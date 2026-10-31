"""每个设备后端都要实现的契约：一张扁平的语义算子表，加一组能力标志。

引擎需要平台给的只有两样东西：每个*语义*设备算子对应一个可调用对象，以及少数
几个模型代码要据以分支的行为事实。用算子名做键的扁平表（而不是一个内核一个类）
让调用点保持统一——``ops.linear(w, x)`` 而不是 ``hip_ops.gemv(w, x, R, U)``——
并且让这张表是全的：每个后端都回答每个名字，没有原生内核的算子回退到纯 torch
参考实现。
Module: ``coffin/backends``。与 ``coffin/backends/registry.py`` 一同位于依赖图
底部：它可以 import ``torch`` 和叶子模块 ``int8_weights``（后者本身只 import
``torch``）。它不得 import ``coffin/ops``、``models``、``engine``、``serving``
或 ``cache``——那些模块从这里 import，绝不反过来，后端层必须能在没有模型、
没有 checkpoint、没有编译扩展的情况下 import。

Constraints:
    * 契约参数只包含模型语义量：形状、dtype、量化存储格式、位置、因果长度。任何
      只因为*这张卡*如何切内核而存在的东西（每块行数、K-unroll、wide-M 变体、
      调试用的 A/B 入口）都留在后端内部，绝不能变成算子名或参数。这就是八个
      ``hip_ops.gemv*`` 条目要收敛成三个 ``linear*`` 名字的原因：新平台不该被迫
      接受在那里毫无意义的调优旋钮。
    * ``w`` 可以是普通 ``torch.Tensor``，也可以是 ``int8_weights.QWeight``；
      int8 分派发生在后端*内部*，调用方传加载器产出的那种存储即可。``QWeight``
      是模型侧的存储格式，不是平台能力。
    * 原地语义是契约的一部分：``out``、``conv_state``、``st`` 以及 prologue 的
      输出都是调用方持有的缓冲区，图捕获路径要求它们地址固定。后端直接写穿它们
      并返回同一个张量，而不是另分配一个替换品。
    * 各后端签名完全一致，包括纯 torch 路径忽略的那些参数：调用点不得按当前
      后端分支。
    * ``Caps`` 只承载模型会据以分支的事实。某个算子是否原生实现看
      ``native_ops``；回退的答案永远是「照常调用」，所以表本身从不告诉调用方
      跳过工作。
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Callable, ClassVar, Mapping, Protocol, runtime_checkable

import torch


#: 一个算子条目。签名按 :class:`Ops` 的字段逐个文档化；后端绑的是闭包和
#: partial，所以这里的可调用类型保持不透明。
OpFn = Callable[..., Any]


#: 规范算子名，按 :class:`Ops` 字段顺序排列，分组方式与 PLATFORMS.md §3.2 的
#: 表格一致。有测试保证它与 dataclass 同步。
_OP_NAMES: tuple[str, ...] = (
    # 投影：按激活行数的档位命名，不是按内核命名。
    "linear",            # 1 行或少数几行
    "linear_m",          # 小 M 的批，一次权重读取
    "linear_prefill",    # 预填充分块的 GEMM（大 M）
    # 归一化。
    "rms_norm",
    "add_rms_norm",
    "rms_norm_gated",
    "kv_quant",
    "silu_mul",
    "rope_apply",
    # int8 KV cache 上的全注意力。
    "attn_decode",
    "attn_paged",
    "attn_paged_batch",
    "attn_paged_multi_q",
    # Gated DeltaNet：深度卷积 + SiLU，然后是循环。
    "conv1d_silu",
    "conv1d_silu_seq",
    "conv1d_silu_batch",
    "delta_prologue",
    "delta_prologue_batch",
    "delta_prep_seq",
    "delta",
    "delta_batch",
    "delta_normed",
    # 三角求解：分块 delta 规则每个块要反解一个单位下三角方程组，那是*库*能力，
    # 不是模型数学。
    "triangular_solve",
)


@dataclass(frozen=True, slots=True)
class Caps:
    """一个后端在这台主机上是什么：身份、dtype，以及模型代码据以分支的事实。

    这里每个字段都替换掉当前写死在后端之外的一处东西：``"cuda"`` 字面量、
    ``ops/dtypes.py`` 的 ``D``/``WF``、``hip_ops.available()`` 检查、
    ``models/qwen35/tuning.py`` 的 ``ATTN_SPLIT``，以及
    ``hasattr(hip_ops, ...)`` 探测。
    """

    #: 后端标识符：``COFFIN_BACKEND`` 接受的那个名字，日志行里打印的也是它。
    name: str
    #: 后端执行所在的 torch 设备*类型*，即 ``torch`` 上的属性名：``"cuda"``、
    #: ``"gcu"``、``"musa"``、``"xpu"``、``"cpu"``。它是 ``torch.device`` 兼容的
    #: 字符串；要解析好的 ``torch.device`` 用 :func:`make_device`（或
    #: ``backends.device()``）。
    #:
    #: 这是命名空间，**不是**厂商。``"cuda"`` 由 ROCm、Iluvatar CoreX 和 MUSA 的
    #: 兼容层分别实现，所以值是 ``"cuda"`` 并不能说明卡是谁家的——
    #: :attr:`device_name` 才承载那个信息。
    device: str
    #: :attr:`device` 内的序号，即 ``torch.device(type, N)`` 的 ``:N``。
    device_index: int = 0
    #: 驱动报出来的板卡名（``"Z200SM_80"``、``"Iluvatar BI-V150"``、
    #: ``"MTT S4000"``），未知时为 ``""``。
    #:
    #: 这不是只写日志的字段：在**共享**设备命名空间里，它是唯一能区分厂商的东西，
    #: 所以后端检测以它为键。不要把它降格成装饰，也不要重新引入
    #: ``is_available()`` 检查当作平台判定。
    device_name: str = ""
    #: 内核编译所针对的目标 ISA，从驱动读取而不是写死（这张 DCU 上是
    #: ``"gfx906"``）；后端不针对具名架构时为 ``""``。编译标志由它推导，所以换
    #: 一颗芯片要换的是这个值，而不是加第二条编译路径。
    arch: str = ""
    #: 本后端用编译内核提供的算子名。它的补集跑参考实现，所以模型的调用点从不需要
    #: 知道两者的差别。纯 torch 参考后端为空；启动行打印
    #: ``len(native)/len(Ops.NAMES)``，那是新平台迁移进度的进度条。
    native: frozenset[str] = frozenset()
    #: 内核和 int8 KV 缓存所用的激活 dtype。替换 ``ops.dtypes.D``。
    act_dtype: torch.dtype = torch.float32
    #: 投影权重 dtype。替换 ``ops.dtypes.WF``；gfx906 上是 fp16，因为那颗芯片有
    #: packed fp16 数学而没有 packed bf16 数学，而有 packed bf16 的芯片应当跟随
    #: :attr:`act_dtype`。
    weight_dtype: torch.dtype = torch.float32
    #: 解码步能捕获进设备图并重放。内核在捕获流之外启动的后端必须报 ``False``：
    #: 那种图重放出来的是过期输出而不是报错，属于静默的正确性 bug，而非性能问题。
    graph: bool = False
    #: 允许加载时按形状基准测试内核以挑选配置。``False`` 表示后端取静态默认值，
    #: 适用于计时原语根本不存在（xpu 上的 ``torch.cuda.Event``）或者扫描的成本
    #: 大于收益的平台。
    autotune: bool = False
    #: 注意力内核的 split-K 宽度。这是内核 ABI，不是调优旋钮：调用方据此分配
    #: ``part_acc``/``part_ml``，不匹配会让注意力结果坏掉而不是悄悄失败一个形状
    #: 检查。``1`` 表示后端不做切分，因而不需要调用方持有的部分和。
    attn_split: int = 1
    #: 批处理分页注意力是原生的，所以一个批桶只花一次注意力启动而不是每行一次。
    #: 这与「表条目存在」是两件事：每个后端都有 ``attn_paged_batch``，但只有部分
    #: 后端原生执行它。
    batch_attn: bool = False
    #: 后端原生读取 ``int8_weights.QWeight``。为 ``False`` 时 ``linear*`` 算子先
    #: 反量化到 :attr:`weight_dtype` 再做 GEMM，结果正确，但每个权重元素要读两倍
    #: 字节。
    int8_weights: bool = False
    #: 后端自己负责线性注意力循环状态的前缀复用快照与恢复。混合模型没有这个就没法
    #: 复用前缀：共享的 KV 配上一段从没见过那些 token 的循环，答案就是错的。
    #: ``False`` 会让本后端的前缀复用直接关掉，而不是悄悄生成错字。
    #:
    #: 实践中每个 torch 后端都是 ``True``：状态就是一块稠密 fp32 张量，
    #: 快照/恢复在引擎侧是 ``clone()``/``copy_()``，不涉及内核。只有当这个拷贝
    #: 不被支持或不可信时才该报 ``False``，不能仅因为自己没有原生快照内核就报
    #: ``False``——那会白白丢掉主要的 token 延迟收益。
    recurrent_state: bool = False
    #: 本后端希望投影权重以哪种布局存放：``"nk"``（checkpoint 的 ``[N, K]``
    #: 行主序，本引擎当前的约定）或 ``"kn"``（``[K, N]``，即 ``x @ Wt``）。
    #:
    #: 这不是装饰性字段。在 Iluvatar BI-V100 上，``nn.Linear`` 产出的 ``[N, K]``
    #: 布局实测 19-30 GB/s，占掉一个解码步 92% 的时间，而 ``[K, N]`` 达到
    #: 362-410 GB/s：一个布局决定换来 14-22 倍。算子契约仍是 ``[N, K]``，因为
    #: checkpoint 就是那样；``"kn"`` 后端自己在加载时转置一次，自行承担成本和
    #: 额外缓冲区。
    weight_layout: str = "nk"
    #: 后端可以传给内核用于位置、槽位表和块索引的整型 dtype。
    #:
    #: 默认 ``int64``，因为本引擎的索引张量就是这个类型。有些平台没有 64 位整型
    #: 通路——Enflame 的 torch_gcu 会静默降级成 32 位并告警，对 ``data_ptr`` 而言
    #: 这还让报出的地址不可信——所以它们必须声明 ``torch.int32``，由调用方把索引
    #: 控制在那个范围内，而不是事后才发现被截断。
    index_dtype: torch.dtype = torch.int64


@dataclass(frozen=True, slots=True)
class Ops:
    """一个后端的扁平设备算子表；字段名就是契约。

    这张表是*全的*：每个后端都回答全部 18 个名字。原生后端返回自己的内核；
    :mod:`coffin.backends` 里的解析器把后端留成 ``None`` 的名字补上参考实现。
    因此 ``None`` 只在原生后端自己的表里才有意义，表示「这里没有内核」——绝不是一个
    会漏到模型代码里的缺口。

    每个条目在每个后端上接受相同的位置参数。名为 ``out``、``conv_state``、``st``
    以及 prologue 的 ``*_out`` 输出的缓冲区归调用方所有并原地写入，因为解码步会被
    捕获进一张缓冲区地址不能变动的图。
    """

    #: ``linear(w, x) -> y``：解码时的权重读取。``x`` 是 ``[K]`` 或 ``[C, K]``，
    #: ``y`` 是 ``[N]`` 或 ``[C, N]``，dtype 跟 ``x`` 走；``w`` 是 ``[N, K]``
    #: 行主序，可以是 ``torch.Tensor`` 也可以是 ``QWeight``。一个名字同时覆盖两种
    #: 存储格式和所有单行内核变体，调用点因此不必携带任何调优选择。
    linear: OpFn | None = None
    #: ``linear_m(w, x) -> y``：小 M 的批（``x`` ``[M, K]``、``y`` ``[M, N]``），
    #: 一次权重读取服务全部 ``M`` 行。后端自己决定特化哪个 ``M``、余下的怎么切，
    #: 调用方只知道自己的行数大于一。
    linear_m: OpFn | None = None
    #: ``linear_prefill(w, x) -> y``：大 M 档位，此时 GEMM 胜过权重流式内核。
    #: 权重为 int8 时，没有反量化 GEMM 的后端会为这次调用把 ``w`` 实体化成
    #: :attr:`Caps.weight_dtype`，这个取舍是对的，因为预填充是算力受限的。
    linear_prefill: OpFn | None = None
    #: ``rms_norm(x, w, stride=0, eps=1e-6) -> y``：对 ``[M, N]`` 最后一轴做零中心
    #: RMSNorm（有效系数 ``1 + w``，由调用方折好）。``stride`` 是行与行之间的元素
    #: 间距，让指向更宽缓冲区的视图能原地归一化而不必拷贝；``0`` 表示「从 ``x`` 里
    #: 读」。不接受单独间距参数的后端忽略该参数。
    rms_norm: OpFn | None = None
    #: ``add_rms_norm(x, add, w, out, stride=0, eps=1e-6) -> out``：一遍完成残差
    #: ``x += add`` 与和值的零中心 RMSNorm。``x`` 原地累加，因为残差流在这层后续
    #: 还要继续用；``out`` 是独立缓冲区而不是与 ``x`` 别名，原因相同。没实现它的
    #: 后端走参考实现：两次启动加一次额外读取而不是一次，更慢但数值上完全一致。
    add_rms_norm: OpFn | None = None
    #: ``rms_norm_gated(x, w, gate, eps=1e-6) -> y``：``rms(x) * w * silu(gate)``，
    #: 按头归一化，所以 ``w`` 是 ``[dv]`` 而不是展平后的宽度。
    rms_norm_gated: OpFn | None = None
    #: ``kv_quant(x, out_q, out_s) -> out_q``：对 ``[M, N]`` 做以行为单位的对称
    #: int8 量化，每行一个 fp32 scale，``out_s`` 为 ``[M]``。它之所以单独成为一个
    #: 条目，是因为 KV 缓存契约把量化方式定死——
    #: ``abs().amax(-1).clamp_min(1e-5) / 127``——某个后端若把这七个算子的链条融合
    #: 成别的样子，缓存就会和读它的内核脱节。两个输出都是调用方持有、原地写入的
    #: 缓冲区。
    kv_quant: OpFn | None = None
    #: ``silu_mul(gate, up, out) -> out``：``out = silu(gate) * up``，MLP 的 SwiGLU
    #: 乘积。它成为一个算子而不是两次 torch 调用，是因为融合的 ``gate_up`` 投影在
    #: 一次权重读取里把两半一起交回来，它们和 down 投影之间只剩一个逐元素乘积：
    #: 写成 ``aten::silu`` + ``aten::mul`` 就是每层两次启动加两个
    #: ``[intermediate]`` 临时张量。``out`` 可以和 ``up`` 别名——自然的写法就是把
    #: 乘积写回 ``up`` 那一半，反正后面没人再读它——这也是为什么缓冲区参数是显式的，
    #: 而不是让条目返回一个新张量。
    silu_mul: OpFn | None = None
    #: ``rope_apply(q, k, cos, sin, q_out, k_out, rotary_dim) -> q_out``：把每个
    #: q/k 头的前 ``rotary_dim`` 个通道按该位置的角旋转，尾部
    #: ``head_dim - rotary_dim`` 个通道原样拷过。
    #:
    #: ``q`` 是 ``[M, nh, hd]``，``k`` 是 ``[M, nkv, hd]``，一个形状同时服务单
    #: token 的解码路径（``M == 1``）和预填充/校验路径（``M`` 行）。``cos``/``sin``
    #: 是 ``[M, rotary_dim]`` fp32，传进来时已经按被旋转的行 gather 好：两个调用点
    #: （mRoPE 与校验）传的是归一化张量而不是表里的一行，让内核接管 gather 就得
    #: 逼所有调用点用同一种索引表达式，只为省一次并非瓶颈的启动。
    #:
    #: 表保持 fp32，在内核里逐元素转换。实体化成 bf16 的表在长上下文上损失的角
    #: 精度足以让 logits 偏移。``q_out``/``k_out`` 不得与 ``q``/``k`` 别名。
    rope_apply: OpFn | None = None
    #: ``attn_decode(q, gate, kcache, vcache, kscale, vscale, out, pos, rep, scale,
    #: part_acc=None, part_ml=None) -> out``：密集 int8 KV 缓存上的单 token 注意力。
    #: ``pos`` 是单元素 long 张量，既用于寻址槽位，也是因果边界（越过它的条目被
    #: 掩掉而不是跳过，形状因此保持静态）。``gate`` 以 ``sigmoid`` 施加。
    #: ``part_acc``/``part_ml`` 是调用方持有的 split-K 中间缓冲区，尺寸由
    #: :attr:`Caps.attn_split` 决定；传 ``None`` 让后端自行分配，结果正确但不可被
    #: 图捕获。
    attn_decode: OpFn | None = None
    #: ``attn_paged(q, gate, kpool, vpool, kscale, vscale, pos2slot, out, pos,
    #: max_pos, rep, scale, part_acc=None, part_ml=None) -> out``：同样的注意力，但
    #: 跑在分页 int8 池的某一层上。``kpool``/``vpool`` 是 ``[nslots, nkv, hd]``
    #: int8，``kscale``/``vscale`` 是 ``[nslots, nkv]`` fp32，``pos2slot`` 是该行的
    #: ``[max_pos]`` 槽位表，只读它的前 ``pos + 1`` 项。``pos2slot`` 是恒等映射时，
    #: 结果必须与 :attr:`attn_decode` 一致。
    attn_paged: OpFn | None = None
    #: ``attn_paged_batch(q, gate, kpool, vpool, kscale, vscale, slots,
    #: context_lens, out, max_seq, rep, scale, part_acc=None, part_ml=None) -> out``：
    #: 一次启动服务 ``B`` 行，第 ``b`` 行只读它自己 ``slots[b]`` 的前
    #: ``context_lens[b]`` 项，混合 prompt 长度的批桶因此能共用一次启动。单独看
    #: 每一行，结果必须与 :attr:`attn_paged` 数值完全一致。
    attn_paged_batch: OpFn | None = None
    #: ``attn_paged_multi_q(q, gate, kpool, vpool, kscale, vscale, slots, ctx_lens,
    #: out, max_seq, rep, scale, part_acc=None, part_ml=None, block_mask=None,
    #: mask_block=64) -> out``：投机校验，``q`` 为 ``[B, c, nh, hd]``，一行的 ``c``
    #: 个草稿查询共享对该行历史的一次遍历。查询 ``j`` 看到 ``ctx_lens[b] + j + 1``
    #: 个 key，即与单独解码该 token 逐位一致。``block_mask``（uint8
    #: ``[B, n_blocks]``，1 表示保留）限定哪些 KV 块参与；``None`` 是精确的全
    #: 注意力。
    attn_paged_multi_q: OpFn | None = None
    #: ``conv1d_silu(conv_w, x, conv_state, out) -> out``：单 token 的深度因果卷积
    #: （卷积核 4）+ SiLU。``conv_w`` 是 ``[C, 4]`` fp32，当前 token 在最末；
    #: ``conv_state`` 是 ``[C, 3]`` fp32，被原地平移，所以它是混合器的短期记忆，
    #: 必须与循环状态一起打快照。
    conv1d_silu: OpFn | None = None
    #: ``conv1d_silu_seq(conv_w, x, conv_state) -> y``：同样的卷积，跑在预填充分块
    #: ``[T, C]`` -> ``[T, C]`` fp32 上，并跨块携带 ``conv_state``，使分块预填充与
    #: 一次性预填充结果完全一致。用 fp32 是因为下游的 delta 扫描要跨块累加。
    conv1d_silu_seq: OpFn | None = None
    #: ``conv1d_silu_batch(conv_w, x, conv_state, out) -> out``：``B`` 行，``x``
    #: ``[B, C]``，``conv_state`` ``[B, C, 3]``；单独看每一行，结果与
    #: :attr:`conv1d_silu` 一致。
    conv1d_silu_batch: OpFn | None = None
    #: ``delta_prologue(q, k, v, b, a, neg_expA, dt_bias, q_out, k_out, v_out,
    #: beta, g, qscale, rep) -> None``：一步头准备——对 q/k 做 L2 归一化，把 key
    #: 头扩展到 value 头（``rep``），并把 ``beta = sigmoid(b)`` 与
    #: ``g = neg_expA * softplus(a + dt_bias)`` 建进 fp32 输出缓冲区。``q`` 额外
    #: 乘 ``qscale``，``k`` 只做归一化。``g`` 必须保持 ``<= 0``，即每步衰减，绝不
    #: 增长。
    delta_prologue: OpFn | None = None
    #: ``delta_prologue_batch(...) -> None``：批处理的 :attr:`delta_prologue`，
    #: ``q``/``k``/``v``/``b``/``a`` 和输出都多一个前导行维；
    #: ``neg_expA``/``dt_bias`` 保持一维，所有行共享。
    delta_prologue_batch: OpFn | None = None
    #: ``delta_prep_seq(qkv, b, a, neg_expA, dt_bias, q_out, k_out, v_out, beta, g,
    #: qscale, rep) -> None``：预填充的头准备。q/k/v 按头主序写成
    #: ``[num_v, T, d]``，beta/g 写成 ``[num_v, T]``，正是分块 delta 扫描直接读取的
    #: 布局，分块因此不需要再做一次转置拷贝。
    delta_prep_seq: OpFn | None = None
    #: ``delta(q, k, v, beta, g, st, out) -> out``：一步 gated delta 解码。原地更新
    #: ``st`` ``[num_v, dk, dv]`` fp32，写出 ``out`` ``[num_v, dv]`` fp32；``out``
    #: 由*更新后*的状态算出。这个循环是本层唯一的跨 token 记忆，且没有闭式回退，
    #: 所以需要回滚的调用方要先给 ``st`` 打快照。
    delta: OpFn | None = None
    #: ``delta_batch(q, k, v, beta, g, st, out) -> out``：与 :attr:`delta` 相同，但
    #: 每个参数都多一个前导行维；单独看每一行，结果必须与 :attr:`delta` 一致。
    delta_batch: OpFn | None = None
    #: ``delta_normed(q, k, v, b, a, neg_expA, dt_bias, z, nw, st, out, qscale,
    #: rep) -> out``：单个解码 token 的整个线性注意力混合器——头准备、循环、带门控
    #: 的输出归一化——一次启动完成。
    #:
    #: 它成为一个算子，是因为它是一个*访存*问题而不是三个。分三段的写法每层每
    #: token 要写入再读回约 48 KB fp32 中间缓冲区（``q_pro``/``k_pro``/``v_pro``
    #: 和 ``delta_out``，各 ``[32, 128]``），就为了搬运一个 block 就能装进共享内存
    #: 和寄存器里的值。``st`` 原地更新，``out`` 承载归一化后的结果，调用方的缓冲区
    #: 和捕获路径的地址都不变。
    #:
    #: ``z`` 是输出归一化的门控投影，``nw`` 是它的 ``[dv]`` 增益：归一化按 value 头
    #: 做，正好一个 block 的范围，所以能并进 epilogue，不需要额外归约。
    delta_normed: OpFn | None = None
    #: ``triangular_solve(lower, rhs) -> out``：解 ``A X = B``，``A`` 是
    #: ``[bn, L, L]`` 单位下三角，``B`` 是 ``[bn, L, n]``，返回 ``[bn, L, n]``，
    #: dtype 跟 ``rhs`` 走。
    #:
    #: 它是算子，是因为该由一个*库*来回答它，而不是模型。分块 delta 规则每层每块
    #: 需要一次 ``A^-1 B``，而显然的写法——``torch.linalg.solve_triangular``——在
    #: CoreX 上不可用：那个构建的 cuSOLVER 是从写死的
    #: ``/opt/sw_home/local/cuda/lib64/libcusolver.so`` 加载的，该文件在集群上不
    #: 存在，于是每条 ``linalg`` 求解路径都在 ``dlopen`` 里失败。用 ``bmm`` 写的
    #: 分块前向代入不依赖任何库，实测与主机侧 fp64 求解相差 1.1e-07。
    #:
    #: ``unitriangular`` 固定为 ``True`` 而不开放：delta 规则的 ``UT`` 方程组对角
    #: 恒为单位元（它是 ``I`` 加上严格下三角乘积），开放成标志就是一个没有调用方
    #: 能有意设置的旋钮。
    triangular_solve: OpFn | None = None

    #: 规范算子名，按字段顺序。
    NAMES: ClassVar[tuple[str, ...]] = _OP_NAMES

    def __post_init__(self) -> None:
        """拒绝既不可调用也不是 ``None`` 的条目。

        Raises:
            TypeError: 某个条目不可调用。把不可调用对象注册进来，会表现为第一步
                解码时一个莫名其妙的 ``TypeError``，而不是建表时就出错，而且只对
                某一个形状暴露。
        """
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None and not callable(value):
                raise TypeError(
                    f"op {field.name!r} must be a callable or None, "
                    f"got {type(value).__name__}")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, OpFn | None]) -> Ops:
        """用一份不完整的 ``{name: callable}`` 映射建表。

        Args:
            mapping: 算子名到可调用对象；``None`` 值多余但可接受。``mapping`` 里
                没有的每个名字都变成 ``None``。

        Returns:
            表。缺席的条目就是解析器要从参考后端补的东西，所以原生后端只列出自己
            实现的部分。

        Raises:
            ValueError: ``mapping`` 里有 :attr:`NAMES` 之外的算子名。否则一个拼写
                错误会产出一张悄悄少了那个算子的表，看起来和一个没有该内核的平台
                一模一样。
        """
        unknown = sorted(set(mapping) - set(_OP_NAMES))
        if unknown:
            raise ValueError(
                f"unknown op(s): {', '.join(unknown)}; "
                f"known: {', '.join(_OP_NAMES)}")
        return cls(**{name: mapping.get(name) for name in _OP_NAMES})

    def as_dict(self) -> dict[str, OpFn | None]:
        """按规范顺序把表返回成 ``{name: callable-or-None}``。

        Returns:
            每个算子名一条，``None`` 的那些也在内。调用方遍历它而不是
            ``dataclasses.fields``，这样类型检查器还能保住「可调用或 ``None``」的
            形态。
        """
        return {name: getattr(self, name) for name in _OP_NAMES}

    def implemented(self) -> tuple[str, ...]:
        """列出这张表自己就能回答的算子名。

        Returns:
            带可调用对象的名字，按规范顺序。在原生后端上这就是它的内核清单；在
            参考后端上是全部 18 个。
        """
        return tuple(name for name, fn in self.as_dict().items() if fn is not None)

    def missing(self) -> tuple[str, ...]:
        """列出这张表留成 ``None`` 的算子名。

        Returns:
            解析器必须从参考后端补的名字，按规范顺序。为空表示表已经是全的。
        """
        return tuple(name for name, fn in self.as_dict().items() if fn is None)


@runtime_checkable
class Backend(Protocol):
    """一个平台：怎么找到它、它能做什么，以及它的算子表。

    registry 只碰这几个成员，所以一个后端就是一个带这六个属性的模块，而不是一棵
    类继承树。每个后端每进程只实例化一次：选后端是 import 时的决定，没有运行时
    热切换。
    """

    #: ``COFFIN_BACKEND`` 接受、并在启动行里打印的标识符。
    name: str

    def probe(self) -> bool:
        """报告本后端针对的硬件是否在本机上。

        必须廉价且无副作用——不编译内核、不分配、不重置设备——因为 registry
        自动探测时会挨个探测所有候选。``False`` 表示「不是这台机器」，不是
        「坏了」。

        Returns:
            后端所需的设备或驱动可见时为 ``True``。
        """

    def usable(self) -> bool:
        """报告本后端现在能否执行模型。

        比 :meth:`probe` 更严：内核编译（如果有）也必须成功。自动探测时 registry
        会继续尝试下一个候选，所以这道门让一个驱动在、工具链却不可用的主机留在
        即时执行路径上，而不是把进程带崩。

        Returns:
            :meth:`ops` 能给出模型可以跑的表时为 ``True``。
        """

    def reason(self) -> str:
        """用一行解释 :meth:`usable` 为什么是 ``False``。

        Returns:
            展示在选择错误和启动日志里的诊断信息；可用时为 ``""``。有了它，一次
            内核编译失败才不会读起来像模型慢。
        """

    def ops(self) -> Ops:
        """返回本后端的算子表。

        首次调用可能很贵（要编译或挂载编译好的内核），必须记忆化：引擎每个请求
        会通过模块级表调用它成千上万次。永远不做部分填充——没有原生内核的算子是
        ``None``，由解析器补齐。

        Returns:
            表。不可用的后端也照样返回一张表（全 ``None``），好让调用方记录并检查
            它，而不是在每个调用点处理异常。
        """

    def caps(self) -> Caps:
        """返回本后端在这台主机上的能力标志。"""

    @property
    def device(self) -> torch.device:
        """本后端所有分配都应该使用的 ``torch.device``。

        Returns:
            ``torch.device(caps.device, caps.device_index)``。模型代码用它替换自己
            的 ``device="cuda"`` 字面量，于是换平台改一个值而不是 173 个调用点。
        """


def make_device(caps: Caps) -> torch.device:
    """把一条 :class:`Caps` 记录解析成要用来分配的 ``torch.device``。

    Args:
        caps: 后端的能力记录。

    Returns:
        ``torch.device(caps.device, caps.device_index)``；``"cpu"`` 时丢掉序号，
        因为并非所有受支持的 torch 版本都接受 ``torch.device("cpu", 0)``。

    Raises:
        ValueError: ``caps.device`` 不是 torch 认识的设备类型。在这里失败，好过在
            模型构建到一半的首次分配时才失败。
    """
    if caps.device == "cpu":
        return torch.device("cpu")
    return torch.device(caps.device, caps.device_index)
