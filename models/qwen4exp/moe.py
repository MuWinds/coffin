"""MoE：TopK 路由、512 个融合 3D 专家、共享专家。

三层结构如下：

* 路由没有归一化输入，也没把输入再过一遍 RMSNorm，它直接吃读门塌缩后的
  2560 宽激活；softmax 在 fp32 上做完才 top-k，权重取自**概率**而不是 logits；
  ``norm_topk_prob`` 为真时权重是"top-k 概率再除以它们自己的和"。
* 专家是 3D Parameter：``gate_up_proj[e]`` 是 ``[2*I, H]``（前 I 行 gate、
  后 I 行 up），``down_proj[e]`` 是 ``[H, I]``。checkpoint 里也是融合张量，
  不是 ``nn.Linear``。
* 共享专家的融合是逐 token 的标量 sigmoid 门（``[H] -> [1]``），不是逐通道；
  融合就是相加，没有再归一化。

数学上整块的输出是
``y[t] = Σ_k w[t,k] * Expert_{idx[t,k]}(x[t]) + sigmoid(g_s[t]) * Shared(x[t])``。

Module: ``coffin/models/qwen4exp``。依赖 ``config`` 与 ``proj``（共享专家是稠密投影）；被
``blocks`` 调用。

Constraints:
    * 聚合必须按专家分组做，不能展开成"每 token 跑 10 次完整 GEMM"：真实配置下专家
      权重是 ``[512, 1280, 2560]``，展开会读 10 倍权重。
    * 融合路径的合并**不许用 ``index_add_``**：它在 CUDA 上走 ``atomicAdd``，浮点加法
      不满足结合律，同一份输入两次前向结果不逐位相同（作业 287/288 实测同版本两次
      greedy token 就不同）。改用「按槽位序 gather + 定序 ``sum(dim=1)``」。
      只有非融合的兜底路径仍然用 ``index_add_``，它不参与服务路径。
    * int8 打包时必须逐 expert 保持 ``[K, N]`` 行主序，不能把 512 个专家摊平成
      一个 2D 矩阵——``grouped_gemm`` 用固定偏移寻址。
    * W4A8 分派路径上不许出现主机往返。``.tolist()`` / ``int(tensor)`` 每次都要
      等一次设备到主机的拷贝，而 decode 一步只有 1 个 token、48 层各来一遍，
      付的全是固定开销：分组内核吃的是设备指针，前缀和没有理由回主机。
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from backends import ops as bops
from models.qwen4exp.config import TextConfig
from models.qwen4exp.proj import dispatch_linear, proj_from_config

#: 块内激活。Qwen 系列固定 silu，这里手写以免引入除 ``torch`` 之外的依赖。
_ACTIVATIONS = {"silu": F.silu, "gelu": F.gelu}

#: 量化分派路径是否逐层把专家下标读回主机校验范围。
#:
#: 默认关闭。那次校验要把 ``min`` / ``max`` 两个设备标量取回主机，48 层 decode
#: 一步就是 96 次同步；而 decode 只有 1 个 token、每层 10 个下标，这份开销与
#: token 数无关，全是白付。下标又由 ``TopKRouter`` 在 ``[0, num_experts)`` 上选出，
#: 正常情况下不可能越界。
#:
#: 需要查融合内核或 EP 映射时设 ``COFFIN_MOE_CHECK=1``（要在进程启动前设好，这里
#: 只在导入时读一次）。越界照旧抛 ``ValueError``，文案与 bf16 路径逐字相同，报错点
#: 不会因为跑的是哪条路径而变。
_CHECK_DISPATCH = bool(os.environ.get("COFFIN_MOE_CHECK"))

#: 打印分组投影的真实形状（`COFFIN_MOE_SHAPE=1`）。只用于测量，默认关闭。
#:
#: 存在理由：这条链上所有关于「每专家几行」的判断一直是**外推**出来的
#: （总 token 数 ÷ 本地专家数），而外推忽略了「每 token 的 top_k 个专家可能落在
#: 同一 rank」这件事，已经因此把 MoE 的 prefill 占比估成 17%（实测 55%）。
#: 内核耗时对 rows 极敏感（rows=8 约 9 µs/专家、64 约 67 µs、160 约 517 µs），
#: 所以这个数必须直接量，不能算。
_SHAPE_TRACE = bool(os.environ.get("COFFIN_MOE_SHAPE"))

#: 每专家行数达到它就用按专家分段的张量核 GEMM，否则用 GEMV 结构的分组内核。
#:
#: 判据是**每专家平均行数**（``rows / num_local_experts``），不是总行数：``offsets``
#: 在设备上，为了拿「非空专家数」回主机读一次是不值的（那正是
#: :meth:`Experts.forward` 里刚删掉的那类同步）。用平均值当代理即可——它把
#: 「EP 分片后本 rank 只有 1/world 的专家」这件事算进去了，而总行数没有。
#:
#: **128 是实测的交叉点，不是经验值**（作业 221 的 ``PFB2``，真几何 E=128、
#: ``gate_up`` K=2560→N=1280，同一份 17.4 MB 权重，只换行数）：
#:
#:     rows/专家   1      8      32     64     128
#:     GEMV µs     1282   4240   14960  29783  59452
#:     GEMM µs     34642  34968  36717  39659  45503
#:     GEMM/GEMV   27x慢  8x慢   2.5x慢 1.3x慢 1.3x**快**
#:
#: 关键在 GEMM 那列**几乎不随行数变**：128 次专家发射共约 35 ms，摊到每次
#: **274 µs**，而同样的活按带宽只要几微秒——成本是「每次专家发射的固定开销」，
#: 不是算术。所以分界必须放在它终于能用工作摊薄的地方，64 那一档 GEMM 仍然慢 1.3 倍。
#:
#: 旧值 64 的依据（``_PREFILL_MIN_M`` 也是 64）只在稠密投影上成立，那里一次调用
#: 只有一次发射；这里一次调用是 `num_local_experts` 次发射，固定开销被乘了 128。
_GEMM_MIN_ROWS = int(os.environ.get("COFFIN_MOE_GEMM_MIN_ROWS", "128"))


class MLP(nn.Module):
    """标准 SwiGLU 前馈块：``down(act(gate(x)) * up(x))``。

    共享专家就是它，与 dense 模型的 MLP 完全同构。

    Attributes:
        gate_proj / up_proj / down_proj: 三个无 bias 线性层。
    """

    def __init__(self, config: TextConfig, intermediate_size: int | None = None) -> None:
        """建三个投影。

        Args:
            config: 文本配置。
            intermediate_size: 中间宽度；``None`` 时退到
                ``shared_expert_intermediate_size``。文本配置里没有 dense MLP
                的 ``intermediate_size`` 字段，非共享专家的路径也不经过这个类。

        Raises:
            ValueError: 激活名不认识。
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = (config.shared_expert_intermediate_size
                                  if intermediate_size is None else intermediate_size)
        # 共享专家是**稠密**的、每卡算一份完整，所以它走投影层，不走下面的分组专家
        # 内核。它不是 ``nn.Linear`` 而是 :class:`~models.qwen4exp.proj.Proj`，因为
        # 它和 GDN/QSA 的投影一样受行主序布局的拖累：``[640, 2560]`` 在 M=1 上实测
        # 0.550 ms，换 ``[K, N]`` 后 0.050 ms，48 层合计约 63 ms（作业 4489/4492）。
        self.gate_proj = proj_from_config(config, self.hidden_size, self.intermediate_size,
                                         bias=False)
        self.up_proj = proj_from_config(config, self.hidden_size, self.intermediate_size,
                                       bias=False)
        self.down_proj = proj_from_config(config, self.intermediate_size, self.hidden_size,
                                        bias=False)
        if config.hidden_act not in _ACTIVATIONS:
            raise ValueError(f"unsupported activation {config.hidden_act!r}.")
        self.act_fn = _ACTIVATIONS[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``down(act(gate(x)) * up(x))``。

        Args:
            x: ``[..., hidden_size]``。

        Returns:
            ``[..., hidden_size]``。
        """
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class TopKRouter(nn.Module):
    """逐 token 选 ``top_k`` 个专家并给出归一化权重。

    Attributes:
        top_k: 每 token 激活的专家数。
        num_experts: 专家总数。
        norm_topk_prob: 是否把 top-k 概率再归一化到和为 1。
    """

    def __init__(self, config: TextConfig) -> None:
        """建路由矩阵。

        Args:
            config: 文本配置。
        """
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))

    def forward(self, hidden_states: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """算路由。

        Args:
            hidden_states: ``[tokens, hidden_size]``（调用方先 flatten）。

        Returns:
            ``(router_logits, router_scores, router_indices)``：前两个是
            ``[tokens, num_experts]`` / ``[tokens, top_k]``，下标是
            ``[tokens, top_k]`` 的 int64。

        Raises:
            ValueError: 最后一维与 ``hidden_dim`` 不符。
        """
        if hidden_states.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"router expects hidden dim {self.hidden_dim}, got {hidden_states.shape[-1]}.")
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        # 走 ``dispatch_linear`` 而不是 ``F.linear``：checkpoint 的 ``[N, K]`` 行主序在本卡
        # 上是慢的那一种。一步 decode 里这 48 次调用原先合计 30.234 ms（作业 q4land6 的
        # ``aten::linear`` 行，629.9 us/次，占那一步设备时间的 11.1%），而权重只有 2.62 MB，
        # 按 575 GB/s 读一遍是 4.6 us。路由矩阵是 ``nn.Parameter`` 而不是 ``Proj``，所以
        # 它请不到 ``Proj.forward`` 那套派发，但必须走同一条规则：两处分开写，某天调
        # ``_SMALL_ROWS`` 就会只生效一半。
        router_logits = dispatch_linear(self.weight, hidden_states, self.num_experts)
        # 一次发射算完 ``softmax(fp32) -> top-k -> 归一化``。这条链原本是 5 个 torch 内核
        # （softmax / topk / div / sum / to），decode 一步在 48 层各来一遍，其中
        # ``aten::topk`` 单次 38 us —— 因为它对 ``[1, 512]`` 取 top-10 时走的是 512 元素的
        # bitonic 排序网络（45 轮比较交换），而挑最大的 10 个不需要排序。
        #
        # 语义与原写法逐项对齐：softmax 仍在 fp32 上做且先减行内最大值（低精度下 top-k
        # 的顺序会抖动，而顺序决定选中哪个专家）、仍按概率降序、``norm_topk_prob`` 为真
        # 时仍除以 top-k 概率自己的和。内核只在 ``__expf`` 与两处归约的求和次序上不同，
        # 量级末位 1-2 ulp，而权重接着要窄化成 bf16。
        #
        # 权重仍是 ``router_logits.dtype``（bf16）：下游 ``Experts`` 拿它乘专家输出，
        # 所以这里不做任何 dtype 变化，只少了那次 ``.to()`` 内核。
        router_indices, router_top_value = bops.moe_router(
            router_logits, self.top_k, self.norm_topk_prob)
        return router_logits, router_top_value, router_indices


class Experts(nn.Module):
    """所有专家权重放在一个 3D 张量里，按需分组计算。

    **专家归属是可注入的**：``local_experts`` 给定时本实例只持有其中的专家
    （EP 分片），路由到别处的 token 由调用方的 all-to-all 负责搬。这个参数
    在单卡上是 ``None``（持有全部），在 EP=4 时是 128 个全局专家号。默认行为
    与参考实现逐位一致，注入后才变成分布式路径，所以 fixture 对齐用的是前者。

    Attributes:
        num_experts: 全局专家数。
        local_experts: 本 rank 持有的全局专家号；``None`` 表示全部。
        gate_up_proj: ``[local, 2 * I, H]``，前一半行是 gate、后一半是 up。
        down_proj: ``[local, H, I]``。
    """

    def __init__(self, config: TextConfig, local_experts: Sequence[int] | None = None,
                 quantized: bool = False) -> None:
        """建融合 3D 参数。

        Args:
            config: 文本配置。
            local_experts: 本 rank 持有的全局专家号（升序）；``None`` 表示全部。

        Raises:
            ValueError: 激活名不认识，或 ``local_experts`` 里有越界 / 重复项。
        """
        super().__init__()
        self.num_experts = config.num_experts
        self.local_experts = None if local_experts is None else [int(e) for e in local_experts]
        if self.local_experts is not None:
            if len(set(self.local_experts)) != len(self.local_experts):
                raise ValueError("local_experts must not contain duplicates.")
            invalid = [e for e in self.local_experts
                       if e < 0 or e >= self.num_experts]
            if invalid:
                raise ValueError(
                    f"local_experts out of range [0, {self.num_experts}): {invalid}.")
            if not self.local_experts:
                raise ValueError("local_experts must not be empty.")
        self.num_local_experts = (self.num_experts if self.local_experts is None
                                  else len(self.local_experts))
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        # 必须显式初始化。``nn.Parameter(torch.empty(...))`` 不做任何初始化，内容是
        # 分配时的内存残留：开发机上常常恰好是零或小值，集群上出现过 2.6e17 的野值，
        # 于是整机前向从第 0 层就出 nan，而报错位置（logits）离真正的原因隔了四层。
        # 参考实现用 ``normal_(std=initializer_range)``（``modeling_qwen4_exp.py:1336-1338``），
        # 照抄它同时也让随机初始化下的数值尺度与真实训练一致。
        init_std = float(getattr(config, "initializer_range", 0.02))
        # ``quantized=True`` 时这两个栈建成 ``meta``：量化路径下它们会被
        # :meth:`set_quantized` 删掉，但**分配发生在删除之前**——48 层每卡 60.4 GB
        # 的 bf16 专家栈会在构造时就要求显存，于是还没轮到挂量化权重就已经 OOM
        # （实测报在 ``.to(device, dtype=...)`` 的 ``Tried to allocate 800 MiB``）。
        shape_gu = (self.num_local_experts, 2 * self.intermediate_dim, self.hidden_dim)
        shape_down = (self.num_local_experts, self.hidden_dim, self.intermediate_dim)
        if quantized:
            with torch.device("meta"):
                self.gate_up_proj = nn.Parameter(torch.empty(*shape_gu))
                self.down_proj = nn.Parameter(torch.empty(*shape_down))
        else:
            self.gate_up_proj = nn.Parameter(
                torch.empty(*shape_gu).normal_(0.0, init_std))
            self.down_proj = nn.Parameter(
                torch.empty(*shape_down).normal_(0.0, init_std))
        if config.hidden_act not in _ACTIVATIONS:
            raise ValueError(f"unsupported activation {config.hidden_act!r}.")
        self.act_fn = _ACTIVATIONS[config.hidden_act]
        #: W4A8 的专家栈与分派函数；``None`` 表示走 bf16 的逐专家循环。
        #: 挂在属性上而不是做成子模块：:class:`W4A8MoEWeight` 不是 ``nn.Module``
        #: （它只持有 packed 缓冲区与 scale），做成子模块会被 ``state_dict`` 收走，
        #: 而它本来就不该出现在 ``state_dict`` 里。
        self._grouped = None
        self._quant_gate_up = None
        self._quant_down = None
        #: 融合分派入口（``bops.moe_dispatch``），后端没有这个算子时为 ``None``，
        #: 那时退回下面那条纯 torch 链。见 :meth:`_make_dispatch`。
        self._dispatch = None
        self._identity_table = None

    def _make_dispatch(self):
        """取后端的 ``moe_dispatch`` 算子，没有就返回 ``None``。

        判据用 ``caps().native`` 而不是「表里有没有这个字段」：参考实现会把每个名字
        都补齐，拿表当开关等于在没有内核的平台上接上一串候选调用。理由与
        ``gated_residual._HYPER_OPS`` 相同。

        Returns:
            ``bops.moe_dispatch``，或 ``None``。
        """
        from backends import caps as _caps
        from backends import ops as _ops

        if "moe_dispatch" not in _caps().native:
            return None
        return getattr(_ops, "moe_dispatch", None)

    def _local_or_identity(self, device: torch.device) -> torch.Tensor:
        """取分派内核要的「全局号 → 本地号」表；非分片时恒等。

        Args:
            device: 目标设备。

        Returns:
            ``[num_experts]`` 的 int32 表。
        """
        if self.local_experts is not None:
            return self._local_table(device).to(torch.int32)
        table = self._identity_table
        if table is None or table.device != device:
            table = torch.arange(self.num_experts, dtype=torch.int32, device=device)
            self._identity_table = table
        return table

    def set_quantized(self, gate_up, down) -> None:
        """把专家栈换成 W4A8 版本，并启用分组投影。

        参数从 ``state_dict`` 里摘掉：量化权重不是 ``nn.Parameter``，留着 bf16 的
        ``Parameter`` 会让显存白白多占一份——48 层的专家是 60.4 GB，那一份正好是
        装得下与装不下之间的差距。

        Args:
            gate_up: :class:`W4A8MoEWeight`，``[E, 2*I, H]``。
            down: :class:`W4A8MoEWeight`，``[E, H, I]``。

        Raises:
            ValueError: 专家数与当前分片不一致。混用两个 checkpoint 的专家栈会
                让第 e 个专家用上别人的权重，而形状完全合法。
        """
        if len(gate_up) != self.num_local_experts or len(down) != self.num_local_experts:
            raise ValueError(
                f"quantized expert stack has {len(gate_up)}/{len(down)} experts, "
                f"this shard holds {self.num_local_experts}")
        self._quant_gate_up = gate_up
        self._quant_down = down
        # 必须在切之前取：``local_experts[0]`` 是**全局**专家号，而
        # ``_grouped`` 要给内核的偏移是本地下标，两者相差一个分片起点。
        self._grouped = self._make_grouped()
        # ``del`` 只在参数确实存在时做：量化路径下调用方可能已经把这两个 meta
        # 参数摘成 ``None``（``to_empty`` 会为每个 meta 参数分配内存，60.4 GB 的
        # 专家栈照样会被分配，所以必须在实体化之前就摘掉）。直接 ``del`` 一个
        # ``None`` 属性会抛 ``AttributeError``，而那个报错完全指不到原因。
        for name in ("gate_up_proj", "down_proj"):
            if getattr(self, name, None) is not None:
                delattr(self, name)
        # 补成 ``None`` 让任何误用立刻报错，而不是悄悄拿回一份旧的 bf16 张量。
        self.gate_up_proj = None
        self.down_proj = None
        # 融合分派只有在分组内核可用时才有意义：它算出的 ``offsets`` / 行序是给
        # ``moe_grouped`` 用的。两者必须一起开，否则会拿一条路的输出喂另一条路。
        self._dispatch = self._make_dispatch()

    def _make_grouped(self):
        """返回一个 ``(x, offsets, down_proj=False) -> y`` 的分派函数。

        按**行数**在两个入口之间选：行少（decode，每专家 1-2 行）走
        ``moe_grouped`` 的 GEMV 结构；行多（prefill，每专家上百行）走
        ``moe_grouped_gemm`` 的按专家分段张量核 GEMM。

        分界取 :data:`_GEMM_MIN_ROWS`。这个数是按「每专家平均行数」定的，而不是
        绝对 tokens——因为决定 GEMV 输赢的是每段有多长：段内只有 1-2 行时张量核的
        64x128 tile 几乎全在做无效行，而段内有上百行时 GEMV 每行都要串行走完 K。
        实测 ``seq=2048``、20480 行散在 128 个专家上、每专家约 160 行
        GEMV 结构 22.7 ms/次而同类 GEMM 2.30 ms/次，差 10 倍。

        Returns:
            闭包。``down_proj=True`` 时投影 ``[tokens, I] -> [tokens, H]``，
            否则 ``[tokens, H] -> [tokens, 2*I]``。
        """
        from backends import caps as _caps
        from backends import ops as bops

        gate_up_weight, down_weight = self._quant_gate_up, self._quant_down
        gemm_ok = ("moe_grouped_gemm" in _caps().native
                   and getattr(bops, "moe_grouped_gemm", None) is not None)

        def grouped(x: torch.Tensor, offsets: torch.Tensor,
                    down_proj: bool = False) -> torch.Tensor:
            """按行数把一次分组投影派给 GEMV 或 GEMM 内核。

            Args:
                x: ``[tokens, K]`` 激活，已按专家号排序。
                offsets: ``[E+1]`` 前缀和，设备侧。
                down_proj: ``True`` 时投影 ``[tokens, I] -> [tokens, H]``，否则
                    ``[tokens, H] -> [tokens, 2*I]``。

            Returns:
                ``[tokens, N]``，dtype 同 ``x``。

            形参不能叫 ``down``：那会遮蔽外层的 ``down_weight``，于是 ``weight.k``
            变成对 ``bool`` 取属性（实测报 ``'bool' object has no attribute 'k'``，
            而错误位置在内核入口）。
            """
            weight = down_weight if down_proj else gate_up_weight
            rows = int(x.shape[0])
            # 分界按「每专家平均行数」判，不是按总行数。
            #
            # 决定 GEMV 输赢的是每段有多长：段内只有 1-2 行时，张量核的 64x128 tile
            # 几乎全在做无效行；段内有上百行时，GEMV 每行都要串行走完 K。
            #
            # 旧写法只看 ``rows >= 64``，于是 **batch decode 掉进了 GEMM 分支**：
            # B=10 时 ``rows = 80`` 跨过阈值，而每专家平均只有 ``80 / 48 = 1.7`` 行，
            # GEMM 的 tile 里 97% 是空行。作业 ``q4bs1`` 量到那时每多一个 token 的
            # 边际成本是 48.4 ms（B=1 147.95 ms -> B=10 584.01 ms），而权重本该在
            # batch 内摊薄、单步耗时接近平坦。这一条正是那次扩展失败的怀疑点。
            rows_per_expert = rows / float(max(1, self.num_local_experts))
            if gemm_ok and rows_per_expert >= _GEMM_MIN_ROWS:
                return bops.moe_grouped_gemm(weight, x, offsets)
            return bops.moe_grouped(weight, x, offsets)

        return grouped

    def local_index(self, global_index: torch.Tensor) -> torch.Tensor:
        """把全局专家号映射成本 rank 的参数下标。

        Args:
            global_index: 任意形状的全局专家号。

        Returns:
            与入参同形状的本地下标；不属于本 rank 的项为 ``-1``。

        Raises:
            ValueError: 本实例没有分片时的行为未定义（此时全局号就是本地号，
                直接返回更快），所以这里只在分片时做映射。调用方必须先过滤。
        """
        if self.local_experts is None:
            return global_index
        table = self._local_table(global_index.device)
        return table[global_index]

    def _local_table(self, device: torch.device) -> torch.Tensor:
        """取全局号到本地下标的查找表。

        这张表只由 ``local_experts`` 与设备决定，一步 decode 里却是 48 层各调一次。
        原写法每次 ``torch.full`` + 两次 index_put，就是每层两次分配加两次内核启动；
        实测一步 decode 有 7025 次内核发射、主机侧 227 ms 中有相当比例是这类
        「每层重建一份常量」的胶水（``q4budget2`` / ``q4host1``）。

        缓存键里带设备：模块 ``.to()`` 之后旧表在另一张卡上，直接复用会索引到错设备。

        Args:
            device: 入参所在的设备。

        Returns:
            ``[num_experts]`` 的 int64 表，本地下标处为 ``0..len-1``，其余为 ``-1``。
        """
        cached = self._local_tables.get(device) if hasattr(self, "_local_tables") else None
        if cached is not None:
            return cached
        table = torch.full((self.num_experts,), -1, dtype=torch.long, device=device)
        local = torch.tensor(self.local_experts, dtype=torch.long, device=device)
        table[local] = torch.arange(len(self.local_experts), dtype=torch.long,
                                    device=device)
        if not hasattr(self, "_local_tables"):
            # 惰性建：``__init__`` 不碰它，免得非分片路径白占一份属性。
            self._local_tables = {}
        self._local_tables[device] = table
        return table

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor,
                top_k_weights: torch.Tensor) -> torch.Tensor:
        """按专家分组算，再把带权输出累加回 token。

        Args:
            hidden_states: ``[tokens, H]``。
            top_k_index: ``[tokens, top_k]`` 的**全局**专家下标。
            top_k_weights: ``[tokens, top_k]`` 的路由权重。

        Returns:
            ``[tokens, H]``。分片模式下，不属于本 rank 的槽位贡献为零——合并
            由调用方的 all-to-all 完成，本函数不做归约。

        Raises:
            ValueError: ``top_k_index`` 与 ``top_k_weights`` 形状不一致，或下标
                越界。越界检查在 W4A8 分派路径上默认跳过（见
                :data:`_CHECK_DISPATCH`），路由正确时它永远不会触发。
        """
        if top_k_index.shape != top_k_weights.shape:
            raise ValueError(
                f"top_k index/weight shape mismatch: {tuple(top_k_index.shape)} vs "
                f"{tuple(top_k_weights.shape)}.")
        if self._grouped is None or _CHECK_DISPATCH:
            if int(top_k_index.max()) >= self.num_experts or int(top_k_index.min()) < 0:
                raise ValueError(
                    f"expert index out of range [0, {self.num_experts}), got "
                    f"[{int(top_k_index.min())}, {int(top_k_index.max())}].")

        final = torch.zeros_like(hidden_states)
        tokens, top_k = top_k_index.shape
        flat_index = top_k_index.reshape(-1)
        flat_weights = top_k_weights.reshape(-1)

        # 融合分派：一次内核算出「前缀和 + 收拢后的行序」，替掉原先这么一串
        # torch 小算子（作业 q4attrib3 按调用点量到，48 层 × 每层各一遍）：
        #   nonzero / flatten / div / argsort / bincount / cumsum / chunk / mul …
        # 合计 14 个内核、672 次/步，按本卡 5.76 us 的发射地板是 3.87 ms/步，
        # 而输入只有 slots 个整数（decode 时 10 个）。见 ``moe_dispatch.cu``。
        #
        # ``local_table`` 是缓存好的全局号→本地号表；非分片时它恒等（见
        # :meth:`_local_table` 与 ``local_experts is None`` 的分支），所以这里
        # 统一走内核、不再分两条路：分派内核在非分片时就是「不减任何槽位」。
        #
        # 分片（EP）下曾经怀疑过这个内核把非本地槽位全丢了（四进程实测里 rank 1–3
        # 的 ``combine`` 输入恒为 0）。**已用「EP 时退回下面的 torch 归组」做过对照：
        # 结果完全一样**，所以嫌疑人不在这个内核，撤回那条守卫，不留一条只为绕开
        # 误判的慢路径。
        if self._dispatch is not None and self._grouped is not None:
            slots = int(flat_index.numel())
            local_table = self._local_or_identity(flat_index.device)
            offsets, slot_pos, token_of_pos, slot_of_pos = self._dispatch(
                flat_index, local_table, slots, self.num_local_experts,
                self.num_experts, top_k)
            # 有效行是**前缀**：分派内核按「每专家前缀和 + 游标」把槽位紧密排进
            # ``[0, valid)``，尾部 ``[valid, slots)`` 保持 0（见 ``moe_dispatch.cu``
            # 的前缀和与打包循环）。所以按 ``valid`` 切开只碰有效行。
            #
            # 这笔账是实测的（作业 284，prefill 512 token）：``slots = tokens x top_k
            # = 5120``，而真正落到本 rank 的只有 ``valid ≈ 1700``（每专家 11-16 行）。
            # 也就是说 gather、``torch.zeros`` 出来的输出、权重乘法、``index_add_``
            # **全都按 slots 缩放，白做 2.8 倍**。
            #
            # 代价是一次 D2H 同步（``.item()``）。原来这里刻意不读 ``offsets[-1]``
            # 是为了避开同步（decode 一步 48 次，实测主机侧被拖到 227 ms），但那是
            # decode 的形状；prefill 里一次这样的同步实测 24 us，96 次共 2.3 ms，
            # 换的是按 slots 缩放的几百毫秒。取舍在 prefill 形状上是反过来的。
            #
            # **但 decode 上这个取舍仍然是反的**（作业 285 实测）：四卡 EP4
            # 141.11 -> 157.65 ms/步（+11.7%），而 prefill 512 token
            # 2153.68 -> 2039.56 ms（−5.3%）。亏的根因不是 96x24us = 2.3 ms，而是
            # **阻塞**：decode 的 CPU 本来跑在 GPU 前面（一步发 9742 个内核），每层插
            # 两次阻塞同步就打断一次主机预取，实测代价比 2.3 ms 大 7 倍。
            #
            # 所以按 **``tokens``** 加门 —— 它是 host 已知的，零同步；不能用 ``valid``
            # 当门限（那正是"要同步才知道"的那个数）。decode 时 tokens 是个位数、
            # prefill 时是几百，量级差两个数量级，64 这个门两边都不踩线。
            #
            # **注意**：分组内核本身只扫 ``offsets[e+1] - offsets[e]`` 行（见
            # ``moe_grouped`` 的内核），尾部它本来就不碰 —— 所以这里省的是**主机侧**
            # 那几项，不是内核时间。
            if int(hidden_states.shape[0]) >= 64:
                valid = int(offsets[-1].item())
                token_of_pos = token_of_pos[:valid]
                slot_of_pos = slot_of_pos[:valid]
            else:
                # 不切：三个张量保持 slots 长度，下游行为与切片前逐位一致。
                valid = int(token_of_pos.shape[0])
            # 收拢激活：一行一个 (token, 槽位)，现在恰好是 ``valid`` 行。
            packed_state = hidden_states[token_of_pos.long()]
            if _SHAPE_TRACE:
                # 分组投影的内核耗时对「每专家行数」极敏感（rows=8 约 9 µs/专家、
                # rows=64 约 67 µs、rows=160 约 517 µs），而这条路径上一直只能靠
                # 「总 token 数 ÷ 本地专家数」去外推，已经因此误判过三次。
                # 这里直接把真实形状打出来：`rows` 是收拢后的行数、`offsets[-1]`
                # 是有效行数（内核只扫到它）、`per` 是平均每专家行数。
                _rows = int(packed_state.shape[0])
                _per = valid / max(1, self.num_local_experts)
                # 偏斜三项。``offsets.diff()`` 就是每专家行数（前缀和的差，逐元素，
                # 不走原子累加，所以这几次读数是确定的）。
                _cnt = offsets.diff()
                _maxp = int(_cnt.max().item()) if _cnt.numel() else 0
                _nz = _cnt[_cnt > 0]
                _minnz = int(_nz.min().item()) if _nz.numel() else 0
                _k = min(5, int(_cnt.numel()))
                _top5 = int(torch.topk(_cnt, _k).values.sum().item()) if _k else 0
                _top5_share = _top5 / max(1, valid)
                print(f"[moe] shape rows={_rows} valid={valid} "
                      f"per_expert={_per:.1f} max_per_expert={_maxp} "
                      f"min_nonzero={_minnz} top5_share={_top5_share:.2f} "
                      f"tokens={int(hidden_states.shape[0])} "
                      f"top_k={top_k} local_experts={self.num_local_experts}", flush=True)
            up_gate = self._grouped(packed_state, offsets)
            gate, up = up_gate.chunk(2, dim=-1)
            current_hidden = self.act_fn(gate) * up
            current_hidden = self._grouped(current_hidden, offsets, down_proj=True)
            if _SHAPE_TRACE:
                # `n`（输出宽度）必须一起打：009 的探针在「128 专家各 40 行」上量到
                # 3.56 ms/次，而生产路径在「每专家 14.3 行」上量到 10.1 ms/次 ——
                # **行少 2.8 倍、时间反多 2.8 倍**。除了每专家行数偏斜，另一个没排除的
                # 混杂项就是 `n` 与探针不同（探针用 gate_up=1280 / down=2560）。
                # 它是 host 已知量，打出来零成本。
                print(f"[moe] out_shape up_gate={tuple(up_gate.shape)} "
                      f"down={tuple(current_hidden.shape)} "
                      f"n_gate_up={int(up_gate.shape[-1])} "
                      f"n_down={int(current_hidden.shape[-1])}", flush=True)
            # 确定性合并：按**槽位序** gather 回来，再对每个 token 的 top_k 项做一次
            # 固定顺序求和。取代原来的 ``index_add_`` —— 那个在 CUDA 上走 atomicAdd，
            # 浮点加法不满足结合律 ⇒ 同一份输入两次前向结果不逐位相同（作业 287/288
            # 实测同版本两次 greedy token 就不同，self_repeat 的根因也是它）。
            # 换成定序求和之后「同一 prompt 两次结果相同」这个承诺可以重新给出。
            #
            # ``slot_pos[t]`` 是「原本第 t 个槽位收拢后的行号」，被丢弃的为 -1
            # （见 ``moe_dispatch.cu`` 的契约）；槽位序是 token-major，所以
            # ``reshape(tokens, top_k, -1).sum(1)`` 正好是每个 token 的归并。
            if current_hidden.shape[0] == 0:
                return final
            pos = slot_pos.long()
            keep = pos >= 0
            # 高级索引而不是 ``index_select``：同形状同 dtype 实测 130.67 vs 244.39 us
            # （作业 307，1.87 倍）。反过来**不要**把这个理由套到上面那个
            # ``hidden_states[token_of_pos]`` 上再改一次，它已经是高级索引了。
            rows = current_hidden[pos.clamp(min=0)]
            # 丢弃槽位**替换成 0**而不是乘 0：若 gather 到的行含 inf/nan，``0 * inf``
            # 会变 nan 并污染该 token 的结果，而 ``torch.where`` 是丢弃、不是相乘。
            rows = torch.where(keep[:, None], rows,
                               torch.zeros((), dtype=rows.dtype, device=rows.device))
            # 累加用 fp32：top_k 项的和在 bf16 下每一步都舍入，fp32 既更准也仍然是定序的。
            merged = (rows * flat_weights.to(rows.dtype)[:, None]).view(tokens, top_k, -1)
            return merged.sum(dim=1, dtype=torch.float32).to(hidden_states.dtype)

        # 分片时先扔掉不归本 rank 的槽位：它们要用 all-to-all 发出去，在这里算
        # 只会白读权重。
        if self.local_experts is not None:
            keep = self.local_index(flat_index) >= 0
            # **绝不能在这里 ``bool(keep.any())``**。那是一次设备到主机的同步：
            # ``any()`` 是设备侧归约，``bool()`` 把结果读回主机，于是主机必须等
            # 整条流跑到这里才能继续下发。一步 decode 有 48 层，就是 48 次同步，
            # 而每次同步都把已经排好的下发队列清空——实测主机侧成本 227 ms 与
            # 端到端 227.02 ms 几乎相等（``q4budget2``），设备只忙了约一半，正是
            # 这条同步在反复打断流水。
            #
            # 空槽位本来就能被下游正确处理：``nonzero`` 返回空表 → ``argsort`` /
            # ``bincount`` 得到全零桶 → 分组内核按 ``offsets`` 看到每段 0 行、
            # 一行都不算，``index_add_`` 加一个空表也是空。所以「整批都丢弃」
            # 这条快速路径**不需要**在主机上做判定，删掉它不改变语义。
            positions_all = torch.nonzero(keep, as_tuple=False).flatten()
        else:
            positions_all = torch.arange(flat_index.numel(), device=flat_index.device)

        local_index = self.local_index(flat_index[positions_all])
        weights_sel = flat_weights[positions_all]
        token_sel = torch.div(positions_all, top_k, rounding_mode="floor")

        # 按专家分组：一次只对一个专家做 GEMM，把该专家覆盖的 (token, 槽位) 行
        # 收集起来。展开成"每 token 跑 10 次完整 GEMM"会把权重读放大 10 倍。
        order = torch.argsort(local_index, stable=True)
        sorted_local = local_index[order]
        # ``minlength`` 不能省：桶数少了会静默丢掉号大的专家。号越界时多出来的
        # 桶反而把前缀和撑长，内核按 ``E+1`` 校验长度会当场拒绝，不会算错。
        counts = torch.bincount(sorted_local, minlength=self.num_local_experts)

        if self._grouped is not None:
            # 前缀和全程留在设备上：``moe_grouped`` 本来就收设备侧 int32 指针，
            # ``.tolist()`` 再 ``torch.tensor(...)`` 只是来回搬一趟，代价是每层
            # 一次主机同步。
            offsets = torch.zeros(counts.numel() + 1, dtype=torch.int32,
                                  device=counts.device)
            offsets[1:] = torch.cumsum(counts, 0)

            # 激活按专家把行收拢一次。``token_sel[order]`` 只物化这一份，累加那
            # 一步直接复用它：下游还要用它做 gather 下标，再算一遍就是一次多余的
            # ``[tokens * top_k]`` 整数索引拷贝。
            gathered_tokens = token_sel[order]
            packed_state = hidden_states[gathered_tokens]
            up_gate = self._grouped(packed_state, offsets)
            # 权重按 (token, 槽位) 取，不能按 token 取：同一 token 的 10 个专家
            # 各有各的权重，按 token 取会把它们串起来。
            gate, up = up_gate.chunk(2, dim=-1)
            current_hidden = self.act_fn(gate) * up
            current_hidden = self._grouped(current_hidden, offsets, down_proj=True)
            current_hidden = current_hidden * weights_sel[order, None]
            # 一次 ``index_add_`` 累加全部槽位，而不是像 bf16 循环那样逐专家累加。
            # 两者差了 bf16 的累加次序（末位一个 ulp），逐专家版才能和循环路径逐位
            # 相同，但那是 128 次启动；decode 一步只有 10 个槽位，不值得。
            final.index_add_(0, gathered_tokens, current_hidden.to(final.dtype))
            return final

        counts_list = counts.tolist()
        start = 0
        for expert_idx, count in enumerate(counts_list):
            if count == 0:
                continue
            rows = order[start:start + count]
            start += count
            token_idx = token_sel[rows]
            current_state = hidden_states[token_idx]
            gate, up = F.linear(current_state,
                                self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden = self.act_fn(gate) * up
            current_hidden = F.linear(current_hidden, self.down_proj[expert_idx])
            # 权重按 (token, 槽位) 取，不能按 token 取：同一 token 的 10 个专家
            # 各有各的权重，按 token 取会把它们串起来。
            current_hidden = current_hidden * weights_sel[rows, None]
            final.index_add_(0, token_idx, current_hidden.to(final.dtype))
        return final


class SparseMoeBlock(nn.Module):
    """路由 + 512 个路由专家 + 一个共享专家。

    Attributes:
        gate: :class:`TopKRouter`。
        experts: :class:`Experts`。
        shared_expert: 标准 SwiGLU 块。
        shared_expert_gate: 共享专家的逐 token 标量门。
    """

    def __init__(self, config: TextConfig,
                 local_experts: Sequence[int] | None = None,
                 quantized: bool = False) -> None:
        """建四件套。

        Args:
            config: 文本配置。
            local_experts: 本 rank 持有的全局专家号（EP 分片）；``None`` 表示
                全部。
            quantized: 专家栈建成 ``meta`` 而不分配 bf16。量化路径下这两个张量
                随后会被 :meth:`set_quantized` 删掉，但分配发生在删除之前，48 层
                每卡 60.4 GB 会在构造时就 OOM。透传给 :class:`Experts`。
        """
        super().__init__()
        self.gate = TopKRouter(config)
        self.experts = Experts(config, local_experts, quantized)
        self.shared_expert = MLP(config, intermediate_size=config.shared_expert_intermediate_size)
        # 逐 token 的标量门只有 1 行输出，仍是整步里最窄的一档：``[1, 2560]`` 行主序
        # 实测 0.782 ms，换 ``[K, N]`` 后 0.056 ms（作业 4489），48 层约 35 ms。
        self.shared_expert_gate = proj_from_config(config, config.hidden_size, 1,
                                                   bias=False)

    def forward(self, hidden_states: torch.Tensor,
                combine: callable | None = None) -> torch.Tensor:
        """``Σ_k w_k Expert_k(x) + sigmoid(g_s) * Shared(x)``。

        Args:
            hidden_states: ``[B, S, H]``；``H`` 是读门塌缩后的一路宽度。
            combine: 路由专家部分的跨 rank 归约，``(x) -> x_reduced``，语义同
                ``all_reduce(SUM)``；``None`` 表示单卡。

        Returns:
            ``[B, S, H]``。

        Raises:
            ValueError: 输入不是 3 维。

        归约**只作用在路由专家**那一路，不能放在最后：共享专家是 dense 的，
        每个 rank 都算了一份完整的 ``sigmoid(g_s) * Shared(x)``，把它也一起
        all-reduce 会让它变成 world 倍。这个错误不会报任何异常，只是 MoE 的
        输出整体偏大——所以归约点选在这里，由函数自己保证。
        """
        if hidden_states.dim() != 3:
            raise ValueError(
                f"SparseMoeBlock expects [B, S, H], got {tuple(hidden_states.shape)}.")
        batch, seq, hidden_dim = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden_dim)
        shared_out = self.shared_expert(flat)
        _, routing_weights, selected_experts = self.gate(flat)
        expert_out = self.experts(flat, selected_experts, routing_weights)
        if combine is not None:
            # 只把路由专家那一路送进归约：共享专家与路由权重每卡一致，重复相加
            # 会把它们放大 world 倍。
            expert_out = combine(expert_out)
        # 逐 token 的标量 sigmoid 门：不是逐通道，所以输出是 [tokens, 1]。
        shared_out = torch.sigmoid(self.shared_expert_gate(flat)) * shared_out
        return (expert_out + shared_out).reshape(batch, seq, hidden_dim)


__all__ = ["Experts", "MLP", "SparseMoeBlock", "TopKRouter"]
