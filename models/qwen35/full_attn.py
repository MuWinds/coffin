"""带输出门、部分 RoPE 和 int8 KV 缓存的 GQA 全注意力。

Module: ``coffin/models/qwen35``。本族组合的两种逐层注意力之一（另一种是
``linear_attn``）；``blocks.Layer`` 按层下标二选一，``model.py`` 建出全部
32 层。向下依赖 ``backends``（注意力算子、设备和 split-K 宽度）、``ops`
（dtype / rope / 归一化 / 投影），以及 ``models/qwen35/batch`` 的批处理投影。
本族之外不许直接 import 它。

32 层里有 8 层是这一种。解码步总是覆盖整个 ``max_seq`` 窗口，用设备侧位置计数器
掩掉越界项，所以形状永远不变，图只捕获一次；预填充在 key 块上做 online softmax，
score 块无论上下文多长都保持在 ~34 MB。split-K 的分片缓冲区住在这里，因为捕获图
需要稳定地址。

Constraints:
    * KV 缓存是 int8，另带一份 fp32 的逐 (头, token) scale。量化器：单 token 用
      ``_quant``，批处理用 ``_quant_rows`` / ``_quant_seq``；三者必须逐位一致，
      因为内核反量化时用的就是写入方存下的那个 scale。
    * 分片缓冲区的 split-K 宽度取自 ``caps.attn_split``，绝不写字面量：这一侧持有
      缓冲区，索引它们的是后端的内核，不一致会弄坏注意力而不是让形状检查失败。
      不切分的后端报 ``1``，缓冲区随之收缩。
    * 解码缓冲区形状静态、地址稳定。捕获图写的东西必须在 ``capture()`` 之前分配
      好，之后绝不重新指派；``enable_batching`` 和 ``attach_pool`` 是仅有的两个
      重定尺寸入口，都必须跑在捕获之前。
    * 本文件没有即时执行的回退分支。每条路径都调算子表，没有原生内核的后端从
      算子表拿到参考实现——正是它消掉了过去每个注意力调用点各写一次的内核可用性
      检查。
    * 这里没有循环状态——KV 按位置索引，回滚就是移动位置计数器——但前缀复用时
      缓存仍必须连同它一起打快照、一起恢复，否则共享的 KV 会把恢复后的位置计数器
      和别人的尾巴配在一起。
    * 所有改写都是单线程的，由 ``llm_engine`` 驱动；没有加锁，也不许加。
"""

from __future__ import annotations

import torch

from backends import caps as _caps
from backends import device as _device
from backends import ops as bops
from models.qwen35.batch import batch_proj
from models.qwen35.tuning import ATTN_SPLIT, PREFILL_ATTN_KB
from ops.dtypes import D, NEG_INF, WF
from ops.gemv import _proj
from ops.norm import RMSNorm

#: 当前选中后端的能力记录，只读一次。``batch_attn`` 决定一个 bucket 是只花一次
#: 注意力 launch 还是每行一次；两者都正确，所以这是速度分支而非回退分支。
_CAPS = _caps()


class FullAttn:
    """带输出门、部分旋转和 int8 KV 缓存的 GQA 注意力。

    注意力覆盖整个 ``max_seq`` 窗口；当前位置之后的项被掩掉，形状因此保持静态，
    便于图捕获。
    """

    def __init__(self, sd: dict, p: str, cfg: dict, bench: bool, max_seq: int,
                 paged: bool = False) -> None:
        """加载一层的注意力权重并确定它的 KV 缓冲区尺寸。

        Args:
            sd: 完整 state dict；本层只读 ``{p}.*`` 这些键。
            p: 该层在 checkpoint 里的前缀，例如 ``model.layers.3.self_attn``。
            cfg: 模型配置；``num_attention_heads``（16）、
                ``num_key_value_heads``（4）、``head_dim``（256）和
                ``rope_parameters.partial_rotary_factor``（0.25 -> 旋转 64 维）
                都来自这里。
            bench: 把这个标志透传给 ``_proj``，由它选择基准权重布局。
            max_seq: 窗口的 token 数；稠密缓存形状是 ``[nkv, max_seq, hd]``。
            paged: 完全跳过稠密缓存，因为 KV 将由池持有。见下面关于分配的注释
                ——这是内存决策，不是模式开关。
        """
        self.nh = cfg["num_attention_heads"]        # 16
        self.nkv = cfg["num_key_value_heads"]       # 4
        self.hd = cfg["head_dim"]                   # 256
        self.rep = self.nh // self.nkv
        self.rot = int(self.hd * cfg["rope_parameters"]["partial_rotary_factor"])  # 64
        self.scale = self.hd ** -0.5
        self.max_seq = max_seq

        self.qkv = _proj(sd, [f"{p}.q_proj", f"{p}.k_proj", f"{p}.v_proj"], bench)
        self.o_proj = _proj(sd, [f"{p}.o_proj"], bench)
        self.q_norm = RMSNorm(sd[f"{p}.q_norm.weight"])
        self.k_norm = RMSNorm(sd[f"{p}.k_norm.weight"])

        # int8 KV 缓存：按 (头, token) 对称量化。挂上分页池之后这些私有缓冲区被丢弃，
        # 改用池（见 ``attach_pool``）。``paged=True`` 干脆一开始就不分配：256k 窗口
        # 下它们是 4 GB 内存，池紧接着就要这块；而分配器没法可靠地把释放掉的每层
        # 0.5 GB 段复用来装池那几个大张量。
        if not paged:
            self.kcache = torch.zeros(self.nkv, max_seq, self.hd, dtype=torch.int8,
                                      device=_device)
            self.vcache = torch.zeros(self.nkv, max_seq, self.hd, dtype=torch.int8,
                                      device=_device)
            self.kscale = torch.zeros(self.nkv, max_seq, dtype=torch.float32, device=_device)
            self.vscale = torch.zeros(self.nkv, max_seq, dtype=torch.float32, device=_device)
        else:
            self.kcache = self.vcache = self.kscale = self.vscale = None
        # 批处理路径的逐行缓冲区：首维是批行（容量 1 即单流引擎，它的内核拿到下面
        # 的第 0 行视图）。``enable_batching`` 会用更多行重新分配它们；
        # ``prefill_row`` 是单流预填充/校验路径用的行，由 ``Qwen35.enable_batching``
        # 与解码行分开。
        self._bcap = 1
        self.prefill_row = 0
        self.attn_out = torch.zeros(1, self.nh, self.hd, dtype=D, device=_device)
        # 解码和批处理路径的量化目标。和这里其他缓冲区一样由调用方持有、就地写入，
        # 理由有三：融合的 ``kv_quant`` 内核只 launch 一次，而不是原来那串六七个；
        # 解码步一旦被捕获，地址就必须稳定；原来那串每次调用分配五个张量，这四个
        # 只分配一次。
        self.k_i8, self.k_s, self.v_i8, self.v_s = self._quant_buffers(1)
        # 融合旋转的目标缓冲区。一个 token 的量就够：解码路径每次调用只有一行，
        # 预填充路径自己分配（见 ``rope``）。
        self.q_rot = torch.empty(1, self.nh, self.hd, dtype=D, device=_device)
        self.k_rot = torch.empty(1, self.nkv, self.hd, dtype=D, device=_device)
        # split 解码注意力用的逐 key 切片分片量（见 attn.cu）；放在这里是因为解码步
        # 是捕获图，需要固定缓冲区。
        self.part_acc = torch.zeros(1, self.nh, ATTN_SPLIT, self.hd, dtype=torch.float32,
                                    device=_device)
        self.part_ml = torch.zeros(1, self.nh, ATTN_SPLIT, 2, dtype=torch.float32,
                                   device=_device)
        self._bind_single_views()
        self.pool = None
        self.pidx = 0

    def _quant_buffers(self, rows: int) -> tuple[torch.Tensor, ...]:
        """分配一次解码或批处理步要量化进去的四个目标缓冲区。

        收成一个辅助函数，解码路径、批处理路径和各重定尺寸入口就不会对布局产生
        分歧：``[rows, nkv, hd]`` 的 int8 和 ``[rows, nkv]`` 的 fp32，正是池的
        索引写入所取的形状，内核和缓存之间因此不夹任何 reshape 或拷贝。

        Args:
            rows: 前导容量；单流解码为一行。

        Returns:
            ``(k_i8, k_scale, v_i8, v_scale)``，全部新分配且未初始化。
        """
        i8 = torch.empty(rows, self.nkv, self.hd, dtype=torch.int8, device=_device)
        sc = torch.empty(rows, self.nkv, dtype=torch.float32, device=_device)
        return i8, sc, i8.clone(), sc.clone()

    def _bind_single_views(self) -> None:
        """单流（预填充 / 校验）路径写入的行视图。

        只重新绑定，从不重新分配：捕获的解码/预填充图就是按这些地址录的，所以
        ``enable_batching`` 在给父缓冲区重定尺寸之后调用它，而不是每次调用都绑定
        一遍视图。
        """
        pr = self.prefill_row
        self.attn_out1 = self.attn_out[pr]
        self.part_acc1 = self.part_acc[pr]
        self.part_ml1 = self.part_ml[pr]
        self.k_i81 = self.k_i8[pr]
        self.k_s1 = self.k_s[pr]
        self.v_i81 = self.v_i8[pr]
        self.v_s1 = self.v_s[pr]

    def rope(self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor,
             sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """旋转 ``q`` 和 ``k``，整次调用只 launch 一次。

        ``ops.rope.apply_rope`` 是十到十二次 launch 的 torch 链——两次表类型转换、四次
        乘法、两次给 ``rotate_half`` 用的 ``cat``、两次加法、两次拼回去的 ``cat``——
        而张量宽度不超过 ``[16, 256]``。每个解码步有 8 个全注意力层，所以这一步约
        80-96 次 launch 都花在这里；在这张卡 5.75 us 的 launch 下限面前，那些时间
        几乎全是 launch 开销而不是算术。融合入口是一次 launch。

        舍入方式微妙到值得写在调用方看得见的地方：内核把 ``cos``/``sin`` 收窄成
        bf16、每个乘积收窄成 bf16、和也收窄成 bf16，因为链里 ``cos.to(q.dtype)``
        之后那几次 bf16 ``mul``/``add`` 就是这么做的。改成 fp32 会有约 30% 的元素
        不同，并改变 logits。

        即使目标缓冲区通常和源一样大，这里也**不是**就地写：旋转把通道 ``i`` 与
        ``i + rotary_dim/2`` 配对，写进同一缓冲区的 lane 可能发现自己的搭档已经被
        另一个 lane 覆盖了。lane 之间没有顺序，所以这是竞态，不是舍入问题。解码
        路径因此写进层自己常驻的缓冲区（顺带消掉链里每步的分配），预填充路径则
        现分配，因为它行数就是分块长度，为一个每请求只跑一次、却要为最坏分块常驻
        几十 MB 每层的路径留缓冲区不值。

        Args:
            q: 查询，``[nh, hd]`` 或 ``[C, nh, hd]``，bf16，连续。
            k: 键，``[nkv, hd]`` 或 ``[C, nkv, hd]``，bf16，连续。
            cos: 余弦行，``[1, rotary_dim]`` 或 ``[C, rotary_dim]``，fp32。
            sin: 正弦行，形状同 ``cos``。

        Returns:
            ``(q_rot, k_rot)``，维数与调用方一致——单 token 是 ``[nh, hd]`` /
            ``[nkv, hd]``，一个分块则是 ``[C, ...]`` 形式。

        Raises:
            ValueError: ``q`` 和 ``k`` 的 token 数不一致，否则会旋转错误的行，而不是
                在内核里让形状检查失败。
        """
        single = q.dim() == 2
        q3 = q.unsqueeze(0) if single else q
        k3 = k.unsqueeze(0) if single else k
        rows = q3.shape[0]
        if k3.shape[0] != rows:
            raise ValueError(f"rope: q has {rows} rows, k has {k3.shape[0]}")
        c2 = cos if cos.dim() == 2 else cos.reshape(rows, -1)
        s2 = sin if sin.dim() == 2 else sin.reshape(rows, -1)
        # 解码路径复用层自己的缓冲区，所以逐 token 的一步不分配任何东西。那里 ``rows``
        # 是 1，预填充路径上是分块长度，这正是那个分支：常驻缓冲区只需覆盖解码那一行。
        if rows == 1:
            qo, ko = self.q_rot, self.k_rot
        else:
            qo = torch.empty(rows, self.nh, self.hd, dtype=D, device=_device)
            ko = torch.empty(rows, self.nkv, self.hd, dtype=D, device=_device)
        bops.rope_apply(q3, k3, c2, s2, qo, ko, self.rot)
        return (qo[0], ko[0]) if single else (qo, ko)

    def enable_batching(self, capacity: int, prefill_row: int) -> None:
        """让注意力缓冲区按批条目各占一行（须在捕获之前）。

        Args:
            capacity: 批需要的行数，已包含预填充行。
            prefill_row: 留给单流预填充和校验的行。批处理解码绝不能写它，所以模型
                把它排除在行池之外。

        Returns:
            无。缓冲区已是这个容量且暂存行未变时是空操作，一个同时跑两种模式的
            引擎因此保持捕获图的地址有效。
        """
        capacity = max(int(capacity), 1)
        if capacity == 1 and self.prefill_row == 0:
            return
        self._bcap = capacity
        self.prefill_row = int(prefill_row)
        self.attn_out = torch.zeros(capacity, self.nh, self.hd, dtype=D, device=_device)
        self.k_i8, self.k_s, self.v_i8, self.v_s = self._quant_buffers(capacity)
        self.part_acc = torch.zeros(capacity, self.nh, ATTN_SPLIT, self.hd,
                                    dtype=torch.float32, device=_device)
        self.part_ml = torch.zeros(capacity, self.nh, ATTN_SPLIT, 2,
                                   dtype=torch.float32, device=_device)
        self._bind_single_views()

    def attach_pool(self, pool, idx: int) -> None:
        """让本层的 KV 从 ``pool`` 的槽位 ``idx`` 取得，而不是稠密缓存。

        必须在捕获任何图之前调用：捕获的解码步会把池张量烘进去，之后再切换只会
        留下一张写错地方的图。私有稠密缓冲区会被释放。

        Args:
            pool: 分页 KV 池（``paged_cache``）；它的槽位表每步都要读，张量就地
                写入。
            idx: 本层在池里的槽位，也就是它在 ``pool.k`` / ``pool.v`` 逐层维度上
                的位置。

        Returns:
            无。调用之后稠密缓存属性都是 ``None``；还去读它们的代码路径会大声
            报错，而不是悄悄写进一个过期的缓冲区。
        """
        self.pool = pool
        self.pidx = idx
        self.kcache = None
        self.vcache = None
        self.kscale = None
        self.vscale = None

    def _pool_views(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """本层在池里的切片，顺序与内核取用一致。

        Returns:
            ``(k, v, kscale, vscale)`` 视图，已按 ``pidx`` 索引。它们指向池内部，
            所以透过它们写入就是缓存写入。
        """
        p = self.pool
        return p.k[self.pidx], p.v[self.pidx], p.kscale[self.pidx], p.vscale[self.pidx]

    @staticmethod
    def _quant(x: torch.Tensor, out_q: torch.Tensor,
               out_s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """[nkv, hd] bf16 -> (int8, 逐头 scale)，一次融合 launch 写进调用方缓冲区。

        这套算术是 int8 KV 缓存的契约，不是调优选择：``1e-5`` 的下限和 ``127`` 的除数
        是与回读缓存的内核之间的约定，所以 ``kv_quant`` 精确复现 torch 链
        ``abs -> amax -> clamp_min -> div -> round -> clamp_ -> to``，本方法只是那条
        链更薄的一种写法，而不是另一种。

        变的是 launch 次数和分配。那条链每次调用六到七次 launch、分配五个张量；这里
        是往层自己持有的缓冲区里发一次 launch。每个全注意力层调两次、共 8 层，那是
        解码步里约 100 次 launch，还顺带消灭了每步的分配抖动。

        Args:
            x: 一个 token 的 K 或 V，``[nkv, hd]``，计算 dtype。
            out_q: int8 目标，``[nkv, hd]``，就地写入。
            out_s: fp32 scale 目标，``[nkv]``，就地写入。

        Returns:
            ``(out_q, out_s)``：``[-127, 127]`` 的 int8 码，以及内核会乘回去的逐头
            正 fp32 scale。
        """
        return bops.kv_quant(x, out_q, out_s), out_s

    def __call__(self, x: torch.Tensor, pos1, cos_tbl, sin_tbl, mask_idx,
                 rope_delta=None) -> torch.Tensor:
        """针对一行的单 token 解码步。

        Args:
            x: 该 token 的层输入，``[hidden]``。
            pos1: 只含一个元素的 long 张量，值是该 token 下标。它寻址 KV 槽位和
                缓存写入，掩码条件是 ``> pos1``。
            cos_tbl: RoPE 余弦表，按位置索引。
            sin_tbl: RoPE 正弦表，按位置索引。
            mask_idx: 设备侧位置计数器，CPU 回退掩码用。
            rope_delta: 可选的 mRoPE 偏移，只加在旋转用的 ``pos1`` 上。纯文本时为
                零（``None``），文本路径因此保持和图像功能出现之前逐位相同。

        Returns:
            ``[hidden]``，经 ``o_proj`` 之后的加门注意力输出。
        """
        qg, k, v = self.qkv(x)
        qg = qg.reshape(self.nh, 2 * self.hd)
        q, gate = qg[:, :self.hd], qg[:, self.hd:]
        q = self.q_norm(q)
        k = self.k_norm(k.reshape(self.nkv, self.hd))
        v = v.reshape(self.nkv, self.hd)

        # 在本方法其余地方 ``pos1`` 始终是*token 下标*（寻址 KV 槽位的就是它）；只有
        # 旋转去读 mRoPE 位置，prompt 含图像之后那就是
        # ``token_index + rope_delta``。没有图像时 ``rope_delta`` 为 0，查表和以前
        # 逐位相同。
        idx = pos1 if rope_delta is None else pos1 + rope_delta
        cos = cos_tbl.index_select(0, idx)
        sin = sin_tbl.index_select(0, idx)
        q, k = self.rope(q, k, cos, sin)

        qk, sk = self._quant(k, self.k_i81, self.k_s1)
        qv, sv = self._quant(v, self.v_i81, self.v_s1)
        if self.pool is not None:
            self.pool.write_slot(self.pidx, self.pool.slots_for(pos1, self.prefill_row),
                                 qk, sk, qv, sv)
        else:
            self.kcache.index_copy_(1, pos1, qk.unsqueeze(1))
            self.vcache.index_copy_(1, pos1, qv.unsqueeze(1))
            self.kscale.index_copy_(1, pos1, sk.unsqueeze(1))
            self.vscale.index_copy_(1, pos1, sv.unsqueeze(1))

        # 每条路径一次算子调用：反量化 k/v、折叠 scale、施加因果掩码、softmax、加门、
        # PV。由哪个内核应答是后端的事；两个入口在每个后端上都可调用，所以既没有
        # 可用性分叉，也没有需要保持数值一致的即时执行孪生体。
        if self.pool is not None:
            kp, vp, kps, vps = self._pool_views()
            out = bops.attn_paged(q.contiguous(), gate.contiguous(), kp, vp, kps, vps,
                                  self.pool.pos2slot[self.prefill_row], self.attn_out1,
                                  pos1, self.max_seq, self.rep, self.scale,
                                  self.part_acc1, self.part_ml1)
        else:
            out = bops.attn_decode(q.contiguous(), gate.contiguous(), self.kcache,
                                   self.vcache, self.kscale, self.vscale,
                                   self.attn_out1, pos1, self.rep, self.scale,
                                   self.part_acc1, self.part_ml1)
        return self.o_proj(out.reshape(-1))

    @staticmethod
    def _quant_rows(x: torch.Tensor, out_q: torch.Tensor,
                    out_s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """[B, nkv, hd] bf16 -> (int8, 逐 (行, 头) scale)，逐行等价于 ``_quant``。

        这里的行是批条目，所以 ``[nkv, hd]`` 那对轴变成 ``kv_quant`` 的扁平行：
        reshape 成 ``[B*nkv, hd]`` 之后，融合内核的一行恰好是一个 (行, 头) 对，正是
        参考实现用 ``amax(-1)`` 做的归约。目标缓冲区是层自己的，按同样方式取视图，
        批处理路径因此不分配，并且直接写进池索引写入所取的布局。

        Args:
            x: 每个批行一个 token，``[B, nkv, hd]``。
            out_q: int8 目标，``[B, nkv, hd]``，就地写入。
            out_s: fp32 目标，``[B, nkv]``，就地写入。

        Returns:
            ``(out_q, out_s)``。每行独立量化，所以第 ``b`` 行的结果与单独对那一行调
            :meth:`_quant` 逐位相同。
        """
        B = x.shape[0]
        flat = out_q.view(B * x.shape[1], x.shape[2])
        scales = out_s.view(B * x.shape[1])
        bops.kv_quant(x.reshape(B * x.shape[1], x.shape[2]), flat, scales)
        return out_q, out_s

    def batch(self, x: torch.Tensor, positions: torch.Tensor, ctx) -> torch.Tensor:
        """``x`` [B, hidden] -> [B, hidden]：批处理投影，逐行注意力。

        这里三个阶段里有两个是批处理的：投影（所有行共用一次权重读取，见
        ``models/qwen35/batch.py``），以及——在后端原生提供批处理分页注意力时——
        注意力本身，于是整个 bucket 只 launch 一次而不是 B 次。每一行仍通过自己的
        ``pos2slot`` 行解析位置、在自己那行的 ``context_lens[b]`` 处停下，内核的
        数值与单行版本完全相同。后端没有批处理入口时，它对行切片逐行调用单 token
        入口；两条路径必须逐行逐位一致。

        ``ctx`` 可以带第 5 项：逐行的 ``rope_delta``，长度 ``[B]``（带图像的请求的
        mRoPE 文本偏移）。它只移动 cos/sin 查找；``positions`` 继续寻址 KV 槽位。

        Args:
            x: 解码行输入，``[B, hidden]``。
            positions: 每行的当前 token 下标，长度 ``[B]``。
            ctx: ``(cos_tbl, sin_tbl, mask_idx[, context_lens[, rope_delta]])``。
                ``context_lens`` 是 ``[B]`` 的 int32，用来选择批处理入口；
                ``rope_delta`` 是 ``[B]`` 的 long，只移动旋转。

        Returns:
            ``[B, hidden]``，经批处理 ``o_proj`` 之后的加门注意力输出。

        Raises:
            RuntimeError: 没有挂分页池。批处理路径要读池的槽位表，所以调用方必须
                在打开批处理之前挂好池，否则就留在单流解码。这是配置错误，不是缺
                内核：每个后端都应答两个注意力入口。
        """
        cos_tbl, sin_tbl, mask_idx = ctx[0], ctx[1], ctx[2]
        ctx_lens = ctx[3] if len(ctx) > 3 else None
        rope_delta = ctx[4] if len(ctx) > 4 else None
        B = x.shape[0]
        if self.pool is None:
            raise RuntimeError("batched attention needs a paged pool (enable_paging())")
        qg, k, v = batch_proj(self.qkv, x)
        qg = qg.reshape(B, self.nh, 2 * self.hd)
        q, gate = qg[..., :self.hd], qg[..., self.hd:]
        q = self.q_norm(q)
        k = self.k_norm(k.reshape(B, self.nkv, self.hd))
        v = v.reshape(B, self.nkv, self.hd)

        # [B, 1, rot]：那个单元素维是块内核期望的 token 轴。
        rope_pos = positions if rope_delta is None else positions + rope_delta[:B]
        cos = cos_tbl.index_select(0, rope_pos)
        sin = sin_tbl.index_select(0, rope_pos)
        q, k = self.rope(q, k, cos, sin)

        kp, vp, kps, vps = self._pool_views()
        if ctx_lens is not None and _CAPS.batch_attn:
            # 整个 bucket 一次 launch：量化每一行、把每行的 K/V 写进各自槽位，然后
            # 一趟分页注意力。
            qk, sk = self._quant_rows(k, self.k_i8[:B], self.k_s[:B])
            qv, sv = self._quant_rows(v, self.v_i8[:B], self.v_s[:B])
            slots = self.pool.pos2slot[:B].gather(1, positions.view(B, 1)).squeeze(1)
            self.pool.write_rows(self.pidx, slots, qk, sk, qv, sv)
            bops.attn_paged_batch(q, gate, kp, vp, kps, vps,
                                  self.pool.pos2slot[:B], ctx_lens[:B],
                                  self.attn_out[:B], self.max_seq, self.rep, self.scale,
                                  self.part_acc[:B], self.part_ml[:B])
        else:
            for b in range(B):
                qk, sk = self._quant(k[b], self.k_i8[b], self.k_s[b])
                qv, sv = self._quant(v[b], self.v_i8[b], self.v_s[b])
                pos_b = positions[b:b + 1]
                self.pool.write_slot(self.pidx, self.pool.slots_for(pos_b, b),
                                     qk, sk, qv, sv)
                bops.attn_paged(q[b].contiguous(), gate[b].contiguous(), kp, vp, kps, vps,
                                self.pool.pos2slot[b], self.attn_out[b], pos_b,
                                self.max_seq, self.rep, self.scale,
                                self.part_acc[b], self.part_ml[b])
        # 按容量定尺寸的缓冲区只写了前 ``B`` 行；切片让投影的 K 维保持为 nh*hd。
        return batch_proj(self.o_proj, self.attn_out[:B].reshape(B, -1))

    @staticmethod
    def _quant_seq(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """[C, nkv, hd] bf16 -> (int8, 逐 (token, 头) scale)。

        和解码、批处理路径不同，这一条会分配：``C`` 是预填充分块长度，逐层常驻缓冲区
        就得按引擎可能接受的最长分块定尺寸，而且整轮运行都占着。预填充是计算受限的，
        且每请求跑一次而不是每 token 一次，所以这里值得要的是省下的那一次 launch，
        分配倒不值得消掉。

        Args:
            x: 若干 token 的 K 或 V，``[C, nkv, hd]``，计算 dtype。

        Returns:
            ``(q, s)``：``[C, nkv, hd]`` 的 int8 码和 ``[C, nkv]`` 的 fp32 scale。
            逐 token 归约与 :meth:`_quant` 相同，所以分块预填充和逐 token 写入的量化
            结果一致。
        """
        C, nkv = x.shape[0], x.shape[1]
        rows = C * nkv
        q = torch.empty(x.shape, dtype=torch.int8, device=x.device)
        s = torch.empty(C, nkv, dtype=torch.float32, device=x.device)
        bops.kv_quant(x.reshape(rows, x.shape[2]), q.view(rows, x.shape[2]), s.view(rows))
        return q, s

    def prefill_project(self, x: torch.Tensor):
        """预填充的投影半边：``[C, hidden]`` -> 原始 ``(qg, k, v)``。

        拆出来是为了让组预填充对若干个序列拼接后的 prompt 只跑一次 GEMM，事后再
        切分结果。

        Args:
            x: 预填充输入，``[C, hidden]``。

        Returns:
            经过 q/k 归一化和 rope 之前的 ``(qg, k, v)``，也就是调用方可以拼接、
            之后再交给 :meth:`prefill_attend` 的原始投影。
        """
        return self.qkv.prefill(x)

    def enable_multi_q(self, capacity: int, width: int) -> None:
        """多查询校验的缓冲区：每个 (行, token) 查询对占一行。

        Args:
            capacity: 解码行数。
            width: 每行草拟的 token 数（``c``）；一行及其 ``c`` 个查询变成首维上的
                ``c`` 个条目。

        Returns:
            无。缓冲区只在这里分配，因为校验图捕获的就是这些地址。
        """
        self._mq_width = int(width)
        n = capacity * int(width)
        self.mq_out = torch.zeros(n, self.nh, self.hd, dtype=D, device=_device)
        self.mq_part_acc = torch.zeros(n, self.nh, ATTN_SPLIT, self.hd,
                                       dtype=torch.float32, device=_device)
        self.mq_part_ml = torch.zeros(n, self.nh, ATTN_SPLIT, 2,
                                      dtype=torch.float32, device=_device)

    def verify_batch_multi(self, qg: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                           positions: torch.Tensor, ctx, rows: int) -> torch.Tensor:
        """整个一次前向校验只调一次的多查询注意力。

        ``qg``/``k``/``v`` 是对 ``rows*c`` 个 token 的层投影（行主序），``positions``
        是 ``[rows, c]``。每个 token 的 K/V 写进各自槽位（``rows*c`` 个 token 一次
        索引写入），然后 ``attn_paged_multi_q`` 用一趟遍历该行历史 KV 的扫描，服务
        该行全部 ``c`` 个查询。返回 ``o_proj`` 之前的注意力输出
        ``[rows*c, nh*hd]``；调用方只跑一次宽投影。

        Args:
            qg: 打包的查询与门，``[rows*c, 2*nh*hd]``。
            k: 键，``[rows*c, nkv*hd]``。
            v: 值，``[rows*c, nkv*hd]``。
            positions: token 下标，``[rows, c]`` 的 long，行内行主序。
            ctx: ``(cos_tbl, sin_tbl[, rope_delta_rows])``；第 5 项是逐行 mRoPE
                偏移，只作用在旋转上。
            rows: 行数 ``b``；``c`` 从 ``positions`` 读。

        Returns:
            ``[rows*c, nh*hd]``，``o_proj`` 之前的加门注意力输出，调用方因此可以对
            整个校验批只做一次那笔权重读取。
        """
        cos_tbl, sin_tbl = ctx[0], ctx[1]
        b = rows
        c = positions.shape[1]
        n = b * c
        qg = qg.view(b, c, self.nh, 2 * self.hd)
        q, gate = qg[..., :self.hd], qg[..., self.hd:]
        q = self.q_norm(q)
        k = self.k_norm(k.view(b, c, self.nkv, self.hd))
        v = v.view(b, c, self.nkv, self.hd)
        # 旋转的位置是 ``token 下标 + rope_delta``；KV 槽位和逐行窗口仍停在 token
        # 下标上。``[b*c]`` 行，一次 launch 就覆盖整个校验窗口。
        rope_pos = positions if len(ctx) < 5 else positions + ctx[4][:b, None]
        cos = cos_tbl.index_select(0, rope_pos.reshape(-1))
        sin = sin_tbl.index_select(0, rope_pos.reshape(-1))
        q, k = self.rope(q.reshape(n, self.nh, self.hd),
                         k.reshape(n, self.nkv, self.hd), cos, sin)
        q = q.view(b, c, self.nh, self.hd)
        k = k.view(b, c, self.nkv, self.hd)

        # 每个校验 token 的 K/V 都写进去，然后多查询注意力为全部 c 个查询只读一次
        # 每行的历史（内核会把查询 j 掩在位置 j，所以一趟就服务整个投机块）。
        kq, ks = self._quant_seq(k.reshape(n, self.nkv, self.hd))
        vq, vs = self._quant_seq(v.reshape(n, self.nkv, self.hd))
        slots = torch.stack([self.pool.pos2slot[i].index_select(0, positions[i])
                             for i in range(b)]).reshape(n)
        self.pool.write_rows(self.pidx, slots, kq, ks, vq, vs)

        kp, vp, kps, vps = self._pool_views()
        lens = positions[:, 0].to(torch.int32).contiguous()
        out = self.mq_out[:n].view(b, c, self.nh, self.hd)
        from models.qwen35.batch import _count
        _count("multi_q")
        bops.attn_paged_multi_q(q.contiguous(), gate.contiguous(), kp, vp, kps, vps,
                                self.pool.pos2slot[:b], lens, out,
                                self.max_seq, self.rep, self.scale,
                                self.mq_part_acc[:n], self.mq_part_ml[:n])
        return out.reshape(n, self.nh * self.hd)

    def verify_batch_step(self, qg: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                          positions: torch.Tensor, ctx) -> torch.Tensor:
        """从已投影好的 q/k/v 出发，为 ``B`` 行处理一个校验 token。

        注意力和解码步相同（融合的分页内核，数字因此与贪心解码一致），整个 bucket
        一次批处理 launch：校验图把 ``B*c`` 个 token 一次投影完，然后让 ``c`` 个
        token 位置逐个走这个入口。``ctx`` 可以把逐行 ``rope_delta`` 作为第 5 项
        带上（图像之后的 mRoPE；``positions`` 继续寻址 KV 槽位）。

        Args:
            qg: 每行 ``c`` 个 token 的已投影查询与门，``[B*c, 2*nh*hd]``。
            k: 已投影的键，``[B*c, nkv*hd]``。
            v: 已投影的值，``[B*c, nkv*hd]``。
            positions: token 下标，``[B, c]`` 的 long；既用来放 KV，也在加上
                ``rope_delta`` 后用来查旋转。
            ctx: ``(cos_tbl, sin_tbl, mask_idx)``；第 5 项是逐行 mRoPE 偏移。

        Returns:
            ``[B, hidden]``，经 ``o_proj`` 之后的加门输出——每个校验 token 一行，
            调用方拿它和草稿比较来决定接受还是拒绝。
        """
        cos_tbl, sin_tbl, mask_idx = ctx[0], ctx[1], ctx[2]
        b = qg.shape[0]
        qg = qg.reshape(b, self.nh, 2 * self.hd)
        q, gate = qg[..., :self.hd], qg[..., self.hd:]
        q = self.q_norm(q)
        k = self.k_norm(k.reshape(b, self.nkv, self.hd))
        v = v.reshape(b, self.nkv, self.hd)
        rope_pos = positions if len(ctx) < 5 else positions + ctx[4][:b]
        cos = cos_tbl.index_select(0, rope_pos)
        sin = sin_tbl.index_select(0, rope_pos)
        q, k = self.rope(q, k, cos, sin)
        qk, sk = self._quant_rows(k, self.k_i8[:b], self.k_s[:b])
        qv, sv = self._quant_rows(v, self.v_i8[:b], self.v_s[:b])
        kp, vp, kps, vps = self._pool_views()
        slots = self.pool.pos2slot[:b].gather(1, positions.view(b, 1)).squeeze(1)
        self.pool.write_rows(self.pidx, slots, qk, sk, qv, sv)
        # ``attn_paged_batch`` 要的是长度而不是末下标，所以 +1 把 token 位置变成计数。
        # 就地做，因为这是捕获路径。
        lens = positions.to(torch.int32).add_(1)
        bops.attn_paged_batch(q, gate, kp, vp, kps, vps,
                              self.pool.pos2slot[:b], lens,
                              self.attn_out[:b], self.max_seq, self.rep, self.scale,
                              self.part_acc[:b], self.part_ml[:b])
        return self.o_proj.prefill(self.attn_out[:b].reshape(b, -1))

    def prefill_attend(self, qg: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                       positions: torch.Tensor, ctx, tiles: list | None = None,
                       row: int | None = None, rope=None) -> torch.Tensor:
        """预填充的注意力半边：原始 ``(qg, k, v)`` -> 加门后的 ``[C, nh*hd]``。

        把 C 个 token 写进 ``row`` 的 KV 槽位，并在该行自己的窗口上做注意力。
        ``o_proj`` 留给调用方，组预填充因此可以对所有序列的输出只跑它一次（那是一笔
        权重读取）。

        ``rope`` 是可选的显式 ``(cos, sin)`` 对，形状 ``[C, rot]``
        （``ops.mrope.MRopeMapper``），用于含图像 token 的分块：那里三根位置轴逐
        token 不同，单次查表给不出旋转，而因果/KV 那一侧仍用 ``positions``。
        ``None`` 时一维表路径保持不变。当参数为 ``None`` 但 ``ctx`` 带了第 4 项时，
        那一项就是 rope 对：预填充调用链（``Layer.prefill`` -> ``prefill_state``）
        只转发 ``ctx``，而 ``FullAttn.batch`` 对 ``ctx_lens`` 用的也是这种透传方式。

        Args:
            qg: 打包的查询与门，``[C, 2*nh*hd]``。
            k: 键，``[C, nkv*hd]``。
            v: 值，``[C, nkv*hd]``。
            positions: 该分块的绝对 token 下标，长度 ``[C]``；它们既寻址 KV 槽位，
                在没有显式 ``rope`` 时也用来查旋转表。
            ctx: ``(cos_tbl, sin_tbl, mask_idx[, rope])``。
            tiles: 来自 ``Qwen35.prefill`` 的 ``(q0, q1, p0, kend)`` 查询块。
                ``None`` 把整个分块当成一块，结果正确，但会浪费 :meth:`prefill`
                里说的上三角工作量。
            row: 要写的槽位表行。默认是暂存行。
            rope: 含图像 token 的分块要用的显式 ``(cos, sin)``，``[C, rot]``；
                这种分块单次查表给不出三轴旋转。

        Returns:
            ``[C, nh*hd]``，``o_proj`` 之前的加门注意力输出。
        """
        if row is None:
            row = self.prefill_row
        cos_tbl, sin_tbl, mask_idx = ctx[0], ctx[1], ctx[2]
        if rope is None and len(ctx) > 3:
            rope = ctx[3]
        C = qg.shape[0]
        # 有 tiles 时 key 范围以它为准：它的边界已经把调用方从前面切掉的前缀算进去了。
        if tiles is None:
            p0, p1 = int(positions[0]), int(positions[-1]) + 1
        else:
            p0, p1 = tiles[0][2], tiles[-1][3]
        qg = qg.reshape(C, self.nh, 2 * self.hd)
        q, gate = qg[..., :self.hd], qg[..., self.hd:]
        q = self.q_norm(q)
        k = self.k_norm(k.reshape(C, self.nkv, self.hd))
        v = v.reshape(C, self.nkv, self.hd)

        if rope is None:
            cos = cos_tbl.index_select(0, positions)
            sin = sin_tbl.index_select(0, positions)
        else:
            # mRoPE 的对到达时已经 gather 好、是 [C, rot] 的 fp32；内核逐元素收窄它，
            # 也就是原来链里 ``.to(dtype)`` 干的事。
            cos = rope[0].to(cos_tbl.dtype)
            sin = rope[1].to(sin_tbl.dtype)
        q, k = self.rope(q, k, cos, sin)

        qk, sk = self._quant_seq(k)
        qv, sv = self._quant_seq(v)
        if self.pool is not None:
            # 分页时：把 C 个新 token 散写进池，再为注意力本身读回一个稠密视图。预填充
            # 是计算受限的，实体化那个视图代价不大，还能让这条路径与稠密版保持同一份
            # 实现（参考实现的分页版也这么做）。
            slots = self.pool.slots_for(positions, row)
            self.pool.write_range(self.pidx, slots, qk, sk, qv, sv)
            kk, vv, kkp, vvp = self.pool.gather(self.pidx, p1, row=row)
        else:
            self.kcache[:, p0:p1] = qk.transpose(0, 1)
            self.vcache[:, p0:p1] = qv.transpose(0, 1)
            self.kscale[:, p0:p1] = sk.transpose(0, 1)
            self.vscale[:, p0:p1] = sv.transpose(0, 1)
            # [nkv, S, hd] 的 int8：这次分块只寻址到 p1 为止的前缀，尾部永远碰不到，
            # 不需要清零。
            kk = self.kcache[:, :p1]
            vv = self.vcache[:, :p1]
            kkp = self.kscale[:, :p1]
            vvp = self.vscale[:, :p1]

        # 两个注意力矩阵乘都用 fp16 操作数，原因和投影一样：这颗芯片有 packed fp16
        # 计算、没有 packed bf16 计算，同一个 einsum 快约 2-3x（两边累加器都仍是
        # fp32）。块里其余部分保持 bf16，KV 量化器和门都是照它写的。加宽是*按 key
        # 块*做的，不是对整段前缀做一次：整段加宽每个上下文 token 要 4 KB（256k 时
        # 1.07 GB），而且那份缓冲区得在整个层调用期间活着，长 prompt 负担不起。
        if tiles is None:
            tiles = [(0, C, p0, p1)]
        KB = PREFILL_ATTN_KB
        out = torch.empty(C, self.nh, self.hd, dtype=D, device=_device)
        for q0, q1, _p0, kend in tiles:
            nq = q1 - q0
            qe = q[q0:q1].reshape(nq, self.nkv, self.rep, self.hd).permute(1, 2, 0, 3)
            qf = qe.to(WF)
            qpos = positions[q0:q1]
            # 在 key 块上做 online softmax：acc 是未归一化输出，m 是运行最大值，l 是
            # 运行分母，全部形状 [nkv, rep, nq, .]
            acc = m = l = None
            for k0 in range(0, kend, KB):
                k1 = min(k0 + KB, kend)
                kb16 = kk[:, k0:k1].to(WF)
                scores = torch.einsum("ghcd,gsd->ghcs", qf, kb16).float()
                # 用一次融合缩放代替两次乘法，并就地施加因果掩码：score 块是层里最大的
                # 张量，多遍历它一遍都值得避免。`tiles` 带的是绝对位置，掩码和缓存
                # 切片因此直接对齐。
                scores *= (kkp[:, k0:k1] * self.scale)[:, None, None, :]
                scores.masked_fill_(
                    (mask_idx[None, k0:k1] > qpos[:, None])[None, None, :, :], NEG_INF)
                bmax = scores.amax(-1, keepdim=True)
                if acc is None:
                    # 第一个块总是包含 key 0，每个查询都看得到它，所以这里 bmax 是
                    # 有限值，不会有 -inf 运算漏进来。
                    m = bmax
                    p = (scores - m).exp()
                    l = p.sum(-1, keepdim=True)
                else:
                    mnew = torch.maximum(m, bmax)
                    alpha = (m - mnew).exp()
                    p = (scores - mnew).exp()
                    l = l * alpha + p.sum(-1, keepdim=True)
                    acc *= alpha
                    m = mnew
                # 把逐 token 的 V scale 折进权重（不分块的版本也这么做），einsum 就在
                # 原始 int8 加宽后的 V 上跑。
                pw = (p * vvp[:, None, None, k0:k1]).to(WF)
                pv = torch.einsum("ghcs,gsd->ghcd", pw, vv[:, k0:k1].to(WF))
                acc = pv if acc is None else acc + pv
            # acc/l 是 (nkv, rep, nq, hd) 下归一化后的切块输出；permute 在加门之前把
            # 它恢复成 (nq, nh, hd)。
            out[q0:q1] = (acc / l).permute(2, 0, 1, 3).reshape(nq, self.nh, self.hd).to(D)
        out = out * gate.sigmoid()
        return out.reshape(C, -1)

    def prefill(self, x: torch.Tensor, positions: torch.Tensor, ctx,
                tiles: list | None = None, row: int | None = None,
                rope=None) -> torch.Tensor:
        """``x`` [C, hidden] -> [C, hidden]；把 C 个 token 写进 KV 缓存。

        给了 ``tiles``（来自 ``Qwen35.prefill`` 的 ``(q0, q1, kend)`` 三元组）时，
        注意力逐查询块跑，key 范围只到该块自己的末尾为止。这不只是 L2 层面的技巧：
        不分块的版本会构造完整的 ``[nkv, rep, C, p1]`` score 块再掩掉上三角，于是
        pos=0 时一半（pos=2048 时四分之一）的矩阵乘和 softmax 计算被白白扔掉。分块
        把 score/softmax/PV 的流量按同样比例砍掉，块还小到能留在缓存里。``row`` 选择
        槽位表行（默认暂存行）；批处理草稿器传各序列自己的行。``rope`` 是含图像 token
        的分块要用的可选显式三维 ``(cos, sin)``（见 :meth:`prefill_attend`）。

        Args:
            x: 预填充输入，``[C, hidden]``。
            positions: 该分块的绝对 token 下标，长度 ``[C]``。
            ctx: ``(cos_tbl, sin_tbl, mask_idx[, rope])``。
            tiles: 来自 ``Qwen35.prefill`` 的 ``(q0, q1, p0, kend)`` 查询块。
            row: 要写的槽位表行；``None`` 用暂存行。
            rope: 含图像 token 的分块要用的显式三维 ``(cos, sin)``。

        Returns:
            ``[C, hidden]``，经 ``o_proj`` 之后的层输出，顺序与 ``x`` 一致，残差
            相加才能对齐。
        """
        proj = self.prefill_project(x)
        out = self.prefill_attend(*proj, positions, ctx, tiles,
                                  self.prefill_row if row is None else row, rope)
        return self.o_proj.prefill(out)

    def verify(self, x: torch.Tensor, positions, cos_tbl, sin_tbl, mask_idx,
               rope_delta=None) -> torch.Tensor:
        """``x`` [C, hidden] -> [C, hidden]，处理 ``positions`` 处的 C 个草稿 token。

        投影是批处理的；之后每个 token 都走与解码步相同的融合注意力，写自己的 KV
        槽位。这里的回滚是免费的：缓存按位置索引，倒回就是移动位置计数器。
        ``rope_delta``（可选，0 维或长度 ``[1]``）是该请求的 mRoPE 偏移：
        ``positions`` 继续寻址 KV 槽位，旋转读的是 ``positions + rope_delta``。

        靠位置计数器回滚之所以成立，是因为被拒的草稿什么都不留：下一次预填充会覆盖
        那些槽位。这也是循环在注意之前先写每个 token 的 KV 的原因——被拒的 token 不会
        再被读到。

        Args:
            x: 草稿 token 的输入，``[C, hidden]``。
            positions: 草稿 token 的 token 下标，长度 ``[C]``。
            cos_tbl: RoPE 余弦表。
            sin_tbl: RoPE 正弦表。
            mask_idx: 设备侧位置计数器，供掩码使用。
            rope_delta: 可选的 mRoPE 偏移，0 维或长度 ``[1]``；只移动旋转。

        Returns:
            ``[C, hidden]`` 层输出，每个草稿 token 一行，采样器拿它和草稿比较，接受
            最长的前缀。
        """
        C = x.shape[0]
        qg, k, v = self.qkv.prefill(x)
        qg = qg.reshape(C, self.nh, 2 * self.hd)
        q, gate = qg[..., :self.hd], qg[..., self.hd:]
        q = self.q_norm(q)
        k = self.k_norm(k.reshape(C, self.nkv, self.hd))
        v = v.reshape(C, self.nkv, self.hd)
        rope_pos = positions if rope_delta is None else positions + rope_delta.reshape(())
        out = torch.empty(C, self.nh, self.hd, dtype=D, device=_device)
        kp, vp, kps, vps = self._pool_views() if self.pool is not None else (None,) * 4
        for i in range(C):
            qi, ki, vi = q[i], k[i], v[i]
            cos = cos_tbl.index_select(0, rope_pos[i:i + 1])
            sin = sin_tbl.index_select(0, rope_pos[i:i + 1])
            qi, ki = self.rope(qi, ki, cos, sin)
            qk, sk = self._quant(ki, self.k_i81, self.k_s1)
            qv, sv = self._quant(vi, self.v_i81, self.v_s1)
            if self.pool is not None:
                self.pool.write_slot(self.pidx,
                                     self.pool.slots_for(positions[i:i + 1],
                                                         self.prefill_row),
                                     qk, sk, qv, sv)
                bops.attn_paged(qi.contiguous(), gate[i].contiguous(), kp, vp, kps, vps,
                                self.pool.pos2slot[self.prefill_row], self.attn_out1,
                                positions[i:i + 1],
                                self.max_seq, self.rep, self.scale,
                                self.part_acc1, self.part_ml1)
            else:
                self.kcache.index_copy_(1, positions[i:i + 1], qk.unsqueeze(1))
                self.vcache.index_copy_(1, positions[i:i + 1], qv.unsqueeze(1))
                self.kscale.index_copy_(1, positions[i:i + 1], sk.unsqueeze(1))
                self.vscale.index_copy_(1, positions[i:i + 1], sv.unsqueeze(1))
                bops.attn_decode(qi.contiguous(), gate[i].contiguous(),
                                 self.kcache, self.vcache,
                                 self.kscale, self.vscale, self.attn_out1,
                                 positions[i:i + 1],
                                 self.rep, self.scale, self.part_acc1, self.part_ml1)
            out[i] = self.attn_out1
        return self.o_proj.prefill(out.reshape(C, -1))

    def reset(self) -> None:
        """在请求之间清空稠密 KV 缓存。

        Returns:
            无。清零对下一次预填充的正确性并非必需——位置在读取前就被覆写了——但上一个
            请求残留的 int8 scale 会被任何读过当前长度的路径反量化，所以缓冲区从干净
            状态开始。

        只清稠密缓存：挂上池之后槽位归会话所有，在这里清零会抹掉别的序列共享的
        前缀。
        """
        if self.pool is not None:
            return
        self.kcache.zero_()
        self.vcache.zero_()
        self.kscale.zero_()
        self.vscale.zero_()
