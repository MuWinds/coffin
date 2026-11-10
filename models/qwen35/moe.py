"""Qwen3.5-MoE 的稀疏前馈块：路由器、共享专家和一整栈路由专家。

同一族里稠密层用 :class:`~models.qwen35.blocks.MLP`，MoE 层用本模块；两者的构造入口
和 ``__call__`` / ``prefill`` / ``batch`` / ``wide`` 四个调用形态一致，所以
:class:`~models.qwen35.blocks.Layer` 只按配置挑一个，其余代码看不出区别。

参考实现是 ``transformers`` 5.8.1 的 ``Qwen3_5MoeSparseMoeBlock``，语义照抄：
``softmax(fp32) -> topk -> 除以 topk 概率之和``，专家侧是逐专家的
``down(silu(gate(x)) * up(x))``，共享专家的输出再乘一个 ``sigmoid(x @ W_gate)`` 标量。
没有 ``norm_topk_prob`` 开关——那个配置项在 5.8.1 里不存在，归一化恒开；把它写成
可配会让一个语义分叉看起来像调优项。

Module: ``coffin/models/qwen35``。依赖 ``ops``（投影、激活）、``backends``（分组专家
内核）和同级 ``batch``、``parallel``；由 ``blocks.Layer`` 构造。

Constraints:
    * 专家权重按融合栈存放，不是 256 个独立投影：int8 是 ``[E, K, N]``（kn 布局，
      N 连续），W4A8 是 packed ``[E, ceil(K/2), N]``，稠密是 ``[E, 2I, H]``。分组内核
      按固定偏移寻址整块连续内存，摊成 256 份会让「第 e 个专家的起点」算错，而形状
      检查照样过。
    * 路由矩阵（``mlp.gate``）与共享专家门（``mlp.shared_expert_gate``）保持稠密。
      4-bit 量化噪声足够翻转 top-8 的选择，而选错专家是整层输出错，不是精度变差。
      ``tools/quant_disk_w4a8.py`` 的 ``DENSE_SUFFIXES`` 是这条约束的执行点。
    * 归并必须定序：``index_add_`` 在本卡上走原子加，同一输入两次前向不逐位相同，
      「同一 prompt 两次 greedy 一致」这条承诺就没了。这里改成按槽位写入
      ``[tokens * top_k, H]`` 再对 top_k 求定序和。
"""

from __future__ import annotations

import os

import torch

from backends import ops as bops
from models.qwen35.batch import batch_proj, wide_proj
from models.qwen35.parallel import context as parallel_context
from models.qwen35.parallel import row_projection
from int8_weights import QWeight, QWeightMoE
from ops.gemv import GEMV, _proj
from w4a8_weights import W4A8MoEWeight, W4A8Weight

#: 分组专家内核与张量核 GEMM 的分界（激活行数）。
#:
#: 两个内核的取舍是实测的：``moe_grouped`` 是「一列一线程、逐行串行」的 GEMV 结构，
#: 每专家 1-2 行时最优；每专家上百行时同一结构要按段重读权重，2048 token 的预填充上
#: 实测 2178 ms，而张量核 GEMM 只要 2.30 ms/次。64 是 ``models/qwen4exp/moe.py`` 在
#: 同一块卡上扫出来的门限，本族每专家宽度更窄（512 对 640），只会更早转入 GEMM。
_GEMM_ROWS = 64


def _fuse_experts(sd: dict, p: str, num_experts: int):
    """把 256 个逐专家投影融合成两整栈，并删掉 state dict 里的源条目。

    参考实现把专家存成 ``gate_up_proj [E, 2I, H]`` 与 ``down_proj [E, H, I]`` 两个三维
    参数；checkpoint 里却是 ``experts.{e}.gate_proj`` / ``up_proj`` / ``down_proj`` 三组
    二维张量。这里在构造期合成同样的三维形状，前向因此只需要一套代码。

    Args:
        sd: 按 checkpoint 参数名索引的 state dict；函数会就地删掉专家条目，因为栈已经
            拷进连续内存，留着只是把构建峰值抬高一份（每层 805 MB，40 层）。
        p: 这一层的 ``mlp`` 前缀。
        num_experts: 专家数；checkpoint 里必须真有这么多组。

    Returns:
        ``(gate_up, down)``。权重是 :class:`~w4a8_weights.W4A8Weight` 时得到
        :class:`~w4a8_weights.W4A8MoEWeight`，否则得到稠密 ``[E, 2I, H]`` /
        ``[E, H, I]`` 张量。

    Raises:
        ValueError: 专家张量数量对不上，或同一层里混了两种存储形式。混用时融合出来的
            栈会一半是真权重一半是垃圾，而形状完全合法。
    """
    # 产物里已经把专家融合成两个三维栈时直接用：量化工具会把 768 个逐专家张量合成
    # ``experts.gate_up_stack`` / ``experts.down_stack``，加载侧因此少读 30720 个张量。
    fused_gu = sd.pop(f"{p}.experts.gate_up_stack", None)
    fused_dn = sd.pop(f"{p}.experts.down_stack", None)
    if fused_gu is not None and fused_dn is not None:
        return fused_gu, fused_dn
    groups = []
    for name in ("gate_proj", "up_proj", "down_proj"):
        groups.append([sd.get(f"{p}.experts.{e}.{name}.weight", None)
                       for e in range(num_experts)])
    missing = [f"{p}.experts.<e>.{name}" for name, items in
               zip(("gate_proj", "up_proj", "down_proj"), groups)
               for item in items if item is None][:1]
    if missing:
        raise ValueError(f"missing expert tensor: {missing[0]}")

    gate, up, down = groups
    # 三种存储形式，全部融合成同一种三维栈；前向只看栈，不看它怎么来的。
    #
    # int8 与 W4A8 各有自己的分组内核：int8 是 ``moe_grouped_i8``（``[E, K, N]`` 加
    # 逐 (专家, 列) scale），W4A8 是 ``moe_grouped``（packed 加逐组 scale）。两者
    # 的分组粒度不同，所以是两个入口而不是一个开关；int8 在 decode 一档进分组内核，
    # 多行时仍退回逐专家 ``bops.linear``（见 ``_experts``）。
    int8 = [isinstance(item, QWeight) for item in gate + up + down]
    if any(int8) and all(int8):
        gate_up = QWeightMoE(
            torch.stack([torch.cat([g.wq, u.wq], dim=0) for g, u in zip(gate, up)]).contiguous(),
            torch.stack([torch.cat([g.scale, u.scale], dim=0) for g, u in zip(gate, up)]).contiguous(),
            k=gate[0].k)
        down_stack = QWeightMoE(
            torch.stack([item.wq for item in down]).contiguous(),
            torch.stack([item.scale for item in down]).contiguous(),
            k=down[0].k)
        _drop_expert_entries(sd, p, num_experts)
        return gate_up, down_stack
    if any(int8):
        raise ValueError(f"{p}: mixed int8 and floating expert weights")
    quantized = [isinstance(item, W4A8Weight) for item in gate + up + down]
    if any(quantized) and not all(quantized):
        raise ValueError(f"{p}: mixed dense and W4A8 expert weights")

    if all(quantized):
        first = gate[0]
        gate_up = W4A8MoEWeight(
            torch.stack([torch.cat([g.packed, u.packed], dim=1)
                         for g, u in zip(gate, up)]).contiguous(),
            torch.stack([torch.cat([g.scale, u.scale], dim=1)
                         for g, u in zip(gate, up)]).contiguous(),
            k=first.k, group_size=first.group_size)
        down_stack = W4A8MoEWeight(
            torch.stack([item.packed for item in down]).contiguous(),
            torch.stack([item.scale for item in down]).contiguous(),
            k=down[0].k, group_size=down[0].group_size)
    else:
        gate_up = torch.stack([torch.cat([g, u], dim=0) for g, u in zip(gate, up)])
        down_stack = torch.stack(down)

    _drop_expert_entries(sd, p, num_experts)
    return gate_up, down_stack


def _drop_expert_entries(sd: dict, p: str, num_experts: int) -> None:
    """删掉 state dict 里已被融合栈取代的逐专家条目。

    Args:
        sd: 按 checkpoint 参数名索引的 state dict，就地修改。
        p: 这一层的 ``mlp`` 前缀。
        num_experts: 专家数。
    """
    for e in range(num_experts):
        for name in ("gate_proj", "up_proj", "down_proj"):
            sd.pop(f"{p}.experts.{e}.{name}.weight", None)


def _native_ops() -> frozenset:
    """本后端原生实现的算子名集合。

    Returns:
        ``caps().native``，是 ``frozenset[str]``。
    """
    from backends import caps as _caps

    return _caps().native


def _expert_project(stack, index: int, x: torch.Tensor) -> torch.Tensor:
    """把一行或几行激活投到第 ``index`` 个专家上。

    三种栈走三条路：int8 交给 ``bops.linear``（本卡的 w8a8 内核）、稠密张量交给
    ``F.linear``、packed W4 只能走分组内核（调用方保证不会到这里）。

    Args:
        stack: :class:`~int8_weights.QWeightMoE`、稠密 ``[E, N, K]`` 或
            :class:`~w4a8_weights.W4A8MoEWeight`。
        index: 专家下标。
        x: 激活，``[rows, K]``。

    Returns:
        ``[rows, N]``。
    """
    if isinstance(stack, QWeightMoE):
        return bops.linear(stack.expert(index), x)
    return torch.nn.functional.linear(x, stack[index])


class SparseMoE:
    """一个 MoE 层：路由器、共享专家、``num_experts`` 个路由专家。

    Attributes:
        num_experts: 专家总数。
        top_k: 每 token 激活的专家数。
        inter: 每个专家的中间宽度；张量并行时是本 rank 的局部值。
        gate_up: ``[E, 2*inter, H]`` 的融合门/升栈，前 ``inter`` 列是 gate。
        down: ``[E, H, inter]``。
    """

    def __init__(self, sd: dict, p: str, cfg: dict, bench: bool) -> None:
        """建路由、共享专家与融合专家栈。

        Args:
            sd: 按 checkpoint 参数名索引的 state dict；专家条目会被本构造消费掉。
            p: 这一层的 ``mlp`` 参数前缀。
            cfg: 文本配置。张量并行时这里已经是 ``parallel.local_config`` 的局部配置，
                所以两个 ``*_intermediate_size`` 读出来就是本 rank 的宽度。
            bench: 透传给投影；``False`` 跳过自动调优。
        """
        self.num_experts = int(cfg["num_experts"])
        self.top_k = int(cfg["num_experts_per_tok"])
        self.hidden = int(cfg["hidden_size"])
        self.inter = int(cfg["moe_intermediate_size"])

        # 路由矩阵保持稠密：它决定选哪 8 个专家，4-bit 噪声会翻转选择。
        self.router = GEMV(sd[f"{p}.gate.weight"], bench)
        self.shared_gate_up = _proj(
            sd, [f"{p}.shared_expert.gate_proj", f"{p}.shared_expert.up_proj"], bench)
        self.shared_down = row_projection(
            _proj(sd, [f"{p}.shared_expert.down_proj"], bench))
        # 共享专家门是 [1, hidden] 的逐 token 标量，不走 GEMV：本卡的一行内核按列块
        # 组织，N=1 这种宽度没有保证；而它只有 4 KB 权重，一次 matmul 的代价可以忽略。
        self.shared_gate_w = sd[f"{p}.shared_expert_gate.weight"]
        #: 门标量的权重预先摊平并转 fp32：它每 token 都要用一次，而 ``reshape(-1)``
        #: 在 C=1 下形状恒定，实测每次 5.8 us、``float()`` 再来一次。一次算好存着。
        self._gate_w32 = self.shared_gate_w.reshape(-1).float()

        self.gate_up, self.down = _fuse_experts(sd, p, self.num_experts)

        self.parallel = parallel_context()
        native = _native_ops()
        self._dispatch = getattr(bops, "moe_dispatch", None) if "moe_dispatch" in native else None
        #: 共享专家门的融合入口，以及它按形状复用的输出缓冲。参考链要六次启动加两次
        #: 分配才能得到一个标量，缓存一个缓冲就把分配消掉了。
        self._gate_scale = (getattr(bops, "shared_gate_scale", None)
                            if "shared_gate_scale" in native else None)
        self._gate_buf: torch.Tensor | None = None
        self._router = getattr(bops, "moe_router", None) if "moe_router" in native else None
        #: 分组投影只有 W4A8 栈能用：内核消费的是 packed 缓冲区和分组 scale，
        #: 稠密栈在 ``backends.reference`` 里另有实现，但本卡上它不是原生路径，
        #: 走它等于在解码循环里跑一遍候选调用。
        self._grouped = (getattr(bops, "moe_grouped", None)
                         if isinstance(self.gate_up, W4A8MoEWeight) else None)
        #: int8 栈的分组入口。只在 decode 那一档接上（见 :meth:`_experts`）：分组内核
        #: 的行就是激活行，多行时同一专家的权重会被按行重读，而 decode 是每专家一行。
        self._grouped_i8 = (getattr(bops, "moe_grouped_i8", None)
                            if ("moe_grouped_i8" in native
                                and isinstance(self.gate_up, QWeightMoE)) else None)
        #: int8 栈的**归并**入口：down 的输出对单条解码来说要的是 top_k 行加权之和，
        #: 让归约核顺手做完，省掉一轮 ``[top_k, H]`` 激活往返（见 :meth:`_experts`）。
        self._grouped_i8_merge = (getattr(bops, "moe_grouped_i8_merge", None)
                                  if ("moe_grouped_i8_merge" in native
                                      and isinstance(self.gate_up, QWeightMoE))
                                  else None)
        self._grouped_gemm = (getattr(bops, "moe_grouped_gemm", None)
                              if "moe_grouped_gemm" in native else None)
        self._identity = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """解码一步：``[H]`` 或 ``[B, H]`` 进，同形出。

        Args:
            x: 归一化后的激活。

        Returns:
            与 ``x`` 同形的 MoE 输出，已含共享专家的贡献。
        """
        return self._forward(x, "decode")

    def prefill(self, x: torch.Tensor) -> torch.Tensor:
        """预填充分块：``[C, H]`` 进，``[C, H]`` 出。

        Args:
            x: 归一化后的分块激活。

        Returns:
            ``[C, H]``。
        """
        return self._forward(x, "prefill")

    def batch(self, x: torch.Tensor) -> torch.Tensor:
        """静态批解码：``[B, H]``，每行独立路由。

        Args:
            x: 归一化后的激活，每个解码序列一行。

        Returns:
            ``[B, H]``。
        """
        return self._forward(x, "batch")

    def wide(self, x: torch.Tensor) -> torch.Tensor:
        """任意行数（一次前向校验）：``[M, H]`` 进，``[M, H]`` 出。

        Args:
            x: 归一化后的激活；``M = B * c``。

        Returns:
            ``[M, H]``。用上宽内核之后各行不必与单条解码逐位相同。
        """
        return self._forward(x, "wide")

    def _forward(self, x: torch.Tensor, kind: str) -> torch.Tensor:
        """MoE 的公共主体：路由、专家分组、定序归并、加共享专家。

        Args:
            x: 归一化后的激活，``[..., H]``。
            kind: 用哪一档投影入口（``decode`` / ``batch`` / ``prefill`` / ``wide``）。

        Returns:
            与 ``x`` 同形。
        """
        # C=1 的 decode 上这几个 reshape 都是恒等的：``x`` 已是 ``[1, H]``，共享专家的
        # 输出也是 ``[1, H]``。每层白付三次元数据构造（各 5.8 us），形状对就原样交出去。
        shape = x.shape
        flat = (x if x.dim() == 2 and x.shape[1] == self.hidden
                else x.reshape(-1, self.hidden))
        weights, index = self._route(x, kind)
        # 路由与分组按二维算（专家内核与 top-k 都只认二维），共享专家则保留原始
        # 维数交给投影：单条解码时 ``x`` 是一维，``GEMV`` 只在一维输入上给出
        # ``[N]``，扁平化会把它变成 ``GEMVGroup`` 不接受的 ``[1, N]``。
        shared = self._shared(x, kind)
        if shared.shape != flat.shape:
            shared = shared.reshape(flat.shape)
        out = self._experts(flat, index, weights) + shared
        return out if out.shape == shape else out.reshape(shape)

    def _project(self, proj, x: torch.Tensor, kind: str):
        """按调用形态选投影入口。

        Args:
            proj: :class:`~ops.gemv.GEMV` 或 :class:`~ops.gemv.GEMVGroup`。
            x: 激活。
            kind: 调用形态。

        Returns:
            投影结果；``GEMVGroup`` 时是按原切分的列表。
        """
        if kind == "prefill":
            return proj.prefill(x)
        if kind == "batch":
            return batch_proj(proj, x)
        if kind == "wide":
            return wide_proj(proj, x)
        if x.dim() == 1:
            return proj(x)
        # ``GEMVGroup.__call__`` 沿第 0 轴切分，只对一维激活成立；多行时走
        # ``linear_m``（沿输出轴切分）。
        return proj.linear_m(x)

    def _route(self, x: torch.Tensor, kind: str):
        """算路由权重与专家下标。

        参考实现是 ``F.linear -> softmax(fp32) -> topk -> 除以 topk 概率之和``，最后
        窄化回 logits 的 dtype。必须用 fp32 做 softmax：bf16 下 top-k 的次序会抖动，
        而次序决定选中哪个专家。

        Args:
            x: ``[tokens, H]`` 的激活。
            kind: 调用形态，决定路由矩阵走哪一档投影。

        Returns:
            ``(weights, index)``，形状同为 ``[tokens, top_k]``；下标是 int64。
        """
        logits = self._project(self.router, x, kind)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        if self._router is not None:
            index, weights = self._router(logits, self.top_k, True)
            return weights, index
        probs = torch.softmax(logits.float(), dim=-1)
        weights, index = torch.topk(probs, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights.to(logits.dtype), index

    def _shared(self, x: torch.Tensor, kind: str) -> torch.Tensor:
        """共享专家：``sigmoid(x @ W_gate) * down(silu(gate(x)) * up(x))``。

        门是逐 token 一个标量，乘在共享专家的**输出**上而不是它的输入上——位置错了
        不会报错，只会让共享专家的贡献按另一条曲线缩放。

        Args:
            x: ``[tokens, H]``。
            kind: 调用形态。

        Returns:
            ``[tokens, H]``。
        """
        gate, up = self._project(self.shared_gate_up, x, kind)
        hidden = self._project(self.shared_down, bops.silu_mul(gate, up, up), kind)
        # 门标量用显式点积，不用 ``F.linear(x, [1, H])``：那个形状在这张卡上一次要
        # **0.62 ms**（实测，见探针的分步拆解），比共享专家两次投影加起来贵 4 倍，
        # 而它做的是 2048 维点积。换成 fp32 乘加归约后是微秒级，且加法归约比 bf16 矩阵
        # 乘更精确；乘回 ``x.dtype`` 是为了让 ``_forward`` 里与专家输出相加时 dtype 一致。
        if self._gate_scale is None:
            scalar = (x.float() * self._gate_w32).sum(-1, keepdim=True).to(x.dtype)
            return torch.sigmoid(scalar) * hidden
        buf = self._gate_buf
        if (buf is None or buf.shape != hidden.shape or buf.dtype != hidden.dtype
                or buf.device != hidden.device):
            self._gate_buf = buf = torch.empty_like(hidden)
        return self._gate_scale(x, self._gate_w32, hidden, buf)

    def _experts(self, x: torch.Tensor, index: torch.Tensor,
                 weights: torch.Tensor) -> torch.Tensor:
        """按专家分组算路由部分，再对每个 token 的 top_k 项定序求和。

        Args:
            x: ``[tokens, H]``。
            index: ``[tokens, top_k]`` 的全局专家下标。
            weights: ``[tokens, top_k]`` 的路由权重。

        Returns:
            ``[tokens, H]``；不发生路由的 token 得到全零行。
        """
        tokens, top_k = index.shape
        flat_index = index.reshape(-1)
        # 多行（prefill）也走分组内核。实测依据：M=512 时逐专家循环要 **173 ms/层**
        # （256 个专家 × nonzero/index_select/两次 linear，一层上千次算子），占单层的 98%。
        # 多行也走分组内核（实测 M=512 时 6.5x、M=128 时 9.2x）。曾经的「多行数值对不上」
        # 查下去是 `pos = token * top_k + slot_of_pos` 把展平位置又叠了一次 token 号，与
        # 内核无关；修完后对 fp32 基准是 1.9e-2（= A8 激活量化的固有误差，decode 那条路
        # 一直就是这个口径）。``COFFIN_MOE_I8_GROUPED=0`` 可退回逐专家路径做对照。
        old_multi_row = (os.environ.get("COFFIN_MOE_I8_GROUPED") == "0"
                         and tokens > 1)
        grouped_i8 = None if old_multi_row else self._grouped_i8
        if self._dispatch is not None and (self._grouped is not None
                                          or grouped_i8 is not None):
            offsets, _slot_pos, token_of_pos, slot_of_pos = self._dispatch(
                flat_index, self._identity_table(flat_index.device), flat_index.numel(),
                self.num_experts, self.num_experts, top_k)
            # 收拢成「按专家排好序」的行，然后两次分组投影：gate/up 一次出中间量，
            # 过 SwiGLU 之后再投一次 down。两次都必须用分组内核——按专家分组的意义
            # 就是让每个专家 2 MB 的权重在一个解码步里只被读一遍。
            packed = x.index_select(0, token_of_pos.long())
            gate, up = self._group(self.gate_up, packed, offsets).chunk(2, dim=-1)
            activated = bops.silu_mul(gate, up, up)
            # ``slot_of_pos`` 是**展平后的**槽位号（``token * top_k + slot``），不是槽位内的
            # 0..top_k-1 —— 实测 tokens=8、slots=64 时它的取值范围是 0..63。所以逐行权重
            # 和回写位置用的都是它本身；再叠一次 ``token * top_k`` 会让下标飞出缓冲区
            # （decode 时 token 恒为 0，两种写法重合，这正是只有多行才炸的原因）。
            pos = slot_of_pos.long()
            flat_weights = weights.reshape(-1).index_select(0, pos)
            if self._grouped_i8_merge is not None and tokens == 1:
                # down 的输出对单条解码来说要的就是 top_k 行加权之和，让归约核顺手做完：
                # 省掉 [top_k, H] 的激活、一次 mul、一次 index_add_ 和一次 sum（实测
                # 0.184 ms，比 down 的投影本身 0.059 ms 还贵）。
                return self._reduce(
                    self._grouped_i8_merge(self.down, activated, offsets, flat_weights),
                    x.dtype)
            hidden = self._group(self.down, activated, offsets)
            # 分派内核把有效行紧密排在前缀里、尾部清 0，所以尾部那几个位置会带着
            # 全零贡献落到槽位 0 上——加法里加 0 不改值，写法比切片简单且不需要同步。
            buf = torch.zeros((tokens * top_k, self.hidden),
                              dtype=x.dtype, device=x.device)
            buf.index_add_(0, pos, hidden * flat_weights[:, None])
            return self._reduce(buf.view(tokens, top_k, self.hidden).sum(dim=1), x.dtype)
        return self._reduce(
            self._experts_dense(x, index, weights, tokens, top_k), x.dtype)

    def _reduce(self, value: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """张量并行下把各 rank 的部分和加起来；单卡直接返回。

        专家的 ``down`` 按 K（中间宽度）分片，每个 rank 只算出一部分和，所以这一步
        不是可选的优化：漏掉它得到的是「每个 rank 各算四分之一」的安静错误。

        Args:
            value: 本 rank 的浮点结果。
            dtype: 返回的 dtype，与激活一致。

        Returns:
            ``value``（单卡）或跨 rank 求和后窄化回 ``dtype`` 的结果。
        """
        # 只有张量并行才需要跨 rank 求和：那时 down 的 K 轴被切开，每个 rank 只算部分
        # 和。流水并行里专家是完整的，再加一次全归约等于把结果乘上世界大小。
        if self.parallel is None or self.parallel.mode != "tp":
            return value
        return self.parallel.reduce(value).to(dtype)

    def _group(self, stack, x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """用分组内核按专家投影一整栈权重。

        Args:
            stack: int8 专家栈（:class:`~int8_weights.QWeightMoE`，``[E, K, N]``）或
                :class:`~w4a8_weights.W4A8MoEWeight`（``[E, N, K]``）。
            x: ``[rows, K]``，已按专家号排好序。
            offsets: ``[E+1]`` 前缀和。

        Returns:
            ``[rows, N]``。
        """
        if isinstance(stack, QWeightMoE):
            # int8 栈只有分组内核这一条路：``moe_grouped`` 与 ``moe_grouped_gemm``
            # 消费的都是 packed W4，把 int8 张量喂进去只会在形状校验里报错。
            return self._grouped_i8(stack, x, offsets)
        if x.shape[0] >= _GEMM_ROWS and self._grouped_gemm is not None:
            # 张量核那条路对 K 与分组的对齐有要求（本族 K=2048/512，分组 128，够用）。
            # 它不满足时会明确抛 ValueError 而不是算错，所以这里退回 GEMV 结构——
            # 慢，但答案对。
            try:
                return self._grouped_gemm(stack, x, offsets)
            except ValueError:
                pass
        return self._grouped(stack, x, offsets)

    def _experts_dense(self, x: torch.Tensor, index: torch.Tensor, weights: torch.Tensor,
                       tokens: int, top_k: int) -> torch.Tensor:
        """没有分组内核时的逐专家循环，语义与参考实现逐行相同。

        只该出现在结构测试和稠密权重的调试路径上：它按专家号 ``index_select`` 再
        ``F.linear``，每个被命中的专家权重读一遍，decode 期每专家 1-2 行时权重读取
        会放大到 token 数的倍数。

        Args:
            x: ``[tokens, H]``。
            index: ``[tokens, top_k]``。
            weights: ``[tokens, top_k]``。
            tokens: 行数。
            top_k: 每 token 的专家数。

        Returns:
            ``[tokens, H]``。

        Raises:
            RuntimeError: 专家栈是量化形式却没有分组内核。这条路径不会自己编一个
                反量化循环——那会把 256 个专家每步实体化一遍。
        """
        if isinstance(self.gate_up, W4A8MoEWeight) and self._grouped is None:
            raise RuntimeError(
                "packed-W4 expert stacks need the grouped-MoE kernel, which this "
                "backend does not provide natively")
        flat = index.reshape(-1)
        if tokens == 1:
            # 单条解码：top_k 个槽位都落在同一行上，所以 "哪些行去哪个专家" 这件事不存在，
            # 直接按槽位顺序算再累加。省掉的不只是几次搬运：``torch.unique(...).tolist()``
            # 与每次 ``.nonzero()`` 都要把结果从设备读回主机，一层里 8 个专家就是 9 次同步；
            # 实测 MoE 的 2.34 ms 里大头花在这些空转上（权重只有 25 MB，按带宽该是 55 us）。
            row = x[0:1]
            picks = flat.tolist()
            scale = weights.reshape(-1)
            acc = None
            for slot, expert in enumerate(picks):
                gate, up = _expert_project(self.gate_up, expert, row).chunk(2, dim=-1)
                hidden = _expert_project(self.down, expert,
                                         torch.nn.functional.silu(gate) * up)
                term = hidden * scale[slot]
                acc = term if acc is None else acc + term
            return acc
        #: 槽位号 = token * top_k + slot；同一专家命中的槽位互不相同，所以是定序写入。
        out = torch.zeros((tokens * top_k, self.hidden), dtype=x.dtype, device=x.device)
        for expert in torch.unique(flat).tolist():
            pos = (flat == expert).nonzero().reshape(-1)
            current = x.index_select(0, pos // top_k)
            gate, up = _expert_project(self.gate_up, expert, current).chunk(2, dim=-1)
            hidden = _expert_project(self.down, expert,
                                     torch.nn.functional.silu(gate) * up)
            out.index_copy_(0, pos, hidden)
        out = out * weights.reshape(-1)[:, None]
        return out.view(tokens, top_k, self.hidden).sum(dim=1)

    def _identity_table(self, device: torch.device) -> torch.Tensor:
        """全局专家号 → 本地号的恒等表（本族不做专家分片，所以是 ``0..E-1``）。

        留着这张表而不是传 ``None``：分派内核要求这张表存在，而专家并行若日后接上，
        改的只是这里一行。

        Args:
            device: 表所在设备。

        Returns:
            ``[E]`` 的 int32 张量。
        """
        table = self._identity
        if table is None or table.device != device:
            table = torch.arange(self.num_experts, dtype=torch.int32, device=device)
            self._identity = table
        return table


__all__ = ["SparseMoE"]
