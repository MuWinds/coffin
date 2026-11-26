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
    * 专家权重按融合栈存放，不是 256 个独立投影：int8 是 ``[E, N, K]``（nk 布局，
      K 连续），稠密是 ``[E, 2I, H]``。分组内核
      按固定偏移寻址整块连续内存，摊成 256 份会让「第 e 个专家的起点」算错，而形状
      检查照样过。
    * 路由矩阵（``mlp.gate``）与共享专家门（``mlp.shared_expert_gate``）保持稠密：
      量化噪声足够翻转 top-8 的选择，而选错专家是整层输出错，不是精度变差。
      ``tools/quant_disk.py`` 的 ``DENSE_SUFFIXES`` 是这条约束的执行点。
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
from models.qwen35 import stages
from int8_weights import QWeight, QWeightMoE
from ops.gemv import GEMV, _proj

#: 分组专家内核与张量核 GEMM 的分界（激活行数）。
#:
#: 两个内核的取舍是实测的：``moe_grouped`` 是「一列一线程、逐行串行」的 GEMV 结构，
#: 每专家 1-2 行时最优；每专家上百行时同一结构要按段重读权重，2048 token 的预填充上
#: 实测 2178 ms，而张量核 GEMM 只要 2.30 ms/次。64 是 ``models/qwen4exp/moe.py`` 在
#: 同一块卡上扫出来的门限，本族每专家宽度更窄（512 对 640），只会更早转入 GEMM。
_GEMM_ROWS = 64

#: 「按专家跨行」预填充核的启用门限（激活行数）。低于它时每专家只有一两行，跨行复用
#: 换不到东西，逐行核（块与行一一对应）反而更省地址运算；高于它时权重按行重读的放大
#: 已经超过一切固定成本。
_EXPERT_ROW_MIN = 64

#: 分组张量核（``moe_grouped_i8_tcu``）的启用门限（激活行数）。``stage_at`` 已支持任意
#: 偏移与非 4 倍数段长（段尾补零），因此不再需要按 4 对齐校验或 D2H 同步；高于此门限时
#: 张量核的 19.2 TMAC/s 远胜 dp4a 的 0.96 TMAC/s。
_TCU_ROW_MIN = int(os.environ.get("COFFIN_MOE_TCU_MIN_ROWS", "256") or "256")

#: 「按专家跨行」核支持 K 为 512 倍数（16B 宽步）与 128 倍数（4B 窄步，覆盖 TP4 down K=128）。
_EXPERT_K_ALIGN = 128

#: 张量核版的专家核按 32 k 分片（前端校验 64 对齐），TP4 down K=128 与 gate_up K=2048 均满足。
_EXPERT_TCU_ALIGN = 64


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
        ``(gate_up, down)``。专家是 int8 ``QWeight`` 时得到
        :class:`~int8_weights.QWeightMoE`，否则得到稠密 ``[E, 2I, H]`` /
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
    # 两种存储形式，全部融合成同一种三维栈；前向只看栈，不看它怎么来的。
    #
    # 量化栈只有 int8 一种：``moe_grouped_i8``（``[E, N, K]`` 加逐 (专家, 列) scale）。
    # 它在 decode 一档进分组内核，多行时仍退回逐专家 ``bops.linear``（见 ``_experts``）。
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
        stack: :class:`~int8_weights.QWeightMoE` 或稠密 ``[E, 2I, H]`` / ``[E, H, I]``。
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

        self.parallel = parallel_context()
        #: 是不是张量并行。TP 下共享专家的门标量在各卡完全一致，所以 ``shared_down`` 保持
        #: 本地部分和，与路由专家的部分和相加后再做**一次**全归约（每层 3 次 → 2 次）。
        self._is_tp = self.parallel is not None and self.parallel.mode == "tp"
        # 路由矩阵保持稠密：它决定选哪 8 个专家，4-bit 噪声会翻转选择。
        self.router = GEMV(sd[f"{p}.gate.weight"], bench)
        self.shared_gate_up = _proj(
            sd, [f"{p}.shared_expert.gate_proj", f"{p}.shared_expert.up_proj"], bench)
        shared_down = _proj(sd, [f"{p}.shared_expert.down_proj"], bench)
        self.shared_down = shared_down if self._is_tp else row_projection(shared_down)
        # 共享专家门是 [1, hidden] 的逐 token 标量，不走 GEMV：本卡的一行内核按列块
        # 组织，N=1 这种宽度没有保证；而它只有 4 KB 权重，一次 matmul 的代价可以忽略。
        self.shared_gate_w = sd[f"{p}.shared_expert_gate.weight"]
        #: 门标量的权重预先摊平并转 fp32：它每 token 都要用一次，而 ``reshape(-1)``
        #: 在 C=1 下形状恒定，实测每次 5.8 us、``float()`` 再来一次。一次算好存着。
        self._gate_w32 = self.shared_gate_w.reshape(-1).float()

        self.gate_up, self.down = _fuse_experts(sd, p, self.num_experts)

        #: 本 rank 的专家数。单卡 / ``pp`` / ``tp`` 下等于全局值；``ep`` 下是
        #: 专家栈的首轴（加载器读盘时就裁好了），不能拿 ``cfg`` 的全局值去覆盖它。
        self.local_experts = self.num_experts
        #: 是不是专家并行。整块驱动与 :meth:`_reduce` 都按它分叉：EP 下驱动只回本 rank
        #: 的专家部分，共享专家必须等跨 rank 求和之后再加一次。
        self._is_ep = self.parallel is not None and self.parallel.mode == "ep"
        if self._is_ep:
            begin, end = self.parallel.expert_range(self.num_experts)
            stacked = (self.gate_up.nk_stack()
                       if isinstance(self.gate_up, QWeightMoE) else self.gate_up)
            self.local_experts = int(stacked.shape[0])
            if self.local_experts != end - begin:
                raise ValueError(
                    f"{p}: this rank holds {self.local_experts} experts but its "
                    f"expert_range is [{begin}, {end}); the loader and the model "
                    f"disagree about the topology, which would silently route "
                    f"tokens to experts that are not loaded")
        native = _native_ops()
        self._dispatch = getattr(bops, "moe_dispatch", None) if "moe_dispatch" in native else None
        #: 共享专家门的融合入口，以及它按形状复用的输出缓冲。参考链要六次启动加两次
        #: 分配才能得到一个标量，缓存一个缓冲就把分配消掉了。
        self._gate_scale = (getattr(bops, "shared_gate_scale", None)
                            if "shared_gate_scale" in native else None)
        self._gate_buf: torch.Tensor | None = None
        #: 多行解码收尾的 ``[tokens, H]`` 累加缓冲；按第一次用到的行数定尺，之后复用。
        self._tail_buf: torch.Tensor | None = None
        self._router = getattr(bops, "moe_router", None) if "moe_router" in native else None
        #: 分组投影的入口只对 int8 栈成立：本仓库的专家权重只有 int8 与稠密两种存储，
        #: 4-bit 那条路连同它的分组内核一起删掉了。稠密栈在
        #: ``backends.reference`` 里另有实现，但本卡上它不是原生路径，走它等于在解码
        #: 循环里跑一遍候选调用。
        self._grouped = None
        #: int8 栈的分组入口。只在 decode 那一档接上（见 :meth:`_experts`）：分组内核
        #: 的行就是激活行，多行时同一专家的权重会被按行重读，而 decode 是每专家一行。
        self._grouped_i8 = (getattr(bops, "moe_grouped_i8", None)
                            if ("moe_grouped_i8" in native
                                and isinstance(self.gate_up, QWeightMoE)) else None)
        #: 预填充那一档的入口：按专家切块、把他的行一趟扫完，权重每层每专家只读一遍。
        #: 专家栈只有 nk 一种布局，所以这里不再按 ``layout`` 分叉。
        self._grouped_i8_expert = (getattr(bops, "moe_grouped_i8_expert", None)
                                   if ("moe_grouped_i8_expert" in native
                                       and isinstance(self.gate_up, QWeightMoE)
                                       and self.hidden % _EXPERT_K_ALIGN == 0
                                       and self.inter % _EXPERT_K_ALIGN == 0) else None)
        #: 同一个「按专家跨行」形态、但把专家 GEMM 交给张量核。同一张卡同一形状上实测
        #: 19.2 TMAC/s 对 dp4a 的 0.96，预填充的 4000 tps 只能靠这条；入口缺席时上面的
        #: dp4a 版本仍然是正确的结果，只是慢。
        self._grouped_i8_tcu = (getattr(bops, "moe_grouped_i8_tcu", None)
                                if ("moe_grouped_i8_tcu" in native
                                    and isinstance(self.gate_up, QWeightMoE)
                                    and self.hidden % _EXPERT_TCU_ALIGN == 0
                                    and self.inter % _EXPERT_TCU_ALIGN == 0) else None)
        #: int8 栈的**归并**入口：down 的输出对单条解码来说要的是 top_k 行加权之和，
        #: 让归约核顺手做完，省掉一轮 ``[top_k, H]`` 激活往返（见 :meth:`_experts`）。
        self._grouped_i8_merge = (getattr(bops, "moe_grouped_i8_merge", None)
                                  if ("moe_grouped_i8_merge" in native
                                      and isinstance(self.gate_up, QWeightMoE))
                                  else None)
        #: int8 栈的**逐 token 归并**入口：单流与多流共用一个入口，权重按槽位号直取、
        #: 有效行掩码在核里判、累加直接写 ``[tokens, H]``（见 :meth:`_experts`）。
        self._merge_tokens = (getattr(bops, "moe_grouped_i8_merge_tokens", None)
                              if ("moe_grouped_i8_merge_tokens" in native
                                  and isinstance(self.gate_up, QWeightMoE))
                              else None)
        #: 取行 + 量化的合成入口与「乘积 + 量化」的合成入口。只在 int8 栈上成立：
        #: 它们省的是 int8 投影前的两次量化发射，别的栈不消费 int8 激活。
        self._gather_quant = (getattr(bops, "gather_quant_i8", None)
                              if ("gather_quant_i8" in native
                                  and isinstance(self.gate_up, QWeightMoE))
                              else None)
        self._silu_quant = (getattr(bops, "silu_mul_quant", None)
                            if ("silu_mul_quant" in native
                                and isinstance(self.gate_up, QWeightMoE))
                            else None)
        self._gather_sum = (getattr(bops, "moe_gather_sum", None)
                            if "moe_gather_sum" in native else None)
        #: 多流那一路的复用缓冲：``(rows, width, dtype, device)`` → 张量。三块分别是
        #: gate/up 的输出、激活乘积、以及归并核的 ``[tokens, H]`` 目的地。按层复用，
        #: 因为它们只在层内活，而每层新建等于每步白付 120 次分配。
        self._act_bufs: dict = {}
        #: 全局专家号 → 本地号；``ep`` 下不归本 rank 的专家是 -1。见 :meth:`_expert_table`。
        self._table: torch.Tensor | None = None
        #: decode 单行的整块驱动。它只省 pybind 编组与 Python 包装（约 4 us/次 x 17 次），
        #: 分配与发射照付，所以上限是个位数百分比；``COFFIN_LAYER_DRIVER=1`` 才启用，
        #: 便于同二进制 A/B。
        self._block_driver = (getattr(bops, "moe_decode_block", None)
                              if "moe_decode_block" in native else None)
        # 专家并行下这条也开着：`moe_block.cu` 现在从映射表的宽度取**全局**专家数
        # （分派内核的边界检查要用它），并把「不属于本 rank 的槽位」的逐行权重在设备上
        # 清零。这两条不加就是「首 token 正确、其后逐 token 变胡话」。
        self._block_w = None
        self._block_b = None
        #: 句柄入口（``moe_decode_prepare`` / ``moe_decode_run``）。装载时把 26 个常量
        #: 编组一次，解码步里每次只传 ``(handle, x)``——省的是 pybind 的逐参转换，
        #: 与算术无关，所以数值与 30 参那条路逐位相同。缺任一个就退回老形态。
        self._block_prepare = (getattr(bops, "moe_decode_prepare", None)
                               if "moe_decode_prepare" in native else None)
        self._block_run = (getattr(bops, "moe_decode_run", None)
                           if "moe_decode_run" in native else None)
        if self._block_prepare is None or self._block_run is None:
            self._block_run = None
        self._block_handle = 0
        #: 「后置归一化 + 整块 + 跨 rank 求和 + 加共享专家」的融合入口。一步 25 ms 里
        #: `add_residual`（79 次）、`moe_decode_run`（40 次）、`all_reduce_sum`（40 次）
        #: 合计约 13.5 ms，而它们的内核时间只有 8 / 73 / 20 us——大头是每层的
        #: **四次 Python→C 往返与适配器准备**。并成一次就是一层省三次往返。
        self._layer_prepare = (getattr(bops, "moe_layer_prepare", None)
                               if "moe_layer_prepare" in native else None)
        self._layer_run = (getattr(bops, "moe_layer_run", None)
                           if "moe_layer_run" in native else None)
        if self._layer_prepare is None or self._layer_run is None:
            self._layer_run = None
        self._layer_handle = 0
        self._ln_buf: torch.Tensor | None = None

    def layer_step(self, x_res: torch.Tensor, h: torch.Tensor,
                   gain: torch.Tensor) -> torch.Tensor | None:
        """MoE 层的整段：``x += h``、归一化、专家、跨 rank 求和、加共享专家。

        替代原来四段 Python 编排（``post_ln.add_residual`` → ``mlp`` → ``_reduce`` →
        加共享专家）。逐位口径不变：残差先累加再归一化，跨 rank 和**先窄化再**加共享
        专家（两次舍入，不能合并成一次 fp32 加法）。

        Args:
            x_res: 残差流，``[H]`` 或 ``[1, H]``，原地累加。
            h: 混合器输出，形状同 ``x_res``。
            gain: ``post_ln`` 的增益，``[H]``，已带 ``1 + w`` 偏移。

        Returns:
            与 ``x_res`` 同形的下一步残差增量；前提不满足时返回 ``None``。
        """
        if self._layer_run is None:
            return None
        p2p = getattr(self.parallel, "_p2p", None) if self.parallel is not None else None
        if p2p is None:
            self._layer_run = None
            return None
        if self._layer_handle == 0:
            if self._block_w is None:
                self._build_block(h)
                if self._block_w is None:
                    self._layer_run = None
                    return None
            if self._ln_buf is None:
                self._ln_buf = torch.empty((1, self.hidden), dtype=torch.bfloat16,
                                           device=x_res.device)
            rows = 1
            blocks, threads = p2p._grid(rows * self.hidden)
            self._layer_handle = self._layer_prepare(
                list(self._block_w) + list(self._block_b), self.top_k, gain,
                self._ln_buf, p2p._out[:rows], self._block_b[-1],
                list(p2p._slot_ptr), list(p2p._flag_ptr), int(p2p.rank),
                int(blocks), int(threads), 1e-6, 0,
                1)
            if not getattr(self, "_lyr_logged", False):
                print(f"[moe] layer fuse on, handle={self._layer_handle} "
                      f"grid={blocks}x{threads}", flush=True)
                self._lyr_logged = True
        # 代际号必须是**整个进程唯一**的一条单调序列：每个层各持一份会撞号，对端看到
        # 「这一代已到」立刻放行、读走别人写了一半的槽位。用 Python 那个全局计数，
        # 先加后用（与 ``ep.py`` 同一口径）。
        round_ = p2p._round + 1
        p2p._round = round_
        # 默认走**只融合前段**：`moe_layer_run(..., with_p2p=0)` 拿回本 rank 部分和，跨 rank
        # 求和与加共享专家留在 Python。前段融合是挣到的（0.6–1.0 ms/步），而**全融合**目前
        # 还不可信：同一份代码（`027_fp` 与 `028_fq`）两次跑出不同质量的文本，其中一次是
        # 标点乱码——贪心解码本该确定，两次不同就说明还有竞态没关掉（见台账 §11.8）。
        # `COFFIN_FUSE_FULL=1` 可切回全融合做结构实验。
        #
        # 已查清的三件事都记在台账 §11：① p2p 的会合与求和**是对的**（已知答案探针在
        # 四张卡上都读到 10）；② 之前那套「逐段自检」的参考链是坏的（`_decode_block` 没
        # 重算，返回的就是融合输出本身），拿它当基准得到的结论全部作废；③ 驱动把自己那块
        # 常驻缓冲当结果交出去会被下一层的 `carry` 就地覆写，必须交副本。
        if os.environ.get("COFFIN_FUSE_FULL") != "1":
            part = self._layer_run(self._layer_handle, x_res, h, round_, 0)
            merged = self._reduce(part.reshape(1, -1), torch.bfloat16) + self._block_b[-2]
            return merged.reshape(-1) if x_res.dim() == 1 else merged
        if os.environ.get("COFFIN_FUSE_CHECK") == "1" and not getattr(
                type(self), "_fuse_checked", False):
            type(self)._fuse_checked = True
            return self._checked_step(p2p, x_res, h, gain, round_)
        return self._layer_run(self._layer_handle, x_res, h, round_, 1)

    def _checked_step(self, p2p, x_res: torch.Tensor, h: torch.Tensor,
                      gain: torch.Tensor, round_: int) -> torch.Tensor:
        """同一次运行里把融合结果和参考链逐段比一次，定位差在哪一段。

        只在 ``COFFIN_FUSE_CHECK=1`` 下、每进程跑一次（第一层第一步）。为了不把残差
        流写坏，先备份 ``x_res``、跑完融合再把备份写回去，然后用**参考链**算一遍。
        最后返回参考结果——这一跑的生成质量因此不受融合路径影响，自检读数才可信。

        Args:
            p2p: EP 的 p2p 归约器（提供轮次与缓冲）。
            x_res: 残差流，``[H]``。
            h: 混合器输出，``[H]``。
            gain: ``post_ln`` 的增益。
            round_: 本次调用已经占用的代际号。

        Returns:
            参考链的结果，形状同 ``x_res``。
        """
        backup = x_res.clone()
        # 三段各自 clone：融合输出落在 ``_block_b[-1]``、累加位落在 ``p2p._out``，参考链
        # 会把这两块缓冲都覆写一遍，不先拷出来，后面读到的就是参考自己的结果。
        fused = self._layer_run(self._layer_handle, x_res, h, round_).clone()
        fused_normed = self._ln_buf.clone()
        fused_sum = p2p._out[:1].clone()
        x_res.copy_(backup)
        # 参考链的归一化：与 ``ops/norm.py`` 的 ``add_residual`` 同一组入参（行视图 +
        # 该行间距），只是这里直接调算子表，不经过 RMSNorm 包装。
        rows_ref = x_res.reshape(1, -1)
        ref_normed_2d = torch.empty_like(rows_ref)
        bops.add_rms_norm(rows_ref, h.reshape(1, -1), gain, ref_normed_2d,
                          self.hidden, 1e-6)
        driven = self._decode_block(ref_normed_2d)
        out = self._reduce(driven, driven.dtype)
        ref_sum = p2p._out[:1].clone()
        if self._is_ep:
            out = out + self._block_b[-2]
        ref = out.reshape(-1) if out.dim() == 2 else out
        def _stat(a, b):
            d = (a.float() - b.float()).abs()
            return f"{float(d.max()):.3e}/{float(d.mean()):.3e}"
        # 全部先算成标量再拼一行：f-string 表达式里换行在 3.10 上是语法错误，而本地
        # 解释器是 3.12（PEP 701 允许），本地语法检查拦不住这类错，节点上一跑就炸。
        d_normed = _stat(fused_normed, ref_normed_2d)
        d_sum = _stat(fused_sum, ref_sum)
        d_out = _stat(fused.reshape(-1), ref)
        d_sum_part = _stat(fused_sum, driven)
        d_out_part = _stat(fused.reshape(-1),
                           (driven.to(torch.bfloat16) + self._block_b[-2]).reshape(-1))
        m_sum = float(ref_sum.float().abs().mean())
        m_out = float(ref.float().abs().mean())
        m_fused = float(fused.float().abs().mean())
        m_part = float(driven.float().abs().mean())
        m_shared = float(self._block_b[-2].float().abs().mean())
        print(f"[fusechk] normed={d_normed} sum32={d_sum} out={d_out} "
              f"sum32_vs_part={d_sum_part} out_vs_part={d_out_part} "
              f"|ref_sum|={m_sum:.3e} |ref_out|={m_out:.3e} |fused_out|={m_fused:.3e} "
              f"|ref_part|={m_part:.3e} |ref_shared|={m_shared:.3e}", flush=True)
        # 头 8 个元素直接摆出来。标量差的判读已经错过一次（四个 rank 的 ``red32`` 逐位
        # 相同本来是「all-reduce 正确」的证据，却被读成「没有求和」），所以这里给第一手
        # 数字：``fs``/``rs`` 是融合与参考的 fp32 累加位，``drv`` 是参考链的本 rank 部分，
        # ``fu`` 是融合输出。若 ``drv`` 与 ``fu`` 头几位相同而 ``|drv|`` 又等于 ``|out|``，
        # 就说明参考链的 ``_decode_block`` 根本没重算。
        def _e8(t):
            vals = t.reshape(-1)[:8].float().tolist()
            return " ".join(format(v, "+.3e") for v in vals)
        e_fs = _e8(fused_sum)
        e_rs = _e8(ref_sum)
        e_drv = _e8(driven)
        e_fu = _e8(fused)
        print(f"[fusechk8] fs={e_fs}", flush=True)
        print(f"[fusechk8] rs={e_rs}", flush=True)
        print(f"[fusechk8] drv={e_drv}", flush=True)
        print(f"[fusechk8] fu={e_fu}", flush=True)
        return ref

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

    def _build_block(self, flat: torch.Tensor) -> None:
        """准备整块驱动的权重视图与复用缓冲；前提不满足时清空入口。

        路由权重在本模型里是**稠密**的（``mlp.gate.weight`` 在 ``DENSE_SUFFIXES`` 里，
        量化噪声会翻转专家选择），所以 ``router_scale`` 传一个零长度张量表示「走 bf16」。

        Args:
            flat: ``[1, H]`` 的归一化激活，用来定 dtype 与设备。
        """
        hidden = self.hidden
        inter = self.inter
        top_k = self.top_k
        device = flat.device
        bf16 = flat.dtype
        group = getattr(self.shared_gate_up, "g", self.shared_gate_up)
        sg = getattr(group, "q", None)
        sd_proj = getattr(self.shared_down, "g", self.shared_down)
        sd = getattr(sd_proj, "q", None)
        if not isinstance(self.gate_up, QWeightMoE) or sg is None or sd is None:
            self._block_driver = None
            return
        #: 专家栈给整块驱动用的就是 checkpoint 原生那一份：``nk_stack()`` 零拷贝、不转置，
        #: 所以整块驱动不会为任何布局留第二份显存。路由与共享专家的权重也按 nk（``[N, K]``）
        #: 直接传下去，不做 ``.t().contiguous()`` ——那份 kn 副本已经整条删掉。
        gu_stack = self.gate_up.nk_stack()
        dn_stack = self.down.nk_stack()
        router_w = self.router.w
        self._block_w = (
            router_w,
            torch.empty(0, dtype=torch.float32, device=device),
            gu_stack, self.gate_up.scale,
            dn_stack, self.down.scale,
            sg.wq, sg.scale,
            sd.wq, sd.scale,
            self._gate_w32.to(device), self._expert_table(device),
        )
        self._block_b = (
            torch.empty((1, top_k), dtype=torch.int64, device=device),
            torch.empty((1, top_k), dtype=bf16, device=device),
            torch.empty(self.local_experts + 1, dtype=torch.int32, device=device),
            torch.empty(top_k, dtype=torch.int32, device=device),
            torch.empty(top_k, dtype=torch.int32, device=device),
            torch.empty(top_k, dtype=torch.int32, device=device),
            torch.empty(self.num_experts, dtype=torch.int32, device=device),
            torch.empty((top_k, hidden), dtype=torch.int8, device=device),
            torch.empty(top_k, dtype=torch.float32, device=device),
            torch.empty((top_k, 2 * inter), dtype=bf16, device=device),
            torch.empty((top_k, inter), dtype=torch.int8, device=device),
            torch.empty(top_k, dtype=torch.float32, device=device),
            torch.empty((1, hidden), dtype=bf16, device=device),
            torch.empty((1, hidden), dtype=bf16, device=device),
        )
        if self._block_run is not None:
            # 26 个常量按 ``moe_decode_block`` 的张量顺序：``_block_w`` 的 12 个权重在前、
            # ``_block_b`` 的 14 个复用缓冲（含输出 ``out``）在后。句柄持有这些张量本身，
            # 所以它们在本层活着期间不会被回收。
            self._block_handle = self._block_prepare(
                list(self._block_w) + list(self._block_b), top_k, 0,
                0 if self._is_ep else 1, 1)
            if not getattr(self, "_blk_logged", False):
                # 一行、每进程一次：句柄入口是**静默降级**的（拿不到就退回 30 参），
                # 所以量下来没变化时必须能一眼看出走的是哪条路。
                print(f"[moe] handle path on, handle={self._block_handle} "
                      f"to_k={top_k} ep={0 if self._is_ep else 1} nk=1",
                      flush=True)
                self._blk_logged = True

    def _decode_block(self, flat: torch.Tensor):
        """用 C++ 驱动算 decode 一层的 MoE 块。

        走句柄那一支：常量在装载时编组一次（``moe_decode_prepare``），每次只传激活与
        句柄号。实测每个张量入参的 pybind 转换约 2–3 us，29 个入参就是 90 us 上下，而
        一层每步调一次、40 层——那是每步 3 ms 级的主机成本，且**与算术无关**。
        扩展没有句柄入口时退回 30 参的历史形态，两条路的数值完全相同。

        Args:
            flat: ``[1, H]`` 的归一化激活。

        Returns:
            ``[1, H]``；开关关闭或前提不满足时返回 ``None``，由调用方走原编排。
        """
        driver = self._block_driver
        if driver is None or os.environ.get("COFFIN_LAYER_DRIVER", "1") != "1":
            return None
        if flat.dim() != 2 or flat.shape[0] != 1 or flat.shape[1] != self.hidden:
            return None
        if self._block_w is None:
            self._build_block(flat)
            if self._block_w is None:
                return None
        if self._block_run is not None:
            return self._block_run(self._block_handle, flat)
        return driver(flat, *self._block_w, *self._block_b, self.top_k, 0,
                      0 if self._is_ep else 1)

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
        if kind == "decode":
            driven = self._decode_block(flat)
            if driven is not None:
                out = self._reduce(driven, driven.dtype)
                if self._is_ep:
                    # 共享专家用**驱动已经算好的那一份**（``_block_b[-2]`` 就是
                    # ``shared_hidden``），不要再调一次 :meth:`_shared`：每层两次投影
                    # 在 decode 上是实打实的浪费，而驱动无论如何都要算它（输出布局
                    # 决定了它必须写进那个缓冲区）。
                    shared = self._block_b[-2]
                    # 不复用 ``out`` 做就地相加：``out`` 可能是驱动自己那块输出缓冲的
                    # 视图，``add_`` 会把它改写掉，而那块缓冲下一层还要当输入读——省
                    # 下的一次分配只值 0.16 ms/步，不值得拿「几百 token 之后才开始走
                    # 样」的风险去换。
                    out = out + (shared if shared.shape == out.shape
                                 else shared.reshape(out.shape))
                return out if out.shape == shape else out.reshape(shape)
        with stages.span("moe.route"):
            weights, index = self._route(x, kind)
        # 路由与分组按二维算（专家内核与 top-k 都只认二维），共享专家则保留原始
        # 维数交给投影：单条解码时 ``x`` 是一维，``GEMV`` 只在一维输入上给出
        # ``[N]``，扁平化会把它变成 ``GEMVGroup`` 不接受的 ``[1, N]``。
        with stages.span("moe.shared"):
            shared = self._shared(x, kind)
        if shared.shape != flat.shape:
            shared = shared.reshape(flat.shape)
        with stages.span("moe.experts"):
            if self._is_tp:
                # TP 下 shared_down 与专家 down 同为局部部分和、且门标量跨卡一致：
                # 在 P2P 核内直接做 ``bf16(experts + shared)`` 并只做一次全归约，省掉
                # 每层一次通信与一次独立加法发射。
                local_exp = self._experts(flat, index, weights, do_reduce=False)
                out = self._reduce(local_exp, x.dtype, add=shared)
            else:
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
                 weights: torch.Tensor, *, do_reduce: bool = True) -> torch.Tensor:
        """按专家分组算路由部分，再对每个 token 的 top_k 项定序求和。

        Args:
            x: ``[tokens, H]``。
            index: ``[tokens, top_k]`` 的全局专家下标。
            weights: ``[tokens, top_k]`` 的路由权重。
            do_reduce: 是否在本函数末尾做跨 rank 归约；TP 融合路径传 ``False``，
                由上层把 ``experts`` 与 ``shared`` 合并成一次全归约。

        Returns:
            ``[tokens, H]``；不发生路由的 token 得到全零行。
        """
        tokens, top_k = index.shape
        flat_index = index.reshape(-1)
        old_multi_row = (os.environ.get("COFFIN_MOE_I8_GROUPED") == "0"
                         and tokens > 1)
        grouped_i8 = None if old_multi_row else self._grouped_i8
        if (self.parallel is not None and self.parallel.mode == "ep"
                and (self._dispatch is None or grouped_i8 is None)):
            raise RuntimeError(
                "expert parallel needs the native moe_dispatch and moe_grouped_i8 "
                "kernels; the per-expert fallback would index a full-width stack "
                "while this rank holds only its own slice")
        if self._dispatch is not None and grouped_i8 is not None:
            offsets, slot_pos, token_of_pos, slot_of_pos = self._dispatch(
                flat_index, self._expert_table(flat_index.device), flat_index.numel(),
                self.local_experts, self.num_experts, top_k)
            rows = int(token_of_pos.numel())
            # 张量核已支持任意段长与非 4 对齐偏移（``stage_at`` 内部段尾补零），
            # 无需 D2H 查表即可在预填充段直接走 19.2 TMAC/s 的张量核；中小段走
            # ``moe_grouped_i8_expert``（支持 K=128 与 K=2048）。
            tcu_ok = (self._grouped_i8_tcu is not None
                      and os.environ.get("COFFIN_MOE_I8_TCU") != "0"
                      and rows >= _TCU_ROW_MIN)
            expert_rows = (tcu_ok or (rows >= _EXPERT_ROW_MIN
                                      and self._grouped_i8_expert is not None))
            if ((grouped_i8 is not None or expert_rows) and self._gather_quant is not None
                    and self._silu_quant is not None
                    and (expert_rows or self._merge_tokens is not None)):
                xq, xs = self._gather_quant(x, token_of_pos)
                gu = self._act_buf(rows, 2 * self.inter, x.dtype, x.device)
                grouped = ((self._grouped_i8_tcu if tcu_ok else self._grouped_i8_expert)
                           if expert_rows else self._grouped_i8)
                gate_up = grouped(self.gate_up, xq, offsets, 0, xs, gu)
                gate, up = gate_up.chunk(2, dim=-1)
                act = self._act_buf(rows, self.inter, x.dtype, x.device)
                act, aq, a_s = self._silu_quant(gate, up, act)
                out = self._act_buf(tokens, self.hidden, x.dtype, x.device)
                if expert_rows:
                    hid = self._act_buf(rows, self.hidden, x.dtype, x.device)
                    grouped(self.down, aq, offsets, 0, a_s, hid)
                    if self._gather_sum is not None:
                        self._gather_sum(hid, slot_pos, weights, tokens, top_k, out=out)
                    else:
                        sl = slot_pos.reshape(-1)
                        if sl.numel() == rows:
                            g = hid.index_select(0, sl).view(tokens, top_k, self.hidden)
                            g.mul_(weights.reshape(tokens, top_k)[:, :, None])
                            out.copy_(g.sum(1))
                        else:
                            out.zero_()
                            out.index_add_(0, token_of_pos.long(),
                                           hid * weights.reshape(-1)[:, None])
                    return self._reduce(out, x.dtype) if do_reduce else out
                merged = self._merge_tokens(self.down, aq, offsets, weights.reshape(-1),
                                            slot_pos, slot_of_pos, tokens, top_k,
                                            scales=a_s, out=out)
                return self._reduce(merged, x.dtype) if do_reduce else merged
            packed = x.index_select(0, token_of_pos.long())
            gate, up = self._group(self.gate_up, packed, offsets).chunk(2, dim=-1)
            activated = bops.silu_mul(gate, up, up)
            if self._merge_tokens is not None:
                merged = self._merge_tokens(self.down, activated, offsets,
                                            weights.reshape(-1), slot_pos, slot_of_pos,
                                            tokens, top_k)
                return self._reduce(merged, x.dtype) if do_reduce else merged
            pos = slot_of_pos.long()
            flat_weights = weights.reshape(-1).index_select(0, pos)
            if self.local_experts != self.num_experts:
                keep = (torch.arange(pos.numel(), dtype=torch.int32, device=pos.device)
                        < offsets[-1])
                flat_weights = flat_weights * keep.to(flat_weights.dtype)
            if self._grouped_i8_merge is not None and tokens == 1:
                merged = self._grouped_i8_merge(self.down, activated, offsets, flat_weights)
                return self._reduce(merged, x.dtype) if do_reduce else merged
            hidden = self._group(self.down, activated, offsets)
            if os.environ.get("COFFIN_MOE_TAIL_FAST", "1") != "0":
                n = tokens
                buf = self._tail_buf
                if (buf is None or buf.shape[0] != n or buf.shape[1] != self.hidden
                        or buf.dtype != x.dtype or buf.device != x.device):
                    self._tail_buf = buf = torch.zeros(
                        (n, self.hidden), dtype=x.dtype, device=x.device)
                buf.zero_()
                buf.index_add_(0, token_of_pos.long(), hidden * flat_weights[:, None])
                return self._reduce(buf, x.dtype) if do_reduce else buf
            buf = torch.zeros((tokens * top_k, self.hidden),
                              dtype=x.dtype, device=x.device)
            buf.index_add_(0, pos, hidden * flat_weights[:, None])
            res = buf.view(tokens, top_k, self.hidden).sum(dim=1)
            return self._reduce(res, x.dtype) if do_reduce else res
        res = self._experts_dense(x, index, weights, tokens, top_k)
        return self._reduce(res, x.dtype) if do_reduce else res

    def _act_buf(self, rows: int, width: int, dtype: torch.dtype,
                 device: torch.device) -> torch.Tensor:
        """取（必要时分配）一块按调用形状复用的激活缓冲。

        Args:
            rows: 行数。
            width: 列数。
            dtype: 元素类型；跟随输入激活，别写死 bf16——投机校验的宽路径可以是 fp32。
            device: 目标设备。

        Returns:
            ``[rows, width]`` 的连续张量；同形状同类型同设备的后续调用复用同一块。
        """
        key = (rows, width, dtype, device)
        buf = self._act_bufs.get(key)
        if buf is None:
            if len(self._act_bufs) > 24:
                self._act_bufs.clear()
            self._act_bufs[key] = buf = torch.empty((rows, width), dtype=dtype,
                                                   device=device)
        return buf

    def _reduce(self, value: torch.Tensor, dtype: torch.dtype,
                add: torch.Tensor | None = None) -> torch.Tensor:
        """把各 rank 的部分和加起来；单卡 / 流水并行直接返回。

        两种并行都只算出部分和，所以这一步不是可选的优化：漏掉它得到的是安静的错误。

        * ``tp``：专家的 ``down`` 按 K（中间宽度）分片，每卡算的是一段 K 的部分和；
          优先走设备侧 P2P 归约（支持融合加本地 ``add``），不可用时退回 Gloo。
        * ``ep``：每卡只算自己那 1/world 专家的槽位，未命中的槽位是零行，走设备侧
          P2P 或同机共享内存求和。

        Args:
            value: 本 rank 的浮点结果。
            dtype: 返回的 dtype，与激活一致。
            add: 可选本地加数（TP 下合并 ``shared`` 时传入）。

        Returns:
            ``value``（单卡 / 流水）或跨 rank 求和后窄化回 ``dtype`` 的结果。
        """
        if self.parallel is None or self.parallel.mode not in {"tp", "ep"}:
            return value if add is None else (value + add)
        if self.parallel.mode == "tp":
            out = self.parallel.reduce_sum(value, add=add)
            return out if out.dtype == dtype else out.to(dtype)
        # 流水并行里专家是完整的，再加一次全归约等于把结果乘上世界大小。
        out = self.parallel.combine(value)
        if add is not None:
            out = out + add
        return out if out.dtype == dtype else out.to(dtype)

    def _group(self, stack, x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """用分组内核按专家投影一整栈 int8 权重。

        Args:
            stack: int8 专家栈（:class:`~int8_weights.QWeightMoE`，``[E, N, K]``）。
            x: ``[rows, K]``，已按专家号排好序。
            offsets: ``[E+1]`` 前缀和。

        Returns:
            ``[rows, N]``。

        Raises:
            ValueError: 栈不是 int8 的。稠密栈走 :meth:`_experts_dense`，那条路根本
                不会走到这里——在这里悄悄换成逐专家循环等于让调用点以为自己在跑内核。
        """
        if not isinstance(stack, QWeightMoE):
            raise ValueError(
                f"grouped projection needs an int8 stack, got {type(stack).__name__}; "
                f"dense stacks take the per-expert loop")
        # int8 栈只有分组内核这一条路：``moe_grouped_i8*`` 吃的是 ``[E, N, K]`` 的
        # int8 栈 + 逐 (专家, 列) scale，别的东西喂进去只会在形状校验里报错。
        return self._grouped_i8(stack, x, offsets)

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
        if isinstance(self.gate_up, QWeightMoE) and self._grouped_i8 is None:
            raise RuntimeError(
                "int8 expert stacks need the grouped int8 kernel, which this backend "
                "does not provide natively")
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

    def _expert_table(self, device: torch.device) -> torch.Tensor:
        """全局专家号 → 本地号的映射表；不归本 rank 的专家写 ``-1``。

        分派内核按这张表决定「哪些槽位留在本地、排到第几行」，所以它同时决定了本
        rank 的专家栈该有多宽。单卡 / ``pp`` / ``tp`` 下专家栈是完整的，它就是恒等
        映射；只有 ``ep`` 退化成 1/world 宽的子映射。

        Args:
            device: 表所在设备。

        Returns:
            ``[num_experts]`` 的 int32 张量；``-1`` 表示该专家的槽位不在这张卡上算。

        Raises:
            ValueError: ``ep`` 下专家数不能被世界大小整除（加载器会在更早的地方报，
                这里只是不让一张半对半错的表流进前向）。
        """
        table = self._table
        if table is not None and table.device == device:
            return table
        if self.parallel is not None and self.parallel.mode == "ep":
            begin, _end = self.parallel.expert_range(self.num_experts)
            table = torch.full((self.num_experts,), -1, dtype=torch.int32, device=device)
            table[begin:begin + self.local_experts] = torch.arange(
                self.local_experts, dtype=torch.int32, device=device)
        else:
            table = torch.arange(self.num_experts, dtype=torch.int32, device=device)
        self._table = table
        return table


__all__ = ["SparseMoE"]


