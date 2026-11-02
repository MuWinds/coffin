"""MoE：TopK 路由、512 个融合 3D 专家、共享专家。

三层结构，逐条对应 ``ARCH_SPEC.md`` §5：

* **路由没有归一化输入**，也没把输入再过一遍 RMSNorm——它直接吃读门塌缩后的
  2560 宽激活；softmax 在 fp32 上做完才 top-k，权重取自**概率**而不是 logits；
  ``norm_topk_prob`` 为真时权重是"top-k 概率再除以它们自己的和"。
* **专家是 3D Parameter**：``gate_up_proj[e]`` 是 ``[2*I, H]``（前 I 行 gate、
  后 I 行 up），``down_proj[e]`` 是 ``[H, I]``。checkpoint 里也是融合张量，
  不是 ``nn.Linear``。
* **共享专家的融合是逐 token 的标量 sigmoid 门**（``[H] -> [1]``），不是逐通道；
  融合就是相加，没有再归一化。

数学上整块的输出是
``y[t] = Σ_k w[t,k] * Expert_{idx[t,k]}(x[t]) + sigmoid(g_s[t]) * Shared(x[t])``。

Module: ``coffin/models/qwen4exp``。依赖 ``config`` 与 ``proj``（共享专家是稠密投影）；被
``blocks`` 调用。

Constraints:
    * 聚合必须按专家分组做（``index_add_``），不能展开成"每 token 跑 10 次完整
      GEMM"：真实配置下专家权重是 ``[512, 1280, 2560]``，展开会读 10 倍权重。
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
        # softmax 在 fp32 上做再 top-k：低精度下 top-k 的顺序会抖动，而顺序决定
        # 选中哪个专家。
        router_probs = F.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            router_top_value = router_top_value / router_top_value.sum(dim=-1, keepdim=True)
        return router_logits, router_top_value.to(router_logits.dtype), router_indices


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

    def _make_grouped(self):
        """返回一个 ``(x, offsets, down_proj=False) -> y`` 的分派函数。

        Returns:
            闭包。``down_proj=True`` 时投影 ``[tokens, I] -> [tokens, H]``，
            否则 ``[tokens, H] -> [tokens, 2*I]``。
        """
        from backends import ops as bops

        gate_up_weight, down_weight = self._quant_gate_up, self._quant_down

        def grouped(x: torch.Tensor, offsets: torch.Tensor,
                    down_proj: bool = False) -> torch.Tensor:
            # 形参不能叫 ``down``：那会遮蔽外层的 ``down_weight``，于是
            # ``weight.k`` 变成对 ``bool`` 取属性（实测报
            # ``'bool' object has no attribute 'k'``，而错误位置在内核入口）。
            weight = down_weight if down_proj else gate_up_weight
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
        table = torch.full((self.num_experts,), -1, dtype=torch.long,
                           device=global_index.device)
        table[torch.tensor(self.local_experts, dtype=torch.long, device=global_index.device)] = \
            torch.arange(len(self.local_experts), dtype=torch.long, device=global_index.device)
        return table[global_index]

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

        # 分片时先扔掉不归本 rank 的槽位：它们要用 all-to-all 发出去，在这里算
        # 只会白读权重。
        if self.local_experts is not None:
            keep = self.local_index(flat_index) >= 0
            if not bool(keep.any()):
                return final
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
