"""Gated DeltaNet：48 层里 36 层（每个四层周期的前 3 层）的 mixer。

递推是带衰减的 delta 规则：状态每步先乘 ``exp(g)``，再用"新值减掉旧状态已经
预测出来的部分"做一次秩一修正，最后**用更新后的状态**读出输出。三个顺序敏感点
（先衰减后更新、读写次序、``kv_mem`` 沿 k 维求和）都是"错一个就整体偏移一个
token、短序列上看着差不多对"的类型。

Module: ``coffin/models/qwen4exp``。依赖 ``config``（几何）、``norm``
（``RMSNormGated``）、``util``（depthwise 卷积）和 ``proj``（稠密投影）；被
``blocks`` 调用。

Constraints:
    * 递推状态是 ``[B, num_v_heads, head_k_dim, head_v_dim]`` 的 **fp32**。
      bf16 在几千 token 后会发散（36 层合计 113 MB/序列，容量规划按这个数算，
      不是按 hidden size）。
    * 卷积状态存的是 ``in_proj_qkv`` 之后、**激活之前**的 ``mixed_qkv``，长度
      ``kernel - 1 = 3``。存激活之后的值会让热启动的每一步都偏。
    * ``l2norm`` 是"平方和 + eps 再 rsqrt"，和 RMSNorm 的"均方 + eps"不是一回事；
      eps 加在开方**里面**。复用同一个算子会让 q/k 的尺度整体差一个常数。
    * 缩放 ``head_k_dim ** -0.5`` 必须发生在 l2norm **之后**。
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F
from torch import nn

from models.qwen4exp.config import TextConfig
from models.qwen4exp.norm import RMSNormGated
from models.qwen4exp.proj import Proj
from models.qwen4exp.util import apply_padding_mask, causal_depthwise_conv


def solve_triangular(lower: torch.Tensor, rhs: torch.Tensor,
                     unitriangular: bool = True) -> torch.Tensor:
    """为单位下三角 ``lower`` 解 ``lower @ X = rhs``。

    优先走 CoreX 的 ``triangular_solve`` 内核。本卡的 torch wheel 在每条
    ``torch.linalg`` 求解路径上都会去 ``dlopen`` 一个不存在的 ``libcusolver.so``，
    所以那条路在卡上是不能用的；而内核版本不需要 cuSOLVER。

    内核不可用时回退到 ``torch.linalg.solve_triangular``，这样纯 CPU 的数值回归
    （那 44 条测试就在这条路上）不需要 CoreX 也能跑。

    Args:
        lower: ``[..., L, L]`` 单位下三角，对角线按 1 处理。
        rhs: ``[..., L, n]`` 右端项。
        unitriangular: 对角线是否为 1。目前只支持 ``True``；``False`` 直接走回退。

    Returns:
        ``[..., L, n]``，dtype 同 ``rhs``。

    Raises:
        RuntimeError: 内核与回退都不可用（只在既没有 CoreX、torch 又答不了时发生）。
    """
    # ``is_cuda`` 是必须的守卫，不是优化：CPU 上 ``backends.corex`` 照样 import
    # 成功，把 CPU 张量喂给设备内核**不报错**，只会算出错的值。纯 CPU 的数值回归
    # 就是被这一点弄红的（gdn prefill rel 0.758），而报错信息完全指不到这里。
    if (unitriangular and lower.dim() >= 3 and rhs.dim() == lower.dim()
            and lower.is_cuda and rhs.is_cuda):
        try:
            from backends.corex import triangular_solve as _solve
            # 内核只接受 3-D ``[bn, L, L]``，而这里拿到的是 5-D：GDN 的形状是
            # ``[batch, num_v_heads, num_chunks, L, L]``。不展平就会被内核以形状
            # 为由拒绝，然后回退到本卡答不了的 ``torch.linalg``——报出来的是
            # cuSOLVER 缺库，与真正的原因（维度数不对）毫无关系。
            lead = lower.shape[:-2]
            flat_lower = lower.reshape(-1, lower.shape[-2], lower.shape[-1])
            flat_rhs = rhs.reshape(-1, rhs.shape[-2], rhs.shape[-1])
            solved = _solve(flat_lower.contiguous(), flat_rhs.contiguous())
            # 内核在内部用 fp32 累加，输出 dtype 因此可能比 ``rhs`` 宽。这里窄回
            # ``rhs`` 的 dtype：下游是 bf16 的矩阵乘，宽出来会报
            # ``expected mat1 and mat2 to have the same dtype: BFloat16 != float``。
            solved = solved.reshape(*lead, solved.shape[-2], solved.shape[-1])
            return solved.to(rhs.dtype)
        except Exception as exc:  # noqa: BLE001
            # 静默回退会掩盖真正的原因：本卡的 ``torch.linalg`` 同样答不了，于是
            # 用户看到的永远是回退那次的 cuSOLVER 报错，内核为什么不可用则完全
            # 看不见。设 ``COFFIN_Q4_SOLVE_DEBUG=1`` 把它打出来。
            if os.environ.get("COFFIN_Q4_SOLVE_DEBUG"):
                print(f"[solve_triangular] kernel unavailable: "
                      f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    return torch.linalg.solve_triangular(lower, rhs, upper=False,
                                         unitriangular=unitriangular)


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """L2 归一化。

    Args:
        x: 输入激活。
        dim: 归一化沿的维。
        eps: 加在**平方和**上的常数（不是均方上）。

    Returns:
        与 ``x`` 同形状。
    """
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def recurrent_gated_delta_rule(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                               g: torch.Tensor, beta: torch.Tensor,
                               initial_state: torch.Tensor | None = None,
                               use_qk_l2norm: bool = True
                               ) -> tuple[torch.Tensor, torch.Tensor]:
    """逐 token 的 gated delta 规则（解码路径）。

    Args:
        query: ``[B, S, H, dk]``。
        key: ``[B, S, H, dk]``。
        value: ``[B, S, H, dv]``。
        g: 对数衰减，``[B, S, H]``，恒 ``<= 0``。
        beta: 写入门，``[B, S, H]``。
        initial_state: 进来的状态，``[B, H, dk, dv]`` 的 fp32；``None`` 表示全零。
        use_qk_l2norm: 是否在 kernel 内做 L2 归一化。两条路径都传 ``True``。

    Returns:
        ``(output, final_state)``：输出 ``[B, S, H, dv]``（与 ``query`` 同 dtype），
        状态 ``[B, H, dk, dv]`` 的 fp32。

    Raises:
        ValueError: ``g`` / ``beta`` 的形状与 query 的头布局不符。
    """
    initial_dtype = query.dtype
    batch, seq, _, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]
    if g.shape != (batch, seq, num_v_heads) or beta.shape != g.shape:
        raise ValueError(
            f"g/beta must be [B, S, H]=[{batch}, {seq}, {num_v_heads}], got "
            f"{tuple(g.shape)} and {tuple(beta.shape)}.")

    # 整个循环在 fp32 里跑：状态跨很多步累积，bf16 会在几千 token 后发散。
    query, key, value, beta, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, g)
    ]
    if use_qk_l2norm:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    # 缩放必须在 l2norm 之后：先缩放会被归一化吃掉。
    query = query / (query.shape[-1] ** 0.5)

    if initial_state is None:
        state = torch.zeros((batch, num_v_heads, k_head_dim, v_head_dim),
                            dtype=value.dtype, device=value.device)
    else:
        state = initial_state.to(value)
    out = torch.zeros_like(value)

    for index in range(seq):
        q_t, k_t, v_t = query[:, :, index], key[:, :, index], value[:, :, index]
        # 先衰减，再用 delta 修正；反过来会把这一步的新信息也乘上衰减。
        state = state * decay[:, :, index].exp()[..., None, None]
        beta_t = beta[:, :, index].unsqueeze(-1)
        # kv_mem 沿 k_head_dim 求和（等价 kᵀS），写成 -1 会把 v 维也求和掉。
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        # 读输出用**更新后**的状态；先读后写会让输出整体偏移一个 token。
        out[:, :, index] = (state * q_t.unsqueeze(-1)).sum(dim=-2)

    out = out.transpose(1, 2).contiguous().to(initial_dtype)
    return out, state


def chunk_gated_delta_rule(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                           g: torch.Tensor, beta: torch.Tensor, chunk_size: int = 64,
                           initial_state: torch.Tensor | None = None,
                           use_qk_l2norm: bool = True
                           ) -> tuple[torch.Tensor, torch.Tensor]:
    """分块形式的 gated delta 规则（预填充路径）。

    两阶段：块内量（衰减矩阵、UT 系统、两次三角求解、衰减后的 q/k）对全部分块
    一次性算完；然后对分块串行扫状态。数学与递推形式逐位等价的只是理想情况，
    fp32 下相对误差约 1e-6。

    Args:
        query: ``[B, S, H, dk]``。
        key: ``[B, S, H, dk]``。
        value: ``[B, S, H, dv]``。
        g: 对数衰减，``[B, S, H]``。
        beta: 写入门，``[B, S, H]``。
        chunk_size: 每个分块的 token 数。
        initial_state: 进来的状态 ``[B, H, dk, dv]`` fp32 或 ``None``。
        use_qk_l2norm: 是否做 L2 归一化。

    Returns:
        ``(output, final_state)``，形状同 :func:`recurrent_gated_delta_rule`。

    Raises:
        ValueError: 序列长度或头布局不合法。
    """
    initial_dtype = query.dtype
    batch, seq, _, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]
    if g.shape != (batch, seq, num_v_heads):
        raise ValueError(
            f"g must be [B, S, H]=[{batch}, {seq}, {num_v_heads}], got {tuple(g.shape)}.")

    query, key, value, beta, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, g)
    ]
    if use_qk_l2norm:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query = query * (query.shape[-1] ** -0.5)

    # 右填零到 chunk_size 整数倍。填出来的行 g=beta=0，对状态没有任何贡献。
    pad = (chunk_size - seq % chunk_size) % chunk_size
    query, key, value = (F.pad(x, (0, 0, 0, pad)) for x in (query, key, value))
    beta, decay = (F.pad(x, (0, pad)) for x in (beta, decay))
    total = seq + pad
    num_chunks = total // chunk_size

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    query, key, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, k_beta, v_beta)
    ]
    decay = decay.reshape(decay.shape[0], decay.shape[1], -1, chunk_size)
    # 严格上三角：因果分块里每个 token 看不到自己之后的位置。
    strictly_upper = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device).triu(1)
    cum_decay = decay.cumsum(dim=3)

    # pairwise_decay[i, j] = exp(cum_i - cum_j)：位置 j 到 i 之间累积的衰减。
    # 上三角先置 -inf 再 exp，否则 exp 出来的正指数会溢出。
    pairwise_decay = cum_decay.unsqueeze(4) - cum_decay.unsqueeze(3)
    pairwise_decay = pairwise_decay.masked_fill(strictly_upper, float("-inf")).exp()

    ut_system = (k_beta @ key.transpose(-1, -2)) * pairwise_decay
    intra_chunk_attn = (query @ key.transpose(-1, -2)) * pairwise_decay
    decayed_k_beta = k_beta * cum_decay.exp().unsqueeze(-1)

    # UT 变换把多个 delta 更新压成几次矩阵乘。解完得到 new_values（论文里的 u）
    # 和 k_cumdecay（用旧状态读那部分 key 所需的衰减）。
    #
    # 求解**不能**用 ``torch.linalg.solve_triangular``：本卡 torch wheel 会去
    # ``dlopen`` 一个不存在的 ``libcusolver.so``，每条 ``torch.linalg`` 求解路径都
    # 直接抛错。``backends/corex/kernels/solve.cu`` 就是为这件事写的，所以这里走
    # :func:`solve_triangular`——它在没有 CoreX 时回退到 torch，在卡上走内核。
    new_values = solve_triangular(ut_system, v_beta, unitriangular=True)
    k_cumdecay = solve_triangular(ut_system, decayed_k_beta, unitriangular=True)

    if initial_state is None:
        state = torch.zeros((batch, num_v_heads, k_head_dim, v_head_dim),
                            dtype=new_values.dtype, device=new_values.device)
    else:
        state = initial_state.to(new_values)
    out = torch.zeros_like(new_values)

    # 衰减只应用一次，而不是每块各乘一遍；这两步必须在 intra_chunk_attn 之后，
    # 否则块内注意力会"提前衰减一次"。
    query = query * cum_decay.exp().unsqueeze(-1)
    key = key * (cum_decay[..., -1:] - cum_decay).exp().unsqueeze(-1)
    chunk_decay = cum_decay[..., -1].exp()[..., None, None]

    for index in range(num_chunks):
        # 旧状态已经预测出来的那部分要减掉，只把修正量写进状态：这就是 delta 规则。
        v_new = new_values[:, :, index] - k_cumdecay[:, :, index] @ state
        inter_chunk_attn = query[:, :, index] @ state
        out[:, :, index] = inter_chunk_attn + intra_chunk_attn[:, :, index] @ v_new
        state = state * chunk_decay[:, :, index] + key[:, :, index].transpose(-1, -2) @ v_new

    out = out.reshape(batch, num_v_heads, -1, v_head_dim)[:, :, :seq]
    out = out.transpose(1, 2).to(initial_dtype, memory_format=torch.contiguous_format)
    return out, state


class GatedDeltaNet(nn.Module):
    """线性注意力 mixer：投影 → 因果卷积 → delta 规则 → 门控归一化 → 输出投影。

    Attributes:
        num_v_heads: value 头数（真实 checkpoint 是 48）。
        num_k_heads: key/query 头数（16）。
        conv_dim: ``2 * key_dim + value_dim``，卷积分支的通道数。
    """

    def __init__(self, config: TextConfig, layer_idx: int) -> None:
        """按配置建投影、卷积、门控归一化与衰减参数。

        Args:
            config: 文本配置。
            layer_idx: 零基层号。
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.key_rep = config.linear_head_rep
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.layer_norm_epsilon = config.rms_norm_eps

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim, out_channels=self.conv_dim, bias=False,
            kernel_size=self.conv_kernel_size, groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )
        # dt_bias 初始化为 1、A_log = log(uniform(0.01, 16))：下限远离 0 是为了
        # 让 log(A) 永远不变 -inf。两者都保持 fp32，被逐元素内核消费。
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads))
        self.norm = RMSNormGated(
            self.head_v_dim, eps=self.layer_norm_epsilon,
            activation=config.output_gate_type or config.hidden_act)
        self.out_proj = Proj(self.value_dim, self.hidden_size)

        self.in_proj_qkv = Proj(self.hidden_size, self.key_dim * 2 + self.value_dim)
        self.in_proj_z = Proj(self.hidden_size, self.value_dim)
        # b/a 只有 48 行（num_v_heads），是整步里最窄的一档：一个 ``[48, 2560]`` 的
        # 行主序矩阵在 M=1 上实测 0.49 ms，换 ``[K, N]`` 后 0.078 ms（作业 4489）。
        # 48 层加起来 216 个这样的投影占掉旧路径 245.2 ms，所以它们必须和宽投影走
        # 同一条路，不能因为「太小了 F.linear 也无所谓」被放过。
        self.in_proj_b = Proj(self.hidden_size, self.num_v_heads)
        self.in_proj_a = Proj(self.hidden_size, self.num_v_heads)

    def forward(self, hidden_states: torch.Tensor, conv_mask: torch.Tensor | None = None,
                conv_state: torch.Tensor | None = None,
                recurrent_state: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """跑一次线性注意力。

        Args:
            hidden_states: ``[B, S, hidden_size]``。
            conv_mask: ``[B, S]`` 的 keep/drop 标记（**不是**因果掩码）；padding
                位置在进卷积前要清零。
            conv_state: 卷积窗口历史，``[B, conv_dim, kernel - 1]``，存的是
                投影之后、激活之前的 ``mixed_qkv``。
            recurrent_state: 递推状态 ``[B, num_v_heads, head_k_dim, head_v_dim]``
                的 fp32 张量；``None`` 表示从零开始。

        Returns:
            ``(output, new_conv_state, new_recurrent_state)``。输出是
            ``[B, S, hidden_size]``。

        Raises:
            ValueError: ``conv_state`` 的宽度不是 ``kernel - 1``。
        """
        hidden_states = apply_padding_mask(hidden_states, conv_mask)
        batch, seq, _ = hidden_states.shape

        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        z = self.in_proj_z(hidden_states).reshape(batch, seq, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        # 卷积状态存投影之后、激活之前的值：激活是卷积之后才做的，把它一起存
        # 会让热启动时多算一次 silu。
        if conv_state is None:
            conv_state = mixed_qkv.new_zeros(batch, self.conv_dim, self.conv_kernel_size - 1)
        elif conv_state.shape[-1] != self.conv_kernel_size - 1:
            raise ValueError(
                f"conv state must be {self.conv_kernel_size - 1} columns, got "
                f"{conv_state.shape[-1]}.")
        combined = torch.cat([conv_state, mixed_qkv], dim=-1)
        new_conv_state = combined[..., -(self.conv_kernel_size - 1):].contiguous()
        convolved = causal_depthwise_conv(mixed_qkv, self.conv1d.weight,
                                         history=conv_state, dilation=1,
                                         activation=self.activation)

        convolved = convolved.transpose(1, 2)
        query, key, value = torch.split(
            convolved, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch, seq, -1, self.head_k_dim)
        key = key.reshape(batch, seq, -1, self.head_k_dim)
        value = value.reshape(batch, seq, -1, self.head_v_dim)

        beta = b.sigmoid()
        # .float() 是必须的：fp16 加载时 A 可能是 -inf。g 恒 <= 0（-exp 恒负、
        # softplus 恒正），是 log 空间的逐步衰减。
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.key_rep > 1:
            # 第 j 个 V 头用第 j // rep 个 K 头，必须在 l2norm 之后对齐。
            query = query.repeat_interleave(self.key_rep, dim=2)
            key = key.repeat_interleave(self.key_rep, dim=2)

        # 参考实现在 seq_len == 1 时走递推、否则走分块；两条路径的数值必须一致，
        # 而递推是 decode 的真实路径，所以这里按同一条件分派。
        if seq == 1:
            core_attn_out, new_state = recurrent_gated_delta_rule(
                query, key, value, g=g, beta=beta, initial_state=recurrent_state,
                use_qk_l2norm=True)
        else:
            core_attn_out, new_state = chunk_gated_delta_rule(
                query, key, value, g=g, beta=beta, initial_state=recurrent_state,
                use_qk_l2norm=True)

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch, seq, -1)
        return self.out_proj(core_attn_out), new_conv_state, new_state


__all__ = ["GatedDeltaNet", "chunk_gated_delta_rule", "l2norm", "recurrent_gated_delta_rule"]
