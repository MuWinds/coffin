"""Gated DeltaNet：48 层里 36 层（每个四层周期的前 3 层）的 mixer。

递推是带衰减的 delta 规则：状态每步先乘 ``exp(g)``，再用"新值减掉旧状态已经
预测出来的部分"做一次秩一修正，最后**用更新后的状态**读出输出。三个顺序敏感点
（先衰减后更新、读写次序、``kv_mem`` 沿 k 维求和）都是"错一个就整体偏移一个
token、短序列上看着差不多对"的类型。

Module: ``coffin/models/qwen4exp``。依赖 ``config``（几何）、``norm``
（``RMSNormGated``）、``util``（depthwise 卷积）、``proj``（稠密投影）和
``backends``（解码步的融合核与能力判定）；被 ``blocks`` 调用。

Constraints:
    * 递推状态是 ``[B, num_v_heads, head_k_dim, head_v_dim]`` 的 **fp32**。
      bf16 在几千 token 后会发散（36 层合计 113 MB/序列，容量规划按这个数算，
      不是按 hidden size）。
    * 解码步只在后端**原生**提供 ``delta_prologue_batch`` / ``delta_batch``、**且**
      融合路径已被真机验证（``_DECODE_FUSED_VERIFIED``）时走融合核。两个条件缺一
      不可：算子表说明内核在，那个常量说明整条路径被观察到跑完了。判据是
      ``caps().native`` 而不是算子表里有没有这个条目——参考后端把每个名字都补齐
      （它的实现就是这两个核的规范），但它一条要发几十次逐元素算子，比它替掉的循环
      还多，在没有内核的平台上接上去只会更慢。
    * 融合路径的输出必须**收窄回激活 dtype**：内核要 fp32 输出缓冲，而下游
      ``RMSNormGated`` / ``out_proj`` 按残差流的 dtype 工作。漏掉这一次舍入会让整条
      残差流变成 fp32，下一层的 ``b`` 跟着变宽，``delta_prologue_batch`` 当场拒绝。
      真机实测记录见 ``_DECODE_FUSED_VERIFIED``。
    * 融合路径**原地推进**传进来的 ``recurrent_state`` 并把同一个张量返回；纯 torch
      路径每步重新绑定新张量、从不改写入参。实测（CPU，``rep=4`` 覆盖头对齐路径）：
      融合关时「入参被就地改写=False」，融合开时「=True、且返回状态与入参同一块内存
      =True」；同一份初始 state 连跑两次融合 decode，两次输出**不**逐位相同
      （第二次被第一次的写污染）。
      所以**快照必须 ``clone()``，存引用会在下一步被就地改写**，而症状是输出微妙偏移、
      形状与 dtype 全对——任何断言都抓不住。
      ``Caps.recurrent_state`` 声明的是「本后端支持快照/恢复」，不是「调用方自己负责」。
      当前不发作：``models/qwen4exp`` 没有 ``snapshot_linear_states`` /
      ``restore_linear_states`` / ``prefill_group`` / ``decode_step`` 这一层，
      ``engine/`` 也完全不 import 本族（那些调用对的是 ``qwen35`` 的接口）。所以这是
      给接入引擎的人留的雷，不是当前 bug。
    * 卷积**不在**融合范围内。``conv1d_silu`` 的 ABI 要 fp32 的卷积权重和 fp32 的
      ``conv_state``，而本文件的卷积状态是 bf16 且由调用方持有、预填充分支也在写它。
      换成 fp32 会静默改变预填充的数值，所以在共享状态换精度之前这条不接。
    * 归一化那一步（``RMSNormGated``）同样不接后端：卡上的 ``rms_norm_gated`` 把门
      写死成 SiLU，而本配置是 ``sigmoid``。接错不报错，只会让 36 层的门控乘错函数。
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

from backends import caps as _caps
from backends import ops as bops
from models.qwen4exp.config import TextConfig
from models.qwen4exp.norm import RMSNormGated
from models.qwen4exp.proj import Proj
from models.qwen4exp.util import apply_padding_mask, causal_depthwise_conv

#: 融合解码步缺一不可的两个内核条目。
#:
#: 命中判定用 ``caps().native`` 而不是"算子表里有没有这个字段"：表在任何后端上都是
#: 满的（参考实现把每个名字都补齐了，它的实现就是这两个核的规范），拿它当开关等于
#: 在没有内核的平台上接上一串逐元素算子，比原来的循环还慢。
_DECODE_OPS = frozenset({"delta_prologue_batch", "delta_batch"})

#: 融合解码步是否已在这张卡上**实测**跑通。做成显式常量而不是从 ``_DECODE_OPS``
#: 推导，因为这是两个不同的问题：算子表说明两个内核都在，这个常量说的是**整条融合
#: 路径**被观察到真的跑完了。和 ``backends/corex/__init__.py`` 的 ``_GRAPH_VERIFIED``
#: 是同一个理由、同一类失败。
#:
#: 2026-11-02 第一次真机失败之后曾置 ``False``（作业 4517）：``_fused_decode`` 抛
#: ``RuntimeError: beta logits must be bf16``。逐层定位（作业 4520）找出病因：
#: **``_fused_decode`` 返回了 fp32**。``buf["out"]`` 是内核要求的 fp32 缓冲，但下游
#: ``RMSNormGated`` 用 ``input_dtype = hidden_states.dtype`` 决定回归宽度、``out_proj``
#: 的 ``_restore(out, x.dtype)`` 又跟着它——于是第 0 层输出 fp32，下一层
#: ``b = in_proj_b(fp32)`` 变 fp32，再下一层的 ``delta_prologue_batch`` 就炸了。
#: 修复是返回前 ``.to(dtype)``（见 :meth:`_fused_decode` 末尾）。
#:
#: 置回 ``True`` 的依据（作业 4521，同一张卡）：修复后 4 个 decode token 全部成功，
#: logits 保持 bf16，48 层残差流全程 bf16。
#:
#: 这个陷阱值得点名：本地是 reference 后端（``native=0``），``_fused_ready()`` 恒假，
#: 所以**单测一次都不会执行到这段代码**。等价实现对拍（rel 2e-7、rec_state 逐位
#: 一致）证明的只是接线逻辑——reference 版 ``_delta_prologue_batch`` 走 ``_head_prep()``，
#: 那条路根本没有 ``b.scalar_type()`` 检查，fp32 进去照样算。本地全绿与真机可用是
#: 两件事，判据必须是「在目标后端上跑过」。
_DECODE_FUSED_VERIFIED = True



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
    # 直接抛错。CoreX 后端的 ``solve.cu`` 就是为这件事写的，所以这里走
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

        # 融合解码步的中间缓冲区，按 (行数, 设备, 激活 dtype) 惰性分配。不能在构造期
        # 分配：量化路径在 ``torch.device("meta")`` 下建模块，那时分配等于直接抛错。
        # 也不能注册成 buffer：``.to(dtype=...)`` 会把它们一起转成 bf16，而
        # prologue/delta 的契约是 fp32——转窄之后内核会拒绝启动，或者更糟，把半精度
        # 当 fp32 读。惰性分配之后地址就不再变，所以第一次解码步之后再捕获图是安全的。
        self._decode_bufs: dict = {}
        #: ``-exp(A_log)`` / ``dt_bias`` 的缓存，见 :meth:`_decay_constants`。
        self._decay_cache: tuple | None = None
        #: 四个 ``in_proj`` 拼成一个权重后的闭包缓存，见 :meth:`_fused_in_proj`。
        #: ``None`` 表示还没算过；``((), None)`` 表示算过且不适用。
        self._fused_in_cache = None

    def _fused_ready(self) -> bool:
        """本后端是否真的能跑融合解码步。

        每次进解码分支都重算：``caps()`` 自己带缓存，而模块可能在 ``to()`` / 重新
        装载之后换了设备，一次性缓存的结果会指向上一台机器上分配过的缓冲区。

        除了算子表，还要 ``_DECODE_FUSED_VERIFIED``：见那个常量上的真机失败记录。
        算子表说明两个内核都在，它说明的是整条融合路径被观察到真的跑完了。

        Returns:
            两个内核都在 ``caps().native`` 里、且融合路径已被真机验证时为 ``True``。
        """
        return _DECODE_FUSED_VERIFIED and _DECODE_OPS <= _caps().native

    def _fused_in_proj(self):
        """把四个 ``in_proj`` 沿输出轴拼成一个权重，返回一次投影的闭包。

        四个投影共享 ``K = hidden_size`` 且都是 W4A8（同 ``group_size``），所以
        ``cat_output`` 的契约正好成立：``packed`` 是 ``[ceil(K/2), N_i]``、
        ``scale`` 是 ``[groups, N_i]``，沿 N 拼接后与拼好的 ``packed`` 一一对应 ——
        **不需要重排，数值逐位等价**。

        为什么值得：decode 一步里 W4A8 投影共 316 次 ``w4a8_dot`` + 约 403 次
        ``quant_act_i8``（``q4budget2``），而 GDN 的四个投影在 36 个线性层上各一次。
        合并后把这部分从「4 次投影 + 4 次量化」压到「1 次投影 + 1 次量化」，
        并且激活只读一遍。

        Returns:
            ``(hidden) -> packed_output`` 的闭包，输出最后一维是
            ``key_dim*2 + value_dim*2 + num_v_heads*2``；任一前置条件不满足时
            （非量化路径、有 bias、形状不合）返回 ``None``，调用方退回四次投影。
        """
        cached = self._fused_in_cache
        if cached is not None:
            return cached[1]

        from w4a8_weights import W4A8Weight

        parts = (self.in_proj_qkv, self.in_proj_z, self.in_proj_b, self.in_proj_a)
        weights = []
        for module in parts:
            weight = getattr(module, "_quantized", None)
            if not isinstance(weight, W4A8Weight) or module.bias is not None:
                # 非量化路径（bf16 ``[K,N]``）也能拼，但那条路不是 decode 的热路径，
                # 而拼接会让 ``pack.as_kn`` 的身份记忆化失效。所以只在 W4A8 上做。
                # 有 bias 的站点同样跳过：``W4A8Weight`` 里没有 bias。
                self._fused_in_cache = ((), None)
                return None
            weights.append(weight)
        merged = W4A8Weight.cat_output(weights)
        total_out = sum(module.out_features for module in parts)

        def project(hidden: torch.Tensor) -> torch.Tensor:
            """一次投影出四段拼接的结果。

            Args:
                hidden: ``[B, S, hidden_size]``。

            Returns:
                ``[B, S, total_out]``，dtype 同 ``hidden``。
            """
            # 走后端入口而不是 ``Proj.forward``：后者会再走一遍派发与 reshape，
            # 而这里形状已知（decode 是 1 行、prefill 是整段）。
            from backends import ops as bops

            flat = hidden.reshape(-1, self.hidden_size)
            return bops.linear_m(merged, flat).reshape(*hidden.shape[:-1], total_out)

        self._fused_in_cache = (tuple(id(w) for w in weights), project)
        return project

    def _decode_scratch(self, rows: int, device: torch.device,
                        dtype: torch.dtype) -> dict:
        """取（必要时分配）融合解码步的 fp32 中间缓冲区。

        按 ``(rows, device, dtype)`` 缓存，命中的条件是三者全等：``dtype`` 也在键里，
        是因为 ``act_dtype`` 决定 ``b``/``a`` 的位宽，换了后端就得换缓存而不是拿旧
        缓冲当新的用。

        Args:
            rows: 前导行数，即 ``batch``。
            device: 缓冲区所在设备。
            dtype: ``b``/``a`` 的 dtype，内核要求 bf16。

        Returns:
            六个键的字典：``q``/``k``/``v``/``beta``/``g``/``out``。
        """
        key = (rows, device, dtype)
        cached = self._decode_bufs.get(key)
        if cached is not None:
            return cached
        nv, dk, dv = self.num_v_heads, self.head_k_dim, self.head_v_dim
        nk = self.num_k_heads
        # 输入和输出**不能共用** ``q``/``k``：prologue 的读是键头布局
        # （``[B, num_k, dk]``）、写是值头布局（``[B, num_v, dk]``），``rep > 1`` 时
        # 两者重数不同。共用一块缓冲区就等于让一个 block 的写入踩掉另一个 block 还没
        # 读的输入，而结果看起来仍然"像个激活"。
        buf = {
            "q_in": torch.empty(rows, nk, dk, dtype=torch.float32, device=device),
            "k_in": torch.empty(rows, nk, dk, dtype=torch.float32, device=device),
            "v_in": torch.empty(rows, nv, dv, dtype=torch.float32, device=device),
            "q": torch.empty(rows, nv, dk, dtype=torch.float32, device=device),
            "k": torch.empty(rows, nv, dk, dtype=torch.float32, device=device),
            "v": torch.empty(rows, nv, dv, dtype=torch.float32, device=device),
            "beta": torch.empty(rows, nv, dtype=torch.float32, device=device),
            "g": torch.empty(rows, nv, dtype=torch.float32, device=device),
            "out": torch.empty(rows, nv, dv, dtype=torch.float32, device=device),
            "state": torch.empty(rows, nv, dk, dv, dtype=torch.float32, device=device),
        }
        self._decode_bufs[key] = buf
        return buf

    def _fused_decode(self, query: torch.Tensor, key: torch.Tensor,
                      value: torch.Tensor, b: torch.Tensor, a: torch.Tensor,
                      recurrent_state: torch.Tensor | None, rows: int,
                      device: torch.device, dtype: torch.dtype
                      ) -> tuple[torch.Tensor, torch.Tensor]:
        """用融合核跑一个解码 token，原地推进递推状态。

        ``query``/``key``/``value`` 是**未经 L2 归一化、未做头扩展**的原始投影切片，
        形状 ``[B, 1, num_k, dk]`` / ``[B, 1, num_v, dv]``——归一化、缩放、头扩展和
        beta/g 的构建全在 ``delta_prologue_batch`` 里做。这正是它与纯 torch 路径的
        分工差异：那边先在 torch 里做完这些再进递推。

        Args:
            query: 每个键头的原始查询，``[B, 1, num_k, dk]``。
            key: 每个键头的原始键，``[B, 1, num_k, dk]``。
            value: 值，``[B, 1, num_v, dv]``。
            b: beta logits，``[B, 1, num_v]``，bf16。
            a: 衰减 logits，``[B, 1, num_v]``，bf16。
            recurrent_state: 进来的递推状态 ``[B, num_v, dk, dv]`` fp32，**原地更新**；
                ``None`` 时新建全零。
            rows: ``B``。
            device: 缓冲区设备。
            dtype: ``b``/``a`` 的 dtype。

        Returns:
            ``(out, state)``：``out`` 是 ``[B, 1, num_v, dv]`` 的激活 dtype 视图，
            ``state`` 就是被原地推进的那块状态。

        Raises:
            ValueError: ``b`` / ``a`` 不是 bf16。这条硬断言**必须留在这里而不是交给
                内核**：``delta_prologue_batch`` 的 ``TORCH_CHECK`` 只会说
                ``beta logits must be bf16``，不含来源信息，而 ``b`` 的 dtype 跟着整条
                残差流走——真机上一次定位这个问题花了两轮作业（4517/4520），原因就是
                报错文案里看不出是"哪个上游把 dtype 提宽了"。在这里拦下并带上
                ``hidden`` 侧的信息，下一次能直接看出该往哪一层找。
        """
        if b.dtype != torch.bfloat16 or a.dtype != torch.bfloat16:
            raise ValueError(
                "fused decode needs bf16 beta/decay logits, got "
                f"b={b.dtype} a={a.dtype}. ``b`` is ``in_proj_b(hidden_states)`` and "
                f"its dtype follows the residual stream (input was {query.dtype}); a "
                "fp32 here means something upstream widened the stream — check the "
                "preceding layer's output dtype, not this kernel.")
        buf = self._decode_scratch(rows, device, dtype)
        # 输入切片在 ``seq == 1`` 时是最内层步长 1 的列切片，``copy_`` 会按步长读。
        # 喂给内核之前必须先物化成连续 fp32：两个核都只接受连续布局。
        buf["q_in"].copy_(query.reshape(rows, self.num_k_heads, self.head_k_dim))
        buf["k_in"].copy_(key.reshape(rows, self.num_k_heads, self.head_k_dim))
        buf["v_in"].copy_(value.reshape(rows, self.num_v_heads, self.head_v_dim))
        neg_expA, dt_bias = self._decay_constants(device)
        bops.delta_prologue_batch(
            buf["q_in"], buf["k_in"], buf["v_in"],
            b.reshape(rows, self.num_v_heads), a.reshape(rows, self.num_v_heads),
            neg_expA, dt_bias,
            buf["q"], buf["k"], buf["v"], buf["beta"], buf["g"],
            self.head_k_dim ** -0.5, self.key_rep)
        if recurrent_state is None:
            # 清零而不是新分配：状态在暖启动时是调用方传回来的同一块内存，这里只负责
            # "从零开始"这一种情况。分配一块新的会把地址换掉，捕获图的缓冲区假设就崩了。
            state = buf["state"].zero_()
        else:
            state = recurrent_state
        bops.delta_batch(buf["q"], buf["k"], buf["v"], buf["beta"], buf["g"],
                         state, buf["out"])
        # 收窄回调用方的激活 dtype 再返回。``buf["out"]`` 必须是 fp32（内核的契约），
        # 但下游 ``RMSNormGated`` 与 ``out_proj`` 期望的是与残差流一致的 bf16：
        # ``RMSNormGated`` 用 ``input_dtype = hidden_states.dtype`` 决定回归宽度，
        # 传 fp32 进去它就一路 fp32，``out_proj`` 的 ``_restore(out, x.dtype)`` 也就
        # 把 fp32 留在残差流里。真机实测（作业 4520）：漏掉这一步时 decode 的第 0 层
        # 输出就是 fp32，下一层的 ``b = in_proj_b(fp32)`` 变 fp32，于是
        # ``delta_prologue_batch`` 抛 ``beta logits must be bf16``。
        #
        # 这也是 ``corex-kernel-check/docs/FUSION.md`` 那条「递推的 fp32 输出必须在归一化前先舍入到
        # bf16」——当时的理由是逐位一致，现在多一条：它还决定 dtype 契约。
        return buf["out"].to(dtype).unsqueeze(1), state

    def _decay_constants(self, device: torch.device
                         ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回内核要的两个**常数**张量：``-exp(A_log)`` 和 ``dt_bias``，都是 fp32。

        缓存是必须的，不是优化：这两个常数由 checkpoint 决定、步骤之间不变，而不带
        缓存每层每 token 要多发 4 个逐元素内核（cast、neg、exp、再加一次 cast），
        36 层合计 144 次启动——正好把融合省下来的那笔又还回去。

        失效靠 ``_version``：``load_into`` 用 ``copy_`` 写参数，那会推进版本号，而
        重新 ``to()`` 会换掉设备。两者任一变化都重算。

        Args:
            device: 目标设备，与 ``A_log`` 不一致时重算。

        Returns:
            ``(neg_expA, dt_bias)``，都是 ``[num_v_heads]`` 的 fp32。
        """
        version = (self.A_log._version, self.dt_bias._version, device)
        if self._decay_cache is not None and self._decay_cache[0] == version:
            return self._decay_cache[1], self._decay_cache[2]
        # 内核的约定是 ``g = neg_expA * softplus(a + dt_bias)``，所以符号在这里翻，
        # 而不是像纯 torch 路径那样写成 ``-A_log.exp() * softplus(...)``。
        neg_expA = -self.A_log.detach().float().exp()
        dt_bias = self.dt_bias.detach().float()
        self._decay_cache = (version, neg_expA, dt_bias)
        return neg_expA, dt_bias

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

        # 四个 in_proj 吃**同一份** hidden_states。分开调用会把激活重复读四遍、
        # 并且各发一次内核（W4A8 路径上每次还额外带一次 ``quant_act_i8``）。
        #
        # 实测依据（作业 ``q4budget2``/``q4attrib3``）：一步 decode 里 ``w4a8_dot`` 316 次、
        # ``quant_act_i8`` 约 403 次，而 GDN 的四个投影在 36 个线性层上各出现一次 ——
        # 合并后这部分从 4 次投影 + 4 次量化降到 1 次投影 + 1 次量化。
        #
        # 用 :meth:`W4A8Weight.cat_output` 沿输出轴拼：四者共享 ``K`` 与 ``group_size``，
        # 而 scale 是逐组的 ``[groups, N]``，沿 N 拼接正好与拼接后的 packed 对齐，
        # 所以**不需要额外转置或重排，数值上也逐位等价**。
        fused = self._fused_in_proj()
        if fused is not None:
            packed = fused(hidden_states)
            qkv_n = self.key_dim * 2 + self.value_dim
            mixed_qkv = packed[..., :qkv_n].transpose(1, 2)
            z = packed[..., qkv_n:qkv_n + self.value_dim].reshape(
                batch, seq, -1, self.head_v_dim)
            b = packed[..., qkv_n + self.value_dim:qkv_n + self.value_dim
                       + self.num_v_heads]
            a = packed[..., qkv_n + self.value_dim + self.num_v_heads:]
        else:
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
        # 融合解码步接手头的归一化/扩展/beta-g，所以它要的是 split 之后的原始切片，
        # 不能再走下面那套 repeat_interleave。
        if seq == 1 and self._fused_ready():
            core_attn_out, new_state = self._fused_decode(
                query, key, value, b, a, recurrent_state, batch,
                hidden_states.device, b.dtype)
        else:
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
