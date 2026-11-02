"""Qwen4-Exp 里两处共享的小工具：padding 清零与因果逐通道卷积。

两件事在族里都有两个调用点（GatedDeltaNet 的 kernel=4 无 dilation 卷积、
PLE 的 kernel=4 dilation=3 卷积），而它们的边界语义正是最容易写错的地方：
"输出第 i 个 token 覆盖输入 ``[i - dilation*(k-1), i]``"。写两遍就有两个版本，
其中一个是错的，形状还都对。

Module: ``coffin/models/qwen4exp``。只依赖 ``torch``；被 ``gdn`` 和 ``ngram``
调用。

Constraints:
    * 卷积一律在 ``[B, C, L]`` 布局上做，权重是 depthwise 的
      ``[C, 1, k]``，``groups = C``。``[B, L, C]`` 上做 depthwise 会把
      通道当成 batch，形状检查抓不到。
    * 左侧历史不足时补 0 而不是复制首行：参考实现补的就是 0，序列开头那几步
      的历史本来是"没有 token"，补首行等于凭空多喂一个 token。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def apply_padding_mask(hidden_states: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """把 padding 位置的激活清零。

    Args:
        hidden_states: ``[B, S, C]`` 的激活。
        mask: ``[B, S]`` 的 keep/drop 标记（1/``True`` 保留）；``None`` 时原样返回。

    Returns:
        清零后的激活，dtype 与入参一致。

    Raises:
        ValueError: ``mask`` 的最后一维与 ``hidden_states`` 的序列维对不上。
            这里必须炸：掩码按 ``[B, S, None]`` 广播，把 4 维的因果掩码误传进来
            会先广播到 ``[B, S, S]`` 再深潜到某个算子才报错，报错点离调用点很远。
    """
    if mask is None:
        return hidden_states
    if mask.shape[-1] != hidden_states.shape[1]:
        raise ValueError(
            "padding mask must be [B, S] matching the activation's sequence axis, got "
            f"mask {tuple(mask.shape)} for activations {tuple(hidden_states.shape)}.")
    dtype = hidden_states.dtype
    return (hidden_states * mask[:, :, None]).to(dtype)


def causal_depthwise_conv(x: torch.Tensor, weight: torch.Tensor, history: torch.Tensor | None = None,
                          dilation: int = 1, activation: str | None = None) -> torch.Tensor:
    """因果逐通道卷积：输出第 i 步只看得到 ``[i - dilation*(k-1), i]``。

    实现方式是"先把历史拼到左边、右侧补零、再把补出来的尾巴切掉"，因此输出的
    第 ``i`` 步永远对齐输入的第 ``i`` 步，不需要调用方再切。

    Args:
        x: 新 token 的输入，``[B, C, S]``。
        weight: depthwise 卷积核，``[C, 1, k]``。
        history: 更早一步的状态，``[B, C, h]``；``h`` 应该等于
            ``dilation * (k - 1)``，多给少给都不会抛错但感受野会偏移。
        dilation: 采样间隔。PLE 的短卷积用 3，GatedDeltaNet 用 1。
        activation: 卷积后的激活名，``None`` 或 ``"silu"``。

    Returns:
        ``[B, C, S]``，与 ``x`` 的序列长度一致。

    Raises:
        ValueError: 权重不是 depthwise 形状，或历史长度不是
            ``dilation * (k - 1)``。两者都会让输出在时间轴上错位，而错位后的
            数值仍然"像激活"。
    """
    if weight.dim() != 3 or weight.shape[1] != 1:
        raise ValueError(f"depthwise conv weight must be [C, 1, k], got {tuple(weight.shape)}.")
    batch, channels, new_len = x.shape
    if weight.shape[0] != channels:
        raise ValueError(
            f"conv weight has {weight.shape[0]} channels but input has {channels}.")
    kernel = weight.shape[-1]
    span = dilation * (kernel - 1)
    if history is None:
        # 没有历史时补零而不是补首行：序列开头那几步的历史本来是"没有 token"，
        # 补首行等于凭空多喂一个 token。补零也让下面的切片偏移恒定。
        history = x.new_zeros(batch, channels, span)
    elif history.shape[-1] != span:
        raise ValueError(
            f"conv history must be {span} columns for kernel={kernel} dilation={dilation}, "
            f"got {history.shape[-1]}.")
    x = torch.cat([history, x], dim=-1)

    # padding 取 span，于是 out[j] 覆盖输入 [j - span, j]，因果性落实成"以第 j 步
    # 结尾"。拼上历史之后第 i 个新 token 位于下标 span + i，对应的输出也就在
    # 那里——输出长度是 L + span，直接切 [span, span + new_len) 即可。
    out = F.conv1d(x.to(weight.dtype), weight, bias=None,
                   padding=span, dilation=dilation, groups=channels)
    out = out[..., span:span + new_len]
    if activation == "silu":
        out = F.silu(out)
    elif activation is not None:
        raise ValueError(f"unsupported conv activation {activation!r}.")
    return out.to(x.dtype)


__all__ = ["apply_padding_mask", "causal_depthwise_conv"]
