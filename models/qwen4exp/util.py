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
    # 两条路径，按"本段长度是否盖过窗口"分：
    #
    #  * ``new_len >= span``（预填充分块的常态）：**不把历史 cat 进去**。``conv1d`` 的
    #    左侧零填充等价于"历史全零"，所以只有最前面 ``span`` 个输出位与"拼上历史"不同
    #    ——差的正是历史上那些抽头的贡献，主卷积算完之后用一块 ``[B, C, span]`` 的小
    #    张量补回来即可。那次 cat 要物化整块 ``[B, C, span + L]``：GatedDeltaNet 的
    #    ``conv_dim = 10240``、prefill 每块 512 个 token 时 20 MB，36 个线性层每层一次
    #    （作业 314：``CatArrayBatchedCopy`` 72 次 39.69 ms，其中一半就是它），而真正
    #    需要搬运的历史只有 ``span`` 列。输出对齐也随之变简单：没有前缀之后
    #    ``out[i]`` 直接覆盖输入 ``[i - span, i]``，切前 ``new_len`` 个即可。
    #
    #  * ``new_len < span``（decode 每步 1 个 token）：保持原来的 cat 路径。那时
    #    历史的字节数和输入同量级（``conv_dim x span``），快路径省不到什么，却会让
    #    唯一那个输出位都多付一次 bf16 舍入——decode 是所有性能红线的口径，不拿它换。
    if span and new_len >= span:
        out = F.conv1d(x.to(weight.dtype), weight, bias=None,
                       padding=span, dilation=dilation, groups=channels)
        out = out[..., :new_len]
        # 修正量用 fp32 算：``conv1d`` 自己也是 fp32 累加，用 bf16 算这一小块会让它比
        # 原来的 cat 版本更不准，而那正是"看起来对、逐位对不上"的来源。
        correction = _history_taps(history.float(), weight[:, 0, :].float(), span,
                                   kernel, dilation, span)
        out[..., :span] += correction.to(out.dtype)
    else:
        extended = torch.cat([history, x], dim=-1)
        out = F.conv1d(extended.to(weight.dtype), weight, bias=None,
                       padding=span, dilation=dilation, groups=channels)
        out = out[..., span:span + new_len]
    if activation == "silu":
        out = F.silu(out)
    elif activation is not None:
        raise ValueError(f"unsupported conv activation {activation!r}.")
    return out.to(x.dtype)


def _history_taps(history: torch.Tensor, taps: torch.Tensor, span: int, kernel: int,
                  dilation: int, width: int) -> torch.Tensor:
    """算出「拼上历史」与「历史当零」在最前面 ``width`` 个输出位上的差。

    ``conv1d`` 是互相关：``out[i] = sum_t in[i + dilation*t - span] * w[t]``。对
    ``i < span``，那些 ``i + dilation*t < span`` 的抽头吃的是历史而不是零填充；
    本函数只把这些项单独算出来，其余位置两种写法逐位相同、不参与。

    Args:
        history: ``[B, C, span]``，本该拼在主卷积左侧的那段历史。
        taps: ``[C, kernel]``，depthwise 的抽头。
        span: 感受野跨度 ``dilation * (kernel - 1)``。
        kernel: 抽头数。
        dilation: 采样间隔。
        width: 要修正的输出位数目，调用方保证 ``width <= min(span, new_len)``。

    Returns:
        ``[B, C, width]`` 的修正量（与 ``history`` 同 dtype、同设备）。
    """
    pos_index, tap_index, src_index = _tap_indices(
        span, kernel, dilation, width, history.device)
    if pos_index.numel() == 0:
        return history.new_zeros(history.shape[0], history.shape[1], width)
    source = history.index_select(2, src_index)
    tap = taps.index_select(1, tap_index).unsqueeze(0)
    return history.new_zeros(
        history.shape[0], history.shape[1], width).index_add(2, pos_index, source * tap)


#: ``(span, kernel, dilation, width, device) -> (pos, tap, src)`` 三张下标表。
#:
#: 缓存的理由：它们只由形状决定，每次前向重建等于每个线性层各做 3 次 H2D 小拷贝
#: （prefill 512 是 36 层，合计 108 次）。用范围很小的表，条目数上界也只有几种形状。
_TAP_INDEX_CACHE: dict = {}
_TAP_INDEX_MAX = 16


def _tap_indices(span: int, kernel: int, dilation: int, width: int,
                 device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """取（必要时建）历史抽头的下标表。

    Args:
        span: 感受野跨度 ``dilation * (kernel - 1)``。
        kernel: 抽头数。
        dilation: 采样间隔。
        width: 要修正的输出位数目。
        device: 目标设备。

    Returns:
        ``(pos, tap, src)`` 三张一维 int64 下标表：输出位、抽头序号、历史列。
    """
    key = (span, kernel, dilation, width, device)
    cached = _TAP_INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    pairs = [(pos, t, pos + dilation * t)
             for pos in range(width) for t in range(kernel)
             if pos + dilation * t < span]
    tables = tuple(torch.tensor([item[axis] for item in pairs], dtype=torch.long,
                                device=device) for axis in range(3))
    if len(_TAP_INDEX_CACHE) >= _TAP_INDEX_MAX:
        _TAP_INDEX_CACHE.clear()
    _TAP_INDEX_CACHE[key] = tables
    return tables


__all__ = ["apply_padding_mask", "causal_depthwise_conv"]
