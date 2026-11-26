"""宿主侧（numpy）参考实现，供卡上自检对拍用。

Module: ``coffin/cxrt``。只 import ``cxrt.dtypes`` 与 numpy，**不 import torch**。

参考必须按内核的运算顺序写，而不是按数学定义写一遍：``conv`` 这条内核用
``fmaf`` 串起来、SiLU 用 ``__expf``，两者的舍入位置与 numpy 的 ``dot`` 不同。差别
只有 1–2 个 fp32 ULP，但**只有顺序一致**时才能说「对拍通过」是真的通过，而不是把
容差放大到看不出问题。
"""

from __future__ import annotations

import numpy as np

#: 因果卷积的抽头数。与 ``kernels/conv_device.inc`` 的 ``CONV_K`` 必须一致。
CONV_K = 4

#: 卷积窗口长度（``CONV_K - 1``）。与内核的 ``CONV_STATE`` 同一来源。
CONV_STATE = CONV_K - 1


def silu(values: np.ndarray) -> np.ndarray:
    """``x * sigmoid(x)``，fp32。

    Args:
        values: fp32 数组。

    Returns:
        同形状 fp32。
    """
    v = np.asarray(values, dtype=np.float32)
    return (v / (np.float32(1.0) + np.exp(-v))).astype(np.float32)


def conv_step(taps: np.ndarray, x_t: np.ndarray, hist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """走一个 token：返回 ``(y, 新窗口)``。

    Args:
        taps: ``[C, 4]`` fp32 抽头。
        x_t: ``[C]`` 当前 token（bf16 已还原成 fp32）。
        hist: ``[C, 3]`` 当前窗口，时间从左到右递增。

    Returns:
        ``(y [C], hist' [C, 3])``。
    """
    acc = (hist[:, 0] * taps[:, 0] + hist[:, 1] * taps[:, 1]
           + hist[:, 2] * taps[:, 2] + x_t * taps[:, 3]).astype(np.float32)
    y = silu(acc)
    hist_next = np.stack([hist[:, 1], hist[:, 2], x_t], axis=1).astype(np.float32)
    return y, hist_next


def conv1d_silu_seq(taps: np.ndarray, xs: np.ndarray,
                    hist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """整条序列走一遍卷积。

    预填充分块在真内核里是「从原始输入重放前 3 个 token 重建起点」，与本函数一趟
    走完逐位相同——因果卷积让分段不改变结果，只改变起点如何重建。

    Args:
        taps: ``[C, 4]`` fp32 抽头。
        xs: ``[T, C]`` 激活（fp32，值来自 bf16 存储）。
        hist: ``[C, 3]`` 起点窗口。

    Returns:
        ``(out [T, C], hist' [C, 3])``。
    """
    tokens, channels = xs.shape
    out = np.empty((tokens, channels), dtype=np.float32)
    h = np.array(hist, dtype=np.float32, copy=True)
    for t in range(tokens):
        out[t], h = conv_step(taps, xs[t], h)
    return out, h


def conv1d_silu_batch(taps: np.ndarray, xs: np.ndarray,
                      hists: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """每行一个窗口，所有行各走一个 token。

    Args:
        taps: ``[C, 4]`` fp32 抽头，所有行共用。
        xs: ``[B, C]`` 当前 token。
        hists: ``[B, C, 3]`` 各行窗口。

    Returns:
        ``(out [B, C], hists' [B, C, 3])``。
    """
    rows, channels = xs.shape
    out = np.empty((rows, channels), dtype=np.float32)
    h = np.array(hists, dtype=np.float32, copy=True)
    for r in range(rows):
        out[r], h[r] = conv_step(taps, xs[r], h[r])
    return out, h


def triangular_solve(lower: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """单位下三角前向替换，fp32 逐步累加（与内核的串行链一致）。

    Args:
        lower: ``[bn, L, L]`` 系数，只读严格下三角。
        rhs: ``[bn, L, n]`` 右端项。

    Returns:
        ``[bn, L, n]`` fp32。
    """
    bn, order, _ = lower.shape
    cols = rhs.shape[2]
    out = np.empty((bn, order, cols), dtype=np.float32)
    a = np.ascontiguousarray(lower, dtype=np.float32)
    b = np.ascontiguousarray(rhs, dtype=np.float32)
    for k in range(bn):
        acc = b[k].copy()
        for i in range(order):
            s = acc[i]
            for j in range(i):
                s = np.float32(s - np.float32(a[k, i, j] * acc[j]))
            acc[i] = s
        out[k] = acc
    return out
