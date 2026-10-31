"""旋转位置编码：一维表以及共用的原语。

纯文本的 mRoPE 退化成普通一维 RoPE，所以表就是简单的
``cos = cat(cos_half, cos_half)`` 布局。它们在加载时一次性建好，形状
``[max_seq, rotary_dim]``，按位置索引。*带图像*的 prompt 每个 token 要三个
位置而不是一个，那条路径在 :mod:`ops.mrope`——它复用
:func:`make_inv_freq` 和同样的 ``cat(half, half)`` 布局，所以纯文本 prompt
走那条路得到的 cos/sin 与查表逐位相同。

Module: ``coffin/ops``。基于 ``torch`` 的纯张量计算；由 ``models/qwen35`` 下的
注意力模块和 :mod:`ops.mrope` 调用。

Constraints:
    * ``cat(half, half)`` 布局是与 ``rotate_half`` 的契约，后者把频率 ``i`` 与
      ``i + rotary_dim/2`` 配对。换成任何别的布局都会转动错误的轴，而且形状
      检查不会报错。
    * 只有前 ``rotary_dim`` 个通道旋转；每个头尾部 ``head_dim -
      rotary_dim`` 个通道原样通过，所以部分旋转的 checkpoint 必须保持这个切分。
    * 表用 fp32 构建、在使用处转换：长上下文下 bf16 的表值损失的角度精度
      足以改变 logits。
"""

from __future__ import annotations

import torch


def make_inv_freq(head_dim: int, rotary_dim: int, theta: float, device) -> torch.Tensor:
    """构建 ``[rotary_dim/2]`` 的 fp32 每位置角度轴。

    频率轴只有这一个来源：下面的一维表和 :mod:`ops.mrope` 里的三维 mRoPE 表
    都从这里出发，纯文本 prompt 上它们的 cos/sin 才能逐位一致。两者一旦分家，
    每个走 mRoPE 路径的无图请求的 logits 都会被悄悄改掉。

    Args:
        head_dim: 注意力头宽度；带着它是为了让频率个数和 checkpoint 配置按同一
            单位读取。
        rotary_dim: 每个头被旋转的通道数，真正决定表宽的是它。
        theta: RoPE 底数（来自 checkpoint 配置的 ``rope_theta``）。
        device: 构建张量所在的 torch 设备。

    Returns:
        角频率，偶数 ``j`` 对应 ``1/theta**(j/(rotary_dim/2))``，从 ``1.0``
        向 ``1/theta`` 递减。
    """
    return 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32,
                                         device=device) / rotary_dim))


def make_rope_tables(head_dim: int, rotary_dim: int, theta: float,
                     max_seq: int, device,
                     inv_freq: torch.Tensor | None = None
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """预先算好按位置索引的 ``[max_seq, rotary_dim]`` cos/sin 表。

    Args:
        head_dim: 注意力头宽度，读取它是为了和配置保持对称。
        rotary_dim: 每个头被旋转的通道数。
        theta: RoPE 底数（来自 checkpoint 配置的 ``rope_theta``）。
        max_seq: 制表的位置数；解码越过这个上标就会读越界，所以它必须覆盖
            引擎的最大序列长度。
        device: 构建表的 torch 设备。
        inv_freq: 可选的现成频率轴。把 mRoPE 路径的频率轴传进来，纯文本
            prompt 无论由哪条路径计算 cos/sin 都逐位相同。

    Returns:
        ``(cos, sin)``，各自 ``[max_seq, rotary_dim]`` 的 fp32，并按
        ``rotate_half`` 的期望在两个半区重复一份。
    """
    if inv_freq is None:
        inv_freq = make_inv_freq(head_dim, rotary_dim, theta, device)
    t = torch.arange(max_seq, dtype=torch.float32, device=device)
    f = t[:, None] * inv_freq[None, :]                     # [S, rotary_dim/2]
    c, s = torch.cos(f), torch.sin(f)
    return (torch.cat([c, c], -1).contiguous(), torch.cat([s, s], -1).contiguous())


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """把头部的前后两半相对旋转，供 RoPE 使用。

    后半区取负，这样 ``x * cos + rotate_half(x) * sin`` 就能复现参考实现施加的
    复数旋转。它只和 ``cat(half, half)`` 的表布局对得上；换任何别的布局，
    配对的就是错误频率的两半。

    Args:
        x: 最后一维是完整 ``rotary_dim`` 的张量，通常是
            ``[..., S, heads, rotary_dim]``。

    Returns:
        与 ``x`` 同形状，按最后一维的一半完成旋转。
    """
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
               rotary_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """旋转每个 q/k 头最前面的 ``rotary_dim`` 个通道。

    q 和 k 在一次调用里一起处理，两者就不可能漂到不同的表上去：那里一旦
    不匹配，注意力分数会退化，却什么都不会抛出。

    Args:
        q: 查询张量，最后一维是一个 ``head_dim`` 通道的头。
        k: 与 ``q`` 同形状的键张量。
        cos: 余弦值，可广播到 ``q[..., :rotary_dim]``。
        sin: 正弦值，布局与 ``cos`` 相同。
        rotary_dim: 要旋转的通道数；其余通道原样通过，部分旋转的 checkpoint
            依赖的正是这一点。

    Returns:
        ``(q, k)``，旋转后拼回 ``head_dim`` 个通道。
    """
    cos = cos.to(q.dtype)
    sin = sin.to(q.dtype)
    qr, qp = q[..., :rotary_dim], q[..., rotary_dim:]
    kr, kp = k[..., :rotary_dim], k[..., rotary_dim:]
    qr = qr * cos + rotate_half(qr) * sin
    kr = kr * cos + rotate_half(kr) * sin
    return torch.cat([qr, qp], dim=-1), torch.cat([kr, kp], dim=-1)
