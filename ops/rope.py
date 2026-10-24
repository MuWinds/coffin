"""Rotary position embeddings: the 1-D tables and the shared primitives.

Text-only mRoPE collapses to plain 1-D RoPE, so the tables use the simple
``cos = cat(cos_half, cos_half)`` layout.  They are built once at load time as
``[max_seq, rotary_dim]`` and indexed by position; a prompt *with images* needs
three positions per token instead of one, and that path lives in
:mod:`ops.mrope` -- it reuses :func:`make_inv_freq` and the same
``cat(half, half)`` layout, so a text-only prompt through it is bit-identical to
a table row.
"""

from __future__ import annotations

import torch


def make_inv_freq(head_dim: int, rotary_dim: int, theta: float, device) -> torch.Tensor:
    """``[rotary_dim/2]`` fp32 angles-per-position, ``1/theta**(j/(rotary_dim/2))``.

    The single source of the frequency axis: the 1-D tables below and the 3-D
    mRoPE tables in :mod:`ops.mrope` both start from this, which is what makes
    their cos/sin agree bit-for-bit on a text-only prompt.
    """
    return 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32,
                                         device=device) / rotary_dim))


def make_rope_tables(head_dim: int, rotary_dim: int, theta: float,
                     max_seq: int, device,
                     inv_freq: torch.Tensor | None = None
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-computed cos/sin tables, ``[max_seq, rotary_dim]``.

    Text-only mrope collapses to 1-D rope, so ``cos = cat(cos_half, cos_half)``.
    ``inv_freq`` lets the caller share one frequency axis with the 3-D mRoPE path
    (:mod:`ops.mrope`), which is what keeps a text-only prompt bit-identical
    whichever path computes its cos/sin.
    """
    if inv_freq is None:
        inv_freq = make_inv_freq(head_dim, rotary_dim, theta, device)
    t = torch.arange(max_seq, dtype=torch.float32, device=device)
    f = t[:, None] * inv_freq[None, :]                     # [S, rotary_dim/2]
    c, s = torch.cos(f), torch.sin(f)
    return (torch.cat([c, c], -1).contiguous(), torch.cat([s, s], -1).contiguous())


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
               rotary_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.to(q.dtype)
    sin = sin.to(q.dtype)
    qr, qp = q[..., :rotary_dim], q[..., rotary_dim:]
    kr, kp = k[..., :rotary_dim], k[..., rotary_dim:]
    qr = qr * cos + rotate_half(qr) * sin
    kr = kr * cos + rotate_half(kr) * sin
    return torch.cat([qr, qp], dim=-1), torch.cat([kr, kp], dim=-1)
