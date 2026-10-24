"""Rotary position embeddings, 1-D for now.

Text-only mRoPE collapses to plain 1-D RoPE, so the tables use the simple
``cos = cat(cos_half, cos_half)`` layout.  They are built once at load time as
``[max_seq, rotary_dim]`` and indexed by position; when multimodal lands, the
position interface (not the table layout) is what changes.
"""

from __future__ import annotations

import torch


def make_rope_tables(head_dim: int, rotary_dim: int, theta: float,
                     max_seq: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-computed cos/sin tables, ``[max_seq, rotary_dim]``.

    Text-only mrope collapses to 1-D rope, so ``cos = cat(cos_half, cos_half)``.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim))
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
