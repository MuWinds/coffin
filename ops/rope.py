"""Rotary position embeddings: the 1-D tables and the shared primitives.

Text-only mRoPE collapses to plain 1-D RoPE, so the tables use the simple
``cos = cat(cos_half, cos_half)`` layout. They are built once at load time as
``[max_seq, rotary_dim]`` and indexed by position. A prompt *with images* needs
three positions per token instead of one, and that path lives in
:mod:`ops.mrope` -- it reuses :func:`make_inv_freq` and the same
``cat(half, half)`` layout, so a text-only prompt through it is bit-identical to
a table row.

Module: ``coffin/ops``. Pure tensor math over ``torch``; called by the attention
modules in ``models/qwen35`` and by :mod:`ops.mrope`.

Constraints:
    * The ``cat(half, half)`` layout is a contract with ``rotate_half``, which
      pairs frequency ``i`` with ``i + rotary_dim/2``. A table laid out any other
      way rotates the wrong axes without any shape error.
    * Only the leading ``rotary_dim`` channels rotate; the trailing ``head_dim -
      rotary_dim`` channels of each head pass through untouched, so partial
      rotary checkpoints must keep that split.
    * Tables are built in fp32 and cast at the use site: bf16 table values lose
      enough angle precision at long context to change the logits.
"""

from __future__ import annotations

import torch


def make_inv_freq(head_dim: int, rotary_dim: int, theta: float, device) -> torch.Tensor:
    """Build the ``[rotary_dim/2]`` fp32 angle-per-position axis.

    The single source of the frequency axis: the 1-D tables below and the 3-D
    mRoPE tables in :mod:`ops.mrope` both start from here, which is what makes
    their cos/sin agree bit-for-bit on a text-only prompt. Diverging them would
    silently change the logits of every image-free request that took the mRoPE
    path.

    Args:
        head_dim: Attention head width; carried so the frequency count is read in
            the same units as the checkpoint config.
        rotary_dim: Rotated channels per head, the actual determinant of the
            table width.
        theta: RoPE base (``rope_theta`` from the checkpoint config).
        device: Torch device to build the tensor on.

    Returns:
        Angular frequencies, ``1/theta**(j/(rotary_dim/2))`` for even ``j``,
        ascending from ``1.0`` down towards ``1/theta``.
    """
    return 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32,
                                         device=device) / rotary_dim))


def make_rope_tables(head_dim: int, rotary_dim: int, theta: float,
                     max_seq: int, device,
                     inv_freq: torch.Tensor | None = None
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute the ``[max_seq, rotary_dim]`` cos/sin tables indexed by position.

    Args:
        head_dim: Attention head width, read for symmetry with the config.
        rotary_dim: Rotated channels per head.
        theta: RoPE base (``rope_theta`` from the checkpoint config).
        max_seq: Positions to tabulate; decoding past this index reads out of
            bounds, so it must cover the engine's maximum sequence length.
        device: Torch device to build the tables on.
        inv_freq: Optional pre-built frequency axis. Passing the mRoPE path's
            axis here is what keeps a text-only prompt bit-identical whichever
            path computes its cos/sin.

    Returns:
        ``(cos, sin)``, each ``[max_seq, rotary_dim]`` fp32 and duplicated across
        the two halves as ``rotate_half`` expects.
    """
    if inv_freq is None:
        inv_freq = make_inv_freq(head_dim, rotary_dim, theta, device)
    t = torch.arange(max_seq, dtype=torch.float32, device=device)
    f = t[:, None] * inv_freq[None, :]                     # [S, rotary_dim/2]
    c, s = torch.cos(f), torch.sin(f)
    return (torch.cat([c, c], -1).contiguous(), torch.cat([s, s], -1).contiguous())


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the two halves of a head against each other for RoPE.

    The second half is negated so ``x * cos + rotate_half(x) * sin`` reproduces
    the complex rotation the reference applies. It only lines up with the
    ``cat(half, half)`` table layout; with any other layout it rotates pairs of
    the wrong frequencies.

    Args:
        x: Tensor whose last axis is the full ``rotary_dim``, normally
            ``[..., S, heads, rotary_dim]``.

    Returns:
        Same shape as ``x``, rotated by half the last axis.
    """
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
               rotary_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate the leading ``rotary_dim`` channels of every q/k head.

    Applied to q and k in one call so the two can never drift into different
    tables: a mismatch there degrades attention scores without raising anything.

    Args:
        q: Query tensor, last axis a head of ``head_dim`` channels.
        k: Key tensor shaped like ``q``.
        cos: Cosine values, broadcastable to ``q[..., :rotary_dim]``.
        sin: Sine values, same layout as ``cos``.
        rotary_dim: Channels to rotate; the remaining ones pass through
            unchanged, which is what partial-rotary checkpoints rely on.

    Returns:
        ``(q, k)`` rotated and concatenated back to ``head_dim`` channels.
    """
    cos = cos.to(q.dtype)
    sin = sin.to(q.dtype)
    qr, qp = q[..., :rotary_dim], q[..., rotary_dim:]
    kr, kp = k[..., :rotary_dim], k[..., rotary_dim:]
    qr = qr * cos + rotate_half(qr) * sin
    kr = kr * cos + rotate_half(kr) * sin
    return torch.cat([qr, qp], dim=-1), torch.cat([kr, kp], dim=-1)
