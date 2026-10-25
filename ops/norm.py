"""RMSNorm in the two conventions this checkpoint uses.

``input_layernorm`` / ``post_attention_layernorm`` / ``q_norm`` / ``k_norm`` are
zero-centred (effective coefficient ``1 + w``, so the stored weights carry
negative values); ``linear_attn.norm`` (RMSNormGated) is the standard ``w * x``
times ``silu(gate)``. Mixing the two up garbles the output without raising
anything, which is why they are two classes and not one flag.

Module: ``coffin/ops``. Uses ``hip_ops`` for the fused kernels and falls back to
``torch.nn.functional`` when the extension is unavailable; called by every block
in ``models/qwen35``.

Constraints:
    * ``eps=1e-6`` is fixed by the checkpoint; changing it shifts the logits.
    * The HIP path only runs on contiguous input (RMSNorm also accepts a
      row-strided slice, e.g. ``q_norm`` on a transposed head tensor), so the
      stride and contiguity tests must stay part of the dispatch condition.
    * The zero-centred convention lives in ``RMSNorm`` only; never reuse its
      weight with the gated class.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import hip_ops
from ops.dtypes import D


class RMSNorm:
    """Zero-centred RMSNorm, ``y = norm(x) * (1 + w)`` (Qwen3.5 convention)."""

    def __init__(self, w: torch.Tensor) -> None:
        """Fold the checkpoint's offset convention and store bf16 weights.

        The ``1 +`` is applied once here rather than per call: the checkpoint
        stores zero-centred weights (they go negative), and adding the offset at
        load time keeps the hot path a single multiply.

        Args:
            w: Checkpoint weight, ``[hidden]``, in whatever precision it was
                saved in; the fold happens in fp32 to keep the offset exact.
        """
        self.g = (1.0 + w.float()).to(D).contiguous()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise and scale the last axis of ``x``.

        Args:
            x: Activations, any leading shape, ``[..., hidden]``. Non-bf16 input
                is cast first because every downstream kernel assumes ``D``.

        Returns:
            Same shape and bf16 dtype as the cast input.
        """
        if x.dtype != D:
            x = x.to(D)
        if hip_ops.available() and x.stride(-1) == 1:
            n = x.shape[-1]
            xf = x.reshape(-1, n)                 # rows may be strided (q_norm slice)
            return hip_ops.rms_norm(xf, self.g, xf.stride(0), 1e-6).view(x.shape)
        return F.rms_norm(x, (x.shape[-1],), self.g, 1e-6)


class RMSNormGated:
    """Standard RMSNorm times ``silu(gate)`` (Qwen3.5 linear-attn output norm)."""

    def __init__(self, w: torch.Tensor) -> None:
        """Store the weight as bf16, applying no offset.

        Args:
            w: Checkpoint weight, ``[hidden]``, already centred (unlike
                :class:`RMSNorm`, where the offset is folded in).
        """
        self.w = w.to(D).contiguous()

    def __call__(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Normalise ``x``, then scale it by the silu of a same-shape gate.

        Args:
            x: Activations, ``[..., hidden]``; cast to bf16 when needed.
            gate: Gate tensor shaped like ``x``. It is a separate projection, so
                the fused kernel needs both operands contiguous.

        Returns:
            Same shape as ``x``, normalised and gated.
        """
        if x.dtype != D:
            x = x.to(D)
        if hip_ops.available() and x.is_contiguous() and gate.is_contiguous():
            n = x.shape[-1]
            y = hip_ops.rms_norm_gated(x.reshape(-1, n), self.w,
                                       gate.reshape(-1, n), 1e-6)
            return y.view(x.shape)
        return F.rms_norm(x, (x.shape[-1],), self.w, 1e-6) * F.silu(gate)
