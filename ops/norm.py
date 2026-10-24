"""RMSNorm in the two conventions this checkpoint uses.

``input_layernorm`` / ``post_attention_layernorm`` / ``q_norm`` / ``k_norm``
are zero-centred (effective coefficient ``1 + w``, weights have negative
values); ``linear_attn.norm`` (RMSNormGated) is the standard ``w * x`` times
``silu(gate)``.  Mixing the two up garbles the output, which is why they are
two separate classes with the convention in the docstring.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import hip_ops
from ops.dtypes import D


class RMSNorm:
    """Zero-centred RMSNorm: ``y = norm(x) * (1 + w)`` (Qwen3.5 convention)."""

    def __init__(self, w: torch.Tensor) -> None:
        self.g = (1.0 + w.float()).to(D).contiguous()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
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
        self.w = w.to(D).contiguous()

    def __call__(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if x.dtype != D:
            x = x.to(D)
        if hip_ops.available() and x.is_contiguous() and gate.is_contiguous():
            n = x.shape[-1]
            y = hip_ops.rms_norm_gated(x.reshape(-1, n), self.w,
                                       gate.reshape(-1, n), 1e-6)
            return y.view(x.shape)
        return F.rms_norm(x, (x.shape[-1],), self.w, 1e-6) * F.silu(gate)
