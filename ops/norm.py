"""RMSNorm in the two conventions this checkpoint uses.

``input_layernorm`` / ``post_attention_layernorm`` / ``q_norm`` / ``k_norm`` are
zero-centred (effective coefficient ``1 + w``, so the stored weights carry
negative values); ``linear_attn.norm`` (RMSNormGated) is the standard ``w * x``
times ``silu(gate)``. Mixing the two up garbles the output without raising
anything, which is why they are two classes and not one flag.

Module: ``coffin/ops``. Binds the backend's ``rms_norm`` / ``add_rms_norm`` /
``rms_norm_gated`` entries and folds the checkpoint's weight convention; called by
every block in ``models/qwen35``.

Constraints:
    * ``eps=1e-6`` is fixed by the checkpoint; changing it shifts the logits.
    * The row pitch travels as an argument, not as a property of the tensor: the
      model normalises column slices of a wider fused buffer (``q_norm`` on a
      transposed head tensor), so a caller-side ``.contiguous()`` would cost a
      full copy of the widest activation in the layer. Only a non-unit *last*
      stride forces the copy, because the 2-D reshaping needs it.
    * ``torch.nn.functional.rms_norm`` must never appear here. It is a 2.4+ API
      and the deployment target runs torch 2.1.0, where it raises
      ``AttributeError``. The hand-written implementation lives in the reference
      backend, so the table is the single place the maths exists.
    * The zero-centred convention lives in ``RMSNorm`` only; never reuse its
      weight with the gated class.
    * ``add_residual`` accumulates into its ``x`` argument. That is the whole
      saving -- the residual stream stays in one buffer instead of being written
      and re-read -- and it is why the caller must own ``x`` exclusively. A tensor
      it shares with anything else is the caller's bug to prevent, because the
      kernel cannot tell an aliased residual from a private one.
"""

from __future__ import annotations

import torch

from backends import ops as bops
from ops.dtypes import D


def _rows(x: torch.Tensor) -> torch.Tensor:
    """View ``[..., n]`` as ``[M, n]`` without disturbing the row pitch.

    Args:
        x: Activations whose last axis is the normalised width. Any leading shape
            is allowed; the fused kernels index it as a flat batch of rows.

    Returns:
        A 2-D view when the last stride is unit, otherwise a contiguous copy
        viewed as 2-D -- a transposed operand has no legal ``[M, n]`` view, and
        copying is the cheaper of the two wrong answers. Rows of the returned
        tensor may still be pitched wider than ``n``; that pitch is what the
        backend's ``stride`` argument carries.
    """
    if x.stride(-1) != 1:
        x = x.contiguous()
    return x.reshape(-1, x.shape[-1])


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
        xf = _rows(x)
        return bops.rms_norm(xf, self.g, xf.stride(0), 1e-6).view(x.shape)

    def add_residual(self, x: torch.Tensor, add: torch.Tensor,
                     out: torch.Tensor) -> torch.Tensor:
        """Fold the residual add into this norm: ``x += add``, then normalise into ``out``.

        A decoder layer's pre-norm half is ``h = mix(norm(x)); x = x + h`` and then
        ``norm(x)`` again for the MLP. As two ops that is two launches and two reads
        of the residual row; at this card's 5.75 us launch floor the launch is the
        cost. Fusing them is the same arithmetic in one pass, so the token ids do not
        move -- the add is still a bf16 rounding of the sum, and the mean square is
        still taken over that rounded value.

        The residual is accumulated **in place** in ``x``. That is the buffer the
        layer keeps live: ``x`` is the residual stream and the next layer adds into it
        again, so writing the sum anywhere else would only move the copy.

        Args:
            x: Residual stream, any leading shape, ``[..., hidden]``, bf16.
                **Mutated**: it holds the sum on return. The caller must own it
                exclusively; a tensor it shares with a live value (the embedding
                output, a view another buffer still reads) is corrupted by this call.
            add: Mixer output to accumulate, shaped like ``x``. It is consumed before
                any output is written, so it is also free to be ``out`` -- which is
                how a layer reuses the mixer's own buffer instead of allocating one.
            out: Dense bf16 destination for the normalised sum, shaped like ``x``. It
                must not alias ``x``: the sum has to survive the norm.

        Returns:
            ``out``, for call chaining.

        Raises:
            TypeError: ``x`` or ``out`` is not bf16. The stream is accumulated in
                place, so there is no wider buffer to fall back to and casting ``x``
                here would strand the sum in a temporary.
            ValueError: ``out`` is not dense, or ``out`` is ``x``. Both would either
                write through a copy -- losing the result without an error -- or
                overwrite the residual.
        """
        if x.dtype != D or out.dtype != D:
            raise TypeError(f"add_residual accumulates {D} in place and writes {D}; "
                            f"got x={x.dtype}, out={out.dtype}")
        # A non-dense ``out`` would be reshaped into a copy and the kernel would write
        # into that copy, so the caller would get ``out`` unchanged and no error. This
        # is the one layout bug in this call that cannot be reported downstream.
        if out.stride(-1) != 1 or _rows(out).data_ptr() != out.data_ptr():
            raise ValueError("add_residual needs a dense destination; "
                             f"got strides {tuple(out.stride())}")
        if out is x or out.data_ptr() == x.data_ptr():
            raise ValueError("add_residual cannot write over the residual it "
                             "accumulates: the sum is still needed")
        if add.dtype != D:
            add = add.to(D)
        xf = _rows(x)
        bops.add_rms_norm(xf, _rows(add), self.g, _rows(out), xf.stride(0), 1e-6)
        return out


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

        The norm is per head, so the weight is ``[dv]`` and not the flattened
        width: flattening first would divide by the RMS of every head at once.

        Args:
            x: Activations, ``[..., dv]``; cast to bf16 when needed.
            gate: Gate tensor shaped like ``x``. It is a separate projection, so
                the fused kernel needs both operands flattened the same way.

        Returns:
            Same shape as ``x``, normalised and gated.
        """
        if x.dtype != D:
            x = x.to(D)
        if gate.dtype != D:
            gate = gate.to(D)
        return bops.rms_norm_gated(_rows(x), self.w, _rows(gate), 1e-6).view(x.shape)
