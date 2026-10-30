"""Decode-time weight reads: a ``[N, K]`` projection against a few activations.

What used to be here was a kernel chooser: six (rows-per-warp, K-unroll)
candidates, a device-timing sweep per weight shape, and four pure-torch fallback
spellings of the same matmul. All of that is a fact about one card's kernels, so
it now lives in the backend behind three semantic entries -- ``linear`` /
``linear_m`` / ``linear_prefill`` -- and this file only decides *which regime a
call is in*. :class:`GEMV` keeps its name and its per-instance weight because ~200
call sites construct one at load time; :class:`GEMVGroup` still concatenates
projections that share one input, because one wide weight read beats several
narrow ones whatever the platform.

Module: ``coffin/ops``. Depends on ``backends.ops`` for the actual kernels and
``int8_weights`` for the quantised storage form; every projection in
``models/qwen35`` goes through it.

Constraints:
    * Weights stay row-major ``[N, K]`` -- the layout the checkpoint ships and the
      one the contract takes. A backend that wants ``[K, N]`` transposes at load
      time behind ``caps.weight_layout``; transposing here would produce plausible
      but wrong numbers on every backend that does not.
    * Nothing in this file may call a timing primitive, read a device API or
      branch on whether a kernel exists. Whether a shape gets an autotuned kernel
      is ``caps.autotune``, decided inside the backend; the table is total, so a
      call site never asks.
    * int8 weights are weight-only: activations stay ``D`` and each row is scaled
      by its own scale. The dispatch on :class:`int8_weights.QWeight` happens
      inside the backend, so the caller passes whichever storage the loader
      produced.
    * ``bench=False`` must stay usable without a GPU: structural tests build these
      objects in a container with no checkpoint and no device.
"""

from __future__ import annotations

import torch

from backends import ops as bops
from int8_weights import QWeight


class GEMV:
    """A ``[N, K]`` weight evaluated as a vector-matrix product at decode time.

    ``w`` is either a bf16/fp16 tensor or an :class:`int8_weights.QWeight`
    (weight-only int8). Both forms are handed to the backend unchanged: the
    quantisation format is model-side storage, not a platform capability.
    """

    def __init__(self, w, bench: bool = True) -> None:
        """Keep the projection weight and record its shape.

        Args:
            w: Projection weight, row-major ``[N, K]``, plain tensor or
                :class:`QWeight`.
            bench: Accepted for call-site compatibility and otherwise unused. The
                per-shape kernel sweep moved to the backend, where it is gated by
                ``caps.autotune`` and cached by shape; a model-side flag could not
                reach it. The parameter stays until every caller (``_proj``,
                ``TiedEmbedding``, the model constructors) drops it, and removing
                it there is a pure signature change.
        """
        self.q = w if isinstance(w, QWeight) else None
        self.N, self.K = (self.q.n, self.q.k) if self.q is not None else w.shape
        #: The weight in the form the op table takes it: the plain tensor, or the
        #: ``QWeight`` carrying its packed buffer and scale. The table dispatches
        #: on the type, so this is the one place the two storage forms are
        #: re-united and the call sites never see the difference.
        self.w = w

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Project one or a few activation rows, ``[K]`` / ``[C, K]`` -> ``[N]`` / ``[C, N]``.

        Args:
            x: Activations, ``[K]`` or ``[C, K]``, bf16.

        Returns:
            Projected activations in the input's dtype and leading shape.
        """
        return bops.linear(self.w, x)

    def linear_m(self, x: torch.Tensor) -> torch.Tensor:
        """Read the weight once for every row of a small batch, ``[M, K]`` -> ``[M, N]``.

        Args:
            x: Activations, ``[M, K]``, bf16, ``M > 1``.

        Returns:
            ``[M, N]`` in ``x``'s dtype. The backend picks the tiling and which
            ``M`` it specialises, so this is the entry the batched decode and the
            speculative verify use instead of looping over :meth:`__call__`.
        """
        return bops.linear_m(self.w, x)

    def prefill(self, x: torch.Tensor) -> torch.Tensor:
        """Project a prefill chunk, ``[C, K]`` -> ``[C, N]``.

        Small ``C`` goes through :meth:`linear_m` because that is the regime where
        one weight pass for all rows beats a GEMM: the verify path lands here with
        ``C <= 8``, and it re-reads the weight once instead of ``C`` times. ``8``
        is the widest verify window the model captures for (``CKPT``), so the
        bound is a model fact rather than a kernel tiling parameter. Above it the
        GEMM wins on every backend measured so far, and an int8 weight is
        materialised at the caps weight dtype for the call -- prefill is
        compute-bound, so that trade is the right one there.

        Args:
            x: Activations, ``[C, K]``, bf16.

        Returns:
            Projected activations, ``[C, N]``, bf16.
        """
        if 2 <= x.shape[0] <= 8:
            return bops.linear_m(self.w, x)
        return bops.linear_prefill(self.w, x)


class GEMVGroup:
    """Several ``[N_i, K]`` projections with the *same* input, fused into one GEMV.

    Reading one wide weight matrix beats several narrow ones: fewer launches, and
    the read is more efficient at large N, so e.g. ``in_proj_qkv/z/b/a`` or
    ``gate/up`` become a single call whose output is then split. That is true on
    every platform, which is why the concatenation stays here rather than moving
    into the backend.
    """

    def __init__(self, ws: list, bench: bool = True) -> None:
        """Concatenate the weights along the output axis and build the fused GEMV.

        The fused matrix is allocated once and filled in place rather than built with
        ``torch.cat``. ``cat`` needs its output and both inputs alive at the same time,
        so on a 27B checkpoint each layer's ``gate``/``up`` pair momentarily costs three
        copies of itself; sixty-four of those transients left the caching allocator
        holding 4.2 GiB of *reserved but unallocated* segments that it could not split
        into the next 170 MiB request, and the build died with 27.0 GiB live on a
        31.7 GiB card. Copying into a pre-allocated destination keeps it at one extra
        copy per layer and lets the source be dropped immediately after.

        Args:
            ws: Projections sharing one input, each row-major ``[N_i, K]``. All
                must be the same storage class (all int8 or all plain) and share
                K, which the concatenation enforces with a shape error.
            bench: Forwarded to :class:`GEMV`.
        """
        if isinstance(ws[0], QWeight):
            self.sizes = [w.n for w in ws]
            k = ws[0].k
            total = sum(self.sizes)
            wq = torch.empty((total, k), dtype=ws[0].wq.dtype, device=ws[0].wq.device)
            scale = torch.empty(total, dtype=ws[0].scale.dtype, device=ws[0].scale.device)
            row = 0
            for w in ws:
                wq[row:row + w.n].copy_(w.wq)
                scale[row:row + w.n].copy_(w.scale)
                row += w.n
            fused = QWeight(wq, scale)
        else:
            self.sizes = [w.shape[0] for w in ws]
            first = ws[0]
            total = sum(self.sizes)
            buf = torch.empty((total, first.shape[1]), dtype=first.dtype,
                              device=first.device)
            row = 0
            for w in ws:
                buf[row:row + w.shape[0]].copy_(w)
                row += w.shape[0]
            fused = buf
        self.g = GEMV(fused, bench)

    def __call__(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Project the shared input and split the result back per weight.

        Args:
            x: Activations, ``[K]`` or ``[C, K]``, bf16.

        Returns:
            One tensor per input weight, in the order given, split along the
            output axis to their original sizes.
        """
        return self.g(x).split(self.sizes, dim=0)

    def linear_m(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Batched variant of :meth:`__call__`, splitting along the output axis.

        Args:
            x: Activations, ``[M, K]``, bf16.

        Returns:
            One ``[M, N_i]`` tensor per input weight, in the order given.
        """
        return self.g.linear_m(x).split(self.sizes, dim=1)

    def prefill(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Prefill-chunk variant of :meth:`__call__`, splitting along the output axis.

        Args:
            x: Activations, ``[C, K]``, bf16.

        Returns:
            One ``[C, N_i]`` tensor per input weight, in the order given.
        """
        return self.g.prefill(x).split(self.sizes, dim=1)


def _proj(sd: dict, prefixes: list[str], bench: bool):
    """Build the projection for one or more weight prefixes of a checkpoint.

    Sharing one helper keeps the single- and multi-prefix cases on the same
    construction path; a lone prefix skips the fusion that would only add a split.

    Args:
        sd: Checkpoint state dict keyed by parameter name.
        prefixes: Parameter prefixes without the ``.weight`` suffix, in the order
            the caller will unpack the result.
        bench: Forwarded to :class:`GEMV`; now inert, see its ``__init__``.

    Returns:
        A :class:`GEMV` for a single prefix, otherwise a :class:`GEMVGroup`.
    """
    ws = [sd[p + ".weight"] for p in prefixes]
    return GEMV(ws[0], bench) if len(ws) == 1 else GEMVGroup(ws, bench)
