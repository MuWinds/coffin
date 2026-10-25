"""Decode-time weight reads: a ``[N, K]`` projection against a few activations.

hipBLAS at M=1 reaches only 55-296 GB/s against a ~700 GB/s copy ceiling and
changes kernel per shape unpredictably, and Triton has no gfx906 backend, so the
hand-written kernel in ``hip_ops``/``kernels/gemv.cu`` is the default path
(540-680 GB/s measured). :class:`GEMV` benchmarks the candidate
(rows-per-warp, K-unroll) choices once per shape at load time -- which is why it
is a class with per-instance state -- and :class:`GEMVGroup` fuses several
projections that share one input into a single wide GEMV.

Module: ``coffin/ops``. Depends on ``hip_ops`` and ``int8_weights``; every
projection in ``models/qwen35`` goes through it, so a kernel choice made here
sets the decode cost of the whole model.

Constraints:
    * Weights are stored row-major ``[N, K]`` -- the layout the checkpoint ships
      and the one ``hip_ops`` kernels index. A caller handing over ``[K, N]``
      would produce plausible but wrong numbers, so never transpose upstream.
    * int8 weights are weight-only: activations stay bf16 (``D``), and each row
      is scaled by its own scale. The non-HIP fallbacks have to cast to the
      weight dtype and back; the HIP kernels do the promotion in-kernel.
    * Kernel choice is per instance and measured on the live shape, so never
      hardcode ``(r, u)`` -- the fastest config changes with N, K and M on this
      part.
    * ``bench=False`` must stay usable without a GPU: structural tests build
      these objects in a container with no checkpoint.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import hip_ops
from int8_weights import QWeight
from ops.dtypes import D, WF


class GEMV:
    """A ``[N, K]`` weight evaluated as a vector-matrix product at decode time.

    ``w`` is either a bf16 tensor or an :class:`int8_weights.QWeight` (weight-only
    int8). Either way the (rows-per-warp, K-unroll) kernel choice is benchmarked
    once per shape, because which one wins changes with the shape on this part.
    """

    #: (rows-per-warp, K-unroll) candidates handed to the per-shape autotuner.
    #: The kernels also instantiate (1,4) and (4,1); leaving them out meant the
    #: tuner could never pick them, which cost a few percent on the short-K and
    #: wide-N shapes.
    _HIP_CFGS = ((1, 1), (2, 1), (1, 2), (2, 2), (1, 4), (4, 1))

    #: U candidates the *batched* small-M kernel instantiates, per M.
    _M_US = {2: (1, 2, 4), 4: (1, 2, 4), 8: (1, 2)}

    def __init__(self, w, bench: bool = True) -> None:
        """Pick a kernel for this weight's shape and keep the form it needs.

        The chosen ``method`` decides what is retained: the HIP path keeps the
        packed/original weight, the transpose path keeps a pre-transposed copy,
        and the dequantise path keeps nothing. Holding all of them would add a
        duplicate of the largest weights in the model.

        Args:
            w: Projection weight, row-major ``[N, K]``, plain tensor or
                :class:`QWeight`.
            bench: Measure every candidate once. ``False`` takes the default
                config, which skips the per-shape sweep for structural tests and
                for loads where the extension is absent.
        """
        self.q = w if isinstance(w, QWeight) else None
        self.N, self.K = (self.q.n, self.q.k) if self.q is not None else w.shape
        self.method, self.r, self.u = self._pick(w) if bench else self._default()
        if hip_ops.available() and self.method == "hip":
            # The batched path (verify, and prefill at C in 2/4/8) used to hardcode
            # U=2 for every shape.  Verify replays it 103 times per step, so it is
            # worth one short sweep per M.
            self.mu = self._pick_m(w) if bench else {}
        else:
            self.mu = {}
        if self.method == "hip":
            self.w, self.wt = (None if self.q is not None else w.contiguous()), None
        elif self.method in ("deq", "mv", "lin"):
            self.w, self.wt = (None if self.method == "deq" else w), None
        else:                                   # "wt": keep a pre-transposed copy
            self.w, self.wt = None, w.t().contiguous()

    def _default(self) -> tuple[str, int, int]:
        """Kernel choice when the autotuner is skipped, as ``(method, r, u)``.

        Returns:
            The HIP config when the extension is loaded, otherwise the cheapest
            pure-torch form for the storage type: dequantise-and-GEMM for int8,
            a pre-transposed matmul for plain weights.
        """
        if hip_ops.available():
            return "hip", 1, 2
        return ("deq", 0, 0) if self.q is not None else ("wt", 0, 0)

    @staticmethod
    def _pick_m(w) -> dict[int, int]:
        """Best batched-GEMV unroll per M in 2/4/8; ``{}`` means "keep the default".

        Args:
            w: Projection weight, row-major ``[N, K]``, plain tensor or
                :class:`QWeight`.

        Returns:
            Mapping ``M -> unroll``. A shape whose candidates all failed to run is
            absent, and the caller then falls back to the default unroll.
        """
        q = w if isinstance(w, QWeight) else None
        K = q.k if q is not None else w.shape[1]
        wc = None if q is not None else w.contiguous()
        out: dict[int, int] = {}
        for M, us in GEMV._M_US.items():
            x = torch.randn(M, K, dtype=D, device="cuda")
            cands = {}
            for u in us:
                if q is not None:
                    cands[u] = lambda u=u: hip_ops.gemv_w8a16_m(q.wq, q.scale, x, u)
                else:
                    cands[u] = lambda u=u: hip_ops.gemv_m(wc, x, u)
            best, best_ms = 2, float("inf")
            for u, fn in cands.items():
                try:
                    for _ in range(3):
                        fn()
                    torch.cuda.synchronize()
                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    s.record()
                    for _ in range(20):
                        fn()
                    e.record()
                    torch.cuda.synchronize()
                    ms = s.elapsed_time(e) / 20
                except Exception:
                    continue
                if ms < best_ms:
                    best, best_ms = u, ms
            out[M] = best
        return out

    @staticmethod
    def _pick(w) -> tuple[str, int, int]:
        """Benchmark every candidate for this shape; returns ``(method, r, u)``.

        Args:
            w: Projection weight, row-major ``[N, K]``, plain tensor or
                :class:`QWeight`.

        Returns:
            ``(method, rows_per_warp, k_unroll)``. The rows/unroll fields are
            ``0`` for the pure-torch methods, which have no such knob.

        Raises:
            Nothing: a candidate that raises (missing extension symbol, unsupported
            dtype) is skipped, so the fallback ordering is the error handling.
        """
        q = w if isinstance(w, QWeight) else None
        K = q.k if q is not None else w.shape[1]
        x = torch.randn(K, dtype=D, device="cuda")
        # The weight is fp16 while the activations are bf16, so every non-HIP
        # candidate has to cast on the way in and back on the way out.
        xw = x if q is not None else x.to(w.dtype)
        wt = None
        if q is not None:
            cands = {"deq": lambda: F.linear(x, q.dequant())}
        else:
            wt = w.t().contiguous()
            cands = {"wt": lambda: (xw @ wt).to(D),
                     "lin": lambda: F.linear(xw, w).to(D),
                     "mv": lambda: torch.mv(w, xw).to(D)}
        if hip_ops.available():
            if q is not None:
                wc, sc = q.wq, q.scale
                for r, u in GEMV._HIP_CFGS:
                    cands[f"hip{r}_{u}"] = (
                        lambda r=r, u=u: hip_ops.gemv_w8a16(wc, sc, x, r, u))
            else:
                wc = w.contiguous()
                for r, u in GEMV._HIP_CFGS:
                    cands[f"hip{r}_{u}"] = (lambda r=r, u=u: hip_ops.gemv(wc, x, r, u))
        best, best_ms = (("deq", 0, 0) if q is not None else ("wt", 0, 0)), float("inf")
        for name, fn in cands.items():
            try:
                for _ in range(5):
                    fn()
                torch.cuda.synchronize()
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                for _ in range(40):
                    fn()
                e.record()
                torch.cuda.synchronize()
                ms = s.elapsed_time(e) / 40
            except Exception:
                continue
            if ms < best_ms:
                if name.startswith("hip"):
                    r, u = (int(v) for v in name[3:].split("_"))
                    best = ("hip", r, u)
                else:
                    best = (name, 0, 0)
                best_ms = ms
        del wt, x, cands
        return best

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Project one or a few activation rows, ``[K]`` / ``[C, K]`` -> ``[N]`` / ``[C, N]``.

        Args:
            x: Activations, ``[K]`` or ``[C, K]``, bf16.

        Returns:
            Projected activations in the input's dtype and leading shape.
        """
        m = self.method
        if m == "hip":
            if self.q is not None:
                return hip_ops.gemv_w8a16(self.q.wq, self.q.scale, x, self.r, self.u)
            return hip_ops.gemv(self.w, x, self.r, self.u)
        if m == "deq":
            return F.linear(x, self.q.dequant()).to(x.dtype)
        # The fallbacks below are dtype-blind and read bf16 activations, so they
        # cast to the weight dtype and back (the HIP kernels do it natively).
        if m == "lin":
            return F.linear(x.to(self.w.dtype), self.w).to(x.dtype)
        if m == "mv":
            return torch.mv(self.w, x.to(self.w.dtype)).to(x.dtype)
        return (x.to(self.wt.dtype) @ self.wt).to(x.dtype)

    def prefill(self, x: torch.Tensor) -> torch.Tensor:
        """Project a prefill chunk, ``[C, K]`` -> ``[C, N]``.

        Small C uses the batched GEMV, which reads each weight row once for all C
        tokens; hipBLAS has no tensor-core path here, so its small-M GEMM costs
        ~C decode steps. Large C uses hipBLAS, which is efficient once M is big --
        for int8 weights that means materialising fp16 weights for the call, since
        prefill is compute-bound and a dequantising GEMM kernel would buy nothing.

        The fp16 GEMM is worth the casts: rocBLAS reaches 13-20 TFLOP/s on fp16
        here (vs 2.3-7 on bf16, and 10 on fp32) because gfx906 has packed fp16
        math only.

        Args:
            x: Activations, ``[C, K]``, bf16.

        Returns:
            Projected activations, ``[C, N]``, bf16.
        """
        C = x.shape[0]
        if hip_ops.available() and self.method == "hip" and C in (2, 4, 8):
            u = self.mu.get(C, 2)
            if self.q is not None:
                return hip_ops.gemv_w8a16_m(self.q.wq, self.q.scale, x.contiguous(), u)
            return hip_ops.gemv_m(self.w, x.contiguous(), u)
        if self.q is not None:
            return F.linear(x.to(WF), self.q.dequant(WF)).to(x.dtype)
        if self.method == "wt":
            return (x.to(self.wt.dtype) @ self.wt).to(x.dtype)
        if self.method == "mv":
            return (x.to(self.w.dtype) @ self.w.t()).to(x.dtype)
        return F.linear(x.to(self.w.dtype), self.w).to(x.dtype)

class GEMVGroup:
    """Several ``[N_i, K]`` projections with the *same* input, fused into one GEMV.

    Reading one wide weight matrix beats several narrow ones: fewer launches, and
    the kernel is more efficient at large N, so e.g. ``in_proj_qkv/z/b/a`` or
    ``gate/up`` become a single call whose output is then split.
    """

    def __init__(self, ws: list, bench: bool = True) -> None:
        """Concatenate the weights along the output axis and build the fused GEMV.

        Args:
            ws: Projections sharing one input, each row-major ``[N_i, K]``. All
                must be the same storage class (all int8 or all plain) and share
                K, which the concatenation enforces with a shape error.
            bench: Forwarded to :class:`GEMV`.
        """
        if isinstance(ws[0], QWeight):
            self.sizes = [w.n for w in ws]
            fused = QWeight(torch.cat([w.wq for w in ws], dim=0),
                            torch.cat([w.scale for w in ws], dim=0))
        else:
            self.sizes = [w.shape[0] for w in ws]
            fused = torch.cat([w.contiguous() for w in ws], dim=0)
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
    autotune path; a lone prefix skips the fusion that would only add a split.

    Args:
        sd: Checkpoint state dict keyed by parameter name.
        prefixes: Parameter prefixes without the ``.weight`` suffix, in the order
            the caller will unpack the result.
        bench: Forwarded to the autotuner.

    Returns:
        A :class:`GEMV` for a single prefix, otherwise a :class:`GEMVGroup`.
    """
    ws = [sd[p + ".weight"] for p in prefixes]
    return GEMV(ws[0], bench) if len(ws) == 1 else GEMVGroup(ws, bench)
