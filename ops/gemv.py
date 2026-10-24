"""Decode-time weight reads: a ``[N, K]`` projection against one vector.

hipBLAS at M=1 reaches only 55-296 GB/s against a ~700 GB/s copy ceiling and
changes kernel per shape unpredictably, and Triton has no gfx906 backend, so
the hand-written kernel in ``hip_ops``/``kernels/gemv.cu`` is the default path
(540-680 GB/s measured).  ``GEMV`` benchmarks the candidate (rows-per-warp,
K-unroll) choices once per shape at load time -- which is why it is a class
with per-instance state -- and ``GEMVGroup`` fuses several projections that
share one input into a single wide GEMV.
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
        if hip_ops.available():
            return "hip", 1, 2
        return ("deq", 0, 0) if self.q is not None else ("wt", 0, 0)

    @staticmethod
    def _pick_m(w) -> dict[int, int]:
        """Best batched-GEMV unroll per M in 2/4/8; ``{}`` means "keep the default"."""
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
        """Benchmark every candidate for this shape; returns ``(method, r, u)``."""
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
        """``x`` [C, K] -> [C, N].

        Small C uses the batched GEMV, which reads each weight row once for all C
        tokens; hipBLAS has no tensor-core path here, so its small-M GEMM costs
        ~C decode steps. Large C uses hipBLAS, which is efficient once M is big --
        for int8 weights that means materialising fp16 weights for the call, since
        prefill is compute-bound and a dequantising GEMM kernel would buy nothing.

        The fp16 GEMM is worth the casts: rocBLAS reaches 13-20 TFLOP/s on fp16
        here (vs 2.3-7 on bf16, and 10 on fp32) because gfx906 has packed fp16
        math only.
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

    Reading one wide weight matrix beats several narrow ones (fewer launches, and
    the kernel is more efficient at large N), so e.g. ``in_proj_qkv/z/b/a`` or
    ``gate/up`` become a single call whose output is then split.
    """

    def __init__(self, ws: list, bench: bool = True) -> None:
        if isinstance(ws[0], QWeight):
            self.sizes = [w.n for w in ws]
            fused = QWeight(torch.cat([w.wq for w in ws], dim=0),
                            torch.cat([w.scale for w in ws], dim=0))
        else:
            self.sizes = [w.shape[0] for w in ws]
            fused = torch.cat([w.contiguous() for w in ws], dim=0)
        self.g = GEMV(fused, bench)

    def __call__(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.g(x).split(self.sizes, dim=0)

    def prefill(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.g.prefill(x).split(self.sizes, dim=1)


def _proj(sd: dict, prefixes: list[str], bench: bool):
    ws = [sd[p + ".weight"] for p in prefixes]
    return GEMV(ws[0], bench) if len(ws) == 1 else GEMVGroup(ws, bench)
