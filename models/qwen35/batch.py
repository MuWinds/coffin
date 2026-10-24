"""Batched weight reads for the static-batch decode path.

Stage 1 batches exactly one thing: the projections.  A decode step is bound by
reading the weights (4 GB of int8 per step, ~700 GB/s), so applying one weight
row to B input vectors costs almost the same as applying it to one -- that is
the whole win of a static batch.

``GEMV.prefill`` already has the small-M path (``gemv_m``), but it indexes it by
the autotuned per-M unroll.  For the batched decode we instead replay the
unroll the *single-stream* kernel chose for that shape, because
``gemv_kernel`` and ``gemv_m_kernel`` accumulate each output row in exactly the
same order for the same ``U`` (same per-lane 8 accumulators, same
``32*U`` K-stride, same warp reduction).  Matching ``U`` therefore keeps every
row bit-identical to a solo decode -- no bf16 rounding drift, which a different
unroll would introduce at a handful of values.  The M=8 kernel tops out at U=2,
so a shape whose single-stream unroll is U=4 is run as two M=4 calls (U=4
exists there); that only halves the weight-read saving for those projections
instead of changing the numbers.
"""

from __future__ import annotations

import os

import torch

import hip_ops
from ops.gemv import GEMV, GEMVGroup

#: Unrolls the batched kernel instantiates per M (keep in sync with gemv.cu).
_M_US = {2: (1, 2, 4), 4: (1, 2, 4), 8: (1, 2)}

#: Opt-in faster projection: ``Q35_FAST_GEMV=1`` routes the batched projection
#: through ``gemv_w8a16_wide_dot2`` (packed-fp16 ``v_dot2`` math, one weight pass
#: for all rows).  Measured 1.18x the M=8 projection segment on this card.  It is
#: *not* bit-identical to the single-stream kernel (bf16 output differs by 1-2 ulp,
#: ~0.02% of elements), so it stays off unless asked for; the default path is
#: untouched.  ``serve --fast-gemv`` sets this variable before the model is built.
_FAST_GEMV = os.environ.get("Q35_FAST_GEMV") == "1"


def _fast_project(proj: GEMV, x: torch.Tensor) -> torch.Tensor | None:
    """dot2 projection for the batched decode widths, or None when unavailable."""
    if not (_FAST_GEMV and hip_ops.available() and proj.method == "hip"
            and proj.q is not None and 2 <= x.shape[0] <= 32
            and x.shape[1] == proj.K):
        return None
    fn = getattr(hip_ops, "gemv_w8a16_wide_dot2", None)
    if fn is None:
        return None
    u = proj.u if proj.u in (2, 4) else 2
    return fn(proj.q.wq, proj.q.scale, x.contiguous(), u, 0)


def _project(proj: GEMV, x: torch.Tensor) -> torch.Tensor:
    """``x`` [M, K] bf16 -> [M, N] bf16, bit-identical per row to ``proj(x[r])``."""
    M = x.shape[0]
    if M == 1:
        # Bucket 1 is a one-row graph: use the plain single-stream GEMV (the
        # batched kernel has no M=1 instantiation) and keep the row dim.
        return proj(x[0]).unsqueeze(0)
    fast = _fast_project(proj, x)          # opt-in (Q35_FAST_GEMV=1)
    if fast is not None:
        return fast
    if not (hip_ops.available() and proj.method == "hip" and M in (2, 4, 8)
            and x.shape[1] == proj.K):
        return proj.prefill(x)               # CPU / non-HIP fallback
    u = proj.u
    if u not in _M_US[M]:
        if M == 8 and u == 4:
            # No M=8/U=4 instantiation: two M=4/U=4 calls keep the exact sums.
            return torch.cat([_project(proj, x[:4]), _project(proj, x[4:])], dim=0)
        u = 2
    x = x.contiguous()
    if proj.q is not None:
        return hip_ops.gemv_w8a16_m(proj.q.wq, proj.q.scale, x, u)
    return hip_ops.gemv_m(proj.w, x, u)


def batch_proj(proj, x: torch.Tensor):
    """Batched projection: ``[M, K] -> [M, N]`` (or a list for a fused group).

    ``proj`` is a :class:`~ops.gemv.GEMV`, a :class:`~ops.gemv.GEMVGroup` (the
    fused ``qkv`` / ``in_proj_qkv/z/b/a`` / ``gate_up`` reads) or a
    :class:`~ops.embed.TiedEmbedding`'s inner GEMV.
    """
    if isinstance(proj, GEMVGroup):
        return _project(proj.g, x).split(proj.sizes, dim=1)
    return _project(proj, x)


#: Names the task-12 wide-M kernels are expected to expose.  Probe with
#: ``hasattr`` so a tree with only the old kernels keeps working (and so the
#: one-forward verify can fall back to the prefill/GEMM path until they land).
_WIDE_W8 = ("gemv_w8a16_wide", "gemv_w8a16_m16", "gemv_w8a16_mM")
_WIDE_BF16 = ("gemv_wide", "gemv_bf16_wide", "gemv_m16", "gemv_bf16_mM")


def wide_kernel_ready() -> bool:
    """True when the extension has a wide-M batched GEMV (one pass for M up to 32)."""
    return any(hasattr(hip_ops, name) for name in _WIDE_W8 + _WIDE_BF16)


#: Names the task-12 multi-query paged attention kernel may expose (the one that
#: reads a row's history KV once for all ``c`` verify queries).
_MULTI_Q = ("attn_paged_multi_q", "attn_paged_multi_q_batch", "attn_paged_multi_query")


def multi_q_ready() -> bool:
    """True when the extension has the multi-query paged attention kernel."""
    return any(hasattr(hip_ops, name) for name in _MULTI_Q)


#: Flipped to True once ``FullAttn.verify_batch_multi`` is wired to the kernel.
MULTI_Q_WIRED = True


#: Counts where ``wide_proj`` sent each module: "dot2" / "wide" / "smallM" /
#: "prefill".  The one-forward verify must be all-dot2 in int8 builds; anything
#: else is the dequant path and shows up as a step-time regression.
WIDE_STATS: dict[str, int] = {}

#: The dot2 wide kernel measures ~50-72 Gelem/s on this part's shapes, while the
#: plain M=8 kernel runs at ~184-200 Gelem/s, so a 16-row verify is often faster
#: as *two* M=8 passes (2 weight reads at 3x the rate) than one M=16 dot2 pass.
#: Default on: it also keeps the per-row numbers identical to the greedy M=8
#: decode step.  ``Q35_WIDE_SPLIT8=0`` switches back to the wide kernel.
import os as _os

_SPLIT8 = _os.environ.get("Q35_WIDE_SPLIT8", "1") != "0"


def _count(kind: str) -> None:
    WIDE_STATS[kind] = WIDE_STATS.get(kind, 0) + 1


def wide_proj(proj, x: torch.Tensor):
    """Batched projection at an arbitrary M (``B*c``), for the one-forward verify.

    Uses the task-12 wide-M GEMV when it exists (one weight read for all rows);
    otherwise falls back to ``GEMV.prefill`` (correct, but on int8 weights that
    materialises the fp16 weights, which is why the one-forward verify is slow
    until the kernel lands).  Like :func:`batch_proj` it accepts a
    :class:`~ops.gemv.GEMVGroup` and returns a list for it.
    """
    if isinstance(proj, GEMVGroup):
        return wide_proj(proj.g, x).split(proj.sizes, dim=1)
    g = proj
    M = x.shape[0]
    x = x.contiguous()
    if g.q is not None:
        # Prefer the packed-fp16 dot2 kernel (kernels task-12): same one-pass
        # weight read, ~1.7x the element rate of the first int8 wide kernel, with
        # a ULP-level difference from the greedy kernel instead of bit equality.
        dot2 = getattr(hip_ops, "gemv_w8a16_wide_dot2", None)
        wide = getattr(hip_ops, "gemv_w8a16_wide", None)
        if _SPLIT8 and M % 8 == 0 and M > 8:
            _count("split8")
            return torch.cat([_project(g, x[i:i + 8]) for i in range(0, M, 8)], dim=0)
        fn = dot2 or wide
        if fn is not None and M in (16, 24, 32):
            u = g.u if g.u in (2, 4) else 2
            _count("dot2" if dot2 is not None else "wide")
            return fn(g.q.wq, g.q.scale, x, u)
        if M in (2, 4, 8):
            _count("smallM")
            return _project(g, x)
    else:
        fn = next((getattr(hip_ops, n) for n in _WIDE_BF16 if hasattr(hip_ops, n)), None)
        if fn is not None and M in (16, 24, 32):
            _count("wide_bf16")
            return fn(g.w, x)
        if M in (2, 4, 8):
            _count("smallM")
            return _project(g, x)
    _count("prefill")
    return g.prefill(x)
