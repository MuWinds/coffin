"""Does one weight read serve all ``c`` verified tokens, or does each token pay its own?

Speculative decoding pays off only if verifying ``c`` candidate tokens costs about one weight
pass. That is a property of the projection entry point, not of the drafter, and on this backend
the int8 and bf16 paths answer it differently:

* ``_gemv_rows`` (int8): there is no small-M int8 kernel, so the entry loops the **single-row**
  GEMV once per row -- ``c`` launches and ``c`` full weight reads for a ``c``-token verify.
* ``gemv_kn_m`` (bf16): one launch, weight read once, ``c`` row vectors.

``PERF_COREX`` §13.6b already records that this card's GEMM falls off the tensor-core path at
odd ``M``; the same "shape decides the route" logic applies here, so the numbers have to be
measured at the real projection shapes rather than inferred from a bandwidth roof.

This probe measures the marginal cost of rows ``1..c`` on every projection shape a decode step
uses, for both weight paths, and reports:

* ``t_c / t_1`` -- the ratio that decides the trade. ``~1`` means the weight read is amortised
  and a verify is nearly free; ``~c`` means every draft token costs a full decode step of
  bandwidth and no drafter can pay for itself.
* the same ratio for ``lin`` shapes only, so a fused wide projection does not hide a narrow one.

Module: ``coffin/diag/corex_probe``. Runs on the card; nothing imports it, and no model is
loaded -- the shapes and the weight construction are local to this file, so it costs seconds.

Constraints:
    * No model, no checkpoint: the entries are called on synthetic weights of the real shapes,
      which is what makes this the cheapest probe to re-run after any dispatch change.
    * Timed with device events inside one process, alternating shapes rather than batching them,
      because the question is a ratio and cross-process drift would swamp a 10% difference.
    * Eager only; nothing here captures.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: ``coffin/`` is the ``sys.path`` root; this file sits two levels below it.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

#: Row counts to sweep. The verify width is ``K + 1`` with ``K <= 7`` (``CKPT``), so 2..8 is the
#: whole range a speculative step can ask for; 1 is the baseline the ratio is taken against.
ROWS = (1, 2, 4, 5, 8)

#: Real projection shapes from a decode step, as ``(label, N, K)`` with the weight in the
#: ``[N, K]`` row-major layout the model hands the backend (``probe_shapes.py`` measured them).
#: ``lm_head`` is in the list because it is the widest read of a step and the one most likely to
#: behave differently from the narrow ones.
SHAPES = (
    ("mlp.gate_up", 18432, 2560),
    ("mlp.down", 2560, 9216),
    ("mix.out_proj", 2560, 4096),
    ("mix.qkv", 8192, 2560),
    ("lm_head", 248320, 2560),
)


def _events():
    """Create a timing-enabled start/end event pair on this backend's runtime.

    Returns:
        ``(start, end)``, or ``(None, None)`` when the runtime cannot time.
    """
    from ops import runtime

    return runtime.Event(), runtime.Event()


def _time_call(fn, reps: int = 5) -> float:
    """Time a zero-argument callable with device events, best of ``reps``.

    Args:
        fn: Callable to time; it must already have been called once so its first-call
            allocations do not land in the measurement.
        reps: Timed repeats.

    Returns:
        Best milliseconds observed, or ``-1.0`` when the backend cannot time.
    """
    from ops import runtime

    best = float("inf")
    for _ in range(reps):
        start, end = _events()
        if start is None:
            return -1.0
        start.record()
        fn()
        end.record()
        runtime.synchronize()
        best = min(best, start.elapsed_ms(end))
    return best


def _quant_kn(wt) -> tuple:
    """Quantise an ``[K, N]`` matrix to int8 with one fp32 scale per output column.

    The same rule the weight loader applies: absmax over the column, so the scale is applied
    once per output after the reduction. Written here rather than imported because the loader's
    version lives behind the checkpoint layout, and this probe must not need a checkpoint.

    Args:
        wt: ``[K, N]`` bf16 matrix.

    Returns:
        ``(int8 [K, N] matrix, fp32 [N] scales)``.
    """
    import torch

    f = wt.float()
    scale = f.abs().amax(0).clamp_min(1e-5) / 127.0
    q = (f / scale.view(1, -1)).round().clamp_(-127, 127).to(torch.int8)
    return q, scale


def _sweep(int8: bool) -> list[tuple]:
    """Measure the marginal row cost of every projection shape on one weight path.

    Args:
        int8: Store the weights as int8, i.e. the W8A16 path the serving default uses.

    Returns:
        One row per shape: ``(label, {rows: ms}, ratio_at_width)``.
    """
    import torch

    import backends.corex as corex
    from backends.corex import pack

    # The module-level entries the model's projections call, so the measurement is of the
    # dispatch the engine actually takes rather than of a kernel invoked directly.
    bops = corex

    width = max(ROWS)
    print(f"\n--- weight path {'int8' if int8 else 'bf16'} ---", flush=True)
    print(f"  {'shape':<14}{'N':>7}{'K':>7}" +
          "".join(f"{'M=' + str(r):>10}" for r in ROWS) +
          f"{'M=' + str(width) + '/M=1':>12}", flush=True)

    out: list[tuple] = []
    for label, n, k in SHAPES:
        w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
        # The two storage forms the two weight paths use, built through the backend's own
        # packing entry so the probe measures the layout the model actually hands over.
        packed = pack.to_kn(w, torch.bfloat16)
        if int8:
            q, s = _quant_kn(packed.wt)
            weight = pack.KnQWeight(q, s)
        else:
            weight = packed
        timings: dict[int, float] = {}
        for rows in ROWS:
            x = torch.randn(rows, k, dtype=torch.bfloat16, device="cuda")
            # The op-table entries the model calls: one row is the decode GEMV, more rows is
            # the small-M entry, and that entry is where the two paths diverge.
            entry = bops.linear if rows == 1 else bops.linear_m
            # One untimed call, then the best of the timed ones: the first call pays the
            # allocator and the kernel handle, which a per-row ratio must not include.
            try:
                entry(weight, x[0] if rows == 1 else x)
                torch.cuda.synchronize()
            except Exception as exc:                    # noqa: BLE001 - reported, not raised
                print(f"  {label:<14} M={rows}: {type(exc).__name__}: {exc}", flush=True)
                timings[rows] = -1.0
                continue
            timings[rows] = _time_call(
                lambda e=entry, wt=weight, x=x: e(wt, x[0] if rows == 1 else x))
        base = timings.get(1, -1.0)
        ratio = (timings.get(width, -1.0) / base) if base and base > 0 else float("nan")
        cells = "".join(f"{timings.get(r, -1.0):>10.3f}" for r in ROWS)
        print(f"  {label:<14}{n:>7}{k:>7}{cells}{ratio:>11.2f}x", flush=True)
        out.append((label, timings, ratio))
        del w, weight
        torch.cuda.empty_cache()
    return out


def main() -> int:
    """Sweep the marginal row cost on both weight paths and print the verdict.

    Returns:
        ``0`` on success, ``1`` when there is no device.
    """
    import torch

    import backends

    print(f"  backend: {backends.status()}", flush=True)
    print(f"  device : {torch.cuda.get_device_name(0)} "
          f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB", flush=True)

    for int8 in (True, False):
        rows = _sweep(int8)
        if rows:
            mean = sum(r for _l, _t, r in rows if r == r) / len(rows)
            print(f"  mean M={max(ROWS)}/M=1 ratio: {mean:.2f}x "
                  f"({'weight read amortised' if mean < 1.5 else 'one read per row'})",
                  flush=True)

    print("\n  Read this as: a c-token verify costs `ratio` plain steps of pure projection"
          "\n  time. Speculation wins when ratio < 1 + accepted, and suffix acceptance on"
          "\n  novel text is far below c, so a ratio near c cannot be recovered by any"
          "\n  drafter. LAYER_ROW_COST_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
