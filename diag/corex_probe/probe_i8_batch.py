"""What the int8 batched GEMV is worth, measured at the level it acts on.

``gemv_kn_i8_m`` exists because ``_gemv_rows`` had no batched int8 entry and looped the
single-row kernel, reading the whole weight once per row. The claim to check is narrow and
mechanical: an ``M``-row projection should now cost about one weight read instead of ``M``.

It is deliberately measured at the projection, not at the decode step, and the reason is worth
recording: ``backends.ops`` is a table frozen at import (``backends/__init__.py``), so a probe
cannot A/B the dispatch by patching ``corex.linear_m`` -- the table already holds the original
callable and both arms run identical code. An earlier version of this probe did exactly that
and reported a perfectly flat 1.00x, which reads like "the kernel did nothing" and is really
"the harness measured one arm twice".

So the two arms here call the two kernel entries directly:

* ``gemv_kn_i8`` per row, which is what ``_gemv_rows`` did before the batched kernel;
* ``gemv_kn_i8_m`` once for all rows, which is what it does now.

Both are the same arithmetic (the row-identity check in ``check_ops.py`` pins that), so the
difference is the weight traffic and nothing else.

Module: ``coffin/diag/corex_probe``. Runs on the card; nothing imports it.

Constraints:
    * Eager only. Nothing here captures, and ``caps.graph`` stays 0 on this backend.
    * Measured on the model's real projection shapes, because bandwidth on this card depends on
      ``N`` (``COREX_PROT`` section 0) and a toy shape would measure the wrong thing.
    * Outputs are compared with ``torch.equal``: a batched path that is merely close would
      change a token once in a long generation, so the equality is the contract, not a nicety.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: ``coffin/`` is the ``sys.path`` root; this file sits two levels below it.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

#: Real projection shapes from a decode step, ``(label, N, K)``. ``lm_head`` is included
#: because it is the widest read of a step and the shape most likely to behave differently.
SHAPES = (
    ("mlp.gate_up", 18432, 2560),
    ("mlp.down", 2560, 9216),
    ("mix.out_proj", 2560, 4096),
    ("mix.qkv", 8192, 2560),
    ("lm_head", 248320, 2560),
)

#: Row counts to sweep. 2/4/8 are the batched-decode buckets; 5 is the odd width the verify
#: path would ask for, which the entry serves by padding to 8.
ROWS = (2, 4, 5, 8)


def _quant_kn(wt):
    """Quantise a ``[K, N]`` matrix to int8 with one fp32 scale per output column.

    The loader's rule, restated locally so this probe needs no checkpoint.

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


def _bench(fn, iters: int = 20, warmup: int = 5) -> float:
    """Time a callable with device events, best of ``iters``.

    Args:
        fn: Zero-argument callable.
        iters: Timed iterations.
        warmup: Untimed iterations first, so a lazy first launch is not measured.

    Returns:
        Best milliseconds observed.
    """
    from ops import runtime

    for _ in range(warmup):
        fn()
    runtime.synchronize()
    best = float("inf")
    for _ in range(iters):
        start, end = runtime.Event(), runtime.Event()
        start.record()
        fn()
        end.record()
        runtime.synchronize()
        best = min(best, start.elapsed_ms(end))
    return best


def main() -> int:
    """Compare the per-row loop against the batched kernel at every shape and width.

    Returns:
        ``0`` on success, ``1`` when there is no device.
    """
    import torch

    import backends
    from backends.corex import pack

    print(f"  backend: {backends.status()}", flush=True)
    print(f"  device : {torch.cuda.get_device_name(0)} "
          f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB", flush=True)

    module = backends.corex.kernels.load_kernels(backends.caps().arch)

    print(f"\n  {'shape':<14}{'M':>3}{'per-row us':>12}{'batched us':>12}{'speedup':>9}"
          f"{'GB/s':>8}  identical", flush=True)
    wins = []
    for label, n, k in SHAPES:
        w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
        q, s = _quant_kn(pack.to_kn(w, torch.bfloat16).wt)
        wq = pack.KnQWeight(q, s)
        nbytes = q.numel() + s.numel() * 4
        for rows in ROWS:
            x = torch.randn(rows, k, dtype=torch.bfloat16, device="cuda")
            loop_out = torch.stack([module.gemv_kn_i8(wq.wt, x[r], wq.scale) for r in range(rows)])
            batched_out = module.gemv_kn_i8_m(wq.wt, x, wq.scale)
            same = torch.equal(loop_out, batched_out)
            loop_ms = _bench(lambda: [module.gemv_kn_i8(wq.wt, x[r], wq.scale)
                                      for r in range(rows)])
            bat_ms = _bench(lambda: module.gemv_kn_i8_m(wq.wt, x, wq.scale))
            gbs = (nbytes / (bat_ms * 1e-3)) / 1e9
            print(f"  {label:<14}{rows:>3}{loop_ms * 1e3:>12.1f}{bat_ms * 1e3:>12.1f}"
                  f"{loop_ms / bat_ms:>8.2f}x{gbs:>8.1f}  {same}", flush=True)
            wins.append(loop_ms / bat_ms)
            if not same:
                print(f"    the batched kernel is not row-identical to the loop", flush=True)
            del x, loop_out, batched_out
        del w, q, s, wq
        torch.cuda.empty_cache()

    print(f"\n  mean projection speed-up over all shapes and widths: "
          f"{sum(wins) / len(wins):.2f}x", flush=True)
    print("  read it as weight traffic: 1.0x would mean the kernel still reads per row,"
          "\n  and M x would mean no amortisation at all.", flush=True)
    print("\nI8_BATCH_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
