"""Why M=2047 costs 17x what M=2048 costs: find the GEMM's M-alignment cliff.

The drafter's prefill is the largest single item in a 2048-token prefill (221 ms of 963 on
bf16), and the previous probe localised all of it to ``GemmEx_kernel::gemmh_new``: five
GEMMs, each called with **M=2047**. Every other GEMM in the model runs at M=2048 and lands
on the tensor-core path.

2047 is not an accident of this model. ``MTP.prefill`` slices ``min(c1, n - 1)`` rows,
because the last prompt token has no next token, so the drafter's GEMM is always one row
short of the main model's -- and one row short is enough to fall off the fast kernel:

    [2048, 2560] x [2560, 18432]   32 calls, 4.29 ms each  ->  45 TFLOP/s  (TCU)
    [2047, 2560] x [2560, 18432]    1 call,  75.70 ms      ->   2.5 TFLOP/s (GemmEx)

So this sweep answers three questions the fix depends on:

1. **Where is the cliff?** M from 2040 to 2056 pins the divisor (256? 64? 8? even/odd?).
2. **Is it M alone, or shape-specific?** The same sweep on three projections, because a
   rule that only holds for one shape cannot be turned into a padding policy.
3. **What does the drafter's prefill cost if M were aligned?** ``MTP.prefill`` called
   directly at n=2047 against n=2048 is the upper bound on the fix, measured rather than
   inferred -- and it also shows whether the padding rows themselves are free.

Module: ``coffin/diag/corex_probe``. Runs on the card; nothing imports it.

Constraints:
    * No model weights are needed for the sweep, so it runs first and answers even if the
      model fails to load. The drafter test does need the checkpoint.
    * Each point is timed three times and the best kept: the cliff is a factor of 17, far
      above any noise here, but a first-call allocator growth would still distort a single
      sample.
    * The GEMMs are timed through ``torch.matmul`` on ``[K, N]`` operands, which is the
      call ``linear_prefill`` makes, so the fallback decision is the one the engine sees.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: ``coffin/`` is the ``sys.path`` root; this file sits two levels below it.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

MODEL = os.environ.get("ENGINE_MODEL", "")

#: Context window for the drafter test.
MAX_SEQ = 4096

#: Real projection shapes from a decode step (``probe_shapes.py``); ``N``/``K`` as the
#: engine hands them to ``linear_prefill`` after the loader's ``[K, N]`` transpose.
SHAPES = (
    ("mlp.gate_up", 18432, 2560),
    ("mlp.down", 2560, 9216),
    ("mix.out_proj", 2560, 4096),
)

#: Row counts to sweep. 2040-2064 finds the local cliff; the far points check whether the
#: fast path returns and whether a large odd M is also slow.
M_SWEEP = (
    1024, 1535, 1536, 2040, 2044, 2046, 2047, 2048, 2049, 2050, 2052, 2056, 2064,
    3071, 3072, 4095, 4096,
)


def _events() -> tuple:
    """Create a start/end device-event pair with timing enabled.

    Returns:
        ``(start, end)`` events.
    """
    import torch

    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def _time_matmul(m: int, n: int, k: int, reps: int = 3) -> tuple[float, float]:
    """Time ``[m, k] @ [k, n]`` and return (ms, TFLOP/s).

    Args:
        m: Rows of the activation operand, i.e. the prefill chunk length.
        n: Output columns, the projection's ``N``.
        k: Reduction depth, the projection's ``K``.
        reps: Timed repeats.

    Returns:
        ``(best_ms, tflops)``.
    """
    import torch

    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(k, n, dtype=torch.bfloat16, device="cuda")
    torch.matmul(x, w)
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        start, end = _events()
        start.record()
        torch.matmul(x, w)
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end))
    del x, w
    torch.cuda.empty_cache()
    return best, 2 * m * n * k / (best * 1e-3) / 1e12


def _sweep() -> None:
    """Print the M-alignment sweep per shape, then a per-shape verdict."""
    print(f"\n{'=' * 78}\nGEMM M-alignment sweep (bf16, [M,K] @ [K,N])\n{'=' * 78}",
          flush=True)
    verdicts = {}
    for label, n, k in SHAPES:
        print(f"\n  --- {label}: N={n} K={k} ---", flush=True)
        print(f"  {'M':>6}{'ms':>10}{'TFLOP/s':>10}{'vs M=2048':>11}", flush=True)
        ref = None
        for m in M_SWEEP:
            ms, tflops = _time_matmul(m, n, k)
            if m == 2048:
                ref = ms
            rel = (ref / ms) if (ref and m != 2048) else 1.0
            print(f"  {m:>6}{ms:>10.2f}{tflops:>10.1f}{rel:>10.2f}x", flush=True)
            verdicts.setdefault(label, {})[m] = (ms, tflops)
        # The threshold is the smallest M at or below 2048 that is still fast, which is
        # what a padding policy has to round up to.
        fast = [m for m, (_ms, tf) in verdicts[label].items()
                if m <= 2048 and tf > 20.0]
        slow = [m for m, (_ms, tf) in verdicts[label].items()
                if m <= 2048 and tf <= 20.0]
        print(f"  fast (<=2048): {sorted(fast)}", flush=True)
        print(f"  SLOW (<=2048): {sorted(slow)}", flush=True)


def _mtp_pad_test() -> None:
    """Time ``MTP.prefill`` at the row counts the padding fix would produce.

    ``MTP.prefill`` is called by ``Qwen35.prefill`` with ``n - 1`` rows, so at n=2048 it
    gets 2047. Calling it directly with an aligned row count is the same computation plus
    a few padding rows, which is exactly what the fix would do.

    Args:
        Nothing; reads ``MODEL`` from the environment.
    """
    import torch

    from engine_api import create_model, load_weights

    print(f"\n{'=' * 78}\nMTP drafter prefill at padded row counts\n{'=' * 78}",
          flush=True)
    sd = load_weights(MODEL, quant=False, vision=False)
    model = create_model(MODEL, max_seq=MAX_SEQ, bench=False, weights=sd)
    if model.mtp is None:
        print("  no MTP head in this checkpoint", flush=True)
        return
    ctx = (model.cos_tbl, model.sin_tbl, model.mask_idx)
    hidden = model.cfg["hidden_size"]
    n = 2048
    print(f"  {'rows':>6}{'ms':>10}{'vs 2047':>10}", flush=True)
    base = None
    for rows in (2047, 2048, 2049, 3072):
        h = torch.randn(rows, hidden, dtype=torch.bfloat16, device="cuda")
        toks = torch.randint(0, 100000, (rows,), dtype=torch.long, device="cuda")
        pos = torch.arange(rows, dtype=torch.long, device="cuda")
        model.mtp.reset()
        model.mtp.prefill(h, toks, pos, ctx)
        torch.cuda.synchronize()
        best = float("inf")
        for _ in range(2):
            model.mtp.reset()
            start, end = _events()
            start.record()
            model.mtp.prefill(h, toks, pos, ctx)
            end.record()
            torch.cuda.synchronize()
            best = min(best, start.elapsed_time(end))
        if rows == 2047:
            base = best
        print(f"  {rows:>6}{best:>10.1f}{(base / best if base else float('nan')):>9.2f}x",
              flush=True)
        del h, toks, pos
        torch.cuda.empty_cache()
    _ = n


def main() -> int:
    """Run the sweep and the drafter padding test.

    Returns:
        ``0`` on success, ``1`` when the model directory is missing.
    """
    import torch

    import backends

    print(f"  backend: {backends.status()}", flush=True)
    print(f"  device : {torch.cuda.get_device_name(0)} "
          f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB",
          flush=True)

    _sweep()

    if MODEL and Path(MODEL).is_dir():
        _mtp_pad_test()
    else:
        print(f"\n  ENGINE_MODEL unset or not a directory: {MODEL!r}; "
              "skipping the drafter test", flush=True)

    print("\nGEMM_ALIGN_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
