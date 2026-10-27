"""Where does a decode step's time go? Measure, do not guess.

The port is correct but slow: 52.41 ms/step against ~12.7 ms/step for the same model on
the reference accelerator. The obvious explanations -- no int8 weights, no graph -- are
both already ruled out as sufficient (the graph is on and worth 1.47x; int8 weights are
worth another 1.4x), so the remaining gap has to be measured rather than assumed.

Two measurements, because they answer different questions:

1. **A per-op budget on the model's real shapes.** Sums the projections as standalone
   kernels and compares the total against one decode step. If the projections account
   for most of the step, the fix is in the GEMV; if they do not, the time is elsewhere
   and tuning the GEMV would be wasted work.
2. **A profiler trace of one eager decode step.** Names the ops that actually dominate
   and how many launches there are, which separates "one slow kernel" from "many small
   kernels whose sum is the step".

The bandwidth each shape achieves is printed next to its time, against the card's
measured copy ceiling, because that ratio -- not tok/s -- is what says how much is left.

Module: ``coffin/diag``. Runs on the card.

Constraints:
    * Eager path only (``model.logits``): the graph path would collapse 2500 launches
      into one replay and hide the per-op cost this probe exists to attribute.
    * One warmup before timing, and events rather than wall clock, so a lazy first
      launch does not land in the measurement.
    * The copy ceiling is measured here rather than quoted, because the number that
      matters is this card's, not a datasheet's.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

MODEL = os.environ.get("ENGINE_MODEL", "")


def _bench(fn, iters: int = 10, warmup: int = 3) -> float:
    """Time a callable with device events, in milliseconds per call.

    Args:
        fn: Zero-argument callable to time.
        iters: Timed iterations.
        warmup: Untimed iterations, so lazy handle creation is not measured.

    Returns:
        Mean milliseconds per call.
    """
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> int:
    import torch

    import backends

    print(f"  backend: {backends.status()}", flush=True)
    print(f"  model  : {MODEL}", flush=True)
    if not MODEL or not Path(MODEL).is_dir():
        print("  ENGINE_MODEL unset or not a directory", flush=True)
        return 1

    table = backends.ops

    # ---------------------------------------------------------------- ceiling
    print("\n=== copy ceiling ===", flush=True)
    n = 64 * 1024 * 1024
    src = torch.empty(n, dtype=torch.uint8, device="cuda")
    dst = torch.empty(n, dtype=torch.uint8, device="cuda")
    copy_ms = _bench(lambda: dst.copy_(src), iters=10, warmup=3)
    ceiling = 2 * n / (copy_ms * 1e-3) / 1e9
    print(f"  D2D copy {copy_ms:.3f} ms -> {ceiling:.0f} GB/s", flush=True)
    del src, dst
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------- per-shape
    # The model's projection geometries, from config.json: hidden 2560,
    # intermediate 9216, fused qkv+gate 8192, o_proj K=4096, vocab 248320.
    shapes = (
        ("mlp_up", 9216, 2560),
        ("mlp_down", 2560, 9216),
        ("qkv_gate", 8192, 2560),
        ("o_proj", 2560, 4096),
        ("lm_head", 248320, 2560),
    )
    print("\n=== decode GEMV per shape (M=1) ===", flush=True)
    print(f"  {'shape':<10} {'N':>7} {'K':>6} {'ms':>8} {'GB/s':>8} {'%ceil':>6}",
          flush=True)
    total = 0.0
    per_shape = {}
    for label, nn, kk in shapes:
        w = torch.randn(nn, kk, dtype=torch.bfloat16, device="cuda")
        x = torch.randn(kk, dtype=torch.bfloat16, device="cuda")
        table.linear(w, x)
        torch.cuda.synchronize()
        ms = _bench(lambda: table.linear(w, x))
        nbytes = nn * kk * 2
        gbs = nbytes / (ms * 1e-3) / 1e9
        per_shape[label] = ms
        print(f"  {label:<10} {nn:>7} {kk:>6} {ms:>8.3f} {gbs:>8.1f} "
              f"{100 * gbs / ceiling:>5.0f}%", flush=True)
        del w, x
        torch.cuda.empty_cache()

    # One layer's projections; the stack is 32 layers. Not every layer has all four
    # (24 are linear-attention, 8 are full), so this is an upper bound on the stack.
    layer_ms = sum(per_shape[k] for k in ("mlp_up", "mlp_down", "qkv_gate", "o_proj"))
    stack_ms = layer_ms * 32
    print(f"\n  one layer's projections : {layer_ms:.3f} ms", flush=True)
    print(f"  x32 layers              : {stack_ms:.2f} ms", flush=True)
    print(f"  lm_head                 : {per_shape['lm_head']:.3f} ms", flush=True)
    print(f"  projections total       : {stack_ms + per_shape['lm_head']:.2f} ms",
          flush=True)

    # ---------------------------------------------------------------- real step
    print("\n=== one eager decode step, end to end ===", flush=True)
    from engine_api import create_model

    model = create_model(MODEL, max_seq=1024, bench=False)
    model.reset()
    logits = model.prefill([1, 2, 3, 4, 5])
    torch.cuda.synchronize()

    def step() -> None:
        """One eager decode step, kept as a named callable for the profiler."""
        nonlocal logits
        logits = model.logits(int(logits.argmax()))

    step()
    torch.cuda.synchronize()
    model.reset()
    logits = model.prefill([1, 2, 3, 4, 5])
    torch.cuda.synchronize()
    step_ms = _bench(step, iters=20, warmup=5)
    print(f"  decode step: {step_ms:.2f} ms  ({1000 / step_ms:.2f} tok/s)", flush=True)
    covered = (stack_ms + per_shape["lm_head"]) / step_ms * 100
    print(f"  projections account for {covered:.0f}% of the step", flush=True)
    print(f"  everything else         : {step_ms - stack_ms - per_shape['lm_head']:.2f} ms",
          flush=True)

    # ---------------------------------------------------------------- trace
    print("\n=== top ops in one step (profiler) ===", flush=True)
    try:
        from torch.profiler import ProfilerActivity, profile

        activities = [ProfilerActivity.CPU]
        if hasattr(torch.profiler.ProfilerActivity, "CUDA"):
            activities.append(ProfilerActivity.CUDA)
        with profile(activities=activities) as prof:
            step()
            torch.cuda.synchronize()
        keys = ("name", "cuda_time_total", "cpu_time_total", "count")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=18),
              flush=True)
    except Exception as exc:                      # noqa: BLE001 - trace is a bonus
        print(f"  profiler unavailable: {type(exc).__name__}: {exc}", flush=True)

    print("\nDECODE_PROFILE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
