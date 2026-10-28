"""How much of a decode step is host dispatch, and how much is device work?

The question this answers: would a single C++ entry point that issues every launch in a loop
buy anything? It removes Python, ATen dispatch and pybind from every op -- but only helps if
that host time is on the *critical path*. If the host finishes issuing step N while the GPU is
still working on step N-1, the cost is already hidden and removing it buys nothing.

The decomposition, which needs no profiler and so carries no profiler overhead:

    A = steps enqueued back-to-back, one synchronise at the end
        -> wall is max(host, device) per step, because the queue absorbs the host work
    S = every step followed by its own synchronise
        -> wall is host + device, because the queue is drained before the next enqueue

    device ~= A          host ~= S - A

That is only valid while ``host < device``; if the host were the slow side, ``A`` would be the
host time and the subtraction would report a near-zero host. The probe therefore checks the
inequality it assumes rather than leaving it implicit, and says so when it does not hold.

Why this and not the existing trace probe: ``probe_step_budget.py`` sums per-kernel durations
from a profiler trace, which this card inflates by ~46% (it reported 33.66 ms of kernels for a
23.10 ms step). That trace answers "which kernel is biggest" well and "how much is host" not at
all. This one measures the host/device split on the wall clock, which is what a launch-overhead
removal would have to beat.

Module: ``coffin/diag/corex_probe``. Runs on the card.

Constraints:
    * Eager path, int8 weights -- the serving path (``load_weights(quant=True)``). CUDA graphs
      stay off: this backend disables them and the driver wedges cards on replay.
    * Warmup runs before either series, and the two series are interleaved rather than run in
      blocks, so a clock or thermal drift over the job cannot land on one of them alone.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

#: ``coffin/`` itself is the ``sys.path`` root; this file is two levels down from it.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

MODEL = os.environ.get("ENGINE_MODEL", "")

#: Steps per series. Enough to average out scheduler noise, short enough that drift is not a
#: factor.
STEPS = 20


def main() -> int:
    """Split one decode step into host and device time.

    Returns:
        ``0`` on success, ``1`` when the model directory is not set.
    """
    import torch

    import backends
    from engine_api import create_model, load_weights

    print(f"  backend: {backends.status()}", flush=True)
    if not MODEL or not Path(MODEL).is_dir():
        print(f"  ENGINE_MODEL unset or not a directory: {MODEL!r}", flush=True)
        return 1

    sd = load_weights(MODEL, quant=True, vision=False)
    model = create_model(MODEL, max_seq=1024, bench=False, weights=sd)

    model.reset()
    logits = model.prefill([1, 2, 3, 4, 5])
    torch.cuda.synchronize()
    state = [int(logits.argmax())]

    def step() -> None:
        """One decode step, kept as a named callable so both series run identical code."""
        state[0] = int(model.logits(state[0]).argmax())

    def reset() -> None:
        """Return the model to a known position so both series do the same work."""
        model.reset()
        lg = model.prefill([1, 2, 3, 4, 5])
        torch.cuda.synchronize()
        state[0] = int(lg.argmax())

    # Warmup: allocator pools, lazy handle creation and the first-touch of every buffer, none
    # of which either series should pay.
    reset()
    for _ in range(5):
        step()
    torch.cuda.synchronize()

    async_ms: list[float] = []
    sync_ms: list[float] = []
    # Interleaved: one async series and one sync series alternately, each of a few steps, so a
    # slow drift affects both equally instead of loading onto whichever ran second.
    for _ in range(4):
        reset()
        start = time.perf_counter()
        for _ in range(STEPS // 4):
            step()
        torch.cuda.synchronize()
        async_ms.append((time.perf_counter() - start) * 1000.0 / (STEPS // 4))

        reset()
        start = time.perf_counter()
        for _ in range(STEPS // 4):
            step()
            torch.cuda.synchronize()
        sync_ms.append((time.perf_counter() - start) * 1000.0 / (STEPS // 4))

    a = sum(async_ms) / len(async_ms)
    s = sum(sync_ms) / len(sync_ms)
    host = s - a
    device = a
    share = 100.0 * host / s if s > 0 else 0.0

    print("\n=== host versus device time on one decode step ===", flush=True)
    print(f"  A  async (queue absorbs host) : {a:7.3f} ms/step", flush=True)
    print(f"  S  sync  (drained every step): {s:7.3f} ms/step", flush=True)
    print(f"  -> device (A)                : {device:7.3f} ms", flush=True)
    print(f"  -> host   (S - A)            : {host:7.3f} ms  ({share:.1f}% of the sync step)",
          flush=True)

    print("\n=== what a C++ single-entry decode_step would recover ===", flush=True)
    # A single C++ entry point removes Python, ATen dispatch and pybind from every op, but it
    # still issues the same number of driver launches, and it cannot remove the host cost of
    # those launches. So the ceiling is the host share, not all of it.
    print(f"  host time on the critical path: {host:.3f} ms/step", flush=True)
    print(f"  best case if ALL of it vanished: {a:.3f} -> {device:.3f} ms "
          f"({device / a:.3f}x, {1000.0 / device:.1f} tok/s)", flush=True)
    if host < 0.15 * s:
        print("  verdict: host is NOT on the critical path. Removing dispatch, pybind and\n"
              "           the Python loop buys at most the number above, minus the launch\n"
              "           cost a C++ entry point still pays.", flush=True)
    else:
        print("  verdict: host is a material share of the step; a lower-overhead entry point\n"
              "           is worth prototyping before more kernel work.", flush=True)

    if host <= 0:
        print("\n  WARNING: S - A <= 0, so the assumed inequality (host < device) is false for\n"
              "           this measurement. Read A as the slow side, not as device time.",
              flush=True)

    print("\nHOST_BUDGET_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
