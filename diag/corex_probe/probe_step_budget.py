"""Split an eager decode step into GPU-busy, CPU-busy and idle, to choose what replaces
the graph.

The graph is off the table: replay measured 46.97 ms/step against eager's 43.46 ms, so it
is now slower as well as unreliable, and one run stopped inside a replay with no traceback.
Whatever replaces it has to beat the eager path on its own terms, which means the first
question is what the eager path actually wastes.

Three numbers decide it:

* **wall** -- device events around one step. What a token costs today.
* **gpu_busy** -- the sum of kernel durations from the profiler for that same step. If it
  is close to wall, the card is working and only fewer/faster kernels help; if it is much
  below wall, the card is idle between launches and the fix is to stop paying per launch.
* **cpu_busy** -- the host time the step spends inside ATen and the launcher. If it exceeds
  wall, the host is the critical path and a C++ entry point that issues the whole step from
  one call is worth more than any kernel change.

A fourth measurement, the per-kernel launch count, says how much a persistent kernel would
have to absorb: every launch is a candidate for fusion.

Module: ``coffin/diag``. Runs on the card.

Constraints:
    * Eager only. Capturing is what is being replaced, so it must not be in the path that
      produces the numbers.
    * Prints and flushes each section before the next begins: the graph experiment runs
      last precisely so that a wedge in it still leaves the attribution on the log.
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

#: Steps to average. Enough to hide scheduler jitter, small enough that a wedged card does
#: not burn the job before the first section prints.
STEPS = 20

#: Steps inside the profiler window. One: the trace is read for per-kernel totals, and a
#: single step is what the timing above measured.
PROFILE_STEPS = 1


def _time_steps(step, n: int) -> float:
    """Time ``n`` calls of ``step`` with device events.

    Args:
        step: Zero-argument callable performing one decode step.
        n: Number of calls.

    Returns:
        Mean milliseconds per call.
    """
    import torch

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n):
        step()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n


def main() -> int:
    # Markers before every import that can be slow. Importing `backends` resolves the
    # backend selection and the capability record, which loads (or, if the build looks
    # stale, rebuilds) the compiled extension -- minutes of work that writes to stderr and
    # nothing to stdout. Without these lines such a wait is indistinguishable from a hang,
    # and a probe that cannot say where it is has already cost several jobs.
    print(f"  [i0] python {sys.version.split()[0]} cwd={os.getcwd()}", flush=True)
    import torch

    print(f"  [i1] torch {torch.__version__}", flush=True)
    import backends

    print(f"  [i2] backends imported", flush=True)
    print(f"  backend: {backends.status()}", flush=True)
    if not MODEL or not Path(MODEL).is_dir():
        print("  ENGINE_MODEL unset or not a directory", flush=True)
        return 1

    from engine_api import create_model, load_weights

    print("  [i3] engine_api imported; loading the model", flush=True)
    # quant=True belongs to the *loader*: it is what makes the state dict hold int8
    # QWeights. A backend's caps.int8_weights only says it can consume them, so a
    # backend with the kernel but without this flag silently measures the bf16 path --
    # which is exactly what the first int8 profile run did.
    weights = load_weights(MODEL, quant=True)
    print(f"  [i4] weights loaded; sample form: "
          f"{type(next(iter(weights.values()))).__name__}", flush=True)
    model = create_model(MODEL, max_seq=1024, bench=False, weights=weights)
    print(f"  loaded, caps.graph={int(backends.caps().graph)}", flush=True)

    state = {"logits": None}

    def step() -> None:
        """Run one eager decode step."""
        nxt = int(state["logits"].argmax())
        state["logits"] = model.logits(nxt)

    def prepare() -> None:
        """Put the model at a fresh prompt, ready to decode."""
        model.reset()
        state["logits"] = model.prefill([1, 2, 3, 4, 5])
        torch.cuda.synchronize()

    # Warmup outside the timing: the first step after a prompt pays allocator and JIT
    # handle costs that every later step does not.
    prepare()
    for _ in range(3):
        step()
    torch.cuda.synchronize()

    print("\n=== A. eager wall time ===", flush=True)
    prepare()
    wall = _time_steps(step, STEPS)
    print(f"  wall/step : {wall:.2f} ms  ({1000 / wall:.2f} tok/s)", flush=True)

    print("\n=== B. kernel durations and launch count (profiler) ===", flush=True)
    gpu_busy = float("nan")
    launches = 0
    try:
        from torch.profiler import ProfilerActivity, profile

        activities = [ProfilerActivity.CPU]
        if hasattr(ProfilerActivity, "CUDA"):
            activities.append(ProfilerActivity.CUDA)
        prepare()
        with profile(activities=activities) as prof:
            for _ in range(PROFILE_STEPS):
                step()
            torch.cuda.synchronize()

        events = prof.key_averages()
        total_cuda_us = 0.0
        total_calls = 0
        rows = []
        for e in events:
            cuda_us = getattr(e, "self_device_time_total", None)
            if cuda_us is None:
                cuda_us = getattr(e, "self_cuda_time_total", 0)
            if cuda_us:
                total_cuda_us += cuda_us
                rows.append((cuda_us, e.key, e.count))
                total_calls += e.count
        rows.sort(reverse=True)
        gpu_busy = total_cuda_us / 1000.0 / PROFILE_STEPS
        launches = total_calls
        print(f"  {'kernel':<44} {'calls':>6} {'ms/step':>9} {'%':>6}", flush=True)
        for cuda_us, key, count in rows[:14]:
            share = 100 * cuda_us / total_cuda_us if total_cuda_us else 0
            print(f"  {key[:44]:<44} {count // PROFILE_STEPS:>6} "
                  f"{cuda_us / 1000.0 / PROFILE_STEPS:>9.3f} {share:>5.0f}%", flush=True)
        print(f"  {'TOTAL':<44} {total_calls // PROFILE_STEPS:>6} "
              f"{gpu_busy:>9.3f} 100%", flush=True)
    except Exception as exc:                      # noqa: BLE001 - trace is informational
        print(f"  profiler unavailable: {type(exc).__name__}: {exc}", flush=True)

    print("\n=== C. host time inside the step (what a C++ entry saves) ===", flush=True)
    # The launcher-style measurement above cannot answer this: it wraps the whole loop in
    # device events, so a host that is merely *ahead* of the device looks free. Here the
    # clock is read on the host around each step, with no synchronise inside the loop, so
    # the total is exactly the Python + ATen dispatch time. If cpu_busy exceeds wall the
    # host is the critical path and issuing the step from one C++ call removes real time;
    # if it is well below wall, the host is already waiting on the device and a C++ entry
    # would only move dispatches off a non-critical path.
    prepare()
    for _ in range(3):
        step()
    torch.cuda.synchronize()
    host_start = time.perf_counter()
    for _ in range(STEPS):
        step()
    cpu_busy = (time.perf_counter() - host_start) * 1000.0 / STEPS
    # Drain so the device work queued by that loop is not left to pollute the next section.
    torch.cuda.synchronize()
    print(f"  cpu busy/step : {cpu_busy:.2f} ms  "
          f"({100 * cpu_busy / wall:.0f}% of wall)", flush=True)
    if cpu_busy >= wall:
        print("  verdict       : HOST BOUND -- one C++ entry point removes real wall time",
              flush=True)
    elif cpu_busy >= 0.75 * wall:
        print("  verdict       : host is close to the critical path; a C++ entry would "
              "recover much of the gap", flush=True)
    else:
        print("  verdict       : host is not the critical path; a C++ entry only moves "
              "dispatch off it", flush=True)

    print("\n=== D. trace sanity (not a verdict) ===", flush=True)
    if gpu_busy == gpu_busy:
        print(f"  trace sum : {gpu_busy:.2f} ms  vs wall {wall:.2f} ms", flush=True)
        print(f"  launches  : {launches} per step", flush=True)
        if gpu_busy >= wall:
            # The trace over-counts by construction: every entry carries the profiler's own
            # cost per kernel, so the sum grows with the launch count rather than with the
            # work. A sum above the wall clock therefore proves the sum is unusable as a
            # busy figure -- and nothing more. Reading it as "the device is busy for the
            # whole step" is the inference PERF_COREX 12.1 had to retract.
            print(f"  inflation : {100 * (gpu_busy / wall - 1):.0f}% over wall", flush=True)
            print("  NOTE      : the sum is unusable as busy time. Use section C: it "
                  "measures host time directly.", flush=True)
        else:
            print("  usage     : the sum is only comparable across runs, never a share of "
                  "wall", flush=True)
            print("  verdict  : device bound -- only kernel work matters", flush=True)
    else:
        print("  no profiler data; wall only", flush=True)

    # There is deliberately no graph comparison here. Capture is what is being replaced,
    # and on this driver a replay can wedge a card permanently with no traceback, so a
    # probe must not touch it even behind a flag -- a probe that can destroy the device it
    # measures is not a probe. The numbers that decide the alternative come from A-C above.
    print("\nSTEP_BUDGET_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
