"""Inventory every kernel launch of one eager int8 decode step, so fusion picks itself.

The step is launch-floor bound, not bandwidth bound. Two measurements say so: the split-K
reduce launches 128 times to move about 0.12 ms worth of bytes in 2.03 ms, and the step as a
whole issues roughly 1500 launches for a 23 ms wall. If a launch costs ~15 us whether it
moves a byte or a megabyte, the only lever that matters is *how many launches there are* --
and the useful question stops being "which kernel is slow" and becomes "which launches can
be removed".

`probe_step_budget.py` answers the second question only down to the top fourteen rows. That
is not enough to plan a refactor: the rows below the cut add up to more launches than the
ones above it. This probe therefore prints the *complete* inventory, and it prints it three
ways, because each view answers a different question:

1. **By kernel name.** What exists at all, and the per-launch cost of each family. A family
   whose average is well under the launch floor (``gemv_reduce_i8_kernel``) is pure waste.
2. **By launch signature, including shapes.** Which distinct shapes a family serves. One
   name covering forty shapes is a different problem from one name covering one shape.
3. **Grouped by Python frame, when the profiler can record stacks.** *Where* the launch is
   issued from. Shape tables cannot say whether a launch belongs to a residual add or to a
   quantisation, and those two want opposite fixes.

The greedy continuation is printed as ids and is compared against the pre-change run: a
fusion that changes the arithmetic is not a fusion.

Module: ``coffin/diag/corex_probe``. Runs on the card.

Constraints:
    * Eager path only. ``caps.graph`` is ``False`` on this backend by policy and capture on
      this driver has wedged cards, so a probe must not touch it.
    * ``quant=True`` on the loader. The serving path loads int8; ``create_model`` alone
      loads dense, and profiling the dense GEMV would tune a kernel no served token uses.
    * Stack recording is opt-in per run and its absence must not fail the probe: torch 2.1
      accepts ``with_stack`` but a backend without a Python unwind still answers, with an
      empty group. Section 3 reports which of the two happened rather than assuming.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

MODEL = os.environ.get("ENGINE_MODEL", "")

#: Tokens to generate greedily, for the equality check against the pre-change run.
GREEDY_TOKENS = 6

#: Steps averaged for the wall-clock line.
STEPS = 20

#: Steps inside the profile window. One: the inventory is a per-step quantity and a second
#: step only doubles the log.
PROFILE_STEPS = 1


def _bench(fn, iters: int, warmup: int) -> float:
    """Time a callable with device events, in milliseconds per call.

    Args:
        fn: Zero-argument callable.
        iters: Timed iterations.
        warmup: Untimed iterations, so a lazy first launch is not measured.

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


def _device_us(event) -> float:
    """Read one profiler event's self device time in microseconds.

    Args:
        event: A ``FunctionEventAvg`` from ``key_averages()``.

    Returns:
        ``self_device_time_total`` when the torch version has it (2.1 does), otherwise the
        older ``self_cuda_time_total``, otherwise zero.
    """
    value = getattr(event, "self_device_time_total", None)
    if value is None:
        value = getattr(event, "self_cuda_time_total", 0)
    return float(value or 0.0)


def _profile(step, with_stack: bool):
    """Run the step once under the profiler.

    Args:
        step: Zero-argument callable performing one decode step.
        with_stack: Record Python stacks. Costs an order of magnitude in trace size and is
            not supported on every backend, which is why the caller decides and the failure
            is reported rather than raised.

    Returns:
        The ``prof`` object, or ``None`` when profiling is unavailable.
    """
    import torch
    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU]
    if hasattr(ProfilerActivity, "CUDA"):
        activities.append(ProfilerActivity.CUDA)
    try:
        prof = profile(activities=activities, with_stack=with_stack)
        with prof:
            step()
            torch.cuda.synchronize()
        return prof
    except Exception as exc:                      # noqa: BLE001 - trace is the whole point
        print(f"  profiler failed (with_stack={with_stack}): "
              f"{type(exc).__name__}: {exc}", flush=True)
        return None


def _print_inventory(prof) -> tuple[int, float]:
    """Print the complete per-name launch inventory.

    Args:
        prof: A finished profiler object.

    Returns:
        ``(total_calls, total_device_ms)`` for the profiled step.
    """
    rows = []
    total_us = 0.0
    total_calls = 0
    for event in prof.key_averages():
        cuda_us = _device_us(event)
        if not cuda_us:
            continue
        calls = int(event.count) // PROFILE_STEPS
        total_us += cuda_us
        total_calls += calls
        rows.append((calls, cuda_us / 1000.0 / PROFILE_STEPS, event.key, cuda_us / calls))
    rows.sort(key=lambda r: -r[1])

    print(f"  {'kernel':<52} {'calls':>6} {'ms':>8} {'us/call':>8} {'%':>5}", flush=True)
    print(f"  {'-' * 52} {'-' * 6} {'-' * 8} {'-' * 8} {'-' * 5}", flush=True)
    for calls, ms, key, us_per in rows:
        share = 100 * ms / (total_us / 1000.0) if total_us else 0.0
        print(f"  {key[:52]:<52} {calls:>6} {ms:>8.3f} {us_per:>8.1f} {share:>4.0f}%",
              flush=True)
    print(f"  {'-' * 52} {'-' * 6} {'-' * 8} {'-' * 8} {'-' * 5}", flush=True)
    print(f"  {'TOTAL':<52} {total_calls:>6} {total_us / 1000.0 / PROFILE_STEPS:>8.3f}",
          flush=True)
    return total_calls, total_us / 1000.0 / PROFILE_STEPS


def _print_shapes(prof) -> None:
    """Print the per-name breakdown by launch signature, so shapes are visible.

    Args:
        prof: A finished profiler object.
    """
    rows = []
    for event in prof.key_averages(group_by_input_shape=True):
        cuda_us = _device_us(event)
        if not cuda_us:
            continue
        calls = int(event.count) // PROFILE_STEPS
        inputs = getattr(event, "input_shapes", None)
        # The signature is carried on the key by group_by_input_shape; keeping it short is
        # what makes the table readable, and a shape that is empty still names the kernel.
        rows.append((calls, cuda_us / 1000.0 / PROFILE_STEPS, event.key[:64], str(inputs)))
    # Only the families that show up more than once: a single-launch family is already
    # visible in the name table above and would bury the interesting rows.
    rows = [r for r in rows if r[0] > 1]
    rows.sort(key=lambda r: -r[1])
    print(f"\n  {'kernel':<64} {'calls':>6} {'ms':>8}  signature", flush=True)
    for calls, ms, key, signature in rows[:40]:
        print(f"  {key:<64} {calls:>6} {ms:>8.3f}  {signature[:70]}", flush=True)


def _print_stacks(prof) -> bool:
    """Print launches grouped by Python call site.

    Args:
        prof: A finished profiler object recorded with ``with_stack=True``.

    Returns:
        ``True`` when at least one row carried a usable stack, so the caller can say which
        of "no stacks recorded" and "no launches" it is looking at.
    """
    try:
        grouped = prof.key_averages(group_by_stack_n=4)
    except Exception as exc:                      # noqa: BLE001 - unsupported on some builds
        print(f"  stack grouping unavailable: {type(exc).__name__}: {exc}", flush=True)
        return False
    rows = []
    for event in grouped:
        cuda_us = _device_us(event)
        if not cuda_us:
            continue
        stack = getattr(event, "stack", "") or ""
        # The bottom frames are the harness (this file, torch's profiler); the last frame
        # that is not them is the site worth printing.
        frames = [f.strip() for f in stack.split(";") if f.strip()]
        site = " <- ".join(frames[-3:]) if frames else ""
        rows.append((int(event.count) // PROFILE_STEPS,
                     cuda_us / 1000.0 / PROFILE_STEPS, event.key[:40], site))
    rows.sort(key=lambda r: -r[1])
    usable = [r for r in rows if r[3]]
    print(f"\n  {'kernel':<40} {'calls':>6} {'ms':>8}  call site", flush=True)
    for calls, ms, key, site in rows[:30]:
        print(f"  {key:<40} {calls:>6} {ms:>8.3f}  {site[:110]}", flush=True)
    return bool(usable)


def main() -> int:
    """Inventory one int8 decode step and print the greedy continuation.

    Returns:
        ``0`` on success, ``1`` when the model directory is not set.
    """
    import torch

    import backends
    from engine_api import create_model, load_weights

    print(f"  python {sys.version.split()[0]}  host={os.uname().nodename}",
          flush=True)
    print(f"  backend: {backends.status()}", flush=True)
    if not MODEL or not Path(MODEL).is_dir():
        print(f"  ENGINE_MODEL unset or not a directory: {MODEL!r}", flush=True)
        return 1

    sd = load_weights(MODEL, quant=True, vision=False)
    model = create_model(MODEL, max_seq=1024, bench=False, weights=sd)

    print("\n=== greedy continuation (must not change) ===", flush=True)
    model.reset()
    logits = model.prefill([1, 2, 3, 4, 5])
    torch.cuda.synchronize()
    ids = []
    probe = logits
    for _ in range(GREEDY_TOKENS):
        nxt = int(probe.argmax())
        ids.append(nxt)
        probe = model.logits(nxt)
        torch.cuda.synchronize()
    print(f"  ids: {ids}", flush=True)

    state = {"tok": 0}

    def prepare() -> None:
        """Put the model at a fresh prompt, ready to decode."""
        model.reset()
        logits_now = model.prefill([1, 2, 3, 4, 5])
        torch.cuda.synchronize()
        state["tok"] = int(logits_now.argmax())

    def step() -> None:
        """One eager decode step."""
        state["tok"] = int(model.logits(state["tok"]).argmax())

    prepare()
    for _ in range(5):
        step()
    torch.cuda.synchronize()

    print("\n=== wall clock ===", flush=True)
    prepare()
    ms = _bench(step, STEPS, 5)
    print(f"  decode step: {ms:.3f} ms  ({1000 / ms:.2f} tok/s)", flush=True)
    print(f"  BASELINE step_ms={ms:.3f} tok_s={1000 / ms:.2f}", flush=True)

    print("\n=== 1. complete inventory by kernel name ===", flush=True)
    prepare()
    prof = _profile(step, with_stack=False)
    if prof is None:
        print("  no profiler data", flush=True)
        print("\nLAUNCH_INVENTORY_DONE", flush=True)
        return 0
    calls, busy_ms = _print_inventory(prof)
    floor_us = ms * 1000.0 / calls if calls else 0.0
    print(f"\n  launches/step={calls}  device_ms={busy_ms:.2f}  wall={ms:.2f}  "
          f"wall/launch={floor_us:.1f} us", flush=True)
    print(f"  if every launch costs the wall-average, removing one launch is worth "
          f"{floor_us:.1f} us", flush=True)

    print("\n=== 2. by launch signature ===", flush=True)
    _print_shapes(prof)

    print("\n=== 3. by Python call site ===", flush=True)
    prepare()
    prof2 = _profile(step, with_stack=True)
    if prof2 is None:
        print("  stack-recording profile unavailable; sections 1-2 still stand", flush=True)
    elif not _print_stacks(prof2):
        print("  no stacks recorded by this build; use sections 1-2", flush=True)

    print("\nLAUNCH_INVENTORY_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
