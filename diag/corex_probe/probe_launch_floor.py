"""Measure what one kernel launch costs on this card, with the profiler out of the way.

Every plan to remove launches from the decode step rests on one number: how much wall clock
a launch costs when it does almost no work. `PERF_COREX.md` §8 previously concluded "device
bound, removing launches cannot help" from a trace whose per-kernel durations summed to 34 ms
against a 23 ms wall. That reading does not survive scrutiny: the trace *over-counts*, and it
over-counts per kernel, so its inflation grows with the launch count rather than the work.
A step whose trace exceeds its wall by 45% while issuing 2407 launches is consistent with a
device that spends most of its time starting work, not doing it.

The profiler cannot settle that, because it is the thing adding the overhead. This probe
measures the launch floor with device events instead:

1. **A trivial launch chain.** ``t.add_(1)`` on 128 floats, in a loop. The arithmetic is
   nanoseconds; what is left is the launch. Dividing gives microseconds per launch.
2. **The same at several element counts**, so the floor can be separated from the marginal
   cost of bytes rather than assumed to be one or the other.
3. **The split-K reduce kernel, alone, at the real shapes.** It is the single clearest waste
   in the step: 128 launches to add up 32 floats each. Its measured cost against the launch
   floor says whether removing it is worth a kernel change.
4. **A representative glue chain** (``amax``/``div``/``round``/``clamp_``/``to``, i.e. the
   int8 quantisation of k/v) at the real tensor sizes.

Module: ``coffin/diag/corex_probe``. Runs on the card. No model, so it is cheap enough to
re-run after every dispatch change.

Constraints:
    * Device events around a loop, with warmup, so the first launch's lazy module load is not
      in the measurement.
    * The launch floor is measured rather than quoted. It is a property of this driver and
      this kernel pair, and a datasheet number would decide a refactor on someone else's card.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

#: Launches per timing sample. Large enough that one event pair is not the noise floor,
#: small enough that a wedged card still gives up within the job.
LAUNCHES = 2000

#: Samples averaged per measurement.
SAMPLES = 5


def _time_launches(fn, launches: int, samples: int = SAMPLES) -> float:
    """Time a callable that issues one launch, in microseconds per launch.

    Args:
        fn: Zero-argument callable issuing exactly one kernel launch.
        launches: Launches per sample.
        samples: Samples to average.

    Returns:
        Mean microseconds per launch, sync excluded from the measurement by recording the
        events on the device timeline around the whole loop.
    """
    import torch

    # One warmup pass: the first launch of a kernel loads its module and allocates its
    # argument buffer, which is a millisecond and would otherwise land in the first sample.
    for _ in range(10):
        fn()
    torch.cuda.synchronize()

    best = float("inf")
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(launches):
            fn()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) * 1000.0 / launches)
    # The minimum, not the mean: the question is what a launch costs when nothing interferes,
    # and a slow sample only says another process was on the card.
    return best


def main() -> int:
    """Run the launch-floor measurements.

    Returns:
        ``0`` on success.
    """
    import torch

    print(f"  python {sys.version.split()[0]}  host={os.uname().nodename}", flush=True)
    print(f"  torch {torch.__version__}  device {torch.cuda.get_device_name(0)}", flush=True)

    print("\n=== 1. the launch floor ===", flush=True)
    print(f"  {'work':<44} {'us/launch':>10}", flush=True)
    rows = []
    for n in (1, 128, 1024, 8192, 65536, 262144, 1048576):
        t = torch.zeros(n, dtype=torch.bfloat16, device="cuda")
        us = _time_launches(lambda: t.add_(1), LAUNCHES)
        rows.append((n, us))
        bytes_moved = n * 2
        gbs = 2 * bytes_moved / (us * 1e-6) / 1e9
        print(f"  add_ bf16 x{n:<29} {us:>10.2f}   ({gbs:>7.1f} GB/s effective)",
              flush=True)
        del t
        torch.cuda.empty_cache()
    floor_us = rows[0][1]
    print(f"\n  launch floor (1 element): {floor_us:.2f} us", flush=True)

    # Where the floor stops mattering: the element count at which the kernel's own work
    # overtakes it. Anything below that crossover is launch, and fusing it is the only win.
    crossover = None
    for n, us in rows:
        if us > 2 * floor_us:
            crossover = n
            break
    if crossover is not None:
        print(f"  a kernel crosses 2x the floor at n={crossover}", flush=True)

    print("\n=== 2. what the split-K reduce costs inside the real op ===", flush=True)
    # The reduce kernel is not bound on its own; it is the tail of ``gemv_kn_i8``. Measuring
    # the op at ``split=1`` and at the split the step actually picks separates the two: the
    # difference is the reduce's launch plus its pass over ``[split, N]`` fp32. That is the
    # number a fused reduce has to beat, and it is measured at the shapes the step calls.
    import backends

    print(f"  backend: {backends.status()}", flush=True)
    module = backends.corex._kernels()
    print(f"  {'N':>8} {'K':>6} {'split':>6} {'us':>9} {'us s=1':>9} {'reduce us':>10}",
          flush=True)
    total_reduce = 0.0
    # (N, K, split the step picks, calls per step) from probe_shapes + pick_split_i8.
    for n, k, split, calls in ((18432, 2560, 14, 32), (2560, 9216, 32, 32),
                               (12352, 2560, 21, 24), (2560, 4096, 32, 32),
                               (10240, 2560, 26, 8), (248320, 2560, 8, 1)):
        wt = torch.randint(-127, 127, (k, n), dtype=torch.int8, device="cuda")
        x = torch.randn(k, dtype=torch.bfloat16, device="cuda")
        scale = torch.rand(n, dtype=torch.float32, device="cuda")
        us_split = _time_launches(
            lambda: module.gemv_kn_i8(wt, x, scale, 0, 0, split, 0), LAUNCHES // 8)
        us_plain = _time_launches(
            lambda: module.gemv_kn_i8(wt, x, scale, 0, 0, 1, 0), LAUNCHES // 8)
        reduce_us = us_split - us_plain
        total_reduce += reduce_us * calls
        print(f"  {n:>8} {k:>6} {split:>6} {us_split:>9.2f} {us_plain:>9.2f} "
              f"{reduce_us:>10.2f}   x{calls}", flush=True)
        del wt, x, scale
        torch.cuda.empty_cache()
    print(f"\n  reduce share of the step (128 launches): "
          f"{total_reduce / 1000.0:.3f} ms", flush=True)
    print(f"  fusing it removes {total_reduce / floor_us:.0f} launches' worth of floor",
          flush=True)

    print("\n=== 3. a glue chain (one layer's k/v int8 quantisation) ===", flush=True)
    # The chain is amax / clamp_min / div / round / clamp_ / to, on [4, 256] k or v. Six
    # launches per tensor and two tensors per full-attention layer.
    x = torch.randn(4, 256, dtype=torch.bfloat16, device="cuda")
    budget = 1e-5

    def quant_chain() -> None:
        """The six-launch quantisation chain ``_quant`` performs."""
        s = x.float().abs().amax(-1).clamp_min(budget) / 127.0
        q = (x.float() / s[:, None]).round().clamp_(-127, 127).to(torch.int8)
        del q

    chain_us = _time_launches(quant_chain, LAUNCHES // 8)
    print(f"  six-launch quant chain: {chain_us:.2f} us "
          f"({chain_us / 6:.2f} us per launch)", flush=True)
    print(f"  a fused version would cost about {floor_us:.2f} us", flush=True)
    print(f"  so fusing it saves about {(chain_us - floor_us) / 6:.2f} us per launch "
          f"removed", flush=True)

    print("\n=== 4. what the step is, given this floor ===", flush=True)
    # The step issues ~2407 launches (probe_launch_inventory, same session). If the mean launch
    # is at the floor, the floor alone accounts for this much of the 23.5 ms wall.
    step_ms = float(os.environ.get("STEP_MS", "23.527") or 23.527)
    launches = int(os.environ.get("STEP_LAUNCHES", "2407") or 2407)
    print(f"  step wall {step_ms:.3f} ms over {launches} launches "
          f"= {step_ms * 1000 / launches:.2f} us per launch", flush=True)
    print(f"  floor {floor_us:.2f} us x {launches} = "
          f"{floor_us * launches / 1000:.2f} ms "
          f"({100 * floor_us * launches / 1000 / step_ms:.0f}% of the wall)", flush=True)

    print("\nLAUNCH_FLOOR_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
