"""How much does it cost to launch a kernel on this part?

Answers whether this engine is limited by device work or by launch submission.
Every measurement so far behaves as if kernels were expensive rather than the
work inside them: the delta scan spends 13 ms on 190 tiny ops, and a prefill of
2048 tokens issues tens of thousands of launches. This measures the per-launch
cost directly (eager vs captured), and then checks how much of a prefill is host
launch time rather than device time.

Read it as three comparisons. Eager versus per-launch device time says whether
the launch itself is visible to the device or only to the host. The graph-replay
row is the floor a captured decode step can reach: if replay is much cheaper per
launch, batching or capture is the fix, not kernel tuning. Finally,
``unaccounted (wall - device)`` in the prefill section is the host submission
time actually paid — a large value with low device time means the fix is fewer,
fatter kernels or a graph, and no amount of in-kernel optimisation will help.

Run inside the container from the engine root, on the DCU:

    python probe_launch_overhead.py [n]

``n`` is the prefill length, default 2048, passed as the chunk size as well so a
single chunk covers the whole prompt.

Module: ``coffin/diag``. Standalone probe: inserts the engine root on
``sys.path`` itself and imports ``models.qwen35``, so it runs from any directory
inside the container. Nothing under ``coffin/`` imports it.

Constraints:
    * ``sys.path.insert`` here is deliberate and must stay: this probe is run as
      a plain script from the container root, and the engine's modules use
      absolute imports that need ``engine/`` itself on the path.
    * Host timing deliberately does not sync inside the loop. It measures
      submission only, and adding a sync would turn it into a copy of the device
      measurement.
    * Reads the real checkpoint, so it needs the weights; the launch cost it
      reports is otherwise host-and-driver specific and does not transfer.
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, "/root/private_data/engine")

import torch

from models.qwen35 import Qwen35

MODEL = "/root/private_data/models/Qwen3.5-4B"
BASE = [9707, 11, 1879, 3671, 374, 264, 3449, 315, 9847, 13,
        3838, 374, 279, 6722, 315, 9625, 323, 11751, 13]


def timeit(fn, iters=20, warmup=5):
    """Time ``fn`` on the device with CUDA events and return milliseconds per call.

    Args:
        fn: Zero-argument callable to time.
        iters: Calls inside the timed region; the span is divided by this.
        warmup: Untimed calls first, to absorb kernel first-load.

    Returns:
        Mean device milliseconds per call across one region. A single region is
        enough here because the quantities being compared differ by orders of
        magnitude, not by percent.

    One region brackets a batch of calls between two events, so per-call launch
    gaps are included: that is the cost under investigation, not an artefact to
    divide away.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def host_timeit(fn, iters=20, warmup=5):
    """Wall time per call with no sync inside: pure host submission cost.

    Args:
        fn: Zero-argument callable to time.
        iters: Calls inside the timed region.
        warmup: Untimed calls first.

    Returns:
        Host microseconds per call. The single sync after the loop means
        asynchronous device work does not block submission, so the number is
        what the CPU spends queueing, which is the figure a launch-bound loop is
        limited by.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    dt = time.perf_counter() - t0
    torch.cuda.synchronize()
    return dt / iters * 1e3


def main():
    """Measure per-launch cost, then split a real prefill into host and device time."""
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    x = torch.zeros(1024, device="cuda")
    y = torch.zeros(1024, device="cuda")

    def one():
        """Issue a single elementwise launch, the smallest possible unit of work."""
        x.add_(1.0)

    def many(k=100):
        """Issue ``k`` identical elementwise launches to measure the repeat cost.

        Args:
            k: Launch count. Repeating one op keeps the dispatcher on a single
                cached kernel, which isolates the launch path from any per-shape
                dispatch cost.
        """
        for _ in range(k):
            x.add_(1.0)

    def many_ops(k=100):
        """Issue ``k`` differently shaped launches so nothing is cached or fused.

        Args:
            k: Launch count. Each call slices a different 8-element view, so the
                dispatcher and any fusion pass see a distinct op every time —
                this is the row that models a real prefill's launch mix.
        """
        # distinct op shapes so nothing can be cached or fused by the dispatcher
        for i in range(k):
            x[i:i + 8].add_(1.0)

    ms1 = timeit(one, iters=200, warmup=20)
    ms100 = timeit(many, iters=20)
    ms100v = timeit(many_ops, iters=20)
    print(f"device time for 1 elemwise launch     : {ms1 * 1e3:8.1f} us")
    print(f"device time for 100 elemwise launches : {ms100 * 1e3:8.1f} us "
          f"({ms100 * 10:6.1f} us/launch)")
    print(f"host   time for 100 elemwise launches : {host_timeit(many, iters=20) * 1e3:8.1f} us "
          f"({host_timeit(many, iters=20) * 10:6.1f} us/launch)")
    print(f"host   time for 100 varied launches   : {host_timeit(many_ops, iters=20) * 1e3:8.1f} us "
          f"({host_timeit(many_ops, iters=20) * 10:6.1f} us/launch)")

    # same again, captured: what a graph replay costs
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            many(100)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        many(100)
    print(f"graph reuse of 100 elemwise launches  : {timeit(lambda: g.replay(), iters=200) * 1e3:8.1f} us "
          f"({timeit(lambda: g.replay(), iters=200) * 10:6.1f} us/launch)")

    # The same measurement on a real prefill: host time versus device time.
    from torch.profiler import ProfilerActivity, profile

    model = Qwen35(MODEL, max_seq=4096, bench=False)
    toks = [BASE[i % len(BASE)] for i in range(n)]
    model.reset()
    model.prefill(toks, chunk=n)
    torch.cuda.synchronize()
    wall = timeit(lambda: (model.reset(), model.prefill(toks, chunk=n))[1], iters=3, warmup=1)
    host = host_timeit(lambda: (model.reset(), model.prefill(toks, chunk=n))[1], iters=3, warmup=1)
    print(f"\nprefill n={n} chunk={n}: wall {wall:.1f} ms, host-side submission {host:.1f} ms")

    model.reset()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        model.prefill(toks, chunk=n)
    torch.cuda.synchronize()
    dev = sum(ev.self_device_time_total for ev in prof.key_averages()) / 1e3
    launches = sum(ev.count for ev in prof.key_averages() if (ev.self_device_time_total or 0) > 0)
    print(f"  device total {dev:.1f} ms over {launches} kernel launches "
          f"({wall / launches * 1e3:.1f} us/launch wall)")
    print(f"  unaccounted (wall - device) = {wall - dev:.1f} ms")
    rows = [(ev.self_device_time_total, ev.count, ev.key) for ev in prof.key_averages()
            if (ev.self_device_time_total or 0) > 0]
    rows.sort(reverse=True)
    for t_us, cnt, key in rows[:12]:
        print(f"{t_us / 1e3:>9.2f} ms {cnt:>6}x  {key[:70]}")
    crows = [(ev.self_cpu_time_total, ev.count, ev.key) for ev in prof.key_averages()
             if (ev.self_cpu_time_total or 0) > 0]
    crows.sort(reverse=True)
    print("  top host-side ops:")
    for t_us, cnt, key in crows[:8]:
        print(f"{t_us / 1e3:>9.2f} ms {cnt:>6}x  {key[:70]}")


if __name__ == "__main__":
    main()
