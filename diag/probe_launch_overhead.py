"""How much does it cost to launch a kernel on this part?

Everything measured so far behaves as if kernels were expensive rather than the
work inside them: the delta scan spends 13 ms on 190 tiny ops, and a prefill of
2048 tokens issues tens of thousands of launches.  This measures the per-launch
cost directly (eager vs captured), and then checks how much of a prefill is host
launch time rather than device time.

Run:  python probe_launch_overhead.py [n]
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, "/root/private_data/engine")

import torch

import qwen35_engine as E

MODEL = "/root/private_data/models/Qwen3.5-4B"
BASE = [9707, 11, 1879, 3671, 374, 264, 3449, 315, 9847, 13,
        3838, 374, 279, 6722, 315, 9625, 323, 11751, 13]


def timeit(fn, iters=20, warmup=5):
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
    """Wall time per call with no sync inside: pure host submission cost."""
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
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    x = torch.zeros(1024, device="cuda")
    y = torch.zeros(1024, device="cuda")

    def one():
        x.add_(1.0)

    def many(k=100):
        for _ in range(k):
            x.add_(1.0)

    def many_ops(k=100):
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

    model = E.Qwen35(MODEL, max_seq=4096, bench=False)
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
