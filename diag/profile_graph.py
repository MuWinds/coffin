"""Profile CUDA-graph decode replays: which kernels own the GPU now that the
CPU is out of the loop.

Run:  python profile_graph.py [n]
"""

from __future__ import annotations

import sys
import time

import torch

import qwen35_engine as E

MODEL = "/root/private_data/models/Qwen3.5-4B"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    model = E.Qwen35(MODEL, max_seq=2048, bench=True)
    model.capture()
    model.reset()

    toks = [9707] * n
    t0 = time.perf_counter()
    for t in toks:
        model.decode_step(t)
    torch.cuda.synchronize()
    print(f"graph: {(time.perf_counter()-t0)/n*1000:.2f} ms/token ({n/(time.perf_counter()-t0):.2f} tok/s)")

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for t in toks:
            model.decode_step(t)
        torch.cuda.synchronize()

    ka = prof.key_averages()
    rows = sorted(ka, key=lambda r: r.self_device_time_total, reverse=True)[:16]
    total = sum(r.self_device_time_total for r in ka)
    print(f"{'kernel':<52}{'calls':>7}{'us/step':>10}{'%':>7}")
    for r in rows:
        print(f"{r.key[:50]:<52}{r.count//n:>7}{r.self_device_time_total/n:>10.1f}"
              f"{100*r.self_device_time_total/total:>7.1f}")
    print(f"{'TOTAL':<52}{'':>7}{total/n:>10.1f}")


if __name__ == "__main__":
    main()
