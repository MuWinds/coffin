"""Per-step decode kernel table, for both weight modes, from the captured graph.

Decode work is bandwidth (weight reads) plus a long tail of small elementwise
kernels; which of the two dominates decides which fusion is worth writing.  The
graph is captured so the CPU is out of the loop, and the table reports calls per
step as well as time per step -- with 1.8 us per kernel inside a graph, 2500
kernels cost ~4.5 ms before any traffic at all.

Run:  python probe_decode_profile.py [steps]
"""

from __future__ import annotations

import sys

sys.path.insert(0, "/root/private_data/engine")

import torch

import qwen35_engine as E

MODEL = "/root/private_data/models/Qwen3.5-4B"
D = torch.bfloat16


def run(model, n, label):
    model.capture()
    model.reset()
    toks = [9707] * n
    model.decode_step(toks[0])
    torch.cuda.synchronize()
    import time
    t0 = time.perf_counter()
    for t in toks:
        model.decode_step(t)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / n * 1000

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for t in toks:
            model.decode_step(t)
        torch.cuda.synchronize()
    ka = list(prof.key_averages())
    total = sum(r.self_device_time_total for r in ka)
    calls = sum(r.count for r in ka)
    rows = sorted(ka, key=lambda r: r.self_device_time_total, reverse=True)[:18]
    print(f"\n=== {label}: {ms:.2f} ms/step ({1000 / ms:.1f} tok/s), "
          f"{calls // n} kernels/step, {total / n:.1f} us device ===")
    print(f"{'kernel':<56}{'x/step':>8}{'us/step':>9}{'%':>7}")
    for r in rows:
        print(f"{r.key[:54]:<56}{r.count // n:>8}{r.self_device_time_total / n:>9.1f}"
              f"{100 * r.self_device_time_total / total:>7.1f}")
    print(f"{'TOTAL':<56}{calls // n:>8}{total / n:>9.1f}")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    print(f"device={torch.cuda.get_device_name(0)}")
    m = E.Qwen35(MODEL, max_seq=2048, bench=True)
    run(m, n, "fp16 weights (default)")
    del m
    torch.cuda.empty_cache()

    sd = E.load_weights(MODEL, quant=True)
    m8 = E.Qwen35(MODEL, max_seq=2048, bench=True, weights=sd)
    run(m8, n, "int8 weights (W8A16)")


if __name__ == "__main__":
    main()
