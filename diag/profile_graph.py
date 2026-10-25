"""Profile CUDA-graph decode replays: which kernels own the GPU now that the
CPU is out of the loop.

Answers what the decode step is device-bound on once launch overhead and Python
have been removed by graph capture. That distinction is the whole point: in an
eager step a large host share hides the kernel mix, so the graph replay is the
only place the real GPU cost distribution is visible.

Read the table as ``us/step`` per kernel against the ``TOTAL`` line, and take
``calls`` as a shape hint — a kernel with many calls per step but little time is
a fusion candidate, while one with few calls and a large share is the bandwidth
target. The ``ms/token`` line above the table is the end-to-end figure those
kernels have to explain; if the table's total is well under it, time is still
going somewhere capture did not cover.

Run inside the container from the engine root, on the DCU:

    python profile_graph.py [n]

``n`` is the number of decode tokens to replay, default 30.

Module: ``coffin/diag``. Standalone probe: imports ``models.qwen35`` and torch,
and nothing under ``coffin/`` imports it.

Constraints:
    * ``capture()`` must run before the timed loop; an uncaptured model would
      measure the eager path and report a different kernel mix entirely.
    * The same token id is fed ``n`` times deliberately. The measurement is the
      per-step kernel mix, and repeated input keeps the branch pattern fixed so
      two runs compare like with like.
    * Needs the real checkpoint, so it runs only where the weights are.
"""

from __future__ import annotations

import sys
import time

import torch

from models.qwen35 import Qwen35

MODEL = "/root/private_data/models/Qwen3.5-4B"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    model = Qwen35(MODEL, max_seq=2048, bench=True)
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
