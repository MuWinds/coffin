"""Prefill throughput: batched path vs the one-token path, across prompt sizes.

Answers how much the batched prefill kernel is worth over feeding tokens through
the single-token path one at a time — the phase-1 fallback. This is the
end-to-end number that ``probe_prefill_split.py`` then attributes to parts.

Read the ``speedup`` column as a function of ``n``: it grows with prompt length
because the one-token path pays its per-call fixed cost n times, so a flat
speedup across sizes means the fixed cost is not what the batch path is saving.
``batch tok/s`` is the throughput to quote; it is the figure a serving decision
should use.

Run inside the container from the engine root, on the DCU:

    python bench_prefill2.py

Module: ``coffin/diag``. Standalone probe: imports ``models.qwen35`` and torch,
and nothing under ``coffin/`` imports it.

Constraints:
    * Every path is warmed once before timing (rocBLAS handles, the memory pool,
      the solver), because a cold first call would otherwise land in the ``n=32``
      row and make the smallest prompt look the slowest.
    * The timed regions include the prefill only, with ``reset`` outside the
      timer where possible, so the column measures prefill and not teardown.
    * Needs the real checkpoint, so it runs only where the weights are.
"""

from __future__ import annotations

import time

import torch

from models.qwen35 import Qwen35

MODEL = "/root/private_data/models/Qwen3.5-4B"


def main():
    print(f"device={torch.cuda.get_device_name(0)}")
    model = Qwen35(MODEL, max_seq=4096, bench=False)
    base = [9707, 11, 1879, 3671, 374, 264, 3449, 315, 9847, 13,
            3838, 374, 279, 6722, 315, 9625, 323, 11751, 13]

    # warm up every code path once (rocBLAS handles, memory pool, solver)
    for n in (16, 64, 256):
        model.reset()
        model.prefill([base[i % len(base)] for i in range(n)])
    model.reset()
    for t in base:
        model.logits(t)
    torch.cuda.synchronize()

    print(f"{'n':>6}{'seq ms':>10}{'seq tok/s':>11}{'batch ms':>11}{'batch tok/s':>13}{'speedup':>9}")
    for n in (32, 256, 2048):
        toks = [base[i % len(base)] for i in range(n)]

        model.reset()
        t0 = time.perf_counter()
        for t in toks:
            model.logits(t)
        torch.cuda.synchronize()
        t_seq = time.perf_counter() - t0

        model.reset()
        t0 = time.perf_counter()
        model.prefill(toks)
        torch.cuda.synchronize()
        t_bat = time.perf_counter() - t0

        print(f"{n:>6}{t_seq*1000:>10.1f}{n/t_seq:>11.0f}{t_bat*1000:>11.1f}"
              f"{n/t_bat:>13.0f}{t_seq/t_bat:>9.1f}x")


if __name__ == "__main__":
    main()
