"""Does the model-level prefill chunk (the GEMM's M) matter?

``Qwen35.prefill(tokens, chunk=256)`` splits the prompt into 256-token model
chunks, so every projection runs at M=256.  hipBLAS/rocBLAS efficiency usually
depends on M, and rewriting the weights per chunk costs bandwidth, so this
sweeps the chunk size with no code change.

Run:  python probe_prefill_chunk.py [n ...]

Module: ``coffin/diag``. Sweeps the ``chunk`` argument of
``Qwen35.prefill``, which is also ``EngineConfig.chunk``.

Constraints:
    * No code change: the chunk size is a config knob, so the sweep must not
      rebuild the weights or re-capture the graph between points.
    * rocBLAS tops out around M=2048 on this card, so a larger chunk should show
      no gain -- if it does, the measurement is off.
"""

from __future__ import annotations

import sys

import torch

from models.qwen35 import Qwen35

MODEL = "/root/private_data/models/Qwen3.5-4B"
BASE = [9707, 11, 1879, 3671, 374, 264, 3449, 315, 9847, 13,
        3838, 374, 279, 6722, 315, 9625, 323, 11751, 13]


def main() -> None:
    sizes = [int(a) for a in sys.argv[1:]] or [512, 2048]
    model = Qwen35(MODEL, max_seq=4096, bench=False)
    for n in sizes:
        toks = [BASE[i % len(BASE)] for i in range(n)]
        print(f"\n=== n={n} ===")
        print(f"{'chunk':>7}{'ms':>10}{'ms/token':>11}{'tok/s':>10}{'vs best':>9}")
        base = None
        for ch in (128, 256, 512, 1024, n):
            if ch > n:
                continue
            model.reset()
            model.prefill(toks, chunk=ch)                     # untimed warmup
            torch.cuda.synchronize()
            best = float("inf")
            for _ in range(2):
                model.reset()
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                model.prefill(toks, chunk=ch)
                e.record()
                torch.cuda.synchronize()
                best = min(best, s.elapsed_time(e))
            base = best if base is None else base
            print(f"{ch:>7}{best:>10.1f}{best / n:>11.3f}{n / best * 1e3:>10.0f}"
                  f"{best / base:>8.2f}x")


if __name__ == "__main__":
    main()
