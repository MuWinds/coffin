"""Group prefill throughput: k prompts in one forward vs k forwards.

The batched engine admits several requests per step; preflighting them one by one
pays the full weight read (4.2 GB int8 -> fp16 dequant per layer) once per
request.  ``Qwen35.prefill_group`` shares the projection/MLP stages over the
concatenated prompts, so the prompt cost should approach the cost of *one*
forward as k grows.  This probe measures that, checks the group answer agrees
with the serial one (argmax), and prints the fixed cost per forward.

Run (inside the container, from the engine directory)::

    python3 diag/probe_prefill_batch.py --length 128

Module: ``coffin/diag``. Exercises ``Qwen35.prefill_group`` through the batched
engine in ``llm_engine``.

Constraints:
    * The comparison is k prompts in one forward against k separate forwards, on
      the same weights, in the same process -- otherwise the weight read dominates
      the difference.
    * Prompt content must be distinct: identical prompts would hit the radix tree
      and measure reuse instead of grouping.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from engine_api import create_model, load_weights

MODEL = os.environ.get("SCNET_MODEL", "/root/private_data/models/Qwen3.5-4B")
BLOCK = 64
BASE = [9707, 11, 1879, 3671, 374, 264, 3449, 315, 9847, 13,
        3838, 374, 279, 6722, 315, 9625, 323, 11751, 13]


def tokens(n: int, off: int = 0) -> list[int]:
    return [BASE[(i + off) % len(BASE)] for i in range(n)]


def alloc(pool, n_tokens: int) -> list[int]:
    return [pool.alloc() for _ in range(-(-(n_tokens + BLOCK) // BLOCK))]


def reset_staging(model) -> None:
    model.pos1.zero_()
    model.zero_linear_states(model.prefill_row)


def serial_prefill(model, pool, prompts) -> float:
    """k separate prefills (what the current scheduler does), returns seconds."""
    blocks = [alloc(pool, len(p)) for p in prompts]
    t0 = time.perf_counter()
    for p, b in zip(prompts, blocks):
        reset_staging(model)
        pool.assign_table(b, 0, row=model.prefill_row)
        model.prefill(p)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    for b in blocks:
        pool.free_blocks(b)
    return dt


def group_prefill(model, pool, prompts) -> float:
    """One forward for all prompts, returns seconds."""
    blocks = [alloc(pool, len(p)) for p in prompts]
    seqs = [SimpleNamespace(token_ids=p, row=r, block_table=b)
            for r, (p, b) in enumerate(zip(prompts, blocks))]
    t0 = time.perf_counter()
    model.prefill_group(seqs)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    for b in blocks:
        pool.free_blocks(b)
    return dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=128)
    ap.add_argument("--sizes", default="1,2,4,8")
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--blocks", type=int, default=256)
    args = ap.parse_args()

    print(f"device={torch.cuda.get_device_name(0)}")
    weights = load_weights(MODEL, quant=True)
    model = create_model(MODEL, max_seq=args.max_seq, bench=True, weights=weights,
                         paged=True)
    pool = model.enable_paging(num_blocks=args.blocks, block_size=BLOCK)
    model.enable_batching(9)

    sizes = [int(s) for s in args.sizes.split(",") if s]
    prompts = [tokens(args.length, off=7 * i) for i in range(max(sizes))]
    for k in sizes:                                  # warm every shape once
        serial_prefill(model, pool, prompts[:k])
        group_prefill(model, pool, prompts[:k])

    # correctness: same next-token argmax as the serial run
    ref_tokens = []
    for p in prompts[:max(sizes)]:
        reset_staging(model)
        b = alloc(pool, len(p))
        pool.assign_table(b, 0, row=model.prefill_row)
        ref_tokens.append(int(model.prefill(p).argmax()))
        pool.free_blocks(b)
    blocks = [alloc(pool, len(p)) for p in prompts[:max(sizes)]]
    seqs = [SimpleNamespace(token_ids=p, row=r, block_table=b)
            for r, (p, b) in enumerate(zip(prompts, blocks))]
    got = [int(lg.argmax()) for lg in model.prefill_group(seqs)[0]]
    for b in blocks:
        pool.free_blocks(b)
    same = sum(a == b for a, b in zip(ref_tokens, got))
    print(f"argmax agreement with serial prefill: {same}/{len(got)}")

    print(f"\nprompt length {args.length}, {max(sizes)} prompts loaded "
          f"(serial cost is the same forward repeated k times)")
    print(f"{'k':>2} | {'serial ms':>9} {'tok/s':>8} | {'group ms':>8} {'tok/s':>8} "
          f"| {'speedup':>7} | {'ms/forward saved':>16}")
    base_group = None
    for k in sizes:
        t_serial = serial_prefill(model, pool, prompts[:k])
        t_group = group_prefill(model, pool, prompts[:k])
        if base_group is None:
            base_group = t_group
        ntok = k * args.length
        print(f"{k:>2} | {t_serial * 1e3:9.1f} {ntok / t_serial:8.1f} | "
              f"{t_group * 1e3:8.1f} {ntok / t_group:8.1f} | "
              f"{t_serial / t_group:7.2f}x | "
              f"{(t_serial - t_group) * 1e3:14.1f} ms")
    print(f"\ngroup k=1 forward: {base_group * 1e3:.1f} ms; "
          f"group k={max(sizes)} costs {group_prefill(model, pool, prompts) * 1e3:.1f} ms "
          f"for {max(sizes)}x the prompt tokens")
    print(f"GPU memory: allocated {torch.cuda.memory_allocated() / 2**20:.0f} MiB, "
          f"reserved {torch.cuda.memory_reserved() / 2**20:.0f} MiB")


if __name__ == "__main__":
    main()
