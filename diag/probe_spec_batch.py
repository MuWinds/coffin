"""Batched speculative decoding: per-step cost across prompt lengths.

Two things to watch:

* short prompt (96): the step is projection-bound.  Greedy reads the weights once
  per step; the token verify reads them ``c`` times unless the one-forward verify
  (``--once``) lands the wide-M projection; the acceptance rate sets how many
  tokens that buy.
* long context (4k / 32k): the verify additionally re-reads the row's history KV
  ``c`` times, which is what the multi-query attention kernel (task-12) removes.
  The crossing point between "projection fixed cost" and "KV read" is the point
  of this probe.

Run (inside the container, from the engine directory)::

    python3 diag/probe_spec_batch.py --lengths 96 --batch 8 --steps 16
    python3 diag/probe_spec_batch.py --lengths 96,4096 --batch 2 --steps 8
    python3 diag/probe_spec_batch.py --lengths 4096 --batch 1 --once
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
BUCKETS = (1, 2, 4, 8)
BASE = [9707, 11, 1879, 3671, 374, 264, 3449, 315, 9847, 13,
        3838, 374, 279, 6722, 315, 9625, 323, 11751, 13]


def tokens(n: int, off: int = 0) -> list[int]:
    return [BASE[(i + off) % len(BASE)] for i in range(n)]


def alloc(pool, n_tokens: int) -> list[int]:
    return [pool.alloc() for _ in range(-(-(n_tokens + 2 * 16) // BLOCK))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lengths", default="96")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--once", action="store_true",
                    help="use the one-forward verify (task-13) instead of c token steps")
    ap.add_argument("--max-seq", type=int, default=0,
                    help="default: max(lengths)+K+steps")
    ap.add_argument("--blocks", type=int, default=0, help="0 = size the pool for the run")
    args = ap.parse_args()

    lengths = [int(x) for x in args.lengths.split(",") if x]
    cap = args.batch
    max_len = max(lengths)
    max_seq = args.max_seq or (max_len + args.k + args.steps + 2 * BLOCK)
    need_positions = max_seq + 2 * BLOCK
    blocks = args.blocks or (cap * (-(-need_positions // BLOCK)) + cap)

    print(f"device={torch.cuda.get_device_name(0)}")
    weights = load_weights(MODEL, quant=True)
    model = create_model(MODEL, max_seq=max_seq, bench=True, weights=weights,
                         paged=True)
    pool = model.enable_paging(num_blocks=blocks, block_size=BLOCK)
    model.enable_batching(cap + 1)
    model.capture_batch(BUCKETS)
    t0 = time.perf_counter()
    model.enable_spec_batch(args.k, buckets=BUCKETS,
                            max_drafter_len=max(16384, max_seq),
                            verify_mode=("once" if args.once else None))
    torch.cuda.synchronize()
    mode = model.verify_mode
    print(f"enable_spec_batch(K={args.k}, verify={mode}): "
          f"{time.perf_counter() - t0:.2f} s, "
          f"allocated {torch.cuda.memory_allocated()/2**20:.0f} MiB, "
          f"free {torch.cuda.mem_get_info()[0]/2**20:.0f} MiB")

    bucket = next(b for b in BUCKETS if b >= cap)
    for length in lengths:
        prompts = [tokens(length, off=7 * i) for i in range(cap)]
        blocks_of = [alloc(pool, len(p)) for p in prompts]
        seqs = [SimpleNamespace(token_ids=p, row=r, block_table=b)
                for r, (p, b) in enumerate(zip(prompts, blocks_of))]
        snap = 512 if length > 2048 else None         # bound prefill activations
        model.reset()
        logits = torch.stack([lg.float() for lg in
                              model.prefill_group(seqs, snapshot_interval=snap)[0]])
        rows_bucket = bucket

        # Greedy decode of the same prompts, for comparison.
        tok = [int(t) for t in logits.argmax(-1).tolist()]
        pos = [len(p) for p in prompts]
        produced = 0
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for s in range(args.steps):
            model.set_rows(tok, [pos[i] + s for i in range(cap)], blocks_of)
            lgs = model.decode_batch(cap, rows_bucket).float()
            tok = [int(t) for t in lgs.argmax(-1).tolist()]
            produced += cap
        torch.cuda.synchronize()
        t_greedy = time.perf_counter() - t0

        # Speculative run, timed the same way.
        def spec_run():
            model.reset()
            lg = torch.stack([t.float() for t in
                              model.prefill_group(seqs, snapshot_interval=snap)[0]])
            cur = [int(t) for t in lg.argmax(-1).tolist()]
            pos = [len(p) for p in prompts]
            produced = accepted = drafted = 0
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for s in range(args.steps):
                toks, logits_next, acc = model.spec_step_batch(
                    cur, pos, blocks_of, rows_bucket, args.k)
                for i in range(cap):
                    produced += len(toks[i])
                    pos[i] += len(toks[i])
                    cur[i] = int(logits_next[i].argmax())
                accepted += sum(acc)
                drafted += cap * args.k
            torch.cuda.synchronize()
            return time.perf_counter() - t0, produced, accepted, drafted

        g_tps = (args.steps * cap) / t_greedy
        t_spec, produced, accepted, drafted = spec_run()

        def report(name, t, prod, acc, dr):
            tps = prod / t
            print(f"  {name:6s}: {t*1e3/args.steps:8.2f} ms/step  {tps:8.1f} tok/s  "
                  f"({prod/(args.steps*cap):.3f} tok/row/step, "
                  f"accept {100.0*acc/max(dr,1):.1f}%)  "
                  f"{tps/g_tps:.2f}x")
            return tps

        print(f"\nlength={length} batch={cap} K={args.k}")
        print(f"  greedy: {t_greedy*1e3/args.steps:8.2f} ms/step  {g_tps:8.1f} tok/s")
        report(f"once", t_spec, produced, accepted, drafted)
        if args.once:
            # same step with the c-token verify, for the curve comparison
            model.set_verify_mode("token")
            t_tok, prod_tok, acc_tok, dr_tok = spec_run()
            report("token", t_tok, prod_tok, acc_tok, dr_tok)
            model.set_verify_mode("once")
            model.reset()
            model.prefill_group(seqs, snapshot_interval=snap)
        for b in blocks_of:
            pool.free_blocks(b)
    print(f"\nmemory: allocated {torch.cuda.memory_allocated()/2**20:.0f} MiB, "
          f"reserved {torch.cuda.memory_reserved()/2**20:.0f} MiB, "
          f"free {torch.cuda.mem_get_info()[0]/2**20:.0f} MiB")


if __name__ == "__main__":
    main()
