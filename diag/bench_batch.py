"""Aggregate decode throughput of the static-batch engine.

Loads the checkpoint once, then drives ``N`` concurrent requests through
``LLMEngine`` and reports the aggregate tokens/s for ``N = 1, 2, 4, 8``:

    MKL_THREADING_LAYER=GNU PYTHONPATH=$PWD python3 diag/bench_batch.py
    ... --prompt-len 512 --max-new 64 --seqs 1,2,4,8

Why this is the number that matters: a decode step is bound by reading the
weights (~4 GB of int8 per step at ~700 GB/s), which one weight read serves for
every row of a batch.  The per-sequence latency barely moves until the attention
/ recurrent work catches up with the memory traffic, so what grows is the
aggregate throughput.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

import engine_api  # noqa: E402
from config import EngineConfig  # noqa: E402
from llm_engine import LLMEngine  # noqa: E402
from sequence import SamplingParams  # noqa: E402

MODEL = os.environ.get("ENGINE_MODEL", "/root/private_data/models/Qwen3.5-4B")


def run(engine: LLMEngine, prompts: list[list[int]], max_new: int) -> tuple[float, int]:
    """Submit every prompt, decode until all finish, return (seconds, tokens).

    The clock starts only after every request has been prefilled (and has
    produced its first token), so the number is the decode step rate the batching
    work is about -- not a mix of prefill and decode.
    """
    seqs = [engine.add_request(p, SamplingParams(max_tokens=max_new, temperature=0.0))
            for p in prompts]
    while not all(s.num_cached_tokens > 0 for s in seqs):
        engine.step()
    t0 = time.time()
    produced = sum(s.num_completion_tokens for s in seqs)
    while not all(s.is_finished for s in seqs):
        for _seq, tok, _done in engine.step():
            produced += tok is not None
    return time.time() - t0, produced


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--max-seq", type=int, default=4096)
    ap.add_argument("--prompt-len", type=int, default=256)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--seqs", default="1,2,4,8")
    ap.add_argument("--batches", default="1,2,4,8")
    ap.add_argument("--num-blocks", type=int, default=0,
                    help="KV pool blocks (0 = 8 x max_seq/block_size)")
    ap.add_argument("--int8", dest="int8", action="store_true", default=True)
    args = ap.parse_args()

    plan = [int(x) for x in args.seqs.split(",") if x.strip()]
    max_batch = max(int(x) for x in args.batches.split(",") if x.strip())
    blocks = args.num_blocks or 8 * (args.max_seq // 64)

    torch.cuda.init()
    print(f"loading {args.model} ...", flush=True)
    t0 = time.time()
    weights = engine_api.load_weights(args.model, quant=args.int8)
    model = engine_api.create_model(args.model, max_seq=args.max_seq, bench=False,
                                    weights=weights, paged=True)
    cfg = EngineConfig(model_dir=args.model, max_seq=args.max_seq, quant=args.int8,
                       prefix_cache=False, block_size=64, num_blocks=blocks,
                       max_num_seqs=max_batch,
                       snapshot_interval=64)
    engine = LLMEngine(cfg, model)
    engine.build()
    print(f"loaded + captured in {time.time() - t0:.1f}s "
          f"(blocks={blocks}, capacity={model.batch_capacity})", flush=True)

    base = [1000 + (i % 5000) for i in range(args.prompt_len)]
    prompts = [[(t + 17 * k) % 248000 for t in base] for k in range(max(plan))]
    for n in plan:
        # Warm the graphs for this batch shape, then time a clean decode run.
        run(engine, prompts[:n], 4)
        secs, produced = run(engine, prompts[:n], args.max_new)
        # One token per sequence is already produced by its prefill step.
        steps = max(args.max_new - 1, 1)
        per_step = secs / steps * 1000
        print(f"B={n:<2d} {produced:>5d} tok in {secs:6.2f}s  = "
              f"{produced / secs:7.2f} tok/s decode  "
              f"({per_step:6.2f} ms/step, {produced / secs / n:6.2f} tok/s per seq)",
              flush=True)


if __name__ == "__main__":
    main()
