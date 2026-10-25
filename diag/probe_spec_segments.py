"""Where does the one-forward verify's time go?

Answers which segment owns the verify step, and whether the wide projection is
actually being dispatched. Decomposes the task-13 verify into device-time
segments (projection / attention / linear recurrence / MLP / lm_head) with CUDA
events, plus the graph replay and host bookkeeping, and counts where
``wide_proj`` sent each module. A healthy int8 build at B=8, K=1 is: every
projection and the lm_head on the dot2 wide kernel (M=16), one
``attn_paged_multi_q`` per full-attention layer, and one ``verify_batch_step``
per linear layer per token.

Read the segment table first: the five rows must sum to the ``total``, and the
printout's own closing hint states the expected range (``proj`` around 25-32 ms).
A ``proj`` well off that range is the signal to look at the ``wide_proj
dispatch`` line above it — if the counts are not all dot2, the wide kernel never
ran and the segment number is measuring the fallback. ``host/accept overhead``
near zero means the step is device-bound; a large value means graph-capture
coverage or acceptance bookkeeping is the problem, not any kernel.

Run from the engine directory::

    python3 diag/probe_spec_segments.py --batch 8 --prompt 96 --k 1

Module: ``coffin/diag``. Standalone probe: imports ``engine_api`` plus
``models.qwen35.batch``, because segment attribution has to instrument the batch
module's dispatch counters. Nothing under ``coffin/`` imports it.

Constraints:
    * Reaches into ``models/qwen35/batch.py`` internals (``WIDE_STATS``,
      ``_project``, ``wide_proj``) on purpose. That coupling is the measurement:
      a refactor that renames them breaks this probe, and fixing it is cheaper
      than losing the attribution.
    * Silent about what it cannot see: an unprofiled segment reports ``0.00 ms``,
      so a suspiciously small row means the event tags changed, not that the
      segment got free.
    * Needs the real checkpoint, a paged model and a captured verify graph, so
      the replay rows measure the path production runs.
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
from models.qwen35 import batch as batch_mod

MODEL = os.environ.get("SCNET_MODEL", "/root/private_data/models/Qwen3.5-4B")
BLOCK = 64
BUCKETS = (1, 2, 4, 8)
BASE = [9707, 11, 1879, 3671, 374, 264, 3449, 315, 9847, 13,
        3838, 374, 279, 6722, 315, 9625, 323, 11751, 13]


def tokens(n: int, off: int = 0) -> list[int]:
    return [BASE[(i + off) % len(BASE)] for i in range(n)]


def alloc(pool, n_tokens: int) -> list[int]:
    return [pool.alloc() for _ in range(-(-(n_tokens + 64) // BLOCK))]


def timed(fn, n: int) -> float:
    """Milliseconds per call, wall clock, synchronized."""
    fn()                                        # untimed warmup
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--prompt", type=int, default=96)
    ap.add_argument("--k", type=int, default=1)
    args = ap.parse_args()

    cap = args.batch
    max_seq = args.prompt + args.k + 3 * BLOCK
    blocks = cap * (-(-(max_seq + 2 * BLOCK) // BLOCK)) + cap
    print(f"device={torch.cuda.get_device_name(0)}")
    weights = load_weights(MODEL, quant=True)
    model = create_model(MODEL, max_seq=max_seq, bench=True, weights=weights,
                         paged=True)
    pool = model.enable_paging(num_blocks=blocks, block_size=BLOCK)
    model.enable_batching(cap + 1)
    model.capture_batch(BUCKETS)
    model.enable_spec_batch(args.k, buckets=BUCKETS, verify_mode="once")
    bucket = next(b for b in BUCKETS if b >= cap)
    c = args.k + 1

    prompts = [tokens(args.prompt, off=7 * i) for i in range(cap)]
    blocks_of = [alloc(pool, len(p)) for p in prompts]
    seqs = [SimpleNamespace(token_ids=p, row=r, block_table=b)
            for r, (p, b) in enumerate(zip(prompts, blocks_of))]
    model.reset()
    logits = torch.stack([lg.float() for lg in model.prefill_group(seqs)[0]])
    cur = [int(t) for t in logits.argmax(-1).tolist()]
    pos = [len(p) for p in prompts]

    # one real step to stage the verify buffers, then profile the eager forward
    model.spec_step_batch(cur, pos, blocks_of, bucket, args.k)
    batch_mod.WIDE_STATS.clear()

    prof: dict[str, list] = {}
    model._verify_batch_once(cap, prof)
    torch.cuda.synchronize()
    seg: dict[str, float] = {}
    for tag, laps in prof.items():
        seg[tag] = sum(s.elapsed_time(e) for s, e in laps)
    once_device = sum(seg.values())

    print(f"\nverify segments (B={cap}, c={c}, M={cap*c}, eager, one call):")
    for tag in ("proj", "attn", "recur", "mlp", "lmhead"):
        print(f"  {tag:7s} {seg.get(tag, 0.0):7.2f} ms")
    print(f"  {'total':7s} {once_device:7.2f} ms")
    print(f"\nwide_proj dispatch: {dict(sorted(batch_mod.WIDE_STATS.items()))}")

    graph = model._verify_batch_graphs[bucket]
    ms_verify = timed(lambda: graph.replay(), 5)
    ms_draft = timed(lambda: model.mtp.draft_batch(bucket, args.k,
                                                   model.spec_toks_b[:bucket]), 5)
    step = lambda: model.spec_step_batch(cur, pos, blocks_of, bucket, args.k)
    ms_step = timed(step, 5)
    ms_greedy = timed(lambda: model.decode_batch(cap, bucket), 5)
    print(f"\nper step:")
    print(f"  greedy decode_batch        {ms_greedy:7.2f} ms  (device+host)")
    print(f"  verify graph replay        {ms_verify:7.2f} ms  (device+host)")
    print(f"  drafter graph replay       {ms_draft:7.2f} ms")
    print(f"  spec_step_batch total      {ms_step:7.2f} ms")
    print(f"  host/accept overhead       {ms_step - ms_verify - ms_draft:7.2f} ms")
    print(f"How to read: once-verify should be ~= segments total; if 'proj' is not"
          f" ~25-32 ms, check the dispatch counts above (want all dot2).")

    # per-shape wide projection cost, to see whether dot2's element rate is
    # shape-dependent (MLP / lm_head dominate the model's weight bytes).
    def bench(proj, label):
        g = proj.g if hasattr(proj, "g") else proj
        n, k = g.q.wq.shape
        x = torch.randn(cap * c, k, dtype=torch.bfloat16, device="cuda")
        ms = timed(lambda: batch_mod.wide_proj(proj, x), 10)
        print(f"  {label:16s} M={cap*c:2d} N={n:5d} K={k:5d}  {ms:7.3f} ms  "
              f"{n*k/1e6/ms:6.1f} Gelem/s")

    print(f"\nwide projection by shape (dot2, M={cap*c}):")
    seen = set()
    for layer in model.layers:
        mix = layer.mix
        name = type(mix).__name__
        if name not in seen:
            seen.add(name)
            bench(mix.qkv if name == "FullAttn" else mix.qkvzb_a, f"{name}.qkv")
            bench(mix.o_proj if name == "FullAttn" else mix.out_proj, f"{name}.o_proj")
            bench(layer.mlp.gate_up, "mlp.gate_up")
            bench(layer.mlp.down, "mlp.down")
    bench(model.embed.gemv, "lm_head")

    # Which kernel/setting is actually fastest per weight byte?  The once path
    # needs the M=B*c kernel to beat the greedy M=8 kernel per element read.
    import hip_ops
    from models.qwen35.batch import _project

    def variants(proj, label):
        g = proj.g if hasattr(proj, "g") else proj
        n, k = g.q.wq.shape
        elems = n * k / 1e6
        x16 = torch.randn(cap * c, k, dtype=torch.bfloat16, device="cuda")
        x8 = torch.randn(8, k, dtype=torch.bfloat16, device="cuda")
        out = [f"  {label:14s}"]
        ms = timed(lambda: _project(g, x8), 10)
        out.append(f"M8 gemv_m {ms:6.3f}ms/{elems/ms:5.1f}Ge")
        ms = timed(lambda: torch.cat([_project(g, x16[i:i + 8]) for i in (0, 8)]), 10)
        out.append(f"split8x2 {ms:6.3f}ms/{elems/ms:5.1f}Ge")
        for name, fn in (("dot2", hip_ops.gemv_w8a16_wide_dot2),
                         ("int8wide", hip_ops.gemv_w8a16_wide)):
            for u in (2, 4):
                try:
                    ms = timed(lambda: fn(g.q.wq, g.q.scale, x16, u), 10)
                    out.append(f"{name}U{u} {ms:6.3f}ms/{elems/ms:5.1f}Ge")
                except Exception as exc:      # bad M/U combo in the kernel
                    out.append(f"{name}U{u} n/a({type(exc).__name__})")
        print("  ".join(out))

    print(f"\nkernel comparison (same weight bytes, M=8 greedy vs M=16 wide):")
    variants(model.layers[0].mlp.gate_up, "mlp.gate_up")
    variants(model.embed.gemv, "lm_head")


if __name__ == "__main__":
    main()
