"""Per-step cost of the static-batch decode path vs the single-stream one.

Measures, on the real model and real weights:

* decode step time for buckets 1/2/4/8 (graph replay + the host-side row staging,
  sampling excluded -- that is ``model_runner``'s job);
* aggregate throughput relative to B=1, i.e. how much of the batch is free;
* the projection pass on its own: the weight read every decode step must pay, at
  M=1 vs M=8.  This is the stage-1 win -- one weight read serves all rows -- so
  it is also the number that explains the step-time scaling;
* captured-graph memory, since the 16 GB card has to hold the pool, the weights
  and one graph pool shared by all buckets.

Run (inside the container, from the engine directory)::

    python3 diag/probe_batch_decode.py --steps 30

Module: ``coffin/diag``. Reads the batched path in ``model_runner`` and
``models/qwen35/batch``; imports the model, so it needs a DCU and the
checkpoint.

Constraints:
    * Pure measurement: no assertion, no failure mode. A number that looks wrong
      means re-measure, not "the test broke".
    * Sampling is excluded from the step time on purpose, so the buckets are
      comparable; including it would mix in the host-side argmax/multinomial.
    * Run on an otherwise idle card: a 16 GB device holding the pool, the
      weights and the graph pool will start swapping if anything else is resident.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from engine_api import create_model, load_weights
from models.qwen35.batch import batch_proj
from models.qwen35.linear_attn import LinearAttn

MODEL = os.environ.get("SCNET_MODEL", "/root/private_data/models/Qwen3.5-4B")
BLOCK = 64
CAP = 8                      # decode rows
BATCH_CAP = CAP + 1          # + the single-stream prefill/verify staging row
BUCKETS = (1, 2, 4, 8)
BASE = [9707, 11, 1879, 3671, 374, 264, 3449, 315, 9847, 13,
        3838, 374, 279, 6722, 315, 9625, 323, 11751, 13]
D = torch.bfloat16


def tokens(n: int) -> list[int]:
    return [BASE[i % len(BASE)] for i in range(n)]


def stage(model, pool, ids: list[int], row: int, total: int | None = None):
    """Prefill ``ids`` on the staging row, then move its state into decode ``row``."""
    n_blocks = -(-(total or len(ids)) // BLOCK)
    blocks = [pool.alloc() for _ in range(n_blocks)]
    model.pos1.zero_()
    model.zero_linear_states(model.prefill_row)
    pool.assign_table(blocks, 0, row=model.prefill_row)
    logits = model.prefill(ids)
    model.move_linear_states(model.prefill_row, row)
    return blocks, int(logits.argmax())


def collect_projs(model):
    """Every GEMV a decode step reads, in layer order."""
    projs = []
    for layer in model.layers:
        mix = layer.mix
        if isinstance(mix, LinearAttn):
            projs.append(("lin.in_proj", mix.qkvzb_a.g))
            projs.append(("lin.out_proj", mix.out_proj))
        else:
            projs.append(("attn.qkv", mix.qkv.g))
            projs.append(("attn.o_proj", mix.o_proj))
        projs.append(("mlp.gate_up", layer.mlp.gate_up.g))
        projs.append(("mlp.down", layer.mlp.down))
    projs.append(("lm_head", model.embed.gemv))
    return projs


def time_projs(projs, B: int, iters: int = 10, autotuned: bool = False) -> tuple[float, float]:
    """(ms per pass, bytes read per pass) for the projection stage at batch ``B``.

    ``autotuned=False`` is the production path: ``batch_proj`` replays the
    single-stream kernel's unroll so every row stays bit-identical to a solo
    decode.  ``autotuned=True`` uses ``GEMV.prefill``'s per-M autotuned unroll
    (faster on some shapes, but it changes fp32 accumulation order and therefore
    a few bf16 logits).
    """
    xs = [torch.randn(B, g.K, dtype=D, device="cuda") for _, g in projs]
    if B == 1:
        fns = [lambda g=g, x=x: g(x[0]) for (_, g), x in zip(projs, xs)]
    elif autotuned:
        fns = [lambda g=g, x=x: g.prefill(x) for (_, g), x in zip(projs, xs)]
    else:
        fns = [lambda g=g, x=x: batch_proj(g, x) for (_, g), x in zip(projs, xs)]
    for fn in fns:                                  # warm every kernel shape
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        for fn in fns:
            fn()
    e.record()
    torch.cuda.synchronize()
    total_bytes = 0
    for _, g in projs:
        elem = 1 if g.q is not None else (2 if g.w is not None and g.w.dtype in
                                          (torch.float16, torch.bfloat16) else 4)
        total_bytes += g.N * g.K * elem
    return s.elapsed_time(e) / iters, total_bytes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--prefix", type=int, default=64)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--blocks", type=int, default=160)
    ap.add_argument("--quant", action="store_true", default=True)
    args = ap.parse_args()

    print(f"device={torch.cuda.get_device_name(0)}")
    weights = load_weights(MODEL, quant=args.quant)
    model = create_model(MODEL, max_seq=args.max_seq, bench=True, weights=weights,
                         paged=True)
    pool = model.enable_paging(num_blocks=args.blocks, block_size=BLOCK)
    model.enable_batching(BATCH_CAP)

    mem0 = torch.cuda.memory_allocated()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.capture_batch(BUCKETS)
    torch.cuda.synchronize()
    t_cap = time.perf_counter() - t0
    print(f"capture_batch{BUCKETS}: {t_cap:.2f} s   "
          f"allocated {torch.cuda.memory_allocated()/2**20:.0f} MiB "
          f"(+{(torch.cuda.memory_allocated()-mem0)/2**20:.0f} MiB for the graphs), "
          f"reserved {torch.cuda.memory_reserved()/2**20:.0f} MiB, "
          f"free {torch.cuda.mem_get_info()[0]/2**20:.0f} MiB")

    ids = tokens(args.prefix)
    total = args.prefix + args.steps + args.warmup + 2 * BLOCK
    blocks = []
    first = []
    for row in range(CAP):
        b, tok = stage(model, pool, ids, row, total)
        blocks.append(b)
        first.append(tok)
    pos = [len(ids)] * CAP
    feed = tokens(CAP * 64)

    rows_out = {}
    for B in BUCKETS:
        model.set_rows(first[:B], pos[:B], blocks[:B])
        for s in range(args.warmup):
            model.decode_batch(B, B)
            model.set_rows([feed[B * 8 + s] % 200000] * B,
                           [pos[0] + 1 + s] * B, blocks[:B])
        torch.cuda.synchronize()
        s_ev = torch.cuda.Event(enable_timing=True)
        e_ev = torch.cuda.Event(enable_timing=True)
        s_ev.record()
        for k in range(args.steps):
            model.set_rows([feed[(B * 16 + k) % len(feed)]] * B,
                           [pos[0] + 1 + args.warmup + k] * B, blocks[:B])
            model.decode_batch(B, B)
        e_ev.record()
        torch.cuda.synchronize()
        ms = s_ev.elapsed_time(e_ev) / args.steps
        rows_out[B] = ms
        print(f"B={B}: {ms:7.3f} ms/step  -> {B / ms * 1000:8.1f} tok/s aggregate"
              f"   ({ms / rows_out[1]:.3f}x the B=1 step time)")

    projs = collect_projs(model)
    n_split = sum(1 for _, g in projs if getattr(g, "u", None) == 4)
    us = {}
    for _, g in projs:
        us[g.u] = us.get(g.u, 0) + 1
    ms1, nbytes = time_projs(projs, 1)
    ms8, _ = time_projs(projs, 8)
    ms8_auto, _ = time_projs(projs, 8, autotuned=True)
    gb = nbytes / 1e9
    print(f"\nprojection pass, M=1: {ms1:7.3f} ms  ({gb / (ms1 / 1000):6.1f} GB/s effective)")
    print(f"projection pass, M=8 (exact, matched unroll): {ms8:7.3f} ms  "
          f"({gb / (ms8 / 1000):6.1f} GB/s effective)")
    print(f"B=8 projection cost vs B=1: {ms8 / ms1:.3f}x  "
          f"(=> {8 / (ms8 / ms1):.1f}x the tokens per weight read)")
    print(f"projection share of the B=8 step: {ms8 / rows_out[8] * 100:.1f}%")
    print(f"single-stream unrolls: {us};  M=8 needs two M=4 calls for "
          f"{n_split}/{len(projs)} projections (u=4 has no M=8 instantiation)")
    print(f"A/B (non-bit-exact) autotuned M=8 path: {ms8_auto:7.3f} ms "
          f"({ms8_auto / ms1:.3f}x M=1) -- not used: it changes accumulation order")
    print(f"weights read per projection pass: {gb:.2f} GB")


if __name__ == "__main__":
    main()
