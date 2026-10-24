"""Long-context profile: prefill cost, peak VRAM and decode speed vs window size.

Why 256k can fit on a 16 GB card at all:

* only 8 of the 32 layers are full attention (``full_attention_interval=4``); the
  other 24 carry the context as a fixed-size recurrent state, so they cost the
  same at 4k and at 256k;
* the KV cache is int8, and there are only 4 kv heads of 256 dims, in 8 full-attn
  layers plus the MTP head's own attention layer:
  ``9 * (4*256*2 + 2*4*4) = 18.3 KB`` per token -> **4.6 GiB at 262144**;
* the weights are int8 (4.0 GiB) instead of fp16 (8.1 GiB) -- run with ``--int8``.

Decode is what the window costs: the step reads that cache once per query head
(16 heads, 4 of them per kv head), i.e. ~17 GB at 256k, plus the 4.0 GB of weights
it always reads. Prefill is quadratic in the prompt (the 8 full-attn layers), so
the first token of a 256k prompt is a ~100 s proposition -- this measures it
rather than guessing.

    MKL_THREADING_LAYER=GNU python probe_longctx.py --int8 \
        --lengths 4096,16384,65536,262144
"""

from __future__ import annotations

import argparse
import random
import time

import torch

import qwen35_engine as E

MODEL = "/root/private_data/models/Qwen3.5-4B"


def sync() -> None:
    torch.cuda.synchronize()


def vram(tag: str) -> str:
    return (f"{tag}: alloc {torch.cuda.memory_allocated() / 2**30:.2f} "
            f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} "
            f"reserved {torch.cuda.memory_reserved() / 2**30:.2f} GiB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--max-seq", type=int, default=262144)
    ap.add_argument("--lengths", default="4096,16384,65536,262144")
    ap.add_argument("--decode", type=int, default=8, help="decode steps timed per length")
    ap.add_argument("--chunk", type=int, default=2048, help="prefill GEMM chunk")
    ap.add_argument("--int8", action="store_true", help="int8 weights (recommended)")
    ap.add_argument("--no-bench", action="store_true")
    ap.add_argument("--mem-dump", type=int, default=0, metavar="LEN",
                    help="prefill LEN tokens with the allocator recorder on, then print "
                         "the largest live allocations with their source line")
    args = ap.parse_args()

    lengths = sorted({int(x) for x in args.lengths.split(",") if x.strip()})
    if lengths[-1] > args.max_seq:
        raise SystemExit(f"length {lengths[-1]} exceeds --max-seq {args.max_seq}")

    random.seed(0)
    toks = [random.randrange(1000, 200000) for _ in range(lengths[-1])]

    t0 = time.time()
    w = E.load_weights(args.model, quant=args.int8)
    print(f"weights {'int8' if args.int8 else 'fp16'}: {time.time() - t0:.1f}s", flush=True)

    torch.cuda.reset_peak_memory_stats()
    m = E.Qwen35(args.model, max_seq=args.max_seq, bench=not args.no_bench, weights=w)
    m.capture()
    sync()
    print(vram("build+capture"), flush=True)

    # KV accounting straight off the live tensors: the 8 full-attn layers plus the
    # MTP head's own full-attention layer, int8 k/v plus the two fp32 row scales.
    full = [lay.mix for lay in m.layers if isinstance(lay.mix, E.FullAttn)]
    if m.mtp is not None and isinstance(getattr(m.mtp, "attn", None), E.FullAttn):
        full.append(m.mtp.attn)
    per_tok = sum(f.nkv * f.hd * 2 + 2 * f.nkv * 4 for f in full)
    print(f"full-attn caches {len(full)}, int8 KV {per_tok / 1024:.2f} KiB/token "
          f"({per_tok * args.max_seq / 2**30:.2f} GiB at {args.max_seq})\n", flush=True)

    if args.mem_dump:
        # Not the state *after* the prefill (that is back to the baseline: the
        # temporaries are freed) but the peak *during* it -- so the alloc/free
        # history is replayed and the live set at its high-water mark is dumped
        # with the source line that allocated each block.
        torch.cuda.memory._record_memory_history(max_entries=1000000)
        m.reset()
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        m.prefill(toks[:args.mem_dump], chunk=args.chunk)
        sync()
        snap = torch.cuda.memory._snapshot()
        torch.cuda.memory._record_memory_history(enabled=None)

        live: dict[int, tuple[int, dict]] = {}
        total = peak = 0
        at_peak: dict[int, tuple[int, dict]] = {}
        for ev in snap.get("device_traces", [[]])[0]:
            act = ev.get("action")
            if act == "alloc":
                live[ev["addr"]] = (ev["size"], ev["frames"][0] if ev.get("frames") else {})
                total += ev["size"]
                if total > peak:
                    peak, at_peak = total, dict(live)
            elif act == "free":
                got = live.pop(ev["addr"], None)
                if got:
                    total -= got[0]
        by_site: dict[tuple, int] = {}
        for size, fr in at_peak.values():
            key = (fr.get("filename", "?"), fr.get("line", 0), fr.get("name", "?"))
            by_site[key] = by_site.get(key, 0) + size
        print(f"\nallocated {base / 2**30:.2f} GiB before, peak "
              f"{peak / 2**30:.2f} GiB during a {args.mem_dump}-token prefill "
              f"(+{(peak - base) / 2**30:.2f})")
        print("largest live groups at the peak:")
        for (fn, ln, name), size in sorted(by_site.items(), key=lambda kv: -kv[1])[:18]:
            print(f"  {size / 2**20:9.1f} MiB  {name}  ({fn}:{ln})")
        free, total_gpu = torch.cuda.mem_get_info()
        print(f"  (reserved {torch.cuda.memory_reserved() / 2**30:.2f}, "
              f"driver free {free / 2**30:.2f} of {total_gpu / 2**30:.2f} GiB)")
        return

    print(f"{'len':>8} {'prefill':>10} {'tok/s':>9} {'peak GiB':>9} "
          f"{'decode ms':>10} {'tok/s':>8} {'cache GiB':>10}")
    for L in lengths:
        try:
            m.reset()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            lg = m.prefill(toks[:L], chunk=args.chunk)
            sync()
            t_pre = time.perf_counter() - t0
            peak = torch.cuda.max_memory_allocated() / 2**30

            nxt = int(lg.argmax())
            for _ in range(2):                       # warm the graph replay
                nxt = int(m.decode_step(nxt).argmax())
            sync()
            t0 = time.perf_counter()
            for _ in range(args.decode):
                nxt = int(m.decode_step(nxt).argmax())
            sync()
            dt = (time.perf_counter() - t0) / max(args.decode, 1)
            print(f"{L:>8} {t_pre:>9.2f}s {L / t_pre:>9.0f} {peak:>9.2f} "
                  f"{dt * 1000:>10.1f} {1 / dt:>8.1f} {per_tok * L / 2**30:>10.2f}",
                  flush=True)
        except Exception as exc:  # noqa: BLE001 - report and try the next length
            print(f"{L:>8} FAILED: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
            m.reset()
            torch.cuda.empty_cache()

    print("\n" + vram("end"), flush=True)


if __name__ == "__main__":
    main()
