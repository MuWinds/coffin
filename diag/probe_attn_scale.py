"""Probe: fused int8-KV decode attention scaling with context length.

Synthetic KV only.  Reuses the engine's cached HIP build (``hip_ops.attn``) and
its Qwen3.8-27B full-layer geometry: nh=24 q heads, nkv=4 kv heads, hd=256,
rep=6 (Qwen3.5-4B's nh=16/rep=4 is measured too, for continuity with NOTES.md).

For every (window, split) it reports ms/call and the effective KV-read GB/s,
then extrapolates to the 16 full-attention layers of Qwen3.8-27B.

    MKL_THREADING_LAYER=GNU PYTHONPATH=/root/private_data/engine \
    TORCH_EXTENSIONS_DIR=/root/private_data/torch_ext \
    python3 coffin/diag/probe_attn_scale.py [--max 262144] [--quick]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx906")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/root/private_data/torch_ext")

import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import hip_ops  # noqa: E402


def bench(fn, iters: int = 5, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=262144)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    D = torch.bfloat16
    dev = "cuda"
    print(f"DEVICE {torch.cuda.get_device_properties(0).name} "
          f"{torch.cuda.get_device_properties(0).total_memory/2**30:.1f}GiB "
          f"hip={torch.version.hip}")

    S_max = args.max
    nkv, hd = 4, 256
    kc = torch.zeros(nkv, S_max, hd, dtype=torch.int8, device=dev)
    vc = torch.zeros(nkv, S_max, hd, dtype=torch.int8, device=dev)
    ks = torch.full((nkv, S_max), 0.01, dtype=torch.float32, device=dev)
    vs = torch.full((nkv, S_max), 0.01, dtype=torch.float32, device=dev)
    pos = torch.zeros(1, dtype=torch.int64, device=dev)

    windows = [8192, 16384, 32768, 65536, 131072]
    if S_max >= 262144:
        windows.append(262144)
    if args.quick:
        windows = [8192, 65536, 262144]
    splits = [8, 16, 32, 64]
    geoms = [(24, 6), (16, 4)]      # (nh, rep) -> 27B full layer, 4B full layer

    rows = []
    for nh, rep in geoms:
        q = torch.randn(nh, hd, dtype=D, device=dev)
        gate = torch.randn(nh, hd, dtype=D, device=dev)
        out = torch.zeros(nh, hd, dtype=D, device=dev)
        print(f"\nGEOM nh={nh} rep={rep} nkv={nkv} hd={hd}")
        for S in windows:
            pos.fill_(S - 1)
            kv_bytes = 2 * nkv * hd * S            # K + V int8
            sc_bytes = 2 * nkv * S * 4             # fp32 scales
            for split in splits:
                part_acc = torch.zeros(nh, split, hd, dtype=torch.float32, device=dev)
                part_ml = torch.zeros(nh, split, 2, dtype=torch.float32, device=dev)
                try:
                    fn = (lambda q=q, gate=gate, out=out, part_acc=part_acc,
                          part_ml=part_ml, rep=rep: hip_ops.attn(
                              q, gate, kc, vc, ks, vs, out, pos, rep, hd ** -0.5,
                              part_acc, part_ml))
                    ms = bench(fn, iters=3 if S >= 131072 else 6)
                    gb = (kv_bytes + sc_bytes) / 1e9 / (ms / 1e3)
                    row = {"nh": nh, "rep": rep, "S": S, "split": split,
                           "ms": round(ms, 3), "gbps": round(gb, 1)}
                    rows.append(row)
                    print("ATTN_JSON " + json.dumps(row))
                except Exception as exc:  # pragma: no cover
                    print(f"ATTN_FAIL nh={nh} S={S} split={split}: {exc}")
                del part_acc, part_ml

        # launch overhead at pos=0 (one key)
        pos.fill_(0)
        split = 8
        part_acc = torch.zeros(nh, split, hd, dtype=torch.float32, device=dev)
        part_ml = torch.zeros(nh, split, 2, dtype=torch.float32, device=dev)
        ms0 = bench(lambda: hip_ops.attn(q, gate, kc, vc, ks, vs, out, pos, rep,
                                         hd ** -0.5, part_acc, part_ml), iters=20)
        print(f"ATTN_OVERHEAD nh={nh} rep={rep} ms={ms0:.4f}")

    # extrapolation to 16 layers at 256k: best split
    best = {}
    for r in rows:
        key = (r["nh"], r["S"])
        if key not in best or r["gbps"] > best[key]["gbps"]:
            best[key] = r
    print("\nLAYER_EXTRAP")
    for (nh, S), r in sorted(best.items()):
        if nh != 24:
            continue
        layer_ms = r["ms"]
        print(f"  S={S} best_split={r['split']} layer_ms={layer_ms:.2f} "
              f"gbps={r['gbps']:.1f} -> 16layers={layer_ms*16:.1f}ms "
              f"decode_tok_s={1000.0/(layer_ms*16):.2f}")
    print("DONE")


if __name__ == "__main__":
    main()
