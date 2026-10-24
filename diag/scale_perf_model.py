#!/usr/bin/env python3
"""Cost model for Qwen3.8-27B on one 16 GB DCU (Z200SM_80).

Every number the 27B plan quotes comes from here, so the constants can be
re-measured and the tables regenerated instead of re-derived by hand.  The model
is deliberately simple: at 256k the per-token wall time is one of three maxima --
weight streaming, cold-KV streaming, or attention compute -- and the point of the
exercise is to see which one wins at which context length.

Usage:
    python3 diag/scale_perf_model.py                    # defaults = measured/assumed
    python3 diag/scale_perf_model.py --pcie 21.5 --w-bw 560
"""

from __future__ import annotations

import argparse

# Qwen3.8-27B text-tower geometry from config.json.
HID = 5120
VOCAB = 248320
FF = 17408
N_LAYERS = 64
N_LINEAR = 48
N_FULL = 16
NH = 24
NKV = 4
HD = 256
KV_HEADS_BYTES_INT8 = 2 * NKV * HD          # K + V per token per layer, 1 byte each
KV_SCALE_BYTES = 2 * NKV * 2                # fp16 scale per (head, token), K + V
KV_INT8_PER_TOKEN = N_FULL * (KV_HEADS_BYTES_INT8 + KV_SCALE_BYTES)
KV_INT4_PER_TOKEN = N_FULL * (KV_HEADS_BYTES_INT8 // 2 + KV_SCALE_BYTES)
KV_INT2_PER_TOKEN = N_FULL * (KV_HEADS_BYTES_INT8 // 4 + KV_SCALE_BYTES)

# param counts (text, no MTP, no vision)
P_EMBED = VOCAB * HID
P_MLP = 3 * HID * FF
P_LIN = (2 * 16 * 128 + 48 * 128) * HID + (48 * 128) * HID + 2 * 48 * HID + (48 * 128) * HID
P_FULL = (2 * NH * HD + 2 * NKV * HD) * HID + NH * HD * HID
P_TEXT = 2 * P_EMBED + N_LINEAR * (P_LIN + P_MLP) + N_FULL * (P_FULL + P_MLP)

W_BITS = {"w4g128": 4.125, "w4g128-fp8sc": 4.0625, "w4g64": 4.25, "w3g128": 3.125, "w2g128": 2.125}


def weights_gib(bits: float) -> float:
    return P_TEXT * bits / 8 / 2**30


def kv_gib(n_tokens: int, per_token: int) -> float:
    return n_tokens * per_token / 2**30


def decode_ms(ctx: int, hot: int, cold_frac: float, w_bw: float, pcie: float,
              kv_bits: str = "int4", attn_tflops: float = 8.0,
              fixed_ms: float = 3.0) -> float:
    """One decode step, weights and cold-KV streaming overlapped, plus attention."""
    per_tok = {"int4": KV_INT4_PER_TOKEN, "int8": KV_INT8_PER_TOKEN,
               "int2": KV_INT2_PER_TOKEN}[kv_bits]
    w_ms = P_TEXT * W_BITS["w4g128"] / 8 / (w_bw * 1e9) * 1e3
    hot_tokens = min(hot, ctx)
    cold_tokens = max(0, ctx - hot_tokens) * cold_frac
    cold_gib = cold_tokens * per_tok / 2**30
    cold_ms = cold_gib / pcie * 1e3
    # attention compute: QK+PV over ctx keys, both in fp16-dot units
    attn_flops = 2 * 2 * NH * HD * ctx * N_FULL
    attn_ms = attn_flops / (attn_tflops * 1e12) * 1e3
    return max(w_ms, cold_ms) + attn_ms + fixed_ms


def prefill_s(ctx: int, chunk: int, gemm_tflops: float = 13.5,
              attn_tflops: float = 12.0, pcie: float = 20.0) -> float:
    lin = 2 * P_TEXT * ctx / (gemm_tflops * 1e12)
    attn = 12288 * ctx * ctx * N_FULL / (attn_tflops * 1e12)
    cold_bytes = (ctx * ctx / (2 * chunk)) * N_FULL * (KV_HEADS_BYTES_INT8 // 2)
    cold = cold_bytes / (pcie * 1e9)
    return lin + attn + cold


def main() -> None:
    ap = argparse.ArgumentParser()
    # measured on the pod, 2026-10-22 (diag/probe_hw.py): pinned H2D 13.1 GB/s,
    # D2H 14.1 GB/s, HBM read 630 GB/s, free VRAM after driver 15.5 GiB.
    ap.add_argument("--pcie", type=float, default=13.1, help="pinned host->device GB/s")
    ap.add_argument("--w-bw", type=float, default=430.0, help="int4 GEMV GB/s (measured, correct kernel)")
    ap.add_argument("--hbm", type=float, default=630.0, help="HBM read GB/s")
    ap.add_argument("--hot-gib", type=float, default=1.8, help="VRAM KV window budget")
    ap.add_argument("--overhead-gib", type=float, default=0.6, help="non-weight VRAM overhead")
    a = ap.parse_args()

    print(f"text params        {P_TEXT/1e9:.2f} B")
    print(f"KV/token  int8 {KV_INT8_PER_TOKEN} B | int4 {KV_INT4_PER_TOKEN} B | int2 {KV_INT2_PER_TOKEN} B")
    for name, bits in W_BITS.items():
        print(f"weights {name:16s} {weights_gib(bits):6.2f} GiB")
    print(f"KV@256k int8 {kv_gib(262144, KV_INT8_PER_TOKEN):.2f} GiB | int4 "
          f"{kv_gib(262144, KV_INT4_PER_TOKEN):.2f} GiB | int2 {kv_gib(262144, KV_INT2_PER_TOKEN):.2f} GiB")
    hot_tokens = int(a.hot_gib * 2**30 / KV_INT4_PER_TOKEN)
    print(f"hot window {a.hot_gib:.2f} GiB = {hot_tokens/1024:.0f}k int4 tokens; "
          f"VRAM headroom after W4 + overhead = {16 - weights_gib(4.125) - a.overhead_gib:.2f} GiB")
    print()

    print("decode tok/s (weights int4 g128; KV int4; cold via PCIe)")
    print(f"{'ctx':>8} {'hot':>6} {'cold%':>6} {'w_ms':>6} {'cold_ms':>8} {'attn_ms':>7} {'ms':>6} {'tok/s':>6}")
    for ctx in (4096, 32768, 65536, 131072, 262144):
        for frac, tag in ((1.0, "full"), (0.10, "quest10%")):
            if frac == 1.0 and ctx >= 262144:
                pass
            ms = decode_ms(ctx, hot_tokens, frac, a.w_bw, a.pcie)
            per = KV_INT4_PER_TOKEN
            hot_t = min(hot_tokens, ctx)
            cold_gib = max(0, ctx - hot_t) * frac * per / 2**30
            w_ms = P_TEXT * 4.125 / 8 / (a.w_bw * 1e9) * 1e3
            attn_ms = 4 * NH * HD * ctx * N_FULL / (8e12) * 1e3
            print(f"{ctx:>8} {hot_t:>6} {frac*100:>5.0f}% {w_ms:>6.1f} {cold_gib/a.pcie*1e3:>8.1f} "
                  f"{attn_ms:>7.1f} {ms:>6.1f} {1000/ms:>6.1f}   {tag}")
    print()

    print("prefill (cold, chunk=2048)")
    for ctx in (8192, 32768, 65536, 131072, 262144):
        s = prefill_s(ctx, 2048, pcie=a.pcie)
        print(f"  {ctx:>7}: {s:>7.1f} s  ({ctx/s:>6.0f} tok/s)")
    print()
    print("incremental append (new_tokens=1024) at ctx")
    for ctx in (32768, 131072, 262144):
        s = prefill_s(1024, 1024, pcie=a.pcie)  # linear+attn part only (no full ctx quadratic)
        attn = 12288 * ctx * 1024 * N_FULL / (12e12)
        lin = 2 * P_TEXT * 1024 / (13.5e12)
        print(f"  {ctx:>7}: {(lin+attn):>6.1f} s")


if __name__ == "__main__":
    main()
