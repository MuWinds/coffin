"""Probe: int4 (g128) weight-only GEMV effective bandwidth vs the engine's int8 GEMV.

Synthetic tensors only -- no model download.  Builds ``gemv_int4_ext.cu`` with
torch cpp_extension (same build path as ``hip_ops.py``: gfx906 + the engine's
TORCH_EXTENSIONS_DIR) and measures:

  * y[N] = W4[N,K] @ x[K]  for the real Qwen3.8-27B projection shapes,
  * the same shapes through ``hip_ops.gemv_w8a16`` (int8 baseline) with the
    (R,U) sweep the engine does at load time,
  * HBM streaming-read ceiling and (optionally) mapped-pinned host read + copy
    bandwidth.

Run in the pod:
    MKL_THREADING_LAYER=GNU PYTHONPATH=/root/private_data/engine \
    TORCH_EXTENSIONS_DIR=/root/private_data/torch_ext \
    python3 coffin/diag/probe_gemv_int4.py [--quick]

Output is one machine-parsable block (``KEY value`` lines) plus a human table.

Module: ``coffin/diag``. Builds ``diag/gemv_int4_ext.cu`` through
``torch.utils.cpp_extension`` with the same flags ``hip_ops.py`` uses.

Constraints:
    * Synthetic tensors only: the question is achievable bandwidth, and a real
      checkpoint would add quantisation error to the comparison.
    * The int8 result must come from ``hip_ops.gemv_w8a16`` itself, so the two
      sides differ only in weight format.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx906")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/root/private_data/torch_ext")

import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(HERE)
sys.path.insert(0, ENGINE)

import hip_ops  # noqa: E402  (engine's int8 kernels, cached build)

EXT = load(
    name="gemv_int4_probe_ext",
    sources=[os.path.join(HERE, "gemv_int4_ext.cu")],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--offload-arch=gfx906"],
    verbose=False,
)

D = torch.bfloat16
DEV = "cuda"


# Quantisers that mirror the engine's conventions.
def quant_int8(w: torch.Tensor, chunk: int = 65536) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-chunked so the 1.27B-parameter lm_head shape never materialises fp32."""
    n, k = w.shape
    q = torch.empty(n, k, dtype=torch.int8, device=w.device)
    s = torch.empty(n, dtype=torch.float32, device=w.device)
    for i in range(0, n, chunk):
        wf = w[i:i + chunk].float()
        si = wf.abs().amax(-1).clamp_min(1e-8) / 127.0
        q[i:i + chunk] = (wf / si[:, None]).round().clamp_(-127, 127).to(torch.int8)
        s[i:i + chunk] = si
    return q, s


def quant_int4_g128(w: torch.Tensor, chunk: int = 32768) -> tuple[torch.Tensor, torch.Tensor]:
    """[N,K] bf16 -> packed uint8 [N,K/2] (u=q+8, low nibble = even k) + fp16 [N,K/128]."""
    n, k = w.shape
    assert k % 128 == 0
    packed = torch.empty(n, k // 2, dtype=torch.uint8, device=w.device)
    s = torch.empty(n, k // 128, dtype=torch.float16, device=w.device)
    for i in range(0, n, chunk):
        m = min(chunk, n - i)
        wf = w[i:i + m].float().reshape(m, k // 128, 128)
        si = wf.abs().amax(-1).clamp_min(1e-8) / 7.0
        q = (wf / si[..., None]).round().clamp_(-8, 7).to(torch.int8) + 8
        u = q.reshape(m, k).to(torch.uint8).to(torch.int16)
        packed[i:i + m] = ((u[:, 0::2] | (u[:, 1::2] << 4)) & 0xFF).to(torch.uint8)
        s[i:i + m] = si.to(torch.float16)
    return packed, s


def x_group_sums(x: torch.Tensor) -> torch.Tensor:
    """XS[c] = 8 * sum of the 8 elements of chunk c (fp32), one-off per activation."""
    return (x.float().reshape(-1, 8).sum(-1) * 8.0).contiguous()


def bench(fn, iters: int = 20, warmup: int = 5) -> float:
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
    ap.add_argument("--quick", action="store_true", help="skip the largest shapes")
    ap.add_argument("--host-bw", action="store_true", help="also probe zero-copy/copy")
    args = ap.parse_args()

    props = torch.cuda.get_device_properties(0)
    print(f"DEVICE {props.name} {props.total_memory/2**30:.1f}GiB")
    print(f"TRITON {torch.__version__} hip={torch.version.hip}")

    # HBM streaming read ceiling (1 GiB)
    gbps = EXT.read_bw(1 << 30, 10)
    print(f"HBM_READ_GBPS {gbps:.1f}")

    if args.host_bw:
        r = EXT.host_read_bw(512 << 20, 5)
        print("HOST_PINNED_JSON " + json.dumps({k: round(float(v), 2) for k, v in r.items()}))

    shapes = [
        (17408, 5120),    # one FFN gate/up projection
        (34816, 5120),    # fused gate+up
        (61440, 5120),    # 5x the full-attn qkv width (q 12288 + k 1024 + v 1024 -> x4)
        (248320, 5120),   # tied lm_head / embed_tokens
        (5120, 17408),    # down_proj: K is the FFN width
    ]
    if args.quick:
        shapes = shapes[:3]

    rows = []
    for n, k in shapes:
        print(f"\nSHAPE N={n} K={k}")
        w = torch.randn(n, k, dtype=torch.bfloat16, device=DEV) * 0.02
        x = torch.randn(k, dtype=torch.bfloat16, device=DEV)

        # int8 quantisation, the path the engine ships.
        wq8, s8 = quant_int8(w)
        best8 = None
        for rr, uu in ((1, 1), (1, 2), (1, 4), (2, 1), (2, 2), (4, 1)):
            try:
                ms = bench(lambda: hip_ops.gemv_w8a16(wq8, s8, x, rr, uu))
            except Exception as exc:  # pragma: no cover
                print(f"  INT8_SKIP R={rr} U={uu} {exc}")
                continue
            nbytes = wq8.numel() + s8.numel() * 4
            gb = nbytes / 1e9 / (ms / 1e3)
            if best8 is None or gb > best8[0]:
                best8 = (gb, ms, rr, uu)
            print(f"  INT8 R={rr} U={uu} ms={ms:.3f} GBps={gb:.1f}")
        del wq8, s8

        # int4 with per-128 group scales.
        w4, s4 = quant_int4_g128(w)
        xs = x_group_sums(x)
        best4 = None
        for uu in (1, 2, 4, 8):
            y = EXT.gemv_w4a16(w4, s4, xs, x, uu)
            torch.cuda.synchronize()
            ms = bench(lambda: EXT.gemv_w4a16(w4, s4, xs, x, uu))
            nbytes = w4.numel() + s4.numel() * 2
            gb = nbytes / 1e9 / (ms / 1e3)
            if best4 is None or gb > best4[0]:
                best4 = (gb, ms, uu)
            print(f"  INT4 U={uu} ms={ms:.3f} GBps={gb:.1f}")
        y4 = EXT.gemv_w4a16(w4, s4, xs, x, best4[2]).float()

        # dequantised reference dot (float64 on CPU, first 8 rows, chunked to save RAM)
        kk = min(8, n)
        wd = w4[:kk].to(torch.int64)
        lo = (wd & 0xF).to(torch.int8) - 8
        hi = ((wd >> 4) & 0xF).to(torch.int8) - 8
        wq = torch.empty(kk, k, dtype=torch.int8, device=DEV)
        wq[:, 0::2] = lo
        wq[:, 1::2] = hi
        s4f = s4[:kk].float().repeat_interleave(128, dim=1)
        ref8 = (wq.float() * s4f) @ x.float()
        err = (y4[:kk] - ref8).abs().max().item()
        print(f"  INT4_CHECK max_abs_err={err:.5f} (rows={kk})")

        speedup = best4[0] / best8[0]
        rows.append({"N": n, "K": k, "int4_gbps": best4[0], "int4_ms": best4[1],
                     "int4_U": best4[2], "int8_gbps": best8[0], "int8_ms": best8[1],
                     "int8_RU": [best8[2], best8[3]], "ratio": speedup,
                     "check_err": err})
        del w, w4, s4, xs, wq
        torch.cuda.empty_cache()

    print("\nSUMMARY")
    for r in rows:
        print("SHAPE_JSON " + json.dumps({k: (round(v, 2) if isinstance(v, float) else v)
                                          for k, v in r.items()}))

    print("\nDONE")


if __name__ == "__main__":
    main()
