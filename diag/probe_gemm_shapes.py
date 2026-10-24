"""Which shape/dtype/precision actually makes the prefill GEMMs fast?

The prefill profile says the projections are ~80% of device time, but at very
different efficiency per shape: ``mlp gate_up`` (N=18432) runs at ~2.1 TFLOP/s
while ``out_proj`` (N=2560, K=4096) reaches ~7.0.  So sweep the levers that do
not need a new kernel, for every real projection shape:

  * dtype -- this part has no bf16 packed-math instruction, but it does have
    fp16 dot products, so fp16 weights may simply be faster;
  * M -- rocBLAS picks its kernel from the shape, and the model's chunk is 256;
  * N-splitting -- a wide N gets a narrow-tile kernel; several narrow calls may
    be faster than one wide one;
  * transposed matmul (W @ x.T) -- turns [M=C, N=wide] into [M=wide, N=C],
    which rocBLAS may like better.

Run:  python probe_gemm_shapes.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

SHAPES = [
    ("in_qkv (linear)", 8192, 2560),
    ("in_z (linear)", 2560, 2560),
    ("out_proj (linear)", 2560, 4096),
    ("mlp gate_up", 18432, 2560),
    ("mlp down", 2560, 9216),
    ("fa qkv", 6144, 2560),
    ("lm_head", 151936, 2560),
]
MS = (128, 256, 512, 1024, 2048)


def bench(fn, iters=5, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main() -> None:
    torch.manual_seed(0)
    print(f"device={torch.cuda.get_device_name(0)}  torch={torch.__version__}")
    for name, N, K in SHAPES:
        print(f"\n=== {name}: [N={N}, K={K}]  (TFLOP/s) ===")
        hdr = f"{'M':>6}{'bf16':>9}{'fp16':>9}{'fp32':>9}"
        for split in (2048, 4096, 8192):
            hdr += f"{'bf16/' + str(split):>11}"
        hdr += f"{'bf16 W@xT':>11}{'fp16 W@xT':>11}"
        print(hdr)
        for M in MS:
            row = f"{M:>6}"
            flops = 2 * M * N * K
            for dt in (torch.bfloat16, torch.float16, torch.float32):
                W = (torch.randn(N, K, dtype=torch.float32, device="cuda") * 0.02).to(dt)
                x = torch.randn(M, K, dtype=dt, device="cuda")
                ms = bench(lambda: F.linear(x, W))
                row += f"{flops / ms / 1e9:>9.1f}"
                del W, x
                torch.cuda.empty_cache()
            Wb = (torch.randn(N, K, dtype=torch.float32, device="cuda") * 0.02).to(torch.bfloat16)
            xb = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
            for split in (2048, 4096, 8192):
                if split >= N:
                    row += f"{'--':>11}"
                    continue
                ws = [w.contiguous() for w in Wb.split(split, 0)]
                ms = bench(lambda: [F.linear(xb, w) for w in ws])
                row += f"{flops / ms / 1e9:>11.1f}"
            Wtb = Wb.t().contiguous()                       # [K, N]
            ms = bench(lambda: (xb @ Wtb).t().contiguous())
            row += f"{flops / ms / 1e9:>11.1f}"
            Wf = Wb.to(torch.float16)
            xt = xb.to(torch.float16)
            Wtf = Wf.t().contiguous()
            ms = bench(lambda: (xt @ Wtf).t().contiguous())
            row += f"{flops / ms / 1e9:>11.1f}"
            print(row)
            del Wb, xb, Wtb, Wf, xt, Wtf
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
