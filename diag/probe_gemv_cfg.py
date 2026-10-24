"""Sweep the GEMV kernel's (rows-per-warp, K-unroll) configs over the engine's
shapes, through the production module. This is the data behind GEMV._HIP_CFGS.

Run:  python probe_gemv_cfg.py
"""

from __future__ import annotations

import torch

import hip_ops

D = torch.bfloat16
CONFIGS = ((1, 1), (1, 2), (1, 4), (2, 1), (2, 2), (4, 1))
SHAPES = [
    ("mlp_gate/up", 9216, 2560),
    ("mlp_down", 2560, 9216),
    ("in_qkv", 8192, 2560),
    ("in_z", 4096, 2560),
    ("out_proj", 2560, 4096),
    ("k/v_proj", 1024, 2560),
    ("lm_head", 248320, 2560),
]


def bench(fn, iters=100, warmup=15, reps=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / iters)
    return best


def main():
    print(f"device={torch.cuda.get_device_name(0)}")
    header = "".join(f"{'R'+str(r)+'U'+str(u):>10}" for r, u in CONFIGS)
    print(f"\n{'shape':<12}{'N':>7}{'K':>6}{header}{'blas':>8}   best")
    for name, N, K in SHAPES:
        x = torch.randn(K, dtype=D, device="cuda")
        W = (torch.randn(N, K, dtype=D, device="cuda") * 0.02).contiguous()
        Wt = W.t().contiguous()
        nbytes = N * K * 2
        ref = x.float() @ Wt.float()

        row = f"{name:<12}{N:>7}{K:>6}"
        best, best_bw = "", 0.0
        for r, u in CONFIGS:
            got = hip_ops.gemv(W, x, r, u).float()
            ok = ((got - ref).abs().max() / ref.abs().max()).item() < 3e-2
            bw = nbytes / bench(lambda: hip_ops.gemv(W, x, r, u)) / 1e6
            if bw > best_bw:
                best, best_bw = f"R{r}U{u}", bw
            row += f"{bw:>9.0f}{'!' if ok else '?'}"
        blas = nbytes / bench(lambda: x @ Wt) / 1e6
        print(row + f"{blas:>8.0f}   {best} {best_bw:.0f} GB/s")

        del x, W, Wt, ref
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
