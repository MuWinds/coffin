"""Clean attribution for the current prefill: kernels only, host ops, and the
strided-store question.

Earlier tables mixed torch *op* rows with kernel rows, which double counts (an
``aten::mm`` row and its ``Cijk_*`` kernel carry the same device time).  This
filters to kernel rows only, names the host-side dispatcher hotspots, attributes
``aten::copy_`` by input shape, and micro-benchmarks the two things the profile
points at: the delta scan's strided per-chunk store, and the causal conv chain.

Run:  python probe_prefill_ops.py [n] [chunk]

Module: ``coffin/diag``. Filters a profiler table down to kernel rows; the host
side belongs to ``aten`` dispatch in ``ops/``.

Constraints:
    * Kernel rows only. Mixing in torch op rows double counts the same device
      time, which is the error this probe exists to correct.
    * The strided-store question is measured here rather than argued about: the
      store pattern is what decides whether a fusion is worth writing.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "/root/private_data/engine")

import torch
import torch.nn.functional as F

from models.qwen35 import Qwen35
from models.qwen35.linear_attn import LinearAttn
from ops.dtypes import D

MODEL = "/root/private_data/models/Qwen3.5-4B"
BASE = [9707, 11, 1879, 3671, 374, 264, 3449, 315, 9847, 13,
        3838, 374, 279, 6722, 315, 9625, 323, 11751, 13]


def bench(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(3):
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
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    chunk = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
    model = Qwen35(MODEL, max_seq=4096, bench=False)
    toks = [BASE[i % len(BASE)] for i in range(n)]

    print(f"device={torch.cuda.get_device_name(0)}  n={n} chunk={chunk}")
    model.reset()
    model.prefill(toks, chunk=chunk)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    ms = bench(lambda: (model.reset(), model.prefill(toks, chunk=chunk))[1], iters=3, warmup=1)
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"prefill: {ms:.1f} ms  {n / ms * 1e3:.0f} tok/s   peak VRAM {peak:.2f} GiB")

    from torch.profiler import ProfilerActivity, DeviceType, profile

    model.reset()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        model.prefill(toks, chunk=chunk)
    torch.cuda.synchronize()

    krows = [(ev.self_device_time_total, ev.count, ev.key) for ev in prof.key_averages()
             if ev.device_type == DeviceType.CUDA and ev.count and "Memcpy" not in ev.key]
    dev = sum(r[0] for r in krows) / 1e3
    launches = sum(r[1] for r in krows)
    print(f"\nkernels only: {dev:.1f} ms device, {launches} launches, "
          f"wall/kernel {ms / launches * 1e3:.1f} us")
    krows.sort(reverse=True)
    for t_us, cnt, key in krows[:22]:
        print(f"{t_us / 1e3:>9.2f} ms {t_us / 1e3 / dev * 100:>5.1f}% {cnt:>6}x  {key[:66]}")

    crows = [(ev.self_cpu_time_total, ev.count, ev.key) for ev in prof.key_averages()
             if ev.self_cpu_time_total]
    crows.sort(reverse=True)
    print("\ntop host-side (self CPU time):")
    for t_us, cnt, key in crows[:14]:
        print(f"{t_us / 1e3:>9.2f} ms {cnt:>6}x  {key[:66]}")

    print("\naten::copy_ by input shape:")
    shaped = [(ev.self_device_time_total, ev.count, ev.key) for ev in
              prof.key_averages(group_by_input_shape=True)
              if ev.key.startswith("aten::copy_") and (ev.self_device_time_total or 0) > 0]
    shaped.sort(reverse=True)
    for t_us, cnt, key in shaped[:10]:
        print(f"{t_us / 1e3:>9.2f} ms {cnt:>6}x  {key[:110]}")

    # Check whether the per-chunk strided store is the ``copy_`` hotspot.
    la = next(l.mix for l in model.layers if isinstance(l.mix, LinearAttn))
    H, L, dv = 32, 64, 128
    nc = 32
    src = torch.randn(H, L, dv, device="cuda")
    strided = torch.empty(H, nc, L, dv, device="cuda")
    blocked = torch.empty(nc, H, L, dv, device="cuda")
    print(f"\nper-chunk store of [H={H},L={L},dv={dv}] fp32 = "
          f"{src.numel() * 4 / 1e6:.2f} MB")
    print(f"  strided  out[:, i] = src   {bench(lambda: strided[:, 5].copy_(src)) * 1e3:8.1f} us")
    print(f"  blocked  out[i]   = src    {bench(lambda: blocked[5].copy_(src)) * 1e3:8.1f} us")
    flat = strided.reshape(-1)[:src.numel()].view_as(src)
    print(f"  contiguous view copy       {bench(lambda: flat.copy_(src)) * 1e3:8.1f} us")

    # The causal conv chain at the model's chunk size.
    C = chunk
    qkv = torch.randn(C, la.conv_dim, dtype=D, device="cuda")
    print(f"\ncausal conv1d+silu chain, C={C}, conv_dim={la.conv_dim}:")

    def chain():
        seq = torch.cat([la.conv_state.t(), qkv.float()], dim=0)
        la.conv_state.copy_(seq[-la.kernel + 1:].t())
        win = seq.unfold(0, la.kernel, 1)
        return F.silu((win * la.conv_w).sum(-1))

    print(f"  current chain              {bench(chain) * 1e3:8.1f} us")
    for name, fn in (("cat only", lambda: torch.cat([la.conv_state.t(), qkv.float()], dim=0)),
                     ("float() only", lambda: qkv.float()),
                     ("unfold+mul+sum", lambda: (torch.cat([la.conv_state.t(), qkv.float()],
                                                            dim=0).unfold(0, la.kernel, 1)
                                                 * la.conv_w).sum(-1))):
        print(f"  {name:<26} {bench(fn) * 1e3:8.1f} us")


if __name__ == "__main__":
    main()
