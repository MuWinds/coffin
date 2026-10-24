"""True GPU cost per op, measured inside a captured graph (removes CPU dispatch).

The plain microbenchmark is CPU-bound on this box (~100us per op dispatch), so
every op looks the same size.  Capturing N copies of an op into a graph and
timing the replay gives the real device time.

Run:  python probe_ops2.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

D = torch.bfloat16
DEV = "cuda"


def graph_time(fn, iters=50, reps=5):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / reps / iters * 1000        # us per op
    del g


def main():
    print(f"device={torch.cuda.get_device_name(0)}")

    print("\nnorms")
    for name, shape in [("hidden2560", (2560,)), ("gated32x128", (32, 128))]:
        x = torch.randn(*shape, dtype=D, device=DEV)
        w = torch.randn(shape[-1], dtype=D, device=DEV)
        g = (1.0 + w.float()).to(D)
        print(f"{name:<12} fused {graph_time(lambda: F.rms_norm(x, (shape[-1],), g, 1e-6)):6.2f}us"
              f"   manual-fp32 {graph_time(lambda: (x.float()*torch.rsqrt(x.float().pow(2).mean(-1,keepdim=True)+1e-6)*w.float()).to(D)):6.2f}us")

    print("\nelementwise on [2560]")
    a = torch.randn(2560, dtype=D, device=DEV)
    b = torch.randn(2560, dtype=D, device=DEV)
    print(f"add      {graph_time(lambda: a+b):6.2f}us   mul {graph_time(lambda: a*b):6.2f}us"
          f"   silu {graph_time(lambda: F.silu(a)):6.2f}us")

    print("\ndelta rule (32x128x128 fp32 state, 2.1 MB)")
    H, dk, dv = 32, 128, 128
    st = torch.randn(H, dk, dv, dtype=torch.float32, device=DEV) * 0.01
    q = torch.randn(H, dk, dtype=torch.float32, device=DEV)
    k = torch.randn(H, dk, dtype=torch.float32, device=DEV)
    v = torch.randn(H, dv, dtype=torch.float32, device=DEV)
    gg = -torch.rand(H, dtype=torch.float32, device=DEV)
    beta = torch.rand(H, dtype=torch.float32, device=DEV)

    def chain():
        st.mul_(gg.exp()[:, None, None])
        kv = (st * k[:, :, None]).sum(-2)
        delta = (v - kv) * beta[:, None]
        st.add_(k[:, :, None] * delta[:, None, :])
        return (st * q[:, :, None]).sum(-2)

    print(f"full chain {graph_time(chain, iters=30):6.2f}us")

    print("\nint8 KV widen (4x2048x256)")
    ki = torch.randint(-127, 127, (4, 2048, 256), dtype=torch.int8, device=DEV)
    ks = torch.rand(4, 2048, dtype=torch.float32, device=DEV)
    print(f"to(bf16) {graph_time(lambda: ki.to(D)):6.2f}us"
          f"   float*scale {graph_time(lambda: (ki.float()*ks[:,:,None]).to(D)):6.2f}us")

    print("\nbatched attention (nkv=4, rep=4, S=2048, hd=256)")
    qe = torch.randn(4, 4, 256, dtype=D, device=DEV)
    kk = torch.randn(4, 2048, 256, dtype=D, device=DEV)
    probs = torch.rand(4, 4, 2048, dtype=D, device=DEV)
    print(f"scores {graph_time(lambda: torch.einsum('ghd,gsd->ghs', qe, kk)):6.2f}us"
          f"   out {graph_time(lambda: torch.einsum('ghs,gsd->ghd', probs, kk)):6.2f}us")

    print("\nconv1d depthwise + silu on [8192]")
    win = torch.randn(8192, 4, dtype=torch.float32, device=DEV)
    cw = torch.randn(8192, 4, dtype=torch.float32, device=DEV)
    print(f"mul+sum {graph_time(lambda: (win*cw).sum(-1)):6.2f}us"
          f"   cat {graph_time(lambda: torch.cat([win[:, :3], win[:, 3:]], 1)):6.2f}us")


if __name__ == "__main__":
    main()
