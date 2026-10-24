"""Parity + timing probe for the batched linear-attn kernels.

Three batched kernels replace the per-row loops of the phase-1 decode path:

  conv1d_silu_batch   depthwise causal conv1d (kernel 4) + SiLU
  delta_prologue_batch q/k L2 norm + head expand + beta/g
  delta_batch         gated-delta-rule state update

A row of each must reproduce the matching single-row kernel, and the fp32
recurrent state must end up identical (the update is the same formula in the
same order, so bit equality is expected).  Shapes are the real Qwen3.5-4B
linear-attn shapes: nk=16, nv=32, dk=dv=128, conv_dim=8192, kernel=4.

Run from the engine root on the DCU container (GPU lock!)::

    flock /root/private_data/gpu.lock -c \
      'cd /root/private_data/engine_kernels && export ... PYTHONPATH=$PWD && \
       python3 diag/probe_delta_batch.py'

``--iters N`` changes the timing loop count.
"""

from __future__ import annotations

import sys

import torch

import hip_ops

NK, NV, DK, DV = 16, 32, 128, 128
REP = NV // NK
KDIM, VDIM = NK * DK, NV * DV
CONV_DIM = 2 * KDIM + VDIM          # 8192
ITERS = 200


def make_conv(B: int, seed: int = 1):
    torch.manual_seed(seed)
    w = torch.randn(CONV_DIM, 4, dtype=torch.float32, device="cuda") * 0.1
    x = (torch.randn(B, CONV_DIM, dtype=torch.float32, device="cuda") * 0.5).bfloat16()
    state = torch.randn(B, CONV_DIM, 3, dtype=torch.float32, device="cuda") * 0.2
    out = torch.empty(B, CONV_DIM, dtype=torch.float32, device="cuda")
    return w, x, state, out


def make_prologue(B: int, seed: int = 2):
    torch.manual_seed(seed)
    q = torch.randn(B, NK, DK, dtype=torch.float32, device="cuda")
    k = torch.randn(B, NK, DK, dtype=torch.float32, device="cuda")
    v = torch.randn(B, NV, DV, dtype=torch.float32, device="cuda")
    b = torch.randn(B, NV, dtype=torch.float32, device="cuda").bfloat16()
    a = torch.randn(B, NV, dtype=torch.float32, device="cuda").bfloat16()
    neg_expA = -torch.rand(NV, dtype=torch.float32, device="cuda") - 0.1
    dt_bias = torch.randn(NV, dtype=torch.float32, device="cuda") * 0.1
    out = [torch.empty(B, NV, DK, dtype=torch.float32, device="cuda"),
           torch.empty(B, NV, DK, dtype=torch.float32, device="cuda"),
           torch.empty(B, NV, DV, dtype=torch.float32, device="cuda"),
           torch.empty(B, NV, dtype=torch.float32, device="cuda"),
           torch.empty(B, NV, dtype=torch.float32, device="cuda")]
    qscale = DK ** -0.5
    return q, k, v, b, a, neg_expA, dt_bias, out, qscale


def make_delta(B: int, seed: int = 3):
    torch.manual_seed(seed)
    q = torch.randn(B, NV, DK, dtype=torch.float32, device="cuda") * 0.1
    k = torch.randn(B, NV, DK, dtype=torch.float32, device="cuda") * 0.1
    v = torch.randn(B, NV, DV, dtype=torch.float32, device="cuda") * 0.5
    beta = torch.rand(B, NV, dtype=torch.float32, device="cuda") * 0.9 + 0.05
    g = -(torch.rand(B, NV, dtype=torch.float32, device="cuda") * 0.3 + 0.001)
    st = torch.randn(B, NV, DK, DV, dtype=torch.float32, device="cuda") * 0.01
    out = torch.empty(B, NV, DV, dtype=torch.float32, device="cuda")
    return q, k, v, beta, g, st, out


# Reference implementations that call the single-row kernels.
def conv_ref(w, x, state, out):
    B = x.shape[0]
    ref = torch.empty_like(out)
    for b in range(B):
        hip_ops.conv1d_silu(w, x[b], state[b], ref[b])
    return ref


def prologue_ref(args):
    q, k, v, b_, a_, neg_expA, dt_bias, out, qscale = args
    B = q.shape[0]
    ref = [torch.empty_like(t) for t in out]
    for i in range(B):
        hip_ops.delta_prologue(q[i].contiguous(), k[i].contiguous(),
                               v[i].contiguous(), b_[i].contiguous(),
                               a_[i].contiguous(), neg_expA, dt_bias,
                               ref[0][i], ref[1][i], ref[2][i], ref[3][i],
                               ref[4][i], qscale, REP)
    return ref


def delta_ref(q, k, v, beta, g, st, out):
    B = q.shape[0]
    ref = torch.empty_like(out)
    st_ref = st.clone()
    for i in range(B):
        hip_ops.delta(q[i], k[i], v[i], beta[i], g[i], st_ref[i], ref[i])
    return ref, st_ref


# The same operations through the batched entry points.
def conv_batch(w, x, state, out):
    hip_ops.conv1d_silu_batch(w, x, state, out)
    return out


def prologue_batch(args):
    q, k, v, b_, a_, neg_expA, dt_bias, out, qscale = args
    hip_ops.delta_prologue_batch(q, k, v, b_, a_, neg_expA, dt_bias,
                                 out[0], out[1], out[2], out[3], out[4],
                                 qscale, REP)
    return out


def delta_batch(q, k, v, beta, g, st, out):
    hip_ops.delta_batch(q, k, v, beta, g, st, out)
    return out


def bench(fn, iters: int = ITERS):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(5):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) * 1000.0 / iters)
    return best


def errs(got, ref):
    d = (got.float() - ref.float()).abs()
    denom = ref.float().abs().max().clamp_min(1e-30)
    exact = (got == ref).float().mean().item()
    return d.max().item(), (d.max() / denom).item(), exact


def report(name, got, ref):
    mx, rel, exact = errs(got, ref)
    tag = "OK " if exact == 1.0 else "BAD"
    print(f"  {name:<34} max_abs {mx:>9.3e}  max_rel {rel:>9.3e}  "
          f"exact {exact:>7.4f}  {tag}")
    return exact == 1.0


def main() -> int:
    global ITERS
    if "--iters" in sys.argv:
        ITERS = int(sys.argv[sys.argv.index("--iters") + 1])

    if not hip_ops.available():
        print(f"HIP extension unavailable: {hip_ops.last_error()}")
        return 1
    print(f"DCU: {torch.cuda.get_device_name(0)}  torch {torch.__version__}  "
          f"hip_ops OK")
    print(f"linear-attn shapes: nk={NK} nv={NV} dk={DK} dv={DV} "
          f"conv_dim={CONV_DIM} rep={REP}")

    print("\n== parity vs single-row kernels ==")
    all_ok = True
    for B in (1, 2, 4, 8):
        w, x, state, out = make_conv(B)
        ref = conv_ref(w, x.clone(), state.clone(), out)
        got = conv_batch(w, x, state, out)
        print(f"B={B} conv1d_silu")
        all_ok &= report("out", got, ref)

        args = make_prologue(B)
        ref = prologue_ref(args)
        got = prologue_batch(args)
        print(f"B={B} delta_prologue")
        for name, a, b in zip(("q_pro", "k_pro", "v_pro", "beta", "g"), got, ref):
            all_ok &= report(name, a, b)

        q, k, v, beta, g, st, out = make_delta(B)
        ref, st_ref = delta_ref(q, k, v, beta, g, st.clone(), out)
        got = delta_batch(q, k, v, beta, g, st, out)
        print(f"B={B} delta")
        all_ok &= report("out", got, ref)
        all_ok &= report("state", st, st_ref)

    # strided prologue inputs: in the model q/k/v are column slices of the
    # [B, conv_dim] conv output (row pitch conv_dim, not packed).
    B = 4
    torch.manual_seed(9)
    qkv = torch.randn(B, CONV_DIM, dtype=torch.float32, device="cuda")
    q = qkv[:, :KDIM].reshape(B, NK, DK)
    k = qkv[:, KDIM:2 * KDIM].reshape(B, NK, DK)
    v = qkv[:, 2 * KDIM:].reshape(B, NV, DV)
    b_ = torch.randn(B, NV, dtype=torch.float32, device="cuda").bfloat16()
    a_ = torch.randn(B, NV, dtype=torch.float32, device="cuda").bfloat16()
    neg_expA = -torch.rand(NV, dtype=torch.float32, device="cuda") - 0.1
    dt_bias = torch.randn(NV, dtype=torch.float32, device="cuda") * 0.1
    out = [torch.empty(B, NV, DK, dtype=torch.float32, device="cuda"),
           torch.empty(B, NV, DK, dtype=torch.float32, device="cuda"),
           torch.empty(B, NV, DV, dtype=torch.float32, device="cuda"),
           torch.empty(B, NV, dtype=torch.float32, device="cuda"),
           torch.empty(B, NV, dtype=torch.float32, device="cuda")]
    qscale = DK ** -0.5
    hip_ops.delta_prologue_batch(q, k, v, b_, a_, neg_expA, dt_bias,
                                 out[0], out[1], out[2], out[3], out[4],
                                 qscale, REP)
    ref = prologue_ref((q.contiguous(), k.contiguous(), v.contiguous(), b_, a_,
                        neg_expA, dt_bias, out, qscale))
    print(f"B={B} delta_prologue (strided q/k/v column slices)")
    for name, g_, r_ in zip(("q_pro", "k_pro", "v_pro", "beta", "g"), out, ref):
        all_ok &= report(name, g_, r_)

    print("\n== per-step latency: batch kernel vs row-by-row loop ==")
    print(f"{'B':>2}  {'kernel':<20} {'batch us':>10} {'loop us':>10} "
          f"{'speedup':>8}")
    for B in (1, 2, 4, 8):
        w, x, state, out = make_conv(B)
        ref_out = torch.empty_like(out)

        def conv_loop():
            for i in range(B):
                hip_ops.conv1d_silu(w, x[i], state[i], ref_out[i])

        tb = bench(lambda: hip_ops.conv1d_silu_batch(w, x, state, out))
        tl = bench(conv_loop)
        print(f"{B:>2}  {'conv1d_silu':<20} {tb:>10.2f} {tl:>10.2f} {tl / tb:>8.2f}x")

        args = make_prologue(B)
        refq = [t.clone() for t in args[7]]
        bq, kq, vq, bb, aa, ne, db, oq, qs = args

        def prologue_loop():
            for i in range(B):
                hip_ops.delta_prologue(bq[i], kq[i], vq[i], bb[i], aa[i], ne, db,
                                       refq[0][i], refq[1][i], refq[2][i],
                                       refq[3][i], refq[4][i], qs, REP)

        tb = bench(lambda: hip_ops.delta_prologue_batch(
            bq, kq, vq, bb, aa, ne, db, oq[0], oq[1], oq[2], oq[3], oq[4], qs, REP))
        tl = bench(prologue_loop)
        print(f"{B:>2}  {'delta_prologue':<20} {tb:>10.2f} {tl:>10.2f} {tl / tb:>8.2f}x")

        q, k, v, beta, g, st, out = make_delta(B)
        st_loop = st.clone()
        out_loop = torch.empty_like(out)

        def delta_loop():
            # states diverge across iterations; timing only
            for i in range(B):
                hip_ops.delta(q[i], k[i], v[i], beta[i], g[i], st_loop[i],
                              out_loop[i])

        tb = bench(lambda: hip_ops.delta_batch(q, k, v, beta, g, st, out))
        tl = bench(delta_loop)
        print(f"{B:>2}  {'delta':<20} {tb:>10.2f} {tl:>10.2f} {tl / tb:>8.2f}x")

    print("\nall rows bit-identical to the single-row kernels"
          if all_ok else "\nSOME ROWS DIFFER -- see BAD lines above")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
