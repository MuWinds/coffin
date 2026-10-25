"""What should the inter-chunk scan become?

``chunked_delta`` is split into batched chunk-local work and a sequential scan.
The scan is still a Python loop that issues ~5 small ops per chunk per layer
(3 batched matmuls, a decay multiply, a strided store), which at n=2048 is ~2.3k
launches -- work that is tiny per op but launch-bound in aggregate.

Phase 1 is held fixed here and only the scan is varied, so the comparison is
about the scan alone:

  V0  the current loop: per chunk v_new, output, decay, state update, store
  V1  store the state entering each chunk, then compute all outputs in 4 batched
      matmuls over [H, nchunk] -- the sequential part drops to 3 ops per chunk
  V2  affine form: st_{i+1} = (cd_i*I - k2_i^T kcd_i) st_i + k2_i^T nv_i
      formed batched, applied with one baddbmm per chunk (fewer, bigger matmuls,
      ~1.8x the FLOPs), outputs batched like V1

Every variant is checked against the current one before being timed, and the
internal chunk size is swept again -- the earlier sweep predates the
strided-view fix, so its numbers are not trustworthy.

Run:  python probe_delta_scan_struct.py [C]

Module: ``coffin/diag``. Targets ``ops/delta.chunked_delta`` and the scan
structure inside ``models/qwen35/linear_attn``.

Constraints:
    * Phase 1 is held fixed so only the scan varies; changing both at once is
      what made the earlier measurements unattributable.
    * Per-op timing is meaningless here -- the scan is launch-bound, so the
      number to compare is total step time at a fixed chunk count.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "/root/private_data/engine")

import torch

from ops.delta import _upper_mask, chunked_delta
from ops.dtypes import NEG_INF

H, DK, DV = 32, 128, 128


def bench(fn, iters=5, warmup=2):
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


def phase1(q, k, v, g, beta, chunk):
    """The batched chunk-local half of ``chunked_delta``."""
    C = q.shape[1]
    nc = (C + chunk - 1) // chunk
    if nc * chunk != C:
        pad = nc * chunk - C
        q = torch.nn.functional.pad(q, (0, 0, 0, pad))
        k = torch.nn.functional.pad(k, (0, 0, 0, pad))
        v = torch.nn.functional.pad(v, (0, 0, 0, pad))
        g = torch.nn.functional.pad(g, (0, pad))
        beta = torch.nn.functional.pad(beta, (0, pad))
    L = chunk
    qc = q.reshape(H, nc, L, DK).contiguous()
    kc = k.reshape(H, nc, L, DK).contiguous()
    vc = v.reshape(H, nc, L, DV).contiguous()
    cum = g.reshape(H, nc, L).contiguous().cumsum(-1)
    pd = (cum.unsqueeze(-1) - cum.unsqueeze(-2)) \
        .masked_fill(_upper_mask(L, q.device), NEG_INF).exp()
    bc = beta.reshape(H, nc, L).contiguous()
    kb = kc * bc.unsqueeze(-1)
    vb = vc * bc.unsqueeze(-1)
    ut = (kb @ kc.transpose(-1, -2)) * pd
    ica = (qc @ kc.transpose(-1, -2)) * pd
    ecum = cum.exp()
    dkb = kb * ecum.unsqueeze(-1)
    nv = torch.linalg.solve_triangular(ut, vb, upper=False, unitriangular=True)
    kcd = torch.linalg.solve_triangular(ut, dkb, upper=False, unitriangular=True)
    q2 = qc * ecum.unsqueeze(-1)
    k2 = kc * (cum[..., -1:] - cum).exp().unsqueeze(-1)
    cd = cum[..., -1].exp()
    return nv, kcd, q2, ica, k2, cd, nc, L


def bmm4(a, b):
    """``[H, nc, m, k] @ [H, nc, k, n] -> [H, nc, m, n]`` (bmm over both batch dims)."""
    H_, nc_, m, k = a.shape
    n = b.shape[-1]
    return torch.bmm(a.reshape(H_ * nc_, m, k),
                     b.reshape(H_ * nc_, k, n)).reshape(H_, nc_, m, n)


def baddbmm4(inp, a, b, alpha=1.0, beta=1.0):
    """Batched ``beta*inp + alpha*(a @ b)`` over ``[H, nc, ...]``."""
    H_, nc_, m, k = a.shape
    n = b.shape[-1]
    return torch.baddbmm(inp.reshape(H_ * nc_, m, n),
                         a.reshape(H_ * nc_, m, k),
                         b.reshape(H_ * nc_, k, n),
                         alpha=alpha, beta=beta).reshape(H_, nc_, m, n)


def to_tokens(out):
    """``[H, nc, L, dv]`` -> the ``[C, H, dv]`` layout the engine returns."""
    H_, nc_, L_, dv_ = out.shape
    return out.permute(1, 2, 0, 3).reshape(nc_ * L_, H_, dv_)


def v0(nv, kcd, q2, ica, k2, cd, nc, L, state):
    """Current: one output store plus 3 matmuls per chunk."""
    out = torch.empty(nc, H, L, DV, dtype=nv.dtype, device=nv.device)
    st = state
    for i in range(nc):
        v_new = torch.baddbmm(nv[:, i], kcd[:, i], st, beta=1.0, alpha=-1.0)
        o_i = torch.bmm(q2[:, i], st)
        out[i] = torch.baddbmm(o_i, ica[:, i], v_new)
        st = torch.baddbmm(st * cd[:, i, None, None], k2[:, i].transpose(-1, -2), v_new)
    return out.permute(0, 2, 1, 3).reshape(nc * L, H, DV), st


def v1(nv, kcd, q2, ica, k2, cd, nc, L, state):
    """Store each chunk's incoming state, then compute outputs batched."""
    sts = torch.empty(H, nc, DK, DV, dtype=nv.dtype, device=nv.device)
    st = state
    for i in range(nc):
        sts[:, i] = st
        v_new = torch.baddbmm(nv[:, i], kcd[:, i], st, beta=1.0, alpha=-1.0)
        st = torch.baddbmm(st * cd[:, i, None, None], k2[:, i].transpose(-1, -2), v_new)
    v_new = nv - bmm4(kcd, sts)
    out = baddbmm4(bmm4(q2, sts), ica, v_new)
    return to_tokens(out), st


def v2(nv, kcd, q2, ica, k2, cd, nc, L, state):
    """Affine scan: A_i = cd_i*I - k2_i^T kcd_i, B_i = k2_i^T nv_i."""
    k2t = k2.transpose(-1, -2)
    A = bmm4(k2t, kcd).neg_()
    A.diagonal(dim1=-2, dim2=-1).add_(cd.unsqueeze(-1))     # + cd_i on the diagonal
    B = bmm4(k2t, nv)
    sts = torch.empty(H, nc, DK, DV, dtype=nv.dtype, device=nv.device)
    st = state
    for i in range(nc):
        sts[:, i] = st
        st = torch.baddbmm(B[:, i], A[:, i], st)
    v_new = nv - bmm4(kcd, sts)
    out = baddbmm4(bmm4(q2, sts), ica, v_new)
    return to_tokens(out), st


def rel(a, b):
    return (a - b).abs().max().item() / b.abs().max().item()


def main():
    C = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    torch.manual_seed(0)
    q = torch.randn(H, C, DK, device="cuda")
    k = torch.randn(H, C, DK, device="cuda")
    v = torch.randn(H, C, DV, device="cuda")
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6) * DK ** -0.5
    k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    beta = torch.rand(H, C, device="cuda")
    g = -torch.rand(H, C, device="cuda") * 0.5
    st0 = torch.randn(H, DK, DV, device="cuda") * 0.01
    st_ref = st0.clone()
    out_st = torch.empty_like(st0)

    print(f"C={C}  H={H} dk={DK} dv={DV}")
    whole = bench(lambda: chunked_delta(q, k, v, g, beta, st0, out_st, 64))
    print(f"whole delta (chunk=64) = {whole:.2f} ms; phase 1 alone:")
    for chunk in (32, 64, 128, 256):
        if C % chunk:
            continue
        print(f"  chunk={chunk:<4} phase1 {bench(lambda c=chunk: phase1(q, k, v, g, beta, c)):6.2f} ms")
    print()
    for chunk in (32, 64, 128, 256):
        if C % chunk:
            continue
        ref = chunked_delta(q, k, v, g, beta, st0, out_st, chunk)
        ref_st = out_st.clone()
        p = phase1(q, k, v, g, beta, chunk)
        nc, L = p[6], p[7]
        names = {}
        for nm, fn in (("v0", v0), ("v1", v1), ("v2", v2)):
            ao, ast = fn(*p, st0.clone())
            names[nm] = bench(lambda fn=fn: fn(*p, st0.clone()))
            e = rel(ao, ref)
            es = rel(ast, ref_st)
            flag = "OK " if (e < 1e-4 and es < 1e-4) else "BAD"
            print(f"  chunk={chunk:<4} nc={nc:<3} {nm}: {names[nm]:6.2f} ms  "
                  f"out {e:.1e} state {es:.1e} {flag}")
        print(f"  chunk={chunk:<4} nc={nc:<3} whole delta "
              f"{bench(lambda c=chunk: chunked_delta(q, k, v, g, beta, st0, out_st, c)):.2f} ms\n")


if __name__ == "__main__":
    main()
