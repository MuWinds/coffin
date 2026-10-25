"""A/B the two scan structures inside one process, end to end.

``probe_delta_scan_struct.py`` says the affine scan is 2.3x faster than the
per-chunk form when called on its own, yet swapping it into the engine did not
move ``bench_prefill2``.  Cross-run noise is 1-2%, so this holds everything else
fixed -- same process, same weights, same warmup -- and alternates the two
implementations through ``engine.chunked_delta`` so the comparison is exact.
It also times one layer's whole prefill and the model, so the delta's real share
is visible rather than assumed.

Run:  python probe_scan_ab.py [rounds]

Module: ``coffin/diag``. A/Bs the two scan implementations inside one process,
end to end, through ``llm_engine`` / ``bench_prefill2``.

Constraints:
    * Same process, same weights, same warmup, alternating rounds: cross-run
      noise on this box is 1-2%, which is larger than the effect being measured, so
      a two-process comparison would not resolve it.
    * The standalone win (2.3x in ``probe_delta_scan_struct``) does not show up
      end to end; this probe is what proves that, so do not "fix" it by re-running
      the microbenchmark.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "/root/private_data/engine")

import torch

import qwen35_engine as E
from qwen35_engine import LinearAttn

MODEL = "/root/private_data/models/Qwen3.5-4B"
BASE = [9707, 11, 1879, 3671, 374, 264, 3449, 315, 9847, 13,
        3838, 374, 279, 6722, 315, 9625, 323, 11751, 13]

F = torch.nn.functional
NEG_INF = E.NEG_INF


def chunked_delta_v0(q, k, v, g, beta, state, out_state, chunk=64):
    """The pre-baddbmm-free per-chunk scan: 3 small matmuls plus a store per chunk."""
    H, C, dk = q.shape
    dv = v.shape[-1]
    nc = (C + chunk - 1) // chunk
    if nc * chunk != C:
        pad = nc * chunk - C
        q = F.pad(q, (0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
        g = F.pad(g, (0, pad))
        beta = F.pad(beta, (0, pad))
    L = chunk
    qc = q.reshape(H, nc, L, dk).contiguous()
    kc = k.reshape(H, nc, L, dk).contiguous()
    vc = v.reshape(H, nc, L, dv).contiguous()
    cum = g.reshape(H, nc, L).contiguous().cumsum(-1)
    pd = (cum.unsqueeze(-1) - cum.unsqueeze(-2)) \
        .masked_fill(E._upper_mask(L, q.device), NEG_INF).exp()
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

    out = torch.empty(nc, H, L, dv, dtype=v.dtype, device=v.device)
    st = state
    for i in range(nc):
        v_new = nv[:, i] - kcd[:, i] @ st
        o_i = q2[:, i] @ st
        out[i] = o_i + ica[:, i] @ v_new
        st = st * cd[:, i, None, None] + k2[:, i].transpose(-1, -2) @ v_new
    out_state.copy_(st)
    return out.permute(0, 2, 1, 3).reshape(nc * L, H, dv)[:C]


def bench(fn, iters=3, warmup=1):
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


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    model = E.Qwen35(MODEL, max_seq=4096, bench=False)
    n = 2048
    toks = [BASE[i % len(BASE)] for i in range(n)]
    affine = E.chunked_delta
    la = next(l.mix for l in model.layers if isinstance(l.mix, LinearAttn))
    x = torch.randn(n, 2560, dtype=E.D, device="cuda") * 0.1

    def layer_prefill():
        return la.prefill(x)

    print(f"n={n}  delta chunk={la.prefill.__defaults__}  rounds={rounds}")
    for rnd in range(rounds):
        for name, impl in (("v0 loop ", chunked_delta_v0), ("affine  ", affine)):
            E.chunked_delta = impl
            E.LinearAttn.prefill.__globals__["chunked_delta"] = impl
            model.reset()
            ms = bench(lambda: (model.reset(), model.prefill(toks))[1], iters=2, warmup=1)
            la_ms = bench(layer_prefill)
            print(f"  round {rnd} {name} model {ms:8.1f} ms ({n / ms * 1e3:5.0f} tok/s)"
                  f"   one layer {la_ms:6.2f} ms")
    E.chunked_delta = affine
    E.LinearAttn.prefill.__globals__["chunked_delta"] = affine


if __name__ == "__main__":
    main()
