"""Where does a batched prefill actually spend its time?

The end-to-end numbers in ``bench_prefill2.py`` say prefill is slow, but not
*what* is slow.  This splits one prefill into

  * the whole model, n = 64..2048;
  * per-part timings of one linear-attention layer (projections / conv /
    head-prep / chunked delta / gated norm / out_proj), one MLP and one
    full-attention layer;
  * a cost model that scales the parts by their layer counts and compares the
    sum against the measured whole-model number;
  * the kernel-level table from ``torch.profiler`` for a whole prefill.

Run:  python probe_prefill_split.py [n ...]
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

import qwen35_engine as E
from qwen35_engine import FullAttn, LinearAttn, chunked_delta

MODEL = "/root/private_data/models/Qwen3.5-4B"

BASE = [9707, 11, 1879, 3671, 374, 264, 3449, 315, 9847, 13,
        3838, 374, 279, 6722, 315, 9625, 323, 11751, 13]


def bench(fn, iters: int = 5, warmup: int = 2) -> float:
    """Milliseconds per call, min-of-batches (CUDA events)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
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


def toks(n: int) -> list[int]:
    return [BASE[i % len(BASE)] for i in range(n)]


def end_to_end(model: E.Qwen35, sizes) -> dict[int, float]:
    print("\n=== whole-model prefill ===")
    print(f"{'n':>6}{'ms':>10}{'ms/token':>11}{'tok/s':>10}")
    out = {}
    for n in sizes:
        t = toks(n)
        model.reset()
        ms = bench(lambda: (model.reset(), model.prefill(t))[1], iters=3, warmup=1)
        out[n] = ms
        print(f"{n:>6}{ms:>10.1f}{ms / n:>11.3f}{n / ms * 1e3:>10.0f}")
    return out


def linear_parts(model: E.Qwen35, n: int) -> dict[str, float]:
    la = next(l.mix for l in model.layers if isinstance(l.mix, LinearAttn))
    x = torch.randn(n, 2560, dtype=E.D, device="cuda") * 0.1
    state = torch.zeros_like(la.state)
    la.reset()

    def proj():
        return la.qkvzb_a.prefill(x)

    qkv, z, b, a = proj()

    # The engine's own fused kernels, not the eager chains they replaced: the
    # conv row used to bench the 3.6 ms unfold chain that conv1d_silu_seq
    # deleted, which made the table lie about where the time goes.
    def conv():
        return E.hip_ops.conv1d_silu_seq(la.conv_w, qkv, la.conv_state)

    qkvc = conv()

    def prep():
        qt = torch.empty(la.num_v, n, la.dk, dtype=torch.float32, device="cuda")
        kt = torch.empty(la.num_v, n, la.dk, dtype=torch.float32, device="cuda")
        vt = torch.empty(la.num_v, n, la.dv, dtype=torch.float32, device="cuda")
        bt = torch.empty(la.num_v, n, dtype=torch.float32, device="cuda")
        gt = torch.empty(la.num_v, n, dtype=torch.float32, device="cuda")
        E.hip_ops.delta_prep_seq(qkvc, b, a, la.neg_expA, la.dt_bias,
                                 qt, kt, vt, bt, gt, la.qscale, la.rep)
        return qt, kt, vt, bt, gt

    qc, kc, vc, bt, gt = prep()
    z3 = z.reshape(n, la.num_v, la.dv)

    def delta_only():
        return chunked_delta(qc, kc, vc, gt, bt, state, state, 64)

    o_ = delta_only().reshape(n, la.num_v, la.dv)
    o_n = la.norm(o_, z3).reshape(n, la.vdim)

    # The transposed-input cost is gone now that the prep kernel writes the
    # scan's layout directly; keep the comparison so a regression shows up.
    qt, kt, vt = (t.transpose(0, 1).contiguous().transpose(0, 1) for t in (qc, kc, vc))
    print(f"  (layout check: contiguous-in {bench(delta_only):.2f} ms vs "
          f"transposed-in {bench(lambda: chunked_delta(qt, kt, vt, gt, bt, state, state, 64)):.2f} ms)")

    return {
        "la.proj": bench(proj),
        "la.conv": bench(conv),
        "la.prep": bench(prep),
        "la.delta": bench(delta_only),
        "la.norm": bench(lambda: la.norm(o_, z3)),
        "la.out_proj": bench(lambda: la.out_proj.prefill(o_n)),
        "la.whole": bench(lambda: la.prefill(x)),
    }


def mlp_parts(model: E.Qwen35, n: int) -> dict[str, float]:
    mlp = model.layers[0].mlp
    x = torch.randn(n, 2560, dtype=E.D, device="cuda") * 0.1
    g, u = mlp.gate_up.prefill(x)
    h = F.silu(g) * u
    return {
        "mlp.gate_up": bench(lambda: mlp.gate_up.prefill(x)),
        "mlp.down": bench(lambda: mlp.down.prefill(h)),
        "mlp.whole": bench(lambda: mlp.prefill(x)),
    }


def fa_parts(model: E.Qwen35, n: int) -> dict[str, float]:
    fa = next(l.mix for l in model.layers if isinstance(l.mix, FullAttn))
    x = torch.randn(n, 2560, dtype=E.D, device="cuda") * 0.1
    pos = torch.arange(n, device="cuda")
    ctx = (model.cos_tbl, model.sin_tbl, model.mask_idx)
    qg, k, v = fa.qkv.prefill(x)
    qg = qg.reshape(n, fa.nh, 2 * fa.hd)
    q = fa.q_norm(qg[..., :fa.hd])
    kn = fa.k_norm(k.reshape(n, fa.nkv, fa.hd))
    vn = v.reshape(n, fa.nkv, fa.hd)
    cos = model.cos_tbl.index_select(0, pos).unsqueeze(1)
    sin = model.sin_tbl.index_select(0, pos).unsqueeze(1)
    qr, kr = E.apply_rope(q, kn, cos, sin, fa.rot)
    qk, sk = fa._quant_seq(kr)                    # [n, nkv, hd]
    qv, sv = fa._quant_seq(vn)
    kk, kkp = qk.transpose(0, 1).to(E.D), sk.transpose(0, 1)
    vv, vvp = qv.transpose(0, 1).to(E.D), sv.transpose(0, 1)

    def eins1():
        qe = qr.reshape(n, fa.nkv, fa.rep, fa.hd).permute(1, 2, 0, 3)
        return torch.einsum("ghcd,gsd->ghcs", qe, kk).float()

    sc = eins1()

    def soft():
        s = sc * kkp[:, None, None, :] * fa.scale
        mask = (model.mask_idx[None, :n] > pos[:, None])[None, None, :, :]
        return s.masked_fill(mask, E.NEG_INF).softmax(-1).to(E.D)

    pr = soft()

    def eins2():
        pv = pr * vvp[:, None, None, :].to(E.D)
        return torch.einsum("ghcs,gsd->ghcd", pv, vv)

    return {
        "fa.qkv": bench(lambda: fa.qkv.prefill(x)),
        "fa.rope+quant": bench(lambda: fa._quant_seq(E.apply_rope(q, kn, cos, sin, fa.rot)[1])),
        "fa.eins1": bench(eins1),
        "fa.softmax": bench(soft),
        "fa.eins2": bench(eins2),
        "fa.whole": bench(lambda: fa.prefill(x, pos, ctx)),
    }


def report(model: E.Qwen35, n: int, chunk: int = 256) -> None:
    n_la = sum(1 for l in model.layers if isinstance(l.mix, LinearAttn))
    n_fa = len(model.layers) - n_la
    parts: dict[str, float] = {}
    for group, fn in (("linear", linear_parts), ("mlp", mlp_parts), ("full", fa_parts)):
        try:
            parts.update(fn(model, n))
        except Exception as exc:                      # keep the rest of the report
            print(f"\n!! {group} parts failed: {type(exc).__name__}: {exc}")
    whole = bench(lambda: (model.reset(), model.prefill(toks(n), chunk=chunk))[1],
                  iters=3, warmup=1)

    owners = {"la": n_la, "mlp": len(model.layers), "fa": n_fa}
    print(f"\n=== cost model, n={n} chunk={chunk} = {whole:.1f} ms "
          f"({whole / n:.3f} ms/token, {n / whole * 1e3:.0f} tok/s) ===")
    print(f"{'part':<16}{'ms':>9}{'x layers':>10}{'ms total':>10}{'share':>8}")
    total = 0.0
    for name, key in (("in/qkv/z/b/a proj", "la.proj"), ("causal conv", "la.conv"),
                      ("head prep (norm/beta/g)", "la.prep"), ("chunked delta", "la.delta"),
                      ("gated norm", "la.norm"), ("out_proj", "la.out_proj"),
                      ("mlp gate_up", "mlp.gate_up"), ("mlp down", "mlp.down"),
                      ("fa qkv proj", "fa.qkv"), ("fa rope+quant", "fa.rope+quant"),
                      ("fa qk^T", "fa.eins1"), ("fa softmax", "fa.softmax"),
                      ("fa probs@v", "fa.eins2")):
        cnt = owners[key.split(".")[0]]
        ms = parts[key] * cnt
        total += ms
        print(f"{name:<16}{parts[key]:>9.2f}{cnt:>10}{ms:>10.1f}{ms / whole * 100:>7.1f}%")
    print(f"{'sum of parts':<16}{'':>9}{'':>10}{total:>10.1f}{total / whole * 100:>7.1f}%")
    print(f"per-layer sanity: la.whole={parts['la.whole']:.2f} "
          f"mlp.whole={parts['mlp.whole']:.2f} fa.whole={parts['fa.whole']:.2f} ms")
    print(f"scaled by layers: {n_la * parts['la.whole']:.1f} + "
          f"{len(model.layers) * parts['mlp.whole']:.1f} + "
          f"{n_fa * parts['fa.whole']:.1f} = "
          f"{n_la * parts['la.whole'] + len(model.layers) * parts['mlp.whole'] + n_fa * parts['fa.whole']:.1f} ms")


def profile_model(model: E.Qwen35, n: int, top: int = 30) -> None:
    from torch.profiler import ProfilerActivity, profile

    t = toks(n)
    model.reset()
    model.prefill(t)                      # untimed warmup
    model.reset()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        model.prefill(t)
    torch.cuda.synchronize()

    rows = [(ev.self_device_time_total, ev.count, ev.key) for ev in prof.key_averages()
            if (ev.self_device_time_total or 0) > 0]
    rows.sort(reverse=True)
    total = sum(r[0] for r in rows)
    print(f"\n=== kernel table, whole prefill n={n}: {total / 1e3:.1f} ms device, "
          f"{sum(r[1] for r in rows)} launches ===")
    for t_us, cnt, key in rows[:top]:
        print(f"{t_us / 1e3:>9.2f} ms {t_us / total * 100:>5.1f}% {cnt:>6}x  {key[:84]}")


def main() -> None:
    sizes = [int(a) for a in sys.argv[1:]] or [256, 512, 2048]
    chunk = int(os.environ.get("CHUNK", "256"))
    print(f"device={torch.cuda.get_device_name(0)}  torch={torch.__version__}")
    print(f"hip_ops={E.hip_ops.available()}  {E.hip_ops.last_error() or ''}")
    model = E.Qwen35(MODEL, max_seq=4096, bench=False)
    model.reset()
    model.prefill(toks(256))              # warm every path (rocBLAS handles, TRSM...)
    if not os.environ.get("SKIP_E2E"):
        end_to_end(model, [64, 256, 512, 1024, 2048])
    for n in sizes:
        report(model, n, chunk)
    profile_model(model, sizes[-1])


if __name__ == "__main__":
    main()
