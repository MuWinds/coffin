"""Where does a batched prefill actually spend its time?

The end-to-end numbers in ``bench_prefill2.py`` say prefill is slow, but not
*what* is slow. This probe answers the attribution question: given one prefill
of ``n`` tokens, which layer, which projection and which kernel own the
milliseconds? It splits one prefill into

  * the whole model, n = 64..2048;
  * per-part timings of one linear-attention layer (projections / conv /
    head-prep / chunked delta / gated norm / out_proj), one MLP and one
    full-attention layer;
  * a cost model that scales the parts by their layer counts and compares the
    sum against the measured whole-model number;
  * the kernel-level table from ``torch.profiler`` for a whole prefill.

Read the cost-model table first: ``share`` is where the time goes, and
``sum of parts`` against the measured whole says whether the accounting is
complete. A sum well below the whole means time is hiding in code no row covers
(the block loop, embedding, sampling) — fix the table before trusting a
``share``. A single part above ~30 % is the next optimisation target; the
``per-layer sanity`` line checks each part list against that layer's own
``whole`` figure, and a part list that sums far above its ``whole`` is measuring
something the layer does not do per call.

Run inside the container, from the engine root, on the DCU:

    python probe_prefill_split.py [n ...]

``n`` defaults to 256, 512 and 2048. ``CHUNK`` selects the delta chunk size
(default 256), and ``SKIP_E2E=1`` skips the model-level sweep when only the
per-part table is wanted.

Module: ``coffin/diag``. Standalone probe: imports the split Qwen3.5 modules plus
torch, and nothing under ``coffin/`` imports this probe.

Constraints:
    * Reads the real checkpoint from ``MODEL`` and builds a real model, so this
      cannot run on a host without the weights; unlike the hardware probes it is
      not synthetic.
    * The direct imports are deliberate: this probe reaches into layer internals
      (``la.conv_state``, ``la.qscale``) that should not be re-exported from the
      family package.
    * Every part is timed on the same ``n`` and the same layer instance, so a
      part list is only comparable with the ``whole`` measured in that same call.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

import hip_ops
from models.qwen35 import Qwen35
from models.qwen35.full_attn import FullAttn
from models.qwen35.linear_attn import LinearAttn
from ops.delta import chunked_delta
from ops.dtypes import D, NEG_INF
from ops.rope import apply_rope

MODEL = "/root/private_data/models/Qwen3.5-4B"

BASE = [9707, 11, 1879, 3671, 374, 264, 3449, 315, 9847, 13,
        3838, 374, 279, 6722, 315, 9625, 323, 11751, 13]


def bench(fn, iters: int = 5, warmup: int = 2) -> float:
    """Time ``fn`` with CUDA events and return milliseconds per call.

    Args:
        fn: Zero-argument callable to time.
        iters: Calls per timed region; the elapsed time is divided by this.
        warmup: Untimed calls first, to absorb rocBLAS handle creation and
            kernel first-load.

    Returns:
        Fastest of three timed regions, in milliseconds per call. The minimum is
        used because a co-tenant can only add time to a region.
    """
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
    """Build ``n`` tokens by repeating a fixed English prompt fragment.

    Args:
        n: Requested prompt length.

    Returns:
        Token ids of length ``n``. The fragment is reused cyclically so every
        part is timed on content that exercises attention over real text rather
        than a single repeated token, which would flatten the softmax and
        understate the attention rows.
    """
    return [BASE[i % len(BASE)] for i in range(n)]


def end_to_end(model: Qwen35, sizes) -> dict[int, float]:
    """Time a full ``model.prefill`` at each prompt length.

    Args:
        model: The loaded model; it is reset before each measurement.
        sizes: Prompt lengths to sweep.

    Returns:
        ``{n: milliseconds}`` for the whole prefill. This is the denominator the
        cost model is judged against, so it is measured with the same
        reset-then-prefill sequence the cost model uses.
    """
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


def linear_parts(model: Qwen35, n: int) -> dict[str, float]:
    """Time each stage of one real linear-attention layer at length ``n``.

    Args:
        model: The loaded model; the first linear-attention layer is used as the
            representative one, since every such layer has the same shapes.
        n: Token count for the synthetic activation.

    Returns:
        ``{part_name: milliseconds}`` for projections, conv, head prep, chunked
        delta, gated norm and out_proj, plus ``la.whole`` for the layer called as
        a unit.

    The stages are chained, not independent: each stage's input comes from
    running the previous one, so the measured activation has the layout and
    magnitude the real layer produces.
    """
    la = next(l.mix for l in model.layers if isinstance(l.mix, LinearAttn))
    x = torch.randn(n, 2560, dtype=D, device="cuda") * 0.1
    state = torch.zeros_like(la.state)
    la.reset()

    def proj():
        return la.qkvzb_a.prefill(x)

    qkv, z, b, a = proj()

    # The engine's own fused kernels, not the eager chains they replaced: the
    # conv row used to bench the 3.6 ms unfold chain that conv1d_silu_seq
    # deleted, which made the table lie about where the time goes.
    def conv():
        return hip_ops.conv1d_silu_seq(la.conv_w, qkv, la.conv_state)

    qkvc = conv()

    def prep():
        """Run delta_prep_seq and return the five scan-layout buffers it writes.

        Returns:
            ``(qt, kt, vt, bt, gt)``, head-major ``[num_v, n, d]`` plus the
            per-head gate tensors. Allocating them inside the callable keeps the
            allocation cost in the row rather than in a cheaper neighbour.
        """
        qt = torch.empty(la.num_v, n, la.dk, dtype=torch.float32, device="cuda")
        kt = torch.empty(la.num_v, n, la.dk, dtype=torch.float32, device="cuda")
        vt = torch.empty(la.num_v, n, la.dv, dtype=torch.float32, device="cuda")
        bt = torch.empty(la.num_v, n, dtype=torch.float32, device="cuda")
        gt = torch.empty(la.num_v, n, dtype=torch.float32, device="cuda")
        hip_ops.delta_prep_seq(qkvc, b, a, la.neg_expA, la.dt_bias,
                               qt, kt, vt, bt, gt, la.qscale, la.rep)
        return qt, kt, vt, bt, gt

    qc, kc, vc, bt, gt = prep()
    z3 = z.reshape(n, la.num_v, la.dv)

    def delta_only():
        return chunked_delta(qc, kc, vc, gt, bt, state, state, 64)

    o_ = delta_only().reshape(n, la.num_v, la.dv)
    o_n = la.norm(o_, z3).reshape(n, la.vdim)

    # The transposed-input cost is gone now that the prep kernel writes the
    # scan's layout directly; keep the comparison so a regression shows up. The
    # two rows are timed on the same data, so a gap means the scan lost its
    # contiguous fast path.
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


def mlp_parts(model: Qwen35, n: int) -> dict[str, float]:
    """Time the gate/up projection, the down projection and the whole MLP.

    Args:
        model: The loaded model; layer 0's MLP stands in for all of them.
        n: Token count.

    Returns:
        ``{part_name: milliseconds}``. ``mlp.whole`` exists so the two
        projections can be checked against it; a gap is the activation function
        and the extra elementwise pass, not a missing row.
    """
    mlp = model.layers[0].mlp
    x = torch.randn(n, 2560, dtype=D, device="cuda") * 0.1
    g, u = mlp.gate_up.prefill(x)
    h = F.silu(g) * u
    return {
        "mlp.gate_up": bench(lambda: mlp.gate_up.prefill(x)),
        "mlp.down": bench(lambda: mlp.down.prefill(h)),
        "mlp.whole": bench(lambda: mlp.prefill(x)),
    }


def fa_parts(model: Qwen35, n: int) -> dict[str, float]:
    """Time each stage of one real full-attention layer at length ``n``.

    Args:
        model: The loaded model; the first full-attention layer is used.
        n: Token count. The causal mask and the RoPE tables are sliced to ``n``
            so the rows measure the same work the model does.

    Returns:
        ``{part_name: milliseconds}`` for qkv projection, RoPE plus INT8
        quantisation, the two einsums and the softmax, plus ``fa.whole``.

    The einsum and softmax stages are timed on pre-quantised inputs, which is why
    ``fa.rope+quant`` is a separate row: quantisation is part of the attention
    cost and is easy to leave out of the accounting entirely.
    """
    fa = next(l.mix for l in model.layers if isinstance(l.mix, FullAttn))
    x = torch.randn(n, 2560, dtype=D, device="cuda") * 0.1
    pos = torch.arange(n, device="cuda")
    ctx = (model.cos_tbl, model.sin_tbl, model.mask_idx)
    qg, k, v = fa.qkv.prefill(x)
    qg = qg.reshape(n, fa.nh, 2 * fa.hd)
    q = fa.q_norm(qg[..., :fa.hd])
    kn = fa.k_norm(k.reshape(n, fa.nkv, fa.hd))
    vn = v.reshape(n, fa.nkv, fa.hd)
    cos = model.cos_tbl.index_select(0, pos).unsqueeze(1)
    sin = model.sin_tbl.index_select(0, pos).unsqueeze(1)
    qr, kr = apply_rope(q, kn, cos, sin, fa.rot)
    qk, sk = fa._quant_seq(kr)                    # [n, nkv, hd]
    qv, sv = fa._quant_seq(vn)
    kk, kkp = qk.transpose(0, 1).to(D), sk.transpose(0, 1)
    vv, vvp = qv.transpose(0, 1).to(D), sv.transpose(0, 1)

    def eins1():
        qe = qr.reshape(n, fa.nkv, fa.rep, fa.hd).permute(1, 2, 0, 3)
        return torch.einsum("ghcd,gsd->ghcs", qe, kk).float()

    sc = eins1()

    def soft():
        """Apply the causal mask and dequant scale, then softmax in fp32.

        Returns:
            Attention probabilities ``[nkv, rep, n, n]``. The mask comes from the
            model's own ``mask_idx`` table rather than a freshly built
            ``triu``, so the row measures the production masking path.
        """
        s = sc * kkp[:, None, None, :] * fa.scale
        mask = (model.mask_idx[None, :n] > pos[:, None])[None, None, :, :]
        return s.masked_fill(mask, NEG_INF).softmax(-1).to(D)

    pr = soft()

    def eins2():
        pv = pr * vvp[:, None, None, :].to(D)
        return torch.einsum("ghcs,gsd->ghcd", pv, vv)

    return {
        "fa.qkv": bench(lambda: fa.qkv.prefill(x)),
        "fa.rope+quant": bench(lambda: fa._quant_seq(apply_rope(q, kn, cos, sin, fa.rot)[1])),
        "fa.eins1": bench(eins1),
        "fa.softmax": bench(soft),
        "fa.eins2": bench(eins2),
        "fa.whole": bench(lambda: fa.prefill(x, pos, ctx)),
    }


def report(model: Qwen35, n: int, chunk: int = 256) -> None:
    """Print the layered cost model for one prompt length.

    Args:
        model: The loaded model.
        n: Prompt length to attribute.
        chunk: Delta chunk size, forwarded to the measured prefill so the
            whole-model denominator uses the same setting as the parts.

    A part group whose timing raises is skipped with a message and the rest of
    the table is printed: one uninterruptible part must not cost the whole
    report. Missing keys then surface as a ``KeyError`` in the table loop, which
    is deliberate — a silently short table would read as an optimisation win.
    """
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


def profile_model(model: Qwen35, n: int, top: int = 30) -> None:
    """Print the kernel-level attribution for one whole prefill.

    Args:
        model: The loaded model.
        n: Prompt length to profile.
        top: Number of kernels to print, sorted by self device time.

    Rows carry self time, not total time, so a kernel is not credited for the
    kernels it launched; ``launches`` is printed next to the device total because
    a prefill dominated by launch count is a fusion problem, not a bandwidth one.
    """
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
    """Run the size sweep, the cost model and the kernel table.

    Prompt lengths come from argv (default 256, 512, 2048); the kernel table is
    taken at the largest one, where launch overhead no longer hides the shapes
    that matter.
    """
    sizes = [int(a) for a in sys.argv[1:]] or [256, 512, 2048]
    chunk = int(os.environ.get("CHUNK", "256"))
    print(f"device={torch.cuda.get_device_name(0)}  torch={torch.__version__}")
    print(f"hip_ops={hip_ops.available()}  {hip_ops.last_error() or ''}")
    model = Qwen35(MODEL, max_seq=4096, bench=False)
    model.reset()
    model.prefill(toks(256))              # warm every path (rocBLAS handles, TRSM...)
    if not os.environ.get("SKIP_E2E"):
        end_to_end(model, [64, 256, 512, 1024, 2048])
    for n in sizes:
        report(model, n, chunk)
    profile_model(model, sizes[-1])


if __name__ == "__main__":
    main()
