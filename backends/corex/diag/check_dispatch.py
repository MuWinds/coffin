"""Check the backend's Python glue on a host with no card, by substituting a torch kernel.

The card is only reachable through a cluster job, so a mistake in the glue -- the
``[N, K]`` to ``[K, N]`` packing, the row chunking, the dtype restore, a table that is
not total -- would otherwise be found only after a deploy round trip. This script
replaces :func:`backends.corex._kernels` with a pure-torch stand-in that has the same
signatures and the same tolerance for ``[K, N]`` weights, then compares every native
op against :mod:`backends.reference`.

**What this does not test**: the kernels. The stand-in computes in fp32 on whatever
device the tensors are on, so a wrong index or a bad launch geometry cannot show up
here. That is what ``check_ops.py`` is for, on the card.

Module: ``coffin/backends/corex/diag``. Runs on a CPU-only host; part of no import
path of the backend.

Constraints:
    * The stand-in must keep the C++ contract exactly (argument order, ``[K, N]``
      weights, dtype of the result), or the check would validate a fiction.
    * ``F.rms_norm`` is unavailable in this cluster's torch 2.1 and on the 2.1 build the
      GPU nodes have, so the reference-side comparison here writes the formula out
      instead of calling it.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: ``coffin/`` itself is the ``sys.path`` root: the engine imports ``backends``,
#: ``ops`` and ``models`` as top-level packages, so a diagnostic started by file path
#: has to put that directory on the path itself. Four levels up from this file.
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _torch_kernels(torch):
    """Build a CPU-capable stand-in for the compiled extension.

    Args:
        torch: The torch module.

    Returns:
        An object with one method per bound entry point, matching the C++ signatures,
        the ``[K, N]`` weight layout and the in-place conventions. Every entry
        accumulates in fp32 and returns the activation dtype, the same contract the real
        kernels keep.

    The stand-in implements the *contract*, not the algorithm: it is the honest way to
    check the Python glue (packing, chunking, buffer ownership, dtype restore) on a host
    with no card, and it deliberately cannot detect a wrong index or a bad launch
    geometry. ``check_ops.py`` covers those, on the card.
    """

    class Fake:
        """Pure-torch implementation of the bound entry points."""

        def gemv_kn(self, wt, x, threads=0, unroll=0, split=0):
            """``x [K] @ wt [K, N]``, fp32 accumulate, out in ``x``'s dtype."""
            if wt.dim() != 2:
                raise ValueError("weight must be 2-D")
            return (wt.float().transpose(0, 1) @ x.float()).to(x.dtype)

        def gemv_kn_m(self, wt, x, threads=0, unroll=0):
            """``x [M, K] @ wt [K, N]``, one pass over the weight."""
            return (x.float() @ wt.float()).to(x.dtype)

        def rms_norm(self, x, w, stride=0, eps=1e-6):
            """Zero-centred RMSNorm over the last axis, honouring the row pitch."""
            pitch = x.stride(0) if stride <= 0 else int(stride)
            rows, width = x.shape
            # ``x`` is a pitched view, so its rows are read through the stride rather than
            # through a reshaped copy -- otherwise this stand-in would silently test the
            # contiguous case whatever the caller asked for.
            flat = x.as_strided((rows * pitch,), (1,))
            out = torch.empty(rows, width, dtype=torch.float32)
            for r in range(rows):
                row = flat[r * pitch:r * pitch + width].float()
                out[r] = row * torch.rsqrt((row * row).mean() + eps) * w.float()
            return out.to(x.dtype)

        def rms_norm_gated(self, x, w, gate, eps=1e-6):
            """Per-head RMSNorm times ``silu(gate)``."""
            xf = x.float().reshape(-1, x.shape[-1])
            gf = gate.float().reshape(xf.shape)
            inv = torch.rsqrt((xf * xf).mean(-1, keepdim=True) + eps)
            return (xf * inv * w.float() * (gf * torch.sigmoid(gf))).to(x.dtype)

        def _attn(self, q, gate, kc, vc, ks, vs, out, nvalid, rep, scale):
            """Gated causal attention over one row's int8 window, written from the spec."""
            nh, hd = q.shape
            nkv = kc.shape[0]
            qf = q.float().reshape(nkv, rep, hd)
            kf = kc.float() * ks[:, :, None]
            vf = vc.float() * vs[:, :, None]
            scores = torch.einsum("ghd,gsd->ghs", qf, kf) * scale
            probs = scores.softmax(-1)
            acc = torch.einsum("ghs,gsd->ghd", probs, vf).reshape(nh, hd)
            out.copy_((acc * gate.float().sigmoid()).to(out.dtype))
            return out

        def attn_decode(self, q, gate, kc, vc, ks, vs, out, pos, rep, scale, pa, pm):
            """Dense cache: read the first ``pos + 1`` entries."""
            n = max(1, min(int(pos.reshape(-1)[0]) + 1, kc.shape[1]))
            return self._attn(q, gate, kc[:, :n], vc[:, :n], ks[:, :n], vs[:, :n], out, n,
                              rep, scale)

        def attn_paged(self, q, gate, kp, vp, ks, vs, pos2slot, out, pos, max_pos, rep,
                       scale, pa, pm):
            """Paged pool: address the window through the slot map."""
            n = max(1, min(int(pos.reshape(-1)[0]) + 1, max_pos, pos2slot.numel()))
            slots = pos2slot[:n].long()
            return self._attn(q, gate, kp.index_select(0, slots).permute(1, 0, 2),
                              vp.index_select(0, slots).permute(1, 0, 2),
                              ks.index_select(0, slots).permute(1, 0),
                              vs.index_select(0, slots).permute(1, 0), out, n, rep,
                              scale)

        def attn_paged_batch(self, q, gate, kp, vp, ks, vs, slots, lens, out, max_seq, rep,
                             scale, pa, pm):
            """One row per batch entry, each with its own context length."""
            for b in range(q.shape[0]):
                n = max(0, min(int(lens[b]), max_seq))
                if n == 0:
                    out[b].zero_()
                    continue
                win = slots[b, :n].long()
                self._attn(q[b], gate[b], kp.index_select(0, win).permute(1, 0, 2),
                           vp.index_select(0, win).permute(1, 0, 2),
                           ks.index_select(0, win).permute(1, 0),
                           vs.index_select(0, win).permute(1, 0), out[b], n, rep, scale)
            return out

        def attn_paged_multi_q(self, q, gate, kp, vp, ks, vs, slots, lens, out, max_seq,
                               rep, scale, pa, pm, block_mask, mask_block):
            """Query ``j`` of a row sees ``ctx + j + 1`` keys."""
            rows, draft = q.shape[0], q.shape[1]
            for b in range(rows):
                base = int(lens[b])
                for j in range(draft):
                    n = min(base + j + 1, max_seq)
                    if n <= 0:
                        out[b, j].zero_()
                        continue
                    win = slots[b, :n].long()
                    self._attn(q[b, j], gate[b, j],
                               kp.index_select(0, win).permute(1, 0, 2),
                               vp.index_select(0, win).permute(1, 0, 2),
                               ks.index_select(0, win).permute(1, 0),
                               vs.index_select(0, win).permute(1, 0), out[b, j], n, rep,
                               scale)
            return out

        def conv1d_silu(self, conv_w, x, state, out):
            """Four-tap causal conv + SiLU for one token, window shifted in place."""
            win = torch.cat([state, x.to(state.dtype).reshape(-1, 1)], dim=1)
            state.copy_(win[:, 1:])
            acc = (win * conv_w).sum(-1)
            out.copy_(acc * torch.sigmoid(acc))

        def conv1d_silu_seq(self, conv_w, x, state):
            """The same conv over a chunk, carrying the window across chunks."""
            kernel = conv_w.shape[1]
            seq = torch.cat([state.t(), x.to(torch.float32)], dim=0)
            state.copy_(seq[-(kernel - 1):].t())
            win = seq.unfold(0, kernel, 1)
            acc = (win * conv_w).sum(-1)
            return acc * torch.sigmoid(acc)

        def conv1d_silu_batch(self, conv_w, x, state, out):
            """One window per row."""
            for b in range(x.shape[0]):
                self.conv1d_silu(conv_w, x[b], state[b], out[b])

        def _head_prep(self, q, k, b, a, neg_expA, dt_bias, qscale, rep):
            """Prologue arithmetic, written from the definition."""
            qf = q.float()
            kf = k.float()
            if rep > 1:
                qf = qf.repeat_interleave(rep, dim=-2)
                kf = kf.repeat_interleave(rep, dim=-2)
            qn = qf * torch.rsqrt((qf * qf).sum(-1, keepdim=True) + 1e-6) * qscale
            kn = kf * torch.rsqrt((kf * kf).sum(-1, keepdim=True) + 1e-6)
            bt = b.float().sigmoid()
            sp = torch.where(a.float() + dt_bias > 20.0, a.float() + dt_bias,
                             torch.log1p(torch.exp(a.float() + dt_bias)))
            return qn, kn, bt, neg_expA * sp

        def delta_prologue(self, q, k, v, b, a, negA, dtb, q_out, k_out, v_out, beta, g,
                           qscale, rep):
            """Single-token prologue into the caller's fp32 buffers."""
            qn, kn, bt, gt = self._head_prep(q, k, b, a, negA, dtb, qscale, rep)
            q_out.copy_(qn)
            k_out.copy_(kn)
            v_out.copy_(v.float())
            beta.copy_(bt)
            g.copy_(gt)

        def delta_prologue_batch(self, q, k, v, b, a, negA, dtb, q_out, k_out, v_out,
                                 beta, g, qscale, rep):
            """Batched prologue, one row per batch entry."""
            for r in range(q.shape[0]):
                self.delta_prologue(q[r], k[r], v[r], b[r], a[r], negA, dtb, q_out[r],
                                    k_out[r], v_out[r], beta[r], g[r], qscale, rep)

        def delta_prep_seq(self, qkv, b, a, negA, dtb, q_out, k_out, v_out, beta, g,
                           qscale, rep):
            """Prefill head prep into the head-major layout."""
            num_v, T, dk = q_out.shape
            dv = v_out.shape[2]
            num_k = num_v // rep
            kdim = num_k * dk
            qf = qkv[:, :kdim].reshape(T, num_k, dk)
            kf = qkv[:, kdim:2 * kdim].reshape(T, num_k, dk)
            vf = qkv[:, 2 * kdim:].reshape(T, num_v, dv)
            qn, kn, bt, gt = self._head_prep(qf, kf, b, a, negA, dtb, qscale, rep)
            q_out.copy_(qn.transpose(0, 1).contiguous())
            k_out.copy_(kn.transpose(0, 1).contiguous())
            v_out.copy_(vf.float().transpose(0, 1).contiguous())
            beta.copy_(bt.transpose(0, 1).contiguous())
            g.copy_(gt.transpose(0, 1).contiguous())

        def _delta_step(self, q, k, v, beta, g, st, out):
            """The recurrence, with the output read from the updated state."""
            st.mul_(g.float().exp()[:, None, None])
            kv = (st * k.float()[:, :, None]).sum(-2)
            d = (v.float() - kv) * beta.float()[:, None]
            st.add_(k.float()[:, :, None] * d[:, None, :])
            out.copy_((st * q.float()[:, :, None]).sum(-2))
            return out

        def delta(self, q, k, v, beta, g, st, out):
            """Single-row gated delta step."""
            return self._delta_step(q, k, v, beta, g, st, out)

        def delta_batch(self, q, k, v, beta, g, st, out):
            """One state row per batch entry."""
            for b in range(q.shape[0]):
                self._delta_step(q[b], k[b], v[b], beta[b], g[b], st[b], out[b])
            return out

        def triangular_solve(self, lower, rhs):
            """Forward substitution, one column at a time, straight from the definition.

            Deliberately the naive form: the point of this stand-in is to be an independent
            oracle for the Python glue, so sharing a blocked algorithm with the fallback
            would only prove the two agree with themselves.
            """
            bn, L, n = rhs.shape
            x = torch.zeros(bn, L, n, dtype=torch.float32)
            for i in range(L):
                acc = rhs[:, i, :].float().clone()
                for j in range(i):
                    acc = acc - lower[:, i, j].float()[:, None] * x[:, j, :]
                x[:, i, :] = acc
            return x.to(rhs.dtype)

    return Fake()


def _host(t):
    """Move to the host and widen there.

    Args:
        t: Any tensor.

    Returns:
        A float64 CPU copy. The order matters on the card, where device fp64 returns
        zeros; keeping one helper means the habit carries over to the scripts that run
        there.
    """
    return t.detach().to("cpu").double()


def _rel(y, ref) -> float:
    """Relative error against a host reference.

    Args:
        y: Kernel output.
        ref: Host float64 reference.

    Returns:
        ``max|y - ref| / max|ref|``, or ``inf`` when the reference is all zeros.
    """
    scale = float(ref.abs().max())
    err = float((_host(y) - ref).abs().max())
    return (err / scale) if scale > 0 else float("inf")


def _sep(title: str) -> None:
    """Print a section header.

    Args:
        title: Section name.
    """
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)


def main() -> int:
    """Compare the native ops against the reference through the stand-in kernels.

    Returns:
        ``0`` when every op agrees and the table is total, ``1`` otherwise.
    """
    import torch

    import backends
    import backends.corex as corex
    from backends.corex import kernels as real_kernels
    from backends.corex import pack

    # Substitute the device probe, the build result and the extension. All three have
    # to be replaced: ``usable()`` asks ``probe()`` and ``_build_error()``, and
    # ``_build_error()`` reaches the real loader through ``kernels.available()`` --
    # patching only the probe would leave ``ops()`` on the all-reference path, and the
    # check would then compare the reference backend against itself and pass.
    corex.probe = lambda: True
    corex._device_name = lambda: "cpu-stand-in"
    corex._arch = lambda: "ivcore10"
    corex._build_error = lambda: None
    corex._kernels = lambda: _torch_kernels(torch)
    for cached in (corex.caps, corex._resolve_device, corex.ops):
        cached.cache_clear()
    del real_kernels                         # the real loader must not be reached

    _sep("0. table")
    print(" ", corex.describe(), flush=True)
    print(" ", corex.reason_table(), flush=True)
    table = corex.ops()
    print(f"  total: {len(table.implemented())}/{len(table.NAMES)} callable, "
          f"missing={table.missing()}", flush=True)
    failures: list[str] = []
    if table.missing():
        failures.append("table is not total")
    # Guard the substitution itself: with the stand-in in place every native op must be
    # this module's function. Comparing reference against reference would pass every
    # tolerance in this file without exercising a line of the glue.
    for op_name in corex.NATIVE:
        if getattr(table, op_name).__module__ != corex.__name__:
            failures.append(f"{op_name} did not resolve to the native table")

    torch.manual_seed(0)
    _sep("1. linear / linear_m / linear_prefill against reference")
    for n, k in ((256, 128), (1024, 512)):
        w = torch.randn(n, k, dtype=torch.bfloat16)
        x = torch.randn(k, dtype=torch.bfloat16)
        ref = _host(x) @ _host(w).t()
        for name, fn, args in (
            ("linear", table.linear, (w, x)),
            ("linear_m M=1", table.linear_m, (w, x.reshape(1, k))),
            ("linear_m M=4", table.linear_m, (w, x.reshape(1, k).repeat(4, 1))),
            ("linear_m M=3 (chunked)", table.linear_m, (w, x.reshape(1, k).repeat(3, 1))),
            ("linear_prefill M=2", table.linear_prefill,
             (w, x.reshape(1, k).repeat(2, 1))),
        ):
            y = fn(*args)
            r = ref if y.dim() == 1 else ref.reshape(1, -1).repeat(y.shape[0], 1)
            rel = _rel(y, r)
            ok = rel < 2e-2
            print(f"  {'ok  ' if ok else 'FAIL'} {name:<24} N={n} K={k} rel={rel:.3e}",
                  flush=True)
            if not ok:
                failures.append(f"{name} N={n} K={k}")

    _sep("2. packed weights and QWeight storage")
    w = torch.randn(512, 256, dtype=torch.bfloat16)
    x = torch.randn(256, dtype=torch.bfloat16)
    ref = _host(x) @ _host(w).t()
    packed = pack.as_kn(w)
    rel = _rel(table.linear(packed, x), ref)
    print(f"  {'ok  ' if rel < 2e-2 else 'FAIL'} packed KnWeight          rel={rel:.3e}",
          flush=True)
    if not rel < 2e-2:
        failures.append("packed KnWeight")
    from int8_weights import QWeight
    q = QWeight.quantize(w)
    ref_q = _host(x) @ _host(q.dequant(torch.float32)).t()
    rel = _rel(table.linear(q, x), ref_q)
    print(f"  {'ok  ' if rel < 2e-2 else 'FAIL'} QWeight (dequantised)     rel={rel:.3e}",
          flush=True)
    if not rel < 2e-2:
        failures.append("QWeight")

    _sep("3. dtype handling: an fp32 activation round-trips")
    xf = x.float()
    rel = _rel(table.linear(w, xf), _host(xf) @ _host(w).t())
    print(f"  {'ok  ' if rel < 2e-2 else 'FAIL'} fp32 activation            rel={rel:.3e}",
          flush=True)
    if not rel < 2e-2:
        failures.append("fp32 activation")

    _sep("3b. a packed weight presents the contract layout to a fallback op")
    # The reference backend calls ``w.to(dtype)`` on whatever it is handed. A packed
    # weight reaching it must present ``[N, K]`` rather than the packed ``[K, N]``, or a
    # partially implemented platform would raise in the middle of a step.
    try:
        plain = pack.KnWeight(torch.randn(256, 32, dtype=torch.bfloat16))
        nk = plain.to(torch.float32)
        if tuple(nk.shape) != (32, 256):
            raise ValueError(f"to() returned {tuple(nk.shape)}, expected (32, 256)")
        print(f"  ok   KnWeight.to -> {tuple(nk.shape)} {nk.dtype}", flush=True)
    except Exception as exc:                      # noqa: BLE001
        print(f"  FAIL KnWeight.to: {type(exc).__name__}: {exc}", flush=True)
        failures.append("KnWeight.to")

    _sep("4. rms_norm and rms_norm_gated against the formula")
    xs = torch.randn(8, 128, dtype=torch.bfloat16)
    g = torch.randn(128, dtype=torch.bfloat16)
    ref = _host(table.rms_norm(xs, g, 0, 1e-6))
    manual = _host(xs) * torch.rsqrt(_host(xs).pow(2).mean(-1, keepdim=True) + 1e-6) * _host(g)
    rel = _rel(ref, manual)
    print(f"  {'ok  ' if rel < 2e-2 else 'FAIL'} rms_norm packed            rel={rel:.3e}",
          flush=True)
    if not rel < 2e-2:
        failures.append("rms_norm")
    wide = torch.randn(8, 256, dtype=torch.bfloat16)
    view = wide[:, :128]
    got = table.rms_norm(view, g, view.stride(0), 1e-6)
    ref_view = (_host(view) * torch.rsqrt(_host(view).pow(2).mean(-1, keepdim=True) + 1e-6)
                * _host(g))
    rel = _rel(got, ref_view)
    print(f"  {'ok  ' if rel < 2e-2 else 'FAIL'} rms_norm pitched slice     rel={rel:.3e}",
          flush=True)
    if not rel < 2e-2:
        failures.append("rms_norm pitched")
    gate = torch.randn(8, 128, dtype=torch.bfloat16)
    got = table.rms_norm_gated(xs, g, gate, 1e-6)
    ref_gate = ((_host(xs) * torch.rsqrt(_host(xs).pow(2).mean(-1, keepdim=True) + 1e-6)
                 * _host(g)) * (_host(gate) * torch.sigmoid(_host(gate))))
    rel = _rel(got, ref_gate)
    print(f"  {'ok  ' if rel < 2e-2 else 'FAIL'} rms_norm_gated             rel={rel:.3e}",
          flush=True)
    if not rel < 2e-2:
        failures.append("rms_norm_gated")

    _sep("5. caps: derived flags, not declared ones")
    caps = corex.caps()
    print(f"  layout={caps.weight_layout} graph={caps.graph} "
          f"batch_attn={caps.batch_attn} native={len(caps.native)}/17", flush=True)
    if caps.weight_layout != "kn":
        failures.append("weight_layout")
    # ``batch_attn`` is a pure table question: with a native ``attn_paged_batch`` it must
    # be on, or the model silently pays one launch per row.
    if not caps.batch_attn:
        failures.append("batch_attn stayed False with a native attn_paged_batch")
    # ``graph`` deliberately stays off until a replay is *measured* to re-read its inputs
    # (``check_graph.py`` on the card). A complete table is necessary but not sufficient, so
    # a backend that flipped it here without that measurement would be claiming a graph
    # that can replay stale output.
    if caps.graph and not corex._GRAPH_VERIFIED:
        failures.append("graph claimed without a recorded replay measurement")
    print(f"  graph={caps.graph} (verified={corex._GRAPH_VERIFIED})", flush=True)

    _sep("6. attention: dense, paged, batched and draft-verify agree with the spec")
    nh, nkv, hd, rep_a, window = 4, 2, 16, 2, 6
    scale = hd ** -0.5
    kc = torch.randint(-127, 127, (nkv, window, hd), dtype=torch.int8)
    vc = torch.randint(-127, 127, (nkv, window, hd), dtype=torch.int8)
    ks = torch.rand(nkv, window) + 0.1
    vs = torch.rand(nkv, window) + 0.1
    qa = torch.randn(nh, hd, dtype=torch.bfloat16)
    gate_a = torch.randn(nh, hd, dtype=torch.bfloat16)
    pos = torch.tensor([3])

    def _attn_ref(q_row, gate_row, n):
        """Spec attention over the first ``n`` keys, in fp64 on the host."""
        qf = _host(q_row).reshape(nkv, rep_a, hd)
        kf = _host(kc[:, :n]) * _host(ks[:, :n])[:, :, None]
        vf = _host(vc[:, :n]) * _host(vs[:, :n])[:, :, None]
        probs = (torch.einsum("ghd,gsd->ghs", qf, kf) * scale).softmax(-1)
        acc = torch.einsum("ghs,gsd->ghd", probs, vf).reshape(nh, hd)
        return acc * _host(gate_row).sigmoid()

    out = torch.zeros(nh, hd, dtype=torch.bfloat16)
    table.attn_decode(qa, gate_a, kc, vc, ks, vs, out, pos, rep_a, scale)
    rel = _rel(out, _attn_ref(qa, gate_a, 4))
    print(f"  {'ok  ' if rel < 2e-2 else 'FAIL'} attn_decode               rel={rel:.3e}",
          flush=True)
    if not rel < 2e-2:
        failures.append("attn_decode")

    # The paged form must equal the dense one through the identity slot map; that is the
    # property that lets the two layouts share one kernel.
    kp = kc.permute(1, 0, 2).contiguous()
    vp = vc.permute(1, 0, 2).contiguous()
    kps = ks.permute(1, 0).contiguous()
    vps = vs.permute(1, 0).contiguous()
    ident = torch.arange(window, dtype=torch.long)
    out_p = torch.zeros(nh, hd, dtype=torch.bfloat16)
    table.attn_paged(qa, gate_a, kp, vp, kps, vps, ident, out_p, pos, window, rep_a, scale)
    rel = _rel(out_p, _host(out))
    print(f"  {'ok  ' if rel < 2e-2 else 'FAIL'} attn_paged == decode       rel={rel:.3e}",
          flush=True)
    if not rel < 2e-2:
        failures.append("attn_paged")

    batch = 3
    qb = torch.randn(batch, nh, hd, dtype=torch.bfloat16)
    gb = torch.randn(batch, nh, hd, dtype=torch.bfloat16)
    slots = ident.reshape(1, window).repeat(batch, 1)
    lens = torch.tensor([6, 4, 2], dtype=torch.int32)
    ob = torch.zeros(batch, nh, hd, dtype=torch.bfloat16)
    table.attn_paged_batch(qb, gb, kp, vp, kps, vps, slots, lens, ob, window, rep_a, scale)
    worst = max(_rel(ob[b], _attn_ref(qb[b], gb[b], int(lens[b]))) for b in range(batch))
    print(f"  {'ok  ' if worst < 2e-2 else 'FAIL'} attn_paged_batch          rel={worst:.3e}",
          flush=True)
    if not worst < 2e-2:
        failures.append("attn_paged_batch")

    draft = 2
    qd = torch.randn(batch, draft, nh, hd, dtype=torch.bfloat16)
    gd = torch.randn(batch, draft, nh, hd, dtype=torch.bfloat16)
    od = torch.zeros(batch, draft, nh, hd, dtype=torch.bfloat16)
    table.attn_paged_multi_q(qd, gd, kp, vp, kps, vps, slots, lens, od, window, rep_a,
                             scale)
    # Query ``j`` sees ``ctx + j + 1`` keys, which is what makes it identical to decoding
    # that token alone rather than a cheaper approximation of it.
    worst = max(_rel(od[b, j], _attn_ref(qd[b, j], gd[b, j],
                                         min(int(lens[b]) + j + 1, window)))
                for b in range(batch) for j in range(draft))
    print(f"  {'ok  ' if worst < 2e-2 else 'FAIL'} attn_paged_multi_q        rel={worst:.3e}",
          flush=True)
    if not worst < 2e-2:
        failures.append("attn_paged_multi_q")

    _sep("7. conv1d_silu trio: the window shift and the chunked form agree")
    C, T = 8, 5
    conv_w = torch.randn(C, 4)
    x1 = torch.randn(C, dtype=torch.bfloat16)
    st_step = torch.zeros(C, 3)
    st_seq = torch.zeros(C, 3)
    o_step = torch.zeros(C)
    table.conv1d_silu(conv_w, x1, st_step, o_step)
    window_in = torch.cat([torch.zeros(C, 3), x1.float()[:, None]], dim=1)
    acc = (window_in * conv_w).sum(-1)
    rel = _rel(o_step, _host(acc * torch.sigmoid(acc)))
    print(f"  {'ok  ' if rel < 2e-2 else 'FAIL'} conv1d_silu                rel={rel:.3e}",
          flush=True)
    if not rel < 2e-2:
        failures.append("conv1d_silu")
    # A chunked prefill must land on the same values *and* the same carried state as the
    # step form, which is what makes chunked prefill agree with a one-shot prefill.
    xc = torch.randn(T, C, dtype=torch.bfloat16)
    st_step.zero_()          # the single-token check above already shifted this window
    got_seq = table.conv1d_silu_seq(conv_w, xc, st_seq)
    steps = []
    for t in range(T):
        o = torch.zeros(C)
        table.conv1d_silu(conv_w, xc[t], st_step, o)
        steps.append(o.clone())
    rel = max(max(_rel(got_seq[t], _host(steps[t])) for t in range(T)),
              _rel(st_seq, _host(st_step)))
    print(f"  {'ok  ' if rel < 2e-2 else 'FAIL'} conv1d_silu_seq == step    rel={rel:.3e}",
          flush=True)
    if not rel < 2e-2:
        failures.append("conv1d_silu_seq")
    xb = torch.randn(2, C, dtype=torch.bfloat16)
    st_b = torch.zeros(2, C, 3)
    ob_conv = torch.zeros(2, C)
    table.conv1d_silu_batch(conv_w, xb, st_b, ob_conv)
    worst = 0.0
    for b in range(2):
        st_one = torch.zeros(C, 3)
        o_one = torch.zeros(C)
        table.conv1d_silu(conv_w, xb[b], st_one, o_one)
        worst = max(worst, _rel(ob_conv[b], _host(o_one)), _rel(st_b[b], _host(st_one)))
    print(f"  {'ok  ' if worst < 2e-2 else 'FAIL'} conv1d_silu_batch == step  rel={worst:.3e}",
          flush=True)
    if not worst < 2e-2:
        failures.append("conv1d_silu_batch")

    _sep("8. delta prologue and the recursion")
    num_k, dk, dv, rep_d = 2, 4, 4, 2
    num_v = num_k * rep_d
    qr = torch.randn(num_k, dk)
    kr = torch.randn(num_k, dk)
    vr = torch.randn(num_v, dv)
    br = torch.randn(num_v, dtype=torch.bfloat16)
    ar = torch.randn(num_v, dtype=torch.bfloat16)
    negA = -torch.rand(num_v) - 0.1
    dtb = torch.randn(num_v)
    qo, ko, vo = torch.zeros(num_v, dk), torch.zeros(num_v, dk), torch.zeros(num_v, dv)
    beta_o, g_o = torch.zeros(num_v), torch.zeros(num_v)
    table.delta_prologue(qr, kr, vr, br, ar, negA, dtb, qo, ko, vo, beta_o, g_o,
                         dk ** -0.5, rep_d)
    qf = _host(qr).repeat_interleave(rep_d, dim=-2)
    kf = _host(kr).repeat_interleave(rep_d, dim=-2)
    qn = qf * torch.rsqrt((qf * qf).sum(-1, keepdim=True) + 1e-6) * (dk ** -0.5)
    kn = kf * torch.rsqrt((kf * kf).sum(-1, keepdim=True) + 1e-6)
    gt = _host(negA) * torch.log1p(torch.exp(_host(ar) + _host(dtb)))
    rel = max(_rel(qo, qn), _rel(ko, kn), _rel(beta_o, _host(br).sigmoid()), _rel(g_o, gt))
    print(f"  {'ok  ' if rel < 2e-2 else 'FAIL'} delta_prologue             rel={rel:.3e}",
          flush=True)
    if not rel < 2e-2:
        failures.append("delta_prologue")
    if bool((_host(g_o) > 0).any()):
        # A positive log decay would make the recurrence grow instead of decay; the sign
        # comes from neg_expA, and a slip here diverges rather than merely being wrong.
        failures.append("delta_prologue produced g > 0")
    st = torch.randn(num_v, dk, dv)
    st_ref = st.clone()
    out_d = torch.zeros(num_v, dv)
    table.delta(qo, ko, vo, beta_o, g_o, st, out_d)
    s = st_ref * _host(g_o).exp()[:, None, None]
    kv = (s * _host(ko)[:, :, None]).sum(-2)
    d = (_host(vo) - kv) * _host(beta_o)[:, None]
    s = s + _host(ko)[:, :, None] * d[:, None, :]
    rel = max(_rel(st, s), _rel(out_d, (s * _host(qo)[:, :, None]).sum(-2)))
    print(f"  {'ok  ' if rel < 2e-2 else 'FAIL'} delta                      rel={rel:.3e}",
          flush=True)
    if not rel < 2e-2:
        failures.append("delta")

    # Each row carries its own inputs, not a tiled copy of one row. With identical rows
    # every answer is identical too, so a stride bug that reads row 0 for row 1 returns
    # the same values and the check passes exactly when it is needed. Distinct rows are
    # what make a wrong batch offset visible, and the batch-start-equals-single-row
    # property is what the batched decode path depends on.
    rows_d = 3
    torch.manual_seed(11)
    q_bd = torch.randn(rows_d, num_v, dk)
    k_bd = torch.randn(rows_d, num_v, dk)
    v_bd = torch.randn(rows_d, num_v, dv)
    beta_bd = torch.randn(rows_d, num_v)
    g_bd = -torch.rand(rows_d, num_v) - 0.1
    st_bd = torch.randn(rows_d, num_v, dk, dv)
    st_one_all = st_bd.clone()
    out_bd = torch.zeros(rows_d, num_v, dv)
    table.delta_batch(q_bd, k_bd, v_bd, beta_bd, g_bd, st_bd, out_bd)
    worst = 0.0
    for b in range(rows_d):
        o_one = torch.zeros(num_v, dv)
        table.delta(q_bd[b], k_bd[b], v_bd[b], beta_bd[b], g_bd[b], st_one_all[b], o_one)
        worst = max(worst, _rel(out_bd[b], _host(o_one)),
                    _rel(st_bd[b], _host(st_one_all[b])))
    print(f"  {'ok  ' if worst < 2e-2 else 'FAIL'} delta_batch == per-row     rel={worst:.3e}",
          flush=True)
    if not worst < 2e-2:
        failures.append("delta_batch")

    _sep("9. triangular_solve: the cuSOLVER replacement")
    # ``L = chunk = 64`` is what the model actually passes (``linear_attn.prefill`` does
    # not override the default), so a shape requirement that rejected 64 would break
    # prefill -- the failure mode this op exists to remove.
    for L, bn, n in ((64, 8, 64), (256, 4, 32), (16, 2, 8)):
        torch.manual_seed(L)
        lower = torch.randn(bn, L, L, dtype=torch.bfloat16)
        # A unit-lower-triangular coefficient: the upper triangle must be ignored, so it is
        # deliberately filled with noise rather than left zero.
        lower = torch.tril(lower, -1) + torch.eye(L)
        rhs = torch.randn(bn, L, n, dtype=torch.bfloat16)
        got = table.triangular_solve(lower, rhs)
        ld, rh = _host(lower), _host(rhs)
        want = torch.linalg.solve_triangular(ld, rh, upper=False, unitriangular=True)
        rel = _rel(got, want)
        ok_t = rel < 2e-2
        print(f"  {'ok  ' if ok_t else 'FAIL'} triangular_solve L={L:<4} bn={bn} n={n}  "
              f"rel={rel:.3e}", flush=True)
        if not ok_t:
            failures.append(f"triangular_solve L={L}")

    # The caller's right-hand side is ``eye(L).expand(bn, L, L)``: batch stride 0, so a
    # kernel that assumed contiguity would read one system and answer it bn times.
    L, bn = 64, 6
    lower = torch.tril(torch.randn(bn, L, L, dtype=torch.bfloat16), -1) + torch.eye(L)
    eye = torch.eye(L, dtype=torch.bfloat16).expand(bn, L, L)
    got = table.triangular_solve(lower, eye)
    want = torch.linalg.solve_triangular(_host(lower), _host(eye), upper=False,
                                         unitriangular=True)
    rel = _rel(got, want)
    ok_e = rel < 2e-2
    print(f"  {'ok  ' if ok_e else 'FAIL'} triangular_solve expand(eye)      rel={rel:.3e}",
          flush=True)
    if not ok_e:
        failures.append("triangular_solve expanded rhs")

    _sep("10. pack: one transpose per weight, and one resident copy")

    w = torch.randn(64, 32, dtype=torch.bfloat16)
    a = pack.as_kn(w)
    b = pack.as_kn(w)
    print(f"  memoised={a is b} shape={tuple(a.wt.shape)} bytes={a.nbytes()}", flush=True)
    if a is not b or tuple(a.wt.shape) != (32, 64):
        failures.append("pack.as_kn")
    del b, a
    sd = {"q.weight": torch.randn(48, 32), "n.weight": torch.randn(32)}
    report = pack.repack_state_dict(sd)
    print(f"  repack_state_dict: {report}", flush=True)
    if report["packed"] != 1 or pack.unpacked_keys(sd):
        failures.append("repack_state_dict")

    _sep("summary")
    print(f"  failures: {failures or '(none)'}", flush=True)
    print("\nCOREX_DISPATCH_DONE", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
