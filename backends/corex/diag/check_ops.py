"""Conformance and bandwidth check of every CoreX kernel against an independent host oracle.

This is the acceptance gate for ``backends/corex`` (COREX_PORT.md §3): each op is
compared against arithmetic written out from the definition -- the specification, not
another kernel -- and its achieved bandwidth is printed next to it. It runs on the
card, so it is submitted through ``platform/scow/tools/deploy.py``, never run on a
dev box.

Module: ``coffin/backends/corex/diag``. Diagnostic only: synthetic tensors, no
checkpoint, no engine import.

Constraints:
    * The oracle is host-side arithmetic, not ``backends.reference``. Two of that
      backend's ops cannot run on this cluster at all (``F.rms_norm`` does not exist in
      torch 2.1) and a third would be measured against the same library GEMM the
      backend avoids, so it is not an independent reference here. The formulas below
      are the spec restated.
    * Every value is moved to the host *before* any dtype change. This card's fp64 is
      "limited support" and returns zeros, so ``y.double()`` on the device would compare
      the kernel against nothing and pass anything (COREX_PORT.md §0).
    * References are fp32, not fp64. ``lm_head`` is 248320 x 2560, which is 2.5 GB as
      host fp32 and 5 GB as fp64; fp32 accumulation over this model's K is ~1e-7
      relative, three orders below the tolerance being checked, so the wider type buys
      nothing but the risk of a host-side OOM on a shared node.
    * One shape at a time. Holding every case's device weight and its host reference at
      once would be ~9 GB of host RAM and ~2.7 GB of device memory for no reason; each
      case frees its tensors before the next is built.
    * No library GEMM in any reference path, for the same reason the backend has none:
      the card's ``cuBLAS`` returns NaN above ``N = 131072``.
    * The tolerance is applied to the magnitude of the reference rather than per
      element: a per-element check on a dot product with cancellation reports a large
      relative error on a value that is numerically noise.
    * Exit status is the gate. A non-zero return means at least one op is out of
      tolerance, so the job's state alone answers "did this pass".
"""

from __future__ import annotations

import sys
from pathlib import Path

#: ``coffin/`` itself is the ``sys.path`` root: the engine imports ``backends``,
#: ``ops`` and ``models`` as top-level packages. Running this file by path puts its own
#: directory on ``sys.path`` instead, so the root is added explicitly. Three levels up
#: from ``backends/corex/diag/check_ops.py``.
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

#: Relative error a bf16 kernel is allowed against the fp32 reference. bf16 carries
#: 8 mantissa bits, so ~4e-3 is the rounding floor of a single operation and 2e-2
#: leaves room for a longer accumulation chain without hiding a real bug.
TOL = 2e-2

#: Columns of a wide weight the host reference covers. ``lm_head`` is 248320 x 2560,
#: which is 2.5 GB as host fp32; the kernel has no column-dependent branch, so a strip
#: is an honest oracle for the columns it covers and the comparison is windowed to
#: exactly those columns.
REF_COLS = 32768

#: Model-real projection shapes, ``(label, N, K)``, from Qwen3.5-4B: mlp gate/up, mlp
#: down, fused qkv+gate, o_proj, and lm_head -- the wide one the library cannot do.
#: Model shapes rather than random ones because bandwidth on this card depends on
#: ``N`` (COREX_PORT.md §0), so a small shape would measure the wrong thing.
SHAPES: tuple[tuple[str, int, int], ...] = (
    ("mlp_up", 9728, 2560),
    ("mlp_down", 2560, 9728),
    ("qkv_gate", 8192, 2560),
    ("o_proj", 2560, 4096),
    ("lm_head", 248320, 2560),
)


def _host(t):
    """Move a tensor to the host and widen it there.

    Args:
        t: Any tensor, on any device.

    Returns:
        A float32 CPU copy. The order is deliberate: ``.double()`` on this device runs
        in the broken fp64 path and returns zeros, and the transfer has to happen
        first either way.
    """
    return t.detach().to("cpu").float()


def _window(y, ref, cols: int | None):
    """Slice a kernel output to the column window a reference covers.

    ``lm_head`` is 248320 columns wide, and referencing all of it would be a 2.5 GB
    host copy, so the reference covers a prefix and the comparison has to cover the
    same prefix. Doing the slicing here -- explicitly, for 1-D and 2-D outputs alike --
    is what keeps the two operands the same shape; the earlier version sliced only
    2-D outputs and therefore subtracted a full-width vector from a prefix reference.

    Args:
        y: Kernel output, ``[N]`` or ``[M, N]``.
        ref: Host reference covering ``ref.shape[-1]`` columns.
        cols: Column count to compare, or ``None`` for the whole of ``ref``.

    Returns:
        ``(got, ref)`` with matching shapes, ``got`` on the host.
    """
    got = _host(y)
    width = ref.shape[-1] if cols is None else min(cols, ref.shape[-1])
    if got.shape[-1] != width:
        got = got[..., :width]
    return got, ref[..., :width]


def _rel(y, ref, cols: int | None = None) -> float:
    """Relative error of a kernel output against a host reference.

    Args:
        y: Kernel output, on any device.
        ref: Host reference.
        cols: Column window to compare; ``None`` compares the whole reference.

    Returns:
        ``max|y - ref| / max|ref|`` over the window, or ``inf`` when the reference is
        all zeros there -- which is itself a finding (a zero reference means the
        *reference* path broke, not that the kernel is exact).
    """
    got, ref = _window(y, ref, cols)
    scale = float(ref.abs().max())
    err = float((got - ref).abs().max())
    return (err / scale) if scale > 0 else float("inf")


def _sep(title: str) -> None:
    """Print a section header.

    Args:
        title: Section name.
    """
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)


def _bench(fn, iters: int = 30, warmup: int = 8) -> float:
    """Time a callable on the device, in milliseconds per call.

    Args:
        fn: Zero-argument callable to time.
        iters: Timed repetitions.
        warmup: Untimed repetitions, to get past first-call allocations.

    Returns:
        Milliseconds per call. Event timing works on this card (the reference engine's
        own probe uses it), so it is the measurement rather than a host clock around
        an asynchronous launch.
    """
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _check(label: str, y, ref, nbytes: int, ms: float | None,
           cols: int | None = None) -> bool:
    """Report one op's relative error and bandwidth, and say whether it passed.

    Args:
        label: Op name and shape, for the log line.
        y: Kernel output.
        ref: Host reference, which may cover only a column prefix of ``y``.
        nbytes: Weight bytes the call had to read, for the bandwidth figure.
        ms: Milliseconds per call, or ``None`` to omit the bandwidth.
        cols: Column window to compare; ``None`` compares the whole reference. Passed
            explicitly rather than inferred, so a width mismatch between the two
            operands is a shape error at the call site instead of a silent slice.

    Returns:
        ``True`` when the relative error is inside :data:`TOL`.
    """
    rel = _rel(y, ref, cols)
    gbs = f"{nbytes / (ms * 1e-3) / 1e9:7.1f} GB/s" if ms else "          n/a"
    ok = rel < TOL
    print(f"  {'ok  ' if ok else 'FAIL'} {label:<34} rel={rel:.3e} {gbs}", flush=True)
    return ok


def _host_linear(w, x, cols: int = 32768):
    """Host reference for ``x [K] @ w [N, K]^T``, computed in column blocks.

    Blocking keeps the host copy of a wide weight at one column strip rather than the
    whole 2.5 GB matrix: the arithmetic is column-independent, so a strip is a valid
    oracle for the columns it covers.

    Args:
        w: Device weight, contract layout ``[N, K]``.
        x: Device activation, ``[K]``.
        cols: How many output columns to reference. ``lm_head`` at 248320 columns would
            cost 2.5 GB of host copy and minutes of transfer for a check that is the
            same per column; the kernel has no column-dependent branch, so a strip is
            the honest and affordable oracle.

    Returns:
        ``[min(N, cols)]`` host float32 reference.
    """
    import torch

    x64 = _host(x)
    n = min(w.shape[0], cols)
    out = torch.empty(n, dtype=torch.float32)
    step = 16384
    for start in range(0, n, step):
        stop = min(n, start + step)
        block = _host(w[start:stop])
        out[start:stop] = block @ x64
        del block
    return out


def _host_formula_norm(x, w, gate=None, eps: float = 1e-6):
    """Host reference for the two norms, written from their definition.

    Not the reference backend: its ``rms_norm`` calls ``F.rms_norm``, which does not
    exist in the torch 2.1 this cluster runs, so using it as the oracle would make the
    check fail for a reason that has nothing to do with the kernel.

    Args:
        x: Device activation, ``[M, N]``; may be a pitched column slice.
        w: Device gain, ``[N]``.
        gate: Optional device gate, same shape as ``x``, for the gated form.
        eps: Variance floor.

    Returns:
        ``[M, N]`` host float32 reference.
    """
    import torch

    xf = _host(x)
    wf = _host(w)
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * wf
    if gate is not None:
        gf = _host(gate)
        out = out * (gf * torch.sigmoid(gf))
    return out


def main() -> int:
    """Run the whole conformance sweep.

    Returns:
        ``0`` when every checked op is inside :data:`TOL`, ``1`` otherwise. A
        diagnostic that always exits 0 would make the job state meaningless.
    """
    import torch

    import backends.corex as corex
    from backends.corex import pack
    from int8_weights import QWeight

    _sep("0. backend state")
    print(" ", corex.describe(), flush=True)
    print(" ", corex.reason_table(), flush=True)
    if not corex.usable():
        print("\nbackend is not usable on this host; nothing to measure", flush=True)
        return 1
    table = corex.ops()
    failures: list[str] = []

    _sep("1. linear: decode GEMV, [K, N] weights, one row")
    for label, n, k in SHAPES:
        w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
        x = torch.randn(k, dtype=torch.bfloat16, device="cuda")
        cols = REF_COLS if n > REF_COLS else None
        ref = _host_linear(w, x, cols or n)
        nbytes = w.numel() * w.element_size()
        y = table.linear(w, x)
        ms = _bench(lambda w=w, x=x: table.linear(w, x))
        if not _check(f"linear {label} N={n} K={k}", y, ref, nbytes, ms, cols):
            failures.append(f"linear {label}")
        del w, x, ref, y
        torch.cuda.empty_cache()

    _sep("1b. where the bf16 error comes from: kernel order vs bf16 output rounding")
    # A relative error of ~2.3e-3 on a dot product of random bf16 operands is the
    # *output's* rounding, not the kernel's accumulation: bf16 holds 8 mantissa bits, so
    # one rounding of the result is already ~2^-9 = 1.95e-3. Four numbers separate the
    # possible causes, and only the last one is about the kernel:
    #   1. fp32 accumulate over the same bf16 inputs, vs the exact fp64 dot product
    #   2. the reference backend's own ``linear``, which is the specification
    #   3. our kernel
    #   4. bf16's own resolution
    # If (1) is ~1e-7 and (2) matches (3), there is no precision bug to chase: both
    # backends round to bf16 because the contract says the result is in the activation
    # dtype, and a model layer sees the same number whichever backend ran it.
    n, k = 9728, 2560
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(k, dtype=torch.bfloat16, device="cuda")
    cols = 4096
    w_strip = _host(w[:cols])
    x_f32 = _host(x)
    ref_exact = (w_strip.double() @ x_f32.double())
    ref_fp32 = ref_exact.float()
    floor = float((ref_fp32.double() - ref_exact).abs().max()) / float(
        ref_exact.abs().max())
    import backends
    ref_table = backends.reference.build(corex.caps())
    ref_backend = _rel(ref_table.linear(w, x), ref_fp32, cols)
    ours = _rel(table.linear(w, x), ref_fp32, cols)
    print(f"  fp32 accumulate vs exact fp64 dot:       {floor:.3e}  <- kernel order",
          flush=True)
    print(f"  reference backend linear:                {ref_backend:.3e}  <- the spec",
          flush=True)
    print(f"  corex kernel linear:                     {ours:.3e}  <- ours", flush=True)
    print(f"  bf16 output rounding alone:              {2.0 ** -9:.3e}  <- the contract",
          flush=True)
    print("  verdict: an error at the bf16 resolution is the contract, not a defect;\n"
          "           a value an order of magnitude above it would be a real bug.",
          flush=True)
    del w, x
    torch.cuda.empty_cache()

    _sep("2. linear_m: small-M batch, one weight pass")
    for m in (2, 4, 8):
        n, k = 9728, 2560
        w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
        xs = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        ref = torch.stack([_host_linear(w, xs[i], n) for i in range(m)])
        nbytes = w.numel() * w.element_size()
        y = table.linear_m(w, xs)
        ms = _bench(lambda w=w, xs=xs: table.linear_m(w, xs))
        if not _check(f"linear_m M={m} N={n} K={k}", y, ref, nbytes, ms):
            failures.append(f"linear_m M={m}")
        del w, xs, ref, y
        torch.cuda.empty_cache()

    _sep("2b. linear_m on the int8 path: one weight pass AND bit-identical rows")
    # The int8 batched entry did not exist until it was added for speculative decoding, and
    # its absence was invisible: ``_gemv_rows`` looped the single-row kernel, which is
    # correct and costs one full weight read per row. Two things have to be shown before that
    # loop may be replaced, and only the first is a tolerance question.
    #
    # 1. Every row equals the single-row entry *bit for bit*. That is the model contract
    #    (batched decode must not change a token), so it is an equality check and not a
    #    tolerance one: a batched path that is merely close silently re-samples tokens.
    # 2. The marginal cost of rows 2..8 is near zero, which is the whole point. Printed per
    #    width so a later dispatch change can see what it buys.
    for m in (2, 4, 8):
        n, k = 9728, 2560
        wq = QWeight.quantize(torch.randn(n, k, dtype=torch.bfloat16, device="cuda"))
        xs = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        got = table.linear_m(wq, xs)
        solo = torch.stack([table.linear(wq, xs[i]) for i in range(m)])
        exact = torch.equal(got, solo)
        nbytes = wq.wq.numel() + wq.scale.numel() * 4
        ms = _bench(lambda wq=wq, xs=xs: table.linear_m(wq, xs))
        one_ms = _bench(lambda wq=wq, xs=xs: table.linear(wq, xs[0]))
        gbs = (nbytes / (ms * 1e-3)) / 1e9
        row = (f"  linear_m int8 M={m} N={n} K={k}: {ms * 1e3:>8.1f} us "
               f"{gbs:>6.1f} GB/s  rows-identical={exact}  vs 1 row: {ms / one_ms:.2f}x")
        print(row, flush=True)
        if not exact:
            # How far off, not just how many: a handful of last-bit differences is the fp32
            # association changing with the split, which is a real (if small) contract break;
            # a large relative error would be a wrong kernel instead, and the two need
            # different fixes. Reported together so the number cannot be read as either.
            worst = int((got != solo).sum())
            rel = float((got.float() - solo.float()).abs().max()
                        / solo.float().abs().max().clamp_min(1e-6))
            print(f"    {worst} of {got.numel()} values differ from the single-row path "
                  f"(max rel {rel:.2e})", flush=True)
            failures.append(f"linear_m int8 M={m} not row-identical")
        del wq, xs, got, solo
        torch.cuda.empty_cache()
    _sep("3. linear_prefill: library GEMM for M >= 64, chunked above LIB_MAX_N")
    for label, m, n, k in (("mlp_up", 512, 9728, 2560), ("lm_head", 64, 248320, 2560)):
        w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
        xs = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        # A row-wise reference: the wide case cannot be transposed into one host
        # matrix without the 2.5 GB copy the column-blocked helper exists to avoid.
        ref = torch.stack([_host_linear(w, xs[i], REF_COLS) for i in range(m)])
        cols = REF_COLS if n > REF_COLS else None
        nbytes = w.numel() * w.element_size()
        y = table.linear_prefill(w, xs)
        ms = _bench(lambda w=w, xs=xs: table.linear_prefill(w, xs), iters=10, warmup=3)
        if not _check(f"linear_prefill {label} M={m} N={n}", y, ref, nbytes, ms, cols):
            failures.append(f"linear_prefill {label}")
        del w, xs, ref, y
        torch.cuda.empty_cache()

    _sep("4. rms_norm: packed rows and a pitched column slice")
    m, n = 8, 2560
    xs = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")
    g = torch.randn(n, dtype=torch.bfloat16, device="cuda")
    y = table.rms_norm(xs, g, 0, 1e-6)
    if not _check(f"rms_norm M={m} N={n}", y, _host_formula_norm(xs, g), xs.numel() * 2,
                  None):
        failures.append("rms_norm")
    wide = torch.randn(m, n * 2, dtype=torch.bfloat16, device="cuda")
    view = wide[:, :n]
    y = table.rms_norm(view, g, view.stride(0), 1e-6)
    if not _check("rms_norm pitched slice", y, _host_formula_norm(view, g),
                  view.numel() * 2, None):
        failures.append("rms_norm pitched")
    del xs, wide, g, y
    torch.cuda.empty_cache()

    _sep("5. rms_norm_gated: per-head norm times silu(gate)")
    heads, dv = 16, 128
    xs = torch.randn(heads, dv, dtype=torch.bfloat16, device="cuda")
    gate = torch.randn(heads, dv, dtype=torch.bfloat16, device="cuda")
    g = torch.randn(dv, dtype=torch.bfloat16, device="cuda")
    y = table.rms_norm_gated(xs, g, gate, 1e-6)
    if not _check("rms_norm_gated [16, 128]", y,
                  _host_formula_norm(xs, g, gate), xs.numel() * 2, None):
        failures.append("rms_norm_gated")
    del xs, gate, g, y
    torch.cuda.empty_cache()

    _sep("6. pack: the loader transpose and its resident cost")
    w = torch.randn(2560, 9728, dtype=torch.bfloat16, device="cuda")
    before = torch.cuda.memory_allocated()
    packed = pack.to_kn(w)
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated()
    grew = after - before
    print(f"  to_kn [2560, 9728] -> {tuple(packed.wt.shape)} resident=+{grew} bytes "
          f"(expect {w.numel() * 2})", flush=True)
    # ``resident`` is compared against the allocator's rounding, not against the exact
    # byte count. A block above 1 MiB is carved from a 2 MiB segment on this build, so
    # 49 807 360 bytes is reported as 50 331 648 -- 24 segments, i.e. exactly 48 MiB. An
    # equality check here failed on a correct transpose and reported it as a backend bug,
    # which cost a round trip; the shape and the logical size are the real invariant.
    seg = 2 * 1024 * 1024
    rounded = (w.numel() * 2 + seg - 1) // seg * seg
    if tuple(packed.wt.shape) != (9728, 2560) or grew not in (w.numel() * 2, rounded):
        failures.append("pack.to_kn")

    _sep("7. attention: dense and paged agree, int8 window in, gated bf16 out")
    nh, nkv, hd, rep_a, window = 16, 4, 256, 4, 64
    scale = hd ** -0.5
    kc = torch.randint(-127, 127, (nkv, window, hd), dtype=torch.int8, device="cuda")
    vc = torch.randint(-127, 127, (nkv, window, hd), dtype=torch.int8, device="cuda")
    ks = torch.rand(nkv, window, device="cuda") + 0.1
    vs = torch.rand(nkv, window, device="cuda") + 0.1
    qa = torch.randn(nh, hd, dtype=torch.bfloat16, device="cuda")
    gate = torch.randn(nh, hd, dtype=torch.bfloat16, device="cuda")
    # ``pos`` is read *by the kernel*, not by the host: the attention entry points take its
    # address and dereference it on the device so a captured graph can be replayed at a new
    # position without a re-capture. A CPU tensor here therefore hands the device a host
    # pointer, and what it reads is that address interpreted as a length. The failure is
    # silent and total -- an out-of-range length leaves the key window empty, every split
    # keeps ``m = -1e30`` and ``l = 0``, and the combine pass divides ``0 / 0`` -- which
    # surfaces as an all-zero output and a relative error of exactly 1, not as an error.
    pos = torch.tensor([window - 1], device="cuda")
    assert pos.is_cuda, "pos is read on the device; a host tensor here reads garbage"

    def _attn_ref(q_row, gate_row, n):
        """Spec attention over the first ``n`` keys, in fp64 on the host."""
        qf = _host(q_row).reshape(nkv, rep_a, hd)
        kf = _host(kc[:, :n]) * _host(ks[:, :n])[:, :, None]
        vf = _host(vc[:, :n]) * _host(vs[:, :n])[:, :, None]
        probs = (torch.einsum("ghd,gsd->ghs", qf, kf) * scale).softmax(-1)
        acc = torch.einsum("ghs,gsd->ghd", probs, vf).reshape(nh, hd)
        return acc * _host(gate_row).sigmoid()

    out = torch.zeros(nh, hd, dtype=torch.bfloat16, device="cuda")
    y = table.attn_decode(qa, gate, kc, vc, ks, vs, out, pos, rep_a, scale)
    kv_bytes = 2 * window * nkv * hd
    ms = _bench(lambda: table.attn_decode(qa, gate, kc, vc, ks, vs, out, pos, rep_a,
                                          scale))
    if not _check(f"attn_decode window={window}", y, _attn_ref(qa, gate, window),
                  kv_bytes, ms):
        failures.append("attn_decode")

    # The paged form must equal the dense one through the identity slot map: that is what
    # lets the two layouts share one kernel, so a drift here is a bug in one of them.
    kp = kc.permute(1, 0, 2).contiguous()
    vp = vc.permute(1, 0, 2).contiguous()
    kps = ks.permute(1, 0).contiguous()
    vps = vs.permute(1, 0).contiguous()
    ident = torch.arange(window, dtype=torch.long, device="cuda")
    out_p = torch.zeros(nh, hd, dtype=torch.bfloat16, device="cuda")
    y = table.attn_paged(qa, gate, kp, vp, kps, vps, ident, out_p, pos, window, rep_a,
                         scale)
    if not _check("attn_paged == decode", y, _host(out), kv_bytes, None):
        failures.append("attn_paged")

    batch = 4
    qb = torch.randn(batch, nh, hd, dtype=torch.bfloat16, device="cuda")
    gb = torch.randn(batch, nh, hd, dtype=torch.bfloat16, device="cuda")
    slots = ident.reshape(1, window).repeat(batch, 1)
    lens = torch.full((batch,), window, dtype=torch.int32, device="cuda")
    ob = torch.zeros(batch, nh, hd, dtype=torch.bfloat16, device="cuda")
    y = table.attn_paged_batch(qb, gb, kp, vp, kps, vps, slots, lens, ob, window, rep_a,
                               scale)
    ref_batch = torch.stack([_attn_ref(qb[b], gb[b], window) for b in range(batch)])
    ms = _bench(lambda: table.attn_paged_batch(qb, gb, kp, vp, kps, vps, slots, lens, ob,
                                               window, rep_a, scale))
    if not _check(f"attn_paged_batch B={batch}", y, ref_batch, batch * kv_bytes, ms):
        failures.append("attn_paged_batch")
    del kc, vc, ks, vs, kp, vp, kps, vps, qa, gate, qb, gb, ob, out, out_p
    torch.cuda.empty_cache()

    _sep("8. conv1d_silu trio and the delta prologue/recursion")
    C, T = 8192, 64
    conv_w = torch.randn(C, 4, device="cuda")
    x1 = torch.randn(C, dtype=torch.bfloat16, device="cuda")
    state = torch.zeros(C, 3, device="cuda")
    o_conv = torch.zeros(C, device="cuda")
    y = table.conv1d_silu(conv_w, x1, state, o_conv)
    # ``y`` aliases the caller-owned ``out`` buffer, so cloning is what makes the
    # comparison below refer to *this* call: ``_bench`` shifts ``conv_state`` 38 more
    # times and overwrites ``o_conv`` in the process.
    got_conv = y.clone()
    state_after = state.clone()
    win = torch.cat([torch.zeros(C, 3, device="cuda"), x1.float()[:, None]], dim=1)
    acc = (win * conv_w).sum(-1)
    # The state must shift by exactly one token, or the next decode step sees the wrong
    # history; that is the part of the contract a value-only check would miss.
    ref_state = torch.cat([torch.zeros(C, 2, device="cuda"), x1.float()[:, None]], dim=1)
    nbytes = C * (4 * 4 + 2 + 3 * 4 + 4)
    rel_conv = max(_rel(got_conv, _host(acc * torch.sigmoid(acc))),
                   _rel(state_after, _host(ref_state)))
    ms = _bench(lambda: table.conv1d_silu(conv_w, x1, state, o_conv))
    ok = rel_conv < TOL
    print(f"  {'ok  ' if ok else 'FAIL'} conv1d_silu C={C}            "
          f"rel={rel_conv:.3e} {nbytes / (ms * 1e-3) / 1e9:7.1f} GB/s", flush=True)
    if not ok:
        failures.append("conv1d_silu")

    # Chunked prefill must agree with the step form on the values *and* on the carried
    # window, or a chunked prefill would diverge from a one-shot one.
    xc = torch.randn(T, C, dtype=torch.bfloat16, device="cuda")
    st_seq = torch.zeros(C, 3, device="cuda")
    st_step = torch.zeros(C, 3, device="cuda")
    seq_out = table.conv1d_silu_seq(conv_w, xc, st_seq)
    steps = []
    for t in range(min(T, 8)):
        o = torch.zeros(C, device="cuda")
        table.conv1d_silu(conv_w, xc[t], st_step, o)
        steps.append(o)
    # Only the first 8 steps are replayed one at a time: recomputing all 64 through the
    # step kernel would dominate the run for a comparison that is already made pointwise.
    rel_seq = max(_rel(seq_out[t], _host(steps[t])) for t in range(len(steps)))
    ms = _bench(lambda: table.conv1d_silu_seq(conv_w, xc, st_seq))
    ok = rel_seq < TOL
    print(f"  {'ok  ' if ok else 'FAIL'} conv1d_silu_seq T={T} C={C}        "
          f"rel={rel_seq:.3e} {T * C * (2 + 4) / (ms * 1e-3) / 1e9:7.1f} GB/s", flush=True)
    if not ok:
        failures.append("conv1d_silu_seq")

    num_k, dk, dv = 16, 128, 128
    rep_d = 2
    num_v = num_k * rep_d
    qr = torch.randn(num_k, dk, device="cuda")
    kr = torch.randn(num_k, dk, device="cuda")
    vr = torch.randn(num_v, dv, device="cuda")
    br = torch.randn(num_v, dtype=torch.bfloat16, device="cuda")
    ar = torch.randn(num_v, dtype=torch.bfloat16, device="cuda")
    negA = -torch.rand(num_v, device="cuda") - 0.1
    dtb = torch.randn(num_v, device="cuda")
    qo = torch.zeros(num_v, dk, device="cuda")
    ko = torch.zeros(num_v, dk, device="cuda")
    vo = torch.zeros(num_v, dv, device="cuda")
    beta_o = torch.zeros(num_v, device="cuda")
    g_o = torch.zeros(num_v, device="cuda")
    table.delta_prologue(qr, kr, vr, br, ar, negA, dtb, qo, ko, vo, beta_o, g_o,
                         dk ** -0.5, rep_d)
    qf = _host(qr).repeat_interleave(rep_d, dim=-2)
    kf = _host(kr).repeat_interleave(rep_d, dim=-2)
    ref_q = qf * torch.rsqrt((qf * qf).sum(-1, keepdim=True) + 1e-6) * (dk ** -0.5)
    ref_g = _host(negA) * torch.log1p(torch.exp(_host(ar) + _host(dtb)))
    rel_p = max(_rel(qo, ref_q),
                _rel(ko, kf * torch.rsqrt((kf * kf).sum(-1, keepdim=True) + 1e-6)),
                _rel(beta_o, _host(br).sigmoid()), _rel(g_o, ref_g))
    ms = _bench(lambda: table.delta_prologue(qr, kr, vr, br, ar, negA, dtb, qo, ko, vo,
                                             beta_o, g_o, dk ** -0.5, rep_d))
    ok = rel_p < TOL
    print(f"  {'ok  ' if ok else 'FAIL'} delta_prologue heads={num_v}        "
          f"rel={rel_p:.3e} {ms * 1e3:7.1f} us", flush=True)
    if not ok:
        failures.append("delta_prologue")
    if bool((_host(g_o) > 0).any()):
        # ``g`` must be a decay. A sign slip makes the recurrence grow, which diverges
        # instead of merely being inaccurate, so it is checked separately.
        failures.append("delta_prologue produced g > 0")

    st = torch.randn(num_v, dk, dv, device="cuda")
    st_ref = st.clone()
    out_d = torch.zeros(num_v, dv, device="cuda")
    y = table.delta(qo, ko, vo, beta_o, g_o, st, out_d)
    # The reference recurrence runs entirely on the host: this card's device fp64 returns
    # zeros, and mixing a device tensor into a host expression is a device error rather
    # than a numerical one.
    s = _host(st_ref) * _host(g_o).exp()[:, None, None]
    kv = (s * _host(ko)[:, :, None]).sum(-2)
    d = (_host(vo) - kv) * _host(beta_o)[:, None]
    s = s + _host(ko)[:, :, None] * d[:, None, :]
    ref_d = (s * _host(qo)[:, :, None]).sum(-2)
    # ``y`` aliases ``out_d`` and the step mutates ``st``, so both have to be captured
    # before the timing loop runs the recurrence another 38 times.
    got_d = y.clone()
    got_state = st.clone()
    state_bytes = num_v * dk * dv * 4 * 3
    ms = _bench(lambda: table.delta(qo, ko, vo, beta_o, g_o, st_ref, out_d))
    rel_d = max(_rel(got_d, ref_d), _rel(got_state, s))
    ok_d = rel_d < TOL
    print(f"  {'ok  ' if ok_d else 'FAIL'} delta heads={num_v} dk={dk}              "
          f"rel={rel_d:.3e} {state_bytes / (ms * 1e-3) / 1e9:7.1f} GB/s", flush=True)
    if not ok_d:
        failures.append("delta")

    # Row ``b`` of the batched step must be identical to the single-row step on that row,
    # which is the property the model's batch bucket relies on.
    #
    # Two things had to be built differently from the obvious version, and both were the
    # reason this check reported a 2.4e-01 failure against a correct kernel:
    #
    #   * Each row gets its **own** inputs. Tiling one row across the batch makes every
    #     row's answer equal, so a stride bug that reads row 0 for row 1 returns the same
    #     values -- the check would pass precisely when it is needed. Distinct rows are
    #     what turn a wrong stride into a visible difference.
    #   * The batch starts from a **fresh** state. ``st`` was advanced one step by the
    #     single-row call above, and ``got_state`` is that result; feeding it to
    #     ``delta_batch`` advances it again, so the comparison was step 2 against step 1.
    #     A fresh start makes both sides the same step.
    rows_d = 3
    torch.manual_seed(11)
    q3 = torch.randn(rows_d, num_v, dk, device="cuda")
    k3 = torch.randn(rows_d, num_v, dk, device="cuda")
    v3 = torch.randn(rows_d, num_v, dv, device="cuda")
    b3 = torch.randn(rows_d, num_v, device="cuda")
    g3 = -torch.rand(rows_d, num_v, device="cuda") - 0.1
    st0 = torch.randn(rows_d, num_v, dk, dv, device="cuda")
    st_b = st0.clone()
    out_b = torch.zeros(rows_d, num_v, dv, device="cuda")
    yb = table.delta_batch(q3, k3, v3, b3, g3, st_b, out_b)
    got_b = yb.clone()
    got_st_b = st_b.clone()
    worst_b = 0.0
    for b in range(rows_d):
        # One row through the single-row op, from the same fresh state.
        st_one = st0[b].clone()
        out_one = torch.zeros(num_v, dv, device="cuda")
        y_one = table.delta(q3[b], k3[b], v3[b], b3[b], g3[b], st_one, out_one)
        worst_b = max(worst_b, _rel(got_b[b], _host(y_one)),
                      _rel(got_st_b[b], _host(st_one)))
    ok_b = worst_b < TOL
    print(f"  {'ok  ' if ok_b else 'FAIL'} delta_batch == per-row  B={rows_d}      "
          f"rel={worst_b:.3e}          n/a", flush=True)
    if not ok_b:
        failures.append("delta_batch")
    del qr, kr, vr, br, ar, qo, ko, vo, st, st_ref, s, q3, k3, v3, st_b, out_b, st0
    torch.cuda.empty_cache()

    _sep("9. pack.repack_state_dict: one transpose per weight, one resident copy")
    sd = {"q.weight": torch.randn(512, 256, dtype=torch.bfloat16, device="cuda"),
          "n.weight": torch.randn(256, dtype=torch.bfloat16, device="cuda")}
    report = pack.repack_state_dict(sd)
    print(f"  {report}  leftover={pack.unpacked_keys(sd)}", flush=True)
    if report["packed"] != 1 or pack.unpacked_keys(sd):
        failures.append("pack.repack_state_dict")

    _sep("summary")
    print(f"  failures: {failures or '(none)'}", flush=True)
    print("\nCOREX_CHECK_OPS_DONE", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
