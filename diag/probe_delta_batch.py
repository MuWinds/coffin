"""Parity + timing probe for the batched linear-attn kernels.

Answers one question before the batched kernels are allowed into the decode
path: does batching preserve the recurrent state bit-for-bit, and what does it
buy per step? Three batched kernels replace the per-row loops of the phase-1
decode path:

  conv1d_silu_batch   depthwise causal conv1d (kernel 4) + SiLU
  delta_prologue_batch q/k L2 norm + head expand + beta/g
  delta_batch         gated-delta-rule state update

Read the output in two parts. In the parity table, ``exact 1.0000`` and ``OK``
mean the batched row reproduced the single-row kernel exactly; ``BAD`` means the
row diverged and the kernel must not be shipped — the fp32 recurrent state is
updated with the same formula in the same order, so bit equality is the
requirement, not a happy accident. In the latency table, ``batch us`` against
``loop us`` is the per-step saving; a speedup at or below 1.00x means the batch
kernel is launch-bound at that B and the loop should stay.

Run inside the container from the engine root, holding the GPU lock so the
timing rows are not stolen by another tenant::

    flock /root/private_data/gpu.lock -c \
      'cd /root/private_data/engine_kernels && export ... PYTHONPATH=$PWD && \
       python3 diag/probe_delta_batch.py'

``--iters N`` changes the timing loop count; it does not affect parity.

Module: ``coffin/diag``. Standalone probe: imports ``hip_ops`` and the kernels
under test, and nothing under ``coffin/`` imports it.

Constraints:
    * The shapes are the production ones (nk=16, nv=32, dk=dv=128, conv_dim=8192,
      kernel=4), so a shape-dependent indexing bug cannot hide behind a toy case.
    * Parity is judged by bit equality, never by a tolerance. A loose threshold
      would pass a kernel whose rounding drifts the state over a long context.
    * The timing loop mutates the state it times and is expected to; only the
      parity pass reads the result. Do not merge the two.
    * ``flock`` is not optional: a co-tenant on the DCU invalidates every microsecond
      in the latency table.
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
    """Build one batch of conv1d_silu inputs with the production shapes and scales.

    Args:
        B: Batch size, the row count under test.
        seed: Seed for the torch RNG. Fixed so the parity pass and the timing
            pass see the same data, and so a divergence is reproducible.

    Returns:
        ``(w, x, state, out)``: weight ``[conv_dim, 4]``, bf16 input
        ``[B, conv_dim]``, fp32 rolling state ``[B, conv_dim, 3]`` and a
        preallocated fp32 output ``[B, conv_dim]``.

    The weights are scaled by 0.1 and the input by 0.5 so the SiLU stays in its
    non-saturated range; larger draws would push the comparison into the tail
    where bf16 rounding dominates and bit equality could never hold.
    """
    torch.manual_seed(seed)
    w = torch.randn(CONV_DIM, 4, dtype=torch.float32, device="cuda") * 0.1
    x = (torch.randn(B, CONV_DIM, dtype=torch.float32, device="cuda") * 0.5).bfloat16()
    state = torch.randn(B, CONV_DIM, 3, dtype=torch.float32, device="cuda") * 0.2
    out = torch.empty(B, CONV_DIM, dtype=torch.float32, device="cuda")
    return w, x, state, out


def make_prologue(B: int, seed: int = 2):
    """Build one batch of delta_prologue inputs, including the five outputs.

    Args:
        B: Batch size.
        seed: Torch RNG seed, fixed for the same reason as ``make_conv``.

    Returns:
        ``(q, k, v, b, a, neg_expA, dt_bias, out, qscale)``. ``out`` is a list of
        five preallocated buffers in kernel order: ``q_pro``, ``k_pro``,
        ``v_pro``, ``beta``, ``g``.

    ``b`` and ``a`` stay bf16 because the real gate path hands them over in bf16,
    and ``neg_expA`` is forced strictly negative so the decay term is always in
    range; a positive value would exercise a branch production never takes.
    """
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
    """Build one batch of delta_batch inputs with a non-singular recurrent state.

    Args:
        B: Batch size.
        seed: Torch RNG seed, fixed for reproducibility.

    Returns:
        ``(q, k, v, beta, g, st, out)``. ``st`` is the fp32 state
        ``[B, NV, DK, DV]`` that the parity pass clones and compares; ``out`` is
        the preallocated ``[B, NV, DV]`` result.

    ``beta`` is offset positive and ``g`` kept small and negative so the
    gated-delta update does not collapse to zero or diverge, which would make a
    broken kernel look bit-identical to the reference.
    """
    torch.manual_seed(seed)
    q = torch.randn(B, NV, DK, dtype=torch.float32, device="cuda") * 0.1
    k = torch.randn(B, NV, DK, dtype=torch.float32, device="cuda") * 0.1
    v = torch.randn(B, NV, DV, dtype=torch.float32, device="cuda") * 0.5
    beta = torch.rand(B, NV, dtype=torch.float32, device="cuda") * 0.9 + 0.05
    g = -(torch.rand(B, NV, dtype=torch.float32, device="cuda") * 0.3 + 0.001)
    st = torch.randn(B, NV, DK, DV, dtype=torch.float32, device="cuda") * 0.01
    out = torch.empty(B, NV, DV, dtype=torch.float32, device="cuda")
    return q, k, v, beta, g, st, out


# Reference implementations that call the single-row kernels. These are the
# oracle: a batched kernel is correct only when it matches them bit for bit.
def conv_ref(w, x, state, out):
    """Run conv1d_silu one row at a time to produce the expected result.

    Args:
        w: Shared conv weight ``[conv_dim, 4]``.
        x: Input rows ``[B, conv_dim]``, bf16.
        state: Rolling state ``[B, conv_dim, 3]``; updated in place, so the
            caller passes a clone.
        out: Preallocated output ``[B, conv_dim]``.

    Returns:
        A new tensor holding the row-by-row reference output.
    """
    B = x.shape[0]
    ref = torch.empty_like(out)
    for b in range(B):
        hip_ops.conv1d_silu(w, x[b], state[b], ref[b])
    return ref


def prologue_ref(args):
    """Run delta_prologue one row at a time to produce the expected result.

    Args:
        args: The tuple from ``make_prologue``. Its ``out`` list is only used for
            shape and dtype; the reference writes fresh buffers.

    Returns:
        A list of five reference tensors matching ``q_pro``, ``k_pro``,
        ``v_pro``, ``beta``, ``g``.

    Each row is passed ``.contiguous()`` because the single-row kernel takes a
    packed row; the batched kernel is expected to handle strided slices itself,
    which the strided case at the end of ``main`` verifies.
    """
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
    """Run delta one row at a time to produce the expected output and state.

    Args:
        q: Queries ``[B, NV, DK]``.
        k: Keys ``[B, NV, DK]``.
        v: Values ``[B, NV, DV]``.
        beta: Gate ``[B, NV]``.
        g: Log decay ``[B, NV]``.
        st: Recurrent state ``[B, NV, DK, DV]``, mutated in place.
        out: Preallocated output ``[B, NV, DV]``.

    Returns:
        ``(ref_out, st_ref)``. The state is returned explicitly because the
        update is in-place: comparing only the output would miss a kernel that
        computes the right row and stores the wrong state.
    """
    B = q.shape[0]
    ref = torch.empty_like(out)
    st_ref = st.clone()
    for i in range(B):
        hip_ops.delta(q[i], k[i], v[i], beta[i], g[i], st_ref[i], ref[i])
    return ref, st_ref


# The same operations through the batched entry points.
def conv_batch(w, x, state, out):
    """Dispatch conv1d_silu_batch and return the output buffer.

    Args:
        w: Shared conv weight ``[conv_dim, 4]``.
        x: Input rows ``[B, conv_dim]``.
        state: Rolling state ``[B, conv_dim, 3]``, updated in place.
        out: Preallocated output ``[B, conv_dim]``.

    Returns:
        ``out``, returned so the caller can hand it straight to ``report``.
    """
    hip_ops.conv1d_silu_batch(w, x, state, out)
    return out


def prologue_batch(args):
    """Dispatch delta_prologue_batch over the whole batch.

    Args:
        args: The tuple from ``make_prologue``; its ``out`` buffers are written
            in place.

    Returns:
        The list of five output buffers, in the same order as ``prologue_ref``
        so the two can be zipped.
    """
    q, k, v, b_, a_, neg_expA, dt_bias, out, qscale = args
    hip_ops.delta_prologue_batch(q, k, v, b_, a_, neg_expA, dt_bias,
                                 out[0], out[1], out[2], out[3], out[4],
                                 qscale, REP)
    return out


def delta_batch(q, k, v, beta, g, st, out):
    """Dispatch delta_batch and return the output buffer.

    Args:
        q: Queries ``[B, NV, DK]``.
        k: Keys ``[B, NV, DK]``.
        v: Values ``[B, NV, DV]``.
        beta: Gate ``[B, NV]``.
        g: Log decay ``[B, NV]``.
        st: Recurrent state ``[B, NV, DK, DV]``, updated in place.
        out: Preallocated output ``[B, NV, DV]``.

    Returns:
        ``out``. The caller compares ``st`` separately, since the state is
        mutated in place.
    """
    hip_ops.delta_batch(q, k, v, beta, g, st, out)
    return out


def bench(fn, iters: int = ITERS):
    """Time ``fn`` with CUDA events and return the best per-call cost in us.

    Args:
        fn: Zero-argument callable to time.
        iters: Calls per timed region. The per-call figure is the elapsed time
            divided by this count, so launch overhead is amortised but the loop
            itself stays dominated by real work.

    Returns:
        Fastest of five timed regions, in microseconds. The minimum rather than
        the mean: a co-tenant's burst can only ever add time, so the best region
        is the closest estimate of the kernel's own cost.

    Launch overhead dominates these kernels at small B, which is why the result
    is reported in microseconds and compared against a row-by-row loop rather
    than against a theoretical roof.
    """
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
    """Measure the three parity statistics for one output tensor.

    Args:
        got: Output produced by the batched kernel.
        ref: Output produced by the single-row oracle.

    Returns:
        ``(max_abs, max_rel, exact)``: largest absolute difference, that
        difference relative to the reference's own peak magnitude (so a caller
        sees whether the error is significant at this output's scale), and the
        fraction of elements that match bit for bit.

    The reference peak is clamped away from zero because an all-zero reference
    row would otherwise produce a division by zero instead of the exact-1.0
    answer it deserves.
    """
    d = (got.float() - ref.float()).abs()
    denom = ref.float().abs().max().clamp_min(1e-30)
    exact = (got == ref).float().mean().item()
    return d.max().item(), (d.max() / denom).item(), exact


def report(name, got, ref):
    """Print one parity row and return whether it is bit-identical.

    Args:
        name: Row label, e.g. ``q_pro`` or ``state``.
        got: Batched-kernel output.
        ref: Single-row oracle output.

    Returns:
        ``True`` when every element matches exactly. ``main`` ANDs these into
        ``all_ok`` and uses it as the process exit code, so a single ``BAD`` row
        fails the run rather than being scrolled past in a table.
    """
    mx, rel, exact = errs(got, ref)
    tag = "OK " if exact == 1.0 else "BAD"
    print(f"  {name:<34} max_abs {mx:>9.3e}  max_rel {rel:>9.3e}  "
          f"exact {exact:>7.4f}  {tag}")
    return exact == 1.0


def main() -> int:
    """Run the parity sweep at B in (1, 2, 4, 8), then the strided case, then the timings.

    Returns:
        ``0`` when every parity row is bit-identical, ``1`` when any row differs
        or the HIP extension is unavailable. The exit code is the gate: a
        non-zero result means the batched kernels must not be used.
    """
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
    # [B, conv_dim] conv output (row pitch conv_dim, not packed). This is the
    # layout the batched kernel sees in production, so a packed-only kernel
    # would pass every row above and still corrupt the model.
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
            """Run the row-by-row conv1d_silu loop this kernel replaces, for timing."""
            for i in range(B):
                hip_ops.conv1d_silu(w, x[i], state[i], ref_out[i])

        tb = bench(lambda: hip_ops.conv1d_silu_batch(w, x, state, out))
        tl = bench(conv_loop)
        print(f"{B:>2}  {'conv1d_silu':<20} {tb:>10.2f} {tl:>10.2f} {tl / tb:>8.2f}x")

        args = make_prologue(B)
        refq = [t.clone() for t in args[7]]
        bq, kq, vq, bb, aa, ne, db, oq, qs = args

        def prologue_loop():
            """Run the row-by-row delta_prologue loop this kernel replaces, for timing."""
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
            """Run the row-by-row delta loop this kernel replaces, for timing."""
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
