"""Parity + timing probe for the batched paged-attention kernel.

Answers the question the batch decode path depends on: is ``attn_paged_batch``
numerically the same function as the single-row ``attn_paged``, and is batching
it worth the extra kernel? The contract is that a row of the batch kernel is the
single-row result: same int8 dequantisation, same per-token scale fold, same
split-K online softmax, same output gate. This probe checks that numerically for
B in {1, 2, 4, 8} and for windows that stress the edges of the split logic --
context_lens of 1, of a non-multiple of 64, of a non-multiple of the block size
1024, and longer than one streamed block -- then times the batch kernel against
the row-by-row loop over the old kernel (the phase-1 path).

Read the output in two parts. In the parity table, ``exact 1.0000`` and ``OK``
mean the row matched the old kernel bit for bit, which is the expected result
because both sides end in bf16; ``BAD`` means it did not, and the batched kernel
must not ship. The ``strided q/gate`` row is the important one: q and gate are
column slices of one fused projection in the model, so a kernel that only
handles packed input passes every other row and still corrupts production. In
the latency tables, ``batch us`` against ``loop us`` is the per-step saving --
``<= 1.00x`` means the batch kernel is launch-bound at that B. The CUDA-graph
table repeats the comparison with launch overhead amortised, which is the
regime a captured decode step actually runs in; a speedup that appears there but
not in the eager table means the win is pure launch reduction.

Run inside the container from the engine root, holding the GPU lock so the
timing rows are not stolen by another tenant::

    flock /root/private_data/gpu.lock -c \
      'cd /root/private_data/engine_kernels && export ... PYTHONPATH=$PWD && \
       python3 diag/probe_attn_batch.py'

``--iters N`` changes the timing loop count.

Module: ``coffin/diag``. Standalone probe: imports ``hip_ops`` and the kernels
under test, and nothing under ``coffin/`` imports it.

Constraints:
    * Shapes are the production ones (nh=16, nkv=4, hd=256, SPLIT=8 matching
      ``ATTN_SPLIT`` in ``attn.cu`` / ``tuning.py``), so a split-boundary bug
      cannot hide behind a toy head count.
    * Parity is bit equality, never a tolerance. Both paths emit bf16, and the
      online-softmax combine is order-sensitive; a tolerance would hide exactly
      the reassociation bug this kernel risks.
    * ``flock`` is not optional: a co-tenant invalidates every microsecond in the
      latency tables.
    * Each row gets its own slot permutation and its own q/gate, so a row's
      result cannot be explained by another row's data.
"""

from __future__ import annotations

import sys
import time

import torch

import hip_ops

NH, NKV, HD = 16, 4, 256
REP = NH // NKV
SCALE = HD ** -0.5
SPLIT = 8                      # ATTN_SPLIT in attn.cu / tuning.py
NSLOTS = 8192
ITERS = 200


def make_pool(seed: int = 0):
    """Build a random INT8 KV pool with per-(slot, head) fp32 scales.

    Args:
        seed: Torch RNG seed, fixed so a failing row is reproducible.

    Returns:
        ``(kpool, vpool, kscale, vscale)``: INT8 codes ``[NSLOTS, NKV, HD]`` and
        fp32 scales ``[NSLOTS, NKV]``.

    The scales are drawn from 0.02..0.021 rather than around 1.0 because the
    kernel folds a per-token scale into the dequantised product; realistic small
    scales keep the accumulation inside bf16 range, where bit equality is
    achievable. Large scales would saturate and make every row look unequal.
    """
    torch.manual_seed(seed)
    kpool = torch.randint(-127, 127, (NSLOTS, NKV, HD), dtype=torch.int8, device="cuda")
    vpool = torch.randint(-127, 127, (NSLOTS, NKV, HD), dtype=torch.int8, device="cuda")
    kscale = (torch.rand(NSLOTS, NKV, device="cuda") * 0.02 + 1e-3).float()
    vscale = (torch.rand(NSLOTS, NKV, device="cuda") * 0.02 + 1e-3).float()
    return kpool, vpool, kscale, vscale


def make_slots(B: int, smax: int) -> torch.Tensor:
    """Build per-row slot maps drawing from one shared pool.

    Args:
        B: Number of rows.
        smax: Window capacity in slots; also the row stride of the map.

    Returns:
        ``[B, smax]`` int64 slot ids, each row an independent random permutation
        of the pool. Independent permutations matter: a kernel that mixed up rows
        would still produce plausible output if every row addressed the same
        slots.
    """
    slots = torch.zeros(B, smax, dtype=torch.int64, device="cuda")
    for b in range(B):
        slots[b] = torch.randperm(NSLOTS, device="cuda")[:smax]
    return slots


def make_qg(B: int):
    """Build a packed bf16 query and output gate for ``B`` rows.

    Args:
        B: Number of rows.

    Returns:
        ``(q, gate)``, each ``[B, nh, hd]`` and contiguous. This is the packed
        layout, used everywhere except the dedicated strided case; the strided
        test exists because the model hands over column slices instead.
    """
    q = torch.randn(B, NH, HD, dtype=torch.float32, device="cuda") * 0.5
    gate = torch.randn(B, NH, HD, dtype=torch.float32, device="cuda")
    return q.bfloat16(), gate.bfloat16()


def ref_rows(q, gate, kpool, vpool, kscale, vscale, slots, lens):
    """Run the single-row kernel once per row to produce the expected output.

    Args:
        q: Queries ``[B, nh, hd]``, bf16.
        gate: Output gate ``[B, nh, hd]``, bf16.
        kpool: INT8 key pool.
        vpool: INT8 value pool.
        kscale: Per-slot key scales.
        vscale: Per-slot value scales.
        slots: Slot maps ``[B, smax]``.
        lens: Per-row context length. The query position is ``lens[b] - 1``,
            i.e. the last token of the row's window.

    Returns:
        Reference output with the same shape and dtype as ``q``.

    Each row is passed ``.contiguous()`` because the single-row kernel takes
    packed input; that is a limitation of the old path, not of the batch kernel
    under test.
    """
    B = q.shape[0]
    out = torch.empty_like(q)
    pa = torch.empty(NH, SPLIT, HD, dtype=torch.float32, device="cuda")
    pm = torch.empty(NH, SPLIT, 2, dtype=torch.float32, device="cuda")
    for b in range(B):
        pos = torch.tensor([lens[b] - 1], dtype=torch.int64, device="cuda")
        hip_ops.attn_paged(q[b].contiguous(), gate[b].contiguous(), kpool, vpool,
                           kscale, vscale, slots[b], out[b], pos, slots.shape[1],
                           REP, SCALE, pa, pm)
    return out


def batch_parts(B, split=SPLIT):
    """Allocate the split-K partial-attention and partial-max scratch for a batch.

    Args:
        B: Number of rows.
        split: Number of split-K partitions; must match ``ATTN_SPLIT`` in the
            kernel, which is why ``SPLIT`` is the default rather than a tunable.

    Returns:
        ``(pa, pm)``: partial accumulators ``[B, nh, split, hd]`` and partial
        max/sum ``[B, nh, split, 2]``, both fp32 and zeroed.

    The buffers are zeroed, not left empty: the kernel combines all ``split``
    partitions, so an uninitialised partition would contribute garbage on a
    short window where some partitions never ran.
    """
    pa = torch.zeros(B, NH, split, HD, dtype=torch.float32, device="cuda")
    pm = torch.zeros(B, NH, split, 2, dtype=torch.float32, device="cuda")
    return pa, pm


def run_batch(q, gate, kpool, vpool, kscale, vscale, slots, lens,
              out=None, pa=None, pm=None):
    """Invoke ``attn_paged_batch`` once and return its output buffer.

    Args:
        q: Queries ``[B, nh, hd]``, bf16, possibly strided.
        gate: Output gate ``[B, nh, hd]``, bf16, possibly strided.
        kpool: INT8 key pool.
        vpool: INT8 value pool.
        kscale: Per-slot key scales.
        vscale: Per-slot value scales.
        slots: Slot maps ``[B, smax]``.
        lens: Per-row context length.
        out: Output buffer; allocated when omitted.
        pa: Partial accumulators; allocated together with ``pm`` when omitted.
        pm: Partial max/sum.

    Returns:
        The output buffer. ``lens`` is converted to the int32 tensor the kernel
        signature expects here, so callers pass a plain Python list.
    """
    B, smax = q.shape[0], slots.shape[1]
    if out is None:
        out = torch.empty_like(q)
    if pa is None:
        pa, pm = batch_parts(B)
    cl = torch.tensor(lens, dtype=torch.int32, device="cuda")
    hip_ops.attn_paged_batch(q, gate, kpool, vpool, kscale, vscale, slots, cl,
                             out, smax, REP, SCALE, pa, pm)
    return out


def bench(fn, iters: int = ITERS):
    """Time ``fn`` eagerly with CUDA events and return the best per-call cost in us.

    Args:
        fn: Zero-argument callable to time.
        iters: Calls per timed region; elapsed time is divided by this.

    Returns:
        Fastest of five timed regions, in microseconds per step. Minimum rather
        than mean, because a co-tenant can only add time. This figure still
        includes launch overhead, which is why ``replay_us`` exists as well.
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
        best = min(best, s.elapsed_time(e) * 1000.0 / iters)      # us / step
    return best


def replay_us(fn, iters: int = ITERS):
    """Per-step cost of ``fn`` replayed from a captured CUDA graph.

    Args:
        fn: Zero-argument callable to capture and replay.
        iters: Replays per timed region.

    Returns:
        Fastest of five timed regions, in microseconds per replay. Launch
        overhead is paid once at capture, so this is the kernel's own cost -- the
        regime a captured decode step runs in, and the only way to tell a real
        batching win from a launch-count win.

    ``fn`` must not perform host synchronisation or allocate inside the captured
    region; both are illegal in a CUDA graph and would fail at capture rather
    than be silently skipped.
    """
    g = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(5):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            g.replay()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) * 1000.0 / iters)
    return best


def errs(got, ref):
    """Measure the three parity statistics for one output tensor.

    Args:
        got: Output from the batched kernel.
        ref: Output from the row-by-row oracle.

    Returns:
        ``(max_abs, max_rel, exact)``: largest absolute difference, that
        difference relative to the reference peak (clamped off zero so an
        all-zero row reports a finite number), and the fraction of elements
        matching bit for bit.
    """
    d = (got.float() - ref.float()).abs()
    denom = ref.float().abs().max().clamp_min(1e-30)
    exact = (got == ref).float().mean().item()
    return d.max().item(), (d.max() / denom).item(), exact


def main() -> int:
    """Run the parity sweep, the strided case, then the eager and graph timings.

    Returns:
        ``0`` always when the extension loads; ``1`` when HIP is unavailable.
        A ``BAD`` parity row is printed but does not change the exit code, so
        read the table rather than trusting the status. The probe is a report,
        not a gate.
    """
    global ITERS
    if "--iters" in sys.argv:
        ITERS = int(sys.argv[sys.argv.index("--iters") + 1])

    if not hip_ops.available():
        print(f"HIP extension unavailable: {hip_ops.last_error()}")
        return 1
    print(f"DCU: {torch.cuda.get_device_name(0)}  torch {torch.__version__}  "
          f"hip_ops OK")
    kpool, vpool, kscale, vscale = make_pool()

    # B = 1 against the old kernel at the window edges: 1 key, a sub-block
    # window, a non-multiple of the 1024-key block, a non-multiple of split,
    # and windows longer than one streamed block. These lens are the cases where
    # split-K partition boundaries and the tail block disagree, so they are where
    # an off-by-one in the online-softmax combine shows up.
    parity = []
    cases = [(1, [1]), (1, [2]), (1, [17]), (1, [63]), (1, [64]), (1, [65]),
             (1, [100]), (1, [333]), (1, [1023]), (1, [1024]), (1, [1025]),
             (1, [4096]),
             (2, [1, 333]), (4, [1, 37, 127, 300]),
             (8, [1, 37, 127, 300, 65, 200, 3, 511]),
             (8, [64, 64, 64, 64, 128, 256, 256, 512])]
    for B, lens in cases:
        smax = max(lens)
        slots = make_slots(B, smax)
        q, gate = make_qg(B)
        ref = ref_rows(q, gate, kpool, vpool, kscale, vscale, slots, lens)
        got = run_batch(q, gate, kpool, vpool, kscale, vscale, slots, lens)
        parity.append((B, lens, *errs(got, ref)))

    # strided q/gate: they are column slices of one fused projection output
    # [B, nh, 2*hd] in the model, so the batch kernel must honour strides. A
    # packed-only kernel passes every case above and still breaks the model.
    B, lens = 4, [1, 65, 200, 300]
    smax = max(lens)
    slots = make_slots(B, smax)
    qg = torch.randn(B, NH, 2 * HD, dtype=torch.float32, device="cuda")
    qg = qg.bfloat16()
    q = qg[:, :, :HD]
    gate = qg[:, :, HD:]
    out = torch.empty(B, NH, HD, dtype=torch.bfloat16, device="cuda")
    pa, pm = batch_parts(B)
    cl = torch.tensor(lens, dtype=torch.int32, device="cuda")
    hip_ops.attn_paged_batch(q, gate, kpool, vpool, kscale, vscale, slots, cl,
                             out, smax, REP, SCALE, pa, pm)
    ref = ref_rows(qg[:, :, :HD].contiguous(), qg[:, :, HD:].contiguous(),
                   kpool, vpool, kscale, vscale, slots, lens)
    strided = errs(out, ref)

    print("\n== parity vs single-row attn_paged (B=1 must be exact) ==")
    print(f"{'B':>2}  {'lens':<44} {'max_abs':>10} {'max_rel':>10} {'exact':>7}")
    for B, lens, mx, rel, exact in parity:
        tag = "OK " if exact == 1.0 else "BAD"
        print(f"{B:>2}  {str(lens):<44} {mx:>10.3e} {rel:>10.3e} {exact:>7.4f} {tag}")
    mx, rel, exact = strided
    tag = "OK " if exact == 1.0 else "BAD"
    print(f"{B:>2}  {'strided q/gate [B,nh,2hd] slice':<44} {mx:>10.3e} {rel:>10.3e} "
          f"{exact:>7.4f} {tag}")

    print("\n== per-step latency: batch kernel vs row-by-row old kernel ==")
    print(f"{'B':>2}  {'regime':<11} {'sum(ctx)':>8} {'batch us':>10} "
          f"{'loop us':>10} {'speedup':>8}")
    timings = []
    regimes = {
        "mixed": [1, 37, 127, 300, 65, 200, 3, 511],
        "long": [4096] * 8,
    }
    for name, base in regimes.items():
        for B in (1, 2, 4, 8):
            lens = base[:B] if name == "mixed" else [base[0]] * B
            smax = max(lens)
            slots = make_slots(B, smax)
            q, gate = make_qg(B)
            out = torch.empty_like(q)
            pa, pm = batch_parts(B)
            cl = torch.tensor(lens, dtype=torch.int32, device="cuda")
            ref_out = torch.empty_like(q)
            pa1 = torch.empty(NH, SPLIT, HD, dtype=torch.float32, device="cuda")
            pm1 = torch.empty(NH, SPLIT, 2, dtype=torch.float32, device="cuda")
            pos = [torch.tensor([lens[b] - 1], dtype=torch.int64, device="cuda")
                   for b in range(B)]

            def batch_call():
                """Issue one batched attention call, for timing or graph capture."""
                hip_ops.attn_paged_batch(q, gate, kpool, vpool, kscale, vscale,
                                         slots, cl, out, smax, REP, SCALE, pa, pm)

            def loop_call():
                """Issue the row-by-row old-kernel loop, for timing or graph capture."""
                for b in range(B):
                    hip_ops.attn_paged(q[b].contiguous(), gate[b].contiguous(),
                                       kpool, vpool, kscale, vscale, slots[b],
                                       ref_out[b], pos[b], smax, REP, SCALE, pa1, pm1)

            tb = bench(batch_call)
            tl = bench(loop_call)
            timings.append((B, name, sum(lens), tb, tl))
            print(f"{B:>2}  {name:<11} {sum(lens):>8} {tb:>10.2f} "
                  f"{tl:>10.2f} {tl / tb:>8.2f}x")

    # Graph replay: inside a captured decode step the comparison is kernel
    # time with launch overhead already amortised per replay.
    print("\n== CUDA-graph replay per step (mixed windows) ==")
    print(f"{'B':>2}  {'batch us':>10} {'loop us':>10} {'speedup':>8} "
          f"{'kernels b/l':>11}")
    for B in (1, 2, 4, 8):
        lens = regimes["mixed"][:B]
        smax = max(lens)
        slots = make_slots(B, smax)
        q, gate = make_qg(B)
        out = torch.empty_like(q)
        pa, pm = batch_parts(B)
        cl = torch.tensor(lens, dtype=torch.int32, device="cuda")
        ref_out = torch.empty_like(q)
        pa1 = torch.empty(NH, SPLIT, HD, dtype=torch.float32, device="cuda")
        pm1 = torch.empty(NH, SPLIT, 2, dtype=torch.float32, device="cuda")
        pos = [torch.tensor([lens[b] - 1], dtype=torch.int64, device="cuda")
               for b in range(B)]

        def batch_call():
            """Issue one batched attention call, for timing or graph capture."""
            hip_ops.attn_paged_batch(q, gate, kpool, vpool, kscale, vscale,
                                     slots, cl, out, smax, REP, SCALE, pa, pm)

        def loop_call():
            """Issue the row-by-row old-kernel loop, for timing or graph capture."""
            for b in range(B):
                hip_ops.attn_paged(q[b].contiguous(), gate[b].contiguous(),
                                   kpool, vpool, kscale, vscale, slots[b],
                                   ref_out[b], pos[b], smax, REP, SCALE, pa1, pm1)

        tb = replay_us(batch_call)
        tl = replay_us(loop_call)
        print(f"{B:>2}  {tb:>10.2f} {tl:>10.2f} {tl / tb:>8.2f}x {2:>5}/{2 * B:<5}")

    print("\ncontext_lens of the mixed regime: "
          f"{regimes['mixed'][:8]};  long regime: 4096 for every row")
    return 0


if __name__ == "__main__":
    sys.exit(main())
