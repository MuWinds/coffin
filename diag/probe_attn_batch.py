"""Parity + timing probe for the batched paged-attention kernel.

The contract for ``attn_paged_batch`` is that a row of the batch kernel is the
single-row ``attn_paged`` result: same int8 dequantisation, same per-token scale
fold, same split-K online softmax, same output gate.  This probe checks that
numerically for B in {1, 2, 4, 8} and for windows that stress the edges of the
split logic -- context_lens of 1, of a non-multiple of 64, of a non-multiple of
the block size 1024, and longer than one streamed block -- then times the batch
kernel against the row-by-row loop over the old kernel (the phase-1 path).

Run from the engine root on the DCU container (GPU lock!)::

    flock /root/private_data/gpu.lock -c \
      'cd /root/private_data/engine_kernels && export ... PYTHONPATH=$PWD && \
       python3 diag/probe_attn_batch.py'

``--iters N`` changes the timing loop count.  Errors are element-wise against
the old kernel; the expected result is bit-identical output (both sides end in
bf16), so the max abs/rel columns should be 0 and ``exact`` 1.0000.
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
    """Random int8 pool + random per-(slot, head) fp32 scales."""
    torch.manual_seed(seed)
    kpool = torch.randint(-127, 127, (NSLOTS, NKV, HD), dtype=torch.int8, device="cuda")
    vpool = torch.randint(-127, 127, (NSLOTS, NKV, HD), dtype=torch.int8, device="cuda")
    kscale = (torch.rand(NSLOTS, NKV, device="cuda") * 0.02 + 1e-3).float()
    vscale = (torch.rand(NSLOTS, NKV, device="cuda") * 0.02 + 1e-3).float()
    return kpool, vpool, kscale, vscale


def make_slots(B: int, smax: int) -> torch.Tensor:
    """[B, smax] device slot maps: random per-row permutation of pool slots."""
    slots = torch.zeros(B, smax, dtype=torch.int64, device="cuda")
    for b in range(B):
        slots[b] = torch.randperm(NSLOTS, device="cuda")[:smax]
    return slots


def make_qg(B: int):
    """[B, nh, hd] bf16 q and gate (contiguous)."""
    q = torch.randn(B, NH, HD, dtype=torch.float32, device="cuda") * 0.5
    gate = torch.randn(B, NH, HD, dtype=torch.float32, device="cuda")
    return q.bfloat16(), gate.bfloat16()


def ref_rows(q, gate, kpool, vpool, kscale, vscale, slots, lens):
    """Single-row ``attn_paged`` per row (the phase-1 fallback path)."""
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
    pa = torch.zeros(B, NH, split, HD, dtype=torch.float32, device="cuda")
    pm = torch.zeros(B, NH, split, 2, dtype=torch.float32, device="cuda")
    return pa, pm


def run_batch(q, gate, kpool, vpool, kscale, vscale, slots, lens,
              out=None, pa=None, pm=None):
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
    """Per-step cost of ``fn`` replayed from a captured CUDA graph."""
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
    d = (got.float() - ref.float()).abs()
    denom = ref.float().abs().max().clamp_min(1e-30)
    exact = (got == ref).float().mean().item()
    return d.max().item(), (d.max() / denom).item(), exact


def main() -> int:
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
    # and windows longer than one streamed block.
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
    # [B, nh, 2*hd] in the model, so the batch kernel must honour strides.
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
                hip_ops.attn_paged_batch(q, gate, kpool, vpool, kscale, vscale,
                                         slots, cl, out, smax, REP, SCALE, pa, pm)

            def loop_call():
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
            hip_ops.attn_paged_batch(q, gate, kpool, vpool, kscale, vscale,
                                     slots, cl, out, smax, REP, SCALE, pa, pm)

        def loop_call():
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
