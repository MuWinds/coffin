"""Multi-query verify attention: bit-exactness and the long-context speedup.

Answers whether the speculative-verify step can fold its c draft tokens into one
kernel without changing the arithmetic. A verify step runs c draft tokens per
sequence row against the same history KV. ``attn_paged_multi_q`` keeps the
per-token decode kernel's arithmetic for every (row, query) but launches the c
queries of a row as adjacent blocks, so the c passes over the history hit L2
instead of re-reading it from HBM.

This probe checks:

1. **bit-exact**: each query of the multi-query kernel equals the per-token
   ``attn_paged`` call (differing element count must be 0), for c in {1,2,4};
2. the block_mask path: an all-ones mask is still exact, and a half-mask skips
   the deselected blocks (timing only -- it is the sparse mode);
3. per-step latency of one multi-query call vs c per-token calls at 4k and 32k
   context, and the speedup.

Read section 1 as a gate: ``differing=0`` is required, because the verify step
feeds a rejection sampler that compares logits, and any arithmetic change would
show up as tokens the verifier accepts that the target model would not have
produced. Section 2's all-ones row must also be 0 -- a non-zero there means the
mask path itself perturbs the result even when it excludes nothing -- while the
half-mask row is *expected* to differ, since it skips blocks on purpose. Section
3 is the payoff: the speedup grows with context because the saving is history
re-reads, so a win at 4k that vanishes at 32k points at the wrong conclusion.

Run inside the container from the engine root, holding the GPU lock and pinning
``TORCH_EXTENSIONS_DIR`` so a concurrent run cannot fight over the extension
build baton::

    TORCH_EXTENSIONS_DIR=/root/private_data/torch_ext_kernels \
    flock /root/private_data/gpu.lock -c \
      'cd /root/private_data/engine_kernels && PYTHONPATH=$PWD python3 -u diag/probe_multi_q.py'

Module: ``coffin/diag``. Standalone probe: imports ``hip_ops`` and the kernels
under test, and nothing under ``coffin/`` imports it.

Constraints:
    * Shapes are the production ones (nh=16, nkv=4, hd=256), and the slot map is
      deliberately the identity mapping, so position p addresses slot p as a real
      sequence does; a permuted map would hide an off-by-one in the position the
      kernel derives per query.
    * Parity is exact element equality, never a tolerance. Both paths emit bf16
      with the same accumulation order, so agreement is the contract.
    * ``flock`` and the isolated ``TORCH_EXTENSIONS_DIR`` are both required: a
      co-tenant invalidates the timings, and a shared extensions directory
      serialises two runs on the same build baton.
"""

from __future__ import annotations

import sys

import torch

import hip_ops

NH, NKV, HD = 16, 4, 256
REP = NH // NKV
SCALE = HD ** -0.5
SPLIT = 8
B = 8


def bench(fn, iters: int, warmup: int = 3, reps: int = 3) -> float:
    """Time ``fn`` with CUDA events and return milliseconds per call.

    Args:
        fn: Zero-argument callable to time.
        iters: Calls per timed region; elapsed time is divided by this.
        warmup: Untimed calls first, to absorb kernel first-load.
        reps: Timed regions; the fastest is kept.

    Returns:
        Best region's elapsed time divided by ``iters``, in milliseconds. At the
        32k rows a single call is long enough that a mean would be dominated by
        one preempted sample.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / iters)
    return best


def make_case(ctx: int, c: int):
    """Build one multi-query problem: pool, slot map, queries and scratch buffers.

    Args:
        ctx: History length shared by every row.
        c: Draft tokens per row, i.e. queries per sequence.

    Returns:
        A dict holding the INT8 pools and scales, the slot map ``[B, smax]``, q
        and gate ``[B, c, NH, HD]``, the output, the split-K scratch sized
        ``[B * c, ...]``, the int32 context lengths and ``smax``/``c`` for the
        callers.

    The slot map is the identity, so position p maps to slot p exactly as a real
    contiguous sequence does, and the split-K scratch is sized by ``B * c``
    because the kernel treats every (row, query) as its own partition owner.
    """
    torch.manual_seed(0)
    smax = ctx + c
    nslots = smax + 8
    kpool = torch.randint(-127, 127, (nslots, NKV, HD), dtype=torch.int8, device="cuda")
    vpool = torch.randint(-127, 127, (nslots, NKV, HD), dtype=torch.int8, device="cuda")
    kscale = (torch.rand(nslots, NKV, device="cuda") * 0.02 + 1e-3).float()
    vscale = (torch.rand(nslots, NKV, device="cuda") * 0.02 + 1e-3).float()
    # slot map: position p -> slot p (contiguous, like a real sequence)
    slots = torch.arange(smax, dtype=torch.int64, device="cuda")
    slots = slots.unsqueeze(0).expand(B, smax).contiguous()
    q = torch.randn(B, c, NH, HD, dtype=torch.float32, device="cuda").bfloat16()
    gate = torch.randn(B, c, NH, HD, dtype=torch.float32, device="cuda").bfloat16()
    out = torch.empty_like(q)
    pa = torch.empty(B * c, NH, SPLIT, HD, dtype=torch.float32, device="cuda")
    pm = torch.empty(B * c, NH, SPLIT, 2, dtype=torch.float32, device="cuda")
    ctx_lens = torch.full((B,), ctx, dtype=torch.int32, device="cuda")
    return dict(kpool=kpool, vpool=vpool, kscale=kscale, vscale=vscale, slots=slots,
                q=q, gate=gate, out=out, pa=pa, pm=pm, ctx_lens=ctx_lens,
                smax=smax, c=c)


def per_token_ref(cs, out_ref, pa1, pm1):
    """Run the old per-token kernel for every (row, query) to build the oracle.

    Args:
        cs: Case dict from ``make_case``.
        out_ref: Output buffer ``[B, c, NH, HD]`` to fill.
        pa1: Split-K accumulator scratch for one query.
        pm1: Split-K max/sum scratch for one query.

    Returns:
        ``out_ref`` after every (b, j) has been written.

    Each query's position is ``ctx_lens[b] + j``, which is the whole point of the
    comparison: the multi-query kernel must derive the same per-query position
    internally, including the causal bound that lets query j see its own slot.
    """
    c = cs["c"]
    for b in range(B):
        for j in range(c):
            pos = torch.tensor([cs["ctx_lens"][b].item() + j],
                               dtype=torch.int64, device="cuda")
            hip_ops.attn_paged(cs["q"][b, j].contiguous(), cs["gate"][b, j].contiguous(),
                               cs["kpool"], cs["vpool"], cs["kscale"], cs["vscale"],
                               cs["slots"][b], out_ref[b, j], pos, cs["smax"],
                               REP, SCALE, pa1, pm1)
    return out_ref


def multi(cs, mask=None, mask_block: int = 64):
    """Invoke ``attn_paged_multi_q`` on the case and return its output buffer.

    Args:
        cs: Case dict from ``make_case``; its ``out`` buffer is written in place.
        mask: Optional per-block ``[B, nblocks]`` uint8 mask, where 1 admits a
            block. ``None`` means dense attention and is the path the parity
            check uses.
        mask_block: Keys per mask bit. 64 is the granularity the sparse path was
            built for; a larger value skips more history per bit but cannot
            express a precise window.

    Returns:
        The output buffer ``[B, c, NH, HD]``.
    """
    return hip_ops.attn_paged_multi_q(cs["q"], cs["gate"], cs["kpool"], cs["vpool"],
                                      cs["kscale"], cs["vscale"], cs["slots"],
                                      cs["ctx_lens"], cs["out"], cs["smax"], REP,
                                      SCALE, cs["pa"], cs["pm"], mask, mask_block)


def main() -> int:
    """Run the parity sweep, the block-mask checks and the latency comparison.

    Returns:
        ``0`` on completion, ``1`` when the HIP extension is unavailable. The
        parity verdicts are printed per row; they are not folded into the exit
        code, so read the ``OK``/``BAD`` column.
    """
    if not hip_ops.available():
        print(f"HIP extension unavailable: {hip_ops.last_error()}")
        return 1
    print(f"DCU: {torch.cuda.get_device_name(0)}  torch {torch.__version__}  "
          f"hip_ops OK")

    print("\n== 1. bit-exactness vs per-token attn_paged ==")
    pa1 = torch.empty(NH, SPLIT, HD, dtype=torch.float32, device="cuda")
    pm1 = torch.empty(NH, SPLIT, 2, dtype=torch.float32, device="cuda")
    for ctx in (7, 333, 4096, 32768):
        for c in (1, 2, 4):
            cs = make_case(ctx, c)
            ref = per_token_ref(cs, torch.empty_like(cs["out"]), pa1, pm1)
            got = multi(cs)
            diff = int((got != ref).sum())
            md = (got.float() - ref.float()).abs().max().item()
            print(f"ctx={ctx:<5} c={c}  differing={diff:<6} max_abs={md:.2e}  "
                  f"{'OK' if diff == 0 else 'BAD'}")
            del cs, ref, got
            torch.cuda.empty_cache()

    print("\n== 2. block_mask: all-ones must stay exact, sparse skips blocks ==")
    cs = make_case(4096, 2)
    full = multi(cs).clone()
    nblocks = (cs["smax"] + 63) // 64
    ones = torch.ones(B, nblocks, dtype=torch.uint8, device="cuda")
    got1 = multi(cs, ones).clone()
    print(f"  all-ones mask differing={(int((got1 != full).sum()))} (expect 0)")
    half = ones.clone()
    half[:, nblocks // 2:] = 0
    got2 = multi(cs, half).clone()
    print(f"  half mask: differing vs full = {int((got2 != full).sum())} "
          f"(sparse mode, expected non-zero)")
    del cs, full, got1, got2, ones, half
    torch.cuda.empty_cache()

    print("\n== 3. per-step latency: 1 multi-q call vs c per-token calls ==")
    print(f"{'ctx':>6} {'c':>2} {'multi us':>9} {'per-token us':>13} {'speedup':>8}")
    for ctx in (4096, 32768):
        for c in (2, 4):
            cs = make_case(ctx, c)
            pa_ref = torch.empty(NH, SPLIT, HD, dtype=torch.float32, device="cuda")
            pm_ref = torch.empty(NH, SPLIT, 2, dtype=torch.float32, device="cuda")
            out_ref = torch.empty_like(cs["out"])
            pos = [torch.tensor([ctx + j], dtype=torch.int64, device="cuda")
                   for j in range(c)]

            t_multi = bench(lambda: multi(cs), 10, warmup=2)

            def loop():
                """Issue the c per-token calls this kernel replaces, for timing."""
                for b in range(B):
                    for j in range(c):
                        hip_ops.attn_paged(cs["q"][b, j].contiguous(),
                                           cs["gate"][b, j].contiguous(),
                                           cs["kpool"], cs["vpool"], cs["kscale"],
                                           cs["vscale"], cs["slots"][b],
                                           out_ref[b, j], pos[j], cs["smax"],
                                           REP, SCALE, pa_ref, pm_ref)

            t_loop = bench(loop, 10, warmup=2)
            print(f"{ctx:>6} {c:>2} {t_multi*1e3:>9.1f} {t_loop*1e3:>13.1f} "
                  f"{t_loop/t_multi:>8.2f}x")
            del cs, out_ref, pa_ref, pm_ref
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
