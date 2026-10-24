"""Multi-query verify attention: bit-exactness and the long-context speedup.

A verify step runs c draft tokens per sequence row against the same history KV.
``attn_paged_multi_q`` keeps the per-token decode kernel's arithmetic for every
(row, query) but launches the c queries of a row as adjacent blocks, so the c
passes over the history hit L2 instead of re-reading it from HBM.

This probe checks:

1. **bit-exact**: each query of the multi-query kernel equals the per-token
   ``attn_paged`` call (differing element count must be 0), for c in {1,2,4};
2. the block_mask path: an all-ones mask is still exact, and a half-mask skips
   the deselected blocks (timing only -- it is the sparse mode);
3. per-step latency of one multi-query call vs c per-token calls at 4k and 32k
   context, and the speedup.

Run from the engine root on the DCU container (GPU lock)::

    TORCH_EXTENSIONS_DIR=/root/private_data/torch_ext_kernels \
    flock /root/private_data/gpu.lock -c \
      'cd /root/private_data/engine_kernels && PYTHONPATH=$PWD python3 -u diag/probe_multi_q.py'
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
    """Reference: the old per-token decode kernel for every (b, j)."""
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
    return hip_ops.attn_paged_multi_q(cs["q"], cs["gate"], cs["kpool"], cs["vpool"],
                                      cs["kscale"], cs["vscale"], cs["slots"],
                                      cs["ctx_lens"], cs["out"], cs["smax"], REP,
                                      SCALE, cs["pa"], cs["pm"], mask, mask_block)


def main() -> int:
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
