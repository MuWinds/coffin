"""Wide-M verify GEMV: packed fp16 (v_dot2_f32_f16) path vs the int8 baseline.

Answers whether the speculative-verify projection should use the packed-fp16
dot2 kernel instead of the scalar INT8 wide kernel. Verify projects c+1
activation vectors per row in one weight pass (M=16 at B=8/K=1, M=32 at K=3).
The scalar int8 wide kernel is bit-exact but element-rate bound (~85 Gelem/s at
M=16 vs 199 for the M=8 kernel), so this probe measures the packed-fp16 variant:
int8 weights -> fp16 (exact), bf16 activations -> fp16 once per call, two MACs
per ``v_dot2_f32_f16`` with fp32 accumulation.

Reports, per shape and M in {16,24,32}:

* numerical difference against the concatenation of the old M=8 kernel's rows
  (max abs diff and fraction of exactly-equal elements -- the dot2 path is fp32
  accumulation with a different rounding order, so it is close, not bit-equal);
* effective bandwidth and element rate for the best (rows, mw, U, threads);
* the weighted verify projection segment before (c+1 x M=8) and after (1x wide).

Read the output bottom-up. Sections 3 and 4 are the decision: ``speedup`` is the
whole 129-module projection segment, so anything near 1.00x means the kernel win
does not survive being summed over the model. Sections 1 and 2 explain why -- a
high ``Gelem/s`` at one M with no segment speedup points at a small-K shape where
launch overhead, not element rate, is the limit. ``exact`` below 1.0 in section 1
is expected, not a bug: accept the dot2 path only if ``max_rel`` stays small
enough for the verifier's acceptance test to be unaffected.

Run inside the container from the engine root, holding the GPU lock and pinning
``TORCH_EXTENSIONS_DIR`` to a per-deployment directory so a concurrent run cannot
fight over the extension build baton::

    TORCH_EXTENSIONS_DIR=/root/private_data/torch_ext_kernels \
    flock /root/private_data/gpu.lock -c \
      'cd /root/private_data/engine_kernels && PYTHONPATH=$PWD python3 -u diag/probe_wide_gemv.py'

Module: ``coffin/diag``. Standalone probe: imports ``hip_ops`` and the kernels
under test, and nothing under ``coffin/`` imports it.

Constraints:
    * All weight matrices stay INT8 row-major ``[N, K]``; the dot2 path converts
      a row to fp16 at load time, which is exact, and never reorders K. A probe
      that pre-transposed the weights would measure a layout production lacks.
    * The module list is the real 129-projection census of Qwen3.5-4B, and the
      per-segment totals are weighted by ``count``. An unweighted average over
      shapes would be dominated by lm_head, which runs once per token.
    * ``flock`` and the isolated ``TORCH_EXTENSIONS_DIR`` are both required: a
      co-tenant invalidates the timings, and a shared extensions directory makes
      two runs rebuild and serialise on the same file baton.
"""

from __future__ import annotations

import sys

import torch

import hip_ops

#: (name, N, K, module count) -- 129 projection modules of Qwen3.5-4B
MODULES = [
    ("linear.in_qkv_z_b_a", 12352, 2560, 24),
    ("linear.out_proj", 2560, 4096, 24),
    ("fa.qkv", 6144, 2560, 8),
    ("fa.o_proj", 2560, 4096, 8),
    ("mlp.gate_up", 18432, 2560, 32),
    ("mlp.down", 2560, 9216, 32),
    ("lm_head", 151936, 2560, 1),
]

WIDE_MS = (16, 24, 32)
#: (rows per warp, activation vectors per warp) for the dot2 kernel
TILINGS = ((1, 16), (1, 8), (2, 8), (4, 4))


def bench(fn, iters: int, warmup: int = 3, reps: int = 3) -> float:
    """Time ``fn`` with CUDA events and return milliseconds per call.

    Args:
        fn: Zero-argument callable to time.
        iters: Calls per timed region. Callers size this by weight bytes so every
            region lasts roughly the same wall time.
        warmup: Untimed calls first, to absorb extension load and first-touch.
        reps: Timed regions; the fastest is kept.

    Returns:
        Best region's elapsed time divided by ``iters``, in milliseconds.
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
        best = min(best, s.elapsed_time(e) / iters)     # ms
    return best


def iters_for(nbytes: float, lo: int = 3, hi: int = 100) -> int:
    """Choose a repetition count that makes a timed region last about 0.1 s.

    Args:
        nbytes: Bytes streamed per call; the cost driver for a bandwidth-bound
            GEMV.
        lo: Floor. Small matrices would otherwise request thousands of calls and
            spend the region's time in Python launch overhead.
        hi: Ceiling. Large matrices would otherwise run once and measure noise.

    Returns:
        Calls per region, clamped to ``[lo, hi]``.
    """
    return max(lo, min(hi, int(0.1e9 / max(nbytes, 1.0))))


def make_w(N, K, seed):
    """Build one INT8 row-major weight matrix and its per-row fp32 scales.

    Args:
        N: Output rows (columns of the projection).
        K: Reduction length.
        seed: Torch RNG seed, derived in ``main`` from the module name and N so
            each shape gets stable but distinct data.

    Returns:
        ``(w, s)``: INT8 codes ``[N, K]`` and fp32 scales ``[N]``. The scale
        range mirrors the checkpoint's, which keeps the dequantised products in
        the range where bf16 output rounding behaves as it does in production.

    The weights are INT8 and row-major because that is the checkpoint layout;
    the kernel consumes a row at a time along K, so a transposed matrix would be
    measured at the wrong stride.
    """
    torch.manual_seed(seed)
    w = torch.randint(-127, 127, (N, K), dtype=torch.int8, device="cuda")
    s = (torch.rand(N, device="cuda") * 0.02 + 1e-3).float()
    return w, s


def split_ref(w, s, X, M, U=2):
    """Build the expected wide-M result by concatenating consecutive M=8 calls.

    Args:
        w: INT8 weights ``[N, K]``.
        s: Per-row fp32 scales.
        X: Activations ``[M, K]``, bf16.
        M: Row count; must be a multiple of 8.
        U: Unroll factor forwarded to the old kernel.

    Returns:
        Reference output ``[M, N]`` from the old kernel. This is the oracle the
        dot2 path is judged against: it defines "same answer" as "same as what
        the shipped kernel produces", not as some idealised arithmetic.
    """
    return torch.cat([hip_ops.gemv_w8a16_m(w, s, X[i:i + 8], U)
                      for i in range(0, M, 8)], dim=0)


def dot2_cands(w, s, X, M):
    """Enumerate the dot2 tilings that are legal for this row count.

    Args:
        w: INT8 weights ``[N, K]``.
        s: Per-row fp32 scales.
        X: Activations ``[M, K]``, bf16.
        M: Row count.

    Returns:
        ``{label: callable}`` for every (rows-per-warp, vectors-per-warp) tiling
        crossed with the unroll factors and block sizes whose warp counts divide
        the row groups.

    The ``wpb % gm`` filter is not a preference: the kernel assigns whole row
    groups to warps, so a block size whose warp count does not divide the group
    count leaves a warp with no work and the launch is invalid. Filtering here
    keeps a config error from looking like a slow kernel.
    """
    cands = {}
    for rows, mw in TILINGS:
        gm = M // mw
        for U in (2, 4):
            for th in (32, 64, 96, 128, 192, 256):
                wpb = th // 32
                if wpb % gm != 0:
                    continue
                name = f"r{rows} mw{mw} u{U} th{th}"
                cands[name] = (lambda rows=rows, mw=mw, U=U, th=th:
                               hip_ops.gemv_w8a16_wide_dot2_dbg(
                                   w, s, X, U, rows, mw, th))
    return cands


def err_stats(got, ref):
    """Measure the numerical gap between the dot2 path and the INT8 oracle.

    Args:
        got: Output from the dot2 kernel.
        ref: Output from the concatenated M=8 oracle.

    Returns:
        ``(max_abs, exact)``: largest absolute difference and the fraction of
        elements that match bit for bit. Bit equality is not expected here --
        fp32 accumulation over a different pairing order rounds differently --
        so ``exact`` is reported as information, and ``max_abs`` is what the
        verifier's tolerance has to survive.
    """
    d = (got.float() - ref.float()).abs()
    return d.max().item(), (got == ref).float().mean().item()


def main() -> int:
    """Run the accuracy/tiling sweep, the element-rate table, and the two segment tables.

    Returns:
        ``0`` on completion, ``1`` when the HIP extension is unavailable. A
        missing tiling shows up as a missing row, so read the tables rather than
        the exit code.
    """
    if not hip_ops.available():
        print(f"HIP extension unavailable: {hip_ops.last_error()}")
        return 1
    print(f"DCU: {torch.cuda.get_device_name(0)}  torch {torch.__version__}  "
          f"hip_ops OK")

    print("\n== 1. dot2 wide-M: accuracy vs old M=8 rows, best tiling ==")
    print(f"{'shape':<22} {'M':>3} {'best cfg':<20} {'max_abs':>9} {'max_rel':>9} "
          f"{'exact':>8} {'GB/s':>6} {'Gelem/s':>8}")
    best_cfg = {}
    for name, N, K, count in MODULES:
        w, s = make_w(N, K, seed=len(name) * 7 + N % 97)
        X = torch.randn(32, K, dtype=torch.bfloat16, device="cuda")
        nbytes = N * K
        it = iters_for(nbytes)
        for M in WIDE_MS:
            ref = split_ref(w, s, X[:M], M)
            cands = dot2_cands(w, s, X[:M], M)
            ms, cfg = 1e9, None
            for cname, fn in cands.items():
                try:
                    t = bench(fn, it)
                except Exception:
                    continue
                if t < ms:
                    ms, cfg = t, cname
            got = cands[cfg]()
            t_def = bench(lambda: hip_ops.gemv_w8a16_wide_dot2(w, s, X[:M], 2, 0), it)
            d = (got.float() - ref.float()).abs()
            mx = d.max().item()
            rel = (d.max() / ref.float().abs().max().clamp_min(1e-30)).item()
            ex = (got == ref).float().mean().item()
            best_cfg[(name, M)] = (ms, cfg)
            print(f"{name:<22} {M:>3} {cfg:<20} {mx:>9.2e} {rel:>9.2e} "
                  f"{ex:>8.5f} {nbytes/1e9/(ms/1e3):>6.0f} {nbytes/(ms/1e3)/1e9:>8.0f} "
                  f"def {nbytes/1e9/(t_def/1e3):>4.0f}GB/s")
        del w, X
        torch.cuda.empty_cache()

    print("\n== 2. element rate vs M (int8, mlp.gate_up, 47.2M weight elements) ==")
    w, s = make_w(18432, 2560, seed=11)
    nelem = 18432 * 2560
    X = torch.randn(32, 2560, dtype=torch.bfloat16, device="cuda")

    def rate(t_ms):
        return nelem / (t_ms / 1e3) / 1e9      # Gelem/s and GB/s (1 byte/elem)

    t = bench(lambda: hip_ops.gemv_w8a16(w, s, X[0], 1, 2), 50)
    print(f"  M=1  old     {rate(t):6.0f} GB/s  {rate(t):6.0f} Gelem/s")
    for M in (2, 4, 8):
        best = 1e9
        for U in (1, 2, 4):
            try:
                best = min(best, bench(
                    lambda U=U: hip_ops.gemv_w8a16_m(w, s, X[:M], U), 40))
            except Exception:
                continue
        print(f"  M={M:<2} old     {rate(best):6.0f} GB/s  {rate(best):6.0f} Gelem/s")
    t = bench(lambda: hip_ops.gemv_w8a16_wide(w, s, X[:16], 2, 0), 30)
    print(f"  M=16 int8wide {rate(t):6.0f} GB/s  {rate(t):6.0f} Gelem/s")
    for M in WIDE_MS:
        t = best_cfg[("mlp.gate_up", M)][0]
        print(f"  M={M:<2} dot2    {rate(t):6.0f} GB/s  {rate(t):6.0f} Gelem/s")

    # dot2 on the normal decode widths too (M=1/2/4/8): a win here helps every
    # batched decode step, not just verify.
    normal = {}
    for M in (1, 2, 4, 8):
        best, cfg = 1e9, None
        for rows, mw in ((1, 8), (2, 8), (4, 4), (1, 4), (2, 4), (4, 2),
                         (1, 2), (2, 2), (4, 1), (1, 1)):
            if M % mw:
                continue
            gm = M // mw
            for th in (96, 128, 192, 256):
                if (th // 32) % gm:
                    continue
                try:
                    tt = bench(lambda rows=rows, mw=mw, th=th:
                               hip_ops.gemv_w8a16_wide_dot2_dbg(
                                   w, s, X[:M], 2, rows, mw, th), 40)
                except Exception:
                    continue
                if tt < best:
                    best, cfg = tt, (rows, mw, th)
        normal[M] = (best, cfg)
        print(f"  M={M:<2} dot2    {rate(best):6.0f} GB/s  {rate(best):6.0f} Gelem/s "
              f"[r{cfg[0]} mw{cfg[1]} th{cfg[2]}]")
    del w, X
    torch.cuda.empty_cache()

    print("\n== 3. verify projection segment, weighted over the 129 modules ==")
    print("   before = (c+1) x M=8 batched calls (one weight pass per token)")
    print("   after  = 1x wide dot2 call")
    print(f"{'K (M)':<8} {'bytes GB':>9} {'before ms':>10} {'after ms':>9} "
          f"{'before GB/s':>11} {'after GB/s':>10} {'speedup':>8}")
    for nspec, M in ((1, 16), (2, 24), (3, 32)):
        seg_bytes = before = after = 0.0
        for name, N, K, count in MODULES:
            w, s = make_w(N, K, seed=len(name) * 7 + N % 97)
            X = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
            t8 = bench(lambda: hip_ops.gemv_w8a16_m(w, s, X[:8], 2), iters_for(N * K))
            before += count * (nspec + 1) * t8
            after += count * best_cfg[(name, M)][0]
            seg_bytes += count * N * K
            del w, X
            torch.cuda.empty_cache()
        print(f"K={nspec} M={M:<3} {seg_bytes/1e9:>9.2f} {before:>10.2f} "
              f"{after:>9.2f} {seg_bytes/1e9/(before/1e3):>11.0f} "
              f"{seg_bytes/1e9/(after/1e3):>10.0f} {before/after:>8.2f}x")

    print("\n== 4. normal decode projection segment: int8 old vs dot2 (M=1/2/4/8) ==")
    print("   one weight pass per step; same 129 modules (3.87 GB int8)")
    print(f"{'M':>2} {'old ms':>8} {'dot2 ms':>8} {'old GB/s':>9} "
          f"{'dot2 GB/s':>10} {'speedup':>8}")
    for M in (1, 2, 4, 8):
        seg_bytes = old = new = 0.0
        for name, N, K, count in MODULES:
            w, s = make_w(N, K, seed=len(name) * 7 + N % 97)
            X = torch.randn(8, K, dtype=torch.bfloat16, device="cuda")
            it = iters_for(N * K)
            if M == 1:
                t_old = bench(lambda: hip_ops.gemv_w8a16(w, s, X[0], 1, 2), it)
            else:
                best_old = 1e9
                for U in (1, 2, 4):
                    try:
                        best_old = min(best_old, bench(
                            lambda U=U: hip_ops.gemv_w8a16_m(w, s, X[:M], U), it))
                    except Exception:
                        continue
                t_old = best_old
            best_new, cfg = 1e9, None
            for rows, mw in ((1, 8), (2, 8), (4, 4), (1, 4), (2, 4), (4, 2),
                             (1, 2), (2, 2), (4, 1), (1, 1)):
                if M % mw:
                    continue
                gm = M // mw
                for th in (128, 192, 256):
                    if (th // 32) % gm:
                        continue
                    try:
                        tt = bench(lambda rows=rows, mw=mw, th=th:
                                   hip_ops.gemv_w8a16_wide_dot2_dbg(
                                       w, s, X[:M], 2, rows, mw, th), it)
                    except Exception:
                        continue
                    if tt < best_new:
                        best_new, cfg = tt, (rows, mw, th)
            old += count * t_old
            new += count * best_new
            seg_bytes += count * N * K
            del w, X
            torch.cuda.empty_cache()
        print(f"{M:>2} {old:>8.2f} {new:>8.2f} {seg_bytes/1e9/(old/1e3):>9.0f} "
              f"{seg_bytes/1e9/(new/1e3):>10.0f} {old/new:>8.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
