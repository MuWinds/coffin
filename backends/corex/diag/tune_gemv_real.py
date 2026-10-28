"""Tune the W8A16 decode GEMV against the shapes a real decode step calls.

The probe tables in this directory were built by reading ``config.json`` and guessing at the
projections, and every one of them is wrong about the fused ones. Measured from a live int8
decode step (``probe_shapes.py``), one step issues 790 ``gemv_kn_i8`` calls over seven
distinct geometries, and the largest traffic item is ``N=18432, K=2560`` -- the fused
gate/up projection, which is twice the size of the ``mlp_up`` shape the earlier sweeps
tuned against. A dispatch table built on the guessed shapes would tune kernels that never
run.

So this sweeps the measured shapes, and separately sweeps a synthetic ``N`` series at fixed
``K`` to locate the point where split-K stops paying. That crossover is what lets
``pick_split`` be a rule rather than a table of seven magic numbers.

Module: ``coffin/backends/corex/diag``. Diagnostic only; runs on the card.

Constraints:
    * Host fp64 reference over a column window: this card's device fp64 returns zeros
      (COREX_PORT.md §0), so a device-side widening compares against nothing.
    * ``lm_head`` is 636 MB int8 and its full reference would cost more than the sweep, so
      every candidate is checked on a window and timed whole.
    * Every candidate is checked, not just the winner: a configuration that is fast because
      it reads the wrong bytes is exactly what a timing-only sweep cannot see.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

#: ``(N, K, calls_per_step)`` measured from a live int8 decode step. The call count is what
#: turns a per-shape bandwidth into a share of the step, so the sweep can be ordered by
#: what actually matters rather than by matrix size.
REAL_SHAPES: tuple[tuple[int, int, int], ...] = (
    (18432, 2560, 32),
    (2560, 9216, 32),
    (12352, 2560, 24),
    (2560, 4096, 32),
    (10240, 2560, 8),
    (248320, 2560, 1),
    (2560, 5120, 2),
)

#: Fixed-``K`` series for the split crossover: at what column count does a single pass
#: already fill the card, so that split-K only adds partial-buffer traffic?
CROSSOVER_N: tuple[int, ...] = (4096, 8192, 16384, 32768, 65536, 131072, 248320)
CROSSOVER_K = 2560

THREADS: tuple[int, ...] = (128, 256, 512)
SPLITS: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
UNROLL = 4

#: Columns the host reference covers on a wide shape.
REF_COLS = 512


def _bench(fn, iters: int = 15, warmup: int = 4) -> float:
    """Time a callable with device events.

    Args:
        fn: Zero-argument callable.
        iters: Timed iterations.
        warmup: Untimed iterations.

    Returns:
        Mean milliseconds per call.
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


def _sweep_shape(module, n: int, k: int, ceiling: float, threads_set, split_set):
    """Sweep one geometry and return ``(best_ms, best_tag, rows)``.

    Args:
        module: The loaded kernel extension.
        n: Output columns.
        k: Reduction length.
        ceiling: Measured copy ceiling in GB/s, for the percentage column.
        threads_set: Thread counts to try.
        split_set: Split-K widths to try.

    Returns:
        ``(best_ms, best_tag, rows)`` where ``rows`` is ``[(ms, tag, gbs, rel, ok)]``.
    """
    import torch

    wt = torch.randint(-127, 128, (k, n), dtype=torch.int8, device="cuda")
    scale = (torch.rand(n, dtype=torch.float32, device="cuda") * 0.01 + 0.001)
    x = torch.randn(k, dtype=torch.bfloat16, device="cuda") * 0.02

    # Reference from the dequantised weight, so quantisation error is not folded into the
    # comparison and a scale bug cannot hide behind the expected few-percent drift.
    cols = min(REF_COLS, n)
    w_deq = (wt[:, :cols].float() * scale[:cols]).cpu().double()
    ref = w_deq.t() @ x.detach().cpu().double()
    scale_ref = float(ref.abs().max()) or 1.0

    nbytes = n * k
    rows = []
    for th in threads_set:
        for sp in split_set:
            tag = f"t{th}/s{sp}"
            try:
                y = module.gemv_kn_i8(wt, x, scale, th, UNROLL, sp)
                torch.cuda.synchronize()
                got = y.detach().cpu().double()[:cols]
                rel = float((got - ref).abs().max()) / scale_ref
                ms = _bench(lambda t=th, s=sp:
                            module.gemv_kn_i8(wt, x, scale, t, UNROLL, s))
                gbs = nbytes / (ms * 1e-3) / 1e9
                ok = rel < 2e-2
                rows.append((ms, tag, gbs, rel, ok))
                print(f"    {tag:<12} {ms:8.4f} ms {gbs:7.1f} GB/s "
                      f"{100 * gbs / ceiling:>3.0f}%  rel={rel:.2e}"
                      f"{'' if ok else '   <-- WRONG'}", flush=True)
            except Exception as exc:              # noqa: BLE001 - a bad combo is data
                print(f"    {tag:<12} FAILED {type(exc).__name__}: {exc}", flush=True)
    del wt, scale, x, w_deq, ref
    torch.cuda.empty_cache()
    good = [r for r in rows if r[4]]
    if not good:
        return (float("inf"), "-", rows)
    good.sort()
    return (good[0][0], good[0][1], rows)


def main() -> int:
    """Sweep the measured decode shapes and the split crossover.

    Returns:
        ``0`` always: the job is read for its numbers, not for a pass/fail.
    """
    import torch

    import backends
    import backends.corex as corex
    from backends.corex import kernels

    print(f"  backend: {backends.status()}", flush=True)
    if not corex.usable():
        print("  backend not usable on this host", flush=True)
        return 1
    module = kernels.load_kernels()

    sms = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"  SMs: {sms}", flush=True)
    ceiling_n = 64 * 1024 * 1024
    src = torch.empty(ceiling_n, dtype=torch.uint8, device="cuda")
    dst = torch.empty(ceiling_n, dtype=torch.uint8, device="cuda")
    ceiling = 2 * ceiling_n / (_bench(lambda: dst.copy_(src), 8, 3) * 1e-3) / 1e9
    print(f"  copy ceiling {ceiling:.0f} GB/s", flush=True)
    del src, dst
    torch.cuda.empty_cache()

    print(f"\n{'=' * 78}\nreal decode shapes: best (threads, split) per geometry\n{'=' * 78}",
          flush=True)
    table = []
    for n, k, calls in REAL_SHAPES:
        nbytes = n * k
        print(f"\n  N={n} K={k}  {nbytes / 1e6:.1f} MB x {calls}/step = "
              f"{nbytes * calls / 1e6:.0f} MB/step", flush=True)
        best_ms, best_tag, rows = _sweep_shape(module, n, k, ceiling, THREADS, SPLITS)
        if rows:
            prod = next((r for r in rows if r[1] == "t256/s8"), None)
            gain = f"{prod[0] / best_ms:.2f}x" if prod and best_ms > 0 else "-"
            print(f"    best {best_tag:<12} {best_ms:8.4f} ms  "
                  f"vs production t256/s8 {gain}", flush=True)
            table.append((n, k, calls, best_tag, best_ms, nbytes * calls, gain, prod))
    torch.cuda.empty_cache()

    print(f"\n{'=' * 78}\nsplit crossover at K={CROSSOVER_K}: where does a single pass "
          f"already fill the card?\n{'=' * 78}", flush=True)
    cross = []
    for n in CROSSOVER_N:
        print(f"\n  N={n} K={CROSSOVER_K}  ({n * CROSSOVER_K / 1e6:.1f} MB)", flush=True)
        best_ms, best_tag, rows = _sweep_shape(module, n, CROSSOVER_K, ceiling, (256,), SPLITS)
        if rows:
            by_split = {r[1]: r for r in rows}
            s1 = by_split.get("t256/s1")
            s32 = by_split.get("t256/s32")
            blocks = (n + 255) // 256
            cross.append((n, blocks, best_tag, best_ms,
                          s1[2] if s1 else 0.0, s32[2] if s32 else 0.0))
        torch.cuda.empty_cache()

    print(f"\n{'=' * 78}\nthe dispatch table\n{'=' * 78}", flush=True)
    print(f"  {'N':>7} {'K':>6} {'calls':>5} {'best':>12} {'ms':>9} {'MB/step':>9} "
          f"{'gain':>6} {'ms/step saved':>13}", flush=True)
    total_before = total_after = 0.0
    for n, k, calls, tag, ms, step_bytes, gain, prod in table:
        before = prod[0] * calls if prod else ms * calls
        after = ms * calls
        total_before += before
        total_after += after
        print(f"  {n:>7} {k:>6} {calls:>5} {tag:>12} {ms:>9.4f} "
              f"{step_bytes / 1e6:>9.0f} {gain:>6} {before - after:>13.3f}", flush=True)
    print(f"  {'':>7} {'':>6} {'':>5} {'':>12} {'':>9} {'':>9} {'':>6} "
          f"{total_before - total_after:>13.3f}  <- total per step", flush=True)
    print(f"\n  projections under the current dispatch: {total_before:.2f} ms/step\n"
          f"  projections with the measured best  : {total_after:.2f} ms/step", flush=True)

    print(f"\n{'=' * 78}\ncrossover summary (blocks = ceil(N/256))\n{'=' * 78}", flush=True)
    print(f"  {'N':>7} {'blocks':>7} {'best':>12} {'s1 GB/s':>9} {'s32 GB/s':>9}", flush=True)
    for n, blocks, tag, _ms, s1, s32 in cross:
        print(f"  {n:>7} {blocks:>7} {tag:>12} {s1:>9.1f} {s32:>9.1f}", flush=True)

    print("\nREAL_TUNE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
