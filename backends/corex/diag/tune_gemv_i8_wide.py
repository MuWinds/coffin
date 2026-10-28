"""Sweep the W8A16 GEMV over weight-load width, threads and split-K, on the real shapes.

The int8 GEMV is the bulk of a decode step and runs at 130-262 GB/s against a ~572 GB/s
copy ceiling. Two causes have been argued for that gap and they are not exclusive:

    * transaction waste -- a 64-lane wavefront loading one byte per lane covers 64 bytes of
      a 128-byte line, so ``width = 2`` (64 lanes x 2 bytes = exactly one line) should be
      the transaction-optimal load;
    * parallelism -- the card wants outstanding loads, and the split-K width sets how many
      blocks are in flight.

The catch is that width and split trade against each other: reading ``width`` columns per
thread divides the thread count by ``width``, so a wide kernel is starved exactly where a
narrow one wants more blocks. Only the product is measurable, which is why every width is
swept against every split instead of one knob at a time. The previous negative result on
the four-wide kernel was measured with split capped at 8; this sweep reaches 32.

Every candidate is checked against a host fp64 reference built from the *dequantised* int8
weight, so a configuration that is fast because it reads the wrong bytes cannot win. That
check runs on every cell rather than on the winner, because the failure it catches is
invisible to a timing table.

Module: ``coffin/backends/corex/diag``. Diagnostic only; runs on the card.

Constraints:
    * No CUDA graphs anywhere in this file. This part's driver wedges cards permanently on
      graph replay, so timing uses events around a loop and never capture.
    * The reference is windowed to the first ``REF_COLS`` columns and the timing is not:
      ``lm_head`` is 635 MB int8 and a full host fp64 reference would cost more than the
      whole sweep.
    * ``caps.autotune`` stays ``True`` and nothing here runs at load: the output is a table
      to hardcode, never a benchmark on the decode path.
    * Host fp64 reference: this card's device fp64 returns zeros, so widening on the device
      would compare against nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: ``coffin/`` itself is the ``sys.path`` root; see ``check_ops.py``.
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

#: The geometries a live int8 decode step actually calls, measured on the serving path
#: (``load_weights(quant=True) -> create_model(weights=sd)``), with the per-step call count.
#: These are not the round numbers the earlier probes assumed -- ``create_model`` alone
#: loads dense weights and measures the bf16 kernel, which is a trap this table exists to
#: avoid. Call counts matter because they weight the step: ``lm_head`` is one call and
#: ``mlp_down`` is thirty-two.
SHAPES: tuple[tuple[str, int, int, int], ...] = (
    ("mlp_up", 18432, 2560, 32),
    ("mlp_down", 2560, 9216, 32),
    ("attn_qkvzb_a", 12352, 2560, 24),
    ("o_proj", 2560, 4096, 32),
    ("full_attn_qkv", 10240, 2560, 8),
    ("lm_head", 248320, 2560, 1),
    ("mlp_k5120", 2560, 5120, 2),
)

#: Columns per thread. 1 is the scalar control, 2 is one 128-byte line per 64-lane
#: Columns per thread. 2 is one 128-byte line per 64-lane wavefront and only halves the
#: thread count, so it is the predicted winner on the narrow shapes; 4 is the pre-existing
#: wide kernel, kept in the grid so the old negative result is retested rather than assumed;
#: 8 and 16 exist for the wide shapes where thread count is not scarce.
#:
#: ``width = 1`` is deliberately absent: it is the scalar control and it has already been
#: swept on these shapes at threads 128/256/512 x split 1..32, so re-timing it would spend
#: the job budget on a row that is known. ``--with-scalar`` puts it back for a session where
#: that reference is not in hand.
WIDTHS: tuple[int, ...] = (2, 4, 8, 16)

#: Threads per block, matching the shipped scalar sweep so the two tables are comparable
#: row for row. 64 is off by default: it was covered by the scalar sweep and it is not a
#: candidate the dispatch would ever pick for these shapes.
THREADS: tuple[int, ...] = (128, 256, 512)

#: Threads added by ``--with-threads64``, for the record that it was covered.
THREADS_EXTRA: tuple[int, ...] = (64,)

#: Split-K widths. 16 and 32 are the point: the shipped dispatch capped at 8, and the
#: narrow shapes are precisely the ones short of blocks. 1 and 8 are the controls that say
#: whether a wide kernel loses only because it was starved -- the old x4 condemnation was
#: taken at splits no larger than 8.
SPLITS: tuple[int, ...] = (1, 2, 4, 8, 16, 32)

#: Unroll is held at 4 for the whole grid. It is a separate axis with its own measured
#: answer, and sweeping it here would triple the job for a knob this one does not move.
UNROLL = 4

#: Columns the host reference covers on a wide shape; the comparison is windowed to these.
REF_COLS = 512

#: Relative error a candidate must stay under, against the fp64 reference off the
#: dequantised weight. Loose because the kernel accumulates in fp32 over a long K; tight
#: enough that a double-applied or missing scale (a factor of ``scale``, not a rounding)
#: cannot pass.
REL_TOL = 2e-2


def _bench(fn, iters: int = 20, warmup: int = 5) -> float:
    """Time a callable with device events.

    Args:
        fn: Zero-argument callable.
        iters: Timed iterations.
        warmup: Untimed iterations, so the first (allocating, cold-cache) call is not timed.

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


def _legacy_split(n: int, k: int, threads: int) -> int:
    """The pre-change int8 split rule, reproduced here to time it as a labelled control.

    ``pick_split``'s bf16 rule, which the int8 path used to reuse unchanged. Kept in the
    diagnostic rather than in the kernel because the kernel no longer has it: without this
    row the table cannot show how much of the win is the new rule and how much is width.

    Args:
        n: Output columns.
        k: Reduction length.
        threads: Threads per block.

    Returns:
        The split width the old dispatch would have chosen.
    """
    if k < 1024:
        return 1
    by_k = min(8, k // 512)
    if by_k < 2:
        return 1
    want = 4 if n >= 4096 else 8
    return min(want, by_k)


def main(argv: list[str] | None = None) -> int:
    """Sweep width x threads x split on the real shapes and report the winner per shape.

    Args:
        argv: Optional shape filter plus flags. A bare shape name restricts the run, e.g.
            ``["mlp_up"]``, which is the cheap way to retry one shape without paying for
            ``lm_head`` again. ``--with-scalar`` adds ``width = 1`` to the grid and
            ``--with-threads64`` adds 64 threads, both off by default because the scalar
            sweep already covers them.

    Returns:
        ``0`` always: a sweep reports numbers rather than passing or failing, and the job
        is read for its table. A wrong candidate is marked in the row, not fatal, so one
        bad cell does not hide the rest of the grid.
    """
    import torch

    import backends
    import backends.corex as corex
    from backends.corex import kernels, pack
    from int8_weights import QWeight

    args = list(argv or [])
    wanted = {a for a in args if not a.startswith("-")}
    shapes = [s for s in SHAPES if not wanted or s[0] in wanted]
    if not shapes:
        print(f"no shape matched {sorted(wanted)}; known: "
              f"{[s[0] for s in SHAPES]}", flush=True)
        return 1

    widths = WIDTHS + ((1,) if "--with-scalar" in args else ())
    threads_all = THREADS + (THREADS_EXTRA if "--with-threads64" in args else ())
    print(f"  grid: width {list(widths)} x threads {list(threads_all)} x "
          f"split {list(SPLITS)} = {len(widths) * len(threads_all) * len(SPLITS)} "
          f"cells/shape", flush=True)

    print(f"  backend: {backends.status()}", flush=True)
    if not corex.usable():
        print("\nbackend is not usable on this host; nothing to sweep", flush=True)
        return 1
    module = kernels.load_kernels()

    props = torch.cuda.get_device_properties(0)
    sms = getattr(props, "multi_processor_count", 0)
    print(f"  SMs: {sms}", flush=True)

    # The ceiling is measured, never hardcoded: a partition of this card was reported at
    # 572-576 GB/s depending on the session, and a stale constant would make every
    # percentage in the table quietly wrong.
    ceiling_n = 64 * 1024 * 1024
    src = torch.empty(ceiling_n, dtype=torch.uint8, device="cuda")
    dst = torch.empty(ceiling_n, dtype=torch.uint8, device="cuda")
    copy_ms = _bench(lambda: dst.copy_(src), iters=10, warmup=3)
    ceiling = 2 * ceiling_n / (copy_ms * 1e-3) / 1e9
    print(f"  copy ceiling {ceiling:.0f} GB/s", flush=True)
    del src, dst
    torch.cuda.empty_cache()

    best_rows: list[str] = []
    step_rows: list[str] = []
    step_totals: list[tuple[float, float]] = []

    for label, n, k, calls in shapes:
        print(f"\n{'=' * 92}\n{label}: N={n} K={k}  ({n * k / 1e6:.1f} MB int8, "
              f"{calls} calls/step)\n{'=' * 92}", flush=True)
        wt = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.02
        x = torch.randn(k, dtype=torch.bfloat16, device="cuda") * 0.02
        qw = QWeight.quantize(wt)
        packed = pack.to_kn(qw)

        # Reference over a column window, from the dequantised weight: the same numbers the
        # kernel is supposed to reconstruct, so quantisation error is not folded into the
        # comparison and a scale bug cannot hide behind the expected few-percent drift.
        cols = min(REF_COLS, n)
        w_deq = (qw.wq[:cols].float() * qw.scale[:cols, None]).cpu().double()
        ref = w_deq @ x.detach().cpu().double()
        scale_ref = float(ref.abs().max()) or 1.0

        def rel(y) -> float:
            """Relative error of ``y`` against the windowed host reference.

            Args:
                y: Kernel output, ``[N]``.

            Returns:
                ``max|y - ref| / max|ref|`` over the reference window.
            """
            got = y.detach().cpu().double()[:cols]
            return float((got - ref).abs().max()) / scale_ref

        nbytes = n * k
        rows: list[tuple[float, str, float, float, bool, int]] = []
        for width in widths:
            for threads in threads_all:
                for split in SPLITS:
                    tag = f"w{width}/t{threads}/s{split}"
                    try:
                        y = module.gemv_kn_i8(packed.wt, x, packed.scale,
                                              threads, UNROLL, split, width)
                        torch.cuda.synchronize()
                        r = rel(y)
                        ms = _bench(lambda w=width, t=threads, s=split:
                                    module.gemv_kn_i8(packed.wt, x, packed.scale,
                                                      t, UNROLL, s, w))
                        gbs = nbytes / (ms * 1e-3) / 1e9
                        ok = r < REL_TOL
                        rows.append((ms, tag, gbs, r, ok, split))
                        print(f"    {tag:<18} {ms:8.4f} ms {gbs:7.1f} GB/s "
                              f"{100 * gbs / ceiling:>3.0f}%  rel={r:.2e}"
                              f"{'' if ok else '   <-- WRONG'}", flush=True)
                    except Exception as exc:      # noqa: BLE001 - a bad combo is data
                        print(f"    {tag:<18} FAILED {type(exc).__name__}: {exc}",
                              flush=True)

        # Controls: what ships today, and what shipped before the int8 split rule changed.
        # Both go through width=0 so they exercise the auto path rather than a guess at it.
        controls: list[tuple[str, float, float]] = []
        for ctl_label, ctl_split in (("production (all auto)", 0),
                                     ("pre-change dispatch", _legacy_split(n, k, 256))):
            try:
                ms = _bench(lambda s=ctl_split:
                            module.gemv_kn_i8(packed.wt, x, packed.scale, 0, 0, s))
                gbs = nbytes / (ms * 1e-3) / 1e9
                controls.append((ctl_label, ms, gbs))
                print(f"    {ctl_label:<18} {ms:8.4f} ms {gbs:7.1f} GB/s "
                      f"{100 * gbs / ceiling:>3.0f}%  (control)", flush=True)
            except Exception as exc:          # noqa: BLE001
                print(f"    {ctl_label:<18} FAILED {type(exc).__name__}: {exc}", flush=True)

        good = [r for r in rows if r[4]]
        if not good:
            print("\n    NO CORRECT CANDIDATE -- this shape is broken, not just slow",
                  flush=True)
            del wt, x, qw, packed, w_deq, ref
            torch.cuda.empty_cache()
            continue

        good.sort()
        ms, tag, gbs, r, _ok, split = good[0]

        # The partial buffer is written and read once per call: an fp32 [split, N] scratch
        # is 8 * split * N bytes against the K * N the weights cost, i.e. 8 * split / K of
        # the weight traffic. That ratio is why a wide shape prefers a small split while a
        # narrow one prefers a large one, and printing it turns a mysterious regression
        # into arithmetic.
        overhead = 8.0 * split / k
        print(f"\n    best correct : {tag:<18} {ms:8.4f} ms {gbs:7.1f} GB/s "
              f"({100 * gbs / ceiling:.0f}% of ceiling)", flush=True)
        print(f"    partial buffer at that split: {8.0 * split * n / 1e6:.1f} MB/call, "
              f"{100 * overhead:.1f}% of the weight read", flush=True)
        for ctl_label, ctl_ms, ctl_gbs in controls:
            print(f"    vs {ctl_label:<18} {ctl_ms / ms:.2f}x faster  "
                  f"({ctl_ms - ms:+.4f} ms/call, "
                  f"{(ctl_ms - ms) * calls:+.2f} ms/step)", flush=True)

        prod = next((c for c in controls if c[0].startswith("production")), None)
        legacy = next((c for c in controls if c[0].startswith("pre-change")), None)
        best_rows.append(
            f"{label:<13} N={n:<7} K={k:<5} best={tag:<16} {ms:8.4f} ms {gbs:7.1f} GB/s "
            f"{100 * gbs / ceiling:>3.0f}%  rel={r:.1e}")
        prod_ms = prod[1] * calls if prod else float("nan")
        row_total = ms * calls
        step_totals.append((row_total, prod_ms))
        step_rows.append(f"{label:<13} {calls:>3}x {row_total:8.3f} ms/step  "
                         f"(production {prod_ms:7.3f} ms/step)")
        if prod and legacy:
            # Separate the two wins: width versus the int8 split rule. They were argued for
            # independently, so the table should not merge them.
            print(f"    win from the int8 split rule alone: "
                  f"{(legacy[1] - prod[1]) * calls:+.2f} ms/step", flush=True)
            print(f"    win from width on top of it     : "
                  f"{(prod[1] - ms) * calls:+.2f} ms/step", flush=True)
        del wt, x, qw, packed, w_deq, ref
        torch.cuda.empty_cache()

    print(f"\n{'=' * 92}\nbest correct configuration per shape\n{'=' * 92}", flush=True)
    for line in best_rows:
        print(f"  {line}", flush=True)

    if step_rows:
        print(f"\n{'=' * 92}\nprojected cost per decode step at the best configuration\n"
              f"{'=' * 92}", flush=True)
        for line in step_rows:
            print(f"  {line}", flush=True)
        total = sum(row for row, _prod in step_totals)
        prod_total = sum(prod for _row, prod in step_totals)
        print(f"\n  total projections: {total:.2f} ms/step at the best row per shape, "
              f"against {prod_total:.2f} ms/step under the shipped dispatch "
              f"({prod_total - total:+.2f} ms)", flush=True)
        print("  Width and the int8 split rule are separated above per shape; do not read "
              "the total as width's win alone.", flush=True)

    print(f"\n  copy ceiling on this card: {ceiling:.0f} GB/s. A shape below ~60% of it is\n"
          f"  parallelism-bound rather than bandwidth-bound.", flush=True)
    print("\nI8_WIDE_TUNE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
