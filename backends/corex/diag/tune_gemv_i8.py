"""Sweep the W8A16 GEMV launch grid on the model's real shapes, with correctness checked.

The int8 path is 50% of a decode step and runs at 130-262 GB/s against a ~575 GB/s copy
ceiling. The dispatch it uses is the bf16 rule, reused unchanged: ``pick_split`` caps at 8
and ``pick_threads`` returns 256 for anything at or above 2048 columns. Neither was
measured for int8, and the two shapes that hurt most -- ``mlp_down`` and ``o_proj``, both
2560 columns -- reach the lowest bandwidth of any shape, which points at parallelism
rather than at bytes.

So the knobs are swept instead of reasoned about. The one that the bf16 table says matters
is ``split``: on a 16-SM part, 2560 columns at 256 threads is 10 blocks, under one wave,
and splitting multiplies the block count without reading one extra weight byte.

Every candidate is checked against a host fp64 reference built from the *dequantised*
weight, so a configuration that is fast because it applies the scale twice cannot win.

Module: ``coffin/backends/corex/diag``. Diagnostic only; runs on the card.

Constraints:
    * Host fp64 reference: this card's device fp64 returns zeros (COREX_PORT.md §0), so
      widening on the device compares against nothing.
    * The reference is windowed to the first ``REF_COLS`` columns, and the timing is not:
      ``lm_head`` is 635 MB int8 and a full host reference would cost more than the sweep.
    * ``caps.autotune`` stays ``True`` and nothing here runs at load: the output is a
      table to hardcode, because a decode step must not pay for a benchmark.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: ``coffin/`` itself is the ``sys.path`` root; see ``check_ops.py``.
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

#: The model's projection geometries, read from its config: hidden 2560, intermediate
#: 9216, fused qkv+gate 8192, ``o_proj`` K=4096, vocabulary 248320.
SHAPES: tuple[tuple[str, int, int], ...] = (
    ("mlp_up", 9216, 2560),
    ("mlp_down", 2560, 9216),
    ("qkv_gate", 8192, 2560),
    ("o_proj", 2560, 4096),
    ("lm_head", 248320, 2560),
)

THREADS: tuple[int, ...] = (64, 128, 256, 512)
UNROLLS: tuple[int, ...] = (2, 4)
#: Split-K widths. Past 8 is the question: the current dispatch never exceeds it, and the
#: narrow shapes are the ones short of blocks. 16 and 32 are tried because the cost of a
#: split is one more slice of a small partial buffer, not another read of the weight.
SPLITS: tuple[int, ...] = (1, 2, 4, 8, 16, 32)

#: Columns the host reference covers on a wide shape; the comparison is windowed to these.
REF_COLS = 512


def _bench(fn, iters: int = 20, warmup: int = 5) -> float:
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


def main() -> int:
    """Sweep the int8 launch grid and report the winner per shape.

    Returns:
        ``0`` always: a sweep reports numbers rather than passing or failing, and the
        job is read for its table. Correctness is printed per candidate so a bad row is
        visible instead of being averaged in.
    """
    import torch

    import backends
    import backends.corex as corex
    from backends.corex import kernels, pack
    from int8_weights import QWeight

    print(f"  backend: {backends.status()}", flush=True)
    if not corex.usable():
        print("\nbackend is not usable on this host; nothing to sweep", flush=True)
        return 1
    module = kernels.load_kernels()

    props = torch.cuda.get_device_properties(0)
    sms = getattr(props, "multi_processor_count", 0)
    print(f"  SMs: {sms}", flush=True)

    ceiling_n = 64 * 1024 * 1024
    src = torch.empty(ceiling_n, dtype=torch.uint8, device="cuda")
    dst = torch.empty(ceiling_n, dtype=torch.uint8, device="cuda")
    copy_ms = _bench(lambda: dst.copy_(src), iters=10, warmup=3)
    ceiling = 2 * ceiling_n / (copy_ms * 1e-3) / 1e9
    print(f"  copy ceiling {ceiling:.0f} GB/s", flush=True)
    del src, dst
    torch.cuda.empty_cache()

    summary: list[str] = []
    for label, n, k in SHAPES:
        print(f"\n{'=' * 78}\n{label}: N={n} K={k}  ({n * k / 1e6:.1f} MB int8)\n{'=' * 78}",
              flush=True)
        wt = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.02
        x = torch.randn(k, dtype=torch.bfloat16, device="cuda") * 0.02
        qw = QWeight.quantize(wt)
        packed = pack.to_kn(qw)

        # Reference over a column window, from the dequantised weight: the same numbers
        # the kernel is supposed to reconstruct, so quantisation error is not folded into
        # the comparison and a scale bug cannot hide behind the expected few-percent drift.
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
        rows = []
        for threads in THREADS:
            for unroll in UNROLLS:
                for split in SPLITS:
                    tag = f"t{threads}/u{unroll}/s{split}"
                    try:
                        y = module.gemv_kn_i8(packed.wt, x, packed.scale,
                                              threads, unroll, split)
                        torch.cuda.synchronize()
                        r = rel(y)
                        ms = _bench(lambda t=threads, u=unroll, s=split:
                                    module.gemv_kn_i8(packed.wt, x, packed.scale, t, u, s))
                        gbs = nbytes / (ms * 1e-3) / 1e9
                        ok = r < 2e-2
                        rows.append((ms, tag, gbs, r, ok))
                        print(f"    {tag:<18} {ms:7.4f} ms {gbs:7.1f} GB/s "
                              f"{100 * gbs / ceiling:>3.0f}%  rel={r:.2e}"
                              f"{'' if ok else '   <-- WRONG'}", flush=True)
                    except Exception as exc:      # noqa: BLE001 - a bad combo is data
                        print(f"    {tag:<18} FAILED {type(exc).__name__}: {exc}",
                              flush=True)
        good = [r for r in rows if r[4]]
        if good:
            good.sort()
            ms, tag, gbs, r, _ok = good[0]
            inc = next((q for q in rows if q[1] == "t256/u4/s8"), None)
            print(f"\n    best correct : {tag:<18} {ms:7.4f} ms {gbs:7.1f} GB/s", flush=True)
            if inc:
                print(f"    current pick : t256/u4/s8        {inc[0]:7.4f} ms "
                      f"{inc[2]:7.1f} GB/s  -> {inc[0] / ms:.2f}x available", flush=True)
            summary.append(f"{label:<9} N={n:<7} K={k:<5} best={tag:<16} "
                           f"{ms:7.4f} ms {gbs:7.1f} GB/s "
                           f"({inc[0] / ms:.2f}x vs current)" if inc else
                           f"{label:<9} N={n:<7} K={k:<5} best={tag:<16} "
                           f"{ms:7.4f} ms {gbs:7.1f} GB/s")
        del wt, x, qw, packed, w_deq, ref
        torch.cuda.empty_cache()

    print(f"\n{'=' * 78}\nthe configuration to hardcode per shape\n{'=' * 78}", flush=True)
    for line in summary:
        print(f"  {line}", flush=True)
    print(f"\n  copy ceiling on this card: {ceiling:.0f} GB/s; a shape below ~60% of it is\n"
          f"  parallelism-bound, not bandwidth-bound", flush=True)
    print("\nINT8_TUNE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
