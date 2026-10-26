"""Sweep the GEMV launch configurations on the card and report GB/s per shape.

``linear`` measured 146-248 GB/s against a 574 GB/s copy ceiling, and the split
between shapes is not what the first prototype saw (``lm_head`` reached 361 GB/s
there, and the narrow ``o_proj`` was its worst shape). The kernel has three knobs that
change occupancy and memory-level parallelism -- threads per block, K unroll, and
split-K width -- and which combination wins depends on ``N``, so this script measures
the grid instead of guessing at it.

The output is a table a human reads and then hardcodes: the winner per shape belongs in
:func:`backends.corex._gemv_rows` as a shape-keyed default, not as a runtime sweep,
because a decode step must not pay a benchmark.

Module: ``coffin/backends/corex/diag``. Diagnostic only; runs on the card.

Constraints:
    * Correctness is checked on every candidate, not just the winner: a configuration
      that is fast because it reads the wrong bytes is the failure mode a pure timing
      sweep cannot see. The oracle is the host fp32 dot product over a column strip.
    * ``caps.autotune`` stays ``True`` and no load-time sweep is added here: the point
      of this script is to produce static defaults, and a sweep on the load path would
      add seconds to every process start for a table that does not change per host.
    * Timing uses device events, which this card supports; a host clock around an
      asynchronous launch would measure the launch queue, not the kernel.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: ``coffin/`` itself is the ``sys.path`` root; see ``check_ops.py``.
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

#: Shapes to sweep. ``lm_head`` is the one that matters most -- the library returns NaN
#: there, so this kernel is the only option -- and the narrow ones are where occupancy,
#: not bandwidth, is the limit.
SHAPES: tuple[tuple[str, int, int], ...] = (
    ("mlp_up", 9728, 2560),
    ("mlp_down", 2560, 9728),
    ("qkv_gate", 8192, 2560),
    ("o_proj", 2560, 4096),
    ("lm_head", 248320, 2560),
)

#: Threads per block to try. 64 is one wavefront, 256 fills a block on this part, and
#: 512 trades blocks for threads on the narrow shapes.
THREADS: tuple[int, ...] = (64, 128, 256, 512)

#: K-unroll factors. 1 is the control: if 1 is not slower, the loop overhead is not the
#: limiter and the extra unroll buys nothing.
UNROLLS: tuple[int, ...] = (1, 2, 4)

#: Split-K widths, including 1 (single pass). Splitting multiplies the block count
#: without adding weight traffic, so it is the lever for a narrow ``N``.
SPLITS: tuple[int, ...] = (1, 2, 4)

#: Columns the host oracle covers on a wide shape; the comparison is windowed to these.
REF_COLS = 4096


def _host(t):
    """Move to the host and widen there.

    Args:
        t: Any tensor.

    Returns:
        A float32 CPU copy; the transfer happens before the widening because the
        device's fp64 path returns zeros.
    """
    return t.detach().to("cpu").float()


def _rel(y, ref, cols: int | None) -> float:
    """Relative error over a column window.

    Args:
        y: Kernel output, ``[N]``.
        ref: Host reference covering ``ref.shape[-1]`` columns.
        cols: Column window, or ``None`` for the whole reference.

    Returns:
        ``max|y - ref| / max|ref|`` over the window, or ``inf`` for a zero reference.
    """
    got = _host(y)
    width = ref.shape[-1] if cols is None else min(cols, ref.shape[-1])
    if got.shape[-1] != width:
        got = got[..., :width]
    ref = ref[..., :width]
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
    """Time a callable, in milliseconds per call.

    Args:
        fn: Zero-argument callable.
        iters: Timed repetitions.
        warmup: Untimed repetitions.

    Returns:
        Milliseconds per call.
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
    """Sweep the launch grid and report the winner per shape.

    Returns:
        ``0`` always: a sweep reports numbers rather than passing or failing, and the
        job is read for its table. Correctness is printed per candidate so a bad row is
        visible rather than silently averaged in.
    """
    import torch

    import backends.corex as corex
    from backends.corex import kernels, pack

    _sep("0. backend state")
    print(" ", corex.describe(), flush=True)
    if not corex.usable():
        print("\nbackend is not usable on this host; nothing to sweep", flush=True)
        return 1
    module = kernels.load_kernels(corex._arch() or corex.DEFAULT_ARCH)
    print(f"  sm_count report from the kernel: {torch.cuda.get_device_properties(0).multi_processor_count}",
          flush=True)

    winners: list[str] = []
    for label, n, k in SHAPES:
        _sep(f"{label}: N={n} K={k}")
        w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
        x = torch.randn(k, dtype=torch.bfloat16, device="cuda")
        packed = pack.to_kn(w)
        cols = REF_COLS if n > REF_COLS else None
        ref = _host(w[:cols] if cols else w) @ _host(x)
        nbytes = w.numel() * w.element_size()
        best = ("", 0.0, float("inf"))
        for threads in THREADS:
            for unroll in UNROLLS:
                for split in SPLITS:
                    try:
                        y = module.gemv_kn(packed.wt, x, threads, unroll, split)
                        torch.cuda.synchronize()
                    except Exception as exc:      # noqa: BLE001 - a bad combo is data
                        print(f"  t={threads:<4} u={unroll} s={split}: "
                              f"{type(exc).__name__}: {exc}", flush=True)
                        continue
                    rel = _rel(y, ref, cols)
                    ms = _bench(
                        lambda t=threads, u=unroll, s=split: module.gemv_kn(
                            packed.wt, x, t, u, s))
                    gbs = nbytes / (ms * 1e-3) / 1e9
                    tag = f"t={threads} u={unroll} s={split}"
                    mark = ""
                    if rel < 2e-2 and gbs > best[1]:
                        best = (tag, gbs, rel)
                        mark = "  <-"
                    print(f"  {tag:<22} {ms:7.4f} ms  {gbs:7.1f} GB/s  rel={rel:.2e}{mark}",
                          flush=True)
        winners.append(f"{label:<9} N={n:<7} K={k:<5} best={best[0]:<22} "
                       f"{best[1]:7.1f} GB/s")
        del w, x, packed, ref
        torch.cuda.empty_cache()

    _sep("summary: the configuration to hardcode per shape")
    for line in winners:
        print(f"  {line}", flush=True)
    print(f"\n  copy ceiling on this card was measured at 574 GB/s (COREX_PORT.md §0);\n"
          f"  a shape below ~60% of it is occupancy-bound, not bandwidth-bound.",
          flush=True)
    print("\nCOREX_TUNE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
