"""Which shapes the split-K GEMV path is hurting, and what the plain kernel gets instead.

The profiler showed two GEMV kernels in one decode step: ``gemv_kn_kernel`` (65 calls,
24.17 ms) and ``gemv_kn_split_kernel`` (64 calls, 11.46 ms). 64 split calls for 64
projections means *every* projection takes the split path, and at 179 us each it costs
almost half of what the non-split kernel costs for the whole step. If the plain kernel
is faster at these shapes, the split threshold is set wrong for this card.

This measures both kernels per shape, with correctness against a host reference so a
"faster" wrong answer cannot win. The split path exists to raise occupancy when N is
small; whether it still helps at N = 2560..9216 on a 16-SM part is exactly the question.

Module: ``coffin/diag``. Runs on the card.

Constraints:
    * Correctness first on every shape: a wrong-but-fast kernel is the one failure mode
      a bandwidth table hides.
    * The reference is host fp64 arithmetic, because this card's device fp64 returns
      zeros (COREX_PORT.md §0).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _bench(fn, iters: int = 20, warmup: int = 5) -> float:
    """Time a callable with device events, in milliseconds per call.

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
    import torch

    import backends
    from backends.corex import kernels
    from backends.corex import pack

    print(f"  backend: {backends.status()}", flush=True)

    ceiling_n = 64 * 1024 * 1024
    src = torch.empty(ceiling_n, dtype=torch.uint8, device="cuda")
    dst = torch.empty(ceiling_n, dtype=torch.uint8, device="cuda")
    copy_ms = _bench(lambda: dst.copy_(src), iters=10, warmup=3)
    ceiling = 2 * ceiling_n / (copy_ms * 1e-3) / 1e9
    print(f"  copy ceiling {ceiling:.0f} GB/s", flush=True)
    del src, dst
    torch.cuda.empty_cache()

    shapes = (
        ("mlp_up", 9216, 2560),
        ("mlp_down", 2560, 9216),
        ("qkv_gate", 8192, 2560),
        ("o_proj", 2560, 4096),
        ("lm_head", 248320, 2560),
    )

    # What the backend exposes for the split threshold, and the kernel entry points it
    # actually has -- printed rather than assumed, so the tuning table is readable next
    # to the numbers it should change.
    # load_kernels is the module's only public entry point; the compiled extension
    # itself stays private so no caller can hold it across a reset.
    mod = kernels.load_kernels()
    print("\n  extension entry points (gemv-related):", flush=True)
    for name in sorted(dir(mod)):
        if "gemv" in name or "split" in name:
            print(f"    {name}", flush=True)

    print(f"\n  {'shape':<10} {'N':>7} {'K':>6} | {'plain ms':>9} {'GB/s':>7} "
          f"{'%c':>4} | {'split ms':>9} {'GB/s':>7} {'%c':>4} | rel(plain) rel(split)",
          flush=True)
    for label, n, k in shapes:
        wt = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
        x = torch.randn(k, dtype=torch.bfloat16, device="cuda")
        packed = pack.as_kn(wt)

        # Host reference: this card's fp64 is unusable, so widen after the transfer.
        ref = (packed.wt.detach().to("cpu").double()
               * x.detach().to("cpu").double().unsqueeze(1)).sum(0)

        def rel(y) -> float:
            """Relative error of ``y`` against the host reference.

            Args:
                y: Kernel output.

            Returns:
                ``max|y - ref| / max|ref|``.
            """
            scale = float(ref.abs().max())
            err = float((y.detach().to("cpu").double() - ref).abs().max())
            return err / scale if scale else float("inf")

        # The public op, which is what the model actually calls.
        y_pub = backends.ops.linear(wt, x)
        ms_pub = _bench(lambda: backends.ops.linear(wt, x))
        nbytes = n * k * 2

        line = (f"  {label:<10} {n:>7} {k:>6} | {ms_pub:>9.3f} "
                f"{nbytes / (ms_pub * 1e-3) / 1e9:>7.1f} "
                f"{100 * nbytes / (ms_pub * 1e-3) / 1e9 / ceiling:>3.0f}%")

        # Try the non-split entry point directly, if the extension exposes one with a
        # signature this uses. Guarded: the point is the comparison, and a missing or
        # differently-shaped entry must not abort the table.
        ms_split, rel_split = float("nan"), float("nan")
        for cand in ("gemv_kn_nosplit", "gemv_kn_plain", "gemv_kn_direct"):
            fn = getattr(mod, cand, None)
            if fn is None:
                continue
            try:
                y = fn(packed.wt, x)
                torch.cuda.synchronize()
                ms_split = _bench(lambda f=fn: f(packed.wt, x))
                rel_split = rel(y)
            except Exception as exc:              # noqa: BLE001
                print(f"      {cand}: {type(exc).__name__}: {exc}", flush=True)
            break

        line += (f" | {ms_split:>9.3f} "
                 f"{nbytes / (ms_split * 1e-3) / 1e9 if ms_split == ms_split else float('nan'):>7.1f}"
                 f" | {rel(y_pub):>8.2e} {rel_split:>8.2e}")
        print(line, flush=True)
        del wt, x, packed, ref
        torch.cuda.empty_cache()

    print("\nGEMV_SPLIT_PROBE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
