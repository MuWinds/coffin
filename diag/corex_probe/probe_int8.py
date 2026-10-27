"""Is the W8A16 GEMV correct, and what does halving the weight bytes actually buy?

The int8 path touches three things that each fail quietly if wrong: the quantised storage
form, the ``[K, N]`` transpose of an int8 matrix, and the per-column scale that must be
applied once after the ``K`` sum rather than per element. A bandwidth table alone would
hide all three, so every shape here is checked against a host fp64 reference computed from
the *dequantised* weight -- the same tensor the kernel is supposed to reconstruct.

Reports bandwidth beside the bf16 kernel at the same shape, because the whole reason for
int8 is the byte count and the ratio is what says whether the kernel is reaching the
traffic reduction or wasting it.

Module: ``coffin/diag``. Runs on the card.

Constraints:
    * Host fp64 reference: this card's device fp64 returns zeros (``COREX_PORT.md`` §0).
    * One shape at a time, freed before the next -- ``lm_head`` alone is 635 MB int8 and
      twice that transient during the transpose.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


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
    import torch

    import backends
    from int8_weights import QWeight

    print(f"  backend: {backends.status()}", flush=True)
    table = backends.ops

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

    print(f"\n  {'shape':<10} {'N':>7} {'K':>6} | {'int8 ms':>8} {'GB/s':>7} {'%c':>4} "
          f"{'rel':>9} | {'bf16 ms':>8} {'GB/s':>7} | speedup", flush=True)
    worst = 0.0
    for label, n, k in shapes:
        wt = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.02
        x = torch.randn(k, dtype=torch.bfloat16, device="cuda") * 0.02

        # The reference is built from the dequantised weight, so it tests the kernel's
        # arithmetic against the same numbers the quantiser produced -- a reference on the
        # original bf16 weight would fold quantisation error into the comparison and hide
        # a scale bug behind an "expected" few-percent difference.
        qw = QWeight.quantize(wt)
        w_deq = qw.dequant(torch.float32).cpu()
        ref = (w_deq.double() @ x.detach().cpu().double()).float()
        scale_ref = float(ref.abs().max()) or 1.0

        y8 = table.linear(qw, x)
        torch.cuda.synchronize()
        rel = float((y8.detach().cpu().float() - ref).abs().max()) / scale_ref
        worst = max(worst, rel)
        ms8 = _bench(lambda: table.linear(qw, x))
        bytes8 = n * k

        yb = table.linear(wt, x)
        torch.cuda.synchronize()
        msb = _bench(lambda: table.linear(wt, x))
        bytesb = n * k * 2

        print(f"  {label:<10} {n:>7} {k:>6} | {ms8:>8.3f} "
              f"{bytes8 / (ms8 * 1e-3) / 1e9:>7.1f} "
              f"{100 * bytes8 / (ms8 * 1e-3) / 1e9 / ceiling:>3.0f}% "
              f"{rel:>9.2e} | {msb:>8.3f} "
              f"{bytesb / (msb * 1e-3) / 1e9:>7.1f} | {msb / ms8:>5.2f}x", flush=True)

        del wt, x, qw, w_deq, ref, y8, yb
        torch.cuda.empty_cache()

    print(f"\n  worst relative error: {worst:.2e}", flush=True)
    # bf16 weights already carry ~2.5e-3 of quantisation error against fp64, and int8 adds
    # its own; anything an order above that means the scale or the layout is wrong.
    verdict = "OK" if worst < 2e-2 else "TOO LARGE -- suspect the scale or the layout"
    print(f"  verdict: {verdict}", flush=True)
    print("\nINT8_PROBE_DONE", flush=True)
    return 0 if worst < 2e-2 else 1


if __name__ == "__main__":
    sys.exit(main())
