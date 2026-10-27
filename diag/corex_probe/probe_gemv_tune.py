"""Sweep the GEMV host knobs on the model's real shapes, with correctness checked.

The kernel exposes three host-side choices -- ``threads``, ``unroll`` and ``split`` --
and the dispatch picks them from the SM count. The profiler says that dispatch is wrong
for this part: every one of the 64 projections took the split path (64 calls,
11.46 ms), which is more than the non-split kernel costs for the whole step.

The dispatch formula's threshold is ``blocks >= 2 * sm_count``. With 16 SMs, a 2560-wide
projection at 256 threads is 10 blocks, well under 32, so it splits. Whether that is
right is a measurement, not an argument -- and the answer changes with N, K and the
card's SM count, so the table is what the dispatch should be built from.

Each configuration is verified against a host fp64 reference before its time is
reported: a fast wrong answer must not win a slot.

Module: ``coffin/diag``. Runs on the card.

Constraints:
    * Host reference, not device: this card's fp64 returns zeros (COREX_PORT.md
      §0), so widening on the device would compare against nothing.
    * One shape in flight, freed before the next: ``lm_head`` is 2.5 GB of host
      reference on its own and the node is shared.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

#: Configurations to try per shape. ``split=0`` means "let the kernel decide"; the
#: explicit 1 is the no-split path, which is the one the dispatch is suspected of
#: skipping too often.
CONFIGS = (
    ("auto", 0, 0, 0),
    ("t256/u4/s1", 256, 4, 1),
    ("t256/u2/s1", 256, 2, 1),
    ("t128/u4/s1", 128, 4, 1),
    ("t256/u4/s2", 256, 4, 2),
    ("t256/u4/s4", 256, 4, 4),
    ("t512/u4/s1", 512, 4, 1),
)


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
    from backends.corex import kernels
    from backends.corex import pack

    print(f"  backend: {backends.status()}", flush=True)
    # load_kernels is the module's only public entry point; the compiled extension
    # itself stays private so no caller can hold it across a reset.
    mod = kernels.load_kernels()

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

    shapes = (
        ("mlp_up", 9216, 2560),
        ("mlp_down", 2560, 9216),
        ("qkv_gate", 8192, 2560),
        ("o_proj", 2560, 4096),
    )

    best = {}
    for label, n, k in shapes:
        wt = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
        x = torch.randn(k, dtype=torch.bfloat16, device="cuda")
        packed = pack.as_kn(wt)
        ref = (packed.wt.detach().to("cpu").double()
               * x.detach().to("cpu").double().unsqueeze(1)).sum(0)
        scale = float(ref.abs().max())
        nbytes = n * k * 2

        def rel(y) -> float:
            """Relative error of ``y`` against the host reference.

            Args:
                y: Kernel output.

            Returns:
                ``max|y - ref| / max|ref|``.
            """
            return float((y.detach().to("cpu").double() - ref).abs().max()) / scale

        print(f"\n  === {label}  N={n} K={k} "
              f"({nbytes / 1e6:.1f} MB of weights) ===", flush=True)
        rows = []
        for name, threads, unroll, split in CONFIGS:
            try:
                y = mod.gemv_kn(packed.wt, x, threads, unroll, split)
                torch.cuda.synchronize()
                r = rel(y)
                ms = _bench(lambda t=threads, u=unroll, s=split:
                            mod.gemv_kn(packed.wt, x, t, u, s))
                gbs = nbytes / (ms * 1e-3) / 1e9
                ok = r < 5e-3
                rows.append((ms, name, gbs, r, ok))
                print(f"    {name:<12} {ms:>7.3f} ms {gbs:>7.1f} GB/s "
                      f"{100 * gbs / ceiling:>3.0f}%  rel={r:.2e} "
                      f"{'' if ok else '<-- WRONG'}", flush=True)
            except Exception as exc:              # noqa: BLE001
                print(f"    {name:<12} FAILED {type(exc).__name__}: {exc}", flush=True)
        good = [r for r in rows if r[4]]
        if good:
            good.sort()
            best[label] = good[0]
            print(f"    best (correct) : {good[0][1]} at {good[0][0]:.3f} ms "
                  f"({good[0][2]:.1f} GB/s)", flush=True)
        del wt, x, packed, ref
        torch.cuda.empty_cache()

    print("\n=== recommended dispatch per shape ===", flush=True)
    for label, (ms, name, gbs, r, _ok) in sorted(best.items()):
        print(f"  {label:<10} {name:<12} {ms:>7.3f} ms {gbs:>7.1f} GB/s", flush=True)
    auto_total = 0.0
    print("\n  (compare against 'auto' in the tables above to see what the current")
    print("   SM-count heuristic costs)")
    print("\nGEMV_TUNE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
