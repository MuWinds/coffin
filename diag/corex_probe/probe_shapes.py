"""What projection shapes does a real decode step actually call?

Every tuning decision so far has been made against a hand-written shape table in the probes
(``mlp_up`` 9216x2560, ``qkv_gate`` 8192x2560, and so on). Those numbers came from reading the
config, not from watching the engine, and a fused projection is not the same shape as its
parts: ``gate_up`` packs two 9216-row weights into one matrix, and the linear-attention layers
fuse four projections into one. A dispatch table keyed on the wrong shapes would tune kernels
that never run.

This records the real (N, K) of every ``gemv_kn_i8`` call in one decode step, by wrapping the
transport entry point the backend uses. It reports call count per shape and the bytes each
shape reads, so the tuning effort can be ordered by what a step actually pays for.

Module: ``coffin/diag/corex_probe``. Diagnostic only; runs on the card.

It lives here rather than under ``backends/corex/diag`` because it builds a model, and the
``backends`` package must not import an engine layer.

Constraints:
    * Eager path only -- there is no graph path on this backend (``_GRAPH_VERIFIED`` is
      ``False``), and CUDA graphs must stay off on this driver regardless.
    * The wrapper is installed after the extension is loaded and calls the original, so the
      numbers are measured through the same code path the engine uses.
    * The model is built from ``load_weights(quant=True)``, which is what
      ``serving/runtime_service.py`` does. ``create_model`` alone loads dense weights and
      would tally the bf16 kernel, reporting nothing about the int8 workload.
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path

#: ``coffin/`` itself is the ``sys.path`` root; this file is two levels down from it.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

MODEL = os.environ.get("ENGINE_MODEL", "")


def main() -> int:
    """Tally the projection shapes of one decode step.

    Returns:
        ``0`` on success, ``1`` when the model directory is missing.
    """
    import torch

    import backends
    import backends.corex as corex
    from backends.corex import kernels

    print(f"  backend: {backends.status()}", flush=True)
    if not MODEL or not Path(MODEL).is_dir():
        print(f"  ENGINE_MODEL unset or not a directory: {MODEL!r}", flush=True)
        return 1

    module = kernels.load_kernels()
    original = module.gemv_kn_i8
    original_dense = module.gemv_kn

    #: ``(N, K) -> [calls, bytes, elapsed_ms]``, filled by the wrappers below.
    tally: dict[tuple[int, int], list] = defaultdict(lambda: [0, 0, 0.0])
    #: Calls that went to the dense kernel instead of the int8 one. Non-zero here means the
    #: step was not on quantised weights, which would make an int8 tuning table meaningless.
    dense_calls = [0]

    def wrapped(wt, x, scale, *args, **kwargs):
        """Record the shape, then run the real kernel.

        Args:
            wt: ``[K, N]`` int8 weight.
            x: ``[K]`` activations.
            scale: ``[N]`` per-column scales.
            *args: Threads/unroll/split overrides, forwarded unchanged.
            **kwargs: Same.

        Returns:
            Whatever the kernel returned.
        """
        k, n = int(wt.shape[0]), int(wt.shape[1])
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = original(wt, x, scale, *args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        slot = tally[(n, k)]
        slot[0] += 1
        slot[1] += n * k
        slot[2] += start.elapsed_time(end)
        return out

    def wrapped_dense(wt, x, *args, **kwargs):
        """Count a dense-path call so a mis-routed probe is visible rather than silent.

        Args:
            wt: ``[K, N]`` weight.
            x: ``[K]`` activations.
            *args: Forwarded unchanged.
            **kwargs: Forwarded unchanged.

        Returns:
            Whatever the kernel returned.
        """
        dense_calls[0] += 1
        return original_dense(wt, x, *args, **kwargs)

    module.gemv_kn_i8 = wrapped
    module.gemv_kn = wrapped_dense
    try:
        from engine_api import create_model, load_weights

        # The serving entry point loads with quant=True by default (serving/runtime_service.py,
        # int8=True), and that is the path the tuning is for. Building the model without it
        # runs the bf16 GEMV, which is a different kernel with different dispatch -- a shape
        # probe on that path reports nothing useful about the int8 workload.
        sd = load_weights(MODEL, quant=True, vision=False)
        model = create_model(MODEL, max_seq=1024, bench=False, weights=sd)
        model.reset()
        logits = model.prefill([1, 2, 3, 4, 5])
        torch.cuda.synchronize()

        print("\n=== one decode step, projection shapes as actually called ===", flush=True)
        logits = model.logits(int(logits.argmax()))
        torch.cuda.synchronize()

        if not tally:
            print(f"  no gemv_kn_i8 calls; dense gemv_kn calls = {dense_calls[0]}", flush=True)
            print("  the step is NOT on quantised weights -- an int8 table would be "
                  "meaningless", flush=True)
            return 1

        # The instrumentation costs two events per call; a second step without it is what
        # the per-shape timings are compared against.
        print(f"\n  {'N':>7} {'K':>6} {'calls':>6} {'MB':>8} {'ms':>8} {'GB/s':>8}",
              flush=True)
        total_bytes = 0
        total_ms = 0.0
        for (n, k), (calls, nbytes, ms) in sorted(tally.items(),
                                                  key=lambda kv: -kv[1][1]):
            gbs = nbytes / (ms * 1e-3) / 1e9 if ms > 0 else 0.0
            print(f"  {n:>7} {k:>6} {calls:>6} {nbytes / 1e6:>8.1f} {ms:>8.3f} "
                  f"{gbs:>8.1f}", flush=True)
            total_bytes += nbytes
            total_ms += ms
        print(f"  {'total':>7} {'':>6} {sum(v[0] for v in tally.values()):>6} "
              f"{total_bytes / 1e6:>8.1f} {total_ms:>8.3f} "
              f"{(total_bytes / (total_ms * 1e-3) / 1e9) if total_ms > 0 else 0.0:>8.1f}",
              flush=True)

        print("\n=== distinct shapes, for the dispatch table ===", flush=True)
        for (n, k) in sorted(tally, key=lambda nk: -tally[nk][1]):
            calls, nbytes, _ms = tally[(n, k)]
            print(f"  N={n:<7} K={k:<6} calls={calls:<5} "
                  f"{nbytes / 1e6:>7.1f} MB/step", flush=True)
    finally:
        module.gemv_kn_i8 = original
        module.gemv_kn = original_dense

    print("\nSHAPE_TALLY_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
