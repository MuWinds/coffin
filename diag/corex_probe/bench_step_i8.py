"""Measure one decode step on the int8 weights the serving path actually uses, and print
the greedy continuation.

``probe_decode_profile.py`` builds its model with ``create_model``, which loads dense
weights: ``load_weights`` defaults to ``quant=False``, so that probe measures the **bf16**
GEMV. The serving path is ``serving/runtime_service.py`` with ``int8=True`` by default, and
that is the kernel a dispatch change has to be judged on. Measuring the dense path here
would report a speedup that never reaches a served token.

The greedy continuation is printed as token ids rather than text: the check that matters
after a dispatch change is that the sequence is unchanged, and ids compare without a
tokenizer rounding difference hiding a real divergence. It is compared against the same
run on unmodified dispatch, not against a quoted string.

Module: ``coffin/diag/corex_probe``. Runs on the card.

It lives here rather than under ``backends/corex/diag`` because it drives the engine: the
``backends`` package must not import an engine layer, and a probe that builds a model does
exactly that. The kernels' own diagnostics stay next to the kernels; anything that needs
``engine_api`` sits here.

Constraints:
    * Eager path only. CUDA graphs are disabled on this backend (``_GRAPH_VERIFIED`` is
      ``False``) and must stay off on this driver regardless.
    * Timed with device events around a batch of steps, so per-step launch overhead is
      included the way the engine pays it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: ``coffin/`` itself is the ``sys.path`` root; this file is two levels down from it.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

MODEL = os.environ.get("ENGINE_MODEL", "")

#: Tokens to generate greedily for the equality check.
GREEDY_TOKENS = 6

#: Decode steps to time after warmup.
STEPS = 20


def _bench(fn, iters: int, warmup: int) -> float:
    """Time a callable with device events.

    Args:
        fn: Zero-argument callable.
        iters: Timed iterations.
        warmup: Untimed iterations, so a lazy first launch is not measured.

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
    """Time an int8 decode step and print the greedy continuation.

    Returns:
        ``0`` on success, ``1`` when the model directory is not set.
    """
    import torch

    import backends
    from engine_api import create_model, load_weights

    print(f"  backend: {backends.status()}", flush=True)
    if not MODEL or not Path(MODEL).is_dir():
        print(f"  ENGINE_MODEL unset or not a directory: {MODEL!r}", flush=True)
        return 1

    # quant=True is what serving/runtime_service.py does; see the module docstring.
    sd = load_weights(MODEL, quant=True, vision=False)
    model = create_model(MODEL, max_seq=1024, bench=False, weights=sd)

    print("\n=== greedy continuation (compare these ids against the pre-change run) ===",
          flush=True)
    model.reset()
    logits = model.prefill([1, 2, 3, 4, 5])
    torch.cuda.synchronize()
    ids = []
    probe = logits
    for _ in range(GREEDY_TOKENS):
        nxt = int(probe.argmax())
        ids.append(nxt)
        probe = model.logits(nxt)
        torch.cuda.synchronize()
    print(f"  ids: {ids}", flush=True)

    print("\n=== one eager int8 decode step ===", flush=True)
    model.reset()
    logits = model.prefill([1, 2, 3, 4, 5])
    torch.cuda.synchronize()

    state = [int(logits.argmax())]

    def step() -> None:
        """One decode step, kept as a named callable for the harness."""
        state[0] = int(model.logits(state[0]).argmax())

    step()
    torch.cuda.synchronize()
    model.reset()
    logits = model.prefill([1, 2, 3, 4, 5])
    torch.cuda.synchronize()
    state[0] = int(logits.argmax())
    ms = _bench(step, STEPS, 5)
    print(f"  decode step: {ms:.2f} ms  ({1000 / ms:.2f} tok/s)", flush=True)
    print(f"  WEIGHT_PATH=int8  step_ms={ms:.3f}  tok_s={1000 / ms:.2f}", flush=True)
    print("\nSTEP_INT8_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
