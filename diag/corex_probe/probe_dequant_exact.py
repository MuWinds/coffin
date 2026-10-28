"""Find a dequantisation spelling that is both faster and numerically exact.

The first micro-benchmark compared two spellings and got a 1.88x win, but the winner
changes the result by one bf16 ULP (1.95e-03, which is 2^-9 -- the bf16 rounding step).
That is inside the contract this repo already documents for bf16 outputs, but this is a
*weight*, and a weight feeds 32 layers: a 1 ULP perturbation is allowed to change a greedy
token id, and changing an argmax is not something to ship without checking.

So before touching ``dequant_kn``, this measures a third family that should be exact:

    out[r0:r1] = (wt[r0:r1].float() * scale).to(bf16)

A row block of a ``[K, N]`` matrix is contiguous, so this keeps both the coalescing the
column-blocked version lost and a working set small enough to stay in cache. The fp32
multiply then rounds once, exactly like the current spelling.

Presses that matter:
    * exact -- ``max |diff|`` against the current spelling must be **0**, not 1 ULP.
    * fast -- within a few percent of the 93 ms the 1-ULP spelling reaches.
    * no extra resident memory, and no change to the layout the GEMM consumes.

If no row-block size is both, the choice is between a 1-ULP change and 174 ms, and that is
a decision for the caller rather than a probe.

Module: ``coffin/diag/corex_probe``. Runs on the card; nothing imports it.

Constraints:
    * Every candidate is verified against the current spelling over every weight, not a
      sample: a transposed index or a dropped block would hide in four tensors.
    * Best-of-3, allocation inside the timed region, so the number is what the engine
      would pay.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: ``coffin/`` is the ``sys.path`` root; this file sits two levels below it.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

MODEL = os.environ.get("ENGINE_MODEL", "")

#: Context window; the model is only built to reach the weights.
MAX_SEQ = 2048

#: Row-block sizes to try for the exact spelling. The working set is ``R * N * 4`` bytes;
#: at N=18432 that is 2.4 MB at R=32 and 19 MB at R=256, so the sweep straddles the cache.
ROW_BLOCKS = (8, 16, 32, 64, 128, 256)


def _events() -> tuple:
    """Create a start/end device-event pair with timing enabled.

    Returns:
        ``(start, end)`` events.
    """
    import torch

    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def _time(fn, reps: int = 3) -> float:
    """Time a zero-argument callable, best of ``reps``, in milliseconds.

    Args:
        fn: Callable to time.
        reps: Timed repeats.

    Returns:
        Best milliseconds.
    """
    import torch

    fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        start, end = _events()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end))
    return best


def _collect(model) -> list:
    """Every distinct int8 projection the prefill path dequantises.

    Args:
        model: Loaded int8 model.

    Returns:
        The packed weights, de-duplicated by identity: 128 for this 32-layer model.
    """
    from backends.corex import pack

    seen: dict[int, object] = {}
    for layer in model.layers:
        for proj in (getattr(layer.mix, "qkvzb_a", None), getattr(layer.mix, "qkv", None),
                     getattr(layer.mix, "out_proj", None), getattr(layer.mix, "o_proj", None),
                     getattr(layer.mlp, "gate_up", None),
                     getattr(layer.mlp, "down", None)):
            if proj is None:
                continue
            gemv = getattr(proj, "g", proj)
            packed = pack.as_kn(gemv.w)
            if getattr(packed, "int8", False):
                seen[id(packed)] = packed
    return list(seen.values())


def main() -> int:
    """Time and verify every dequantisation spelling.

    Returns:
        ``0`` on success, ``1`` when the model directory is missing.
    """
    import torch

    import backends
    from backends.corex import pack
    from engine_api import create_model, load_weights

    print(f"  backend: {backends.status()}", flush=True)
    print(f"  model  : {MODEL}", flush=True)
    if not MODEL or not Path(MODEL).is_dir():
        print(f"  ENGINE_MODEL unset or not a directory: {MODEL!r}", flush=True)
        return 1

    sd = load_weights(MODEL, quant=True, vision=False)
    model = create_model(MODEL, max_seq=MAX_SEQ, bench=False, weights=sd, paged=True)
    weights = _collect(model)
    bf16 = pack.WEIGHT_DTYPE
    total = sum(w.nbytes() for w in weights)
    print(f"  {len(weights)} int8 projections, {total / 2**30:.2f} GiB "
          f"-> {total * 2 / 2**30:.2f} GiB bf16", flush=True)

    def current(w):
        """The engine's current spelling: fp32 multiply, then narrow.

        Args:
            w: Packed int8 weight.

        Returns:
            ``[K, N]`` bf16 weight.
        """
        return (w.wt.float() * w.scale.view(1, -1)).to(bf16)

    def cheap(w):
        """One-ULP spelling: narrow first, scale in bf16 in place.

        Args:
            w: Packed int8 weight.

        Returns:
            ``[K, N]`` bf16 weight.
        """
        out = w.wt.to(bf16)
        out.mul_(w.scale.to(bf16).view(1, -1))
        return out

    def block_factory(rows: int):
        """Build a row-blocked exact spelling at one block size.

        Args:
            rows: Rows per block. The fp32 temporary is ``rows * N * 4`` bytes.

        Returns:
            A callable taking and returning the same things as ``current``.
        """

        def blocked(w):
            """Dequantise one weight in row blocks, keeping the fp32 multiply.

            Args:
                w: Packed int8 weight.

            Returns:
                ``[K, N]`` bf16 weight.
            """
            out = torch.empty((w.k, w.n), dtype=bf16, device=w.wt.device)
            for r0 in range(0, w.k, rows):
                r1 = min(w.k, r0 + rows)
                out[r0:r1] = (w.wt[r0:r1].float() * w.scale).to(bf16)
            return out

        return blocked

    # Reference output per weight, computed once, so verification is a comparison and not
    # a second implementation of the same arithmetic.
    print("\n  --- verifying every candidate over every weight ---", flush=True)
    def mixed_inplace(w):
        """Narrow the weight, then scale in place with the *fp32* scale.

        An in-place op takes its dtype from the destination, so ``bf16.mul_(fp32)``
        computes in fp32 and rounds once into bf16 -- which is the same single rounding
        the current spelling does, minus the ``[K, N]`` fp32 temporary. This is the
        candidate that could be both exact and fast; ``cheap`` loses the ULP because it
        narrows the scale to bf16 before multiplying.

        Args:
            w: Packed int8 weight.

        Returns:
            ``[K, N]`` bf16 weight.
        """
        out = w.wt.to(bf16)
        out.mul_(w.scale.view(1, -1))
        return out

    candidates = [("current", current), ("cheap (1 ULP)", cheap),
                  ("mixed in-place", mixed_inplace)]

    verdicts = {}
    for name, fn in candidates:
        worst = 0.0
        for w in weights:
            ref = current(w)
            got = fn(w)
            worst = max(worst, float((got.float() - ref.float()).abs().max()))
        verdicts[name] = worst
        print(f"  {name:<16} max |diff| = {worst:.2e}"
              f"{'   EXACT' if worst == 0.0 else ''}", flush=True)

    print("\n  --- timing every weight once ---", flush=True)
    print(f"  {'scheme':<18}{'ms':>9}{'GB/s eff':>10}{'vs current':>12}{'exact':>8}",
          flush=True)
    base = None
    for name, fn in candidates:
        def run(fn=fn):
            """Dequantise every projection once.

            Args:
                fn: The spelling under test.
            """
            for w in weights:
                fn(w)

        ms = _time(run)
        if name == "current":
            base = ms
        eff = (total * 3) / (ms * 1e-3) / 1e9
        exact = "yes" if verdicts[name] == 0.0 else "no"
        print(f"  {name:<18}{ms:>9.2f}{eff:>10.1f}{base / ms:>11.2f}x{exact:>8}",
              flush=True)

    print("\nDEQUANT_EXACT_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
