"""Validate the three int8 prefill fixes end to end, before any of them is written.

The scheme micro-benchmark settled *how* to dequantise faster (174.7 -> 93.0 ms over all
128 projections, 1.88x, no extra memory). It did not settle *how often*, and that is the
larger question on a long prompt: ``Qwen35.prefill`` is chunk-major --

    for c0 in range(0, n, chunk):        # chunks
        for layer in self.layers:        # layers

-- so every weight is dequantised once *per chunk*. At n=8192 with chunk=2048 that is
four passes over 3.33 GiB instead of one. The chunk sweep at n=8192 already hinted at it
(4573 ms at 4 chunks vs 4108 at 1), but that comparison also changes the GEMM's M and the
attention tiling, so it cannot be attributed to dequantisation.

This isolates it. Three configurations, each measured end to end on the real prefill path:

1. **as-is** -- the current spelling, one dequant per chunk.
2. **cheap spelling** -- ``pack.KnQWeight.dequant_kn`` monkeypatched to cast straight to
   bf16 and scale in place. This is the exact one-function change the micro-benchmark
   recommends, so the end-to-end number is the number that change would deliver.
3. **hoisted** -- every weight pre-dequantised once, so the prefill pays no dequantisation
   at all. That is the upper bound, and the +6.65 GiB it costs is what decides whether it
   is worth having.

Configurations 1 and 2 are run at two prompts (2048 and 8192) and, for the long prompt,
two chunk sizes -- because if the remaining gap is "per chunk", then the fix is either a
cheaper per-call (2) or fewer calls (bigger chunk / layer-major loop), and the numbers
have to say which.

Module: ``coffin/diag/corex_probe``. Runs on the card; nothing imports it.

Constraints:
    * The monkeypatch is installed and removed around the timed region, and verified
      against the original output: a cheaper-but-wrong dequantisation must not win.
    * ``mtp_prefill`` stays off throughout -- it is worth 1.29x (PERF_COREX.md §12.2a) and
      leaving it on would bury the effect being measured.
    * The hoisted run adds a resident bf16 copy on top of the int8 one (3.33 + 6.66 GiB),
      so it runs last and can be dropped.
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

#: Context window; 16k so the long prompt fits.
MAX_SEQ = 16384

#: Tokens per block in the paged pool.
BLOCK = 64

#: Prompts: one chunk at the default size, and four chunks of it.
CASES = ((2048, 2048), (8192, 2048), (8192, 8192))


def _events() -> tuple:
    """Create a start/end device-event pair with timing enabled.

    Returns:
        ``(start, end)`` events.
    """
    import torch

    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def _tokens(n: int) -> list[int]:
    """Build a deterministic prompt of ``n`` legal token ids.

    Args:
        n: Prompt length.

    Returns:
        ``n`` ids from the arithmetic sequence the other probes use.
    """
    return [(i * 7919 + 13) % 100000 for i in range(n)]


def _time_prefill(model, toks: list[int], chunk: int, reps: int = 2) -> float:
    """Time one prefill, best of ``reps``, in milliseconds.

    Args:
        model: Loaded model.
        toks: Prompt token ids.
        chunk: Tokens per GEMM pass.
        reps: Timed repeats.

    Returns:
        Best whole-prompt milliseconds.
    """
    import torch

    model.reset()
    model.prefill(toks, chunk=chunk)
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        model.reset()
        start, end = _events()
        start.record()
        model.prefill(toks, chunk=chunk)
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end))
    return best


def _patch_cheap_dequant():
    """Replace ``dequant_kn`` with a bf16-cast-plus-in-place-multiply spelling.

    Returns:
        A zero-argument callable that restores the original method.
    """
    from backends.corex import pack

    original = pack.KnQWeight.dequant_kn

    def dequant_kn(self, dtype=pack.WEIGHT_DTYPE):
        """Dequantise without the fp32 intermediate.

        ``self.wt.float()`` allocates 4 bytes per element and is then narrowed back to 2,
        so the current spelling moves 4x the payload the answer needs. Casting first and
        scaling in place is two passes at the final width. The scale is narrowed too:
        leaving it fp32 promotes the multiply and reinstates the wide intermediate.

        Args:
            self: The packed int8 weight.
            dtype: Storage dtype of the result.

        Returns:
            A :class:`pack.KnWeight` over the dequantised ``[K, N]`` matrix.
        """
        out = self.wt.to(dtype)
        out.mul_(self.scale.to(dtype).view(1, -1))
        return pack.KnWeight(out)

    pack.KnQWeight.dequant_kn = dequant_kn

    def restore() -> None:
        """Put the original method back."""
        pack.KnQWeight.dequant_kn = original

    return restore


def _hoist_all(model) -> int:
    """Pre-dequantise every int8 projection and point the model at the dense form.

    Args:
        model: Loaded int8 model.

    Returns:
        Extra resident bytes, so the log can state the cost rather than imply it.
    """
    import torch

    from backends.corex import pack

    extra = 0
    for layer in model.layers:
        for proj in (getattr(layer.mix, "qkvzb_a", None), getattr(layer.mix, "qkv", None),
                     getattr(layer.mix, "out_proj", None), getattr(layer.mix, "o_proj", None),
                     getattr(layer.mlp, "gate_up", None),
                     getattr(layer.mlp, "down", None)):
            if proj is None:
                continue
            gemv = getattr(proj, "g", proj)
            packed = pack.as_kn(gemv.w)
            if not getattr(packed, "int8", False):
                continue
            dense = packed.dequant_kn(pack.WEIGHT_DTYPE)
            extra += dense.nbytes()
            gemv.w = dense
            gemv.q = None
    torch.cuda.synchronize()
    return extra


def main() -> int:
    """Run as-is, cheap-spelling and hoisted prefill against the same prompts.

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
    print(f"  device : {torch.cuda.get_device_name(0)} "
          f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB",
          flush=True)

    sd = load_weights(MODEL, quant=True, vision=False)
    model = create_model(MODEL, max_seq=MAX_SEQ, bench=False, weights=sd, paged=True)
    blocks = -(-MAX_SEQ // BLOCK)
    pool = model.enable_paging(num_blocks=blocks, block_size=BLOCK)
    pool.assign_table(list(range(blocks)), 0, row=model.prefill_row)
    model.mtp_prefill = False

    # The cheap spelling must produce the same weights as the current one before it is
    # allowed to produce a faster number.
    sample = next(iter(sd.values()))
    _ = sample
    restore = _patch_cheap_dequant()
    checked = 0
    for layer in model.layers[:1]:
        for proj in (getattr(layer.mix, "qkvzb_a", None), getattr(layer.mlp, "gate_up", None)):
            if proj is None:
                continue
            gemv = getattr(proj, "g", proj)
            packed = pack.as_kn(gemv.w)
            if not getattr(packed, "int8", False):
                continue
            fast = packed.dequant_kn(pack.WEIGHT_DTYPE).wt.float()
            restore()
            slow = packed.dequant_kn(pack.WEIGHT_DTYPE).wt.float()
            _patch_cheap_dequant()
            diff = float((fast - slow).abs().max())
            print(f"  cheap-vs-current dequant max |diff| = {diff:.2e}", flush=True)
            checked += 1
    restore()
    if not checked:
        print("  no int8 weight found to verify", flush=True)

    results: dict[tuple[str, int, int], float] = {}
    for label in ("as-is", "cheap spelling"):
        if label == "cheap spelling":
            restore = _patch_cheap_dequant()
        for n, chunk in CASES:
            ms = _time_prefill(model, _tokens(n), chunk)
            results[(label, n, chunk)] = ms
            print(f"  {label:<16} n={n:<5} chunk={chunk:<5} {ms:>9.1f} ms "
                  f"{n / ms * 1e3:>7.0f} tok/s", flush=True)
        if label == "cheap spelling":
            restore()

    extra = _hoist_all(model)
    for n, chunk in CASES:
        ms = _time_prefill(model, _tokens(n), chunk)
        results[("hoisted", n, chunk)] = ms
        print(f"  {'hoisted':<16} n={n:<5} chunk={chunk:<5} {ms:>9.1f} ms "
              f"{n / ms * 1e3:>7.0f} tok/s", flush=True)

    print(f"\n  --- summary (ms, and gain over as-is) ---", flush=True)
    print(f"  {'n':>6}{'chunk':>7}{'as-is':>10}{'cheap':>10}{'gain':>8}"
          f"{'hoisted':>10}{'gain':>8}", flush=True)
    for n, chunk in CASES:
        a = results[("as-is", n, chunk)]
        b = results[("cheap spelling", n, chunk)]
        c = results[("hoisted", n, chunk)]
        print(f"  {n:>6}{chunk:>7}{a:>10.1f}{b:>10.1f}{a / b:>7.2f}x"
              f"{c:>10.1f}{a / c:>7.2f}x", flush=True)
    print(f"\n  hoisted resident cost: {extra / 2**30:.2f} GiB extra", flush=True)

    print("\nINT8_PREFILL_APPLY_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
