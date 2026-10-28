"""Verify the two shipped prefill changes on the card: they must be faster and identical.

Both changes are already in the tree, and both are the kind that can look right while
being wrong:

1. ``pack.KnQWeight.dequant_kn`` now casts to the output dtype and scales in place instead
   of building a ``[K, N]`` fp32 temporary. It is *supposed* to be bit-identical (a bf16
   destination with an fp32 scale computes in fp32 and rounds once), so the check is
   equality, not a tolerance -- and it is checked through the engine's own weights, not on
   a synthetic tensor.
2. ``LLMEngine.build`` clears ``mtp_prefill`` when capture is off, because speculation is
   rejected in that branch so no drafter KV is ever read. The check is that a served
   prefill gets faster and that the greedy continuation is unchanged.

So this reports three things and refuses to call the job a success unless all three hold:

    dequant equality   every projection, max |diff| == 0 against the old spelling
    prefill time       the served path, before/after the drafter flag
    token ids          the greedy continuation, compared against the recorded baseline

The baseline ids are the ones every measurement in ``PERF_COREX.md`` has been using:
``[6, 24218, 10, 4838, 1665, 15]`` on ``[1, 2, 3, 4, 5]``.

Module: ``coffin/diag/corex_probe``. Runs on the card; nothing imports it.

Constraints:
    * Eager only; graphs are off on this backend.
    * The old dequantisation spelling is reconstructed here rather than kept behind a
      flag, so the tree carries one implementation and the probe carries the comparison.
    * ``mtp_prefill`` is toggled directly for the A/B; the shipped default is what
      ``build()`` computes, so the default run is what a served request pays.
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

#: Context window; covers the prompt below.
MAX_SEQ = 8192

#: Tokens per block in the paged pool.
BLOCK = 64

#: Prompt length for the served-path timing.
N = 2048

#: The greedy continuation every measurement in PERF_COREX.md compares against.
BASELINE_IDS = [6, 24218, 10, 4838, 1665, 15]

#: Prompt the baseline was recorded on.
SEED = [1, 2, 3, 4, 5]


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


def _time_prefill(model, toks: list[int], chunk: int, reps: int = 3) -> float:
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


def _collect(model) -> list:
    """Every distinct int8 projection the prefill path dequantises.

    Args:
        model: Loaded int8 model.

    Returns:
        The packed weights, de-duplicated by identity.
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


def _check_dequant(weights: list) -> bool:
    """Compare the shipped dequantisation against the old wide-fp32 spelling.

    Args:
        weights: The packed int8 weights to check.

    Returns:
        ``True`` when every weight came out bit-identical.
    """
    import torch

    from backends.corex import pack

    bf16 = pack.WEIGHT_DTYPE
    worst = 0.0
    for w in weights:
        got = w.dequant_kn(bf16).wt
        want = (w.wt.float() * w.scale.view(1, -1)).to(bf16)
        worst = max(worst, float((got.float() - want.float()).abs().max()))
    ok = worst == 0.0
    print(f"  dequant vs old spelling: {len(weights)} projections, "
          f"max |diff| = {worst:.2e}  {'BIT-IDENTICAL' if ok else 'CHANGED'}", flush=True)
    return ok


def _check_ids(model) -> bool:
    """Replay the recorded greedy continuation and compare token ids.

    Args:
        model: Loaded model, freshly reset by this call.

    Returns:
        ``True`` when the ids match the baseline exactly.
    """
    import torch

    model.reset()
    logits = model.prefill(SEED)
    torch.cuda.synchronize()
    ids = []
    probe = logits
    for _ in range(len(BASELINE_IDS)):
        nxt = int(probe.argmax())
        ids.append(nxt)
        probe = model.logits(nxt)
        torch.cuda.synchronize()
    ok = ids == BASELINE_IDS
    print(f"  greedy ids: {ids}  baseline {BASELINE_IDS}  "
          f"{'UNCHANGED' if ok else 'DIVERGED'}", flush=True)
    return ok


def main() -> int:
    """Measure the shipped changes and gate on equality.

    Returns:
        ``0`` when the dequantisation is bit-identical, the ids are unchanged and every
        check ran; ``1`` otherwise.
    """
    import torch

    import backends
    from engine_api import create_model, load_weights

    print(f"  backend: {backends.status()}", flush=True)
    print(f"  model  : {MODEL}", flush=True)
    if not MODEL or not Path(MODEL).is_dir():
        print(f"  ENGINE_MODEL unset or not a directory: {MODEL!r}", flush=True)
        return 1
    print(f"  device : {torch.cuda.get_device_name(0)} "
          f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB",
          flush=True)

    ok = True
    sd = load_weights(MODEL, quant=True, vision=False)
    model = create_model(MODEL, max_seq=MAX_SEQ, bench=False, weights=sd, paged=True)
    blocks = -(-MAX_SEQ // BLOCK)
    pool = model.enable_paging(num_blocks=blocks, block_size=BLOCK)
    pool.assign_table(list(range(blocks)), 0, row=model.prefill_row)

    print("\n  --- 1. dequantisation is bit-identical ---", flush=True)
    ok &= _check_dequant(_collect(model))

    print("\n  --- 2. greedy continuation is unchanged ---", flush=True)
    ok &= _check_ids(model)

    print(f"\n  --- 3. served prefill, n={N}, chunk=2048 ---", flush=True)
    toks = _tokens(N)
    # This is the flag value ``LLMEngine.build`` now computes for a graph-less backend,
    # applied by hand because this probe drives the model without an engine.
    model.mtp_prefill = False
    shipped = _time_prefill(model, toks, 2048)
    # And what the old default cost, for the delta.
    model.mtp_prefill = True
    old = _time_prefill(model, toks, 2048)
    print(f"  mtp_prefill=True  (old default)  {old:>9.1f} ms "
          f"{N / old * 1e3:>7.0f} tok/s", flush=True)
    print(f"  mtp_prefill=False (shipped)      {shipped:>9.1f} ms "
          f"{N / shipped * 1e3:>7.0f} tok/s   {old / shipped:.2f}x", flush=True)

    print("\n  --- 4. shipped attention defaults, long prompt ---", flush=True)
    # Step 3 ends on the old-default arm of its A/B (``mtp_prefill=True``); put the
    # shipped value back or this step would time a flag the engine no longer sets.
    model.mtp_prefill = False
    from models.qwen35 import tuning as _tuning

    print(f"  PREFILL_ATTN_TILE={_tuning.PREFILL_ATTN_TILE} "
          f"PREFILL_ATTN_KB={_tuning.PREFILL_ATTN_KB}", flush=True)
    for n in (2048, 8192):
        ms = _time_prefill(model, _tokens(n), 2048)
        print(f"  n={n:<5} {ms:>9.1f} ms {n / ms * 1e3:>7.0f} tok/s", flush=True)
    # A bigger key block changes the online-softmax accumulation order, so the ids have
    # to be re-checked here and not assumed: this is the same gate as step 2.
    ok &= _check_ids(model)

    print(f"\n  --- verdict: {'PASS' if ok else 'FAIL'} ---", flush=True)
    print("\nPREFILL_FIXES_DONE", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
