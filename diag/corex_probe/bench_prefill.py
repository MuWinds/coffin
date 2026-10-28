"""Measure prefill throughput on this card, and attribute it to stages.

Nothing in the tree has ever measured prefill on CoreX. ``PERF_COREX.md`` is decode only, and
the only prefill figure anywhere is ``prefill 1056.5 ms (5 tokens)`` from the first end-to-end
run -- a prompt short enough that ``Qwen35.prefill`` routes it through the decode path, so it
measures nothing about batched prefill at all.

So this reports four things that answer different questions:

1. **Headline tok/s at a few prompt lengths**, so the number can be quoted.
2. **The chunk sweep.** ``chunk`` is the prefill GEMM's ``M``; on the DCU bigger was
   monotonically better up to 2048, but that tuning ran against a rocBLAS GEMM. Whether it
   holds where the library is only used below ``LIB_MAX_N`` columns is a measurement.
3. **A profiler ranking of one prefill.** Named ops with device time, which separates "the
   GEMMs dominate" from "the glue between them dominates".
4. **A per-shape ``linear_prefill`` micro-benchmark on the real projection shapes**, dense
   against int8. The int8 entry dequantises the entire weight per call
   (``pack.KnQWeight.dequant_kn``), and the size of that cost decides whether an int8-served
   model should prefill from int8 storage at all.

The MTP drafter prefill is on by default for single-stream prefill (``mtp_prefill``) and adds
one full-attention layer plus a projection over the whole prompt, so the same prompt is timed
with it on and off.

Module: ``coffin/diag/corex_probe``. Runs on the card; nothing imports it.

Constraints:
    * Eager only. Graphs are off on this backend and stay off on this driver.
    * Both weight paths in one process, because loading the checkpoint costs ~35 s: the int8
      model is measured, dropped, and only then is the dense one built. 4.6 GB and 8.8 GB of
      weights do not coexist comfortably beside the activations on a 32 GB card.
    * The size list adapts after the first measurements. If this card turns out an order of
      magnitude slower than the DCU at 512 tokens, the 2048-token rows are skipped rather
      than left to eat the job's whole wall clock.
"""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

#: ``coffin/`` is the ``sys.path`` root; this file sits two levels below it.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

MODEL = os.environ.get("ENGINE_MODEL", "")

#: Context window to build with. Only has to cover the longest prompt below; the dense
#: full-attention caches of the un-paged model are sized from it.
MAX_SEQ = 4096

#: Prompt lengths to report, longest last. 128 is the smallest size comfortably inside the
#: batched path (under 4 tokens the model routes to decode); 2048 is the engine's default
#: chunk and the length the DCU figures are quoted at.
SIZES = (128, 512, 2048)

#: Chunk sizes to sweep at the largest prompt length. 2048 is the default; the rest bracket
#: it so a peak in the middle is visible.
CHUNKS = (256, 512, 1024, 2048)

#: A 512-token prefill slower than this (seconds) means the 2048-token rows would cost several
#: minutes each, so they are skipped. The DCU needs ~0.3 s here.
_SLOW_S = 20.0

#: Real projection shapes of this model, from ``probe_shapes.py`` on a decode step. The fused
#: projections are not their parts' shapes, which is why these are the measured ones.
PROJ_SHAPES = (
    ("gate_up", 18432, 2560),
    ("mlp_down", 2560, 9216),
    ("linear_attn_qkvzba", 12352, 2560),
    ("out_proj", 2560, 4096),
    ("full_attn_qkvz", 10240, 2560),
)


def _events() -> tuple:
    """Create a start/end device-event pair for one timed region.

    Returns:
        ``(start, end)`` events with timing enabled.
    """
    import torch

    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def _prefill_ms(model, toks: list[int], chunk: int, reps: int = 2) -> float:
    """Time ``model.prefill`` on one prompt, best of ``reps``.

    Best rather than mean because the first repeat can still absorb an allocator growth; a
    mean over two samples would carry that into the headline.

    Args:
        model: A loaded model; ``reset()`` runs before each timed pass.
        toks: Prompt token ids.
        chunk: Tokens per prefill GEMM pass.
        reps: Timed repeats.

    Returns:
        Best milliseconds for the whole prompt.
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


def _tokens(n: int) -> list[int]:
    """Build a deterministic prompt of ``n`` legal token ids.

    An arithmetic sequence rather than one repeated token: a flat prompt is the case most
    likely to hide a bad row somewhere in the middle.

    Args:
        n: Prompt length.

    Returns:
        ``n`` token ids below any plausible vocabulary size.
    """
    return [(i * 7919 + 13) % 100000 for i in range(n)]


def _bench_proj(shape: tuple[str, int, int]) -> None:
    """Time ``linear_prefill`` at M=2048 for one shape, dense against int8.

    The dense row is the reference; the int8 row pays a full dequantisation of the weight
    inside the call, so the gap between them is the price of serving prefill from int8
    storage on this backend.

    Args:
        shape: ``(label, N, K)`` of the projection.
    """
    import torch

    import backends
    from int8_weights import QWeight

    label, n, k = shape
    table = backends.ops
    m = 2048
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    rows = []
    for tag, weight in (("dense", w), ("int8", QWeight.quantize(w))):
        table.linear_prefill(weight, x)
        torch.cuda.synchronize()
        best = float("inf")
        for _ in range(3):
            start, end = _events()
            start.record()
            table.linear_prefill(weight, x)
            end.record()
            torch.cuda.synchronize()
            best = min(best, start.elapsed_time(end))
        nbytes = n * k * (1 if tag == "int8" else 2)
        flops = 2 * m * n * k
        rows.append((best, nbytes / (best * 1e-3) / 1e9, flops / (best * 1e-3) / 1e12))
    d, q = rows
    print(f"  {label:<20} N={n:>6} K={k:<5} "
          f"dense {d[0]:>7.2f} ms {d[1]:>6.0f} GB/s {d[2]:>6.1f} TFLOP/s | "
          f"int8 {q[0]:>7.2f} ms {q[1]:>6.0f} GB/s ({q[0] / d[0]:.2f}x)", flush=True)
    del x, w
    torch.cuda.empty_cache()


def _profile_one(model, toks: list[int], chunk: int) -> None:
    """Print the profiler's top ops for one prefill.

    Device time per op name is the attribution this exists for; the count column says whether
    the cost is one slow kernel or many small ones.

    Args:
        model: Loaded model.
        toks: Prompt token ids.
        chunk: Tokens per GEMM pass.
    """
    import torch
    from torch.profiler import ProfilerActivity, profile

    model.reset()
    model.prefill(toks, chunk=chunk)
    torch.cuda.synchronize()
    activities = [ProfilerActivity.CPU]
    if hasattr(ProfilerActivity, "CUDA"):
        activities.append(ProfilerActivity.CUDA)
    model.reset()
    with profile(activities=activities) as prof:
        model.prefill(toks, chunk=chunk)
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=22), flush=True)


def _run_weight_path(label: str, quant: bool, sizes: tuple[int, ...],
                     chunks: tuple[int, ...]) -> None:
    """Measure one weight path: headline sizes, chunk sweep, MTP A/B, profiler.

    Args:
        label: ``"int8"`` or ``"dense"``, for the printed header.
        quant: Passed to ``load_weights``; selects the weight storage form.
        sizes: Prompt lengths for the headline table.
        chunks: Chunk sizes to sweep at the largest prompt length.
    """
    import time

    import backends
    from engine_api import create_model, load_weights

    print(f"\n{'=' * 78}\nweight path: {label}\n{'=' * 78}", flush=True)
    t0 = time.time()
    sd = load_weights(MODEL, quant=quant, vision=False)
    model = create_model(MODEL, max_seq=MAX_SEQ, bench=False, weights=sd)
    print(f"  loaded in {time.time() - t0:.1f}s  "
          f"caps graph={int(backends.caps().graph)} mtp={model.mtp is not None}", flush=True)

    print("\n  --- headline (chunk=2048) ---", flush=True)
    print(f"  {'n':>6}{'ms':>10}{'tok/s':>10}{'ms/token':>11}", flush=True)
    measured: dict[int, float] = {}
    for n in sizes:
        ms = _prefill_ms(model, _tokens(n), 2048)
        measured[n] = ms
        print(f"  {n:>6}{ms:>10.1f}{n / ms * 1e3:>10.0f}{ms / n:>11.3f}", flush=True)
        if n == 512 and ms > _SLOW_S * 1e3:
            print(f"  (512 tokens took {ms / 1e3:.0f} s: skipping the longer rows)", flush=True)
            break

    top = max(measured)
    if top < sizes[-1]:
        print(f"\n  --- chunk sweep at n={top} (the largest size measured) ---", flush=True)
    else:
        print(f"\n  --- chunk sweep at n={top} ---", flush=True)
    toks = _tokens(top)
    print(f"  {'chunk':>7}{'ms':>10}{'tok/s':>10}{'vs 2048':>10}", flush=True)
    base = None
    for chunk in chunks:
        if chunk > top:
            continue
        ms = _prefill_ms(model, toks, chunk)
        if chunk == 2048:
            base = ms
        print(f"  {chunk:>7}{ms:>10.1f}{top / ms * 1e3:>10.0f}"
              f"{(base / ms if base else float('nan')):>9.2f}x", flush=True)

    print(f"\n  --- MTP drafter during prefill (n={top}) ---", flush=True)
    if model.mtp is not None:
        for flag in (True, False):
            model.mtp_prefill = flag
            ms = _prefill_ms(model, toks, 2048)
            print(f"  mtp_prefill={str(flag):<5} {ms:>9.1f} ms "
                  f"{top / ms * 1e3:>8.0f} tok/s", flush=True)
        model.mtp_prefill = True
    else:
        print("  no MTP head in this checkpoint", flush=True)

    probe_n = min(512, top)
    print(f"\n  --- profiler, one prefill of {probe_n} tokens ---", flush=True)
    try:
        _profile_one(model, _tokens(probe_n), 2048)
    except Exception as exc:                      # noqa: BLE001 - trace is a bonus
        print(f"  profiler unavailable: {type(exc).__name__}: {exc}", flush=True)

    del model, sd
    gc.collect()


def main() -> int:
    """Run every measurement and print the tables.

    Returns:
        ``0`` on success, ``1`` when the model directory is missing.
    """
    import torch

    import backends

    print(f"  backend: {backends.status()}", flush=True)
    print(f"  model  : {MODEL}", flush=True)
    if not MODEL or not Path(MODEL).is_dir():
        print(f"  ENGINE_MODEL unset or not a directory: {MODEL!r}", flush=True)
        return 1
    print(f"  device : {torch.cuda.get_device_name(0)} "
          f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB",
          flush=True)

    _run_weight_path("int8", True, SIZES, CHUNKS)
    torch.cuda.empty_cache()
    _run_weight_path("dense", False, SIZES, CHUNKS)

    print(f"\n{'=' * 78}\nlinear_prefill per real shape (M=2048)\n{'=' * 78}",
          flush=True)
    for shape in PROJ_SHAPES:
        try:
            _bench_proj(shape)
        except Exception as exc:                  # noqa: BLE001 - one shape may OOM
            print(f"  {shape[0]:<20} FAILED {type(exc).__name__}: {exc}", flush=True)

    print("\nPREFILL_BENCH_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
