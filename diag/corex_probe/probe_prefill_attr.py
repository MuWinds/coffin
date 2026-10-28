"""Find the prefill costs that a fix can actually remove.

The first two measurements left four specific unknowns, all of them "what is this
number made of" rather than "how fast is it":

1. **Five GEMMs that cost ~10.4 ms each** in the n=512 profile, against ~0.76 ms for the
   other 104. Their kernel is the non-tensor-core ``GemmEx::gemmh_new`` fallback, so
   something in the prefill is asking for a GEMM shape the tensor-core path refuses.
   ``record_shapes`` names it.
2. **The delta scan's chunk length.** ``chunked_delta`` is called with its default
   ``chunk=64`` no matter what the model-level ``chunk`` is, so a 2048-token prefill runs
   32 sequential scan steps per linear layer (24 layers x 32 = 768 ``baddbmm`` launches,
   visible in the previous profile). The scan is exact for any split, so this is a pure
   knob -- and the previous run showed it as 56-91 ms, one of the largest single items.
3. **The int8 prefill dequantisation.** ``linear_prefill`` calls
   ``KnQWeight.dequant_kn`` on every call, which materialises the whole weight at bf16 --
   4.4 GB in, 8.8 GB out, per prefill. That is affordable as a one-off at load and
   indefensible per request. Pre-dequantising the weights in the probe simulates a cached
   version exactly, so the size of the prize is measurable before anyone writes it.
4. **The stage split, correctly this time.** The previous attempt installed its hooks
   before the warmup and summed the warmup's event pairs along with the timed ones, which
   is why the buckets added up to 197% of the wall clock. Buckets are cleared here.

Module: ``coffin/diag/corex_probe``. Runs on the card; nothing imports it.

Constraints:
    * Eager only; there is no graph on this backend.
    * Every A/B is run twice and the second value is reported, so a first-call allocator
      growth cannot be read as a win.
    * The dequantisation experiment *adds* a resident bf16 copy and does not drop the
      int8 one, which is the honest simulation of a cache but also the peak-VRAM cost:
      13.2 GB of weights on a 32 GB card. It is the last thing this probe does.
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

#: Context window; covers every prompt below.
MAX_SEQ = 4096

#: Tokens per block in the paged pool.
BLOCK = 64

#: The prompt length the DCU figures are quoted at, and the one used throughout.
N = 2048

#: Prompt length the first profile used, so the five slow GEMMs show up again.
SMALL_N = 512

#: Delta-scan chunk lengths to try. 64 is the hardcoded default; 128 is the next step up.
#: 256 is deliberately absent: ``triangular_solve`` holds the whole ``L x L`` block in
#: 48 KB of shared memory, so 256 rows needs 64 KB and fails to launch. The kernel says
#: so rather than silently falling back, which is why the limit is a fact and not a guess.
DELTA_CHUNKS = (64, 128)


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


def _time(model, toks: list[int], chunk: int, reps: int = 3) -> float:
    """Time one prefill, best of ``reps``, in milliseconds.

    Args:
        model: Loaded model.
        toks: Prompt token ids.
        chunk: Tokens per model-level GEMM pass.
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


def _load(quant: bool):
    """Build a paged single-stream model with the drafter's prefill turned off.

    ``mtp_prefill=False`` is not a probe convenience: with ``caps.graph == 0`` the engine
    clears speculative decoding, so the drafter's prefill KV is never read.

    Args:
        quant: Load int8 weights (the serving default) when ``True``.

    Returns:
        ``(model, pool)``.
    """
    from engine_api import create_model, load_weights

    sd = load_weights(MODEL, quant=quant, vision=False)
    model = create_model(MODEL, max_seq=MAX_SEQ, bench=False, weights=sd, paged=True)
    blocks = -(-MAX_SEQ // BLOCK)
    pool = model.enable_paging(num_blocks=blocks, block_size=BLOCK)
    pool.assign_table(list(range(blocks)), 0, row=model.prefill_row)
    model.mtp_prefill = False
    return model, pool


def _stage_split(model, toks: list[int], chunk: int) -> None:
    """Split one prefill into projection, mixer-state and MLP time.

    Buckets are cleared after the warmup. The first version of this summed the warmup's
    own event pairs, which made the stages add up to 197% of the wall clock -- an
    arithmetic impossibility that was easy to misread as overlap.

    Args:
        model: Loaded model.
        toks: Prompt token ids.
        chunk: Tokens per GEMM pass.
    """
    import torch

    from models.qwen35.linear_attn import LinearAttn

    buckets: dict[str, list] = {}
    undo: list[tuple[object, str, object]] = []

    def wrap(owner, name: str, label: str) -> None:
        """Bracket one method with two device events and file them under ``label``.

        Args:
            owner: Object holding the method.
            name: Attribute name; the instance gets the binding, not the class.
            label: Bucket name.
        """
        original = getattr(owner, name)
        bucket = buckets.setdefault(label, [])

        def wrapped(*args, **kw):
            begin, _ = _events()
            begin.record()
            out = original(*args, **kw)
            end, _ = _events()
            end.record()
            bucket.append((begin, end))
            return out

        undo.append((owner, name, original))
        setattr(owner, name, wrapped)

    for layer in model.layers:
        linear = isinstance(layer.mix, LinearAttn)
        wrap(layer, "prefill_state", "mix_state_linear" if linear else "mix_state_full")
        wrap(layer.mix, "prefill_project",
             "mix_proj_linear" if linear else "mix_proj_full")
        wrap(layer.mlp, "prefill", "mlp")

    model.reset()
    model.prefill(toks, chunk=chunk)
    torch.cuda.synchronize()
    for bucket in buckets.values():
        bucket.clear()

    start, end = _events()
    model.reset()
    start.record()
    model.prefill(toks, chunk=chunk)
    end.record()
    torch.cuda.synchronize()

    for owner, name, original in undo:
        setattr(owner, name, original)

    wall = start.elapsed_time(end)
    print(f"\n  --- stage split, n={len(toks)}, chunk={chunk} ---", flush=True)
    print(f"  {'whole prefill':<28}{wall:>9.1f} ms", flush=True)
    accounted = 0.0
    for label in ("mix_proj_linear", "mix_state_linear", "mix_proj_full",
                  "mix_state_full", "mlp"):
        ms = sum(s.elapsed_time(e) for s, e in buckets.get(label, ()))
        accounted += ms
        print(f"  {label:<28}{ms:>9.1f} ms  ({100 * ms / wall:>5.1f}%)", flush=True)
    print(f"  {'accounted':<28}{accounted:>9.1f} ms  "
          f"({100 * accounted / wall:>5.1f}%)", flush=True)
    print(f"  {'residual (norms, add, head)':<28}{wall - accounted:>9.1f} ms  "
          f"({100 * (wall - accounted) / wall:>5.1f}%)", flush=True)


def _profile_shapes(model, toks: list[int], chunk: int, tag: str) -> None:
    """Profile one prefill, ranking kernels and then matmuls by input shape.

    Args:
        model: Loaded model.
        toks: Prompt token ids.
        chunk: Tokens per GEMM pass.
        tag: Label printed in the headers, so two profiles in one log stay apart.
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
    with profile(activities=activities, record_shapes=True) as prof:
        model.prefill(toks, chunk=chunk)
        torch.cuda.synchronize()

    print(f"\n  --- kernels, {tag}: n={len(toks)}, chunk={chunk} ---", flush=True)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=14), flush=True)
    print(f"\n  --- matmuls by input shape, {tag} ---", flush=True)
    rows = []
    for evt in prof.key_averages(group_by_input_shape=True):
        if evt.key not in ("aten::mm", "aten::matmul", "aten::bmm", "aten::baddbmm",
                           "aten::einsum"):
            continue
        if not evt.cuda_time_total:
            continue
        shapes = " ".join("[" + ",".join(str(d) for d in item) + "]"
                          for item in (evt.input_shapes or ()))
        rows.append((evt.cuda_time_total, evt.count, evt.key, shapes))
    rows.sort(reverse=True)
    print(f"  {'us total':>10}{'calls':>7}  {'op':<14} shapes", flush=True)
    for us, count, key, shapes in rows[:14]:
        print(f"  {us:>10.0f}{count:>7}  {key:<14} {shapes}", flush=True)


def _delta_chunk_sweep(model, toks: list[int], chunk: int) -> None:
    """Re-run the prefill with a different chunked-delta chunk length.

    ``Layer.prefill`` does not pass ``chunk`` to ``LinearAttn.prefill``, so the scan runs
    at its default 64. Rebinding the mixer's method to bind a different value is the same
    code path the engine would take with the constant changed, and the scan is exact for
    any split, so this is a speed knob with no output change.

    Args:
        model: Loaded model.
        toks: Prompt token ids.
        chunk: Model-level GEMM chunk to hold fixed.
    """
    from models.qwen35.linear_attn import LinearAttn

    print(f"\n  --- delta-scan chunk (model chunk fixed at {chunk}) ---", flush=True)
    print(f"  {'delta chunk':>12}{'ms':>10}{'tok/s':>10}{'vs 64':>9}", flush=True)
    original = LinearAttn.prefill_state
    base = None
    for delta_chunk in DELTA_CHUNKS:
        def patched(self, qkv, z, b, a, row=None, chunk=64, _dc=delta_chunk):
            """``LinearAttn.prefill_state`` with the scan chunk bound to ``_dc``.

            Args:
                qkv: Fused q/k/v projection.
                z: Gate projection.
                b: Beta logits.
                a: Decay logits.
                row: State row.
                chunk: Ignored; the scan chunk is ``_dc``.
            """
            return original(self, qkv, z, b, a, row=row, chunk=_dc)

        LinearAttn.prefill_state = patched
        try:
            ms = _time(model, toks, chunk)
        except RuntimeError as exc:
            # A chunk the triangular-solve kernel cannot launch is a real answer, not a
            # probe failure: report it and keep the rest of the sweep.
            print(f"  {delta_chunk:>12}  unsupported: {exc}", flush=True)
            continue
        finally:
            LinearAttn.prefill_state = original
        if delta_chunk == 64:
            base = ms
        print(f"  {delta_chunk:>12}{ms:>10.1f}{len(toks) / ms * 1e3:>10.0f}"
              f"{(base / ms if base else float('nan')):>8.2f}x", flush=True)


def _dequant_cache_experiment(model, toks: list[int], chunk: int) -> None:
    """Pre-dequantise every int8 projection, then re-time the prefill.

    ``linear_prefill`` dequantises its weight on each call. Replacing each projection's
    weight with the already-dequantised ``[K, N]`` bf16 form removes that per-call work
    and nothing else, so the before/after gap is exactly what a cached dequantisation
    would save -- at the cost of holding both forms resident.

    Args:
        model: Loaded model with int8 weights.
        toks: Prompt token ids.
        chunk: Tokens per GEMM pass.
    """
    import torch

    from backends.corex import pack

    before = _time(model, toks, chunk)
    n_packed = 0
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
            n_packed += 1
    torch.cuda.synchronize()
    after = _time(model, toks, chunk)
    print(f"\n  --- cached dequantisation ({n_packed} projections) ---", flush=True)
    print(f"  int8, dequant per call   {before:>9.1f} ms  "
          f"{len(toks) / before * 1e3:>6.0f} tok/s", flush=True)
    print(f"  int8, dequant once       {after:>9.1f} ms  "
          f"{len(toks) / after * 1e3:>6.0f} tok/s   ({before / after:.2f}x)", flush=True)
    print(f"  extra resident bf16      {extra / 2**30:>9.2f} GiB", flush=True)


def main() -> int:
    """Run the four experiments on the int8 serving configuration.

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

    model, pool = _load(quant=True)
    print(f"  paged slots={pool.num_slots}  mtp_prefill=False", flush=True)
    toks = _tokens(N)

    _stage_split(model, toks, 2048)
    _delta_chunk_sweep(model, toks, 2048)
    _profile_shapes(model, _tokens(SMALL_N), 2048, "small")
    _profile_shapes(model, toks, 2048, "full")
    _dequant_cache_experiment(model, toks, 2048)

    print("\nPREFILL_ATTR_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
