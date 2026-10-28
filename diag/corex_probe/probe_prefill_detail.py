"""Attribute one prefill to its stages, its GEMM shapes and its weight form.

The first measurement (``bench_prefill.py``) produced three results that need a cause
rather than a description:

1. ``chunk=1024`` beats the default ``2048`` on dense weights and loses on int8 weights.
   Chunk size cannot be tuned against two contradictory answers, so the cost that flips
   the sign has to be named.
2. The int8 weight path is *slower* despite reading half the bytes. The prime suspect is
   that ``linear_prefill`` dequantises the whole weight on every call, which predicts a
   specific profile shape; that is checked here rather than assumed.
3. A handful of GEMM launches cost ~10.5 ms each against ~0.76 ms for the tensor-core
   ones. Grouping the profiler by input shape names their ``M``/``N``/``K``.

And one gap: the served path is the **paged** KV pool, while every earlier number came
from the dense cache. Paged prefill materialises a dense view of the pool
(``PagedKVCache.gather``), so it is not free and a figure taken on the wrong cache is
not the figure a request pays.

Module: ``coffin/diag/corex_probe``. Runs on the card; nothing imports it.

Constraints:
    * Eager only; there is no graph on this backend.
    * The stage split is read off the device timeline (events between stages, no sync
      inside the prefill). A synchronise per layer would serialise the pipeline and
      report a sum of stalls rather than the work.
    * ``record_shapes`` is on for the profiler run only: it is the only way to learn the
      M/N/K of a slow GEMM instead of guessing, and it costs memory the other runs do
      not need.
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

#: Context window to build with; covers every prompt below.
MAX_SEQ = 4096

#: Text tokens per block in the pool, the engine's own default.
BLOCK = 64

#: Prompt lengths for the headline table.
SIZES = (128, 512, 2048)

#: Prompt length the attribution runs use; the DCU figures are quoted at it.
N = 2048

#: Chunk sizes to sweep. 2048 is the engine default, and 4096 is above it because the
#: sweep on the DCU stopped at 2048 only because the card's library GEMM topped out there.
CHUNKS = (512, 1024, 2048, 4096)


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
        ``n`` ids from the arithmetic sequence the other probes use, so runs are
        comparable across jobs.
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


def _stage_timing(model, toks: list[int], chunk: int) -> None:
    """Split one prefill into per-layer projection, mixer state and MLP time.

    Hooks the methods ``Layer.prefill`` dispatches to, so every number is a real region
    of the call graph rather than a guess about which op belongs where. The residual
    adds and the norm live between the hooks and land in the "layer overhead" line.

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
        """Bracket one method with two device events.

        Args:
            owner: Object holding the method.
            name: Attribute name, bound on the instance so the class is untouched.
            label: Bucket the event pair is filed under.
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
        # ``Layer.prefill`` is projection, then ``prefill_state`` (the stateful half),
        # then the residual and the MLP. Hooking the middle one keeps the mixer's own
        # GEMM out of the state bucket, which is the split that decides whether the
        # cost is a projection or the recurrence/attention.
        wrap(layer, "prefill_state", "mix_state_linear" if linear else "mix_state_full")
        wrap(layer.mix, "prefill_project", "mix_proj_linear" if linear
             else "mix_proj_full")
        wrap(layer.mlp, "prefill", "mlp")

    model.reset()
    model.prefill(toks, chunk=chunk)
    torch.cuda.synchronize()

    start, end = _events()
    model.reset()
    start.record()
    model.prefill(toks, chunk=chunk)
    end.record()
    torch.cuda.synchronize()

    for owner, name, original in undo:
        setattr(owner, name, original)

    wall = start.elapsed_time(end)

    def total(label: str) -> float:
        """Sum one bucket's event pairs.

        Args:
            label: Bucket name.

        Returns:
            Milliseconds; ``0.0`` for a bucket that was never entered.
        """
        return sum(s.elapsed_time(e) for s, e in buckets.get(label, ()))

    print(f"\n  --- stage split, n={len(toks)}, chunk={chunk}, mtp off ---", flush=True)
    print(f"  {'whole prefill':<28}{wall:>9.1f} ms", flush=True)
    layers = 0.0
    for label in ("mix_proj_linear", "mix_state_linear", "mix_proj_full",
                  "mix_state_full", "mlp"):
        ms = total(label)
        layers += ms
        print(f"  {label:<28}{ms:>9.1f} ms  ({100 * ms / wall:>5.1f}%)", flush=True)
    print(f"  {'layer total':<28}{layers:>9.1f} ms  ({100 * layers / wall:>5.1f}%)",
          flush=True)
    print(f"  {'embed + mtp + head + glue':<28}{wall - layers:>9.1f} ms  "
          f"({100 * (wall - layers) / wall:>5.1f}%)", flush=True)


def _profile_shapes(model, toks: list[int], chunk: int) -> None:
    """Profile one prefill and rank the matmuls by input shape.

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
    with profile(activities=activities, record_shapes=True) as prof:
        model.prefill(toks, chunk=chunk)
        torch.cuda.synchronize()

    print(f"\n  --- top kernels, n={len(toks)}, chunk={chunk} ---", flush=True)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=18), flush=True)
    print(f"\n  --- matmuls grouped by input shape ---", flush=True)
    rows = []
    for evt in prof.key_averages(group_by_input_shape=True):
        if evt.key not in ("aten::mm", "aten::matmul", "aten::bmm", "aten::baddbmm",
                           "aten::einsum", "aten::_scaled_dot_product"):
            continue
        if not evt.cuda_time_total:
            continue
        shapes = " ".join("[" + ",".join(str(d) for d in item) + "]"
                          for item in (evt.input_shapes or ()))
        rows.append((evt.cuda_time_total, evt.count, evt.key, shapes))
    rows.sort(reverse=True)
    print(f"  {'us total':>10}{'calls':>7}  {'op':<14} shapes", flush=True)
    for us, count, key, shapes in rows[:20]:
        print(f"  {us:>10.0f}{count:>7}  {key:<14} {shapes}", flush=True)


def _bench_linear_prefill(model) -> None:
    """Time ``linear_prefill`` per shape on the model's own packed weights.

    Args:
        model: Loaded model; its layers supply the packed weights.
    """
    import torch

    import backends
    from backends.corex import pack

    table = backends.ops
    print(f"\n  --- linear_prefill on this model's weights (M=2048) ---", flush=True)
    print(f"  {'label':<22}{'N':>7}{'K':>6}{'ms':>9}{'TFLOP/s':>10}{'GB/s':>8}{'form':>7}",
          flush=True)
    seen = set()
    for layer in model.layers:
        # A fused projection is a ``GEMVGroup``; its concatenated weight is what the
        # kernel is handed, so the group's inner ``GEMV`` is the one to time.
        named = (("mix.in_proj", getattr(layer.mix, "qkvzb_a", None)
                  or getattr(layer.mix, "qkv", None)),
                 ("mix.out_proj", getattr(layer.mix, "out_proj", None)
                  or getattr(layer.mix, "o_proj", None)),
                 ("mlp.gate_up", getattr(layer.mlp, "gate_up", None)),
                 ("mlp.down", getattr(layer.mlp, "down", None)))
        for label, proj in named:
            if proj is None:
                continue
            gemv = getattr(proj, "g", proj)
            packed = pack.as_kn(gemv.w)
            key = (label, packed.n, packed.k)
            if key in seen:
                continue
            seen.add(key)
            x = torch.randn(2048, packed.k, dtype=torch.bfloat16, device="cuda")
            table.linear_prefill(gemv.w, x)
            torch.cuda.synchronize()
            best = float("inf")
            for _ in range(3):
                begin, end = _events()
                begin.record()
                table.linear_prefill(gemv.w, x)
                end.record()
                torch.cuda.synchronize()
                best = min(best, begin.elapsed_time(end))
            flops = 2 * 2048 * packed.n * packed.k
            nbytes = packed.n * packed.k * (1 if packed.int8 else 2)
            form = "int8" if packed.int8 else str(packed.wt.dtype).split(".")[-1]
            print(f"  {label:<22}{packed.n:>7}{packed.k:>6}{best:>9.2f}"
                  f"{flops / (best * 1e-3) / 1e12:>10.1f}"
                  f"{nbytes / (best * 1e-3) / 1e9:>8.0f}{form:>7}", flush=True)
            del x
    del table


def main() -> int:
    """Run the attribution suite on the paged, single-stream configuration.

    Returns:
        ``0`` on success, ``1`` when the model directory is missing.
    """
    import time

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

    # >384 slots is ~= the model's own vocabulary, one per prompt token (up to 16k).
    for label, quant in (("int8", True), ("dense", False)):
        print(f"\n{'=' * 78}\nweight path: {label}\n{'=' * 78}", flush=True)
        t0 = time.time()
        sd = load_weights(MODEL, quant=quant, vision=False)
        model = create_model(MODEL, max_seq=MAX_SEQ, bench=False, weights=sd, paged=True)
        blocks = -(-MAX_SEQ // BLOCK)
        pool = model.enable_paging(num_blocks=blocks, block_size=BLOCK)
        pool.assign_table(list(range(blocks)), 0, row=model.prefill_row)
        # The served configuration on this backend: ``caps.graph`` is 0, so the engine
        # clears speculative decoding, so the drafter's prefill KV can never be used.
        model.mtp_prefill = False
        print(f"  loaded in {time.time() - t0:.1f}s  paged slots={pool.num_slots} "
              f"mtp_prefill=False", flush=True)

        print(f"\n  --- chunk sweep, paged, n={N} ---", flush=True)
        print(f"  {'chunk':>7}{'ms':>10}{'tok/s':>10}{'vs 2048':>10}", flush=True)
        toks = _tokens(N)
        base = None
        for chunk in CHUNKS:
            ms = _time_prefill(model, toks, chunk)
            if chunk == 2048:
                base = ms
            print(f"  {chunk:>7}{ms:>10.1f}{N / ms * 1e3:>10.0f}"
                  f"{(base / ms if base else float('nan')):>9.2f}x", flush=True)

        print("\n  --- headline at the best chunk ---", flush=True)
        print(f"  {'n':>6}{'ms':>10}{'tok/s':>10}{'ms/token':>11}", flush=True)
        for n in SIZES:
            ms = _time_prefill(model, _tokens(n), 2048)
            print(f"  {n:>6}{ms:>10.1f}{n / ms * 1e3:>10.0f}{ms / n:>11.3f}", flush=True)

        _stage_timing(model, toks, 1024)
        _profile_shapes(model, toks, 1024)
        _bench_linear_prefill(model)

        del model, sd, pool
        gc.collect()
        torch.cuda.empty_cache()

    print("\nPREFILL_DETAIL_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
