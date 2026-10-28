"""Close the two gaps the prefill measurement left open.

Both are gaps in *evidence*, not new questions, and both are cheap to close in one job:

1. **What does the MTP drafter's prefill actually cost, op by op?** Turning it off is worth
   221.8 ms at n=2048 (1142.7 -> 920.9 on int8), which is the largest single win found so
   far and needs no kernel work. But the first profile that showed five ~10.4 ms
   non-tensor-core GEMMs had ``mtp_prefill=True`` at n=512, where those five would cost
   ~52 ms -- not 222. Something else is paying, and a probe that measures instead of
   inferring is the only way to say what. So: one ``record_shapes`` profile with the
   drafter's prefill **on**, at n=2048, and the difference against the same profile with
   it off.
2. **bf16 + paged + ``mtp_prefill=True``**, the one cell of the headline table that was
   extrapolated (952.5 ms) rather than measured. Paged costs 1.1-1.4% over the dense cache,
   so the extrapolation assumed the *dense* number plus that, which is two assumptions
   stacked. One measurement replaces both.

Module: ``coffin/diag/corex_probe``. Runs on the card; nothing imports it.

Constraints:
    * Eager only; there is no graph on this backend.
    * The two profiles are taken in one process, drafter on then off, on the same model
      and the same prompt, so the difference is the drafter and nothing else.
    * ``record_shapes`` is on: the point is the M/N/K of the GEMMs the drafter adds.
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
MAX_SEQ = 4096

#: Tokens per block in the paged pool.
BLOCK = 64

#: The prompt length the headline figures use.
N = 2048


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


def _profile(model, toks: list[int], chunk: int, tag: str) -> dict:
    """Profile one prefill and return the per-kernel and per-shape device times.

    Args:
        model: Loaded model.
        toks: Prompt token ids.
        chunk: Tokens per GEMM pass.
        tag: Label for the printed headers.

    Returns:
        ``{"device": {name: us}, "shapes": [(us, calls, op, shapes)]}`` so the caller can
        diff two profiles instead of eyeballing two tables.
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

    device = {}
    for evt in prof.key_averages():
        # Device kernels only: the ``aten::*`` rows are the host-side dispatchers, and
        # their CUDA totals nest into the kernels below them.
        if evt.key.startswith("aten::"):
            continue
        if evt.cuda_time_total:
            device[evt.key] = evt.cuda_time_total

    shapes = []
    for evt in prof.key_averages(group_by_input_shape=True):
        if evt.key not in ("aten::mm", "aten::matmul", "aten::bmm", "aten::baddbmm"):
            continue
        if not evt.cuda_time_total:
            continue
        shape = " ".join("[" + ",".join(str(d) for d in item) + "]"
                         for item in (evt.input_shapes or ()))
        shapes.append((evt.cuda_time_total, evt.count, evt.key, shape))

    print(f"\n  --- kernels, {tag} ---", flush=True)
    ranked = sorted(device.items(), key=lambda kv: -kv[1])
    print(f"  {'us':>10}  kernel", flush=True)
    for name, us in ranked[:12]:
        print(f"  {us:>10.0f}  {name[:76]}", flush=True)
    total = sum(device.values())
    print(f"  {total:>10.0f}  TOTAL device kernels", flush=True)

    print(f"\n  --- matmuls by shape, {tag} ---", flush=True)
    print(f"  {'us total':>10}{'calls':>7}  {'op':<14} shapes", flush=True)
    for us, count, key, shape in sorted(shapes, reverse=True)[:16]:
        print(f"  {us:>10.0f}{count:>7}  {key:<14} {shape}", flush=True)

    return {"device": device, "shapes": shapes}


def _report_ab(on: dict, off: dict) -> None:
    """Print what the drafter's prefill adds, kernel by kernel.

    Args:
        on: Profile taken with ``mtp_prefill=True``.
        off: Profile taken with ``mtp_prefill=False``.
    """
    print(f"\n  --- what mtp_prefill adds at n={N} ---", flush=True)
    names = set(on["device"]) | set(off["device"])
    rows = []
    for name in names:
        delta = on["device"].get(name, 0.0) - off["device"].get(name, 0.0)
        if abs(delta) > 1000.0:                   # >1 ms is above the noise here
            rows.append((delta, name))
    rows.sort(reverse=True)
    print(f"  {'delta us':>10}  kernel", flush=True)
    for delta, name in rows[:12]:
        print(f"  {delta:>10.0f}  {name[:76]}", flush=True)
    total = sum(on["device"].values()) - sum(off["device"].values())
    print(f"  {total:>10.0f}  TOTAL added by the drafter", flush=True)

    print(f"\n  --- matmuls the drafter adds (by shape) ---", flush=True)
    off_map = {(k, s): (us, c) for us, c, k, s in off["shapes"]}
    added = []
    for us, count, key, shape in on["shapes"]:
        before_us, before_c = off_map.get((key, shape), (0.0, 0))
        if us - before_us > 500.0:
            added.append((us - before_us, count - before_c, key, shape))
    added.sort(reverse=True)
    print(f"  {'delta us':>10}{'delta calls':>13}  {'op':<14} shapes", flush=True)
    for us, count, key, shape in added[:12]:
        print(f"  {us:>10.0f}{count:>13}  {key:<14} {shape}", flush=True)


def main() -> int:
    """Measure the two open cells: the drafter's cost, and bf16+paged+drafter.

    Returns:
        ``0`` on success, ``1`` when the model directory is missing.
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

    toks = _tokens(N)
    blocks = -(-MAX_SEQ // BLOCK)

    for label, quant in (("int8", True), ("bf16", False)):
        print(f"\n{'=' * 78}\nweight path: {label}\n{'=' * 78}", flush=True)
        sd = load_weights(MODEL, quant=quant, vision=False)
        model = create_model(MODEL, max_seq=MAX_SEQ, bench=False, weights=sd, paged=True)
        pool = model.enable_paging(num_blocks=blocks, block_size=BLOCK)
        pool.assign_table(list(range(blocks)), 0, row=model.prefill_row)

        print(f"  --- headline, paged, n={N}, chunk=2048 ---", flush=True)
        if model.mtp is None:
            model.mtp_prefill = False
            ms = _time(model, toks, 2048)
            print(f"  no MTP head {ms:>9.1f} ms {N / ms * 1e3:>8.0f} tok/s", flush=True)
            prof_off = None
        else:
            for flag in (True, False):
                model.mtp_prefill = flag
                ms = _time(model, toks, 2048)
                print(f"  mtp_prefill={str(flag):<5} {ms:>9.1f} ms "
                      f"{N / ms * 1e3:>8.0f} tok/s", flush=True)
            model.mtp_prefill = True
            prof_on = _profile(model, toks, 2048, f"{label}, mtp ON")
            model.mtp_prefill = False
            prof_off = _profile(model, toks, 2048, f"{label}, mtp OFF")
            _report_ab(prof_on, prof_off)

        del model, sd, pool
        torch.cuda.empty_cache()

    print("\nPREFILL_MTP_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
