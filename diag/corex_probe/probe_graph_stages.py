"""Find which stage of a graph capture is unsafe, one bounded stage at a time.

A whole-step capture once hung until the scheduler killed the job, and a hang gives
no stack trace -- so the useful question is not "does the decode step capture" but
"which piece of it stops being safe". Answering that needs the experiment split so
each stage is individually bounded and the log says where it stopped.

**A hang can leave the device context unusable, so each stage is a separate job
with a short cap.** Running all five in one process would mean a hang in stage 1
costs the whole run *and* risks poisoning the context for every later stage. Pass
``--stage N`` and submit one small job per stage; each stage announces itself with
``flush=True`` before doing the work, so the last line in the log names the stage
that hung.

Stage ladder, each a strict superset of the previous:

1. one kernel, captured and replayed                     (known to work)
2. two kernels in sequence, captured and replayed
3. eager prefill + eager decode, **no graph**            (the control)
4. ``Model.capture()`` only, no replay
5. prefill then exactly ONE graph replay                 (this is where it stopped)

Module: ``coffin/diag``. Runs on the card, always under ``run_capped.sh``.

Constraints:
    * Always run under a hard cap. A device-side spin produces no output and no
      traceback, and the only way to bound it is a supervisor that kills the process.
    * Never run two stages in one job when a hang is plausible: the point of
      splitting is that a hang costs one short job, not the whole investigation.
    * ``capture()`` resets the model and runs its own prefill, so it must precede
      the real prefill. Capturing after a prefill wipes the KV cache and recurrent
      state, and a replay then drives attention over a window never written -- the
      exact mistake that produced the original hang.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

MODEL = os.environ.get("ENGINE_MODEL", "")


def _stage(n: int, text: str) -> None:
    """Announce a stage before running it.

    Printed first and flushed, because when the process hangs the last
    announcement is the only record of where it happened.

    Args:
        n: Stage number, for grepping the log.
        text: What the stage is about to do.
    """
    print(f"\n[stage {n}] {text}", flush=True)


def _ok(text: str) -> None:
    """Report a completed stage.

    Args:
        text: Result summary.
    """
    print(f"[stage ok] {text}", flush=True)


def _linear_stage() -> None:
    """Capture and replay a single projection.

    The smallest thing that can prove a capture works at all. If this hangs, the
    problem is in the kernel or the stream discipline, not in the model.
    """
    import torch

    import backends

    table = backends.ops
    n, k = 2560, 2560
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(k, dtype=torch.bfloat16, device="cuda")
    # Warm up so the allocator and any lazy handle exist before the capture.
    for _ in range(3):
        table.linear(w, x)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, pool=torch.cuda.graph_pool_handle()):
        out = table.linear(w, x)
    g.replay()
    torch.cuda.synchronize()
    _ok(f"replay finite={bool(torch.isfinite(out).all())} shape={tuple(out.shape)}")


def _two_op_stage() -> None:
    """Capture and replay a norm followed by a projection.

    Two kernels is the point where a capture starts to depend on op ordering and on
    intermediate buffers rather than on one kernel's own arguments.
    """
    import torch

    import backends

    table = backends.ops
    m, n = 4, 2560
    xs = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")
    gain = torch.randn(n, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(n, n, dtype=torch.bfloat16, device="cuda")
    for _ in range(3):
        table.linear(w, table.rms_norm(xs, gain, 0, 1e-6)[0])
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, pool=torch.cuda.graph_pool_handle()):
        h = table.rms_norm(xs, gain, 0, 1e-6)
        out = table.linear(w, h[0])
    g.replay()
    torch.cuda.synchronize()
    _ok(f"replay finite={bool(torch.isfinite(out).all())}")


def _eager_stage(model) -> None:
    """Prefill and decode with no graph involved at all.

    The control. If this is slow-but-correct, the graph is the only suspect; if this
    hangs too, the hang has nothing to do with capture.
    """
    import torch

    model.reset()
    t0 = time.time()
    logits = model.prefill([1, 2, 3, 4])
    torch.cuda.synchronize()
    _ok(f"4-token eager prefill in {(time.time() - t0) * 1000:.0f} ms")
    for _ in range(3):
        logits = model.logits(int(logits.argmax()))
    torch.cuda.synchronize()
    _ok(f"3 eager decode steps, finite={bool(torch.isfinite(logits).all())}")


def _capture_stage(model) -> None:
    """Call ``Model.capture()`` and stop there, without replaying.

    Split from the replay on purpose: capture and replay fail for different reasons,
    and a stage that does both cannot say which half hung.
    """
    import torch

    model.reset()
    t0 = time.time()
    model.capture()
    torch.cuda.synchronize()
    _ok(f"capture() returned in {time.time() - t0:.1f}s, graph_ready={model.graph_ready}")


def _replay_stage(model) -> None:
    """Prefill, then replay the decode graph exactly once.

    One replay, not a loop. A loop is what turns a slow-but-working replay into a
    wall-clock timeout with no information; a single step either returns or it does
    not, and the cap bounds the alternative.
    """
    import torch

    if not model.graph_ready:
        print("  graph not captured (run stage 4 first in the same process); skipping",
              flush=True)
        return
    model.reset()
    logits = model.prefill([1, 2, 3, 4, 5])
    nxt = int(logits.argmax())
    t0 = time.time()
    logits = model.decode_step(nxt)
    torch.cuda.synchronize()
    _ok(f"one replay in {(time.time() - t0) * 1000:.1f} ms, "
        f"finite={bool(torch.isfinite(logits).all())}")


def main() -> int:
    import argparse

    import backends

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", type=int, default=0,
                        help="run only this stage (0 = all, which is only safe when "
                             "no stage is expected to hang). One stage per job bounds "
                             "a hang to a short cap and keeps a wedged context from "
                             "contaminating the next measurement.")
    args = parser.parse_args()
    only = args.stage

    print("=" * 78, flush=True)
    print("graph staging diagnosis", flush=True)
    print("=" * 78, flush=True)
    print(f"  backend: {backends.status()}", flush=True)
    print(f"  model  : {MODEL}", flush=True)
    print(f"  stage  : {'all' if only == 0 else only}", flush=True)
    if only == 0:
        print("  WARNING: running every stage in one process. If a stage hangs, the "
              "context may be unusable for the rest -- use --stage N.", flush=True)
    if not MODEL or not Path(MODEL).is_dir():
        print("  ENGINE_MODEL unset or not a directory", flush=True)
        return 1

    from engine_api import create_model

    # Only stages 3-5 need the model; skipping the load keeps a kernel-only stage
    # from being blamed for a load-time problem (and saves a minute per job).
    model = None
    if only == 0 or only >= 3:
        _stage(0, "load the model (no graph involved)")
        t0 = time.time()
        model = create_model(MODEL, max_seq=1024, bench=True)
        _ok(f"loaded in {time.time() - t0:.1f}s")

    runners = {
        1: ("single kernel, capture + replay", _linear_stage, False),
        2: ("two kernels in sequence, capture + replay", _two_op_stage, False),
        3: ("eager prefill + eager decode, no graph (control)", _eager_stage, True),
        4: ("Model.capture() only, no replay", _capture_stage, True),
        5: ("prefill then exactly one graph replay", _replay_stage, True),
    }
    for n, (text, fn, needs_model) in runners.items():
        if only not in (0, n):
            continue
        _stage(n, text)
        if needs_model and model is None:
            print("  model not loaded; skipping", flush=True)
            continue
        try:
            fn(model) if needs_model else fn()
        except Exception:
            print("[stage FAIL]", flush=True)
            traceback.print_exc()

    print("\n[stages done]", flush=True)
    print("GRAPH_STAGES_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
