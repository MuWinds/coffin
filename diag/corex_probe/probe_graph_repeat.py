"""Is the graph replay failure a per-run race, or state corruption across iterations?

Two runs of identical code on the same node disagreed: one printed its text after
``decode path: graph replay``, the other stopped there with no traceback and no stderr
-- a device-side spin. Same commit, same card, same 3-second capture. That is either a
race inside one replay or damage that carries over from one decode step to the next,
and the two need different fixes.

One process, repeated attempts, one line of output per attempt. The *index of the last
completed iteration* is the answer:

* failing inside iteration 0 only, with success on some runs, is a race;
* failing at a consistent index greater than zero is cumulative corruption, which
  points at a buffer the replay writes and a later step reads (the stack's in-place
  recurrent state or the KV pool);
* failing at a random index is an allocator/address problem.

Every iteration is announced before the work it introduces, so a hang names its own
iteration rather than leaving it to be inferred.

Module: ``coffin/diag``. Runs on the card.

Constraints:
    * ``CUDA_LAUNCH_BLOCKING`` is honoured when set, because the first question about an
      asynchronous failure is whether making every launch synchronous changes it.
    * No wall-clock cap inside the process: the caller's job limit is the cap, and a
      timeout here is a data point rather than an error.
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

#: Attempts to run. Enough to expose a race that reproduces roughly half the time, small
#: enough that a hang does not consume the whole job.
ATTEMPTS = int(os.environ.get("ATTEMPTS", "6"))


def main() -> int:
    import torch

    import backends

    print(f"  backend      : {backends.status()}", flush=True)
    print(f"  model        : {MODEL}", flush=True)
    print(f"  CUDA_LAUNCH_BLOCKING: {os.environ.get('CUDA_LAUNCH_BLOCKING', '(unset)')}",
          flush=True)
    print(f"  attempts     : {ATTEMPTS}", flush=True)
    if not MODEL or not Path(MODEL).is_dir():
        print("  ENGINE_MODEL unset or not a directory", flush=True)
        return 1

    from engine_api import create_model

    t0 = time.time()
    model = create_model(MODEL, max_seq=1024, bench=True)
    print(f"  loaded in {time.time() - t0:.1f}s, caps.graph={int(backends.caps().graph)}",
          flush=True)
    if not backends.caps().graph:
        print("  caps.graph is False; nothing to test. Set _GRAPH_VERIFIED to get here.",
              flush=True)
        return 2

    prompt = [0, 1, 2, 3, 4]
    decode_steps = 4
    completed = []
    for attempt in range(ATTEMPTS):
        print(f"\n[attempt {attempt}] reset + capture", flush=True)
        try:
            model.reset()
            t0 = time.time()
            model.capture()
            cap_s = time.time() - t0
            print(f"  capture ok in {cap_s:.1f}s  graph_ready={model.graph_ready}",
                  flush=True)

            # The order that produced both outcomes in the field: capture, reset,
            # prefill, then replay.
            model.reset()
            print(f"  prefill {len(prompt)} tokens", flush=True)
            logits = model.prefill(prompt)
            torch.cuda.synchronize()
            print(f"  prefill ok, logits finite={bool(torch.isfinite(logits).all())}",
                  flush=True)

            tokens = []
            for step in range(decode_steps):
                print(f"  replay step {step}", flush=True)
                nxt = int(logits.argmax())
                tokens.append(nxt)
                logits = model.decode_step(nxt)
                torch.cuda.synchronize()
            print(f"  decode ok, tokens={tokens} "
                  f"finite={bool(torch.isfinite(logits).all())}", flush=True)
            completed.append(attempt)
        except Exception:
            print(f"[attempt {attempt}] RAISED", flush=True)
            traceback.print_exc()

    print(f"\n  completed attempts: {completed} of {ATTEMPTS}", flush=True)
    if len(completed) == ATTEMPTS:
        print("  ALL_ATTEMPTS_OK", flush=True)
    print("  (a missing 'completed attempts' line below the last [attempt N] means the")
    print("   process hung inside that attempt)")
    print("\nGRAPH_REPEAT_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
