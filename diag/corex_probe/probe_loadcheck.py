"""Where does the model load stop? One question, bounded, per-run.

An end-to-end run began hanging at load time -- before the first "loaded in" line --
while the same run had completed in seconds an hour earlier. Between the two, the
kernels gained closing barriers in their reducers. A barrier added inside a reducer
that is called from a *divergent* context (only some threads reach the call) blocks
forever: the waiting threads never see the others arrive, and the device spins
without producing a traceback. That is the shape of the symptom, so the probe tests
it directly instead of guessing.

The split that localises it:

* ``bench=False`` skips load-time kernel selection, so if that path completes the
  hang is inside the autotune sweep rather than in weight loading.
* ``bench=True`` runs the sweep, which calls every op on the model's real shapes --
  precisely the shapes ``check_ops.py`` never used.

Runs under a hard cap (``launch_loadcheck.sh``), because a barrier deadlock is a
device-side spin: no exception, no stderr, and without a supervisor it consumes the
whole job.

Module: ``coffin/diag``. Diagnostic only.

Constraints:
    * Never call this without a cap: the failure mode is a hang, not an error.
    * One mode per invocation (``--bench`` or not) so a hang names its own cause
      instead of leaving "which of the two hung" ambiguous.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

MODEL = os.environ.get("ENGINE_MODEL", "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bench", action="store_true",
                        help="run load-time kernel selection (the suspected hang)")
    parser.add_argument("--max-seq", type=int, default=1024)
    args = parser.parse_args()

    import torch

    import backends

    print(f"  backend : {backends.status()}", flush=True)
    print(f"  model   : {MODEL}", flush=True)
    print(f"  bench   : {args.bench}", flush=True)
    print(f"  max_seq : {args.max_seq}", flush=True)
    if not MODEL or not Path(MODEL).is_dir():
        print("  ENGINE_MODEL unset or not a directory", flush=True)
        return 1

    from engine_api import create_model

    print("  [1] importing done; calling create_model", flush=True)
    t0 = time.time()
    try:
        model = create_model(MODEL, max_seq=args.max_seq, bench=args.bench)
    except Exception:
        print("  LOAD_RAISED", flush=True)
        traceback.print_exc()
        return 1
    print(f"  [2] create_model returned in {time.time() - t0:.1f}s", flush=True)

    # A load that returns is not proof the kernels are healthy: the sweep may have
    # skipped a shape. One tiny forward is the cheapest way to exercise the real
    # call graph once.
    print("  [3] eager prefill of 5 tokens", flush=True)
    t0 = time.time()
    try:
        model.reset()
        logits = model.prefill([1, 2, 3, 4, 5])
        torch.cuda.synchronize()
        print(f"  [4] prefill ok in {time.time() - t0:.1f}s, "
              f"finite={bool(torch.isfinite(logits).all())}", flush=True)
        nxt = int(logits.argmax())
        logits = model.logits(nxt)
        torch.cuda.synchronize()
        print(f"  [5] one eager decode step ok, "
              f"finite={bool(torch.isfinite(logits).all())}", flush=True)
    except Exception:
        print("  FORWARD_RAISED", flush=True)
        traceback.print_exc()
        return 1

    print("LOADCHECK_OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
