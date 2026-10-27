"""Does the rewritten delta kernel survive graph capture? One question, staged.

The eager paths all pass: the 18-op conformance check reports ``delta rel=3.270e-07`` at
101.5 GB/s, and a profiled decode step runs at 37.98 ms. The end-to-end probe printed
``captured decode graph in 2970 ms`` and then stopped producing output, which means the
capture itself succeeded and something after it -- the dummy prefill, the prompt prefill,
or the first replay -- wedged.

Those three are separable by printing before each one, and the answer decides where to
look: a hang in replay means the captured launch arguments differ from what the graph
recorded (a kernel that reads its grid or block size back is the usual cause); a hang in
prefill means the graph-legal path is fine and something about the wider launch is not.

The kernel change under suspicion computes ``j`` from ``blockIdx.y * blockDim.x +
threadIdx.x``. That is stable across a replay, but the launch now picks ``threads`` from
``dv``, so a shape whose ``dv`` differs between capture and replay would record one block
size and replay another. Nothing in the model varies ``dv``, so the probe also prints the
sizes it sees, in case the assumption is wrong.

Module: ``coffin/diag``. Runs on the card.

Constraints:
    * Every stage prints before it runs, so a hang names its own stage.
    * ``CUDA_LAUNCH_BLOCKING`` is honoured when set: making launches synchronous is the
      cheapest way to tell an asynchronous wedge from a synchronisation bug.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

MODEL = os.environ.get("ENGINE_MODEL", "")


def main() -> int:
    import torch

    import backends

    print(f"  backend: {backends.status()}", flush=True)
    print(f"  CUDA_LAUNCH_BLOCKING={os.environ.get('CUDA_LAUNCH_BLOCKING', '(unset)')}",
          flush=True)
    if not MODEL or not Path(MODEL).is_dir():
        print("  ENGINE_MODEL unset or not a directory", flush=True)
        return 1

    from engine_api import create_model

    model = create_model(MODEL, max_seq=1024, bench=False)
    print(f"  loaded, caps.graph={int(backends.caps().graph)}", flush=True)

    # The delta state's shape, read from the live model rather than assumed: the kernel's
    # block size is derived from ``dv``, so a value other than the expected 128 would
    # change the recorded launch.
    for name, attr in (("state", "state"),):
        mod = getattr(model, "layers", None) or getattr(model, "model", None)
        print(f"  model attrs: {type(model).__name__}", flush=True)
        del name, attr, mod

    print("\n[A] reset", flush=True)
    model.reset()
    torch.cuda.synchronize()
    print("[A] ok", flush=True)

    print("[B] capture (calls reset + dummy prefill internally)", flush=True)
    t0 = time.time()
    model.capture()
    torch.cuda.synchronize()
    print(f"[B] ok in {(time.time() - t0) * 1000:.0f} ms, "
          f"graph_ready={model.graph_ready}", flush=True)

    print("[C] reset after capture", flush=True)
    model.reset()
    torch.cuda.synchronize()
    print("[C] ok", flush=True)

    print("[D] real prefill", flush=True)
    t0 = time.time()
    logits = model.prefill([1, 2, 3, 4, 5])
    torch.cuda.synchronize()
    print(f"[D] ok in {(time.time() - t0) * 1000:.1f} ms, "
          f"finite={bool(torch.isfinite(logits).all())}", flush=True)

    print("[E] one replay step", flush=True)
    t0 = time.time()
    nxt = int(logits.argmax())
    logits = model.decode_step(nxt)
    torch.cuda.synchronize()
    print(f"[E] ok in {(time.time() - t0) * 1000:.1f} ms, "
          f"finite={bool(torch.isfinite(logits).all())}", flush=True)

    print("[F] four more replay steps", flush=True)
    tokens = [nxt]
    for i in range(4):
        print(f"  [F{i}] argmax", flush=True)
        nxt = int(logits.argmax())
        tokens.append(nxt)
        logits = model.decode_step(nxt)
        torch.cuda.synchronize()
        print(f"  [F{i}] ok, finite={bool(torch.isfinite(logits).all())}", flush=True)
    print(f"[F] ok, tokens={tokens}", flush=True)

    print("\nGRAPH_DELTA_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
