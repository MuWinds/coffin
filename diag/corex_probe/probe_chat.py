"""Does the ported engine produce text on the CoreX card?

Two things can each be true alone and both be wrong: the kernels can match their
host oracle while the model still emits garbage (a wiring bug between ops), and
the unit suite can pass while the model never loads (the suite skips GPU cases).
So this runs the engine end to end -- load the real checkpoint through
``create_model``, prefill a prompt, decode greedily -- and prints the text.

The prompt is the one this engine has been checked with since the first version
("The capital of France is" -> " Paris."), so the expected answer is known without
a reference run: if it comes back as `` Paris.``, the whole stack is wired right.

Module: ``coffin/diag``. Runs on the card through the SCOW client.

Constraints:
    * ``COFFIN_BACKEND=corex`` and ``--no-graph``: the operator table is not yet
      all-native, so ``caps.graph`` is 0 and the runtime takes the eager substitute
      on its own. Passing ``--no-graph`` anyway keeps this probe's cost independent
      of that decision, so the same script works before and after the graph flips.
    * Greedy decoding only, so a run is reproducible and a difference means a bug
      rather than a sampler.
    * A small ``--max-seq``: this is a smoke test for wiring, not a long-context
      measurement, and a smaller KV pool is what keeps a stall in prefill from
      turning into a job that sits on the card for its full time limit.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

#: ``coffin/`` is the ``sys.path`` root: the engine imports its packages as
#: top-level names. Two levels up from ``diag/corex_probe/probe_chat.py``.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

#: The checkpoint, from the environment so this script carries no home path.
MODEL = os.environ.get("ENGINE_MODEL", "")

#: ``(label, prompt, tokens)`` cases. The first is the historical smoke test; the
#: second checks that the chat template and the stop handling still work, which is
#: a different code path from a raw continuation.
CASES = (
    ("raw continuation", "The capital of France is", 12),
    ("raw arithmetic", "2 + 2 =", 8),
)


def _sep(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)


def main() -> int:
    import backends
    import torch
    from tokenizers import Tokenizer

    from engine_api import create_model, load_weights, describe

    _sep("0. backend")
    print(" ", backends.status(), flush=True)
    for line in backends.describe_all():
        print("  ", line, flush=True)
    print(f"  model: {MODEL}", flush=True)
    if not MODEL or not Path(MODEL).is_dir():
        print("  ENGINE_MODEL is unset or not a directory", flush=True)
        return 1

    _sep("1. load")
    tok = Tokenizer.from_file(str(Path(MODEL) / "tokenizer.json"))
    t0 = time.time()
    try:
        # COFFIN_QUANT=1 loads the projections as int8, which is the only way to exercise
        # the W8A16 kernel end to end: the flag belongs to the loader, not the backend, so
        # a backend that supports int8 still builds a bf16 model without it.
        quant = os.environ.get("COFFIN_QUANT") == "1"
        weights = load_weights(MODEL, quant=quant) if quant else None
        model = create_model(MODEL, max_seq=1024, bench=True, weights=weights)
    except Exception:
        print("LOAD FAILED", flush=True)
        traceback.print_exc()
        return 1
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)
    print(f"  describe: {describe(MODEL)}", flush=True)
    print(f"  caps: graph={int(backends.caps().graph)} "
          f"native={len(backends.caps().native)}/{len(backends.ops.__class__.NAMES)}",
          flush=True)

    failures = []
    for label, prompt, n in CASES:
        _sep(f"2. {label}: {prompt!r}")
        try:
            ids = tok.encode(prompt).ids
            # Order matters and is not free to rearrange: `capture()` starts by
            # resetting the model and then runs its own dummy prefill, so capturing
            # *after* a real prefill wipes the prompt's KV cache and recurrent state
            # and leaves `decode_step` replaying against a state that never saw the
            # prompt. The attention kernels read their key-loop bound from device
            # memory, so a wrecked state shows up as a very long spin rather than an
            # error -- which is exactly how an earlier version of this probe burned
            # a job's whole wall clock. This is the order `run_chat.generate` uses.
            model.reset()
            if backends.caps().graph and not model.graph_ready:
                t0 = time.time()
                try:
                    model.capture()
                    print(f"  captured decode graph in {(time.time() - t0) * 1000:.0f} ms",
                          flush=True)
                except Exception as exc:              # noqa: BLE001 - measurement either way
                    print(f"  capture FAILED: {type(exc).__name__}: {exc}", flush=True)
                model.reset()

            step = model.decode_step if model.graph_ready else model.logits
            print(f"  decode path: {'graph replay' if model.graph_ready else 'eager'}",
                  flush=True)

            t0 = time.time()
            logits = model.prefill(ids)
            prefill_s = time.time() - t0

            t0 = time.time()
            out = list(ids)
            for _ in range(n):
                nxt = int(logits.argmax())
                out.append(nxt)
                logits = step(nxt)
            decode_s = time.time() - t0
            text = tok.decode(out)
            finite = bool(torch.isfinite(logits).all())
            print(f"  text     : {text!r}", flush=True)
            print(f"  prefill  : {prefill_s * 1000:.1f} ms ({len(ids)} tokens)", flush=True)
            print(f"  decode   : {decode_s * 1000 / max(n, 1):.2f} ms/step "
                  f"({n / max(decode_s, 1e-9):.2f} tok/s)", flush=True)
            print(f"  logits finite: {finite}", flush=True)
            if not finite:
                failures.append(f"{label}: non-finite logits")
        except Exception:
            print("  FAILED", flush=True)
            traceback.print_exc()
            failures.append(f"{label}: raised")

    _sep("summary")
    print(f"  backend : {backends.caps().name} "
          f"({len(backends.caps().native)}/{len(backends.ops.__class__.NAMES)} native)",
          flush=True)
    print(f"  failures: {failures or '(none)'}", flush=True)
    print("\nCOREX_CHAT_DONE", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
