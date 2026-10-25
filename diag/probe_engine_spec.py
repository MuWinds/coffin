"""Engine-level check of batch-friendly MTP: speculation must not change tokens.

Builds a dense model first for the greedy reference, then the batched engine with
``spec_k=1, spec_batch=True`` and runs the same prompts concurrently.  Every row's
tokens must equal the dense greedy reference: speculation only changes *how fast*
the tokens are produced, never which ones.

    python3 diag/probe_engine_spec.py                     # default model dir
    QWEN_MODEL=/root/private_data/models/Qwen3.5-4B python3 diag/probe_engine_spec.py

Module: ``coffin/diag``. End-to-end check over ``engine_api`` / ``llm_engine``;
needs the checkpoint and a DCU.

Constraints:
    * The dense greedy run is the oracle: speculation may only change speed, so
      any token divergence is a bug in the verify path, not noise.
    * Both engines run in one process, which is why only one model is resident at
      a time -- two of them do not fit on the card.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import engine_api  # noqa: E402
from config import EngineConfig  # noqa: E402
from llm_engine import LLMEngine  # noqa: E402
from sequence import SamplingParams  # noqa: E402

MODEL_DIR = os.environ.get("QWEN_MODEL", "/root/private_data/models/Qwen3.5-4B")
PROMPTS = [
    ["The quick brown fox jumps over the lazy dog."],
    ["In 1789 the French Revolution began, and the century that followed reshaped "
     "European politics."],
    ["def fibonacci(n):\n    return n if n < 2 else fibonacci(n - 1) + fibonacci(n - 2)"],
]
NEW = 4


def _ids(tok, text: str) -> list[int]:
    return tok.encode(text)


def main() -> None:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    prompts = [_ids(tok, p[0]) for p in PROMPTS]
    weights = engine_api.load_weights(MODEL_DIR, quant=True)

    # 1) dense, single-stream greedy reference
    dense = engine_api.create_model(MODEL_DIR, max_seq=1024, bench=False,
                                    weights=weights, paged=False)
    dense.capture()
    refs = []
    for prompt in prompts:
        dense.reset()
        logits = dense.prefill(prompt)
        out = []
        for _ in range(NEW):
            token = int(logits.argmax())
            out.append(token)
            logits = dense.decode_step(token)
        refs.append(out)
    dense.close()

    # 2) batched engine with speculation on
    paged = engine_api.create_model(MODEL_DIR, max_seq=1024, bench=False,
                                    weights=weights, paged=True)
    cfg = EngineConfig(model_dir=MODEL_DIR, max_seq=1024, bench=False, quant=True,
                       prefix_cache=False, block_size=64, num_blocks=1024,
                       spec_k=1, spec_batch=True, max_num_seqs=4)
    engine = LLMEngine(cfg, paged)
    engine.build()
    print(f"spec enabled: {engine.runner.spec} k={engine.runner.spec_k}", flush=True)

    seqs = [engine.add_request(p, SamplingParams(max_tokens=NEW, temperature=0.0))
            for p in prompts]
    out: list[list[int]] = [[] for _ in seqs]
    t0 = time.time()
    while not all(s.is_finished for s in seqs):
        for seq, token, _done in engine.step():
            for i, target in enumerate(seqs):
                if seq is target and token is not None:
                    out[i].append(token)
    wall = time.time() - t0

    ok = out == refs
    for i, (got, want) in enumerate(zip(out, refs)):
        print(f"row {i}: {'OK ' if got == want else 'DIFF'} {got} vs {want}", flush=True)
    produced = sum(len(o) for o in out)
    print(f"{produced} tokens in {wall:.2f}s = {produced / wall:.1f} tok/s "
          f"(spec, batched, {len(seqs)} rows)", flush=True)
    print("PASS" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
