"""Single-box entry point: load one checkpoint, generate greedily, print it.

This is the debugging front door for one request on one card -- no server, no
scheduler, no paged cache. It exists to answer "does this build still produce
text, and how fast", which is the check to run after a kernel change; the
serving path is ``serve.py`` and is what any real deployment uses.

    python run_chat.py --raw "The capital of France is" -n 32
    python run_chat.py --chat "introduce yourself" -n 64
    python run_chat.py --bench -n 128

Module: ``coffin/run_chat.py``. Top of the import graph, a sibling of
``serve.py``: it imports ``engine_api`` (the model-facing surface) and reads the
tokenizer from the checkpoint directory, and no engine module imports it back.
Model selection is delegated to ``create_model``, so this script never names a
model family -- the checkpoint's ``config.json`` decides.

Constraints:
    * Decoding is greedy, ``argmax`` with no sampler. That is what makes the
      same prompt reproducible run to run, which is the point of a smoke test;
      ``--bench`` numbers are compared against earlier runs and a sampler would
      make them meaningless.
    * ``--bench`` and ``--no-graph`` change the measured path itself, not just
      the output: the decode loop either replays the captured graph or calls
      ``logits`` eagerly, and those are different costs.
    * Nothing here is thread-safe or re-entrant. One process, one model, one
      request at a time; concurrency is ``serve.py``'s job.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from engine_api import Model, create_model, describe

#: Deployment default; any checkpoint registered in engine_api works.
MODEL = os.environ.get("ENGINE_MODEL", "/root/private_data/models/Qwen3.5-4B")


def build_prompt(tok: Tokenizer, args) -> list[int]:
    """Turn the CLI's ``--raw``/``--chat``/``--system`` into prompt token ids.

    The tags are encoded with the tokenizer rather than copied as literals,
    because ``<|im_start|>``/``<|im_end|>`` are single special tokens in this
    vocabulary and string concatenation would tokenize them as ordinary text.

    Args:
        tok: The checkpoint's tokenizer.
        args: Parsed CLI arguments; ``raw`` bypasses the template, ``chat`` and
            ``system`` fill it.

    Returns:
        The prompt ids, ending in the open ``assistant`` turn so generation
        continues the assistant reply rather than starting a new message.

    Raises:
        AttributeError: Neither ``raw`` nor ``chat`` was given; the resulting
            ``None`` fails on concatenation. A missing prompt is a CLI mistake,
            so failing here beats sending the model an empty one.
    """
    if args.raw:
        return tok.encode(args.raw).ids
    im_start = tok.encode("<|im_start|>").ids
    im_end = tok.encode("<|im_end|>").ids
    ids: list[int] = []
    if args.system:
        ids += im_start + tok.encode("system\n" + args.system).ids + im_end + tok.encode("\n").ids
    ids += im_start + tok.encode("user\n" + args.chat).ids + im_end + tok.encode("\n").ids
    ids += im_start + tok.encode("assistant\n").ids
    return ids


@torch.inference_mode()
def generate(model: Model, tok: Tokenizer, prompt: list[int], n: int, eos: int,
             graph: bool = True) -> tuple[list[int], float, float]:
    """Prefill ``prompt`` and decode greedily for at most ``n`` tokens.

    Two decode paths, and the choice is made per call: ``decode_step`` only
    exists once a graph has been captured, so the first call captures it and
    every token after that replays it. With ``graph=False`` each step calls
    ``logits`` eagerly -- slower, but it exercises the code the capture hides,
    which is why ``--no-graph`` exists as a diagnostic.

    Args:
        model: A loaded model; its state is consumed and left dirty.
        tok: Tokenizer for the checkpoint, kept for callers that decode with it.
        prompt: Prompt ids to prefill.
        n: Maximum number of tokens to generate, ``eos`` included.
        eos: Stop id; generation ends without emitting it.
        graph: Use the captured decode graph when the model can provide one.

    Returns:
        ``(tokens, prefill_s, decode_s)``: the generated ids with the stop token
        excluded, and the two wall-clock times reported separately because a
        prefill-bound change and a decode-bound change look identical in a
        combined number.
    """
    model.reset()
    if graph and not model.graph_ready:
        model.capture()
        model.reset()
    step = model.decode_step if (graph and model.graph_ready) else model.logits

    t0 = time.time()
    logits = model.prefill(prompt)       # batched GEMMs over the whole prompt
    t_prefill = time.time() - t0

    out: list[int] = []
    t1 = time.time()
    for _ in range(n):
        nxt = int(logits.argmax())
        if nxt == eos:
            break
        out.append(nxt)
        logits = step(nxt)
    dt = time.time() - t1
    return out, t_prefill, dt


def main() -> None:
    """Parse the CLI, load the checkpoint and print either a reply or timings.

    ``--bench`` runs a fixed prompt and reports throughput instead of the text,
    so successive builds are compared on the same input.

    Raises:
        OSError: The checkpoint directory is missing its ``tokenizer.json`` or
            weights; point ``--model`` at a complete checkpoint.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--raw", help="raw prompt (no chat template)")
    ap.add_argument("--chat", help="user message (chat template)")
    ap.add_argument("--system", default=None)
    ap.add_argument("-n", "--max-new", type=int, default=64)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--max-seq", type=int, default=4096)
    ap.add_argument("--no-bench", action="store_true", help="skip per-shape GEMV method picking")
    ap.add_argument("--no-graph", action="store_true", help="disable the CUDA-graph decode path")
    args = ap.parse_args()

    tok = Tokenizer.from_file(str(Path(args.model) / "tokenizer.json"))

    # ``<|im_end|>`` of this checkpoint. Unlike the server, which derives the
    # stop set from the tokenizer, this smoke test hard-codes the one id it
    # needs: a wrong value here shows up as a reply that runs past its turn,
    # which is exactly the regression to notice.
    eos = 248044

    print("loading weights ...", flush=True)
    t0 = time.time()
    model = create_model(args.model, max_seq=args.max_seq, bench=not args.no_bench)
    info = describe(args.model)
    print(f"loaded {info['model']} ({info['arch']}) in {time.time()-t0:.1f}s", flush=True)
    graph = not args.no_graph

    if args.bench:
        # Fixed prompt on purpose: the token count is part of the reported
        # tok/s, so a varying prompt would make successive runs incomparable.
        prompt = tok.encode("The quick brown fox").ids
        out, tp, dt = generate(model, tok, prompt, args.max_new, eos, graph)
        print(f"prefill: {len(prompt)} tokens in {tp*1000:.0f}ms ({len(prompt)/tp:.0f} tok/s)")
        print(f"decode : {len(out)} tokens in {dt:.2f}s = {len(out)/dt:.2f} tok/s"
              f"  (graph={graph and model.graph_ready})")
        return

    prompt = build_prompt(tok, args)
    print(f"prompt tokens: {len(prompt)}")
    out, tp, dt = generate(model, tok, prompt, args.max_new, eos, graph)
    text = tok.decode(out, skip_special_tokens=False)
    print("output:")
    print(text)
    print(f"prefill {len(prompt)} tok in {tp*1000:.0f}ms ({len(prompt)/tp:.0f} tok/s); "
          f"decode {len(out)} tok in {dt:.2f}s ({len(out)/dt:.2f} tok/s)"
          f"  (graph={graph and model.graph_ready})")


if __name__ == "__main__":
    main()
