"""Run the text engine (greedy) on the DCU; the checkpoint decides the model.

    python run_chat.py --raw "The capital of France is" -n 32
    python run_chat.py --chat "你好，介绍一下你自己" -n 64
    python run_chat.py --bench -n 128
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
    eos = 248044

    print("loading weights ...", flush=True)
    t0 = time.time()
    model = create_model(args.model, max_seq=args.max_seq, bench=not args.no_bench)
    info = describe(args.model)
    print(f"loaded {info['model']} ({info['arch']}) in {time.time()-t0:.1f}s", flush=True)
    graph = not args.no_graph

    if args.bench:
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
