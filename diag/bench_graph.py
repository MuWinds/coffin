"""A/B the static-shape engine: plain step vs CUDA-graph replay.

Run:  python bench_graph.py [n]
"""

from __future__ import annotations

import sys
import time

import torch

import qwen35_engine as E

MODEL = "/root/private_data/models/Qwen3.5-4B"


def timed(model, n, tokens, use_graph):
    model.reset()
    t0 = time.perf_counter()
    for i in range(n):
        if use_graph:
            logits = model.decode_step(tokens[i % len(tokens)])
        else:
            logits = model.logits(tokens[i % len(tokens)])
        nxt = int(logits.argmax())
        tokens.append(nxt)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    print(f"device={torch.cuda.get_device_name(0)}")
    model = E.Qwen35(MODEL, max_seq=2048, bench=True)

    prompt = [9707, 11, 1879, 3671, 374]      # "Hello, my name is"
    for t in prompt:                          # warm up the plain path
        model.logits(t)
    torch.cuda.synchronize()

    ms_plain = timed(model, n, list(prompt), use_graph=False)
    print(f"plain : {ms_plain:6.2f} ms/token  ({1000/ms_plain:5.2f} tok/s)")

    # correctness: same greedy trajectory with and without the graph
    model.reset()
    a = []
    for i in range(20):
        tok = prompt[i] if i < len(prompt) else int(model.logits(tok).argmax())
        a.append(int(model.logits(tok).argmax()))

    model.capture()
    model.reset()
    b = []
    for i in range(20):
        tok = prompt[i] if i < len(prompt) else b[-1]
        b.append(int(model.decode_step(tok).argmax()))
    print(f"graph tokens: {b[:12]}")

    model.reset()
    c = []
    for i in range(20):
        tok = prompt[i] if i < len(prompt) else c[-1]
        c.append(int(model.logits(tok).argmax()))
    print(f"plain tokens: {c[:12]}")
    print(f"match: {b == c}")

    ms_graph = timed(model, n, list(prompt), use_graph=True)
    print(f"graph : {ms_graph:6.2f} ms/token  ({1000/ms_graph:5.2f} tok/s)")
    print(f"speedup: {ms_plain/ms_graph:.2f}x")


if __name__ == "__main__":
    main()
