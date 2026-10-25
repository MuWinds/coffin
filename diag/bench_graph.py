"""A/B the static-shape engine: plain step vs CUDA-graph replay.

Answers what graph capture buys a single-sequence decode step, and whether it
buys it without changing the output. That second half is the reason this is an
A/B and not just a timing run: capture replays fixed addresses, so a stale
buffer or a pointer captured before a reset shows up as a diverging token
trajectory rather than a crash.

Read in this order. ``match: True`` first — if the graph and plain trajectories
differ, the ``speedup`` below it is a speedup of doing the wrong thing and the
number should be ignored. Then ``speedup``: the gap is the host submission time
removed by capture, so a small gap means this step is already device-bound and
the graph is not the lever.

Run inside the container from the engine root, on the DCU:

    python bench_graph.py [n]

``n`` is the number of decode tokens per timed run, default 50.

Module: ``coffin/diag``. Standalone probe: imports ``models.qwen35`` and torch,
and nothing under ``coffin/`` imports it.

Constraints:
    * ``model.capture()`` must run before the graph measurement; an uncaptured
      model would silently re-time the plain path and report a 1.00x speedup.
    * Both paths are warmed before timing, because the plain path's first call
      pays rocBLAS handle creation and would make capture look better than it is.
    * Needs the real checkpoint, so it runs only where the weights are.
"""

from __future__ import annotations

import sys
import time

import torch

from models.qwen35 import Qwen35

MODEL = "/root/private_data/models/Qwen3.5-4B"


def timed(model, n, tokens, use_graph):
    """Decode ``n`` tokens through one path and return milliseconds per token.

    Args:
        model: A loaded model, already captured when ``use_graph`` is set.
        n: Token count to decode; the wall span is divided by this.
        tokens: Seed prompt; the method appends each generated token, so the
            run advances a real greedy trajectory instead of repeating one id.
        use_graph: Replay the captured step instead of the eager path.

    Returns:
        Mean milliseconds per token over the whole run, including host time.
    """
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
    """Print the plain and graph token rates, their ratio, and a trajectory match check."""
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    print(f"device={torch.cuda.get_device_name(0)}")
    model = Qwen35(MODEL, max_seq=2048, bench=True)

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
