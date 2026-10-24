"""Concurrent-request throughput against a running ``serve.py``.

Unlike ``bench_batch.py`` (which drives ``LLMEngine`` in-process), this talks to
the unix socket: it starts ``N`` client threads at the same time and measures how
fast the server retires them.  That is the number the service is judged on, and
the one batching is supposed to move.

    python3 serve.py --serve --max-num-seqs 8 --max-seq 4096 --num-blocks 512 \
        --sock /tmp/qwen35_batch.sock &
    python3 diag/bench_serve.py --sock /tmp/qwen35_batch.sock --clients 1,2,4,8
    # shared system prompt: per-client unique tails, radix reuses the prefix
    python3 diag/bench_serve.py --sock ... --tokens 0 --prefix-tokens 512

``--tokens N`` sends N raw token ids (use a small N to measure decode rather than
prefill); ``--prefix-tokens P`` sends a P-token shared prefix plus a 16-token
per-client tail, which is what prefix reuse is supposed to make cheap.
"""

from __future__ import annotations

import argparse
import json
import socket
import threading
import time

PROMPT = ("You are a careful assistant. Answer the question about the history of "
          "Paris concisely, and mention the century when it matters. Question: when "
          "was the city founded and by whom? ")


def _call(sock_path: str, req: dict) -> dict:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(1800.0)
    s.connect(sock_path)
    s.sendall((json.dumps(req) + "\n").encode())
    buf = b""
    while not buf.endswith(b"\n"):
        chunk = s.recv(65536)
        if not chunk:
            break
        buf += chunk
    s.close()
    return json.loads(buf.decode())


def _build_prompts(n: int, n_tokens: int, prefix_tokens: int) -> list[list[int]]:
    if n_tokens:
        ids = [1000 + (i % 5000) for i in range(n_tokens)]
        return [ids] * n
    if prefix_tokens:
        prefix = [1000 + (i % 5000) for i in range(prefix_tokens)]
        return [prefix + [9000 + 37 * k + i for i in range(16)] for k in range(n)]
    return [[1000 + i for i in range(16)]] * n


def run_clients(sock_path: str, prompts: list[list[int]], max_new: int
                ) -> tuple[float, int, list[float]]:
    """Fire every request simultaneously; return (wall_s, tokens, latencies)."""
    n = len(prompts)
    barrier = threading.Barrier(n)
    results: list[tuple[int, float]] = [None] * n  # type: ignore[list-item]

    def worker(i: int) -> None:
        barrier.wait()
        t0 = time.time()
        rep = _call(sock_path, {"op": "generate", "tokens": prompts[i],
                                "max_new": max_new, "temperature": 0.0})
        results[i] = (len(rep.get("tokens") or []), time.time() - t0)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0
    latencies = [r[1] for r in results]
    return wall, sum(r[0] for r in results), latencies


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sock", default="/tmp/qwen35_batch.sock")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--clients", default="1,2,4,8")
    ap.add_argument("--tokens", type=int, default=0,
                    help="send N raw token ids shared by every client (small N = "
                         "decode-dominated)")
    ap.add_argument("--prefix-tokens", type=int, default=0,
                    help="send a shared P-token prefix plus a unique 16-token tail")
    args = ap.parse_args()

    plan = [int(x) for x in args.clients.split(",") if x.strip()]
    for n in plan:
        prompts = _build_prompts(n, args.tokens, args.prefix_tokens)
        run_clients(args.sock, prompts, 8)              # untimed warmup
        wall, tokens, lat = run_clients(args.sock, prompts, args.max_new)
        print(f"clients={n:<2d} prompt={len(prompts[0]):>4d} tok  "
              f"{tokens:>5d} tok in {wall:6.2f}s = {tokens / wall:7.2f} tok/s  "
              f"(latency p50 {sorted(lat)[len(lat) // 2]:.2f}s, "
              f"max {max(lat):.2f}s)", flush=True)


if __name__ == "__main__":
    main()
