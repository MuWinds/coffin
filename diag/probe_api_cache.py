"""Does a repeated prefix actually skip work, end to end through the API?

The agent pattern: turn 2 resends turn 1 and appends. If the radix prefix cache is
wired into the serving path, the second request must report
``usage.prompt_tokens_details.cached_tokens > 0`` and must be measurably faster --
that is the whole point of the paged pool, and it is invisible in the engine's own
checks (which never go through the HTTP layer).

    python diag/probe_api_cache.py [port]
"""

from __future__ import annotations

import json
import random
import sys
import time
import urllib.request

WORDS = ["the", "quick", "brown", "fox", "jumps", "over", "a", "lazy", "dog",
         "scnet", "dcu", "qwen", "engine", "context", "attention", "token",
         "prefix", "cache", "block", "radix"]


def post(port: int, messages: list, max_new: int = 8) -> dict:
    body = json.dumps({"model": "qwen3.5-4b", "messages": messages,
                       "max_completion_tokens": max_new, "temperature": 0.0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.load(r)
    out["_wall"] = time.time() - t0
    return out


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 18080
    rng = random.Random(0)
    long_text = " ".join(rng.choices(WORDS, k=3000))
    q_first = "\n\nQuestion: what is the very first word above? Answer with that word only."
    q_last = "Question: what is the very last word above? Answer with that word only."

    cold = post(port, [{"role": "user", "content": long_text + q_first}])
    warm = post(port, [{"role": "user", "content": long_text + q_first},
                       {"role": "assistant", "content": cold["choices"][0]["message"]["content"] or ""},
                       {"role": "user", "content": q_last}])

    def line(tag: str, out: dict) -> None:
        u = out["usage"]
        det = u.get("prompt_tokens_details") or {}
        print(f"{tag:<5} prompt {u['prompt_tokens']:>5} tok  cached {det.get('cached_tokens', 0):>5}"
              f"  wall {out['_wall']:6.2f}s  finish={out['choices'][0]['finish_reason']}"
              f"  answer={out['choices'][0]['message']['content']!r}")

    line("cold", cold)
    line("warm", warm)

    cached = (warm["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    speed = cold["_wall"] / warm["_wall"] if warm["_wall"] else 0.0
    ok = cached > 2000 and warm["_wall"] < cold["_wall"] * 0.5
    print(f"\ncached {cached} tokens, cold {cold['_wall']:.2f}s -> warm {warm['_wall']:.2f}s "
          f"({speed:.1f}x)  {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
