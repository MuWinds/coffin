"""Does a repeated prefix actually skip work, end to end through the API?

Answers whether prefix reuse survives the HTTP layer, which the engine's own
checks cannot see. The agent pattern: turn 2 resends turn 1 and appends. If the
radix prefix cache is wired into the serving path, the second request must report
``usage.prompt_tokens_details.cached_tokens > 0`` and must be measurably faster --
that is the whole point of the paged pool, and it is invisible in the engine's own
checks (which never go through the HTTP layer).

Read the ``cached`` column on the warm row: above 2000 tokens and a wall time
under half the cold run is a pass, and both are required — a high ``cached`` count
with no speedup means the tokens were reported as reused but the prefill still
ran. The ``answer`` on both rows should name the same word, which is what proves
the reused KV actually carries the text.

    python diag/probe_api_cache.py [port]

Module: ``coffin/diag``. Standalone probe: drives the OpenAI adapter over HTTP
and imports nothing from ``coffin/``.

Constraints:
    * The long text is generated from a fixed seed, so cold and warm requests send
      byte-identical prefixes; a different prefix each run would make the cache
      miss and the probe report a false failure.
    * ``max_completion_tokens`` is small (8) so the measurement is prefill and
      caching, not decode; a long completion would swamp the speedup ratio.
    * ``temperature`` is 0 so the warm request's own content, which is fed back
      into the prompt, is reproducible.
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
    """Send one chat-completion request and return the parsed reply with its wall time.

    Args:
        port: Server port on localhost.
        messages: Full message list, so the caller controls exactly what prefix
            the second request resends.
        max_new: Completion budget. Small on purpose: this probe measures prefill
            and cache reuse, and a long completion would dominate the wall time.

    Returns:
        The parsed reply with an added ``_wall`` of wall seconds.
    """
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
    """Send a cold request, then a warm one that resends and extends it, and compare."""
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
        """Print one request's token counts, wall time and answer.

        Args:
            tag: ``cold`` or ``warm``.
            out: Parsed reply from ``post``. ``cached_tokens`` is read from the
                OpenAI usage detail block, which is the only place a client can
                see prefix reuse.
        """
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
