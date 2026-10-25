"""One end-to-end check of the case that used to fail: a long prompt via the API.

Answers whether a prompt beyond the old context limit is accepted *and read*. A
16k-token prompt exceeded the old ``max_seq=4096``; this sends one through the
OpenAI endpoint and asks a question whose answer can only come from the context,
so "it was accepted" and "the model actually read it" are checked together.

Read the ``answer`` line against the expected ``'the quick brown'``: a correct
answer proves the whole prompt reached the model, while a plausible but wrong one
means the prompt was silently truncated, which the ``prompt`` token count in the
line above exposes. An HTTP error instead of output means the request exceeded
``--max-seq`` on the server.

    python diag/probe_api_long.py [words] [port]

Module: ``coffin/diag``. Standalone probe: drives the OpenAI adapter over HTTP
and imports nothing from ``coffin/``.

Constraints:
    * The filler text comes from a fixed seed, so the prompt length and content are
      reproducible and two runs can be compared.
    * The tail question asks for the *first* three words on purpose: it can only be
      answered from the beginning of the context, which is the part a truncated
      prefill would drop.
    * ``max_tokens`` stays tiny (16) so the run measures prefill of a long prompt
      rather than generation.
"""

from __future__ import annotations

import json
import random
import sys
import time
import urllib.request

WORDS = ["the", "quick", "brown", "fox", "jumps", "over", "a", "lazy", "dog",
         "scnet", "dcu", "qwen", "engine", "context", "attention", "token"]
TAIL = "\n\nQuestion: what are the first three words of the text above? " \
       "Answer with those three words only."


def main() -> None:
    """Post one oversized prompt and print its token counts, wall time and answer."""
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16000
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 18080
    text = " ".join(random.Random(0).choices(WORDS, k=n)) + TAIL
    body = json.dumps({"model": "qwen3.5-4b",
                       "messages": [{"role": "user", "content": text}],
                       "max_tokens": 16, "temperature": 0.0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    print(f"posting ~{n} words ({len(text)} chars)...", flush=True)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.load(r)
    dt = time.time() - t0
    usage = out["usage"]
    pt = usage["prompt_tokens"]
    print(f"wall {dt:.1f}s  prompt {pt} tok ({pt / dt:.0f} tok/s)  "
          f"completion {usage['completion_tokens']} tok  "
          f"finish={out['choices'][0]['finish_reason']}")
    print(f"answer: {out['choices'][0]['message']['content']!r}  "
          f"(expected 'the quick brown')")


if __name__ == "__main__":
    main()
