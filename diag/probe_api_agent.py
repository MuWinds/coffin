"""One request that reproduces both faults the agent clients hit.

Answers whether the two OpenAI-compatibility faults that broke agent clients are
fixed, in a single request each:

1. a ``developer``-role message (OpenAI's newer system role) used to make the
   checkpoint's template raise *"Unexpected message role."* and fail the request;
2. a budget sent as ``max_completion_tokens`` used to be ignored, falling back to
   256 tokens -- which truncates exactly what a long-context model is asked for.

The prompt asks for a long file of code, so "did it get the whole answer" is
visible in the completion length and the finish reason.

Read the two boolean lines under the first request: ``role accepted`` must be
``True`` or the request never reached the model, and ``beyond the old 256
default`` must be ``True`` or the budget is still being dropped — a completion of
256 with ``finish=length`` is that failure exactly. The final block compares
streamed against non-streamed text for the same prompt; a mismatch prints both
strings, which is the incremental decoder's bug rather than the protocol's.

    python diag/probe_api_agent.py [port]

Module: ``coffin/diag``. Standalone probe: drives ``openai_server.py`` over HTTP
and imports nothing from ``coffin/``.

Constraints:
    * The port defaults to 18080 because that is the deployment's server port; the
      optional argument exists so a second instance can be probed.
    * ``temperature`` is 0 so a truncation is attributable to the budget and not
      to sampling; a nonzero value makes the length check meaningless.
    * ``timeout`` is an hour by design: a 1200-token completion on a cold DCU is
      slow, and a short timeout would read as a protocol failure.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request

ASK = ("Write a complete, self-contained HTML file that animates an SVG bicycle "
       "with rotating wheels and moving pedals. Output the whole file, no "
       "explanation.")


def post(port: int, body: dict, timeout: float = 3600.0) -> dict:
    """Send one non-streamed chat-completion request and return the parsed reply.

    Args:
        port: Server port on localhost.
        body: OpenAI request body.
        timeout: Seconds to wait; long by default because a long completion on a
            cold engine is slow and a timeout would look like a protocol failure.

    Returns:
        The parsed reply with an extra ``_wall`` key holding the request's wall
        seconds. ``_wall`` is added here rather than at each call site so the
        timing is measured the same way for every request.
    """
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    out["_wall"] = time.time() - t0
    return out


def main() -> None:
    """Send the developer-role request, a plain control, and a stream/plain comparison."""
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 18080

    body = {"model": "qwen3.5-4b",
            "messages": [{"role": "developer", "content": "You are a coding agent."},
                         {"role": "user", "content": ASK}],
            "max_completion_tokens": 1200,
            "temperature": 0.0}
    out = post(port, body)
    u = out["usage"]
    msg = out["choices"][0]["message"]
    text = msg.get("content") or ""
    print(f"[developer role + max_completion_tokens=1200]")
    print(f"  finish={out['choices'][0]['finish_reason']}  completion={u['completion_tokens']}"
          f"  prompt={u['prompt_tokens']}  wall={out['_wall']:.1f}s")
    print(f"  {len(text)} chars, starts {text[:60]!r} ends {text[-40:]!r}")
    ok_roles = "developer" not in json.dumps(out)
    ok_len = u["completion_tokens"] > 256
    print(f"  role accepted: {ok_roles}   beyond the old 256 default: {ok_len}")

    body2 = {"model": "qwen3.5-4b",
             "messages": [{"role": "user", "content": "Name three primes."}],
             "max_tokens": 64}
    out2 = post(port, body2)
    print(f"[plain request, max_tokens=64] finish={out2['choices'][0]['finish_reason']} "
          f"completion={out2['usage']['completion_tokens']} "
          f"content={(out2['choices'][0]['message'].get('content') or '')[:60]!r}")

    # streaming must produce the same text as the non-streaming path (the agent
    # clients all stream, and the incremental decoder is new)
    ask3 = "Write a haiku about winter, with Chinese characters."
    body3 = {"model": "qwen3.5-4b",
             "messages": [{"role": "user", "content": ask3}], "max_tokens": 96}
    once = (post(port, body3)["choices"][0]["message"].get("content") or "")
    chunks = []
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps({**body3, "stream": True}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r.read().decode().splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                delta = json.loads(line[6:])["choices"][0]["delta"]
                chunks.append(delta.get("content") or "")
    streamed = "".join(chunks)
    same = streamed.strip() == once.strip()
    print(f"[stream vs non-stream] {len(chunks)} chunks  identical={same}")
    if not same:
        print(f"  once    : {once[:120]!r}")
        print(f"  streamed: {streamed[:120]!r}")


if __name__ == "__main__":
    main()
