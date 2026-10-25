"""``tool_choice`` against the *real* model, not the protocol stub.

Answers whether the OpenAI wire semantics of ``tool_choice`` survive contact with
an actual sampled decode. The definition says ``required`` means "the model
**must** call one or more tools" and a named choice "forces the model to call
that tool". The interesting cases are therefore the ones where the model would
rather answer in prose: with ``auto`` it talks, with ``required`` the API must
hand back a call anyway. A prompt that begs for a tool would pass every case and
prove nothing, which is why ``PROSE`` asks for a joke.

Cases A-F must each print ``OK``; ``MISMATCH`` means the server returned the
wrong shape (no call under ``required``, the wrong function name, or a
``finish_reason`` that disagrees with whether a call came back). Case G checks
the streaming usage chunk: ``stream_options.include_usage`` must emit exactly one
extra chunk, because the OpenAI client reads usage there and nowhere else.

Run inside the container, against a server that is already up; the argument is
the port, default 18080:

    cd /root/private_data/engine && MKL_THREADING_LAYER=GNU python3 diag/probe_tool_choice.py

Module: ``coffin/diag``. Standalone probe: drives the OpenAI adapter over HTTP
and imports nothing from ``coffin/``.

Constraints:
    * Streams every request. The tool-call parser has a separate streaming path
      that accumulates ``tool_calls`` deltas by index, and a non-streamed probe
      would never exercise it.
    * ``temperature`` is 0 so a mismatch is a server bug, not sampling variance;
      do not raise it to "get more variety", that turns the run unreproducible.
    * The Chinese strings below are request and sample data, not prose, and are
      quoted verbatim from the tool schema users send.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 18080
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"

TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather", "description": "查询指定城市的当前天气",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                       "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "multiply", "description": "计算两个整数的乘积",
        "parameters": {"type": "object",
                       "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                       "required": ["a", "b"]}}},
]
# a request the model is happy to answer in prose -- so "did it call a tool" is not
# confounded by the prompt begging for one
PROSE = "讲个关于猫的冷笑话。"
WEATHER = "北京天气怎么样？"


def post(body: dict, timeout: float = 1800.0) -> tuple[dict, float]:
    """Send one chat-completion request and normalise streamed and unstreamed replies.

    Args:
        body: OpenAI request body, already fully built.
        timeout: Seconds to wait. The default is deliberately long because a
            cold engine may still be loading weights or building graphs; a short
            timeout would look like a tool-choice failure.

    Returns:
        ``(result, wall_seconds)``. ``result`` always carries ``finish``,
        ``content`` and ``calls`` as a list of ``{name, arguments}``, plus
        ``usage`` for a non-streamed reply or ``reasoning``/``usage_chunks`` for a
        streamed one.

    A streamed reply is reassembled here: content and reasoning deltas are
    concatenated, and ``tool_calls`` fragments are merged by their ``index``,
    because the wire format splits one call across many chunks and the argument
    JSON is only valid once every fragment has arrived.
    """
    req = urllib.request.Request(URL, data=json.dumps(body, ensure_ascii=False).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode()
    wall = time.time() - t0
    if not body.get("stream"):
        out = json.loads(raw)
        ch = out["choices"][0]
        return ({"finish": ch.get("finish_reason"),
                 "content": ch["message"].get("content") or "",
                 "calls": [{"name": (c.get("function") or {}).get("name"),
                            "arguments": (c.get("function") or {}).get("arguments")}
                           for c in (ch["message"].get("tool_calls") or [])],
                 "usage": out.get("usage")}, wall)
    calls: dict[int, dict] = {}
    content, reasoning, finish, usage_chunks = [], [], None, 0
    for line in raw.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        ch = json.loads(line[6:])
        if ch.get("choices") == []:
            usage_chunks += 1
            continue
        ch0 = ch["choices"][0]
        d = ch0.get("delta") or {}
        if d.get("content"):
            content.append(d["content"])
        if d.get("reasoning_content"):
            reasoning.append(d["reasoning_content"])
        for tc in d.get("tool_calls") or []:
            slot = calls.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
            fn = tc.get("function") or {}
            slot["name"] += fn.get("name") or ""
            slot["arguments"] += fn.get("arguments") or ""
        if ch0.get("finish_reason"):
            finish = ch0["finish_reason"]
    return ({"finish": finish, "content": "".join(content), "reasoning": "".join(reasoning),
             "calls": list(calls.values()), "usage_chunks": usage_chunks}, wall)


def run(tag: str, prompt: str, **extra) -> dict:
    """Send one case with a fixed model, tool set and temperature, and print it.

    Args:
        tag: Case label printed before the result, e.g. ``B tools + tool_choice=required``.
        prompt: User message.
        **extra: Request fields merged last, which is how each case sets
            ``tool_choice`` or ``stream_options``.

    Returns:
        The normalised reply from ``post``. ``temperature`` stays 0 and
        ``max_tokens`` 300 so a run is repeatable and a runaway prose answer
        cannot push the case past the server's own limit.
    """
    body = {"model": "qwen3.5-4b", "messages": [{"role": "user", "content": prompt}],
            "tools": TOOLS, "max_tokens": 300, "temperature": 0.0, "stream": True, **extra}
    out, wall = post(body)
    print(f"--- {tag}")
    print(f"    finish={out['finish']} wall={wall:.1f}s calls={len(out['calls'])} "
          f"content={len(out['content'])}c reasoning={len(out.get('reasoning') or '')}c "
          f"usage_chunks={out.get('usage_chunks', 0)}")
    for c in out["calls"]:
        print(f"    -> {c['name']}({str(c['arguments'])[:160]})")
    if out["content"].strip():
        print(f"    content[:120]={out['content'].strip()[:120]!r}")
    return out


def main() -> None:
    """Run the six tool-choice cases plus the usage-chunk case and print the tally.

    The verdict on the last line, ``n/7 PASS``, is the whole result: a partial
    pass is not "mostly working", because each case pins a distinct wire
    guarantee, so read which tags failed rather than the fraction.
    """
    ok: list[tuple[str, bool]] = []

    def expect(tag: str, out: dict, want_call: bool, want_name: str = "") -> None:
        """Record one expected outcome and print the observed reply beside it.

        Args:
            tag: Case label, reused in the final PASS/FAIL listing.
            out: Normalised reply from ``run``.
            want_call: Whether a tool call is expected at all.
            want_name: Function that must be called, or empty for "any or none".

        A case counts as passing only when call presence, function name and
        ``finish_reason`` all agree: a server that emits a call but reports
        ``stop`` breaks clients that branch on the finish reason.
        """
        got = bool(out["calls"])
        named = (not want_name) or (out["calls"] and out["calls"][0]["name"] == want_name)
        good = got == want_call and named and out["finish"] == ("tool_calls" if got else "stop")
        ok.append((tag, good))
        print(f"    {'OK' if good else 'MISMATCH'}: calls={got} want={want_call} "
              f"name={out['calls'][0]['name'] if out['calls'] else None} want_name={want_name or '-'}")

    expect("A auto (default), prose prompt -> the model is free to talk",
           run("A tools, no tool_choice (auto), prose prompt", PROSE), False)
    expect("B required, prose prompt -> a call is forced",
           run("B tools + tool_choice=required, prose prompt", PROSE,
               tool_choice="required"), True)
    expect("C required, weather prompt -> a call",
           run("C tools + tool_choice=required, weather prompt", WEATHER,
               tool_choice="required"), True)
    expect("D named get_weather, prose prompt -> that function",
           run("D tool_choice={get_weather}, prose prompt", PROSE,
               tool_choice={"type": "function", "function": {"name": "get_weather"}}),
           True, "get_weather")
    expect("E named multiply, weather prompt -> that function",
           run("E tool_choice={multiply}, weather prompt", WEATHER,
               tool_choice={"type": "function", "function": {"name": "multiply"}}),
           True, "multiply")
    expect("F none, weather prompt -> never a call",
           run("F tools + tool_choice=none, weather prompt", WEATHER,
               tool_choice="none"), False)

    out = run("G auto + stream_options.include_usage", WEATHER,
              stream_options={"include_usage": True})
    ok.append(("G include_usage emits its own chunk",
               out.get("usage_chunks") == 1 and out["finish"] == "tool_calls"))
    print(f"    {'OK' if ok[-1][1] else 'MISMATCH'}: usage_chunks={out.get('usage_chunks')}")

    print()
    n = sum(1 for _, good in ok if good)
    for tag, good in ok:
        print(f"[{'PASS' if good else 'FAIL'}] {tag}")
    print(f"\n{n}/{len(ok)} PASS")


if __name__ == "__main__":
    main()
