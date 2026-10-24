"""``tool_choice`` against the *real* model, not the protocol stub.

The definition says ``required`` means "the model **must** call one or more tools"
and a named choice "forces the model to call that tool".  The interesting cases are
therefore the ones where the model would rather answer in prose: with ``auto`` it
talks, with ``required`` the API must hand back a call anyway.

    cd /root/private_data/engine && MKL_THREADING_LAYER=GNU python3 diag/probe_tool_choice.py
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
    ok: list[tuple[str, bool]] = []

    def expect(tag: str, out: dict, want_call: bool, want_name: str = "") -> None:
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
