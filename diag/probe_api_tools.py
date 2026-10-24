"""Does a *real agent-shaped* request come back as a tool call?

The minimal case (one tool, one user turn, socket level) already works; agent
clients differ in three ways, and any of them can be the reason "the model
doesn't call the tool" while it obviously can:

* they **stream** (``stream: true``) and assemble ``tool_calls`` from deltas;
* they send a **large** tool set and an agent **system prompt**;
* they hold a **multi-turn** conversation: assistant ``tool_calls`` + ``role: tool``
  results go back into the prompt, and the model must keep going.

Every scenario here goes over HTTP against the running server (18080), which is the
path the client actually takes -- no in-process shortcuts.

    cd /root/private_data/engine && MKL_THREADING_LAYER=GNU python3 diag/probe_api_tools.py
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 18080
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"

SYSTEM = ("You are a coding agent working in the repository at /root/private_data. "
          "Use the provided tools to inspect and change files. Prefer calling a tool "
          "over answering from memory. Do not explain what you are about to do; call "
          "the tool.")

# A tool set the size of a real agent's, with real-looking schemas.
SPECS = [
    ("read_file", "读取一个文件的内容", {"path": ("string", "文件路径")}, ["path"]),
    ("write_file", "把内容写入文件（覆盖）", {"path": ("string", "文件路径"),
                                            "content": ("string", "要写入的内容")},
     ["path", "content"]),
    ("run_command", "在容器里执行一条 shell 命令", {"command": ("string", "命令行")},
     ["command"]),
    ("list_dir", "列出一个目录", {"path": ("string", "目录路径")}, ["path"]),
    ("search", "在仓库里搜索文本", {"pattern": ("string", "正则表达式")}, ["pattern"]),
    ("get_weather", "查询指定城市的当前天气",
     {"city": ("string", "城市名，例如 北京"), "unit": ("string", "单位")}, ["city"]),
    ("multiply", "计算两个整数的乘积", {"a": ("integer", "第一个数"),
                                       "b": ("integer", "第二个数")}, ["a", "b"]),
]
# pad to a realistic count
for i in range(14):
    SPECS.append((f"helper_{i:02d}", f"辅助工具 {i}，用于外部系统交互",
                  {"query": ("string", "查询串")}, ["query"]))


def build_tools() -> list[dict]:
    out = []
    for name, desc, props, required in SPECS:
        out.append({"type": "function", "function": {
            "name": name, "description": desc,
            "parameters": {"type": "object",
                           "properties": {k: {"type": t, "description": d}
                                          for k, (t, d) in props.items()},
                           "required": required}}})
    return out


TOOLS = build_tools()


def post(body: dict, timeout: float = 1800.0) -> tuple[dict, float]:
    req = urllib.request.Request(URL, data=json.dumps(body, ensure_ascii=False).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode()
    wall = time.time() - t0
    if body.get("stream"):
        calls: dict[int, dict] = {}
        content, reasoning, finish = [], [], None
        for line in raw.splitlines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ch = json.loads(line[6:])
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
        out = {"choices": [{"finish_reason": finish,
                            "message": {"content": "".join(content),
                                        "reasoning_content": "".join(reasoning),
                                        "tool_calls": [{"type": "function", "function": v}
                                                       for _, v in sorted(calls.items())]}}],
               "usage": {}}
        return out, wall
    return json.loads(raw), wall


def show(tag: str, out: dict, wall: float) -> list[dict]:
    ch = out["choices"][0]
    msg = ch["message"]
    tcs = msg.get("tool_calls") or []
    content = msg.get("content") or ""
    print(f"--- {tag}")
    print(f"    finish={ch.get('finish_reason')} wall={wall:.1f}s "
          f"content={len(content)}c reasoning={len(msg.get('reasoning_content') or '')}c "
          f"tool_calls={len(tcs)}")
    for tc in tcs:
        fn = tc.get("function") or {}
        print(f"    -> {fn.get('name')}({fn.get('arguments')})")
    if content.strip():
        print(f"    content[:200]={content.strip()[:200]!r}")
    return tcs


def main() -> None:
    print(f"tools sent: {len(TOOLS)}")

    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "北京天气怎么样？"}]
    base = {"model": "qwen3.5-4b", "messages": msgs, "tools": TOOLS,
            "max_tokens": 400, "temperature": 0.0}

    out, wall = post({**base, "stream": False})
    print(f"[A] non-stream  prompt_tokens={out['usage'].get('prompt_tokens')}")
    show("A non-stream, 21 tools, agent system prompt", out, wall)

    out, wall = post({**base, "stream": True})
    print(f"[B] stream      prompt_tokens={out['usage'].get('prompt_tokens')}")
    tcs = show("B stream, same request", out, wall)

    # [C] two-turn: feed the call's result back the way a client does, then ask for
    # something that needs a *different* tool -- the prompt now contains an assistant
    # tool_calls message and a role:tool message.
    if tcs:
        fn = tcs[0]["function"]
        try:
            args = json.loads(fn["arguments"])
        except Exception:
            args = {}
        msgs2 = msgs + [
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "call_1", "type": "function",
                             "function": {"name": fn["name"], "arguments": args}}]},
            {"role": "tool", "name": fn["name"],
             "content": json.dumps({"city": args.get("city"), "temp": 24, "desc": "晴"},
                                   ensure_ascii=False)},
            {"role": "user", "content": "好，再帮我算一下 12 乘 8。"},
        ]
        out, wall = post({**base, "messages": msgs2, "stream": True})
        show("C turn 2, assistant tool_calls + role:tool in the prompt", out, wall)

    # [D] thinking on (the server default is off; an agent UI may ask for it)
    out, wall = post({**base, "stream": True, "enable_thinking": True,
                      "max_tokens": 400})
    show("D stream + enable_thinking=true", out, wall)


if __name__ == "__main__":
    main()
