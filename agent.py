"""Tool-calling agent loop against the persistent engine server.

The server answers one assistant turn (``chat``); this client runs whatever tools
the model asked for and feeds the results back until it stops calling tools or the
round budget is exhausted.

    python agent.py --user "北京天气怎么样？再算一下 12 乘 8" --tools tools_demo.json
    python agent.py --user "..." --tools tools.json --exec my_tools.py

A tool module exposes ``execute(name, args) -> result``. Without ``--exec`` a small
built-in weather/multiply executor is used so the loop runs out of the box.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import serve

#: Follow serve.py's default, which is $ENGINE_SOCK-aware.
SOCK = serve.SOCK


def _builtin(name: str, args: dict):
    if name == "get_weather":
        return {"city": args.get("city"), "temp": 24,
                "unit": args.get("unit", "celsius"), "desc": "晴"}
    if name == "multiply":
        return {"result": int(args["a"]) * int(args["b"])}
    return {"error": f"unknown tool: {name}"}


def _load_executor(path: str):
    spec = importlib.util.spec_from_file_location("agent_tools", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.execute


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sock", default=SOCK)
    ap.add_argument("--user", required=True)
    ap.add_argument("--system", default=None)
    ap.add_argument("--tools", default=None, help="JSON file with an OpenAI tools array")
    ap.add_argument("--exec", dest="exec_path", default=None,
                    help="python file exposing execute(name, args)")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--think", action="store_true")
    args = ap.parse_args()

    execute = _load_executor(args.exec_path) if args.exec_path else _builtin
    tools = json.loads(Path(args.tools).read_text()) if args.tools else None
    messages: list[dict] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.user})

    for rnd in range(args.rounds):
        r = serve.call(args.sock, {"op": "chat", "messages": messages, "tools": tools,
                                   "enable_thinking": args.think, "max_new": args.max_new})
        if r.get("thinking"):
            print(f"[think] {r['thinking']}")
        if r.get("text"):
            print(f"[assistant] {r['text']}")
        calls = r.get("tool_calls") or []
        if not calls:
            break
        # the template renders ``content`` *and* ``tool_calls``, so keep only the
        # natural-language lead-in here and let the calls render from the structure.
        # ``arguments`` must be a dict: the template iterates it with ``|items``.
        lead = r["text"].split("<tool_call>")[0].strip()
        tcs = [{"function": {"name": c["function"]["name"],
                             "arguments": json.loads(c["function"]["arguments"])}}
               for c in calls]
        asst: dict = {"role": "assistant", "content": lead, "tool_calls": tcs}
        if r.get("thinking"):
            asst["reasoning_content"] = r["thinking"]
        messages.append(asst)
        for c in calls:
            name = c["function"]["name"]
            argv = json.loads(c["function"]["arguments"])
            res = execute(name, argv)
            print(f"  -> {name}({argv}) = {res}")
            messages.append({"role": "tool", "name": name,
                             "content": json.dumps(res, ensure_ascii=False)})


if __name__ == "__main__":
    main()
