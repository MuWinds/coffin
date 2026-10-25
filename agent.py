"""Tool-calling agent loop against the persistent engine server.

The server answers one assistant turn (``chat``); this client runs whatever tools
the model asked for and feeds the results back until it stops calling tools or the
round budget is exhausted.

    python agent.py --user "Beijing weather? Also 12 times 8" --tools tools_demo.json
    python agent.py --user "..." --tools tools.json --exec my_tools.py

A tool module exposes ``execute(name, args) -> result``. Without ``--exec`` a small
built-in weather/multiply executor is used so the loop runs out of the box.

Module: ``coffin/agent.py``. Top of the import graph: it imports ``serve`` for
the socket client and the default socket path, and nothing imports it back.
``tool_call.py`` and ``run_chat.py`` are siblings, not dependencies.

Constraints:
    * The loop must echo the model's own call structure back, not the parsed
      ``ToolCall`` objects. The chat template renders ``content`` *and*
      ``tool_calls`` from the message dict, so it needs ``arguments`` as a dict
      (it iterates them with ``|items``) while the wire format carries a JSON
      string.
    * The natural-language lead-in has to be truncated at ``<tool_call>``: the
      template re-renders the calls from the structure, and keeping the raw tags
      in ``content`` would render every call twice.
    * Every turn is one request/response on the socket; this client holds no
      engine state, so it can be killed between rounds without leaking a
      sequence.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import serve

#: ``serve.SOCK`` already resolves ``$ENGINE_SOCK``, so reusing it keeps the
#: agent and the server on one socket default instead of two that can drift.
SOCK = serve.SOCK


def _builtin(name: str, args: dict):
    """Stand-in executor so the loop has something to call on a fresh checkout.

    It is deliberately trivial: it exists to show the loop's shape, not to be a
    useful tool, and ``--exec`` replaces it.

    Args:
        name: Tool name the model asked for.
        args: Parsed arguments for that tool.

    Returns:
        A JSON-serialisable result, or an ``{"error": ...}`` dict for an unknown
        name -- an error the model can read and recover from beats an exception
        that ends the loop.
    """
    if name == "get_weather":
        return {"city": args.get("city"), "temp": 24,
                "unit": args.get("unit", "celsius"), "desc": "晴"}
    if name == "multiply":
        return {"result": int(args["a"]) * int(args["b"])}
    return {"error": f"unknown tool: {name}"}


def _load_executor(path: str):
    """Import a user tool module and hand back its ``execute`` function.

    Loaded by path rather than by module name: the file is an arbitrary script
    the user points at, so it is not on ``sys.path``.

    Args:
        path: Python file exposing ``execute(name, args)``.

    Returns:
        The module's ``execute`` callable, unwrapped so the loop's call site
        does not care where it came from.
    """
    spec = importlib.util.spec_from_file_location("agent_tools", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.execute


def main() -> None:
    """Run the round loop: ask the server, execute the calls, feed results back.

    The budget is a round count, not a token count: each round is one full
    assistant turn plus its tool results, and a model that keeps calling tools
    has to be stopped by something.

    Raises:
        OSError: No server is listening on ``--sock``; start ``serve.py`` first.
    """
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

        # The template renders ``content`` *and* ``tool_calls``, so only the
        # natural-language lead-in stays in ``content`` and the calls render from
        # the structure. ``arguments`` must be a dict: the template iterates it
        # with ``|items``.
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
