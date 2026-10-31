"""面向常驻引擎服务端的工具调用 agent 循环。

服务端应答一个 assistant 回合（``chat``）；这个客户端执行模型要求的所有工具、
把结果喂回去，直到模型不再调用工具或者轮次预算耗尽。

    python agent.py --user "Beijing weather? Also 12 times 8" --tools tools_demo.json
    python agent.py --user "..." --tools tools.json --exec my_tools.py

工具模块暴露 ``execute(name, args) -> result``。不给 ``--exec`` 时用一个内置的
天气/乘法小执行器，循环开箱即可跑。

Module: ``coffin/agent.py``。导入图的最顶层：它导入常驻的 socket 传输及其默认
socket 路径，没有东西反向导入它。``utils.tool_call`` 和 ``run_chat.py`` 是各自
独立的入口。

Constraints:
    * 循环必须把模型自己的调用结构原样回显，而不是解析出来的 ``ToolCall`` 对象。
      chat 模板会从消息字典里同时渲染 ``content`` *和* ``tool_calls``，所以它需要
      ``arguments`` 是个字典（模板用 ``|items`` 迭代它），而线上格式携带的是
      JSON 字符串。
    * 自然语言引子必须在 ``<tool_call>`` 处截断：模板会从结构重新渲染调用，把原始
      标签留在 ``content`` 里会让每个调用渲染两遍。
    * 每个回合一进一出走 socket；这个客户端不持有引擎状态，所以轮次之间被杀掉也不会
      泄漏序列。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from serving.runtime_support import SOCK
from serving.runtime_transport import call

#: 复用运行时的默认值，agent 和服务端因此落在同一条 socket 路径上。


def _builtin(name: str, args: dict):
    """占位执行器，让刚 clone 下来的仓库里循环也有东西可调。

    它刻意做得很简单：存在的目的是展示循环的形状，不是当
    有用的工具；``--exec`` 会替换它。

    Args:
        name: 模型请求的工具名。
        args: 该工具解析后的参数。

    Returns:
        可 JSON 化的结果；未知工具名返回 ``{"error": ...}`` 字典——
        一个模型能读到并从中恢复的错误，胜过直接终止循环的
        异常。
    """
    if name == "get_weather":
        return {"city": args.get("city"), "temp": 24,
                "unit": args.get("unit", "celsius"), "desc": "晴"}
    if name == "multiply":
        return {"result": int(args["a"]) * int(args["b"])}
    return {"error": f"unknown tool: {name}"}


def _load_executor(path: str):
    """导入用户的工具模块，把它的 ``execute`` 函数交回来。

    按路径加载而不是按模块名：那文件是用户随手指定的脚本，不在 ``sys.path`` 上。

    Args:
        path: 暴露 ``execute(name, args)`` 的 Python 文件。

    Returns:
        模块的 ``execute`` 可调用对象，直接取出，循环的调用点不必关心它来自哪里。
    """
    spec = importlib.util.spec_from_file_location("agent_tools", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.execute


def main() -> None:
    """跑轮次循环：问服务端、执行调用、把结果喂回去。

    预算是轮次数而不是 token 数：一轮是一个完整的 assistant 回合加它的工具结果，
    一直调工具的模型总得有个东西叫停。

    Raises:
        OSError: ``--sock`` 上没人监听；先把运行时 CLI 起起来。
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
        r = call(args.sock, {"op": "chat", "messages": messages, "tools": tools,
                                   "enable_thinking": args.think, "max_new": args.max_new})
        if r.get("thinking"):
            print(f"[think] {r['thinking']}")
        if r.get("text"):
            print(f"[assistant] {r['text']}")
        calls = r.get("tool_calls") or []
        if not calls:
            break

        # 模板会同时渲染 ``content`` *和* ``tool_calls``，所以 ``content`` 里只留自然
        # 语言引子，调用由结构渲染。``arguments`` 必须是字典：模板用 ``|items`` 迭代它。
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
