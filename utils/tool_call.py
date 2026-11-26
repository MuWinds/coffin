"""Qwen3.5 / Occamy 工具调用线格式的解析与序列化。

格式是 checkpoint 定的，不是我们定的：对话模板教模型用下面这样的正文回答工具
请求，所以解析器必须恰好接受这种文本，不能再宽容。下面每个开标签都对应一个同名
的闭合标签（写法见 ``chat_protocol.DEFAULT_MARKERS``）：

    <tool_call>
    <function=get_weather>
    <parameter=city>
    Beijing
    <闭合标签>
    <闭合标签>
    <闭合标签>

一个调用可以带多个 ``<function=...>`` 块和多个 ``<parameter=...>`` 块。参数值是
原始文本而不是 JSON：格式里没有引号概念，所以 ``0755`` 和 ``celsius`` 的取值都以
字符串到达，类型由工具 schema 声明的那一个决定（见 ``chat_protocol.convert_param``）。
``<tool_call>`` 里直接放 ``{"name": ..., "arguments": {...}}`` 的 JSON 兜底形式也
接受，因为有 checkpoint 就是这么输出的。

开着思考时，推理段在 ``<think>`` 与它的闭合标签之间，且可能出现在调用之前；
``split_thinking`` 把两者分开，OpenAI 外壳才能把 ``reasoning_content`` 和答案一起
返回。

Module: ``coffin/utils/tool_call.py``。依赖图的叶子：只用标准库，所以
``serving/`` 和 ``tests/test_openai.py`` 都能依赖它。

Constraints:
    * 标签和配套的模板是 checkpoint 契约。模板文本是标记的权威来源，本模块的
      默认值只是后备（见 ``chat_protocol``）。不重新训练模板就改这些标签，会让
      所有工具调用静默地识别不出来——模型输出看起来还是对的。
    * 解析永远不为畸形输入抛异常：截断或只写了一半的块只是产出更少的调用，
      因为这段代码跑在已经降级的生成结果上，把它变成服务端错误等于杀掉整个
      请求。
    * 解析是纯函数：不做 I/O 也没有全局状态，所以对流式返回的半截回答调用
      是安全的。
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from utils import chat_protocol


@dataclass
class ToolCall:
    """一个解析出来的调用，形状同时满足对话模板和 OpenAI 两边的要求。

    ``to_openai_dict`` 会把 ``arguments`` 重新序列化成 JSON *字符串*，因为 OpenAI
    线格式用文本承载参数；对话模板则把它当字典遍历。两个消费方读的是同一个
    对象，所以两种表示集中在这里维护。
    """

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    id: str = ""
    raw: str = ""

    def __post_init__(self) -> None:
        """线格式没带 id 时在这里生成一个。

        OpenAI 的响应契约要求每个调用都有 id，而 Qwen 模板不产出 id，所以由
        解析器统一补上，省得每个调用方各写一遍。
        """
        if not self.id:
            self.id = "call_" + uuid.uuid4().hex[:24]

    def to_openai_dict(self) -> dict[str, Any]:
        """渲染成 OpenAI ``tool_calls`` 数组里的一项。

        Returns:
            ``{"id", "type": "function", "function": {"name", "arguments"}}``，
            其中 ``arguments`` 已 JSON 编码，非 ASCII 保持可读——客户端会原样
            解析回去。
        """
        return {"id": self.id, "type": "function",
                "function": {"name": self.name,
                             "arguments": json.dumps(self.arguments, ensure_ascii=False)}}

    def __repr__(self) -> str:
        """给 agent 循环里 ``-> name(args)`` 那种追踪用的一行形式。

        Returns:
            名字和参数；``raw`` 被省略，免得一个长工具调用块把日志刷屏。
        """
        return f"ToolCall(name={self.name!r}, arguments={self.arguments!r})"


def split_thinking(text: str, markers: dict[str, str] | None = None) -> tuple[str, str]:
    """把推理段和答案分开，返回 ``(thinking, answer)``。

    转发到 ``chat_protocol.split_thinking``，保留这个名字是因为服务层、agent
    循环和测试都按它引用。

    Args:
        text: 从 prompt 之后切出来的原始生成文本。
        markers: 标记表；``None`` 用这一族的后备值。

    Returns:
        ``(thinking, answer)``，两者都已去空白。
    """
    return chat_protocol.split_thinking(text, markers)


def strip_thinking(text: str, markers: dict[str, str] | None = None) -> str:
    """``text`` 里属于答案的那一半，供丢弃推理段的调用方使用。

    Args:
        text: 原始生成文本；允许为空。
        markers: 标记表；``None`` 用这一族的后备值。

    Returns:
        去掉推理块并去过空白的答案。
    """
    return chat_protocol.strip_thinking(text, markers)


def _coerce(value: str) -> Any:
    """把无歧义的 JSON 字面量去壳，其余原样保留。

    只在没有工具 schema 时使用（本地 agent 循环、裸探针）。走 OpenAI 的那条路
    一定带 schema，类型由 :func:`chat_protocol.convert_param` 按声明决定：这里
    猜得越宽，``0755`` 这种字符串型参数被改成数字的机会越大。

    Args:
        value: 一对 ``<parameter=...>`` 之间的原始文本。

    Returns:
        解码后的字面量；不是字面量时返回去过空白的原文。
    """
    s = value.strip()
    if s in ("true", "false", "null"):
        return json.loads(s)
    if s[:1] in "{[" or (s[:1] in "-0123456789" and s == s.strip()):
        try:
            return json.loads(s)
        except Exception:
            return s
    return s


def _spans(text: str, marker_block: dict[str, str], open_key: str,
           stops: tuple[str, ...] = ()) -> list[tuple[str, str]]:
    """按标记表扫出 ``<name=...>`` 块的 ``(名字, 正文)``。

    正文到闭合标签、下一个同种开标签或调用方点名的停靠标签为止，所以被截断的
    块（生成撞上 token 预算）仍然能交出一段残缺的取值，而不是整块作废。

    Args:
        text: 要扫的文本。
        marker_block: 标记表。
        open_key: 开标签在标记表里的键（``function_open`` / ``param_open``）。
        stops: 额外的正文终止标记。

    Returns:
        ``(名字, 正文)`` 列表，按出现顺序排列。
    """
    open_tag = marker_block[open_key]
    close_tag = marker_block["function_close" if open_key == "function_open"
                             else "param_close"]
    out: list[tuple[str, str]] = []
    pos = 0
    while True:
        i = text.find(open_tag, pos)
        if i < 0:
            return out
        head_end = text.find(">", i + len(open_tag))
        if head_end < 0:
            return out                      # 名字还没写完，这一块不算
        name = text[i + len(open_tag):head_end].strip()
        body = head_end + 1
        ends = [k for k in (text.find(close_tag, body),) if k >= 0]
        for stop in stops:
            k = text.find(stop, body)
            if k >= 0:
                ends.append(k)
        end = min(ends) if ends else len(text)
        out.append((name, text[body:end]))
        pos = end + (len(close_tag) if text.startswith(close_tag, end) else 0)


def _parse_block(block: str, markers: dict[str, str] | None,
                 specs: dict[str, dict[str, str]] | None) -> list[ToolCall]:
    """解析一个 ``<tool_call>`` 块的内部，先试模板形式。

    原生形式先试且一试到底：含 ``<function=...>`` 的块绝不会再落到 JSON 分支，
    取值恰好是 JSON 的参数因此不会被误认成整个调用。

    Args:
        block: ``<tool_call>`` 标签之间的文本，不含标签本身。
        markers: 标记表；``None`` 用后备值。
        specs: 工具 schema 的 ``函数名 -> 参数类型`` 表；``None`` 表示本条请求
            没有 schema，退回按字面量猜类型的老规则。

    Returns:
        块里的全部调用，按顺序排列；两种形式都不匹配时为空——畸形的块被当成
        「没有调用」，而不是一个错误。
    """
    mk = chat_protocol.markers_from(markers)
    calls: list[ToolCall] = []
    fns = _spans(block, mk, "function_open", (mk["function_open"],))
    if fns:
        for name, body in fns:
            args: dict[str, Any] = {}
            for pname, raw in _spans(body, mk, "param_open",
                                     (mk["param_open"], mk["function_close"])):
                if specs is None:
                    args[pname] = _coerce(raw)
                else:
                    args[pname] = chat_protocol.convert_param(
                        raw, chat_protocol.param_type(specs, name, pname))
            calls.append(ToolCall(name=name, arguments=args,
                                  raw=f"{mk['function_open']}{name}{mk['function_close']}"))
        return calls
    try:
        obj = json.loads(block.strip())
    except Exception:
        return calls
    for it in (obj if isinstance(obj, list) else [obj]):
        if not isinstance(it, dict):
            continue
        fn = it.get("function") if "function" in it else it
        name = fn.get("name") or it.get("name")
        if not name:
            continue
        args = fn.get("arguments", it.get("arguments", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"value": args}
        calls.append(ToolCall(name=name, arguments=args or {}, raw=block.strip()))
    return calls


def _blocks(text: str, markers: dict[str, str] | None) -> list[str]:
    """取出每个 ``<tool_call>`` 块的内部。

    窗口截断成半个标记时没有闭合标签，此时尾部整体算一个块——这是生成撞上
    ``max_tokens`` 时的常见形状，丢掉它等于把已经想好的调用扔掉。

    Args:
        text: assistant 文本。
        markers: 标记表。

    Returns:
        块内文本列表，不含标签本身。
    """
    mk = chat_protocol.markers_from(markers)
    open_tag, close_tag = mk["tool_open"], mk["tool_close"]
    out: list[str] = []
    pos = 0
    while True:
        i = text.find(open_tag, pos)
        if i < 0:
            return out
        body = i + len(open_tag)
        j = text.find(close_tag, body)
        if j < 0:
            out.append(text[body:])             # 生成被截断
            return out
        out.append(text[body:j])
        pos = j + len(close_tag)


def parse_tool_calls(text: str, tools=None,
                     markers: dict[str, str] | None = None) -> list[ToolCall]:
    """从模型回复里取出所有工具调用，没有则返回空列表。

    Args:
        text: 已经去掉 prompt 的 assistant 文本；其中可能仍有推理块。
        tools: 请求里的工具 schema 数组。给了就按它声明的类型定型，没给（本地
            agent 循环）才按字面量猜。
        markers: 标记表；``None`` 用这一族的后备值。

    Returns:
        按生成顺序排列的调用。未闭合的 ``<tool_call>``——生成到一半撞上 token
        预算时的常见形状——按它残缺的正文解析，被截断的回答因此仍可用。
    """
    if not text:
        return []
    specs = None if tools is None else chat_protocol.param_types(tools)
    calls: list[ToolCall] = []
    for block in _blocks(text, markers):
        calls.extend(_parse_block(block, markers, specs))
    return calls


def has_tool_call(text: str, tools=None, markers: dict[str, str] | None = None) -> bool:
    """``text`` 里有没有 ``parse_tool_calls`` 能变成调用的内容。

    Args:
        text: assistant 文本；可以为空。
        tools: 工具 schema；含义同 :func:`parse_tool_calls`。
        markers: 标记表。

    Returns:
        至少解析出一个调用时为 ``True``，也就是循环该去执行工具，而不是把
        这个回答当成最终答案。
    """
    return bool(parse_tool_calls(text, tools, markers))


def tool_result_message(call: ToolCall, result: Any, name: str | None = None) -> dict[str, str]:
    """构造把工具结果喂回模型的那条对话消息。

    Args:
        call: 正在被回答的调用；它提供默认的 ``name``。
        result: 工具的返回值。字符串原样传递，其余一律 JSON 编码，因为模板是
            逐字渲染 ``content`` 的。
        name: 覆盖 ``name`` 字段，用于结果并非由被调用的那个工具产生的情况。

    Returns:
        一条 ``{"role": "tool", "name", "content"}`` 消息，可以直接追加到历史里。
    """
    content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    return {"role": "tool", "name": name or call.name, "content": content}
