"""Qwen3.5 工具调用线格式的解析与序列化。

格式是 checkpoint 定的，不是我们定的：对话模板教模型用下面这样的正文回答工具
请求，所以解析器必须恰好接受这种文本，不能再宽容。

    <tool_call>
    <function=get_weather>
    <parameter=city>
    Beijing
    </parameter>
    </function>
    </tool_call>

一个调用可以带多个 ``<function=...>`` 块和多个 ``<parameter=...>`` 块。
参数值是原始文本而不是 JSON：格式里没有引号概念，所以 ``0755`` 和 ``celsius``
都会以字符串到达，而能被解析成 JSON 字面量的值必须刻意去壳（见 ``_coerce``）。
``<tool_call>`` 里直接放 ``{"name": ..., "arguments": {...}}`` 的 JSON 兜底形式也
接受，因为有 checkpoint 就是这么输出的。

开着思考时，推理段在 ``<think>...</think>`` 里，且可能出现在调用之前；
``split_thinking`` 把两者分开，OpenAI 外壳才能把 ``reasoning_content`` 和答案一起
返回。

Module: ``coffin/utils/tool_call.py``。依赖图的叶子：只用标准库，所以
``serving/`` 和 ``tests/test_openai.py`` 都能依赖它。

Constraints:
    * 标签和配套的模板是 checkpoint 契约。不重新训练模板就改这些正则，会让
      所有工具调用静默地识别不出来——模型输出看起来还是对的。
    * 解析永远不为畸形输入抛异常：截断或只写了一半的块只是产出更少的调用，
      因为这段代码跑在已经降级的生成结果上，把它变成服务端错误等于杀掉整个
      请求。
    * 解析是纯函数：不做 I/O 也没有全局状态，所以对流式返回的半截回答调用
      是安全的。
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

_TC = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_TC_OPEN = "<tool_call>"
_FN = re.compile(r"<function=([^>\n]+)>(.*?)</function>", re.DOTALL)
_PARAM = re.compile(r"<parameter=([^>\n]+)>(.*?)</parameter>", re.DOTALL)
_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL)


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


def split_thinking(text: str) -> tuple[str, str]:
    """把推理段和答案分开，返回 ``(thinking, answer)``。

    模板已经把开头的 ``<think>`` 发在 prompt 里了，所以生成文本通常只带闭合的
    ``</think>``；两种形状以及被截断（未闭合）的 ``<think>`` 都处理。

    Args:
        text: 从 prompt 之后切出来的原始生成文本。

    Returns:
        ``(thinking, answer)``，两者都已去空白。模型没有推理时 ``thinking``
        是 ``""``。遇到未闭合的 ``<think>`` 时，标签之前的部分算答案，残缺的
        推理段算 ``thinking``，被截断的一轮因此仍有可用文本。
    """
    if not text:
        return "", ""
    m = _THINK.search(text)
    if m:
        return m.group(1).strip(), (text[:m.start()] + text[m.end():]).strip()
    if "</think>" in text:
        i = text.index("</think>")
        return text[:i].strip(), text[i + len("</think>"):].strip()
    if "<think>" in text:
        i = text.index("<think>")
        return text[i + len("<think>"):].strip(), text[:i].strip()
    return "", text.strip()


def strip_thinking(text: str) -> str:
    """``text`` 里属于答案的那一半，供丢弃推理段的调用方使用。

    Args:
        text: 原始生成文本；允许为空。

    Returns:
        去掉推理块并去过空白的答案。
    """
    return split_thinking(text)[1]


def _coerce(value: str) -> Any:
    """把无歧义的 JSON 字面量去壳，其余原样保留。

    只转换 ``true``/``false``/``null``、对象/数组和数字，所以 ``"Beijing"``、
    ``"0755"``、``"celsius"`` 仍是字符串。这个格式本身不携带类型信息，猜得再宽
    一点（日期、写成 ``yes`` 的布尔）就会改坏工具自己去校验的参数。

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


def _parse_block(block: str) -> list[ToolCall]:
    """解析一个 ``<tool_call>`` 块的内部，先试模板形式。

    原生形式先试且一试到底：含 ``<function=...>`` 的块绝不会再落到 JSON 分支，
    取值恰好是 JSON 的参数因此不会被误认成整个调用。

    Args:
        block: ``<tool_call>`` 标签之间的文本，不含标签本身。

    Returns:
        块里的全部调用，按顺序排列；两种形式都不匹配时为空——畸形的块被当成
        「没有调用」，而不是一个错误。
    """
    calls: list[ToolCall] = []
    fns = list(_FN.finditer(block))
    if fns:
        for m in fns:
            args = {pm.group(1).strip(): _coerce(pm.group(2))
                    for pm in _PARAM.finditer(m.group(2))}
            calls.append(ToolCall(name=m.group(1).strip(), arguments=args, raw=m.group(0)))
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


def parse_tool_calls(text: str) -> list[ToolCall]:
    """从模型回复里取出所有工具调用，没有则返回空列表。

    Args:
        text: 已经去掉 prompt 的 assistant 文本；其中可能仍有推理块。

    Returns:
        按生成顺序排列的调用。未闭合的 ``<tool_call>``——生成到一半撞上 token
        预算时的常见形状——按它残缺的正文解析，被截断的回答因此仍可用。
    """
    if not text:
        return []
    blocks = [m.group(1) for m in _TC.finditer(text)]
    if not blocks and _TC_OPEN in text:
        blocks = [text.split(_TC_OPEN, 1)[1]]      # 生成被截断
    calls: list[ToolCall] = []
    for b in blocks:
        calls.extend(_parse_block(b))
    return calls


def has_tool_call(text: str) -> bool:
    """``text`` 里有没有 ``parse_tool_calls`` 能变成调用的内容。

    Args:
        text: assistant 文本；可以为空。

    Returns:
        至少解析出一个调用时为 ``True``，也就是循环该去执行工具，而不是把
        这个回答当成最终答案。
    """
    return bool(parse_tool_calls(text))


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
