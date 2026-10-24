"""Parser for the Qwen3.5 tool-call format.

The chat template asks the model to answer a tool request with::

    <tool_call>
    <function=get_weather>
    <parameter=city>
    北京
    </parameter>
    </function>
    </tool_call>

Reasoning, when thinking is on, is wrapped in ``<think>...</think>`` and may
precede the call. ``parse_tool_calls`` turns that raw text into structured calls;
a JSON fallback (``{"name": ..., "arguments": {...}}`` inside ``<tool_call>``) is
also accepted because some checkpoints emit it.
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
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    id: str = ""
    raw: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = "call_" + uuid.uuid4().hex[:24]

    def to_openai_dict(self) -> dict[str, Any]:
        return {"id": self.id, "type": "function",
                "function": {"name": self.name,
                             "arguments": json.dumps(self.arguments, ensure_ascii=False)}}

    def __repr__(self) -> str:
        return f"ToolCall(name={self.name!r}, arguments={self.arguments!r})"


def split_thinking(text: str) -> tuple[str, str]:
    """Split reasoning from the answer, returning ``(thinking, answer)``.

    The template already emits the opening ``<think>`` in the prompt, so the
    generated text usually carries only the closing ``</think>``; both shapes and
    a truncated (unclosed) ``<think>`` are handled.
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
    return split_thinking(text)[1]


def _coerce(value: str) -> Any:
    """Unwrap values that are unambiguous JSON literals; keep the rest verbatim.

    Only ``true``/``false``/``null``, objects/arrays and numbers are converted, so
    ``"北京"``, ``"0755"`` and ``"celsius"`` stay strings.
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
    """Extract every tool call from a model response (empty list if none)."""
    if not text:
        return []
    blocks = [m.group(1) for m in _TC.finditer(text)]
    if not blocks and _TC_OPEN in text:
        blocks = [text.split(_TC_OPEN, 1)[1]]      # truncated generation
    calls: list[ToolCall] = []
    for b in blocks:
        calls.extend(_parse_block(b))
    return calls


def has_tool_call(text: str) -> bool:
    return bool(parse_tool_calls(text))


def tool_result_message(call: ToolCall, result: Any, name: str | None = None) -> dict[str, str]:
    """Build the chat message that feeds a tool result back to the model."""
    content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    return {"role": "tool", "name": name or call.name, "content": content}
