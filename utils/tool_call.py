"""Parser and serializer for the Qwen3.5 tool-call wire format.

The format is the checkpoint's, not ours: the chat template teaches the model to
answer a tool request with the body below, so the parser has to accept exactly
that text and nothing more forgiving.

    <tool_call>
    <function=get_weather>
    <parameter=city>
    Beijing
    </parameter>
    </function>
    </tool_call>

One call may carry several ``<function=...>`` blocks and several
``<parameter=...>`` blobs. Parameter values are raw text, not JSON: the format
has no quoting, so ``0755`` and ``celsius`` arrive as strings, and a value that
parses as a JSON literal has to be un-wrapped deliberately (see ``_coerce``).
A JSON fallback (``{"name": ..., "arguments": {...}}`` inside ``<tool_call>``)
is also accepted because some checkpoints emit it.

When thinking is on, the reasoning sits in ``<think>...</think>`` and may precede
the call; ``split_thinking`` separates the two so the OpenAI shell can return
``reasoning_content`` next to the answer.

Module: ``coffin/utils/tool_call.py``. Leaf of the import graph: standard library
only, so ``serving/`` and ``tests/test_openai.py`` can both depend on it.

Constraints:
    * The tags and the surrounding template are a checkpoint contract. Changing
      the patterns without retraining the template silently stops every tool
      call from being recognised -- the model output would still look right.
    * Parsing never raises on malformed input: a truncated or half-written block
      yields fewer calls, because this runs on a degraded generation and turning
      that into a server error would kill the request.
    * Parsing is pure: no I/O and no global state, so it is safe to call on a
      streamed partial answer.
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
    """One parsed call, in the shape both the chat template and OpenAI expect.

    ``to_openai_dict`` re-serialises ``arguments`` to a JSON *string*, because
    the OpenAI wire format carries arguments as text; the chat template instead
    iterates them as a dict. Both consumers read this same object, so the two
    representations live in one place.
    """

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    id: str = ""
    raw: str = ""

    def __post_init__(self) -> None:
        """Mint an id when the wire format carried none.

        The OpenAI response contract requires every call to have an id, and the
        Qwen template does not emit one, so the parser invents it here rather
        than making each caller do so.
        """
        if not self.id:
            self.id = "call_" + uuid.uuid4().hex[:24]

    def to_openai_dict(self) -> dict[str, Any]:
        """Render as one entry of an OpenAI ``tool_calls`` array.

        Returns:
            ``{"id", "type": "function", "function": {"name", "arguments"}}``
            with ``arguments`` JSON-encoded and non-ASCII kept readable, since
            the client parses it back verbatim.
        """
        return {"id": self.id, "type": "function",
                "function": {"name": self.name,
                             "arguments": json.dumps(self.arguments, ensure_ascii=False)}}

    def __repr__(self) -> str:
        """One-line form for the agent loop's ``-> name(args)`` trace.

        Returns:
            The name and arguments, with ``raw`` left out so a long tool-call
            block does not flood the log.
        """
        return f"ToolCall(name={self.name!r}, arguments={self.arguments!r})"


def split_thinking(text: str) -> tuple[str, str]:
    """Split reasoning from the answer, returning ``(thinking, answer)``.

    The template already emits the opening ``<think>`` in the prompt, so the
    generated text usually carries only the closing ``</think>``; both shapes and
    a truncated (unclosed) ``<think>`` are handled.

    Args:
        text: Raw generated text, after the prompt has been stripped from it.

    Returns:
        ``(thinking, answer)``, both stripped. ``thinking`` is ``""`` when the
        model did not reason. For an unclosed ``<think>`` the answer is whatever
        preceded the tag and the partial reasoning becomes ``thinking``, so a
        truncated turn still yields usable text.
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
    """The answer half of ``text``, for callers that discard the reasoning.

    Args:
        text: Raw generated text; empty is accepted.

    Returns:
        The answer with any thinking block removed, stripped.
    """
    return split_thinking(text)[1]


def _coerce(value: str) -> Any:
    """Unwrap values that are unambiguous JSON literals; keep the rest verbatim.

    Only ``true``/``false``/``null``, objects/arrays and numbers are converted, so
    ``"Beijing"``, ``"0755"`` and ``"celsius"`` stay strings. The formats share no
    type information, so guessing wider (dates, booleans spelled ``yes``) would
    corrupt arguments the tool validates itself.

    Args:
        value: Raw text between one ``<parameter=...>`` pair.

    Returns:
        The decoded literal, or the stripped text when it is not one.
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
    """Parse the inside of one ``<tool_call>`` block, template form first.

    The native form is tried first and wins outright: a block that contains
    ``<function=...>`` never falls through to JSON, so a parameter whose value
    happens to be JSON cannot be mistaken for a whole call.

    Args:
        block: Text between the ``<tool_call>`` tags, tags excluded.

    Returns:
        Every call the block contains, in order; empty when neither form
        matches, which leaves a malformed block as "no call" rather than an
        error.
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
    """Extract every tool call from a model response (empty list if none).

    Args:
        text: Assistant text with the prompt already stripped; may still contain
            the thinking block.

    Returns:
        The calls in generation order. An unterminated ``<tool_call>`` -- the
        usual shape when generation hits the token budget mid-call -- is parsed
        from its partial body, which is what keeps a truncated answer usable.
    """
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
    """Whether ``text`` contains anything ``parse_tool_calls`` can turn into a call.

    Args:
        text: Assistant text; may be empty.

    Returns:
        ``True`` when at least one call parsed, i.e. when the loop should run a
        tool instead of treating the answer as final.
    """
    return bool(parse_tool_calls(text))


def tool_result_message(call: ToolCall, result: Any, name: str | None = None) -> dict[str, str]:
    """Build the chat message that feeds a tool result back to the model.

    Args:
        call: The call being answered; it supplies the default ``name``.
        result: The tool's return value. A string is passed through unchanged,
            anything else is JSON-encoded, because the template renders
            ``content`` verbatim.
        name: Override for the ``name`` field, for a result produced by a
            different tool than the one called.

    Returns:
        A ``{"role": "tool", "name", "content"}`` message ready to append to the
        history.
    """
    content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    return {"role": "tool", "name": name or call.name, "content": content}
