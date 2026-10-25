"""Incremental reasoning and tool-call stream parsers.

Module: ``coffin.serving``. The parsers convert the engine's raw token stream
into OpenAI-compatible deltas without touching the model or HTTP transport.
"""

from __future__ import annotations

import json
import re
import uuid

TOOL_OPEN = "<tool_call>"
_CALL_OPEN = "<tool_call>"
_FN_OPEN = re.compile(r"<function=([^>\n]+)>")
_PARAM_OPEN = re.compile(r"<parameter=([^>\n]+)>")
_PARAM_CLOSE = "</parameter>"
_FN_CLOSE = "</function>"
_CALL_CLOSE = "</tool_call>"
_JSON_LITERAL_HEAD = "-0123456789tfn{["
_SHORT_ESCAPE = {"\n": "\\n", "\r": "\\r", "\t": "\\t", "\b": "\\b", "\f": "\\f"}

class AnswerFilter:
    """Splits the raw stream into (reasoning, content) increments.

    Two things are handled here and both are load-bearing for a thinking model:

    * the reasoning block -- with thinking on the *prompt* opens ``<think>``, so
      the generated text starts inside it. The reasoning is handed out as
      ``reasoning_content`` deltas while it is being produced, and ``content``
      starts only after ``</think>``. Dropping it, or replaying it at the end, is
      what left an agent UI's thinking pane empty.
    * ``<tool_call>`` markup -- once it appears, content stops and the call is
      delivered structured instead; a partial marker is held back so it cannot
      leak, and ``finish`` releases that tail at end of stream.

    Deliberately incremental: each ``push`` only scans the newly appended text,
    so the filter costs O(tokens) rather than rescanning the whole answer.
    """

    def __init__(self, prompt_opens_think: bool, marker: str = TOOL_OPEN) -> None:
        """Start a filter for one reply.

        Args:
            prompt_opens_think: Whether the rendered prompt already ended inside
                an open ``<think>``, which the engine reports on its ``start``
                line. When it did not, the whole answer is content from the first
                token.
            marker: The tool-call opener to cut content at.
        """
        self.raw = ""
        self.marker = marker
        self.past_think = not prompt_opens_think
        self.think_pos = 0            # how much of the reasoning has been sent
        self.emitted = 0              # how much of the content has been sent
        self.done_tools = False

    def _content(self) -> str:
        """Next content increment, holding back a possible partial tool marker.

        Returns:
            The content text that is safe to emit now. Empty while the tail could
            still become the marker: emitting it and finding out later would leak
            markup into the answer.
        """
        if self.done_tools:
            return ""
        body = self.raw[self.emitted:]
        k = body.find(self.marker)
        if k >= 0:
            self.done_tools = True
            out, self.emitted = body[:k], self.emitted + k
            return out
        keep = len(self.marker) - 1          # never emit a partial marker
        if len(body) <= keep:
            return ""
        out = body[:len(body) - keep]
        self.emitted += len(out)
        return out

    def push(self, piece: str) -> tuple[str, str]:
        """Split one streamed piece into its reasoning and content increments.

        Args:
            piece: The engine's incremental text for one token.

        Returns:
            ``(reasoning_delta, content_delta)``. Either may be empty; the caller
            emits only the non-empty one as its own SSE delta.
        """
        self.raw += piece
        if self.past_think:
            return "", self._content()
        i = self.raw.find("</think>", self.think_pos)
        if i < 0:                            # still inside the reasoning block
            out = self.raw[self.think_pos:]
            self.think_pos = len(self.raw)
            return out, ""
        reasoning = self.raw[self.think_pos:i]
        self.past_think = True
        self.think_pos = len(self.raw)
        self.emitted = i + len("</think>")
        return reasoning, self._content()

    def finish(self, has_tools: bool) -> tuple[str, str]:
        """Flush the tails at end of stream.

        The content hold-back exists so a partial ``<tool_call>`` never reaches the
        client; at end of stream it has to be released, or the last few characters
        of every answer would be lost. With tool calls the tail *is* markup
        (already delivered structured), so it is dropped.

        Args:
            has_tools: Whether the final engine reply carried parsed tool calls.

        Returns:
            ``(reasoning_tail, content_tail)``; either may be empty.
        """
        if not self.past_think:              # the reasoning never closed
            out = self.raw[self.think_pos:]
            self.think_pos = len(self.raw)
            return out, ""
        if self.done_tools:
            return "", ""
        body = self.raw[self.emitted:]
        if has_tools:
            k = body.find(self.marker)
            body = body[:k] if k >= 0 else ""
        self.emitted += len(body)
        return "", body


def _json_escape(text: str) -> str:
    """Escape a fragment exactly the way ``json.dumps(ensure_ascii=False)`` does.

    Byte-for-byte agreement matters: the fragments a client concatenates have to
    be the same string the non-streaming path returns, or a client that falls back
    to the non-stream JSON of the same call would see two different argument
    strings.

    Args:
        text: Raw value text from the tool-call markup.

    Returns:
        The escaped fragment, without the enclosing quotes.
    """
    out = []
    for ch in text:
        if ch == '"' or ch == "\\":
            out.append("\\" + ch)
        elif ch in _SHORT_ESCAPE:
            out.append(_SHORT_ESCAPE[ch])
        elif ord(ch) < 0x20:
            out.append("\\u%04x" % ord(ch))
        else:
            out.append(ch)
    return "".join(out)


def _coerce_value(value: str):
    """Keep the value typed like ``tool_call._coerce`` (the non-streaming path).

    The two paths must agree on the type, or the same call would arrive as
    ``{"n": 3}`` when streamed and ``{"n": "3"}`` when not.

    Args:
        value: The raw text between ``<parameter=...>`` and ``</parameter>``.

    Returns:
        The parsed JSON value when the text is a JSON literal (including nested
        objects and arrays), else the stripped string. Text that looks numeric but
        does not parse -- ``1.2.3`` -- stays a string rather than failing the call.
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


class ToolStream:
    """Turn the Qwen ``<tool_call>`` markup into OpenAI ``tool_calls`` deltas as it
    is generated, in the shape vLLM's streaming tool parsers emit.

    The engine streams the call token by token, but parsing only at the end (what
    this server used to do) means a client gets one fat delta after the whole call
    is complete: no progress for a long argument, and a UI that looks hung. So the
    same state machine vLLM runs per choice lives here instead:

    * the first fragment of a call carries ``index``/``id``/``type``/``function.name``
      (with ``arguments: ""``), so the client knows *which* tool is being called
      before its arguments exist;
    * ``function.arguments`` arrives in fragments -- a parameter's value streams as
      it is generated when it is a string, and is emitted whole (typed: numbers,
      true/false/null, nested JSON) when it looks like a JSON literal, because that
      decision cannot be unmade half way through;
    * ``</tool_call>`` closes the call with a final empty ``arguments`` fragment, and
      the next call gets ``index + 1``.

    ``arguments`` accumulates to exactly ``json.dumps(parsed_arguments)`` -- the same
    string the non-streaming reply carries -- so a client can use either path.

    The buffer is append-only and ``pos`` only ever moves forward past text that is
    already decided, which is what lets a token that splits a marker across two
    pieces still parse: an incomplete marker is not matched until its last
    character arrives.
    """

    def __init__(self, seed: str = "", max_calls: int | None = None,
                 marker: str = _CALL_OPEN) -> None:
        """Start a parser for one reply.

        Args:
            seed: Text to parse before any generated piece. Used to inject the
                forced ``tool_choice: required`` opener, which lives in the prompt
                rather than in the generated text.
            max_calls: Stop emitting after this many calls, for
                ``parallel_tool_calls: false``. ``None`` emits every call.
            marker: The tool-call opener to look for.
        """
        self.buf = seed or ""
        self.pos = 0
        self.marker = marker
        self.max_calls = max_calls
        self.index = -1
        self.call_id = ""
        self.in_call = False
        self.name_sent = False
        self.first_param = True
        self.param = None
        self.mode = ""            # "", "stream" (string value) or "hold" (JSON literal)
        self.hold = ""            # held value text for "hold"
        self.ws = ""              # trailing whitespace of a streamed value
        self.value_started = False
        self.open_quote = False   # the JSON string's opening quote is still owed
        self.emitted_any = False

    def _delta(self, arguments: str, name: str | None = None) -> dict | None:
        """Build one ``tool_calls`` delta, or ``None`` when it must be suppressed.

        Args:
            arguments: The ``function.arguments`` fragment for this delta.
            name: The function name on the fragment that introduces the call;
                ``None`` for every following fragment, which is what tells the
                client whether to expect a new ``id``/``type`` pair.

        Returns:
            A ``{"tool_calls": [...]}`` payload, or ``None`` once
            ``parallel_tool_calls: false`` has been satisfied -- the caller then
            emits nothing for the extra call.
        """
        if self.max_calls is not None and self.index >= self.max_calls:
            return None                     # parallel_tool_calls: false -> drop it
        fn: dict = {"arguments": arguments}
        if name is not None:
            fn["name"] = name
        tc: dict = {"index": self.index, "function": fn}
        if name is not None:                # the fragment that introduces the call
            tc["id"] = self.call_id
            tc["type"] = "function"
        self.emitted_any = True
        return {"tool_calls": [tc]}

    def _value_fragment(self, text: str) -> list[dict]:
        """Feed raw value text and emit whatever argument fragment is now final.

        A string value can stream as it arrives; a JSON literal cannot, because a
        number read as a string cannot be turned back into a number once the client
        has already received the quotes. So the first character decides the mode
        once and the value follows it to the end: ``hold`` buffers the text and
        dumps it coerced at the closing tag, ``stream`` escapes and forwards it.

        Args:
            text: Newly available value text, without its closing tag.

        Returns:
            Zero or more ``tool_calls`` deltas. Trailing whitespace is held back
            instead of emitted, so the value of ``"1 "`` loses its pad and matches
            what ``_coerce`` produces on the non-streaming path.
        """
        if not text:
            return []
        if not self.value_started:
            text = text.lstrip()            # ``_coerce`` strips the value's ends
            if not text:
                return []
            self.value_started = True
            if text[0] in _JSON_LITERAL_HEAD:
                self.mode = "hold"
            else:
                self.mode = "stream"
                self.open_quote = True      # a string value opens with a quote
        if self.mode == "hold":
            self.hold += text
            return []
        combined = self.ws + text
        body = combined.rstrip()            # trailing whitespace is not emitted yet
        if not body:
            self.ws = combined
            return []
        self.ws = combined[len(body):]
        fragment = _json_escape(body)
        if self.open_quote:
            fragment = '"' + fragment
            self.open_quote = False
        d = self._delta(fragment)
        return [d] if d else []

    def _close_param(self) -> list[dict]:
        """Emit the fragment that terminates the current parameter's value.

        Returns:
            Zero or one delta. A ``hold`` value is coerced and dumped whole so the
            argument stream ends exactly like ``json.dumps`` would; a ``stream``
            value just closes its quote.
        """
        out: list[dict] = []
        if self.mode == "hold":
            d = self._delta(json.dumps(_coerce_value(self.hold), ensure_ascii=False))
        elif self.mode == "stream":
            d = self._delta('"')            # close the value string
        else:
            d = self._delta('""')           # empty (or whitespace-only) value
        if d:
            out.append(d)
        self.param = None
        self.mode = ""
        self.hold = self.ws = ""
        self.value_started = False
        self.open_quote = False
        return out

    def push(self, piece: str) -> list[dict]:
        """Parse one streamed text piece.

        Args:
            piece: The engine's incremental text for one token.

        Returns:
            The ``tool_calls`` deltas that became final with this piece; empty
            while a marker or a held JSON literal is still incomplete.
        """
        self.buf += piece
        return self._drain(final=False)

    def finish(self) -> list[dict]:
        """Flush a truncated call at end of stream.

        Returns:
            The remaining deltas, including the closing fragment for a call whose
            markup ended early. Without this a call cut off by ``max_tokens`` would
            reach the client with an unterminated ``arguments`` string.
        """
        return self._drain(final=True)

    def _drain(self, final: bool) -> list[dict]:
        """Advance the state machine over whatever the buffer now holds.

        Args:
            final: ``True`` at end of stream, which closes a truncated parameter
                instead of waiting for its closing tag.

        Returns:
            The deltas produced by this pass, in emission order.
        """
        out: list[dict] = []
        while True:
            if not self.in_call:
                i = self.buf.find(self.marker, self.pos)
                if i < 0:
                    # nothing to do until the opener is complete: the buffer is
                    # append-only and these markers only match in full, so ``pos``
                    # must stay put or later pieces would be scanned past.
                    break
                self.pos = i + len(self.marker)
                self.in_call = True
                self.index += 1
                self.call_id = "call_" + uuid.uuid4().hex[:24]
                self.name_sent = False
                self.first_param = True
                self.mode = ""
                self.hold = self.ws = ""
                self.value_started = False
                self.open_quote = False
                continue

            if self.param is not None:
                end = self.buf.find(_PARAM_CLOSE, self.pos)
                if end < 0:
                    if final:                       # truncated call: close it here
                        out += self._value_fragment(self.buf[self.pos:])
                        self.pos = len(self.buf)
                        out += self._close_param()
                        break
                    safe = len(self.buf) - (len(_PARAM_CLOSE) - 1)
                    if safe > self.pos:
                        out += self._value_fragment(self.buf[self.pos:safe])
                        self.pos = safe
                    break
                out += self._value_fragment(self.buf[self.pos:end])
                self.pos = end + len(_PARAM_CLOSE)
                out += self._close_param()
                continue

            m_fn = _FN_OPEN.search(self.buf, self.pos)
            m_param = _PARAM_OPEN.search(self.buf, self.pos)
            closes = [(self.buf.find(tok, self.pos), tok)
                      for tok in (_FN_CLOSE, _CALL_CLOSE)]
            nxt = [(m_fn.start(), "fn") if m_fn else None,
                   (m_param.start(), "param") if m_param else None,
                   *[(p, "close") for p, _ in closes if p >= 0]]
            nxt = [c for c in nxt if c is not None]
            if not nxt:
                # wait for the marker to be complete (see above): do not advance pos
                break
            pos, kind = min(nxt, key=lambda c: c[0])
            if kind == "fn":
                name = m_fn.group(1).strip()
                d = self._delta("", name) if not self.name_sent else None
                self.name_sent = True
                if d:
                    out.append(d)
                self.pos = m_fn.end()
                continue
            if kind == "param":
                key = m_param.group(1).strip()
                prefix = ("{" if self.first_param else ", ")
                d = self._delta(prefix + json.dumps(key, ensure_ascii=False) + ": ")
                if d:
                    out.append(d)
                self.first_param = False
                self.param = key
                self.pos = m_param.end()
                continue
            if self.buf.startswith(_CALL_CLOSE, pos):
                d = self._delta("")               # vLLM's end-of-call empty fragment
                if d:
                    out.append(d)
                self.pos = pos + len(_CALL_CLOSE)
                self.in_call = False
                self.param = None
                continue
            # </function>
            d = self._delta("{}" if self.first_param else "}")
            if d:
                out.append(d)
            self.pos = pos + len(_FN_CLOSE)
        return out
