"""OpenAI-compatible HTTP API in front of the resident engine.

``serve.py`` is a unix-socket JSON server; this is the HTTP/SSE skin that OpenAI
clients (openai-python, LangChain, Open WebUI, ``curl``) expect.  It is
deliberately **stdlib-only** so it adds nothing to install inside the container,
and it never imports torch: it is a pure client of the socket protocol.  The
served model id and architecture are read from the engine (``ping``), so this
file names no checkpoint either.

Run it next to the engine:

    MKL_THREADING_LAYER=GNU python serve.py --serve \
        --model /root/private_data/models/Qwen3.5-4B &
    python openai_server.py --sock /tmp/qwen35.sock --port 18080

Endpoints:
    GET  /health                 engine liveness, graph/VRAM state
    GET  /v1/models              the served model id (for client discovery)
    POST /v1/chat/completions    messages, tools, stream (SSE)

Concurrency:
The engine is **single-stream**: the decode step is one CUDA graph over one
token and all the state (KV pool, position, recurrent states) belongs to the one
sequence in flight (see NOTES.md §7).  Every engine call is therefore serialized
behind one lock -- a second concurrent request waits its turn instead of
corrupting the first one's state.  Batching is *not* emulated here.

Streaming:
``stream: true`` maps onto the engine's ``chat_stream`` op (one reply line per
token) and is re-emitted as ``chat.completion.chunk`` SSE events.  Two things are
filtered on the way out:

* the reasoning block -- with thinking on, the template opens ``<think>`` in the
  prompt, so the generated text starts inside the reasoning; content is held back
  until ``</think>`` (the engine reports whether the prompt opened it).  If
  generation stops *inside* the block (``max_tokens`` before ``</think>``), the
  text is delivered as ``reasoning_content`` instead of being passed off as the
  answer;
* the tool-call markup -- once ``<tool_call>`` appears, content stops, and the
  parsed calls are delivered structured (OpenAI ``tool_calls``) in a final delta,
  so clients never see the XML.

Known gaps (kept honest, not silently ignored): ``top_p`` / ``n`` / penalties /
``stop`` are accepted and ignored (the engine samples with temperature only, and
greedy when temperature is 0); a client that hangs up leaves the engine
generating to the end (no cancel path in the socket protocol yet); ``max_tokens``
beyond ``--max-seq`` is truncated by the engine rather than rejected here.

Tool calling follows the definition's ``ChatCompletionToolChoiceOption`` rather
than only passing ``tools`` through: ``none`` drops the tools, ``auto`` (the
default when tools are present) lets the model choose, ``required`` **forces** the
turn to open with the template's call header, and
``{"type": "function", "function": {"name": X}}`` forces that one function; the
deprecated ``functions`` / ``function_call`` pair maps onto the same two fields.
``parallel_tool_calls: false`` returns the first call only, and
``stream_options.include_usage`` adds the definition's extra ``choices: []`` chunk
with the totals just before ``[DONE]``.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import itertools
import json
import os
import re
import socket
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

#: Default matches serve.py; $ENGINE_SOCK overrides both (a second model gets
#: its own socket and, from there, its own HTTP port).
SOCK = os.environ.get("ENGINE_SOCK", "/tmp/qwen35.sock")
TOOL_OPEN = "<tool_call>"
STATE: dict = {}

# ---------------------------------------------------------------------------
# Image parts: what this process validates, and what it deliberately does not
# ---------------------------------------------------------------------------
# This file is stdlib-only on purpose (no torch, no PIL) so the HTTP skin can run
# anywhere the engine socket is reachable.  Everything here is therefore a *cheap*
# check -- the shape of the data URL, the decoded byte counts, the image count --
# and it exists to answer 400 before a byte reaches the engine.  The expensive
# checks (PIL decode, decompression-bomb size, the processor's token expansion)
# belong to serve.py, which owns those libraries and re-validates authoritatively
# against its own configured limits.
#
#: Decoded bytes of one image.  A 1 MP PNG is ~1-3 MB, so 8 MiB is generous and
#: still bounded; base64 inflates it 4/3 on the wire.
MAX_IMAGE_BYTES = int(os.environ.get("ENGINE_MAX_IMAGE_BYTES", 8 * 1024 * 1024))
#: Images per request.  Four 1 MP images are already ~4k image tokens.
MAX_IMAGES = int(os.environ.get("ENGINE_MAX_IMAGES", 4))
#: Base64 characters of all image parts of one request, before decoding.
MAX_IMAGE_TOTAL_BYTES = int(os.environ.get("ENGINE_MAX_IMAGE_TOTAL_BYTES", 16 * 1024 * 1024))
#: MIME types accepted on the data URL.  The processor decodes with PIL, so the
#: set is "what PIL opens reliably and the model was trained on".
IMAGE_MIMES = ("image/png", "image/jpeg", "image/jpg", "image/webp", "image/bmp",
               "image/gif")
_DATA_URL = re.compile(r"^data:(?P<mime>[-\w.+]+/[-\w.+]+)?(?P<params>[^,]*),"
                       r"(?P<data>.*)$", re.DOTALL)


class ImagePartError(ValueError):
    """A client-supplied image part this process refuses (-> HTTP 400)."""


def _decode_data_url(url: str) -> tuple[bytes, str]:
    """``data:image/png;base64,...`` -> ``(raw bytes, mime)``.

    Only base64 data URLs are accepted: v1 does not fetch ``http(s)`` URLs (the
    container has no egress and a fetch would be an SSRF surface), and a URL that
    would make the *server* do the I/O is exactly the thing a caller cannot audit.
    """
    if not isinstance(url, str) or not url:
        raise ImagePartError("image_url.url must be a non-empty string")
    m = _DATA_URL.match(url.strip())
    if not m:
        if url.lower().startswith(("http://", "https://")):
            raise ImagePartError(
                "only `data:` URLs are supported for image_url; this build does not "
                "fetch remote images -- send the bytes inline as "
                "`data:image/png;base64,...`")
        raise ImagePartError(
            "image_url.url must be a `data:image/<type>;base64,<payload>` URL")
    mime = (m.group("mime") or "image/png").lower()
    if mime == "image/jpg":
        mime = "image/jpeg"
    if mime not in IMAGE_MIMES:
        raise ImagePartError(
            f"unsupported image type {mime!r}; supported: {', '.join(sorted(set(IMAGE_MIMES)))}")
    params = (m.group("params") or "").lower()
    if "base64" not in params:
        raise ImagePartError("image_url.url must be base64-encoded (data:...;base64,)")
    b64 = re.sub(r"\s+", "", m.group("data") or "")
    if not b64:
        raise ImagePartError("image_url.url carries no data")
    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImagePartError(f"image_url.url is not valid base64: {exc}") from None
    if not raw:
        raise ImagePartError("image_url.url decodes to zero bytes")
    if len(raw) > MAX_IMAGE_BYTES:
        raise ImagePartError(
            f"image is {len(raw) / 2**20:.2f} MiB decoded, over the "
            f"{MAX_IMAGE_BYTES / 2**20:.0f} MiB per-image limit; downscale it first")
    return raw, mime


def _split_parts(content, budget: dict) -> tuple[object, list[dict]]:
    """Split one message's ``content`` into (template-safe content, image payloads).

    An OpenAI vision message is a *parts* array in which one part carries the
    image bytes and the rest carry text.  The checkpoint's chat template knows only
    two things: text, and a ``{"type": "image"}`` marker that it renders as
    ``<|vision_start|><|image_pad|><|vision_end|>`` (the placeholders the engine's
    processor later expands to one token per visual patch).  So the image parts are
    lifted out of the array -- their bytes become ``images[]`` on the engine
    request -- and replaced in place by that marker, which keeps the *order* of
    text and images exactly as the client wrote it.

    ``budget`` is a mutable per-request dict (``{"images": n, "b64": chars}``) so
    the count and total-size limits apply to the whole request, not per message.
    ``b64`` counts the *data URL* characters the client sent (what has to travel
    over the wire and be held in memory), not just the payload.

    Returns ``(content, images)``: ``content`` is the untouched value when the
    message has no image part, which is what keeps the text-only request path
    byte-for-byte what it was.
    """
    if not isinstance(content, list):
        return content, []
    parts: list = []
    images: list[dict] = []
    for part in content:
        if isinstance(part, str):
            parts.append({"type": "text", "text": part})
            continue
        if not isinstance(part, dict):
            raise ImagePartError(f"content part must be an object, got {type(part).__name__}")
        kind = part.get("type")
        if kind in (None, "text"):
            parts.append({"type": "text", "text": part.get("text") or ""})
            continue
        if kind in ("video", "video_url", "input_video"):
            raise ImagePartError(
                "video input is not supported: frame count -> token count is not "
                "bounded by this build's image limits")
        if kind == "input_audio":
            raise ImagePartError("audio input is not supported by this model")
        if kind not in ("image_url", "image"):
            raise ImagePartError(f"unsupported content part type {kind!r}")
        holder = part.get("image_url")
        if isinstance(holder, dict):
            url = holder.get("url")
        elif isinstance(holder, str):
            url = holder
        else:
            url = None
        if url is None and kind == "image":
            url = part.get("image")
            if isinstance(url, dict):
                url = url.get("url")
        if url is None:
            raise ImagePartError(
                f"`{kind}` part has no `image_url.url`; send "
                "{{\"type\": \"image_url\", \"image_url\": {{\"url\": "
                "\"data:image/png;base64,...\"}}}}")
        # Count and total-size limits are checked before the decode: a request that
        # already has too many images should not pay to base64-decode one more.
        if budget.get("images", 0) >= MAX_IMAGES:
            raise ImagePartError(
                f"too many images: {MAX_IMAGES} per request is the limit")
        budget["images"] = budget.get("images", 0) + 1
        budget["b64"] = budget.get("b64", 0) + len(url)
        if budget["b64"] > MAX_IMAGE_TOTAL_BYTES:
            raise ImagePartError(
                f"images total {budget['b64'] / 2**20:.2f} MiB of base64, over the "
                f"{MAX_IMAGE_TOTAL_BYTES / 2**20:.0f} MiB per-request limit; send fewer "
                f"or smaller images")
        raw, mime = _decode_data_url(url)
        images.append({"b64": base64.b64encode(raw).decode(), "mime": mime})
        parts.append({"type": "image"})      # the template's own marker
    return parts, images


def _split_messages(messages: list) -> tuple[list, list[dict], str | None]:
    """Pull every image out of a request's ``messages``.

    ``-> (messages, images, error)``.  ``error`` is a human-readable 400 message,
    or ``None``.  The three rules that cannot be deferred to the engine:

    * a ``system`` message may not carry an image -- the template raises on it, so
      the client would otherwise get an opaque 500 from a generation it cannot
      debug;
    * the per-request count/size budget (see ``_split_parts``);
    * an image part in a role whose content the template never renders as media.
    """
    budget: dict = {"images": 0, "b64": 0}
    out: list = []
    images: list[dict] = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        m = dict(msg)
        role = str(m.get("role") or "user").lower()
        try:
            content, found = _split_parts(m.get("content"), budget)
        except ImagePartError as exc:
            return [], [], str(exc)
        if found:
            if role in ("system", "developer"):
                # The template itself raises for this, so reject it with a message
                # that says what to do instead of surfacing a jinja traceback.
                return [], [], ("a system message cannot contain images: the chat "
                                "template rejects it; put the image in a user turn")
            if role == "assistant":
                # Historical assistant turns carry text; an image there has no
                # template rendering and would be silently dropped.
                return [], [], ("an assistant message cannot carry images; put the "
                                "image in a user or tool message")
            # ``tool`` IS accepted: an agent that calls a screenshot tool gets the
            # capture back as the tool result, and that is the single most common
            # way images enter this server.  The template renders it exactly like a
            # user image (vision_start/image_pad/vision_end inside the message), and
            # the processor expands it in prompt order, so no special case is
            # needed beyond collecting the payload here.
            images.extend(found)
        m["content"] = content
        out.append(m)
    return out, images, None


class EngineClient:
    """One client of the engine's line-oriented unix socket.

    Each request opens its own connection: the engine accepts connections
    concurrently and batches whatever requests are in flight, so a lock here
    would serialize exactly the work the batch scheduler is supposed to overlap.
    """

    def __init__(self, sock_path: str, timeout: float = 900.0) -> None:
        self.sock_path = sock_path
        self.timeout = timeout

    def _open(self):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        s.connect(self.sock_path)
        return s, s.makefile("rb")

    @staticmethod
    def _close(s, f) -> None:
        for c in (f, s):
            try:
                c.close()
            except Exception:
                pass

    def _send(self, s, req: dict) -> None:
        s.sendall((json.dumps(req, ensure_ascii=False) + "\n").encode())

    def call(self, req: dict) -> dict:
        """One request, one reply."""
        s, f = self._open()
        try:
            self._send(s, req)
            line = f.readline()
            if not line:
                raise RuntimeError("engine closed the connection")
            return json.loads(line.decode())
        finally:
            self._close(s, f)

    def stream(self, req: dict):
        """Yield reply lines until the terminal one (``type: done`` / an error)."""
        s, f = self._open()
        try:
            self._send(s, req)
            while True:
                line = f.readline()
                if not line:
                    raise RuntimeError("engine closed the connection")
                obj = json.loads(line.decode())
                yield obj
                if obj.get("type") == "done" or obj.get("ok") is False:
                    return
        finally:
            self._close(s, f)


def _text_of(content) -> str:
    """Flatten OpenAI's string-or-parts content into the plain string the template wants."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in (None, "text"):
                out.append(part.get("text") or "")
            elif isinstance(part, str):
                out.append(part)
        return "".join(out)
    return str(content)


def _has_image_part(content) -> bool:
    return isinstance(content, list) and any(
        isinstance(p, dict) and p.get("type") in ("image", "image_url") for p in content)


def _template_content(content):
    """What the chat template should receive for one message's ``content``.

    Text-only content is flattened to a string (exactly as before, so a text-only
    request renders byte-for-byte what it used to).  A message that carries an
    image keeps its *parts array*, because that is the only shape the template
    understands for media: each ``{"type": "image"}`` marker becomes
    ``<|vision_start|><|image_pad|><|vision_end|>`` in place, preserving the
    interleaving of text and images the client wrote.
    """
    if not _has_image_part(content):
        return _text_of(content)
    out = []
    for part in content:
        if isinstance(part, str):
            part = {"type": "text", "text": part}
        if not isinstance(part, dict):
            continue
        if part.get("type") in ("image", "image_url"):
            out.append({"type": "image"})
        else:
            out.append({"type": "text", "text": part.get("text") or ""})
    return out


def _norm_messages(messages: list) -> list[dict]:
    """Make the client's message list safe for the checkpoint's jinja template.

    Three shapes actually break it, all of them seen from real clients:

    * ``content`` as a parts array;
    * a role the template does not know -- OpenAI's newer ``developer`` (or a
      harness inventing its own) makes the template raise *"Unexpected message
      role."* and the whole request fails;
    * a ``system`` message that is not first, or no user turn at all, which the
      template rejects outright;

    and one whose ``arguments`` is the OpenAI JSON *string* rather than a mapping
    -- the template iterates that field with ``|items``.

    A parts array that carries images is *not* flattened: the template renders
    ``{"type": "image"}`` itself (see ``_template_content``).
    """
    out: list[dict] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        m = dict(msg)
        role = str(m.get("role") or "user").lower()
        if role == "developer":              # the newer name for the system role
            role = "system"
        elif role == "function":
            role = "tool"
        elif role not in ("system", "user", "assistant", "tool"):
            role = "user"
        m["role"] = role
        m["content"] = _template_content(m.get("content"))
        calls = m.get("tool_calls")
        if isinstance(calls, list):
            fixed = []
            for c in calls:
                if not isinstance(c, dict):
                    continue
                fn = dict(c.get("function") or {})
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                fn["arguments"] = args or {}
                fixed.append({"id": c.get("id") or "", "type": "function", "function": fn})
            m["tool_calls"] = fixed
        out.append(m)

    if out and out[0]["role"] == "system":
        pass
    elif any(m["role"] == "system" for m in out):     # hoist it, demote the rest
        first = next(m for m in out if m["role"] == "system")
        out = [first] + [{**m, "role": "user"} if m["role"] == "system" else m
                         for m in out if m is not first]
    if out and not any(m["role"] == "user" for m in out):
        out.append({"role": "user", "content": "(continue)"})
    return out


def _max_tokens(body: dict) -> int:
    """The output budget the client asked for, else the configured default.

    ``max_completion_tokens`` is the field the newer OpenAI clients (and coding
    agents) send; reading only ``max_tokens`` used to fall through to a small
    default, which truncates exactly the answers people ask a long-context model
    for -- a file of code.

    When the client states nothing at all, ``--default-max-tokens`` applies rather
    than "the whole window": the engine is single-stream, so an unbounded request
    that nobody is waiting for any more keeps every later request queued until it
    finishes on its own (measured: a 117 s generation with no client attached).
    """
    for key in ("max_completion_tokens", "max_tokens", "max_new_tokens"):
        val = body.get(key)
        if val is not None:
            try:
                return max(1, int(val))
            except (TypeError, ValueError):
                pass
    return int(STATE.get("default_max_tokens", -1))



def _thinking_flag(body: dict, default: bool) -> bool:
    """Does this request want the model's reasoning block?

    Clients spell it differently, so all the shapes actually seen are accepted:
    ``enable_thinking`` (vLLM/Qwen), ``chat_template_kwargs.enable_thinking``,
    ``thinking`` as a bool/string/``{"type": "enabled"}``, and ``reasoning_effort``
    (anything but none/off, which is how the newer OpenAI-style agents ask).
    """
    for key in ("enable_thinking", "thinking"):
        val = body.get(key)
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.strip().lower() not in ("", "false", "none", "off", "disabled")
        if isinstance(val, dict):
            inner = val.get("type", val.get("enabled"))
            if isinstance(inner, bool):
                return inner
            if isinstance(inner, str):
                return inner.strip().lower() in ("enabled", "true", "on", "auto")
    kw = body.get("chat_template_kwargs")
    if isinstance(kw, dict) and "enable_thinking" in kw:
        return bool(kw["enable_thinking"])
    effort = body.get("reasoning_effort")
    if isinstance(effort, str):
        return effort.strip().lower() not in ("", "none", "off", "disabled")
    return default


def _tool_plan(body: dict) -> tuple[list | None, str]:
    """``(tools to render, forced assistant opener)`` per OpenAI's tool_choice.

    The definition (``ChatCompletionToolChoiceOption``) says: ``none`` -- the model
    must not call a tool; ``auto`` (the default when tools are present) -- it picks;
    ``required`` -- it **must** call one or more tools; and
    ``{"type": "function", "function": {"name": X}}`` -- it must call exactly X.
    The deprecated-but-still-in-the-definition ``functions`` / ``function_call``
    fields mean the same thing and map onto the same two fields here.

    ``required`` is honoured by *forcing the opener*: the engine appends the
    template's call header to the prompt, so the turn can only continue as a call
    (a 4B model otherwise often answers in prose while tools are attached -- which
    is exactly the reported symptom).  With a named choice the function name is
    forced too.
    """
    tools = body.get("tools")
    legacy = body.get("functions")
    if not tools and isinstance(legacy, list) and legacy:
        tools = [{"type": "function", "function": f}
                 for f in legacy if isinstance(f, dict)]
    if not isinstance(tools, list) or not tools:
        return None, ""

    choice = body.get("tool_choice")
    if choice is None and "function_call" in body:
        choice = body.get("function_call")

    if choice is None:
        return tools, ""
    if isinstance(choice, str):
        mode = choice.strip().lower()
        if mode == "none":
            return None, ""
        if mode in ("required", "any"):
            return tools, TOOL_OPEN + "\n"
        return tools, ""                       # "auto" (and anything unrecognised)
    if isinstance(choice, dict):
        kind = str(choice.get("type") or "").lower()
        if kind == "none":
            return None, ""
        if kind == "auto":
            return tools, ""
        if kind == "function":
            fn = choice.get("function") if isinstance(choice.get("function"), dict) else {}
            name = str(fn.get("name") or choice.get("name") or "")
            if name:
                named = [t for t in tools
                         if isinstance(t, dict)
                         and (t.get("function") or {}).get("name") == name]
                return (named or tools), f"{TOOL_OPEN}\n<function={name}>\n"
            return tools, TOOL_OPEN + "\n"
    return tools, ""


def _include_usage(body: dict) -> bool:
    """``stream_options.include_usage``: one extra chunk with ``choices: []``."""
    opts = body.get("stream_options")
    return bool(isinstance(opts, dict) and opts.get("include_usage"))


class AnswerFilter:
    """Splits the raw stream into (reasoning, content) increments.

    Two things are handled here and both are load-bearing for a thinking model:

    * the reasoning block -- with thinking on the *prompt* opens ``<think>``, so
      the generated text starts inside it.  The reasoning is handed out as
      ``reasoning_content`` deltas while it is being produced, and ``content``
      starts only after ``</think>``.  (Dropping it, or replaying it at the end,
      is what left an agent UI's thinking pane empty.)
    * ``<tool_call>`` markup -- once it appears, content stops and the call is
      delivered structured instead; a partial marker is held back so it cannot
      leak, and ``finish`` releases that tail at end of stream.
    """

    def __init__(self, prompt_opens_think: bool, marker: str = TOOL_OPEN) -> None:
        self.raw = ""
        self.marker = marker
        self.past_think = not prompt_opens_think
        self.think_pos = 0            # how much of the reasoning has been sent
        self.emitted = 0              # how much of the content has been sent
        self.done_tools = False

    def _content(self) -> str:
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
        """-> ``(reasoning_delta, content_delta)`` for this piece of text."""
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
        """Flush the tails at end of stream: ``(reasoning, content)``.

        The content hold-back exists so a partial ``<tool_call>`` never reaches the
        client; at end of stream it must be released, or the last few characters of
        every answer would be lost.  With tool calls the tail *is* markup (already
        delivered structured), so it is dropped.
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


# streaming tool calls
_CALL_OPEN = "<tool_call>"
_FN_OPEN = re.compile(r"<function=([^>\n]+)>")
_PARAM_OPEN = re.compile(r"<parameter=([^>\n]+)>")
_PARAM_CLOSE = "</parameter>"
_FN_CLOSE = "</function>"
_CALL_CLOSE = "</tool_call>"
_JSON_LITERAL_HEAD = "-0123456789tfn{["
_SHORT_ESCAPE = {"\n": "\\n", "\r": "\\r", "\t": "\\t", "\b": "\\b", "\f": "\\f"}


def _json_escape(text: str) -> str:
    """Escape a fragment exactly the way ``json.dumps(ensure_ascii=False)`` does.

    Byte-for-byte agreement matters: the fragments a client concatenates have to be
    the same string the non-streaming path returns, or a client that falls back to
    the non-stream JSON of the same call would see two different argument strings.
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
    """Keep the value typed like ``tool_call._coerce`` (the non-streaming path)."""
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
    """Turn the Qwen ``<tool_call>`` markup into OpenAI ``tool_calls`` deltas *as it
    is generated* -- the shape vLLM's streaming tool parsers emit.

    The engine streams the call token by token, but parsing only at the end (what
    this server used to do) means a client gets one fat delta after the whole call
    is complete: no progress for a long argument, and a UI that looks hung.  So the
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
    """

    def __init__(self, seed: str = "", max_calls: int | None = None,
                 marker: str = _CALL_OPEN) -> None:
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
        self.buf += piece
        return self._drain(final=False)

    def finish(self) -> list[dict]:
        return self._drain(final=True)

    def _drain(self, final: bool) -> list[dict]:
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


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "dcu-engine/0.1"

    def log_message(self, fmt, *args):  # noqa: A003 - stdlib hook name
        print(f"[http] {self.address_string()} {fmt % args}", flush=True)

    def _json(self, code: int, obj) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code: int, message: str, err_type: str = "invalid_request_error") -> None:
        self._json(code, {"error": {"message": message, "type": err_type, "code": code}})

    def _sse_start(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.close_connection = True         # SSE ends at EOF: no Content-Length

    def _sse(self, obj) -> None:
        self.wfile.write(f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _body(self) -> dict | None:
        try:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")
        except Exception as exc:  # noqa: BLE001
            self._error(400, f"bad request body: {exc}")
            return None

    def do_OPTIONS(self):  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "content-type, authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path in ("/health", "/healthz"):
            try:
                rep = STATE["engine"].call({"op": "ping"})
                # ``rep`` already carries model / model_path / arch from the engine
                self._json(200, {"status": "ok",
                                 "served_model_name": STATE["served_name"], **rep})
            except Exception as exc:  # noqa: BLE001
                self._json(503, {"status": "unavailable", "error": str(exc)})
        elif path == "/v1/models":
            self._json(200, {"object": "list", "data": [{
                "id": STATE["served_name"], "object": "model",
                "created": STATE["created"], "owned_by": "dcu-engine"}]})
        else:
            self._error(404, f"no route {self.path}")

    def do_POST(self):  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path != "/v1/chat/completions":
            self._error(404, f"no route {self.path}")
            return
        body = self._body()
        if body is None:
            return
        try:
            self._chat(body)
        except (BrokenPipeError, ConnectionResetError):
            print(f"[http] {self.address_string()} client hung up", flush=True)
        except Exception as exc:  # noqa: BLE001 - never kill the server on one bad request
            traceback.print_exc()
            try:
                self._error(500, f"{type(exc).__name__}: {exc}", "server_error")
            except Exception:
                pass

    def _chat(self, body: dict) -> None:
        raw_messages, images, err = _split_messages(body.get("messages") or [])
        if err:
            self._error(400, err)
            return
        messages = _norm_messages(raw_messages)
        if not messages:
            self._error(400, "`messages` must be a non-empty array")
            return
        if images and STATE.get("vision") is False:
            self._error(503, "this server was started with --no-vision: image inputs "
                             "are disabled", "server_error")
            return
        max_new = _max_tokens(body)
        temperature = float(body.get("temperature") or 0.0)
        stream = bool(body.get("stream"))
        model = body.get("model") or STATE["served_name"]
        roles = sorted({m["role"] for m in messages})
        think = _thinking_flag(body, STATE["enable_thinking"])
        tools, forced = _tool_plan(body)
        single_call = body.get("parallel_tool_calls") is False
        include_usage = _include_usage(body)
        think_keys = sorted(k for k in ("enable_thinking", "thinking", "reasoning_effort",
                                        "chat_template_kwargs") if k in body)
        print(f"[http] chat: {len(messages)} msgs {roles} max_new={max_new} "
              f"stream={stream} think={think} tools={bool(tools)} "
              f"images={len(images)} "
              f"tool_choice={json.dumps(body.get('tool_choice'), ensure_ascii=False)} "
              f"forced_call={bool(forced)} usage_chunk={include_usage} "
              f"think_keys={think_keys}", flush=True)
        req = {"op": "chat_stream" if stream else "chat",
               "messages": messages, "tools": tools,
               "tool_prefix": forced or None,
               # a forced call has to open the answer, so it cannot be reasoning first
               "enable_thinking": think and not forced,
               "max_new": max_new, "temperature": temperature,
               # the engine's own wall-clock cap: the only cutoff that also fires when
               # the client goes quiet without closing its socket
               "max_seconds": STATE.get("max_decode_seconds") or None}
        if images:
            # Base64 (already validated and re-encoded here) travels on the socket
            # JSON: the engine decodes, preprocesses and expands the placeholder
            # tokens itself with the checkpoint's own processor, so this process
            # stays stdlib-only and never has to agree with it numerically.
            req["images"] = images
        cid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())
        if stream:
            self._chat_stream(req, cid, created, model, single_call, include_usage)
        else:
            self._chat_once(req, cid, created, model, single_call)

    @staticmethod
    def _finish_reason(out: dict) -> str:
        """``length`` when the *effective* budget (after the window clamp) ran out."""
        if out.get("tool_calls"):
            return "tool_calls"
        if out.get("stopped") == "deadline":
            # the definition has no "deadline" value; ``length`` is the one that
            # means "cut short by the server" rather than "the model finished"
            return "length"
        n_new = int(out.get("n_new") or 0)
        n_max = int(out.get("max_new") or 0)
        return "length" if n_max > 0 and n_new >= n_max else "stop"

    @staticmethod
    def _usage(out: dict) -> dict:
        pt = int(out.get("n_prompt") or 0)
        ct = int(out.get("n_new") or 0)
        return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct,
                # how much of the prompt the radix prefix cache had already computed
                "prompt_tokens_details": {"cached_tokens":
                                          int(out.get("cached_tokens") or 0)}}

    def _chat_once(self, req: dict, cid: str, created: int, model: str,
                   single_call: bool = False) -> None:
        out = STATE["engine"].call(req)
        if not out.get("ok", True):
            # The engine says *why* and with which status: a bad image is a 400, a
            # server without a vision tower a 503.  Collapsing all of them into 500
            # would tell a client to retry a request that can never succeed.
            self._error(int(out.get("status") or 500),
                        f"engine: {out.get('error')}",
                        "invalid_request_error" if int(out.get("status") or 500) < 500
                        else "server_error")
            return
        calls = out.get("tool_calls") or []
        if single_call and len(calls) > 1:      # parallel_tool_calls: false
            calls = calls[:1]
        text = out.get("text") or ""
        if calls:
            # keep only the natural-language lead-in: the calls travel structured,
            # exactly like the streamed content (the template renders both).
            text = text.split(TOOL_OPEN, 1)[0].strip()
        message: dict = {"role": "assistant", "content": text or None}
        if calls:
            message["tool_calls"] = calls
        if out.get("thinking"):
            message["reasoning_content"] = out["thinking"]
        self._json(200, {
            "id": cid, "object": "chat.completion", "created": created, "model": model,
            "choices": [{"index": 0, "message": message,
                         "finish_reason": self._finish_reason({**out, "tool_calls": calls})}],
            "usage": self._usage(out),
        })

    def _chat_stream(self, req: dict, cid: str, created: int, model: str,
                     single_call: bool = False, include_usage: bool = False) -> None:
        base = {"id": cid, "object": "chat.completion.chunk", "created": created,
                "model": model}
        # The definition says: with ``stream_options.include_usage`` every other
        # chunk carries ``usage: null`` and one extra chunk with ``choices: []``
        # carries the real numbers just before ``[DONE]``.
        marker = {"usage": None} if include_usage else {}
        t_start = time.time()
        sent = 0

        def emit(choices: list, **extra) -> None:
            nonlocal sent
            sent += len(choices)
            self._sse({**base, "choices": choices, **marker, **extra})

        flt: AnswerFilter | None = None
        done: dict | None = None
        # One streaming tool-call parser for this reply, seeded with the forced
        # opener so a ``tool_choice: required`` call streams exactly like a natural
        # one (the opener lives in the prompt, not in the generated text).
        tstream = ToolStream(seed=req.get("tool_prefix") or "",
                             max_calls=1 if single_call else None)
        # The engine validates the request (image decode, processor expansion, the
        # placeholder self-check) *before* its ``start`` line, so its first line can
        # still be a rejection.  Nothing is written before that line arrives: an
        # image request that is refused then gets a real 400/503 with the JSON error
        # body an OpenAI client expects, instead of a 200 SSE stream carrying an
        # error event -- which no client retries or reports usefully.
        lines = STATE["engine"].stream(req)
        try:
            first = next(lines)
        except StopIteration:
            first = None
        if first is not None and first.get("type") == "error":
            self._error(int(first.get("status") or 400),
                        f"engine: {first.get('error') or 'rejected the request'}")
            return
        self._sse_start()
        emit([{"index": 0, "delta": {"role": "assistant", "content": ""},
               "finish_reason": None}])
        try:
            for line in itertools.chain([first] if first is not None else [], lines):
                if line is None:
                    continue
                if line.get("type") == "start":
                    flt = AnswerFilter(bool(line.get("prompt_opens_think")))
                    if line.get("forced_tool"):
                        # ``tool_choice: required``: the opener is already in the
                        # prompt, so the generated text is the *body* of the call --
                        # it must never go out as content, the call is delivered
                        # structured below.
                        flt.done_tools = True
                    continue
                if line.get("type") != "token":
                    done = line
                    break
                piece = line.get("text") or ""
                delta_r, delta_c = flt.push(piece) if flt else ("", "")
                for td in tstream.push(piece):
                    emit([{"index": 0, "delta": td, "finish_reason": None}])
                if delta_r:
                    # the reasoning block goes out as it is produced, so a client
                    # with a thinking pane shows it live (and never as content)
                    emit([{"index": 0, "delta": {"reasoning_content": delta_r},
                           "finish_reason": None}])
                if delta_c:
                    emit([{"index": 0, "delta": {"content": delta_c},
                           "finish_reason": None}])
        except Exception as exc:  # noqa: BLE001 - report into the stream, then close it
            traceback.print_exc()
            # ``client_hung_up`` is the fact that decides who cut the stream: if this
            # is False the engine itself stopped early, if it is True the answer the
            # client lost was cut on the way out (client, relay), not by the model.
            print(f"[http] stream aborted id={cid} sent={sent} "
                  f"wall={time.time() - t_start:.1f}s "
                  f"client_hung_up={isinstance(exc, (BrokenPipeError, ConnectionResetError))}",
                  flush=True)
            emit([{"index": 0, "delta": {}, "finish_reason": "error"}],
                 error={"message": f"{type(exc).__name__}: {exc}",
                        "type": "server_error"})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        if done is None or not done.get("ok", True):
            emit([{"index": 0, "delta": {}, "finish_reason": "error"}],
                 error={"message": f"engine: {(done or {}).get('error', 'no reply')}",
                        "type": "server_error"})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        if flt is not None and not flt.done_tools:
            tail_r, tail_c = flt.finish(bool(done.get("tool_calls")))
            if tail_r:      # the stream ended inside the reasoning block
                emit([{"index": 0, "delta": {"reasoning_content": tail_r},
                       "finish_reason": None}])
            if tail_c:
                emit([{"index": 0, "delta": {"content": tail_c}, "finish_reason": None}])
        for td in tstream.finish():          # truncated call: close what we have
            emit([{"index": 0, "delta": td, "finish_reason": None}])
        calls = done.get("tool_calls") or []
        if single_call and len(calls) > 1:       # parallel_tool_calls: false
            calls = calls[:1]
        if calls and not tstream.emitted_any:
            # Fallback for a call the streaming parser did not see (it would be a
            # bug in it): deliver the whole call in one delta, as before.  When the
            # fragments went out, repeating the complete call here would make the
            # client concatenate the arguments twice.
            emit([{"index": 0, "delta": {
                "tool_calls": [{"index": i, **c} for i, c in enumerate(calls)]},
                "finish_reason": None}])
        last = {"index": 0, "delta": {},
                "finish_reason": self._finish_reason({**done, "tool_calls": calls})}
        if include_usage:
            emit([last])
            self._sse({**base, "choices": [], "usage": self._usage(done)})
        else:
            # OpenAI omits usage unless it is asked for; keeping it on the last
            # chunk is the lenient behaviour existing clients (the bridge, the
            # probes) already read, so it stays when nobody asked for the extra chunk.
            self._sse({**base, "choices": [last], "usage": self._usage(done)})
        print(f"[http] stream done id={cid} n_new={done.get('n_new')} "
              f"finish={last['finish_reason']} stopped={done.get('stopped')} "
              f"tokens_sent={sent} wall={time.time() - t_start:.1f}s "
              f"prefill_ms={done.get('prefill_ms')} client_hung_up=False", flush=True)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def main() -> None:
    global MAX_IMAGES, MAX_IMAGE_BYTES, MAX_IMAGE_TOTAL_BYTES

    ap = argparse.ArgumentParser(description="OpenAI-compatible HTTP API in front of "
                                     "engine serve.py")
    ap.add_argument("--sock", default=SOCK,
                    help="engine unix socket (default: $ENGINE_SOCK or the serve.py default)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--served-model-name", default="",
                    help="model id reported by /v1/models (default: the engine's "
                         "checkpoint directory name)")
    ap.add_argument("--model", default="",
                    help="override the checkpoint path reported by /health "
                         "(default: whatever serve.py loaded)")
    ap.add_argument("--default-max-tokens", type=int, default=8192,
                    help="output budget when the client states none (0 = the whole "
                         "window). The engine is single-stream with no cancel path, so "
                         "an unbounded request that nobody waits for keeps later "
                         "requests queued until it ends by itself")
    ap.add_argument("--enable-thinking", action="store_true",
                    help="let the model reason by default (per-request override wins)")
    ap.add_argument("--max-decode-seconds", type=float, default=180.0,
                    help="wall-clock cap the engine enforces on one generation "
                         "(0 = none). This is the cutoff that also fires when a client "
                         "goes quiet without closing its socket: the single stream "
                         "would otherwise keep decoding for a request nobody reads, and "
                         "every later request queues behind it")
    ap.add_argument("--timeout", type=float, default=900.0,
                    help="per-request engine timeout, seconds")
    # -- image limits -------------------------------------------------------
    # These mirror the defaults serve.py starts with; they are *this* process's
    # cheap pre-check, and serve.py re-validates against its own configured limits
    # (whatever --image-max-pixels / --max-images it was actually started with).
    # Raising one here without raising it there just moves the 400 to the engine.
    ap.add_argument("--max-images", type=int, default=MAX_IMAGES,
                    help=f"images per request (default {MAX_IMAGES}); serve.py "
                         f"re-checks against its own --max-images")
    ap.add_argument("--max-image-bytes", type=int, default=MAX_IMAGE_BYTES,
                    help="decoded bytes allowed per image (default 8 MiB)")
    ap.add_argument("--max-image-total-bytes", type=int, default=MAX_IMAGE_TOTAL_BYTES,
                    help="base64 characters of all image parts of one request "
                         "(default 16 MiB)")
    args = ap.parse_args()

    # The module-level defaults are what the pure functions use; the flags only
    # exist because a deployment may want tighter limits than the defaults.
    MAX_IMAGES = int(args.max_images)
    MAX_IMAGE_BYTES = int(args.max_image_bytes)
    MAX_IMAGE_TOTAL_BYTES = int(args.max_image_total_bytes)

    engine = EngineClient(args.sock, timeout=args.timeout)
    for _ in range(3):
        try:
            info = engine.call({"op": "ping"})
            break
        except Exception as exc:  # noqa: BLE001
            print(f"[server] engine not reachable on {args.sock}: {exc}", flush=True)
            time.sleep(2)
    else:
        raise SystemExit(f"engine socket {args.sock} is not answering — start serve.py first")

    # The engine knows which checkpoint is loaded; only fall back to the flags.
    served = args.served_model_name or info.get("model") or "default"
    model_path = args.model or info.get("model_path") or ""
    STATE.update({"engine": engine, "model_path": model_path, "served_name": served,
                  "enable_thinking": args.enable_thinking,
                  "default_max_tokens": args.default_max_tokens or -1,
                  "max_decode_seconds": args.max_decode_seconds,
                  # what the engine can actually do with an image: the server may
                  # have been started with --no-vision, and answering 503 here saves
                  # a round trip that can only fail
                  "vision": bool(info.get("vision", True)),
                  "image_limits": {"max_images": MAX_IMAGES,
                                   "max_image_bytes": MAX_IMAGE_BYTES,
                                   "max_image_total_bytes": MAX_IMAGE_TOTAL_BYTES},
                  "created": int(time.time())})
    print(f"[server] engine up: {json.dumps(info, ensure_ascii=False)}", flush=True)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    print(f"[server] listening on http://{args.host}:{args.port}/v1 "
          f"(model id {served}, arch {info.get('arch')}, thinking={args.enable_thinking}, "
          f"max_decode_seconds={args.max_decode_seconds or 'none'}, "
          f"vision={STATE['vision']}, images<={MAX_IMAGES}x"
          f"{MAX_IMAGE_BYTES / 2**20:.0f}MiB)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
