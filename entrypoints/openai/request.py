"""OpenAI request normalization and cheap image validation.

Module: ``coffin.serving``. This module is standard-library only and runs before
anything reaches the engine socket.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import re

TOOL_OPEN = "<tool_call>"
MAX_IMAGE_BYTES = int(os.environ.get("ENGINE_MAX_IMAGE_BYTES", 8 * 1024 * 1024))
MAX_IMAGES = int(os.environ.get("ENGINE_MAX_IMAGES", 4))
MAX_IMAGE_TOTAL_BYTES = int(
    os.environ.get("ENGINE_MAX_IMAGE_TOTAL_BYTES", 16 * 1024 * 1024)
)
IMAGE_MIMES = (
    "image/png", "image/jpeg", "image/jpg", "image/webp", "image/bmp", "image/gif",
)
_DATA_URL = re.compile(
    r"^data:(?P<mime>[-\w.+]+/[-\w.+]+)?(?P<params>[^,]*),"
    r"(?P<data>.*)$",
    re.DOTALL,
)

class ImagePartError(ValueError):
    """A client-supplied image part this process refuses.

    Subclasses ``ValueError`` so a caller that only catches broad request errors
    still handles it; the handler maps it to HTTP 400, because every trigger is
    something the client sent wrong and can fix by rewriting the request.
    """


def _decode_data_url(url: str) -> tuple[bytes, str]:
    """``data:image/png;base64,...`` -> ``(raw bytes, mime)``.

    Only base64 data URLs are accepted: v1 does not fetch ``http(s)`` URLs (the
    container has no egress and a fetch would be an SSRF surface), and a URL that
    would make the *server* do the I/O is exactly the thing a caller cannot audit.

    Args:
        url: The ``image_url.url`` value as the client sent it.

    Returns:
        ``(raw, mime)``: the decoded bytes and the normalised media type. There
        is no need to re-encode here beyond what the engine expects -- the caller
        base64s the raw bytes for the socket request.

    Raises:
        ImagePartError: The value is empty, is not a ``data:`` URL (a remote
            ``http(s)`` URL names the supported form in the message), carries an
            unsupported media type, is not base64, decodes to nothing, or exceeds
            ``MAX_IMAGE_BYTES``. All are request errors: report 400 and let the
            client resend a smaller or correctly encoded image.
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
    image bytes and the rest carry text. The checkpoint's chat template knows only
    two things: text, and a ``{"type": "image"}`` marker that it renders as
    ``<|vision_start|><|image_pad|><|vision_end|>`` (the placeholders the engine's
    processor later expands to one token per visual patch). So the image parts are
    lifted out of the array -- their bytes become ``images[]`` on the engine
    request -- and replaced in place by that marker, which keeps the *order* of
    text and images exactly as the client wrote it.

    Args:
        content: The message's ``content``: a string, a parts array, or ``None``.
        budget: Mutable per-request accumulator ``{"images": n, "b64": chars}``, so
            the limits apply to the whole request rather than per message. ``b64``
            counts the *data URL* characters the client sent -- what travels over
            the wire and is held in memory -- not just the payload.

    Returns:
        ``(content, images)``. ``content`` is the untouched value when the message
        has no image part, which is what keeps the text-only request path
        byte-for-byte what it was.

    Raises:
        ImagePartError: A part is not an object, is a video or audio part (this
            build cannot bound their token count), is an unknown type, carries no
            usable ``image_url.url``, or the request is over its image count or
            total-size budget. The handler turns it into 400.
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
        # Checked before the decode: a request that already has too many images
        # should not pay to base64-decode one more.
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

    The three rules that cannot be deferred to the engine, because the engine
    would surface them as an opaque failure:

    * a ``system`` message may not carry an image -- the template raises on it, so
      the client would otherwise get a 500 from a generation it cannot debug;
    * the per-request count and size budget (see ``_split_parts``);
    * an image part in a role whose content the template never renders as media.

    Args:
        messages: The request's raw ``messages`` array, unvalidated.

    Returns:
        ``(messages, images, error)``. ``error`` is a human-readable 400 message
        or ``None``; when it is set the other two are empty. ``images`` are the
        engine-request payloads (``{"b64", "mime"}``), in prompt order.
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
            # way images enter this server. The template renders it exactly like a
            # user image (vision_start/image_pad/vision_end inside the message),
            # and the processor expands it in prompt order, so collecting the
            # payload here is the only special case needed.
            images.extend(found)
        m["content"] = content
        out.append(m)
    return out, images, None


def _text_of(content) -> str:
    """Flatten OpenAI's string-or-parts content into the plain string the template wants.

    Args:
        content: A string, a parts array, ``None``, or anything else a client sent.

    Returns:
        The concatenated text. Image parts contribute nothing here -- the caller
        only reaches this for a message with no image (see ``_template_content``).
        A value that is neither a string nor a list is stringified rather than
        dropped, so an odd client shape still renders something.
    """
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
    """Whether a ``content`` parts array carries an image marker.

    Args:
        content: The message's ``content``.

    Returns:
        ``True`` for a parts array with at least one image part. Drives the
        string-vs-parts decision in ``_template_content``.
    """
    return isinstance(content, list) and any(
        isinstance(p, dict) and p.get("type") in ("image", "image_url") for p in content)


def _template_content(content):
    """What the chat template should receive for one message's ``content``.

    Text-only content is flattened to a string, exactly as before, so a text-only
    request renders byte-for-byte what it used to. A message that carries an image
    keeps its *parts array*, because that is the only shape the template
    understands for media: each ``{"type": "image"}`` marker becomes
    ``<|vision_start|><|image_pad|><|vision_end|>`` in place, preserving the
    interleaving of text and images the client wrote.

    Args:
        content: The message's ``content``.

    Returns:
        A plain string for text-only content, or a rewritten parts array for
        content with an image.
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

    Args:
        messages: The request's messages, after ``_split_messages`` removed the
            image payloads.

    Returns:
        The normalised messages in the template's expected shape. A non-dict entry
        is dropped rather than raising, because a partial request is still worth
        answering. The template also requires a user turn and a leading system
        turn, so those are synthesised when the client left them out.
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


def _max_tokens(body: dict, state: dict | None = None) -> int:
    """The output budget the client asked for, else the configured default.

    ``max_completion_tokens`` is the field the newer OpenAI clients (and coding
    agents) send; reading only ``max_tokens`` used to fall through to a small
    default, which truncates exactly the answers people ask a long-context model
    for -- a file of code.

    When the client states nothing at all, ``--default-max-tokens`` applies rather
    than "the whole window": the engine is single-stream, so an unbounded request
    that nobody is waiting for any more keeps every later request queued until it
    finishes on its own (measured: a 117 s generation with no client attached).

    Args:
        body: The request body.

    Returns:
        The budget in tokens. ``-1`` means "the whole window", and the engine
        clamps it to the room the prompt leaves. An unparsable value falls through
        to the next spelling rather than failing the request.
    """
    for key in ("max_completion_tokens", "max_tokens", "max_new_tokens"):
        val = body.get(key)
        if val is not None:
            try:
                return max(1, int(val))
            except (TypeError, ValueError):
                pass
    return int((state or {}).get("default_max_tokens", -1))


def _thinking_flag(body: dict, default: bool) -> bool:
    """Does this request want the model's reasoning block?

    Clients spell it differently, so all the shapes actually seen are accepted:
    ``enable_thinking`` (vLLM/Qwen), ``chat_template_kwargs.enable_thinking``,
    ``thinking`` as a bool/string/``{"type": "enabled"}``, and ``reasoning_effort``
    (anything but none/off, which is how the newer OpenAI-style agents ask).

    The precedence is fixed by the order of the checks below: an explicit
    ``enable_thinking`` or ``thinking`` wins over the ``reasoning_effort``
    approximation, so setting both gives the specific field.

    Args:
        body: The request body.
        default: The server-wide ``--enable-thinking`` choice, used when the
            request says nothing.

    Returns:
        Whether the template should open the reasoning block.
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
    is exactly the reported symptom). With a named choice the function name is
    forced too.

    Args:
        body: The request body.

    Returns:
        ``(tools, opener)``. ``tools`` is ``None`` when the choice is ``none``, so
        the template renders no tool schemas at all. ``opener`` is the template
        text appended to the assistant header, empty when nothing is forced. An
        unrecognised choice string is treated as ``auto``: that is what a client
        means when it sends a value this build does not know.
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
        return tools, ""                       # "auto", and anything unrecognised
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
    """``stream_options.include_usage``: one extra chunk with ``choices: []``.

    Args:
        body: The request body.

    Returns:
        Whether the client asked for the definition's separate usage chunk. It
        is absent by default because the definition says usage is omitted unless
        requested.
    """
    opts = body.get("stream_options")
    return bool(isinstance(opts, dict) and opts.get("include_usage"))
