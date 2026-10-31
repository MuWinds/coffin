"""OpenAI 请求归一化，以及代价很低的图像校验。

Module: ``coffin.serving``。只用标准库，并且跑在任何东西到达引擎 socket 之前。

Constraints:
    * 校验必须便宜：只做 base64 解码和大小检查，不解码像素。真正的图像
      处理在引擎侧的处理器里。
    * 纯文本请求的渲染结果必须与改造前逐字节一致，任何归一化都只能作用于
      带图消息。
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
    """客户端送来的、本进程拒绝接收的图像片段。

    继承 ``ValueError``，只捕获宽泛请求异常的调用方也能兜住；处理器把它映射
    成 HTTP 400，因为每次触发都是客户端送错了东西，改一下请求就能解决。
    """


def _decode_data_url(url: str) -> tuple[bytes, str]:
    """把 ``data:image/png;base64,...`` 解成 ``(原始字节, mime)``。

    只接受 base64 data URL：v1 不抓 ``http(s)`` URL（容器没有出网能力，抓取
    本身就是 SSRF 面），而且让*服务端*去做 I/O 的 URL 正是调用方无法审计的
    东西。

    Args:
        url: 客户端原样送来的 ``image_url.url`` 值。

    Returns:
        ``(raw, mime)``：解码后的字节和归一化后的媒体类型。这里不需要任何
        超出引擎所需的重编码——调用方会再把这些字节 base64 一次发给 socket。

    Raises:
        ImagePartError: 值为空；不是 ``data:`` URL（远端 ``http(s)`` URL 会在
            报错信息里点名受支持的写法）；媒体类型不支持；不是 base64；解码
            结果为空；或超过 ``MAX_IMAGE_BYTES``。这些都是请求错误：回 400，
            让客户端重发一张更小或编码正确的图像。
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
    """把一条消息的 ``content`` 拆成（模板可用的内容, 图像载荷）。

    OpenAI 的视觉消息是 *parts* 数组，其中一段放图像字节，其余放文本。
    checkpoint 的对话模板只认识两样东西：文本，以及 ``{"type": "image"}`` 标记；
    模板把这个标记渲染成 ``<|vision_start|><|image_pad|><|vision_end|>``（引擎
    侧处理器随后再把这些占位符展开成每个视觉 patch 一个 token）。所以图像
    片段要从数组里摘出来——字节变成引擎请求上的 ``images[]``——原位只留那个
    标记，文本与图像的*顺序*因此与客户端写的完全一致。

    Args:
        content: 消息的 ``content``：字符串、parts 数组或 ``None``。
        budget: 每个请求共用的可变累加器 ``{"images": n, "b64": 字符数}``，
            让限额作用在整个请求而不是单条消息上。``b64`` 统计的是客户端
            发来的 *data URL* 字符数——真正走线、也真正占内存的量——而不是
            只有载荷部分。

    Returns:
        ``(content, images)``。消息里没有图像片段时 ``content`` 保持原值，
        纯文本请求路径因此逐字节不变。

    Raises:
        ImagePartError: 某个片段不是对象；是视频或音频片段（本构建无法给它们
            的 token 数设上界）；类型未知；没有可用的 ``image_url.url``；或
            请求超出图像数量、总大小预算。处理器把它转成 400。
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
        # 解码之前先查：已经超量了的请求不该再为多解一张图付出 base64 的代价。
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
        parts.append({"type": "image"})      # 模板自己的标记，不是 OpenAI 的字段
    return parts, images


def _split_messages(messages: list) -> tuple[list, list[dict], str | None]:
    """把请求 ``messages`` 里的图像全部摘出来。

    有三条规则不能留给引擎处理，否则客户端只会收到一个看不懂的失败：

    * ``system`` 消息不能带图——模板会直接抛异常，客户端就会拿到一个来自
      自己无法调试的生成的 500；
    * 每请求的图像数量与总大小预算（见 ``_split_parts``）；
    * 图像片段出现在模板从不把其内容渲染成媒体的角色里。

    Args:
        messages: 请求原始的 ``messages`` 数组，未校验。

    Returns:
        ``(messages, images, error)``。``error`` 是可读的 400 提示或 ``None``；
        一旦它有值，另外两个都是空的。``images`` 是引擎请求的载荷
        （``{"b64", "mime"}``），按 prompt 顺序排列。
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
                # 模板自己会为这个抛异常，所以这里回一条说明该怎么办的提示，
                # 别把 jinja 的 traceback 甩给客户端。
                return [], [], ("a system message cannot contain images: the chat "
                                "template rejects it; put the image in a user turn")
            if role == "assistant":
                # 历史的 assistant 轮次只承载文本；图像在那里没有对应的模板
                # 渲染，会被静默丢掉。
                return [], [], ("an assistant message cannot carry images; put the "
                                "image in a user or tool message")
            # ``tool`` 是接受的：调了截图工具的 agent 会把截屏当成工具结果拿
            # 回来，这是图像进入本服务最常见的途径。模板渲染它的方式与用户
            # 图像完全相同（消息里出现 vision_start/image_pad/vision_end），
            # 处理器按 prompt 顺序展开，所以在这里收下载荷就是唯一需要的
            # 特殊处理。
            images.extend(found)
        m["content"] = content
        out.append(m)
    return out, images, None


def _text_of(content) -> str:
    """把 OpenAI 的「字符串或 parts 数组」内容压成模板要的纯字符串。

    Args:
        content: 字符串、parts 数组、``None``，或客户端发来的任何别的东西。

    Returns:
        拼接后的文本。图像片段在这里不出力——调用方只在消息无图时才会走到
        这里（见 ``_template_content``）。既不是字符串也不是列表的值会被
        转成字符串而不是丢掉，形状再怪的客户端也还能渲染出点东西。
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
    """``content`` parts 数组里是否带图像标记。

    Args:
        content: 消息的 ``content``。

    Returns:
        至少含一个图像片段的 parts 数组返回 ``True``。``_template_content``
        据此决定给模板字符串还是 parts 数组。
    """
    return isinstance(content, list) and any(
        isinstance(p, dict) and p.get("type") in ("image", "image_url") for p in content)


def _template_content(content):
    """一条消息的 ``content`` 应该以什么形状交给对话模板。

    纯文本内容压成字符串，与改造前完全一样，纯文本请求因此逐字节不变。带图
    消息保留 *parts 数组*，因为这是模板唯一能理解媒体的形状：每个
    ``{"type": "image"}`` 标记原位变成
    ``<|vision_start|><|image_pad|><|vision_end|>``，客户端写的图文交替顺序
    原样保留。

    Args:
        content: 消息的 ``content``。

    Returns:
        纯文本内容返回纯字符串，带图内容返回重写后的 parts 数组。
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
    """把客户端的消息列表修成 checkpoint 里 jinja 模板能接受的样子。

    真正会把它弄坏的有三种形状，都是从真实客户端上见到的：

    * ``content`` 是 parts 数组；
    * 模板不认识的 role——OpenAI 较新的 ``developer``（或者某个自造 role 的
      harness）会让模板抛 *"Unexpected message role."*，整个请求直接失败；
    * ``system`` 消息不在首位，或者根本没有 user 轮次，模板都会直接拒绝；

    还有一种：``arguments`` 是 OpenAI 的 JSON *字符串*而不是映射，而模板要用
    ``|items`` 遍历这个字段。

    带图像的 parts 数组*不*压平：``{"type": "image"}`` 由模板自己渲染
    （见 ``_template_content``）。

    Args:
        messages: 已经由 ``_split_messages`` 摘掉图像载荷的消息。

    Returns:
        模板期望形状的归一化消息。非字典项直接丢掉而不是抛异常，请求残缺
        也仍然值得回一个结果。模板还要求有一个 user 轮次和位于首位的 system
        轮次，客户端没写时这里补上。
    """
    out: list[dict] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        m = dict(msg)
        role = str(m.get("role") or "user").lower()
        if role == "developer":              # system 角色的新名字
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
    elif any(m["role"] == "system" for m in out):     # 提到最前，其余的降级
        first = next(m for m in out if m["role"] == "system")
        out = [first] + [{**m, "role": "user"} if m["role"] == "system" else m
                         for m in out if m is not first]
    if out and not any(m["role"] == "user" for m in out):
        out.append({"role": "user", "content": "(continue)"})
    return out


def _max_tokens(body: dict, state: dict | None = None) -> int:
    """客户端要的输出预算；没写就用配置的默认值。

    ``max_completion_tokens`` 是较新 OpenAI 客户端（以及编码 agent）发的字段；
    以前只读 ``max_tokens`` 会落到一个很小的默认值上，而被截断的恰好是人们
    向长上下文模型要的东西——一整个代码文件。

    客户端什么都没说时，用的是 ``--default-max-tokens`` 而不是「整个窗口」：
    引擎是单流的，一个已经没人在等、又没有长度上界的请求会让后面所有请求
    一直排队，直到它自己跑完（实测：没人连着也能生成 117 s）。

    Args:
        body: 请求体。

    Returns:
        以 token 计的预算。``-1`` 表示「整个窗口」，引擎会再按 prompt 占掉的
        空间收紧。解析不出来的值会往下一个写法退，而不是让请求失败。
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
    """这个请求要不要模型的推理段。

    各家客户端的写法不同，实际见过的形状都接受：``enable_thinking``
    （vLLM/Qwen）、``chat_template_kwargs.enable_thinking``、``thinking``
    （布尔/字符串/``{"type": "enabled"}``），以及 ``reasoning_effort``
    （只要不是 none/off，新的 OpenAI 风格 agent 就是这么问的）。

    优先级由下面的检查顺序决定：显式的 ``enable_thinking`` 或 ``thinking``
    压过 ``reasoning_effort`` 这个近似写法，两个都设时以具体字段为准。

    Args:
        body: 请求体。
        default: 服务级的 ``--enable-thinking`` 选择，请求没表态时用它。

    Returns:
        模板是否应该打开推理段。
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
    """按 OpenAI 的 tool_choice 给出 ``(要渲染的 tools, 强制的 assistant 开头)``。

    定义（``ChatCompletionToolChoiceOption``）规定：``none``——模型不许调用工具；
    ``auto``（带 tools 时的默认值）——由模型自己决定；``required``——它**必须**
    调用一个或多个工具；``{"type": "function", "function": {"name": X}}``——必须
    恰好调用 X。已废弃但仍在定义里的 ``functions`` / ``function_call`` 字段
    意思相同，在这里映射到同样两个字段。

    ``required`` 靠*强制开头*落实：引擎把模板的调用头追加到 prompt 后面，这一轮
    就只能续写成一个调用（4B 模型在挂了工具时经常改用散文回答——正是被报告
    的那个症状）。指定了名字时函数名也一起强制。

    Args:
        body: 请求体。

    Returns:
        ``(tools, opener)``。选择为 ``none`` 时 ``tools`` 是 ``None``，模板
        因此一个工具 schema 都不渲染。``opener`` 是追加到 assistant 头后面的
        模板文本，没有强制时为空。不认识的选择字符串按 ``auto`` 处理：客户端
        发来本构建不认识的取值时，它的意思就是让模型自己决定。
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
        return tools, ""                       # "auto"，以及任何认不出的值
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
    """``stream_options.include_usage``：额外一块 ``choices: []`` 的用量块。

    Args:
        body: 请求体。

    Returns:
        客户端是否要了定义里那个独立的用量块。默认不发，因为定义规定除非
        显式要求，否则省略 usage。
    """
    opts = body.get("stream_options")
    return bool(isinstance(opts, dict) and opts.get("include_usage"))
