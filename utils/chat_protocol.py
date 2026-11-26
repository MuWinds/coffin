"""Qwen3.5 一族的对话线格式：推理标记、工具调用标记和参数类型的唯一来源。

引擎、服务层和 OpenAI 外壳都从这里取同一份定义，qwen3.5（4B）与 occamy 1.0
（``qwen3_5_moe``）因此共用一套解析规则；两个 checkpoint 的差别只可能来自它们
各自的对话模板和各自的工具 schema，而不是散落在三个模块里的字面量。

Module: ``coffin/utils``。依赖图的叶子：只用标准库，``serving/``、
``entrypoints/openai/`` 和 ``tests/`` 都能依赖它，OpenAI 进程也不必 import
torch 或任何模型实现。

Constraints:
    * 标记的权威来源是 checkpoint 的对话模板。这里的默认值只是「模板没写出
      这一项」时的兜底，绝不覆盖模板里真实出现的写法——把 ``<function=``
      换成模板里的另一个词，会让所有工具调用静默地识别不出来。
    * 参数类型只认工具 schema 声明的那一个。模型输出的取值是纯文本，格式本身
      不携带类型；按字面量猜类型会把 ``0755``、``00123``、``13800138000`` 这种
      字符串型参数改成数字，工具自己校验时就会失败。
    * 闭合标记一律由 ``_close`` 拼出来，不写字面量。源码里出现完整的一对
      ``<tag>`` / 闭合标签时，仓库里的 XML 风格工具链会把它当成自己的标签结束，
      把文件内容截断。
"""

from __future__ import annotations

import ast
import json
from pathlib import Path


def _close(tag: str) -> str:
    """一个开标签对应的闭合标签。

    Args:
        tag: 标签名，不带尖括号。

    Returns:
        ``<``、``/``、标签名和 ``>`` 拼成的闭合标签。
    """
    return "</" + tag + ">"


#: 参数闭合标记。与工具 schema、模板三处共用同一份，避免各模块各写一遍。
PARAM_CLOSE = _close("parameter")

#: 后备标记表：Qwen3 / Qwen3.5 / Qwen3.5-MoE 模板里实际出现的字面量。
DEFAULT_MARKERS: dict[str, str] = {
    "think_open": "<think>",
    "think_close": _close("think"),
    "tool_open": "<tool_call>",
    "tool_close": _close("tool_call"),
    "function_open": "<function=",
    "function_close": _close("function"),
    "param_open": "<parameter=",
    "param_close": PARAM_CLOSE,
}

#: 标记表里每一项在模板文本里应该能找到的写法。模板缺了哪一项，工具调用或推理段
#: 就只可能由模型自己凭空写出来，而不是这一族的训练格式。
_MARKER_KEYS = tuple(DEFAULT_MARKERS)

#: schema 里按「字符串」处理的一批写法。OpenAI 的工具定义允许 ``type`` 写成
#: 数组形式（``["string", "null"]``），分解在 :func:`param_types` 里完成。
_STRING_TYPES = ("string", "str", "text", "varchar", "char", "enum")

#: 整数族的类型名前缀。
_INT_PREFIXES = ("int", "uint", "long", "short", "unsigned")

#: 浮点族的类型名前缀。
_FLOAT_PREFIXES = ("num", "float", "double", "decimal")

#: 布尔族的类型名。
_BOOL_TYPES = ("boolean", "bool", "binary")


def markers_from(payload) -> dict[str, str]:
    """把外部给的一份标记表补齐成完整的一份。

    引擎在 ``ping`` 和流式 ``start`` 行里把标记表带下来，外壳不该相信它总是
    完整的：缺项时用这一族的后备值补上，取值不是字符串的项直接丢掉。

    Args:
        payload: 引擎给的映射，或 ``None``（没有引擎信息时）。

    Returns:
        键齐全、值非空的标记表。
    """
    out = dict(DEFAULT_MARKERS)
    if isinstance(payload, dict):
        for key in _MARKER_KEYS:
            value = payload.get(key)
            if isinstance(value, str):
                out[key] = value
    return out


def template_presence(template: str | None) -> dict[str, bool]:
    """模板文本里出现了哪些线格式标记。

    启动时用它把「这个 checkpoint 的模板根本没提工具调用」和「模型不肯调用」
    区分开：前者是配置问题，日志里必须点名，否则整套适配看起来正常却永远解析
    不出调用。

    Args:
        template: checkpoint 的对话模板原文；没有模板时为 ``None``。

    Returns:
        标记名 -> 模板里是否出现。没有模板时全部为 ``False``。
    """
    if not isinstance(template, str):
        return {key: False for key in _MARKER_KEYS}
    return {key: value in template for key, value in DEFAULT_MARKERS.items()}


#: checkpoint 目录里除 ``tokenizer_config.json`` 之外还可能放模板的文件名。
_TEMPLATE_FILES = ("chat_template.jinja", "chat_template.txt")


def ensure_template(tok, model_dir) -> str | None:
    """保证分词器手里有一份可用的对话模板。

    Occamy 1.0 把模板放在 checkpoint 目录下的 ``chat_template.jinja`` 单文件里，
    而不是写进 ``tokenizer_config.json``；``AutoTokenizer`` 不读那个单文件，
    ``apply_chat_template`` 于是直接报「chat template is not defined」。两个模型族
    的模板来源不同，所以统一在这里补齐，调用方只看 ``tok.chat_template``。

    Args:
        tok: 已加载的分词器。
        model_dir: checkpoint 目录。

    Returns:
        生效的模板文本。分词器自己带模板时返回它；连文件都没有时返回 ``None``，
        调用方据此在请求上报一个可读的拒绝，而不是让 jinja 抛 traceback。
    """
    tpl = getattr(tok, "chat_template", None)
    if isinstance(tpl, dict):
        # transformers 允许多命名模板；不带名字渲染时它取 ``default``。
        tpl = tpl.get("default") or next(
            (v for v in tpl.values() if isinstance(v, str)), None)
    if isinstance(tpl, str) and tpl.strip():
        try:
            tok.chat_template = tpl
        except Exception:      # noqa: BLE001 - 只读属性时保持原样即可
            pass
        return tpl
    path = Path(model_dir)
    for name in _TEMPLATE_FILES:
        candidate = path / name
        if not candidate.is_file():
            continue
        text = candidate.read_text(encoding="utf-8")
        if not text.strip():
            continue
        try:
            tok.chat_template = text
        except Exception:      # noqa: BLE001
            return None
        return text
    return None


def opens_reasoning(text: str, markers: dict[str, str] | None = None) -> bool:
    """渲染好的 prompt 是否停在未闭合的推理段内部。

    模板把开标签发在 prompt 末尾时（Qwen3.5 起的新式模板），生成的第一个 token
    就已经在推理段里；流式过滤器必须知道这一点，否则整段推理会被当成答案。

    Args:
        text: 渲染好的 prompt 原文。
        markers: 标记表；``None`` 用这一族的后备值。

    Returns:
        最后出现的是开标签、且它后面只剩空白时为 ``True``。
    """
    if not text:
        return False
    mk = markers_from(markers)
    tail = text.rstrip()
    i = tail.rfind(mk["think_open"])
    if i < 0:
        return False
    return i > tail.rfind(mk["think_close"]) and i + len(mk["think_open"]) == len(tail)


def split_thinking(text: str, markers: dict[str, str] | None = None) -> tuple[str, str]:
    """把推理段和答案分开，返回 ``(thinking, answer)``。

    两种模板形态都要认：Qwen3.5 起模板把开标签发在 prompt 里，生成文本只剩闭合
    标签；更早的模板和裸 prompt 会由模型自己写出整对标签。推理段被截断（没有闭合
    标签）时，标签之后的部分仍然算推理——那一轮没有答案。

    Args:
        text: 从 prompt 之后切出来的原始生成文本。
        markers: 标记表；``None`` 用这一族的后备值。

    Returns:
        ``(thinking, answer)``，两者都已去空白。没有推理段时 ``thinking`` 为
        ``""``。
    """
    if not text:
        return "", ""
    mk = markers_from(markers)
    open_tag, close_tag = mk["think_open"], mk["think_close"]
    i_open = text.find(open_tag)
    i_close = text.find(close_tag)
    if i_open >= 0 and i_close > i_open:
        body = text[i_open + len(open_tag):i_close]
        rest = text[:i_open] + text[i_close + len(close_tag):]
        return body.strip(), rest.strip()
    if i_close >= 0:
        return text[:i_close].strip(), text[i_close + len(close_tag):].strip()
    if i_open >= 0:
        return text[i_open + len(open_tag):].strip(), text[:i_open].strip()
    return "", text.strip()


def strip_thinking(text: str, markers: dict[str, str] | None = None) -> str:
    """``text`` 里属于答案的那一半，供丢弃推理段的调用方使用。

    Args:
        text: 原始生成文本；允许为空。
        markers: 标记表；``None`` 用这一族的后备值。

    Returns:
        去掉推理块并去过空白的答案。
    """
    return split_thinking(text, markers)[1]


def param_types(tools) -> dict[str, dict[str, str]]:
    """工具 schema 里每个函数声明了哪些参数、各是什么类型。

    两种形状都接受：OpenAI 的 ``{"type": "function", "function": {...}}``，以及
    已弃用但仍有客户端在发的扁平 ``{"name": ..., "parameters": ...}``。参数表既
    可以嵌在 ``properties`` 下，也有工具把参数直接平铺在 ``parameters`` 里。

    Args:
        tools: 请求里的 ``tools`` 数组，或 ``None``。

    Returns:
        函数名 -> (参数名 -> 小写类型名)。没声明类型的参数对应空串，表示「不要
        猜类型，原样返回文本」。没有 ``tools`` 时是空字典。
    """
    out: dict[str, dict[str, str]] = {}
    for item in tools or []:
        if not isinstance(item, dict):
            continue
        fn = item.get("function")
        if not isinstance(fn, dict):
            fn = item
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            continue
        params = fn.get("parameters")
        props = None
        if isinstance(params, dict):
            props = params.get("properties")
            if not isinstance(props, dict):
                props = params
        spec: dict[str, str] = {}
        for key, value in (props or {}).items():
            if not isinstance(key, str):
                continue
            if isinstance(value, dict):
                kind = value.get("type")
                if isinstance(kind, list):
                    kind = next((k for k in kind if k != "null"), None)
                spec[key] = str(kind).strip().lower() if kind else ""
            else:
                spec[key] = ""
        out[name] = spec
    return out


def param_type(specs: dict[str, dict[str, str]], function: str,
               param: str) -> str:
    """``function`` 上名为 ``param`` 的参数声明了什么类型。

    Args:
        specs: :func:`param_types` 的结果。
        function: 函数名；流式路径在读到 ``<function=`` 开头时才知道。
        param: 参数名。

    Returns:
        小写类型名；函数或参数没声明时是空串——调用方据此保持字符串，而不是替
        模型猜一个类型。
    """
    return (specs.get(function) or {}).get(param, "")


def is_string_type(kind: str) -> bool:
    """这个类型名是不是「原样返回文本」的那一类。

    Args:
        kind: 小写类型名，允许为空。

    Returns:
        字符串族（含 ``enum``）或没有声明类型时为 ``True``。
    """
    return not kind or kind in _STRING_TYPES


def convert_param(value: str, kind: str) -> object:
    """按工具 schema 声明的类型，把一段原始取值转成参数值。

    没有声明类型时原样返回文本：模型输出的取值不带类型信息，猜得越宽，改坏工具
    自己校验的参数的机会越大。字符串族同样原样返回，``0755`` 和 ``00123`` 因此
    保持字符串。

    Args:
        value: 参数标签之间的原始文本，格式里它单独占一行。
        kind: 小写类型名；空串表示未声明。

    Returns:
        声明了数字/布尔/对象/数组时是对应的 Python 值；转换失败、未声明或属于
        字符串族时是去过空白后的原文。字符串参数里的 ``null`` 也按原文保留，
        因为这种工具想要的正是那四个字符。
    """
    s = value.strip()
    if is_string_type(kind):
        return s
    if s.lower() == "null":
        return None
    if kind.startswith(_INT_PREFIXES):
        try:
            return int(s)
        except (TypeError, ValueError):
            return s
    if kind.startswith(_FLOAT_PREFIXES):
        try:
            v = float(s)
        except (TypeError, ValueError):
            return s
        # 整数写法的 ``3.0`` 收敛回 ``3``：工具 schema 里 number 通常也接受整数，
        # 而 JSON 里带小数点的整数值会让严格校验的实现拒收。
        return int(v) if v.is_integer() else v
    if kind in _BOOL_TYPES:
        low = s.lower()
        # 只有 true/false 才算布尔。``yes``、``1`` 这类写法留给工具自己判，
        # 静默取 False 等于把用户说的「是」变成「否」。
        return True if low == "true" else False if low == "false" else s
    if kind.startswith(("object", "dict", "array", "arr", "list")):
        try:
            return json.loads(s)
        except (TypeError, ValueError):
            return s
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError, TypeError):
        return s
