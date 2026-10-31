"""推理段与工具调用的增量流解析器。

Module: ``coffin.serving``。这两个解析器把引擎的原始 token 流转成 OpenAI 兼容
的 delta，全程不碰模型，也不碰 HTTP 传输层。

Constraints:
    * 增量解析：每个 ``push`` 只扫新追加的文本，整体代价是 O(tokens)，绝不
      重扫整段回答。
    * 流式与非流式两条路径产出的 ``arguments`` 必须是同一个字符串，否则
      客户端在同一次调用的两种模式之间切换会看到不同的参数。
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
    """把原始流切成（推理, 内容）两种增量。

    这里处理两件事，对思考型模型都是承重的：

    * 推理段——思考开在 *prompt* 里时会先出现 ``<think>``，生成的文本因此从
      推理段内部开始。推理边产生边以 ``reasoning_content`` 增量发出，
      ``content`` 要等 ``</think>`` 之后才开始。把它丢掉或者压到最后再补，
      正是 agent 界面思考面板空白的原因。
    * ``<tool_call>`` 标记——一旦出现，内容输出停止，调用改以结构化字段交付；
      不完整的标记会被扣住以免泄漏，``finish`` 在流末尾放行这段尾巴。

    刻意做成增量：每个 ``push`` 只扫新追加的文本，所以过滤器是 O(tokens)
    而不是每次重扫整段回答。
    """

    def __init__(self, prompt_opens_think: bool, marker: str = TOOL_OPEN) -> None:
        """为一次回复建一个过滤器。

        Args:
            prompt_opens_think: 渲染后的 prompt 是否已经在未闭合的 ``<think>``
                内部结束，这个信息由引擎在 ``start`` 行里给出。为否时，整段
                回答从第一个 token 起就是 content。
            marker: 用来截断内容的工具调用开头标记。
        """
        self.raw = ""
        self.marker = marker
        self.past_think = not prompt_opens_think
        self.think_pos = 0            # 已发出的推理文本长度
        self.emitted = 0              # 已发出的内容文本长度
        self.done_tools = False

    def _content(self) -> str:
        """下一个内容增量，同时扣住可能是半个工具标记的尾巴。

        Returns:
            现在可以安全发出的内容文本。尾巴仍可能长成完整标记时返回空——先
            发出去、后来才发现，就会把标记文本漏进回答里。
        """
        if self.done_tools:
            return ""
        body = self.raw[self.emitted:]
        k = body.find(self.marker)
        if k >= 0:
            self.done_tools = True
            out, self.emitted = body[:k], self.emitted + k
            return out
        keep = len(self.marker) - 1          # 半个标记绝不发出
        if len(body) <= keep:
            return ""
        out = body[:len(body) - keep]
        self.emitted += len(out)
        return out

    def push(self, piece: str) -> tuple[str, str]:
        """把一段流式片段拆成推理增量和内容增量。

        Args:
            piece: 引擎为一个 token 给出的增量文本。

        Returns:
            ``(reasoning_delta, content_delta)``。任一侧都可能为空；调用方只把
            非空的那个作为一条 SSE delta 发出。
        """
        self.raw += piece
        if self.past_think:
            return "", self._content()
        i = self.raw.find("</think>", self.think_pos)
        if i < 0:                            # 还在推理段内部
            out = self.raw[self.think_pos:]
            self.think_pos = len(self.raw)
            return out, ""
        reasoning = self.raw[self.think_pos:i]
        self.past_think = True
        self.think_pos = len(self.raw)
        self.emitted = i + len("</think>")
        return reasoning, self._content()

    def finish(self, has_tools: bool) -> tuple[str, str]:
        """在流末尾把扣住的尾巴放出来。

        扣住内容的目的是让半个 ``<tool_call>`` 永远到不了客户端；流结束时必须
        放行，否则每个回答的最后几个字符都会丢。有工具调用时这段尾巴*就是*
        标记文本（已经以结构化字段交付过），直接丢弃。

        Args:
            has_tools: 引擎的最终回包是否带了已解析的工具调用。

        Returns:
            ``(reasoning_tail, content_tail)``；任一侧都可能为空。
        """
        if not self.past_think:              # 推理段始终没有闭合
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
    """按 ``json.dumps(ensure_ascii=False)`` 的方式转义一个片段。

    必须逐字节一致：客户端拼起来的片段得和非流式路径返回的是同一个字符串，
    否则客户端在同一个调用上退回非流式 JSON 时会看到两份不同的 arguments。

    Args:
        text: 工具调用标记里的原始取值文本。

    Returns:
        转义后的片段，不含外层引号。
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
    """按 ``tool_call._coerce``（非流式路径）的规则保住取值的类型。

    两条路径的类型必须一致，否则同一个调用流式过来是 ``{"n": 3}``，不走流式
    却变成 ``{"n": "3"}``。

    Args:
        value: ``<parameter=...>`` 与 ``</parameter>`` 之间的原始文本。

    Returns:
        文本是 JSON 字面量（含嵌套对象和数组）时返回解析后的值，否则返回
        去空白后的字符串。看着像数字却解析不了的东西——比如 ``1.2.3``——
        保持字符串，而不是让整个调用失败。
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
    """把 Qwen 的 ``<tool_call>`` 标记边生成边转成 OpenAI 的 ``tool_calls`` 增量，
    形状与 vLLM 的流式工具解析器一致。

    引擎是逐 token 流出调用的，但如果只在末尾解析（本服务以前的做法），客户端
    要在整个调用结束之后才拿到一个很胖的 delta：长参数期间毫无进度，界面看起来
    像卡死。所以这里放的是 vLLM 每个 choice 各跑一份的同一台状态机：

    * 调用的第一个片段带 ``index``/``id``/``type``/``function.name``（``arguments``
      为空串），客户端因此能在参数存在之前就知道*要调哪个*工具；
    * ``function.arguments`` 分片到达——参数值是字符串时边生成边流，看起来像
      JSON 字面量时则整个发出（带类型：数字、true/false/null、嵌套 JSON），
      因为这个判断一旦做下就没法在半途反悔；
    * ``</tool_call>`` 用最后一个空的 ``arguments`` 片段收尾，下一个调用的
      ``index`` 加一。

    ``arguments`` 累加起来恰好等于 ``json.dumps(parsed_arguments)``——与非流式
    回复里携带的字符串完全相同——所以客户端两条路径都能用。

    缓冲区只追加，``pos`` 只向前越过已经定下来的文本，这才使得一个把标记切成
    两半的 token 仍然能正确解析：不完整的标记要等最后一个字符到齐才匹配。
    """

    def __init__(self, seed: str = "", max_calls: int | None = None,
                 marker: str = _CALL_OPEN) -> None:
        """为一次回复建一个解析器。

        Args:
            seed: 在任何生成的片段之前先解析的文本。用来注入
                ``tool_choice: required`` 产生的强制开头——它在 prompt 里，
                不在生成文本里。
            max_calls: 发满这么多个调用后就停，用于
                ``parallel_tool_calls: false``。``None`` 表示全都发。
            marker: 要找的工具调用开头标记。
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
        self.mode = ""            # ""、"stream"（字符串取值）或 "hold"（JSON 字面量）
        self.hold = ""            # "hold" 模式下扣住的取值文本
        self.ws = ""              # 流式取值尾部暂扣的空白
        self.value_started = False
        self.open_quote = False   # JSON 字符串的开引号还没发出去
        self.emitted_any = False

    def _delta(self, arguments: str, name: str | None = None) -> dict | None:
        """构造一条 ``tool_calls`` 增量；该被压掉时返回 ``None``。

        Args:
            arguments: 这条增量的 ``function.arguments`` 片段。
            name: 只在引入调用的那个片段上给函数名；后续片段一律 ``None``，
                客户端据此判断是否需要期待新的 ``id``/``type`` 对。

        Returns:
            ``{"tool_calls": [...]}`` 载荷；``parallel_tool_calls: false`` 已经
            满足后返回 ``None``——调用方据此对多余的调用什么都不发。
        """
        if self.max_calls is not None and self.index >= self.max_calls:
            return None                     # parallel_tool_calls: false，多余的直接丢
        fn: dict = {"arguments": arguments}
        if name is not None:
            fn["name"] = name
        tc: dict = {"index": self.index, "function": fn}
        if name is not None:                # 引入这个调用的片段
            tc["id"] = self.call_id
            tc["type"] = "function"
        self.emitted_any = True
        return {"tool_calls": [tc]}

    def _value_fragment(self, text: str) -> list[dict]:
        """吃进原始取值文本，把此刻已经定型的参数片段发出去。

        字符串值可以边到边流；JSON 字面量不行，因为客户端已经收到引号之后，一个
        被当成字符串读走的数字就没法再变回数字。所以第一个字符一次性定下模式，
        取值一路跟随：``hold`` 把文本攒着，到闭合标记时强制类型后整体 dump；
        ``stream`` 转义后直接转发。

        Args:
            text: 新到手的取值文本，不含闭合标记。

        Returns:
            零条或多条 ``tool_calls`` 增量。尾部空白暂扣不发，``"1 "`` 的值因此
            去掉补白，与非流式路径上 ``_coerce`` 的结果一致。
        """
        if not text:
            return []
        if not self.value_started:
            text = text.lstrip()            # ``_coerce`` 会去掉取值两端的空白
            if not text:
                return []
            self.value_started = True
            if text[0] in _JSON_LITERAL_HEAD:
                self.mode = "hold"
            else:
                self.mode = "stream"
                self.open_quote = True      # 字符串值以一个引号开头
        if self.mode == "hold":
            self.hold += text
            return []
        combined = self.ws + text
        body = combined.rstrip()            # 尾部空白暂时不发
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
        """发出结束当前参数取值的那个片段。

        Returns:
            零条或一条增量。``hold`` 模式先把值强制类型再整体 dump，参数流因此
            与 ``json.dumps`` 的结果完全一致；``stream`` 模式只补上收尾引号。
        """
        out: list[dict] = []
        if self.mode == "hold":
            d = self._delta(json.dumps(_coerce_value(self.hold), ensure_ascii=False))
        elif self.mode == "stream":
            d = self._delta('"')            # 收掉字符串的引号
        else:
            d = self._delta('""')           # 空值（或只有空白）
        if d:
            out.append(d)
        self.param = None
        self.mode = ""
        self.hold = self.ws = ""
        self.value_started = False
        self.open_quote = False
        return out

    def push(self, piece: str) -> list[dict]:
        """解析一段流式文本。

        Args:
            piece: 引擎为一个 token 给出的增量文本。

        Returns:
            这一段之后才定型的 ``tool_calls`` 增量；标记或扣住的 JSON 字面量
            还没收齐时为空。
        """
        self.buf += piece
        return self._drain(final=False)

    def finish(self) -> list[dict]:
        """在流末尾把被截断的调用收尾。

        Returns:
            剩下的增量，包括标记提前结束的调用的收尾片段。没有这一步，被
            ``max_tokens`` 截断的调用会让客户端收到一个没有结束的
            ``arguments`` 字符串。
        """
        return self._drain(final=True)

    def _drain(self, final: bool) -> list[dict]:
        """按缓冲区当前的进展推进状态机。

        Args:
            final: 流末尾时为 ``True``，此时直接收掉被截断的参数，不再等它的
                闭合标记。

        Returns:
            本趟产出的增量，按发出顺序排列。
        """
        out: list[dict] = []
        while True:
            if not self.in_call:
                i = self.buf.find(self.marker, self.pos)
                if i < 0:
                    # 开头标记没收完整之前什么都不做：缓冲区只追加，而这些
                    # 标记只按完整形式匹配，所以 ``pos`` 必须原地不动，否则
                    # 后面的片段会被跳过。
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
                    if final:                       # 调用被截断：就地收尾
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
                # 等标记收完整（见上）：不要推进 pos
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
                d = self._delta("")               # vLLM 收尾用的空片段
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
