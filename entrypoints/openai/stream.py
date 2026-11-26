"""推理段与工具调用的增量流解析器。

Module: ``coffin.serving``。这两个解析器把引擎的原始 token 流转成 OpenAI 兼容
的 delta，全程不碰模型，也不碰 HTTP 传输层。

Constraints:
    * 增量解析：每个 ``push`` 只扫新追加的文本，整体代价是 O(tokens)，绝不
      重扫整段回答。
    * 流式与非流式两条路径产出的 ``arguments`` 必须是同一个字符串，否则
      客户端在同一次调用的两种模式之间切换会看到不同的参数。
    * 工具调用只从*答案*段解析，推理段不算。模型在思考里写「我应该调用
      ``get_weather``」是最常见的形状，把它当成真调用会让客户端执行一个模型
      最终并没有选择的工具。
    * 标记由 :mod:`utils.chat_protocol` 提供，默认值只是后备：引擎在流式
      ``start`` 行里带下 checkpoint 模板用的那一份。
"""

from __future__ import annotations

import json
import uuid

from utils import chat_protocol

#: 兼容旧引用：工具调用开标记的后备值。
TOOL_OPEN = chat_protocol.DEFAULT_MARKERS["tool_open"]

#: 会被当成「可能长出 JSON 字面量」的首字符；没有 schema 时靠它决定取值要不要
#: 扣到最后一起发。
_JSON_LITERAL_HEAD = "-0123456789tfn{["
_SHORT_ESCAPE = {"\n": "\\n", "\r": "\\r", "\t": "\\t", "\b": "\\b", "\f": "\\f"}


class AnswerFilter:
    """把原始流切成（推理, 内容, 工具段）三种增量。

    这里处理三件事，对思考型模型都是承重的：

    * 推理段——思考开在 *prompt* 里时会先出现 ``<think>``，生成的文本因此从
      推理段内部开始。推理边产生边以 ``reasoning_content`` 增量发出，
      ``content`` 要等闭合标签之后才开始。把它丢掉或者压到最后再补，正是
      agent 界面思考面板空白的原因。
    * 模型自己写的 ``<think>``——裸 prompt 或老模板下，开标签不在 prompt 里而是
      由模型生成，还可能出现在一小段答案之后。漏掉这条，整段内心独白会带着标签
      落进 ``content``。
    * ``<tool_call>`` 标记——一旦出现，内容输出停止，调用改以结构化字段交付；
      不完整的标记会被扣住以免泄漏，``finish`` 在流末尾放行这段尾巴。

    第三条返回值是给工具解析器的：它是「答案段」的原文，含调用标记、不含推理段。
    两个解析器跑在同一段流上，但必须看到不同的切片，否则思考里的假调用会被执行。

    刻意做成增量：每个 ``push`` 只扫新追加的文本，所以过滤器是 O(tokens)
    而不是每次重扫整段回答。
    """

    def __init__(self, prompt_opens_think: bool, markers=None) -> None:
        """为一次回复建一个过滤器。

        Args:
            prompt_opens_think: 渲染后的 prompt 是否已经在未闭合的 ``<think>``
                内部结束，这个信息由引擎在 ``start`` 行里给出。为否时，整段
                回答从第一个 token 起都是答案段。
            markers: 标记表；``None`` 用这一族的后备值。
        """
        self.raw = ""
        self.markers = chat_protocol.markers_from(markers)
        self.in_reasoning = bool(prompt_opens_think)
        self.r_pos = 0               # 已发出的推理文本长度
        self.c_pos = 0               # 已发出的内容文本长度
        self.t_pos = 0               # 已交给工具解析器的文本长度
        self.done_tools = False

    def _reasoning_until_close(self) -> str:
        """推理段的下一个增量，在闭合标记处停下。

        Returns:
            现在可以安全发出的推理文本。闭合标记还不完整时先扣住那一截尾巴，
            否则半个标记会漏进 ``reasoning_content``。
        """
        close = self.markers["think_close"]
        i = self.raw.find(close, self.r_pos)
        if i >= 0:
            out = self.raw[self.r_pos:i]
            self.r_pos = i + len(close)
            self.in_reasoning = False
            # 答案段从闭合标记之后开始；两个游标一起挪，工具解析器才不会拿到
            # 刚才那段推理。
            self.c_pos = self.r_pos
            self.t_pos = self.r_pos
            return out
        safe = len(self.raw) - (len(close) - 1)
        if safe <= self.r_pos:
            return ""
        out = self.raw[self.r_pos:safe]
        self.r_pos = safe
        return out

    def _content(self) -> str:
        """下一个内容增量，同时扣住可能是半个标记的尾巴。

        Returns:
            现在可以安全发出的内容文本。尾巴仍可能长成完整标记时返回空——先
            发出去、后来才发现，就会把标记文本漏进回答里。
        """
        if self.done_tools or self.in_reasoning:
            return ""
        mk = self.markers
        open_tags = (mk["tool_open"], mk["think_open"])
        while True:
            body = self.raw[self.c_pos:]
            found = []
            for tag in open_tags:
                k = body.find(tag)
                if k >= 0:
                    found.append((k, tag))
            if found:
                k, tag = min(found, key=lambda c: c[0])
                if tag == mk["tool_open"]:
                    self.done_tools = True
                    out = body[:k]
                    self.c_pos += k
                    return out
                # 模型在答案中途自己打开了思考段：先交出它前面的答案，再切回推理。
                out = body[:k]
                self.c_pos += k + len(tag)
                self.r_pos = self.c_pos
                self.t_pos = self.c_pos
                self.in_reasoning = True
                return out
            keep = max(len(t) for t in open_tags) - 1     # 半个标记绝不发出
            if len(body) <= keep:
                return ""
            out = body[:len(body) - keep]
            self.c_pos += len(out)
            return out

    def push(self, piece: str) -> tuple[str, str, str]:
        """把一段流式片段拆成推理、内容和工具段三种增量。

        Args:
            piece: 引擎为一个 token 给出的增量文本。

        Returns:
            ``(reasoning_delta, content_delta, tool_delta)``。任一侧都可能为空；
            调用方只把非空的推理/内容作为 SSE delta 发出，``tool_delta`` 交给
            工具解析器，永远不进 ``content``。
        """
        self.raw += piece
        reasoning = self._reasoning_until_close() if self.in_reasoning else ""
        content = self._content()
        if self.in_reasoning:
            # 推理段还没结束：答案段一个字都还没开始，工具解析器也不该看到。
            tool = ""
        else:
            tool = self.raw[self.t_pos:]
            self.t_pos = len(self.raw)
        return reasoning, content, tool

    def finish(self, has_tools: bool) -> tuple[str, str, str]:
        """在流末尾把扣住的尾巴放出来。

        扣住内容的目的是让半个 ``<tool_call>`` 永远到不了客户端；流结束时必须
        放行，否则每个回答的最后几个字符都会丢。有工具调用时这段尾巴*就是*
        标记文本（已经以结构化字段交付过），直接丢弃。

        Args:
            has_tools: 引擎的最终回包是否带了已解析的工具调用。

        Returns:
            ``(reasoning_tail, content_tail, tool_tail)``；任一侧都可能为空。
        """
        reasoning = ""
        content = ""
        in_think = self.in_reasoning
        if in_think:                         # 推理段始终没有闭合
            reasoning = self.raw[self.r_pos:]
            self.r_pos = len(self.raw)
            self.in_reasoning = False
        else:
            body = self.raw[self.c_pos:]
            if has_tools or self.done_tools:
                # 调用标记之后是调用正文，已经以结构化字段交付过。被 truncation
                # 结束的调用也一样：那截正文绝不能回头当答案发出去。
                k = body.find(self.markers["tool_open"])
                body = body[:k] if k >= 0 else ""
            content = body
            self.c_pos += len(body)
        # 推理段没闭合时，那段文本全是推理，绝不能流到工具解析器里。
        tool = "" if in_think else self.raw[self.t_pos:]
        self.t_pos = len(self.raw)
        return reasoning, content, tool


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


class ToolStream:
    """把 Qwen 的 ``<tool_call>`` 标记边生成边转成 OpenAI 的 ``tool_calls`` 增量，
    形状与 vLLM 的流式工具解析器一致。

    引擎是逐 token 流出调用的，但如果只在末尾解析（本服务以前的做法），客户端
    要在整个调用结束之后才拿到一个很胖的 delta：长参数期间毫无进度，界面看起来
    像卡死。所以这里放的是 vLLM 每个 choice 各跑一份的同一台状态机：

    * 调用的第一个片段带 ``index``/``id``/``type``/``function.name``（``arguments``
      为空串），客户端因此能在参数存在之前就知道*要调哪个*工具；
    * ``function.arguments`` 分片到达——参数值是字符串时边生成边流；schema 声明
      了别的类型时整个取值扣到闭合标记再一次性发出（带类型：数字、true/false/null、
      嵌套 JSON），因为一个已经按字符串发出去的数字没法在半途反悔；
    * ``</tool_call>`` 用最后一个空的 ``arguments`` 片段收尾，下一个调用的
      ``index`` 加一。

    ``arguments`` 累加起来恰好等于 ``json.dumps(parsed_arguments)``——与非流式
    回复里携带的字符串完全相同——所以客户端两条路径都能用。

    缓冲区只追加，``pos`` 只向前越过已经定下来的文本，这才使得一个把标记切成
    两半的 token 仍然能正确解析：不完整的标记要等最后一个字符到齐才匹配。
    """

    def __init__(self, seed: str = "", max_calls: int | None = None,
                 markers=None, specs: dict[str, dict[str, str]] | None = None) -> None:
        """为一次回复建一个解析器。

        Args:
            seed: 在任何生成的片段之前先解析的文本。用来注入
                ``tool_choice: required`` 产生的强制开头——它在 prompt 里，
                不在生成文本里。
            max_calls: 发满这么多个调用后就停，用于
                ``parallel_tool_calls: false``。``None`` 表示全都发。
            markers: 标记表；``None`` 用这一族的后备值。
            specs: 工具 schema 的 ``函数名 -> (参数名 -> 类型)`` 表。空表表示请求
                没带 schema，取值按字面量猜类型（老路径）。
        """
        self.buf = seed or ""
        self.pos = 0
        self.markers = chat_protocol.markers_from(markers)
        self.specs = specs or {}
        self.max_calls = max_calls
        self.index = -1
        self.call_id = ""
        self.function = ""
        self.in_call = False
        self.name_sent = False
        self.first_param = True
        self.param = None
        self.param_type = ""
        self.mode = ""            # ""、"stream"（字符串取值）或 "hold"（其余）
        self.hold = ""            # "hold" 模式下扣住的取值文本
        self.ws = ""              # 流式取值尾部暂扣的空白
        self.value_started = False
        self.open_quote = False   # JSON 字符串的开引号还没发出去
        # ``arguments`` 那个 JSON 对象的左花括号已经发出去、右花括号还没发。
        # 正常流在函数闭合时补右括号；被截断的调用靠 ``finish`` 补，否则客户端
        # 拼出来的 ``arguments`` 不是合法 JSON。
        self.args_open = False
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

        字符串值可以边到边流；非字符串不行，因为客户端已经收到引号之后，一个
        被当成字符串读走的数字就没法再变回数字。模式由 schema 声明的类型决定，
        没有 schema 时才退回按第一个字符猜。``hold`` 把文本攒着，到闭合标记时
        强制类型后整体 dump；``stream`` 转义后直接转发。

        Args:
            text: 新到手的取值文本，不含闭合标记。

        Returns:
            零条或多条 ``tool_calls`` 增量。尾部空白暂扣不发，``"1 "`` 的值因此
            去掉补白，与非流式路径上 ``convert_param`` 的结果一致。
        """
        if not text:
            return []
        if not self.value_started:
            text = text.lstrip()            # 取值两端的空白是格式的一部分
            if not text:
                return []
            self.value_started = True
            if self.param_type:
                # 声明了类型：只有字符串族能边到边流。
                self.mode = ("stream" if chat_protocol.is_string_type(self.param_type)
                             else "hold")
            else:
                self.mode = "hold" if text[0] in _JSON_LITERAL_HEAD else "stream"
            # 流式取值的第一片要自带 JSON 的开引号；卡在「等第一个字符」上的
            # 客户端靠它知道参数值开始了。
            self.open_quote = self.mode == "stream"
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
            零条或一条增量。``hold`` 模式先把值按 schema 类型转好再整体 dump，
            参数流因此与 ``json.dumps`` 的结果完全一致；``stream`` 模式只补上
            收尾引号。
        """
        out: list[dict] = []
        if self.mode == "hold":
            value = chat_protocol.convert_param(self.hold, self.param_type)
            d = self._delta(json.dumps(value, ensure_ascii=False))
        elif self.mode == "stream":
            d = self._delta('"')            # 收掉字符串的引号
        else:
            d = self._delta('""')           # 空值（或只有空白）
        if d:
            out.append(d)
        self.param = None
        self.param_type = ""
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
            这一段之后才定型的 ``tool_calls`` 增量；标记或扣住的类型还没收齐时
            为空。
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

    def _marker_name(self, pos: int, tag: str) -> tuple[int, str] | None:
        """一个 ``<name=...>`` 开标签的名字与右尖括号下标。

        名字里不允许有换行：模板把名字写在同一行，换行说明这一截是正文而不是
        新标记。

        Args:
            pos: 开标签的起始下标。
            tag: 开标签本身。

        Returns:
            ``(名字右尖括号下标, 名字)``；名字还没写完时为 ``None``。
        """
        start = pos + len(tag)
        head_end = self.buf.find(">", start)
        newline = self.buf.find("\n", start)
        if head_end < 0 or (0 <= newline < head_end):
            return None
        return head_end, self.buf[start:head_end].strip()

    def _next_marker(self) -> tuple[int, str, str] | None:
        """缓冲区里最靠前的下一个结构标记。

        Returns:
            ``(下标, 种类, 名字)``；种类是 ``fn`` / ``param`` / ``fn_close`` /
            ``call_close``，闭合标记的名字为空。没有任何标记时为 ``None``。开
            标记的名字还没写完时也返回 ``None``：那一截要等下一个 token。
        """
        best: tuple[int, str, str] | None = None
        for kind, key in (("fn", "function_open"), ("param", "param_open"),
                          ("fn_close", "function_close"),
                          ("call_close", "tool_close")):
            tag = self.markers[key]
            i = self.buf.find(tag, self.pos)
            if i < 0:
                continue
            name = ""
            if kind in ("fn", "param"):
                parsed = self._marker_name(i, tag)
                if parsed is None:
                    continue                # 名字没写完，等它到齐
                name = parsed[1]
            if best is None or i < best[0]:
                best = (i, kind, name)
        return best

    def _drain(self, final: bool) -> list[dict]:
        """按缓冲区当前的进展推进状态机。

        Args:
            final: 流末尾时为 ``True``，此时直接收掉被截断的参数，不再等它的
                闭合标记。

        Returns:
            本趟产出的增量，按发出顺序排列。
        """
        mk = self.markers
        out: list[dict] = []
        while True:
            if not self.in_call:
                i = self.buf.find(mk["tool_open"], self.pos)
                if i < 0:
                    # 开头标记没收完整之前什么都不做：缓冲区只追加，而这些
                    # 标记只按完整形式匹配，所以 ``pos`` 必须原地不动，否则
                    # 后面的片段会被跳过。
                    break
                self.pos = i + len(mk["tool_open"])
                self.in_call = True
                self.index += 1
                self.call_id = "call_" + uuid.uuid4().hex[:24]
                self.function = ""
                self.name_sent = False
                self.first_param = True
                self.mode = ""
                self.hold = self.ws = ""
                self.value_started = False
                self.open_quote = False
                self.args_open = False
                continue

            if self.param is not None:
                end = self.buf.find(mk["param_close"], self.pos)
                if end < 0:
                    if final:                       # 调用被截断：就地收尾
                        out += self._value_fragment(self.buf[self.pos:])
                        self.pos = len(self.buf)
                        out += self._close_param()
                        break
                    safe = len(self.buf) - (len(mk["param_close"]) - 1)
                    if safe > self.pos:
                        out += self._value_fragment(self.buf[self.pos:safe])
                        self.pos = safe
                    break
                out += self._value_fragment(self.buf[self.pos:end])
                self.pos = end + len(mk["param_close"])
                out += self._close_param()
                continue

            nxt = self._next_marker()
            if nxt is None:
                break                           # 等标记收完整（见上）
            pos, kind, name = nxt
            if kind == "fn":
                self.function = name
                d = self._delta("", name) if not self.name_sent else None
                self.name_sent = True
                if d:
                    out.append(d)
                self.pos = self._marker_name(pos, mk["function_open"])[0] + 1
                continue
            if kind == "param":
                prefix = ("{" if self.first_param else ", ")
                d = self._delta(prefix + json.dumps(name, ensure_ascii=False) + ": ")
                if d:
                    out.append(d)
                self.first_param = False
                self.args_open = True
                self.param = name
                self.param_type = chat_protocol.param_type(self.specs, self.function, name)
                self.pos = self._marker_name(pos, mk["param_open"])[0] + 1
                continue
            if kind == "call_close":
                # 正常流已经在函数闭合时补过右括号；缺函数闭合标记的畸形流
                # 在这里补，免得客户端拿到不完整的 JSON。
                d = self._delta("}" if self.args_open else "")
                if d:
                    out.append(d)
                self.args_open = False
                self.pos = pos + len(mk["tool_close"])
                self.in_call = False
                self.param = None
                continue
            # 函数闭合：补上 JSON 的右花括号
            d = self._delta("}" if self.args_open else "{}")
            if d:
                out.append(d)
            self.args_open = False
            self.pos = pos + len(mk["function_close"])
        # 被 token 预算截断的调用：右括号不能缺，否则累积出来的 arguments 解析
        # 不动，而调用本身已经完整（参数值可能只缺尾部）。
        if final and self.in_call and self.args_open:
            d = self._delta("}")
            if d:
                out.append(d)
            self.args_open = False
        return out
