"""对话线格式的单元测试：不依赖 torch，也不需要 GPU 或 checkpoint。

Module: ``coffin/tests``。这里锁的是 qwen3.5 与 occamy 1.0 共用的那一份规则：
推理段怎么切、标记怎么来、参数类型跟谁走。三者都曾经靠各模块里的字面量拼出来，
所以每条都单独断言，而不是只通过 OpenAI 报文间接覆盖。

Constraints:
    * 只用标准库：``utils.chat_protocol`` 是依赖图的叶子，这里也不往上走。
    * 标记不写字面量，一律从 ``chat_protocol.DEFAULT_MARKERS`` 或拼接取——仓库
      里 XML 风格的标签解析会把完整的闭合标签当成自己的结束符。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import support  # noqa: F401   # 副作用导入：把 coffin/ 加进 sys.path

from utils import chat_protocol
from utils.tool_call import parse_tool_calls

#: 参数闭合标记。拼出来，避免源码里出现完整的一对开闭标签。
PC = "</" + "parameter>"
MK = chat_protocol.DEFAULT_MARKERS


class SplitThinkingTest(unittest.TestCase):
    """推理段与答案的切分：两种模板形态、被截断的推理、以及没有推理。"""

    def test_new_style_prompt_opens_the_block(self):
        """模板把开标记发在 prompt 里，生成文本从推理段内部开始，只剩闭合标记。"""
        thinking, answer = chat_protocol.split_thinking(
            "先算一下\n" + MK["think_close"] + "答案是 4。")
        self.assertEqual(thinking, "先算一下")
        self.assertEqual(answer, "答案是 4。")

    def test_old_style_model_writes_both_tags(self):
        """裸 prompt 或老模板：模型自己写出整对标签，标签前的文字仍属于答案。"""
        text = ("开场 " + MK["think_open"] + "想一想" + MK["think_close"] + " 收尾")
        thinking, answer = chat_protocol.split_thinking(text)
        self.assertEqual(thinking, "想一想")
        self.assertEqual(answer, "开场  收尾")
        self.assertNotIn(MK["think_open"], answer)

    def test_truncated_reasoning_has_no_answer(self):
        """推理段没闭合（撞上 token 预算）时，标签之后全是推理，不能当答案。"""
        thinking, answer = chat_protocol.split_thinking(
            MK["think_open"] + "才想到一半")
        self.assertEqual(thinking, "才想到一半")
        self.assertEqual(answer, "")

    def test_no_reasoning_at_all(self):
        """没有标记时整段都是答案——这是关掉思考的普通回复。"""
        self.assertEqual(chat_protocol.split_thinking("直接回答"), ("", "直接回答"))
        self.assertEqual(chat_protocol.split_thinking(""), ("", ""))


class OpensReasoningTest(unittest.TestCase):
    """prompt 是否停在未闭合的推理段内部。"""

    def test_template_ends_with_the_open_tag(self):
        prompt = "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n<think>\n"
        self.assertTrue(chat_protocol.opens_reasoning(prompt))

    def test_plain_assistant_header_does_not_open_it(self):
        prompt = "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n"
        self.assertFalse(chat_protocol.opens_reasoning(prompt))

    def test_closed_block_in_history_is_not_an_open_block(self):
        prompt = ("<|im_start|>assistant\n<think>旧的推理" + MK["think_close"]
                  + "\n旧答案<|im_end|>\n<|im_start|>user\n再说一次<|im_end|>\n"
                    "<|im_start|>assistant\n")
        self.assertFalse(chat_protocol.opens_reasoning(prompt))


class ParamTypesTest(unittest.TestCase):
    """工具 schema -> 参数类型表。"""

    def test_nested_properties(self):
        tool = {"type": "function", "function": {
            "name": "lookup",
            "parameters": {"type": "object", "properties": {
                "zip": {"type": "string"}, "n": {"type": "integer"}}},
        }}
        specs = chat_protocol.param_types([tool])
        self.assertEqual(specs, {"lookup": {"zip": "string", "n": "integer"}})

    def test_type_array_collapses_to_the_non_null_half(self):
        """``["string", "null"]`` 这种可空写法要落到 ``string``，不是整串。"""
        tool = {"type": "function", "function": {
            "name": "f",
            "parameters": {"properties": {"v": {"type": ["string", "null"]}}},
        }}
        specs = chat_protocol.param_types([tool])
        self.assertEqual(specs["f"]["v"], "string")

    def test_flat_and_legacy_shapes(self):
        """参数直接平铺在 ``parameters`` 下、以及已弃用的顶层 ``name`` 都要认。"""
        flat = chat_protocol.param_types([{"function": {
            "name": "f", "parameters": {"city": {"type": "string"}}}}])
        legacy = chat_protocol.param_types([{
            "name": "g", "parameters": {"properties": {"n": {"type": "integer"}}}}])
        self.assertEqual(flat["f"]["city"], "string")
        self.assertEqual(legacy["g"]["n"], "integer")

    def test_missing_schema_is_an_empty_table(self):
        self.assertEqual(chat_protocol.param_types(None), {})
        self.assertEqual(chat_protocol.param_types([]), {})


class ConvertParamTest(unittest.TestCase):
    """参数取值按 schema 声明的类型定型。"""

    def test_string_params_stay_text(self):
        """``0755`` 是邮编、``00123`` 是工号：猜成数字就把工具的参数改坏了。"""
        self.assertEqual(chat_protocol.convert_param("0755", "string"), "0755")
        self.assertEqual(chat_protocol.convert_param("00123", ""), "00123")
        self.assertEqual(chat_protocol.convert_param("null", "string"), "null")

    def test_numbers_and_booleans(self):
        self.assertEqual(chat_protocol.convert_param("12", "integer"), 12)
        self.assertEqual(chat_protocol.convert_param("2.5", "number"), 2.5)
        # 整数写法的 number 收敛回 int：严格校验的 schema 会拒收 3.0
        self.assertEqual(chat_protocol.convert_param("3.0", "number"), 3)
        self.assertIs(chat_protocol.convert_param("true", "boolean"), True)
        self.assertIs(chat_protocol.convert_param("false", "boolean"), False)
        # ``yes`` 不该被静默读成 False：留给工具自己判
        self.assertEqual(chat_protocol.convert_param("yes", "boolean"), "yes")

    def test_null_and_structures(self):
        self.assertIsNone(chat_protocol.convert_param("null", "integer"))
        self.assertEqual(chat_protocol.convert_param('["a", "b"]', "array"), ["a", "b"])
        self.assertEqual(chat_protocol.convert_param('{"k": 1}', "object"), {"k": 1})
        # 解析不动时退回原文，绝不把一次工具调用变成异常
        self.assertEqual(chat_protocol.convert_param("[坏", "array"), "[坏")

    def test_unconvertible_numbers_fall_back_to_text(self):
        self.assertEqual(chat_protocol.convert_param("十二", "integer"), "十二")


class ParseToolCallsTest(unittest.TestCase):
    """解析器只按 schema 定型，不替模型猜类型。"""

    def test_declared_types_are_applied(self):
        tools = [{"type": "function", "function": {
            "name": "lookup", "parameters": {"type": "object", "properties": {
                "zip": {"type": "string"}, "n": {"type": "integer"}}}}}]
        text = ("<tool_call>\n<function=lookup>\n<parameter=zip>\n0755\n" + PC
                + "\n<parameter=n>\n12\n" + PC + "\n\n</tool_call>")
        call = parse_tool_calls(text, tools)[0]
        self.assertEqual(call.name, "lookup")
        self.assertEqual(call.arguments, {"zip": "0755", "n": 12})

    def test_without_schema_the_literal_heuristic_still_applies(self):
        """没有 schema 的本地 agent 循环保持老行为：明显是数字的取值按数字给。"""
        text = ("<tool_call>\n<function=multiply>\n<parameter=a>\n12\n" + PC
                + "\n\n</tool_call>")
        self.assertEqual(parse_tool_calls(text)[0].arguments, {"a": 12})


class EnsureTemplateTest(unittest.TestCase):
    """模板兜底：occamy 1.0 把模板放在 ``chat_template.jinja`` 单文件里。"""

    class _Tok:
        """只有一个 ``chat_template`` 属性的最小分词器替身。"""

        def __init__(self, template=None):
            self.chat_template = template

    def test_sidecar_file_is_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chat_template.jinja"
            path.write_text("{{ messages }}", encoding="utf-8")
            tok = self._Tok()
            self.assertEqual(chat_protocol.ensure_template(tok, tmp), "{{ messages }}")
            self.assertEqual(tok.chat_template, "{{ messages }}")

    def test_existing_template_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "chat_template.jinja").write_text("sidecar", encoding="utf-8")
            tok = self._Tok("<think>")
            self.assertEqual(chat_protocol.ensure_template(tok, tmp), "<think>")

    def test_no_template_anywhere(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(chat_protocol.ensure_template(self._Tok(), tmp))


class MarkersTest(unittest.TestCase):
    """标记表：引擎下发的配置可以覆盖，但不该把表弄残。"""

    def test_partial_payload_is_completed(self):
        markers = chat_protocol.markers_from({"tool_open": "<call>"})
        self.assertEqual(markers["tool_open"], "<call>")
        self.assertEqual(markers["think_open"], MK["think_open"])

    def test_junk_values_are_ignored(self):
        markers = chat_protocol.markers_from({"tool_open": None, "param_close": 7})
        self.assertEqual(markers["tool_open"], MK["tool_open"])
        self.assertEqual(markers["param_close"], MK["param_close"])

    def test_template_presence_flags_the_missing_pieces(self):
        present = chat_protocol.template_presence(
            MK["think_open"] + MK["think_close"] + MK["tool_open"])
        self.assertTrue(present["think_open"])
        self.assertTrue(present["tool_open"])
        self.assertFalse(present["tool_close"])
        self.assertFalse(present["param_open"])


if __name__ == "__main__":
    unittest.main()
