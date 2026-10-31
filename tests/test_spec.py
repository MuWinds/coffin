"""投机解码必须精确，其草稿模型还得真能预测出下一个 token。

Module: ``coffin/tests``。它经 ``support.ModelTestCase`` 驱动 ``model.spec_step``
与 ``model.mtp``，也是唯一捕获 K+1 宽校验图的测试，因此必须独占进程，并让
模型以这个宽度加载。

Constraints:
    * 不变量是精确性而不是速度：贪心投机只允许改变墙钟时间，输出必须逐 token
      等于纯贪心。对不上就是 draft/verify 记账的正确性 bug，绝不能用「近似可
      接受」搪塞过去。
    * 校验图按 K+1 个 token 捕获，这是该步骤真正需要的宽度：每多校验一个
      token，24 个线性注意力层里每层都要多做一次完整的循环状态更新。捕获得
      比需要的宽会拖慢每一步；比需要的窄则在重放时直接失败。
    * 接受率衡量的是草稿模型预测的质量，不是时间阈值；加速比本身由 diag 下的
      探针测量。

移植自 ``diag/check_spec.py``（耗时与开销拆分的打印留在那边）。
"""

from __future__ import annotations

import functools
import unittest

import support

PROMPT = ("The capital of France is Paris. The capital of Germany is Berlin. "
          "The capital of Italy is Rome. Tell me about the history of")
N = 64


@functools.lru_cache(maxsize=1)
def _prompt_ids() -> list[int]:
    """把固定 prompt 只分词一次，让每个子测试都从同一串 id 出发。"""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(support.MODEL_DIR))
    return tok.encode(PROMPT)


@support.needs_model
class SpecDecodeTest(support.ModelTestCase):
    """贪心投机只该改速度；token 流必须一字不差。"""

    QUANT = True
    MAX_SEQ = 2048

    def setUp(self) -> None:
        """复用 prompt id，并保证任何投机步骤之前校验图已经捕获完成。"""
        super().setUp()
        self.ids = _prompt_ids()
        support.ensure_captured(self.model)

    def _plain(self, n: int) -> list[int]:
        """贪心基线：每步取 argmax 回灌，一次一个 token。"""
        model = self.model
        model.reset()
        logits = model.prefill(self.ids)
        out: list[int] = []
        for _ in range(n):
            tok = int(logits.argmax())
            out.append(tok)
            logits = model.decode_step(tok)
        return out

    def _spec(self, n: int, k: int) -> tuple[list[int], dict]:
        """在草稿深度 ``k`` 下跑 ``n`` 个 token 的投机循环。

        故意丢掉两次预热步骤：最初几次重放会摸到尚未填过的图内存，留下来会
        污染统计。
        """
        model = self.model
        model.capture_verify(k + 1)        # 只为真正要校验的 token 付代价
        model.reset()
        logits = model.prefill(self.ids)
        for _ in range(2):                 # 先把重放路径预热
            _, logits = model.spec_step(logits, k)
        model.reset()
        logits = model.prefill(self.ids)
        model.n_drafted = model.n_accepted = model.n_spec_steps = 0
        out: list[int] = []
        while len(out) < n:
            toks, logits = model.spec_step(logits, k)
            out.extend(toks)
        return out[:n], model.spec_stats

    def test_capture_builds_the_verify_and_draft_graphs(self):
        """``spec_step`` 要重放两张图，缺一次捕获必须在这里失败，而不是生成到一半才炸。"""
        self.assertIsNotNone(self.model._verify_graph)
        self.assertIsNotNone(self.model.mtp)
        self.assertIsNotNone(self.model.mtp._graph)

    def test_spec_matches_plain_greedy(self):
        """无论深度多少，被接受的草稿加回退都必须精确复现纯贪心的输出。"""
        plain = self._plain(N)
        for k in (1, 3, 7):
            with self.subTest(K=k):
                spec, stats = self._spec(N, k)
                self.assertEqual(spec, plain,
                                 f"K={k}: speculative output differs from greedy "
                                 f"({sum(a == b for a, b in zip(spec, plain))}/{N} tokens)")

    def test_drafter_acceptance_rate(self):
        _, stats = self._spec(N, 1)
        # K=1 实测约 73%；这条量的是草稿模型的预测质量，不是时间阈值
        # （加速比本身由 diag 下的探针测量）。
        self.assertGreater(stats["accept_rate"], 0.5,
                           f"K=1 acceptance {stats['accept_rate']:.1%}")
