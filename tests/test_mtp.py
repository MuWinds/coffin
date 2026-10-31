"""MTP 头充当投机草稿模型：它的接受率是否值回代价。

Module: ``coffin/tests``。经 ``support.ModelTestCase`` 摸进 ``model.mtp`` 与层栈；
只测草稿模型本身，不碰 ``test_spec.py`` 覆盖的校验路径。

Constraints:
    * 草稿模型吃的是主模型的真实隐含状态，不是它自己的输出：它就是在这些状态上
      训练的，喂自己的输出只会量到漂移，而不是预测质量。
    * 草稿模型自己的 KV cache 先用真实 token 填满再清空；留着不清会把前缀泄漏到
      下一个测试。
    * 更深的草稿不可能比更浅的更容易被接受，所以单调性检查写成断言，而不是和某
      个实测数字比大小。
    * k=1 处的 0.5 门槛是业务阈值：低于大约一半时，投机路径每步的代价超过它省
      下的那一步，草稿模型再准也不值得保留。
"""

from __future__ import annotations

import functools
import unittest

import support

import torch

K = 5
TEXT = ("The capital of France is Paris. The capital of Germany is Berlin. "
        "The capital of Italy is Rome. The capital of Spain is Madrid. "
        "The capital of Portugal is Lisbon. The capital of Greece is Athens. "
        "Water boils at 100 degrees Celsius at sea level. The Moon orbits the "
        "Earth roughly once every 27 days. Light travels at about 300,000 km/s.")


@functools.lru_cache(maxsize=1)
def _prompt_ids() -> list[int]:
    """把固定 prompt 只分词一次；接受率扫描要和这串 id 逐位对齐。"""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(support.MODEL_DIR))
    return tok.encode(TEXT)


@support.needs_model
class MtpDrafterTest(support.ModelTestCase):
    """草稿模型的预测得足够准，投机才比不投机划算。"""

    QUANT = True
    MAX_SEQ = 1024

    def setUp(self) -> None:
        """预取 prompt id，并留住草稿模型需要的 RoPE 表与 mask 上下文。"""
        super().setUp()
        self.ids = _prompt_ids()
        self.ctx = (self.model.cos_tbl, self.model.sin_tbl, self.model.mask_idx)

    def _main_hidden(self, ids: list[int]) -> torch.Tensor:
        """每个 prompt token 之后的主模型隐含状态，也就是草稿模型的对照真值。"""
        model = self.model
        toks = torch.tensor(ids, device="cuda")
        x = model.embed.lookup(toks)
        positions = torch.arange(len(ids), device="cuda")
        for layer in model.layers:
            x = layer.prefill(x, positions, self.ctx)
        return x

    def test_chained_acceptance_by_depth(self):
        """各深度的接受率必须越过 k=1 门槛且保持单调，否则草稿链就是白做的功。"""
        model = self.model
        ids = self.ids
        assert len(ids) > K + 4, "prompt too short for the acceptance sweep"
        hidden = self._main_hidden(ids)

        # 用真实 token 把草稿模型自己的 KV cache 按整个 prompt 填满。
        model.mtp.reset()
        for j in range(len(ids) - 1):
            model.mtp.pos.fill_(j)
            model.mtp.step(hidden[j], torch.tensor([ids[j + 1]], device="cuda"), self.ctx)

        accepted = [0] * (K + 1)
        tested = 0
        for i in range(len(ids) - 2 - K, 0, -1):
            hits = 0
            h = hidden[i]
            tok_next = ids[i + 1]
            for k in range(K):
                model.mtp.pos.fill_(i + k)
                h, logits = model.mtp.step(h, torch.tensor([tok_next], device="cuda"), self.ctx)
                pred = int(logits.argmax())
                if pred != ids[i + 2 + k]:
                    break
                hits += 1
                tok_next = pred
            for k in range(1, hits + 1):
                accepted[k] += 1
            tested += 1
        model.mtp.reset()

        self.assertGreater(tested, 10, "not enough positions to measure acceptance")
        rates = [accepted[k] / tested for k in range(1, K + 1)]
        self.assertGreater(rates[0], 0.5, f"k=1 acceptance {rates[0]:.1%}")
        self.assertGreater(rates[1], 0.2, f"k=2 acceptance {rates[1]:.1%}")
        for shallow, deep in zip(rates, rates[1:]):
            self.assertGreaterEqual(shallow + 1e-9, deep,
                                    f"acceptance not monotone: {rates}")
