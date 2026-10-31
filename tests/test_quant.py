"""权重格式的精度账：换成 fp16 存储或压成 int8，模型的答案会不会被挪动。

Module: ``coffin/tests``。经 ``support.ModelFixture`` 先后加载两个量化方式不同的
模型，所以必须单独跑或放到最后：两张 4B 模型的权重加 KV 同时驻留，一张卡放不下。

Constraints:
    * 一律 teacher forcing，绝不放任自由生成：任何扰动迟早会翻转一个 argmax，
      之后两条续写给出的是两份不同（但都合法）的文本，所以自由生成的逐 token
      相等不是正确性该有的性质。
    * 两个模型必须吃同一段真实前缀，top-1 一致率和真实下一个 token 的困惑度
      才可比。
    * logprob 只要不是有限值就算失败，哪怕一致率看着很好：fp16 的指数位比 bf16
      窄，这个格式真正的风险是范围溢出，一个 inf 会污染困惑度而不是把自己报出来。

移植自 ``diag/check_int8.py``、``diag/check_int8_ppl.py`` 和
``diag/check_f16_ppl.py``（只保留精度那一半，速度那一半留在探针里）。实测数字：

* fp16 对 bf16 权重：top-1 一致率 98.5%（130/132），ppl 5.108 对 5.113；
* int8（W8A16）对 fp16：一致率 96.2%（127/132），ppl +0.29%。
"""

from __future__ import annotations

import functools
import math
import unittest

import support

import torch

TEXT = (
    "The history of Paris spans more than two thousand years. It began as a "
    "settlement on the banks of the Seine, known to the Romans as Lutetia. "
    "Over the centuries it grew into the political, cultural and economic "
    "centre of France, surviving revolutions, occupations and rebuilding. "
    "Its medieval streets were gradually replaced by broad boulevards, and the "
    "city became a model for urban planning across Europe. Today it remains "
    "one of the most visited cities in the world, and its museums, universities "
    "and libraries hold collections that reach back to antiquity. The Seine "
    "still divides the city into a left bank and a right bank, and the islands "
    "in its channel are where the settlement first took root."
)


@functools.lru_cache(maxsize=1)
def _text_ids() -> list[int]:
    """把固定段落只分词一次；逐用例重新分词可能落在不同的 merge 上。"""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(support.MODEL_DIR))
    return tok.encode(TEXT)


def _teacher_forced(ids: list[int], *, quant: bool, weight_dtype=None):
    """逐位置给出真实下一个 token 的 logprob 与模型自己的 top-1 选择。

    **这个写法只测得到权重格式，测不到 W8A8。** 它先 ``prefill(ids[:4])`` 再逐 token
    ``decode_step``，而引擎在 ``M < 64`` 时把投影走 GEMV——所以激活量化那条路径一次都
    不会被调用。用它对照 W8A8 与 W8A16 会得到逐位相同的结果，读起来像「W8A8 精度无损」，
    实际是两条臂跑了同一段代码。

    这条陷阱在本仓库真实发生过：``probe_w8a8_accuracy.py`` 第一版就这么写，靠加路径
    计数器自证（W8A16 臂 ``w8a8_calls=0``、W8A8 臂 9576）才发现。要量 W8A8 必须让每个
    计分位置都走 ``M >= 64`` 的预填充，也就是逐前缀预填充而不是增量 decode。

    Args:
        ids: 两个模型共同吃进的真实 token id。
        quant: 以 INT8 权重加载模型。这选的是**权重存储形式**；``W8A8`` 是另一件事。
        weight_dtype: 覆盖存储 dtype；``None`` 沿用 checkpoint 自带的。

    Returns:
        ``(logprobs, top1)``。``logprobs[i]`` 是真正下一个 token 的 log-softmax，
        ``top1[i]`` 是该位置上模型自己的 argmax；两者配合即可把「存得不一样」和
        「预测得不一样」分开。
    """
    model = support.ModelFixture.get(quant=quant, max_seq=1024,
                                     weight_dtype=weight_dtype)
    support.ensure_captured(model)
    model.reset()
    logits = model.prefill(ids[:4])
    logprobs: list[float] = []
    top1: list[int] = []
    for i in range(4, len(ids) - 1):
        lf = logits.float()
        lp = torch.log_softmax(lf, dim=-1)
        logprobs.append(float(lp[ids[i]]))
        top1.append(int(lf.argmax()))
        logits = model.decode_step(ids[i])
    return torch.tensor(logprobs), torch.tensor(top1)


def _report(lp: torch.Tensor) -> tuple[float, float]:
    """打分位置上返回 ``(平均负对数似然, 困惑度)``。"""
    nll = -lp.mean().item()
    return nll, math.exp(nll)


@support.needs_model
class WeightFormatQualityTest(unittest.TestCase):
    """fp16 与 int8 的存储选择不得改变模型给出的答案。"""

    def test_fp16_weights_match_bf16_quality(self):
        """fp16 存储省一半字节、换来更快的读取，但它不许改变模型选哪个 token。"""
        ids = _text_ids()
        lp16, top16 = _teacher_forced(ids, quant=False, weight_dtype=torch.float16)
        lpbf, topbf = _teacher_forced(ids, quant=False, weight_dtype=torch.bfloat16)

        # fp16 的尾数位比 bf16 多，指数位却更窄；风险在取值范围上，所以 logprob
        # 一旦非有限就判失败，哪怕一致率看着没问题。
        self.assertTrue(torch.isfinite(lp16).all(), "fp16 weights produced non-finite logprob")
        self.assertTrue(torch.isfinite(lpbf).all(), "bf16 weights produced non-finite logprob")

        agree = (top16 == topbf).float().mean().item()
        _, ppl16 = _report(lp16)
        _, pplbf = _report(lpbf)
        self.assertEqual(tuple(lp16.shape), tuple(lpbf.shape))
        # 实测 98.5%；门槛压到 95% 是给换 prompt 或换 seed 留余量。
        self.assertGreater(agree, 0.95,
                           f"fp16 vs bf16 top-1 agreement {agree:.1%} ({(top16 == topbf).sum()}/{len(ids) - 5})")
        # 实测 5.108 对 5.113，差异小到可以忽略。
        self.assertLess(abs(ppl16 / pplbf - 1.0), 0.02,
                        f"fp16 ppl {ppl16:.3f} vs bf16 {pplbf:.3f}")

    def test_int8_weights_keep_quality(self):
        """INT8 是服务默认格式，误差预算因此是产品决策：漂移必须停在实测水平以内。"""
        ids = _text_ids()
        # 线上参照是 fp16（WG 权重），INT8 量化才是服务默认。
        lp16, top16 = _teacher_forced(ids, quant=False, weight_dtype=torch.float16)
        lp8, top8 = _teacher_forced(ids, quant=True)

        self.assertTrue(torch.isfinite(lp8).all(), "int8 weights produced non-finite logprob")
        agree = (top16 == top8).float().mean().item()
        _, ppl16 = _report(lp16)
        _, ppl8 = _report(lp8)
        self.assertEqual(tuple(lp8.shape), tuple(lp16.shape))
        # 实测 96.2%（127/132）；按行做的 W8 粒度粗，所以门槛只压在 90%。
        self.assertGreater(agree, 0.90,
                           f"int8 vs fp16 top-1 agreement {agree:.1%}")
        # 实测 +0.29%；2% 的窗口足够抓住真正的量化回归。
        self.assertLess(abs(ppl8 / ppl16 - 1.0), 0.02,
                        f"int8 ppl {ppl8:.3f} vs fp16 {ppl16:.3f}")
