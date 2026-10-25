"""Weight-format quality: does fp16 storage or int8 quantisation change the model?

Module: ``coffin/tests``. Loads two differently-quantised models through
``support.ModelFixture``, so it must run alone or last: two 4B models plus KV do
not fit on the card at once.

Constraints:
    * Teacher forcing, never free-running generation: any perturbation flips one
      argmax eventually and the two continuations then describe different (both
      valid) text, so free-running equality is not a property of correctness.
    * Both models must see the same ground-truth prefix; that is what makes the
      top-1 agreement and the perplexity of the true next token comparable.
    * A non-finite logprob is a failure even when the agreement looks fine:
      fp16 has a narrower exponent than bf16, so the risk of the format is range
      overflow, and an inf would corrupt the perplexity rather than flag itself.

Port of ``diag/check_int8.py``, ``diag/check_int8_ppl.py`` and
``diag/check_f16_ppl.py`` (accuracy halves; the speed halves stay in the probes).
The documented measurements are:

* fp16 vs bf16 weights: 98.5% (130/132) top-1 agreement, ppl 5.108 vs 5.113;
* int8 (W8A16) vs fp16: 96.2% (127/132) agreement, ppl +0.29%.
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
    """Tokenise the fixed passage once; a per-test tokenisation could differ by a merge."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(support.MODEL_DIR))
    return tok.encode(TEXT)


def _teacher_forced(ids: list[int], *, quant: bool, weight_dtype=None):
    """Logprob of the true next token and the top-1 pick, position by position.

    Args:
        ids: Ground-truth token ids both models are fed.
        quant: Load the model with INT8 weights.
        weight_dtype: Override the storage dtype; ``None`` keeps the checkpoint's.

    Returns:
        ``(logprobs, top1)``. ``logprobs[i]`` is the log-softmax of the token that
        actually comes next, ``top1[i]`` the model's own argmax at that position,
        so comparing the two across formats separates "stored differently" from
        "predicts differently".
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
    """``(mean negative log-likelihood, perplexity)`` over the scored positions."""
    nll = -lp.mean().item()
    return nll, math.exp(nll)


@support.needs_model
class WeightFormatQualityTest(unittest.TestCase):
    """The fp16/int8 storage choices must not move the model's answers."""

    def test_fp16_weights_match_bf16_quality(self):
        """fp16 stores weights for speed and halves the bytes; it may not change which token the model picks."""
        ids = _text_ids()
        lp16, top16 = _teacher_forced(ids, quant=False, weight_dtype=torch.float16)
        lpbf, topbf = _teacher_forced(ids, quant=False, weight_dtype=torch.bfloat16)

        # fp16 has more mantissa bits but a narrower exponent; the risk is range,
        # so a non-finite logprob is a failure even if the agreement looks fine.
        self.assertTrue(torch.isfinite(lp16).all(), "fp16 weights produced non-finite logprob")
        self.assertTrue(torch.isfinite(lpbf).all(), "bf16 weights produced non-finite logprob")

        agree = (top16 == topbf).float().mean().item()
        _, ppl16 = _report(lp16)
        _, pplbf = _report(lpbf)
        self.assertEqual(tuple(lp16.shape), tuple(lpbf.shape))
        # Measured 98.5%; 95% leaves room for a different prompt/seed.
        self.assertGreater(agree, 0.95,
                           f"fp16 vs bf16 top-1 agreement {agree:.1%} ({(top16 == topbf).sum()}/{len(ids) - 5})")
        # Measured 5.108 vs 5.113 (penny-level difference).
        self.assertLess(abs(ppl16 / pplbf - 1.0), 0.02,
                        f"fp16 ppl {ppl16:.3f} vs bf16 {pplbf:.3f}")

    def test_int8_weights_keep_quality(self):
        """INT8 is the serve default, so its error budget is a product decision: it must stay inside the measured drift."""
        ids = _text_ids()
        # fp16 is the production reference (WG weights), int8 is the serve default.
        lp16, top16 = _teacher_forced(ids, quant=False, weight_dtype=torch.float16)
        lp8, top8 = _teacher_forced(ids, quant=True)

        self.assertTrue(torch.isfinite(lp8).all(), "int8 weights produced non-finite logprob")
        agree = (top16 == top8).float().mean().item()
        _, ppl16 = _report(lp16)
        _, ppl8 = _report(lp8)
        self.assertEqual(tuple(lp8.shape), tuple(lp16.shape))
        # Measured 96.2% (127/132); per-row W8 is coarse, so keep the bar at 90%.
        self.assertGreater(agree, 0.90,
                           f"int8 vs fp16 top-1 agreement {agree:.1%}")
        # Measured +0.29%; 2% catches a real quantisation regression.
        self.assertLess(abs(ppl8 / ppl16 - 1.0), 0.02,
                        f"int8 ppl {ppl8:.3f} vs fp16 {ppl16:.3f}")
