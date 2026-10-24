"""Blocked attention paths must agree with the unblocked ones.

Long context needed two restructurings, and neither may move a single number:

* prefill streams keys in blocks of ``PREFILL_ATTN_KB`` with a running max
  (online softmax) -- the score block used to be O(context), 8.6 GB at 256k;
* the decode kernel streams its window in blocks of ``ATTN_KEY_BLOCK`` with
  online softmax -- it used to hold the whole window in ``[S]`` floats in shared
  memory, which does not fit past ~15k.

So this is an A/B on the same model and the same prompt.  Port of
``diag/check_attn_block.py``; the batched-vs-sequential prefill check is the
port of ``diag/check_prefill.py``.

Agreement is judged in bf16 ULPs, not fp32 relative error: everything this
engine emits is bf16, so two identical fp32 computations that differ only in
accumulation order still land on different bf16 values at rounding boundaries.
"""

from __future__ import annotations

import unittest

import support

import torch

#: >1024 so a decode through the cache spans several kernel blocks, and >2048 so
#: the last prefill chunk still has a full-width score block to mask.
PREFILL_TOKENS = 2048
KB_SWEEP = (4096, 1024, 256, 64, 32)     # 4096 covers the whole 2048-token prompt
ATTN_KEY_BLOCK = 1024                     # keep in sync with kernels/attn.cu

BASE = [9707, 11, 1879, 3671, 374, 264, 3449, 315, 9847, 13,
        3838, 374, 279, 6722, 315, 9625, 323, 11751, 13]


def _tokens(n: int) -> list[int]:
    return [BASE[i % len(BASE)] for i in range(n)]


def _ulp(x: torch.Tensor) -> float:
    """One bf16 ULP at the magnitude of the largest entry of ``x``."""
    return max(float(x.abs().max()), 1e-6) * 2.0 ** -8


def _compare(a: torch.Tensor, b: torch.Tensor, max_ulps: float = 4.0):
    """(agree, max_diff, differing, argmax_same) of two logit tensors."""
    unit = _ulp(b)
    diff = float((a - b).abs().max())
    differing = int((a != b).sum())
    same = int(a.argmax()) == int(b.argmax())
    return same and diff <= max_ulps * unit, diff, differing, same


@support.needs_hip
@support.needs_model
class AttnBlockTest(support.ModelTestCase):
    QUANT = False
    MAX_SEQ = 4096

    def test_prefill_key_block_width_is_a_no_op(self):
        from models.qwen35 import full_attn

        toks = _tokens(PREFILL_TOKENS)
        reference = None
        reference_kb = None
        try:
            for kb in KB_SWEEP:
                full_attn.PREFILL_ATTN_KB = kb
                self.model.reset()
                logits = self.model.prefill(toks).float().clone()
                torch.cuda.synchronize()
                if reference is None:
                    reference, reference_kb = logits, kb
                    continue
                ok, diff, differing, same = _compare(logits, reference)
                with self.subTest(kb=kb):
                    self.assertTrue(
                        same,
                        f"KB={kb} argmax differs from KB={reference_kb}")
                    self.assertLessEqual(
                        diff, 4.0 * _ulp(reference),
                        f"KB={kb} vs {reference_kb}: max|d|={diff:.3e}, "
                        f"differing {differing}/{logits.numel()}")
        finally:
            full_attn.PREFILL_ATTN_KB = 1024

    def test_decode_kernel_matches_eager_attention(self):
        """The blocked decode kernel vs the unblocked eager reference."""
        import hip_ops
        from models.qwen35 import full_attn
        from ops.dtypes import D, NEG_INF

        def eager_attn(q, gate, kcache, vcache, kscale, vscale, out, pos, rep,
                       scale, part_acc=None, part_ml=None):
            """Whole-window attention: the maths the kernel replaced."""
            nh, hd = q.shape
            nkv = kcache.shape[0]
            seq = kcache.shape[1]
            mask_idx = torch.arange(seq, device=q.device)
            kk = kcache.to(D)
            qe = q.reshape(nkv, rep, hd)
            scores = torch.einsum("ghd,gsd->ghs", qe, kk).float()
            scores = scores * kscale[:, None, :] * scale
            mask = (mask_idx[None, :] > pos[:, None])[None, :, :]
            scores = scores.masked_fill(mask, NEG_INF)
            probs = scores.softmax(-1).to(D)
            pv = probs * vscale[:, None, :].to(D)
            y = torch.einsum("ghs,gsd->ghd", pv, vcache.to(D)).reshape(nh, hd)
            return y * gate.sigmoid()

        toks = _tokens(PREFILL_TOKENS)
        real = hip_ops.attn
        try:
            for step in range(3):
                outs = []
                for fn in (real, eager_attn):
                    self.model.reset()
                    self.model.prefill(toks)
                    for i in range(step):
                        self.model.logits(BASE[i])
                    hip_ops.attn = fn
                    outs.append(self.model.logits(BASE[step]).float().clone())
                hip_ops.attn = real
                pos = PREFILL_TOKENS + step
                blocks = -(-(pos + 1) // ATTN_KEY_BLOCK)
                ok, diff, differing, same = _compare(outs[0], outs[1])
                with self.subTest(pos=pos, blocks=blocks):
                    self.assertTrue(same, f"decode pos={pos}: argmax differs from eager")
                    self.assertLessEqual(
                        diff, 4.0 * _ulp(outs[1]),
                        f"decode pos={pos} ({blocks} blocks): max|d|={diff:.3e}, "
                        f"differing {differing}/{outs[0].numel()}")
        finally:
            hip_ops.attn = real


@support.needs_model
class PrefillEquivalenceTest(support.ModelTestCase):
    """Batched prefill must pick the same next token as walking the prompt."""

    QUANT = False
    MAX_SEQ = 4096

    def test_batched_matches_token_by_token(self):
        for n in (5, 100, 700):
            with self.subTest(n=n):
                toks = _tokens(n)
                self.model.reset()
                last = None
                for tok in toks:
                    last = self.model.logits(tok).float().clone()
                torch.cuda.synchronize()

                self.model.reset()
                batched = self.model.prefill(toks).float().clone()
                torch.cuda.synchronize()

                self.assertEqual(int(last.argmax()), int(batched.argmax()),
                                 f"n={n}: batched prefill picked a different token")
