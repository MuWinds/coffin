"""分块注意力路径必须与未分块的路径给出相同结果。

Module: ``coffin/tests``。加载模型和 HIP 扩展，然后在运行中的模型上替换注意力
入口；套件里其他测试不得这样 monkey-patch ``backends.ops``。

Constraints:
    * 一致性按 bf16 ULP 判定，而不是 fp32 相对误差：这个引擎吐出的张量全是
      bf16，两次完全相同的 fp32 计算只要累加顺序不同，就会在舍入边界上落到
      不同的 bf16 值。把界收紧只会让它因为舍入失败，而不是因为 bug。
    * 这里替换的内核全局量是模块状态，所以每次替换都在 ``finally`` 里还原；
      泄漏的替换会悄悄改掉后面测试实际测量的东西。
    * prompt 必须长于被测的最大块，否则块扫描永远跨不过边界，什么也证明不了。

长上下文逼出过两次重构，两次都不许移动任何一个数字：

* 预填充按 ``PREFILL_ATTN_KB`` 分块流式读取 key，配一个滚动最大值（online
  softmax）—— 原来的分数块是 O(context)，256k 时要 8.6 GB；
* 解码内核按 ``ATTN_KEY_BLOCK`` 分块流式处理窗口，同样用 online softmax ——
  原来把整个窗口放进共享内存的 ``[S]`` 个 float 里，超过约 15k 就放不下。

于是这里是同一模型、同一 prompt 上的 A/B 对比。移植自
``diag/check_attn_block.py``；批处理与逐 token 预填充的对照移植自
``diag/check_prefill.py``。
"""

from __future__ import annotations

import unittest

import support

import torch

#: 取大于 1024 是为了让一次经过缓存的解码跨过好几个内核块；大于 2048 是
#: 为了让最后一个预填充分块里还有满宽的分数块可以被 mask 掉。
PREFILL_TOKENS = 2048
KB_SWEEP = (4096, 1024, 256, 64, 32)     # 4096 能覆盖整个 2048 token 的 prompt
ATTN_KEY_BLOCK = 1024                     # 与 kernels/attn.cu 保持一致

BASE = [9707, 11, 1879, 3671, 374, 264, 3449, 315, 9847, 13,
        3838, 374, 279, 6722, 315, 9625, 323, 11751, 13]


def _tokens(n: int) -> list[int]:
    return [BASE[i % len(BASE)] for i in range(n)]


def _ulp(x: torch.Tensor) -> float:
    """``x`` 绝对值最大的那个元素所在量级上的一个 bf16 ULP。"""
    return max(float(x.abs().max()), 1e-6) * 2.0 ** -8


def _compare(a: torch.Tensor, b: torch.Tensor, max_ulps: float = 4.0):
    """比较两个 logits 张量，返回 ``(agree, max_diff, differing, argmax_same)``。"""
    unit = _ulp(b)
    diff = float((a - b).abs().max())
    differing = int((a != b).sum())
    same = int(a.argmax()) == int(b.argmax())
    return same and diff <= max_ulps * unit, diff, differing, same


@support.needs_hip
@support.needs_model
class AttnBlockTest(support.ModelTestCase):
    """块宽是性能旋钮：改它不许挪动任何一个 bf16 值。"""

    QUANT = False
    MAX_SEQ = 4096

    def test_prefill_key_block_width_is_a_no_op(self):
        """用 online softmax 的预填充在每个块宽下都必须落到同一份 logits 上。"""
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
        """分块的解码内核对手写未分块的 eager 参考实现。"""
        import dataclasses

        import backends
        from models.qwen35 import full_attn
        from ops.dtypes import D, NEG_INF
        def eager_attn(q, gate, kcache, vcache, kscale, vscale, out, pos, rep,
                       scale, part_acc=None, part_ml=None):
            """整窗注意力：内核替换掉的那套算法，作为对照留在这里。"""
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
        toks = _tokens(PREFILL_TOKENS)
        # ``backends.ops`` 是 frozen dataclass，不能就地改写字段。替代入口的正确做法是
        # 换一张表再挂回 ``full_attn`` 的模块属性 —— 旧版 monkey-patch ``hip_ops.attn``，
        # 而那个顶层模块在 CoreX 移植里搬进了 ``backends/hip/``，import 直接失败。
        real = backends.ops
        eager_table = dataclasses.replace(real, attn_decode=eager_attn)
        try:
            for step in range(3):
                outs = []
                for table in (real, eager_table):
                    self.model.reset()
                    self.model.prefill(toks)
                    for i in range(step):
                        self.model.logits(BASE[i])
                    full_attn.bops = table
                    outs.append(self.model.logits(BASE[step]).float().clone())
                full_attn.bops = real
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
            full_attn.bops = real

@support.needs_model
class PrefillEquivalenceTest(support.ModelTestCase):
    """批处理预填充必须挑出跟逐 token 走完 prompt 一样的下一个 token。"""

    QUANT = False
    MAX_SEQ = 4096

    def test_batched_matches_token_by_token(self):
        """批处理预填充和一次一个 token 地走完 prompt，必须挑出同一个下一个 token。"""
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
