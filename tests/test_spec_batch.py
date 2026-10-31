"""批处理的投机解码：逐行草稿、校验与回滚。

Module: ``coffin/tests``。经批处理模型接口驱动 ``model.spec_step_batch``；
调度器目前还没有在生产路径上调用它，因此这个文件是它唯一的功能门禁。

Constraints:
    * 投机只允许改变速度：批处理投机路径吐出的每个 token，都必须是纯批处理
      贪心路径在同一位置会吐出的那个。对逐 token 校验路径，要逐行按位一致。
    * 批处理图、MTP 图和回滚图的峰值约 6 GiB，16 GB 的卡上放不进同一批的其他
      测试类旁边；所以用 ``RUN_SPEC_BATCH`` 选择加入，而不是当成正常测试。
    * 行空洞是契约的一部分，不是边角情况：引擎按序列固定行，并用哨兵填洞，
      只测连续行的用例永远覆盖不到填充逻辑。
    * 一次性前向校验路径按 token 一致判定，而不比对逐位 logits：它的宽 M 内核
      与贪心内核存在设计内的 ULP 级差异。

批处理投机只允许改变速度：它吐出的每个 token，都必须是纯批处理贪心路径在同一
位置会吐出的那个。这里验证的就是这条强断言——逐行、按位：

* 草稿模型（MTP 头，现在带逐行分页 KV 与桶图）为每行提出 K 个 token；
* 主模型在一次捕获好的前向里校验每行的 ``K+1`` 个 token，每校验完一个就把
  循环状态打一次快照；
* 每行各自接受匹配的前缀，从那个 token 的快照恢复自己的状态后继续。
"""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace

import support

import torch

BLOCK = 64
DECODE_CAP = 8
BATCH_CAP = DECODE_CAP + 1
BUCKETS = (1, 2, 4, 8)
K = 1
WIDTH = K + 1
STEPS = 10                     # 投机步数；每一步至少产出一个 token
GEN = 8                        # 用来比对的贪心 token 数
NUM_BLOCKS = 256

#: 这个类要占模型、批处理图、批处理 MTP 图和逐行回滚快照（峰值约 6 GiB），
#: 16 GB 的卡上放不进同一批的其他测试类旁边。所以全量套件会跳过它，单独跑：
#:     RUN_SPEC_BATCH=1 python3 -m unittest discover -s tests -p "test_spec_batch.py" -v
RUN_SPEC_BATCH = os.environ.get("RUN_SPEC_BATCH") == "1"

BASE = [9707, 11, 1879, 3671, 374, 264, 3449, 315, 9847, 13,
        3838, 374, 279, 6722, 315, 9625, 323, 11751, 13]


def _tokens(n: int, off: int = 0) -> list[int]:
    return [BASE[(i + off) % len(BASE)] for i in range(n)]


@support.needs_model
@unittest.skipUnless(RUN_SPEC_BATCH,
                     "needs ~6 GiB; run with RUN_SPEC_BATCH=1")
class BatchSpecTest(support.ModelTestCase):
    """批处理投机循环必须逐行产出与纯批处理贪心完全相同的 token。"""

    QUANT = True
    MAX_SEQ = 2048
    PAGED = True

    @classmethod
    def setUpClass(cls) -> None:
        """先捕获批处理图，再捕获重放它们的投机图。"""
        super().setUpClass()
        cls.model = support.ModelFixture.get(quant=cls.QUANT, max_seq=cls.MAX_SEQ,
                                             paged=True, bench=False)
        cls.pool = cls.model.enable_paging(num_blocks=NUM_BLOCKS, block_size=BLOCK)
        cls.model.enable_batching(BATCH_CAP)
        cls.model.capture_batch(BUCKETS)
        cls.model.enable_spec_batch(K, buckets=BUCKETS)

    @classmethod
    def tearDownClass(cls) -> None:
        """释放约 6 GiB 的投机图；下一个模块要用这张卡。"""
        support.ModelFixture.close()

    def _alloc(self, n_tokens: int) -> list[int]:
        """prompt 加上每一步可能追加的投机余量所需的块。"""
        return [self.pool.alloc() for _ in range(-(-(n_tokens + STEPS * WIDTH + WIDTH) // BLOCK))]

    def _free(self, blocks: list[int]) -> None:
        """把块还给池；每次 ``_alloc`` 都对应一次这个调用。"""
        self.pool.free_blocks(blocks)

    def _prefill(self, prompts: list[list[int]], rows=None):
        """按行分组预填充，返回块表和最后一个 token 的 logits。"""
        rows = list(range(len(prompts))) if rows is None else list(rows)
        blocks = [self._alloc(len(p)) for p in prompts]
        seqs = [SimpleNamespace(token_ids=p, row=r, block_table=b)
                for p, r, b in zip(prompts, rows, blocks)]
        logits, _states = self.model.prefill_group(seqs)
        return blocks, [lg.float().clone() for lg in logits]

    def _greedy(self, prompts: list[list[int]], rows=None) -> list[list[int]]:
        """对照实现：在同样的行上对同样的 prompt 做纯批处理贪心解码。"""
        k = len(prompts)
        rows = list(range(k)) if rows is None else list(rows)
        blocks, logits = self._prefill(prompts, rows)
        try:
            bucket = next(b for b in BUCKETS if b > max(rows))
            pad = [r for r in range(bucket) if r not in rows]
            chains = [[int(lg.argmax())] for lg in logits]
            for s in range(GEN - 1):
                self.model.set_rows([c[-1] for c in chains],
                                    [len(p) + s for p in prompts],
                                    blocks, row_indices=rows)
                lgs = self.model.decode_batch(max(rows) + 1, bucket, pad_rows=pad).float()
                for i, row in enumerate(rows):
                    chains[i].append(int(lgs[row].argmax()))
            return chains
        finally:
            for blk in blocks:
                self._free(blk)

    def _spec(self, prompts: list[list[int]], rows=None):
        """跑批处理投机循环，返回每行产出的 token 与接受记录。"""
        k = len(prompts)
        rows = list(range(k)) if rows is None else list(rows)
        blocks, logits = self._prefill(prompts, rows)
        try:
            bucket = next(b for b in BUCKETS if b > max(rows))
            cur = [int(lg.argmax()) for lg in logits]
            pos = [len(p) for p in prompts]
            produced = [[] for _ in range(k)]
            accepted: list[int] = []
            for _ in range(STEPS):
                toks, logits_next, acc = self.model.spec_step_batch(
                    cur, pos, blocks, bucket, K, row_indices=rows)
                for i in range(k):
                    produced[i].extend(toks[i])
                    pos[i] += len(toks[i])
                    cur[i] = int(logits_next[i].argmax())
                accepted.extend(acc)
            return produced, accepted
        finally:
            for blk in blocks:
                self._free(blk)

    def test_spec_matches_greedy_per_batch_size(self):
        """每个批大小都必须精确复现贪心；回滚 bug 只有在两行以不同方式回滚时才会露出来。"""
        lengths = [37, 53, 71, 44, 89, 64, 102, 47]
        for k in (1, 2, 4, 8):
            with self.subTest(k=k):
                prompts = [_tokens(lengths[i], off=5 * i) for i in range(k)]
                greedy = self._greedy(prompts)
                produced, accepted = self._spec(prompts)
                for row in range(k):
                    n = min(GEN, len(produced[row]))
                    self.assertGreaterEqual(n, GEN, f"k={k} row={row}: too few tokens")
                    self.assertEqual(produced[row][:GEN], greedy[row][:GEN],
                                     f"k={k} row={row}: spec tokens diverge from greedy")

    def test_spec_with_row_holes(self):
        """固定行之间留一个空洞（引擎调度时的形状）。"""
        prompts = [_tokens(45, 0), _tokens(70, 2)]
        rows = [0, 2]
        greedy = self._greedy(prompts)
        produced, _acc = self._spec(prompts, rows=rows)
        for i in range(len(prompts)):
            self.assertEqual(produced[i][:GEN], greedy[i][:GEN],
                             f"row={rows[i]}: spec tokens diverge from greedy")

    def test_bucket_padding_does_not_leak(self):
        """桶 4 里只有三行是真的：哨兵行不得影响它们。"""
        prompts = [_tokens(40, 0), _tokens(55, 1), _tokens(72, 2)]
        greedy = self._greedy(prompts)
        produced, _acc = self._spec(prompts)
        for i in range(len(prompts)):
            self.assertEqual(produced[i][:GEN], greedy[i][:GEN],
                             f"row={i}: padded-bucket spec tokens diverge")


@support.needs_model
@unittest.skipUnless(RUN_SPEC_BATCH, "needs ~6 GiB; run with RUN_SPEC_BATCH=1")
class BatchSpecOnceTest(support.ModelTestCase):
    """一次性前向校验（task-13）也必须与贪心逐 token 相同。

    在没有 task-12 宽 M GEMV 的代码树上，它会回退到预填充投影路径（正确但慢）；
    这里关心的是这个一次性前向校验的图 / 缓冲区 / 回滚布局，不是它的速度。
    """

    QUANT = True
    MAX_SEQ = 2048
    PAGED = True

    @classmethod
    def setUpClass(cls) -> None:
        """以一次性前向校验模式捕获一组更窄的桶；紧随其后的断言钉住本类所测的模式。"""
        super().setUpClass()
        cls.model = support.ModelFixture.get(quant=cls.QUANT, max_seq=cls.MAX_SEQ,
                                             paged=True, bench=False)
        cls.pool = cls.model.enable_paging(num_blocks=NUM_BLOCKS, block_size=BLOCK)
        cls.model.enable_batching(BATCH_CAP)
        cls.model.capture_batch((1, 2, 4))
        cls.model.enable_spec_batch(K, buckets=(1, 2, 4), verify_mode="once")
        assert cls.model.verify_mode == "once"

    @classmethod
    def tearDownClass(cls) -> None:
        """释放模型；这个类同样要吃满整张卡。"""
        support.ModelFixture.close()

    # 复用主类的辅助函数与缓冲区
    _alloc = BatchSpecTest._alloc
    _free = BatchSpecTest._free
    _prefill = BatchSpecTest._prefill
    _greedy = BatchSpecTest._greedy
    _spec = BatchSpecTest._spec

    def test_one_forward_verify_matches_greedy(self):
        """一次性前向校验必须同时与逐 token 路径和贪心一致，这样更宽的图才只是速度变化。

        一次性前向校验需要的宽 M 投影内核（kernels task-12）与贪心内核存在 ULP 级
        差异，所以这条路径按 token 一致判定（上面的逐 token 路径仍然按位比对）。
        """
        prompts = [_tokens(41, off=0), _tokens(63, off=2)]
        greedy = self._greedy(prompts)
        produced_once, accepted = self._spec(prompts)
        self.assertTrue(accepted and isinstance(accepted[0], int))

        # 同一步、换回逐 token 校验路径、同一模型（图会重新捕获），
        # 用来做 once 与 token 两条路径的一致性检查。
        self.model.set_verify_mode("token")
        produced_tok, _acc = self._spec(prompts)
        self.model.set_verify_mode("once")

        total = matched = 0
        for row in range(len(prompts)):
            for a, b in zip(produced_once[row][:GEN], greedy[row][:GEN]):
                total += 1
                matched += int(a == b)
            self.assertEqual(produced_once[row][:GEN], produced_tok[row][:GEN],
                             f"once-verify row={row}: differs from the token path")
        self.assertGreaterEqual(matched / total, 0.999,
                                f"once-verify agrees with greedy on {matched}/{total} tokens")


if __name__ == "__main__":
    unittest.main()
