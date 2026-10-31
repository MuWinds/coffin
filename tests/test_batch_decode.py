"""静态批解码（B 取 1/2/4/8）必须复现单流解码的数字。

Module: ``coffin/tests``。直接驱动批处理模型 API（分页、批捕获、按行搬移状态），
不经过 ``llm_engine``；同一条断言在调度器层面的版本放在 ``test_engine.py``。

Constraints:
    * 比较是逐位相等，绝不放宽成容差：``models/qwen35/batch.py`` 对每种形状重放
      单流内核选中的同一份 GEMV 展开，部分和因此完全一致，bf16 舍入没有漂移的
      余地。容差恰好会掩盖这个测试要抓的那类 bug。
    * 每次批运行都按行对照单独 ``decode_step`` 的参考结果：两次批运行互相吻合
      时，共用的 bug 一样能通过。
    * 最后一行是暂存行，解码行到 ``capacity-2`` 为止；解码后打的状态快照必须在
      下一个 bucket 之前恢复，因为计算图是原地写入的。
    * 块在 ``finally`` 里释放；池小到泄漏一次就会改变哪个 bucket 还装得下。

stage-1 的批处理路径只批投影——也就是权重的读取——注意力和 delta-net 循环按行
经由单流引擎所用的同一批内核，在按行状态缓冲的行切片上执行。

行：``enable_batching(capacity)`` 把最后一行留给单流的预填充/校验路径
（``prefill_row``）；解码行是 ``0 .. capacity-2``。新请求先在暂存行上预填充，
再用 ``move_linear_states`` 把它的循环状态搬到某个解码行——这样解码步骤之间的
预填充不会扰动正在解码的行。

五条断言：

1. 单流计算图（``decode_step``）等于 ``decode_batch(1, 1)``；
2. 同一 prompt 复制进 B 行：每一行都等于单独解码；
3. B 行使用不同 prompt 长度（块表和位置各不相同）：每行等于它自己的单独解码；
4. bucket 大于真实行数：哨兵填充行不扰动真实行，且不同 bucket 结果一致；
5. 交叉执行：在解码中途预填充新请求，不影响已经在解码的行。
"""

from __future__ import annotations

import unittest

import support

import torch

BLOCK = 64
DECODE_CAP = 8            # 真实的解码行数
BATCH_CAP = DECODE_CAP + 1   # 再加上单流预填充/校验的暂存行
BUCKETS = (1, 2, 4, 8)
STEPS = 6
NUM_BLOCKS = 192

BASE = [9707, 11, 1879, 3671, 374, 264, 3449, 315, 9847, 13,
        3838, 374, 279, 6722, 315, 9625, 323, 11751, 13]


def _tokens(n: int, off: int = 0) -> list[int]:
    return [BASE[(i + off) % len(BASE)] for i in range(n)]


@support.needs_model
class BatchDecodeTest(support.ModelTestCase):
    """批解码中的一行必须与该序列单独解码的结果逐位相同。"""

    QUANT = True
    MAX_SEQ = 2048
    PAGED = True

    @classmethod
    def setUpClass(cls) -> None:
        """建好分页、bucket 与两张计算图：比较需要批捕获和单流捕获同时在场。"""
        super().setUpClass()
        cls.model = support.ModelFixture.get(quant=cls.QUANT, max_seq=cls.MAX_SEQ,
                                             paged=True, bench=False)
        cls.pool = cls.model.enable_paging(num_blocks=NUM_BLOCKS, block_size=BLOCK)
        cls.model.enable_batching(BATCH_CAP)
        cls.model.capture_batch(BUCKETS)
        # 单流引擎自己的图（``decode_step``）。``_forward`` 写的是暂存行，
        # 因此它和解码行互不干扰。
        cls.model.capture()
        cls.prefill_row = cls.model.prefill_row
        assert cls.prefill_row == BATCH_CAP - 1

    @classmethod
    def tearDownClass(cls) -> None:
        """在下一个模块申请自己的显存之前，把模型和它占用的显存交还回去。"""
        support.ModelFixture.close()

    def _alloc(self, n_tokens: int) -> list[int]:
        """按池的块大小分配足以覆盖 ``n_tokens`` 的块。"""
        return [self.pool.alloc() for _ in range(-(-n_tokens // BLOCK))]

    def _free(self, blocks: list[int]) -> None:
        """把块还回池；每个 ``_alloc`` 都配一次本调用。"""
        self.pool.free_blocks(blocks)

    def _zero_prefill_row(self) -> None:
        """清空暂存行，否则上一个请求的位置会漏进这次预填充。"""
        self.model.pos1.zero_()
        self.model.zero_linear_states(self.prefill_row)

    def _stage(self, ids: list[int], row: int) -> tuple[list[int], torch.Tensor]:
        """在暂存行上预填充 ``ids``，再把它的状态搬到 ``row``。

        KV 落在序列自己的块里（经由暂存行的槽位表写入）；循环状态则拷进接下来
        继续生成的那个解码行。
        """
        blocks = self._alloc(len(ids))
        self._zero_prefill_row()
        self.pool.assign_table(blocks, 0, row=self.prefill_row)
        logits = self.model.prefill(ids).float().clone()
        self.model.move_linear_states(self.prefill_row, row)
        return blocks, logits

    def _reference(self, ids: list[int]) -> tuple[list[int], list[torch.Tensor]]:
        """``ids`` 的单独解码：``STEPS+1`` 个 token，每一步一个 logits 张量。"""
        blocks = self._alloc(len(ids))
        try:
            self.model.reset()
            self.pool.assign_table(blocks, 0, row=self.prefill_row)
            logits = self.model.prefill(ids)
            tokens: list[int] = []
            logits_out: list[torch.Tensor] = []
            for _ in range(STEPS):
                tok = int(logits.argmax())
                tokens.append(tok)
                logits = self.model.decode_step(tok)
                logits_out.append(logits.float().clone())
            tokens.append(int(logits.argmax()))
            return tokens, logits_out
        finally:
            self._free(blocks)

    def _snapshot(self, rows: int):
        """拷出每一行的循环状态；计算图原地写入，所以一次 bucket 运行必须可撤销。"""
        return [self.model.snapshot_linear_states(row) for row in range(rows)]

    def _restore(self, snap) -> None:
        """把快照里的状态放回去，让下一个 bucket 从同一起点开始。"""
        for row, states in enumerate(snap):
            self.model.restore_linear_states(states, row)

    def _batch_decode(self, rows: int, bucket: int, blocks: list[list[int]],
                      first_tokens: list[int], first_pos: list[int]):
        """让 ``rows`` 条序列走 ``bucket`` 对应的计算图。

        每行返回 ``STEPS+1`` 个已生成 token，并按步骤返回有效行的 logits
        （``STEPS`` 个张量）。
        """
        tokens: list[list[int]] = [list(first_tokens)]
        logits_out: list[torch.Tensor] = []
        tok = list(first_tokens)
        for s in range(STEPS):
            self.model.set_rows(tok, [p + s for p in first_pos], blocks)
            logits = self.model.decode_batch(rows, bucket).float().clone()
            logits_out.append(logits)
            tok = [int(t) for t in logits[:rows].argmax(-1).tolist()]
            tokens.append(tok)
        return tokens, logits_out

    def _batch_decode_rows(self, live_rows: list[int], bucket: int,
                           blocks: dict[int, list[int]], first: dict[int, int],
                           pos: dict[int, int]):
        """在固定、可能不连续的活动行上解码（行空洞）。"""
        tokens: dict[int, list[int]] = {r: [first[r]] for r in live_rows}
        logits_out: list[torch.Tensor] = []
        pad = [r for r in range(bucket) if r not in live_rows]
        rows = max(live_rows) + 1
        for s in range(STEPS):
            self.model.set_rows(
                [tokens[r][-1] for r in live_rows],
                [pos[r] + s for r in live_rows],
                [blocks[r] for r in live_rows],
                row_indices=live_rows)
            logits = self.model.decode_batch(rows, bucket, pad_rows=pad).float().clone()
            logits_out.append(logits)
            for r in live_rows:
                tokens[r].append(int(logits[r].argmax()))
        return tokens, logits_out

    def _assert_exact(self, got: torch.Tensor, ref: torch.Tensor, label: str) -> None:
        """逐位相等，并用数量报告差异：一个 ULP 的偏差和整体发散能区分开。"""
        self.assertEqual(tuple(got.shape), tuple(ref.shape), label)
        diff = float((got - ref).abs().max())
        differing = int((got != ref).sum())
        self.assertEqual(differing, 0,
                         f"{label}: {differing} logits differ from solo decode "
                         f"(max abs diff {diff:.3e}, {got.numel()} values)")

    def test_single_stream_graph_matches_bucket1(self):
        """Bucket 1 是批处理路径的退化情形；它一旦对不上，更大的 bucket 都不必信。"""
        ids = _tokens(48)
        ref_tokens, ref_logits = self._reference(ids)
        blocks, logits = self._stage(ids, 0)
        try:
            got_tokens, got_logits = self._batch_decode(
                1, 1, [blocks], [int(logits.argmax())], [len(ids)])
            for s in range(STEPS + 1):
                self.assertEqual(got_tokens[s][0], ref_tokens[s],
                                 f"bucket 1 vs decode_step step={s}: token differs")
            for s in range(STEPS):
                self._assert_exact(got_logits[s][0], ref_logits[s],
                                   f"bucket 1 vs decode_step step={s}")
        finally:
            self._free(blocks)

    def test_same_prompt_every_row_matches_solo(self):
        """内容相同的行必须保持相同：按行状态或槽位表串了行，会表现为两行分叉。"""
        ids = _tokens(48)
        ref_tokens, ref_logits = self._reference(ids)
        blocks: list[list[int]] = []
        first: list[int] = []
        pos: list[int] = []
        for row in range(DECODE_CAP):
            blk, logits = self._stage(ids, row)
            blocks.append(blk)
            first.append(int(logits.argmax()))
            pos.append(len(ids))
        snap = self._snapshot(DECODE_CAP)
        try:
            for B in BUCKETS:
                with self.subTest(B=B):
                    self._restore(snap[:B])
                    got_tokens, got_logits = self._batch_decode(
                        B, B, blocks[:B], first[:B], pos[:B])
                    for row in range(B):
                        for s in range(STEPS + 1):
                            self.assertEqual(got_tokens[s][row], ref_tokens[s],
                                             f"B={B} row={row} step={s}: token differs")
                        for s in range(STEPS):
                            self._assert_exact(got_logits[s][row], ref_logits[s],
                                               f"B={B} row={row} step={s}")
        finally:
            for blk in blocks:
                self._free(blk)

    def test_different_prefix_lengths_match_their_own_solo(self):
        """长度不同意味着位置和块表都不同，共用位置缓冲正是在这里露出破绽。"""
        lengths = [40, 57, 71, 96, 130, 66, 85, 112]
        prompts = [_tokens(n, off=i) for i, n in enumerate(lengths)]
        refs = [self._reference(p) for p in prompts]
        blocks: list[list[int]] = []
        first: list[int] = []
        pos: list[int] = []
        for row, ids in enumerate(prompts):
            blk, logits = self._stage(ids, row)
            blocks.append(blk)
            first.append(int(logits.argmax()))
            pos.append(len(ids))
        snap = self._snapshot(len(prompts))
        try:
            for B in BUCKETS:
                with self.subTest(B=B):
                    self._restore(snap[:B])
                    got_tokens, got_logits = self._batch_decode(
                        B, B, blocks[:B], first[:B], pos[:B])
                    for row in range(B):
                        ref_tokens, ref_logits = refs[row]
                        for s in range(STEPS + 1):
                            self.assertEqual(got_tokens[s][row], ref_tokens[s],
                                             f"B={B} row={row} step={s}: token differs")
                        for s in range(STEPS):
                            self._assert_exact(got_logits[s][row], ref_logits[s],
                                               f"B={B} row={row} step={s}")
        finally:
            for blk in blocks:
                self._free(blk)

    def test_bucket_padding_does_not_disturb_valid_rows(self):
        """bucket 由最高的活动行决定，所以多数步骤都带着哨兵行；哨兵若写了真实状态就会污染相邻行。"""
        rows = 3
        lengths = [45, 70, 120]
        prompts = [_tokens(n, off=3 * i) for i, n in enumerate(lengths)]
        refs = [self._reference(p) for p in prompts]
        blocks: list[list[int]] = []
        first: list[int] = []
        pos: list[int] = []
        for row, ids in enumerate(prompts):
            blk, logits = self._stage(ids, row)
            blocks.append(blk)
            first.append(int(logits.argmax()))
            pos.append(len(ids))
        snap = self._snapshot(rows)
        runs = {}
        try:
            # 同样 3 个真实行用两个不同的 bucket：4（1 个哨兵行）和 8（5 个哨兵行）。
            # 填充必须完全不可见。
            for bucket in (4, 8):
                with self.subTest(bucket=bucket):
                    self._restore(snap)
                    runs[bucket] = self._batch_decode(
                        rows, bucket, blocks, first, pos)
                    for row in range(rows):
                        ref_tokens, ref_logits = refs[row]
                        for s in range(STEPS + 1):
                            self.assertEqual(runs[bucket][0][s][row], ref_tokens[s],
                                             f"bucket={bucket} row={row} step={s}: token differs")
                        for s in range(STEPS):
                            self._assert_exact(runs[bucket][1][s][row], ref_logits[s],
                                               f"bucket={bucket} row={row} step={s}")
        finally:
            for blk in blocks:
                self._free(blk)
        # 这两次带填充的运行在有效行上也必须彼此吻合
        for s in range(STEPS):
            self._assert_exact(runs[8][1][s][:rows], runs[4][1][s][:rows],
                               f"bucket 8 vs 4 step={s}")

    def test_interleaved_prefill_does_not_disturb_decode_rows(self):
        """连续批处理的形态：先解码一批，中途插入一次预填充，再继续解码。"""
        prompts = [_tokens(45, 0), _tokens(57, 1), _tokens(70, 2)]
        refs = [self._reference(p) for p in prompts]
        blocks: list[list[int]] = []
        first: list[int] = []
        pos: list[int] = []
        for row, ids in enumerate(prompts[:2]):
            blk, logits = self._stage(ids, row)
            blocks.append(blk)
            first.append(int(logits.argmax()))
            pos.append(len(ids))
        try:
            # 只用前两行先走两步（bucket 2）
            toks = list(first)
            for s in range(2):
                self.model.set_rows(toks, [p + s for p in pos], blocks)
                logits = self.model.decode_batch(2, 2).float().clone()
                for row in range(2):
                    self._assert_exact(logits[row], refs[row][1][s],
                                       f"pre-insert row={row} step={s}")
                toks = [int(t) for t in logits.argmax(-1).tolist()]
            # 插入第三个请求：在暂存行上预填充，再搬到第 2 行
            blk, logits = self._stage(prompts[2], 2)
            blocks.append(blk)
            first.append(int(logits.argmax()))
            pos.append(len(prompts[2]))
            self.assertEqual(int(logits.argmax()), refs[2][0][0],
                             "inserted request's first token differs")
            # 继续：第 0/1 行已到第 2 步，第 2 行还在第 0 步
            tok2 = [toks[0], toks[1], first[2]]
            for k in range(STEPS - 2):
                self.model.set_rows(tok2, [pos[0] + 2 + k, pos[1] + 2 + k, pos[2] + k],
                                    blocks)
                logits = self.model.decode_batch(3, 4).float().clone()
                for row in range(2):
                    self._assert_exact(logits[row], refs[row][1][2 + k],
                                       f"post-insert row={row} step={2 + k}")
                self._assert_exact(logits[2], refs[2][1][k],
                                   f"post-insert row=2 step={k}")
                tok2 = [int(t) for t in logits[:3].argmax(-1).tolist()]
        finally:
            for blk in blocks:
                self._free(blk)


    def test_row_holes_are_padded_with_sentinels(self):
        """Runner 的形态：活动行是 0 和 2，第 1 行是空洞；两次解码都必须等于单独解码。"""
        prompts = [_tokens(45, 0), _tokens(70, 2)]
        refs = [self._reference(p) for p in prompts]
        live_rows = [0, 2]
        blocks: dict[int, list[int]] = {}
        first: dict[int, int] = {}
        pos: dict[int, int] = {}
        try:
            for r, ids in zip(live_rows, prompts):
                blk, logits = self._stage(ids, r)
                blocks[r] = blk
                first[r] = int(logits.argmax())
                pos[r] = len(ids)
            # 第 1 行是空洞：先让它持有的状态变成有限值，再让 decode_batch
            # 用哨兵覆盖它的 token/pos/槽位表。
            self.model.zero_linear_states(1)
            bucket = 4                       # 大于 max(live row) 的最小 2 的幂
            tokens, logits = self._batch_decode_rows(
                live_rows, bucket, blocks, first, pos)
            for i, r in enumerate(live_rows):
                ref_tokens, ref_logits = refs[i]
                for s in range(STEPS + 1):
                    self.assertEqual(tokens[r][s], ref_tokens[s],
                                     f"row={r} step={s}: token differs")
                for s in range(STEPS):
                    self._assert_exact(logits[s][r], ref_logits[s],
                                       f"hole row={r} step={s}")
        finally:
            for blk in blocks.values():
                self._free(blk)


if __name__ == "__main__":
    unittest.main()
