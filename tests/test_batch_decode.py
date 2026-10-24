"""Static-batch decode (B in 1/2/4/8) must return the single-stream numbers.

The stage-1 batch path batches only the projections -- the weight reads -- and
runs attention / the delta-net recurrence once per row through the *same*
kernels the single-stream engine uses, on row slices of per-row state buffers.
Every row therefore has to match a solo decode exactly, not merely closely:
``models/qwen35/batch.py`` replays the same GEMV unroll the single-stream kernel
picked for each shape, so the partial sums are identical and no bf16 rounding
drifts.  The logit tensors are compared bit for bit here; a tolerance would hide
exactly the class of bug this test exists for.

Rows: ``enable_batching(capacity)`` reserves the last row for the single-stream
prefill/verify path (``prefill_row``); decode rows are ``0 .. capacity-2``.  A
new request is prefilled on the staging row and its recurrent state is moved to
a decode row with ``move_linear_states`` -- so a prefill between decode steps
cannot disturb a live decode row.

Five claims:

1. the single-stream graph (``decode_step``) equals ``decode_batch(1, 1)``;
2. the same prompt copied into B rows: every row equals the solo decode;
3. B rows with different prompt lengths (different block tables and positions):
   each row equals *its own* solo decode;
4. a bucket larger than the number of real rows: sentinel padding rows do not
   disturb the real rows, and different buckets agree;
5. interleaving: prefilling a new request in the middle of a decode does not
   disturb the rows already decoding.
"""

from __future__ import annotations

import unittest

import support

import torch

BLOCK = 64
DECODE_CAP = 8            # real decode rows
BATCH_CAP = DECODE_CAP + 1   # + the single-stream prefill/verify staging row
BUCKETS = (1, 2, 4, 8)
STEPS = 6
NUM_BLOCKS = 192

BASE = [9707, 11, 1879, 3671, 374, 264, 3449, 315, 9847, 13,
        3838, 374, 279, 6722, 315, 9625, 323, 11751, 13]


def _tokens(n: int, off: int = 0) -> list[int]:
    return [BASE[(i + off) % len(BASE)] for i in range(n)]


@support.needs_model
class BatchDecodeTest(support.ModelTestCase):
    QUANT = True
    MAX_SEQ = 2048
    PAGED = True

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.model = support.ModelFixture.get(quant=cls.QUANT, max_seq=cls.MAX_SEQ,
                                             paged=True, bench=False)
        cls.pool = cls.model.enable_paging(num_blocks=NUM_BLOCKS, block_size=BLOCK)
        cls.model.enable_batching(BATCH_CAP)
        cls.model.capture_batch(BUCKETS)
        # The single-stream engine's own graph (``decode_step``); ``_forward``
        # writes the staging row, so it is independent of the decode rows.
        cls.model.capture()
        cls.prefill_row = cls.model.prefill_row
        assert cls.prefill_row == BATCH_CAP - 1

    @classmethod
    def tearDownClass(cls) -> None:
        support.ModelFixture.close()

    def _alloc(self, n_tokens: int) -> list[int]:
        return [self.pool.alloc() for _ in range(-(-n_tokens // BLOCK))]

    def _free(self, blocks: list[int]) -> None:
        self.pool.free_blocks(blocks)

    def _zero_prefill_row(self) -> None:
        self.model.pos1.zero_()
        self.model.zero_linear_states(self.prefill_row)

    def _stage(self, ids: list[int], row: int) -> tuple[list[int], torch.Tensor]:
        """Prefill ``ids`` on the staging row, then move its state to ``row``.

        The KV lands in the sequence's own blocks (written through the staging
        row's slot map); the recurrent state is copied into the decode row that
        will carry on generating.
        """
        blocks = self._alloc(len(ids))
        self._zero_prefill_row()
        self.pool.assign_table(blocks, 0, row=self.prefill_row)
        logits = self.model.prefill(ids).float().clone()
        self.model.move_linear_states(self.prefill_row, row)
        return blocks, logits

    def _reference(self, ids: list[int]) -> tuple[list[int], list[torch.Tensor]]:
        """Solo decode of ``ids``: ``STEPS+1`` tokens and one logit tensor per step."""
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
        return [self.model.snapshot_linear_states(row) for row in range(rows)]

    def _restore(self, snap) -> None:
        for row, states in enumerate(snap):
            self.model.restore_linear_states(states, row)

    def _batch_decode(self, rows: int, bucket: int, blocks: list[list[int]],
                      first_tokens: list[int], first_pos: list[int]):
        """Run ``rows`` sequences through ``bucket``'s graph.

        Returns ``STEPS+1`` generated tokens per row and, per step, the logits of
        the valid rows (``STEPS`` tensors).
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
        """Decode with fixed, possibly non-contiguous live rows (row holes)."""
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
        self.assertEqual(tuple(got.shape), tuple(ref.shape), label)
        diff = float((got - ref).abs().max())
        differing = int((got != ref).sum())
        self.assertEqual(differing, 0,
                         f"{label}: {differing} logits differ from solo decode "
                         f"(max abs diff {diff:.3e}, {got.numel()} values)")

    def test_single_stream_graph_matches_bucket1(self):
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
            # two different buckets for the same 3 real rows: 4 (1 sentinel row)
            # and 8 (5 sentinel rows).  Padding must be invisible.
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
        # and the two padded runs agree with each other on the valid rows
        for s in range(STEPS):
            self._assert_exact(runs[8][1][s][:rows], runs[4][1][s][:rows],
                               f"bucket 8 vs 4 step={s}")

    def test_interleaved_prefill_does_not_disturb_decode_rows(self):
        """The continuous-batching shape: decode a batch, insert a prefill, go on."""
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
            # two steps with the first two rows only (bucket 2)
            toks = list(first)
            for s in range(2):
                self.model.set_rows(toks, [p + s for p in pos], blocks)
                logits = self.model.decode_batch(2, 2).float().clone()
                for row in range(2):
                    self._assert_exact(logits[row], refs[row][1][s],
                                       f"pre-insert row={row} step={s}")
                toks = [int(t) for t in logits.argmax(-1).tolist()]
            # insert a third request: prefill on the staging row, move to row 2
            blk, logits = self._stage(prompts[2], 2)
            blocks.append(blk)
            first.append(int(logits.argmax()))
            pos.append(len(prompts[2]))
            self.assertEqual(int(logits.argmax()), refs[2][0][0],
                             "inserted request's first token differs")
            # continue: rows 0/1 are at step 2, row 2 is at step 0
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
        """Runner shape: live rows 0 and 2, row 1 a hole; two decodes must match solo."""
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
            # row 1 is a hole: make sure whatever state it holds is finite, then
            # let decode_batch overwrite its token/pos/slot map with the sentinel.
            self.model.zero_linear_states(1)
            bucket = 4                       # min power of two > max(live row)
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
