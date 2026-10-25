"""Group prefill: several prompts in one forward.

Module: ``coffin/tests``. Drives ``Qwen35.prefill_group`` through the batched
model API (paging, batch capture, per-row decode); the scheduler-level use of the
same path is covered by ``test_spec_batch.py`` and ``test_prefix_batch.py``.

Constraints:
    * Agreement is argued in bf16 ULPs, never bit equality across batch sizes: the
      GEMM's M changed, so the fp32 accumulation order changed too, and
      bit-identical simply is not on the table. Only the one-sequence case is
      bitwise, because it takes the same M.
    * The model has ``capacity = max_bucket + 1`` rows, the last being the prefill
      staging row; a test that decodes on that row corrupts the next prefill, so
      states are always moved off it first.
    * Every block allocated here is freed in a ``finally``; a leak would make the
      later prefix-reuse assertions pass for the wrong reason (a full pool).
    * ``_solo`` is the oracle for every test: comparing group prefill against
      anything else would not separate a batching bug from a model bug.

The batched engine prefills each admitted request on its own, so a burst of k
requests pays the weight read k times.  ``Qwen35.prefill_group`` concatenates the
prompts, runs each layer's projection / MLP once over all the tokens, and slices
the result back per sequence for the parts that are genuinely per sequence
(attention over its own KV window and positions, the delta-net scan on its own
recurrent row).  The weights are read once per layer instead of once per prompt.

The acceptance contract:

1. a group prefill must agree with running the same prompts one at a time:
   argmax identical and logits within bf16 noise;
2. with one sequence the group path *is* bit-identical to ``prefill``;
3. the states are routed to the right rows: after a group prefill each sequence
   keeps decoding into tokens that match its own solo run;
4. the single-sequence ``prefill`` path itself is unchanged (covered by the rest
   of the suite).
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import support

import torch

BLOCK = 64
DECODE_CAP = 8
BATCH_CAP = DECODE_CAP + 1     # + the prefill/verify staging row
BUCKETS = (1, 2, 4, 8)
STEPS = 4
NUM_BLOCKS = 256

BASE = [9707, 11, 1879, 3671, 374, 264, 3449, 315, 9847, 13,
        3838, 374, 279, 6722, 315, 9625, 323, 11751, 13]


def _tokens(n: int, off: int = 0) -> list[int]:
    return [BASE[(i + off) % len(BASE)] for i in range(n)]


@support.needs_model
class PrefillBatchTest(support.ModelTestCase):
    """Group prefill must reproduce the per-sequence solo run, states and all."""

    QUANT = True
    MAX_SEQ = 2048
    PAGED = True

    @classmethod
    def setUpClass(cls) -> None:
        """Paged, batched and bucketed: the group path only exists in this configuration."""
        super().setUpClass()
        cls.model = support.ModelFixture.get(quant=cls.QUANT, max_seq=cls.MAX_SEQ,
                                             paged=True, bench=False)
        cls.pool = cls.model.enable_paging(num_blocks=NUM_BLOCKS, block_size=BLOCK)
        cls.model.enable_batching(BATCH_CAP)
        cls.model.capture_batch(BUCKETS)
        cls.prefill_row = cls.model.prefill_row

    @classmethod
    def tearDownClass(cls) -> None:
        """Return the model and its VRAM; the shared fixture outlives this class."""
        support.ModelFixture.close()

    def _alloc(self, n_tokens: int, extra: int = STEPS) -> list[int]:
        """Allocate blocks for ``n_tokens`` plus room for the decode steps that follow."""
        nb = -(-(n_tokens + extra) // BLOCK)
        return [self.pool.alloc() for _ in range(nb)]

    def _free(self, blocks: list[int]) -> None:
        """Return blocks to the pool; every ``_alloc`` has to be paired with this."""
        self.pool.free_blocks(blocks)

    def _solo(self, ids: list[int]):
        """One sequence end to end: solo prefill, then ``STEPS`` solo decode steps."""
        blocks = self._alloc(len(ids))
        try:
            self.model.reset()
            self.pool.assign_table(blocks, 0, row=self.prefill_row)
            logits = self.model.prefill(ids).float().clone()
            # Hand the prompt state to decode row 0 and decode through bucket 1
            # (which is bit-identical to the single-stream graph).
            self.model.zero_linear_states(0)
            self.model.move_linear_states(self.prefill_row, 0)
            tok = int(logits.argmax())
            chain = [tok]
            for s in range(STEPS):
                self.model.set_rows([tok], [len(ids) + s], [blocks], row_indices=[0])
                lg = self.model.decode_batch(1, 1).float().clone()
                tok = int(lg[0].argmax())
                chain.append(tok)
            return logits, chain
        finally:
            self._free(blocks)

    def _group(self, prompts: list[list[int]], snapshot_interval=None):
        """Group-prefill ``prompts`` on decode rows 0..k-1, then ``STEPS`` decode."""
        k = len(prompts)
        blocks = [self._alloc(len(p)) for p in prompts]
        try:
            rows = list(range(k))
            seqs = [SimpleNamespace(token_ids=p, row=r, block_table=b)
                    for p, r, b in zip(prompts, rows, blocks)]
            logits_list, states = self.model.prefill_group(
                seqs, snapshot_interval=snapshot_interval)
            logits = [lg.float().clone() for lg in logits_list]
            bucket = next(b for b in BUCKETS if b >= k)
            chain = [[int(lg.argmax())] for lg in logits]
            for s in range(STEPS):
                self.model.set_rows(
                    [c[-1] for c in chain],
                    [len(p) + s for p in prompts],
                    blocks, row_indices=rows)
                lgs = self.model.decode_batch(k, bucket).float()
                for i, row in enumerate(rows):
                    chain[i].append(int(lgs[row].argmax()))
            return logits, chain, states
        finally:
            for blk in blocks:
                self._free(blk)

    def _assert_close(self, got: torch.Tensor, ref: torch.Tensor, label: str,
                      max_ulps: float = 16.0):
        """Assert argmax equality plus a ULP bound, so a tolerance failure names which one broke."""
        self.assertEqual(tuple(got.shape), tuple(ref.shape), label)
        diff = float((got - ref).abs().max())
        unit = max(float(ref.abs().max()), 1e-6) * 2.0 ** -8    # one bf16 ULP
        self.assertEqual(int(got.argmax()), int(ref.argmax()),
                         f"{label}: argmax changed (max abs diff {diff:.4f})")
        self.assertLessEqual(
            diff, max_ulps * unit,
            f"{label}: max abs diff {diff:.4f} ({diff / unit:.1f} bf16 ULP, "
            f"{int((got != ref).sum())} of {got.numel()} differ)")

    def test_group_prefill_matches_solo(self):
        """Every group size must reproduce the solo prefill and the next STEPS decoded tokens."""
        lengths = [37, 53, 71, 44, 89, 64, 102, 47]
        for k in (1, 2, 4, 8):
            with self.subTest(k=k):
                prompts = [_tokens(lengths[i], off=5 * i) for i in range(k)]
                refs = [self._solo(p) for p in prompts]
                logits, chains, _states = self._group(prompts)
                for i in range(k):
                    ref_logits, ref_chain = refs[i]
                    self._assert_close(logits[i], ref_logits, f"k={k} seq={i} prefill")
                    self.assertEqual(chains[i], ref_chain,
                                     f"k={k} seq={i}: decode tokens diverged")

    def test_same_prompt_grouped(self):
        """Identical prompts must give identical rows: a row-indexing bug shows up as two rows disagreeing."""
        ids = _tokens(96)
        ref_logits, ref_chain = self._solo(ids)
        logits, chains, _states = self._group([list(ids) for _ in range(DECODE_CAP)])
        for row in range(DECODE_CAP):
            self._assert_close(logits[row], ref_logits, f"row={row} prefill")
            self.assertEqual(chains[row], ref_chain, f"row={row}: decode tokens diverged")

    def test_single_sequence_group_is_bitwise_prefill(self):
        """With one sequence the GEMM shape is unchanged, so this case can and must be bit-identical."""
        ids = _tokens(88, off=3)
        blocks = self._alloc(len(ids))
        try:
            self.model.reset()
            self.pool.assign_table(blocks, 0, row=self.prefill_row)
            ref = self.model.prefill(ids).float().clone()

            self.model.zero_linear_states(0)
            seq = [SimpleNamespace(token_ids=list(ids), row=0, block_table=blocks)]
            got = self.model.prefill_group(seq)[0][0].float().clone()
            self.assertEqual(int((got != ref).sum()), 0,
                             f"1-sequence group prefill differs from prefill: "
                             f"max abs diff {float((got - ref).abs().max()):.4e}")
        finally:
            self._free(blocks)


    def test_prefix_snapshots_and_matched_suffix(self):
        """Phase 2b shape: snapshots come out of the group run, a later (matched) suffix reuses them."""
        prefix = _tokens(192, off=11)
        ids_a = prefix + _tokens(16, off=1)
        ids_b = prefix + _tokens(24, off=2)
        ref_a = self._solo(ids_a)
        ref_b = self._solo(ids_b)
        blocks_a = self._alloc(len(ids_a))
        blocks_b = self._alloc(len(ids_b))
        try:
            seqs = [SimpleNamespace(token_ids=ids_a, row=0, block_table=blocks_a),
                    SimpleNamespace(token_ids=ids_b, row=1, block_table=blocks_b)]
            logits, states = self.model.prefill_group(seqs, snapshot_interval=BLOCK)
            self.assertTrue(states[0], "group prefill produced no snapshots")
            self.assertIn(len(prefix), states[0])
            for pos in states[0]:
                self.assertEqual(pos % BLOCK, 0,
                                 f"snapshot at non-block boundary {pos}")
            self._assert_close(logits[0].float(), ref_a[0], "prefix seq=0")
            self._assert_close(logits[1].float(), ref_b[0], "prefix seq=1")
            chains = [[int(lg.argmax())] for lg in logits]
            for s in range(STEPS):
                self.model.set_rows(
                    [c[-1] for c in chains],
                    [len(ids_a) + s, len(ids_b) + s],
                    [blocks_a, blocks_b], row_indices=[0, 1])
                lgs = self.model.decode_batch(2, 2).float()
                for row in (0, 1):
                    chains[row].append(int(lgs[row].argmax()))
            self.assertEqual(chains[0], ref_a[1], "prefix seq=0 decode diverged")
            self.assertEqual(chains[1], ref_b[1], "prefix seq=1 decode diverged")

            # A third sequence shares the first three blocks and the snapshot they
            # produced: prefilling only its tail (positions 192..) must reproduce
            # the solo run over the whole prompt.
            ids_c = prefix + _tokens(20, off=3)
            ref_logits_c, ref_chain_c = self._solo(ids_c)
            tail_blocks = self._alloc(len(ids_c) - len(prefix))
            blocks_c = blocks_a[:len(prefix) // BLOCK] + tail_blocks
            try:
                self.model.restore_linear_states(states[0][len(prefix)], row=2)
                seq_c = [SimpleNamespace(token_ids=ids_c, row=2,
                                         block_table=blocks_c, matched=len(prefix))]
                logits_c, states_c = self.model.prefill_group(
                    seq_c, snapshot_interval=BLOCK)
                self._assert_close(logits_c[0].float(), ref_logits_c,
                                   "matched suffix", max_ulps=64.0)
                chain = [int(logits_c[0].argmax())]
                for s in range(STEPS):
                    self.model.set_rows([chain[-1]], [len(ids_c) + s],
                                        [blocks_c], row_indices=[2])
                    lg = self.model.decode_batch(3, 4, pad_rows=[0, 1]).float()
                    chain.append(int(lg[2].argmax()))
                self.assertEqual(chain, ref_chain_c,
                                 "matched suffix decode diverged")
            finally:
                self._free(tail_blocks)
        finally:
            self._free(blocks_a)
            self._free(blocks_b)


if __name__ == "__main__":
    unittest.main()
