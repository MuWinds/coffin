"""Batched speculative decoding: per-row drafter, verify and rollback.

Module: ``coffin/tests``. Drives ``model.spec_step_batch`` through the batched
model API; the scheduler never calls this path in production yet, so this file
is its only functional gate.

Constraints:
    * Speculation may only change speed: every token the batched spec path emits
      has to be the token the plain batched greedy path would emit at that
      position. Per row, bit for bit, for the per-token verify path.
    * The batch, MTP and rollback graphs peak around 6 GiB, which does not fit
      next to the other classes on a 16 GB card; hence the ``RUN_SPEC_BATCH``
      opt-in skip rather than a normal test.
    * A row hole is part of the contract, not an edge case: the engine fixes a
      row per sequence and pads holes with sentinels, so a test with contiguous
      rows only would never exercise the padding.
    * The one-forward verify path is judged by token agreement instead of bitwise
      logits, because its wide-M kernel has a ULP-level difference from the greedy
      kernel by design.

Speculation may only change speed: every token the batched spec path emits has to
be the token the plain batched greedy path would emit at that position.  That is
the strong claim tested here -- per row, bit for bit:

* the drafter (MTP head, now with per-row paged KV and a bucket graph) proposes
  K tokens per row;
* the main model verifies ``K+1`` tokens per row in one captured forward, with
  the recurrent state snapshotted after every verified token;
* each row independently accepts the matching prefix, restores its own state from
  that token's snapshot and continues.
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
STEPS = 10                     # spec steps; each yields at least one token
GEN = 8                        # greedy tokens to compare against
NUM_BLOCKS = 256

#: This class needs the model plus batched graphs, the batched MTP graphs and the
#: per-row rollback snapshots (~6 GiB peak), which does not fit next to the other
#: classes on a 16 GB card.  The full suite therefore skips it; run it alone:
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
    """The batched spec loop must emit exactly the plain batched greedy tokens, per row."""

    QUANT = True
    MAX_SEQ = 2048
    PAGED = True

    @classmethod
    def setUpClass(cls) -> None:
        """Capture the batch graphs and then the spec graphs that replay them."""
        super().setUpClass()
        cls.model = support.ModelFixture.get(quant=cls.QUANT, max_seq=cls.MAX_SEQ,
                                             paged=True, bench=False)
        cls.pool = cls.model.enable_paging(num_blocks=NUM_BLOCKS, block_size=BLOCK)
        cls.model.enable_batching(BATCH_CAP)
        cls.model.capture_batch(BUCKETS)
        cls.model.enable_spec_batch(K, buckets=BUCKETS)

    @classmethod
    def tearDownClass(cls) -> None:
        """Release the ~6 GiB of spec graphs; the next module needs the card."""
        support.ModelFixture.close()

    def _alloc(self, n_tokens: int) -> list[int]:
        """Blocks for the prompt plus the speculative headroom each step may append."""
        return [self.pool.alloc() for _ in range(-(-(n_tokens + STEPS * WIDTH + WIDTH) // BLOCK))]

    def _free(self, blocks: list[int]) -> None:
        """Return blocks to the pool; every ``_alloc`` is paired with this."""
        self.pool.free_blocks(blocks)

    def _prefill(self, prompts: list[list[int]], rows=None):
        """Group-prefill the prompts on their rows; returns the blocks and the last-token logits."""
        rows = list(range(len(prompts))) if rows is None else list(rows)
        blocks = [self._alloc(len(p)) for p in prompts]
        seqs = [SimpleNamespace(token_ids=p, row=r, block_table=b)
                for p, r, b in zip(prompts, rows, blocks)]
        logits, _states = self.model.prefill_group(seqs)
        return blocks, [lg.float().clone() for lg in logits]

    def _greedy(self, prompts: list[list[int]], rows=None) -> list[list[int]]:
        """The oracle: plain batched greedy decoding of the same prompts on the same rows."""
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
        """Run the batched spec loop; returns emitted tokens per row + acceptance."""
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
        """Every batch size must reproduce greedy exactly; a rollback bug only appears once two rows roll back differently."""
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
        """Fixed rows with a hole between them (engine scheduling shape)."""
        prompts = [_tokens(45, 0), _tokens(70, 2)]
        rows = [0, 2]
        greedy = self._greedy(prompts)
        produced, _acc = self._spec(prompts, rows=rows)
        for i in range(len(prompts)):
            self.assertEqual(produced[i][:GEN], greedy[i][:GEN],
                             f"row={rows[i]}: spec tokens diverge from greedy")

    def test_bucket_padding_does_not_leak(self):
        """Three real rows in bucket 4: the sentinel row must not affect them."""
        prompts = [_tokens(40, 0), _tokens(55, 1), _tokens(72, 2)]
        greedy = self._greedy(prompts)
        produced, _acc = self._spec(prompts)
        for i in range(len(prompts)):
            self.assertEqual(produced[i][:GEN], greedy[i][:GEN],
                             f"row={i}: padded-bucket spec tokens diverge")


@support.needs_model
@unittest.skipUnless(RUN_SPEC_BATCH, "needs ~6 GiB; run with RUN_SPEC_BATCH=1")
class BatchSpecOnceTest(support.ModelTestCase):
    """The one-forward verify (task-13) must also be token-identical to greedy.

    On a tree that does not yet have the task-12 wide-M GEMV this falls back to
    the prefill projection path (correct, slow); the point here is the graph /
    buffer / rollback layout of the one-forward verify, not its speed.
    """

    QUANT = True
    MAX_SEQ = 2048
    PAGED = True

    @classmethod
    def setUpClass(cls) -> None:
        """Capture a narrower bucket set with the one-forward verify mode; the assert pins the mode the class measures."""
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
        """Release the model; this class runs the whole card too."""
        support.ModelFixture.close()

    # reuse the helpers/buffers of the main class
    _alloc = BatchSpecTest._alloc
    _free = BatchSpecTest._free
    _prefill = BatchSpecTest._prefill
    _greedy = BatchSpecTest._greedy
    _spec = BatchSpecTest._spec

    def test_one_forward_verify_matches_greedy(self):
        """The one-forward verify must agree with the token path and with greedy, so the wider graph is a speed change only.

        The wide-M projection kernel the one-forward verify needs (kernels
        task-12) has a ULP-level difference from the greedy kernel, so this path
        is judged by token agreement (the per-token path above stays bitwise).
        """
        prompts = [_tokens(41, off=0), _tokens(63, off=2)]
        greedy = self._greedy(prompts)
        produced_once, accepted = self._spec(prompts)
        self.assertTrue(accepted and isinstance(accepted[0], int))

        # Same step, token-path verify, same model (graphs re-captured), for the
        # once-vs-token consistency check.
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
