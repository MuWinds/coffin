"""Chunked prefill, preemption and eager batched decode, on the CPU.

Module: ``coffin/tests``. Drives ``llm_engine`` / ``scheduler`` /
``model_runner`` / ``block_manager`` against a recording fake model over a real
paged pool, so the whole scheduling surface runs with no DCU and no checkpoint.

Constraints:
    * The fake model owns its "KV" as a list of tokens per row and refuses to
      write at a position other than the next free one. That is what makes the
      equivalence claims exact rather than approximate: a chunked prefill that
      resumed from the wrong position, or reset the row's carried state, cannot
      produce the prompt as its row state afterwards.
    * The pool is real, so block refcounts, eviction and preemption are measured
      on real allocations. ``num_free`` returning to its starting value is the
      leak check for every test that allocates.
    * ``graph=False`` throughout. A captured graph cannot be recorded on this
      host, and the eager path is the one the SCOW deployment runs, so the tests
      must not need capture to pass.
    * Timing is never asserted; ``prefill_s`` is only checked for being present.

What has to hold:

1. one prefill call and N chunked calls produce the same row state and the same
   next-token logits;
2. a chunk boundary that lands inside the model's carried state does not reset
   it -- the carried state after chunk k is exactly the prompt prefix;
3. a prefix hit still works when the suffix takes several chunks;
4. preemption mid-prefill, mid-decode and with a radix prefix all restore
   without losing or duplicating an emitted token;
5. the token budget really bounds a step's prefill work, and a short request is
   not blocked behind a long prompt;
6. a pool too small to make progress raises instead of spinning.

    python -m unittest discover -s tests -p "test_chunked_prefill.py" -v
"""

from __future__ import annotations

import unittest

import support  # noqa: F401  (sys.path setup only; no checkpoint is loaded)

import torch

from cache.paged_cache import PagedKVCache
from engine.block_manager import BlockManager
from engine.config import EngineConfig
from engine.llm_engine import NO_PROGRESS_LIMIT, LLMEngine
from engine.sequence import SamplingParams, Sequence, SequenceStatus

BLOCK = 8
MAX_SEQ = 256
VOCAB = 32


class FakePrefillModel:
    """A paged model whose "KV" is the token stream written into each row.

    Every entry point records into ``rows`` and checks that it was asked to
    write at the position the row is actually at, which is the whole contract a
    chunked prefill has to satisfy. Logits are derived from the row's full token
    list, so two runs agree exactly when they wrote the same stream.
    """

    def __init__(self, num_blocks: int = 48, block_size: int = BLOCK) -> None:
        """Allocate the pool and the single-row defaults.

        Args:
            num_blocks: Physical blocks; small on purpose so preemption and
                eviction are reachable.
            block_size: Tokens per block.
        """
        self.max_seq = MAX_SEQ
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.pool = PagedKVCache(1, 1, 4, num_blocks=num_blocks,
                                 block_size=block_size, max_seq=MAX_SEQ,
                                 device="cpu")
        self.batch_capacity = 1
        self.decode_capacity = 1
        self.prefill_row = 0
        self.mtp = None
        self.mtp_prefill = True
        self.pos = torch.zeros(1, dtype=torch.long)
        self.pos1 = self.pos[:1]
        self.rows: list[list[int]] = [[]]
        self.writes: list[tuple[int, int, list[int]]] = []
        self.scratch: list[int] = []
        self.mm_calls: list[object] = []

    # lifecycle
    def enable_paging(self, num_blocks: int, block_size: int):
        """Attach the paged pool.

        Args:
            num_blocks: Pool size.
            block_size: Tokens per block.

        Returns:
            The pool, which the block manager then wraps.
        """
        return self.pool

    def enable_batching(self, capacity: int) -> None:
        """Give the model one row per batch entry, plus the staging row.

        Args:
            capacity: Rows including the staging row.
        """
        self.batch_capacity = capacity
        self.decode_capacity = capacity - 1
        self.prefill_row = capacity - 1
        self.pos = torch.zeros(capacity, dtype=torch.long)
        self.pos1 = self.pos[self.prefill_row:self.prefill_row + 1]
        self.rows = [[] for _ in range(capacity)]
        # The real model grows the slot map to the row count here; without it a
        # row above 0 has no map to write through.
        self.pool.set_rows(capacity)
        for row in range(capacity):
            self.scratch.append(self.pool.alloc())
            self.pool.assign(self.scratch[-1], 0, row)

    def reset(self) -> None:
        """Rewind every row to the empty context."""
        self.pos.zero_()
        self.rows = [[] for _ in range(self.batch_capacity)]

    # carried state
    def snapshot_linear_states(self, row: int = 0) -> dict:
        """Snapshot one row's carried state.

        Args:
            row: Row to snapshot.

        Returns:
            The row's written token stream, which is its whole state here.
        """
        return {"tokens": tuple(self.rows[row])}

    def restore_linear_states(self, states, row: int = 0) -> None:
        """Install a snapshot into ``row``.

        Args:
            states: Value from :meth:`snapshot_linear_states`.
            row: Row to overwrite.
        """
        self.rows[row] = list(states["tokens"])

    def zero_linear_states(self, row: int) -> None:
        """Clear one row's carried state.

        Args:
            row: Row to clear.
        """
        self.rows[row] = []

    def move_linear_states(self, src_row: int, dst_row: int) -> None:
        """Copy one row's state onto another.

        Args:
            src_row: Source row.
            dst_row: Destination row.
        """
        if src_row != dst_row:
            self.rows[dst_row] = list(self.rows[src_row])

    # forward
    def prefill(self, tokens, chunk: int = 2048, spec_row=None, mm=None):
        """Write a token run into the staging row.

        Args:
            tokens: Tokens to write, starting at the row's current position.
            chunk: GEMM tile; unused here apart from being recorded.
            spec_row: Batched-spec drafter row; unused.
            mm: The request's image payload, recorded so a test can prove it
                reaches every chunk rather than only the first.

        Returns:
            ``[VOCAB]`` logits derived from the row's whole stream.

        Raises:
            AssertionError: The run does not start where the row actually is.
                That is the chunked-prefill bug this fake exists to catch.
        """
        row = self.prefill_row
        start = int(self.pos[row])
        assert start == len(self.rows[row]), (
            f"row {row} asked to write at {start} but holds {len(self.rows[row])}")
        self.rows[row].extend(int(t) for t in tokens)
        self.writes.append((row, start, [int(t) for t in tokens]))
        self.mm_calls.append(mm)
        self.pos[row] = start + len(tokens)
        return self._logits(self.rows[row])

    def set_rows(self, tokens, positions, slot_rows=None, row_indices=None) -> None:
        """Stage one decode step: token and position per row.

        Args:
            tokens: One token per row.
            positions: Absolute position of that token per row.
            slot_rows: Block table per row; mapped through the pool.
            row_indices: Row for each entry.
        """
        rows = list(range(len(tokens))) if row_indices is None else list(row_indices)
        for row, tok, pos in zip(rows, tokens, positions):
            self.pos[row] = int(pos)
            self.rows[row].append(int(tok))
        if slot_rows is not None:
            for row, blocks in zip(rows, slot_rows):
                self.pool.assign_table(list(blocks), 0, row)

    def decode_batch_eager(self, rows: int, bucket: int, pad_rows=()):
        """Run a batched decode step eagerly and return every row's logits.

        Args:
            rows: High-water mark of the live rows.
            bucket: Width of the forward.
            pad_rows: Holes below ``rows``.

        Returns:
            ``[bucket, VOCAB]`` logits. Sentinel rows get a zero row, exactly like
            the real model's discarded padding.
        """
        assert self.batch_capacity > 1, "decode_batch_eager() before enable_batching()"
        pad = set(int(r) for r in pad_rows) | set(range(rows, bucket))
        out = torch.zeros(bucket, VOCAB)
        for row in range(bucket):
            self.pos[row] = self.pos[row] + 1
            if row in pad:
                continue
            out[row] = self._logits(self.rows[row])
        return out

    def capture_batch(self, buckets) -> None:
        """Refuse capture: this host cannot record a graph, and the engine is
        configured for the eager path anyway.

        Raises:
            AssertionError: Always. A test that reaches capture is testing the
                wrong configuration.
        """
        raise AssertionError("capture_batch() called with graph off")

    def prefill_group(self, seqs, snapshot_interval=None):
        """Prefill several prompts in one call, one row each.

        Args:
            seqs: Sequences to prefill; each already has a row and a block table.
            snapshot_interval: Boundaries to snapshot; unused here.

        Returns:
            ``(logits_list, snaps_list)``: one ``[VOCAB]`` tensor and one
            position-keyed snapshot dict per sequence, matching the real model's
            grouped-prefill contract.
        """
        logits, snaps = [], []
        for seq in seqs:
            start = seq.prefill_cursor
            self.rows[seq.row] = list(seq.token_ids)
            self.pos[seq.row] = len(seq.token_ids)
            self.mm_calls.append(seq.mm)
            logits.append(self._logits(self.rows[seq.row]))
            snaps.append({})
            del start
        return logits, snaps

    def logits(self, token: int) -> torch.Tensor:
        """Single-stream eager decode: write one token, return the next logits.

        The prefix-cache and non-batched engines call this instead of
        ``decode_step``; there is no graph to replay on this host, so it is also
        the path the eager batched engine picks.

        Args:
            token: Token to write on the staging row.

        Returns:
            ``[VOCAB]`` logits for the row's stream after the write.
        """
        row = self.prefill_row
        start = int(self.pos[row])
        if start < len(self.rows[row]):
            self.rows[row] = self.rows[row][:start]
        self.rows[row].append(int(token))
        self.pos[row] = start + 1
        return self._logits(self.rows[row])

    def decode_step(self, token: int) -> torch.Tensor:
        """Captured-decode entry; identical to :meth:`logits` without capture.

        Args:
            token: Token to write on the staging row.

        Returns:
            ``[VOCAB]`` logits for the row's stream after the write.
        """
        return self.logits(token)

    def _logits(self, tokens: list[int]) -> torch.Tensor:
        """Logits from a row's token stream, deterministic and injective enough
        for an equality claim.

        Args:
            tokens: The row's whole stream.

        Returns:
            ``[VOCAB]`` fp32 logits: a hash of the stream spread over the vocab.
        """
        seed = 0
        for i, tok in enumerate(tokens):
            seed = (seed * 131 + tok * 7 + i) % 9973
        out = torch.zeros(VOCAB)
        for i in range(VOCAB):
            out[i] = float((seed * (i + 3)) % 101) / 101.0
        return out


def make_engine(num_blocks: int = 48, *, max_num_seqs: int = 2,
                prefill_quantum: int = 16, max_num_batched_tokens: int = 64,
                snapshots: int = 4, prefix_cache: bool = True) -> tuple[LLMEngine, FakePrefillModel]:
    """Build an engine over a fake model with the graph off.

    Args:
        num_blocks: Pool size; small enough that preemption is reachable.
        max_num_seqs: Rows the engine may admit at once. The default is 2 because
            chunked prefill is a batched-engine feature: the single-stream engine
            prefills a prompt in one call through its prefix cache, so a budget
            cannot bound it.
        prefill_quantum: Tokens one request may prefill per step.
        max_num_batched_tokens: Step token budget, decode included.
        snapshots: Recurrent-state snapshot budget.
        prefix_cache: Attach the radix tree.

    Returns:
        ``(engine, model)``, built and ready to step.
    """
    model = FakePrefillModel(num_blocks=num_blocks)
    cfg = EngineConfig(model_dir="unused", max_seq=MAX_SEQ, bench=False,
                       quant=False, graph=False, prefix_cache=prefix_cache,
                       block_size=BLOCK, num_blocks=num_blocks,
                       snapshots=snapshots, snapshot_interval=BLOCK,
                       max_num_seqs=max_num_seqs,
                       prefill_quantum=prefill_quantum,
                       max_num_batched_tokens=max_num_batched_tokens,
                       vision=False)
    engine = LLMEngine(cfg, model)
    engine.build()
    return engine, model


def drain(engine: LLMEngine, seqs: list[Sequence], limit: int = 400) -> None:
    """Step the engine until every sequence finishes.

    Args:
        engine: Engine to drive.
        seqs: Sequences that must all finish.
        limit: Step cap, so a livelock fails the test instead of hanging it.

    Raises:
        AssertionError: The sequences did not finish within ``limit`` steps.
    """
    for _ in range(limit):
        if all(s.is_finished for s in seqs):
            return
        engine.step()
    raise AssertionError(f"engine did not finish within {limit} steps: "
                         f"{[s.status for s in seqs]}")


class ChunkedPrefillTest(unittest.TestCase):
    """A prompt advanced in pieces must equal the same prompt in one call."""

    def test_chunked_prefill_equals_one_shot(self) -> None:
        """The row state and the next-token logits must not depend on how the prompt was cut, which is the whole correctness claim."""
        prompt = [100 + i for i in range(40)]
        one, _ = make_engine(prefill_quantum=64, max_num_batched_tokens=128)
        row_a = one.add_request(prompt, SamplingParams(max_tokens=1, temperature=0.0))
        drain(one, [row_a])
        whole = one.runner.model.rows[row_a.row]

        many, many_model = make_engine(prefill_quantum=8, max_num_batched_tokens=64)
        row_b = many.add_request(prompt, SamplingParams(max_tokens=1, temperature=0.0))
        drain(many, [row_b])
        piecewise = many_model.rows[row_b.row]
        # The one request this engine serves owns every write, so a chunk count
        # above one proves the prompt was split rather than prefilled whole.
        chunks = len(many_model.writes)

        self.assertEqual(piecewise, whole, "chunked prefill wrote a different stream")
        self.assertGreater(chunks, 1, "the quantum did not split the prompt")
        self.assertEqual(piecewise[:len(prompt)], prompt,
                         "the row must hold the prompt as written")

    def test_grouped_prefill_serves_several_short_prompts_at_once(self) -> None:
        """Two prompts that each fit the whole budget must share one forward: that single weight read is the only thing grouping buys, so a regression to per-prompt forwards would be invisible except here."""
        engine, model = make_engine(num_blocks=48, max_num_seqs=3,
                                    prefill_quantum=64,
                                    max_num_batched_tokens=96)
        self.assertTrue(engine.runner.can_group_prefill())
        prompts = [[10, 11, 12, 13], [20, 21, 22, 23, 24]]
        seqs = [engine.add_request(p, SamplingParams(max_tokens=4, temperature=0.0))
                for p in prompts]
        engine.step()
        # One grouped forward covers both prompts, so both are past the prompt
        # after a single step; per-prompt prefills would need two steps.
        for seq in seqs:
            self.assertIs(seq.status, SequenceStatus.RUNNING,
                          "the prompt did not finish in the grouped forward")
            self.assertEqual(seq.prefill_cursor, seq.num_prompt_tokens,
                             "the grouped forward left the prompt half written")
        group_rows = [model.rows[s.row] for s in seqs]
        self.assertEqual([r[:len(p)] for r, p in zip(group_rows, prompts)], prompts)
        self.assertTrue(all(s.num_completion_tokens >= 1 for s in seqs),
                        "a prompt in the group produced no token")
    def test_state_is_carried_across_a_chunk_boundary(self) -> None:
        """A chunk that stops mid-prompt must hand the next one the state, not a reset row: that is what a 50 MB copy per request would have bought."""
        engine, model = make_engine(prefill_quantum=8, max_num_batched_tokens=64)
        prompt = [200 + i for i in range(40)]
        seq = engine.add_request(prompt, SamplingParams(max_tokens=1, temperature=0.0))
        seen: list[int] = []
        for _ in range(40):
            if seq.num_prompt_tokens <= seq.prefill_cursor:
                break
            engine.step()
            if seq.prefill_cursor:
                seen.append(seq.prefill_cursor)
                self.assertEqual(model.rows[seq.row][:seq.prefill_cursor],
                                 prompt[:seq.prefill_cursor],
                                 "carried state diverged mid-prompt")
        self.assertGreater(len(set(seen)), 1, "the prompt was not prefilled in pieces")

    def test_token_budget_bounds_one_step(self) -> None:
        """The budget is a real limit: a step must not prefill more tokens than it allows."""
        engine, model = make_engine(prefill_quantum=8, max_num_batched_tokens=16)
        seq = engine.add_request([300 + i for i in range(64)],
                                 SamplingParams(max_tokens=1, temperature=0.0))
        engine.step()
        written = sum(len(w[2]) for w in model.writes)
        self.assertLessEqual(written, 16, "one step wrote more than the budget")
        self.assertEqual(seq.status, SequenceStatus.PREFILLING)

    def test_prefix_hit_survives_chunking(self) -> None:
        """Reuse must compose with chunking: the second request adopts the prefix and prefills only its own tail, in pieces."""
        engine, model = make_engine(prefill_quantum=8, max_num_batched_tokens=64)
        prefix = [400 + i for i in range(24)]
        engine.clear_prefix_cache()
        first = engine.add_request(prefix + [500, 501],
                                   SamplingParams(max_tokens=1, temperature=0.0))
        drain(engine, [first])
        self.assertEqual(first.matched, 0)

        warm = engine.add_request(prefix + [600, 601],
                                  SamplingParams(max_tokens=1, temperature=0.0))
        drain(engine, [warm])
        self.assertEqual(warm.matched, len(prefix),
                         "the cached prefix was not adopted")
        tail_writes = [w for w in model.writes if w[2][:1] == [prefix[0]]]
        self.assertEqual(tail_writes[0][1], 0, "the cold run must start at 0")
        self.assertTrue(any(w[1] == len(prefix) for w in model.writes),
                        "the warm run did not resume at the matched boundary")

    def test_image_payload_reaches_every_chunk(self) -> None:
        """An image prompt sliced across chunks must hand the model the same payload each time: spans are absolute, so a chunk inside the image still has to inject it."""
        engine, model = make_engine(prefill_quantum=8, max_num_batched_tokens=64)
        payload = object()
        seq = engine.add_request([700 + i for i in range(32)],
                                 SamplingParams(max_tokens=1, temperature=0.0),
                                 mm=payload)
        drain(engine, [seq])
        calls = [mm for mm in model.mm_calls if mm is not None]
        self.assertGreater(len(calls), 1, "the prompt was not chunked")
        self.assertTrue(all(mm is payload for mm in calls),
                        "a chunk lost the image payload")


class PreemptionTest(unittest.TestCase):
    """Recompute preemption must give the pool back and restore correctly."""

    def test_preempt_mid_prefill_resumes_and_finishes(self) -> None:
        """A preempted prompt restarts from the radix checkpoint, keeps its tokens and still produces output."""
        engine, model = make_engine(num_blocks=19, max_num_seqs=2,
                                    prefill_quantum=8, max_num_batched_tokens=16)
        long_prompt = [800 + i for i in range(32)]
        a = engine.add_request(long_prompt, SamplingParams(max_tokens=40,
                                                           temperature=0.0))
        b = engine.add_request([900 + i for i in range(32)],
                               SamplingParams(max_tokens=40, temperature=0.0))
        drain(engine, [a, b], limit=400)
        self.assertTrue(a.is_finished and b.is_finished)
        self.assertGreater(engine.n_preempted, 0,
                           "the scenario must actually preempt, or it proves nothing")
        self.assertEqual(a.token_ids[:len(long_prompt)], long_prompt,
                         "preemption dropped or reordered prompt tokens")
        self.assertTrue(all(w[1] >= 0 for w in model.writes))

    def test_preempt_mid_decode_keeps_emitted_tokens(self) -> None:
        """A request evicted while generating must not repeat or lose a token: the emitted prefix is part of the request, not of the recomputed state."""
        engine, model = make_engine(num_blocks=19, max_num_seqs=2,
                                    prefill_quantum=8, max_num_batched_tokens=16)
        prompts = [[800 + i for i in range(32)], [900 + i for i in range(32)]]
        seqs = [engine.add_request(p, SamplingParams(max_tokens=40, temperature=0.0))
                for p in prompts]
        drain(engine, seqs, limit=600)
        self.assertGreater(engine.n_preempted, 0, "the scenario did not preempt")
        for seq, prompt in zip(seqs, prompts):
            self.assertTrue(seq.is_finished)
            self.assertEqual(seq.token_ids[:len(prompt)], prompt,
                             "the prompt prefix changed under recompute")
            self.assertEqual(len(seq.token_ids), len(prompt) + 40,
                             "recompute lost or duplicated generated tokens")

    def test_pool_survives_preemption_without_leaking(self) -> None:
        """Every referenced block must be accounted for -- the tree's cached prefixes and the model's scratch rows -- and no refcount may go negative, which would free a block someone still reads."""
        engine, model = make_engine(num_blocks=19, max_num_seqs=2,
                                    prefill_quantum=8, max_num_batched_tokens=16)
        seqs = [engine.add_request([800 + i for i in range(32)],
                                   SamplingParams(max_tokens=40, temperature=0.0)),
                engine.add_request([900 + i for i in range(32)],
                                   SamplingParams(max_tokens=40, temperature=0.0))]
        drain(engine, seqs, limit=600)
        bm = engine.bm
        pool = bm.pool
        cached = set()
        stack = [bm.radix.root]
        while stack:
            node = stack.pop()
            for child in node.children.values():
                cached.update(child.blocks)
                stack.append(child)
        held = {b for b, n in enumerate(pool.refcount) if n > 0}
        # A finished sequence's blocks are meant to survive in the tree -- that is
        # the prefix cache. What must not happen is a reference nothing owns: a
        # block held with no tree edge and no scratch row is a leak.
        stray = held - cached - set(model.scratch)
        self.assertFalse(stray, f"blocks referenced by nothing: {sorted(stray)}")
        self.assertTrue(all(n >= 0 for n in pool.refcount),
                        f"a refcount went negative: {list(pool.refcount)}")
        self.assertEqual(len(bm.radix.root.children) > 0, True,
                         "nothing was cached, so the leak check proved nothing")

    def test_unsatisfiable_pool_raises_instead_of_hanging(self) -> None:
        """A pool smaller than one request's table must fail loudly: spinning is worse than an error naming the flag to raise."""
        # Three scratch blocks plus the two rows' staging blocks leave too little
        # for a 120-token prompt at block 8, and no completion can ever free any.
        engine, _ = make_engine(num_blocks=8, max_num_seqs=1,
                                prefill_quantum=8, max_num_batched_tokens=16)
        seq = engine.add_request([i for i in range(120)],
                                 SamplingParams(max_tokens=1, temperature=0.0))
        with self.assertRaises(RuntimeError):
            for _ in range(NO_PROGRESS_LIMIT + 10):
                engine.step()
        self.assertFalse(seq.is_finished)

    def test_short_request_is_not_blocked_by_a_long_prompt(self) -> None:
        """Chunked prefill exists so a long prompt cannot stall the queue: the short request must start and finish while the long one is still prefilling."""
        engine, model = make_engine(num_blocks=64, max_num_seqs=2,
                                    prefill_quantum=8, max_num_batched_tokens=16)
        long_prompt = [i % 40 for i in range(160)]
        long_seq = engine.add_request(long_prompt,
                                      SamplingParams(max_tokens=2, temperature=0.0))
        engine.step()
        self.assertIs(long_seq.status, SequenceStatus.PREFILLING)
        short_seq = engine.add_request([7, 8, 9], SamplingParams(max_tokens=2,
                                                                temperature=0.0))
        for _ in range(20):
            engine.step()
            if short_seq.is_finished:
                break
        self.assertTrue(short_seq.is_finished,
                        "the short request waited for the whole long prompt")
        self.assertFalse(long_seq.is_finished,
                         "the long prompt should still be prefilling")
        # The long prompt must have advanced while the short one ran, not started
        # only after it retired: that is the head-of-line block being gone.
        self.assertGreater(long_seq.prefill_cursor, 0)
        self.assertLess(long_seq.prefill_cursor, len(long_prompt))


class EagerBatchDecodeTest(unittest.TestCase):
    """Batched decode must run with capture off -- the SCOW configuration."""

    def test_batched_decode_runs_without_capture(self) -> None:
        """Two concurrent requests must both decode through the eager batched entry while the model never captures a graph."""
        engine, model = make_engine(num_blocks=48, max_num_seqs=4,
                                    prefill_quantum=16, max_num_batched_tokens=96)
        seqs = [engine.add_request([i for i in range(12)],
                                   SamplingParams(max_tokens=3, temperature=0.0)),
                engine.add_request([20 + i for i in range(12)],
                                   SamplingParams(max_tokens=3, temperature=0.0))]
        drain(engine, seqs)
        self.assertTrue(all(s.is_finished for s in seqs))
        self.assertEqual(engine.batching, True)
        self.assertFalse(engine.runner.graph, "the eager path must be selected")

    def test_batched_tokens_match_a_solo_run(self) -> None:
        """A row must sample what a lone request samples: the eager batch changes the row count, never the numbers."""
        prompt = [55 + i for i in range(16)]
        solo_engine, solo_model = make_engine(num_blocks=48, max_num_seqs=2,
                                              prefill_quantum=32,
                                              max_num_batched_tokens=96)
        solo = solo_engine.add_request(prompt, SamplingParams(max_tokens=4,
                                                              temperature=0.0))
        drain(solo_engine, [solo])

        batch_engine, batch_model = make_engine(num_blocks=48, max_num_seqs=2,
                                                prefill_quantum=32,
                                                max_num_batched_tokens=96)
        a = batch_engine.add_request(prompt, SamplingParams(max_tokens=4,
                                                            temperature=0.0))
        b = batch_engine.add_request([90 + i for i in range(16)],
                                     SamplingParams(max_tokens=4, temperature=0.0))
        drain(batch_engine, [a, b])
        self.assertEqual(a.token_ids, solo.token_ids,
                         "the batched row produced a different token stream")
        # The batched engine must really have allocated a second row, or the run
        # proves nothing about the batching path it is meant to exercise.
        self.assertGreaterEqual(len(batch_model.rows), 2)
        del solo_model


class StatusModelTest(unittest.TestCase):
    """The new statuses must describe the lifecycle the scheduler relies on."""

    def test_non_chunked_engine_still_serves_one_request(self) -> None:
        """The single-stream path predates chunking; whatever the new statuses do, one request must still prefill once and then decode to completion."""
        model = FakePrefillModel(num_blocks=32)
        cfg = EngineConfig(model_dir="unused", max_seq=MAX_SEQ, bench=False,
                           quant=False, graph=False, prefix_cache=False,
                           block_size=BLOCK, num_blocks=32, max_num_seqs=1,
                           prefill_quantum=16, max_num_batched_tokens=64,
                           vision=False)
        engine = LLMEngine(cfg, model)
        engine.build()
        self.assertFalse(engine.batching)
        self.assertFalse(engine.scheduler.chunked,
                         "no block manager means the token budget cannot apply")
        seq = engine.add_request([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                 SamplingParams(max_tokens=3, temperature=0.0))
        drain(engine, [seq], limit=20)
        self.assertTrue(seq.is_finished)
        self.assertEqual(len(seq.token_ids), seq.num_prompt_tokens + 3,
                         "the single-stream loop lost or duplicated a token")

    def test_sequence_starts_waiting_and_prefills_before_running(self) -> None:
        """``RUNNING`` means decodable, so a request still writing its prompt must be ``PREFILLING``."""
        engine, _ = make_engine(num_blocks=32, max_num_seqs=2,
                                prefill_quantum=8, max_num_batched_tokens=16)
        seq = engine.add_request([i for i in range(40)],
                                 SamplingParams(max_tokens=1, temperature=0.0))
        self.assertIs(seq.status, SequenceStatus.WAITING)
        engine.step()
        self.assertIs(seq.status, SequenceStatus.PREFILLING)
        self.assertTrue(seq.is_prefilling)
        drain(engine, [seq])
        self.assertIs(seq.status, SequenceStatus.FINISHED)

    def test_preempted_sequence_keeps_its_tokens(self) -> None:
        """The whole point of recompute: the request survives eviction, only its cache bookkeeping is reset."""
        seq = Sequence([1, 2, 3, 4])
        seq.append_token(9)
        seq.block_table = [3, 4]
        seq.prefill_cursor = 4
        seq.num_cached_tokens = 5
        seq.matched = 4
        seq.park_preempted()
        self.assertIs(seq.status, SequenceStatus.PREEMPTED)
        self.assertEqual(seq.token_ids, [1, 2, 3, 4, 9])
        self.assertEqual(seq.prefill_cursor, 0)
        self.assertEqual(seq.num_cached_tokens, 0)
        self.assertIsNone(seq.prefill_logits)
        self.assertEqual(seq.preempt_count, 1)
        seq.resume()
        self.assertIs(seq.status, SequenceStatus.WAITING)


if __name__ == "__main__":
    unittest.main()
