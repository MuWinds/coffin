"""Drives one model forward for the scheduler.

Module: ``coffin/`` (middle layer), owned by ``llm_engine``. Keeping the runner
separate is what makes batching possible: the runner owns *how* a step is
executed (prefill path, paged write window, graph replay, sampling) and the
scheduler owns *which* sequences go into the step.

Two execution modes live here:

* single stream (``batching=False``) — the pre-refactor path: one sequence, one
  token, the decode graph captured for one slot map;
* static batch (``batching=True``) — up to ``max_num_batched`` sequences share
  one forward. The projections are batched (one weight read serves every row)
  while attention and the delta-net recurrence run the existing single-token
  kernels once per row.

Constraints:
    * Row layout is fixed: rows ``0..capacity-2`` are decode rows and row
      ``capacity-1`` is the prefill staging row. That reserve is why
      interleaving a prefill with running decodes cannot clobber a decode row's
      recurrent state.
    * A sequence keeps its row until it finishes. Rows may therefore have holes,
      and every batched step runs the smallest bucket above the highest live row
      with the holes padded by the model's sentinel row. Never compact rows: the
      saving is not worth permuting recurrent state every step.
    * ``seq.num_cached_tokens`` is refreshed from ``model.pos1`` after every
      forward, never incremented locally — a speculative step advances the model
      by more than one token.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .config import EngineConfig
    from .block_manager import BlockManager
    from .prefix_engine import PrefixCacheEngine
    from .sequence import Sequence


#: Batch sizes the model captures one graph per; a step runs the smallest bucket
#: that fits (the rest are sentinel rows whose logits are dropped). Powers of two
#: keep the padding waste under 100% at every bucket boundary.
BATCH_BUCKETS = (1, 2, 4, 8)


class ModelRunner:
    """Execution of one step of the engine's model."""

    def __init__(self, model, cfg: "EngineConfig") -> None:
        """Bind a runner to one loaded model.

        Args:
            model: A ``engine_api.Model`` / ``PagedModel`` implementation.
            cfg: Engine configuration; the prefill chunk size and the snapshot
                budget are read from it.
        """
        self.model = model
        self.cfg = cfg
        self.pc: "PrefixCacheEngine | None" = None
        self.bm: "BlockManager | None" = None
        self.batching = False
        self.buckets: tuple[int, ...] = ()
        #: Batch-friendly speculative decoding (MTP), off unless the engine asks.
        self.spec = False
        self.spec_k = 0
        self.spec_sparse_k = 0
        #: decode row -> seq_id. Rows are stable for a sequence's lifetime, which
        #: is what lets a batched step skip permuting recurrent state.
        self._row_owner: dict[int, int] = {}
        #: Set by the engine when a prefix cache is attached; used by ``ping``.
        self.last_prefill_s = 0.0

    def attach_prefix_cache(self, pc: "PrefixCacheEngine") -> None:
        """Drive prefills through a prefix-cache engine.

        Args:
            pc: The cache engine. It also supplies the block manager, so the
                two can never disagree about which blocks a sequence holds.
        """
        self.pc = pc
        self.bm = pc.bm

    def attach_blocks(self, bm: "BlockManager") -> None:
        """Attach a block manager without a prefix cache (batched path).

        Args:
            bm: Manager built with ``with_radix=False``; blocks are allocated
                per sequence and never shared.
        """
        self.bm = bm

    def release_row(self, seq: "Sequence") -> None:
        """Free the batch row a finished sequence held.

        Args:
            seq: Finished sequence. Its ``row`` is reset to ``-1`` so a late
                second call cannot free somebody else's row.
        """
        if seq.row >= 0:
            self._row_owner.pop(seq.row, None)
            seq.row = -1

    def _claim_row(self, seq: "Sequence") -> int:
        """Give ``seq`` the lowest free decode row for its whole lifetime.

        Args:
            seq: Sequence being admitted.

        Returns:
            The claimed row index.

        Raises:
            RuntimeError: Every row is taken. That means the scheduler admitted
                past ``max_num_seqs``, and running with a shared row would
                corrupt recurrent state rather than merely slow down.
        """
        capacity = int(getattr(self.model, "decode_capacity", 1))
        for row in range(capacity):
            if row not in self._row_owner:
                self._row_owner[row] = seq.seq_id
                seq.row = row
                return row
        raise RuntimeError(
            f"no free decode row (capacity {capacity}, running {len(self._row_owner)})")

    def enable_batching(self, max_batch: int,
                        buckets: tuple[int, ...] = BATCH_BUCKETS) -> None:
        """Capture one decode graph per bucket.

        Args:
            max_batch: Upper bound on decode rows. Buckets above it are dropped,
                because a graph larger than the row count can never run.
            buckets: Candidate bucket sizes. Always includes 1, so a single
                running sequence still has a graph to replay.
        """
        buckets = tuple(b for b in buckets if b <= max_batch) or (1,)
        self.buckets = buckets
        # +1 is the prefill staging row; without it a grouped prefill would have
        # to share a row with a sequence already decoding.
        self.model.enable_batching(max(self.buckets) + 1)
        self.model.capture_batch(self.buckets)
        self.batching = True

    def enable_spec(self, k: int, sparse_topk: int = 0) -> None:
        """Turn on batch-friendly MTP with ``k`` drafts per row.

        Args:
            k: Draft tokens per step.
            sparse_topk: Verify against only the top-k historical KV blocks per
                row (approximate). 0 keeps full attention, which is the exact
                mode; the approximation exists because the batched verify
                otherwise pays one weight pass per candidate token.

        Raises:
            RuntimeError: Batching is off. Single-stream spec has its own path
                (``spec_generate``) and does not go through this switch.
        """
        if not self.batching:
            raise RuntimeError("enable_spec() needs the batched engine")
        kwargs = {"sparse_topk": sparse_topk} if sparse_topk else {}
        self.model.enable_spec_batch(K=k, buckets=self.buckets, **kwargs)
        self.spec_k = int(k)
        self.spec_sparse_k = int(sparse_topk)
        self.spec = True

    @torch.inference_mode()
    def prefill(self, seq: "Sequence") -> torch.Tensor:
        """Feed the prompt (minus the cached prefix) and return next-token logits.

        Three paths exist because there are three engines, not because the call
        differs: a prefix cache, the batched runner, and the plain single-stream
        model. Each keeps its own invariants around the KV write window.

        Args:
            seq: Sequence to prefill. Its ``num_cached_tokens`` and
                ``num_scheduled_tokens`` are refreshed from the model afterwards.

        Returns:
            Logits for the last prompt position, ``[vocab]``.
        """
        t0 = time.time()
        if self.pc is not None:
            logits = self.pc.prefill(seq.token_ids, mm=seq.mm)
            seq.matched = self.pc.last_matched
            if self.pc.seq is not None:
                seq.block_table = list(self.pc.seq.blocks)
        elif self.batching:
            logits = self._prefill_batched(seq)
        else:
            self.model.reset()
            logits = self.model.prefill(seq.token_ids, mm=seq.mm)
        seq.num_cached_tokens = int(self.model.pos1)
        seq.num_scheduled_tokens = seq.num_cached_tokens - seq.matched
        self.last_prefill_s = time.time() - t0
        return logits

    def _prefill_batched(self, seq: "Sequence") -> torch.Tensor:
        """Prefill one sequence into its decode row, reusing a radix prefix.

        The sequence owns blocks from the moment it is admitted: the matched
        prefix blocks are refcounted by the radix tree (shared with whoever else
        holds them) and the suffix gets fresh blocks. The suffix is prefilled on
        the staging row with its recurrent state restored from the deepest
        snapshot the tree has.

        Args:
            seq: Sequence to prefill; already holds its decode row.

        Returns:
            Logits for the last prompt position.

        Raises:
            RuntimeError: No block manager is attached. The batched path cannot
                run without one, since it must allocate KV before the forward.
        """
        model, bm = self.model, self.bm
        if bm is None:
            raise RuntimeError("batched prefill needs a block manager")
        bs = bm.block_size
        radix = bm.radix
        row = self._claim_row(seq)

        matched, states = 0, None
        if radix is not None:
            # Reserve before matching: eviction can only pick unreferenced
            # leaves, so once this sequence holds blocks they are no longer
            # evictable and the pool could come up short mid-prefill.
            bm.reserve((seq.num_tokens + bs - 1) // bs + 1)
            # Drop the last token: a match including it would leave nothing to
            # run the forward on and no logits to return.
            blocks, matched, states = bm.match_prefix(seq.token_ids[:-1],
                                                      require_states=True,
                                                      mm=seq.mm)
            seq.block_table = list(blocks)
            seq.matched = matched
        bm.ensure_blocks(seq, seq.num_tokens)

        stage = model.prefill_row
        bm.assign(seq.block_table, row=stage)
        model.pos[stage:stage + 1].fill_(matched)
        if states is not None:
            model.restore_linear_states(states, row=stage)
        else:
            model.zero_linear_states(stage)

        # Snapshots are only reusable on block boundaries, so the block-aligned
        # head is prefilled in `snapshot_interval` slices (one snapshot each) and
        # the tail last -- only its logits matter, nothing downstream can reuse it.
        states_at: dict = {}
        if states is not None and matched:
            states_at[matched] = states
        rest = seq.token_ids[matched:]
        n_total = len(seq.token_ids)
        head_len = max(0, (n_total // bs) * bs - matched)
        step = max(bs, self.cfg.snapshot_interval // bs * bs)
        # With MTP on, the prefill also fills that sequence's drafter row so its
        # first speculative step has context. This affects acceptance rate only,
        # never correctness.
        spec_row = row if self.spec else None
        logits = None
        for c0 in range(0, head_len, step):
            c1 = min(c0 + step, head_len)
            logits = model.prefill(rest[c0:c1], chunk=self.cfg.chunk,
                                   spec_row=spec_row, mm=seq.mm)
            states_at[matched + c1] = model.snapshot_linear_states(row=stage)
            # Budget is enforced deepest-first so the *shallow* boundaries
            # survive: a request is far likelier to share a short prefix than a
            # long one, and `matched` itself must never be dropped.
            while len(states_at) > self.cfg.snapshots:
                droppable = [k for k in states_at if k != matched]
                if not droppable:
                    break
                del states_at[min(droppable)]
        if head_len < len(rest) or logits is None:
            logits = model.prefill(rest[head_len:], chunk=self.cfg.chunk,
                                   spec_row=spec_row, mm=seq.mm)

        model.move_linear_states(stage, row)
        if radix is not None:
            radix.insert(seq.token_ids, seq.block_table, states_at, mm=seq.mm)
        return logits

    def can_group_prefill(self) -> bool:
        """Whether several prompts can share one prefill forward.

        Returns:
            ``True`` when batching is on and a block manager is attached; both
            are prerequisites for a grouped prefill.
        """
        return self.batching and self.bm is not None

    def _match_for_prefill(self, seq: "Sequence") -> tuple[int, dict | None]:
        """Adopt ``seq``'s longest cached prefix.

        Args:
            seq: Sequence about to be prefilled; its block table and ``matched``
                are updated in place.

        Returns:
            ``(matched, states)``, both zero / ``None`` without a radix tree.
        """
        bm = self.bm
        if bm is None or bm.radix is None:
            return 0, None
        bs = bm.block_size
        bm.reserve((seq.num_tokens + bs - 1) // bs + 1)
        blocks, matched, states = bm.match_prefix(seq.token_ids[:-1],
                                                  require_states=True,
                                                  mm=seq.mm)
        seq.block_table = list(blocks)
        seq.matched = matched
        return matched, states

    @torch.inference_mode()
    def prefill_group(self, seqs: list["Sequence"]) -> list[torch.Tensor]:
        """Prefill several prompts in one forward.

        Prefix reuse composes with this: each sequence adopts its matched blocks
        and recurrent state first (into its own row), the group forward starts
        every prompt at its own ``matched`` offset, and the block-boundary
        snapshots it returns are inserted back into the radix tree.

        Args:
            seqs: Prompts to prefill. Each gets a row if it has none yet.

        Returns:
            One logits tensor per sequence, in the same order.

        Raises:
            RuntimeError: No block manager is attached.
        """
        t0 = time.time()
        bm = self.bm
        if bm is None:
            raise RuntimeError("grouped prefill needs a block manager")
        radix = bm.radix
        matched_states: list[tuple[int, dict | None]] = []
        for seq in seqs:
            if seq.row < 0:
                self._claim_row(seq)
            matched, states = self._match_for_prefill(seq)
            bm.ensure_blocks(seq, seq.num_tokens)
            if states is not None:
                self.model.restore_linear_states(states, row=seq.row)
            matched_states.append((matched, states))

        # Image prompts pass their payload on ``seq.mm``; ``model.prefill_group``
        # reads it from the sequences, so this path needs no mm argument.
        logits, snaps = self.model.prefill_group(
            seqs, snapshot_interval=(self.cfg.snapshot_interval
                                     if radix is not None else None))
        for seq, (matched, states), snap in zip(seqs, matched_states, snaps):
            seq.num_cached_tokens = len(seq.token_ids)
            seq.num_scheduled_tokens = seq.num_cached_tokens - matched
            if radix is None:
                continue
            states_at = dict(snap)
            if matched and states is not None:
                states_at[matched] = states
            while len(states_at) > self.cfg.snapshots:
                droppable = [k for k in states_at if k != matched]
                if not droppable:
                    break
                del states_at[min(droppable)]
            radix.insert(seq.token_ids, seq.block_table, states_at, mm=seq.mm)
        self.last_prefill_s = time.time() - t0
        return logits

    def begin_token(self, seq: "Sequence") -> None:
        """Open the one-token write window (prefix-cache path only).

        The window has to be opened before the KV write, and a partial tail block
        that is still shared with the radix tree is copied at that point, which is
        what keeps the cached prefix read-only.

        Args:
            seq: Sequence about to decode one token.
        """
        if self.pc is not None and self.pc.seq is not None:
            self.pc.seq.reserve(1)

    def decode_step(self, seq: "Sequence", token: int) -> torch.Tensor:
        """Advance one token through the captured single-stream decode graph.

        Args:
            seq: Sequence being advanced.
            token: Token to feed, i.e. the previous output.

        Returns:
            Logits for the next position.
        """
        logits = self.model.decode_step(token)
        if self.pc is not None and self.pc.seq is not None:
            self.pc.seq.commit_upto(self.model.pos1)
        seq.num_cached_tokens = int(self.model.pos1)
        return logits

    @torch.inference_mode()
    def decode(self, seqs: list["Sequence"]) -> list[int]:
        """Next token for each sequence, batched when more than one is running.

        Args:
            seqs: Running sequences to advance by one token.

        Returns:
            One sampled token id per sequence, in the same order.
        """
        if self.batching:
            return self._decode_batch(seqs)
        out: list[int] = []
        for seq in seqs:
            self.begin_token(seq)
            logits = self.decode_step(seq, seq.token_ids[-1])
            out.append(self.sample(logits, seq.params.temperature))
        return out

    @torch.inference_mode()
    def spec_step(self, seqs: list["Sequence"]):
        """One batch-friendly speculative step.

        Args:
            seqs: Running sequences. Each must already have its pending token
                appended by the engine.

        Returns:
            ``(accepted, logits)``. ``accepted[i]`` starts with sequence ``i``'s
            pending token followed by the drafts the main model accepted, so the
            caller appends ``accepted[i][1:]``. ``None`` when the drafter's
            window cannot cover a row — the caller must then fall back to a plain
            batched decode step rather than skip the sequence.

        Raises:
            RuntimeError: Batching or speculation is off. Falling back is the
                caller's job, so this is a programming error, not a runtime
                condition.
        """
        model, bm = self.model, self.bm
        if bm is None or not self.spec:
            raise RuntimeError("spec_step() needs the batched spec engine")
        rows = [seq.row for seq in seqs]
        active = set(rows)
        bucket = next(b for b in self.buckets if b > max(active))
        tokens: list[int] = []
        positions: list[int] = []
        tables: list[list[int]] = []
        for seq in seqs:
            pos = seq.num_cached_tokens
            # +1 for the pending token, +k for the drafts: all of them get KV.
            bm.ensure_blocks(seq, pos + self.spec_k + 1)
            tokens.append(seq.token_ids[-1])
            positions.append(pos)
            tables.append(seq.block_table)
        try:
            accepted, logits, _counts = model.spec_step_batch(
                tokens, positions, tables, bucket, K=self.spec_k,
                row_indices=rows,
                pad_rows=[r for r in range(bucket) if r not in active])
        except RuntimeError:
            # The drafter was captured for a fixed window; a row whose context
            # no longer fits makes it refuse. Signalling the caller is cheaper
            # than rebuilding the drafter mid-generation.
            return None
        for seq, acc in zip(seqs, accepted):
            seq.num_cached_tokens += len(acc)
        return accepted, logits

    def _decode_batch(self, seqs: list["Sequence"]) -> list[int]:
        """Advance several sequences through one captured decode graph.

        Args:
            seqs: Running sequences, each with a stable row.

        Returns:
            One sampled token id per sequence.

        Raises:
            RuntimeError: No block manager is attached.
        """
        model, bm = self.model, self.bm
        if bm is None:
            raise RuntimeError("batched decode needs a block manager")
        # Rows are stable for a sequence's lifetime, so a finished sequence leaves
        # a hole rather than forcing a permutation of recurrent state.  The step
        # runs the smallest bucket above the highest live row and the model pads
        # every hole with its sentinel row.
        rows = [seq.row for seq in seqs]
        active = set(rows)
        bucket = next(b for b in self.buckets if b > max(active))
        pad_rows = [r for r in range(bucket) if r not in active]
        tokens: list[int] = []
        positions: list[int] = []
        tables: list[list[int]] = []
        for seq in seqs:
            pos = seq.num_cached_tokens
            bm.ensure_blocks(seq, pos + 1)
            tokens.append(seq.token_ids[-1])
            positions.append(pos)
            tables.append(seq.block_table)
        model.set_rows(tokens, positions, tables, row_indices=rows)
        logits = model.decode_batch(bucket, bucket, pad_rows=pad_rows)
        for seq, pos in zip(seqs, positions):
            seq.num_cached_tokens = pos + 1
        return self._sample_rows(logits, seqs)

    @staticmethod
    def _sample_rows(logits: torch.Tensor, seqs: list["Sequence"]) -> list[int]:
        """Sample each sequence from its own row of the batched logits.

        Args:
            logits: Batched logits, ``[bucket, vocab]``.
            seqs: Sequences in row order.

        Returns:
            One token id per sequence. A single-row logits tensor is read at
            index 0 regardless of ``seq.row``, because the bucket-1 graph always
            writes row 0 even for a sequence that kept a higher row.
        """
        temps = [s.params.temperature for s in seqs]
        out: list[int] = []
        for seq in seqs:
            row = logits[seq.row] if logits.shape[0] > 1 else logits[0]
            temp = seq.params.temperature
            if temp > 0:
                p = torch.softmax(row.float() / temp, dim=-1)
                out.append(int(torch.multinomial(p, 1)))
            else:
                out.append(int(row.argmax()))
        return out

    @staticmethod
    def sample(logits: torch.Tensor, temperature: float) -> int:
        """Pick one token from ``logits``.

        Args:
            logits: ``[vocab]`` logits, possibly bf16.
            temperature: ``0`` selects the argmax; the sampling path upcasts to
                fp32 first because bf16 softmax loses too much tail mass.

        Returns:
            The sampled token id.
        """
        if temperature > 0:
            p = torch.softmax(logits.float() / temperature, dim=-1)
            return int(torch.multinomial(p, 1))
        return int(logits.argmax())
