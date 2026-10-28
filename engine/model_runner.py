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
    * A prompt may be prefilled over several steps (chunked prefill). The
      sequence's recurrent state lives on its own decode row between chunks and
      only visits the staging row for the duration of one chunk, so a paused
      request costs no extra VRAM: a copy of the state is ~50 MB.
    * Decode may not capture a graph (``EngineConfig.graph`` is a correctness
      switch, not a tuning knob). Both decode entries produce the same numbers,
      so the only difference is which one is called.

Chunked prefill is three calls, and the split is what makes it resumable:

    begin_prefill(seq)      claim a row, adopt a radix prefix, place the state
    prefill_chunk(seq, n)   advance the cursor by at most n tokens
    finish_prefill(seq)     record the last logits, insert the prefix, go RUNNING

``begin_prefill`` runs once per admission and ``prefill_chunk`` as often as the
scheduler grants budget; everything the second call needs is on the sequence, so
a step boundary is a legal place to pause. A preempted request goes through the
same three calls again from whatever prefix the radix tree still holds.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch

from .sequence import SequenceStatus

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
            cfg: Engine configuration; the prefill chunk size, the prefill
                quantum and the snapshot budget are read from it.
        """
        self.model = model
        self.cfg = cfg
        self.pc: "PrefixCacheEngine | None" = None
        self.bm: "BlockManager | None" = None
        self.batching = False
        self.buckets: tuple[int, ...] = ()
        #: Whether the decode step may replay a captured graph. ``False`` routes
        #: every token through the model's eager forward instead: the numbers are
        #: the same, but nothing is recorded, which is the only safe path on a
        #: backend whose capture is untrusted (a graph replaying stale outputs is a
        #: silent wrong answer). Set from ``EngineConfig.graph`` by the engine.
        self.graph = cfg.graph
        #: Batch-friendly speculative decoding (MTP), off unless the engine asks.
        self.spec = False
        self.spec_k = 0
        self.spec_sparse_k = 0
        #: decode row -> seq_id. Rows are stable for a sequence's lifetime, which
        #: is what lets a batched step skip permuting recurrent state.
        self._row_owner: dict[int, int] = {}
        #: Set by the engine when a prefix cache is attached; used by ``ping``.
        self.last_prefill_s = 0.0
        #: Tokens one chunk may cover. Kept here rather than read from the
        #: scheduler so a runner-only caller (tests, probes) chunks identically.
        self.prefill_quantum = max(cfg.block_size, cfg.prefill_quantum)

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

    def can_chunk_prefill(self) -> bool:
        """Whether this engine can advance a prompt in pieces.

        The single-stream prefix engine prefills a prompt in one call and keeps
        its own cache bookkeeping, so a token budget cannot bound it; the batched
        path allocates per sequence and therefore can.

        Returns:
            ``True`` for the batched runner with a block manager attached.
        """
        return self.batching and self.bm is not None

    def free_blocks(self) -> int:
        """Blocks the pool can hand out right now.

        The scheduler asks this before admitting a request, so a client that
        cannot be given the blocks its next chunk needs waits in the queue
        instead of being admitted, found unrunnable and evicted again. Without
        the question the two requests trade the pool every step and neither
        finishes.

        Returns:
            Free block count, or a large number when no pool is attached (the
            single-stream engines allocate nothing per request).
        """
        bm = self.bm
        return 1 << 30 if bm is None else bm.num_free

    def blocks_for(self, tokens: int) -> int:
        """Blocks needed to hold ``tokens`` positions.

        Args:
            tokens: Positions the caller is about to occupy.

        Returns:
            Block count at the configured block size.
        """
        bm = self.bm
        if bm is None:
            return 0
        return (tokens + bm.block_size - 1) // bm.block_size

    def reserve_for(self, tokens: int) -> int:
        """Evict cached prefixes until ``tokens`` positions can be held.

        The scheduler asks this before admitting a request, because the pool's
        free count alone understates what is available: most of a full pool is
        usually cached prefix that nobody is reading from and that eviction can
        return. Checking without evicting first would leave a request waiting
        behind blocks the tree could have given up.

        Args:
            tokens: Positions the request needs to hold.

        Returns:
            Free blocks after the eviction pass.
        """
        bm = self.bm
        if bm is None:
            return 1 << 30
        bm.reserve(self.blocks_for(tokens))
        return bm.num_free

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
        """Prepare the batched decode path, capturing one graph per bucket.

        With capture on, each bucket is a recorded replay. With it off the model
        allocates the same row buffers and staging blocks but records nothing,
        and :meth:`_decode_batch` calls the eager entry instead -- so the two
        modes agree and a backend that must not capture still batches.

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
        if self.graph:
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
        """Feed the whole prompt (minus the cached prefix); return the logits.

        The unchunked entry point. It exists for the engines that cannot chunk
        (single-stream prefix cache) and as the "one chunk covering everything"
        case of the batched path, so both keep exactly the numbers they had.

        Args:
            seq: Sequence to prefill. Its ``num_cached_tokens`` and
                ``num_scheduled_tokens`` are refreshed from the model afterwards.

        Returns:
            Logits for the last prompt position, ``[vocab]``.

        Raises:
            RuntimeError: The batched path was asked to finish a prompt in one
                chunk and the cursor did not reach the end. That means the call
                ran a chunk smaller than the prompt, which is a caller bug: use
                :meth:`prefill_chunk` for chunked prefill.
        """
        t0 = time.time()
        if self.pc is not None:
            logits = self.pc.prefill(seq.token_ids, mm=seq.mm)
            seq.matched = self.pc.last_matched
            if self.pc.seq is not None:
                seq.block_table = list(self.pc.seq.blocks)
            seq.prefill_cursor = seq.num_tokens
        elif self.batching:
            self.begin_prefill(seq)
            logits = self.prefill_chunk(seq, seq.num_tokens - seq.prefill_cursor)
            if logits is None:                       # pragma: no cover - defensive
                raise RuntimeError("prefill_chunk() did not finish the prompt")
        else:
            self.model.reset()
            logits = self.model.prefill(seq.token_ids, mm=seq.mm)
            seq.prefill_cursor = seq.num_tokens
        seq.num_cached_tokens = int(self.model.pos1)
        seq.num_scheduled_tokens = seq.num_cached_tokens - seq.matched
        self.last_prefill_s = time.time() - t0
        return logits

    @torch.inference_mode()
    def begin_prefill(self, seq: "Sequence") -> None:
        """Open a resumable prefill: row, prefix match, starting state.

        Runs once per admission. The prefix match is taken over the whole token
        stream (prompt plus anything already generated, minus the last token
        whose logits have to come out of a real forward), so a resumed request
        restarts from the deepest block boundary the radix tree can justify
        instead of from token 0.

        Only the blocks the first chunk needs are reserved here. Reserving the
        whole prompt would let one long request evict the cache of every other
        request for blocks it may not reach for many steps.

        Args:
            seq: Sequence to admit. It must not already hold a row; a resumed
                sequence keeps its row and blocks, so the caller does not repeat
                this call for it.

        Raises:
            RuntimeError: No block manager is attached. The batched path cannot
                run without one, since it must allocate KV before the forward.
        """
        bm, model = self.bm, self.model
        if bm is None:
            raise RuntimeError("chunked prefill needs a block manager")
        radix = bm.radix
        if seq.row < 0:
            row = self._claim_row(seq)
        else:
            row = seq.row

        matched, states = 0, None
        if radix is not None:
            # Blocks from an earlier admission must go back before the table is
            # replaced by the matched prefix: a resumed sequence normally arrives
            # with an empty table, and silently dropping one that is not empty
            # would leak the references the pool counts.
            if seq.block_table:
                bm.release(seq)
            bm.reserve(1)
            blocks, matched, states = bm.match_prefix(seq.token_ids[:-1],
                                                      require_states=True,
                                                      mm=seq.mm)
            seq.block_table = list(blocks)
        seq.matched = matched
        seq.prefill_cursor = matched
        seq.prefill_states_at = {}
        seq.prefill_logits = None
        seq.prefill_started = True
        # The state is restored on the staging row and then moved to the
        # sequence's own row, which is where it lives between chunks. Restoring
        # straight into the decode row would need that row to be addressable
        # before any forward has run, and the staging row is the only row the
        # one-token paths are guaranteed to have.
        stage = model.prefill_row
        if states is not None:
            seq.prefill_states_at[matched] = states
            model.restore_linear_states(states, row=stage)
        else:
            model.zero_linear_states(stage)
        model.move_linear_states(stage, row)
        seq.num_cached_tokens = matched
        seq.num_scheduled_tokens = 0

    @torch.inference_mode()
    def prefill_chunk(self, seq: "Sequence", budget: int) -> torch.Tensor | None:
        """Advance the prompt by at most ``budget`` tokens.

        The chunk runs on the staging row, with the sequence's recurrent state
        moved there for the duration and moved back afterwards, so between
        chunks the state sits on the sequence's own decode row and no paused
        request costs a private copy.

        Args:
            seq: Sequence whose ``prefill_cursor`` is where this chunk starts.
                ``begin_prefill`` must have run for it.
            budget: Maximum tokens this chunk may cover. A larger remainder is
                left for the next chunk; a smaller one finishes the prompt.

        Returns:
            The prompt's last-token logits when this chunk completes the prompt,
            ``None`` when more chunks are needed. Completion also runs
            :meth:`finish_prefill`, so the caller only has to sample.

        Raises:
            RuntimeError: ``begin_prefill`` did not run for this sequence, or no
                block manager is attached.
        """
        bm, model = self.bm, self.model
        if bm is None:
            raise RuntimeError("chunked prefill needs a block manager")
        if not seq.prefill_started:
            raise RuntimeError("prefill_chunk() before begin_prefill()")
        t0 = time.time()
        bs = bm.block_size
        radix = bm.radix
        stage = model.prefill_row
        row = seq.row
        cursor = seq.prefill_cursor
        prompt_len = seq.num_prompt_tokens
        # The stream a resumed request recomputes is prompt plus the tokens it
        # already emitted: the prompt boundary is where new tokens are sampled
        # from, not where the forward stops.
        total = seq.num_tokens
        end = min(cursor + max(int(budget), 1), total)
        if end <= cursor:                       # pragma: no cover - caller guards
            self.last_prefill_s = time.time() - t0
            return seq.prefill_logits

        # Evict before allocating: once this request holds the blocks it is
        # asking for, they are no longer candidates for the eviction that has to
        # make room for them.
        held = len(seq.block_table) * bs
        if end > held:
            bm.reserve((end - held + bs - 1) // bs + 1)
        bm.ensure_blocks(seq, end)

        model.move_linear_states(row, stage)
        bm.assign(seq.block_table, row=stage)
        model.pos[stage:stage + 1].fill_(cursor)

        # Two granularities, for two different reasons. `step` is the snapshot
        # interval rounded up to a whole block: a recurrent state is only
        # reusable on a block boundary, and one snapshot per block of a long
        # prompt costs more VRAM than the weights, so the boundaries are thinned.
        # `cfg.chunk` is the GEMM's M inside one call and is purely a tuning
        # number (measured 865/1035/1094 tok/s at 256/512/2048). Keeping them
        # apart is what lets the GEMM be re-tuned without changing which
        # boundaries exist.
        step = max(bs, self.cfg.snapshot_interval // bs * bs)
        # End of the block-aligned head: from there on only logits matter, so the
        # unaligned tail (and any generated tokens) runs as one pass.
        head_len = max(0, (prompt_len // bs) * bs - cursor)
        spec_row = row if self.spec else None
        logits = None
        done = cursor
        while done < end:
            stop = end if done - cursor >= head_len else min(done + step, end)
            logits = model.prefill(seq.token_ids[done:stop], chunk=self.cfg.chunk,
                                   spec_row=spec_row, mm=seq.mm)
            done = stop
            seq.prefill_cursor = done
            seq.num_cached_tokens = done
            if radix is not None and done % bs == 0 and done <= prompt_len:
                seq.prefill_states_at[done] = model.snapshot_linear_states(row=stage)
                self._trim_states(seq)
                # Insert per boundary rather than at the end of the prompt: a
                # request preempted mid-prefill then leaves the blocks it has
                # already computed reusable, and a second request can join at
                # them while the first is still prefilling.
                self._insert_prefix(seq, radix)

        model.move_linear_states(stage, row)
        seq.num_scheduled_tokens = seq.prefill_cursor - seq.matched
        if seq.prefill_cursor < total:
            self.last_prefill_s = time.time() - t0
            return None
        seq.prefill_logits = logits
        self.finish_prefill(seq)
        self.last_prefill_s = time.time() - t0
        return logits

    def finish_prefill(self, seq: "Sequence") -> None:
        """Close a completed prefill and make the request decodable.

        Inserts the prompt into the radix tree (including any boundary the last
        chunk produced), then flips the sequence to ``RUNNING``. Sampling the
        first generated token is the caller's job: it holds the logits this
        prefill just returned.

        Args:
            seq: Sequence whose ``prefill_cursor`` reached the end of its tokens.
        """
        bm = self.bm
        if bm is not None and bm.radix is not None:
            self._insert_prefix(seq, bm.radix)
        seq.status = SequenceStatus.RUNNING
        seq.num_cached_tokens = seq.num_tokens

    def _insert_prefix(self, seq: "Sequence", radix) -> None:
        """Record the block-aligned part of ``seq``'s stream in the radix tree.

        Called after every chunk rather than once at the end: a request
        preempted mid-prefill then leaves everything it already computed
        reusable, and a second request can join at any completed block boundary
        while the first is still prefilling.

        Args:
            seq: Sequence holding the blocks to record.
            radix: The prefix tree. Not ``None``; the caller checks.
        """
        bs = self.bm.block_size
        full = min(seq.prefill_cursor, (seq.num_tokens // bs) * bs)
        if full <= 0:
            return
        radix.insert(seq.token_ids[:full], seq.block_table[:full // bs],
                     dict(seq.prefill_states_at), mm=seq.mm)

    def _trim_states(self, seq: "Sequence") -> None:
        """Cap a sequence's live snapshots at the tree's own budget.

        A snapshot is ~50 MB (24 linear-attention layers), so keeping every
        block boundary of a long prompt would cost more than the weights. The
        deepest boundaries go first, and the prefix the sequence joined at is
        never dropped: it is where a resumed request would restart.

        Args:
            seq: Sequence whose ``prefill_states_at`` is trimmed in place.
        """
        states_at = seq.prefill_states_at
        while len(states_at) > self.cfg.snapshots:
            droppable = [k for k in states_at if k != seq.matched]
            if not droppable:
                break
            del states_at[min(droppable)]

    @torch.inference_mode()
    def preempt(self, seq: "Sequence") -> None:
        """Evict a request and leave its prefix recoverable.

        Recomputation is the cheapest policy here: the blocks are int8 KV on the
        device, so swapping them to the host would cost a large synchronous copy
        per preemption, while the radix tree already holds every block-aligned
        prefix. The sequence therefore keeps its tokens and loses only the
        position bookkeeping that recompute rebuilds.

        Args:
            seq: Sequence to evict. It must be in the scheduler's running set;
                putting it back in the waiting queue is the caller's job so the
                engine can keep scheduler mutation in one place.
        """
        bm = self.bm
        if bm is not None:
            if bm.radix is not None and seq.prefill_cursor > 0:
                self._insert_prefix(seq, bm.radix)
            bm.release(seq)
        self.release_row(seq)
        seq.block_table.clear()
        seq.park_preempted()

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
            seq.prefill_cursor = seq.num_tokens
            seq.prefill_started = False
            seq.prefill_logits = None
            seq.num_cached_tokens = len(seq.token_ids)
            seq.num_scheduled_tokens = seq.num_cached_tokens - matched
            # The whole prompt went through this one forward, so the sequence is
            # decodable now. This path does not go through ``finish_prefill``, and
            # leaving ``PREFILLING`` behind would keep it out of every later
            # decode step.
            seq.status = SequenceStatus.RUNNING
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
        """Advance one token through the decode path (graph replay, or eager).

        Args:
            seq: Sequence being advanced.
            token: Token to feed, i.e. the previous output.

        Returns:
            Logits for the next position.
        """
        logits = self._forward_token(token)
        if self.pc is not None and self.pc.seq is not None:
            self.pc.seq.commit_upto(self.model.pos1)
        seq.num_cached_tokens = int(self.model.pos1)
        return logits

    def _forward_token(self, token: int) -> torch.Tensor:
        """One token through whichever decode path this engine captured.

        The two model entry points differ only in bookkeeping: ``decode_step``
        replays a captured graph, ``logits`` runs the same forward eagerly. Both
        stage the token on the prefill row and both advance the position counter,
        so everything downstream (the KV write window, ``commit_upto``) is
        identical and only the launch mechanism changes.

        Args:
            token: Token id to feed.

        Returns:
            Logits for the next position, a live buffer either way.
        """
        if self.graph:
            return self.model.decode_step(token)
        return self.model.logits(token)

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
        """Advance several sequences through one batched decode forward.

        Args:
            seqs: Running sequences, each with a stable row. The caller has
                already made room for the one position each of them writes; this
                method must not allocate, because an allocation here cannot evict
                or preempt and would leave a half-grown batch.

        Returns:
            One sampled token id per sequence.

        Raises:
            RuntimeError: No block manager is attached, or a sequence's table
                cannot hold the position it is about to write. Both mean the
                caller skipped its capacity check.
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
            if (pos + 1 + bm.block_size - 1) // bm.block_size > len(seq.block_table):
                raise RuntimeError(
                    f"decode row {seq.row} has no block for position {pos}: "
                    "the step's capacity check is missing")
            tokens.append(seq.token_ids[-1])
            positions.append(pos)
            tables.append(seq.block_table)
        model.set_rows(tokens, positions, tables, row_indices=rows)
        # Same staging, same body, same buffers: the only difference is whether
        # the forward was recorded into a graph. A backend that must not capture
        # (the SCOW driver) therefore still gets a batched step.
        if self.graph:
            logits = model.decode_batch(bucket, bucket, pad_rows=pad_rows)
        else:
            logits = model.decode_batch_eager(bucket, bucket, pad_rows=pad_rows)
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
        out: list[int] = []
        for seq in seqs:
            row = logits[seq.row] if logits.shape[0] > 1 else logits[0]
            out.append(ModelRunner.sample(row, seq.params.temperature))
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
