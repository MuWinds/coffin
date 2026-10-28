"""One request, from admission to retirement.

Module: ``coffin/`` (top level). This is the unit the scheduler queues, the
model runner advances and the prefix tree refcounts. ``config.py`` and this
module import nothing else from the engine, so everything can depend on them.

Constraints:
    * The sequence owns **lengths**, not device buffers. The authoritative
      position after a speculative step is the model's own counter
      (``model.pos1``), so the scheduler refreshes ``num_cached_tokens`` from
      the runner instead of incrementing it here.
    * ``block_table`` is shared state: the radix tree refcounts the same block
      ids, and the batched path hands them to other sequences. Never cache a
      derived view of it across a call that can allocate or release.
    * ``token_ids`` is the request's own record of what was asked and what was
      emitted, so it must survive every transition. Preemption resets the *cache*
      fields (cursor, logits, matched, cached length) and never the tokens: a
      recompute replays the same prefix, and a dropped generated token would be
      lost output rather than recomputed state.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum, auto


class SequenceStatus(Enum):
    """Lifecycle of one request, as far as the scheduler can observe it.

    ``PREFILLING`` and ``RUNNING`` are both schedulable states and both live in
    the scheduler's running list; the split exists because only ``RUNNING`` may
    be handed to a decode step. A sequence whose prompt is still being written
    has partial KV, so decoding it would attend over tokens the model has not
    computed yet.

    ``PREEMPTED`` is the parked state of the recompute policy: the sequence keeps
    its tokens and its generation budget but holds no blocks and no batch row, so
    it is back in the waiting queue waiting to be admitted again.
    """

    WAITING = auto()
    PREFILLING = auto()
    RUNNING = auto()
    PREEMPTED = auto()
    FINISHED = auto()


@dataclass(slots=True)
class SamplingParams:
    """Per-request decoding policy.

    ``eos`` and ``deadline`` are filled in by the engine rather than the caller:
    the tokenizer knows the stop ids and the server owns the wall-clock budget,
    so neither belongs to whatever builds the request.
    """

    temperature: float = 0.0
    max_tokens: int = 64
    ignore_eos: bool = False
    eos: set[int] = field(default_factory=set)
    deadline: float = 0.0


class Sequence:
    """One generation request and the block table it currently holds."""

    _counter = itertools.count()

    def __init__(self, token_ids: list[int], params: SamplingParams | None = None,
                 seq_id: int | None = None) -> None:
        """Wrap a prompt in the state the engine mutates as it generates.

        Args:
            token_ids: Prompt tokens. Copied, because callers keep reusing the
                list they tokenised into.
            params: Decoding policy. ``None`` means greedy defaults, which is
                what the internal callers want.
            seq_id: Explicit id for tests that need a deterministic allocation;
                otherwise taken from a process-wide counter.
        """
        self.seq_id = next(Sequence._counter) if seq_id is None else seq_id
        self.status = SequenceStatus.WAITING
        self.token_ids: list[int] = list(token_ids)
        self.num_prompt_tokens: int = len(self.token_ids)
        #: Prompt tokens whose KV is in the pool right now, counted from token 0.
        #: Chunked prefill advances this a quantum at a time, so it trails
        #: ``num_prompt_tokens`` while the request is ``PREFILLING``; it is what
        #: the next chunk starts from. Distinct from ``num_cached_tokens``: this
        #: one counts prompt tokens only, that one counts every position written,
        #: generated tokens included.
        self.prefill_cursor: int = 0
        #: Next-token logits of the finished prompt, held between the prefill that
        #: produced them and the decode step that samples from them. Chunked
        #: prefill spreads those two across steps (the last chunk ends a step with
        #: the request already ``RUNNING``), so the tensor has to survive the gap.
        self.prefill_logits = None
        #: Whether :meth:`ModelRunner.begin_prefill` has run for the current
        #: admission. A resumed sequence keeps its row and blocks, so the runner
        #: must not re-match the prefix or re-claim the row for it.
        self.prefill_started: bool = False
        #: Block-boundary recurrent-state snapshots this prefill has taken so far,
        #: keyed by absolute prompt position. They are inserted into the radix tree
        #: as each chunk completes rather than at the end, so a preemption
        #: mid-prefill leaves everything already computed reusable.
        self.prefill_states_at: dict = {}
        #: Scheduler step that last put this sequence in a batch. Feeds the
        #: preemption score, which prefers a victim that has already been starved.
        self.last_scheduled_step: int = 0
        #: How many times this request has been preempted and restarted. The
        #: preemption score discounts a request that has been evicted already, so
        #: one long prompt cannot be evicted on every step while the rest of the
        #: queue runs.
        self.preempt_count: int = 0
        #: Scheduler step of the most recent preemption. Admission skips a request
        #: preempted this step, which is what stops it from being evicted and
        #: immediately re-admitted into the same blocks it just gave up.
        self.preempted_step: int = -1
        self.num_cached_tokens: int = 0
        self.num_scheduled_tokens: int = 0
        self.params: SamplingParams = params or SamplingParams()
        #: Blocks this sequence holds a reference on. Shared with the radix tree
        #: and, on the batched path, with other sequences.
        self.block_table: list[int] = []
        #: Row in the batched model buffers. ``-1`` = state not yet staged.
        self.row: int = -1
        #: Wall-clock of this request's prefill, set by the engine.
        self.prefill_s: float = 0.0
        #: Tokens adopted from the radix tree by the last prefill.
        self.matched: int = 0
        #: Set by the runner when the model itself ended the sequence.
        self.stop_reason: str | None = None
        #: Multimodal payload of the prompt (``mm_input.MMInput``) or ``None``.
        #: Carried here so the runner, the block manager and the radix tree all
        #: see the image digest without a parallel side table to keep in sync.
        self.mm = None

    def __len__(self) -> int:
        return len(self.token_ids)

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def num_completion_tokens(self) -> int:
        return self.num_tokens - self.num_prompt_tokens

    @property
    def completion_token_ids(self) -> list[int]:
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    @property
    def is_prefilling(self) -> bool:
        """Whether the prompt is still being written into the cache.

        Returns:
            ``True`` while the sequence is ``PREFILLING``. A decode step may only
            run when this is ``False``: KV below ``prefill_cursor`` is partial,
            and attention over it would read slots the model has not filled.
        """
        return self.status == SequenceStatus.PREFILLING

    @property
    def num_blocks_owned(self) -> int:
        """Blocks this sequence currently holds a reference on.

        Returns:
            The length of ``block_table``, i.e. live references only: a preempted
            sequence has already dropped its table. The preemption score reads
            this to find where the blocks are.
        """
        return len(self.block_table)

    def blocks_needed(self, block_size: int) -> int:
        """Physical blocks this sequence occupies once prefilled.

        Args:
            block_size: Tokens per block, from ``EngineConfig``.

        Returns:
            Block count, rounded up. The partial tail block is charged in full
            because KV is allocated in whole blocks.
        """
        return (self.num_tokens + block_size - 1) // block_size

    def append_token(self, token_id: int) -> None:
        """Append one decoded token.

        Args:
            token_id: Token sampled by the runner, already checked against the
                stop ids.
        """
        self.token_ids.append(token_id)

    def finish(self, reason: str = "eos") -> None:
        """Mark the request finished.

        Args:
            reason: ``"eos"``, ``"length"``, ``"deadline"`` or a model-supplied
                reason. It reaches the client as ``finish_reason``, so it has to
                be one of the values the OpenAI surface accepts.
        """
        self.status = SequenceStatus.FINISHED
        self.stop_reason = reason

    def park_preempted(self) -> None:
        """Reset the cache state of a sequence that is about to be evicted.

        Recomputed from scratch on the next admission, so everything that
        describes *where its KV is* goes back to zero while the tokens -- prompt
        and already-emitted ones -- stay: the emitted prefix is part of the
        request now, and dropping it would duplicate or lose output tokens.

        The caller must already have inserted the finished blocks into the radix
        tree and released the sequence's own references; this method deliberately
        does not touch ``block_table``, so a caller that has not released it
        fails on the next allocation rather than silently leaking.
        """
        self.status = SequenceStatus.PREEMPTED
        self.prefill_cursor = 0
        self.prefill_started = False
        self.prefill_logits = None
        self.prefill_states_at = {}
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.matched = 0
        self.preempt_count += 1

    def resume(self) -> None:
        """Put a preempted sequence back in the admission queue.

        Admission is the same path a fresh request takes: match the prefix over
        the whole token stream (``prompt + generated``), restore the boundary's
        recurrent state and prefill from there. The status change is what makes
        the scheduler treat it as waiting rather than as a victim.
        """
        self.status = SequenceStatus.WAITING

    def over_budget(self, now: float) -> bool:
        """Whether the request outlived its wall-clock budget.

        Args:
            now: Current wall-clock time. Passed in rather than read here so one
                scheduling pass judges every sequence against the same instant.

        Returns:
            ``True`` when a deadline is set and has passed. A deadline of ``0``
            means "no budget" and never expires.
        """
        return bool(self.params.deadline) and now >= self.params.deadline
