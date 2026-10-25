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
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum, auto


class SequenceStatus(Enum):
    """Lifecycle of one request, as far as the scheduler can observe it."""

    WAITING = auto()
    RUNNING = auto()
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
