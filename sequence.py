"""One request, from admission to retirement.

The engine used to pass bare token lists around and keep the block table hidden
inside ``prefix_engine.SeqKV``.  :class:`Sequence` is that state, made explicit:
it is the unit the scheduler queues, the model runner advances and the prefix
tree refcounts.

The sequence owns **lengths**, not device buffers.  The authoritative position
after a speculative step is the model's own counter (``model.pos1``); the
scheduler therefore refreshes ``num_cached_tokens`` from the runner rather than
incrementing it itself.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum, auto


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


@dataclass(slots=True)
class SamplingParams:
    """Per-request decoding policy.

    ``eos`` and ``deadline`` are filled in by the engine: the tokenizer knows the
    stop ids and the server owns the wall-clock budget, neither belongs to the
    model.
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
        self.seq_id = next(Sequence._counter) if seq_id is None else seq_id
        self.status = SequenceStatus.WAITING
        self.token_ids: list[int] = list(token_ids)
        self.num_prompt_tokens: int = len(self.token_ids)
        self.num_cached_tokens: int = 0
        self.num_scheduled_tokens: int = 0
        self.params: SamplingParams = params or SamplingParams()
        #: Blocks this sequence holds a reference on.  Shared with the radix tree
        #: (and, once batching lands, with other sequences).
        self.block_table: list[int] = []
        #: Row in the batched model buffers.  ``-1`` = state not yet staged.
        self.row: int = -1
        #: Wall-clock of this request's prefill, set by the engine.
        self.prefill_s: float = 0.0
        #: Tokens adopted from the radix tree by the last prefill.
        self.matched: int = 0
        #: Set by the runner when the model itself ended the sequence.
        self.stop_reason: str | None = None

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
        """Physical blocks this sequence occupies once prefilled."""
        return (self.num_tokens + block_size - 1) // block_size

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)

    def finish(self, reason: str = "eos") -> None:
        self.status = SequenceStatus.FINISHED
        self.stop_reason = reason

    def over_budget(self, now: float) -> bool:
        """True when the request outlived its wall-clock budget."""
        return bool(self.params.deadline) and now >= self.params.deadline
