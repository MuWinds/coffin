"""Which sequences run in the next step.

Module: ``coffin/`` (top level). Sits between ``llm_engine`` (which drives the
loop) and ``model_runner`` (which executes one step), so it may only import
``sequence`` / ``config`` and never the model or kernel layers.

Constraints:
    * The batch row count is the real capacity limit, not the sequence count:
      row ``capacity-1`` is reserved for a prefill while the decode rows stay
      live, so admit against ``max_num_seqs`` and let the runner pick a bucket.
    * All mutation is single-threaded, driven by the engine's step loop. Nothing
      here is re-entrant and no lock is taken.
    * One scheduling step may run at most one prefill chunk when chunked prefill
      is on: there is a single staging row, so two chunks in one step would have
      to serialise their recurrent-state moves anyway. The exception is a batch
      that fits the budget as a whole, which the runner executes as one grouped
      forward instead.

Scheduling order, and why it is this order:

* decode first, because a request that is already emitting tokens must not wait
  behind a prompt that just arrived (that is the head-of-line block chunked
  prefill exists to remove);
* then one prefill chunk with whatever token budget is left after the decode
  tokens, capped by ``prefill_quantum``;
* a step whose remaining budget is below one block is a decode-only step: a
  sub-block chunk costs a full recurrent-state move for a few tokens.

Admission no longer requires the whole prompt to fit the budget. A prompt longer
than the budget is admitted and advances one quantum per step, so it starts
producing instead of stalling every shorter request behind it.

Preemption is recompute-based and only reached from an allocation failure: the
engine asks for a victim when growing a block table fails even after the cache
has evicted everything it can. Victim selection is deliberately simple and
explainable -- see :meth:`Scheduler.preempt_victim`.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .sequence import SequenceStatus

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .config import EngineConfig
    from .model_runner import ModelRunner
    from .sequence import Sequence


#: Steps a sequence must have been left alone before it can be preempted. Zero
#: because the requester is already excluded by :meth:`Scheduler.preempt_victim`
#: and preempting a *different* sequence that was also selected this step is
#: exactly what makes room for the one that cannot grow: the victim's decode for
#: this step has not run yet, so no work is thrown away.
PREEMPT_MIN_GAP = 0

#: Steps a preempted sequence sits out before it can be admitted again. Without
#: it the victim comes straight back, re-matches the prefix it just gave up and
#: takes the same blocks, so a full pool becomes an endless eviction ping-pong
#: instead of one request finishing and freeing room for the other.
PREEMPT_COOLDOWN = 1


@dataclass(frozen=True, slots=True)
class PrefillChunk:
    """How many more prompt tokens one request may prefill in this step.

    Attributes:
        seq: Request to advance. It is already in the running list with status
            ``PREFILLING``, so the engine only has to execute the chunk.
        tokens: Tokens the chunk may cover, counted from ``seq.prefill_cursor``.
            This is a budget, not an exact length: a shorter remainder is
            prefilled and finishes the request's prompt.
    """

    seq: "Sequence"
    tokens: int


class Scheduler:
    """Admission and retirement of requests."""

    def __init__(self, cfg: "EngineConfig", runner: "ModelRunner | None" = None) -> None:
        """Build an empty scheduler bound to one engine.

        Args:
            cfg: Engine configuration; ``max_num_seqs``, ``max_num_batched_tokens``
                and ``prefill_quantum`` are the admission limits.
            runner: Optional back-reference used by the engine to compare the
                scheduler's view of a row against what the runner actually
                staged. ``None`` is valid for scheduler-only tests, and makes the
                scheduler take the non-chunked path.
        """
        self.cfg = cfg
        self.runner = runner
        self.waiting: deque["Sequence"] = deque()
        self.running: deque["Sequence"] = deque()
        self.max_num_seqs = max(1, cfg.max_num_seqs)
        self.max_num_batched_tokens = cfg.max_num_batched_tokens
        self.prefill_quantum = max(cfg.block_size, cfg.prefill_quantum)
        self.block_size = max(1, cfg.block_size)
        self.eos: set[int] = set()
        self.n_finished = 0
        #: Monotone step counter, stamped on every sequence a step touches. The
        #: preemption score reads it to prefer a victim that has been starved.
        self.step_idx = 0

    @property
    def chunked(self) -> bool:
        """Whether prompts may be advanced in pieces.

        Chunking needs the batched runner: the single-stream prefix engine
        prefills a prompt in one call and keeps its own cache bookkeeping, so the
        token budget below cannot bound it. Read on every step rather than cached
        in ``__init__`` because the runner only becomes batched in
        ``LLMEngine.build``, which runs after the scheduler is constructed. Without
        a runner this is a scheduler-only test, which is the same non-chunked
        path.

        Returns:
            ``True`` when the runner can chunk and the engine is therefore allowed
            to split a prompt across steps.
        """
        return bool(self.runner is not None and self.runner.can_chunk_prefill())

    def add(self, seq: "Sequence") -> None:
        """Queue a new request.

        Args:
            seq: Request to admit. If it carries no stop ids of its own it
                inherits the engine-wide set, which is how the tokenizer's
                defaults reach requests built by the API layer.
        """
        args = seq.params
        if not args.eos:
            args.eos = set(self.eos)
        self.waiting.append(seq)

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def schedule(self) -> tuple[list["Sequence"], list[PrefillChunk]]:
        """Pick the next step: what to decode and what to prefill.

        Decode is chosen first and its cost is subtracted from the token budget,
        so a prompt can never take the slots a generating request needs. Prefill
        then takes what is left, at most one chunk of it.

        Returns:
            ``(decode_seqs, prefill_chunks)``. ``decode_seqs`` holds sequences
            whose prompt is complete and which may run a decode step this step.
            ``prefill_chunks`` is empty, one chunk, or -- when several freshly
            admitted prompts fit the budget as a whole -- one chunk per prompt,
            which the runner executes as a single grouped forward. Both empty
            means nothing was runnable.
        """
        self.step_idx += 1
        decode = [s for s in self.running if s.status is SequenceStatus.RUNNING]
        for seq in decode:
            seq.last_scheduled_step = self.step_idx
        if not self.chunked:
            return decode, self._schedule_atomic()
        return decode, self._schedule_chunked(self.max_num_batched_tokens - len(decode))

    def _schedule_atomic(self) -> list[PrefillChunk]:
        """Admit one whole prompt, ignoring the token budget.

        The non-chunked engines prefilled the first admission regardless of the
        budget, because a prompt longer than the whole budget must still make
        progress. One sequence is the row limit on those paths, so there is
        nothing to group either.

        Returns:
            Zero or one chunk, always covering the request's whole remainder.
        """
        if not self.waiting or len(self.running) >= self.max_num_seqs:
            return []
        seq = self.waiting.popleft()
        self._begin(seq)
        # The sequence joins ``running`` here, not when the forward returns:
        # admission is what makes it a live row, and a caller that crashed
        # mid-prefill must still see it in the running set rather than have it
        # vanish from both queues.
        self.running.append(seq)
        return [PrefillChunk(seq, len(seq.token_ids) - seq.prefill_cursor)]

    def _schedule_chunked(self, budget: int) -> list[PrefillChunk]:
        """Admit within the budget, then pick one chunk to run.

        Args:
            budget: Tokens left after the decode step is paid for.

        Returns:
            Empty when the budget cannot cover a block (a decode-only step), a
            whole batch of chunks when those fit as a group, otherwise the least
            recently served prefilling request's chunk.
        """
        if budget < self.block_size:
            return []
        group = self._admit(budget)
        if group:
            return group
        prefilling = [s for s in self.running if s.status is SequenceStatus.PREFILLING]
        if not prefilling:
            return []
        # Round-robin: the request that has waited longest goes next, and the
        # stamp below is what makes the *other* one go next step. Without it the
        # deque order would let the oldest admission hog every step.
        seq = min(prefilling, key=lambda s: (s.last_scheduled_step, s.seq_id))
        remaining = len(seq.token_ids) - seq.prefill_cursor
        tokens = min(self.prefill_quantum, remaining, budget)
        if tokens <= 0:
            return []
        seq.last_scheduled_step = self.step_idx
        return [PrefillChunk(seq, tokens)]

    def _begin(self, seq: "Sequence") -> None:
        """Move a sequence from the waiting queue into the prefilling state.

        Admission does not size the prompt against the budget: a long prompt is
        advanced a quantum at a time, so requiring it to fit would reintroduce the
        head-of-line block chunking is here to remove.

        Args:
            seq: Sequence just popped from ``waiting``. A preempted one is put
                back in the admission flow first, so the runner treats it like a
                fresh request.
        """
        if seq.status is SequenceStatus.PREEMPTED:
            seq.resume()
        seq.status = SequenceStatus.PREFILLING
        seq.last_scheduled_step = self.step_idx

    def _admit(self, budget: int) -> list[PrefillChunk]:
        """Admit waiting requests while rows and budget allow.

        A batch is only worth grouping when every admitted prompt fits the
        remaining budget as a whole: the grouped forward reads the weights once,
        which is the only thing it saves, and a prompt split across chunks cannot
        take part in it.

        The *first* admission ignores the budget. A prompt longer than the whole
        budget must still start, and with chunking it starts by taking one
        quantum rather than by being refused.

        Args:
            budget: Tokens available for the whole prefill phase.

        Returns:
            The chunks to run as one grouped forward, or an empty list when the
            batch is not groupable. Admitted sequences are in ``running`` either
            way, so a non-grouped admission still becomes a chunk candidate.
        """
        admitted: list[PrefillChunk] = []
        spare = budget
        while self.waiting and len(self.running) < self.max_num_seqs:
            nxt = self.waiting[0]
            if self.step_idx - nxt.preempted_step <= PREEMPT_COOLDOWN:
                # Just evicted: admitting it now would re-take the blocks the
                # eviction was meant to free, and the two requests would trade
                # places every step without either finishing.
                break
            need = len(nxt.token_ids) - nxt.prefill_cursor
            if need <= 0:
                # An empty prompt can never make progress; leaving it at the head
                # would stall the queue forever.
                self.waiting.popleft()
                continue
            chunk = min(self.prefill_quantum, need, spare)
            if not self._has_room(nxt, chunk):
                # The blocks for this chunk are not there and the running set is
                # what holds them. Admitting anyway means being evicted again on
                # the same step, so wait for a completion to return them.
                break
            if admitted and need > spare:
                break
            self.waiting.popleft()
            self._begin(nxt)
            self.running.append(nxt)
            admitted.append(PrefillChunk(nxt, need))
            spare -= need
        if len(admitted) >= 2 and all(c.tokens <= budget for c in admitted):
            return admitted
        return []

    def _has_room(self, seq: "Sequence", chunk: int) -> bool:
        """Whether the pool can hold ``seq``'s first chunk right now.

        Admission has to ask: a request admitted without the blocks to start is
        found unrunnable on the same step and either evicted or skipped, and
        either way it goes back to the queue having taken a slot from someone who
        could have run. The prefix match may hand it shared blocks, so the answer
        is deliberately optimistic -- it counts only the fresh tail -- and the
        engine still checks again when it actually grows the table.

        Args:
            seq: Candidate for admission.
            chunk: Tokens its first chunk covers.

        Returns:
            ``True`` when the fresh blocks for that chunk are free. Without a
            runner (a scheduler-only test) nothing is allocated per request, so
            the answer is always yes.
        """
        runner = self.runner
        if runner is None or not self.chunked:
            return True
        target = min(seq.prefill_cursor + chunk, seq.num_tokens)
        # A full pool is mostly cached prefix, which eviction can return; ask for
        # the room through the evicting path so a request does not wait behind
        # blocks the tree could give up.
        return runner.reserve_for(target) >= runner.blocks_for(target)

    def expire(self, now: float | None = None) -> list["Sequence"]:
        """Retire every running request whose wall-clock budget ran out.

        Args:
            now: Instant to judge against; ``None`` reads the clock once so the
                whole pass is consistent.

        Returns:
            The retired sequences, already passed to :meth:`finish`.
        """
        now = time.time() if now is None else now
        expired = [s for s in self.running if s.over_budget(now)]
        for seq in expired:
            self.finish(seq, "deadline")
        return expired

    def preempt_victim(self, protected: set["Sequence"] | frozenset = frozenset()
                       ) -> "Sequence | None":
        """Pick the request to evict when the pool cannot satisfy an allocation.

        The policy trades fairness for explainability on purpose: the pool is
        what must not deadlock, so a request that is only partway through its
        prompt is evicted before one that is already generating, the biggest
        holder goes first because that is where the blocks are, and the score
        discounts a request that has been evicted before so one long prompt
        cannot be picked on every step while the rest of the queue runs.

        Args:
            protected: Sequences that must not be evicted, normally the caller
                itself plus the sequences already selected to decode this step.

        Returns:
            The victim, or ``None`` when nothing may be evicted -- the caller
            must then let its own request wait rather than force the allocation.
        """
        best = None
        best_key = None
        for seq in self.running:
            if seq in protected or seq.num_blocks_owned <= 0:
                continue
            if self.step_idx - seq.last_scheduled_step < PREEMPT_MIN_GAP:
                continue
            score = (seq.num_blocks_owned * 4
                     + (self.step_idx - seq.last_scheduled_step)
                     - seq.preempt_count * 8)
            # A prefilling request is the better victim: it has no generated
            # tokens that a restart would have to reproduce token for token.
            key = (seq.status is SequenceStatus.PREFILLING, score, -seq.seq_id)
            if best_key is None or key > best_key:
                best, best_key = seq, key
        return best

    def park(self, seq: "Sequence") -> None:
        """Take a preempted request out of the running set and requeue it.

        Args:
            seq: Sequence the engine has already stripped of its blocks and batch
                row. It is put back at the *head* of the waiting queue: a
                preempted request has already waited once, and letting a fresh
                arrival overtake it would make the recompute cost unbounded.
        """
        try:
            self.running.remove(seq)
        except ValueError:
            pass
        seq.park_preempted()
        seq.preempted_step = self.step_idx
        self.waiting.appendleft(seq)

    def finish(self, seq: "Sequence", reason: str = "eos") -> None:
        """Retire one request.

        Idempotent on purpose: the engine retires from both the decode loop and
        the expiry pass, and a request can be finished twice without corrupting
        ``n_finished``.

        Args:
            seq: Request to retire.
            reason: Value forwarded to the client as ``finish_reason``.
        """
        seq.finish(reason)
        for queue in (self.running, self.waiting):
            try:
                queue.remove(seq)
            except ValueError:
                continue
        self.n_finished += 1

    # Accounting helpers used by the engine's streaming loop.
    @staticmethod
    def timed_out(seq: "Sequence", now: float | None = None) -> bool:
        """Whether one sequence has exceeded its deadline.

        Args:
            seq: Request to judge.
            now: Instant to judge against; ``None`` reads the clock.

        Returns:
            ``True`` when the request has a deadline and it has passed.
        """
        return seq.over_budget(time.time() if now is None else now)
