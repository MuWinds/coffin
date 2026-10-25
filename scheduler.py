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

Admission policy:

* a prefill is a second, eager forward pass on the staging row, so admitting
  several at once is only worth it when they share one grouped prefill forward
  (one weight read instead of one per prompt);
* every running sequence that already owns KV is decoded in the same step, so a
  prefill does not stall the decode stream (continuous batching);
* a sequence keeps its batch row until it finishes (see ``model_runner``).

A finished sequence is retired by :meth:`Scheduler.finish`, which also drops it
from the running list.
"""

from __future__ import annotations

import time
from collections import deque
from typing import TYPE_CHECKING

from sequence import SequenceStatus

if TYPE_CHECKING:  # pragma: no cover - typing only
    from config import EngineConfig
    from model_runner import ModelRunner
    from sequence import Sequence


class Scheduler:
    """Admission and retirement of requests."""

    def __init__(self, cfg: "EngineConfig", runner: "ModelRunner | None" = None) -> None:
        """Build an empty scheduler bound to one engine.

        Args:
            cfg: Engine configuration; ``max_num_seqs`` and
                ``max_num_batched_tokens`` are the admission limits.
            runner: Optional back-reference used by the engine to compare the
                scheduler's view of a row against what the runner actually
                staged. ``None`` is valid for scheduler-only tests.
        """
        self.cfg = cfg
        self.runner = runner
        self.waiting: deque["Sequence"] = deque()
        self.running: deque["Sequence"] = deque()
        self.max_num_seqs = max(1, cfg.max_num_seqs)
        self.max_num_batched_tokens = cfg.max_num_batched_tokens
        self.eos: set[int] = set()
        self.n_finished = 0

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

    def schedule(self) -> tuple[list["Sequence"], list["Sequence"]]:
        """Pick the next step: ``(decode_seqs, prefill_seqs)``.

        Grouping the prefills of one step into a single forward is what makes a
        burst of prompts cost one weight read instead of one per prompt. The
        token budget stops the group before it turns one cheap prefill into an
        expensive one.

        A request admitted this step is not in ``decode_seqs``: it has no KV
        before its prefill runs, so it joins the decode list next step.

        Returns:
            ``(decode_seqs, prefill_seqs)``. Either list may be empty; both
            empty means there was nothing runnable.
        """
        prefill: list["Sequence"] = []
        budget = self.max_num_batched_tokens
        while self.waiting and len(self.running) < self.max_num_seqs:
            nxt = self.waiting[0]
            tokens = len(nxt)
            # The first admission ignores the budget: a prompt longer than the
            # whole budget must still make progress, or it would deadlock the
            # queue. Chunked prefill is not implemented on this path yet.
            if prefill and tokens > budget:
                break
            self.waiting.popleft()
            nxt.status = SequenceStatus.RUNNING
            self.running.append(nxt)
            prefill.append(nxt)
            budget -= tokens
            if budget <= 0:
                break
        decode = [s for s in self.running
                  if s.num_cached_tokens > 0 and s not in prefill]
        return decode, prefill

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
        try:
            self.running.remove(seq)
        except ValueError:
            pass
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
