"""Which sequences run in the next step.

nano-vllm's scheduler has two queues and one job: pack as many sequences as the
token budget allows into the next forward pass, preferring prefill over decode.
This module keeps that shape, with the batch row count as the capacity limit.

Admission policy (phase 1/2):

* at most one *new* request is admitted per step, and only while fewer than
  ``max_num_seqs`` sequences are running -- a prefill is a second, eager forward
  pass on the staging row and would otherwise double a step's latency;
* every running sequence that already owns KV is decoded in the same step, so a
  prefill does not stall the decode stream (continuous batching);
* a sequence keeps its batch row until it finishes; the runner re-packs rows
  before each batched step (moving recurrent state when a hole appears).

A finished sequence is retired by :meth:`finish`, which also frees its row.
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
        self.cfg = cfg
        self.runner = runner
        self.waiting: deque["Sequence"] = deque()
        self.running: deque["Sequence"] = deque()
        self.max_num_seqs = max(1, cfg.max_num_seqs)
        self.max_num_batched_tokens = cfg.max_num_batched_tokens
        self.eos: set[int] = set()
        self.n_finished = 0

    def add(self, seq: "Sequence") -> None:
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

        ``decode_seqs`` are the running sequences whose KV already exists and can
        go through the batched decode.  ``prefill_seqs`` are the new requests
        admitted this step: several of them share one grouped prefill forward as
        long as the token budget and the row capacity allow, which is what makes a
        burst of prompts cost one weight read instead of one per prompt.

        A request that was just admitted is not in ``decode_seqs`` for this step
        (it has no KV before its prefill runs); it joins the decode list next
        step.
        """
        prefill: list["Sequence"] = []
        budget = self.max_num_batched_tokens
        while self.waiting and len(self.running) < self.max_num_seqs:
            nxt = self.waiting[0]
            tokens = len(nxt)
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
        """Retire every running request whose wall-clock budget ran out."""
        now = time.time() if now is None else now
        expired = [s for s in self.running if s.over_budget(now)]
        for seq in expired:
            self.finish(seq, "deadline")
        return expired

    def finish(self, seq: "Sequence", reason: str = "eos") -> None:
        seq.finish(reason)
        try:
            self.running.remove(seq)
        except ValueError:
            pass
        self.n_finished += 1

    # accounting helpers used by the engine's streaming loop
    @staticmethod
    def timed_out(seq: "Sequence", now: float | None = None) -> bool:
        return seq.over_budget(time.time() if now is None else now)
