"""The engine, assembled: model + runner + scheduler + block manager.

This is the module a server talks to.  It owns the request lifecycle, the block
pool and the model's captured graphs; ``serve.py`` is left with the socket, the
tokenizer and the response protocol.

    LLMEngine
      ├── ModelRunner      how one step executes (graph replay / paged write)
      ├── Scheduler        which sequences are in the next step
      └── BlockManager     the paged pool, its refcounts (and the radix tree
                           when single-stream prefix reuse is on)

One :meth:`step` is one iteration of the serving loop: at most one new request
is prefilled (on the staging row) and every running sequence with KV is decoded
in the same batched forward.  ``step`` returns the tokens it produced, so both
the blocking :meth:`generate` and the streaming :meth:`stream` are thin loops
around it, and a serving thread can interleave several requests by calling
:meth:`add_request` while stepping.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Iterator

import torch

from block_manager import BlockManager
from config import EngineConfig
from model_runner import ModelRunner
from scheduler import Scheduler
from sequence import SamplingParams, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    from prefix_engine import PrefixCacheEngine


class LLMEngine:
    """One loaded checkpoint, driven request by request."""

    def __init__(self, cfg: EngineConfig, model) -> None:
        self.cfg = cfg
        self.model = model
        self.runner = ModelRunner(model, cfg)
        self.scheduler = Scheduler(cfg, self.runner)
        self.batching = cfg.max_num_seqs > 1
        self.pc: "PrefixCacheEngine | None" = None
        self.bm: "BlockManager | None" = None

        # One pool serves both modes.  The radix tree is per-sequence state, so it
        # is attached whenever prefix caching is on -- including the batched mode,
        # where the runner matches a prefix when it admits a sequence and inserts
        # it back after its prefill.
        if cfg.prefix_cache or self.batching:
            self.bm = BlockManager.attach(
                model, cfg.num_blocks, cfg.block_size, cfg.snapshots,
                with_radix=cfg.prefix_cache)
            self.runner.attach_blocks(self.bm)
        if cfg.prefix_cache and not self.batching:
            from prefix_engine import PrefixCacheEngine

            self.pc = PrefixCacheEngine(
                model, num_blocks=cfg.num_blocks, block_size=cfg.block_size,
                max_state_snapshots=cfg.snapshots,
                snapshot_interval=cfg.snapshot_interval, bm=self.bm)
            self.runner.attach_prefix_cache(self.pc)
            # ``capture()`` warms the batched prefill path, which writes KV through
            # the slot map.  Reserve a block the pool can never hand out, so a
            # later capture cannot scribble over a cached prefix.
            self._scratch = self.pc.pool.alloc()
            self.pc.pool.assign(self._scratch, 0)
        else:
            self._scratch = None

        self.eos: set[int] = set()
        self.n_requests = 0
        self.n_prompt_tokens = 0
        self.n_generated = 0
        self.load_s = 0.0
        self.build_s = 0.0
        #: Prompt tokens the last prefill adopted from the radix tree.
        self.last_matched = 0

    def build(self) -> None:
        """Capture the decode graph(s) and warm up the paths a step will use."""
        t0 = time.time()
        if self.batching:
            self.runner.enable_batching(self.cfg.max_num_seqs)
            if self.cfg.spec_k and (self.cfg.spec_batch or self.cfg.spec_sparse_k):
                if self.model.mtp is None:
                    raise RuntimeError("--spec-k needs the MTP head in the checkpoint")
                self.runner.enable_spec(self.cfg.spec_k, self.cfg.spec_sparse_k)
        else:
            if not self.model.graph_ready:
                self.model.capture()
            if self.cfg.spec_k:
                if self.model.mtp is None:
                    raise RuntimeError("--spec-k needs the MTP head in the checkpoint")
                self.model.capture_verify(self.cfg.spec_k + 1)
            self.model.reset()
        self.build_s = time.time() - t0

    def add_request(self, token_ids: list[int],
                    params: SamplingParams | None = None) -> Sequence:
        params = params or SamplingParams()
        if not params.eos:
            params.eos = set(self.eos)
        seq = Sequence(token_ids, params)
        self.scheduler.add(seq)
        self.n_requests += 1
        self.n_prompt_tokens += len(seq)
        return seq

    def step(self) -> list[tuple[Sequence, int | None, bool]]:
        """One serving iteration; returns ``(seq, token, finished)`` events.

        The token is ``None`` when the event only reports retirement (deadline).
        A finishing step still reports the token it produced, so a stream client
        sees the last token before the done line.
        """
        expired = self.scheduler.expire()
        events: list[tuple[Sequence, int | None, bool]] = [
            (seq, None, True) for seq in expired]
        for seq in expired:
            self._release(seq)
        decode, prefill = self.scheduler.schedule()
        if prefill:
            # Several prompts admitted in one step share a grouped prefill when
            # there is no radix tree to populate; with prefix caching on they run
            # one by one so each can be matched/inserted (phase 2b).
            if len(prefill) > 1 and self.runner.can_group_prefill():
                logits_list = self.runner.prefill_group(prefill)
                group_s = self.runner.last_prefill_s
                for seq, logits in zip(prefill, logits_list):
                    seq.prefill_s = group_s
                    tok = self.runner.sample(logits, seq.params.temperature)
                    events += self._commit(seq, tok)
            else:
                for seq in prefill:
                    logits = self.runner.prefill(seq)
                    seq.prefill_s = self.runner.last_prefill_s
                    self.last_matched = seq.matched
                    tok = self.runner.sample(logits, seq.params.temperature)
                    events += self._commit(seq, tok)
        if decode:
            use_spec = (self.runner.spec
                        and all(s.params.temperature == 0 for s in decode))
            if use_spec:
                # Speculation only pays once the KV read dominates: below the
                # context threshold a plain batched decode step is cheaper, so the
                # step can mix both paths (rows are independent).
                thr = self.cfg.spec_min_ctx
                spec_seqs = ([s for s in decode if s.num_cached_tokens >= thr]
                             if thr else list(decode))
                plain = [s for s in decode if s not in spec_seqs]
                if spec_seqs:
                    events += self._spec_commit(spec_seqs)
                if plain:
                    tokens = self.runner.decode(plain)
                    for seq, tok in zip(plain, tokens):
                        events += self._commit(seq, tok)
            else:
                tokens = self.runner.decode(decode)
                for seq, tok in zip(decode, tokens):
                    events += self._commit(seq, tok)
        return events

    def _spec_commit(self, seqs: list[Sequence]) -> list[tuple[Sequence, int | None, bool]]:
        """Commit one batch-friendly MTP step and sample the next token per row.

        Speculation only changes speed: the model accepts a draft only when it
        equals its own argmax, so the tokens emitted here are exactly the greedy
        tokens a plain decode step would have produced.
        """
        result = self.runner.spec_step(seqs)
        if result is None:                     # drafter window exceeded: plain step
            tokens = self.runner.decode(seqs)
            events: list[tuple[Sequence, int | None, bool]] = []
            for seq, tok in zip(seqs, tokens):
                events += self._commit(seq, tok)
            return events
        accepted, logits = result
        events = []
        for seq, acc, row_logits in zip(seqs, accepted, logits):
            finished = False
            # ``acc[0]`` is the pending token the engine already appended.
            for tok in acc[1:]:
                if tok in seq.params.eos and not seq.params.ignore_eos:
                    self.scheduler.finish(seq, "eos")
                    self._release(seq)
                    events.append((seq, None, True))
                    finished = True
                    break
                seq.append_token(tok)
                self.n_generated += 1
                if seq.num_completion_tokens >= seq.params.max_tokens:
                    self.scheduler.finish(seq, "length")
                    self._release(seq)
                    events.append((seq, tok, True))
                    finished = True
                    break
                events.append((seq, tok, False))
            if finished:
                continue
            nxt = self.runner.sample(row_logits, seq.params.temperature)
            events += self._commit(seq, nxt)
        return events

    def _commit(self, seq: Sequence, tok: int) -> list[tuple[Sequence, int | None, bool]]:
        params = seq.params
        if tok in params.eos and not params.ignore_eos:
            self.scheduler.finish(seq, "eos")
            self._release(seq)
            return [(seq, None, True)]
        seq.append_token(tok)
        self.n_generated += 1
        if seq.num_completion_tokens >= params.max_tokens:
            self.scheduler.finish(seq, "length")
            self._release(seq)
            return [(seq, tok, True)]
        return [(seq, tok, False)]

    def _release(self, seq: Sequence) -> None:
        """Return a finished batched sequence's blocks and its batch row."""
        if self.batching and self.bm is not None:
            self.bm.release(seq)
            self.runner.release_row(seq)

    # convenience loops (single request at a time, used by tests and run_chat)
    def stream(self, seq: Sequence) -> Iterator[int]:
        """Yield the tokens ``seq`` generates while stepping the engine."""
        while not seq.is_finished:
            for other, tok, _done in self.step():
                if other is seq and tok is not None:
                    yield tok
            if not self.scheduler.has_work():
                break

    @torch.inference_mode()
    def generate(self, token_ids: list[int],
                 params: SamplingParams | None = None) -> dict:
        """One full request; returns tokens plus the prefill/decode split."""
        seq = self.add_request(token_ids, params)
        t0 = time.time()
        out = list(self.stream(seq))
        total = time.time() - t0
        pre = seq.prefill_s
        return {"tokens": out, "prefill_s": pre, "decode_s": max(total - pre, 0.0),
                "matched": seq.matched, "stop_reason": seq.stop_reason}

    def reset(self) -> None:
        """Drop the active sequence's blocks and rewind every piece of state."""
        if self.pc is not None:
            self.pc.release()
        self.model.reset()

    def release(self) -> None:
        if self.pc is not None:
            self.pc.release()

    def clear_prefix_cache(self) -> None:
        """Drop every cached prefix; only valid while the engine is idle."""
        if self.scheduler.running or self.scheduler.waiting:
            raise RuntimeError("clear_prefix_cache() needs an idle engine")
        if self.bm is not None:
            self.bm.clear_prefix_cache()

    def stats(self) -> dict:
        out = {
            "requests": self.n_requests,
            "prompt_tokens": self.n_prompt_tokens,
            "generated_tokens": self.n_generated,
            "finished": self.scheduler.n_finished,
            "waiting": len(self.scheduler.waiting),
            "running": len(self.scheduler.running),
            "batching": self.batching,
        }
        if self.pc is not None:
            out["prefix"] = self.pc.stats()
        elif self.bm is not None and self.bm.radix is not None:
            out["prefix"] = self.bm.stats()
        if self.bm is not None:
            out["blocks_used"] = self.bm.num_used
            out["blocks_total"] = self.bm.num_blocks
        return out

    @property
    def prefix(self) -> "PrefixCacheEngine | None":
        return self.pc
