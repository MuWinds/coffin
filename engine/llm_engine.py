"""The engine, assembled: model + runner + scheduler + block manager.

Module: ``coffin/`` (top layer). This is what the runtime service talks to; the
entry point owns the socket, tokenizer and response protocol. The engine owns
the request lifecycle, the block pool and the model's captured graphs.

    LLMEngine
      ├── ModelRunner      how one step executes (graph replay / paged write)
      ├── Scheduler        which sequences are in the next step
      └── BlockManager     the paged pool, its refcounts (and the radix tree
                           when prefix reuse is on)

Constraints:
    * One :meth:`step` is one iteration of the serving loop: new requests are
      prefilled (on the staging row) and every running sequence with KV is
      decoded in the same batched forward. :meth:`generate` and :meth:`stream`
      are thin loops around it, so the loop itself holds no request state.
    * A serving thread may call :meth:`add_request` between steps, never during
      one: ``step`` mutates the scheduler's running list and no lock is taken.
    * A finished sequence is released by whoever detects the finish
      (:meth:`_commit`, :meth:`_spec_commit`, the expiry pass). Missing one call
      leaks a batch row, and the next admission then fails outright.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Iterator

import torch

from .block_manager import BlockManager
from .config import EngineConfig
from .model_runner import ModelRunner
from .scheduler import Scheduler
from .sequence import SamplingParams, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .prefix_engine import PrefixCacheEngine


class LLMEngine:
    """One loaded checkpoint, driven request by request."""

    def __init__(self, cfg: EngineConfig, model) -> None:
        """Assemble the layers and provision the KV pool.

        Args:
            cfg: Engine configuration; decides batching, prefix caching and the
                pool size.
            model: Loaded ``engine_api`` model. Paging is enabled on it here, so
                the caller must not allocate KV before this returns.
        """
        self.cfg = cfg
        self.model = model
        self.runner = ModelRunner(model, cfg)
        self.scheduler = Scheduler(cfg, self.runner)
        self.batching = cfg.max_num_seqs > 1
        self.pc: "PrefixCacheEngine | None" = None
        self.bm: "BlockManager | None" = None

        # One pool serves both modes. The radix tree is attached whenever prefix
        # caching is on, including the batched mode, where the runner matches a
        # prefix when it admits a sequence and inserts it back after its prefill.
        # Two pools would give the two modes separate block id spaces.
        if cfg.prefix_cache or self.batching:
            self.bm = BlockManager.attach(
                model, cfg.num_blocks, cfg.block_size, cfg.snapshots,
                with_radix=cfg.prefix_cache)
            self.runner.attach_blocks(self.bm)
        if cfg.prefix_cache and not self.batching:
            from .prefix_engine import PrefixCacheEngine

            self.pc = PrefixCacheEngine(
                model, num_blocks=cfg.num_blocks, block_size=cfg.block_size,
                max_state_snapshots=cfg.snapshots,
                snapshot_interval=cfg.snapshot_interval, bm=self.bm,
                graph=cfg.graph)
            self.runner.attach_prefix_cache(self.pc)
            # ``capture()`` warms the batched prefill path, which writes KV
            # through the slot map. Reserving a block the pool can never hand out
            # keeps that warm-up from scribbling over a cached prefix: the write
            # needs a slot, and this one is not part of any prefix.
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
        """Capture the decode graph(s) and warm up the paths a step will use.

        Must run once before the first :meth:`step`. With ``cfg.graph`` on (the
        default) this records the decode forward into a device graph, which is what
        makes a step a replay instead of a re-trace. With it off the engine stays
        eager, which is slower per token but is the only safe choice on a backend
        whose replay is not trusted: a graph that replays stale outputs is a
        silent wrong answer rather than a crash.

        Batching and speculation are graph-only in this engine -- both replay a
        per-bucket graph -- so the caller must have cleared them before arriving
        here when capture is off; :class:`~coffin.serving.runtime_service.Server`
        does that from ``--no-graph``.

        Raises:
            RuntimeError: Capture is off but batching or speculation is still
                configured. Both would replay a graph that was never captured, so
                this fails here instead of at the first decode step.
            RuntimeError: Speculative decoding is configured but the checkpoint
                has no MTP head. Failing at build time beats producing subtly
                different tokens later.
        """
        t0 = time.time()
        if not self.cfg.graph:
            if self.batching or self.cfg.spec_k:
                raise RuntimeError(
                    "graph capture is off but batching/speculation is configured; "
                    "both replay a captured graph, so they need caps.graph")
            self.model.reset()
        elif self.batching:
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
                # The verify graph is captured for k+1 tokens: the pending token
                # plus the k drafts the drafter proposes.
                self.model.capture_verify(self.cfg.spec_k + 1)
            self.model.reset()
        self.build_s = time.time() - t0

    def add_request(self, token_ids: list[int],
                    params: SamplingParams | None = None,
                    mm=None) -> Sequence:
        """Queue one prompt.

        The image payload rides on the sequence so the runner, the block manager
        and the radix tree all read the same object: the tree needs
        ``mm.digest`` / ``mm.spans`` to salt its block keys, and the model needs
        the embeddings. A parallel side table would have to be kept in sync
        across three layers for no benefit.

        Args:
            token_ids: Prompt tokens, already rendered by the chat template.
            params: Decoding policy; ``None`` uses greedy defaults with the
                engine-wide stop ids.
            mm: ``mm_input.MMInput`` for an image prompt, ``None`` for text.

        Returns:
            The queued sequence. The caller keeps it to observe completion and
            to read ``stop_reason`` afterwards.
        """
        params = params or SamplingParams()
        if not params.eos:
            params.eos = set(self.eos)
        seq = Sequence(token_ids, params)
        seq.mm = mm
        self.scheduler.add(seq)
        self.n_requests += 1
        self.n_prompt_tokens += len(seq)
        return seq

    def step(self) -> list[tuple[Sequence, int | None, bool]]:
        """Run one serving iteration.

        Returns:
            ``(seq, token, finished)`` events. The token is ``None`` when the
            event only reports retirement, which happens on a deadline — there
            is no token to report. A finishing step still reports the token it
            produced, so a stream client sees the last token before the done
            line rather than an empty tail.
        """
        expired = self.scheduler.expire()
        events: list[tuple[Sequence, int | None, bool]] = [
            (seq, None, True) for seq in expired]
        for seq in expired:
            self._release(seq)
        decode, prefill = self.scheduler.schedule()
        if prefill:
            # Several prompts admitted in one step share a grouped prefill when
            # there is no radix tree to populate. With prefix caching on they run
            # one by one, because the grouped forward would still have to match
            # and insert each sequence separately.
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
                # Speculation only pays once the KV read dominates. Below the
                # context threshold a plain batched step is cheaper, so one step
                # can mix both paths: rows are independent, and a mixed step
                # costs no more than the two paths run back to back.
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

        Speculation only changes speed: a draft is accepted only when it equals
        the model's own argmax, so the tokens emitted here are exactly the greedy
        tokens a plain decode step would have produced.

        Args:
            seqs: Sequences to advance. All must decode at temperature 0; a
                stochastic request cannot be verified against an argmax.

        Returns:
            ``(seq, token, finished)`` events, in the engine's usual shape.
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
        """Append one sampled token and apply the stop conditions.

        Args:
            seq: Sequence that produced ``tok``.
            tok: Sampled token id.

        Returns:
            Events for the caller. On a stop the sequence is finished and
            released here, so no caller has to remember to do it.
        """
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
        """Return a finished batched sequence's blocks and its batch row.

        Single-stream and prefix-cache engines own their blocks through
        ``PrefixCacheEngine``, so this is a no-op for them; only the batched path
        allocates per sequence.

        Args:
            seq: Finished sequence. Its row must come back before the next
                admission looks for a free one.
        """
        if self.batching and self.bm is not None:
            self.bm.release(seq)
            self.runner.release_row(seq)

    # Convenience loops: one request at a time, used by tests and run_chat.
    def stream(self, seq: Sequence) -> Iterator[int]:
        """Yield the tokens ``seq`` generates while stepping the engine.

        Args:
            seq: Request previously queued with :meth:`add_request`.

        Yields:
            Token ids as they are produced. Events for other sequences are
            filtered out, so a caller driving several requests still sees only
            its own stream.
        """
        while not seq.is_finished:
            for other, tok, _done in self.step():
                if other is seq and tok is not None:
                    yield tok
            if not self.scheduler.has_work():
                break

    @torch.inference_mode()
    def generate(self, token_ids: list[int],
                 params: SamplingParams | None = None,
                 mm=None) -> dict:
        """Run one request to completion.

        Args:
            token_ids: Prompt tokens.
            params: Decoding policy; ``None`` uses greedy defaults.
            mm: ``mm_input.MMInput`` for an image prompt.

        Returns:
            ``{"tokens", "prefill_s", "decode_s", "matched", "stop_reason"}``.
            The time split is computed here because it is what the benchmark
            numbers in ``NOTES.md`` report; each caller would otherwise redo it
            slightly differently.
        """
        seq = self.add_request(token_ids, params, mm=mm)
        t0 = time.time()
        out = list(self.stream(seq))
        total = time.time() - t0
        pre = seq.prefill_s
        return {"tokens": out, "prefill_s": pre, "decode_s": max(total - pre, 0.0),
                "matched": seq.matched, "stop_reason": seq.stop_reason}

    def reset(self) -> None:
        """Drop the active sequence's blocks and rewind every piece of state.

        Single-stream reuse: the model keeps its position and KV between
        requests, so a new request must clear both or it would read the previous
        prompt's prefix.
        """
        if self.pc is not None:
            self.pc.release()
        self.model.reset()

    def release(self) -> None:
        """Drop the prefix cache's reference on the engine's live sequence."""
        if self.pc is not None:
            self.pc.release()

    def clear_prefix_cache(self) -> None:
        """Drop every cached prefix.

        Only valid while the engine is idle: the radix tree refcounts blocks that
        running sequences are actively writing into.

        Raises:
            RuntimeError: A sequence is waiting or running. Clearing then would
                free blocks still in use, so this refuses instead of corrupting
                the pool.
        """
        if self.scheduler.running or self.scheduler.waiting:
            raise RuntimeError("clear_prefix_cache() needs an idle engine")
        if self.bm is not None:
            self.bm.clear_prefix_cache()

    def stats(self) -> dict:
        """Engine and cache counters, as reported by the runtime ``ping``.

        Returns:
            Request and token totals plus the block/prefix counters when a pool
            exists. The key names are what the runtime serialises, so they are
            part of the API surface rather than free-form labels.
        """
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
