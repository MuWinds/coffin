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
from .sequence import SamplingParams, Sequence, SequenceStatus

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .prefix_engine import PrefixCacheEngine


#: Steps the loop may produce no event at all before the engine reports the pool
#: as too small. Two deferred steps are normal (a prefill chunk that only advanced
#: its cursor); a long run of them cannot resolve itself, because every live
#: sequence is holding the blocks the others are waiting for.
NO_PROGRESS_LIMIT = 200


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
        #: Recompute preemptions so far. A rising count with a healthy pool means
        #: the pool is undersized for the concurrency the scheduler admits.
        self.n_preempted = 0
        #: Consecutive steps that produced no event; see :meth:`_check_progress`.
        self._stalled_steps = 0
        #: Sequences already evicted this step. A sequence cannot be a second
        #: victim in the same step, and it also cannot be the one whose growth is
        #: being served.
        self._preempted_this_step: set[Sequence] = set()
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

        Batching survives capture being off: the runner calls the model's eager
        batched entry instead of replaying a bucket graph, which stages the same
        buffers and runs the same body. Speculation does not -- a verify step is
        a captured replay by construction -- so it is what the graph-off branch
        rejects.

        Raises:
            RuntimeError: Capture is off but speculative decoding is configured.
                A verify step replays a graph that was never captured, so this
                fails here instead of at the first speculative step.
            RuntimeError: Speculative decoding is configured but the checkpoint
                has no MTP head. Failing at build time beats producing subtly
                different tokens later.
        """
        t0 = time.time()
        if self.batching and self.bm is not None:
            # Paged batched serving: one pool and one row set, decoded either by a
            # captured bucket graph or by the eager body of the same bucket. This
            # branch runs before the capture check: with capture off no graph is
            # ever recorded, and ``_decode_batch`` takes the eager entry.
            self.runner.enable_batching(self.cfg.max_num_seqs)
            if self.cfg.spec_k and (self.cfg.spec_batch or self.cfg.spec_sparse_k):
                if not self.cfg.graph:
                    raise RuntimeError(
                        "--spec-k needs graph capture; a verify step is a captured "
                        "replay by construction")
                if self.model.mtp is None:
                    raise RuntimeError("--spec-k needs the MTP head in the checkpoint")
                self.runner.enable_spec(self.cfg.spec_k, self.cfg.spec_sparse_k)
        elif not self.cfg.graph:
            if self.cfg.spec_k:
                raise RuntimeError(
                    "graph capture is off but speculation is configured; a verify "
                    "step is a captured replay, so it needs caps.graph")
            # Speculation is rejected above, so no drafter KV this prefill writes can
            # ever be read. Feeding it is one full-attention layer plus five projections
            # over the whole prompt: 214.8 ms of a 962.5 ms prefill at n=2048 on the part
            # this was measured on (PERF_COREX.md §13.2a). Cleared here rather than in
            # ``Qwen35.__init__`` because a caller can also drive ``spec_generate``
            # directly, which replays the eager substitute captures and does read that KV.
            if self.model.mtp is not None:
                self.model.mtp_prefill = False
            self.model.reset()
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
        decode, chunks = self.scheduler.schedule()
        # Only the requester is off limits: a victim has to be somebody, and
        # preempting a request that this step is about to advance is exactly what
        # makes room for the one that cannot grow. Its decode has not run yet, so
        # nothing is thrown away.
        self._preempted_this_step = set()
        # Decode first: a prompt must not delay a request that is already
        # producing. This pass may also evict, so the chunk list is re-checked
        # afterwards -- a chunk whose sequence lost its blocks must not run.
        decode = self._make_room_for_decode(decode)
        if decode:
            events += self._run_decode(decode)
        chunks = [c for c in chunks
                  if c.seq.status is SequenceStatus.PREFILLING]
        if chunks:
            events += self._run_prefill(chunks)
        self._check_progress(events)
        return events

    def _check_progress(self, events: list[tuple[Sequence, int | None, bool]]) -> None:
        """Fail loudly when the pool is too small to make progress at all.

        Admission and block growth can both defer a request for a step, which is
        the right answer while some other request is finishing and will return its
        blocks. If nothing at all has moved for many steps in a row, no completion
        is coming: every live sequence is waiting for a block the others hold, and
        the pool is smaller than the working set. Raising here turns that into a
        reported error instead of a hung serving loop.

        Args:
            events: Events this step produced. Any token or retirement is progress.

        Raises:
            RuntimeError: The engine has made no progress for
                :data:`NO_PROGRESS_LIMIT` consecutive steps.
        """
        if events:
            self._stalled_steps = 0
            return
        self._stalled_steps += 1
        if self._stalled_steps < NO_PROGRESS_LIMIT:
            return
        self._stalled_steps = 0
        raise RuntimeError(
            f"KV pool too small: no progress for {NO_PROGRESS_LIMIT} steps with "
            f"{len(self.scheduler.running)} running and "
            f"{self.bm.num_free if self.bm is not None else 0} free blocks of "
            f"{self.bm.num_blocks if self.bm is not None else 0}; raise --num-blocks "
            "or lower --max-num-seqs")

    def _run_prefill(self, chunks) -> list[tuple[Sequence, int | None, bool]]:
        """Execute the step's prefill work and sample what completes it.

        A batch of chunks means every admitted prompt fits the token budget as a
        whole, which is the one case where one grouped forward is worth it. A
        single chunk is one request advancing by at most ``prefill_quantum``
        tokens; if it does not finish the prompt the sequence stays ``PREFILLING``
        and the next step picks it up.

        Args:
            chunks: The scheduler's chunks for this step.

        Returns:
            ``(seq, token, finished)`` events, in the engine's usual shape. A
            chunk that only advanced the cursor emits nothing.
        """
        runner = self.runner
        if len(chunks) > 1 and runner.can_group_prefill():
            seqs = [chunk.seq for chunk in chunks]
            # A group is only formed from prompts that fit the budget whole, so
            # every one of them is about to be finished by this forward.
            for chunk in chunks:
                self._ensure_capacity(chunk.seq,
                                      chunk.seq.prefill_cursor + chunk.tokens)
            logits_list = runner.prefill_group(seqs)
            group_s = runner.last_prefill_s
            events: list[tuple[Sequence, int | None, bool]] = []
            for seq, logits in zip(seqs, logits_list):
                seq.prefill_s = group_s
                tok = runner.sample(logits, seq.params.temperature)
                events += self._commit(seq, tok)
            return events

        events = []
        for chunk in chunks:
            seq = chunk.seq
            if not runner.can_chunk_prefill():
                # Single-stream and prefix-cache engines prefill in one call and
                # keep their own block bookkeeping, so the token budget does not
                # apply to them. The whole prompt is written by this call, which
                # is why the sequence becomes ``RUNNING`` here rather than through
                # ``finish_prefill``; without it the next step would not decode it.
                logits = runner.prefill(seq)
                seq.prefill_s = runner.last_prefill_s
                self.last_matched = seq.matched
                seq.status = SequenceStatus.RUNNING
                tok = runner.sample(logits, seq.params.temperature)
                events += self._commit(seq, tok)
                continue
            # Admission first: matching the radix prefix claims the blocks it can
            # share, which is exactly the part the capacity check then does not
            # have to find room for.
            if not seq.prefill_started:
                self.runner.begin_prefill(seq)
            target = seq.prefill_cursor + chunk.tokens
            # Blocks are checked here rather than inside the forward: growing a
            # table can evict, and eviction must run before the request owns the
            # blocks it is asking for.
            if not self._ensure_capacity(seq, target):
                # Nothing left to reclaim: leave the request prefilling and let a
                # later step retry. Failing it would turn a full pool into a
                # dropped client.
                continue
            logits = runner.prefill_chunk(seq, chunk.tokens)
            seq.prefill_s = runner.last_prefill_s
            if logits is None:
                # The prompt is not finished: nothing to sample, and the sequence
                # stays PREFILLING for a later step.
                continue
            self.last_matched = seq.matched
            # ``prefill_chunk`` records what it produced; reading it back here is
            # what makes the field the single record of "the logits the finished
            # prompt ended on", rather than a second copy of the return value.
            tok = runner.sample(seq.prefill_logits, seq.params.temperature)
            events += self._commit(seq, tok)
        return events

    def _run_decode(self, decode: list[Sequence]
                    ) -> list[tuple[Sequence, int | None, bool]]:
        """Advance every decodable sequence by one token.

        Args:
            decode: Sequences whose prompt is complete.

        Returns:
            ``(seq, token, finished)`` events, in the engine's usual shape.
        """
        use_spec = (self.runner.spec
                    and all(s.params.temperature == 0 for s in decode))
        events: list[tuple[Sequence, int | None, bool]] = []
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

    def _ensure_capacity(self, seq: Sequence, tokens: int) -> bool:
        """Make the pool able to hold ``seq``'s next ``tokens`` positions.

        Eviction comes first: the cache holds prefixes nobody is generating from,
        and dropping one costs a later recompute rather than a live request. Only
        when it cannot free enough does a running request become a preemption
        candidate.

        Args:
            seq: Sequence about to grow. It is the one sequence this step will not
                preempt: it is the request that needs the room.
            tokens: Tokens the forward covers.

        Returns:
            ``True`` when the blocks are available. ``False`` means nothing left to
            reclaim; the caller then leaves ``seq`` for a later step instead of
            evicting work it cannot replace.
        """
        bm = self.bm
        if bm is None:
            return True
        bs = bm.block_size
        # A table of `k` blocks addresses `k*bs` positions, and the positions
        # already written inside it are paid for; only the difference has to be
        # allocated.
        held = len(seq.block_table) * bs
        need = max(0, (tokens - held + bs - 1) // bs)
        if need <= 0:
            return True
        bm.reserve(need)
        # The eviction above may have freed less than asked, so preempt one
        # request at a time and re-check. Preempting a sequence whose blocks the
        # tree took over only drops a reference; the tree's copy becomes the
        # eviction candidate, which is why the reserve is repeated rather than
        # assumed to have freed the whole request.
        for _ in range(len(self.scheduler.running) + 1):
            if bm.num_free >= need:
                return True
            victim = self.scheduler.preempt_victim({seq} | self._preempted_this_step)
            if victim is None:
                return False
            self._preempt(victim)
            bm.reserve(need)
        return bm.num_free >= need

    def _make_room_for_decode(self, decode: list[Sequence]) -> list[Sequence]:
        """Keep only the decoding sequences this step can actually advance.

        A decode step writes exactly one new position, so a sequence needs a free
        block only when its table is full. Dropping a row that cannot be served
        keeps the batched forward consistent: half the batch advanced and half
        not would leave the sequences out of step with their KV.

        Args:
            decode: Sequences the scheduler chose for this step.

        Returns:
            The rows that can run. Empty means this step is decode-free.
        """
        bm = self.bm
        if bm is None or self.runner.pc is not None:
            # The prefix-cache path owns its own write window and allocates
            # through ``SeqKV.reserve``, not through the block manager.
            return decode
        bs = bm.block_size
        runnable = []
        for seq in decode:
            if seq.status is not SequenceStatus.RUNNING:
                # An earlier iteration's preemption may have taken this sequence
                # out of the batch; running it now would allocate a fresh table
                # and decode on state that was reset to be recomputed.
                continue
            pos = seq.num_cached_tokens
            if len(seq.block_table) * bs >= pos + 1:
                runnable.append(seq)
            elif self._ensure_capacity(seq, pos + 1):
                # The room exists now, so the table is grown here rather than in
                # the forward: allocating inside the batched step would have no
                # way to evict or preempt once half the rows had already run.
                bm.ensure_blocks(seq, pos + 1)
                runnable.append(seq)
        return runnable

    def _preempt(self, seq: Sequence) -> None:
        """Evict one sequence and put it back in the admission queue.

        Args:
            seq: Victim chosen by the scheduler. Its blocks and row go back to the
                pool and its cursor is reset; the tokens it has already emitted
                stay on the sequence, so a recompute neither loses nor duplicates
                output.
        """
        self.runner.preempt(seq)
        self.scheduler.park(seq)
        self._preempted_this_step.add(seq)
        self.n_preempted += 1

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
            "preempted": self.n_preempted,
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
