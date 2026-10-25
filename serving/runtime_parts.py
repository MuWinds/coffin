"""Runtime mixins for the resident inference service.

Module: ``coffin.serving``. Each mixin owns one lifecycle concern and operates
on the state created by ``runtime_service.Server``.

Constraints:
    * Mixins do not own process state; ``Server.__init__`` remains the single
      place that provisions model, tokenizer and request queues.
    * The methods are kept body-equivalent to the original service so protocol
      behavior and engine contracts do not change during the structural split.
"""

from __future__ import annotations

import gc
import importlib
import json
import os
import queue
import threading
import time
from pathlib import Path

import torch

import engine_api
import hip_ops
from engine import llm_engine
from engine.config import EngineConfig
from engine.sequence import SamplingParams
from utils.tool_call import parse_tool_calls, split_thinking

from .runtime_support import MMError, _decode_image



class RuntimeLifecycleMixin:

    def _build(self) -> None:
        """Create the model and the engine, then bring up the vision tower.

        Split out of ``__init__`` because ``_reload`` runs exactly this step
        again against the tensors that are already resident. ``build_s`` is
        reported by ``ping`` and the startup log, so a slow build is visible
        without a profiler.
        """
        t0 = time.time()
        # The checkpoint config decides which family implementation is built;
        # ``serve`` itself only knows the engine_api.Model contract.
        self.model = engine_api.create_model(self.model_dir, max_seq=self.max_seq,
                                            bench=self.bench, weights=self.weights,
                                            paged=self.prefix_cache)
        # The engine owns the prefix cache, the block pool, the scheduler and the
        # captured graphs; ``serve`` keeps only the transport and rendering.
        self.engine = llm_engine.LLMEngine(self.cfg, self.model)
        self.engine.eos = set(self.eos)
        self.engine.build()
        self.pc = self.engine.pc
        # The tower is built here, after the engine's pools and graphs exist: it
        # is an eager module that never enters the decode graph, so it can come
        # up last and still be resident for the first image request.
        # ``--no-vision`` skips it, which is the only way to get the 0.62 GiB
        # back.
        if self.vision and self.model.vision_sd and self.model.vision is None:
            try:
                t_v = time.time()
                self.model.enable_vision(quant=self.vision_int8)
                print(f"[serve] vision tower ready in {time.time() - t_v:.1f}s "
                      f"(int8={self.vision_int8}, "
                      f"{len(self.model.vision_sd)} tensors)", flush=True)
            except Exception as exc:  # noqa: BLE001 - text serving must still work
                print(f"[serve] vision tower unavailable: {type(exc).__name__}: {exc}",
                      flush=True)
                self.vision = False
        elif not self.vision:
            self.model.vision_sd = {}
        self.build_s = time.time() - t0

    def _reload(self) -> None:
        """Re-import the engine (and the kernel extension) and rebuild from VRAM.

        The old model (buffers plus its captured graph pool) is torn down first,
        otherwise the two models do not both fit in 16 GB. The graph is nulled
        explicitly inside ``close()``: a live ``CUDAGraph`` pins its private
        memory pool, and that pool is ~half the model's footprint (all the
        grouped-projection copies), so merely dropping ``self.model`` is not
        enough.
        """
        import gc

        def _mem(tag: str) -> None:
            print(f"[reload] {tag}: alloc={torch.cuda.memory_allocated() / 2**30:.2f} "
                  f"reserved={torch.cuda.memory_reserved() / 2**30:.2f} GiB", flush=True)

        _mem("before")
        old = self.model
        self.engine = None
        self.model = None
        if old is not None:
            old.close()          # nulls the graphs (they pin their pools) + layers
        del old
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        _mem("after teardown")

        importlib.reload(hip_ops)

        # Every engine module, dependencies first: reloading in the other order
        # would leave old classes bound to the new tensors.
        engine_api.reload()
        self.processor = None    # re-imported transformers classes: rebuild lazily
        self._processor_failed = None
        self.n_reload += 1
        self._build()
        _mem("after build")

    def start(self) -> None:
        """Run the serving loop on its own thread.

        Idempotent: a second call is a no-op, so ``serve`` can start the loop
        before it accepts connections without tracking the state itself.
        """
        if self._loop_thread is not None:
            return
        self._loop_thread = threading.Thread(target=self._engine_loop,
                                             name="engine-loop", daemon=True)
        self._loop_thread.start()

    def stop(self) -> None:
        """Ask the engine thread to return and wait up to 5 s for it.

        The thread also polls the condition with a 0.2 s timeout, so a missed
        notification costs one step rather than a hang. A join that times out is
        not an error: the thread is a daemon and the process is exiting anyway.
        """
        self._stop = True
        with self._cv:
            self._cv.notify_all()
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=5.0)
            self._loop_thread = None

    def _engine_loop(self) -> None:
        """One ``engine.step()`` per iteration, dispatching tokens to requests.

        The 0.2 s wait bounds the cost of a lost notification: the loop must also
        notice ``_paused`` (a diagnostic is touching the model) and ``_stop``,
        neither of which may be waited on indefinitely.
        """
        while not self._stop:
            with self._cv:
                while not self._stop:
                    engine = self.engine
                    if engine is not None and not self._paused \
                            and engine.scheduler.has_work():
                        break
                    self._cv.wait(0.2)
                if self._stop:
                    return
            try:
                with self._lock:
                    events = self.engine.step()
            except Exception as exc:  # keep serving; report to every active request
                print(f"[serve] engine step failed: {exc}", flush=True)
                self._fail(str(exc))
                self.engine.reset()
                continue
            for seq, tok, done in events:
                with self._lock:
                    out_q = self._outputs.get(seq.seq_id)
                if out_q is not None:
                    out_q.put({"token": tok, "done": done, "error": None,
                               "prefill_s": seq.prefill_s})
                if done:
                    with self._lock:
                        self._outputs.pop(seq.seq_id, None)

    def _fail(self, message: str) -> None:
        """Report one engine failure to every in-flight request and drop the queues.

        A failed ``step()`` has already lost the batch's state, so each waiting
        connection is answered with the error rather than left blocked on a queue
        nothing will feed again. The dict is swapped under the lock so a request
        submitted concurrently cannot land in the discarded copy.

        Args:
            message: The failure text, sent verbatim to every waiting connection.
        """
        with self._lock:
            pending, self._outputs = self._outputs, {}
        for out_q in pending.values():
            out_q.put({"token": None, "done": True, "error": message,
                       "prefill_s": 0.0})


class RuntimeGenerationMixin:

    def _submit(self, ids: list[int], params: SamplingParams, mm=None):
        """Queue one request and return its sequence plus the token queue.

        ``mm`` rides on the :class:`~sequence.Sequence` (``seq.mm``), which is what
        the runner, the block manager and the radix tree all read: the tree salts
        its block keys with ``mm.digest`` so two different images can never share a
        cached KV prefix. The queue is registered *inside* the same lock that the
        engine loop's ``step()`` takes, so a prefilled sequence always has a queue
        waiting for its tokens.

        Args:
            ids: Prompt token ids, already rendered and (for an image request)
                already expanded to the processor's layout.
            params: Sampling and stopping policy for this request.
            mm: The request's ``MMInput``, or ``None`` for a text prompt.

        Returns:
            ``(seq, out_q)``: the engine-side sequence and the queue that will
            receive ``{"token", "done", "error", "prefill_s"}`` events.
        """
        out_q: queue.Queue = queue.Queue()
        with self._lock:
            seq = self.engine.add_request(ids, params, mm=mm)
            self._outputs[seq.seq_id] = out_q
            self._cv.notify()
        return seq, out_q

    def _blocking_generate(self, prompt_ids: list[int], max_new: int,
                           temperature: float, deadline: float = 0.0, mm=None):
        """Submit a request and collect its tokens until the engine says done.

        Args:
            prompt_ids: Prompt token ids.
            max_new: Output budget in tokens, already clamped by ``_room``.
            temperature: 0 for greedy.
            deadline: Epoch-seconds cutoff, 0 for none.
            mm: The request's ``MMInput``, or ``None``.

        Returns:
            ``(tokens, prefill_s, decode_s)``. ``decode_s`` excludes the prefill
            and is floored at 0 because the two timings are read from different
            clocks and can invert by a few microseconds on a short answer.

        Raises:
            RuntimeError: The engine reported an error for this request. The
                failed step has already been reset, so the caller may retry a new
                request, but not this sequence.
        """
        seq, out_q = self._submit(prompt_ids,
                                  self._sampling(max_new, temperature, deadline), mm=mm)
        t0 = time.time()
        out: list[int] = []
        while True:
            ev = out_q.get()
            if ev["error"]:
                raise RuntimeError(ev["error"])
            if ev["token"] is not None:
                out.append(ev["token"])
            if ev["done"]:
                break
        total = time.time() - t0
        pre = seq.prefill_s
        return out, pre, max(total - pre, 0.0)

    def _room(self, n_prompt: int, max_new) -> tuple[int, bool]:
        """Clamp the output budget to what the context window has left.

        The window is shared by prompt *and* output (262144 tokens total), and the
        decode step has no bounds check of its own: it writes the KV cache at
        ``pos`` and the rope tables are ``[max_seq]``, so a budget larger than the
        room would run off the end of both. Long-context clients routinely ask for
        a full window without looking at their prompt length, so the engine clamps
        instead of faulting.

        Args:
            n_prompt: Prompt length in tokens, after image expansion.
            max_new: Requested output budget; ``None`` or negative means "as much
                as fits", the caller stating no preference.

        Returns:
            ``(budget, window_full)``. ``budget`` is what the request may actually
            generate, never negative -- a prompt that exactly fills the window
            generates nothing. ``window_full`` says the *request* was cut by the
            window rather than finishing on its own. Both travel back in the
            reply so the client can tell "done" from "ran out of room".
        """
        room = max(0, self.max_seq - n_prompt)
        if max_new is None or int(max_new) < 0:
            return room, False
        return max(0, min(int(max_new), room)), int(max_new) > room

    def _sampling(self, max_new: int, temperature: float,
                  deadline: float = 0.0) -> SamplingParams:
        """Build the per-request decoding policy the engine's scheduler sees.

        Args:
            max_new: Output budget, after ``_room`` clamping.
            temperature: 0 for greedy, which is what enables the spec path.
            deadline: Epoch-seconds cutoff, 0 for none.

        Returns:
            The sampling parameters, with this server's full stop-id set attached
            so every path stops on the template's own end-of-turn token.
        """
        return SamplingParams(temperature=temperature, max_tokens=max_new,
                              eos=set(self.eos), deadline=deadline)

    def _deadline(self, req: dict) -> float:
        """Wall-clock cap for this generation (epoch seconds, 0 = none).

        The request may only shorten the server-wide ``--max-seconds``, never
        extend it: the cap exists to cover a client that goes quiet without
        closing its socket, so a client cannot opt out of it.

        Args:
            req: The request dict; ``max_seconds`` overrides the server default.

        Returns:
            The absolute cutoff, or 0 when no cap applies.
        """
        secs = float(req.get("max_seconds") or self.max_seconds or 0.0)
        return time.time() + secs if secs > 0 else 0.0

    def _cached_tokens(self) -> int:
        """How much of the last prompt came out of the prefix cache.

        Returns:
            The matched prefix length, 0 when nothing was reused. Read after the
            request finished, because a batched step updates it per sequence and
            only the reply needs it.
        """
        if self.pc is not None:
            return self.pc.last_matched
        return int(getattr(self.engine, "last_matched", 0))

    @torch.inference_mode()
    def _generate(self, prompt_ids: list[int], max_new: int, temperature: float,
                  deadline: float = 0.0, mm=None) -> tuple[list[int], float, float]:
        """Non-streaming generation through the engine's batched path.

        Args:
            prompt_ids: Prompt token ids.
            max_new: Output budget, after ``_room`` clamping.
            temperature: 0 for greedy.
            deadline: Epoch-seconds cutoff, 0 for none.
            mm: The request's ``MMInput``, or ``None``.

        Returns:
            ``(tokens, prefill_s, decode_s)``, as ``_blocking_generate`` returns
            them.

        Raises:
            RuntimeError: The engine reported an error for this request.
        """
        return self._blocking_generate(prompt_ids, max_new, temperature, deadline, mm=mm)

    @torch.inference_mode()
    def _spec_generate(self, prompt_ids: list[int], max_new: int, k: int,
                       deadline: float = 0.0,
                       mm=None) -> tuple[list[int], float, float]:
        """Greedy speculative decode, driving the model instead of the scheduler.

        This bypasses the engine's request queue: it resets the model, prefills,
        then loops on ``spec_step`` directly. That is what lets the ``spec`` op
        A/B it against the plain path, and it means the verify graph must already
        be captured for ``k + 1`` tokens (``capture_verify``) or the first step
        raises.

        Only greedy is supported: the accept/reject test compares the main
        model's argmax against the draft, which is exact for greedy and not for
        sampling.

        Args:
            prompt_ids: Prompt token ids.
            max_new: Output budget, after ``_room`` clamping.
            k: Drafts per step; the verify graph must already be captured for
                ``k + 1`` tokens.
            deadline: Epoch-seconds cutoff, 0 for none.
            mm: The request's ``MMInput``, or ``None``. Supported, not refused:
                ``prefill(mm=...)`` writes the request's ``rope_delta`` and both
                the drafter and the verify graph read it per step (``spec_step``
                copies it into ``mtp.rope_delta``), so an image prompt speculates
                exactly like a text one.

        Returns:
            ``(tokens, prefill_s, decode_s)``. ``decode_s`` covers the spec loop
            only, so it is directly comparable with the plain path's decode time.
        """
        m = self.model
        m.reset()
        t0 = time.time()
        logits = m.prefill(prompt_ids, mm=mm)
        t_pre = time.time() - t0
        m.n_drafted = m.n_accepted = m.n_spec_steps = 0
        out: list[int] = []
        t1 = time.time()
        while len(out) < max_new:
            if int(m.pos1) + m.verify_c > m.max_seq:
                break
            if deadline and time.time() >= deadline:
                break
            toks, logits = m.spec_step(logits, k)
            stop = False
            for t in toks:
                if t in self.eos:
                    stop = True
                    break
                out.append(t)
                if len(out) >= max_new:
                    break
            if stop:
                break
        return out[:max_new], t_pre, time.time() - t1


class RuntimeChatMixin:

    def _render_with_text(self, messages: list[dict], tools, enable_thinking: bool,
                          forced_prefix: str = "") -> tuple[list[int], str]:
        """Render a conversation with the checkpoint's own jinja template.

        Using the checkpoint's template rather than a hand-built string is what
        keeps the tool schemas, the thinking mode and the special tokens exactly
        as the model was trained to see them.

        Args:
            messages: Conversation turns in OpenAI chat format.
            tools: Tool schemas, or ``None``; the template renders them into the
                system turn, so a text prompt with tools is not a plain prompt.
            enable_thinking: Whether the template opens the reasoning block.
            forced_prefix: Template opener appended to the assistant turn, used to
                implement ``tool_choice: required``.

        Returns:
            ``(token_ids, text)``. The text is kept because the image path has to
            hand the *rendered* prompt to the processor, and because the caller
            inspects its tail to see whether thinking was opened.
        """
        text = self.tok.apply_chat_template(
            messages, tools=tools, enable_thinking=enable_thinking,
            tokenize=False, add_generation_prompt=True)
        if forced_prefix:
            # ``tool_choice: required`` (OpenAI's ``ChatCompletionToolChoiceOption``)
            # is implemented by continuing the assistant turn: the rendered prompt
            # already ends at the assistant header, so appending the template's
            # opener leaves the model no way out but to finish the call.
            text += forced_prefix
        return self.tok.encode(text), text

    def _render(self, messages: list[dict], tools, enable_thinking: bool,
                forced_prefix: str = "") -> list[int]:
        """Token ids of a rendered conversation; the text form is not needed.

        Args:
            messages: Conversation turns in OpenAI chat format.
            tools: Tool schemas, or ``None``.
            enable_thinking: Whether the template opens the reasoning block.
            forced_prefix: Template opener appended to the assistant turn.

        Returns:
            The prompt token ids, ready to submit.
        """
        return self._render_with_text(messages, tools, enable_thinking, forced_prefix)[0]

    @staticmethod
    def _split(raw: str, enable_thinking: bool) -> tuple[str, str]:
        """Split reasoning from answer, knowing whether thinking was asked for.

        ``split_thinking`` cannot tell "no reasoning" from "reasoning that was cut
        off before the closing marker": with thinking on and no ``</think>`` in the
        text, everything generated so far *is* reasoning. Returning it as the
        answer is how a thinking model's monologue ends up looking like a reply.

        Args:
            raw: The full generated text, forced prefix included.
            enable_thinking: Whether the caller asked the template for reasoning.

        Returns:
            ``(thinking, answer)``. Both are empty strings when the model produced
            neither; the answer is empty for a truncated reasoning block.
        """
        thinking, answer = split_thinking(raw)
        if enable_thinking and not thinking and "</think>" not in raw:
            thinking, answer = answer, ""
        return thinking, answer

    def _render_request(self, req: dict, thinking_on: bool, prefix: str):
        """Render one request's prompt -> ``(token_ids, text, MMInput|None)``.

        The text-only path is render, encode, no ``mm``. A request carrying
        ``images`` renders the *same* way (the template turns its
        ``{"type": "image"}`` markers into the vision delimiters) and then hands
        the rendered text to the processor, which is the only difference -- that
        keeps one rendering rule for both paths, so adding vision could never
        change how a text-only prompt renders.

        Args:
            req: The request dict: ``messages``, optional ``tools`` and ``images``.
            thinking_on: Whether the template should open the reasoning block.
            prefix: The forced tool-call opener, empty when the caller did not ask
                for ``tool_choice: required``.

        Returns:
            ``(token_ids, text, mm)``; ``mm`` is ``None`` for a text request. For
            an image request the token ids are the processor's expanded ones, not
            the tokenizer's.

        Raises:
            MMError: The request carries ``images`` while a forced tool call is
                active (400), or the image path refused (the status from
                ``_prepare_mm``). The two are mutually exclusive because the
                forced call needs the assistant turn to open the call, which
                leaves no room for the vision delimiters.
        """
        messages = req.get("messages", [])
        images = req.get("images") or []
        ids, prompt = self._render_with_text(messages, req.get("tools"), thinking_on,
                                             prefix)
        if not images:
            return ids, prompt, None
        if prefix:
            raise MMError("tool_choice=required forces the assistant turn to open "
                          "with a call; it cannot be combined with images")
        mm_ids, mm = self._prepare_mm(images, prompt)
        return mm_ids, prompt, mm

    @torch.inference_mode()
    def stream_chat(self, req: dict):
        """``chat`` as a generator: one reply line per step, for SSE clients.

        Yields ``type: "start"`` (prompt length and whether the prompt itself
        opened the reasoning block), then one ``type: "token"`` per decoded piece,
        then a final ``type: "done"`` with exactly the fields of the ``chat``
        reply. That three-line framing is the whole contract the OpenAI entry point
        forwards, so a new field belongs in ``done``, never in a new line type.

        ``text`` on a token line is *incremental*: only the new token's ids are
        decoded each step, so streaming stays linear in the answer length. The
        previous step's ids are kept only while they end in the middle of a
        multi-byte character, which is never more than a couple.

        A refused request is yielded as ``type: "error"`` with its status,
        *before* the ``start`` line: SSE has already committed a 200 by then, so
        the OpenAI entry point maps that line's status into the stream event instead of
        pretending the request succeeded.

        Speculative decode is deliberately not used here: the spec loop only
        returns after a whole K+1 verification window, which is the opposite of
        what a stream wants.

        Args:
            req: The request dict: ``messages``, optional ``tools``, ``images``,
                ``max_new``, ``temperature``, ``enable_thinking``,
                ``tool_prefix`` and ``max_seconds``.

        Raises:
            RuntimeError: The engine reported an error for this request. It is
                raised after the ``start`` line, so the transport closes the
                stream rather than sending a ``done`` that claims success.
        """
        prefix = str(req.get("tool_prefix") or "")

        # A forced call must start the answer, so reasoning is off for that turn:
        # the template would otherwise open ``<think>`` where the call has to go.
        thinking_on = bool(req.get("enable_thinking", False)) and not prefix
        try:
            ids, prompt, mm = self._render_request(req, thinking_on, prefix)
        except MMError as exc:
            yield {"type": "error", "ok": False, "error": str(exc),
                   "status": exc.status}
            return
        n, window_full = self._room(len(ids), req.get("max_new"))
        temp = float(req.get("temperature", 0.0))
        deadline = self._deadline(req)
        yield {"type": "start", "n_prompt": len(ids), "max_new": n,
               "prompt_opens_think": prompt.rstrip().endswith("<think>"),
               "forced_tool": bool(prefix),
               "n_images": mm.num_images if mm is not None else 0,
               "image_tokens": len(mm) if mm is not None else 0}
        t0 = time.time()
        seq, out_q = self._submit(ids, self._sampling(n, temp, deadline), mm=mm)
        out: list[int] = []
        hold: list[int] = []             # token ids held back: partial UTF-8
        t1 = time.time()
        while True:
            ev = out_q.get()
            if ev["error"]:
                raise RuntimeError(ev["error"])
            nxt = ev["token"]
            if nxt is not None:
                out.append(nxt)

                # Incremental decode: decoding the whole list every step is O(n^2)
                # and dominates a long answer (a 20k-token reply would spend
                # minutes in the tokenizer). A single token can end in the middle
                # of a UTF-8 sequence, so its ids are held back and decoded
                # together with the next ones until the character completes.
                held = hold + [nxt] if hold else [nxt]
                text = self.tok.decode(held, skip_special_tokens=True)
                if text.endswith("\ufffd"):
                    hold = held
                else:
                    hold = []
                    if text:
                        yield {"type": "token", "id": nxt, "text": text}
            if ev["done"]:
                break
        if hold:                         # the sequence never completed (truncated)
            text = self.tok.decode(hold, skip_special_tokens=True)
            if text:
                yield {"type": "token", "id": hold[-1], "text": text}
        t_dec = time.time() - t1
        t_pre = seq.prefill_s
        raw = self.tok.decode(out, skip_special_tokens=False)

        # The forced opener lives in the *prompt*, so it is put back in front of
        # the generated text before the call is parsed -- otherwise the block
        # would look like `<function=...>` with no enclosing `<tool_call>`.
        full = prefix + raw
        thinking, answer = self._split(full, thinking_on)
        calls = parse_tool_calls(answer)
        yield {"type": "done", "ok": True, "text": answer, "thinking": thinking,
               "raw": full, "tokens": out,
               "tool_calls": [c.to_openai_dict() for c in calls],
               "n_prompt": len(ids), "n_new": len(out), "max_new": n,
               "window_full": window_full, "cached_tokens": self._cached_tokens(),
               "prefill_ms": round(t_pre * 1000, 1),
               "decode_ms": round(t_dec * 1000, 1),
               "decode_tok_s": round(len(out) / t_dec, 2) if t_dec else None,
               # why generation stopped, when the engine decided it itself
               "stopped": ("deadline" if deadline and time.time() >= deadline else None),
               "spec_k": 0, "spec": None}

    def handle(self, req: dict) -> dict:
        """Serve one non-streaming op.

        Diagnostics (bench/profile/ops/spec/reload/reset) touch the model
        directly, so they are refused while requests are in flight and pause the
        engine loop while they run: ``step`` and the diagnostic take the same
        lock. Pausing is what makes them safe at all -- a reload tears the model
        down, and a profiler capture would otherwise interleave with a live step.

        Args:
            req: One request dict; ``op`` defaults to ``ping``.

        Returns:
            The reply dict, always with ``ok``. A refusal is ``ok: False`` plus an
            ``error`` string, not an exception: the transport turns that into a
            normal JSON line the client can read.
        """
        op = req.get("op", "ping")
        debug = op in ("reload", "reset", "bench", "profile", "ops", "spec")
        if not debug:
            return self._handle_impl(req)
        engine = self.engine
        if op in ("bench", "profile", "ops", "spec") and engine is not None \
                and engine.batching:
            return {"ok": False, "error": "diagnostic ops need the single-stream "
                                          "engine (run without --max-num-seqs)"}
        if engine is not None and (engine.scheduler.waiting or engine.scheduler.running):
            return {"ok": False, "error": "engine busy: diagnostic ops need an idle engine"}
        with self._lock:
            self._paused = True
            try:
                return self._handle_impl(req)
            finally:
                self._paused = False
                self._cv.notify_all()

    def _handle_impl(self, req: dict) -> dict:
        """Dispatch one op and build its reply.

        Every reply carries the same timing fields (``prefill_ms``,
        ``decode_ms``, ``decode_tok_s``, ``cached_tokens``) so a probe can compare
        paths without special-casing them. No locking here: the caller holds it.

        Args:
            req: The request dict; its keys depend on ``op``.

        Returns:
            The reply dict; ``{"ok": False, "error": ...}`` for an unknown op.
        """
        op = req.get("op", "ping")
        self.n_req += 1
        if op == "ping":
            out = {"ok": True, **self.info,
                   "graph": self.model.graph_ready, "max_seq": self.max_seq,
                   "load_s": round(self.load_s, 1), "build_s": round(self.build_s, 1),
                   "n_req": self.n_req, "n_reload": self.n_reload,
                   "hip": hip_ops.available(), "int8": self.int8,
                   "vram_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
                   "batch": bool(self.engine is not None and self.engine.batching),
                   "decode_capacity": int(getattr(self.model, "decode_capacity", 1)),
                   # what image requests this server accepts; the OpenAI shell reads
                   # ``vision`` to answer 503 up front instead of forwarding a request
                   # the engine will reject anyway
                   "vision": bool(self.vision and self.model.vision_sd),
                   "vision_int8": self.vision_int8,
                   "image_max_pixels": self.image_max_pixels,
                   "image_min_pixels": self.image_min_pixels,
                   "max_images": self.max_images,
                   "n_image_req": self.n_image_req,
                   "processor": ("loaded" if self.processor is not None
                                 else ("failed" if self._processor_failed else "lazy"))}

            if self.engine is not None:
                out["queue"] = {"waiting": len(self.engine.scheduler.waiting),
                                "running": len(self.engine.scheduler.running)}
            if self.pc is not None:
                out["prefix"] = self.pc.stats()
            elif self.engine is not None and self.engine.bm is not None \
                    and self.engine.bm.radix is not None:
                out["prefix"] = self.engine.bm.stats()
            return out
        if op == "reload":
            self._reload()
            return {"ok": True, "build_s": round(self.build_s, 1), "n_reload": self.n_reload,
                    "hip": hip_ops.available(), "hip_err": hip_ops.last_error()}
        if op == "reset":
            self.engine.reset()
            return {"ok": True}
        if op == "stop":
            return {"ok": True, "stopping": True}
        if op == "generate":
            ids = req.get("tokens") or self.tok.encode(req.get("prompt", ""))
            n, window_full = self._room(len(ids), req.get("max_new"))
            temp = float(req.get("temperature", 0.0))
            deadline = self._deadline(req)
            k = int(req.get("spec_k", self.spec_k))
            spec = None
            if k and temp == 0:
                out, t_pre, t_dec = self._spec_generate(ids, n, k, deadline)
                spec = self.model.spec_stats
            else:
                out, t_pre, t_dec = self._generate(ids, n, temp, deadline)
            return {"ok": True, "tokens": out, "text": self.tok.decode(out),
                    "n_prompt": len(ids), "n_new": len(out), "max_new": n,
                    "window_full": window_full, "cached_tokens": self._cached_tokens(),
                    "prefill_ms": round(t_pre * 1000, 1), "decode_ms": round(t_dec * 1000, 1),
                    "decode_tok_s": round(len(out) / t_dec, 2) if t_dec else None,
                    "stopped": ("deadline" if deadline and time.time() >= deadline else None),
                    "spec_k": k if spec else 0, "spec": spec}
        if op == "chat":
            prefix = str(req.get("tool_prefix") or "")
            thinking_on = bool(req.get("enable_thinking", False)) and not prefix
            try:
                ids, _prompt, mm = self._render_request(req, thinking_on, prefix)
            except MMError as exc:
                return {"ok": False, "error": str(exc), "status": exc.status}
            n, window_full = self._room(len(ids), req.get("max_new"))
            temp = float(req.get("temperature", 0.0))
            deadline = self._deadline(req)
            k = int(req.get("spec_k", self.spec_k))
            spec = None
            if k and temp == 0:
                out, t_pre, t_dec = self._spec_generate(ids, n, k, deadline, mm=mm)
                spec = self.model.spec_stats
            else:
                out, t_pre, t_dec = self._generate(ids, n, temp, deadline, mm=mm)
            raw = self.tok.decode(out, skip_special_tokens=False)
            full = prefix + raw
            thinking, answer = self._split(full, thinking_on)
            calls = parse_tool_calls(answer)
            return {"ok": True, "text": answer, "thinking": thinking, "raw": full,
                    "tool_calls": [c.to_openai_dict() for c in calls], "forced_tool": bool(prefix),
                    "n_prompt": len(ids), "n_new": len(out), "max_new": n,
                    "window_full": window_full, "cached_tokens": self._cached_tokens(),
                    "prefill_ms": round(t_pre * 1000, 1),
                    "decode_tok_s": round(len(out) / t_dec, 2) if t_dec else None,
                    "stopped": ("deadline" if deadline and time.time() >= deadline else None),
                    "spec_k": k if spec else 0, "spec": spec,
                    # what the image path actually did, so a probe can report it
                    "n_images": mm.num_images if mm is not None else 0,
                    "image_tokens": len(mm) if mm is not None else 0}
        if op == "spec":
            # A/B the two decode paths on the same prompt. Both are warmed before
            # timing, and the spec output is compared against plain greedy because
            # a speculative loop that is only "about as good" is not a decoder.
            n = int(req.get("n", 48))
            ids = self.tok.encode(req.get("prompt") or "The quick brown fox")
            k = int(req.get("spec_k", self.spec_k or 1))
            self.model.capture_verify(k + 1)
            self._generate(ids, 8, 0.0)
            plain, _, t_a = self._generate(ids, n, 0.0)
            self._spec_generate(ids, 8, k)
            spec, _, t_b = self._spec_generate(ids, n, k)
            same = sum(a == b for a, b in zip(plain, spec))
            return {"ok": True, "n": n, "k": k, "n_prompt": len(ids),
                    "plain_tok_s": round(len(plain) / t_a, 2),
                    "spec_tok_s": round(len(spec) / t_b, 2),
                    "speedup": round((len(spec) / t_b) / (len(plain) / t_a), 3),
                    "exact": f"{same}/{min(len(plain), len(spec))}",
                    "accept": self.model.spec_stats}
        if op == "bench":
            n = int(req.get("n", 30))
            ids = self.tok.encode(req.get("prompt") or "The quick brown fox")
            self.model.reset()
            logits = self.model.prefill(ids)
            for _ in range(3):  # untimed: the first replay pays graph cache warming
                logits = self.model.decode_step(int(logits.argmax()))
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(n):
                logits = self.model.decode_step(int(logits.argmax()))
            torch.cuda.synchronize()
            dt = time.time() - t0
            return {"ok": True, "n": n, "ms_per_step": round(dt / n * 1000, 2),
                    "decode_tok_s": round(n / dt, 2)}
        if op == "profile":
            n = int(req.get("n", 30))
            ids = self.tok.encode(req.get("prompt") or "The quick brown fox")
            self.model.reset()
            logits = self.model.prefill(ids)
            for _ in range(3):
                logits = self.model.decode_step(int(logits.argmax()))
            torch.cuda.synchronize()
            with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                for _ in range(n):
                    logits = self.model.decode_step(int(logits.argmax()))
                torch.cuda.synchronize()
            ka = prof.key_averages()
            total = sum(r.self_device_time_total for r in ka)
            rows = sorted(ka, key=lambda r: r.self_device_time_total, reverse=True)[:18]
            return {"ok": True, "total_us": round(total / n, 1),
                    "rows": [{"kernel": r.key[:60], "calls": r.count // n,
                              "us": round(r.self_device_time_total / n, 1),
                              "pct": round(100 * r.self_device_time_total / total, 1)}
                             for r in rows]}
        if op == "ops":
            # eager (un-graphed) forward so each aten op is individually visible,
            # grouped by op name with its self device time -> per-op attribution
            n = int(req.get("n", 5))
            ids = self.tok.encode(req.get("prompt") or "The quick brown fox")
            self.model.reset()
            tok = int(self.model.prefill(ids).argmax())
            with torch.inference_mode(), torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA]) as prof:
                for _ in range(n):
                    self.model.logits(tok)
                torch.cuda.synchronize()
            ka = prof.key_averages()
            total = sum(r.self_device_time_total for r in ka)
            rows = sorted(ka, key=lambda r: r.self_device_time_total, reverse=True)[:25]
            return {"ok": True, "n": n, "total_us": round(total / n, 1),
                    "rows": [{"op": r.key[:70], "calls": r.count // n,
                              "us": round(r.self_device_time_total / n, 1),
                              "pct": round(100 * r.self_device_time_total / total, 1)}
                             for r in rows]}
        return {"ok": False, "error": f"unknown op {op!r}"}
