"""OpenAI-compatible HTTP/SSE adapter.

Module: ``coffin.serving``. Request validation, socket I/O, stream parsing and
HTTP response writing live in separate modules; this file owns only the Handler
and process CLI wiring.

Constraints:
    * Standard library only. The model process is reached through the socket.
    * Public helper names are re-exported here for existing tests and clients.
"""

from __future__ import annotations

import json
import os
import threading
import time
import traceback
import uuid
import itertools
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .client import EngineClient
from .request import (
    IMAGE_MIMES,
    _decode_data_url,
    _has_image_part,
    _include_usage,
    _norm_messages,
    _template_content,
    _text_of,
    _thinking_flag,
    _tool_plan,
    ImagePartError,
)
from . import request as _request
from .stream import AnswerFilter, ToolStream, TOOL_OPEN

SOCK = os.environ.get("ENGINE_SOCK", "/tmp/qwen35.sock")
STATE: dict = {}
MAX_IMAGE_BYTES = _request.MAX_IMAGE_BYTES
MAX_IMAGES = _request.MAX_IMAGES
MAX_IMAGE_TOTAL_BYTES = _request.MAX_IMAGE_TOTAL_BYTES


def _split_messages(messages: list):
    """Split images using the limits patched on this compatibility module."""
    _request.MAX_IMAGE_BYTES = MAX_IMAGE_BYTES
    _request.MAX_IMAGES = MAX_IMAGES
    _request.MAX_IMAGE_TOTAL_BYTES = MAX_IMAGE_TOTAL_BYTES
    return _request._split_messages(messages)


def _max_tokens(body: dict) -> int:
    """Read the output budget using this module's shared server state."""
    return _request._max_tokens(body, STATE)

class Handler(BaseHTTPRequestHandler):
    """One HTTP request; the OpenAI wire shapes live in this class.

    ``ThreadingHTTPServer`` gives each request its own thread, and the engine,
    not this class, is what serializes the actual generations.

    The response shapes are fixed by the OpenAI definition: a JSON
    ``chat.completion`` object, ``chat.completion.chunk`` objects for SSE, and an
    ``{"error": {...}}`` body with the status code the client will branch on.
    """

    protocol_version = "HTTP/1.1"
    # A vendor-neutral identity: this adapter fronts the DCU build and a CoreX
    # build alike, and a client that branches on the string must not have to
    # learn a new one each time the engine moves to another accelerator.
    server_version = "coffin-engine/0.1"

    def log_message(self, fmt, *args):  # noqa: A003 - stdlib hook name
        print(f"[http] {self.address_string()} {fmt % args}", flush=True)

    def _json(self, code: int, obj) -> None:
        """Write one JSON response with an explicit ``Content-Length``.

        The length matters: the connection is HTTP/1.1 keep-alive, and a missing
        length would leave the client waiting on a body that never ends.
        ``Access-Control-Allow-Origin`` is open because the usual client is a
        browser UI on another origin.

        Args:
            code: HTTP status.
            obj: The JSON-serialisable body.
        """
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code: int, message: str, err_type: str = "invalid_request_error") -> None:
        """Write the OpenAI error envelope, which clients parse field by field.

        Args:
            code: HTTP status.
            message: Human-readable cause; it is shown to the user, so it names
                what the client should change.
            err_type: ``invalid_request_error`` for a fixable request,
                ``server_error`` when the failure is ours.
        """
        self._json(code, {"error": {"message": message, "type": err_type, "code": code}})

    def _sse_start(self) -> None:
        """Commit the 200 and the SSE headers for a streaming reply.

        Called only after the engine's first line proved the request was accepted:
        once these headers are out, a rejection can no longer be a 400.
        ``close_connection`` is set because SSE ends at EOF -- there is no length
        to send up front.
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.close_connection = True         # SSE ends at EOF: no Content-Length

    def _sse(self, obj) -> None:
        """Write one SSE frame and flush it.

        Args:
            obj: The chunk object to serialise as the frame's ``data:`` payload.
                The double newline is what terminates a frame, and the flush is
                what makes a token arrive before the next one is generated.
        """
        self.wfile.write(f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _body(self) -> dict | None:
        """Read and parse the request body.

        Returns:
            The parsed dict, or ``None`` when a 400 has already been written -- the
            caller then returns, because the error reply is complete.
        """
        try:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")
        except Exception as exc:  # noqa: BLE001
            self._error(400, f"bad request body: {exc}")
            return None

    def do_OPTIONS(self):  # noqa: N802
        """Answer the CORS preflight a browser client sends before the POST."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "content-type, authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):  # noqa: N802
        """Serve the two discovery/liveness routes a client polls.

        ``/health`` reports engine state by forwarding ``ping``; ``/v1/models``
        is what client libraries call before anything else, so it must answer from
        the cached state rather than touching the engine.
        """
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path in ("/health", "/healthz"):
            try:
                rep = STATE["engine"].call({"op": "ping"})

                # ``rep`` already carries model / model_path / arch from the engine
                self._json(200, {"status": "ok",
                                 "served_model_name": STATE["served_name"], **rep})
            except Exception as exc:  # noqa: BLE001
                self._json(503, {"status": "unavailable", "error": str(exc)})
        elif path == "/v1/models":
            self._json(200, {"object": "list", "data": [{
                "id": STATE["served_name"], "object": "model",
                "created": STATE["created"], "owned_by": "coffin-engine"}]})
        else:
            self._error(404, f"no route {self.path}")

    def do_POST(self):  # noqa: N802
        """Route the single POST endpoint and keep the server alive on any failure.

        A client that hangs up mid-stream is not an error to log as a server fault:
        it is the expected consequence of an SSE client being closed.
        """
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path != "/v1/chat/completions":
            self._error(404, f"no route {self.path}")
            return
        body = self._body()
        if body is None:
            return
        try:
            self._chat(body)
        except (BrokenPipeError, ConnectionResetError):
            print(f"[http] {self.address_string()} client hung up", flush=True)
        except Exception as exc:  # noqa: BLE001 - never kill the server on one bad request
            traceback.print_exc()
            try:
                self._error(500, f"{type(exc).__name__}: {exc}", "server_error")
            except Exception:
                pass

    def _chat(self, body: dict) -> None:
        """Validate one chat-completions request and hand it to a reply path.

        Validation happens here and in ``_split_messages`` so that everything the
        client got wrong is a 400 before the engine sees the request. The engine
        would otherwise answer late, after a prefill, with the same problem.

        Args:
            body: The parsed request body.
        """
        raw_messages, images, err = _split_messages(body.get("messages") or [])
        if err:
            self._error(400, err)
            return
        messages = _norm_messages(raw_messages)
        if not messages:
            self._error(400, "`messages` must be a non-empty array")
            return
        if images and STATE.get("vision") is False:
            self._error(503, "this server was started with --no-vision: image inputs "
                             "are disabled", "server_error")
            return
        max_new = _max_tokens(body)
        temperature = float(body.get("temperature") or 0.0)
        stream = bool(body.get("stream"))
        model = body.get("model") or STATE["served_name"]
        roles = sorted({m["role"] for m in messages})
        think = _thinking_flag(body, STATE["enable_thinking"])
        tools, forced = _tool_plan(body)
        single_call = body.get("parallel_tool_calls") is False
        include_usage = _include_usage(body)
        think_keys = sorted(k for k in ("enable_thinking", "thinking", "reasoning_effort",
                                        "chat_template_kwargs") if k in body)
        print(f"[http] chat: {len(messages)} msgs {roles} max_new={max_new} "
              f"stream={stream} think={think} tools={bool(tools)} "
              f"images={len(images)} "
              f"tool_choice={json.dumps(body.get('tool_choice'), ensure_ascii=False)} "
              f"forced_call={bool(forced)} usage_chunk={include_usage} "
              f"think_keys={think_keys}", flush=True)
        req = {"op": "chat_stream" if stream else "chat",
               "messages": messages, "tools": tools,
               "tool_prefix": forced or None,

               # a forced call has to open the answer, so it cannot be reasoning first
               "enable_thinking": think and not forced,
               "max_new": max_new, "temperature": temperature,

               # the engine's own wall-clock cap: the only cutoff that also fires
               # when the client goes quiet without closing its socket
               "max_seconds": STATE.get("max_decode_seconds") or None}
        if images:
            # Base64 (already validated and re-encoded here) travels on the socket
            # JSON: the engine decodes, preprocesses and expands the placeholder
            # tokens itself with the checkpoint's own processor, so this process
            # stays stdlib-only and never has to agree with it numerically.
            req["images"] = images
        cid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())
        if stream:
            self._chat_stream(req, cid, created, model, single_call, include_usage)
        else:
            self._chat_once(req, cid, created, model, single_call)

    @staticmethod
    def _finish_reason(out: dict) -> str:
        """Map the engine's stop reason onto the definition's enum.

        Args:
            out: The engine's reply (or ``done`` line).

        Returns:
            ``tool_calls`` when a call was parsed, ``length`` when the effective
            budget (after the window clamp) ran out, else ``stop``. A server-side
            deadline also maps to ``length``: the definition has no "deadline"
            value, and ``length`` is the one that means "cut short" rather than
            "the model finished".
        """
        if out.get("tool_calls"):
            return "tool_calls"
        if out.get("stopped") == "deadline":
            return "length"
        n_new = int(out.get("n_new") or 0)
        n_max = int(out.get("max_new") or 0)
        return "length" if n_max > 0 and n_new >= n_max else "stop"

    @staticmethod
    def _usage(out: dict) -> dict:
        """Build the definition's ``usage`` object from the engine's counters.

        Args:
            out: The engine's reply (or ``done`` line).

        Returns:
            ``prompt_tokens`` / ``completion_tokens`` / ``total_tokens``, plus the
            ``prompt_tokens_details.cached_tokens`` extension that reports how much
            of the prompt the radix prefix cache had already computed.
        """
        pt = int(out.get("n_prompt") or 0)
        ct = int(out.get("n_new") or 0)
        return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct,
                "prompt_tokens_details": {"cached_tokens":
                                          int(out.get("cached_tokens") or 0)}}

    def _chat_once(self, req: dict, cid: str, created: int, model: str,
                   single_call: bool = False) -> None:
        """Answer a non-streaming request with one ``chat.completion`` object.

        Args:
            req: The engine request.
            cid: The ``chatcmpl-`` id for this reply.
            created: Epoch seconds reported as the completion's creation time.
            model: The model id the client asked for (or the served default).
            single_call: Honour ``parallel_tool_calls: false`` by keeping one call.
        """
        out = STATE["engine"].call(req)
        if not out.get("ok", True):
            # The engine says *why* and with which status: a bad image is a 400, a
            # server without a vision tower a 503. Collapsing all of them into 500
            # would tell a client to retry a request that can never succeed.
            self._error(int(out.get("status") or 500),
                        f"engine: {out.get('error')}",
                        "invalid_request_error" if int(out.get("status") or 500) < 500
                        else "server_error")
            return
        calls = out.get("tool_calls") or []
        if single_call and len(calls) > 1:      # parallel_tool_calls: false
            calls = calls[:1]
        text = out.get("text") or ""
        if calls:
            # keep only the natural-language lead-in: the calls travel structured,
            # exactly like the streamed content (the template renders both).
            text = text.split(TOOL_OPEN, 1)[0].strip()
        message: dict = {"role": "assistant", "content": text or None}
        if calls:
            message["tool_calls"] = calls
        if out.get("thinking"):
            message["reasoning_content"] = out["thinking"]
        self._json(200, {
            "id": cid, "object": "chat.completion", "created": created, "model": model,
            "choices": [{"index": 0, "message": message,
                         "finish_reason": self._finish_reason({**out, "tool_calls": calls})}],
            "usage": self._usage(out),
        })

    def _chat_stream(self, req: dict, cid: str, created: int, model: str,
                     single_call: bool = False, include_usage: bool = False) -> None:
        """Answer a ``stream: true`` request as ``chat.completion.chunk`` SSE frames.

        The chunks are the definition's: a role-only opening delta, one delta per
        reasoning/content/tool-call fragment, then a final empty delta carrying
        ``finish_reason``, then optionally the usage chunk, then ``data: [DONE]``.
        Every frame repeats ``id``/``object``/``created``/``model`` because clients
        treat each chunk as a standalone object.

        Args:
            req: The engine request; its ``tool_prefix`` seeds the streaming
                parser so a forced call fragments exactly like a natural one.
            cid: The ``chatcmpl-`` id shared by every chunk.
            created: Epoch seconds reported in every chunk.
            model: The model id reported in every chunk.
            single_call: Honour ``parallel_tool_calls: false`` by capping the
                parser at one call.
            include_usage: Emit the definition's extra ``choices: []`` usage chunk.
        """
        base = {"id": cid, "object": "chat.completion.chunk", "created": created,
                "model": model}

        # The definition says: with ``stream_options.include_usage`` every other
        # chunk carries ``usage: null`` and one extra chunk with ``choices: []``
        # carries the real numbers just before ``[DONE]``.
        marker = {"usage": None} if include_usage else {}
        t_start = time.time()
        sent = 0

        def emit(choices: list, **extra) -> None:
            """Write one chunk, tracking how many choices have gone out.

            Args:
                choices: The chunk's ``choices`` array.
                **extra: Additional top-level fields (an ``error`` object on the
                    failure path).
            """
            nonlocal sent
            sent += len(choices)
            self._sse({**base, "choices": choices, **marker, **extra})

        flt: AnswerFilter | None = None
        done: dict | None = None
        # One streaming tool-call parser for this reply, seeded with the forced
        # opener so a ``tool_choice: required`` call streams exactly like a natural
        # one (the opener lives in the prompt, not in the generated text).
        tstream = ToolStream(seed=req.get("tool_prefix") or "",
                             max_calls=1 if single_call else None)
        # The engine validates the request (image decode, processor expansion, the
        # placeholder self-check) *before* its ``start`` line, so its first line can
        # still be a rejection. Nothing is written before that line arrives: an
        # image request that is refused then gets a real 400/503 with the JSON
        # error body an OpenAI client expects, instead of a 200 SSE stream carrying
        # an error event -- which no client retries or reports usefully.
        lines = STATE["engine"].stream(req)
        try:
            first = next(lines)
        except StopIteration:
            first = None
        if first is not None and first.get("type") == "error":
            self._error(int(first.get("status") or 400),
                        f"engine: {first.get('error') or 'rejected the request'}")
            return
        self._sse_start()
        emit([{"index": 0, "delta": {"role": "assistant", "content": ""},
               "finish_reason": None}])
        try:
            for line in itertools.chain([first] if first is not None else [], lines):
                if line is None:
                    continue
                if line.get("type") == "start":
                    flt = AnswerFilter(bool(line.get("prompt_opens_think")))
                    if line.get("forced_tool"):
                        # ``tool_choice: required``: the opener is already in the
                        # prompt, so the generated text is the *body* of the call --
                        # it must never go out as content, the call is delivered
                        # structured below.
                        flt.done_tools = True
                    continue
                if line.get("type") != "token":
                    done = line
                    break
                piece = line.get("text") or ""
                delta_r, delta_c = flt.push(piece) if flt else ("", "")
                for td in tstream.push(piece):
                    emit([{"index": 0, "delta": td, "finish_reason": None}])
                if delta_r:
                    # the reasoning block goes out as it is produced, so a client
                    # with a thinking pane shows it live (and never as content)
                    emit([{"index": 0, "delta": {"reasoning_content": delta_r},
                           "finish_reason": None}])
                if delta_c:
                    emit([{"index": 0, "delta": {"content": delta_c},
                           "finish_reason": None}])
        except Exception as exc:  # noqa: BLE001 - report into the stream, then close it
            traceback.print_exc()

            # ``client_hung_up`` decides who cut the stream: when it is False the
            # engine stopped early, when it is True the answer was lost on the way
            # out (client or relay), not by the model.
            print(f"[http] stream aborted id={cid} sent={sent} "
                  f"wall={time.time() - t_start:.1f}s "
                  f"client_hung_up={isinstance(exc, (BrokenPipeError, ConnectionResetError))}",
                  flush=True)
            emit([{"index": 0, "delta": {}, "finish_reason": "error"}],
                 error={"message": f"{type(exc).__name__}: {exc}",
                        "type": "server_error"})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        if done is None or not done.get("ok", True):
            emit([{"index": 0, "delta": {}, "finish_reason": "error"}],
                 error={"message": f"engine: {(done or {}).get('error', 'no reply')}",
                        "type": "server_error"})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        if flt is not None and not flt.done_tools:
            tail_r, tail_c = flt.finish(bool(done.get("tool_calls")))
            if tail_r:      # the stream ended inside the reasoning block
                emit([{"index": 0, "delta": {"reasoning_content": tail_r},
                       "finish_reason": None}])
            if tail_c:
                emit([{"index": 0, "delta": {"content": tail_c}, "finish_reason": None}])
        for td in tstream.finish():          # truncated call: close what we have
            emit([{"index": 0, "delta": td, "finish_reason": None}])
        calls = done.get("tool_calls") or []
        if single_call and len(calls) > 1:       # parallel_tool_calls: false
            calls = calls[:1]
        if calls and not tstream.emitted_any:
            # Fallback for a call the streaming parser did not see (which would be
            # a bug in it): deliver the whole call in one delta. When the fragments
            # did go out, repeating the complete call here would make the client
            # concatenate the arguments twice.
            emit([{"index": 0, "delta": {
                "tool_calls": [{"index": i, **c} for i, c in enumerate(calls)]},
                "finish_reason": None}])
        last = {"index": 0, "delta": {},
                "finish_reason": self._finish_reason({**done, "tool_calls": calls})}
        if include_usage:
            emit([last])
            self._sse({**base, "choices": [], "usage": self._usage(done)})
        else:
            # OpenAI omits usage unless it is asked for; keeping it on the last
            # chunk is the lenient behaviour existing clients (the bridge, the
            # probes) already read, so it stays when nobody asked for the extra
            # chunk.
            self._sse({**base, "choices": [last], "usage": self._usage(done)})
        print(f"[http] stream done id={cid} n_new={done.get('n_new')} "
              f"finish={last['finish_reason']} stopped={done.get('stopped')} "
              f"tokens_sent={sent} wall={time.time() - t_start:.1f}s "
              f"prefill_ms={done.get('prefill_ms')} client_hung_up=False", flush=True)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
