"""Self-test for ``openai_server.py`` against a fake engine socket (no GPU needed).

The HTTP/SSE layer is pure protocol code, so it can be checked anywhere: a stub
engine socket plays the canned token stream, ``openai_server`` is pointed at it,
and the OpenAI responses are asserted -- including the two things that are easy to
get wrong and invisible in a passing generation:

* the reasoning block is suppressed when the *prompt* opened ``<think>``;
* the ``<tool_call>`` markup never leaks into ``content``, even when the marker is
  split across two token pieces (the filter holds back a partial marker).

Port of ``diag/check_openai.py``.  The fake engine speaks TCP because that is
also what a dev box without ``AF_UNIX`` can do; only ``_open`` differs from the
shipped client, everything above it is the real line protocol.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import unittest
import urllib.request

import support  # noqa: F401  (sets MKL + sys.path)

import openai_server as srv
from tool_call import parse_tool_calls, split_thinking


class TcpEngine(srv.EngineClient):
    """``EngineClient`` over TCP instead of AF_UNIX."""

    def _open(self):
        conn = socket.create_connection(("127.0.0.1", int(self.sock_path)),
                                        timeout=self.timeout)
        return conn, conn.makefile("rb")


PLAIN = ["The", " capital", " of", " France", " is", " Paris", "."]
THINK = ["We", " need", " the", " capital", ".", "</think>", "The", " capital",
         " is", " Paris", "."]
THINK_TRUNC = ["We", " need", " to", " think", " about", " it", " first", "."]
CALL = ("<tool_call>\n<function=get_weather>\n<parameter=city>\n北京\n"
        "</parameter>\n</function>\n</tool_call>")
CALL2 = ("<tool_call>\n<function=multiply>\n<parameter=a>\n12\n</parameter>\n"
         "<parameter=b>\n8\n</parameter>\n</function>\n</tool_call>")
# With tool_choice: required the engine finds the opener already in the prompt, so
# what it generates -- and streams -- is only the body of the call.
CALL_BODY = CALL[len("<tool_call>"):]
TOOLS = ["Let", " me", " check", ".", CALL]
TOOLS_SPLIT = ["Let", " me", " check", ".", "<tool_ca", "ll>" + CALL[len("<tool_call>"):]]
CALL_FRAG_CITY = ["<tool_call>", "\n<function=", "get_weather", ">\n<parameter=", "city",
                  ">\n", "北", "京", "\n</par", "ameter>\n</fun", "ction>\n</tool", "_call>"]
CALL_FRAG_INT = ["<tool_call>", "\n<function=multiply>\n<parameter=a>\n1", "2\n</parameter>\n",
                 "<parameter=b>\n", "8", "\n</parameter>\n</function>\n</tool_call>"]
CALL_FRAG_ESC = ["<tool_call>", "\n<function=multiply>\n<parameter=x>\n", 'he said "hi"',
                 "\nline2\\path", "\n</parameter>\n</function>\n</tool_call>"]

SEEN: list[dict] = []          # every request the fake engine received
_MODULE: dict = {}


def _pieces(req: dict) -> tuple[list[str], bool]:
    """The canned stream + 'prompt opened <think>' flag for a request."""
    text = " ".join(str(m.get("content") or "") for m in req.get("messages") or [])
    if req.get("tool_prefix"):
        return [CALL_BODY], False
    if req.get("tools"):
        if "fragcity" in text:
            pieces = CALL_FRAG_CITY
        elif "fragint" in text:
            pieces = CALL_FRAG_INT
        elif "fragesc" in text:
            pieces = CALL_FRAG_ESC
        elif "two" in text:
            pieces = [CALL, CALL2]
        elif "split" in text:
            pieces = TOOLS_SPLIT
        else:
            pieces = TOOLS
    elif req.get("enable_thinking"):
        return THINK_TRUNC if "trunc" in text else THINK, True
    else:
        pieces = PLAIN
    n = int(req.get("max_new", -1))
    return (pieces[:n] if n >= 0 else pieces), False


def _reply(req: dict) -> dict:
    pieces, opens = _pieces(req)
    prefix = str(req.get("tool_prefix") or "")
    raw = prefix + "".join(pieces)              # the opener lives in the prompt
    thinking, answer = split_thinking(raw)
    if (req.get("enable_thinking") and not prefix
            and not thinking and "</think>" not in raw):
        thinking, answer = answer, ""           # same rule as serve.Server._split
    return {"ok": True, "text": answer, "thinking": thinking, "raw": raw,
            "tool_calls": [c.to_openai_dict() for c in parse_tool_calls(answer)],
            "forced_tool": bool(prefix),
            "n_prompt": 11, "n_new": len(pieces),
            "max_new": int(req.get("max_new", -1)), "window_full": False,
            "cached_tokens": 0,
            "prefill_ms": 120.0, "decode_tok_s": 42.0, "spec_k": 0, "spec": None}


def _fake_engine(port: int, stop: threading.Event) -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", port))
    server.listen(8)
    server.settimeout(0.5)
    stop.server_sock = server          # type: ignore[attr-defined]
    while not stop.is_set():
        try:
            conn, _ = server.accept()
        except socket.timeout:
            continue
        with conn:
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = conn.recv(65536)
                if not chunk:
                    break
                buf += chunk
            try:
                req = json.loads(buf.decode() or "{}")
                if req.get("op") != "ping":
                    SEEN.append(req)
                if req.get("op") == "ping":
                    conn.sendall(b'{"ok":true,"graph":true,"max_seq":4096}\n')
                elif req.get("op") == "chat_stream":
                    pieces, opens = _pieces(req)
                    conn.sendall((json.dumps({"type": "start", "n_prompt": 11,
                                              "prompt_opens_think": opens,
                                              "forced_tool": bool(req.get("tool_prefix"))}
                                             ) + "\n").encode())
                    for piece in pieces:
                        conn.sendall((json.dumps({"type": "token", "text": piece}) + "\n").encode())
                    done = _reply(req)
                    done["type"] = "done"
                    conn.sendall((json.dumps(done, ensure_ascii=False) + "\n").encode())
                else:
                    conn.sendall((json.dumps(_reply(req), ensure_ascii=False) + "\n").encode())
            except Exception as exc:  # noqa: BLE001
                conn.sendall((json.dumps({"ok": False, "error": str(exc)}) + "\n").encode())
    server.close()


def post(port: int, body: dict) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def sse(port: int, body: dict) -> list[dict]:
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps({**body, "stream": True}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        text = r.read().decode()
    out = []
    for line in text.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            out.append(json.loads(line[6:]))
    return out


def get(port: int, path: str) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=30) as r:
        return json.loads(r.read())


def content_of(chunks: list[dict]) -> str:
    return "".join(c["choices"][0]["delta"].get("content") or ""
                   for c in chunks if c["choices"])


def reasoning_of(chunks: list[dict]) -> str:
    return "".join(c["choices"][0]["delta"].get("reasoning_content") or ""
                   for c in chunks if c["choices"])


def tool_calls_of(chunks: list[dict]) -> list[dict]:
    """Accumulate streamed tool_calls fragments the way an OpenAI client does."""
    acc: dict[int, dict] = {}
    for chunk in chunks:
        if not chunk["choices"]:
            continue
        for tc in chunk["choices"][0]["delta"].get("tool_calls") or []:
            index = tc.get("index", 0)
            slot = acc.setdefault(index, {"id": "", "type": "",
                                          "function": {"name": "", "arguments": ""}})
            if tc.get("id"):
                slot["id"] = tc["id"]
            if tc.get("type"):
                slot["type"] = tc["type"]
            fn = tc.get("function") or {}
            if fn.get("name"):
                slot["function"]["name"] += fn["name"]
            slot["function"]["arguments"] += fn.get("arguments") or ""
    return [acc[i] for i in sorted(acc)]


def fragments_of(chunks: list[dict]) -> list[dict]:
    return [tc for c in chunks if c["choices"]
            for tc in (c["choices"][0]["delta"].get("tool_calls") or [])]


TOOLS_SCHEMA = [{"type": "function",
                 "function": {"name": "get_weather", "parameters": {"type": "object"}}}]


def setUpModule() -> None:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    engine_port = probe.getsockname()[1]
    probe.close()
    stop = threading.Event()
    thread = threading.Thread(target=_fake_engine, args=(engine_port, stop), daemon=True)
    thread.start()
    for _ in range(50):                       # wait for the stub to bind
        if getattr(stop, "server_sock", None) is not None:
            break
        time.sleep(0.05)

    srv.STATE.update({"engine": TcpEngine(engine_port),
                      "model_path": "/models/Qwen3.5-4B",
                      "served_name": "qwen3.5-4b", "enable_thinking": False,
                      "created": int(time.time())})
    httpd = srv.ThreadingHTTPServer(("127.0.0.1", 0), srv.Handler)
    httpd.daemon_threads = True
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    _MODULE.update({"httpd": httpd, "thread": thread, "stop": stop, "port": port})


def tearDownModule() -> None:
    _MODULE["stop"].set()
    _MODULE["httpd"].shutdown()
    _MODULE["httpd"].server_close()
    _MODULE["thread"].join(timeout=2)


class OpenAIServerTest(unittest.TestCase):
    @property
    def port(self) -> int:
        return _MODULE["port"]

    def test_health_and_models(self):
        health = get(self.port, "/health")
        self.assertEqual(health.get("status"), "ok", json.dumps(health))
        self.assertTrue(health.get("graph") is True)
        models = get(self.port, "/v1/models")
        self.assertEqual(models["object"], "list")
        self.assertEqual(models["data"][0]["id"], "qwen3.5-4b")

    def test_chat_completion(self):
        r = post(self.port, {"model": "qwen3.5-4b",
                             "messages": [{"role": "user", "content": "hi"}]})
        msg = r["choices"][0]["message"]
        self.assertEqual(msg["content"], "The capital of France is Paris.")
        self.assertEqual(r["choices"][0]["finish_reason"], "stop")
        self.assertEqual(r["usage"]["completion_tokens"], len(PLAIN))

    def test_content_parts_are_joined(self):
        r = post(self.port, {"messages": [{"role": "user", "content": [
            {"type": "text", "text": "hi"}]}]})
        self.assertTrue(r["choices"][0]["message"]["content"].endswith("Paris."))

    def test_max_tokens_sets_finish_length(self):
        r = post(self.port, {"messages": [{"role": "user", "content": "hi"}],
                             "max_tokens": 3})
        self.assertEqual(r["choices"][0]["finish_reason"], "length")
        self.assertEqual(r["usage"]["completion_tokens"], 3)

    def test_sse_plain(self):
        ch = sse(self.port, {"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(content_of(ch), "The capital of France is Paris.")
        self.assertEqual(ch[0]["choices"][0]["delta"].get("role"), "assistant")
        self.assertEqual(ch[-1]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(ch[-1]["usage"]["total_tokens"], 11 + len(PLAIN))

    def test_thinking_is_suppressed_from_content(self):
        ch = sse(self.port, {"messages": [{"role": "user", "content": "hi"}],
                             "enable_thinking": True})
        self.assertEqual(content_of(ch), "The capital is Paris.")
        reasoning = reasoning_of(ch)
        self.assertTrue(reasoning.startswith("We need the capital"), repr(reasoning))
        self.assertTrue(reasoning.endswith("."), repr(reasoning))

        r = post(self.port, {"messages": [{"role": "user", "content": "hi"}],
                             "enable_thinking": True})
        message = r["choices"][0]["message"]
        self.assertEqual(message["content"], "The capital is Paris.")
        self.assertTrue(message["reasoning_content"].startswith("We need"))

    def test_cut_off_reasoning_is_not_passed_off_as_the_answer(self):
        r = post(self.port, {"messages": [{"role": "user", "content": "trunc"}],
                             "enable_thinking": True, "max_tokens": 8})
        message = r["choices"][0]["message"]
        self.assertFalse(message.get("content") or "")
        self.assertTrue(message["reasoning_content"].startswith("We need to think"))

        ch = sse(self.port, {"messages": [{"role": "user", "content": "trunc"}],
                             "enable_thinking": True, "max_tokens": 8})
        self.assertEqual(content_of(ch), "")
        self.assertTrue(reasoning_of(ch).startswith("We need to think"))

    def test_tool_call_is_parsed_and_not_leaked(self):
        r = post(self.port, {"messages": [{"role": "user", "content": "北京?"}],
                             "tools": TOOLS_SCHEMA})
        message = r["choices"][0]["message"]
        call = (message.get("tool_calls") or [{}])[0].get("function", {})
        self.assertEqual(r["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(call.get("name"), "get_weather")
        self.assertEqual(json.loads(call["arguments"])["city"], "北京")

    def test_sse_tool_fragments_accumulate_to_the_non_stream_call(self):
        ch = sse(self.port, {"messages": [{"role": "user", "content": "北京?"}],
                             "tools": TOOLS_SCHEMA})
        once = post(self.port, {"messages": [{"role": "user", "content": "北京?"}],
                                "tools": TOOLS_SCHEMA}
                    )["choices"][0]["message"]["tool_calls"][0]
        acc = tool_calls_of(ch)
        self.assertGreaterEqual(len(fragments_of(ch)), 4)
        self.assertEqual(acc[0]["function"]["name"], "get_weather")
        self.assertEqual(acc[0]["type"], "function")
        self.assertTrue(acc[0]["id"])
        self.assertEqual(acc[0]["function"]["arguments"], once["function"]["arguments"])
        self.assertEqual(json.loads(acc[0]["function"]["arguments"])["city"], "北京")
        first = fragments_of(ch)[0]
        self.assertEqual((first.get("function") or {}).get("name"), "get_weather")
        self.assertTrue(first.get("id"))
        self.assertEqual(content_of(ch), "Let me check.")
        self.assertNotIn("<tool_call>", content_of(ch))

    def test_split_marker_is_held_back(self):
        r = post(self.port, {"messages": [{"role": "user", "content": "split it please"}],
                             "tools": TOOLS_SCHEMA})
        self.assertEqual((r["choices"][0]["message"].get("tool_calls") or [{}])[0]
                         ["function"]["name"], "get_weather")
        ch = sse(self.port, {"messages": [{"role": "user", "content": "split it please"}],
                             "tools": TOOLS_SCHEMA})
        self.assertEqual(content_of(ch), "Let me check.")

    def test_fragmented_arguments_match_the_non_stream_arguments(self):
        cases = (("string value split across tokens", "fragcity"),
                 ("integer values", "fragint"),
                 ("value with quotes/backslash/newline", "fragesc"))
        for label, text in cases:
            with self.subTest(case=label):
                got = tool_calls_of(sse(self.port, {
                    "messages": [{"role": "user", "content": text}],
                    "tools": TOOLS_SCHEMA}))
                want = post(self.port, {
                    "messages": [{"role": "user", "content": text}],
                    "tools": TOOLS_SCHEMA})["choices"][0]["message"]["tool_calls"]
                self.assertEqual([c["function"]["arguments"] for c in got],
                                 [c["function"]["arguments"] for c in want])
                self.assertEqual(len(got), len(want))
                json.loads(got[0]["function"]["arguments"])   # valid JSON, no duplicates

    def test_tool_choice_none_drops_tools(self):
        r = post(self.port, {"messages": [{"role": "user", "content": "北京?"}],
                             "tools": TOOLS_SCHEMA, "tool_choice": "none"})
        message = r["choices"][0]["message"]
        self.assertEqual(r["choices"][0]["finish_reason"], "stop")
        self.assertFalse(message.get("tool_calls"))
        self.assertEqual(message["content"], "The capital of France is Paris.")
        self.assertFalse(SEEN[-1].get("tools"))

    def test_tool_choice_required_forces_the_opener_and_returns_the_call(self):
        r = post(self.port, {"messages": [{"role": "user", "content": "北京?"}],
                             "tools": TOOLS_SCHEMA, "tool_choice": "required"})
        self.assertEqual(SEEN[-1].get("tool_prefix"), "<tool_call>\n")
        message = r["choices"][0]["message"]
        self.assertEqual(r["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual((message.get("tool_calls") or [{}])[0]["function"]["name"],
                         "get_weather")
        self.assertFalse(message.get("content") or "")
        self.assertNotIn("<tool_call>", json.dumps(message))

        ch = sse(self.port, {"messages": [{"role": "user", "content": "北京?"}],
                             "tools": TOOLS_SCHEMA, "tool_choice": "required"})
        calls = tool_calls_of(ch)
        self.assertEqual(content_of(ch), "")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        self.assertGreaterEqual(len(fragments_of(ch)), 2)

    def test_named_tool_choice_forces_that_function(self):
        r = post(self.port, {"messages": [{"role": "user", "content": "北京?"}],
                             "tools": TOOLS_SCHEMA,
                             "tool_choice": {"type": "function",
                                             "function": {"name": "get_weather"}}})
        self.assertEqual(SEEN[-1].get("tool_prefix"), "<tool_call>\n<function=get_weather>\n")
        self.assertEqual((r["choices"][0]["message"].get("tool_calls") or [{}])[0]
                         ["function"]["name"], "get_weather")

    def test_legacy_functions_pair_maps_onto_tools(self):
        r = post(self.port, {
            "messages": [{"role": "user", "content": "北京?"}],
            "functions": [{"name": "get_weather", "parameters": {"type": "object"}}],
            "function_call": "required"})
        self.assertEqual(SEEN[-1].get("tool_prefix"), "<tool_call>\n")
        self.assertTrue(SEEN[-1].get("tools"))
        self.assertEqual((r["choices"][0]["message"].get("tool_calls") or [{}])[0]
                         ["function"]["name"], "get_weather")

    def test_parallel_tool_calls_flag(self):
        r = post(self.port, {"messages": [{"role": "user", "content": "two calls"}],
                             "tools": TOOLS_SCHEMA})
        self.assertEqual(len(r["choices"][0]["message"]["tool_calls"]), 2)
        r = post(self.port, {"messages": [{"role": "user", "content": "two calls"}],
                             "tools": TOOLS_SCHEMA, "parallel_tool_calls": False})
        self.assertEqual(len(r["choices"][0]["message"]["tool_calls"]), 1)

    def test_stream_include_usage(self):
        ch = sse(self.port, {"messages": [{"role": "user", "content": "hi"}],
                             "stream_options": {"include_usage": True}})
        usage_chunks = [c for c in ch if c["choices"] == []]
        self.assertEqual(len(usage_chunks), 1)
        self.assertEqual(usage_chunks[0]["usage"]["total_tokens"], 11 + len(PLAIN))
        self.assertTrue(all(c.get("usage", "absent") is None
                            for c in ch if c["choices"]))
        self.assertEqual(content_of(ch), "The capital of France is Paris.")
