"""Self-test for the image (``image_url``) path of ``openai_server.py`` (no GPU).

Two things are under test and they are deliberately separated:

* **this process's job** -- parsing ``data:`` URLs out of a parts array, replacing
  them with the template's ``{"type": "image"}`` marker, enforcing the count/size
  limits, and turning every refusal into a readable ``400`` *before* the engine is
  asked to generate anything.  A stub engine socket plays the engine, so the tests
  run anywhere (the same trick as ``test_openai.py``).
* **the contract with the engine** -- an image request must arrive as
  ``images=[{"b64": ..., "mime": ...}]`` on the socket JSON, the messages must keep
  their interleaved order, and ``usage.prompt_tokens`` must be the engine's
  ``n_prompt`` (which already counts the image tokens).

The no-image regression matters as much as the new behaviour: a text-only request
must produce byte-for-byte the same socket payload it produced before images
existed, which is asserted directly on the captured request.
"""

from __future__ import annotations

import base64
import json
import os
import socket
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

# This is a pure-protocol test: a stub socket plays the engine, so nothing here
# needs a DCU -- and nothing here imports ``support``, because that module calls
# ``torch.cuda.is_available()`` at import time, which *blocks* while another
# process holds the card (a test suite that hangs on a busy GPU is a test suite
# nobody runs).  Only the two things ``support`` would have set up are done here.
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
_ENGINE_DIR = Path(__file__).resolve().parent.parent
if str(_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(_ENGINE_DIR))

import openai_server as srv  # noqa: E402  (after sys.path is prepared)

#: A tiny valid PNG (1x1, opaque white).  Real bytes, so the base64/decoding path
#: is exercised rather than mocked; the *image* content never matters here.
PNG_1PX = base64.b64decode(
    b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFAAH/"
    b"q842iQAAAABJRU5ErkJggg==")
PNG_B64 = base64.b64encode(PNG_1PX).decode()
DATA_URL = f"data:image/png;base64,{PNG_B64}"

#: What the engine reports for one image: its placeholders expand to this many
#: tokens, and the prompt grows by exactly that much.
IMAGE_TOKENS = 64
PLAIN = ["The", " image", " says", " SCNET", "."]

SEEN: list[dict] = []
_MODULE: dict = {}


def _pieces(req: dict) -> tuple[list[str], bool]:
    """The canned answer, truncated to ``max_new`` like the engine would."""
    pieces = PLAIN
    n = int(req.get("max_new", -1))
    return (pieces[:n] if n >= 0 else pieces), False


def _text_of(req: dict) -> str:
    out = []
    for m in req.get("messages") or []:
        content = m.get("content")
        if isinstance(content, str):
            out.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    out.append(part.get("text") or "")
    return " ".join(out)


def _n_prompt(req: dict) -> int:
    """11 text tokens + one image's worth per image part, like the real engine."""
    base = 11
    return base + IMAGE_TOKENS * len(req.get("images") or [])


def _reply(req: dict) -> dict:
    pieces, _opens = _pieces(req)
    return {"ok": True, "text": "".join(pieces), "thinking": "", "raw": "".join(pieces),
            "tool_calls": [], "forced_tool": bool(req.get("tool_prefix")),
            "n_prompt": _n_prompt(req), "n_new": len(pieces),
            "max_new": int(req.get("max_new", -1)), "window_full": False,
            "cached_tokens": 0, "prefill_ms": 120.0, "decode_tok_s": 42.0,
            "spec_k": 0, "spec": None,
            "n_images": len(req.get("images") or []),
            "image_tokens": IMAGE_TOKENS * len(req.get("images") or [])}


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
                chunk = conn.recv(1 << 20)
                if not chunk:
                    break
                buf += chunk
            try:
                req = json.loads(buf.decode() or "{}")
                if req.get("op") != "ping":
                    SEEN.append(req)
                if req.get("op") == "ping":
                    conn.sendall((json.dumps({
                        "ok": True, "graph": True, "max_seq": 4096,
                        "vision": _MODULE.get("vision", True),
                    }) + "\n").encode())
                elif req.get("op") == "chat_stream":
                    pieces, opens = _pieces(req)
                    done = _reply(req)
                    done["type"] = "done"
                    conn.sendall((json.dumps({"type": "start",
                                              "n_prompt": _n_prompt(req),
                                              "prompt_opens_think": opens,
                                              "forced_tool": False}) + "\n").encode())
                    for piece in pieces:
                        conn.sendall((json.dumps({"type": "token", "text": piece})
                                      + "\n").encode())
                    conn.sendall((json.dumps(done, ensure_ascii=False) + "\n").encode())
                else:
                    conn.sendall((json.dumps(_reply(req), ensure_ascii=False)
                                  + "\n").encode())
            except Exception as exc:  # noqa: BLE001
                conn.sendall((json.dumps({"ok": False, "error": str(exc)}) + "\n").encode())
    server.close()


def post(port: int, body: dict) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def post_status(port: int, body: dict) -> tuple[int, dict]:
    """POST and return ``(status, body)`` even for a 4xx/5xx (no exception)."""
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


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


def content_of(chunks: list[dict]) -> str:
    return "".join(c["choices"][0]["delta"].get("content") or ""
                   for c in chunks if c["choices"])


def image_message(text: str = "What does it say?", n: int = 1) -> dict:
    return {"role": "user", "content": [{"type": "image_url",
                                         "image_url": {"url": DATA_URL}}
                                        for _ in range(n)]
            + [{"type": "text", "text": text}]}


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

    srv.STATE.update({"engine": _TcpEngine(engine_port),
                      "model_path": "/models/Qwen3.5-4B",
                      "served_name": "qwen3.5-4b", "enable_thinking": False,
                      "vision": True, "max_decode_seconds": 0,
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


class _TcpEngine(srv.EngineClient):
    """``EngineClient`` over TCP (a dev box may have no ``AF_UNIX``)."""

    def _open(self):
        conn = socket.create_connection(("127.0.0.1", int(self.sock_path)),
                                        timeout=self.timeout)
        return conn, conn.makefile("rb")


class UnitTest(unittest.TestCase):
    """The pure functions, without the socket: what exactly leaves this process."""

    def test_data_url_is_split_into_b64_and_the_template_marker(self):
        msgs, images, err = srv._split_messages([image_message("hi")])
        self.assertIsNone(err)
        self.assertEqual(len(images), 1)
        self.assertEqual(base64.b64decode(images[0]["b64"]), PNG_1PX)
        self.assertEqual(images[0]["mime"], "image/png")
        # order preserved: the image marker comes before the text part
        content = msgs[0]["content"]
        self.assertEqual([p["type"] for p in content], ["image", "text"])
        self.assertEqual(content[1]["text"], "hi")

    def test_http_url_is_refused_with_a_readable_message(self):
        msgs, images, err = srv._split_messages([{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": "https://example.com/a.png"}}]}])
        self.assertEqual(msgs, [])
        self.assertEqual(images, [])
        self.assertIn("data:", err)
        self.assertIn("does not fetch", err)

    def test_video_parts_are_refused(self):
        for kind in ("video", "video_url"):
            with self.subTest(kind=kind):
                _m, _i, err = srv._split_messages([{"role": "user", "content": [
                    {"type": kind, kind: {"url": "data:video/mp4;base64,AAAA"}}]}])
                self.assertIsNotNone(err)
                self.assertIn("video", err)

    def test_system_message_with_an_image_is_refused(self):
        _m, _i, err = srv._split_messages([
            {"role": "system", "content": [{"type": "image_url",
                                            "image_url": {"url": DATA_URL}}]},
            {"role": "user", "content": "hi"}])
        self.assertIsNotNone(err)
        self.assertIn("system", err)
        self.assertIn("template", err)

    def test_too_many_images_is_refused(self):
        n = srv.MAX_IMAGES + 1
        _m, _i, err = srv._split_messages([image_message(n=n)])
        self.assertIsNotNone(err)
        self.assertIn("too many images", err)

    def test_oversized_image_is_refused(self):
        old = srv.MAX_IMAGE_BYTES
        srv.MAX_IMAGE_BYTES = len(PNG_1PX) - 1
        try:
            _m, _i, err = srv._split_messages([image_message()])
        finally:
            srv.MAX_IMAGE_BYTES = old
        self.assertIsNotNone(err)
        self.assertIn("per-image limit", err)

    def test_total_base64_budget_is_refused(self):
        old = srv.MAX_IMAGE_TOTAL_BYTES
        srv.MAX_IMAGE_TOTAL_BYTES = len(DATA_URL) * 2 - 1
        try:
            _m, _i, err = srv._split_messages([image_message(n=2)])
        finally:
            srv.MAX_IMAGE_TOTAL_BYTES = old
        self.assertIsNotNone(err)
        self.assertIn("per-request limit", err)

    def test_invalid_base64_and_unknown_mime_are_refused(self):
        cases = {
            "bad base64": "data:image/png;base64,!!!notbase64!!!",
            "no payload": "data:image/png;base64,",
            "not a url": "not-a-url",
            "svg": "data:image/svg+xml;base64," + PNG_B64,
        }
        for label, url in cases.items():
            with self.subTest(case=label):
                _m, _i, err = srv._split_messages([{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": url}}]}])
                self.assertIsNotNone(err, f"{label} should be refused")

    def test_text_only_content_is_untouched(self):
        plain = [{"role": "user", "content": "hello"},
                 {"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        msgs, images, err = srv._split_messages(plain)
        self.assertIsNone(err)
        self.assertEqual(images, [])
        norm = srv._norm_messages(msgs)
        self.assertEqual([m["content"] for m in norm], ["hello", "hi"])

    def test_template_content_keeps_a_parts_array_only_when_it_has_an_image(self):
        self.assertEqual(srv._template_content("hi"), "hi")
        self.assertEqual(srv._template_content([{"type": "text", "text": "a"},
                                                {"type": "text", "text": "b"}]), "ab")
        self.assertEqual(
            srv._template_content([{"type": "image"}, {"type": "text", "text": "x"}]),
            [{"type": "image"}, {"type": "text", "text": "x"}])


class HttpTest(unittest.TestCase):
    """End to end over HTTP: the socket payload the engine receives, and the reply."""

    @property
    def port(self) -> int:
        return _MODULE["port"]

    def test_image_request_reaches_the_engine_with_images(self):
        SEEN.clear()
        r = post(self.port, {"model": "qwen3.5-4b",
                             "messages": [image_message("what is in scnetimage?")]})
        req = SEEN[-1]
        self.assertEqual(len(req["images"]), 1)
        self.assertEqual(base64.b64decode(req["images"][0]["b64"]), PNG_1PX)
        self.assertEqual(req["images"][0]["mime"], "image/png")
        # the placeholder the template needs, not the URL
        self.assertEqual(req["messages"][0]["content"][0], {"type": "image"})
        self.assertNotIn("data:", json.dumps(req["messages"]))
        # usage counts the image tokens because it is the engine's n_prompt
        self.assertEqual(r["usage"]["prompt_tokens"], 11 + IMAGE_TOKENS)
        self.assertEqual(r["choices"][0]["message"]["content"],
                         "The image says SCNET.")

    def test_no_image_request_payload_is_unchanged(self):
        """A text-only request's socket JSON must be what it was before images.

        Asserted as *exact equality against the literal payload*, not just "no
        images key": the point of the image feature is that it costs the text path
        nothing, and a stray ``images: []`` or a re-shaped content array would be
        invisible to a looser check (and would change radix-block keys downstream).
        """
        SEEN.clear()
        post(self.port, {"messages": [{"role": "user", "content": "hi"}],
                         "max_tokens": 5})
        self.assertEqual(SEEN[-1], {
            "op": "chat",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": None, "tool_prefix": None, "enable_thinking": False,
            "max_new": 5, "temperature": 0.0, "max_seconds": None})

    def test_text_parts_array_still_flattens_to_a_string(self):
        SEEN.clear()
        post(self.port, {"messages": [{"role": "user", "content": [
            {"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]})
        self.assertEqual(SEEN[-1]["messages"][0]["content"], "ab")
        self.assertNotIn("images", SEEN[-1])

    def test_multi_image_order_is_preserved(self):
        SEEN.clear()
        other = base64.b64encode(PNG_1PX + b"\x00").decode()
        body = {"messages": [{"role": "user", "content": [
            {"type": "text", "text": "first"},
            {"type": "image_url", "image_url": {"url": DATA_URL}},
            {"type": "text", "text": "second"},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{other}"}}]}]}
        r = post(self.port, body)
        req = SEEN[-1]
        self.assertEqual(len(req["images"]), 2)
        # the engine sees one marker per image, in the client's order
        kinds = [p["type"] for p in req["messages"][0]["content"]]
        self.assertEqual(kinds, ["text", "image", "text", "image"])
        self.assertEqual([p.get("text") for p in req["messages"][0]["content"]
                          if p["type"] == "text"], ["first", "second"])
        self.assertEqual(base64.b64decode(req["images"][1]["b64"]), PNG_1PX + b"\x00")
        self.assertEqual(r["usage"]["prompt_tokens"], 11 + 2 * IMAGE_TOKENS)

    def test_image_request_streams(self):
        SEEN.clear()
        chunks = sse(self.port, {"messages": [image_message("scnetimage")]})
        self.assertEqual(content_of(chunks), "The image says SCNET.")
        self.assertEqual(chunks[-1]["usage"]["prompt_tokens"], 11 + IMAGE_TOKENS)
        self.assertEqual(len(SEEN[-1]["images"]), 1)

    def test_error_codes_are_400_and_readable(self):
        cases = {
            "http url": {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": "http://example.com/x.png"}}]},
            "video": {"role": "user", "content": [
                {"type": "video", "video": {"url": "data:video/mp4;base64,AAAA"}}]},
            "system image": {"role": "system", "content": [
                {"type": "image_url", "image_url": {"url": DATA_URL}}]},
            "broken payload": {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": "data:image/png;base64,@@@@@@@"}}]},
        }
        for label, message in cases.items():
            with self.subTest(case=label):
                SEEN.clear()
                status, body = post_status(self.port, {"messages": [message]})
                self.assertEqual(status, 400, f"{label}: {body}")
                self.assertEqual(body["error"]["code"], 400)
                self.assertTrue(body["error"]["message"])
                # nothing was forwarded: a refused image never reaches the engine
                self.assertFalse([r for r in SEEN if r.get("images")],
                                 f"{label} leaked an image to the engine")

    def test_too_many_images_is_400(self):
        status, body = post_status(self.port,
                                   {"messages": [image_message(n=srv.MAX_IMAGES + 1)]})
        self.assertEqual(status, 400, body)
        self.assertIn("too many images", body["error"]["message"])

    def test_no_vision_server_answers_503(self):
        srv.STATE["vision"] = False
        try:
            status, body = post_status(self.port,
                                       {"messages": [image_message()]})
        finally:
            srv.STATE["vision"] = True
        self.assertEqual(status, 503, body)
        self.assertIn("--no-vision", body["error"]["message"])

    def test_text_only_request_unchanged_with_vision_enabled(self):
        r = post(self.port, {"messages": [{"role": "user", "content": "hi"}],
                             "max_tokens": 3})
        self.assertEqual(r["usage"]["prompt_tokens"], 11)
        self.assertEqual(r["usage"]["completion_tokens"], 3)
        self.assertEqual(r["choices"][0]["finish_reason"], "length")


class EngineErrorTest(unittest.TestCase):
    """An engine-side rejection has to come back out with *its* status code."""

    @property
    def port(self) -> int:
        return _MODULE["port"]

    def test_engine_400_is_forwarded_on_the_non_stream_path(self):
        class _Rejecting(srv.EngineClient):
            def call(self, req):
                return {"ok": False, "error": "image placeholder mismatch",
                        "status": 400}

        real = srv.STATE["engine"]
        srv.STATE["engine"] = _Rejecting("ignored")
        try:
            status, body = post_status(self.port, {"messages": [image_message()]})
        finally:
            srv.STATE["engine"] = real
        self.assertEqual(status, 400, body)
        self.assertIn("placeholder mismatch", body["error"]["message"])
        self.assertEqual(body["error"]["type"], "invalid_request_error")

    def test_engine_503_is_forwarded_on_the_stream_path(self):
        class _Rejecting(srv.EngineClient):
            def stream(self, req):
                yield {"type": "error", "ok": False, "status": 503,
                       "error": "vision tower unavailable"}

        real = srv.STATE["engine"]
        srv.STATE["engine"] = _Rejecting("ignored")
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{self.port}/v1/chat/completions",
                data=json.dumps({"messages": [image_message()],
                                 "stream": True}).encode(),
                headers={"Content-Type": "application/json"})
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(req, timeout=30)
            status, body = ctx.exception.code, json.loads(ctx.exception.read())
        finally:
            srv.STATE["engine"] = real
        # a rejection before ``start`` is a real HTTP status, not a 200 SSE event
        self.assertEqual(status, 503, body)
        self.assertIn("vision tower", body["error"]["message"])


if __name__ == "__main__":
    unittest.main()

class ToolRoleImageTest(unittest.TestCase):
    """An agent that returns a screenshot as the *tool* result must work.

    This is how images actually arrive from an IDE/agent client: the model asks
    for a screenshot and the tool sends the capture back in a ``tool`` message.
    The server used to reject that with 400 ("only user turns accept image
    parts"), which a streaming client surfaced as "Stream completed without a
    choice" because the error never reached it.  Both halves of that bug are
    pinned here.
    """

    def _tool_messages(self):
        img = {"type": "image_url", "image_url": {"url": DATA_URL}}
        return [
            {"role": "user", "content": "take a screenshot"},
            {"role": "assistant", "content": "ok",
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "shot", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1",
             "content": [{"type": "text", "text": "result:"}, img]},
        ]

    def test_tool_role_images_are_accepted(self):
        msgs, images, err = srv._split_messages(self._tool_messages())

        self.assertIsNone(err)
        self.assertEqual(len(images), 1)
        self.assertEqual(len(msgs), 3)

    def test_system_role_images_are_rejected_with_advice(self):
        msgs, images, err = srv._split_messages(
            [{"role": "system",
              "content": [{"type": "image_url", "image_url": {"url": DATA_URL}}]}])

        self.assertEqual(msgs, [])
        self.assertIn("system", err)

    def test_assistant_role_images_are_rejected(self):
        msgs, images, err = srv._split_messages(
            [{"role": "assistant",
              "content": [{"type": "image_url", "image_url": {"url": DATA_URL}}]}])

        self.assertEqual(msgs, [])
        self.assertIn("assistant", err)

