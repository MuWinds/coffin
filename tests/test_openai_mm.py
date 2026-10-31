"""OpenAI adapter 的图片（``image_url``）路径自测，不需要 GPU。

Module: ``coffin/tests``。``test_openai.py`` 的多模态对照版：导入 OpenAI
adapter，用 socket 桩代替引擎把整条链路跑一遍。

Constraints:
    * 不能 import ``support``：那个模块在 import 时就调用
      ``torch.cuda.is_available()``，别的进程占着卡时会阻塞，而协议测试没有
      理由排在 GPU 后面。所以 ``MKL_THREADING_LAYER`` 和 ``sys.path`` 都由
      本文件自己设。同样因为如此，adapter 必须在 ``sys.path`` 准备好之后才
      import，这个顺序正是那条 import 带 ``# noqa: E402`` 的原因。
    * 真正解码 PNG 字节，不打桩：被测的正是 base64 与大小限制这条路径，
      假 payload 会把它整段跳过。
    * 每个拒绝都断言发生在调用引擎之前：被拒绝的图片若仍到达 socket，既是
      资源泄漏，生成完才回来的 400 也只是一次浪费掉的请求。
    * ``srv.STATE`` 和 ``MAX_*`` 限制原地改、在 ``finally`` 里复原；HTTP
      服务器跑在本进程的线程里，漏掉的补丁会改变后续测试量到的东西。

被测的是刻意分开的两件事：

* **本进程的职责** —— 从 parts 数组里解析 ``data:`` URL，替换成模板的
  ``{"type": "image"}`` 标记，执行数量/大小上限，并在要求引擎生成之前把
  每一种拒绝变成可读的 ``400``。socket 桩扮演引擎，所以测试在哪都能跑
  （与 ``test_openai.py`` 同一手法）。
* **与引擎的契约** —— 图片请求必须以 ``images=[{"b64": ..., "mime": ...}]``
  出现在 socket JSON 上，messages 保持交错顺序，``usage.prompt_tokens``
  必须是引擎的 ``n_prompt``（它已经算上了图片 token）。

无图回归和新行为同等重要：纯文本请求的 socket 报文必须与图片功能出现之前
逐字节一致，这一点直接在捕获到的请求上断言。
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

# 纯协议测试：socket 桩扮演引擎，不需要 DCU。也不 import ``support``——那个
# 模块在 import 时调用 ``torch.cuda.is_available()``，别的进程占着卡时它会
# 阻塞（挂在繁忙 GPU 上的测试套件没人会跑）。``support`` 本来会做的设置只有
# 下面这两件，就地做完。
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
_ENGINE_DIR = Path(__file__).resolve().parent.parent
if str(_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(_ENGINE_DIR))

# sys.path 在上一行才补齐，这条 import 必须排在它之后，否则导入就失败。
from entrypoints.openai import server as srv  # noqa: E402

#: 一张 1x1 的不透明白色 PNG。用真实字节，base64/解码路径才会被走到而不是
#: 被打桩；图片内容本身在这里从不重要。
PNG_1PX = base64.b64decode(
    b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFAAH/"
    b"q842iQAAAABJRU5ErkJggg==")
PNG_B64 = base64.b64encode(PNG_1PX).decode()
DATA_URL = f"data:image/png;base64,{PNG_B64}"

#: 引擎为一幅图报告的 token 数：占位符展开成这么多 token，prompt 也正好按这个
#: 数增长。
IMAGE_TOKENS = 64
PLAIN = ["The", " image", " says", " SCNET", "."]

SEEN: list[dict] = []
_MODULE: dict = {}


def _pieces(req: dict) -> tuple[list[str], bool]:
    """预先定好的回答，像引擎那样按 ``max_new`` 截断。"""
    pieces = PLAIN
    n = int(req.get("max_new", -1))
    return (pieces[:n] if n >= 0 else pieces), False


def _text_of(req: dict) -> str:
    """把请求 messages 里所有 text 片段拼成一串，两种 content 形状都覆盖。"""
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
    """11 个文本 token，再加每个图片片段一幅图的量，与真实引擎一致。"""
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
    """在 ``port`` 上按引擎的行协议服务，直到 ``stop`` 被置位。

    0.5 秒的 accept 超时让循环有机会看到 ``stop``；阻塞式 accept 会把
    ``tearDownModule`` 永久挂住。1 MB 的 recv 缓冲足够装下大小限制测试发出的
    base64 payload。
    """
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
    """发一个非流式请求并解析 JSON 回复；4xx 以异常形式向外传播。"""
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def post_status(port: int, body: dict) -> tuple[int, dict]:
    """发请求并返回 ``(status, body)``，4xx/5xx 也不抛异常。"""
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def sse(port: int, body: dict) -> list[dict]:
    """发一个流式请求，返回解析后的 ``data:`` 分块，去掉终止标记。"""
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
    """为整个模块起一个引擎桩和一个 HTTP 服务器，并等到端口绑定完成。

    ``STATE`` 是原地改而不是重新构造，因为那个 dict 就是服务器的进程级配置；
    每个测试把自己动过的部分复原。
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    engine_port = probe.getsockname()[1]
    probe.close()
    stop = threading.Event()
    thread = threading.Thread(target=_fake_engine, args=(engine_port, stop), daemon=True)
    thread.start()
    for _ in range(50):                       # 等桩把端口绑上
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
    """停掉引擎桩并关掉 HTTP 服务器，让下一轮能干净地重新绑定。"""
    _MODULE["stop"].set()
    _MODULE["httpd"].shutdown()
    _MODULE["httpd"].server_close()
    _MODULE["thread"].join(timeout=2)


class _TcpEngine(srv.EngineClient):
    """走 TCP 的 ``EngineClient``（开发机可能没有 ``AF_UNIX``）。"""

    def _open(self):
        conn = socket.create_connection(("127.0.0.1", int(self.sock_path)),
                                        timeout=self.timeout)
        return conn, conn.makefile("rb")


class UnitTest(unittest.TestCase):
    """纯函数，不碰 socket：离开本进程的到底是什么。"""

    def test_data_url_is_split_into_b64_and_the_template_marker(self):
        """引擎存的是字节而不是 URL：data URL 必须原地变成 ``{"b64", "mime"}``，再加模板自己的标记。"""
        msgs, images, err = srv._split_messages([image_message("hi")])
        self.assertIsNone(err)
        self.assertEqual(len(images), 1)
        self.assertEqual(base64.b64decode(images[0]["b64"]), PNG_1PX)
        self.assertEqual(images[0]["mime"], "image/png")
        # 顺序保持：图片标记排在 text 片段之前
        content = msgs[0]["content"]
        self.assertEqual([p["type"] for p in content], ["image", "text"])
        self.assertEqual(content[1]["text"], "hi")

    def test_http_url_is_refused_with_a_readable_message(self):
        """拉取远程 URL 既是 SSRF 也是延迟黑洞，所以服务器直接拒绝而不下载；报错信息里会说清这一点。"""
        msgs, images, err = srv._split_messages([{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": "https://example.com/a.png"}}]}])
        self.assertEqual(msgs, [])
        self.assertEqual(images, [])
        self.assertIn("data:", err)
        self.assertIn("does not fetch", err)

    def test_video_parts_are_refused(self):
        """``video`` 和 ``video_url`` 是合法的 OpenAI 片段类型，但本引擎没有视觉塔的视频部分，接受它们等于悄悄丢帧。"""
        for kind in ("video", "video_url"):
            with self.subTest(kind=kind):
                _m, _i, err = srv._split_messages([{"role": "user", "content": [
                    {"type": kind, kind: {"url": "data:video/mp4;base64,AAAA"}}]}])
                self.assertIsNotNone(err)
                self.assertIn("video", err)

    def test_system_message_with_an_image_is_refused(self):
        """system 轮在对话模板里没有图片槽位；放在那里的图片会把后面每个占位符都错位。"""
        _m, _i, err = srv._split_messages([
            {"role": "system", "content": [{"type": "image_url",
                                            "image_url": {"url": DATA_URL}}]},
            {"role": "user", "content": "hi"}])
        self.assertIsNotNone(err)
        self.assertIn("system", err)
        self.assertIn("template", err)

    def test_too_many_images_is_refused(self):
        """这个上限约束的是视觉塔的单请求显存，所以必须在解码任何一张图之前生效。"""
        n = srv.MAX_IMAGES + 1
        _m, _i, err = srv._split_messages([image_message(n=n)])
        self.assertIsNotNone(err)
        self.assertIn("too many images", err)

    def test_oversized_image_is_refused(self):
        """单图字节上限用来防止一个请求独占上传路径；这里改的是限制值，不是图片。"""
        old = srv.MAX_IMAGE_BYTES
        srv.MAX_IMAGE_BYTES = len(PNG_1PX) - 1
        try:
            _m, _i, err = srv._split_messages([image_message()])
        finally:
            srv.MAX_IMAGE_BYTES = old
        self.assertIsNotNone(err)
        self.assertIn("per-image limit", err)

    def test_total_base64_budget_is_refused(self):
        """只有单图上限的话，N 张合法图片加起来仍可能超过预算，所以请求总量另查一遍。"""
        old = srv.MAX_IMAGE_TOTAL_BYTES
        srv.MAX_IMAGE_TOTAL_BYTES = len(DATA_URL) * 2 - 1
        try:
            _m, _i, err = srv._split_messages([image_message(n=2)])
        finally:
            srv.MAX_IMAGE_TOTAL_BYTES = old
        self.assertIsNotNone(err)
        self.assertIn("per-request limit", err)

    def test_invalid_base64_and_unknown_mime_are_refused(self):
        """畸形 payload 必须在这里就以 400 失败；解码错误漏到引擎会在生成中途变成 500。"""
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
        """文本请求比图片先存在，两种 content 形状都必须归一到同一个字符串，且不带图片列表。"""
        plain = [{"role": "user", "content": "hello"},
                 {"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        msgs, images, err = srv._split_messages(plain)
        self.assertIsNone(err)
        self.assertEqual(images, [])
        norm = srv._norm_messages(msgs)
        self.assertEqual([m["content"] for m in norm], ["hello", "hi"])

    def test_template_content_keeps_a_parts_array_only_when_it_has_an_image(self):
        """模板对文本接受纯字符串；只有 parts 数组里含图片标记时，数组才必须原样保留。"""
        self.assertEqual(srv._template_content("hi"), "hi")
        self.assertEqual(srv._template_content([{"type": "text", "text": "a"},
                                                {"type": "text", "text": "b"}]), "ab")
        self.assertEqual(
            srv._template_content([{"type": "image"}, {"type": "text", "text": "x"}]),
            [{"type": "image"}, {"type": "text", "text": "x"}])


class HttpTest(unittest.TestCase):
    """走 HTTP 的端到端测试：引擎收到的 socket 报文，以及回复。"""

    @property
    def port(self) -> int:
        return _MODULE["port"]

    def test_image_request_reaches_the_engine_with_images(self):
        """socket 契约：``images`` 里是字节加 mime，messages 里是标记（绝不是 URL），prompt token 数包含图片。"""
        SEEN.clear()
        r = post(self.port, {"model": "qwen3.5-4b",
                             "messages": [image_message("what is in scnetimage?")]})
        req = SEEN[-1]
        self.assertEqual(len(req["images"]), 1)
        self.assertEqual(base64.b64decode(req["images"][0]["b64"]), PNG_1PX)
        self.assertEqual(req["images"][0]["mime"], "image/png")
        # 模板要的占位符，不是 URL
        self.assertEqual(req["messages"][0]["content"][0], {"type": "image"})
        self.assertNotIn("data:", json.dumps(req["messages"]))
        # usage 之所以算上图片 token，是因为它直接取引擎的 n_prompt
        self.assertEqual(r["usage"]["prompt_tokens"], 11 + IMAGE_TOKENS)
        self.assertEqual(r["choices"][0]["message"]["content"],
                         "The image says SCNET.")

    def test_no_image_request_payload_is_unchanged(self):
        """纯文本请求的 socket JSON 必须与图片功能出现之前完全一致。

        断言方式是与字面 payload 精确相等，而不是只查有没有 images 键：图片
        功能的全部意义就在于文本路径不付任何代价，多一个空的 ``images: []``
        或 content 数组被改了形状，松一点的检查根本看不出来（而这两样都会改变
        下游的基数块键）。
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
        """不带图片的 parts 数组仍算原来的文本请求；留成 list 只会白白改变基数块键。"""
        SEEN.clear()
        post(self.port, {"messages": [{"role": "user", "content": [
            {"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]})
        self.assertEqual(SEEN[-1]["messages"][0]["content"], "ab")
        self.assertNotIn("images", SEEN[-1])

    def test_multi_image_order_is_preserved(self):
        """图片是位置相关的：相对文本重排会悄悄把每道题配到错的那张图上。"""
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
        # 引擎按客户端给的顺序，每张图看到一个标记
        kinds = [p["type"] for p in req["messages"][0]["content"]]
        self.assertEqual(kinds, ["text", "image", "text", "image"])
        self.assertEqual([p.get("text") for p in req["messages"][0]["content"]
                          if p["type"] == "text"], ["first", "second"])
        self.assertEqual(base64.b64decode(req["images"][1]["b64"]), PNG_1PX + b"\x00")
        self.assertEqual(r["usage"]["prompt_tokens"], 11 + 2 * IMAGE_TOKENS)

    def test_image_request_streams(self):
        """流式路径必须携带与非流式相同的 usage，否则只读流的客户端会把 prompt 算少。"""
        SEEN.clear()
        chunks = sse(self.port, {"messages": [image_message("scnetimage")]})
        self.assertEqual(content_of(chunks), "The image says SCNET.")
        self.assertEqual(chunks[-1]["usage"]["prompt_tokens"], 11 + IMAGE_TOKENS)
        self.assertEqual(len(SEEN[-1]["images"]), 1)

    def test_error_codes_are_400_and_readable(self):
        """每种拒绝都必须是客户端能展示的 400，且不能有任何东西被转发：被拒的图片不得到达引擎。"""
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
                # 什么都没转发：被拒绝的图片从不到达引擎
                self.assertFalse([r for r in SEEN if r.get("images")],
                                 f"{label} leaked an image to the engine")

    def test_too_many_images_is_400(self):
        """单元级的上限在 HTTP 上也得能触发；回 200 却在 body 里塞错误会打坏客户端的错误处理。"""
        status, body = post_status(self.port,
                                   {"messages": [image_message(n=srv.MAX_IMAGES + 1)]})
        self.assertEqual(status, 400, body)
        self.assertIn("too many images", body["error"]["message"])

    def test_no_vision_server_answers_503(self):
        """以 ``--no-vision`` 启动的服务器必须回 503 并点明这个开关，而不是在预填充内部某处失败。"""
        srv.STATE["vision"] = False
        try:
            status, body = post_status(self.port,
                                       {"messages": [image_message()]})
        finally:
            srv.STATE["vision"] = True
        self.assertEqual(status, 503, body)
        self.assertIn("--no-vision", body["error"]["message"])

    def test_text_only_request_unchanged_with_vision_enabled(self):
        """开启视觉不能让文本路径付代价：token 数相同，finish reason 也相同。"""
        r = post(self.port, {"messages": [{"role": "user", "content": "hi"}],
                             "max_tokens": 3})
        self.assertEqual(r["usage"]["prompt_tokens"], 11)
        self.assertEqual(r["usage"]["completion_tokens"], 3)
        self.assertEqual(r["choices"][0]["finish_reason"], "length")


class EngineErrorTest(unittest.TestCase):
    """引擎侧拒绝必须带着它自己的状态码原样透出。"""

    @property
    def port(self) -> int:
        return _MODULE["port"]

    def test_engine_400_is_forwarded_on_the_non_stream_path(self):
        """引擎自己的状态码和消息必须原封不动送到客户端；压成 500 就藏掉了是哪个组件拒绝的。"""
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
        """``start`` 之前的拒绝必须是 HTTP 状态码，而不是 200 里的 SSE 错误事件：流式客户端会直接忽略后者。"""
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
        # ``start`` 之前的拒绝是真 HTTP 状态码，不是 200 的 SSE 事件
        self.assertEqual(status, 503, body)
        self.assertIn("vision tower", body["error"]["message"])


if __name__ == "__main__":
    unittest.main()

class ToolRoleImageTest(unittest.TestCase):
    """agent 把截图作为 *tool* 结果返回时，链路必须通。

    图在 IDE/agent 客户端上实际就是这样进来的：模型要一张截图，工具把截屏
    放进 ``tool`` 消息送回。服务器以前会用 400 拒绝（"only user turns accept
    image parts"），流式客户端因为收不到这个错误，最后只看到 "Stream
    completed without a choice"。这个 bug 的两半都在这里钉住。
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
        """agent 客户端就是这样交付截图的，拒绝它等于废掉整个工具循环场景。"""
        msgs, images, err = srv._split_messages(self._tool_messages())

        self.assertIsNone(err)
        self.assertEqual(len(images), 1)
        self.assertEqual(len(msgs), 3)

    def test_system_role_images_are_rejected_with_advice(self):
        """拒绝信息必须点出 system 角色，客户端才知道要把图片挪进 user 轮。"""
        msgs, images, err = srv._split_messages(
            [{"role": "system",
              "content": [{"type": "image_url", "image_url": {"url": DATA_URL}}]}])

        self.assertEqual(msgs, [])
        self.assertIn("system", err)

    def test_assistant_role_images_are_rejected(self):
        """模型「发出」的图片既没有模板槽位也没有视觉塔输入，必须拒绝而不是悄悄丢掉。"""
        msgs, images, err = srv._split_messages(
            [{"role": "assistant",
              "content": [{"type": "image_url", "image_url": {"url": DATA_URL}}]}])

        self.assertEqual(msgs, [])
        self.assertIn("assistant", err)

