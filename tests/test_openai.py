"""用一个假引擎套接字自测 OpenAI 入口，不需要 GPU。

Module: ``coffin/tests``。唯一走 HTTP 的测试：它导入 OpenAI 适配器和
``utils.tool_call``，接到桩引擎上，断言 OpenAI 报文格式，所以没有 DCU 的机器
也能跑。

Constraints:
    * 桩引擎讲 TCP 而不是 ``AF_UNIX``：没有 ``AF_UNIX`` 的开发机也能连上，与真正
      的客户端相比只有 ``_open`` 不同，传输层以上仍是真实的行协议。
    * ``SEEN`` 是模块级全局，测试读它的最后一条来断言服务端往下游发了什么；
      请求因此必须串行，这一点 unittest 已经保证。
    * 整个模块只有一个 HTTP 服务和一个桩引擎（``setUpModule``）：每个测试各绑
      一次端口既慢，又会留下 TIME_WAIT 状态的套接字。
    * 预置 token 流故意把标记和参数拆到不同片段里；只发完整标记的流测不到
      扣住不发的路径。

HTTP/SSE 这一层是纯协议代码，在哪都能验：桩引擎套接字播预置 token 流，把 OpenAI
适配器接上去，再断言 OpenAI 侧的响应——包括两处容易写错、生成正常时又看不出来
的地方：

* prompt 自己打开了 ``<think>`` 时，推理块要被抑制；
* ``<tool_call>`` 标记绝不漏进 ``content``，即使标记被拆到两个 token 片段里
  （过滤器把不完整的标记扣住不发）。

移植自 ``diag/check_openai.py``。
"""

from __future__ import annotations

import json
import socket
import threading
import time
import unittest
import urllib.request

# 副作用导入：设置 MKL_THREADING_LAYER，并把 coffin/ 加进 sys.path。
import support  # noqa: F401

from entrypoints.openai import server as srv
from utils.tool_call import parse_tool_calls, split_thinking


class TcpEngine(srv.EngineClient):
    """把 ``EngineClient`` 的传输层换成 TCP，而不是 AF_UNIX。"""

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
# tool_choice: required 时引擎在 prompt 里已经看到开标记，所以它生成并流式发出的
# 只是这次调用的正文部分。
CALL_BODY = CALL[len("<tool_call>"):]
TOOLS = ["Let", " me", " check", ".", CALL]
TOOLS_SPLIT = ["Let", " me", " check", ".", "<tool_ca", "ll>" + CALL[len("<tool_call>"):]]
CALL_FRAG_CITY = ["<tool_call>", "\n<function=", "get_weather", ">\n<parameter=", "city",
                  ">\n", "北", "京", "\n</par", "ameter>\n</fun", "ction>\n</tool", "_call>"]
CALL_FRAG_INT = ["<tool_call>", "\n<function=multiply>\n<parameter=a>\n1", "2\n</parameter>\n",
                 "<parameter=b>\n", "8", "\n</parameter>\n</function>\n</tool_call>"]
CALL_FRAG_ESC = ["<tool_call>", "\n<function=multiply>\n<parameter=x>\n", 'he said "hi"',
                 "\nline2\\path", "\n</parameter>\n</function>\n</tool_call>"]

SEEN: list[dict] = []          # 假引擎收到的每个请求
_MODULE: dict = {}


def _pieces(req: dict) -> tuple[list[str], bool]:
    """给一条请求挑出预置 token 流，外加「prompt 自己打开了 ``<think>``」标志。

    唯一的开关是 prompt 文本：桩引擎里没有模型，所以用例靠标记词（``"fragcity"``、
    ``"two"``、``"split"`` 等）来选，而不是看服务端算出了什么。
    """
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
    """桩引擎返回的非流式应答，形状与真实的 ``done`` payload 一致。

    ``_reply`` 复刻运行时 chat 拆分的那条「prompt 打开了思考」规则，这样测试断言的
    是服务端的输出，而不是自己重算一遍。
    """
    pieces, opens = _pieces(req)
    prefix = str(req.get("tool_prefix") or "")
    raw = prefix + "".join(pieces)              # 开标记已经落在 prompt 里
    thinking, answer = split_thinking(raw)
    if (req.get("enable_thinking") and not prefix
            and not thinking and "</think>" not in raw):
        thinking, answer = answer, ""           # 与运行时 chat 拆分同一条规则
    return {"ok": True, "text": answer, "thinking": thinking, "raw": raw,
            "tool_calls": [c.to_openai_dict() for c in parse_tool_calls(answer)],
            "forced_tool": bool(prefix),
            "n_prompt": 11, "n_new": len(pieces),
            "max_new": int(req.get("max_new", -1)), "window_full": False,
            "cached_tokens": 0,
            "prefill_ms": 120.0, "decode_tok_s": 42.0, "spec_k": 0, "spec": None}


def _fake_engine(port: int, stop: threading.Event) -> None:
    """在 ``port`` 上按真实引擎的行协议应答，直到 ``stop`` 被置位。

    accept 超时设成 0.5 s，循环才有机会看到 ``stop``；阻塞式 accept 会卡住
    ``tearDownModule``，端口一直占着不放。
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
    """发一条非流式的 ``/v1/chat/completions`` 请求，并解出 JSON 应答。"""
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def sse(port: int, body: dict) -> list[dict]:
    """发一条流式请求，返回解析后的 ``data:`` 块，不含终止符。

    ``[DONE]`` 哨兵在这里就被丢掉：它不带 JSON，而每个调用方只看解出来的块列表。
    """
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
    """向被测服务 GET 一个路径，并解出 JSON 正文。"""
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=30) as r:
        return json.loads(r.read())


def content_of(chunks: list[dict]) -> str:
    return "".join(c["choices"][0]["delta"].get("content") or ""
                   for c in chunks if c["choices"])


def reasoning_of(chunks: list[dict]) -> str:
    return "".join(c["choices"][0]["delta"].get("reasoning_content") or ""
                   for c in chunks if c["choices"])


def tool_calls_of(chunks: list[dict]) -> list[dict]:
    """按 OpenAI 客户端的做法累积流式 ``tool_calls`` 片段。"""
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
    """整个模块只用一套桩引擎和 HTTP 服务。

    发第一条请求之前要先等桩线程起来：否则 HTTP 层会和 bind 抢跑，在慢机器上
    间歇性失败。
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    engine_port = probe.getsockname()[1]
    probe.close()
    stop = threading.Event()
    thread = threading.Thread(target=_fake_engine, args=(engine_port, stop), daemon=True)
    thread.start()
    for _ in range(50):                       # 等桩引擎完成绑定
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
    """停掉桩引擎、关掉 HTTP 服务，让重跑能绑到同样的状态。"""
    _MODULE["stop"].set()
    _MODULE["httpd"].shutdown()
    _MODULE["httpd"].server_close()
    _MODULE["thread"].join(timeout=2)


class OpenAIServerTest(unittest.TestCase):
    """OpenAI 兼容面必须对上客户端解析的报文格式，而不只是产出文本。"""

    @property
    def port(self) -> int:
        """``setUpModule`` 启动的服务的 HTTP 端口。"""
        return _MODULE["port"]

    def test_health_and_models(self):
        """``/health`` 要报出负载均衡器轮询的图状态，``/v1/models`` 要报出客户端
        据以选模型的服务名。"""
        health = get(self.port, "/health")
        self.assertEqual(health.get("status"), "ok", json.dumps(health))
        self.assertTrue(health.get("graph") is True)
        models = get(self.port, "/v1/models")
        self.assertEqual(models["object"], "list")
        self.assertEqual(models["data"][0]["id"], "qwen3.5-4b")

    def test_chat_completion(self):
        """非流式形状：``content``、``finish_reason`` 和 usage 都得填上，客户端对每一项都会分支处理。"""
        r = post(self.port, {"model": "qwen3.5-4b",
                             "messages": [{"role": "user", "content": "hi"}]})
        msg = r["choices"][0]["message"]
        self.assertEqual(msg["content"], "The capital of France is Paris.")
        self.assertEqual(r["choices"][0]["finish_reason"], "stop")
        self.assertEqual(r["usage"]["completion_tokens"], len(PLAIN))

    def test_content_parts_are_joined(self):
        """多段 ``content`` 是视觉类客户端发的东西；丢掉其中一段就等于丢掉 prompt。"""
        r = post(self.port, {"messages": [{"role": "user", "content": [
            {"type": "text", "text": "hi"}]}]})
        self.assertTrue(r["choices"][0]["message"]["content"].endswith("Paris."))

    def test_max_tokens_sets_finish_length(self):
        """客户端靠 ``finish_reason`` 得知回答被截断；报成 ``stop`` 会让截断的回复看起来完整。"""
        r = post(self.port, {"messages": [{"role": "user", "content": "hi"}],
                             "max_tokens": 3})
        self.assertEqual(r["choices"][0]["finish_reason"], "length")
        self.assertEqual(r["usage"]["completion_tokens"], 3)

    def test_sse_plain(self):
        """流必须以 role delta 开头、以 finish 块收尾，否则客户端的累积器永远等不到结束。"""
        ch = sse(self.port, {"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(content_of(ch), "The capital of France is Paris.")
        self.assertEqual(ch[0]["choices"][0]["delta"].get("role"), "assistant")
        self.assertEqual(ch[-1]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(ch[-1]["usage"]["total_tokens"], 11 + len(PLAIN))

    def test_thinking_is_suppressed_from_content(self):
        """推理内容必须走 ``reasoning_content``：漏进 ``content`` 就会把模型的草稿
        过程塞进用户看到的答案里。"""
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
        """在思考中途撞上 token 上限的回复没有答案，``content`` 必须为空，而不是塞一段被截断的思考。"""
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
        """``<tool_call>`` 标记是内部方言：必须转成 ``tool_calls``，且绝不出现在 ``content`` 里。"""
        r = post(self.port, {"messages": [{"role": "user", "content": "北京?"}],
                             "tools": TOOLS_SCHEMA})
        message = r["choices"][0]["message"]
        call = (message.get("tool_calls") or [{}])[0].get("function", {})
        self.assertEqual(r["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(call.get("name"), "get_weather")
        self.assertEqual(json.loads(call["arguments"])["city"], "北京")

    def test_sse_tool_fragments_accumulate_to_the_non_stream_call(self):
        """客户端从片段拼回调用，因此累积结果必须与非流式调用逐字相同；切法不同
        就会把参数 JSON 拼坏。"""
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
        """标记是分片到达的，过滤器必须把不完整的 ``<tool_call>`` 扣在缓冲区里，而不是直接发出去。"""
        r = post(self.port, {"messages": [{"role": "user", "content": "split it please"}],
                             "tools": TOOLS_SCHEMA})
        self.assertEqual((r["choices"][0]["message"].get("tool_calls") or [{}])[0]
                         ["function"]["name"], "get_weather")
        ch = sse(self.port, {"messages": [{"role": "user", "content": "split it please"}],
                             "tools": TOOLS_SCHEMA})
        self.assertEqual(content_of(ch), "Let me check.")

    def test_fragmented_arguments_match_the_non_stream_arguments(self):
        """这三种分片正是以前把 JSON 搞坏的情形：字符串值被拆开、整数被拆开，以及转义。"""
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
                json.loads(got[0]["function"]["arguments"])   # 解析得动，且参数没有重复

    def test_tool_choice_none_drops_tools(self):
        """``tool_choice: none`` 必须在引擎看到之前就把 tools 摘掉，否则 prompt 仍会
        把模型往调用上引。"""
        r = post(self.port, {"messages": [{"role": "user", "content": "北京?"}],
                             "tools": TOOLS_SCHEMA, "tool_choice": "none"})
        message = r["choices"][0]["message"]
        self.assertEqual(r["choices"][0]["finish_reason"], "stop")
        self.assertFalse(message.get("tool_calls"))
        self.assertEqual(message["content"], "The capital of France is Paris.")
        self.assertFalse(SEEN[-1].get("tools"))

    def test_tool_choice_required_forces_the_opener_and_returns_the_call(self):
        """``required`` 靠把开标记种进 prompt 实现，所以服务端不能再把它当 content
        回显，也不能重复一遍。"""
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
        """具名选择会种下开标记加函数名，模型就没法改挑别的工具。"""
        r = post(self.port, {"messages": [{"role": "user", "content": "北京?"}],
                             "tools": TOOLS_SCHEMA,
                             "tool_choice": {"type": "function",
                                             "function": {"name": "get_weather"}}})
        self.assertEqual(SEEN[-1].get("tool_prefix"), "<tool_call>\n<function=get_weather>\n")
        self.assertEqual((r["choices"][0]["message"].get("tool_calls") or [{}])[0]
                         ["function"]["name"], "get_weather")

    def test_legacy_functions_pair_maps_onto_tools(self):
        """已弃用的 ``functions`` / ``function_call`` 组合必须映射到 ``tools``；
        还有客户端在发，悄悄丢掉就会丢失这次调用。"""
        r = post(self.port, {
            "messages": [{"role": "user", "content": "北京?"}],
            "functions": [{"name": "get_weather", "parameters": {"type": "object"}}],
            "function_call": "required"})
        self.assertEqual(SEEN[-1].get("tool_prefix"), "<tool_call>\n")
        self.assertTrue(SEEN[-1].get("tools"))
        self.assertEqual((r["choices"][0]["message"].get("tool_calls") or [{}])[0]
                         ["function"]["name"], "get_weather")

    def test_parallel_tool_calls_flag(self):
        """``parallel_tool_calls: false`` 必须截到单个调用；客户端要禁止并行调用
        只有这一个开关。"""
        r = post(self.port, {"messages": [{"role": "user", "content": "two calls"}],
                             "tools": TOOLS_SCHEMA})
        self.assertEqual(len(r["choices"][0]["message"]["tool_calls"]), 2)
        r = post(self.port, {"messages": [{"role": "user", "content": "two calls"}],
                             "tools": TOOLS_SCHEMA, "parallel_tool_calls": False})
        self.assertEqual(len(r["choices"][0]["message"]["tool_calls"]), 1)

    def test_stream_include_usage(self):
        """开了 ``include_usage`` 之后，usage 搭在一个额外的无 choice 块上；挂在
        内容块上会打断按 ``choices[0]`` 取值的客户端。"""
        ch = sse(self.port, {"messages": [{"role": "user", "content": "hi"}],
                             "stream_options": {"include_usage": True}})
        usage_chunks = [c for c in ch if c["choices"] == []]
        self.assertEqual(len(usage_chunks), 1)
        self.assertEqual(usage_chunks[0]["usage"]["total_tokens"], 11 + len(PLAIN))
        self.assertTrue(all(c.get("usage", "absent") is None
                            for c in ch if c["choices"]))
        self.assertEqual(content_of(ch), "The capital of France is Paris.")
