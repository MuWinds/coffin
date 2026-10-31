"""OpenAI 兼容的 HTTP/SSE 适配器。

Module: ``coffin.serving``。请求校验、socket I/O、流解析和 HTTP 响应写出分别
在独立模块里；本文件只管 Handler 和进程内的命令行装配。

Constraints:
    * 只用标准库。模型进程通过 socket 访问。
    * 一些公开的辅助函数名在这里转出，现有测试和客户端直接引用它们。
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
    """按本兼容模块上打过补丁的限额拆分图像。

    限额放在这里而不是直接从 ``request`` 取，是为了让 ``--max-images`` 之类的
    命令行开关在运行时改写它们，而 ``request`` 里的默认值只在导入时算一次。

    Args:
        messages: 请求的 ``messages`` 数组。

    Returns:
        ``(messages, images, error)``，见 ``request._split_messages``。
    """
    _request.MAX_IMAGE_BYTES = MAX_IMAGE_BYTES
    _request.MAX_IMAGES = MAX_IMAGES
    _request.MAX_IMAGE_TOTAL_BYTES = MAX_IMAGE_TOTAL_BYTES
    return _request._split_messages(messages)


def _max_tokens(body: dict) -> int:
    """用本模块共享的服务状态读取输出预算。

    Args:
        body: 请求体。

    Returns:
        以 token 计的输出预算；``-1`` 表示整个窗口。
    """
    return _request._max_tokens(body, STATE)

class Handler(BaseHTTPRequestHandler):
    """处理一次 HTTP 请求；OpenAI 的线格式都集中在这个类里。

    ``ThreadingHTTPServer`` 给每个请求一条线程，真正把生成串行化的是引擎，
    不是这个类。

    响应形状由 OpenAI 定义固定：非流式是 JSON 的 ``chat.completion`` 对象，
    SSE 是 ``chat.completion.chunk`` 对象，出错则是 ``{"error": {...}}`` 响应体
    加上客户端拿来分支判断的状态码。
    """

    protocol_version = "HTTP/1.1"
    # 厂商中立的标识：本适配器同时面向 DCU 构建和 CoreX 构建，按这个字符串
    # 做分支的客户端不该在引擎换到别的加速卡时还要重新认一遍。
    server_version = "coffin-engine/0.1"

    def log_message(self, fmt, *args):  # noqa: A003 - stdlib hook name
        print(f"[http] {self.address_string()} {fmt % args}", flush=True)

    def _json(self, code: int, obj) -> None:
        """写一个带明确 ``Content-Length`` 的 JSON 响应。

        长度必须给：连接是 HTTP/1.1 keep-alive，缺了长度客户端会一直等一个
        永不结束的响应体。``Access-Control-Allow-Origin`` 全开，因为常见客户端
        是另一个 origin 上的浏览器界面。

        Args:
            code: HTTP 状态码。
            obj: 可 JSON 序列化的响应体。
        """
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code: int, message: str, err_type: str = "invalid_request_error") -> None:
        """写出 OpenAI 的错误信封，客户端是逐字段解析它的。

        Args:
            code: HTTP 状态码。
            message: 可读的原因。它会展示给用户，所以要写清客户端该改什么。
            err_type: 请求本身能改时用 ``invalid_request_error``，错在我们这边
                时用 ``server_error``。
        """
        self._json(code, {"error": {"message": message, "type": err_type, "code": code}})

    def _sse_start(self) -> None:
        """为流式回复提交 200 和 SSE 头。

        只在引擎的第一行证明请求已被接受之后调用：头一旦发出去，拒绝就再也
        变不成 400 了。``close_connection`` 必须置上，因为 SSE 以 EOF 结束——
        开头没有长度可发。
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.close_connection = True         # SSE 以 EOF 结束：没有 Content-Length

    def _sse(self, obj) -> None:
        """写一帧 SSE 并立刻刷出。

        Args:
            obj: 要序列化成该帧 ``data:`` 载荷的 chunk 对象。两个换行才是帧的
                终止符；刷出则让一个 token 在下一个 token 生成之前就到达客户端。
        """
        self.wfile.write(f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _body(self) -> dict | None:
        """读取并解析请求体。

        Returns:
            解析后的字典；已经写出 400 时返回 ``None``——调用方据此直接返回，
            错误响应已经发完了。
        """
        try:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")
        except Exception as exc:  # noqa: BLE001
            self._error(400, f"bad request body: {exc}")
            return None

    def do_OPTIONS(self):  # noqa: N802
        """回应浏览器客户端在 POST 之前发的 CORS 预检。"""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "content-type, authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):  # noqa: N802
        """响应客户端会轮询的两个发现/存活路由。

        ``/health`` 转发 ``ping`` 报告引擎状态；``/v1/models`` 是客户端库最先
        调用的接口，所以必须直接用缓存状态回答，不去碰引擎。
        """
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path in ("/health", "/healthz"):
            try:
                rep = STATE["engine"].call({"op": "ping"})

                # ``rep`` 已经带着引擎给的 model / model_path / arch
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
        """路由唯一的 POST 端点，并且任何失败都不能把服务弄挂。

        流到一半客户端断开不该记成服务端故障：那是 SSE 客户端被关掉之后的
        正常结果。
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
        # 一个坏请求绝不能杀死整个服务，所以这里捕获所有异常并回 500。
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            try:
                self._error(500, f"{type(exc).__name__}: {exc}", "server_error")
            except Exception:
                pass

    def _chat(self, body: dict) -> None:
        """校验一条 chat-completions 请求，然后交给对应的回复路径。

        校验放在这里和 ``_split_messages`` 里，是为了让客户端所有送错的地方在
        引擎看到请求之前就变成 400。否则引擎要等到预填充之后才回同样的错误，
        晚了整整一个预填充。

        Args:
            body: 解析后的请求体。
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

               # 强制调用必须开在答案最前面，所以不能先出推理段
               "enable_thinking": think and not forced,
               "max_new": max_new, "temperature": temperature,

               # 引擎自己的墙钟上限：客户端静默下来却不关 socket 时，也只有这一刀
               # 能把它切掉
               "max_seconds": STATE.get("max_decode_seconds") or None}
        if images:
            # base64（这里已经校验并重编码过）随 socket 上的 JSON 一起走：解码、
            # 预处理和占位符 token 的展开都由引擎用 checkpoint 自带的处理器完成，
            # 所以本进程只用标准库，也不必和它在数值上保持一致。
            req["images"] = images
        cid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())
        if stream:
            self._chat_stream(req, cid, created, model, single_call, include_usage)
        else:
            self._chat_once(req, cid, created, model, single_call)

    @staticmethod
    def _finish_reason(out: dict) -> str:
        """把引擎的停止原因映射到定义里的枚举。

        Args:
            out: 引擎的回包（或 ``done`` 行）。

        Returns:
            解析出工具调用时是 ``tool_calls``；生效预算（按窗口收紧之后）用尽
            时是 ``length``；其余是 ``stop``。服务端的截止时间同样映射成
            ``length``：定义里没有「截止」这个取值，而 ``length`` 是唯一表示
            「被截断」而不是「模型自己写完」的那一个。
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
        """用引擎的计数器拼出定义里的 ``usage`` 对象。

        Args:
            out: 引擎的回包（或 ``done`` 行）。

        Returns:
            ``prompt_tokens`` / ``completion_tokens`` / ``total_tokens``，再加
            扩展字段 ``prompt_tokens_details.cached_tokens``，它报告 prompt 里
            有多少已经被基数前缀缓存算过了。
        """
        pt = int(out.get("n_prompt") or 0)
        ct = int(out.get("n_new") or 0)
        return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct,
                "prompt_tokens_details": {"cached_tokens":
                                          int(out.get("cached_tokens") or 0)}}

    def _chat_once(self, req: dict, cid: str, created: int, model: str,
                   single_call: bool = False) -> None:
        """用一个 ``chat.completion`` 对象回应非流式请求。

        Args:
            req: 引擎请求。
            cid: 本次回复的 ``chatcmpl-`` id。
            created: 作为生成时间上报的 epoch 秒。
            model: 客户端要的模型 id（或服务默认值）。
            single_call: 为 ``parallel_tool_calls: false`` 只保留一个调用。
        """
        out = STATE["engine"].call(req)
        if not out.get("ok", True):
            # 引擎会说明*为什么*失败、该用哪个状态码：图像有问题回 400，没有
            # 视觉塔的服务回 503。把它们一律压成 500，等于让客户端去重试一个
            # 永远不可能成功的请求。
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
            # 只保留自然语言的开场白：调用是结构化字段走的，和流式内容一样
            # （两者都由模板渲染）。
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
        """把 ``stream: true`` 的请求按 ``chat.completion.chunk`` 的 SSE 帧回出去。

        帧的顺序按定义来：先一个只带 role 的开头 delta，然后推理/内容/工具调用
        每个片段一个 delta，最后一个空的 delta 带 ``finish_reason``，需要时再
        补用量块，最后是 ``data: [DONE]``。每一帧都要重复
        ``id``/``object``/``created``/``model``，因为客户端把每个 chunk 当成
        独立对象处理。

        Args:
            req: 引擎请求；它的 ``tool_prefix`` 用来给流式解析器播种，让强制
                调用和自然产生的调用切出来的片段完全一样。
            cid: 所有 chunk 共用的 ``chatcmpl-`` id。
            created: 每一帧都上报的 epoch 秒。
            model: 每一帧都上报的模型 id。
            single_call: 为 ``parallel_tool_calls: false`` 把解析器的调用数上限
                压到一个。
            include_usage: 是否发定义里那个额外的 ``choices: []`` 用量块。
        """
        base = {"id": cid, "object": "chat.completion.chunk", "created": created,
                "model": model}

        # 定义规定：带 ``stream_options.include_usage`` 时其余每帧都写
        # ``usage: null``，并在 ``[DONE]`` 之前单独补一帧 ``choices: []`` 携带
        # 真实数字。
        marker = {"usage": None} if include_usage else {}
        t_start = time.time()
        sent = 0

        def emit(choices: list, **extra) -> None:
            """写一个 chunk，并记下已经发出去多少个 choice。

            Args:
                choices: 本 chunk 的 ``choices`` 数组。
                **extra: 额外的顶层字段（失败路径上的 ``error`` 对象）。
            """
            nonlocal sent
            sent += len(choices)
            self._sse({**base, "choices": choices, **marker, **extra})

        flt: AnswerFilter | None = None
        done: dict | None = None
        # 本次回复只用一个流式工具调用解析器，用强制开头播种，这样
        # ``tool_choice: required`` 产生的调用流出来和自然产生的完全一样
        # （开头在 prompt 里，不在生成文本里）。
        tstream = ToolStream(seed=req.get("tool_prefix") or "",
                             max_calls=1 if single_call else None)
        # 引擎在自己的 ``start`` 行*之前*就完成了校验（图像解码、处理器展开、
        # 占位符自检），所以它的第一行仍可能是拒绝。这一行到达之前什么都不写：
        # 被拒的带图请求因此能拿到真正的 400/503 和 OpenAI 客户端期待的 JSON
        # 错误体，而不是一个 200 的 SSE 流里塞着错误事件——那种东西没有客户端
        # 会重试，也没法有效上报。
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
                        # ``tool_choice: required``：开头已经在 prompt 里了，
                        # 所以生成的文本是调用的*正文*——它绝不能当成 content
                        # 发出去，调用会在下面以结构化字段交付。
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
                    # 推理段边产生边发出，带思考面板的客户端因此能实时显示
                    # （而且永远不会混进 content）
                    emit([{"index": 0, "delta": {"reasoning_content": delta_r},
                           "finish_reason": None}])
                if delta_c:
                    emit([{"index": 0, "delta": {"content": delta_c},
                           "finish_reason": None}])
        except Exception as exc:  # noqa: BLE001 - report into the stream, then close it
            traceback.print_exc()

            # ``client_hung_up`` 决定是谁切断了流：为 False 是引擎提前停了，
            # 为 True 则是答案在出去的路上丢了（客户端或中间转发），不是模型
            # 的问题。
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
            if tail_r:      # 流在推理段内部就结束了
                emit([{"index": 0, "delta": {"reasoning_content": tail_r},
                       "finish_reason": None}])
            if tail_c:
                emit([{"index": 0, "delta": {"content": tail_c}, "finish_reason": None}])
        for td in tstream.finish():          # 调用被截断：把已有的部分收尾
            emit([{"index": 0, "delta": td, "finish_reason": None}])
        calls = done.get("tool_calls") or []
        if single_call and len(calls) > 1:       # parallel_tool_calls: false
            calls = calls[:1]
        if calls and not tstream.emitted_any:
            # 流式解析器没看见的调用走这里兜底（那说明解析器有 bug）：整个调用
            # 一次性发出去。如果片段已经发出去了，这里再重复完整调用会让客户端
            # 把 arguments 拼接两遍。
            emit([{"index": 0, "delta": {
                "tool_calls": [{"index": i, **c} for i, c in enumerate(calls)]},
                "finish_reason": None}])
        last = {"index": 0, "delta": {},
                "finish_reason": self._finish_reason({**done, "tool_calls": calls})}
        if include_usage:
            emit([last])
            self._sse({**base, "choices": [], "usage": self._usage(done)})
        else:
            # OpenAI 不主动给 usage，除非客户端要；把它挂在最后一个 chunk 上是
            # 现有客户端（bridge、各种探针）已经在读的宽松行为，所以没人要那个
            # 额外用量块时它仍然保留。
            self._sse({**base, "choices": [last], "usage": self._usage(done)})
        print(f"[http] stream done id={cid} n_new={done.get('n_new')} "
              f"finish={last['finish_reason']} stopped={done.get('stopped')} "
              f"tokens_sent={sent} wall={time.time() - t_start:.1f}s "
              f"prefill_ms={done.get('prefill_ms')} client_hung_up=False", flush=True)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
