"""常驻引擎 socket 的行协议客户端。

Module: ``coffin.serving``。只用标准库，HTTP 进程因此不需要 import torch 或
任何模型实现就能起来。

Constraints:
    * 单行一条 JSON，读端必须用 ``makefile`` 包一层——``recv`` 会把一行切成
      两半，按字节直接解析迟早出错。
"""

from __future__ import annotations

import json
import socket

class EngineClient:
    """引擎行协议 unix socket 的一个客户端。

    每个请求各开一条连接：引擎并发接受连接，并把在途请求凑成批，这里加锁
    恰好会把批调度本该重叠起来的工作串行化。

    超时是整条 socket 的超时，不是逐 token 的：一次长生成会在整个过程中占着
    连接，所以默认给到 900 s——调小会把正常的长回答截断。
    """

    def __init__(self, sock_path: str, timeout: float = 900.0) -> None:
        self.sock_path = sock_path
        self.timeout = timeout

    def _open(self):
        """打开一条连接和执行缓冲的读取器。

        Returns:
            ``(socket, file)``；调用方统一通过 ``_close`` 关掉两者。用读取器
            包一层才有真正按行读的语义——直接读 socket 会把一行 JSON 拆进两次
            ``recv``。
        """
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        s.connect(self.sock_path)
        return s, s.makefile("rb")

    @staticmethod
    def _close(s, f) -> None:
        """关掉 socket 和它的文件对象，两边都失败也照样继续。

        两个都必须尝试：读端已经出错时单独关文件对象会把描述符留着，而服务端
        每条泄漏的连接都对应一个泄漏的线程。

        Args:
            s: ``_open`` 返回的 socket。
            f: 与它配对的缓冲读取器。
        """
        for c in (f, s):
            try:
                c.close()
            except Exception:
                pass

    def _send(self, s, req: dict) -> None:
        s.sendall((json.dumps(req, ensure_ascii=False) + "\n").encode())

    def call(self, req: dict) -> dict:
        """一发一收。

        Args:
            req: 引擎请求字典；``op`` 决定走哪个处理分支。

        Returns:
            解码后的回包字典。回包可能带 ``ok: False``；HTTP 状态码由调用方
            决定，只有它知道上下文。

        Raises:
            RuntimeError: 引擎没回任何一行就关了连接。这是引擎没起来或正在
                重启，调用方应映射成 503，而不是当成请求错误。
            json.JSONDecodeError: 回包不是 JSON，说明两端对协议的理解完全
                对不上——这是版本不匹配，不是客户端问题。
        """
        s, f = self._open()
        try:
            self._send(s, req)
            line = f.readline()
            if not line:
                raise RuntimeError("engine closed the connection")
            return json.loads(line.decode())
        finally:
            self._close(s, f)

    def stream(self, req: dict):
        """逐行产出回包，直到终止行（``type: done`` 或一条错误）。

        Args:
            req: 引擎请求字典；这里是 ``chat_stream`` 操作。

        Yields:
            按顺序解码出的每一行：``start``、若干条 ``token``，最后是
            ``done``。错误行也会产出，随后迭代结束。

        Raises:
            RuntimeError: 在终止行之前引擎就关了连接。这条流没法续上，调用方
                只能关掉 SSE 响应并把失败写进日志。
            json.JSONDecodeError: 某一行不是 JSON。
        """
        s, f = self._open()
        try:
            self._send(s, req)
            while True:
                line = f.readline()
                if not line:
                    raise RuntimeError("engine closed the connection")
                obj = json.loads(line.decode())
                yield obj
                if obj.get("type") == "done" or obj.get("ok") is False:
                    return
        finally:
            self._close(s, f)
