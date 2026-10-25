"""Line-oriented client for the resident engine socket.

Module: ``coffin.serving``. It is standard-library only so the HTTP process can
run without importing torch or a model implementation.
"""

from __future__ import annotations

import json
import socket

class EngineClient:
    """One client of the engine's line-oriented unix socket.

    Each request opens its own connection: the engine accepts connections
    concurrently and batches whatever requests are in flight, so a lock here
    would serialize exactly the work the batch scheduler is supposed to overlap.

    The timeout is per socket, not per token stream: a long generation holds the
    connection open for its whole duration, which is why the default is 900 s
    rather than something that would cut a legitimate long answer.
    """

    def __init__(self, sock_path: str, timeout: float = 900.0) -> None:
        self.sock_path = sock_path
        self.timeout = timeout

    def _open(self):
        """Open one connection and its buffered reader.

        Returns:
            ``(socket, file)``; the caller closes both through ``_close``. The
            reader is what turns ``readline`` into a proper line read -- reading
            the socket directly could split a JSON line across two ``recv`` calls.
        """
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        s.connect(self.sock_path)
        return s, s.makefile("rb")

    @staticmethod
    def _close(s, f) -> None:
        """Close the socket and its file object, ignoring whatever already failed.

        Both have to be attempted: closing the file object alone can leave the
        descriptor open when the read side already errored, and the server leaks a
        thread per leaked connection.

        Args:
            s: The socket returned by ``_open``.
            f: The buffered reader paired with it.
        """
        for c in (f, s):
            try:
                c.close()
            except Exception:
                pass

    def _send(self, s, req: dict) -> None:
        s.sendall((json.dumps(req, ensure_ascii=False) + "\n").encode())

    def call(self, req: dict) -> dict:
        """One request, one reply.

        Args:
            req: The engine request dict; ``op`` selects the handler.

        Returns:
            The decoded reply dictionary. It may carry ``ok: False``; the caller
            decides the HTTP status, because only it knows the context.

        Raises:
            RuntimeError: The engine closed the connection without a reply line.
                The engine is down or restarting; the caller should map this to a
                503 rather than a request error.
            json.JSONDecodeError: The reply line was not JSON, which means the
                socket speakers agree on nothing -- a version mismatch, not a
                client problem.
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
        """Yield reply lines until the terminal one (``type: done`` / an error).

        Args:
            req: The engine request dict; this is the ``chat_stream`` op.

        Yields:
            Each decoded line in order: ``start``, one ``token`` per piece, then
            ``done``. An error line is yielded too, then iteration stops.

        Raises:
            RuntimeError: The engine closed the connection before the terminal
                line. The stream cannot be resumed; the caller closes the SSE
                response and reports the failure into the log.
            json.JSONDecodeError: A line was not JSON.
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
