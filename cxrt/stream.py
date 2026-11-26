"""流与事件：显式句柄，代替 ``torch.cuda.Stream`` / ``torch.cuda.Event``。

Module: ``coffin/cxrt``。只 import ``rt.driver`` 与标准库，**不 import
torch**。

为什么要这层薄壳而不是直接把整数句柄传下去：句柄需要一个所有者，否则没人负责销毁。
``Stream`` 与 ``Event`` 就是这个所有者，并且把「同步发生在什么时候」写成显式方法，
而不是依赖 ``with torch.cuda.stream(...)`` 这种全局状态切换。

Constraints:
    * 默认流用 ``0`` 表示，与 cudart 一致。``Stream.default()`` 返回的对象不拥有句柄，
      它的 ``close()`` 是空操作——销毁默认流会把整个上下文带走。
    * ``Event`` 的计时只在同一条流上有意义。跨流的 ``elapsed_ms`` 在驱动上仍然可算，
      但它量的是「两次打点之间的墙钟」，不是「这段计算用了多久」。要在诊断里区分这两
      件事，看调用点，不要看这里。
"""

from __future__ import annotations

from cxrt.driver import Cudart


class Stream:
    """一条设备流。

    Attributes:
        handle: cudart 的 ``cudaStream_t`` 值，``0`` 是默认流。
    """

    __slots__ = ("_rt", "_handle", "_owned")

    def __init__(self, rt: Cudart, handle: int, owned: bool) -> None:
        self._rt = rt
        self._handle = handle
        self._owned = owned

    @classmethod
    def create(cls, rt: Cudart) -> "Stream":
        """建一条新的非默认流。

        Args:
            rt: 运行时句柄。

        Returns:
            一条由本对象拥有的流。

        Raises:
            RtError: 驱动拒绝建流。
        """
        return cls(rt, rt.stream_create(), True)

    @classmethod
    def default(cls, rt: Cudart) -> "Stream":
        """返回默认流的包装，不拥有底层句柄。

        Args:
            rt: 运行时句柄。

        Returns:
            ``handle == 0`` 的包装对象。
        """
        return cls(rt, 0, False)

    @property
    def handle(self) -> int:
        """底层 ``cudaStream_t`` 值，可直接透传给内核出口。"""
        return self._handle

    def sync(self) -> None:
        """等这条流完成。"""
        self._rt.stream_sync(self._handle)

    def wait(self, event: "Event") -> None:
        """让这条流等某个事件。

        Args:
            event: 已打过点的事件。
        """
        raise NotImplementedError("cudaStreamWaitEvent is not bound yet; "
                                  "add it to driver.Cudart before using cross-stream waits")

    def close(self) -> None:
        """销毁这条流。默认流不拥有句柄，因此什么也不做。"""
        if self._owned and self._handle:
            self._rt.stream_destroy(self._handle)
            self._handle = 0
            self._owned = False

    def __enter__(self) -> "Stream":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:                    # pragma: no cover - 解释器退出路径
        try:
            self.close()
        except Exception:
            pass


class Event:
    """一个可打点、可计时的设备事件。

    Attributes:
        handle: ``cudaEvent_t`` 值。
    """

    __slots__ = ("_rt", "_handle")

    def __init__(self, rt: Cudart, handle: int) -> None:
        self._rt = rt
        self._handle = handle

    @classmethod
    def create(cls, rt: Cudart) -> "Event":
        """建一个事件。

        Args:
            rt: 运行时句柄。

        Returns:
            新事件；用完必须 :meth:`close`。
        """
        return cls(rt, rt.event_create())

    def record(self, stream: Stream | None = None) -> None:
        """在流上打点。

        Args:
            stream: 目标流；``None`` 表示默认流。
        """
        self._rt.event_record(self._handle, stream.handle if stream else 0)

    def sync(self) -> None:
        """等到这个事件发生。"""
        self._rt.event_sync(self._handle)

    def ms_since(self, earlier: "Event") -> float:
        """返回 ``earlier -> self`` 的毫秒数。

        Args:
            earlier: 先打点的事件。

        Returns:
            墙钟毫秒。
        """
        return self._rt.event_elapsed_ms(earlier._handle, self._handle)

    def close(self) -> None:
        """销毁事件。"""
        if self._handle:
            self._rt.event_destroy(self._handle)
            self._handle = 0

    def __del__(self) -> None:                    # pragma: no cover - 解释器退出路径
        try:
            self.close()
        except Exception:
            pass
