"""给捕获执行路径用的设备运行时管线：流、图以及释放显存的调用。

模型必须做四件不属于算子的事：在旁路流上跑预热、把一段缓冲区地址固定的主体
捕获进设备图、重放它，以及让分配器把内存还回来。这四件事直接写进 ``models/``
时都会点名某一家厂商的运行时 API，所以它们放在这里，藏在一个开关后面——
:attr:`Caps.graph <coffin.backends.protocol.Caps.graph>`。

那份即时执行的替代品不是降级模式。只有当主体里*每一个*算子都是后端能捕获的，
图才捕得下来；而在 Iluvatar BI-V100 上这从构造上就不成立：它的库 GEMM 根本
捕不了（实测报 ``CUBLAS_STATUS_ALLOC_FAILED``），手写内核却捕获、重放都很正常。
所以算子表混合的后端必须即时执行这段主体，那对它而言是正确的路径，而不是仅仅
避免崩溃的回退——:func:`capture` 返回的对象其 ``replay()`` 会重新执行主体，
调用方拿到的数字和图本会产生的一样。

Module: ``coffin/ops``。只依赖 ``backends.caps()`` 和 ``torch``；被
``models/qwen35/*`` 和 ``serving`` 的启动路径调用。这里不许 import
``models`` 或它上面任何一层。

Constraints:
    * 设备命名空间来自 ``caps().device``，绝不来自字面量。DCU 和 CoreX 都认同一个
      命名空间，推导出来才能让一次调用同时服务两者——以及日后加入的 ``gcu`` 或
      ``musa`` 后端。写死字面量会把本文件钉死在某一家厂商的软件栈上。
    * ``caps.graph`` 只控制捕获，不控制运行时 API 整体：CoreX 有可用的事件计时器
      和可用的流 API，却捕不了库 GEMM，所以 ``Event``/``Stream``/``empty_cache``
      永远是真调用，只有 :func:`capture` 分支。
    * import 时不许分配、不许捕获、不许调用设备：纯 CPU 主机会 import 本模块来跑
      参考后端。
    * 不支持 ``caps.graph`` 的后端照样拿到捕获句柄，调用方永远不必分支；句柄的
      ``captured`` 属性用于报告当前走的是哪条路径，供日志和测试使用。
"""

from __future__ import annotations

import torch

from backends import caps as _caps

#: 当前选中后端的能力记录，在 import 时读一次。后端选择是 import 期的决定，
#: 没有热切换，所以读一次就够。
_CAPS = _caps()

#: 本后端设备命名空间对应的运行时模块——``torch.<device>``，在 DCU 和 CoreX 上
#: 都解析到 CUDA 兼容运行时。只解析一次，因为解码路径上要读它。
_RT = getattr(torch, _CAPS.device)


def enabled() -> bool:
    """报告本后端能否捕获解码图。

    Returns:
        ``Caps.graph``。``False`` 表示执行路径走即时执行，这是正确但更慢的路，
        模块 docstring 解释了它为什么是真实路径而不是回退。
    """
    return bool(_CAPS.graph)


def stream():
    """为捕获前的预热建一条旁路流。

    Returns:
        本后端设备上的一条新流；设备没有流的概念时（CPU 参考实现）返回
        ``None``。调用方把 ``None`` 当作「在当前线程上跑」，那本来也是 CPU 即时
        执行的默认行为。
    """
    make = getattr(_RT, "Stream", None)
    return None if make is None else make()


def current_stream():
    """返回调用方当前所在的流。

    Returns:
        当前流，用来给预热流和调用方的流排出先后顺序；设备没有流时返回
        ``None``。
    """
    get = getattr(_RT, "current_stream", None)
    return None if get is None else get()


def empty_cache() -> None:
    """把分配器缓存着的块还给驱动。

    在两次模型加载之间、以及释放稠密 KV 缓存之后调用：在 16 GB 的卡上，这点
    差别决定分页池放不放得下，而分配器不会自己把段释放掉。运行时没有缓存分配器
    的设备没有东西可还，所以那里是空操作，而不是在模型构造函数里崩掉。
    """
    release = getattr(_RT, "empty_cache", None)
    if release is not None:
        release()


def graph_pool_handle():
    """为一组互斥的图返回共享的内存池句柄。

    Returns:
        池句柄；本设备的运行时没有这个概念时（例如 CPU 参考实现）返回
        ``None``。永远不会并发重放的图可以共用一个池，于是一组 bucket 的内存
        成本是最宽那个 bucket 而不是总和；``None`` 只是表示每张图各用各的。
    """
    handle = getattr(_RT, "graph_pool_handle", None)
    return None if handle is None else handle()


class Capture:
    """被捕获的主体，由 ``replay()`` 重放；后端捕不了时走即时执行。

    两条路径共用一种类型正是要点：调用点捕获一次、重放很多次，而
    ``caps.graph`` 改变的只是主体被录进了图，还是留成一个闭包。

    Attributes:
        captured: 主体确实是设备图时为 ``True``。``False`` 表示 ``replay()``
            会重新执行主体。
    """

    __slots__ = ("_body", "_graph", "_pool", "captured")

    def __init__(self, body, graph=None, pool=None) -> None:
        """包住一张已捕获的图，或者一个要即时重跑的主体。

        Args:
            body: 图就是从这个零参数可调用对象捕获来的。即时路径上也保留它，
                捕获路径上预热调用的也是它，两条路径因此共用同一份主体定义。
            graph: 设备图；即时路径为 ``None``。
            pool: 可选的共享内存池句柄，图就是带着它捕获的。
        """
        self._body = body
        self._graph = graph
        self._pool = pool
        self.captured = graph is not None

    def replay(self) -> None:
        """运行主体：重放图，即时执行时则直接调用。

        Returns:
            无。捕获路径上主体读写的是同一批固定缓冲区，所以调用方这里不需要
            分支。
        """
        if self._graph is not None:
            self._graph.replay()
        else:
            self._body()


def capture(body, warmup: int = 3, pool=None, reset=None) -> Capture:
    """把 ``body`` 录进设备图，或者包起来供即时执行。

    预热先跑是有理由的：主体第一次调用会分配临时缓冲、创建库句柄，把它录进图
    等于把一次性初始化烘进每一次重放。

    Args:
        body: 读写固定缓冲区的零参数可调用对象。在捕获路径上，它不许分配任何
            活过这次调用的东西，因为图重放的是同一批地址。
        warmup: 捕获前的即时执行轮数。允许为 0，已经预热过主体的调用方就这么用。
        pool: 可选的池句柄，来自 :func:`graph_pool_handle`，让几张互斥的图共用
            一个内存池。
        reset: 可选的可调用对象，在预热之后、捕获之前调用。会推进位置计数器或
            改写循环状态的主体，预热后会留下脏状态，而捕获的图必须按它首次重放
            时看到的状态来录。即时路径上也会调用，两条路径因此把模型留在同样的
            状态。

    Returns:
        :class:`Capture`；``captured`` 报告走了哪条路径。``caps.graph`` 为
        false 时主体被包起来而不是录进图，在算子表混合了可捕获与不可捕获算子的
        后端上这是正确的。

    Raises:
        RuntimeError: 运行时的图机制拒绝捕获。直接上抛而不是悄悄降级：在这里
            捕获失败的主体，下一次重放还会失败；静默回退只会藏住一个在捕获流之外
            启动的内核。
    """
    if not enabled():
        for _ in range(warmup):
            body()
        if reset is not None:
            reset()
        return Capture(body)
    side = stream()
    if side is not None:
        side.wait_stream(current_stream())
        with _RT.stream(side):
            for _ in range(warmup):
                body()
        current_stream().wait_stream(side)
    else:
        for _ in range(warmup):
            body()
    if reset is not None:
        reset()
    graph = _RT.CUDAGraph()
    ctx = _RT.graph(graph, pool=pool) if pool is not None else _RT.graph(graph)
    with ctx:
        body()
    return Capture(body, graph, pool)


class Event:
    """给解码路径的某一段计时的设备事件。

    只有诊断计数器和 ``--profile`` 路径用它，生产解码循环里从不用：事件每段都
    要在设备侧记一笔，在 0.3 ms 一步的情况下这笔开销不便宜。

    运行时没有计时事件的设备会得到一个惰性事件，永远报 ``0.0``：在那种主机上
    跑 profiler 应当读作「没有计时能力」，而不是让本来能跑的解码中断。

    Attributes:
        _event: 底层的运行时事件；运行时没有可计时的事件时为 ``None``。
    """

    __slots__ = ("_event",)

    def __init__(self) -> None:
        """在本后端的运行时上创建一个可计时事件，前提是它支持。"""
        try:
            self._event = _RT.Event(enable_timing=True)
        except (AttributeError, TypeError):
            self._event = None

    def record(self) -> None:
        """在调用方流的当前位置标记这个事件。"""
        if self._event is not None:
            self._event.record()

    def elapsed_ms(self, other: "Event") -> float:
        """从本事件到 ``other`` 之间的毫秒数。

        Args:
            other: 同一设备上更晚记录的事件。

        Returns:
            流逝时间（毫秒）；本设备不能计时时为 ``0.0``。同步是调用方的事，
            所以在工作完成前读到的数字没有意义——见 :func:`synchronize`。
        """
        if self._event is None or other._event is None:
            return 0.0
        return self._event.elapsed_time(other._event)


def synchronize() -> None:
    """阻塞主机侧，直到本设备上排队的每个内核都完成。

    没有异步队列的设备（CPU 参考实现）没有可等的东西，所以那里是空操作。
    """
    sync = getattr(_RT, "synchronize", None)
    if sync is not None:
        sync()
