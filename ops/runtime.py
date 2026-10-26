"""Device-runtime plumbing for the captured execution paths: streams, graphs and free-memory calls.

The model has to do four things that are not ops: run warmup work on a side
stream, capture a fixed buffer-address body into a device graph, replay it, and
ask the allocator to hand memory back. All four name one vendor's runtime API when
written directly in ``models/``, so they live here, behind one flag --
:attr:`Caps.graph <coffin.backends.protocol.Caps.graph>`.

The eager substitute is not a degraded mode. A graph can only be captured when
*every* op inside the body is one the backend can capture, and on the Iluvatar
BI-V100 that is false by construction: its library GEMM cannot be captured at all
(measured ``CUBLAS_STATUS_ALLOC_FAILED``) while the hand-written kernels capture
and replay fine. So a backend whose table is mixed must run the body eagerly, and
that is the correct path for it, not a fallback that merely avoids crashing --
:func:`capture` returns an object whose ``replay()`` re-executes the body, and the
numbers a caller gets are the same ones the graph would have produced.

Module: ``coffin/ops``. Depends on ``backends.caps()`` and ``torch`` only; called
by ``models/qwen35/*`` and by ``serving``'s bring-up path. Nothing here may import
``models`` or any layer above it.

Constraints:
    * The device namespace comes from ``caps().device``, never from a literal. The
      DCU and the CoreX both answer to the same namespace, so deriving it is what
      lets one call serve both -- and a ``gcu`` or ``musa`` backend when one is
      added. A literal here would pin this file to one vendor's stack.
    * ``caps.graph`` gates capture only, not the runtime API in general: CoreX has
      a working event timer and a working stream API while being unable to capture
      a library GEMM, so ``Event``/``Stream``/``empty_cache`` are always the real
      calls and only :func:`capture` branches.
    * No allocation, no capture and no device call may happen at import: a CPU-only
      host imports this module to run the reference backend.
    * A backend with no ``caps.graph`` support still gets a capture handle, so a
      caller never has to branch; the handle's ``captured`` property is what
      reports which path it is on, for logging and for tests.
"""

from __future__ import annotations

import torch

from backends import caps as _caps

#: The selected backend's capability record, read once at import. The backend
#: selection is an import-time decision with no hot swap, so one read is enough.
_CAPS = _caps()

#: The runtime module for this backend's device namespace -- ``torch.<device>``,
#: which resolves to the CUDA-compatible runtime on both the DCU and the CoreX.
#: Resolved once because it is read on the decode path.
_RT = getattr(torch, _CAPS.device)


def enabled() -> bool:
    """Report whether this backend can capture a decode graph.

    Returns:
        ``Caps.graph``. ``False`` means the execution paths run eagerly, which is
        correct and slower; see the module docstring for why that is a real path
        rather than a fallback.
    """
    return bool(_CAPS.graph)


def stream():
    """Create a side stream for capture warmup.

    Returns:
        A fresh stream on this backend's device, or ``None`` when the device has
        no stream concept (the CPU reference). Callers treat ``None`` as "run it
        on the current thread", which is what an eager CPU run does anyway.
    """
    make = getattr(_RT, "Stream", None)
    return None if make is None else make()


def current_stream():
    """Return the stream the caller is presently running on.

    Returns:
        The current stream, used to order the warmup stream against the caller's,
        or ``None`` on a device with no streams.
    """
    get = getattr(_RT, "current_stream", None)
    return None if get is None else get()


def empty_cache() -> None:
    """Return cached allocator blocks to the driver.

    Called between model loads and after releasing the dense KV caches: on a
    16 GB card the difference decides whether a paged pool fits, and the allocator
    does not release segments on its own. A device whose runtime has no caching
    allocator has nothing to return, so this is a no-op there rather than a crash
    in the model constructor.
    """
    release = getattr(_RT, "empty_cache", None)
    if release is not None:
        release()


def graph_pool_handle():
    """Return a shared memory-pool handle for a group of mutually exclusive graphs.

    Returns:
        A pool handle, or ``None`` when this device's runtime has no such concept
        (the CPU reference, for instance). Graphs that never replay concurrently
        can share one pool, so a bucket set costs the largest bucket's memory
        rather than the sum; ``None`` simply means each graph gets its own.
    """
    handle = getattr(_RT, "graph_pool_handle", None)
    return None if handle is None else handle()


class Capture:
    """A captured body, replayed by ``replay()``; eager when the backend cannot capture.

    One type for both paths is the point: call sites capture once and replay many
    times, and the only thing that changes with ``caps.graph`` is whether the body
    was recorded into a graph or kept as a closure.

    Attributes:
        captured: ``True`` when the body really is a device graph. ``False`` means
            ``replay()`` re-executes the body.
    """

    __slots__ = ("_body", "_graph", "_pool", "captured")

    def __init__(self, body, graph=None, pool=None) -> None:
        """Wrap a captured graph, or a body to re-run eagerly.

        Args:
            body: The zero-argument callable the graph was captured from. Retained
                even on the eager path, and on the captured path it is what the
                warmup calls, so both paths share one body definition.
            graph: The device graph, or ``None`` for the eager path.
            pool: Optional shared memory-pool handle the graph was captured with.
        """
        self._body = body
        self._graph = graph
        self._pool = pool
        self.captured = graph is not None

    def replay(self) -> None:
        """Run the body: replay the graph, or call it directly when eager.

        Returns:
            Nothing. On the captured path the body reads and writes the same fixed
            buffers either way, which is why callers need no branch here.
        """
        if self._graph is not None:
            self._graph.replay()
        else:
            self._body()


def capture(body, warmup: int = 3, pool=None, reset=None) -> Capture:
    """Record ``body`` into a device graph, or wrap it for eager execution.

    The warmup runs first for a reason: a body's first call allocates scratch and
    creates library handles, and recording that into the graph would bake one-time
    setup into every replay.

    Args:
        body: Zero-argument callable that reads and writes fixed buffers. It must
            not allocate anything that outlives the call on the captured path,
            because the graph replays the same addresses.
        warmup: Eager iterations before the capture. Zero is legal and is what a
            caller that already warmed the body uses.
        pool: Optional pool handle from :func:`graph_pool_handle`, so several
            mutually exclusive graphs share one memory pool.
        reset: Optional callable invoked after the warmup and before the capture.
            A body that advances position counters or mutates recurrent state
            leaves them dirty from the warmup, and the captured graph must be
            recorded against the state its first replay will see. Called on the
            eager path too, so both paths leave the model in the same state.

    Returns:
        A :class:`Capture`; ``captured`` reports which path it took. With
        ``caps.graph`` false the body is wrapped rather than recorded, which is
        correct on a backend whose table mixes capturable and non-capturable ops.

    Raises:
        RuntimeError: The runtime's graph machinery refused the capture.
            Propagated rather than silently degrading: a body that fails to
            capture here fails again on the next replay, and a silent fallback
            would hide a kernel that launches outside the capturing stream.
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
    """A device event for timing a segment of the decode path.

    Only used by the diagnostic counters and the ``--profile`` path, never in the
    production decode loop: an event costs a device-side record per segment, and at
    a 0.3 ms step that is not free.

    A device whose runtime has no timed event yields an inert one that reports
    ``0.0``: a profiler run on such a host should read as "no timing available"
    rather than abort a decode that is otherwise working.

    Attributes:
        _event: The underlying runtime event, or ``None`` when the runtime has no
            timing-enabled event.
    """

    __slots__ = ("_event",)

    def __init__(self) -> None:
        """Create a timing-enabled event on this backend's runtime, if it has one."""
        try:
            self._event = _RT.Event(enable_timing=True)
        except (AttributeError, TypeError):
            self._event = None

    def record(self) -> None:
        """Mark this event at the current point in the caller's stream."""
        if self._event is not None:
            self._event.record()

    def elapsed_ms(self, other: "Event") -> float:
        """Milliseconds from this event to ``other``.

        Args:
            other: An event recorded later on the same device.

        Returns:
            The elapsed time in milliseconds, or ``0.0`` when this device cannot
            time. Synchronising is the caller's job, so a reading taken before the
            work finishes is meaningless -- see :func:`synchronize`.
        """
        if self._event is None or other._event is None:
            return 0.0
        return self._event.elapsed_time(other._event)


def synchronize() -> None:
    """Block the host until every queued kernel on this device has finished.

    A device with no asynchronous queue (the CPU reference) has nothing to wait
    for, so this is a no-op there.
    """
    sync = getattr(_RT, "synchronize", None)
    if sync is not None:
        sync()
