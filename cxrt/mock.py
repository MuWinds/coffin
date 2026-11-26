"""假运行时：在没有卡、没有工具链、没有 torch 的主机上把 ``rt`` 的逻辑跑起来。

Module: ``coffin/cxrt``。只 import ``rt`` 内的模块与 numpy，不 import torch。

它提供两样东西：

* :class:`MockCudart` —— 与 :class:`~coffin.cxrt.driver.Cudart` 同一套
  Python 方法，但内存是宿主上的 numpy 数组，地址是伪造的整数。分配、视图、H2D/D2H、
  填充、流与事件因此都能单测。
* :class:`NumpyNormLib` —— 三个 C ABI 出口的 numpy 复刻，签名与 ``rt/shim.py`` 的
  ``NormShim`` 一致。所以 ``RMSNormOps`` 的形状校验与调用顺序可以被完整验证，
  不必等一张卡。

Constraints:
    * 地址空间必须支持**内部指针**：``Array`` 的视图（行、列切片）指向缓冲区中间，
      真实驱动当然接受这种指针，所以假运行时要按「哪个缓冲区包含这个地址」解析，
      不能只认分配起点。
    * 这里算的是 fp32 复刻，不是逐位复刻。它的用途是验形状与数据流，不是验内核数值；
      逐位对拍只能在卡上做（``rt/selftest.py``）。
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from cxrt.dtypes import DType, bf16_bits, bf16_to_f32
from cxrt.errors import RtError, ShapeError


class MockCudart:
    """用宿主内存模拟的设备运行时。"""

    def __init__(self) -> None:
        self._device: dict[int, np.ndarray] = {}
        self._host: dict[int, np.ndarray] = {}
        self._next = 0x10000
        self._streams = {0: 0}
        self._stream_next = 1
        self._events: dict[int, float] = {}
        self._event_next = 1
        self.copies = 0
        self.peak = 0
        self.allocated = 0

    # 地址空间

    def _new(self, nbytes: int, table: dict[int, np.ndarray]) -> int:
        ptr = self._next
        self._next += (nbytes + 0xFF) & ~0xFF
        table[ptr] = np.zeros(nbytes, dtype=np.uint8)
        return ptr

    def _resolve(self, table: dict[int, np.ndarray], ptr: int, nbytes: int,
                 what: str) -> np.ndarray:
        for base, block in table.items():
            if base <= ptr and ptr + nbytes <= base + block.size:
                offset = ptr - base
                return block[offset:offset + nbytes]
        raise RtError(f"{what}: address {ptr:#x}+{nbytes} is not in the mock space")

    # 内存

    def malloc(self, nbytes: int) -> int:
        if nbytes <= 0:
            raise RtError(f"cudaMalloc({nbytes}): size must be positive")
        ptr = self._new(nbytes, self._device)
        self.allocated += nbytes
        self.peak = max(self.peak, self.allocated)
        return ptr

    def free(self, ptr: int) -> None:
        if not ptr:
            return
        block = self._device.pop(ptr, None)
        if block is None:
            raise RtError(f"cudaFree({ptr:#x}): not an allocation start")
        self.allocated -= block.size

    def malloc_host(self, nbytes: int) -> int:
        return self._new(nbytes, self._host)

    def free_host(self, ptr: int) -> None:
        if not ptr:
            return
        if self._host.pop(ptr, None) is None:
            raise RtError(f"cudaFreeHost({ptr:#x}): not an allocation start")

    def host_view(self, ptr: int, dtype: DType, count: int) -> np.ndarray:
        """把一块宿主内存按 ``dtype`` 观察成数组，支持内部指针。"""
        nbytes = count * dtype.itemsize
        return self._resolve(self._host, ptr, nbytes, "host_view").view(dtype.numpy)

    # 拷贝

    def copy_h2d(self, dst: int, src: int, nbytes: int, stream: int | None = None) -> None:
        self._resolve(self._device, dst, nbytes, "copy_h2d")[:] = \
            self._resolve(self._host, src, nbytes, "copy_h2d")
        self.copies += 1

    def copy_d2h(self, dst: int, src: int, nbytes: int, stream: int | None = None) -> None:
        self._resolve(self._host, dst, nbytes, "copy_d2h")[:] = \
            self._resolve(self._device, src, nbytes, "copy_d2h")
        self.copies += 1

    def copy_d2d(self, dst: int, src: int, nbytes: int, stream: int | None = None) -> None:
        self._resolve(self._device, dst, nbytes, "copy_d2d")[:] = \
            self._resolve(self._device, src, nbytes, "copy_d2d").copy()
        self.copies += 1

    def memset(self, ptr: int, value: int, nbytes: int, stream: int | None = None) -> None:
        self._resolve(self._device, ptr, nbytes, "memset")[:] = value & 0xFF

    # 流与事件

    def stream_create(self) -> int:
        handle = self._stream_next
        self._stream_next += 1
        self._streams[handle] = handle
        return handle

    def stream_destroy(self, handle: int) -> None:
        self._streams.pop(handle, None)

    def stream_sync(self, handle: int) -> None:
        return None

    def device_sync(self) -> None:
        return None

    def event_create(self) -> int:
        handle = self._event_next
        self._event_next += 1
        self._events[handle] = time.perf_counter()
        return handle

    def event_destroy(self, handle: int) -> None:
        self._events.pop(handle, None)

    def event_record(self, handle: int, stream: int | None = None) -> None:
        self._events[handle] = time.perf_counter()

    def event_sync(self, handle: int) -> None:
        return None

    def event_elapsed_ms(self, start: int, end: int) -> float:
        return (self._events[end] - self._events[start]) * 1e3

    # 设备

    def device_count(self) -> int:
        return 1

    def set_device(self, index: int) -> None:
        if index != 0:
            raise RtError(f"cudaSetDevice({index}): mock has one device")

    def current_device(self) -> int:
        return 0

    def mem_info(self) -> tuple[int, int]:
        return (32 << 30, 32 << 30)

    def device_name(self) -> str:
        return "Iluvatar BI-V100 (mock)"

    def error_string(self, code: int) -> str:
        return f"mock cudaError {code}"


class NumpyConvLib:
    """卷积三个出口的 numpy 复刻，签名与 ``ConvShim`` 一致。

    复刻按「一条递推」实现：``acc = s0*w0 + s1*w1 + s2*w2 + x3*w3``，``y = silu(acc)``，
    然后环平移。预填充分段在真内核里是分段重放，结果与整条走下来逐位相同，所以这里
    一趟走完就够。
    """

    def __init__(self, rt: MockCudart) -> None:
        self._rt = rt
        self.path = "mock"
        self.calls: list[str] = []

    def _read_f32(self, ptr: int, rows: int, cols: int, pitch: int) -> np.ndarray:
        """读 ``[rows, cols]`` fp32，行间距 ``pitch``。"""
        out = np.empty((rows, cols), dtype=np.float32)
        for r in range(rows):
            raw = self._rt._resolve(  # noqa: SLF001 - 同一个包内的假运行时
                self._rt._device, ptr + r * pitch * 4, cols * 4, "read f32 row")
            out[r] = raw.view(np.float32)
        return out

    def _write_f32(self, ptr: int, rows: int, cols: int, pitch: int,
                   values: np.ndarray) -> None:
        """写 ``[rows, cols]`` fp32，行间距 ``pitch``。"""
        for r in range(rows):
            dst = self._rt._resolve(  # noqa: SLF001
                self._rt._device, ptr + r * pitch * 4, cols * 4, "write f32 row")
            dst.view(np.float32)[:] = values[r].astype(np.float32)

    def _read_bf16(self, ptr: int, rows: int, cols: int, pitch: int) -> np.ndarray:
        """读 ``[rows, cols]`` bf16，行间距 ``pitch``。"""
        out = np.empty((rows, cols), dtype=np.float32)
        for r in range(rows):
            raw = self._rt._resolve(  # noqa: SLF001
                self._rt._device, ptr + r * pitch * 2, cols * 2, "read bf16 row")
            out[r] = bf16_to_f32(raw.view(np.uint16))
        return out

    def conv1d_silu(self, wp: int, xp: int, sp: int, op: int, channels: int,
                    stream: int) -> None:
        """单 token。"""
        self.calls.append("conv1d_silu")
        taps = self._read_f32(wp, channels, 4, 4)
        x = self._read_bf16(xp, 1, channels, channels)[0]
        hist = self._read_f32(sp, 1, channels * 3, channels * 3).reshape(channels, 3)
        acc = (hist[:, 0] * taps[:, 0] + hist[:, 1] * taps[:, 1]
               + hist[:, 2] * taps[:, 2] + x * taps[:, 3])
        y = (acc / (1.0 + np.exp(-acc))).astype(np.float32)
        hist = np.stack([hist[:, 1], hist[:, 2], x], axis=1)
        self._write_f32(sp, 1, channels * 3, channels * 3, hist.reshape(1, -1))
        self._write_f32(op, 1, channels, channels, y.reshape(1, channels))

    def conv1d_silu_seq(self, wp: int, xp: int, sp: int, op: int, tokens: int,
                        channels: int, ldx: int, stream: int) -> None:
        """预填充分块（一趟走完）。"""
        self.calls.append("conv1d_silu_seq")
        taps = self._read_f32(wp, channels, 4, 4)
        x = self._read_bf16(xp, tokens, channels, ldx)
        hist = self._read_f32(sp, 1, channels * 3, channels * 3).reshape(channels, 3)
        out = np.empty((tokens, channels), dtype=np.float32)
        for t in range(tokens):
            acc = (hist[:, 0] * taps[:, 0] + hist[:, 1] * taps[:, 1]
                   + hist[:, 2] * taps[:, 2] + x[t] * taps[:, 3])
            out[t] = (acc / (1.0 + np.exp(-acc))).astype(np.float32)
            hist = np.stack([hist[:, 1], hist[:, 2], x[t]], axis=1)
        self._write_f32(sp, 1, channels * 3, channels * 3, hist.reshape(1, -1))
        self._write_f32(op, tokens, channels, channels, out)

    def conv1d_silu_batch(self, wp: int, xp: int, sp: int, op: int, rows: int, channels: int,
                          ldx: int, lds: int, ldo: int, stream: int) -> None:
        """批处理单 token。"""
        self.calls.append("conv1d_silu_batch")
        taps = self._read_f32(wp, channels, 4, 4)
        x = self._read_bf16(xp, rows, channels, ldx)
        hist = self._read_f32(sp, rows, channels * 3, lds).reshape(rows, channels, 3)
        out = np.empty((rows, channels), dtype=np.float32)
        for r in range(rows):
            acc = (hist[r, :, 0] * taps[:, 0] + hist[r, :, 1] * taps[:, 1]
                   + hist[r, :, 2] * taps[:, 2] + x[r] * taps[:, 3])
            out[r] = (acc / (1.0 + np.exp(-acc))).astype(np.float32)
            hist[r] = np.stack([hist[r, :, 1], hist[r, :, 2], x[r]], axis=1)
        self._write_f32(sp, rows, channels * 3, lds, hist.reshape(rows, channels * 3))
        self._write_f32(op, rows, channels, ldo, out)


class NumpySolveLib:
    """单出口求解的 numpy 复刻，签名与 ``SolveShim`` 一致。

    复刻顺序与内核一致：前向替换在 fp32 里做（内核的解缓冲也是 fp32），只在写出的
    那一行进操作数 dtype；对角线按 1 处理，上三角从不读。
    """

    def __init__(self, rt: MockCudart) -> None:
        self._rt = rt
        self.path = "mock"
        self.calls: list[str] = []

    def _tensor(self, ptr: int, bn: int, rows: int, cols: int, stride: int,
                code: int) -> np.ndarray:
        """读一个 ``[bn, rows, cols]`` 视图，批步长显式。"""
        dtype = _dtype(code)
        out = np.empty((bn, rows, cols), dtype=np.float32)
        for b in range(bn):
            base = ptr + b * stride * dtype.itemsize
            for r in range(rows):
                raw = self._rt._resolve(  # noqa: SLF001 - 同一个包内的假运行时
                    self._rt._device, base + r * cols * dtype.itemsize,
                    cols * dtype.itemsize, "read row")
                out[b, r] = bf16_to_f32(raw.view(np.uint16)) if code == 1 else raw.view(np.float32)
        return out

    def _write(self, ptr: int, bn: int, rows: int, cols: int, stride: int,
               values: np.ndarray, code: int) -> None:
        """写一个 ``[bn, rows, cols]`` 视图。"""
        dtype = _dtype(code)
        for b in range(bn):
            base = ptr + b * stride * dtype.itemsize
            for r in range(rows):
                dst = self._rt._resolve(  # noqa: SLF001
                    self._rt._device, base + r * cols * dtype.itemsize,
                    cols * dtype.itemsize, "write row")
                if code == 1:
                    dst.view(np.uint16)[:] = bf16_bits(values[b, r])
                else:
                    dst.view(np.float32)[:] = values[b, r].astype(np.float32)

    def triangular_solve(self, lowerp: int, rhsp: int, outp: int, code: int, bn: int, order: int,
                         cols: int, ldb: int, ldr: int, stream: int) -> None:
        """``A X = B``，``A`` 单位下三角，逐系统前向替换。"""
        self.calls.append("triangular_solve")
        a = self._tensor(lowerp, bn, order, order, ldb, code)
        rhs = self._tensor(rhsp, bn, order, cols, ldr, code)
        x = np.empty((bn, order, cols), dtype=np.float32)
        for k in range(bn):
            acc = rhs[k].astype(np.float64)
            for i in range(order):
                acc[i] -= a[k, i, :i].astype(np.float64) @ acc[:i]
            x[k] = acc.astype(np.float32)
        self._write(outp, bn, order, cols, ldr, x, code)


class NumpyNormLib:
    """三个归一化出口的 numpy 复刻，签名与 ``NormShim`` 一致。"""

    def __init__(self, rt: MockCudart) -> None:
        self._rt = rt
        self.path = "mock"
        self.calls: list[str] = []

    # 数据面

    def _rows(self, ptr: int, m: int, n: int, pitch: int, code: int) -> np.ndarray:
        """按行间距读一个 ``[m, n]`` 矩阵。"""
        dtype = _dtype(code)
        out = np.empty((m, n), dtype=np.float32)
        for r in range(m):
            row_ptr = ptr + r * pitch * dtype.itemsize
            raw = self._rt._resolve(  # noqa: SLF001 - 同一个包内的假运行时
                self._rt._device, row_ptr, n * dtype.itemsize, "read row")
            if dtype.code == 1:
                out[r] = bf16_to_f32(raw.view(np.uint16))
            else:
                out[r] = raw.view(np.float32)
        return out

    def _write_rows(self, ptr: int, m: int, n: int, values: np.ndarray, code: int) -> None:
        """按行间距写一个 ``[m, n]`` 矩阵。"""
        dtype = _dtype(code)
        for r in range(m):
            row_ptr = ptr + r * n * dtype.itemsize
            dst = self._rt._resolve(  # noqa: SLF001
                self._rt._device, row_ptr, n * dtype.itemsize, "write row")
            if dtype.code == 1:
                dst.view(np.uint16)[:] = bf16_bits(values[r])
            else:
                dst.view(np.float32)[:] = values[r].astype(np.float32)

    def _vec(self, ptr: int, n: int, code: int) -> np.ndarray:
        """读一个 ``[n]`` 增益。"""
        dtype = _dtype(code)
        raw = self._rt._resolve(self._rt._device, ptr, n * dtype.itemsize, "read vec")  # noqa: SLF001
        return bf16_to_f32(raw.view(np.uint16)) if dtype.code == 1 else raw.view(np.float32)

    # 出口

    def rms_norm(self, xp: int, wp: int, yp: int, m: int, n: int, pitch: int,
                 eps: float, code: int, stream: int) -> None:
        """``y = x * rsqrt(mean(x^2) + eps) * w``。"""
        self.calls.append("rms_norm")
        x = self._rows(xp, m, n, pitch, code)
        w = self._vec(wp, n, code)
        inv = _inv_rms(x, eps)
        self._write_rows(yp, m, n, x * inv[:, None] * w[None, :], code)

    def add_rms_norm(self, addp: int, xp: int, wp: int, yp: int, m: int, n: int, pitch: int,
                     eps: float, code: int, stream: int) -> None:
        """``x += add`` 后归一化，``x`` 原地写回舍入后的和。"""
        self.calls.append("add_rms_norm")
        add = self._rows(addp, m, n, pitch, code)
        x = self._rows(xp, m, n, pitch, code)
        total = x + add
        if code == 1:
            total = bf16_to_f32(bf16_bits(total))
        self._write_pitched(xp, m, n, pitch, total, code)
        w = self._vec(wp, n, code)
        inv = _inv_rms(total, eps)
        self._write_rows(yp, m, n, total * inv[:, None] * w[None, :], code)

    def rms_norm_gated(self, xp: int, wp: int, gatep: int, yp: int, m: int, n: int,
                       eps: float, code: int, stream: int) -> None:
        """``y = rms(x) * w * silu(gate)``。"""
        self.calls.append("rms_norm_gated")
        x = self._rows(xp, m, n, n, code)
        w = self._vec(wp, n, code)
        g = self._rows(gatep, m, n, n, code)
        silu = g / (1.0 + np.exp(-g))
        inv = _inv_rms(x, eps)
        self._write_rows(yp, m, n, x * inv[:, None] * w[None, :] * silu, code)

    def rms_norm_i8(self, xp: int, wp: int, qp: int, sp: int, m: int, n: int, pitch: int,
                    eps: float, threads: int, code: int, stream: int) -> None:
        """``q, s = quantize(rms(x) * w)``，absmax 取自舍入成 ``T`` 之后的值。"""
        self.calls.append("rms_norm_i8")
        x = self._rows(xp, m, n, pitch, code)
        w = self._vec(wp, n, code)
        inv = _inv_rms(x, eps)
        y = x * inv[:, None] * w[None, :]
        if code == 1:
            # 与内核同一顺序：先舍入再取 absmax，再算 scale。
            y = bf16_to_f32(bf16_bits(y))
        q, s = _quant_rows(y)
        self._write_i8(qp, m, n, q)
        self._write_vec_f32(sp, m, s)

    def add_rms_norm_i8(self, addp: int, xp: int, wp: int, qp: int, sp: int, m: int, n: int,
                        pitch: int, eps: float, threads: int, code: int,
                        stream: int) -> None:
        """``x += add``（原地），``q, s = quantize(rms(x) * w)``。"""
        self.calls.append("add_rms_norm_i8")
        add = self._rows(addp, m, n, pitch, code)
        x = self._rows(xp, m, n, pitch, code)
        total = x + add
        if code == 1:
            total = bf16_to_f32(bf16_bits(total))
        self._write_pitched(xp, m, n, pitch, total, code)
        w = self._vec(wp, n, code)
        inv = _inv_rms(total, eps)
        y = total * inv[:, None] * w[None, :]
        if code == 1:
            y = bf16_to_f32(bf16_bits(y))
        q, s = _quant_rows(y)
        self._write_i8(qp, m, n, q)
        self._write_vec_f32(sp, m, s)

    def _write_i8(self, ptr: int, m: int, n: int, q: np.ndarray) -> None:
        """把 ``[m, n]`` int8 稠密写回设备。"""
        for r in range(m):
            dst = self._rt._resolve(self._rt._device, ptr + r * n, n, "write int8 row")  # noqa: SLF001
            dst.view(np.int8)[:] = q[r].astype(np.int8)

    def _write_vec_f32(self, ptr: int, count: int, values: np.ndarray) -> None:
        """把 ``[count]`` fp32 向量写回设备。"""
        dst = self._rt._resolve(self._rt._device, ptr, count * 4, "write fp32 vec")  # noqa: SLF001
        dst.view(np.float32)[:] = values.astype(np.float32)

    def _write_pitched(self, ptr: int, m: int, n: int, pitch: int,
                       values: np.ndarray, code: int) -> None:
        """按行间距把 ``values`` 写回设备。"""
        dtype = _dtype(code)
        for r in range(m):
            row_ptr = ptr + r * pitch * dtype.itemsize
            dst = self._rt._resolve(self._rt._device, row_ptr, n * dtype.itemsize, "write row")  # noqa: SLF001
            if dtype.code == 1:
                dst.view(np.uint16)[:] = bf16_bits(values[r])
            else:
                dst.view(np.float32)[:] = values[r].astype(np.float32)


def _dtype(code: int) -> DType:
    """把 ABI 码折回 dtype。

    Raises:
        ShapeError: 码值不在表里。
    """
    from cxrt.dtypes import BY_CODE

    dtype = BY_CODE.get(int(code))
    if dtype is None:
        raise ShapeError(f"unknown dtype code {code}")
    return dtype


def _inv_rms(x: np.ndarray, eps: float) -> np.ndarray:
    """按行的 ``1 / sqrt(mean(x^2) + eps)``，fp32 累加。"""
    ss = np.sum(x * x, axis=1, dtype=np.float32)
    return (1.0 / np.sqrt(ss / np.float32(x.shape[1]) + np.float32(eps))).astype(np.float32)


def _quant_rows(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """按行做 ``absmax / 127`` 量化，与内核的 ``rms_norm_i8`` 同一套约定。

    下限 ``1e-5`` 与 ``quant_act_i8`` 一致：全零行量化成零，而不是除出 inf。

    Args:
        y: ``[M, N]`` fp32 归一化值。

    Returns:
        ``(codes, scales)``：``[M, N]`` int8 与 ``[M]`` fp32。
    """
    amax = np.max(np.abs(y), axis=1).astype(np.float32)
    sc = np.maximum(amax, np.float32(1e-5)) / np.float32(127.0)
    q = np.clip(np.rint(y / sc[:, None]), -127, 127)
    return q, sc


def host_reference(kind: str, x: np.ndarray, w: np.ndarray, *, add: np.ndarray | None = None,
                   gate: np.ndarray | None = None, eps: float = 1e-6) -> np.ndarray:
    """host 侧参考实现，fp64 计算后回 fp32。

    与 :class:`NumpyNormLib` 的区别只在累加精度，所以两者的差就是「复刻够不够像」的
    指标；真正的验收在卡上对内核做逐位对比。

    Args:
        kind: ``"rms"`` / ``"add"`` / ``"gated"``。
        x: ``[M, N]`` 激活，fp32。
        w: ``[N]`` 增益。
        add: ``add`` 变体的加数。
        gate: 门控变体的门。
        eps: 方差下限。

    Returns:
        ``[M, N]`` fp32 结果；``add`` 变体还额外把原地和写回 ``x``（由调用方处理）。
    """
    activ = x.astype(np.float64)
    if kind == "add":
        assert add is not None
        activ = activ + add.astype(np.float64)
    inv = 1.0 / np.sqrt(np.mean(activ * activ, axis=1) + eps)
    out = activ * inv[:, None] * w.astype(np.float64)[None, :]
    if kind == "gated":
        assert gate is not None
        g = gate.astype(np.float64)
        out = out * (g / (1.0 + np.exp(-g)))
    return out.astype(np.float32)


def make_ops(code_name: str = "f32") -> Any:
    """造一套「假驱动 + numpy 内核」的运行环境，供单测直接使用。

    Args:
        code_name: ``"f32"`` 或 ``"bf16"``，决定用例默认的 dtype。

    Returns:
        ``(rt, alloc, ops)``：假运行时、分配器、``RMSNormOps``。
    """
    from cxrt.dtypes import BY_NAME
    from cxrt.memory import Allocator
    from cxrt.norm import RMSNormOps

    if code_name not in BY_NAME:
        raise ShapeError(f"unknown dtype name {code_name}")
    rt = MockCudart()
    alloc = Allocator(rt, retain_bytes=1 << 24)
    return rt, alloc, RMSNormOps(NumpyNormLib(rt), rt, alloc)  # type: ignore[arg-type]


def make_solve_ops(code_name: str = "f32") -> Any:
    """造一套「假驱动 + numpy 求解」的运行环境，供单测直接使用。

    Args:
        code_name: ``"f32"`` 或 ``"bf16"``，决定用例默认的 dtype。

    Returns:
        ``(rt, alloc, ops)``：假运行时、分配器、``SolveOps``。
    """
    from cxrt.dtypes import BY_NAME
    from cxrt.memory import Allocator
    from cxrt.solve import SolveOps

    if code_name not in BY_NAME:
        raise ShapeError(f"unknown dtype name {code_name}")
    rt = MockCudart()
    alloc = Allocator(rt, retain_bytes=1 << 24)
    return rt, alloc, SolveOps(NumpySolveLib(rt))  # type: ignore[arg-type]


def make_conv_ops(code_name: str = "bf16") -> Any:
    """造一套「假驱动 + numpy 卷积」的运行环境，供单测直接使用。

    Args:
        code_name: 占位参数，保持与 ``make_ops`` 一致的调用形状（卷积只有 bf16 激活）。

    Returns:
        ``(rt, alloc, ops)``：假运行时、分配器、``ConvOps``。
    """
    from cxrt.conv import ConvOps
    from cxrt.dtypes import BY_NAME
    from cxrt.memory import Allocator

    if code_name not in BY_NAME:
        raise ShapeError(f"unknown dtype name {code_name}")
    rt = MockCudart()
    alloc = Allocator(rt, retain_bytes=1 << 24)
    return rt, alloc, ConvOps(NumpyConvLib(rt))  # type: ignore[arg-type]
