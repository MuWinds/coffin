"""W8A8 族激活量化的 C ABI 面：``quant_act_i8`` / ``gather_quant_i8`` / ``epilogue``。

Module: ``coffin/cxrt``。只 import ``ctypes`` 与 ``cxrt.*``，**不 import torch**。

前两条是 W8A8 预填充的**写方**：一行一个 token，`amax` 只在 K 上归约，`1e-5` 下限让
全零行量化成零而不是 inf；读方是库 GEMM 与 :meth:`W8A8Ops.epilogue`。``gather_quant_i8``
是 MoE 分派之后「按索引表搬行 + 量化」，与先 `index_select` 再 `quant_act_i8` 逐位相同。

``epilogue`` 是 int8 预填充三步里的最后一步：``cublas`` 的 ``GemmEx`` 只给 int32 累加器，
两个 scale（每 token 一个、每输出列一个）必须由这一条施加并**只窄化一次**。它单独走一条
出口，是因为把累加器交回 torch 做 `mul` 就等于给推理链路续上 torch。

Constraints:
    * 激活必须**连续**。包装层原本就先做 `x.contiguous()`，调用点给的也都是投影输出的
      稠密片；带间距的需求不存在，所以这里不收 pitch 参数，也不假装支持。
    * ``gather_quant_i8`` 整行先进共享内存，共享内存大小是 ``K * sizeof(T)``，由 C 侧算；
      Python 不重复这个公式。
    * ``epilogue`` 的累加器必须连续（C 侧按 ``c + m * N`` 走行），但 ``out`` 只要求行间距
      ``>= 列数``：沿 N 切块时每块写回更宽缓冲区的自己那一段，不必再拷一次。
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

from cxrt import tensors as T
from cxrt.dtypes import BY_CODE, F32, I8
from cxrt.errors import ShapeError
from cxrt.shim import Shim


class W8A8Shim:
    """两个量化出口的薄转发层。"""

    def __init__(self, shim: Shim) -> None:
        """绑定出口。

        Args:
            shim: 已装载的共享库。
        """
        p = ctypes.c_void_p
        i = ctypes.c_int
        ll = ctypes.c_longlong
        self.quant_act_i8 = shim.export(
            "coffin_quant_act_i8", [p, p, p, ll, ll, i, p])
        self.gather_quant_i8 = shim.export(
            "coffin_gather_quant_i8", [p, p, p, p, ll, ll, ll, i, p])
        self.epilogue = shim.export(
            "coffin_w8a8_epilogue", [p, p, p, p, ll, ll, ll, i, p])
        # nk 布局的 dp4a GEMM：融合输出模式由调用方显式给（ABI 面没有 dtype 可读）。
        self.dp4a_nk = shim.export(
            "coffin_w8a8_dp4a_nk", [p, p, p, p, p, p, i, i, i, i, i, p])
        self.path = shim.path


def load_w8a8_shim(lib_path: str | Path) -> W8A8Shim:
    """装载 W8A8 shim。

    Args:
        lib_path: ``cxrt.build`` 产出的 ``.so`` 路径。

    Returns:
        绑好两个出口的 :class:`W8A8Shim`。
    """
    return W8A8Shim(Shim(lib_path))


class W8A8Ops:
    """两个量化出口加 epilogue 的可调用面。

    Attributes:
        shim: :class:`W8A8Shim`。
    """

    def __init__(self, shim: W8A8Shim) -> None:
        self.shim = shim

    def quant_act_i8(self, x: T.TensorLike, out_q: T.TensorLike | None = None,
                     out_s: T.TensorLike | None = None,
                     stream: int | None = None) -> tuple[T.TensorLike, T.TensorLike]:
        """按行对称 int8 量化，一行一个块。

        Args:
            x: ``[M, K]`` 稠密激活，bf16 或 fp32。
            out_q: ``[M, K]`` int8；``None`` 时分配。
            out_s: ``[M]`` fp32 每行 scale；``None`` 时分配。
            stream: 流句柄。

        Returns:
            ``(out_q, out_s)``。

        Raises:
            ShapeError: 不是二维、dtype 不对，或目的地形状/类型不符。
        """
        rows, width, code = self._check_2d(x)
        q, s = self._dests(x, rows, width, out_q, out_s)
        self.shim.quant_act_i8(T.ptr_of(x), T.ptr_of(q), T.ptr_of(s), int(rows), int(width),
                               code, _stream(stream))
        return q, s

    def gather_quant_i8(self, x: T.TensorLike, index: T.TensorLike, rows: int = 0,
                        out_q: T.TensorLike | None = None, out_s: T.TensorLike | None = None,
                        stream: int | None = None) -> tuple[T.TensorLike, T.TensorLike]:
        """按 ``index`` 搬行并量化，一次启动。

        Args:
            x: 源激活，``[M, K]`` 稠密。
            index: 设备 int32 行号表，``[rows]``。
            rows: 索引表长度；``0`` 表示读 ``index`` 的元素数。
            out_q: ``[rows, K]`` int8；``None`` 时分配。
            out_s: ``[rows]`` fp32 每行 scale；``None`` 时分配。
            stream: 流句柄。

        Returns:
            ``(out_q, out_s)``。

        Raises:
            ShapeError: 索引表 dtype/长度、目的地形状或激活布局不符。
        """
        src_rows, width, code = self._check_2d(x)
        if T.code_of(index) != T.I32C:
            raise ShapeError(f"row index must be int32, got {BY_CODE[T.code_of(index)].name}")
        n = int(rows) or T.numel_of(index)
        if n != T.numel_of(index):
            raise ShapeError(f"index holds {T.numel_of(index)} rows, caller said {n}")
        q, s = self._dests(x, n, width, out_q, out_s)
        self.shim.gather_quant_i8(T.ptr_of(x), T.ptr_of(index), T.ptr_of(q), T.ptr_of(s),
                                  int(src_rows), int(n), int(width), code, _stream(stream))
        return q, s

    def epilogue(self, c: T.TensorLike, xs: T.TensorLike, ws: T.TensorLike,
                 out: T.TensorLike | None = None, stride: int = 0,
                 stream: int | None = None) -> T.TensorLike:
        """施加两个 W8A8 scale 并把 int32 累加器窄化一次。

        Args:
            c: ``[M, N]`` int32 累加器，连续。
            xs: ``[M]`` fp32 每 token 激活 scale。
            ws: ``[N]`` fp32 每输出列权重 scale，与本块列段对齐。
            out: ``[M, N]`` bf16 或 fp32；``None`` 时按 ``c`` 分配连续缓冲。
            stride: ``out`` 的行间距，单位元素；``0`` 表示读 ``out`` 自己的。
            stream: 流句柄。

        Returns:
            ``out``。

        Raises:
            ShapeError: 累加器不是 int32、两个 scale 长度不符，或 ``out`` 的 dtype/
                行间距吃不下这次写入。
        """
        shape = T.shape_of(c)
        if len(shape) != 2:
            raise ShapeError(f"accumulator must be 2-D [M, N], got {shape}")
        if T.code_of(c) != T.I32C:
            raise ShapeError(f"accumulator must be int32, got {BY_CODE[T.code_of(c)].name}")
        rows, cols = int(shape[0]), int(shape[1])
        if T.code_of(xs) != F32.code or T.numel_of(xs) != rows:
            raise ShapeError(f"xs must be fp32 with {rows} values (one per token)")
        if T.code_of(ws) != F32.code or T.numel_of(ws) != cols:
            raise ShapeError(f"ws must be fp32 with {cols} values (one per column)")

        given = out is not None
        if not given:
            # 目的地 dtype 跟着激活走：这里没有激活可看，只有 ``c``，所以默认按 bf16
            # 落地——corex 上模型张量就是 bf16，fp32 是调用方显式要的那一种。
            out = T.new_empty(c, (rows, cols), T.BF16C)
        else:
            oshape = T.shape_of(out)
            if len(oshape) != 2 or int(oshape[0]) != rows:
                raise ShapeError(f"out must be [{rows}, {cols}] or a column view of a "
                                 f"wider buffer, got {oshape}")
            if T.code_of(out) not in (T.BF16C, T.F32):
                raise ShapeError(f"out must be bf16 or fp32, got {BY_CODE[T.code_of(out)].name}")
            ostrides = T.strides_of(out)
            if int(ostrides[-1]) != 1:
                raise ShapeError(f"out must have unit inner stride, got {ostrides}")
        # 自己分配的目的地按构造就是连续的，行距就是列数；只有调用方给的才需要问。
        pitch = int(stride) or (int(T.strides_of(out)[0]) if given else cols)
        if pitch < cols:
            raise ShapeError(f"out pitch {pitch} narrower than {cols} columns")
        self.shim.epilogue(T.ptr_of(c), T.ptr_of(xs), T.ptr_of(ws), T.ptr_of(out), int(rows),
                           int(cols), pitch, T.code_of(out), _stream(stream))
        return out

    # 校验

    def _check_2d(self, x: T.TensorLike) -> tuple[int, int, int]:
        """检查二维激活，返回 ``(行数, 行宽, dtype 码)``。

        Args:
            x: 激活。

        Returns:
            ``(rows, width, code)``。

        Raises:
            ShapeError: 不是二维，或不是 bf16/fp32。
        """
        shape = T.shape_of(x)
        if len(shape) != 2:
            raise ShapeError(f"activation must be 2-D [M, K], got {shape}")
        code = T.code_of(x)
        if code not in (T.BF16C, T.F32):
            raise ShapeError(f"activation must be bf16 or fp32, got {BY_CODE[code].name}")
        return int(shape[0]), int(shape[1]), code

    def _dests(self, x: T.TensorLike, rows: int, width: int, out_q: T.TensorLike | None,
               out_s: T.TensorLike | None) -> tuple[T.TensorLike, T.TensorLike]:
        """决定两个目的地：用调用方的，或分配。

        Args:
            x: 激活，用作分配锚点。
            rows: 行数。
            width: 行宽。
            out_q: 给定或 ``None``。
            out_s: 给定或 ``None``。

        Returns:
            ``(out_q, out_s)``。

        Raises:
            ShapeError: 目的地形状或 dtype 不符。
        """
        if out_q is None:
            out_q = T.new_empty(x, (rows, width), I8.code)
        elif T.shape_of(out_q) != (rows, width) or T.code_of(out_q) != I8.code:
            raise ShapeError(
                f"out_q must be dense [{rows}, {width}] int8, got {T.shape_of(out_q)}")
        if out_s is None:
            out_s = T.new_empty(x, (rows,), F32.code)
        elif T.numel_of(out_s) != rows or T.code_of(out_s) != F32.code:
            raise ShapeError(f"out_s must be fp32 with {rows} values")
        return out_q, out_s


def _stream(stream: int | None) -> Any:
    """流句柄。

    Args:
        stream: 句柄号或 ``None``。

    Returns:
        ``c_void_p`` 可接受的值。
    """
    return None if stream is None else int(stream)
