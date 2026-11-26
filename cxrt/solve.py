"""求解族的 C ABI 面：``triangular_solve``（单位下三角 ``A X = B``）。

Module: ``coffin/cxrt``。只 import ``ctypes`` 与 ``cxrt.*``，**不 import torch**。

这是 delta 规则每层每块都要走的那一步。它之所以是一个算子而不是模型里几行 Python，
是因为本卡的 ``torch.linalg`` 求解路径全部在 ``dlopen`` 里失败（wheel 写死了
``/opt/sw_home/local/cuda/lib64/libcusolver.so``），而用 ``bmm`` 链写的分块前向代入要
每块发 4-8 次启动。

Constraints:
    * 批步长从**张量自身**读取，所以唯一块（``eye(L).expand(bn, L, L)``，批步长为 0）
      可以原样传进来，不必为一次读实体化 ``bn * L * L`` 个元素。
    * 行布局固定：最内层步长为 1、批内行间距等于行宽。放宽它就得再往内核传两个步长。
    * 对角线按 1 处理、上三角永不读取——这是 delta 规则 ``UT`` 的语义，不是本实现的
      简化。
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

from cxrt import tensors as T
from cxrt.dtypes import BY_CODE
from cxrt.errors import ShapeError
from cxrt.shim import Shim


class SolveShim:
    """单个求解出口的薄转发层。"""

    def __init__(self, shim: Shim) -> None:
        """绑定出口。

        Args:
            shim: 已装载的共享库。
        """
        p = ctypes.c_void_p
        i = ctypes.c_int
        ll = ctypes.c_longlong
        # coffin_triangular_solve(lower, rhs, out, dtype, bn, L, n, ldb, ldr, stream,
        #                          err, errlen)
        # ``Shim.export`` 自己追加 err/errlen，所以这里只列前 10 个——多写两个会让
        # ctypes 在调用点报 "takes at least 14 arguments (12 given)"。
        self.triangular_solve = shim.export(
            "coffin_triangular_solve", [p, p, p, i, i, i, i, ll, ll, p])
        self.path = shim.path


def load_solve_shim(lib_path: str | Path) -> SolveShim:
    """装载求解 shim。

    Args:
        lib_path: ``cxrt.build`` 产出的 ``.so`` 路径。

    Returns:
        绑好出口的 :class:`SolveShim`。
    """
    return SolveShim(Shim(lib_path))


class SolveOps:
    """求解算子的可调用面。

    Attributes:
        shim: :class:`SolveShim`。
    """

    def __init__(self, shim: SolveShim) -> None:
        self.shim = shim

    def triangular_solve(self, lower: T.TensorLike, rhs: T.TensorLike,
                         out: T.TensorLike | None = None,
                         stream: int | None = None) -> T.TensorLike:
        """解 ``A X = B``，``A`` 是单位下三角。

        Args:
            lower: ``[bn, L, L]`` 系数，bf16 或 fp32；只读严格下三角。
            rhs: ``[bn, L, n]`` 右端项，dtype 与 ``lower`` 相同。
            out: ``[bn, L, n]`` 目的地；``None`` 时按 ``rhs`` 分配。必须与输入不重叠。
            stream: 流句柄。

        Returns:
            解张量 ``[bn, L, n]``。

        Raises:
            ShapeError: 维数、dtype、行布局或目的地不符。
        """
        lshape = T.shape_of(lower)
        rshape = T.shape_of(rhs)
        if len(lshape) != 3 or len(rshape) != 3:
            raise ShapeError(f"lower/rhs must both be 3-D, got {lshape} and {rshape}")
        code = T.code_of(rhs)
        if code not in (T.BF16C, T.F32):
            raise ShapeError(f"operands must be bf16 or fp32, got {BY_CODE[code].name}")
        if T.code_of(lower) != code:
            raise ShapeError("lower and rhs must share a dtype")
        bn, order, cols = (int(v) for v in rshape)
        if tuple(int(v) for v in lshape) != (bn, order, order):
            raise ShapeError(f"lower must be [{bn}, {order}, {order}], got {lshape}")
        lstr = T.strides_of(lower)
        rstr = T.strides_of(rhs)
        if lstr[2] != 1 or rstr[2] != 1:
            raise ShapeError(f"operands must have unit innermost stride, got {lstr}/{rstr}")
        if lstr[1] != order or rstr[1] != cols:
            raise ShapeError(
                f"operands must be row-contiguous within a batch entry, got {lstr}/{rstr}")
        # ``lower`` 允许批步长 0：``eye(L).expand(bn, L, L)`` 的唯一块被所有系统共享，
        # 读取它不需要实体化。右端项不行——解是照着右端项的批步长写出去的，步长 0 会让
        # ``bn`` 次求解互相覆盖。
        if lstr[0] != 0 and lstr[0] < order * order:
            raise ShapeError(f"lower batch stride {lstr[0]} is narrower than {order * order}")
        if rstr[0] < order * cols:
            raise ShapeError(f"rhs batch stride {rstr[0]} is narrower than {order * cols}")
        if out is None:
            out = T.new_empty(rhs, (bn, order, cols), code)
        elif T.shape_of(out) != (bn, order, cols) or T.code_of(out) != code:
            raise ShapeError(
                f"out must be dense [{bn}, {order}, {cols}] {BY_CODE[code].name}, "
                f"got {T.shape_of(out)}")
        # 内核按 ``rhs`` 的批步长写 ``out``（torch 前端也是 ``out`` 新分配、沿用同一个
        # ``ldr``），所以目的地的批步长必须与之相同。
        if T.strides_of(out) != (rstr[0], cols, 1):
            raise ShapeError(f"out must have batch stride {rstr[0]}, got {T.strides_of(out)}")
        self.shim.triangular_solve(T.ptr_of(lower), T.ptr_of(rhs), T.ptr_of(out),
                                   code, bn, order, cols, int(lstr[0]), int(rstr[0]),
                                   _stream(stream))
        return out


def _stream(stream: int | None) -> Any:
    """流句柄。

    Args:
        stream: 句柄号或 ``None``。

    Returns:
        ``c_void_p`` 可接受的值。
    """
    return None if stream is None else int(stream)
