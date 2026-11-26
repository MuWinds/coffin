"""投影族的 C ABI 面：``ctypes`` 直调 ``abi/gemv_abi.cu``，不经过 torch。

Module: ``coffin/cxrt``。只 import ``ctypes`` 与 ``cxrt.*``，**不 import torch**：
这一层存在的全部理由就是 corex 的投影路径不再需要框架——张量只按「指针 + dtype 码 +
形状」参与，输出由调用方持有。

只有 nk 权重（``[N, K]``，一行权重一个输出列）。corex 的 int8 权重本来就是这个布局，
以后为它写的适配层默认也是 nk；``[K, N]`` 的那两族入口没有搬进 ABI 面，免得再养一条
没人走的路径。真要 kn，先在权重侧转置成 nk 再进来。

与 ``norm.py`` 的分工相同：:class:`GemvShim` 只保证「按 ABI 传参」，
:class:`GemvOps` 做形状/dtype 校验。

Constraints:
    * ``out`` 永远由调用方给或者新分配，绝不复用——它是返回给上层的值，复用会让上一步
      的结果被就地覆盖（而调用方常常还攥着它做后续加法）。
    * 参数顺序与 ``abi/gemv_abi.cu`` 逐字对齐。``ctypes`` 少写一个参数不报错，只会把
      后面所有参数错位一格，症状是内核读到垃圾。
    * nk 路径没有 split-K，所以没有 ``part``/``counters`` 暂存，也没有计划查询入口：
      一个输出列就是一个 32 线程组，网格天然铺满 ``N``。
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

from cxrt import tensors as T
from cxrt.dtypes import BY_CODE
from cxrt.errors import ShapeError
from cxrt.shim import Shim


class GemvShim:
    """nk 投影两个出口的薄转发层。"""

    def __init__(self, shim: Shim) -> None:
        """绑定出口。

        Args:
            shim: 已装载的共享库。
        """
        p = ctypes.c_void_p
        i = ctypes.c_int
        ll = ctypes.c_longlong
        self.nk_i8 = shim.export(
            "coffin_gemv_nk_i8", [p, p, i, p, ll, ll, p, i, i, i, p])
        self.nk_i8_m = shim.export(
            "coffin_gemv_nk_i8_m", [p, p, i, p, ll, ll, ll, p, i, i, i, p])
        self.nk_dense = shim.export(
            "coffin_gemv_nk_dense", [p, p, i, ll, ll, p, i, p])
        self.nk_dense_m = shim.export(
            "coffin_gemv_nk_dense_m", [p, p, i, ll, ll, ll, p, i, p])
        self.path = shim.path


def load_gemv_shim(lib_path: str | Path) -> GemvShim:
    """装载投影 shim。

    Args:
        lib_path: ``cxrt.build`` 产出的 ``.so`` 路径。

    Returns:
        绑好两个出口的 :class:`GemvShim`。
    """
    return GemvShim(Shim(lib_path))


class GemvOps:
    """投影族的可调用面：校验形状、分配输出、调 ABI。

    Attributes:
        shim: :class:`GemvShim`。
    """

    def __init__(self, shim: GemvShim) -> None:
        self.shim = shim

    def nk_i8(self, wtn: T.TensorLike, x: T.TensorLike, scale: T.TensorLike,
              out: T.TensorLike | None = None, threads: int = 0, unroll: int = 0,
              width: int = 0, stream: int | None = None) -> T.TensorLike:
        """``y[n] = scale[n] * dot(wtn[n, :], x)``，``wtn`` 为 ``[N, K]`` int8。

        Args:
            wtn: ``[N, K]`` int8 权重，连续。
            x: ``[K]`` 激活。
            scale: ``[N]`` fp32。
            out: ``[N]`` 输出；``None`` 表示按 ``x`` 分配（dtype 跟随 ``x``）。
            threads: 每块线程数，``0`` 自动；必须是 32 的倍数。
            unroll: 展开度，``0`` 表示 4。
            width: 每组列数，``0`` 表示 1；只接受 1 或 2。
            stream: 透传的流句柄。

        Returns:
            ``out``，dtype 同 ``x``。

        Raises:
            ShapeError: 形状、dtype 或对齐对不上。
        """
        n, k, _ = self._check_nk(wtn, x, scale)
        if out is None:
            out = T.new_empty(x, (n,), T.code_of(x))
        self._check_nk_out(out, (n,))
        self._call(self.shim.nk_i8, wtn, x, scale, n, k, 0, out, threads, unroll, width,
                   stream)
        return out

    def nk_i8_m(self, wtn: T.TensorLike, x: T.TensorLike, scale: T.TensorLike,
                out: T.TensorLike | None = None, threads: int = 0, unroll: int = 0,
                width: int = 0, stream: int | None = None) -> T.TensorLike:
        """批处理形态的 :meth:`nk_i8`：``x`` 为 ``[M, K]``。

        Args:
            wtn: ``[N, K]`` int8 权重。
            x: ``[M, K]`` 激活。
            scale: ``[N]`` fp32。
            out: ``[M, N]`` 输出；``None`` 表示按 ``x`` 分配。
            threads: 每块线程数，``0`` 自动。
            unroll: 展开度，``0`` 表示 4。
            width: 每组列数，``0`` 表示 1。
            stream: 流。

        Returns:
            ``out``，dtype 同 ``x``。

        Raises:
            ShapeError: 形状、dtype 或对齐对不上。
        """
        n, k, m = self._check_nk(wtn, x, scale)
        if out is None:
            out = T.new_empty(x, (m, n), T.code_of(x))
        self._check_nk_out(out, (m, n))
        self._call(self.shim.nk_i8_m, wtn, x, scale, n, k, m, out, threads, unroll, width,
                   stream)
        return out

    @staticmethod
    def _call(entry: Any, wtn: T.TensorLike, x: T.TensorLike, scale: T.TensorLike,
              n: int, k: int, m: int, out: T.TensorLike, threads: int, unroll: int,
              width: int, stream: int | None) -> None:
        """两个出口共用的参数编组：``(wtn, x, x_code, scale, N, K[, M], y, ...)``。

        Args:
            entry: 已绑定的出口。
            wtn: 权重。
            x: 激活。
            scale: 每列 scale。
            n: 输出列数。
            k: 内维。
            m: 行数（``0`` 表示单行出口，不带这个参数）。
            out: 输出。
            threads: 线程数请求。
            unroll: 展开度请求。
            width: 每组列数请求。
            stream: 流。
        """
        args = [T.ptr_of(wtn), T.ptr_of(x), T.code_of(x), T.ptr_of(scale), int(n),
                int(k)]
        if m:
            args.append(int(m))
        args += [T.ptr_of(out), _threads_nk(threads), int(unroll) or 4, int(width),
                 None if stream is None else int(stream)]
        entry(*args)

    def _check_nk(self, wtn: T.TensorLike, x: T.TensorLike,
                  scale: T.TensorLike) -> tuple[int, int, int]:
        """检查 nk 权重、激活与 scale，返回 ``(N, K, M)``。

        Args:
            wtn: ``[N, K]`` int8 权重。
            x: ``[K]`` 或 ``[M, K]`` 激活。
            scale: ``[N]`` fp32。

        Returns:
            ``(输出列数, 内维, 行数)``。

        Raises:
            ShapeError: 形状、dtype、``K % 4`` 或 bf16 对齐对不上。
        """
        wshape = T.shape_of(wtn)
        if len(wshape) != 2:
            raise ShapeError(f"nk weight must be [N, K], got {wshape}")
        if T.code_of(wtn) != T.I8:
            raise ShapeError(f"nk weight must be int8, got {BY_CODE[T.code_of(wtn)].name}")
        n, k = wshape
        if k % 4:
            raise ShapeError(f"nk GEMV needs K % 4 == 0, got K={k}")
        xshape = T.shape_of(x)
        if len(xshape) not in (1, 2) or xshape[-1] != k:
            raise ShapeError(f"activation {xshape} does not match K={k}")
        if T.code_of(x) not in (T.F32, T.BF16C):
            raise ShapeError(f"activation dtype {BY_CODE[T.code_of(x)].name} is not bf16/fp32")
        if T.code_of(x) == T.BF16C and T.ptr_of(x) % 8:
            raise ShapeError("bf16 activation must be 8-byte aligned (nk reads four halves "
                             "as one word)")
        if T.numel_of(scale) != n:
            raise ShapeError(f"scale must hold N={n} values, got {T.numel_of(scale)}")
        if T.code_of(scale) != T.F32:
            raise ShapeError(f"scale must be fp32, got {BY_CODE[T.code_of(scale)].name}")
        m = 1 if len(xshape) == 1 else xshape[0]
        return n, k, m

    def _check_nk_out(self, out: T.TensorLike, shape: tuple[int, ...]) -> None:
        """检查输出形状与 dtype 跟随激活。

        Args:
            out: 输出张量。
            shape: 期望形状。

        Raises:
            ShapeError: 形状或 dtype 对不上。
        """
        if T.shape_of(out) != shape:
            raise ShapeError(f"out{tuple(T.shape_of(out))} != {shape}")
        if T.code_of(out) not in (T.F32, T.BF16C):
            raise ShapeError(f"out dtype {BY_CODE[T.code_of(out)].name} is not bf16/fp32")


def _threads_nk(threads: int) -> int:
    """nk 路径的线程数：``0`` 表示 128（``pick_nk_threads`` 的取值，卡片上可查）。

    Args:
        threads: 请求值。

    Returns:
        线程数。
    """
    return int(threads) if threads > 0 else 128
