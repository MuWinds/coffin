"""``Array``：设备缓冲 + 形状 + 步长 + dtype，``torch.Tensor`` 在 corex 路径上的替代物。

Module: ``coffin/cxrt``。只 import ``rt.memory`` / ``rt.dtypes`` /
``rt.driver`` 与 ``numpy``，**不 import torch**。

为什么不是「直接用 numpy 的 ndarray 包一层指针」：模型侧需要的东西很少——
``empty`` / ``zeros`` / 形状与步长 / 行视图 / 一条列的切片——但每一样都必须能算出
「哪个字节」，因为内核只收裸指针加间距。``Array`` 把这件事集中在一处，内核与调用点
就都不必各自做指针算术。

Constraints:
    * 视图零拷贝：``row`` / ``narrow`` / ``view`` 只改 ``offset`` 与 ``shape``/``strides``，
    共享同一个 :class:`~coffin.cxrt.memory.Buffer`。谁的生命周期更长由
    Python 引用计数决定，不需要额外登记表。
    * 步长的单位是**元素**，与 torch 一致；指针算术时再乘 ``dtype.itemsize``。混用元素与
    字节是本文件唯一能算错的东西，所有偏移都只在一个地方乘。
    * 不实现广播、类型提升与运算符重载。这些语义在模型层换成 ``Array`` 之后由
    显式的 ``ops`` 调用承担——在设备数组上做隐式广播，等价于在解码主循环里藏一次
    临时分配，那正是这一层要消灭的东西。
"""

from __future__ import annotations

import numpy as np

from cxrt.driver import Cudart
from cxrt.dtypes import BF16, BY_CODE, DType, bf16_bits, from_numpy
from cxrt.errors import RtError, ShapeError
from cxrt.memory import Allocator, BorrowedBuffer, Buffer, HostBuffer


def _dense_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """返回行主序的连续步长。

    Args:
        shape: 形状，元素个数。

    Returns:
        步长元组，最后一个元素是 1。
    """
    strides: list[int] = []
    acc = 1
    for size in reversed(shape):
        strides.append(acc)
        acc *= size
    return tuple(reversed(strides))


class Array:
    """一块设备内存上的多维视图。

    Attributes:
        shape: 各维长度。
        strides: 各维步长，单位是元素。
        dtype: 存储类型。
        offset: 视图起点相对缓冲的字节偏移。
    """

    __slots__ = ("_buf", "offset", "shape", "strides", "dtype", "_closed", "_rt")

    def __init__(self, rt: Cudart, buf: Buffer, shape: tuple[int, ...],
                 strides: tuple[int, ...], dtype: DType, offset: int = 0) -> None:
        self._rt = rt
        self._buf = buf
        self.shape = tuple(int(s) for s in shape)
        self.strides = tuple(int(s) for s in strides)
        self.dtype = dtype
        self.offset = int(offset)
        self._closed = False

    # 构造

    @classmethod
    def empty(cls, rt: Cudart, alloc: Allocator, shape: tuple[int, ...],
              dtype: DType) -> "Array":
        """分配一块未初始化的设备内存并包成视图。

        Args:
            rt: 运行时句柄。
            alloc: 分配器。
            shape: 形状。
            dtype: 存储类型。

        Returns:
            新数组，内容是未初始化的。

        Raises:
            RtError: 形状为空或显存不足。
        """
        if not shape or any(int(s) <= 0 for s in shape):
            raise ShapeError(f"Array.empty{shape}: every dimension must be positive")
        numel = 1
        for size in shape:
            numel *= int(size)
        buf = alloc.malloc(numel * dtype.itemsize)
        return cls(rt, buf, tuple(int(s) for s in shape), _dense_strides(shape), dtype)

    @classmethod
    def zeros(cls, rt: Cudart, alloc: Allocator, shape: tuple[int, ...],
              dtype: DType) -> "Array":
        """分配并清零。

        Args:
            rt: 运行时句柄。
            alloc: 分配器。
            shape: 形状。
            dtype: 存储类型。

        Returns:
            全零数组。
        """
        out = cls.empty(rt, alloc, shape, dtype)
        out.fill(0)
        return out

    @classmethod
    def from_numpy(cls, rt: Cudart, alloc: Allocator, values: np.ndarray,
                   dtype: DType | None = None) -> "Array":
        """把宿主数组拷到设备。

        Args:
            rt: 运行时句柄。
            alloc: 分配器。
            values: 宿主数组；非连续或非本机字节序会先规整。
            dtype: 目标类型；``None`` 时按 numpy dtype 映射，bf16 必须显式给出
                （numpy 没有 bf16，映射不到）。

        Returns:
            设备上的新数组。

        Raises:
            ShapeError: ``dtype`` 与内容无法无损对应（例如 fp64 没有设备侧对应）。
            RtError: 显存不足。
        """
        source = np.ascontiguousarray(values)
        if dtype is None:
            dtype = _dtype_of_host(source)
        if dtype is BF16:
            if source.dtype == np.uint16:
                payload = source
            else:
                payload = bf16_bits(source.astype(np.float32))
        else:
            payload = source.astype(dtype.numpy, copy=False)
        out = cls.empty(rt, alloc, payload.shape, dtype)
        out.copy_from(payload)
        return out

    # 元数据

    @classmethod
    def borrow(cls, t: object, rt: Cudart | None = None) -> "Array":
        """把框架张量包成 ``Array`` 视图，零拷贝、不接管所有权。

        这是「算子面写 ``Array``、调用方攥着 ``torch.Tensor``」这两边中间的桥：地址、
        形状、步长、dtype 码全部从对方自己问出来（``cxrt.tensors`` 里那套中立访问器），
        缓冲包一层 :class:`~cxrt.memory.BorrowedBuffer`，所以视图关掉时不会把别人的
        内存还进 cxrt 的池子。

        Args:
            t: ``torch.Tensor``（或任何有 ``data_ptr`` / ``shape`` / ``stride`` /
                ``dtype`` 的对象）。
            rt: 运行时句柄。只有 ``copy_from`` / ``to_numpy`` / ``fill`` 需要它；纯指针
                运算（本包所有内核入口）不需要，可以省。

        Returns:
            指向同一块设备内存的 ``Array`` 视图。

        Raises:
            ShapeError: dtype 在设备侧没有对应存储。
        """
        from cxrt.tensors import code_of, ptr_of, shape_of, strides_of

        shape = shape_of(t)
        strides = strides_of(t)
        dtype = BY_CODE[code_of(t)]
        start = ptr_of(t)
        storage = start
        offset_of = getattr(t, "storage_offset", None)
        if offset_of is not None:
            # 存储起点 = 视图起点 - 偏移。**不**用 ``untyped_storage()``：那会构造一个
            # Python ``Storage`` 对象，热路径上（每个解码步几十次归一化）一次就是十几
            # 微秒，而这里只需要一个地址。
            storage = start - int(offset_of()) * dtype.itemsize
        span = 1
        for size, step in zip(shape, strides):
            span += (size - 1) * step
        buf = BorrowedBuffer(start, span * dtype.itemsize, storage)
        return cls(rt, buf, shape, strides, dtype)

    @property
    def ndim(self) -> int:
        """维数。"""
        return len(self.shape)

    @property
    def numel(self) -> int:
        """元素个数。"""
        n = 1
        for size in self.shape:
            n *= size
        return n

    @property
    def nbytes(self) -> int:
        """这个视图覆盖的字节数（按最外围跨度算，含行间填充）。"""
        if not self.shape:
            return self.dtype.itemsize
        return (self.strides[0] * (self.shape[0] - 1) + self.shape[-1]) * self.dtype.itemsize \
            if self.ndim == 2 else self.numel * self.dtype.itemsize

    @property
    def ptr(self) -> int:
        """视图起点的设备地址。"""
        if self._closed:
            raise RtError("Array is closed")
        return self._buf.ptr + self.offset

    @property
    def contiguous(self) -> bool:
        """是否行主序连续。"""
        return self.strides == _dense_strides(self.shape)

    @property
    def row_pitch(self) -> int:
        """二维数组的行间距，单位元素。

        Returns:
            第 0 维步长。带间距的激活（更宽缓冲区的一列切片）就靠它寻址，**不要**
            假定它等于 ``shape[1]``。
        """
        if self.ndim != 2:
            raise ShapeError(f"row_pitch needs a 2-D array, got {self.ndim}-D")
        return self.strides[0]

    def describe(self) -> str:
        """返回一行形状/步长/dtype 描述，错误消息与自检用。"""
        return (f"{self.dtype.name}{list(self.shape)} strides={list(self.strides)} "
                f"ptr={self.ptr:#x}")

    # 视图

    def view(self, shape: tuple[int, ...]) -> "Array":
        """在连续数组上换一个形状，零拷贝。

        Args:
            shape: 新形状，元素总数必须等于原数组。

        Returns:
            共享同一块内存的新视图。

        Raises:
            ShapeError: 原数组不连续，或元素总数不匹配。非连续数组换形状需要真的
                重排，而重排不是一次元数据操作——悄悄拷贝会破坏「视图零拷贝」的承诺。
        """
        if not self.contiguous:
            raise ShapeError(f"view needs a contiguous array, got {self.describe()}")
        if int(np.prod(shape)) != self.numel:
            raise ShapeError(f"view{shape} does not fit {self.describe()}")
        return Array(self._rt, self._buf, tuple(int(s) for s in shape),
                     _dense_strides(shape), self.dtype, self.offset)

    def row(self, index: int) -> "Array":
        """二维数组的一行，零拷贝。

        Args:
            index: 行号。

        Returns:
            形状 ``(shape[1],)`` 的视图。

        Raises:
            ShapeError: 不是二维或行号越界。
        """
        if self.ndim != 2:
            raise ShapeError(f"row() needs a 2-D array, got {self.describe()}")
        if not 0 <= index < self.shape[0]:
            raise ShapeError(f"row {index} out of range for {self.describe()}")
        return Array(self._rt, self._buf, (self.shape[1],), (1,), self.dtype,
                     self.offset + index * self.strides[0] * self.dtype.itemsize)

    def rows(self, start: int, stop: int) -> "Array":
        """二维数组的行区间，零拷贝。

        Args:
            start: 起始行（含）。
            stop: 结束行（不含）。

        Returns:
            形状 ``(stop-start, shape[1])`` 的视图。

        Raises:
            ShapeError: 不是二维或区间越界。
        """
        if self.ndim != 2:
            raise ShapeError(f"rows() needs a 2-D array, got {self.describe()}")
        if not 0 <= start < stop <= self.shape[0]:
            raise ShapeError(f"rows({start},{stop}) out of range for {self.describe()}")
        return Array(self._rt, self._buf, (stop - start, self.shape[1]),
                     self.strides, self.dtype,
                     self.offset + start * self.strides[0] * self.dtype.itemsize)

    def narrow(self, width: int) -> "Array":
        """把二维视图的最后一维收窄到 ``width``，行间距不变。

        这是带间距激活的入口：``q_norm`` 归一化的是更宽缓冲区里的一条列切片，列切片
        的「行间距」是原缓冲区的宽度，而不是 ``width``。内核按间距寻址，所以这里只改
        ``shape``、保留 ``strides``。

        Args:
            width: 新的列数，必须不大于原列数。

        Returns:
            形状 ``(rows, width)`` 的视图，步长不变。
        """
        if self.ndim != 2:
            raise ShapeError(f"narrow() needs a 2-D array, got {self.describe()}")
        if not 0 < width <= self.shape[1]:
            raise ShapeError(f"narrow({width}) out of range for {self.describe()}")
        return Array(self._rt, self._buf, (self.shape[0], width), self.strides, self.dtype,
                     self.offset)

    def transpose2d(self) -> "Array":
        """二维转置视图，零拷贝。

        Returns:
            形状互换、步长互换的视图。它**不再连续**，所以不能被 ``view``。

        Raises:
            ShapeError: 不是二维。
        """
        if self.ndim != 2:
            raise ShapeError(f"transpose2d() needs a 2-D array, got {self.describe()}")
        return Array(self._rt, self._buf, (self.shape[1], self.shape[0]),
                     (self.strides[1], self.strides[0]), self.dtype, self.offset)

    def same_storage(self, other: "Array") -> bool:
        """两块视图是否落在同一块缓冲上。

        Args:
            other: 另一块数组。

        Returns:
            同缓冲（含视图关系）为 ``True``。先比缓冲身份，再比地址区间——后者是
            :meth:`borrow` 带来的：同一个框架张量被包成两次视图时各有各的
            ``BorrowedBuffer``，身份比较会漏掉「目的地其实压着残差」这种别名。
        """
        if self._buf is None or other._buf is None:
            return False
        if self._buf is other._buf:
            return True
        a_storage = getattr(self._buf, "storage", None)
        b_storage = getattr(other._buf, "storage", None)
        if a_storage is not None and b_storage is not None:
            return a_storage == b_storage
        a0, a1 = self._buf.ptr, self._buf.ptr + self._buf.nbytes
        b0, b1 = other._buf.ptr, other._buf.ptr + other._buf.nbytes
        return a0 < b1 and b0 < a1

    # 数据搬运

    def copy_from(self, values: np.ndarray, stream: int | None = None) -> None:
        """把宿主数组写进这块设备内存。

        Args:
            values: 宿主数组，形状与元素数必须与本数组一致。

        Raises:
            ShapeError: 元素数不匹配。
            RtError: 拷贝失败。
        """
        source = np.ascontiguousarray(values)
        if source.size != self.numel:
            raise ShapeError(f"copy_from: host has {source.size} elements, "
                            f"array has {self.numel}")
        if source.dtype != self.dtype.numpy:
            raise ShapeError(f"copy_from: host dtype {source.dtype} != {self.dtype.name}")
        if not self.contiguous:
            raise ShapeError(f"copy_from needs a contiguous destination, got {self.describe()}")
        staging = HostBuffer(self._rt, self.numel * self.dtype.itemsize)
        try:
            staging.view(self.dtype, self.numel)[:] = source.reshape(-1)
            self._rt.copy_h2d(self.ptr, staging.ptr, self.numel * self.dtype.itemsize, stream)
        finally:
            staging.close()

    def to_numpy(self, stream: int | None = None) -> np.ndarray:
        """把设备内容读回宿主，返回一份独立拷贝。

        Args:
            stream: 拷贝所在的流；``None`` 表示默认流（同步）。

        Returns:
            形状同 ``shape`` 的 numpy 数组。bf16 以 ``uint16`` 位模式返回。

        Raises:
            RtError: 拷贝失败。
        """
        if not self.contiguous:
            raise ShapeError(f"to_numpy needs a contiguous array, got {self.describe()}")
        staging = HostBuffer(self._rt, self.numel * self.dtype.itemsize)
        try:
            self._rt.copy_d2h(staging.ptr, self.ptr, self.numel * self.dtype.itemsize, stream)
            if stream:
                self._rt.stream_sync(stream)
            return staging.view(self.dtype, self.numel).reshape(self.shape).copy()
        finally:
            staging.close()

    def fill(self, value: float) -> None:
        """把整块内存填成一个字节模式。

        Args:
            value: 填进去的字节值，取低 8 位。

        Raises:
            RtError: 驱动拒绝。
        """
        if not self.contiguous:
            raise ShapeError(f"fill needs a contiguous array, got {self.describe()}")
        self._rt.memset(self.ptr, int(value) & 0xFF, self.numel * self.dtype.itemsize)

    # 生命周期

    def close(self) -> None:
        """解除本视图对缓冲的引用。幂等。

        只丢掉这一份引用，**不**关掉缓冲：行/列视图与父数组共享同一个
        :class:`~cxrt.memory.Buffer`，最后一个持有者释放时缓冲才真的回到池里。
        ``close()`` 直接关缓冲会让「父数组关掉、视图还能用」变成「父数组关掉、视图指向
        已回收的地址」——这是最难查的一类悬垂指针。
        """
        if self._closed:
            return
        self._closed = True
        self._buf = None

    def __enter__(self) -> "Array":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"Array({self.dtype.name}{list(self.shape)})"


def _dtype_of_host(values: np.ndarray) -> DType:
    """按 numpy dtype 找设备 dtype。

    Args:
        values: 宿主数组。

    Returns:
        对应的 :class:`DType`；bf16 只能由调用方显式指定，所以这里遇到 ``uint16``
        返回的是 ``u16`` 而不是 ``bf16``。

    Raises:
        ShapeError: 该 dtype 在设备侧没有对应。
    """
    return from_numpy(values.dtype)


__all__ = ["Array"]
