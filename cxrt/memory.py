"""设备与宿主内存：按尺寸分档的池，加上 pinned 暂存。

Module: ``coffin/cxrt``。只 import ``rt.driver`` / ``rt.dtypes`` 与
``numpy``，**不 import torch**。``torch.cuda.caching_allocator`` 的替代物。

为什么自己池化而不是每次 ``cudaMalloc``：``cudaMalloc`` 在本平台是同步的、并在驱动里
做一次页表操作，解码步里每个算子都分配一次中间张量的话，分配会排进启动清单（原来
``q4budget2`` 实测一步 7025 次发射）。池只做「按 2 的幂档位复用」，不做碎片整理：
所有块的尺寸都是算子的形状，重复性极高，档位复用命中率接近 100%。

Constraints:
    * 档位按 2 的幂向上取整，最小 256 B。按精确尺寸做键会让每个不同 batch 尺寸各留
      一份永不回收的块，最后表现成「显存慢慢漏光」。
    * 池只回收它自己发出的块。``free`` 一个不是它分配的地址直接忽略并计数，绝不传给
      ``cudaFree``——那会释放别人的内存，症状是几层之后随机 NaN。
    * 保留上限默认 1 GiB，超过就把空闲块真的还给驱动。KV 池与权重都不走这个池（它们
      寿命是整个进程），所以这个上限只管激活与中间量。
"""

from __future__ import annotations

import ctypes

import numpy as np

from cxrt.driver import Cudart
from cxrt.dtypes import DType
from cxrt.errors import RtError

#: 最小档位。比它小的请求都落到这一档，避免一个 8 B 的标量占一个独立块。
_MIN_BUCKET = 256

#: 默认保留上限，超过之后空闲块被真正释放。可用 ``COFFIN_RT_POOL_BYTES`` 覆盖。
_DEFAULT_RETAIN = 1 << 30


def _bucket(nbytes: int) -> int:
    """把请求尺寸向上取到 2 的幂档位。

    Args:
        nbytes: 请求字节数，正数。

    Returns:
        档位字节数，至少 :data:`_MIN_BUCKET`。
    """
    size = _MIN_BUCKET
    while size < nbytes:
        size <<= 1
    return size


class Buffer:
    """一块设备内存的所有权凭证。

    ``Array`` 的视图共享同一个 ``Buffer``（Python 引用计数保证寿命），所以视图不需要
    单独登记。``close()`` 幂等，重复释放返回池里不会发生。

    Attributes:
        ptr: 设备地址。
        nbytes: 实际分配到的字节数（档位大小，可能大于请求）。
        requested: 调用方请求的字节数，诊断用。
    """

    __slots__ = ("ptr", "nbytes", "requested", "_allocator", "_closed")

    def __init__(self, ptr: int, nbytes: int, requested: int, allocator: "Allocator") -> None:
        self.ptr = ptr
        self.nbytes = nbytes
        self.requested = requested
        self._allocator = allocator
        self._closed = False

    def close(self) -> None:
        """把这块内存还给分配器。幂等。

        顺序很关键：先把所有权交给分配器，再把本地引用清空。反过来会让 ``free`` 看到
        一个 ``_allocator is None`` 的缓冲，把它当成外来指针丢弃——内存于是永不回收，
        症状是池里留下越来越多的死块。
        """
        if self._closed:
            return
        self._closed = True
        allocator = self._allocator
        if allocator is not None:
            allocator.free(self)
            self._allocator = None

    def __enter__(self) -> "Buffer":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:                    # pragma: no cover - 解释器退出路径
        try:
            self.close()
        except Exception:
            pass


class BorrowedBuffer:
    """别人（例如框架张量）拥有的设备内存，``Array`` 只是借来用。

    存在的理由：``cxrt`` 的算子面（``norm`` / ``conv`` / ``attn`` / ``solve``）写的是
    ``Array``，而调用方在 torch 那条路上攥着的是 ``torch.Tensor``。把后者包成
    ``Array`` 视图需要一块「有地址、但**不**归我释放」的缓冲——拿 ``Buffer`` 装上它
    会在视图销毁时把别人的内存还进 cxrt 的池子，下一次 ``malloc`` 就发同一块地址，
    症状是数值在几步之后随机漂移。

    Attributes:
        ptr: 设备地址。
        nbytes: 借用范围覆盖的字节数。
        storage: 对方「整块存储」的起始地址；框架张量拿得到 ``untyped_storage()``
            时填它。别名判定比的是这个，不是本视图的地址区间——和 torch 的
            ``is_alias_of`` 同口径：同一块存储上的两个不相交视图也算别名。
    """

    __slots__ = ("ptr", "nbytes", "storage")

    def __init__(self, ptr: int, nbytes: int, storage: int | None = None) -> None:
        self.ptr = int(ptr)
        self.nbytes = int(nbytes)
        self.storage = None if storage is None else int(storage)

    def close(self) -> None:
        """``Buffer`` 协议的形状（幂等、什么都不做）；所有权不在本对象手上。"""

    def __enter__(self) -> "BorrowedBuffer":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class Allocator:
    """按档位复用的设备内存池。

    Attributes:

    Attributes:
        retained: 当前留在池里的空闲字节数。
    """

    def __init__(self, rt: Cudart, retain_bytes: int | None = None) -> None:
        """建一个池。

        Args:
            rt: 运行时句柄。
            retain_bytes: 空闲块保留上限；``None`` 取 :data:`_DEFAULT_RETAIN`。
        """
        self._rt = rt
        self._free: dict[int, list[int]] = {}
        self._retain = _DEFAULT_RETAIN if retain_bytes is None else int(retain_bytes)
        self.retained = 0
        self.live = 0
        self.peak = 0
        self.foreign_frees = 0

    def malloc(self, nbytes: int) -> Buffer:
        """取一块至少 ``nbytes`` 的设备内存。

        Args:
            nbytes: 请求字节数，必须为正。

        Returns:
            新的所有权凭证。

        Raises:
            RtError: 显存不足。
        """
        if nbytes <= 0:
            raise RtError(f"Allocator.malloc({nbytes}): size must be positive")
        size = _bucket(nbytes)
        spare = self._free.get(size)
        if spare:
            ptr = spare.pop()
            self.retained -= size
        else:
            ptr = self._rt.malloc(size)
        self.live += size
        self.peak = max(self.peak, self.live + self.retained)
        return Buffer(ptr, size, nbytes, self)

    def free(self, buffer: Buffer) -> None:
        """回收一块由本池发出的内存。

        Args:
            buffer: 待回收的凭证。不属于本池的地址会被计数后丢弃，绝不 ``cudaFree``。
        """
        if buffer._allocator is not self:          # noqa: SLF001 - 所有权检查
            self.foreign_frees += 1
            return
        size = buffer.nbytes
        buffer._allocator = None                  # noqa: SLF001 - 防止再次回收
        self.live -= size
        if self.retained + size > self._retain:
            self._rt.free(buffer.ptr)
            return
        self._free.setdefault(size, []).append(buffer.ptr)
        self.retained += size

    def trim(self) -> None:
        """把所有空闲块还给驱动。OOM 排查与测试收尾用。"""
        for size, ptrs in self._free.items():
            for ptr in ptrs:
                self._rt.free(ptr)
        self._free.clear()
        self.retained = 0
        self.live = 0

    def stats(self) -> str:
        """返回一行诊断：档位分布、在用与保留量、峰值、外来释放次数。"""
        buckets = ", ".join(f"{size >> 10}K*{len(p)}" for size, p in sorted(self._free.items())
                            if p) or "empty"
        return (f"pool live={self.live >> 20} MiB retained={self.retained >> 20} MiB "
                f"peak={self.peak >> 20} MiB buckets=[{buckets}] "
                f"foreign_frees={self.foreign_frees}")


class HostBuffer:
    """一块 pinned 宿主内存，按 numpy 视图读写。

    用于 H2D/D2H 的暂存：普通宿主内存也能拷，但带宽只有 pinned 的三分之一到一半
    （本平台实测 PCIe 13-14 GB/s 是 pinned 的数字），权重加载会直接差出几十秒。
    """

    __slots__ = ("ptr", "nbytes", "_rt", "_closed")

    def __init__(self, rt: Cudart, nbytes: int) -> None:
        if nbytes <= 0:
            raise RtError(f"HostBuffer({nbytes}): size must be positive")
        self._rt = rt
        self.ptr = rt.malloc_host(nbytes)
        self.nbytes = nbytes
        self._closed = False

    def view(self, dtype: DType, count: int) -> np.ndarray:
        """把这块内存按 ``dtype`` 观察成 numpy 数组。

        Args:
            dtype: 观察类型。
            count: 元素个数，``count * dtype.itemsize`` 不得超过 :attr:`nbytes`。

        Returns:
            ``count`` 个元素的数组视图，写入即写入这块 pinned 内存。

        Raises:
            RtError: 尺寸超出。
        """
        need = count * dtype.itemsize
        if need > self.nbytes:
            raise RtError(f"host view needs {need} B, buffer holds {self.nbytes} B")
        # 假驱动（rt/mock.py）的地址不是真地址，``from_address`` 在那里会解引用到
        # 未映射内存并段错误。给它一条出口，真实运行时没有这个方法。
        hook = getattr(self._rt, "host_view", None)
        if hook is not None:
            return hook(self.ptr, dtype, count)
        block = (ctypes.c_char * need).from_address(self.ptr)
        return np.frombuffer(block, dtype=dtype.numpy, count=count)

    def close(self) -> None:
        """还给驱动。幂等。"""
        if self._closed:
            return
        self._closed = True
        self._rt.free_host(self.ptr)
        self.ptr = 0

    def __del__(self) -> None:                    # pragma: no cover - 解释器退出路径
        try:
            self.close()
        except Exception:
            pass
