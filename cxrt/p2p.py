"""四卡设备内存 all-to-all 归约的 C ABI 装载面。

Module: ``coffin/cxrt``。只 import ``ctypes`` 与 ``cxrt.*``，**不 import torch**。
供 TP4 与 EP4 共同使用：IPC 句柄取/开/关，以及奇偶双缓冲 bf16 归约核的发射。

Constraints:
    * 签名必须与 ``backends/corex/kernels/abi/p2p_reduce_abi.cu`` 逐字一致。
    * 固定 4 卡（``world == 4``），且 ``blocks <= 16`` 保证 BI-V100 全网格常驻。
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from cxrt.shim import Shim

#: ``cudaIpcMemHandle_t`` 的固定字节数。
IPC_HANDLE_BYTES = 64


class P2pShim:
    """四卡 P2P 归约四个 C ABI 出口的薄转发层。"""

    def __init__(self, shim: Shim) -> None:
        """绑定四个出口。

        Args:
            shim: 已装载的 ``libcoffin_p2p.so``。
        """
        p = ctypes.c_void_p
        i = ctypes.c_int
        ll = ctypes.c_longlong
        u64 = ctypes.c_uint64
        self._alloc = shim.export("coffin_p2p_alloc", [u64, ctypes.POINTER(u64)])
        self._free = shim.export("coffin_p2p_free", [u64])
        self._read_abort = shim.export("coffin_p2p_read_abort", [u64, ctypes.POINTER(ll)])
        self._get_handle = shim.export("coffin_p2p_get_handle", [p, p])
        self._open_handle = shim.export("coffin_p2p_open_handle", [p, ctypes.POINTER(u64)])
        self._close_handle = shim.export("coffin_p2p_close_handle", [u64])
        self.combine_bf16 = shim.export(
            "coffin_p2p_combine_bf16",
            [p, p, p, u64, u64, u64, u64, u64, u64, u64, u64, ll, ll, i, ll, i, i, p],
        )
        self.path = shim.path

    def alloc(self, nbytes: int) -> int:
        """用 ``cudaMalloc`` 分配并清零一块独立设备缓冲，返回段基址。"""
        out = ctypes.c_uint64(0)
        self._alloc(int(nbytes), ctypes.byref(out))
        return int(out.value)

    def free(self, dev_ptr: int) -> None:
        """释放 :meth:`alloc` 分配的独立设备缓冲。"""
        if dev_ptr:
            self._free(int(dev_ptr))

    def read_abort(self, flag_ptr: int) -> int:
        """读回本卡 ``flags`` 里的中止锁存轮次（0 表示正常）。"""
        out = ctypes.c_longlong(0)
        self._read_abort(int(flag_ptr), ctypes.byref(out))
        return int(out.value)

    def get_handle(self, dev_ptr: int) -> bytes:
        """取一块设备缓冲的 64 字节 IPC 句柄。

        Args:
            dev_ptr: 设备内存首址。

        Returns:
            长度为 64 的原始句柄字节串。
        """
        buf = (ctypes.c_ubyte * IPC_HANDLE_BYTES)()
        self._get_handle(dev_ptr, ctypes.cast(buf, ctypes.c_void_p))
        return bytes(buf)

    def open_handle(self, handle_bytes: bytes) -> int:
        """打开对端 64 字节 IPC 句柄，返回本进程可寻址的设备指针。

        Args:
            handle_bytes: 对端 :meth:`get_handle` 产出的 64 字节串。

        Returns:
            对端设备内存的整数地址。

        Raises:
            ValueError: 句柄长度不是 64 字节。
        """
        if len(handle_bytes) != IPC_HANDLE_BYTES:
            raise ValueError(f"IPC handle must be {IPC_HANDLE_BYTES} bytes, got {len(handle_bytes)}")
        buf = (ctypes.c_ubyte * IPC_HANDLE_BYTES).from_buffer_copy(handle_bytes)
        out = ctypes.c_uint64(0)
        self._open_handle(ctypes.cast(buf, ctypes.c_void_p), ctypes.byref(out))
        return int(out.value)

    def close_handle(self, peer_ptr: int) -> None:
        """关闭已打开的对端 IPC 指针。"""
        if peer_ptr:
            self._close_handle(int(peer_ptr))


def load_p2p_shim(lib_path: str | Path) -> P2pShim:
    """装载 P2P 归约 shim。

    Args:
        lib_path: ``cxrt.build`` 产出的 ``.so`` 路径。

    Returns:
        绑好出口的 :class:`P2pShim`。
    """
    return P2pShim(Shim(lib_path))
