"""C ABI 装载：``ctypes`` 直接调 shim ``.so``，取代 pybind11 与 ``TORCH_CHECK``。

Module: ``coffin/cxrt``。只 import ``ctypes`` 与标准库，**不 import
torch**。

约定（两侧必须逐字一致，``kernels/abi/*.cu`` 是另一半）：

* 每个出口返回 ``int``：``0`` 成功，非零失败。
* 指针一律 ``void*``（``ctypes.c_void_p``），整数用 ``c_int`` / ``c_long``，
  浮点用 ``c_float``。不用 ``size_t``：它在不同平台宽度不同，跨语言传它是自找麻烦。
* 出口的最后两个参数固定是 ``char* err, int errlen``。失败时把原因写进去，Python
  侧把整段消息抬成 :class:`~coffin.cxrt.errors.RtError`。

为什么不用 pybind11：pybind 本身虽然不带 torch，但它的类型转换器是按 torch 张量写的；
自己写一套类型转换器等于重造 ABI，还不如直接暴露扁平指针参数——内核要的本来就只有
指针、形状和间距。

Constraints:
    * 每个出口的 ``argtypes`` 必须写全，且必须与 ``.cu`` 里的签名逐字对齐。``ctypes``
      少写一个参数不会报错，只会把后面所有参数整体错位一格，症状是内核读到垃圾。
    * 这里只做「调用 + 错误码翻译」，不做任何形状推导。形状属于算子语义，在
      ``rt/norm.py`` 那一层判，那里能给出带数组描述的错误消息。
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

from cxrt.errors import RtError

#: 错误缓冲长度。够装下一个文件名加一句描述，又不至于每次调用都分配大对象。
ERR_LEN = 512


class Shim:
    """一个 shim 共享库，以及它导出的入口。

    Attributes:
        path: ``.so`` 路径。
    """

    def __init__(self, path: str | Path) -> None:
        """装载共享库。

        Args:
            path: ``.so`` 路径。

        Raises:
            RtError: 装载失败（缺依赖、架构不符、符号缺失）。
        """
        self.path = Path(path)
        try:
            self._lib = ctypes.CDLL(str(self.path))
        except OSError as exc:                    # pragma: no cover - 需要真库
            raise RtError(f"cannot load shim {self.path}: {exc}") from exc

    def export(self, name: str, argtypes: list[Any]) -> Any:
        """取一个出口并包成「自动查错误码」的可调用对象。

        Args:
            name: 出口名，例如 ``coffin_rms_norm``。
            argtypes: 除错误缓冲之外的参数类型，顺序与 ``.cu`` 一致。

        Returns:
            可调用对象：按 ``argtypes`` 传参，成功返回 ``None``，失败抛
            :class:`RtError`。

        Raises:
            RtError: 库里没有这个符号。缺符号通常意味着编的是另一份源码，早点报出来
                比在内核里读到垃圾好。
        """
        try:
            fn = getattr(self._lib, name)
        except AttributeError as exc:             # pragma: no cover - 需要真库
            raise RtError(f"{self.path.name} has no export {name}") from exc
        fn.argtypes = argtypes + [ctypes.c_char_p, ctypes.c_int]
        fn.restype = ctypes.c_int
        err = ctypes.create_string_buffer(ERR_LEN)

        def call(*args: Any) -> None:
            code = fn(*args, err, ERR_LEN)
            if code:
                message = err.value.decode("utf-8", "replace") or f"error code {code}"
                raise RtError(f"{name}: {message}")

        call.__name__ = name
        return call

    def export_code(self, name: str, argtypes: list[Any]) -> Any:
        """取一个出口，包成「返回错误码」的可调用对象。

        给**可失败**的加速路径用：形状不合时调用方要回退，而不是抛异常。语义与
        :meth:`export` 只差这一点——成功都返回 ``0``。

        Args:
            name: 出口名，例如 ``coffin_kv_pool_store``。
            argtypes: 除错误缓冲之外的参数类型，顺序与 ``.cu`` 一致。

        Returns:
            可调用对象：按 ``argtypes`` 传参，返回 ``0`` 或非零错误码。

        Raises:
            RtError: 库里没有这个符号。
        """
        try:
            fn = getattr(self._lib, name)
        except AttributeError as exc:             # pragma: no cover - 需要真库
            raise RtError(f"{self.path.name} has no export {name}") from exc
        fn.argtypes = argtypes + [ctypes.c_char_p, ctypes.c_int]
        fn.restype = ctypes.c_int
        err = ctypes.create_string_buffer(ERR_LEN)

        def call(*args: Any) -> int:
            return int(fn(*args, err, ERR_LEN))

        call.__name__ = name
        return call


class NormShim:
    """归一化族五个出口的 C ABI 面。

    这是一个极薄的转发层：参数名与顺序就是 ``kernels/abi/norm_abi.cu`` 的签名。
    ``cxrt/norm.py`` 负责形状与 dtype 校验，这里只保证「按 ABI 传参」。
    """

    def __init__(self, shim: Shim) -> None:
        """绑定五个出口。

        Args:
            shim: 已装载的共享库。
        """
        p = ctypes.c_void_p
        i = ctypes.c_int
        l = ctypes.c_long
        f = ctypes.c_float
        self.rms_norm = shim.export(
            "coffin_rms_norm", [p, p, p, i, i, l, f, i, p])
        self.add_rms_norm = shim.export(
            "coffin_add_rms_norm", [p, p, p, p, i, i, l, f, i, p])
        self.rms_norm_gated = shim.export(
            "coffin_rms_norm_gated", [p, p, p, p, i, i, f, i, p])
        self.rms_norm_i8 = shim.export(
            "coffin_rms_norm_i8", [p, p, p, p, i, i, l, f, i, i, p])
        self.add_rms_norm_i8 = shim.export(
            "coffin_add_rms_norm_i8", [p, p, p, p, p, i, i, l, f, i, i, p])
        self.path = shim.path


def load_norm_shim(lib_path: str | Path) -> NormShim:
    """装载 norm shim。

    Args:
        lib_path: ``cxrt.build`` 产出的 ``.so`` 路径。

    Returns:
        绑好五个出口的 :class:`NormShim`。
    """
    return NormShim(Shim(lib_path))
