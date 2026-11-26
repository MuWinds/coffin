"""CoreX 设备运行时：用 ctypes 直接调 ``libcudart.so``，取代 ``torch.cuda``。

Module: ``coffin/cxrt``。叶子模块，只 import ``ctypes``、``numpy``
与标准库，**不 import torch**。它是 ``rt/memory.py``、``rt/array.py``、
``rt/stream.py`` 之下唯一碰驱动的层。

为什么是 cudart 而不是驱动 API：CoreX 的 ``libcudart.so`` 与 CUDA 的 ABI 一致，
``cudaMalloc`` / ``cudaMemcpy`` / ``cudaStreamCreate`` 这些入口在 `/usr/local/corex/lib64`
里都有；而驱动 API 那条路要求自己从 fatbin 里取 cubin 再 ``cuModuleLoadData``，
在厂商工具链（``nvcc`` 是空壳，只有 clang 能用）上不成立。用 cudart，编译产物是一个
普通 ``.so``，Python 用 ``ctypes`` 装载，torch 的扩展加载器就彻底不在链路里。

Constraints:
    * 每个入口的 ``argtypes`` 必须写全。ctypes 默认把 Python ``int`` 按 ``c_int`` 传，
      指针在 64 位上会被截成 32 位——这是最典型的「看着能跑、偶发非法访问」。
    * 所有返回码统一走 ``_check``：驱动错误的字符串自身不指出是哪次调用，所以消息里
      必须带入口名与尺寸。
    * ``stream == 0`` 表示默认流。这里不提供「当前流」这个隐式全局状态：torch 的
      ``getCurrentCUDAStream`` 在 ``rt`` 里没有对应物，流一律由调用方显式持有并透传。
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from typing import Any

from cxrt.errors import RtError

#: ``cudaMemcpy`` 的方向常量。抄自 CUDA 运行时 ABI，CoreX 沿用同一套值。
_MEMCPY_H2H = 0
_MEMCPY_H2D = 1
_MEMCPY_D2H = 2
_MEMCPY_D2D = 3

#: 候选库里最先找到的那个被采用。环境变量优先，因为同一台机器上可能装着两套 SDK。
_LIB_ENV = "COFFIN_COREX_RT_LIB"
_ROOT_ENV = "COREX_ROOT"


def _candidate_libs() -> list[str]:
    """按优先级列出 ``libcudart.so`` 的候选路径。

    Returns:
        绝对路径列表，最后一项是 ``find_library`` 的结果（可能为 ``None``，会被过滤）。
    """
    out: list[str] = []
    explicit = os.environ.get(_LIB_ENV, "").strip()
    if explicit:
        out.append(explicit)
    root = os.environ.get(_ROOT_ENV, "").strip() or "/usr/local/corex"
    out.append(os.path.join(root, "lib64", "libcudart.so"))
    found = ctypes.util.find_library("cudart")
    if found:
        out.append(found)
    return out


class Cudart:
    """``libcudart`` 的最小绑定：分配、拷贝、流、事件、设备查询。

    实例化即 ``dlopen``，但**不** ``cudaSetDevice``：设备选择是调用方的事，
    在 import 期动设备状态会让「只是枚举后端」变成一次真实初始化。
    """

    def __init__(self, path: str | None = None) -> None:
        """装载运行库并绑定入口。

        Args:
            path: 显式库路径；``None`` 时按 :func:`_candidate_libs` 的顺序找。

        Raises:
            RtError: 一个候选都没装载成功。消息里带上所有试过的路径，因为集群上
                「库不在默认位置」和「库存在但缺符号」看起来一样。
        """
        errors: list[str] = []
        lib = None
        for candidate in ([path] if path else _candidate_libs()):
            if not candidate:
                continue
            try:
                lib = ctypes.CDLL(candidate)
                break
            except OSError as exc:                # pragma: no cover - 需要真库
                errors.append(f"{candidate}: {exc}")
        if lib is None:
            raise RtError("cannot load libcudart: " + "; ".join(errors or ["no candidate"]))
        self._lib = lib
        self._path = candidate
        self._bind()

    # 绑定

    def _bind(self) -> None:
        """给每个入口写全 ``argtypes`` / ``restype``。

        Raises:
            RtError: 库里缺某个符号。缺符号比「返回错误码」更早暴露工具链装错了，
                所以这里直接失败而不是在第一次分配时才炸。
        """
        lib = self._lib
        p = ctypes.c_void_p
        sz = ctypes.c_size_t
        i = ctypes.c_int

        def fn(name: str, args: list[Any], res: Any = ctypes.c_int) -> Any:
            try:
                f = getattr(lib, name)
            except AttributeError as exc:         # pragma: no cover - 需要真库
                raise RtError(f"{self._path} has no symbol {name}") from exc
            f.argtypes = args
            f.restype = res
            return f

        self._cudaMalloc = fn("cudaMalloc", [ctypes.POINTER(p), sz])
        self._cudaFree = fn("cudaFree", [p])
        self._cudaMallocHost = fn("cudaMallocHost", [ctypes.POINTER(p), sz])
        self._cudaFreeHost = fn("cudaFreeHost", [p])
        self._cudaMemcpy = fn("cudaMemcpy", [p, p, sz, i])
        self._cudaMemcpyAsync = fn("cudaMemcpyAsync", [p, p, sz, i, p])
        self._cudaMemsetAsync = fn("cudaMemsetAsync", [p, i, sz, p])
        self._cudaStreamCreate = fn("cudaStreamCreate", [ctypes.POINTER(p)])
        self._cudaStreamDestroy = fn("cudaStreamDestroy", [p])
        self._cudaStreamSynchronize = fn("cudaStreamSynchronize", [p])
        self._cudaDeviceSynchronize = fn("cudaDeviceSynchronize", [])
        self._cudaEventCreate = fn("cudaEventCreate", [ctypes.POINTER(p)])
        self._cudaEventDestroy = fn("cudaEventDestroy", [p])
        self._cudaEventRecord = fn("cudaEventRecord", [p, p])
        self._cudaEventSynchronize = fn("cudaEventSynchronize", [p])
        # cudaEventElapsedTime 把毫秒写进第一个参数，所以它是 float*。
        self._cudaEventElapsedTime = fn("cudaEventElapsedTime",
                                        [ctypes.POINTER(ctypes.c_float), p, p])
        self._cudaGetLastError = fn("cudaGetLastError", [])
        self._cudaGetErrorString = fn("cudaGetErrorString", [i], ctypes.c_char_p)
        self._cudaSetDevice = fn("cudaSetDevice", [i])
        self._cudaGetDevice = fn("cudaGetDevice", [ctypes.POINTER(i)])
        self._cudaGetDeviceCount = fn("cudaGetDeviceCount", [ctypes.POINTER(i)])
        self._cudaMemGetInfo = fn("cudaMemGetInfo", [ctypes.POINTER(sz), ctypes.POINTER(sz)])

    # 错误

    def error_string(self, code: int) -> str:
        """把驱动返回码翻成人能读的字符串。

        Args:
            code: ``cudaError`` 值，必须非零。

        Returns:
            ``cudaGetErrorString`` 给出的描述；取不到时返回码本身。
        """
        raw = self._cudaGetErrorString(code)
        return raw.decode("utf-8", "replace") if raw else f"cudaError {code}"

    def _check(self, code: int, what: str) -> None:
        """非零返回码抛 :class:`RtError`，消息带入口名。

        Args:
            code: 驱动返回码。
            what: 出错的入口与关键参数，例如 ``cudaMalloc(1048576)``。

        Raises:
            RtError: ``code`` 非零。
        """
        if code:
            self._cudaGetLastError()             # 清掉粘滞错误，免得下次误报
            raise RtError(f"{what} -> {code} ({self.error_string(code)})")

    # 设备

    def device_count(self) -> int:
        """返回可见设备数。

        Returns:
            设备个数；驱动不可用时抛 :class:`RtError`。
        """
        count = ctypes.c_int(0)
        self._check(self._cudaGetDeviceCount(ctypes.byref(count)), "cudaGetDeviceCount")
        return count.value

    def set_device(self, index: int) -> None:
        """选择后续调用落在那张卡上。

        Args:
            index: 设备序号。

        Raises:
            RtError: 序号越界或驱动拒绝。
        """
        self._check(self._cudaSetDevice(index), f"cudaSetDevice({index})")

    def current_device(self) -> int:
        """返回当前设备序号。

        Returns:
            设备序号。
        """
        value = ctypes.c_int(0)
        self._check(self._cudaGetDevice(ctypes.byref(value)), "cudaGetDevice")
        return value.value

    def mem_info(self) -> tuple[int, int]:
        """返回 ``(free, total)`` 字节。

        Returns:
            空闲与总显存。启动横幅与 OOM 排查只看这两个数。
        """
        free = ctypes.c_size_t(0)
        total = ctypes.c_size_t(0)
        self._check(self._cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)),
                    "cudaMemGetInfo")
        return int(free.value), int(total.value)

    def device_name(self) -> str:
        """返回板卡名，优先问驱动 API，缺库时退回空串。

        名字是后端探测唯一的厂商判据（``cuda`` 这个命名空间由 ROCm、CoreX 和 NVIDIA
        共处），所以它必须真来自驱动，不能由常量兜底。

        Returns:
            ``"Iluvatar BI-V100"`` 这类字符串；``libcuda.so.1`` 不可用时为 ``""``。
        """
        driver = _DriverApi.open()
        if driver is None:
            return ""
        return driver.name(self.current_device())

    # 内存

    def malloc(self, nbytes: int) -> int:
        """分配设备内存。

        Args:
            nbytes: 字节数，必须为正。

        Returns:
            设备地址，作为整数返回（``Array`` 用它算偏移）。

        Raises:
            RtError: 显存不足或尺寸非法。
        """
        if nbytes <= 0:
            raise RtError(f"cudaMalloc({nbytes}): size must be positive")
        ptr = ctypes.c_void_p(0)
        self._check(self._cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes)),
                    f"cudaMalloc({nbytes})")
        return int(ptr.value or 0)

    def free(self, ptr: int) -> None:
        """释放设备内存。

        Args:
            ptr: :meth:`malloc` 返回的地址。``0`` 直接忽略，与 ``cudaFree(NULL)``
                的语义一致。

        Raises:
            RtError: 驱动拒绝（通常是重复释放或指针来自别的分配器）。
        """
        if not ptr:
            return
        self._check(self._cudaFree(ctypes.c_void_p(ptr)), f"cudaFree({ptr:#x})")

    def malloc_host(self, nbytes: int) -> int:
        """分配 pinned 宿主内存，供 H2D/D2H 走满带宽。

        Args:
            nbytes: 字节数。

        Returns:
            宿主地址。

        Raises:
            RtError: 页锁定内存不足。
        """
        ptr = ctypes.c_void_p(0)
        self._check(self._cudaMallocHost(ctypes.byref(ptr), ctypes.c_size_t(nbytes)),
                    f"cudaMallocHost({nbytes})")
        return int(ptr.value or 0)

    def free_host(self, ptr: int) -> None:
        """释放 pinned 宿主内存。

        Args:
            ptr: :meth:`malloc_host` 返回的地址。
        """
        if not ptr:
            return
        self._check(self._cudaFreeHost(ctypes.c_void_p(ptr)), f"cudaFreeHost({ptr:#x})")

    # 拷贝

    def _memcpy(self, dst: int, src: int, nbytes: int, kind: int, stream: int | None,
                what: str) -> None:
        """同步或异步地搬 ``nbytes`` 字节，按 ``stream`` 决定走哪条入口。

        Args:
            dst: 目的地址。
            src: 源地址。
            nbytes: 字节数。
            kind: ``cudaMemcpyKind``。
            stream: 流句柄；``None``/``0`` 表示默认流，走同步入口。
            what: 错误消息里的入口描述。
        """
        if not nbytes:
            return
        if stream:
            self._check(self._cudaMemcpyAsync(ctypes.c_void_p(dst), ctypes.c_void_p(src),
                                             ctypes.c_size_t(nbytes), kind,
                                             ctypes.c_void_p(stream)),
                        f"{what}({nbytes}B, async)")
        else:
            self._check(self._cudaMemcpy(ctypes.c_void_p(dst), ctypes.c_void_p(src),
                                         ctypes.c_size_t(nbytes), kind),
                        f"{what}({nbytes}B)")

    def copy_h2d(self, dst: int, src: int, nbytes: int, stream: int | None = None) -> None:
        """宿主到设备。``src`` 是 pinned 或普通宿主地址（普通地址只能同步拷贝）。"""
        self._memcpy(dst, src, nbytes, _MEMCPY_H2D, stream, "cudaMemcpyHostToDevice")

    def copy_d2h(self, dst: int, src: int, nbytes: int, stream: int | None = None) -> None:
        """设备到宿主。``dst`` 是 pinned 或普通宿主地址。"""
        self._memcpy(dst, src, nbytes, _MEMCPY_D2H, stream, "cudaMemcpyDeviceToHost")

    def copy_d2d(self, dst: int, src: int, nbytes: int, stream: int | None = None) -> None:
        """设备到设备。"""
        self._memcpy(dst, src, nbytes, _MEMCPY_D2D, stream, "cudaMemcpyDeviceToDevice")

    def memset(self, ptr: int, value: int, nbytes: int, stream: int | None = None) -> None:
        """把 ``nbytes`` 字节填成 ``value`` 的低 8 位。

        Args:
            ptr: 目的地址。
            value: 填充值，取低 8 位。
            nbytes: 字节数。
            stream: 流句柄。
        """
        if not nbytes:
            return
        self._check(self._cudaMemsetAsync(ctypes.c_void_p(ptr), ctypes.c_int(value & 0xFF),
                                         ctypes.c_size_t(nbytes),
                                         ctypes.c_void_p(stream or 0)),
                    f"cudaMemsetAsync({nbytes}B)")

    # 流与事件

    def stream_create(self) -> int:
        """建一条非默认流。

        Returns:
            流句柄。

        Raises:
            RtError: 驱动拒绝建流（通常是上下文已损坏）。
        """
        handle = ctypes.c_void_p(0)
        self._check(self._cudaStreamCreate(ctypes.byref(handle)), "cudaStreamCreate")
        return int(handle.value or 0)

    def stream_destroy(self, handle: int) -> None:
        """销毁流。销毁前不隐式同步，驱动语义如此，调用方自己决定。"""
        if not handle:
            return
        self._check(self._cudaStreamDestroy(ctypes.c_void_p(handle)),
                    f"cudaStreamDestroy({handle:#x})")

    def stream_sync(self, handle: int) -> None:
        """等这条流上的工作全部完成。"""
        self._check(self._cudaStreamSynchronize(ctypes.c_void_p(handle or 0)),
                    f"cudaStreamSynchronize({handle:#x})")

    def device_sync(self) -> None:
        """等整个设备完成。"""
        self._check(self._cudaDeviceSynchronize(), "cudaDeviceSynchronize")

    def event_create(self) -> int:
        """建一个计时事件。

        Returns:
            事件句柄。
        """
        handle = ctypes.c_void_p(0)
        self._check(self._cudaEventCreate(ctypes.byref(handle)), "cudaEventCreate")
        return int(handle.value or 0)

    def event_destroy(self, handle: int) -> None:
        """销毁事件。"""
        if not handle:
            return
        self._check(self._cudaEventDestroy(ctypes.c_void_p(handle)),
                    f"cudaEventDestroy({handle:#x})")

    def event_record(self, handle: int, stream: int | None = None) -> None:
        """在流上打点。"""
        self._check(self._cudaEventRecord(ctypes.c_void_p(handle),
                                          ctypes.c_void_p(stream or 0)),
                    f"cudaEventRecord({handle:#x})")

    def event_sync(self, handle: int) -> None:
        """等事件到达。"""
        self._check(self._cudaEventSynchronize(ctypes.c_void_p(handle)),
                    f"cudaEventSynchronize({handle:#x})")

    def event_elapsed_ms(self, start: int, end: int) -> float:
        """返回两个事件之间的毫秒数。

        Args:
            start: 先打点的事件。
            end: 后打点的事件。

        Returns:
            毫秒，``float``。设备事件的分辨率约 0.5 us，量 20 us 级内核够用。
        """
        ms = ctypes.c_float(0.0)
        self._check(self._cudaEventElapsedTime(ctypes.byref(ms), ctypes.c_void_p(start),
                                             ctypes.c_void_p(end)),
                    "cudaEventElapsedTime")
        return float(ms.value)


class _DriverApi:
    """``libcuda.so.1`` 上只需三个入口：``cuInit`` / ``cuDeviceGet`` / ``cuDeviceGetName``。

    只为了拿板卡名。名字是后端探测的厂商判据，而 cudart 的 ``cudaGetDeviceProperties``
    要摆一个大结构体，字段布局在 CoreX 上不一定与 CUDA 文档一致——读错一个偏移就是
    读到垃圾名。驱动 API 的 ``cuDeviceGetName`` 只有一个 ``char*``，不会读错。
    """

    _instance: "_DriverApi | None" = None
    _failed = False

    def __init__(self, lib: Any) -> None:
        self._lib = lib
        self._cuInit = lib.cuInit
        self._cuInit.argtypes = [ctypes.c_uint]
        self._cuInit.restype = ctypes.c_int
        self._cuDeviceGet = lib.cuDeviceGet
        self._cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self._cuDeviceGet.restype = ctypes.c_int
        self._cuDeviceGetName = lib.cuDeviceGetName
        self._cuDeviceGetName.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        self._cuDeviceGetName.restype = ctypes.c_int

    @classmethod
    def open(cls) -> "_DriverApi | None":
        """装载驱动库，失败时返回 ``None`` 并记住这次失败。

        Returns:
            可用的实例，或 ``None``（缺失 ``libcuda.so.1``，例如纯 CPU 主机）。
        """
        if cls._instance is not None or cls._failed:
            return cls._instance
        candidates = []
        explicit = os.environ.get("COFFIN_COREX_RT_DRIVER", "").strip()
        if explicit:
            candidates.append(explicit)
        root = os.environ.get(_ROOT_ENV, "").strip() or "/usr/local/corex"
        candidates.append(os.path.join(root, "lib64", "libcuda.so.1"))
        candidates.append("libcuda.so.1")
        for candidate in candidates:
            try:
                lib = ctypes.CDLL(candidate)
            except OSError:
                continue
            try:
                api = cls(lib)
            except AttributeError:
                continue
            api._cuInit(0)
            cls._instance = api
            return api
        cls._failed = True
        return None

    def name(self, index: int) -> str:
        """返回第 ``index`` 张卡的板卡名。

        Args:
            index: 设备序号。

        Returns:
            名字；驱动拒绝时为空串。空串是「说不清」，不能当成「不是本厂商」之外的
            任何含义。
        """
        device = ctypes.c_int(0)
        if self._cuDeviceGet(ctypes.byref(device), index):
            return ""
        buf = ctypes.create_string_buffer(256)
        if self._cuDeviceGetName(buf, 256, device.value):
            return ""
        return buf.value.decode("utf-8", "replace")
