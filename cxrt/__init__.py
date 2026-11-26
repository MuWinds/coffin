"""``rt``：corex 推理链上不依赖 torch 的设备运行时。

这一层的目的很直接：用天数智芯的卡做推理时，整条链路不 import torch。它只依赖
``ctypes``、``numpy`` 与标准库，把原来分散在 ``torch.cuda`` / ``torch.utils.
cpp_extension`` / ``torch.Tensor`` 上的四件事各接一个替代物：

===============  ============================================  ==========================
原来            现在                                         模块
===============  ============================================  ==========================
``torch.cuda``   ``Cudart``：ctypes 直调 ``libcudart.so``      ``rt/driver.py``
流与事件          ``Stream`` / ``Event``：显式句柄               ``rt/stream.py``
分配器            ``Allocator``：按尺寸分档的池 + pinned 暂存     ``rt/memory.py``
``torch.Tensor``  ``Array``：指针 + 形状 + 步长 + dtype         ``rt/array.py``
``cpp_extension`` ``build``：厂商 clang 直接编 ``.so``          ``rt/build.py``
``pybind11``      ``Shim``：ctypes + 返回码 ABI                 ``rt/shim.py``
===============  ============================================  ==========================

用法（卡上）::

    import cxrt as rt

    runtime = rt.open_runtime()                 # 装载驱动、建流与池
    norm = runtime.build_norm()                 # 编（或复用）norm shim 并绑定
    y = norm.rms_norm(x, gain, eps=1e-6)        # x/gain 都是 rt.Array
    runtime.close()

主机上（无卡、无 torch、无工具链）也能跑一半：``rt.mock`` 提供假驱动与 numpy 版
内核，``tests/test_corex_rt.py`` 就是靠它在本地验证形状、视图与数据流。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cxrt.array import Array
from cxrt.build import build, cache_key, corex_root
from cxrt.driver import Cudart
from cxrt.dtypes import BF16, BY_CODE, BY_NAME, F32, I8, I32, I64, U8, U32, DType
from cxrt.errors import BuildError, RtError, ShapeError
from cxrt.memory import Allocator, Buffer, HostBuffer
from cxrt.attn import AttnOps, AttnShim, load_attn_shim
from cxrt.conv import ConvOps, ConvShim, load_conv_shim
from cxrt.gemv import GemvOps, GemvShim, load_gemv_shim
from cxrt.linalg import Cublas, GemmOps
from cxrt.moe import MoeOps, MoeShim, load_moe_shim
from cxrt.p2p import P2pShim, load_p2p_shim
from cxrt.solve import SolveOps, SolveShim, load_solve_shim
from cxrt.w8a8 import W8A8Ops, W8A8Shim, load_w8a8_shim
from cxrt.norm import RMSNormOps
from cxrt.shim import NormShim, Shim, load_norm_shim
from cxrt.stream import Event, Stream

__all__ = [
    "Array", "Allocator", "AttnOps", "AttnShim", "BF16", "BY_CODE", "BY_NAME", "Buffer",
    "ConvOps", "ConvShim", "Cublas",
    "BuildError", "Cudart", "DType", "Event", "F32", "GemmOps", "GemvOps", "GemvShim",
    "HostBuffer",
    "I8", "I32", "I64", "MoeOps", "MoeShim", "NormShim", "P2pShim", "RMSNormOps", "RtError", "Runtime",
    "ShapeError",
    "Shim", "SolveOps", "SolveShim", "Stream", "U8", "U32", "W8A8Ops", "W8A8Shim",
    "build", "build_gemm", "cache_key", "corex_root", "kernels_dir", "load_attn_shim", "load_conv_shim", "load_gemv_shim",
    "load_moe_shim", "load_norm_shim", "load_p2p_shim", "load_solve_shim", "load_w8a8_shim", "open_runtime",
]

#: 默认目标 ISA。CoreX 的 ``clang`` 用 ``--cuda-gpu-arch=ivcore10``。
DEFAULT_ARCH = "ivcore10"


def kernels_dir() -> Path:
    """返回内核源码目录。

    Returns:
        ``coffin/backends/corex/kernels``，``abi/`` 与 ``norm_device.inc`` 都在它下面。
        内核源码留在 corex 后端目录里（它属于那个后端的资产），运行时只是按路径去编它。
    """
    return Path(__file__).resolve().parent.parent / "backends" / "corex" / "kernels"


@dataclass
class Runtime:
    """一台设备的运行时上下文：驱动、分配器、流、设备序号。

    Attributes:
        cudart: 驱动绑定。
        alloc: 设备内存池。
        stream: 计算流（非默认流，这样与任何遗留的默认流工作解耦）。
        index: 设备序号。
    """

    cudart: Cudart
    alloc: Allocator
    stream: Stream
    index: int

    def build_norm(self, arch: str = DEFAULT_ARCH) -> RMSNormOps:
        """编（或复用）``norm_abi.cu`` 并绑成算子面。

        Args:
            arch: 目标 ISA。

        Returns:
            可以直接调 ``rms_norm`` / ``add_rms_norm`` / ``rms_norm_gated`` 的对象。

        Raises:
            BuildError: 工具链缺失或编译失败。
        """
        root = kernels_dir()
        lib = build(
            sources=[root / "abi" / "norm_abi.cu"],
            includes=[root],
            arch=arch,
            name="coffin_norm",
        )
        return RMSNormOps(load_norm_shim(lib), self.cudart, self.alloc)

    def build_gemv(self, arch: str = DEFAULT_ARCH) -> GemvOps:
        """编（或复用）``gemv_abi.cu`` 并绑成投影族算子面。

        Args:
            arch: 目标 ISA。

        Returns:
            :class:`GemvOps`：``nk_i8`` / ``nk_i8_m``（nk 权重，`[N, K]`），参数都是
            张量（``Array`` 或任何暴露 ``data_ptr()`` 的对象），不是 torch 张量。

        Raises:
            BuildError: 工具链缺失或编译失败。
        """
        root = kernels_dir()
        lib = build(
            sources=[root / "abi" / "gemv_abi.cu"],
            includes=[root],
            arch=arch,
            name="coffin_gemv",
        )
        return GemvOps(load_gemv_shim(lib))

    def build_attn(self, arch: str = DEFAULT_ARCH) -> AttnOps:
        """编（或复用）``attn_abi.cu`` 并绑成注意力小算子面。

        Args:
            arch: 目标 ISA。

        Returns:
            :class:`AttnOps`：``silu_mul`` / ``silu_mul_quant`` / ``rope_apply``，加上
            ``kv_store`` / ``kv_pool_store`` 两个写缓存的出口（后者可失败）、
            ``attn_slice``（四个公开入口共用的切片+合并）、QSA 的 ``attn_sparse`` 与
            ``qsa_block_keys``。

        Raises:
            BuildError: 工具链缺失或编译失败。
        """
        root = kernels_dir()
        lib = build(
            sources=[root / "abi" / "attn_abi.cu", root / "abi" / "kv_store_abi.cu",
                     root / "abi" / "hyper_norm_abi.cu", root / "abi" / "delta_abi.cu",
                     root / "abi" / "delta_prologue_abi.cu",
                     root / "abi" / "attn_slice_abi.cu",
                     root / "abi" / "attn_sparse_abi.cu",
                     root / "abi" / "qsa_block_keys_abi.cu"],
            includes=[root],
            arch=arch,
            name="coffin_attn",
        )
        return AttnOps(load_attn_shim(lib))

    def build_moe(self, arch: str = DEFAULT_ARCH) -> MoeOps:
        """编（或复用）``abi/moe_epilog_abi.cu`` 并绑成 MoE 小算子面。

        Args:
            arch: 目标 ISA。

        Returns:
            :class:`MoeOps`：``shared_gate_scale`` 与 ``moe_router``；``moe_dispatch``
            只有出口（四个输出都是整数表，校验在出口里）。``moe_grouped_i8*`` 四条
            要连分组偏移一起定契约，还没搬。

        Raises:
            BuildError: 工具链缺失或编译失败。
        """
        root = kernels_dir()
        lib = build(
            sources=[root / "abi" / "moe_epilog_abi.cu", root / "abi" / "moe_router_abi.cu",
                     root / "abi" / "moe_dispatch_abi.cu",
                     root / "abi" / "moe_grouped_i8_expert_abi.cu",
                     root / "abi" / "moe_grouped_i8_abi.cu",
                     root / "abi" / "moe_i8_tcu_abi.cu",
                     root / "abi" / "moe_grouped_i8_merge_abi.cu"],
            includes=[root],
            arch=arch,
            name="coffin_moe",
        )
        return MoeOps(load_moe_shim(lib))

    def build_w8a8(self, arch: str = DEFAULT_ARCH) -> W8A8Ops:
        """编（或复用）``abi/w8a8_abi.cu`` + ``abi/w8a8_epilog_abi.cu`` 并绑成量化面。

        Args:
            arch: 目标 ISA。

        Returns:
            :class:`W8A8Ops`：``quant_act_i8`` / ``gather_quant_i8`` / ``epilogue``。
            这三条加上 :meth:`build_gemm` 的 ``i8_nk`` 就是完整的 W8A8 预填充：
            ``quant_act_i8``(写) → ``GemmEx``(int32 累加) → ``epilogue``(两个 scale + 一次
            窄化)，中间没有任何一步回到 torch。MoE 分组 GEMM 还没搬——它带 ``[E, N, K]``
            的分组偏移，是另一份契约。

        Raises:
            BuildError: 工具链缺失或编译失败。
        """
        root = kernels_dir()
        lib = build(
            sources=[root / "abi" / "w8a8_abi.cu", root / "abi" / "w8a8_epilog_abi.cu",
                     root / "abi" / "w8a8_dp4a_abi.cu"],
            includes=[root],
            arch=arch,
            name="coffin_w8a8",
        )
        return W8A8Ops(load_w8a8_shim(lib))

    def build_p2p(self, arch: str = DEFAULT_ARCH) -> P2pShim:
        """编（或复用）``abi/p2p_reduce_abi.cu`` 并绑成四卡 P2P 归约面。

        Args:
            arch: 目标 ISA。

        Returns:
            :class:`P2pShim`。

        Raises:
            BuildError: 工具链缺失或编译失败。
        """
        root = kernels_dir()
        lib = build(
            sources=[root / "abi" / "p2p_reduce_abi.cu"],
            includes=[root],
            arch=arch,
            name="coffin_p2p",
        )
        return load_p2p_shim(lib)

    def build_gemm(self, lib: str | None = None) -> GemmOps:
        """绑定预填充 GEMM 的库面，不编任何内核、不 import torch。

        走 ``libcublas``（装了 ixblas 的话就是 ixblas 的同名替换）：``i8_nk`` 是 W8A8
        的读方，``bf16_nk`` 是 ``--fp16`` 权重那一支。两条都按 nk 原缓冲取操作数
        （``lda = ldb = K``、``transa = T``），所以那份转置副本整条都不存在。

        Args:
            lib: 显式 ``libcublas.so`` 路径；``None`` 时按 ``COFFIN_CUBLAS`` 与常规位置找。

        Returns:
            :class:`GemmOps`。

        Raises:
            RtError: 找不到库，或建句柄被拒。
        """
        return GemmOps(Cublas(lib))

    def build_conv(self, arch: str = DEFAULT_ARCH) -> ConvOps:
        """编（或复用）``conv_abi.cu`` 并绑成卷积算子面。

        Args:
            arch: 目标 ISA。

        Returns:
            :class:`ConvOps`：``conv1d_silu`` / ``_seq`` / ``_batch``——qwen35 每条
            线性注意力层的入口。

        Raises:
            BuildError: 工具链缺失或编译失败。
        """
        root = kernels_dir()
        lib = build(
            sources=[root / "abi" / "conv_abi.cu"],
            includes=[root],
            arch=arch,
            name="coffin_conv",
        )
        return ConvOps(load_conv_shim(lib))

    def build_solve(self, arch: str = DEFAULT_ARCH) -> SolveOps:
        """编（或复用）``solve_abi.cu`` 并绑成求解算子面。

        Args:
            arch: 目标 ISA。

        Returns:
            :class:`SolveOps`：``triangular_solve``（单位下三角 ``A X = B``），delta
            规则每层每块都要走它。

        Raises:
            BuildError: 工具链缺失或编译失败。
        """
        root = kernels_dir()
        lib = build(
            sources=[root / "abi" / "solve_abi.cu"],
            includes=[root],
            arch=arch,
            name="coffin_solve",
        )
        return SolveOps(load_solve_shim(lib))

    def stats(self) -> str:
        """返回一行状态：板卡名、设备序号、池占用。"""
        return (f"device={self.cudart.device_name() or 'unknown'} index={self.index} "
                f"{self.alloc.stats()}")

    def close(self) -> None:
        """释放流并把池里的空闲块还给驱动。幂等。"""
        self.stream.close()
        self.alloc.trim()


def build_gemm(lib: str | None = None) -> GemmOps:
    """模块级快捷入口：绑定预填充 GEMM 的库面，不编任何内核、不 import torch。"""
    return GemmOps(Cublas(lib))


def open_runtime(index: int = 0, lib: str | None = None,
                 retain_bytes: int | None = None) -> Runtime:
    """打开一台设备。

    Args:
        index: 设备序号。
        lib: 显式 ``libcudart.so`` 路径；``None`` 时按 ``rt.driver`` 的顺序找。
        retain_bytes: 池的空闲保留上限；``None`` 取默认 1 GiB。

    Returns:
        运行时上下文。

    Raises:
        RtError: 驱动装载失败或设备序号越界。
    """
    cudart = Cudart(lib)
    count = cudart.device_count()
    if not 0 <= index < count:
        raise RtError(f"device {index} out of range: {count} visible")
    cudart.set_device(index)
    return Runtime(cudart=cudart, alloc=Allocator(cudart, retain_bytes),
                   stream=Stream.create(cudart), index=index)
