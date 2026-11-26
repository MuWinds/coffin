"""预填充 GEMM 的 C 面：``ctypes`` 直调 ``libcublas``（含 ixblas 同名替换），不经过 torch。

Module: ``coffin/cxrt``。只 import ``ctypes`` 与 ``cxrt.*``，**不 import torch**：
corex 的预填充投影不再有 ``torch.matmul`` / ``F.linear`` / ``torch._int_mm`` 三条出口，
张量只按「指针 + dtype 码 + 形状」参与。

两条出口，同一个布局：

* :meth:`GemmOps.i8_nk` —— int8 权重的 W8A8 预填充，``i8 x i8 -> i32``，累加器由调用方
  持有，scale 在 ``w8a8_epilogue`` 里乘。
* :meth:`GemmOps.bf16_nk` —— bf16 权重（``--fp16`` 在 corex 上落的就是这一支）的预填充，
  输出 bf16 或 fp32。

## 权重为什么一次都不用转

库的列主序视角下，``[N, K]`` 行主序权重**就是** ``[K, N]`` 列主序、``ldb = K`` 的那块
内存，所以 nk 原缓冲直接当操作数用。要凑出「行主序 ``[M, N]`` 输出」这个约定，cublas
的三个尺寸要交换：

    C[N, M] = op(A)[N, K] * B[K, M]        （列主序，``ldc = N``）

    A = 权重 ``[N, K]``，``lda = K``，``op(A) = T``
    B = 激活 ``[M, K]``，``ldb = K``，``op(B) = N``
    C 的内存按下标 ``n + m * N`` 铺，正好是行主序 ``[M, N]`` 的 ``m * N + n``

也就是 ``transa = T``、``transb = N``、``(m, n, k) = (N, M, K)``、``lda = ldb = K``、
``ldc = N``。这一组正是 ixblas 记的 **TN**：``K % 128 == 0`` 的锯齿配上 ``ld + 16`` 后
int8 到 41.8~46.2 TOPS、f16 到 18.9 TFLOP/s；反过来为了凑 NT 而转置权重，才是把
4.3 GB 的 kn 副本请回来的那条路。

## 为什么不是「自己写内核」

已删的自研稠密张量核（``w8a8_gemm_tcu``）在 4B 真实形状上对库是 45~60 TOPS 对 33~48，
``gate_up`` / ``full_attn`` / ``linear_attn`` / ``o_proj`` 赢 1.4~1.8 倍。所以路线是
两段：先由这里把 torch 摘掉（库的 TN 路径够用），再把自研 nk 内核搬进 ``abi/``
（``coffin_gemm_i8_nk``，吃同一块 ``[N, K]``，一次转置都不做）替掉赢的那几个形状。
两者共用同一个「权重是 ``[N, K]``」的契约，所以切换不影响上层。

Constraints:
    * **``computeType`` 的取值随 SDK 世代变**。CUDA 11 起是 ``CUBLAS_COMPUTE_*``
      （32I = 0、32F = 1、16F = 2），10.2 那代传的是 ``cudaDataType``（32I = 10、32F = 0、
      16F = 2）。传错的表现是 ``NOT_SUPPORTED(15)`` 或「跑完但数不对」，两种都很难从
      现象反推。默认走新表，``COFFIN_CUBLAS_COMPUTE=legacy`` 切旧表；第一次上卡必须
      用 ``tools/probe_gemm_nk.py --computes both`` 把两套各试一遍再定。
    * int8 的 ``alpha``/``beta`` 是**指向 int32 的指针**，且 ixblas 的 i8 接管要求
      ``alpha == 1``、``M % 4 == 0``（预填充量子是 8 的倍数，天然满足；尾部不足 4 行
      由 ``w8a8_gemm`` 的窄形状路径接走）。
    * 三个操作数都必须**连续**。``lm_head`` 那种 ``out[:, start:stop]`` 的列切片不能直接
      进来，先写到整块再切，或者等 ``ldc`` 参数化——现在不收 ``ldc`` 是有意的：收一个
      半吊子的间距参数，错的时候是静默写花别人的行。
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from pathlib import Path
from typing import Any

from cxrt import tensors as T
from cxrt.dtypes import BY_CODE
from cxrt.errors import RtError, ShapeError

#: ``cublasOperation_t``。
OP_N = 0
OP_T = 1

#: ``cudaDataType``：A/B/C 的类型码，跨世代稳定，可以照抄。
#: 注意 ``CUDA_R_8I = 3`` 而不是 ``8``——``8`` 是 ``CUDA_R_8U``。这个差值在卡上不报错，
#: 只在 ``cublasGemmEx`` 里回一个 ``not supported``，看起来像「库不支持 int8 的 TN」。
#: 上卡（2026-11-25）实测：把它改回 3 之后 int8 就能跑，改成 8 时三档 dtype 里唯独 int8
#: 全形状失败。
CUDA_R_32F = 0
CUDA_R_16F = 2
CUDA_R_8I = 3
CUDA_R_32I = 10
CUDA_R_16BF = 14

#: ``CUBLAS_COMPUTE_*``（CUDA 11 起）。
COMPUTE_32I = 0
COMPUTE_32F = 1
COMPUTE_16F = 2

#: 10.2 那代把 ``cudaDataType`` 当 computeType 传。
COMPUTE_LEGACY_32I = 10
COMPUTE_LEGACY_32F = 0

#: ``cublasStatus_t`` 到人话。只列真会碰到的几个。
STATUS = {
    0: "success",
    1: "not initialized",
    3: "alloc failed",
    7: "invalid value",
    8: "arch mismatch",
    11: "mapping error",
    13: "execution failed",
    14: "internal error",
    15: "not supported",
}

#: ``cublasGemmAlgo_t``：``-1`` 是 ``CUBLAS_GEMM_DEFAULT``，让库自己选。
ALGO_DEFAULT = -1


def _compute_code(name: str) -> int:
    """按世代取 computeType 的码值。

    Args:
        name: ``"32i"`` / ``"32f"`` / ``"16f"``。

    Returns:
        当前世代下应传的整数。

    Raises:
        RtError: 名字不认识，或环境变量给了非 ``legacy``/``new`` 的值。
    """
    legacy = os.environ.get("COFFIN_CUBLAS_COMPUTE", "").strip().lower()
    if legacy not in ("", "new", "legacy"):
        raise RtError(f"COFFIN_CUBLAS_COMPUTE={legacy!r}; expected 'new' or 'legacy'")
    table = {"32i": (COMPUTE_32I, COMPUTE_LEGACY_32I),
             "32f": (COMPUTE_32F, COMPUTE_LEGACY_32F),
             "16f": (COMPUTE_16F, CUDA_R_16F)}
    if name not in table:
        raise RtError(f"no compute type {name!r}")
    # 默认走**旧表**：上卡实测这块卡的厂商 ``libcublas`` 对新表的 ``computeType``
    # 一律回 ``not supported``（bf16/f32/i8 三档都一样，见 card_report.md §3）。
    # ``COFFIN_CUBLAS_COMPUTE=new`` 显式要新表；ixblas 在 f32 上认新表，切库时才用得上。
    return table[name][0] if legacy == "new" else table[name][1]


def find_libcublas(explicit: str | Path | None = None) -> str:
    """定位 ``libcublas``。

    Args:
        explicit: 调用方给的路径；``None`` 时按环境变量与常规位置找。

    Returns:
        可以被 ``ctypes.CDLL`` 装载的路径或名字。

    Raises:
        RtError: 一处都没找到。
    """
    if explicit:
        return str(explicit)
    env = os.environ.get("COFFIN_CUBLAS", "")
    if env:
        return env
    for name in ("libcublas.so.10", "libcublas.so.11", "libcublas.so"):
        found = ctypes.util.find_library("cublas")
        if found:
            return found
        for root in ("/usr/local/corex/lib64", "/usr/local/cuda/lib64", "/usr/lib/x86_64-linux-gnu"):
            candidate = Path(root) / name
            if candidate.exists():
                return str(candidate)
    raise RtError("libcublas not found; set COFFIN_CUBLAS=/path/to/libcublas.so")


class Cublas:
    """一个 cublas 句柄，以及唯一的 ``gemm_ex`` 出口。

    Attributes:
        path: 装载的库路径。
    """

    def __init__(self, lib: str | Path | None = None) -> None:
        """装载库并建句柄。

        Args:
            lib: 显式库路径；``None`` 时 :func:`find_libcublas` 去找。

        Raises:
            RtError: 装载失败、缺符号，或建句柄被拒。
        """
        self.path = Path(find_libcublas(lib))
        try:
            self._lib = ctypes.CDLL(str(self.path))
        except OSError as exc:                    # pragma: no cover - 需要真库
            raise RtError(f"cannot load {self.path}: {exc}") from exc
        self._handle = ctypes.c_void_p()
        self._bind()
        status = self._create(ctypes.byref(self._handle))
        if status:
            raise RtError(f"cublasCreate failed: {STATUS.get(status, status)}")

    def _bind(self) -> None:
        """把四个符号的 argtypes 写全。

        Raises:
            RtError: 缺符号。缺符号通常意味着装的是另一份库。
        """
        p, i = ctypes.c_void_p, ctypes.c_int
        try:
            self._create = self._lib.cublasCreate_v2
            self._create.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
            self._create.restype = i
            self._set_stream = self._lib.cublasSetStream_v2
            self._set_stream.argtypes = [p, p]
            self._set_stream.restype = i
            self._destroy = self._lib.cublasDestroy_v2
            self._destroy.argtypes = [p]
            self._destroy.restype = i
            self._gemm = self._lib.cublasGemmEx
            self._gemm.argtypes = [p, i, i, i, i, i, p, p, i, i, p, i, i, p, p, i, i, i, i]
            self._gemm.restype = i
        except AttributeError as exc:             # pragma: no cover - 需要真库
            raise RtError(f"{self.path.name} lacks a cublas entry: {exc}") from exc

    @property
    def handle(self) -> Any:
        """底层句柄，供诊断打印用。"""
        return self._handle

    def set_stream(self, stream: int | None) -> None:
        """把后续调用挂到一条流上。

        Args:
            stream: ``cudaStream_t`` 值；``0``/``None`` 表示默认流。

        Raises:
            RtError: 驱动拒绝换流。
        """
        status = self._set_stream(self._handle, ctypes.c_void_p(int(stream or 0)))
        if status:
            raise RtError(f"cublasSetStream failed: {STATUS.get(status, status)}")

    def gemm_ex(self, *, m: int, n: int, k: int, alpha: Any, beta: Any,
                a: int, a_type: int, lda: int, b: int, b_type: int, ldb: int,
                c: int, c_type: int, ldc: int, compute: int, transa: int, transb: int) -> None:
        """转发一次 ``cublasGemmEx``。

        参数名与签名逐字对齐；尺寸、op 与 ld 的推导在 :class:`GemmOps` 里做，这里只
        保证「不写漏参数」。``ctypes`` 少写一个参数不会报错，只会把后面整体错位一格。

        Raises:
            RtError: 非 ``0`` 的状态码，消息里带人话。
        """
        status = self._gemm(self._handle, int(transa), int(transb), int(m), int(n), int(k),
                            alpha, a, int(a_type), int(lda), b, int(b_type), int(ldb),
                            beta, c, int(c_type), int(ldc), int(compute), int(ALGO_DEFAULT))
        if status:
            raise RtError(f"cublasGemmEx({m}x{n}x{k}, ld={lda}/{ldb}/{ldc}) "
                          f"-> {STATUS.get(status, status)}")

    def close(self) -> None:
        """销毁句柄。幂等。"""
        if self._handle:
            self._destroy(self._handle)
            self._handle = ctypes.c_void_p()

    def __del__(self) -> None:                    # pragma: no cover - 解释器退出时
        try:
            self.close()
        except Exception:
            pass


class GemmOps:
    """预填充 GEMM 的可调用面：校验形状、推 cublas 参数、调库。

    三条出口共用同一个布局（见模块文档），差别只在 dtype 与累加器。

    Attributes:
        cublas: :class:`Cublas`。
    """

    def __init__(self, cublas: Cublas) -> None:
        self.cublas = cublas
        # 1.0f / 0.0f 与 1 / 0 两份常数在句柄生命周期内复用，避免每次调用都建临时对象。
        self._one_f = ctypes.c_float(1.0)
        self._zero_f = ctypes.c_float(0.0)
        self._one_i = ctypes.c_int32(1)
        self._zero_i = ctypes.c_int32(0)

    def i8_nk(self, xq: T.TensorLike, wtn: T.TensorLike, acc: T.TensorLike | None = None,
              stream: int | None = None) -> T.TensorLike:
        """``acc[m, n] = sum_k xq[m, k] * wtn[n, k]``，int8 入、int32 出。

        这是 int8 在这一层**唯一**的入口：权重按 nk 原缓冲直接进 ``OP_T``。曾经为
        「转一份 kn 再走 ``OP_N/OP_N``」留过 ``i8_kn`` / ``i8_kn_cols``，卡上量出来
        NN 比 TN 快 1.2~1.25 倍（49.3 vs 39.5 TOPS），但代价是一份 int8 权重副本；
        这条副本已经整条删掉——NK 是模型权重唯一的布局，也是后面要专门优化的方向。

        Args:
            xq: ``[M, K]`` int8 激活（``quant_act_i8`` 的产物）。
            wtn: ``[N, K]`` int8 权重，nk 原缓冲，不转置。
            acc: ``[M, N]`` int32 累加器；``None`` 时按 ``xq`` 分配。
            stream: 流句柄。

        Returns:
            ``acc``。

        Raises:
            ShapeError: 维度、dtype、连续性不符。
        """
        m, k, n = self._operands(xq, wtn, T.I8, "i8")
        if acc is None:
            acc = T.new_empty(xq, (m, n), T.I32C)
        self._check_out(acc, (m, n), T.I32C, "acc")
        if m == 0 or n == 0:
            return acc
        self.cublas.set_stream(stream)
        self.cublas.gemm_ex(m=n, n=m, k=k,
                            alpha=ctypes.byref(self._one_i), beta=ctypes.byref(self._zero_i),
                            a=T.ptr_of(wtn), a_type=CUDA_R_8I, lda=k,
                            b=T.ptr_of(xq), b_type=CUDA_R_8I, ldb=k,
                            c=T.ptr_of(acc), c_type=CUDA_R_32I, ldc=n,
                            compute=_compute_code("32i"), transa=OP_T, transb=OP_N)
        return acc

    def bf16_nk(self, x: T.TensorLike, wtn: T.TensorLike, out: T.TensorLike | None = None,
                out_code: int = T.BF16C, stream: int | None = None) -> T.TensorLike:
        """``out[m, n] = sum_k x[m, k] * wtn[n, k]``，bf16 入。

        ``--fp16`` 在 corex 上落的权重就是 bf16，这条把 ``torch.matmul`` 摘掉。库的
        bf16 路径在 ixblas 下先按 ``ld`` 转 f16 再走自研 fp16 张量核（fp32 累加），
        厂商那侧是 34.8 TFLOP/s 且算对；本卡 bf16 与 f16 都到 32.6。

        Args:
            x: ``[M, K]`` bf16 激活。
            wtn: ``[N, K]`` bf16 权重，nk 原缓冲。
            out: ``[M, N]`` 目的地，bf16 或 fp32；``None`` 时按 ``out_code`` 分配。
            out_code: ``T.BF16C`` 或 ``T.F32``，只在 ``out`` 为 ``None`` 时用。
            stream: 流句柄。

        Returns:
            ``out``。

        Raises:
            ShapeError: 维度、dtype、连续性不符。
        """
        m, k, n = self._operands(x, wtn, T.BF16C, "bf16")
        if out is None:
            out = T.new_empty(x, (m, n), out_code)
        code = T.code_of(out)
        if code not in (T.BF16C, T.F32):
            raise ShapeError(f"out must be bf16 or fp32, got {BY_CODE[code].name}")
        self._check_out(out, (m, n), code, "out")
        if m == 0 or n == 0:
            return out
        self.cublas.set_stream(stream)
        self.cublas.gemm_ex(m=n, n=m, k=k,
                            alpha=ctypes.byref(self._one_f), beta=ctypes.byref(self._zero_f),
                            a=T.ptr_of(wtn), a_type=CUDA_R_16BF, lda=k,
                            b=T.ptr_of(x), b_type=CUDA_R_16BF, ldb=k,
                            c=T.ptr_of(out), c_type=CUDA_R_16BF if code == T.BF16C else CUDA_R_32F,
                            ldc=n, compute=_compute_code("32f"), transa=OP_T, transb=OP_N)
        return out

    def f32_nk(self, x: T.TensorLike, wtn: T.TensorLike, out: T.TensorLike | None = None,
               stream: int | None = None) -> T.TensorLike:
        """``out[m, n] = sum_k x[m, k] * wtn[n, k]``，fp32 入出。

        参考实现与对拍用：ixblas 的 f32 TN 在 ``ld + 16`` 后 18.9 TFLOP/s，远高于厂商
        那条 7.2，所以「慢」不是拒绝它的理由。

        Args:
            x: ``[M, K]`` fp32 激活。
            wtn: ``[N, K]`` fp32 权重。
            out: ``[M, N]`` fp32；``None`` 时分配。
            stream: 流句柄。

        Returns:
            ``out``。

        Raises:
            ShapeError: 维度、dtype、连续性不符。
        """
        m, k, n = self._operands(x, wtn, T.F32, "f32")
        if out is None:
            out = T.new_empty(x, (m, n), T.F32)
        self._check_out(out, (m, n), T.F32, "out")
        if m == 0 or n == 0:
            return out
        self.cublas.set_stream(stream)
        self.cublas.gemm_ex(m=n, n=m, k=k,
                            alpha=ctypes.byref(self._one_f), beta=ctypes.byref(self._zero_f),
                            a=T.ptr_of(wtn), a_type=CUDA_R_32F, lda=k,
                            b=T.ptr_of(x), b_type=CUDA_R_32F, ldb=k,
                            c=T.ptr_of(out), c_type=CUDA_R_32F, ldc=n,
                            compute=_compute_code("32f"), transa=OP_T, transb=OP_N)
        return out

    # 校验

    def _operands(self, x: T.TensorLike, wtn: T.TensorLike, code: int,
                  name: str) -> tuple[int, int, int]:
        """检查两个操作数，返回 ``(M, K, N)``。

        Args:
            x: ``[M, K]`` 激活。
            wtn: ``[N, K]`` 权重。
            code: 两者共同的 dtype 码。
            name: 出错消息里的族名。

        Returns:
            ``(M, K, N)``。

        Raises:
            ShapeError: 不是二维、dtype 不同、K 对不上，或不连续。
        """
        sx, sw = T.shape_of(x), T.shape_of(wtn)
        if len(sx) != 2 or len(sw) != 2:
            raise ShapeError(f"{name} gemm needs 2-D operands, got {sx} / {sw}")
        if T.code_of(x) != code or T.code_of(wtn) != code:
            raise ShapeError(
                f"{name} gemm needs {BY_CODE[code].name} on both sides, got "
                f"{BY_CODE[T.code_of(x)].name} / {BY_CODE[T.code_of(wtn)].name}")
        m, k = int(sx[0]), int(sx[1])
        n, kw = int(sw[0]), int(sw[1])
        if k != kw:
            raise ShapeError(f"K mismatch: activation {sx} vs weight {sw}")
        for tensor, shape, tag in ((x, (m, k), "activation"), (wtn, (n, kw), "weight")):
            if T.strides_of(tensor) != (shape[1], 1):
                raise ShapeError(f"{tag} must be dense row-major, got strides "
                                 f"{T.strides_of(tensor)} for {shape}")
        return m, k, n

    def _check_out(self, out: T.TensorLike, shape: tuple[int, int], code: int,
                   tag: str) -> None:
        """检查目的地。

        Args:
            out: 目的地。
            shape: 期望 ``(rows, cols)``。
            code: 期望 dtype 码。
            tag: 消息里的名字。

        Raises:
            ShapeError: 形状、dtype 或连续性不符。
        """
        if T.shape_of(out) != shape:
            raise ShapeError(f"{tag} must be {shape}, got {T.shape_of(out)}")
        if T.code_of(out) != code:
            raise ShapeError(f"{tag} must be {BY_CODE[code].name}, got "
                             f"{BY_CODE[T.code_of(out)].name}")
        if T.strides_of(out) != (shape[1], 1):
            raise ShapeError(f"{tag} must be dense row-major, got strides {T.strides_of(out)}")


def available(lib: str | Path | None = None) -> bool:
    """库能不能装载。

    引擎在迁移期用它决定走这条还是退回原来那条；返回 ``False`` 只是「这台机器上没有
    可用的 libcublas」，不该被当成错误抛出去。

    Args:
        lib: 显式库路径。

    Returns:
        能装载且能建句柄为 ``True``。
    """
    try:
        handle = Cublas(lib)
    except RtError:
        return False
    handle.close()
    return True
