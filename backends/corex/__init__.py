"""Iluvatar CoreX（BI-V100）后端：手写 nk（``[N, K]``）内核，垫在算子契约之下。

这是 ``coffin`` 里第二个原生后端。同一份生产代码在 Hygon DCU 和这张卡上跑；变的
是算子表、权重布局和 dtype 策略，三者都属于本包。两个实测事实决定了这里的一切
（docs/qwen4/COREX-PLATFORM.md）：这张卡在 ``lm_head`` 的宽度上 ``cuBLAS`` 返回 NaN，
所以超过 ``pack.LIB_MAX_N`` 的输出列不许进库；另外权重的读取效率完全由「一条 32
条通道的分组每步能搬多少连续字节」决定——nk 的 ``[N, K]`` 让这一组每步覆盖 128 个
连续字节，``[K, N]``（kn）只能每步 32 个，实测差 1.02-1.63 倍。

**``[K, N]``（kn）内核在本包里已经一个都不剩**：稠密 kn GEMM、``gemv_kn*``、
``hyper_down/up``、``w8a8_gemm`` 的 kn 包装、``moe_grouped_i8`` 的 kn 主体核、
``w8a8_dp4a`` 的 kn 家族，连同一份 ``pack.KnWeight`` / ``KnQWeight`` 与
``as_kn`` / ``to_kn`` 的转换机制一起删除。推理里只有 nk 一种布局，也没有为 int8
转出来的 ``[K, N]`` 副本。

Module: ``coffin/backends/corex``。import ``backends.protocol``、
``backends.reference``（用来补齐还没有内核的算子）、``int8_weights`` 以及自己的
``pack`` / ``kernels`` 子模块，不从 ``coffin/ops``、``models``、``engine``、
``serving`` 或 ``cache`` 里 import 任何东西。import 它不得构建任何东西：扩展在第
一次 :func:`ops` 或 :func:`usable` 调用时才加载，所以纯 CPU 主机可以只枚举后端
而不被索要工具链。

Constraints:
    * ``caps.weight_layout == "nk"``：契约布局就是加载器手里的那份 ``int8/bf16 [N, K]``，
      一行输出通道、K 轴在内。加载器不再替任何后端转置，所以这里也不假装自己收到的是
      ``[K, N]``。**预填充**直接吃这份 nk 原缓冲：``cublas`` 的 ``transa = T`` 让行主序的
      ``[N, K]`` 就是列主序的 ``[K, N]``（``lda = ldb = K``），于是 ``pack.as_kn`` 那份
      整份转置副本（4B 上 4.3 GB）在默认路径上不再产生。**解码**同样吃原缓冲：int8 走
      ``GemvOps.nk_i8(_m)``（一列一行、K 在行内连续，一组 32 条通道每步读 128 个连续
      字节，实测比 ``[K, N]`` 快 1.02-1.63 倍），稠密走库的 ``GemmEx``。

      **本包不再自己造任何副本。** 曾经为 int8 加过「转一份 ``[K,N]`` 走
      ``OP_N/OP_N``（49.3 TOPS，TN 是 39.5）」和 ``COFFIN_I8_GEMM=tcu`` 那条分支，
      两者都已删除：NK 是模型权重唯一的布局，也是后面要专门优化的方向，一份 int8
      权重副本换来的两成速度不值得。
    * 本包**不再调用任何 torch 算子**（``torch.matmul`` / ``mm`` / ``bmm`` / ``F.linear``
      一个都没有）：投影走 ``cublas``，其余算子走 ``abi/*.cu``。``torch.empty`` /
      ``zeros`` 只是分配，``.to()`` 只是窄化，两者不算算子。唯一的例外是
      :func:`_solve_blocked`——那是「扩展不可用」时才走的纯 torch 回退，不在推理链路上。
      验收脚本：``python tools/audit_torch_path.py``。
    * 输出列超过 ``pack.LIB_MAX_N`` 时不许用库 GEMM。本卡在 ``N = 248320`` 实测
      NaN，在 ``N = 131072`` 结果正确，所以预填充路径沿 ``N`` 分块，解码路径则完全
      不调库。这是 ``lm_head`` 算错还是算慢的分界。
    * 每次内核启动都在 ``c10::cuda::getCurrentCUDAStream()`` 上（``.cu`` 文件自己
      完成）。在 legacy stream 上启动不属于 ``torch.cuda.graph`` 的捕获范围，被捕获
      的解码步会重放陈旧输出——这是会被误读成数值问题的正确性 bug。走 ``cxrt`` 的那些
      出口（``abi/*.cu`` 与 ``cublas``）由 Python 侧显式传流
      （``torch.cuda.current_stream().cuda_stream``，见 :func:`_stream`）：``cxrt`` 不查
      任何全局「当前流」，所以两条流不会在同一批字节上竞态。
    * ``caps.graph`` 和 ``caps.batch_attn`` 是从算子表推导出来的，不是声明的。只有
      当没有任何算子回退到参考实现时捕获才安全，因为那条回退会走到 ``cuBLAS``，而
      它在这里无法捕获；同理 ``batch_attn`` 的含义是 ``attn_paged_batch`` 有内核。
      两者随清单补齐自动变 ``True``，所以都不会漂成一个谎话。
    * 全程不用 fp64。本设备对它返回 0，所以所有累加器和所有比较都停在 fp32 或
      更窄。

内核清单逐个算子增长；没有内核的算子不进 :data:`_NATIVE`，由 :func:`ops` 从参考
后端补齐，这就是一个只实现了一半的平台仍然能跑整个模型的原因。
"""

from __future__ import annotations

import functools
import os
import sys
from typing import Any

import torch

from backends import reference
from backends.corex import pack
from backends.protocol import Backend, Caps, Ops, OpFn, make_device, native_ops
from int8_weights import QWeight, QWeightMoE

#: 后端标识符，即 ``COFFIN_BACKEND=corex`` 选中的名字。
NAME = "corex"

#: 模块级别名，让本模块满足 :class:`~coffin.backends.protocol.Backend`。
name = NAME

#: 驱动报告不出可用的 ISA 时，扩展按这个架构编译。``ivcore10`` 是 BI-V100 的
#: 目标（docs/qwen4/COREX-PLATFORM.md）；报告了不同架构串的卡就用那个值，所以换新部件时构建
#: 选项永远不必改。
DEFAULT_ARCH = "ivcore10"

#: 在共享的 ``cuda`` 命名空间里标识本厂商的设备名标记。按子串不分大小写匹配：
#: 厂商自己的代码是对精确串 ``"Iluvatar BI-V100"`` 做比较的，到 BI-V150 就会坏
#: 掉，这个元组存在的意义就是避开那个错误。
_NAME_MARKERS: tuple[str, ...] = ("iluvatar", "bi-v100", "bi-v150", "corex")

#: 激活行数超过它，投影就是算力受限，库 GEMM 胜过流式读权重。低于它时 GEMV 把
#: 权重读一遍，而 GEMM 要 *每个 tile* 读一遍，这就是两种情形被拆成两个算子的原因。
_PREFILL_MIN_M = 64

#: 为了避开库而必须分块时，每次内核启动处理的行数。:func:`_gemv_rows` 最多把这么
#: 多行并进一次权重读取，正好对上小 M 内核的特化范围。
#:
#: **不要把它调大。** 试过改成 64（理由是内核现在支持任意 M，按 8 切会让 10 行付
#: 两遍权重），实测 B=10 从 466.71 变成 475.58 ms —— **更慢**。原因是 nk 批处理入口
#: （``gemv_nk_i8_m``）按宽度特化，宽度之外的行由上一档承接。
_ROW_CHUNK = 8


#: ``gemv_nk_i8_m`` 每个通道组认领的列数：``0``/``1`` 一列一组，``2`` 两列一组。
#:
#: M >= 2 时每 128 字节权重只有 1 次权重读，却有 M 次激活读加 4M 次 FMA —— 压住这一档的
#: 是 L1 与发射，不是 HBM。两列一组把每步的激活读摊到两列上（M 次 -> M/2 次），同时在
#: 飞的权重字节翻倍；代价是块数减半，而 R8 那边 N=2560 仍有 320 个块、每 SM 20 个。
#: M = 1 不适用（那里本来就只有 1 次激活读，块数砍半是纯损失），所以只在行数 > 1 的
#: 分支里生效。默认 1 列，量过再改。
_GEMV_NK_WIDTH = int(os.environ.get("COFFIN_GEMV_NK_WIDTH", "0") or "0")

#: ``gemv_nk_i8_m`` 的 K 循环展开度：``0`` 表示内核默认值 4，可选 ``2``/``8``。
#:
#: 为什么值得动：解码里 ``N = 12352`` 那个投影（grid 3088 x 128 线程）在自己的时间轴上
#: 是 **127.63 us** 一发，按 ``N * K = 25.3 MB`` 算只有 **198 GB/s** —— 离这张卡的带宽
#: 还差一个量级，而在飞字节数由展开度决定：U=4 时每个 warp 只有 ``4 x 128 = 512 B``
#: 在天上。把 U 抬到 8 就是 1 KB，正好补上这条「带宽-延迟积」的缺口。
#: 默认 0（不动内核默认 4），量过再改默认值。
_GEMV_NK_UNROLL = int(os.environ.get("COFFIN_GEMV_NK_UNROLL", "0") or "0")

#: 投机校验的 W8A8 DP4A 路径；``COFFIN_COREX_W8A8_DP4A_VERIFY=1`` 打开以对照原有 GEMV。
#:
#: **实测更慢，所以默认关**（作业 5077，2026-11-12，gpu-node-06，B=8、K=1、M=B*c=16，
#: 一次前向校验的整步墙钟）：关 82.82 ms，开 210.88 ms，即 **2.55x 退化**。分段看是
#: 全面退化而不只是投影：mlp 段 101.00 -> 347.15 ms（三次累计）、proj 38.39 -> 86.00。
#: 也就是说 ``w8a8_dp4a_nk`` 省下的权重重读抵不过它自己的代价——激活要先量化一遍
#: （``quant_act_i8``）、DP4A 的 ALU 吞吐在这张卡上不如 GEMV 路径的 FMA 摊销，
#: 而且 M=16 本来也只是 ``_ROW_CHUNK=8`` 的 2 块、重读代价只有 2 倍。
#:
#: 保留这个开关是因为它描述的路径本身没被否掉：真正缺的是「一次 launch 摊掉任意 M 行
#: 且不比 GEMV 慢」的宽 M 内核，DP4A 是现成的一个候选而不是答案。要重开它，得先让
#: mlp 段在 M=16 上按元素胜出。
_W8A8_DP4A_VERIFY = os.environ.get("COFFIN_COREX_W8A8_DP4A_VERIFY", "0").strip() == "1"

#: 预填充是否走 nk 原生张量核、不做那份 ``[K, N]`` 转置（``COFFIN_PREFILL_NK=1``）。
#: 默认关，理由与实测数字见 :func:`_prefill_nk_tcu` 的 Note：4B 上省 4.3 GB 显存
#: 但预填充 2468 -> 662 tok/s。只有「权重加副本放不下」时才值得开（Occamy）。
_PREFILL_NK = os.environ.get("COFFIN_PREFILL_NK", "0").strip() == "1"

#: 宽指令张量核 GEMM 的形状门限，按**权重字节数**而不是 K 或 N 单独判。
#:


@functools.lru_cache(maxsize=1)
def _is_corex_host() -> tuple[bool, str]:
    """在 torch 共享的 ``cuda`` 命名空间里认出本厂商。

    ``torch.cuda.is_available()`` 不是厂商测试：ROCm、CoreX 和普通 NVIDIA 驱动都
    答 True，而 ``cuda`` 这个设备类型说明不了卡是谁造的。共享命名空间下设备
    *名字* 是唯一的判据，所以这里向驱动要名字，也所以 ROCm 后端额外要求自己的
    构建标记。

    **必须记忆化。** ``probe()`` 从 :func:`_arch` 与 :func:`usable` 两条路被每次
    算子调用间接问道，而这里的 ``get_device_name`` 会初始化 CUDA 运行时并做一次
    驱动查询。一步 decode 有几千次算子调用（``q4budget2`` 实测 7025 次发射、
    主机侧 99.4% 的时间），把它留在热路径上等于每次投影都付一次驱动往返。
    「这张卡是谁造的」在进程生命周期内不会变，所以缓存是安全的。

    Returns:
        ``(matched, device_name)``。驱动调用失败是 ``(False, "")``：「说不清」
        绝不能变成「是我的」，否则一个坏驱动会被本后端认领，然后在一次内核启动里
        失败。
    """
    if getattr(torch.version, "hip", None) is not None:
        # 这是 ROCm 构建。即便有 CoreX 卡通过它应答，能针对 HIP ISA 构建的也是 HIP
        # 后端，所以本后端让路。
        return (False, "")
    try:
        if not torch.cuda.is_available():
            return (False, "")
        device_name = str(torch.cuda.get_device_name(0))
    except Exception:                             # pragma: no cover - driver error
        return (False, "")
    lowered = device_name.lower()
    return (any(marker in lowered for marker in _NAME_MARKERS), device_name)


def probe() -> bool:
    """报告本主机上有没有 Iluvatar 设备。

    只读驱动：不构建扩展、不分配、不重置设备。自动探测时注册表会挨个探测所有候选
    后端，纯 CPU 机器上也一样。

    Returns:
        仅当 ``cuda`` 命名空间的设备可见 *且* 名字匹配本厂商时才为 ``True``。只凭
        名字答 ``True`` 会让本后端认领一张别人的卡，然后为一个它没有的 ISA 编译。
    """
    return _is_corex_host()[0]


@functools.lru_cache(maxsize=1)
def _device_name() -> str:
    """只读一次板卡名，供启动横幅和架构回退使用。

    Returns:
        ``torch.cuda.get_device_name(0)``，没有设备时为 ``""``。这个调用会初始化
        运行时，所以被记忆化，而且只在 :func:`probe` 已经报过本厂商之后才走到。
    """
    if not probe():
        return ""
    return _is_corex_host()[1]


@functools.lru_cache(maxsize=1)
def _arch() -> str:
    """报告该为哪个 ISA 编译，优先用驱动暴露的值。

    架构是构建选项，所以编造一个值比用实测的默认值更糟：编译器不认识的串会让构建
    失败，而它默默接受的串会指向错误的 ISA。因此只有驱动交回 CoreX 形状的名字时
    才用它；否则返回 :data:`DEFAULT_ARCH`，也就是这张卡被实测时用的值
    （docs/qwen4/COREX-PLATFORM.md）。

    **必须记忆化。** :func:`_kernels` 每次调用都问它，而一次 decode 里有几千次算子
    调用（``q4budget2`` 实测 7025 次内核发射）。这里的 ``torch.cuda.get_device_properties``
    会走一遍驱动查询，把它留在热路径上等于给每次投影加一次驱动往返。ISA 是进程级
    常量，第一次问到的答案永远是正确答案。

    Returns:
        给 ``--cuda-gpu-arch=`` 用的 ISA 串；在完全不是 CoreX 的主机上返回 ``""``
        （为一张不在场的卡构建从来没用）。
    """
    if not probe():
        return ""
    try:
        props = torch.cuda.get_device_properties(0)
    except Exception:                             # pragma: no cover - driver error
        return DEFAULT_ARCH
    for field in ("arch", "gcnArchName", "device_arch"):
        value = getattr(props, field, None)
        if isinstance(value, str) and value.lower().startswith("ivcore"):
            return value.split(":")[0]
    return DEFAULT_ARCH


#: 一次构建尝试的结果，``None`` 表示工具链可用。记忆化：坏工具链每进程只花一次尝试。
_SHIM_ERROR: str | None = None
#: 是否已经尝试过。
_SHIM_TRIED = False


def _build_error() -> str | None:
    """构建最小的那个 ``cxrt`` 库，判断工具链能不能用（记忆化）。

    Returns:
        构建失败时为 ``"<ExceptionType>: <message>"``，否则 ``None``。

    Note:
        判据从「torch 扩展能不能编出来」换成了「``cxrt`` 的 ABI 库能不能编出来」：
        链路已经一个 torch 符号都不碰，能不能服务模型只取决于 C ABI 那条构建。
    """
    global _SHIM_ERROR, _SHIM_TRIED
    if _SHIM_TRIED:
        return _SHIM_ERROR
    _SHIM_TRIED = True
    try:                                   # noqa: SIM105 - 失败文本要留下
        from cxrt import open_runtime
        open_runtime().build_norm(_arch() or DEFAULT_ARCH)
    except Exception as exc:               # noqa: BLE001 - 任何失败都算工具链不可用
        _SHIM_ERROR = f"{type(exc).__name__}: {exc}"
    return _SHIM_ERROR


def usable() -> bool:
    """报告本后端此刻能不能执行模型。

    Returns:
        仅当设备在场 *且* 扩展构建成功时为 ``True``。否则注册表会落到下一个候选，
        这样有驱动但没有工具链的主机仍然通过参考后端服务模型，而不是在第一步解码时
        把进程带走。
    """
    return probe() and _build_error() is None


def reason() -> str:
    """用一行说明 :func:`usable` 为什么是 ``False``。

    Returns:
        可用时为 ``""``，否则是设备状态或构建异常。没有它，回退到参考实现看起来就
        像一次神秘变慢的部署，而不是缺了内核构建。
    """
    if not probe():
        seen = _is_corex_host()[1]
        detail = f" (got {seen!r})" if seen else ""
        return f"no Iluvatar CoreX device visible through torch.cuda{detail}"
    err = _build_error()
    return "" if err is None else f"kernel build failed: {err}"


def _kernels() -> Any:
    """已被删除的 torch JIT 扩展。

    Returns:
        从不返回：这个函数现在是**桩**。扩展连同 18 个 torch 宿主 ``.cu`` 一起删掉了
        （``docs/COREX-NO-TORCH.md`` §7），解码链路整条走 ``cxrt`` 的 C ABI。

    Raises:
        RuntimeError: 永远。仍在调用它的是两处还没迁移的旁路——``models/qwen35/ep.py``
            的跨 rank 归约与 ``models/qwen4exp/parallel.py`` 的映射内存注册。它们不在
            单卡解码路径上，但一旦被调到必须立刻报错，而不是静默地退到参考实现。
    """
    raise RuntimeError(
        "the CoreX torch JIT extension was deleted (docs/COREX-NO-TORCH.md §7); the EP "
        "collectives and mapped-memory paths still need their own ABI migration")


def _compute_dtype(x: torch.Tensor) -> torch.Tensor:
    """把激活拓宽成内核有实例化的 dtype。

    内核按 bf16 和 fp32 编译。别的一律拓宽到 fp32（例如调用方攥着旧张量传进来的
    fp16 激活），对任何 16 位浮点都是无损的，结果由调用方窄化回去。这里故意不用
    参考后端兜底：它的 ``rms_norm`` 调 ``F.rms_norm``，而本集群的 torch 2.1 上
    没有那个函数。

    Args:
        x: 任意浮点 dtype 的激活张量。

    Returns:
        ``x`` 自己是 bf16 或 fp32 时原样返回，否则返回 fp32 副本。
    """
    if x.dtype in (torch.bfloat16, torch.float32):
        return x
    return x.float()


def _restore(y: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """把内核结果窄化回调用方的 dtype。

    Args:
        y: 内核输出，bf16 或 fp32。
        dtype: 调用方传入并期望拿回的 dtype。

    Returns:
        ``y`` 已经是 ``dtype`` 时原样返回，否则返回 ``dtype`` 的副本。
    """
    return y if y.dtype == dtype else y.to(dtype)


# 预填充的 torch-free 算子面
#
# 这三条出口（``quant_act_i8`` / ``coffin_w8a8_epilogue`` 在 ABI 里，``GemmEx`` 在
# libcublas 里）合起来就是一次 int8 预填充投影，中间没有一步回到 torch。缓存的是
# **算子对象**：``build_w8a8`` 会编一次 ``.so``，``build_gemm`` 会开一个 cublas 句柄，
# 都得在一遍请求里只做一次。

#: :class:`cxrt.GemmOps`，首次使用时建。
_GEMM_OPS: Any = None

#: :class:`cxrt.W8A8Ops`，首次使用时建（会编 ``abi/w8a8_epilog_abi.cu``）。
_W8A8_OPS: Any = None

#: :class:`cxrt.GemvOps`，首次使用时建（会编 ``abi/gemv_abi.cu``）。
_GEMV_OPS: Any = None

#: :class:`cxrt.RMSNormOps`，首次使用时建（会编 ``abi/norm_abi.cu``）。
_NORM_OPS: Any = None

#: :class:`cxrt.ConvOps`，首次使用时建（会编 ``abi/conv_abi.cu``）。
_CONV_OPS: Any = None

#: :class:`cxrt.AttnOps`，首次使用时建（会编 ``abi/attn_abi.cu``）。
_ATTN_OPS: Any = None

#: :class:`cxrt.SolveOps`，首次使用时建（会编 ``abi/solve_abi.cu``）。
_SOLVE_OPS: Any = None



#: 快速取当前 CUDA 流句柄的 C 扩展入口缓存，避免每步 870+ 次走
#: ``torch.cuda.current_stream()``（其内部每次读环境变量并构造 Python ``Stream``）。
_FAST_STREAM: Any = None
_FAST_DEV: Any = None
_FAST_SINGLE_DEV: bool = False


def _stream() -> int | None:
    """当前流的句柄：**没有 torch 时就是默认流 0**。

    这条出口不是「顺手取一下」：``cxrt`` 的 ABI 出口、``cublas`` 句柄上的队列与内核
    启动必须落在同一条流上，否则两个算子会在同一批字节上竞态（表现为偶发的错值，不是
    崩溃）。以前这条流只能去问 torch 要，因为链路上跑的张量归 torch 的分配器管。

    现在它**先看 torch 在不在场**：``sys.modules`` 里没有 torch，或者那一个是
    :mod:`cxrt.torchshim`（本项目的壳，``cuda`` 上没有 ``current_stream``），就返回
    ``0``——``cudart`` 的默认流本来就是这条链唯一在用的流，我们自己建的流与它之间不需要
    任何同步，因为链上没有第二个所有者了。真 torch 在场时走 ``_cuda_getCurrentRawStream``
    直取裸流号，语义与 ``torch.cuda.current_stream().cuda_stream`` 完全一致。

    Returns:
        流句柄号，或 ``None``（没有 CUDA 上下文时）。
    """
    global _FAST_STREAM, _FAST_DEV, _FAST_SINGLE_DEV
    if _FAST_STREAM is not None:
        return _FAST_STREAM(0) if _FAST_SINGLE_DEV else _FAST_STREAM(_FAST_DEV())
    tl = sys.modules.get("torch")
    if tl is None:
        return 0
    cuda = getattr(tl, "cuda", None)
    current = getattr(cuda, "current_stream", None)
    if current is None:
        # 壳没有流的概念：它跑的是 numpy，不存在跨流竞态。
        return 0
    s = int(current().cuda_stream)
    c_mod = getattr(tl, "_C", None)
    raw = getattr(c_mod, "_cuda_getCurrentRawStream", None)
    dev = getattr(c_mod, "_cuda_getDevice", None)
    if raw is not None and dev is not None:
        try:
            if int(raw(int(dev()))) == s:
                _FAST_STREAM = raw
                _FAST_DEV = dev
                _FAST_SINGLE_DEV = (int(cuda.device_count()) == 1)
        except Exception:
            pass
    return s


def _gemm_ops() -> Any:
    """返回预填充 GEMM 的库面，首次调用时绑定。

    Returns:
        :class:`cxrt.GemmOps`。

    Raises:
        RtError: 找不到 ``libcublas``（装了 ixblas 就是它的同名替换）。
    """
    global _GEMM_OPS
    if _GEMM_OPS is None:
        from cxrt import build_gemm

        _GEMM_OPS = build_gemm()
    return _GEMM_OPS


def _w8a8_ops() -> Any:
    """返回 W8A8 的 ABI 面（量化 + epilogue），首次调用时编译并装载。

    Returns:
        :class:`cxrt.W8A8Ops`。

    Raises:
        BuildError: 工具链缺失或编译失败。
    """
    global _W8A8_OPS
    if _W8A8_OPS is None:
        from cxrt import open_runtime

        _W8A8_OPS = open_runtime().build_w8a8(_arch() or DEFAULT_ARCH)
    return _W8A8_OPS


def _gemv_ops() -> Any:
    """返回投影 GEMV 的 ABI 面，首次调用时编译并装载。

    ``coffin_gemv_nk_i8(_m)`` 与 torch 扩展里那两条是**同一个内核**（设备代码是同一份
    ``gemv_device.inc``），所以换过来是纯搬家：数值逐位相同，唯一的变化是参数不再是
    ``torch::Tensor``。

    Returns:
        :class:`cxrt.GemvOps`。

    Raises:
        BuildError: 工具链缺失或编译失败。
    """
    global _GEMV_OPS
    if _GEMV_OPS is None:
        from cxrt import open_runtime

        _GEMV_OPS = open_runtime().build_gemv(_arch() or DEFAULT_ARCH)
    return _GEMV_OPS

#: MoE 族 ABI 句柄。``None`` 表示还没编过。
_MOE_OPS: Any = None

#: 四卡 P2P 归约 ABI 句柄。``None`` 表示还没编过。
_P2P_OPS: Any = None

#: 分组张量核转置激活缓存：``(rows, k, device)`` → ``[k, rows]`` int8。
_MOE_AT_SCRATCH: dict = {}


def _p2p_ops() -> Any:
    """返回四卡 P2P 归约的 ABI 面，首次调用时编译并装载。"""
    global _P2P_OPS
    if _P2P_OPS is None:
        from cxrt import open_runtime

        _P2P_OPS = open_runtime().build_p2p(_arch() or DEFAULT_ARCH)
    return _P2P_OPS


def _moe_at_scratch(tokens: int, k: int, device: torch.device) -> torch.Tensor:
    """取（必要时分配）分组张量核专用的 ``[k, tokens]`` int8 转置缓冲。"""
    key = (tokens, k, device)
    buf = _MOE_AT_SCRATCH.get(key)
    if buf is None:
        if len(_MOE_AT_SCRATCH) >= 8:
            _MOE_AT_SCRATCH.clear()
        buf = torch.empty((k, tokens), dtype=torch.int8, device=device)
        _MOE_AT_SCRATCH[key] = buf
    return buf


def _norm_ops() -> Any:
    """返回归一化族的 ABI 面，首次调用时编译并装载。

    设备代码是同一份 ``norm_device.inc``（``norm.cu`` 与 ``abi/norm_abi.cu`` 各包含
    一次），所以从 torch 扩展换到这里是纯搬家：同样的五个 ``__global__``、同样的
    整数预算，数值逐位相同。区别只有两处：目的地由调用方给（``norm_abi`` 不分配），
    以及参数不再是 ``torch::Tensor`` 而是 ``cxrt.Array`` 视图。

    Returns:
        :class:`cxrt.RMSNormOps`。

    Raises:
        BuildError: 工具链缺失或编译失败。
    """
    global _NORM_OPS
    if _NORM_OPS is None:
        from cxrt import open_runtime

        _NORM_OPS = open_runtime().build_norm(_arch() or DEFAULT_ARCH)
    return _NORM_OPS


def _conv_ops() -> Any:
    """返回线性注意力入口那个深度卷积的 ABI 面，首次调用时编译并装载。

    设备代码是同一份 ``conv_device.inc``（``linear_attn.cu`` 与 ``abi/conv_abi.cu``
    各包含一次），三个入口的算术因此逐位相同；批处理那条启动的就是单 token 那条
    内核（行轴收成 ``rows``），不是另一份实现。

    Returns:
        :class:`cxrt.ConvOps`。

    Raises:
        BuildError: 工具链缺失或编译失败。
    """
    global _CONV_OPS
    if _CONV_OPS is None:
        from cxrt import open_runtime

        _CONV_OPS = open_runtime().build_conv(_arch() or DEFAULT_ARCH)
    return _CONV_OPS


def _attn_ops() -> Any:
    """返回注意力族里已经搬进 ABI 的四个算子，首次调用时编译并装载。

    设备代码与 ``attn.cu`` 里那四条是同一份 ``attn_device.inc``（两条前端各包含一次），
    所以 ``silu_mul`` / ``silu_mul_quant`` / ``rope_apply`` / ``kv_quant`` 换过来是纯
    搬家。分页注意力那几条要连块表一起定契，还在扩展里，见
    ``docs/COREX-NO-TORCH.md`` §6.2。

    Returns:
        :class:`cxrt.AttnOps`。

    Raises:
        BuildError: 工具链缺失或编译失败。
    """
    global _ATTN_OPS
    if _ATTN_OPS is None:
        from cxrt import open_runtime

        _ATTN_OPS = open_runtime().build_attn(_arch() or DEFAULT_ARCH)
    return _ATTN_OPS


def _solve_ops() -> Any:
    """返回前向替换求解的 ABI 面，首次调用时编译并装载。

    Returns:
        :class:`cxrt.SolveOps`。

    Raises:
        BuildError: 工具链缺失或编译失败。
    """
    global _SOLVE_OPS
    if _SOLVE_OPS is None:
        from cxrt import open_runtime

        _SOLVE_OPS = open_runtime().build_solve(_arch() or DEFAULT_ARCH)
    return _SOLVE_OPS


def _moe_ops() -> Any:
    """返回 MoE 族已经搬进 ABI 的算子，首次调用时编译并装载。

    设备代码与 ``moe_epilogue.cu`` 里那条是同一份 ``shared_gate_device.inc``（两条前端
    各包含一次），所以 ``shared_gate_scale`` 换过来是纯搬家。``moe_router`` /
    ``moe_dispatch`` / ``moe_grouped_i8*`` 要连分组偏移与槽位表一起定契，还在扩展里。

    Returns:
        :class:`cxrt.MoeOps`。

    Raises:
        BuildError: 工具链缺失或编译失败。
    """
    global _MOE_OPS
    if _MOE_OPS is None:
        from cxrt import open_runtime

        _MOE_OPS = open_runtime().build_moe(_arch() or DEFAULT_ARCH)
    return _MOE_OPS


def _ptr(t: torch.Tensor) -> int:
    """取设备地址。

    热路径上用这个而不是 ``Array.borrow``：一次借用要问形状、步长、dtype 码、存储
    偏移，再种两个 Python 对象；实测把一次归一化的主机侧成本从 6.7 us 抬到 30 us
    以上，而一个解码步里有七十多次归一化——折下来正好是每步 5 ms，占 16.7 ms 步时
    的三成。``cxrt`` 的 ``Array`` 面留着给离机单测和 ``selftest`` 用，这里直接进
    同一个 ``extern "C"`` 入口，参数只给地址。

    Args:
        t: 设备张量。

    Returns:
        设备地址（整数）。
    """
    return int(t.data_ptr())


#: torch.dtype → ABI 整数码缓存，省去每次 ``_dcode`` 的局部 import 与字符串拆分。
_DCODE_MAP: dict[Any, int] = {}


def _dcode(t: torch.Tensor) -> int:
    """取 dtype 的内核整数码。

    Args:
        t: 设备张量。

    Returns:
        ``cx::Tensor`` 认的 dtype 码。
    """
    dt = t.dtype
    code = _DCODE_MAP.get(dt)
    if code is not None:
        return code
    from cxrt.tensors import code_of

    code = code_of(dt)
    _DCODE_MAP[dt] = code
    return code


def _norm_out(shape: tuple[int, ...], like: torch.Tensor) -> torch.Tensor:
    """按``like`` 的 dtype/设备分配一块稠密目的地。

    ``norm_abi`` 的契约是「目的地归调用方」：捕获/重放的路径要求地址稳定，所以这一
    层显式分配，而不是让 ABI 层偷偷分配一块 cxrt 自己的缓冲。

    Args:
        shape: 目的地形状，``[rows, width]``。
        like: 对 dtype 与设备的参照张量。

    Returns:
        新张量，内容是未初始化的。
    """
    return torch.empty(shape, dtype=like.dtype, device=like.device)


def _gemv_nk(nkw: torch.Tensor, ws: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
    """nk 原生的 int8 解码投影，直调 ABI shim，不经过 torch 扩展与多层包装校验。

    收的是**裸缓冲**而不是 ``QWeight``：``QWeight.wq`` 就是 ``[N, K]`` 的那一份，
    零拷贝传进来即可，于是「哪条权重来源」与「怎么算」在这里分开。

    Args:
        nkw: ``[N, K]`` int8 权重，连续。
        ws: ``[N]`` fp32 每列 scale。
        act: 激活，``[K]`` 或 ``[C, K]``，bf16 或 fp32，连续。

    Returns:
        ``[N]`` 或 ``[C, N]``，dtype 同 ``act``。

    Raises:
        ShapeError: 形状、dtype 或对齐不符（``K % 4``、bf16 需 8 字节对齐）。
    """
    shim = _gemv_ops().shim
    stream = _stream() or 0
    n = int(nkw.shape[0])
    k = int(nkw.shape[1])
    xcode = _dcode(act)
    unroll = _GEMV_NK_UNROLL or 4
    w_ptr = nkw.data_ptr()
    s_ptr = ws.data_ptr()
    if act.dim() == 1:
        out = torch.empty((n,), dtype=act.dtype, device=act.device)
        shim.nk_i8(w_ptr, act.data_ptr(), xcode, s_ptr, n, k, out.data_ptr(),
                   128, unroll, _GEMV_NK_WIDTH, stream)
        return out
    rows = int(act.shape[0])
    if rows <= _ROW_CHUNK:
        out = torch.empty((rows, n), dtype=act.dtype, device=act.device)
        shim.nk_i8_m(w_ptr, act.data_ptr(), xcode, s_ptr, n, k, rows,
                     out.data_ptr(), 128, unroll, _GEMV_NK_WIDTH, stream)
        return out
    out = torch.empty((rows, n), dtype=act.dtype, device=act.device)
    for start in range(0, rows, _ROW_CHUNK):
        stop = min(rows, start + _ROW_CHUNK)
        chunk = act[start:stop]
        if not chunk.is_contiguous():
            chunk = chunk.contiguous()
        dst = out[start:stop]
        shim.nk_i8_m(w_ptr, chunk.data_ptr(), xcode, s_ptr, n, k, stop - start,
                     dst.data_ptr(), 128, unroll, _GEMV_NK_WIDTH, stream)
    return out


def _gemm_nk_dense(w: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
    """稠密（bf16/fp32）权重的一次投影：小 M 走自研 nk GEMV，大 M 走库 GEMM。

    在 BI-V100 上 ``cublasGemmEx`` 算解码路由 ``[1, 2048] x [256, 2048]^T`` 单次要
    471 us（40 层即 19 ms/步），而 ``coffin_gemv_nk_dense`` 约 12-15 us。这里在
    ``M <= _ROW_CHUNK``（1..8）且权重大致规整时直调 nk 稠密 GEMV，大 M 仍走库。

    Args:
        w: ``[N, K]`` 稠密权重，bf16 或 fp32。
        act: ``[K]`` 或 ``[C, K]`` 激活。

    Returns:
        与 ``act`` 前导维数一致的输出。
    """
    if (w.dtype in (torch.bfloat16, torch.float32) and w.is_contiguous()
            and int(w.shape[1]) % 4 == 0):
        n, k = int(w.shape[0]), int(w.shape[1])
        lhs = act if act.dtype == w.dtype else act.to(w.dtype)
        if not lhs.is_contiguous():
            lhs = lhs.contiguous()
        if lhs.dim() == 1:
            out = torch.empty((n,), dtype=w.dtype, device=w.device)
            _gemv_ops().shim.nk_dense(w.data_ptr(), lhs.data_ptr(), _dcode(lhs),
                                      n, k, out.data_ptr(), 128, _stream() or 0)
            return out
        rows = int(lhs.shape[0])
        if rows <= _ROW_CHUNK:
            out = torch.empty((rows, n), dtype=w.dtype, device=w.device)
            _gemv_ops().shim.nk_dense_m(w.data_ptr(), lhs.data_ptr(), _dcode(lhs),
                                        n, k, rows, out.data_ptr(), 128, _stream() or 0)
            return out
    if act.dim() == 2:
        return _gemm_nk_rows(w, act)
    return _gemm_nk_rows(w, act.unsqueeze(0))[0]


# 投影


def linear(w: torch.Tensor | QWeight, x: torch.Tensor) -> torch.Tensor:
    """解码期投影：一行激活对 ``[N, K]`` 的 ``w`` 做乘加。

    Args:
        w: ``[N, K]`` 投影权重——普通张量、``QWeight``（保持 int8，由 W8A16 内核
            承担，这正是声明 ``int8_weights`` 的意义），或加载器已经转置好的打包
            权重。
        x: 激活，``[K]`` 或 ``[C, K]``。

    Returns:
        ``[N]`` 或 ``[C, N]``，dtype 同 ``x``。

    Raises:
        RtError: 找不到 ``libcublas``（装了 ixblas 就是它的同名替换）。这条入口
            现在完全不需要 torch 扩展：int8 走 ``coffin_gemv_nk_i8``，稠密走 ``GemmEx``。
    """
    if x.dim() != 1:
        return _gemv_rows(w, x)
    dtype = x.dtype
    act = _compute_dtype(x).contiguous()
    if isinstance(w, QWeight):
        # nk 原生：盘上那一份就是内核要的几何，直接交出去（``w.wq`` 即 ``[N, K]``）。
        # 早先这里会先转一份 ``[K, N]``（4B 是 4.3 GB）再把原缓冲要回来——那份副本除
        # 了吃显存什么都不做，已经连同转换机制一起删除。
        # 传输走 :func:`_gemv_nk`：同一个内核的 ABI 出口，参数不再是 ``torch::Tensor``，
        # 所以这条路径连 torch 扩展都不必装载。
        return _restore(_gemv_nk(w.wq, w.scale, act), dtype)
    # 稠密（bf16/fp32）：nk 权重直接交给库的 TN 出口，不转置。
    return _restore(_gemm_nk_dense(w, act), dtype)


def linear_m(w: torch.Tensor | QWeight, x: torch.Tensor) -> torch.Tensor:
    """小 M 投影：``M`` 行激活只把权重读一遍。

    Args:
        w: 任意可接受存储形式的 ``[N, K]`` 投影权重。
        x: 激活，``[M, K]``。

    Returns:
        ``[M, N]``，dtype 同 ``x``。更大的 ``M`` 交给
        :func:`linear_prefill`，因为过了小 M 区间之后，即便 GEMM 每个 tile 都要
        重读权重，它在形状上也更便宜。

    Raises:
        RuntimeError: 扩展不可用。
    """
    if x.shape[0] > _ROW_CHUNK:
        return linear_prefill(w, x)
    return _gemv_rows(w, x)


def linear_m_verify(w: torch.Tensor | QWeight,
                    x: torch.Tensor) -> torch.Tensor:
    """为投机校验的 W8A8 投影选择带 LDS 重排的 DP4A 内核。

    Args:
        w: int8 ``QWeight`` 或其他普通投影权重。
        x: 激活，``[M, K]``。

    Returns:
        ``[M, N]``，dtype 同 ``x``。未启用开关或权重不是 int8 时沿用
        :func:`linear_m`。
    """
    if not _W8A8_DP4A_VERIFY or x.shape[0] < 2:
        return linear_m(w, x)
    if isinstance(w, QWeight):
        if w.k % 4:
            return linear_m(w, x)
        if w.wq.is_contiguous():
            wt, ws = w.wq, w.scale
        else:
            # 不连续的 nk 权重在这里补一次连续。dp4a 核本来吃 nk（``utrans = 0``）。
            wt, ws = w.wq.contiguous(), w.scale
    else:
        return linear_m(w, x)

    act = _compute_dtype(x).contiguous()
    rows, k = act.shape
    n = int(ws.numel())
    if rows < 2 or k % 4:
        return linear_m(w, x)

    ops = _w8a8_ops()
    stream = _stream()
    xq = torch.empty((rows, k), dtype=torch.int8, device=act.device)
    xs = torch.empty(rows, dtype=torch.float32, device=act.device)
    acc = torch.empty((rows, n), dtype=torch.int32, device=act.device)
    out = torch.empty((rows, n), dtype=act.dtype, device=act.device)
    ops.quant_act_i8(act, xq, xs, stream=stream)
    # 融合模式由这里的 dtype 决定：ABI 出口拿的是裸指针，读不到 ``out`` 的类型。
    ops.shim.dp4a_nk(_ptr(xq), _ptr(wt), _ptr(acc), _ptr(out), _ptr(xs), _ptr(ws),
                     1 if out.dtype == torch.bfloat16 else 2, int(n), int(rows), int(k),
                     int(n), stream)
    return _restore(out, x.dtype)


def _gemv_rows(w: torch.Tensor | QWeight,
               x: torch.Tensor) -> torch.Tensor:
    """用自写 GEMV 分块投影至多 :data:`_ROW_CHUNK` 行。

    单独一个辅助函数是因为三个入口都要它：调用方传了 batch 时的单行入口、小 M
    入口，以及绝不能碰库的宽 ``N`` 预填充路径。按 :data:`_ROW_CHUNK` 分块保证每次
    启动都落在一个内核真的有的特化上。

    Args:
        w: 任意可接受存储形式的 ``[N, K]`` 投影权重。
        x: 激活，``[C, K]``，``C >= 1``。

    Returns:
        ``[C, N]``，dtype 同 ``x``。
    """
    dtype = x.dtype
    act = _compute_dtype(x).contiguous()
    if isinstance(w, QWeight):
        # 同 :func:`linear`：``w.wq`` 就是内核要的 ``[N, K]``，一次转置都不做；分块
        # （``_ROW_CHUNK``）在 :func:`_gemv_nk` 里做，一次启动落在一个内核真的有的特化上。
        return _restore(_gemv_nk(w.wq, w.scale, act), dtype)
    # 稠密（bf16/fp32）：nk 权重交给库。原先这里是 ``gemv_kn`` / ``gemv_kn_m_split``
    # （torch 扩展），整套 kn 内核已删除；稠密 bf16 没有 nk GEMV，库的 ``N = 1`` 出口
    # 吃得了这个形状。
    return _restore(_gemm_nk_dense(w, act), dtype)

def shared_gate_scale(x: torch.Tensor, w: torch.Tensor, src: torch.Tensor,
                      out: torch.Tensor) -> torch.Tensor:
    """一次启动算 ``out = src * sigmoid(x · w)``，共享专家的门标量。

    参考链是 ``x.float() * w`` → ``sum`` → ``to`` → ``sigmoid`` → ``mul``：六次启动
    加两次分配，只为得到一个 fp32 标量。``out`` 由调用方按形状预分配并复用，
    所以这一步在稳态下不分配任何内存。

    Args:
        x: 激活，``[rows, width]`` 或 ``[width]``，最内层步长为 1。
        w: fp32 门权重，``[width]``，连续。
        src: 被缩放的输入，形状同 ``x``。
        out: 目的地，形状同 ``x``，允许与 ``src`` 重叠。

    Returns:
        ``out``，便于调用链串接。

    Raises:
        RuntimeError: 扩展不可用。
    """
    act = _compute_dtype(x)
    width = int(act.shape[-1])
    if x.dim() == 1:
        rows, x_pitch = 1, width
        s_pitch = o_pitch = width
    else:
        rows = int(act.reshape(-1, width).shape[0])
        x_pitch = int(act.reshape(-1, width).stride(0))
        s_pitch = int(src.reshape(-1, width).stride(0))
        o_pitch = int(out.reshape(-1, width).stride(0))
    _moe_ops().shim.shared_gate_scale(_ptr(act), _ptr(w), _ptr(src), _ptr(out), rows,
                                      width, x_pitch, s_pitch, o_pitch, _dcode(act),
                                      _stream() or 0)
    return out


def moe_grouped_i8(w, x: torch.Tensor, offsets: torch.Tensor,
                   slices: int = 0, scales: torch.Tensor | None = None,
                   out: torch.Tensor | None = None) -> torch.Tensor:
    """按专家分组做 int8 投影，一整栈权重在一次启动里读完。

    权重是 ``[E, N, K]``（nk 契约布局，也是**唯一**的布局）的 int8 加逐 (专家, 列) 的
    scale；``[E, K, N]`` 的栈与它的内核已经删除。int8 让 K 方向
    分片可以先把整数和相加、最后一次乘两个 scale，所以部分和是 int32（见
    ``kernels/moe_grouped_i8.cu``）。

    激活按行量化一次再交给内核：量化与「哪些行属于哪个专家」无关，因此不必按段各量化
    一次。约定与预填充的 ``quant_act_i8`` 完全一致（absmax/127，下限 1e-5），两处的
    量化误差口径才对得上。

    Args:
        w: int8 专家栈，:class:`~int8_weights.QWeightMoE`（``[E, N, K]`` int8 加
            ``[E, N]`` fp32）。
        x: 激活，``[tokens, K]``，已按专家号排序。bf16 时本函数就地量化；int8 时必须
            同时给出 ``scales``（上游已经量化过，例如 :func:`gather_quant_i8`、
            :func:`silu_mul_quant` 的输出），这样一次量化只做一次。
        offsets: ``[E+1]`` 前缀和，int32 或 int64；空段表示该专家没有行。
        slices: K 方向分片数；``<= 0`` 自动。
        scales: 逐行激活 scale，``[tokens]`` fp32；``x`` 已是 int8 时必填。
        out: 可选目的地，``[tokens, N]``；不给则分配。

    Returns:
        ``[tokens, N]``，dtype 同 ``x``（int8 输入时为 fp32）。

    Raises:
        ValueError: 激活或前缀和的形状与权重不自洽。
    """
    if x.dim() != 2 or x.shape[-1] != w.k:
        raise ValueError("grouped MoE expects [tokens, K] activations")
    if int(offsets.numel()) != len(w) + 1:
        raise ValueError(f"offsets must have {len(w) + 1} entries")
    tokens = x.shape[0]
    dtype = x.dtype
    if tokens == 0:
        return torch.empty((0, w.n), dtype=dtype, device=x.device)

    if x.dtype == torch.int8:
        # 上游刚量化过的情形（``gather_quant_i8`` / ``silu_mul_quant``）：直接用它的
        # int8 与 scale，量化一次只付一次。结果口径与在这里现量化逐位相同——两个量化
        # 核用的是同一套 absmax/127、下限 1e-5 与 ``__float2int_rn``。
        if scales is None:
            raise ValueError("int8 activations need their per-row scales")
        xq, xs = x, scales
        act_dtype = torch.float32
    else:
        act_dtype = x.dtype if x.dtype in (torch.bfloat16, torch.float32) else torch.float32
        act = x if x.dtype == act_dtype else x.float()
        if act.shape != (tokens, w.k) or not act.is_contiguous():
            act = act.reshape(tokens, w.k).contiguous()
        xq, xs = _moe_scratch(tokens, w.k, act.device)
        _w8a8_ops().quant_act_i8(act, xq, xs)
    # 内核覆盖每一行的每一列（网格的 y 维就是行数），所以这里用 ``empty`` 而不是
    # ``zeros``：一行清零的发射在 decode 一步里要付 80 次。
    #
    # ``out`` 的 dtype 由调用方定，不跟着激活走：int8 输入时可以指定 bf16 目的地
    # （下游核只吃 bf16，而 fp32 会把逐位口径改掉），内核在写回时才窄化。
    if out is None or out.shape != (tokens, w.n) or out.device != x.device \
            or out.dtype not in (torch.bfloat16, torch.float32):
        out = torch.empty((tokens, w.n), dtype=act_dtype, device=x.device)
    if offsets.dtype == torch.int32 and offsets.is_contiguous():
        bounds = offsets
    else:
        bounds = offsets.to(dtype=torch.int32, device=x.device).contiguous()
    # `w.nk_stack()` 是**零拷贝**（就是盘上那一份），专家栈只剩 nk 一种布局，所以
    # 这里不再有布局开关，也不会有任何整栈转置。
    rc = _moe_ops().shim.grouped_i8(_ptr(xq), _ptr(xs), _ptr(w.nk_stack()), _ptr(w.scale),
                                    _ptr(bounds), _ptr(out), 0, _dcode(out),
                                    int(tokens), int(w.k), int(w.n), len(w), int(slices),
                                    _stream() or 0)
    if rc:
        raise RuntimeError(f"coffin_moe_grouped_i8 rejected the geometry "
                           f"(tokens={tokens} K={w.k} N={w.n} E={len(w)} slices={slices}, "
                           f"code={rc})")
    return out if x.dtype == torch.int8 else _restore(out, dtype)


def moe_grouped_i8_expert(w, x: torch.Tensor, offsets: torch.Tensor, slices: int = 0,
                          scales: torch.Tensor | None = None,
                          out: torch.Tensor | None = None) -> torch.Tensor:
    """按专家分组做 int8 投影，**一个专家一趟**地把它那些行一起算完。

    与 :func:`moe_grouped_i8` 的算术逐位相同（int32 部分和、最后一次性乘激活与权重两个
    scale），差别只在并行切法：那个入口的块与激活行一一对应，一行一个块，于是一层
    805 MB 的专家权重会被按行重读。decode 每专家 1-2 行时那是最优；预填充一个 512 token
    的块摊到 256 个专家是每专家 16 行，权重流量放大 16 倍（实测单次 10.6 ms、700 GB/s，
    带宽已经打满，只是全花在重复字节上）。这里按专家切块，行在块内成趟复用权重。

    只接受 **nk 原生**专家栈（``[E, N, n]``，一列的 K 连续）：这条几何是一条线程认领
    一整列、一个 32 字节步铺满一个扇区；``[E, K, N]`` 的栈要先转置一份整模型的副本才
    谈得上，那条分支（连同内核）已经删除。

    Args:
        w: int8 专家栈，:class:`~int8_weights.QWeightMoE`（nk，``[E, N, K]``）。
        x: 激活，``[tokens, K]``，已按专家号排序；int8 时必须同时给出 ``scales``。
        offsets: ``[E+1]`` 前缀和，int32 或 int64。
        slices: 与 :func:`moe_grouped_i8` 同形保留，本形态忽略（并行度来自列块 x 专家）。
        scales: 逐行激活 scale，``[tokens]`` fp32。
        out: 可选目的地，``[tokens, N]``。

    Returns:
        ``[tokens, N]``，dtype 同 ``x``（int8 输入时为 fp32）。

    Raises:
        ValueError: 权重不是 nk 布局，或激活、前缀和与权重不自洽。
    """
    # 专家栈只有 nk 一种布局（kn 的 `[E, K, N]` 与它的内核已删除），`nk_stack()`
    # 零拷贝交出的就是 `[E, N, K]` 那一份，所以不再有布局分支要检查。
    if x.dim() != 2 or x.shape[-1] != w.k:
        raise ValueError("grouped MoE expects [tokens, K] activations")
    if int(offsets.numel()) != len(w) + 1:
        raise ValueError(f"offsets must have {len(w) + 1} entries")
    tokens = x.shape[0]
    dtype = x.dtype
    if tokens == 0:
        return torch.empty((0, w.n), dtype=dtype, device=x.device)

    if x.dtype == torch.int8:
        if scales is None:
            raise ValueError("int8 activations need their per-row scales")
        xq, xs = x, scales
        act_dtype = torch.float32
    else:
        act_dtype = x.dtype if x.dtype in (torch.bfloat16, torch.float32) else torch.float32
        act = x if x.dtype == act_dtype else x.float()
        if act.shape != (tokens, w.k) or not act.is_contiguous():
            act = act.reshape(tokens, w.k).contiguous()
        xq, xs = _moe_scratch(tokens, w.k, act.device)
        _w8a8_ops().quant_act_i8(act, xq, xs)

    if out is None or out.shape != (tokens, w.n) or out.device != x.device \
            or out.dtype not in (torch.bfloat16, torch.float32):
        out = torch.empty((tokens, w.n), dtype=act_dtype, device=x.device)
    if offsets.dtype == torch.int32 and offsets.is_contiguous():
        bounds = offsets
    else:
        bounds = offsets.to(dtype=torch.int32, device=x.device).contiguous()
    # ``nk_stack()`` 在 nk 原生栈上是零拷贝：直接就是 checkpoint 里那一份。
    _moe_ops().shim.grouped_i8_expert(_ptr(xq), _ptr(xs), _ptr(w.nk_stack()), _ptr(w.scale),
                                       _ptr(bounds), _ptr(out),
                                       1 if out.dtype == torch.bfloat16 else 0, int(tokens),
                                       int(w.k), int(w.n), len(w), _stream())
    return out if x.dtype == torch.int8 else _restore(out, dtype)


def moe_grouped_i8_tcu(w, x: torch.Tensor, offsets: torch.Tensor, slices: int = 0,
                       scales: torch.Tensor | None = None,
                       out: torch.Tensor | None = None) -> torch.Tensor:
    """按专家分组、但把专家 GEMM 交给**张量核**的 int8 投影。

    与 :func:`moe_grouped_i8_expert` 的算术逐位相同，差别在算力：同一张卡上同一形状
    （4096 x 2048 -> 1024）实测 ``moe_grouped_i8_expert`` 8.99 ms（0.96 TMAC/s，dp4a
    的天花板），而张量核这条 0.446 ms（19.2 TMAC/s）。PP2 预填充要
    4000 tps，每卡需要约 2 TMAC/s，所以预填充必须走这条。

    nk 原生栈在这里**不需要转置**：内核把两个操作数互换（mat-a 吃激活、mat-b 吃权重），
    于是 ``[E, N, K]`` 里「固定 n 的 4 个相邻 k 连续」正好是 mat-b 要的字。代价是激活
    要按 k 主序再摆一份，即下面那次 ``t().contiguous()``。

    Args:
        w: int8 专家栈，:class:`~int8_weights.QWeightMoE`（nk，``[E, N, K]``）。
        x: 激活，``[tokens, K]``，已按专家号排序；int8 时必须同时给出 ``scales``。
        offsets: ``[E+1]`` 前缀和，int32 或 int64。
        slices: 保留同形，本形态忽略。
        scales: 逐行激活 scale，``[tokens]`` fp32。
        out: 可选目的地，``[tokens, N]``。

    Returns:
        ``[tokens, N]``，dtype 同 ``x``。

    Raises:
        ValueError: 权重不是 nk 布局、K 不是 64 的整数倍，或**每专家不足 4 行**——
            后者是内核的已知缺陷（见下），在这里挡住，调用方按 ``ValueError`` 退回
            :func:`moe_grouped_i8` 的 GEMV 结构。
    """
    # 同上：只有 nk 一种专家栈，没有布局分支要检查。
    if w.k % 64 != 0:
        raise ValueError("moe_grouped_i8_tcu needs K to be a multiple of 64")
    if x.dim() != 2 or x.shape[-1] != w.k:
        raise ValueError("grouped MoE expects [tokens, K] activations")
    if int(offsets.numel()) != len(w) + 1:
        raise ValueError(f"offsets must have {len(w) + 1} entries")
    tokens = x.shape[0]
    dtype = x.dtype
    if tokens == 0:
        return torch.empty((0, w.n), dtype=dtype, device=x.device)

    if x.dtype == torch.int8:
        if scales is None:
            raise ValueError("int8 activations need their per-row scales")
        xq, xs = x, scales
        act_dtype = torch.float32
    else:
        act_dtype = x.dtype if x.dtype in (torch.bfloat16, torch.float32) else torch.float32
        act = x if x.dtype == act_dtype else x.float()
        if act.shape != (tokens, w.k) or not act.is_contiguous():
            act = act.reshape(tokens, w.k).contiguous()
        xq, xs = _moe_scratch(tokens, w.k, act.device)
        _w8a8_ops().quant_act_i8(act, xq, xs)

    if out is None or out.shape != (tokens, w.n) or out.device != x.device \
            or out.dtype not in (torch.bfloat16, torch.float32):
        out = torch.empty((tokens, w.n), dtype=act_dtype, device=x.device)
    if offsets.dtype == torch.int32 and offsets.is_contiguous():
        bounds = offsets
    else:
        bounds = offsets.to(dtype=torch.int32, device=x.device).contiguous()
    # 复用 ``[k, tokens]`` 转置缓冲，避免预填充每层 2 次 × 40 层重复分配。
    at = _moe_at_scratch(tokens, int(w.k), x.device)
    ops = _moe_ops().shim
    stream = _stream() or 0
    rc = ops.transpose_i8(_ptr(xq), _ptr(at), int(tokens), int(w.k), stream)
    if rc:
        raise RuntimeError(f"coffin_transpose_i8 rejected the geometry "
                           f"(rows={tokens} K={w.k}, code={rc})")
    rc = ops.grouped_i8_tcu(_ptr(at), _ptr(xs), _ptr(w.nk_stack()), _ptr(w.scale),
                            _ptr(bounds), _ptr(out), _dcode(out),
                            int(tokens), int(w.k), int(w.n), len(w), stream)
    if rc:
        raise RuntimeError(f"coffin_moe_grouped_i8_tcu rejected the geometry "
                           f"(rows={tokens} K={w.k} N={w.n} E={len(w)}, code={rc})")
    return out if x.dtype == torch.int8 else _restore(out, dtype)


def moe_gather_sum(hid: torch.Tensor, slot_pos: torch.Tensor,
                   weights: torch.Tensor, tokens: int, top_k: int,
                   out: torch.Tensor | None = None) -> torch.Tensor:
    """按槽位收集专家输出、乘路由权重并在 fp32 中定序累加到 ``[tokens, hidden]``。"""
    rows, hidden = int(hid.shape[0]), int(hid.shape[1])
    if out is None or out.shape != (tokens, hidden) or out.device != hid.device \
            or out.dtype != hid.dtype:
        out = torch.empty((tokens, hidden), dtype=hid.dtype, device=hid.device)
    if tokens == 0:
        return out
    hid_c = hid if hid.is_contiguous() else hid.contiguous()
    sp = (slot_pos if slot_pos.dtype == torch.int32 and slot_pos.is_contiguous()
          else slot_pos.to(dtype=torch.int32, device=hid.device).contiguous())
    rw = weights if weights.is_contiguous() else weights.contiguous()
    if rw.dtype not in (torch.bfloat16, torch.float32):
        rw = rw.float()
    _moe_ops().shim.moe_gather_sum(_ptr(hid_c), _ptr(sp), _ptr(rw), _ptr(out),
                                   _dcode(hid_c), _dcode(rw), int(tokens), int(top_k),
                                   int(hidden), int(rows), _stream() or 0)
    return out


def gather_quant_i8(x: torch.Tensor, index: torch.Tensor,
                    out_q: torch.Tensor | None = None,
                    out_s: torch.Tensor | None = None):
    """按索引表取行，并在同一次启动里完成激活量化。

    原本是 ``index_select`` 加 ``quant_act_i8`` 两次发射：前者还要为 ``[rows, K]``
    的中间结果付一次写回（EP4 的 MoE 上就是每层 64 x 2048 的 bf16 往返），后者再把它
    读回来算 absmax。合起来之后中间结果不出共享内存，发射也少一次。

    Args:
        x: 源激活，``[tokens, K]`` 连续。
        index: ``[rows]`` 的行号表，int32；``rows`` 是取出的行数（MoE 里是
            ``tokens * top_k``），元素可重复也可越序。
        out_q: 可选目的地，``[rows, K]`` int8；不给时用分组 MoE 的激活暂存。
        out_s: 可选目的地，``[rows]`` fp32；不给时用分组 MoE 的激活暂存。

    Returns:
        ``(xq, xs)``：``[rows, K]`` int8 与 ``[rows]`` fp32。量化口径与
        :func:`quant_act_i8` 逐位相同（absmax/127、下限 1e-5、``__float2int_rn``）。

    Raises:
        ValueError: 源激活不是二维连续张量，或索引表的 dtype 不是 int32。
    """
    if x.dim() != 2 or not x.is_contiguous():
        raise ValueError("gather_quant_i8 expects a contiguous [tokens, K] source")
    if index.dtype != torch.int32:
        raise ValueError("gather_quant_i8 needs an int32 index table")
    rows = int(index.numel())
    if out_q is None or out_s is None:
        out_q, out_s = _moe_scratch(rows, x.shape[1], x.device)
    return _w8a8_ops().gather_quant_i8(x, index, rows=rows, out_q=out_q, out_s=out_s,
                                       stream=_stream())


def silu_mul_quant(gate: torch.Tensor, up: torch.Tensor,
                   out: torch.Tensor | None = None,
                   out_q: torch.Tensor | None = None,
                   out_s: torch.Tensor | None = None):
    """``silu(gate) * up`` 落成 bf16，并顺手量化成 int8。

    替掉 ``silu_mul`` 加 ``quant_act_i8`` 两次发射：乘积只算一遍，amax 直接在寄存器里
    归约，不必把 bf16 结果写回再读回来。

    舍入口径与 ``silu_mul`` 相同（fp32 算 sigmoid、窄化一次、乘积再窄化一次），
    amax 按**落盘的 bf16** 算，所以与「先 ``silu_mul`` 再 ``quant_act_i8``」逐位相同。

    Args:
        gate: ``[rows, K]``，已按专家号排序的门投影。
        up: 同形，同上投影。
        out: 可选目的地，``[rows, K]`` bf16；不给时新建。
        out_q: 可选目的地，``[rows, K]`` int8；不给时用分组 MoE 的激活暂存。
        out_s: 可选目的地，``[rows]`` fp32；不给时用分组 MoE 的激活暂存。

    Returns:
        ``(product, xq, xs)``：bf16 乘积、int8 激活与逐行 fp32 scale。

    Raises:
        ValueError: 两个输入形状不一致。
    """
    if gate.shape != up.shape or gate.dim() != 2:
        raise ValueError("silu_mul_quant needs two same-shaped [rows, K] inputs")
    rows, width = int(gate.shape[0]), int(gate.shape[1])
    if out is None:
        out = torch.empty_like(gate)
    if out_q is None or out_s is None:
        out_q, out_s = _moe_scratch(rows, width, gate.device)
    _attn_ops().shim.silu_mul_quant(_ptr(gate), _ptr(up), _ptr(out), _ptr(out_q),
                                    _ptr(out_s), rows, width, int(gate.stride(0)),
                                    int(up.stride(0)), int(out.stride(0)),
                                    _dcode(gate), _stream() or 0)
    return out, out_q, out_s


def moe_grouped_i8_merge(w, x: torch.Tensor, offsets: torch.Tensor,
                         row_weight: torch.Tensor,
                         slices: int = 0) -> torch.Tensor:
    """按专家分组做 int8 投影，并把各行按路由权重加权累加成一行。

    decode 时一个 token 激活 top_k 个专家，调用方要的是它们加权后的和而不是这 top_k
    行本身。把这一步并进归约核，省掉 ``[top_k, N]`` 的激活缓冲与 ``mul`` /
    ``index_add_`` / ``sum`` 三次发射：实测那几步 0.184 ms，比 down 的投影本身
    （0.059 ms）还贵。

    Args:
        w: int8 专家栈，``[E, K, N]`` 加 ``[E, N]`` fp32 scale。
        x: 激活，``[rows, K]``，已按专家号排序。
        offsets: ``[E+1]`` 前缀和。
        row_weight: ``[rows]`` 的逐行路由权重。

    Returns:
        ``[1, N]``，dtype 同 ``x``。

    Raises:
        ValueError: 激活行数、前缀和长度或逐行权重与权重栈不自洽。
    """
    if x.dim() != 2 or x.shape[-1] != w.k:
        raise ValueError("grouped MoE expects [rows, K] activations")
    rows = int(x.shape[0])
    if int(offsets.numel()) != len(w) + 1:
        raise ValueError("offsets must have " + str(len(w) + 1) + " entries")
    dtype = x.dtype
    act = x if x.dtype in (torch.bfloat16, torch.float32) else x.float()
    if act.shape != (rows, w.k) or not act.is_contiguous():
        act = act.reshape(rows, w.k).contiguous()
    if row_weight.dtype != torch.float32 or not row_weight.is_contiguous():
        row_weight = row_weight.to(dtype=torch.float32,
                                   device=act.device).contiguous()
    if int(row_weight.numel()) != rows:
        raise ValueError("row_weight needs one entry per row")
    xq, xs = _moe_scratch(rows, w.k, act.device)
    if offsets.dtype == torch.int32 and offsets.is_contiguous():
        bounds = offsets
    else:
        bounds = offsets.to(dtype=torch.int32, device=act.device).contiguous()
    out = torch.empty((1, w.n), dtype=act.dtype, device=act.device)
    _w8a8_ops().quant_act_i8(act, xq, xs)
    stack = w.nk_stack()
    rc = _moe_ops().shim.merge(_ptr(xq), _ptr(xs), _ptr(stack), _ptr(w.scale),
                               _ptr(bounds), _ptr(row_weight), _ptr(out), _dcode(out),
                               int(rows), int(w.k), int(w.n), len(w), int(slices),
                               _stream() or 0)
    if rc:
        raise RuntimeError(f"coffin_moe_grouped_i8_merge rejected the geometry "
                           f"(rows={rows} K={w.k} N={w.n} E={len(w)} slices={slices}, "
                           f"code={rc})")
    return _restore(out, dtype)


def moe_grouped_i8_merge_tokens(w, x: torch.Tensor, offsets: torch.Tensor,
                                row_weight: torch.Tensor,
                                slot_pos: torch.Tensor, slot_of_pos: torch.Tensor,
                                tokens: int, top_k: int,
                                scales: torch.Tensor | None = None,
                                out: torch.Tensor | None = None,
                                slices: int = 0) -> torch.Tensor:
    """归并形态的多流版本：每个 token 只累加自己那 ``top_k`` 行。

    与 :func:`moe_grouped_i8_merge` 是同一个数学，区别在于三件外层原本要先做的事现在
    都在核里：路由权重按槽位号直接寻址（不再 ``index_select`` 成逐行表、也不再转 fp32）、
    专家并行的有效行掩码由内核按 ``offsets[E]`` 自己判（不再算 ``arange < valid`` 再相乘）、
    多流的累加写进 ``[tokens, N]``（不再 ``zero_`` / ``mul`` / ``index_add_`` 三步）。

    Args:
        w: int8 专家栈，``[E, K, N]`` 加 ``[E, N]`` fp32 scale。
        x: 激活，``[rows, K]``，已按专家号排序。bf16 时本函数就地量化；int8 时必须同时
            给出 ``scales``。
        offsets: ``[E+1]`` 前缀和。
        row_weight: ``[rows]`` 的路由权重，bf16 或 fp32，第 ``r`` 行对应槽位
            ``slot_of_pos[r]``。
        slot_pos: ``[rows]``，槽位 → 行号，路由到别的 rank 的槽位为 ``-1``。
        slot_of_pos: ``[rows]``，行号 → 槽位。
        tokens: 解码行数，即 ``rows / top_k``。``1`` 时走行序，与
            :func:`moe_grouped_i8_merge` 逐位相同。
        top_k: 每 token 的专家数。
        scales: 逐行激活 scale，``[rows]`` fp32；``x`` 已是 int8 时必填。
        out: 可选目的地，``[tokens, N]`` bf16 或 fp32；不给则分配。
        slices: K 方向分片数；``<= 0`` 自动。

    Returns:
        ``[tokens, N]``，dtype 同 ``x``（int8 输入时为 fp32 的同形结果）。

    Raises:
        ValueError: 激活行数、权重形状或两张位置表的长度与 ``tokens * top_k`` 不自洽。
    """
    if x.dim() != 2 or x.shape[-1] != w.k:
        raise ValueError("grouped MoE expects [rows, K] activations")
    rows = int(x.shape[0])
    if int(offsets.numel()) != len(w) + 1:
        raise ValueError("offsets must have " + str(len(w) + 1) + " entries")
    if rows != tokens * top_k:
        raise ValueError(f"{rows} activation rows do not split into {tokens} x {top_k}")
    if int(slot_pos.numel()) != rows or int(slot_of_pos.numel()) != rows:
        raise ValueError("slot tables need one entry per row")
    if int(row_weight.numel()) != rows:
        raise ValueError("row_weight needs one entry per row")
    if rows == 0:
        n = w.n
        return out if out is not None else torch.empty(
            (tokens, n), dtype=torch.float32, device=x.device)
    dtype = x.dtype
    if x.dtype == torch.int8:
        if scales is None:
            raise ValueError("int8 activations need their per-row scales")
        xq, xs = x, scales
        act_dtype = torch.float32
    else:
        act_dtype = x.dtype if x.dtype in (torch.bfloat16, torch.float32) else torch.float32
        act = x if x.dtype == act_dtype else x.float()
        if not act.is_contiguous():
            act = act.reshape(rows, w.k).contiguous()
        xq, xs = _moe_scratch(rows, w.k, act.device)
        _w8a8_ops().quant_act_i8(act, xq, xs)
    if not row_weight.is_contiguous():
        row_weight = row_weight.contiguous()
    if row_weight.dtype not in (torch.bfloat16, torch.float32):
        row_weight = row_weight.float()
    if offsets.dtype == torch.int32 and offsets.is_contiguous():
        bounds = offsets
    else:
        bounds = offsets.to(dtype=torch.int32, device=x.device).contiguous()
    if slot_pos.dtype != torch.int32 or not slot_pos.is_contiguous():
        slot_pos = slot_pos.to(dtype=torch.int32, device=x.device).contiguous()
    if slot_of_pos.dtype != torch.int32 or not slot_of_pos.is_contiguous():
        slot_of_pos = slot_of_pos.to(dtype=torch.int32, device=x.device).contiguous()
    # ``out`` 的 dtype 由调用方定，与 :func:`moe_grouped_i8` 同一条理由：int8 输入时
    # 上游给的可能是 bf16 目的地，而 fp32 会一路漏到 EP 的 p2p 归约上（那个核只收
    # bf16，报「p2p combine needs bf16, got torch.float32」）。
    if out is None or out.shape != (tokens, w.n) or out.device != x.device \
            or out.dtype not in (torch.bfloat16, torch.float32):
        out = torch.empty((tokens, w.n), dtype=act_dtype, device=x.device)
    rc = _moe_ops().shim.merge_tokens(_ptr(xq), _ptr(xs), _ptr(w.nk_stack()), _ptr(w.scale),
                                      _ptr(bounds), _ptr(row_weight), _ptr(slot_pos),
                                      _ptr(slot_of_pos), _ptr(out), _dcode(out), _dcode(row_weight),
                                      int(rows), int(w.k), int(w.n), len(w), int(tokens),
                                      int(top_k), int(slices), _stream() or 0)
    if rc:
        raise RuntimeError(f"coffin_moe_grouped_i8_merge_tokens rejected the geometry "
                           f"(rows={rows} K={w.k} N={w.n} E={len(w)} tokens={tokens} "
                           f"top_k={top_k} slices={slices}, code={rc})")
    return out if x.dtype == torch.int8 else _restore(out, dtype)
#: ``(tokens, k, device)`` → ``(xq, xs)`` 的分组 MoE 激活暂存。
#:
#: 一步 decode 里 ``moe_grouped`` 被调 96 次（48 层 x gate_up/down），原来每次新建
#: ``xq``/``xs``/``out`` 三个张量 = 288 次分配。这里只缓存前两个：它们是纯暂存，
#: 从不返回、也没有别名；``out`` 必须每次新建（见上）。
#:
#: 单线程引擎主循环（``llm_engine`` 单线程驱动），不需要锁。条数上限是防御性的。
_MOE_SCRATCH: dict = {}
_MOE_SCRATCH_MAX = 16


def _moe_scratch(tokens: int, k: int, device: torch.device):
    """取（必要时分配）分组 MoE 的激活暂存。

    Args:
        tokens: 行数。
        k: 激活宽度。
        device: 目标设备。

    Returns:
        ``(xq, xs)``：``[tokens, k]`` 的 int8 与 ``[tokens]`` 的 fp32。
    """
    key = (tokens, k, device)
    pair = _MOE_SCRATCH.get(key)
    if pair is None:
        if len(_MOE_SCRATCH) >= _MOE_SCRATCH_MAX:
            _MOE_SCRATCH.clear()
        pair = (torch.empty((tokens, k), dtype=torch.int8, device=device),
                torch.empty(tokens, dtype=torch.float32, device=device))
        _MOE_SCRATCH[key] = pair
    return pair

def moe_dispatch(expert_idx: torch.Tensor, local_table: torch.Tensor,
                 slots: int, local_experts: int, num_experts: int,
                 top_k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                      torch.Tensor]:
    """一次内核算出 MoE 分组要的 ``(offsets, slot_pos, token_of_pos, slot_of_pos)``。

    替掉模型侧原本的一串 torch 小算子。作业 ``q4attrib3`` 按调用点量到，一步 decode
    里这段在 **48 层 × 每层** 各跑一遍：``topk`` / ``div`` / ``sum`` / ``to`` /
    ``nonzero`` / ``flatten`` / ``div`` / ``argsort`` / ``bincount`` / ``cumsum`` /
    ``chunk`` / ``mul`` / ``mul`` / ``to`` = **14 个内核,合计 672 次/步**。
    按本卡 5.76 us 的发射地板（``q4floor1``）是 **3.87 ms/步**,而这些算术的输入
    只有 ``slots`` 个整数（decode 时 **10** 个）。

    Args:
        expert_idx: ``[slots]`` 展平的全局专家号,int32 连续。
        local_table: ``[num_experts]`` 全局号 → 本地号,int32,不归本 rank 为 -1。
        slots: 槽位数 ``tokens * top_k``。
        local_experts: 本 rank 的专家数。
        num_experts: 全局专家数。
        top_k: 每 token 的专家数,用来从槽位号反推 token 号。

    Returns:
        ``(offsets, slot_pos, token_of_pos, slot_of_pos)``：
        ``offsets`` 是 ``[local_experts + 1]`` 前缀和；``slot_pos`` 是 ``[slots]``
        收拢后的行号（丢弃的为 -1）；``token_of_pos`` / ``slot_of_pos`` 是
        ``[slots]`` 的「收拢后第 p 行来自哪个 token / 哪个槽位」。
        **有效行数之外的尾部被清 0**,所以两张位置表可以整块拿去索引。

    Raises:
        ValueError: 专家数或槽位数与 ``local_table`` 不自洽。
        RuntimeError: 扩展不可用。
    """
    if local_table.dtype != torch.int32:
        local_table = local_table.to(torch.int32)
    device = expert_idx.device
    offsets = torch.empty(local_experts + 1, dtype=torch.int32, device=device)
    slot_pos = torch.empty(slots, dtype=torch.int32, device=device)
    token_of_pos = torch.empty(slots, dtype=torch.int32, device=device)
    slot_of_pos = torch.empty(slots, dtype=torch.int32, device=device)
    scratch = _disp_scratch(local_experts, device)
    # ``expert_idx`` 不窄化：出口按 dtype 选模板（int32 / int64 各一条），读 int64 与
    # 读 int32 在只有 ``slots`` 个元素的分派核上没有可测差别，而窄化是一次独立发射
    # （实测 11 us/层）。
    if expert_idx.dtype not in (torch.int32, torch.int64):
        expert_idx = expert_idx.to(torch.int32)
    _moe_ops().shim.moe_dispatch(_ptr(expert_idx), _ptr(local_table), _ptr(offsets),
                                 _ptr(slot_pos), _ptr(token_of_pos), _ptr(slot_of_pos),
                                 _ptr(scratch), slots, local_experts, num_experts,
                                 int(top_k), _dcode(expert_idx), _stream() or 0)
    return offsets, slot_pos, token_of_pos, slot_of_pos


#: ``(local_experts, device)`` → 分派暂存。一步 48 层共用一块,不必每层分配。
_DISP_SCRATCH: dict = {}


def _disp_scratch(local_experts: int, device: torch.device) -> torch.Tensor:
    """取（必要时分配）分派内核的逐桶游标暂存。

    Args:
        local_experts: 桶数。
        device: 目标设备。

    Returns:
        ``[local_experts]`` 的 int32 张量。
    """
    key = (local_experts, device)
    buf = _DISP_SCRATCH.get(key)
    if buf is None:
        buf = torch.empty(local_experts, dtype=torch.int32, device=device)
        _DISP_SCRATCH[key] = buf
    return buf


def moe_router(logits: torch.Tensor, top_k: int,
               normalize: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """融合的 ``softmax -> top-k -> 归一化``，一次发射写完下标与权重。

    替掉 ``models/qwen4exp/moe.py`` 里 ``TopKRouter`` 那条 torch 链
    ``softmax -> topk -> div -> sum -> to``。作业 ``q4attrib3`` 量到它在 decode 一步里
    是 48 层 × 5 个内核，其中 ``aten::topk`` 单次 38 us（``[1, 512]`` 取 top-10 在本卡上
    走的是 bitonic 排序网络，45 轮比较交换）——挑最大的 10 个不需要排序，理由见
    ``kernels/moe_grouped.cu`` 里那个内核上方的注释。

    **返回的两个张量每次新建**，不做缓存复用：它们在本层结束后就被丢掉，但「返回值
    不被别人持有」这件事没有写在契约里，而 ``moe_grouped`` 的 ``out`` 已经因为同一条
    理由不复用。两次小分配比一次静默的别名事故便宜。

    **近似在哪、影响多大**（这一轮的精度政策是「只要输出正常」，所以要能一眼看到边界）：
    ``__expf`` 相对误差 ≤ 2e-6、softmax 分母的求和换序 ≤ 3e-5 最坏 / ~1e-6 典型、归一化
    少一次除法 ≤ 1.2e-7 相对 —— 三者都远小于权重落地时的 bf16 量化步长 3.9e-3，不可观测。
    唯一会改变最终选择的是**等值**：两个概率完全相等时这里取小下标，而 ``torch.topk``
    的次序不保证；两个选择数学上都对（等值就是同一个数，权重逐位相同），换来的是同一输入
    两次运行逐位相同。**这一档不是罕见情形**：实测 bf16 randn logits 上 ``tokens=512`` 时
    340/512 行（66.4%）的 top-11 内有完全等值（top-10 挤在 2.5-3.5，而 bf16 在那里只有约
    50 个位模式），fp32 对照 0/5120。所以**验收不能按位置比下标**，要比值（``w_set_diff``
    非 0 才是真错误）；详见 ``kernels/moe_grouped.cu`` 里那段。

    Args:
        logits: ``[tokens, num_experts]`` 的 bf16 或 fp32 logits。
        top_k: 每 token 选的专家数。
        normalize: 是否把 top-k 概率再归一化（``norm_topk_prob``）。

    Returns:
        ``(index, weight)``：前者 ``[tokens, top_k]`` int64 按概率降序，后者同形、
        dtype 同 ``logits``。

    Raises:
        ValueError: 激活不是二维，或 ``top_k`` / dtype 超出内核的实例化范围。此时没有
            可用的融合路径，报错比悄悄换回慢路径好——后者会让「为什么这一步变慢了」
            只能靠计时器发现。
    """
    if logits.dim() != 2:
        raise ValueError(
            f"router logits must be 2-D [tokens, experts], got {tuple(logits.shape)}.")
    tokens, experts = int(logits.shape[0]), int(logits.shape[1])
    # 内核的 top_k 上限是 32：一个 32-lane 组里每个 lane 只保存**自己那一槽**的结果，
    # 超过它就得用动态下标的寄存器数组，那会掉进 local memory，反而更慢。
    if top_k <= 0 or top_k > 32 or top_k > experts:
        raise ValueError(
            f"fused router needs 0 < top_k <= min(32, {experts}), got {top_k}.")
    if logits.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(f"fused router needs bf16 or fp32 logits, got {logits.dtype}.")
    if not logits.is_contiguous():
        # 内核按 ``token * experts`` 取行，带行间距的视图会读错行（出口也会拒）。
        logits = logits.contiguous()
    index = torch.empty((tokens, top_k), dtype=torch.int64, device=logits.device)
    weight = torch.empty((tokens, top_k), dtype=logits.dtype, device=logits.device)
    if tokens == 0:
        return index, weight
    _moe_ops().shim.moe_router(_ptr(logits), _ptr(index), _ptr(weight), tokens, experts,
                               int(top_k), 1 if normalize else 0, _dcode(logits),
                               _stream() or 0)
    return index, weight


def moe_router_dispatch(logits: torch.Tensor, index: torch.Tensor,
                        weight: torch.Tensor, layout: torch.Tensor) -> None:
    """对一行 logits 做归一化路由，向调用方的连续缓冲写入下标、权重和四张分派表。"""
    if logits.ndim != 2 or logits.shape[0] != 1:
        raise ValueError("moe_router_dispatch expects logits shaped [1, experts]")
    experts = logits.shape[1]
    top_k = index.numel()
    if not 0 < top_k <= min(experts, 32) or experts > 1024:
        raise ValueError(f"invalid router geometry: experts={experts}, top_k={top_k}")
    if logits.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(f"router needs bf16 or fp32 logits, got {logits.dtype}")
    if (index.shape != (1, top_k) or weight.shape != index.shape
            or layout.shape != (experts + 1 + 3 * top_k,)
            or index.dtype != torch.int64 or weight.dtype != logits.dtype
            or layout.dtype != torch.int32):
        raise ValueError("invalid router output shape or dtype")
    if any(t.device != logits.device or not t.is_contiguous() for t in (index, weight, layout)):
        raise ValueError("router outputs must be contiguous and on the logits device")
    logits = logits.contiguous()
    _moe_ops().shim.moe_router_dispatch(_ptr(logits), _ptr(index), _ptr(weight), _ptr(layout),
                                        experts, top_k, _dcode(logits), _stream() or 0)


def attn_sparse(q: torch.Tensor, kcache: torch.Tensor, vcache: torch.Tensor,
                sel_idx: torch.Tensor, sel_len: torch.Tensor,
                ctx: torch.Tensor, out: torch.Tensor | None = None,
                scale: float | None = None) -> torch.Tensor:
    """QSA 的稀疏注意力：只读 ``sel_idx`` 指出的位置，KV 不物化。

    替掉 ``qsa.py`` 里那三段纯 torch（``repeat_kv`` + ``matmul`` + ``softmax`` +
    ``matmul``）。作业 ``q4attn1`` 量到它们在 kv=32768 时是 16.68 ms/层/步，其中
    ``matmul`` 占 55%，而它扫的是整个 ``kv_length`` —— 索引器预算只有 2048，所以
    94% 的打分作用在已经被判定为「不看」的位置上。

    Args:
        q: ``[rows, nh, hd]`` bf16，**未 repeat 的查询头**，已旋转。
        kcache: ``[nkv, max_seq, hd]`` bf16，头主序、**未 repeat**。
        vcache: 同 ``kcache``。
        sel_idx: ``[rows, max_sel]`` int32，行内升序，无效槽为 -1。
        sel_len: ``[rows]`` int32 每行有效位置数。
        ctx: ``[rows]`` int32 每行因果上界；``>= ctx`` 的位置被跳过。
        out: 可选输出缓冲；不给则分配。
        scale: 打分缩放；``None`` 时取 ``hd ** -0.5``，与 ``Attention.scaling`` 一致。

    Returns:
        ``[rows, nh, hd]`` bf16。

    Raises:
        ValueError: 形状与权重不自洽。
        RuntimeError: 扩展不可用。
    """
    module = _attn_ops().shim
    if q.dim() != 3:
        raise ValueError("attn_sparse expects q as [rows, nh, hd]")
    rows, nh, hd = q.shape
    if out is None:
        out = torch.empty((rows, nh, hd), dtype=q.dtype, device=q.device)
    if scale is None:
        scale = hd ** -0.5
    if sel_idx.dtype != torch.int32:
        sel_idx = sel_idx.to(torch.int32)
    if sel_len.dtype != torch.int32:
        sel_len = sel_len.to(torch.int32)
    if ctx.dtype != torch.int32:
        ctx = ctx.to(torch.int32)
    # 几何全给裸指针出口：ABI 面没有形状，``rows`` 只能由调用方从 q 读出来。
    module.attn_sparse(_ptr(q), _ptr(kcache), _ptr(vcache), _ptr(sel_idx), _ptr(sel_len),
                       _ptr(ctx), _ptr(out), float(scale), int(rows), int(nh),
                       int(kcache.shape[0]), int(hd), int(kcache.shape[1]),
                       int(sel_idx.shape[-1]), _stream())
    return out


def qsa_block_keys(flat_keys: torch.Tensor, table: torch.Tensor,
                   weight: torch.Tensor, flat_cos: torch.Tensor,
                   flat_sin: torch.Tensor, *, seq: int, tokens: int,
                   rotary_dim: int, eps: float = 1e-6,
                   out: torch.Tensor | None = None) -> torch.Tensor:
    """QSA 块 key：gather → 池化 → RMSNorm → partial RoPE，一次启动。

    替掉 ``qsa.py`` 的 ``_batched_block_keys`` 那串 torch 算子。作业 ``q4sel1``
    量到那一段在 kv=32768 时是 1.067 ms/层/步（12 层 12.8 ms），而它要读的数据只有
    ``max_blocks x ratio x head_dim`` 个 bf16 = 8.4 MB —— 按 575 GB/s 的屋顶应是
    0.015 ms，差 71 倍。差在中间量上：gather、``.float()`` 的 fp32 池化、layernorm
    的多次 elementwise、RoPE 的 ``cat``，每一步都是一次全量读写。

    Meta 里那块 ``[blk * max_blocks, ratio, head_dim]`` 的 fp32 临时量也不再有，
    所以 prefill 的显存峰值会跟着降（见 ``_POOL_ROW_CHUNK`` 的历史）。

    Args:
        flat_keys: ``[batch * tokens, head_dim]`` bf16。
        table: ``[rows, max_blocks, ratio]`` int32，``-1`` 是补位槽。
        weight: ``[head_dim]`` bf16，**零中心**增益（内核会乘 ``1 + w``）。
        flat_cos: ``[batch * tokens, rotary_dim]`` bf16。
        flat_sin: 同 ``flat_cos``。
        seq: 每个 batch 行的 query 数，用来把块行映射回 batch 行。
        tokens: 每行的 token 数。
        rotary_dim: 参与旋转的维数（前若干维），其余原样输出。
        eps: RMSNorm 的 epsilon。
        out: 可选输出缓冲；不给则分配。

    Returns:
        ``[rows, max_blocks, head_dim]`` bf16。

    Raises:
        ValueError: 形状不自洽或 ``ratio`` 超出内核上限。
        RuntimeError: 扩展不可用。
    """
    module = _attn_ops().shim
    if table.dim() != 3:
        raise ValueError("table must be [rows, max_blocks, ratio]")
    rows, max_blocks, _ratio = table.shape
    head_dim = flat_keys.shape[1]
    if out is None:
        out = torch.empty((rows, max_blocks, head_dim), dtype=flat_keys.dtype,
                          device=flat_keys.device)
    if table.dtype != torch.int32:
        table = table.to(torch.int32)
    # 块行 → query 行 → batch 行。两条都只是 int32 除法，但内核要它们连续。
    blk_total = rows * max_blocks
    arange = torch.arange(blk_total, device=table.device, dtype=torch.int32)
    block_row = torch.div(arange, max_blocks, rounding_mode="floor")
    block_batch = torch.div(block_row, int(seq), rounding_mode="floor")
    module.qsa_block_keys(_ptr(flat_keys), _ptr(table), _ptr(weight), _ptr(flat_cos),
                          _ptr(flat_sin), _ptr(block_row), _ptr(block_batch), _ptr(out),
                          int(rows), int(seq), int(tokens), int(max_blocks), int(_ratio),
                          int(head_dim), int(rotary_dim), float(eps), _stream())
    return out


def _prefill_nk_tcu(w: QWeight, x: torch.Tensor):
    """把一个 nk 投影当成「只有一个专家」交给分组张量核 GEMM。

    为什么值得单开一条：``linear_prefill`` 原来一律先 ``pack.as_kn(w)``，对 nk 原生的
    ``QWeight`` 这意味着为这个权重转置一整份 ``[K, N]`` 副本——4B 是 4.3 GB，Occamy 是
    35 GB，而卡只有 31.7 GiB，两份布局根本放不下。分组内核本来就是 nk 原生（mat-b
    吃的就是「固定 n 的 4 个相邻 k」），所以 ``E = 1``、``offsets = [0, rows]`` 正好
    表达「整个投影一次算完」，一次转置都不用做。

    Args:
        w: nk 原生的 int8 投影，``wq`` 是连续 ``[N, K]``。
        x: 激活，``[rows, K]``。

    Returns:
        ``[rows, N]``，dtype 同 ``x``；内核缺失或短段守卫拒绝时返回 ``None``，
        调用方退回库的 TN 出口（``_w8a8_lib_rows``）。

    Note:
        默认不开（``COFFIN_PREFILL_NK=1`` 才走）。4B 上实测这一档把吞吐从 2468
        掉到 662 tok/s（预填充 512 行从 207 ms 变成 774 ms），因为分组内核按专家切
        并行度，``E = 1`` 只剩一维可用；它换来的是峰值显存 9.41 -> 6.01 GiB——
        也就是库里那一趟 workspace。权重放得下时走库的 TN 出口（本来就不转置），
        放不下时（Occamy 的 35 GB 在 31.7 GiB 的卡上）才换这一档。
    """
    if not _PREFILL_NK:
        return None
    rows, k = x.shape
    if rows < 4:
        return None
    stack = QWeightMoE(w.wq.unsqueeze(0), w.scale.unsqueeze(0).contiguous(),
                       k=k)
    bounds = torch.tensor([0, rows], dtype=torch.int32, device=x.device)
    try:
        return moe_grouped_i8_tcu(stack, x, bounds)
    except ValueError:
        return None


def _w8a8_lib_rows(w: QWeight, act: torch.Tensor) -> torch.Tensor:
    """int8 预填充投影，三步全在 torch 之外：量化 → ``GemmEx`` → epilogue。

    权重就是盘上那份 nk 原缓冲 ``[N, K]``，直接按 ``OP_T`` 递进 ``cublasGemmEx``
    （TN：`[N, K]` 的 nk 行主序缓冲在列主序下就是 `[K, N]`），**一次转置都不做**。

    ``N`` 超过 :data:`pack.LIB_MAX_N` 时沿**行**切：nk 布局下这一步是连续切片，切出来
    仍是一块 ``[width, K]`` 的 nk 权重，``GemmOps.i8_nk`` 照原样吃，不需要指针偏移。

    Args:
        w: nk 原生的 int8 权重，``wq`` 是连续 ``[N, K]``，``scale`` 是 ``[N]`` fp32。
        act: 激活，``[M, K]``，bf16 或 fp32。

    Returns:
        ``[M, N]``，dtype 同 ``act``。

    Raises:
        ValueError: 权重不是 nk 连续，或 dtype/形状不符。
        RtError: 找不到 ``libcublas``，或 GEMM 启动失败。
    """
    rows, k = act.shape
    n = int(w.wq.shape[0])
    if int(w.wq.shape[1]) != int(k):
        raise ValueError(f"weight holds K = {w.wq.shape[1]}, activation has {k}")
    stream = _stream()
    xq = torch.empty((rows, k), dtype=torch.int8, device=act.device)
    xs = torch.empty(rows, dtype=torch.float32, device=act.device)
    out = torch.empty((rows, n), dtype=act.dtype, device=act.device)
    w8a8 = _w8a8_ops()
    gemm = _gemm_ops()
    # 一份激活只量化一次：切块只沿 N 做，每块共用同一份 int8 激活和同一份行 scale。
    w8a8.quant_act_i8(act, xq, xs, stream=stream)
    for start in range(0, n, pack.LIB_MAX_N):
        stop = min(n, start + pack.LIB_MAX_N)
        width = stop - start
        block = w.wq[start:stop]
        if not block.is_contiguous():
            block = block.contiguous()
        acc = torch.empty((rows, width), dtype=torch.int32, device=act.device)
        gemm.i8_nk(xq, block, acc, stream=stream)
        # 行距传 ``n``：``out`` 的列切片在最内层连续到 ``n``，用块宽当行距会把第二行
        # 写回第一行的尾部。
        w8a8.epilogue(acc, xs, w.scale[start:stop], out[:, start:stop], stride=n,
                      stream=stream)
    return out


def _gemm_nk_rows(w: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
    """稠密（bf16/fp32）预填充投影，直接把 ``[N, K]`` 交给库，不再转 ``kn``。

    ``torch.matmul([M, K], [K, N])`` 在这里被 ``GemmEx`` 顶掉：``transa = T`` 让
    ``[N, K]`` 的行主序缓冲就是库要的列主序 ``[K, N]``，``lda = K`` 就是它的行宽，
    所以整份转置副本（4B 上 4.3 GB）根本不存在。

    Args:
        w: ``[N, K]`` 稠密权重，bf16 或 fp32。
        act: 激活，``[M, K]``，bf16 或 fp32。

    Returns:
        ``[M, N]``，dtype 同 ``w``。

    Raises:
        ShapeError: 形状或 dtype 不符。
        RtError: 找不到 ``libcublas``。
    """
    rows, k = act.shape
    n = int(w.shape[0])
    if int(w.shape[1]) != int(k):
        raise ValueError(f"weight holds K = {w.shape[1]}, activation has {k}")
    wt = w if w.is_contiguous() else w.contiguous()
    if wt.dtype not in (torch.bfloat16, torch.float32):
        # 库的 TN 出口只有 bf16 与 fp32 两个实例。fp16 权重（参考实现那一档的遗留）
        # 先拓宽成 bf16——bf16 的 8 位指数装得下 fp16 的 5 位，这次转换不丢信息。
        wt = wt.to(torch.bfloat16)
    # GEMM 两个操作数必须同一 dtype，与旧路径同一条规矩：由权重决定，激活跟着它。
    lhs = act if act.dtype == wt.dtype else act.to(wt.dtype)
    if not lhs.is_contiguous():
        lhs = lhs.contiguous()
    gemm = _gemm_ops()
    stream = _stream()
    # bf16 是被要求「必须做」的那一档：库在 ixblas 下把它按 ``ld`` 转 f16 再走自研
    # fp16 张量核（fp32 累加，39.7 TFLOP/s），厂商那侧 bf16→f32 是 34.8 且算对。
    # fp32 权重（参考实现那一档）走 ``f32_nk``，两者形状契约一样。
    run = gemm.f32_nk if wt.dtype == torch.float32 else gemm.bf16_nk
    if n < 8:
        # 与 ``torch.matmul`` 那条同样的理由（N < 8 时整 tile 大部分空转，补零到 8 列
        # 再切回是 13.4x，结果逐位相同）：这里补的是**行**——``[8, K]``，因为 nk 布局
        # 下每行才是连续的一整段权重，补列等于转置。
        pad = torch.zeros((8, k), dtype=wt.dtype, device=wt.device)
        pad[:n] = wt
        return run(lhs, pad, stream=stream)[:, :n]
    if n <= pack.LIB_MAX_N:
        return run(lhs, wt, stream=stream)
    out = run(lhs, wt[:pack.LIB_MAX_N], stream=stream)
    full = torch.empty((rows, n), dtype=out.dtype, device=out.device)
    full[:, :pack.LIB_MAX_N] = out
    for start in range(pack.LIB_MAX_N, n, pack.LIB_MAX_N):
        stop = min(n, start + pack.LIB_MAX_N)
        full[:, start:stop] = run(lhs, wt[start:stop], stream=stream)
    return full


def linear_prefill(w: torch.Tensor | QWeight,
                   x: torch.Tensor) -> torch.Tensor:
    """预填充分块投影（大 ``M``），此时 GEMM 胜过流式读权重。

    库只用在这张卡已知能算对的宽度内：输出列超过 :data:`pack.LIB_MAX_N` 时它的
    GEMM 在 ``N = 248320``（就是 ``lm_head``）返回 NaN，而那正是最经不起这种错误
    的一个形状。切块判定在 :func:`_w8a8_lib_rows` / :func:`_gemm_nk_rows` 里，内核
    里还有第二道断言。切块沿**权重行**做：nk 布局下每行才是连续的一整段权重，所以
    按块递进库不需要任何拷贝。

    ``M`` 低于 :data:`_PREFILL_MIN_M` 时仍然走自写 GEMV，int8 权重也一样：那条
    分界是实测的（``27B-4CARD-PLAN.md`` §3.4，M=1 上 W8A8 慢 42%），不是可调
    参数。这里额外把分界判断提到 int8 反量化之前，所以小 ``M`` 的 int8 预填充
    不再先付一次全量反量化再把它丢掉。

    两条主要分支都不经过 torch 的算子：int8 走 ``quant_act_i8``(ABI) + ``GemmEx``
    + ``epilogue``(ABI)，稠密 bf16/fp32 走 ``GemmEx`` 的 TN（``[N, K]`` 原缓冲即
    列主序 ``[K, N]``）。``torch.empty`` 只是分配，不是算子。

    Args:
        w: 任意可接受存储形式的 ``[N, K]`` 投影权重。
        x: 激活，``[C, K]``。

    Returns:
        ``[C, N]``，dtype 同 ``x``。

    Raises:
        RuntimeError: 扩展不可用，而宽 ``N`` 形状需要它。
    """
    rows = x.shape[0]
    if rows < _PREFILL_MIN_M:
        # 低于预填充区间时 GEMM 每个 tile 都要读一遍权重，输给整次调用只读一遍的
        # GEMV。分界必须走在打包之后、任何按 dtype 的分支之前。
        return _gemv_rows(w, x)
    dtype = x.dtype
    # ``quant_act_i8`` 与 ``GemmEx`` 都按连续行主序寻址；上游 ``GEMVGroup.prefill``
    # 切出来的 ``up``（``silu_mul(gate, up, up)``）带行间距 ``2 * inter``，不先做
    # 连续化会让量化核跨行错读 ``gate`` 与 ``up``。
    act = _compute_dtype(x).contiguous()
    if isinstance(w, QWeight) and w.wq.is_contiguous():
        # nk 原生的 int8 权重：量化 → ``GemmEx`` → epilogue，三步都在 torch 之外，
        # 权重用**盘上那份**原缓冲，一次转置都不做。
        if _PREFILL_NK:
            out = _prefill_nk_tcu(w, x)
            if out is not None:
                return _restore(out, dtype)
        return _restore(_w8a8_lib_rows(w, act), dtype)
    # 稠密 nk 权重（bf16/fp32）：``[N, K]`` 直接进库、也不 ``torch.matmul``。
    # 这是 ``--fp16`` 那一支，bf16 GEMM 就从这里接进链路。
    return _restore(_gemm_nk_rows(w, act), dtype)


# 归一化


def _pitch_rows(x: torch.Tensor, width: int) -> torch.Tensor:
    """把 ``x`` 按它自己的行间距看作 ``[rows, width]``，不拷贝。

    ``x.reshape(-1, width)`` 是直觉写法，但在这里是错的：更宽缓冲区的一列切片不是
    连续存放的，reshape 于是悄悄拷贝，交给内核的间距描述的也就成了这份拷贝而不是
    缓冲区本身。数值结果仍然对，这正是浪费的带宽容易被忽略的原因。

    Args:
        x: 最后一轴是 ``width`` 的激活。
        width: 最后一轴长度。

    Returns:
        行间距一致时（契约描述的情形）返回 ``[rows, width]`` 视图，否则返回
        reshape 过的（可能已拷贝）张量。
    """
    if x.dim() == 2 and x.stride(-1) == 1:
        return torch.as_strided(x, (x.shape[0], width), (x.stride(0), 1))
    return x.reshape(-1, width)


def rms_norm(x: torch.Tensor, w: torch.Tensor, stride: int = 0,
             eps: float = 1e-6) -> torch.Tensor:
    """经融合内核做 ``x`` 最后一轴上的零中心 RMSNorm。

    Args:
        x: 激活，``[..., N]``；可以是更宽缓冲区的一列切片，此时行间距大于 ``N``。
        w: 增益，``[N]``，已经带上了 checkpoint 的 ``1 + w`` 偏移。
        stride: 行之间的元素间距；``0`` 表示从 ``x`` 自己读。
        eps: 方差下限。

    Returns:
        形状与 dtype 同 ``x``。

    Raises:
        RuntimeError: 扩展不可用。
    """
    dtype = x.dtype
    width = x.shape[-1]
    act = _compute_dtype(x)
    rows = _pitch_rows(act, width)
    gain = w if w.dtype == act.dtype else w.to(act.dtype)
    out = _norm_out((rows.shape[0], width), act)
    _norm_ops().shim.rms_norm(_ptr(rows), _ptr(gain.contiguous()), _ptr(out),
                              rows.shape[0], width,
                              int(rows.stride(0) if stride == 0 else stride),
                              float(eps), _dcode(act), _stream() or 0)
    return _restore(out.view(x.shape), dtype)


def rms_norm_gated(x: torch.Tensor, w: torch.Tensor, gate: torch.Tensor,
                   eps: float = 1e-6) -> torch.Tensor:
    """经融合内核做按头的 RMSNorm，再乘 ``silu(gate)``。

    Args:
        x: 激活，``[..., dv]``。
        w: 按头的增益，``[dv]``：不是展平后的宽度，因为归一化是按头做的。
        gate: 形状同 ``x`` 的门张量。
        eps: 方差下限。

    Returns:
        形状与 dtype 同 ``x``。

    Raises:
        RuntimeError: 扩展不可用。
    """
    dtype = x.dtype
    width = x.shape[-1]
    act = _compute_dtype(x)
    gate_c = _compute_dtype(gate)
    gain = w if w.dtype == act.dtype else w.to(act.dtype)
    x2 = act.reshape(-1, width)
    g2 = gate_c.reshape(-1, width)
    out = _norm_out((x2.shape[0], width), act)
    _norm_ops().shim.rms_norm_gated(_ptr(x2), _ptr(gain.contiguous()), _ptr(g2),
                                    _ptr(out), x2.shape[0], width, float(eps),
                                    _dcode(act), _stream() or 0)
    return _restore(out.view(x.shape), dtype)


def add_rms_norm(x: torch.Tensor, add: torch.Tensor, w: torch.Tensor,
                 out: torch.Tensor, stride: int = 0,
                 eps: float = 1e-6) -> torch.Tensor:
    """一次启动内完成「把残差加进 ``x`` 再做零中心 RMSNorm」。

    上游写法是两次启动、两遍读残差流；融合掉省下每次调用一次启动（按本卡 5.75 us
    的启动下限算）以及整个 ``[hidden]`` 向量的一次读取。

    Args:
        x: 残差流，``[..., N]``，原地累加。
        add: 形状同 ``x`` 的混合器输出。
        w: 增益，``[N]``，已经带上了 checkpoint 的 ``1 + w`` 偏移。
        out: 归一化后和的去向，形状同 ``x``。不得与 ``x`` 重叠：残差必须活过这次
            归一化。
        stride: 行之间的元素间距；``0`` 表示从 ``x`` 自己读。
        eps: 方差下限。

    Returns:
        ``out``，便于调用链串接。

    Raises:
        RuntimeError: 扩展不可用。
    """
    width = x.shape[-1]
    act = _compute_dtype(x)
    act_add = _compute_dtype(add)
    rows = _pitch_rows(act, width)
    rows_add = _pitch_rows(act_add, width)
    gain = w if w.dtype == act.dtype else w.to(act.dtype)
    stride0 = rows.stride(0) if stride == 0 else stride
    _norm_ops().shim.add_rms_norm(_ptr(rows_add), _ptr(rows), _ptr(gain.contiguous()),
                                  _ptr(out.view(-1, width)), rows.shape[0], width,
                                  int(stride0), float(eps), _dcode(act), _stream() or 0)
    return out


def kv_quant(x: torch.Tensor, out_q: torch.Tensor,
             out_s: torch.Tensor) -> torch.Tensor:
    """对 ``[M, N]`` 做对称 int8 量化，每行一个 fp32 scale，一次启动。

    替代 ``_quant`` 用的那条七算子 torch 链：后者每次调用约 82 us，而本卡的启动
    下限是 5.75 us，并且每个全注意力层要跑两次。

    Args:
        x: 待量化的激活，``[M, N]``。
        out_q: int8 目的地，``[M, N]``，原地写入。
        out_s: 每行 scale 的 fp32 目的地，``[M]``，原地写入。

    Returns:
        ``out_q``，便于调用链串接。

    Raises:
        RuntimeError: 扩展不可用。
    """
    act = _compute_dtype(x)
    width = act.shape[-1]
    rows = _pitch_rows(act, width)
    _attn_ops().shim.kv_quant(_ptr(rows), _ptr(out_q), _ptr(out_s), rows.shape[0], width,
                              int(rows.stride(0)), _dcode(act), _stream() or 0)
    return out_q


def kv_store(k: torch.Tensor, v: torch.Tensor, ks: torch.Tensor, vs: torch.Tensor,
             slot: torch.Tensor, kc: torch.Tensor, vc: torch.Tensor,
             ksc: torch.Tensor, vsc: torch.Tensor) -> torch.Tensor:
    """一次启动写入 k/v 缓存与它们的每头 scale。

    替代 ``FullAttn.__call__`` 里那四次 ``index_copy_``（k、v、kscale、vscale 各一次）：
    全注意力层每层四次、10 层就是每步 40 次发射，而它们共用同一个槽位号、搬的字节只有
    ``nkv * hd``。

    槽位号留在设备上读：``pos1`` 本来就是 CUDA 张量，取回主机会多一次 D2H 同步。

    Args:
        k: ``[nkv, hd]`` int8 量化后的 key。
        v: 同形状的 value。
        ks: ``[nkv]`` fp32 key 每头 scale。
        vs: 同形状的 value scale。
        slot: ``[1]`` CUDA int32/int64 槽位号。
        kc: ``[nkv, max_seq, hd]`` int8 缓存，原地写入。
        vc: 同上。
        ksc: ``[nkv, max_seq]`` fp32，原地写入。
        vsc: 同上。

    Returns:
        ``kc``，便于调用链串接。

    Raises:
        RuntimeError: 扩展不可用。
    """
    nkv, hd = int(k.shape[0]), int(k.shape[1])
    max_seq = int(kc.shape[1])
    _attn_ops().shim.kv_store(_ptr(k), _ptr(v), _ptr(ks), _ptr(vs), _ptr(slot), _ptr(kc),
                              _ptr(vc), _ptr(ksc), _ptr(vsc), nkv, hd, max_seq,
                              1 if slot.dtype == torch.int64 else 0, _stream() or 0)
    return kc


def kv_pool_store(k: torch.Tensor, v: torch.Tensor, ks: torch.Tensor, vs: torch.Tensor,
                  slot: torch.Tensor, kc: torch.Tensor, vc: torch.Tensor,
                  ksc: torch.Tensor, vsc: torch.Tensor, layer: int,
                  head_major: bool) -> bool:
    """一次启动把一层的一个 token 写进 paged k/v 池。

    ``PagedCache.write_slot`` 原来是五次发射：一次 ``full_like`` 造层号张量、再四次
    ``index_put_``。全注意力层每层一次、10 层就是每步 50 次发射，而它们写的是同一个
    ``[layer, slot]`` 单元、搬的字节只有 ``nkv * hd``。层号在这里是主机侧常量，所以那次
    ``full_like`` 也不必存在。

    形状不合（多槽位、非连续、dtype 不符、层号越界）时返回 ``False``，调用方照旧走逐张量
    写回。

    Args:
        k: ``[nkv, hd]`` int8 key 行。
        v: 同形状的 value 行。
        ks: ``[nkv]`` fp32 key scale。
        vs: 同形状的 value scale。
        slot: ``[1]`` CUDA int32/int64 槽位号，设备上读。
        kc: ``[layers, slots, nkv, hd]`` 或 ``[layers, nkv, slots, hd]`` int8 池。
        vc: 同上。
        ksc: ``[layers, slots, nkv]`` 或 ``[layers, nkv, slots]`` fp32。
        vsc: 同上。
        layer: 全注意力层在池层轴上的下标。
        head_major: 池是否按 ``[layers, nkv, slots, hd]`` 排。

    Returns:
        是否真的走了融合入口。``False`` 时调用方必须回退到逐张量写回。

    Raises:
        RuntimeError: 扩展不可用（此时不会返回 ``False``，而不是静默降级）。
    """
    nkv, hd = int(k.shape[0]), int(k.shape[1])
    # 池的两种布局：``head_major`` 时槽位轴在 size(2)，否则在 size(1)。
    nslots = int(kc.shape[2] if head_major else kc.shape[1])
    code = _attn_ops().shim.kv_pool_store(_ptr(k), _ptr(v), _ptr(ks), _ptr(vs), _ptr(slot),
                                          _ptr(kc), _ptr(vc), _ptr(ksc), _ptr(vsc), nkv, hd,
                                          nslots, int(layer), 1 if head_major else 0,
                                          1 if slot.dtype == torch.int64 else 0,
                                          _stream() or 0)
    return code == 0


def silu_mul(gate: torch.Tensor, up: torch.Tensor,
             out: torch.Tensor) -> torch.Tensor:
    """一次启动算 ``out = silu(gate) * up``，即 SwiGLU 乘积。

    这里不为了把操作数弄连续而拷贝任何东西，这一条是承重的而不是洁癖。
    ``GEMVGroup.__call__`` 把两半作为同一个融合结果的
    ``.split(sizes, dim=1)`` 交回来，所以在每条二维路径（预填充、批处理、宽形状）
    上 ``gate`` 和 ``up`` 都是更宽缓冲区的 *列视图*，并不连续。在这里把它们稠密化
    会让每个 MLP 多两次 ``_to_copy`` 启动，总共三次，而
    ``F.silu(gate) * up`` 原本是两次；这种退化还只会出现在预填充路径上，那恰恰
    不是被测量的路径。内核改为接受按操作数给的行间距，所以带步长的列视图可以原地
    服务。

    Args:
        gate: 融合投影门的那一半，``[...]``，最内层步长为 1。
        up: 另一半，形状同 ``gate``，最内层步长为 1（不必连续）。
        out: 目的地，形状相同。可以与 ``up`` 重叠。

    Returns:
        ``out``，便于调用链串接。

    Raises:
        RuntimeError: 扩展不可用。
    """
    if gate.dim() == 1:
        g, u, o, rows = gate, up, out, 1
    else:
        width = gate.shape[-1]
        g = _pitch_rows(gate, width)
        u = _pitch_rows(up, width)
        o = _pitch_rows(out, width)
        rows = int(g.shape[0])
    width = int(g.shape[-1])
    # 内核对单行也按 pitch 走指针，要求 pitch 不窄于行宽；一维连续向量的 stride(0)
    # 是 1，直接送进去会被出口以「row pitch narrower than the row width」挡下。
    dense = rows == 1
    gp = width if dense else int(g.stride(0))
    up_p = width if dense else int(u.stride(0))
    op = width if dense else int(o.stride(0))
    _attn_ops().shim.silu_mul(_ptr(g), _ptr(u), _ptr(o), rows, width, gp, up_p, op,
                              _dcode(g), _stream() or 0)
    return out


def rope_apply(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor,
               sin: torch.Tensor, q_out: torch.Tensor, k_out: torch.Tensor,
               rotary_dim: int) -> torch.Tensor:
    """一次启动内旋转每个 q/k 头最前面 ``rotary_dim`` 个通道。

    Args:
        q: 查询，``[M, nh, hd]``。
        k: 键，``[M, nkv, hd]``。
        cos: 被旋转那些行的余弦，``[M, rotary_dim]`` fp32。
        sin: 正弦，布局同 ``cos``。
        q_out: 旋转后查询的去向，形状同 ``q``。
        k_out: 旋转后键的去向，形状同 ``k``。
        rotary_dim: 要旋转的通道数。

    Returns:
        ``q_out``，便于调用链串接。

    Raises:
        RuntimeError: 扩展不可用。
    """
    _attn_ops().shim.rope_apply(_ptr(q), _ptr(k), _ptr(cos), _ptr(sin),
                                _ptr(q_out), _ptr(k_out), int(q.shape[0]),
                                int(q.shape[1]), int(k.shape[1]), int(q.shape[2]),
                                int(rotary_dim), _stream() or 0)
    return q_out


# 注意力


def _split_k(part_acc: torch.Tensor | None, part_ml: torch.Tensor | None,
             shape: tuple[int, ...], like: torch.Tensor) -> tuple[torch.Tensor,
                                                                  torch.Tensor]:
    """返回 split-K 的暂存，调用方没给就分配。

    在这里分配正确但不可被图捕获：捕获路径要求缓冲区地址固定，所以每次重放都新
    分配会让图写到一个已死的张量上。因此模型的捕获路径总是自己传缓冲区进来。

    Args:
        part_acc: 调用方自有的分子，或 ``None``。
        part_ml: 调用方自有的运行最大值／分母，或 ``None``。
        shape: split 轴之前的维度，例如 ``(nh,)`` 或 ``(B, nh)``。
        like: 分配跟随其设备的张量。

    Returns:
        适配本后端 ``attn_split`` 的 ``(part_acc, part_ml)``。
    """
    split = caps().attn_split
    if part_acc is None:
        part_acc = torch.zeros(*shape, split, like.shape[-1], dtype=torch.float32,
                               device=like.device)
    if part_ml is None:
        part_ml = torch.zeros(*shape, split, 2, dtype=torch.float32, device=like.device)
    return part_acc, part_ml


def _attn_slice_call(q: torch.Tensor, gate: torch.Tensor, kcache: torch.Tensor,
                     vcache: torch.Tensor, kscale: torch.Tensor, vscale: torch.Tensor,
                     out: torch.Tensor, part_acc: torch.Tensor, part_ml: torch.Tensor,
                     *, flat: int, nh: int, hd: int, rep: int, scale: float, draft: int,
                     mask: torch.Tensor | None, mask_block: int, stride: int, max_seq: int,
                     stride_head: int, map_rows: int, ctx_anchor: int,
                     pos2slot: torch.Tensor | None, ctx_lens: torch.Tensor | None,
                     pos: torch.Tensor | None) -> None:
    """把一次注意力调用交给 C ABI 出口，几何不合就抛。

    四个公开入口只差 ``AttnLayout`` 的那几个数，所以它们在这一个函数里汇合——与设备侧
    汇到 ``launch_attn_slice`` 是同一个理由：稠密和分页两条路不能各自漂移。

    Args:
        q, gate: ``[flat, nh, hd]`` bf16 查询与输出门（稠密）。
        kcache, vcache: int8 缓存，布局由 ``stride`` / ``stride_head`` 选。
        kscale, vscale: fp32 反量化 scale。
        out: ``[flat, nh, hd]`` bf16 目的地。
        part_acc, part_ml: split-K 暂存。
        flat: 行数。
        nh: 查询头数。
        hd: 头宽。
        rep: 每个 KV 头对应的查询头数。
        scale: ``hd ** -0.5``。
        draft: 共用一行历史的查询数。
        mask: 可选的 uint8 保留掩码。
        mask_block: 掩码的块大小。
        stride: 相邻槽位之间的行距。
        max_seq: 窗口上限。
        stride_head: 相邻 KV 头之间的行距。
        map_rows: ``pos2slot`` 是否整张 ``[B, max_seq]`` 映射。
        ctx_anchor: ``ctx_lens`` 是否装锚点下标。
        pos2slot, ctx_lens, pos: 三个设备侧窗口定义，至少给一个。

    Raises:
        RuntimeError: 出口拒绝了这组几何。
    """
    rc = _attn_ops().shim.attn_slice(
        _ptr(q), _ptr(gate), _ptr(kcache), _ptr(vcache), _ptr(kscale), _ptr(vscale), _ptr(out),
        _ptr(part_acc), _ptr(part_ml),
        0 if mask is None else _ptr(mask),
        0 if pos2slot is None else _ptr(pos2slot),
        0 if ctx_lens is None else _ptr(ctx_lens),
        0 if pos is None else _ptr(pos),
        float(scale), int(flat), int(nh), int(hd), int(rep), int(part_acc.shape[-2]),
        int(draft), int(mask_block), int(stride), int(max_seq), int(stride_head),
        int(map_rows), int(ctx_anchor), _stream() or 0)
    if rc:
        raise RuntimeError(
            f"coffin_attn_slice rejected the geometry (flat={flat} nh={nh} hd={hd} rep={rep} "
            f"stride={stride}/{stride_head} max_seq={max_seq} code={rc})")


def attn_decode(q: torch.Tensor, gate: torch.Tensor, kcache: torch.Tensor,
                vcache: torch.Tensor, kscale: torch.Tensor, vscale: torch.Tensor,
                out: torch.Tensor, pos: torch.Tensor, rep: int, scale: float,
                part_acc: torch.Tensor | None = None,
                part_ml: torch.Tensor | None = None) -> torch.Tensor:
    """经融合内核在稠密 int8 KV 缓存上算一个解码 token。

    Args:
        q: 查询，``[nh, hd]``，调用方已经旋转并缩放好。
        gate: 输出门，``[nh, hd]``；内核施加 ``sigmoid``。
        kcache: int8 键，``[nkv, max_seq, hd]``。
        vcache: int8 值，``[nkv, max_seq, hd]``。
        kscale: fp32 K scale，``[nkv, max_seq]``。
        vscale: fp32 V scale，``[nkv, max_seq]``。
        out: 输出缓冲区，``[nh, hd]``，原地写入。
        pos: 单元素 long 张量；token 下标兼因果上界。在设备上读取，所以被捕获的
            重放看到的是当前的位置，而不是捕获那一刻冻住的那个。
        rep: 每个 KV 头对应的查询头数。
        scale: ``hd ** -0.5``。
        part_acc: split-K 累加器，``[nh, split, hd]`` fp32；``None`` 表示分配。
        part_ml: split-K 运行最大值／分母，``[nh, split, 2]`` fp32。

    Returns:
        ``out``，便于调用链串接。

    Raises:
        RuntimeError: 扩展不可用。
    """
    nh, hd = int(q.shape[0]), int(q.shape[1])
    seq = int(kcache.shape[1])
    pa, pm = _split_k(part_acc, part_ml, (nh,), q)
    _attn_slice_call(q.contiguous(), gate.contiguous(), kcache, vcache, kscale, vscale,
                     out, pa, pm, flat=1, nh=nh, hd=hd, rep=rep, scale=scale, draft=1,
                     mask=None, mask_block=0, stride=1, max_seq=seq, stride_head=seq,
                     map_rows=0, ctx_anchor=0, pos2slot=None, ctx_lens=None, pos=pos)
    return out


def attn_paged(q: torch.Tensor, gate: torch.Tensor, kpool: torch.Tensor,
               vpool: torch.Tensor, kscale: torch.Tensor, vscale: torch.Tensor,
               pos2slot: torch.Tensor, out: torch.Tensor, pos: torch.Tensor,
               max_pos: int, rep: int, scale: float,
               part_acc: torch.Tensor | None = None,
               part_ml: torch.Tensor | None = None,
               head_major: bool = False) -> torch.Tensor:
    """在某一层的分页 int8 池上算一个解码 token。

    Args:
        q: 查询，``[nh, hd]``。
        gate: 输出门，``[nh, hd]``。
        kpool: 本层的 int8 K 池，``[nslots, nkv, hd]``。
        vpool: 本层的 int8 V 池，``[nslots, nkv, hd]``。
        kscale: fp32 K scale，``[nslots, nkv]``。
        vscale: fp32 V scale，``[nslots, nkv]``。
        pos2slot: 本行的槽位表；只读前 ``pos + 1`` 项。
        out: 输出缓冲区，``[nh, hd]``，原地写入。
        pos: 单元素 long 张量；token 下标兼因果上界。
        max_pos: 窗口长度，即槽位表的行间距。
        rep: 每个 KV 头对应的查询头数。
        scale: ``hd ** -0.5``。
        part_acc: split-K 累加器，``[nh, split, hd]`` fp32。
        part_ml: split-K 运行最大值／分母，``[nh, split, 2]`` fp32。

    Returns:
        ``out``，便于调用链串接。

    Raises:
        RuntimeError: 扩展不可用。
    """
    nh, hd = int(q.shape[0]), int(q.shape[1])
    nkv = int(kpool.shape[0]) if head_major else int(kpool.shape[1])
    nslots = int(kpool.shape[1]) if head_major else int(kpool.shape[0])
    stride_slot = 1 if head_major else nkv
    stride_head = nslots if head_major else 1
    pa, pm = _split_k(part_acc, part_ml, (nh,), q)
    # ``pos2slot`` 是 ``pool.pos2slot[row]``，首个元素已经是本行第一个位置，所以内核的不加
    # 行偏移（``map_rows=0``）；搞错这一步，单次解码步就会读到相邻行的键。
    _attn_slice_call(q.contiguous(), gate.contiguous(), kpool, vpool, kscale, vscale,
                     out, pa, pm, flat=1, nh=nh, hd=hd, rep=rep, scale=scale, draft=1,
                     mask=None, mask_block=0, stride=stride_slot, max_seq=int(max_pos),
                     stride_head=stride_head, map_rows=0, ctx_anchor=0, pos2slot=pos2slot,
                     ctx_lens=None, pos=pos)
    return out


def attn_paged_batch(q: torch.Tensor, gate: torch.Tensor, kpool: torch.Tensor,
                     vpool: torch.Tensor, kscale: torch.Tensor, vscale: torch.Tensor,
                     slots: torch.Tensor, context_lens: torch.Tensor,
                     out: torch.Tensor, max_seq: int, rep: int, scale: float,
                     part_acc: torch.Tensor | None = None,
                     part_ml: torch.Tensor | None = None,
                     head_major: bool = False) -> torch.Tensor:
    """整个桶的行一次启动算一个解码 token。

    Args:
        q: 查询，``[B, nh, hd]``。
        gate: 输出门，``[B, nh, hd]``。
        kpool: 本层的 int8 K 池，``[nslots, nkv, hd]``。
        vpool: 本层的 int8 V 池，``[nslots, nkv, hd]``。
        kscale: fp32 K scale，``[nslots, nkv]``。
        vscale: fp32 V scale，``[nslots, nkv]``。
        slots: 每行的槽位表，``[B, max_seq]``。
        context_lens: 每行的键数，``[B]`` int32。
        out: 输出缓冲区，``[B, nh, hd]``，原地写入。
        max_seq: 窗口长度，即 ``slots`` 的行间距。
        rep: 每个 KV 头对应的查询头数。
        scale: ``hd ** -0.5``。
        part_acc: split-K 累加器，``[B, nh, split, hd]`` fp32。
        part_ml: split-K 运行最大值／分母，``[B, nh, split, 2]`` fp32。

    Returns:
        ``out``，便于调用链串接。

    Raises:
        RuntimeError: 扩展不可用。
    """
    rows, nh, hd = int(q.shape[0]), int(q.shape[1]), int(q.shape[2])
    nkv = int(kpool.shape[0]) if head_major else int(kpool.shape[1])
    nslots = int(kpool.shape[1]) if head_major else int(kpool.shape[0])
    stride_slot = 1 if head_major else nkv
    stride_head = nslots if head_major else 1
    pa, pm = _split_k(part_acc, part_ml, (rows, nh), q)
    # 这里的 ``context_lens`` 就是键数，锚点那次转换不适用（``ctx_anchor=0``）；``slots``
    # 是整个 ``[B, max_seq]`` 映射，所以内核要偏移到第 ``b`` 行（``map_rows=1``）。
    #
    # 在边界上把操作数变成稠密的，而不是相信调用方传的。内核通过 ``data_ptr()`` 把
    # ``q`` 和 ``gate`` 当成一整块平的 ``[B, nh, hd]`` 来索引，所以列视图会悄悄读到
    # 错误的字节——而构造门的自然写法正是这样的视图（融合 ``[.., 2*hd]`` 投影的
    # ``qg[..., hd:]``，最内层步长是 1，但行间距是 ``2*hd``）。实测：用带步长的门
    # 时，这个入口和第 3 层的 ``attn_paged_multi_q`` 在 3834/4096 个元素上不一致，
    # 而查询和池里的字节逐位相同，这个现象被当成内核缺陷查了好几个会话
    # （``docs/STAGE_LOCALISED.md``）。调用方本来就传稠密张量时这次拷贝是空操作，
    # 所以它属于边界而不属于每个调用点。
    _attn_slice_call(q.contiguous(), gate.contiguous(), kpool, vpool, kscale, vscale,
                     out, pa, pm, flat=rows, nh=nh, hd=hd, rep=rep, scale=scale, draft=1,
                     mask=None, mask_block=0, stride=stride_slot, max_seq=int(max_seq),
                     stride_head=stride_head, map_rows=1, ctx_anchor=0,
                     pos2slot=slots.contiguous() if slots is not None else None,
                     ctx_lens=context_lens.contiguous(), pos=None)
    return out


def attn_paged_multi_q(q: torch.Tensor, gate: torch.Tensor, kpool: torch.Tensor,
                       vpool: torch.Tensor, kscale: torch.Tensor, vscale: torch.Tensor,
                       slots: torch.Tensor, ctx_lens: torch.Tensor, out: torch.Tensor,
                       max_seq: int, rep: int, scale: float,
                       part_acc: torch.Tensor | None = None,
                       part_ml: torch.Tensor | None = None,
                       block_mask: torch.Tensor | None = None,
                       mask_block: int = 64,
                       head_major: bool = False) -> torch.Tensor:
    """投机校验：每行 ``c`` 个草稿查询共用一遍历史读取。

    Args:
        q: 草稿查询，``[B, c, nh, hd]``。
        gate: 输出门，形状相同。
        kpool: 本层的 int8 K 池，``[nslots, nkv, hd]``。
        vpool: 本层的 int8 V 池，``[nslots, nkv, hd]``。
        kscale: fp32 K scale，``[nslots, nkv]``。
        vscale: fp32 V scale，``[nslots, nkv]``。
        slots: 每行的槽位表，``[B, max_seq]``。
        ctx_lens: 每行的历史长度，``[B]`` int32；第 ``j`` 个查询看到
            ``ctx_lens[b] + j + 1`` 个键，所以和单独解码那个 token 完全一致。
        out: 输出缓冲区，``[B, c, nh, hd]``，原地写入。
        max_seq: 窗口长度，即 ``slots`` 的行间距。
        rep: 每个 KV 头对应的查询头数。
        scale: ``hd ** -0.5``。
        part_acc: split-K 累加器，``[B*c, nh, split, hd]`` fp32。
        part_ml: split-K 运行最大值／分母，``[B*c, nh, split, 2]`` fp32。
        block_mask: 可选的 uint8 ``[B, n_blocks]`` 保留掩码（1 = 保留）。
        mask_block: 掩码按多大的块表达。

    Returns:
        ``out``，便于调用链串接。

    Raises:
        RuntimeError: 扩展不可用。
    """
    rows, draft, nh, hd = (int(v) for v in q.shape)
    nkv = int(kpool.shape[0]) if head_major else int(kpool.shape[1])
    nslots = int(kpool.shape[1]) if head_major else int(kpool.shape[0])
    stride_slot = 1 if head_major else nkv
    stride_head = nslots if head_major else 1
    pa, pm = _split_k(part_acc, part_ml, (rows * draft, nh), q)
    # 契约里掩码是可选的；「精确注意力」是以空张量传来的，把它当成掩码会把每个键都清零。
    masked = block_mask if block_mask is not None and int(block_mask.numel()) > 0 else None
    # 这里的 ``ctx_lens`` 是锚点的最后下标而不是计数：第 ``j`` 个查询需要
    # ``anchor + 1 + j`` 个键，内核按 ``ctx_anchor`` 把两项都加上。``slots`` 是整张映射。
    #
    # 在边界上把操作数变成稠密的，理由见 ``attn_paged_batch``：内核通过
    # ``data_ptr()`` 把门当成一整块平来索引，而每个调用方都是把门构造成融合投影的
    # 列切片。
    _attn_slice_call(q.contiguous(), gate.contiguous(), kpool, vpool, kscale, vscale,
                     out, pa, pm, flat=rows * draft, nh=nh, hd=hd, rep=rep, scale=scale,
                     draft=draft, mask=masked, mask_block=int(mask_block) if masked else 0,
                     stride=stride_slot, max_seq=int(max_seq), stride_head=stride_head,
                     map_rows=1, ctx_anchor=1, pos2slot=slots.contiguous(),
                     ctx_lens=ctx_lens.contiguous(), pos=None)
    return out


# 门控 DeltaNet


def conv1d_silu(conv_w: torch.Tensor, x: torch.Tensor, conv_state: torch.Tensor,
                out: torch.Tensor) -> torch.Tensor:
    """单 token 的深度卷积（核 4）加 SiLU，同时平移窗口。

    Args:
        conv_w: 逐通道的抽头，``[C, 4]`` fp32。
        x: 本 token 的投影，``[C]``。
        conv_state: 前 3 个 token，``[C, 3]`` fp32；原地平移。
        out: 输出缓冲区，``[C]`` fp32，原地写入。

    Returns:
        ``out``，便于调用链串接。

    Raises:
        RuntimeError: 扩展不可用。
    """
    _conv_ops().shim.conv1d_silu(_ptr(conv_w), _ptr(x), _ptr(conv_state), _ptr(out),
                                 int(conv_w.shape[0]), _stream() or 0)
    return out


def conv1d_silu_seq(conv_w: torch.Tensor, x: torch.Tensor,
                    conv_state: torch.Tensor) -> torch.Tensor:
    """同一个卷积跑在预填充分块上，窗口跨分块携带。

    Args:
        conv_w: 逐通道的抽头，``[C, 4]`` fp32。
        x: 分块的投影，``[T, C]``。
        conv_state: 前 3 个 token，``[C, 3]`` fp32；原地更新。

    Returns:
        ``[T, C]`` fp32 的卷积加 SiLU 结果。

    Raises:
        RuntimeError: 扩展不可用。
    """
    tokens, channels = int(x.shape[0]), int(x.shape[1])
    out = torch.empty((tokens, channels), dtype=torch.float32, device=x.device)
    _conv_ops().shim.conv1d_silu_seq(_ptr(conv_w), _ptr(x), _ptr(conv_state), _ptr(out),
                                     tokens, channels, int(x.stride(0)), _stream() or 0)
    return out


def conv1d_silu_batch(conv_w: torch.Tensor, x: torch.Tensor, conv_state: torch.Tensor,
                      out: torch.Tensor) -> torch.Tensor:
    """批处理的深度因果卷积加 SiLU，每行一个窗口。

    Args:
        conv_w: 逐通道的抽头，``[C, 4]`` fp32。
        x: 每行一个 token，``[B, C]``。
        conv_state: 每行的窗口，``[B, C, 3]`` fp32；原地平移。
        out: 输出缓冲区，``[B, C]`` fp32，原地写入。

    Returns:
        ``out``，便于调用链串接。

    Raises:
        RuntimeError: 扩展不可用。
    """
    _conv_ops().shim.conv1d_silu_batch(_ptr(conv_w), _ptr(x), _ptr(conv_state), _ptr(out),
                                       int(x.shape[0]), int(x.shape[1]),
                                       int(x.stride(0)), int(conv_state.stride(0)),
                                       int(out.stride(0)), _stream() or 0)
    return out


def delta_prologue(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, b: torch.Tensor,
                   a: torch.Tensor, neg_expA: torch.Tensor, dt_bias: torch.Tensor,
                   q_out: torch.Tensor, k_out: torch.Tensor, v_out: torch.Tensor,
                   beta: torch.Tensor, g: torch.Tensor, qscale: float,
                   rep: int) -> None:
    """单 token 的 q/k 归一化、头扩展与 beta/g 门，融合在一起。

    Args:
        q: 每个键头的查询，``[num_k, dk]``。
        k: 每个键头的键，``[num_k, dk]``。
        v: 每个值头的值，``[num_v, dv]``。
        b: beta logits，``[num_v]``。
        a: 衰减 logits，``[num_v]``。
        neg_expA: ``-exp(A_log)``，``[num_v]`` fp32。
        dt_bias: 衰减偏置，``[num_v]`` fp32。
        q_out: 输出查询，``[num_v, dk]`` fp32。
        k_out: 输出键，``[num_v, dk]`` fp32。
        v_out: 输出值，``[num_v, dv]`` fp32。
        beta: 输出写强度，``[num_v]`` fp32。
        g: 输出对数衰减 ``<= 0``，``[num_v]`` fp32。
        qscale: ``dk ** -0.5``。
        rep: 每个键头对应的值头数。

    Returns:
        无；每个输出缓冲区都是原地写入。

    Raises:
        RuntimeError: 扩展不可用。
    """
    nv, dk, dv = int(v.shape[0]), int(q.shape[1]), int(v.shape[1])
    _attn_ops().shim.delta_prologue(_ptr(q), _ptr(k), _ptr(v), _ptr(b), _ptr(a), _ptr(neg_expA),
                                    _ptr(dt_bias), _ptr(q_out), _ptr(k_out), _ptr(v_out),
                                    _ptr(beta), _ptr(g), float(qscale), int(rep), nv, dk, dv,
                                    _stream() or 0)


def delta_prologue_batch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         b: torch.Tensor, a: torch.Tensor, neg_expA: torch.Tensor,
                         dt_bias: torch.Tensor, q_out: torch.Tensor, k_out: torch.Tensor,
                         v_out: torch.Tensor, beta: torch.Tensor, g: torch.Tensor,
                         qscale: float, rep: int) -> None:
    """批处理版的 :func:`delta_prologue`，每个参数多一个前导行维度。

    Args:
        q: 每个键头的查询，``[B, num_k, dk]``。
        k: 每个键头的键，``[B, num_k, dk]``。
        v: 每个值头的值，``[B, num_v, dv]``。
        b: beta logits，``[B, num_v]``。
        a: 衰减 logits，``[B, num_v]``。
        neg_expA: ``-exp(A_log)``，``[num_v]``，所有行共用。
        dt_bias: 衰减偏置，``[num_v]``，所有行共用。
        q_out: 输出查询，``[B, num_v, dk]`` fp32。
        k_out: 输出键，``[B, num_v, dk]`` fp32。
        v_out: 输出值，``[B, num_v, dv]`` fp32。
        beta: 输出写强度，``[B, num_v]`` fp32。
        g: 输出对数衰减 ``<= 0``，``[B, num_v]`` fp32。
        qscale: ``dk ** -0.5``。
        rep: 每个键头对应的值头数。

    Returns:
        无；缓冲区都是原地写入。

    Raises:
        RuntimeError: 扩展不可用。
    """
    rows, nv, dk, dv = (int(v.shape[0]), int(v.shape[1]), int(q.shape[2]),
                       int(v.shape[2]))
    _attn_ops().shim.delta_prologue_batch(_ptr(q), _ptr(k), _ptr(v), _ptr(b), _ptr(a),
                                          _ptr(neg_expA), _ptr(dt_bias), _ptr(q_out),
                                          _ptr(k_out), _ptr(v_out), _ptr(beta), _ptr(g),
                                          float(qscale), int(rep), rows, nv, dk, dv,
                                          _stream() or 0)


def delta_prep_seq(qkv: torch.Tensor, b: torch.Tensor, a: torch.Tensor,
                   neg_expA: torch.Tensor, dt_bias: torch.Tensor, q_out: torch.Tensor,
                   k_out: torch.Tensor, v_out: torch.Tensor, beta: torch.Tensor,
                   g: torch.Tensor, qscale: float, rep: int) -> None:
    """预填充的头预处理，写出分块扫描读取的头主序布局。

    Args:
        qkv: 卷积输出，``[T, conv_dim]`` fp32。
        b: beta logits，``[T, num_v]``。
        a: 衰减 logits，``[T, num_v]``。
        neg_expA: ``-exp(A_log)``，``[num_v]`` fp32。
        dt_bias: 衰减偏置，``[num_v]`` fp32。
        q_out: 输出查询，``[num_v, T, dk]`` fp32。
        k_out: 输出键，``[num_v, T, dk]`` fp32。
        v_out: 输出值，``[num_v, T, dv]`` fp32。
        beta: 输出写强度，``[num_v, T]`` fp32。
        g: 输出对数衰减 ``<= 0``，``[num_v, T]`` fp32。
        qscale: ``dk ** -0.5``。
        rep: 每个键头对应的值头数。

    Returns:
        无；每个输出缓冲区都是原地写入。

    Raises:
        RuntimeError: 扩展不可用。
    """
    T, nv = int(qkv.shape[0]), int(q_out.shape[0])
    dk, dv = int(q_out.shape[2]), int(v_out.shape[2])
    kdim = dk * (nv // int(rep))
    _attn_ops().shim.delta_prep_seq(_ptr(qkv), _ptr(b), _ptr(a), _ptr(neg_expA), _ptr(dt_bias),
                                    _ptr(q_out), _ptr(k_out), _ptr(v_out), _ptr(beta), _ptr(g),
                                    float(qscale), int(rep), T, nv, dk, dv, kdim,
                                    int(qkv.shape[1]), int(b.stride(0)), int(a.stride(0)),
                                    _stream() or 0)


def _delta_dims(st: torch.Tensor) -> tuple[int, int, int, int]:
    """从状态张量读 ``(rows, heads, dk, dv)``。

    单行入口的状态是 ``[heads, dk, dv]``（decode 一行），批处理是 ``[B, heads, dk, dv]``。
    内核按 ``b * heads * dk * dv`` 算行偏移，所以 ``rows`` 必须与实际秩一致。

    Args:
        st: 循环状态。

    Returns:
        ``(rows, heads, dk, dv)``。
    """
    shape = tuple(int(v) for v in st.shape)
    if len(shape) == 4:
        return shape[0], shape[1], shape[2], shape[3]
    return 1, shape[0], shape[1], shape[2]


def _delta_call(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor,
                g: torch.Tensor, st: torch.Tensor, out: torch.Tensor,
                record: torch.Tensor | None, slot: int) -> None:
    """把一次门控 delta 步交给 C ABI 出口。

    Args:
        q, k, v, beta, g: 输入。
        st: 状态，原地更新。
        out: 输出缓冲区。
        record: 可选的更新日志；``None`` 表示不记录。
        slot: 日志中的位置。
    """
    rows, heads, dk, dv = _delta_dims(st)
    _attn_ops().shim.delta(_ptr(q), _ptr(k), _ptr(v), _ptr(beta), _ptr(g), _ptr(st), _ptr(out),
                           0 if record is None else _ptr(record), int(slot),
                           0 if record is None else int(record.shape[1]), rows, heads, dk, dv,
                           _stream() or 0)


def delta(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor,
          g: torch.Tensor, st: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """门控 delta 的一个解码步，原地更新循环状态。

    Args:
        q: 归一化后的查询，``[num_v, dk]`` fp32。
        k: 归一化后的键，``[num_v, dk]`` fp32。
        v: 值，``[num_v, dv]`` fp32。
        beta: 写强度，``[num_v]`` fp32。
        g: 对数衰减 ``<= 0``，``[num_v]`` fp32。
        st: 循环状态 ``[num_v, dk, dv]`` fp32，原地更新。
        out: 输出缓冲区 ``[num_v, dv]`` fp32，原地写入。

    Returns:
        ``out``，便于调用链串接。

    Raises:
        RuntimeError: 扩展不可用。
    """
    _delta_call(q, k, v, beta, g, st, out, None, 0)
    return out


def delta_batch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor,
                g: torch.Tensor, st: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """批处理的门控 delta 步，每个批条目一行状态。

    Args:
        q: 归一化后的查询，``[B, num_v, dk]`` fp32。
        k: 归一化后的键，``[B, num_v, dk]`` fp32。
        v: 值，``[B, num_v, dv]`` fp32。
        beta: 写强度，``[B, num_v]`` fp32。
        g: 对数衰减 ``<= 0``，``[B, num_v]`` fp32。
        st: 循环状态 ``[B, num_v, dk, dv]`` fp32，原地更新。
        out: 输出缓冲区，``[B, num_v, dv]`` fp32，原地写入。

    Returns:
        ``out``，便于调用链串接。

    Raises:
        RuntimeError: 扩展不可用。
    """
    _delta_call(q, k, v, beta, g, st, out, None, 0)
    return out


def delta_record(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                 beta: torch.Tensor, g: torch.Tensor, st: torch.Tensor,
                 out: torch.Tensor, record: torch.Tensor, slot: int) -> torch.Tensor:
    """推进一条 GDN 状态，并记录可重放的秩一更新。

    Args:
        q, k, v, beta, g, st, out: 一步递推的输入、原地状态和输出缓冲。
        record: ``[C, heads, dk + dv + 1]`` fp32 更新日志。
        slot: 当前校验位置。

    Returns:
        ``out``。

    Raises:
        RuntimeError: 扩展不可用或输入布局不符合内核契约。
    """
    _delta_call(q, k, v, beta, g, st, out, record, int(slot))
    return out


def delta_batch_record(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                       beta: torch.Tensor, g: torch.Tensor, st: torch.Tensor,
                       out: torch.Tensor, record: torch.Tensor,
                       slot: int) -> torch.Tensor:
    """批量推进 GDN 状态，并逐行记录可重放的秩一更新。

    Args:
        q, k, v, beta, g, st, out: 批量递推的输入、原地状态和输出缓冲。
        record: ``[B, C, heads, dk + dv + 1]`` fp32 更新日志。
        slot: 当前校验位置。

    Returns:
        ``out``。

    Raises:
        RuntimeError: 扩展不可用或输入布局不符合内核契约。
    """
    _delta_call(q, k, v, beta, g, st, out, record, int(slot))
    return out


def delta_commit(base: torch.Tensor, record: torch.Tensor, st: torch.Tensor,
                 steps: int) -> torch.Tensor:
    """从基态按顺序重放若干步 GDN 秩一更新。

    Args:
        base: ``[heads, dk, dv]`` fp32 校验前状态。
        record: ``[C, heads, dk + dv + 1]`` fp32 更新日志。
        st: ``[heads, dk, dv]`` fp32 提交目标。
        steps: 要接受并重放的前缀长度。

    Returns:
        更新后的 ``st``。

    Raises:
        RuntimeError: 扩展不可用或输入布局不符合内核契约。
    """
    heads, dk, dv = (int(v) for v in base.shape)
    _attn_ops().shim.delta_commit(_ptr(base), _ptr(record), _ptr(st), heads, dk, dv,
                                  int(steps), _stream() or 0)
    return st


def delta_normed(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                 b: torch.Tensor, a: torch.Tensor, neg_expA: torch.Tensor,
                 dt_bias: torch.Tensor, z: torch.Tensor, nw: torch.Tensor,
                 st: torch.Tensor, out: torch.Tensor, qscale: float,
                 rep: int, record: torch.Tensor | None = None,
                 slot: int = 0) -> torch.Tensor:
    """单解码 token 的头预处理、递推与门控输出归一化，一次启动。

    替代 ``delta_prologue`` -> ``delta`` -> ``rms_norm_gated`` 那条链：它每层每个
    token 要写并重读约 48 KB 的 fp32 暂存，只为在三个内核之间传三个值。为什么每个
    值头一个 block 就够把这一切留在片上，见 ``kernels/delta.cu`` 里的
    :func:`delta_normed`。

    Args:
        q: 原始查询，每个键头一行，``[num_k, dk]`` fp32。
        k: 原始键，``[num_k, dk]`` fp32。
        v: 值，``[num_v, dv]`` fp32。
        b: beta logits，``[num_v]`` bf16。
        a: 衰减 logits，``[num_v]`` bf16。
        neg_expA: ``-exp(A_log)``，``[num_v]`` fp32。
        dt_bias: 衰减偏置，``[num_v]`` fp32。
        z: 输出归一化用的门投影，``[num_v, dv]``。
        nw: 门控归一化的增益，``[dv]`` bf16，按值头。
        st: 循环状态，``[num_v, dk, dv]`` fp32，原地更新。
        out: 输出缓冲区，``[num_v, dv]`` fp32，原地写入归一化后的结果。
        qscale: ``dk ** -0.5``。
        rep: 每个键头对应的值头数。
        record: 可选的 ``[width, num_v, dk + dv + 1]`` fp32 更新日志。
        slot: 日志中的校验位置。

    Returns:
        ``out``，便于调用链串接。

    Raises:
        RuntimeError: 扩展不可用。
    """
    shim = _attn_ops().shim
    nv, dk, dv = (int(v_) for v_ in st.shape)
    rc = shim.delta_normed(_ptr(q), _ptr(k), _ptr(v), _ptr(b), _ptr(a), _ptr(neg_expA),
                           _ptr(dt_bias), _ptr(z), _ptr(nw), _ptr(st), _ptr(out),
                           float(qscale), int(rep), nv, dk, dv,
                           _dcode(out), _dcode(z),
                           0 if record is None else _ptr(record), int(slot),
                           0 if record is None else int(record.shape[0]), _stream() or 0)
    if rc:
        raise RuntimeError(
            f"coffin_delta_normed rejected the geometry (dk={dk} dv={dv} rep={rep}, "
            f"code={rc}); keep the delta_prologue/delta/rms_norm_gated chain for this model")
    return out


def delta_normed_seq(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     b: torch.Tensor, a: torch.Tensor, neg_expA: torch.Tensor,
                     dt_bias: torch.Tensor, z: torch.Tensor, nw: torch.Tensor,
                     st: torch.Tensor, out: torch.Tensor, qscale: float,
                     rep: int, record: torch.Tensor | None = None,
                     slot: int = 0) -> torch.Tensor:
    """c 个校验 token 的融合链，一次启动跑完整个投机窗口。

    算术与逐 token 调用 :func:`delta_normed` 完全相同，只是把 c 步串进同一个内核的
    循环里。省下来的不是计算量而是发射——单 token 那次调用无论一行还是八行都要
    70–105 us，几乎不随行数变化，所以校验窗口里每层多付的 ``c - 1`` 次调用是纯加性
    的固定延迟。

    Args:
        q: 原始查询，``[steps, num_k, dk]`` fp32；最内层连续，行距可大于行宽。
        k: 原始键，形状同 ``q``。
        v: 值，``[steps, num_v, dv]`` fp32。
        b: beta logits，``[steps, num_v]`` bf16。
        a: 衰减 logits，``[steps, num_v]`` bf16。
        neg_expA: ``-exp(A_log)``，``[num_v]`` fp32。
        dt_bias: 衰减偏置，``[num_v]`` fp32。
        z: 输出归一化用的门投影，``[steps, num_v, dv]``。
        nw: 门控归一化的增益，``[dv]`` bf16，按值头。
        st: 循环状态，``[num_v, dk, dv]`` fp32，跨 step 延续并原地更新。
        out: 输出缓冲区，``[steps, num_v, dv]``，原地写入归一化后的结果。
        qscale: ``dk ** -0.5``。
        rep: 每个键头对应的值头数。
        record: 可选的 ``[width, num_v, dk + dv + 1]`` fp32 更新日志。
        slot: 第一个 step 在日志里的位置，后续 step 依次 +1。

    Returns:
        ``out``，便于调用链串接。

    Raises:
        RuntimeError: 扩展不可用。
    """
    shim = _attn_ops().shim
    nv, dk, dv = (int(v_) for v_ in st.shape)
    rc = shim.delta_normed_seq(_ptr(q), _ptr(k), _ptr(v), _ptr(b), _ptr(a), _ptr(neg_expA),
                               _ptr(dt_bias), _ptr(z), _ptr(nw), _ptr(st), _ptr(out),
                               float(qscale), int(rep), nv, dk, dv,
                               _dcode(out), _dcode(z), int(q.shape[0]),
                               int(q.stride(0)), int(k.stride(0)), int(v.stride(0)),
                               int(b.stride(0)), int(z.stride(0)),
                               0 if record is None else _ptr(record), int(slot),
                               0 if record is None else int(record.shape[0]), _stream() or 0)
    if rc:
        raise RuntimeError(
            f"coffin_delta_normed_seq rejected the geometry (dk={dk} dv={dv} rep={rep}, "
            f"code={rc}); keep the three-op chain for this model")
    return out


# 超连接（Gated Residual）
#
# 只剩分组归一化这一支：``kernels/hyper.cu`` 里的 ``grouped_rms_norm_kernel``。原来的
# ``hyper_down`` / ``hyper_up`` 是两个吃 ``[K, N]``（kn）权重的投影内核，kn 布局整条
# 删除后它们也一起撤下，``models/qwen4exp/gated_residual.py`` 会退回
# ``backends/reference`` 的同名实现。归一化不吃投影权重，所以留在这里。


def _hyper_rows(x: torch.Tensor) -> torch.Tensor:
    """把 ``[..., width]`` 摊平成内核要求的 ``[rows, width]`` 连续张量。

    不用 :func:`_pitch_rows`：那个按行间距做 ``as_strided`` 视图，而
    ``hyper.cu`` 的内核用 ``ptr + row * width`` 直接寻址、并在入口
    ``TORCH_CHECK(is_contiguous())``。带间距的视图会被内核拒绝；更糟的是把它
    ``reshape`` 成连续的内存会静默拷贝，于是内核写进临时张量、调用方拿到旧值。

    Args:
        x: 最后一维是通道宽的激活。

    Returns:
        二维连续视图（本身连续时是同一个张量的视图，不是拷贝）。

    Raises:
        ValueError: ``x`` 不是连续布局。这里显式拒绝，因为静默拷贝会让写入丢失。
    """
    if not x.is_contiguous():
        raise ValueError("hyper ops need contiguous activations")
    return x.reshape(-1, x.shape[-1])


def hyper_norm(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor,
               hidden: int, hc: int, eps: float = 1e-6) -> torch.Tensor:
    """经融合内核做分组零中心 RMSNorm：``hc`` 路流各自在 ``hidden`` 维上归一化。

    不复用 :func:`rms_norm`：那个对整条最后一维归一化，用在 4 路残差流上会把各路
    能量摊平，形状全对、尺度全错。

    Args:
        x: 残差流，``[..., hidden * hc]`` 连续。
        w: 零中心权重，``[hidden * hc]``，前向按 ``1 + w``。内核只接受 fp32，
            这里只在 dtype 不符时拷一次。
        out: 目的地，形状与 dtype 同 ``x``，原地写入。
        hidden: 每路宽度。
        hc: 路数，内核上限 8。
        eps: 方差下限。

    Returns:
        ``out``，便于调用链串接。

    Raises:
        RuntimeError: 扩展不可用。
        ValueError: 某个操作数不是连续布局。
    """
    act = _compute_dtype(x)
    gain = w if w.dtype == torch.float32 else w.to(torch.float32)
    a = _hyper_rows(act)
    o = _hyper_rows(out)
    _attn_ops().shim.hyper_norm(_ptr(a), _ptr(gain.contiguous()), _ptr(o), int(a.shape[0]),
                                int(hidden), int(hc), float(eps), _dcode(a), _stream() or 0)
    return out



#: 三角求解里每个块的行数，从大到小。在本卡 L=512 上对块大小 32/64/128/256 实测：
#: 256 最快（2.48 ms），128 是病态的（9.3 ms），所以这个选择比算法本身更要紧。试过
#: 的每种块大小都正确到 1e-07，因此它纯粹是个速度旋钮——这也是行数不能被最优块整除
#: 时改挑更小的块的原因。
_SOLVE_BLOCKS = (256, 128, 64, 32, 16, 8, 4, 2, 1)


def _solve_block(rows: int) -> int:
    """挑一个能整除 ``rows`` 的最大块大小。

    行数是 delta 规则的分块长度，它并不总是大到能容下最快块大小的 2 的幂：短
    prompt 用 64 的分块预填充，而 64 没有 256 行的块。拒绝这种情况会让整个预填充
    失败，所以块大小改为自适应。

    Args:
        rows: 三角系统的行数。

    Returns:
        :data:`_SOLVE_BLOCKS` 里能整除 ``rows`` 的最大选择器。

    Raises:
        ValueError: ``rows`` 为零或负数，那是调用方的 bug 而不是能自适应的形状。
    """
    if rows <= 0:
        raise ValueError(f"triangular_solve: row count must be positive, got {rows}")
    for block in _SOLVE_BLOCKS:
        if rows % block == 0:
            return block
    return 1


def _solve_blocked(lower: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """纯 torch 的块回代，作为 :func:`triangular_solve` 的回退。

    只在扩展不可用时使用。留着而不是删掉，因为它完全不需要工具链，所以一个能枚举
    后端、走参考实现路径的主机仍然可以端到端跑这个算子。

    Args:
        lower: ``[bn, L, L]`` 单位下三角系数矩阵。只读严格下三角。
        rhs: ``[bn, L, n]`` 右端项。

    Returns:
        ``[bn, L, n]``，dtype 同 ``rhs``。在 L = 512/1024/2048 上对主机侧 fp64
        解验证过，相对误差 1.1e-07。

    Raises:
        ValueError: 行数不是正数。任何正数都处理得了：块大小会自适应，所以 64 行的
            短预填充分块也能跑。
    """
    bn, rows, cols = rhs.shape
    block = _solve_block(rows)
    out = torch.empty_like(rhs)
    eye = torch.eye(block, dtype=rhs.dtype, device=rhs.device)
    for start in range(0, rows, block):
        stop = start + block
        acc = rhs[:, start:stop, :]
        if start:
            # 已经解出的部分每块贡献一次 GEMM，正是它把串行深度从 rows 降到
            # rows/block。
            acc = acc - torch.bmm(lower[:, start:stop, :start], out[:, :start, :])
        diag = lower[:, start:stop, start:stop]
        # ``eye`` 用广播而不是每块实体化：在预填充能到的批大小上，bn 份 256x256
        # fp32 块就是 64 MB。
        inv = eye.expand(bn, block, block)
        twice = 2.0 * inv
        for _ in range(max(1, (block - 1).bit_length())):
            inv = torch.bmm(inv, twice - torch.bmm(diag, inv))
        out[:, start:stop, :] = torch.bmm(inv, acc)
    return out


def triangular_solve(lower: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """为单位下三角 ``A`` 解 ``A X = B``，完全不调库。

    ``torch.linalg.solve_triangular`` 和别的 ``linalg`` 求解路径在这个平台上都用
    不了：CoreX 的 torch 构建从一个写死的
    ``/opt/sw_home/local/cuda/lib64/libcusolver.so`` 里加载 cuSOLVER，而本集群上
    没有那个文件，所以调用在 ``dlopen`` 里就失败，而不是抛一个正常的「不支持」错误。
    测了五条路径，五条都这样失败，这就是为什么这个操作属于契约：回退是平台事实，
    不是模型的数学。

    内核就是一次朴素前向替换，每个右端项一列一个线程。替换在不同列之间是独立的，
    所以它不需要屏障，也不对 warp 宽度作任何假设——而且它一次启动解完整个分块，
    而下面的块形式要 ``rows / block`` 次 GEMM 加每块 ``log2(block)`` 次批量矩阵乘
    才建得起每个对角块的逆。

    Args:
        lower: ``[bn, L, L]`` 单位下三角系数矩阵。对角线按 1 处理，严格上三角永远
            不读。
        rhs: ``[bn, L, n]`` 右端项。

    Returns:
        ``[bn, L, n]``，dtype 同 ``rhs``。

    Raises:
        RuntimeError: 操作数不是内核能索引的形状，包括 ``L`` 大到超过一个 block 的
            共享内存。形状错误不走块回退：在形状错误上悄悄换实现会把调用方的 bug
            藏到一条更慢的路径后面。
    """
    try:
        solve = _solve_ops()
    except (RuntimeError, Exception):
        # 本主机没有工具链：纯 torch 替换仍然给出答案，让结构性调用或参考实现路径的
        # 调用方照常工作。
        return _solve_blocked(lower, rhs)
    # 两个操作数都必须做成连续的，而 ``rhs`` 这一条是在真卡上被否掉之后才确认的。
    #
    # ``solve.cu`` 的文件头确实写着它「支持批步长 0，也就是 ``eye.expand`` 直接可用」，
    # 而 ``solve_kernel`` 里的 ``rhs_b = rhs + b * ldr`` 也确实是按 ``ldr`` 读的 ——
    # 但**同一个 ``ldr`` 也被用来定位输出**：``out_b = out + b * ldr``，而 ``out`` 是
    # 内核自己 ``torch::empty`` 出来的连续张量。于是 ``rhs`` 是 ``expand`` 时 ``ldr == 0``，
    # 全部 ``bn`` 个批条目都写进第 0 个 —— 其余部分是未初始化的显存。
    #
    # 实测后果（作业 4363）：贪心 ids 从 ``[6, 24218, 10, 4838, 1665, 15]`` 变成
    # ``[2, 18, 22, 22, 22, 22]``，且两轮之间还在变（读到的垃圾每次都不同）。
    # 所以「省掉一次实体化」这句话在契约上是对的、在实现上是错的：要省这 16.78 MB，
    # 得给 ``rhs`` 和 ``out`` 各传一个批步长，那是改内核，不是改这一行。
    return solve.triangular_solve(lower.contiguous(), rhs.contiguous(),
                                  stream=_stream())


#: 本后端用内核承担的算子。按「适配器与算子同名」的约定从本模块收集，所以写了适配器
#: 就是原生条目，没写就自然交给 :func:`ops` 从参考后端补齐。
#:
#: 手写清单曾经是第三个注册点，而且它是这里唯一会静默失败的一个：一个适配器、一条头文件
#: 声明、一个绑定、一个协议字段和一份参考实现都可以先落地，而算子在这里仍然缺席——然后
#: 它就会在本后端上悄悄跑 *参考* 实现。那种状态下的测量读出来是一个干净的零，正是本仓库
#: 用另一种伪装记录过两次的「陈旧缓存是看不见的」失败。现在名字即注册：``tests``
#: 只需检查同名约定本身，而不是逐条比对两份清单。
#:
#: 顺序来自 :attr:`Ops.NAMES`，与旧手写 dict 的插入顺序不同；表按名字取值，顺序无语义。
_NATIVE: dict[str, OpFn] = native_ops(globals())

#: 本后端所有内核的表；缺席的由解析器补齐。
_OPS = Ops.from_mapping(_NATIVE)

#: 本后端在 *某种* 主机上有内核的名字。按设计就与主机无关：*这里* 能跑什么看
#: ``caps().native``。
NATIVE: frozenset[str] = frozenset(_NATIVE)

#: 被捕获的解码步是否已在这张卡上 *实测* 能用。做成显式常量而不是推导出来的，因为
#: 这是两个不同的问题：完整的算子表说明每个内核都在，而这里说的是 **整个解码步** 的
#: 重放被观察到确实重读了输入，并且和 eager 路径一致。
#:
#: 2026-10-26 一次真实失败之后置为 ``False``，它的失败方式本身就是教训。
#: ``backends/corex/diag/check_graph.py`` 只捕获了 ``linear`` 和 ``rms_norm`` 两个
#: 算子（而且是孤立的），报告 ``capture rel=2.551e-03, replay rel=2.744e-03,
#: output moved by 2.910e+02``。那是对这两个算子的真实测量，却被当成解码步可捕获的
#: 证据。两者不是同一个命题：逐算子捕获既没有走过 32 层的调用图，也没有走到分页
#: 注意力窗口、int8 池和原地循环状态。
#:
#: 第一次真正重放解码步就把设备 **挂死了**。``capture()`` 本身成功（2396 ms），日志
#: 打出 ``decode path: graph replay``，然后什么都没有：任务烧完整个 25 分钟的墙钟被
#: 判为 TIMEOUT，没有 traceback，也没有 ``.err`` 文件。这种特征说明内核在设备上自旋
#: 或等待，而不是段错误——段错误会立刻退出并留下痕迹。
#:
#: 那次挂死是 **调用顺序** 的 bug，不是捕获的 bug：探针先预填充 prompt，*然后* 才调
#: ``capture()``，而它开头会重置模型并跑自己的预填充。那会把 prompt 的 KV 缓存和
#: 循环状态抹掉，于是重放驱动注意力看过一个从未写过的窗口。
#: ``run_chat.generate`` 的顺序反过来——``reset()``、``capture()``、``reset()``，
#: 然后 ``prefill()``——当时的 ``probe_chat.py`` 也是这样（该探针已随 ``diag/`` 删除）。
#:
#: 顺序修正之后，整个解码步被完整测量过。验收跑是一个进程里六轮
#: capture+prefill+replay 的探针——每一轮都以 *相同*
#: 的 token ``[5, 6, 24218, 10]`` 和有限 logits 完成：
#:
#:   backend=corex native=18/18 graph=1, loaded in 35.1 s
#:   capture ok in 3.1s / 2.0s / ...        graph_ready=True (every round)
#:   prefill ok, logits finite=True
#:   decode ok, tokens=[5, 6, 24218, 10] finite=True     (x6)
#:   completed attempts: [0, 1, 2, 3, 4, 5] of 6
#:
#: 反复 capture+replay 输出完全相同，才证明了重放确实重读输入；六次重复则证明它不是
#: 一次侥幸。同一 prompt 上对 eager 路径的速度：52.41 ms/step 对 77.09 ms/step，即
#: **1.47x**。
#:
#: **观察到过一次挂死，而那不是图的问题。** 一次运行在
#: ``graph replay`` 之后停住且没有 traceback，几分钟后 *同一台节点* 连 8x8 矩阵乘都
#: 做不完——它的 context 早就在自旋了。所以操作
#: 规则是：先在一台节点上跑个 8x8 矩阵乘，再怀疑某次运行，因为卡住的 context 和卡住
#: 的内核产生同样的日志，而日志里没有任何东西能区分它们。每个
#: 启动器现在都先做这项检查，检查不过就以退出码 64 结束，连模型都不碰。
#:
#: ``check_graph.py`` 仍然只覆盖两个算子，它的文件头也这么写；它是廉价的回归检查，
#: 不是验收门禁。
#:
#: **2026-10-27 重新置回 ``False``，而且必须保持 ``False``。**
#:
#: 三条独立的理由，任何一条都足够：
#:
#: 1. *它会随机弄死卡。* 相同代码在可比节点上产生过两种结果——一次 52.41 ms 的完整
#:    步，和一次在 ``graph replay`` 之后彻底停住、无 traceback 无 stderr 的运行。卡死
#:    之后杀不掉：进程停在不可中断状态而设备在自旋，节点不重置这张卡就废了。移植期间
#:    这个分区上这样丢了四张卡。只是「有时安全」的东西不值得在共享硬件上打开。
#: 2. *它不再更快。* 上面的 1.47x 是对着一条尚未优化的 eager 路径测的。split-K 修复
#:    （50.53 -> 43.46 ms）和 delta 重写（43.46 -> 37.98 ms）之后，同一 prompt 上重放
#:    实测 46.97 ms——**0.93x，是退化**。图消除的是启动开销；内核变快之后，剩下的那点
#:    开销已经小于重放本身的代价。
#: 3. *它针对的开销在别处消除更合适。* ``launch`` 次数和主机时间在
#:    ``docs/qwen4/QWEN4-PERF.md`` 里直接在 eager 路径上攻，那里出 bug 会报错，而
#:    不是把设备吃掉。
#:
#: 不要为了追那个旧的 1.47x 重新打开它。上面的重复运行证据表明图的 *机制* 在这个驱动
#: 上（当它真的能用时）是可行的；它没有表明这个取舍可以接受，而且它是对着一个已不存在
#: 的基线测的。将来任何尝试都必须用可证明跨运行稳定的重放打败 37.98 ms 的 eager，并且
#: 只能在赔得起的卡上做。
_GRAPH_VERIFIED = False

#: 被捕获的解码步会调用的算子。只有当其中每一个都是自写内核时设备图才安全：参考实现
#: 回退会走到 ``cuBLAS``，它在这张卡上无法捕获，于是捕获要么失败，要么（更糟）重放
#: 陈旧输出。``linear_prefill`` 故意 *不* 在这个集合里：它是预填充区间，从来不被
#: 捕获，而且是唯一一个会调库的条目（分块调，以留在这张卡能算对的宽度内）。
_DECODE_OPS: frozenset[str] = frozenset({
    "linear", "linear_m", "linear_m_verify", "rms_norm",
    "add_rms_norm", "rms_norm_gated",
    "kv_quant", "silu_mul", "rope_apply", "attn_decode",
    "attn_paged", "attn_paged_batch", "attn_paged_multi_q", "conv1d_silu",
    "conv1d_silu_seq", "conv1d_silu_batch", "delta_prologue", "delta_prologue_batch",
    "delta_prep_seq", "delta", "delta_record", "delta_batch",
    "delta_batch_record", "delta_commit", "delta_normed",
    "hyper_norm",
})


@functools.lru_cache(maxsize=1)
def caps() -> Caps:
    """返回本后端在这台主机上的能力记录。

    Returns:
        一个记录：``device="cuda"``（CoreX 实现了那个命名空间），两种 dtype 都是
        bf16，因为预填充 GEMM 在本卡上 bf16 和 fp16 都到 32.6 TFLOP/s，转换不值
        它的流量；``weight_layout="nk"``：契约与加载器一致，都是 ``[N, K]``。库 GEMM 与张量核
        GEMM 用 ``transa = T`` 直接吃这份 ``[N, K]``，一次转置都不做（当初库的 ``[N, K]``
        解码 GEMV 只有 16-30 GB/s、而库的 ``[K, N]`` 有 360-406 GB/s，那是一次库对库的
        比较，不是布局对布局；现在库 GEMM 走 TN、解码 int8 走 nk，实测差 1.02-1.63 倍）；
        ``attn_split=32``：切片轴就是内核的 ``grid.y``，切片数由 ``part_acc.size(-2)`` 读，
        所以改这里就够、不用改内核。切分是给键范围*分区*，各块读的是**不相交**的区间，所以
        提高它不增加 KV 流量，只多出 ``part_acc``/``part_ml`` 那一趟（几十 KB）和 combine。

        早先写的是 8，依据是单层孤立内核实测（合成 int8 池）4k 1.407→0.827 ms、
        32k 11.17→3.55、64k 24.49→6.22，8 与 16 处在同一平台因此取了 8。那是**孤立内核**
        的结论，整步是另一回事：2026-11-12 在整步上量 8 → 32（Qwen3.5-4B-int8kn、单流），
        decode 在 4k/16k/65k 是 25.81→24.01、36.66→34.12、70.13→62.62 ms
        （-7.0% / -6.9% / -10.7%），verify c=4 在同上下文是 47.68→44.78、76.73→72.37、
        187.19→169.52 ms（-6.1% / -5.7% / -9.4%）。上下文越长收益越大，说明这段的瓶颈是
        **并行度不足以掩盖访存延迟**，不是带宽、也不是 ALU 吞吐——这也解释了为什么几次
        「省流量」的改法（折叠 draft、W8A8 宽 M）全部无效。

        不要再往上加：128（16k 上下文）实测启动约 20 秒后把整个沙箱作业的 runner 卡死，
        心跳写盘停止、作业从调度器消失，属于驱动侧不可中断等待那一类。``tuning.py`` 因此
        把 ``COFFIN_ATTN_SPLIT`` 的上限压到后端值的 4 倍。

        ``graph`` 和 ``batch_attn`` 是推导出来的，不是声明的。只有当解码步碰到的
        每个算子都有自己的内核（:data:`_DECODE_OPS`）时 ``graph`` 才为真，因为参考
        实现回退会走到 ``cuBLAS``，而它在这里无法捕获。两者都靠推导，横幅就无法
        声称表里没有的能力。

        ``graph`` 还额外要求 :data:`_GRAPH_VERIFIED`：表完整是必要条件而不是充分
        条件，因为捕获还必须被 *实测* 在这张卡上重读输入。在那项检查通过之前，这里
        填 ``True`` 等于承诺一张会重放陈旧输出的图，而那正是这个标志要挡的失败模式。
    """
    native = frozenset(_OPS.implemented()) if usable() else frozenset()
    return Caps(
        name=NAME,
        device="cuda",
        device_index=0,
        device_name=_device_name(),
        arch=_arch(),
        native=native,
        act_dtype=torch.bfloat16,
        weight_dtype=pack.WEIGHT_DTYPE,
        graph=_GRAPH_VERIFIED and _DECODE_OPS <= native,
        autotune=True,
        attn_split=32,
        batch_attn="attn_paged_batch" in native,
        int8_weights=True,
        recurrent_state=True,
        weight_layout="nk",
    )


@functools.lru_cache(maxsize=1)
def _resolve_device() -> torch.device:
    """返回本后端所有分配所在的设备。

    Returns:
        ``torch.device("cuda", 0)``，懒解析，所以仅仅 import 本模块不会初始化运行时。
    """
    return make_device(caps())


def __getattr__(attr: str) -> Any:
    """解析 :class:`Backend` 协议里那些懒计算的模块属性。

    ``device`` 不能做成模块常量：构建能力记录要读设备名，那会初始化驱动，而在纯
    CPU 主机上 import 本模块不能做这件事。Python 只对模块没有定义的名字调用这个
    钩子，所以正常的属性路径不受影响。

    Args:
        attr: 正在查找的属性。

    Returns:
        解析后的属性。

    Raises:
        AttributeError: 该名字不是懒解析的属性之一。
    """
    if attr == "device":
        return _resolve_device()
    raise AttributeError(f"module {__name__!r} has no attribute {attr!r}")


@functools.lru_cache(maxsize=1)
def ops() -> Ops:
    """返回本后端的总算子表。

    扩展在第一次调用时构建（或附着到缓存构建），结果被记忆化，因为引擎每个请求要
    访问这张表几千次。没有内核的条目由参考后端补齐，这既让表保持完整，也让调用点
    不必写分支。

    Returns:
        一张每个契约名字都可调用的表：有内核的地方用本后端的内核，其余用参考实现。
        没有设备或构建失败时整张表都是参考实现，诊断调用方仍然可以检查它；
        :func:`usable` 是拦住这种表进不了请求的那道关。
    """
    # 先调 ``probe()``：没有卡的主机绝不能走到构建路径，因为一次失败的尝试什么都不会
    # 编出来，却会记忆化一个读起来像工具链问题的错误。
    if not usable():
        return reference.fill(Ops(), caps())
    return reference.fill(_OPS, caps())


def describe() -> str:
    """把本后端渲染成一行诊断信息。

    Returns:
        ``"corex: <state> device=... arch=... native=N/M"``，启动横幅打印的就是
        这一行，诊断转储可以 grep 它。和 ``backends.status()`` 不同，它还报告设备名
        和内核被构建时针对的 ISA，而一个错的架构构建只能从这两项诊断出来。
    """
    state = "usable" if usable() else ("present, unusable" if probe() else "absent")
    record = caps()
    suffix = "" if usable() else f" ({reason()})"
    return (f"{NAME}: {state}{suffix} device={record.device}:{record.device_index} "
            f"name={record.device_name!r} arch={record.arch!r} "
            f"native={len(record.native)}/{len(Ops.NAMES)} "
            f"layout={record.weight_layout} graph={int(record.graph)}")


def reason_table() -> str:
    """返回原生／回退的分界，供诊断转储。

    Returns:
        一行说明哪些算子跑原生、哪些来自参考实现，这样一个只做了一半的后端能报告
        自己的进度，而不是看起来要么完整要么缺席。
    """
    return (f"native={', '.join(_OPS.implemented())}; "
            f"fallback={', '.join(_OPS.missing()) or '(none)'}")


#: 模块本身满足协议；在这里点名协议是为了让签名漂移不会被类型检查器漏掉。
_PROTOCOL: type[Backend] = Backend




