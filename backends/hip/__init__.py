"""gfx906（Hygon DCU）后端：架在本包内核之上的一层契约适配器。

平台迁移的 S1 阶段。内核和它们的主机侧 shim 现在都在本包内
（``backends/hip/kernels/`` 和 ``backends/hip/shim.py``）；本模块是*契约*适配器，
让它们看起来像平台后端之一，也是 ``coffin`` 里唯一允许知道 arch 字符串、ROCm 环境
变量和 split-K 宽度的位置。

shim 的四个 ``gemv*`` 家族在这里全部收敛成三个 ``linear*`` 条目：``R``/``U``（每块
行数、K-unroll）、wide-M 内核以及 ``_dbg`` A/B 入口都是内核切块细节，所以它们变成
按形状索引的内部选择，而不是契约里的参数。

Module: ``coffin/backends/hip``。import ``coffin.backends.protocol``、
``coffin.backends.reference``（用来补没有内核的算子）以及*惰性*的
``coffin.backends.hip.shim``——惰性是为了在没有 ROCm 工具链的主机上 import 本包时
不去设置构建环境变量、也不 import ``torch.utils.cpp_extension``。

Constraints:
    * import shim 有模块级副作用：它设置 ``PYTORCH_ROCM_ARCH`` 和
      ``TORCH_EXTENSIONS_DIR``，并 import CUDA/HIP 扩展加载器，后者在 import 时读这两个
      变量。因此这一步只在 :func:`_hip` 里做，绝不在模块作用域做：一台只是让 registry
      枚举后端的纯 CPU 机器不能因此被改写配置，一个已经选中 CoreX 卡的进程也不能继承
      gfx906 的 arch。
    * 扩展在首次调用 :func:`ops()`（或首次 :func:`usable`）时才编译，不在 import 时
      编译：坏掉的工具链必须退化成即时执行路径而不是把进程带崩，且失败要记忆化，让每个
      进程只付一次尝试的代价。
    * 这里的 ``ATTN_SPLIT`` 必须等于 ``kernels/attn.cu`` 里的 ``ATTN_SPLIT``：split-K
      部分和缓冲区由这一侧持有，不匹配会让注意力结果坏掉而不是悄悄失败一个形状检查。
      S3 阶段它会连同缓冲区尺寸计算一起搬进 ``backends/hip/tuning.py``。
    * 每个条目都保留即时执行路径可达：没有内核的算子返回 ``None``，由 :func:`ops`
      从参考后端补齐，所以只编译了一部分的扩展也能跑模型。
"""

from __future__ import annotations

import functools
from typing import Any

import torch
import torch.nn.functional as F

from backends import reference
from backends.protocol import Caps, Ops, OpFn, make_device
from int8_weights import QWeight

#: 后端标识符，即 ``COFFIN_BACKEND=hip`` 选中的那个。
NAME = "hip"

#: 模块级别名，让本模块满足 :class:`~coffin.backends.protocol.Backend`。
name = NAME

#: ``kernels/attn.cu`` 的 split-K 宽度。这是内核 ABI：调用方据此分配
#: ``part_acc``/``part_ml``，内核用同样的宽度去索引它们。S3 阶段会搬进
#: ``backends/hip/tuning.py``。
ATTN_SPLIT = 8

#: 本后端内核编译所针对的 arch。选择时从驱动读取（:func:`caps`），驱动没报时用它兜底，
#: 这样构建标志不必在任何调用点写死。
DEFAULT_ARCH = "gfx906"

#: GEMV 内核实例化的「行块 / K-unroll」组合。这些是内核切块参数而不是契约：只留在本模块
#: 内，由 :data:`_TUNING` 按形状索引。
_GEMV_CFGS: tuple[tuple[int, int], ...] = ((1, 1), (2, 1), (1, 2), (2, 2), (1, 4), (4, 1))

#: 还没人调优过的形状所用的静态默认值。``(1, 2)`` 是契约里旧 ``gemv`` 的默认值，
#: 在模型的真实形状上与逐个形状的最优值相差几个百分点。
_DEFAULT_CFG = (1, 2)

#: 按形状键控的内核选择，由 :func:`_cfg` 填充。键是 ``(N, K, is_int8)`` 而不是权重张量
#: 本身，好让几何相同的两个层共用一个条目。这是后端内部缓存：模型绝不能读它或传它。
_TUNING: dict[tuple[int, int, bool], tuple[int, int]] = {}


def _hip() -> Any:
    """首次使用时 import 本后端的内核 shim。

    shim 放在本文件旁边（``backends/hip/shim.py``）而不是包根目录，所以 import 它是本
    后端的私事。它在 import 时设置 ``PYTORCH_ROCM_ARCH`` / ``TORCH_EXTENSIONS_DIR``，
    那是一个进程级副作用，不能仅仅因为有人 import 了后端包就发生：在 CoreX 主机上那会
    把扩展加载器指向 gfx906 的 arch。

    Returns:
        shim 模块。

    Raises:
        ImportError: shim 或 torch 扩展加载器缺失。调用方把这当作「没有内核」处理，
            那是一种受支持的状态。
    """
    from backends.hip import shim
    return shim


def _is_rocm_build() -> bool:
    """报告本次 torch 构建是否至少链接了 ROCm/HIP 运行时。

    ``torch.cuda.is_available()`` 不是厂商判定。好几家非 NVIDIA 加速卡通过 *CUDA*
    命名空间暴露自己，因为它们实现了那套 API：Iluvatar CoreX 是其中之一——它自己的
    引擎无条件设 ``device="cuda"``——Moore Threads 或 Hygon 的卡也是。只按命名空间
    探测，会让本后端认领一张别人的卡、报告自己可用，然后在第一次内核启动里失败，或者
    更糟：用 ``--offload-arch=gfx906`` 编译完照样跑起来。

    ``torch.version.hip`` 是区分两者的标记：每个 ROCm 构建上它被设为 HIP 版本，而在
    CUDA 构建以及本引擎瞄准的所有 CUDA 兼容厂商构建上它是 ``None``。
    ``get_device_properties`` 里的 ``gcnArchName`` 是第二个、设备侧的标记，由
    :func:`_arch` 使用。

    Returns:
        torch 构建本身就是 HIP 构建时为 ``True``。
    """
    return getattr(torch.version, "hip", None) is not None


def probe() -> bool:
    """报告 torch 是否看得到 ROCm/HIP 设备。

    只读驱动：不 import 内核、不编译、不重置设备。自动探测期间它会在每台主机上运行，
    包括纯 CPU 主机。

    两半都算数：没有插设备的 HIP 构建是 ``False``，别家的 CUDA 兼容卡也是 ``False``
    （见 :func:`_is_rocm_build`）。自动探测随后会落到那家厂商的后端，或者参考实现。

    Returns:
        仅当既是 HIP 构建*又*有设备存在时为 ``True``。
    """
    try:
        return _is_rocm_build() and bool(torch.cuda.is_available())
    except Exception:                             # pragma: no cover - 驱动报错
        return False


def _build_error() -> str | None:
    """返回记忆化的内核编译失败原因，并尝试编译一次。

    Returns:
        编译失败时的 ``"<ExceptionType>: <message>"``；扩展已加载时为 ``None``。这次
        尝试在 shim 里做了记忆化，所以再次调用不花钱。
    """
    try:
        module = _hip()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    if module.available():
        return None
    return module.last_error() or "the kernel extension did not build"


def usable() -> bool:
    """报告内核是否已编译或已缓存，这张表因此可以原生执行。

    Returns:
        没有设备或扩展编译失败时为 ``False``；registry 随后会继续尝试下一个候选，于是
        一个驱动在、工具链却不可用的主机依然能通过参考后端服务模型。
    """
    return probe() and _build_error() is None


def reason() -> str:
    """解释 :func:`usable` 为什么是 ``False``，供选择错误的文案使用。

    Returns:
        可用时为 ``""``，否则是设备状态或编译抛出的异常。
    """
    if not probe():
        return "no ROCm/HIP device visible (torch.cuda.is_available() is False)"
    err = _build_error()
    return "" if err is None else f"kernel build failed: {err}"


@functools.lru_cache(maxsize=1)
def _device_name() -> str:
    """只读一次设备名；横幅要用它，挑选 arch 也要用它。

    Returns:
        ``torch.cuda.get_device_name(0)``，没有设备时为 ``""``。这次调用会初始化 HIP
        运行时，所以做了记忆化，并且只有在 :func:`probe` 已经说过有设备之后才会到达。
    """
    if not probe():
        return ""
    try:
        return str(torch.cuda.get_device_name(0))
    except Exception:                             # pragma: no cover - 驱动报错
        return ""


@functools.lru_cache(maxsize=1)
def _arch() -> str:
    """报告目标 ISA，从驱动读取而不是写死。

    ``gcnArchName`` 是 ROCm 独有的字段，所以它同时充当厂商判定的第二半：如果一张非
    ROCm 的卡走 ``cuda`` 命名空间到了这个函数，说明 :func:`probe` 已经说过不行了，
    这里返回 ``""``，而不是虚构一个 ``gfx906`` 再照着编译。

    Returns:
        ``get_device_properties(0).gcnArchName``（DCU 上是 ``"gfx906"``），去掉
        ``:sramecc`` 之类的后缀；驱动是 ROCm 却没报 arch 时返回 :data:`DEFAULT_ARCH`。
        构建标志由这个值推导，所以换一颗芯片要换的是字符串，而不是加第二条编译路径。
    """
    if not probe():
        return ""
    try:
        props = torch.cuda.get_device_properties(0)
        arch = getattr(props, "gcnArchName", "") or ""
    except Exception:                             # pragma: no cover - 驱动报错
        arch = ""
    return str(arch).split(":")[0] or DEFAULT_ARCH


@functools.lru_cache(maxsize=1)
def caps() -> Caps:
    """返回本后端的能力记录，只探测设备一次。

    Returns:
        一条记录：``device="cuda"``（torch 的 ROCm 后端保留了这个名字）、fp16 投影
        权重（因为 gfx906 只有 packed fp16 数学）、图捕获与加载时 autotune 均开启、
        ``recurrent_state=True``（本后端的快照/恢复就是普通的 ``copy_``）。
    """
    natives = frozenset(Ops.NAMES) if usable() else frozenset()
    return Caps(
        name=NAME,
        device="cuda",
        device_index=0,
        device_name=_device_name(),
        arch=_arch(),
        native=natives,
        act_dtype=torch.bfloat16,
        # 用 fp16 而不是 bf16：这颗芯片有 packed fp16 数学而没有 packed bf16 数学，所以
        # 预填充 GEMM 在 fp16 权重上达到 13-20 TFLOP/s，bf16 上只有 2.3 TFLOP/s。激活
        # 保持 bf16，因为 int8 KV 缓存和每个融合内核都是照着它写的。
        weight_dtype=torch.float16,
        graph=True,
        autotune=True,
        attn_split=ATTN_SPLIT,
        batch_attn=True,
        int8_weights=True,
        recurrent_state=True,
    )


@functools.lru_cache(maxsize=1)
def _resolve_device() -> torch.device:
    """返回本后端所有分配应该落到的设备。

    Returns:
        ``torch.device("cuda", 0)``。模型代码用它替代 ``device="cuda"`` 字面量。
    """
    return make_device(caps())


def __getattr__(name: str) -> Any:
    """解析 :class:`Backend` 协议里那些惰性计算的模块属性。

    ``device`` 不能做成模块级常量：构建能力记录要读设备名，那会初始化 HIP 运行时，而在
    纯 CPU 主机上 import 本模块绝不能做这件事。Python 只对模块没定义的名字调用这个钩子，
    所以正常属性访问路径不受影响。

    Args:
        name: 正在查找的属性名。

    Returns:
        解析出的属性值。

    Raises:
        AttributeError: 该名字不是惰性解析的属性之一。
    """
    if name == "device":
        return _resolve_device()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _cfg(w: torch.Tensor | QWeight) -> tuple[int, int]:
    """返回这个形状该用的 ``(rows_per_block, k_unroll)``。

    选哪个是后端的私事：按形状缓存，绝不由调用方传入，也绝不存进模型族。加载时的扫描
    （S3）在 ``caps().autotune`` 开启时填充 :data:`_TUNING`；在那之前每个形状都取
    :data:`_DEFAULT_CFG`。

    Args:
        w: 投影权重，普通张量或 int8 权重。

    Returns:
        要传给内核的 ``(R, U)`` 组合。
    """
    n, k = (w.n, w.k) if isinstance(w, QWeight) else w.shape
    return _TUNING.get((n, k, isinstance(w, QWeight)), _DEFAULT_CFG)


def _split_k(part_acc: torch.Tensor | None, part_ml: torch.Tensor | None,
             nh: int, hd: int, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 split-K 中间缓冲区，调用方没给时就地分配。

    Args:
        part_acc: 调用方持有的累加器，``[nh, ATTN_SPLIT, hd]`` fp32；或 ``None``。
        part_ml: 调用方持有的滑动最大值/分母，``[nh, ATTN_SPLIT, 2]`` fp32；或 ``None``。
        nh: 查询头数。
        hd: 头维。
        like: 一个张量，分配跟随它的设备。

    Returns:
        ``(part_acc, part_ml)``。在这里分配结果正确但不可被图捕获，所以模型的捕获路径
        总是传自己的缓冲区。
    """
    if part_acc is None:
        part_acc = torch.zeros(nh, ATTN_SPLIT, hd, dtype=torch.float32,
                               device=like.device)
    if part_ml is None:
        part_ml = torch.zeros(nh, ATTN_SPLIT, 2, dtype=torch.float32,
                              device=like.device)
    return part_acc, part_ml


def _split_k_batch(part_acc: torch.Tensor | None, part_ml: torch.Tensor | None,
                   rows: int, nh: int, hd: int,
                   like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`_split_k` 的按行版本，供批处理注意力条目使用。

    Args:
        part_acc: 调用方持有的 ``[rows, nh, ATTN_SPLIT, hd]`` fp32；或 ``None``。
        part_ml: 调用方持有的 ``[rows, nh, ATTN_SPLIT, 2]`` fp32；或 ``None``。
        rows: 缓冲区的行数。
        nh: 查询头数。
        hd: 头维。
        like: 一个张量，分配跟随它的设备。

    Returns:
        ``(part_acc, part_ml)``，缺席时新分配。
    """
    if part_acc is None:
        part_acc = torch.zeros(rows, nh, ATTN_SPLIT, hd, dtype=torch.float32,
                               device=like.device)
    if part_ml is None:
        part_ml = torch.zeros(rows, nh, ATTN_SPLIT, 2, dtype=torch.float32,
                              device=like.device)
    return part_acc, part_ml


# 投影


def linear(w: torch.Tensor | QWeight, x: torch.Tensor) -> torch.Tensor:
    """解码时的权重读取：一行或少数几行激活对 ``w`` 做乘。

    ``QWeight`` 走 int8 内核，普通张量走 fp16/bf16 内核；二维激活转交
    :func:`linear_m`，因为单行内核只为单个向量编译。

    Args:
        w: ``[N, K]`` 行主序权重，普通张量或 ``QWeight``。
        x: 激活，``[K]`` 或 ``[C, K]``。

    Returns:
        ``[N]`` 或 ``[C, N]``，dtype 跟 ``x`` 走。

    Raises:
        RuntimeError: 扩展不可用却直接调用到了这个条目。经由 :func:`ops` 时会由参考
            实现替代，所以只有在调用方绕过表直接取用时才会发生。
    """
    if x.dim() != 1:
        return linear_m(w, x)
    module = _hip()
    r, u = _cfg(w)
    if isinstance(w, QWeight):
        return module.gemv_w8a16(w.wq, w.scale, x, r, u)
    return module.gemv(w, x, r, u)


def linear_m(w: torch.Tensor | QWeight, x: torch.Tensor) -> torch.Tensor:
    """小 M 的批投影：一次权重读取服务 ``x`` 的每一行。

    在三种实现之间选：小 M 内核（``M`` 为 2/4/8）、``M`` 为 16/24/32 时的 wide-M 内核，
    其余情况走预填充 GEMM。三者数值上都等价于逐行调用 :func:`linear`，所以这个选择对
    调用方不可见——也正是同一个契约名能覆盖它们的原因。

    Args:
        w: ``[N, K]`` 行主序权重，普通张量或 ``QWeight``。
        x: 激活，``[M, K]``。

    Returns:
        ``[M, N]``，dtype 跟 ``x`` 走。

    Raises:
        RuntimeError: 扩展不可用。
    """
    module = _hip()
    m = x.shape[0]
    x = x.contiguous()
    int8 = isinstance(w, QWeight)
    if m in (2, 4, 8):
        _, u = _cfg(w)
        if int8:
            return module.gemv_w8a16_m(w.wq, w.scale, x, u)
        return module.gemv_m(w, x, u)
    if int8 and m in (16, 24, 32):
        _, u = _cfg(w)
        return module.gemv_w8a16_wide(w.wq, w.scale, x, u)
    return linear_prefill(w, x)


def linear_prefill(w: torch.Tensor | QWeight, x: torch.Tensor) -> torch.Tensor:
    """预填充分块的投影（大 M），此处 GEMM 胜过流式读权重。

    Args:
        w: ``[N, K]`` 行主序权重，普通张量或 ``QWeight``。
        x: 激活，``[C, K]``。

    Returns:
        ``[C, N]``，dtype 跟 ``x`` 走。int8 权重会为这次调用反量化到
        :attr:`Caps.weight_dtype`：预填充是算力受限的，实体化一次比多次反量化的内核
        启动更便宜。
    """
    weight_dtype = caps().weight_dtype
    dense = w.dequant(weight_dtype) if isinstance(w, QWeight) else w.to(weight_dtype)
    return F.linear(x.to(weight_dtype), dense).to(x.dtype)


# 归一化


def rms_norm(x: torch.Tensor, w: torch.Tensor, stride: int = 0,
             eps: float = 1e-6) -> torch.Tensor:
    """对 ``[M, N]`` 最后一轴做零中心 RMSNorm，走融合内核。

    Args:
        x: 待归一化的行，``[M, N]``；可以是列切片，此时行的实际间距比 ``N`` 宽。
        w: 增益，``[N]``，已经带着 ``1 + w`` 的偏移。
        stride: 行与行之间的元素间距；``0`` 表示从 ``x`` 里读。
        eps: 方差下限。

    Returns:
        归一化后的行，形状与 dtype 同 ``x``。

    Raises:
        RuntimeError: 扩展不可用。
    """
    module = _hip()
    stride = x.stride(0) if stride == 0 else stride
    return module.rms_norm(x, w, stride, eps)


def rms_norm_gated(x: torch.Tensor, w: torch.Tensor, gate: torch.Tensor,
                   eps: float = 1e-6) -> torch.Tensor:
    """按头做的 RMSNorm，再乘 ``silu(gate)``，走融合内核。

    Args:
        x: 激活，``[..., dv]``。
        w: 每头的增益，``[dv]``：不是展平后的宽度，因为归一化是按头做的。
        gate: 形状与 ``x`` 相同的门控张量。
        eps: 方差下限。

    Returns:
        形状与 dtype 同 ``x``。

    Raises:
        RuntimeError: 扩展不可用。
    """
    module = _hip()
    return module.rms_norm_gated(x, w, gate, eps)


# 注意力


def attn_decode(q: torch.Tensor, gate: torch.Tensor, kcache: torch.Tensor,
                vcache: torch.Tensor, kscale: torch.Tensor, vscale: torch.Tensor,
                out: torch.Tensor, pos: torch.Tensor, rep: int, scale: float,
                part_acc: torch.Tensor | None = None,
                part_ml: torch.Tensor | None = None) -> torch.Tensor:
    """一个解码 token 在稠密 int8 KV 缓存上的注意力，走融合的 split-K 内核。

    Args:
        q: 查询，``[nh, hd]``，调用方已完成旋转和缩放。
        gate: 输出门控，``[nh, hd]``；``sigmoid`` 由内核施加。
        kcache: int8 key，``[nkv, max_seq, hd]``。
        vcache: int8 value，``[nkv, max_seq, hd]``。
        kscale: fp32 的 K scale，``[nkv, max_seq]``。
        vscale: fp32 的 V scale，``[nkv, max_seq]``。
        out: 输出缓冲区，``[nh, hd]``，原地写。
        pos: 单元素 long 张量；既是 token 索引也是因果边界。内核把越过它的条目掩掉
            而不是缩短循环，捕获下来的形状因此保持静态。
        rep: 每个 KV 头对应的查询头数。
        scale: ``hd ** -0.5``。
        part_acc: split-K 累加器，``[nh, ATTN_SPLIT, hd]`` fp32。传 ``None`` 会分配，
            结果正确但不可被图捕获。
        part_ml: split-K 的滑动最大值/分母，``[nh, ATTN_SPLIT, 2]`` fp32。

    Returns:
        ``out``，便于链式调用。

    Raises:
        RuntimeError: 扩展不可用。
    """
    module = _hip()
    pa, pm = _split_k(part_acc, part_ml, q.shape[0], q.shape[1], q)
    return module.attn(q.contiguous(), gate.contiguous(), kcache, vcache,
                       kscale, vscale, out, pos, rep, scale, pa, pm)


def attn_paged(q: torch.Tensor, gate: torch.Tensor, kpool: torch.Tensor,
               vpool: torch.Tensor, kscale: torch.Tensor, vscale: torch.Tensor,
               pos2slot: torch.Tensor, out: torch.Tensor, pos: torch.Tensor,
               max_pos: int, rep: int, scale: float,
               part_acc: torch.Tensor | None = None,
               part_ml: torch.Tensor | None = None) -> torch.Tensor:
    """一个解码 token 在某层分页 int8 池上的注意力。

    Args:
        q: 查询，``[nh, hd]``。
        gate: 输出门控，``[nh, hd]``。
        kpool: 本层的 int8 K 池，``[nslots, nkv, hd]``。
        vpool: 本层的 int8 V 池，``[nslots, nkv, hd]``。
        kscale: fp32 的 K scale，``[nslots, nkv]``。
        vscale: fp32 的 V scale，``[nslots, nkv]``。
        pos2slot: 本行的槽位表；只读它的前 ``pos + 1`` 项。
        out: 输出缓冲区，``[nh, hd]``，原地写。
        pos: 单元素 long 张量；既是 token 索引也是因果边界。
        max_pos: 窗口长度，即槽位表的行距。
        rep: 每个 KV 头对应的查询头数。
        scale: ``hd ** -0.5``。
        part_acc: split-K 累加器，``[nh, ATTN_SPLIT, hd]`` fp32。
        part_ml: split-K 的滑动最大值/分母，``[nh, ATTN_SPLIT, 2]`` fp32。

    Returns:
        ``out``，便于链式调用。

    Raises:
        RuntimeError: 扩展不可用。
    """
    module = _hip()
    pa, pm = _split_k(part_acc, part_ml, q.shape[0], q.shape[1], q)
    return module.attn_paged(q.contiguous(), gate.contiguous(), kpool, vpool,
                             kscale, vscale, pos2slot, out, pos, max_pos, rep,
                             scale, pa, pm)


def attn_paged_batch(q: torch.Tensor, gate: torch.Tensor, kpool: torch.Tensor,
                     vpool: torch.Tensor, kscale: torch.Tensor, vscale: torch.Tensor,
                     slots: torch.Tensor, context_lens: torch.Tensor,
                     out: torch.Tensor, max_seq: int, rep: int, scale: float,
                     part_acc: torch.Tensor | None = None,
                     part_ml: torch.Tensor | None = None) -> torch.Tensor:
    """一个批桶里所有行各解一个 token，整个桶只启动一次。

    Args:
        q: 查询，``[B, nh, hd]``。
        gate: 输出门控，``[B, nh, hd]``。
        kpool: 本层的 int8 K 池，``[nslots, nkv, hd]``。
        vpool: 本层的 int8 V 池，``[nslots, nkv, hd]``。
        kscale: fp32 的 K scale，``[nslots, nkv]``。
        vscale: fp32 的 V scale，``[nslots, nkv]``。
        slots: 每行的槽位表，``[B, max_seq]``。
        context_lens: 每行的 key 数，``[B]`` int32。
        out: 输出缓冲区，``[B, nh, hd]``，原地写。
        max_seq: 窗口长度，即 ``slots`` 的行距。
        rep: 每个 KV 头对应的查询头数。
        scale: ``hd ** -0.5``。
        part_acc: split-K 累加器，``[B, nh, ATTN_SPLIT, hd]`` fp32。
        part_ml: split-K 的滑动最大值/分母，``[B, nh, ATTN_SPLIT, 2]`` fp32。

    Returns:
        ``out``，便于链式调用。

    Raises:
        RuntimeError: 扩展不可用。
    """
    module = _hip()
    rows, nh, hd = q.shape[0], q.shape[1], q.shape[2]
    pa, pm = _split_k_batch(part_acc, part_ml, rows, nh, hd, q)
    return module.attn_paged_batch(q, gate, kpool, vpool, kscale, vscale, slots,
                                   context_lens, out, max_seq, rep, scale, pa, pm)


def attn_paged_multi_q(q: torch.Tensor, gate: torch.Tensor, kpool: torch.Tensor,
                       vpool: torch.Tensor, kscale: torch.Tensor, vscale: torch.Tensor,
                       slots: torch.Tensor, ctx_lens: torch.Tensor,
                       out: torch.Tensor, max_seq: int, rep: int, scale: float,
                       part_acc: torch.Tensor | None = None,
                       part_ml: torch.Tensor | None = None,
                       block_mask: torch.Tensor | None = None,
                       mask_block: int = 64) -> torch.Tensor:
    """投机校验：每行的 ``c`` 个草稿查询共享对历史 KV 的一次遍历。

    Args:
        q: 草稿查询，``[B, c, nh, hd]``。
        gate: 输出门控，形状同上。
        kpool: 本层的 int8 K 池，``[nslots, nkv, hd]``。
        vpool: 本层的 int8 V 池，``[nslots, nkv, hd]``。
        kscale: fp32 的 K scale，``[nslots, nkv]``。
        vscale: fp32 的 V scale，``[nslots, nkv]``。
        slots: 每行的槽位表，``[B, max_seq]``。
        ctx_lens: 每行的历史长度，``[B]`` int32。
        out: 输出缓冲区，``[B, c, nh, hd]``，原地写。
        max_seq: 窗口长度，即 ``slots`` 的行距。
        rep: 每个 KV 头对应的查询头数。
        scale: ``hd ** -0.5``。
        part_acc: split-K 累加器，``[B*c, nh, ATTN_SPLIT, hd]`` fp32。
        part_ml: split-K 的滑动最大值/分母，``[B*c, nh, ATTN_SPLIT, 2]`` fp32。
        block_mask: 可选的 uint8 ``[B, n_blocks]`` 保留掩码（1 表示保留）。
        mask_block: 掩码所表达的块大小。

    Returns:
        ``out``，便于链式调用。

    Raises:
        RuntimeError: 扩展不可用。
    """
    module = _hip()
    rows, nh, hd = q.shape[0] * q.shape[1], q.shape[2], q.shape[3]
    pa, pm = _split_k_batch(part_acc, part_ml, rows, nh, hd, q)
    return module.attn_paged_multi_q(q, gate, kpool, vpool, kscale, vscale, slots,
                                     ctx_lens, out, max_seq, rep, scale, pa, pm,
                                     block_mask, mask_block)


# 线性注意力（Gated DeltaNet）


def conv1d_silu(conv_w: torch.Tensor, x: torch.Tensor, conv_state: torch.Tensor,
                out: torch.Tensor) -> torch.Tensor:
    """单 token 的深度因果卷积（卷积核 4）+ SiLU，并平移 ``conv_state``。

    Args:
        conv_w: 每通道的抽头系数，``[C, 4]`` fp32。
        x: 本 token 的投影，``[C]``。
        conv_state: 前 ``3`` 个 token，``[C, 3]`` fp32；原地平移。
        out: 输出缓冲区，``[C]`` fp32，原地写。

    Returns:
        ``out``，便于链式调用。

    Raises:
        RuntimeError: 扩展不可用。
    """
    return _hip().conv1d_silu(conv_w, x, conv_state, out)


def conv1d_silu_seq(conv_w: torch.Tensor, x: torch.Tensor,
                    conv_state: torch.Tensor) -> torch.Tensor:
    """同样的卷积跑在预填充分块上，并跨块携带 ``conv_state``。

    Args:
        conv_w: 每通道的抽头系数，``[C, 4]`` fp32。
        x: 分块的投影，``[T, C]``。
        conv_state: 前 ``3`` 个 token，``[C, 3]`` fp32；原地平移。

    Returns:
        ``[T, C]`` fp32 的卷积 + SiLU 输出。

    Raises:
        RuntimeError: 扩展不可用。
    """
    return _hip().conv1d_silu_seq(conv_w, x, conv_state)


def conv1d_silu_batch(conv_w: torch.Tensor, x: torch.Tensor,
                      conv_state: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """批处理的深度因果卷积 + SiLU，每个批条目一行。

    Args:
        conv_w: 每通道的抽头系数，``[C, 4]`` fp32。
        x: 每行一个 token，``[B, C]``。
        conv_state: 每行的窗口，``[B, C, 3]`` fp32；原地平移。
        out: 输出缓冲区，``[B, C]`` fp32，原地写。

    Returns:
        ``out``，便于链式调用。

    Raises:
        RuntimeError: 扩展不可用。
    """
    return _hip().conv1d_silu_batch(conv_w, x, conv_state, out)


def delta_prologue(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   b: torch.Tensor, a: torch.Tensor,
                   neg_expA: torch.Tensor, dt_bias: torch.Tensor,
                   q_out: torch.Tensor, k_out: torch.Tensor, v_out: torch.Tensor,
                   beta: torch.Tensor, g: torch.Tensor,
                   qscale: float, rep: int) -> None:
    """单 token 的 q/k 归一化、头扩展与 beta/g 构建，融合在一个内核里。

    Args:
        q: 每个 key 头的查询，``[num_k, dk]``。
        k: 每个 key 头的 key，``[num_k, dk]``。
        v: 每个 value 头的 value，``[num_v, dv]``。
        b: beta 的 logits，``[num_v]``。
        a: 衰减的 logits，``[num_v]``。
        neg_expA: ``-exp(A_log)``，``[num_v]`` fp32。
        dt_bias: 衰减偏置，``[num_v]`` fp32。
        q_out: 输出查询，``[num_v, dk]`` fp32。
        k_out: 输出 key，``[num_v, dk]`` fp32。
        v_out: 输出 value，``[num_v, dv]`` fp32。
        beta: 输出的写入强度，``[num_v]`` fp32。
        g: 输出的对数衰减 ``<= 0``，``[num_v]`` fp32。
        qscale: ``dk ** -0.5``。
        rep: 每个 key 头对应的 value 头数。

    Returns:
        无。每个输出缓冲区都是原地写入。

    Raises:
        RuntimeError: 扩展不可用。
    """
    _hip().delta_prologue(q, k, v, b, a, neg_expA, dt_bias, q_out, k_out, v_out,
                          beta, g, qscale, rep)


def delta_prologue_batch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         b: torch.Tensor, a: torch.Tensor,
                         neg_expA: torch.Tensor, dt_bias: torch.Tensor,
                         q_out: torch.Tensor, k_out: torch.Tensor,
                         v_out: torch.Tensor, beta: torch.Tensor,
                         g: torch.Tensor, qscale: float, rep: int) -> None:
    """批处理的 :func:`delta_prologue`，每个参数都多一个前导行维。

    Args:
        q: 每个 key 头的查询，``[B, num_k, dk]``。
        k: 每个 key 头的 key，``[B, num_k, dk]``。
        v: 每个 value 头的 value，``[B, num_v, dv]``。
        b: beta 的 logits，``[B, num_v]``。
        a: 衰减的 logits，``[B, num_v]``。
        neg_expA: ``-exp(A_log)``，``[num_v]``，所有行共享。
        dt_bias: 衰减偏置，``[num_v]``，所有行共享。
        q_out: 输出查询，``[B, num_v, dk]`` fp32。
        k_out: 输出 key，``[B, num_v, dk]`` fp32。
        v_out: 输出 value，``[B, num_v, dv]`` fp32。
        beta: 输出的写入强度，``[B, num_v]`` fp32。
        g: 输出的对数衰减 ``<= 0``，``[B, num_v]`` fp32。
        qscale: ``dk ** -0.5``。
        rep: 每个 key 头对应的 value 头数。

    Returns:
        无。缓冲区都是原地写入。

    Raises:
        RuntimeError: 扩展不可用。
    """
    _hip().delta_prologue_batch(q, k, v, b, a, neg_expA, dt_bias, q_out, k_out,
                                v_out, beta, g, qscale, rep)


def delta_prep_seq(qkv: torch.Tensor, b: torch.Tensor, a: torch.Tensor,
                   neg_expA: torch.Tensor, dt_bias: torch.Tensor,
                   q_out: torch.Tensor, k_out: torch.Tensor, v_out: torch.Tensor,
                   beta: torch.Tensor, g: torch.Tensor,
                   qscale: float, rep: int) -> None:
    """预填充的头准备，写出分块扫描直接读取的头主序布局。

    Args:
        qkv: 卷积输出，``[T, conv_dim]`` fp32。
        b: beta 的 logits，``[T, num_v]``。
        a: 衰减的 logits，``[T, num_v]``。
        neg_expA: ``-exp(A_log)``，``[num_v]`` fp32。
        dt_bias: 衰减偏置，``[num_v]`` fp32。
        q_out: 输出查询，``[num_v, T, dk]`` fp32。
        k_out: 输出 key，``[num_v, T, dk]`` fp32。
        v_out: 输出 value，``[num_v, T, dv]`` fp32。
        beta: 输出的写入强度，``[num_v, T]`` fp32。
        g: 输出的对数衰减 ``<= 0``，``[num_v, T]`` fp32。
        qscale: ``dk ** -0.5``。
        rep: 每个 key 头对应的 value 头数。

    Returns:
        无。每个输出缓冲区都是原地写入。

    Raises:
        RuntimeError: 扩展不可用。
    """
    _hip().delta_prep_seq(qkv, b, a, neg_expA, dt_bias, q_out, k_out, v_out,
                          beta, g, qscale, rep)


def delta(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor,
          g: torch.Tensor, st: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """一步 gated delta 解码，原地更新 ``st``。

    Args:
        q: 归一化后的查询，``[num_v, dk]`` fp32。
        k: 归一化后的 key，``[num_v, dk]`` fp32。
        v: value，``[num_v, dv]`` fp32。
        beta: 写入强度，``[num_v]`` fp32。
        g: 对数衰减 ``<= 0``，``[num_v]`` fp32。
        st: 循环状态 ``[num_v, dk, dv]`` fp32，原地更新。
        out: 输出缓冲区 ``[num_v, dv]`` fp32，原地写。

    Returns:
        ``out``，便于链式调用。

    Raises:
        RuntimeError: 扩展不可用。
    """
    return _hip().delta(q, k, v, beta, g, st, out)


def delta_batch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                beta: torch.Tensor, g: torch.Tensor, st: torch.Tensor,
                out: torch.Tensor) -> torch.Tensor:
    """批处理的 gated delta 步，每个批条目一行状态。

    Args:
        q: 归一化后的查询，``[B, num_v, dk]`` fp32。
        k: 归一化后的 key，``[B, num_v, dk]`` fp32。
        v: value，``[B, num_v, dv]`` fp32。
        beta: 写入强度，``[B, num_v]`` fp32。
        g: 对数衰减 ``<= 0``，``[B, num_v]`` fp32。
        st: 循环状态 ``[B, num_v, dk, dv]`` fp32，原地更新。
        out: 输出缓冲区 ``[B, num_v, dv]`` fp32，原地写。

    Returns:
        ``out``，便于链式调用。

    Raises:
        RuntimeError: 扩展不可用。
    """
    return _hip().delta_batch(q, k, v, beta, g, st, out)


#: 本后端用内核提供的 17 个名字。``_NATIVE`` 就是 :func:`caps` 报告的内容；这里逐个
#: 写出来，好让缺失的内核在横幅里表现为更短的 ``native=N/17``，而不是悄悄降级。
_NATIVE: dict[str, OpFn] = {
    "linear": linear,
    "linear_m": linear_m,
    "linear_prefill": linear_prefill,
    "rms_norm": rms_norm,
    "rms_norm_gated": rms_norm_gated,
    "attn_decode": attn_decode,
    "attn_paged": attn_paged,
    "attn_paged_batch": attn_paged_batch,
    "attn_paged_multi_q": attn_paged_multi_q,
    "conv1d_silu": conv1d_silu,
    "conv1d_silu_seq": conv1d_silu_seq,
    "conv1d_silu_batch": conv1d_silu_batch,
    "delta_prologue": delta_prologue,
    "delta_prologue_batch": delta_prologue_batch,
    "delta_prep_seq": delta_prep_seq,
    "delta": delta,
    "delta_batch": delta_batch,
}

#: 本后端已有的全部内核组成的表；解析器会补齐缺席的部分。
_OPS = Ops.from_mapping(_NATIVE)


@functools.lru_cache(maxsize=1)
def ops() -> Ops:
    """返回本后端的完整算子表，并确保扩展已加载。

    首次调用时编译内核（或挂到已缓存的构建上），结果被记忆化：失败的记忆留在 shim
    里，所以坏掉的工具链每个进程只花一次尝试。没有内核的条目由参考后端补齐，整张表
    因此保持完整。

    Returns:
        17 个条目全部可调用的表：有内核的地方是本后端的实现，其余是参考实现。没有
        设备或编译失败时，整张表都是参考实现，诊断调用方仍能检查它；:func:`usable`
        才是拦着这张表进入请求的东西。
    """
    # 先看 ``probe()``，这样没有设备的主机永远到不了 ``_build_error``，也就永远不会
    # import 那个会设置构建环境变量的 shim。
    if not probe() or _build_error() is not None:
        return reference.fill(Ops(), caps())
    return reference.fill(_OPS, caps())


def reason_table() -> str:
    """返回原生/回退的划分，供诊断转储使用。

    Returns:
        一行，点名哪些算子走原生、哪些回退到参考实现，例如供 ``test_backends`` 断言
        这个划分。
    """
    table = _OPS
    return (f"native={', '.join(table.implemented())}; "
            f"fallback={', '.join(table.missing()) or '(none)'}")
