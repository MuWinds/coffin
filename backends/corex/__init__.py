"""Iluvatar CoreX（BI-V100）后端：手写 ``[K, N]`` 内核，垫在算子契约之下。

这是 ``coffin`` 里第二个原生后端。同一份生产代码在 Hygon DCU 和这张卡上跑；变的
是算子表、权重布局和 dtype 策略，三者都属于本包。两个实测事实决定了这里的一切
（COREX_PORT.md §0）：这张卡按 ``[K, N]`` 流式读权重是 168-361 GB/s，而
checkpoint 的 ``[N, K]`` 形式只有 19-30 GB/s；另外它的 ``cuBLAS`` 在
``lm_head`` 的宽度上返回 NaN。

Module: ``coffin/backends/corex``。import ``backends.protocol``、
``backends.reference``（用来补齐还没有内核的算子）、``int8_weights`` 以及自己的
``pack`` / ``kernels`` 子模块，不从 ``coffin/ops``、``models``、``engine``、
``serving`` 或 ``cache`` 里 import 任何东西。import 它不得构建任何东西：扩展在第
一次 :func:`ops` 或 :func:`usable` 调用时才加载，所以纯 CPU 主机可以只枚举后端
而不被索要工具链。

Constraints:
    * ``caps.weight_layout == "kn"`` 是一句承诺：本后端跑的每个投影都要
      ``[K, N]``。契约里的 ``w`` 仍然是 ``[N, K]``（那是 checkpoint），所以
      :func:`pack.as_kn` 每个权重转置一次并记忆化；调用
      :func:`pack.repack_state_dict` 的加载器只付这一次代价，之后常驻一份拷贝。
    * 输出列超过 ``pack.LIB_MAX_N`` 时不许用库 GEMM。本卡在 ``N = 248320`` 实测
      NaN，在 ``N = 131072`` 结果正确，所以预填充路径沿 ``N`` 分块，解码路径则完全
      不调库。这是 ``lm_head`` 算错还是算慢的分界。
    * 每次内核启动都在 ``c10::cuda::getCurrentCUDAStream()`` 上（``.cu`` 文件自己
      完成）。在 legacy stream 上启动不属于 ``torch.cuda.graph`` 的捕获范围，被捕获
      的解码步会重放陈旧输出——这是会被误读成数值问题的正确性 bug。
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
from typing import Any

import torch

from backends import reference
from backends.corex import kernels, pack
from backends.protocol import Backend, Caps, Ops, OpFn, make_device
from int8_weights import QWeight

#: 后端标识符，即 ``COFFIN_BACKEND=corex`` 选中的名字。
NAME = "corex"

#: 模块级别名，让本模块满足 :class:`~coffin.backends.protocol.Backend`。
name = NAME

#: 驱动报告不出可用的 ISA 时，扩展按这个架构编译。``ivcore10`` 是 BI-V100 的
#: 目标（COREX_PORT.md §0）；报告了不同架构串的卡就用那个值，所以换新部件时构建
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
_ROW_CHUNK = 8


def _is_corex_host() -> tuple[bool, str]:
    """在 torch 共享的 ``cuda`` 命名空间里认出本厂商。

    ``torch.cuda.is_available()`` 不是厂商测试：ROCm、CoreX 和普通 NVIDIA 驱动都
    答 True，而 ``cuda`` 这个设备类型说明不了卡是谁造的。共享命名空间下设备
    *名字* 是唯一的判据，所以这里向驱动要名字，也所以 ROCm 后端额外要求自己的
    构建标记。

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
    （COREX_PORT.md §0）。

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


def _build_error() -> str | None:
    """返回记忆下来的扩展构建失败，必要时尝试构建一次。

    Returns:
        构建失败时为 ``"<ExceptionType>: <message>"``，扩展已加载时为 ``None``。
        这次尝试在 :mod:`backends.corex.kernels` 里被记忆化，所以再问不花代价。
    """
    if kernels.available():
        return None
    return kernels.last_error() or "backends.corex.kernels.available() is False"


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
    """返回已加载的扩展，首次使用时构建。

    Returns:
        带 ``gemv_kn`` / ``gemv_kn_m`` / ``rms_norm`` / ``rms_norm_gated`` 入口的
        扩展模块。

    Raises:
        RuntimeError: 构建失败或 ``TORCH_EXTENSIONS_DIR`` 未设置。经 :func:`ops`
            走的话这种主机上永远到不了受影响的入口，因为整张表就是参考实现；直接
            调用意味着调用方绕过了表。
    """
    return kernels.load_kernels(_arch() or DEFAULT_ARCH)


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


# 投影


def linear(w: torch.Tensor | QWeight | pack.KnWeight, x: torch.Tensor) -> torch.Tensor:
    """解码期投影：一行激活对 ``[N, K]`` 的 ``w`` 做乘加。

    Args:
        w: ``[N, K]`` 投影权重——普通张量、``QWeight``（保持 int8，由 W8A16 内核
            承担，这正是声明 ``int8_weights`` 的意义），或加载器已经转置好的打包
            权重。
        x: 激活，``[K]`` 或 ``[C, K]``。

    Returns:
        ``[N]`` 或 ``[C, N]``，dtype 同 ``x``。

    Raises:
        RuntimeError: 扩展不可用且直接走到了这个入口；经 :func:`ops` 走的话由参考
            实现应答。
    """
    if x.dim() != 1:
        return _gemv_rows(w, x)
    packed = pack.as_kn(w)
    module = _kernels()
    dtype = x.dtype
    act = _compute_dtype(x).contiguous()
    if packed.int8:
        return _restore(module.gemv_kn_i8(packed.wt, act, packed.scale), dtype)
    return _restore(module.gemv_kn(packed.wt, act), dtype)


def linear_m(w: torch.Tensor | QWeight | pack.KnWeight, x: torch.Tensor) -> torch.Tensor:
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


def _gemv_rows(w: torch.Tensor | QWeight | pack.KnWeight,
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
    packed = pack.as_kn(w)
    module = _kernels()
    dtype = x.dtype
    act = _compute_dtype(x).contiguous()
    rows = act.shape[0]
    if packed.int8:
        # 每个分块只读一遍权重。在 gemv_kn_i8_m 出现之前，这个循环走的是单行内核，
        # 于是一批行每行都把整个权重读一遍：实测 M=8 是 M=1 的 8.19 倍
        # （probe_row_cost.py，五个解码形状），这就是为什么一次 c-token 投机校验要付
        # c 遍完整权重读取，在任何草稿深度都无法打平（../../../PERF_COREX.md 第 14 节）。
        #
        # 内核只接受它有实例化的宽度，而本辅助函数会被以不超过 _ROW_CHUNK 的任意行数
        # 调用（比如某个退回这里的分块预填充），所以下面 bf16 分支做的分块这里也必须
        # 做：把 24 行交给 TORCH_CHECK 写着 [1, 8] 的内核会抛异常，这正是加分块循环
        # 之前这个分支的行为。
        if rows <= _ROW_CHUNK:
            return _restore(module.gemv_kn_i8_m(packed.wt, act, packed.scale), dtype)
        out = torch.empty((rows, packed.n), dtype=act.dtype, device=act.device)
        for start in range(0, rows, _ROW_CHUNK):
            stop = min(rows, start + _ROW_CHUNK)
            out[start:stop] = module.gemv_kn_i8_m(packed.wt, act[start:stop].contiguous(),
                                                  packed.scale)
        return _restore(out, dtype)
    if rows == 1:
        return _restore(module.gemv_kn(packed.wt, act[0]), dtype)
    if rows <= _ROW_CHUNK:
        return _restore(module.gemv_kn_m(packed.wt, act), dtype)
    out = torch.empty((rows, packed.n), dtype=act.dtype, device=act.device)
    for start in range(0, rows, _ROW_CHUNK):
        stop = min(rows, start + _ROW_CHUNK)
        chunk = act[start:stop]
        if chunk.shape[0] == 1:
            out[start] = module.gemv_kn(packed.wt, chunk[0])
        else:
            out[start:stop] = module.gemv_kn_m(packed.wt, chunk.contiguous())
    return _restore(out, dtype)


def linear_prefill(w: torch.Tensor | QWeight | pack.KnWeight,
                   x: torch.Tensor) -> torch.Tensor:
    """预填充分块投影（大 ``M``），此时 GEMM 胜过流式读权重。

    库只用在这张卡已知能算对的宽度内：输出列超过 :data:`pack.LIB_MAX_N` 时它的
    GEMM 在 ``N = 248320``（就是 ``lm_head``）返回 NaN，而那正是最经不起这种错误
    的一个形状。所以宽 ``N`` 的预填充改为按行分块跑自写 ``[K, N]`` GEMV：受限的是
    带宽而不是算力，但结果是对的。

    Args:
        w: 任意可接受存储形式的 ``[N, K]`` 投影权重。
        x: 激活，``[C, K]``。

    Returns:
        ``[C, N]``，dtype 同 ``x``。

    Raises:
        RuntimeError: 扩展不可用，而宽 ``N`` 形状需要它。
    """
    packed = pack.as_kn(w)
    rows = x.shape[0]
    if packed.int8:
        # int8 是为解码 GEMV 存在的，那里减半字节数占主导。预填充 GEMM 每个 tile 要
        # 消费很多次权重，而库在这里不接受 int8 操作数，所以宽路径只反量化一次、改用
        # bf16 GEMM。这样显存里只有一种存储形式，而不是每个权重两份拷贝。
        packed = packed.dequant_kn(pack.WEIGHT_DTYPE)
    if rows < _PREFILL_MIN_M:
        # 低于预填充区间时 GEMM 每个 tile 都要读一遍权重，输给整次调用只读一遍的
        # GEMV。
        return _gemv_rows(w, x)
    dtype = x.dtype
    act = _compute_dtype(x)
    weight_dtype = packed.wt.dtype
    # GEMM 两个操作数必须同一 dtype，而本卡 bf16 和 fp16 的 TFLOP/s 一样（M=2048 时
    # 32.6），所以由权重自己的 dtype 决定，激活跟着它，而不是反过来。
    lhs = act.to(weight_dtype)
    if packed.n <= pack.LIB_MAX_N:
        return _restore(torch.matmul(lhs, packed.wt), dtype)
    out = torch.empty((rows, packed.n), dtype=weight_dtype, device=x.device)
    for start in range(0, packed.n, pack.LIB_MAX_N):
        stop = min(packed.n, start + pack.LIB_MAX_N)
        # ``[K, N]`` 矩阵的列切片不连续；这次拷贝无法避免，但会摊到 ``M`` 行上。
        # 结果拷出来而不是通过 ``out=`` 写进去，因为 ``out`` 的列切片同样不连续，
        # 而 ``matmul`` 不接受那样的输出。
        block = packed.wt[:, start:stop].contiguous()
        out[:, start:stop].copy_(torch.matmul(lhs, block))
    return _restore(out, dtype)


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
    module = _kernels()
    dtype = x.dtype
    width = x.shape[-1]
    act = _compute_dtype(x)
    rows = _pitch_rows(act, width)
    gain = w if w.dtype == act.dtype else w.to(act.dtype)
    y = module.rms_norm(rows, gain.contiguous(),
                        rows.stride(0) if stride == 0 else stride, eps)
    return _restore(y.view(x.shape), dtype)


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
    module = _kernels()
    dtype = x.dtype
    width = x.shape[-1]
    act = _compute_dtype(x)
    gate_c = _compute_dtype(gate)
    gain = w if w.dtype == act.dtype else w.to(act.dtype)
    y = module.rms_norm_gated(act.reshape(-1, width), gain.contiguous(),
                              gate_c.reshape(-1, width), eps)
    return _restore(y.view(x.shape), dtype)


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
    module = _kernels()
    width = x.shape[-1]
    act = _compute_dtype(x)
    act_add = _compute_dtype(add)
    rows = _pitch_rows(act, width)
    rows_add = _pitch_rows(act_add, width)
    gain = w if w.dtype == act.dtype else w.to(act.dtype)
    y = module.add_rms_norm(rows, rows_add, gain.contiguous(), out, 
                            rows.stride(0) if stride == 0 else stride, eps)
    return y.view(x.shape) if y.dtype == out.dtype else y.to(out.dtype).view(out.shape)


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
    module = _kernels()
    act = _compute_dtype(x)
    width = act.shape[-1]
    return module.kv_quant(_pitch_rows(act, width), out_q, out_s)


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
    module = _kernels()
    if gate.dim() == 1:
        return module.silu_mul(gate, up, out)
    width = gate.shape[-1]
    return module.silu_mul(_pitch_rows(gate, width), _pitch_rows(up, width),
                           _pitch_rows(out, width))


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
    return _kernels().rope_apply(q, k, cos, sin, q_out, k_out, rotary_dim)


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
    module = _kernels()
    pa, pm = _split_k(part_acc, part_ml, (q.shape[0],), q)
    return module.attn_decode(q.contiguous(), gate.contiguous(), kcache, vcache, kscale,
                              vscale, out, pos, rep, scale, pa, pm)


def attn_paged(q: torch.Tensor, gate: torch.Tensor, kpool: torch.Tensor,
               vpool: torch.Tensor, kscale: torch.Tensor, vscale: torch.Tensor,
               pos2slot: torch.Tensor, out: torch.Tensor, pos: torch.Tensor,
               max_pos: int, rep: int, scale: float,
               part_acc: torch.Tensor | None = None,
               part_ml: torch.Tensor | None = None) -> torch.Tensor:
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
    module = _kernels()
    pa, pm = _split_k(part_acc, part_ml, (q.shape[0],), q)
    return module.attn_paged(q.contiguous(), gate.contiguous(), kpool, vpool, kscale,
                             vscale, pos2slot, out, pos, max_pos, rep, scale, pa, pm)


def attn_paged_batch(q: torch.Tensor, gate: torch.Tensor, kpool: torch.Tensor,
                     vpool: torch.Tensor, kscale: torch.Tensor, vscale: torch.Tensor,
                     slots: torch.Tensor, context_lens: torch.Tensor,
                     out: torch.Tensor, max_seq: int, rep: int, scale: float,
                     part_acc: torch.Tensor | None = None,
                     part_ml: torch.Tensor | None = None) -> torch.Tensor:
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
    module = _kernels()
    rows, nh = q.shape[0], q.shape[1]
    pa, pm = _split_k(part_acc, part_ml, (rows, nh), q)
    # 在边界上把操作数变成稠密的，而不是相信调用方传的。内核通过 ``data_ptr()`` 把
    # ``q`` 和 ``gate`` 当成一整块平的 ``[B, nh, hd]`` 来索引，所以列视图会悄悄读到
    # 错误的字节——而构造门的自然写法正是这样的视图（融合 ``[.., 2*hd]`` 投影的
    # ``qg[..., hd:]``，最内层步长是 1，但行间距是 ``2*hd``）。实测：用带步长的门
    # 时，这个入口和第 3 层的 ``attn_paged_multi_q`` 在 3834/4096 个元素上不一致，
    # 而查询和池里的字节逐位相同，这个现象被当成内核缺陷查了好几个会话
    # （``docs/STAGE_LOCALISED.md``）。调用方本来就传稠密张量时这次拷贝是空操作，
    # 所以它属于边界而不属于每个调用点。
    return module.attn_paged_batch(q.contiguous(), gate.contiguous(), kpool, vpool, kscale,
                                   vscale, slots, context_lens.contiguous(), out, max_seq,
                                   rep, scale, pa, pm)


def attn_paged_multi_q(q: torch.Tensor, gate: torch.Tensor, kpool: torch.Tensor,
                       vpool: torch.Tensor, kscale: torch.Tensor, vscale: torch.Tensor,
                       slots: torch.Tensor, ctx_lens: torch.Tensor, out: torch.Tensor,
                       max_seq: int, rep: int, scale: float,
                       part_acc: torch.Tensor | None = None,
                       part_ml: torch.Tensor | None = None,
                       block_mask: torch.Tensor | None = None,
                       mask_block: int = 64) -> torch.Tensor:
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
    module = _kernels()
    rows, draft, nh = q.shape[0], q.shape[1], q.shape[2]
    pa, pm = _split_k(part_acc, part_ml, (rows * draft, nh), q)
    mask = block_mask if block_mask is not None else torch.empty(
        0, dtype=torch.uint8, device=q.device)
    # 在边界上把操作数变成稠密的，理由见 ``attn_paged_batch``：内核通过
    # ``data_ptr()`` 把门当成一整块平来索引，而每个调用方都是把门构造成融合投影的
    # 列切片。
    return module.attn_paged_multi_q(q.contiguous(), gate.contiguous(), kpool, vpool, kscale,
                                     vscale, slots, ctx_lens.contiguous(), out, max_seq,
                                     rep, scale, pa, pm, mask, mask_block)


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
    _kernels().conv1d_silu(conv_w, x, conv_state, out)
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
    return _kernels().conv1d_silu_seq(conv_w, x, conv_state)


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
    _kernels().conv1d_silu_batch(conv_w, x, conv_state, out)
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
    _kernels().delta_prologue(q, k, v, b, a, neg_expA, dt_bias, q_out, k_out, v_out,
                              beta, g, qscale, rep)


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
    _kernels().delta_prologue_batch(q, k, v, b, a, neg_expA, dt_bias, q_out, k_out,
                                    v_out, beta, g, qscale, rep)


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
    _kernels().delta_prep_seq(qkv, b, a, neg_expA, dt_bias, q_out, k_out, v_out, beta, g,
                              qscale, rep)


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
    return _kernels().delta(q, k, v, beta, g, st, out)


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
    return _kernels().delta_batch(q, k, v, beta, g, st, out)


def delta_normed(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                 b: torch.Tensor, a: torch.Tensor, neg_expA: torch.Tensor,
                 dt_bias: torch.Tensor, z: torch.Tensor, nw: torch.Tensor,
                 st: torch.Tensor, out: torch.Tensor, qscale: float,
                 rep: int) -> torch.Tensor:
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

    Returns:
        ``out``，便于调用链串接。

    Raises:
        RuntimeError: 扩展不可用。
    """
    return _kernels().delta_normed(q, k, v, b, a, neg_expA, dt_bias, z, nw, st, out,
                                   qscale, rep)


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
        module = _kernels()
    except RuntimeError:
        # 本主机没有工具链：纯 torch 替换仍然给出答案，让结构性调用或参考实现路径的
        # 调用方照常工作。
        return _solve_blocked(lower, rhs)
    return module.triangular_solve(lower.contiguous(), rhs.contiguous())


#: 本后端用内核承担的算子，按 :class:`Ops` 字段顺序。缺席的都由 :func:`ops` 从参考
#: 后端补齐，这就是只实现了一半的平台仍然能应答每个名字的原因。
#:
#: 这个 dict 是手工维护的，而这正是值得点名的陷阱：一个适配器、一条头文件声明、一个
#: 绑定、一个协议字段和一份参考实现都可以先落地，而算子在这里仍然缺席——然后它就会
#: 在本后端上悄悄跑 *参考* 实现。那种状态下的测量读出来是一个干净的零，正是本仓库
#: 用另一种伪装记录过两次的「陈旧缓存是看不见的」失败。``tests/test_backends.py``
#: 把它和已存在的适配器钉在一起，所以新内核不可能不带注册就跑起来。
_NATIVE: dict[str, OpFn] = {
    "linear": linear,
    "linear_m": linear_m,
    "linear_prefill": linear_prefill,
    "rms_norm": rms_norm,
    "add_rms_norm": add_rms_norm,
    "rms_norm_gated": rms_norm_gated,
    "kv_quant": kv_quant,
    "silu_mul": silu_mul,
    "rope_apply": rope_apply,
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
    "delta_normed": delta_normed,
    "triangular_solve": triangular_solve,
}

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
#: 然后 ``prefill()``——``diag/corex/probe_chat.py`` 也一样。
#:
#: 顺序修正之后，整个解码步被完整测量过。验收跑是 2026-10-26 在 gpu-node-07 上的
#: 历史 graph 验收探针——一个进程里六轮 capture+prefill+replay，每一轮都以 *相同*
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
#: **观察到过一次挂死，而那不是图的问题。** gpu-node-06 上一次运行在
#: ``graph replay`` 之后停住且没有 traceback，几分钟后 *同一个节点* 连 8x8 矩阵乘都
#: 做不完——它的 context 早就在自旋了。node-06 承载过先前那次失败的捕获。所以操作
#: 规则是：先在一个节点上跑个 8x8 矩阵乘，再怀疑某次运行，因为卡住的 context 和卡住
#: 的内核产生同样的日志，而日志里没有任何东西能区分它们。``diag/corex/`` 下的每个
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
#:    ``../../../PERF_COREX.md`` §8 里直接在 eager 路径上攻，那里出 bug 会报错，而
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
    "linear", "linear_m", "rms_norm", "add_rms_norm", "rms_norm_gated",
    "kv_quant", "silu_mul", "rope_apply", "attn_decode",
    "attn_paged", "attn_paged_batch", "attn_paged_multi_q", "conv1d_silu",
    "conv1d_silu_seq", "conv1d_silu_batch", "delta_prologue", "delta_prologue_batch",
    "delta_prep_seq", "delta", "delta_batch", "delta_normed",
})


@functools.lru_cache(maxsize=1)
def caps() -> Caps:
    """返回本后端在这台主机上的能力记录。

    Returns:
        一个记录：``device="cuda"``（CoreX 实现了那个命名空间），两种 dtype 都是
        bf16，因为预填充 GEMM 在本卡上 bf16 和 fp16 都到 32.6 TFLOP/s，转换不值
        它的流量；``weight_layout="kn"``，因为那个布局在本卡上对解码值 14-22 倍；
        ``attn_split=1``，因为注意力内核还不会按 K 切分——那是内核 ABI，而一个内核
        没实现的数值会把注意力算错而不是让它失败。

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
        attn_split=1,
        batch_attn="attn_paged_batch" in native,
        int8_weights=True,
        recurrent_state=True,
        weight_layout="kn",
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
        一张 17 个名字都可调用的表：有内核的地方用本后端的内核，其余用参考实现。
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
        ``"corex: <state> device=... arch=... native=N/17"``，启动横幅打印的就是
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
