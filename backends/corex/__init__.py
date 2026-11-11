"""Iluvatar CoreX（BI-V100）后端：手写 ``[K, N]`` 内核，垫在算子契约之下。

这是 ``coffin`` 里第二个原生后端。同一份生产代码在 Hygon DCU 和这张卡上跑；变的
是算子表、权重布局和 dtype 策略，三者都属于本包。两个实测事实决定了这里的一切
（docs/qwen4/COREX-PLATFORM.md）：这张卡按 ``[K, N]`` 流式读权重是 168-361 GB/s，而
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
import os
from typing import Any

import torch

from backends import reference
from backends.corex import kernels, pack
from backends.protocol import Backend, Caps, Ops, OpFn, make_device
from int8_weights import QWeight
from w4a8_weights import W4A8Weight

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

#: W4 权重走「反量化成 bf16 + 厂商 GEMM」的最小行数。低于它时反量化那 79 MB 的读写
#: 摊不开（``w4_dequant`` 实测 198.4 us / 401.3 GB/s，是纯带宽的固定成本），而小 M 下
#: ``_w4_rows`` 的 GEMV 只把权重读一遍就够。
#:
#: 起手取 256，实测再调：M=512 时「反量化 + matmul」935.9 us，对 ``w4_gemm_tcu`` 的
#: 1934.8 us 是 2.07 倍（作业 371，q_proj 形状 512x12288x2560）。
_W4_DEQUANT_MIN_M = int(os.environ.get("COFFIN_W4_DEQUANT_MIN_M", "256"))

#: 反量化的显存闸门：``K * N * 2`` 字节超过它就放弃这条路、回退 ``_w4_rows``。
#: 63 MB（q_proj）是正常量级；``lm_head`` 那类 248320 列的形状反量化要 1.27 GB，
#: 虽然不会立刻 OOM，但没有理由为它付这个峰值 —— 判据 4 要求「per-call 只活一张权重」。
_W4_DEQUANT_MAX_BYTES = int(os.environ.get("COFFIN_W4_DEQUANT_MAX_MB", "256")) << 20

#: 为了避开库而必须分块时，每次内核启动处理的行数。:func:`_gemv_rows` 最多把这么
#: 多行并进一次权重读取，正好对上小 M 内核的特化范围。
#:
#: **不要把它调大。** 试过改成 64（理由是内核现在支持任意 M，按 8 切会让 10 行付
#: 两遍权重），实测 B=10 从 466.71 变成 475.58 ms —— **更慢**。原因是 ``gemv_kn_m``
#: 的 grid 只按 ``N`` 分块、与 ``M`` 无关，所以行数超过 8 之后并行度不涨、每线程的
#: 串行 FMA 变多，反而比「切两块、各自占满 SM」差。这个 8 是被内核结构钉住的，
#: 不是随便定的。
_ROW_CHUNK = 8

#: 投机校验才启用的 W8A8 DP4A 路径；可用 0 关闭以对照原有 GEMV。
_W8A8_DP4A_VERIFY = os.environ.get("COFFIN_COREX_W8A8_DP4A_VERIFY", "0").strip() == "1"

#: 宽指令张量核 GEMM 的形状门限，按**权重字节数**而不是 K 或 N 单独判。
#:
#: 实测（4B 的真实预填充形状，M=2048，同一张卡同进程，``docs/TCU_INT8.md``）：
#:
#:   形状            K      N      TCU ms   库 ms    相对
#:   gate_up       2560  18432     3.44     5.75   1.67x
#:   full_attn     2560  10240     1.80     3.26   1.81x
#:   linear_attn   2560  12352     2.35     3.92   1.66x
#:   o_proj        4096   2560     0.84     1.18   1.41x
#:   k5120         5120   2560     1.12     1.33   1.20x
#:   mlp_down      9216   2560     2.14     2.01   0.94x   <- 唯一输
#:
#: 判据**不是** ``K * N``：``o_proj``（10.5 MB）与 ``k5120``（13.1 MB）都远小于任意
#: 一个 24 MB 门限，却都赢；而 ``mlp_down``（23.6 MB）最大却输。真正的区别在
#: **K 有多深**——``mlp_down`` 是 K=9216 的深归约、N 只有 2560，一次预填充要沿 K 走
#: 144 个 k-tile，TCU 的块间固定开销摊不薄；其余形状 N 宽 K 浅。所以门限落在 K 上。
#:
#: 这个数只决定走哪条路，两条路的**数值逐位相同**（int32 累加是精确的，epilogue 的
#: 舍入次数也一样），所以分派错了只会慢、不会错。``COFFIN_I8_GEMM`` 可覆盖：
#: ``tcu`` 强制走 TCU，``lib`` 强制走库做 A/B。
_TCU_MAX_K = 8192


def _use_tcu_gemm(rows: int, k: int, n: int, available: bool) -> bool:
    """决定这次预填充投影走宽指令 TCU 内核还是库。

    Args:
        rows: 激活行数 ``M``。
        k: 内维。
        n: 输出列数。
        available: 扩展里有没有 ``w8a8_gemm_tcu``。旧构建没有它，而「扩展能加载」
            不等于「这个入口在」，所以两者分开判。

    Returns:
        走 TCU 时为 ``True``。

    Raises:
        ValueError: ``COFFIN_I8_GEMM`` 是个不认识的值。静默回退会让一次 A/B 看起来
            像「两边一样快」，而实际两边跑的是同一条路。
    """
    if not available:
        return False
    forced = os.environ.get("COFFIN_I8_GEMM", "").strip().lower()
    if forced in {"tcu", "tensor", "i8x8"}:
        return True
    if forced in {"lib", "cublas"}:
        return False
    if forced:
        raise ValueError(
            f"COFFIN_I8_GEMM={forced!r} is not one of 'tcu' / 'lib'")
    return k <= _TCU_MAX_K


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


#: 已加载的扩展模块，由 :func:`_kernels` 填一次。
#:
#: 缓存的是**模块对象**而不是 ``load_kernels()`` 的返回值：那个函数自己也是记忆化的，
#: 但它前面还压着 ``_arch()`` → ``probe()`` → 驱动查询那条链。一步 decode 有几千次
#: 算子调用、每次都要问一遍，一次属性读取比那条链便宜得多。构建失败时不缓存（保持
#: ``None``），让异常路径每次照常抛。
_KERNEL_MODULE: Any = None


def _kernels() -> Any:
    """返回已加载的扩展，首次使用时构建。

    这是每次算子调用的必经之路（``linear`` / ``moe_grouped`` / ``rms_norm`` … 全走
    它），所以第一次之后只允许剩一次属性读取。实测（作业 ``q4host1``）扩展直调
    ``rms_norm`` 的主机侧成本是 6.7 us/次，而原生 ``torch`` 逐元素是 4.9 us ——
    差额就是这里的 Python 开销，必须压到接近零。

    Returns:
        带 ``gemv_kn`` / ``gemv_kn_m`` / ``rms_norm`` / ``rms_norm_gated`` 入口的
        扩展模块。

    Raises:
        RuntimeError: 构建失败或 ``TORCH_EXTENSIONS_DIR`` 未设置。经 :func:`ops`
            走的话这种主机上永远到不了受影响的入口，因为整张表就是参考实现；直接
            调用意味着调用方绕过了表。
    """
    global _KERNEL_MODULE
    module = _KERNEL_MODULE
    if module is None:
        module = kernels.load_kernels(_arch() or DEFAULT_ARCH)
        _KERNEL_MODULE = module
    return module


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
    if isinstance(w, W4A8Weight):
        return _w4_rows(w, x)
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
    if isinstance(w, W4A8Weight):
        return _w4_rows(w, x)
    if x.shape[0] > _ROW_CHUNK:
        return linear_prefill(w, x)
    return _gemv_rows(w, x)


def linear_m_verify(w: torch.Tensor | QWeight | pack.KnWeight,
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
            packed = pack.as_kn(w)
            wt, ws = packed.dp4a_weight(), packed.scale
    elif isinstance(w, pack.KnQWeight):
        if w.k % 4:
            return linear_m(w, x)
        wt, ws = w.dp4a_weight(), w.scale
    else:
        return linear_m(w, x)

    act = _compute_dtype(x).contiguous()
    rows, k = act.shape
    n = int(ws.numel())
    if rows < 2 or k % 4:
        return linear_m(w, x)

    module = _kernels()
    xq = torch.empty((rows, k), dtype=torch.int8, device=act.device)
    xs = torch.empty(rows, dtype=torch.float32, device=act.device)
    acc = torch.empty((rows, n), dtype=torch.int32, device=act.device)
    out = torch.empty((rows, n), dtype=act.dtype, device=act.device)
    module.quant_act_i8(act, xq, xs)
    module.w8a8_dp4a_nk(xq, wt, acc, out, xs, ws, 0)
    return _restore(out, x.dtype)


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
        # c 遍完整权重读取，在任何草稿深度都无法打平（docs/qwen4/QWEN4-PERF.md）。
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


def _dequant_w4(w: W4A8Weight) -> torch.Tensor:
    """把 packed W4 权重反量化成 ``[K, N]`` bf16。

    每次调用**新建**输出，绝不缓存到 module 上：``q_proj`` 一份是 63 MB，128 个投影
    常驻就是 4.5 GB，会在这个已经贴着显存上限的模型上直接 OOM。反量化的成本是
    79.6 MB 的读写（实测 198.4 us / 401.3 GB/s），只有 ``M`` 足够大时才摊得薄，
    所以这条路只在 :func:`linear_prefill` 的 ``rows >= _W4_DEQUANT_MIN_M`` 分支上用。

    Args:
        w: W4 权重，``packed`` 是 ``[ceil(K/2), N]`` uint8、``scale`` 是
            ``[ceil(K/group), N]`` fp32（**K 在行、N 在列**）。

    Returns:
        ``[K, N]`` bf16 连续张量，可直接喂 ``torch.matmul(lhs, wt)``。

    Raises:
        RuntimeError: 扩展不可用，或形状不满足 ``w4_dequant`` 的向量化约束
            （``N % 4 != 0``、``packed`` 不是 ``[ceil(K/2), N]``）。
    """
    module = _kernels()
    return module.w4_dequant(w.packed, w.scale, w.k, w.n, w.group_size)


def _w4_rows(w: W4A8Weight, x: torch.Tensor,
             activation_scale: torch.Tensor | None = None) -> torch.Tensor:
    """量化激活到 int8，再以逐组 int32 累加消费 packed W4 权重。

    主机侧成本是这个入口的主要问题：一步 decode 有 316 次单行 W4A8 投影
    （``w4a8_dot`` 计数，作业 ``q4budget2``），而实测 ``bops.linear`` 的 W4A8
    一档主机耗时 **36.4 us/次**（作业 ``q4host1``），其中原生 torch 分发只占
    4.9 us。差额几乎全是这里每次调用新建的三个张量（``xq`` / ``xs`` / ``out``）
    加 ``quant_act_i8`` 的适配：

    * ``xq`` / ``xs`` 是**纯暂存**，形状只由 ``(rows, K)`` 决定，调用方之间不共享
      也不该看见它们。按 ``(形状, 设备)`` 缓存起来复用，省掉每步 316 次分配。
    * ``out`` 必须每次新建：它是返回给调用方的结果，复用会让上一步的结果被就地
      覆盖（而调用方常常还攥着它做后续加法）。

    缓存只按形状键取，不按内容；暂存之间没有别名关系，所以复用是安全的。键里带
    设备与 dtype：``.to()`` 之后旧缓冲在另一张卡上。

    Args:
        w: W4A8 权重，逻辑形状 ``[N, K]``。
        x: 激活，``[K]``、``[M, K]`` 或它们的任意前导形状。
        activation_scale: 已有的逐行 scale；给了就复用它、不重新分配。

    Returns:
        与 ``x`` 同前导形状、最后一维为 ``N`` 的张量，dtype 同 ``x``。

    Raises:
        ValueError: 激活的最后一维不是 ``K``，或与权重不在同一设备。
    """
    module = _kernels()
    dtype = x.dtype
    if x.dim() not in (1, 2) or x.shape[-1] != w.k:
        raise ValueError("W4 projection expects [K] or [M, K] activations")
    if x.device != w.packed.device:
        raise ValueError("W4 weight and activation must share a device")
    act = _compute_dtype(x)
    # 只在真的需要时 reshape/contiguous。``x`` 在 decode 路径上几乎总是已经
    # ``[1, K]``（或 ``[K]``）且连续，而 ``.reshape(-1, K)`` 与 ``.contiguous()``
    # 即使什么都不做也各付一次 Python 分发 —— 一步 316 次 W4A8 投影就是 632 次。
    if act.dim() != 2 or act.shape[1] != w.k:
        act = act.reshape(-1, w.k)
    if not act.is_contiguous():
        act = act.contiguous()
    rows = act.shape[0]
    xq, xs = _w4_scratch(rows, w.k, act.device, activation_scale)
    out = torch.empty((rows, w.n), dtype=act.dtype, device=act.device)
    if rows:
        # 分支提到循环外：这几个条件只由形状决定，而它每次调用都要判一遍。
        #
        # 分界 64 是实测的（作业 009，形状 ``[12288, 2560]``、单卡、bf16 输出）：
        #
        #     M=1    张量核 484.4 us / GEMV   59.3 us   -> GEMV 快 8.2 倍
        #     M=32   张量核 510.6 us / GEMV  419.1 us   -> GEMV 快 18%
        #     M=512  张量核 1998.2 us / GEMV 6734.7 us  -> 张量核快 3.4 倍
        #
        # 32 这一档必须走 GEMV：张量核的 warp tile 在 M 方向恒为 64 行，M=32 时一半是
        # padding，而它还得付整套分块装载。从 64 起张量核领先。
        use_gemm = rows >= 64 and w.k % 4 == 0 and w.group_size % 64 == 0
        # M tile：从 256 行起用 256（实测 M=512 上比 128 快 11%，作业 259），
        # 64-255 行仍用 128 —— 这一档用 256 会把行数补到 256，白算一倍。
        m_tile = 256 if rows >= 256 else 0
        # **不套 ``torch.cuda.device(...)``**。那个上下文管理器每次都走一遍
        # ``c10::cuda::OptionalCUDAGuard``：读当前设备、比较、必要时切换、出块再
        # 恢复。而扩展自己的每个入口都已经有 ``CUDAGuard guard(t.device())``
        # （见 ``corex_ops.h`` 的 Constraints），张量也在正确的设备上，所以在
        # 单卡路径上它什么也没做——一步 316 次投影就是 316 次纯开销。
        # 多卡时由调用方保证当前设备，或由扩展的 guard 兜住。
        if activation_scale is None:
            module.quant_act_i8(act, xq, xs)
        else:
            module.quant_act_i8_scaled(act, xq, xs)
        if use_gemm:
            module.w4a8_gemm_tcu(xq, w.packed, xs, w.scale, out, w.group_size, m_tile)
        else:
            module.w4a8_gemv(xq, w.packed, xs, w.scale, out, w.group_size)
    return _restore(out[0] if x.dim() == 1 else out, dtype)


#: ``(rows, k, device)`` → ``(xq, xs)`` 的 W4A8 激活暂存池。
#:
#: 只在单线程的引擎主循环里被读写（``llm_engine`` 单线程驱动，见 ``moe.py`` 的
#: Constraints），所以不需要锁。条数上限是防御性的：形状随 prompt 长度变化，
#: 不设限会把每个见过的行数都留一份 int8 缓冲。
_W4_SCRATCH: dict = {}
_W4_SCRATCH_MAX = 32


def _w4_scratch(rows: int, k: int, device: torch.device,
                activation_scale: torch.Tensor | None):
    """取（必要时分配）一对 W4A8 激活暂存。

    Args:
        rows: 本次的行数。
        k: 激活宽度。
        device: 目标设备。
        activation_scale: 调用方已经持有的 scale；给了就直接用它，不碰缓存。

    Returns:
        ``(xq, xs)``：``[rows, k]`` 的 int8 与 ``[rows]`` 的 fp32。
    """
    if activation_scale is not None:
        xq = _W4_SCRATCH.get((rows, k, device, True))
        if xq is None:
            xq = torch.empty((rows, k), dtype=torch.int8, device=device)
            _W4_SCRATCH[(rows, k, device, True)] = xq
        return xq, activation_scale
    key = (rows, k, device, False)
    pair = _W4_SCRATCH.get(key)
    if pair is None:
        if len(_W4_SCRATCH) >= _W4_SCRATCH_MAX:
            _W4_SCRATCH.clear()
        pair = (torch.empty((rows, k), dtype=torch.int8, device=device),
                torch.empty(rows, dtype=torch.float32, device=device))
        _W4_SCRATCH[key] = pair
    return pair


def moe_grouped(w, x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """按专家分组做 W4A8 投影，每个专家的权重一步只读一遍。

    与 :func:`_w4_rows` 的区别只在并行维：那个是"一行激活对一个投影"，这个是
    "一个专家对其全部路由行"。decode 期每个专家平均只分到 1-2 行，按 token 分组
    会把 2.4 MB 的专家权重读到 token 数的倍数；按专家分组只读一遍。

    激活先整体量成 int8，再交给内核：量化是逐 token 的，与哪些 token 属于哪个专家
    无关，所以可以在分组之前一次做完，不必按段各量化一次。

    内核侧按**每专家行数**分两档，调用方不需要知道：每专家 1 行时走「8 路展开 +
    （K 够长才切）weight-group 分片」的那一档，实测 gate_up 203.3 -> 62.9 us、
    down 58.5 -> 35.9 us（作业 213，逐位相同）；每专家多行时走单趟形态 —— 分片在
    每专家 4 行以上反而慢 26%，因为部分和与归约是固定成本。判据与反例见
    ``kernels/moe_grouped.cu`` 里那段启动注释。

    Args:
        w: 专家栈，:class:`W4A8MoEWeight`，``[E, N, K]``。
        x: 激活，``[tokens, K]``，已按专家号排序。
        offsets: ``[E+1]`` 前缀和，int32 或 int64；空段表示该专家没有行。

    Returns:
        ``[tokens, N]``，dtype 同 ``x``。

    Raises:
        ValueError: 激活或前缀和的形状与权重不自洽。
    """
    module = _kernels()
    if x.dim() != 2 or x.shape[-1] != w.k:
        raise ValueError("grouped MoE expects [tokens, K] activations")
    if int(offsets.numel()) != len(w) + 1:
        raise ValueError(f"offsets must have {len(w) + 1} entries")
    tokens = x.shape[0]
    dtype = x.dtype
    if tokens == 0:
        return torch.empty((0, w.n), dtype=dtype, device=x.device)

    # 这一段是**主机侧**的主要成本（一条路径 48 层 x 2 次）。原来每次调用做四件事：
    # ``_compute_dtype`` / ``reshape`` / ``contiguous`` / 两个 ``torch.empty`` /
    # 一个 ``torch.zeros`` / 一个 ``offsets.to(int32)`` —— 六次分配加两次转换，
    # 而它们描述的东西在一步之内完全不变（形状只由 tokens 决定，``offsets`` 本来就是
    # int32 且在各 rank 间同构）。
    #
    # 所以这里按「形状键」把 xq/xs/out 复用起来，并短路掉那些已经是目标状态的转换。
    act = x if x.dtype in (torch.bfloat16, torch.float32) else x.float()
    if act.shape != (tokens, w.k) or not act.is_contiguous():
        act = act.reshape(tokens, w.k).contiguous()
    # 只复用 ``xq`` / ``xs`` 这对**纯暂存**：它们从不被返回、只在这次调用里被内核
    # 读写，所以复用无别名风险。``out`` 不复用 —— 它是返回值，调用方可能把它挂进
    # 残差流或是留着做后续加法，复用会让上一步的结果被就地覆盖。这一条与
    # ``_w4_rows`` 里"out 必须每次新建"是同一条理由。
    xq, xs = _moe_scratch(tokens, w.k, act.device)
    # out 必须清零：分组内核只写 ``offsets`` 覆盖到的行，尾部一个字节都不写，而
    # 调用方会把整块结果乘一个含 0 的权重掩码再加回去 —— ``0 * nan`` 还是 ``nan``。
    out = torch.zeros((tokens, w.n), dtype=act.dtype, device=act.device)
    # ``offsets`` 已经是 int32 连续时不要 ``.to()``：``.to`` 同 dtype 仍是一次
    # Python 层分发，而这里每层要付两次。
    if offsets.dtype == torch.int32 and offsets.is_contiguous():
        bounds = offsets
    else:
        bounds = offsets.to(dtype=torch.int32, device=act.device).contiguous()
    # 同样不套 ``torch.cuda.device(...)``：扩展入口自带 ``CUDAGuard``，而单卡路径
    # 上当前设备本来就是对的。一步 96 次就是 96 次纯开销。
    module.quant_act_i8(act, xq, xs)
    module.moe_grouped(xq, xs, w.packed, w.scale, bounds, out, w.group_size)
    return _restore(out, dtype)


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
    module = _kernels()
    module.shared_gate_scale(x, w, src, out)
    return out


def moe_grouped_i8(w, x: torch.Tensor, offsets: torch.Tensor,
                   slices: int = 0) -> torch.Tensor:
    """按专家分组做 int8 投影，一整栈权重在一次启动里读完。

    与 :func:`moe_grouped` 的区别是权重的量化格式：那个吃 packed W4 加逐 (weight-group,
    列) 的 scale，这个吃 ``[E, K, N]`` 的 int8 加逐 (专家, 列) 的 scale。后者让 K 方向
    分片可以先把整数和相加、最后一次乘两个 scale，所以部分和是 int32（见
    ``kernels/moe_grouped_i8.cu``）。

    激活按行量化一次再交给内核：量化与「哪些行属于哪个专家」无关，因此不必按段各量化
    一次。约定与预填充的 ``quant_act_i8`` 完全一致（absmax/127，下限 1e-5），两处的
    量化误差口径才对得上。

    Args:
        w: int8 专家栈，:class:`~int8_weights.QWeightMoE`（``[E, K, N]`` int8 加
            ``[E, N]`` fp32）。
        x: 激活，``[tokens, K]``，已按专家号排序。
        offsets: ``[E+1]`` 前缀和，int32 或 int64；空段表示该专家没有行。

    Returns:
        ``[tokens, N]``，dtype 同 ``x``。

    Raises:
        ValueError: 激活或前缀和的形状与权重不自洽。
    """
    module = _kernels()
    if x.dim() != 2 or x.shape[-1] != w.k:
        raise ValueError("grouped MoE expects [tokens, K] activations")
    if int(offsets.numel()) != len(w) + 1:
        raise ValueError(f"offsets must have {len(w) + 1} entries")
    tokens = x.shape[0]
    dtype = x.dtype
    if tokens == 0:
        return torch.empty((0, w.n), dtype=dtype, device=x.device)

    act = x if x.dtype in (torch.bfloat16, torch.float32) else x.float()
    if act.shape != (tokens, w.k) or not act.is_contiguous():
        act = act.reshape(tokens, w.k).contiguous()
    xq, xs = _moe_scratch(tokens, w.k, act.device)
    # 内核覆盖每一行的每一列（网格的 y 维就是行数），所以这里用 ``empty`` 而不是
    # ``zeros``：一行清零的发射在 decode 一步里要付 80 次。
    out = torch.empty((tokens, w.n), dtype=act.dtype, device=act.device)
    if offsets.dtype == torch.int32 and offsets.is_contiguous():
        bounds = offsets
    else:
        bounds = offsets.to(dtype=torch.int32, device=act.device).contiguous()
    module.quant_act_i8(act, xq, xs)
    module.moe_grouped_i8(xq, xs, w.wq, w.scale, bounds, out, slices)
    return _restore(out, dtype)




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
    module = _kernels()
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
    module.quant_act_i8(act, xq, xs)
    module.moe_grouped_i8_merge(xq, xs, w.wq, w.scale, bounds, row_weight, out,
                                slices)
    return _restore(out, dtype)
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
    module = _kernels()
    if expert_idx.dtype != torch.int32:
        expert_idx = expert_idx.to(torch.int32)
    if local_table.dtype != torch.int32:
        local_table = local_table.to(torch.int32)
    device = expert_idx.device
    offsets = torch.empty(local_experts + 1, dtype=torch.int32, device=device)
    slot_pos = torch.empty(slots, dtype=torch.int32, device=device)
    token_of_pos = torch.empty(slots, dtype=torch.int32, device=device)
    slot_of_pos = torch.empty(slots, dtype=torch.int32, device=device)
    scratch = _disp_scratch(local_experts, device)
    with torch.cuda.device(device):
        module.moe_dispatch(expert_idx, local_table, offsets, slot_pos,
                            token_of_pos, slot_of_pos, scratch, slots,
                            local_experts, num_experts, top_k)
    return offsets, slot_pos, token_of_pos, slot_of_pos


#: ``(local_experts, device)`` → 分派暂存。一步 48 层共用一块,不必每层分配。
_DISP_SCRATCH: dict = {}


def moe_decode_block(*args) -> torch.Tensor:
    """decode 单层 MoE 块的合成入口：一次调用顶掉十余次 pybind 往返。

    权重与全部中间缓冲都由调用方给，本入口不做分配——分配本身实测只有 1.4 us/次，
    把它搬进 C++ 省不下那笔钱，而这个入口赚的是参数编组与 Python 包装那约 4 us/次。

    Args:
        *args: 与扩展入口同序的位置参数，见 ``kernels/corex_ops.h`` 的声明。

    Returns:
        最终 MoE 输出张量（即调用方传入的 ``out``）。
    """
    return _kernels().moe_decode_block(*args)


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


def moe_grouped_gemm(w, x: torch.Tensor, offsets: torch.Tensor,
                     out: torch.Tensor | None = None) -> torch.Tensor:
    """按专家分段做**张量核 GEMM**，用于 prefill 的大 M。

    与 :func:`moe_grouped` 的区别只在用哪类内核：那个是「一列一线程、逐行串行」的
    GEMV 结构，适合 decode 时每专家 1-2 行；这里是每段一次 ``w4a8_gemm_tcu``，
    适合 prefill 时每专家上百行。

    **为什么需要它**：作业 ``q4prebase3`` 的 ``seq=2048`` 账本里
    ``moe_grouped_kernel`` 是 **2178 ms / 30.4%**，第一项。那时
    ``tokens = 2048 x top_k(10) = 20480`` 行散在 128 个本地专家上、每专家约 160 行，
    而 GEMV 结构下每行都要串行走完整个 K —— 权重被按段重读，有效带宽只有几 GB/s。
    同一批行按段喂给 ``w4a8_gemm_tcu`` 则是标准的 ``[160, 2560] x [2560, 1024]``，
    那是本卡张量核最擅长的形状（实测 2.3 ms/次、算力受限档）。

    Args:
        w: 专家栈，:class:`W4A8MoEWeight`，``[E, N, K]``。
        x: 激活，``[tokens, K]``，**已按专家号排好序**。
        offsets: ``[E+1]`` int32 前缀和，设备侧。
        out: 可选输出缓冲，``[tokens, N]``；不给则分配。

    Returns:
        ``[tokens, N]``，dtype 同 ``x``。

    Raises:
        ValueError: 形状不自洽，或 K/group 不满足张量核的对齐要求（此时调用方应当
            继续用 :func:`moe_grouped`）。
    """
    module = _kernels()
    tokens = x.shape[0]
    dtype = x.dtype
    if tokens == 0:
        return torch.zeros((0, w.n), dtype=dtype, device=x.device)
    act = _compute_dtype(x).reshape(tokens, w.k).contiguous()
    xq = torch.zeros(act.shape, dtype=torch.int8, device=act.device)
    xs = torch.zeros(tokens, dtype=torch.float32, device=act.device)
    result = (torch.zeros((tokens, w.n), dtype=act.dtype, device=act.device)
              if out is None else out)
    if out is not None:
        out.zero_()
    bounds = offsets.to(dtype=torch.int32, device=act.device).contiguous()
    module.quant_act_i8(act, xq, xs)
    module.moe_grouped_gemm(xq, xs, w.packed, w.scale, bounds, result,
                            w.group_size)
    return _restore(result, dtype)


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
    module = _kernels()
    if not logits.is_contiguous():
        logits = logits.contiguous()
    index = torch.empty((tokens, top_k), dtype=torch.int64, device=logits.device)
    weight = torch.empty((tokens, top_k), dtype=logits.dtype, device=logits.device)
    if tokens == 0:
        return index, weight
    # 不套 ``torch.cuda.device(...)``：扩展入口自带 ``CUDAGuard``，而 decode 一步要发
    # 48 次，每次都是一个纯开销的上下文管理器。
    module.moe_router(logits, index, weight, top_k, normalize)
    return index, weight


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
    module = _kernels()
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
    module.attn_sparse(q, kcache, vcache, sel_idx, sel_len, ctx, out, float(scale))
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
    module = _kernels()
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
    module.qsa_block_keys(flat_keys, table, weight, flat_cos, flat_sin,
                          block_row, block_batch, out, int(seq), int(tokens),
                          int(rotary_dim), float(eps))
    return out


def _w8a8_rows(packed: pack.KnQWeight, act: torch.Tensor) -> torch.Tensor:
    """用三个 W8A8 内核做一次预填充投影：量化激活、int8 GEMM、乘两个 scale。

    这是本后端 int8 权重的预填充实现。它替掉的是「每次调用把整份权重反量化成
    bf16 再 GEMM」那条路，那条路在 4B 上每个请求固定多付一次全量权重转换
    （``pack.KnQWeight.dequant_kn`` 的 docstring 记的是 174.8 ms 一档的代价），
    而这里权重全程留在 int8。

    ``N`` 超过 :data:`pack.LIB_MAX_N` 时沿列切块，每块的累加器单独分配：``[K, N]``
    矩阵的列切片不连续，而 ``w8a8_gemm`` 要求最内层连续，正确写法是切一次权重再
    GEMM，而不是把不连续的视图递进去。

    Args:
        packed: int8 权重，``[K, N]`` 矩阵加 ``[N]`` 个 fp32 列 scale。
        act: 激活，``[M, K]``，bf16 或 fp32。

    Returns:
        ``[M, N]``，dtype 同 ``act``。

    Raises:
        RuntimeError: 扩展不可用。这条路径没有纯 torch 的回退——调用方
            :func:`linear_prefill` 只在 int8 权重且 ``M`` 够大时才走到这里，
            而那种形状上退回参考实现比慢更糟。
    """
    module = _kernels()
    rows, k = act.shape
    n = packed.n
    xq = torch.empty((rows, k), dtype=torch.int8, device=act.device)
    xs = torch.empty(rows, dtype=torch.float32, device=act.device)
    out = torch.empty((rows, n), dtype=act.dtype, device=act.device)
    # 量化排在循环外：一份激活只按行量化一次，切块只在 N 上做，所以每块共用同一
    # 份 int8 激活和同一份 scale。
    module.quant_act_i8(act, xq, xs)
    use_tcu = _use_tcu_gemm(rows, k, n, hasattr(module, "w8a8_gemm_tcu"))
    for start in range(0, n, pack.LIB_MAX_N):
        stop = min(n, start + pack.LIB_MAX_N)
        width = stop - start
        block = packed.wt[:, start:stop]
        if not block.is_contiguous():
            block = block.contiguous()
        if use_tcu:
            # 融合路径没有 int32 缓冲，也没有第二个内核：累加器留在寄存器里，
            # 乘完两个 scale 直接出浮点。
            module.w8a8_gemm_tcu(xq, block, xs, packed.scale[start:stop],
                                 out[:, start:stop])
            continue
        acc = torch.empty((rows, width), dtype=torch.int32, device=act.device)
        module.w8a8_gemm(xq, block, acc)
        # 行距传 ``n`` 而不是块宽：``out`` 的列切片在最内层仍然连续到 ``n``，
        # 用块宽当行距会让第二行写回到第一行的尾部。
        module.w8a8_epilogue(acc, xs, packed.scale[start:stop],
                             out[:, start:stop], n)
    return out


def linear_prefill(w: torch.Tensor | QWeight | pack.KnWeight,
                   x: torch.Tensor) -> torch.Tensor:
    """预填充分块投影（大 ``M``），此时 GEMM 胜过流式读权重。

    库只用在这张卡已知能算对的宽度内：输出列超过 :data:`pack.LIB_MAX_N` 时它的
    GEMM 在 ``N = 248320``（就是 ``lm_head``）返回 NaN，而那正是最经不起这种错误
    的一个形状。切块判定在 :func:`_w8a8_rows` 里，内核里还有第二道断言。

    ``M`` 低于 :data:`_PREFILL_MIN_M` 时仍然走自写 GEMV，int8 权重也一样：那条
    分界是实测的（``27B-4CARD-PLAN.md`` §3.4，M=1 上 W8A8 慢 42%），不是可调
    参数。这里额外把分界判断提到 int8 反量化之前，所以小 ``M`` 的 int8 预填充
    不再先付一次全量反量化再把它丢掉。

    Args:
        w: 任意可接受存储形式的 ``[N, K]`` 投影权重。
        x: 激活，``[C, K]``。

    Returns:
        ``[C, N]``，dtype 同 ``x``。

    Raises:
        RuntimeError: 扩展不可用，而宽 ``N`` 形状需要它。
    """
    if isinstance(w, W4A8Weight):
        rows_w4 = int(x.shape[0])
        # 大 M：反量化成 bf16 走厂商 GEMM。**必须放在读 ``packed.wt`` 之前** ——
        # ``W4A8Weight`` 没有 ``.wt`` 字段，激活 dtype 只能由这里自己定（bf16，与
        # ``w4_dequant`` 的输出对齐）。
        #
        # 为什么 M=512 才值得（作业 371，q_proj 512x12288x2560）：
        #   反量化 198.4 us（79.6 MB @ 401.3 GB/s）+ 厂商 bf16 GEMM 728.2 us
        #   = 935.9 us，对 ``w4_gemm_tcu`` 的 1934.8 us 是 2.07 倍，且逐位可复原。
        # 小 M 不走：反量化那 79 MB 是固定成本，摊不薄。
        # `x.dim() != 2` 时交给 `_w4_rows`：它自己会 reshape，而这条分支只做二维。
        if (x.dim() == 2 and rows_w4 >= _W4_DEQUANT_MIN_M and (w.n & 3) == 0
                and w.k * w.n * 2 <= _W4_DEQUANT_MAX_BYTES):
            act = _compute_dtype(x).reshape(rows_w4, w.k)
            lhs = act if act.dtype == torch.bfloat16 else act.to(torch.bfloat16)
            if not lhs.is_contiguous():
                lhs = lhs.contiguous()
            return _restore(torch.matmul(lhs, _dequant_w4(w)), x.dtype)
        return _w4_rows(w, x)
    packed = pack.as_kn(w)
    rows = x.shape[0]
    if rows < _PREFILL_MIN_M:
        # 低于预填充区间时 GEMM 每个 tile 都要读一遍权重，输给整次调用只读一遍的
        # GEMV。分界必须走在打包之后、任何按 dtype 的分支之前。
        return _gemv_rows(w, x)
    dtype = x.dtype
    act = _compute_dtype(x)
    if packed.int8:
        # int8 权重不再反量化。反量化那条路留给没有 W8A8 内核的后端；这里权重全程
        # 保持 int8，省下的不只是时间还有一整份 bf16 副本的显存。
        return _restore(_w8a8_rows(packed, act), dtype)
    weight_dtype = packed.wt.dtype
    # GEMM 两个操作数必须同一 dtype，而本卡 bf16 和 fp16 的 TFLOP/s 一样（M=2048 时
    # 32.6），所以由权重自己的 dtype 决定，激活跟着它，而不是反过来。
    lhs = act.to(weight_dtype)
    if packed.n < 8:
        # **N < 8 一律补零到 8 列，取回前 ``n`` 列**：GEMM 的 N 方向 tile 是 8 的倍数，
        # N 小于 8 时整 tile 里大部分通道在空转。实测同一个 ``[512, 2560] x [2560, 1]``：
        # 直接乘 **853 us**（3.07 GFLOP/s），补零到 8 列再切回是 **63.6 us（13.4x）**，
        # 且结果**逐位相同**（补的是 0，点积里多出来的项精确为 0）。补到 16/32 无进一步
        # 收益（63.5 / 63.0 us），所以取 8。
        #
        # 覆盖到的真实形状（``COFFIN_MOE_SHAPE=1`` 与 ``record_shapes`` 都量过）：
        #   * ``N = 1``：每层一次的标量投影（48 次/遍，曾 1.03 ms/次 = 49.3 ms）；
        #   * ``N = 4``：``gated_residual`` 的 ``block_inject_weight``（``hc_count=4``，
        #     96 次/遍，``[512, 10240] x [10240, 4]`` 曾 0.36 ms/次 = 34.5 ms）。
        # 两者都是「无效 FLOPs 极便宜、tile 却大部分空转」的典型。
        #
        # 走过的弯路，别重试：``torch.mv`` 会让算子链变成 ``aten::mv -> aten::addmv_``，
        # **仍是 1.04 ms/次**（端到端 1402.99 -> 1408.86，噪声内）。瓶颈不在 API 选择，
        # 在**形状**——不换 API、只补零就能快 13 倍。归因见作业 315 / 380。
        pad = 8
        width = int(packed.n)
        wp = torch.zeros((packed.k, pad), dtype=weight_dtype, device=lhs.device)
        wp[:, :width] = packed.wt
        return _restore(torch.matmul(lhs, wp)[:, :width], dtype)
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
    module = _kernels()
    pa, pm = _split_k(part_acc, part_ml, (q.shape[0],), q)
    return module.attn_paged(q.contiguous(), gate.contiguous(), kpool, vpool, kscale,
                             vscale, pos2slot, out, pos, max_pos, rep, scale, pa, pm, head_major)


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
                                   rep, scale, pa, pm, head_major)


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
                                     rep, scale, pa, pm, mask, mask_block, head_major)


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
    return _kernels().delta_record(q, k, v, beta, g, st, out, record, slot)


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
    return _kernels().delta_batch_record(q, k, v, beta, g, st, out, record, slot)


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
    return _kernels().delta_commit(base, record, st, steps)


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
    module = _kernels()
    args = (q, k, v, b, a, neg_expA, dt_bias, z, nw, st, out, qscale, rep)
    if record is None:
        return module.delta_normed(*args)
    return module.delta_normed_record(*args, record, slot)


# 超连接（Gated Residual）
#
# 这三个适配器背后是 ``kernels/hyper.cu`` 的三个内核，它们一层里被调两套、48 层就是
# 96 套，每套从八次 torch 启动压到三次。三者的权重全部按内核的 ``[K, N]`` 契约收，
# 转置由 ``pack.as_kn`` 按对象身份记忆化承担一次——与 ``linear`` 同一条路，所以
# 模型侧不需要知道本后端的布局。


def _hyper_rows(x: torch.Tensor) -> torch.Tensor:
    """把 ``[..., width]`` 摊平成内核要求的 ``[rows, width]`` 连续张量。

    不用 :func:`_pitch_rows`：那个按行间距做 ``as_strided`` 视图，而
    ``hyper.cu`` 的三个内核都用 ``ptr + row * width`` 直接寻址、并在入口
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


#: 超连接权重从 W4A8 实体化成 bf16 的缓存，键是 ``(id(W4A8Weight), 激活 dtype)``。
#:
#: 为什么键里带 dtype：内核对权重与激活的 dtype 一致性有硬断言，而同一个
#: ``W4A8Weight`` 在不同臂上可能配不同的激活 dtype；只按 ``id()`` 记住一份会让第二次
#: 拿到 dtype 不符的矩阵，然后被内核拒绝。
#:
#: 为什么必须实体化：``hyper.cu`` 的三个内核只吃 bf16/fp32 的致密矩阵，没有反量化
#: 通路（那是 ``w4a8_gemv`` 的事，而它的输出形状是"激活 × 列"，与这里的
#: ``[lowrank, K]`` 权重转置对不上）。而真机量化产物确实覆盖了超连接的投影：它们是
#: 二维、行数 10240/320，全部过了 ``classify`` 的 q4 判据。
#:
#: 代价是每套超连接多 58.7 MB 常驻：``input_mix_weight_down`` 的 ``[10240, 2560]``
#: 是 52.4 MB、``input_mix_weight_up`` 的 ``[10240, 320]`` 是 6.55 MB、写门的
#: ``[4, 10240]`` 是 81.9 KB（bf16 实体化）。48 层 × 2 套 = 96 套合计 **约 5.6 GB**。
#: 这是拿显存换 768 次发射（768 × 5.75 us 的启动地板 = 4.4 ms）的取舍，不是免费的，
#: 也不是笔误。真机上必须把 ``resident=`` 打出来并与基线比：那个差值是这笔账的实付金额。
#:
#: 用普通 dict 而不是 weakref 表：``W4A8Weight`` 带 ``__slots__`` 且没有
#: ``__weakref__`` 槽位，弱引用建不出来；而这个字典的条目数是模型的固定几何
#: （每层两套 × 三个权重），不是一个随调用增长的缓存，同时持有强引用正好挡住
#: ``id()`` 被回收后复用。
_HYPER_DENSE: dict[tuple[int, torch.dtype], "pack.KnWeight"] = {}


def _hyper_weight(w, act_dtype: torch.dtype) -> "pack.KnWeight":
    """把超连接的权重备成内核能读的 ``[K, N]`` 致密矩阵，dtype 强制对齐激活。

    ``hyper.cu`` 的入口断言是 ``wt.scalar_type() == normed.scalar_type()``，所以权重
    dtype 必须显式跟着激活走，不能指望"加载器已经把它转成 bf16 了"：

    * ``W4A8Weight`` 没有转置形态，只能先实体化成 ``act_dtype`` 再转置（这是当前唯一
      能跑的路，代价见 :data:`_HYPER_DENSE`）。
    * ``pack.as_kn`` 对**已经打包**的权重是幂等直返的，不会改 dtype。而 ``KnWeight``
      不是 ``nn.Module`` 也不是 buffer，``model.to(dtype=...)`` 搬不动它——于是真机上
      会出现"激活 bf16、权重 fp32"，内核直接拒绝（作业 4543，on 臂第一层就死在这里）。
      这一条与 ``gdn.py`` 记录的 fp32 输出外溢是同一类事故：**dtype 契约必须由适配器
      自己收口，不能靠上游自觉**。

    Args:
        w: 契约布局 ``[N, K]`` 的权重：``nn.Parameter``、``pack.KnWeight`` 或
            :class:`W4A8Weight`。
        act_dtype: 归一化激活的 dtype，内核要求权重与它一致。

    Returns:
        ``[K, N]`` 连续致密矩阵的打包形式，dtype 为 ``act_dtype``。

    Raises:
        ValueError: 既不是可识别的权重类型，也不是二维张量。
    """
    if isinstance(w, W4A8Weight):
        key = (id(w), act_dtype)
        hit = _HYPER_DENSE.get(key)
        if hit is not None:
            return hit
        dense = pack.to_kn(w.to(act_dtype), act_dtype)
        _HYPER_DENSE[key] = dense
        return dense
    packed = pack.as_kn(w, act_dtype)
    if packed.wt.dtype == act_dtype:
        return packed
    # 已打包但 dtype 不符（``as_kn`` 幂等直返那条路）。这一步只在首次见到该权重时
    # 发生一次，之后由 ``as_kn`` 的对象身份缓存接管。
    return pack.KnWeight(packed.wt.to(act_dtype).contiguous())


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
    module = _kernels()
    act = _compute_dtype(x)
    gain = w if w.dtype == torch.float32 else w.to(torch.float32)
    return module.hyper_norm(_hyper_rows(act), gain.contiguous(),
                             _hyper_rows(out), hidden, hc, eps)


def hyper_down(normed: torch.Tensor, wt: torch.Tensor, out: torch.Tensor,
               hc: int) -> torch.Tensor:
    """经融合内核做读门的降维投影，并 ``silu(v / hc)``。

    除以 ``hc`` 发生在 silu **之前**，与内核一致：silu 不是线性的，顺序换了就是
    另一条曲线。

    Args:
        normed: 归一化后的激活，``[..., K]`` 连续。
        wt: 降维权重，契约布局 ``[lowrank, K]``；内核要 ``[K, lowrank]``，转置由
            :func:`pack.as_kn` 按对象身份记忆化承担，与 :func:`linear` 同一条路。
        out: 目的地，``[..., lowrank]``，dtype 同 ``normed``，原地写入。
        hc: 读门除数。

    Returns:
        ``out``，便于调用链串接。

    Raises:
        RuntimeError: 扩展不可用。
        ValueError: 某个操作数不是连续布局。
    """
    module = _kernels()
    act = _compute_dtype(normed)
    packed = _hyper_weight(wt, act.dtype)
    return module.hyper_down(_hyper_rows(act), packed.wt, _hyper_rows(out), hc)


def hyper_up(normed: torch.Tensor, up_wt: torch.Tensor, low: torch.Tensor,
             inj_wt: torch.Tensor, mixed: torch.Tensor, inject: torch.Tensor,
             hidden: int, hc: int, lowrank: int) -> torch.Tensor:
    """经融合内核做读门升维、加权 mean 与写门，一次启动。

    读门塌缩是 ``mean`` 不是 ``sum``，写门是 ``2 * sigmoid(dot / hc)``；三处都在
    内核里，这里的职责只是把两个权重备成 ``[K, N]``。

    Args:
        normed: 归一化后的激活，``[..., hidden * hc]`` 连续。
        up_wt: 升维权重，契约布局 ``[hidden * hc, lowrank]``。
        low: 降维输出，``[..., lowrank]`` 连续。
        inj_wt: 写门权重，契约布局 ``[hc, hidden * hc]``。
        mixed: 读门塌缩结果的目的地，``[..., hidden]``，原地写入。
        inject: 写门的目的地，``[..., hc]`` fp32，原地写入。
        hidden: 每路宽度。
        hc: 路数。
        lowrank: 降维秩。

    Returns:
        ``mixed``，便于调用链串接。

    Raises:
        RuntimeError: 扩展不可用。
        ValueError: 某个操作数不是连续布局。
    """
    module = _kernels()
    act = _compute_dtype(normed)
    packed_up = _hyper_weight(up_wt, act.dtype)
    packed_inj = _hyper_weight(inj_wt, act.dtype)
    return module.hyper_up(_hyper_rows(act), packed_up.wt, _hyper_rows(low),
                           packed_inj.wt, _hyper_rows(mixed), _hyper_rows(inject),
                           hidden, hc, lowrank)


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
    "linear_m_verify": linear_m_verify,
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
    "delta_record": delta_record,
    "delta_batch": delta_batch,
    "delta_batch_record": delta_batch_record,
    "delta_commit": delta_commit,
    "delta_normed": delta_normed,
    "triangular_solve": triangular_solve,
    "moe_grouped": moe_grouped,
    "moe_grouped_gemm": moe_grouped_gemm,
    "moe_grouped_i8": moe_grouped_i8,    "moe_grouped_i8_merge": moe_grouped_i8_merge,
    "moe_decode_block": moe_decode_block,
    "shared_gate_scale": shared_gate_scale,
    "attn_sparse": attn_sparse,
    "qsa_block_keys": qsa_block_keys,
    "moe_dispatch": moe_dispatch,
    # 路由的一条融合路径：替掉 softmax/topk/div/sum/to 五个内核，其中 topk 走的是
    # 512 元素的 bitonic 排序网络。没有它这一层会退回参考实现（慢但对）。
    "moe_router": moe_router,
    "hyper_norm": hyper_norm,
    "hyper_down": hyper_down,
    "hyper_up": hyper_up,
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
    "hyper_norm", "hyper_down", "hyper_up",
})


@functools.lru_cache(maxsize=1)
def caps() -> Caps:
    """返回本后端在这台主机上的能力记录。

    Returns:
        一个记录：``device="cuda"``（CoreX 实现了那个命名空间），两种 dtype 都是
        bf16，因为预填充 GEMM 在本卡上 bf16 和 fp16 都到 32.6 TFLOP/s，转换不值
        它的流量；``weight_layout="kn"``，因为那个布局在本卡上对解码值 14-22 倍；
        ``attn_split=8``：切片轴就是内核的 ``grid.y``，切片数由 ``part_acc.size(-2)`` 读，
        所以改这里就够、不用改内核。实测（合成 int8 池）4k 1.407→0.827 ms、32k 11.17→3.55、
        64k 24.49→6.22，对 split=1 的相对误差 ≤2.8e-3。取 8 不取 16：两者在 4k 与 64k 都
        处在同一平台（0.78 / 6.2 ms），而 8 的切片缓冲只有一半。原先这里写「内核还不会按 K
        切分」，是把「能力记录里没开」误当成「内核没实现」——那次没有实测就下了结论。

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
        attn_split=8,
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
