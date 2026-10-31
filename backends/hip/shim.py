"""编译并暴露 ``backends/hip/kernels/`` 里的 HIP 内核。

Module: ``coffin/backends/hip``，与 ``backends/hip/__init__.py`` 一起位于依赖图底部，
后者是唯一的 import 方。内核是 C++/HIP 源码，一次性编译进 ``TORCH_EXTENSIONS_DIR``
下的缓存扩展；本模块只是编译/加载的 shim，所以必须不依赖模型、算子表和协议。

Constraints:
    * 在 ``at::cuda::getCurrentCUDAStream()`` 上启动（旧的默认流不属于
      ``torch.cuda.graph`` 捕获范围，在那里启动的内核会从图里消失，解码步就会静默读到
      过期输出）。
    * 所有形状保持静态：解码步只捕获一次然后对每个位置重放，所以随 token 索引变化的
      形状无法被捕获。
    * 内核写入的缓冲区（split 部分和、KV 状态、卷积状态）归调用方所有，出于同样的理由
      必须保持地址固定。
    * 扩展只针对 gfx906（``PYTORCH_ROCM_ARCH`` / ``--offload-arch``）。换一颗芯片就该
      有自己的后端，而不是在这里多写一个 arch 字符串。
    * ``TORCH_EXTENSIONS_DIR`` 必须按部署隔离。共享缓存目录会让每个新 checkout 都触发
      一次完整 HIP 重编，并争抢同一把构建锁；并发的部署随后会一直等下去。
    * 两处 ``os.environ`` 写入留在 ``torch`` 的 import 之上，因为扩展加载器在 import 时
      就会读它们。它们是进程级副作用：把它们关在一个后端里（而不是放在包根目录），
      才不至于让 ``PYTORCH_ROCM_ARCH=gfx906`` 泄漏进一个选了别的后端的进程。
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx906")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/root/private_data/torch_ext")

import torch
from torch.utils.cpp_extension import load

_KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
_SOURCES = ("bindings.cpp", "gemv.cu", "gemv_int8.cu", "delta.cu", "attn.cu", "norm.cu",
            "linear_attn.cu", "batch.cu", "gemv_wide.cu")

_mod = None
_error: str | None = None


def _load():
    """编译一次内核扩展（或挂到它的缓存构建上）。

    结果记忆化在 ``_mod``，失败记忆化在 ``_error``，于是坏掉的工具链每进程只花一次
    编译尝试而不是每次调用一次——引擎每个请求要进这个模块成千上万次。

    Returns:
        扩展模块；编译失败时为 ``None``。调用方必须把 ``None`` 当成「走即时执行路径」，
        绝不能当成空模块。失败异常本身留在 ``_error`` 里供 :func:`last_error` 使用，
        因为缺少工具链时必须退化成 hipBLAS 兜底，而不是在 import 阶段把进程带崩。
    """
    global _mod, _error
    if _mod is None and _error is None:
        try:
            _mod = load(
                name="hip_ops_dcu",
                sources=[str(_KERNEL_DIR / name) for name in _SOURCES],
                extra_include_paths=[str(_KERNEL_DIR)],
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3", "--offload-arch=gfx906"],
                verbose=False,
            )
        except Exception as exc:  # pragma: no cover - depends on the toolchain
            _error = f"{type(exc).__name__}: {exc}"
    return _mod


def available() -> bool:
    """扩展是否已编译（或本来就在缓存里）。

    Returns:
        工具链缺失或编译失败时为 ``False``。调用方据此挑选即时执行的兜底路径；因为加载
        做了记忆化，这个值在进程整个生命周期内不变。
    """
    return _load() is not None


def last_error() -> str | None:
    """报告扩展为什么不可用，供启动日志使用。

    Returns:
        编译失败时的 ``"<ExceptionType>: <message>"``；编译成功或从未尝试时为 ``None``。
        引擎在启动时打一次这条信息；没有它，一次悄悄回退到 hipBLAS 看起来就像模型慢，
        而不是部署坏了。
    """
    return _error


def gemv(w: torch.Tensor, x: torch.Tensor, R: int = 1, U: int = 2) -> torch.Tensor:
    """``w`` [N, K] fp16 或 bf16 连续存放，``x`` [K] bf16，返回 ``y`` [N] bf16。

    实践中权重是 fp16（见 ``models/qwen35/weights.py``）；保留 bf16 入口是为了手里拿着
    bf16 张量的调用方。

    Args:
        w: 权重矩阵 ``[N, K]``，行主序，fp16 或 bf16，连续存放。
        x: 激活向量 ``[K]``，bf16。
        R: 每个线程块计算多少行输出；调优旋钮来自 ``models/qwen35/tuning.py``。
        U: K 方向展开因子（2 或 4）。取大能减半循环开销，代价是 K 维必须保持它的
            整数倍。

    Returns:
        ``[N]`` bf16 输出。

    Raises:
        RuntimeError: 扩展不可用。回退到 ``ops/gemv`` 的即时执行路径；两边数值必须
            一致。
    """
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    if w.dtype == torch.float16:
        return module.gemv_f16(w, x, R, U)
    return module.gemv_bf16(w, x, R, U)


def gemv_m(w: torch.Tensor, x: torch.Tensor, U: int = 2) -> torch.Tensor:
    """``w`` [N, K] fp16/bf16，``x`` [M, K] bf16（M 为 2/4/8），返回 ``y`` [M, N]。

    Args:
        w: 权重矩阵 ``[N, K]``，行主序，连续存放。
        x: 激活矩阵 ``[M, K]`` bf16。``M`` 必须是 2、4 或 8：内核就是按这几个宽度编译
            的，别的 M 会静默读错行数。
        U: K 方向展开因子（2 或 4）。

    Returns:
        ``[M, N]`` bf16 输出，全部 M 行共用一次权重读取。

    Raises:
        RuntimeError: 扩展不可用。
    """
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    if w.dtype == torch.float16:
        return module.gemv_f16_m(w, x, U)
    return module.gemv_bf16_m(w, x, U)


def gemv_w8a16(w: torch.Tensor, scale: torch.Tensor, x: torch.Tensor,
               R: int = 1, U: int = 2) -> torch.Tensor:
    """``w`` [N, K] int8 配每行 ``scale`` [N]，``x`` [K] bf16 -> [N] bf16。

    Args:
        w: int8 权重矩阵 ``[N, K]``，行主序，在内核里反量化。
        scale: 每行的 fp32 反量化 scale，``[N]``。
        x: 激活向量 ``[K]`` bf16。
        R: 每个线程块的输出行数。
        U: K 方向展开因子（2 或 4）。

    Returns:
        ``[N]`` bf16 输出。改成把权重实体化成 bf16 会多占 ``[N, K]`` 的常驻字节，而
        int8 布局存在的意义正是省掉这些。

    Raises:
        RuntimeError: 扩展不可用。
    """
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    return module.gemv_w8a16(w, scale, x, R, U)


def gemv_w8a16_m(w: torch.Tensor, scale: torch.Tensor, x: torch.Tensor,
                 U: int = 2) -> torch.Tensor:
    """``w`` [N, K] int8，``x`` [M, K] bf16（M 为 2/4/8）-> [M, N] bf16。

    Args:
        w: int8 权重矩阵 ``[N, K]``。
        scale: 每行的 fp32 scale，``[N]``。
        x: 激活矩阵 ``[M, K]`` bf16，``M`` 为 2/4/8。
        U: K 方向展开因子（2 或 4）。

    Returns:
        ``[M, N]`` bf16 输出。

    Raises:
        RuntimeError: 扩展不可用。
    """
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    return module.gemv_w8a16_m(w, scale, x, U)


def delta(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor,
          g: torch.Tensor, st: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """gated delta 规则的一步解码；原地更新 ``st`` 并返回 ``out``。

    Args:
        q: 每个 value 头归一化后的查询，``[num_v, dk]`` fp32。
        k: 每个 value 头归一化后的 key，``[num_v, dk]`` fp32。
        v: 每个 value 头的 value，``[num_v, dv]`` fp32。
        beta: 每头的写入强度，取值在 ``[0, 1]``，``[num_v]`` fp32。
        g: 每头的对数衰减，``<= 0``，``[num_v]`` fp32。
        st: 循环状态 ``[num_v, dk, dv]`` fp32，原地更新。它是本层唯一的跨 token
            记忆，需要回滚的调用方必须先给它打快照。
        out: 输出缓冲区 ``[num_v, dv]`` fp32，原地写。

    Returns:
        ``out`` 本身，便于链式调用。

    Raises:
        RuntimeError: 扩展不可用。
    """
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    return module.delta_step(q, k, v, beta, g, st, out)


def delta_prologue(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   b: torch.Tensor, a: torch.Tensor,
                   neg_expA: torch.Tensor, dt_bias: torch.Tensor,
                   q_out: torch.Tensor, k_out: torch.Tensor, v_out: torch.Tensor,
                   beta: torch.Tensor, g: torch.Tensor,
                   qscale: float, rep: int) -> None:
    """对 q/k 做 L2 归一化、把 key 头扩成 value 头、把 beta/g 建进 fp32 缓冲区。

    一个内核替代约 16 个小的逐元素算子（mul、sum、rsqrt、sigmoid、softplus、add、
    repeat_interleave），解码时它们占了混合器启动次数的大部分。

    Args:
        q: 每个 key 头的查询，``[num_k, dk]``。
        k: 每个 key 头的 key，``[num_k, dk]``。
        v: 每个 value 头的 value，``[num_v, dv]``。
        b: beta 的 logits，``[num_v]``；sigmoid 在这里施加。
        a: 衰减的 logits，``[num_v]``；softplus 加 ``dt_bias`` 在这里施加。
        neg_expA: ``-exp(A_log)``，``[num_v]`` fp32。
        dt_bias: 衰减偏置，``[num_v]`` fp32。
        q_out: 输出查询，``[num_v, dk]`` fp32，已完成头扩展和缩放。
        k_out: 输出 key，``[num_v, dk]`` fp32。
        v_out: 输出 value，``[num_v, dv]`` fp32。
        beta: 输出的写入强度，``[num_v]`` fp32。
        g: 输出的对数衰减 ``<= 0``，``[num_v]`` fp32。
        qscale: ``dk ** -0.5``，只施加到查询上。
        rep: 每个 key 头对应的 value 头数，即扩多少倍。

    Returns:
        无。每个输出缓冲区都是原地写入。

    Raises:
        RuntimeError: 扩展不可用。
    """
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    module.delta_prologue(q, k, v, b, a, neg_expA, dt_bias,
                          q_out, k_out, v_out, beta, g, qscale, rep)


def conv1d_silu(conv_w: torch.Tensor, x: torch.Tensor, conv_state: torch.Tensor,
                out: torch.Tensor) -> torch.Tensor:
    """融合的深度因果 conv1d（卷积核 4）+ SiLU；平移 ``conv_state``。

    Args:
        conv_w: 深度卷积的抽头系数 ``[conv_dim, 4]`` fp32。
        x: 当前 token 的投影，``[conv_dim]``。
        conv_state: 前 3 个 token，``[conv_dim, 3]`` fp32。原地平移，所以它是混合器的
            短期记忆，必须与循环状态一起打快照。
        out: 输出缓冲区 ``[conv_dim]`` fp32，原地写。

    Returns:
        ``out`` 本身，便于链式调用。

    Raises:
        RuntimeError: 扩展不可用。``linear_attn.__call__`` 里的即时执行路径改用
            cat + copy_ + mul + sum + silu。
    """
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    return module.conv1d_silu(conv_w, x, conv_state, out)


def conv1d_silu_seq(conv_w: torch.Tensor, x: torch.Tensor,
                    conv_state: torch.Tensor) -> torch.Tensor:
    """同样的卷积跑在预填充分块上：``x`` [T, C] bf16 -> [T, C] fp32。

    Args:
        conv_w: 深度卷积的抽头系数 ``[C, 4]`` fp32。
        x: 分块的投影 ``[T, C]``，每行一个 token。
        conv_state: 跨块携带的卷积窗口 ``[C, 3]`` fp32，原地平移，使分块预填充与一次性
            预填充结果完全一致。

    Returns:
        ``[T, C]`` fp32 的卷积 + SiLU 输出。用 fp32 是因为下游的 delta 扫描要跨块累加。

    Raises:
        RuntimeError: 扩展不可用。
    """
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    return module.conv1d_silu_seq(conv_w, x, conv_state)


def delta_prep_seq(qkv: torch.Tensor, b: torch.Tensor, a: torch.Tensor,
                   neg_expA: torch.Tensor, dt_bias: torch.Tensor,
                   q_out: torch.Tensor, k_out: torch.Tensor, v_out: torch.Tensor,
                   beta: torch.Tensor, g: torch.Tensor,
                   qscale: float, rep: int) -> None:
    """融合的预填充头准备，把 q/k/v 写成 [H, T, d]、beta/g 写成 [H, T]。

    头主序布局正是这次融合的意义：分块 delta 扫描直接读 ``[H, T, d]``，在这里产出它就
    省掉了整个分块的一次转置拷贝。

    Args:
        qkv: 卷积输出 ``[T, conv_dim]`` fp32，即融合的 q/k/v 投影。
        b: beta 的 logits，``[T, num_v]``。
        a: 衰减的 logits，``[T, num_v]``。
        neg_expA: ``-exp(A_log)``，``[num_v]`` fp32。
        dt_bias: 衰减偏置，``[num_v]`` fp32。
        q_out: 输出查询，``[num_v, T, dk]`` fp32。
        k_out: 输出 key，``[num_v, T, dk]`` fp32。
        v_out: 输出 value，``[num_v, T, dv]`` fp32。
        beta: 输出的写入强度，``[num_v, T]`` fp32。
        g: 输出的对数衰减 ``<= 0``，``[num_v, T]`` fp32。
        qscale: ``dk ** -0.5``，只施加到查询上。
        rep: 每个 key 头对应的 value 头数。

    Returns:
        无。每个输出缓冲区都是原地写入。

    Raises:
        RuntimeError: 扩展不可用。
    """
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    module.delta_prep_seq(qkv, b, a, neg_expA, dt_bias, q_out, k_out, v_out,
                          beta, g, qscale, rep)


def attn(q: torch.Tensor, gate: torch.Tensor, kcache: torch.Tensor, vcache: torch.Tensor,
         kscale: torch.Tensor, vscale: torch.Tensor, out: torch.Tensor,
         pos: torch.Tensor, rep: int, scale: float,
         part_acc: torch.Tensor, part_ml: torch.Tensor) -> torch.Tensor:
    """在 int8 KV 缓存上做融合的单 token 注意力；返回 ``out``。

    ``part_acc``/``part_ml`` 是 online softmax 按 key 切片的部分和
    （``[nh, ATTN_SPLIT, hd]`` fp32 与 ``[nh, ATTN_SPLIT, 2]`` fp32）：内核在每头上跑
    ``ATTN_SPLIT`` 个块，所以这些缓冲区归调用方所有，并为图捕获保持静止。

    Args:
        q: 本 token 的查询，``[nh, hd]``，已完成缩放和旋转。
        gate: 输出门控，``[nh, hd]``；``sigmoid`` 由内核施加。
        kcache: int8 key，``[nkv, max_seq, hd]``。
        vcache: int8 value，``[nkv, max_seq, hd]``。
        kscale: 每 (头, token) 的 fp32 K scale，``[nkv, max_seq]``。
        vscale: 每 (头, token) 的 fp32 V scale，``[nkv, max_seq]``。
        out: 输出缓冲区 ``[nh, hd]``，原地写。
        pos: 单元素 long 张量，装本 token 的索引；它同时是因果边界，越过它的条目被掩掉
            而不是从循环里排除。
        rep: 每个 KV 头对应的查询头数（GQA 组大小）。
        scale: ``hd ** -0.5``，由内核折进 K scale。
        part_acc: split-K 累加器，``[nh, ATTN_SPLIT, hd]`` fp32，归调用方所有。
        part_ml: split-K 的滑动最大值/分母，``[nh, ATTN_SPLIT, 2]`` fp32。

    Returns:
        ``out`` 本身，便于链式调用。

    Raises:
        RuntimeError: 扩展不可用。
    """
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    return module.attn_decode(q, gate, kcache, vcache, kscale, vscale, out, pos, rep,
                              scale, part_acc, part_ml)


def attn_paged(q: torch.Tensor, gate: torch.Tensor, kpool: torch.Tensor,
               vpool: torch.Tensor, kscale: torch.Tensor, vscale: torch.Tensor,
               pos2slot: torch.Tensor, out: torch.Tensor, pos: torch.Tensor,
               max_pos: int, rep: int, scale: float,
               part_acc: torch.Tensor, part_ml: torch.Tensor) -> torch.Tensor:
    """同样的融合注意力，跑在某层分页 int8 池上。

    Args:
        q: 本 token 的查询，``[nh, hd]``。
        gate: 输出门控，``[nh, hd]``。
        kpool: 本层的 int8 K 池，``[nslots, nkv, hd]``。
        vpool: 本层的 int8 V 池，``[nslots, nkv, hd]``。
        kscale: 每槽位的 fp32 K scale，``[nslots, nkv]``。
        vscale: 每槽位的 fp32 V scale，``[nslots, nkv]``。
        pos2slot: 本行的槽位表，只读它的前 ``pos + 1`` 项。
        out: 输出缓冲区 ``[nh, hd]``，原地写。
        pos: 单元素 long 张量，装本 token 的索引；即因果边界。
        max_pos: 窗口长度，即内核扫描到的那个槽位的行距。
        rep: 每个 KV 头对应的查询头数。
        scale: ``hd ** -0.5``。
        part_acc: split-K 累加器，``[nh, ATTN_SPLIT, hd]`` fp32。
        part_ml: split-K 的滑动最大值/分母，``[nh, ATTN_SPLIT, 2]`` fp32。

    Returns:
        ``out`` 本身，便于链式调用。

    Raises:
        RuntimeError: 扩展不可用。
    """
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    return module.attn_paged_decode(q, gate, kpool, vpool, kscale, vscale,
                                    pos2slot, out, pos, max_pos, rep, scale,
                                    part_acc, part_ml)


def rms_norm(x: torch.Tensor, w: torch.Tensor, stride: int, eps: float = 1e-6) -> torch.Tensor:
    """在 [M, N] 的多行上做融合的 ``x * rsqrt(mean(x^2)+eps) * w``。

    Args:
        x: 输入行 ``[M, N]``。
        w: 增益，``[N]``（也可以是带间距的，见 ``stride``）。
        stride: ``x`` 相邻两行之间的元素间距，让指向更宽缓冲区的视图不必拷贝就能归一化
            同样的这些行。
        eps: 方差下限，与 checkpoint 的 ``layer_norm_eps`` 对齐。

    Returns:
        归一化后的张量，形状同 ``x``。

    Raises:
        RuntimeError: 扩展不可用。
    """
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    return module.rms_norm(x, w, stride, eps)


def rms_norm_gated(x: torch.Tensor, w: torch.Tensor, gate: torch.Tensor,
                   eps: float = 1e-6) -> torch.Tensor:
    """融合的 ``rms(x) * w * silu(gate)``。

    Args:
        x: 输入 ``[num_v, dv]``，每个 value 头一行。
        w: 每头增益 ``[dv]``。它**不是** ``[num_v*dv]``：归一化是按头做的，传展平后的
            宽度会读到张量边界之外。
        gate: 门控投影 ``[num_v, dv]``；silu 在这里施加。
        eps: 方差下限。

    Returns:
        带门控的归一化张量，形状同 ``x``。

    Raises:
        RuntimeError: 扩展不可用。
    """
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    return module.rms_norm_gated(x, w, gate, eps)


# ``kernels/batch.cu`` 的批处理解码入口。
#
# B 个序列一次启动而不是 B 次；上面的单行入口保留为 B = 1 / 投机校验路径。批内核里
# 的每一行都与对应的单行内核数值一致，注意力内核每行只遍历该行自己的
# ``context_lens[b]`` 个 key。


def attn_paged_batch(q: torch.Tensor, gate: torch.Tensor,
                     kpool: torch.Tensor, vpool: torch.Tensor,
                     kscale: torch.Tensor, vscale: torch.Tensor,
                     slots: torch.Tensor, context_lens: torch.Tensor,
                     out: torch.Tensor, max_seq: int, rep: int, scale: float,
                     part_acc: torch.Tensor, part_ml: torch.Tensor) -> torch.Tensor:
    """按行在 int8 池窗口上做批处理分页注意力。

    ``q``/``gate``/``out`` 是 ``[B, nh, hd]``（bf16），``kpool``/``vpool`` 是
    ``[nslots, nkv, hd]`` int8 配 ``[nslots, nkv]`` fp32 scale，``slots`` 是
    ``[B, max_seq]`` 的设备侧槽位表（只读第 ``b`` 行的前 ``context_lens[b]`` 项），
    ``context_lens`` 是 ``[B]`` int32。``part_acc``/``part_ml`` 是 split-K 的
    online-softmax 部分和，``[B, nh, split, hd]`` 与 ``[B, nh, split, 2]`` fp32，归
    调用方所有，形状因此为图捕获保持静态。

    Args:
        q: 查询 ``[B, nh, hd]`` bf16，已完成旋转和缩放。
        gate: 输出门控 ``[B, nh, hd]`` bf16。
        kpool: 本层的 int8 K 池，``[nslots, nkv, hd]``。
        vpool: 本层的 int8 V 池，``[nslots, nkv, hd]``。
        kscale: 每槽位的 fp32 K scale，``[nslots, nkv]``。
        vscale: 每槽位的 fp32 V scale，``[nslots, nkv]``。
        slots: 每行的设备侧槽位表，``[B, max_seq]``。
        context_lens: 每行的上下文长度 ``[B]`` int32。第 ``b`` 行只读它自己槽位表的前
            ``context_lens[b]`` 项，混合 prompt 长度的批桶因此能共用一次启动。
        out: 输出缓冲区 ``[B, nh, hd]``，原地写。
        max_seq: 窗口长度，即 ``slots`` 的行距。
        rep: 每个 KV 头对应的查询头数。
        scale: ``hd ** -0.5``。
        part_acc: split-K 累加器，``[B, nh, split, hd]`` fp32。
        part_ml: split-K 的滑动最大值/分母，``[B, nh, split, 2]`` fp32。

    Returns:
        ``out`` 本身，便于链式调用。

    Raises:
        RuntimeError: 扩展不可用。
    """
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    return module.attn_paged_batch(q, gate, kpool, vpool, kscale, vscale, slots,
                                   context_lens, out, max_seq, rep, scale,
                                   part_acc, part_ml)


def attn_paged_multi_q(q: torch.Tensor, gate: torch.Tensor,
                       kpool: torch.Tensor, vpool: torch.Tensor,
                       kscale: torch.Tensor, vscale: torch.Tensor,
                       slots: torch.Tensor, ctx_lens: torch.Tensor,
                       out: torch.Tensor, max_seq: int, rep: int, scale: float,
                       part_acc: torch.Tensor, part_ml: torch.Tensor,
                       block_mask: torch.Tensor | None = None,
                       mask_block: int = 64) -> torch.Tensor:
    """校验注意力：每行的 ``c`` 个草稿查询共享一次历史 KV 遍历。

    ``q``/``gate``/``out`` 是 ``[B, c, nh, hd]`` bf16，``ctx_lens`` 是每行的历史长度
    （查询 ``j`` 看到 ``ctx + j + 1`` 个 key，即历史加上前 ``j+1`` 个新 token）。每个
    查询拿到与 ``attn_paged_batch`` 相同的 split-K / online-softmax / 门控运算，所以
    与单独解码该 token 逐位一致。``part_acc``/``part_ml`` 是
    ``[B*c, nh, split, hd]`` / ``[B*c, nh, split, 2]`` fp32。

    ``block_mask``（uint8 ``[B, n_blocks]``，1 表示保留，块大小为 ``mask_block``）选定
    哪些 KV 块参与；``None`` 是精确的全注意力。

    Args:
        q: 草稿查询 ``[B, c, nh, hd]`` bf16，已完成旋转和缩放。
        gate: 输出门控，形状同上。
        kpool: 本层的 int8 K 池，``[nslots, nkv, hd]``。
        vpool: 本层的 int8 V 池，``[nslots, nkv, hd]``。
        kscale: 每槽位的 fp32 K scale，``[nslots, nkv]``。
        vscale: 每槽位的 fp32 V scale，``[nslots, nkv]``。
        slots: 每行的设备侧槽位表，``[B, max_seq]``。
        ctx_lens: 每行的历史长度 ``[B]`` int32。查询 ``j`` 看到 ``ctx_lens[b] + j + 1``
            个 key，即历史加上截至并包含它自己的新 token。
        out: 输出缓冲区 ``[B, c, nh, hd]``，原地写。
        max_seq: 窗口长度，即 ``slots`` 的行距。
        rep: 每个 KV 头对应的查询头数。
        scale: ``hd ** -0.5``。
        part_acc: split-K 累加器，``[B*c, nh, split, hd]`` fp32。
        part_ml: split-K 的滑动最大值/分母，``[B*c, nh, split, 2]`` fp32。
        block_mask: 可选的 uint8 块保留掩码，``[B, n_blocks]``。
        mask_block: 掩码所表达的块大小。

    Returns:
        ``out`` 本身，便于链式调用。

    Raises:
        RuntimeError: 扩展不可用。
    """
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    if block_mask is None:
        block_mask = torch.empty(0, dtype=torch.uint8, device=q.device)
    return module.attn_paged_multi_q(q, gate, kpool, vpool, kscale, vscale,
                                     slots, ctx_lens, out, max_seq, rep, scale,
                                     part_acc, part_ml, block_mask, mask_block)


def conv1d_silu_batch(conv_w: torch.Tensor, x: torch.Tensor,
                      conv_state: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """批处理的融合深度因果 conv1d + SiLU。

    ``x`` [B, C] bf16，``conv_state`` [B, C, 3] fp32（原地平移），``out`` [B, C] fp32。

    Args:
        conv_w: 深度卷积的抽头系数 ``[C, 4]`` fp32。
        x: 每行一个 token，``[B, C]``。
        conv_state: 每行的卷积窗口 ``[B, C, 3]`` fp32，原地平移。
        out: 输出缓冲区 ``[B, C]`` fp32，原地写。

    Returns:
        ``out`` 本身，便于链式调用。

    Raises:
        RuntimeError: 扩展不可用。
    """
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    return module.conv1d_silu_batch(conv_w, x, conv_state, out)


def delta_prologue_batch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         b: torch.Tensor, a: torch.Tensor,
                         neg_expA: torch.Tensor, dt_bias: torch.Tensor,
                         q_out: torch.Tensor, k_out: torch.Tensor,
                         v_out: torch.Tensor, beta: torch.Tensor,
                         g: torch.Tensor, qscale: float, rep: int) -> None:
    """批处理的 q/k 归一化 + 头扩展 + beta/g 构建（见 ``delta_prologue``）。

    Args:
        q: 每个 key 头的查询，``[B, num_k, dk]``。
        k: 每个 key 头的 key，``[B, num_k, dk]``。
        v: 每个 value 头的 value，``[B, num_v, dv]``。
        b: beta 的 logits，``[B, num_v]``。
        a: 衰减的 logits，``[B, num_v]``。
        neg_expA: ``-exp(A_log)``，``[num_v]`` fp32，所有行共享。
        dt_bias: 衰减偏置，``[num_v]`` fp32，所有行共享。
        q_out: 输出查询，``[B, num_v, dk]`` fp32。
        k_out: 输出 key，``[B, num_v, dk]`` fp32。
        v_out: 输出 value，``[B, num_v, dv]`` fp32。
        beta: 输出的写入强度，``[B, num_v]`` fp32。
        g: 输出的对数衰减 ``<= 0``，``[B, num_v]`` fp32。
        qscale: ``dk ** -0.5``。
        rep: 每个 key 头对应的 value 头数。

    Returns:
        无。每个输出缓冲区都是原地写入；单独看每一行，结果与 :func:`delta_prologue`
        逐位一致。

    Raises:
        RuntimeError: 扩展不可用。
    """
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    module.delta_prologue_batch(q, k, v, b, a, neg_expA, dt_bias,
                                q_out, k_out, v_out, beta, g, qscale, rep)


def delta_batch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                beta: torch.Tensor, g: torch.Tensor,
                st: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """批处理的 gated delta 更新；原地更新 ``st`` [B, nv, dk, dv]。

    Args:
        q: 每个 value 头归一化后的查询，``[B, num_v, dk]`` fp32。
        k: 每个 value 头归一化后的 key，``[B, num_v, dk]`` fp32。
        v: 每个 value 头的 value，``[B, num_v, dv]`` fp32。
        beta: 写入强度，取值在 ``[0, 1]``，``[B, num_v]``。
        g: 对数衰减 ``<= 0``，``[B, num_v]``。
        st: 每行的循环状态 ``[B, num_v, dk, dv]`` fp32，原地更新。
        out: 输出缓冲区 ``[B, num_v, dv]`` fp32，原地写。

    Returns:
        ``out`` 本身，便于链式调用。

    Raises:
        RuntimeError: 扩展不可用。
    """
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    return module.delta_batch(q, k, v, beta, g, st, out)


def gemv_w8a16_wide(w: torch.Tensor, scale: torch.Tensor, x: torch.Tensor,
                    U: int = 2, threads: int = 0) -> torch.Tensor:
    """Wide-M 的 int8 W8A16 GEMV：``w`` [N, K] int8，``scale`` [N]，``x`` [M, K] bf16。

    ``M`` 必须是 16、24 或 32；一次权重读取，不做反量化。每一行都与对该行的 8 激活切片
    调用 ``gemv_w8a16_m`` 逐位一致。``U`` 是 K 方向展开（2 或 4）；``threads=0`` 选一个
    能把 warp 均匀分进各激活组的块大小（M=24 时为 192，其余为 256）。

    Args:
        w: int8 权重矩阵 ``[N, K]``，行主序。
        scale: 每行的 fp32 scale，``[N]``。
        x: 激活 ``[M, K]`` bf16，``M`` 取 {16, 24, 32}。
        U: K 方向展开因子（2 或 4）。
        threads: 每块线程数；``0`` 表示采用上面那个均匀切分的默认值。

    Returns:
        ``[M, N]`` bf16 输出。改成实体化权重则每次调用要多占 ``[N, K]`` 的常驻字节。

    Raises:
        RuntimeError: 扩展不可用。
    """
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    return module.gemv_w8a16_wide(w, scale, x, U, threads)


def gemv_w8a16_wide_dbg(w: torch.Tensor, scale: torch.Tensor, x: torch.Tensor,
                        U: int, rows: int, mw: int, threads: int = 0) -> torch.Tensor:
    """A/B 入口：每个 warp 处理 ``rows`` 行输出和 ``mw`` 个激活向量。

    Args:
        w: int8 权重矩阵 ``[N, K]``。
        scale: 每行的 fp32 scale，``[N]``。
        x: 激活 ``[M, K]`` bf16。
        U: K 方向展开因子（2 或 4）。
        rows: 每块覆盖的输出行数，覆盖调优后的默认值。
        mw: 每个 warp 的激活向量数，覆盖调优后的默认值。
        threads: 每块线程数；``0`` 保持编译时的默认值。

    Returns:
        ``[M, N]`` bf16 输出，算术与 :func:`gemv_w8a16_wide` 相同；只有启动形状不同，
        所以两者逐位一致，可以在基准测试里互相卡住对方。

    Raises:
        RuntimeError: 扩展不可用。
    """
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    return module.gemv_w8a16_wide_dbg(w, scale, x, U, rows, mw, threads)


def gemv_w8a16_wide_dot2(w: torch.Tensor, scale: torch.Tensor, x: torch.Tensor,
                         U: int = 2, threads: int = 0) -> torch.Tensor:
    """用 packed fp16 ``v_dot2_f32_f16`` 数学做的 Wide-M int8 权重 GEMV。

    ``M`` 必须是 16、24 或 32；fp32 累加，激活每次调用只转一次 fp16。数值上接近
    ``gemv_w8a16_wide``（不逐位一致）。

    Args:
        w: int8 权重矩阵 ``[N, K]``，行主序。
        scale: 每行的 fp32 scale，``[N]``。
        x: 激活 ``[M, K]`` bf16，``M`` 取 {16, 24, 32}。
        U: K 方向展开因子（2 或 4）。
        threads: 每块线程数；``0`` 保持编译时的默认值。

    Returns:
        ``[M, N]`` bf16 输出。packed 的 ``v_dot2_f32_f16`` 路径相对标量 int8 路径把每个
        权重元素的算术指令数减半，但它与 :func:`gemv_w8a16_wide` 不逐位一致，所以只在
        调用方能接受这点舍入差异的地方启用。

    Raises:
        RuntimeError: 扩展不可用。
    """
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    return module.gemv_w8a16_wide_dot2(w, scale, x, U, threads)


def gemv_w8a16_wide_dot2_dbg(w: torch.Tensor, scale: torch.Tensor, x: torch.Tensor,
                             U: int, rows: int, mw: int,
                             threads: int = 0) -> torch.Tensor:
    """A/B 入口：每个 warp 处理 ``rows`` 行输出和 ``mw`` 个激活向量。

    Args:
        w: int8 权重矩阵 ``[N, K]``。
        scale: 每行的 fp32 scale，``[N]``。
        x: 激活 ``[M, K]`` bf16。
        U: K 方向展开因子（2 或 4）。
        rows: 每块覆盖的输出行数，覆盖调优后的默认值。
        mw: 每个 warp 的激活向量数，覆盖调优后的默认值。
        threads: 每块线程数；``0`` 保持编译时的默认值。

    Returns:
        ``[M, N]`` bf16 输出，算术与 :func:`gemv_w8a16_wide_dot2` 相同，只有启动形状
        变了。

    Raises:
        RuntimeError: 扩展不可用。
    """
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    return module.gemv_w8a16_wide_dot2_dbg(w, scale, x, U, rows, mw, threads)

