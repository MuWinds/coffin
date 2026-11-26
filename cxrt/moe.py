"""MoE 族已经搬进 C ABI 的算子面：``shared_gate_scale``。

Module: ``coffin/cxrt``。只 import ``ctypes`` 与 ``cxrt.*``，**不 import torch**。

为什么先只有这一条：MoE 里其余入口（``moe_router`` / ``moe_dispatch`` /
``moe_grouped_i8*``）都要连分组偏移、槽位表一起定契约，而共享专家的门标量是
「一个 fp32 标量决定一整行」——形状最简单，也没有块表。它在 decode 每一步、每层
各来一次，是这一族里最先被量到的启动开销。

Constraints:
    * 行间距一律**显式**传参，与 C ABI 一致。一维操作数（decode 传的是 ``[H]``）的
      行间距必须写成 ``width``：内核按 pitch 走指针，写成 stride(0)=1 会读回第 0 行的
      通道块，而且不会让任何形状检查失败。
    * ``out`` 允许与 ``src`` 共享存储，其余重叠是调用方的 bug，本层不检测。
    * ``moe_dispatch`` 只有出口、没有 ``MoeOps`` 方法：它的四个输出全是 ``[slots]``
      量级的整数表，校验（几何 + 共享内存预算）在出口里，Python 侧只需要一次
      ``shim`` 调用。等有离机单测要跑它时再加 ``Array`` 面。
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

from cxrt import tensors as T
from cxrt.dtypes import BY_CODE, DType, I64
from cxrt.errors import ShapeError
from cxrt.shim import Shim


class MoeShim:
    """MoE 出口的薄转发层。"""

    def __init__(self, shim: Shim) -> None:
        """绑定出口。

        Args:
            shim: 已装载的共享库。
        """
        p = ctypes.c_void_p
        i = ctypes.c_int
        ll = ctypes.c_longlong
        self.shared_gate_scale = shim.export(
            "coffin_shared_gate_scale", [p, p, p, p, i, i, ll, ll, ll, i, p])
        self.moe_gather_sum = shim.export(
            "coffin_moe_gather_sum", [p, p, p, p, i, i, i, i, i, i, p])
        self.moe_router = shim.export(
            "coffin_moe_router", [p, p, p, i, i, i, i, i, p])
        self.moe_dispatch = shim.export(
            "coffin_moe_dispatch", [p, p, p, p, p, p, p, i, i, i, i, i, p])
        # 分组 int8 的「一个专家一趟」核：目的地 dtype 由调用方给（ABI 面没有 dtype 可读）。
        self.grouped_i8_expert = shim.export(
            "coffin_moe_grouped_i8_expert", [p, p, p, p, p, p, i, i, i, i, i, p])
        # 分组 int8 的单趟核：``part`` 只在分片（``slices > 1``）时需要，nk 自动片数恒为 1。
        self.grouped_i8 = shim.export(
            "coffin_moe_grouped_i8", [p] * 7 + [i] * 6 + [p])
        # 预填充那条：先 ``transpose_i8`` 把激活摆成 k 主序，再进分组张量核。
        self.transpose_i8 = shim.export("coffin_transpose_i8", [p, p, i, i, p])
        self.grouped_i8_tcu = shim.export(
            "coffin_moe_grouped_i8_tcu", [p] * 6 + [i] * 5 + [p])
        # 归并形态两条：分片缓冲与到达计数由出口自己按容量缓存（原来在这里的是 torch 张量）。
        self.merge = shim.export(
            "coffin_moe_grouped_i8_merge", [p] * 7 + [i] * 6 + [p])
        self.merge_tokens = shim.export(
            "coffin_moe_grouped_i8_merge_tokens", [p] * 9 + [i] * 9 + [p])
        self.path = shim.path


def load_moe_shim(lib_path: str | Path) -> MoeShim:
    """装载 MoE shim。

    Args:
        lib_path: ``cxrt.build`` 产出的 ``.so`` 路径。

    Returns:
        绑好出口的 :class:`MoeShim`。
    """
    return MoeShim(Shim(lib_path))


class MoeOps:
    """MoE 算子的可调用面。

    Attributes:
        shim: :class:`MoeShim`。
    """

    def __init__(self, shim: MoeShim) -> None:
        """绑定出口。

        Args:
            shim: 已装载的共享库。
        """
        self.shim = shim

    def shared_gate_scale(self, x: T.TensorLike, w: T.TensorLike, src: T.TensorLike,
                          out: T.TensorLike | None = None,
                          stream: int | None = None) -> T.TensorLike:
        """``out = src * sigmoid(x . w)``，逐行一个标量。

        Args:
            x: ``[rows, width]`` 或 ``[width]``，最内层步长 1。
            w: ``[width]`` fp32 门权重，连续。
            src: 被缩放的操作数，形状同 ``x``。
            out: 同形状目的地；``None`` 时分配成稠密。
            stream: 流句柄。

        Returns:
            ``out``。

        Raises:
            ShapeError: 秩、dtype、系统或不连续。
        """
        shape = T.shape_of(x)
        if not shape:
            raise ShapeError(f"activation must not be empty, got {shape}")
        width = int(shape[-1])
        rows = 1 if len(shape) <= 1 else int(T.numel_of(x) // width)
        for name, operand in (("source", src), ("destination", out)):
            if operand is None:
                continue
            other = T.shape_of(operand)
            if other != shape:
                raise ShapeError(f"{name} shape {other} must match the activation {shape}")
        if T.code_of(w) != DType.F32.code:
            raise ShapeError(
                f"gate weight must be fp32, got {BY_CODE[T.code_of(w)].name}")
        if T.code_of(x) not in (DType.BF16.code, DType.F32.code):
            raise ShapeError(f"operands must be bf16 or fp32, got {BY_CODE[T.code_of(x)].name}")
        code = T.code_of(x)
        x_pitch = width if len(shape) <= 1 else int(T.strides_of(x)[-2])
        s_pitch = width if len(shape) <= 1 else int(T.strides_of(src)[-2])
        if out is None:
            out = T.new_empty(x, (rows, width), code)
            o_pitch = width
        else:
            o_pitch = width if len(shape) <= 1 else int(T.strides_of(out)[-2])
        self.shim.shared_gate_scale(T.ptr_of(x), T.ptr_of(w), T.ptr_of(src), T.ptr_of(out),
                                    rows, width, x_pitch, s_pitch, o_pitch, code,
                                   (stream or 0))
        return out

    def moe_router(self, logits: T.TensorLike, top_k: int, normalize: bool = True,
                   stream: int | None = None) -> tuple[T.TensorLike, T.TensorLike]:
        """融合的 ``softmax -> top-k -> 归一化``：一次发射写完下标与权重。

        Args:
            logits: ``[tokens, experts]`` 连续 bf16 或 fp32。
            top_k: 每 token 的专家数，``(0, 32]`` 且不超过专家数。
            normalize: 是否把 top-k 概率再归一化。
            stream: 流句柄。

        Returns:
            ``(index, weight)``：前者 ``[tokens, top_k]`` int64 按概率降序，后者同形、
            dtype 同 ``logits``。

        Raises:
            ShapeError: 不是二维、专家数超上界或 top_k 越界。
        """
        shape = T.shape_of(logits)
        if len(shape) != 2:
            raise ShapeError(f"router logits must be [tokens, experts], got {shape}")
        tokens, experts = int(shape[0]), int(shape[1])
        if top_k <= 0 or top_k > 32 or top_k > experts:
            raise ShapeError(f"top_k must be in (0, min(32, {experts})], got {top_k}")
        code = T.code_of(logits)
        if code not in (DType.BF16.code, DType.F32.code):
            raise ShapeError(f"router logits must be bf16 or fp32, got {BY_CODE[code].name}")
        # 内核按 ``token * n`` 取行，带间距的视图会读错行。
        if T.strides_of(logits) != (experts, 1):
            raise ShapeError(f"router logits must be contiguous, got strides "
                             f"{T.strides_of(logits)} with width {experts}")
        index = T.new_empty(logits, (tokens, top_k), I64.code)
        weight = T.new_empty(logits, (tokens, top_k), code)
        if tokens == 0:
            return index, weight
        self.shim.moe_router(T.ptr_of(logits), T.ptr_of(index), T.ptr_of(weight), tokens,
                             experts, int(top_k), 1 if normalize else 0, code,
                             (stream or 0))
        return index, weight
