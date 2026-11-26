"""卷积族的 C ABI 面：``conv1d_silu`` / ``conv1d_silu_seq`` / ``conv1d_silu_batch``。

Module: ``coffin/cxrt``。只 import ``ctypes`` 与 ``cxrt.*``，**不 import torch**。

这是 qwen35 每条线性注意力层的入口：4 抽头深度因果卷积接 SiLU，历史存在一个三元素
的环里原地平移。三条入口共用一份内核——批处理那条启动的就是单 token 那条，行轴收成
``rows``——所以它们的算术不可能各自漂移。

Constraints:
    * 激活只能是 **bf16**：内核读 ``uint16_t``。fp32 在这里是错误，不是可选的另一半。
    * ``state`` 原地更新是语义：它是这一层的短期记忆，必须与循环状态一起推进；下一个
      解码步接着同一条递推，这正是分块预填充能与一次性预填充一致的原因。
    * 行间距一律从张量自身读。校验主体传进来的是 ``qkv[:, j]``（``[rows, c, conv_dim]``
      的列切片），行间距是 ``c * C``；按 ``C`` 去读会在 ``rows > 1`` 时静默取错行。
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

from cxrt import tensors as T
from cxrt.dtypes import BY_CODE, F32
from cxrt.errors import ShapeError
from cxrt.shim import Shim

#: 抽头数与历史长度。与 ``conv_device.inc`` 的 ``CONV_K`` / ``CONV_STATE`` 同源，改一处
#: 必须改两处。
CONV_K = 4
CONV_STATE = CONV_K - 1


class ConvShim:
    """三个卷积出口的薄转发层。"""

    def __init__(self, shim: Shim) -> None:
        """绑定出口。

        Args:
            shim: 已装载的共享库。
        """
        p = ctypes.c_void_p
        i = ctypes.c_int
        ll = ctypes.c_longlong
        # ``Shim.export`` 自己追加 ``err`` / ``errlen``，所以每个列表只列到 ``stream``
        # 为止。多列两项会让 ctypes 在调用点报 "takes at least N arguments"。
        self.conv1d_silu = shim.export(
            "coffin_conv1d_silu", [p, p, p, p, i, p])
        self.conv1d_silu_seq = shim.export(
            "coffin_conv1d_silu_seq", [p, p, p, p, i, i, ll, p])
        self.conv1d_silu_batch = shim.export(
            "coffin_conv1d_silu_batch", [p, p, p, p, i, i, ll, ll, ll, p])
        self.path = shim.path


def load_conv_shim(lib_path: str | Path) -> ConvShim:
    """装载卷积 shim。

    Args:
        lib_path: ``cxrt.build`` 产出的 ``.so`` 路径。

    Returns:
        绑好三个出口的 :class:`ConvShim`。
    """
    return ConvShim(Shim(lib_path))


class ConvOps:
    """卷积三兄弟的可调用面。

    Attributes:
        shim: :class:`ConvShim`。
    """

    def __init__(self, shim: ConvShim) -> None:
        self.shim = shim

    def conv1d_silu(self, conv_w: T.TensorLike, x: T.TensorLike, conv_state: T.TensorLike,
                    out: T.TensorLike | None = None,
                    stream: int | None = None) -> T.TensorLike:
        """单 token：``out = silu(conv(x))``，窗口原地平移。

        Args:
            conv_w: ``[C, 4]`` fp32 抽头。
            x: ``[C]`` bf16 激活。
            conv_state: ``[C, 3]`` fp32 历史。
            out: ``[C]`` fp32 目的地；``None`` 时分配。
            stream: 流句柄。

        Returns:
            ``[C]`` fp32 结果。

        Raises:
            ShapeError: dtype、形状或历史长度不符。
        """
        channels = self._check(conv_w, x, 1)
        self._check_state(conv_state, 1, channels)
        dest = self._dest(x, (channels,), out)
        self.shim.conv1d_silu(T.ptr_of(conv_w), T.ptr_of(x), T.ptr_of(conv_state),
                              T.ptr_of(dest), channels, _stream(stream))
        return dest

    def conv1d_silu_seq(self, conv_w: T.TensorLike, x: T.TensorLike,
                        conv_state: T.TensorLike, out: T.TensorLike | None = None,
                        stream: int | None = None) -> T.TensorLike:
        """预填充分块：沿 ``T`` 分段跑同一个卷积，最后一段回写窗口。

        Args:
            conv_w: ``[C, 4]`` fp32 抽头。
            x: ``[T, C]`` bf16 激活，最内层步长 1；行间距可以大于 ``C``。
            conv_state: ``[C, 3]`` fp32 历史，原地更新。
            out: ``[T, C]`` fp32 目的地；``None`` 时分配。
            stream: 流句柄。

        Returns:
            ``[T, C]`` fp32 结果。

        Raises:
            ShapeError: 不是二维、最内层步长不是 1、dtype 或历史长度不符。
        """
        if len(T.shape_of(x)) != 2:
            raise ShapeError(f"activation must be [T, C], got {T.shape_of(x)}")
        tokens, channels = (int(v) for v in T.shape_of(x))
        self._check(conv_w, x, tokens)
        self._check_state(conv_state, 1, channels)
        ldx = T.strides_of(x)[0]
        if T.strides_of(x)[1] != 1:
            raise ShapeError(f"activation must have unit inner stride, got {T.strides_of(x)}")
        dest = self._dest(x, (tokens, channels), out)
        self.shim.conv1d_silu_seq(T.ptr_of(conv_w), T.ptr_of(x), T.ptr_of(conv_state),
                                  T.ptr_of(dest), tokens, channels, int(ldx),
                                  _stream(stream))
        return dest

    def conv1d_silu_batch(self, conv_w: T.TensorLike, x: T.TensorLike,
                          conv_state: T.TensorLike, out: T.TensorLike | None = None,
                          stream: int | None = None) -> T.TensorLike:
        """批处理单 token：所有行一次启动，每行一个窗口。

        Args:
            conv_w: ``[C, 4]`` fp32 抽头，所有行共用。
            x: ``[B, C]`` bf16 激活，最内层步长 1。
            conv_state: ``[B, C, 3]`` fp32 历史，原地平移。
            out: ``[B, C]`` fp32 目的地；``None`` 时分配。
            stream: 流句柄。

        Returns:
            ``[B, C]`` fp32 结果。

        Raises:
            ShapeError: 秩、dtype 或历史长度不符。
        """
        if len(T.shape_of(x)) != 2:
            raise ShapeError(f"activation must be [B, C], got {T.shape_of(x)}")
        rows, channels = (int(v) for v in T.shape_of(x))
        self._check(conv_w, x, rows)
        self._check_state(conv_state, rows, channels)
        strides = T.strides_of(x)
        if strides[1] != 1:
            raise ShapeError(f"activation must have unit inner stride, got {strides}")
        dest = self._dest(x, (rows, channels), out)
        self.shim.conv1d_silu_batch(T.ptr_of(conv_w), T.ptr_of(x), T.ptr_of(conv_state),
                                    T.ptr_of(dest), rows, channels, int(strides[0]),
                                    int(T.strides_of(conv_state)[0]),
                                    int(T.strides_of(dest)[0]), _stream(stream))
        return dest

    # 校验

    def _check(self, conv_w: T.TensorLike, x: T.TensorLike, rows: int) -> int:
        """检查抽头与激活，返回通道数。

        Args:
            conv_w: 抽头。
            x: 激活。
            rows: 期望行数。

        Returns:
            通道数 ``C``。

        Raises:
            ShapeError: 抽头不是 ``[C, 4]``、激活不是 bf16，或行数不符。
        """
        code = T.code_of(x)
        if code != T.BF16C:
            raise ShapeError(f"activation must be bf16, got {BY_CODE[code].name}")
        taps = T.shape_of(conv_w)
        if len(taps) != 2 or taps[1] != CONV_K:
            raise ShapeError(f"taps must be [C, {CONV_K}], got {taps}")
        if T.code_of(conv_w) != T.F32:
            raise ShapeError(f"taps must be fp32, got {BY_CODE[T.code_of(conv_w)].name}")
        width = int(taps[0])
        shape = T.shape_of(x)
        if len(shape) == 1:
            if rows != 1:
                raise ShapeError(f"a flat activation is one row, got rows={rows}")
            if shape[0] != width:
                raise ShapeError(f"activation must be [{width}], got {shape}")
            return width
        if shape[1] != width:
            raise ShapeError(f"activation must be C={width} wide, got {shape}")
        if shape[0] != rows:
            raise ShapeError(f"activation has {shape[0]} rows, expected {rows}")
        return width

    def _check_state(self, conv_state: T.TensorLike, rows: int, channels: int) -> None:
        """历史必须是 ``[rows, C, 3]`` 的连续块。

        Args:
            conv_state: 历史。
            rows: 行数。
            channels: 通道数。

        Raises:
            ShapeError: dtype 或元素数不符。
        """
        if T.code_of(conv_state) != T.F32:
            raise ShapeError(
                f"conv state must be fp32, got {BY_CODE[T.code_of(conv_state)].name}")
        need = rows * channels * CONV_STATE
        if T.numel_of(conv_state) != need:
            raise ShapeError(f"conv state must hold {need} values, got {T.numel_of(conv_state)}")

    def _dest(self, anchor: T.TensorLike, shape: tuple[int, ...],
              out: T.TensorLike | None) -> T.TensorLike:
        """决定目的地：用调用方的，或按 fp32 分配。

        Args:
            anchor: 分配锚点（激活）。
            shape: 目的地形状。
            out: 给定或 ``None``。

        Returns:
            目的地。

        Raises:
            ShapeError: 形状或 dtype 不符。
        """
        if out is None:
            return T.new_empty(anchor, shape, F32.code)
        if T.shape_of(out) != shape or T.code_of(out) != T.F32:
            raise ShapeError(
                f"out must be {list(shape)} fp32, got {T.shape_of(out)} "
                f"{BY_CODE[T.code_of(out)].name}")
        return out


def _stream(stream: int | None) -> Any:
    """流句柄。

    Args:
        stream: 句柄号或 ``None``。

    Returns:
        ``c_void_p`` 可接受的值。
    """
    return None if stream is None else int(stream)
