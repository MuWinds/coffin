"""设备 dtype 的描述与 bf16 位级转换，``rt`` 全层的类型词汇表。

Module: ``coffin/cxrt``。叶子模块，只 import ``numpy`` 与标准库。
**不 import torch**：``torch.dtype`` 是 corex 路径上要消失的东西，这里给出它的替代。

Constraints:
    * bf16 一律用 ``uint16`` 承载位模式，不引入 ``ml_dtypes`` 之类的第三方包。
      CoreX 的 clang 工具链与集群的 numpy 版本都不保证装得上额外 dtype 扩展，而
      内核本来就是按位容器（``uint16_t``）实例化的，所以宿主侧也按位看待它。
    * 窄化按就近取偶（round-half-to-even），与内核 ``CxScalar<uint16_t>`` 和
      ``torch`` 的转换一致。截断式转换会让每个投影系统性偏低，穿过 32 层累积。
    * 码值是内核 ABI 的一部分（``rt/shim.py`` 与 ``kernels/abi/*.cu`` 两边各写一份），
      改一个数就是改一次 ABI，必须同时改两处。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class DType:
    """一种设备存储类型。

    ``numpy`` 字段是这个 dtype 在宿主侧的观察方式：bf16 观察成 ``uint16``（位模式），
    其余一一对应。用「观察方式」而不是「等价类型」，是因为宿主侧从不对 bf16 做算术，
    只做搬运与对拍。

    Attributes:
        name: 人类可读名字，出现在错误消息与自检输出里。
        itemsize: 每个元素的字节数。
        code: 传给内核 ABI 的整数码。**改它就是改 ABI**。
        numpy: 宿主侧的观察 dtype。
        floating: 是不是浮点。整型不参与 ``eps`` / ``rsqrt`` 这类参数校验。
    """

    name: str
    itemsize: int
    code: int
    numpy: np.dtype
    floating: bool = True


#: 内核 ABI 实例化的浮点类型，码值与 ``kernels/abi/norm_abi.cu`` 的 ``CX_DTYPE_*`` 一致。
F32 = DType("f32", 4, 0, np.dtype(np.float32))
BF16 = DType("bf16", 2, 1, np.dtype(np.uint16))

#: 权重与 KV 用到的整型。码值同样与 ABI 对齐。
I8 = DType("i8", 1, 2, np.dtype(np.int8), floating=False)
I32 = DType("i32", 4, 3, np.dtype(np.int32), floating=False)
I64 = DType("i64", 8, 4, np.dtype(np.int64), floating=False)
U8 = DType("u8", 1, 5, np.dtype(np.uint8), floating=False)
U16 = DType("u16", 2, 6, np.dtype(np.uint16), floating=False)
U32 = DType("u32", 4, 7, np.dtype(np.uint32), floating=False)

#: 名字到 dtype，供自检与诊断按字符串取用。
BY_NAME: dict[str, DType] = {
    d.name: d for d in (F32, BF16, I8, I32, I64, U8, U16, U32)
}

#: 码值到 dtype。内核出口只收码值，所以这是 ABI 的唯一反向表。
BY_CODE: dict[int, DType] = {d.code: d for d in BY_NAME.values()}


def from_numpy(dt: np.dtype) -> DType:
    """把宿主 numpy dtype 映射成设备 dtype。

    Args:
        dt: 宿主数组的 dtype。

    Returns:
        对应的 :class:`DType`。

    Raises:
        ShapeError: 该 dtype 在设备侧没有对应存储（例如 float16 尚未接入，bf16 必须
            由调用方显式给出位容器）。静默挑一个「差不多」的类型会让一次转换错误
            表现为数值漂移，而不是参数错误。
    """
    from cxrt.errors import ShapeError

    for candidate in BY_NAME.values():
        if candidate.numpy == dt:
            return candidate
    raise ShapeError(f"no device dtype for host dtype {dt}; cast explicitly first")


def bf16_bits(values: np.ndarray) -> np.ndarray:
    """把 fp32 数组按就近取偶窄化成 bf16 的位模式。

    内核里 ``CxScalar<uint16_t>::from_float`` 是同一套位运算，这里逐位复刻它，所以
    host 侧参考与设备侧输出可以按位对拍，而不是比一个容差。

    Args:
        values: ``float32`` 或可无损转成 ``float32`` 的数组。

    Returns:
        ``uint16`` 数组，长度与形状同 ``values``。
    """
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32).astype(np.uint64)
    # 加 0x7fff 再按第 16 位补一位，就是 round-half-to-even；NaN 的载荷保持不动，
    # 因为加法溢出进指数位时结果仍是 NaN。
    bits = bits + 0x7FFF + ((bits >> 16) & 1)
    return (bits >> 16).astype(np.uint16)


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    """把 bf16 位模式还原成 fp32，供宿主侧参考计算。

    Args:
        bits: ``uint16`` 位模式数组。

    Returns:
        ``float32`` 数组。
    """
    wide = np.ascontiguousarray(bits, dtype=np.uint16).astype(np.uint32) << 16
    return wide.view(np.float32)
