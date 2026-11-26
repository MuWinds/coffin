"""torch 形状的薄壳：让 ``models/`` / ``engine/`` 的推理逻辑在没有 torch 的进程里跑。

Module: ``coffin/cxrt``。这不是「再写一个 torch」，只覆盖推理链实际用到的面：

- 数据结构：``Tensor`` + dtype 常量 + ``device``；
- 容器与常量：``zeros`` / ``empty`` / ``full`` / ``arange`` / ``cat`` / ``stack`` / ``split``；
- 少量真算子：``matmul`` / ``bmm`` / ``softmax`` / ``where`` / ``nonzero`` / ``cumsum``；
- 训练期残留的壳：``nn.Module`` / ``Parameter`` / ``Linear`` / ``Embedding`` / ``F``；
- 上下文管理：``no_grad`` / ``inference_mode``。

**为什么先做成 numpy 后端**：模型层的改造分两步走，先要「逻辑跑通、能逐层对拍
logits」，再谈把重算子接回 ``cxrt`` 的 ABI。numpy 后端让第一步不依赖卡、不依赖编译，
在任何机器上都能跑；等模型对齐之后，把 ``Tensor`` 底下的存储换成 ``cxrt.Array``、
把 ``matmul`` / ``softmax`` 换成 ABI 出口，改动都收在这个文件里。

数值约定：``bfloat16`` 用 fp32 存放、在写入时按 bf16 舍入（8 位尾数）。这样
``dtype`` 语义对得上，又不用先做一套 uint16 存储的算术；代价是中间计算比真 bf16
更准——对拍 logits 时以「真 bf16 内核」为准，而不是以此为金标准。

约束：本模块只 import numpy，**不 import torch**；``load`` 走 ``cxrt.store``。
"""

from __future__ import annotations

import math
import types
from typing import Any, Iterable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# dtype
# ---------------------------------------------------------------------------


class dtype:
    """dtype 标记。只带一个名字与对应的 numpy dtype。

    Attributes:
        name: ``"float32"`` 之类。
        np_dtype: numpy 侧的类型。
    """

    __slots__ = ("name", "np_dtype", "_nbytes")

    def __init__(self, name: str, np_dtype: Any, nbytes: int) -> None:
        self.name = name
        self.np_dtype = np_dtype
        self._nbytes = nbytes

    def __repr__(self) -> str:                             # pragma: no cover - 展示
        return f"torch.{self.name}"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, dtype) and other.name == self.name

    def __hash__(self) -> int:
        return hash(self.name)

    @property
    def itemsize(self) -> int:
        """每个元素的字节数。"""
        return self._nbytes


float32 = dtype("float32", np.float32, 4)
float16 = dtype("float16", np.float16, 2)
bfloat16 = dtype("bfloat16", np.float32, 2)
int8 = dtype("int8", np.int8, 1)
int16 = dtype("int16", np.int16, 2)
int32 = dtype("int32", np.int32, 4)
int64 = dtype("int64", np.int64, 8)
uint8 = dtype("uint8", np.uint8, 1)
bool = dtype("bool", np.bool_, 1)
long = int64
half = float16
float_ = float32
double = dtype("float64", np.float64, 8)

#: 名字到 dtype 的映射，供 ``to("float32")`` 与配置里读到的字符串使用。
DTYPES = {d.name: d for d in (float32, float16, bfloat16, int8, int16, int32,
                              int64, uint8, bool, double)}

#: 与 ``cxrt.dtypes`` 的 dtype 码一一对应，接 ABI 时用它换算。
CODE_OF = {float32: 0, bfloat16: 1, int8: 2, int32: 3, int64: 4, uint8: 5}

#: torch 的类型推进阶：宽者胜，浮点压整数。``bf16 + fp32 -> fp32`` 是真的（不是
#: 更窄的那个赢），``fp32 + int64 -> fp32`` 也是。推理链里出现的组合都在表内，
#: 表外的组合退化成 numpy 自己的 ``result_type``。
_PROMOTE_RANK = {
    bool: 0, uint8: 1, int8: 1, int16: 2, int32: 3, int64: 4,
    float16: 5, bfloat16: 6, float32: 7, double: 8,
}


def promote_types(a: Any, b: Any) -> dtype:
    """torch 的类型推进：返回两者运算后的 dtype。

    Args:
        a: dtype 或带 ``dtype`` 的对象。
        b: 同上。

    Returns:
        推进后的 dtype。

    Raises:
        TypeError: 传进来的不是 dtype。
    """
    da = a.dtype if not isinstance(a, dtype) else a
    db = b.dtype if not isinstance(b, dtype) else b
    if not isinstance(da, dtype) or not isinstance(db, dtype):
        raise TypeError(f"promote_types got {type(a)!r}, {type(b)!r}")
    ra, rb = _PROMOTE_RANK.get(da, 99), _PROMOTE_RANK.get(db, 99)
    return da if ra >= rb else db


def _bf16_round(arr: np.ndarray) -> np.ndarray:
    """按 bf16 的 7 位显式尾数舍入（就近偶数），保持 fp32 存放。

    Args:
        arr: 任意浮点数组。

    Returns:
        同形状 fp32。
    """
    v = np.asarray(arr, dtype=np.float32)
    bits = v.view(np.uint32)
    keep = bits & np.uint32(0xFFFF0000)
    round_bit = (bits >> np.uint32(15)) & np.uint32(1)
    sticky = (bits & np.uint32(0x7FFF)) != 0
    lsb = (keep >> np.uint32(16)) & np.uint32(1)
    bump = (round_bit & (sticky | lsb)).astype(np.uint32)
    out = (keep + (bump << np.uint32(16))).view(np.float32)
    return out.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# device
# ---------------------------------------------------------------------------


class device:
    """设备标记。推理链只用到 ``cuda:0`` 与 ``cpu`` 两种。"""

    __slots__ = ("type", "index")

    def __init__(self, spec: Any = "cuda", index: int | None = None) -> None:
        if isinstance(spec, device):
            self.type, self.index = spec.type, spec.index
            return
        text = str(spec)
        if ":" in text:
            head, tail = text.split(":", 1)
            self.type = head
            self.index = int(tail) if tail.isdigit() else None
        else:
            self.type = text
            self.index = index

    def __repr__(self) -> str:                              # pragma: no cover - 展示
        return f"device(type='{self.type}', index={self.index})"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            other = device(other)
        return isinstance(other, device) and (other.type, other.index) == (self.type, self.index)

    def __hash__(self) -> int:
        return hash((self.type, self.index))


#: 进程级默认设备。推理链里由启动脚本决定，这里只给一个能被盖掉的默认值。
_default_device = device("cpu")


def set_default_device(spec: Any) -> None:
    """设置默认设备（等价于 ``torch.set_default_device``）。

    Args:
        spec: ``"cpu"`` / ``"cuda"`` / ``"cuda:0"``。
    """
    global _default_device
    _default_device = device(spec)


def get_default_device() -> device:
    """当前默认设备。"""
    return _default_device


# ---------------------------------------------------------------------------
# Tensor
# ---------------------------------------------------------------------------


def _wrap(arr: np.ndarray, dtype_: "dtype", dev: "device") -> "Tensor":
    """用已有数组造张量，**不转换 dtype、不拷贝**。

    ``Tensor(...)`` 的构造器会按目标 dtype 重排一份数组（bf16 要舍入），这样
    ``reshape``/``transpose``/切片这些纯元数据操作会悄悄变成拷贝——最直接的后果
    是 ``x.reshape(-1, n)`` 与 ``x`` 不再共享缓冲区，而 ``ops/norm.py`` 正是靠
    ``data_ptr`` 相等来判断「这个目标是不是稠密视图」。所以元数据路径走这一条。
    """
    out = Tensor.__new__(Tensor)
    out.data = arr
    out.dtype = dtype_
    out.device = dev
    return out


class Tensor:
    """numpy 支撑的张量，语义对齐推理链用到的那部分 torch。

    Attributes:
        data: ``np.ndarray``。
        dtype: :class:`dtype`。
        device: :class:`device`。
    """

    #: 让 ``isinstance(x, Tensor)`` 与 torch 的鸭子类型兼容。
    __array_priority__ = 1000

    def __init__(self, data: Any, dtype_: dtype | None = None,
                 dev: device | None = None) -> None:
        arr = data if isinstance(data, np.ndarray) else np.asarray(data)
        if dtype_ is not None:
            arr = _cast_array(arr, dtype_)
        self.data = arr
        self.dtype = _dtype_of_array(arr, dtype_)
        self.device = dev or _default_device

    # 构造与属性 ---------------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        """形状。"""
        return tuple(int(v) for v in self.data.shape)

    @property
    def ndim(self) -> int:
        """维数。"""
        return int(self.data.ndim)

    @property
    def T(self) -> Tensor:
        """最后两维转置。"""
        return self.permute(tuple(range(self.ndim - 2)) + (self.ndim - 1, self.ndim - 2)) \
            if self.ndim >= 2 else self.transpose(0, 0)

    def t(self) -> Tensor:
        """torch 的 ``.t()``：2-D 转置（1-D 原样返回）。"""
        if self.ndim < 2:
            return self
        return self.transpose(self.ndim - 2, self.ndim - 1)

    def dim(self) -> int:
        """维数（torch 的方法形式）。"""
        return self.ndim

    def size(self, dim: int | None = None) -> Any:
        """形状，或某一维的长度。"""
        if dim is None:
            return self.shape
        return self.shape[dim]

    def numel(self) -> int:
        """元素个数。"""
        return int(self.data.size)

    def element_size(self) -> int:
        """每个元素的字节数（按 dtype 报，不按存放类型）。"""
        return self.dtype.itemsize

    def item(self) -> Any:
        """单元素张量转 Python 标量。"""
        return self.data.reshape(-1)[0].item()

    def __int__(self) -> int:
        """单元素张量转 ``int``：``int(t.argmax())`` 是驱动里最常见的写法。"""
        return int(self.item())

    def __float__(self) -> float:
        """单元素张量转 ``float``。"""
        return float(self.item())

    def __index__(self) -> int:
        """让单元素张量能直接当下标用（切片、``range`` 都走这条）。"""
        return int(self.item())

    def tolist(self) -> Any:
        """转 Python 列表。"""
        return self.data.tolist()

    def numpy(self) -> np.ndarray:
        """转 numpy（bf16 给的是 fp32 视图，值已按 bf16 舍入）。"""
        return self.data

    def __array__(self, dtype_: Any = None) -> np.ndarray:   # pragma: no cover - 互操作
        return self.data if dtype_ is None else self.data.astype(dtype_)

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __repr__(self) -> str:                              # pragma: no cover - 展示
        return f"Tensor({self.data!r}, dtype={self.dtype.name})"

    # dtype / device 迁移 -------------------------------------------------

    def to(self, target: Any = None, *args: Any, **kwargs: Any) -> Tensor:
        """换 dtype 或设备；两者都给时按 torch 的位置参数语义处理。

        Args:
            target: ``dtype`` / ``device`` / 字符串，或 ``device=`` 关键字。
            *args: 第二个位置参数（dtype 或 device）。
            **kwargs: ``dtype=`` / ``device=`` / ``non_blocking=`` 等。

        Returns:
            新的 :class:`Tensor`（同值）。
        """
        want_dtype: dtype | None = None
        want_device: Any = None
        for item in (target, *args):
            if item is None:
                continue
            if isinstance(item, dtype):
                want_dtype = item
            elif isinstance(item, device):
                want_device = item
            elif isinstance(item, str) and item in DTYPES:
                want_dtype = DTYPES[item]
            elif isinstance(item, str):
                want_device = device(item)
        want_dtype = kwargs.get("dtype", want_dtype)
        want_device = kwargs.get("device", want_device)
        out = self
        if want_dtype is not None and want_dtype != self.dtype:
            out = Tensor(_cast_array(self.data, want_dtype), want_dtype, self.device)
        if want_device is not None:
            out = Tensor(out.data, out.dtype, device(want_device) if not isinstance(
                want_device, device) else want_device)
        return out

    def type_as(self, other: Tensor) -> Tensor:
        """对齐另一个张量的 dtype。"""
        return self.to(other.dtype)

    def float(self) -> Tensor:
        """转 fp32。"""
        return self.to(float32)

    def half(self) -> Tensor:
        """转 fp16。"""
        return self.to(float16)

    def bfloat16(self) -> Tensor:
        """转 bf16。"""
        return self.to(bfloat16)

    def int(self) -> Tensor:
        """转 int32。"""
        return self.to(int32)

    def long(self) -> Tensor:
        """转 int64。"""
        return self.to(int64)

    def byte(self) -> Tensor:
        """转 uint8。"""
        return self.to(uint8)

    def bool(self) -> Tensor:
        """转 bool。"""
        return self.to(bool)

    def cpu(self) -> Tensor:
        """放到 CPU。"""
        return self.to(device("cpu"))

    def cuda(self, index: int | None = None) -> Tensor:
        """放到卡上。"""
        return self.to(device("cuda", index))

    def pin_memory(self) -> Tensor:
        """无操作，保留接口。"""
        return self

    def detach(self) -> Tensor:
        """去掉梯度语义（本壳没有梯度，直接返回自身）。"""
        return self

    def clone(self) -> Tensor:
        """深拷贝。"""
        return Tensor(self.data.copy(), self.dtype, self.device)

    def contiguous(self, *args: Any, **kwargs: Any) -> Tensor:
        """保证 C 连续。"""
        if self.data.flags["C_CONTIGUOUS"]:
            return self
        return Tensor(np.ascontiguousarray(self.data), self.dtype, self.device)

    def copy_(self, other: Tensor) -> Tensor:
        """原地写入，返回自身（torch 语义）。"""
        self.data[...] = _as_array(other)
        return self

    def fill_(self, value: Any) -> Tensor:
        """原地填值。"""
        self.data[...] = value
        return self

    def zero_(self) -> Tensor:
        """原地清零。"""
        return self.fill_(0)

    def add_(self, other: Any, alpha: Any = 1) -> Tensor:
        """原地加。"""
        self.data += alpha * _as_array(other)
        return self

    def mul_(self, other: Any) -> Tensor:
        """原地乘。"""
        self.data *= _as_array(other)
        return self

    # 形状 -----------------------------------------------------------------

    def reshape(self, *shape: Any) -> Tensor:
        """改形状（可拷贝）。"""
        return _wrap(self.data.reshape(_shape_tuple(shape)), self.dtype, self.device)

    def view(self, *shape: Any) -> Tensor:
        """改形状（要求连续）。"""
        return _wrap(self.data.reshape(_shape_tuple(shape)), self.dtype, self.device)

    def view_as(self, other: Tensor) -> Tensor:
        """按另一个张量的形状 reshape。"""
        return self.reshape(other.shape)

    def flatten(self, start: int = 0, end: int = -1) -> Tensor:
        """展平一段维度。"""
        return self.reshape(self.shape[:start] + (-1,) + self.shape[end + 1:])

    def squeeze(self, dim: int | None = None) -> Tensor:
        """去掉长度 1 的维度。"""
        data = np.squeeze(self.data, axis=dim)
        return _wrap(data, self.dtype, self.device)

    def unsqueeze(self, dim: int) -> Tensor:
        """插入长度 1 的维度。"""
        return _wrap(np.expand_dims(self.data, dim), self.dtype, self.device)

    def transpose(self, dim0: int, dim1: int) -> Tensor:
        """交换两维。"""
        return _wrap(np.swapaxes(self.data, dim0, dim1), self.dtype, self.device)

    def permute(self, *dims: Any) -> Tensor:
        """重排维度。"""
        axes = _shape_tuple(dims)
        return _wrap(np.transpose(self.data, axes), self.dtype, self.device)

    def stride(self, dim: int | None = None) -> int | tuple[int, ...]:
        """步长，**按元素**而不是按字节报（torch 的口径）。

        模型层用它判「最后一维连不连续」，例如 ``x.stride(-1) != 1``。numpy 的
        ``strides`` 是字节，所以这里要除以元素宽度；按字节报会让每个 fp32 张量
        都显得不连续。
        """
        strides = tuple(int(s) // (self.data.dtype.itemsize or 1)
                         for s in self.data.strides)
        if dim is None:
            return strides
        return strides[dim]

    def expand(self, *shape: Any) -> Tensor:
        """广播到指定形状（长度 1 的维度可改大）。"""
        target = _shape_tuple(shape)
        data = np.broadcast_to(self.data, target)
        return Tensor(np.array(data, copy=False), self.dtype, self.device)

    def repeat(self, *shape: Any) -> Tensor:
        """按倍数重复。"""
        return Tensor(np.tile(self.data, _shape_tuple(shape)), self.dtype, self.device)

    def narrow(self, dim: int, start: int, length: int) -> Tensor:
        """沿某维取一段。"""
        sl = [slice(None)] * self.ndim
        sl[dim] = slice(start, start + length)
        return _wrap(self.data[tuple(sl)], self.dtype, self.device)

    def split(self, sizes: int | Sequence[int], dim: int = 0) -> tuple[Tensor, ...]:
        """沿某维切分。"""
        return split(self, sizes, dim)

    def chunk(self, parts: int, dim: int = 0) -> tuple[Tensor, ...]:
        """沿某维均分。"""
        return chunk(self, parts, dim)

    def index_select(self, dim: int, index: Tensor) -> Tensor:
        """按索引表取值。"""
        idx = _as_array(index).astype(np.int64)
        return Tensor(np.take(self.data, idx, axis=dim), self.dtype, self.device)

    def gather(self, dim: int, index: Tensor) -> Tensor:
        """沿某维按索引取值（同形状索引）。"""
        idx = _as_array(index).astype(np.int64)
        return Tensor(np.take_along_axis(self.data, idx, axis=dim), self.dtype, self.device)

    def masked_fill(self, mask: Tensor, value: Any) -> Tensor:
        """掩码位置填值，掩码按 numpy 规则广播（``[L,L]`` 可以盖到 ``[H,nc,L,L]``）。"""
        m = _as_array(mask).astype(np.bool_)
        out = np.where(m, _as_array(value), self.data)
        return Tensor(out, self.dtype, self.device)

    def masked_select(self, mask: Tensor) -> Tensor:
        """展平后取掩码为真的元素。"""
        return Tensor(self.data[_as_array(mask).astype(np.bool_)], self.dtype, self.device)

    def where(self, condition: Tensor, other: Tensor) -> Tensor:
        """``condition ? self : other``（torch 的三参形式）。"""
        return where(condition, self, other)

    def __getitem__(self, key: Any) -> Tensor:
        if isinstance(key, Tensor):
            key = _as_array(key)
        if isinstance(key, tuple):
            key = tuple(_as_array(k) if isinstance(k, Tensor) else k for k in key)
        return _wrap(self.data[key], self.dtype, self.device)

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(key, Tensor):
            key = _as_array(key)
        if isinstance(key, tuple):
            key = tuple(_as_array(k) if isinstance(k, Tensor) else k for k in key)
        self.data[key] = _as_array(value)

    # 算术 -----------------------------------------------------------------

    def _binary(self, other: Any, op: Any) -> Tensor:
        """按「任一 bf16 则结果 bf16」的规则算二元运算。"""
        rhs = other.data if isinstance(other, Tensor) else other
        left = self.data
        out = op(left, rhs)
        out_dtype = _promote(self.dtype, other.dtype if isinstance(other, Tensor) else None)
        return Tensor(_cast_array(np.asarray(out), out_dtype), out_dtype, self.device)

    def __add__(self, other: Any) -> Tensor:
        return self._binary(other, np.add)

    def __radd__(self, other: Any) -> Tensor:
        return self._binary(other, lambda a, b: np.add(b, a))

    def __sub__(self, other: Any) -> Tensor:
        return self._binary(other, np.subtract)

    def __rsub__(self, other: Any) -> Tensor:
        return self._binary(other, lambda a, b: np.subtract(b, a))

    def __mul__(self, other: Any) -> Tensor:
        return self._binary(other, np.multiply)

    def __rmul__(self, other: Any) -> Tensor:
        return self._binary(other, lambda a, b: np.multiply(b, a))

    def __truediv__(self, other: Any) -> Tensor:
        return self._binary(other, np.true_divide)

    def __rtruediv__(self, other: Any) -> Tensor:
        return self._binary(other, lambda a, b: np.true_divide(b, a))

    def __pow__(self, exponent: Any) -> Tensor:
        return self._binary(exponent, np.power)

    def __rpow__(self, base: Any) -> Tensor:
        """``theta ** tensor``：rope 的 ``theta`` 是 Python 标量，这里兜住反向幂。"""
        return self._binary(base, lambda a, b: np.power(b, a))

    def __floordiv__(self, other: Any) -> Tensor:
        return self._binary(other, np.floor_divide)

    def __neg__(self) -> Tensor:
        return Tensor(-self.data, self.dtype, self.device)

    def __abs__(self) -> Tensor:
        return Tensor(np.abs(self.data), self.dtype, self.device)

    def __matmul__(self, other: Tensor) -> Tensor:
        return matmul(self, other)

    def __lt__(self, other: Any) -> Tensor:
        return Tensor(np.less(self.data, _as_array(other)), bool, self.device)

    def __le__(self, other: Any) -> Tensor:
        return Tensor(np.less_equal(self.data, _as_array(other)), bool, self.device)

    def __gt__(self, other: Any) -> Tensor:
        return Tensor(np.greater(self.data, _as_array(other)), bool, self.device)

    def __ge__(self, other: Any) -> Tensor:
        return Tensor(np.greater_equal(self.data, _as_array(other)), bool, self.device)

    def __eq__(self, other: Any) -> Tensor:                 # type: ignore[override]
        return Tensor(np.equal(self.data, _as_array(other)), bool, self.device)

    def __ne__(self, other: Any) -> Tensor:                 # type: ignore[override]
        return Tensor(np.not_equal(self.data, _as_array(other)), bool, self.device)

    def __hash__(self) -> int:                              # pragma: no cover
        return id(self)

    # 归约与逐元素 ---------------------------------------------------------

    def sum(self, dim: Any = None, keepdim: bool = False) -> Tensor:
        """求和。"""
        return Tensor(np.sum(self.data, axis=dim, keepdims=keepdim), self.dtype, self.device)

    def mean(self, dim: Any = None, keepdim: bool = False) -> Tensor:
        """平均值。"""
        return Tensor(np.mean(self.data, axis=dim, keepdims=keepdim), self.dtype, self.device)

    def max(self, dim: Any = None) -> Any:
        """最大值；给了 ``dim`` 时返回 ``(values, indices)``。"""
        if dim is None:
            return Tensor(np.max(self.data), self.dtype, self.device)
        vals = np.max(self.data, axis=dim)
        idx = np.argmax(self.data, axis=dim)
        return (Tensor(vals, self.dtype, self.device),
                Tensor(idx.astype(np.int64), int64, self.device))

    def min(self, dim: Any = None) -> Any:
        """最小值；给了 ``dim`` 时返回 ``(values, indices)``。"""
        if dim is None:
            return Tensor(np.min(self.data), self.dtype, self.device)
        vals = np.min(self.data, axis=dim)
        idx = np.argmin(self.data, axis=dim)
        return (Tensor(vals, self.dtype, self.device),
                Tensor(idx.astype(np.int64), int64, self.device))

    def argmax(self, dim: int | None = None) -> Tensor:
        """最大值下标。"""
        return Tensor(np.argmax(self.data, axis=dim).astype(np.int64), int64, self.device)

    def argmin(self, dim: int | None = None) -> Tensor:
        """最小值下标。"""
        return Tensor(np.argmin(self.data, axis=dim).astype(np.int64), int64, self.device)

    def cumsum(self, dim: int) -> Tensor:
        """累加。"""
        return Tensor(np.cumsum(self.data, axis=dim), self.dtype, self.device)

    def softmax(self, dim: int = -1) -> Tensor:
        """数值稳定的 softmax。"""
        return softmax(self, dim)

    def sigmoid(self) -> Tensor:
        """logistic。"""
        return sigmoid(self)

    def exp(self) -> Tensor:
        """指数。"""
        return Tensor(np.exp(self.data), self.dtype, self.device)

    def sqrt(self) -> Tensor:
        """开方。"""
        return Tensor(np.sqrt(self.data), self.dtype, self.device)

    def rsqrt(self) -> Tensor:
        """平方根的倒数。"""
        return rsqrt(self)

    def tanh(self) -> Tensor:
        """双曲正切。"""
        return Tensor(np.tanh(self.data), self.dtype, self.device)

    def cos(self) -> Tensor:
        """余弦（RoPE 的 cos 表就是它算的）。"""
        return Tensor(np.cos(self.data), self.dtype, self.device)

    def sin(self) -> Tensor:
        """正弦。"""
        return Tensor(np.sin(self.data), self.dtype, self.device)

    def abs(self) -> Tensor:
        """绝对值。"""
        return Tensor(np.abs(self.data), self.dtype, self.device)

    def clamp(self, min: Any = None, max: Any = None) -> Tensor:
        """裁剪区间。"""
        return Tensor(np.clip(self.data, min, max), self.dtype, self.device)

    def pow(self, exponent: Any) -> Tensor:
        """幂。"""
        return Tensor(np.power(self.data, _as_array(exponent)), self.dtype, self.device)

    # 补齐的方法形式：模块级函数已经有，这里只是让 ``x.xxx()`` 也能用 ----------

    def log(self) -> Tensor:
        """自然对数。"""
        return Tensor(np.log(self.data), self.dtype, self.device)

    def round(self) -> Tensor:
        """四舍五入（torch 是就近偶数，numpy 也是）。"""
        return Tensor(np.round(self.data), self.dtype, self.device)

    def floor(self) -> Tensor:
        """向下取整。"""
        return Tensor(np.floor(self.data), self.dtype, self.device)

    def ceil(self) -> Tensor:
        """向上取整。"""
        return Tensor(np.ceil(self.data), self.dtype, self.device)

    def neg(self) -> Tensor:
        """取负。"""
        return Tensor(-self.data, self.dtype, self.device)

    def amax(self, dim: int | None = None, keepdim: bool = False) -> Tensor:
        """最大值。"""
        axis = None if dim is None else dim
        return Tensor(np.amax(self.data, axis=axis, keepdims=keepdim),
                      self.dtype, self.device)

    def amin(self, dim: int | None = None, keepdim: bool = False) -> Tensor:
        """最小值。"""
        axis = None if dim is None else dim
        return Tensor(np.amin(self.data, axis=axis, keepdims=keepdim),
                      self.dtype, self.device)

    def flip(self, dims: Any) -> Tensor:
        """沿给定维翻转。"""
        axes = (dims,) if isinstance(dims, int) else tuple(dims)
        return Tensor(np.flip(self.data, axis=axes), self.dtype, self.device)

    def sort(self, dim: int = -1, descending: bool = False
             ) -> tuple[Tensor, Tensor]:
        """沿某维排序，返回 ``(值, 下标)``。

        下标就是 ``argsort`` 的结果：``indices[i]`` 是排在第 ``i`` 位的原始位置，
        与 torch 的约定一致。
        """
        data = self.data
        axis = dim % data.ndim
        order = np.argsort(-data if descending else data, axis=axis, kind="stable")
        return (Tensor(np.take_along_axis(data, order, axis=axis), self.dtype, self.device),
                Tensor(order.astype(np.int64), int64, self.device))

    def topk(self, k: int, dim: int = -1, largest: bool = True,
             sorted: bool = True) -> tuple[Tensor, Tensor]:
        """方法形式的 :func:`topk`。"""
        return topk(self, k, dim, largest, sorted)

    def tril(self, diagonal: int = 0) -> Tensor:
        """下三角（其余清零）。"""
        return tril(self, diagonal)

    def triu(self, diagonal: int = 0) -> Tensor:
        """上三角（其余清零）。"""
        return triu(self, diagonal)

    def sub_(self, other: Any, alpha: Any = 1) -> Tensor:
        """就地减法。"""
        self.data[...] = _as_array(self) - _as_array(other) * _as_array(alpha)
        return self

    def div_(self, other: Any) -> Tensor:
        """就地除法。"""
        self.data[...] = _as_array(self) / _as_array(other)
        return self

    def new_ones(self, *shape: Any, dtype: Any = None, device: Any = None) -> Tensor:
        """同样设备上的全 1 张量。"""
        return ones(*shape, dtype=dtype or self.dtype, device=device or self.device)

    def new_tensor(self, value: Any, dtype: Any = None, device: Any = None) -> Tensor:
        """同设备上的新张量。"""
        return Tensor(np.array(value), dtype or self.dtype, device or self.device)

    def double(self) -> Tensor:
        """转 fp64。"""
        return self.to(double)

    def index_select(self, dim: int, index: Tensor) -> Tensor:
        """按下标沿某维挑选。"""
        return Tensor(np.take(self.data, index.data.astype(np.int64), axis=dim),
                      self.dtype, self.device)

    def unfold(self, dim: int, size: int, step: int) -> Tensor:
        """滑动窗口，窗口维追加在最后（torch 的 ``unfold`` 口径）。

        卷积的三出口都靠它把 ``[K-1+T, C]`` 展开成 ``[T, C, K]``；numpy 的
        ``sliding_window_view`` 也是把窗口放在最后一维，所以形状天然对齐，
        只是要按 ``step`` 抽稀。
        """
        data = self.data
        axis = dim % data.ndim
        view = np.lib.stride_tricks.sliding_window_view(data, int(size), axis=axis)
        sl = [slice(None)] * view.ndim
        sl[axis] = slice(None, None, int(step))
        return Tensor(np.array(view[tuple(sl)]), self.dtype, self.device)

    def repeat_interleave(self, repeats: int, dim: int = 0) -> Tensor:
        """把每个元素重复 ``repeats`` 次（``[h, ...] -> [h*rep, ...]``）。"""
        return Tensor(np.repeat(self.data, int(repeats), axis=dim % self.ndim),
                      self.dtype, self.device)

    # 剩下的方法形式：矩阵/就地/查询，一次补齐，避免每次作业只为缺一个名字重跑 ----

    def is_contiguous(self) -> bool:
        """是否行主序连续。"""
        return self.data.flags["C_CONTIGUOUS"]

    def mm(self, other: Tensor) -> Tensor:
        """2-D 矩阵乘。"""
        return matmul(self, other)

    def bmm(self, other: Tensor) -> Tensor:
        """批量矩阵乘。"""
        return matmul(self, other)

    def einsum(self, equation: str, *operands: Tensor) -> Tensor:
        """方法形式的 :func:`einsum`。"""
        return einsum(equation, self, *operands)

    def unbind(self, dim: int = 0) -> tuple[Tensor, ...]:
        """沿某维拆成一串张量。"""
        moved = np.moveaxis(self.data, dim % self.ndim, 0)
        return tuple(Tensor(moved[i], self.dtype, self.device)
                     for i in range(moved.shape[0]))

    def diagonal(self, offset: int = 0, dim1: int = 0, dim2: int = 1) -> Tensor:
        """取对角，返回**可写视图**。

        numpy 的 ``np.diagonal`` 给的是只读拷贝，而 ``ops/delta.py`` 会写
        ``A.diagonal().add_(...)``——它依赖 torch 的「对角是视图」这一语义。这里用
        ``as_strided`` 造同一条对角线视图，写完就落回原张量。
        """
        a = np.moveaxis(self.data, (dim1 % self.ndim, dim2 % self.ndim), (-2, -1))
        row, col = (int(offset), 0) if offset >= 0 else (0, -int(offset))
        rows, cols = a.shape[-2], a.shape[-1]
        length = rows - row if rows - row < cols - col else cols - col
        if length <= 0:
            return Tensor(np.zeros(a.shape[:-2] + (0,), dtype=np.float32),
                          self.dtype, self.device)
        base = a[..., row:, col:]
        strides = base.strides[:-2] + (base.strides[-2] + base.strides[-1],)
        view = np.lib.stride_tricks.as_strided(
            base, shape=a.shape[:-2] + (length,), strides=strides, writeable=True)
        return _wrap(view, self.dtype, self.device)

    def prod(self, dim: int | None = None, keepdim: bool = False) -> Tensor:
        """连乘。"""
        axis = None if dim is None else dim
        return Tensor(np.prod(self.data, axis=axis, keepdims=keepdim),
                      self.dtype, self.device)

    def maximum(self, other: Any) -> Tensor:
        """逐元素取大。"""
        return Tensor(np.maximum(self.data, _as_array(other)), self.dtype, self.device)

    def minimum(self, other: Any) -> Tensor:
        """逐元素取小。"""
        return Tensor(np.minimum(self.data, _as_array(other)), self.dtype, self.device)

    def nonzero(self, **kwargs: Any) -> Tensor:
        """非零元素下标，``[n, ndim]`` int64。"""
        return Tensor(np.argwhere(self.data).astype(np.int64), int64, self.device)

    def is_floating_point(self) -> bool:
        """是否浮点。"""
        return self.dtype in (float32, float16, bfloat16, double)

    def data_ptr(self) -> int:
        """缓冲区地址（无 torch 环境里没有意义，只为兼容签名）。"""
        return int(self.data.ctypes.data)

    def addcmul_(self, tensor1: Any, tensor2: Any, value: Any = 1) -> Tensor:
        """就地 ``self += value * t1 * t2``。"""
        self.data[...] = (self.data + _as_array(value) * _as_array(tensor1)
                          * _as_array(tensor2))
        return self

    def index_add_(self, dim: int, index: Tensor, source: Tensor) -> Tensor:
        """就地按 ``index`` 累加。"""
        idx = index.data.astype(np.int64)
        target = np.moveaxis(self.data, dim % self.ndim, 0)
        src = np.moveaxis(source.data, dim % source.ndim, 0)
        for row in range(idx.shape[0]):
            target[idx[row]] += src[row]
        return self

    def index_copy_(self, dim: int, index: Tensor, source: Tensor) -> Tensor:
        """就地按 ``index`` 覆盖。"""
        idx = index.data.astype(np.int64)
        target = np.moveaxis(self.data, dim % self.ndim, 0)
        src = np.moveaxis(source.data, dim % source.ndim, 0)
        for row in range(idx.shape[0]):
            target[idx[row]] = src[row]
        return self

    def masked_fill_(self, mask: Any, value: Any) -> Tensor:
        """就地把 ``mask`` 处填成 ``value``（掩码同样按 numpy 规则广播）。"""
        raw = mask.data if isinstance(mask, Tensor) else np.asarray(mask)
        self.data[...] = np.where(np.asarray(raw, dtype=np.bool_), _as_array(value),
                                  self.data)
        return self

    def clamp_(self, min: Any = None, max: Any = None) -> Tensor:
        """就地裁剪。"""
        self.data[...] = np.clip(self.data, min, max)
        return self

    def clamp_min(self, min: Any) -> Tensor:
        """下限裁剪（函数形式）。"""
        return Tensor(np.maximum(self.data, _as_array(min)), self.dtype, self.device)

    def clamp_min_(self, min: Any) -> Tensor:
        """就地下限裁剪。"""
        self.data[...] = np.maximum(self.data, _as_array(min))
        return self

    def neg_(self) -> Tensor:
        """就取负。"""
        self.data[...] = -self.data
        return self

    # 便捷构造 -------------------------------------------------------------

    def new_empty(self, shape: Sequence[int], dtype_: dtype | None = None) -> Tensor:
        """新分配（未初始化）。"""
        return empty(shape, dtype=dtype_ or self.dtype, device=self.device)

    def new_zeros(self, shape: Sequence[int], dtype_: dtype | None = None) -> Tensor:
        """新分配并清零。"""
        return zeros(shape, dtype=dtype_ or self.dtype, device=self.device)

    def new_full(self, shape: Sequence[int], value: Any, dtype_: dtype | None = None) -> Tensor:
        """新分配并填值。"""
        return full(shape, value, dtype=dtype_ or self.dtype, device=self.device)

    def to_tensor_list(self) -> list[Tensor]:
        """沿第 0 维拆成若干张量（内部辅助）。"""
        return [self[i] for i in range(self.shape[0])]


# ---------------------------------------------------------------------------
# 容器与常量
# ---------------------------------------------------------------------------


def _as_array(value: Any) -> np.ndarray:
    """把 ``Tensor`` / 标量 / 序列统一成 numpy 数组。

    Args:
        value: 任意输入。

    Returns:
        numpy 数组（``Tensor`` 取底层数据）。
    """
    if isinstance(value, Tensor):
        return value.data
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def _shape_tuple(shape: Any) -> tuple[int, ...]:
    """把 ``reshape(2, 3)`` / ``reshape((2, 3))`` / ``reshape([2, 3])`` 归一。

    Args:
        shape: 位置参数元组或其唯一元素。

    Returns:
        维度元组。
    """
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        return tuple(int(v) for v in shape[0])
    return tuple(int(v) for v in shape)


def _dtype_of_array(arr: np.ndarray, fallback: dtype | None) -> dtype:
    """由 numpy dtype 反推 :class:`dtype`。

    Args:
        arr: 数组。
        fallback: 优先采用的外部声明。

    Returns:
        dtype 标记。
    """
    if fallback is not None:
        return fallback
    mapping = {np.dtype(np.float32): float32, np.dtype(np.float16): float16,
               np.dtype(np.int8): int8, np.dtype(np.int16): int16,
               np.dtype(np.int32): int32, np.dtype(np.int64): int64,
               np.dtype(np.uint8): uint8, np.dtype(np.bool_): bool,
               np.dtype(np.float64): double}
    return mapping.get(arr.dtype, float32)


def _cast_array(arr: np.ndarray, target: dtype) -> np.ndarray:
    """按 :class:`dtype` 转换数组，bf16 走 8 位尾数舍入。

    Args:
        arr: 输入数组。
        target: 目标 dtype。

    Returns:
        转换后的数组。
    """
    if target == bfloat16:
        return _bf16_round(np.asarray(arr, dtype=np.float32))
    if target == bool:
        return np.asarray(arr, dtype=np.bool_)
    return np.asarray(arr, dtype=target.np_dtype)


def _promote(left: dtype, right: dtype | None) -> dtype:
    """二元运算的结果 dtype：浮点优先、宽者优先，bf16 与 fp32 混合给 bf16。

    Args:
        left: 左侧 dtype。
        right: 右侧 dtype，``None`` 表示标量。

    Returns:
        结果 dtype。
    """
    if right is None:
        return left
    if left == right:
        return left
    order = [bfloat16, float16, float32, double]
    if left in order and right in order:
        if bfloat16 in (left, right) and float32 not in (left, right):
            return bfloat16
        return left if order.index(left) > order.index(right) else right
    if left in order:
        return left
    if right in order:
        return right
    return left


def _dev(dev: Any) -> device:
    """归一 ``device=`` 参数。"""
    if dev is None:
        return _default_device
    return dev if isinstance(dev, device) else device(dev)


def _norm_dtype(dtype_: Any, default: dtype = float32) -> dtype:
    """归一 ``dtype=`` 参数，允许传字符串。"""
    if dtype_ is None:
        return default
    if isinstance(dtype_, dtype):
        return dtype_
    if isinstance(dtype_, str):
        return DTYPES[dtype_]
    raise TypeError(f"unsupported dtype {dtype_!r}")


def zeros(*shape: Any, dtype: Any = None, device: Any = None, **kwargs: Any) -> Tensor:
    """全零张量。"""
    d = _norm_dtype(dtype, float32)
    return Tensor(np.zeros(_shape_tuple(shape), dtype=_np_kind(d)), d, _dev(device))


def ones(*shape: Any, dtype: Any = None, device: Any = None, **kwargs: Any) -> Tensor:
    """全一张量。"""
    d = _norm_dtype(dtype, float32)
    return Tensor(np.ones(_shape_tuple(shape), dtype=_np_kind(d)), d, _dev(device))


def empty(*shape: Any, dtype: Any = None, device: Any = None, **kwargs: Any) -> Tensor:
    """未初始化张量（numpy 下就是 zeros，避免出现脏值）。"""
    d = _norm_dtype(dtype, float32)
    return Tensor(np.zeros(_shape_tuple(shape), dtype=_np_kind(d)), d, _dev(device))


def full(shape: Any, value: Any, dtype: Any = None, device: Any = None,
         **kwargs: Any) -> Tensor:
    """填值张量。"""
    d = _norm_dtype(dtype, float32)
    shape_t = _shape_tuple((shape,)) if not isinstance(shape, (tuple, list, Tensor)) \
        else _shape_tuple((shape,))
    return Tensor(np.full(shape_t, value, dtype=_np_kind(d)), d, _dev(device))


def _np_kind(d: dtype) -> Any:
    """按 dtype 给出 numpy 的存放类型（bf16 用 fp32 存放）。"""
    return np.float32 if d == bfloat16 else d.np_dtype


def zeros_like(other: Tensor, dtype: Any = None, device: Any = None, **kwargs: Any) -> Tensor:
    """同形状全零。"""
    return zeros(other.shape, dtype=_norm_dtype(dtype, other.dtype), device=device or other.device)


def ones_like(other: Tensor, dtype: Any = None, device: Any = None, **kwargs: Any) -> Tensor:
    """同形状全一。"""
    return ones(other.shape, dtype=_norm_dtype(dtype, other.dtype), device=device or other.device)


def empty_like(other: Tensor, dtype: Any = None, device: Any = None, **kwargs: Any) -> Tensor:
    """同形状未初始化。"""
    return empty(other.shape, dtype=_norm_dtype(dtype, other.dtype), device=device or other.device)


def full_like(other: Tensor, value: Any, dtype: Any = None, device: Any = None,
              **kwargs: Any) -> Tensor:
    """同形状填值。"""
    return full(other.shape, value, dtype=_norm_dtype(dtype, other.dtype),
                device=device or other.device)


def arange(start: Any, end: Any = None, step: Any = 1, dtype: Any = None,
           device: Any = None, **kwargs: Any) -> Tensor:
    """等差数列。"""
    d = _norm_dtype(dtype, int64)
    if end is None:
        start, end = 0, start
    out = np.arange(start, end, step, dtype=_np_kind(d))
    return Tensor(out, d, _dev(device))


def linspace(start: Any, end: Any, steps: int, dtype: Any = None,
             device: Any = None, **kwargs: Any) -> Tensor:
    """等分点。"""
    d = _norm_dtype(dtype, float32)
    return Tensor(np.linspace(start, end, steps, dtype=_np_kind(d)), d, _dev(device))


def tensor(data: Any, dtype: Any = None, device: Any = None, **kwargs: Any) -> Tensor:
    """由数据建张量（拷贝）。"""
    arr = _as_array(data).copy() if not isinstance(data, Tensor) else data.data.copy()
    d = _norm_dtype(dtype, _dtype_of_array(arr, None))
    return Tensor(_cast_array(arr, d), d, _dev(device))


def as_tensor(data: Any, dtype: Any = None, device: Any = None, **kwargs: Any) -> Tensor:
    """由数据建张量（尽量不拷贝）。"""
    if isinstance(data, Tensor):
        return data.to(dtype) if dtype is not None else data
    arr = _as_array(data)
    d = _norm_dtype(dtype, _dtype_of_array(arr, None))
    return Tensor(_cast_array(arr, d), d, _dev(device))


def from_numpy(arr: np.ndarray) -> Tensor:
    """包装 numpy 数组（不拷贝）。"""
    return Tensor(arr, _dtype_of_array(arr, None))


def frombuffer(buf: Any, dtype: Any = None, count: int = -1, offset: int = 0) -> Tensor:
    """由字节缓冲建一维张量。"""
    d = _norm_dtype(dtype, float32)
    raw = np.frombuffer(buf, dtype=_np_kind(d), count=count, offset=offset)
    return Tensor(np.array(raw, copy=True), d)


def is_tensor(value: Any) -> bool:
    """是否 :class:`Tensor`。"""
    return isinstance(value, Tensor)


def cat(items: Iterable[Tensor], dim: int = 0) -> Tensor:
    """拼接。"""
    seq = list(items)
    first = seq[0]
    out = np.concatenate([_as_array(x) for x in seq], axis=dim)
    return Tensor(out, first.dtype, first.device)


concat = cat


def stack(items: Iterable[Tensor], dim: int = 0) -> Tensor:
    """堆叠。"""
    seq = list(items)
    first = seq[0]
    return Tensor(np.stack([_as_array(x) for x in seq], axis=dim), first.dtype, first.device)


def split(value: Tensor, sizes: int | Sequence[int], dim: int = 0) -> tuple[Tensor, ...]:
    """按份数或按每份长度切分。"""
    total = value.shape[dim]
    if isinstance(sizes, int):
        parts = [sizes] * (total // sizes)
        if total % sizes:
            parts.append(total % sizes)
    else:
        parts = list(sizes)
    out: list[Tensor] = []
    offset = 0
    for size in parts:
        sl = [slice(None)] * value.ndim
        sl[dim] = slice(offset, offset + int(size))
        out.append(_wrap(value.data[tuple(sl)], value.dtype, value.device))
        offset += int(size)
    return tuple(out)


def chunk(value: Tensor, parts: int, dim: int = 0) -> tuple[Tensor, ...]:
    """尽量均分成 ``parts`` 份。"""
    total = value.shape[dim]
    size = math.ceil(total / parts)
    return split(value, size, dim)


def where(condition: Any, a: Any = None, b: Any = None) -> Tensor:
    """三参选择，或单参 ``nonzero``。"""
    if a is None and b is None:
        return nonzero(condition)
    cond = _as_array(condition).astype(np.bool_)
    left = _as_array(a)
    right = _as_array(b)
    ref = a if isinstance(a, Tensor) else (b if isinstance(b, Tensor) else None)
    out = np.where(cond, left, right)
    d = ref.dtype if ref is not None else _dtype_of_array(np.asarray(out), None)
    return Tensor(_cast_array(np.asarray(out), d), d, ref.device if ref is not None else _default_device)


def nonzero(value: Any, as_tuple: bool = False) -> Any:
    """非零下标。"""
    idx = np.nonzero(_as_array(value))
    if as_tuple:
        return tuple(Tensor(i.astype(np.int64), int64) for i in idx)
    if not idx:
        return zeros((0, 0), dtype=int64)
    return Tensor(np.stack(idx, axis=1).astype(np.int64), int64)


def cumsum(value: Tensor, dim: int) -> Tensor:
    """累加。"""
    return value.cumsum(dim)


def baddbmm(input: Tensor, batch1: Tensor, batch2: Tensor, *, beta: float = 1.0,
            alpha: float = 1.0) -> Tensor:
    """``beta * input + alpha * batch1 @ batch2``（逐批）。"""
    body = np.matmul(batch1.data, batch2.data)
    out = beta * input.data + alpha * body
    return Tensor(out, input.dtype, input.device)


def maximum(a: Any, b: Any) -> Tensor:
    """逐元素取大。"""
    ta = a if isinstance(a, Tensor) else Tensor(np.asarray(a, dtype=np.float32), float32)
    tb = b if isinstance(b, Tensor) else Tensor(np.asarray(b, dtype=np.float32), float32)
    d = promote_types(ta.dtype, tb.dtype)
    return Tensor(_cast_array(np.maximum(ta.data, tb.data).astype(np.float32), d), d)


def minimum(a: Any, b: Any) -> Tensor:
    """逐元素取小。"""
    ta = a if isinstance(a, Tensor) else Tensor(np.asarray(a, dtype=np.float32), float32)
    tb = b if isinstance(b, Tensor) else Tensor(np.asarray(b, dtype=np.float32), float32)
    d = promote_types(ta.dtype, tb.dtype)
    return Tensor(_cast_array(np.minimum(ta.data, tb.data).astype(np.float32), d), d)


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """矩阵乘（支持 >=3 维的批）。"""
    out = np.matmul(_as_array(a), _as_array(b))
    d = _promote(a.dtype, b.dtype)
    return Tensor(_cast_array(np.asarray(out), d), d, a.device)


def bmm(a: Tensor, b: Tensor) -> Tensor:
    """批量矩阵乘（三维）。"""
    return matmul(a, b)


def softmax(value: Tensor, dim: int = -1) -> Tensor:
    """数值稳定的 softmax。"""
    x = _as_array(value).astype(np.float32)
    shifted = x - np.max(x, axis=dim, keepdims=True)
    e = np.exp(shifted)
    out = e / np.sum(e, axis=dim, keepdims=True)
    d = _promote(value.dtype, float32) if value.dtype != bfloat16 else bfloat16
    return Tensor(_cast_array(out.astype(np.float32), d), d, value.device)


def sigmoid(value: Tensor) -> Tensor:
    """logistic。"""
    x = _as_array(value).astype(np.float32)
    out = 1.0 / (1.0 + np.exp(-x))
    return Tensor(_cast_array(out, value.dtype), value.dtype, value.device)


def exp(value: Tensor) -> Tensor:
    """指数。"""
    return value.exp()


def rsqrt(value: Tensor) -> Tensor:
    """平方根的倒数。"""
    x = _as_array(value).astype(np.float32)
    return Tensor(_cast_array(1.0 / np.sqrt(x), value.dtype), value.dtype, value.device)


def sqrt(value: Tensor) -> Tensor:
    """开方。"""
    return value.sqrt()


def cos(value: Tensor) -> Tensor:
    """余弦。"""
    return value.cos()


def sin(value: Tensor) -> Tensor:
    """正弦。"""
    return value.sin()


def sum(value: Tensor, dim: Any = None, keepdim: bool = False) -> Tensor:
    """求和。"""
    return value.sum(dim, keepdim)


def max(value: Tensor, dim: Any = None) -> Any:
    """最大值。"""
    return value.max(dim)


def min(value: Tensor, dim: Any = None) -> Any:
    """最小值。"""
    return value.min(dim)


def argmax(value: Tensor, dim: int | None = None) -> Tensor:
    """最大值下标。"""
    return value.argmax(dim)


def isfinite(value: Tensor) -> Tensor:
    """有限性判断。"""
    return Tensor(np.isfinite(_as_array(value)), bool, value.device)


def isnan(value: Tensor) -> Tensor:
    """NaN 判断。"""
    return Tensor(np.isnan(_as_array(value)), bool, value.device)


def allclose(a: Tensor, b: Tensor, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """接近判定（``torch.allclose`` 的接口）。"""
    return bool(np.allclose(_as_array(a), _as_array(b), rtol=rtol, atol=atol))


def equal(a: Tensor, b: Tensor) -> bool:
    """逐元素相等。"""
    return bool(np.array_equal(_as_array(a), _as_array(b)))


# ---------------------------------------------------------------------------
# 上下文管理（推理链里只用来关梯度）
# ---------------------------------------------------------------------------


class _NoGrad:
    """``no_grad`` / ``inference_mode`` 的空实现：本壳没有梯度。"""

    def __enter__(self) -> "_NoGrad":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def __call__(self, fn: Any) -> Any:
        """当装饰器用。"""
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            with self:
                return fn(*args, **kwargs)
        return wrapped


no_grad = _NoGrad
inference_mode = _NoGrad
enable_grad = _NoGrad


class profiler:
    """``torch.profiler`` 的壳。"""

    @staticmethod
    def profile(*args: Any, **kwargs: Any) -> _NoGrad:
        """无操作地进入。"""
        return _NoGrad()


class cuda:
    """``torch.cuda`` 的壳；真正是否在卡上由 ``cxrt`` 的运行时决定。"""

    @staticmethod
    def is_available() -> bool:
        """本壳不做真实探测，恒为假。"""
        return False

    @staticmethod
    def synchronize() -> None:
        """无操作。"""
        return None

    @staticmethod
    def current_device() -> int:
        """当前设备号。"""
        return 0

    @staticmethod
    def device_count() -> int:
        """设备数。"""
        return 0

    @staticmethod
    def memory_allocated() -> int:
        """已分配字节数。"""
        return 0

    @staticmethod
    def empty_cache() -> None:
        """无操作。"""
        return None


class cpu:
    """``torch.cpu`` 的壳：numpy 后端恒可用。"""

    @staticmethod
    def is_available() -> bool:
        """CPU 一直在。"""
        return True

    @staticmethod
    def synchronize() -> None:
        """无操作。"""
        return None

    @staticmethod
    def current_device() -> str:
        """当前设备名。"""
        return "cpu"


def manual_seed(seed: int) -> None:
    """设置 numpy 全局种子（派生 ``randn`` 用）。"""
    np.random.seed(seed)


def randn(*shape: Any, dtype: Any = None, device: Any = None, **kwargs: Any) -> Tensor:
    """标准正态。"""
    d = _norm_dtype(dtype, float32)
    out = np.random.standard_normal(_shape_tuple(shape)).astype(_np_kind(d))
    return Tensor(_cast_array(out, d), d, _dev(device))


def rand(*shape: Any, dtype: Any = None, device: Any = None, **kwargs: Any) -> Tensor:
    """[0,1) 均匀。"""
    d = _norm_dtype(dtype, float32)
    out = np.random.random(_shape_tuple(shape)).astype(_np_kind(d))
    return Tensor(_cast_array(out, d), d, _dev(device))


#: 壳的版本号。下游有拿它做分支的代码，给一个明确的字串比留空更安全。
__version__ = "2.6.0+cxrt"


class version:
    """``torch.version`` 的壳：全是 ``None``，表示「不是 CUDA 构建、也不是 ROCm」。

    后端探测里会读 ``torch.version.hip``/``cuda`` 判断是不是厂商构建；本壳两者都
    没有，探测因此会走「非厂商」分支——这正是要用的事实，不是缺省值。
    """

    cuda = None
    hip = None
    miopen = None
    cudnn = None
    nccl = None


#: ``torch.Size`` 就是 tuple；模型里会 ``torch.Size([2, 3])``。
Size = tuple


class autograd:
    """``torch.autograd`` 的壳：没有梯度。"""

    @staticmethod
    def no_grad() -> _NoGrad:
        """无梯度上下文。"""
        return _NoGrad()

    @staticmethod
    def set_grad_enabled(mode: bool) -> _NoGrad:    # noqa: D401 - 语义同上下文
        """无梯度上下文。"""
        return _NoGrad()

    @staticmethod
    def backward(*args: Any, **kwargs: Any) -> None:
        """无操作。"""
        return None


class jit:
    """``torch.jit`` 的壳：装饰器原样返回被装饰对象。"""

    @staticmethod
    def script(obj: Any = None, **kwargs: Any) -> Any:
        """直接返回原对象。"""
        return obj

    @staticmethod
    def ignore(*args: Any, **kwargs: Any) -> Any:
        """当装饰器用。"""
        if args and callable(args[0]):
            return args[0]

        def _wrap(fn: Any) -> Any:
            return fn
        return _wrap

    @staticmethod
    def is_scripting() -> bool:
        """恒假。"""
        return False


class amp:
    """``torch.amp`` 的壳。"""

    @staticmethod
    def autocast(*args: Any, **kwargs: Any) -> _NoGrad:
        """无操作地进入。"""
        return _NoGrad()

    class GradScaler:
        """空的梯度缩放器。"""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def scale(self, value: Any) -> Any:
            return value


def set_grad_enabled(mode: bool) -> _NoGrad:
    """无梯度上下文。"""
    return _NoGrad()


def is_grad_enabled() -> bool:
    """恒假（没有梯度）。"""
    return False


def set_float32_matmul_precision(precision: str) -> None:
    """无操作，保留接口。"""
    return None


def use_deterministic_algorithms(mode: bool, **kwargs: Any) -> None:
    """无操作，保留接口。"""
    return None


class _Finfo:
    """``torch.finfo``。"""

    def __init__(self, d: Any) -> None:
        target = _norm_dtype(d)
        if target == bfloat16:
            # bf16 的 numpy 侧没有对应类型（这里用 fp32 存放），所以数值属性只能按
            # 定义算：7 位显式尾数 -> eps = 2^-8，指数域与 fp32 相同。
            self.bits = 16
            self.eps = float(2.0 ** -8)
            self.max = float(np.finfo(np.float32).max)
            self.min = float(-np.finfo(np.float32).max)
            self.tiny = float(2.0 ** -126)
            return
        info = np.finfo(_np_kind(target))
        self.bits = int(info.bits)
        self.eps = float(info.eps)
        self.max = float(info.max)
        self.min = float(info.min)
        self.tiny = float(info.tiny)


class _Iinfo:
    """``torch.iinfo``。"""

    def __init__(self, d: Any) -> None:
        kind = _np_kind(_norm_dtype(d))
        info = np.iinfo(kind)
        self.bits = int(info.bits)
        self.max = int(info.max)
        self.min = int(info.min)


def finfo(d: Any) -> _Finfo:
    """浮点 dtype 的数值范围。"""
    return _Finfo(d)


def iinfo(d: Any) -> _Iinfo:
    """整数 dtype 的范围。"""
    return _Iinfo(d)


def tril(value: Tensor, diagonal: int = 0) -> Tensor:
    """取严格/包含对角线的下三角。"""
    return Tensor(np.tril(_as_array(value), diagonal), value.dtype, value.device)


def triu(value: Tensor, diagonal: int = 0) -> Tensor:
    """取上三角。"""
    return Tensor(np.triu(_as_array(value), diagonal), value.dtype, value.device)


def einsum(spec: str, *operands: Any) -> Tensor:
    """爱因斯坦求和。"""
    arrays = [x.data if isinstance(x, Tensor) else np.asarray(x) for x in operands]
    ref = next((x for x in operands if isinstance(x, Tensor)), None)
    out = np.einsum(spec, *arrays)
    d = ref.dtype if ref is not None else _dtype_of_array(np.asarray(out), None)
    return Tensor(_cast_array(np.asarray(out), d), d, ref.device if ref is not None else _default_device)


def gelu(value: Tensor, approximate: str = "none") -> Tensor:
    """GELU（转给 ``ttnn.functional``，避免两处实现走散）。"""
    from cxrt import ttnn

    return ttnn.functional.gelu(value, approximate=approximate)


def relu(value: Tensor) -> Tensor:
    """ReLU。"""
    return Tensor(np.maximum(_as_array(value), 0), value.dtype, value.device)


def tanh(value: Tensor) -> Tensor:
    """双曲正切。"""
    return Tensor(np.tanh(_as_array(value)), value.dtype, value.device)


def norm(value: Tensor, p: Any = 2, dim: Any = None, keepdim: bool = False) -> Tensor:
    """范数。"""
    out = np.linalg.norm(_as_array(value).astype(np.float32), ord=p, axis=dim, keepdims=keepdim)
    return Tensor(out.astype(np.float32), value.dtype, value.device)


def topk(value: Tensor, k: int, dim: int = -1, largest: bool = True,
         sorted: bool = True) -> tuple[Tensor, Tensor]:
    """沿某维取前 ``k`` 个，返回 ``(值, 下标)``。

    路由（MoE 的 ``softmax -> topk``）就靠它；下标是 ``int64``，与 torch 一致。
    同值时的顺序不定：torch 也没承诺，而路由在数值相同处取谁，本来就是任意的。

    Args:
        value: 输入张量。
        k: 取的个数。
        dim: 沿哪一维。
        largest: ``True`` 取最大，``False`` 取最小。
        sorted: 是否把结果按值排序（本实现总是排好，参数保留以兼容）。

    Returns:
        ``(values, indices)``，形状同 ``value`` 但该维变成 ``k``。
    """
    data = _as_array(value)
    axis = dim % data.ndim
    taken = int(k) if int(k) <= data.shape[axis] else data.shape[axis]
    order = np.argsort(-data, axis=axis, kind="stable") if largest else \
        np.argsort(data, axis=axis, kind="stable")
    idx = np.take(order, np.arange(taken), axis=axis)
    vals = np.take_along_axis(data, idx, axis=axis)
    return (Tensor(vals.astype(np.float32), value.dtype, value.device),
            Tensor(idx.astype(np.int64), int64, value.device))


class _Linalg:
    """``torch.linalg`` 用到的那一小块：三角求解。"""

    @staticmethod
    def solve_triangular(lower: Tensor, rhs: Tensor, *, upper: bool = False,
                         unitriangular: bool = False) -> Tensor:
        """解 ``L X = B``（或 ``U X = B``），逐批前代/回代。

        ``lower`` 是 ``[bn, L, L]``、``rhs`` 是 ``[bn, L, n]``。前代是顺序依赖的，
        所以按行循环；``L`` 在推理链里是一个窗口大小（16 上下），代价可忽略。

        Args:
            lower: 三角系数矩阵。
            rhs: 右端项，行数与 ``lower`` 的阶数相同。
            upper: ``True`` 时按上三角解。
            unitriangular: 对角线视为 1（不读对角）。

        Returns:
            与 ``rhs`` 同形同 dtype。

        Raises:
            ValueError: 阶数与右端行数不匹配。
        """
        a = _as_array(lower).astype(np.float64)
        b = _as_array(rhs).astype(np.float64)
        if a.ndim != 3 or a.shape[-1] != a.shape[-2]:
            raise ValueError(f"coefficient must be [bn, L, L], got {a.shape}")
        if b.shape[:-2] != a.shape[:-2] or b.shape[-2] != a.shape[-1]:
            raise ValueError(f"rhs {b.shape} does not match {a.shape}")
        order = a.shape[-1]
        out = np.zeros_like(b)
        if not upper:
            for i in range(order):
                acc = b[..., i, :].copy()
                if i:
                    acc = acc - np.einsum("bj,bjn->bn", a[..., i, :i], out[..., :i, :])
                diag = 1.0 if unitriangular else a[..., i, i][:, None]
                out[..., i, :] = acc / diag
        else:
            for i in range(order - 1, -1, -1):
                acc = b[..., i, :].copy()
                if i + 1 < order:
                    acc = acc - np.einsum("bj,bjn->bn", a[..., i, i + 1:],
                                          out[..., i + 1:, :])
                diag = 1.0 if unitriangular else a[..., i, i][:, None]
                out[..., i, :] = acc / diag
        return Tensor(out.astype(np.float32), rhs.dtype, rhs.device)


linalg = _Linalg()


def eye(n: int, m: int | None = None, dtype: Any = None, device: Any = None) -> Tensor:
    """单位矩阵（``torch.eye``）。"""
    rows, cols = int(n), int(n) if m is None else int(m)
    return Tensor(np.eye(rows, cols, dtype=np.float32), dtype or float32, device)


def linspace(start: float, end: float, steps: int, dtype: Any = None,
             device: Any = None) -> Tensor:
    """等距数列。"""
    return Tensor(np.linspace(start, end, int(steps), dtype=np.float32),
                  dtype or float32, device)


def compile(model: Any = None, **kwargs: Any) -> Any:
    """``torch.compile`` 的壳：原样返回（本壳没有图编译）。"""
    if callable(model) and not isinstance(model, type):
        return model
    return model


def _dynamo_stub() -> Any:
    """``torch._dynamo`` 的空壳。"""
    return types.SimpleNamespace(config=types.SimpleNamespace(
        suppress_errors=True), disable=lambda fn=None, **k: (fn if fn else (lambda f: f)))


def load(path: Any, map_location: Any = None, **kwargs: Any) -> Any:
    """读 checkpoint——交给 :mod:`cxrt.store`，那里不依赖 torch。

    Args:
        path: ``.pt`` / ``.bin`` 路径。
        map_location: 兼容 torch 的参数名，本壳忽略。
        **kwargs: 其余 torch 参数（``weights_only`` 等），忽略。

    Returns:
        反序列化出来的对象（dict / Tensor / 标量）。
    """
    from cxrt import store

    return store.load(path)


def save(obj: Any, path: Any, **kwargs: Any) -> None:
    """写 numpy 存档（``.npz``），只用于自有产物，不追求 torch 兼容。"""
    from cxrt import store

    store.save(obj, path)
