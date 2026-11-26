"""张量家族中立层：只靠实例方法读写张量，不 import 任何框架。

Module: ``coffin/cxrt``。叶子模块，只 import ``cxrt.array`` / ``cxrt.dtypes`` 与
标准库。**不 import torch**：corex 后端要在这层之上重写，而 ``torch`` 正是那里
要消失的东西。

为什么需要这一层：corex 后端过去直接调 ``torch.empty(dtype=torch.int8, ...)``、
``x.to(dtype=torch.int32)``、``x.dtype in (torch.bfloat16, ...)``。这些调用同时
绑定了三件互不相干的事——**分配**、**类型转换**、**类型判定**——而它们全都可以
从张量实例自己身上问出来：

    * 分配：``anchor.new_empty(shape)`` / ``anchor.new_zeros(shape)``（任何持有
      分配器的张量都有这一对方法；``Array`` 在 :mod:`cxrt.array` 里补上）。
    * 转换：``x.float()`` / ``x.bfloat16()`` / ``x.int()`` / ``x.char()`` ——
      numpy 与 torch 都认这套语义化方法名，所以它是家族中立的。
    * 判定：``x.dtype`` 的**名字**（``"torch.bfloat16"``、``"float32"``、``"bf16"``）
      映射到 ABI 的整数码，判等只在整数上做。

于是调用方不再需要框架的类型对象，只需要一个锚张量（anchor）。锚张量总是有的：
corex 的每个入口都至少收到一个激活。

Constraints:
    * 这里的每个函数都收「张量」而不是 ``torch.Tensor``。判断家族只看实例属性
      （``data_ptr`` vs ``ptr``），**不做 isinstance**——``isinstance`` 需要框架类型，
      那正是要避开的东西。
    * :func:`code_of` 对未知 dtype 抛 :class:`~cxrt.errors.ShapeError`，不猜一个
      「差不多」的码：猜错会让一次类型错误表现为数值漂移。
    * :func:`cast` 在 ``Array`` 上是**宿主机往返**，只作迁移期的权宜实现：它把设备
      内存读回、按 numpy 转、再写回。corex 的正式实现要走设备侧转换内核，这里出现
      往返只应出现在冷路径（索引张量的 int32 归一化）。这条约束写在函数里，而不是
      留在某人的记忆里。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cxrt.array import Array
from cxrt.dtypes import BF16, BY_CODE, BY_NAME, DType, I32, bf16_bits, bf16_to_f32
from cxrt.errors import ShapeError

#: 任何「形状 + dtype + 设备指针」的对象，含 ``Array`` 与框架张量。
TensorLike = Any

#: ABI 码值，导出成短名字：调用点写 ``F32`` 而不是 ``torch.float32``。
F32 = BY_NAME["f32"].code
BF16C = BF16.code
I8 = BY_NAME["i8"].code
I32C = I32.code
I64 = BY_NAME["i64"].code
U8 = BY_NAME["u8"].code
U16 = BY_NAME["u16"].code
U32 = BY_NAME["u32"].code

#: ``str(dtype)`` 的后缀到码值。框架的 dtype 打印成 ``torch.bfloat16``，numpy 的
#: 打印成 ``float32`` / ``int8``；两者都按最后一段认。
_SUFFIX_CODES: dict[str, int] = {
    "float32": F32, "float": F32, "f32": F32,
    "bfloat16": BF16C, "bf16": BF16C,
    "int8": I8, "i8": I8, "char": I8,
    "int32": I32C, "i32": I32C, "int": I32C,
    "int64": I64, "i64": I64, "long": I64,
    "uint8": U8, "u8": U8, "byte": U8,
    "uint16": U16, "u16": U16,
    "uint32": U32, "u32": U32,
}

#: 码值到家族中立转换方法名。numpy 与 torch 的张量都有这套方法。
_CAST_METHODS: dict[int, str] = {
    F32: "float", BF16C: "bfloat16", I8: "char", I32C: "int", I64: "long",
    U8: "byte",
}


def is_array(t: TensorLike) -> bool:
    """判断是不是 ``cxrt.Array``。

    Args:
        t: 任意张量。

    Returns:
        ``Array`` 为 ``True``，其它（框架张量）为 ``False``。只在同进程内用，
        所以 ``isinstance`` 在这里是安全的——``Array`` 是本包自己的类型。
    """
    return isinstance(t, Array)


#: dtype 对象（``torch.dtype`` / ``np.dtype`` / ``DType``）到 ABI 整数码的缓存，
#: 避免热路径上每次 ``code_of`` 都走 ``str(dtype).rsplit``。
_DTYPE_OBJ_CACHE: dict[Any, int] = {}


def code_of(t_or_dtype: TensorLike | DType | Any) -> int:
    """把 dtype（或一个张量的 dtype）映射成 ABI 整数码。

    Args:
        t_or_dtype: 张量、:class:`~cxrt.dtypes.DType`，或任何能 ``str()`` 出
            ``"<module>.<name>"`` 的 dtype 对象。

    Returns:
        内核 ABI 的 dtype 码。

    Raises:
        ShapeError: 名字认不出来。抛而不是回退到 fp32：静默回退会把「传错了
            dtype」变成一次数值漂移，而且只在那条形状上出现。
    """
    dtype = getattr(t_or_dtype, "dtype", t_or_dtype)
    try:
        cached = _DTYPE_OBJ_CACHE.get(dtype)
        if cached is not None:
            return cached
    except TypeError:
        cached = None
    if isinstance(dtype, DType):
        _DTYPE_OBJ_CACHE[dtype] = dtype.code
        return dtype.code
    name = str(dtype)
    suffix = name.rsplit(".", 1)[-1]
    code = _SUFFIX_CODES.get(suffix)
    if code is None:
        raise ShapeError(f"unknown dtype {name!r}; add it to cxrt.tensors._SUFFIX_CODES")
    try:
        _DTYPE_OBJ_CACHE[dtype] = code
    except TypeError:
        pass
    return code


def dtype_of(t: TensorLike) -> DType:
    """返回张量的 dtype 在 ``cxrt`` 里的对应记录。

    Args:
        t: 任意张量。

    Returns:
        :class:`~cxrt.dtypes.DType`。

    Raises:
        ShapeError: dtype 在设备侧没有对应存储。
    """
    return BY_CODE[code_of(t)]


def ptr_of(t: TensorLike) -> int:
    """取设备地址。

    Args:
        t: 任意张量。``Array`` 走 ``.ptr``，框架张量走 ``.data_ptr()``。

    Returns:
        设备指针（整数）。

    Raises:
        ShapeError: 对象没有这两种形式之一——说明它根本不是一块设备内存。
    """
    if is_array(t):
        return int(t.ptr)
    data_ptr = getattr(t, "data_ptr", None)
    if data_ptr is not None:
        return int(data_ptr())
    attr_ptr = getattr(t, "ptr", None)
    if isinstance(attr_ptr, int):
        return attr_ptr
    raise ShapeError(f"{type(t).__name__} exposes neither .data_ptr() nor .ptr")


def shape_of(t: TensorLike) -> tuple[int, ...]:
    """返回形状。

    Args:
        t: 任意张量。

    Returns:
        各维长度。
    """
    return tuple(int(s) for s in t.shape)


def numel_of(t: TensorLike) -> int:
    """返回元素个数。

    Args:
        t: 任意张量。

    Returns:
        ``shape`` 的乘积。框架张量的 ``numel`` 是方法，``Array`` 是属性，两种都认。
    """
    numel = getattr(t, "numel", None)
    if callable(numel):
        return int(numel())
    if isinstance(numel, int):
        return numel
    out = 1
    for size in shape_of(t):
        out *= size
    return out


def strides_of(t: TensorLike) -> tuple[int, ...]:
    """返回元素为单位的步长。

    Args:
        t: 任意张量。``Array`` 读 ``.strides``，框架张量读 ``.stride()``。

    Returns:
        各维步长；与 :func:`shape_of` 等长。

    Raises:
        ShapeError: 两种形式都没有——说明它不是一块可索引的存储。
    """
    if is_array(t):
        return tuple(int(s) for s in t.strides)
    stride = getattr(t, "stride", None)
    if callable(stride):
        return tuple(int(s) for s in stride())
    raise ShapeError(f"{type(t).__name__} exposes no .stride()")


def device_key(t: TensorLike) -> Any:
    """返回可以比较的设备身份。

    Args:
        t: 任意张量。

    Returns:
        ``Array`` 返回它的运行时对象（同运行时即同设备）；框架张量返回 ``.device``。
        只用于「两个张量在不在一张卡上」这类判断，不用于构造设备对象。
    """
    if is_array(t):
        return t._rt                                    # noqa: SLF001 - 同包内视图
    return getattr(t, "device", None)


def cast(t: TensorLike, code: int) -> TensorLike:
    """把张量转到目标 dtype。

    Args:
        t: 任意张量。
        code: 目标 ABI 码值。

    Returns:
        已是目标 dtype 时原样返回 ``t``（省掉一次与 dtype 相关的分发，调用点大量
        传进来的本来就是对的），否则返回新张量。

    Raises:
        ShapeError: 该 dtype 没有家族中立的转换方法（例如 ``uint16``）。

    Constraints:
        ``Array`` 上的转换是宿主机往返，**只用于冷路径**：正式实现应当走设备侧
        转换内核。这里保留它是为了让迁移期的索引张量归一化（``int32``）不至于
        无处可走，而不是给热路径提供一条能跑的替代品。
    """
    if code_of(t) == code:
        return t
    if is_array(t):
        return _array_cast(t, code)
    method_name = _CAST_METHODS.get(code)
    if method_name is None:
        raise ShapeError(
            f"no family-neutral cast to {BY_CODE[code].name}; "
            f"add a method name to cxrt.tensors._CAST_METHODS")
    method = getattr(t, method_name, None)
    if method is None:
        raise ShapeError(f"{type(t).__name__} has no .{method_name}()")
    return method()


def _array_cast(t: Array, code: int) -> Array:
    """``Array`` 的权宜转换：读回宿主、按 numpy 转、写回设备。

    Args:
        t: 源数组。
        code: 目标码值。

    Returns:
        新数组，目标 dtype。

    Raises:
        ShapeError: 分配器不可用。
    """
    target = BY_CODE[code]
    values = t.to_numpy()
    if code_of(t) == BF16C:
        values = bf16_to_f32(values)
    if target.code == BF16C:
        bits = bf16_bits(values.astype(np.float32, copy=False))
        out = Array.empty(t._rt, _alloc_of(t), shape_of(t), target)  # noqa: SLF001
        out.copy_from(bits)
        return out
    out = Array.empty(t._rt, _alloc_of(t), shape_of(t), target)      # noqa: SLF001
    out.copy_from(np.ascontiguousarray(values, dtype=target.numpy))
    return out


def _alloc_of(t: Array) -> Any:
    """取出数组背后的分配器，供新分配用。

    Args:
        t: 数组。

    Returns:
        分配器。

    Raises:
        ShapeError: 缓冲已经关闭，分配器被清掉了。
    """
    alloc = getattr(t._buf, "_allocator", None)                     # noqa: SLF001
    if alloc is None:
        raise ShapeError("source Array is closed; no allocator to allocate from")
    return alloc


def new_empty(anchor: TensorLike, shape: tuple[int, ...], code: int) -> TensorLike:
    """按锚张量的家族与设备分配一块未初始化内存。

    Args:
        anchor: 提供家族与设备的张量。
        shape: 形状。
        code: dtype 码值。

    Returns:
        新张量，dtype 为目标码值。

    Raises:
        ShapeError: 锚张量浮不起来（没有 ``new_empty``），或形状非法。
    """
    shape = tuple(int(s) for s in shape)
    if not shape or any(s <= 0 for s in shape):
        raise ShapeError(f"new_empty{shape}: every dimension must be positive")
    if is_array(anchor):
        return Array.empty(anchor._rt, _alloc_of(anchor), shape,  # noqa: SLF001
                           BY_CODE[code])
    prototype = cast(anchor, code)
    new = getattr(prototype, "new_empty", None)
    if new is None:
        raise ShapeError(f"{type(anchor).__name__} has no .new_empty()")
    return new(shape)


def new_zeros(anchor: TensorLike, shape: tuple[int, ...], code: int) -> TensorLike:
    """与 :func:`new_empty` 相同，但内容清零。

    Args:
        anchor: 提供家族与设备的张量。
        shape: 形状。
        code: dtype 码值。

    Returns:
        全零新张量。

    Raises:
        ShapeError: 家族没有 ``new_zeros``；``Array`` 走 :meth:`Array.zeros`。
    """
    shape = tuple(int(s) for s in shape)
    if not shape or any(s <= 0 for s in shape):
        raise ShapeError(f"new_zeros{shape}: every dimension must be positive")
    if is_array(anchor):
        return Array.zeros(anchor._rt, _alloc_of(anchor), shape,  # noqa: SLF001
                           BY_CODE[code])
    prototype = cast(anchor, code)
    new = getattr(prototype, "new_zeros", None)
    if new is None:
        raise ShapeError(f"{type(anchor).__name__} has no .new_zeros()")
    return new(shape)


def new_empty_like(t: TensorLike, code: int | None = None) -> TensorLike:
    """按另一个张量的形状与 dtype 分配。

    Args:
        t: 形状与 dtype 的来源。
        code: 目标码值；``None`` 表示跟随 ``t``。

    Returns:
        新张量。
    """
    return new_empty(t, shape_of(t), code_of(t) if code is None else code)


def as_strided(t: TensorLike, shape: tuple[int, ...],
               strides: tuple[int, ...]) -> TensorLike:
    """按给定形状/步长重新解释同一块内存（零拷贝视图）。

    Args:
        t: 源张量。
        shape: 视图形状。
        strides: 元素为单位的步长。

    Returns:
        视图。框架张量走它自己的 ``as_strided``；``Array`` 直接构造视图对象，
        共享同一个缓冲。

    Raises:
        ShapeError: 家族没有这种视图，或视图越界。
    """
    shape = tuple(int(s) for s in shape)
    strides = tuple(int(s) for s in strides)
    if is_array(t):
        need = 1 + sum((s - 1) * st for s, st in zip(shape, strides))
        if need > numel_of(t):
            raise ShapeError(f"as_strided{shape}/{strides} overruns {numel_of(t)} elements")
        return Array(t._rt, t._buf, shape, strides, t.dtype, t.offset)  # noqa: SLF001
    method = getattr(t, "as_strided", None)
    if method is None:
        raise ShapeError(f"{type(t).__name__} has no .as_strided()")
    return method(shape, strides)
