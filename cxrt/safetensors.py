"""safetensors 读取：不依赖 torch、不依赖 safetensors 包。

Module: ``coffin/cxrt``。仓库里的 checkpoint（``Qwen3.5-4B-int8-nk``、
``occamy-1.0-W8A8``）落盘是 safetensors，不是 ``torch.save`` 的 zip：文件头 8 字节是
JSON 头的长度，紧随其后是一张按张量名索引的表，每个条目给出 ``dtype``、``shape`` 与
``data_offsets``。数据区紧接着 JSON 头，按 8 字节对齐排布。

所以读它只需要三件事：解析头、把 ``data_offsets`` 加上前缀、按 dtype 解释字节。这里
用 ``np.memmap`` 直接映射数据区，**不把整份权重读进内存**——4B 的 w8a8 checkpoint 是
2.5 GB 级别，拷贝一遍就把机器逼到交换区。

bf16 的处理：safetensors 里它按 uint16 存（每个元素 16 位），这里还原成 fp32 的
:class:`cxrt.tt.Tensor`，与 ``cxrt.store`` 的口径一致。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from cxrt.tt import DTYPES, Tensor, bfloat16, from_numpy

#: safetensors 的 dtype 串到 ``cxrt.tt`` 的 dtype 的映射。
_DTYPES: dict[str, Any] = {
    "F64": DTYPES["float64"],
    "F32": DTYPES["float32"],
    "F16": DTYPES["float16"],
    "BF16": bfloat16,
    "I64": DTYPES["int64"],
    "I32": DTYPES["int32"],
    "I16": DTYPES["int16"],
    "I8": DTYPES["int8"],
    "U8": DTYPES["uint8"],
    "BOOL": DTYPES["bool"],
}

#: numpy 侧的存放类型（bf16 用 uint16 先读出，再移位成 fp32）。
_RAW_KIND: dict[str, Any] = {
    "F64": np.float64, "F32": np.float32, "F16": np.float16, "BF16": np.uint16,
    "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8,
    "U8": np.uint8, "BOOL": np.bool_,
}


def read_header(path: Any) -> dict[str, Any]:
    """读出 safetensors 的 JSON 头。

    Args:
        path: 文件路径。

    Returns:
        头里除 ``__metadata__`` 之外的条目；另把元数据挂在 ``"__metadata__"`` 上。

    Raises:
        ValueError: 头长度或 JSON 不合法。
    """
    fp = Path(path)
    with fp.open("rb") as handle:
        head = handle.read(8)
        if len(head) < 8:
            raise ValueError(f"{fp}: truncated safetensors header")
        length = int(np.frombuffer(head, dtype=np.uint64)[0])
        if length <= 0 or length > 100 * 2**20:
            raise ValueError(f"{fp}: implausible header length {length}")
        raw = handle.read(length)
        if len(raw) < length:
            raise ValueError(f"{fp}: header claims {length} bytes, file has {len(raw)}")
    try:
        header = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{fp}: bad header json ({exc})") from None
    return header


def load(path: Any, keep_alive: bool = True) -> dict[str, Tensor]:
    """读一个 safetensors 文件。

    Args:
        path: 文件路径。
        keep_alive: 保留 ``np.memmap`` 对象（``True`` 时张量与文件共享页，省内存）。

    Returns:
        ``{张量名: Tensor}``，bf16 已还原成 fp32。

    Raises:
        ValueError: 头不合法，或 dtype 不支持。
    """
    fp = Path(path)
    header = read_header(fp)
    # 数据区起点 = 8 字节长度字段 + 文件里写的那个头长度（不要用重新序列化后的长度：
    # JSON 的空白与键序都可能变，字节数就不再等于原值）。
    with fp.open("rb") as handle:
        base = 8 + int(np.frombuffer(handle.read(8), dtype=np.uint64)[0])

    out: dict[str, Tensor] = {}
    mm = np.memmap(fp, dtype=np.uint8, mode="r")
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype_name = info.get("dtype", "")
        if dtype_name not in _RAW_KIND:
            raise ValueError(f"{fp}: unsupported dtype {dtype_name!r} for {name}")
        begin, end = (int(v) for v in info["data_offsets"])
        shape = tuple(int(v) for v in info.get("shape", ()))
        kind = _RAW_KIND[dtype_name]
        count = end - begin
        raw = mm[base + begin: base + end]
        if dtype_name == "BF16":
            bits = np.frombuffer(raw, dtype=np.uint16, count=count // 2)
            value = (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)
            value = value.reshape(shape).astype(np.float32, copy=False)
            out[name] = Tensor(np.ascontiguousarray(value), bfloat16)
        else:
            value = np.frombuffer(raw, dtype=kind, count=count // np.dtype(kind).itemsize)
            value = value.reshape(shape)
            out[name] = Tensor(np.ascontiguousarray(value) if not keep_alive else value,
                               _DTYPES[dtype_name])
    return out


def is_safetensors(path: Any) -> bool:
    """按后缀与头判断是不是 safetensors。

    Args:
        path: 文件路径。

    Returns:
        是则为真。
    """
    fp = Path(path)
    if fp.suffix == ".safetensors":
        return True
    try:
        read_header(fp)
    except (ValueError, OSError):
        return False
    return True


__all__ = ["is_safetensors", "load", "read_header"]
