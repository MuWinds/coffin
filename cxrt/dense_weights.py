"""把预量化的 w8a8 checkpoint 折成一份稠密 state dict（不依赖 torch）。

Module: ``coffin/cxrt``。盘上的预量化目录是 **nk** 布局：``int8 [N, K]``（每行一个
输出通道、K 轴连续）加 ``fp32 [N]`` 每行标尺。这里把每个 int8 张量乘上它那行标尺、
还原成 bf16，再交给 :class:`models.qwen35.Qwen35` 的 ``weights=`` 入口——模型的
``Linear``/``GEMV`` 本来就按 ``[N, K]`` 建，所以**一次转置都不用做**。

为什么要有这条路：它是「模型能不能在无 torch 进程里跑出 token」的最短验证。它不需要
23 个还没搬的算子，也不需要把 ``backends`` 适配层重写——只需要模型代码本身能在这套
壳上跑。等 token 出来、数值对齐之后，再逐个把稠密路径换成 ABI 内核，改动才有对照。

内存：4B 的 int8 权重是 4.3 GB，折成 bf16 是 8.6 GB 量级；用 ``np.memmap`` 逐分片读、
逐分片折，峰值只多一个分片。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from cxrt import safetensors
from cxrt.tt import Tensor, bfloat16, float32

#: 标尺键的后缀。量化目录里每个需量化的张量都配一条 ``<key>#scale``。
SCALE_SUFFIX = "#scale"


def shards(model_dir: Path) -> list[Path]:
    """列出目录下的 safetensors 分片。

    Args:
        model_dir: checkpoint 目录。

    Returns:
        按名字排序的分片路径。
    """
    return sorted(p for p in Path(model_dir).glob("*.safetensors"))


def dequantize(weight: Tensor, scale: Tensor, target: Tensor | None = None) -> Tensor:
    """``w8a8`` 反量化：``int8 [N, K]`` × 每行标尺 ``[N]`` → ``bf16 [N, K]``。

    Args:
        weight: ``int8 [N, K]``——盘上的 nk 存储序，一行输出通道连续。
        scale: ``float32 [N]``，每个输出行一个标尺。
        target: 目标 dtype；``None`` 取 bf16（模型的稠密路径吃这个）。

    Returns:
        稠密 ``[N, K]`` 张量，形状与盘上一致（nk 与模型的行主序本来就一致）。

    Raises:
        ValueError: 维度或长度对不上——对不上时乘出来是「尺寸正确、内容错位」的权重。
    """
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2-D, got {weight.shape}")
    rows = int(weight.shape[0])
    if scale.ndim != 1 or int(scale.numel()) != rows:
        raise ValueError(f"scale must be [{rows}] (one per output row), got {scale.shape}")
    dense = weight.data.astype(np.float32) * scale.data.astype(np.float32)[:, None]
    out_dtype = target.dtype if isinstance(target, Tensor) else bfloat16
    return Tensor(np.ascontiguousarray(dense), out_dtype).to(out_dtype)


def load_dense(model_dir: Path, *, limit_gib: float | None = None,
               verbose: bool = True) -> dict[str, Tensor]:
    """读整个量化目录，返回稠密 state dict。

    Args:
        model_dir: checkpoint 目录。
        limit_gib: 只读若干 GiB 就停（做形状探针时用）；``None`` 读全部。
        verbose: 打印每个分片的处理量。

    Returns:
        ``{参数名: Tensor}``，int8 已折成 bf16，``#scale`` 键已消化掉。

    Raises:
        FileNotFoundError: 目录里没有 safetensors。
        ValueError: 某个 int8 张量缺标尺。
    """
    files = shards(model_dir)
    if not files:
        raise FileNotFoundError(f"no *.safetensors under {model_dir}")

    out: dict[str, Tensor] = {}
    seen = 0.0
    for path in files:
        table = safetensors.load(path)
        for key, value in table.items():
            if key.endswith(SCALE_SUFFIX):
                continue
            if value.dtype.name == "int8" and value.ndim == 2:
                scale_key = key + SCALE_SUFFIX
                scale = table.get(scale_key)
                if scale is None:
                    raise ValueError(f"{path.name}: {key} has no {scale_key}")
                out[key] = dequantize(value, scale)
                seen += value.numel() / 2**30
            else:
                out[key] = value
        if verbose:
            print(f"[dense] {path.name}: cumulative {seen:.2f} GiB dequantised, "
                  f"{len(out)} tensors", flush=True)
        if limit_gib is not None and seen >= limit_gib:
            break
    return out


def iter_keys(model_dir: Path) -> Iterator[str]:
    """只枚举键与形状，不折权重（给形状探针和 ``keep`` 过滤用）。

    Args:
        model_dir: checkpoint 目录。

    Yields:
        每个参数名。
    """
    for path in shards(model_dir):
        header = safetensors.read_header(path)
        for key in header:
            if key != "__metadata__" and not key.endswith(SCALE_SUFFIX):
                yield key


__all__ = ["SCALE_SUFFIX", "dequantize", "iter_keys", "load_dense", "shards"]
