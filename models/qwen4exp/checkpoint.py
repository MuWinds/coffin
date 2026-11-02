"""从真实 checkpoint 分片流式读张量，供 ``weights.load_into`` 直接使用。

参考 checkpoint 是 131 个 safetensors 分片、共 360 GB，本机无法整体驻留，所以
这里**只在被请求时才读**：``get(name)`` 才打开对应分片、读出单个张量，读完立刻
释放句柄。配合 ``weights.load_into`` 的逐张量装载，峰值内存是"一个张量 + 模型
本身"，而不是"整个 checkpoint + 模型"。

两个必须处理的细节：

* **分片定位**：先扫一遍所有 shard 的 header 建 ``名字 -> 分片`` 索引（只读 header，
  不读数据），否则每次 ``get`` 都要线性扫 131 个文件。
* **n-gram 表不预读**：单张表 95 GiB，任何路径下都不许整个 load 起来。它的 128 个
  分片由 :class:`StreamingSource` 按需一次读一片，``weights.load_into`` 负责拼接；
  真要装进一张表就要求调用方有对应内存，所以默认路径是 ``stream_ngram=True``。

Module: ``coffin/models/qwen4exp``。族内的 I/O 边界；依赖 ``safetensors``，
**只在集群上有**，所以导入放在方法内部——本机 import 本模块不应该失败。

Constraints:
    * header 索引必须缓存：``get`` 会被调用 1658 次，每次重扫 131 个 header 是
      分钟级开销。
    * 读出来的张量必须 ``.contiguous()``：safetensors 直接映射文件，返回的可能是
      非连续视图，后续 ``copy_`` 到参数上会慢一个数量级。
    * 不在这里做 dtype 转换：目标参数什么 dtype 由 ``load_into`` 的 ``prepare``
      决定，多转一次就多一份峰值内存。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

#: safetensors 文件的后缀。分片名形如 ``model-00001-of-00131.safetensors``。
SHARD_SUFFIX = ".safetensors"


@dataclass
class ShardIndex:
    """``张量名 -> 所在分片`` 的索引，只由 header 建成。

    Attributes:
        shards: 分片路径列表。
        locations: 张量名到分片路径的映射。
        dtypes: 张量名到 header 里记录的 dtype 字符串。
        shapes: 张量名到 header 里记录的形状。
    """

    shards: list[Path] = field(default_factory=list)
    locations: dict[str, Path] = field(default_factory=dict)
    dtypes: dict[str, str] = field(default_factory=dict)
    shapes: dict[str, list[int]] = field(default_factory=dict)

    def __len__(self) -> int:
        """索引里的张量总数。"""
        return len(self.locations)


def build_index(model_dir: str | Path) -> ShardIndex:
    """扫所有分片的 header，建张量名到分片的索引。

    只读 header（safetensors 格式把 header 放在文件开头），因此对 360 GB 的目录
    也是秒级；不会把任何权重读进内存。

    Args:
        model_dir: 权重目录。

    Returns:
        建好的索引。

    Raises:
        ImportError: 环境里没有 ``safetensors``（本机就是这样）。
        FileNotFoundError: 目录里一个分片都没有。
    """
    from safetensors import safe_open

    model_dir = Path(model_dir)
    index = ShardIndex(shards=sorted(model_dir.glob(f"*{SHARD_SUFFIX}")))
    if not index.shards:
        raise FileNotFoundError(f"no *{SHARD_SUFFIX} under {model_dir}")
    for path in index.shards:
        with safe_open(str(path), "pt") as handle:
            for name in handle.keys():
                if name in index.locations:
                    raise ValueError(
                        f"{name} appears in both {index.locations[name]} and {path}")
                index.locations[name] = path
                slice_info = handle.get_slice(name)
                index.dtypes[name] = str(slice_info.get_dtype())
                index.shapes[name] = list(slice_info.get_shape())
    return index


class StreamingSource:
    """按需从 safetensors 分片读单个张量。

    实现 :class:`~weights.TensorSource` 协议，所以 ``weights.plan`` /
    ``weights.load_into`` 不需要为它改一行——它们的测试用 ``DictSource``，线上用
    本类，走的是同一条装载逻辑。

    Attributes:
        index: header 索引。
        device: 读出的张量落到哪个设备；``None`` 表示保持 CPU。
    """

    def __init__(self, model_dir: str | Path, device: torch.device | str | None = None,
                 index: ShardIndex | None = None) -> None:
        """建索引（或复用调用方已经建好的那份）。

        Args:
            model_dir: 权重目录。
            device: 目标设备；``None`` 表示 CPU。
            index: 复用已有的索引，避免多 rank 各扫一遍 header。
        """
        self.index = build_index(model_dir) if index is None else index
        self.device = None if device is None else torch.device(device)

    def keys(self):
        """列出 checkpoint 里所有张量名。"""
        return self.index.locations.keys()

    def get(self, name: str) -> torch.Tensor:
        """读出一个张量。

        Args:
            name: checkpoint 里的完整名字。

        Returns:
            该张量，已 ``contiguous``；``device`` 非空时搬到该设备。

        Raises:
            KeyError: 名字不在索引里。
        """
        from safetensors import safe_open

        path = self.index.locations.get(name)
        if path is None:
            raise KeyError(f"{name} is not in the checkpoint index")
        with safe_open(str(path), "pt") as handle:
            tensor = handle.get_tensor(name)
        # 映射出来的视图不一定连续，``copy_`` 到参数上会慢一个数量级。
        tensor = tensor.contiguous()
        if self.device is not None:
            tensor = tensor.to(self.device, non_blocking=True)
        return tensor

    def shape_of(self, name: str) -> list[int]:
        """返回 header 里记录的形状，不读数据。

        Args:
            name: 张量名。

        Returns:
            形状列表。

        Raises:
            KeyError: 名字不在索引里。
        """
        return list(self.index.shapes[name])

    def summarize(self) -> dict[str, Any]:
        """汇总分片数、张量数与按前缀分组的计数，用于日志与配置核对。

        Returns:
            形如 ``{"shards": 131, "tensors": 1658, "prefixes": {...}}`` 的字典。
        """
        prefixes: dict[str, int] = {}
        for name in self.index.locations:
            head = name.split(".", 2)
            key = ".".join(head[:2]) if len(head) > 1 else head[0]
            prefixes[key] = prefixes.get(key, 0) + 1
        return {"shards": len(self.index.shards), "tensors": len(self.index),
                "prefixes": dict(sorted(prefixes.items()))}


def assert_inventory_matches(model_dir: str | Path, inventory_path: str | Path) -> list[str]:
    """核对 header 索引与 ``inventory.json`` 完全一致。

    这是加载前最便宜的一道门：360 GB 的下载少一个分片、或者混进两个版本的权重，
    都会在 ``build_index`` 之后立刻表现为张量数或形状不符。放到"装载完再前向"去
    发现，代价是几十分钟的加载。

    Args:
        model_dir: 权重目录。
        inventory_path: ``docs/qwen4/inventory.json`` 的路径。

    Returns:
        差异描述列表；空列表表示完全一致。

    Raises:
        ImportError: 环境里没有 ``safetensors``。
    """
    index = build_index(model_dir)
    inventory = json.loads(Path(inventory_path).read_text(encoding="utf-8"))["tensors"]
    problems: list[str] = []
    missing = sorted(set(inventory) - set(index.locations))
    if missing:
        problems.append(f"{len(missing)} tensors missing from the checkpoint, "
                        f"e.g. {missing[:3]}")
    extra = sorted(set(index.locations) - set(inventory))
    if extra:
        problems.append(f"{len(extra)} unexpected tensors, e.g. {extra[:3]}")
    mismatched = [name for name in sorted(set(inventory) & set(index.locations))
                  if list(inventory[name]["shape"]) != index.shapes[name]]
    if mismatched:
        problems.append(f"{len(mismatched)} shape mismatches, e.g. {mismatched[:3]}")
    return problems


__all__ = [
    "SHARD_SUFFIX",
    "ShardIndex",
    "StreamingSource",
    "assert_inventory_matches",
    "build_index",
]
