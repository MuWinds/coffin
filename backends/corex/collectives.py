"""为单进程四卡张量并行提供主机暂存归约。

Module: ``coffin/backends/corex``。由模型并行调度调用，只依赖 torch；不依赖
NCCL、Gloo 或跨卡算术。

Constraints:
    * 各卡在自己的当前流上排队，CPU 读取前等待下载事件；上传完成前不得复用暂存区。
    * 主机以 fp32 累加并只窄化一次，避免 bf16 大数抵消时丢失其他 rank 的贡献。
    * 同一实例由模型线程串行调用，不允许 CUDA graph 捕获。
"""

from __future__ import annotations

from collections import OrderedDict

import torch


class HostAllReduce:
    """复用 pinned 缓冲区，把同形状的各卡贡献求和并回传各卡。"""

    def __init__(self, devices: list[int], cache_entries: int = 2) -> None:
        """固定 rank 顺序，并限制暂存区缓存的形状数量。

        Args:
            devices: 按 rank 排列的逻辑 CUDA 设备编号，不可重复。
            cache_entries: 保留多少种形状的 pinned 暂存区。

        Raises:
            ValueError: 设备为空、重复或缓存容量小于一。
        """
        if not devices or len(set(devices)) != len(devices) or cache_entries < 1:
            raise ValueError("distinct devices and positive cache_entries required")
        self.devices = tuple(torch.device("cuda", d) for d in devices)
        self.cache_entries = cache_entries
        self._cache = OrderedDict()

    @staticmethod
    def _wait_uploads(entry: dict) -> None:
        """异步 H2D 仍可能读取 pinned 内存，复用或淘汰前必须等它结束。"""
        for event in entry["uploaded"]:
            event.synchronize()
        entry["uploaded"].clear()

    def all_reduce(self, tensors: list[torch.Tensor],
                   out: list[torch.Tensor] | None = None) -> list[torch.Tensor]:
        """求各 rank 的和，结果排在每张卡调用时的当前流上。

        Args:
            tensors: 与设备列表同序的 bf16/fp16/fp32 张量，形状及 dtype 相同。
            out: 可选的逐卡输出；允许与输入别名，下载完成后才写回。

        Returns:
            逐卡结果；后续消费须使用各卡同一当前流，或自行建立流依赖。

        Raises:
            ValueError: 输入数量、设备、形状、dtype 或输出契约不一致。
        """
        if len(tensors) != len(self.devices):
            raise ValueError("one tensor per rank required")
        first = tensors[0]
        if first.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError("floating point reduction required")
        for device, tensor in zip(self.devices, tensors):
            if (tensor.device != device or tensor.shape != first.shape
                    or tensor.dtype != first.dtype):
                raise ValueError("rank device, shape or dtype mismatch")
        if out is None:
            out = [torch.empty_like(t) for t in tensors]
        if len(out) != len(tensors) or any(
                o.shape != t.shape or o.dtype != t.dtype or o.device != t.device
                for o, t in zip(out, tensors)):
            raise ValueError("output must match each rank input")
        key = (tuple(first.shape), first.dtype)
        entry = self._cache.pop(key, None)
        if entry is None:
            if len(self._cache) >= self.cache_entries:
                _, old = self._cache.popitem(last=False)
                self._wait_uploads(old)
            entry = {
                "incoming": [torch.empty(first.shape, dtype=first.dtype, pin_memory=True)
                             for _ in tensors],
                "sum": torch.empty(first.shape, dtype=torch.float32, pin_memory=True),
                "result": torch.empty(first.shape, dtype=first.dtype, pin_memory=True),
                "uploaded": [],
            }
        self._cache[key] = entry
        self._wait_uploads(entry)
        downloaded = []
        for device, tensor, host in zip(self.devices, tensors, entry["incoming"]):
            with torch.cuda.device(device):
                host.copy_(tensor, non_blocking=True)
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(device))
                downloaded.append(event)
        for event in downloaded:
            event.synchronize()
        total = entry["sum"]
        total.copy_(entry["incoming"][0])
        for host in entry["incoming"][1:]:
            total.add_(host)
        entry["result"].copy_(total)
        for device, target in zip(self.devices, out):
            with torch.cuda.device(device):
                target.copy_(entry["result"], non_blocking=True)
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(device))
                entry["uploaded"].append(event)
        return out

    def synchronize(self) -> None:
        """等待本实例尚未完成的上传，供计时、释放或外部换流前调用。"""
        for entry in self._cache.values():
            self._wait_uploads(entry)
