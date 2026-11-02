"""四卡同机的进程间归约：共享内存实现，替掉 gloo。

实测（`diag/qwen4/probe_shm_allreduce.py`，作业 4410，卡 3/4/5/6）：

| 实现 | `[1, 2560]` fp32 单次 | 48 层折算 | decode 上限 |
| --- | --- | --- | --- |
| gloo `all_reduce` | **291–420 µs** | 14.0–20.2 ms | 50–72 tok/s |
| 共享内存 barrier | **32.9 µs** | 1.6 ms | 633 tok/s |

差 8.9 倍。原因是机制选错了：四个 rank 全在同一台机器上，走 TCP loopback 做同机
进程间通信要经过两次系统调用、两次 skb 拷贝和 Nagle 定时器，而共享内存是每个
rank 直接写对方可见的缓存行。

**这个结论改变并行拓扑的选择。** 原本按 232 µs/次 算，TP=4 每层归约一次就是
11 ms、每层两次是 22 ms，会逼着去选 EP 来躲通信；换成 33 µs 之后，每层归约一次
是 1.6 ms，**通信不再是决策变量**，拓扑可以纯粹按「哪个算得快」来挑。

Module: ``coffin/models/qwen4exp``。模型族内的并行原语。依赖只有 ``torch`` 和
标准库——它不碰 ``backends``，因为这条路径只交换 CPU 上的 fp32，与平台无关；
``blocks.py`` / ``model.py`` 在构造时按 rank 拿到它，单卡时传 ``None``。

Constraints:
    * **每个 rank 只写自己的格子，绝不共享一个计数器。** 第一版用「四个 rank 都对
      一个 int64 读-改-写」当 barrier，实测直接死锁：读-改-写不是原子的，两个 rank
      会读到同一个值再都写回 ``cur+1``，计数器永远到不了 ``world * round``。
    * **绝不在这里 ftruncate 到 0。** 文件由调用方在任何进程启动之前建好清好；
      在这里截断会让四个进程互相拆台，报 ``mmap length is greater than file size``，
      而就算侥幸建好也会把别人写进去的数据清零。
    * 自旋有上限。一个 rank 挂了其余会永久空转，所以到点就抛 ``TimeoutError``，
      让外面的 ``timeout`` 收尸——绝不无限等。
    * 成本与 payload 的关系只在 rows<=8 时近似常数（32.9 → 65.5 µs），说明它仍是
      「次数受限」的；rows=64 时 Python 层 4 次全量 copy 吃掉 188 ms，那是调用方的
      开销不是通信的。**大 payload 不要直接拿这个类去 reduce 一个 prefill chunk。**
    * 没有序列号，所以各 rank 必须以相同顺序调用；顺序错了会把两轮的数据混在一起
      而不是报错。
"""

from __future__ import annotations

import mmap
import os
import struct
import time

import torch

#: 归约的宽度。qwen4_exp 的 ``hidden_size``，MoE 的 combine 归约的就是它。
HIDDEN = 2560


class ShmAllReduce:
    """若干 rank 通过一块 mmap 做全归约，语义同 ``dist.all_reduce(SUM)``。

    Args:
        path: mmap 文件路径。必须由调用方在**任何进程启动之前**创建并清空。
        rank: 本进程编号。
        world: 进程数。
        rows: 每次归约的行数，``[rows, 2560]`` fp32。

    Raises:
        ValueError: ``rows`` 或 ``world`` 非正。
    """

    HEADER = 64

    def __init__(self, path: str, rank: int, world: int, rows: int = 1) -> None:
        """打开共享板，为自己的 rank 建好写入口。

        各 rank 的槽位在构造时就算好偏移并缓存成 ``memoryview``：归约热路径上
        每轮要读 ``world`` 次，临时切片的开销在这个尺度上是可观的。

        Args:
            path: mmap 文件路径。调用方必须在起任何进程之前创建并清零。
            rank: 本进程编号，决定写哪一格、以及哪一格的到达计数归自己。
            world: 进程数。
            rows: 每次归约的行数，``[rows, 2560]`` fp32。

        Raises:
            ValueError: ``rows`` 或 ``world`` 非正。
        """
        self.rank = rank
        self.world = world
        self.rows = rows
        self.slot_elems = rows * HIDDEN
        self.slot_bytes = self.slot_elems * 4
        self.total = self.HEADER + world * self.slot_bytes
        self.round = 0
        self._fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
        # 只保证「至少这么大」，绝不缩到 0：清空由调用方在起进程之前做完，
        # 那是唯一没有并发者的时刻。
        if os.fstat(self._fd).st_size < self.total:
            os.ftruncate(self._fd, self.total)
        self._mm = mmap.mmap(self._fd, self.total)
        self._arrivals = memoryview(self._mm)[:world * 8]
        # 只读视图与只写视图分开，避免反复切片产生临时对象。
        self._slots = [memoryview(self._mm)[self.HEADER + i * self.slot_bytes:
                                              self.HEADER + (i + 1) * self.slot_bytes]
                       for i in range(world)]
        # 复用一个临时张量承接每个槽位，避免每次归约分配 world 次。
        self._stage = torch.empty(rows, HIDDEN, dtype=torch.float32)
        self._stage_view = memoryview(self._stage.numpy()).cast("B")

    def __call__(self, tensor: torch.Tensor, out: torch.Tensor,
                 *, spin_limit: float = 20.0) -> torch.Tensor:
        """把各 rank 的 ``tensor`` 求和写进所有 rank 的 ``out``。

        Args:
            tensor: 本 rank 的 fp32 CPU 张量，形状 ``[rows, 2560]``，连续。
            out: 结果写入的张量，形状同 ``tensor``，会被覆盖。
            spin_limit: 自旋等待上限（秒）。

        Returns:
            ``out``，便于链式调用。

        Raises:
            TimeoutError: 某个 rank 在 ``spin_limit`` 内没有到达。
            ValueError: 形状或 dtype 与构造时不一致。
        """
        if tensor.shape != (self.rows, HIDDEN) or tensor.dtype != torch.float32:
            raise ValueError(
                f"expected [{self.rows}, {HIDDEN}] fp32, got {tuple(tensor.shape)} "
                f"{tensor.dtype}")

        src = memoryview(tensor.contiguous().numpy()).cast("B")
        self._slots[self.rank][:len(src)] = src

        off = self.rank * 8
        cur = struct.unpack_from("<q", self._arrivals, off)[0]
        struct.pack_into("<q", self._arrivals, off, cur + 1)
        self.round += 1

        want = self.round * self.world
        deadline = time.perf_counter() + spin_limit
        while True:
            total = 0
            for i in range(self.world):
                total += struct.unpack_from("<q", self._arrivals, i * 8)[0]
            if total >= want:
                break
            if time.perf_counter() > deadline:
                raise TimeoutError(
                    f"rank {self.rank} timed out waiting at round {self.round}")

        out.zero_()
        for i in range(self.world):
            self._stage_view[:] = self._slots[i]
            out.add_(self._stage)
        return out

    def close(self) -> None:
        """释放映射和文件描述符。"""
        try:
            self._slots.clear()
            self._arrivals.release()
            self._mm.close()
        finally:
            os.close(self._fd)
