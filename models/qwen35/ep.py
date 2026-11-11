"""同机多进程的全归约：一块 mmap 加自旋到达计数，语义同 ``dist.all_reduce(SUM)``。

Module: ``coffin/models/qwen35``。EP 的通信底座：``moe.py`` 每个 MoE 层调一次
``all_reduce_sum``，除此之外没有调用方。它只认 torch 和标准库，不 import
``backends`` / ``models`` 里的任何东西——后端换来换去，这一层不该跟着动。

Constraints:
    * 所有 rank 必须以**完全相同的顺序**调用同一个实例。协议里没有序号校验（省一次读），
      调用错开时不报错，只是把两轮的数据混在一起。
    * 一个板文件只给一个实例用。两个实例共用一个文件时，它们往同一组 arrival 格子上各写
      自己的 round，某一个 rank 的 ``want`` 会被另一块板提前满足，于是读到别人那一轮的数据
      ——不报错，只是数值错。载荷大小不同就开不同的文件。
    * 每个 rank 只写自己那一格计数，绝不读-改-写共享计数器：那组操作不是原子的，
      qwen4exp 的第一版就是四个 rank 读到同一个值再都写回 ``cur+1``，计数器永远到不了
      ``world * round``，实测直接死锁。
    * ``__init__`` 绝不 ftruncate。文件由启动器在**任何**进程启动之前建好：并发 ftruncate
      会把别人刚建好的文件截到 0，随后报文 ``mmap length is greater than file size``。
    * 自旋有上限，到点抛 ``TimeoutError``。一个 rank 挂了，其余的不该永久空转——那样外面
      的 ``timeout`` 连尸都收不到。
    * 槽里一律是 fp32。bf16 四个 rank 直接相加要丢两三位尾数，而这里省下的只是窄化那一次
      拷贝，不值得。
    * 可见性只保证 x86-64：先写载荷再写自己的到达计数，靠 TSO 的写序让对端「看到计数就
      一定看到载荷」。换到弱序架构（ARM）要在两次写之间补一条内存屏障。
"""

from __future__ import annotations

import gc
import mmap
import os
import struct
import time

import torch


class EpShmReduce:
    """四进程共享内存全归约，``[1, 2560]`` 一次往返在几十微秒量级。"""

    def __init__(self, rank: int, world: int, path: str, payload_bytes: int) -> None:
        """打开已存在的共享板，并算好自己的槽位偏移。

        Args:
            rank: 本进程编号，决定写哪一格计数、哪一个槽。
            world: 进程数，也决定头部要几个 arrival 格子。
            path: 共享板文件路径。调用方必须在起任何进程之前建好并清零，长度至少
                ``(world + 1) * 8 + world * payload_bytes``。
            payload_bytes: 每个 rank 的载荷区容量，不是四个 rank 的合计。

        Raises:
            ValueError: ``world`` 非正、``rank`` 越界或 ``payload_bytes`` 非正。
            RuntimeError: 文件不存在，或比需要的短。这里绝不自己扩容：并发扩容正是
                上面那条死锁的同源问题。
        """
        if world <= 0:
            raise ValueError(f"world must be positive, got {world}")
        if not 0 <= rank < world:
            raise ValueError(f"rank {rank} out of range [0, {world})")
        if payload_bytes <= 0:
            raise ValueError(f"payload_bytes must be positive, got {payload_bytes}")

        self.rank = int(rank)
        self.world = int(world)
        self.path = path
        self.payload_bytes = int(payload_bytes)
        #: 头部 = world 个到达计数 + 一个 epoch，全部 int64。
        self.header_bytes = (self.world + 1) * 8
        self.total_bytes = self.header_bytes + self.world * self.payload_bytes
        #: 自旋上限（秒）。给得比一次 decode 步大三个数量级，只用来把死锁变成异常。
        self.spin_limit = 20.0
        self._round = 0
        self._calls = 0
        self._closed = False
        self._views: dict[tuple[int, int, int], tuple[torch.Tensor, ...]] = {}
        self._acc: torch.Tensor | None = None

        try:
            self._fd = os.open(path, os.O_RDWR)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"共享板不存在：{path}。启动器要先建好 {self.total_bytes} 字节") from exc
        size = os.fstat(self._fd).st_size
        if size < self.total_bytes:
            os.close(self._fd)
            raise RuntimeError(
                f"共享板太短：需要 {self.total_bytes} 字节"
                f"（头部 {self.header_bytes} + {self.world}×{self.payload_bytes}），"
                f"实际 {size} 字节")
        self._mm = mmap.mmap(self._fd, self.total_bytes)
        # 头部只取一次视图：自旋里每轮要按 rank 读 world 次，临时切片在这个尺度上是噪声。
        self._head = memoryview(self._mm)[:self.header_bytes]

    def _slot_views(self, slots: int, numel: int, index: int) -> tuple[torch.Tensor, ...]:
        """取本轮各 rank 槽位在本进程的零拷贝视图，按 rank 号排。

        Args:
            slots: 每个 rank 的载荷区切成几块。
            numel: 本次载荷的元素数（fp32）。
            index: 本轮用第几块，由调用次数对 ``slots`` 取模。

        Returns:
            长度为 ``world`` 的元组，第 ``r`` 项是 rank ``r`` 那一段 ``[numel]`` fp32。
        """
        key = (slots, numel, index)
        cached = self._views.get(key)
        if cached is not None:
            return cached
        slot_bytes = self.payload_bytes // slots
        first = self.header_bytes + index * slot_bytes
        views = tuple(
            torch.frombuffer(self._mm, dtype=torch.float32, count=numel,
                             offset=first + r * self.payload_bytes)
            for r in range(self.world))
        # 槽数被载荷撑开时缓存会无界增长，最多留八组：decode 只会用到一两组。
        if len(self._views) >= 8:
            self._views.pop(next(iter(self._views)))
        self._views[key] = views
        return views

    def _accumulator(self, numel: int) -> torch.Tensor:
        """取本进程的 fp32 累加缓冲，不够长就换一条更大的。

        Args:
            numel: 需要的元素数。

        Returns:
            ``[numel]`` 的 fp32 CPU 张量，复用上一次的分配。
        """
        acc = self._acc
        if acc is None or acc.numel() < numel:
            acc = torch.empty(numel, dtype=torch.float32)
            self._acc = acc
        return acc[:numel]

    def all_reduce_sum(self, x: torch.Tensor, slots: int | None = None) -> torch.Tensor:
        """把所有 rank 的 ``x`` 求和，返回同 dtype / shape / device 的新张量。

        Args:
            x: 本 rank 的贡献，浮点张量，CPU 或 CUDA 都行，非连续也可（会先拉平）。
            slots: 每个 rank 的载荷区切成几块轮转使用。省略时按容量推导，取 2 到 4 块：
                只有两块时上一轮的读和这一轮的写靠奇偶分开，更多块只是让不同步长的调用
                不撞同一段，再往上只浪费容量。

        Returns:
            求和结果，形状与 dtype 同 ``x``，device 也同 ``x``。是新分配的张量，调用方
            可以一直留着——下一次调用不会覆盖它。

        Raises:
            ValueError: ``x`` 不是浮点、``slots`` 非正、或载荷超出单槽容量。
            RuntimeError: 实例已 ``close()``。
            TimeoutError: 某个 rank 在 ``spin_limit`` 内没有到达。
        """
        if self._closed:
            raise RuntimeError("shared board already closed")
        if not x.is_floating_point():
            raise ValueError(f"all_reduce_sum needs floats, got {x.dtype}")

        numel = x.numel()
        if numel == 0:
            # 空张量也要走完两次 barrier，否则本 rank 和别人的调用序列错开一格。
            self.barrier()
            self.barrier()
            return x.clone()

        need = numel * 4
        if slots is None:
            slots = max(2, min(4, self.payload_bytes // need))
        slots = int(slots)
        if slots <= 0:
            raise ValueError(f"slots must be positive, got {slots}")
        slot_bytes = self.payload_bytes // slots
        if slot_bytes < need:
            raise ValueError(
                f"载荷 {need} 字节放不进 {slots} 个槽（每槽 {slot_bytes} 字节）："
                f"payload_bytes 至少要 {2 * need}，当前 {self.payload_bytes}")

        index = self._calls % slots
        self._calls += 1
        views = self._slot_views(slots, numel, index)
        # 写自己的槽。``copy_`` 顺便把 dtype（bf16→fp32）和 device（CUDA→CPU）一次转完，
        # 所以这里没有额外的一次中转。
        views[self.rank].copy_(x.reshape(-1))
        self.barrier()

        acc = self._accumulator(numel)
        # 按 rank 号顺序累加、只窄化一次：bf16 逐次累加在 2560 维上会偏。第一次用
        # ``out=`` 直接落地，省掉一次独立的拷贝调度——这里每次调度在 CPU 上是微秒级。
        if self.world == 1:
            acc.copy_(views[0])
        else:
            torch.add(views[0], views[1], out=acc)
            for slot in views[2:]:
                acc.add_(slot)
        # 第二次 barrier 不是冗余：没有它，跑得快的 rank 会在别人还没读完时写下一轮。
        self.barrier()

        out = torch.empty_like(x)
        out.copy_(acc.view(x.shape))
        return out

    def barrier(self) -> None:
        """让所有 rank 在这里对齐一次。

        Raises:
            TimeoutError: 某个 rank 在 ``spin_limit`` 内没有到达。
            RuntimeError: 实例已 ``close()``。
        """
        if self._closed:
            raise RuntimeError("shared board already closed")
        self._round += 1
        # 只写自己那一格：共享计数器上的读-改-写不是原子的，两个 rank 会互相覆盖。
        struct.pack_into("<q", self._head, self.rank * 8, self._round)
        want = self._round
        deadline = time.perf_counter() + self.spin_limit
        while True:
            behind = -1
            for i in range(self.world):
                if struct.unpack_from("<q", self._head, i * 8)[0] < want:
                    behind = i
                    break
            if behind < 0:
                break
            if time.perf_counter() > deadline:
                epoch = struct.unpack_from("<q", self._head, self.world * 8)[0]
                raise TimeoutError(
                    f"rank {self.rank} 在 round {want} 等 rank {behind} 没到；"
                    f"rank 0 停在 epoch {epoch}")
        if self.rank == 0:
            # 只给超时诊断看：它回答「队列是从第几轮开始没动的」，这是区分「别人卡住」
            # 和「整队根本没起来」的唯一线索。
            struct.pack_into("<q", self._head, self.world * 8, want)

    def close(self) -> None:
        """释放映射与文件描述符。可重复调用。"""
        if self._closed:
            return
        self._closed = True
        self._views.clear()
        self._acc = None
        # ``torch.frombuffer`` 的视图持有 mmap 的导出指针，引用计数掉下来才关得掉。
        # 关不掉就交给进程退出收尸，绝不在 close 里抛异常打断收尾。
        gc.collect()
        try:
            self._head.release()
        except (BufferError, ValueError):
            pass
        try:
            self._mm.close()
        except BufferError:
            # 关不掉就交给进程退出收尸：在这里抛只会打断收尾，而映射本来就是进程级资源。
            pass
        finally:
            os.close(self._fd)
