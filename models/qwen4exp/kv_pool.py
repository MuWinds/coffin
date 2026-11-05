"""qwen4exp 的块记账池：给 ``BlockManager`` / ``RadixCache`` / ``PrefixCacheEngine``
用的适配器。

## 为什么不是直接用 ``cache.paged_cache.PagedKVCache``

那个池持有 int8 KV 张量，模型族的注意力靠 ``pos2slot`` 槽位表去读它，是 qwen35
那套「内核直接读共享池」的契约。qwen4exp 的 QSA 现在用的是另一套：
:class:`~models.qwen4exp.qsa.PagedAttentionCache` 在模型内部按块预分配
``[B, kv_heads, capacity, head_dim]`` 的 bf16 池，注意力走连续视图，indexer 也有
自己的一块池。把三块池都改成「按槽位表跳着读」需要改注意力内核与索引器内核，
是另一轮工作。

本适配器因此只实现**块生命周期与前缀树记账**这一层接口（``alloc`` /
``incref`` / ``decref`` / ``assign_table`` / ``num_free`` …），让调度侧的
``RadixCache``、``BlockManager``、``PrefixCacheEngine`` 能原样挂上来。KV 内容仍由
模型自己的池持有，前缀命中时通过 :meth:`ModelStates.snapshot` 恢复整份状态。

## 为什么 ``copy_block`` 可以只记账

``SeqKV.reserve`` 在「缓存前缀的尾块仍被树持有」时会调 ``copy_block`` 做写时复制，
防止新请求覆写共享的 KV。这里不需要：状态快照是**整份深拷贝**（GDN 递推/卷积、
PLE 上下文/卷积、QSA 的 K/V/索引池全部 ``clone``），树里存的快照与正在前向的状态
物理隔离，命中后继续写不会碰到树里那一份。所以 COW 退化成一次 ``n_cows`` 记账，
正确性由快照的深拷贝保证，而不是由块复制保证。

Module: ``coffin/models/qwen4exp``。只被 ``Qwen4Engine.enable_paging`` 构造，
不 import 引擎层，依赖方向保持单向。

Constraints:
    * 不分配 KV 张量，也不持有模型引用；这里只有整数块 id 与一张兼容用的
      ``pos2slot`` 表。真正的 KV 容量由模型的 ``max_seq`` 决定，两者必须一致。
    * 不加锁：引擎的步进循环是唯一写者（与 ``paged_cache`` 相同约定）。
"""

from __future__ import annotations

import torch


class OutOfBlocks(RuntimeError):
    """池里已经没有无人引用的块可以发出去了。"""


class Qwen4KVPool:
    """块记账池，实现 ``PagedKVCache`` 在调度侧被用到的那一片接口。

    Attributes:
        bs: 每块的 token 数，同时也是前缀匹配的粒度。
        max_seq: 槽位表能寻址的最大位置，等于 ``num_blocks * bs``。
        num_blocks: 物理块数。
        n_cows: 写时复制次数（这里只记账，不搬数据，理由见文件头）。
    """

    def __init__(self, num_blocks: int, block_size: int = 64,
                 device: torch.device | str = "cpu") -> None:
        """开一张空闲块表与引用计数表。

        Args:
            num_blocks: 物理块数。
            block_size: 每块的 token 数。
            device: ``pos2slot`` 兼容表所在的设备。

        Raises:
            ValueError: 块数或块大小非正。这是显存/位置预算的承诺，不该静默接受 0。
        """
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError(
                f"KV pool needs positive sizes, got num_blocks={num_blocks} "
                f"block_size={block_size}")
        self.bs = int(block_size)
        self.num_blocks = int(num_blocks)
        self.max_seq = self.num_blocks * self.bs
        self.free: list[int] = list(range(self.num_blocks - 1, -1, -1))
        self.refcount: list[int] = [0] * self.num_blocks
        self.n_cows = 0
        self.n_allocs = 0
        #: 与 ``PagedKVCache`` 同形状的兼容表。qwen4exp 的注意力暂时按连续位置读
        #: 自己的池，不查这张表；留着它是为了让只依赖 ``assign_table`` 记账的调用
        #: 方（前缀缓存、批块表）行为与原池一致，将来接槽位寻址的内核时不必再改。
        self.pos2slot = torch.zeros(1, self.max_seq, dtype=torch.int64, device=device)

    @property
    def num_free(self) -> int:
        """当前无人引用的块数。"""
        return len(self.free)

    @property
    def num_used(self) -> int:
        """已被树或序列持有的块数。"""
        return self.num_blocks - len(self.free)

    @property
    def rows(self) -> int:
        """槽位表的行数（单流为 1）。"""
        return self.pos2slot.shape[0]

    def set_rows(self, rows: int) -> None:
        """把兼容槽位表扩缩到 ``rows`` 行，第 0 行内容保留。

        Args:
            rows: 至少 1 行；与当前相同时是空操作。
        """
        rows = max(int(rows), 1)
        if rows == self.rows:
            return
        new = torch.zeros(rows, self.max_seq, dtype=torch.int64,
                          device=self.pos2slot.device)
        keep = min(rows, self.rows)
        new[:keep].copy_(self.pos2slot[:keep])
        self.pos2slot = new

    def alloc(self) -> int:
        """取一个无人引用的块发出去，引用计数置 1。

        Returns:
            块 id。

        Raises:
            OutOfBlocks: 没有空闲块。调用方应先淘汰缓存前缀再重试。
        """
        if not self.free:
            raise OutOfBlocks(
                f"no free KV blocks (used {self.num_used}/{self.num_blocks})")
        block = self.free.pop()
        self.refcount[block] = 1
        self.n_allocs += 1
        return block

    def incref(self, block: int) -> None:
        """给已缓存的块再加一份引用（匹配前缀时由基数树调用）。"""
        self.refcount[block] += 1

    def decref(self, block: int) -> None:
        """丢掉一份引用，归零时把块还回空闲表。"""
        self.refcount[block] -= 1
        if self.refcount[block] == 0:
            self.free.append(block)

    def free_blocks(self, blocks: list[int]) -> None:
        """对一串块各丢一份引用。

        Args:
            blocks: 调用方持有的块 id 列表。
        """
        for block in blocks:
            self.decref(block)

    def copy_block(self, src: int, dst: int) -> None:
        """写时复制钩子：这里只记账，不搬数据。

        Args:
            src: 被共享的源块。
            dst: 新分配的私有块。
        """
        del src, dst                      # 快照是深拷贝，没有共享 KV 需要复制
        self.n_cows += 1

    def assign(self, block: int, start: int, row: int = 0) -> None:
        """把 ``[start, start+bs)`` 的位置映射到 ``block`` 的槽位。

        Args:
            block: 已持有引用的块 id。
            start: 该块起始的 token 位置。
            row: 兼容槽位表的行。
        """
        lo, hi = start, min(start + self.bs, self.max_seq)
        if hi <= lo:
            return
        self.pos2slot[row, lo:hi] = block * self.bs + torch.arange(
            hi - lo, device=self.pos2slot.device)

    def assign_table(self, blocks: list[int], start: int = 0, row: int = 0) -> None:
        """把一串连续的块按块对齐铺进某一行槽位表。

        Args:
            blocks: 一条序列的块表，按位置顺序排列。
            start: 第一个块的起始位置。
            row: 要重写的槽位表行。
        """
        for index, block in enumerate(blocks):
            self.assign(block, start + index * self.bs, row)


__all__ = ["OutOfBlocks", "Qwen4KVPool"]
