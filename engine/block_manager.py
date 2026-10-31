"""块的归属：分页池、它的引用计数和基数前缀树。

Module: ``coffin/``（顶层），由 ``llm_engine`` 和 ``prefix_engine`` 使用。池
（``paged_cache.PagedKVCache``）负责发放和共享块；基数树
（``radix_cache.RadixCache``）知道哪些前缀值得留。本对象同时持有两者，回答
调度器真正在意的问题：这个请求能不能准入、它能复用哪些块、它马上要释放的
块还有谁拿着。

Constraints:
    * 块 id 只有在被基数树或某个 ``Sequence.block_table`` 引用时（或两者都有）
      才有效。``free`` 丢掉的是池里的那份引用，对树仍在缓存的块调它会造成
      数据损坏，而不是泄漏。
    * ``assign`` 只重写池里的槽位表，不取引用。调用方必须已经持有这些块。
    * 不加锁。引擎的步进循环是唯一的写者。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from cache.paged_cache import PagedKVCache
    from cache.radix_cache import RadixCache


#: 默认块粒度；运行时命令行把它暴露成 ``--block-size``。64 与快照间隔的下限
#: 一致，循环状态快照因此永远不会落在块中间。
DEFAULT_BLOCK_SIZE = 64


class BlockManager:
    """池、它的前缀树，以及两者之间的引用计数簿记。"""

    def __init__(self, pool: "PagedKVCache", radix: "RadixCache | None" = None,
                 block_size: int = DEFAULT_BLOCK_SIZE) -> None:
        """包装一个已经建好的池。

        Args:
            pool: 分页 KV 池，通常由 :meth:`attach` 调 ``model.enable_paging``
                建出来。
            radix: 建在同一池上的前缀树；批处理解码用的池不共享前缀，传
                ``None``。
            block_size: 每块的 token 数。必须与池自己的块大小一致，因为它要
                用来把 token 数换算成块数。
        """
        self.pool = pool
        self.radix = radix
        self.block_size = block_size

    @classmethod
    def attach(cls, model, num_blocks: int, block_size: int = DEFAULT_BLOCK_SIZE,
               snapshots: int = 8, with_radix: bool = True) -> "BlockManager":
        """在 ``model`` 上建分页池，再用基数树把它包起来。

        Args:
            model: 一个 ``PagedModel``，必须带 ``enable_paging``。
            num_blocks: 池里的物理块数。``-1`` 由配置层解析，不在这里处理。
            block_size: 每块的 token 数。
            snapshots: 交给基数树的循环状态快照预算。
            with_radix: 传 ``False`` 建的是批处理解码用的池——存储和引用计数
                一样，但不做前缀复用。在有按序列前缀之前，批调度器自己分配
                序列的块。

        Returns:
            持有新建池的管理器。
        """
        from cache.radix_cache import RadixCache

        pool = model.enable_paging(num_blocks, block_size)
        radix = RadixCache(pool, snapshots) if with_radix else None
        return cls(pool, radix, block_size)

    # 按序列分配，批处理路径使用。
    def ensure_blocks(self, seq, num_tokens: int) -> list[int]:
        """分配块，直到 ``seq`` 放得下 ``num_tokens`` 个位置。

        只增不减。在这里缩减会把一条还在解码的序列的尾巴丢掉，所以释放是另一个
        单独的调用。

        Args:
            seq: ``block_table`` 被就地扩展的序列。
            num_tokens: 序列必须能容纳的位置数。

        Returns:
            该序列的块表，按需扩展过。

        Raises:
            OutOfBlocks: 池空了。调用方必须先淘汰前缀缓存，还不够就抢占一条
                序列；这个请求没法在这里长大。
        """
        need = (num_tokens + self.block_size - 1) // self.block_size
        while len(seq.block_table) < need:
            seq.block_table.append(self.pool.alloc())
        return seq.block_table

    def release(self, seq) -> None:
        """丢掉 ``seq`` 在它持有的每个块上的引用。

        树对自己缓存的东西另有一份引用，所以被抢占的序列调完这个，按块对齐的
        前缀仍然能被下一次准入复用。没有别的持有者的块回到池里。

        Args:
            seq: 正在退休或被抢占的序列。它的块表会被清空，所以再调一次不会
                重复释放同一批 id。
        """
        self.free(seq.block_table)
        seq.block_table.clear()

    def reserve(self, blocks_needed: int) -> None:
        """淘汰缓存前缀，直到空出 ``blocks_needed`` 个块。

        Args:
            blocks_needed: 下一次预填充需要的块数。必须在*分配之前*调用，这样
                淘汰时还有无人引用的叶子可挑；分配之后，请求自己就占住了唯一
                的候选。
        """
        if self.radix is not None:
            self.radix.reserve(blocks_needed)

    def match_prefix(self, token_ids: list[int], require_states: bool = False,
                     mm=None):
        """``token_ids`` 里最长的已缓存块前缀。

        Args:
            token_ids: 完整的请求 token。
            require_states: 要求命中点处有循环状态快照（混合模型上这是必须的，
                那里光有 KV 不够）。
            mm: 请求的图像载荷（``mm_input.MMInput | None``）。基数树拿它给
                块键加盐，所以带*不同*图像的 prompt 无法复用这段前缀的 KV。
                ``None`` 是纯文本路径，不额外花代价。

        Returns:
            ``(blocks, length, states)``；没挂前缀树时是 ``([], 0, None)``。
        """
        if self.radix is None:
            return [], 0, None
        return self.radix.match_prefix(token_ids, require_states=require_states,
                                       mm=mm)

    def insert(self, token_ids: list[int], blocks: list[int], states,
               mm=None) -> None:
        """把一段已预填充的前缀记进树里。

        Args:
            token_ids: 这段前缀覆盖的 token，按块对齐。
            blocks: 存着该前缀 KV 的物理块。
            states: 块边界处的循环状态快照，没打到任何快照时为 ``None``。
            mm: 用来给键加盐的媒体载荷；必须与之后调用 :meth:`match_prefix`
                时传的一致。
        """
        if self.radix is not None:
            self.radix.insert(token_ids, blocks, states, mm=mm)

    def assign(self, blocks: list[int], row: int = 0) -> None:
        """把池里第 ``row`` 行的槽位表指向 ``blocks``。

        ``row == 0`` 是单流路径，直接省掉行参数，热解码循环因此不必多付一次
        分派。

        Args:
            blocks: 要映射的块 id。调用方必须已经持有引用。
            row: 映射到哪一行批槽位。
        """
        if row == 0:
            self.pool.assign_table(blocks, 0)
        else:
            self.pool.assign_table(blocks, 0, row=row)

    def free(self, blocks: list[int]) -> None:
        """把块还给池。

        Args:
            blocks: 要释放的 id。不能包含基数树仍在缓存的块——那种块要通过树
                去淘汰。
        """
        self.pool.free_blocks(blocks)

    def clear_prefix_cache(self) -> None:
        """丢掉所有缓存前缀。

        A/B 冷启动测量用它，需要已知冷引擎但不想重建池的测试也用它。
        """
        if self.radix is not None:
            self.radix.clear()

    @property
    def num_blocks(self) -> int:
        return self.pool.num_blocks

    @property
    def num_used(self) -> int:
        return self.pool.num_used

    @property
    def num_free(self) -> int:
        return self.pool.num_free

    def stats(self) -> dict:
        """前缀缓存的计数器，运行时 ``ping`` 上报的就是这些。

        Returns:
            块总数总是有；基数树的计数器只在挂了树时才有。``snapshot_mib``
            保留一位小数，因为看它的人是在盯 16 GB 预算。
        """
        out = {"blocks_used": self.num_used, "blocks_total": self.num_blocks}
        if self.radix is not None:
            out.update({
                "cached_blocks": self.radix.num_cached_blocks(),
                "snapshots": self.radix.num_snapshots(),
                "snapshot_mib": round(self.radix.snapshot_bytes() / 2**20, 1),
                "evicted_blocks": self.radix.num_evicted_blocks,
            })
        return out
