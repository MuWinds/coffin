"""带块共享和写时复制的分页 KV 存储。

Module: ``coffin/``（顶层，不感知模型）。``radix_cache`` 在这里发出的块上给前缀
定键，两者都归 ``block_manager`` 持有；模型族的注意力代码只通过槽位表访问池。
本模块不许 import 模型族，这里也不能有任何地方假定分词器或层数。

解码时的注意力窗口是序列不断增长的前缀，所以每层的 KV 缓存不必是一整块连续的
``[nkv, max_seq, hd]`` 缓冲区。定长块换来了稠密布局做不到的两件事：共享同一段
prompt 前缀的两条序列指向*同一批*物理块（引用计数 + 写时复制，而不是防守性
拷贝，这也正是 :mod:`radix_cache` 存在的价值）；结束的序列把块还给池，KV 显存
因此跟着活着的序列走，而不是 ``max_seq`` 乘以并发序列数。

模型所有的全注意力层共用一个 int8 池 ``[L, num_slots, nkv, hd]``，每个（头,
token）配一个 fp32 scale——和稠密缓存用的是同一套 int8 方案，反量化的注意力内核
除了寻址行那一处之外完全不用改。一条序列拥有自己的*槽位表*：每个绝对位置对应
一个 int64 池槽位，以 ``pos2slot`` 的形式留在设备上。块 ``b`` 覆盖位置
``[b*bs, (b+1)*bs)``，占用槽位 ``[b*bs, (b+1)*bs)``，所以
``slot = block*bs + (pos % bs)``。

Constraints:
    * int8 布局是与注意力内核之间的契约，不是存储细节：``k``/``v`` 是
      ``[n_layers, num_slots, nkv, hd]`` 的 int8，``kscale``/``vscale`` 是
      ``[n_layers, num_slots, nkv]`` 的 fp32，都按池槽位寻址。改 dtype、改
      scale 的秩、改轴顺序都会破坏内核的反量化，所以换后端时不许碰它。
    * ``pos2slot`` 留在设备内存里，因为解码步是 capture 成图的：注意力靠索引
      这个张量来解析位置；如果把位置当作参数在 capture 时烧进去，就得为每条
      序列、每个位置各 capture 一张图。留成张量，一张图就能服务任意序列的
      任意位置。
    * 槽位表是 ``[rows, max_seq]``，永远不是一维的。第 0 行是单流那份表，
      ``slots_for`` 和 ``assign`` 默认用它，那些调用点因此不用改；批解码给
      每个批行一份自己的表，把行视图交给同一批内核。行步长是 captured 内核
      烧进去的一部分，所以 ``capture()`` 跑过之后行数就固定了。
    * 不加锁。``llm_engine`` 的步进循环是唯一的写者，块 id 也只有被引用计数
      持有时才有效。
    * 默认设备从 ``backends.device`` 来，绝不写字面量：池是进程里最大的单笔
      分配，默认值弄错就会把它放到模型从来不读的设备上。
"""

from __future__ import annotations

import torch

import backends


class OutOfBlocks(RuntimeError):
    """池里已经没有无人引用的块可以发出去了。

    抛异常而不是返回哨兵值，是因为调用方无从恢复：分配失败时序列已经被准入，
    只能让请求失败并释放它的块。把池开到不会触发这个异常，是
    ``BlockManager.reserve`` 的职责，不是调用方的。
    """


class PagedKVCache:
    """所有全注意力层共用的 int8 块池。

    ``num_blocks`` x ``block_size`` 个槽位对*所有*序列开放，而序列的块是带引用
    计数的，所以通过基数树共享的块会一直活到最后一个持有者放手为止——这份共享
    引用正是池的意义所在，而多释放一次就会破坏某条活着的序列的 KV。
    """

    def __init__(self, n_layers: int, nkv: int, hd: int, num_blocks: int = 512,
                 block_size: int = 64, max_seq: int = 4096,
                 device: "torch.device | str | None" = None, rows: int = 1) -> None:
        """一次性把整个池开出来，这是一笔显存承诺。

        Args:
            n_layers: 共用这个池的全注意力层数；每层在每个块里各占一片。
            nkv: 每层的 key/value 头数。GQA 在这里打包，所以它比查询头数小。
            hd: 头的维度。
            num_blocks: 物理块数。``-1`` 在传到这里之前已由配置层解析掉。
            block_size: 每块的 token 数；同时也是基数匹配的粒度。
            max_seq: 槽位表能寻址的最大位置。比它更长的序列即使还有空闲块也
                表示不出来。
            device: int8 池和槽位表放在哪。``None`` 表示取
                ``backends.device``，即选中后端的设备；测试显式传 ``"cpu"``，
                这样池不需要加速卡。
            rows: 槽位表的行数。单流传 ``1``；批解码传批容量，第
                ``capacity-1`` 行是预填充行。
        """
        if device is None:
            device = backends.device
        self.n_layers = n_layers
        self.nkv = nkv
        self.hd = hd
        self.num_blocks = num_blocks
        self.bs = block_size
        self.max_seq = max_seq
        self.num_slots = num_blocks * block_size
        shape = (n_layers, self.num_slots, nkv, hd)
        self.k = torch.zeros(shape, dtype=torch.int8, device=device)
        self.v = torch.zeros(shape, dtype=torch.int8, device=device)
        self.kscale = torch.zeros((n_layers, self.num_slots, nkv), dtype=torch.float32,
                                  device=device)
        self.vscale = torch.zeros((n_layers, self.num_slots, nkv), dtype=torch.float32,
                                  device=device)

        #: 位置 -> 池槽位，每个批行一行（常驻设备）。第 0 行是单流那份表，
        #: 第 1 行起是批解码的行。
        self.pos2slot = torch.zeros(rows, max_seq, dtype=torch.int64, device=device)

        self.free: list[int] = list(range(num_blocks - 1, -1, -1))
        self.refcount: list[int] = [0] * num_blocks
        self.n_cows = 0
        self.n_allocs = 0

    @property
    def num_free(self) -> int:
        return len(self.free)

    @property
    def num_used(self) -> int:
        return self.num_blocks - len(self.free)

    def bytes_per_block(self) -> int:
        """一个块在所有全注意力层上的常驻字节数。

        其中 `2 * n` 是 int8 的 K/V 载荷（各 `bs * nkv * hd` 字节），
        `2 * bs * nkv * 4` 是 K 和 V 每个（头, token）的 fp32 scale。scale 不是
        可有可无的记账：注意力内核靠它反量化，而 fp16 的 scale 在这些权重量级
        约 1e-2 的范围上会掉精度。

        Returns:
            一个块包含所有层的字节数。
        """
        n = self.bs * self.nkv * self.hd
        return self.n_layers * (2 * n + 2 * self.bs * self.nkv * 4)

    def memory_bytes(self) -> int:
        """池在当前的块数下的常驻总字节数。

        Returns:
            ``bytes_per_block() * num_blocks``，也就是整份 int8 缓存。
        """
        return self.bytes_per_block() * self.num_blocks

    def alloc(self) -> int:
        """取一个无人引用的块发出去，引用计数置 1。

        Returns:
            块 id。它由调用方用 :meth:`decref` 释放，并且只有引用计数还撑着时
            才有效。

        Raises:
            OutOfBlocks: 已经没有无人引用的块。池是每个活请求的 KV 预算，所以
                调用方必须先淘汰缓存前缀
                （:meth:`radix_cache.RadixCache.reserve`）再重试。
        """
        if not self.free:
            raise OutOfBlocks(f"no free KV blocks (used {self.num_used}/{self.num_blocks})")
        b = self.free.pop()
        self.refcount[b] = 1
        self.n_allocs += 1
        return b

    def incref(self, b: int) -> None:
        """给 ``b`` 加一份引用，用来把一个已缓存的块再共享给一个持有者。

        Args:
            b: 已经有人持有的块 id——基数树或另一条序列。加引用永远不分配，
                所以不会失败。
        """
        self.refcount[b] += 1

    def decref(self, b: int) -> None:
        """丢掉 ``b`` 的一份引用，归零时把它还给空闲表。

        Args:
            b: 调用方持有的块 id。释放一个自己并不拥有的块会把计数打成负数，
                并把别人还在读的块还回去，所以每次 :meth:`alloc` /
                :meth:`incref` 都必须恰好有一次对应的调用。
        """
        self.refcount[b] -= 1
        if self.refcount[b] == 0:
            self.free.append(b)

    def free_blocks(self, blocks: list[int]) -> None:
        """对 ``blocks`` 里每个块各丢一份引用。

        Args:
            blocks: 调用方持有的 id，通常是一条序列的整张块表。基数树仍在缓存
                的块会保留树那份引用，所以序列退休后它们的 KV 依然有效。
        """
        for b in blocks:
            self.decref(b)

    def copy_block(self, src: int, dst: int) -> None:
        """把 ``src`` 在每个层上的槽位复制到 ``dst``（写时复制）。

        Args:
            src: 序列当前正在共享的块。
            dst: 为同一批位置新分配的私有块。调用方在块边界处打开写入窗口，
                把槽位表指向 ``dst``；改去写 ``src`` 会破坏所有共享它的序列。
        """
        bs = self.bs
        self.k[:, dst * bs:(dst + 1) * bs].copy_(self.k[:, src * bs:(src + 1) * bs])
        self.v[:, dst * bs:(dst + 1) * bs].copy_(self.v[:, src * bs:(src + 1) * bs])
        self.kscale[:, dst * bs:(dst + 1) * bs].copy_(self.kscale[:, src * bs:(src + 1) * bs])
        self.vscale[:, dst * bs:(dst + 1) * bs].copy_(self.vscale[:, src * bs:(src + 1) * bs])
        self.n_cows += 1

    @property
    def rows(self) -> int:
        """槽位表的行数，也就是批容量。``1`` 表示只做单流。

        Returns:
            ``pos2slot`` 的行数；批处理模型把它设成桶容量，最后一行留给
            预填充。
        """
        return self.pos2slot.shape[0]

    def set_rows(self, rows: int) -> None:
        """把槽位表扩缩到 ``rows`` 行，第 0 行的内容保留。

        必须在任何图 capture 之前跑：captured 的注意力内核把它收到的行视图烧了
        进去，``pos2slot`` 的行步长是其中一部分。扩容会重新分配这张表，所以图
        一旦存在就不能再做。

        Args:
            rows: 想要的行数；至少取 1，与当前相同时是空操作。超出旧容量的行
                从零开始，所以调用方必须先给行赋一张表再用它。
        """
        rows = max(int(rows), 1)
        if rows == self.rows:
            return
        new = torch.zeros(rows, self.max_seq, dtype=torch.int64,
                          device=self.pos2slot.device)
        keep = min(rows, self.rows)
        new[:keep].copy_(self.pos2slot[:keep])
        self.pos2slot = new

    def assign(self, block: int, start: int, row: int = 0) -> None:
        """把第 ``row`` 行的位置 ``[start, start+bs)`` 指向 ``block``。

        Args:
            block: 这些位置现在住进它的槽位。调用方必须已经持有引用；
                ``assign`` 只重写映射。
            start: 这个块起始的 token 位置，通常是块大小的整数倍。超出
                ``max_seq`` 的 start 什么都不写。
            row: 槽位表的行；第 0 行是单流路径。
        """
        lo, hi = start, min(start + self.bs, self.max_seq)
        if hi <= lo:
            return
        self.pos2slot[row, lo:hi] = block * self.bs + torch.arange(
            hi - lo, device=self.pos2slot.device)

    def assign_table(self, blocks: list[int], start: int = 0, row: int = 0) -> None:
        """把一串连续的块按块对齐地铺进某一行槽位表。

        在这之前分页池的槽位表全是 0，所以跳过这一步的预填充会让每个位置都寻址
        到池槽位 0。要在 ``enable_paging()`` 之后、``capture()`` 或
        ``prefill()`` 之前调用。

        Args:
            blocks: 一条序列的块表，按位置顺序排列。
            start: 第一个块的起始位置；后续块各隔一个块大小。
            row: 要重写的槽位表行。
        """
        for i, b in enumerate(blocks):
            self.assign(b, start + i * self.bs, row)

    @staticmethod
    def _quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """把 ``[..., hd]`` 按行量化成 int8，每行一个 scale。

        Args:
            x: bf16 或 fp32 的行。scale 下界取 ``1e-5``，让全零行除以一个正数
                而不是产出 NaN。

        Returns:
            int8 张量和 fp32 scale，满足 ``q.float() * s[..., None]`` 能以 int8
            精度还原出 ``x``。
        """
        xf = x.float()
        s = xf.abs().amax(-1).clamp_min(1e-5) / 127.0
        q = (xf / s.unsqueeze(-1)).round().clamp_(-127, 127).to(torch.int8)
        return q, s

    def write_slot(self, layer: int, slot: torch.Tensor, k: torch.Tensor,
                   sk: torch.Tensor, v: torch.Tensor, sv: torch.Tensor) -> None:
        """把某一层的一个 token 写到 ``slot``（已经是量化好的 int8）。

        Args:
            layer: 池的层轴下标，也就是该层在*全注意力*层里的序号，不是模型的
                块下标。
            slot: 要写的池槽位，设备标量，通常来自 ``slots_for(pos1)``。
            k: ``[nkv, hd]`` 的 int8 key 行。
            sk: ``[nkv]`` 的 fp32 key 反量化 scale。
            v: ``[nkv, hd]`` 的 int8 value 行。
            sv: ``[nkv]`` 的 fp32 value 反量化 scale。
        """
        li = torch.full_like(slot, layer)
        self.k.index_put_((li, slot), k.unsqueeze(0))
        self.v.index_put_((li, slot), v.unsqueeze(0))
        self.kscale.index_put_((li, slot), sk.unsqueeze(0))
        self.vscale.index_put_((li, slot), sv.unsqueeze(0))

    def write_range(self, layer: int, slots: torch.Tensor, k: torch.Tensor,
                    sk: torch.Tensor, v: torch.Tensor, sv: torch.Tensor) -> None:
        """把某一层的 ``T`` 个 token 写到 ``slots``（``[T]``）。

        Args:
            layer: 在全注意力层里的下标。
            slots: ``[T]`` 个池槽位，从这些 token 正在写入的那一行槽位表解析
                而来。
            k: ``[T, nkv, hd]`` 的 int8 key。
            sk: ``[T, nkv]`` 的 fp32 key scale。
            v: ``[T, nkv, hd]`` 的 int8 value。
            sv: ``[T, nkv]`` 的 fp32 value scale。
        """
        li = torch.full_like(slots, layer)
        self.k.index_put_((li, slots), k)
        self.v.index_put_((li, slots), v)
        self.kscale.index_put_((li, slots), sk)
        self.vscale.index_put_((li, slots), sv)

    def write_rows(self, layer: int, slots: torch.Tensor, k: torch.Tensor,
                   sk: torch.Tensor, v: torch.Tensor, sv: torch.Tensor) -> None:
        """把某一层的一个 token 按 ``B`` 行各写到各自的 ``slots``（``[B]``）。

        :meth:`write_slot` 的批处理版本：``k``/``v`` 是 ``[B, nkv, hd]`` 的
        int8，scale 是 ``[B, nkv]``，所以所有行用每个池张量一次带下标写入就够，
        而不是 ``B`` 次独立启动。

        Args:
            layer: 在全注意力层里的下标。
            slots: 每个批行一个池槽位（``[B]``），通过该行自己的槽位表行收集。
            k: ``[B, nkv, hd]`` 的 int8 key。
            sk: ``[B, nkv]`` 的 fp32 key scale。
            v: ``[B, nkv, hd]`` 的 int8 value。
            sv: ``[B, nkv]`` 的 fp32 value scale。
        """
        li = torch.full_like(slots, layer)
        self.k.index_put_((li, slots), k)
        self.v.index_put_((li, slots), v)
        self.kscale.index_put_((li, slots), sk)
        self.vscale.index_put_((li, slots), sv)

    def slots_for(self, positions: torch.Tensor, row: int = 0) -> torch.Tensor:
        """给定位置在第 ``row`` 行槽位表里当前映射到的池槽位。

        Args:
            positions: 设备上的 int64 位置；captured 的图会把位置计数器传进来，
                而不是烧进去一个常数。
            row: 槽位表的行；第 0 行是单流路径。

        Returns:
            每个位置一个池槽位，顺序一致。
        """
        return self.pos2slot[row].index_select(0, positions)

    def gather(self, layer: int, n_tokens: int, row: int = 0) -> tuple[torch.Tensor, torch.Tensor,
                                                                      torch.Tensor, torch.Tensor]:
        """当前序列前 ``n_tokens`` 个位置的稠密 K/V。

        返回形状是 ``[nkv, S, hd]``（scale 是 ``[nkv, S]``），与稠密缓存的
        head-major 布局一致，预填充/验证的注意力代码因此两边共用一套。那两条
        路径是批处理、受算力限制的，在那里把池实体化成稠密视图是便宜的选择——
        参考分页实现也是这么做的。

        Args:
            layer: 在全注意力层里的下标。
            n_tokens: 要实体化的前缀长度；从该行槽位表的开头读，所以比整条
                序列更短的前缀也可以。
            row: 正在预填充的序列所用的槽位表行。

        Returns:
            ``(k, v, kscale, vscale)``，是这些位置按位置顺序排列的 head-major
            稠密视图。
        """
        slots = self.pos2slot[row][:n_tokens]
        return (self.k[layer].index_select(0, slots).transpose(0, 1),
                self.v[layer].index_select(0, slots).transpose(0, 1),
                self.kscale[layer].index_select(0, slots).transpose(0, 1),
                self.vscale[layer].index_select(0, slots).transpose(0, 1))


class SeqKV:
    """一条序列眼中的池：一张块表加一张槽位表。

    ``blocks`` 里的每个块都是本对象持有的一份引用，所以 :meth:`free` 是唯一
    正确的退休方式。``num_tokens`` 是已提交的长度——模型已经写好的 KV——而叠在
    它上面的 ``reserve``/``commit`` 窗口就是即将写入的那段。
    """

    def __init__(self, cache: PagedKVCache, blocks: list[int] | None = None,
                 num_tokens: int = 0) -> None:
        """包装一张调用方已经持有引用的块表。

        Args:
            cache: 这些块所属的池。
            blocks: 块表，原样收下；这里不做 ``incref``，所以调用方的引用转移
                给了本对象。
            num_tokens: 已经写进这些块的 token 数。不向块边界取整：前缀匹配
                可以停在块中间。
        """
        self.cache = cache
        self.blocks: list[int] = list(blocks or [])
        self.num_tokens = num_tokens
        self._pending = 0
        self._start = num_tokens

    def __len__(self) -> int:
        """已提交长度，也就是解码步可以放心注意到的位置范围。

        Returns:
            KV 已经写好的 token 数；还在途中的预留不计入，因为它还读不到。
        """
        return self.num_tokens

    @property
    def bs(self) -> int:
        """池的块大小，以 token 计。

        Returns:
            每块的 token 数；直接读池，因为离开它序列的块表没有意义。
        """
        return self.cache.bs

    @property
    def total_tokens(self) -> int:
        """含尚未写入的那段预留的长度。

        Returns:
            已提交长度加上打开的写入窗口，也就是调用方已经规划好要占多少个
            token。
        """
        return self.num_tokens + self._pending

    def reserve(self, t: int, radix=None) -> None:
        """为 ``t`` 个新 token 分配块，并打开写入窗口。

        仍在与树（或别的序列）共享的尾块会在窗口打开之前被复制，接下来的写入
        因此不会破坏共享前缀——这次复制就是整套池设计存在的那个写时复制。

        Args:
            t: 马上要写入的 token 数。``0`` 表示不分配也不复制，只关上窗口，
                这正是解码步什么都没生成时想要的空操作。
            radix: 不参与；仅为调用点兼容而保留。淘汰归
                ``BlockManager.reserve``，引擎会在本方法分配之前调它，所以空闲
                表此时已经补过了。

        Raises:
            OutOfBlocks: 分配途中池被抽干。只有调用方跳过了淘汰步骤才可能走到
                这里；此时必须让序列失败并释放，而不是重试。
        """
        if t <= 0:
            self._start, self._pending = self.num_tokens, 0
            return
        bs = self.bs
        cache = self.cache
        if self.blocks and self.num_tokens % bs != 0:
            tail = self.blocks[-1]
            if cache.refcount[tail] > 1:
                new = cache.alloc()
                cache.copy_block(tail, new)
                cache.decref(tail)
                self.blocks[-1] = new
                cache.assign(new, self.num_tokens - self.num_tokens % bs)
        total = self.num_tokens + t
        need = (total + bs - 1) // bs - len(self.blocks)
        for _ in range(need):
            blk = cache.alloc()
            self.blocks.append(blk)
            cache.assign(blk, (len(self.blocks) - 1) * bs)
        self._start, self._pending = self.num_tokens, t

    def commit(self) -> None:
        """关闭写入窗口，把预留的 token 计为已写入。"""
        self.num_tokens += self._pending
        self._pending = 0

    def commit_upto(self, n_tokens: int) -> None:
        """用模型自己的位置计数器设定序列长度。

        投机解码每一步推进的 token 数不固定，所以长度取自模型（它才是「写了多少
        KV」的权威），而不是来自 reserve/commit 的记账。

        Args:
            n_tokens: 这一步之后模型的位置计数器。传一个比已提交长度更小的值
                会丢掉模型仍在注意的 KV，所以调用方都读 ``model.pos1``。
        """
        self.num_tokens = int(n_tokens)
        self._pending = 0

    def free(self) -> None:
        """把每个块还给池，并清空这条序列。

        第二次调用是空操作，因为块表已经清空：把同一批 id 释放两次会破坏空闲
        表，把还活着的块发给下一次分配。
        """
        self.cache.free_blocks(self.blocks)
        self.blocks = []
        self.num_tokens = 0
