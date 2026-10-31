"""为调度器驱动一次模型前向。

Module: ``coffin/``（中间层），由 ``llm_engine`` 持有。把 runner 独立出来正是
批处理得以成立的原因：runner 负责一步*怎么*执行（预填充路径、分页写入窗口、
图回放、采样），调度器负责*哪些*序列进入这一步。

这里有两种执行模式：

* 单流（``batching=False``）——重构前的路径：一条序列、一个 token，解码图按
  一张槽位表 capture；
* 静态批（``batching=True``）——最多 ``max_num_batched`` 条序列共用一个前向。
  投影部分走批处理（一次权重读取服务所有行），而注意力和 delta-net 递推仍然
  按行跑现有的单 token 内核。

Constraints:
    * 行布局固定：``0..capacity-2`` 是解码行，行 ``capacity-1`` 是预填充暂存
      行。留这一行，才使得把一个预填充插进在跑的解码之间时不会踩坏某个解码行
      的循环状态。
    * 序列一直占着自己的行直到结束。行因此可能有空洞，每个批处理步都跑「高于
      最高活跃行的最小桶」，空洞用模型的哨兵行填上。永远不要压缩行：省下来的
      那点空间不值得每一步都去置换循环状态。
    * ``seq.num_cached_tokens`` 在每次前向之后从 ``model.pos1`` 刷新，绝不在
      本地自增——一个投机步会让模型前进不止一个 token。
    * 一个 prompt 可以跨多个步预填充（分块预填充）。序列的循环状态在分块之间
      住在它自己的解码行上，只在某一个分块期间借用暂存行，所以被暂停的请求不
      额外占显存：一份状态拷贝约 50 MB。
    * 解码可以不做图 capture（``EngineConfig.graph`` 是正确性开关，不是调优
      项）。两个解码入口产出的数字一样，区别只在于调的是哪一个。

分块预填充是三个调用，拆开才让它可中断：

    begin_prefill(seq)      认领一行、沿用基数前缀、把状态放好
    prefill_chunk(seq, n)   把游标推进最多 n 个 token
    finish_prefill(seq)     记下最后的 logits、插入前缀、转入 RUNNING

``begin_prefill`` 每次准入跑一次，``prefill_chunk`` 则按调度器给的预算跑任意
多次；第二次调用需要的东西全在序列上，所以步边界是合法的暂停点。被抢占的请求
会从基数树仍然持有的那段前缀开始，把这三个调用再走一遍。
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch

from .sequence import SequenceStatus

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .config import EngineConfig
    from .block_manager import BlockManager
    from .prefix_engine import PrefixCacheEngine
    from .sequence import Sequence


#: 模型为每种批大小各 capture 一张图；一步跑能装下的最小桶（其余是哨兵行，其
#: logits 会被丢弃）。取 2 的幂，是为了让每个桶边界上的填充浪费都低于 100%。
BATCH_BUCKETS = (1, 2, 4, 8)


class ModelRunner:
    """执行引擎模型的一步。"""

    def __init__(self, model, cfg: "EngineConfig") -> None:
        """把一个 runner 绑到一个已加载的模型上。

        Args:
            model: ``engine_api.Model`` / ``PagedModel`` 的一个实现。
            cfg: 引擎配置；预填充分块大小、预填充 quantum 和快照预算都从这里读。
        """
        self.model = model
        self.cfg = cfg
        self.pc: "PrefixCacheEngine | None" = None
        self.bm: "BlockManager | None" = None
        self.batching = False
        self.buckets: tuple[int, ...] = ()
        #: 解码步是否可以回放 capture 下来的图。``False`` 时每个 token 都走模型的
        #: eager 前向：数字一样，但什么都不录，这是 capture 不可信的后端上唯一
        #: 安全的路（回放过期输出的图会静默给出错误答案）。由引擎从
        #: ``EngineConfig.graph`` 设置。
        self.graph = cfg.graph
        #: 批友好的投机解码（MTP），引擎不要求就是关的。
        self.spec = False
        self.spec_k = 0
        self.spec_sparse_k = 0
        #: 解码行 -> seq_id。行在序列存活期间是稳定的，批处理步因此可以不做循环
        #: 状态的置换。
        self._row_owner: dict[int, int] = {}
        #: 挂上前缀缓存时由引擎设置；``ping`` 会读它。
        self.last_prefill_s = 0.0
        #: 一个分块可覆盖的 token 数。放在这里而不是从调度器读，是为了让只驱动
        #: runner 的调用方（测试、探针）切出完全相同的分块。
        self.prefill_quantum = max(cfg.block_size, cfg.prefill_quantum)

    def attach_prefix_cache(self, pc: "PrefixCacheEngine") -> None:
        """让预填充走某个前缀缓存引擎。

        Args:
            pc: 缓存引擎。块管理器也由它提供，两者因此不会对「一条序列持有
                哪些块」产生分歧。
        """
        self.pc = pc
        self.bm = pc.bm

    def attach_blocks(self, bm: "BlockManager") -> None:
        """挂一个没有前缀缓存的块管理器（批处理路径）。

        Args:
            bm: 用 ``with_radix=False`` 建的管理器；块按序列分配，从不共享。
        """
        self.bm = bm

    def can_chunk_prefill(self) -> bool:
        """这个引擎能不能把 prompt 分片推进。

        单流的前缀引擎一次调用就把 prompt 预填充完，缓存记账也自己管，token 预算
        限不住它；批处理路径按序列分配，所以可以。

        Returns:
            挂了块管理器的批处理 runner 返回 ``True``。
        """
        return self.batching and self.bm is not None

    def free_blocks(self) -> int:
        """池现在能发出去的块数。

        调度器在准入之前会问这一句，这样拿不到下一个分块所需块的客户端就在
        队列里等着，而不是被准入、发现跑不动、又被淘汰一次。不问这一句，两个
        请求就会每一步互换一次池子，谁也跑不完。

        Returns:
            空闲块数；没挂池时返回一个很大的数（单流引擎不按请求分配）。
        """
        bm = self.bm
        return 1 << 30 if bm is None else bm.num_free

    def blocks_for(self, tokens: int) -> int:
        """容纳 ``tokens`` 个位置需要的块数。

        Args:
            tokens: 调用方即将占用的位置数。

        Returns:
            按配置的块大小算出的块数。
        """
        bm = self.bm
        if bm is None:
            return 0
        return (tokens + bm.block_size - 1) // bm.block_size

    def reserve_for(self, tokens: int) -> int:
        """淘汰缓存前缀，直到能容纳 ``tokens`` 个位置。

        调度器在准入之前问这一句，因为只看池的空闲数会低估可用量：满池子里装的
        大多是没人读的缓存前缀，淘汰能把它们还回来。先不淘汰就去检查，请求会
        被树本可以让出的块挡在后面干等。

        Args:
            tokens: 请求需要容纳的位置数。

        Returns:
            淘汰之后剩下的空闲块数。
        """
        bm = self.bm
        if bm is None:
            return 1 << 30
        bm.reserve(self.blocks_for(tokens))
        return bm.num_free

    def release_row(self, seq: "Sequence") -> None:
        """释放一条已结束序列占着的批行。

        Args:
            seq: 已结束的序列。它的 ``row`` 会被重置成 ``-1``，这样迟到的第二次
                调用不会去释放别人的行。
        """
        if seq.row >= 0:
            self._row_owner.pop(seq.row, None)
            seq.row = -1

    def _claim_row(self, seq: "Sequence") -> int:
        """给 ``seq`` 分一个最低的空闲解码行，整个生命周期都用它。

        Args:
            seq: 正在准入的序列。

        Returns:
            认领到的行号。

        Raises:
            RuntimeError: 所有行都被占了。这说明调度器准入时越过了
                ``max_num_seqs``；共用一行会破坏循环状态，而不只是变慢。
        """
        capacity = int(getattr(self.model, "decode_capacity", 1))
        for row in range(capacity):
            if row not in self._row_owner:
                self._row_owner[row] = seq.seq_id
                seq.row = row
                return row
        raise RuntimeError(
            f"no free decode row (capacity {capacity}, running {len(self._row_owner)})")

    def enable_batching(self, max_batch: int,
                        buckets: tuple[int, ...] = BATCH_BUCKETS) -> None:
        """把批解码路径准备好，每个桶 capture 一张图。

        capture 打开时，每个桶都是一次录制好的回放。关掉时模型照样分配同样的
        行缓冲区和暂存块，只是什么都不录，:meth:`_decode_batch` 改走 eager
        入口——两种模式因此一致，不允许 capture 的后端也仍然能批处理。

        Args:
            max_batch: 解码行的上限。超过它的桶会被丢掉，因为比行数还大的图
                永远跑不起来。
            buckets: 候选桶大小。总包含 1，这样只有一条在跑的序列时也有图可
                回放。
        """
        buckets = tuple(b for b in buckets if b <= max_batch) or (1,)
        self.buckets = buckets
        # +1 那一行是预填充暂存行；没有它，分组预填充就得和一条已经在解码的
        # 序列共用行。
        self.model.enable_batching(max(self.buckets) + 1)
        if self.graph:
            self.model.capture_batch(self.buckets)
        self.batching = True

    def enable_spec(self, k: int, sparse_topk: int = 0) -> None:
        """开启批友好的 MTP，每行 ``k`` 个草稿。

        Args:
            k: 每个投机步的草稿 token 数。
            sparse_topk: 每行只对历史 KV 块里得分最高的 k 个做验证（近似）。
                0 表示保留全注意力，也就是精确模式；做近似是因为批验证否则
                每个候选 token 都要过一遍权重。

        Raises:
            RuntimeError: 批处理没开。单流投机有自己的路径
                （``spec_generate``），不走这个开关。
        """
        if not self.batching:
            raise RuntimeError("enable_spec() needs the batched engine")
        kwargs = {"sparse_topk": sparse_topk} if sparse_topk else {}
        self.model.enable_spec_batch(K=k, buckets=self.buckets, **kwargs)
        self.spec_k = int(k)
        self.spec_sparse_k = int(sparse_topk)
        self.spec = True

    @torch.inference_mode()
    def prefill(self, seq: "Sequence") -> torch.Tensor:
        """吃下整个 prompt（已缓存前缀除外），返回 logits。

        不分块的入口。它的存在是为了服务不能分块的引擎（单流前缀缓存），以及
        作为批处理路径「一个分块覆盖全部」的那种情况，两条路的数字因此都保持
        原样。

        Args:
            seq: 要预填充的序列。之后会从模型刷新它的 ``num_cached_tokens``
                和 ``num_scheduled_tokens``。

        Returns:
            最后一个 prompt 位置的 logits，形状 ``[vocab]``。

        Raises:
            RuntimeError: 批处理路径被要求用一个分块跑完 prompt，但游标没到
                末尾。这说明调用跑的分块比 prompt 还短，是调用方的 bug：分块
                预填充应该用 :meth:`prefill_chunk`。
        """
        t0 = time.time()
        if self.pc is not None:
            logits = self.pc.prefill(seq.token_ids, mm=seq.mm)
            seq.matched = self.pc.last_matched
            if self.pc.seq is not None:
                seq.block_table = list(self.pc.seq.blocks)
            seq.prefill_cursor = seq.num_tokens
        elif self.batching:
            self.begin_prefill(seq)
            logits = self.prefill_chunk(seq, seq.num_tokens - seq.prefill_cursor)
            if logits is None:                       # pragma: no cover - defensive
                raise RuntimeError("prefill_chunk() did not finish the prompt")
        else:
            self.model.reset()
            logits = self.model.prefill(seq.token_ids, mm=seq.mm)
            seq.prefill_cursor = seq.num_tokens
        seq.num_cached_tokens = int(self.model.pos1)
        seq.num_scheduled_tokens = seq.num_cached_tokens - seq.matched
        self.last_prefill_s = time.time() - t0
        return logits

    @torch.inference_mode()
    def begin_prefill(self, seq: "Sequence") -> None:
        """开一次可续的预填充：占行、匹配前缀、摆好起始状态。

        每次准入跑一次。前缀匹配在整个 token 流上做（prompt 加上已经生成的
        部分，再去掉必须由真实前向产出 logits 的最后一个 token），所以被恢复的
        请求会从基数树撑得住的最深块边界重新开始，而不是从 token 0 重来。

        这里只预留第一个分块需要的块。预留给整个 prompt 会让一个长请求为了它
        好几步之后才用得上的块，把其他所有请求的缓存都淘汰掉。

        Args:
            seq: 要准入的序列。它必须还没占行；被恢复的序列保留自己的行和块，
                调用方不会为它重复这次调用。

        Raises:
            RuntimeError: 没挂块管理器。批处理路径缺了它跑不了，因为前向之前
                必须先把 KV 分配出来。
        """
        bm, model = self.bm, self.model
        if bm is None:
            raise RuntimeError("chunked prefill needs a block manager")
        radix = bm.radix
        if seq.row < 0:
            row = self._claim_row(seq)
        else:
            row = seq.row

        matched, states = 0, None
        if radix is not None:
            # 上一次准入拿到的块必须在块表被匹配到的前缀替换之前还回去：被恢复
            # 的序列通常空表进来，而悄悄丢掉一张非空表会泄漏池在计的那些引用。
            if seq.block_table:
                bm.release(seq)
            bm.reserve(1)
            blocks, matched, states = bm.match_prefix(seq.token_ids[:-1],
                                                      require_states=True,
                                                      mm=seq.mm)
            seq.block_table = list(blocks)
        seq.matched = matched
        seq.prefill_cursor = matched
        seq.prefill_states_at = {}
        seq.prefill_logits = None
        seq.prefill_started = True
        # 状态先在暂存行恢复，再搬到序列自己那一行——分块之间它就住在那里。
        # 直接恢复到解码行会要求那一行在任何前向跑过之前就可寻址，而暂存行是
        # 单 token 路径唯一能保证存在的一行。
        stage = model.prefill_row
        if states is not None:
            seq.prefill_states_at[matched] = states
            model.restore_linear_states(states, row=stage)
        else:
            model.zero_linear_states(stage)
        model.move_linear_states(stage, row)
        seq.num_cached_tokens = matched
        seq.num_scheduled_tokens = 0

    @torch.inference_mode()
    def prefill_chunk(self, seq: "Sequence", budget: int) -> torch.Tensor | None:
        """把 prompt 推进最多 ``budget`` 个 token。

        分块在暂存行上跑，期间把序列的循环状态搬过去、跑完再搬回来，所以分块
        之间状态待在序列自己的解码行上，被暂停的请求不会多占一份私有拷贝。

        Args:
            seq: 分块从它的 ``prefill_cursor`` 开始。必须已经为它跑过
                ``begin_prefill``。
            budget: 本分块最多覆盖的 token 数。剩余更多就留给下一个分块；剩余
                更少就把 prompt 跑完。

        Returns:
            本分块跑完 prompt 时返回最后一个 token 的 logits，还需要更多分块时
            返回 ``None``。跑完时也会调用 :meth:`finish_prefill`，调用方只需要
            采样。

        Raises:
            RuntimeError: 没为这条序列跑 ``begin_prefill``，或者没挂块管理器。
        """
        bm, model = self.bm, self.model
        if bm is None:
            raise RuntimeError("chunked prefill needs a block manager")
        if not seq.prefill_started:
            raise RuntimeError("prefill_chunk() before begin_prefill()")
        t0 = time.time()
        bs = bm.block_size
        radix = bm.radix
        stage = model.prefill_row
        row = seq.row
        cursor = seq.prefill_cursor
        prompt_len = seq.num_prompt_tokens
        # 被恢复的请求要重算的流是 prompt 加上它已经生成的 token：prompt 边界是
        # 新 token 的采样位置，不是前向的停止位置。
        total = seq.num_tokens
        end = min(cursor + max(int(budget), 1), total)
        if end <= cursor:                       # pragma: no cover - caller guards
            self.last_prefill_s = time.time() - t0
            return seq.prefill_logits

        # 先淘汰再分配：一旦这个请求握住了它要的块，这些块就不再是那趟替它腾
        # 空间的淘汰的候选了。
        held = len(seq.block_table) * bs
        if end > held:
            bm.reserve((end - held + bs - 1) // bs + 1)
        bm.ensure_blocks(seq, end)

        model.move_linear_states(row, stage)
        bm.assign(seq.block_table, row=stage)
        model.pos[stage:stage + 1].fill_(cursor)

        # 两个粒度，出于两种不同的原因。`step` 是快照间隔向上取整到整块：循环
        # 状态只在块边界上可复用，而长 prompt 每个块打一次快照比权重还费显存，
        # 所以边界要放稀。`cfg.chunk` 是一次调用里 GEMM 的 M，纯属调优数字
        # （实测 256/512/2048 时是 865/1035/1094 tok/s）。把两者分开，重新调
        # GEMM 时才不会连带改变哪些边界存在。
        step = max(bs, self.cfg.snapshot_interval // bs * bs)
        # 块对齐头部的终点：从那以后只有 logits 重要，所以没对齐的尾巴（以及
        # 已生成的 token）一趟跑完。
        head_len = max(0, (prompt_len // bs) * bs - cursor)
        spec_row = row if self.spec else None
        logits = None
        done = cursor
        while done < end:
            stop = end if done - cursor >= head_len else min(done + step, end)
            logits = model.prefill(seq.token_ids[done:stop], chunk=self.cfg.chunk,
                                   spec_row=spec_row, mm=seq.mm)
            done = stop
            seq.prefill_cursor = done
            seq.num_cached_tokens = done
            if radix is not None and done % bs == 0 and done <= prompt_len:
                seq.prefill_states_at[done] = model.snapshot_linear_states(row=stage)
                self._trim_states(seq)
                # 每个边界插一次，而不是等 prompt 跑完再插：预填充中途被抢占的
                # 请求因此把已经算好的块留成可复用，第二个请求还能在第一个仍在
                # 预填充时就从这些边界接上。
                self._insert_prefix(seq, radix)

        model.move_linear_states(stage, row)
        seq.num_scheduled_tokens = seq.prefill_cursor - seq.matched
        if seq.prefill_cursor < total:
            self.last_prefill_s = time.time() - t0
            return None
        seq.prefill_logits = logits
        self.finish_prefill(seq)
        self.last_prefill_s = time.time() - t0
        return logits

    def finish_prefill(self, seq: "Sequence") -> None:
        """收掉一次跑完的预填充，让请求可以被解码。

        把 prompt 插进基数树（含最后一个分块产出的边界），然后把序列翻成
        ``RUNNING``。采第一个生成 token 是调用方的事：它手里有这次预填充刚
        返回的 logits。

        Args:
            seq: ``prefill_cursor`` 已经到达 token 末尾的序列。
        """
        bm = self.bm
        if bm is not None and bm.radix is not None:
            self._insert_prefix(seq, bm.radix)
        seq.status = SequenceStatus.RUNNING
        seq.num_cached_tokens = seq.num_tokens

    def _insert_prefix(self, seq: "Sequence", radix) -> None:
        """把 ``seq`` 的流里按块对齐的那部分记进基数树。

        每个分块之后都调一次，而不是最后一次性调：预填充中途被抢占的请求因此
        把已经算出来的东西都留成可复用，第二个请求也能在第一个仍在预填充时从
        任意已完成的块边界接上。

        Args:
            seq: 持有要记录的块的序列。
            radix: 前缀树。不会是 ``None``；调用方已检查。
        """
        bs = self.bm.block_size
        full = min(seq.prefill_cursor, (seq.num_tokens // bs) * bs)
        if full <= 0:
            return
        radix.insert(seq.token_ids[:full], seq.block_table[:full // bs],
                     dict(seq.prefill_states_at), mm=seq.mm)

    def _trim_states(self, seq: "Sequence") -> None:
        """把一条序列的存活快照压到树自己的预算以内。

        一个快照约 50 MB（24 层线性注意力），长 prompt 的每个块边界都留着会比
        权重还费显存。最深的边界先丢，而序列接上的那个前缀永远不丢：被恢复的
        请求会从那里重新开始。

        Args:
            seq: 就地裁剪其 ``prefill_states_at`` 的序列。
        """
        states_at = seq.prefill_states_at
        while len(states_at) > self.cfg.snapshots:
            droppable = [k for k in states_at if k != seq.matched]
            if not droppable:
                break
            del states_at[min(droppable)]

    @torch.inference_mode()
    def preempt(self, seq: "Sequence") -> None:
        """淘汰一个请求，同时让它的前缀可恢复。

        重算是这里最便宜的策略：块是设备上的 int8 KV，把它们换出到主机侧意味着
        每次抢占都要一次大的同步拷贝，而基数树本来就留着每一个按块对齐的前缀。
        序列因此保留自己的 token，只丢掉重算时会重建的位置记账。

        Args:
            seq: 要淘汰的序列。它必须在调度器的运行集合里；把它放回等待队列是
                调用方的事，这样引擎才能把对调度器的改动都留在一处。
        """
        bm = self.bm
        if bm is not None:
            if bm.radix is not None and seq.prefill_cursor > 0:
                self._insert_prefix(seq, bm.radix)
            bm.release(seq)
        self.release_row(seq)
        seq.block_table.clear()
        seq.park_preempted()

    def can_group_prefill(self) -> bool:
        """多个 prompt 能不能共用一个预填充前向。

        Returns:
            开了批处理且挂了块管理器时返回 ``True``；两者都是分组预填充的
            前提。
        """
        return self.batching and self.bm is not None

    def _match_for_prefill(self, seq: "Sequence") -> tuple[int, dict | None]:
        """沿用 ``seq`` 最长的已缓存前缀。

        Args:
            seq: 马上要预填充的序列；它的块表和 ``matched`` 会被就地更新。

        Returns:
            ``(matched, states)``；没有基数树时分别是 0 和 ``None``。
        """
        bm = self.bm
        if bm is None or bm.radix is None:
            return 0, None
        bs = bm.block_size
        bm.reserve((seq.num_tokens + bs - 1) // bs + 1)
        blocks, matched, states = bm.match_prefix(seq.token_ids[:-1],
                                                  require_states=True,
                                                  mm=seq.mm)
        seq.block_table = list(blocks)
        seq.matched = matched
        return matched, states

    @torch.inference_mode()
    def prefill_group(self, seqs: list["Sequence"]) -> list[torch.Tensor]:
        """一趟前向预填充多个 prompt。

        前缀复用的逻辑也接在这里：每条序列先沿用自己匹配到的块和循环状态（放到
        自己那一行），分组前向让每个 prompt 从各自的 ``matched`` 偏移开始，最后
        把它返回的块边界快照插回基数树。

        Args:
            seqs: 要预填充的 prompt。还没有行的会先分到一行。

        Returns:
            每条序列一个 logits 张量，顺序一致。

        Raises:
            RuntimeError: 没挂块管理器。
        """
        t0 = time.time()
        bm = self.bm
        if bm is None:
            raise RuntimeError("grouped prefill needs a block manager")
        radix = bm.radix
        matched_states: list[tuple[int, dict | None]] = []
        for seq in seqs:
            if seq.row < 0:
                self._claim_row(seq)
            matched, states = self._match_for_prefill(seq)
            bm.ensure_blocks(seq, seq.num_tokens)
            if states is not None:
                self.model.restore_linear_states(states, row=seq.row)
            matched_states.append((matched, states))

        # 带图 prompt 把载荷挂在 ``seq.mm`` 上；``model.prefill_group`` 自己从
        # 序列里读，所以这条路径不用再传 mm 参数。
        logits, snaps = self.model.prefill_group(
            seqs, snapshot_interval=(self.cfg.snapshot_interval
                                     if radix is not None else None))
        for seq, (matched, states), snap in zip(seqs, matched_states, snaps):
            seq.prefill_cursor = seq.num_tokens
            seq.prefill_started = False
            seq.prefill_logits = None
            seq.num_cached_tokens = len(seq.token_ids)
            seq.num_scheduled_tokens = seq.num_cached_tokens - matched
            # 整个 prompt 都走完了这一趟前向，序列现在可解码。这条路径不走
            # ``finish_prefill``，而留着 ``PREFILLING`` 状态会让它进不了之后
            # 任何一次解码步。
            seq.status = SequenceStatus.RUNNING
            if radix is None:
                continue
            states_at = dict(snap)
            if matched and states is not None:
                states_at[matched] = states
            while len(states_at) > self.cfg.snapshots:
                droppable = [k for k in states_at if k != matched]
                if not droppable:
                    break
                del states_at[min(droppable)]
            radix.insert(seq.token_ids, seq.block_table, states_at, mm=seq.mm)
        self.last_prefill_s = time.time() - t0
        return logits

    def begin_token(self, seq: "Sequence") -> None:
        """打开单 token 写入窗口（仅前缀缓存路径）。

        窗口必须在写 KV 之前打开，仍在与基数树共享的尾块会在那一刻被复制，
        已缓存的前缀因此保持只读。

        Args:
            seq: 马上要解码一个 token 的序列。
        """
        if self.pc is not None and self.pc.seq is not None:
            self.pc.seq.reserve(1)

    def decode_step(self, seq: "Sequence", token: int) -> torch.Tensor:
        """沿解码路径推进一个 token（图回放或 eager）。

        Args:
            seq: 正在推进的序列。
            token: 要喂进去的 token，也就是上一次的输出。

        Returns:
            下一个位置的 logits。
        """
        logits = self._forward_token(token)
        if self.pc is not None and self.pc.seq is not None:
            self.pc.seq.commit_upto(self.model.pos1)
        seq.num_cached_tokens = int(self.model.pos1)
        return logits

    def _forward_token(self, token: int) -> torch.Tensor:
        """按本引擎 capture 的那条解码路径走一个 token。

        两个模型入口只在记账上不同：``decode_step`` 回放 capture 下来的图，
        ``logits`` 用 eager 跑同一个前向。两者都把 token 铺到预填充行上、都会
        推进位置计数器，所以下游（KV 写入窗口、``commit_upto``）完全一样，变的
        只是启动方式。

        Args:
            token: 要喂进去的 token id。

        Returns:
            下一个位置的 logits；两种路径下都是活缓冲区。
        """
        if self.graph:
            return self.model.decode_step(token)
        return self.model.logits(token)

    @torch.inference_mode()
    def decode(self, seqs: list["Sequence"]) -> list[int]:
        """每条序列的下一个 token，多于一条在跑时走批处理。

        Args:
            seqs: 要各推进一个 token 的运行中序列。

        Returns:
            每条序列一个采样出的 token id，顺序一致。
        """
        if self.batching:
            return self._decode_batch(seqs)
        out: list[int] = []
        for seq in seqs:
            self.begin_token(seq)
            logits = self.decode_step(seq, seq.token_ids[-1])
            out.append(self.sample(logits, seq.params.temperature))
        return out

    @torch.inference_mode()
    def spec_step(self, seqs: list["Sequence"]):
        """一个批友好的投机步。

        Args:
            seqs: 运行中的序列。每条的待定 token 必须已经由引擎追加过。

        Returns:
            ``(accepted, logits)``。``accepted[i]`` 以序列 ``i`` 的待定 token 开头，
            后面是主模型接受的草稿，所以调用方追加的是 ``accepted[i][1:]``。
            草稿模型的窗口覆盖不了某一行时返回 ``None``——此时调用方必须退回
            普通批解码步，而不是跳过这条序列。

        Raises:
            RuntimeError: 批处理或投机没开。退避是调用方的责任，所以这里是
                编程错误，不是运行期状况。
        """
        model, bm = self.model, self.bm
        if bm is None or not self.spec:
            raise RuntimeError("spec_step() needs the batched spec engine")
        rows = [seq.row for seq in seqs]
        active = set(rows)
        bucket = next(b for b in self.buckets if b > max(active))
        tokens: list[int] = []
        positions: list[int] = []
        tables: list[list[int]] = []
        for seq in seqs:
            pos = seq.num_cached_tokens
            # 待定 token 占 1 个、草稿占 k 个：全都要写 KV。
            bm.ensure_blocks(seq, pos + self.spec_k + 1)
            tokens.append(seq.token_ids[-1])
            positions.append(pos)
            tables.append(seq.block_table)
        try:
            accepted, logits, _counts = model.spec_step_batch(
                tokens, positions, tables, bucket, K=self.spec_k,
                row_indices=rows,
                pad_rows=[r for r in range(bucket) if r not in active])
        except RuntimeError:
            # 草稿模型是按固定窗口 capture 的；某行的上下文放不进去就会拒绝。
            # 通知调用方比在生成中途重建草稿模型便宜得多。
            return None
        for seq, acc in zip(seqs, accepted):
            seq.num_cached_tokens += len(acc)
        return accepted, logits

    def _decode_batch(self, seqs: list["Sequence"]) -> list[int]:
        """用一趟批解码前向推进多条序列。

        Args:
            seqs: 运行中的序列，每条都有稳定的行。调用方已经为它们各自要写的
                那个位置腾好了空间；本方法不许分配，因为这里的分配既不能淘汰
                也不能抢占，只会留下一个长到一半的批。

        Returns:
            每条序列一个采样出的 token id。

        Raises:
            RuntimeError: 没挂块管理器，或者某条序列的块表放不下它即将写的
                位置。两者都说明调用方漏了容量检查。
        """
        model, bm = self.model, self.bm
        if bm is None:
            raise RuntimeError("batched decode needs a block manager")
        # 行在序列存活期间是稳定的，所以结束的序列留下的是空洞，而不是逼着大家
        # 去置换循环状态。这一步跑「高于最高活跃行的最小桶」，模型用哨兵行填上
        # 每个空洞。
        rows = [seq.row for seq in seqs]
        active = set(rows)
        bucket = next(b for b in self.buckets if b > max(active))
        pad_rows = [r for r in range(bucket) if r not in active]
        tokens: list[int] = []
        positions: list[int] = []
        tables: list[list[int]] = []
        for seq in seqs:
            pos = seq.num_cached_tokens
            if (pos + 1 + bm.block_size - 1) // bm.block_size > len(seq.block_table):
                raise RuntimeError(
                    f"decode row {seq.row} has no block for position {pos}: "
                    "the step's capacity check is missing")
            tokens.append(seq.token_ids[-1])
            positions.append(pos)
            tables.append(seq.block_table)
        model.set_rows(tokens, positions, tables, row_indices=rows)
        # 同样的铺设、同样的实现、同样的缓冲区：唯一区别是这一趟前向有没有被
        # 录进图。所以不允许 capture 的后端（SCOW 驱动）也照样能走批处理步。
        if self.graph:
            logits = model.decode_batch(bucket, bucket, pad_rows=pad_rows)
        else:
            logits = model.decode_batch_eager(bucket, bucket, pad_rows=pad_rows)
        for seq, pos in zip(seqs, positions):
            seq.num_cached_tokens = pos + 1
        return self._sample_rows(logits, seqs)

    @staticmethod
    def _sample_rows(logits: torch.Tensor, seqs: list["Sequence"]) -> list[int]:
        """每条序列从批 logits 里自己那一行采样。

        Args:
            logits: 批 logits，``[bucket, vocab]``。
            seqs: 按行序排列的序列。

        Returns:
            每条序列一个 token id。只有一行的 logits 张量一律读下标 0，不看
            ``seq.row``，因为 bucket-1 的图无论序列占着多高的行，写的都是第 0
            行。
        """
        out: list[int] = []
        for seq in seqs:
            row = logits[seq.row] if logits.shape[0] > 1 else logits[0]
            out.append(ModelRunner.sample(row, seq.params.temperature))
        return out

    @staticmethod
    def sample(logits: torch.Tensor, temperature: float) -> int:
        """从 ``logits`` 里挑一个 token。

        Args:
            logits: ``[vocab]`` 的 logits，可能是 bf16。
            temperature: 取 ``0`` 时选 argmax；采样路径会先升到 fp32，因为 bf16
                的 softmax 会丢掉太多尾部概率。

        Returns:
            采样出的 token id。
        """
        if temperature > 0:
            p = torch.softmax(logits.float() / temperature, dim=-1)
            return int(torch.multinomial(p, 1))
        return int(logits.argmax())
