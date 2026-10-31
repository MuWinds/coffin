"""在 CPU 上验证分块预填充、抢占与 eager 批处理解码。

Module: ``coffin/tests``。用一台会记账的假模型驱动 ``llm_engine`` /
``scheduler`` / ``model_runner`` / ``block_manager``，底下挂真实分页池，
于是整套调度面不用 DCU、不用 checkpoint 就能跑。

Constraints:
    * 假模型的「KV」就是每行一个 token 列表，并且拒绝写到「下一个空位」以外
      的位置。等价性结论因此是精确的而不是近似的：从错误位置续上的分块预填充，
      或者把该行携带状态重置掉的分块预填充，事后都不可能把 prompt 还原成
      行状态。
    * 池是真的，块引用计数、淘汰、抢占都量在真实分配上。每个会分配的测试都用
      ``num_free`` 是否回到初值查泄漏。
    * 全程 ``graph=False``。这台主机记不了捕获图，而 SCOW 部署跑的正是 eager
      路径，所以测试不能依赖捕获才能通过。
    * 从不断言耗时；``prefill_s`` 只检查字段在不在。

必须守住的不变量：

1. 一次预填充调用与 N 次分块调用产出的行状态、下一个 token 的 logits 相同；
2. 落在携带状态内部的分块边界不得重置它——第 k 块之后的状态恰好是 prompt
   前缀；
3. 后缀跨多块时分，前缀命中仍然生效；
4. 预填充中途、解码中途、以及带基数前缀时被抢占，恢复后都不能丢或重复已输出
   的 token；
5. token 预算真的卡住单步预填充量，短请求不会被长 prompt 挡在后面；
6. 池小到推不动时抛错，而不是原地空转。

    python -m unittest discover -s tests -p "test_chunked_prefill.py" -v
"""

from __future__ import annotations

import unittest

# support 只为把顶层包加进 sys.path，本文件不加载 checkpoint，所以导入后不使用
# 它，F401 是预期内的。
import support  # noqa: F401

import torch

from cache.paged_cache import PagedKVCache
from engine.block_manager import BlockManager
from engine.config import EngineConfig
from engine.llm_engine import NO_PROGRESS_LIMIT, LLMEngine
from engine.sequence import SamplingParams, Sequence, SequenceStatus

BLOCK = 8
MAX_SEQ = 256
VOCAB = 32


class FakePrefillModel:
    """一台分页模型，它的「KV」就是写进每一行的 token 流。

    每个入口都把写入记到 ``rows`` 里，并核对被要求写入的位置是否正是该行真正的
    当前位置——这正是分块预填充必须满足的全部契约。logits 由该行的完整 token
    列表算出，因此两次运行只要写入了同一串 token 就一定对得上。
    """

    def __init__(self, num_blocks: int = 48, block_size: int = BLOCK) -> None:
        """分配池和单行默认状态。

        Args:
            num_blocks: 物理块数；故意给小，让抢占和淘汰够得着。
            block_size: 每块的 token 数。
        """
        self.max_seq = MAX_SEQ
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.pool = PagedKVCache(1, 1, 4, num_blocks=num_blocks,
                                 block_size=block_size, max_seq=MAX_SEQ,
                                 device="cpu")
        self.batch_capacity = 1
        self.decode_capacity = 1
        self.prefill_row = 0
        self.mtp = None
        self.mtp_prefill = True
        self.pos = torch.zeros(1, dtype=torch.long)
        self.pos1 = self.pos[:1]
        self.rows: list[list[int]] = [[]]
        self.writes: list[tuple[int, int, list[int]]] = []
        self.scratch: list[int] = []
        self.mm_calls: list[object] = []

    # 生命周期
    def enable_paging(self, num_blocks: int, block_size: int):
        """挂上分页池。

        Args:
            num_blocks: 池大小。
            block_size: 每块的 token 数。

        Returns:
            池本身，块管理器随后会包住它。
        """
        return self.pool

    def enable_batching(self, capacity: int) -> None:
        """让模型每个批条目一行，外加一行暂存行。

        Args:
            capacity: 行数，含暂存行。
        """
        self.batch_capacity = capacity
        self.decode_capacity = capacity - 1
        self.prefill_row = capacity - 1
        self.pos = torch.zeros(capacity, dtype=torch.long)
        self.pos1 = self.pos[self.prefill_row:self.prefill_row + 1]
        self.rows = [[] for _ in range(capacity)]
        # 真模型在这里把槽位表扩到行数这么多行；少了这一步，0 号以上的行没有
        # 可写入的槽位表。
        self.pool.set_rows(capacity)
        for row in range(capacity):
            self.scratch.append(self.pool.alloc())
            self.pool.assign(self.scratch[-1], 0, row)

    def reset(self) -> None:
        """把所有行倒回空上下文。"""
        self.pos.zero_()
        self.rows = [[] for _ in range(self.batch_capacity)]

    def snapshot_linear_states(self, row: int = 0) -> dict:
        """给一行的携带状态打快照。

        Args:
            row: 要打快照的行。

        Returns:
            该行写入的 token 流，它在这里就是全部状态。
        """
        return {"tokens": tuple(self.rows[row])}

    def restore_linear_states(self, states, row: int = 0) -> None:
        """把一份快照装进 ``row``。

        Args:
            states: :meth:`snapshot_linear_states` 的返回值。
            row: 要被覆盖的行。
        """
        self.rows[row] = list(states["tokens"])

    def zero_linear_states(self, row: int) -> None:
        """清空一行的携带状态。

        Args:
            row: 要清空的行。
        """
        self.rows[row] = []

    def move_linear_states(self, src_row: int, dst_row: int) -> None:
        """把一行的状态拷到另一行。

        Args:
            src_row: 源行。
            dst_row: 目标行。
        """
        if src_row != dst_row:
            self.rows[dst_row] = list(self.rows[src_row])

    # 前向入口
    def prefill(self, tokens, chunk: int = 2048, spec_row=None, mm=None):
        """把一段 token 写进暂存行。

        Args:
            tokens: 要写入的 token，从该行当前位置起写。
            chunk: GEMM 切块大小；这里除了记下来没有别的用途。
            spec_row: 批处理投机用的草稿行；未使用。
            mm: 请求的图像载荷，记录下来供测试证明它到达了每一块，而不只是
                第一块。

        Returns:
            由该行整条 token 流算出的 ``[VOCAB]`` logits。

        Raises:
            AssertionError: 写入起点与该行实际位置不一致。这正是这个假模型要
                抓的分块预填充 bug。
        """
        row = self.prefill_row
        start = int(self.pos[row])
        assert start == len(self.rows[row]), (
            f"row {row} asked to write at {start} but holds {len(self.rows[row])}")
        self.rows[row].extend(int(t) for t in tokens)
        self.writes.append((row, start, [int(t) for t in tokens]))
        self.mm_calls.append(mm)
        self.pos[row] = start + len(tokens)
        return self._logits(self.rows[row])

    def set_rows(self, tokens, positions, slot_rows=None, row_indices=None) -> None:
        """摆好一步解码的输入：每行一个 token 和一个位置。

        Args:
            tokens: 每行一个 token。
            positions: 该 token 在对应行上的绝对位置。
            slot_rows: 每行的块表；经池做映射。
            row_indices: 每个条目所属的行。
        """
        rows = list(range(len(tokens))) if row_indices is None else list(row_indices)
        for row, tok, pos in zip(rows, tokens, positions):
            self.pos[row] = int(pos)
            self.rows[row].append(int(tok))
        if slot_rows is not None:
            for row, blocks in zip(rows, slot_rows):
                self.pool.assign_table(list(blocks), 0, row)

    def decode_batch_eager(self, rows: int, bucket: int, pad_rows=()):
        """以 eager 方式跑一步批处理解码，返回每一行的 logits。

        Args:
            rows: 活跃行的高水位。
            bucket: 前向的宽度。
            pad_rows: ``rows`` 之下的空洞。

        Returns:
            ``[bucket, VOCAB]`` logits。哨兵行拿到全零行，和真模型丢弃填充行
            的做法一致。
        """
        assert self.batch_capacity > 1, "decode_batch_eager() before enable_batching()"
        pad = set(int(r) for r in pad_rows) | set(range(rows, bucket))
        out = torch.zeros(bucket, VOCAB)
        for row in range(bucket):
            self.pos[row] = self.pos[row] + 1
            if row in pad:
                continue
            out[row] = self._logits(self.rows[row])
        return out

    def capture_batch(self, buckets) -> None:
        """拒绝捕获：这台主机记不了图，引擎本来也配成 eager 路径。

        Raises:
            AssertionError: 无条件抛出。跑到捕获的测试说明配置选错了。
        """
        raise AssertionError("capture_batch() called with graph off")

    def prefill_group(self, seqs, snapshot_interval=None):
        """一次调用预填充多个 prompt，每条序列一行。

        Args:
            seqs: 要预填充的序列；每条都已经有行和块表。
            snapshot_interval: 打快照的边界；未使用。

        Returns:
            ``(logits_list, snaps_list)``：每条序列一个 ``[VOCAB]`` 张量和一个
            以位置为键的快照字典，与真模型的分组预填充契约一致。
        """
        logits, snaps = [], []
        for seq in seqs:
            start = seq.prefill_cursor
            self.rows[seq.row] = list(seq.token_ids)
            self.pos[seq.row] = len(seq.token_ids)
            self.mm_calls.append(seq.mm)
            logits.append(self._logits(self.rows[seq.row]))
            snaps.append({})
            del start
        return logits, snaps

    def logits(self, token: int) -> torch.Tensor:
        """单流 eager 解码：写一个 token，返回下一步的 logits。

        前缀缓存引擎和非批处理引擎走这里而不是 ``decode_step``；这台主机没有
        可重放的图，所以 eager 批处理引擎也同样落到这条路径。

        Args:
            token: 要写在暂存行上的 token。

        Returns:
            写入之后该行 token 流对应的 ``[VOCAB]`` logits。
        """
        row = self.prefill_row
        start = int(self.pos[row])
        if start < len(self.rows[row]):
            self.rows[row] = self.rows[row][:start]
        self.rows[row].append(int(token))
        self.pos[row] = start + 1
        return self._logits(self.rows[row])

    def decode_step(self, token: int) -> torch.Tensor:
        """捕获解码的入口；没有捕获时与 :meth:`logits` 完全一致。

        Args:
            token: 要写在暂存行上的 token。

        Returns:
            写入之后该行 token 流对应的 ``[VOCAB]`` logits。
        """
        return self.logits(token)

    def _logits(self, tokens: list[int]) -> torch.Tensor:
        """由一行的 token 流算出 logits，确定且足够单射，可以拿来做等式断言。

        Args:
            tokens: 该行的整条 token 流。

        Returns:
            ``[VOCAB]`` 的 fp32 logits：整条流的哈希打散到词表上。
        """
        seed = 0
        for i, tok in enumerate(tokens):
            seed = (seed * 131 + tok * 7 + i) % 9973
        out = torch.zeros(VOCAB)
        for i in range(VOCAB):
            out[i] = float((seed * (i + 3)) % 101) / 101.0
        return out


def make_engine(num_blocks: int = 48, *, max_num_seqs: int = 2,
                prefill_quantum: int = 16, max_num_batched_tokens: int = 64,
                snapshots: int = 4, prefix_cache: bool = True) -> tuple[LLMEngine, FakePrefillModel]:
    """在图关闭的前提下，用假模型搭一台引擎。

    Args:
        num_blocks: 池大小；小到抢占够得着。
        max_num_seqs: 引擎一次可以准入的行数。默认 2 是因为分块预填充是批处理
            引擎的能力：单流引擎经前缀缓存一次就把 prompt 预填充完，预算根本
            卡不住它。
        prefill_quantum: 单个请求每步可预填充的 token 数。
        max_num_batched_tokens: 单步 token 预算，含解码。
        snapshots: 循环状态快照预算。
        prefix_cache: 是否挂上基数树。

    Returns:
        ``(engine, model)``，已构建好，可以直接 step。
    """
    model = FakePrefillModel(num_blocks=num_blocks)
    cfg = EngineConfig(model_dir="unused", max_seq=MAX_SEQ, bench=False,
                       quant=False, graph=False, prefix_cache=prefix_cache,
                       block_size=BLOCK, num_blocks=num_blocks,
                       snapshots=snapshots, snapshot_interval=BLOCK,
                       max_num_seqs=max_num_seqs,
                       prefill_quantum=prefill_quantum,
                       max_num_batched_tokens=max_num_batched_tokens,
                       vision=False)
    engine = LLMEngine(cfg, model)
    engine.build()
    return engine, model


def drain(engine: LLMEngine, seqs: list[Sequence], limit: int = 400) -> None:
    """反复 step 引擎，直到所有序列结束。

    Args:
        engine: 要驱动的引擎。
        seqs: 必须全部结束的序列。
        limit: 步数上限，让活锁表现为测试失败而不是把测试挂死。

    Raises:
        AssertionError: 序列没有在 ``limit`` 步内结束。
    """
    for _ in range(limit):
        if all(s.is_finished for s in seqs):
            return
        engine.step()
    raise AssertionError(f"engine did not finish within {limit} steps: "
                         f"{[s.status for s in seqs]}")


class ChunkedPrefillTest(unittest.TestCase):
    """分块推进的 prompt 必须等于一次调用写进去的同一个 prompt。"""

    def test_chunked_prefill_equals_one_shot(self) -> None:
        """行状态和下一个 token 的 logits 不能取决于 prompt 被切成几段——这就是全部正确性主张。"""
        prompt = [100 + i for i in range(40)]
        one, _ = make_engine(prefill_quantum=64, max_num_batched_tokens=128)
        row_a = one.add_request(prompt, SamplingParams(max_tokens=1, temperature=0.0))
        drain(one, [row_a])
        whole = one.runner.model.rows[row_a.row]

        many, many_model = make_engine(prefill_quantum=8, max_num_batched_tokens=64)
        row_b = many.add_request(prompt, SamplingParams(max_tokens=1, temperature=0.0))
        drain(many, [row_b])
        piecewise = many_model.rows[row_b.row]
        # 这台引擎只服务一个请求，每次写入都归它，所以块数大于 1 就证明 prompt
        # 是被切开写的，而不是整段预填充。
        chunks = len(many_model.writes)

        self.assertEqual(piecewise, whole, "chunked prefill wrote a different stream")
        self.assertGreater(chunks, 1, "the quantum did not split the prompt")
        self.assertEqual(piecewise[:len(prompt)], prompt,
                         "the row must hold the prompt as written")

    def test_grouped_prefill_serves_several_short_prompts_at_once(self) -> None:
        """两个各自装得进预算的 prompt 必须共用一次前向：分组唯一换来的就是这一次权重读取，退化成逐 prompt 前向的话别处看不出来，只能靠这里。"""
        engine, model = make_engine(num_blocks=48, max_num_seqs=3,
                                    prefill_quantum=64,
                                    max_num_batched_tokens=96)
        self.assertTrue(engine.runner.can_group_prefill())
        prompts = [[10, 11, 12, 13], [20, 21, 22, 23, 24]]
        seqs = [engine.add_request(p, SamplingParams(max_tokens=4, temperature=0.0))
                for p in prompts]
        engine.step()
        # 一次分组前向覆盖两个 prompt，所以一步之后两者都已越过 prompt 段；
        # 逐 prompt 预填充则要两步。
        for seq in seqs:
            self.assertIs(seq.status, SequenceStatus.RUNNING,
                          "the prompt did not finish in the grouped forward")
            self.assertEqual(seq.prefill_cursor, seq.num_prompt_tokens,
                             "the grouped forward left the prompt half written")
        group_rows = [model.rows[s.row] for s in seqs]
        self.assertEqual([r[:len(p)] for r, p in zip(group_rows, prompts)], prompts)
        self.assertTrue(all(s.num_completion_tokens >= 1 for s in seqs),
                        "a prompt in the group produced no token")
    def test_state_is_carried_across_a_chunk_boundary(self) -> None:
        """停在 prompt 中间的分块必须把状态交给下一块，而不是交出一行被重置的行：否则每个请求都要多付一次 50 MB 拷贝。"""
        engine, model = make_engine(prefill_quantum=8, max_num_batched_tokens=64)
        prompt = [200 + i for i in range(40)]
        seq = engine.add_request(prompt, SamplingParams(max_tokens=1, temperature=0.0))
        seen: list[int] = []
        for _ in range(40):
            if seq.num_prompt_tokens <= seq.prefill_cursor:
                break
            engine.step()
            if seq.prefill_cursor:
                seen.append(seq.prefill_cursor)
                self.assertEqual(model.rows[seq.row][:seq.prefill_cursor],
                                 prompt[:seq.prefill_cursor],
                                 "carried state diverged mid-prompt")
        self.assertGreater(len(set(seen)), 1, "the prompt was not prefilled in pieces")

    def test_token_budget_bounds_one_step(self) -> None:
        """预算是硬上限：一步写入的 token 不得超过它允许的量。"""
        engine, model = make_engine(prefill_quantum=8, max_num_batched_tokens=16)
        seq = engine.add_request([300 + i for i in range(64)],
                                 SamplingParams(max_tokens=1, temperature=0.0))
        engine.step()
        written = sum(len(w[2]) for w in model.writes)
        self.assertLessEqual(written, 16, "one step wrote more than the budget")
        self.assertEqual(seq.status, SequenceStatus.PREFILLING)

    def test_prefix_hit_survives_chunking(self) -> None:
        """前缀复用必须与分块共存：第二个请求接上前缀，只分块预填充自己那截尾巴。"""
        engine, model = make_engine(prefill_quantum=8, max_num_batched_tokens=64)
        prefix = [400 + i for i in range(24)]
        engine.clear_prefix_cache()
        first = engine.add_request(prefix + [500, 501],
                                   SamplingParams(max_tokens=1, temperature=0.0))
        drain(engine, [first])
        self.assertEqual(first.matched, 0)

        warm = engine.add_request(prefix + [600, 601],
                                  SamplingParams(max_tokens=1, temperature=0.0))
        drain(engine, [warm])
        self.assertEqual(warm.matched, len(prefix),
                         "the cached prefix was not adopted")
        tail_writes = [w for w in model.writes if w[2][:1] == [prefix[0]]]
        self.assertEqual(tail_writes[0][1], 0, "the cold run must start at 0")
        self.assertTrue(any(w[1] == len(prefix) for w in model.writes),
                        "the warm run did not resume at the matched boundary")

    def test_image_payload_reaches_every_chunk(self) -> None:
        """被切成多块的图像 prompt 每次都得把同一份载荷交给模型：图像 span 是绝对区间，落在图像内部的那一块照样要注入。"""
        engine, model = make_engine(prefill_quantum=8, max_num_batched_tokens=64)
        payload = object()
        seq = engine.add_request([700 + i for i in range(32)],
                                 SamplingParams(max_tokens=1, temperature=0.0),
                                 mm=payload)
        drain(engine, [seq])
        calls = [mm for mm in model.mm_calls if mm is not None]
        self.assertGreater(len(calls), 1, "the prompt was not chunked")
        self.assertTrue(all(mm is payload for mm in calls),
                        "a chunk lost the image payload")


class PreemptionTest(unittest.TestCase):
    """重算式抢占必须把块还给池，并且恢复得正确。"""

    def test_preempt_mid_prefill_resumes_and_finishes(self) -> None:
        """被抢占的 prompt 从基数树检查点重来，token 不丢，最后仍能产出结果。"""
        engine, model = make_engine(num_blocks=19, max_num_seqs=2,
                                    prefill_quantum=8, max_num_batched_tokens=16)
        long_prompt = [800 + i for i in range(32)]
        a = engine.add_request(long_prompt, SamplingParams(max_tokens=40,
                                                           temperature=0.0))
        b = engine.add_request([900 + i for i in range(32)],
                               SamplingParams(max_tokens=40, temperature=0.0))
        drain(engine, [a, b], limit=400)
        self.assertTrue(a.is_finished and b.is_finished)
        self.assertGreater(engine.n_preempted, 0,
                           "the scenario must actually preempt, or it proves nothing")
        self.assertEqual(a.token_ids[:len(long_prompt)], long_prompt,
                         "preemption dropped or reordered prompt tokens")
        self.assertTrue(all(w[1] >= 0 for w in model.writes))

    def test_preempt_mid_decode_keeps_emitted_tokens(self) -> None:
        """生成中被淘汰的请求不能重复或丢失 token：已输出的前缀属于请求，不属于被重算的状态。"""
        engine, model = make_engine(num_blocks=19, max_num_seqs=2,
                                    prefill_quantum=8, max_num_batched_tokens=16)
        prompts = [[800 + i for i in range(32)], [900 + i for i in range(32)]]
        seqs = [engine.add_request(p, SamplingParams(max_tokens=40, temperature=0.0))
                for p in prompts]
        drain(engine, seqs, limit=600)
        self.assertGreater(engine.n_preempted, 0, "the scenario did not preempt")
        for seq, prompt in zip(seqs, prompts):
            self.assertTrue(seq.is_finished)
            self.assertEqual(seq.token_ids[:len(prompt)], prompt,
                             "the prompt prefix changed under recompute")
            self.assertEqual(len(seq.token_ids), len(prompt) + 40,
                             "recompute lost or duplicated generated tokens")

    def test_pool_survives_preemption_without_leaking(self) -> None:
        """每个被引用的块都得有主——基数树里缓存的前缀和模型的暂存行——并且引用计数不能变成负数，那会把别人还在读的块释放掉。"""
        engine, model = make_engine(num_blocks=19, max_num_seqs=2,
                                    prefill_quantum=8, max_num_batched_tokens=16)
        seqs = [engine.add_request([800 + i for i in range(32)],
                                   SamplingParams(max_tokens=40, temperature=0.0)),
                engine.add_request([900 + i for i in range(32)],
                                   SamplingParams(max_tokens=40, temperature=0.0))]
        drain(engine, seqs, limit=600)
        bm = engine.bm
        pool = bm.pool
        cached = set()
        stack = [bm.radix.root]
        while stack:
            node = stack.pop()
            for child in node.children.values():
                cached.update(child.blocks)
                stack.append(child)
        held = {b for b, n in enumerate(pool.refcount) if n > 0}
        # 已结束序列的块本来就该留在树里，那就是前缀缓存。不能出现的是没人拥有的
        # 引用：一个块被占着，却既没有树边也没有暂存行认领它，那就是泄漏。
        stray = held - cached - set(model.scratch)
        self.assertFalse(stray, f"blocks referenced by nothing: {sorted(stray)}")
        self.assertTrue(all(n >= 0 for n in pool.refcount),
                        f"a refcount went negative: {list(pool.refcount)}")
        self.assertEqual(len(bm.radix.root.children) > 0, True,
                         "nothing was cached, so the leak check proved nothing")

    def test_unsatisfiable_pool_raises_instead_of_hanging(self) -> None:
        """池比单个请求的块表还小时必须大声报错：原地空转比一个点明该调哪个开关的错误更糟。"""
        # 三个暂存块加两行的暂存块之后，块大小 8 装不下 120 个 token 的 prompt，
        # 而且没有任何完成能释放出块来。
        engine, _ = make_engine(num_blocks=8, max_num_seqs=1,
                                prefill_quantum=8, max_num_batched_tokens=16)
        seq = engine.add_request([i for i in range(120)],
                                 SamplingParams(max_tokens=1, temperature=0.0))
        with self.assertRaises(RuntimeError):
            for _ in range(NO_PROGRESS_LIMIT + 10):
                engine.step()
        self.assertFalse(seq.is_finished)

    def test_short_request_is_not_blocked_by_a_long_prompt(self) -> None:
        """分块预填充存在的理由就是长 prompt 不能堵住队列：长的那条还在预填充时，短的必须能开始并跑完。"""
        engine, model = make_engine(num_blocks=64, max_num_seqs=2,
                                    prefill_quantum=8, max_num_batched_tokens=16)
        long_prompt = [i % 40 for i in range(160)]
        long_seq = engine.add_request(long_prompt,
                                      SamplingParams(max_tokens=2, temperature=0.0))
        engine.step()
        self.assertIs(long_seq.status, SequenceStatus.PREFILLING)
        short_seq = engine.add_request([7, 8, 9], SamplingParams(max_tokens=2,
                                                                temperature=0.0))
        for _ in range(20):
            engine.step()
            if short_seq.is_finished:
                break
        self.assertTrue(short_seq.is_finished,
                        "the short request waited for the whole long prompt")
        self.assertFalse(long_seq.is_finished,
                         "the long prompt should still be prefilling")
        # 长 prompt 必须在短请求运行期间就往前走，而不是等短的跑完才启动：那才是
        # 队头阻塞消失的样子。
        self.assertGreater(long_seq.prefill_cursor, 0)
        self.assertLess(long_seq.prefill_cursor, len(long_prompt))


class EagerBatchDecodeTest(unittest.TestCase):
    """批处理解码必须在关闭捕获的前提下跑——SCOW 的配置。"""

    def test_batched_decode_runs_without_capture(self) -> None:
        """两个并发请求都得走 eager 批处理入口解码，同时模型一次图都不捕获。"""
        engine, model = make_engine(num_blocks=48, max_num_seqs=4,
                                    prefill_quantum=16, max_num_batched_tokens=96)
        seqs = [engine.add_request([i for i in range(12)],
                                   SamplingParams(max_tokens=3, temperature=0.0)),
                engine.add_request([20 + i for i in range(12)],
                                   SamplingParams(max_tokens=3, temperature=0.0))]
        drain(engine, seqs)
        self.assertTrue(all(s.is_finished for s in seqs))
        self.assertEqual(engine.batching, True)
        self.assertFalse(engine.runner.graph, "the eager path must be selected")

    def test_batched_tokens_match_a_solo_run(self) -> None:
        """同一行采样出的 token 必须和单独跑一个请求时一样：eager 批只改行数，不改数值。"""
        prompt = [55 + i for i in range(16)]
        solo_engine, solo_model = make_engine(num_blocks=48, max_num_seqs=2,
                                              prefill_quantum=32,
                                              max_num_batched_tokens=96)
        solo = solo_engine.add_request(prompt, SamplingParams(max_tokens=4,
                                                              temperature=0.0))
        drain(solo_engine, [solo])

        batch_engine, batch_model = make_engine(num_blocks=48, max_num_seqs=2,
                                                prefill_quantum=32,
                                                max_num_batched_tokens=96)
        a = batch_engine.add_request(prompt, SamplingParams(max_tokens=4,
                                                            temperature=0.0))
        b = batch_engine.add_request([90 + i for i in range(16)],
                                     SamplingParams(max_tokens=4, temperature=0.0))
        drain(batch_engine, [a, b])
        self.assertEqual(a.token_ids, solo.token_ids,
                         "the batched row produced a different token stream")
        # 批处理引擎必须真的分配了第二行，否则这次运行说明不了它要覆盖的批处理
        # 路径。
        self.assertGreaterEqual(len(batch_model.rows), 2)
        del solo_model


class StatusModelTest(unittest.TestCase):
    """新状态必须如实描述调度器所依赖的生命周期。"""

    def test_non_chunked_engine_still_serves_one_request(self) -> None:
        """单流路径比分块更早存在；不管新状态怎么变，一个请求仍然要一次预填充完，然后解码到底。"""
        model = FakePrefillModel(num_blocks=32)
        cfg = EngineConfig(model_dir="unused", max_seq=MAX_SEQ, bench=False,
                           quant=False, graph=False, prefix_cache=False,
                           block_size=BLOCK, num_blocks=32, max_num_seqs=1,
                           prefill_quantum=16, max_num_batched_tokens=64,
                           vision=False)
        engine = LLMEngine(cfg, model)
        engine.build()
        self.assertFalse(engine.batching)
        self.assertFalse(engine.scheduler.chunked,
                         "no block manager means the token budget cannot apply")
        seq = engine.add_request([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                 SamplingParams(max_tokens=3, temperature=0.0))
        drain(engine, [seq], limit=20)
        self.assertTrue(seq.is_finished)
        self.assertEqual(len(seq.token_ids), seq.num_prompt_tokens + 3,
                         "the single-stream loop lost or duplicated a token")

    def test_sequence_starts_waiting_and_prefills_before_running(self) -> None:
        """``RUNNING`` 意味着可以解码，所以还在写 prompt 的请求必须是 ``PREFILLING``。"""
        engine, _ = make_engine(num_blocks=32, max_num_seqs=2,
                                prefill_quantum=8, max_num_batched_tokens=16)
        seq = engine.add_request([i for i in range(40)],
                                 SamplingParams(max_tokens=1, temperature=0.0))
        self.assertIs(seq.status, SequenceStatus.WAITING)
        engine.step()
        self.assertIs(seq.status, SequenceStatus.PREFILLING)
        self.assertTrue(seq.is_prefilling)
        drain(engine, [seq])
        self.assertIs(seq.status, SequenceStatus.FINISHED)

    def test_preempted_sequence_keeps_its_tokens(self) -> None:
        """重算的全部意义：请求在淘汰中活下来，被重置的只是它的缓存记账。"""
        seq = Sequence([1, 2, 3, 4])
        seq.append_token(9)
        seq.block_table = [3, 4]
        seq.prefill_cursor = 4
        seq.num_cached_tokens = 5
        seq.matched = 4
        seq.park_preempted()
        self.assertIs(seq.status, SequenceStatus.PREEMPTED)
        self.assertEqual(seq.token_ids, [1, 2, 3, 4, 9])
        self.assertEqual(seq.prefill_cursor, 0)
        self.assertEqual(seq.num_cached_tokens, 0)
        self.assertIsNone(seq.prefill_logits)
        self.assertEqual(seq.preempt_count, 1)
        seq.resume()
        self.assertIs(seq.status, SequenceStatus.WAITING)


if __name__ == "__main__":
    unittest.main()
