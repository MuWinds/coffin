"""组装好的引擎：模型 + runner + 调度器 + 块管理器。

Module: ``coffin/``（顶层）。运行时服务对接的就是它；socket、分词器和响应协议
归入口所有。引擎负责请求生命周期、块池和模型 capture 下来的图。

    LLMEngine
      ├── ModelRunner      一步怎么执行（图回放 / 分页写入）
      ├── Scheduler        下一步有哪些序列
      └── BlockManager     分页池及其引用计数（开了前缀复用时还有基数树）

Constraints:
    * 一次 :meth:`step` 就是服务循环的一次迭代：新请求在暂存行上预填充，所有
      已有 KV 的运行中序列在同一个批前向里解码。:meth:`generate` 和
      :meth:`stream` 只是它外面的薄循环，循环本身不持有任何请求状态。
    * 服务线程只能在步与步之间调 :meth:`add_request`，绝不能在步中调用：
      ``step`` 会改调度器的运行列表，而且全程不加锁。
    * 结束的序列由发现结束的那一方释放（:meth:`_commit`、:meth:`_spec_commit`、
      超时处理）。漏掉一次就会泄漏一个批行，下一次准入会直接失败。
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Iterator

import torch

from .block_manager import BlockManager
from .config import EngineConfig
from .model_runner import ModelRunner
from .scheduler import Scheduler
from .sequence import SamplingParams, Sequence, SequenceStatus

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .prefix_engine import PrefixCacheEngine


#: 循环可以连续多少步一个事件都不产出，超过就认定池太小。偶尔两步没有产出是
#: 正常的（某个预填充分块只推进了游标）；连续很长一段则自己解不开，因为每条
#: 活着的序列都占着别人正在等的块。
NO_PROGRESS_LIMIT = 200


class LLMEngine:
    """一个加载好的 checkpoint，按请求驱动。"""

    def __init__(self, cfg: EngineConfig, model) -> None:
        """把各层装配起来，并把 KV 池开出来。

        Args:
            cfg: 引擎配置；决定是否批处理、是否用前缀缓存以及池的大小。
            model: 已加载的 ``engine_api`` 模型。分页在这里对它开启，所以调用方
                在本方法返回之前不能自己分配 KV。
        """
        self.cfg = cfg
        self.model = model
        self.runner = ModelRunner(model, cfg)
        self.scheduler = Scheduler(cfg, self.runner)
        self.batching = cfg.max_num_seqs > 1
        self.pc: "PrefixCacheEngine | None" = None
        self.bm: "BlockManager | None" = None

        # 两种模式共用同一个池。只要开了前缀缓存就挂基数树，批处理模式也不例外：
        # 那里 runner 在准入序列时匹配前缀，预填充完再插回去。两套池会让两种
        # 模式各有各的块 id 空间。
        if cfg.prefix_cache or self.batching:
            self.bm = BlockManager.attach(
                model, cfg.num_blocks, cfg.block_size, cfg.snapshots,
                with_radix=cfg.prefix_cache)
            self.runner.attach_blocks(self.bm)
        if cfg.prefix_cache and not self.batching:
            from .prefix_engine import PrefixCacheEngine

            self.pc = PrefixCacheEngine(
                model, num_blocks=cfg.num_blocks, block_size=cfg.block_size,
                max_state_snapshots=cfg.snapshots,
                snapshot_interval=cfg.snapshot_interval, bm=self.bm,
                graph=cfg.graph)
            self.runner.attach_prefix_cache(self.pc)
            # ``capture()`` 会预热批预填充路径，那条路径是通过槽位表写 KV 的。
            # 预留一个池永远不会再发出去的块，预热写才不会涂到缓存前缀上：写入
            # 需要一个槽位，而这个槽位不属于任何前缀。
            self._scratch = self.pc.pool.alloc()
            self.pc.pool.assign(self._scratch, 0)
        else:
            self._scratch = None

        self.eos: set[int] = set()
        self.n_requests = 0
        self.n_prompt_tokens = 0
        self.n_generated = 0
        #: 到目前为止因重算而抢占的次数。池看着健康但这个数在涨，说明池的容量
        #: 相对于调度器放进的并发度偏小。
        self.n_preempted = 0
        #: 连续多少个步没有产出事件；见 :meth:`_check_progress`。
        self._stalled_steps = 0
        #: 本步已经被淘汰的序列。同一步里一条序列不能当第二次受害者，也不能是
        #: 当前正在为它长块表的那条。
        self._preempted_this_step: set[Sequence] = set()
        self.load_s = 0.0
        self.build_s = 0.0
        #: 上一次预填充从基数树沿用的 prompt token 数。
        self.last_matched = 0

    def build(self) -> None:
        """capture 解码图，并把一步会走到的路径都预热一遍。

        必须在第一次 :meth:`step` 之前跑一次。``cfg.graph`` 打开时（默认）把解码
        前向录成一张设备图，一步才变成回放而不是重新描一遍。关掉时引擎保持
        eager，每 token 更慢，但这是回放不可信的后端上唯一安全的选择：一张回放
        过期输出的图不会崩，只会给出静默的错误答案。

        关掉 capture 之后批处理仍然可用：runner 调模型的 eager 批入口，而不是
        回放某个桶的图，铺的还是同一组缓冲区、跑的还是同一段代码。投机就不行
        ——验证步从构造上就是一次 capture 回放——所以关图的分支会把它拒掉。

        Raises:
            RuntimeError: 关了 capture 却配了投机解码。验证步要回放一张从未
                capture 过的图，所以在这里失败，而不是等到第一个投机步。
            RuntimeError: 配了投机解码，但 checkpoint 没有 MTP 头。构建时失败
                好过之后产出微妙不同的 token。
        """
        t0 = time.time()
        if self.batching and self.bm is not None:
            # 分页批处理服务：一个池、一组行，解码要么走 capture 好的桶图，要么
            # 走同一个桶的 eager 实现。这个分支在 capture 检查之前跑：关掉
            # capture 时根本不会录图，``_decode_batch`` 走 eager 入口。
            self.runner.enable_batching(self.cfg.max_num_seqs)
            if self.cfg.spec_k and (self.cfg.spec_batch or self.cfg.spec_sparse_k):
                if not self.cfg.graph:
                    raise RuntimeError(
                        "--spec-k needs graph capture; a verify step is a captured "
                        "replay by construction")
                if self.model.mtp is None:
                    raise RuntimeError("--spec-k needs the MTP head in the checkpoint")
                self.runner.enable_spec(self.cfg.spec_k, self.cfg.spec_sparse_k)
        elif not self.cfg.graph:
            if self.cfg.spec_k:
                raise RuntimeError(
                    "graph capture is off but speculation is configured; a verify "
                    "step is a captured replay, so it needs caps.graph")
            # 上面已经拒掉了投机，所以这次预填充写下的草稿模型 KV 永远不会被读到。
            # 喂它是整个 prompt 上一层全注意力加五个投影：实测的这块卡上 n=2048 时
            # 962.5 ms 的预填充里有 214.8 ms 花在这（../../PERF_COREX.md §13.2a）。
            # 在这里清而不是在 ``Qwen35.__init__`` 里清，是因为调用方也可以直接驱动
            # ``spec_generate``，那条路会回放 eager 的替代 capture，确实会读那份 KV。
            if self.model.mtp is not None:
                self.model.mtp_prefill = False
            self.model.reset()
        else:
            if not self.model.graph_ready:
                self.model.capture()
            if self.cfg.spec_k:
                if self.model.mtp is None:
                    raise RuntimeError("--spec-k needs the MTP head in the checkpoint")
                # 验证图按 k+1 个 token 来 capture：待定 token 加上草稿模型提的
                # k 个草稿。
                self.model.capture_verify(self.cfg.spec_k + 1)
            self.model.reset()
        self.build_s = time.time() - t0

    def add_request(self, token_ids: list[int],
                    params: SamplingParams | None = None,
                    mm=None) -> Sequence:
        """把一个 prompt 放进队列。

        图像载荷挂在序列上，runner、块管理器和基数树读的都是同一个对象：树要
        ``mm.digest`` / ``mm.spans`` 来给块键加盐，模型要嵌入。另开一张旁表就得
        在三个层之间保持同步，一点好处都没有。

        Args:
            token_ids: prompt 的 token，已经由对话模板渲染好。
            params: 解码策略；``None`` 表示用贪心默认值加引擎级停止 id。
            mm: 带图 prompt 传 ``mm_input.MMInput``，纯文本传 ``None``。

        Returns:
            排入队列的序列。调用方留着它来观察是否完成，以及之后读
            ``stop_reason``。
        """
        params = params or SamplingParams()
        if not params.eos:
            params.eos = set(self.eos)
        seq = Sequence(token_ids, params)
        seq.mm = mm
        self.scheduler.add(seq)
        self.n_requests += 1
        self.n_prompt_tokens += len(seq)
        return seq

    def step(self) -> list[tuple[Sequence, int | None, bool]]:
        """跑一次服务迭代。

        Returns:
            ``(seq, token, finished)`` 事件。事件只报告退休时 token 为 ``None``，
            这发生在超时的情况下——本来就无 token 可报。结束的那一步仍然会报告
            它产出的 token，流式客户端因此在 done 行之前收到最后一个 token，而
            不是一个空尾巴。
        """
        expired = self.scheduler.expire()
        events: list[tuple[Sequence, int | None, bool]] = [
            (seq, None, True) for seq in expired]
        for seq in expired:
            self._release(seq)
        decode, chunks = self.scheduler.schedule()
        # 只有发起方是禁区：受害者总得有人当，而抢占一条本步即将推进的请求，
        # 恰好就是给长不大的那个腾地方。它的解码还没跑，所以不会白做。
        self._preempted_this_step = set()
        # 先解码：prompt 不该拖慢一个已经在产出的请求。这一趟也可能触发淘汰，
        # 所以之后要重新检查分块列表——序列已经丢掉块的分块不能再跑。
        decode = self._make_room_for_decode(decode)
        if decode:
            events += self._run_decode(decode)
        chunks = [c for c in chunks
                  if c.seq.status is SequenceStatus.PREFILLING]
        if chunks:
            events += self._run_prefill(chunks)
        self._check_progress(events)
        return events

    def _check_progress(self, events: list[tuple[Sequence, int | None, bool]]) -> None:
        """池小到完全推不动时，直接大声失败。

        准入和块增长都可能把一个请求推迟一步，这在别的请求快要跑完、马上会把
        块还回来时是正确答案。但如果连续很多步什么都没动，就不会有人跑完了：
        每条活着的序列都在等别人手里的块，说明池比工作集还小。在这里抛异常，
        把服务循环挂死变成一个能上报的错误。

        Args:
            events: 本步产出的事件。任何 token 或退休都算有进展。

        Raises:
            RuntimeError: 引擎连续 :data:`NO_PROGRESS_LIMIT` 步没有任何进展。
        """
        if events:
            self._stalled_steps = 0
            return
        self._stalled_steps += 1
        if self._stalled_steps < NO_PROGRESS_LIMIT:
            return
        self._stalled_steps = 0
        raise RuntimeError(
            f"KV pool too small: no progress for {NO_PROGRESS_LIMIT} steps with "
            f"{len(self.scheduler.running)} running and "
            f"{self.bm.num_free if self.bm is not None else 0} free blocks of "
            f"{self.bm.num_blocks if self.bm is not None else 0}; raise --num-blocks "
            "or lower --max-num-seqs")

    def _run_prefill(self, chunks) -> list[tuple[Sequence, int | None, bool]]:
        """执行本步的预填充工作，并对跑完 prompt 的那个采样。

        一次拿到多个分块意味着每个已准入的 prompt 都能整体塞进 token 预算，这是
        唯一值得做分组前向的情况。单个分块则是一个请求最多推进 ``prefill_quantum``
        个 token；没跑完 prompt 时序列保持 ``PREFILLING``，下一步继续。

        Args:
            chunks: 调度器为本步给出的分块。

        Returns:
            引擎惯用形状的 ``(seq, token, finished)`` 事件。只推进了游标的分块
            不产出任何事件。
        """
        runner = self.runner
        if len(chunks) > 1 and runner.can_group_prefill():
            seqs = [chunk.seq for chunk in chunks]
            # 一组只由能整体塞进预算的 prompt 组成，所以这一趟前向会把它们全部
            # 跑完。
            for chunk in chunks:
                self._ensure_capacity(chunk.seq,
                                      chunk.seq.prefill_cursor + chunk.tokens)
            logits_list = runner.prefill_group(seqs)
            group_s = runner.last_prefill_s
            events: list[tuple[Sequence, int | None, bool]] = []
            for seq, logits in zip(seqs, logits_list):
                seq.prefill_s = group_s
                tok = runner.sample(logits, seq.params.temperature)
                events += self._commit(seq, tok)
            return events

        events = []
        for chunk in chunks:
            seq = chunk.seq
            if not runner.can_chunk_prefill():
                # 单流和前缀缓存引擎一次调用就把 prompt 预填充完，块记账也自己
                # 管，所以 token 预算对它们不适用。整个 prompt 由这一次调用写完，
                # 这也是序列在这里直接变成 ``RUNNING`` 而不是走
                # ``finish_prefill`` 的原因；不改状态下一步就不会解码它。
                logits = runner.prefill(seq)
                seq.prefill_s = runner.last_prefill_s
                self.last_matched = seq.matched
                seq.status = SequenceStatus.RUNNING
                tok = runner.sample(logits, seq.params.temperature)
                events += self._commit(seq, tok)
                continue
            # 先准入：匹配基数前缀会认领它能共享的那些块，容量检查因此正好不必
            # 再为这部分找地方。
            if not seq.prefill_started:
                self.runner.begin_prefill(seq)
            target = seq.prefill_cursor + chunk.tokens
            # 块在这里检查而不是在前向内部检查：长块表可能触发淘汰，而淘汰必须在
            # 请求占住它要的块之前发生。
            if not self._ensure_capacity(seq, target):
                # 已经没有可回收的东西：让请求保持预填充状态，等后面的步重试。
                # 让它失败等于把「池满了」变成一个被丢掉的客户端连接。
                continue
            logits = runner.prefill_chunk(seq, chunk.tokens)
            seq.prefill_s = runner.last_prefill_s
            if logits is None:
                # prompt 还没跑完：没什么可采样的，序列继续以 ``PREFILLING``
                # 状态留给后面的步。
                continue
            self.last_matched = seq.matched
            # ``prefill_chunk`` 已经记下它产出的东西；在这里读回来，这个字段
            # 才是「跑完的 prompt 最终落在哪个 logits」的唯一记录，而不是返回值
            # 的第二份拷贝。
            tok = runner.sample(seq.prefill_logits, seq.params.temperature)
            events += self._commit(seq, tok)
        return events

    def _run_decode(self, decode: list[Sequence]
                    ) -> list[tuple[Sequence, int | None, bool]]:
        """把每条可解码的序列各推进一个 token。

        Args:
            decode: prompt 已经跑完的序列。

        Returns:
            引擎惯用形状的 ``(seq, token, finished)`` 事件。
        """
        use_spec = (self.runner.spec
                    and all(s.params.temperature == 0 for s in decode))
        events: list[tuple[Sequence, int | None, bool]] = []
        if use_spec:
            # 只有 KV 读取成为主要开销之后，投机才划算。低于这个上下文档位时
            # 普通批处理步更便宜，所以一个步里可以混走两条路径：行与行互相独立，
            # 混着跑不比两条路径前后脚各跑一遍更贵。
            thr = self.cfg.spec_min_ctx
            spec_seqs = ([s for s in decode if s.num_cached_tokens >= thr]
                         if thr else list(decode))
            plain = [s for s in decode if s not in spec_seqs]
            if spec_seqs:
                events += self._spec_commit(spec_seqs)
            if plain:
                tokens = self.runner.decode(plain)
                for seq, tok in zip(plain, tokens):
                    events += self._commit(seq, tok)
        else:
            tokens = self.runner.decode(decode)
            for seq, tok in zip(decode, tokens):
                events += self._commit(seq, tok)
        return events

    def _ensure_capacity(self, seq: Sequence, tokens: int) -> bool:
        """让池放得下 ``seq`` 接下来的 ``tokens`` 个位置。

        先淘汰：缓存里那些前缀已经没人在生成，丢掉它只会在以后多算一次，不会
        影响活着的请求。只有淘汰都腾不出足够空间时，运行中的请求才会成为抢占
        候选。

        Args:
            seq: 马上要长块表的序列。本步唯一不会被抢占的就是它：它正是需要
                空间的那个请求。
            tokens: 这趟前向覆盖的 token 数。

        Returns:
            块可用时返回 ``True``。``False`` 表示已经无可回收；调用方于是把
            ``seq`` 留给后面的步，而不是去淘汰一份补不回来的工作。
        """
        bm = self.bm
        if bm is None:
            return True
        bs = bm.block_size
        # 一张 `k` 块的块表能寻址 `k*bs` 个位置，其中已经写过的部分早就付过账，
        # 只需要分配差额。
        held = len(seq.block_table) * bs
        need = max(0, (tokens - held + bs - 1) // bs)
        if need <= 0:
            return True
        bm.reserve(need)
        # 上面的淘汰可能腾得不够，所以一次只抢占一个请求，抢完再查。抢占一条块
        # 已被树接管的序列只是丢掉一份引用；树里的那份拷贝会变成淘汰候选，这
        # 正是要重复 reserve、而不能假定一次抢占就释放了整个请求的原因。
        for _ in range(len(self.scheduler.running) + 1):
            if bm.num_free >= need:
                return True
            victim = self.scheduler.preempt_victim({seq} | self._preempted_this_step)
            if victim is None:
                return False
            self._preempt(victim)
            bm.reserve(need)
        return bm.num_free >= need

    def _make_room_for_decode(self, decode: list[Sequence]) -> list[Sequence]:
        """只留下本步真能推进的那些解码序列。

        一个解码步恰好写一个新位置，所以序列只有在块表满的时候才需要一个新块。
        把服务不了的行剔掉，批前向才一致：一半推进、一半没推进会让序列和它们
        的 KV 对不上。

        Args:
            decode: 调度器为本步选中的序列。

        Returns:
            能跑的行。为空表示本步不解码。
        """
        bm = self.bm
        if bm is None or self.runner.pc is not None:
            # 前缀缓存路径自己管写入窗口，块是通过 ``SeqKV.reserve`` 分配的，
            # 不走块管理器。
            return decode
        bs = bm.block_size
        runnable = []
        for seq in decode:
            if seq.status is not SequenceStatus.RUNNING:
                # 本轮更早的一次抢占可能已经把这条序列踢出了批；现在跑它就会
                # 分配一张新块表，并在已经被重置、等着重算的状态上解码。
                continue
            pos = seq.num_cached_tokens
            if len(seq.block_table) * bs >= pos + 1:
                runnable.append(seq)
            elif self._ensure_capacity(seq, pos + 1):
                # 空间现在有了，所以块表在这里长，而不是留到前向里：批处理步
                # 跑到一半时再分配，就没有办法淘汰或抢占了。
                bm.ensure_blocks(seq, pos + 1)
                runnable.append(seq)
        return runnable

    def _preempt(self, seq: Sequence) -> None:
        """淘汰一条序列，并把它放回准入队列。

        Args:
            seq: 调度器选出的受害者。它的块和行回到池里，游标被重置；已经生成
                的 token 留在序列上，所以重算既不会丢输出也不会重复输出。
        """
        self.runner.preempt(seq)
        self.scheduler.park(seq)
        self._preempted_this_step.add(seq)
        self.n_preempted += 1

    def _spec_commit(self, seqs: list[Sequence]) -> list[tuple[Sequence, int | None, bool]]:
        """提交一个批友好的 MTP 步，并为每行采样下一个 token。

        投机只改变速度：草稿只有等于模型自己的 argmax 才会被接受，所以这里发出
        的 token 与普通解码步会产生的贪心 token 完全一致。

        Args:
            seqs: 要推进的序列。全部必须以温度 0 解码；随机采样无法用 argmax
                去验证。

        Returns:
            引擎惯用形状的 ``(seq, token, finished)`` 事件。
        """
        result = self.runner.spec_step(seqs)
        if result is None:                     # 超出草稿窗口：退回普通步
            tokens = self.runner.decode(seqs)
            events: list[tuple[Sequence, int | None, bool]] = []
            for seq, tok in zip(seqs, tokens):
                events += self._commit(seq, tok)
            return events
        accepted, logits = result
        events = []
        for seq, acc, row_logits in zip(seqs, accepted, logits):
            finished = False
            # ``acc[0]`` 是引擎已经追加过的待定 token。
            for tok in acc[1:]:
                if tok in seq.params.eos and not seq.params.ignore_eos:
                    self.scheduler.finish(seq, "eos")
                    self._release(seq)
                    events.append((seq, None, True))
                    finished = True
                    break
                seq.append_token(tok)
                self.n_generated += 1
                if seq.num_completion_tokens >= seq.params.max_tokens:
                    self.scheduler.finish(seq, "length")
                    self._release(seq)
                    events.append((seq, tok, True))
                    finished = True
                    break
                events.append((seq, tok, False))
            if finished:
                continue
            nxt = self.runner.sample(row_logits, seq.params.temperature)
            events += self._commit(seq, nxt)
        return events

    def _commit(self, seq: Sequence, tok: int) -> list[tuple[Sequence, int | None, bool]]:
        """追加一个采样出的 token，并应用停止条件。

        Args:
            seq: 产出 ``tok`` 的序列。
            tok: 采样出的 token id。

        Returns:
            给调用方的事件。触发停止时序列在这里被结束并释放，调用方不必自己
            记得去做。
        """
        params = seq.params
        if tok in params.eos and not params.ignore_eos:
            self.scheduler.finish(seq, "eos")
            self._release(seq)
            return [(seq, None, True)]
        seq.append_token(tok)
        self.n_generated += 1
        if seq.num_completion_tokens >= params.max_tokens:
            self.scheduler.finish(seq, "length")
            self._release(seq)
            return [(seq, tok, True)]
        return [(seq, tok, False)]

    def _release(self, seq: Sequence) -> None:
        """归还一条已结束的批处理序列的块和它的批行。

        单流和前缀缓存引擎的块由 ``PrefixCacheEngine`` 持有，所以这里对它们是
        空操作；只有批处理路径是按序列分配的。

        Args:
            seq: 已结束的序列。它的行必须在下次准入去找空闲行之前还回来。
        """
        if self.batching and self.bm is not None:
            self.bm.release(seq)
            self.runner.release_row(seq)

    # 便利循环：一次一个请求，测试和 run_chat 用。
    def stream(self, seq: Sequence) -> Iterator[int]:
        """在推进引擎的同时产出 ``seq`` 生成的 token。

        Args:
            seq: 事先用 :meth:`add_request` 排进队列的请求。

        Yields:
            生成即产出的 token id。其他序列的事件会被滤掉，所以同时驱动多个
            请求的调用方仍然只看到自己那条流。
        """
        while not seq.is_finished:
            for other, tok, _done in self.step():
                if other is seq and tok is not None:
                    yield tok
            if not self.scheduler.has_work():
                break

    @torch.inference_mode()
    def generate(self, token_ids: list[int],
                 params: SamplingParams | None = None,
                 mm=None) -> dict:
        """把一个请求跑到底。

        Args:
            token_ids: prompt 的 token。
            params: 解码策略；``None`` 用贪心默认值。
            mm: 带图 prompt 的 ``mm_input.MMInput``。

        Returns:
            ``{"tokens", "prefill_s", "decode_s", "matched", "stop_reason"}``。
            时间拆分在这里算，是因为 ``NOTES.md`` 里那些基准数字报的就是它；
            否则每个调用方都会各自算出一个略有差别的版本。
        """
        seq = self.add_request(token_ids, params, mm=mm)
        t0 = time.time()
        out = list(self.stream(seq))
        total = time.time() - t0
        pre = seq.prefill_s
        return {"tokens": out, "prefill_s": pre, "decode_s": max(total - pre, 0.0),
                "matched": seq.matched, "stop_reason": seq.stop_reason}

    def reset(self) -> None:
        """丢掉当前活跃序列的块，并把所有状态倒回去。

        单流复用的前提：模型在请求之间保留自己的位置和 KV，新请求必须把两者都
        清掉，否则它会读到上一个 prompt 的前缀。
        """
        if self.pc is not None:
            self.pc.release()
        self.model.reset()

    def release(self) -> None:
        """丢掉前缀缓存对引擎当前活跃序列的引用。"""
        if self.pc is not None:
            self.pc.release()

    def clear_prefix_cache(self) -> None:
        """丢掉所有缓存前缀。

        只在引擎空闲时合法：基数树给块计数的那些块，运行中的序列可能正在往里写。

        Raises:
            RuntimeError: 还有序列在等待或运行。此时清理会释放仍在使用的块，
                所以这里直接拒绝，而不是把池弄坏。
        """
        if self.scheduler.running or self.scheduler.waiting:
            raise RuntimeError("clear_prefix_cache() needs an idle engine")
        if self.bm is not None:
            self.bm.clear_prefix_cache()

    def stats(self) -> dict:
        """引擎与缓存计数器，运行时 ``ping`` 上报的就是这些。

        Returns:
            请求数和 token 总数，有池时再加块/前缀计数器。这些键名就是运行时
            序列化出去的名字，属于 API 面，不是可以随便改的标签。
        """
        out = {
            "requests": self.n_requests,
            "prompt_tokens": self.n_prompt_tokens,
            "generated_tokens": self.n_generated,
            "finished": self.scheduler.n_finished,
            "waiting": len(self.scheduler.waiting),
            "running": len(self.scheduler.running),
            "batching": self.batching,
            "preempted": self.n_preempted,
        }
        if self.pc is not None:
            out["prefix"] = self.pc.stats()
        elif self.bm is not None and self.bm.radix is not None:
            out["prefix"] = self.bm.stats()
        if self.bm is not None:
            out["blocks_used"] = self.bm.num_used
            out["blocks_total"] = self.bm.num_blocks
        return out

    @property
    def prefix(self) -> "PrefixCacheEngine | None":
        return self.pc
