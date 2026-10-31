"""下一步跑哪些序列。

Module: ``coffin/``（顶层）。夹在 ``llm_engine``（驱动循环）和 ``model_runner``
（执行一步）之间，所以它只许 import ``sequence`` / ``config``，绝不碰模型层或
内核层。

Constraints:
    * 真正的容量上限是批里的行数，不是序列数：解码行都活着时第 ``capacity-1``
      行留给预填充，所以准入按 ``max_num_seqs`` 来，再由 runner 挑桶。
    * 所有改动都是单线程的，由引擎的步进循环驱动。这里没有任何可重入的东西，
      也不加锁。
    * 开启分块预填充时，一个调度步最多跑一个预填充分块：暂存行只有一个，一步里
      两个分块反正也得把各自的循环状态搬移串起来。例外是一个能整体塞进预算的
      批，runner 会把它作为一个分组前向执行。

调度顺序，以及为什么是这个顺序：

* 先解码，因为已经在吐 token 的请求不该排在一个刚到的 prompt 后面（分块预填充
  存在的意义就是消除这种队头阻塞）；
* 然后用解码之后剩下的预算跑一个预填充分块，上限是 ``prefill_quantum``；
* 剩余预算不足一个块的步就是纯解码步：不足一块的分块要为几个 token 付掉一次
  完整的循环状态搬移。

准入不再要求整个 prompt 塞进预算。比预算长的 prompt 照样准入，每个调度步推进
一个 quantum，于是它开始产出，而不是把后面每个更短的请求都堵死。

抢占基于重算，且只能由分配失败触发：缓存已经淘汰到极限、块表还是长不大时，
引擎才会点名一个受害者。受害者的挑选刻意做得简单可解释——见
:meth:`Scheduler.preempt_victim`。
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .sequence import SequenceStatus

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .config import EngineConfig
    from .model_runner import ModelRunner
    from .sequence import Sequence


#: 一条序列要空闲多少步才能被抢占。取 0，是因为发起方已经被
#: :meth:`Scheduler.preempt_victim` 排除掉了，而抢占一条本步同样被选中的*别的*
#: 序列，恰好就是给长不大的那个腾地方：受害者在本次的 decode 还没跑，所以不会
#: 丢掉任何已做的工作。
PREEMPT_MIN_GAP = 0

#: 被抢占的序列要等多少步才能再次准入。没有它，受害者会立刻回到队列、重新
#: 匹配上自己刚让出的前缀、又占回同一批块，池一满就变成无休止的淘汰乒乓，
#: 而不是让一个请求先跑完、给另一个腾出空间。
PREEMPT_COOLDOWN = 1


@dataclass(frozen=True, slots=True)
class PrefillChunk:
    """本步一个请求还能预填充多少 prompt token。

    Attributes:
        seq: 要推进的请求。它已经在运行列表里、状态是 ``PREFILLING``，引擎只需
            执行这个分块。
        tokens: 分块可覆盖的 token 数，从 ``seq.prefill_cursor`` 起算。这是
            预算而不是精确长度：剩余更短时就预填充剩余部分，把 prompt 跑完。
    """

    seq: "Sequence"
    tokens: int


class Scheduler:
    """请求的准入与退休。"""

    def __init__(self, cfg: "EngineConfig", runner: "ModelRunner | None" = None) -> None:
        """建一个空的、绑定到某个引擎的调度器。

        Args:
            cfg: 引擎配置；``max_num_seqs``、``max_num_batched_tokens`` 和
                ``prefill_quantum`` 是准入限额。
            runner: 可选的反向引用，引擎用它把调度器眼里的行和 runner 实际铺进去
                的东西对一下。仅测调度器的测试传 ``None`` 也合法，此时走非分块
                路径。
        """
        self.cfg = cfg
        self.runner = runner
        self.waiting: deque["Sequence"] = deque()
        self.running: deque["Sequence"] = deque()
        self.max_num_seqs = max(1, cfg.max_num_seqs)
        self.max_num_batched_tokens = cfg.max_num_batched_tokens
        self.prefill_quantum = max(cfg.block_size, cfg.prefill_quantum)
        self.block_size = max(1, cfg.block_size)
        self.eos: set[int] = set()
        self.n_finished = 0
        #: 单调步计数器，一个步碰过的每条序列都会被打上。抢占评分读它来优先挑
        #: 已经被饿过的受害者。
        self.step_idx = 0

    @property
    def chunked(self) -> bool:
        """prompt 是否可以分片推进。

        分块需要批处理 runner：单流的前缀引擎一次调用就把 prompt 预填充完，缓存
        记账也自己管，下面的 token 预算限不住它。每步都重新读而不是在
        ``__init__`` 里缓存，是因为 runner 要到 ``LLMEngine.build`` 里才变成批
        处理，而那发生在调度器构造之后。没有 runner 时就是仅测调度器的场景，
        同样走非分块路径。

        Returns:
            runner 能分块时返回 ``True``，此时才允许引擎把一个 prompt 拆到多个
            调度步里。
        """
        return bool(self.runner is not None and self.runner.can_chunk_prefill())

    def add(self, seq: "Sequence") -> None:
        """把一个新请求放进队列。

        Args:
            seq: 要准入的请求。它自己没带停止 id 时继承引擎级的那一份，分词器的
                默认值就是这样传到 API 层构造出来的请求上的。
        """
        args = seq.params
        if not args.eos:
            args.eos = set(self.eos)
        self.waiting.append(seq)

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def schedule(self) -> tuple[list["Sequence"], list[PrefillChunk]]:
        """定下这一步：解码哪些、预填充哪些。

        先选解码，并把它的开销从 token 预算里扣掉，prompt 因此永远占不到正在
        生成的请求需要的槽位。预填充再拿剩下的，最多一个分块。

        Returns:
            ``(decode_seqs, prefill_chunks)``。``decode_seqs`` 里的序列 prompt
            已经跑完，本步可以走解码。``prefill_chunks`` 可能是空、一个分块，
            或者——当几个刚准入的 prompt 能整体塞进预算时——每个 prompt 一个
            分块，runner 会把它们作为一次分组前向执行。两者都为空表示没有可跑
            的东西。
        """
        self.step_idx += 1
        decode = [s for s in self.running if s.status is SequenceStatus.RUNNING]
        for seq in decode:
            seq.last_scheduled_step = self.step_idx
        if not self.chunked:
            return decode, self._schedule_atomic()
        return decode, self._schedule_chunked(self.max_num_batched_tokens - len(decode))

    def _schedule_atomic(self) -> list[PrefillChunk]:
        """整段准入一个 prompt，不看 token 预算。

        非分块的引擎本来就不管预算地预填充第一个准入的请求，因为比整份预算还
        长的 prompt 也必须能推进。那些路径上一条序列就是行的上限，所以也没有
        什么可分组的。

        Returns:
            零个或一个分块，总是覆盖该请求的全部剩余部分。
        """
        if not self.waiting or len(self.running) >= self.max_num_seqs:
            return []
        seq = self.waiting.popleft()
        self._begin(seq)
        # 序列在这里就加入 ``running``，而不是等前向返回：是准入让它成为一个活
        # 着的行，而且预填充中途崩掉的调用方仍然要在运行集合里看到它，而不是让
        # 它从两个队列里同时消失。
        self.running.append(seq)
        return [PrefillChunk(seq, len(seq.token_ids) - seq.prefill_cursor)]

    def _schedule_chunked(self, budget: int) -> list[PrefillChunk]:
        """在预算内准入，然后挑一个分块来跑。

        Args:
            budget: 付掉解码步之后剩下的 token 数。

        Returns:
            预算不够一个块（纯解码步）时为空；能整组放下时返回一整批分块；
            否则返回最久没被服务的那条预填充中请求的分块。
        """
        if budget < self.block_size:
            return []
        group = self._admit(budget)
        if group:
            return group
        prefilling = [s for s in self.running if s.status is SequenceStatus.PREFILLING]
        if not prefilling:
            return []
        # 轮转：等得最久的那条先走，而下面的时间戳让*另一条*下一步走。没有它，
        # deque 的顺序会让最早准入的那条一直霸占每一步。
        seq = min(prefilling, key=lambda s: (s.last_scheduled_step, s.seq_id))
        remaining = len(seq.token_ids) - seq.prefill_cursor
        tokens = min(self.prefill_quantum, remaining, budget)
        if tokens <= 0:
            return []
        seq.last_scheduled_step = self.step_idx
        return [PrefillChunk(seq, tokens)]

    def _begin(self, seq: "Sequence") -> None:
        """把一条序列从等待队列移进预填充状态。

        准入不拿预算去量 prompt：长 prompt 一次推进一个 quantum，要求它塞得下
        等于把分块本来要消除的队头阻塞又请回来。

        Args:
            seq: 刚从 ``waiting`` 弹出的序列。被抢占过的先放回准入流程，runner
                因此会像对待新请求一样对待它。
        """
        if seq.status is SequenceStatus.PREEMPTED:
            seq.resume()
        seq.status = SequenceStatus.PREFILLING
        seq.last_scheduled_step = self.step_idx

    def _admit(self, budget: int) -> list[PrefillChunk]:
        """在行数和预算允许的范围内准入等待中的请求。

        只有当每个已准入的 prompt 都能整体塞进剩余预算时，成批才值得分组：分组
        前向只省一次权重读取，而一个被拆到多个分块的 prompt 没法参与其中。

        *第一个*准入不看预算。比整份预算还长的 prompt 也必须能开始，开了分块之后
        它是靠拿走一个 quantum 开始的，而不是被拒绝。

        Args:
            budget: 整个预填充阶段可用的 token 数。

        Returns:
            要作为一次分组前向执行的分块；批不可分组时返回空列表。无论哪种情况，
            准入的序列都已在 ``running`` 里，所以没被分组的准入仍会成为分块候选。
        """
        admitted: list[PrefillChunk] = []
        spare = budget
        while self.waiting and len(self.running) < self.max_num_seqs:
            nxt = self.waiting[0]
            if self.step_idx - nxt.preempted_step <= PREEMPT_COOLDOWN:
                # 刚被淘汰过：现在准它进来会把淘汰本该腾出的块又占回去，两个
                # 请求就这样每步互换位置，谁也跑不完。
                break
            need = len(nxt.token_ids) - nxt.prefill_cursor
            if need <= 0:
                # 空 prompt 永远推进不了；把它留在队头会把整个队列永久堵死。
                self.waiting.popleft()
                continue
            chunk = min(self.prefill_quantum, need, spare)
            if not self._has_room(nxt, chunk):
                # 这个分块要的块不在池里，而占着它们的正是运行集合。硬准入只会
                # 在同一步里再被淘汰一次，所以等一个请求跑完把块还回来。
                break
            if admitted and need > spare:
                break
            self.waiting.popleft()
            self._begin(nxt)
            self.running.append(nxt)
            admitted.append(PrefillChunk(nxt, need))
            spare -= need
        if len(admitted) >= 2 and all(c.tokens <= budget for c in admitted):
            return admitted
        return []

    def _has_room(self, seq: "Sequence", chunk: int) -> bool:
        """池现在放得下 ``seq`` 的第一个分块吗。

        准入必须问这一句：没有开头要用的块就进来的请求，会在同一步里发现跑不动，
        于是要么被淘汰要么被跳过，两种结果都是它回到队列，同时挤掉了一个本来能
        跑的请求的槽位。前缀匹配可能直接给它共享块，所以这里的答案刻意偏乐观
        ——只数新增的那段尾巴——而引擎在真正长块表时还会再查一次。

        Args:
            seq: 待准入的候选。
            chunk: 它第一个分块覆盖的 token 数。

        Returns:
            这个分块要的新块都空着时返回 ``True``。没有 runner（仅测调度器）时
            不会按请求分配任何东西，所以答案恒为是。
        """
        runner = self.runner
        if runner is None or not self.chunked:
            return True
        target = min(seq.prefill_cursor + chunk, seq.num_tokens)
        # 满池子里装的主要是被缓存的前缀，淘汰能把这些还回来；所以走带淘汰的
        # 那条路去问空间，请求才不会被树本可以让出的块挡在后面。
        return runner.reserve_for(target) >= runner.blocks_for(target)

    def expire(self, now: float | None = None) -> list["Sequence"]:
        """退休所有墙钟预算已耗尽的运行中请求。

        Args:
            now: 用于判断的时刻；``None`` 表示读一次时钟，让整趟处理基于同一个
                时刻。

        Returns:
            已退休的序列，都已经过 :meth:`finish`。
        """
        now = time.time() if now is None else now
        expired = [s for s in self.running if s.over_budget(now)]
        for seq in expired:
            self.finish(seq, "deadline")
        return expired

    def preempt_victim(self, protected: set["Sequence"] | frozenset = frozenset()
                       ) -> "Sequence | None":
        """池满足不了分配时，挑一个请求淘汰掉。

        这套策略是在用公平性换可解释性，而且是故意的：不能死锁的是池，所以
        prompt 还没跑完的请求优先于已经在生成的请求被淘汰；持有块最多的先走，
        因为块就在它身上；评分还会给被淘汰过的请求打折，免得一个长 prompt 在
        队列其余部分推进时每一步都被点名。

        Args:
            protected: 不许被淘汰的序列，通常是调用方自己，加上本步已被选中
                去做解码的那些。

        Returns:
            受害者；没有任何可淘汰对象时返回 ``None``——这时调用方只能让自己
            的请求等着，而不是硬把分配做下去。
        """
        best = None
        best_key = None
        for seq in self.running:
            if seq in protected or seq.num_blocks_owned <= 0:
                continue
            if self.step_idx - seq.last_scheduled_step < PREEMPT_MIN_GAP:
                continue
            score = (seq.num_blocks_owned * 4
                     + (self.step_idx - seq.last_scheduled_step)
                     - seq.preempt_count * 8)
            # 预填充中的请求是更好的受害者：它还没有生成任何 token，重来一遍
            # 不必逐 token 复现。
            key = (seq.status is SequenceStatus.PREFILLING, score, -seq.seq_id)
            if best_key is None or key > best_key:
                best, best_key = seq, key
        return best

    def park(self, seq: "Sequence") -> None:
        """把被抢占的请求移出运行集合并重新排队。

        Args:
            seq: 引擎已经拿走它的块和批槽位的序列。它被放回等待队列的*头部*：
                被抢占的请求已经等过一次了，让新到的请求插到它前面，重算代价就
                没有上界了。
        """
        try:
            self.running.remove(seq)
        except ValueError:
            pass
        seq.park_preempted()
        seq.preempted_step = self.step_idx
        self.waiting.appendleft(seq)

    def finish(self, seq: "Sequence", reason: str = "eos") -> None:
        """退休一个请求。

        刻意做成幂等：引擎在解码循环和超时处理两处都会退休请求，同一条请求被
        finish 两次也不能把 ``n_finished`` 记乱。

        Args:
            seq: 要退休的请求。
            reason: 作为 ``finish_reason`` 转发给客户端的值。
        """
        seq.finish(reason)
        for queue in (self.running, self.waiting):
            try:
                queue.remove(seq)
            except ValueError:
                continue
        self.n_finished += 1

    # 引擎流式循环使用的记账辅助函数。
    @staticmethod
    def timed_out(seq: "Sequence", now: float | None = None) -> bool:
        """某条序列是否已经超过它的截止时间。

        Args:
            seq: 要判断的请求。
            now: 用于判断的时刻；``None`` 表示读一次时钟。

        Returns:
            请求设了截止时间且已经过去时返回 ``True``。
        """
        return seq.over_budget(time.time() if now is None else now)
