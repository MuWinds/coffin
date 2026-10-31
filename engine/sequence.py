"""一个请求，从准入到退休。

Module: ``coffin/``（顶层）。调度器排的是它，runner 推进的是它，前缀树给它
计数。``config.py`` 和本模块不 import 引擎里的任何其他模块，所以谁都可以依赖
它们。

Constraints:
    * 序列持有的是**长度**，不是设备缓冲区。投机步之后的权威位置是模型自己的
      计数器（``model.pos1``），所以调度器从 runner 那边刷新
      ``num_cached_tokens``，而不是在这里自增。
    * ``block_table`` 是共享状态：基数树给同一批块 id 计数，批处理路径还会把
      它们交给别的序列。凡是可能分配或释放块的调用之间，都不要缓存它的派生
      视图。
    * ``token_ids`` 是这个请求自己的记录：问过什么、生成过什么，所以它必须
      穿越每一次状态转换。抢占重置的是*缓存*字段（游标、logits、matched、
      已缓存长度），永远不动 token：重算会重放同一段前缀，而丢掉的生成 token
      就是真的丢输出，不是能重算的状态。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum, auto


class SequenceStatus(Enum):
    """一个请求的生命周期，按调度器能观察到的粒度划分。

    ``PREFILLING`` 和 ``RUNNING`` 都是可调度状态，也都在调度器的运行列表里；
    之所以分开，是因为只有 ``RUNNING`` 能交给解码步。prompt 还在写的序列只有
    部分 KV，解码它就会去注意模型还没算过的 token。

    ``PREEMPTED`` 是重算策略的停泊状态：序列保留自己的 token 和生成预算，但
    不占块、不占批里的行，等于回到等待队列里重新排队。
    """

    WAITING = auto()
    PREFILLING = auto()
    RUNNING = auto()
    PREEMPTED = auto()
    FINISHED = auto()


@dataclass(slots=True)
class SamplingParams:
    """单个请求的解码策略。

    ``eos`` 和 ``deadline`` 由引擎填，不由调用方填：停止 id 只有分词器知道，
    墙钟预算归服务端所有，两者都不该由构造请求的那一层决定。
    """

    temperature: float = 0.0
    max_tokens: int = 64
    ignore_eos: bool = False
    eos: set[int] = field(default_factory=set)
    deadline: float = 0.0


class Sequence:
    """一次生成请求，以及它当前持有的块表。"""

    _counter = itertools.count()

    def __init__(self, token_ids: list[int], params: SamplingParams | None = None,
                 seq_id: int | None = None) -> None:
        """把 prompt 包进引擎在生成过程中不断改写的状态里。

        Args:
            token_ids: prompt 的 token。必须拷贝，因为调用方会反复复用自己
                分词用的那个列表。
            params: 解码策略。``None`` 表示贪心默认值，内部调用方要的就是它。
            seq_id: 给需要确定性分配的测试用的显式 id；否则取进程级计数器。
        """
        self.seq_id = next(Sequence._counter) if seq_id is None else seq_id
        self.status = SequenceStatus.WAITING
        self.token_ids: list[int] = list(token_ids)
        self.num_prompt_tokens: int = len(self.token_ids)
        #: 当前已经写进池里的 prompt token 数，从 token 0 起算。分块预填充一次
        #: 推进一个 quantum，所以请求处于 ``PREFILLING`` 时它落后于
        #: ``num_prompt_tokens``；下一个分块就从这里开始。它与
        #: ``num_cached_tokens`` 不同：前者只数 prompt token，后者数所有写过的
        #: 位置，含已生成的 token。
        self.prefill_cursor: int = 0
        #: prompt 跑完后下一个 token 的 logits，在产出它的预填充和采样它的解码
        #: 步之间一直保存着。分块预填充把这两者摊到不同的调度步里（最后一个
        #: 分块结束时请求已经是 ``RUNNING``），所以这个张量必须跨过那段间隔。
        self.prefill_logits = None
        #: 本次准入是否已经跑过 :meth:`ModelRunner.begin_prefill`。被恢复的序列
        #: 保留自己的行和块，runner 因此不能再为它重做前缀匹配或重新认领行。
        self.prefill_started: bool = False
        #: 本次预填充到目前为止在块边界上打的循环状态快照，按 prompt 绝对位置
        #: 索引。每完成一个分块就插进基数树，而不是最后一次性插入，所以预填充
        #: 中途被抢占时，已经算出来的东西也都能复用。
        self.prefill_states_at: dict = {}
        #: 最近一次把这个序列放进批里的调度步号。供抢占评分使用，评分会优先
        #: 挑已经被饿过的受害者。
        self.last_scheduled_step: int = 0
        #: 这个请求被抢占并重来的次数。抢占评分会对已经被淘汰过的请求打折，
        #: 免得一个长 prompt 在队列其余部分推进时每一步都被踢出去。
        self.preempt_count: int = 0
        #: 最近一次被抢占时的调度步号。准入会跳过本步刚被抢占的请求，从而避免
        #: 它刚让出块就被立刻重新放进同一批块里。
        self.preempted_step: int = -1
        self.num_cached_tokens: int = 0
        self.num_scheduled_tokens: int = 0
        self.params: SamplingParams = params or SamplingParams()
        #: 这个序列持有引用的块。与基数树共享，批处理路径上还与其他序列共享。
        self.block_table: list[int] = []
        #: 在批处理模型缓冲区里的行号。``-1`` 表示状态还没铺进去。
        self.row: int = -1
        #: 这个请求预填充的墙钟耗时，由引擎写入。
        self.prefill_s: float = 0.0
        #: 上一次预填充从基数树里直接沿用的 token 数。
        self.matched: int = 0
        #: 模型自己结束序列时由 runner 写入。
        self.stop_reason: str | None = None
        #: prompt 的多模态载荷（``mm_input.MMInput``）或 ``None``。挂在这里，
        #: runner、块管理器和基数树都能看到图像 digest，不必再维护一张要同步的
        #: 旁表。
        self.mm = None

    def __len__(self) -> int:
        return len(self.token_ids)

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def num_completion_tokens(self) -> int:
        return self.num_tokens - self.num_prompt_tokens

    @property
    def completion_token_ids(self) -> list[int]:
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    @property
    def is_prefilling(self) -> bool:
        """prompt 是否还在往缓存里写。

        Returns:
            序列处于 ``PREFILLING`` 时为 ``True``。只有它为 ``False`` 才能跑
            解码步：``prefill_cursor`` 以下的 KV 是不完整的，对它做注意力会
            读到模型还没填过的槽位。
        """
        return self.status == SequenceStatus.PREFILLING

    @property
    def num_blocks_owned(self) -> int:
        """这个序列当前持有的块引用数。

        Returns:
            ``block_table`` 的长度，也就是只有活引用：被抢占的序列已经把自己的
            块表丢了。抢占评分读它来判断块在谁手里。
        """
        return len(self.block_table)

    def blocks_needed(self, block_size: int) -> int:
        """预填充完成后这个序列占用的物理块数。

        Args:
            block_size: 每块的 token 数，来自 ``EngineConfig``。

        Returns:
            向上取整后的块数。最后一个不满的块按整块计，因为 KV 是按整块分配的。
        """
        return (self.num_tokens + block_size - 1) // block_size

    def append_token(self, token_id: int) -> None:
        """追加一个解码出来的 token。

        Args:
            token_id: runner 采样出的 token，已经比对过停止 id。
        """
        self.token_ids.append(token_id)

    def finish(self, reason: str = "eos") -> None:
        """把请求标记为结束。

        Args:
            reason: ``"eos"``、``"length"``、``"deadline"`` 或模型给出的原因。
                它会作为 ``finish_reason`` 送到客户端，所以必须是 OpenAI 那一
                层接受的取值之一。
        """
        self.status = SequenceStatus.FINISHED
        self.stop_reason = reason

    def park_preempted(self) -> None:
        """重置一个即将被淘汰的序列的缓存状态。

        下次准入时会从头重算，所以凡是描述*它的 KV 在哪*的字段都清零，而 token
        ——prompt 的和已经生成的——都保留：已生成的这段现在就是请求的一部分，
        丢掉它要么导致输出重复，要么直接丢 token。

        调用方必须已经把用完的块插进基数树并释放了这个序列自己的引用；本方法
        故意不碰 ``block_table``，于是没释放的调用方会在下一次分配时失败，而
        不是静默泄漏。
        """
        self.status = SequenceStatus.PREEMPTED
        self.prefill_cursor = 0
        self.prefill_started = False
        self.prefill_logits = None
        self.prefill_states_at = {}
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.matched = 0
        self.preempt_count += 1

    def resume(self) -> None:
        """把被抢占的序列放回准入队列。

        准入走的是新请求的同一条路径：在整个 token 流（``prompt + 已生成``）上
        匹配前缀，恢复边界处的循环状态，再从那里开始预填充。状态一改，调度器
        才会把它当等待中的请求，而不是待牺牲的受害者。
        """
        self.status = SequenceStatus.WAITING

    def over_budget(self, now: float) -> bool:
        """这个请求是否已经用超了墙钟预算。

        Args:
            now: 当前墙钟时间。由外部传入而不是在这里读，一趟调度才能拿同一个
                时刻去衡量所有序列。

        Returns:
            设了截止时间且已经过去时返回 ``True``。截止时间为 ``0`` 表示「不
            限」，永不过期。
        """
        return bool(self.params.deadline) and now >= self.params.deadline
