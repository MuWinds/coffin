"""分页 KV 池之上带基数缓存的服务，一次处理一个请求。

Module: ``coffin/``（顶层）。夹在 ``llm_engine`` 和池之间：它为一个请求驱动
``model``（一个 ``PagedModel``）、``block_manager`` 和 ``radix_cache``，自己不持有
任何存储。它不许知道自己在服务哪个模型族；与模型族相关的部分以 ``Model`` /
``PagedModel`` 契约的形式从 ``engine_api`` 传入。

处理一个请求时它把这些串起来：

    匹配前缀缓存 -> 沿用它的块并恢复循环状态 -> 只对没见过的后缀做预填充
    -> 把新前缀插进树里 -> 解码 -> 释放序列在块上的引用

这带来什么：见过一次的前缀——系统提示、工具前言、few-shot 开头、重新生成的
回答——只算一次，而不是每个请求算一次。可复用的单位是*按块对齐*的前缀，所以
``block_size=64`` 时，512 token 的系统提示会让下一个请求的预填充只剩一个块的
后缀。

Constraints:
    * 从构造上就是单流：解码步是一张针对一个 token 和一张槽位表的 CUDA 图，
      所以引擎一次只服务一条序列，池的槽位表属于当前活跃序列。多序列批解码
      需要另一种步进和另一张图，那在 ``model_runner`` 里，不在这里。
    * ``prefill`` 永远留下至少一个 token 重算。最后一个 token 的 logits 总得有
      出处，所以匹配是在 ``tokens[:-1]`` 上做的，这把复用上限卡在「除最后一块
      之外的全部」，也是全命中时仍能拿到有效 logits 的原因。
    * ``pos1`` 是模型自己的位置计数器，也是「写了多少 KV」的权威；这里的序列
      长度从它刷新（``commit_upto``），而不是另记一份。
    * 投机解码可以和这套并存，但 MTP 草稿模型自己的 KV 缓存不在前缀快照里：
      命中前缀之后草稿模型看到的是过期上下文，草稿质量变差（接受率下降），
      而*输出*仍然精确等同贪心，因为每份草稿都要主模型验证。在块边界恢复 MTP
      缓存能修好接受率，是个小后续（每个边界 4 MB，对比约 50 MB 的循环状态
      快照）。
    * 不加锁：``llm_engine`` 从它唯一的步进循环里调用本模块。
"""

from __future__ import annotations

import time

import torch

from cache.paged_cache import PagedKVCache, SeqKV
from .block_manager import BlockManager
from cache.radix_cache import RadixCache


class PrefixCacheEngine:
    """在分页池上做顺序请求服务，带基数前缀复用。"""

    def __init__(self, model, num_blocks: int = 512, block_size: int = 64,
                 max_state_snapshots: int = 8, snapshot_interval: int = 512,
                 bm: BlockManager | None = None, graph: bool = True) -> None:
        """挂到某个模型的池上，并定下快照策略。

        Args:
            model: 一个 ``PagedModel``。会对它调用 ``prefill``、``reset``、
                ``snapshot_linear_states`` 和 ``restore_linear_states``，所以它
                必须提供这些方法。
            num_blocks: 池大小，仅在本引擎自己建池时使用。
            block_size: 本引擎自己建池时每块的 token 数。它同时也是前缀复用的
                粒度。
            max_state_snapshots: 基数树最多保留的循环状态快照数。每个约
                50 MB，所以这是一笔显存预算。
            snapshot_interval: 两次快照之间的 token 数，向上取整到整块；它换来
                什么、放弃什么见属性的注释。
            bm: 与引擎其余部分共用的池和树。引擎会传一个进来，因为批处理和
                单流两条路径必须共用同一套块 id，而不是各建一个池。
            graph: 每个 token 回放 capture 下来的解码图。``False`` 跑 eager
                前向，这是 capture 不可信的后端唯一能走的路；token 一样，只有
                启动方式不同。
        """
        self.model = model
        #: 解码路径选择器；见 ``graph`` 参数。
        self.graph = bool(graph)
        #: 池和基数树归块管理器所有；本引擎只是在它们之上编排一个请求的
        #: 预填充/解码。
        self.bm = bm or BlockManager.attach(model, num_blocks, block_size, max_state_snapshots)
        self.pool: PagedKVCache = self.bm.pool
        self.radix: RadixCache = self.bm.radix
        #: 每多少个 token 打一次快照（向上取整到整块）。可复用单位是按块对齐的
        #: 前缀，但*每个*块边界都打快照在两方面都浪费：每个快照约 50 MB（24 层
        #: 的 [32,128,128] fp32），16k prompt 的全部 256 个边界加起来就是
        #: 12.8 GB；而且每 64 token 调一次 ``model.prefill``，会为一个很小的 M
        #: 付掉全套 GEMM/TRSM/调度准备开销。边界粗一点只多算衔接点到 prompt
        #: 之间的那些 token，对每轮增长几百 token 的对话而言可以忽略。
        self.snapshot_interval = snapshot_interval
        self.keep_states = max_state_snapshots
        self.seq: SeqKV | None = None
        self.last_matched = 0
        self.n_requests = 0
        self.n_prompt_tokens = 0
        self.n_hit_tokens = 0

    @torch.inference_mode()
    def prefill(self, tokens: list[int], mm=None) -> torch.Tensor:
        """沿用一段已缓存前缀，预填充剩下的部分，再把结果插回去。

        返回的东西与完整预填充 ``tokens`` 完全一致：下一个 token 的 logits。

        ``mm``（``mm_input.MMInput | None``）是请求的图像载荷。它起两个作用：
        前缀*匹配*会用图像 digest 加盐，所以带不同图像的 prompt 无法沿用这段
        KV；后缀预填充也会拿到它，模型因此能把图像嵌入拼进去并使用三维 mRoPE
        位置。``matched`` 仍然只会落在块边界上；它可能落在某个图像区间的*内部*，
        这没关系，因为嵌入是按 prompt 绝对下标切片的。

        Args:
            tokens: 完整 prompt，至少一个 token。最后一个 token 永远不从缓存里
                取——见模块级约束。
            mm: 请求的图像载荷，纯文本时为 ``None``。它必须与被要求给基数树
                加盐的是同一个对象，否则插入会落到之后任何匹配都不会看一眼的
                分支上。

        Returns:
            下一个 token 的 logits；prompt 最后一段总是真的跑过前向，所以前缀
            全命中时也仍有 logits。

        Raises:
            ValueError: ``tokens`` 为空。没有最后一个 token 可以出 logits，所以
                调用方必须在走到这里之前就拒掉请求。
        """
        if not tokens:
            raise ValueError("empty prompt")
        model, pool, bm = self.model, self.pool, self.bm
        bs = pool.bs
        # 先淘汰再分配：请求一旦握住了块，淘汰本该回收的叶子恰好就是它马上要
        # 占用的那些。
        bm.reserve((len(tokens) + bs - 1) // bs + 1)

        blocks, matched, states = bm.match_prefix(tokens[:-1], require_states=True,
                                                  mm=mm)
        if self.seq is not None:
            self.seq.free()
        seq = SeqKV(pool, blocks, matched)
        self.seq = seq
        bm.assign(blocks, 0)
        if states is not None:
            model.restore_linear_states(states)
        else:
            # 没有可用快照：把整个模型倒回去（连 pos1 和 MTP 缓存一起清），
            # 重放 prompt 而不是续上。
            model.reset()
        model.pos1.fill_(matched)

        states_at: dict[int, dict] = {}
        if matched and states is not None:
            # 命中点本身已经是一个快照，可以按同一个键重新插回去。
            states_at[matched] = states
        rest = tokens[matched:]
        seq.reserve(len(rest))
        logits = None
        # 快照只在*块*边界上才可复用，而块正是基数树匹配的单位，所以 prompt 在
        # 它最后一个块边界处切开：对齐的头部按 `snapshot_interval` 分片预填充，
        # 每片末尾打一个快照；没对齐的尾巴最后预填充，只有它的 logits 有意义。
        # 分片比一个块宽是为了速度——64 token 的 ``model.prefill`` 会为一个很小
        # 的 M 付掉全套 GEMM/TRSM/调度准备，所以 16k token 以前要调 256 次。
        # 存活快照数被基数树的预算卡住，因为每个约 50 MB（24 层的
        # [32,128,128] fp32），守住 16k prompt 的每个边界要 12.8 GB。不断增长的
        # 对话命中的是最深的边界，所以最早的那些先被丢掉。
        n_total = len(tokens)
        head_len = max(0, (n_total // bs) * bs - matched)
        step = max(bs, self.snapshot_interval // bs * bs)
        for c0 in range(0, head_len, step):
            c1 = min(c0 + step, head_len)
            logits = model.prefill(rest[c0:c1], mm=mm)
            states_at[matched + c1] = model.snapshot_linear_states()
            while len(states_at) > self.keep_states:
                # 保住命中点：那是下一个请求最可能接上的边界，永远不丢它。
                droppable = [k for k in states_at if k != matched]
                if not droppable:
                    break
                del states_at[min(droppable)]
        if head_len < len(rest):
            logits = model.prefill(rest[head_len:], mm=mm)
        seq.commit_upto(model.pos1)

        self.last_matched = matched
        self.n_requests += 1
        self.n_prompt_tokens += len(tokens)
        self.n_hit_tokens += matched
        radix = self.radix
        radix.insert(tokens, seq.blocks, states_at, mm=mm)
        return logits

    @torch.inference_mode()
    def decode_iter(self, logits: torch.Tensor, max_new: int,
                    eos: set[int] | None = None, temperature: float = 0.0):
        """逐 token 产出生成的 id（列表形式的 ``decode``）。

        每一步先开再关它自己的单 token 写入窗口，正是这一步让共享中的尾块在
        仍被树持有时走写时复制，而不是覆盖掉已缓存的前缀。

        Args:
            logits: :meth:`prefill` 或上一步给出的下一 token logits。每轮迭代
                都会替换；调用方的张量只被读取。
            max_new: 产出 token 数的上限。遇到 EOS 会提前结束生成器，所以实际
                可能更少。
            eos: 停止 token id。生成的 EOS 会被丢弃而不是产出，以匹配只期待
                内容 token 的线协议。
            temperature: 采样温度。``0`` 取 argmax，这是唯一逐次运行都能复现的
                模式。

        Yields:
            一次一个 token id，流式服务因此能在整段答案还没生成完时就转发每个
            token。
        """
        eos = eos or set()
        model, seq = self.model, self.seq
        out = 0
        while out < max_new:
            seq.reserve(1)
            if temperature > 0:
                p = torch.softmax(logits.float() / temperature, dim=-1)
                nxt = int(torch.multinomial(p, 1))
            else:
                nxt = int(logits.argmax())
            if nxt in eos:
                break
            yield nxt
            out += 1
            # 同一个前向有两个入口：图回放和 eager。两者都会铺好 token 并推进
            # ``pos1``，所以下面的提交无论走哪条路读到的都是同一个计数器。
            logits = model.decode_step(nxt) if self.graph else model.logits(nxt)
            # 模型的计数器才是权威：这一步实际写了几个位置是那张 capture 的图
            # 决定的，不是这个循环。
            seq.commit_upto(model.pos1)

    @torch.inference_mode()
    def decode(self, logits: torch.Tensor, max_new: int, eos: set[int] | None = None,
               spec_k: int = 0) -> list[int]:
        """在当前序列上解码，返回新生成的 token id。

        Args:
            logits: :meth:`prefill` 给出的下一 token logits。
            max_new: 新 token 数的上限。
            eos: 停止 token id；遇到 EOS 结束解码且不返回该 token。
            spec_k: 仅为签名兼容而接受，实际忽略：这条单流路径从不投机，因为
                草稿模型要靠 ``model_runner`` 的批验证才划得来。

        Returns:
            按顺序排列的生成 token id；第一个 token 就是 EOS 时可能为空。
        """
        return list(self.decode_iter(logits, max_new, eos))

    @torch.inference_mode()
    def generate(self, tokens: list[int], max_new: int, eos: set[int] | None = None,
                 spec_k: int = 0, mm=None) -> dict:
        """一次完整请求，返回生成的 token 以及预填充/解码的耗时拆分。

        Args:
            tokens: 完整 prompt。
            max_new: 生成 token 数的上限。
            eos: 停止 token id。
            spec_k: 这里忽略；见 :meth:`decode`。
            mm: 请求的图像载荷，纯文本时为 ``None``。

        Returns:
            ``tokens``（生成的 id）、``prefill_s`` 和 ``decode_s``（墙钟时间，
            拆开是为了让延迟报告能归因到贵的那一半）、``n_prompt``、``n_new``
            以及 ``matched``（前缀缓存命中的 token 数，也就是这个请求不必重算
            的部分）。

        Raises:
            ValueError: ``tokens`` 为空，由 :meth:`prefill` 向上抛出。
        """
        t0 = time.time()
        logits = self.prefill(tokens, mm=mm)
        t_pre = time.time() - t0
        t1 = time.time()
        out = self.decode(logits, max_new, eos, spec_k)
        return {"tokens": out, "prefill_s": t_pre, "decode_s": time.time() - t1,
                "n_prompt": len(tokens), "n_new": len(out), "matched": self.last_matched}

    def release(self) -> None:
        """丢掉当前活跃序列在它的块上的引用。

        必须在下一个请求沿用块之前跑，否则上一条序列的块表仍被计数，池就泄漏
        了；只要基数树还在缓存，块本身不会消失。
        """
        if self.seq is not None:
            self.seq.free()
            self.seq = None

    def stats(self) -> dict:
        """服务端状态回复用的前缀缓存与池计数器。

        Returns:
            请求数、token 数和块总数，加上复用比例、快照数及其估算 MiB、写时
            复制次数和被淘汰的块数。``hit_rate`` 做了取整，因为它是拿来上报的，
            不是拿来比较的。
        """
        return {
            "requests": self.n_requests,
            "prompt_tokens": self.n_prompt_tokens,
            "hit_tokens": self.n_hit_tokens,
            "hit_rate": round(self.n_hit_tokens / self.n_prompt_tokens, 4)
            if self.n_prompt_tokens else 0.0,
            "blocks_used": self.pool.num_used, "blocks_total": self.pool.num_blocks,
            "cached_blocks": self.radix.num_cached_blocks(),
            "snapshots": self.radix.num_snapshots(),
            "snapshot_mib": round(self.radix.snapshot_bytes() / 2**20, 1),
            "cows": self.pool.n_cows, "evicted_blocks": self.radix.num_evicted_blocks,
        }
