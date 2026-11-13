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

预填充/解码分离时它同时是交接点：:meth:`PrefixCacheEngine.export_state` 把
预填充完的序列打包成纯 CPU 字典，:meth:`PrefixCacheEngine.adopt` 在另一个进程
里装回去继续解码。搬的只是「池里的 KV + 循环状态 + 位置」，块号不搬。

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

import os
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
        #: :meth:`adopt` 装进来的下一 token logits，来自预填充实例。
        self.pending_logits: torch.Tensor | None = None
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
        _trace = bool(os.environ.get("COFFIN_PC_TRACE"))
        _t0 = time.perf_counter() if _trace else 0.0
        blocks, matched, states = bm.match_prefix(tokens[:-1], require_states=True,
                                                  mm=mm)
        _t1 = time.perf_counter() if _trace else 0.0
        if self.seq is not None:
            self.seq.free()
        seq = SeqKV(pool, blocks, matched)
        self.seq = seq
        # 先接住匹配前缀，再释放旧请求，最后只为未命中部分腾空间。旧请求仍持有
        # 块时淘汰会失败，而按完整 prompt 淘汰还会提前丢掉本可复用的前缀。
        bm.reserve((len(tokens) + bs - 1) // bs - len(blocks) + 1)
        bm.assign(blocks, 0)
        if states is not None:
            model.restore_linear_states(states)
        else:
            # 没有可用快照：把整个模型倒回去（连 pos1 和 MTP 缓存一起清），
            # 重放 prompt 而不是续上。
            model.reset()
        model.pos1.fill_(matched)
        _t2 = time.perf_counter() if _trace else 0.0

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
        _t3 = time.perf_counter() if _trace else 0.0
        radix.insert(tokens, seq.blocks, states_at, mm=mm)
        if _trace:
            # 分段的秒表是这套命中链路上唯一能分开「匹配/恢复快照/后缀前向/插树」
            # 四件事的东西：它们在客户端只看得到合起来的 prefill_ms。
            from models.qwen35.parallel import context as _pctx

            _p = _pctx()
            _t4 = time.perf_counter()
            print(f"[pc] rank={0 if _p is None else _p.rank} n={len(tokens)} "
                  f"matched={matched} states={'y' if states is not None else 'n'} "
                  f"match={(_t1 - _t0) * 1e3:.1f} restore={(_t2 - _t1) * 1e3:.1f} "
                  f"suffix={(_t3 - _t2) * 1e3:.1f} insert={(_t4 - _t3) * 1e3:.1f} "
                  f"total={(_t4 - _t0) * 1e3:.1f}", flush=True)
        return logits

    @torch.inference_mode()
    def export_state(self) -> dict:
        """把当前序列在本 rank 上的状态打包成可以跨进程搬运的纯 CPU 字典。

        预填充/解码分离时，预填充实例跑完 prompt 就把状态交给解码实例。切法相同
        时 rank r 只导出自己那几层，接收端 rank r 原样装回去；切法不同（例如 PP2
        预填充对 EP4 解码）时用 :meth:`adopt_merged`，按这里给的全局层号把几份
        导出拼成一份。

        导出的是「池里已经写好的内容」，不是模型权重：线性注意力层的递推态按
        模型局部层号，全注意力层的 KV 按池的层轴下标。两者不是同一套编号，所以
        除了局部键还带上 ``linear_global`` / ``kv_global`` 两张换算表，跨切法
        拼装只认全局层号。

        Returns:
            ``pos``/``n_tokens``/``block_size``/``layer_begin``/``layer_end``/
            ``n_layers``/``nkv``/``hd`` 是 int；``blocks`` 是块号 list；
            ``linear`` 是 ``{局部层号: (递推态, 卷积态)}`` 的 CPU fp32 张量；
            ``kv`` 是 ``{池层轴下标: (k, v, kscale, vscale)}``，其中 ``k``/``v``
            是 ``[nkv, pos, hd]`` 的 int8、scale 是 ``[nkv, pos]`` 的 fp32；
            ``linear_global`` 是 ``{局部层号: 全局层号}``、``kv_global`` 是
            ``{池层轴下标: 全局全注意力层序}``；模型带 ``rope_delta`` 时多一个
            同名字段。整份字典可以直接 ``torch.save``。

        Raises:
            RuntimeError: 没有活跃序列；或模型没有暴露 ``layer_begin``/
                ``layer_end``，此时无法证明两端拓扑一致。
            ValueError: 序列长度与 ``pos1`` 不一致。那说明有人动过位置计数器却没
                提交，导出的状态会自带一个错位的长度，装到对面就是「输出是另一个
                模型」。
        """
        seq = self.seq
        if seq is None:
            raise RuntimeError("no active sequence to export")
        model, pool = self.model, self.pool
        layer_begin, layer_end = _layer_span(model)
        pos = int(model.pos1)
        if int(seq.num_tokens) != pos:
            raise ValueError(
                f"sequence length {int(seq.num_tokens)} disagrees with pos1 {pos}; "
                "the position counter moved without a commit")
        if len(block_ids := list(seq.blocks)) != _blocks_for(pos, pool.bs):
            raise ValueError(
                f"{len(block_ids)} blocks cannot hold {pos} tokens at "
                f"block_size {pool.bs}")
        linear = {i: (st.detach().to("cpu").contiguous(),
                      conv.detach().to("cpu").contiguous())
                  for i, (st, conv) in model.snapshot_linear_states().items()}
        kv = {}
        for idx in sorted(_pool_layers(model)):
            k, v, sk, sv = pool.gather(idx, pos)
            kv[idx] = (k.detach().to("cpu").contiguous(),
                       v.detach().to("cpu").contiguous(),
                       sk.detach().to("cpu").contiguous(),
                       sv.detach().to("cpu").contiguous())
        state = {
            "pos": pos,
            "n_tokens": pos,
            "blocks": block_ids,
            "block_size": int(pool.bs),
            "layer_begin": layer_begin,
            "layer_end": layer_end,
            "n_layers": int(pool.n_layers),
            "nkv": int(pool.nkv),
            "hd": int(pool.hd),
            "linear": linear,
            "kv": kv,
            # 跨切法拼装只认这两张表：局部编号在 PP 各段里从 0 重来一遍，按局部
            # 编号合并会把 20..39 层状态装到 0..19 层上，而且不会报错。
            "linear_global": {int(i): _global_linear(model, i) for i in linear},
            "kv_global": {int(idx): _global_pooled(model, idx) for idx in kv},
        }
        rope = getattr(model, "rope_delta", None)
        if torch.is_tensor(rope):
            state["rope_delta"] = rope[0].detach().to("cpu").contiguous()
        return state

    @torch.inference_mode()
    def adopt(self, state: dict, logits: torch.Tensor, tokens: list[int]) -> None:
        """把预填充实例交过来的状态装回本引擎，等价于「本地刚预填充完」。

        装完之后的 :meth:`decode_iter` 与在同一个实例里 ``prefill`` 完直接解码
        逐 token 相同——因为搬的正是解码要读的那几样东西：池里的 KV、线性注意力
        的递推态、位置计数器，以及块表。

        块号不搬。块 id 只在一个进程的池里有效，接收端自己 ``alloc`` 出等量的块
        再重建槽位表；``state`` 里的 ``blocks`` 只用来核对长度。

        Args:
            state: :meth:`export_state` 的产物，来自切法完全相同的另一个实例。
            logits: 预填充实例给出的下一个 token logits；存进
                ``self.pending_logits``，紧接着的 :meth:`decode_iter` 用它作起点。
            tokens: 那条序列的完整 prompt。它的长度必须等于状态里的 ``n_tokens``，
                否则 logits 与位置对不上。

        Raises:
            KeyError: 缺字段。半份状态装进去只会得到看似正常的错误输出。
            ValueError: 层区间、块大小、池层数、头数、头维、张量形状/dtype、
                token 数或块表中任何一处与本地不一致。这些对不上说明两个实例不是
                同一个拓扑，或者状态是用别的配置导出的。
        """
        model, pool, bm = self.model, self.pool, self.bm
        _check_state(state, model, pool, tokens)
        # logits 也先在动池之前查完：装了一半再抛，引擎停在半个序列上。
        if not torch.is_tensor(logits) or logits.numel() != model.vocab:
            raise ValueError(
                f"logits must hold {model.vocab} values, got "
                f"{getattr(logits, 'shape', type(logits))}")
        pos = int(state["n_tokens"])
        n_blocks = _blocks_for(pos, pool.bs)
        if self.seq is not None:
            self.seq.free()
            self.seq = None
        bm.reserve(n_blocks)
        blocks = [pool.alloc() for _ in range(n_blocks)]
        pool.assign_table(blocks, 0)
        self.seq = SeqKV(pool, blocks, pos)
        slots = pool.slots_for(torch.arange(pos, device=pool.pos2slot.device))
        for idx, (k, v, sk, sv) in state["kv"].items():
            # 导出的是 [nkv, pos, hd]，池的写入入口要 [pos, nkv, hd]。头轴的位置
            # 是两套契约里唯一不同的一处；nkv == pos 时写反了不会报错。
            pool.write_range(
                int(idx), slots,
                k.to(pool.k.device).transpose(0, 1).contiguous(),
                sk.to(pool.kscale.device).transpose(0, 1).contiguous(),
                v.to(pool.v.device).transpose(0, 1).contiguous(),
                sv.to(pool.vscale.device).transpose(0, 1).contiguous())
        states = {int(i): (st.to(model.pos1.device), conv.to(model.pos1.device))
                  for i, (st, conv) in state["linear"].items()}
        model.restore_linear_states(states)
        model.pos1.fill_(pos)
        if "rope_delta" in state:
            model.rope_delta[0].copy_(state["rope_delta"])
        self.pending_logits = logits.to(model.pos1.device)
        self.last_matched = 0

    @torch.inference_mode()
    def adopt_merged(self, parts: list[dict], logits: torch.Tensor,
                     tokens: list[int]) -> None:
        """把来自另一种切法的几份预填充状态拼成一份，装回本引擎。

        与 :meth:`adopt` 的差别只有一处：那边要求对面是这个实例的镜像，
        这里只认全局层号，所以 PP2 预填充可以直接交给 EP4 解码：
        每份 ``part`` 的``linear``/``kv`` 用 ``linear_global``/``kv_global``
        换算成全局层号后取并集，四个解码 rank 各自把同一份并集写进自己的池。

        装完之后的 :meth:`decode_iter` 等价于「本地刚用同一份 prompt 预填充
        完」，因为搬的正是解码要读的四样东西：池里的 KV、线性注意力递推态、
        位置计数器和块表。

        块号不搬：``part["blocks"]`` 只用来核对长度，接收端自己 ``alloc``。

        Args:
            parts: 若干份 :meth:`export_state` 的产物，覆盖互补的层区间；同一
                个全局层号出现两次视为切法写错。
            logits: 预填充端给出的下一个 token logits（PP 各段都算 logits，取
                第一段的即可）。
            tokens: 那条序列的完整 prompt；长度必须等于状态里的 ``n_tokens``。

        Raises:
            KeyError: 某份 ``part`` 缺字段。
            ValueError: 长度/块大小/头数/头维在各份之间不一致，并集与本地模型的
                线性层或池化层对不上，张量形状或 dtype 不符，prompt 长度与状态
                不符。
        """
        model, pool, bm = self.model, self.pool, self.bm
        pos, linear, kv, rope_delta = merge_parts(parts, tokens, model, pool)
        if not torch.is_tensor(logits) or logits.numel() != model.vocab:
            raise ValueError(
                f"logits must hold {model.vocab} values, got "
                f"{getattr(logits, 'shape', type(logits))}")
        n_blocks = _blocks_for(pos, pool.bs)
        if self.seq is not None:
            self.seq.free()
            self.seq = None
        _trace = bool(os.environ.get("COFFIN_PD_TRACE")) and self._pd_rank() == 0

        def _say(tag: str) -> None:
            """逐段打点：卡住时能一眼看出是分配、写 KV 还是恢复状态。"""
            if _trace:
                print(f"[adopt] {tag} t={time.time():.2f}", flush=True)

        _say(f"start pos={pos} blocks={n_blocks}")
        bm.reserve(n_blocks)
        blocks = [pool.alloc() for _ in range(n_blocks)]
        pool.assign_table(blocks, 0)
        self.seq = SeqKV(pool, blocks, pos)
        _say("blocks assigned")
        slots = pool.slots_for(torch.arange(pos, device=pool.pos2slot.device))
        for idx, (k, v, sk, sv) in kv.items():
            # 与 :meth:`adopt` 同一条契约：导出是 [nkv, pos, hd]，池要
            # [pos, nkv, hd]。
            pool.write_range(
                int(idx), slots,
                k.to(pool.k.device).transpose(0, 1).contiguous(),
                sk.to(pool.kscale.device).transpose(0, 1).contiguous(),
                v.to(pool.v.device).transpose(0, 1).contiguous(),
                sv.to(pool.vscale.device).transpose(0, 1).contiguous())
        _say("kv written")
        states = {int(i): (st.to(model.pos1.device), conv.to(model.pos1.device))
                  for i, (st, conv) in linear.items()}
        model.restore_linear_states(states)
        model.pos1.fill_(pos)
        if rope_delta is not None:
            model.rope_delta[0].copy_(rope_delta)
        self.pending_logits = logits.to(model.pos1.device)
        self.last_matched = 0
        _say("done")

    def _pd_rank(self) -> int:
        """本进程的张量并行 rank，供 :meth:`adopt_merged` 的追踪用。"""
        return int(os.environ.get("RANK", 0))

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


def _layer_span(model) -> tuple[int, int]:
    """本实例负责的层区间，也就是两个实例「切法相同」的唯一凭据。

    Args:
        model: 一个 ``PagedModel``。

    Returns:
        ``(layer_begin, layer_end)``，全模型口径的层号。

    Raises:
        RuntimeError: 模型没有暴露这两个属性。没有它们就无法证明对面的 rank r
            装的是同一段层，宁可拒绝。
    """
    begin = getattr(model, "layer_begin", None)
    end = getattr(model, "layer_end", None)
    if not isinstance(begin, int) or not isinstance(end, int):
        raise RuntimeError(
            "model does not expose layer_begin/layer_end; the two instances' "
            "pipeline split cannot be compared")
    return begin, end


def _pool_layers(model) -> dict[int, int]:
    """池层轴下标映射到模型局部层号。

    这是两套编号：池的层轴只数全注意力层，``snapshot_linear_states`` 的键只数
    线性注意力层，都用局部序号。把一套当另一套用不会报错，只会把状态装到别的
    层上。

    Args:
        model: 一个已经 ``enable_paging()`` 的 ``PagedModel``。

    Returns:
        ``{池层轴下标: 模型局部层号}``；没有全注意力层时为空。
    """
    out: dict[int, int] = {}
    for i, layer in enumerate(getattr(model, "layers", ())):
        pidx = getattr(getattr(layer, "mix", None), "pidx", None)
        if isinstance(pidx, int):
            out[pidx] = i
    return out


def _linear_layers(model) -> set[int]:
    """本实例里带循环状态的层，按模型局部层号。

    Args:
        model: 一个 ``PagedModel``。

    Returns:
        线性注意力层的局部层号集合，也就是 ``snapshot_linear_states`` 该有的键。
    """
    return {i for i, layer in enumerate(getattr(model, "layers", ()))
            if hasattr(getattr(layer, "mix", None), "state")}


#: :meth:`PrefixCacheEngine.adopt_merged` 要求的字段；跨切法拼装靠
#: ``linear_global``/``kv_global`` 两列，缺了就退化成按局部编号对齐。
_PD_REQUIRED = ("pos", "n_tokens", "block_size", "nkv", "hd", "linear", "kv",
                "linear_global", "kv_global")


def _global_linear(model, local_i: int) -> int:
    """局部线性层号换算成全模型层号。

    Args:
        model: 一个 ``PagedModel``，带 ``layer_begin``。
        local_i: 本实例的局部层号。

    Returns:
        全模型的层号，也就是 ``layer_begin + local_i``。
    """
    begin, _ = _layer_span(model)
    return begin + int(local_i)


def _global_pooled(model, local_pidx: int) -> int:
    """池层轴下标换算成全模型的全注意力层序。

    池的层轴只数全注意力层，而且从本实例的第一层开始数；全局层序要把本实例
    之前的所有全注意力层算进去，否则 PP 第二段会把第 6 个全注意力层当成第 1 个。

    Args:
        model: 一个 ``PagedModel``，带 ``layer_begin`` 和 checkpoint 配置。
        local_pidx: 池的层轴下标。

    Returns:
        全模型口径的全注意力层序号。
    """
    begin, _ = _layer_span(model)
    if begin == 0:
        return int(local_pidx)
    types = (getattr(model, "cfg", None) or {}).get("layer_types") or []
    if len(types) <= begin:
        raise RuntimeError(
            "checkpoint exposes no layer_types before layer "
            f"{begin}; the global pooled-layer index cannot be computed")
    return sum(1 for t in types[:begin] if t == "full_attention") + int(local_pidx)


def _blocks_for(n_tokens: int, block_size: int) -> int:
    """``n_tokens`` 个位置要几个整块，口径与 ``SeqKV.reserve`` 一致。

    Args:
        n_tokens: token 数。
        block_size: 每块的 token 数。

    Returns:
        向上取整的块数；``0`` 个 token 要 ``0`` 个块。
    """
    return (n_tokens + block_size - 1) // block_size


def _check_state(state: dict, model, pool, tokens: list[int]) -> None:
    """在动池之前把一份外来状态与本地契约逐项对齐。

    每一项对不上都意味着「装进去也能跑，但输出属于另一个模型」，所以这里全部
    抛异常而不是纠正或跳过。

    Args:
        state: :meth:`PrefixCacheEngine.export_state` 的产物。
        model: 本实例的 ``PagedModel``。
        pool: 本实例的池。
        tokens: 那条序列的完整 prompt。

    Raises:
        KeyError: 缺字段。
        ValueError: 层区间、块大小、池层数、头数、头维、张量形状或 dtype、token
            数、块表长度中任何一项不一致。
    """
    required = ("pos", "n_tokens", "blocks", "block_size", "layer_begin",
                "layer_end", "n_layers", "nkv", "hd", "linear", "kv")
    missing = [key for key in required if key not in state]
    if missing:
        raise KeyError(f"prefill state is missing {missing}")
    begin, end = _layer_span(model)
    if (int(state["layer_begin"]), int(state["layer_end"])) != (begin, end):
        raise ValueError(
            f"this instance owns layers [{begin}, {end}) but the state was "
            f"exported from [{int(state['layer_begin'])}, "
            f"{int(state['layer_end'])}); the pipeline split differs")
    if int(state["block_size"]) != int(pool.bs):
        raise ValueError(
            f"block_size {int(state['block_size'])} != local {int(pool.bs)}; "
            "the block tables are not interchangeable")
    if int(state["n_layers"]) != int(pool.n_layers):
        raise ValueError(
            f"state has {int(state['n_layers'])} pooled layers, local pool has "
            f"{int(pool.n_layers)}")
    if (int(state["nkv"]), int(state["hd"])) != (int(pool.nkv), int(pool.hd)):
        raise ValueError(
            f"state is {int(state['nkv'])}x{int(state['hd'])} per head, local "
            f"pool is {int(pool.nkv)}x{int(pool.hd)}")
    pos, n_tokens = int(state["pos"]), int(state["n_tokens"])
    if pos != n_tokens:
        raise ValueError(f"state disagrees with itself: pos {pos} vs n {n_tokens}")
    if n_tokens < 1:
        raise ValueError("state carries no tokens; nothing to decode from")
    if n_tokens > pool.max_seq:
        raise ValueError(
            f"state holds {n_tokens} tokens, beyond the local max_seq "
            f"{pool.max_seq}")
    if len(tokens) != n_tokens:
        raise ValueError(
            f"state holds {n_tokens} tokens but the prompt has {len(tokens)}; "
            "logits would be decoded from the wrong position")
    if len(state["blocks"]) != _blocks_for(n_tokens, pool.bs):
        raise ValueError(
            f"{len(state['blocks'])} blocks cannot hold {n_tokens} tokens at "
            f"block_size {int(state['block_size'])}")
    local_linear = _linear_layers(model)
    if set(map(int, state["linear"])) != local_linear:
        raise ValueError(
            f"state covers linear layers {sorted(map(int, state['linear']))}, "
            f"local model has {sorted(local_linear)}")
    for i, pair in state["linear"].items():
        st, conv = pair
        want = model.layers[int(i)].mix
        for name, got, ref in (("state", st, want.state[0]),
                               ("conv", conv, want.conv_state[0])):
            if (not torch.is_tensor(got) or got.shape != ref.shape
                    or got.dtype != ref.dtype):
                raise ValueError(
                    f"linear layer {int(i)} {name} must be {tuple(ref.shape)} "
                    f"{ref.dtype}, got "
                    f"{tuple(got.shape) if torch.is_tensor(got) else type(got)} "
                    f"{getattr(got, 'dtype', None)}")
    if set(map(int, state["kv"])) != set(_pool_layers(model)):
        raise ValueError(
            f"state covers pooled layers {sorted(map(int, state['kv']))}, local "
            f"model has {sorted(_pool_layers(model))}")
    for idx, quad in state["kv"].items():
        k, v, sk, sv = quad
        for name, got, want_dtype, want_shape in (
                ("k", k, pool.k.dtype, (pool.nkv, n_tokens, pool.hd)),
                ("v", v, pool.v.dtype, (pool.nkv, n_tokens, pool.hd)),
                ("kscale", sk, pool.kscale.dtype, (pool.nkv, n_tokens)),
                ("vscale", sv, pool.vscale.dtype, (pool.nkv, n_tokens))):
            if (not torch.is_tensor(got) or tuple(got.shape) != want_shape
                    or got.dtype != want_dtype):
                raise ValueError(
                    f"pooled layer {int(idx)} {name} must be {want_shape} "
                    f"{want_dtype}, got "
                    f"{tuple(got.shape) if torch.is_tensor(got) else type(got)} "
                    f"{getattr(got, 'dtype', None)}")
    if "rope_delta" in state:
        rope = getattr(model, "rope_delta", None)
        got = state["rope_delta"]
        if not torch.is_tensor(rope) or not torch.is_tensor(got) \
                or got.shape != rope[0].shape or got.dtype != rope[0].dtype:
            raise ValueError("state carries a rope_delta this model cannot take")


def merge_parts(parts: list[dict], tokens: list[int], model, pool
                ) -> tuple[int, dict[int, tuple[torch.Tensor, torch.Tensor]],
                           dict[int, tuple[torch.Tensor, ...]], torch.Tensor | None]:
    """把几种切法导出的状态按**全局层号**并成一份，并逐项校验。

    单流引擎把它装进自己的池（``adopt_merged``），批处理引擎把它按行装进
    批槽（``runtime_parts._pd_decode_batch``，每一行一次）。两处唯一的差别是
    往哪个行号写，所以合并与校验只留这一份。

    Args:
        parts: 若干份 :meth:`PrefixCacheEngine.export_state` 的产物，覆盖互补的
            层区间。
        tokens: 那条序列的完整 prompt。
        model: 目标实例的 ``PagedModel``。
        pool: 目标实例的池。

    Returns:
        ``(pos, linear, kv, rope_delta)``：位置数；按全局层号索引的线性态；
        按全局全注意力层序索引的 ``(k, v, kscale, vscale)``（``[nkv, pos, hd]``）；
        以及可选的位置偏置。

    Raises:
        KeyError: 某份 ``part`` 缺字段。
        ValueError: 各份之间长度/块大小/头数/头维不一致，并集与本地模型的线性层
            或池化层对不上，张量形状或 dtype 不符，prompt 长度与状态不符。
    """
    if not parts:
        raise ValueError("no prefill states to adopt")
    pos = None
    linear: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    kv: dict[int, tuple[torch.Tensor, ...]] = {}
    rope_delta = None
    for part in parts:
        missing = [key for key in _PD_REQUIRED if key not in part]
        if missing:
            raise KeyError(f"prefill state is missing {missing}")
        if int(part["block_size"]) != int(pool.bs):
            raise ValueError(
                f"block_size {int(part['block_size'])} != local {int(pool.bs)}; "
                "the block tables are not interchangeable")
        if (int(part["nkv"]), int(part["hd"])) != (int(pool.nkv), int(pool.hd)):
            raise ValueError(
                f"state is {int(part['nkv'])}x{int(part['hd'])} per head, "
                f"local pool is {int(pool.nkv)}x{int(pool.hd)}")
        if int(part["pos"]) != int(part["n_tokens"]):
            raise ValueError(
                f"state disagrees with itself: pos {int(part['pos'])} vs "
                f"n {int(part['n_tokens'])}")
        here = int(part["n_tokens"])
        if pos is None:
            pos = here
        elif pos != here:
            raise ValueError(
                f"stages disagree on length: {here} vs {pos}; they cannot "
                "describe the same request")
        if set(map(int, part["linear_global"])) != set(map(int, part["linear"])):
            raise ValueError("linear_global does not cover the linear layers")
        if set(map(int, part["kv_global"])) != set(map(int, part["kv"])):
            raise ValueError("kv_global does not cover the pooled layers")
        for i, pair in part["linear"].items():
            g = int(part["linear_global"][i])
            if g in linear:
                raise ValueError(f"linear layer {g} appears in two parts")
            linear[g] = pair
        for idx, quad in part["kv"].items():
            g = int(part["kv_global"][idx])
            if g in kv:
                raise ValueError(f"pooled layer {g} appears in two parts")
            kv[g] = quad
        if "rope_delta" in part:
            rope_delta = part["rope_delta"]
    assert pos is not None
    if pos < 1:
        raise ValueError("state carries no tokens; nothing to decode from")
    if pos > pool.max_seq:
        raise ValueError(
            f"state holds {pos} tokens, beyond the local max_seq "
            f"{int(pool.max_seq)}")
    if len(tokens) != pos:
        raise ValueError(
            f"state holds {pos} tokens but the prompt has {len(tokens)}; "
            "logits would be decoded from the wrong position")
    local_linear = _linear_layers(model)
    if set(linear) != local_linear:
        raise ValueError(
            f"merged state covers linear layers {sorted(linear)}, local "
            f"model has {sorted(local_linear)}")
    for i, (st, conv) in linear.items():
        want = model.layers[int(i)].mix
        for name, got, ref in (("state", st, want.state[0]),
                               ("conv", conv, want.conv_state[0])):
            if (not torch.is_tensor(got) or got.shape != ref.shape
                    or got.dtype != ref.dtype):
                raise ValueError(
                    f"linear layer {int(i)} {name} must be {tuple(ref.shape)} "
                    f"{ref.dtype}, got "
                    f"{tuple(got.shape) if torch.is_tensor(got) else type(got)} "
                    f"{getattr(got, 'dtype', None)}")
    if set(kv) != set(_pool_layers(model)):
        raise ValueError(
            f"merged state covers pooled layers {sorted(kv)}, local model has "
            f"{sorted(_pool_layers(model))}")
    for idx, (k, v, sk, sv) in kv.items():
        for name, got, want_dtype, want_shape in (
                ("k", k, pool.k.dtype, (pool.nkv, pos, pool.hd)),
                ("v", v, pool.v.dtype, (pool.nkv, pos, pool.hd)),
                ("kscale", sk, pool.kscale.dtype, (pool.nkv, pos)),
                ("vscale", sv, pool.vscale.dtype, (pool.nkv, pos))):
            if (not torch.is_tensor(got) or tuple(got.shape) != want_shape
                    or got.dtype != want_dtype):
                raise ValueError(
                    f"pooled layer {int(idx)} {name} must be {want_shape} "
                    f"{want_dtype}, got "
                    f"{tuple(got.shape) if torch.is_tensor(got) else type(got)} "
                    f"{getattr(got, 'dtype', None)}")
    if rope_delta is not None:
        rope = getattr(model, "rope_delta", None)
        if not torch.is_tensor(rope) or not torch.is_tensor(rope_delta) \
                or rope_delta.shape != rope[0].shape \
                or rope_delta.dtype != rope[0].dtype:
            raise ValueError("state carries a rope_delta this model cannot take")
    return pos, linear, kv, rope_delta
