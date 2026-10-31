"""多 token 预测头：投机解码的草稿器。

Module: ``coffin/models/qwen35``。架在本族自己的块之上：复用 ``FullAttn``（作为
``mtp.layers.0``）、``MLP``、绑定嵌入和批处理投影，由 ``model.py`` 驱动
（``spec_generate`` / ``spec_step`` / ``spec_step_batch``）。它持有第二套分页池
给草稿器 KV，不能比构建它的模型活得更久。

一个全注意力层：给它主模型在位置 ``i`` 的隐含状态和 token ``i+1`` 的嵌入，它预测
``i+2``。草稿过程是固定缓冲区上的一张捕获图，链式 argmax 在设备上解析，所以 K 份
草稿就是 K 次重放，主机在中间看不到任何 token。

Constraints:
    * 草稿链完全活在设备缓冲区里（``h_in``、``tok_in``、``logits_out``、
      ``b_*``）。重放之间读一次主机，每个草稿 token 就要一次完整同步（batch 1 时
      约 0.5 ms），投机也就没有意义了，所以链式 argmax 必须留在设备上。
    * 图会把缓冲区地址和池张量烘进去。``capture_draft`` / ``capture_draft_batch``
      必须在每次分配之后跑，而它们写的缓冲区在图存活期间绝不能被重新分配。
    * 捕获经 ``ops.runtime``，所以捕不了的后端（算子表混了库内核和手写内核）改为
      即时执行同一段主体。两条路径下链式 argmax 都留在设备上，无主机往返的约束
      因此在两边都成立。
    * 草稿器 KV 在主模型的草稿位置被改写、由主校验回滚，所以它只需要同样的窗口；
      超过 ``batch_len`` 的 prompt 会被截断并通过 ``check_len`` 上报，而不是写
      越池。
    * ``self.pos`` 是 KV 地址，``rope_delta`` 是 mRoPE 偏移。mRoPE 只动旋转——
      和主注意力是同一种拆分，两者搞混会在图像之后用错误的位置旋转而不报错。
    * 所有改写都是单线程的，由 ``llm_engine`` 驱动；没有加锁，也不许加。
"""

from __future__ import annotations

import torch

from backends import device as _device
from models.qwen35.batch import batch_proj
from models.qwen35.blocks import MLP
from models.qwen35.full_attn import FullAttn
from models.qwen35.tuning import PREFILL_ATTN_TILE, PREFILL_MTP_CHUNK
from ops import runtime
from ops.dtypes import D
from ops.embed import TiedEmbedding
from ops.gemv import _proj
from ops.norm import RMSNorm
from cache.paged_cache import PagedKVCache


class MTP:
    """多 token 预测头：一个全注意力层，给它主模型的隐含状态 ``h_i`` 和 token
    ``i+1`` 的嵌入，它预测 ``i+2``。

    草稿过程把自己的输出串起来（上一步的 ``h`` 喂给下一步），下一个 token 是设备上
    的 ``logits.argmax()``，所以一整份草稿从不同步回主机。之后由主模型校验这份
    草稿。
    """

    def __init__(self, sd: dict, cfg: dict, bench: bool, max_seq: int) -> None:
        """加载 MTP 头并分配单流草稿缓冲区。

        Args:
            sd: 完整 state dict；这个头读 ``mtp.*`` 这些键，外加绑定的
                ``model.language_model.embed_tokens.weight``。
            cfg: 模型配置；``hidden_size`` 和注意力几何直接传给内嵌的
                ``FullAttn`` 层。
            bench: 透传给 ``_proj``，用于选择基准权重布局。
            max_seq: 交给内嵌全注意力层的上下文窗口，它据此确定稠密（非分页）
                KV 缓存的尺寸。
        """
        p = "mtp"
        self.embed = TiedEmbedding(sd["model.language_model.embed_tokens.weight"], bench)
        self.pre_norm_emb = RMSNorm(sd[f"{p}.pre_fc_norm_embedding.weight"])
        self.pre_norm_hid = RMSNorm(sd[f"{p}.pre_fc_norm_hidden.weight"])
        self.fc = _proj(sd, [f"{p}.fc"], bench)
        self.in_ln = RMSNorm(sd[f"{p}.layers.0.input_layernorm.weight"])
        self.attn = FullAttn(sd, f"{p}.layers.0.self_attn", cfg, bench, max_seq)
        self.post_ln = RMSNorm(sd[f"{p}.layers.0.post_attention_layernorm.weight"])
        self.mlp = MLP(sd, f"{p}.layers.0.mlp", bench)
        self.norm = RMSNorm(sd[f"{p}.norm.weight"])
        self.pos = torch.zeros(1, dtype=torch.long, device=_device)
        #: 正在草拟的请求的 mRoPE 偏移，与 ``Qwen35.rope_delta`` 对应：草稿器的旋转
        #: 读 ``pos + rope_delta``，而 ``pos`` 继续寻址 KV 槽位。调用方
        #: （``Qwen35.spec_step`` / ``spec_step_batch``）在草稿前把该行的值拷进来，
        #: 因为捕获的草稿图只能加一个缓冲区，不能按它分支。
        self.rope_delta = torch.zeros(1, dtype=torch.long, device=_device)

        # 草稿过程是一张捕获图。链 (h, token) -> (h', token') 完全活在这些设备
        # 缓冲区里，所以 K 份草稿就是 K 次重放，没有主机往返、也没有逐算子 launch
        # 开销；链式 argmax 在重放之间于设备上解析。
        hidden = cfg["hidden_size"]
        self.tok_in = torch.zeros(1, dtype=torch.long, device=_device)
        self.h_in = torch.zeros(hidden, dtype=D, device=_device)
        self.logits_out = torch.zeros(self.embed.shape[0], dtype=D, device=_device)
        self._ctx = None
        self._graph: runtime.Capture | None = None

        # 批处理草稿器（每个序列一行）。与上面的单流缓冲区完全分开，投机路径因此
        # 不受影响。
        self._sd, self._cfg, self._bench = sd, cfg, bench
        # 即使调用方在模型构建后清空 ``sd``，也要让注意力张量活着（批处理草稿器
        # 会延迟构建它的分页孪生体）。
        self._attn_sd = {k: v for k, v in sd.items()
                         if k.startswith("mtp.layers.0.self_attn.")}
        self._max_seq = max_seq
        self.batch_capacity = 0
        self.batch_buckets: tuple[int, ...] = ()
        self.batch_len = 0               # 一行能寻址的草稿器位置数
        self.pool: PagedKVCache | None = None
        self.attn_b: FullAttn | None = None
        self.b_pos = self.b_tok = self.b_h = self.b_logits = None
        self.b_h_prev = None
        self._ctx_lens = None
        self._draft_graphs: dict[int, runtime.Capture] = {}

    def reset(self) -> None:
        """把单流草稿器倒回位置 0。

        Returns:
            无。MTP 的 KV 按位置索引且从不读越过 ``pos``，所以把计数器移回去就是
            完整回滚；内嵌注意力层自己的缓存也一并清空。
        """
        self.attn.reset()
        self.pos.zero_()
        self.rope_delta.zero_()

    def _draft_impl(self) -> None:
        """在固定缓冲区上跑一步草稿；尾部的 token 喂给下一步。

        Returns:
            无。读 ``tok_in`` / ``h_in`` / ``pos``，写回新的隐含状态、logits 和
            推进后的位置——全部就地，因为捕获图重放的就是这段主体。
        """
        cos_tbl, sin_tbl, mask_idx = self._ctx[0], self._ctx[1], self._ctx[2]
        e = self.embed.lookup(self.tok_in).view(-1)
        x = self.fc(torch.cat([self.pre_norm_emb(e), self.pre_norm_hid(self.h_in)], dim=-1))
        h = x + self.attn(self.in_ln(x), self.pos, cos_tbl, sin_tbl, mask_idx,
                          self.rope_delta)
        h = h + self.mlp(self.post_ln(h))
        h = self.norm(h)
        self.h_in.copy_(h)
        self.logits_out.copy_(self.embed(h))
        self.pos.add_(1)

    def capture_draft(self, ctx, warmup: int = 3) -> None:
        """捕获一步草稿；:meth:`draft` 把它的 K 次重放串起来。

        Args:
            ctx: 主模型的 ``(cos_tbl, sin_tbl, mask_idx)``。只持有不拷贝，因为图
                读的就是这几个张量本身。
            warmup: 捕获之前在旁路流上的重放次数。内核要先按形状预热，否则捕获会
                录下一次它无法重放的分配。

        Returns:
            无。``_graph`` 持有捕获好的那一步；再次调用会替换它，捕获前后各重置
            一次位置计数器，好让第一次重放从 0 开始。
        """
        self._ctx = ctx
        self.reset()
        self._graph = runtime.capture(self._draft_impl, warmup)
        self.reset()

    def draft(self, h_prev: torch.Tensor, tok0: torch.Tensor, K: int,
              out_toks: torch.Tensor) -> None:
        """从 ``(h_prev, tok0)`` 串出 ``K`` 份草稿；把 K+1 个 id 写进 ``out_toks``。

        ``self.pos`` 必须已经停在 ``h_prev`` 对应的位置上。

        Args:
            h_prev: 主模型对最后一个已接受 token 的隐含状态。
            tok0: 那个 token 的 id，也就是链的第一个输入。
            K: 要产出的草稿数；即捕获步的 ``K`` 次重放。
            out_toks: 至少 ``K+1`` 项的 long 缓冲区。``out_toks[0]`` 是 ``tok0``，
                ``out_toks[i]`` 是第 ``i`` 份草稿——校验步骤按这个布局索引。

        Raises:
            AssertionError: 图还没捕获。``capture_draft`` 必须在模型构建之后、
                第一次草稿之前跑。
        """
        assert self._graph is not None, "draft graph not captured"
        self.h_in.copy_(h_prev)
        self.tok_in.copy_(tok0.reshape(1))
        # 用切片而不是索引：一维张量上的 ``out_toks[0]`` 是零维视图，把 [1] 张量
        # ``copy_`` 进去过不了形状检查。
        out_toks[0:1].copy_(self.tok_in)
        for i in range(1, K + 1):
            self._graph.replay()
            self.tok_in.copy_(self.logits_out.argmax())
            out_toks[i:i + 1].copy_(self.tok_in)

    def step(self, h_prev: torch.Tensor, tok: torch.Tensor, ctx) -> tuple[torch.Tensor, torch.Tensor]:
        """在 ``self.pos`` 处做一步即时执行的草稿；返回 (h', logits) 并推进 pos。

        ``ctx`` 是主模型的 ``(cos_tbl, sin_tbl, mask_idx)``（可选的第四项被忽略）；
        旋转会加上 :attr:`rope_delta`，调用方让它和主模型对应行保持一致。

        Args:
            h_prev: 主模型在当前位置的隐含状态。
            tok: 从这个 token id 开始草拟。
            ctx: 主模型的 ``(cos_tbl, sin_tbl, mask_idx[, rope])``；这里忽略 rope
                那一项，因为这条路径不处理任何图像分块。

        Returns:
            预测下一个 token 用的 ``(h, logits)``。调用方从 ``logits`` 里采样；
            这条路径上草稿器不提交 token。
        """
        cos_tbl, sin_tbl, mask_idx = ctx[0], ctx[1], ctx[2]
        e = self.embed.lookup(tok.reshape(1)).view(-1)
        x = self.fc(torch.cat([self.pre_norm_emb(e), self.pre_norm_hid(h_prev)], dim=-1))
        h = x + self.attn(self.in_ln(x), self.pos, cos_tbl, sin_tbl, mask_idx,
                          self.rope_delta)
        h = h + self.mlp(self.post_ln(h))
        h = self.norm(h)
        self.pos.add_(1)
        return h, self.embed(h)

    def prefill(self, H: torch.Tensor, next_toks: torch.Tensor, positions: torch.Tensor,
                ctx, rope=None) -> None:
        """在一段 prompt 位置上填充 MTP 的 KV 缓存。

        ``H[i]`` 是主模型在 ``positions[i]`` 的隐含状态，``next_toks[i]`` 是紧跟它
        的那个 token——MTP 头正是按这一对训练的，所以调用方传进来的是已经移过位的
        token（这里不再移位，否则分块调用方会移两次）。

        ``rope`` 是可选的显式 ``(cos, sin)`` 对，形状 ``[n, rot]``，用于含图像
        token 的 prompt 区域（``ops.mrope``）：草稿器是覆盖与主模型相同位置的全
        注意力层，必须同样地旋转。``None`` 走一维表路径；两种情况下 ``positions``
        都寻址 KV 槽位，``ctx`` 仍是主模型的元组。

        查询照主注意力那样切块、key 流式读取（``PREFILL_ATTN_TILE`` /
        ``PREFILL_ATTN_KB``）：不做查询切块的话 score 块是
        ``[nkv, rep, n, kend]``，16k token 时 1 GiB、64k 时 17 GB。

        Args:
            H: 主模型的隐含状态，``[n, hidden]``。
            next_toks: 跟在 ``H`` 每一行之后的 token id，长度 ``[n]``。
            positions: 那些 token 的绝对位置，长度 ``[n]``。
            ctx: 主模型的 ``(cos_tbl, sin_tbl, mask_idx[, rope])``。
            rope: 含图像区域用的显式三维 ``(cos, sin)``。

        Returns:
            无。缓存被填好，``self.pos`` 移到最后一个位置之后，下一份草稿从那继续。
        """
        n = min(int(positions.shape[0]), int(H.shape[0]), int(next_toks.shape[0]))
        if n <= 0:
            return
        # 调用方可能把 rope 对放在 ctx 里而不是参数里。
        if len(ctx) > 3:
            if rope is None:
                rope = ctx[3]
            ctx = ctx[:3]
        e = self.embed.lookup(next_toks[:n])
        # ``fc`` 必须走批处理路径：融合入口只吃单个 [K] 向量，用 [n, 2*hidden]
        # 调它只会静默读到第 0 行。
        x = self.fc.prefill(torch.cat([self.pre_norm_emb(e), self.pre_norm_hid(H[:n])],
                                      dim=-1))
        Q = PREFILL_ATTN_TILE
        for c0 in range(0, n, PREFILL_MTP_CHUNK):
            c1 = min(c0 + PREFILL_MTP_CHUNK, n)
            n_loc = c1 - c0
            base = int(positions[c0])
            # 切块坐标是绝对的：``(q0, q1, p0, kend)`` 都在分块坐标里，所以最后一个
            # 切块的因果上界就是分块末尾。
            tiles = ([(q0, min(q0 + Q, n_loc), base, base + min(q0 + Q, n_loc))
                      for q0 in range(0, n_loc, Q)]
                     if n_loc > Q else None)
            rope_c = None if rope is None else (rope[0][c0:c1], rope[1][c0:c1])
            x[c0:c1] = x[c0:c1] + self.attn.prefill(self.in_ln(x[c0:c1]),
                                                    positions[c0:c1], ctx, tiles,
                                                    rope=rope_c)
            x[c0:c1] = x[c0:c1] + self.mlp.prefill(self.post_ln(x[c0:c1]))
            # 最后那一步归一化不按分块施加：调用方经 ``_draft_impl`` 消费隐含状态，
            # 那里会在末尾统一归一化一次。在这里算要完整跑一遍 [n, hidden]，而且改变
            # 不了下一次重放读到的任何东西。
            _ = self.norm(x[c0:c1])
        self.pos.fill_(int(positions[n - 1]) + 1)

    def enable_spec_batch(self, capacity: int, buckets: tuple[int, ...],
                          ctx, block_size: int = 64,
                          max_len: int = 16384) -> None:
        """给草稿器配好逐行缓冲区、分页 KV 池和捕获图。

        草稿器的 KV 在主模型草稿所在的相同位置被改写、并由主校验回滚，所以它只
        需要同样的上下文窗口：每行 ``min(max_seq, max_len)`` 个位置（这个上限让池
        小到能与解码图共存；更长的上下文由 :meth:`check_len` 上报给调用方）。

        Args:
            capacity: 解码行数。
            buckets: 要为哪些批大小各捕获一张草稿图；每个一张，且每个都必须
                ``<= capacity``。
            ctx: 主模型的 ``(cos_tbl, sin_tbl, mask_idx)``。
            block_size: 分页 KV 的块大小；每行的块数 ``batch_len / block_size``
                向下取整，所以一行总是寻址整块。
            max_len: 每行草稿器位置数的上限。

        Returns:
            无。``(capacity, buckets)`` 不变时幂等；一旦变化就先丢掉旧池和旧图，
            因为捕获的图没法指向新池。
        """
        capacity = max(int(capacity), 1)
        if (self.pool is not None and self.batch_capacity == capacity
                and self.batch_buckets == tuple(buckets)):
            return
        if self.pool is not None:
            # 为不同的 bucket 重建（例如在同一个模型上跑第二个测试类）：先丢掉旧的
            # 草稿器状态，再分配新的。
            self.pool = None
            self.attn_b = None
            self._draft_graphs = {}
        self._ctx = ctx
        self.batch_capacity = capacity
        self.batch_buckets = tuple(buckets)
        hidden = self._cfg["hidden_size"]
        bs = int(block_size)
        bpr = max(1, min(int(max_len), int(self._max_seq)) // bs)
        self.batch_len = bpr * bs
        nkv, hd = self.attn.nkv, self.attn.hd
        self.pool = PagedKVCache(1, nkv, hd, num_blocks=capacity * bpr,
                                 block_size=bs, max_seq=self.batch_len,
                                 rows=capacity)
        # 静态块表：第 ``r`` 行拥有块 ``[r*bpr, (r+1)*bpr)``，所以草稿路径既不跑
        # 分配器，也不改块表。
        for row in range(capacity):
            self.pool.assign_table([row * bpr + i for i in range(bpr)], 0, row)
        self.attn_b = FullAttn(self._attn_sd, "mtp.layers.0.self_attn", self._cfg,
                               self._bench, self.batch_len, paged=True)
        self.attn_b.attach_pool(self.pool, 0)
        self.attn_b.enable_batching(capacity, 0)
        self.b_pos = torch.zeros(capacity, dtype=torch.long, device=_device)
        self.b_tok = torch.zeros(capacity, dtype=torch.long, device=_device)
        self.b_h = torch.zeros(capacity, hidden, dtype=D, device=_device)
        self.b_h_prev = torch.zeros(capacity, hidden, dtype=D, device=_device)
        self.b_logits = torch.zeros(capacity, self.embed.shape[0], dtype=D, device=_device)
        self._ctx_lens = torch.ones(capacity, dtype=torch.int32, device=_device)
        #: 批处理草稿器的逐行 mRoPE 偏移，由调用方（``Qwen35.spec_step_batch``）从
        #: 主模型的 ``rope_delta`` 镜像过来；纯文本 bucket 为 0，默认 bucket 因此
        #: 不变。
        self.b_rope_delta = torch.zeros(capacity, dtype=torch.long, device=_device)
        self.capture_draft_batch()

    def check_len(self, max_position: int) -> None:
        """某一行的上下文会超出草稿器池时抛出。

        Args:
            max_position: 该行最后一个位置，即上下文长度减一。

        Raises:
            RuntimeError: 位置等于或超过 ``batch_len``。池里没有槽位可写，调用方
                必须为这个请求关掉投机或者调高 ``max_len``；悄悄截断会对着一个与
                主模型不一致的 KV 草拟。
        """
        if self.pool is not None and max_position >= self.batch_len:
            raise RuntimeError(
                f"batch spec needs context < {self.batch_len} (got {max_position}); "
                "disable spec for this request or raise max_len")

    def _draft_batch_impl(self, rows: int) -> None:
        """为 ``rows`` 行做一步草稿：链的尾部喂给下一次重放。

        Args:
            rows: 活跃行数，即每个 ``b_*`` 缓冲区的前导切片。

        Returns:
            无。全部状态就地迁移，因为这段主体就是捕获的 bucket 图。
        """
        cos_tbl, sin_tbl, mask_idx = self._ctx[0], self._ctx[1], self._ctx[2]
        # 就地 +1：这个入口要的是长度而不是末下标；而且它跑在捕获图里，不能分配。
        self._ctx_lens[:rows].copy_(
            self.b_pos[:rows].to(torch.int32).add_(1))
        e = self.embed.lookup(self.b_tok[:rows])
        x = batch_proj(self.fc, torch.cat([self.pre_norm_emb(e),
                                           self.pre_norm_hid(self.b_h[:rows])], dim=-1))
        h = x + self.attn_b.batch(self.in_ln(x), self.b_pos[:rows],
                                  (cos_tbl, sin_tbl, mask_idx, self._ctx_lens,
                                   self.b_rope_delta))
        h = h + self.mlp.batch(self.post_ln(h))
        h = self.norm(h)
        self.b_h[:rows].copy_(h)
        self.b_logits[:rows].copy_(batch_proj(self.embed.gemv, h))
        self.b_pos[:rows].add_(1)

    def capture_draft_batch(self) -> None:
        """每个草稿器 bucket 一张图；:meth:`draft_batch` 把它的 K 次重放串起来。

        Returns:
            无。捕获前后都把位置计数器清零，每张 bucket 图因此都从位置 0 开始。

        Raises:
            AssertionError: 还没有池，也就是模型构建之后没调过
                ``enable_spec_batch``。
        """
        assert self.pool is not None, "enable_spec_batch() first"
        self._draft_graphs = {}
        for b in self.batch_buckets:
            impl = lambda b=b: self._draft_batch_impl(b)
            self.b_pos.zero_()
            self._draft_graphs[b] = runtime.capture(impl, warmup=2)
        self.b_pos.zero_()

    def draft_batch(self, bucket: int, K: int, out_toks: torch.Tensor) -> None:
        """为 ``bucket`` 的每一行串出 ``K`` 份草稿；写出 ``[bucket, K+1]``。

        调用方必须先把 ``b_h``/``b_tok``/``b_pos`` 备好；链式 argmax 和单流草稿
        一样，在重放之间留在设备上。

        Args:
            bucket: 要重放哪张图的批大小。它必须是已捕获的 bucket 之一：更大的会
                冲出缓冲区，更小的会留下没草拟的行。
            K: 每行要产出的草稿数。
            out_toks: long 缓冲区 ``[bucket, K+1]``；第 0 列是输入 token，第 ``i``
                列是第 ``i`` 份草稿。
        """
        g = self._draft_graphs[bucket]
        out_toks[:, 0].copy_(self.b_tok[:bucket])
        for i in range(1, K + 1):
            g.replay()
            self.b_tok[:bucket].copy_(self.b_logits[:bucket].argmax(-1))
            out_toks[:, i].copy_(self.b_tok[:bucket])

    @torch.no_grad()
    def prefill_row(self, row: int, H: torch.Tensor, next_toks: torch.Tensor,
                    positions: torch.Tensor, ctx, rope=None) -> None:
        """在一段 prompt 位置上填充某一行的分页草稿器 KV。

        数学和 :meth:`prefill` 相同，但注意力读写的是 ``row`` 的槽位表，而不是单流
        的稠密缓存。``rope`` 仍是那个可选的显式三维 ``(cos, sin)`` 对（本区域含
        图像 token）。

        Args:
            row: 其块表指向草稿器池的解码行。
            H: 主模型的隐含状态，``[n, hidden]``。
            next_toks: 跟在 ``H`` 每一行之后的 token id，长度 ``[n]``。
            positions: 那些 token 的绝对位置，长度 ``[n]``。
            ctx: 主模型的 ``(cos_tbl, sin_tbl, mask_idx[, rope])``。
            rope: 含图像区域用的显式三维 ``(cos, sin)``。

        Returns:
            无。会越过 ``batch_len`` 的那段被截到放得下的部分，该行后面的位置于是
            被丢掉，调用方经 :meth:`check_len` 回退，而不是写越池。

        Raises:
            AssertionError: ``enable_spec_batch`` 还没建出分页草稿器。
        """
        assert self.attn_b is not None, "enable_spec_batch() first"
        n = min(int(positions.shape[0]), int(H.shape[0]), int(next_toks.shape[0]))
        if n <= 0:
            return
        # 调用方可能把 rope 对放在 ctx 里而不是参数里。
        if len(ctx) > 3:
            if rope is None:
                rope = ctx[3]
            ctx = ctx[:3]
        # 草稿器池只寻址 [0, batch_len) 内的位置：比草稿窗口更长的 prompt 不许写
        # 越它。只预填充放得下的前缀，让 spec_step_batch 经 check_len() 上报溢出，
        # 调用方于是回退到普通解码步，而不是在注意力内核里崩掉。
        n = min(n, self.batch_len - int(positions[0]))
        if n <= 0:
            return
        e = self.embed.lookup(next_toks[:n])
        x = self.fc.prefill(torch.cat([self.pre_norm_emb(e),
                                       self.pre_norm_hid(H[:n])], dim=-1))
        Q = PREFILL_ATTN_TILE
        for c0 in range(0, n, PREFILL_MTP_CHUNK):
            c1 = min(c0 + PREFILL_MTP_CHUNK, n)
            n_loc = c1 - c0
            base = int(positions[c0])
            tiles = ([(q0, min(q0 + Q, n_loc), base, base + min(q0 + Q, n_loc))
                      for q0 in range(0, n_loc, Q)]
                     if n_loc > Q else None)
            rope_c = None if rope is None else (rope[0][c0:c1], rope[1][c0:c1])
            x[c0:c1] = x[c0:c1] + self.attn_b.prefill(
                self.in_ln(x[c0:c1]), positions[c0:c1], ctx, tiles, row=row,
                rope=rope_c)
            x[c0:c1] = x[c0:c1] + self.mlp.prefill(self.post_ln(x[c0:c1]))

    def close(self) -> None:
        """丢弃已捕获的草稿图；它们各自钉住私有的内存池。

        Returns:
            无。这些图存在期间占着的设备内存任何分配器都收不回，所以重载模型前
            必须先调它，否则下一个模型看到的空闲显存会比这个用掉的还少。
        """
        self._graph = None
        self._draft_graphs = {}
