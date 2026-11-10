"""Qwen3.5-4B 文本引擎：从权重到层，再到捕获图。

模型族的组装点：它持有层列表、rope 表、固定的图缓冲区和三条执行路径——解码（重放
一张图）、预填充（批处理 GEMM 加切块的因果注意力），以及校验（为投机解码捕获的 C
token 前向）。它拼装所用的原语住在 ``ops/``；逐层的块就在本模块旁边。

Module: ``coffin/models/qwen35``。族的顶层，也是 ``engine_api`` 注册的那个类；
它 import ``ops``、``backends`` 和每个同级模块，但不许 import ``engine_api``、
``model_runner`` 或族以上的任何东西。

Constraints:
    * 图只捕获一次、之后反复重放，所以捕获路径要写的每个缓冲区都必须在
      ``capture()`` / ``capture_batch()`` 之前以稳定地址存在。捕获之后再重新分配
      ``pos``、``token_in`` 或 ``rope_delta``，图会悄悄写进一个已经死掉的张量。
    * 捕获经 ``ops.runtime``，这正是本文件能在完全捕不了图的后端上工作的原因：
      ``caps.graph`` 为 false 时它返回一个句柄，其 ``replay()`` 重新执行主体，而
      每个缓冲区地址仍然是固定的，下面这些路径因此不需要分支。那不是降级模式——
      算子表混合了库内核和手写内核的后端从构造上就捕不了（它的库 GEMM 不可捕获，
      自己写的内核却可以），在那里即时执行就是正确的路径。
    * ``pos`` 寻址 KV 槽位和上下文窗口，而旋转读的是 ``pos + rope_delta``；
      把两者分开，才让图像 prompt 之后的解码仍然只是一次重放。
    * 线性注意力的循环状态必须和 KV 同行，绝不能单独走：前缀复用、快照/恢复和行
      搬移都必须同时带两者，否则模型会从一个从没见过那些共享 key 的状态继续。
    * 行分配：解码行是 ``0 .. capacity-2``，``capacity-1`` 是预填充暂存行。中间的
      空洞用哨兵行填上而不是压缩，状态因此永远不需要搬离自己那一行。
    * 不加锁。``llm_engine`` 单线程驱动所有改写，所以行的簿记就是普通的字典和列表。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch

from backends import device as _device
from models.qwen35.batch import (MULTI_Q_WIRED, batch_proj, multi_q_ready,
                                 wide_kernel_ready, wide_proj)
from models.qwen35.blocks import Layer
from models.qwen35.full_attn import FullAttn
from models.qwen35.linear_attn import LinearAttn
from models.qwen35.mtp import MTP
from models.qwen35.tuning import CKPT, PREFILL_ATTN_TILE
from models.qwen35.weights import load_weights
from ops import runtime
from ops.dtypes import D, WF
from ops.embed import TiedEmbedding
from ops.mrope import mrope_cos_sin, scatter_image_embeds
from ops.norm import RMSNorm
from ops.rope import make_inv_freq, make_rope_tables
from cache.paged_cache import PagedKVCache


def _drop_prefix(sd: dict, prefix: str) -> None:
    """删掉 ``sd`` 里 ``prefix`` 之下的每一项。

    在一层建好之后调用，构建的峰值占用因此是最终的模型，而不是模型加上读进来的那个
    字典。这不是微优化：27B 的 int8 checkpoint 约 26 GiB，卡是 31.7 GiB，而
    ``GEMVGroup`` 会把每对 ``gate``/``up`` 拷进一个融合矩阵，所以一直留着源张量的
    构建会冲过显存，在构造中途抛 ``OutOfMemoryError``。

    只有在调用方拥有该字典时才是安全的——调用方提供的字典必须活着，同一批张量才能
    重建模型，这也正是 ``engine_api.load_weights`` 被拆成独立调用的全部理由。

    Args:
        sd: 要就地裁剪的 state dict。
        prefix: 刚建好那层的参数名前缀。

    Returns:
        无。张量在最后一个引用消失时释放，也就是在拷贝它们的那个投影之后。
    """
    for k in [k for k in sd if k.startswith(prefix)]:
        del sd[k]


class Qwen35:
    """Qwen3.5-4B 模型：层、rope 表、图缓冲区和执行路径。"""

    def __init__(self, model_dir: str | Path, max_seq: int = 4096, bench: bool = True,
                 weights: dict | None = None, paged: bool = False,
                 quant: bool = False) -> None:
        """加载 checkpoint，并分配执行路径需要的一切。

        Args:
            model_dir: checkpoint 目录；它的 ``config.json`` 决定层数、层类型的
                交错方式和 rope 参数。
            max_seq: 引擎会跑的最长序列。它定 RoPE 表和掩码下标的尺寸，所以 ``prefill``
                越过表尾会被它挡住。
            bench: 到场时对投影做自动调优。``False`` 取默认内核配置，结构测试需要的
                正是不付扫描代价就能跑。
            weights: 预加载好的 state dict，用它而不是从磁盘读，改完代码重建模型
                因此不必重新加载。所有权仍在调用方；本实例不会清空它。
            paged: 不给全注意力层建私有稠密 KV 缓存，留给之后要调 :meth:`enable_paging`
                的调用方（PrefixCacheEngine 就是）。MTP 层不分页——它的缓存不参与
                前缀快照——所以它保留自己的。
            quant: 本次调用自己读 checkpoint 时，把文本投影存成 int8。给了
                ``weights`` 时忽略：那个字典已经定下了存储形式。

        Raises:
            ValueError: ``model_dir`` 里没有 ``config.json`` 或没有
                ``text_config``，由 JSON/配置访问处暴露出来。
            FileNotFoundError: 找不到期望前缀的 safetensors 分片。报出不匹配的
                checkpoint；引擎起不来。
        """
        model_dir = Path(model_dir)
        self.raw_cfg = json.loads((model_dir / "config.json").read_text())
        self.cfg = self.raw_cfg["text_config"]
        from models.qwen35.parallel import context, local_config

        self.parallel = context()
        if self.parallel is not None:
            self.cfg = local_config(self.cfg, self.parallel.world_size)
        #: 流水并行上下文；``None`` 表示这一层栈是完整的。
        #:
        #: TP 与 PP 的差别不是「切哪儿」而是「每 token 通信几次」：TP 每层两次 all-reduce
        #: （40 层 80 次），PP 每 token 三次点对点。这张卡的跨卡归约实测 1 ms 量级，
        #: 所以 MoE 这种 40 层的模型在 TP 下每 token 要付 80 ms 通信，而 PP 只付 0.2 ms。
        self.pipe = (self.parallel if self.parallel is not None
                     and self.parallel.mode == "pp" else None)
        total_layers = int(self.cfg["num_hidden_layers"])
        self.layer_begin, self.layer_end = (
            self.pipe.layer_range(total_layers) if self.pipe is not None
            else (0, total_layers))
        self.hidden = int(self.cfg["hidden_size"])
        self.vocab = int(self.cfg["vocab_size"])
        self.max_seq = max_seq
        #: 本实例是否拥有该 state dict、可以在消费张量的同时释放它们。自己加载时是；
        #: 调用方提供的字典保持完整，同一批张量才能在重载时重建模型。
        #:
        #: 所有权决定构建的峰值占用，这正是 27B checkpoint 能不能加载的前提：
        #: ``GEMVGroup`` 用一份融合拷贝替换两个源矩阵，不排空的话两种形式同时常驻，
        #: 峰值就是字典加上融合拷贝。排空让每个源在它那层建好时释放，峰值因此维持在
        #: 接近最终大小、而不是两倍（实测：26 GiB 的 int8 27B 峰值冲过 31.7 GiB，
        #: 在构建中途抛 OutOfMemoryError）。
        #:
        #: 融合拷贝也按层释放。``GEMVGroup`` 把每对投影汇成一个更宽的矩阵，这些临时量
        #: 即使两个源已经没了，也会记在缓存分配器的段上：一次 27B 构建达到 27.00 GiB
        #: 已分配加 4.57 GiB *已保留但未分配*，然后在一次 170 MiB 的 ``cat`` 上失败。
        #: 每层之后把段还回去，保留池就跟着活集合走，而不是跟着最大那一层的峰值走。
        own_sd = weights is None
        sd = weights if weights is not None else load_weights(model_dir, quant=quant)

        # 首段持有词嵌入（它把 token 变成 hidden），末段持有最后的归一化与 lm_head
        # （它把 hidden 变成 logits）；中间的段两者都不建，省下的显存正好装层。
        last_stage = self.pipe is not None and self.pipe.rank == self.pipe.world_size - 1
        first_stage = self.pipe is None or self.pipe.rank == 0
        self.embed = (TiedEmbedding(sd["model.language_model.embed_tokens.weight"], bench)
                     if first_stage else None)
        self.norm = RMSNorm(sd["model.language_model.norm.weight"])
        self.layers = []
        for i in range(self.layer_begin, self.layer_end):
            self.layers.append(Layer(sd, i, self.cfg, bench, max_seq, paged))
            # 建好的层已经持有它需要的一切；它在字典里剩下的条目都是 `GEMVGroup`
            # 已经拷过的，纯粹是峰值。
            if own_sd:
                _drop_prefix(sd, f"model.language_model.layers.{i}.")
                runtime.empty_cache()
        # 绑定还是不绑定是 checkpoint 的属性，不是架构的：4B 和 27B 都是
        # ``Qwen3_5ForConditionalGeneration``，只有 27B 单独带了 ``lm_head.weight``。
        # 按张量而不是按族名来判断，才让一条代码路径同时服务两者——反过来的做法不是
        # 崩溃，而是头悄悄改读嵌入，用错误的矩阵产生看似合理的 logits。
        head_w = sd.get("lm_head.weight")
        if head_w is not None:
            self.lm_head = TiedEmbedding(head_w, bench)
            self.head_is_tied = False
        elif self.pipe is not None:
            # 共享词嵌入的模型（``tie_word_embeddings=True``，如 Qwen3.5-0.8B）：输出头就是
            # 词嵌入本身。流水下每个 rank 都要有它 —— ``_forward`` 是「末段回传 4 KB
            # hidden、各段自己算 logits」的结构，``pipe_keeps`` 因此也把 embed_tokens 发给
            # 了每个 rank。这里直接另建一个 ``TiedEmbedding``，不依赖 ``self.embed``（非首段
            # 的 ``self.embed`` 是 ``None``）。
            tied_w = sd.get("model.language_model.embed_tokens.weight")
            if tied_w is None:
                raise ValueError(
                    "tied-head model under pipeline parallel needs embed_tokens on every "
                    "stage; pipe_keeps should have kept it")
            self.lm_head = TiedEmbedding(tied_w, bench)
            self.head_is_tied = True
        else:
            self.lm_head = self.embed
            self.head_is_tied = True
        self.mtp = MTP(sd, self.cfg, bench, max_seq) if "mtp.fc.weight" in sd else None
        #: 视觉张量留在常驻字典里，但只在第一次图像请求时才变成视觉塔（见
        #: ``enable_vision``）；纯文本服务端因此背着那 0.62 GiB，却从不构建该模块。
        self.vision_sd = {k: v for k, v in sd.items()
                          if k.startswith("model.visual.")}
        self.vision = None
        if weights is None:                 # 调用方持有的字典保持常驻
            sd.clear()
        runtime.empty_cache()

        rp = self.cfg["rope_parameters"]
        self.rot = int(self.cfg["head_dim"] * rp["partial_rotary_factor"])
        #: 唯一的频率轴：一维解码表和三维 mRoPE 表都从它推导，纯文本 prompt 在两条
        #: 路径上因此得到相同的 cos/sin。
        self.inv_freq = make_inv_freq(self.cfg["head_dim"], self.rot,
                                      rp["rope_theta"], _device)
        self.cos_tbl, self.sin_tbl = make_rope_tables(
            self.cfg["head_dim"], self.rot, rp["rope_theta"], max_seq, _device,
            inv_freq=self.inv_freq)

        # 捕获图寻址的固定缓冲区。用 ``enable_batching`` 时 ``pos``/``token_in``/
        # ``logits_out`` 会长出一个前导行维；默认容量 1 时它们精确保持单流形状。
        # 批处理打开时，``prefill_row``（最后一行）留给单流预填充 / 校验路径，
        # 新请求的预填充因此永远碰不到某个解码行（见 ``enable_batching``）。
        self.batch_capacity = 1
        self.decode_capacity = 1
        self.prefill_row = 0
        #: 批处理引擎里是 ``False``：投机解码是单流的，MTP 头的预填充在那里是白做的
        #: 工作，会被跳过。
        self.mtp_prefill = True
        self.pool = None
        self._batch_graphs: dict[int, runtime.Capture] = {}
        self._scratch_blocks: list[int] = []
        self._row_tables: dict[int, tuple] = {}
        self._sentinel_token = 0
        self.pos = torch.zeros(1, dtype=torch.long, device=_device)
        self.pos1 = self.pos[:1]
        self.mask_idx = torch.arange(max_seq, dtype=torch.long, device=_device)
        #: 正在解码的请求的逐行 mRoPE 偏移：token ``i`` 的 mRoPE 位置是
        #: ``i + rope_delta``。纯文本请求为 0（三根轴退化成 1 维，这就是恒等），
        #: 含图像的 prompt 之后是恒定的 ``MMInput.rope_delta``——图像块结束于
        #: ``max(h, w) / merge``，而只消耗了 ``h*w/merge**2`` 个 token，所以此后位置
        #: 会跑在 token 下标前面。把它保持成捕获图里的*缓冲区加法*，解码才能仍然只是
        #: 一次重放：只有 cos/sin 那一行动，每个 KV 槽位仍按 token 下标寻址。
        self.rope_delta = torch.zeros(1, dtype=torch.long, device=_device)
        self.rope_delta1 = self.rope_delta[:1]
        self.token_in = torch.zeros(1, dtype=torch.long, device=_device)
        self.logits_out = torch.zeros(self.vocab, dtype=D, device=_device)
        self.logits_single = self.logits_out
        self._graph = None

        # 批量投机解码（逐行草稿器加逐行校验）；全部由 ``enable_spec_batch`` 分配，
        # 默认关闭，单流投机路径和普通批处理路径因此不受影响。
        self.batch_spec = False
        self.spec_width = 0
        self.spec_buckets: tuple[int, ...] = ()
        self.verify_mode = "token"      # "once" = 对全部 B*c 个 token 一次前向
        self.sparse_topk = 0            # >0：选择性 KV（需要带掩码的多查询）
        self.verify_toks_b = None
        self.verify_pos_b = None
        self.verify_hidden_b = None
        self.verify_logits_b = None
        self.spec_toks_b = None
        self.spec_logits_b = None
        self.h_prev_b = None
        self._verify_batch_graphs: dict[int, runtime.Capture] = {}

        # 投机校验的状态；覆盖它的图由 ``capture_verify`` 按 ``verify_c`` 最终定下的
        # 宽度捕获。
        hidden = self.cfg["hidden_size"]
        self.h_prev = torch.zeros(hidden, dtype=D, device=_device)
        self.verify_c = 4
        self.verify_off = torch.arange(self.verify_c, dtype=torch.long, device=_device)
        self.verify_toks = torch.zeros(self.verify_c, dtype=torch.long, device=_device)
        self.verify_pos = torch.zeros(self.verify_c, dtype=torch.long, device=_device)
        self.verify_hidden = torch.zeros(self.verify_c, hidden, dtype=D, device=_device)
        self.verify_logits = torch.zeros(self.verify_c, self.vocab, dtype=D, device=_device)
        self._verify_graph = None

        # 上一次 ``spec_generate`` 的接受统计，由 ``spec_stats`` 读取。
        self.n_drafted = 0
        self.n_accepted = 0
        self.n_spec_steps = 0

    def enable_batching(self, capacity: int) -> None:
        """给每个逐 token 缓冲区分配一个 ``capacity`` 长的前导行维。

        阶段 1 让 ``B`` 个序列走一次前向：投影是批处理的（所有行共用一次权重读取，见
        ``models/qwen35/batch.py``），有状态的 mixer 仍在行切片上逐行调用单流内核。
        所以捕获图要写的每个缓冲区都必须在捕获之前、以稳定地址存在。

        Args:
            capacity: 要分配的行数，*包含*单流暂存行。解码行是
                ``0 .. capacity-2``；第 ``capacity-1`` 行是 ``prefill`` 写的那一行
                （``prefill_row``，也是 ``pos1`` 指向的行）。正是这个切分让连续批处理
                能在解码步之间预填充新请求，而不踩坏某个活着的解码行的循环状态。
                ``capacity=1`` 时单流缓冲区原样不动，默认路径因此不变。

        Raises:
            ValueError: ``capacity < 1``。引擎不能构建零行的批；在调用方那里拒绝。
            RuntimeError: 已经有图被捕获。要在 ``capture()``/``capture_batch()``
                之前调用，因为图会把这里分配的缓冲区地址烘进去。
        """
        capacity = int(capacity)
        if capacity < 1:
            raise ValueError(f"batch capacity must be >= 1 (got {capacity})")
        if self._graph is not None or self._verify_graph is not None:
            raise RuntimeError("enable_batching() must be called before capture()")
        self.batch_capacity = capacity
        if capacity == 1:
            self.decode_capacity = 1
            self.prefill_row = 0
            return
        # 一行是预填充/校验的暂存行，其余都是解码行。
        self.decode_capacity = capacity - 1
        self.prefill_row = capacity - 1
        self.mtp_prefill = False        # 投机解码只有单流
        self.pos = torch.zeros(capacity, dtype=torch.long, device=_device)
        self.pos1 = self.pos[self.prefill_row:self.prefill_row + 1]
        self.rope_delta = torch.zeros(capacity, dtype=torch.long, device=_device)
        self.rope_delta1 = self.rope_delta[self.prefill_row:self.prefill_row + 1]
        self.token_in = torch.zeros(capacity, dtype=torch.long, device=_device)
        self.logits_out = torch.zeros(capacity, self.vocab, dtype=D, device=_device)
        self.logits_single = self.logits_out[self.prefill_row]
        # 阶段 3 批处理注意力内核用的逐行注意力窗口长度（``pos+1``）；用 int32 是因为
        # 内核要在设备上读它。
        self.ctx_lens = torch.ones(capacity, dtype=torch.int32, device=_device)
        for layer in self.layers:
            layer.mix.enable_batching(capacity, self.prefill_row)
        self._row_tables.clear()
        if self.pool is not None:
            self._prepare_pool_rows()

    def _prepare_pool_rows(self) -> None:
        """扩出槽位表，并为每一行预留一个暂存块。

        捕获图会跑完它那个 bucket 的每一行，包括调用方没填的行（bucket 用哨兵行补
        齐）。在捕获*之前*给每行预留一个块、并把该行的表指向它是关键：捕获时的预热
        因此不会涂掉某个真实序列拥有的块——运行时服务给它的单个暂存块用的是同一招。
        预填充行也分一个，因为 ``capture()`` 会预热它的预填充路径。
        """
        pool = self.pool
        if pool is None or self.batch_capacity <= 1:
            return
        pool.set_rows(self.batch_capacity)
        if self._scratch_blocks:
            return
        for row in range(self.batch_capacity):
            blk = pool.alloc()
            self._scratch_blocks.append(blk)
            pool.assign(blk, 0, row)

    def enable_paging(self, num_blocks: int = 512, block_size: int = 64):
        """让全注意力层从共享的分页 int8 块池取 KV。

        必须在 :meth:`capture` 之前跑：捕获的解码步会把池张量（以及设备侧常驻的槽位
        表）烘进去，之后再切换只会留下一张写错地方的图。

        调用方契约：**在任何预填充或捕获之前先分配块表**。``pos2slot`` 初值全是零，
        所以在刚分页化的模型上跑 ``prefill()``/``capture()`` 会让每个位置都指向池的
        0 号槽位，连续几次预填充互相读到对方的 KV。症状看着像模型不确定，实际是调用
        方的错误；``PrefixCacheEngine`` 通过分配块并调用 ``assign_table`` 满足契约。

        Args:
            num_blocks: 池里的块数。池是所有活请求合起来的 KV 预算，所以池太小会表现
                成调度卡顿，而不是报错。
            block_size: 每块的 token 数；也是基数树共享前缀的粒度。

        Returns:
            池本身，调用方通过 :class:`~radix_cache.RadixCache` 驱动它。

        Raises:
            RuntimeError: 已经有图被捕获；池张量必须在捕获之前就位。
            ValueError: 请求的块数覆盖不了 ``max_seq`` 时由
                :class:`~paged_cache.PagedKVCache` 抛出；给池更多块。
        """
        if self._graph is not None:
            raise RuntimeError("enable_paging() must be called before capture()")
        full = [layer.mix for layer in self.layers if isinstance(layer.mix, FullAttn)]
        # 在分配池*之前*释放私有稠密缓存。256k 窗口下每份约 4.5 GB，两者同时持有
        # 再叠上 4 GB 的 int8 权重根本放不下——池一存在它们就死了，先丢掉它们把
        # 「启动 OOM」变成「完全没有额外峰值」。
        for mix in full:
            mix.kcache = mix.vcache = mix.kscale = mix.vscale = None
        runtime.empty_cache()
        pool = PagedKVCache(len(full), self.cfg["num_key_value_heads"], self.cfg["head_dim"],
                            num_blocks=num_blocks, block_size=block_size,
                            max_seq=self.max_seq, rows=max(self.batch_capacity, 1))
        for i, mix in enumerate(full):
            mix.attach_pool(pool, i)
        self.pool = pool
        runtime.empty_cache()
        if self.batch_capacity > 1:
            self._prepare_pool_rows()
        return pool

    def enable_vision(self, quant: bool = False, dtype=D) -> "VisionTower":
        """接上视觉塔，按需构建，纯文本服务因此不付任何代价。

        视觉塔不在解码图里：一张图在它的 token 被预填充之前、每个请求只编码一次，所以
        它可以是个普通的即时执行模块。它的权重和文本的一起放在常驻字典里，由
        :meth:`close` 释放。

        Args:
            quant: 把视觉塔的 GEMM 权重存成 int8 加逐行 scale。只有当放不下的是 fp16
                的视觉塔时才值得。
            dtype: 视觉塔激活和权重的计算 dtype。

        Returns:
            视觉塔，:meth:`image_embeds` 之后每个图像批调用它一次。

        Raises:
            RuntimeError: 模型不是按分页方式构建的，或者它的 checkpoint 没有
                ``vision_config``。只有服务路径支持视觉，所以宁可拒绝请求，也不要
                在没有任何嵌入的情况下预填充图像 token。
        """
        from models.qwen35.vision import VisionTower

        if self.pool is None:
            raise RuntimeError("enable_vision() needs the paged model (serve path)")
        cfg = self.raw_cfg.get("vision_config")
        if not cfg:
            raise RuntimeError("this checkpoint has no vision_config")
        if self.vision is None:
            self.vision = VisionTower(self.vision_sd, cfg, device=_device,
                                      quant=quant, dtype=dtype)
        return self.vision

    def image_embeds(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """把一批图像编码成 LLM token 嵌入 ``[Σtokens, hidden]``。

        结果就是 prompt 顺序，恰好是 prompt 里
        ``<|image_pad|>`` token 的顺序，调用方因此不用做任何重排就能把它拼进去：正是
        这个对齐让 ``scatter_image_embeds`` 只靠绝对区间就能工作。


        Args:
            pixel_values: processor 展平后的 patch 张量，
                ``[Σ t*h*w, C*tps*p*p]``.
            grid_thw: ``(t, h, w)`` patch 网格，每图一行，按 prompt
                顺序。

        Returns:
            ``[Σtokens, hidden]`` 的 bf16 嵌入，可以直接覆写预填充 gather 缓冲区的
            图像行。

        Raises:
            RuntimeError: 视觉塔从未接上。先调 :meth:`enable_vision`；纯文本服务端
                只有调用方出 bug 才会走到这里。
        """
        if self.vision is None:
            raise RuntimeError("vision tower not enabled (enable_vision())")
        pv = pixel_values.to(device=_device, dtype=WF)
        gt = grid_thw.to(device=_device, dtype=torch.long)
        with torch.no_grad():
            return self.vision(pv, gt).to(D)

    def snapshot_linear_states(self, row: int = 0) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        """克隆 ``row`` 那一行上每个线性注意力层的循环状态和卷积状态。

        前缀共享需要它：缓存的 KV 前缀只有连同产生它的循环状态一起才能复用，因为
        那 24 层是用一个定长的状态、而不是 KV 来携带前缀向前走的。


        Args:
            row: 要打快照的是哪一行的序列；预填充行或某个解码行。批处理调用方传它
                想要的那一行，而不是默认值。

        Returns:
            ``{层下标: (循环状态, 卷积状态)}``，只含线性注意力层。克隆是独立的，
            调用方可以让它们活过下一次预填充。

        """
        out: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for i, layer in enumerate(self.layers):
            mix = layer.mix
            if isinstance(mix, LinearAttn):
                out[i] = (mix.state[row].clone(), mix.conv_state[row].clone())
        return out

    def restore_linear_states(self, states, row: int = 0) -> None:
        """把 :meth:`snapshot_linear_states` 打下的快照装进 ``row``。

        Args:
            states: :meth:`snapshot_linear_states` 返回的那种字典；它的键必须是本模型
                的线性注意力层下标。
            row: 要装入状态的行，通常是该序列自己的解码行。

        Raises:
            KeyError: ``states`` 点出了本模型没有的层下标——也就是来自另一套架构的
                快照。宁可拒绝，也不要恢复一个残缺的状态。
        """
        for i, (st, conv) in states.items():
            mix = self.layers[i].mix
            mix.state[row].copy_(st)
            mix.conv_state[row].copy_(conv)

    def move_linear_states(self, src_row: int, dst_row: int) -> None:
        """把 ``src_row`` 那一行的循环状态拷进 ``dst_row``。

        批处理步骤两半之间的桥：新请求先在暂存行（``prefill_row``）上预填充，然后它的
        循环状态被搬进那个即将接着生成解码行。该行的 :attr:`rope_delta` 也跟着走——
        它属于序列而不属于行，丢了它会让图像 prompt 后续 token 按错误的角度旋转。

        Args:
            src_row: 持有待搬移状态的行，通常是 ``prefill_row``。
            dst_row: 目标解码行。两者相等时是空操作，也就是单流的情形。
        """
        if src_row == dst_row:
            return
        for layer in self.layers:
            mix = layer.mix
            if isinstance(mix, LinearAttn):
                mix.state[dst_row].copy_(mix.state[src_row])
                mix.conv_state[dst_row].copy_(mix.conv_state[src_row])
        self.rope_delta[dst_row].copy_(self.rope_delta[src_row])

    def zero_linear_states(self, row: int) -> None:
        """在新序列占用它之前，清空某一行的循环状态。

        Args:
            row: 要交给新请求的解码行。该行的 ``rope_delta`` 随状态一起清空：上一个
                占用者残留的偏移会把新请求转错。
        """
        for layer in self.layers:
            mix = layer.mix
            if isinstance(mix, LinearAttn):
                mix.state[row].zero_()
                mix.conv_state[row].zero_()
        self.rope_delta[row].zero_()

    def _forward(self) -> None:
        """单流路径（预填充/暂存行）上的一个 token。

        这就是 ``capture()`` 为单流引擎录下的图，也是批处理路径拿来对照的即时执行
        参考；它写暂存行，永远不写解码行。

        旋转读 ``pos + rope_delta``，而 KV 槽位仍停在 ``pos``（见 ``rope_delta``）；
        纯文本请求里这个缓冲区是 0，于是恰好就是原来的单一项查表。
        """
        pr = self.prefill_row
        if self.pipe is None or self.pipe.rank == 0:
            x = self.embed.lookup(self.token_in[pr:pr + 1]).view(-1)
        else:
            x = self.pipe.await_relay((1, self.hidden), self.pipe.rank - 1,
                                      _device, D).view(-1)
        ctx = (self.pos1, self.cos_tbl, self.sin_tbl, self.mask_idx,
               self.rope_delta1)
        for layer in self.layers:
            x = layer(x, ctx)
        self.pos1.add_(1)
        if self.pipe is None:
            self.logits_single.copy_(self.lm_head(self.norm(x)))
            return
        # 段间只传激活：中间段把 hidden 交给下一段，末段把 logits 送回 rank 0。
        # 末段必须在这里早退，否则它会去等一个永远不会来的回包。
        # 段间只传 hidden（[hidden] = 4 KB）。末段算完自己那层后把 hidden 回给其他各段，
        # 各段用自己那份输出头算 logits —— 每个 rank 因此拿到**逐位相同**的 logits，
        # 引擎的采样决定自然一致，而线上跑的是 4 KB 而不是 1 MB。
        # 段间只传 4 KB 的 hidden；末段把 hidden 回给各段，各段用自己的头算 logits。
        # 这样每段都能在等对端的时候把自己的头 GEMV 藏在等待里，实测比「末段算、
        # 广播 logits」快 22%（26.24 vs 21.48 tok/s）。
        nxt = self.pipe.rank + 1
        if nxt >= self.pipe.world_size:
            for dst in range(self.pipe.world_size - 1):
                self.pipe.relay(x.reshape(1, -1), dst)
            h = x
        else:
            self.pipe.relay(x.reshape(1, -1), nxt)
            h = self.pipe.await_relay((self.hidden,), self.pipe.world_size - 1,
                                      _device, D)
        self.logits_single.copy_(self.lm_head(self.norm(h)))

    def _forward_batch(self, rows: int) -> None:
        """在一次前向里为 ``rows`` 个序列各跑一个 token。

        投影走 M=rows 的批处理 GEMV 路径（捕获图里行数是静态的）；注意力和 delta-net
        递推仍逐行调用现有的单 token 内核，作用在各行缓冲区的行切片上，所以每一行的
        数字与单条解码精确一致。``positions`` 仍是裸的 token 下标（它寻址 KV 槽位和
        上下文窗口）；``ctx`` 的第 5 项携带逐行 ``rope_delta``，只让旋转发生偏移
        （图像之后的 mRoPE）。

        Args:
            rows: 这张图跑多少行，也就是 bucket 大小。它必须等于捕获时的宽度；它碰
                的缓冲区是捕获的固定件，不是参数。
        """
        if self.pipe is None or self.pipe.rank == 0:
            x = self.embed.lookup(self.token_in[:rows])       # [rows, hidden]
        else:
            x = self.pipe.await_relay((rows, self.hidden), self.pipe.rank - 1,
                                      _device, D)
        positions = self.pos[:rows]
        ctx = (self.cos_tbl, self.sin_tbl, self.mask_idx, self.ctx_lens,
               self.rope_delta)
        for layer in self.layers:
            x = layer.batch(x, positions, ctx)
        if self.pipe is not None:
            # 与 ``_verify_impl`` 同一套走法：末段把 hidden 回给各段、各段自己算 logits。
            # 让末段算完再广播 logits 实测慢 22%（文档 §8.8），因为各段算 logits 的
            # 那点开销本来就躲在流水等待窗口里。
            nxt = self.pipe.rank + 1
            if nxt >= self.pipe.world_size:
                for dst in range(self.pipe.world_size - 1):
                    self.pipe.relay(x, dst)
            else:
                self.pipe.relay(x, nxt)
                x = self.pipe.await_relay((rows, self.hidden), self.pipe.world_size - 1,
                                          _device, D)
        x = self.norm(x)
        # 输出头用 ``self.lm_head`` 而不是 ``self.embed.gemv``：后者只在首段存在，
        # 而这行在每个 rank 上都会执行（occamy 的输出头是独立的，不与嵌入共享）。
        self.logits_out[:rows].copy_(self.lm_head.prefill(x))
        self.pos[:rows].add_(1)

    def reset(self) -> None:
        """把逐行位置和循环状态全部清零。

        在请求之间、以及每次捕获预热之前调用，图因此是在已知状态下录的，而不是对着
        上一个请求留下的残余。
        """
        self.pos.zero_()
        self.rope_delta.zero_()
        for layer in self.layers:
            layer.reset()
        if self.mtp is not None:
            self.mtp.reset()
        if self.batch_spec:
            self.h_prev_b.zero_()
            self.mtp.b_pos.zero_()

    def logits(self, token: int) -> torch.Tensor:
        """让一个 token 走一遍未捕获的前向，返回它的 logits。

        捕获解码路径的即时执行参考，``prefill`` 也用它处理短到不值得批处理的 prompt。


        Args:
            token: 要喂的下一个 token id；它被放到预填充行上，该行的位置加一。


        Returns:
            该位置的 ``[vocab]`` bf16 logits。这是活缓冲区而不是拷贝——下一次调用会
            覆盖它。
        """
        pr = self.prefill_row
        self.token_in[pr:pr + 1].fill_(token)
        self._forward()
        return self.logits_single

    @torch.no_grad()
    def prefill(self, tokens: list[int], chunk: int = 2048,
                spec_row: int | None = None,
                mm: "MMInput | None" = None,
                need_logits: bool = True) -> torch.Tensor | None:
        """用批处理 GEMM 处理整个 prompt；返回最后一个 token 的 logits。

        ``chunk`` 就是 GEMM 的 M，它很值钱：fp16 上的库 GEMM 在 M=256 时约
        13 TFLOP/s、M>=1024 时约 20；更大的分块还意味着每 token 更少的、只有 4-3 个
        token 大小的逐元素算子。n=2048 实测：256 时 865 tok/s，512 时 1035，2048 时
        1094。它只多花激活内存（注意力 score 是 [nkv, rep, C, p1] 的 fp32），这也是
        它不用 ``max_seq`` 的原因。

        Args:
            tokens: 整个 prompt，按顺序。位置从当前的
                ``pos1`` 开始，分块预填充因此接着上一次调用停下的地方继续。
            chunk: 每次 GEMM 处理的 token 数；见上面的实测数据。
            spec_row: 哪个解码行的分页草稿器 KV 接收这个 prompt 的隐含状态。只有打开
                批量投机解码时才用；``None`` 表示不碰 MTP 头。
            mm: 该请求的图像载荷，类型 :class:`mm_input.MMInput`，纯文本时为 ``None``。
                它恰好改变两件事，都是逐分块的，且都用*绝对* prompt 坐标
                （``tokens`` 是整个 prompt，所以分块 ``[c0, c1)`` 就是 prompt 的
                ``[pos0+c0, pos0+c1)``）：gather 缓冲区的图像行被视觉塔的嵌入覆写
                （``mm.spans`` 允许重叠，所以从图像中间开始的分块只拷自己那几行——
                不假设分块边界），以及全注意力层改用这些 token 的显式三维 cos/sin
                （对 ``mm.slice_positions`` 调 ``ops.mrope.mrope_cos_sin``），
                而不是一维表。

                它刻意*不*改的东西：因果掩码、KV 槽位和上下文窗口仍用 token 下标，
                解码之后因此只需要那个标量 ``rope_delta``。``mm=None`` 就是旧路径，
                逐位相同。

        Returns:
            prompt 最后一个 token 的 ``[vocab]`` logits，第一个生成的 token 从这里
            采样。短于 4 个 token 的 prompt 这里是解码路径的最后一次输出，而不是批处理
            的结果。

        Raises:
            ValueError: ``tokens`` 为空，或者放不进当前位置与 ``max_seq`` 之间的
                区间。在调度器那里拒绝该请求，那是唯一能缩短或拒绝它的层。

        """
        n = len(tokens)
        pos0 = int(self.pos1)
        if n == 0:
            raise ValueError("empty prompt")
        if n > self.max_seq - pos0:
            raise ValueError(f"prompt of {n} exceeds max_seq={self.max_seq}")
        if n < 4:
            # 极短的 prompt 受内存限制：M<4 时库 GEMM 比专用的解码入口还慢，所以让
            # 它们走解码路径。
            # ``rope_delta`` 是解码图的输入，必须在第一次 ``logits`` 调用之前就位。

            if mm is not None:
                self.rope_delta[self.prefill_row] = int(mm.rope_delta)
            lg = None
            for t in tokens:
                lg = self.logits(t)
            return lg

        if self.pipe is not None:
            # 分块预填充：每段只跑自己那 10 层，块沿管线逐段传递。这是参考实现
            # （enginex-vllm-bi100-qwen36）的核心配置 ``--max-num-batched-tokens 4096
            # --enable-chunked-prefill`` 在层切分下的等价物——逐 token 驱动时 M=1，
            # 张量核完全用不上，5 个 token 要跑 5 个完整的管线来回。
            if mm is not None:
                raise RuntimeError(
                    "pipeline parallel does not support image inputs: the vision tower "
                    "lives on stage 0 and its embeddings would have to cross the wire")
            # 分块路径目前只对**单块**（n <= 512）验证过：空多块会让段间协议失步
            # （实测 2001 token 时 rank0 在等末段的回包超时，而同一 prompt 走逐 token
            # 是通的，代价是 prefill 89.4 s）。所以按块数分流：单块走分块，多块回退，
            # 等把多块的失步修掉再放开。
            # 三档：``auto``（默认）按长度选路、``1`` 强制分块、``0`` 一律逐 token。
            #
            # 阈值 64 是实测的：23 token 时逐 token 更快（935 vs 1220 ms，M 太小，库 GEMM
            # 还不如解码入口），2001 token 时分块快 8.4 倍（10.8 s vs 90.8 s）。中间的
            # 24..512 没有逐点测过，取 64 是保守的一侧 —— 宁可早一点切到分块。
            mode = os.environ.get("COFFIN_PP_CHUNKED", "auto")
            if not (mode == "1" or (mode == "auto" and n >= 64)):
                # 逐 token 的兜底：短 prompt 上它反而更快（M 太小，库 GEMM 不如解码入口），
                # 多块长 prompt 上它是唯一正确的路。
                lg = None
                for t in tokens:
                    lg = self.logits(int(t))
                return lg
            toks = torch.tensor(tokens, dtype=torch.long, device=_device)
            ctx = (self.cos_tbl, self.sin_tbl, self.mask_idx)
            # 块宽按段间载荷定：C 行 hidden 是 C*H*4 字节，C=512 时 4 MB，接收缓冲
            # （32 MB）装得下整批分片，不用等对端边收边放。
            step = max(1, min(int(chunk), 512))
            comm_before = self.pipe.comm_calls
            last_row = None
            for c0 in range(0, n, step):
                c1 = min(c0 + step, n)
                if self.pipe.rank == 0:
                    x = self.embed.lookup(toks[c0:c1])
                else:
                    x = self.pipe.await_relay((c1 - c0, self.hidden),
                                              self.pipe.rank - 1, _device, D)
                positions = torch.arange(pos0 + c0, pos0 + c1, device=_device)
                _tl = time.perf_counter()
                for layer in self.layers:
                    x = layer.prefill(x, positions, ctx, None)
                _tlayers = (time.perf_counter() - _tl) * 1e3
                nxt = self.pipe.rank + 1
                if nxt < self.pipe.world_size:
                    self.pipe.relay(x, nxt)
                if os.environ.get("COFFIN_PREFILL_TRACE") == "1":
                    print(f"[pp-prefill] rank={self.pipe.rank} n={n} c=[{c0},{c1}) "
                          f"layers={_tlayers:.1f}ms comm={self.pipe.comm_calls}",
                          flush=True)
                last_row = x[-1]
            if os.environ.get("COFFIN_PP_DEBUG"):
                # 多块失步时唯一能定位的东西就是「每个 rank 各发/收了几次」。期望值：
                # rank0 = 块数 + 1，中间段 = 2 * 块数 + 1，末段 = 块数 + (段数 - 1)。
                print(f"[pipe] rank={self.pipe.rank} n={n} step={step} "
                      f"chunks={(n + step - 1) // step} "
                      f"comm={self.pipe.comm_calls - comm_before}", flush=True)
            self.pos1.fill_(pos0 + n)
            # 分块预填充的中间块不需要 logits：尾部的段间回传 + 每个 rank 各算一次
            # ``lm_head``（2048 -> 248320 的 GEMV，权重 254 MB）在 2001 token 的 4 块
            # 上实测要 3.5 s，而只有最后一块的结果会被用到。四 rank 的判定依据
            # （``need_logits``）由调用方按同一个 ``stop == total`` 传下来，所以跳过时
            # 各 rank 的 relay 圈数仍然一致。
            if not need_logits:
                return None
            if self.pipe.rank == self.pipe.world_size - 1:
                for dst in range(self.pipe.world_size - 1):
                    self.pipe.relay(last_row.reshape(1, -1), dst)
            else:
                last_row = self.pipe.await_relay((self.hidden,),
                                                 self.pipe.world_size - 1,
                                                 _device, D)
            return self.lm_head(self.norm(last_row))

        toks = torch.tensor(tokens, dtype=torch.long, device=_device)
        ctx = (self.cos_tbl, self.sin_tbl, self.mask_idx)
        Q = PREFILL_ATTN_TILE
        last = None
        for c0 in range(0, n, chunk):
            c1 = min(c0 + chunk, n)
            if self.pipe is None or self.pipe.rank == 0:
                x = self.embed.lookup(toks[c0:c1])
            else:
                x = self.pipe.await_relay((c1 - c0, self.hidden), self.pipe.rank - 1,
                                          _device, D)
            positions = torch.arange(pos0 + c0, pos0 + c1, device=_device)
            # 因果注意力的查询切块：每块只读到自己的最后一个位置，score 块里被掩掉的那
            # 一半因此根本不会被构造。切块坐标在主机侧算成整数，每层不会多一次设备同步。
            # key 范围和查询范围都是*绝对*位置，缓存切片和 mask_idx 就是按它索引的。
            n_loc, base = c1 - c0, pos0 + c0
            tiles = ([(q0, min(q0 + Q, n_loc), base, base + min(q0 + Q, n_loc))
                      for q0 in range(0, n_loc, Q)]
                     if n_loc > Q else None)
            layer_ctx = ctx
            rope = None
            if mm is not None:
                x = scatter_image_embeds(x, mm.embeds, mm.spans, pos0 + c0, pos0 + c1)
                a, b = pos0 + c0, pos0 + c1
                rope = mrope_cos_sin(mm.slice_positions(a, b), self.inv_freq)
                layer_ctx = ctx + (rope,)
            for layer in self.layers:
                x = layer.prefill(x, positions, layer_ctx, tiles)

            if self.mtp is not None and self.mtp_prefill:
                # 把 MTP 头按分块喂，而不是把所有分块的隐含状态都攒着最后拼接：256k 的
                # prompt 其 `H` 是 1.34 GB bf16，拼接拷贝还要再翻一倍。最后一个 prompt
                # token 没有下一个 token，所以每个分块都在末尾少喂一个。MTP 自己的
                # 注意力是覆盖相同位置的全注意力层，所以图像分块也把自己的 rope 切片交给
                # 它（覆盖 ``m1 - c0`` 个 token，是该分块的严格前缀）。


                m1 = min(c1, n - 1)
                if m1 > c0:
                    self.mtp.prefill(x[:m1 - c0], toks[c0 + 1:m1 + 1],
                                     torch.arange(pos0 + c0, pos0 + m1, device=_device),
                                     layer_ctx,
                                     rope=None if rope is None
                                     else (rope[0][:m1 - c0], rope[1][:m1 - c0]))
            elif self.batch_spec and self.mtp is not None and spec_row is not None:
                # 批量投机：同样的分块，写进该行的分页草稿器 KV。
                m1 = min(c1, n - 1)
                if m1 > c0:
                    self.mtp.prefill_row(
                        spec_row, x[:m1 - c0], toks[c0 + 1:m1 + 1],
                        torch.arange(pos0 + c0, pos0 + m1, device=_device), layer_ctx,
                        rope=None if rope is None
                        else (rope[0][:m1 - c0], rope[1][:m1 - c0]))
            last = x[-1]
        self.pos1.fill_(pos0 + n)
        # 此后这个请求的每个解码步都按 ``token_index + rope_delta`` 旋转；纯文本
        # prompt（或 ``mm=None``）写入 0，那正是捕获图构建时的恒等值。

        self.rope_delta[self.prefill_row] = int(mm.rope_delta) if mm is not None else 0

        if self.mtp is not None and self.mtp_prefill and n >= 2:
            self.h_prev.copy_(last)
        elif self.batch_spec and spec_row is not None:
            self.h_prev_b[spec_row].copy_(last)

        if self.pipe is not None:
            # 流水模式在函数开头就按 token 早退了；还能走到这里说明分块路径和流水被同时
            # 打开，而那条路的段间分块边界没有核对过，宁可响亮地失败。
            raise RuntimeError("pipeline prefill must take the per-token path")
        return self.lm_head(self.norm(last))

    @torch.no_grad()
    def prefill_group(self, seqs: list, snapshot_interval: int | None = None):
        """用一次前向预填充多个 prompt。

        只共享投影/MLP 阶段：每一层里所有序列的隐含状态先拼接，投影对 ``C_total`` 个
        token 只跑一次（更大的 GEMM，而且每层一次权重反量化而不是每序列一次），结果按
        序列切开，在该序列自己的位置和行上做有状态的注意力 / delta 扫描，然后输出投影
        和 MLP 再批处理跑一次。预填充是即时执行的（没有捕获图），所以动态的
        ``C_total`` 不花代价。

        设了 ``snapshot_interval`` 时，每个序列的后缀按块对齐的 ``snapshot_interval``
        分块遍历（token 优先：每一块都要跑完*所有*层，下一块才开始），循环状态于是
        能在块边界打快照，和单序列前缀路径完全一样。不设时整个后缀是一块，这也让
        ``k == 1`` 与 ``prefill`` 逐位相同。



        换来的是*一批* prompt 的延迟而不是它们之和：权重每层只读一次。它和逐个跑
        prompt 并非逐位一致——不同的 GEMM M 会改变 fp32 累加次序——所以契约是「argmax
        相同，logits 在几个 bf16 ULP 之内」（单序列的 ``prefill`` 路径不变，仍然是精确的）。

        Args:
            seqs: 鸭子类型的序列对象，每个需要 ``token_ids``（整个
                prompt）、``matched``（基数树已经持有的前导 prompt token 数，默认 0）、
                ``row``（它终身占用的解码行）和 ``block_table``（它的 KV 块，已分配——
                命中的块与树共享，后缀块是新的）。有 ``mm`` 属性时会读它。



                把命中边界的循环状态恢复进 ``row`` 是调用方的责任；本方法只在
                ``matched == 0`` 时清空该行。预填充行上不做暂存：KV 和循环状态直接
                落在各序列自己的解码行里，之后不需要搬移任何状态。


            snapshot_interval: 每这么多个 token 记一次循环状态快照，向上取整到整块。
                ``None`` 表示不记，调用方不打算把前缀插进基数树时这是对的选择。



        Returns:
            ``(logits_list, states_at_list)``：每个序列一个下一 token 的 logits 张量；
            以及每个序列在预填充过程中于块边界打下的快照
            （``{absolute_position: snapshot_linear_states(row)}``），供基数树插入。
            ``snapshot_interval`` 为 ``None`` 时 ``states_at_list[i]`` 为空。


        Raises:
            RuntimeError: 模型没有分页池。组预填充直接写进逐行的块表，所以先调
                :meth:`enable_paging`。

            ValueError: 某个 prompt 为空或超过 ``max_seq``、``matched`` 落在 prompt 之外、
                两个序列抢占同一个解码行，或某行超出 ``0 .. decode_capacity-1``。在调度
                器那里拒绝这一批；这些是请求形状的 bug，否则会弄坏另一个序列的行。


        """
        if self.pool is None:
            raise RuntimeError("prefill_group needs a paged pool (enable_paging())")
        n = len(seqs)
        if n == 0:
            return [], []
        lens = [len(s.token_ids) for s in seqs]
        if min(lens) < 1:
            raise ValueError("empty prompt")
        if max(lens) > self.max_seq:
            raise ValueError(f"prompt of {max(lens)} exceeds max_seq={self.max_seq}")
        matched = [int(getattr(s, "matched", 0) or 0) for s in seqs]
        for length, m in zip(lens, matched):
            if not 0 <= m < length:
                raise ValueError(f"matched={m} outside 0..{length - 1}")
        rows = [int(s.row) for s in seqs]
        if len(set(rows)) != n:
            raise ValueError(f"duplicate decode rows: {rows}")
        if min(rows) < 0 or max(rows) >= self.decode_capacity:
            raise ValueError(f"row outside 0..{self.decode_capacity - 1}: {rows}")

        ctx = (self.cos_tbl, self.sin_tbl, self.mask_idx)
        bs = self.pool.bs
        suffix = [length - m for length, m in zip(lens, matched)]
        # 逐序列的图像载荷，从序列对象上读，这样多模态请求在这里不需要额外参数
        # （runner 已经把它挂在 ``Sequence.mm`` 上）。全为 ``None`` 时——也就是纯文本
        # 引擎——走的正是原来的代码路径：没有 rope 对、不覆写嵌入、delta 为 0。

        mm_list = [getattr(s, "mm", None) for s in seqs]
        for s, row, m, mm_i in zip(seqs, rows, matched, mm_list):
            self.pool.assign_table(list(s.block_table), 0, row)
            if m == 0:
                self.zero_linear_states(row)      # 新序列
                # matched > 0：调用方已把边界状态恢复进这一行，rope 偏移也一起恢复了——
                # 在这里重写会是同一个值，但复用一行给*文本*请求时必须回到 0，所以
                # 无条件写一次。

            self.rope_delta[row] = int(mm_i.rope_delta) if mm_i is not None else 0

        # token 优先的调度。有前缀复用时，后缀按块对齐的快照边界切开：块对齐的头部
        # （树能复用的那段）按 ``snapshot_interval`` 分块遍历、每块打一份快照；
        # 不足一块的尾巴放到最后，只有它的 logits 有用。

        states_at: list[dict[int, dict]] = [{} for _ in range(n)]
        plans: list[list[tuple[int, int, int]]] = []
        # 每个序列可打快照头部的最后一个 token 下标（``matched`` 之上的整数块）；
        # 其余是尾巴。
        head = [max(0, (length // bs) * bs - m) for length, m in zip(lens, matched)]
        if snapshot_interval is None:
            plans.append([(i, 0, suffix[i]) for i in range(n)])
        else:
            step = max(bs, int(snapshot_interval) // bs * bs)
            n_head = max((h + step - 1) // step for h in head)
            for j in range(n_head):
                rng = []
                for i in range(n):
                    c0 = min(j * step, head[i])
                    c1 = min(c0 + step, head[i])
                    if c1 > c0:
                        rng.append((i, c0, c1))
                if rng:
                    plans.append(rng)
            tail = [(i, head[i], suffix[i]) for i in range(n) if head[i] < suffix[i]]
            if tail:
                plans.append(tail)

        last_hidden: list[torch.Tensor | None] = [None] * n
        Q = PREFILL_ATTN_TILE
        # 流水线：只有首段做嵌入，其余段从上一段收 hidden；每个分块跑完本段的层之后
        # 立刻把 hidden 转发给下一段。这与单序列 ``prefill`` 的收尾是同一套约定，
        # 也是分组预填充能在 PP 下跑起来的前提 —— 非首段没有 ``embed``，无条件调
        # ``self.embed.lookup`` 会抛 ``NoneType has no attribute lookup``。
        pipe = self.pipe
        stage = 0 if pipe is None else pipe.rank
        world = 1 if pipe is None else pipe.world_size
        first = stage == 0
        for chunk in plans:
            parts: list[torch.Tensor] = []
            positions: list[torch.Tensor] = []
            ropes: list[tuple | None] = []
            tiles: list[list | None] = []
            starts = [0]
            for i, c0, c1 in chunk:
                length = c1 - c0
                base = matched[i] + c0
                mm_i = mm_list[i]
                if first:
                    ids = list(seqs[i].token_ids[base:base + length])
                    emb = self.embed.lookup(torch.as_tensor(
                        ids, dtype=torch.long, device=_device))
                    if mm_i is not None:
                        emb = scatter_image_embeds(emb, mm_i.embeds, mm_i.spans,
                                                   base, base + length)
                        ropes.append(mrope_cos_sin(
                            mm_i.slice_positions(base, base + length), self.inv_freq))
                    else:
                        ropes.append(None)
                    parts.append(emb)
                else:
                    ropes.append(None)
                positions.append(torch.arange(base, base + length,
                                              dtype=torch.long, device=_device))
                # 绝对 key 范围：一个切块只读到自己的末尾。
                tiles.append(
                    [(q0, min(q0 + Q, length), base, base + min(q0 + Q, length))
                     for q0 in range(0, length, Q)]
                    if length > Q else None)
                starts.append(starts[-1] + length)
            total = starts[-1]
            if first:
                x = torch.cat(parts, dim=0)
            else:
                x = pipe.await_relay((total, self.hidden), stage - 1, _device, D)

            for layer in self.layers:
                mix, proj = layer.prefill_group_project(x)
                pieces = []
                for k, (i, _c0, _c1) in enumerate(chunk):
                    sl = slice(starts[k], starts[k + 1])
                    proj_k = tuple(t[sl] for t in proj)
                    # rope 对搭在 ``ctx`` 上（``FullAttn.batch`` 对 ``ctx_lens`` 用的也是
                    # 这种透传方式）：层辅助函数原样转发 ``ctx``，所以带图像的序列拿到自己
                    # 的三维旋转，纯文本序列拿到一维表。

                    ctx_k = ctx if ropes[k] is None else ctx + (ropes[k],)
                    pieces.append(layer.prefill_group_state(
                        mix, proj_k, positions[k], ctx_k, tiles[k], rows[i]))
                out = layer.prefill_group_out(mix, torch.cat(pieces, dim=0))
                x = x + out
                x = x + layer.mlp.prefill(layer.post_ln(x))

            if stage + 1 < world:
                pipe.relay(x, stage + 1)
            for k, (i, c0, c1) in enumerate(chunk):
                if c1 == suffix[i]:
                    last_hidden[i] = x[starts[k + 1] - 1].clone()
                # 批量投机：把这个分块的主模型隐含状态喂给该行的草稿器 KV（位置 p 处的
                # H，下一个 token p+1）——与单序列预填充交给 ``MTP.prefill`` 的同一对。

                if self.batch_spec and self.mtp is not None:
                    base = matched[i] + c0
                    length = c1 - c0
                    if length > 1:
                        h = x[starts[k]:starts[k + 1] - 1]
                        nxt = torch.as_tensor(
                            list(seqs[i].token_ids[base + 1:base + length]),
                            dtype=torch.long, device=_device)
                        pos = torch.arange(base, base + length - 1,
                                           dtype=torch.long, device=_device)
                        self.mtp.prefill_row(rows[i], h, nxt, pos, ctx)
                    if c1 == suffix[i]:
                        self.h_prev_b[rows[i]].copy_(x[starts[k + 1] - 1])
                # 可复用的快照只存在于块对齐头部内部的块边界上（和单序列路径同一规则）。

                if snapshot_interval is not None and c1 % bs == 0 and c1 <= head[i]:
                    states_at[i][matched[i] + c1] = \
                        self.snapshot_linear_states(row=rows[i])

        # 所有序列共用一次 lm_head 读取：[n, hidden] -> [n, vocab]。批处理 GEMV
        # 路径处理 n 为 2/4/8 的情形；其他计数退回逐行单个 GEMV（这里重要的是日常
        # 的 bucket 大小）。
        if pipe is None:
            # 单卡保留原来的批处理 lm_head 读取：[n, hidden] -> [n, vocab]。
            last = torch.stack([self.norm(h) for h in last_hidden])
            if n in (2, 4, 8):
                logits = self.embed.prefill(last)
                return [logits[i] for i in range(n)], states_at
            return [self.lm_head(last[i]) for i in range(n)], states_at
        # 流水线：末段算完把每条序列的最后一个 hidden 回传给其余各段，各段用自己那份
        # ``lm_head`` 算 logits —— 与单序列 ``prefill`` 的收尾同一套约定。这样每段
        # 返回的 logits 一致，非首段也不需要 ``embed``（它在这些段上是 None）。
        last = torch.stack([h for h in last_hidden])
        if stage == world - 1:
            for dst in range(world - 1):
                pipe.relay(last, dst)
        else:
            last = pipe.await_relay((n, self.hidden), world - 1, _device, D)
        return [self.lm_head(self.norm(last[i])) for i in range(n)], states_at

    def capture(self, warmup: int = 3) -> None:
        """把 ``_forward`` 捕获进设备图，让解码步只需一次 launch。

        Args:
            warmup: 捕获前的即时执行轮数，用来安置好分配器和每个惰性内核句柄。
                不预热就捕获，录进图里的是初始化工作而不是稳态。

        Raises:
            RuntimeError: 流正忙或已有图在活动时由运行时的图机制抛出；捕获必须在
                引擎开始服务之前跑完。只有 ``caps.graph`` 为真的后端才会走到。
        """
        self.reset()
        self.token_in.zero_()

        def _reset() -> None:
            """在预热和捕获之间把模型倒回去。

            预热会推进 ``pos`` 和循环状态；图必须按它第一次重放会看到的状态来录，
            所以两个计数器在中间都要归零——``token_in`` 也一样，录下的主体因此从
            token 0 开始，而不是从预热留下的那个开始。
            """
            self.reset()
            self.token_in.zero_()

        self._graph = runtime.capture(self._forward, warmup, reset=_reset)
        # 批处理预填充路径也预热一下：否则它的第一次调用要付库句柄创建、三角求解器和
        # 池增长的代价。
        self.prefill([1, 2, 3, 4, 5, 6, 7, 8] * 4)
        self.reset()
        if self.mtp is not None:
            self.capture_verify()
            self.mtp.capture_draft((self.cos_tbl, self.sin_tbl, self.mask_idx))
            self.reset()

    @torch.inference_mode()
    def set_rows(self, tokens, positions, slot_rows=None, row_indices=None) -> None:
        """为活跃序列铺好一个解码步。

        这里的一切都是捕获图之外的即时执行主机/设备写入，因为捕获的解码是去读这些
        缓冲区，而不是把它们当参数收。

        Args:
            tokens: 逐行输入 token id，顺序与 ``row_indices`` 一致。
            positions: 逐行绝对位置；它同时寻址 KV 槽位和上下文窗口。
            slot_rows: 可选的逐行块表。每个都会被拷进该行的 ``pos2slot`` 表，没变的
                表会被跳过，稳态的一步因此在这里什么都不写。
            row_indices: 每个条目对应的解码行，用于活跃序列不是连续行 ``0..n-1``
                的情况。只有列出的行会被碰，已结束序列的行因此可以留成洞，交给
                ``decode_batch(pad_rows=...)`` 用哨兵补齐。

        Raises:
            RuntimeError: 这个模型从未调过 ``enable_batching``；单流缓冲区没有可写的
                行维。
            ValueError: ``tokens`` 与 ``positions`` 长度不同、``row_indices`` 也和它们
                对不上，或某个行落在 ``0 .. decode_capacity-1`` 之外，会覆盖暂存行或
                另一个序列的状态。在 runner 那里拒绝这一步。
        """
        if self.batch_capacity <= 1:
            raise RuntimeError("set_rows() needs enable_batching(capacity>1)")
        tokens = list(tokens)
        positions = list(positions)
        n = len(tokens)
        if len(positions) != n:
            raise ValueError(f"{n} tokens vs {len(positions)} positions")
        if row_indices is None:
            rows = list(range(n))
        else:
            rows = [int(r) for r in row_indices]
            if len(rows) != n:
                raise ValueError(f"{n} rows vs {len(rows)} row_indices")
        if any(r < 0 or r >= self.decode_capacity for r in rows):
            raise ValueError(f"row out of 0..{self.decode_capacity - 1}: {rows}")
        if rows == list(range(n)):
            self.token_in[:n].copy_(torch.as_tensor(tokens, dtype=torch.long, device=_device))
            pos_t = torch.as_tensor(positions, dtype=torch.long, device=_device)
            self.pos[:n].copy_(pos_t)
            # 绝对位置 P 意味着位置 0..P 都存在，也就是 P+1 个 key：批处理注意力内核
            # 只在自己这一行的窗口上循环。
            self.ctx_lens[:n].copy_(pos_t.to(torch.int32).add_(1))
        else:
            for r, tok, pos in zip(rows, tokens, positions):
                self.token_in[r] = tok
                self.pos[r] = pos
                self.ctx_lens[r] = pos + 1
        if self.pool is not None and slot_rows is not None:
            for row, blocks in zip(rows, slot_rows):
                blocks = list(blocks)
                key = tuple(blocks)
                if self._row_tables.get(row) == key:
                    continue
                self.pool.assign_table(blocks, 0, row)
                self._row_tables[row] = key

    def capture_batch(self, buckets: tuple[int, ...] = (1, 2, 4, 8),
                      warmup: int = 3) -> None:
        """每个 bucket 捕获一张解码图，它们共用一个内存池。

        每张图都是固定 ``rows`` 的前向，所以投影形状和里面的每次内核 launch 都是静态
        的——这正是批量权重读取得以可能的原因。这些图共用一个 CUDA graph 内存池：它们
        互斥（同一时间只重放一张），所以池只需大到能容下最大的 bucket，而不是各 bucket
        之和。bucket 1 也走*批处理*路径（作为解码行）捕获，而不是走 ``_forward``；
        ``_forward`` 属于单流引擎（``capture()``/``decode_step``），写的是暂存行。

        Args:
            buckets: 要捕获的 bucket 大小。每个必须是 (1, 2, 4, 8) 之一，也就是批处理
                内核实例化的宽度；bucket 1 总会捕获，单个活序列因此永远不会回退。
            warmup: 每个 bucket 捕获之前的即时执行轮数。

        Raises:
            RuntimeError: 没开批处理，或者模型没分页。写时复制块才是让捕获预热不去
                覆写真实序列的东西，所以两个前提都必需。
            ValueError: 某个 bucket 不在 (1, 2, 4, 8) 内。选一个支持的宽度；内核没有
                别的实例化。
        """
        if self.batch_capacity <= 1:
            raise RuntimeError("capture_batch() needs enable_batching(capacity>1)")
        if self.pool is None:
            raise RuntimeError("capture_batch() needs a paged pool (enable_paging())")
        for b in buckets:
            if b not in (1, 2, 4, 8):
                raise ValueError(f"bucket {b} not in (1, 2, 4, 8)")
        order = sorted(set(int(b) for b in buckets) | {1})
        pool_handle = runtime.graph_pool_handle()
        self._batch_graphs = {}
        if self.ctx_lens is not None:
            self.ctx_lens.fill_(1)       # 捕获时的窗口都是 pos 0
        self.reset()
        for b in order:
            if b > self.decode_capacity:
                continue
            impl = lambda b=b: self._forward_batch(b)
            self._batch_graphs[b] = runtime.capture(impl, warmup, pool_handle,
                                                    reset=self.reset)
        self.reset()

    def _stage_sentinels(self, rows: int, bucket: int, pad_rows=()) -> None:
        """把调用方没填的每一行变成无害的哨兵行。

        bucket 图会跑满 ``bucket`` 行——批处理内核不带活掩码，它们就执行构造时的宽度
        ——所以一个洞必须是一行可运行的行，而不是被跳过的行：该行的槽位表指向它自己的
        暂存块，它的 token、位置和上下文长度是首次出现的孤立 token 0 对应的那组（位置
        0、窗口 1）。它的结果会算出来，然后由调用方丢弃。旋转偏移是这次重置的一部分：
        上一个占用者留下的负 ``rope_delta`` 会索引到 cos/sin 表第一行之前。

        Args:
            rows: 活行的高水位：它以下的一切都是活的，``pad_rows`` 里的行除外。
            bucket: 消费这些缓冲区的图（或即时前向）的行数。``rows..bucket-1`` 的行
                也会被铺上。
            pad_rows: ``0 .. rows-1`` 内部的洞，也就是序列已经结束、而后面行仍然活着
                的那些行。

        Raises:
            ValueError: 某个 pad 行落在 ``0 .. bucket-1`` 之外。这样的行没有缓冲区也
                没有暂存块，会弄坏某个活行的状态或索引越界；在 runner 那里拒绝。
        """
        pad = {int(r) for r in pad_rows}
        if any(r < 0 or r >= bucket for r in pad):
            raise ValueError(f"pad_rows outside 0..{bucket - 1}: {sorted(pad)}")
        pad.update(range(rows, bucket))
        for row in sorted(pad):
            if self.pool is not None:
                self.pool.assign(self._scratch_blocks[row], 0, row)
                # 池里的表被改成了暂存块，所以 set_rows 的行表缓存必须一起失效。
                # 不清的话：该行被新序列拿到、且它的块表恰好等于上一个占用者留在缓存里的
                # 那一份（块回收后同序复用很常见），set_rows 就会跳过 assign_table，
                # 于是这一行继续指向只有一块的暂存区——上下文超出一个块之后开始出错，
                # 而按 seq.block_table 做的别名检查发现不了（Python 侧的列表是对的）。
                self._row_tables.pop(int(row), None)
            self.token_in[row] = self._sentinel_token
            self.pos[row] = 0
            self.rope_delta[row] = 0
            if self.ctx_lens is not None:
                self.ctx_lens[row] = 1

    @torch.inference_mode()
    def decode_batch(self, rows: int, bucket: int, pad_rows=()) -> torch.Tensor:
        """重放 ``bucket`` 对应的图；返回全部 ``bucket`` 行的 logits。

        runner 让每个序列待在固定的解码行上，所以一个 bucket 的活行不必连续；
        :meth:`_stage_sentinels` 负责填洞。:meth:`decode_batch_eager` 是同一步但
        不带图，用于不允许捕获的后端。


        Args:
            rows: 活行的高水位：它以下的一切都被占用，``pad_rows`` 里的行除外。
            bucket: 要重放哪张捕获图。它同时固定行数，所以 ``rows..bucket-1`` 的行
                总是作为哨兵跑。
            pad_rows: ``0 .. rows-1`` 内部的洞，也就是序列已经结束、而后面行仍然活着
                的那些行。

        Returns:
            logits 的 ``[bucket, vocab]`` 视图。调用方按 ``seq.row`` 采样；返回的
            缓冲区是活的而不是拷贝，所以只在下一次重放之前有效。

        Raises:
            ValueError: No graph was captured for ``bucket``, ``rows`` is outside
                ``1..bucket``, or a pad row is outside the bucket. The runner
                should size the bucket from the live row count instead of retrying.
        """
        graph = self._batch_graphs.get(bucket)
        if graph is None:
            raise ValueError(f"no graph captured for bucket {bucket}")
        if not 1 <= rows <= bucket:
            raise ValueError(f"rows={rows} outside 1..{bucket}")
        self._stage_sentinels(rows, bucket, pad_rows)
        graph.replay()
        return self.logits_out[:bucket]

    @torch.inference_mode()
    def decode_batch_eager(self, rows: int, bucket: int, pad_rows=()) -> torch.Tensor:
        """即时执行 ``bucket`` 行的批处理解码，不用捕获图。

        给那些绝不能捕获的后端用的入口：SCOW 的驱动把图重放搞坏到能损坏显卡，所以
        那里禁止捕获，也从不调用 :meth:`capture_batch`。铺行、哨兵行和 ``bucket`` 宽
        的输出都与 :meth:`decode_batch` 相同，主体也是同一个
        :meth:`_forward_batch`。那段主体不读任何图——它只读固定缓冲区并推进
        ``pos[:bucket]``——所以这里一个 bucket 的 logits 与同宽捕获图的 logits 相等。

        bucket 集合刻意不校验。runner 的调度器按活行数挑 ``bucket``，而「这个宽度
        捕获过没有」的检查属于图路径，那是唯一存在捕获集合的地方；捕获关掉时根本没
        东西可查。

        Args:
            rows: 活行的高水位：它以下的一切都被占用，``pad_rows`` 里的行除外。
            bucket: 这一步要跑的行数，也就是批处理前向的宽度。任何不超过
                ``decode_capacity`` 的宽度都合法。
            pad_rows: ``0 .. rows-1`` 内部的洞，也就是序列已经结束、而后面行仍然活着
                的那些行。

        Returns:
            logits 的 ``[bucket, vocab]`` 视图。调用方按 ``seq.row`` 采样；缓冲区是
            活的而不是拷贝，所以只在下一次调用之前有效。

        Raises:
            RuntimeError: :meth:`enable_batching` was never called with a capacity
                above 1, or the model has no paged pool. Each sentinel row writes
                KV through its own scratch block, which only those two calls
                reserve (``_prepare_pool_rows``), so without them a staged hole
                would address a block no sequence owns.
            ValueError: ``rows`` and ``bucket`` are not within
                ``1 <= rows <= bucket <= decode_capacity``, or a pad row is outside
                the bucket. Request-shape bugs; reject them at the runner.
        """
        if self.batch_capacity <= 1:
            raise RuntimeError("decode_batch_eager() needs enable_batching(capacity>1)")
        if self.pool is None:
            raise RuntimeError("decode_batch_eager() needs a paged pool (enable_paging())")
        if not 1 <= rows <= bucket:
            raise ValueError(f"rows={rows} outside 1..{bucket}")
        if bucket > self.decode_capacity:
            raise ValueError(
                f"bucket {bucket} exceeds decode_capacity {self.decode_capacity}")
        self._stage_sentinels(rows, bucket, pad_rows)
        self._forward_batch(bucket)
        return self.logits_out[:bucket]

    def enable_spec_batch(self, K: int, buckets: tuple[int, ...] = (1, 2, 4, 8),
                          block_size: int | None = None,
                          max_drafter_len: int = 16384,
                          sparse_topk: int = 0,
                          verify_mode: str | None = None) -> None:
        """分配逐行的草稿器/校验状态，并捕获它们的图。

        每步 ``K`` 份草稿意味着每行校验 ``c = K + 1`` 个 token。草稿器有自己的一个小
        分页 KV 池（每个解码行一张恒等块表）和一张 bucket 图，每次重放推进每一行一次。

        两套校验实现共用同一批缓冲区、同一个逐步接口（:meth:`spec_step_batch`）：

        * ``"once"``——对全部 ``B*c`` 个 token 一次前向：宽 M 投影加多查询分页注意力
          内核（task-12）；快路径；
        * ``"token"``——``c`` 次串行批处理解码前向；更慢，但只需 stage-3 的内核，
          留作回退。

        每行 ``c`` 份回滚快照是贵的那部分（每份 24 层 × 2 MB fp32）；
        ``enable_verify_batch`` 按这个宽度定它们的尺寸。

        Args:
            K: 每个投机步的草稿数。校验宽度是 ``K + 1``，因为主模型自己的下一个 token
                也要被校验。
            buckets: 要为哪些批宽度捕获校验图；超过解码容量的会被丢掉。
            block_size: 草稿器池的块大小；``None`` 复用主池的。
            max_drafter_len: 草稿器池能覆盖的上下文。超过之后 :meth:`MTP.check_len`
                会抛错，调用方可以回退到普通解码。
            sparse_topk: 可选的选择性 KV 校验。在带掩码的多查询内核落地之前必须是 0。
            verify_mode: ``"once"`` 或 ``"token"``；``None`` 时读 ``Q35_SPEC_VERIFY``，
                所需内核缺失时回退到 ``"token"``。默认仍是 c token 校验，因为 B=8 下
                即便 4k 上下文实测都低于贪心（task-13 的结论）；一次性前向的形式作为
                参考路径保留，直到有宽 M 内核按元素胜出 M=8 的那个。

        Raises:
            RuntimeError: 没开批处理、模型没分页，或 checkpoint 里没有 MTP 头。投机
                解码三样都要；否则引擎应当不带它启动。
            ValueError: ``K`` 在 ``1..CKPT-1`` 之外（校验图和回滚快照按 ``CKPT`` 定
                尺寸）、没有 bucket 放得进解码容量，或 ``verify_mode`` 不是那两个名字
                之一。
            NotImplementedError: ``sparse_topk > 0``；带掩码的内核还不存在。
        """
        if self.batch_capacity <= 1:
            raise RuntimeError("enable_spec_batch() needs enable_batching(capacity>1)")
        if self.pool is None:
            raise RuntimeError("enable_spec_batch() needs a paged pool")
        if self.mtp is None:
            raise RuntimeError("enable_spec_batch() needs the MTP head in the checkpoint")
        if int(sparse_topk) > 0:
            raise NotImplementedError(
                "sparse Top-K verify needs the masked attn_paged_multi_q kernel "
                "(task-12); pass sparse_topk=0 for now")
        if verify_mode is None:
            # 默认仍是 c token 校验。一次性前向校验是选项（``Q35_SPEC_VERIFY=once``）：
            # B=8 下即便 4k 上下文实测都低于贪心（task-13 的结论），所以它作为参考路径
            # 保留，直到出现按元素胜出 M=8 的宽 M 内核。

            verify_mode = os.environ.get("Q35_SPEC_VERIFY", "token")
            if verify_mode not in ("once", "token"):
                verify_mode = "token"
            if verify_mode == "once" and not (wide_kernel_ready() and multi_q_ready()
                                              and MULTI_Q_WIRED):
                verify_mode = "token"
        if verify_mode not in ("once", "token"):
            raise ValueError(f"verify_mode must be 'once' or 'token' (got {verify_mode!r})")
        self.verify_mode = verify_mode
        self.sparse_topk = int(sparse_topk)
        # 在为新配置分配（可能尺寸不同的）缓冲区之前，先丢掉上一个配置捕获的图。

        self._verify_batch_graphs = {}
        width = int(K) + 1
        if width < 2 or width > CKPT:
            raise ValueError(f"K must be in 1..{CKPT - 1} (got {K})")
        cap = self.decode_capacity
        buckets = tuple(b for b in buckets if b <= cap)
        if not buckets:
            raise ValueError("no bucket fits the decode capacity")
        self.batch_spec = True
        self.spec_width = width
        self.spec_buckets = buckets
        hidden = self.cfg["hidden_size"]
        vocab = self.vocab
        self.verify_toks_b = torch.zeros(cap, width, dtype=torch.long, device=_device)
        self.verify_pos_b = torch.zeros(cap, width, dtype=torch.long, device=_device)
        self.verify_hidden_b = torch.zeros(cap, width, hidden, dtype=D, device=_device)
        self.verify_logits_b = torch.zeros(cap, width, vocab, dtype=D, device=_device)
        self.spec_toks_b = torch.zeros(cap, width, dtype=torch.long, device=_device)
        self.spec_logits_b = torch.zeros(cap, vocab, dtype=D, device=_device)
        self.h_prev_b = torch.zeros(cap, hidden, dtype=D, device=_device)
        bs = int(block_size or self.pool.bs)
        self.mtp.enable_spec_batch(
            self.batch_capacity, buckets,
            ctx=(self.cos_tbl, self.sin_tbl, self.mask_idx),
            block_size=bs, max_len=max_drafter_len)
        for layer in self.layers:
            mix = layer.mix
            if isinstance(mix, LinearAttn):
                mix.enable_verify_batch(cap, width)
            elif isinstance(mix, FullAttn) and hasattr(mix, "enable_multi_q"):
                mix.enable_multi_q(cap, width)
        self.capture_verify_batch()

    def _verify_batch_once(self, rows: int, prof: dict | None = None) -> None:
        """对每一行的 ``c`` 个 token 做一次前向（task-13 的快路径）。

        布局：``x`` 是行主序的 ``[rows*c, hidden]``（一行那 c 个被校验的 token 连续
        存放）。每层里投影和 MLP 用宽 M 内核对全部 rows*c 个 token 各跑一次；两个有
        状态的一半仍用贪心解码的内核：

        * 全注意力——先把全部 c 行 K/V 写进去，然后一次多查询分页注意力调用，用该行的
          历史服务全部 c 个查询（内核 task-12）。在调用前把它们全部写完是安全的：内核
          会把每个查询掩在它自己的位置上；
        * 线性注意力——卷积/delta 递推对 c 个 token 是串行的（状态依赖），每 token 一次
          批处理 launch，第 j 个 token 之后立刻打状态快照供回滚。

        多查询内核缺失时回退到逐 token 的注意力入口（结果正确，只是这条路径的意义就
        没了）。

        Args:
            rows: 这张图跑多少行，即校验的 bucket 大小。
            prof: 可选的 ``{tag: [(start_event, end_event), ...]}`` 累加器。给了就用
                CUDA 事件给每一段计时以便归因；``None`` 跳过事件，捕获图因此更精简。
        """
        c = self.spec_width
        total = rows * c
        hidden = self.cfg["hidden_size"]
        # 第 5 项是逐行 mRoPE 偏移：每个被校验的 token 按 ``pos + rope_delta`` 旋转，
        # 而它的 KV 槽位仍停在 ``pos``。请求内它是常量，所以就是捕获图读的一个缓冲区。

        ctx = (self.cos_tbl, self.sin_tbl, self.mask_idx, self.ctx_lens,
               self.rope_delta)
        poss = self.verify_pos_b[:rows]
        x = self.embed.lookup(self.verify_toks_b[:rows].reshape(total))
        multi_q = multi_q_ready()
        prof = prof if prof is not None else None

        def lap(tag, start):
            """结束打开着的计时段 ``tag``，返回下一个起点。

            Args:
                tag: 段名，例如 "proj" / "recur" / "attn" / "mlp" / "lmhead"；
                    它会成为 ``prof`` 里的一个键。
                start: 在这一段之前记录的事件；关闭 profiling 时为 ``None``，那就什么
                    都不记。

            Returns:
                原样返回 ``start``。调用方的模式是
                ``s = ...; s.record(); lap(tag, s); s = ...``，所以这个值只在下一个
                事件创建之前作为占位有意义。
            """
            if prof is not None:
                end = runtime.Event()
                end.record()
                prof.setdefault(tag, []).append((start, end))
            return start

        for layer in self.layers:
            mix = layer.mix
            s = runtime.Event() if prof is not None else None
            if s is not None:
                s.record()
            h = layer.in_ln(x)
            if isinstance(mix, LinearAttn):
                qkv, z, b, a = wide_proj(mix.qkvzb_a, h)
                if s is not None:
                    lap("proj", s)
                    s = runtime.Event()
                    s.record()
                qkv = qkv.view(rows, c, -1)
                z = z.view(rows, c, -1)
                b = b.view(rows, c, -1)
                a = a.view(rows, c, -1)
                pieces = [mix.verify_batch_step(qkv[:, j], z[:, j], b[:, j], a[:, j], j)
                          for j in range(c)]
                out = wide_proj(mix.out_proj,
                                torch.stack(pieces, dim=1).reshape(total, -1))
                if s is not None:
                    lap("recur", s)
                    s = runtime.Event()
                    s.record()
            else:
                qg, k, v = wide_proj(mix.qkv, h)
                if s is not None:
                    lap("proj", s)
                    s = runtime.Event()
                    s.record()
                if multi_q:
                    # 多查询入口返回的是 out_proj 之前的注意力输出
                    pre = mix.verify_batch_multi(qg, k, v, poss, ctx, rows)
                    out = wide_proj(mix.o_proj, pre)
                else:
                    # 回退：逐 token 入口已经含了 o_proj
                    qg = qg.view(rows, c, -1)
                    k = k.view(rows, c, -1)
                    v = v.view(rows, c, -1)
                    pieces = [mix.verify_batch_step(qg[:, j], k[:, j], v[:, j],
                                                    poss[:, j], ctx)
                              for j in range(c)]
                    out = torch.stack(pieces, dim=1).reshape(total, -1)
                if s is not None:
                    lap("attn", s)
                    s = runtime.Event()
                    s.record()
            x = x + out
            x = x + layer.mlp.wide(layer.post_ln(x))
            if s is not None:
                lap("mlp", s)
        s = runtime.Event() if prof is not None else None
        if s is not None:
            s.record()
        x = self.norm(x)
        logits = wide_proj(self.embed.gemv, x)
        self.verify_hidden_b[:rows].copy_(x.view(rows, c, hidden))
        self.verify_logits_b[:rows].copy_(
            logits.view(rows, c, self.vocab))
        if s is not None:
            lap("lmhead", s)

    def _verify_batch_impl(self, rows: int) -> None:
        """``c`` 次串行的批处理解码步，每步之后打状态快照。

        每一步都是批处理贪心解码前向，所以一行的校验 logits 恰好就是普通贪心步在那个
        token 上会产出的 logits；逐 token 的循环状态快照就是回滚点。

        （单次多 token 前向——全部 ``B*c`` 个 token 一趟投影——实现并测过，但在
        ``M = B*c = 16`` 时投影退回预填充路径，连带完整的 int8->fp16 反量化（这颗芯片上
        约 250-290 ms 固定成本），比这 ``c`` 次批处理解码步慢 6.9 倍。能一次权重读取
        搞定 M=16/32 的宽 M 批处理 GEMV 才能让单趟校验值得做；在那个内核出现之前，
        这个形态更快。）

        Args:
            rows: 这张图跑多少行，即校验的 bucket 大小。
        """
        ctx_base = (self.cos_tbl, self.sin_tbl, self.mask_idx)
        c = self.spec_width
        ctx = ctx_base + (self.ctx_lens, self.rope_delta)
        for j in range(c):
            toks = self.verify_toks_b[:rows, j]
            poss = self.verify_pos_b[:rows, j]
            self.ctx_lens[:rows].copy_(poss.to(torch.int32).add_(1))
            x = self.embed.lookup(toks)
            for layer in self.layers:
                x = layer.batch(x, poss, ctx)
            x = self.norm(x)
            self.verify_hidden_b[:rows, j].copy_(x)
            self.verify_logits_b[:rows, j].copy_(batch_proj(self.embed.gemv, x))
            for layer in self.layers:
                mix = layer.mix
                if isinstance(mix, LinearAttn):
                    mix.snapshot_batch(rows, j)

    def set_verify_mode(self, mode: str) -> None:
        """切换所捕获的校验实现（``"once"`` / ``"token"``）。

        两者用同一批缓冲区、同一个 ``spec_step_batch`` 接口，所以图可以当场重捕；用于
        A/B 测量，也用于引擎想要便宜回退的时候。

        Args:
            mode: 从今往后要捕获的实现。设成已经在用的模式是空操作，重复请求因此不必
                再付一次捕获代价。

        Raises:
            ValueError: ``mode`` 不是那两个名字之一。
            RuntimeError: 批量投机解码从未启用；没有东西可捕获。
        """
        if mode not in ("once", "token"):
            raise ValueError(f"verify_mode must be 'once' or 'token' (got {mode!r})")
        if mode == self.verify_mode:
            return
        self.verify_mode = mode
        self.capture_verify_batch()

    def capture_verify_batch(self, warmup: int = 2) -> None:
        """每个 bucket 捕获一张校验图，它们共用一个 graph 内存池。

        图录下的是 ``verify_mode`` 选中的那个实现；两者的逐步接口和回滚缓冲区完全
        相同。

        Args:
            warmup: 每个 bucket 捕获之前的即时执行轮数，用来安置分配器和内核句柄。

        Raises:
            RuntimeError: ``enable_spec_batch`` 从未被调用，没有缓冲区可供捕获。
        """
        if not self.batch_spec:
            raise RuntimeError("enable_spec_batch() first")
        handle = runtime.graph_pool_handle()
        self._verify_batch_graphs = {}
        body = (self._verify_batch_once if self.verify_mode == "once"
                else self._verify_batch_impl)
        for b in self.spec_buckets:
            impl = lambda b=b: body(b)
            self._verify_batch_graphs[b] = runtime.capture(impl, warmup, handle)

    @torch.inference_mode()
    def spec_step_batch(self, tokens: list[int], positions: list[int],
                        slot_rows: list[list[int]], bucket: int, K: int,
                        row_indices=None,
                        pad_rows=()) -> tuple[list[list[int]], list[torch.Tensor], list[int]]:
        """在 ``tokens``/``positions`` 上做一个批量投机步。

        Args:
            tokens: 每个序列已经采样出的 token——草稿器就是从它开始。
            positions: 每个序列那个 token 的绝对位置。
            slot_rows: 主模型给这些行用的块表，参数和 :meth:`set_rows` 相同。
            bucket: 要重放哪几张捕获图。bucket 的每一行都会跑，没用上的作为哨兵，
                所以 bucket 必须是捕获过的之一。
            K: 这一步每个序列做多少份草稿；必须放得进 ``spec_width``。
            row_indices: 每个条目固定的解码行；默认是 ``i``，只有活行连续时才正确。
            pad_rows: bucket 内部的洞，也就是本步不参与的序列所在的行。

        Returns:
            ``(accepted_tokens, logits_next, accepted_counts)``：每个输入序列被接受的
            token（*包含* ``tokens[i]``）、被接受前缀之后的 logits（下一个 token 从这里
            采样），以及接受了多少份草稿。调用方按 ``len(accepted_tokens[i])`` 推进该
            序列并铺下一步。

        Raises:
            RuntimeError: 批量投机解码从未启用。
            ValueError: 行下标与 token 数不符或有重复、bucket 没捕获过、``K`` 超过
                捕获的校验宽度，或上下文会冲出草稿器池。
        """
        if not self.batch_spec:
            raise RuntimeError("enable_spec_batch() first")
        n = len(tokens)
        active_rows = (list(range(n)) if row_indices is None
                       else [int(r) for r in row_indices])
        if len(active_rows) != n or len(set(active_rows)) != n:
            raise ValueError(f"{n} tokens vs row_indices {active_rows}")
        if n < 1 or max(active_rows) >= bucket or bucket not in self._verify_batch_graphs:
            raise ValueError(f"bad rows/bucket ({active_rows}, {bucket})")
        if self.mtp is None:
            raise RuntimeError(
                "this checkpoint has no MTP head (mtp_num_hidden_layers = 0), so batched "
                "speculative decoding is unavailable")
        self.mtp.check_len(max(positions) + K + 1)
        width = K + 1
        if width > self.spec_width:
            raise ValueError(f"K={K} needs width {width} > {self.spec_width}")

        self.set_rows([tokens[i] for i in range(n)],
                      [positions[i] for i in range(n)],
                      [slot_rows[i] for i in range(n)],
                      row_indices=active_rows)
        # 给 bucket 图会执行、但我们不用的每一行铺哨兵。
        pad = set(int(r) for r in pad_rows)
        pad |= {r for r in range(bucket) if r not in active_rows}
        for row in sorted(pad):
            self.pool.assign(self._scratch_blocks[row], 0, row)
            self.token_in[row] = self._sentinel_token
            self.pos[row] = 0
            self.ctx_lens[row] = 1
            # 取 0 让旋转留在表内：上一个占用者残留的负 delta 会索引到表的第一行之前。

            self.rope_delta[row] = 0

        # 铺草稿器：上一个被接受 token 处的 h、它的 token，以及草稿器位置（主模型位置
        # 减一：MTP 预测的是 h 之后的 token）。每行的 mRoPE 偏移被拷过去，草稿器因此
        # 和主模型转得一模一样；被填充的行拿 0（它跑在哨兵上）。

        mtp = self.mtp
        mtp.b_rope_delta.zero_()
        for i, row in enumerate(active_rows):
            mtp.b_tok[row] = int(tokens[i])
            mtp.b_h[row].copy_(self.h_prev_b[row])
            mtp.b_pos[row] = int(positions[i]) - 1
            mtp.b_rope_delta[row] = self.rope_delta[row]
        for row in sorted(pad):
            mtp.b_tok[row] = 0
            mtp.b_h[row].zero_()
            mtp.b_pos[row] = 0

        mtp.draft_batch(bucket, K, self.spec_toks_b[:bucket])
        self.verify_toks_b[:bucket, :width].copy_(self.spec_toks_b[:bucket, :width])
        for i, row in enumerate(active_rows):
            self.verify_pos_b[row, :width].copy_(
                torch.arange(int(positions[i]), int(positions[i]) + width,
                             device=_device))
        for row in sorted(pad):
            self.verify_pos_b[row, :width].copy_(
                torch.arange(width, device=_device))
        self._verify_batch_graphs[bucket].replay()

        # 逐活行判断接受：logits[j].argmax() 必须等于下一份草稿。
        pred = self.verify_logits_b[:bucket, :K].argmax(-1)      # [bucket, K]
        match = (pred == self.verify_toks_b[:bucket, 1:width]).tolist()
        accepted_tokens: list[list[int]] = []
        accepted_counts: list[int] = []
        logits_next: list[torch.Tensor] = []
        for row in active_rows:
            j = 0
            while j < K and match[row][j]:
                j += 1
            accepted_counts.append(j)
            accepted_tokens.append(self.verify_toks_b[row, :j + 1].tolist())
            logits_next.append(self.verify_logits_b[row, j].clone())
            for layer in self.layers:
                mix = layer.mix
                if isinstance(mix, LinearAttn):
                    mix.restore_batch(row, j)
            self.h_prev_b[row].copy_(self.verify_hidden_b[row, j])
        return accepted_tokens, logits_next, accepted_counts

    def _verify_impl(self) -> None:
        """用一次穿过各层的即时前向校验铺好的 ``c`` 个 token。

        这是 ``capture_verify`` 录下的主体；它读固定的校验缓冲区，并就地写
        ``verify_hidden``/``verify_logits``，所以图在重放时既不需要参数也不需要分配。

        流水并行下这段和单行 ``_forward`` 用同一套走法：首段自己做嵌入，中间段收
        ``[c, hidden]`` 再往下发；末段算完把 hidden 回给各段、各段自己算 logits。
        不能让末段算完再广播 logits——那条路实测慢 22%（文档 §8.8）。
        """
        c = self.verify_c
        if self.pipe is None or self.pipe.rank == 0:
            x = self.embed.lookup(self.verify_toks)
        else:
            x = self.pipe.await_relay((c, self.hidden), self.pipe.rank - 1,
                                      _device, D)
        # 第 4 项：本请求的 mRoPE 偏移（是个缓冲区，图因此原样重放；``verify_pos``
        # 继续寻址 KV 槽位）。
        ctx = (self.cos_tbl, self.sin_tbl, self.mask_idx, self.rope_delta1)
        for layer in self.layers:
            x = layer.verify(x, self.verify_pos, ctx)
        if self.pipe is None:
            self.verify_hidden.copy_(x)
            self.verify_logits.copy_(self.lm_head.prefill(self.norm(x)))
            return
        nxt = self.pipe.rank + 1
        if nxt >= self.pipe.world_size:
            for dst in range(self.pipe.world_size - 1):
                self.pipe.relay(x, dst)
            h = x
        else:
            self.pipe.relay(x, nxt)
            h = self.pipe.await_relay((c, self.hidden), self.pipe.world_size - 1,
                                      _device, D)
        self.verify_hidden.copy_(h)
        self.verify_logits.copy_(self.lm_head.prefill(self.norm(h)))

    def capture_verify(self, c: int = 4, warmup: int = 3) -> None:
        """把 c token 的校验前向捕获进它自己的图。

        校验 k 份草稿的成本必须大约等于一次权重读取。即时执行时它要花约 k 个解码步，
        主要来自逐 token 递推更新的 CPU launch 开销，所以整个前向捕获一次、每步重放。

        Args:
            c: 校验宽度，也是草稿宽度：每次重放校验 ``K = c - 1`` 份草稿。批处理
                GEMV 只接受 M 属于 {2, 4, 8}，而线性注意力层为回滚保存 ``CKPT`` 份
                状态快照，所以 ``c`` 必须是这些宽度之一。
            warmup: 捕获之前的即时执行轮数。

        Raises:
            ValueError: ``c`` 不是 2、4 或 8，或者超过 ``CKPT``。回滚快照环按 ``CKPT``
                定尺寸，更宽的校验没法倒回。
        """
        if c not in (2, 4, 8) or c > CKPT:
            raise ValueError(f"verify width must be 2, 4 or 8 (got {c})")
        self.verify_c = c
        hidden = self.cfg["hidden_size"]
        self.verify_off = torch.arange(c, dtype=torch.long, device=_device)
        self.verify_toks = torch.zeros(c, dtype=torch.long, device=_device)
        self.verify_pos = self.verify_off.clone()
        self.verify_hidden = torch.zeros(c, hidden, dtype=D, device=_device)
        self.verify_logits = torch.zeros(c, self.vocab, dtype=D, device=_device)
        self._verify_graph = runtime.capture(self._verify_impl, warmup, reset=self.reset)
        self.reset()

    @torch.inference_mode()
    def spec_step(self, logits_prev: torch.Tensor, K: int,
                  draft_tokens: list[int] | None = None) -> tuple[list[int], torch.Tensor]:
        """草拟 K 个 token，一次校验全部 K+1 个，接受匹配的前缀并把循环状态倒回
        到那里。

        草稿有两个来源：``draft_tokens`` 为空时用 checkpoint 自带的 MTP 头；给了就
        用外部草稿器（独立的小模型）的产出。后者是 ``mtp_num_hidden_layers == 0``
        的模型上唯一可用的路——它没有 MTP 头可草拟，但校验和回滚这两半是模型无关的。

        两半都是图重放，每个位置/载荷都留在设备上，所以每步唯一的主机往返就是接受
        判断——而调用方本来就要它才能吐出 token。

        Args:
            logits_prev: 主模型对最后一个已提交 token 的 logits；它的 argmax 给草稿器
                做种子。
            K: 要草拟多少份。随后在一次重放里校验 ``K + 1`` 个 token，所以 ``K + 1``
                必须放得进捕获的校验宽度。
            draft_tokens: 外部草稿器的 ``K`` 个 token；``None`` 表示用 MTP 头自己草拟。

        Returns:
            ``(accepted_token_ids, logits_after_prefix)``。主模型自己的下一个 token
            总会被接受，所以至少推进一个 token；返回的 logits 就是用来采样后续 token
            的那份。

        Raises:
            AssertionError: 没有校验图，或者 ``K + 1`` 超过 ``verify_c``。两者都是
                配置错误：按引擎会用的最宽 ``K`` 捕获校验图。
            RuntimeError: 既没给 ``draft_tokens``、checkpoint 又没有 MTP 头——
                那种模型只能靠外部草稿器，调用方必须自己提供草稿。
            ValueError: ``draft_tokens`` 的长度不是 ``K``。
        """
        assert self._verify_graph is not None, "verification graph is not captured"
        assert K + 1 <= self.verify_c, f"K+1={K + 1} exceeds verify width {self.verify_c}"

        # 草拟：t0 是主模型自己的下一个 token，然后是 K 个串起来的 MTP 步。草稿器在
        # 和主模型相同的 mRoPE 位置上旋转，所以第一次重放之前先把请求的偏移拷过去。

        t0 = logits_prev.argmax()
        if draft_tokens is None:
            if self.mtp is None:
                raise RuntimeError(
                    "this checkpoint has no MTP head (mtp_num_hidden_layers = 0) "
                    "and no external drafts were supplied")
            self.mtp.pos.copy_(self.pos1).sub_(1)
            self.mtp.rope_delta.copy_(self.rope_delta1)
            self.mtp.draft(self.h_prev, t0, K, self.verify_toks[:K + 1])
        else:
            if len(draft_tokens) != K:
                raise ValueError(f"draft_tokens has {len(draft_tokens)} entries, "
                                 f"but K={K}")
            self.verify_toks[0].copy_(t0)
            self.verify_toks[1:K + 1].copy_(
                torch.tensor(draft_tokens, dtype=torch.long, device=t0.device))

        # 校验：重放一次 c token 的图
        self.verify_pos.copy_(self.verify_off).add_(self.pos1)
        self._verify_graph.replay()

        match = (self.verify_logits[:K].argmax(-1) == self.verify_toks[1:K + 1]).tolist()
        j = 0
        while j < K and match[j]:
            j += 1
        toks = self.verify_toks[:j + 1].tolist()

        # 提交被接受的前缀，其余全部倒回
        self.n_drafted += K
        self.n_accepted += j
        self.n_spec_steps += 1
        self.pos1.add_(j + 1)
        self.h_prev.copy_(self.verify_hidden[j])
        # 外部草稿器不需要 MTP 头；只有 MTP 那条路才要同步草稿器的位置缓冲。
        if self.mtp is not None:
            self.mtp.pos.copy_(self.pos1).sub_(1)
        for layer in self.layers:
            mix = layer.mix
            if isinstance(mix, LinearAttn):
                mix.state[self.prefill_row].copy_(mix.ckpt_state[j])
                mix.conv_state[self.prefill_row].copy_(mix.ckpt_conv[j])
        return toks, self.verify_logits[j]

    @torch.inference_mode()
    def spec_generate(self, prompt: list[int], max_new: int, K: int = 3,
                      eos: set[int] | None = None) -> list[int]:
        """贪心投机解码；返回最多 ``max_new`` 个新 token id。

        给基准测试和单请求使用提供便利的入口；服务引擎自己驱动 :meth:`spec_step`，
        以便交错处理请求。


        Args:
            prompt: prompt token id，在第一步之前预填充。
            max_new: 生成这么多 token 之后停下。
            K: 每步的草稿数；透传给 :meth:`spec_step`。
            eos: 结束请求的 token id；``None`` 表示没有。

        Returns:
            按顺序生成的 token；若 EOS token 或 ``max_seq`` 上界先停住了循环，可能
            少于 ``max_new``。

        Raises:
            ValueError: prompt 为空或超过 ``max_seq``；见 :meth:`prefill`。

        """
        eos = eos or set()
        self.n_drafted = self.n_accepted = self.n_spec_steps = 0
        logits = self.prefill(prompt)
        out: list[int] = []
        while len(out) < max_new:
            if int(self.pos1) + self.verify_c > self.max_seq:
                break
            toks, logits = self.spec_step(logits, K)
            for t in toks:
                if t in eos:
                    return out
                out.append(t)
                if len(out) >= max_new:
                    break
        return out

    @property
    def spec_stats(self) -> dict:
        """上一次 :meth:`spec_generate` 运行的接受情况报告。

        Returns:
            单次运行的计数：``steps``、``drafted``、``accepted``、``tokens``，外加
            ``accept_rate``（主模型认可的*草稿* token 占比）和 ``tokens_per_step``。
            后两个才是与加速相关的数字，因为每一步的成本约等于一次校验。


        """
        steps = max(self.n_spec_steps, 1)
        tokens = self.n_accepted + self.n_spec_steps      # 草稿数 + 每步一个额外 token
        return {
            "steps": self.n_spec_steps,
            "drafted": self.n_drafted,
            "accepted": self.n_accepted,
            "tokens": tokens,
            "accept_rate": round(self.n_accepted / max(self.n_drafted, 1), 4),
            "tokens_per_step": round(tokens / steps, 3),
        }

    def decode_step(self, token: int) -> torch.Tensor:
        """把一个 token 喂给捕获的单流图，返回它的 logits。

        Args:
            token: 要追加的 token；它被放到预填充行上，该行的位置随后推进。

        Returns:
            该行的 ``[vocab]`` logits 缓冲区。是活的而不是拷贝：下一次重放会覆盖它，
            所以要在再次调用之前采样。

        Raises:
            AttributeError: ``capture()`` 从未被调用，没有图可重放。让引擎走它正常的
                启动路径。
        """
        pr = self.prefill_row
        self.token_in[pr:pr + 1].fill_(token)
        self._graph.replay()
        return self.logits_single

    @property
    def graph_ready(self) -> bool:
        """报告单流解码图是否已捕获。

        Returns:
            :meth:`capture` 成功之后为 ``True``，从那时起才可以调
            :meth:`decode_step`。
        """
        return self._graph is not None

    def close(self) -> None:
        """丢掉捕获图和层对象。

        运行时服务在重新加载代码之后会重建模型；活着的 ``CUDAGraph`` 会钉住它私有的
        内存池（约占模型占用的一半），所以只丢掉模型引用是不够的。

        """
        self._graph = None
        self._verify_graph = None
        self._batch_graphs = {}
        self._verify_batch_graphs = {}
        if self.mtp is not None:
            self.mtp.close()
        self.layers = []
        self.embed = None
        self.lm_head = None
        self.mtp = None
        self.vision = None
        self.vision_sd = {}
