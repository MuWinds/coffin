"""QSA（Qwen Sparse Attention）：块级索引器 + 带输出门的全注意力。

参考实现是掩码版而不是稀疏核：索引器把 KV 按 ``compress_ratio`` 分块、每块
池化成一个 key、用 indexer 的查询头打分、选出 ``block_topk`` 个块，再把整块的
token 放回可见集合；注意力本身仍然跑完整 ``Q @ Kᵀ``，只是把没选的 key 打分置
``-inf``。QSA 因此不省任何矩阵乘——这是正确性基准，稀疏核是后面的事。

* ``q_proj`` 输出是 ``[head0(q), head0(gate), head1(q), head1(gate), ...]``
  交错布局，必须 ``view(..., -1, 2*head_dim)`` 后再 ``chunk(2, -1)``；把前一半
  当 q、后一半当 gate 会让门控随机。
* indexer 的 raw key 是 k_layernorm 之前、RoPE 之前的原始 key，直接进
  cache；主 KV 存的是归一化并旋转之后的 K。两者不能共用一块池。
* 块 key 的旋转角用**块首 token 的绝对位置**、且从**全表** cos/sin 取。
* 打分是 ``relu`` 逐头施加、在求和之前，再除 ``sqrt(index_head_dim)``。
* 最终掩码是因果掩码与索引器掩码的交集；写成并集会泄露未来 token。

Module: ``coffin/models/qwen4exp``。依赖 ``config``、``norm``、``rope``；被
``blocks`` 调用。

Constraints:
    * 选块有两条实现，必须逐位同掩码：``select`` 是批量版（``[B*S, num_blocks]``
      打分 + 一次 top-k），``select_reference`` 是参考实现的逐 query 循环，只留作
      对拍基线。改其中一条就必须让 ``tests/test_qwen4exp_qsa_select.py`` 两边都对得起。
    * 批量版的打分走一次大 matmul，与逐条 matmul 相比最后一位的舍入可能不同；
      掩码在正常数据上逐位一致，但精确并列且只差 1 ulp 的两个块可能换序。这是
      换来 1536 次循环降到 12 次 Python 循环的代价。
    * 注意力在 fp32 上做 softmax，再 cast 回 query 的 dtype。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from models.qwen4exp.config import TextConfig
from models.qwen4exp.norm import RMSNorm
from models.qwen4exp.proj import Proj, proj_from_config
from models.qwen4exp.rope import apply_rotary_pos_emb

from backends import ops as bops

#: 走稀疏注意力的最小 ``kv_length``。
#:
#: 上面那张表是**旧内核时代**的（作业 ``q4sspeed2``，那时稠密路径走的是库内核，
#: kv=2048 只有 0.587 ms/层）。本轮实测（作业 230-232，纯内核、每层一次）：
#:
#:     kv      DENSE_FULL  其中两次 matmul   SPARSE(sel=全部可见)   加速
#:     1024      918.56 us      379.53 us          459.27 us        2.00x
#:     2048     1977.18         732.78             907.95           2.18x
#:     2304     2209.86         822.75            1016.87           2.17x
#:     4096     3780.73        1480.27            1800.44           2.10x
#:
#: 也就是说稀疏核在**每一档**都是稠密的一半上下，而它读的字节数和稠密完全一样
#: （``sel`` 是全部可见位置）。原来那个 4096 的分界白扔了 1024-4096 整段的 2 倍收益，
#: 所以降到 1024 —— 1024 那一档实测仍有 2.00x；**1024 以下没有实测，不要凭感觉往下压**。
#: 另外三条替代路径里 ``attn_decode``（int8 KV）在同一张表上是 1298/2553/2700/3436 us，
#: **比稠密还慢**，所以 KV 缓存保持 bf16 不动。
#:
#: 落到 ``kv=2048``：12 层从 ``12 x 1977 us = 23.7 ms`` 降到 ``12 x 908 us = 10.9 ms``。
_SPARSE_MIN_KV = 1024

#: 批量 decode 仍用稀疏路径的最小 ``kv_length``。
#:
#: ``_sparse_attention`` 在 ``batch > 1`` 时是**逐行各发一次内核**（内核接口只收一条
#: 序列的 KV），所以「稀疏比稠密快 2 倍」这个结论只对单行成立。1024-4096 这一段批量
#: 继续走稠密；4096 以上保持原来的行为（那里稀疏的读放大优势压倒逐行发射）。
#: 等 ``ep4-fix`` 的 B=4 数据出来再决定要不要把这一段也放开给批量。
_SPARSE_BATCH_MIN_KV = 4096

#: 「全看」的块数上限相对预算的倍数，见 :meth:`QSAIndexer.select`。
#:
#: 慢路径（``nonzero`` / 规整块表 / 打分 / top-k / scatter）是**固定成本**：作业 228
#: 在 kv=2304 上量到 ``aten::nonzero`` 6.73 ms/83 次、``aten::topk`` 2.87 ms/12 次、
#: 内核数从 9777 涨到 12178，合计 **+31 ms/步**；而它这一段能剪掉的块只有
#: ``max_blocks - block_topk`` 个（kv=2304 时是 576-512=64 个，约 11% 的 token）。
#: 也就是说上下文不到预算两倍时，慢路径的固定成本远贵于它省下的 KV 读取。
#:
#: 这是一处**近似**：``2048 < kv <= 4096`` 时 QSA 不再按预算丢块，等于把预算临时
#: 放大一倍。它比"丢掉 11% 的块"更保守（数值上更接近全注意力）。
_FULL_SELECT_FACTOR = 2

#: ``(掩码对象身份, kv_length, batch) -> (sel_idx, sel_len, ctx, 掩码)`` 的稀疏选择缓存。
#:
#: 只在 ``_sparse_attention`` 收到 ``full=True``（这一步选择器走的是"全看"快路径）
#: 时使用：全看意味着可见集合就是当步那份 ``attention_mask``，12 个 QSA 层完全相同，
#: 而构造 ``sel_idx`` 要一次 ``nonzero``（设备同步）加两次 H2D 小张量，实测约 50 us/层。
#: 非全看时选择结果依赖各层自己的 index key，绝不能复用。键里带掩码的对象身份、并且
#: 缓存里留住掩码本身，是为了让"同一个键"真的等于"同一份可见集合"。
_SEL_CACHE: dict = {}
_SEL_CACHE_MAX = 8

#: 稠密注意力改用「按 KV 头逐组算」的最小规模，单位是 ``batch * kv_length``。
#:
#: 两种写法由两件事拉扯：逐组省掉「把 KV 复制成 24 份」的拷贝（随位置数线性涨），
#: 付出约两倍的内核数（下限是 ~240 us/层）。实测每层（作业 330，纯内核，decode 形状
#: ``b=1, s=1``；逐组那一列几乎不随位置数变，说明它由固定成本主导）：
#:
#:     kv         7     32     64    128    256    384    512
#:     repeat   146    160    180    221    311    466    566
#:     逐组     252    241    241    242    246    248    248
#:     胜者    repeat repeat repeat repeat 逐组   逐组   逐组
#:
#: 预填充形状（``s=512``）的交叉点更靠右：kv=512 时两者打平（750 vs 763 us，1.7%），
#: kv=2048 时逐组赢 304 us；``B=4`` 从 kv=512 起就是逐组碾压（1885 vs 254 us，因为
#: 要复制 4 倍的 KV）。
#:
#: 所以按 256 分档：低于它走重复写法（与改动前逐位相同，单卡 decode 就在这一档），
#: 高于它走逐组（预填充与批量 decode 都在这一档）。256 落在实测的 128（重复赢 21 us）
#: 与 256（逐组赢 65 us）之间。
_GROUPED_MIN_SPAN = 256


@dataclass
class AttentionCache:
    """一个 QSA 层跨步必须保留的三块状态。

    Attributes:
        keys: 旋转之后的 K，``[B, kv_heads, T, head_dim]``。
        values: V，``[B, kv_heads, T, head_dim]``。
        index_keys: 索引器池，``[B, T, indexer_head_dim]``，
            **k_layernorm 之前、RoPE 之前**的原始 key。它与上面两块是三个独立
            的池：混淆任何一个，检索都会对着错的 key 池做选择。
    """

    keys: torch.Tensor
    values: torch.Tensor
    index_keys: torch.Tensor

    def append(self, keys: torch.Tensor, values: torch.Tensor,
               index_keys: torch.Tensor) -> "AttentionCache":
        """把本步的 K/V/索引 key 接在历史后面。

        Args:
            keys: 本步旋转后的 K，``[B, kv_heads, S, head_dim]``。
            values: 本步 V，形状同 ``keys``。
            index_keys: 本步原始 key，``[B, S, indexer_head_dim]``。

        Returns:
            新的缓存对象（不原地改，前缀复用要能同时持有两份）。
        """
        return AttentionCache(
            keys=torch.cat([self.keys, keys], dim=2),
            values=torch.cat([self.values, values], dim=2),
            index_keys=torch.cat([self.index_keys, index_keys], dim=1),
        )

    def clone(self) -> "AttentionCache":
        """拷一份，供前缀快照 / 投机回滚用。

        稠密缓存本来就是不可变的（``append`` 返回新对象），但**快照必须拷张量**：
        与前进共享同一份 K/V 时，回滚之后长度对、内容已经被写过。

        Returns:
            与 ``self`` 内容相同的新对象，张量是拷贝。
        """
        return AttentionCache(keys=self.keys.clone(), values=self.values.clone(),
                              index_keys=self.index_keys.clone())

    def truncate(self, length: int) -> "AttentionCache":
        """截到前 ``length`` 个位置。

        Args:
            length: 要保留的位置数。

        Returns:
            新的稠密缓存，张量已 ``contiguous``。
        """
        return AttentionCache(keys=self.keys[:, :, :length].contiguous(),
                              values=self.values[:, :, :length].contiguous(),
                              index_keys=self.index_keys[:, :length].contiguous())


class PagedAttentionCache:
    """块粒度预分配的 QSA 缓存：追加是**写入**而不是 ``torch.cat``。

    为什么需要它：稠密实现对每一步都 ``torch.cat`` 一次历史，长度为 ``T`` 的序列
    生成 ``L`` 个 token 就累计 ``O(L^2)`` 的键值拷贝——上下文一长，光这次拷贝就能
    盖过注意力本身。这里改成预分配一块按块对齐的池子，追加只写新位置（``O(S)``），
    没有历史拷贝。

    **当前的分配是顺序的**：一个序列占池子里连续的一段，因此 ``keys`` 仍能返回
    ``[B, :cursor]`` 的连续视图，注意力路径与稠密实现完全一样（数值逐位相同，有单测）。
    真正的「按块表跳着放」要配合分页注意力内核（``attn_paged`` 收 slot map），以及
    前缀复用时按块引用计数与写时复制——那一步等这里的顺序版站住再做，因为分页内核
    的接口要求先有「每行一张块表」这个对象。

    Attributes:
        block_size: 分配粒度（token 数），与 ``cache/block_manager`` 的块同义。
        capacity: 池子能装多少个 token（按 ``block_size`` 向上取整）。
        cursor: 已经写了多少个 token；也是有效前缀的长度。
    """

    def __init__(self, batch: int, max_seq: int, num_kv_heads: int, head_dim: int,
                 indexer_dim: int, *, dtype: torch.dtype, device: torch.device,
                 block_size: int = 64) -> None:
        """按 ``max_seq`` 预分配三块池子（K、V、索引 key）。

        Args:
            batch: 批大小。批维放在最前面，静态批的每行各占一段。
            max_seq: 这个池子要能装下的最长上下文。
            num_kv_heads: KV 头数。
            head_dim: 主注意力头宽。
            indexer_dim: 索引器 key 的宽度（``indexer_head_dim``）。
            dtype: 池子的 dtype，与激活一致。
            device: 设备。
            block_size: 分配粒度。

        Raises:
            ValueError: ``max_seq``、``batch`` 或 ``block_size`` 非正。池子大小是
                显存承诺，形状不合法必须在分配之前就拒绝。
        """
        if batch <= 0 or max_seq <= 0 or block_size <= 0:
            raise ValueError(
                f"paged cache needs positive sizes, got batch={batch} "
                f"max_seq={max_seq} block_size={block_size}")
        self.block_size = int(block_size)
        self.capacity = ((int(max_seq) + self.block_size - 1) // self.block_size
                         ) * self.block_size
        self.cursor = 0
        # 池子按 [B, kv_heads, T, head_dim] 排：这样 `keys` 视图的形状与稠密
        # 实现完全一致，注意力路径走同一批内核，数值才能逐位相同。
        shape = (int(batch), int(num_kv_heads), self.capacity, int(head_dim))
        self.kpool = torch.zeros(shape, dtype=dtype, device=device)
        self.vpool = torch.zeros(shape, dtype=dtype, device=device)
        self.ipool = torch.zeros((int(batch), self.capacity, int(indexer_dim)),
                                 dtype=dtype, device=device)

    @property
    def keys(self) -> torch.Tensor:
        """有效前缀的 K 视图，``[B, cursor, kv_heads, head_dim]``（无拷贝）。"""
        return self.kpool[:, :, :self.cursor]

    @property
    def values(self) -> torch.Tensor:
        """有效前缀的 V 视图，形状同 :attr:`keys`。"""
        return self.vpool[:, :, :self.cursor]

    @property
    def index_keys(self) -> torch.Tensor:
        """有效前缀的索引器 key 视图，``[B, cursor, indexer_dim]``。"""
        return self.ipool[:, :self.cursor]

    def append(self, keys: torch.Tensor, values: torch.Tensor,
               index_keys: torch.Tensor) -> "PagedAttentionCache":
        """把本步的 K/V/索引 key 写进池子，返回自身。

        与稠密实现不同，这里**原地写并返回 self**：池子已经是一份独立的显存，
        再返回一个新对象只会让调用方误以为拿到的是一份可以并存的快照——而快照要
        拷贝（见 :meth:`clone`）。

        Args:
            keys: 本步旋转后的 K，``[B, kv_heads, S, head_dim]``。
            values: 本步 V，形状同 ``keys``。
            index_keys: 本步原始 key，``[B, S, indexer_dim]``。

        Returns:
            ``self``。

        Raises:
            ValueError: 写不下（超过 ``max_seq``）或形状与池子不符。越界写会静默
                覆盖池子尾部之外的内存，报在别处。
        """
        step = int(keys.shape[2])
        if step <= 0:
            return self
        end = self.cursor + step
        if end > self.capacity:
            raise ValueError(
                f"paged cache overflow: cursor {self.cursor} + {step} > "
                f"capacity {self.capacity}（把 max_seq 调大或换一行）")
        if keys.device != self.kpool.device:
            keys = keys.to(self.kpool.device)
            values = values.to(self.vpool.device)
            index_keys = index_keys.to(self.ipool.device)
        self.kpool[:, :, self.cursor:end].copy_(keys)
        self.vpool[:, :, self.cursor:end].copy_(values)
        self.ipool[:, self.cursor:end].copy_(index_keys)
        self.cursor = end
        return self

    def clone(self) -> "PagedAttentionCache":
        """整份快照：只拷有效前缀，不改 ``self``。

        用于前缀复用与投机回滚。代价是一次 ``O(T)`` 拷贝，但只发生在快照边界上，
        而稠密实现是**每一步**都付这个代价。

        Returns:
            新的池子，``cursor`` 与 ``self`` 相同。
        """
        out = PagedAttentionCache(self.kpool.shape[0], self.capacity,
                                  self.kpool.shape[1], self.kpool.shape[3],
                                  self.ipool.shape[2], dtype=self.kpool.dtype,
                                  device=self.kpool.device,
                                  block_size=self.block_size)
        if self.cursor:
            out.kpool[:, :, :self.cursor].copy_(self.kpool[:, :, :self.cursor])
            out.vpool[:, :, :self.cursor].copy_(self.vpool[:, :, :self.cursor])
            out.ipool[:, :self.cursor].copy_(self.ipool[:, :self.cursor])
        out.cursor = self.cursor
        return out

    def truncate(self, length: int) -> "PagedAttentionCache":
        """截到前 ``length`` 个位置（只移动 ``cursor``，不搬运数据）。

        Args:
            length: 要保留的位置数。

        Returns:
            新的池子，``cursor == length``。

        Raises:
            ValueError: ``length`` 超过已经写入的长度。截断到更长的位置会把池子里
                没写过的零当成历史，形状合法、内容全错。
        """
        if length < 0 or length > self.cursor:
            raise ValueError(
                f"cannot truncate paged cache to {length}; it holds {self.cursor}.")
        out = self.clone()
        out.cursor = int(length)
        return out


@dataclass
class IndexReuse:
    """一次可复用的**块选择结果**：只存块号，不存「选中了哪些 token」。

    技术报告 §Implementation Details：*"multi-step MTP reuses top-k indices across
    prediction steps to further reduce draft-model inference costs"*，照的是 GLM 的
    做法。vLLM 的 ``QSAIndexer.skip_topk`` 把它实现成「整个选择行原样返回」，而
    **coffin 不能照抄**，两条例由都是本仓的既有事实：

    1. coffin 的 QSA 掩码是**当步 kv_length 宽的稠密 bool**（:meth:`QSAIndexer.select`
       返回 ``[B, 1, S, kv_length]``），而草稿链上每一步的 ``kv_length`` 都不同。
       复用一个旧宽度的掩码形状直接对不上。
    2. :meth:`QSAIndexer.select` 在 ``max_blocks <= block_topk`` 时有一条整段快路径，
       结果**逐位等于 visible**。``block_topk = 2048/4 = 512``，所以短上下文全部落在
       这条路径上——那里「复用选择」等于复用整个可见掩码，必错。

    所以只复用**段 A**（块打分 + top-k 得到的块号），**段 B**（尾部槽位）永远按当步
    ``visible`` 重算。于是：

    * 尾部恒包含当前位置，最新 token 不会掉出可见集（上游整行复用会让
      ``p+1`` 的 query 看不到它自己）；
    * 因果性严格成立：段 A 的块集合来自 ``[0, p]``、段 B 的尾部来自当步可见集，
      两者都 ⊆ 当步可见集，只会少看、不会多看；
    * 近似只剩「``[p+1, p+k]`` 区间里新凑满的那几个块没被选中」，占比
      ``k / (kv_length / ratio)``。

    Attributes:
        picked: 段 A 选出的块号，``[rows, max_picked]``。
        max_blocks: 产出这份块表时的 ``max_blocks``。下一次调用必须一致——不一致
            说明上下文长度跨了一整块，块号的含义已经变了。
        ratio: 产出时的 ``compress_ratio``。块宽是块号含义的一部分，也必须一致。
    """

    picked: torch.Tensor
    max_blocks: int
    ratio: int

    def check(self, max_blocks: int, ratio: int) -> None:
        """确认这份块表还能用在这一次选块上。

        Args:
            max_blocks: 本次的块数上界。
            ratio: 本次的压缩比。

        Raises:
            ValueError: 块数或压缩比与产出时不一致。静默沿用会让块号指到别的
                位置——形状合法、选出来的 token 全错。
        """
        if ratio != self.ratio:
            raise ValueError(
                f"reusable block table used compress_ratio={self.ratio} but this call "
                f"needs {ratio}; block ids are not comparable across ratios.")
        if max_blocks != self.max_blocks:
            raise ValueError(
                f"reusable block table was built for max_blocks={self.max_blocks} but "
                f"this call has {max_blocks}; the context crossed a block boundary and "
                "the block ids no longer address the same tokens.")


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """把 KV 头复制成 Q 头的份数。

    Args:
        hidden_states: ``[B, kv_heads, S, head_dim]``。
        n_rep: 每个 KV 头对应的 Q 头数。

    Returns:
        ``[B, kv_heads * n_rep, S, head_dim]``。
    """
    if n_rep == 1:
        return hidden_states
    batch, kv_heads, seq, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, None, :, :].expand(batch, kv_heads, n_rep, seq, head_dim)
    return expanded.reshape(batch, kv_heads * n_rep, seq, head_dim)


def _additive_mask(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """把 bool 掩码转成加性的 ``-inf`` 掩码。

    Args:
        mask: 4 维 bool 掩码，``True`` 表示可见。
        dtype: 目标浮点 dtype，用来取最小值。

    Returns:
        与 ``mask`` 同形状的加性掩码：可见处 0、不可见处 ``finfo.min``。
    """
    return torch.where(mask, torch.zeros((), dtype=dtype, device=mask.device),
                       torch.full((), torch.finfo(dtype).min, dtype=dtype, device=mask.device))


class QSAIndexer(nn.Module):
    """块级索引器：把 KV 池化成块、选出 ``block_topk`` 个块、还原成稠密掩码。

    Attributes:
        index_n_heads: 索引器的 query 头数。
        index_head_dim: 索引器头宽。
        token_budget: 每个 query 最多从完整块里选多少 token。
        compress_ratio: 多少连续 token 平均成一个块。
        block_topk: ``token_budget // compress_ratio``，即最多选多少块。
    """

    def __init__(self, config: TextConfig, layer_idx: int) -> None:
        """建索引器的两个投影与两处归一化。

        Args:
            config: 文本配置。

        Raises:
            ValueError: QSA 字段没配齐。索引器缺一个字段时形状仍然能算出来
                （``None`` 会直接炸在乘法上），但错误信息毫无指向性。
        """
        super().__init__()
        if config.indexer_n_heads is None:
            raise ValueError("QSA indexer requires indexer_n_heads in the config.")
        self.layer_idx = layer_idx
        self.index_n_heads = int(config.indexer_n_heads)
        self.index_kv_heads = int(config.indexer_kv_heads)
        self.index_head_dim = int(config.indexer_head_dim)
        self.token_budget = int(config.indexer_budget)
        self.compress_ratio = int(config.indexer_compress_ratio)
        self.block_topk = config.indexer_block_topk
        # 索引器投影同样走 ``Proj``：它是 12 个 QSA 层里每层都跑的窄投影
        # （``N = (4 + 1) * 128 = 640``），而 ``[N, K]`` 行主序在 M=1 下实测只有
        # 17.3 GB/s，换成后端的 ``[K, N]`` GEMV 是它的十倍量级。留成 ``nn.Linear``
        # 会让这 12 层每步都退回那条慢路，而 logits 完全正常，只有账本能发现。
        self.index_qk_proj = proj_from_config(
            config,
            config.hidden_size,
            (self.index_n_heads + self.index_kv_heads) * self.index_head_dim,
            bias=False)
        self.q_layernorm = RMSNorm(self.index_head_dim, eps=config.rms_norm_eps)
        self.k_layernorm = RMSNorm(self.index_head_dim, eps=config.rms_norm_eps)
        #: ``(kv_length, rows)`` → ``max_blocks`` 的缓存，见 :meth:`select`。
        #:
        #: ``max_blocks`` 由可见掩码唯一决定，而 decode 一步里 12 个 QSA 层的
        #: ``kv_length`` 完全相同，所以它一层算一次就够。不加这个缓存的话每层都要
        #: 做一次 ``int(num_blocks.max())``，那是**设备到主机的同步**，每步 12 次。
        self._max_blocks_cache: dict = {}
        #: 上一次 :meth:`select` 是否走了"全看"快路径。``Attention`` 用它判断这一步
        #: 的可见集合是不是就是因果掩码（是的话 12 层完全相同，可以让
        #: ``_sparse_attention`` 共用一份 ``sel_idx``）。
        self._full_select = False

    def project(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """把隐藏状态拆成索引器 query（已归一化）与原始 key。

        Args:
            hidden_states: ``[B, S, hidden_size]``。

        Returns:
            ``(query, raw_keys)``：``query`` 是 ``[B, S, index_n_heads, index_dim]``
            且已过 ``q_layernorm``（**未**旋转）；``raw_keys`` 是
            ``[B, S, index_head_dim]``，归一化与旋转都没做——它就是缓存里的东西。

        Raises:
            ValueError: ``index_kv_heads != 1``。block 池化打分公式建立在"共享
                单个 K 头"上，多头需要另一套打分。
        """
        if self.index_kv_heads != 1:
            raise ValueError("Qwen4-Exp QSA requires indexer_kv_heads=1.")
        batch, seq, _ = hidden_states.shape
        hidden_shape = (batch, seq, -1, self.index_head_dim)
        qk = self.index_qk_proj(hidden_states)
        q, token_k = torch.split(
            qk, [self.index_n_heads * self.index_head_dim,
                 self.index_kv_heads * self.index_head_dim], dim=-1)
        q = q.reshape(*hidden_shape)
        raw_keys = token_k.reshape(*hidden_shape).squeeze(2)
        return self.q_layernorm(q), raw_keys

    def block_keys(self, raw_keys: torch.Tensor, block_token_indices: torch.Tensor,
                   full_cos: torch.Tensor, full_sin: torch.Tensor) -> torch.Tensor:
        """池化一个 query 可见的各块，并归一化、按块首位置旋转。

        Args:
            raw_keys: ``[S, index_head_dim]``，索引器的原始 key。
            block_token_indices: ``[nblocks, compress_ratio]`` 的绝对下标。
            full_cos: 全表余弦，``[T, rotary_dim]``。
            full_sin: 全表正弦，同形状。

        Returns:
            ``[nblocks, index_head_dim]`` 的块 key。

        Raises:
            ValueError: ``block_token_indices`` 的第二维不是 ``compress_ratio``。
        """
        if block_token_indices.shape[-1] != self.compress_ratio:
            raise ValueError(
                f"blocks must be {self.compress_ratio} tokens wide, got "
                f"{tuple(block_token_indices.shape)}.")
        flat = block_token_indices.flatten()
        key_groups = raw_keys.index_select(0, flat).view(
            *block_token_indices.shape, self.index_head_dim)
        # 池化在 fp32 里做：bf16 上先平均再归一化会把小块的能量差抹掉。
        pooled = key_groups.float().mean(dim=1).to(raw_keys.dtype)
        pooled = self.k_layernorm(pooled)
        group_starts = block_token_indices[:, 0]
        rotated = apply_rotary_pos_emb(
            pooled.unsqueeze(1),
            cos=full_cos.index_select(0, group_starts),
            sin=full_sin.index_select(0, group_starts),
        )
        return rotated.squeeze(1)

    def select(self, query: torch.Tensor, raw_keys: torch.Tensor,
               visible: torch.Tensor,
               full_cos: torch.Tensor, full_sin: torch.Tensor,
               kv_length: int,
               reuse: "IndexReuse | None" = None,
               out_reuse: list | None = None) -> torch.Tensor:
        """批量选块：所有 query 的块打分拼成一次 matmul、一次 top-k。

        参考实现是 ``B * S`` 次 Python 循环，每次都是 nonzero / index_select /
        matmul / topk / cat / scatter 一串小算子。prefill seq=128 摊到 12 个 QSA
        层上是 1536 次循环，而每次循环的 GPU 时间远小于发射开销，所以先把它压成
        每个 ``compress_ratio`` 一组的规整表。

        每个 query 能看到的元素个数不同，变量长度无法直接拼成批量张量，因此分两步：
        先把 ``[B*S, kv_length]`` 的可见掩码一次性 nonzero 成"按 query 连续"的
        下标流，用一次 cumsum 求出每个 query 的起点、再用减法还原出它在自己行内的
        名次；名次小于 ``num_blocks * compress_ratio`` 的前缀就是整块部分，其余是
        无条件保留的尾部。整块部分散写进 ``[B*S, max_blocks, compress_ratio]`` 的
        规整表，后面的池化、旋转、打分、top-k 全在表上批量做，一次算子覆盖全部
        query；买不到表的位置（该 query 的块数少于 ``max_blocks``）留着 -1，打分前
        用 ``-inf`` 压掉。

        Args:
            query: 已归一化、已旋转的索引器 query，``[B, S, index_n_heads, dim]``。
            raw_keys: ``[B, S, index_dim]``。
            visible: 4 维 bool 可见掩码，``[B, 1, S, kv_length]``。
            full_cos: 全表余弦 ``[B, T, rotary_dim]``。
            full_sin: 全表正弦，同形状。
            kv_length: 掩码的最后一维宽度。
            reuse: 跨投机步复用来的**块选择结果**（:class:`IndexReuse`）。给了就
                跳过段 A（块池化 + 打分 + top-k），**但段 B 照常按当步 ``visible``
                重算**。理由见 :class:`IndexReuse`。
            out_reuse: 非空列表时，本函数把段 A 的结果（一个 :class:`IndexReuse`）
                append 进去。草稿链的步 0 用它留下 step 1+ 要复用的块表；主干层传
                ``None``，不产生任何额外对象。

        Returns:
            ``[B, 1, S, kv_length]`` 的 bool 掩码：选中块的全部 token、加上不足
            一整块的尾部。

        Raises:
            ValueError: ``reuse`` 的块表宽度与本次 ``max_blocks`` 不符。那说明两次
                调用的上下文长度差得太多，复用的块号已经失去意义。
        """
        batch, seq = query.shape[:2]
        rows = batch * seq
        ratio = self.compress_ratio
        head_dim = self.index_head_dim
        tokens = raw_keys.shape[1]

        # ``max_blocks`` 决定后面所有张量的宽度，所以它必须是一个 Python int。
        #
        # ``decode`` 时每层可见的 token 数只随位置 +1 增长，而 ``num_blocks =
        # floor(visible / ratio)`` —— 它每 ``ratio`` 步才变一次。这里原来每层都
        # ``int(num_blocks.max())``，即一步 12 次设备到主机的同步。改成按
        # ``(kv_length, rows)`` 缓存：同一个 ``kv_length`` 下 ``max_blocks`` 由可见
        # 掩码唯一决定，而 decode 一步里各层的 ``kv_length`` 相同。
        cache_key = (kv_length, rows)
        max_blocks = self._max_blocks_cache.get(cache_key)
        if max_blocks is None:
            # ``counts`` 要算才知道每行可见多少 token；而 ``_flatten_visible`` 会顺带
            # 做一次 ``nonzero``。为了不重复算，这里先只求 ``counts``，只有真正需要
            # 走慢路径时才要下标流。
            counts = visible.reshape(rows, kv_length).sum(dim=1, dtype=torch.long)
            max_blocks = int((counts // ratio).max())
            if len(self._max_blocks_cache) >= 64:
                self._max_blocks_cache.clear()
            self._max_blocks_cache[cache_key] = max_blocks
        else:
            counts = visible.reshape(rows, kv_length).sum(dim=1, dtype=torch.long)

        # 整段快路径：块数没超过预算时，选出来的掩码就是 ``visible`` 本身。
        #
        # 推理：``max_picked = min(block_topk, max_blocks)``，所以 ``max_blocks <=
        # block_topk`` 时 ``max_picked == max_blocks``，``scores.topk(max_blocks)``
        # 从 ``max_blocks`` 个里取 ``max_blocks`` 个，全取。而每行的
        # ``block_part`` 覆盖它自己的 ``num_blocks * ratio`` 个 token、``tail_part``
        # 覆盖剩下的，两者并起来正是该行**全部**可见 token。既然块全被选中、尾部又
        # 无条件保留，结果必然等于 ``visible``。逐位相等，不是近似。
        #
        # 这不是微优化：QSA 层每层 6.61 ms、12 层合计 79.3 ms，占一步 45%
        # 而下面那段 ``_block_table`` / ``_batched_block_keys``
        # / ``.float()`` 的 matmul / ``relu().sum()`` / ``topk`` / ``index_select`` /
        # ``dense_mask`` 在 ``S=1`` 时是纯固定开销（约 60 次 torch 分发）。
        #
        # 覆盖范围：``block_topk = budget / ratio = 2048 / 4 = 512``，而
        # ``max_blocks = floor(kv_length / ratio)``，所以纯按预算算是
        # ``kv_length <= 2048`` 走这里；再乘 :data:`_FULL_SELECT_FACTOR` 之后实际
        # 覆盖到 ``kv_length <= 4096``。
        #
        # 为什么放宽到两倍预算——**这不是降精度近似，而是去掉一段亏本剪枝**。
        # 分两段来看（``_SPARSE_MIN_KV`` 现在是 1024，所以 1024 以上都会走稀疏核）：
        #
        #   * ``kv <= 1024``：注意力走稠密 ``matmul``，稠密会把**整个 kv_length** 都算
        #     一遍再给没选中的位置加 ``finfo.min``。这一段剪掉的块一个字节的读取都省
        #     不到，纯亏。
        #   * ``1024 < kv <= 4096``：走稀疏核，剪枝**确实**能少读几个位置（kv=2304 时
        #     能从 2304 降到 2048，约 11%）。但慢路径是固定成本：作业 228 在 kv=2304
        #     上量到 ``aten::nonzero`` 6.73 ms/83 次、``aten::topk`` 2.87 ms/12 次、
        #     内核数 9777 -> 12178，合计 **+31 ms/步**；而省下的读取按 232 的
        #     1017 us/层折算只有约 1.3 ms/步。31 换 1.3，仍然亏。
        #
        # 代价要写清：``2048 < kv <= 4096`` 时 QSA 实际不再按预算丢块，等于把预算临时
        # 放大一倍；token 会与旧行为**不同**（旧行为把那些位置压成 softmax 权重 0，
        # 新行为保留它们的贡献）。换来的是这一段不再付 31 ms/步，且注意力看到全部
        # 可见上下文——返回 ``visible`` 是"全部因果可见"，比剪枝后的集合更精确。
        #
        # 超过 4096 之后剪枝才开始赚：那时 ``max_blocks`` 是预算的两倍以上，慢路径的
        # 固定成本被"少读一大截 KV"压过去（32768 时稠密 7.56 ms/层 vs 稀疏 0.90）。
        #
        # **这条路径下复用是无害的**（值得写下来，因为 ``IndexReuse`` 的注里点名了
        # 它）：既然全部块都被选中，正确答案就是 ``visible`` 本身，而 ``visible``
        # 本来就是当步算的。所以草稿链从短上下文跨进长上下文时，``reuse`` 会被
        # 忽略、结果仍然正确；反过来从长上下文退回短上下文时同理。真正需要
        # 「复用块、重算尾」的只有下面那条慢路径。
        self._full_select = max_blocks <= self.block_topk * _FULL_SELECT_FACTOR
        if self._full_select:
            if visible.dtype == torch.bool:
                return visible
            return visible.to(torch.bool)

        # 可见掩码的第二维是头，恒为 1；按行展平后下标流就是 [B*S] 行连续。
        entries, _counts = self._flatten_visible(visible, rows, kv_length)
        entry_rows, entry_cols = entries
        num_blocks = counts // ratio
        block_starts = num_blocks * ratio
        # 每个 query 能选中的块数受自己的块数与全局预算双重限制，批量版只能按
        # min(block_topk, max_blocks) 统一取，多出来的名额由 -inf 掩码滤掉。
        max_picked = min(self.block_topk, max_blocks)

        row_first = torch.cumsum(counts, 0) - counts
        entry_rank = torch.arange(entry_rows.numel(), device=query.device) \
            - row_first.index_select(0, entry_rows)
        block_part = entry_rank < block_starts.index_select(0, entry_rows)
        tail_part = ~block_part
        # 尾部无条件保留，槽位紧跟在"最多能选中的块 token"之后。用上限而不是实际
        # 选中数当起点，槽位就只由块数决定、与 top-k 选了什么无关；槽位之间的相对
        # 顺序不影响最终掩码，所以这个静态偏移足够。
        tail_slot = max_picked * ratio + (entry_rank - block_starts.index_select(0, entry_rows))

        if max_blocks > 0:
            table = self._block_table(entry_rows, entry_cols, entry_rank, block_part,
                                      rows, max_blocks, ratio)
            if reuse is not None:
                # 段 A 复用：块池化、打分、top-k 全部跳过，只把块号搬过来。这就是
                # 上游 ``skip_topk`` 省下的那一整串。
                reuse.check(max_blocks, ratio)
                picked = reuse.picked
            else:
                keys = self._batched_block_keys(raw_keys, table, rows, seq, tokens,
                                                full_cos, full_sin, batch)
                scores = torch.matmul(
                    query.reshape(rows, -1, head_dim).float(),
                    keys.float().transpose(1, 2)).transpose(1, 2)
                # relu 逐头施加、在求和之前；除数是 sqrt(index_head_dim)。
                scores = torch.relu(scores).sum(dim=-1) / math.sqrt(head_dim)
                # 够不着 max_blocks 的槽位是补出来的 -1 块，必须先压成 -inf 再
                # top-k，否则它们会被选进结果、再散写成越界的 token 下标。
                block_rank = torch.arange(max_blocks, device=query.device)
                scores = scores.masked_fill(
                    block_rank.unsqueeze(0) >= num_blocks.unsqueeze(1), float("-inf"))
                picked = scores.topk(max_picked, dim=1).indices
            offsets = torch.arange(ratio, device=query.device)
            slot = torch.arange(rows, device=query.device).unsqueeze(1)
            # 选中块的 token 在规整表里是连续的一段，一次 index_select 就展平回
            # "按 query 连续的 token 下标"。
            picked_tokens = table.reshape(-1).index_select(
                0, ((slot * (max_blocks * ratio) + picked * ratio).unsqueeze(-1)
                    + offsets).reshape(-1)).view(rows, max_picked * ratio)
            # 段 B **永远按当步 visible 重算**：尾部不看复用来的任何东西。这就是
            # coffin 与上游必须分开的那一处——上游整行复用会把过期尾部一起搬过来，
            # 于是位置 ``p+1`` 的 query 看不到它自己。
            tail_rows = entry_rows[tail_part]
            tail_tokens = entry_cols[tail_part]
        else:
            # 一个整块都没有：全部 query 只剩尾部，下标流原样搬过去。
            picked_tokens = torch.zeros((rows, 0), dtype=torch.long, device=query.device)
            tail_rows, tail_tokens = entry_rows, entry_cols

        # 缓冲宽度按本次实际需要的列数算，不必固定成 token_budget + ratio - 1：
        # 短上下文（decode、小 prefill）下宽出几十倍，scatter 和随后的稠密化都白做。
        width = max(int(tail_slot.max()) + 1, max_picked * ratio) if entry_rank.numel() \
            else max_picked * ratio
        selected = torch.full((rows, width), -1, dtype=torch.long, device=query.device)
        if picked_tokens.numel():
            selected[:, :max_picked * ratio] = picked_tokens
        if tail_rows.numel():
            selected.view(-1)[tail_rows * width + tail_slot[tail_part]] = tail_tokens
        if out_reuse is not None:
            # 把段 A 的结果交出去，供草稿链后续步复用。只在真的做了选择时才有
            # 意义：``reuse is not None`` 时 ``picked`` 就是搬过来的那一份，再存一次
            # 没有坏处，但调用方不该期待它变成新的来源。
            out_reuse.append(IndexReuse(picked=picked, max_blocks=max_blocks, ratio=ratio))
        return self.dense_mask(selected.view(batch, seq, width), kv_length)

    def _flatten_visible(self, visible: torch.Tensor, rows: int,
                         kv_length: int) -> tuple[tuple[torch.Tensor, torch.Tensor],
                                                  torch.Tensor]:
        """把 ``[B, 1, S, kv_length]`` 的可见掩码展成按行连续的下标流。

        ``torch.nonzero`` 在整表上按行优先遍历，所以同一行的列下标天然按升序
        连续出现——参考实现逐 query 拿到的 ``local_visible`` 正是这个顺序，分块
        才和它对得上。这里把这个性质当作契约用，不再逐行排序。

        Args:
            visible: ``[B, 1, S, kv_length]`` 的 bool 掩码。
            rows: ``B * S``。
            kv_length: 掩码的最后一维宽度。

        Returns:
            ``((entry_rows, entry_cols), counts)``：下标流里每个元素属于哪一行、
            是哪一列，以及每行的可见元素数。
        """
        flat = visible.reshape(rows, kv_length)
        entries = torch.nonzero(flat, as_tuple=False)
        return (entries[:, 0], entries[:, 1]), flat.sum(dim=1, dtype=torch.long)

    @staticmethod
    def _block_table(entry_rows: torch.Tensor, entry_cols: torch.Tensor,
                     entry_rank: torch.Tensor, block_part: torch.Tensor,
                     rows: int, max_blocks: int, ratio: int) -> torch.Tensor:
        """把整块部分的下标散写进 ``[B*S, max_blocks, ratio]`` 的规整表。

        规整表是批量化的前提：池化与旋转要求每组恰好 ``ratio`` 个连续槽位。块数
        不足 ``max_blocks`` 的行由 -1 补齐，取数前统一 clamp，选到哪个补位槽都
        不越界——它们绝不会被选中，因为打分前已经被压成 ``-inf``。

        Args:
            entry_rows: 整块部分元素所属的行。
            entry_cols: 整块部分元素在 ``kv_length`` 里的列下标。
            entry_rank: 整块部分元素在自己行内的名次。
            block_part: 只保留整块部分元素的总掩码。
            rows: ``B * S``。
            max_blocks: 本批最大的整块数。
            ratio: ``compress_ratio``。

        Returns:
            ``[B*S, max_blocks, ratio]`` 的 long 表，空位是 -1。
        """
        width = max_blocks * ratio
        table = torch.full((rows, width), -1, dtype=torch.long, device=entry_cols.device)
        rows_sel = entry_rows[block_part]
        table.view(-1)[rows_sel * width + entry_rank[block_part]] = entry_cols[block_part]
        return table.view(rows, max_blocks, ratio)

    #: 块池化一次处理多少个 query 行。
    #:
    #: 未分块时 ``_batched_block_keys`` 会把 ``[rows * max_blocks, ratio, head_dim]``
    #: 整体 ``.float()`` 一把：``rows = B*S``、``max_blocks ≈ S/ratio``，所以那个中间量
    #: 随序列长度**平方**增长。真机实测（作业 q4prebase1）prefill ``seq=2048`` 时显存
    #: 占到 26.04 GiB / 31.72 GiB 然后 OOM，报错点就在这一行的 ``gathered.float()``。
    #:
    #: 按行切成 256 行一批，峰值降到 ``[256 * max_blocks, ratio, head_dim]``，与总行数
    #: 无关。256 是权衡：再小则每批 launch 的固定开销占比变高，再大则 ``seq=2048`` 仍会
    #: 顶到 1 GiB 量级的临时量。
    _POOL_ROW_CHUNK = 256

    def _batched_block_keys(self, raw_keys: torch.Tensor, table: torch.Tensor,
                            rows: int, seq: int, tokens: int,
                            full_cos: torch.Tensor, full_sin: torch.Tensor,
                            batch: int) -> torch.Tensor:
        """在规整表上批量做块池化 → k_layernorm → 块首旋转。

        与 :meth:`block_keys` 同一套语义，只是把 ``[nblocks, ratio]`` 换成
        ``[B*S*max_blocks, ratio]``：池化仍在 fp32 里做，块首取规整表的第 0 列。
        索引用的是一维偏移 ``batch * T + 位置``，因为规整表里同一 query 的各块
        分属不同 batch 行，二维 index_select 反而要先构造配对下标。

        **按行分块做**，见 :attr:`_POOL_ROW_CHUNK`：整批一把做时那个 fp32 中间量随
        ``S`` 平方增长，``seq=2048`` 直接 OOM。分块只改变处理顺序、不改变任何算术：
        每一行各自池化、各自归一化、各自旋转，行与行之间没有归约。

        Args:
            raw_keys: ``[B, T, index_head_dim]`` 的全部原始 key。
            table: ``[B*S, max_blocks, ratio]`` 的整块下表。
            rows: ``B * S``。
            seq: 每个 batch 行的 query 数。
            tokens: ``T``。
            full_cos: 全表余弦 ``[B, T, rotary_dim]``。
            full_sin: 全表正弦，同形状。
            batch: ``B``。

        Returns:
            ``[B*S, max_blocks, index_head_dim]`` 的块 key。
        """
        ratio = self.compress_ratio
        head_dim = self.index_head_dim
        max_blocks = table.shape[1]
        if rows == 0 or max_blocks == 0:
            return table.new_zeros((rows, max_blocks, head_dim),
                                   dtype=raw_keys.dtype)

        # 这些量只由形状决定，先在块外算一次：每批重复算会把多次 launch 白付回去。
        flat_keys = raw_keys.reshape(batch * tokens, head_dim)
        flat_cos = full_cos.reshape(batch * tokens, -1)
        flat_sin = full_sin.reshape(batch * tokens, -1)

        # 融合快路径：gather → 池化 → RMSNorm → partial RoPE 一次启动。
        #
        # 作业 ``q4blks4`` 量到生产 torch 链与融合内核的差距：
        #
        #     kv      torch 链   融合内核   加速
        #     4096    0.328 ms   0.017 ms   18.9x
        #     8192    0.404       0.024      17.1x
        #     32768   1.071       0.055      19.6x
        #
        # 原链要读的数据只有 ``max_blocks * ratio * head_dim`` 个 bf16（kv=32768 时
        # 8.4 MB，屋顶 0.015 ms），所以融合后基本就是屋顶本身。差在中间量上：
        # gather 出的 ``[blk*max_blocks, ratio, head_dim]``、它的 fp32 副本、
        # layernorm 的多次 elementwise、RoPE 的 ``cat`` —— 每一步都是一次全量读写。
        #
        # 这条**不走分块**：分块（``_POOL_ROW_CHUNK``）原本是为了压住那个 fp32
        # 中间量的显存峰值（``seq=2048`` 会 OOM），而融合后那个中间量根本不存在。
        #
        # **真实入参与直觉不同的三处**（作业 ``q4bdtype2`` 实测，都是踩过的坑）：
        #
        #   1. ``raw_keys`` 是 bf16 但 ``contiguous=False``（它来自转置过的投影输出），
        #      所以必须显式 ``contiguous()``，否则内核直接拒绝。
        #   2. ``table`` 是 **int64**，不是 int32。内核按 int32 读，所以要转。
        #   3. ``flat_cos``/``flat_sin`` 的最后一维是 **64**，而 ``index_head_dim`` 是
        #      128 —— 这是 partial rotary，只有前 64 维旋转、其余原样输出。内核已经
        #      支持（``rotary_dim`` 参数），接线时用 ``flat_cos.shape[-1]`` 传进去。
        blk_keys_op = getattr(bops, "qsa_block_keys", None)
        if blk_keys_op is not None and flat_keys.dtype == torch.bfloat16:
            fused = blk_keys_op(
                flat_keys.contiguous(),
                table.to(torch.int32) if table.dtype != torch.int32 else table,
                # 内核里的增益是 ``1 + weight``（与 ``norm.py`` 的零中心语义一致），
                # 所以传零中心参数本身，而不是 ``_folded_gain()``。
                self.k_layernorm.weight.detach().to(raw_keys.dtype).contiguous(),
                flat_cos, flat_sin,
                seq=seq, tokens=tokens, rotary_dim=flat_cos.shape[-1],
                eps=self.k_layernorm.eps)
            if fused is not None:
                return fused.to(raw_keys.dtype)

        out = torch.empty((rows, max_blocks, head_dim), dtype=raw_keys.dtype,
                          device=raw_keys.device)

        for start in range(0, rows, self._POOL_ROW_CHUNK):
            stop = min(rows, start + self._POOL_ROW_CHUNK)
            blk = stop - start
            block_row = (torch.arange(start * max_blocks, stop * max_blocks,
                                      device=table.device) // max_blocks)
            block_batch = block_row // seq
            # -1 补位槽统一取 0 号 token：它们的 key 是垃圾，但打分会被 -inf 压掉，
            # 任何取值都不影响结果，而 clamp 换掉了逐块判空的分支。
            offsets = table[start:stop].reshape(blk * max_blocks, ratio).clamp(min=0)
            gathered = flat_keys.index_select(
                0, (block_batch.unsqueeze(1) * tokens + offsets).reshape(-1)
            ).view(blk * max_blocks, ratio, head_dim)
            # 池化在 fp32 里做：bf16 上先平均再归一化会把小块的能量差抹掉。
            pooled = self.k_layernorm(gathered.float().mean(dim=1).to(raw_keys.dtype))
            block_start = table[start:stop, :, 0].reshape(blk * max_blocks).clamp(min=0)
            embed_index = block_batch * tokens + block_start
            rotated = apply_rotary_pos_emb(
                pooled.unsqueeze(1),
                cos=flat_cos.index_select(0, embed_index),
                sin=flat_sin.index_select(0, embed_index),
            )
            out[start:stop] = rotated.squeeze(1).view(blk, max_blocks, head_dim)
            # 释放本批的 fp32 临时量，别让它攒到下一批之前：``pooled``/``gathered``
            # 各自是几十 MB 量级，靠 Python 引用计数在循环回到这里时回收。
            del gathered, pooled, rotated
        return out

    def select_reference(self, query: torch.Tensor, raw_keys: torch.Tensor,
                         visible: torch.Tensor,
                         full_cos: torch.Tensor, full_sin: torch.Tensor,
                         kv_length: int) -> torch.Tensor:
        """逐 query 选块的参考实现，只用来给 :meth:`select` 对拍。

        ``B * S`` 次 Python 循环，每次一串小算子，prefill 上慢到不可接受；保留它
        是因为它逐字对应参考实现，改批量版时唯一可靠的判据就是它与批量版逐位
        同掩码。生产路径改用 :meth:`select`。

        Args:
            query: 已归一化、已旋转的索引器 query，``[B, S, index_n_heads, dim]``。
            raw_keys: ``[B, S, index_dim]``。
            visible: 4 维 bool 可见掩码，``[B, 1, S, kv_length]``。
            full_cos: 全表余弦 ``[B, T, rotary_dim]``。
            full_sin: 全表正弦，同形状。
            kv_length: 掩码的最后一维宽度。

        Returns:
            ``[B, 1, S, kv_length]`` 的 bool 掩码。
        """
        batch, seq = query.shape[:2]
        buffer_width = self.token_budget + self.compress_ratio - 1
        selected = torch.full((batch, seq, buffer_width), -1, dtype=torch.long,
                              device=query.device)
        for batch_idx in range(batch):
            for query_idx in range(seq):
                local_visible = torch.nonzero(
                    visible[batch_idx, 0, query_idx], as_tuple=False).flatten()
                num_blocks = int(local_visible.shape[-1]) // self.compress_ratio
                if num_blocks > 0:
                    # 分块按"可见顺序"从头每 compress_ratio 个一组，不是按绝对
                    # 位置对齐。无 padding 的因果前缀下两者等价；有左侧 padding
                    # 时分组会跨过被屏蔽的位置，行为以参考实现为准。
                    block_tokens = local_visible[:num_blocks * self.compress_ratio].view(
                        num_blocks, self.compress_ratio)
                    keys = self.block_keys(raw_keys[batch_idx], block_tokens,
                                           full_cos[batch_idx], full_sin[batch_idx])
                    scores = torch.matmul(
                        query[batch_idx, query_idx].float(), keys.float().transpose(-1, -2)
                    ).transpose(-1, -2)
                    # relu 逐头施加、在求和之前；除数是 sqrt(index_head_dim)。
                    scores = torch.relu(scores).sum(dim=-1) / math.sqrt(self.index_head_dim)
                    picked = scores.topk(min(self.block_topk, num_blocks), dim=0).indices
                    picked_tokens = block_tokens.index_select(0, picked).flatten()
                else:
                    picked_tokens = local_visible.new_zeros(0)
                # 不足一整块的尾部（最近 <compress_ratio 个）无条件保留。
                tail = local_visible[num_blocks * self.compress_ratio:]
                picked_tokens = torch.cat([picked_tokens, tail])
                selected[batch_idx, query_idx, :picked_tokens.numel()] = picked_tokens
        return self.dense_mask(selected, kv_length)

    @staticmethod
    def dense_mask(selected: torch.Tensor, kv_length: int) -> torch.Tensor:
        """把选中的 token 下标还原成稠密 bool 掩码。

        Args:
            selected: ``[B, S, buffer]`` 下标，未用到的槽位是 -1。
            kv_length: 输出掩码的最后一维宽度。

        Returns:
            ``[B, 1, S, kv_length]``：被选中的位置为 ``True``。

        缓冲区里留着的 ``-1`` 槽位统一 scatter 到多出来的第 ``kv_length`` 列、
        再切掉，于是 padding 槽位就此丢弃，不需要逐条比较。
        """
        buffer = torch.zeros((*selected.shape[:-1], kv_length + 1),
                             device=selected.device, dtype=torch.bool)
        scatter_indices = torch.where(selected >= 0, selected,
                                      torch.full_like(selected, kv_length))
        mask = buffer.scatter(-1, scatter_indices, True)[..., :kv_length]
        return mask.unsqueeze(1)

    def forward(self, hidden_states: torch.Tensor,
                position_embeddings: tuple[torch.Tensor, torch.Tensor],
                attention_mask: torch.Tensor,
                past_index_keys: torch.Tensor | None = None,
                reuse: "IndexReuse | None" = None,
                out_reuse: list | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """算索引器掩码，同时把本步的原始 key 交还给调用方去缓存。

        Args:
            hidden_states: ``[B, S, hidden_size]``。
            position_embeddings: 全表 ``(cos, sin)``，形状 ``[B, T, rotary_dim]``；
                只有最后 ``S`` 行属于当前 chunk。
            attention_mask: 4 维因果掩码（bool 或加性浮点），``[B, 1, S, kv_length]``。
            past_index_keys: 之前各步的索引器原始 key，``[B, T - S, index_head_dim]``；
                ``None`` 表示这是第一步。块池化必须看到**全部**历史 key——只拿
                当前 chunk 的 key 去索引绝对下标会越界，而就算不越界，选出来的块
                也不是模型训练时选的那些。
            reuse: 跨投机步复用的**块选择结果**（:class:`IndexReuse`），或 ``None``
                表示本步自己选块。

                MTP 的跨投机步复用走这一支：技术报告明确写了 *"multi-step MTP
                reuses top-k indices across prediction steps"*，vLLM 的
                ``Qwen4ExpMultiTokenPredictor.set_skip_topk`` /
                ``compact_topk_indices`` 就是这件事。coffin 只复用**段 A（块号）**、
                段 B（尾部）按当步可见集重算——理由见 :class:`IndexReuse`，那里也
                写了为什么不能照抄上游的整行复用。
            out_reuse: 非空列表时把本步段 A 的结果 append 进去，供草稿链后续步
                复用。主干层传 ``None``。
        Returns:
            ``(selected_token_mask, raw_keys)``。掩码的语义**跟随入参**：
            bool 掩码进、bool 掩码出；加性浮点掩码进，则出"选中=0 / 未选中=
            ``finfo.min``"的浮点掩码。``raw_keys`` 是
            ``[B, S, index_head_dim]``，用于追加索引器缓存。

        Raises:
            ValueError: 掩码不是 4 维。3 维掩码在这里会广播成一团形状合法的
                垃圾，必须提前拒绝。
        """
        if attention_mask.dim() != 4:
            raise ValueError(
                f"QSA requires a 4-D attention mask, got shape {tuple(attention_mask.shape)}.")
        seq = hidden_states.shape[1]
        full_cos, full_sin = position_embeddings
        current_cos = full_cos[:, -seq:, :]
        current_sin = full_sin[:, -seq:, :]
        query, raw_keys = self.project(hidden_states)
        # 索引器的布局是 [B, S, H, D]，所以 cos/sin 在序列轴之后插头维。
        query = apply_rotary_pos_emb(query, cos=current_cos, sin=current_sin,
                                     unsqueeze_dim=2)
        visible = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
        kv_length = attention_mask.shape[-1]
        # 块池化在完整历史上做：掩码里的下标是绝对位置，而 raw_keys 只有当前
        # chunk。拼接之后再索引，块才和训练时选的是同一批。
        if past_index_keys is None:
            all_keys = raw_keys
        else:
            all_keys = torch.cat([past_index_keys, raw_keys], dim=1)
        if all_keys.shape[1] < kv_length:
            # 掩码比索引池长：说明调用方漏传了历史 key。补 eos 宽度只会产出垃圾，
            # 直接拒绝。
            raise ValueError(
                f"indexer pool has {all_keys.shape[1]} tokens but the mask covers "
                f"{kv_length}; pass past_index_keys for cached steps.")
        mask = self.select(query, all_keys, visible, full_cos, full_sin, kv_length,
                           reuse=reuse, out_reuse=out_reuse)
        if attention_mask.is_floating_point():
            # 浮点掩码路径下参考实现也返回加性掩码：选中 0、未选中 finfo.min。
            mask = _additive_mask(mask, attention_mask.dtype)
        return mask, raw_keys


class Attention(nn.Module):
    """带 QSA 索引器与输出门的全注意力。

    Attributes:
        head_dim: QSA 的头宽（真实 checkpoint 是 256）。
        num_key_value_groups: 每个 KV 头对应的 Q 头数。
        scaling: ``head_dim ** -0.5``。
    """

    def __init__(self, config: TextConfig, layer_idx: int) -> None:
        """建四个投影、两处 QK 归一化与索引器。

        Args:
            config: 文本配置。
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = int(config.head_dim)
        self.num_key_value_groups = config.num_key_value_groups
        self.scaling = self.head_dim ** -0.5
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        # q_proj 的行数是 2 倍：一半 query、一半输出门，按 head 交错排列。
        # 五个投影都走 ``Proj``，理由是真机账本：48 层里 12 个 QSA 层的这五个投影
        # 在一步 decode 里合计 110 ms 的 ``GemmEx_kernel::gemmh`，而同一份权重换成
        # ``[K, N]`` + 后端 GEMV 是 4.5 ms。``bias`` 由 ``proj_from_config`` 按
        # ``attention_bias`` 决定，与参考实现一致。
        self.q_proj = proj_from_config(
            config, config.hidden_size, self.num_heads * self.head_dim * 2)
        self.k_proj = proj_from_config(
            config, config.hidden_size, self.num_key_value_heads * self.head_dim)
        self.v_proj = proj_from_config(
            config, config.hidden_size, self.num_key_value_heads * self.head_dim)
        self.o_proj = proj_from_config(
            config, self.num_heads * self.head_dim, config.hidden_size)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.indexer = QSAIndexer(config, layer_idx)

    def _sparse_attention(self, query_states: torch.Tensor,
                          cache: "AttentionCache",
                          merged_mask: torch.Tensor,
                          kv_length: int,
                          attn_sparse,
                          full: bool = False,
                          mask: torch.Tensor | None = None) -> torch.Tensor | None:
        """上下文够长时走稀疏内核，把选中集合直接喂给它。

        选中集合就是 ``merged_mask``（索引器选中的 ∩ 因果可见）。**不重新计算**：
        把掩码转成 ``sel_idx``/``sel_len`` 比再跑一遍索引器便宜得多，而且保证
        两条路的可见集合逐位一致——那是等价性的前提。

        Args:
            query_states: ``[B, num_heads, 1, head_dim]``，已旋转但**未缩放**——稠密
                路径也是先 ``matmul`` 再乘 ``self.scaling``，所以缩放只施加一次，
                由内核的 ``scale`` 参数负责。
            cache: 含 ``keys``/``values``，形状 ``[B, num_kv_heads, kv_length, head_dim]``。
            merged_mask: ``[B, 1, 1, kv_length]`` 的 bool 掩码。
            kv_length: 缓存里的键数（内核用它做因果上界）。
            attn_sparse: 后端算子；``None`` 表示不可用。
            full: 索引器这一步是否走了"全看"快路径。为真时 12 个 QSA 层看到的可见
                集合就是同一份 ``attention_mask``，``sel_idx`` 可以缓存起来共用。
            mask: 当步的 ``attention_mask`` 本体（不是 ``merged_mask``）。全看时要靠它的
                对象身份当缓存键，并且缓存里必须留住它——否则它被回收之后同一块地址
                会被别的张量复用，缓存就会命中到错的可见集合。

        Returns:
            ``[B, num_heads, 1, head_dim]`` 的注意力输出（未乘输出门），或 ``None``
            表示这个情形不该走稀疏路径。批量时逐行各调一次内核再按 batch 叠回，
            每行的 ``kcache`` 与 ``sel_idx`` 都来自它自己那一行。
        """
        if attn_sparse is None or query_states.shape[2] != 1:
            return None
        batch = query_states.shape[0]
        if batch < 1:
            return None
        if merged_mask.dtype != torch.bool:
            # 加性掩码在这里要反推可见集合，容易出错；生产路径传的是 bool。
            return None
        # 稀疏核按 ``rep`` 把查询头映射到 KV 头，所以 ``nkv`` 必须整除 ``nh``。
        keys = cache.keys
        values = cache.values
        if keys.shape[1] == 0 or query_states.shape[1] % keys.shape[1] != 0:
            return None
        # 内核要求 ``[rows, nh, hd]`` 三维。``query_states`` 是
        # ``[B, nh, 1, hd]``，所以按 ``[:, :, 0, :]`` 收紧**中间的 1** 保留 batch 轴；
        # 写成 ``[0, :, 0, :]`` 会把 batch 轴也去掉、退化成二维，内核直接拒绝。
        q_in = query_states[:, :, 0, :].contiguous()
        # 全看时 12 个 QSA 层的可见集合完全一样（都是同一个 ``attention_mask``），
        # 所以 ``sel_idx`` 可以共用。键用**掩码对象身份**而不是内容：同一个对象 +
        # 全看，才保证各层看到的可见集相同；内容比较要一次设备同步，反而更贵。
        # 缓存里留住掩码本身，这块地址才不会被别的张量复用。
        cache_key = (id(mask), kv_length, batch) if (full and mask is not None) else None
        cached = _SEL_CACHE.get(cache_key) if cache_key is not None else None
        if cached is None and cache_key is not None:
            sels = []
            lens = []
            for row in range(batch):
                visible = merged_mask[row].reshape(kv_length)
                sel = torch.nonzero(visible, as_tuple=False).flatten().to(torch.int32)
                if sel.numel() == 0:
                    return None
                sels.append(sel)
                lens.append(int(sel.numel()))
            # 行宽可能不同（padding 位置），内核契约是 ``[rows, max_sel]`` + 无效槽 -1。
            width = max(lens)
            device = merged_mask.device
            table = torch.full((batch, width), -1, dtype=torch.int32, device=device)
            for row, sel in enumerate(sels):
                table[row, :sel.numel()] = sel
            cached = (table, torch.tensor(lens, dtype=torch.int32, device=device),
                      torch.full((batch,), kv_length, dtype=torch.int32, device=device),
                      mask)
            if len(_SEL_CACHE) >= _SEL_CACHE_MAX:
                _SEL_CACHE.clear()
            _SEL_CACHE[cache_key] = cached
        pieces: list[torch.Tensor] = []
        for row in range(batch):
            # 稀疏核的 ``kcache`` 是一个序列的 KV，``sel_idx`` 是每行一条。
            # 批量 decode 没有「每行一份 KV 池」的内核接口，所以逐行各调一次：
            # 每行只读自己缓存里被选中的位置，而不是让 B 行都乘一遍完整的
            # ``[kv, hd]``。kv=32768 时稠密一层 7.5 ms，逐行稀疏 8 行也只有
            # 8 × 0.9 ms，对现有内核最省事的正确接法，等批量稀疏核落地再
            # 换成一次发射。
            if cached is not None:
                sel, sel_len, ctx = (cached[0][row:row + 1], cached[1][row:row + 1],
                                     cached[2][row:row + 1])
            else:
                visible = merged_mask[row].reshape(kv_length)
                sel = torch.nonzero(visible, as_tuple=False).flatten().to(torch.int32)
                if sel.numel() == 0:
                    return None
                sel = sel.unsqueeze(0)
                sel_len = torch.tensor([sel.numel()], dtype=torch.int32,
                                       device=sel.device)
                ctx = torch.tensor([kv_length], dtype=torch.int32, device=sel.device)
            kcache = keys[row].contiguous()
            vcache = values[row].contiguous()
            out = attn_sparse(q_in[row:row + 1], kcache, vcache, sel, sel_len, ctx,
                              scale=self.scaling)
            pieces.append(out)
        return torch.cat(pieces, dim=0).unsqueeze(2)

    def forward(self, hidden_states: torch.Tensor,
                position_embeddings: tuple[torch.Tensor, torch.Tensor],
                attention_mask: torch.Tensor,
                cache: AttentionCache | None = None,
                reuse: "IndexReuse | None" = None,
                out_reuse: list | None = None,
                ) -> tuple[torch.Tensor, AttentionCache]:
        """跑一次 QSA 前向。

        Args:
            hidden_states: ``[B, S, hidden_size]``，已经是读门塌缩后的一路激活。
            position_embeddings: 全表 ``(cos, sin)``，``[B, T, rotary_dim]``。
            attention_mask: 4 维因果掩码。
            cache: 之前的 KV 与索引器池；``None`` 表示从空上下文开始。
            reuse: 选块复用结果（:class:`IndexReuse`），只给草稿链的步 1+ 用。
            out_reuse: 非空列表时把本步的块选择结果 append 进去；草稿链的步 0 靠
                它把结果留给后续步。

        Returns:
            ``(attn_output, cache)``。``cache`` 是新的一份，调用方决定要不要保留。

        Raises:
            ValueError: 掩码不是 4 维，或浮点掩码的 dtype 与 q 不一致到无法相加。
        """
        past_index_keys = None if cache is None else cache.index_keys
        selected_token_mask, raw_index_keys = self.indexer(
            hidden_states, position_embeddings, attention_mask,
            past_index_keys=past_index_keys, reuse=reuse, out_reuse=out_reuse)
        # 交集，不是并集：索引器只能从因果集合里**删**，写成并集会泄露未来 token。
        # 两种掩码的语义在这里对齐：bool 走 `&`，加性浮点走相加。
        if attention_mask.dtype == torch.bool:
            merged_mask = attention_mask & selected_token_mask
        else:
            merged_mask = attention_mask + selected_token_mask

        seq = hidden_states.shape[1]
        cos_full, sin_full = position_embeddings
        # 主注意力的 cos/sin 也只有最后 S 行属于当前 chunk（indexer 需要全表）。
        cos = cos_full[:, -seq:, :]
        sin = sin_full[:, -seq:, :]
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # 先 view 成 [..., num_heads, 2*head_dim] 再 chunk：q_proj 的行是按 head
        # 交错的 [q0, g0, q1, g1, ...]，先 chunk 再 view 会拿到错位的 q。
        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states, key_states = apply_rotary_pos_emb(
            query_states, cos, sin, k=key_states, unsqueeze_dim=1)

        if cache is None:
            new_cache = AttentionCache(keys=key_states, values=value_states,
                                       index_keys=raw_index_keys)
        else:
            new_cache = cache.append(key_states, value_states, raw_index_keys)

        # 稀疏快路径：上下文够长时，索引器已经把可见集合缩到 ``budget`` 个 token，
        # 而稠密路径仍要扫整个 ``kv_length``（先算出来再被 ``-inf`` 压掉）。
        #
        # 实测：
        #
        #     kv      dense     sparse    加速
        #     512     0.205 ms  0.236 ms  0.87x
        #     2048    0.587     0.906     0.65x
        #     4096    1.229     0.905     1.36x
        #     8192    2.158     0.905     2.38x
        #     32768   7.560     0.904     8.36x
        #
        # 交叉点在 kv = 4096：短上下文下稀疏核「每位置两次归约」的开销比稠密的
        # 库内核贵，长上下文下「只读 2048 个位置」的优势压倒一切。所以按长度分档，
        # 不给短上下文添乱（12 层折算在 kv=2048 时稀疏反而多花 3.83 ms）。
        #
        # 只在 ``seq == 1`` 启用：prefill 的注意力是 ``[S, kv]`` 矩阵乘，本来就吃满
        # 算力，而稀疏核是「一头一行一个 block」，在 prefill 上会退化。
        #
        # ``kv_len`` 取**缓存实际的键数**：``Attention.forward`` 的作用域里没有索引器
        # 那个 ``kv_length``（它在 ``QSAIndexer.select`` 内），而缓存长度才是本层真正
        # 要扫的长度。
        kv_len = int(new_cache.keys.shape[2])
        # 1024 <= kv < 4096 这一段只对**单行**放开稀疏：稀疏核对 batch>1 是逐行发射，
        # 而这一段稠密还不算贵。kv >= 4096 保持原来的行为（对任何 batch 都试稀疏）。
        sparse_min = (_SPARSE_MIN_KV if new_cache.keys.shape[0] == 1
                      else _SPARSE_BATCH_MIN_KV)
        if seq == 1 and kv_len >= sparse_min:
            sparse_out = self._sparse_attention(
                query_states, new_cache, merged_mask, kv_len,
                getattr(bops, "attn_sparse", None),
                full=bool(self.indexer._full_select),
                mask=attention_mask)
            if sparse_out is not None:
                attn_output = sparse_out.reshape(*input_shape, -1)
                attn_output = attn_output * torch.sigmoid(gate)
                return self.o_proj(attn_output), new_cache

        # 稠密分支：不重复 KV 头，按 KV 头分组后让 batch 维广播。
        #
        # 原来先 ``repeat_kv`` 把 ``[B, nkv, kv, hd]`` 复制成 ``[B, nh, kv, hd]``，
        # 再拿它做两次 matmul。那两次复制是**纯广播**：prefill 512、nkv=2、hd=256 时
        # 每层每份 6 MB（K 与 V 各一份 = 12 MB/层），kv=2048 的批解码更是 25 MB/层。
        # 按 KV 头分组之后，把 q 看成 ``[B, nkv, rep, S, hd]``、k 看成
        # ``[B, nkv, 1, hd, kv]``，batch 维自己就会广播——数学完全一样（同一批点积），
        # 但不再物化重复的 KV。
        #
        # 注意 rep == 1 时分组是恒等变换，这条路径与原来逐位相同。
        attn_output = _grouped_attention(
            query_states, new_cache.keys, new_cache.values, merged_mask, self.scaling)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1)

        # 输出门是逐元素 sigmoid，不是 silu。
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output), new_cache


def _grouped_attention(query_states: torch.Tensor, keys: torch.Tensor,
                       values: torch.Tensor, mask: torch.Tensor,
                       scaling: float) -> torch.Tensor:
    """稠密注意力：按 KV 头分组做两次 matmul，不物化重复的 KV 头。

    与 ``repeat_kv`` 之后再算**数学上完全相同**（同一批点积、同样的 softmax、同样的
    V 加权），但省掉两次广播拷贝：``repeat_kv`` 把 ``[B, nkv, kv, hd]`` 展成
    ``[B, nh, kv, hd]``，prefill 512 时每层每份 6 MB（K/V 各一份），kv=2048 的批解码
    是每层 25 MB。分组之后 k/v 的 batch 维是 ``1``，靠广播复用。

    Args:
        query_states: ``[B, nh, S, hd]``，已旋转、**未缩放**。头序必须是「KV 头在前、
            组内连续」，与 :func:`repeat_kv` 的展开顺序一致（第 ``h`` 个头属于第
            ``h // rep`` 个 KV 头）。
        keys: ``[B, nkv, kv, hd]``。
        values: ``[B, nkv, kv, hd]``。
        mask: 与 ``[B, 1, S, kv]`` 可广播的掩码：bool 走交集转加性，浮点直接相加。
        scaling: 打分缩放。

    Returns:
        ``[B, nh, S, hd]``，与 ``repeat_kv`` 版本的输出同形状、同 dtype。

    Raises:
        ValueError: 查询头数不是 KV 头数的整数倍。那说明头映射本身不成立，广播会把
            它算成一个形状合法但错位的注意力。
    """
    batch, heads, seq, head_dim = query_states.shape
    kv_heads = keys.shape[1]
    if kv_heads == 0 or heads % kv_heads != 0:
        raise ValueError(
            f"attention needs heads ({heads}) to be a multiple of kv_heads ({kv_heads})")
    kv_len = keys.shape[2]
    rep = heads // kv_heads
    if rep == 1:
        # 一组一头：分组是恒等变换，直接算，省掉两次 reshape 的分发。
        scores = torch.matmul(query_states, keys.transpose(2, 3)) * scaling
        weights = _masked_softmax(scores, mask, query_states.dtype)
        return torch.matmul(weights, values)
    # 按 KV 头逐组算，每组两次 **3-D 批** matmul（batch = B），全程没有 broadcast
    # 的 batch 维。这一条是量出来的（作业 325，us/层）：
    #
    #     形状                          repeat_kv   5-D 广播   逐组 3-D（这里）
    #     prefill 512 @ kv=512             745.48     886.76        758.36
    #     prefill 512 @ kv=8192          12144.46   18118.47      11348.45
    #     B=4 decode @ kv=2048            6868.61    7106.95        272.05
    #
    # 5-D 广播那一版的 batch 维 stride 是 0，``bmm`` 要求 batch 连续，于是 torch 内部
    # 把它物化成 ``[1, nkv, rep, hd, kv]``——kv=8192 时 **50 MB/次**、每层两次。作业 323
    # 的同树 A/B/A 里它让 kv=8192 的预填充多付了约 0.5 s。
    #
    # 但逐组也不是处处赢：位置很少时它多出来的调度更贵（作业 328，同进程 A/B/A、
    # prompt 8、gen 12：repeat_kv 113.45 ms/步，逐组 117.34 / 116.01）。所以按
    # ``batch * kv_length`` 分档，见 :data:`_GROUPED_MIN_SPAN`。
    if batch * kv_len < _GROUPED_MIN_SPAN:
        flat_k = repeat_kv(keys, rep)
        flat_v = repeat_kv(values, rep)
        scores = torch.matmul(query_states, flat_k.transpose(2, 3)) * scaling
        weights = _masked_softmax(scores, mask, query_states.dtype)
        return torch.matmul(weights, flat_v)
    view = query_states.reshape(batch, kv_heads, rep, seq, head_dim)
    out = torch.empty(batch, kv_heads, rep, seq, head_dim,
                      dtype=query_states.dtype, device=query_states.device)
    for head in range(kv_heads):
        flat_q = view[:, head].reshape(batch, rep * seq, head_dim)
        scores = torch.matmul(flat_q, keys[:, head].transpose(1, 2)) * scaling
        grouped = _masked_softmax(scores.reshape(batch, rep, seq, kv_len), mask,
                                  query_states.dtype)
        out[:, head] = torch.matmul(
            grouped.reshape(batch, rep * seq, kv_len), values[:, head]).reshape(
                batch, rep, seq, head_dim)
    return out.reshape(batch, heads, seq, head_dim)


def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor,
                    dtype: torch.dtype) -> torch.Tensor:
    """加掩码并在 fp32 上做 softmax，再窄化回调用方的 dtype。

    softmax 走 fp32 是承重的：bf16 上做 2048 个位置的概率归一化，尾部会看到明显的
    舍入，而它决定每个位置的权重。这与改动前的写法一致。

    Args:
        scores: ``[B, nh, S, kv]`` 的打分。
        mask: bool 或加性浮点掩码，需与 ``scores`` 可广播。
        dtype: 输出的目标 dtype（调用方的激活 dtype）。

    Returns:
        与 ``scores`` 同形状、``dtype`` 的注意力权重。
    """
    if mask.dtype != torch.bool:
        # 加性掩码本身已经带 ``finfo.min``，直接相加即可。
        scores = scores + mask
    else:
        scores = scores + _additive_mask(mask, scores.dtype)
    return nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(dtype)


__all__ = ["Attention", "AttentionCache", "QSAIndexer", "repeat_kv"]

