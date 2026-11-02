"""QSA（Qwen Sparse Attention）：块级索引器 + 带输出门的全注意力。

参考实现是**掩码版**而不是稀疏核：索引器把 KV 按 ``compress_ratio`` 分块、每块
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
    * 索引器逐 query 选块是 ``B * S`` 次 Python 循环——参考实现就是这么写的，
      正确性基准也只需要它对。向量化（``[B*S, num_blocks]`` 批量 top-k）属于
      预填充性能工作，换实现时必须与本模块逐位对齐。
    * 注意力在 fp32 上做 softmax，再 cast 回 query 的 dtype。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from models.qwen4exp.config import TextConfig
from models.qwen4exp.norm import RMSNorm
from models.qwen4exp.rope import apply_rotary_pos_emb


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
        self.index_qk_proj = nn.Linear(
            config.hidden_size,
            (self.index_n_heads + self.index_kv_heads) * self.index_head_dim,
            bias=False,
        )
        self.q_layernorm = RMSNorm(self.index_head_dim, eps=config.rms_norm_eps)
        self.k_layernorm = RMSNorm(self.index_head_dim, eps=config.rms_norm_eps)

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
               kv_length: int) -> torch.Tensor:
        """逐 query 选块，返回稠密 bool 掩码 ``[1, S, kv_length]``。

        Args:
            query: 已归一化、已旋转的索引器 query，``[B, S, index_n_heads, dim]``。
            raw_keys: ``[B, S, index_dim]``。
            visible: 4 维 bool 可见掩码，``[B, 1, S, kv_length]``。
            full_cos: 全表余弦 ``[B, T, rotary_dim]``。
            full_sin: 全表正弦，同形状。
            kv_length: 掩码的最后一维宽度。

        Returns:
            ``[B, 1, S, kv_length]`` 的 bool 掩码：选中块的全部 token、加上不足
            一整块的尾部。
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
                past_index_keys: torch.Tensor | None = None
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
        mask = self.select(query, all_keys, visible, full_cos, full_sin, kv_length)
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
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim * 2,
                                bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim,
                                bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim,
                                bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size,
                                bias=config.attention_bias)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.indexer = QSAIndexer(config, layer_idx)

    def forward(self, hidden_states: torch.Tensor,
                position_embeddings: tuple[torch.Tensor, torch.Tensor],
                attention_mask: torch.Tensor,
                cache: AttentionCache | None = None
                ) -> tuple[torch.Tensor, AttentionCache]:
        """跑一次 QSA 前向。

        Args:
            hidden_states: ``[B, S, hidden_size]``，已经是读门塌缩后的一路激活。
            position_embeddings: 全表 ``(cos, sin)``，``[B, T, rotary_dim]``。
            attention_mask: 4 维因果掩码。
            cache: 之前的 KV 与索引器池；``None`` 表示从空上下文开始。

        Returns:
            ``(attn_output, cache)``。``cache`` 是新的一份，调用方决定要不要保留。

        Raises:
            ValueError: 掩码不是 4 维，或浮点掩码的 dtype 与 q 不一致到无法相加。
        """
        past_index_keys = None if cache is None else cache.index_keys
        selected_token_mask, raw_index_keys = self.indexer(
            hidden_states, position_embeddings, attention_mask,
            past_index_keys=past_index_keys)
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

        keys = repeat_kv(new_cache.keys, self.num_key_value_groups)
        values = repeat_kv(new_cache.values, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, keys.transpose(2, 3)) * self.scaling
        if merged_mask.dtype != torch.bool:
            attn_weights = attn_weights + merged_mask
        else:
            attn_weights = attn_weights + _additive_mask(merged_mask, attn_weights.dtype)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights.to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, values)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1)

        # 输出门是逐元素 sigmoid，不是 silu。
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output), new_cache


__all__ = ["Attention", "AttentionCache", "QSAIndexer", "repeat_kv"]

