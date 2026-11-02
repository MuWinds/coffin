"""一个 decoder 层的装配：PLE 注入 + 读门 → mixer → 写门 → 读门 → MoE → 写门。

Qwen4-Exp 的层与 Qwen3.5 的 ``Layer`` 结构上不兼容：

* **层内没有 pre-norm**。既没有 ``input_layernorm`` 也没有
  ``post_attention_layernorm``，归一化全在 ``GatedResidual`` 里、作用在 4 路
  残差流上。补一个会让模型变成另一个模型。
* 残差流是 ``hc_count * hidden_size`` 宽，第一层从 ``inputs_embeds`` 用
  ``repeat(1, 1, hc_count)`` 拉宽而来，4 路初始时逐元素相同。
* mixer 吃的是读门塌缩后的**一路** 2560 宽激活，输出按逐流标量门写回 4 路。
* PLE 在超连接**之前**、直接加进 4 路残差流，10240 宽加法、没有门。

数据流（去掉 PLE）：:

    x                    : [B, S, 10240]
    mixed, hyper, inj_w  = attn_hyper_connection(x)
    h                    = mixer(mixed)          # [B, S, 2560]
    x                    = hyper + (h[:,:,None,:] * inj_w[:,:,:,None]).flatten(-2)
    mixed, hyper, inj_w  = mlp_hyper_connection(x)
    h                    = mlp(mixed)
    x                    = hyper + (h[:,:,None,:] * inj_w[:,:,:,None]).flatten(-2)

Module: ``coffin/models/qwen4exp``。依赖同族的 ``config``、``gated_residual``、
``gdn``、``qsa``、``moe``、``ngram``；被 ``model`` 组装成层栈。

Constraints:
    * ``conv_mask``（``[B, S]`` 的 keep/drop）与 ``attention_mask``（4 维因果
      掩码）是两个不同的参数，绝不能互换：前者按 ``[:, :, None]`` 广播，把 4 维
      的传进去会把序列轴广播进头轴。
    * 层类型从 ``config.layer_types`` 读，不从下标推断。
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from models.qwen4exp.config import INDEXED_ATTENTION, TextConfig
from models.qwen4exp.gated_residual import GatedResidual, apply_injection
from models.qwen4exp.gdn import GatedDeltaNet
from models.qwen4exp.moe import SparseMoeBlock
from models.qwen4exp.ngram import PLELayer
from models.qwen4exp.qsa import Attention, AttentionCache


class DecoderLayer(nn.Module):
    """一个完整的 decoder 层。

    Attributes:
        layer_type: ``linear_attention`` 或 ``indexed_attention``。
        linear_attn: 线性注意力 mixer；只在 ``linear_attention`` 层存在。
        self_attn: QSA mixer；只在 ``indexed_attention`` 层存在。
        mlp: 稀疏 MoE 块，每层恒有一个。
        ple: 只有 ``ple_layer_ids`` 指定的层才有。
    """

    def __init__(self, config: TextConfig, layer_idx: int,
                 quantized: bool = False,
                 local_experts: Sequence[int] | None = None,
                 stream_ngram: bool = False) -> None:
        """按层类型建 mixer，另加 MoE、可选 PLE 与两套超连接。

        Args:
            config: 文本配置。
            layer_idx: 零基层号。
            local_experts: 本 rank 持有的全局专家号（EP 分片）；``None`` 表示
                全部，此时单卡路径与参考实现逐位一致。
            stream_ngram: PLE 的 n-gram 表不分配，改由调用方逐次传入。真实表
                320001536×160 bf16 约 95 GiB，凡是要在 CPU 上跑真实形状的路径
                都必须打开。
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(config, layer_idx)
            self.self_attn = None
        else:
            self.self_attn = Attention(config, layer_idx)
            self.linear_attn = None
        self.mlp = SparseMoeBlock(config, local_experts=local_experts,
                                  quantized=quantized)
        ple_layer_index = config.ple_layer_index(layer_idx)
        self.ple = (PLELayer(config, layer_idx, ple_layer_index,
                             stream_table=stream_ngram)
                    if ple_layer_index is not None else None)
        self.attn_hyper_connection = GatedResidual(config)
        self.mlp_hyper_connection = GatedResidual(config)

    def forward(self, hidden_states: torch.Tensor,
                position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
                attention_mask: torch.Tensor | None = None,
                conv_mask: torch.Tensor | None = None,
                ple_input_ids: torch.Tensor | None = None,
                attention_cache: AttentionCache | None = None,
                conv_state: torch.Tensor | None = None,
                recurrent_state: torch.Tensor | None = None,
                ple_context: torch.Tensor | None = None,
                ple_conv_state: torch.Tensor | None = None,
                ple_table: torch.Tensor | None = None,
                combine: callable | None = None,
                ) -> tuple[torch.Tensor, dict]:
        """跑一层。

        Args:
            hidden_states: 4 路残差流，``[B, S, hc_count * hidden_size]``。
            position_embeddings: 全表 ``(cos, sin)``；线性注意力层不用。
            attention_mask: 4 维因果掩码，只给 ``indexed_attention`` 层。
            conv_mask: ``[B, S]`` 的 keep/drop 标记，喂 GDN 与 PLE。
            ple_input_ids: ``[B, S]`` 的 token id；padding 位置调用方已换成 eos。
            attention_cache: QSA 的 KV + 索引器池。
            conv_state: GDN 卷积窗口。
            recurrent_state: GDN 递推状态。
            ple_context: n-gram 历史 ``[B, context_len]``。
            ple_conv_state: PLE 短卷积历史。
            ple_table: PLE 的 n-gram 嵌入表；``None`` 时用层内自带的那张。
                流式模式下必须给（真实表 ~95 GiB）。
            combine: 路由专家输出的跨 rank 归约，``(x) -> x_reduced``，语义同
                ``all_reduce(SUM)``；``None`` 表示单卡、专家全量持有。EP 分片时
                每个 rank 只算自己那批专家，**必须**每层归约一次——不归约不会
                报错，只是每个 token 少了别人那几个专家的贡献。归约作用在 MoE
                的路由专家那一路，不含共享专家。

        Returns:
            ``(hidden_states, state)``。``state`` 是本层更新后的缓存，键名固定为
            ``attn_cache`` / ``conv_state`` / ``recurrent_state`` / ``ple_context``
            / ``ple_conv_state``，无关的项为 ``None``。

        Raises:
            ValueError: 当前层是 QSA 层但没给 ``position_embeddings`` 或
                ``attention_mask``。QSA 的索引器需要全表位置和因果掩码，缺一个
                都会退化成"看不到历史"或"看到未来"。
        """
        state: dict = {
            "attn_cache": None, "conv_state": None, "recurrent_state": None,
            "ple_context": None, "ple_conv_state": None,
        }
        if self.ple is not None:
            # PLE 在超连接之前、直接加进 4 路残差流；10240 宽加法、没有门。
            # return_state=True 把 n-gram 上下文与短卷积窗口一并交出去：漏掉它们
            # 解码第 2 步之后 PLE 就对着一个永远是 eos 的假上下文哈希，形状全对。
            ple_out, new_ple_context, new_ple_conv_state = self.ple(
                hidden_states, ple_input_ids, previous_context=ple_context,
                history=ple_conv_state, conv_mask=conv_mask, table=ple_table,
                return_state=True)
            hidden_states = hidden_states + ple_out
            state["ple_context"] = new_ple_context
            state["ple_conv_state"] = new_ple_conv_state

        mixed, hyper_input, inject_weights = self.attn_hyper_connection(hidden_states)
        if self.layer_type == INDEXED_ATTENTION:
            if position_embeddings is None or attention_mask is None:
                raise ValueError(
                    "indexed_attention layer needs position_embeddings and attention_mask.")
            mixer_out, new_cache = self.self_attn(
                mixed, position_embeddings, attention_mask, cache=attention_cache)
            state["attn_cache"] = new_cache
        else:
            mixer_out, new_conv_state, new_recurrent_state = self.linear_attn(
                mixed, conv_mask=conv_mask, conv_state=conv_state,
                recurrent_state=recurrent_state)
            state["conv_state"] = new_conv_state
            state["recurrent_state"] = new_recurrent_state

        hidden_states = apply_injection(mixer_out, hyper_input, inject_weights)

        mixed, hyper_input, inject_weights = self.mlp_hyper_connection(hidden_states)
        # combine 交给 MoE 内部：归约必须只覆盖**路由专家**那一路，共享专家是
        # dense 的、每卡都算了一份完整的。放在这里归约会把它放大 world 倍。
        mlp_out = self.mlp(mixed, combine=combine)
        hidden_states = apply_injection(mlp_out, hyper_input, inject_weights)
        return hidden_states, state


__all__ = ["DecoderLayer"]
