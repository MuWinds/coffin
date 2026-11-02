"""文本塔的整机：48 层栈 + 末端塌缩 + lm_head。

三处最容易写错、又都不会抛异常的地方：

* **残差流是 ``hc_count * hidden_size`` 宽**。``inputs_embeds`` 用
  ``repeat(1, 1, hc_count)`` 拉宽，展开顺序是 ``[ch0, ch0, ch0, ch0]``——4 路
  初始时逐元素完全相同，只在第一次写回时开始分叉。忘了这一句数值直接崩。
* **末端没有 final RMSNorm**。``Qwen4ExpTextModel`` 只有 ``embed_tokens``、
  ``layers``、``rotary_emb``、``hyper_connection_mixer``；``lm_head`` 直接吃
  mixer 的输出。补一个会改变 logits 尺度。
* 末端 mixer 是 ``use_combine=False`` 的 ``GatedResidual``，它把
  ``[B, S, 10240]`` 塌回 ``[B, S, 2560]``，这一步就是收口。

Module: ``coffin/models/qwen4exp``。族的顶层之一：依赖同族所有模块；不 import
``backends`` / ``ops``，因为它是与硬件无关的数值基准。

Constraints:
    * 位置表是**全表**，要跨步缓存：indexer 给压缩块算 RoPE 需要"块首 token 的
      绝对位置"，块首可能远在过去。漏掉它，解码第 2 步之后 indexer 的 key 旋转
      角就是错的（不报错，只是检索质量缓慢劣化）。
    * 四类状态（GDN 卷积窗口 / GDN 递推状态 / PLE 短卷积窗口 / n-gram 上下文）
      必须与 KV **同行**：前缀复用、快照、行搬移都要一起带走，否则会把共享的
      KV 配上一个从没见过的递推状态。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from torch import nn

from models.qwen4exp.blocks import DecoderLayer
from models.qwen4exp.config import TextConfig
from models.qwen4exp.gated_residual import GatedResidual
from models.qwen4exp.rope import RotaryEmbedding


@dataclass
class LayerStates:
    """一层跨步需要保留的全部状态。

    ``conv_mask`` 之外的每一项都在 ``DecoderLayer.forward`` 的返回里，按层号
    收集成 :class:`ModelStates`。字段名与 ``blocks`` 里的键名一致。

    Attributes:
        attn_cache: QSA 的 KV + 索引器池；线性层为 ``None``。
        conv_state: GDN 卷积窗口，``[B, conv_dim, kernel - 1]``。
        recurrent_state: GDN 递推状态，``[B, V, dk, dv]`` 的 fp32。
        ple_context: n-gram 历史 token id，``[B, context_len]``。
        ple_conv_state: PLE 短卷积窗口，``[B, hc_hidden, state_len]``。
    """

    attn_cache: object | None = None
    conv_state: torch.Tensor | None = None
    recurrent_state: torch.Tensor | None = None
    ple_context: torch.Tensor | None = None
    ple_conv_state: torch.Tensor | None = None

    @classmethod
    def from_dict(cls, state: dict) -> "LayerStates":
        """从 ``DecoderLayer.forward`` 的状态字典构造。

        Args:
            state: ``DecoderLayer`` 返回的第二项。

        Returns:
            对应的状态对象。
        """
        return cls(
            attn_cache=state.get("attn_cache"),
            conv_state=state.get("conv_state"),
            recurrent_state=state.get("recurrent_state"),
            ple_context=state.get("ple_context"),
            ple_conv_state=state.get("ple_conv_state"),
        )

    def as_dict(self) -> dict:
        """转回 ``DecoderLayer.forward`` 认识的关键字参数。

        Returns:
            可以直接 ``**`` 展开给 ``DecoderLayer`` 的字典。
        """
        return {
            "attention_cache": self.attn_cache,
            "conv_state": self.conv_state,
            "recurrent_state": self.recurrent_state,
            "ple_context": self.ple_context,
            "ple_conv_state": self.ple_conv_state,
        }


@dataclass
class ModelStates:
    """全模型的状态：逐层状态 + 全位置表。

    Attributes:
        layers: 与 ``layers`` 一一对应的状态列表。
        position_ids: 全位置表，``[3, B, T]`` 的 int64。Qwen4 相比 Qwen3.5 新增的
            必需缓存。
    """

    layers: list[LayerStates] = field(default_factory=list)
    position_ids: torch.Tensor | None = None

    @classmethod
    def empty(cls, num_layers: int) -> "ModelStates":
        """建一个全空的初始状态。

        Args:
            num_layers: 层数。

        Returns:
            每层状态都为 ``None`` 的容器。
        """
        return cls(layers=[LayerStates() for _ in range(num_layers)])


class TextModel(nn.Module):
    """Qwen4-Exp 文本塔。

    Attributes:
        embed_tokens: 主词表嵌入，``[vocab_size, hidden_size]``。
        layers: ``num_hidden_layers`` 个 :class:`~blocks.DecoderLayer`。
        rotary_emb: mrope 频率表。
        hyper_connection_mixer: 末端塌缩用的 ``GatedResidual(use_combine=False)``。
    """

    def __init__(self, config: TextConfig,
                 local_experts: Sequence[int] | None = None,
                 stream_ngram: bool = False,
                 quantized: bool = False,
                 combine: callable | None = None) -> None:
        """建嵌入、层栈、频率表与末端 mixer。

        Args:
            config: 文本配置。
            local_experts: 本 rank 持有的全局专家号（EP 分片）；``None`` 表示
                全部。EP=4 时每 rank 128 个，MoE 的 dispatch/combine 由调用方
                的 all-to-all 负责，本模型自己做零次集合通信。
            stream_ngram: PLE 的 n-gram 表不分配。真实 checkpoint 的表约 95 GiB，
                大于任何单机的内存，所以凡是要在 CPU 上跑真实形状的路径都必须
                打开；小配置的 fixture 对齐用默认的 ``False``。
            combine: MoE 专家输出的跨 rank 归约，``(x) -> x_reduced``。EP 分片时
                每层在 MoE 之后调一次，语义同 ``all_reduce(SUM)``；``None`` 表示
                单卡。入参形状是 ``[B, S, hidden_size]`` 的 fp32，即
                ``parallel.ShmAllReduce`` 要的那个。
        """
        super().__init__()
        self.config = config
        self.combine = combine
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size,
                                         config.pad_token_id)
        self.layers = nn.ModuleList(
            [DecoderLayer(config, layer_idx, local_experts=local_experts,
                          stream_ngram=stream_ngram, quantized=quantized)
             for layer_idx in range(config.num_hidden_layers)])
        self.rotary_emb = RotaryEmbedding(config)
        # 末端没有 final RMSNorm，塌缩由这个 use_combine=False 的 mixer 完成。
        self.hyper_connection_mixer = GatedResidual(config, use_combine=False)

    def prepare_position_ids(self, position_ids: torch.Tensor | None, batch: int, seq: int,
                             past_len: int = 0, device: torch.device | None = None
                             ) -> tuple[torch.Tensor, torch.Tensor]:
        """把外部位置表规整成 (文本位置, 三轴 mrope 位置)。

        接受参考实现的四种输入形态：``None``（从 ``past_len`` 起 arange）、
        4 行（第 0 行文本位置、第 1..3 行三轴）、1 行（文本位置，扩成三轴）、
        2 维（先扩一维再按 1 行处理）。

        Args:
            position_ids: 外部位置表，形状见上；``None`` 时自动生成。
            batch: batch 大小。
            seq: 本次 chunk 的 token 数。
            past_len: 已经处理过的 token 数（`position_ids` 为 ``None`` 时用）。
            device: 生成位置表用的设备。

        Returns:
            ``(text_position_ids, mrope_position_ids)``：前者 ``[B, S]`` 给因果
            掩码，后者 ``[3, B, S]`` 给 RoPE。

        Raises:
            ValueError: 位置表的行数既不是 1、3 也不是 4。
        """
        if position_ids is None:
            base = torch.arange(seq, device=device, dtype=torch.long) + past_len
            position_ids = base.view(1, 1, -1).expand(4, batch, -1)
        if position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)
        rows = position_ids.shape[0]
        if rows == 4:
            return position_ids[0], position_ids[1:]
        if rows == 1:
            return position_ids[0], position_ids.expand(3, -1, -1)
        if rows == 3:
            # 纯文本服务可以只传三轴；文本位置与第 0 轴相同。
            return position_ids[0], position_ids
        raise ValueError(
            f"position_ids must have 1, 3 or 4 rows, got {rows}.")

    def forward(self, input_ids: torch.Tensor | None = None,
                inputs_embeds: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None,
                conv_mask: torch.Tensor | None = None,
                position_ids: torch.Tensor | None = None,
                ple_input_ids: torch.Tensor | None = None,
                states: ModelStates | None = None,
                ple_tables: dict[int, torch.Tensor] | None = None,
                use_cache: bool = True
                ) -> tuple[torch.Tensor, ModelStates]:
        """跑文本塔。

        Args:
            input_ids: ``[B, S]`` 的 token id；与 ``inputs_embeds`` 二选一。
            inputs_embeds: ``[B, S, hidden_size]``；与 ``input_ids`` 二选一。
            attention_mask: QSA 层的 4 维因果掩码（或其来源）。
            conv_mask: ``[B, S]`` 的 keep/drop 标记，喂 GDN 与 PLE。
            position_ids: 位置表，见 :meth:`prepare_position_ids`。
            ple_input_ids: PLE 用的原始 token id；``None`` 时取 ``input_ids``。
            states: 上一步的状态；``None`` 表示从空上下文开始。
            ple_tables: 按层号给出的 n-gram 表，覆盖层内自带的那张。流式模式下
                必须给——真实表 ~95 GiB，调用方按分片查完行再传进来。
            use_cache: 是否返回更新后的状态。

        Returns:
            ``(hidden_states, new_states)``：``[B, S, hidden_size]`` 与更新后的
            状态。``use_cache=False`` 时第二项仍是容器，但层内状态为 ``None``。

        Raises:
            ValueError: ``input_ids`` 与 ``inputs_embeds`` 都给了或都没给，或
                PLE 开着却没有 token id。PLE 需要 token id，光有嵌入不行——参考
                实现是拿嵌入矩阵反查，引擎侧永远有 id。
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("specify exactly one of input_ids or inputs_embeds.")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if ple_input_ids is None and self.config.ple_layer_ids:
            if input_ids is None:
                raise ValueError(
                    "PLE needs token ids; pass ple_input_ids when using inputs_embeds.")
            ple_input_ids = input_ids

        batch, seq, _ = inputs_embeds.shape
        states = ModelStates.empty(len(self.layers)) if states is None else states
        past_len = 0 if states.position_ids is None else states.position_ids.shape[-1]
        text_positions, mrope_positions = self.prepare_position_ids(
            position_ids, batch, seq, past_len, inputs_embeds.device)
        # 全表：带 cache 时把历史位置接在前面，indexer 算块首旋转角要用。
        if states.position_ids is not None:
            full_positions = torch.cat([states.position_ids, mrope_positions], dim=-1)
        else:
            full_positions = mrope_positions
        new_states = ModelStates(layers=list(states.layers),
                                 position_ids=full_positions)
        cos, sin = self.rotary_emb.cos_sin(full_positions, dtype=inputs_embeds.dtype)

        # repeat 在最后一维复制，展开顺序是 [ch0, ch0, ch0, ch0]：4 路初始时
        # 逐元素完全相同，第一次写回才开始分叉。
        hidden_states = inputs_embeds.repeat(1, 1, self.config.hc_count)

        new_layers: list[LayerStates] = []
        for layer_idx, layer in enumerate(self.layers):
            previous = states.layers[layer_idx] if layer_idx < len(states.layers) else LayerStates()
            hidden_states, state = layer(
                hidden_states,
                position_embeddings=(cos, sin),
                attention_mask=attention_mask,
                conv_mask=conv_mask,
                ple_input_ids=ple_input_ids,
                ple_table=None if ple_tables is None else ple_tables.get(layer_idx),
                combine=self.combine,
                **previous.as_dict())
            new_layers.append(LayerStates.from_dict(state) if use_cache else LayerStates())
        new_states.layers = new_layers

        hidden_states = self.hyper_connection_mixer(hidden_states)
        return hidden_states, new_states


class CausalLM(nn.Module):
    """文本塔加语言模型头。

    Attributes:
        model: :class:`TextModel`。
        lm_head: ``[vocab_size, hidden_size]``，**不带** final norm。
    """

    def __init__(self, config: TextConfig,
                 local_experts: Sequence[int] | None = None,
                 stream_ngram: bool = False,
                 quantized: bool = False,
                 combine: callable | None = None) -> None:
        """建文本塔与 lm_head。

        Args:
            config: 文本配置。
            local_experts: 本 rank 持有的全局专家号（EP 分片）；见
                :class:`TextModel`。
            stream_ngram: 不分配 PLE 的 n-gram 表；见 :class:`TextModel`。
            combine: MoE 专家输出的跨 rank 归约；见 :class:`TextModel`。
        """
        super().__init__()
        self.config = config
        self.model = TextModel(config, local_experts=local_experts,
                               quantized=quantized,
                               stream_ngram=stream_ngram, combine=combine)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor | None = None,
                inputs_embeds: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None,
                conv_mask: torch.Tensor | None = None,
                position_ids: torch.Tensor | None = None,
                ple_input_ids: torch.Tensor | None = None,
                states: ModelStates | None = None,
                ple_tables: dict[int, torch.Tensor] | None = None,
                logits_to_keep: int = 0,
                use_cache: bool = True
                ) -> tuple[torch.Tensor, ModelStates]:
        """跑一次前向并返回 logits。

        参考实现在 ``hidden_states`` 与 ``lm_head`` 之间**没有** anything else
        ——没有 final RMSNorm，也没有额外的 cast。

        Args:
            input_ids: 见 :meth:`TextModel.forward`。
            inputs_embeds: 见 :meth:`TextModel.forward`。
            attention_mask: 见 :meth:`TextModel.forward`。
            conv_mask: 见 :meth:`TextModel.forward`。
            position_ids: 见 :meth:`TextModel.forward`。
            ple_input_ids: 见 :meth:`TextModel.forward`。
            states: 见 :meth:`TextModel.forward`。
            ple_tables: 见 :meth:`TextModel.forward`。
            logits_to_keep: 只算最后这么多 token 的 logits；``0`` 表示全算。
            use_cache: 是否返回状态。

        Returns:
            ``(logits, new_states)``：``[B, S', vocab_size]``。
        """
        hidden_states, new_states = self.model(
            input_ids=input_ids, inputs_embeds=inputs_embeds,
            attention_mask=attention_mask, conv_mask=conv_mask,
            position_ids=position_ids, ple_input_ids=ple_input_ids,
            states=states, ple_tables=ple_tables, use_cache=use_cache)
        slice_indices = slice(-logits_to_keep, None) if logits_to_keep else slice(None)
        return self.lm_head(hidden_states[:, slice_indices, :]), new_states


__all__ = ["CausalLM", "LayerStates", "ModelStates", "TextModel"]
