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

Module: ``coffin/models/qwen4exp``。族的顶层之一：依赖同族所有模块；它不 import
``backends`` / ``ops``，因为它是与硬件无关的数值基准。唯一的例外是 ``proj``：稠密
投影的布局与分派由后端决定，这是全族唯一需要平台事实的地方，所以那件事整个搬到
``proj`` 里，本文件仍然只写几何。

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
from models.qwen4exp.mtp import DraftIndexPlan, MTP
from models.qwen4exp.proj import Proj, QuantizedEmbedding, release_quantized_placeholders
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

    def snapshot(self) -> "ModelStates":
        """拷一份状态，供投机校验之后回滚。

        为什么需要它：GQA/QSA 那几层按位置索引、可以截断，但**线性注意力**（GDN）
        的 ``conv_state`` / ``recurrent_state`` 是按步推进的递推量，没有「第几个位置」
        的概念——它只能整份存、整份还。投机校验一次吃掉 ``K+1`` 行，接受 ``a`` 份之后
        要回到「``ctx + a + 1`` 个位置」那个状态，而那个状态只有在校验**之前**存下来
        才有。

        上游（vLLM）的做法是给 GDN 的 Mamba 缓存额外分配 ``num_speculative_tokens``
        个槽位、按位置回填；这里没有分页池，所以退一档用整份快照。代价是一次
        ``[B, V, dk, dv]`` 的 fp32 拷贝（真机单行约 12 MiB 量级），而收益是正确性。

        Returns:
            与 ``self`` 内容相同的**新**容器；张量是拷贝，不是视图。``None`` 状态原样
            保留（空状态没有可回滚的东西）。
        """
        layers: list[LayerStates] = []
        for layer in self.layers:
            # ``attn_cache`` 也要拷：``AttentionCache.append`` 虽然返回**新**对象，
            # 但它的 K/V 是 ``torch.cat`` 出来的新张量、而索引 key 池在别处可能被
            # 原地续写。真正致命的是「快照与前进共享同一份 K/V」——那样回滚之后
            # 长度是对的、内容却已经被校验步写过了，症状是重放出来的 logits 与
            # 干净路径差一点点，而前缀 token 全都对。
            cache = layer.attn_cache
            if cache is not None:
                cache = type(cache)(
                    keys=cache.keys.clone(),
                    values=cache.values.clone(),
                    index_keys=cache.index_keys.clone(),
                )
            layers.append(LayerStates(
                attn_cache=cache,
                conv_state=None if layer.conv_state is None
                else layer.conv_state.clone(),
                recurrent_state=None if layer.recurrent_state is None
                else layer.recurrent_state.clone(),
                ple_context=None if layer.ple_context is None
                else layer.ple_context.clone(),
                ple_conv_state=None if layer.ple_conv_state is None
                else layer.ple_conv_state.clone(),
            ))
        return ModelStates(
            layers=layers,
            position_ids=None if self.position_ids is None else self.position_ids.clone())

    def truncate(self, length: int) -> "ModelStates":
        """把每层的 KV 与索引 key 截到 ``length`` 个位置。

        投机校验回滚要用它：一次吃掉 ``K+1`` 行之后只接受 ``a`` 份草稿，主模型的
        缓存必须退回 ``ctx + a + 1``。**线性注意力的递推状态不能这样截**——它不是
        按位置索引的，回滚需要快照。所以这个方法在遇到非 ``None`` 的
        ``conv_state`` / ``recurrent_state`` 时直接拒绝，而不是悄悄只截注意力的
        那一半（那样会得到一个长度对、递推状态多跑了几步的模型）。

        混合架构（本项目就是：每 4 层一个 QSA、其余是 GDN）因此必须用
        :meth:`snapshot` + 整份还，而不是截断。这条拒绝是刻意留的——它把
        「这条路上有推不回去的状态」变成一次明确的调用失败，而不是一个静默错值。

        Args:
            length: 要保留多少个位置。

        Returns:
            新的状态容器（不原地改）。

        Raises:
            ValueError: 长度超出当前缓存，或遇到无法按位置截断的线性注意力状态。
        """
        new_layers: list[LayerStates] = []
        for index, layer in enumerate(self.layers):
            if layer.conv_state is not None or layer.recurrent_state is not None:
                raise ValueError(
                    f"layer {index} carries a linear-attention recurrent state, which "
                    "cannot be truncated by position; take a snapshot before the verify "
                    "step instead of truncating after it.")
            cache = layer.attn_cache
            if cache is not None:
                have = int(cache.keys.shape[2])
                if length < 0 or length > have:
                    raise ValueError(
                        f"cannot truncate layer {index} KV to {length}; it holds {have}.")
                cache = type(cache)(keys=cache.keys[:, :, :length].contiguous(),
                                    values=cache.values[:, :, :length].contiguous(),
                                    index_keys=cache.index_keys[:, :length].contiguous())
            new_layers.append(LayerStates(
                attn_cache=cache,
                conv_state=layer.conv_state,
                recurrent_state=layer.recurrent_state,
                ple_context=layer.ple_context,
                ple_conv_state=layer.ple_conv_state,
            ))
        positions = self.position_ids
        if positions is not None:
            if length > int(positions.shape[-1]):
                raise ValueError(
                    f"cannot truncate the position table to {length}; it holds "
                    f"{positions.shape[-1]}.")
            positions = positions[..., :length]
        return ModelStates(layers=new_layers, position_ids=positions)


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
            quantized: 稠密投影与 embed_tokens 的 bf16 权重不建，改为等待量化产物
                （``proj.set_quantized``）。量化路径必须打开：不打开的话
                ``to_empty`` 会先分配一份马上要丢的 bf16 矩阵，48 层是 9 GiB 量级。
        """
        super().__init__()
        self.config = config
        self.combine = combine
        self.embed_tokens = QuantizedEmbedding(config.vocab_size, config.hidden_size,
                                               config.pad_token_id,
                                               quantized=quantized)
        self.layers = nn.ModuleList(
            [DecoderLayer(config, layer_idx, local_experts=local_experts,
                          stream_ngram=stream_ngram, quantized=quantized)
             for layer_idx in range(config.num_hidden_layers)])
        self.rotary_emb = RotaryEmbedding(config)
        # 末端没有 final RMSNorm，塌缩由这个 use_combine=False 的 mixer 完成。
        self.hyper_connection_mixer = GatedResidual(config, use_combine=False)
        if quantized:
            self.release_quantized_placeholders()

    def release_quantized_placeholders(self) -> int:
        """摘掉本子树里所有稠密投影的 bf16 占位权重，交给量化产物供货。

        只在 ``quantized=True`` 的构造末尾调用一次。层内的投影由 ``DecoderLayer``
        建，而 ``blocks.py`` 不在这次改动的范围里，所以这一步放在模型侧统一走一遍
        ——它要做的事恰好就是「按名字把所有 :class:`Proj` 找出来」，而名字在构造
        完成时已经定型。

        Returns:
            摘掉占位参数的模块个数，供装载日志核对量化接上了几个投影。
        """
        return release_quantized_placeholders(self)

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
                use_cache: bool = True,
                return_multi: bool = False
                ) -> tuple[torch.Tensor, ModelStates] | tuple[torch.Tensor,
                                                              torch.Tensor, ModelStates]:
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
            return_multi: 是否**额外**返回末端 mixer 之前的 4 路残差流。
                MTP 头需要它：草稿器的 ``pre_fc_norm_hidden`` 是 ``[hc*H]`` 宽，
                吃的就是这个未塌缩状态（见 ``mtp.py`` 文件头）。主干路径传默认值
                ``False``，此时返回值与加这个参数之前完全相同。

        Returns:
            ``(hidden_states, new_states)``：``[B, S, hidden_size]`` 与更新后的
            状态。``return_multi`` 为真时是 ``(hidden_states, multi, new_states)``，
            ``multi`` 是 ``[B, S, hc_count * hidden_size]`` 的未塌缩影射。

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

        # 末端 mixer 的入参就是「未塌缩的 4 路残差流」——它自己是
        # ``use_combine=False`` 的 GatedResidual，读门一过 hc 那一路就被加权平均掉，
        # 之后再想还原不可能。所以这里在进 mixer 之前先留一份引用（是引用不是拷贝：
        # 塌缩只读不写）。
        multi = hidden_states
        hidden_states = self.hyper_connection_mixer(hidden_states)
        if return_multi:
            return hidden_states, multi, new_states
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
                 combine: callable | None = None,
                 with_mtp: bool = False) -> None:
        """建文本塔、lm_head，以及（可选的）MTP 草稿头。

        Args:
            config: 文本配置。
            local_experts: 本 rank 持有的全局专家号（EP 分片）；见
                :class:`TextModel`。
            stream_ngram: 不分配 PLE 的 n-gram 表；见 :class:`TextModel`。
            combine: MoE 专家输出的跨 rank 归约；见 :class:`TextModel`。
            quantized: 稠密投影与 embed_tokens 不建 bf16 权重；见
                :class:`TextModel`。``lm_head`` 是 ``[248320, 2560]``，bf16 是 1.27 GB，
                量化后 0.64 GB，而它是每步读得最宽的一个投影。
            with_mtp: 是否建 MTP 草稿头。它额外占一份 **BF16** 权重（约 4.86 GiB
                的 2.607B 参数）与一层 QSA 的 KV——真实 checkpoint 的 ``mtp.*``
                全是 BF16，量化产物里没有它们。所以只有真要跑投机解码时才开；
                开之前 ``config.mtp_num_hidden_layers`` 必须非零（权重里得有那 31 个
                张量），否则就是建了一个永远装不满的空壳。

        Raises:
            ValueError: ``with_mtp`` 为真但配置声明没有 MTP 头。
        """
        super().__init__()
        self.config = config
        self.model = TextModel(config, local_experts=local_experts,
                               quantized=quantized,
                               stream_ngram=stream_ngram, combine=combine)
        self.lm_head = Proj(config.hidden_size, config.vocab_size, quantized=quantized)
        if with_mtp and not config.mtp_num_hidden_layers:
            raise ValueError(
                "with_mtp=True needs config.mtp_num_hidden_layers > 0; this checkpoint "
                "declares no MTP head, so the draft layer would never be loaded.")
        self.mtp = (MTP(config, quantized=quantized, stream_ngram=stream_ngram)
                    if with_mtp else None)
        #: 投机草稿链是否复用 QSA 的选块结果。
        #:
        #: 技术报告说 MTP 复用跨步 top-k 下标；vLLM 把它做成
        #: ``index_share_for_mtp_iteration``，默认关。这里也默认关，因为
        #: ``docs/qwen4/Q4EXP-MTP-SPEC.md`` §6.5 算出它的收益上限是**每 draft 步
        #: 约 1.7 ms（仅长上下文）**——MTP 只有一层，不是主干那 12 层。打开它不该
        #: 改变结果（技术报告 Table 4：接受长度 4.07 对 4.06），所以它是个纯开关，
        #: 用于「复用 vs 不复用」的对拍。
        self.index_reuse = False

    def _mtp_forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                     seq: int, states: "ModelStates | None"
                     ) -> tuple[torch.Tensor, torch.Tensor, "ModelStates"]:
        """主模型前向，取最后 ``seq`` 行的 logits、未塌缩状态与新状态。

        单独成一个方法是为了让 ``spec_generate`` 里两个调用点（prompt 预填充、
        投机校验）走**同一条**路径：分开写两份的话，「logits 取哪几行」与「multi
        取哪几行」这两处切片很容易只在一处改对。

        Args:
            input_ids: ``[1, S]``。
            attention_mask: ``[1, 1, S, past + S]``。
            seq: 要保留多少行的输出（从末尾数）。
            states: 上一回合留下的状态；``None`` 表示从空上下文开始。

        Returns:
            ``(logits, multi, new_states)``：``[1, seq, vocab]``、
            ``[1, seq, hc*H]``、更新后的状态。

        Raises:
            RuntimeError: 模型没建 MTP 头。
            ValueError: ``seq`` 超过本次输入的行数。
        """
        if self.mtp is None:
            raise RuntimeError("_mtp_forward needs the MTP head (with_mtp=True)")
        rows = int(input_ids.shape[1])
        if seq > rows:
            raise ValueError(f"asked for {seq} output rows but the input has {rows}.")
        logits, multi, new_states = self.forward(
            input_ids=input_ids, attention_mask=attention_mask, states=states,
            logits_to_keep=seq, return_multi=True)
        return logits, multi[:, rows - seq:, :], new_states

    def _shifted_embed(self, input_ids: torch.Tensor,
                       next_token: int) -> torch.Tensor:
        """右移一位的嵌入：位置 ``i`` 配 token ``i+1``。

        MTP 头是按「给定位置 ``i`` 的状态，预测 ``i+1``」训练的，所以草稿层在位置
        ``i`` 的 query 要喂 ``i+1`` 的嵌入。最后一行没有「``prompt`` 里的下一个」，
        它要喂的是**主模型刚采样出来的那个 token**——那正是位置 ``len(prompt)`` 上的
        token，而草稿层预填充的最后一行要预测的就是它的下一个。

        填 eos 是个**看起来能用**的错误：形状、掩码、KV 全都正常，只是最后那格的草稿
        永远基于一个错误的嵌入，于是接受长度恒接近 0，而没有任何一处报错。这一条
        只能靠「草稿是不是真的从正确输入算出来的」这类判据发现。

        Args:
            input_ids: ``[1, n]``。
            next_token: 位置 ``n`` 上的那个 token（主模型对 prompt 末尾的 argmax）。

        Returns:
            ``[1, n, H]`` 的嵌入。

        Raises:
            ValueError: ``input_ids`` 不是二维或为空。
        """
        if input_ids.dim() != 2 or input_ids.shape[1] < 1:
            raise ValueError(
                f"shifted embedding needs a non-empty [1, n] token tensor, got "
                f"{tuple(input_ids.shape)}.")
        tail = torch.full((input_ids.shape[0], 1), int(next_token),
                          dtype=torch.long, device=input_ids.device)
        return self.model.embed_tokens(torch.cat([input_ids[:, 1:], tail], dim=1))

    def _pos_emb(self, n: int,
                 past: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """``[0, past + n)`` 这段位置的 ``(cos, sin)``。

        草稿层的 RoPE 与主干同表（``rope_theta`` 都是 1e7、``partial_rotary_factor``
        都是 0.25）。文本场景下三轴位置相同，所以把一维位置扩成三行。

        Args:
            n: 要多少个位置。
            past: 起始位置。

        Returns:
            ``(cos, sin)``，形状 ``[1, n, rotary_dim]``。

        Raises:
            ValueError: ``n`` 非正。
        """
        if n <= 0:
            raise ValueError(f"position table needs n > 0, got {n}.")
        base = torch.arange(past, past + n, dtype=torch.long,
                            device=self.model.embed_tokens.weight.device)
        return self.model.rotary_emb.cos_sin(base.view(1, 1, -1).expand(3, 1, -1))

    def _pos_emb_at(self, position: int) -> tuple[torch.Tensor, torch.Tensor]:
        """:meth:`_pos_emb` 的单位置版本。

        Args:
            position: 绝对位置。

        Returns:
            ``(cos, sin)``，形状 ``[1, 1, rotary_dim]``。

        Raises:
            ValueError: ``position`` 为负。
        """
        return self._pos_emb(1, position)

    def forward(self, input_ids: torch.Tensor | None = None,
                inputs_embeds: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None,
                conv_mask: torch.Tensor | None = None,
                position_ids: torch.Tensor | None = None,
                ple_input_ids: torch.Tensor | None = None,
                states: ModelStates | None = None,
                ple_tables: dict[int, torch.Tensor] | None = None,
                logits_to_keep: int = 0,
                use_cache: bool = True,
                return_multi: bool = False
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
            return_multi: 是否额外返回末端 mixer 之前的 4 路残差流（MTP 要用）。

        Returns:
            ``(logits, new_states)``：``[B, S', vocab_size]``。``return_multi`` 为真
            时是 ``(logits, multi, new_states)``。
        """
        if return_multi:
            hidden_states, multi, new_states = self.model(
                input_ids=input_ids, inputs_embeds=inputs_embeds,
                attention_mask=attention_mask, conv_mask=conv_mask,
                position_ids=position_ids, ple_input_ids=ple_input_ids,
                states=states, ple_tables=ple_tables, use_cache=use_cache,
                return_multi=True)
            slice_indices = slice(-logits_to_keep, None) if logits_to_keep else slice(None)
            return self.lm_head(hidden_states[:, slice_indices, :]), multi, new_states
        hidden_states, new_states = self.model(
            input_ids=input_ids, inputs_embeds=inputs_embeds,
            attention_mask=attention_mask, conv_mask=conv_mask,
            position_ids=position_ids, ple_input_ids=ple_input_ids,
            states=states, ple_tables=ple_tables, use_cache=use_cache)
        slice_indices = slice(-logits_to_keep, None) if logits_to_keep else slice(None)
        return self.lm_head(hidden_states[:, slice_indices, :]), new_states

    def spec_generate(self, prompt: list[int], max_new: int, K: int = 3,
                      eos: int | None = None, temperature: float = 0.0
                      ) -> tuple[list[int], dict]:
        """贪心投机解码：MTP 草拟 ``K`` 个，主模型一次校验 ``K+1`` 个。

        这是**单请求**路径，与 ``engine/`` 的批调度无关；它的用途是把 MTP 的数学
        与状态契约验到位，并给出接受长度。批处理、分页与前缀复用要另外接。

        一次回合的形状：

        * 草稿：``h_prev`` 是主模型最后一步吐的**未塌缩**残差流，``tok`` 是它的下一个
          token。链上 K 步，每步 argmax 得到下一个草稿 token，``multi`` 喂下一步。
        * 校验：把 ``[tok, d1, ..., dK]`` 整段喂给主模型（S=K+1），从每个位置取
          argmax，与草稿逐位比。第一个不一致处用主模型的 argmax 替换并停止接受。

        链式 argmax 全程留在设备上：``draft_chain`` 只把 token 写进设备缓冲，不读回
        主机。K 份草稿因此没有 K 次同步——这是 ``models/qwen35/mtp.py`` 文件头里那条
        约束的同一条。

        Args:
            prompt: 已 tokenize 的 prompt，至少 1 个 token。
            max_new: 最多生成多少个 token。
            K: 每回合草拟几个。
            eos: 结束 token；遇到即停。
            temperature: 参与校验的主模型采样温度。``0`` 是纯贪心，也是唯一能与
                「逐位接受」判据对齐的档——非零温度下草稿与校验都得走概率拒绝采样，
                那是另一套算术，本条路径不提供。

        Returns:
            ``(tokens, stats)``：``tokens`` 是生成出来的 token（不含 prompt），
            ``stats`` 含 ``rounds`` / ``drafted`` / ``accepted`` / ``acceptance``
            四个计数。

        Raises:
            RuntimeError: 模型没建 MTP 头。
            ValueError: ``prompt`` 为空、``K < 1``，或 ``temperature`` 非零。
        """
        if self.mtp is None:
            raise RuntimeError("spec_generate() needs the MTP head (with_mtp=True)")
        if not prompt:
            raise ValueError("prompt must not be empty.")
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}.")
        if temperature != 0:
            raise ValueError(
                "spec_generate() implements greedy verification only; sampling needs the "
                f"rejection sampler, got temperature={temperature}.")

        device = self.lm_head.weight.device if self.lm_head.weight is not None \
            else next(self.parameters()).device
        prompt_ids = torch.tensor([prompt], dtype=torch.long, device=device)
        n_prompt = len(prompt)
        logits, multi, states = self._mtp_forward(
            prompt_ids, causal_mask(1, n_prompt, 0), n_prompt, states=None)
        nxt = int(logits[0, -1].argmax())
        out: list[int] = [nxt]
        stats = {"rounds": 0, "drafted": 0, "accepted": 0, "acceptance": 0.0,
                 "index_reuse": 0}
        # 草稿层的 KV 每回合**从头重建**，不做增量维护。
        #
        # 为什么放弃增量：草稿池的高度、第 0 份草稿来自哪一格、回滚后要补几格——
        # 这三件事互相耦合，而每错一格都只表现为「接受率慢慢掉」，不会有一处报错。
        # 重建的代价是每回合一次 ``[len(ctx), hc*H]`` 的草稿层前向，而草稿链本身
        # 已经要跑 K 次同样的前向，所以它与已有成本同量级；换来的是**没有第二套
        # 不变量**。增量要等这个基线站住之后再谈。
        #
        # 重建用**右移一位**的嵌入：位置 ``i`` 的 query 拿 token ``i+1`` 的嵌入，
        # 末尾补 ``out[-1]``。预填充的末行输出即位置 ``len(full)-1`` 的草稿。
        if eos is not None and nxt == eos:
            return out, stats
        while len(out) < max_new:
            stats["rounds"] += 1
            plan = DraftIndexPlan(enabled=self.index_reuse)
            drafts = torch.zeros(K, dtype=torch.long, device=device)
            # 本回合的唯一事实来源：``full = prompt + out[:-1]`` 整串重算主模型。
            #
            # ``out[-1]`` 不在 ``full`` 里，因为它是「已采样、还没校验落库」的那个
            # token —— 校验步吃 ``[out[-1], drafts...]``，所以它必须是第一个 query。
            #
            # 草稿层用的是**同一个**位置网格（``full`` 的 ``len(full)`` 格），它的
            # 位置 ``i`` 的 query 拿 token ``full[i+1]`` 的嵌入、末尾补 ``out[-1]``。
            # 两层的 KV 因此从同一个起点、同一组位置上长出来，不会再出现「草稿层比
            # 主模型多一格/少一格」这类只有接受率才看得出的错位。
            #
            # 这与 vLLM 的 ``_update_positions_dependent_metadata`` 是同一件事的两种
            # 写法：上游靠增量推进 KV 元数据，coffin 没有分页池，就直接把整串重算。
            full = list(prompt) + out[:-1]
            full_ids = torch.tensor([full], dtype=torch.long, device=device)
            _lg, multi_all, states = self._mtp_forward(
                full_ids, causal_mask(1, len(full), 0), len(full), states=None)
            pending_sample, draft_cache = self.mtp.prefill(
                self._shifted_embed(full_ids, out[-1]),
                multi_all, self._pos_emb(len(full)),
                causal_mask(1, len(full), 0))
            h_multi = multi_all[:, -1:, :]
            tok = out[-1]
            for step in range(K):
                if step == 0:
                    sample = pending_sample[:, -1:, :]
                else:
                    pool_len = len(full) + (step - 1)
                    sample, h_multi, draft_cache = self.mtp.run(
                        self.model.embed_tokens(torch.tensor(
                            [[tok]], dtype=torch.long, device=device)),
                        h_multi, self._pos_emb(pool_len + 1),
                        causal_mask(1, 1, pool_len), cache=draft_cache, step=step,
                        index_plan=plan)
                tok = int(self.lm_head(sample)[0, -1].argmax())
                drafts[step] = tok
            stats["drafted"] += K
            stats["index_reuse"] += plan.used
            # 校验：主模型一次吃 K+1 行 ``[out[-1], drafts...]``，从 ``ctx_len`` 起。
            # ``ctx_len`` 是 ``out[-1]`` 的位置，也就是它之前已落库的 token 数。
            ctx_len = n_prompt + len(out) - 1
            snapshot = states.snapshot()
            verify_ids = torch.cat([torch.tensor([out[-1]], dtype=torch.long,
                                                 device=device), drafts])
            logits, _multi_v, _states_v = self._mtp_forward(
                verify_ids.view(1, -1), causal_mask(1, K + 1, ctx_len), K + 1,
                states=states)
            picked = logits[0].argmax(dim=-1)
            accepted = 0
            for i in range(K):
                if int(picked[i]) != int(drafts[i]):
                    break
                accepted += 1
            # ``picked[accepted]`` 是「接受 accepted 份草稿之后本该出现的下一个
            # token」：全接受时它是 bonus token，中途失配时它是替换 token。
            new_tokens = [int(drafts[i]) for i in range(accepted)] + [int(picked[accepted])]
            out.extend(new_tokens)
            stats["accepted"] += accepted
            if eos is not None and eos in new_tokens:
                break
            if len(out) >= max_new:
                break
        stats["acceptance"] = (stats["accepted"] / stats["drafted"]
                               if stats["drafted"] else 0.0)
        return out[:max_new], stats


def causal_mask(batch: int, seq: int, past: int) -> torch.Tensor:
    """建 4 维因果掩码，``[B, 1, S, past + S]``。

    Args:
        batch: batch 大小。
        seq: 本次 chunk 的 query 数。
        past: 已缓存的 token 数（本 chunk 的 query 全部在它们之后）。

    Returns:
        bool 掩码，``True`` 表示可见。

    Raises:
        ValueError: ``batch`` 或 ``seq`` 非正。
    """
    if batch <= 0 or seq <= 0:
        raise ValueError(f"mask needs positive batch/seq, got {batch} and {seq}.")
    rows = torch.arange(past, past + seq).unsqueeze(1)
    cols = torch.arange(past + seq).unsqueeze(0)
    return (cols <= rows).expand(batch, -1, -1).unsqueeze(1)


__all__ = ["CausalLM", "LayerStates", "ModelStates", "TextModel", "causal_mask"]