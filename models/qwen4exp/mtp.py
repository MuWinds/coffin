"""MTP 头：本族的投机解码草稿器（一层 QSA + MoE，共享词表嵌入与 lm_head）。

参考实现（``docs/qwen4/modeling_qwen4_exp.py``）**没有实现 MTP**——它只把 ``mtp.*``
列进 ``_keys_to_ignore_on_load_unexpected``。所以这一份的规格不是从参考实现抄的，
而是从两处反推，逐条记在 ``docs/qwen4/Q4EXP-MTP-SPEC.md``：

* **真实 checkpoint 的 31 个 ``mtp.*`` 张量形状**（合计 2607150848 个 BF16 参数，
  已用集群 safetensors header 逐条核对）；
* **vLLM 官方 qwen4_exp MTP 源码** ``vllm/models/qwen4_exp/nvidia/mtp.py`` 的
  ``Qwen4ExpMultiTokenPredictor``。

形状为什么能定死结构：

    mtp.fc_embedding.weight            [2560, 2560]   嵌入那一路的投影
    mtp.fc_hidden.weight               [2560, 2560]   4 路残差流那一路的投影
    mtp.pre_fc_norm_embedding.weight   [2560]         作用在嵌入上
    mtp.pre_fc_norm_hidden.weight      [10240]        **10240 = hc_count*H**，
                                                      作用在未塌缩的 4 路流上
    mtp.hyper_connection_mixer.*       3 项           末端收口，use_combine=False
                                                      （没有 block_inject_weight）
    mtp.layers.0.*                     完整 QSA 层    含 indexer、MoE、两套超连接

``fc_embedding`` 与 ``fc_hidden`` 都是 ``[H, H]``、``pre_fc_norm_hidden`` 是
``[10240]``——这三条一起排除了「两路拼接成 ``[2H, H]``」的读法（那是 Qwen3.5 MTP
的 ``mtp.fc``）。vLLM 的写法印证：``fc_hidden`` 作用在 ``[T, hc, H]`` 的最后一维，
``fc_embedding`` 的输出作为 ``prev_block_output`` 以**单位权重**并进 4 路流。

Module: ``coffin/models/qwen4exp``。依赖同族的 ``blocks``（复用 ``DecoderLayer``）、
``gated_residual``、``norm``、``proj``、``qsa``；被 ``model.CausalLM`` 持有。

Constraints:
    * **不带 PLE。** 第 2 层才有 PLE，而 MTP 层用的是 ``num_hidden_layers`` 号，本来
      也不命中——但那是巧合，不是契约，所以显式关掉。真实 PLE 表是 95 GiB 的 n-gram
      嵌入，草稿步每步查它是纯浪费，vLLM 也明确写了 MTP "forces PLE off"。
    * **每步吐两路。** ``run`` 返回 ``(sample, multi, cache)``：``sample`` 是末端
      mixer 塌缩后的 ``[B,S,H]``（喂 lm_head），``multi`` 是**未塌缩**的
      ``[B,S,hc*H]``（喂下一步的 ``hidden_projection``）。漏掉 ``multi`` 就只能每步
      从 token id 重新起步，链就断了。
    * **嵌入残差只在「首步」加。** ``prev_block_output`` 只在 ``add_embedding=True``
      时并进残差流；草稿链的步 1+ 必须传 ``False``。每步都加会把嵌入重复叠加进
      残差流——形状全对、数值全错。
    * **KV 与索引 key 单独一块池。** 草稿层有自己的 ``k_proj`` / ``v_proj`` /
      ``o_proj`` 与 ``indexer.index_qk_proj``，与主干任何一层都不共享（31 个张量里
      没有一处共享），所以主干的 KV 不能被它复用。
    * **K/V 与索引 key 必须与主干同步截断。** 校验回滚、前缀复用命中时，草稿池的
      位置必须和主干 KV 停在同一个点上，否则草稿会对着一段长度不对的 KV 起跑。
      见 :meth:`MTP.truncate`。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from models.qwen4exp.blocks import DecoderLayer
from models.qwen4exp.config import TextConfig
from models.qwen4exp.gated_residual import GatedResidual
from models.qwen4exp.norm import RMSNorm
from models.qwen4exp.proj import Proj
from models.qwen4exp.qsa import AttentionCache, IndexReuse


@dataclass
class DraftIndexPlan:
    """一条草稿链上的 QSA 选块复用簿记。

    技术报告 §「Implementation Details」：*"multi-step MTP reuses top-k indices
    across prediction steps to further reduce draft-model inference costs"*，且明确
    说这是照 GLM 的做法。vLLM 在 ``Qwen4ExpMultiTokenPredictor`` 上给了两个方法：
    ``set_skip_topk(skip)``（*"Select on MTP step 0 and reuse its QSA indices on
    later steps."*）与 ``compact_topk_indices(row_indices)``（*"Keep each request's
    target-aligned step-0 sparse-index row."*）。

    coffin 的复用**只复用块号**，不复用整份掩码——理由写在 :class:`~qsa.IndexReuse`
    上，核心两条本仓事实：掩码宽度随 ``kv_length`` 变；而
    ``kv_length <= 2048`` 时选块有一条整段快路径、结果就等于可见掩码本身。

    Attributes:
        shared: 本轮要复用的块表；``None`` 表示还没产出（步 0 之前）。
        enabled: 是否启用复用。关掉时步 1+ 也自己选块，用于「复用 vs 不复用」的对拍
            ——技术报告 Table 4 的结论是两者平均接受长度无显著差异（4.07 对 4.06），
            所以这条开关必须存在，否则无法验证那个结论在本实现上成立。
        used: 本次复用实际生效了几次。短上下文下步 1+ 会被整段快路径接管，那个
            计数器会停在 0——这也正是它必须被记下来的原因，而不是假设「开了就在用」。
    """

    shared: IndexReuse | None = None
    enabled: bool = True
    used: int = 0

    def before_step(self, step: int) -> IndexReuse | None:
        """这一步该不该复用；该复用时返回块表。

        Args:
            step: 从 0 开始的 draft 步号。

        Returns:
            步 0、或关闭复用、或还没产出块表时返回 ``None``（自己选）；其余返回
            :attr:`shared`。

        Raises:
            ValueError: 步号是负的。负步号说明调用方把「已经推进几步」记错了，
                静默按步 0 处理会让整轮都退化成不复用而看不出来。
        """
        if step < 0:
            raise ValueError(f"draft step index must be >= 0, got {step}.")
        if not self.enabled or step == 0 or self.shared is None:
            return None
        self.used += 1
        return self.shared

    def record(self, produced: list) -> None:
        """把步 0 产出的块表留下来。

        Args:
            produced: ``DecoderLayer`` 通过 ``index_out`` 填出来的列表；空列表表示
                那一步走了整段快路径（短上下文），没有块表可复用——此时保持
                :attr:`shared` 为 ``None``，后续步继续自己选，结果同样正确。

        Raises:
            ValueError: 列表里有多于一项。MTP 只有一层，一层只 append 一次；多于
                一项说明调用方把主干层的输出也塞了进来。
        """
        if len(produced) > 1:
            raise ValueError(
                f"one MTP layer produced {len(produced)} index plans; only one is expected.")
        self.shared = produced[0] if produced else None


class MTP(nn.Module):
    """一层 QSA 草稿器：给它主模型的 4 路残差流与下一个 token，它预测再下一个。

    Attributes:
        config: 文本配置。
        fc_embedding: ``[H, H]``，投影（已归一化的）token 嵌入。
        fc_hidden: ``[H, H]``，投影 4 路残差流；作用在最后一维上，各分支共享。
        pre_fc_norm_embedding: 零中心 RMSNorm，宽度 ``H``。
        pre_fc_norm_hidden: 零中心 RMSNorm，宽度 ``hc_count * H``。
        layers: 恰好一个 :class:`~blocks.DecoderLayer`；类型取
            ``config.mtp_layer_type``。包成 ``ModuleList`` 是为了键名与 checkpoint 的
            ``mtp.layers.0.*`` 逐字相同。
        hyper_connection_mixer: 末端收口，``use_combine=False``。
    """

    def __init__(self, config: TextConfig, quantized: bool = False,
                 stream_ngram: bool = False) -> None:
        """建草稿器。

        Args:
            config: 文本配置；``mtp_num_hidden_layers`` 必须是 1。
            quantized: 稠密投影是否等量化产物供货（``proj.Proj`` 的语义）。
                注意真实 checkpoint 的 ``mtp.*`` **全是 BF16**，量化产物里没有它们；
                所以这个开关在 MTP 上通常传 ``False``。
            stream_ngram: 透传给 ``DecoderLayer``。草稿层不带 PLE，这个开关在它上面
                没有效果；留着是为了调用点不必分两套签名。

        Raises:
            ValueError: ``mtp_num_hidden_layers`` 不是 1。大于 1 时每步用的是不同的
                层，而 :meth:`run` 只驱动一层；静默按第 0 层跑会让第 2..N 层的权重
                永远不参与前向（装载校验能过、答案错）。
        """
        super().__init__()
        if config.mtp_num_hidden_layers != 1:
            raise ValueError(
                "qwen4exp MTP implements exactly one draft layer; checkpoint declares "
                f"mtp_num_hidden_layers={config.mtp_num_hidden_layers}.")
        self.config = config
        hidden = config.hidden_size
        hc_hidden = config.hc_hidden_size
        self.fc_embedding = Proj(hidden, hidden, quantized=quantized)
        self.fc_hidden = Proj(hidden, hidden, quantized=quantized)
        # 两处 pre-norm 的宽度不同是这条规格里最容易看错的一处：嵌入那一路是
        # ``[H]``，残差流那一路是 ``[hc*H]``——后者正是「喂进去的是未塌缩的 4 路流」
        # 这件事唯一的形状证据。
        self.pre_fc_norm_embedding = RMSNorm(hidden, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = RMSNorm(hc_hidden, eps=config.rms_norm_eps)
        # 层号取 ``num_hidden_layers``：vLLM 的 ``mtp_start_layer_idx`` 就是这么定的，
        # 它让这层在权重名字上落在主干之外（``mtp.layers.0`` 而不是
        # ``model.layers.48``）。``layer_type`` 显式传，理由见 ``DecoderLayer``。
        #
        # 包成 ``ModuleList`` 而不是直接赋 ``self.layer``：checkpoint 的键是
        # ``mtp.layers.0.*``（复数、带下标），而 ``self.layer = DecoderLayer(...)``
        # 只产出 ``mtp.layer.*``。差这一个 ``s.0`` 就是那 20 多个张量落不到模块上
        # ——装载器会报「模型参数没有源张量」，而那句话读起来像「权重少了」。
        # 层数恒为 1（上面已校验），所以下标只可能是 0。
        self.layers = nn.ModuleList([DecoderLayer(
            config, config.num_hidden_layers,
            quantized=quantized,
            stream_ngram=stream_ngram,
            layer_type=config.mtp_layer_type,
            with_ple=False,
        )])
        self.hyper_connection_mixer = GatedResidual(config, use_combine=False)

    def embed_projection(self, token_embed: torch.Tensor) -> torch.Tensor:
        """把 token 嵌入投影成块输出候选，``[B, S, H]``。

        Args:
            token_embed: 查表得到的嵌入，``[B, S, H]``。

        Returns:
            投影后的嵌入。它作为 ``prev_block_output`` 以单位权重并进 4 路残差流。

        Raises:
            ValueError: 最后一维不是 ``hidden_size``。
        """
        if token_embed.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"token embedding must be {self.config.hidden_size} wide, got "
                f"{tuple(token_embed.shape)}.")
        return self.fc_embedding(self.pre_fc_norm_embedding(token_embed))

    def hidden_projection(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """把 4 路残差流投影成草稿层的输入，``[B, S, hc*H]``。

        ``pre_fc_norm_hidden`` 是**按整条 10240 维**做一次 RMSNorm（不是按 2560
        分组），所以先 flatten 再归一化；``fc_hidden`` 只作用在最后一维上，因此把
        ``[B,S,hc*H]`` 拆回 ``[B,S,hc,H]``、投影完再拼回去。逐个分支各调一次
        ``[H,H]`` 在数值上完全等价，但那是 hc 倍的调用次数。

        Args:
            hidden_states: 未塌缩的 4 路残差流，``[B, S, hc_count * H]``。

        Returns:
            投影后的 ``[B, S, hc_count * H]``。

        Raises:
            ValueError: 最后一维不是 ``hc_count * hidden_size``。
        """
        hc, hidden = self.config.hc_count, self.config.hidden_size
        if hidden_states.shape[-1] != hc * hidden:
            raise ValueError(
                f"residual stream must be {hc * hidden} wide, got "
                f"{tuple(hidden_states.shape)}.")
        lead = hidden_states.shape[:-1]
        normed = self.pre_fc_norm_hidden(hidden_states.reshape(-1, hc * hidden))
        branches = normed.view(*lead, hc, hidden)
        return self.fc_hidden(branches).reshape(*lead, hc * hidden)

    def run(self, token_embed: torch.Tensor, hidden_states: torch.Tensor,
            position_embeddings, attention_mask: torch.Tensor,
            cache: AttentionCache | None = None,
            step: int = 0, add_embedding: bool | None = None,
            index_plan: DraftIndexPlan | None = None,
            combine: callable | None = None
            ) -> tuple[torch.Tensor, torch.Tensor, AttentionCache]:
        """跑 ``S`` 行草稿层，返回 ``(sample, multi, cache)``。

        这是唯一的执行入口：预填充（``S = n``，只关心把 KV 建起来）与草稿单步
        （``S = 1``，只关心输出）都走它。分开写两份会让「KV 建对了但输出算错了」
        这类问题在两条路之间摇摆。

        Args:
            token_embed: 本步输入 token 的嵌入，``[B, S, H]``。
            hidden_states: 4 路残差流，``[B, S, hc*H]``。首步来自主模型的**未塌缩**
                输出；后续步传上一步返回的 ``multi``。
            position_embeddings: ``(cos, sin)``，最后 ``S`` 行对应本次的位置。
            attention_mask: 4 维因果掩码，``[B, 1, S, kv_length]``。
            cache: 草稿层上一轮留下的 KV 与索引 key 池；``None`` 表示从空上下文开始。
            step: 从 0 开始的 draft 步号，决定要不要复用选块。
            add_embedding: 是否把投影后的嵌入以单位权重并进 4 路流。``None`` 表示
                按 ``step == 0`` 决定——预填充时显式传 ``True``（每一行都是「首步」），
                草稿步 1+ 传 ``False``。
            index_plan: 本轮投机的选块复用簿记；``None`` 表示每步都自己选。
            combine: MoE 的跨 rank 归约；``None`` 表示单卡。

        Returns:
            ``(sample, multi, cache)``：``sample`` 是 ``[B,S,H]`` 的塌缩输出（喂
            lm_head 出 logits），``multi`` 是 ``[B,S,hc*H]`` 的未塌缩状态（喂下一步的
            :meth:`hidden_projection`），``cache`` 是追加了本次 ``S`` 个位置之后的新缓存。

        Raises:
            ValueError: ``attention_mask`` 不是 4 维。3 维掩码在 QSA 里会广播成一团
                形状合法的垃圾。
        """
        if attention_mask.dim() != 4:
            raise ValueError(
                f"MTP needs a 4-D attention mask, got {tuple(attention_mask.shape)}.")
        if add_embedding is None:
            add_embedding = step == 0
        block_output = self.embed_projection(token_embed)
        projected = self.hidden_projection(hidden_states)
        reuse = None if index_plan is None else index_plan.before_step(step)
        produced: list | None = ([] if index_plan is not None and reuse is None else None)
        hidden_states, state, pending = self.layers[0](
            projected,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            attention_cache=cache,
            combine=combine,
            prev_block_output=block_output if add_embedding else None,
            return_block=True,
            index_plan=reuse,
            index_out=produced,
        )
        if index_plan is not None and produced is not None:
            index_plan.record(produced)
        sample, multi = self.hyper_connection_mixer.combine_and_mix(
            pending["hyper_input"], pending["block_output"],
            pending["injection_weights"])
        return sample, multi, state["attn_cache"]

    def prefill(self, token_embed: torch.Tensor, hidden_states: torch.Tensor,
                position_embeddings, attention_mask: torch.Tensor,
                cache: AttentionCache | None = None,
                combine: callable | None = None
                ) -> tuple[torch.Tensor, AttentionCache]:
        """用主模型在整段 prompt 上的未塌缩状态建起草稿层的 KV。

        草稿层在位置 ``p`` 的 query 要看 ``[0, p]`` 上**它自己**算出来的 K/V，而那些
        K/V 只能由草稿层自己从 prompt 每一行的状态算出来——所以每段 prompt 都要过
        一次这个头。这是 MTP 的固定成本，没有便宜做法。

        **最后一行已经是第一份草稿**，所以这里把 ``sample`` 一并交出去：位置
        ``n-1`` 的 query 拿的是「主模型在最后一个 prompt 位置的隐含状态 + 右移一位后
        埋在那里的那个 token 的嵌入」，预测的正是那个 token 之后的下一个。调用方
        因此不必再为 step 0 单独前向一次——再跑一次会把同一个位置写两遍。

        Args:
            token_embed: prompt 各位置的 token 嵌入，``[B, n, H]``。**已经右移一位**
                ——第 ``i`` 行配的应该是第 ``i+1`` 个 token 的嵌入（给定位置 ``i`` 的
                状态，预测 ``i+1``）。移位由调用方做，这里不再移第二次。
            hidden_states: 主模型在 prompt 各位置的未塌缩状态，``[B, n, hc*H]``。
            position_embeddings: ``(cos, sin)``，前 ``n`` 行对应 prompt 的位置。
            attention_mask: 4 维因果掩码，``[B, 1, n, n]``。
            cache: 已有的草稿缓存；``None`` 表示从空上下文开始。
            combine: MoE 的跨 rank 归约。

        Returns:
            ``(sample, cache)``：``sample`` 是 ``[B, n, H]`` 的塌缩输出（**最后一行**
            就是第一份草稿的 logits 来源，前 ``n-1`` 行只是前向的副产品），
            ``cache`` 是追加了整段 prompt 之后的草稿缓存。

        Raises:
            ValueError: 两个入参的 batch/序列长度不一致。错位会让草稿层把位置 ``i``
                的 query 配到位置 ``j`` 的 KV 上，形状全对。
        """
        if token_embed.shape[:2] != hidden_states.shape[:2]:
            raise ValueError(
                "MTP prefill token embedding and hidden states disagree on shape: "
                f"{tuple(token_embed.shape)} vs {tuple(hidden_states.shape)}.")
        sample, _multi, cache = self.run(
            token_embed, hidden_states, position_embeddings, attention_mask,
            cache=cache, step=0, add_embedding=True, combine=combine)
        return sample, cache

    def append(self, token_embed: torch.Tensor, hidden_states: torch.Tensor,
               position_embeddings, attention_mask: torch.Tensor,
               cache: AttentionCache, combine: callable | None = None
               ) -> tuple[torch.Tensor, AttentionCache]:
        """往已有的草稿缓存尾部追加 ``S`` 个位置，返回末端输出与新缓存。

        这是「一轮投机结束后把接受下来的 token 补进草稿层」的那一步。它和
        :meth:`prefill` 的算术完全相同，区别只在入参是**一段增量**而不是整段
        prompt：``attention_mask`` 覆盖 ``[0, 已有长度 + S)``，而 ``cache`` 里已有前
        半段。两者分开命名是为了让调用点上的意图清楚——预填充与增量追加接错，症状
        是草稿层的上下文比主模型短一截，而接受长度会慢慢掉下去。

        Args:
            token_embed: ``[B, S, H]``，**已经右移一位**的嵌入。
            hidden_states: ``[B, S, hc*H]``，主模型在这 ``S`` 个位置上的未塌缩状态。
            position_embeddings: ``(cos, sin)``，覆盖 ``[0, 已有长度 + S)``。
            attention_mask: ``[B, 1, S, 已有长度 + S]``。
            cache: 已有的草稿缓存，不能是 ``None``——从空上下文开始的场景应该走
                :meth:`prefill`。
            combine: MoE 的跨 rank 归约。

        Returns:
            ``(sample, cache)``：``sample`` 是 ``[B, S, H]`` 的塌缩输出（末行就是末端
            位置的草稿），``cache`` 是追加之后的缓存。

        Raises:
            ValueError: ``cache`` 是 ``None``，或三个入参的长度不一致。前者应该走
                :meth:`prefill`；后者会让草稿层把位置配错。
        """
        if cache is None:
            raise ValueError("MTP.append() needs an existing cache; use prefill() to start.")
        if token_embed.shape[:2] != hidden_states.shape[:2]:
            raise ValueError(
                "MTP append token embedding and hidden states disagree on shape: "
                f"{tuple(token_embed.shape)} vs {tuple(hidden_states.shape)}.")
        sample, _multi, cache = self.run(
            token_embed, hidden_states, position_embeddings, attention_mask,
            cache=cache, add_embedding=True, combine=combine)
        return sample, cache

    @staticmethod
    def truncate(cache: AttentionCache | None, length: int) -> AttentionCache | None:
        """把草稿缓存截到 ``length`` 个位置。

        投机校验回滚与前缀复用命中都靠它。草稿池按位置索引，截断就是完整回滚——
        没有别的状态（不像 GDN 有递推状态要恢复）。

        Args:
            cache: 要截的缓存；``None`` 原样返回。
            length: 保留多少个位置。

        Returns:
            新的缓存对象，或 ``None``。

        Raises:
            ValueError: ``length`` 超过当前长度，或为负。前者说明调用方把「接受了几
                个」算错了，静默放大到当前长度会让草稿看到一段没被校验过的 KV。
        """
        if cache is None:
            return None
        have = int(cache.keys.shape[2])
        if length < 0 or length > have:
            raise ValueError(
                f"cannot truncate the draft cache to {length}; it holds {have} positions.")
        return AttentionCache(
            keys=cache.keys[:, :, :length].contiguous(),
            values=cache.values[:, :, :length].contiguous(),
            index_keys=cache.index_keys[:, :length].contiguous(),
        )


def mtp_parameter_names(prefix: str = "") -> tuple[str, ...]:
    """MTP 子树里的两个顶点参数名，默认**相对于 ``MTP`` 自己**。

    装载器与校验都用它来确认「这个 checkpoint 到底有没有 MTP 头」，所以它必须是
    常量而不是从模块实例上现算——建模型之前就要能回答这个问题。

    ``prefix`` 默认空串而不是 ``"mtp."``：这些名字取自 ``MTP.state_dict()``，而那个
    dict 的键本来就相对模块自己。要拿引擎侧的全名就传 ``"mtp."``——两个方向都得对，
    所以 ``prefix`` 是参数、不是写死的前缀拼接。

    Args:
        prefix: 要加在前面的前缀。

    Returns:
        一组参数名；只要源里一个都找不到就说明没有 MTP 头。

    Raises:
        ValueError: ``prefix`` 不以 ``mtp`` 开头。那样拼出来的名字不会落在引擎的
            MTP 子树上，而调用方看起来「查过了」。
    """
    if prefix and not prefix.startswith("mtp"):
        raise ValueError(
            f"prefix {prefix!r} does not point into the MTP subtree; expected it to "
            "start with 'mtp'.")
    return (f"{prefix}fc_embedding.weight", f"{prefix}fc_hidden.weight")


__all__ = ["DraftIndexPlan", "MTP", "mtp_parameter_names"]
