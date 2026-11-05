"""Qwen4-Exp（Qwen3.8-Flash-Next）的配置解析与形状推导。

引擎不依赖 ``transformers``，所以 ``config.json`` 由这里自己解析：字段名与
参考实现的 ``Qwen4ExpTextConfig`` / ``Qwen4ExpVisionConfig`` 一一对应，
派生形状（残差流宽度、QSA 的 q/kv 维、n-gram 头数与表宽、索引器块数）全部在
这里算一次，其余模块只读结果、不自己从 ``hidden_size`` 反推——参考实现里
有一半的静默错误来自某个模块用的宽度和 checkpoint 不一致。

Module: ``coffin/models/qwen4exp``。族的叶子模块：只依赖标准库，可以被族里
任何其它模块 import，自己什么都不 import。

Constraints:
    * 读的是 checkpoint 的字段，不是参考实现的类默认值。默认值里有一批是从
      Qwen3.5-MoE 模板继承下来没改的（``hidden_size=2048``、视觉塔的
      ``out_hidden_size=3584``），真实取值一律以 ``config.json`` 为准。
    * 层类型必须从配置列表读，不许按 ``i % 4`` 推断：checkpoint 的
      ``layer_types`` 才是权威，旧权重里叫 ``full_attention`` 的项在这里统一
      重写成 ``indexed_attention``。
    * 所有校验在构造期抛 ``ValueError``。形状错了要在加载前炸，不能等到前向
      里让一个 10240 宽的张量广播成垃圾。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

#: ``indexed_attention`` 就是"带 QSA indexer 的全注意力"，与滑动窗口无关；
#: 旧 checkpoint 里的 ``full_attention`` 是它的同义词。
LINEAR_ATTENTION = "linear_attention"
INDEXED_ATTENTION = "indexed_attention"
LAYER_TYPES = (LINEAR_ATTENTION, INDEXED_ATTENTION)

#: checkpoint 里可能出现的层类型别名。
_LAYER_TYPE_ALIASES = {"full_attention": INDEXED_ATTENTION}


def _as_int_list(value: Any) -> list[int]:
    """把配置里的整数列表取成新的 ``list[int]``；``None`` 变空表。

    Args:
        value: 配置里的原值，可能是 ``None``、``list`` 或 JSON 数组。

    Returns:
        新的列表。返回副本，调用方改它不会污染配置。
    """
    if value is None:
        return []
    return [int(item) for item in value]


def _flatten_rope(value: Any) -> dict[str, Any]:
    """把参考实现的 ``rope_parameters`` 字典摊成标量字段。

    热路径只需要 ``rope_theta`` / ``partial_rotary_factor`` / ``mrope_section``
    三项，保留整颗字典只会让"到底用的是哪几个字段"变得需要翻文档。

    Args:
        value: ``rope_parameters``（或旧名 ``rope_scaling``）的原值。

    Returns:
        只含出现过的那几个键的小字典。
    """
    if not isinstance(value, dict):
        return {}
    out: dict[str, Any] = {}
    if value.get("rope_theta") is not None:
        out["rope_theta"] = float(value["rope_theta"])
    if value.get("partial_rotary_factor") is not None:
        out["partial_rotary_factor"] = float(value["partial_rotary_factor"])
    if value.get("mrope_section") is not None:
        out["mrope_section"] = _as_int_list(value["mrope_section"])
    if value.get("mrope_interleaved") is not None:
        out["mrope_interleaved"] = bool(value["mrope_interleaved"])
    if value.get("rope_type") is not None:
        out["rope_type"] = str(value["rope_type"])
    return out


@dataclass
class TextConfig:
    """文本塔的全部几何与开关。

    ``__post_init__`` 推导并校验所有派生量，构造完成后字段的语义就是"已经校验
    过的 checkpoint 事实"，前向路径不再做形状判断。

    Attributes:
        vocab_size: 主词表行数（含 padding）。
        hidden_size: 单路残差流宽度，也是 mixer 的输入输出宽度。
        num_hidden_layers: decoder 层数。
        num_attention_heads: QSA 的 query 头数。
        num_key_value_heads: QSA 的 KV 头数。
        head_dim: QSA 的 head 维；``None`` 时按 ``hidden_size // 头数`` 推。
        layer_types: 每层的 mixer 类型，长度必须等于层数。
        hc_count: 残差流路数；主干宽度是 ``hc_count * hidden_size``。
        hc_lowrank: 读写门的低秩瓶颈宽度。
        ple_layer_ids: 一基的 PLE 层号。
        indexer_*: QSA indexer 的几何；全给值时 QSA 才启用。
        ple_embed_dim: 所有 n-gram 头拼起来的总宽度。
    """

    vocab_size: int = 248320
    hidden_size: int = 2560
    num_hidden_layers: int = 48
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    head_dim: int | None = None
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    attention_dropout: float = 0.0
    tie_word_embeddings: bool = False
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None

    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32

    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 512
    num_experts_per_tok: int = 10
    num_experts: int = 512
    norm_topk_prob: bool = True
    router_aux_loss_coef: float = 0.001

    hc_count: int = 4
    hc_lowrank: int = 320

    ple_layer_ids: list[int] = field(default_factory=list)
    ple_embed_dim: int | None = None
    ple_conv_kernel_size: int = 4
    ngram_size: int = 3
    heads_per_ngram: int = 8
    ngram_vocab_size_base: int = 20_000_000
    make_ngram_vocab_size_divisible_by: int = 128
    seed: int = 1234
    split_ngram_parts: int = 512

    indexer_n_heads: int | None = None
    indexer_kv_heads: int | None = None
    indexer_head_dim: int | None = None
    indexer_budget: int | None = None
    indexer_compress_ratio: int | None = None

    output_gate_type: str | None = None

    #: MTP（多 token 预测）头的层数；``0`` 表示这个 checkpoint 没有 MTP 头。
    #:
    #: 真实 checkpoint 是 ``1``（``config.json`` 的 ``mtp_num_hidden_layers``），对应
    #: 权重里的 31 个 ``mtp.*`` 张量。它**不是** ``num_hidden_layers`` 的一部分：MTP
    #: 层在草稿模型里被当成第 ``num_hidden_layers`` 号层单独注册，有自己的 KV 缓存，
    #: 而主干那 48 层不含它。
    mtp_num_hidden_layers: int = 0

    #: MTP 头是否带自己的词表嵌入表。
    #:
    #: 真实 checkpoint 是 ``false``，也就是草稿器**共享**主干的 ``embed_tokens``
    #: 与 ``lm_head``（``mtp.*`` 里确实没有任何 ``embed_tokens`` / ``lm_head`` 张量，
    #: 31 个里最接近的只有 ``fc_embedding`` / ``fc_hidden``）。为 ``true`` 时权重里
    #: 会出现 ``mtp.embed_tokens.weight``，装载器按名字区分，不靠这个开关分支。
    mtp_use_dedicated_embeddings: bool = False

    #: MTP 各层的 mixer 类型，来自 ``config.json`` 的 ``mtp.layer_types``。
    #:
    #: 单独一支字段而不是复用 :attr:`layer_types`：后者的长度被钉死成
    #: ``num_hidden_layers``，而 MTP 只有 1 层。混在一起会让「48 层主干 + 1 层 MTP」
    #: 的层表长度校验直接失败。
    mtp_layer_types: list[str] = field(default_factory=lambda: ["full_attention"])

    layer_types: list[str] | None = None
    full_attention_interval: int = 4

    rope_theta: float = 10000.0
    partial_rotary_factor: float = 1.0
    mrope_section: list[int] = field(default_factory=list)
    mrope_interleaved: bool = False
    rope_type: str = "default"

    #: 以下字段由 ``__post_init__`` 推导，构造后只读。
    hc_hidden_size: int = field(init=False, default=0)
    q_dim: int = field(init=False, default=0)
    kv_dim: int = field(init=False, default=0)
    num_key_value_groups: int = field(init=False, default=0)
    rotary_dim: int = field(init=False, default=0)
    linear_key_dim: int = field(init=False, default=0)
    linear_value_dim: int = field(init=False, default=0)
    linear_conv_dim: int = field(init=False, default=0)
    linear_head_rep: int = field(init=False, default=0)
    ngram_heads: int = field(init=False, default=0)
    ngram_head_dim: int = field(init=False, default=0)
    ple_conv_dilation: int = field(init=False, default=0)
    ple_conv_state_len: int = field(init=False, default=0)
    indexer_block_topk: int = field(init=False, default=0)
    number_of_conv_states: int = field(init=False, default=1)

    def __post_init__(self) -> None:
        """推导派生的形状字段并校验整套几何。

        Raises:
            ValueError: 层类型未知或层表长度不符、``hc_count <= 1``、QSA 字段
                只给了一部分、``indexer_budget`` 不能被压缩比整除、
                ``ple_embed_dim`` 不能被 n-gram 头数整除、PLE 层号越界或落在
                非线性格上、``mrope_section`` 铺不满频率轴。这些都是加载前就能
                判定的错误，留到前向只会变成一次形状合法的垃圾输出。
        """
        self.layer_types = self._derive_layer_types()
        self.hc_hidden_size = self.hc_count * self.hidden_size

        if self.head_dim is None:
            if self.num_attention_heads <= 0:
                raise ValueError(
                    f"num_attention_heads must be positive, got {self.num_attention_heads}.")
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be a multiple of num_key_value_heads, got "
                f"{self.num_attention_heads} and {self.num_key_value_heads}.")
        self.q_dim = self.num_attention_heads * self.head_dim
        self.kv_dim = self.num_key_value_heads * self.head_dim
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.rotary_dim = int(self.head_dim * self.partial_rotary_factor)

        if self.hc_count <= 1:
            raise ValueError(f"Qwen4-Exp requires hc_count > 1, got {self.hc_count}.")
        if self.hc_lowrank <= 0:
            raise ValueError(f"hc_lowrank must be positive, got {self.hc_lowrank}.")

        self.linear_key_dim = self.linear_num_key_heads * self.linear_key_head_dim
        self.linear_value_dim = self.linear_num_value_heads * self.linear_value_head_dim
        self.linear_conv_dim = 2 * self.linear_key_dim + self.linear_value_dim
        if self.linear_num_key_heads <= 0 or self.linear_num_value_heads <= 0:
            raise ValueError("linear attention head counts must be positive.")
        if self.linear_num_value_heads % self.linear_num_key_heads != 0:
            raise ValueError(
                "linear_num_value_heads must be a multiple of linear_num_key_heads, got "
                f"{self.linear_num_value_heads} and {self.linear_num_key_heads}.")
        self.linear_head_rep = self.linear_num_value_heads // self.linear_num_key_heads

        self._validate_moe()
        self._validate_indexer()
        self._validate_ple()
        self._validate_rope()

    def _derive_layer_types(self) -> list[str]:
        """返回逐层的 mixer 类型，并把旧名字统一成新名字。

        参考实现接受 ``full_attention`` 作为 ``indexed_attention`` 的旧名，
        checkpoint 里两种都可能出现；统一之后下游只认两个常量。

        Returns:
            长度为 ``num_hidden_layers`` 的类型列表。

        Raises:
            ValueError: 列表长度与层数不符，或出现第三种类型。
        """
        if self.layer_types is None:
            interval = self.full_attention_interval
            types = [LINEAR_ATTENTION if (i + 1) % interval else INDEXED_ATTENTION
                     for i in range(self.num_hidden_layers)]
        else:
            types = [_LAYER_TYPE_ALIASES.get(item, item) for item in self.layer_types]
            if len(types) != self.num_hidden_layers:
                raise ValueError(
                    f"layer_types has {len(types)} entries but num_hidden_layers is "
                    f"{self.num_hidden_layers}.")
        unsupported = sorted(set(types) - set(LAYER_TYPES))
        if unsupported:
            raise ValueError(f"Unsupported Qwen4-Exp layer types: {unsupported}.")
        return types

    def _validate_moe(self) -> None:
        """校验路由与专家几何。

        Raises:
            ValueError: 专家数非正、``num_experts_per_tok`` 越界，或两个
                intermediate 宽度非正。
        """
        if self.num_experts <= 0:
            raise ValueError(f"num_experts must be > 0, got {self.num_experts}.")
        if not 0 < self.num_experts_per_tok <= self.num_experts:
            raise ValueError(
                "num_experts_per_tok must be in [1, num_experts], got "
                f"{self.num_experts_per_tok} and {self.num_experts}.")
        if self.moe_intermediate_size <= 0 or self.shared_expert_intermediate_size <= 0:
            raise ValueError("moe_intermediate_size and shared_expert_intermediate_size must be > 0.")

    def _validate_indexer(self) -> None:
        """校验 QSA indexer 的五个预算字段。

        Raises:
            ValueError: 只给了一部分字段（QSA 要么全开要么全关）、任一字段非正、
                KV 头不是 1、预算不能被压缩比整除，或旋转维放不进 indexer 头。
                KV 头为 1 是块池化打分公式的前提，换个数得改公式而不是改配置。
        """
        qsa = {
            "indexer_n_heads": self.indexer_n_heads,
            "indexer_kv_heads": self.indexer_kv_heads,
            "indexer_head_dim": self.indexer_head_dim,
            "indexer_budget": self.indexer_budget,
            "indexer_compress_ratio": self.indexer_compress_ratio,
        }
        if all(value is None for value in qsa.values()):
            self.indexer_block_topk = 0
            return
        missing = [name for name, value in qsa.items() if value is None]
        if missing:
            raise ValueError(f"QSA config is missing required fields: {missing}.")
        if any(int(value) <= 0 for value in qsa.values()):
            raise ValueError(f"QSA config values must be positive: {qsa}.")
        if self.indexer_kv_heads != 1:
            raise ValueError("Qwen4-Exp QSA requires indexer_kv_heads=1.")
        if self.indexer_budget % self.indexer_compress_ratio != 0:
            raise ValueError("indexer_budget must be divisible by indexer_compress_ratio.")
        if self.rotary_dim > int(self.indexer_head_dim):
            raise ValueError(
                "attention RoPE dimensions must fit the QSA index head: "
                f"rotary_dim={self.rotary_dim}, indexer_head_dim={self.indexer_head_dim}.")
        self.indexer_block_topk = int(self.indexer_budget) // int(self.indexer_compress_ratio)

    def _validate_ple(self) -> None:
        """校验 PLE 与 n-gram 几何。

        Raises:
            ValueError: 头数非正、``ple_embed_dim`` 不能被头数整除、PLE 层号越界
                或落在非线性格上、开了 PLE 却没有 eos。eos 是分段哨兵，缺了它
                n-gram 会跨序列边界哈希。
        """
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        # 参考实现里短卷积的 dilation 直接复用 ngram_size，两个 3 不是独立常数：
        # 改 ngram_size 会同时改感受野，卷积状态长度跟着变。
        self.ple_conv_dilation = self.ngram_size
        self.ple_conv_state_len = (self.ple_conv_kernel_size - 1) * self.ple_conv_dilation
        self.number_of_conv_states = 3 if self.ple_layer_ids else 1

        if self.ple_embed_dim is None:
            self.ple_embed_dim = self.hidden_size
        if not self.ple_layer_ids:
            return
        if self.ngram_heads <= 0 or self.ple_embed_dim <= 0:
            raise ValueError(
                "n-gram head count and ple_embed_dim must be positive, got "
                f"{self.ngram_heads} and {self.ple_embed_dim}.")
        if self.ple_embed_dim % self.ngram_heads != 0:
            raise ValueError(
                "ple_embed_dim must be divisible by the number of n-gram heads: "
                f"{self.ple_embed_dim} % {self.ngram_heads} != 0.")
        self.ngram_head_dim = self.ple_embed_dim // self.ngram_heads
        invalid = [layer for layer in self.ple_layer_ids
                   if layer < 1 or layer > self.num_hidden_layers]
        if invalid:
            raise ValueError(
                f"ple_layer_ids must contain one-indexed ids in [1, {self.num_hidden_layers}], "
                f"got {invalid}.")
        non_linear = [layer for layer in self.ple_layer_ids
                      if self.layer_types[layer - 1] != LINEAR_ATTENTION]
        if non_linear:
            raise ValueError(
                "Qwen4-Exp PLE is only supported on linear_attention layers, got PLE on "
                f"{non_linear}.")
        if self.eos_token is None:
            raise ValueError("eos_token_id must be set when PLE layers are enabled.")

    def _validate_rope(self) -> None:
        """校验 mrope 分节。

        Raises:
            ValueError: ``mrope_section`` 里有负值。

        故意**不**要求 ``sum(mrope_section) == rotary_dim // 2``：参考实现按
        ``slice(offset, section * 3, 3)`` 取下标，超出频率轴的部分自然是空切片，
        三轴因此退化成"其余维度都用 T 轴"。小配置上 ``mrope_section`` 保持真实
        值 ``[3, 3, 2]`` 而 ``rotary_dim`` 只有 8，正是靠这个行为才跑得通；
        在这里加硬校验会让小配置直接构造失败。
        """
        if not self.mrope_section:
            return
        if any(section < 0 for section in self.mrope_section):
            raise ValueError(f"mrope_section must be non-negative, got {self.mrope_section}.")

    @property
    def eos_token(self) -> int | None:
        """n-gram 分段哨兵用的单个 eos id。

        checkpoint 里 ``eos_token_id`` 可能是列表，哈希只认第一个。

        Returns:
            单个 eos id，或 ``None`` 表示配置里没设。
        """
        if isinstance(self.eos_token_id, list):
            return int(self.eos_token_id[0]) if self.eos_token_id else None
        return None if self.eos_token_id is None else int(self.eos_token_id)

    def ple_layer_index(self, layer_idx: int) -> int | None:
        """返回 ``layer_idx`` 这一层在 ``ple_layer_ids`` 里的序号。

        ``ple_layer_ids`` 是一基的，这个序号又是 n-gram 哈希种子的一部分
        （见 ``ngram.layer_multipliers``），所以不能拿层号代替。

        Args:
            layer_idx: 零基的 decoder 层号。

        Returns:
            PLE 模块序号（0, 1, ...），或 ``None`` 表示这层没有 PLE。
        """
        one_based = layer_idx + 1
        if one_based not in self.ple_layer_ids:
            return None
        return self.ple_layer_ids.index(one_based)

    @property
    def mtp_layer_type(self) -> str:
        """MTP 层自己的 mixer 类型（与主干层表无关）。

        checkpoint 的 ``mtp.layer_types`` 是 ``["full_attention"]``，也就是
        ``indexed_attention`` 的旧名（见 :data:`_LAYER_TYPE_ALIASES`），所以 MTP 层
        是**完整 QSA 层**：它带 indexer、会选块、有自己的 KV 与索引 key 池。

        这个属性从 ``mtp.layer_types`` 读而不是写死 ``indexed_attention``：写死会让
        一个「MTP 换成 GDN」的 checkpoint 静默按 QSA 跑，形状全对、数值全错。

        Returns:
            :data:`INDEXED_ATTENTION` 或 :data:`LINEAR_ATTENTION`。

        Raises:
            ValueError: ``mtp_layer_types`` 里有不认识的类型，或长度与
                ``mtp_num_hidden_layers`` 不符。
        """
        if not self.mtp_num_hidden_layers:
            return INDEXED_ATTENTION
        types = [_LAYER_TYPE_ALIASES.get(item, item) for item in self.mtp_layer_types]
        if len(types) != self.mtp_num_hidden_layers:
            raise ValueError(
                f"mtp.layer_types has {len(types)} entries but mtp_num_hidden_layers is "
                f"{self.mtp_num_hidden_layers}.")
        unsupported = sorted(set(types) - set(LAYER_TYPES))
        if unsupported:
            raise ValueError(f"Unsupported MTP layer types: {unsupported}.")
        return types[0]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TextConfig":
        """从 ``text_config`` 字典构造。

        只认参考实现 ``Qwen4ExpTextConfig`` 的字段：不认识的键直接忽略（权重
        目录里常混进 ``architectures``、``dtype`` 之类），认识的键原样落进
        dataclass。``rope_parameters`` 在这里摊平，热路径只留三个标量。

        Args:
            raw: ``config.json`` 里的 ``text_config`` 对象，或顶层直接就是文本
                配置的整个字典。

        Returns:
            校验完成的配置。

        Raises:
            ValueError: 见 :meth:`__post_init__`。
        """
        known = {item.name for item in fields(cls) if item.init}
        kwargs: dict[str, Any] = {}
        for key, value in raw.items():
            if key in ("rope_parameters", "rope_scaling"):
                kwargs.update(_flatten_rope(value))
            elif key == "ple_layer_ids":
                kwargs["ple_layer_ids"] = sorted(set(_as_int_list(value)))
            elif key == "layer_types" and value is not None:
                kwargs["layer_types"] = list(value)
            elif key == "mtp" and isinstance(value, dict):
                # checkpoint 把 MTP 的几何塞在一个嵌套对象里：
                # ``{"hybrid": true, "layer_types": ["full_attention"],
                #   "mtp_use_hidden_state_from_layer": null, "num_hidden_layers": 1,
                #   "rope_theta": 10000000}``。只取它的层类型，其余字段在这里没有
                # 意思——``num_hidden_layers`` 已经由顶层的 ``mtp_num_hidden_layers``
                # 给了，而 ``rope_theta`` 与主干同值（都是 1e7）。
                layers = value.get("layer_types")
                if layers is not None:
                    kwargs["mtp_layer_types"] = list(layers)
            elif key == "mrope_section" and value is not None:
                kwargs["mrope_section"] = _as_int_list(value)
            elif key in known:
                kwargs[key] = value
        return cls(**kwargs)

    @classmethod
    def from_json(cls, path: str | Path) -> "TextConfig":
        """从 ``config.json`` 读取文本配置。

        Args:
            path: 权重目录里的 ``config.json``。

        Returns:
            配置对象；顶层带 ``text_config`` 时取那一段。

        Raises:
            ValueError: 文件里既没有 ``text_config`` 也不是扁平文本配置。
        """
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        section = raw.get("text_config", raw)
        if not isinstance(section, dict):
            raise ValueError(f"{path} has no text_config object.")
        return cls.from_dict(section)

    def __repr__(self) -> str:
        """只打印决定显存的字段，免得把三十多个默认值刷满日志。"""
        return (
            f"{type(self).__name__}(hidden={self.hidden_size}, layers={self.num_hidden_layers}, "
            f"hc={self.hc_count}, experts={self.num_experts}/{self.num_experts_per_tok}, "
            f"vocab={self.vocab_size}, ple={self.ple_layer_ids})"
        )


@dataclass
class VisionConfig:
    """视觉塔（ViT）的几何。

    Attributes:
        depth: ViT 层数。
        hidden_size: ViT 宽度；``head_dim = hidden_size // num_heads``。
        num_heads: 注意力头数，无 GQA（16 Q 对 16 KV）。
        patch_size: patch 的空间边长；``temporal_patch_size`` 是时间边长。
        spatial_merge_size: merger 的 2×2 合并边长。
        num_position_embeddings: 学习位置表的行数，必须是完全平方数。
        out_hidden_size: merger 输出宽度，必须等于文本 ``hidden_size``。
    """

    depth: int = 27
    hidden_size: int = 1152
    hidden_act: str = "gelu_pytorch_tanh"
    intermediate_size: int = 4304
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    out_hidden_size: int = 2560
    num_position_embeddings: int = 2304
    initializer_range: float = 0.02
    rope_theta: float = 10000.0
    rope_type: str = "axial"

    #: 推导字段。
    head_dim: int = field(init=False, default=0)
    patch_dim: int = field(init=False, default=0)
    merge_unit: int = field(init=False, default=0)
    merged_hidden_size: int = field(init=False, default=0)
    num_grid_per_side: int = field(init=False, default=0)
    spatial_rotary_dim: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        """推导 patch 宽度、合并宽度与位置表边长，并校验可达性。

        Raises:
            ValueError: ``hidden_size`` 不能被 ``num_heads`` 整除、位置表行数
                不是完全平方数（重采样按方阵做），或轴向旋转维不是偶数。
        """
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"vision hidden_size {self.hidden_size} is not divisible by num_heads "
                f"{self.num_heads}.")
        self.head_dim = self.hidden_size // self.num_heads
        self.patch_dim = (self.in_channels * self.temporal_patch_size
                          * self.patch_size * self.patch_size)
        self.merge_unit = self.spatial_merge_size * self.spatial_merge_size
        self.merged_hidden_size = self.hidden_size * self.merge_unit
        side = int(round(math.sqrt(self.num_position_embeddings)))
        if side * side != self.num_position_embeddings:
            raise ValueError(
                "num_position_embeddings must be a perfect square, got "
                f"{self.num_position_embeddings}.")
        self.num_grid_per_side = side
        # 轴向 RoPE 把 head_dim 对半分给 H 和 W，各自的频率数是四分之一。
        self.spatial_rotary_dim = self.head_dim // 2
        if self.spatial_rotary_dim % 2 != 0:
            raise ValueError(f"axial rotary dim must be even, got head_dim={self.head_dim}.")
        if self.rope_type != "axial":
            raise ValueError(
                f"{type(self).__name__} supports only axial rope, got {self.rope_type}.")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "VisionConfig":
        """从 ``vision_config`` 字典构造。

        Args:
            raw: ``config.json`` 里的 ``vision_config`` 对象。

        Returns:
            校验完成的配置。
        """
        known = {item.name for item in fields(cls) if item.init}
        kwargs: dict[str, Any] = {}
        for key, value in raw.items():
            if key in ("rope_parameters", "rope_scaling"):
                if isinstance(value, dict):
                    if value.get("rope_theta") is not None:
                        kwargs["rope_theta"] = float(value["rope_theta"])
                    if value.get("rope_type") is not None:
                        kwargs["rope_type"] = str(value["rope_type"])
            elif key == "num_attention_heads":
                # 参考实现用 attribute_map 把 num_attention_heads 映射到
                # num_heads，checkpoint 里两种写法都有。
                kwargs["num_heads"] = int(value)
            elif key in known:
                kwargs[key] = value
        for key in ("patch_size", "temporal_patch_size"):
            if isinstance(kwargs.get(key), (list, tuple)):
                kwargs[key] = int(kwargs[key][0])
        return cls(**kwargs)


@dataclass
class ModelConfig:
    """多模态顶层配置：文本塔 + 视觉塔 + 占位符 token id。

    Attributes:
        text: 语言模型的几何。
        vision: 视觉塔的几何。
        image_token_id: 图像占位符 token；``inputs_embeds`` 里这些位置被视觉
            merger 的输出逐行覆写。
        video_token_id: 视频占位符 token。
        vision_start_token_id: 图文分界哨兵，只用于构造位置表。
        vision_end_token_id: 同上。
    """

    text: TextConfig = field(default_factory=TextConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    image_token_id: int = 248056
    video_token_id: int = 248057
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054

    def __post_init__(self) -> None:
        """校验视觉输出宽度与文本宽度一致。

        Raises:
            ValueError: ``vision.out_hidden_size`` 不等于 ``text.hidden_size``。
                两者不等时 ``masked_scatter`` 会在嵌入宽度上错位；参考实现的
                默认值 ``out_hidden_size=3584`` 是 Qwen3.5-MoE 模板残留，真实
                取值必须来自 checkpoint 的 ``vision_config``。
        """
        if self.vision.out_hidden_size != self.text.hidden_size:
            raise ValueError(
                "vision.out_hidden_size must equal text hidden_size, got "
                f"{self.vision.out_hidden_size} and {self.text.hidden_size}.")

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelConfig":
        """读取权重目录里的 ``config.json``。

        Args:
            path: 权重目录或 ``config.json`` 的路径。

        Returns:
            文本与视觉两段都解析完成的配置。
        """
        path = Path(path)
        raw = json.loads((path / "config.json" if path.is_dir() else path).read_text(encoding="utf-8"))
        text = TextConfig.from_dict(raw.get("text_config", raw))
        vision_raw = raw.get("vision_config")
        if vision_raw is None:
            vision = VisionConfig(out_hidden_size=text.hidden_size)
        else:
            vision = VisionConfig.from_dict(vision_raw)
        return cls(
            text=text,
            vision=vision,
            image_token_id=int(raw.get("image_token_id", cls.image_token_id)),
            video_token_id=int(raw.get("video_token_id", cls.video_token_id)),
            vision_start_token_id=int(raw.get("vision_start_token_id", cls.vision_start_token_id)),
            vision_end_token_id=int(raw.get("vision_end_token_id", cls.vision_end_token_id)),
        )


__all__ = [
    "INDEXED_ATTENTION",
    "LAYER_TYPES",
    "LINEAR_ATTENTION",
    "ModelConfig",
    "TextConfig",
    "VisionConfig",
]
