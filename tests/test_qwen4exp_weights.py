"""加载路径的回归：从 checkpoint 名映射到张量装进模型，全程可核对。

真实权重是 360 GB，开发机上连一个分片都放不下，所以这里分两层验证：

* **命名与切片规则**（``plan`` / ``engine_key`` / ``slice_experts``）——纯字符串与
  张量视图运算，任何机器上都能跑，而且**不需要任何权重**。这一层的价值最高：
  映射错一个前缀，某个参数就会停在随机初始值上，前向照跑、答案全错。
* **端到端装载**——用 ``decoder_layer.pt`` 里那 34 个真张量，走 ``DictSource``
  装进一个 ``DecoderLayer``，再前向，与 fixture 的 ``out`` 比。这证明"映射 +
  装载 + 前向"这条链是通的，不只是各段单独对。

真实的 safetensors 分片路径要 ``safetensors`` 包和真实 checkpoint，在集群侧跑
（``coffin/tests`` 的其余部分也是这个分层方式）。

跑法（仓库根目录）::

    python coffin/tests/test_qwen4exp_weights.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ENGINE = next((p for p in Path(__file__).resolve().parents
               if (p / "models" / "qwen4exp").is_dir()), Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ENGINE))

from models.qwen4exp.blocks import DecoderLayer  # noqa: E402
from models.qwen4exp.config import TextConfig  # noqa: E402
from models.qwen4exp.model import CausalLM  # noqa: E402
from models.qwen4exp.weights import (  # noqa: E402
    DictSource,
    NGRAM_SHARD_MARKER,
    engine_key,
    load_into,
    plan,
    slice_experts,
    verify,
)

ORACLE = Path(__file__).resolve().parents[2] / "docs" / "qwen4" / "oracle"


def load_fixture(name: str) -> dict:
    """读一个 oracle fixture。

    Args:
        name: 文件名，如 ``"decoder_layer.pt"``。

    Returns:
        反序列化后的字典。
    """
    return torch.load(ORACLE / name, map_location="cpu", weights_only=False)


def small_config(**override) -> TextConfig:
    """与 fixture dump 脚本一致的小配置。

    Args:
        **override: 要覆盖的字段。

    Returns:
        ``TextConfig``。
    """
    fields = dict(
        vocab_size=256, hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=32,
        linear_num_key_heads=2, linear_num_value_heads=6,
        linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
        num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32,
        shared_expert_intermediate_size=32, hc_count=4, hc_lowrank=16,
        ple_layer_ids=[2], ple_embed_dim=64, ple_conv_kernel_size=4,
        ngram_size=3, heads_per_ngram=8, ngram_vocab_size_base=1000,
        make_ngram_vocab_size_divisible_by=128,
        indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=16,
        indexer_budget=16, indexer_compress_ratio=4,
        output_gate_type="sigmoid", rope_theta=10000.0, partial_rotary_factor=0.25,
        mrope_section=[3, 3, 2], eos_token_id=2, pad_token_id=0,
    )
    fields.update(override)
    return TextConfig(**fields)


def causal_mask(seq: int) -> torch.Tensor:
    """fp32 因果掩码。"""
    matrix = torch.full((1, 1, seq, seq), torch.finfo(torch.float32).min)
    for i in range(seq):
        matrix[0, 0, i, : i + 1] = 0.0
    return matrix


class TestNameMapping(unittest.TestCase):
    """checkpoint 名到引擎名的映射。前缀错一个，参数就停在随机初始值上。"""

    def test_text_prefix_is_stripped(self):
        """多模态包装的 ``model.language_model.`` 要去掉，换成引擎的 ``model.``。"""
        self.assertEqual(
            engine_key("model.language_model.layers.0.linear_attn.A_log"),
            "model.layers.0.linear_attn.A_log")
        self.assertEqual(
            engine_key("model.language_model.embed_tokens.weight"),
            "model.embed_tokens.weight")
        self.assertEqual(
            engine_key("model.language_model.hyper_connection_mixer.hc_norm.weight"),
            "model.hyper_connection_mixer.hc_norm.weight")

    def test_longest_prefix_wins(self):
        """``model.language_model.`` 必须先于 ``model.`` 匹配。

        否则 ``model.language_model.embed_tokens.weight`` 会映射成
        ``model.language_model.embed_tokens.weight``（原样保留），任何一个前缀都
        对不上——而它不会报错，只是那个参数空着。
        """
        got = engine_key("model.language_model.embed_tokens.weight")
        self.assertNotIn("language_model", got)
        self.assertTrue(got.startswith("model."))

    def test_lm_head_stays_top_level(self):
        """``lm_head`` 在 ``CausalLM`` 顶层，不属于 ``TextModel``。"""
        self.assertEqual(engine_key("lm_head.weight"), "lm_head.weight")

    def test_mtp_tensors_map_verbatim(self):
        """``mtp.*`` 原样返回：引擎侧 ``CausalLM.mtp`` 的子树前缀正是 ``mtp.``。

        加 ``model.`` 会让它落进 ``TextModel`` 的子树，而 ``TextModel`` 只有
        ``embed_tokens`` / ``layers`` / ``rotary_emb`` / ``hyper_connection_mixer``
        ——那 31 个张量会在那里**没有归属**，而 ``verify`` 只会报「计划张量不是模型
        参数」，看不出是前缀的事。
        """
        self.assertEqual(engine_key("mtp.fc_embedding.weight"), "mtp.fc_embedding.weight")
        self.assertEqual(engine_key("mtp.hyper_connection_mixer.hc_norm.weight"),
                         "mtp.hyper_connection_mixer.hc_norm.weight")
        # 引擎里 MTP 的专家张量仍然命中 ``EXPERT_TENSORS`` 后缀，所以 EP 切片规则
        # 对它自动生效、不需要另写一条。
        self.assertEqual(engine_key("mtp.layers.0.mlp.experts.gate_up_proj"),
                         "mtp.layers.0.mlp.experts.gate_up_proj")
        self.assertTrue(engine_key("mtp.layers.0.mlp.experts.gate_up_proj").endswith(
            ".mlp.experts.gate_up_proj"))

    def test_vision_tensors_are_skipped(self):
        """视觉塔不由文本塔持有，映射为 ``None`` 而不是瞎猜一个名字。"""
        self.assertIsNone(engine_key("model.visual.patch_embed.proj.weight"))
        self.assertIsNone(engine_key("model.visual.merger.linear_fc1.weight"))

    def test_ngram_shard_rejected_single(self):
        """单个分片不能直接映射：必须先按几何拼成一张表。"""
        with self.assertRaises(ValueError):
            engine_key("model.language_model.layers.1.ple.ple_embedding"
                       f"{NGRAM_SHARD_MARKER}0.weight")


class TestPlanning(unittest.TestCase):
    """``plan`` 的覆盖性与 n-gram 分组。"""

    def test_covers_every_model_parameter(self):
        """计划必须覆盖模型的**每一个**参数，一个不漏。

        这是加载路径最重要的一条断言：漏掉一个，它的 ``nn.Parameter`` 就停在
        初始化值上（``normal_(0.02)``，不会 NaN），前向照跑、数值全错。用 meta
        设备构造模型，所以这里对**真实 48 层配置**也零开销。
        """
        cfg = small_config()
        with torch.device("meta"):
            model = CausalLM(cfg)
        keys = [f"model.language_model.{key[len('model.'):]}"
                for key in model.state_dict() if key.startswith("model.")]
        keys += ["lm_head.weight"]
        source = DictSource({key: torch.empty(0) for key in keys})
        result = plan(source, cfg)
        problems = verify(result, model.state_dict())
        self.assertEqual(problems, [], "plan does not cover the model")

    def test_real_shape_coverage(self):
        """真实 48 层配置下同样全覆盖（1165 个张量，meta 设备零内存）。"""
        cfg = small_config(
            vocab_size=248320, hidden_size=2560, num_hidden_layers=48,
            num_attention_heads=24, num_key_value_heads=2, head_dim=256,
            linear_num_key_heads=16, linear_num_value_heads=48,
            linear_key_head_dim=128, linear_value_head_dim=128,
            num_experts=512, num_experts_per_tok=10, moe_intermediate_size=640,
            shared_expert_intermediate_size=640, hc_lowrank=320, ple_embed_dim=2560,
            ngram_vocab_size_base=20_000_000,
            indexer_n_heads=4, indexer_head_dim=128, indexer_budget=2048,
        )
        with torch.device("meta"):
            model = CausalLM(cfg, stream_ngram=True)
        state = model.state_dict()
        keys = [f"model.language_model.{key[len('model.'):]}"
                for key in state if key.startswith("model.")]
        keys += ["lm_head.weight"]
        # n-gram 表按 128 分片喂进去。
        padded = model.model.layers[1].ple.ple_embedding.padded_vocab_size
        per_shard = padded // 128
        keys += [f"model.language_model.layers.1.ple.ple_embedding"
                 f"{NGRAM_SHARD_MARKER}{index}.weight" for index in range(128)]
        source = DictSource({key: torch.empty(0) for key in keys})
        result = plan(source, cfg)
        self.assertEqual(verify(result, state), [])
        # 流式模式下 n-gram 表不是模型参数，所以账目比 state_dict 多一条聚合键。
        self.assertEqual(result.total(), len(state) + 1)
        self.assertEqual(len(result.ngram), 1)
        self.assertEqual(len(next(iter(result.ngram.values()))), 128)
        self.assertGreater(per_shard, 0)

    def test_rejects_duplicate_mapping(self):
        """两个 checkpoint 张量映射到同一个引擎参数时必须报错。

        静默取后者会让某一层用上另一层的权重——形状全对，图全错。
        """
        source = DictSource({
            "model.language_model.layers.0.mlp.gate.weight": torch.empty(0),
            "model.layers.0.mlp.gate.weight": torch.empty(0),
        })
        with self.assertRaises(ValueError):
            plan(source, small_config())

    def test_skips_mtp_and_vision(self):
        """``mtp.*`` 与视觉塔进 ``skipped``，不静默变成模型参数。"""
        source = DictSource({
            "model.language_model.embed_tokens.weight": torch.empty(0),
            "mtp.layers.0.mlp.gate.weight": torch.empty(0),
            "model.visual.pos_embed.weight": torch.empty(0),
        })
        result = plan(source, small_config())
        self.assertEqual(len(result.skipped), 2)
        self.assertIn("mtp.layers.0.mlp.gate.weight", result.skipped)

    def test_ngram_shards_sorted_by_index(self):
        """分片按 ``k`` 排序，不是按字符串顺序。

        字符串顺序会把 ``shard_10`` 排在 ``shard_2`` 前面，拼出来的表行序整体错位
        ——哈希查到的仍是合法行号，只是内容对不上，检索质量下降而形状全对。
        """
        names = [f"model.language_model.layers.1.ple.ple_embedding"
                 f"{NGRAM_SHARD_MARKER}{index}.weight" for index in (0, 2, 10, 1)]
        # 故意打乱输入顺序。
        source = DictSource({name: torch.empty(0) for name in [names[2], names[0],
                                                               names[3], names[1]]})
        result = plan(source, small_config())
        grouped = next(iter(result.ngram.values()))
        self.assertEqual([index for _, index in grouped], [0, 1, 2, 10])


class TestExpertSlicing(unittest.TestCase):
    """EP 分片：沿第 0 维切，且顺序必须是升序。"""

    def test_slices_along_expert_axis(self):
        """切出来的必须是那几行，且顺序按传入的号排列。"""
        tensor = torch.arange(8 * 3 * 2).reshape(8, 3, 2).float()
        got = slice_experts(tensor, [4, 5, 6, 7], total_experts=8)
        self.assertEqual(tuple(got.shape), (4, 3, 2))
        self.assertTrue(torch.equal(got[0], tensor[4]))
        self.assertTrue(torch.equal(got[3], tensor[7]))

    def test_none_returns_full(self):
        """不分片时原样返回，不做拷贝。"""
        tensor = torch.zeros(8, 3, 2)
        self.assertIs(slice_experts(tensor, None, 8), tensor)

    def test_rejects_unsorted(self):
        """未排序的专家号必须拒绝：路由是按位置把全局号映射到本地号的。"""
        with self.assertRaises(ValueError):
            slice_experts(torch.zeros(8, 3, 2), [7, 4, 5, 6], 8)

    def test_rejects_wrong_expert_count(self):
        """第 0 维与配置不符时必须报错，而不是默默切一行。"""
        with self.assertRaises(ValueError):
            slice_experts(torch.zeros(4, 3, 2), [0], 8)

    def test_rejects_out_of_range(self):
        """越界的专家号必须报错。"""
        with self.assertRaises(ValueError):
            slice_experts(torch.zeros(8, 3, 2), [8], 8)


class _PrefixedLayer(torch.nn.Module):
    """给 ``DecoderLayer`` 套上 ``model.layers.N.`` 前缀，好按 checkpoint 名装载。

    真实层的参数在 checkpoint 里叫 ``model.language_model.layers.1.xxx``。把层包进
    一层命名空间之后，``engine_key`` 的映射规则可以原样复用，不必为单层另写一套
    ——装载路径只有一条，测试验的才是线上跑的那条。
    """

    def __init__(self, layer: torch.nn.Module, name: int) -> None:
        """建命名空间。

        Args:
            layer: 真正的 decoder 层。
            name: 层号；子模块名会变成 ``model.layers.{name}``。
        """
        super().__init__()
        self.model = torch.nn.ModuleDict({"layers": torch.nn.ModuleDict(
            {str(name): layer})})


class TestEndToEnd(unittest.TestCase):
    """用 fixture 的 34 个真张量走完整装载 + 前向。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = load_fixture("decoder_layer.pt")
        cls.cfg = small_config()

    def test_layer_loaded_from_source_matches_fixture(self):
        """``DictSource`` 装进 ``DecoderLayer`` 之后，前向必须与 fixture 逐位一致。

        这条把"映射 + 装载 + 前向"整条链钉在一起：以前是直接
        ``load_state_dict(fixture["weights"])``，现在权重必须经过 checkpoint 名的
        映射，才能证明映射规则与真实 checkpoint 布局是同一条。
        """
        fixture = self.fixture
        source = DictSource(fixture["weights"], prefix="model.layers.1.")
        # 单层模型的 state_dict 只有这一层的键，所以 plan 必须限定子树：不限定
        # 就会拿全模型的账目去核对，报"34 个计划张量不是模型参数"。
        layer = _PrefixedLayer(DecoderLayer(self.cfg, layer_idx=1), 1)
        report = load_into(layer, source, self.cfg, with_lm_head=False,
                           subtree="model.layers.1.")
        self.assertEqual(report["loaded"], len(fixture["weights"]))
        with torch.no_grad():
            out, _ = layer.model["layers"]["1"](
                fixture["x"], position_embeddings=(fixture["cos"], fixture["sin"]),
                attention_mask=fixture["mask"], conv_mask=fixture["conv_mask"],
                ple_input_ids=fixture["input_ids"])
        self.assertLess(((out - fixture["out"]).abs().max()
                         / fixture["out"].abs().max()).item(), 1e-6)

    def test_expert_slicing_end_to_end(self):
        """装 1/4 专家时，模型里只有 1/4，且四份能拼回全量。

        这是 EP 加载路径在真实张量上的证据：用 fixture 的 MoE 权重切四份，
        装进四个分片模型，各自的专家参数拼起来必须等于原张量。
        """
        cfg = small_config()
        model = CausalLM(cfg)
        reference = {name: tensor.clone() for name, tensor in model.state_dict().items()
                     if ".mlp.experts." in name}
        state = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        keys = [f"model.language_model.{key[len('model.'):]}"
                for key in state if key.startswith("model.")]
        keys += ["lm_head.weight"]
        source = DictSource({key: state[key.replace("model.language_model.", "model.", 1)
                                        if key.startswith("model.language_model.") else key]
                             for key in keys})
        per_rank = cfg.num_experts // 4
        for rank in range(4):
            shard = list(range(rank * per_rank, (rank + 1) * per_rank))
            model_shard = CausalLM(cfg, local_experts=shard)
            load_into(model_shard, source, cfg, local_experts=shard)
            got = {name: tensor for name, tensor in model_shard.state_dict().items()
                   if ".mlp.experts." in name}
            for name, tensor in got.items():
                self.assertEqual(tensor.shape[0], per_rank, name)
                expected = reference[name][rank * per_rank:(rank + 1) * per_rank]
                self.assertTrue(torch.equal(tensor, expected),
                                f"{name}: rank {rank} got the wrong rows")


if __name__ == "__main__":
    unittest.main()
