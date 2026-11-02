"""真实形状的整机前向自测：48 层 + 12 层 QSA + PLE + 末端塌缩。

真实 checkpoint 是 360 GB（MoE 专家 246 GB、n-gram 表 102 GB），任何单机 CPU 都
放不下，所以这里用两种互补的"真实"来覆盖：

* **宽度真实、层数截断**：hidden 2560、24/2 头、head_dim 256、conv_dim 10240、
  512 专家、rank 320，只留 4 层（覆盖 3 层 GDN + 1 层 QSA 的完整周期）。权重
  随机初始化，验证的是形状推导、状态几何、EP 分片、数值稳定性。
* **层数真实、宽度截断**：48 层小宽度，验证层栈、PLE 只挂一层、状态逐层收集。

两者都跑 CPU、都不需要 fixture。真正需要在真机跑的是"真实权重 + W4A8"，
那属于量化与并行那一层的验收。

**默认套件只跑零开销的那部分。** 真实宽度下 4 层就是约 6.7 B 参数、13 GB 权重，
CPU 上单次前向十几分钟，`decode_step` 一条就能超过十分钟墙钟——留在默认套件里的
后果是整个模块被砍掉、看起来像"通过了 2 条"。真正跑数值的用例挂在
``Q4_REAL_SHAPE=1`` 后面，见 :class:`TestRealShape`。

跑法（仓库根目录）::

    python coffin/tests/test_qwen4exp_model.py
    Q4_REAL_SHAPE=1 python -m unittest tests.test_qwen4exp_model   # 独占机器

Module: ``coffin/tests``。依赖 ``models.qwen4exp``；不加载 checkpoint、不碰 GPU。

Constraints:
    * 默认套件只跑 ``meta`` 设备与截断配置的用例。真实宽度的前向挂在
      ``Q4_REAL_SHAPE=1`` 后面：它不是更严格的测试，只是更慢的同一个测试，
      而慢到被墙钟砍掉时连便宜的用例一起丢。
    * 真实宽度的配置只能从 ``real_config()`` 来，截断只许改 ``num_hidden_layers``
      和 ``ngram_vocab_size_base``——层宽、专家数、头数一改，"真实形状"就没有意义了。
    * ``stream_ngram=True`` 必须传：否则 PLE 会去分配 320 M × 160 的那张表。
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import torch

ENGINE = next((p for p in Path(__file__).resolve().parents
               if (p / "models" / "qwen4exp").is_dir()), Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ENGINE))

from models.qwen4exp.config import TextConfig  # noqa: E402
from models.qwen4exp.model import CausalLM, ModelStates  # noqa: E402

#: 真实 checkpoint 的文本几何（``README-modelcard.md:45-71``）。
REAL = dict(
    vocab_size=248320, hidden_size=2560, num_hidden_layers=48, num_attention_heads=24,
    num_key_value_heads=2, head_dim=256,
    linear_num_key_heads=16, linear_num_value_heads=48,
    linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4,
    num_experts=512, num_experts_per_tok=10, moe_intermediate_size=640,
    shared_expert_intermediate_size=640, hc_count=4, hc_lowrank=320,
    ple_layer_ids=[2], ple_embed_dim=2560, ple_conv_kernel_size=4,
    ngram_size=3, heads_per_ngram=8, ngram_vocab_size_base=20_000_000,
    make_ngram_vocab_size_divisible_by=128,
    indexer_n_heads=4, indexer_kv_heads=1, indexer_head_dim=128,
    indexer_budget=2048, indexer_compress_ratio=4,
    output_gate_type="sigmoid", rope_theta=10000000.0, partial_rotary_factor=0.25,
    mrope_section=[11, 11, 10], mrope_interleaved=True, eos_token_id=248044,
)


def real_config(**override):
    """返回真实几何的配置；``override`` 用来截断层数或换掉 n-gram 基数。

    Args:
        **override: 要覆盖的字段。

    Returns:
        ``TextConfig``。
    """
    fields = dict(REAL)
    fields.update(override)
    return TextConfig(**fields)


def tiny_ngram_table(model: "CausalLM") -> torch.Tensor:
    """按模型的真实行数造一张替代表：真实表 320001536 行放不进内存。

    行数必须取模型自己算出来的 ``padded_vocab_size``，不能随便给一个小数：
    哈希出来的行号是配置的函数，给少了会越界，给多了会掩盖"行数推导错了"这件事。

    Args:
        model: 已经建好的模型，用来读 PLE 层的表几何。

    Returns:
        ``[padded_vocab_size, head_dim]`` 的 fp32 表。
    """
    ple_embedding = model.model.layers[1].ple.ple_embedding
    rows = ple_embedding.padded_vocab_size
    return torch.rand(rows, ple_embedding.head_dim_per_ngram) * 0.02


def causal_mask(seq: int, past: int = 0) -> torch.Tensor:
    """fp32 因果掩码，可见处 0、屏蔽处 ``finfo.min``。

    ``past > 0`` 时 query ``i`` 的绝对位置是 ``past + i``，可见范围是
    ``[0, past + i]``——解码步的掩码必须按这个对齐，否则第 i 个 query 会看到
    比它自己更多的历史，而形状完全正确。

    Args:
        seq: 本次 chunk 的 query 数。
        past: 已经处理过的 token 数。

    Returns:
        ``[1, 1, seq, past + seq]``。
    """
    total = past + seq
    matrix = torch.full((1, 1, seq, total), torch.finfo(torch.float32).min)
    for i in range(seq):
        matrix[0, 0, i, : past + i + 1] = 0.0
    return matrix


class TestRealShapeGeometry(unittest.TestCase):
    """真实宽度下的形状与几何，全程 ``meta`` 设备，零内存、零计算。

    单独成类而不是和数值用例放一起，是因为它们守的东西完全不同：这里只要
    "形状推导对不对"，在 meta 上毫秒级就能答；数值等价性要真跑前向，那要十几分钟。
    混在一个类里的后果是后者把前者一起拖慢到被墙钟砍掉——早期的版本就是这样，
    默认套件里整个模块只跑出 2 条就没了。
    """

    def test_state_dict_matches_inventory_shapes(self):
        """每个张量的形状必须与 ``docs/qwen4/inventory.json`` 一致。

        参考实现的 checkpoint 是权威；这里的文本塔键名前缀不同（``model.`` vs
        ``model.language_model.``），按相对名逐条比对。没有 inventory 时跳过——
        引擎仓是独立仓库，``docs/`` 不一定在旁边。
        """
        import json

        inv_path = (Path(__file__).resolve().parents[2] / "docs" / "qwen4"
                    / "inventory.json")
        if not inv_path.is_file():
            self.skipTest(f"no inventory.json next to the repo ({inv_path})")
        inventory = json.loads(inv_path.read_text(encoding="utf-8"))["tensors"]
        expected = {}
        for name, meta in inventory.items():
            if name.startswith("model.language_model."):
                expected[name[len("model.language_model."):]] = meta["shape"]
            elif name == "lm_head.weight":
                expected["lm_head.weight"] = meta["shape"]
        # meta 设备不分配内存，所以这里可以用**完整的 48 层真实配置**去对——这正是
        # 要验的东西：层数、专家数与 PLE 位置全都来自 checkpoint，不是本文件里那几个
        # 截断过的测试实例。
        with torch.device("meta"):
            model = CausalLM(real_config(), stream_ngram=True)
        actual = {(key[len("model."):] if key.startswith("model.") else key): tensor
                  for key, tensor in model.state_dict().items()}
        checked = 0
        shards: dict[str, int] = {}
        for key, shape in expected.items():
            if ".ngram_embedding." in key:
                # checkpoint 把那张 320001536 行的表切成 128 个分片落盘；运行时它是
                # 一张拼好的表（流式模式下由调用方按分片取行）。这里按分片累加行数，
                # 再与模型算出的 padded_vocab_size 对账——这正是"行数推导正确"的证据。
                stem = key.split(".ngram_embedding.")[0]
                shards[stem] = shards.get(stem, 0) + int(shape[0])
                continue
            if key not in actual:
                self.fail(f"missing tensor {key}")
            self.assertEqual(list(actual[key].shape), list(shape), f"{key}: shape differs")
            checked += 1
        for stem, rows in shards.items():
            ple_embedding = model.model.layers[int(stem.split(".")[1])].ple.ple_embedding
            self.assertEqual(rows, ple_embedding.padded_vocab_size,
                             f"{stem}: checkpoint shards total {rows} rows but the "
                             f"engine derives {ple_embedding.padded_vocab_size}")
            checked += 1
        self.assertGreater(checked, 1000, "inventory comparison covered too few tensors")

    def test_ep_sharding(self):
        """EP 分片：真实宽度下每 rank 128 个专家，切片维度正确。

        Returns 之前的口径（四份部分和相加等于全量）在 ``test_qwen4exp_ep``：
        那里用小配置，能同时验证归约点的位置（每层一次、只覆盖路由专家）。
        """
        world = 4
        cfg = real_config()
        ranks = [list(range(rank * (cfg.num_experts // world),
                            (rank + 1) * (cfg.num_experts // world)))
                 for rank in range(world)]
        self.assertEqual([len(r) for r in ranks], [128, 128, 128, 128])
        self.assertEqual(ranks[0][0], 0)
        self.assertEqual(ranks[-1][-1], cfg.num_experts - 1)
        with torch.device("meta"):
            model = CausalLM(cfg, local_experts=ranks[2], stream_ngram=True)
        experts = model.model.layers[0].mlp.experts
        self.assertEqual(experts.gate_up_proj.shape,
                         (128, 2 * cfg.moe_intermediate_size, cfg.hidden_size))
        self.assertEqual(experts.down_proj.shape,
                         (128, cfg.hidden_size, cfg.moe_intermediate_size))


class TestRealShape(unittest.TestCase):
    """宽度真实、层数截断的整机前向。

    **默认跳过。** 这里的配置是真实宽度（hidden 2560、每层 512 个专家、专家中间维
    640），4 层约 6.7 B 参数、13 GB 权重；CPU 上一次前向十几分钟，`decode_step`
    单条就超过十分钟墙钟。那不是单元测试的尺度——留在默认套件里，结果是整个模块
    被墙钟砍掉、看起来还像"通过了 2 条"。

    设 ``Q4_REAL_SHAPE=1`` 打开，且要求独占机器。真机上跑它守的是**数值等价性**
    （分块与整段逐位一致、解码步正确接续）；形状与状态几何已经由
    ``test_state_dict_matches_inventory_shapes`` 和 ``test_ep_sharding`` 在
    ``meta`` 设备上零开销覆盖了。
    """

    @classmethod
    def setUpClass(cls) -> None:
        if os.environ.get("Q4_REAL_SHAPE") != "1":
            raise unittest.SkipTest(
                "real-width forward costs ~10 min per case on CPU; "
                "set Q4_REAL_SHAPE=1 on an exclusive machine to run it")
        cls.cfg = real_config(num_hidden_layers=4, ngram_vocab_size_base=1000)
        torch.manual_seed(0)
        # stream_ngram：不然会去分配 320M×160 的表。
        cls.model = CausalLM(cls.cfg, stream_ngram=True)
        cls.model.eval()

    def run_forward(self, seq: int = 4, local_experts=None):
        """跑一次前向并返回 ``(logits, states)``。

        Args:
            seq: token 数。
            local_experts: 覆盖模型上的 EP 分片；``None`` 用全量。

        Returns:
            ``(logits, ModelStates)``。
        """
        model = self.model if local_experts is None else CausalLM(
            self.cfg, local_experts=local_experts, stream_ngram=True).eval()
        ids = torch.randint(0, 1000, (1, seq))
        table = tiny_ngram_table(self.model)
        with torch.no_grad():
            logits, states = model(
                input_ids=ids, attention_mask=causal_mask(seq),
                conv_mask=torch.ones(1, seq, dtype=torch.bool),
                position_ids=torch.arange(seq).view(1, 1, -1).expand(3, 1, -1),
                ple_tables={1: table, 2: table, 3: table})
        return logits, states

    def test_logits_shape(self):
        """logits 是 ``[B, S, vocab]``，末端没有 final RMSNorm 所以直接是 hidden。"""
        logits, _ = self.run_forward(4)
        self.assertEqual(tuple(logits.shape), (1, 4, self.cfg.vocab_size))
        self.assertTrue(torch.isfinite(logits).all(), "logits contain NaN/Inf")

    def test_state_geometry(self):
        """四类状态的形状必须是真实 checkpoint 的量级。

        这些数字直接决定显存规划（GDN 递推 3.1 MB/层/序列、PLE 卷积 10240×9），
        写错了不会报错，只会让前缀复用搬错大小。
        """
        _, states = self.run_forward(4)
        layer0 = states.layers[0]
        self.assertEqual(tuple(layer0.conv_state.shape),
                         (1, self.cfg.linear_conv_dim, self.cfg.linear_conv_kernel_dim - 1))
        self.assertEqual(tuple(layer0.recurrent_state.shape),
                         (1, self.cfg.linear_num_value_heads,
                          self.cfg.linear_key_head_dim, self.cfg.linear_value_head_dim))
        self.assertEqual(layer0.recurrent_state.dtype, torch.float32,
                         "递推状态必须 fp32，bf16 在几千 token 后会发散")
        layer1 = states.layers[1]
        self.assertEqual(tuple(layer1.ple_conv_state.shape),
                         (1, self.cfg.hc_hidden_size, self.cfg.ple_conv_state_len))
        self.assertEqual(tuple(layer1.ple_context.shape), (1, self.cfg.ngram_size - 1))
        layer3 = states.layers[3]
        self.assertEqual(tuple(layer3.attn_cache.keys.shape),
                         (1, self.cfg.num_key_value_heads, 4, self.cfg.head_dim))
        # indexer 池是 [B, T, index_head_dim]，K 是旋转后的 [B, nkv, T, hd]：
        # 两个不同的池，混用不会报错但检索会对着错的 key 做选择。
        self.assertEqual(tuple(layer3.attn_cache.index_keys.shape),
                         (1, 4, int(self.cfg.indexer_head_dim)))

    def test_layer_types(self):
        """``i % 4 == 3`` 是 QSA，其余是 GDN；PLE 只在 1 号层（一基的 2）。"""
        types = [layer.layer_type for layer in self.model.model.layers]
        self.assertEqual(types, ["linear_attention"] * 3 + ["indexed_attention"])
        has_ple = [layer.ple is not None for layer in self.model.model.layers]
        self.assertEqual(has_ple, [False, True, False, False])

    def test_decode_step(self):
        """解码一步：传入上一步状态后 logits 是 ``[B, 1, vocab]``。

        全位置表必须跟着一起长——indexer 给压缩块算 RoPE 要用块首的绝对位置，
        漏掉它解码第 2 步之后的旋转角就是错的。
        """
        logits, states = self.run_forward(4)
        prefix_positions = states.position_ids
        self.assertEqual(tuple(prefix_positions.shape), (3, 1, 4))
        step2 = CausalLM(self.cfg, stream_ngram=True).eval()
        step2.load_state_dict(self.model.state_dict())
        table = tiny_ngram_table(self.model)
        with torch.no_grad():
            logits2, states2 = step2(
                input_ids=torch.tensor([[7]]),
                attention_mask=causal_mask(1, past=4),
                conv_mask=torch.ones(1, 1, dtype=torch.bool),
                ple_tables={1: table, 2: table, 3: table},
                states=states)
        self.assertEqual(tuple(logits2.shape), (1, 1, self.cfg.vocab_size))
        self.assertEqual(tuple(states2.position_ids.shape), (3, 1, 5))
        self.assertTrue(torch.isfinite(logits2).all())

    def test_chunked_matches_full(self):
        """一次前向整个序列，与"先 4 个 token、再 2 个"必须逐位一致。

        这是整个前缀缓存契约的地基：GDN 的递推状态、卷积窗口、QSA 的 KV 与
        indexer 池、PLE 的 n-gram 上下文与短卷积窗口，任何一处没被正确接续，
        分块结果都会偏——而两条路径的**形状完全一样**，只有这个比较能发现。

        实测两次真实缺陷都是它抓到的：PLE 从不更新上下文（解码第 2 步起对着
        永远是 eos 的假上下文哈希）、indexer 只拿当前 chunk 的 key 去索引绝对
        下标（缓存步直接越界）。
        """
        seq = 6
        ids = torch.randint(0, 1000, (1, seq))
        table = tiny_ngram_table(self.model)
        tables = {1: table, 2: table, 3: table}

        def run(model, chunk, past, states=None):
            with torch.no_grad():
                return model(
                    input_ids=ids[:, past:past + chunk],
                    attention_mask=causal_mask(chunk, past=past),
                    conv_mask=torch.ones(1, chunk, dtype=torch.bool),
                    position_ids=(torch.arange(past, past + chunk)
                                  .view(1, 1, -1).expand(3, 1, -1)),
                    ple_tables=tables, states=states)

        full, _ = run(self.model, seq, 0)
        _, state = run(self.model, 4, 0)
        chunked, _ = run(self.model, 2, 4, states=state)
        self.assertLess((full[0, -2:] - chunked[0]).abs().max().item(), 1e-5,
                        "chunked prefill diverged from the full forward")


class TestDeepStack(unittest.TestCase):
    """层数真实的 48 层栈，宽度截断到能在 CPU 上跑。"""

    def test_48_layers(self):
        """48 层 = 12 × (GDN ×3 + QSA)，逐层状态齐全。"""
        cfg = TextConfig(
            vocab_size=256, hidden_size=64, num_hidden_layers=48, num_attention_heads=4,
            num_key_value_heads=2, head_dim=32,
            linear_num_key_heads=2, linear_num_value_heads=6,
            linear_key_head_dim=16, linear_value_head_dim=16,
            linear_conv_kernel_dim=4,
            num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32,
            shared_expert_intermediate_size=32, hc_count=4, hc_lowrank=16,
            ple_layer_ids=[2], ple_embed_dim=64, ple_conv_kernel_size=4,
            ngram_size=3, heads_per_ngram=8, ngram_vocab_size_base=1000,
            make_ngram_vocab_size_divisible_by=128,
            indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=16,
            indexer_budget=16, indexer_compress_ratio=4,
            output_gate_type="sigmoid", rope_theta=10000.0,
            partial_rotary_factor=0.25, mrope_section=[3, 3, 2],
            eos_token_id=2,
        )
        torch.manual_seed(2)
        model = CausalLM(cfg).eval()
        seq = 3
        ids = torch.randint(0, 256, (1, seq))
        with torch.no_grad():
            logits, states = model(
                input_ids=ids, attention_mask=causal_mask(seq),
                conv_mask=torch.ones(1, seq, dtype=torch.bool))
        self.assertEqual(tuple(logits.shape), (1, seq, cfg.vocab_size))
        self.assertEqual(len(states.layers), 48)
        for index, layer in enumerate(states.layers):
            if cfg.layer_types[index] == "linear_attention":
                self.assertIsNotNone(layer.recurrent_state, f"layer {index} lost rec state")
                self.assertIsNone(layer.attn_cache, f"layer {index} should have no attn cache")
            else:
                self.assertIsNotNone(layer.attn_cache, f"layer {index} lost attn cache")
                self.assertIsNone(layer.recurrent_state,
                                  f"layer {index} should have no rec state")

    def test_state_reuse_from_empty(self):
        """``ModelStates.empty`` 的层数必须与层栈一致，否则状态会串层。"""
        cfg = TextConfig(
            vocab_size=128, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
            num_key_value_heads=1, head_dim=16,
            linear_num_key_heads=1, linear_num_value_heads=2,
            linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=4,
            num_experts=4, num_experts_per_tok=1, moe_intermediate_size=16,
            shared_expert_intermediate_size=16, hc_count=2, hc_lowrank=8,
            output_gate_type="sigmoid", rope_theta=10000.0,
            partial_rotary_factor=0.5, mrope_section=[1, 1, 1],
            indexer_n_heads=1, indexer_kv_heads=1, indexer_head_dim=16,
            indexer_budget=4, indexer_compress_ratio=2,
        )
        states = ModelStates.empty(cfg.num_hidden_layers)
        self.assertEqual(len(states.layers), 2)
        self.assertIsNone(states.position_ids)
        for layer in states.layers:
            self.assertIsNone(layer.attn_cache)
            self.assertIsNone(layer.recurrent_state)


if __name__ == "__main__":
    unittest.main()
