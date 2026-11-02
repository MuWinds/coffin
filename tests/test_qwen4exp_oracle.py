"""qwen4_exp 各模块与 golden fixture 的逐位对齐。

fixture 由参考实现（transformers 5.20.2 的 ``modeling_qwen4_exp.py``）在**小配置**下
dump 出来，路径由 ``Q4_ORACLE`` 环境变量指定；不设时按仓库布局找
``<repo>/docs/qwen4/oracle``。之所以允许覆盖：引擎仓和移植文档是两个仓库，集群上
引擎被单独同步过去时 docs/ 并不在它旁边。

判据必须是逐位对齐而不是「跑起来了」：这些模块的形状检查都通不过有意义的信息。
写错残差宽度、漏掉 ``2*sigmoid``、把 ``.mean`` 当 ``.sum``、把 ``rotate_half`` 写成
交错式——形状全对，只有数值错。

跑法（仓库任意位置）::

    python coffin/tests/test_qwen4exp_oracle.py
    Q4_ORACLE=/path/to/oracle python -m unittest -v test_qwen4exp_oracle
"""

import os
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
# `ROOT / "coffin"` 是工作区布局（`coffin-repo/coffin/tests/...`）；集群上引擎树
# 被单独上传，`parents[2]` 指到了它上面一层，包就 import 不到。挑第一个真正含
# `models/` 的祖先，两种布局都不用特判——`support.py` 对它的 `parents[1]` 也是
# 同一个假设。
ENGINE = next((p for p in Path(__file__).resolve().parents
               if (p / "models" / "qwen4exp").is_dir()), ROOT / "coffin")
sys.path.insert(0, str(ENGINE))

from models.qwen4exp.blocks import DecoderLayer  # noqa: E402
from models.qwen4exp.config import TextConfig  # noqa: E402
from models.qwen4exp.gated_residual import GatedResidual  # noqa: E402
from models.qwen4exp.gdn import GatedDeltaNet  # noqa: E402
from models.qwen4exp.moe import SparseMoeBlock  # noqa: E402
from models.qwen4exp.ngram import NGramEmbedding, PLELayer  # noqa: E402
from models.qwen4exp.norm import RMSNorm, RMSNormGated  # noqa: E402
from models.qwen4exp.qsa import Attention  # noqa: E402
from models.qwen4exp.rope import RotaryEmbedding  # noqa: E402

#: fixture 所在目录。默认按仓库布局找，集群上由 ``Q4_ORACLE`` 覆盖。
ORACLE = Path(os.environ.get("Q4_ORACLE") or (ROOT / "docs" / "qwen4" / "oracle"))

#: fp32 下的容差。参考实现的运算是 fp32，两条同样正确的实现之间的差别只来自
#: 求和顺序，量级在 1e-6 的舍入误差上；1e-5 足够严到能挡住任何公式错误
#: （公式错是 O(1) 量级），也足够松到不会因顺序不同而假红。
TOL = 1e-5


def rel(got, ref):
    """相对 ``ref`` 最大幅值的最大绝对误差。

    Args:
        got: 被测实现的结果。
        ref: fixture 里的参照结果。

    Returns:
        无量纲的相对误差，可以跨层深直接比较。
    """
    return ((got - ref).abs().max() / ref.abs().max()).item()


def load(name):
    """读一个 fixture。

    Args:
        name: fixture 文件名，如 ``"ngram.pt"``。

    Returns:
        反序列化后的字典。
    """
    return torch.load(ORACLE / name, map_location="cpu", weights_only=False)


def cfg(**kw):
    """构造与 fixture dump 脚本完全一致的小配置。

    常数必须与 ``diag/qwen4/dump_reference.py:text_config`` 逐项相同：哈希几何、
    块数、头的分组全都由它们决定，任何一个不同都会让 fixture 对不上，而报错会
    指向别处。

    Args:
        **kw: 要覆盖的字段。

    Returns:
        ``TextConfig``。
    """
    base = dict(
        vocab_size=256, hidden_size=64, num_hidden_layers=8, num_attention_heads=4,
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
        output_gate_type="sigmoid", rms_norm_eps=1e-6, max_position_embeddings=512,
        eos_token_id=2, bos_token_id=1,
        rope_theta=10000.0, partial_rotary_factor=0.25, mrope_section=[3, 3, 2],
        mrope_interleaved=True,
    )
    base.update(kw)
    return TextConfig(**base)


class OracleCase(unittest.TestCase):
    """所有对齐用例的基类：缺 fixture 时整类跳过，而不是红。"""

    @classmethod
    def setUpClass(cls):
        if not (ORACLE / "manifest.json").is_file():
            raise unittest.SkipTest(f"no qwen4 oracle fixtures under {ORACLE}")

    def assertClose(self, got, ref, what, tol=TOL):
        """断言 ``got`` 与 ``ref`` 的相对误差在 ``tol`` 内。

        Args:
            got: 被测实现的结果。
            ref: 参照结果。
            what: 失败信息里用来定位的名字。
            tol: 相对误差上限。
        """
        self.assertEqual(tuple(got.shape), tuple(ref.shape), f"{what}: shape mismatch")
        err = rel(got, ref)
        self.assertLess(err, tol, f"{what}: rel_err {err:.3e} >= {tol:.0e}")


class TestNorm(OracleCase):
    """两种 RMSNorm：零中心 ``1+w`` 与带门的 ``w*x*sigmoid(gate)``。"""

    def test_plain(self):
        """零中心约定：权重是偏移量，前向是 ``(1 + w) * x / rms(x)``。"""
        d = load("rmsnorm.pt")
        norm = RMSNorm(64, eps=1e-6)
        norm.weight.data.copy_(d["plain_w"])
        self.assertClose(norm(d["x"]), d["plain_out"], "rmsnorm plain")

    def test_grouped(self):
        """``group_size`` 让 4 路残差流各自归一：每 2560 维独立算 RMS。"""
        d = load("rmsnorm.pt")
        norm = RMSNorm(64, group_size=16, eps=1e-6)
        norm.weight.data.copy_(d["grouped_w"])
        self.assertClose(norm(d["x"]), d["grouped_out"], "rmsnorm grouped")

    def test_gated(self):
        """门控归一化：权重**不**加偏移，且归一化在门之前。"""
        d = load("rmsnorm.pt")
        norm = RMSNormGated(16, eps=1e-6, activation="sigmoid")
        norm.weight.data.copy_(d["gated_w"])
        self.assertClose(norm(d["gate"], d["gate"]), d["gated_out"], "rmsnorm gated")


class TestGatedResidual(OracleCase):
    """读门是 ``.mean``、写门是 ``2*sigmoid(w/hc_count)``，两处都只有数值证据。"""

    def test_mix_and_inject(self):
        """读门塌缩与写门的逐流标量。"""
        d = load("gated_residual.pt")
        c = cfg()
        mix = GatedResidual(c)
        mix.hc_norm.weight.data.copy_(d["hc_norm_w"])
        mix.input_mix_weight_down.weight.data.copy_(d["down_w"])
        mix.input_mix_weight_up.weight.data.copy_(d["up_w"])
        mix.block_inject_weight.weight.data.copy_(d["inject_w"])
        mixed, hyper, inj = mix(d["x"])
        self.assertClose(mixed, d["mixed"], "gated_residual mixed")
        self.assertClose(hyper, d["hyper"], "gated_residual hyper")
        self.assertClose(inj, d["inject_weights"], "gated_residual inject")

    def test_tail_collapse(self):
        """末端 ``use_combine=False``：只返回读门那一支，10240 -> 2560。"""
        d = load("gated_residual.pt")
        tail = GatedResidual(cfg(), use_combine=False)
        tail.hc_norm.weight.data.copy_(d["tail_hc_norm_w"])
        tail.input_mix_weight_down.weight.data.copy_(d["tail_down_w"])
        tail.input_mix_weight_up.weight.data.copy_(d["tail_up_w"])
        self.assertClose(tail(d["tail_x"]), d["tail_out"], "gated_residual tail")


class TestNGram(OracleCase):
    """哈希几何与查表结果。几何错了形状永远对，所以两块分开断言。"""

    def test_geometry(self):
        """层乘子、每头词表大小与偏移、补齐后的行数。"""
        d = load("ngram.pt")
        c = cfg()
        emb = NGramEmbedding(c, c.ple_embed_dim, layer_idx=1, ple_layer_index=0)
        self.assertTrue(torch.equal(emb.layer_multipliers, d["layer_multipliers"]),
                        "layer multipliers differ")
        self.assertTrue(torch.equal(emb.ngram_heads_vocab_sizes, d["head_vocab_sizes"]),
                        "head vocab sizes differ")
        self.assertTrue(torch.equal(emb.ngram_heads_offsets, d["head_offsets"]),
                        "head offsets differ")
        self.assertEqual(emb.ngram_embedding.num_embeddings, d["padded_rows"].item())

    def test_lookup(self):
        """给定同一张表，行号必须逐位一致（含 eos 分段边界）。"""
        d = load("ngram.pt")
        c = cfg()
        emb = NGramEmbedding(c, c.ple_embed_dim, layer_idx=1, ple_layer_index=0)
        emb.ngram_embedding.weight.data.copy_(d["ngram_table"])
        self.assertClose(emb(d["ids"], previous_context=None), d["ngram_out"],
                         "ngram lookup")


class TestPLE(OracleCase):
    """PLE 的完整前向：查表 -> key/value 投影 -> 门 -> 膨胀卷积。"""

    def test_forward(self):
        """逐项灌入 fixture 权重后，输出必须与参考实现一致。"""
        d = load("ngram.pt")
        c = cfg()
        ple = PLELayer(c, layer_idx=1, ple_layer_index=0)
        ple.ple_embedding.ngram_embedding.weight.data.copy_(d["ple_table"])
        ple.key_proj.weight.data.copy_(d["ple_key_w"])
        ple.value_proj.weight.data.copy_(d["ple_value_w"])
        ple.norm_key.weight.data.copy_(d["ple_norm_key_w"])
        ple.norm_query.weight.data.copy_(d["ple_norm_query_w"])
        ple.norm_conv.weight.data.copy_(d["ple_norm_conv_w"])
        ple.conv1d.weight.data.copy_(d["ple_conv_w"])
        self.assertClose(ple(d["ple_hidden"], d["ple_ids"], previous_context=None),
                         d["ple_out"], "PLE forward")


class TestGDN(OracleCase):
    """Gated DeltaNet 的分块预填充路径。"""

    def test_prefill(self):
        """分块 delta rule：衰减在对角、三角解、扫描三段都要对。"""
        d = load("gdn_prefill.pt")
        gdn = GatedDeltaNet(cfg(), layer_idx=0)
        gdn.in_proj_qkv.weight.data.copy_(d["in_proj_qkv_w"])
        gdn.in_proj_z.weight.data.copy_(d["in_proj_z_w"])
        gdn.in_proj_b.weight.data.copy_(d["in_proj_b_w"])
        gdn.in_proj_a.weight.data.copy_(d["in_proj_a_w"])
        gdn.out_proj.weight.data.copy_(d["out_proj_w"])
        gdn.conv1d.weight.data.copy_(d["conv1d_w"])
        gdn.norm.weight.data.copy_(d["norm_w"])
        gdn.A_log.data.copy_(d["A_log"])
        gdn.dt_bias.data.copy_(d["dt_bias"])
        out, _, _ = gdn(d["x"])
        self.assertClose(out, d["out"], "gdn prefill")


class TestMoE(OracleCase):
    """路由、3D 融合专家张量、标量门控的共享专家。"""

    def test_router(self):
        """softmax 在 fp32、top-k 后再归一化、索引逐位一致。"""
        d = load("moe.pt")
        c = cfg()
        moe = SparseMoeBlock(c)
        moe.gate.weight.data.copy_(d["gate_w"])
        logits, weights, indices = moe.gate(d["x"].reshape(-1, c.hidden_size))
        self.assertClose(logits, d["router_logits"], "moe router logits")
        self.assertClose(weights, d["router_weights"], "moe router weights")
        self.assertTrue(torch.equal(indices, d["router_indices"]),
                        "router indices differ")

    def test_forward(self):
        """路由专家加共享专家（逐 token 标量 sigmoid 门）。"""
        d = load("moe.pt")
        c = cfg()
        moe = SparseMoeBlock(c)
        moe.gate.weight.data.copy_(d["gate_w"])
        moe.experts.gate_up_proj.data.copy_(d["gate_up_proj"])
        moe.experts.down_proj.data.copy_(d["down_proj"])
        moe.shared_expert.gate_proj.weight.data.copy_(d["shared_gate_proj_w"])
        moe.shared_expert.up_proj.weight.data.copy_(d["shared_up_proj_w"])
        moe.shared_expert.down_proj.weight.data.copy_(d["shared_down_proj_w"])
        moe.shared_expert_gate.weight.data.copy_(d["shared_expert_gate_w"])
        self.assertClose(moe(d["x"]), d["out"], "moe forward")


class TestRope(OracleCase):
    """mrope 重组：三轴位置表 -> 32 个频率 -> 复制成 64 维旋转。"""

    def test_cos_sin(self):
        """图像段的三轴位置与文本段不同，重组顺序错了这里就会红。"""
        d = load("rope.pt")
        cos, sin = RotaryEmbedding(cfg()).cos_sin(d["position_ids"])
        self.assertClose(cos, d["cos"], "rope cos")
        self.assertClose(sin, d["sin"], "rope sin")


class TestQSA(OracleCase):
    """QSA indexer 的块选择与主注意力。"""

    def _build(self):
        d = load("qsa.pt")
        c = cfg()
        attn = Attention(c, layer_idx=3)
        for name, key in (("q_proj", "q_proj_w"), ("k_proj", "k_proj_w"),
                          ("v_proj", "v_proj_w"), ("o_proj", "o_proj_w"),
                          ("q_norm", "q_norm_w"), ("k_norm", "k_norm_w")):
            getattr(attn, name).weight.data.copy_(d[key])
        attn.indexer.index_qk_proj.weight.data.copy_(d["index_qk_proj_w"])
        attn.indexer.q_layernorm.weight.data.copy_(d["index_q_ln_w"])
        attn.indexer.k_layernorm.weight.data.copy_(d["index_k_ln_w"])
        return d, c, attn

    def test_indexer_mask(self):
        """块选择必须**逐格**一致：选出不同的块但输出碰巧接近是可能的。"""
        d, c, attn = self._build()
        rope = RotaryEmbedding(c)
        cos, sin = rope.cos_sin(d["position_ids"])
        # 索引器的返回形式跟随入参：float 掩码进 -> 加性掩码出（选中 0、
        # 未选中 finfo.min）。比较前转回 bool，免得 0/-inf 的表示差异掩盖问题。
        indexer_mask, _ = attn.indexer(d["x"], (cos, sin), d["mask"])
        got_bool = indexer_mask == 0
        ref_bool = d["indexer_mask"] == 0
        mismatched = int((got_bool != ref_bool).sum())
        self.assertEqual(mismatched, 0,
                         f"indexer selected a different block set ({mismatched} cells)")
        self.assertGreater(int(ref_bool.sum()), 0, "fixture mask selects nothing")

    def test_attention_output(self):
        """用正确（float, -inf）掩码跑出的输出，这是 oracle。

        fixture 另存了 ``out_bool_mask``——那是参考实现在 bool 掩码下的输出，
        它把 True 当 +1 加进 logits，未来 token 只是低一分而不是被屏蔽。两者
        必须分叉，这条断言同时钉住"我们没抄那个 bug"。
        """
        d, c, attn = self._build()
        rope = RotaryEmbedding(c)
        cos, sin = rope.cos_sin(d["position_ids"])
        out, _ = attn(d["x"], (cos, sin), d["mask"])
        self.assertClose(out, d["out"], "qsa out (float mask)")

        # 同一份实现喂 bool 掩码，也应该得到 float 版的结果：bool 的语义就是
        # 屏蔽，不是加一。参考实现在这里会分叉 0.048。
        out_bool, _ = attn(d["x"], (cos, sin), d["mask_bool"])
        self.assertClose(out_bool, d["out"], "qsa out (bool mask)")
        gap = (d["out"] - d["out_bool_mask"]).abs().max().item()
        self.assertGreater(gap, 1e-2,
                           "fixture's bool path did not diverge; the recorded defect "
                           "may have been fixed upstream and this test should be revisited")


class TestDecoderLayer(OracleCase):
    """整层装配：PLE + GDN + MoE + 两套 GatedResidual。"""

    def test_forward(self):
        """权重从 fixture 的 ``state_dict`` 直接灌，不靠重放 RNG。"""
        d = load("decoder_layer.pt")
        layer = DecoderLayer(cfg(), layer_idx=1)  # 一基第 2 层，挂 PLE
        missing, unexpected = layer.load_state_dict(d["weights"], strict=False)
        self.assertEqual(list(missing), [], f"weights not consumed: {missing}")
        self.assertEqual(list(unexpected), [], f"fixture keys unused: {unexpected}")
        # fixture 的 mask 是 4 维因果掩码，conv_mask 是 [B, S] 的 keep/drop，
        # 两者是不同参数——混用会在 PLE 里把序列轴广播进头轴。
        out, _ = layer(d["x"], position_embeddings=(d["cos"], d["sin"]),
                       attention_mask=d["mask"], conv_mask=d["conv_mask"],
                       ple_input_ids=d["input_ids"])
        self.assertClose(out, d["out"], "decoder layer")


if __name__ == "__main__":
    unittest.main(verbosity=2)
