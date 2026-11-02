"""QSA 选块的批量版与参考循环版必须给出逐位相同的掩码。

``QSAIndexer.select`` 是 prefill 的热点：参考实现对每个 (batch, query) 跑一次
Python 循环，每次一串 nonzero / index_select / matmul / topk / cat / scatter。
seq=128 摊到 12 个 QSA 层上是 1536 次循环。批量版改成 ``[B*S, num_blocks]`` 的
一次打分 + 一次 top-k，代价是分块的下标得先拼成规整表。

这个改写最容易错的地方全是**形状合法、数值只差一点**的：因果前缀下每个 query
能看到的元素数不同，块数因此每行不一样；尾部不足一整块的 token 必须无条件保留；
块数少于全局预算时 top-k 要自己截断。三者错任何一个都不会抛异常，只会选中错的
块——而掩码是 bool，所以判据可以直接用 ``torch.equal``，不需要容差。

覆盖：S=1（decode）、S=8、S=128（真实 prefill 长度）、续解码（past>0）、左侧
padding、块数为 0/1/多、``token_budget`` 小于块数的边界，以及 ratio=1/2 与不同
头数的退化几何。

Module: ``coffin/tests``。纯 CPU 算术，不碰 checkpoint、不加载扩展。

Constraints:
    * 对拍必须用 ``torch.equal``：掩码是 bool，任何容差都会掩盖「选中的块换了」。
    * 随机权重每例重新播种。两条实现对同一份权重打分，共同的随机种子是唯一让
      逐位比较成立的来源。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ENGINE = next((p for p in Path(__file__).resolve().parents
               if (p / "models" / "qwen4exp").is_dir()), Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ENGINE))

from models.qwen4exp.config import TextConfig  # noqa: E402
from models.qwen4exp.qsa import QSAIndexer  # noqa: E402
from models.qwen4exp.rope import RotaryEmbedding  # noqa: E402


def make_config(compress_ratio: int = 4, budget: int = 2048, n_heads: int = 4,
                head_dim: int = 128) -> TextConfig:
    """造一个索引器几何可调的小配置。

    Args:
        compress_ratio: 多少连续 token 平均成一个块。
        budget: 每个 query 从完整块里最多选多少 token。
        n_heads: 索引器的 query 头数。
        head_dim: 索引器头宽。

    Returns:
        ``TextConfig``，其余宽度取能让索引器独立跑起来的最小值。
    """
    return TextConfig(
        vocab_size=64, hidden_size=64, num_hidden_layers=4, num_attention_heads=2,
        num_key_value_heads=1, head_dim=32,
        linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=4,
        num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16,
        shared_expert_intermediate_size=16, hc_count=2, hc_lowrank=8,
        ple_layer_ids=[], ple_conv_kernel_size=4, ngram_size=2, heads_per_ngram=2,
        ngram_vocab_size_base=256, make_ngram_vocab_size_divisible_by=128,
        indexer_n_heads=n_heads, indexer_kv_heads=1, indexer_head_dim=head_dim,
        indexer_budget=budget, indexer_compress_ratio=compress_ratio,
        output_gate_type="sigmoid", rope_theta=10000.0,
        partial_rotary_factor=0.5, mrope_section=[1, 1, 1],
    )


def causal_visible(batch: int, seq: int, past: int, pad_left: int = 0) -> torch.Tensor:
    """造 `[B, 1, S, T]` 的因果可见掩码，可选左侧 padding。

    Args:
        batch: 批大小。
        seq: 本次 query 数。
        past: 已经处理过的 token 数。
        pad_left: 最前面的多少个位置在所有 query 上都不可见。

    Returns:
        bool 掩码，``True`` 表示可见。

    Raises:
        ValueError: ``pad_left`` 吃掉了某个 query 的全部可见位置。这时参考实现
            与批量版都只会选出一堆 -1，"对拍通过"不再说明任何事情，用例应该
            显式跳过而不是假装有覆盖。
    """
    total = past + seq
    visible = torch.zeros(batch, 1, seq, total, dtype=torch.bool)
    for batch_idx in range(batch):
        for query_idx in range(seq):
            visible[batch_idx, 0, query_idx, : past + query_idx + 1] = True
    if pad_left:
        if pad_left >= total:
            raise ValueError(f"pad_left={pad_left} covers the whole {total}-token context.")
        visible[:, :, :, :pad_left] = False
    return visible


def build_case(seed: int, batch: int, seq: int, past: int = 0, pad_left: int = 0,
               compress_ratio: int = 4, budget: int = 2048, n_heads: int = 4,
               head_dim: int = 128):
    """按给定几何建一个索引器，并造出它需要的一组输入。

    Returns:
        ``(indexer, query, raw_keys, visible, full_cos, full_sin, tokens)``。
    """
    config = make_config(compress_ratio, budget, n_heads, head_dim)
    torch.manual_seed(seed)
    indexer = QSAIndexer(config, layer_idx=0).eval()
    with torch.no_grad():
        indexer.index_qk_proj.weight.normal_(0.0, 0.2)
        indexer.q_layernorm.weight.normal_(0.0, 0.2)
        indexer.k_layernorm.weight.normal_(0.0, 0.2)
    tokens = past + seq
    visible = causal_visible(batch, seq, past, pad_left)
    positions = torch.arange(tokens).view(1, 1, -1).expand(3, batch, -1)
    full_cos, full_sin = RotaryEmbedding(config).cos_sin(positions, dtype=torch.float32)
    torch.manual_seed(seed + 1)
    raw_keys = torch.randn(batch, tokens, head_dim) * 0.3
    with torch.no_grad():
        # 直接造已归一化、已旋转的 query：被测的是选块，不是投影。
        query = indexer.q_layernorm(torch.randn(batch, seq, n_heads, head_dim) * 0.5)
    return indexer, query, raw_keys, visible, full_cos, full_sin, tokens


class SelectEquivalenceCase(unittest.TestCase):
    """对拍用例的共用断言。"""

    def assert_same_selection(self, seed: int, batch: int, seq: int, past: int = 0,
                              pad_left: int = 0, compress_ratio: int = 4,
                              budget: int = 2048, n_heads: int = 4,
                              head_dim: int = 128) -> torch.Tensor:
        """跑两条实现并要求掩码逐位相同。

        Returns:
            参考实现给出的掩码，供调用方再断言它确实选中了东西。
        """
        indexer, query, raw_keys, visible, cos, sin, tokens = build_case(
            seed, batch, seq, past, pad_left, compress_ratio, budget, n_heads, head_dim)
        with torch.no_grad():
            reference = indexer.select_reference(query, raw_keys, visible, cos, sin, tokens)
            batched = indexer.select(query, raw_keys, visible, cos, sin, tokens)
        self.assertEqual(tuple(batched.shape), tuple(reference.shape),
                         "批量版与参考版的掩码形状不同")
        mismatched = int((batched != reference).sum())
        self.assertEqual(mismatched, 0,
                         f"批量版选了不同的块（{mismatched} 格不同）")
        return reference


class TestSequenceLengths(SelectEquivalenceCase):
    """序列长度：decode 的一步、小 prefill、真实 prefill 长度。"""

    def test_decode_step(self):
        """S=1 是 decode；此时每行块数都受限于 ``past``，批量表几乎全是补位。"""
        reference = self.assert_same_selection(1, batch=1, seq=1, past=0)
        self.assertGreater(int(reference.sum()), 0)

    def test_decode_with_history(self):
        """续解码：历史够长时 S=1 也有多个整块，块首位置必须取全表。"""
        for past in (4, 8, 63):
            with self.subTest(past=past):
                self.assert_same_selection(2, batch=1, seq=1, past=past)

    def test_prefill_eight(self):
        """S=8：块数与尾部混合，max_blocks 由最长的那行决定。"""
        self.assert_same_selection(3, batch=2, seq=8)

    def test_prefill_eight_with_history(self):
        """chunked prefill：S=8 但前面已有 7 个 token，尾部分组因此整体错位。"""
        self.assert_same_selection(4, batch=1, seq=8, past=7)

    def test_prefill_128(self):
        """S=128 是真实 prefill 长度：32 个整块全部要过一遍打分。"""
        reference = self.assert_same_selection(5, batch=1, seq=128)
        self.assertGreater(int(reference.sum()), 0)

    def test_prefill_128_with_history(self):
        """S=128 接在 896 个 token 后面，块池化要索引到远处的历史。"""
        self.assert_same_selection(6, batch=1, seq=128, past=896)


class TestVariableVisibleSets(SelectEquivalenceCase):
    """可见集合的形态：块数为 0/1/多，以及左侧 padding 造成的不对齐分组。"""

    def test_no_complete_block(self):
        """可见 token 不足 ``compress_ratio`` 个：一个整块都没有，只剩尾部。

        这时批量表根本不建，尾部直接来自下标流。参考实现在同一条分支上走
        ``new_zeros(0)`` 再 cat，两条路径的结构完全不同。
        """
        indexer, query, raw_keys, visible, cos, sin, tokens = build_case(
            7, batch=1, seq=3)
        with torch.no_grad():
            reference = indexer.select_reference(query, raw_keys, visible, cos, sin, tokens)
            batched = indexer.select(query, raw_keys, visible, cos, sin, tokens)
        self.assertTrue(torch.equal(batched, reference))
        # 第 i 个 query 只看到 i+1 个 token；全部小于一个块。
        self.assertEqual(int(reference[0, 0, 0].sum()), 1)
        self.assertEqual(int(reference[0, 0, 2].sum()), 3)

    def test_exactly_one_block(self):
        """恰好一个整块：top-k 的 k 退化成 1，尾部为 0。"""
        self.assert_same_selection(8, batch=1, seq=4)

    def test_one_block_plus_tail(self):
        """一个整块加一条尾巴：块选中与尾部保留必须同时成立。"""
        self.assert_same_selection(9, batch=1, seq=6)

    def test_left_padding(self):
        """左侧 padding 让可见集合不从绝对位置 0 开始。

        参考实现的分组是「按可见顺序」从头每 ``compress_ratio`` 个一组，不是按
        绝对位置对齐。批量版用同一套下标流，所以两者必须仍然一致。
        """
        for pad_left in (1, 3, 5):
            with self.subTest(pad_left=pad_left):
                self.assert_same_selection(10, batch=2, seq=16, pad_left=pad_left)

    def test_ragged_batch(self):
        """整批里最长行与最短行的块数相差一整个数量级。"""
        indexer, query, raw_keys, visible, cos, sin, tokens = build_case(
            11, batch=4, seq=2, past=0)
        # 第 0 个 query 只看到 1 个 token，最后一个看到 tokens 个：块数从 0 到多。
        with torch.no_grad():
            reference = indexer.select_reference(query, raw_keys, visible, cos, sin, tokens)
            batched = indexer.select(query, raw_keys, visible, cos, sin, tokens)
        self.assertTrue(torch.equal(batched, reference))


class TestTokenBudgetBoundary(SelectEquivalenceCase):
    """``token_budget`` 的边界：它同时约束每 query 的块数与缓冲宽度。"""

    def test_budget_shorter_than_blocks(self):
        """预算只够一半的块：top-k 必须截断，且不能把补位槽选进来。"""
        self.assert_same_selection(12, batch=1, seq=8, past=32, budget=8)

    def test_budget_equals_one_block(self):
        """预算正好一个块：``block_topk == 1``，批量 top-k 退化到最小宽度。"""
        self.assert_same_selection(13, batch=1, seq=8, past=32, budget=4)

    def test_budget_above_block_count(self):
        """预算远大于块数：k 只能取块数，多出来的名额不得选到补位槽。"""
        self.assert_same_selection(14, batch=1, seq=8, past=8, budget=4096)

    def test_budget_one_block_no_tail(self):
        """预算 = 一块、且可见长度正好是块的整数倍：尾部为空。"""
        self.assert_same_selection(15, batch=1, seq=4, past=4, budget=4)


class TestDegenerateGeometry(SelectEquivalenceCase):
    """``compress_ratio`` 与头数的退化几何。"""

    def test_compress_ratio_one(self):
        """``compress_ratio == 1``：每块一个 token，块表退化成下标流本身。"""
        self.assert_same_selection(16, batch=1, seq=8, past=4, compress_ratio=1, budget=8)

    def test_compress_ratio_two(self):
        """``compress_ratio == 2``：块内平均只有两项，池化的分组仍要正确。"""
        self.assert_same_selection(17, batch=1, seq=16, past=8, compress_ratio=2)

    def test_single_index_head(self):
        """只有一个索引头时打分退化成单列 relu 求和。"""
        self.assert_same_selection(18, batch=1, seq=8, past=4, n_heads=1, head_dim=16)

    def test_multi_index_head_wide(self):
        """4 头 128 维是真实 checkpoint 的索引器几何。"""
        self.assert_same_selection(19, batch=1, seq=64, past=64, n_heads=4, head_dim=128)


if __name__ == "__main__":
    unittest.main(verbosity=2)
