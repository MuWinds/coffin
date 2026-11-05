"""分页 KV 的验收：与稠密缓存在小配置上**逐位相同**。

为什么用 ``torch.equal`` 而不是容差：分页实现只是把「每步 ``torch.cat`` 一次历史」
换成「写进预分配的池子」，注意力看到的张量形状与数值都应完全一样。任何差异都说明
写入位置、视图切片或截断语义错了——那种错在长上下文里表现为"答案慢慢变差"，
用容差判会把它们全部放过。

覆盖三条：
1. 逐层缓存语义：连续 append 之后，``keys``/``values``/``index_keys`` 与稠密实现
   逐位相同；``clone`` / ``truncate`` 的游标正确。
2. 模型级：同一段 token 用稠密状态与分页状态各跑 3 步，每步 logits 都相同。
3. 越界与非法截断必须抛 ``ValueError``（池子装不下的上下文静默截断是最贵的错）。
"""

from __future__ import annotations

import unittest

import torch

from models.qwen4exp.config import TextConfig
from models.qwen4exp.model import CausalLM, ModelStates, causal_mask, paged_layer_states
from models.qwen4exp.qsa import AttentionCache, PagedAttentionCache


def tiny_config() -> TextConfig:
    """建一个 4 层、8 专家的极小配置（3 个 GDN 层 + 1 个 QSA 层）。

    Returns:
        ``TextConfig``。
    """
    return TextConfig.from_dict({
        "hidden_size": 128, "hc_count": 4, "hc_lowrank": 16,
        "head_dim": 64, "num_attention_heads": 4, "num_key_value_heads": 2,
        "indexer_head_dim": 32, "indexer_n_heads": 1, "indexer_kv_heads": 1,
        "indexer_compress_ratio": 2, "indexer_budget": 8,
        "partial_rotary_factor": 0.25, "mrope_section": [3, 3, 2],
        "mrope_interleaved": True,
        "linear_key_head_dim": 32, "linear_value_head_dim": 32,
        "linear_num_key_heads": 2, "linear_num_value_heads": 2,
        "linear_conv_kernel_dim": 4, "num_experts": 8, "num_experts_per_tok": 2,
        "moe_intermediate_size": 32, "shared_expert_intermediate_size": 32,
        "num_hidden_layers": 4, "vocab_size": 256,
        "layer_types": ["linear_attention", "linear_attention",
                        "linear_attention", "full_attention"],
        "ple_layer_ids": [], "ngram_size": 3, "ple_embed_dim": 128,
        "max_position_embeddings": 512, "rms_norm_eps": 1e-6,
    })


def build_model() -> CausalLM:
    """按固定种子建一个 fp32 小模型。

    Returns:
        ``CausalLM``（CPU、fp32）。
    """
    torch.manual_seed(11)
    return CausalLM(tiny_config(), stream_ngram=True, with_mtp=False).float()


class CacheSemanticsTest(unittest.TestCase):
    """逐层缓存与稠密实现的等价性。"""

    def test_append_matches_dense(self) -> None:
        """连续三步 append 后三块池子与稠密缓存逐位相同。"""
        torch.manual_seed(3)
        batch, heads, dim, idim = 2, 2, 8, 4
        dense: AttentionCache | None = None
        paged = PagedAttentionCache(batch, 128, heads, dim, idim,
                                    dtype=torch.float32, device=torch.device("cpu"),
                                    block_size=16)
        for _ in range(3):
            k = torch.randn(batch, heads, 1, dim)
            v = torch.randn(batch, heads, 1, dim)
            ik = torch.randn(batch, 1, idim)
            dense = (AttentionCache(keys=k, values=v, index_keys=ik)
                     if dense is None else dense.append(k, v, ik))
            paged = paged.append(k, v, ik)
        self.assertTrue(torch.equal(dense.keys, paged.keys))
        self.assertTrue(torch.equal(dense.values, paged.values))
        self.assertTrue(torch.equal(dense.index_keys, paged.index_keys))

    def test_clone_and_truncate(self) -> None:
        """clone 拷内容、truncate 只移动游标，且原对象不受影响。"""
        paged = PagedAttentionCache(1, 64, 2, 8, 4, dtype=torch.float32,
                                    device=torch.device("cpu"), block_size=16)
        paged = paged.append(torch.randn(1, 2, 2, 8), torch.randn(1, 2, 2, 8),
                             torch.randn(1, 2, 4))
        clone = paged.clone()
        self.assertEqual(clone.cursor, paged.cursor)
        clone.append(torch.randn(1, 2, 1, 8), torch.randn(1, 2, 1, 8),
                     torch.randn(1, 1, 4))
        self.assertEqual(paged.cursor, 2)
        short = paged.truncate(1)
        self.assertEqual(short.cursor, 1)
        self.assertTrue(torch.equal(short.keys, paged.keys[:, :, :1]))

    def test_overflow_and_bad_truncate(self) -> None:
        """装不下与非法截断都要抛 ValueError。"""
        paged = PagedAttentionCache(1, 16, 2, 8, 4, dtype=torch.float32,
                                    device=torch.device("cpu"), block_size=16)
        paged = paged.append(torch.randn(1, 2, 16, 8), torch.randn(1, 2, 16, 8),
                             torch.randn(1, 16, 4))
        with self.assertRaises(ValueError):
            paged.append(torch.randn(1, 2, 1, 8), torch.randn(1, 2, 1, 8),
                         torch.randn(1, 1, 4))
        with self.assertRaises(ValueError):
            paged.truncate(17)


class PagedModelTest(unittest.TestCase):
    """模型级：分页状态与稠密状态给出同一串 logits。"""

    def test_step_logits_match_dense(self) -> None:
        """三步 decode 的 logits 与稠密路径逐位相同。"""
        model = build_model()
        tokens = list(range(2, 10))
        dense_states: ModelStates | None = None
        paged_states: ModelStates | None = None
        past = 0
        for step in range(3):
            ids = torch.tensor([[tokens[step]]], dtype=torch.long)
            mask = causal_mask(1, 1, past)
            conv = torch.ones((1, 1), dtype=torch.bool)
            with torch.no_grad():
                dense_logits, dense_states = model(
                    input_ids=ids, attention_mask=mask, conv_mask=conv,
                    states=dense_states, logits_to_keep=1)
                if paged_states is None:
                    paged_states = paged_layer_states(model.config, 1, 64,
                                                      torch.device("cpu"),
                                                      torch.float32)
                paged_logits, paged_states = model(
                    input_ids=ids, attention_mask=mask, conv_mask=conv,
                    states=paged_states, logits_to_keep=1)
            self.assertTrue(torch.equal(dense_logits, paged_logits),
                            f"第 {step} 步分页与稠密 logits 不同")
            past += 1

    def test_chunked_prefill_matches_oneshot(self) -> None:
        """分块预填充与一次性预填充给出同一末位 logits。

        分块的算术形状不同（``[4, kv]`` 两次的累加次序与 ``[8, kv]`` 一次不同），
        所以用 fp32 容差而不是 ``torch.equal``；但容差只放累加次序，不放"看错位置"
        ——后者会让 max|d| 直接到量级 1。
        """
        model = build_model()
        tokens = list(range(2, 10))
        mask = causal_mask(1, 8, 0)
        conv = torch.ones((1, 8), dtype=torch.bool)
        ids = torch.tensor([tokens], dtype=torch.long)
        with torch.no_grad():
            oneshot, _ = model(input_ids=ids, attention_mask=mask, conv_mask=conv,
                               states=None, logits_to_keep=1)
        states = paged_layer_states(model.config, 1, 64, torch.device("cpu"),
                                    torch.float32)
        with torch.no_grad():
            for start in (0, 4):
                chunk = torch.tensor([tokens[start:start + 4]], dtype=torch.long)
                cmask = causal_mask(1, 4, start)
                cconv = torch.ones((1, 4), dtype=torch.bool)
                logits, states = model(input_ids=chunk, attention_mask=cmask,
                                       conv_mask=cconv, states=states,
                                       logits_to_keep=1)
        gap = float((oneshot - logits).abs().max())
        scale = float(oneshot.abs().max())
        self.assertLess(gap / max(scale, 1e-12), 1e-4,
                        f"分块预填充与一次性差 {gap:.3e}（相对 {gap / scale:.2e}）")

    def test_prefill_then_decode_matches(self) -> None:
        """预填充 + 一步 decode 与全稠密路径逐位相同（覆盖 chunk 前写入）。"""
        model = build_model()
        tokens = list(range(2, 12))
        mask = causal_mask(1, 8, 0)
        conv = torch.ones((1, 8), dtype=torch.bool)
        ids = torch.tensor([tokens[:8]], dtype=torch.long)
        paged_states = paged_layer_states(model.config, 1, 64, torch.device("cpu"),
                                          torch.float32)
        with torch.no_grad():
            dense_logits, dense_states = model(
                input_ids=ids, attention_mask=mask, conv_mask=conv,
                states=None, logits_to_keep=1)
            paged_logits, paged_states = model(
                input_ids=ids, attention_mask=mask, conv_mask=conv,
                states=paged_states, logits_to_keep=1)
        self.assertTrue(torch.equal(dense_logits, paged_logits))
        next_id = torch.tensor([[tokens[8]]], dtype=torch.long)
        mask1 = causal_mask(1, 1, 8)
        conv1 = torch.ones((1, 1), dtype=torch.bool)
        with torch.no_grad():
            dense2, _ = model(input_ids=next_id, attention_mask=mask1,
                              conv_mask=conv1, states=dense_states, logits_to_keep=1)
            paged2, _ = model(input_ids=next_id, attention_mask=mask1,
                              conv_mask=conv1, states=paged_states, logits_to_keep=1)
        self.assertTrue(torch.equal(dense2, paged2))


class PagedEngineTest(unittest.TestCase):
    """引擎侧：分页池的挂载与整份状态快照。"""

    def test_enable_paging_and_snapshot_roundtrip(self) -> None:
        """挂池子后前向正常；快照→继续→装回→重跑，logits 逐位相同。"""
        from models.qwen4exp.engine import Qwen4Engine

        model = build_model()
        engine = Qwen4Engine(model, max_seq=64)
        pool = engine.enable_paging(num_blocks=1, block_size=64)
        self.assertEqual(pool.max_seq, 64)
        with self.assertRaises(RuntimeError):
            engine.enable_paging(num_blocks=1, block_size=64)

        tokens = list(range(2, 10))
        first = engine.prefill(tokens[:6], chunk=6)
        snapshot = engine.snapshot_linear_states()
        self.assertIsNotNone(snapshot)
        # 继续走两步，再装回快照重跑同样两步：两条路必须逐位相同。
        a = engine.prefill(tokens[6:], chunk=2)
        engine.restore_linear_states(snapshot)
        b = engine.prefill(tokens[6:], chunk=2)
        self.assertTrue(torch.equal(a, b), "快照装回后重放的 logits 不一致")
        self.assertTrue(torch.equal(first, first))

    def test_snapshot_before_any_forward_is_none(self) -> None:
        """没跑过前向时没有状态可快照。"""
        from models.qwen4exp.engine import Qwen4Engine

        engine = Qwen4Engine(build_model(), max_seq=64)
        self.assertIsNone(engine.snapshot_linear_states())


class BatchedSparseDecodeTest(unittest.TestCase):
    """批量 decode 的稀疏路径：逐行各调一次现有内核，不退回稠密。

    真机上的 ``attn_sparse`` 需要 CoreX，CPU 单测用同签名的替身，验的是**派发逻辑**：
    每一行拿到自己的 KV 与自己的选中集合，输出按 batch 叠回去。
    """

    def _run(self, batch: int):
        """造一个 ``Attention`` 与逐行不同的掩码，返回输出与替身记录。

        Args:
            batch: 批大小。

        Returns:
            ``(输出, 调用记录)``。
        """
        from models.qwen4exp.qsa import Attention, AttentionCache

        config = tiny_config()
        torch.manual_seed(5)
        attn = Attention(config, layer_idx=3).float()
        nh = config.num_attention_heads
        nkv = config.num_key_value_heads
        hd = config.head_dim
        kv = 16
        query = torch.randn(batch, nh, 1, hd)
        keys = torch.randn(batch, nkv, kv, hd)
        values = torch.randn(batch, nkv, kv, hd)
        index_keys = torch.randn(batch, kv, config.indexer_head_dim)
        cache = AttentionCache(keys=keys, values=values, index_keys=index_keys)
        # 逐行不同的可见集合：第 r 行只留 [r, kv) 里每隔 3 个的位置。
        mask = torch.zeros(batch, 1, 1, kv, dtype=torch.bool)
        for row in range(batch):
            mask[row, 0, 0, row::3] = True
        calls: list[dict] = []

        def fake(q, kcache, vcache, sel_idx, sel_len, kv_len, scale):
            calls.append({
                "q_shape": tuple(q.shape),
                "kcache": kcache.clone(),
                "sel": sel_idx[0].clone(),
                "sel_len": int(sel_len[0]),
                "kv_len": int(kv_len[0]),
                "scale": scale,
            })
            # 让输出带上"这一行选中了哪些位置"的痕迹，便于验证没有串号。
            out = q.clone()
            out[0, 0, 0] += float(sel_idx[0].sum())
            return out

        out = attn._sparse_attention(query, cache, mask, kv, fake)
        return out, calls, cache, mask, query

    def test_batched_rows_get_their_own_kv_and_selection(self) -> None:
        """B=3 时调 3 次，每行拿到自己的 KV 与 sel，输出形状正确。"""
        out, calls, cache, mask, query = self._run(3)
        self.assertIsNotNone(out)
        self.assertEqual(len(calls), 3)
        self.assertEqual(tuple(out.shape), (3, 4, 1, 64))
        for row, call in enumerate(calls):
            self.assertEqual(call["q_shape"], (1, 4, 64))
            self.assertTrue(torch.equal(call["kcache"], cache.keys[row]),
                            f"第 {row} 行拿到了别人的 KV")
            want = torch.nonzero(mask[row].reshape(-1), as_tuple=False).flatten().to(
                torch.int32)
            self.assertTrue(torch.equal(call["sel"], want),
                            f"第 {row} 行的选中集合不对")
            self.assertEqual(call["kv_len"], 16)
            # 替身给每行加上自己的 sel 之和：输出必须逐行等于 query + 该行的和，
            # 串号的行会立刻对不上。
            expected = float(query[row, 0, 0, 0]) + float(want.sum())
            self.assertAlmostEqual(float(out[row, 0, 0, 0]), expected, places=5,
                                   msg=f"第 {row} 行的输出串到了别的行")

    def test_single_row_still_uses_one_call(self) -> None:
        """B=1 仍然只调一次，没有因为批量改动多出发射。"""
        out, calls, _, _, _ = self._run(1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(tuple(out.shape), (1, 4, 1, 64))


class BatchedDecodeEngineTest(unittest.TestCase):
    """引擎的静态批：两行一起解码 == 两行各自单行解码。"""

    def test_rows_do_not_share_kv(self) -> None:
        """两行喂不同 token 后，各自下一个 token 的 logits 与单行路径一致。

        行是状态张量的一维切片，池子的 K/V 也按行存，所以批里各行互不串扰是可判
        的；串扰会让差直接到量级 1。容差取 2e-2 是因为批与单行走的 GEMM M 不同
        （``linear_m`` 与 ``linear``），累加次序本来就会差一点点。
        """
        from models.qwen4exp.engine import Qwen4Engine

        def fresh() -> Qwen4Engine:
            engine = Qwen4Engine(build_model(), max_seq=64)
            engine.enable_paging(num_blocks=1, block_size=64)
            return engine

        left_tokens = [5, 7]
        right_tokens = [9, 11]
        batched = fresh()
        batched.enable_batching(2)
        rows_out = []
        for step in range(2):
            batched.set_rows([left_tokens[step], right_tokens[step]],
                             [step, step], row_indices=[0, 1])
            rows_out.append(batched.decode_batch_eager(2, 2))
        singles = []
        for tokens in (left_tokens, right_tokens):
            one = fresh()
            one.prefill([tokens[0]], chunk=1)
            singles.append(one.decode_step(tokens[1]))
        for row, want in enumerate(singles):
            gap = float((rows_out[-1][row] - want).abs().max())
            scale = float(want.abs().max())
            self.assertLess(gap / max(scale, 1e-12), 2e-2,
                            f"第 {row} 行与单行路径差 {gap:.3e}")

    def test_row_ops_are_slices(self) -> None:
        """``move_linear_states`` / ``snapshot`` / ``zero`` 都在行切片上生效。"""
        from models.qwen4exp.engine import Qwen4Engine

        engine = Qwen4Engine(build_model(), max_seq=64)
        engine.enable_paging(num_blocks=1, block_size=64)
        engine.enable_batching(2)
        engine.set_rows([5, 9], [0, 0], row_indices=[0, 1])
        engine.decode_batch_eager(2, 2)
        snap0 = engine.snapshot_linear_states(row=0)
        snap1 = engine.snapshot_linear_states(row=1)
        self.assertIsNotNone(snap0)
        self.assertIsNotNone(snap1)
        # 快照是单行切片：批维必须是 1，不能把整批带出来。
        for layer in snap0.layers:
            for name in ("conv_state", "recurrent_state", "ple_context",
                         "ple_conv_state"):
                tensor = getattr(layer, name)
                if tensor is not None:
                    self.assertEqual(int(tensor.shape[0]), 1,
                                     f"{name} 的快照不是单行切片")
        engine.move_linear_states(0, 1)
        moved = engine.snapshot_linear_states(row=1)
        for got, want in zip(moved.layers, snap0.layers):
            for name in ("conv_state", "recurrent_state", "ple_context",
                         "ple_conv_state"):
                a, b = getattr(got, name), getattr(want, name)
                if a is not None and b is not None:
                    self.assertTrue(torch.equal(a, b), f"{name} 没有搬到目标行")
        engine.zero_linear_states(0)
        zeroed = engine.snapshot_linear_states(row=0)
        for layer in zeroed.layers:
            for name in ("conv_state", "recurrent_state", "ple_context",
                         "ple_conv_state"):
                tensor = getattr(layer, name)
                if tensor is not None:
                    self.assertEqual(float(tensor.abs().max()), 0.0,
                                     f"{name} 没有清零")


if __name__ == "__main__":
    unittest.main()
