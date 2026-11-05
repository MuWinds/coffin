"""qwen4exp 接进调度与前缀缓存的端到端冒烟（CPU 小配置）。

判据是三条，缺一条都不算"接上了"：

1. ``LLMEngine`` 能把 ``BlockManager`` / ``RadixCache`` / ``PrefixCacheEngine``
   挂到 :class:`Qwen4KVPool` 上而不报 ``NotImplementedError``；
2. 两个共享前缀的请求，第二个的 ``last_matched`` 必须 > 0（真的命中了块边界的
   前缀），并且命中点上恢复的是整份状态（GDN/PLE/QSA 一起）；
3. 命中前缀后生成的 token 与"从头重算同一个 prompt"逐位相同——混合模型只恢复
   KV 不恢复线性状态会静默给出另一个答案，这条能把它抓住。

Module: ``coffin/tests``。纯 CPU，不碰加速卡，也不需要 checkpoint 目录。
"""

from __future__ import annotations

import unittest

import torch

from engine.config import EngineConfig
from engine.llm_engine import LLMEngine
from engine.sequence import SamplingParams
from models.qwen4exp.config import TextConfig
from models.qwen4exp.engine import Qwen4Engine
from models.qwen4exp.model import CausalLM


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


def make_engine() -> LLMEngine:
    """按固定种子建一个开了前缀缓存的 LLMEngine。

    Returns:
        已 ``build()`` 的引擎；CPU、eager（平台禁图）。
    """
    torch.manual_seed(17)
    model = Qwen4Engine(CausalLM(tiny_config(), stream_ngram=True,
                                 with_mtp=False).float(), max_seq=256)
    cfg = EngineConfig(model_dir="", max_seq=256, graph=False, prefix_cache=True,
                       max_num_seqs=1, num_blocks=4, block_size=8,
                       snapshots=4, snapshot_interval=8, chunk=64,
                       prefill_quantum=256, bench=False, quant=False,
                       vision=False)
    engine = LLMEngine(cfg, model)
    engine.build()
    return engine


class PrefixCacheIntegrationTest(unittest.TestCase):
    """调度 / 基数树 / 整份状态快照三者接起来。"""

    def test_pool_is_a_real_block_pool(self) -> None:
        """``BlockManager.attach`` 拿到的是能记账的池，不是描述字典。"""
        engine = make_engine()
        self.assertIsNotNone(engine.bm)
        self.assertIsNotNone(engine.pc)
        self.assertEqual(engine.bm.pool.bs, 8)
        self.assertEqual(engine.bm.pool.num_blocks, 4)
        # build() 预留了一个预热块，所以它确实从池里取过引用。
        self.assertGreater(engine.bm.pool.n_allocs, 0)

    def test_second_request_reuses_prefix(self) -> None:
        """两个共享前缀的请求：第二个命中，且生成的 token 与从头算一致。"""
        engine = make_engine()
        prompt = list(range(2, 20))              # 18 token，跨过两个 8-token 块
        params = SamplingParams(max_tokens=3, eos=set())

        first = engine.add_request(prompt, SamplingParams(max_tokens=3, eos=set()))
        tokens1 = self._collect(engine, first, 3)

        second = engine.add_request(prompt, params)
        tokens2 = self._collect(engine, second, 3)
        # prefill 在第一次 step() 里跑，之后 last_matched 才是第二次请求的命中数。
        matched = engine.pc.last_matched
        self.assertGreater(matched, 0,
                           "第二个请求没有命中任何块边界前缀")
        self.assertEqual(tokens1, tokens2,
                         "命中前缀后生成的 token 与从头算不一致")

    @staticmethod
    def _collect(engine: LLMEngine, seq, n: int) -> list[int]:
        """驱动 ``engine.step()`` 直到收满 ``n`` 个 token 或序列结束。

        Args:
            engine: 引擎。
            seq: 要观察的序列。
            n: 最多收多少 token。

        Returns:
            该序列产出的 token 列表。
        """
        out: list[int] = []
        for _ in range(200):
            events = engine.step()
            for got_seq, token, _finished in events:
                if got_seq is seq and token is not None:
                    out.append(int(token))
            if len(out) >= n or seq.status.name in ("FINISHED", "ABORTED"):
                break
        return out[:n]


if __name__ == "__main__":
    unittest.main()
