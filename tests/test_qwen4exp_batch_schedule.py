"""批调度接入：``LLMEngine`` 的批路径能不能真的驱动 qwen4exp。

判据不是"引擎自己有 ``set_rows`` / ``decode_batch_eager``"，而是 ``LLMEngine``
既有的 ``begin_prefill`` / ``_run_prefill`` / ``_decode_batch`` 这条链路能跑完两条
请求。这条用例守的是三处曾经缺失的接口：

* ``decode_capacity``：``model_runner._claim_row`` 读它决定能放几条序列。写死成 1
  时第二个请求报 ``no free decode row (capacity 1)``，看起来像调度器越界准入；
* 桶宽与容量的关系：runner 按桶（``max(active_row)+1``）传宽，而状态是按容量开的，
  所以必须"按容量跑、只返回前 bucket 行"；要求两者相等会让每个 decode 步都失败；
* ``prefill_group``：分组预填充要能收多条等长 prompt。

模块级限制写在 ``Qwen4Engine`` 的文件头：分页池只有一个共享游标，所以批里各行的
上下文必须等长。**分块预填充（每行各自游标）因此不在这条路径上**，
``model_runner.prefill_chunk`` 会撞上"批维 3 对 1"的形状错。
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
    """4 层极小配置（3 个 GDN 层 + 1 个 QSA 层）。

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


def make_engine(seed: int = 23) -> LLMEngine:
    """建一个两行的批调度引擎（无前缀缓存，纯批路径）。

    Args:
        seed: 权重随机种子。

    Returns:
        已 ``build()`` 的引擎。
    """
    torch.manual_seed(seed)
    model = Qwen4Engine(CausalLM(tiny_config(), stream_ngram=True,
                                 with_mtp=False).float(), max_seq=128)
    cfg = EngineConfig(model_dir="", max_seq=128, graph=False, prefix_cache=False,
                       max_num_seqs=2, num_blocks=8, block_size=8,
                       snapshots=2, snapshot_interval=8, chunk=32,
                       prefill_quantum=32, max_num_batched_tokens=64,
                       bench=False, quant=False, vision=False)
    engine = LLMEngine(cfg, model)
    engine.build()
    return engine


class BatchScheduleTest(unittest.TestCase):
    """两条请求走完整批路径。"""

    def test_two_requests_finish_through_the_runner(self) -> None:
        """分组预填充 + 一步批解码，两条请求都正常结束并各出两个 token。"""
        engine = make_engine()
        first = engine.add_request(list(range(2, 10)),
                                   SamplingParams(max_tokens=2, eos=set()))
        second = engine.add_request(list(range(20, 28)),
                                    SamplingParams(max_tokens=2, eos=set()))
        produced: dict[int, list[int]] = {first.seq_id: [], second.seq_id: []}
        for _ in range(12):
            for seq, token, _finished in engine.step():
                if token is not None:
                    produced[seq.seq_id].append(int(token))
            if all(seq.status.name in ("FINISHED", "ABORTED")
                   for seq in (first, second)):
                break
        self.assertEqual(len(produced[first.seq_id]), 2)
        self.assertEqual(len(produced[second.seq_id]), 2)
        # 两条 prompt 不同，逐位相同就说明批里串行了。
        self.assertNotEqual(produced[first.seq_id], produced[second.seq_id])

    def test_chunked_prefill_is_not_on_this_path(self) -> None:
        """分块预填充在批路径上明确不支持：状态按容量开，单行前向对不上批维。

        这个用例把那句"不支持"钉成行为，而不是让它以后变成一条难查的形状错误。
        """
        engine = make_engine(seed=29)
        engine.add_request(list(range(2, 42)), SamplingParams(max_tokens=1,
                                                              eos=set()))
        with self.assertRaises(RuntimeError):
            for _ in range(20):
                engine.step()


if __name__ == "__main__":
    unittest.main()
