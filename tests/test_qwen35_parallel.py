"""四卡张量分片的边界必须保留原矩阵的 packed 和 scale 对齐。

Module: ``coffin/tests``。纯 CPU 检查并行加载器的切片规则，不依赖设备或进程组。

Constraints:
    * 所有分片拼回去必须逐位等于原 packed 权重及 scale。
    * K 轴切分必须按组边界进行，避免尾 nibble 或 scale 移到相邻 rank。
"""

from __future__ import annotations

import unittest

import torch

from models.qwen35.parallel import local_config, shard_weight
from tools.quant_disk_w4a8 import quantize_projection
from w4a8_weights import W4A8Weight


class ParallelShardTest(unittest.TestCase):
    """QKV、MLP 与 K 分片的局部轴必须符合各自的并行方向。"""

    def setUp(self):
        self.cfg = {
            "num_attention_heads": 24, "num_key_value_heads": 4,
            "linear_num_key_heads": 16, "linear_num_value_heads": 48,
            "linear_key_head_dim": 128, "linear_value_head_dim": 128,
            "intermediate_size": 17408,
        }

    def test_model_dimensions_partition_evenly(self):
        local = local_config(self.cfg, 4)
        self.assertEqual(local["num_attention_heads"], 6)
        self.assertEqual(local["num_key_value_heads"], 1)
        self.assertEqual(local["linear_num_key_heads"], 4)
        self.assertEqual(local["linear_num_value_heads"], 12)
        self.assertEqual(local["intermediate_size"], 4352)
        self.assertEqual(self.cfg["num_attention_heads"], 24)

    def test_fused_linear_qkv_and_convolution_keep_matching_segments(self):
        widths = [2048, 2048, 6144]
        dense = torch.arange(sum(widths) * 3).reshape(sum(widths), 3)
        conv = torch.arange(sum(widths) * 4).reshape(sum(widths), 1, 4)
        for rank in range(4):
            got = shard_weight("model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
                               dense, self.cfg, rank, 4)
            cg = shard_weight("model.language_model.layers.0.linear_attn.conv1d.weight",
                              conv, self.cfg, rank, 4)
            expected = torch.cat([dense[a + rank * (width // 4):
                                        a + (rank + 1) * (width // 4)]
                                  for a, width in ((0, 2048), (2048, 2048),
                                                   (4096, 6144))])
            expected_conv = torch.cat([conv[a + rank * (width // 4):
                                            a + (rank + 1) * (width // 4), 0]
                                       for a, width in ((0, 2048), (2048, 2048),
                                                        (4096, 6144))])
            self.assertTrue(torch.equal(got, expected))
            self.assertTrue(torch.equal(cg[:, 0], expected_conv))

    def test_packed_output_and_input_shards_reassemble_exactly(self):
        source = torch.arange(64 * 512, dtype=torch.float32).reshape(64, 512).sin()
        packed, scales = quantize_projection(source, group_size=128)
        full = W4A8Weight(packed, scales, k=512, group_size=128)
        output_parts, input_parts = [], []
        for rank in range(4):
            output_parts.append(shard_weight(
                "model.language_model.layers.0.mlp.gate_proj.weight",
                full, self.cfg, rank, 4))
            input_parts.append(shard_weight(
                "model.language_model.layers.0.mlp.down_proj.weight",
                full, self.cfg, rank, 4))
        self.assertTrue(torch.equal(torch.cat([x.packed for x in output_parts], 1), packed))
        self.assertTrue(torch.equal(torch.cat([x.scale for x in output_parts], 1), scales))
        self.assertTrue(torch.equal(torch.cat([x.packed for x in input_parts], 0), packed))
        self.assertTrue(torch.equal(torch.cat([x.scale for x in input_parts], 0), scales))


if __name__ == "__main__":
    unittest.main()
