"""W4 离线权重格式的打包、重建和 checkpoint 往返测试。

Module: ``coffin/tests``。只用 CPU 和 ``safetensors`` 验证磁盘格式，不要求 CoreX
驱动或模型 checkpoint。

Constraints:
    * 解包后必须与索引记载的 K、group size 和逐组 scale 对齐；错一个 nibble 都会
      污染整条投影，却可能继续产出有限 logits。
    * 输出端拒绝覆盖已有文件，测试临时目录与 W8 往返测试保持隔离。
"""

from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

import support

import torch
from tools import quant_disk_w4a8

try:
    from safetensors import safe_open
    from safetensors.torch import save_file
    _HAS_SAFETENSORS = True
except Exception:
    _HAS_SAFETENSORS = False

SCRATCH = Path(__file__).resolve().parents[1] / ".tmp_purity" / "quant_w4a8"


def _scratch(name: str) -> Path:
    path = SCRATCH / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def tearDownModule() -> None:
    shutil.rmtree(SCRATCH, ignore_errors=True)


class ProjectionFormatTest(unittest.TestCase):
    """打包规则必须可逆到量化值，含分组尾部和奇数 K。"""

    def test_low_nibble_is_even_k_and_high_nibble_is_odd_k(self):
        w = torch.tensor([[-7.0, 7.0, -3.5, 3.5]], dtype=torch.float32)
        packed, scales = quant_disk_w4a8.quantize_projection(w, group_size=2)
        self.assertEqual(packed.tolist(), [[0xF1], [0xF1]])
        self.assertTrue(torch.equal(scales, torch.tensor([[1.0], [0.5]])))
        got = quant_disk_w4a8.dequantize_projection(packed, scales, k=4,
                                                     group_size=2)
        self.assertTrue(torch.equal(got, w))

    def test_odd_k_and_partial_group_round_trip(self):
        w = torch.tensor([[1.0, -2.0, 3.0, -4.0, 5.0],
                          [-2.0, 4.0, -6.0, 8.0, -10.0]])
        packed, scales = quant_disk_w4a8.quantize_projection(w, group_size=3)
        self.assertEqual(tuple(packed.shape), (3, 2))
        self.assertEqual(tuple(scales.shape), (2, 2))
        got = quant_disk_w4a8.dequantize_projection(packed, scales, k=5,
                                                     group_size=3)
        want = w.clone()
        want[:, :3] = torch.round(w[:, :3] / (w[:, :3].abs().amax(1, keepdim=True) / 7)) \
            .clamp(-8, 7) * (w[:, :3].abs().amax(1, keepdim=True) / 7)
        tail_scale = w[:, 3:].abs().amax(1, keepdim=True) / 7
        want[:, 3:] = torch.round(w[:, 3:] / tail_scale).clamp(-8, 7) * tail_scale
        self.assertTrue(torch.equal(got, want))

    def test_invalid_shape_or_group_size_is_rejected(self):
        with self.assertRaises(ValueError):
            quant_disk_w4a8.quantize_projection(torch.ones(4), 128)
        with self.assertRaises(ValueError):
            quant_disk_w4a8.quantize_projection(torch.ones(2, 4), 0)


@unittest.skipUnless(_HAS_SAFETENSORS, "needs the safetensors package")
class CheckpointRoundTripTest(unittest.TestCase):
    """量化 shard 与 sidecar 组成可审计的独立产物，原 checkpoint 不变。"""

    def setUp(self) -> None:
        self.root = _scratch("roundtrip")
        self.addCleanup(shutil.rmtree, self.root, True)
        self.ckpt = self.root / "checkpoint"
        self.ckpt.mkdir()
        (self.ckpt / "config.json").write_text(
            json.dumps({"model_type": "qwen3_5"}), encoding="utf-8")
        self.tensors = {
            "model.language_model.layers.0.mlp.down_proj.weight":
                torch.randn(10, 12, dtype=torch.bfloat16),
            "model.language_model.norm.weight": torch.randn(12, dtype=torch.bfloat16),
            "model.visual.patch_embed.weight": torch.randn(4, 8, dtype=torch.bfloat16),
        }
        save_file(self.tensors, str(self.ckpt / "model-00001.safetensors"))
        self.out = self.root / "w4"

    def _convert(self, *, include_vision: bool = False) -> dict:
        return quant_disk_w4a8.convert(
            self.ckpt, self.out, weight_dtype=torch.bfloat16,
            device=torch.device("cpu"), prefixes=quant_disk_w4a8.quant_disk.DEFAULT_PREFIXES,
            include_vision=include_vision, copy_sidecars=True, group_size=4)

    def test_manifest_and_shard_record_the_packed_layout(self):
        index = self._convert(include_vision=True)
        key = "model.language_model.layers.0.mlp.down_proj.weight"
        meta = index["tensors"][key]
        self.assertEqual(meta["kind"], "q4")
        self.assertEqual(meta["shape"], [10, 12])
        self.assertEqual(meta["packed_shape"], [6, 10])
        self.assertEqual(meta["scale_shape"], [3, 10])
        self.assertTrue(index["vision"])
        shard = next(iter(index["files"]))
        with safe_open(str(self.out / shard), framework="pt", device="cpu") as handle:
            packed = handle.get_tensor(key)
            scales = handle.get_tensor(meta["scale_key"])
            visual = handle.get_tensor("model.visual.patch_embed.weight")
        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(scales.dtype, torch.float32)
        self.assertTrue(torch.equal(visual, self.tensors["model.visual.patch_embed.weight"]))
        restored = quant_disk_w4a8.dequantize_projection(packed, scales, k=12,
                                                          group_size=4)
        self.assertLess((restored - self.tensors[key].float()).abs().max().item(), 0.3)

    def test_vision_is_skipped_by_default_and_source_is_unchanged(self):
        source_before = (self.ckpt / "model-00001.safetensors").read_bytes()
        index = self._convert()
        self.assertFalse(index["vision"])
        self.assertNotIn("model.visual.patch_embed.weight", index["tensors"])
        self.assertEqual((self.ckpt / "model-00001.safetensors").read_bytes(), source_before)

    def test_nonempty_output_directory_is_never_overwritten(self):
        self.out.mkdir()
        sentinel = self.out / "keep.txt"
        sentinel.write_text("user data", encoding="utf-8")
        with self.assertRaises(SystemExit):
            self._convert()
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "user data")


if __name__ == "__main__":
    unittest.main()
