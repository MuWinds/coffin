"""防止旧版 AutoProcessor 把纯分词器缓存为可用视觉处理器。

Module: coffin/tests。通过运行时 mixin 验证惰性处理器加载，不构建 GPU 模型。

Constraints:
    * 兼容路径只能用于 checkpoint 明确声明的静态图片算法。
    * 加载失败后必须保留 503 语义，不能让非主 rank 因 AttributeError 退出。
"""

import json
import shutil
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from serving.runtime_parts import RuntimeChatMixin
from serving.runtime_support import MMError

#: 临时配置目录。不用 ``tempfile``：Windows 上 ``mkdtemp`` 建的目录带沙箱令牌的
#: DACL，写不进也删不掉，整个模块会在 ``setUp`` 里直接报 ``PermissionError``。
TMP_ROOT = Path(__file__).resolve().parent / ".tmp_runtime_processor"


class RuntimeProcessorTest(unittest.TestCase):
    """处理器能力校验必须在发布缓存之前完成。"""

    def setUp(self):
        shutil.rmtree(TMP_ROOT, ignore_errors=True)
        TMP_ROOT.mkdir(parents=True, exist_ok=True)
        self.addCleanup(shutil.rmtree, TMP_ROOT, ignore_errors=True)
        self.temp = SimpleNamespace(name=TMP_ROOT)
        self.server = SimpleNamespace(
            model_dir=self.temp.name, processor=None, _processor_failed=None,
            tok=object(), image_min_pixels=65536, image_max_pixels=1048576)
        self.transformers = SimpleNamespace(
            AutoProcessor=Mock(), AutoImageProcessor=Mock(), Qwen2VLProcessor=Mock())
        self.patcher = patch.dict(sys.modules, {"transformers": self.transformers})
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def _config(self, processor_class="Qwen3VLProcessor"):
        Path(self.temp.name, "preprocessor_config.json").write_text(json.dumps({
            "processor_class": processor_class,
            "image_processor_type": "Qwen2VLImageProcessorFast",
        }), encoding="utf-8")

    def test_native_processor_is_cached_without_compatibility_loading(self):
        """原生处理器直接复用，不需要触碰旧版图片组件。"""
        native = SimpleNamespace(image_processor=object())
        self.transformers.AutoProcessor.from_pretrained.return_value = native
        self.assertIs(RuntimeChatMixin._ensure_processor(self.server), native)
        self.assertIs(RuntimeChatMixin._ensure_processor(self.server), native)
        self.transformers.AutoProcessor.from_pretrained.assert_called_once()
        self.transformers.AutoImageProcessor.from_pretrained.assert_not_called()

    def test_tokenizer_fallback_preserves_tokenizer_and_pixel_limits(self):
        """兼容图片必须与服务模板使用同一个分词器和尺寸预算。"""
        self._config()
        self.transformers.AutoProcessor.from_pretrained.return_value = object()
        image = object()
        self.transformers.AutoImageProcessor.from_pretrained.return_value = image
        processor = SimpleNamespace(image_processor=image)
        self.transformers.Qwen2VLProcessor.return_value = processor
        self.assertIs(RuntimeChatMixin._ensure_processor(self.server), processor)
        self.transformers.AutoImageProcessor.from_pretrained.assert_called_once_with(
            self.temp.name, use_fast=True, min_pixels=65536, max_pixels=1048576)
        self.transformers.Qwen2VLProcessor.assert_called_once_with(
            image_processor=image, tokenizer=self.server.tok)

    def test_unrecognized_fallback_is_rejected_and_failure_is_cached(self):
        """未知视觉模型不能套用 Qwen 算法，也不能缓存纯分词器。"""
        self._config("OtherProcessor")
        self.transformers.AutoProcessor.from_pretrained.return_value = object()
        for _ in range(2):
            with self.assertRaises(MMError) as raised:
                RuntimeChatMixin._ensure_processor(self.server)
            self.assertEqual(raised.exception.status, 503)
        self.assertIsNone(self.server.processor)
        self.transformers.AutoProcessor.from_pretrained.assert_called_once()

    def test_image_loader_failure_does_not_publish_partial_processor(self):
        """组件加载抛错后仍走受控的 503，不留下半初始化状态。"""
        self._config()
        self.transformers.AutoProcessor.from_pretrained.return_value = object()
        self.transformers.AutoImageProcessor.from_pretrained.side_effect = ValueError("invalid")
        with self.assertRaises(MMError):
            RuntimeChatMixin._ensure_processor(self.server)
        self.assertIsNone(self.server.processor)
        self.assertIn("invalid", self.server._processor_failed)


if __name__ == "__main__":
    unittest.main()
