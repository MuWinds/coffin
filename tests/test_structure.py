"""引擎的包结构、架构分派与 reload 行为。

Module: ``coffin/tests``。只导入 ``engine_api`` 与 ``support``；它是唯一必须
在没有 DCU 的机器上跑通的测试模块，所以模块作用域里不许导入 ``hip_ops`` 或
某个模型族。

Constraints:
    * 不读权重、不跑内核，因此改完代码先跑这个文件，确认包结构没坏，再去付
      加载模型的时间。
    * 模型族的导入留在 ``engine_api`` 背后；导入这个测试不能在纯 CPU 主机上
      把已编译的内核扩展带进来。

测试直接钉住包注册表。生产导入图里没有扁平的兼容模块。
"""

from __future__ import annotations

import json
import unittest

import support

import engine_api

QWEN_CONFIG = {"architectures": ["Qwen3_5ForConditionalGeneration"],
               "text_config": {"model_type": "qwen3_5_text"}}


class PackageTest(unittest.TestCase):
    """注册表必须通过新包布局暴露模型族。"""

    def test_registered_model_is_the_qwen35_package_model(self):
        self.assertEqual(engine_api._REGISTRY["qwen3_5"].model.__module__,
                         "models.qwen35.model")

    def test_reload_rebuilds_the_registered_package_objects(self):
        """按 ``_RELOAD_ORDER`` 重建包之后，注册表必须指向新类而不是旧类。

        旧类对象留在注册表里，调用方却拿到新导入的类，这种类型不匹配只会在
        运行时才炸。
        """
        before = engine_api._REGISTRY["qwen3_5"].model
        engine_api.reload()
        after = engine_api._REGISTRY["qwen3_5"].model
        self.assertEqual(after.__module__, "models.qwen35.model")
        self.assertIsNot(before, after)


class RegistryTest(unittest.TestCase):
    """checkpoint 配置到实现的映射，也是唯一与模型相关的分派点。"""

    def test_qwen35_is_registered(self):
        self.assertIn("qwen3_5", engine_api.registered())

    def test_detect_from_model_type_and_architectures(self):
        self.assertEqual(engine_api.detect_architecture(QWEN_CONFIG), "qwen3_5")

    def test_detect_from_text_geometry_when_metadata_is_missing(self):
        self.assertEqual(
            engine_api.detect_architecture({"text_config": {"linear_num_value_heads": 16}}),
            "qwen3_5")

    def test_unknown_architecture_raises_instead_of_guessing(self):
        """未知模型族必须直接报错；静默回退会加载错误的权重。"""
        with self.assertRaises(RuntimeError):
            engine_api.detect_architecture({"model_type": "llama"})

    def test_describe_reports_name_and_family(self):
        """启动日志和 ``/v1/models`` 都读 ``describe``，它的键因此算一份小型公开契约。"""
        info = engine_api.describe(support.FIXTURE_MODEL_DIR)
        self.assertEqual(info["model"], "qwen3_5")
        self.assertEqual(info["arch"], "qwen3_5")
        self.assertEqual(info["model_path"], str(support.FIXTURE_MODEL_DIR))

    def test_create_model_dispatches_and_forwards_arguments(self):
        """``create_model`` 必须把每个开关原样转交模型族构造器。

        自己再套一层默认值会让调用方显式传的 ``max_seq`` / ``paged`` 被悄悄
        丢弃。
        """
        seen: dict = {}

        class _Dummy:
            # ``quant`` 只在这次调用自己加载 checkpoint 时才转交；仍把它留在
            # 签名里，是为了将来给 ``create_model`` 加了开关却没同步转交时，
            # 这个测试就失败，而不是等到调用点才炸。
            def __init__(self, model_dir, max_seq=4096, bench=True,
                         weights=None, paged=False, quant=False):
                seen["ctor"] = (model_dir, max_seq, bench, weights, paged, quant)

        real = engine_api._REGISTRY["qwen3_5"]
        engine_api._REGISTRY["qwen3_5"] = engine_api._Entry(_Dummy, real.load_weights)
        try:
            model = engine_api.create_model(support.FIXTURE_MODEL_DIR, max_seq=8192,
                                            bench=False, paged=True)
        finally:
            engine_api._REGISTRY["qwen3_5"] = real
        self.assertIsInstance(model, _Dummy)
        self.assertEqual(seen["ctor"][1:], (8192, False, None, True, False))

    def test_load_weights_dispatches_to_the_family_loader(self):
        """钉住每个模型族都必须接受的加载器签名。

        分派加了参数而某个族没跟上时，坏在加载，不坏在这个测试自己的调用。
        """
        seen: dict = {}

        # 模型族加载器的契约：位置参数 ``model_dir``，加上文本量化的开关和视觉
        # 那几个开关，都带默认值，纯文本族可以忽略它们。视觉相关的关键字参数
        # 随多模态改造加入，这个测试负责让分派和契约对所有族都保持同步，而不
        # 只是 qwen35。
        def _fake_loader(model_dir, quant=False, weight_dtype=None,
                         vision=None, vision_quant=False):
            seen["call"] = (model_dir, quant, weight_dtype, vision, vision_quant)
            return {"ok": True}

        real = engine_api._REGISTRY["qwen3_5"]
        engine_api._REGISTRY["qwen3_5"] = engine_api._Entry(real.model, _fake_loader)
        try:
            weights = engine_api.load_weights(support.FIXTURE_MODEL_DIR, quant=True)
        finally:
            engine_api._REGISTRY["qwen3_5"] = real
        self.assertEqual(weights, {"ok": True})
        self.assertEqual(seen["call"][1:], (True, None, None, False))


@unittest.skipUnless(support.has_model(), f"no checkpoint at {support.MODEL_DIR}")
class RealCheckpointTest(unittest.TestCase):
    """仓库里带的 fixture 只有配置形状，所以必须有一个测试去读真的 ``config.json``。"""

    def test_dispatch_real_checkpoint(self):
        """分派必须认得部署 checkpoint 自己的元数据，而不只是 fixture。"""
        config = json.loads((support.MODEL_DIR / "config.json").read_text())
        self.assertEqual(engine_api.detect_architecture(config), "qwen3_5")
        self.assertEqual(engine_api.describe(support.MODEL_DIR)["arch"], "qwen3_5")


if __name__ == "__main__":
    unittest.main()
