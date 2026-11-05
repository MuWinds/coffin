"""引擎的包结构、模型族注册表与 reload 行为。

Module: ``coffin/tests``。只导入 ``engine_api`` 与 ``support``，不读权重、不跑
内核，所以在没有加速卡的机器上也能跑。注册表相关的断言只看「登记项的形状」，
不写死某个模型族：模型族会增删，钉死键名的测试会跟着一起过时。

Constraints:
    * 模型族的导入留在 ``engine_api`` 背后；导入本文件不能在纯 CPU 主机上把已
      编译的内核扩展带进来，所以这里不 import ``models`` / ``backends``。
    * 分派测试用临时登记的替身族。每个用例在 ``finally`` 里还原注册表，泄漏的
      登记会让别的模块看到一份多出来的架构。
    * 临时配置目录放在 ``tests/.tmp_structure``，不用 ``tempfile``：Windows 上
      ``mkdtemp`` 建的目录带沙箱令牌的 DACL，删不掉也写不进去。
"""

from __future__ import annotations

import contextlib
import json
import shutil
import unittest
from pathlib import Path
from typing import Callable

# 导入本模块只为它的两个副作用：导出 MKL 线程层，并把 ``coffin/`` 放进
# ``sys.path``。这里的 ``# noqa`` 压的正是「导入了却没用到」那条告警。
import support  # noqa: F401

import engine_api

#: 替身族的注册键。带前缀是为了不会撞上任何真实模型族的 ``model_type`` 子串。
DUMMY_KEY = "engine_api_test_family"

#: 临时目录根。随目录跟着 ``tests/``，删得掉，且被 ``.gitignore`` 忽略。
TMP_ROOT = Path(__file__).resolve().parent / ".tmp_structure"

#: 构造器替身记录到的最近一次调用，``(位置参数, 关键字参数)``。
_MODEL_CALLS: list[tuple[tuple, dict]] = []
#: 加载器替身记录到的最近一次调用，形式同上。
_LOADER_CALLS: list[tuple[tuple, dict]] = []


class _RecorderModel:
    """记录 ``create_model`` 转交了哪些参数，本身不做任何事。"""

    def __init__(self, *args, **kwargs) -> None:
        _MODEL_CALLS.append((args, kwargs))


def _recorder_loader(*args, **kwargs) -> dict:
    """记录 ``load_weights`` 转交了哪些参数，返回一个可断言的占位结果。"""
    _LOADER_CALLS.append((args, kwargs))
    return {"ok": True}


@contextlib.contextmanager
def _temporary_entry(model: type, loader: Callable[..., dict]):
    """临时把 ``DUMMY_KEY`` 指向给定的类与加载器，退出时还原。"""
    previous = engine_api._REGISTRY.get(DUMMY_KEY)
    engine_api.register((DUMMY_KEY,), model, loader)
    try:
        yield
    finally:
        if previous is None:
            engine_api._REGISTRY.pop(DUMMY_KEY, None)
        else:
            engine_api._REGISTRY[DUMMY_KEY] = previous


def _write_config(name: str, model_type: str) -> Path:
    """在临时根下造一个只含 ``config.json`` 的 checkpoint 目录。"""
    root = TMP_ROOT / name
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps({"model_type": model_type}),
                                      encoding="utf-8")
    return root


def setUpModule() -> None:
    """写出两个配置目录：一个指向替身族，一个指向未登记的族。"""
    global _DISPATCH_DIR, _OVERRIDE_DIR, _REGISTERED_DIR
    _DISPATCH_DIR = _write_config("dispatch", DUMMY_KEY)
    _OVERRIDE_DIR = _write_config("override", "definitely-not-registered")
    # 用当前构建登记的第一个键：它的配置形状与这份代码树一致，又不需要写死名字。
    _REGISTERED_DIR = _write_config("registered", engine_api.registered()[0])


def tearDownModule() -> None:
    """删掉临时目录，避免它成为下一次扫描的输入。"""
    shutil.rmtree(TMP_ROOT, ignore_errors=True)


class RegistryTest(unittest.TestCase):
    """注册表是 checkpoint 到实现的唯一分派点，形状错了就只能加载错误的权重。"""

    def test_registered_keys_are_sorted_and_lowercase(self):
        """键要能被 ``architecture=`` 原样传回，所以只能是小写且有序的。"""
        keys = engine_api.registered()
        self.assertTrue(keys)
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(keys, [key.lower() for key in keys])

    def test_every_entry_lives_in_a_model_package(self):
        """每个登记项都得是指向 ``models/`` 下某个族的类和加载器。

        指向包外或者 ``None`` 时，``create_model`` 会在服务启动那一刻才炸，而
        报错信息不会指出是哪个族登记错了。
        """
        for key, entry in engine_api._REGISTRY.items():
            self.assertTrue(entry.model.__module__.startswith("models."),
                            f"{key}: {entry.model!r}")
            self.assertTrue(callable(entry.load_weights), f"{key}: loader is not callable")

    def test_detection_is_case_insensitive_and_stays_registered(self):
        """``config.json`` 的大小写不统一；检测结果必须是已登记的键。"""
        keys = engine_api.registered()
        for key in keys:
            for spelled in (key, key.upper()):
                detected = engine_api.detect_architecture({"architectures": [spelled]})
                self.assertIn(detected, keys)
                self.assertIn(detected, spelled.lower())

    def test_detection_reads_nested_text_config(self):
        """多模态 checkpoint 只在 ``text_config`` 里点文本模型的名。"""
        keys = engine_api.registered()
        for key in keys:
            detected = engine_api.detect_architecture(
                {"text_config": {"model_type": key}})
            self.assertIn(detected, keys)
            self.assertIn(detected, key)

    def test_unknown_configuration_raises_with_the_known_keys(self):
        """未知架构必须直接报错；静默回退会拿错实现去服务一个 checkpoint。

        报错里必须列出可用键，否则调用方只能去读注册表源码。
        """
        with self.assertRaises(RuntimeError) as caught:
            engine_api.detect_architecture({"model_type": "definitely-not-registered"})
        for key in engine_api.registered():
            self.assertIn(key, str(caught.exception))

    def test_describe_reports_path_basename_and_architecture(self):
        """启动日志与 ``/v1/models`` 都读 ``describe``，它的三个键是公开契约。"""
        info = engine_api.describe(_REGISTERED_DIR)
        self.assertEqual(info, {"model_path": str(_REGISTERED_DIR),
                                "model": _REGISTERED_DIR.name,
                                "arch": engine_api.registered()[0]})


class DispatchTest(unittest.TestCase):
    """``create_model`` / ``load_weights`` 必须把每个开关原样交给模型族。

    自己再套一层默认值会让调用方显式传的 ``max_seq`` / ``paged`` 被悄悄丢弃，
    而模型照常加载、只是行为不对。
    """

    def test_create_model_forwards_every_switch(self):
        """分派必须完整转交构造参数，包括显式传入的 ``False`` 与 ``None``。"""
        _MODEL_CALLS.clear()
        with _temporary_entry(_RecorderModel, _recorder_loader):
            model = engine_api.create_model(_DISPATCH_DIR, max_seq=8192,
                                            bench=False, paged=True)
        self.assertIsInstance(model, _RecorderModel)
        args, kwargs = _MODEL_CALLS[-1]
        self.assertEqual(Path(args[0]), _DISPATCH_DIR)
        self.assertEqual(kwargs, {"max_seq": 8192, "bench": False, "weights": None,
                                  "paged": True, "quant": False})

    def test_load_weights_forwards_every_switch(self):
        """加载器的开关同样逐个转交；``weight_dtype=None`` 不能变成显式参数。"""
        sentinel = object()
        _LOADER_CALLS.clear()
        with _temporary_entry(_RecorderModel, _recorder_loader):
            weights = engine_api.load_weights(_DISPATCH_DIR, quant=True,
                                              weight_dtype=sentinel, vision=False,
                                              vision_quant=True)
        self.assertEqual(weights, {"ok": True})
        args, kwargs = _LOADER_CALLS[-1]
        self.assertEqual(Path(args[0]), _DISPATCH_DIR)
        self.assertEqual(kwargs, {"quant": True, "vision": False,
                                  "vision_quant": True, "weight_dtype": sentinel})

    def test_explicit_architecture_overrides_detection(self):
        """元数据说谎的 checkpoint 靠 ``architecture=`` 覆盖；覆盖值同样要查表。"""
        with _temporary_entry(_RecorderModel, _recorder_loader):
            model = engine_api.create_model(_OVERRIDE_DIR, architecture=DUMMY_KEY)
        self.assertIsInstance(model, _RecorderModel)

    def test_unknown_architecture_override_is_rejected(self):
        """覆盖值没登记时必须报错，不能悄悄退回检测结果。"""
        with self.assertRaises(RuntimeError):
            engine_api.create_model(_OVERRIDE_DIR, architecture="definitely-not-registered")


class ReloadTest(unittest.TestCase):
    """reload 之后注册表必须指向重新导入的类，否则调用方拿到的是旧类。"""

    def test_reload_rebuilds_registered_classes(self):
        """旧类对象留在注册表里、调用方却用新导入的类，这种类型不匹配只在运行时炸。"""
        key = engine_api.registered()[0]
        before = engine_api._REGISTRY[key].model
        engine_api.reload()
        after = engine_api._REGISTRY[key].model
        self.assertEqual(after.__module__, before.__module__)
        self.assertIsNot(after, before)


if __name__ == "__main__":
    unittest.main()
