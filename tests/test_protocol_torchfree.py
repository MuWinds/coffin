"""``backends/protocol.py`` 在没有 torch 的机器上可导入：契约层不再是 torch 硬依赖。

Module: ``coffin/tests``。用 ``importlib`` 按文件路径单独装载 ``protocol.py``，
绕开 ``backends/__init__.py``（那里面还有 ``vendor`` 的 torch 导入，属于另一件事）。
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import support  # noqa: F401 - 侧效：把 coffin/ 放进 sys.path

from cxrt.dtypes import F32

PROTOCOL = Path(support.ENGINE_DIR) / "backends" / "protocol.py"


def load_standalone():
    """不经过 ``backends`` 包装载协议模块。

    Returns:
        模块对象。
    """
    spec = importlib.util.spec_from_file_location("protocol_standalone", PROTOCOL)
    mod = importlib.util.module_from_spec(spec)
    prev = sys.modules.get("protocol_standalone")
    sys.modules["protocol_standalone"] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        if prev is None:
            del sys.modules["protocol_standalone"]
        else:
            sys.modules["protocol_standalone"] = prev
        raise
    return mod


class ProtocolTorchFreeTest(unittest.TestCase):
    """没有 torch 时契约仍然成立：dtype 退到 ``cxrt``，设备退到 ``DeviceRef``。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = load_standalone()

    def test_torch_is_none(self) -> None:
        self.assertIsNone(self.mod.torch)

    def test_torch_not_imported(self) -> None:
        self.assertNotIn("torch", sys.modules)

    def test_caps_defaults_are_neutral_dtype(self) -> None:
        caps = self.mod.Caps(name="corex", device="cuda")
        self.assertIs(caps.act_dtype, F32)
        self.assertIs(caps.weight_dtype, F32)

    def test_make_device_returns_device_ref(self) -> None:
        caps = self.mod.Caps(name="corex", device="cuda", device_index=1)
        dev = self.mod.make_device(caps)
        self.assertIsInstance(dev, self.mod.DeviceRef)
        self.assertEqual((dev.type, dev.index), ("cuda", 1))
        self.assertEqual(str(dev), "cuda:1")

    def test_cpu_drops_index(self) -> None:
        caps = self.mod.Caps(name="reference", device="cpu", device_index=3)
        self.assertEqual(str(self.mod.make_device(caps)), "cpu:0")

    def test_caps_still_frozen_and_slotted(self) -> None:
        caps = self.mod.Caps(name="corex", device="cuda")
        with self.assertRaises(Exception):
            caps.device = "xpu"


if __name__ == "__main__":
    unittest.main()
