"""``cxrt.install`` 的用例：别名装进去之后，既有代码的 import 形状必须照旧成立。

Module: ``coffin/tests``。要点三条：

1. ``import torch`` / ``import torch.nn as nn`` / ``import torch.nn.functional as F`` /
   ``import torch.distributed as dist`` 四种写法都要能走通（缺一个就说明 ``sys.modules``
   里少了子模块名，而不是少了属性）；
2. ``sys.modules['torch']`` 必须是壳本身，``verify()`` 才能为真；
3. 真的 torch 在场时要拒绝覆盖——静默替换是最难查的一类错误。
"""

from __future__ import annotations

import sys
import unittest


class InstallTests(unittest.TestCase):
    """别名安装的行为。"""

    def setUp(self) -> None:
        from cxrt import install

        self.install = install

    def tearDown(self) -> None:
        for name in list(sys.modules):
            if name == "torch" or name.startswith("torch."):
                sys.modules.pop(name, None)

    def test_import_forms_work(self) -> None:
        """四种 import 形状全部成立，且都不需要真的 torch。"""
        self.install.install()
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torch.distributed as dist

        self.assertTrue(self.install.verify())
        self.assertIs(sys.modules["torch"], torch)
        self.assertEqual(nn.Module.__name__, "Module")
        self.assertEqual(F.pad(torch.zeros(1, 3), (1, 1)).shape, (1, 5))
        self.assertEqual(dist.get_world_size(), 1)
        self.assertFalse(dist.is_initialized())

    def test_version_and_cpu_namespace(self) -> None:
        """``torch.version.hip`` 与 ``torch.cpu`` 是后端探测会读的两个名字。"""
        self.install.install()
        import torch

        self.assertIsNone(torch.version.cuda)
        self.assertIsNone(torch.version.hip)
        self.assertTrue(torch.cpu.is_available())
        self.assertFalse(torch.cuda.is_available())

    def test_refuses_to_shadow_a_real_torch(self) -> None:
        """``sys.modules`` 里已有别的 ``torch`` 时要拒绝，除非显式 ``force``。"""
        import types

        fake = types.ModuleType("torch")
        fake.__file__ = "/usr/lib/python3/site-packages/torch/__init__.py"
        sys.modules["torch"] = fake
        with self.assertRaises(RuntimeError):
            self.install.install()
        # force 之后必须换成壳
        self.install.install(force=True)
        self.assertTrue(self.install.verify())

    def test_loaded_submodules_are_tracked(self) -> None:
        """装过什么要能查，排查「为什么 import 走了缓存」时靠它。"""
        self.install.install()
        seen = self.install.modules()
        for name in ("torch", "torch.nn", "torch.nn.functional", "torch.distributed"):
            self.assertIn(name, seen)


if __name__ == "__main__":                                  # pragma: no cover
    unittest.main()
