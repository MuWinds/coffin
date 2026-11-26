"""``cxrt.store`` 的用例：用**真的 torch 容器格式**验证无 torch 读取。

Module: ``coffin/tests``。主机上没有 torch，所以这里不自欺地「读一个自己写的
npz」，而是按 ``torch.save`` 的 zip 布局（``data.pkl`` + ``data/<key>`` +
``version``）现场造一个归档，pickle 里写的 global 名与 torch 完全一致
（``torch._utils._rebuild_tensor_v2``、``torch.FloatStorage``），再看 loader 能否
把它解回张量。

造归档时借用了 ``sys.modules['torch']`` 两个替身模块——只在造归档的那一瞬间存在，
``tearDown`` 里删掉，且断言跑完之后 ``torch`` 不在 ``sys.modules`` 里。
"""

from __future__ import annotations

import io
import pickle
import sys
import types
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np


class _StorageRef:
    """pickle 里代表一块存储的占位对象。"""

    def __init__(self, key: str, numel: int) -> None:
        self.key = key
        self.numel = numel


class _FakeTensor:
    """按 ``_rebuild_tensor_v2`` 的参数表写成的假张量。"""

    def __init__(self, ref: _StorageRef, offset: int, size: tuple[int, ...],
                 stride: tuple[int, ...], bf16: bool = False) -> None:
        self.ref = ref
        self.offset = offset
        self.size = size
        self.stride = stride
        self.bf16 = bf16

    def __reduce__(self) -> tuple:
        # 从 ``sys.modules`` 拿替身：``__module__`` 已改成 ``torch._utils``，pickle 就会
        # 写出与真 torch 完全同名的 global，这正是要验的那件事。
        rebuild = sys.modules["torch._utils"]._rebuild_tensor_v2
        return (rebuild, (self.ref, self.offset, self.size, self.stride, False, None))


def _torch_stub() -> dict[str, types.ModuleType]:
    """造出 ``torch`` / ``torch._utils`` 的替身模块，只为让 pickle 写出同名 global。

    Returns:
        ``{"torch": mod, "utils": mod}``。
    """
    if "torch" in sys.modules:
        raise RuntimeError("a real torch is installed; this test would be meaningless")

    torch_mod = types.ModuleType("torch")
    utils_mod = types.ModuleType("torch._utils")

    def _rebuild_tensor_v2(*args: object, **kwargs: object) -> None:   # pragma: no cover
        raise AssertionError("stub rebuild must never be executed")

    _rebuild_tensor_v2.__module__ = "torch._utils"
    _rebuild_tensor_v2.__qualname__ = "_rebuild_tensor_v2"
    utils_mod._rebuild_tensor_v2 = _rebuild_tensor_v2

    class FloatStorage:                                     # pragma: no cover - 仅做标记
        pass

    FloatStorage.__module__ = "torch"
    FloatStorage.__qualname__ = "FloatStorage"

    class BFloat16Storage:                                  # pragma: no cover - 仅做标记
        pass

    BFloat16Storage.__module__ = "torch"
    BFloat16Storage.__qualname__ = "BFloat16Storage"

    torch_mod._utils = utils_mod
    torch_mod.FloatStorage = FloatStorage
    torch_mod.BFloat16Storage = BFloat16Storage
    sys.modules["torch"] = torch_mod
    sys.modules["torch._utils"] = utils_mod
    return {"torch": torch_mod, "utils": utils_mod}


class _Pickler(pickle.Pickler):
    """把 :class:`_StorageRef` 写成 ``('storage', type, key, numel)``。"""

    def __init__(self, buf: io.BytesIO, kind: Any, store_mod: Any) -> None:
        super().__init__(buf, protocol=2)
        self.kind = kind
        self.store_mod = store_mod

    def persistent_id(self, obj: object) -> Any:
        if isinstance(obj, _StorageRef):
            return ("storage", self.kind, obj.key, obj.numel)
        return None


def _build_archive(path: Path, entries: dict[str, _FakeTensor],
                   storages: dict[str, bytes], kind: Any, store_mod: Any) -> None:
    """按 torch 的 zip 布局写出归档。

    Args:
        path: 目标文件。
        entries: ``{key: _FakeTensor}``。
        storages: ``{storage_key: 原始字节}``。
        kind: 存储类的替身（``sys.modules["torch"].FloatStorage`` 之类）；pid 里放的
            是类对象本身，pickle 才会写出 ``GLOBAL torch FloatStorage``。
        store_mod: ``cxrt.store``。
    """
    payload = {}
    for key, item in entries.items():
        payload[key] = item
    buf = io.BytesIO()
    _Pickler(buf, kind, store_mod).dump(payload)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", buf.getvalue())
        archive.writestr("archive/version", b"3\n")
        archive.writestr("archive/byteorder", b"little")
        for key, raw in storages.items():
            archive.writestr(f"archive/data/{key}", raw)


class StoreTests(unittest.TestCase):
    """loader 的行为。"""

    def setUp(self) -> None:
        from cxrt import store

        self.store = store
        self._stub = _torch_stub()
        self._tmp = TemporaryDirectory()

    def tearDown(self) -> None:
        self._tmp.cleanup()
        for name in ("torch", "torch._utils"):
            sys.modules.pop(name, None)
        self.assertNotIn("torch", sys.modules)

    def _fp32_checkpoint(self, path: Path) -> None:
        """写一个 fp32 checkpoint：一块 8 元素的存储，切出两个张量（含步长视图）。"""
        rng = np.random.default_rng(7)
        data = rng.standard_normal(8, dtype=np.float32)
        _build_archive(
            path,
            {"w": _FakeTensor(_StorageRef("0", 8), 0, (2, 2), (4, 1)),
             "b": _FakeTensor(_StorageRef("0", 8), 5, (3,), (1,))},
            {"0": data.tobytes()},
            sys.modules["torch"].FloatStorage,
            self.store,
        )
        self._data = data

    def test_reads_fp32_with_offset_and_stride(self) -> None:
        """偏移 + 行步长要按元素解释：``[2,2]`` 视图的行步长是 4，不是 2。"""
        path = Path(self._tmp.name) / "fp32.pt"
        self._fp32_checkpoint(path)
        obj = self.store.load(path)
        w = obj["w"]
        self.assertEqual(w.shape, (2, 2))
        want = np.array([[self._data[0], self._data[1]],
                         [self._data[4], self._data[5]]], dtype=np.float32)
        self.assertTrue(np.array_equal(w.data, want))
        b = obj["b"]
        self.assertTrue(np.array_equal(b.data, self._data[5:8]))

    def test_reads_bf16_storage(self) -> None:
        """bf16 存储是 uint16，要按 7 位尾数还原，而不是当 uint16 直接给。"""
        path = Path(self._tmp.name) / "bf16.pt"
        bits = np.array([0x3F80, 0x4000, 0xBF80], dtype=np.uint16)
        _build_archive(path,
                       {"w": _FakeTensor(_StorageRef("0", 3), 0, (3,), (1,), bf16=True)},
                       {"0": bits.tobytes()}, sys.modules["torch"].BFloat16Storage, self.store)
        got = self.store.load(path)["w"]
        self.assertEqual(got.tolist(), [1.0, 2.0, -1.0])

    def test_state_dict_of_drops_metadata(self) -> None:
        """``state_dict_of`` 只留张量键，元数据键不进权重表。"""
        path = Path(self._tmp.name) / "fp32.pt"
        self._fp32_checkpoint(path)
        table = self.store.state_dict_of(path)
        self.assertEqual(sorted(table), ["b", "w"])

    def test_rejects_non_zip(self) -> None:
        """非 zip 容器要给明确的错误，而不是猜。"""
        path = Path(self._tmp.name) / "legacy.bin"
        path.write_bytes(b"not a zip")
        with self.assertRaises(ValueError):
            self.store.load(path)

    def test_roundtrip_own_format(self) -> None:
        """自有 ``save`` / ``load`` 走 npz，能来回。"""
        from cxrt import tt

        path = Path(self._tmp.name) / "mine.npz"
        self.store.save({"w": tt.ones(2, 2)}, path)
        back = np.load(path)["w"]
        self.assertEqual(back.tolist(), [[1.0, 1.0], [1.0, 1.0]])


if __name__ == "__main__":                                  # pragma: no cover
    unittest.main()
