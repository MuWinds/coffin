"""``cxrt.safetensors`` 的用例：按 safetensors 真实布局造文件再读回来。

Module: ``coffin/tests``。格式极简，所以这里直接按规范手写字节：8 字节小端头长度 +
JSON 头 + 数据区。覆盖三种最容易读错的 dtype：fp32（原样）、bf16（uint16 还原）、
int8（量化权重的主体）。
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


def _write(path: Path, entries: list[tuple[str, str, tuple[int, ...], bytes]]) -> None:
    """按 safetensors 规范写出一个文件。

    Args:
        path: 目标文件。
        entries: ``[(名字, dtype 串, 形状, 字节)]``，按顺序紧排。
    """
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name, dtype_name, shape, blob in entries:
        header[name] = {"dtype": dtype_name, "shape": list(shape),
                        "data_offsets": [offset, offset + len(blob)]}
        offset += len(blob)
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(len(raw).to_bytes(8, "little"))
        handle.write(raw)
        for _, _, _, blob in entries:
            handle.write(blob)


class SafetensorsTests(unittest.TestCase):
    """读回 fp32 / bf16 / int8。"""

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.path = Path(self._tmp.name) / "quant.safetensors"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_fp32_and_int8(self) -> None:
        """fp32 逐位一致，int8 原样（量化权重的主体是 int8）。"""
        from cxrt import safetensors

        w = np.arange(6, dtype=np.float32).reshape(2, 3)
        q = np.array([-127, 0, 127, -1], dtype=np.int8)
        _write(self.path, [("w", "F32", (2, 3), w.tobytes()),
                           ("q", "I8", (4,), q.tobytes())])
        table = safetensors.load(self.path)
        self.assertEqual(table["w"].shape, (2, 3))
        self.assertTrue(np.array_equal(table["w"].data, w))
        self.assertEqual(table["q"].dtype.name, "int8")
        self.assertTrue(np.array_equal(table["q"].data, q))

    def test_bf16_restored_to_fp32(self) -> None:
        """bf16 存的是 uint16，读出来要是 1.0 / 2.0 / -1.0，而不是三个整数。"""
        from cxrt import safetensors

        bits = np.array([0x3F80, 0x4000, 0xBF80], dtype=np.uint16)
        _write(self.path, [("w", "BF16", (3,), bits.tobytes())])
        table = safetensors.load(self.path)
        self.assertEqual(table["w"].dtype.name, "bfloat16")
        self.assertEqual(table["w"].tolist(), [1.0, 2.0, -1.0])

    def test_offsets_are_absolute_from_data_region(self) -> None:
        """同文件多张量时偏移要累加正确，否则第二个张量会读到第一个的尾巴。"""
        from cxrt import safetensors

        a = np.arange(8, dtype=np.float32).reshape(2, 4)
        b = np.array([[1.5, -2.5]], dtype=np.float32)
        _write(self.path, [("a", "F32", (2, 4), a.tobytes()),
                           ("b", "F32", (1, 2), b.tobytes())])
        table = safetensors.load(self.path)
        self.assertTrue(np.array_equal(table["a"].data, a))
        self.assertTrue(np.array_equal(table["b"].data, b))

    def test_store_dispatch_by_extension(self) -> None:
        """``cxrt.store.load`` 见到 ``.safetensors`` 就转给这个模块。"""
        from cxrt import safetensors, store

        _write(self.path, [("w", "F32", (2, 2), np.ones((2, 2), np.float32).tobytes())])
        table = store.load(self.path)
        self.assertEqual(sorted(table), ["w"])
        self.assertTrue(safetensors.is_safetensors(self.path))

    def test_bad_header_rejected(self) -> None:
        """头长度离谱时要报错，而不是把整个文件当地址读崩。"""
        from cxrt import safetensors

        self.path.write_bytes((200 * 2**20).to_bytes(8, "little") + b"{}")
        with self.assertRaises(ValueError):
            safetensors.read_header(self.path)


if __name__ == "__main__":                                  # pragma: no cover
    unittest.main()
