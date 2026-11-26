"""``cxrt.w8a8`` 的单元测试：两条量化出口的编组与校验。

Module: ``coffin/tests``。没有卡时能验的是「送进去的东西对不对」：参数个数与顺序、
目的地的形状/dtype、索引表类型与长度。数值由卡上对拍负责（``w8a8_abi.cu`` 与
``w8a8_device.inc`` 共用同一份内核）。
"""

from __future__ import annotations

import sys
import unittest

import support  # noqa: F401 - 侧效：把 coffin/ 放进 sys.path

import numpy as np

from cxrt import tensors as T
from cxrt import w8a8 as W
from cxrt.array import Array
from cxrt.dtypes import F32, I8
from cxrt.errors import ShapeError
from cxrt.memory import Allocator
from cxrt.mock import MockCudart


class FakeTensor:
    """只有实例方法的最小张量，dtype 打印成 ``torch.<name>``。"""

    def __init__(self, shape, dtype_name="bfloat16", base=0x4000) -> None:
        self.shape = tuple(int(s) for s in shape)
        self.dtype = _FakeDType(dtype_name)
        self._base = base

    def data_ptr(self) -> int:
        return self._base

    @property
    def device(self) -> str:
        return "fake:0"

    def numel(self) -> int:
        out = 1
        for s in self.shape:
            out *= s
        return out

    def new_empty(self, shape):
        return FakeTensor(tuple(shape), self.dtype._name, self._base + 0x100)

    def new_zeros(self, shape):
        return FakeTensor(tuple(shape), self.dtype._name, self._base + 0x200)

    def _retype(self, name: str):
        return FakeTensor(self.shape, name, self._base)

    def float(self): return self._retype("float32")
    def bfloat16(self): return self._retype("bfloat16")
    def int(self): return self._retype("int32")
    def char(self): return self._retype("int8")
    def long(self): return self._retype("int64")
    def byte(self): return self._retype("uint8")


class _FakeDType:
    def __init__(self, name: str) -> None:
        self._name = name

    def __str__(self) -> str:
        return f"torch.{self._name}"


class StubShim:
    """记下每次调用参数的桩。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []

    def quant_act_i8(self, *args):
        self.calls.append(("quant_act_i8", args))

    def gather_quant_i8(self, *args):
        self.calls.append(("gather_quant_i8", args))

    def epilogue(self, *args):
        self.calls.append(("epilogue", args))


class QuantActTest(unittest.TestCase):
    """``quant_act_i8``：一行一个 token，两个目的地都可复用。"""

    def setUp(self) -> None:
        self.ops = W.W8A8Ops(StubShim())

    def test_allocates_int8_and_scale(self) -> None:
        q, s = self.ops.quant_act_i8(FakeTensor((4, 32)))
        self.assertEqual((q.shape, s.shape), ((4, 32), (4,)))
        self.assertEqual(T.code_of(q), T.I8)
        self.assertEqual(T.code_of(s), T.F32)

    def test_args_are_in_abi_order(self) -> None:
        self.ops.quant_act_i8(FakeTensor((4, 32)), FakeTensor((4, 32), "int8"),
                              FakeTensor((4,), "float32"))
        name, args = self.ops.shim.calls[-1]
        self.assertEqual(name, "quant_act_i8")
        self.assertEqual(len(args), 7)
        self.assertEqual((args[3], args[4]), (4, 32))       # M, K
        self.assertEqual(args[5], T.BF16C)                  # dtype 码

    def test_rejects_1d_activation(self) -> None:
        with self.assertRaises(ShapeError):
            self.ops.quant_act_i8(FakeTensor((32,)))

    def test_rejects_fp64_activation(self) -> None:
        with self.assertRaises(ShapeError):
            self.ops.quant_act_i8(FakeTensor((4, 32), "float64"))

    def test_bad_int8_destination_raises(self) -> None:
        with self.assertRaises(ShapeError):
            self.ops.quant_act_i8(FakeTensor((4, 32)), out_q=FakeTensor((4, 33), "int8"))

    def test_bad_scale_destination_raises(self) -> None:
        with self.assertRaises(ShapeError):
            self.ops.quant_act_i8(FakeTensor((4, 32)), out_s=FakeTensor((3,), "float32"))


class GatherQuantTest(unittest.TestCase):
    """``gather_quant_i8``：按索引表搬行，行的数量来自索引表而不是源激活。"""

    def setUp(self) -> None:
        self.ops = W.W8A8Ops(StubShim())

    def test_rows_come_from_index(self) -> None:
        q, s = self.ops.gather_quant_i8(FakeTensor((8, 32)), FakeTensor((3,), "int32"))
        self.assertEqual((q.shape, s.shape), ((3, 32), (3,)))

    def test_args_are_in_abi_order(self) -> None:
        self.ops.gather_quant_i8(FakeTensor((8, 32)), FakeTensor((3,), "int32"),
                                 out_q=FakeTensor((3, 32), "int8"),
                                 out_s=FakeTensor((3,), "float32"))
        name, args = self.ops.shim.calls[-1]
        self.assertEqual(name, "gather_quant_i8")
        self.assertEqual(len(args), 9)
        self.assertEqual((args[4], args[5], args[6]), (8, 3, 32))   # 源行数, rows, K

    def test_rejects_non_int32_index(self) -> None:
        with self.assertRaises(ShapeError):
            self.ops.gather_quant_i8(FakeTensor((8, 32)), FakeTensor((3,), "int64"))

    def test_explicit_rows_must_match_index(self) -> None:
        with self.assertRaises(ShapeError):
            self.ops.gather_quant_i8(FakeTensor((8, 32)), FakeTensor((3,), "int32"), rows=4)


class EpilogueTest(unittest.TestCase):
    """``epilogue``：两个 scale 加一次窄化，写入按行距走。"""

    def setUp(self) -> None:
        self.rt = MockCudart()
        self.alloc = Allocator(self.rt, retain_bytes=1 << 20)
        self.shim = StubShim()
        self.ops = W.W8A8Ops(self.shim)

    def _acc(self, rows=5, cols=11):
        return FakeTensor((rows, cols), "int32")

    def _out(self, rows=5, cols=11):
        return Array.from_numpy(self.rt, self.alloc,
                                np.zeros((rows, cols), dtype=np.float32), F32)

    def test_args_are_in_abi_order(self) -> None:
        self.ops.epilogue(self._acc(), FakeTensor((5,), "float32"),
                          FakeTensor((11,), "float32"), self._out())
        name, args = self.shim.calls[-1]
        self.assertEqual(name, "epilogue")
        self.assertEqual(len(args), 9)
        self.assertEqual((args[4], args[5], args[6]), (5, 11, 11))   # rows, cols, pitch
        self.assertEqual(args[7], T.F32)                             # dtype 码

    def test_pitch_can_be_wider_than_the_block(self) -> None:
        self.ops.epilogue(self._acc(), FakeTensor((5,), "float32"),
                          FakeTensor((11,), "float32"), self._out(), stride=16)
        _, args = self.shim.calls[-1]
        self.assertEqual(args[6], 16)

    def test_allocates_out_when_absent(self) -> None:
        out = self.ops.epilogue(self._acc(), FakeTensor((5,), "float32"),
                               FakeTensor((11,), "float32"))
        self.assertEqual(out.shape, (5, 11))
        self.assertEqual(T.code_of(out), T.BF16C)

    def test_rejects_non_int32_accumulator(self) -> None:
        with self.assertRaises(ShapeError):
            self.ops.epilogue(FakeTensor((5, 11), "float32"), FakeTensor((5,), "float32"),
                              FakeTensor((11,), "float32"), self._out())

    def test_rejects_short_weight_scale(self) -> None:
        with self.assertRaises(ShapeError):
            self.ops.epilogue(self._acc(), FakeTensor((5,), "float32"),
                              FakeTensor((10,), "float32"), self._out())

    def test_rejects_pitch_narrower_than_columns(self) -> None:
        with self.assertRaises(ShapeError):
            self.ops.epilogue(self._acc(), FakeTensor((5,), "float32"),
                              FakeTensor((11,), "float32"), self._out(), stride=10)


class ArrayPathTest(unittest.TestCase):
    """同样的调用在 ``Array`` 家族上也要成立。"""

    def setUp(self) -> None:
        self.rt = MockCudart()
        self.alloc = Allocator(self.rt, retain_bytes=1 << 20)

    def test_quant_outputs_are_arrays(self) -> None:
        ops = W.W8A8Ops(StubShim())
        x = Array.from_numpy(self.rt, self.alloc,
                             np.zeros((4, 32), dtype=np.float32), F32)
        q, s = ops.quant_act_i8(x)
        self.assertIsInstance(q, Array)
        self.assertEqual(q.dtype, I8)
        self.assertEqual(s.dtype, F32)
        self.assertEqual((q.shape, s.shape), ((4, 32), (4,)))


class TorchFreedomTest(unittest.TestCase):
    """模块链里不许出现 torch。"""

    def test_torch_absent(self) -> None:
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
