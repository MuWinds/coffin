"""``cxrt.attn`` 的单元测试：三个小算子的编组与校验。

Module: ``coffin/tests``。没有卡时能验的是「送进去的东西对不对」：参数个数与顺序、
自动分配的 dtype/形状、行间距的默认与拒绝规则。数值由卡上自检负责（``attn_abi.cu``
与 ``attn_device.inc`` 共用同一份内核）。
"""

from __future__ import annotations

import ctypes
import sys
import unittest

import support  # noqa: F401 - 侧效：把 coffin/ 放进 sys.path

import numpy as np

from cxrt import attn as A
from cxrt import tensors as T
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

    def silu_mul(self, *args):
        self.calls.append(("silu_mul", args))

    def silu_mul_quant(self, *args):
        self.calls.append(("silu_mul_quant", args))

    def rope_apply(self, *args):
        self.calls.append(("rope_apply", args))

    def kv_quant(self, *args):
        self.calls.append(("kv_quant", args))


class SiluMulTest(unittest.TestCase):
    """``out = silu(gate) * up``：两半来自同一个更宽缓冲区，所以间距可不同。"""

    def setUp(self) -> None:
        self.ops = A.AttnOps(StubShim())

    def test_allocates_output_with_operand_shape(self) -> None:
        g = FakeTensor((4, 16))
        u = FakeTensor((4, 16))
        out = self.ops.silu_mul(g, u)
        self.assertEqual(out.shape, (4, 16))
        self.assertEqual(str(out.dtype), "torch.bfloat16")

    def test_args_are_in_abi_order(self) -> None:
        g = FakeTensor((4, 16), base=0x1000)
        u = FakeTensor((4, 16), base=0x2000)
        out = FakeTensor((4, 16), base=0x3000)
        self.ops.silu_mul(g, u, out, g_pitch=32, u_pitch=32, o_pitch=32)
        name, args = self.ops.shim.calls[-1]
        self.assertEqual(name, "silu_mul")
        self.assertEqual(len(args), 10)
        self.assertEqual(args[3:5], (4, 16))               # rows, width
        self.assertEqual(args[5:8], (32, 32, 32))           # 三个行间距
        self.assertEqual(args[8], T.BF16C)                  # dtype 码

    def test_pitch_defaults_to_width(self) -> None:
        self.ops.silu_mul(FakeTensor((3, 8)), FakeTensor((3, 8)))
        _name, args = self.ops.shim.calls[-1]
        self.assertEqual(args[5:8], (8, 8, 8))

    def test_1d_operand_is_one_row(self) -> None:
        self.ops.silu_mul(FakeTensor((32,)), FakeTensor((32,)))
        _name, args = self.ops.shim.calls[-1]
        self.assertEqual(args[3:5], (1, 32))

    def test_pitch_narrower_than_row_raises(self) -> None:
        with self.assertRaises(ShapeError):
            self.ops.silu_mul(FakeTensor((3, 8)), FakeTensor((3, 8)), g_pitch=7)

    def test_shape_mismatch_raises(self) -> None:
        with self.assertRaises(ShapeError):
            self.ops.silu_mul(FakeTensor((3, 8)), FakeTensor((3, 9)))

    def test_dtype_mismatch_raises(self) -> None:
        with self.assertRaises(ShapeError):
            self.ops.silu_mul(FakeTensor((3, 8)), FakeTensor((3, 8), "float32"))

    def test_out_must_match_shape(self) -> None:
        with self.assertRaises(ShapeError):
            self.ops.silu_mul(FakeTensor((3, 8)), FakeTensor((3, 8)), FakeTensor((3, 9)))


class SiluMulQuantTest(unittest.TestCase):
    """融合量化：多出来的两个目的地形状固定为 ``[rows, width]`` int8 与 ``[rows]`` fp32。"""

    def setUp(self) -> None:
        self.ops = A.AttnOps(StubShim())

    def test_allocates_int8_and_scale(self) -> None:
        out, q, s = self.ops.silu_mul_quant(FakeTensor((4, 16)), FakeTensor((4, 16)))
        self.assertEqual((out.shape, q.shape, s.shape), ((4, 16), (4, 16), (4,)))
        self.assertEqual(T.code_of(q), T.I8)
        self.assertEqual(T.code_of(s), T.F32)

    def test_args_are_in_abi_order(self) -> None:
        self.ops.silu_mul_quant(FakeTensor((4, 16)), FakeTensor((4, 16)),
                                FakeTensor((4, 16)), FakeTensor((4, 16), "int8"),
                                FakeTensor((4,), "float32"), g_pitch=32)
        name, args = self.ops.shim.calls[-1]
        self.assertEqual(name, "silu_mul_quant")
        self.assertEqual(len(args), 12)
        self.assertEqual(args[5:7], (4, 16))                # rows, width
        self.assertEqual(args[7], 32)                        # g_pitch

    def test_bad_int8_destination_raises(self) -> None:
        with self.assertRaises(ShapeError):
            self.ops.silu_mul_quant(FakeTensor((4, 16)), FakeTensor((4, 16)),
                                    out_q=FakeTensor((4, 17), "int8"))

    def test_bad_scale_destination_raises(self) -> None:
        with self.assertRaises(ShapeError):
            self.ops.silu_mul_quant(FakeTensor((4, 16)), FakeTensor((4, 16)),
                                    out_s=FakeTensor((3,), "float32"))


class RopeApplyTest(unittest.TestCase):
    """RoPE：三个维度 + 旋转宽度，只吃 bf16。"""

    def setUp(self) -> None:
        self.ops = A.AttnOps(StubShim())

    def test_allocates_both_outputs(self) -> None:
        q = FakeTensor((2, 4, 64))
        k = FakeTensor((2, 2, 64))
        cos = FakeTensor((2, 64), "float32")
        sin = FakeTensor((2, 64), "float32")
        qo, ko = self.ops.rope_apply(q, k, cos, sin)
        self.assertEqual(qo.shape, (2, 4, 64))
        self.assertEqual(ko.shape, (2, 2, 64))
        self.assertEqual(T.code_of(qo), T.BF16C)

    def test_rotary_dim_defaults_to_table_width(self) -> None:
        q = FakeTensor((1, 8, 128))
        k = FakeTensor((1, 2, 128))
        cos = FakeTensor((1, 64), "float32")
        self.ops.rope_apply(q, k, cos, FakeTensor((1, 64), "float32"))
        _name, args = self.ops.shim.calls[-1]
        self.assertEqual(args[6:11], (1, 8, 2, 128, 64))

    def test_rejects_fp32_q(self) -> None:
        with self.assertRaises(ShapeError):
            self.ops.rope_apply(FakeTensor((1, 8, 128), "float32"), FakeTensor((1, 2, 128)),
                                FakeTensor((1, 64), "float32"),
                                FakeTensor((1, 64), "float32"))

    def test_rejects_odd_rotary_dim(self) -> None:
        q = FakeTensor((1, 8, 128))
        k = FakeTensor((1, 2, 128))
        cos = FakeTensor((1, 63), "float32")
        with self.assertRaises(ShapeError):
            self.ops.rope_apply(q, k, cos, FakeTensor((1, 63), "float32"), rotary_dim=63)

    def test_rejects_mismatched_head_width(self) -> None:
        with self.assertRaises(ShapeError):
            self.ops.rope_apply(FakeTensor((1, 8, 128)), FakeTensor((1, 2, 64)),
                                FakeTensor((1, 64), "float32"),
                                FakeTensor((1, 64), "float32"))

    def test_rejects_fp32_table(self) -> None:
        with self.assertRaises(ShapeError):
            self.ops.rope_apply(FakeTensor((1, 8, 128)), FakeTensor((1, 2, 128)),
                                FakeTensor((1, 64)), FakeTensor((1, 64)))


class KvQuantTest(unittest.TestCase):
    """对称 int8 量化：一个带间距的二维板，两个目的地都由内核写满。"""

    def setUp(self) -> None:
        self.ops = A.AttnOps(StubShim())

    def test_allocates_int8_and_scale(self) -> None:
        q, s = self.ops.kv_quant(FakeTensor((5, 16)))
        self.assertEqual((q.shape, s.shape), ((5, 16), (5,)))
        self.assertEqual(T.code_of(q), T.I8)
        self.assertEqual(T.code_of(s), T.F32)

    def test_args_are_in_abi_order(self) -> None:
        self.ops.kv_quant(FakeTensor((5, 16)), FakeTensor((5, 16), "int8"),
                          FakeTensor((5,), "float32"), pitch=32)
        name, args = self.ops.shim.calls[-1]
        self.assertEqual(name, "kv_quant")
        self.assertEqual(len(args), 8)
        self.assertEqual((args[3], args[4], args[5]), (5, 16, 32))   # M, N, pitch
        self.assertEqual(args[6], T.BF16C)

    def test_pitch_defaults_to_width(self) -> None:
        self.ops.kv_quant(FakeTensor((5, 16)))
        _name, args = self.ops.shim.calls[-1]
        self.assertEqual(args[5], 16)

    def test_pitch_narrower_than_row_raises(self) -> None:
        with self.assertRaises(ShapeError):
            self.ops.kv_quant(FakeTensor((5, 16)), pitch=15)

    def test_bad_int8_destination_raises(self) -> None:
        with self.assertRaises(ShapeError):
            self.ops.kv_quant(FakeTensor((5, 16)), out_q=FakeTensor((5, 17), "int8"))

    def test_bad_scale_destination_raises(self) -> None:
        with self.assertRaises(ShapeError):
            self.ops.kv_quant(FakeTensor((5, 16)), out_s=FakeTensor((4,), "float32"))


class ArrayPathTest(unittest.TestCase):
    """同样的调用在 ``Array`` 家族上也要成立。"""

    def setUp(self) -> None:
        self.rt = MockCudart()
        self.alloc = Allocator(self.rt, retain_bytes=1 << 20)

    def test_silu_mul_output_is_array(self) -> None:
        ops = A.AttnOps(StubShim())
        g = Array.from_numpy(self.rt, self.alloc,
                             np.zeros((4, 16), dtype=np.float32), F32)
        u = Array.from_numpy(self.rt, self.alloc,
                             np.zeros((4, 16), dtype=np.float32), F32)
        out = ops.silu_mul(g, u)
        self.assertIsInstance(out, Array)
        self.assertEqual(out.shape, (4, 16))
        self.assertEqual(out.dtype, F32)

    def test_silu_mul_quant_returns_array_triple(self) -> None:
        ops = A.AttnOps(StubShim())
        g = Array.from_numpy(self.rt, self.alloc,
                             np.zeros((2, 8), dtype=np.float32), F32)
        u = Array.from_numpy(self.rt, self.alloc,
                             np.zeros((2, 8), dtype=np.float32), F32)
        out, q, s = ops.silu_mul_quant(g, u)
        self.assertIsInstance(q, Array)
        self.assertEqual(q.dtype, I8)
        self.assertEqual(s.dtype, F32)


class TorchFreedomTest(unittest.TestCase):
    """模块链里不许出现 torch。"""

    def test_torch_absent(self) -> None:
        self.assertNotIn("torch", sys.modules)

    def test_bit_pattern_codes_are_distinct(self) -> None:
        self.assertEqual((T.BF16C, T.F32, T.I8), (1, 0, 2))


if __name__ == "__main__":
    unittest.main()
