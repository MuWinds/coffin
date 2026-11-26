"""``cxrt.gemv`` 的单元测试：ABI 参数编组与形状校验（只有 nk 权重）。

Module: ``coffin/tests``。先导入 ``support``，随后只导入 ``cxrt.*``、numpy 与这里
定义的桩。

没有卡的时候唯一能验的是「送进去的东西对不对」：参数个数与顺序、输出按谁的 dtype
分配、校验在哪一步拦下坏形状。数值由卡上自检负责（``abi/gemv_abi.cu`` 与
``gemv_device.inc`` 共用同一份内核，所以两侧的算术不可能各自漂移）。
"""

from __future__ import annotations

import sys
import unittest

import support  # noqa: F401 - 侧效：把 coffin/ 放进 sys.path

import numpy as np

from cxrt import gemv as G
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

    def _retype(self, name: str):
        return FakeTensor(self.shape, name, self._base)

    def float(self): return self._retype("float32")
    def bfloat16(self): return self._retype("bfloat16")
    def int(self): return self._retype("int32")
    def char(self): return self._retype("int8")
    def long(self): return self._retype("int64")
    def byte(self): return self._retype("uint8")

    def new_empty(self, shape):
        return FakeTensor(tuple(shape), self.dtype._name, self._base + 0x100)

    def new_zeros(self, shape):
        return FakeTensor(tuple(shape), self.dtype._name, self._base + 0x200)


class _FakeDType:
    def __init__(self, name: str) -> None:
        self._name = name

    def __str__(self) -> str:
        return f"torch.{self._name}"


class StubShim:
    """记下每次调用参数的桩：只用于校验 ABI 面上的编组。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []

    def nk_i8(self, *args):
        self.calls.append(("nk_i8", args))

    def nk_i8_m(self, *args):
        self.calls.append(("nk_i8_m", args))


class SurfaceTest(unittest.TestCase):
    """ABI 面只接受 nk 权重；kn 那两族不许再长回来。"""

    def test_no_kn_entry_points(self) -> None:
        ops = G.GemvOps(StubShim())
        self.assertFalse(hasattr(ops, "kn"))
        self.assertFalse(hasattr(ops, "kn_m_split"))
        self.assertFalse(hasattr(ops, "plan"))


class NkSingleTest(unittest.TestCase):
    """单行 nk 路径：形状、dtype、参数顺序。"""

    def setUp(self) -> None:
        self.ops = G.GemvOps(StubShim())

    def test_allocates_output_from_activation_dtype(self) -> None:
        out = self.ops.nk_i8(FakeTensor((4096, 2560), "int8"), FakeTensor((2560,)),
                             FakeTensor((4096,), "float32"))
        self.assertEqual(out.shape, (4096,))
        self.assertEqual(str(out.dtype), "torch.bfloat16")

    def test_args_are_in_abi_order(self) -> None:
        w = FakeTensor((4096, 2560), "int8")
        x = FakeTensor((2560,))
        s = FakeTensor((4096,), "float32")
        out = FakeTensor((4096,))
        self.ops.nk_i8(w, x, s, out, threads=256, unroll=8, width=2)
        name, args = self.ops.shim.calls[-1]
        self.assertEqual(name, "nk_i8")
        self.assertEqual(len(args), 11)
        self.assertEqual(args[1], x.data_ptr())           # 激活紧跟在权重后面
        self.assertEqual(args[2], T.BF16C)                # x_code
        self.assertEqual(args[3], s.data_ptr())           # scale
        self.assertEqual((args[4], args[5]), (4096, 2560))  # N, K（nk 是 [N, K]）
        self.assertEqual(args[7], 256)                    # threads
        self.assertEqual(args[8], 8)                      # unroll
        self.assertEqual(args[9], 2)                      # width
        self.assertIsNone(args[10])                       # stream

    def test_k_must_be_multiple_of_four(self) -> None:
        with self.assertRaises(ShapeError):
            self.ops.nk_i8(FakeTensor((4096, 2561), "int8"), FakeTensor((2561,)),
                           FakeTensor((4096,), "float32"))

    def test_scale_length_must_match_n(self) -> None:
        with self.assertRaises(ShapeError):
            self.ops.nk_i8(FakeTensor((4096, 2560), "int8"), FakeTensor((2560,)),
                           FakeTensor((4095,), "float32"))

    def test_weight_must_be_int8(self) -> None:
        with self.assertRaises(ShapeError):
            self.ops.nk_i8(FakeTensor((4096, 2560)), FakeTensor((2560,)),
                           FakeTensor((4096,), "float32"))

    def test_activation_must_match_k(self) -> None:
        with self.assertRaises(ShapeError):
            self.ops.nk_i8(FakeTensor((4096, 2560), "int8"), FakeTensor((1280,)),
                           FakeTensor((4096,), "float32"))

    def test_misaligned_bf16_activation_raises(self) -> None:
        w = FakeTensor((4096, 2560), "int8")
        x = FakeTensor((2560,), base=0x4004)            # 4 字节对齐，不是 8
        with self.assertRaises(ShapeError):
            self.ops.nk_i8(w, x, FakeTensor((4096,), "float32"))

    def test_bad_out_shape_raises(self) -> None:
        with self.assertRaises(ShapeError):
            self.ops.nk_i8(FakeTensor((4096, 2560), "int8"), FakeTensor((2560,)),
                           FakeTensor((4096,), "float32"), FakeTensor((4095,)))


class NkBatchTest(unittest.TestCase):
    """批处理路径：``[M, K]`` 激活、``[M, N]`` 输出、多出来的 M 参数位置。"""

    def setUp(self) -> None:
        self.ops = G.GemvOps(StubShim())

    def test_batch_shape_and_dtype(self) -> None:
        out = self.ops.nk_i8_m(FakeTensor((4096, 2560), "int8"), FakeTensor((3, 2560)),
                               FakeTensor((4096,), "float32"))
        self.assertEqual(out.shape, (3, 4096))
        self.assertEqual(str(out.dtype), "torch.bfloat16")

    def test_m_sits_between_k_and_y(self) -> None:
        out = FakeTensor((3, 4096))
        self.ops.nk_i8_m(FakeTensor((4096, 2560), "int8"), FakeTensor((3, 2560)),
                         FakeTensor((4096,), "float32"), out)
        name, args = self.ops.shim.calls[-1]
        self.assertEqual(name, "nk_i8_m")
        self.assertEqual(len(args), 12)
        self.assertEqual(args[6], 3)                      # M
        self.assertEqual(args[7], out.data_ptr())          # y 在 M 之后

    def test_bad_out_shape_raises(self) -> None:
        with self.assertRaises(ShapeError):
            self.ops.nk_i8_m(FakeTensor((4096, 2560), "int8"), FakeTensor((3, 2560)),
                             FakeTensor((4096,), "float32"), FakeTensor((4, 4096)))


class ArrayPathTest(unittest.TestCase):
    """同样的调用在 ``Array`` 家族上也要成立：锚张量换成 ``Array``。"""

    def setUp(self) -> None:
        self.rt = MockCudart()
        self.alloc = Allocator(self.rt, retain_bytes=1 << 20)

    def test_output_is_array_with_activation_dtype(self) -> None:
        ops = G.GemvOps(StubShim())
        w = Array.from_numpy(self.rt, self.alloc,
                             np.zeros((32, 16), dtype=np.int8), I8)
        x = Array.from_numpy(self.rt, self.alloc,
                             np.zeros((16,), dtype=np.float32), F32)
        s = Array.from_numpy(self.rt, self.alloc,
                             np.zeros((32,), dtype=np.float32), F32)
        out = ops.nk_i8(w, x, s)
        self.assertIsInstance(out, Array)
        self.assertEqual(out.dtype, F32)
        self.assertEqual(out.shape, (32,))


class TorchFreedomTest(unittest.TestCase):
    """模块链里不许出现 torch。"""

    def test_torch_absent(self) -> None:
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
