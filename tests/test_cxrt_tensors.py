"""``cxrt.tensors`` 的单元测试：家族中立协议在没有 torch 的进程里必须成立。

Module: ``coffin/tests``。先导入 ``support``（``sys.path`` 跳板），随后只导入
``cxrt.*``、numpy 与这个文件里定义的假张量。

这里的假张量刻意**不叫** ``torch.Tensor``：它只有 ``data_ptr`` / ``new_empty`` /
``.float()`` 这些实例方法，dtype 打印成 ``"torch.bfloat16"`` 这样的串。corex 后端
将来就是靠这一套东西工作，所以测试也照这一套验：只要协议对，家族是谁不重要。

覆盖范围与不覆盖范围：
* 覆盖：dtype 名字到 ABI 码值、按锚张量分配、家族中立转换、视图、设备身份比较，
  以及「模块里没有 torch」这条断言。
* 不覆盖：真正内核的数值，以及框架张量对本协议的实现细节（那要在装了 torch 的
  机器上验）。
"""

from __future__ import annotations

import sys
import unittest

import support  # noqa: F401 - 侧效：把 coffin/ 放进 sys.path

import numpy as np

from cxrt import tensors as T
from cxrt.array import Array
from cxrt.dtypes import BF16, F32, I8, I32, BY_CODE
from cxrt.errors import ShapeError
from cxrt.memory import Allocator
from cxrt.mock import MockCudart


class FakeDType:
    """打印成 ``torch.<name>`` 的 dtype：模拟框架 dtype，但没有框架代码。"""

    def __init__(self, name: str) -> None:
        self._name = name

    def __str__(self) -> str:
        return f"torch.{self._name}"

    def __repr__(self) -> str:
        return f"FakeDType({self._name})"


class FakeTensor:
    """只有实例方法的最小张量：``data_ptr`` / ``new_empty`` / 语义化转换。"""

    def __init__(self, shape: tuple[int, ...], dtype_name: str = "bfloat16",
                 base: int = 0x1000) -> None:
        self.shape = tuple(int(s) for s in shape)
        self.dtype = FakeDType(dtype_name)
        self._base = base

    # 设备地址
    def data_ptr(self) -> int:
        return self._base

    @property
    def device(self) -> str:
        return "fake:0"

    def numel(self) -> int:
        out = 1
        for size in self.shape:
            out *= size
        return out

    # 分配：家族保持，形状与 dtype 由调用方给
    def _retype(self, name: str) -> "FakeTensor":
        return FakeTensor(self.shape, name, self._base)

    def float(self) -> "FakeTensor":
        return self._retype("float32")

    def bfloat16(self) -> "FakeTensor":
        return self._retype("bfloat16")

    def int(self) -> "FakeTensor":
        return self._retype("int32")

    def char(self) -> "FakeTensor":
        return self._retype("int8")

    def long(self) -> "FakeTensor":
        return self._retype("int64")

    def byte(self) -> "FakeTensor":
        return self._retype("uint8")

    def new_empty(self, shape) -> "FakeTensor":
        return FakeTensor(tuple(shape), self.dtype._name, self._base + 0x100)

    def new_zeros(self, shape) -> "FakeTensor":
        return FakeTensor(tuple(shape), self.dtype._name, self._base + 0x200)

    def as_strided(self, shape, strides) -> "FakeTensor":
        out = FakeTensor(tuple(shape), self.dtype._name, self._base)
        out.strides = tuple(strides)
        return out


class CodeOfTest(unittest.TestCase):
    """dtype 名字到 ABI 码值。"""

    def test_accepts_cxrt_dtype(self) -> None:
        self.assertEqual(T.code_of(F32), F32.code)
        self.assertEqual(T.code_of(BF16), BF16.code)

    def test_accepts_foreign_dtype_name(self) -> None:
        """框架 dtype 只靠打印串认——不需要 import 那个框架。"""
        self.assertEqual(T.code_of(FakeDType("bfloat16")), BF16.code)
        self.assertEqual(T.code_of(FakeDType("float32")), F32.code)
        self.assertEqual(T.code_of(FakeDType("int8")), I8.code)
        self.assertEqual(T.code_of(FakeDType("int32")), I32.code)

    def test_accepts_tensor_and_numpy_dtype(self) -> None:
        self.assertEqual(T.code_of(FakeTensor((2, 3))), BF16.code)
        self.assertEqual(T.code_of(np.dtype(np.float32)), F32.code)

    def test_unknown_dtype_raises(self) -> None:
        """认不出来就报错：静默回退到 fp32 会把类型错误变成数值漂移。"""
        with self.assertRaises(ShapeError):
            T.code_of(FakeDType("float64"))


class PtrTest(unittest.TestCase):
    """设备地址：两种家族的两种形式。"""

    def setUp(self) -> None:
        self.rt = MockCudart()
        self.alloc = Allocator(self.rt, retain_bytes=1 << 20)

    def test_array_uses_ptr_property(self) -> None:
        a = Array.empty(self.rt, self.alloc, (4, 8), F32)
        self.assertEqual(T.ptr_of(a), a.ptr)
        self.assertEqual(T.shape_of(a), (4, 8))
        self.assertEqual(T.numel_of(a), 32)

    def test_foreign_uses_data_ptr(self) -> None:
        t = FakeTensor((3, 5))
        self.assertEqual(T.ptr_of(t), 0x1000)
        self.assertEqual(T.numel_of(t), 15)

    def test_object_without_either_raises(self) -> None:
        with self.assertRaises(ShapeError):
            T.ptr_of(object())


class CastTest(unittest.TestCase):
    """转换：同 dtype 原样返回，不同 dtype 走家族中立方法。"""

    def setUp(self) -> None:
        self.rt = MockCudart()
        self.alloc = Allocator(self.rt, retain_bytes=1 << 20)

    def test_same_dtype_is_identity(self) -> None:
        """调用点大量传进来的本来就是对的，这一档不能分配。"""
        t = FakeTensor((4,))
        self.assertIs(T.cast(t, BF16.code), t)

    def test_foreign_cast_uses_method_name(self) -> None:
        t = FakeTensor((4,))
        out = T.cast(t, F32.code)
        self.assertEqual(str(out.dtype), "torch.float32")
        ints = T.cast(t, I8.code)
        self.assertEqual(str(ints.dtype), "torch.int8")

    def test_unsupported_target_raises(self) -> None:
        with self.assertRaises(ShapeError):
            T.cast(FakeTensor((2,)), BY_CODE[6].code)      # u16

    def test_array_cast_roundtrips_values(self) -> None:
        """``Array`` 的权宜转换必须保住数值（bf16 → fp32 无损）。"""
        values = np.array([[1.0, -2.5], [0.0, 3.25]], dtype=np.float32)
        a = Array.from_numpy(self.rt, self.alloc, values, BF16)
        wide = T.cast(a, F32.code)
        self.assertEqual(wide.dtype, F32)
        self.assertTrue(np.array_equal(wide.to_numpy(), values))


class NewTest(unittest.TestCase):
    """分配：形状与 dtype 由调用方给，家族与设备跟锚张量走。"""

    def setUp(self) -> None:
        self.rt = MockCudart()
        self.alloc = Allocator(self.rt, retain_bytes=1 << 20)

    def test_foreign_family_and_dtype(self) -> None:
        anchor = FakeTensor((1, 8))
        out = T.new_empty(anchor, (4, 8), I8.code)
        self.assertEqual(out.shape, (4, 8))
        self.assertEqual(str(out.dtype), "torch.int8")

    def test_foreign_zeros(self) -> None:
        out = T.new_zeros(FakeTensor((1, 8)), (2, 8), I32.code)
        self.assertEqual(str(out.dtype), "torch.int32")

    def test_array_family(self) -> None:
        anchor = Array.empty(self.rt, self.alloc, (1, 8), F32)
        out = T.new_empty(anchor, (3, 8), BF16.code)
        self.assertIsInstance(out, Array)
        self.assertEqual(out.dtype, BF16)
        self.assertEqual(out.shape, (3, 8))

    def test_empty_like(self) -> None:
        anchor = FakeTensor((2, 5))
        out = T.new_empty_like(anchor)
        self.assertEqual(out.shape, (2, 5))
        self.assertEqual(str(out.dtype), "torch.bfloat16")

    def test_bad_shape_raises(self) -> None:
        with self.assertRaises(ShapeError):
            T.new_empty(FakeTensor((1, 8)), (0, 8), F32.code)


class ViewTest(unittest.TestCase):
    """视图：与源共享内存，不改内容。"""

    def setUp(self) -> None:
        self.rt = MockCudart()
        self.alloc = Allocator(self.rt, retain_bytes=1 << 20)

    def test_foreign_as_strided(self) -> None:
        t = FakeTensor((5, 40))
        narrow = T.as_strided(t, (5, 32), (40, 1))
        self.assertEqual(narrow.shape, (5, 32))
        self.assertEqual(narrow.strides, (40, 1))

    def test_array_as_strided_shares_storage(self) -> None:
        a = Array.empty(self.rt, self.alloc, (5, 40), F32)
        narrow = T.as_strided(a, (5, 32), (40, 1))
        self.assertTrue(narrow.same_storage(a))
        self.assertEqual(narrow.ptr, a.ptr)

    def test_array_as_strided_overrun_raises(self) -> None:
        a = Array.empty(self.rt, self.alloc, (5, 40), F32)
        with self.assertRaises(ShapeError):
            T.as_strided(a, (5, 48), (40, 1))


class TorchFreedomTest(unittest.TestCase):
    """这个模块与它的依赖链里不许出现 torch。"""

    def test_torch_absent_after_import(self) -> None:
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
