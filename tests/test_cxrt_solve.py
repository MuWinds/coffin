"""``cxrt.solve`` 的单元测试：单出口求解的编组、校验与 numpy 复刻。

Module: ``coffin/tests``。没有卡时能验三件事：参数个数与顺序（错了只会整体错位一格）、
目的地与布局校验、以及「批步长为 0 的唯一块」被原样穿过。数值的最终验收在卡上——
``abi/solve_abi.cu`` 与 ``solve.cu`` 共用同一份 ``solve_kernel``。
"""

from __future__ import annotations

import sys
import unittest

import support  # noqa: F401 - 侧效：把 coffin/ 放进 sys.path

import numpy as np

from cxrt import solve as S
from cxrt import tensors as T
from cxrt.array import Array
from cxrt.dtypes import BF16, F32
from cxrt.errors import ShapeError
from cxrt.mock import make_solve_ops


class FakeTensor3D:
    """带 ``stride()`` 的最小三维张量，dtype 打印成 ``torch.<name>``。"""

    def __init__(self, shape, strides=None, dtype_name="float32", base=0x5000) -> None:
        self.shape = tuple(int(s) for s in shape)
        self._strides = tuple(int(s) for s in strides) if strides else _dense(self.shape)
        self.dtype = _FakeDType(dtype_name)
        self._base = base

    def data_ptr(self) -> int:
        return self._base

    def stride(self) -> tuple[int, ...]:
        return self._strides

    def new_empty(self, shape):
        """分配（假分配）：换一块基址，形状按调用方。"""
        return FakeTensor3D(tuple(shape), dtype_name=self.dtype._name, base=self._base + 0x100)

    def new_zeros(self, shape):
        """同上（``new_empty`` 的零填充版本在这里等价）。"""
        return self.new_empty(shape)

    @property
    def device(self) -> str:
        return "fake:0"


class _FakeDType:
    def __init__(self, name: str) -> None:
        self._name = name

    def __str__(self) -> str:
        return f"torch.{self._name}"


def _dense(shape):
    """稠密步长。"""
    out, acc = [], 1
    for size in reversed(shape):
        out.insert(0, acc)
        acc *= size
    return tuple(out)


class StubShim:
    """记下每次调用参数的桩。"""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def triangular_solve(self, *args):
        self.calls.append(args)


class ArgumentOrderTest(unittest.TestCase):
    """``coffin_triangular_solve`` 的实参必须与 ``argtypes`` 逐字对齐。"""

    def test_order_and_scalars(self) -> None:
        """六个标量按 dtype/bn/L/n/ldb/ldr 的顺序落在中间。"""
        stub = StubShim()
        ops = S.SolveOps(stub)
        lower = FakeTensor3D((2, 3, 3), dtype_name="float32", base=0x1000)
        rhs = FakeTensor3D((2, 3, 4), dtype_name="float32", base=0x2000)
        out = ops.triangular_solve(lower, rhs)
        self.assertEqual(len(stub.calls), 1)
        args = stub.calls[0]
        self.assertEqual(len(args), 10)
        self.assertEqual(args[:3], (0x1000, 0x2000, T.ptr_of(out)))
        # dtype 码 = F32；bn/L/n = 2/3/4；批步长 9/12；流 = None。
        self.assertEqual(args[3:], (0, 2, 3, 4, 9, 12, None))

    def test_batch_stride_zero_passes_through(self) -> None:
        """唯一块的批步长是 0，原样穿过，不实体化。"""
        stub = StubShim()
        ops = S.SolveOps(stub)
        lower = FakeTensor3D((2, 3, 3), strides=(0, 3, 1), dtype_name="float32")
        rhs = FakeTensor3D((2, 3, 3), dtype_name="float32")
        ops.triangular_solve(lower, rhs)
        self.assertEqual(stub.calls[0][7], 0)


class GuardTest(unittest.TestCase):
    """布局与 dtype 的拒绝点。"""

    def setUp(self) -> None:
        self.ops = S.SolveOps(StubShim())

    def test_wrong_dim(self) -> None:
        """二维输入直接拒绝。"""
        with self.assertRaises(ShapeError):
            self.ops.triangular_solve(FakeTensor3D((3, 3)), FakeTensor3D((3, 3)))

    def test_lower_shape_mismatch(self) -> None:
        """``lower`` 必须与右端项的 ``L`` 对齐。"""
        with self.assertRaises(ShapeError):
            self.ops.triangular_solve(FakeTensor3D((2, 4, 4)), FakeTensor3D((2, 3, 2)))

    def test_dtype_mismatch(self) -> None:
        """两个操作数必须同类型。"""
        with self.assertRaises(ShapeError):
            self.ops.triangular_solve(FakeTensor3D((2, 3, 3), dtype_name="bfloat16"),
                                      FakeTensor3D((2, 3, 2), dtype_name="float32"))

    def test_int8_rejected(self) -> None:
        """int8 没有内核实例化。"""
        with self.assertRaises(ShapeError):
            self.ops.triangular_solve(FakeTensor3D((2, 3, 3), dtype_name="int8"),
                                      FakeTensor3D((2, 3, 2), dtype_name="int8"))

    def test_row_stride_must_match_width(self) -> None:
        """批内行间距必须等于行宽：内核按 ``i * n`` 索引。"""
        with self.assertRaises(ShapeError):
            self.ops.triangular_solve(FakeTensor3D((2, 3, 3), strides=(9, 5, 1)),
                                      FakeTensor3D((2, 3, 2)))

    def test_narrow_batch_stride(self) -> None:
        """批步长窄于一整块时读穿分配，拒绝。"""
        with self.assertRaises(ShapeError):
            self.ops.triangular_solve(FakeTensor3D((2, 3, 3), strides=(4, 3, 1)),
                                      FakeTensor3D((2, 3, 2)))

    def test_out_shape(self) -> None:
        """调用方给的目的地形状必须对得上。"""
        with self.assertRaises(ShapeError):
            self.ops.triangular_solve(FakeTensor3D((2, 3, 3)), FakeTensor3D((2, 3, 2)),
                                      out=FakeTensor3D((2, 3, 3)))


def _reference(lower: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """fp64 前向替换，对角线按 1。

    Args:
        lower: ``[bn, L, L]``。
        rhs: ``[bn, L, n]``。

    Returns:
        ``[bn, L, n]`` fp32。
    """
    bn, order, _ = lower.shape
    cols = rhs.shape[2]
    out = np.empty((bn, order, cols), dtype=np.float32)
    for k in range(bn):
        acc = rhs[k].astype(np.float64)
        for i in range(order):
            acc[i] -= lower[k, i, :i].astype(np.float64) @ acc[:i]
        out[k] = acc.astype(np.float32)
    return out


class NumericTest(unittest.TestCase):
    """假驱动 + numpy 内核：编组确实把正确的地址与步长送到了内核。"""

    def _run(self, code_name: str, dtype) -> None:
        """跑一遍并比对。

        Args:
            code_name: ``"f32"``/``"bf16"``。
            dtype: 对应的 ``cxrt.dtypes`` 常量。
        """
        rt, alloc, ops = make_solve_ops(code_name)
        rng = np.random.default_rng(7)
        order, cols = 4, 3
        lower = np.tril(rng.normal(size=(2, order, order)).astype(np.float32), -1)
        rhs = rng.normal(size=(2, order, cols)).astype(np.float32)
        if dtype is BF16:
            lower = _round_bf16(lower)
            rhs = _round_bf16(rhs)
        a = Array.from_numpy(rt, alloc, lower, dtype)
        b = Array.from_numpy(rt, alloc, rhs, dtype)
        x = ops.triangular_solve(a, b)
        got = x.to_numpy()
        if dtype is BF16:
            got = _from_bf16(got)
        want = _reference(lower, rhs)
        tol = 4e-6 if dtype is F32 else 2e-2 * float(np.abs(want).max())
        np.testing.assert_allclose(got, want, rtol=0, atol=tol)
        alloc.trim()

    def test_f32(self) -> None:
        """fp32：解缓冲与写出同精度，误差只来自 fp32 累加。"""
        self._run("f32", F32)

    def test_bf16(self) -> None:
        """bf16：只有写出的那一行被窄化。"""
        self._run("bf16", BF16)

    def test_expanded_unit_block(self) -> None:
        """``eye(L).expand(bn, L, L)``（批步长 0）能直接求解。"""
        rt, alloc, ops = make_solve_ops("f32")
        order = 3
        eye = Array.from_numpy(rt, alloc, np.eye(order, dtype=np.float32), F32)
        lower = T.as_strided(eye, (2, order, order), (0, order, 1))
        rhs = Array.from_numpy(rt, alloc,
                               np.arange(2 * order * 2, dtype=np.float32).reshape(2, order, 2),
                               F32)
        x = ops.triangular_solve(lower, rhs)
        np.testing.assert_allclose(x.to_numpy(), rhs.to_numpy(), atol=1e-6)


def _round_bf16(values: np.ndarray) -> np.ndarray:
    """把 fp32 就近取偶舍入到 bf16 再回 fp32。"""
    from cxrt.dtypes import bf16_bits, bf16_to_f32

    return bf16_to_f32(bf16_bits(values))


def _from_bf16(bits: np.ndarray) -> np.ndarray:
    """把 ``to_numpy`` 给出的 bf16 位模式转回 fp32。"""
    from cxrt.dtypes import bf16_to_f32

    return bf16_to_f32(bits.view(np.uint16))


if __name__ == "__main__":
    unittest.main()
