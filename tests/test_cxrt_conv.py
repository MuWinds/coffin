"""``cxrt.conv`` 的单元测试：线性注意力入口卷积的编组、校验与 numpy 复刻。

Module: ``coffin/tests``。没有卡时能验的是「送进去的东西对不对」：三条出口的参数顺序、
激活 dtype、状态长度、行间距；数值由 ``abi/conv_abi.cu`` 与 ``linear_attn.cu`` 共用同一份
``conv_device.inc`` 保证，最终在卡上对拍。
"""

from __future__ import annotations

import unittest

import support  # noqa: F401 - 侧效：把 coffin/ 放进 sys.path

import numpy as np

from cxrt import conv as C
from cxrt import tensors as T
from cxrt.array import Array
from cxrt.dtypes import BF16, F32, bf16_bits, bf16_to_f32
from cxrt.errors import ShapeError
from cxrt.mock import make_conv_ops


class FakeTensor:
    """只有实例属性的最小张量，dtype 打印成 ``torch.<name>``。"""

    def __init__(self, shape, strides=None, dtype_name="bfloat16", base=0x6000) -> None:
        self.shape = tuple(int(s) for s in shape)
        self.strides = tuple(int(s) for s in strides) if strides else _dense(self.shape)
        self.dtype = _FakeDType(dtype_name)
        self._base = base

    def data_ptr(self) -> int:
        return self._base

    def stride(self) -> tuple[int, ...]:
        return self.strides

    @property
    def device(self) -> str:
        return "fake:0"

    def new_empty(self, shape):
        return FakeTensor(tuple(shape), dtype_name="float32", base=self._base + 0x200)

    def float(self):
        """fp32 版本（``tensors.new_empty`` 的 dtype 转换会走这里）。"""
        return FakeTensor(self.shape, dtype_name="float32", base=self._base)

    def bfloat16(self):
        return FakeTensor(self.shape, strides=self.strides, dtype_name="bfloat16",
                           base=self._base)

    def new_zeros(self, shape):
        return self.new_empty(shape)


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
        self.calls: list[tuple[str, tuple]] = []

    def conv1d_silu(self, *args):
        self.calls.append(("conv1d_silu", args))

    def conv1d_silu_seq(self, *args):
        self.calls.append(("conv1d_silu_seq", args))

    def conv1d_silu_batch(self, *args):
        self.calls.append(("conv1d_silu_batch", args))


class ArgumentOrderTest(unittest.TestCase):
    """三条出口的实参必须与 ``argtypes`` 逐字对齐。"""

    def test_single_token(self) -> None:
        """单 token：五个实参里只有 C 是标量。"""
        stub = StubShim()
        ops = C.ConvOps(stub)
        taps = FakeTensor((8, 4), dtype_name="float32", base=0x100)
        x = FakeTensor((8,), base=0x200)
        state = FakeTensor((8, 3), dtype_name="float32", base=0x300)
        out = ops.conv1d_silu(taps, x, state)
        args = stub.calls[0][1]
        self.assertEqual(len(args), 6)
        self.assertEqual(args[:4], (0x100, 0x200, 0x300, T.ptr_of(out)))
        self.assertEqual(args[4:], (8, None))

    def test_seq_uses_activation_pitch(self) -> None:
        """行间距来自激活本身，不是 ``C``。"""
        stub = StubShim()
        ops = C.ConvOps(stub)
        taps = FakeTensor((6, 4), dtype_name="float32")
        x = FakeTensor((5, 6), strides=(12, 1))
        state = FakeTensor((6, 3), dtype_name="float32")
        ops.conv1d_silu_seq(taps, x, state)
        args = stub.calls[0][1]
        self.assertEqual(len(args), 8)
        self.assertEqual(args[4:], (5, 6, 12, None))

    def test_batch_carries_three_pitches(self) -> None:
        """批处理：激活、状态、目的地三个间距各自传。"""
        stub = StubShim()
        ops = C.ConvOps(stub)
        taps = FakeTensor((6, 4), dtype_name="float32")
        x = FakeTensor((4, 6), strides=(12, 1))
        state = FakeTensor((4, 18), dtype_name="float32", base=0x700)
        out = ops.conv1d_silu_batch(taps, x, state)
        args = stub.calls[0][1]
        self.assertEqual(len(args), 10)
        self.assertEqual(args[4:8], (4, 6, 12, 18))
        self.assertEqual(args[8], 6)


class GuardTest(unittest.TestCase):
    """dtype、形状与间距的拒绝点。"""

    def setUp(self) -> None:
        self.ops = C.ConvOps(StubShim())

    def test_fp32_activation_rejected(self) -> None:
        """卷积内核只读 bf16，fp32 是错误而不是可选的另一半。"""
        with self.assertRaises(ShapeError):
            self.ops.conv1d_silu(FakeTensor((8, 4), dtype_name="float32"),
                                 FakeTensor((8,), dtype_name="float32"),
                                 FakeTensor((8, 3), dtype_name="float32"))

    def test_taps_must_be_c_by_4(self) -> None:
        """抽头必须是 ``[C, 4]``。"""
        with self.assertRaises(ShapeError):
            self.ops.conv1d_silu(FakeTensor((8, 3), dtype_name="float32"),
                                 FakeTensor((8,), dtype_name="bfloat16"),
                                 FakeTensor((8, 3), dtype_name="float32"))

    def test_taps_must_be_fp32(self) -> None:
        """抽头是 checkpoint 常量，fp32。"""
        with self.assertRaises(ShapeError):
            self.ops.conv1d_silu(FakeTensor((8, 4), dtype_name="bfloat16"),
                                 FakeTensor((8,), dtype_name="bfloat16"),
                                 FakeTensor((8, 3), dtype_name="float32"))

    def test_state_length(self) -> None:
        """历史必须是 ``C * 3`` 个值。"""
        with self.assertRaises(ShapeError):
            self.ops.conv1d_silu(FakeTensor((8, 4), dtype_name="float32"),
                                 FakeTensor((8,), dtype_name="bfloat16"),
                                 FakeTensor((8, 2), dtype_name="float32"))

    def test_activation_width(self) -> None:
        """激活宽度必须等于通道数。"""
        with self.assertRaises(ShapeError):
            self.ops.conv1d_silu(FakeTensor((8, 4), dtype_name="float32"),
                                 FakeTensor((6,), dtype_name="bfloat16"),
                                 FakeTensor((8, 3), dtype_name="float32"))

    def test_inner_stride(self) -> None:
        """最内层步长必须是 1，否则内核的合并读会跨列。"""
        with self.assertRaises(ShapeError):
            self.ops.conv1d_silu_seq(FakeTensor((6, 4), dtype_name="float32"),
                                     FakeTensor((5, 6), strides=(24, 2)),
                                     FakeTensor((6, 3), dtype_name="float32"))


def _sigmoid(v: np.ndarray) -> np.ndarray:
    """fp32 sigmoid。"""
    return (1.0 / (1.0 + np.exp(-v))).astype(np.float32)


def _reference(taps, xs, hist):
    """逐 token 的参考实现，返回 ``(out, new_hist)``。

    Args:
        taps: ``[C, 4]`` fp32。
        xs: ``[T, C]`` fp32 激活（已从 bf16 还原）。
        hist: ``[C, 3]`` fp32 历史。

    Returns:
        ``([T, C] fp32, [C, 3] fp32)``。
    """
    out = np.empty(xs.shape, dtype=np.float32)
    h = hist.copy()
    for t in range(xs.shape[0]):
        acc = (h[:, 0] * taps[:, 0] + h[:, 1] * taps[:, 1] + h[:, 2] * taps[:, 2]
               + xs[t] * taps[:, 3])
        out[t] = (acc * _sigmoid(acc)).astype(np.float32)
        h = np.stack([h[:, 1], h[:, 2], xs[t]], axis=1)
    return out, h


class NumericTest(unittest.TestCase):
    """假驱动 + numpy 内核：编组确实把正确的地址与间距送到了内核。"""

    def setUp(self) -> None:
        self.rt, self.alloc, self.ops = make_conv_ops("bf16")
        self.rng = np.random.default_rng(3)

    def _taps(self, channels: int) -> np.ndarray:
        """随机抽头。"""
        return self.rng.normal(size=(channels, 4)).astype(np.float32)

    def test_single_token(self) -> None:
        """单 token：输出与窗口都推进一格。"""
        channels = 8
        taps = self._taps(channels)
        xs = bf16_to_f32(bf16_bits(self.rng.normal(size=(1, channels)).astype(np.float32)))
        hist = self.rng.normal(size=(1, channels, 3)).astype(np.float32)
        w = Array.from_numpy(self.rt, self.alloc, taps, F32)
        x = Array.from_numpy(self.rt, self.alloc, bf16_bits(xs[0]), BF16)
        st = Array.from_numpy(self.rt, self.alloc, hist, F32)
        out = self.ops.conv1d_silu(w, x, st)
        want, want_hist = _reference(taps, xs, hist[0])
        np.testing.assert_allclose(out.to_numpy(), want[0], rtol=0, atol=1e-6)
        np.testing.assert_allclose(st.to_numpy().reshape(channels, 3), want_hist, atol=1e-7)

    def test_prefill_sequence(self) -> None:
        """预填充分块：一趟走完与逐 token 同一条递推。"""
        channels, tokens = 6, 9
        taps = self._taps(channels)
        xs = bf16_to_f32(bf16_bits(self.rng.normal(size=(tokens, channels)).astype(np.float32)))
        hist = self.rng.normal(size=(1, channels, 3)).astype(np.float32)
        w = Array.from_numpy(self.rt, self.alloc, taps, F32)
        x = Array.from_numpy(self.rt, self.alloc, bf16_bits(xs), BF16)
        st = Array.from_numpy(self.rt, self.alloc, hist, F32)
        out = self.ops.conv1d_silu_seq(w, x, st)
        want, want_hist = _reference(taps, xs, hist[0])
        np.testing.assert_allclose(out.to_numpy(), want, rtol=0, atol=1e-6)
        np.testing.assert_allclose(st.to_numpy().reshape(channels, 3), want_hist, atol=1e-7)

    def test_batch_with_pitched_rows(self) -> None:
        """带行间距的列切片：行间距是 ``2C`` 时第 1 行不能读到第 0 行的通道块。"""
        channels, rows, wide = 6, 3, 2
        taps = self._taps(channels)
        xs = bf16_to_f32(bf16_bits(
            self.rng.normal(size=(rows, channels)).astype(np.float32)))
        padded = np.zeros((rows, channels * wide), dtype=np.float32)
        padded[:, :channels] = xs
        hist = self.rng.normal(size=(rows, channels, 3)).astype(np.float32)
        w = Array.from_numpy(self.rt, self.alloc, taps, F32)
        xw = Array.from_numpy(self.rt, self.alloc, bf16_bits(padded), BF16)
        x = T.as_strided(xw, (rows, channels), (channels * wide, 1))
        st = Array.from_numpy(self.rt, self.alloc, hist, F32)
        out = self.ops.conv1d_silu_batch(w, x, st)
        want, _ = _reference(taps, xs, hist[0])
        got = out.to_numpy()
        for r in range(rows):
            w_row, _ = _reference(taps, xs[r:r + 1], hist[r])
            np.testing.assert_allclose(got[r], w_row[0], rtol=0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
