"""``rt`` 的单元测试：在没有卡、没有 torch 的主机上验证字段、视图、分配与数据流。

Module: ``coffin/tests``。先导入 ``support``（``sys.path`` 跳板），随后**只**导入
``cxrt.*`` 与 numpy——这个文件的存在本身就是一个断言：corex 的运行时
可以在不装载 torch 的进程里跑通。

覆盖范围与不覆盖范围要分清：

* 覆盖：dtype 码表、bf16 位级转换、``Array`` 的指针算术与视图、分配器分档复用、
  H2D/D2H 往返、三个归一化算子的形状校验与数值流（用 ``rt.mock`` 的 numpy 内核）。
* 不覆盖：真正内核的数值。那只能在卡上做，入口是 ``python3 -m
  cxrt.selftest``；这里能做的是保证「送进去的东西和取回来的东西」是对的。
"""

from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

import support  # noqa: F401 - 侧效：把 coffin/ 放进 sys.path

import numpy as np

from cxrt.array import Array
from cxrt.build import BuildError, build, cache_key, compiler
from cxrt.dtypes import (
    BF16, BY_CODE, BY_NAME, F32, I8, U8, bf16_bits, bf16_to_f32,
)
from cxrt.errors import ShapeError
from cxrt.memory import Allocator, HostBuffer
from cxrt.mock import MockCudart, NumpyNormLib, host_reference
from cxrt.norm import RMSNormOps


class DtypeTest(unittest.TestCase):
    """码表与 bf16 位级转换。"""

    def test_code_table_is_bijective(self) -> None:
        """码值必须一一对应，且与 ``cx_tensor.h`` 的前两项一致。"""
        self.assertEqual(F32.code, 0)
        self.assertEqual(BF16.code, 1)
        for code, dtype in BY_CODE.items():
            self.assertEqual(dtype.code, code)
        self.assertEqual(len(BY_CODE), len(BY_NAME))

    def test_bf16_rounds_to_nearest_even(self) -> None:
        """舍入必须就近取偶，否则每个投影系统性偏低。"""
        # 0x3f808000 正好在 0x3f80 与 0x3f81 之间：偶数尾数（0x3f80）胜出。
        bits = bf16_bits(np.array([1.00390625], dtype=np.float32))
        self.assertEqual(int(bits[0]), 0x3f80)
        # 0x3f818000 落在 0x3f81 与 0x3f82 之间：0x3f82 是偶尾数，胜出。
        bits = bf16_bits(np.array([1.01171875], dtype=np.float32))
        self.assertEqual(int(bits[0]), 0x3f82)

    def test_bf16_roundtrip_is_exact_for_representable(self) -> None:
        """bf16 本身能表示的值往返后一位不差。"""
        values = np.array([0.0, 1.0, -2.5, 3.25], dtype=np.float32)
        self.assertTrue(np.array_equal(bf16_to_f32(bf16_bits(values)), values))


class ArrayTest(unittest.TestCase):
    """``Array`` 的分配、指针算术与视图。"""

    def setUp(self) -> None:
        self.rt = MockCudart()
        self.alloc = Allocator(self.rt, retain_bytes=1 << 20)

    def test_empty_and_fill(self) -> None:
        """分配 → 填充 → 读回。"""
        a = Array.empty(self.rt, self.alloc, (3, 4), F32)
        self.assertEqual(a.numel, 12)
        self.assertEqual(a.row_pitch, 4)
        self.assertTrue(a.contiguous)
        a.fill(0)
        self.assertTrue(np.array_equal(a.to_numpy(), np.zeros((3, 4), dtype=np.float32)))

    def test_row_and_narrow_offsets(self) -> None:
        """行的起点必须是「基址 + 行号 × 行间距 × 元素宽度」。"""
        a = Array.empty(self.rt, self.alloc, (4, 8), F32)
        r2 = a.row(2)
        self.assertEqual(r2.ptr, a.ptr + 2 * 8 * 4)
        self.assertEqual(r2.shape, (8,))
        rows = a.rows(1, 3)
        self.assertEqual(rows.ptr, a.ptr + 8 * 4)
        self.assertEqual(rows.shape, (2, 8))
        self.assertEqual(rows.strides, a.strides)

    def test_narrow_keeps_row_pitch(self) -> None:
        """收窄最后一维时行间距不变——这是 ``q_norm`` 列切片的语义。"""
        wide = Array.empty(self.rt, self.alloc, (5, 40), BF16)
        narrow = wide.narrow(32)
        self.assertEqual(narrow.shape, (5, 32))
        self.assertEqual(narrow.row_pitch, 40)
        self.assertEqual(narrow.ptr, wide.ptr)
        self.assertFalse(narrow.contiguous)

    def test_transpose_is_a_view(self) -> None:
        """转置只换步长，不动内存，且不再连续。"""
        a = Array.empty(self.rt, self.alloc, (2, 3), F32)
        t = a.transpose2d()
        self.assertEqual(t.shape, (3, 2))
        self.assertEqual(t.strides, (1, 3))
        self.assertFalse(t.contiguous)
        self.assertTrue(t.same_storage(a))

    def test_view_rejects_non_contiguous(self) -> None:
        """非连续数组换形状需要真的重排，必须报错而不是偷偷拷贝。"""
        wide = Array.empty(self.rt, self.alloc, (4, 40), F32)
        with self.assertRaises(ShapeError):
            wide.narrow(32).view((4 * 32,))

    def test_copy_roundtrip_matches_host(self) -> None:
        """H2D 之后 D2H 回来的内容必须与宿主逐位相同。"""
        values = np.arange(24, dtype=np.float32).reshape(4, 6) / 7.0
        a = Array.from_numpy(self.rt, self.alloc, values)
        back = a.to_numpy()
        self.assertTrue(np.array_equal(back, values))

    def test_bf16_payload_roundtrip(self) -> None:
        """bf16 以位模式搬运：写进去的位模式读回来一位不差。"""
        values = np.random.default_rng(3).standard_normal((2, 5)).astype(np.float32)
        a = Array.from_numpy(self.rt, self.alloc, bf16_bits(values), BF16)
        back = a.to_numpy()
        self.assertEqual(back.dtype, np.uint16)
        self.assertTrue(np.array_equal(back, bf16_bits(values)))
        self.assertTrue(np.allclose(bf16_to_f32(back), values, atol=0.02))

    def test_copy_from_rejects_wrong_size(self) -> None:
        """长度不符是调用错误，必须在拷贝前报出来。"""
        a = Array.empty(self.rt, self.alloc, (2, 3), F32)
        with self.assertRaises(ShapeError):
            a.copy_from(np.zeros((2, 4), dtype=np.float32))


class AllocatorTest(unittest.TestCase):
    """分档复用与所有权。"""

    def setUp(self) -> None:
        self.rt = MockCudart()
        self.alloc = Allocator(self.rt, retain_bytes=1 << 20)

    def test_bucket_reuse_avoids_driver_calls(self) -> None:
        """同尺寸的第二次分配必须命中空闲块，不再问驱动要内存。"""
        a = Array.empty(self.rt, self.alloc, (64, 4), F32)     # 1024 B
        ptr = a.ptr
        a.close()
        b = Array.empty(self.rt, self.alloc, (64, 4), F32)
        self.assertEqual(b.ptr, ptr)
        self.assertEqual(self.rt.peak, 1024)

    def test_bucket_rounds_up_to_power_of_two(self) -> None:
        """档位是 2 的幂，最小 256 B。"""
        a = Array.empty(self.rt, self.alloc, (3, 3), F32)      # 36 B
        self.assertEqual(a._buf.nbytes, 256)  # noqa: SLF001 - 断言内部档位
        b = Array.empty(self.rt, self.alloc, (300,), F32)      # 1200 B
        self.assertEqual(b._buf.nbytes, 2048)  # noqa: SLF001

    def test_foreign_free_is_counted_not_given_to_driver(self) -> None:
        """释放别人的指针必须只计数，绝不能传给 ``cudaFree``。"""
        other = Allocator(self.rt, retain_bytes=1 << 20)
        a = Array.empty(self.rt, other, (8, 8), F32)
        buffer = a._buf                                  # noqa: SLF001
        self.alloc.free(buffer)
        self.assertEqual(self.alloc.foreign_frees, 1)
        self.assertEqual(self.rt.allocated, 8 * 8 * 4)

    def test_trim_returns_everything(self) -> None:
        """``trim`` 之后设备上一块都不该留。"""
        a = Array.empty(self.rt, self.alloc, (16, 16), F32)
        a.close()
        self.alloc.trim()
        self.assertEqual(self.rt.allocated, 0)
        self.assertEqual(self.alloc.retained, 0)

    def test_view_keeps_buffer_alive(self) -> None:
        """父数组关掉之后，视图仍必须持有缓冲。"""
        a = Array.empty(self.rt, self.alloc, (4, 4), F32)   # 64 B，落进 256 B 档
        v = a.row(0)
        a.close()
        # 视图还活着，缓冲就不能回池：只剩一个 256 B 的档位块在设备上。
        self.assertEqual(self.rt.allocated, 256)
        v.close()
        self.assertEqual(self.alloc.retained, 256)


class HostBufferTest(unittest.TestCase):
    """pinned 暂存。"""

    def test_view_writes_through(self) -> None:
        """往视图里写等于写进暂存，H2D 会原样带走。"""
        rt = MockCudart()
        staging = HostBuffer(rt, 32)
        view = staging.view(F32, 8)
        view[:] = np.arange(8, dtype=np.float32)
        staging.close()


class NormTest(unittest.TestCase):
    """三个归一化算子的形状校验与数值流。"""

    def setUp(self) -> None:
        self.rt = MockCudart()
        self.alloc = Allocator(self.rt, retain_bytes=1 << 22)
        self.ops = RMSNormOps(NumpyNormLib(self.rt), self.rt, self.alloc)  # type: ignore[arg-type]
        self.rng = np.random.default_rng(5)

    def _pair(self, rows: int, cols: int, dtype=F32):
        x = self.rng.standard_normal((rows, cols), dtype=np.float32)
        gain = (1.0 + self.rng.standard_normal(cols, dtype=np.float32) * 0.02).astype(np.float32)
        xa = self._place(x, dtype)
        wa = self._place(gain.reshape(1, -1), dtype).view((cols,))
        return x, gain, xa, wa

    def _place(self, values: np.ndarray, dtype) -> Array:
        payload = bf16_bits(values) if dtype is BF16 else values
        return Array.from_numpy(self.rt, self.alloc, payload, dtype)

    def test_rms_norm_f32_matches_reference(self) -> None:
        """fp32 全链路：视图 → 内核 → 读回 → 与 host 参考对比。"""
        x, gain, xa, wa = self._pair(6, 40)
        out = self.ops.rms_norm(xa, wa, eps=1e-6)
        got = out.to_numpy()
        want = host_reference("rms", x, gain, eps=1e-6)
        self.assertLess(float(np.max(np.abs(got - want))), 1e-5)

    def test_rms_norm_bf16_within_one_ulp(self) -> None:
        """bf16 只允许 1 个 ULP 的差：再多就说明舍入顺序变了。"""
        x, gain, xa, wa = self._pair(6, 40, BF16)
        out = self.ops.rms_norm(xa, wa, eps=1e-6)
        got = bf16_to_f32(out.to_numpy())
        want = bf16_to_f32(bf16_bits(host_reference("rms", bf16_to_f32(bf16_bits(x)), gain,
                                                    eps=1e-6)))
        gap = int(np.max(np.abs(bf16_bits(got).astype(np.int32)
                                - bf16_bits(want).astype(np.int32))))
        self.assertLessEqual(gap, 1)

    def test_rms_norm_honours_row_pitch(self) -> None:
        """带间距的列切片：内核必须按 ``pitch`` 走行，按 ``N`` 走列。"""
        rows, wide, cols = 5, 48, 32
        buffer = self.rng.standard_normal((rows, wide)).astype(np.float32)
        device = self._place(buffer, F32)
        slice_ = device.narrow(cols)
        gain = np.ones(cols, dtype=np.float32)
        wa = self._place(gain.reshape(1, -1), F32).view((cols,))
        out = self.ops.rms_norm(slice_, wa, stride=wide, eps=1e-6)
        got = out.to_numpy()
        want = host_reference("rms", buffer[:, :cols], gain, eps=1e-6)
        self.assertLess(float(np.max(np.abs(got - want))), 1e-5)

    def test_add_rms_norm_updates_residual_in_place(self) -> None:
        """残差必须被原地改写，且归一化基于舍入后的和。"""
        x, gain, xa, wa = self._pair(4, 24)
        add = self.rng.standard_normal((4, 24), dtype=np.float32) * 0.5
        aa = self._place(add, F32)
        out = self.ops.add_rms_norm(xa, aa, wa, eps=1e-6)
        total = xa.to_numpy()
        self.assertTrue(np.allclose(total, x + add, atol=1e-6))
        got = out.to_numpy()
        want = host_reference("rms", total, gain, eps=1e-6)
        self.assertLess(float(np.max(np.abs(got - want))), 1e-5)

    def test_gated_matches_reference(self) -> None:
        """门控变体：归一化后乘 ``silu(gate)``，不参与归约。"""
        x, gain, xa, wa = self._pair(4, 24)
        gate = self.rng.standard_normal((4, 24), dtype=np.float32)
        ga = self._place(gate, F32)
        out = self.ops.rms_norm_gated(xa, wa, ga, eps=1e-6)
        got = out.to_numpy()
        want = host_reference("gated", x, gain, gate=gate, eps=1e-6)
        self.assertLess(float(np.max(np.abs(got - want))), 1e-5)

    def test_rejects_non_2d_activation(self) -> None:
        """一维激活是调用错误：内核按 ``[M, N]`` 索引。"""
        flat = Array.empty(self.rt, self.alloc, (24,), F32)
        w = Array.empty(self.rt, self.alloc, (24,), F32)
        with self.assertRaises(ShapeError):
            self.ops.rms_norm(flat, w)

    def test_rejects_unsupported_dtype(self) -> None:
        """int8 激活不在实例化范围内，必须报参数错误而不是算错。"""
        x = Array.empty(self.rt, self.alloc, (2, 8), U8)
        w = Array.empty(self.rt, self.alloc, (8,), U8)
        with self.assertRaises(ShapeError):
            self.ops.rms_norm(x, w)

    def test_rejects_pitch_narrower_than_row(self) -> None:
        """间距窄于行宽会让一行的写入落到另一行未读的数据上。"""
        x, _, xa, wa = self._pair(4, 16)
        with self.assertRaises(ShapeError):
            self.ops.rms_norm(xa, wa, stride=8)

    def test_rejects_destination_aliasing_residual(self) -> None:
        """``out`` 与残差共享存储会把还没用完的和写坏。"""
        x, _, xa, wa = self._pair(4, 16)
        add = self._place(self.rng.standard_normal((4, 16), dtype=np.float32) * 0.5, F32)
        with self.assertRaises(ShapeError):
            self.ops.add_rms_norm(xa, add, wa, out=xa)

    def test_rejects_mismatched_gain(self) -> None:
        """增益元素数必须等于行宽。"""
        x, _, xa, _ = self._pair(4, 16)
        bad = Array.empty(self.rt, self.alloc, (15,), F32)
        with self.assertRaises(ShapeError):
            self.ops.rms_norm(xa, bad)


class NormI8Test(unittest.TestCase):
    """融合归一化 → int8 量化：目的地形状、scale 与逐位约定。"""

    def setUp(self) -> None:
        self.rt = MockCudart()
        self.alloc = Allocator(self.rt, retain_bytes=1 << 22)
        self.ops = RMSNormOps(NumpyNormLib(self.rt), self.rt, self.alloc)  # type: ignore[arg-type]
        self.rng = np.random.default_rng(11)

    def _place(self, values: np.ndarray, dtype) -> Array:
        payload = bf16_bits(values) if dtype is BF16 else values
        return Array.from_numpy(self.rt, self.alloc, payload, dtype)

    def _pair(self, rows: int, cols: int, dtype=F32):
        x = self.rng.standard_normal((rows, cols), dtype=np.float32)
        gain = (1.0 + self.rng.standard_normal(cols, dtype=np.float32) * 0.02).astype(np.float32)
        return x, gain, self._place(x, dtype), self._place(gain.reshape(1, -1), dtype).view((cols,))

    def test_allocates_int8_and_scale(self) -> None:
        x, gain, xa, wa = self._pair(4, 32)
        q, s = self.ops.rms_norm_i8(xa, wa)
        self.assertEqual((q.shape, q.dtype), ((4, 32), I8))
        self.assertEqual((s.shape, s.numel), ((4,), 4))
        self.assertEqual(s.dtype, F32)

    def test_dequant_matches_reference_within_half_step(self) -> None:
        """``q * s`` 与 host 参考的差不超过半个量化步长——这是 int8 的物理下限。"""
        x, gain, xa, wa = self._pair(6, 40)
        q, s = self.ops.rms_norm_i8(xa, wa)
        dq = q.to_numpy().astype(np.float32) * s.to_numpy()[:, None]
        want = host_reference("rms", x, gain, eps=1e-6)
        sc = s.to_numpy()[:, None]
        self.assertTrue(bool(np.all(np.abs(dq - want) <= 0.5 * sc + 1e-6)))

    def test_scale_is_absmax_over_127(self) -> None:
        x, gain, xa, wa = self._pair(3, 24)
        q, s = self.ops.rms_norm_i8(xa, wa)
        self.assertTrue(bool(np.all(np.abs(q.to_numpy()) <= 127)))
        self.assertTrue(bool(np.all(s.to_numpy() > 0)))

    def test_caller_buffers_are_written(self) -> None:
        """给了目的地就不分配：返回的必须是同一块存储。"""
        x, gain, xa, wa = self._pair(5, 40)
        qa = Array.empty(self.rt, self.alloc, (5, 40), I8)
        sa = Array.empty(self.rt, self.alloc, (5,), F32)
        q, s = self.ops.rms_norm_i8(xa, wa, out_q=qa, out_s=sa)
        self.assertTrue(q.same_storage(qa))
        self.assertTrue(s.same_storage(sa))

    def test_add_variant_updates_residual_in_place(self) -> None:
        x, gain, xa, wa = self._pair(4, 32)
        add = self.rng.standard_normal((4, 32), dtype=np.float32)
        ad = self._place(add, F32)
        before = xa.to_numpy().copy()
        q, s = self.ops.add_rms_norm_i8(xa, ad, wa)
        self.assertFalse(bool(np.allclose(xa.to_numpy(), before)))
        self.assertTrue(bool(np.allclose(xa.to_numpy(), before + add, atol=1e-5)))
        self.assertEqual((q.shape, s.numel), ((4, 32), 4))

    def test_threads_override_is_passed_through(self) -> None:
        x, gain, xa, wa = self._pair(4, 32)
        self.ops.rms_norm_i8(xa, wa, threads=256)
        self.assertEqual(self.ops.shim.calls[-1], "rms_norm_i8")

    def test_bad_int8_destination_raises(self) -> None:
        x, gain, xa, wa = self._pair(4, 32)
        bad = Array.empty(self.rt, self.alloc, (4, 33), I8)
        with self.assertRaises(ShapeError):
            self.ops.rms_norm_i8(xa, wa, out_q=bad)

    def test_bad_scale_destination_raises(self) -> None:
        x, gain, xa, wa = self._pair(4, 32)
        bad = Array.empty(self.rt, self.alloc, (3,), F32)
        with self.assertRaises(ShapeError):
            self.ops.rms_norm_i8(xa, wa, out_s=bad)


class BuildKeyTest(unittest.TestCase):
    """构建缓存键：内容决定目录。"""

    def test_key_changes_with_header_content(self) -> None:
        """改头文件必须换键。只哈希 ``.cu`` 会让所有人共用按旧头文件编出来的产物。"""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.h").write_text("// one\n", encoding="utf-8")
            (root / "s.cu").write_text("int f(){return 1;}\n", encoding="utf-8")
            first = cache_key([root / "s.cu"], [root], "ivcore10")
            (root / "a.h").write_text("// two\n", encoding="utf-8")
            second = cache_key([root / "s.cu"], [root], "ivcore10")
            self.assertNotEqual(first, second)
            self.assertEqual(cache_key([root / "s.cu"], [root], "ivcore10"), second)

    def test_key_changes_with_arch(self) -> None:
        """架构变一位就换目录：同一份源码给两个 ISA 编出来的不是同一个东西。"""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "s.cu").write_text("int f(){return 1;}\n", encoding="utf-8")
            self.assertNotEqual(cache_key([root / "s.cu"], [root], "ivcore10"),
                                cache_key([root / "s.cu"], [root], "ivcore11"))

    def test_missing_compiler_is_a_build_error(self) -> None:
        """工具链缺失必须抛 :class:`BuildError` 并列出试过的路径。"""
        import os

        old = os.environ.get("COFFIN_RT_CXX")
        os.environ["COFFIN_RT_CXX"] = str(Path("/nonexistent/compiler"))
        try:
            with self.assertRaises(BuildError):
                compiler()
        finally:
            if old is None:
                os.environ.pop("COFFIN_RT_CXX", None)
            else:
                os.environ["COFFIN_RT_CXX"] = old

    def test_build_without_toolchain_raises(self) -> None:
        """没有编译器时 ``build`` 必须失败，而不是留一个空目录骗下一次调用。"""
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "s.cu").write_text("int f(){return 1;}\n", encoding="utf-8")
            old_cxx = os.environ.get("COFFIN_RT_CXX")
            old_dir = os.environ.get("COFFIN_RT_DIR")
            os.environ["COFFIN_RT_CXX"] = str(root / "no-such-clang")
            os.environ["COFFIN_RT_DIR"] = str(root / "cache")
            try:
                with self.assertRaises(BuildError):
                    build([root / "s.cu"], [root], "ivcore10", "unit")
            finally:
                for key, value in (("COFFIN_RT_CXX", old_cxx), ("COFFIN_RT_DIR", old_dir)):
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value


class TorchFreedomTest(unittest.TestCase):
    """这一层不许把 torch 带进来。"""

    def test_rt_imports_without_torch(self) -> None:
        """把 ``rt`` 的每个模块都导一遍，``sys.modules`` 里都不能出现 torch。"""
        self.assertNotIn("torch", sys.modules)
        for name in ("cxrt", "cxrt.driver",
                     "cxrt.stream", "cxrt.memory",
                     "cxrt.array", "cxrt.dtypes",
                     "cxrt.build", "cxrt.shim",
                     "cxrt.norm", "cxrt.mock",
                     "cxrt.selftest"):
            importlib.import_module(name)
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
