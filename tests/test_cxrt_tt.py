"""``cxrt.tt`` / ``cxrt.ttnn`` / ``cxrt.torchshim`` 的用例。

Module: ``coffin/tests``。三条主线：

1. 张量语义要和 torch 对齐到「模型层能直接换 import」的程度（形状、dtype、bf16
   舍入、``to`` 的位置参数、``split``/``cat`` 的返回形状）；
2. ``nn`` 半边要能把扁平点号键的权重灌进去（``state_dict`` / ``load_state_dict``）；
3. facade 装出来的模块名字面必须覆盖模型里真实用到的那些（``torch.nn.Module``、
   ``torch.nn.functional``、``torch.Tensor``），且全程不出现 torch。

运行：``python -m unittest discover -s tests -p "test_cxrt_tt.py"``
"""

from __future__ import annotations

import sys
import unittest

import numpy as np


class TensorTests(unittest.TestCase):
    """张量本身的行为。"""

    def setUp(self) -> None:
        from cxrt import tt

        self.tt = tt

    def test_shape_dtype_device(self) -> None:
        """形状、dtype、设备读得回来。"""
        x = self.tt.zeros(2, 3, dtype=self.tt.bfloat16)
        self.assertEqual(x.shape, (2, 3))
        self.assertEqual(x.dtype, self.tt.bfloat16)
        self.assertEqual(x.numel(), 6)
        self.assertEqual(x.dim(), 2)
        self.assertEqual(x.size(1), 3)

    def test_bf16_rounding_is_not_identity(self) -> None:
        """bf16 必须真的按 8 位尾数舍入，否则 dtype 只是个标签。

        bf16 的尾数是 7 位显式（8 位有效），所以在 1.0 附近的步长是
        ``2^-7 = 0.0078125``：``1.00390625``（``1 + 2^-8``，正好半格）会被舍成 1.0；
        而 ``1.0078125`` 恰好落在格点上，不该被动。
        """
        values = np.array([1.0078125, 0.1, 12345.678], dtype=np.float32)
        x = self.tt.tensor(values, dtype=self.tt.bfloat16)
        dense = self.tt.tensor(values, dtype=self.tt.float32)
        self.assertNotEqual(float(x.data[1]), float(dense.data[1]))
        self.assertEqual(float(x.data[0]), 1.0078125)
        halfway = self.tt.tensor(np.array([1.00390625], dtype=np.float32),
                                dtype=self.tt.bfloat16)
        self.assertEqual(float(halfway.data[0]), 1.0)

    def test_to_accepts_positional_dtype_and_device(self) -> None:
        """``to`` 的位置参数、字符串与关键字三种写法都要认。"""
        x = self.tt.ones(2, 2)
        self.assertEqual(x.to(self.tt.bfloat16).dtype, self.tt.bfloat16)
        self.assertEqual(x.to("bfloat16").dtype, self.tt.bfloat16)
        self.assertEqual(x.to(dtype=self.tt.float16).dtype, self.tt.float16)
        self.assertEqual(x.to("cuda:0").device, self.tt.device("cuda", 0))

    def test_arithmetic_promotes_like_torch(self) -> None:
        """类型推进按 torch 的规则：bf16 与 fp32 混合得 fp32，标量不抬 dtype。"""
        a = self.tt.ones(2, 2, dtype=self.tt.bfloat16)
        b = self.tt.ones(2, 2, dtype=self.tt.float32)
        self.assertEqual((a + b).dtype, self.tt.float32)
        self.assertEqual((a + a).dtype, self.tt.bfloat16)
        self.assertEqual((b + 1.0).dtype, self.tt.float32)

    def test_matmul_matches_numpy(self) -> None:
        """矩阵乘与 numpy 一致，含三维批。"""
        rng = np.random.default_rng(0)
        a = self.tt.from_numpy(rng.standard_normal((2, 3), dtype=np.float32))
        b = self.tt.from_numpy(rng.standard_normal((3, 4), dtype=np.float32))
        self.assertTrue(np.allclose((a @ b).data, a.data @ b.data, atol=1e-6))
        p = self.tt.from_numpy(rng.standard_normal((2, 5, 3), dtype=np.float32))
        q = self.tt.from_numpy(rng.standard_normal((2, 3, 4), dtype=np.float32))
        self.assertTrue(np.allclose(self.tt.bmm(p, q).data, np.matmul(p.data, q.data), atol=1e-5))

    def test_softmax_rows_sum_to_one(self) -> None:
        """softmax 沿最后一维归一。"""
        x = self.tt.from_numpy(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
        out = x.softmax(dim=-1)
        self.assertAlmostEqual(float(out.data.sum()), 1.0, places=6)

    def test_split_and_chunk_carry_remainder(self) -> None:
        """``split`` 按长度切、``chunk`` 均分时余数进最后一块。"""
        x = self.tt.arange(10)
        parts = self.tt.split(x, 3)
        self.assertEqual([p.shape for p in parts], [(3,), (3,), (3,), (1,)])
        self.assertEqual(len(self.tt.chunk(x, 4)), 4)

    def test_cat_and_stack(self) -> None:
        """拼接与堆叠的形状。"""
        a = self.tt.zeros(2, 3)
        b = self.tt.ones(2, 3)
        self.assertEqual(self.tt.cat([a, b], dim=0).shape, (4, 3))
        self.assertEqual(self.tt.cat([a, b], dim=1).shape, (2, 6))
        self.assertEqual(self.tt.stack([a, b], dim=0).shape, (2, 2, 3))

    def test_where_and_nonzero(self) -> None:
        """三参选择与单参 ``nonzero``。"""
        cond = self.tt.from_numpy(np.array([[True, False]], dtype=bool_()))
        out = self.tt.where(cond, self.tt.ones(1, 2), self.tt.zeros(1, 2))
        self.assertEqual(float(out.data[0, 0]), 1.0)
        self.assertEqual(float(out.data[0, 1]), 0.0)
        idx = self.tt.nonzero(self.tt.from_numpy(np.array([0, 1, 0, 2], dtype=np.int8)))
        self.assertEqual(idx.shape, (2, 1))
        self.assertEqual(int(idx.data[0, 0]), 1)

    def test_getitem_and_view_share_semantics(self) -> None:
        """切片、``view``、``reshape`` 的结果形状与内容。"""
        x = self.tt.arange(12).reshape(3, 4)
        self.assertEqual(x[1].shape, (4,))
        self.assertEqual(x[:, 2].shape, (3,))
        self.assertEqual(x.view(2, 6).shape, (2, 6))
        self.assertEqual(float(x[2, 3].item()), 11.0)

    def test_masked_fill_and_index_select(self) -> None:
        """掩码填值与按索引取值。"""
        x = self.tt.ones(2, 2)
        mask = self.tt.from_numpy(np.array([[True, False], [False, True]], dtype=bool_()))
        out = x.masked_fill(mask, 0.0)
        self.assertEqual(out.tolist(), [[0.0, 1.0], [1.0, 0.0]])
        table = self.tt.arange(6).reshape(3, 2)
        picked = table.index_select(0, self.tt.from_numpy(np.array([2, 0], dtype=np.int64)))
        self.assertEqual(picked.tolist(), [[4.0, 5.0], [0.0, 1.0]])


class ModuleTests(unittest.TestCase):
    """``nn`` 半边的行为。"""

    def setUp(self) -> None:
        from cxrt import ttnn

        self.nn = ttnn

    def test_linear_forward_and_shapes(self) -> None:
        """``Linear`` 的权重按 ``[out, in]`` 布局，前向是 ``x @ W^T + b``。"""
        from cxrt import tt

        lin = self.nn.Linear(4, 2, bias=True)
        w = np.arange(8, dtype=np.float32).reshape(2, 4)
        lin.weight.data[...] = w
        lin.bias.data[...] = np.array([1.0, -1.0], dtype=np.float32)
        x = tt.ones(3, 4)
        out = lin(x)
        want = np.ones((3, 4), np.float32) @ w.T + np.array([1.0, -1.0], np.float32)
        self.assertEqual(out.shape, (3, 2))
        self.assertTrue(np.allclose(out.data, want, atol=1e-5))

    def test_embedding_lookup(self) -> None:
        """查表按 id 取行。"""
        from cxrt import tt

        emb = self.nn.Embedding(5, 3)
        emb.weight.data[...] = np.arange(15, dtype=np.float32).reshape(5, 3)
        out = emb(tt.from_numpy(np.array([2, 0], dtype=np.int64)))
        self.assertEqual(out.tolist(), [[6.0, 7.0, 8.0], [0.0, 1.0, 2.0]])

    def test_state_dict_uses_dotted_keys(self) -> None:
        """``state_dict`` 的键是点号扁平的，与 checkpoint 命名一致。"""
        outer = self.nn.Module()
        outer.blk = self.nn.Linear(2, 2)
        keys = sorted(outer.state_dict())
        self.assertEqual(keys, ["blk.bias", "blk.weight"])

    def test_load_state_dict_writes_through(self) -> None:
        """灌权重写进 ``Parameter`` 的底层数组，而不是换对象。"""
        from cxrt import tt

        lin = self.nn.Linear(2, 2)
        target = lin.weight
        payload = {"weight": tt.ones(2, 2), "bias": tt.zeros(2)}
        result = lin.load_state_dict(payload)
        self.assertEqual(result.missing_keys, [])
        self.assertIs(lin.weight, target)
        self.assertEqual(lin.weight.tolist(), [[1.0, 1.0], [1.0, 1.0]])

    def test_module_getattr_reaches_submodule(self) -> None:
        """子模块挂上后按名字取得到。"""
        blk = self.nn.Module()
        blk.attn = self.nn.Linear(2, 2)
        self.assertEqual(blk.attn.weight.shape, (2, 2))

    def test_functional_pad_matches_torch_order(self) -> None:
        """``F.pad`` 的 ``(left, right, ...)`` 顺序按维度由内向外。"""
        from cxrt import tt

        x = tt.ones(2, 3)
        out = self.nn.functional.pad(x, (1, 2))
        self.assertEqual(out.shape, (2, 6))
        out2 = self.nn.functional.pad(x, (0, 0, 1, 1))
        self.assertEqual(out2.shape, (4, 3))


class FacadeTests(unittest.TestCase):
    """装出来的 torch 形状模块。"""

    def test_facade_exposes_nn_and_functional(self) -> None:
        """``torch.nn`` / ``torch.nn.functional`` / ``torch.Tensor`` 都在。"""
        from cxrt import torchshim as torch

        self.assertTrue(callable(torch.zeros))
        self.assertTrue(callable(torch.cat))
        self.assertEqual(torch.nn.Module.__name__, "Module")
        self.assertTrue(callable(torch.nn.functional.pad))
        self.assertTrue(callable(torch.nn.functional.silu))
        self.assertEqual(torch.F.pad, torch.nn.functional.pad)
        self.assertIs(torch.Tensor, __import__("cxrt.tt", fromlist=["Tensor"]).Tensor)

    def test_dtype_names_match_torch_spelling(self) -> None:
        """dtype 名字按 torch 的写法提供，模型里的 ``dtype=`` 常量不用改。"""
        from cxrt import torchshim as torch

        for name in ("float32", "float16", "bfloat16", "int8", "int32", "int64",
                     "bool", "long", "half"):
            self.assertTrue(hasattr(torch, name), name)

    def test_no_torch_leaks_in(self) -> None:
        """整条导入路径上不许把 torch 拉进 ``sys.modules``。"""
        from cxrt import torchshim  # noqa: F401 - 触发导入

        self.assertNotIn("torch", sys.modules)


def bool_() -> np.dtype:
    """numpy 的 bool 类型（避免与 ``cxrt.tt.bool`` 的名字冲突）。"""
    return np.dtype(np.bool_)


if __name__ == "__main__":                                  # pragma: no cover
    unittest.main()
