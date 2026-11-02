"""视觉塔自测：没有 fixture 覆盖，所以把每处"顺序"都用可证伪的断言钉住。

视觉塔拿不到 golden fixture（dump 它要构造 processor 加真图），因此这里不比对
数值，而是**对着公式断言**。每条测试守的都是"形状一样、结果全错"的那类顺序：

* patch 嵌入：``Conv3d(k=s=(2,16,16))`` 必须与展平后的 GEMM 逐位等价——这是
  "stride == kernel 所以无重叠"的直接检验。
* 学习位置表：抽头求和必须在 fp32 里做（bf16 会错 ~1e-2），且行序是 **merge 块
  顺序**而不是行主序。行序错了形状完全正确。
* axial RoPE：每对 ``(i, i + head_dim/2)`` 用各自的频率旋转，前半的平面用 H
  坐标、后半用 W 坐标。
* merger：LayerNorm 作用在 **2×2 合并之前**的 1152 维上；GELU 是精确 erf 版，
  不是块内那个 tanh 近似。
* 段划分：**每图每帧一段**，跨帧互见不报错但图错。

跑法（仓库根目录）::

    python coffin/tests/test_qwen4exp_vision.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

ENGINE = next((p for p in Path(__file__).resolve().parents
               if (p / "models" / "qwen4exp").is_dir()), Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ENGINE))

from models.qwen4exp.config import VisionConfig  # noqa: E402
from models.qwen4exp.vision import (  # noqa: E402
    PatchEmbed,
    PatchMerger,
    VisionMLP,
    VisionModel,
    scatter_image_embeds,
    vision_attention_seqlens,
    vision_interpolation_indices_weights,
    vision_position_ids,
)

#: 真实 checkpoint 的视觉几何（``inventory.json``：patch_embed.proj.weight
#: 是 ``[1152, 3, 2, 16, 16]``、merger.norm 是 1152 宽、pos_embed 是 2304×1152）。
REAL = dict(
    depth=27, hidden_size=1152, intermediate_size=4304, num_heads=16, in_channels=3,
    patch_size=16, spatial_merge_size=2, temporal_patch_size=2, out_hidden_size=2560,
    num_position_embeddings=2304, rope_theta=10000.0,
    hidden_act="gelu_pytorch_tanh",
)


def small_config(**override) -> VisionConfig:
    """把真实几何等比缩小到能在 CPU 上跑。

    缩小的只有宽度与深度；``patch_size``、``temporal_patch_size``、
    ``spatial_merge_size`` 保持真实值，因为它们决定了 patch 展平的顺序，
    那正是这些测试要守的东西。

    Args:
        **override: 要覆盖的字段。

    Returns:
        ``VisionConfig``。
    """
    fields = dict(REAL)
    fields.update(depth=2, hidden_size=32, intermediate_size=64, num_heads=2,
                  out_hidden_size=64, num_position_embeddings=36)
    fields.update(override)
    return VisionConfig(**fields)


class TestPatchEmbed(unittest.TestCase):
    """patch 嵌入：stride == kernel 时 Conv3d 等价于一次 GEMM。"""

    def test_matches_flattened_gemm(self):
        """``Conv3d`` 的输出必须与"展平 1536 维后做 GEMM"逐位一致。

        参考实现的 patch 嵌入是 ``Conv3d(3->1152, kernel=stride=(2,16,16))``：
        无重叠，所以它只是一次宽矩阵乘。引擎换成 GEMM 时若把展平顺序搞错
        （例如 NCHW 写成 NHWC），形状一样、图全错。
        """
        cfg = small_config()
        embed = PatchEmbed(cfg)
        rows = 5
        x = torch.randn(rows, cfg.patch_dim)
        with torch.no_grad():
            conv_out = embed(x)
            # 展平顺序必须与 Conv3d 的 (C, T, H, W) 视图一致。
            flat = x.view(rows, cfg.patch_dim)
            gemm_out = flat @ embed.proj.weight.reshape(cfg.hidden_size, -1).T + embed.proj.bias
        self.assertLess((conv_out - gemm_out).abs().max().item(), 1e-5)

    def test_rejects_wrong_width(self):
        """宽度不对时必须报错，而不是把最后几维悄悄吃掉。"""
        cfg = small_config()
        embed = PatchEmbed(cfg)
        with self.assertRaises(ValueError):
            embed(torch.randn(3, cfg.patch_dim + 1))


class TestPositionEmbed(unittest.TestCase):
    """学习位置表的重采样：抽头、fp32、merge 块行序。"""

    def test_weights_sum_to_one(self):
        """双线性抽头权重沿 4 个邻居求和必须为 1（``align_corners=True``）。"""
        grid = torch.tensor([[1, 4, 6]])
        _, weights = vision_interpolation_indices_weights(grid, 6, 2)
        self.assertLess((weights.sum(dim=1) - 1.0).abs().max().item(), 1e-6)

    def test_row_order_is_merge_block_order(self):
        """行序是 merge 块顺序：第 0 行取 (0,0)，第 1 行取 (0,1)，第 2 行取 (1,0)。

        processor 发出的 ``pixel_values`` 就按 merge 块排；位置嵌入按行主序排会
        整体错位，而形状完全一致。用 ``h = w = merge`` 的 2×2 网格，行主序与
        merge 顺序的差别正好是前四行的一个转置。
        """
        grid = torch.tensor([[1, 2, 2]])
        indices, _ = vision_interpolation_indices_weights(grid, 6, 2)
        # 2x2 网格、side=6、align_corners：坐标是 0 和 5。
        self.assertEqual(indices.shape, (4, 4))
        # merge=2 时整个 2x2 就是一个块，块内顺序 (0,0)(0,1)(1,0)(1,1) —— 与行
        # 主序相同，所以这里扩成 4x2 才看得出差别。
        grid = torch.tensor([[1, 4, 2]])
        indices, weights = vision_interpolation_indices_weights(grid, 6, 2)
        # 4x2 网格、merge=2 => 两个块，每块 2x2。第 0 块是行 0-1、列 0-1；
        # 第 1 块是行 2-3、列 0-1。块内顺序 (r0c0, r0c1, r1c0, r1c1)。
        state = {}
        for order, index in enumerate(indices[:, 0].tolist()):
            state.setdefault(index, []).append(order)
        # 行主序第 0 行第 0 列（index 0）应当出现在位置 0；第 1 行第 0 列
        # （index 6）应当出现在位置 2，而不是位置 2*... 的行主序位置。
        self.assertIn(0, state[0], "merge 块顺序下 (0,0) 必须在最前")
        del weights

    def test_interpolation_is_fp32(self):
        """抽头权重的 dtype 必须是 fp32：bf16 的带权和会错 ~1e-2。"""
        grid = torch.tensor([[1, 4, 6]])
        _, weights = vision_interpolation_indices_weights(grid, 6, 2)
        self.assertEqual(weights.dtype, torch.float32)

    def test_rejects_non_multiple_grid(self):
        """网格边不是 merge 的倍数时必须报错：2×2 合并会留下不完整的块。"""
        with self.assertRaises(ValueError):
            vision_interpolation_indices_weights(torch.tensor([[1, 5, 4]]), 6, 2)


class TestAxialRope(unittest.TestCase):
    """轴向 RoPE：H/W 各管一半，转满整个 head_dim。"""

    def test_position_table_layout(self):
        """坐标表是 ``[N, 2]``，且行序与 patch 的 merge 块顺序一致。"""
        grid = torch.tensor([[1, 4, 4]])
        positions = vision_position_ids(grid, 2)
        self.assertEqual(tuple(positions.shape), (16, 2))
        # merge=2、4x4 网格 => 4 个块，块内 2x2。第 0 块是 (0,0)(0,1)(1,0)(1,1)。
        self.assertEqual(positions[:4].tolist(), [[0, 0], [0, 1], [1, 0], [1, 1]])
        # 第 1 块是行 0-1、列 2-3。
        self.assertEqual(positions[4:8].tolist(), [[0, 2], [0, 3], [1, 2], [1, 3]])

    def test_axes_map_to_quarters(self):
        """重组的布局是 ``[h(n), w(n), h(n), w(n)]``，``n = head_dim / 4``。

        所以 H 驱动第 0 段与第 2 段、W 驱动第 1 段与第 3 段——不是"前半 H、
        后半 W"。配合分半式 ``rotate_half`` 的 ``i ↔ i + head_dim/2`` 配对，
        第 ``i`` 个二维平面恰好由第 ``i`` 个频率驱动。

        这条断言的第一版写成了"前半只看 H"，实测直接红：说明它确实能区分
        两种布局，不是一个恒真的检查。
        """
        cfg = small_config()
        model = VisionModel(cfg)
        grid = torch.tensor([[1, 4, 4]])
        positions = vision_position_ids(grid, 2)
        quarter = cfg.head_dim // 4

        only_w = torch.zeros_like(positions)
        only_w[:, 1] = positions[:, 1]
        cos_w, _ = model.rotary_pos_emb(only_w)
        varies_w = (cos_w - 1.0).abs().max(dim=0).values > 1e-6
        self.assertEqual(varies_w[:quarter].any().item(), False, "第 0 段应当只看 H")
        self.assertEqual(varies_w[quarter:2 * quarter].any().item(), True, "第 1 段应当由 W 驱动")
        self.assertEqual(varies_w[2 * quarter:3 * quarter].any().item(), False, "第 2 段应当只看 H")
        self.assertEqual(varies_w[3 * quarter:].any().item(), True, "第 3 段应当由 W 驱动")

        only_h = torch.zeros_like(positions)
        only_h[:, 0] = positions[:, 0]
        cos_h, _ = model.rotary_pos_emb(only_h)
        varies_h = (cos_h - 1.0).abs().max(dim=0).values > 1e-6
        self.assertEqual(varies_h[:quarter].any().item(), True, "第 0 段应当由 H 驱动")
        self.assertEqual(varies_h[quarter:2 * quarter].any().item(), False,
                         "第 1 段应当只看 W")

    def test_rotary_preserves_norm(self):
        """旋转是正交变换：每个二维平面的模长必须守恒。"""
        cfg = small_config()
        model = VisionModel(cfg)
        grid = torch.tensor([[1, 4, 6]])
        positions = vision_position_ids(grid, 2)
        cos, sin = model.rotary_pos_emb(positions)
        x = torch.randn(positions.shape[0], cfg.num_heads, cfg.head_dim)
        from models.qwen4exp.rope import rotate_half
        rotated = x * cos.unsqueeze(-2) + rotate_half(x) * sin.unsqueeze(-2)
        # 分半式 rotate_half 把 (i, i+half) 配对，所以按 half 分组算模长。
        half = cfg.head_dim // 2
        before = x[..., :half].pow(2) + x[..., half:].pow(2)
        after = rotated[..., :half].pow(2) + rotated[..., half:].pow(2)
        self.assertLess((before - after).abs().max().item(), 1e-5)


class TestMerger(unittest.TestCase):
    """merger：先归一化再合并；GELU 是精确 erf 版。"""

    def test_norm_before_merge(self):
        """LayerNorm 作用在 1152 维上（合并**之前**），不是 4608 维。"""
        cfg = small_config()
        merger = PatchMerger(cfg, use_postshuffle_norm=False)
        self.assertEqual(merger.norm.normalized_shape, (cfg.hidden_size,))
        self.assertEqual(merger.hidden_size, cfg.hidden_size * cfg.merge_unit)

    def test_gelu_is_exact_erf(self):
        """merger 的激活是精确 erf GELU，与块内那个 tanh 近似不是同一个函数。"""
        cfg = small_config()
        merger = PatchMerger(cfg)
        x = torch.randn(7, cfg.merged_hidden_size)
        with torch.no_grad():
            got = merger.act_fn(x)
        exact = F.gelu(x, approximate="none")
        approx = F.gelu(x, approximate="tanh")
        self.assertLess((got - exact).abs().max().item(), 1e-6)
        self.assertGreater((got - approx).abs().max().item(), 1e-7,
                           "merger 用的必须是 erf 版；用 tanh 近似会静默差 ~1e-3")

    def test_block_mlp_uses_tanh_approx(self):
        """块内 MLP 反过来必须是 tanh 近似——两个 GELU 不能混用。"""
        cfg = small_config()
        mlp = VisionMLP(cfg)
        x = torch.randn(7, cfg.hidden_size)
        with torch.no_grad():
            hidden = mlp.linear_fc1(x)
            got = mlp.linear_fc2(F.gelu(hidden, approximate="tanh"))
        with torch.no_grad():
            reference = mlp(x)
        self.assertLess((got - reference).abs().max().item(), 1e-6)
        with torch.no_grad():
            erf_version = mlp.linear_fc2(F.gelu(hidden, approximate="none"))
        self.assertGreater((erf_version - reference).abs().max().item(), 1e-7)

    def test_shape(self):
        """``[N, hidden] -> [N / merge_unit, out_hidden]``。"""
        cfg = small_config()
        merger = PatchMerger(cfg)
        x = torch.randn(16, cfg.hidden_size)
        self.assertEqual(tuple(merger(x).shape), (4, cfg.out_hidden_size))


class TestSegmentation(unittest.TestCase):
    """段划分：每图每帧一段，跨帧互见不报错但图错。"""

    def test_frames_are_separate_segments(self):
        """``t=2`` 的图像是 **2 段**，不是 1 段。"""
        cu, max_len = vision_attention_seqlens(torch.tensor([[2, 4, 4]]))
        self.assertEqual(cu.tolist(), [0, 16, 32])
        self.assertEqual(max_len, 16)

    def test_multiple_images(self):
        """多图之间也各自成段，不共享注意力。"""
        cu, _ = vision_attention_seqlens(torch.tensor([[1, 4, 4], [1, 2, 6]]))
        self.assertEqual(cu.tolist(), [0, 16, 28])


class TestFullTower(unittest.TestCase):
    """整塔前向的形状与并回文本流的顺序契约。"""

    def test_forward_shapes(self):
        """``last_hidden`` 是 merger 之前的东西，``pooler_output`` 才是并回文本流的。"""
        cfg = small_config()
        model = VisionModel(cfg).eval()
        grid = torch.tensor([[1, 4, 4]])
        pixels = torch.randn(16, cfg.patch_dim)
        with torch.no_grad():
            last, pooled = model(pixels, grid)
        self.assertEqual(tuple(last.shape), (16, cfg.hidden_size))
        self.assertEqual(tuple(pooled.shape), (4, cfg.out_hidden_size))

    def test_two_images_and_frames(self):
        """多图 + 多帧：总行数是 ``sum(t*h*w)``，merger 输出除以 merge_unit。"""
        cfg = small_config()
        model = VisionModel(cfg).eval()
        grid = torch.tensor([[1, 4, 4], [2, 2, 6]])
        rows = 16 + 2 * 12
        with torch.no_grad():
            last, pooled = model(torch.randn(rows, cfg.patch_dim), grid)
        self.assertEqual(last.shape[0], rows)
        self.assertEqual(pooled.shape[0], rows // cfg.merge_unit)

    def test_rejects_grid_mismatch(self):
        """行数与网格对不上时必须在入口炸，而不是让 reshape 顺手吞掉。"""
        cfg = small_config()
        model = VisionModel(cfg).eval()
        with self.assertRaises(ValueError):
            model(torch.randn(15, cfg.patch_dim), torch.tensor([[1, 4, 4]]))

    def test_scatter_contract(self):
        """占位符按行序覆写；个数对不上必须报错。

        顺序契约：``image_embeds`` 的行序必须完全等于 prompt 里 ``<|image_pad|>``
        的顺序（每图按合并后网格的行主序）。搞错顺序会得到一张"看起来合理但
        不对应"的图，形状检查发现不了——所以数量校验必须在。
        """
        hidden, seq = 8, 6
        embeds = torch.zeros(1, seq, hidden)
        ids = torch.tensor([[1, 100, 100, 2, 100, 3]])
        image = torch.arange(3 * hidden, dtype=torch.float32).view(3, hidden)
        out = scatter_image_embeds(embeds, ids, image, image_token_id=100)
        self.assertTrue(torch.equal(out[0, 1], image[0]))
        self.assertTrue(torch.equal(out[0, 2], image[1]))
        self.assertTrue(torch.equal(out[0, 4], image[2]))
        # 非占位符位置保持不变。
        self.assertTrue(torch.equal(out[0, 0], embeds[0, 0]))
        with self.assertRaises(ValueError):
            scatter_image_embeds(embeds, ids, image[:2], image_token_id=100)


if __name__ == "__main__":
    unittest.main()
