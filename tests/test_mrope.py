"""mRoPE 与图像嵌入注入：视觉路径里能在 CPU 上校验的那一半。

这里的东西全部不需要 DCU 就能跑。必须守住三条性质：

* **三维布局** —— ``mrope_cos_sin`` 逐元素复现参考实现的
  ``recomposition_frequencies``；在纯文本 prompt 上，它与解码图捕获时所用的
  一维表逐位相同（``rope_delta = 0``，三个轴相等）；
* **位置记账** —— ``mm_input.mrope_positions`` 复现 ``get_rope_index``
  （``.tmp_dsh/vision/qwen35_vision_ref.py`` 里的参考移植版在下面被*内联*
  重写，而不是导入），覆盖 1024 token 的图像、图像位于 prompt 的头/中/尾、
  以及多图的情况，并要求图像之后那段文本保持恒定的 ``rope_delta``；
* **注入** —— ``scatter_image_embeds`` 按绝对 prompt 下标改写预填充分块里
  恰好属于图像的那些行，边界落在图像内部的分块也算在内。

数学上精确的地方用 ``torch.equal`` 断言；浮点求和顺序不同（但数学等价）
无法避免的地方放宽到 1 ULP。

Module: ``coffin/tests``。只在 CPU 上跑，不加载 checkpoint：这条视觉路径
的这一半不用 DCU 也能校验，正是它存在的理由。

Constraints:
    * 参考版 ``get_rope_index`` 内联重写而不是从 ``.tmp_dsh`` 导入，测试
      因此不依赖那棵临时目录树。
    * ``scatter_image_embeds`` 就地改写传入的缓冲区；任何检查非图像行的
      用例都必须自建输入，否则上一个用例的写入会漏进来。
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

#: 故意*不* ``import support``：那个模块导入时执行的 ``skipUnless`` 探针会调
#: ``has_hip()``，后者构建并加载 HIP 扩展、初始化 DCU。这里每个用例都是纯
#: CPU 算术（布局与注入必须能在没有容器时验证，这正是重点），导入 ``support``
#: 会把 30 ms 的测试跑成几分钟的扩展加载，还要和别的进程抢那一张卡。
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")   # 容器里的 torch 依赖它

_ENGINE = Path(__file__).resolve().parent.parent
if str(_ENGINE) not in sys.path:
    sys.path.insert(0, str(_ENGINE))

import torch

import utils.mm_input as mm_input
from ops.mrope import mrope_cos_sin, scatter_image_embeds
from ops.rope import make_inv_freq, make_rope_tables

#: checkpoint 里的旋转位置配置（``text_config.rope_parameters``）。
HEAD_DIM = 256
PARTIAL = 0.25
ROT = int(HEAD_DIM * PARTIAL)          # 64
THETA = 10_000_000.0
MROPE_SECTION = (11, 11, 10)
IMAGE_TOKEN_ID = mm_input.IMAGE_TOKEN_ID


# 参考实现（内联：测试不得导入 .tmp_dsh）

def _ref_recompose(position_ids: torch.Tensor, inv_freq: torch.Tensor,
                   section=(11, 11, 10), dtype=None):
    """按参考规范 §4.2 复述的 ``Qwen3_5TextRotaryEmbedding``。

    ``position_ids`` 是 ``[3, C]``（行依次为 T、H、W）或 ``[3, B, C]``；参考
    实现把它扩成 ``(3, B, S)``，对我们的二维输入即单批次。
    """
    if position_ids.dim() == 2:
        position_ids = position_ids.unsqueeze(1)        # (3, 1, C)
    pos = position_ids.expand(3, -1, -1)                # (3, B, S)
    freqs = pos[..., None].float() * inv_freq           # (3, B, S, half)
    out = freqs[0].clone()
    for dim, offset in ((1, 1), (2, 2)):
        length = section[dim] * 3
        idx = slice(offset, length, 3)
        out[..., idx] = freqs[dim][..., idx]
    cos = torch.cat((out.cos(), out.cos()), dim=-1)
    sin = torch.cat((out.sin(), out.sin()), dim=-1)
    if position_ids.shape[1] == 1:
        cos, sin = cos[0], sin[0]                       # 还原成 (C, rot)
    if dtype is not None:
        cos, sin = cos.to(dtype), sin.to(dtype)
    return cos, sin


def _ref_llm_vision_positions(start: int, grid, spatial_merge: int = 2) -> torch.Tensor:
    """单张静态图像（t == 1）的 ``get_llm_vision_position_ids``。"""
    t, h, w = (int(x) for x in grid)
    llm_t, llm_h, llm_w = t // 1, h // spatial_merge, w // spatial_merge
    pos_t = torch.arange(llm_t, dtype=torch.float32).long()
    pos_h = torch.arange(llm_h) + start
    pos_w = torch.arange(llm_w) + start
    tg, hg, wg = torch.meshgrid(pos_t, pos_h, pos_w, indexing="ij")
    pids = torch.stack([tg, hg, wg], dim=0).reshape(3, -1)
    pids[0] += start
    return pids


def _ref_rope_index(ids: list[int], grids: list[tuple[int, int, int]],
                    spatial_merge: int = 2):
    """单条未补齐序列的 ``Qwen3_5Model.get_rope_index``。

    ``mm_token_type_ids`` 按 processor 的算法从 id 推出
    （``tt[ids == IMAGE_TOKEN_ID] = 1``）；``vision_start/end`` 仍按文本处理。
    """
    tt = [1 if t == IMAGE_TOKEN_ID else 0 for t in ids]
    runs, start = [], 0
    for i in range(1, len(tt) + 1):
        if i == len(tt) or tt[i] != tt[start]:
            runs.append((tt[start], start, i))
            start = i
    current, chunks, gi = 0, [], 0
    for modality, s, e in runs:
        if modality == 0:
            n = e - s
            chunks.append(torch.arange(n).view(1, -1).expand(3, -1) + current)
            current += n
        else:
            grid = grids[gi]
            gi += 1
            chunks.append(_ref_llm_vision_positions(current, grid, spatial_merge))
            current += max(int(grid[1]), int(grid[2])) // spatial_merge
    pos = torch.cat(chunks, dim=1).reshape(3, -1)
    return pos, int(pos.max()) + 1 - len(ids)


def _prompt_with_image(text_head: int, text_tail: int, grid, n_images: int = 1,
                       between: int = 0):
    """构造 ``文本 … <img> … 文本`` 的 token id 与网格，图像尺寸由 ``grid`` 给定。"""
    t, h, w = grid
    n_img = t * (h // 2) * (w // 2)
    ids = [100 + i for i in range(text_head)]
    ids += [mm_input.VISION_START_ID]
    ids += [IMAGE_TOKEN_ID] * n_img
    ids += [mm_input.VISION_END_ID]
    grids = [grid]
    for _ in range(n_images - 1):
        ids += [200 + i for i in range(between)]
        ids += [mm_input.VISION_START_ID] + [IMAGE_TOKEN_ID] * n_img
        ids += [mm_input.VISION_END_ID]
        grids.append(grid)
    ids += [300 + i for i in range(text_tail)]
    return ids, grids


# 1. cos/sin 布局

class MRopeLayoutTest(unittest.TestCase):
    """``mrope_cos_sin`` 必须逐位等于参考实现的组合方式。"""

    def setUp(self):
        self.inv_freq = make_inv_freq(HEAD_DIM, ROT, THETA, "cpu")

    def test_matches_reference_formula(self):
        """输入任意三维位置时，cos/sin 都要逐元素命中参考公式。"""
        torch.manual_seed(0)
        pos = torch.randint(0, 512, (3, 7), dtype=torch.long)
        cos, sin = mrope_cos_sin(pos, self.inv_freq)
        ref_cos, ref_sin = _ref_recompose(pos, self.inv_freq)
        self.assertEqual(tuple(cos.shape), (7, ROT))
        self.assertTrue(torch.equal(cos, ref_cos), "cos differs from the reference")
        self.assertTrue(torch.equal(sin, ref_sin), "sin differs from the reference")

    def test_axis_to_dimension_map(self):
        """第 i 个频率取自第 i % 3 个轴（T、H、W 循环）。"""
        pos = torch.zeros(3, 1, dtype=torch.long)
        pos[0, 0], pos[1, 0], pos[2, 0] = 3, 5, 7      # 三轴刻意取不同值
        half = ROT // 2
        freq = pos.float()[:, 0, None] * self.inv_freq
        cos, _ = mrope_cos_sin(pos, self.inv_freq)
        for i in range(half):
            expect = freq[i % 3, i].cos()
            self.assertAlmostEqual(float(cos[0, i]), float(expect), places=6,
                                   msg=f"frequency {i} should come from axis {i % 3}")
        # H 占下标 1,4,...,31（11 个），W 占 2,5,...,29（10 个），T 占其余
        # 11 个：11+11+10 == 32，与 checkpoint 的 section 配置对得上。
        t_idx = [i for i in range(half) if i % 3 == 0]
        h_idx = [i for i in range(half) if i % 3 == 1]
        w_idx = [i for i in range(half) if i % 3 == 2]
        self.assertEqual((len(t_idx), len(h_idx), len(w_idx)), MROPE_SECTION)
        self.assertEqual((t_idx[-1], h_idx[-1], w_idx[-1]), (30, 31, 29))

    def test_second_half_is_a_copy(self):
        """前半段频率被原样复制到后半段，这是 RoPE 拼接两条 sin/cos 的约定。"""
        pos = torch.arange(36, dtype=torch.long).reshape(3, -1)
        cos, sin = mrope_cos_sin(pos, self.inv_freq)
        half = ROT // 2
        self.assertTrue(torch.equal(cos[:, :half], cos[:, half:]))
        self.assertTrue(torch.equal(sin[:, :half], sin[:, half:]))

    def test_text_only_matches_the_1d_table(self):
        """三轴相等且 delta 为 0 时，必须复现 ``make_rope_tables`` 的一维表。"""
        max_seq = 512
        cos_tbl, sin_tbl = make_rope_tables(HEAD_DIM, ROT, THETA, max_seq, "cpu",
                                            inv_freq=self.inv_freq)
        t = torch.arange(max_seq, dtype=torch.long)
        pos = t.view(1, -1).expand(3, -1)
        cos, sin = mrope_cos_sin(pos, self.inv_freq)
        self.assertTrue(torch.equal(cos, cos_tbl), "text-only cos != 1-D table")
        self.assertTrue(torch.equal(sin, sin_tbl), "text-only sin != 1-D table")

    def test_rope_delta_shifts_every_row(self):
        """图像之后的解码：整行查表，位移是一个常量。"""
        delta = -4
        t = torch.arange(40, 60, dtype=torch.long)
        pos = (t + delta).view(1, -1).expand(3, -1)
        cos, sin = mrope_cos_sin(pos, self.inv_freq)
        cos_tbl, sin_tbl = make_rope_tables(HEAD_DIM, ROT, THETA, 4096, "cpu",
                                            inv_freq=self.inv_freq)
        self.assertTrue(torch.equal(cos, cos_tbl.index_select(0, t + delta)))
        self.assertTrue(torch.equal(sin, sin_tbl.index_select(0, t + delta)))


# 2. 位置与 rope_delta 对 get_rope_index 校验

class RopeIndexTest(unittest.TestCase):
    """``mm_input.mrope_positions`` 必须等于参考实现 ``get_rope_index``。

    已知缺陷：``mm_input.mrope_positions``（v1）拿*token 下标*当三维游标，
    于是图像之后的每个位置都用 token 下标记数，``rope_delta`` 塌缩成 0
    （位移的符号和大小都丢了）。下面的参考实现是规范里的双游标算法，已逐
    元素对照 ``get_rope_index`` 校验过；契约用例拿它做对比，在辅助函数修好
    之前标记 ``expectedFailure``，这样修好当天测试就会转绿，而不是悄悄一直
    红着。
    """

    #: 1024 个 token 的图 = 一张 1024² 的图：网格 ``[1, 64, 64]``。
    GRID_1024 = (1, 64, 64)

    def _check(self, ids, grids, label=""):
        """在 ``mm_input`` 与参考实现之间比对位置矩阵、形状与 rope_delta。"""
        spans = mm_input.find_image_spans(ids)
        self.assertEqual(len(spans), len(grids), f"{label}: span count")
        pos, delta = mm_input.mrope_positions(spans, grids, len(ids), device="cpu")
        ref_pos, ref_delta = _ref_rope_index(ids, grids)
        self.assertEqual(tuple(pos.shape), (3, len(ids)), f"{label}: shape")
        self.assertTrue(torch.equal(pos, ref_pos),
                        f"{label}: positions differ from get_rope_index")
        self.assertEqual(int(delta), int(ref_delta), f"{label}: rope_delta")
        return pos, int(delta)

    def test_image_in_the_middle(self):
        """图像夹在文本中间：两侧文本都要按双游标接上图像的位移。"""
        ids, grids = _prompt_with_image(5, 6, self.GRID_1024)
        pos, delta = self._check(ids, grids, "middle")
        n_img = 1024
        self.assertEqual(sum(e - s for s, e in mm_input.find_image_spans(ids)), n_img)

    def test_image_at_the_head(self):
        """图像在 prompt 开头：没有前置文本可借位，最容易暴露下标偏移。"""
        ids, grids = _prompt_with_image(0, 3, self.GRID_1024)
        self._check(ids, grids, "head")

    def test_image_at_the_tail(self):
        """图像收尾、后面没有文本：位移只体现在总长度上。"""
        ids, grids = _prompt_with_image(4, 0, self.GRID_1024)
        self._check(ids, grids, "tail")

    def test_two_images(self):
        """两张图之间夹一段文本：第二个图像跨度要从已累计的位移接着算。"""
        ids, grids = _prompt_with_image(3, 4, (1, 32, 32), n_images=2, between=3)
        self._check(ids, grids, "two images")

    def test_small_image_and_non_square(self):
        """小图和非方形图：合并后的网格会取整，这类尺寸最容易算错行列数。"""
        for grid in ((1, 16, 16), (1, 16, 32), (1, 32, 48)):
            with self.subTest(grid=grid):
                ids, grids = _prompt_with_image(3, 3, grid)
                self._check(ids, grids, f"grid {grid}")

    def test_text_after_an_image_has_constant_delta(self):
        """图像块在 max(h, w)/merge 处结束：之后的每个 token 都被整体位移。"""
        ids, grids = _prompt_with_image(5, 8, self.GRID_1024)
        pos, delta = self._check(ids, grids, "delta")
        spans = mm_input.find_image_spans(ids)
        img_start, img_end = spans[-1]
        # 5 个文本 token 加 <|vision_start|> 把图像顶到位置 6；1024² 是
        # 64x64 个 patch，合并后 32x32，所以图像最后一个 H/W 位置是
        # 6 + 31 = 37，而这一段吃掉了 1024 个 token —— 这个差值就是 delta。
        self.assertEqual(img_start, 6)
        self.assertEqual(int(pos[:, img_end - 1].max()), 37)
        # 37 是图像的最后位置，尾部接着跑 38..46，于是整个 prompt 结束在 46，
        # 却装着 1039 个 token —— 这个差值同样是 delta。
        self.assertEqual(int(pos.max()), 46)
        tail = torch.arange(img_end, len(ids), dtype=torch.long) + delta
        self.assertTrue(torch.equal(pos[0, img_end:], tail), "T axis after the image")
        self.assertTrue(torch.equal(pos[1, img_end:], tail), "H axis after the image")
        self.assertTrue(torch.equal(pos[2, img_end:], tail), "W axis after the image")
        # 并且 delta 恰好等于最后位置所能推出的那个差值。
        self.assertEqual(delta, int(pos.max()) + 1 - len(ids))

    def test_text_only_prompt_is_identity(self):
        """纯文本 prompt 上位置就是下标本身，delta 必须为 0，三轴完全一致。"""
        ids = [11, 12, 13, 14, 15]
        pos, delta = mm_input.mrope_positions([], [], len(ids), device="cpu")
        self.assertEqual(delta, 0)
        self.assertTrue(torch.equal(pos[0], torch.arange(len(ids))))
        self.assertTrue(torch.equal(pos[0], pos[1]))
        self.assertTrue(torch.equal(pos[0], pos[2]))

    def test_delta_is_negative_for_a_1024_image(self):
        """校验符号：1024 个 token 只占 32 个位置，delta 必须为负。"""
        ids, grids = _prompt_with_image(5, 6, self.GRID_1024)
        _, delta = self._check(ids, grids, "sign")
        self.assertLess(delta, 0)


# 3. 注入

class ScatterInjectTest(unittest.TestCase):
    """``scatter_image_embeds`` 必须恰好落在分块的图像行上。"""

    HIDDEN = 8

    def setUp(self):
        """造一段带 64 个图像 token 的 prompt，并预先生成对应行数的嵌入。"""
        ids, grids_1024 = _prompt_with_image(4, 4, (1, 16, 16))       # 64 个图像 token
        self.ids = ids
        self.spans = mm_input.find_image_spans(ids)
        n = sum(e - s for s, e in self.spans)
        torch.manual_seed(0)
        self.embeds = torch.randn(n, self.HIDDEN)

    def _gather(self, length):
        return torch.zeros(length, self.HIDDEN)

    def test_whole_prompt_in_one_chunk(self):
        """整条 prompt 一块塞进去：图像行按顺序取嵌入，其余行必须仍是零。"""
        x = self._gather(len(self.ids))
        scatter_image_embeds(x, self.embeds, self.spans, 0, len(self.ids))
        mask = torch.zeros(len(self.ids), dtype=torch.bool)
        base = 0
        for s, e in self.spans:
            mask[s:e] = True
            x[s:e] = self.embeds[base:base + (e - s)]          # 期望值就是恒等映射
            base += e - s
        self.assertTrue(torch.equal(x[mask], self.embeds))
        self.assertEqual(int(x[~mask].abs().sum()), 0)

    def test_chunk_inside_an_image(self):
        """起止都落在图像内部分块：只拷贝该分块覆盖的那几行，偏移不能从头算。"""
        s, e = self.spans[0]
        start, end = s + 10, s + 25
        x = self._gather(end - start)
        scatter_image_embeds(x, self.embeds, self.spans, start, end)
        self.assertTrue(torch.equal(x, self.embeds[10:25]))

    def test_chunk_ending_inside_an_image(self):
        """分块在图像中途结束：前面几行是文本（保持零），后面接着图像的头几行。"""
        s, e = self.spans[0]
        start, end = s - 3, s + 5
        x = self._gather(end - start)
        scatter_image_embeds(x, self.embeds, self.spans, start, end)
        self.assertEqual(int(x[:3].abs().sum()), 0)          # 图像之前是文本
        self.assertTrue(torch.equal(x[3:], self.embeds[:5]))

    def test_chunk_starting_inside_an_image(self):
        """分块从图像中途开始：取的是图像嵌入的尾部，块尾的文本行保持零。"""
        s, e = self.spans[0]
        start, end = e - 4, e + 4
        x = self._gather(end - start)
        scatter_image_embeds(x, self.embeds, self.spans, start, end)
        self.assertTrue(torch.equal(x[:4], self.embeds[-4:]))
        self.assertEqual(int(x[4:].abs().sum()), 0)          # 图像之后是文本

    def test_chunk_without_image_is_untouched(self):
        """不含图像的分块必须原封不动，否则说明偏移算漏了边界判断。"""
        s, _e = self.spans[0]
        x = self._gather(s)                     # 只有图像之前那段文本
        scatter_image_embeds(x, self.embeds, self.spans, 0, s)
        self.assertEqual(int(x.abs().sum()), 0)

    def test_multiple_images_offsets(self):
        """分块横跨多张图时，每个跨度都从自己那份嵌入的偏移处取，不能接错。"""
        ids, _ = _prompt_with_image(2, 2, (1, 16, 16), n_images=3, between=1)
        spans = mm_input.find_image_spans(ids)
        n = sum(e - s for s, e in spans)
        embeds = torch.randn(n, self.HIDDEN)
        x = self._gather(len(ids))
        scatter_image_embeds(x, embeds, spans, 0, len(ids))
        base = 0
        for s, e in spans:
            self.assertTrue(torch.equal(x[s:e], embeds[base:base + (e - s)]))
            base += e - s
        self.assertEqual(base, n)

    def test_image_after_a_cached_prefix(self):
        """共享前缀的后半段：注入要从正确的行开始，否则整块嵌入会整体错位。"""
        s, e = self.spans[0]
        start = s - 1                     # 紧挨图像之前的缓存边界
        x = self._gather(len(self.ids) - start)
        scatter_image_embeds(x, self.embeds, self.spans, start, len(self.ids))
        self.assertEqual(int(x[0].abs().sum()), 0)
        self.assertTrue(torch.equal(x[1:1 + (e - s)], self.embeds))


class ChunkedPrefillEquivalenceTest(unittest.TestCase):
    """逐块切片必须等于整条 prompt：分块预填充的正确性全靠这一点。

    ``Qwen35.prefill`` 按 ``chunk`` 大小走 prompt，把请求位置的一个*切片*、
    以及各分块与图像跨度的交集分别交给每一块。任何一处切片错开一个块边界，
    同一条 prompt 在 chunk=64 和 chunk=4096 下嵌入或旋转的结果就会不同——
    所以这个用例直接重放那套算术（不建模型、不用 GPU），要求逐位相等。
    """

    HIDDEN = 8
    CHUNK = 5

    def test_chunked_injection_and_rope_equal_whole_prompt(self):
        """图像跨度故意横跨多个 5 token 分块，逐块拼接仍需与整条 prompt 完全一致。"""
        inv_freq = make_inv_freq(HEAD_DIM, ROT, THETA, "cpu")
        # 构造一条 prompt，让它的图像跨度故意落在几个 5 token 分块的中间。
        ids, grids = _prompt_with_image(4, 4, (1, 16, 16))
        spans = mm_input.find_image_spans(ids)
        pos, delta = mm_input.mrope_positions(spans, grids, len(ids), device="cpu")
        n_img = sum(e - s for s, e in spans)
        embeds = torch.arange(n_img * self.HIDDEN, dtype=torch.float32).reshape(
            n_img, self.HIDDEN)

        full = torch.zeros(len(ids), self.HIDDEN)
        scatter_image_embeds(full, embeds, spans, 0, len(ids))
        full_cos, full_sin = mrope_cos_sin(pos, inv_freq)

        embeds_out, cos_out, sin_out = [], [], []
        for c0 in range(0, len(ids), self.CHUNK):
            c1 = min(c0 + self.CHUNK, len(ids))
            x = torch.zeros(c1 - c0, self.HIDDEN)
            scatter_image_embeds(x, embeds, spans, c0, c1)
            c, s = mrope_cos_sin(pos[:, c0:c1], inv_freq)
            embeds_out.append(x)
            cos_out.append(c)
            sin_out.append(s)

        self.assertTrue(torch.equal(torch.cat(embeds_out), full),
                        "chunked image injection differs from whole-prompt")
        self.assertTrue(torch.equal(torch.cat(cos_out), full_cos),
                        "chunked cos differs from whole-prompt")
        self.assertTrue(torch.equal(torch.cat(sin_out), full_sin),
                        "chunked sin differs from whole-prompt")
        self.assertNotEqual(delta, 0, "this prompt must actually shift the positions")


# 4. 模型侧的接线

class ModelPlumbingTest(unittest.TestCase):
    """``Qwen35``/``MTP`` 必须持有形状正确的 ``rope_delta`` 缓冲区。

    这些是刻意做成源码级断言：``models.qwen35`` 导入 ``safetensors``
    （checkpoint I/O），开发机上没有这个包，在这里真去导入模块会把一个能在
    CPU 上跑的契约用例变成 skip。缓冲区在 ``__init__`` / ``enable_batching``
    里分配，捕获的图会按下标索引它们，所以缓冲区缺失或形状不对就是 DCU 上的
    加载期失败——钉住源码是本地最便宜的守卫。
    """

    @staticmethod
    def _src(name: str) -> str:
        return (Path(__file__).resolve().parent.parent / name).read_text(encoding="utf-8")

    def test_model_has_per_row_rope_delta(self):
        """每行一个 rope_delta；reset() 清零，新序列因此不会继承上一张图的偏移。"""
        src = self._src("models/qwen35/model.py")
        self.assertIn("self.rope_delta = torch.zeros(1", src)
        self.assertIn("self.rope_delta = torch.zeros(capacity", src)
        self.assertIn("self.rope_delta1 = self.rope_delta[", src)
        # reset() 会清掉它，否则新序列会继承上一张图的偏移。
        self.assertIn("self.pos.zero_()\n        self.rope_delta.zero_()", src)

    def test_rope_delta_follows_the_row_state(self):
        """行偏移必须跟着它的循环状态一起搬运，并在行被复用时清零。"""
        src = self._src("models/qwen35/model.py")
        self.assertIn("self.rope_delta[dst_row].copy_(self.rope_delta[src_row])", src)
        self.assertIn("self.rope_delta[row].zero_()", src)

    def test_prefill_signature_keeps_mm_last_and_optional(self):
        """``prefill`` 的 ``mm`` 必须是最后一个参数且可选，批处理接口不额外收它。"""
        src = self._src("models/qwen35/model.py")
        self.assertIn("def prefill(self, tokens: list[int], chunk: int = 2048,\n"
                      "                spec_row: int | None = None,\n"
                      '                mm: "MMInput | None" = None) -> torch.Tensor:', src)
        # prefill_group 不收 mm：它读的是每条序列自带的载荷。
        self.assertIn("def prefill_group(self, seqs: list, snapshot_interval: int | None = None):",
                      src)
        self.assertIn('mm_list = [getattr(s, "mm", None) for s in seqs]', src)

    def test_mtp_has_a_rope_delta(self):
        """草稿模型也要有自己的 rope_delta，否则投机步骤会按错位的行旋转。"""
        src = self._src("models/qwen35/mtp.py")
        self.assertIn("self.rope_delta = torch.zeros(1", src)
        self.assertIn("self.b_rope_delta = torch.zeros(capacity", src)


if __name__ == "__main__":
    unittest.main()
