"""视觉塔：引擎的 ``VisionTower`` 对照内联的 HF 参考实现。

分两层，数值在任何机器上都能校验：

* **CPU（总会执行）** —— 约 150 行的内联 HF 4.57.6 视觉路径复刻
  （``fast_pos_embed_interpolate`` / ``rot_pos_emb`` / 按图像打包的注意力 /
  merger）。写在这里而不是导入，是为了对着公式比，而不是对着 ``.tmp_dsh``
  的另一份拷贝比。随机权重、fp32、微小与真实几何、多张图像。
* **DCU + checkpoint（否则跳过）** —— 真实的 ``model.visual.*`` 权重对照
  ``transformers`` 4.57.6 的 ``Qwen3VLVisionModel``（M0 对照实现），以及转储的
  ``oracle_512.pt`` fixture。门限：``rel_err <= 2e-3``。

两种跑法都可以：

    PYTHONPATH=coffin python -m unittest coffin/tests/test_vision.py
    # 在容器里、引擎目录内：
    python3 -m unittest tests.test_vision -v

Module: ``coffin/tests``。按设备分层：分词器/处理器与 mRoPE 的检查在任何机器
上都能跑，塔的前向需要 DCU 和 checkpoint，缺了就跳过。

Constraints:
    * 数学上精确的地方容差按 1 ULP 量级给，只有跨 dtype 时才放宽；为了放行某个
      用例去调宽容差，会把真实回归掩盖掉。
    * 共享的模型 fixture 在不同权重格式之间要关闭重开，因为两份拷贝同时放不
      进卡里。
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import unittest

import torch
import torch.nn.functional as F

try:
    import support
except ImportError:  # 直接以 `python -m unittest coffin/tests/test_vision.py` 运行时
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import support


def _vision_module():
    """取引擎的 ``models.qwen35.vision``，同时不被迫拉起整个模型族。

    走 ``import models.qwen35.vision`` 会先执行包的 ``__init__``，它再导入
    ``model.py`` -> ``safetensors``；开发机没有容器那套依赖时这一步就失败，
    所以回退到直接按路径加载文件。两条路径拿到的被测模块是同一个（它唯一
    的导入是 ``int8_weights``）。
    """
    try:
        from models.qwen35 import vision
        return vision
    except ImportError:
        for name in [n for n in sys.modules if n == "models" or n.startswith("models.")]:
            del sys.modules[name]
        path = pathlib.Path(support.ENGINE_DIR) / "models" / "qwen35" / "vision.py"
        spec = importlib.util.spec_from_file_location("qwen35_vision_under_test", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


V = _vision_module()
VisionTower = V.VisionTower


# 内联参考实现：对着公式写，不照抄引擎
def ref_pos_embed(table, h, w, merge, t=1):
    """``fast_pos_embed_interpolate``：48x48（side=sqrt(n_pos)）重采样到每张图的网格。"""
    side = int(round(table.shape[0] ** 0.5))
    views = table.reshape(side, side, -1).permute(2, 0, 1).unsqueeze(0)
    up = F.interpolate(views, size=(h, w), mode="bilinear", align_corners=True)
    flat = up[0].permute(1, 2, 0)                     # (h, w, hidden)，行主序
    blocks = flat.reshape(h // merge, merge, w // merge, merge, -1)
    out = blocks.permute(0, 2, 1, 3, 4).reshape(h * w, -1)   # 按 merge 块排序
    return out.repeat(t, 1)


def ref_hw(grid_thw, merge):
    """``rot_pos_emb`` 的 ``pos_ids``：每个 token 的 patch 网格 (row, col)，按 merge 块排序。"""
    rows, cols = [], []
    for t, h, w in grid_thw:
        mh, mw = h // merge, w // merge
        r = torch.arange(mh)[:, None, None, None] * merge + torch.arange(merge)[None, None, :, None]
        c = torch.arange(mw)[None, :, None, None] * merge + torch.arange(merge)[None, None, None, :]
        r = r.expand(mh, mw, merge, merge).reshape(-1)
        c = c.expand(mh, mw, merge, merge).reshape(-1)
        rows.append(r.repeat(t))
        cols.append(c.repeat(t))
    return torch.cat(rows), torch.cat(cols)


def ref_rope(grid_thw, merge, head_dim, theta=10000.0):
    """轴向 rope 的 cos/sin ``[N, head_dim]``，频段布局为 ``[h | w | h | w]``。"""
    spatial = head_dim // 2
    inv = 1.0 / (theta ** (torch.arange(0, spatial, 2, dtype=torch.float32) / spatial))
    rows, cols = ref_hw(grid_thw, merge)
    fh = rows.float()[:, None] * inv
    fw = cols.float()[:, None] * inv
    dup = lambda x: torch.cat([x, x], dim=-1)
    cos = dup(torch.cat([fh.cos(), fw.cos()], dim=-1))
    sin = dup(torch.cat([fh.sin(), fw.sin()], dim=-1))
    return cos, sin


def ref_rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def ref_attention(x, sd, prefix, cfg, grid_thw, cos, sin, merge):
    """把打包的非因果注意力写成一次稠密的块对角掩码矩阵乘。"""
    del merge
    heads, hd = cfg["num_heads"], cfg["hidden_size"] // cfg["num_heads"]
    n = x.shape[0]
    qkv = F.linear(x, sd[prefix + "qkv.weight"], sd[prefix + "qkv.bias"])
    q, k, v = qkv.reshape(n, 3, heads, hd).permute(1, 0, 2, 3).unbind(0)   # (n,heads,hd)
    qf, kf = q.float(), k.float()                       # HF 的 rope 在 fp32 里算
    qf = qf * cos[:, None] + ref_rotate_half(qf) * sin[:, None]
    kf = kf * cos[:, None] + ref_rotate_half(kf) * sin[:, None]
    q, k = qf.to(q.dtype), kf.to(k.dtype)
    q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)     # (heads,n,hd)
    seg = torch.zeros(n)
    off = 0
    for t, h, w in grid_thw:
        for _f in range(t):                             # 每帧一段，段间不能互相看见
            seg[off:off + h * w] = off
            off += h * w
    blocked = seg[:, None] != seg[None, :]
    scores = torch.matmul(q, k.transpose(-1, -2)) * (hd ** -0.5)
    scores = scores.masked_fill(blocked[None], float("-inf"))
    p = torch.softmax(scores, dim=-1)                   # 掩码项是 -inf，fp32 下算得稳
    out = torch.matmul(p.to(q.dtype), v)                # (heads,n,hd)
    return F.linear(out.transpose(0, 1).reshape(n, heads * hd),
                    sd[prefix + "proj.weight"],
                    sd[prefix + "proj.bias"])


def ref_forward(sd, cfg, pixel_values, grid_thw):
    """整座塔逐形状对照 HF（``Qwen3VLVisionModel.forward``）。"""
    hidden = cfg["hidden_size"]
    heads = cfg["num_heads"]
    hd = hidden // heads
    patch = cfg["patch_size"]
    tps = cfg["temporal_patch_size"]
    ch = cfg["in_channels"]
    merge = cfg["spatial_merge_size"]
    eps = 1e-6
    gt = [tuple(int(v) for v in row) for row in grid_thw.tolist()]

    x = F.linear(pixel_values, sd["patch_embed.proj.weight"].reshape(hidden, -1),
                 sd["patch_embed.proj.bias"])
    pos = torch.cat([ref_pos_embed(sd["pos_embed.weight"], h, w, merge, t) for t, h, w in gt])
    x = x + pos
    cos, sin = ref_rope(gt, merge, hd)

    for i in range(cfg["depth"]):
        p = f"blocks.{i}."
        h = F.layer_norm(x, (hidden,), sd[p + "norm1.weight"], sd[p + "norm1.bias"], eps)
        x = x + ref_attention(h, sd, p + "attn.", cfg, gt, cos, sin, merge)
        h = F.layer_norm(x, (hidden,), sd[p + "norm2.weight"], sd[p + "norm2.bias"], eps)
        h = F.linear(h, sd[p + "mlp.linear_fc1.weight"], sd[p + "mlp.linear_fc1.bias"])
        h = F.gelu(h, approximate="tanh")
        x = x + F.linear(h, sd[p + "mlp.linear_fc2.weight"], sd[p + "mlp.linear_fc2.bias"])

    merge_dim = hidden * merge * merge
    y = F.layer_norm(x, (hidden,), sd["merger.norm.weight"], sd["merger.norm.bias"], eps)
    y = y.reshape(-1, merge_dim)
    y = F.linear(y, sd["merger.linear_fc1.weight"], sd["merger.linear_fc1.bias"])
    y = F.gelu(y)                                    # 精确 erf GELU，与块内的 tanh 近似不同
    return F.linear(y, sd["merger.linear_fc2.weight"], sd["merger.linear_fc2.bias"])


# fixture 构造
def tiny_cfg(**over) -> dict:
    """能跑通前向的最小配置，``over`` 用来单独放大某一维。

    默认值都不大，是为了让 CPU 层的用例在几秒内跑完；要验证某条形状约束
    时只覆盖对应的键，其余保持最小。
    """
    cfg = {"depth": 2, "hidden_size": 64, "intermediate_size": 128, "num_heads": 4,
           "in_channels": 3, "patch_size": 2, "spatial_merge_size": 2,
           "temporal_patch_size": 2, "num_position_embeddings": 16,
           "out_hidden_size": 32, "hidden_act": "gelu_pytorch_tanh",
           "deepstack_visual_indexes": []}
    cfg.update(over)
    return cfg


def random_sd(cfg, seed=0, dtype=torch.float32):
    """按真实 checkpoint 的键名和形状造一份随机权重。

    键名必须与 ``model.visual.*`` 去掉前缀后一致，加载路径才能原样走通；
    ``mul_(0.2)`` 把激活压到个位数量级，误差界限才有意义。
    """
    g = torch.Generator().manual_seed(seed)
    hidden = cfg["hidden_size"]
    inter = cfg["intermediate_size"]
    merge_dim = hidden * cfg["spatial_merge_size"] ** 2
    patch_dim = cfg["in_channels"] * cfg["temporal_patch_size"] * cfg["patch_size"] ** 2

    def r(*shape):
        return torch.randn(*shape, generator=g).mul_(0.2).to(dtype)

    sd = {"patch_embed.proj.weight": r(hidden, cfg["in_channels"],
                                       cfg["temporal_patch_size"], cfg["patch_size"],
                                       cfg["patch_size"]),
          "patch_embed.proj.bias": r(hidden),
          "pos_embed.weight": r(cfg["num_position_embeddings"], hidden),
          "merger.norm.weight": torch.ones(hidden), "merger.norm.bias": r(hidden),
          "merger.linear_fc1.weight": r(merge_dim, merge_dim),
          "merger.linear_fc1.bias": r(merge_dim),
          "merger.linear_fc2.weight": r(cfg["out_hidden_size"], merge_dim),
          "merger.linear_fc2.bias": r(cfg["out_hidden_size"])}
    for i in range(cfg["depth"]):
        p = f"blocks.{i}."
        sd[p + "norm1.weight"] = torch.ones(hidden)
        sd[p + "norm1.bias"] = r(hidden)
        sd[p + "norm2.weight"] = torch.ones(hidden)
        sd[p + "norm2.bias"] = r(hidden)
        sd[p + "attn.qkv.weight"] = r(3 * hidden, hidden)
        sd[p + "attn.qkv.bias"] = r(3 * hidden)
        sd[p + "attn.proj.weight"] = r(hidden, hidden)
        sd[p + "attn.proj.bias"] = r(hidden)
        sd[p + "mlp.linear_fc1.weight"] = r(inter, hidden)
        sd[p + "mlp.linear_fc1.bias"] = r(inter)
        sd[p + "mlp.linear_fc2.weight"] = r(hidden, inter)
        sd[p + "mlp.linear_fc2.bias"] = r(hidden)
    del patch_dim
    return sd


def make_tower(cfg, sd=None, **kwargs):
    sd = random_sd(cfg) if sd is None else sd
    return VisionTower(sd, cfg, device="cpu", dtype=torch.float32, **kwargs)


def patch_dim(cfg) -> int:
    return cfg["in_channels"] * cfg["temporal_patch_size"] * cfg["patch_size"] ** 2


def rel_norm(got, ref) -> float:
    """相对 Frobenius 误差，对照实现的门限用的也是这个度量。"""
    return ((got - ref).norm() / ref.norm()).item()


#: 「数学相同、归约顺序不同」的容差。这些随机权重塔的激活在 10 量级，24 层
#: 累加下来 fp32 的求和顺序差异是相对 ~1e-7、绝对 ~2e-5。真正必须让它失败的
#: 是公式错误（merge 块顺序、头序、rope 布局都是 O(1)）或 GELU 用错
#: （相对 ~1e-3），所以两个界都在舍入噪声之上约 100 倍。
REL_TOL = 1e-5
ABS_TOL = 5e-5


def blocked_pixels(cfg, grid_thw):
    """为 ``grid_thw`` 造形状对的 ``pixel_values``，内容在这里无关紧要。"""
    return torch.randn(int(torch.as_tensor(grid_thw).prod(-1).sum()), patch_dim(cfg))


# CPU 层：只验公式
class PosEmbedTest(unittest.TestCase):
    """学到的 2304=48x48 位置表的双线性重采样，这是最容易静默出错的一处。"""

    def test_matches_bilinear_reference_on_real_side(self):
        """侧边取真实的 48，否则重采样的缩放比例跟 checkpoint 对不上。"""
        # 用窄的 hidden 保持开销低。
        cfg = tiny_cfg(hidden_size=8, num_heads=2, intermediate_size=16,
                       spatial_merge_size=2, num_position_embeddings=2304,
                       out_hidden_size=8, depth=0)
        tower = make_tower(cfg)
        for grid in ([[1, 16, 16]], [[1, 32, 32]], [[1, 64, 64]], [[1, 2, 2]],
                     [[1, 12, 20]]):
            grid_thw = torch.tensor(grid)
            got = tower._pos_offsets(grid_thw, torch.float32)
            ref = torch.cat([ref_pos_embed(tower.pos_embed, h, w, 2, t)
                             for t, h, w in grid_thw.tolist()])
            self.assertEqual(tuple(got.shape), tuple(ref.shape))
            self.assertLess((got - ref).abs().max().item(), 3e-5,
                            f"grid {grid}: pos-embed resampling disagrees")

    def test_multiple_images_and_frames(self):
        """多图多帧时位置嵌入必须逐段拼接，不能复用上一张图的偏移。"""
        cfg = tiny_cfg(num_position_embeddings=16, hidden_size=8, num_heads=2,
                       intermediate_size=16, out_hidden_size=8, depth=0)
        tower = make_tower(cfg)
        grid_thw = torch.tensor([[1, 4, 4], [2, 2, 4], [1, 8, 2]])
        got = tower._pos_offsets(grid_thw, torch.float32)
        ref = torch.cat([ref_pos_embed(tower.pos_embed, h, w, 2, t)
                         for t, h, w in grid_thw.tolist()])
        self.assertLess((got - ref).abs().max().item(), 3e-5)

    def test_taps_are_cached_per_size(self):
        """同一尺寸的插值结果必须命中缓存；解码时每帧都会问一次。"""
        tower = make_tower(tiny_cfg(num_position_embeddings=16))
        tower._pos_offsets(torch.tensor([[1, 4, 4]]), torch.float32)
        tower._pos_offsets(torch.tensor([[1, 4, 4]]), torch.float32)
        self.assertEqual(len(tower._pos_cache), 1)


class RopeTest(unittest.TestCase):
    """轴向 rope：16 个频率、``[h|w|h|w]`` 布局、按 merge 块排列的位置。"""

    def test_cos_sin_match_reference(self):
        """频率表或拼接顺序错一格，注意力就会看到错误的相对位置。"""
        cfg = tiny_cfg()
        tower = make_tower(cfg)
        hd = cfg["hidden_size"] // cfg["num_heads"]
        for grid_thw in (torch.tensor([[1, 4, 6]]), torch.tensor([[1, 8, 8], [2, 2, 4]])):
            cos, sin = tower._rope(grid_thw)
            rcos, rsin = ref_rope([tuple(int(v) for v in r) for r in grid_thw.tolist()], 2, hd)
            self.assertLess((cos - rcos).abs().max().item(), 1e-6)
            self.assertLess((sin - rsin).abs().max().item(), 1e-6)

    def test_merge_block_token_order(self):
        """token 必须按 2x2 merge 块内先横后纵的顺序排，HF 的 ``pos_ids`` 就是这么排的。"""
        tower = make_tower(tiny_cfg())
        rows, cols = tower._hw_positions(4, 6)
        # 前 4 个 token 是第一个 2x2 merge 块：(0,0) (0,1) (1,0) (1,1)
        self.assertEqual(rows[:4].tolist(), [0, 0, 1, 1])
        self.assertEqual(cols[:4].tolist(), [0, 1, 0, 1])
        self.assertEqual(rows[4:8].tolist(), [0, 0, 1, 1])
        self.assertEqual(cols[4:8].tolist(), [2, 3, 2, 3])

    def test_frequency_count(self):
        """频率个数由 head_dim 推出来，少一对就会和 HF 的布局错位。"""
        tower = make_tower(tiny_cfg())
        # head_dim 16 -> spatial_dim 8 -> 每轴 4 个频率
        self.assertEqual(tower.inv_freq.numel(), 4)
        self.assertAlmostEqual(float(tower.inv_freq[0]), 1.0, places=6)


class PackedAttentionTest(unittest.TestCase):
    """查询切块不能改变结果，各段之间必须互相隔离。"""

    def _tower(self, q_tile):
        """造一座塔并把查询切块宽度顶到 ``q_tile``，只给切块相关的用例用。"""
        cfg = tiny_cfg()
        tower = make_tower(cfg)
        tower._q_tile_override = q_tile
        return tower

    def test_query_tiles_match_dense_mask(self):
        """任何切块宽度都必须等价于稠密掩码参考；切块只该影响显存，不该影响数值。"""
        cfg = tiny_cfg()
        sd = random_sd(cfg)
        grid_thw = torch.tensor([[1, 8, 8], [1, 4, 8]])
        pv = blocked_pixels(cfg, grid_thw)
        ref = ref_forward(sd, cfg, pv, grid_thw)
        for tile in (3, 16, 512):
            old = V.Q_TILE
            V.Q_TILE = tile
            try:
                tower = VisionTower(sd, cfg, device="cpu", dtype=torch.float32)
                got = tower(pv, grid_thw)
            finally:
                V.Q_TILE = old
            self.assertLess(rel_norm(got, ref), REL_TOL,
                            f"Q_TILE={tile} diverges from the dense-mask reference")
            self.assertLess((got - ref).abs().max().item(), ABS_TOL)

    def test_segments_are_per_frame(self):
        """每帧自成一段，``cu_seqlens`` 按帧累加，不按图像累加。"""
        tower = make_tower(tiny_cfg())
        cu = tower.cu_seqlens(torch.tensor([[2, 4, 6], [1, 2, 2]]))
        self.assertEqual(cu.dtype, torch.int32)
        self.assertEqual(cu.tolist(), [0, 24, 48, 52])

    def test_segment_isolation(self):
        """两张图像不能互相看见：一起编码必须等于分别编码。"""
        cfg = tiny_cfg()
        sd = random_sd(cfg)
        tower = VisionTower(sd, cfg, device="cpu", dtype=torch.float32)
        g1, g2 = [1, 4, 8], [1, 6, 4]
        pv1 = blocked_pixels(cfg, torch.tensor([g1]))
        pv2 = blocked_pixels(cfg, torch.tensor([g2]))
        both = tower(torch.cat([pv1, pv2]), torch.tensor([g1, g2]))
        n1 = g1[1] * g1[2] // 4
        # M 不同时 fp32 矩阵乘的累加顺序也不同，所以按相对误差比
        # （段之间真的串了会是 O(1)，不是 O(1e-6)）。
        for got, want in ((both[:n1], tower(pv1, torch.tensor([g1]))),
                          (both[n1:], tower(pv2, torch.tensor([g2])))):
            rel = ((got - want).norm() / want.norm()).item()
            self.assertLess(rel, 1e-5)


class FullTowerTest(unittest.TestCase):
    """随机权重上的端到端：切块、rope、merge 顺序、GELU 选型。

    这些用例覆盖的是整条前向串联后仍然正确的组合，单点公式对、组合错的情况
    只能在这一层暴露。
    """

    def test_matches_reference_single_image(self):
        """单图端到端：输出形状先去掉了 merge 的降采样，再逐元素对参考。"""
        cfg = tiny_cfg(depth=3)
        sd = random_sd(cfg, seed=1)
        tower = VisionTower(sd, cfg, device="cpu", dtype=torch.float32)
        grid_thw = torch.tensor([[1, 6, 8]])
        pv = blocked_pixels(cfg, grid_thw)
        got = tower(pv, grid_thw)
        ref = ref_forward(sd, cfg, pv, grid_thw)
        self.assertEqual(tuple(got.shape), (6 * 8 // 4, cfg["out_hidden_size"]))
        self.assertLess(rel_norm(got, ref), REL_TOL)
        self.assertLess((got - ref).abs().max().item(), ABS_TOL)

    def test_matches_reference_multi_image_with_frames(self):
        """多图加多帧时 merge 之后的 token 总数必须是各段各自降采样后相加。"""
        cfg = tiny_cfg(depth=2)
        sd = random_sd(cfg, seed=2)
        tower = VisionTower(sd, cfg, device="cpu", dtype=torch.float32)
        grid_thw = torch.tensor([[1, 4, 4], [2, 2, 4]])
        pv = blocked_pixels(cfg, grid_thw)
        got = tower(pv, grid_thw)
        ref = ref_forward(sd, cfg, pv, grid_thw)
        self.assertEqual(got.shape[0], (16 + 16) // 4)
        self.assertLess(rel_norm(got, ref), REL_TOL)
        self.assertLess((got - ref).abs().max().item(), ABS_TOL)

    def test_merger_uses_exact_gelu_and_block_mlp_tanh(self):
        """两处 GELU 不能互换：换错会在该处造成 ~1e-3 的偏差，这个用例就是拿它当哨兵。"""
        cfg = tiny_cfg(depth=1)
        sd = random_sd(cfg, seed=3)
        tower = VisionTower(sd, cfg, device="cpu", dtype=torch.float32)
        grid_thw = torch.tensor([[1, 4, 4]])
        pv = blocked_pixels(cfg, grid_thw)
        ref = ref_forward(sd, cfg, pv, grid_thw)
        got = tower(pv, grid_thw)
        self.assertLess(rel_norm(got, ref), REL_TOL)
        self.assertLess((got - ref).abs().max().item(), ABS_TOL)

    def test_shape_validation(self):
        """行数与 ``grid_thw`` 不符、或 patch 宽度不对，必须在入口就抛 ``ValueError``。"""
        cfg = tiny_cfg()
        tower = make_tower(cfg)
        with self.assertRaises(ValueError):
            tower(blocked_pixels(cfg, torch.tensor([[1, 4, 4]])), torch.tensor([[1, 4, 6]]))
        with self.assertRaises(ValueError):
            tower(torch.randn(4, 7), torch.tensor([[1, 4, 4]]))


class ConfigAndLayoutTest(unittest.TestCase):
    """配置解析、键名前缀和 int8 驻留格式，这些在纯 CPU 上就能定死。"""

    def test_missing_key_names_the_tensor(self):
        """缺权重时报错必须点名缺的是哪个张量，否则 297 个键里根本查不出来。"""
        cfg = tiny_cfg()
        sd = random_sd(cfg)
        del sd["blocks.1.attn.proj.bias"]
        with self.assertRaises(KeyError) as ctx:
            VisionTower(sd, cfg, device="cpu")
        self.assertIn("blocks.1.attn.proj.bias", str(ctx.exception))

    def test_non_square_pos_embed_is_rejected(self):
        """位置表张量数不是完全平方数时无法还原成 48x48，只能拒绝而不能猜。"""
        cfg = tiny_cfg(num_position_embeddings=1000)
        with self.assertRaises(ValueError):
            VisionTower(random_sd(tiny_cfg()), cfg, device="cpu")

    def test_stripped_or_prefixed_keys_both_load(self):
        """checkpoint 里的键带 ``model.visual.`` 前缀，直接喂的裸键不带，两种都得认。"""
        cfg = tiny_cfg()
        sd = random_sd(cfg)
        prefixed = {"model.visual." + k: v for k, v in sd.items()}
        a = VisionTower(sd, cfg, device="cpu", dtype=torch.float32)
        b = VisionTower(prefixed, cfg, device="cpu", dtype=torch.float32)
        pv = blocked_pixels(cfg, torch.tensor([[1, 4, 4]]))
        grid = torch.tensor([[1, 4, 4]])
        self.assertLess((a(pv, grid) - b(pv, grid)).abs().max().item(), 1e-7)

    def test_int8_round_trip_keeps_matrices_close(self):
        """量化再反量化后矩阵要落在 0.01 以内，这是 int8 驻留可用的前提。"""
        cfg = tiny_cfg()
        sd = random_sd(cfg)
        tower = VisionTower(sd, cfg, device="cpu", dtype=torch.float32, quant=True)
        q = tower.blocks[0].qkv_w
        self.assertTrue(isinstance(q, V.QWeight))
        self.assertEqual(q.wq.dtype, torch.int8)
        deq = tower._dequant(q)
        self.assertLess((deq - sd["blocks.0.attn.qkv.weight"]).abs().max().item(), 0.01)
        # 一维张量和 gather 表保持计算 dtype，不参与量化。
        self.assertIsInstance(tower.pos_embed, torch.Tensor)
        self.assertIsInstance(tower.blocks[0].norm1_w, torch.Tensor)

    def test_int8_and_fp16_agree_loosely(self):
        """int8 塔与 fp16 塔只需大致一致：3e-2 挡住布局错，但不该卡死量化噪声。"""
        cfg = tiny_cfg(depth=2)
        sd = random_sd(cfg, seed=7)
        grid = torch.tensor([[1, 4, 4]])
        pv = blocked_pixels(cfg, grid)
        f16 = VisionTower(sd, cfg, device="cpu", dtype=torch.float32)
        q8 = VisionTower(sd, cfg, device="cpu", dtype=torch.float32, quant=True)
        a, b = f16(pv, grid), q8(pv, grid)
        self.assertLess(((a - b).norm() / a.norm()).item(), 3e-2)

    def test_summary_reports_the_checkpoint_geometry(self):
        """``summary()`` 的参数量必须和 ``num_params()`` 对得上，它是显存预算的唯一来源。"""
        cfg = tiny_cfg()
        tower = make_tower(cfg)
        s = tower.summary()
        self.assertEqual(s["params"], tower.num_params())
        self.assertFalse(s["quant"])


class VisionQuantPolicyTest(unittest.TestCase):
    """``weights.py`` 视觉分支的键名策略，纯 CPU、不需要 checkpoint。

    与卡无关、必须恒成立的三条：只有 GEMM 权重能从 int8 受益；gather 表和卷积
    核保持 fp16；前缀要保留，``Qwen35.vision_sd`` 靠它把视觉权重筛回去。
    """

    #: 每个块的二维矩阵乘个数：attn.qkv、attn.proj、mlp.linear_fc1、mlp.linear_fc2。
    #: 偏置和两个 LayerNorm 是一维的；merger 另外加 fc1 和 fc2。
    GEMMS_PER_BLOCK = V.GEMMS_PER_BLOCK

    def test_quantizable_keys(self):
        """按形状和用途区分：能走 GEMM 的量化，按下标读的 gather 表不量化。"""
        self.assertTrue(V.is_vision_quantizable("model.visual.blocks.0.attn.qkv.weight",
                                                torch.zeros(3072, 1024)))
        self.assertTrue(V.is_vision_quantizable("model.visual.blocks.23.mlp.linear_fc1.weight",
                                                torch.zeros(4096, 1024)))
        self.assertTrue(V.is_vision_quantizable("model.visual.merger.linear_fc2.weight",
                                                torch.zeros(2560, 4096)))
        # 行 gather 表按下标读，不走 GEMM，量化它只会给 gather 路径多一步反量化。
        self.assertFalse(V.is_vision_quantizable("model.visual.pos_embed.weight",
                                                 torch.zeros(2304, 1024)))
        # 一维归一化和偏置逐元素消费，总共才 ~0.3 MB，量化反而不划算。
        self.assertFalse(V.is_vision_quantizable("model.visual.blocks.0.norm1.weight",
                                                 torch.zeros(1024)))
        self.assertFalse(V.is_vision_quantizable("model.visual.blocks.0.attn.qkv.bias",
                                                 torch.zeros(3072)))

    def test_quantized_key_count_matches_the_tower(self):
        """每个块 4 个 GEMM 权重乘 24 再加 merger 的 2 个，297 个张量里量化 98 个。"""
        cfg = tiny_cfg()
        sd = random_sd(cfg)
        gemms = [k for k, v in sd.items()
                 if V.is_vision_quantizable("model.visual." + k, v)]
        self.assertEqual(len(gemms), self.GEMMS_PER_BLOCK * cfg["depth"] + 2)
        # 297 = patch_embed 2 + pos_embed 1 + 24 blocks x 12 + merger 6
        self.assertEqual(len(sd), 9 + 12 * cfg["depth"])

    def test_prefix_is_the_contract(self):
        """``model.visual.`` 是 ``Qwen35.vision_sd`` 用来筛选的契约前缀。"""
        self.assertEqual(V.VISUAL_PREFIX, "model.visual.")

    def test_vision_is_in_the_reload_order(self):
        """``engine_api.reload()`` 只重建 ``_RELOAD_ORDER`` 里列出的模块。

        通过 ``weights.py`` 间接导入却没登记的模块会让 ``reload()`` 抛错，
        而不是悄悄继续服务一个过期的类。
        """
        self.assertTrue(hasattr(V, "__name__"))
        try:
            import engine_api
        except Exception as exc:
            self.skipTest(f"engine_api not importable here: {exc}")
        self.assertIn("models.qwen35.vision", engine_api._RELOAD_ORDER)

    def test_quant_policy_applied_to_a_tiny_tower(self):
        """策略落到实际驻留上：块内只有 4 个 GEMM 变成 ``QWeight``，merger 两个，其余不动。"""
        cfg = tiny_cfg()
        sd = random_sd(cfg)
        tower = VisionTower(sd, cfg, device="cpu", dtype=torch.float32, quant=True)
        n_q = sum(1 for blk in tower.blocks for name in V._Block.__slots__
                  if isinstance(getattr(blk, name), V.QWeight))
        self.assertEqual(n_q, self.GEMMS_PER_BLOCK * cfg["depth"])
        self.assertIsInstance(tower.merger_fc1_w, V.QWeight)
        self.assertIsInstance(tower.merger_fc2_w, V.QWeight)
        self.assertIsInstance(tower.pos_embed, torch.Tensor)
        self.assertIsInstance(tower.patch_w, torch.Tensor)


# DCU + checkpoint 层：真实权重对照 transformers 的对照实现
#: 已评审的门限（VISION.md 第 7 节）。主门限是 fp32 对 fp32，这是唯一能把
#: *公式* 误差单独隔离出来的比法：这座塔 24 层会把激活推到 ~1e3，bf16 每步
#: 舍入 ~8e-3 并逐层累积 —— 一个 bf16 对照实现本身就离 fp32 有 1e-1，所以
#: 「对 bf16 对照实现做到 rel_err <= 2e-3」任何实现都达不到。
#:
#: DCU 上 256^2 实测：1.203e-05。残差来自归约顺序而不是公式 —— rocBLAS 切分
#: fp32 的 K 维方式和 oracle 里 HF 走的 CPU 路径不同，而这座塔叠了 24 层。
#: 它相对输出幅度约 7e-6，比任何公式错误（merge 顺序、头序、rope 布局都是
#: O(1)）低约 1e4 倍，所以门限定在 2e-5；收到 1e-5 只会变成在量两个库的求和
#: 顺序之差。
GATE_FP32 = 2e-5
GATE_FP16 = 3e-2       # 引擎 fp16 对 HF fp32 —— 这是 fp16 的精度代价（HF 的 fp16
                       #                      自己离 fp32 也有 1.58e-2）
GATE_BF16_SMOKE = 1.5e-1   # 引擎 fp16 对 HF bf16 —— 冒烟用：能跑、量级对、顺序对，
                           #                      不是数值门限

ORACLE_FIXTURE = pathlib.Path("/root/private_data/engine_vision/oracle_512.pt")


def _load_visual(dtype):
    """从 checkpoint 分片里取出 ``model.visual.*``，保留前缀，并转成 ``dtype``。"""
    import json
    from safetensors import safe_open

    md = support.MODEL_DIR
    index = json.loads((md / "model.safetensors.index.json").read_text())["weight_map"]
    shards = sorted({v for k, v in index.items() if k.startswith("model.visual.")})
    sd = {}
    for shard in shards:
        with safe_open(str(md / shard), "pt") as f:
            for k in f.keys():
                if k.startswith("model.visual."):
                    sd[k] = f.get_tensor(k).to(dtype)
    return sd


def _hf_vision_model(dtype):
    """装配 4.57.6 的 HF 对照实现模型，并装入 checkpoint 的视觉权重。

    缺键和多键都直接抛 ``AssertionError``：对照实现装错了权重，后面的误差
    数字就毫无意义。
    """
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    import json
    cfg = json.loads((support.MODEL_DIR / "config.json").read_text())
    vc = {k: v for k, v in cfg["vision_config"].items() if k != "model_type"}
    hf = Qwen3VLVisionModel(Qwen3VLVisionConfig(**vc)).to("cuda", dtype=dtype).eval()
    sd = _load_visual(dtype)
    missing, unexpected = hf.load_state_dict(
        {k[len("model.visual."):]: v for k, v in sd.items()}, strict=False)
    if missing or unexpected:
        raise AssertionError(f"oracle load: missing={missing[:3]} unexpected={unexpected[:3]}")
    del sd
    return hf, cfg["vision_config"]


def _hf_forward(hf, pv, grid, dtype):
    """跑一次 HF 塔，按 ``float().cpu()`` 返回，好与引擎侧逐元素比。

    不同版本的 HF 前向返回值形状不一（元组与带 ``pooler_output`` 的对象），
    两种情况都要认。
    """
    with torch.no_grad():
        out = hf(pv.to("cuda", dtype=dtype), grid_thw=grid.to("cuda"))
    out = out[0] if isinstance(out, (tuple, list)) else out.pooler_output
    return out.float().cpu()


def _sample_image(size):
    """造一张 ``size`` 见方的合成图：横纵渐变加一个红方块，确定且无外部依赖。"""
    import numpy as np
    from PIL import Image
    a = np.zeros((size, size, 3), np.uint8)
    a[..., 0] = np.linspace(0, 255, size, dtype=np.uint8)[None, :]
    a[..., 1] = np.linspace(0, 255, size, dtype=np.uint8)[:, None]
    a[size // 4:size // 2, size // 4:size // 2] = [255, 0, 0]
    return Image.fromarray(a)


@unittest.skipUnless(support.has_cuda() and support.has_model(),
                     f"needs a DCU and the checkpoint at {support.MODEL_DIR}")
class RealWeightsTest(unittest.TestCase):
    """真实权重上的公式正确性、精度代价与常驻显存占用。

    ``VISION_TEST_SIZES`` 限定跑几个尺寸（默认一个：整套要能塞进共享卡上的
    一个槽位）；``VISION_SIZES=256,512,1024`` 会把三个都跑一遍。
    """

    @classmethod
    def setUpClass(cls):
        """先数视觉张量个数：对不上 297 说明 checkpoint 不完整，整类直接跳过。"""
        sd = _load_visual(torch.float16)
        if len(sd) != 297:
            raise unittest.SkipTest(f"expected 297 vision tensors, found {len(sd)}")
        cls.n_tensors = len(sd)
        del sd
        cls.sizes = [int(s) for s in
                     os.environ.get("VISION_TEST_SIZES", "512").split(",") if s.strip()]

    def _features(self, size):
        """用真实的 processor 走一遍预处理，拿到 ``pixel_values`` 和 ``image_grid_thw``。

        预处理必须和线上同源，手工拼张量会绕过 patch 切分和 resize 的细节。
        """
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(str(support.MODEL_DIR))
        text = proc.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"},
                                          {"type": "text", "text": "Describe."}]}],
            tokenize=False, add_generation_prompt=True)
        out = proc(text=[text], images=[_sample_image(size)], return_tensors="pt")
        return out["pixel_values"], out["image_grid_thw"]

    def test_primary_gate_fp32_matches_transformers(self):
        """引擎 fp32 对 HF fp32：公式门限，完全没有 dtype 噪声。"""
        hf, vcfg = _hf_vision_model(torch.float32)
        try:
            for size in self.sizes:
                with self.subTest(size=size):
                    pv, grid = self._features(size)
                    ref = _hf_forward(hf, pv, grid, torch.float32)
                    tower = VisionTower(_load_visual(torch.float32), vcfg,
                                        device="cuda", dtype=torch.float32)
                    with torch.no_grad():
                        got = tower(pv.to("cuda"), grid).float().cpu()
                    e = rel_norm(got, ref)
                    print(f"\n  {size}x{size} seq={pv.shape[0]}: "
                          f"engine fp32 vs HF fp32 rel_err={e:.2e} (gate {GATE_FP32:.0e})")
                    self.assertEqual(tuple(got.shape), tuple(ref.shape))
                    self.assertLessEqual(e, GATE_FP32)
                    del tower, got, ref, pv, grid
                    torch.cuda.empty_cache()
        finally:
            del hf
            torch.cuda.empty_cache()

    def test_fp16_precision_cost(self):
        """引擎 fp16 对 HF fp32 要落在 3e-2 内，并与 HF 自己用 fp16 的代价对照。"""
        hf, vcfg = _hf_vision_model(torch.float32)
        try:
            for size in self.sizes:
                with self.subTest(size=size):
                    pv, grid = self._features(size)
                    ref = _hf_forward(hf, pv, grid, torch.float32)
                    tower = VisionTower(_load_visual(torch.float16), vcfg,
                                        device="cuda", dtype=torch.float16)
                    with torch.no_grad():
                        got = tower(pv, grid).float().cpu()
                    e = rel_norm(got, ref)
                    print(f"\n  {size}x{size}: engine fp16 vs HF fp32 "
                          f"rel_err={e:.2e} (gate {GATE_FP16:.0e})")
                    self.assertLessEqual(e, GATE_FP16)
                    del tower, got, ref, pv, grid
                    torch.cuda.empty_cache()
        finally:
            del hf
            torch.cuda.empty_cache()

    def test_bf16_oracle_smoke(self):
        """引擎 fp16 对 HF bf16：只验量级和顺序，门限说明见 ``GATE_*`` 注释。"""
        hf, vcfg = _hf_vision_model(torch.bfloat16)
        try:
            for size in self.sizes:
                with self.subTest(size=size):
                    pv, grid = self._features(size)
                    ref = _hf_forward(hf, pv, grid, torch.bfloat16)
                    tower = VisionTower(_load_visual(torch.float16), vcfg,
                                        device="cuda", dtype=torch.float16)
                    with torch.no_grad():
                        got = tower(pv, grid).float().cpu()
                    e = rel_norm(got, ref)
                    print(f"\n  {size}x{size}: engine fp16 vs HF bf16 "
                          f"rel_err={e:.2e} (smoke gate {GATE_BF16_SMOKE:.1e})")
                    self.assertEqual(tuple(got.shape), tuple(ref.shape))
                    self.assertTrue(torch.isfinite(got).all())
                    self.assertLessEqual(e, GATE_BF16_SMOKE)
                    del tower, got, ref, pv, grid
                    torch.cuda.empty_cache()
        finally:
            del hf
            torch.cuda.empty_cache()

    def test_against_fixture_oracle_512(self):
        """对 M0 转储的 fixture：只验形状和量级，不卡 dtype 精度。"""
        if not ORACLE_FIXTURE.exists():
            self.skipTest(f"no oracle fixture at {ORACLE_FIXTURE}")
        o = torch.load(ORACLE_FIXTURE, map_location="cpu")
        import json
        cfg = json.loads((support.MODEL_DIR / "config.json").read_text())["vision_config"]
        tower = VisionTower(_load_visual(torch.float16), cfg, device="cuda",
                            dtype=torch.float16)
        with torch.no_grad():
            got = tower(o["pixel_values"].to("cuda", dtype=torch.float16),
                        o["grid"].to("cuda")).float().cpu()
        ref = o["embeds"].float()
        self.assertEqual(tuple(got.shape), tuple(ref.shape))
        e = rel_norm(got, ref)
        print(f"\n  fixture oracle_512.pt ({ref.dtype}): rel_err={e:.2e}")
        # 这份 fixture 是 bf16 转储，同样带着 1e-1 的底噪；所以断言量级，
        # 而不是断言一个很紧的界。
        self.assertLessEqual(e, GATE_BF16_SMOKE)
        self.assertAlmostEqual(float(got.abs().mean()), float(ref.abs().mean()),
                               delta=0.1 * float(ref.abs().mean()))


@unittest.skipUnless(support.has_cuda() and support.has_model(),
                     f"needs a DCU and the checkpoint at {support.MODEL_DIR}")
class WeightLoadingTest(unittest.TestCase):
    """``weights.py`` 的视觉分支：常驻权重字典里必须有什么、必须没有什么。"""

    def test_vision_branch_adds_the_tower_and_keeps_the_prefix(self):
        """默认不量化视觉塔，但键前缀保留，文本侧照旧走 int8。"""
        import json
        import models.qwen35.weights as W

        cfg = json.loads((support.MODEL_DIR / "config.json").read_text())
        self.assertIn("vision_config", cfg)
        sd = W.load_weights(support.MODEL_DIR, quant=True)
        try:
            vis = [k for k in sd if k.startswith("model.visual.")]
            self.assertEqual(len(vis), 297)
            # 视觉分支不能混进文本的 int8 路径：量化视觉塔要显式开启
            # （vision_quant），因为它是计算受限而不是带宽受限。
            self.assertFalse(any(isinstance(sd[k], W.QWeight) for k in vis))
            # 文本侧仍然量化，保持原有行为。
            text_q = [k for k in sd if not k.startswith("model.visual.")
                      and isinstance(sd[k], W.QWeight)]
            self.assertGreater(len(text_q), 100)
            n_bytes = sum(sd[k].numel() * 2 for k in vis)
            print(f"\n  vision tensors {len(vis)}, fp16 {n_bytes / 1024**3:.3f} GiB")
        finally:
            sd = None
            torch.cuda.empty_cache()

    def test_vision_none_skips_the_tower(self):
        """``vision=False`` 时一个视觉张量都不该进常驻字典，语言侧不受影响。"""
        import models.qwen35.weights as W
        sd = W.load_weights(support.MODEL_DIR, vision=False)
        try:
            vis = [k for k in sd if k.startswith("model.visual.")]
            self.assertEqual(vis, [])
            self.assertIn("model.language_model.norm.weight", sd)
        finally:
            sd = None
            torch.cuda.empty_cache()

    def test_vision_quant_key_set(self):
        """int8 驻留：只量化 GEMM 权重，gather 表和一维张量保持原 dtype。"""
        import models.qwen35.weights as W
        from models.qwen35.vision import is_vision_quantizable
        self.assertFalse(is_vision_quantizable("model.visual.pos_embed.weight",
                                               torch.zeros(2304, 1024)))
        self.assertFalse(is_vision_quantizable("model.visual.patch_embed.proj.weight",
                                               torch.zeros(1024, 3, 2, 16, 16)))
        self.assertTrue(is_vision_quantizable("model.visual.blocks.0.attn.qkv.weight",
                                              torch.zeros(3072, 1024)))
        sd = W.load_weights(support.MODEL_DIR, vision_quant=True)
        try:
            vis = {k: v for k, v in sd.items() if k.startswith("model.visual.")}
            n_q = sum(1 for v in vis.values() if isinstance(v, W.QWeight))
            n_bytes = sum(v.nbytes() if isinstance(v, W.QWeight)
                          else v.numel() * v.element_size() for v in vis.values())
            print(f"\n  int8 tower: {n_q} QWeight of {len(vis)} tensors, "
                  f"{n_bytes / 1024**3:.3f} GiB")
            # 每块 4 个矩阵乘乘 24 块，加 merger 的 2 个 fc。
            self.assertEqual(n_q, 4 * 24 + 2)
            # 0.307 GiB 的 int8 加上 0.008 GiB 的一维/归一化/gather 残留，总量仍在一张卡放得下。
            self.assertLess(n_bytes, 0.33 * 1024**3)
        finally:
            sd = None
            torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
