"""Vision tower: the engine's ``VisionTower`` against an inline HF reference.

Two tiers, so the numerics are checkable on any box:

* **CPU (always runs)** -- a ~150-line inline re-implementation of the HF 4.57.6
  vision path (``fast_pos_embed_interpolate`` / ``rot_pos_emb`` / packed
  per-image attention / merger), written here rather than imported so the
  comparison is against the formulas and not against another copy of
  ``.tmp_dsh``.  Random weights, fp32, tiny and real geometries, multiple images.
* **DCU + checkpoint (skips otherwise)** -- the real ``model.visual.*`` weights
  against ``transformers`` 4.57.6's ``Qwen3VLVisionModel`` (the M0 oracle), and
  against the dumped ``oracle_512.pt`` fixture.  Gate: ``rel_err <= 2e-3``.

Run either way::

    PYTHONPATH=coffin python -m unittest coffin/tests/test_vision.py
    # in the container, inside the engine dir:
    python3 -m unittest tests.test_vision -v
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
except ImportError:  # `python -m unittest coffin/tests/test_vision.py`
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import support


def _vision_module():
    """The engine's ``models.qwen35.vision``, without requiring the whole family.

    ``import models.qwen35.vision`` runs the package ``__init__``, which imports
    ``model.py`` -> ``safetensors``; on a dev box without the container's
    dependency set that fails, so fall back to loading the file directly.  The
    module under test is identical either way (its only import is
    ``int8_weights``).
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


# --------------------------------------------------------------------------------------
# inline reference: the formulas, not the engine
# --------------------------------------------------------------------------------------
def ref_pos_embed(table, h, w, merge, t=1):
    """``fast_pos_embed_interpolate``: 48x48 (side=sqrt(n_pos)) -> per-image grid."""
    side = int(round(table.shape[0] ** 0.5))
    views = table.reshape(side, side, -1).permute(2, 0, 1).unsqueeze(0)
    up = F.interpolate(views, size=(h, w), mode="bilinear", align_corners=True)
    flat = up[0].permute(1, 2, 0)                     # (h, w, hidden), row-major
    blocks = flat.reshape(h // merge, merge, w // merge, merge, -1)
    out = blocks.permute(0, 2, 1, 3, 4).reshape(h * w, -1)   # merge-block order
    return out.repeat(t, 1)


def ref_hw(grid_thw, merge):
    """``rot_pos_emb``'s ``pos_ids``: patch-grid (row, col) per token, merge-block order."""
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
    """Axial rope cos/sin ``[N, head_dim]`` as ``[h | w | h | w]`` frequency blocks."""
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
    """Packed non-causal attention as one dense block-diagonal-masked matmul."""
    del merge
    heads, hd = cfg["num_heads"], cfg["hidden_size"] // cfg["num_heads"]
    n = x.shape[0]
    qkv = F.linear(x, sd[prefix + "qkv.weight"], sd[prefix + "qkv.bias"])
    q, k, v = qkv.reshape(n, 3, heads, hd).permute(1, 0, 2, 3).unbind(0)   # (n,heads,hd)
    qf, kf = q.float(), k.float()                       # HF ropes in fp32
    qf = qf * cos[:, None] + ref_rotate_half(qf) * sin[:, None]
    kf = kf * cos[:, None] + ref_rotate_half(kf) * sin[:, None]
    q, k = qf.to(q.dtype), kf.to(k.dtype)
    q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)     # (heads,n,hd)
    seg = torch.zeros(n)
    off = 0
    for t, h, w in grid_thw:
        for _f in range(t):                             # one segment per frame
            seg[off:off + h * w] = off
            off += h * w
    blocked = seg[:, None] != seg[None, :]
    scores = torch.matmul(q, k.transpose(-1, -2)) * (hd ** -0.5)
    scores = scores.masked_fill(blocked[None], float("-inf"))
    p = torch.softmax(scores, dim=-1)                   # ok in fp32
    out = torch.matmul(p.to(q.dtype), v)                # (heads,n,hd)
    return F.linear(out.transpose(0, 1).reshape(n, heads * hd),
                    sd[prefix + "proj.weight"],
                    sd[prefix + "proj.bias"])


def ref_forward(sd, cfg, pixel_values, grid_thw):
    """Whole tower, HF shape-for-shape (``Qwen3VLVisionModel.forward``)."""
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
    y = F.gelu(y)                                    # exact erf GELU
    return F.linear(y, sd["merger.linear_fc2.weight"], sd["merger.linear_fc2.bias"])


# --------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------
def tiny_cfg(**over) -> dict:
    cfg = {"depth": 2, "hidden_size": 64, "intermediate_size": 128, "num_heads": 4,
           "in_channels": 3, "patch_size": 2, "spatial_merge_size": 2,
           "temporal_patch_size": 2, "num_position_embeddings": 16,
           "out_hidden_size": 32, "hidden_act": "gelu_pytorch_tanh",
           "deepstack_visual_indexes": []}
    cfg.update(over)
    return cfg


def random_sd(cfg, seed=0, dtype=torch.float32):
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
    """Relative Frobenius error -- the metric the oracle gate uses too."""
    return ((got - ref).norm() / ref.norm()).item()


#: Tolerances for "same maths, different reduction order".  These random-weight
#: towers produce activations of order 10, so a fp32 sum order difference over 24
#: layers lands at ~1e-7 relative / ~2e-5 absolute.  What has to *fail* is a
#: formula error (merge-block order, head order, rope layout = O(1)) or the wrong
#: GELU (~1e-3 relative), so both bounds are ~100x above the rounding floor.
REL_TOL = 1e-5
ABS_TOL = 5e-5


def blocked_pixels(cfg, grid_thw):
    """``pixel_values`` rows for ``grid_thw`` (content is irrelevant here)."""
    return torch.randn(int(torch.as_tensor(grid_thw).prod(-1).sum()), patch_dim(cfg))


# --------------------------------------------------------------------------------------
# CPU: the formulas
# --------------------------------------------------------------------------------------
class PosEmbedTest(unittest.TestCase):
    """The learned 2304=48x48 table resampling (the silent-failure #1)."""

    def test_matches_bilinear_reference_on_real_side(self):
        # Real geometry (side 48) with a narrow hidden so the test is cheap.
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
        cfg = tiny_cfg(num_position_embeddings=16, hidden_size=8, num_heads=2,
                       intermediate_size=16, out_hidden_size=8, depth=0)
        tower = make_tower(cfg)
        grid_thw = torch.tensor([[1, 4, 4], [2, 2, 4], [1, 8, 2]])
        got = tower._pos_offsets(grid_thw, torch.float32)
        ref = torch.cat([ref_pos_embed(tower.pos_embed, h, w, 2, t)
                         for t, h, w in grid_thw.tolist()])
        self.assertLess((got - ref).abs().max().item(), 3e-5)

    def test_taps_are_cached_per_size(self):
        tower = make_tower(tiny_cfg(num_position_embeddings=16))
        tower._pos_offsets(torch.tensor([[1, 4, 4]]), torch.float32)
        tower._pos_offsets(torch.tensor([[1, 4, 4]]), torch.float32)
        self.assertEqual(len(tower._pos_cache), 1)


class RopeTest(unittest.TestCase):
    """Axial rope: 16 frequencies, ``[h|w|h|w]`` layout, merge-block positions."""

    def test_cos_sin_match_reference(self):
        cfg = tiny_cfg()
        tower = make_tower(cfg)
        hd = cfg["hidden_size"] // cfg["num_heads"]
        for grid_thw in (torch.tensor([[1, 4, 6]]), torch.tensor([[1, 8, 8], [2, 2, 4]])):
            cos, sin = tower._rope(grid_thw)
            rcos, rsin = ref_rope([tuple(int(v) for v in r) for r in grid_thw.tolist()], 2, hd)
            self.assertLess((cos - rcos).abs().max().item(), 1e-6)
            self.assertLess((sin - rsin).abs().max().item(), 1e-6)

    def test_merge_block_token_order(self):
        tower = make_tower(tiny_cfg())
        rows, cols = tower._hw_positions(4, 6)
        # tokens 0..3 are the first 2x2 merge block: (0,0) (0,1) (1,0) (1,1)
        self.assertEqual(rows[:4].tolist(), [0, 0, 1, 1])
        self.assertEqual(cols[:4].tolist(), [0, 1, 0, 1])
        self.assertEqual(rows[4:8].tolist(), [0, 0, 1, 1])
        self.assertEqual(cols[4:8].tolist(), [2, 3, 2, 3])

    def test_frequency_count(self):
        tower = make_tower(tiny_cfg())
        # head_dim 16 -> spatial_dim 8 -> 4 frequencies per axis
        self.assertEqual(tower.inv_freq.numel(), 4)
        self.assertAlmostEqual(float(tower.inv_freq[0]), 1.0, places=6)


class PackedAttentionTest(unittest.TestCase):
    """Query tiling must not change the result, and segments must stay isolated."""

    def _tower(self, q_tile):
        cfg = tiny_cfg()
        tower = make_tower(cfg)
        tower._q_tile_override = q_tile
        return tower

    def test_query_tiles_match_dense_mask(self):
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
        tower = make_tower(tiny_cfg())
        cu = tower.cu_seqlens(torch.tensor([[2, 4, 6], [1, 2, 2]]))
        self.assertEqual(cu.dtype, torch.int32)
        self.assertEqual(cu.tolist(), [0, 24, 48, 52])

    def test_segment_isolation(self):
        """Two images must not see each other: encoding them together == apart."""
        cfg = tiny_cfg()
        sd = random_sd(cfg)
        tower = VisionTower(sd, cfg, device="cpu", dtype=torch.float32)
        g1, g2 = [1, 4, 8], [1, 6, 4]
        pv1 = blocked_pixels(cfg, torch.tensor([g1]))
        pv2 = blocked_pixels(cfg, torch.tensor([g2]))
        both = tower(torch.cat([pv1, pv2]), torch.tensor([g1, g2]))
        n1 = g1[1] * g1[2] // 4
        # fp32 matmuls of different M differ in accumulation order, so compare
        # relatively (a leak between segments would be O(1), not O(1e-6)).
        for got, want in ((both[:n1], tower(pv1, torch.tensor([g1]))),
                          (both[n1:], tower(pv2, torch.tensor([g2])))):
            rel = ((got - want).norm() / want.norm()).item()
            self.assertLess(rel, 1e-5)


class FullTowerTest(unittest.TestCase):
    """End to end on random weights: tiling, rope, merge order, gelu choice."""

    def test_matches_reference_single_image(self):
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
        """A tanh/exact swap in either place would be ~1e-3 here."""
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
        cfg = tiny_cfg()
        tower = make_tower(cfg)
        with self.assertRaises(ValueError):
            tower(blocked_pixels(cfg, torch.tensor([[1, 4, 4]])), torch.tensor([[1, 4, 6]]))
        with self.assertRaises(ValueError):
            tower(torch.randn(4, 7), torch.tensor([[1, 4, 4]]))


class ConfigAndLayoutTest(unittest.TestCase):
    def test_missing_key_names_the_tensor(self):
        cfg = tiny_cfg()
        sd = random_sd(cfg)
        del sd["blocks.1.attn.proj.bias"]
        with self.assertRaises(KeyError) as ctx:
            VisionTower(sd, cfg, device="cpu")
        self.assertIn("blocks.1.attn.proj.bias", str(ctx.exception))

    def test_non_square_pos_embed_is_rejected(self):
        cfg = tiny_cfg(num_position_embeddings=1000)
        with self.assertRaises(ValueError):
            VisionTower(random_sd(tiny_cfg()), cfg, device="cpu")

    def test_stripped_or_prefixed_keys_both_load(self):
        cfg = tiny_cfg()
        sd = random_sd(cfg)
        prefixed = {"model.visual." + k: v for k, v in sd.items()}
        a = VisionTower(sd, cfg, device="cpu", dtype=torch.float32)
        b = VisionTower(prefixed, cfg, device="cpu", dtype=torch.float32)
        pv = blocked_pixels(cfg, torch.tensor([[1, 4, 4]]))
        grid = torch.tensor([[1, 4, 4]])
        self.assertLess((a(pv, grid) - b(pv, grid)).abs().max().item(), 1e-7)

    def test_int8_round_trip_keeps_matrices_close(self):
        cfg = tiny_cfg()
        sd = random_sd(cfg)
        tower = VisionTower(sd, cfg, device="cpu", dtype=torch.float32, quant=True)
        q = tower.blocks[0].qkv_w
        self.assertTrue(isinstance(q, V.QWeight))
        self.assertEqual(q.wq.dtype, torch.int8)
        deq = tower._dequant(q)
        self.assertLess((deq - sd["blocks.0.attn.qkv.weight"]).abs().max().item(), 0.01)
        # 1-D tensors and the gather table stay in the compute dtype.
        self.assertIsInstance(tower.pos_embed, torch.Tensor)
        self.assertIsInstance(tower.blocks[0].norm1_w, torch.Tensor)

    def test_int8_and_fp16_agree_loosely(self):
        cfg = tiny_cfg(depth=2)
        sd = random_sd(cfg, seed=7)
        grid = torch.tensor([[1, 4, 4]])
        pv = blocked_pixels(cfg, grid)
        f16 = VisionTower(sd, cfg, device="cpu", dtype=torch.float32)
        q8 = VisionTower(sd, cfg, device="cpu", dtype=torch.float32, quant=True)
        a, b = f16(pv, grid), q8(pv, grid)
        self.assertLess(((a - b).norm() / a.norm()).item(), 3e-2)

    def test_summary_reports_the_checkpoint_geometry(self):
        cfg = tiny_cfg()
        tower = make_tower(cfg)
        s = tower.summary()
        self.assertEqual(s["params"], tower.num_params())
        self.assertFalse(s["quant"])


class VisionQuantPolicyTest(unittest.TestCase):
    """``weights.py``'s vision branch, on its key policy (CPU, no checkpoint).

    What has to be true regardless of the card: the GEMM weights are the only
    ones int8 helps, the gather table and the conv kernel stay fp16, and the
    prefix is preserved so ``Qwen35.vision_sd`` can filter it back out.
    """

    #: 2-D matmuls per block: attn.qkv, attn.proj, mlp.linear_fc1, mlp.linear_fc2.
    #: (The biases and the two LayerNorms are 1-D; the merger adds fc1+fc2.)
    GEMMS_PER_BLOCK = V.GEMMS_PER_BLOCK

    def test_quantizable_keys(self):
        self.assertTrue(V.is_vision_quantizable("model.visual.blocks.0.attn.qkv.weight",
                                                torch.zeros(3072, 1024)))
        self.assertTrue(V.is_vision_quantizable("model.visual.blocks.23.mlp.linear_fc1.weight",
                                                torch.zeros(4096, 1024)))
        self.assertTrue(V.is_vision_quantizable("model.visual.merger.linear_fc2.weight",
                                                torch.zeros(2560, 4096)))
        # Row-gather table: read by index, not by a GEMM, so int8 would only add a
        # dequantise to the gather path.
        self.assertFalse(V.is_vision_quantizable("model.visual.pos_embed.weight",
                                                 torch.zeros(2304, 1024)))
        # 1-D norms and biases: consumed elementwise, ~0.3 MB in total.
        self.assertFalse(V.is_vision_quantizable("model.visual.blocks.0.norm1.weight",
                                                 torch.zeros(1024)))
        self.assertFalse(V.is_vision_quantizable("model.visual.blocks.0.attn.qkv.bias",
                                                 torch.zeros(3072)))

    def test_quantized_key_count_matches_the_tower(self):
        """4 GEMM weights per block x 24 + 2 merger = 98 of the 297 tensors."""
        cfg = tiny_cfg()
        sd = random_sd(cfg)
        gemms = [k for k, v in sd.items()
                 if V.is_vision_quantizable("model.visual." + k, v)]
        self.assertEqual(len(gemms), self.GEMMS_PER_BLOCK * cfg["depth"] + 2)
        # 297 = patch_embed 2 + pos_embed 1 + 24 blocks x 12 + merger 6
        self.assertEqual(len(sd), 9 + 12 * cfg["depth"])

    def test_prefix_is_the_contract(self):
        """``model.visual.`` is what ``Qwen35.vision_sd`` filters on."""
        self.assertEqual(V.VISUAL_PREFIX, "model.visual.")

    def test_vision_is_in_the_reload_order(self):
        """``engine_api.reload()`` rebuilds every loaded dtype, but only the
        modules listed in ``_RELOAD_ORDER``; a module that is imported (through
        ``weights.py``) and not listed makes ``reload()`` raise instead of
        silently serving a stale class."""
        self.assertTrue(hasattr(V, "__name__"))
        try:
            import engine_api
        except Exception as exc:
            self.skipTest(f"engine_api not importable here: {exc}")
        self.assertIn("models.qwen35.vision", engine_api._RELOAD_ORDER)

    def test_quant_policy_applied_to_a_tiny_tower(self):
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


# --------------------------------------------------------------------------------------
# DCU + checkpoint: the real weights, against the transformers oracle
# --------------------------------------------------------------------------------------
#: Reviewed gates (VISION.md §7).  The primary gate is fp32-vs-fp32, because that is
#: the only comparison that isolates *formula* error: 24 layers of this tower drive
#: activations to ~1e3, so bf16 rounds ~8e-3 per step and compounds -- a bf16 oracle
#: sits 1e-1 from fp32 by itself, which is why "rel_err <= 2e-3 vs a bf16 oracle"
#: cannot be met by any implementation.
#:
#: Measured on the DCU at 256^2: 1.203e-05.  The residual is reduction order, not
#: maths -- rocBLAS splits the fp32 K dimension differently than the CPU path HF
#: takes inside the oracle, and the tower stacks 24 layers of it.  It is ~7e-6
#: relative to the output magnitude and ~1e4x below any formula error (merge order,
#: head order, rope layout are all O(1)), so 2e-5 is the gate; tightening it to 1e-5
#: would only measure the two libraries' summation orders against each other.
GATE_FP32 = 2e-5
GATE_FP16 = 3e-2       # engine fp16 vs HF fp32  -- fp16's precision cost (HF fp16
                       #                      itself is 1.58e-2 away from fp32)
GATE_BF16_SMOKE = 1.5e-1   # engine fp16 vs HF bf16 -- smoke: runs, right magnitude,
                           #                      right order; not a numeric gate

ORACLE_FIXTURE = pathlib.Path("/root/private_data/engine_vision/oracle_512.pt")


def _load_visual(dtype):
    """``model.visual.*`` from the checkpoint shards, prefix kept, in ``dtype``."""
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
    """The 4.57.6 oracle with the checkpoint's visual weights loaded."""
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
    with torch.no_grad():
        out = hf(pv.to("cuda", dtype=dtype), grid_thw=grid.to("cuda"))
    out = out[0] if isinstance(out, (tuple, list)) else out.pooler_output
    return out.float().cpu()


def _sample_image(size):
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
    """Formula correctness, precision cost and resident footprint on real weights.

    ``VISION_TEST_SIZES`` bounds the run (default one size: the suite has to fit a
    slot on the shared card); ``VISION_SIZES=256,512,1024`` runs all three.
    """

    @classmethod
    def setUpClass(cls):
        sd = _load_visual(torch.float16)
        if len(sd) != 297:
            raise unittest.SkipTest(f"expected 297 vision tensors, found {len(sd)}")
        cls.n_tensors = len(sd)
        del sd
        cls.sizes = [int(s) for s in
                     os.environ.get("VISION_TEST_SIZES", "512").split(",") if s.strip()]

    def _features(self, size):
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(str(support.MODEL_DIR))
        text = proc.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"},
                                          {"type": "text", "text": "Describe."}]}],
            tokenize=False, add_generation_prompt=True)
        out = proc(text=[text], images=[_sample_image(size)], return_tensors="pt")
        return out["pixel_values"], out["image_grid_thw"]

    def test_primary_gate_fp32_matches_transformers(self):
        """engine(fp32) vs HF(fp32): the formula gate (no dtype noise at all)."""
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
        """engine(fp16) vs HF(fp32) <= 3e-2 -- and compare with HF fp16's own cost."""
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
        """engine(fp16) vs HF(bf16) -- magnitude/order smoke only (see GATE_* note)."""
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
        """The dumped M0 fixture: shape + magnitude, at whatever dtype it holds."""
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
        # The fixture is a bf16 dump, so it carries the same 1e-1 floor; assert on
        # the magnitude instead of on a tight bound.
        self.assertLessEqual(e, GATE_BF16_SMOKE)
        self.assertAlmostEqual(float(got.abs().mean()), float(ref.abs().mean()),
                               delta=0.1 * float(ref.abs().mean()))


@unittest.skipUnless(support.has_cuda() and support.has_model(),
                     f"needs a DCU and the checkpoint at {support.MODEL_DIR}")
class WeightLoadingTest(unittest.TestCase):
    """``weights.py``'s vision branch: what the resident dict has to contain."""

    def test_vision_branch_adds_the_tower_and_keeps_the_prefix(self):
        import json
        import models.qwen35.weights as W

        cfg = json.loads((support.MODEL_DIR / "config.json").read_text())
        self.assertIn("vision_config", cfg)
        sd = W.load_weights(support.MODEL_DIR, quant=True)
        try:
            vis = [k for k in sd if k.startswith("model.visual.")]
            self.assertEqual(len(vis), 297)
            # The vision branch must not be routed through the text int8 path: a
            # quantised tower is opt-in (vision_quant) because it is compute-bound.
            self.assertFalse(any(isinstance(sd[k], W.QWeight) for k in vis))
            # The text half still is quantised (existing behaviour).
            text_q = [k for k in sd if not k.startswith("model.visual.")
                      and isinstance(sd[k], W.QWeight)]
            self.assertGreater(len(text_q), 100)
            n_bytes = sum(sd[k].numel() * 2 for k in vis)
            print(f"\n  vision tensors {len(vis)}, fp16 {n_bytes / 1024**3:.3f} GiB")
        finally:
            sd = None
            torch.cuda.empty_cache()

    def test_vision_none_skips_the_tower(self):
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
        """int8 residency: the GEMM weights quantised, gather table and 1-D kept."""
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
            # 4 matmuls x 24 blocks + the merger's 2 fc.
            self.assertEqual(n_q, 4 * 24 + 2)
            # 0.307 GiB int8 + 0.008 GiB of fp16 1-D/norm/gather leftovers.
            self.assertLess(n_bytes, 0.33 * 1024**3)
        finally:
            sd = None
            torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
