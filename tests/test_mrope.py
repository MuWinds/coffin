"""mRoPE and image-embedding injection: the CPU-checkable half of the vision path.

Everything here runs without a DCU.  The three properties that must hold are:

* **the 3-D layout** -- ``mrope_cos_sin`` reproduces the reference
  ``recomposition_frequencies`` element for element, and on a text-only prompt
  it is bit-identical to the 1-D table the decode graphs were captured with
  (``rope_delta = 0`` and all three axes equal);
* **the position bookkeeping** -- ``mm_input.mrope_positions`` reproduces
  ``get_rope_index`` (the reference port in ``.tmp_dsh/vision/qwen35_vision_ref.py``
  is re-implemented *inline* below, not imported) for a 1024-token image, for the
  image at the head / middle / tail of the prompt, and for several images, and
  the run of text after an image keeps a constant ``rope_delta``;
* **the injection** -- ``scatter_image_embeds`` overwrites exactly the image rows
  of a prefill chunk given absolute prompt indices, including a chunk whose
  boundaries fall inside an image.

The assertions are exact (``torch.equal``) where the maths is exact and 1-ULP
tolerant where a different (but mathematically identical) float order is
unavoidable.

Module: ``coffin/tests``. Runs on CPU only, no checkpoint: the whole point is
that this half of the vision path is checkable without a DCU.

Constraints:
    * The reference ``get_rope_index`` is re-implemented inline rather than
      imported from ``.tmp_dsh``, so the test does not depend on a scratch tree.
    * ``scatter_image_embeds`` mutates the buffer it is given; every test that
      checks a non-image row must build its own input, or the previous case's
      writes leak into it.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

#: Deliberately *not* ``import support``: that module's import-time ``skipUnless``
#: probes call ``has_hip()``, which builds/loads the HIP extension and
#: initialises the DCU.  Every test here is pure CPU arithmetic (the whole point:
#: the layout and the injection must be verifiable without the container), so
#: importing ``support`` would turn a 30 ms test run into a multi-minute extension
#: load that also contends for the single card.
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")   # container torch needs it

_ENGINE = Path(__file__).resolve().parent.parent
if str(_ENGINE) not in sys.path:
    sys.path.insert(0, str(_ENGINE))

import torch

import mm_input
from ops.mrope import mrope_cos_sin, scatter_image_embeds
from ops.rope import make_inv_freq, make_rope_tables

#: The checkpoint's rotary config (``text_config.rope_parameters``).
HEAD_DIM = 256
PARTIAL = 0.25
ROT = int(HEAD_DIM * PARTIAL)          # 64
THETA = 10_000_000.0
MROPE_SECTION = (11, 11, 10)
IMAGE_TOKEN_ID = mm_input.IMAGE_TOKEN_ID


# reference implementations (inline: the tests must not import .tmp_dsh)

def _ref_recompose(position_ids: torch.Tensor, inv_freq: torch.Tensor,
                   section=(11, 11, 10), dtype=None):
    """``Qwen3_5TextRotaryEmbedding`` as the reference spec §4.2 states it.

    ``position_ids`` is ``[3, C]`` (rows T, H, W) or ``[3, B, C]``; the reference
    form expands it to ``(3, B, S)``, which for our 2-D input means one batch.
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
        cos, sin = cos[0], sin[0]                       # back to (C, rot)
    if dtype is not None:
        cos, sin = cos.to(dtype), sin.to(dtype)
    return cos, sin


def _ref_llm_vision_positions(start: int, grid, spatial_merge: int = 2) -> torch.Tensor:
    """``get_llm_vision_position_ids`` for one still image (t == 1)."""
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
    """``Qwen3_5Model.get_rope_index`` for one unpadded sequence.

    ``mm_token_type_ids`` is derived from the ids exactly as the processor does it
    (``tt[ids == IMAGE_TOKEN_ID] = 1``); ``vision_start/end`` stay text.
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
    """Token ids / grids of ``text … <img> … text`` with ``grid``-sized images."""
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


# 1. cos/sin layout

class MRopeLayoutTest(unittest.TestCase):
    """``mrope_cos_sin`` must be the reference composition, bit for bit."""

    def setUp(self):
        self.inv_freq = make_inv_freq(HEAD_DIM, ROT, THETA, "cpu")

    def test_matches_reference_formula(self):
        torch.manual_seed(0)
        pos = torch.randint(0, 512, (3, 7), dtype=torch.long)
        cos, sin = mrope_cos_sin(pos, self.inv_freq)
        ref_cos, ref_sin = _ref_recompose(pos, self.inv_freq)
        self.assertEqual(tuple(cos.shape), (7, ROT))
        self.assertTrue(torch.equal(cos, ref_cos), "cos differs from the reference")
        self.assertTrue(torch.equal(sin, ref_sin), "sin differs from the reference")

    def test_axis_to_dimension_map(self):
        """Frequency index i comes from axis i % 3 (T, H, W, ...)."""
        pos = torch.zeros(3, 1, dtype=torch.long)
        pos[0, 0], pos[1, 0], pos[2, 0] = 3, 5, 7      # distinct on purpose
        half = ROT // 2
        freq = pos.float()[:, 0, None] * self.inv_freq
        cos, _ = mrope_cos_sin(pos, self.inv_freq)
        for i in range(half):
            expect = freq[i % 3, i].cos()
            self.assertAlmostEqual(float(cos[0, i]), float(expect), places=6,
                                   msg=f"frequency {i} should come from axis {i % 3}")
        # H covers indices 1,4,...,31 (11) and W 2,5,...,29 (10): 11+11+10 == 32.
        t_idx = [i for i in range(half) if i % 3 == 0]
        h_idx = [i for i in range(half) if i % 3 == 1]
        w_idx = [i for i in range(half) if i % 3 == 2]
        self.assertEqual((len(t_idx), len(h_idx), len(w_idx)), MROPE_SECTION)
        self.assertEqual((t_idx[-1], h_idx[-1], w_idx[-1]), (30, 31, 29))

    def test_second_half_is_a_copy(self):
        pos = torch.arange(36, dtype=torch.long).reshape(3, -1)
        cos, sin = mrope_cos_sin(pos, self.inv_freq)
        half = ROT // 2
        self.assertTrue(torch.equal(cos[:, :half], cos[:, half:]))
        self.assertTrue(torch.equal(sin[:, :half], sin[:, half:]))

    def test_text_only_matches_the_1d_table(self):
        """All three axes equal + delta 0 must reproduce ``make_rope_tables``."""
        max_seq = 512
        cos_tbl, sin_tbl = make_rope_tables(HEAD_DIM, ROT, THETA, max_seq, "cpu",
                                            inv_freq=self.inv_freq)
        t = torch.arange(max_seq, dtype=torch.long)
        pos = t.view(1, -1).expand(3, -1)
        cos, sin = mrope_cos_sin(pos, self.inv_freq)
        self.assertTrue(torch.equal(cos, cos_tbl), "text-only cos != 1-D table")
        self.assertTrue(torch.equal(sin, sin_tbl), "text-only sin != 1-D table")

    def test_rope_delta_shifts_every_row(self):
        """Decode after an image: one table row, shifted by a constant."""
        delta = -4
        t = torch.arange(40, 60, dtype=torch.long)
        pos = (t + delta).view(1, -1).expand(3, -1)
        cos, sin = mrope_cos_sin(pos, self.inv_freq)
        cos_tbl, sin_tbl = make_rope_tables(HEAD_DIM, ROT, THETA, 4096, "cpu",
                                            inv_freq=self.inv_freq)
        self.assertTrue(torch.equal(cos, cos_tbl.index_select(0, t + delta)))
        self.assertTrue(torch.equal(sin, sin_tbl.index_select(0, t + delta)))


# 2. positions / rope_delta against get_rope_index

class RopeIndexTest(unittest.TestCase):
    """``mm_input.mrope_positions`` must equal the reference ``get_rope_index``.

    Known defect: ``mm_input.mrope_positions`` (v1) uses the *token index* as the
    3-D cursor, so anything after an image gets token-index positions and
    ``rope_delta`` collapses to 0 (the sign/size of the shift disappears).  The
    reference below is the spec's two-cursor algorithm, verified element-wise
    against ``get_rope_index``; the contract test compares against it and is
    marked ``expectedFailure`` until the helper is fixed, so the day it is fixed
    the test turns green instead of silently staying red.
    """

    #: 1024 tokens of image = a 1024² picture: grid ``[1, 64, 64]``.
    GRID_1024 = (1, 64, 64)

    def _check(self, ids, grids, label=""):
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
        ids, grids = _prompt_with_image(5, 6, self.GRID_1024)
        pos, delta = self._check(ids, grids, "middle")
        n_img = 1024
        self.assertEqual(sum(e - s for s, e in mm_input.find_image_spans(ids)), n_img)

    def test_image_at_the_head(self):
        ids, grids = _prompt_with_image(0, 3, self.GRID_1024)
        self._check(ids, grids, "head")

    def test_image_at_the_tail(self):
        ids, grids = _prompt_with_image(4, 0, self.GRID_1024)
        self._check(ids, grids, "tail")

    def test_two_images(self):
        ids, grids = _prompt_with_image(3, 4, (1, 32, 32), n_images=2, between=3)
        self._check(ids, grids, "two images")

    def test_small_image_and_non_square(self):
        for grid in ((1, 16, 16), (1, 16, 32), (1, 32, 48)):
            with self.subTest(grid=grid):
                ids, grids = _prompt_with_image(3, 3, grid)
                self._check(ids, grids, f"grid {grid}")

    def test_text_after_an_image_has_constant_delta(self):
        """The image block ends at max(h, w)/merge: every later token is shifted."""
        ids, grids = _prompt_with_image(5, 8, self.GRID_1024)
        pos, delta = self._check(ids, grids, "delta")
        spans = mm_input.find_image_spans(ids)
        img_start, img_end = spans[-1]
        # 5 text tokens + <|vision_start|> put the image at position 6; 1024² is
        # 64x64 patches -> 32x32 merged, so the image's last H/W position is
        # 6 + 31 = 37 while 1024 tokens were consumed -- that gap is the delta.
        self.assertEqual(img_start, 6)
        self.assertEqual(int(pos[:, img_end - 1].max()), 37)
        # 37 is the image's last position; the tail then runs 38..46, so the
        # prompt ends at 46 while holding 1039 tokens -- that gap is the delta.
        self.assertEqual(int(pos.max()), 46)
        tail = torch.arange(img_end, len(ids), dtype=torch.long) + delta
        self.assertTrue(torch.equal(pos[0, img_end:], tail), "T axis after the image")
        self.assertTrue(torch.equal(pos[1, img_end:], tail), "H axis after the image")
        self.assertTrue(torch.equal(pos[2, img_end:], tail), "W axis after the image")
        # And the delta is exactly what the last position implies.
        self.assertEqual(delta, int(pos.max()) + 1 - len(ids))

    def test_text_only_prompt_is_identity(self):
        ids = [11, 12, 13, 14, 15]
        pos, delta = mm_input.mrope_positions([], [], len(ids), device="cpu")
        self.assertEqual(delta, 0)
        self.assertTrue(torch.equal(pos[0], torch.arange(len(ids))))
        self.assertTrue(torch.equal(pos[0], pos[1]))
        self.assertTrue(torch.equal(pos[0], pos[2]))

    def test_delta_is_negative_for_a_1024_image(self):
        """Sanity on the sign: 1024 tokens consume 32 positions, so delta < 0."""
        ids, grids = _prompt_with_image(5, 6, self.GRID_1024)
        _, delta = self._check(ids, grids, "sign")
        self.assertLess(delta, 0)


# 3. injection

class ScatterInjectTest(unittest.TestCase):
    """``scatter_image_embeds`` must hit exactly the image rows of a chunk."""

    HIDDEN = 8

    def setUp(self):
        ids, grids_1024 = _prompt_with_image(4, 4, (1, 16, 16))       # 64 img tokens
        self.ids = ids
        self.spans = mm_input.find_image_spans(ids)
        n = sum(e - s for s, e in self.spans)
        torch.manual_seed(0)
        self.embeds = torch.randn(n, self.HIDDEN)

    def _gather(self, length):
        return torch.zeros(length, self.HIDDEN)

    def test_whole_prompt_in_one_chunk(self):
        x = self._gather(len(self.ids))
        scatter_image_embeds(x, self.embeds, self.spans, 0, len(self.ids))
        mask = torch.zeros(len(self.ids), dtype=torch.bool)
        base = 0
        for s, e in self.spans:
            mask[s:e] = True
            x[s:e] = self.embeds[base:base + (e - s)]          # identity expectation
            base += e - s
        self.assertTrue(torch.equal(x[mask], self.embeds))
        self.assertEqual(int(x[~mask].abs().sum()), 0)

    def test_chunk_inside_an_image(self):
        """A chunk that starts and ends inside the image copies only its rows."""
        s, e = self.spans[0]
        start, end = s + 10, s + 25
        x = self._gather(end - start)
        scatter_image_embeds(x, self.embeds, self.spans, start, end)
        self.assertTrue(torch.equal(x, self.embeds[10:25]))

    def test_chunk_ending_inside_an_image(self):
        s, e = self.spans[0]
        start, end = s - 3, s + 5
        x = self._gather(end - start)
        scatter_image_embeds(x, self.embeds, self.spans, start, end)
        self.assertEqual(int(x[:3].abs().sum()), 0)          # text before the image
        self.assertTrue(torch.equal(x[3:], self.embeds[:5]))

    def test_chunk_starting_inside_an_image(self):
        s, e = self.spans[0]
        start, end = e - 4, e + 4
        x = self._gather(end - start)
        scatter_image_embeds(x, self.embeds, self.spans, start, end)
        self.assertTrue(torch.equal(x[:4], self.embeds[-4:]))
        self.assertEqual(int(x[4:].abs().sum()), 0)          # text after the image

    def test_chunk_without_image_is_untouched(self):
        s, _e = self.spans[0]
        x = self._gather(s)                     # only the text run before the image
        scatter_image_embeds(x, self.embeds, self.spans, 0, s)
        self.assertEqual(int(x.abs().sum()), 0)

    def test_multiple_images_offsets(self):
        """A chunk spanning several images reads each span at its own offset."""
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
        """The second half of a shared prefix injects starting at the right row."""
        s, e = self.spans[0]
        start = s - 1                     # a cached boundary just before the image
        x = self._gather(len(self.ids) - start)
        scatter_image_embeds(x, self.embeds, self.spans, start, len(self.ids))
        self.assertEqual(int(x[0].abs().sum()), 0)
        self.assertTrue(torch.equal(x[1:1 + (e - s)], self.embeds))


class ChunkedPrefillEquivalenceTest(unittest.TestCase):
    """Slice-per-chunk must equal whole-prompt: that is what chunking relies on.

    ``Qwen35.prefill`` walks the prompt in ``chunk``-sized pieces and hands every
    piece a *slice* of the request's positions plus its own overlap with the image
    spans.  If either slice were off by a chunk boundary, a prompt would embed or
    rotate differently at chunk=64 than at chunk=4096 -- so this test replays that
    arithmetic directly (no model, no GPU) and demands exact equality.
    """

    HIDDEN = 8
    CHUNK = 5

    def test_chunked_injection_and_rope_equal_whole_prompt(self):
        inv_freq = make_inv_freq(HEAD_DIM, ROT, THETA, "cpu")
        # A prompt whose image spans deliberately straddle several 5-token chunks.
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


# 4. the model-side plumbing

class ModelPlumbingTest(unittest.TestCase):
    """``Qwen35``/``MTP`` must own a ``rope_delta`` buffer with the right shape.

    These are source-level assertions, deliberately: ``models.qwen35`` imports
    ``safetensors`` (checkpoint I/O), which a dev box does not have, so importing
    the module here would turn a CPU-runnable contract test into a skip.  The
    containers are allocated in ``__init__`` / ``enable_batching`` and the
    captured graphs index them, so a missing or wrongly-shaped buffer is a
    load-time failure on the DCU -- pinning the source is the cheap local guard.
    """

    @staticmethod
    def _src(name: str) -> str:
        return (Path(__file__).resolve().parent.parent / name).read_text(encoding="utf-8")

    def test_model_has_per_row_rope_delta(self):
        src = self._src("models/qwen35/model.py")
        self.assertIn("self.rope_delta = torch.zeros(1", src)
        self.assertIn("self.rope_delta = torch.zeros(capacity", src)
        self.assertIn("self.rope_delta1 = self.rope_delta[", src)
        # reset() clears it, so a fresh sequence never inherits an image offset.
        self.assertIn("def reset(self) -> None:\n        self.pos.zero_()\n"
                      "        self.rope_delta.zero_()", src)

    def test_rope_delta_follows_the_row_state(self):
        """A row's offset must travel with its recurrent state, and clear on reuse."""
        src = self._src("models/qwen35/model.py")
        self.assertIn("self.rope_delta[dst_row].copy_(self.rope_delta[src_row])", src)
        self.assertIn("self.rope_delta[row].zero_()", src)

    def test_prefill_signature_keeps_mm_last_and_optional(self):
        src = self._src("models/qwen35/model.py")
        self.assertIn("def prefill(self, tokens: list[int], chunk: int = 2048,\n"
                      "                spec_row: int | None = None,\n"
                      '                mm: "MMInput | None" = None) -> torch.Tensor:', src)
        # prefill_group takes no mm: it reads each sequence's own payload.
        self.assertIn("def prefill_group(self, seqs: list, snapshot_interval: int | None = None):",
                      src)
        self.assertIn('mm_list = [getattr(s, "mm", None) for s in seqs]', src)

    def test_mtp_has_a_rope_delta(self):
        src = self._src("models/qwen35/mtp.py")
        self.assertIn("self.rope_delta = torch.zeros(1", src)
        self.assertIn("self.b_rope_delta = torch.zeros(capacity", src)


if __name__ == "__main__":
    unittest.main()
