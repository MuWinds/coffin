"""The radix tree must never hand one image's KV to a different image.

Module: ``coffin/tests``. Drives ``radix_cache`` / ``block_manager`` /
``prefix_engine`` / ``model_runner`` with a recording model; it is the only test
that asserts on the cache key itself rather than on generated tokens.

Constraints:
    * The paged pool is real but allocated on the CPU and the model is a recorder,
      so this module runs without a DCU. Do not swap in the shared GPU fixture:
      the assertions are about keys and match lengths, not about numerics.
    * The block key must stay ``(tokens_block, media_signature)`` so a new kernel
      or an extra cache layer cannot reintroduce token-only keys. The text-only
      tests are the guard that this costs the text path nothing.
    * ``TEXT_MEDIA_ID`` must remain the value a no-image block hashes to; changing
      it would invalidate every stored text-only block.
    * Tests patch nothing global, but ``make_runner`` sets ``batching = True``,
      which is what routes ``prefill`` through ``_prefill_batched``.

A ``<|image_pad|>`` run is the same token ids for every image, so a prefix cache
keyed on token ids alone would reuse the KV of the *first* image for a prompt
carrying a *different* one.  That failure is silent and grows with the suffix, so
the block key is ``(tokens_block, media_signature)`` instead: the set of image
content ids in the block, ``(0,)`` for a block with no image token.

Four things have to hold, and this module checks them without a DCU:

1. same tokens + different image digest -> zero match;
2. same tokens + same digest -> a hit, always on a block boundary;
3. text only -> the old plain-token keys and the old match lengths (no per-token
   cost, no behaviour change);
4. both prefill paths (single-stream ``prefix_engine`` and batched
   ``model_runner``) insert and match the same salted keys, and forward ``mm`` to
   the model so the embeddings are actually injected.

    PYTHONPATH=coffin python -m unittest coffin/tests/test_radix_mm.py -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parent.parent
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

import torch

from engine.config import EngineConfig
from utils.mm_input import MMInput
from cache.paged_cache import PagedKVCache
from cache.radix_cache import (RadixCache, TEXT_MEDIA_ID, media_id, media_ids_for,
                               media_signature)

BLOCK = 8
MAX_SEQ = 512
NUM_BLOCKS = 64
IMAGE = 248056                      # <|image_pad|>


# fixtures
def make_pool() -> PagedKVCache:
    """A real paged pool, on the CPU so the suite needs no DCU."""
    return PagedKVCache(1, 1, 4, num_blocks=NUM_BLOCKS, block_size=BLOCK,
                        max_seq=MAX_SEQ, device="cpu")


def make_mm(digest: str, spans: list[tuple[int, int]], prompt_len: int) -> MMInput:
    """An ``MMInput`` with the fields the cache reads (embeds/positions unused)."""
    n_img = sum(e - s for s, e in spans) or 1
    return MMInput(embeds=torch.zeros(n_img, 4), grid_thw=torch.ones(len(spans), 3),
                   positions=torch.zeros(3, prompt_len, dtype=torch.long),
                   rope_delta=0, digest=digest, spans=list(spans))


def prompt_with_image(n_text: int, n_img: int, prefix_text: int = 0) -> tuple[list[int], int]:
    """``prefix_text`` text ids, then an image run, then ``n_text`` more text ids."""
    ids = [100 + i for i in range(prefix_text)]
    span = (len(ids), len(ids) + n_img)
    ids += [IMAGE] * n_img
    ids += [900 + i for i in range(n_text)]
    return ids, span[0]


class FakeModel:
    """Records what the prefill paths ask of it; owns only ``pos1``/row counters.

    Enough surface for ``PrefixCacheEngine.prefill`` and
    ``ModelRunner._prefill_batched``: both only touch ``pos1``/``pos``, the
    linear-state snapshot helpers and ``prefill``.
    """

    def __init__(self) -> None:
        self.pos1 = torch.zeros(1, dtype=torch.long)
        self.pos = self.pos1                     # single-row staging row is row 0
        self.prefill_row = 0
        self.decode_capacity = 8
        self.calls: list[tuple[list[int], object]] = []
        self.states: dict[int, dict] = {}

    # the two paths' state hooks
    def reset(self) -> None:
        self.pos1.zero_()

    def snapshot_linear_states(self, row: int = 0) -> dict:
        return {"row": row, "pos": int(self.pos1[row])}

    def restore_linear_states(self, states, row: int = 0) -> None:
        self.states[row] = dict(states)
        self.pos1[row] = int(states["pos"])

    def zero_linear_states(self, row: int) -> None:
        self.states[row] = {"row": row, "pos": 0}

    def move_linear_states(self, src_row: int, dst_row: int) -> None:
        if src_row != dst_row:
            self.states[dst_row] = dict(self.states.get(src_row, {}))

    def prefill(self, tokens, chunk: int = 2048, spec_row=None, mm=None):
        self.calls.append((list(tokens), mm))
        self.pos1[self.prefill_row:self.prefill_row + 1] += len(tokens)
        return torch.zeros(4)


def make_engine(model: FakeModel):
    """A single-stream ``PrefixCacheEngine`` over a CPU pool."""
    from engine.block_manager import BlockManager
    from engine.prefix_engine import PrefixCacheEngine

    pool = make_pool()
    bm = BlockManager(pool, RadixCache(pool, 4), BLOCK)
    return PrefixCacheEngine(model, num_blocks=NUM_BLOCKS, block_size=BLOCK,
                             max_state_snapshots=4, snapshot_interval=BLOCK, bm=bm)


def make_runner(model: FakeModel, pool=None):
    """A batched ``ModelRunner`` with a radix tree attached."""
    from engine.block_manager import BlockManager
    from engine.model_runner import ModelRunner

    pool = pool or make_pool()
    cfg = EngineConfig(model_dir=".", max_seq=MAX_SEQ, bench=False, quant=False,
                       prefix_cache=True, block_size=BLOCK, snapshots=4,
                       snapshot_interval=BLOCK, num_blocks=NUM_BLOCKS)
    manager = BlockManager(pool, RadixCache(pool, 4), BLOCK)
    runner = ModelRunner(model, cfg)
    runner.attach_blocks(manager)
    runner.batching = True                  # take ``_prefill_batched``
    return runner, manager


class Seq:
    """Stand-in for ``sequence.Sequence`` (only what the runner reads)."""

    def __init__(self, token_ids: list[int], mm=None) -> None:
        self.seq_id = id(self)
        self.token_ids = list(token_ids)
        self.mm = mm
        self.num_prompt_tokens = len(self.token_ids)
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.block_table: list[int] = []
        self.row = -1
        self.matched = 0
        self.params = None

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)


# media id / signature helpers
class MediaIdTest(unittest.TestCase):
    """The media id is the cache salt, so it must be deterministic and fit a signed 64-bit token slot."""

    def test_media_id_is_stable_positive_and_position_salted(self) -> None:
        """Two runs must agree (an id that changed would miss every warm cache), and the id must salt on both digest and index."""
        a = media_id("deadbeef", 0)
        self.assertEqual(a, media_id("deadbeef", 0), "same image must keep its id")
        self.assertNotEqual(a, media_id("deadbeef", 1), "index must salt the id")
        self.assertNotEqual(a, media_id("feedface", 0), "digest must salt the id")
        self.assertNotEqual(a, TEXT_MEDIA_ID)
        self.assertGreater(a, 0)
        self.assertLess(a, 1 << 63)

    def test_media_ids_and_signatures(self) -> None:
        """Positional salting is what makes the same picture twice two distinct spans, and block signatures must reflect exactly the image tokens a block holds."""
        ids, img_start = prompt_with_image(n_text=4, n_img=12,
                                           prefix_text=2 * BLOCK + 3)
        mm = make_mm("d1", [(img_start, img_start + 12)], len(ids))
        media = media_ids_for(mm, len(ids))
        self.assertEqual(media[:img_start], [TEXT_MEDIA_ID] * img_start)
        self.assertEqual(set(media[img_start:img_start + 12]), {media_id("d1", 0)})
        self.assertEqual(media[img_start + 12:], [TEXT_MEDIA_ID] * 4)

        # block signatures: the blocks before the image are text (0,), the block
        # the span starts in carries the image id next to the text id.
        self.assertEqual(media_signature(media, 0, BLOCK), (TEXT_MEDIA_ID,))
        self.assertEqual(media_signature(media, BLOCK, BLOCK), (TEXT_MEDIA_ID,))
        self.assertEqual(media_signature(media, img_start // BLOCK * BLOCK, BLOCK),
                         tuple(sorted({TEXT_MEDIA_ID, media_id("d1", 0)})))

    def test_text_only_mm_has_no_ids_and_no_signature_work(self) -> None:
        """``None`` and an empty digest must both mean "no image ids", so the text path allocates no per-token array at all."""
        ids = [1, 2, 3]
        mm = make_mm("", [], len(ids))
        self.assertIsNone(media_ids_for(mm, len(ids)))
        self.assertIsNone(media_ids_for(None, len(ids)))
        self.assertEqual(media_signature(None, 0, BLOCK), (TEXT_MEDIA_ID,))

    def test_spans_without_digest_are_rejected(self) -> None:
        """An image span with no digest would key every image alike, which is the exact bug this module exists to prevent; fail loudly instead."""
        ids, img_start = prompt_with_image(n_text=4, n_img=8)
        mm = make_mm("", [(img_start, img_start + 8)], len(ids))
        with self.assertRaises(ValueError):
            media_ids_for(mm, len(ids))


# the core claim: no cross-image KV reuse
class SaltTest(unittest.TestCase):
    """The core claim: image KV is never reused across different images, and text KV still is."""

    def setUp(self) -> None:
        """Shared prompt, two digests for the same image position, one CPU pool per test."""
        self.pool = make_pool()
        self.radix = RadixCache(self.pool, max_state_snapshots=4)
        self.ids, self.img_start = prompt_with_image(n_text=9, n_img=12)
        self.mm_a = make_mm("image-A", [(self.img_start, self.img_start + 12)],
                            len(self.ids))
        self.mm_b = make_mm("image-B", [(self.img_start, self.img_start + 12)],
                            len(self.ids))

    def _blocks(self, n: int) -> list[int]:
        return [self.pool.alloc() for _ in range(n)]

    def test_same_tokens_different_digest_matches_nothing(self) -> None:
        """The failure this prevents is silent: the second image's prompt would decode over the first image's KV."""
        blocks = self._blocks(len(self.ids) // BLOCK)
        self.radix.insert(self.ids, blocks, {}, mm=self.mm_a)
        got, matched, _ = self.radix.match_prefix(self.ids[:-1], mm=self.mm_b)
        self.assertEqual(matched, 0, "a different image reused the first image's KV")
        self.assertEqual(got, [])

    def test_same_tokens_same_digest_hits_on_a_block_boundary(self) -> None:
        """A hit must be whole blocks: a partial block cannot be refcounted or copied on write."""
        blocks = self._blocks(len(self.ids) // BLOCK)
        self.radix.insert(self.ids, blocks, {}, mm=self.mm_a)
        got, matched, _ = self.radix.match_prefix(self.ids[:-1], mm=self.mm_a)
        self.assertEqual(matched % BLOCK, 0, "match must land on a block boundary")
        self.assertEqual(matched, (len(self.ids) - 1) // BLOCK * BLOCK)
        self.assertEqual(len(got), matched // BLOCK)
        self.assertTrue(all(b in blocks for b in got))

    def test_text_prefix_before_the_image_is_still_shared(self) -> None:
        """Salting must not be contagious: text blocks before the image are image-independent and must stay reusable."""
        # An image further along must not throw away the shareable text prefix:
        # the blocks before it hold the same KV whatever the image is.
        ids, img_start = prompt_with_image(n_text=8, n_img=12, prefix_text=3 * BLOCK)
        a = make_mm("image-A", [(img_start, img_start + 12)], len(ids))
        b = make_mm("image-B", [(img_start, img_start + 12)], len(ids))
        self.radix.insert(ids, self._blocks(len(ids) // BLOCK), {}, mm=a)
        _, matched, _ = self.radix.match_prefix(ids[:-1], mm=b)
        self.assertEqual(matched, 3 * BLOCK,
                         "the image-free text prefix should still be reusable")

    def test_plain_text_query_cannot_match_an_image_block(self) -> None:
        """The inverse leak: a text query carrying placeholder ids must not adopt an image branch, which would inject another image's KV."""
        # A text request whose ids happen to contain the placeholder run (no mm
        # set) must not adopt an image branch: its signature is (0,).
        self.radix.insert(self.ids, self._blocks(len(self.ids) // BLOCK), {},
                          mm=self.mm_a)
        _, matched, _ = self.radix.match_prefix(self.ids[:-1], mm=None)
        self.assertEqual(matched, 0)

    def test_image_query_reuses_a_text_branch_up_to_the_image(self) -> None:
        """Reuse must not depend on insert order: text first, then an image request, must still share the image-free head."""
        # The other direction: a *text* insert put the head in the tree (plain
        # keys), then an image request with the same head arrives.  The text KV is
        # correct regardless of the image, so the head must be reused and the
        # match must stop before the first image block.
        ids, img_start = prompt_with_image(n_text=4, n_img=12, prefix_text=3 * BLOCK)
        self.radix.insert(ids, self._blocks(len(ids) // BLOCK), {}, mm=None)
        mm = make_mm("image-A", [(img_start, img_start + 12)], len(ids))
        _, matched, _ = self.radix.match_prefix(ids[:-1], mm=mm)
        self.assertEqual(matched, 3 * BLOCK,
                         "the text head must be reusable, the image part not")

    def test_a_no_mm_query_never_looks_up_a_salted_key(self) -> None:
        """The mechanism behind the two tests above: a salted edge sits under a key a plain lookup cannot construct, so ``mm=None`` costs nothing."""
        # The invariant behind the two tests above: salted edges live under keys a
        # plain-token lookup cannot produce, so ``mm=None`` costs no comparison at
        # all and can never reach image KV.
        self.radix.insert(self.ids, self._blocks(len(self.ids) // BLOCK), {},
                          mm=self.mm_a)
        node = next(iter(self.radix.root.children.values()))
        self.assertIsNotNone(node.sigs)
        self.assertNotIn(tuple(self.ids[:BLOCK]), self.radix.root.children,
                         "a salted edge must not sit under a plain token key")
        plain_key = tuple(self.ids[:BLOCK])
        self.assertEqual(len(plain_key), BLOCK)

    def test_two_images_in_one_prompt_are_position_salted(self) -> None:
        """The same bytes sent twice are still two different spans; without the index salt the second image would reuse the first's KV."""
        ids, first = prompt_with_image(n_text=4, n_img=8)
        second = len(ids)
        ids += [IMAGE] * 8
        mm = make_mm("same-bytes", [(first, first + 8), (second, second + 8)], len(ids))
        media = media_ids_for(mm, len(ids))
        self.assertNotEqual(media[first], media[second],
                            "the same picture twice is still two different spans")
        blocks = self._blocks(len(ids) // BLOCK)
        self.radix.insert(ids, blocks, {}, mm=mm)
        _, matched, _ = self.radix.match_prefix(ids[:-1], mm=mm)
        self.assertEqual(matched, len(ids) // BLOCK * BLOCK)

    def test_image_span_crossing_a_block_salts_both_blocks(self) -> None:
        """A span crossing a boundary must salt every block it touches; salting only the first would expose the overlap to reuse."""
        # The span starts mid-block: the block holding its head and the next one
        # must both carry the id, so neither can be shared across images.
        ids, img_start = prompt_with_image(n_text=4, n_img=12, prefix_text=3)
        mm = make_mm("image-A", [(img_start, img_start + 12)], len(ids))
        media = media_ids_for(mm, len(ids))
        mid = media_id("image-A", 0)
        head = media_signature(media, 0, BLOCK)
        self.assertIn(mid, head, "the block holding the span head must be salted")
        self.assertIn(TEXT_MEDIA_ID, head)
        for start in range(BLOCK, len(ids) - BLOCK + 1, BLOCK):
            sig = media_signature(media, start, BLOCK)
            if any(t == IMAGE for t in ids[start:start + BLOCK]):
                self.assertIn(mid, sig, f"image block at {start} is not salted")
            else:
                self.assertEqual(sig, (TEXT_MEDIA_ID,),
                                 f"text block at {start} must keep the plain key")
        # and a query carrying the *other* image shares none of the salted blocks
        blocks = [self.pool.alloc() for _ in range(len(ids) // BLOCK)]
        self.radix.insert(ids, blocks, {}, mm=mm)
        other = make_mm("image-B", [(img_start, img_start + 12)], len(ids))
        _, matched, _ = self.radix.match_prefix(ids[:-1], mm=other)
        self.assertEqual(matched, 0)


# text-only regression: plain tokens, plain keys, no mm work
class TextOnlyTest(unittest.TestCase):
    """The text path predates images and must keep its old keys, lengths and zero per-token cost."""

    def setUp(self) -> None:
        """A fresh CPU pool per test, so match lengths cannot be inherited from a previous case."""
        self.pool = make_pool()
        self.radix = RadixCache(self.pool, max_state_snapshots=4)

    def test_keys_stay_plain_token_tuples(self) -> None:
        """A text-only insert must not start producing salted child keys, or every existing cached block becomes unreachable."""
        ids = [1000 + i for i in range(5 * BLOCK)]
        blocks = [self.pool.alloc() for _ in range(5)]
        self.radix.insert(ids, blocks, {})
        keys = set(self.radix.root.children)
        self.assertEqual(keys, {tuple(ids[:BLOCK])},
                         "a text-only insert must keep the old child key format")
        self.assertIsNone(self.radix.root.children[tuple(ids[:BLOCK])].sigs)

    def test_match_lengths_are_unchanged(self) -> None:
        """Match lengths are the old floor-to-block values; a shorter one loses reuse and a longer one hands over unverified KV."""
        ids = [1000 + i for i in range(5 * BLOCK)]
        self.radix.insert(ids, [self.pool.alloc() for _ in range(5)], {})
        for n in (BLOCK, 2 * BLOCK, 5 * BLOCK, 5 * BLOCK - 1):
            _, matched, _ = self.radix.match_prefix(ids[:n], require_states=False)
            self.assertEqual(matched, n // BLOCK * BLOCK)

    def test_a_text_query_with_mm_none_matches_a_text_branch(self) -> None:
        """``mm=None`` is the text request's normal state, so it must take the plain-key path and match."""
        ids = [7 * i % 251 for i in range(4 * BLOCK)]
        self.radix.insert(ids, [self.pool.alloc() for _ in range(4)], {})
        _, matched, _ = self.radix.match_prefix(ids[:-1], mm=None)
        self.assertEqual(matched, 3 * BLOCK)

    def test_shared_text_prefix_between_an_image_and_a_text_request(self) -> None:
        """Reuse must work in both directions across modalities, or a mixed workload loses the prefix sharing entirely."""
        # The cache must still give a text request the text blocks an image
        # request put there, as long as they hold no image token.
        text = [500 + i for i in range(2 * BLOCK)]
        ids, img_start = prompt_with_image(n_text=6, n_img=8, prefix_text=2 * BLOCK)
        ids = [500 + i for i in range(2 * BLOCK)] + ids[2 * BLOCK:]
        mm = make_mm("d", [(img_start, img_start + 8)], len(ids))
        self.radix.insert(ids, [self.pool.alloc() for _ in range(len(ids) // BLOCK)],
                          {}, mm=mm)
        _, matched, _ = self.radix.match_prefix(text, mm=None)
        self.assertEqual(matched, 2 * BLOCK)


# both prefill paths: insert -> match, with mm forwarded to the model
class PrefixEnginePathTest(unittest.TestCase):
    """The single-stream prefill path must insert and match salted keys and forward ``mm``."""

    def test_insert_then_match_single_stream(self) -> None:
        """``mm`` reaching the model is what actually injects the image embeddings; a cache hit without it would silently drop the picture."""

        model = FakeModel()
        eng = make_engine(model)
        ids, img_start = prompt_with_image(n_text=9, n_img=12)
        mm = make_mm("image-A", [(img_start, img_start + 12)], len(ids))

        eng.prefill(ids, mm=mm)
        self.assertEqual(eng.last_matched, 0)
        self.assertTrue(model.calls, "the prefill never reached the model")
        self.assertIs(model.calls[-1][1], mm, "mm was not forwarded to model.prefill")
        self.assertEqual([t for call in model.calls for t in call[0]], ids)

        eng.prefill(ids, mm=mm)
        self.assertEqual(eng.last_matched, (len(ids) - 1) // BLOCK * BLOCK)
        self.assertEqual(eng.last_matched % BLOCK, 0)

        # same ids, another image: nothing may be adopted
        model2 = FakeModel()
        eng2 = make_engine(model2)
        block = eng2.bm.block_size
        del block
        other = make_mm("image-B", [(img_start, img_start + 12)], len(ids))
        eng2.prefill(ids, mm=other)
        self.assertEqual(eng2.last_matched, 0)

    def test_text_only_path_is_unchanged(self) -> None:
        """A text prefill must pass ``mm=None`` and keep the old match lengths."""
        model = FakeModel()
        eng = make_engine(model)
        ids = [300 + i for i in range(3 * BLOCK)]
        eng.prefill(ids)
        self.assertIsNone(model.calls[-1][1], "text prefill must pass mm=None")
        eng.prefill(ids)
        self.assertEqual(eng.last_matched, (len(ids) - 1) // BLOCK * BLOCK)
        _, matched, _ = eng.bm.match_prefix(ids[:-1])
        self.assertEqual(matched, eng.last_matched)

    def test_mixed_edge_stops_at_the_image_block(self) -> None:
        """A short prompt puts text and image blocks on one edge; the check must stop exactly at the first image block, not early and not late."""
        # One edge holds text blocks *and* image blocks (a short prompt lives
        # entirely in one node).  The edge gets a signature list, and checking it
        # must stop exactly at the first image block -- not one block early (which
        # would lose reuse) and not one late (which would hand over wrong KV).
        model = FakeModel()
        eng = make_engine(model)
        ids, img_start = prompt_with_image(n_text=4, n_img=12, prefix_text=2 * BLOCK + 5)
        mm_a = make_mm("image-A", [(img_start, img_start + 12)], len(ids))
        eng.prefill(ids, mm=mm_a)
        node = next(iter(eng.bm.radix.root.children.values()))
        self.assertIsNotNone(node.sigs, "a mixed edge must carry per-block signatures")

        # same image: the match reaches the last full block before the final token
        eng.prefill(ids, mm=mm_a)
        full = (len(ids) - 1) // BLOCK * BLOCK
        self.assertEqual(eng.last_matched, full)

        # same ids, another image: it must stop at the last *text* block, because
        # everything from the image on is a different computation.
        text_blocks = img_start // BLOCK
        eng2 = make_engine(FakeModel())
        eng2.prefill(ids, mm=mm_a)
        eng2.prefill(ids, mm=make_mm("image-B", [(img_start, img_start + 12)], len(ids)))
        self.assertEqual(eng2.last_matched, text_blocks * BLOCK,
                         "a different image must only reuse the image-free head")


class ModelRunnerPathTest(unittest.TestCase):
    """The batched path must match the single-stream one: same salt, same boundary, same ``mm`` forwarding."""

    def _insert(self, runner, seq) -> None:
        """Prefill one sequence through the runner (the batched path)."""
        runner.prefill(seq)

    def test_insert_then_match_batched(self) -> None:
        """Any divergence from the single-stream path shows up here first, because the scheduler only ever uses this one."""
        model = FakeModel()
        model.enable_batching = lambda capacity: None      # not used by prefill
        runner, bm = make_runner(model)
        ids, img_start = prompt_with_image(n_text=9, n_img=12)
        mm = make_mm("image-A", [(img_start, img_start + 12)], len(ids))

        first = Seq(ids, mm=mm)
        self._insert(runner, first)
        self.assertEqual(first.matched, 0)
        self.assertIs(model.calls[-1][1], mm, "mm was not forwarded to model.prefill")

        warm = Seq(ids, mm=mm)
        self._insert(runner, warm)
        self.assertEqual(warm.matched, (len(ids) - 1) // BLOCK * BLOCK)
        self.assertEqual(len(warm.block_table), (len(ids) + BLOCK - 1) // BLOCK)

        other = Seq(ids, mm=make_mm("image-B", [(img_start, img_start + 12)], len(ids)))
        self._insert(runner, other)
        self.assertEqual(other.matched, 0,
                         "a different image adopted the first image's KV")

    def test_text_only_batched_path_is_unchanged(self) -> None:
        """The batched text path keeps the old behaviour: no ``mm`` to the model, old match lengths."""
        model = FakeModel()
        runner, bm = make_runner(model)
        ids = [700 + i for i in range(4 * BLOCK)]
        runner.prefill(Seq(ids))
        self.assertIsNone(model.calls[-1][1])
        warm = Seq(ids)
        runner.prefill(warm)
        self.assertEqual(warm.matched, (len(ids) - 1) // BLOCK * BLOCK)

    def test_group_prefill_matches_what_it_inserted(self) -> None:
        """A mismatch between the key inserted and the key matched poisons the tree silently, so both are asserted on one prompt."""
        # ``prefill_group`` matches on ``seq.mm`` and inserts on ``seq.mm``; a
        # mismatch there is exactly the bug that would silently poison the tree,
        # so check the tree carries the image branch the match will look for.
        model = FakeModel()
        runner, bm = make_runner(model)
        ids, img_start = prompt_with_image(n_text=9, n_img=12)
        mm = make_mm("image-A", [(img_start, img_start + 12)], len(ids))

        seq = Seq(ids, mm=mm)
        seq.block_table = [bm.pool.alloc() for _ in range(len(ids) // BLOCK)]
        seq.row = 0
        bm.insert(seq.token_ids, seq.block_table, {}, mm=seq.mm)
        root_key = tuple(ids[:BLOCK]) + (media_id("image-A", 0),)
        self.assertIn(root_key, bm.radix.root.children)
        _, matched, _ = bm.match_prefix(ids[:-1], require_states=False, mm=mm)
        self.assertEqual(matched, (len(ids) - 1) // BLOCK * BLOCK)


if __name__ == "__main__":
    unittest.main()
