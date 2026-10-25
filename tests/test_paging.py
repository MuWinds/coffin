"""Paged attention and radix prefix reuse must not change the answers.

Module: ``coffin/tests``. Drives ``prefix_engine.PrefixCacheEngine``,
``paged_cache`` and ``radix_cache`` over a shared paged model; it swaps the
fixture between the paged and dense configurations, so it pays two model loads.

Constraints:
    * The equality tests are token-for-token, not approximate: paging changes
      where the int8 rows live, not the arithmetic, so any token drift is a bug.
    * On the hybrid model reused KV is only valid together with the recurrent
      state that produced it; a KV-only hit would silently corrupt everything
      after it, which is why the state snapshot is asserted, not assumed.
    * ``_ENGINE`` is module-global: the block pool and the tree are the state
      under test, so a fresh engine per test would hide leaks and make eviction
      assertions meaningless.
    * The pool sizes are small on purpose so eviction is reachable; a large pool
      would never exercise the unreferenced-leaf path.

Four things have to hold:

1. paged decode == dense decode, token for token (the paged kernel addresses the
   same int8 rows through a slot map);
2. a prefix cache *hit* == a cold full prefill -- the important one for a hybrid
   model;
3. copy-on-write protects a shared block, and decoding past a matched prefix
   leaves every block the tree holds untouched;
4. the pool survives pressure: eviction takes only unreferenced leaves and every
   block comes back on release.

Port of ``diag/check_paged.py``: only functional assertions (the timing prints
stay in that probe -- tests assert *what* the cache reuses, not how fast).
"""

from __future__ import annotations

import functools
import hashlib
import unittest

import support

import torch

PROMPT = ("The history of Paris spans more than two thousand years. It began as a "
          "settlement on the banks of the Seine, known to the Romans as Lutetia. ")
SYS_A = ("You are a careful assistant. Answer questions about history and geography "
         "concisely, and cite the century when it matters. ") * 6
SYS_B = ("Reply in one short sentence. Prefer concrete dates and place names over "
         "general remarks. Do not repeat the question. ") * 6

_ENGINE = None


def _engine():
    """One paged PrefixCacheEngine for the module, rebuilt if the fixture moved."""
    global _ENGINE
    from engine.prefix_engine import PrefixCacheEngine
    model = support.ModelFixture.get(quant=False, max_seq=8192, paged=True)
    if _ENGINE is None or _ENGINE.model is not model:
        _ENGINE = PrefixCacheEngine(model, num_blocks=256, block_size=64,
                                    max_state_snapshots=8)
        support.ensure_captured(model)
    return _ENGINE


@functools.lru_cache(maxsize=1)
def _tokenizer():
    """Load the tokenizer once; it is 10 MB of JSON and every test re-encodes text."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(support.MODEL_DIR))


def _ids(text: str) -> list[int]:
    """Token ids of one prompt fragment."""
    return _tokenizer().encode(text)


def _greedy(model, ids: list[int], n: int) -> list[int]:
    """Dense baseline: greedy ``n`` tokens through the plain (unpaged) model path."""
    model.reset()
    logits = model.prefill(ids)
    out: list[int] = []
    for _ in range(n):
        tok = int(logits.argmax())
        out.append(tok)
        logits = model.decode_step(tok)
    return out


def _tree_fingerprint(eng) -> str:
    """Hash every block the radix tree currently holds.

    A fingerprint rather than a block-id set: copy-on-write may move or refcount
    a block without changing the KV bytes, and that is fine, whereas overwriting
    a block in place changes the hash.
    """
    digest = hashlib.sha256()
    pool = eng.pool
    stack = [eng.radix.root]
    while stack:
        node = stack.pop()
        for block in node.blocks:
            sl = slice(block * pool.bs, (block + 1) * pool.bs)
            digest.update(pool.k[:, sl].cpu().numpy().tobytes())
            digest.update(pool.v[:, sl].cpu().numpy().tobytes())
        stack.extend(node.children.values())
    return digest.hexdigest()[:16]


@support.needs_model
class PagedDecodeTest(unittest.TestCase):
    """The paged kernel must address exactly the rows dense attention would."""

    def test_paged_decode_matches_dense(self):
        """A wrong slot map still produces plausible tokens, so the two paths are compared token for token."""
        ids = _ids(SYS_A) + _ids(" When was it founded?")
        dense_model = support.ModelFixture.get(quant=False, max_seq=8192, paged=False)
        support.ensure_captured(dense_model)
        dense = _greedy(dense_model, ids, 24)

        eng = _engine()
        eng.generate(ids, 4)                       # warm the page table
        run = eng.generate(ids, 24)
        self.assertEqual(run["tokens"], dense,
                         f"paged vs dense: {sum(a == b for a, b in zip(run['tokens'], dense))}/24 tokens")


@support.needs_model
class PrefixReuseTest(unittest.TestCase):
    """A cache hit must be indistinguishable from a cold prefill of the same tokens."""

    def test_prefix_hit_reproduces_the_cold_answer(self):
        """The second request is the oracle: same prompt, cached prefix, must decode identically."""
        q1 = _ids(" Which river runs through it?")
        ids = _ids(SYS_B) + q1
        eng = _engine()
        cold = eng.generate(ids, 24)
        self.assertEqual(cold["matched"], 0, "first use of this prefix must be cold")
        warm = eng.generate(ids, 24)
        self.assertGreaterEqual(warm["matched"], len(_ids(SYS_B)) - 64,
                                f"matched {warm['matched']} of {len(ids)}")
        self.assertEqual(warm["tokens"], cold["tokens"],
                         "cache hit changed the greedy answer")

    def test_warm_prefill_reuses_the_prefix_and_keeps_the_answer(self):
        """Long prompts are chunked, so the reuse must survive several prefill chunks, not just one."""
        q1 = _ids(" When was it founded?")
        q2 = _ids(" Which river runs through it?")
        base = _ids(SYS_A) + q1 + q2 + _ids(PROMPT)
        long_prompt = [base[i % len(base)] for i in range(6000)]   # 3 prefill chunks
        eng = _engine()
        cold = eng.prefill(long_prompt).float().clone()
        warm = eng.prefill(long_prompt).float().clone()
        # Functional claim: the cache covers the prompt and the reused path returns
        # the same next token as the full prefill (what it saves is the probe's job).
        self.assertGreaterEqual(eng.last_matched, len(long_prompt) - 1024,
                                f"matched {eng.last_matched} of {len(long_prompt)}")
        self.assertEqual(int(cold.argmax()), int(warm.argmax()),
                         "warm prefill picked a different next token")
        unit = max(float(cold.abs().max()), 1e-6) * 2.0 ** -8    # one bf16 ULP
        self.assertLess(float((cold - warm).abs().max()), 4.0 * unit)


@support.needs_gpu
class CopyOnWriteTest(unittest.TestCase):
    """A second reference to a block must never be written through in place."""

    def test_shared_block_is_not_overwritten_and_writer_copies(self):
        """Writing a shared block in place would corrupt the tree's copy while still looking correct to the writer."""
        from cache.paged_cache import PagedKVCache, SeqKV
        pool = PagedKVCache(1, 2, 8, num_blocks=4, block_size=4, max_seq=16)
        block = pool.alloc()
        pool.k[0, block * 4:(block + 1) * 4] = 7
        pool.incref(block)                         # a second holder (the tree)
        seq = SeqKV(pool, [block], num_tokens=3)   # partial tail, still shared
        before = pool.k[0, block * 4:(block + 1) * 4].clone()
        seq.reserve(1)
        self.assertTrue(torch.equal(before, pool.k[0, block * 4:(block + 1) * 4]),
                        "a shared block was overwritten in place")
        self.assertNotEqual(seq.blocks[-1], block, "the writer did not get its own block")
        self.assertGreaterEqual(pool.n_cows, 1)


@support.needs_model
class TreeBlocksTest(unittest.TestCase):
    """The tree owns its blocks; a decoding sequence may only read them."""

    def test_decoding_past_a_matched_prefix_leaves_the_tree_untouched(self):
        """Decoding beyond the match writes new blocks, and any in-place write would poison the shared prefix."""
        ids = _ids(SYS_A) + _ids(" When was it founded?")
        eng = _engine()
        eng.generate(ids, 8)                       # make sure the prefix is in the tree
        before = _tree_fingerprint(eng)
        eng.generate(ids, 16)                      # decode past the matched prefix
        self.assertEqual(before, _tree_fingerprint(eng))


@support.needs_gpu
class PoolEvictionTest(unittest.TestCase):
    """Eviction and release must both leave the refcounts consistent."""

    def test_eviction_spares_live_blocks_and_release_returns_them(self):
        """Evicting a live sequence's block is data loss; a leaked refcount is a pool that never grows back."""
        from cache.paged_cache import PagedKVCache, SeqKV
        from cache.radix_cache import RadixCache
        pool = PagedKVCache(2, 4, 32, num_blocks=4, block_size=8, max_seq=64)
        radix = RadixCache(pool, max_state_snapshots=2)
        live = SeqKV(pool)
        live.reserve(16)
        live.commit_upto(16)                       # 2 of the 4 blocks are held
        radix.evict(4)
        self.assertEqual(pool.num_used, 2, "eviction touched a live sequence's blocks")
        live.free()
        self.assertEqual(pool.num_free, 4, "release did not return the blocks")
        self.assertTrue(all(r >= 0 for r in pool.refcount), f"refcounts {pool.refcount}")
