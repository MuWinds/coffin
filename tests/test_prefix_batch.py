"""Per-sequence prefix reuse in the batched engine.

The radix tree is per-sequence state (block hashes + the delta-net recurrent
snapshot that produced them), so batching must not switch it off: a request is
admitted, matches the longest cached block prefix, restores that boundary's
recurrent state, prefills only the suffix and inserts what it computed.

The reference for a warm run is the same engine's *cold* run: the test clears the
tree in between (``LLMEngine.clear_prefix_cache``), so no second model has to fit
on the card.

    python3 -m unittest discover -s tests -p "test_prefix_batch.py" -v
"""

from __future__ import annotations

import unittest

import support

from config import EngineConfig
from llm_engine import LLMEngine
from sequence import SamplingParams

BLOCK = 64
PREFIX = [1000 + i for i in range(192)]          # three full blocks
TAIL_A = [4000 + i for i in range(16)]
TAIL_B = [5000 + i for i in range(24)]


def make_engine(model, *, max_seq: int, max_num_seqs: int = 4) -> LLMEngine:
    cfg = EngineConfig(model_dir=str(support.MODEL_DIR), max_seq=max_seq,
                       bench=False, quant=True, prefix_cache=True,
                       block_size=BLOCK, snapshot_interval=BLOCK,
                       max_num_seqs=max_num_seqs, num_blocks=1024)
    engine = LLMEngine(cfg, model)
    engine.build()
    return engine


@support.needs_model
class BatchedPrefixTest(unittest.TestCase):
    MAX_SEQ = 1024

    @classmethod
    def setUpClass(cls) -> None:
        cls.MODEL = support.ModelFixture.get(quant=True, max_seq=cls.MAX_SEQ,
                                             paged=True, bench=False)
        cls.ENGINE = make_engine(cls.MODEL, max_seq=cls.MAX_SEQ)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.ENGINE = None
        cls.MODEL = None
        support.ModelFixture.close()

    def _generate(self, prompt: list[int], n: int = 4) -> dict:
        return self.ENGINE.generate(prompt, SamplingParams(max_tokens=n,
                                                           temperature=0.0))

    def test_cold_then_warm_identical_prompt(self) -> None:
        engine = self.ENGINE
        engine.clear_prefix_cache()
        cold = self._generate(PREFIX + TAIL_A)
        self.assertEqual(cold["matched"], 0)
        warm = self._generate(PREFIX + TAIL_A)
        self.assertGreaterEqual(warm["matched"], BLOCK)
        self.assertEqual(warm["tokens"], cold["tokens"])

    def test_shared_system_prompt_with_different_tails(self) -> None:
        engine = self.ENGINE
        engine.clear_prefix_cache()
        cold_a = self._generate(PREFIX + TAIL_A)
        warm_b = self._generate(PREFIX + TAIL_B)
        self.assertEqual(warm_b["matched"], len(PREFIX))
        engine.clear_prefix_cache()
        cold_b = self._generate(PREFIX + TAIL_B)
        self.assertEqual(cold_b["matched"], 0)
        self.assertEqual(warm_b["tokens"], cold_b["tokens"])

    def test_concurrent_identical_prompts_reuse_the_prefix(self) -> None:
        engine = self.ENGINE
        engine.clear_prefix_cache()
        prompt = PREFIX + TAIL_A
        # First request is admitted alone so its prefill populates the tree; the
        # second one is admitted while it decodes and must reuse the prefix.
        # Its first token comes out of that first step, so collect the event too.
        first = engine.add_request(prompt, SamplingParams(max_tokens=4,
                                                          temperature=0.0))
        out: list[list[int]] = [
            [tok for seq, tok, _done in engine.step()
             if seq is first and tok is not None],
            [],
        ]
        seqs = [first,
                engine.add_request(prompt, SamplingParams(max_tokens=4,
                                                          temperature=0.0))]
        while not all(s.is_finished for s in seqs):
            for seq, tok, _done in engine.step():
                for i, target in enumerate(seqs):
                    if seq is target and tok is not None:
                        out[i].append(tok)
        self.assertEqual(out[0], out[1])
        self.assertGreaterEqual(seqs[1].matched, BLOCK,
                                "the second concurrent request should reuse the prefix")

    def test_grouped_prefill_populates_the_prefix_cache(self) -> None:
        engine = self.ENGINE
        engine.clear_prefix_cache()
        # Two prompts admitted together go through one grouped prefill; it still
        # has to leave block-aligned snapshots behind for the next request.
        a = engine.add_request(PREFIX + TAIL_A,
                               SamplingParams(max_tokens=3, temperature=0.0))
        b = engine.add_request(PREFIX + TAIL_B,
                               SamplingParams(max_tokens=3, temperature=0.0))
        out_a: list[int] = []
        while not all(s.is_finished for s in (a, b)):
            for seq, tok, _done in engine.step():
                if seq is a and tok is not None:
                    out_a.append(tok)
        self.assertEqual(a.matched, 0)
        warm = self._generate(PREFIX + TAIL_A, 3)
        self.assertGreaterEqual(warm["matched"], BLOCK)
        self.assertEqual(warm["tokens"], out_a)

    def test_prefix_stats_report_cached_blocks(self) -> None:
        engine = self.ENGINE
        engine.clear_prefix_cache()
        self._generate(PREFIX + TAIL_A, 2)
        stats = engine.stats()
        self.assertIn("prefix", stats)
        self.assertGreater(stats["prefix"]["cached_blocks"], 0)
        self.assertGreater(stats["prefix"]["snapshots"], 0)
