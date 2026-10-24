"""Engine-level regression: ``LLMEngine`` must agree with the paths it wraps.

The split into ``LLMEngine`` / ``Scheduler`` / ``ModelRunner`` / ``BlockManager``
is only worth having if a request still produces exactly the tokens the direct
model loop produces.  These tests pin that down for both KV modes, plus the
prefix-cache round trip and the streaming iterator.

    python3 -m unittest discover -s tests -p "test_engine.py" -v
"""

from __future__ import annotations

import functools
import unittest

import support  # noqa: F401  (sets MKL + sys.path)

from config import EngineConfig
from llm_engine import LLMEngine
from sequence import SamplingParams

#: Any in-vocab ids work: the model only sees embeddings, not text.  Two blocks
#: long, so the radix tree has a block-aligned prefix to match.
PROMPT = [1000 + i for i in range(128)]
#: A second, disjoint prompt: prefix tests must not see each other's cache.
PROMPT_ALT = [2000 + i for i in range(128)]
TAIL = [23, 24, 25, 26, 27]


@functools.lru_cache(maxsize=1)
def _tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(support.MODEL_DIR))


def _ids(text: str) -> list[int]:
    return _tokenizer().encode(text)


def make_engine(model, *, prefix_cache: bool, max_seq: int,
                snapshot_interval: int = 512, max_num_seqs: int = 1) -> LLMEngine:
    cfg = EngineConfig(model_dir=str(support.MODEL_DIR), max_seq=max_seq,
                       bench=False, quant=True, prefix_cache=prefix_cache,
                       block_size=64, snapshot_interval=snapshot_interval,
                       max_num_seqs=max_num_seqs,
                       num_blocks=max(64, (max_seq + 63) // 64))
    engine = LLMEngine(cfg, model)
    engine.build()
    return engine


def greedy_reference(model, prompt: list[int], n: int) -> list[int]:
    """The pre-refactor loop: reset, prefill, then argmax + decode_step."""
    model.reset()
    logits = model.prefill(prompt)
    out: list[int] = []
    for _ in range(n):
        tok = int(logits.argmax())
        out.append(tok)
        logits = model.decode_step(tok)
    return out


class DenseEngineTest(support.ModelTestCase):
    """No prefix cache: the engine must add nothing on top of the model."""

    MAX_SEQ = 2048
    PAGED = False

    def test_generate_matches_the_reference_loop(self) -> None:
        engine = make_engine(self.model, prefix_cache=False, max_seq=self.MAX_SEQ)
        ref = greedy_reference(self.model, PROMPT, 6)
        res = engine.generate(PROMPT, SamplingParams(max_tokens=6, temperature=0.0))
        self.assertEqual(res["tokens"], ref)
        self.assertEqual(res["matched"], 0)
        self.assertEqual(engine.stats()["generated_tokens"], 6)

    def test_streaming_yields_the_same_tokens_as_generate(self) -> None:
        engine = make_engine(self.model, prefix_cache=False, max_seq=self.MAX_SEQ)
        res = engine.generate(PROMPT, SamplingParams(max_tokens=5, temperature=0.0))

        seq = engine.add_request(PROMPT, SamplingParams(max_tokens=5, temperature=0.0))
        streamed = list(engine.stream(seq))
        self.assertEqual(streamed, res["tokens"])

    def test_max_tokens_and_eos_stop_the_request(self) -> None:
        engine = make_engine(self.model, prefix_cache=False, max_seq=self.MAX_SEQ)
        res = engine.generate(PROMPT, SamplingParams(max_tokens=3, temperature=0.0))
        self.assertEqual(len(res["tokens"]), 3)
        self.assertEqual(res["stop_reason"], "length")

        eos = res["tokens"][2]
        expected: list[int] = []
        for tok in res["tokens"]:
            if tok == eos:
                break
            expected.append(tok)
        seq = engine.add_request(PROMPT,
                                 SamplingParams(max_tokens=8, temperature=0.0, eos={eos}))
        self.assertEqual(list(engine.stream(seq)), expected)
        self.assertEqual(seq.stop_reason, "eos")


@support.needs_model
class PagedEngineTest(unittest.TestCase):
    """Paged KV + radix prefix cache, one engine for the class."""

    MAX_SEQ = 1024
    ENGINE: LLMEngine | None = None
    MODEL = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.MODEL = support.ModelFixture.get(quant=True, max_seq=cls.MAX_SEQ,
                                             paged=True, bench=False)
        # The engine attaches the paged pool, so it has to exist before any reset.
        # ``snapshot_interval=64`` puts a recurrent-state snapshot on every block
        # boundary, which is what makes a 127-token prefix matchable at 64.
        cls.ENGINE = make_engine(cls.MODEL, prefix_cache=True, max_seq=cls.MAX_SEQ,
                                 snapshot_interval=64)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.ENGINE = None
        cls.MODEL = None
        support.ModelFixture.close()

    def test_engine_matches_the_prefix_engine_api(self) -> None:
        engine = self.ENGINE
        pc = engine.pc
        logits = pc.prefill(PROMPT_ALT)
        ref = list(pc.decode_iter(logits, 4, set(), 0.0))
        pc.release()
        res = engine.generate(PROMPT_ALT, SamplingParams(max_tokens=4, temperature=0.0))
        self.assertEqual(res["tokens"], ref)

    def test_second_prompt_hits_the_prefix(self) -> None:
        engine = self.ENGINE
        first = engine.generate(PROMPT, SamplingParams(max_tokens=3, temperature=0.0))
        second = engine.generate(PROMPT, SamplingParams(max_tokens=3, temperature=0.0))
        self.assertEqual(first["tokens"], second["tokens"])
        # ``tokens[:-1]`` is the matchable prefix, so only complete 64-token
        # blocks count; the longer prompt reuses one more block.
        self.assertEqual(second["matched"], (len(PROMPT) - 1) // 64 * 64)
        third = engine.generate(PROMPT + TAIL,
                                SamplingParams(max_tokens=3, temperature=0.0))
        self.assertEqual(third["matched"],
                         (len(PROMPT + TAIL) - 1) // 64 * 64)


@support.needs_model
class BatchedEngineTest(unittest.TestCase):
    """Static batch: concurrent requests must each match a solo decode.

    The prompts are real text (tokenised) rather than arbitrary ids: on synthetic
    ids the logits are near-uniform and a few bf16 ULPs from a different GEMM M
    flip the argmax, which is not the thing this test is about.
    """

    MAX_SEQ = 1024
    TEXTS = [
        "The quick brown fox jumps over the lazy dog. " * 2,
        "In 1789 the French Revolution began, and the century that followed "
        "reshaped European politics. ",
        "def fibonacci(n):\n    return n if n < 2 else fibonacci(n - 1) + "
        "fibonacci(n - 2)\n",
    ]

    @classmethod
    def setUpClass(cls) -> None:
        cls.PROMPTS = [list(_ids(text)) for text in cls.TEXTS]
        cls.MODEL = support.ModelFixture.get(quant=True, max_seq=cls.MAX_SEQ,
                                             paged=True, bench=False)
        cls.ENGINE = make_engine(cls.MODEL, prefix_cache=False, max_seq=cls.MAX_SEQ,
                                 max_num_seqs=4)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.ENGINE = None
        cls.MODEL = None
        support.ModelFixture.close()

    def _solo(self, prompt: list[int], budget: int) -> list[int]:
        engine = self.ENGINE
        seq = engine.add_request(prompt, SamplingParams(max_tokens=budget,
                                                        temperature=0.0))
        return list(engine.stream(seq))

    def _concurrent(self, budgets: list[int]) -> list[list[int]]:
        engine = self.ENGINE
        seqs = [engine.add_request(p, SamplingParams(max_tokens=b, temperature=0.0))
                for p, b in zip(self.PROMPTS, budgets)]
        out: list[list[int]] = [[] for _ in seqs]
        while not all(s.is_finished for s in seqs):
            for seq, tok, _done in engine.step():
                for i, target in enumerate(seqs):
                    if seq is target and tok is not None:
                        out[i].append(tok)
        return out

    def test_concurrent_requests_match_their_solo_decode(self) -> None:
        budgets = [5, 2, 4]          # different lengths leave a row hole mid-run
        refs = [self._solo(p, b) for p, b in zip(self.PROMPTS, budgets)]
        self.assertEqual(self._concurrent(budgets), refs)

    def test_grouped_prefill_matches_solo_prefill(self) -> None:
        budgets = [4, 4, 4]
        refs = [self._solo(p, b) for p, b in zip(self.PROMPTS, budgets)]
        self.assertEqual(self._concurrent(budgets), refs)
