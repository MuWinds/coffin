"""Engine-level regression: ``LLMEngine`` must agree with the paths it wraps.

Module: ``coffin/tests``. The highest layer of the suite: it drives
``llm_engine.LLMEngine`` over ``config`` / ``sequence`` / ``scheduler`` /
``block_manager`` / ``model_runner`` and compares the result against the direct
model loop, so a scheduler or runner bug surfaces here rather than in the runtime
entry point.

Constraints:
    * The reference is the pre-refactor model loop (``greedy_reference``), never a
      second engine: two engines agreeing would pass with a shared bug.
    * Temperature 0 everywhere; sampling would make two runs incomparable.
    * The batch tests tokenise real text instead of using arbitrary ids, because
      on synthetic ids the logits are near-uniform and a few bf16 ULPs from a
      different GEMM M flip the argmax, which is not the thing under test.
    * Each class owns its engine and closes the fixture in ``tearDownClass``; the
      card cannot hold the dense and paged models at once.

The split into ``LLMEngine`` / ``Scheduler`` / ``ModelRunner`` / ``BlockManager``
is only worth having if a request still produces exactly the tokens the direct
model loop produces.  These tests pin that down for both KV modes, plus the
prefix-cache round trip and the streaming iterator.

    python3 -m unittest discover -s tests -p "test_engine.py" -v
"""

from __future__ import annotations

import functools
import unittest

# Side-effect import: sets MKL_THREADING_LAYER and puts coffin/ on sys.path.
import support  # noqa: F401

from engine.config import EngineConfig
from engine.llm_engine import LLMEngine
from engine.sequence import SamplingParams

#: Any in-vocab ids work: the model only sees embeddings, not text.  Two blocks
#: long, so the radix tree has a block-aligned prefix to match.
PROMPT = [1000 + i for i in range(128)]
#: A second, disjoint prompt: prefix tests must not see each other's cache.
PROMPT_ALT = [2000 + i for i in range(128)]
TAIL = [23, 24, 25, 26, 27]


@functools.lru_cache(maxsize=1)
def _tokenizer():
    """Load the tokenizer once; the batched tests reuse it for every prompt."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(support.MODEL_DIR))


def _ids(text: str) -> list[int]:
    """Token ids of one prompt text."""
    return _tokenizer().encode(text)


def make_engine(model, *, prefix_cache: bool, max_seq: int,
                snapshot_interval: int = 512, max_num_seqs: int = 1) -> LLMEngine:
    """Build an engine around an already-loaded model.

    ``num_blocks`` is sized from ``max_seq`` so a test cannot run out of KV and
    pass for the wrong reason; the floor of 64 keeps the smallest case valid.
    """
    cfg = EngineConfig(model_dir=str(support.MODEL_DIR), max_seq=max_seq,
                       bench=False, quant=True, prefix_cache=prefix_cache,
                       block_size=64, snapshot_interval=snapshot_interval,
                       max_num_seqs=max_num_seqs,
                       num_blocks=max(64, (max_seq + 63) // 64))
    engine = LLMEngine(cfg, model)
    engine.build()
    return engine


def greedy_reference(model, prompt: list[int], n: int) -> list[int]:
    """The pre-refactor loop: reset, prefill, then argmax + decode_step.

    This is the oracle for every engine test: it is the baseline the engine was
    refactored out of, so it must stay independent of scheduler and runner code.
    """
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
        """With the cache off, engine overhead must be invisible in the token stream."""
        engine = make_engine(self.model, prefix_cache=False, max_seq=self.MAX_SEQ)
        ref = greedy_reference(self.model, PROMPT, 6)
        res = engine.generate(PROMPT, SamplingParams(max_tokens=6, temperature=0.0))
        self.assertEqual(res["tokens"], ref)
        self.assertEqual(res["matched"], 0)
        self.assertEqual(engine.stats()["generated_tokens"], 6)

    def test_streaming_yields_the_same_tokens_as_generate(self) -> None:
        """``stream`` and ``generate`` share one loop; a divergence means a token is dropped or duplicated at a step boundary."""
        engine = make_engine(self.model, prefix_cache=False, max_seq=self.MAX_SEQ)
        res = engine.generate(PROMPT, SamplingParams(max_tokens=5, temperature=0.0))

        seq = engine.add_request(PROMPT, SamplingParams(max_tokens=5, temperature=0.0))
        streamed = list(engine.stream(seq))
        self.assertEqual(streamed, res["tokens"])

    def test_max_tokens_and_eos_stop_the_request(self) -> None:
        """Both stop conditions are protocol: the length cap bounds latency, the EOS cap keeps the EOS token out of the reply."""
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


class GraphOffTest(unittest.TestCase):
    """What ``build()`` must do when the backend cannot capture a graph.

    Uses stand-ins rather than a checkpoint: the behaviour under test is the branch
    ``build()`` takes before any weight is touched, so a real model would only make the
    test slower and platform-dependent.
    """

    class _Model:
        """The parts of ``Qwen35`` that ``build()`` reads on the graph-off branch."""

        def __init__(self, has_mtp: bool) -> None:
            """Record whether this stand-in has an MTP head.

            Args:
                has_mtp: Value of the ``mtp`` attribute; the drafter flag is only
                    meaningful when a head exists.
            """
            self.mtp = object() if has_mtp else None
            self.mtp_prefill = True
            self.reset_calls = 0

        def reset(self) -> None:
            """Count resets, so the test can prove the branch ran."""
            self.reset_calls += 1

    def _engine(self, graph: bool, batching: bool, spec_k: int, has_mtp: bool) -> LLMEngine:
        """Build an engine with the branch's inputs set directly.

        Args:
            graph: ``EngineConfig.graph``.
            batching: Whether the engine is configured for batching.
            spec_k: Drafts per speculative step.
            has_mtp: Whether the stand-in model has an MTP head.

        Returns:
            An unbuilt engine wrapping the stand-in model.
        """
        # ``prefix_cache=False`` and one sequence keep the constructor off the block
        # manager, which is the only thing in it that would need a real paged model.
        cfg = EngineConfig(model_dir="unused", max_seq=64, bench=False, graph=graph,
                           spec_k=spec_k, prefix_cache=False, max_num_seqs=1)
        engine = LLMEngine(cfg, self._Model(has_mtp))
        engine.batching = batching
        return engine

    def test_graph_off_disables_the_drafter_prefill(self):
        """Speculation cannot run without capture, so any drafter KV a prefill writes is unread work.

        That KV is one full-attention layer plus five projections over the whole prompt,
        214.8 ms of a 962.5 ms prefill at n=2048 on the card it was measured on. Nothing
        else in the graph-off branch notices, so the flag has to be cleared here.
        """
        engine = self._engine(graph=False, batching=False, spec_k=0, has_mtp=True)
        engine.build()
        self.assertFalse(engine.model.mtp_prefill)
        self.assertEqual(engine.model.reset_calls, 1)

    def test_graph_off_without_an_mtp_head_is_still_fine(self):
        """A checkpoint with no drafter must not be asked about one."""
        engine = self._engine(graph=False, batching=False, spec_k=0, has_mtp=False)
        engine.build()
        self.assertEqual(engine.model.reset_calls, 1)

    def test_graph_off_still_rejects_speculation(self):
        """Clearing the drafter flag must not soften the check that caught the real misconfiguration."""
        engine = self._engine(graph=False, batching=False, spec_k=2, has_mtp=True)
        with self.assertRaises(RuntimeError):
            engine.build()


@support.needs_model
class PagedEngineTest(unittest.TestCase):
    """Paged KV + radix prefix cache, one engine for the class."""

    MAX_SEQ = 1024
    ENGINE: LLMEngine | None = None
    MODEL = None

    @classmethod
    def setUpClass(cls) -> None:
        """One engine for the class: building it captures graphs, which is the slow part."""
        cls.MODEL = support.ModelFixture.get(quant=True, max_seq=cls.MAX_SEQ,
                                             paged=True, bench=False)
        # The engine attaches the paged pool, so it has to exist before any reset.
        # ``snapshot_interval=64`` puts a recurrent-state snapshot on every block
        # boundary, which is what makes a 127-token prefix matchable at 64.
        cls.ENGINE = make_engine(cls.MODEL, prefix_cache=True, max_seq=cls.MAX_SEQ,
                                 snapshot_interval=64)

    @classmethod
    def tearDownClass(cls) -> None:
        """Drop the engine before closing the model, so the pool is released first."""
        cls.ENGINE = None
        cls.MODEL = None
        support.ModelFixture.close()

    def test_engine_matches_the_prefix_engine_api(self) -> None:
        """The engine must produce the tokens of the raw prefix engine it wraps, not its own variant."""
        engine = self.ENGINE
        pc = engine.pc
        logits = pc.prefill(PROMPT_ALT)
        ref = list(pc.decode_iter(logits, 4, set(), 0.0))
        pc.release()
        res = engine.generate(PROMPT_ALT, SamplingParams(max_tokens=4, temperature=0.0))
        self.assertEqual(res["tokens"], ref)

    def test_second_prompt_hits_the_prefix(self) -> None:
        """The match must be block-aligned and bounded by the length actually verified, or a reused block would carry unseen KV."""
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
        """Tokenise the texts once and build a four-sequence engine with no prefix cache."""
        cls.PROMPTS = [list(_ids(text)) for text in cls.TEXTS]
        cls.MODEL = support.ModelFixture.get(quant=True, max_seq=cls.MAX_SEQ,
                                             paged=True, bench=False)
        cls.ENGINE = make_engine(cls.MODEL, prefix_cache=False, max_seq=cls.MAX_SEQ,
                                 max_num_seqs=4)

    @classmethod
    def tearDownClass(cls) -> None:
        """Drop the engine before closing the model, so the pool is released first."""
        cls.ENGINE = None
        cls.MODEL = None
        support.ModelFixture.close()

    def _solo(self, prompt: list[int], budget: int) -> list[int]:
        """Generate one prompt through the engine alone: the per-row reference."""
        engine = self.ENGINE
        seq = engine.add_request(prompt, SamplingParams(max_tokens=budget,
                                                        temperature=0.0))
        return list(engine.stream(seq))

    def _concurrent(self, budgets: list[int]) -> list[list[int]]:
        """Admit all prompts at once and step the engine until every sequence finishes."""
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
        """Batching must not change any request's tokens, including when a row finishes early."""
        budgets = [5, 2, 4]          # different lengths leave a row hole mid-run
        refs = [self._solo(p, b) for p, b in zip(self.PROMPTS, budgets)]
        self.assertEqual(self._concurrent(budgets), refs)

    def test_grouped_prefill_matches_solo_prefill(self) -> None:
        """Grouped prefill changes the GEMM shape, so this test is the guard that it still lands on the same tokens."""
        budgets = [4, 4, 4]
        refs = [self._solo(p, b) for p, b in zip(self.PROMPTS, budgets)]
        self.assertEqual(self._concurrent(budgets), refs)
