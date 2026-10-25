"""Speculative decoding must be exact, and its drafter must predict the next tokens.

Module: ``coffin/tests``. Drives ``model.spec_step`` and ``model.mtp`` through
``support.ModelTestCase``; it is the only test that captures the K+1-wide verify
graph, so it must run in its own process with the model loaded at this width.

Constraints:
    * Exactness is the invariant, not speed: greedy speculation may only change
      wall-clock, so the output must equal plain greedy token for token. A
      mismatch is a correctness bug in the draft/verify bookkeeping, never an
      acceptable approximation.
    * The verify graph is captured at K+1 tokens, the width the step needs,
      because every extra verified token costs a full recurrent update in each of
      the 24 linear-attention layers. Capturing wider than needed slows every
      step; capturing narrower fails at replay.
    * Acceptance rate is a quality property of the drafter's predictions, not a
      timing threshold; the speed-up itself is measured by the probes.

Port of ``diag/check_spec.py`` (the timing/cost-breakdown prints stay there).
"""

from __future__ import annotations

import functools
import unittest

import support

PROMPT = ("The capital of France is Paris. The capital of Germany is Berlin. "
          "The capital of Italy is Rome. Tell me about the history of")
N = 64


@functools.lru_cache(maxsize=1)
def _prompt_ids() -> list[int]:
    """Tokenise the fixed prompt once, so every subtest starts from the same ids."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(support.MODEL_DIR))
    return tok.encode(PROMPT)


@support.needs_model
class SpecDecodeTest(support.ModelTestCase):
    """Greedy speculation is a speed change only; the token stream must stay identical."""

    QUANT = True
    MAX_SEQ = 2048

    def setUp(self) -> None:
        """Share the prompt ids and make sure a verify graph exists before any spec step."""
        super().setUp()
        self.ids = _prompt_ids()
        support.ensure_captured(self.model)

    def _plain(self, n: int) -> list[int]:
        """Greedy baseline: argmax of each step fed back in, one token at a time."""
        model = self.model
        model.reset()
        logits = model.prefill(self.ids)
        out: list[int] = []
        for _ in range(n):
            tok = int(logits.argmax())
            out.append(tok)
            logits = model.decode_step(tok)
        return out

    def _spec(self, n: int, k: int) -> tuple[list[int], dict]:
        """Run the speculative loop for ``n`` tokens at draft depth ``k``.

        Two warm-up steps are discarded on purpose: the first replays hit
        unpopulated graph memory and would otherwise be counted in the stats.
        """
        model = self.model
        model.capture_verify(k + 1)        # pay only for the tokens we verify
        model.reset()
        logits = model.prefill(self.ids)
        for _ in range(2):                 # warm the replays
            _, logits = model.spec_step(logits, k)
        model.reset()
        logits = model.prefill(self.ids)
        model.n_drafted = model.n_accepted = model.n_spec_steps = 0
        out: list[int] = []
        while len(out) < n:
            toks, logits = model.spec_step(logits, k)
            out.extend(toks)
        return out[:n], model.spec_stats

    def test_capture_builds_the_verify_and_draft_graphs(self):
        """``spec_step`` replays both graphs, so a missing capture must fail here and not mid-generation."""
        self.assertIsNotNone(self.model._verify_graph)
        self.assertIsNotNone(self.model.mtp)
        self.assertIsNotNone(self.model.mtp._graph)

    def test_spec_matches_plain_greedy(self):
        """At every depth, accepted drafts plus fallbacks must reproduce plain greedy exactly."""
        plain = self._plain(N)
        for k in (1, 3, 7):
            with self.subTest(K=k):
                spec, stats = self._spec(N, k)
                self.assertEqual(spec, plain,
                                 f"K={k}: speculative output differs from greedy "
                                 f"({sum(a == b for a, b in zip(spec, plain))}/{N} tokens)")

    def test_drafter_acceptance_rate(self):
        _, stats = self._spec(N, 1)
        # Measured ~73% for K=1; this is the drafter's prediction quality, not a
        # timing threshold (the speed-up itself is measured by the probes).
        self.assertGreater(stats["accept_rate"], 0.5,
                           f"K=1 acceptance {stats['accept_rate']:.1%}")
