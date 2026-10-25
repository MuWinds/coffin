"""The MTP head is the speculative drafter: is its acceptance rate worth it?

Module: ``coffin/tests``. Reaches into ``model.mtp`` and the layer stack through
``support.ModelTestCase``; it measures the drafter alone, without the verify
path that ``test_spec.py`` exercises.

Constraints:
    * The drafter is fed the main model's real hidden states, not its own: it is
      trained on those, and feeding it its own output would measure drift rather
      than prediction quality.
    * The drafter's own KV cache is filled with the real tokens first, then
      reset; leaving it populated would leak a prefix into the next test.
    * Deeper drafts can never be accepted more often than shallower ones, which
      is why the monotonicity check is an assertion and not a comparison against
      a measured number.
    * The 0.5 bar at k=1 is the business threshold: below roughly half the
      speculative path costs more per step than the step it saves, so a drafter
      under it is not worth keeping regardless of its exact rate.

Port of ``diag/check_mtp.py`` (its GEMV/profile sections stay in that probe; this
is the functional acceptance test, using the integrated ``model.mtp``).
"""

from __future__ import annotations

import functools
import unittest

import support

import torch

K = 5
TEXT = ("The capital of France is Paris. The capital of Germany is Berlin. "
        "The capital of Italy is Rome. The capital of Spain is Madrid. "
        "The capital of Portugal is Lisbon. The capital of Greece is Athens. "
        "Water boils at 100 degrees Celsius at sea level. The Moon orbits the "
        "Earth roughly once every 27 days. Light travels at about 300,000 km/s.")


@functools.lru_cache(maxsize=1)
def _prompt_ids() -> list[int]:
    """Tokenise the fixed prompt once; the acceptance sweep compares against these exact ids."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(support.MODEL_DIR))
    return tok.encode(TEXT)


@support.needs_model
class MtpDrafterTest(support.ModelTestCase):
    """The drafter's predictions must be good enough that speculating beats not speculating."""

    QUANT = True
    MAX_SEQ = 1024

    def setUp(self) -> None:
        """Cache the prompt ids and the shared RoPE/mask context the drafter expects."""
        super().setUp()
        self.ids = _prompt_ids()
        self.ctx = (self.model.cos_tbl, self.model.sin_tbl, self.model.mask_idx)

    def _main_hidden(self, ids: list[int]) -> torch.Tensor:
        """Hidden state after every prompt token (the drafter's ground truth)."""
        model = self.model
        toks = torch.tensor(ids, device="cuda")
        x = model.embed.lookup(toks)
        positions = torch.arange(len(ids), device="cuda")
        for layer in model.layers:
            x = layer.prefill(x, positions, self.ctx)
        return x

    def test_chained_acceptance_by_depth(self):
        """Per-depth acceptance rates must clear the k=1 bar and stay monotone, or the draft chain is wasted work."""
        model = self.model
        ids = self.ids
        assert len(ids) > K + 4, "prompt too short for the acceptance sweep"
        hidden = self._main_hidden(ids)

        # Fill the drafter's own KV cache over the prompt with the real tokens.
        model.mtp.reset()
        for j in range(len(ids) - 1):
            model.mtp.pos.fill_(j)
            model.mtp.step(hidden[j], torch.tensor([ids[j + 1]], device="cuda"), self.ctx)

        accepted = [0] * (K + 1)
        tested = 0
        for i in range(len(ids) - 2 - K, 0, -1):
            hits = 0
            h = hidden[i]
            tok_next = ids[i + 1]
            for k in range(K):
                model.mtp.pos.fill_(i + k)
                h, logits = model.mtp.step(h, torch.tensor([tok_next], device="cuda"), self.ctx)
                pred = int(logits.argmax())
                if pred != ids[i + 2 + k]:
                    break
                hits += 1
                tok_next = pred
            for k in range(1, hits + 1):
                accepted[k] += 1
            tested += 1
        model.mtp.reset()

        self.assertGreater(tested, 10, "not enough positions to measure acceptance")
        rates = [accepted[k] / tested for k in range(1, K + 1)]
        # Measured ~75% / 47% / 24% / 13% / 4% for k=1..5.
        self.assertGreater(rates[0], 0.5, f"k=1 acceptance {rates[0]:.1%}")
        self.assertGreater(rates[1], 0.2, f"k=2 acceptance {rates[1]:.1%}")
        # Deeper drafts cannot be accepted more often than shallower ones.
        for shallow, deep in zip(rates, rates[1:]):
            self.assertGreaterEqual(shallow + 1e-9, deep,
                                    f"acceptance not monotone: {rates}")
