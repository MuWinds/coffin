"""Suffix drafter contract: the drafts must be real continuations of the observed history.

Module: ``coffin/tests``. Imports only ``ops.suffix``, so it runs on a CPU-only
host with no card, no checkpoint and no ``support`` harness; the on-card probe in
``diag`` is what re-checks the same rule against the live target model.

Constraints:
    * Every draft is checked against the history the drafter observed, never
      against a hand-written expectation of the whole list. A test that hardcodes
      the answer stops guarding "the draft is a real slice of history" the moment
      the lookup rule changes, which is the one property the verifier depends on.
    * No GPU, no ``torch`` and no device setup in this file: the engine's whole
      reason for a host-side drafter is that speculation must cost no device
      work, and a test that needs a card cannot gate that on CI.
    * The assertions are on tokens and lengths, not on timing. Acceptance rate is
      a quality property measured by the probes; a wall-clock bound here would
      flake on a loaded machine.
"""

from __future__ import annotations

import random
import unittest

from ops.suffix import SuffixDrafter


def _ids(words: list[str]) -> list[int]:
    """Map a word list to stable token ids so a test reads as the text it models.

    Args:
        words: Tokens of the sample text, in order.

    Returns:
        One integer per word, assigned in first-appearance order. Distinct words
        get distinct ids, so a match in the test can only come from the text.
    """
    vocab: dict[str, int] = {}
    out: list[int] = []
    for word in words:
        if word not in vocab:
            vocab[word] = len(vocab) + 1
        out.append(vocab[word])
    return out


def _continuation(history: list[int], draft: list[int]) -> int:
    """Find the position whose continuation equals ``draft``, or ``-1``.

    Args:
        history: The drafter's full history.
        draft: The tokens the drafter returned.

    Returns:
        The end position ``p`` such that ``history[p+1 : p+1+len(draft)] == draft``,
        with ``p <= len(history) - 2``; ``-1`` when no such position exists.
    """
    width = len(draft)
    for p in range(len(history) - 1 - width + 1):
        if history[p + 1 : p + 1 + width] == draft:
            return p
    return -1


class SuffixDrafterTest(unittest.TestCase):
    """The draft must be findable verbatim in the history the drafter was given."""

    def test_repeated_phrase_drafts_the_continuation(self):
        """A phrase seen once must predict its own continuation, which is the whole payoff.

        If this fails the drafter is not finding earlier occurrences at all, and
        every later assertion about the selection rule would pass vacuously on an
        empty draft.
        """
        words = "the cat sat on the mat. the cat sat on the".split()
        history = _ids(words)
        vocab = dict(zip(words, history))

        drafter = SuffixDrafter(max_match=4, min_match=2)
        drafter.reset(history)

        # The tail is "cat sat on the", seen once before at positions 1-4, and
        # the text right after that earlier occurrence is "mat. the cat".
        self.assertEqual(drafter.draft(3), [vocab[w] for w in ("mat.", "the", "cat")])
        self.assertEqual(drafter.draft(1), [vocab["mat."]])

    def test_disjoint_history_drafts_nothing(self):
        """No earlier occurrence of the suffix must yield an empty draft, not a guess.

        An empty draft has to be the honest answer: a speculative step with no
        draft still produces the plain greedy token, while a fabricated token
        would be rejected and waste the verify slot.
        """
        drafter = SuffixDrafter(max_match=4, min_match=2)
        drafter.reset([1, 2, 3, 4, 5])
        self.assertEqual(drafter.draft(3), [])

    def test_non_positive_k_returns_nothing(self):
        """``k <= 0`` must be a no-op, since callers disable drafting by passing zero.

        Returning the whole remainder of the history here would silently turn a
        disabled draft into an unbounded one.
        """
        drafter = SuffixDrafter()
        drafter.reset(_ids("the cat sat on the mat. the cat sat on the".split()))
        self.assertEqual(drafter.draft(0), [])
        self.assertEqual(drafter.draft(-3), [])

    def test_empty_history_returns_nothing(self):
        """A fresh request has no history, so draft must answer before any lookup.

        The engine drafts on the first decode step after prefill; an index error
        there would take down the request rather than cost one slow step.
        """
        drafter = SuffixDrafter()
        drafter.reset([])
        self.assertEqual(drafter.draft(4), [])

    def test_draft_length_and_slice_are_bounded(self):
        """Every draft is at most ``k`` tokens and a verbatim slice of the history.

        The verifier accepts a draft only because the target model can check it
        against its own output; a draft token that never occurred in the history
        would make an accepted token unverifiable, so the slice property is
        checked over the whole stream rather than on one example.
        """
        rng = random.Random(0)
        stream = [rng.randrange(6) for _ in range(120)]
        drafter = SuffixDrafter(max_match=4, min_match=2)
        for i, token in enumerate(stream):
            drafter.observe([token])
            if i < 8:
                continue
            for k in (1, 3, 6):
                draft = drafter.draft(k)
                self.assertLessEqual(len(draft), k, f"draft {draft} at step {i} exceeds k={k}")
                if draft:
                    self.assertGreaterEqual(
                        _continuation(stream[: i + 1], draft), 0,
                        f"draft {draft} at step {i} is not a continuation of the history",
                    )

    def test_observe_matches_reset(self):
        """Incremental indexing must equal a one-shot rebuild, results and index size alike.

        The engine appends one token per step, so ``observe`` is the hot path and
        ``reset`` is the rare one; if they diverge, a request drafted from an
        incrementally built index silently behaves differently from a
        prefix-cache-hit request that rebuilt it.
        """
        rng = random.Random(7)
        stream = [rng.randrange(5) for _ in range(80)]
        incremental = SuffixDrafter(max_match=4, min_match=2)
        rebuilt = SuffixDrafter(max_match=4, min_match=2)
        for i, token in enumerate(stream):
            incremental.observe([token])
            rebuilt.reset(stream[: i + 1])
            self.assertEqual(incremental.index_size, rebuilt.index_size, f"index size at step {i}")
            for k in (1, 2, 5):
                self.assertEqual(incremental.draft(k), rebuilt.draft(k), f"draft k={k} at step {i}")

    def test_suffix_does_not_match_itself(self):
        """The terminal suffix must not count as its own occurrence.

        If it did, ``p`` would be the last history position and the "draft" would
        be empty-or-past-the-end plus a fabricated first token; the guard is what
        keeps a first-occurrence-only suffix reported as no match.
        """
        # Both suffix tokens appeared earlier, but never as this pair, and the
        # pair itself ends the history: the only occurrence is the suffix.
        drafter = SuffixDrafter(max_match=4, min_match=2)
        drafter.reset([8, 1, 2, 8, 9])
        self.assertEqual(drafter.draft(2), [])

    def test_longest_match_beats_follow_frequency(self):
        """When a 4-gram and a 2-gram both occur, the 4-gram continuation wins.

        The 2-gram "3 4" is followed by token 8 twice and by token 7 once, so a
        frequency-first rule would draft 8. The 4-gram "1 2 3 4" is followed by 7
        and only occurs once. Preferring the longer suffix is what turns a match
        into several accepted tokens per verify instead of one lucky one.
        """
        history = [1, 2, 3, 4, 7, 3, 4, 8, 3, 4, 8, 0, 1, 2, 3, 4]
        longest = SuffixDrafter(max_match=4, min_match=2)
        longest.reset(history)
        self.assertEqual(longest.draft(1), [7])

        # The same history read with a 2-token cap is the frequency-only view;
        # asserting it drafts 8 shows the two rules really do disagree here.
        shortest = SuffixDrafter(max_match=2, min_match=2)
        shortest.reset(history)
        self.assertEqual(shortest.draft(1), [8])

    def test_history_shorter_than_min_match_is_safe(self):
        """A history below ``min_match`` has no legal lookup window and must not raise.

        Prefill can be as short as one token, so the drafter is asked for a draft
        before any match is possible; a raise here would fail a valid request.
        """
        drafter = SuffixDrafter(max_match=4, min_match=2)
        drafter.reset([42])
        self.assertEqual(drafter.draft(3), [])
        drafter.observe([43])
        self.assertEqual(drafter.draft(3), [])

    def test_draft_does_not_mutate_history(self):
        """Drafting is read-only, so repeated calls return the same tokens.

        The step calls ``draft`` and then ``observe`` in either order during
        rollback; a draft that grew the history would corrupt the next verify's
        notion of position.
        """
        history = _ids("the cat sat on the mat. the cat sat on the".split())
        drafter = SuffixDrafter()
        drafter.reset(history)
        fresh = SuffixDrafter()
        fresh.reset(history)
        first = drafter.draft(4)
        second = drafter.draft(4)
        self.assertEqual(first, second)
        self.assertEqual(drafter.index_size, fresh.index_size)


if __name__ == "__main__":
    unittest.main()
