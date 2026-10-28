"""Host-side suffix drafter: speculative decoding from the request's own token history.

Module: ``coffin/ops``. Called by the spec step in the engine in place of the MTP
drafter; nothing in ``coffin`` calls down into this module, so it sits off to the
side of the graph and carries no dependency of its own.

Constraints:
    * Pure Python, stdlib only, no ``torch``, no device access, no threads. The
      card this engine targets is damaged by CUDA graphs (AGENTS.md), which is
      what makes a draft model unusable, so the speculation has to come from the
      tokens themselves and must not perturb the device at all. The ``ops``
      package ``__init__`` imports torch for the kernels; nothing here may add to
      that, or this drafter would stop being testable on a CPU-only host.
    * The index is rebuilt only by ``reset`` and extended only by ``observe``, so
      a draft can never contain a token the model has not produced yet. The
      history is fed from the target model's own output; a bug that fed drafts
      back in would make the index self-fulfilling and the acceptance rate
      meaningless.
    * Longest suffix first, then follow-frequency. The target model verifies every
      draft, so a wrong long match costs one rejected slot, while a short match
      was usually reachable anyway; preferring the longer match is what keeps the
      accepted-token count per verify up.
    * One draft is a few dictionary lookups and zero weight reads: the cost model
      that justifies this module against a draft model at all, given 20.26 ms per
      decode step on this part.
"""

from __future__ import annotations


class SuffixDrafter:
    """Predict the next tokens by looking up earlier occurrences of the recent suffix.

    The drafter holds the token history plus an index from every m-token window
    (``1 <= m <= max_match``) to the positions at which that window ends. Drafting
    the same suffix twice is a hash hit, and no weight is read.

    The index is append-only between resets, which is what makes ``observe`` cost
    ``O(len(tokens) * max_match)`` rather than a rebuild per step.
    """

    def __init__(self, max_match: int = 4, min_match: int = 2) -> None:
        """Configure the longest and shortest suffix a lookup may use.

        Args:
            max_match: Longest suffix length in tokens. Four is the default
                because a longer lookup rarely occurs twice inside one prompt,
                and each miss is a wasted probe of the index.
            min_match: Shortest suffix that still counts as a match. At one token
                a lookup hits everywhere and drafts pure noise, which costs a
                verify slot on every step.

        Raises:
            ValueError: ``min_match < 1`` or ``max_match < min_match``. Both make
                the match loop empty or meaningless, and silently drafting
                nothing would look like a quality problem instead of a config
                bug.
        """
        if min_match < 1:
            raise ValueError(f"min_match must be >= 1, got {min_match}")
        if max_match < min_match:
            raise ValueError(
                f"max_match ({max_match}) must be >= min_match ({min_match})"
            )
        self.max_match = max_match
        self.min_match = min_match
        self._history: list[int] = []
        # One dict per m, keyed by the m-token window, valued by ascending end
        # positions. Splitting by m keeps the window itself out of the key, and a
        # per-m dict is smaller than one dict keyed by (m, window) pairs.
        self._index: list[dict[tuple[int, ...], list[int]]] = [
            {} for _ in range(max_match + 1)
        ]
        self._index_size = 0

    @property
    def index_size(self) -> int:
        """Report how many (window, position) entries the index holds.

        Reads the counter rather than the dicts, so a probe can watch the index
        grow per step without walking it. Equal values for an ``observe``-fed and
        a ``reset``-fed drafter prove the incremental path indexes the same
        history, which is the invariant a stale counter would hide.

        Returns:
            Total number of indexed positions across all window lengths.
        """
        return self._index_size

    def reset(self, tokens: list[int]) -> None:
        """Replace the history with ``tokens`` and rebuild the index from scratch.

        Used at the start of a request and after a prefix-cache hit, where the
        reused prefix arrives as one block. The rebuild is the same loop
        ``observe`` runs, so the two paths can never disagree about which
        positions exist.

        Args:
            tokens: The request's tokens in order, prompt first. An empty list
                clears the drafter.
        """
        self._history = list(tokens)
        self._index = [{} for _ in range(self.max_match + 1)]
        self._index_size = 0

        # Index from the reset state rather than from an empty drafter plus a
        # copy of the old index: one code path means one notion of "positioned".
        self._index_positions(0)

    def observe(self, tokens: list[int]) -> None:
        """Append ``tokens`` to the history and index only the new positions.

        This is the per-step entry point: a decode step produces one token, so the
        work here is ``max_match`` slice-and-insert operations, not a rescan of
        the prompt.

        Args:
            tokens: Newly produced tokens in order. Empty input is a no-op.
        """
        if not tokens:
            return
        start = len(self._history)
        self._history.extend(tokens)
        self._index_positions(start)

    def draft(self, k: int) -> list[int]:
        """Predict up to ``k`` tokens that continue the end of the history.

        The selection rule, in order:

        1. Try suffix length ``m`` from ``max_match`` down to ``min_match`` and
           take the first length that occurs earlier in the history. An occurrence
           is an end position ``p`` whose window ``history[p-m+1 : p+1]`` equals
           the suffix, with ``p <= len(history) - 2`` so the suffix cannot match
           itself, and only positions with a following token count.
        2. Among the occurrences of that length, take as first draft the token
           that most often follows the suffix; ties go to the choice whose latest
           occurrence is closest to the end, because recent text predicts the
           next token better than old text.
        3. Continue from the occurrence of that first token closest to the end,
           so the extension is a real continuation rather than a splice of two
           different occurrences.
        4. Return ``history[p+1 : p+1+k]``, truncated at the history end.

        The draft is a slice of observed history, never a synthesis, so a token it
        proposes outside the model's own output is impossible by construction.

        Args:
            k: Maximum number of tokens to predict. Zero or negative returns
                nothing, which lets the caller disable drafting without a branch.

        Returns:
            Between ``1`` and ``k`` tokens, or ``[]`` when no suffix of at least
            ``min_match`` tokens occurs earlier in the history.
        """
        history = self._history
        n = len(history)
        if k <= 0 or n < self.min_match:
            return []

        # p == n - 1 is the suffix itself; allowing it would draft the token that
        # follows the suffix inside the suffix's own window.
        last_usable = n - 2
        for m in range(min(self.max_match, n), self.min_match - 1, -1):
            positions = self._index[m].get(tuple(history[n - m :]))
            if not positions:
                continue

            counts: dict[int, int] = {}
            latest: dict[int, int] = {}
            for p in positions:
                if p > last_usable:
                    continue
                token = history[p + 1]
                counts[token] = counts.get(token, 0) + 1
                if p > latest.get(token, -1):
                    latest[token] = p
            if not counts:
                continue

            # count first, recency second; two tokens cannot share a latest
            # position, so the order is total and the draft is reproducible.
            first = max(counts, key=lambda token: (counts[token], latest[token]))
            best = latest[first]
            # best <= n - 2, so at least history[best + 1] exists and a k >= 1
            # draft always has a first token.
            return history[best + 1 : best + 1 + k]

        return []

    def _index_positions(self, start: int) -> None:
        """Index every window that ends at or after ``start``, for all lengths.

        Only windows ending at a position at or after ``start`` are new: an
        earlier end position was indexed when its token arrived, so touching it
        again would duplicate it in the occurrence list and double its weight in
        the follow-frequency vote.

        Args:
            start: First history index whose window may be new.
        """
        history = self._history
        n = len(history)
        for end in range(start, n):
            for m in range(1, self.max_match + 1):
                if end + 1 < m:
                    continue
                position_list = self._index[m].setdefault(
                    tuple(history[end - m + 1 : end + 1]), []
                )
                position_list.append(end)
                self._index_size += 1
