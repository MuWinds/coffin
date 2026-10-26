"""The tied embedding matrix: input row gather and shared ``lm_head`` projection.

One weight serves two very different roles -- a row gather per token id, and the
final projection, which is the single largest weight read of a decode step -- so
the gather dequantises only the rows it touches while the projection goes
through :class:`ops.gemv.GEMV`.

Module: ``coffin/ops``. Sits on ``ops/gemv``; used by ``models/qwen35/model.py``
for both the embedding lookup and the logits projection.

Constraints:
    * The gather must return bf16 (``D``) regardless of how the weight is stored:
      the KV cache and every kernel are written against bf16.
    * The int8 path dequantises per row with the row's own scale -- a whole-matrix
      dequantise would cost a full pass over the largest weight in the model.
    * Gather and projection are separate entry points on purpose; keeping the
      projection's kernel selection off the gather path is what makes the gather
      cheap.
"""

from __future__ import annotations

import torch

from int8_weights import QWeight
from ops.dtypes import D
from ops.gemv import GEMV


class TiedEmbedding:
    """The tied input-embedding / lm_head matrix, held as bf16 or int8.

    One weight serves two very different roles: a row gather per token id, and the
    final projection. Quantised, the gather dequantises only the rows it touches
    (a few thousand elements) while the projection reads the whole matrix -- the
    single largest weight read of a decode step -- so the two get separate paths.
    """

    def __init__(self, w, bench: bool = True) -> None:
        """Wrap the shared weight and build its projection path.

        Args:
            w: The embedding matrix, either a plain tensor or an
                :class:`int8_weights.QWeight`.
            bench: Forwarded to :class:`GEMV`; ``False`` skips the per-shape
                autotune, which is what makes structural tests cheap.
        """
        self.q = w if isinstance(w, QWeight) else None
        self.gemv = GEMV(w, bench)
        self.w = None if self.q is not None else w

    @property
    def shape(self) -> tuple[int, int]:
        """Weight shape, readable whichever storage form is live.

        Returns:
            ``(vocab, hidden)``; the int8 branch reports the logical shape of the
            dequantised matrix, not the packed buffer's.
        """
        return (self.q.n, self.q.k) if self.q is not None else tuple(self.w.shape)

    def lookup(self, ids: torch.Tensor) -> torch.Tensor:
        """Gather ``[T]`` token ids into their ``[T, hidden]`` bf16 embeddings.

        The int8 branch dequantises only the selected rows: a prefill of 2k tokens
        touches a few thousand rows of a ``[vocab, hidden]`` matrix, so a
        whole-matrix dequantise would dominate the step.

        Args:
            ids: Token ids, ``[T]``, already validated against the vocabulary by
                the caller.

        Returns:
            ``[T, hidden]`` bf16 activations.
        """
        if self.q is None:
            return self.w.index_select(0, ids).to(D)
        rows, scale = self.q.rows(ids)
        return (rows.float() * scale[:, None]).to(D)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Project one activation row against the tied matrix (decode path).

        Args:
            x: Activations, ``[hidden]`` or ``[C, hidden]``.

        Returns:
            Logits, ``[vocab]`` or ``[C, vocab]``.
        """
        return self.gemv(x)

    def prefill(self, x: torch.Tensor) -> torch.Tensor:
        """Project many activation rows against the tied matrix (prefill path).

        Args:
            x: Activations, ``[C, hidden]``.

        Returns:
            Logits, ``[C, vocab]``; see :meth:`ops.gemv.GEMV.prefill` for why the
            small-``C`` case takes a different kernel.
        """
        return self.gemv.prefill(x)
