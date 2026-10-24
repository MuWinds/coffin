"""The tied embedding matrix: input gather and shared ``lm_head`` overlap.

One weight serves two very different roles -- a row gather per token id, and
the final projection (the single largest weight read of a decode step) -- so
the gather dequantises only the rows it touches while the projection goes
through ``GEMV``.
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
        self.q = w if isinstance(w, QWeight) else None
        self.gemv = GEMV(w, bench)
        self.w = None if self.q is not None else w

    @property
    def shape(self) -> tuple[int, int]:
        return (self.q.n, self.q.k) if self.q is not None else tuple(self.w.shape)

    def lookup(self, ids: torch.Tensor) -> torch.Tensor:
        """``[T]`` token ids -> ``[T, hidden]`` bf16 embeddings."""
        if self.q is None:
            return self.w.index_select(0, ids).to(D)
        rows, scale = self.q.rows(ids)
        return (rows.float() * scale[:, None]).to(D)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.gemv(x)

    def prefill(self, x: torch.Tensor) -> torch.Tensor:
        return self.gemv.prefill(x)
