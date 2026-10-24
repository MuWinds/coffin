"""Qwen3.5-4B (text-only): blocks, MTP head, weight loader and model class.

The checkpoint is a hybrid-attention model -- 24 Gated-DeltaNet layers and 8
GQA layers, plus a single MTP layer used as a speculative drafter -- so the
pieces here are the ones that are specific to that shape.  The math and the
kernels they run on live in ``ops/``.
"""

from models.qwen35.model import Qwen35
from models.qwen35.weights import load_weights

__all__ = ["Qwen35", "load_weights"]
