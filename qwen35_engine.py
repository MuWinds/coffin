"""Compatibility shim for the old single-module engine import.

The engine is split by responsibility now --

    ops/                  device primitives (GEMV, norms, rope, delta, embed)
    models/qwen35/        blocks, MTP head, weight loader, ``Qwen35``
    engine_api.py         the serving-facing surface: model contract, registry,
                          checkpoint dispatch and reload

-- and this module only re-exports the flat names that the ~50 ``diag/`` probes
and older callers import (``import qwen35_engine as E`` / ``from qwen35_engine
import chunked_delta``).  It adds no behaviour.  New code should import from
``engine_api``, ``ops`` or ``models.qwen35`` directly.
"""

from __future__ import annotations

import importlib
import sys

import hip_ops  # noqa: F401  -- probes reach it as ``E.hip_ops``
import torch
import torch.nn.functional as F

from engine_api import Model, create_model, register  # noqa: F401
from models.qwen35.blocks import Layer, MLP
from models.qwen35.full_attn import FullAttn
from models.qwen35.linear_attn import LinearAttn
from models.qwen35.model import Qwen35
from models.qwen35.mtp import MTP
from models.qwen35.tuning import (ATTN_SPLIT, CKPT, PREFILL_ATTN_KB,
                                  PREFILL_ATTN_TILE, PREFILL_MTP_CHUNK)
from models.qwen35.weights import load_weights
from ops.delta import _UPPER_MASK, _upper_mask, chunked_delta
from ops.dtypes import D, NEG_INF, WF
from ops.embed import TiedEmbedding
from ops.gemv import GEMV, GEMVGroup, _proj
from ops.norm import RMSNorm, RMSNormGated
from ops.rope import apply_rope, make_rope_tables, rotate_half

__all__ = [
    "D", "WF", "NEG_INF", "CKPT",
    "PREFILL_ATTN_TILE", "PREFILL_ATTN_KB", "PREFILL_MTP_CHUNK", "ATTN_SPLIT",
    "GEMV", "GEMVGroup", "_proj", "TiedEmbedding",
    "RMSNorm", "RMSNormGated",
    "make_rope_tables", "rotate_half", "apply_rope",
    "_UPPER_MASK", "_upper_mask", "chunked_delta",
    "MLP", "Layer", "LinearAttn", "FullAttn", "MTP", "Qwen35", "load_weights",
    "Model", "create_model", "register", "reload",
]


def reload() -> None:
    """Reload the engine modules, then this shim (see :func:`engine_api.reload`)."""
    from engine_api import reload as _reload

    _reload()
    importlib.reload(sys.modules[__name__])
