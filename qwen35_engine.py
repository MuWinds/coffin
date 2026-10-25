"""Legacy flat-name facade over the split engine; new code must not use it.

It adds no behaviour. It exists because ~50 ``diag/`` probes still reach the
engine through the old single-module names (``import qwen35_engine as E``,
``from qwen35_engine import chunked_delta``), and those probes are compared
against runs already recorded, so their imports are kept working rather than
rewritten. Import from ``engine_api``, ``ops`` or ``models.qwen35`` directly.

The split this facade hides:

    ops/                  device primitives (GEMV, norms, rope, delta, embed)
    models/qwen35/        blocks, MTP head, weight loader, ``Qwen35``
    engine_api.py         the serving-facing surface: model contract, registry,
                          checkpoint dispatch and reload

Module: ``coffin/qwen35_engine.py``. Top of the import graph and off it: it
imports ``engine_api`` / ``ops`` / ``models.qwen35``, and only ``diag/`` probes
import it, so the engine must never depend on this file.

Constraints:
    * Each ``__all__`` entry is the real object, re-exported unchanged. Probes
      compare and monkeypatch through these names, so a wrapper or a copy here
      would silently split the object the probe patches from the one the engine
      calls.
    * ``hip_ops`` is imported although nothing below references it: probes reach
      the compiled kernels as ``E.hip_ops``.
    * ``reload()`` must reload ``engine_api`` before this module. The flat names
      are bound at import time, so reloading the shim alone would leave them
      pointing at the stale module objects.
"""

from __future__ import annotations

import importlib
import sys

# Probes reach the compiled kernels as ``E.hip_ops``, so the name is re-exported
# here even though no code below touches it.
import hip_ops  # noqa: F401
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
    """Reload ``engine_api``, then this shim, so probes see the new modules.

    The order is load-bearing: the re-exports below are plain names bound at
    import time, so only re-running this module's import statements moves them
    onto the reloaded classes. Reloading ``engine_api`` first means those
    statements bind to the new objects instead of the stale ones.
    """
    from engine_api import reload as _reload

    _reload()
    importlib.reload(sys.modules[__name__])
