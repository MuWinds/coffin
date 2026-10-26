"""Qwen3.5-4B family entry point: model class, weight loader, tuning knobs.

The checkpoint is a hybrid-attention model -- 24 Gated-DeltaNet layers and 8
GQA layers, plus a single MTP layer used as a speculative drafter -- so the
pieces here are the ones specific to that shape. The math and the kernels they
run on live in ``ops/``.

Module: ``coffin/models/qwen35``. Top of the family: it may import ``ops`` and
its own siblings (``blocks``, ``batch``, ``linear_attn``, ``full_attn``, ``mtp``,
``vision``, ``weights``, ``tuning``), and is itself reached only through
``engine_api``, which dispatches on the checkpoint's ``config.json``.

Constraints:
    * Absolute imports only (``from models.qwen35.model import Qwen35``): the
      entry points put ``coffin/`` on ``sys.path``, so ``models`` is a top-level
      package next to ``ops`` and ``backends``.
    * Nothing in the family may import ``engine_api`` or anything above it --
      that module is the one place allowed to know which families exist.
    * No hardware detail: the device, the dtypes and the split-K width all come
      from ``backends``, so a new accelerator changes nothing in this package.
"""

from models.qwen35.model import Qwen35
from models.qwen35.weights import load_weights

__all__ = ["Qwen35", "load_weights"]
