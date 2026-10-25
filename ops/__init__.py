"""Device-side primitives, deliberately independent of any one checkpoint family.

A model family decides the shapes and the wiring; these modules decide how a
projection, a norm, RoPE or the delta recurrence is *executed* on this part.
Every one of them was benchmarked into its current form, so the measurement
notes travel with the code they justify.

Module: ``coffin/ops``. Bottom of the dependency graph: ``models/*`` and the
engine import from here, never the reverse. Each module below is re-exported so
callers can write ``from ops import GEMV`` instead of reaching into a submodule.

Constraints:
    * May not import ``models/`` or anything above it in the graph; this layer
      has to stay loadable, and unit-testable, without a checkpoint.
    * Absolute imports (``from ops.gemv import GEMV``) because the entry points
      put ``coffin/`` itself on ``sys.path``, making ``ops`` a top-level package
      next to ``hip_ops`` and the ``cache`` package -- see
      ``entrypoints/cli/runtime.py`` and ``run_chat.py``.
    * Keep the measured TFLOP/s and GB/s figures in the module docstrings: they
      are the only record of why the current kernel form won on gfx906.
"""

from ops.delta import chunked_delta
from ops.dtypes import D, NEG_INF, WF
from ops.embed import TiedEmbedding
from ops.gemv import GEMV, GEMVGroup, _proj
from ops.norm import RMSNorm, RMSNormGated
from ops.rope import apply_rope, make_rope_tables, rotate_half

__all__ = [
    "D", "NEG_INF", "WF",
    "GEMV", "GEMVGroup", "_proj",
    "TiedEmbedding",
    "RMSNorm", "RMSNormGated",
    "make_rope_tables", "rotate_half", "apply_rope",
    "chunked_delta",
]
