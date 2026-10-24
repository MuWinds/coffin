"""Device-side primitives, independent of any one checkpoint architecture.

A model family decides the shapes and the wiring; these modules decide how a
projection, a norm, RoPE or the delta recurrence is *executed* on this part.
They are the pieces that were benchmarked into their current form (see
``ENGINE.md`` §3 and ``NOTES.md``), so keep the measurement notes with them.

Imports are deliberately absolute (``from ops.gemv import GEMV``): the engine
directory itself is what the entry points put on ``sys.path`` -- see
``serve.py``/``run_chat.py`` -- so ``ops`` and ``models`` are top-level packages
next to ``hip_ops`` and ``paged_cache``.
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
