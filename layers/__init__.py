"""Model-independent layer and kernel entry points.

Module: ``coffin.layers``. The implementation currently lives in ``ops`` and
``models`` for compatibility; this namespace is the stable place for new layer
families to be exposed.
"""

from ops import (
    GEMV,
    GEMVGroup,
    RMSNorm,
    RMSNormGated,
    TiedEmbedding,
    apply_rope,
    chunked_delta,
    make_rope_tables,
    rotate_half,
)

__all__ = [
    "GEMV",
    "GEMVGroup",
    "RMSNorm",
    "RMSNormGated",
    "TiedEmbedding",
    "apply_rope",
    "chunked_delta",
    "make_rope_tables",
    "rotate_half",
]
