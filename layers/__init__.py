"""与模型无关的层与内核入口。

Module: ``coffin.layers``。实现目前为了兼容仍留在 ``ops`` 和 ``models`` 里；
新加入的层族应当暴露在这个命名空间下，它是稳定的位置。

Constraints:
    * 这里只重导出，不定义实现。导入面稳定是它存在的全部理由，所以不要
      在这里加逻辑，否则上层会绑到具体实现上。
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
