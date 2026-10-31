"""分页 KV 和前缀缓存的存储原语。

Module: ``coffin.cache``。本包不感知模型，由 ``coffin.engine`` 使用；不许
import ``serving`` 或模型族代码，否则依赖图会反向成环。

Constraints:
    * 只做存储与索引，不碰设备、不做算子派发；形状和 dtype 由调用方
      保证。
"""

from .paged_cache import OutOfBlocks, PagedKVCache, SeqKV
from .radix_cache import RadixCache, RadixNode

__all__ = ["OutOfBlocks", "PagedKVCache", "SeqKV", "RadixCache", "RadixNode"]
