"""验证前缀共享结束后块能重新进入淘汰候选。

Module: coffin/tests。直接使用 CPU 分页池和基数树，覆盖调度器依赖的引用契约。

Constraints:
    * 活序列持有的块不可淘汰，最后一个序列释放后不得继续钉住缓存。
"""

import unittest

from cache.paged_cache import PagedKVCache
from cache.radix_cache import RadixCache


class RadixLifetimeTest(unittest.TestCase):
    """匹配、快照回退和边拆分都必须与按块释放相容。"""

    def make_cache(self):
        """建满池缓存，淘汰是否成功直接体现在空闲块数上。"""
        pool = PagedKVCache(1, 1, 4, num_blocks=4, block_size=4,
                            max_seq=16, device="cpu")
        radix = RadixCache(pool, 4)
        tokens = list(range(16))
        blocks = [pool.alloc() for _ in range(4)]
        radix.insert(tokens, blocks, {8: {"pos": 8}})
        pool.free_blocks(blocks)
        return pool, radix, tokens

    def test_last_reader_release_allows_eviction(self):
        """两个读者逐个释放；第一个结束不能回收，最后一个结束必须能回收。"""
        pool, radix, tokens = self.make_cache()
        first, _, _ = radix.match_prefix(tokens)
        second, _, _ = radix.match_prefix(tokens)
        pool.free_blocks(first)
        self.assertEqual(radix.evict(4), 0)
        pool.free_blocks(second)
        self.assertEqual(radix.evict(4), 4)
        self.assertEqual(pool.num_free, 4)

    def test_state_fallback_does_not_pin_discarded_match(self):
        """完整 KV 匹配回退到中间快照后，只释放返回的块也必须能清空池。"""
        pool, radix, tokens = self.make_cache()
        blocks, matched, states = radix.match_prefix(tokens, require_states=True)
        self.assertEqual(matched, 8)
        self.assertEqual(states, {"pos": 8})
        self.assertEqual(radix.evict(4), 0)
        pool.free_blocks(blocks)
        self.assertEqual(radix.evict(4), 4)
        self.assertEqual(pool.num_free, 4)

    def test_split_while_shared_does_not_leave_node_pinned(self):
        """读者存活时拆边，归还其块后两个节点都应恢复可淘汰。"""
        pool, radix, tokens = self.make_cache()
        blocks, _, _ = radix.match_prefix(tokens)
        radix.insert(tokens[:8], blocks[:2], {8: {"pos": 8}})
        self.assertEqual(radix.evict(4), 0)
        pool.free_blocks(blocks)
        self.assertEqual(radix.evict(4), 4)
        self.assertEqual(pool.num_free, 4)


if __name__ == "__main__":
    unittest.main()
