"""分页注意力与基数前缀复用不得改变答案。

Module: ``coffin/tests``。驱动 ``prefix_engine.PrefixCacheEngine``、
``paged_cache`` 与 ``radix_cache``，跑在同一个分页模型上；fixture 会在分页与
dense 两种配置之间切换，所以这里要付两次模型加载。

Constraints:
    * 相等性断言按 token 逐个比对，不做近似：分页只改变 int8 行存在哪里，不改
      算术，任何 token 漂移都是 bug。
    * 混合模型上，复用的 KV 只有配上产生它的循环状态才有效；只命中 KV 会让它
      之后的一切静默出错，所以状态快照是断言出来的，不是假定的。
    * ``_ENGINE`` 是模块级全局：块池和树就是被测的状态，每个测试新建一个引擎会
      把泄漏藏起来，淘汰断言也就失去意义。
    * 池容量故意设得很小，好让淘汰真的能发生；池一大就永远走不到无引用叶子那
      条路径。

四件事必须成立：

1. 分页解码 == dense 解码，逐 token 相等（分页内核通过槽位表寻址同样的 int8 行）；
2. 前缀缓存命中 == 冷的完整预填充——混合模型上这一条最关键；
3. 写时复制保护共享块，越过已匹配前缀继续解码不动树持有的任何块；
4. 池扛得住压力：淘汰只拿走无引用的叶子，释放时每个块都回来。

移植自 ``diag/check_paged.py``：只保留功能断言（计时打印留在那个探针里——测试
断言缓存复用了什么，不关心它有多快）。
"""

from __future__ import annotations

import functools
import hashlib
import unittest

import support

import torch

PROMPT = ("The history of Paris spans more than two thousand years. It began as a "
          "settlement on the banks of the Seine, known to the Romans as Lutetia. ")
SYS_A = ("You are a careful assistant. Answer questions about history and geography "
         "concisely, and cite the century when it matters. ") * 6
SYS_B = ("Reply in one short sentence. Prefer concrete dates and place names over "
         "general remarks. Do not repeat the question. ") * 6

_ENGINE = None


def _engine():
    """全模块共用一个分页 ``PrefixCacheEngine``；fixture 换了才重建。"""
    global _ENGINE
    from engine.prefix_engine import PrefixCacheEngine
    model = support.ModelFixture.get(quant=False, max_seq=8192, paged=True)
    if _ENGINE is None or _ENGINE.model is not model:
        _ENGINE = PrefixCacheEngine(model, num_blocks=256, block_size=64,
                                    max_state_snapshots=8)
        support.ensure_captured(model)
    return _ENGINE


@functools.lru_cache(maxsize=1)
def _tokenizer():
    """分词器只加载一次：它是 10 MB 的 JSON，而每个测试都要重新编码文本。"""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(support.MODEL_DIR))


def _ids(text: str) -> list[int]:
    """把一段 prompt 片段编成 token id。"""
    return _tokenizer().encode(text)


def _greedy(model, ids: list[int], n: int) -> list[int]:
    """dense 基线：走未分页的普通路径贪心生成 ``n`` 个 token。"""
    model.reset()
    logits = model.prefill(ids)
    out: list[int] = []
    for _ in range(n):
        tok = int(logits.argmax())
        out.append(tok)
        logits = model.decode_step(tok)
    return out


def _tree_fingerprint(eng) -> str:
    """对基数树当前持有的每个块内容取哈希。

    取指纹而不是块 id 集合：写时复制可能挪动块或改引用计数而不改 KV 字节，那是
    允许的；原地覆写块才会改变哈希。
    """
    digest = hashlib.sha256()
    pool = eng.pool
    stack = [eng.radix.root]
    while stack:
        node = stack.pop()
        for block in node.blocks:
            sl = slice(block * pool.bs, (block + 1) * pool.bs)
            digest.update(pool.k[:, sl].cpu().numpy().tobytes())
            digest.update(pool.v[:, sl].cpu().numpy().tobytes())
        stack.extend(node.children.values())
    return digest.hexdigest()[:16]


@support.needs_model
class PagedDecodeTest(unittest.TestCase):
    """分页内核必须寻址到 dense 注意力会寻址的那些行，一行不多一行不少。"""

    def test_paged_decode_matches_dense(self):
        """槽位表错了照样能吐出像样的 token，所以两条路径必须逐 token 比对。"""
        ids = _ids(SYS_A) + _ids(" When was it founded?")
        dense_model = support.ModelFixture.get(quant=False, max_seq=8192, paged=False)
        support.ensure_captured(dense_model)
        dense = _greedy(dense_model, ids, 24)

        eng = _engine()
        eng.generate(ids, 4)                       # 先把页表预热
        run = eng.generate(ids, 24)
        self.assertEqual(run["tokens"], dense,
                         f"paged vs dense: {sum(a == b for a, b in zip(run['tokens'], dense))}/24 tokens")


@support.needs_model
class PrefixReuseTest(unittest.TestCase):
    """缓存命中必须和同样 token 的冷预填充分辨不出区别。"""

    def test_prefix_hit_reproduces_the_cold_answer(self):
        """第二次请求就是对照实现：同样的 prompt、带缓存的前缀，解码结果必须一致。"""
        q1 = _ids(" Which river runs through it?")
        ids = _ids(SYS_B) + q1
        eng = _engine()
        cold = eng.generate(ids, 24)
        self.assertEqual(cold["matched"], 0, "first use of this prefix must be cold")
        warm = eng.generate(ids, 24)
        self.assertGreaterEqual(warm["matched"], len(_ids(SYS_B)) - 64,
                                f"matched {warm['matched']} of {len(ids)}")
        self.assertEqual(warm["tokens"], cold["tokens"],
                         "cache hit changed the greedy answer")

    def test_warm_prefill_reuses_the_prefix_and_keeps_the_answer(self):
        """长 prompt 会被分块，所以复用要跨过好几个预填充分块，而不只是一个。"""
        q1 = _ids(" When was it founded?")
        q2 = _ids(" Which river runs through it?")
        base = _ids(SYS_A) + q1 + q2 + _ids(PROMPT)
        long_prompt = [base[i % len(base)] for i in range(6000)]   # 3 个预填充分块
        eng = _engine()
        cold = eng.prefill(long_prompt).float().clone()
        warm = eng.prefill(long_prompt).float().clone()
        # 只断言功能：缓存覆盖了 prompt，复用路径给出的下一个 token 和完整预填充
        # 相同（省下多少时间由探针去量）。
        self.assertGreaterEqual(eng.last_matched, len(long_prompt) - 1024,
                                f"matched {eng.last_matched} of {len(long_prompt)}")
        self.assertEqual(int(cold.argmax()), int(warm.argmax()),
                         "warm prefill picked a different next token")
        unit = max(float(cold.abs().max()), 1e-6) * 2.0 ** -8    # 一个 bf16 ULP
        self.assertLess(float((cold - warm).abs().max()), 4.0 * unit)


@support.needs_gpu
class CopyOnWriteTest(unittest.TestCase):
    """一个块的第二个引用绝不允许被原地写穿。"""

    def test_shared_block_is_not_overwritten_and_writer_copies(self):
        """原地写共享块会改坏树里那份拷贝，而写者自己还觉得一切正常。"""
        from cache.paged_cache import PagedKVCache, SeqKV
        pool = PagedKVCache(1, 2, 8, num_blocks=4, block_size=4, max_seq=16)
        block = pool.alloc()
        pool.k[0, block * 4:(block + 1) * 4] = 7
        pool.incref(block)                         # 第二个持有者（树）
        seq = SeqKV(pool, [block], num_tokens=3)   # 尾部只写了一部分，仍是共享的
        before = pool.k[0, block * 4:(block + 1) * 4].clone()
        seq.reserve(1)
        self.assertTrue(torch.equal(before, pool.k[0, block * 4:(block + 1) * 4]),
                        "a shared block was overwritten in place")
        self.assertNotEqual(seq.blocks[-1], block, "the writer did not get its own block")
        self.assertGreaterEqual(pool.n_cows, 1)


@support.needs_model
class TreeBlocksTest(unittest.TestCase):
    """树拥有自己的块；解码中的序列只能读它们。"""

    def test_decoding_past_a_matched_prefix_leaves_the_tree_untouched(self):
        """越过命中点继续解码要写新块，任何原地写入都会毒化共享前缀。"""
        ids = _ids(SYS_A) + _ids(" When was it founded?")
        eng = _engine()
        eng.generate(ids, 8)                       # 先让前缀进入树
        before = _tree_fingerprint(eng)
        eng.generate(ids, 16)                      # 越过已匹配前缀继续解码
        self.assertEqual(before, _tree_fingerprint(eng))


@support.needs_gpu
class PoolEvictionTest(unittest.TestCase):
    """淘汰和释放之后引用计数都必须仍然自洽。"""

    def test_eviction_spares_live_blocks_and_release_returns_them(self):
        """淘汰活序列的块就是数据丢失；引用计数泄漏则等于池再也长不回来。"""
        from cache.paged_cache import PagedKVCache, SeqKV
        from cache.radix_cache import RadixCache
        pool = PagedKVCache(2, 4, 32, num_blocks=4, block_size=8, max_seq=64)
        radix = RadixCache(pool, max_state_snapshots=2)
        live = SeqKV(pool)
        live.reserve(16)
        live.commit_upto(16)                       # 4 个块里占了 2 个
        radix.evict(4)
        self.assertEqual(pool.num_used, 2, "eviction touched a live sequence's blocks")
        live.free()
        self.assertEqual(pool.num_free, 4, "release did not return the blocks")
        self.assertTrue(all(r >= 0 for r in pool.refcount), f"refcounts {pool.refcount}")
