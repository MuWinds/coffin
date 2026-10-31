"""基数树绝不能把一张图的 KV 交给另一张图。

Module: ``coffin/tests``。用一个记录型模型驱动 ``radix_cache`` / ``block_manager``
/ ``prefix_engine`` / ``model_runner``；它是唯一对缓存键本身断言、而不是对生成
token 断言的测试。

Constraints:
    * 分页池是真的，但分配在 CPU 上，模型只负责记录，所以本模块不需要 DCU。
      不要换成共享的 GPU fixture：这里的断言针对键与命中长度，与数值无关。
    * 块键必须保持 ``(tokens_block, media_signature)``，新内核或者多出来的一层
      缓存都无法把纯 token 键带回来。纯文本用例守的是「这么做没让文本路径付出
      代价」。
    * ``TEXT_MEDIA_ID`` 必须保持无图块哈希到的那个值；改了它，所有已存的纯文本
      块全部失效。
    * 测试没有打全局补丁，但 ``make_runner`` 会置 ``batching = True``，正是它把
      ``prefill`` 路由到 ``_prefill_batched``。

一段 ``<|image_pad|>`` 在每张图上都是同一串 token id，所以只按 token id 做键的
前缀缓存，会把*第一张*图的 KV 复用给携带*另一张*图的 prompt。这种故障是静默的，
而且随后缀一起增长，于是块键改成了 ``(tokens_block, media_signature)``：块内图像
内容 id 的集合，无图像 token 的块取 ``(0,)``。

四件事必须成立，本模块在没有 DCU 的条件下逐条检查：

1. token 相同、图像摘要不同 -> 零命中；
2. token 相同、摘要相同 -> 命中，且总落在块边界上；
3. 纯文本 -> 仍是原来的纯 token 键与原命中长度（不增加逐 token 开销，行为不变）；
4. 两条预填充路径（单流的 ``prefix_engine`` 与批处理的 ``model_runner``）插入并
   匹配同一套加盐键，并把 ``mm`` 转发给模型，让嵌入真正注入。

    PYTHONPATH=coffin python -m unittest coffin/tests/test_radix_mm.py -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parent.parent
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

import torch

from engine.config import EngineConfig
from utils.mm_input import MMInput
from cache.paged_cache import PagedKVCache
from cache.radix_cache import (RadixCache, TEXT_MEDIA_ID, media_id, media_ids_for,
                               media_signature)

BLOCK = 8
MAX_SEQ = 512
NUM_BLOCKS = 64
IMAGE = 248056                      # <|image_pad|>


# fixture
def make_pool() -> PagedKVCache:
    """造一个真的分页池，放在 CPU 上，整套用例因此不需要 DCU。"""
    return PagedKVCache(1, 1, 4, num_blocks=NUM_BLOCKS, block_size=BLOCK,
                        max_seq=MAX_SEQ, device="cpu")


def make_mm(digest: str, spans: list[tuple[int, int]], prompt_len: int) -> MMInput:
    """造一个 ``MMInput``，只填缓存真正读取的字段（embeds 与 positions 用不到）。"""
    n_img = sum(e - s for s, e in spans) or 1
    return MMInput(embeds=torch.zeros(n_img, 4), grid_thw=torch.ones(len(spans), 3),
                   positions=torch.zeros(3, prompt_len, dtype=torch.long),
                   rope_delta=0, digest=digest, spans=list(spans))


def prompt_with_image(n_text: int, n_img: int, prefix_text: int = 0) -> tuple[list[int], int]:
    """构造 ``prefix_text`` 个文本 id，接着一段图像，再接 ``n_text`` 个文本 id。"""
    ids = [100 + i for i in range(prefix_text)]
    span = (len(ids), len(ids) + n_img)
    ids += [IMAGE] * n_img
    ids += [900 + i for i in range(n_text)]
    return ids, span[0]


class FakeModel:
    """记录预填充路径向它要了些什么；自己只维护 ``pos1`` 和行计数。

    对 ``PrefixCacheEngine.prefill`` 与 ``ModelRunner._prefill_batched`` 提供的接口
    够用即可：两者只碰 ``pos1``/``pos``、线性状态快照辅助方法和 ``prefill``。
    """

    def __init__(self) -> None:
        """建一个空的调用记录和一个单行的位置计数器。"""
        self.pos1 = torch.zeros(1, dtype=torch.long)
        self.pos = self.pos1                     # 单行的暂存行就是第 0 行
        self.prefill_row = 0
        self.decode_capacity = 8
        self.calls: list[tuple[list[int], object]] = []
        self.states: dict[int, dict] = {}

    # 两条路径共用的状态钩子
    def reset(self) -> None:
        """把位置计数器清零，让下一条序列从 0 开始。"""
        self.pos1.zero_()

    def snapshot_linear_states(self, row: int = 0) -> dict:
        return {"row": row, "pos": int(self.pos1[row])}

    def restore_linear_states(self, states, row: int = 0) -> None:
        self.states[row] = dict(states)
        self.pos1[row] = int(states["pos"])

    def zero_linear_states(self, row: int) -> None:
        self.states[row] = {"row": row, "pos": 0}

    def move_linear_states(self, src_row: int, dst_row: int) -> None:
        """把一行的状态搬到另一行；同一行时不搬，免得自己覆盖自己。"""
        if src_row != dst_row:
            self.states[dst_row] = dict(self.states.get(src_row, {}))

    def prefill(self, tokens, chunk: int = 2048, spec_row=None, mm=None):
        """记下这次预填充的 token 与 ``mm``，并按 token 数推进暂存行的位置。

        多模态输入只能靠 ``mm`` 传进来，测试断言 ``mm`` 是否原样到达模型，所以
        它必须留在调用记录里。
        """
        self.calls.append((list(tokens), mm))
        self.pos1[self.prefill_row:self.prefill_row + 1] += len(tokens)
        return torch.zeros(4)


def make_engine(model: FakeModel):
    """在 CPU 池上搭一个单流的 ``PrefixCacheEngine``。"""
    from engine.block_manager import BlockManager
    from engine.prefix_engine import PrefixCacheEngine

    pool = make_pool()
    bm = BlockManager(pool, RadixCache(pool, 4), BLOCK)
    return PrefixCacheEngine(model, num_blocks=NUM_BLOCKS, block_size=BLOCK,
                             max_state_snapshots=4, snapshot_interval=BLOCK, bm=bm)


def make_runner(model: FakeModel, pool=None):
    """搭一个挂上基数树的批处理 ``ModelRunner``。"""
    from engine.block_manager import BlockManager
    from engine.model_runner import ModelRunner

    pool = pool or make_pool()
    cfg = EngineConfig(model_dir=".", max_seq=MAX_SEQ, bench=False, quant=False,
                       prefix_cache=True, block_size=BLOCK, snapshots=4,
                       snapshot_interval=BLOCK, num_blocks=NUM_BLOCKS)
    manager = BlockManager(pool, RadixCache(pool, 4), BLOCK)
    runner = ModelRunner(model, cfg)
    runner.attach_blocks(manager)
    runner.batching = True                  # 让它走 ``_prefill_batched``
    return runner, manager


class Seq:
    """``sequence.Sequence`` 的替身，只提供 runner 会读的字段。"""

    def __init__(self, token_ids: list[int], mm=None) -> None:
        """按 token 列表建一条序列，计数器全部归零。"""
        self.seq_id = id(self)
        self.token_ids = list(token_ids)
        self.mm = mm
        self.num_prompt_tokens = len(self.token_ids)
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.block_table: list[int] = []
        self.row = -1
        self.matched = 0
        self.params = None

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)


# 媒体 id 与签名辅助函数
class MediaIdTest(unittest.TestCase):
    """媒体 id 就是缓存的盐，因此必须确定，而且能塞进有符号 64 位的 token 槽位。"""

    def test_media_id_is_stable_positive_and_position_salted(self) -> None:
        """两次运行必须一致（id 一变，所有热缓存全部落空），且摘要与位置都要参与加盐。"""
        a = media_id("deadbeef", 0)
        self.assertEqual(a, media_id("deadbeef", 0), "same image must keep its id")
        self.assertNotEqual(a, media_id("deadbeef", 1), "index must salt the id")
        self.assertNotEqual(a, media_id("feedface", 0), "digest must salt the id")
        self.assertNotEqual(a, TEXT_MEDIA_ID)
        self.assertGreater(a, 0)
        self.assertLess(a, 1 << 63)

    def test_media_ids_and_signatures(self) -> None:
        """位置加盐让同一张图出现两次算作两个不同的跨度；块签名必须与块内实际持有的图像 token 精确对应。"""
        ids, img_start = prompt_with_image(n_text=4, n_img=12,
                                           prefix_text=2 * BLOCK + 3)
        mm = make_mm("d1", [(img_start, img_start + 12)], len(ids))
        media = media_ids_for(mm, len(ids))
        self.assertEqual(media[:img_start], [TEXT_MEDIA_ID] * img_start)
        self.assertEqual(set(media[img_start:img_start + 12]), {media_id("d1", 0)})
        self.assertEqual(media[img_start + 12:], [TEXT_MEDIA_ID] * 4)

        # 块签名：图像之前的块是文本块 ``(0,)``，跨度起始所在的块把图像 id 和
        # 文本 id 放在一起。
        self.assertEqual(media_signature(media, 0, BLOCK), (TEXT_MEDIA_ID,))
        self.assertEqual(media_signature(media, BLOCK, BLOCK), (TEXT_MEDIA_ID,))
        self.assertEqual(media_signature(media, img_start // BLOCK * BLOCK, BLOCK),
                         tuple(sorted({TEXT_MEDIA_ID, media_id("d1", 0)})))

    def test_text_only_mm_has_no_ids_and_no_signature_work(self) -> None:
        """``None`` 与空摘要是同一个意思——没有图像 id，于是文本路径一个逐 token 数组都不分配。"""
        ids = [1, 2, 3]
        mm = make_mm("", [], len(ids))
        self.assertIsNone(media_ids_for(mm, len(ids)))
        self.assertIsNone(media_ids_for(None, len(ids)))
        self.assertEqual(media_signature(None, 0, BLOCK), (TEXT_MEDIA_ID,))

    def test_spans_without_digest_are_rejected(self) -> None:
        """图像跨度没有摘要，所有图像就会撞成同一个键，正是本模块要防的那个 bug；宁可当场抛错。"""
        ids, img_start = prompt_with_image(n_text=4, n_img=8)
        mm = make_mm("", [(img_start, img_start + 8)], len(ids))
        with self.assertRaises(ValueError):
            media_ids_for(mm, len(ids))


# 核心主张：不跨图像复用 KV
class SaltTest(unittest.TestCase):
    """核心主张：图像 KV 绝不跨不同图像复用，而文本 KV 照旧复用。"""

    def setUp(self) -> None:
        """同一份 prompt，同一图像位置给两个摘要，每个用例一个独立的 CPU 池。"""
        self.pool = make_pool()
        self.radix = RadixCache(self.pool, max_state_snapshots=4)
        self.ids, self.img_start = prompt_with_image(n_text=9, n_img=12)
        self.mm_a = make_mm("image-A", [(self.img_start, self.img_start + 12)],
                            len(self.ids))
        self.mm_b = make_mm("image-B", [(self.img_start, self.img_start + 12)],
                            len(self.ids))

    def _blocks(self, n: int) -> list[int]:
        return [self.pool.alloc() for _ in range(n)]

    def test_same_tokens_different_digest_matches_nothing(self) -> None:
        """它防的故障是静默的：第二张图的 prompt 会直接在第一张图的 KV 上解码。"""
        blocks = self._blocks(len(self.ids) // BLOCK)
        self.radix.insert(self.ids, blocks, {}, mm=self.mm_a)
        got, matched, _ = self.radix.match_prefix(self.ids[:-1], mm=self.mm_b)
        self.assertEqual(matched, 0, "a different image reused the first image's KV")
        self.assertEqual(got, [])

    def test_same_tokens_same_digest_hits_on_a_block_boundary(self) -> None:
        """命中必须是整块：半个块既没法记引用计数，也没法写时复制。"""
        blocks = self._blocks(len(self.ids) // BLOCK)
        self.radix.insert(self.ids, blocks, {}, mm=self.mm_a)
        got, matched, _ = self.radix.match_prefix(self.ids[:-1], mm=self.mm_a)
        self.assertEqual(matched % BLOCK, 0, "match must land on a block boundary")
        self.assertEqual(matched, (len(self.ids) - 1) // BLOCK * BLOCK)
        self.assertEqual(len(got), matched // BLOCK)
        self.assertTrue(all(b in blocks for b in got))

    def test_text_prefix_before_the_image_is_still_shared(self) -> None:
        """加盐不能传染：图像之前的文本块与图像无关，必须继续可复用。"""
        # 图像在后面，不代表可共享的文本前缀要跟着丢掉：它之前的块不管换成哪张
        # 图，持有的都是同一份 KV。
        ids, img_start = prompt_with_image(n_text=8, n_img=12, prefix_text=3 * BLOCK)
        a = make_mm("image-A", [(img_start, img_start + 12)], len(ids))
        b = make_mm("image-B", [(img_start, img_start + 12)], len(ids))
        self.radix.insert(ids, self._blocks(len(ids) // BLOCK), {}, mm=a)
        _, matched, _ = self.radix.match_prefix(ids[:-1], mm=b)
        self.assertEqual(matched, 3 * BLOCK,
                         "the image-free text prefix should still be reusable")

    def test_plain_text_query_cannot_match_an_image_block(self) -> None:
        """反方向的泄漏：带占位 id 的纯文本查询不能认领图像分支，那会把另一张图的 KV 注进来。"""
        # id 里恰好含那段占位符、但没有设置 mm 的文本请求，签名是 (0,)，不能
        # 接上图像分支。
        self.radix.insert(self.ids, self._blocks(len(self.ids) // BLOCK), {},
                          mm=self.mm_a)
        _, matched, _ = self.radix.match_prefix(self.ids[:-1], mm=None)
        self.assertEqual(matched, 0)

    def test_image_query_reuses_a_text_branch_up_to_the_image(self) -> None:
        """复用不该依赖插入顺序：先插文本、再来图像请求，仍须共享无图像的头部。"""
        # 反方向：*文本*插入先把头部放进树里（纯 token 键），随后来了一个头部
        # 相同的图像请求。文本 KV 与图像无关，头部必须复用，而命中要停在第一个
        # 图像块之前。
        ids, img_start = prompt_with_image(n_text=4, n_img=12, prefix_text=3 * BLOCK)
        self.radix.insert(ids, self._blocks(len(ids) // BLOCK), {}, mm=None)
        mm = make_mm("image-A", [(img_start, img_start + 12)], len(ids))
        _, matched, _ = self.radix.match_prefix(ids[:-1], mm=mm)
        self.assertEqual(matched, 3 * BLOCK,
                         "the text head must be reusable, the image part not")

    def test_a_no_mm_query_never_looks_up_a_salted_key(self) -> None:
        """上面两个用例背后的机制：加盐边藏在纯 token 查法造不出来的键下面，所以 ``mm=None`` 一分钱不花。"""
        # 上面两个用例的不变量：加盐边所在的键，纯 token 查法根本产生不出来，
        # 于是 ``mm=None`` 连一次比较都不做，也永远够不到图像 KV。
        self.radix.insert(self.ids, self._blocks(len(self.ids) // BLOCK), {},
                          mm=self.mm_a)
        node = next(iter(self.radix.root.children.values()))
        self.assertIsNotNone(node.sigs)
        self.assertNotIn(tuple(self.ids[:BLOCK]), self.radix.root.children,
                         "a salted edge must not sit under a plain token key")
        plain_key = tuple(self.ids[:BLOCK])
        self.assertEqual(len(plain_key), BLOCK)

    def test_two_images_in_one_prompt_are_position_salted(self) -> None:
        """同一份字节发两次仍是两个不同的跨度；不加位置盐，第二张图就会复用第一张的 KV。"""
        ids, first = prompt_with_image(n_text=4, n_img=8)
        second = len(ids)
        ids += [IMAGE] * 8
        mm = make_mm("same-bytes", [(first, first + 8), (second, second + 8)], len(ids))
        media = media_ids_for(mm, len(ids))
        self.assertNotEqual(media[first], media[second],
                            "the same picture twice is still two different spans")
        blocks = self._blocks(len(ids) // BLOCK)
        self.radix.insert(ids, blocks, {}, mm=mm)
        _, matched, _ = self.radix.match_prefix(ids[:-1], mm=mm)
        self.assertEqual(matched, len(ids) // BLOCK * BLOCK)

    def test_image_span_crossing_a_block_salts_both_blocks(self) -> None:
        """跨越块边界的跨度必须给它碰到的每个块加盐；只给第一个加盐，重叠部分就暴露给复用了。"""
        # 跨度从块中间开始：装它头部的块和下一个块都得带上这个 id，两者都不可
        # 跨图像共享。
        ids, img_start = prompt_with_image(n_text=4, n_img=12, prefix_text=3)
        mm = make_mm("image-A", [(img_start, img_start + 12)], len(ids))
        media = media_ids_for(mm, len(ids))
        mid = media_id("image-A", 0)
        head = media_signature(media, 0, BLOCK)
        self.assertIn(mid, head, "the block holding the span head must be salted")
        self.assertIn(TEXT_MEDIA_ID, head)
        for start in range(BLOCK, len(ids) - BLOCK + 1, BLOCK):
            sig = media_signature(media, start, BLOCK)
            if any(t == IMAGE for t in ids[start:start + BLOCK]):
                self.assertIn(mid, sig, f"image block at {start} is not salted")
            else:
                self.assertEqual(sig, (TEXT_MEDIA_ID,),
                                 f"text block at {start} must keep the plain key")
        # 换成*另一张*图的查询，一个加盐块都共享不到
        blocks = [self.pool.alloc() for _ in range(len(ids) // BLOCK)]
        self.radix.insert(ids, blocks, {}, mm=mm)
        other = make_mm("image-B", [(img_start, img_start + 12)], len(ids))
        _, matched, _ = self.radix.match_prefix(ids[:-1], mm=other)
        self.assertEqual(matched, 0)


# 纯文本回归：纯 token、纯键、不做 mm 相关工作
class TextOnlyTest(unittest.TestCase):
    """文本路径先于图像存在，必须保住旧的键、旧的命中长度和零逐 token 开销。"""

    def setUp(self) -> None:
        """每个用例一个全新的 CPU 池，命中长度就不会从上一个用例里继承。"""
        self.pool = make_pool()
        self.radix = RadixCache(self.pool, max_state_snapshots=4)

    def test_keys_stay_plain_token_tuples(self) -> None:
        """纯文本插入不能开始产出加盐子键，否则已缓存的块全部变成不可达。"""
        ids = [1000 + i for i in range(5 * BLOCK)]
        blocks = [self.pool.alloc() for _ in range(5)]
        self.radix.insert(ids, blocks, {})
        keys = set(self.radix.root.children)
        self.assertEqual(keys, {tuple(ids[:BLOCK])},
                         "a text-only insert must keep the old child key format")
        self.assertIsNone(self.radix.root.children[tuple(ids[:BLOCK])].sigs)

    def test_match_lengths_are_unchanged(self) -> None:
        """命中长度仍是旧的「向下取整到块」值；偏短丢复用，偏长把未校验的 KV 交出去。"""
        ids = [1000 + i for i in range(5 * BLOCK)]
        self.radix.insert(ids, [self.pool.alloc() for _ in range(5)], {})
        for n in (BLOCK, 2 * BLOCK, 5 * BLOCK, 5 * BLOCK - 1):
            _, matched, _ = self.radix.match_prefix(ids[:n], require_states=False)
            self.assertEqual(matched, n // BLOCK * BLOCK)

    def test_a_text_query_with_mm_none_matches_a_text_branch(self) -> None:
        """``mm=None`` 是文本请求的常态，所以必须走纯键路径并命中。"""
        ids = [7 * i % 251 for i in range(4 * BLOCK)]
        self.radix.insert(ids, [self.pool.alloc() for _ in range(4)], {})
        _, matched, _ = self.radix.match_prefix(ids[:-1], mm=None)
        self.assertEqual(matched, 3 * BLOCK)

    def test_shared_text_prefix_between_an_image_and_a_text_request(self) -> None:
        """跨模态的复用必须双向都成立，否则混合负载会完全失去前缀共享。"""
        # 只要那些块里没有图像 token，缓存就得照样把图像请求写进去的文本块给
        # 文本请求用。
        text = [500 + i for i in range(2 * BLOCK)]
        ids, img_start = prompt_with_image(n_text=6, n_img=8, prefix_text=2 * BLOCK)
        ids = [500 + i for i in range(2 * BLOCK)] + ids[2 * BLOCK:]
        mm = make_mm("d", [(img_start, img_start + 8)], len(ids))
        self.radix.insert(ids, [self.pool.alloc() for _ in range(len(ids) // BLOCK)],
                          {}, mm=mm)
        _, matched, _ = self.radix.match_prefix(text, mm=None)
        self.assertEqual(matched, 2 * BLOCK)


# 两条预填充路径：插入 -> 匹配，且 mm 转发到模型
class PrefixEnginePathTest(unittest.TestCase):
    """单流预填充路径必须插入并匹配加盐键，且把 ``mm`` 转发出去。"""

    def test_insert_then_match_single_stream(self) -> None:
        """``mm`` 到达模型才是图像嵌入真正注入的时刻；缺了它，一次缓存命中会静默地把图丢掉。"""

        model = FakeModel()
        eng = make_engine(model)
        ids, img_start = prompt_with_image(n_text=9, n_img=12)
        mm = make_mm("image-A", [(img_start, img_start + 12)], len(ids))

        eng.prefill(ids, mm=mm)
        self.assertEqual(eng.last_matched, 0)
        self.assertTrue(model.calls, "the prefill never reached the model")
        self.assertIs(model.calls[-1][1], mm, "mm was not forwarded to model.prefill")
        self.assertEqual([t for call in model.calls for t in call[0]], ids)

        eng.prefill(ids, mm=mm)
        self.assertEqual(eng.last_matched, (len(ids) - 1) // BLOCK * BLOCK)
        self.assertEqual(eng.last_matched % BLOCK, 0)

        # id 相同、换成另一张图：一个块都不能认领
        model2 = FakeModel()
        eng2 = make_engine(model2)
        block = eng2.bm.block_size
        del block
        other = make_mm("image-B", [(img_start, img_start + 12)], len(ids))
        eng2.prefill(ids, mm=other)
        self.assertEqual(eng2.last_matched, 0)

    def test_text_only_path_is_unchanged(self) -> None:
        """文本预填充必须传 ``mm=None``，并保持旧的命中长度。"""
        model = FakeModel()
        eng = make_engine(model)
        ids = [300 + i for i in range(3 * BLOCK)]
        eng.prefill(ids)
        self.assertIsNone(model.calls[-1][1], "text prefill must pass mm=None")
        eng.prefill(ids)
        self.assertEqual(eng.last_matched, (len(ids) - 1) // BLOCK * BLOCK)
        _, matched, _ = eng.bm.match_prefix(ids[:-1])
        self.assertEqual(matched, eng.last_matched)

    def test_mixed_edge_stops_at_the_image_block(self) -> None:
        """短 prompt 会把文本块和图像块放进同一条边；检查必须恰好停在第一个图像块上，不早也不晚。"""
        # 一条边同时装着文本块和图像块（短 prompt 整条都落在同一个节点里）。
        # 这条边会拿到一张签名表，逐块检查必须恰好在第一个图像块处停住——早一块
        # 丢复用，晚一块交出错误的 KV。
        model = FakeModel()
        eng = make_engine(model)
        ids, img_start = prompt_with_image(n_text=4, n_img=12, prefix_text=2 * BLOCK + 5)
        mm_a = make_mm("image-A", [(img_start, img_start + 12)], len(ids))
        eng.prefill(ids, mm=mm_a)
        node = next(iter(eng.bm.radix.root.children.values()))
        self.assertIsNotNone(node.sigs, "a mixed edge must carry per-block signatures")

        # 同一张图：命中走到最后一个 token 之前那个整块
        eng.prefill(ids, mm=mm_a)
        full = (len(ids) - 1) // BLOCK * BLOCK
        self.assertEqual(eng.last_matched, full)

        # id 相同、换成另一张图：必须停在最后一个*文本*块，因为从图像开始往后
        # 都是另一次计算。
        text_blocks = img_start // BLOCK
        eng2 = make_engine(FakeModel())
        eng2.prefill(ids, mm=mm_a)
        eng2.prefill(ids, mm=make_mm("image-B", [(img_start, img_start + 12)], len(ids)))
        self.assertEqual(eng2.last_matched, text_blocks * BLOCK,
                         "a different image must only reuse the image-free head")


class ModelRunnerPathTest(unittest.TestCase):
    """批处理路径必须与单流路径一致：同一套盐、同一个边界、同样转发 ``mm``。"""

    def _insert(self, runner, seq) -> None:
        """经 runner（批处理路径）预填充一条序列。"""
        runner.prefill(seq)

    def test_insert_then_match_batched(self) -> None:
        """与单流路径的任何分歧都会最先在这里暴露，因为调度器只用这一条。"""
        model = FakeModel()
        model.enable_batching = lambda capacity: None      # 预填充用不到
        runner, bm = make_runner(model)
        ids, img_start = prompt_with_image(n_text=9, n_img=12)
        mm = make_mm("image-A", [(img_start, img_start + 12)], len(ids))

        first = Seq(ids, mm=mm)
        self._insert(runner, first)
        self.assertEqual(first.matched, 0)
        self.assertIs(model.calls[-1][1], mm, "mm was not forwarded to model.prefill")

        warm = Seq(ids, mm=mm)
        self._insert(runner, warm)
        self.assertEqual(warm.matched, (len(ids) - 1) // BLOCK * BLOCK)
        self.assertEqual(len(warm.block_table), (len(ids) + BLOCK - 1) // BLOCK)

        other = Seq(ids, mm=make_mm("image-B", [(img_start, img_start + 12)], len(ids)))
        self._insert(runner, other)
        self.assertEqual(other.matched, 0,
                         "a different image adopted the first image's KV")

    def test_text_only_batched_path_is_unchanged(self) -> None:
        """批处理文本路径行为不变：不往模型传 ``mm``，命中长度照旧。"""
        model = FakeModel()
        runner, bm = make_runner(model)
        ids = [700 + i for i in range(4 * BLOCK)]
        runner.prefill(Seq(ids))
        self.assertIsNone(model.calls[-1][1])
        warm = Seq(ids)
        runner.prefill(warm)
        self.assertEqual(warm.matched, (len(ids) - 1) // BLOCK * BLOCK)

    def test_group_prefill_matches_what_it_inserted(self) -> None:
        """插入用的键和匹配用的键对不上会静默毒化整棵树，所以在同一个 prompt 上把两者都断言一遍。"""
        # ``prefill_group`` 按 ``seq.mm`` 匹配、也按 ``seq.mm`` 插入；这里对不上
        # 正是会静默毒化树的那个 bug，所以要确认树里确实有匹配将去找的图像分支。
        model = FakeModel()
        runner, bm = make_runner(model)
        ids, img_start = prompt_with_image(n_text=9, n_img=12)
        mm = make_mm("image-A", [(img_start, img_start + 12)], len(ids))

        seq = Seq(ids, mm=mm)
        seq.block_table = [bm.pool.alloc() for _ in range(len(ids) // BLOCK)]
        seq.row = 0
        bm.insert(seq.token_ids, seq.block_table, {}, mm=seq.mm)
        root_key = tuple(ids[:BLOCK]) + (media_id("image-A", 0),)
        self.assertIn(root_key, bm.radix.root.children)
        _, matched, _ = bm.match_prefix(ids[:-1], require_states=False, mm=mm)
        self.assertEqual(matched, (len(ids) - 1) // BLOCK * BLOCK)


if __name__ == "__main__":
    unittest.main()
