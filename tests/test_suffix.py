"""后缀草稿模型的契约：草稿必须是观察到的历史的真实延续。

Module: ``coffin/tests``。只 import ``ops.suffix``，因此在没有卡、没有
checkpoint、也没有 ``support`` 测试骨架的纯 CPU 主机上就能跑；``diag`` 里的
上卡探针负责对活的目标模型复查同一条规则。

Constraints:
    * 每条草稿都对着草稿模型观察过的历史校验，绝不对着事先写死的整段期望值。
      把答案硬编码的测试，在查找规则一改的瞬间就不再守「草稿是历史的真实切片」
      这条性质，而它正是校验方唯一依赖的东西。
    * 本文件不碰 GPU、不 import ``torch``、不做设备初始化：引擎要在主机侧做草稿
      模型的全部理由就是投机不能带来设备开销，而需要卡的测试没法在 CI 上守住
      这一点。
    * 断言只看 token 和长度，不看时间。接受率是由探针测量的质量指标；在这里加
      墙钟阈值，机器一忙就会闪断。
"""

from __future__ import annotations

import random
import unittest

from ops.suffix import SuffixDrafter


def _ids(words: list[str]) -> list[int]:
    """把词表映射成稳定的 token id，让测试读起来就是它所建模的那段文本。

    Args:
        words: 样例文本的 token，按顺序。

    Returns:
        每个词一个整数，按首次出现的顺序分配。不同词拿到不同 id，所以测试里
        命中只可能来自文本本身，不会来自编号巧合。
    """
    vocab: dict[str, int] = {}
    out: list[int] = []
    for word in words:
        if word not in vocab:
            vocab[word] = len(vocab) + 1
        out.append(vocab[word])
    return out


def _continuation(history: list[int], draft: list[int]) -> int:
    """找出延续正好等于 ``draft`` 的位置，找不到返回 ``-1``。

    Args:
        history: 草稿模型持有的完整历史。
        draft: 草稿模型返回的 token。

    Returns:
        结束位置 ``p``，满足 ``history[p+1 : p+1+len(draft)] == draft`` 且
        ``p <= len(history) - 2``；不存在这样的位置时返回 ``-1``。
    """
    width = len(draft)
    for p in range(len(history) - 1 - width + 1):
        if history[p + 1 : p + 1 + width] == draft:
            return p
    return -1


class SuffixDrafterTest(unittest.TestCase):
    """草稿必须能在交给草稿模型的历史里原样找到。"""

    def test_repeated_phrase_drafts_the_continuation(self):
        """只出现过一次的词组也必须能续出自己后面那段，这正是全部收益所在。

        这条失败说明草稿模型根本没找到更早的出现位置，后面所有关于选择规则的
        断言都会在空草稿上空洞地通过。
        """
        words = "the cat sat on the mat. the cat sat on the".split()
        history = _ids(words)
        vocab = dict(zip(words, history))

        drafter = SuffixDrafter(max_match=4, min_match=2)
        drafter.reset(history)

        # 尾部是 "cat sat on the"，此前在位置 1~4 出现过一次，
        # 那次出现之后紧接着的文本是 "mat. the cat"。
        self.assertEqual(drafter.draft(3), [vocab[w] for w in ("mat.", "the", "cat")])
        self.assertEqual(drafter.draft(1), [vocab["mat."]])

    def test_disjoint_history_drafts_nothing(self):
        """后缀在更早处没有出现过就返回空草稿，而不是猜一个。

        空草稿必须是诚实答案：没有草稿的投机步骤仍然产出普通的贪心 token，
        而编造的 token 会被拒绝，白白占掉一个校验槽位。
        """
        drafter = SuffixDrafter(max_match=4, min_match=2)
        drafter.reset([1, 2, 3, 4, 5])
        self.assertEqual(drafter.draft(3), [])

    def test_non_positive_k_returns_nothing(self):
        """``k <= 0`` 必须什么都不做，因为调用方就是靠传零来关掉投机。

        在这里返回历史剩余的全部内容，会让一个本该关闭的草稿悄悄变成无限长的
        草稿。
        """
        drafter = SuffixDrafter()
        drafter.reset(_ids("the cat sat on the mat. the cat sat on the".split()))
        self.assertEqual(drafter.draft(0), [])
        self.assertEqual(drafter.draft(-3), [])

    def test_empty_history_returns_nothing(self):
        """新请求没有历史，``draft`` 必须在做任何查找之前就作答。

        引擎在预填充后的第一个解码步就要草稿；这里抛的索引错误会直接弄死整个
        请求，而不是只多花一个慢步骤。
        """
        drafter = SuffixDrafter()
        drafter.reset([])
        self.assertEqual(drafter.draft(4), [])

    def test_draft_length_and_slice_are_bounded(self):
        """每条草稿最多 ``k`` 个 token，且是历史的一段原样切片。

        校验方肯接受一条草稿，前提是目标模型能拿自己的输出去核对它；一个从未在
        历史里出现过的草稿 token 会让被接受的 token 无法校验，所以切片性质要在
        整条流上查，而不是只看一个例子。
        """
        rng = random.Random(0)
        stream = [rng.randrange(6) for _ in range(120)]
        drafter = SuffixDrafter(max_match=4, min_match=2)
        for i, token in enumerate(stream):
            drafter.observe([token])
            if i < 8:
                continue
            for k in (1, 3, 6):
                draft = drafter.draft(k)
                self.assertLessEqual(len(draft), k, f"draft {draft} at step {i} exceeds k={k}")
                if draft:
                    self.assertGreaterEqual(
                        _continuation(stream[: i + 1], draft), 0,
                        f"draft {draft} at step {i} is not a continuation of the history",
                    )

    def test_observe_matches_reset(self):
        """增量建索引必须等于一次性重建，结果和索引规模都要一致。

        引擎每步追加一个 token，所以 ``observe`` 是热路径、``reset`` 是冷路径；
        两者一旦分叉，靠增量索引起草的请求就会和前缀缓存命中后重建索引的请求
        表现不同，而且不会报错。
        """
        rng = random.Random(7)
        stream = [rng.randrange(5) for _ in range(80)]
        incremental = SuffixDrafter(max_match=4, min_match=2)
        rebuilt = SuffixDrafter(max_match=4, min_match=2)
        for i, token in enumerate(stream):
            incremental.observe([token])
            rebuilt.reset(stream[: i + 1])
            self.assertEqual(incremental.index_size, rebuilt.index_size, f"index size at step {i}")
            for k in (1, 2, 5):
                self.assertEqual(incremental.draft(k), rebuilt.draft(k), f"draft k={k} at step {i}")

    def test_suffix_does_not_match_itself(self):
        """末尾的后缀不能算作它自己的一次出现。

        若算，``p`` 就会是历史的最后一个位置，「草稿」会变成空或者是越界内容再
        拼上一个编造的 token；这道保护正是让只出现一次的后缀被判为未命中。
        """
        # 后缀的两个 token 此前各自出现过，但从没以这个组合出现过，而这个组合
        # 本身就在历史末尾：唯一的一次出现就是后缀本身。
        drafter = SuffixDrafter(max_match=4, min_match=2)
        drafter.reset([8, 1, 2, 8, 9])
        self.assertEqual(drafter.draft(2), [])

    def test_longest_match_beats_follow_frequency(self):
        """4-gram 与 2-gram 都存在时，4-gram 的延续获胜。

        2-gram "3 4" 后面跟过两次 token 8、一次 token 7，所以频率优先的规则会
        起 8。4-gram "1 2 3 4" 后面跟的是 7，且只出现一次。优先更长的后缀，才能
        把一次命中变成每次校验多接受几个 token，而不是只中一个。
        """
        history = [1, 2, 3, 4, 7, 3, 4, 8, 3, 4, 8, 0, 1, 2, 3, 4]
        longest = SuffixDrafter(max_match=4, min_match=2)
        longest.reset(history)
        self.assertEqual(longest.draft(1), [7])

        # 同一段历史在 2-token 上限下就是纯频率视角；
        # 断言它起 8，说明这两条规则在这里确实给出了不同答案。
        shortest = SuffixDrafter(max_match=2, min_match=2)
        shortest.reset(history)
        self.assertEqual(shortest.draft(1), [8])

    def test_history_shorter_than_min_match_is_safe(self):
        """历史短于 ``min_match`` 时没有合法的查找窗口，且不许抛异常。

        预填充可以短到一个 token，草稿模型在任何匹配可能出现之前就会被调用；
        这里抛异常会让一个合法请求失败。
        """
        drafter = SuffixDrafter(max_match=4, min_match=2)
        drafter.reset([42])
        self.assertEqual(drafter.draft(3), [])
        drafter.observe([43])
        self.assertEqual(drafter.draft(3), [])

    def test_draft_does_not_mutate_history(self):
        """起草是只读操作，重复调用必须返回同一批 token。

        回滚时这一步会先调 ``draft`` 再调 ``observe``，顺序也可能反过来；一条会
        往历史里追加内容的草稿，会弄坏下一次校验对位置的认知。
        """
        history = _ids("the cat sat on the mat. the cat sat on the".split())
        drafter = SuffixDrafter()
        drafter.reset(history)
        fresh = SuffixDrafter()
        fresh.reset(history)
        first = drafter.draft(4)
        second = drafter.draft(4)
        self.assertEqual(first, second)
        self.assertEqual(drafter.index_size, fresh.index_size)


if __name__ == "__main__":
    unittest.main()
