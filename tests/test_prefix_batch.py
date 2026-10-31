"""批处理引擎里逐序列的前缀复用。

Module: ``coffin/tests``。把分页模型包进 ``llm_engine.LLMEngine``，再经
``config`` / ``sequence`` / ``scheduler`` 驱动它；它是唯一对调度器可见的
``matched`` 计数做断言的测试。

Constraints:
    * 热跑的参照是同一个引擎的冷跑，中间清空基数树；若换第二个模型做对照，
      16 GB 卡上就要多放一份 4B 权重。
    * 准入顺序本身是被测对象的地方用 ``add_request`` 加手工 ``step``，因为
      ``generate`` 拥有整个循环，会把准入时机藏起来。
    * ``tearDownClass`` 必须关掉 fixture：这个类独占一个引擎，泄漏的块池会让
      后面每个模块都占着卡。

基数树是逐序列的状态（块哈希，加上产出这些哈希的 delta-net 循环状态快照），
所以批处理不能把它关掉：一个请求被准入后，命中已缓存的最长块前缀，恢复该
边界处的循环状态，只预填充后缀，再把它算出的内容插进树里。

    python3 -m unittest discover -s tests -p "test_prefix_batch.py" -v
"""

from __future__ import annotations

import unittest

import support

from engine.config import EngineConfig
from engine.llm_engine import LLMEngine
from engine.sequence import SamplingParams

BLOCK = 64
PREFIX = [1000 + i for i in range(192)]          # 192 = 3 * BLOCK：共享前缀正好落在块边界上
TAIL_A = [4000 + i for i in range(16)]
TAIL_B = [5000 + i for i in range(24)]


def make_engine(model, *, max_seq: int, max_num_seqs: int = 4) -> LLMEngine:
    """建一个块大小与模型快照间隔对齐的引擎。

    ``snapshot_interval=BLOCK`` 不是调参选择：循环状态快照只在块边界上才有
    意义，两个常量必须相等，否则恢复出来的状态配不上被复用的那段 KV。
    """
    cfg = EngineConfig(model_dir=str(support.MODEL_DIR), max_seq=max_seq,
                       bench=False, quant=True, prefix_cache=True,
                       block_size=BLOCK, snapshot_interval=BLOCK,
                       max_num_seqs=max_num_seqs, num_blocks=1024)
    engine = LLMEngine(cfg, model)
    engine.build()
    return engine


@support.needs_model
class BatchedPrefixTest(unittest.TestCase):
    """批处理不能关掉前缀复用；复用到的前缀也不能改变答案。"""

    MAX_SEQ = 1024

    @classmethod
    def setUpClass(cls) -> None:
        """加载分页模型，并建好本类所有测试共用的引擎。"""
        cls.MODEL = support.ModelFixture.get(quant=True, max_seq=cls.MAX_SEQ,
                                             paged=True, bench=False)
        cls.ENGINE = make_engine(cls.MODEL, max_seq=cls.MAX_SEQ)

    @classmethod
    def tearDownClass(cls) -> None:
        """释放引擎和模型，让下一个模块从一张空卡开始。"""
        cls.ENGINE = None
        cls.MODEL = None
        support.ModelFixture.close()

    def _generate(self, prompt: list[int], n: int = 4) -> dict:
        """走共享引擎做贪心生成；温度 0 让不同次运行之间可以逐 token 比较。"""
        return self.ENGINE.generate(prompt, SamplingParams(max_tokens=n,
                                                           temperature=0.0))

    def test_cold_then_warm_identical_prompt(self) -> None:
        """第二次相同请求必须命中前缀，而输出仍要与冷跑逐 token 相同。"""
        engine = self.ENGINE
        engine.clear_prefix_cache()
        cold = self._generate(PREFIX + TAIL_A)
        self.assertEqual(cold["matched"], 0)
        warm = self._generate(PREFIX + TAIL_A)
        self.assertGreaterEqual(warm["matched"], BLOCK)
        self.assertEqual(warm["tokens"], cold["tokens"])

    def test_shared_system_prompt_with_different_tails(self) -> None:
        """共享前缀配不同后缀时，命中长度必须恰好等于前缀，且与它自己的冷跑答案一致。"""
        engine = self.ENGINE
        engine.clear_prefix_cache()
        cold_a = self._generate(PREFIX + TAIL_A)
        warm_b = self._generate(PREFIX + TAIL_B)
        self.assertEqual(warm_b["matched"], len(PREFIX))
        engine.clear_prefix_cache()
        cold_b = self._generate(PREFIX + TAIL_B)
        self.assertEqual(cold_b["matched"], 0)
        self.assertEqual(warm_b["tokens"], cold_b["tokens"])

    def test_grouped_prefill_populates_the_prefix_cache(self) -> None:
        """分块预填充走的是另一条代码路径，但它同样要留下冷跑命中所需的、按块对齐的状态快照。"""
        engine = self.ENGINE
        engine.clear_prefix_cache()
        # 两个请求一起准入会走一次分块预填充（grouped prefill）；这条路径同样
        # 得为下一个请求留下按块对齐的状态快照。
        a = engine.add_request(PREFIX + TAIL_A,
                               SamplingParams(max_tokens=3, temperature=0.0))
        b = engine.add_request(PREFIX + TAIL_B,
                               SamplingParams(max_tokens=3, temperature=0.0))
        out_a: list[int] = []
        while not all(s.is_finished for s in (a, b)):
            for seq, tok, _done in engine.step():
                if seq is a and tok is not None:
                    out_a.append(tok)
        self.assertEqual(a.matched, 0)
        warm = self._generate(PREFIX + TAIL_A, 3)
        self.assertGreaterEqual(warm["matched"], BLOCK)
        self.assertEqual(warm["tokens"], out_a)

    def test_prefix_stats_report_cached_blocks(self) -> None:
        """``prefix`` 这一块 stats 直接喂给服务端的缓存指标；它缺失或为空，运维就看不见树已经坏了。"""
        engine = self.ENGINE
        engine.clear_prefix_cache()
        self._generate(PREFIX + TAIL_A, 2)
        stats = engine.stats()
        self.assertIn("prefix", stats)
        self.assertGreater(stats["prefix"]["cached_blocks"], 0)
        self.assertGreater(stats["prefix"]["snapshots"], 0)
