"""引擎层回归：``LLMEngine`` 必须与它包装的那几条路径给出相同结果。

Module: ``coffin/tests``。测试套件里的最高层：它驱动
``llm_engine.LLMEngine`` 走过 ``config`` / ``sequence`` / ``scheduler`` /
``block_manager`` / ``model_runner``，再把结果与直接的模型循环对比，于是
调度器或 runner 的 bug 在这里就暴露，而不是拖到运行时入口。

Constraints:
    * 参照物是重构前的模型循环（``greedy_reference``），绝不是第二个引擎：
      两个引擎互相对上，可能共享同一个 bug 而一起通过。
    * 一律 temperature 0；一旦采样，两次运行就无法比较。
    * 批处理测试分词真实文本，而不用随手造的 id：合成 id 上 logits 接近均匀，
      来自不同 GEMM M 的几 ULP bf16 误差就能翻转 argmax，那不是这里要测的
      东西。
    * 每个类自己持有引擎，并在 ``tearDownClass`` 里关掉 fixture；这张卡同时
      装不下稠密模型和分页模型。

把 ``LLMEngine`` / ``Scheduler`` / ``ModelRunner`` / ``BlockManager`` 拆开
是否值得，只看一条：一个请求产出的 token 是否仍与直接跑模型循环完全一致。
这些测试把这条钉在两种 KV 模式下，另外覆盖前缀缓存的往返与流式迭代器。

    python3 -m unittest discover -s tests -p "test_engine.py" -v
"""

from __future__ import annotations

import functools
import unittest

# 副作用导入：设置 MKL_THREADING_LAYER，并把 coffin/ 加进 sys.path。
import support  # noqa: F401

from engine.config import EngineConfig
from engine.llm_engine import LLMEngine
from engine.sequence import SamplingParams

#: 词表内的任意 id 都行：模型只看到嵌入，看不到文本。长度取两个块，
#: 基数树才有块对齐的前缀可以匹配。
PROMPT = [1000 + i for i in range(128)]
#: 第二个不相交的 prompt：前缀测试彼此不能看到对方的缓存。
PROMPT_ALT = [2000 + i for i in range(128)]
TAIL = [23, 24, 25, 26, 27]


@functools.lru_cache(maxsize=1)
def _tokenizer():
    """分词器只加载一次，批处理测试的每个 prompt 都复用它。"""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(support.MODEL_DIR))


def _ids(text: str) -> list[int]:
    return _tokenizer().encode(text)


def make_engine(model, *, prefix_cache: bool, max_seq: int,
                snapshot_interval: int = 512, max_num_seqs: int = 1) -> LLMEngine:
    """围绕已加载的模型搭一个引擎。

    ``num_blocks`` 按 ``max_seq`` 推算，测试不会因为 KV 耗尽而以错误的原因
    通过；下限 64 让最小的用例也放得下。
    """
    cfg = EngineConfig(model_dir=str(support.MODEL_DIR), max_seq=max_seq,
                       bench=False, quant=True, prefix_cache=prefix_cache,
                       block_size=64, snapshot_interval=snapshot_interval,
                       max_num_seqs=max_num_seqs,
                       num_blocks=max(64, (max_seq + 63) // 64))
    engine = LLMEngine(cfg, model)
    engine.build()
    return engine


def greedy_reference(model, prompt: list[int], n: int) -> list[int]:
    """重构前的循环：reset、prefill，然后 argmax 加 decode_step。

    这是所有引擎测试的对照实现：引擎正是从这段循环重构出来的，所以它必须保持
    独立于调度器和 runner 代码。
    """
    model.reset()
    logits = model.prefill(prompt)
    out: list[int] = []
    for _ in range(n):
        tok = int(logits.argmax())
        out.append(tok)
        logits = model.decode_step(tok)
    return out


class DenseEngineTest(support.ModelTestCase):
    """关掉前缀缓存：引擎不能在模型之上再引入任何差异。"""

    MAX_SEQ = 2048
    PAGED = False

    def test_generate_matches_the_reference_loop(self) -> None:
        """缓存关掉时，引擎的调度开销不得在 token 流里露出来。"""
        engine = make_engine(self.model, prefix_cache=False, max_seq=self.MAX_SEQ)
        ref = greedy_reference(self.model, PROMPT, 6)
        res = engine.generate(PROMPT, SamplingParams(max_tokens=6, temperature=0.0))
        self.assertEqual(res["tokens"], ref)
        self.assertEqual(res["matched"], 0)
        self.assertEqual(engine.stats()["generated_tokens"], 6)

    def test_streaming_yields_the_same_tokens_as_generate(self) -> None:
        """``stream`` 与 ``generate`` 共用一条循环；两边分叉就说明有 token 在步边界上被丢掉或重复输出。"""
        engine = make_engine(self.model, prefix_cache=False, max_seq=self.MAX_SEQ)
        res = engine.generate(PROMPT, SamplingParams(max_tokens=5, temperature=0.0))

        seq = engine.add_request(PROMPT, SamplingParams(max_tokens=5, temperature=0.0))
        streamed = list(engine.stream(seq))
        self.assertEqual(streamed, res["tokens"])

    def test_max_tokens_and_eos_stop_the_request(self) -> None:
        """两个停止条件都是协议的一部分：长度上限约束延迟，EOS 上限把 EOS token 挡在回复之外。"""
        engine = make_engine(self.model, prefix_cache=False, max_seq=self.MAX_SEQ)
        res = engine.generate(PROMPT, SamplingParams(max_tokens=3, temperature=0.0))
        self.assertEqual(len(res["tokens"]), 3)
        self.assertEqual(res["stop_reason"], "length")

        eos = res["tokens"][2]
        expected: list[int] = []
        for tok in res["tokens"]:
            if tok == eos:
                break
            expected.append(tok)
        seq = engine.add_request(PROMPT,
                                 SamplingParams(max_tokens=8, temperature=0.0, eos={eos}))
        self.assertEqual(list(engine.stream(seq)), expected)
        self.assertEqual(seq.stop_reason, "eos")


class GraphOffTest(unittest.TestCase):
    """后端无法捕获计算图时 ``build()`` 该做什么。

    用替身而不是 checkpoint：被测行为是 ``build()`` 在碰到任何权重之前走的那条
    分支，真模型只会让测试更慢，还会绑死平台。
    """

    class _Model:
        """``Qwen35`` 里 ``build()`` 在关图分支上会读到的那几部分。"""

        def __init__(self, has_mtp: bool) -> None:
            """记下这个替身有没有 MTP 头。

            Args:
                has_mtp: ``mtp`` 属性的取值；草稿模型开关只在头存在时才有意义。
            """
            self.mtp = object() if has_mtp else None
            self.mtp_prefill = True
            self.reset_calls = 0

        def reset(self) -> None:
            """数 reset 次数，测试靠它证明分支真的走到了。"""
            self.reset_calls += 1

    def _engine(self, graph: bool, batching: bool, spec_k: int, has_mtp: bool) -> LLMEngine:
        """直接设好分支需要的输入，搭一个引擎。

        Args:
            graph: ``EngineConfig.graph``。
            batching: 引擎是否按批处理配置。
            spec_k: 每个投机步的草稿数。
            has_mtp: 替身模型有没有 MTP 头。

        Returns:
            一个尚未 build、包着替身模型的引擎。
        """
        # ``prefix_cache=False`` 加单序列让构造函数不去碰块管理器，那是构造里
        # 唯一需要真的分页模型的东西。
        cfg = EngineConfig(model_dir="unused", max_seq=64, bench=False, graph=graph,
                           spec_k=spec_k, prefix_cache=False, max_num_seqs=1)
        engine = LLMEngine(cfg, self._Model(has_mtp))
        engine.batching = batching
        return engine

    def test_graph_off_disables_the_drafter_prefill(self):
        """没有图捕获就跑不了投机，预填充写下的草稿 KV 全是白做的工作。

        这份 KV 是整个 prompt 上的一层全注意力加五次投影，在测它的那张卡上
        n=2048 时占 962.5 ms 预填充里的 214.8 ms。关图分支里没有别的东西会
        发现这件事，所以开关必须在这里清掉。
        """
        engine = self._engine(graph=False, batching=False, spec_k=0, has_mtp=True)
        engine.build()
        self.assertFalse(engine.model.mtp_prefill)
        self.assertEqual(engine.model.reset_calls, 1)

    def test_graph_off_without_an_mtp_head_is_still_fine(self):
        """没有草稿模型的 checkpoint，不能要求它给出草稿模型。"""
        engine = self._engine(graph=False, batching=False, spec_k=0, has_mtp=False)
        engine.build()
        self.assertEqual(engine.model.reset_calls, 1)

    def test_graph_off_still_rejects_speculation(self):
        """清掉草稿模型开关，不能连带放松那个抓出真实错配的检查。"""
        engine = self._engine(graph=False, batching=False, spec_k=2, has_mtp=True)
        with self.assertRaises(RuntimeError):
            engine.build()


@support.needs_model
class PagedEngineTest(unittest.TestCase):
    """分页 KV 加基数前缀缓存，整个类共用一个引擎。"""

    MAX_SEQ = 1024
    ENGINE: LLMEngine | None = None
    MODEL = None

    @classmethod
    def setUpClass(cls) -> None:
        """整个类只建一个引擎：build 要捕获计算图，那是耗时的大头。"""
        cls.MODEL = support.ModelFixture.get(quant=True, max_seq=cls.MAX_SEQ,
                                             paged=True, bench=False)
        # 引擎会挂上分页池，所以池必须先于任何 reset 存在。
        # ``snapshot_interval=64`` 在每个块边界留一份循环状态快照，正是它让
        # 127 个 token 的前缀能在 64 处命中。
        cls.ENGINE = make_engine(cls.MODEL, prefix_cache=True, max_seq=cls.MAX_SEQ,
                                 snapshot_interval=64)

    @classmethod
    def tearDownClass(cls) -> None:
        """先放掉引擎再关模型，让分页池先释放。"""
        cls.ENGINE = None
        cls.MODEL = None
        support.ModelFixture.close()

    def test_engine_matches_the_prefix_engine_api(self) -> None:
        """引擎产出的 token 必须来自它包装的裸前缀引擎，而不是自己另算一版。"""
        engine = self.ENGINE
        pc = engine.pc
        logits = pc.prefill(PROMPT_ALT)
        ref = list(pc.decode_iter(logits, 4, set(), 0.0))
        pc.release()
        res = engine.generate(PROMPT_ALT, SamplingParams(max_tokens=4, temperature=0.0))
        self.assertEqual(res["tokens"], ref)

    def test_second_prompt_hits_the_prefix(self) -> None:
        """命中必须按块对齐，并以真正校验过的长度为界，否则复用的块会带上没见过的 KV。"""
        engine = self.ENGINE
        first = engine.generate(PROMPT, SamplingParams(max_tokens=3, temperature=0.0))
        second = engine.generate(PROMPT, SamplingParams(max_tokens=3, temperature=0.0))
        self.assertEqual(first["tokens"], second["tokens"])
        # 可匹配的前缀是 ``tokens[:-1]``，所以只数得整的 64 token 块；
        # 更长的 prompt 多复用一个块。
        self.assertEqual(second["matched"], (len(PROMPT) - 1) // 64 * 64)
        third = engine.generate(PROMPT + TAIL,
                                SamplingParams(max_tokens=3, temperature=0.0))
        self.assertEqual(third["matched"],
                         (len(PROMPT + TAIL) - 1) // 64 * 64)


@support.needs_model
class BatchedEngineTest(unittest.TestCase):
    """静态批：并发请求各自都必须等于单独解码的结果。

    prompt 用分词后的真实文本，而不是随手造的 id：合成 id 上 logits 接近均匀，
    来自不同 GEMM M 的几 ULP bf16 误差就能翻转 argmax，那不是这个测试要测的
    东西。
    """

    MAX_SEQ = 1024
    TEXTS = [
        "The quick brown fox jumps over the lazy dog. " * 2,
        "In 1789 the French Revolution began, and the century that followed "
        "reshaped European politics. ",
        "def fibonacci(n):\n    return n if n < 2 else fibonacci(n - 1) + "
        "fibonacci(n - 2)\n",
    ]

    @classmethod
    def setUpClass(cls) -> None:
        """文本只分词一次，再建一个四序列、不带前缀缓存的引擎。"""
        cls.PROMPTS = [list(_ids(text)) for text in cls.TEXTS]
        cls.MODEL = support.ModelFixture.get(quant=True, max_seq=cls.MAX_SEQ,
                                             paged=True, bench=False)
        cls.ENGINE = make_engine(cls.MODEL, prefix_cache=False, max_seq=cls.MAX_SEQ,
                                 max_num_seqs=4)

    @classmethod
    def tearDownClass(cls) -> None:
        """先放掉引擎再关模型，让分页池先释放。"""
        cls.ENGINE = None
        cls.MODEL = None
        support.ModelFixture.close()

    def _solo(self, prompt: list[int], budget: int) -> list[int]:
        """单独把一条 prompt 走完引擎：逐行的参照。"""
        engine = self.ENGINE
        seq = engine.add_request(prompt, SamplingParams(max_tokens=budget,
                                                        temperature=0.0))
        return list(engine.stream(seq))

    def _concurrent(self, budgets: list[int]) -> list[list[int]]:
        """一次性准入全部 prompt，逐步驱动引擎直到每条序列结束。"""
        engine = self.ENGINE
        seqs = [engine.add_request(p, SamplingParams(max_tokens=b, temperature=0.0))
                for p, b in zip(self.PROMPTS, budgets)]
        out: list[list[int]] = [[] for _ in seqs]
        while not all(s.is_finished for s in seqs):
            for seq, tok, _done in engine.step():
                for i, target in enumerate(seqs):
                    if seq is target and tok is not None:
                        out[i].append(tok)
        return out

    def test_concurrent_requests_match_their_solo_decode(self) -> None:
        """批处理不得改变任何请求的 token，包括某一行提前结束的时候。"""
        budgets = [5, 2, 4]          # 长度不同，运行中途会空出一行
        refs = [self._solo(p, b) for p, b in zip(self.PROMPTS, budgets)]
        self.assertEqual(self._concurrent(budgets), refs)

    def test_grouped_prefill_matches_solo_prefill(self) -> None:
        """分组预填充改变了 GEMM 形状，这个测试守住它仍然落在同一串 token 上。"""
        budgets = [4, 4, 4]
        refs = [self._solo(p, b) for p, b in zip(self.PROMPTS, budgets)]
        self.assertEqual(self._concurrent(budgets), refs)
