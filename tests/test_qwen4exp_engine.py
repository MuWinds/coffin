"""qwen4exp 的引擎契约：单序列、eager、不分页。

守两条线。第一条是**数值**：引擎的 ``prefill`` 必须与直接调 ``CausalLM.forward``
逐位相同，``decode_step`` 必须与「每次把整段 token 重跑一遍」接得上，``reset`` 之后
重跑同一段 prompt 必须回到同一份 logits。第二条是**契约**：``engine_api`` 的族表里
两个族都在、分派认得 ``qwen4_exp_text``，而批处理/分页/投机/视觉这些本 lane 没有的
入口必须明确拒绝，而不是给出一个能跑但算错的实现。

两条线缺一不可：只测数值，一个悄悄接了分页的 ``enable_paging`` 会带着错 KV 通过；
只测契约，``prefill`` 走错掩码也能"有返回值"。

跑法（仓库根目录）::

    cd coffin
    PYTHONPATH=. python -m unittest tests.test_qwen4exp_engine -v

Module: ``coffin/tests``。依赖 ``models.qwen4exp`` 与 ``engine_api``；不加载
checkpoint、不碰 GPU、不 import ``support``（这个模块要在纯 CPU 机器上单独跑通）。

Constraints:
    * 全部用**小配置**：hidden 64、4 层、8 个专家。真实配置是 360 GB 权重，任何单机
      都放不下，而这里要验的是契约与状态搬运，不是数值规模。
    * ``prefill`` 的逐位比较只在整段一次前向时成立：分块会改变 GEMM 的 M，fp32 的
      末位跟着变。所以分块那条用例用 ULP 容差，并且把原因写在断言的注释里。
    * checkpoint 目录只写 ``config.json``（safetensors 不在本机，也不需要——权重走
      ``DictSource``）；临时目录建在引擎根下并在 ``tearDownClass`` 里删掉。
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import torch

#: 引擎根（``coffin/``）：``ops`` / ``models`` / ``engine_api`` 都是它的顶层包。
ENGINE = next((parent for parent in Path(__file__).resolve().parents
               if (parent / "models" / "qwen4exp").is_dir()),
              Path(__file__).resolve().parents[1])
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

import engine_api  # noqa: E402
from engine.config import EngineConfig  # noqa: E402
from engine.llm_engine import LLMEngine  # noqa: E402
from engine.sequence import SamplingParams  # noqa: E402
from models.qwen4exp.config import TextConfig  # noqa: E402
from models.qwen4exp.engine import Qwen4Engine, load_weights  # noqa: E402
from models.qwen4exp.model import CausalLM, causal_mask  # noqa: E402
from models.qwen4exp.weights import DictSource  # noqa: E402

#: 一个覆盖 3 层 GDN + 1 层 QSA 的完整周期的小配置；宽度 64 让 CPU 上一秒内跑完。
TINY = dict(
    vocab_size=256, hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
    num_key_value_heads=2, head_dim=32,
    linear_num_key_heads=2, linear_num_value_heads=6,
    linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
    num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32,
    shared_expert_intermediate_size=32, hc_count=4, hc_lowrank=16,
    ple_layer_ids=[2], ple_embed_dim=64, ple_conv_kernel_size=4,
    ngram_size=3, heads_per_ngram=8, ngram_vocab_size_base=1000,
    make_ngram_vocab_size_divisible_by=128,
    indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=16,
    indexer_budget=16, indexer_compress_ratio=4,
    output_gate_type="sigmoid", rope_theta=10000.0,
    partial_rotary_factor=0.25, mrope_section=[3, 3, 2], eos_token_id=2,
)

#: 5 个 prompt token 加 4 个解码 token：够长到让位置表、KV 和 n-gram 上下文都不
#: 停在第一格（长度 1 的序列会让「上下文没接上」这类 bug 整体隐身）。
PROMPT = [5, 7, 11, 13, 17]
STEPS = [19, 23, 29, 31]

#: 一次前向跑完整段 prompt 时窗口就够，解码用例另算。
MAX_SEQ = 64


def raw_config(with_ple: bool = True) -> dict:
    """把 :data:`TINY` 包成 checkpoint ``config.json`` 的形状。

    ``TextConfig`` 也从这里构造，所以「写到磁盘的配置」与「用来造参考模型的配置」是
    同一份字典——否则两个小配置会悄悄分叉，逐位比较就变成在比两个不同的模型。

    Args:
        with_ple: 是否保留 ``ple_layer_ids=[2]``（第 2 层挂 PLE）。

    Returns:
        可以直接 ``json.dumps`` 的字典。
    """
    fields = dict(TINY)
    if not with_ple:
        fields["ple_layer_ids"] = []
    return {"model_type": "qwen4_exp_text", "text_config": fields}


def tiny_config(with_ple: bool = True) -> TextConfig:
    """小配置；``with_ple`` 关掉 PLE 层，用来测「没有 PLE 时状态快照可用」。

    Args:
        with_ple: 是否保留 PLE 层。

    Returns:
        校验通过的 ``TextConfig``。
    """
    return TextConfig.from_dict(raw_config(with_ple)["text_config"])


def checkpoint_dict(model: CausalLM) -> dict[str, torch.Tensor]:
    """把引擎侧的 ``state_dict`` 改写成 checkpoint 的名字布局。

    参考 checkpoint 的文本塔在 ``model.language_model.`` 下，而 ``CausalLM`` 的文本塔
    叫 ``model.``——``weights.engine_key`` 反过来映射的就是这一步。名字若照抄引擎布局，
    装载器的映射表就形同虚设，那条路也就测不到了。

    Args:
        model: 参考模型。

    Returns:
        张量名到张量的字典，键是 checkpoint 布局。
    """
    out: dict[str, torch.Tensor] = {}
    for key, tensor in model.state_dict().items():
        if key.startswith("model."):
            out["model.language_model." + key[len("model."):]] = tensor.clone()
        else:
            out[key] = tensor.clone()
    return out


def reference_logits(model, ids, dtype=None) -> torch.Tensor:
    """参考实现的整段前向，``[1, n, vocab]``。

    掩码与 ``conv_mask`` 与引擎内部造的那份逐元素相同，所以两条路的结果可以在
    ``torch.equal`` 下直接比。``dtype`` 给了就先转模型——bf16 与 fp32 的容差结论
    完全不同，只在 fp32 上测过的用例说明不了 bf16 的逐位性。

    Args:
        model: 参考 ``CausalLM``。
        ids: 整段 token。
        dtype: 模型的 dtype；``None`` 表示不动它。

    Returns:
        整段 logits。
    """
    if dtype is not None:
        model = model.to(dtype)
    with torch.no_grad():
        out, _states = model(
            input_ids=torch.tensor([ids], dtype=torch.long),
            attention_mask=causal_mask(1, len(ids), 0),
            conv_mask=torch.ones(1, len(ids), dtype=torch.bool))
    return out


def fp32_tolerance(reference: torch.Tensor) -> float:
    """整段前向与单 token 前向在 fp32 下的合理末位差。

    ``M=1`` 的 GEMV 与 ``M=n`` 的 GEMM 对 K 维的累加次序不同，两者在 fp32 上本来
    就不会逐位相同（本机小配置实测 3.3e-7，logits 幅值约 0.93）。这个常数把「真的
    算错了」与「只是换了 GEMM 形状」分开；bf16 下舍入会把这点差抹平，那条路径直接
    要求逐位。

    Args:
        reference: 参考 logits，用它的幅值把容差归一化。

    Returns:
        绝对容差。
    """
    scale = max(float(reference.abs().max()), 1.0)
    return 8.0 * float(torch.finfo(torch.float32).eps) * scale


class RegistryTest(unittest.TestCase):
    """族表与分派：``engine_api`` 是唯一知道有哪些族的地方。"""

    def test_detect_returns_a_qwen4exp_key(self):
        """``model_type: qwen4_exp_text`` 必须落到 qwen4exp 的键上。

        分派靠 ``model_type`` 的子串匹配，键写成 ``qwen4_exp`` / ``qwen4exp`` 两个
        形式，是为了同时认下划线写法与紧凑写法。
        """
        key = engine_api.detect_architecture({"model_type": "qwen4_exp_text"})
        self.assertIn(key, ("qwen4_exp", "qwen4exp"))

    def test_both_families_are_registered(self):
        """qwen35 与 qwen4exp 必须同时在场；漏掉一个就等于砍掉一个族。"""
        keys = engine_api.registered()
        self.assertIn("qwen3_5", keys)
        self.assertTrue({"qwen4_exp", "qwen4exp"} & set(keys), keys)

    def test_the_registered_class_is_the_engine_module(self):
        """注册的类必须是本族的 ``engine.py`` 里那个，而不是参考模型或旧类。"""
        entry = engine_api._REGISTRY[
            engine_api.detect_architecture({"model_type": "qwen4_exp_text"})]
        self.assertEqual(entry.model.__module__, "models.qwen4exp.engine")
        self.assertIs(entry.model, Qwen4Engine)

    def test_qwen35_detection_is_unchanged(self):
        """新族进来不能把 qwen3_5 的 checkpoint 认领走：子串匹配没有优先级。"""
        self.assertEqual(
            engine_api.detect_architecture(
                {"architectures": ["Qwen3_5ForConditionalGeneration"],
                 "text_config": {"model_type": "qwen3_5_text"}}),
            "qwen3_5")


class InjectedModelTest(unittest.TestCase):
    """直接注入一个 ``CausalLM``（fp32），验引擎的四个入口与契约属性。

    注入路径不碰 checkpoint、不碰 ``safetensors``，所以它也是这条路上唯一能在纯
    CPU 开发机上真跑数值的部分。
    """

    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)
        cls.model = CausalLM(tiny_config()).eval()
        cls.engine = Qwen4Engine(cls.model, max_seq=MAX_SEQ)
        cls.engine.reset()

    def setUp(self) -> None:
        self.engine.reset()

    def test_prefill_matches_the_reference_forward_bitwise(self):
        """``prefill`` 的末位 logits 必须与 ``CausalLM.forward`` 逐位相同。

        这是整个接入的地基：掩码、位置表、conv_mask、PLE 的 token id 任何一处与参考
        实现不同，都会在这里变成末位差。``allclose`` 会放过它——层数一多，掩码里错
        一位的影响也就几个 ULP，而答案已经错了。
        """
        got = self.engine.prefill(PROMPT)
        want = reference_logits(self.model, PROMPT)[0, -1]
        self.assertEqual(tuple(got.shape), (TINY["vocab_size"],))
        self.assertTrue(torch.equal(got, want),
                        f"max abs diff {float((got - want).abs().max())}")

    def test_prefill_advances_the_position_counter(self):
        """``pos1`` 是引擎记账的权威，服务层与 runner 都读它。"""
        self.engine.prefill(PROMPT)
        self.assertEqual(int(self.engine.pos1), len(PROMPT))

    def test_decode_steps_match_a_full_rerun(self):
        """4 次 ``decode_step`` 必须接得上「整段重跑一遍前向」的末位。

        fp32 下这两条路不会逐位相同：单 token 是 ``M=1`` 的 GEMV，整段是 ``M=n``
        的 GEMM，K 维的累加次序不同（bf16 的舍入会把这个差抹平，见
        :class:`Bf16BitwiseTest`）。所以这里用「几个 ULP」判，而不是放宽到
        ``allclose`` 的默认值——后者对 logits 尺度来说太松，真错了也能过。
        """
        self.engine.prefill(PROMPT)
        got = torch.stack([self.engine.decode_step(token).clone() for token in STEPS])
        want = reference_logits(self.model, PROMPT + STEPS)[0, len(PROMPT):]
        self.assertEqual(tuple(got.shape), tuple(want.shape))
        diff = float((got - want).abs().max())
        self.assertLessEqual(diff, fp32_tolerance(want),
                             f"decode diverged from the full rerun: {diff}")

    def test_reset_clears_the_whole_context(self):
        """``reset`` 之后重跑同一段 prompt，logits 必须与首次逐位相同。

        ``ModelStates`` 里装的是 KV、GDN 的递推/卷积、PLE 的 n-gram 上下文与短卷积
        窗口、全位置表。漏清任何一条，第二次 prefill 就会读到上一次的尾巴，而那在
        贪心解码里通常只表现为「答案不太一样」，不会报错。
        """
        first = self.engine.prefill(PROMPT).clone()
        self.engine.reset()
        self.assertEqual(int(self.engine.pos1), 0)
        second = self.engine.prefill(PROMPT)
        self.assertTrue(torch.equal(first, second))

    def test_chunked_prefill_matches_one_forward(self):
        """分块推进与一次跑完的末位 logits 在几个 ULP 之内。

        分块改变的是 GEMM 的 M，所以差在 fp32 的末位；但状态必须真的接上——接不上的
        话差的是量级，不是 ULP。
        """
        want = reference_logits(self.model, PROMPT)[0, -1]
        got = self.engine.prefill(PROMPT, chunk=2)
        self.assertEqual(int(self.engine.pos1), len(PROMPT))
        diff = float((got - want).abs().max())
        self.assertLessEqual(diff, fp32_tolerance(want),
                             f"chunked prefill diverged: {diff}")

    def test_contract_attributes(self):
        """服务层与 runner 直接读的那几个属性，取值与语义都要对。"""
        engine = self.engine
        self.assertEqual(engine.max_seq, MAX_SEQ)
        self.assertEqual(tuple(engine.pos1.shape), (1,))
        self.assertEqual(int(engine.pos1), 0)
        # 没有验证图、没有 MTP 头：verify_c=0 与 mtp=None 是这两件事唯一诚实的表示。
        self.assertEqual(engine.verify_c, 0)
        self.assertIsNone(engine.mtp)
        self.assertEqual(engine.prefill_row, 0)
        self.assertEqual(engine.decode_capacity, 1)
        self.assertFalse(engine.graph_ready)
        # 服务层的 _build 会无条件读这两个视觉属性。
        self.assertIsNone(engine.vision)
        self.assertEqual(engine.vision_sd, {})

    def test_capture_is_a_no_op_and_reports_no_graph(self):
        """``capture`` 是空实现（平台不许用图），``graph_ready`` 因此恒假。"""
        self.engine.capture()
        self.engine.capture(warmup=0)
        self.assertFalse(self.engine.graph_ready)

    def test_decode_step_equals_logits(self):
        """两个解码入口必须是同一条 eager 前向：契约里它们只是启动方式不同。"""
        ids = [7, 9]
        self.engine.reset()
        self.engine.prefill(ids)
        a = self.engine.logits(31).clone()
        self.engine.reset()
        self.engine.prefill(ids)
        b = self.engine.decode_step(31).clone()
        self.assertTrue(torch.equal(a, b))

    def test_close_drops_the_model(self):
        """``close`` 之后重建前用它是编程错误，而不是一个静默的空前向。"""
        engine = Qwen4Engine(CausalLM(tiny_config()).eval(), max_seq=MAX_SEQ)
        engine.close()
        self.assertIsNone(engine.model)
        with self.assertRaises(RuntimeError):
            engine.prefill(PROMPT)
        # reset 仍然可用：服务层在失败路径上会顺手调它。
        engine.reset()


class Bf16BitwiseTest(unittest.TestCase):
    """bf16 下的逐位性：这是真机上的 dtype，也是契约里「末位容差」的实际口径。"""

    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(1)
        cls.model = CausalLM(tiny_config()).eval().to(torch.bfloat16)
        cls.engine = Qwen4Engine(cls.model, max_seq=MAX_SEQ)

    def test_prefill_and_decode_are_bitwise(self):
        """bf16 里 prefill 与 4 次 decode 都必须与参考前向逐位相同。

        舍入把 fp32 那点 GEMM 形状差抹掉了，所以这条路径可以要求 ``torch.equal``
        ——比契约要求的「末位容差」更严，而且它正是真机上判断「状态有没有接错」的
        唯一可用判据。
        """
        engine = self.engine
        engine.reset()
        got = engine.prefill(PROMPT)
        want = reference_logits(self.model, PROMPT)[0, -1]
        self.assertTrue(torch.equal(got, want))

        steps = torch.stack([engine.decode_step(token).clone() for token in STEPS])
        rerun = reference_logits(self.model, PROMPT + STEPS)[0, len(PROMPT):]
        self.assertTrue(torch.equal(steps, rerun),
                        f"max abs diff {float((steps - rerun).abs().max())}")


class RefusalTest(unittest.TestCase):
    """本 lane 没有的入口必须明确拒绝，而不是给一个能跑但算错的实现。"""

    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(2)
        cls.model = CausalLM(tiny_config()).eval()
        cls.engine = Qwen4Engine(cls.model, max_seq=MAX_SEQ)

    def setUp(self) -> None:
        self.engine.reset()

    def test_paged_construction_refuses(self):
        """``paged=True`` 要的是分页 KV 池，而这条路只有稠密 ``torch.cat`` 缓存。"""
        with self.assertRaises(NotImplementedError):
            Qwen4Engine(self.model, max_seq=MAX_SEQ, paged=True)

    def test_batching_and_paging_entries_refuse(self):
        """批处理的四个入口仍然拒绝：没有第二行、没有块表。"""
        with self.assertRaises(NotImplementedError):
            self.engine.enable_batching(2)
        with self.assertRaises(NotImplementedError):
            self.engine.set_rows([1], [0], [[0]])
        with self.assertRaises(NotImplementedError):
            self.engine.decode_batch_eager(1, 1)
        with self.assertRaises(NotImplementedError):
            self.engine.prefill_group([object()])

    def test_paging_attaches_a_pool(self):
        """分页池现在能挂上了：返回容量描述，重复挂载要拒绝。"""
        pool = self.engine.enable_paging(4, 8)
        self.assertEqual(pool["capacity"], 32)
        self.assertEqual(self.engine._paged["block_size"], 8)
        with self.assertRaises(RuntimeError):
            self.engine.enable_paging(4, 8)

    def test_enable_batching_one_is_a_no_op(self):
        """``capacity == 1`` 就是单流本身；runner 只在批处理时才传更大的值。"""
        self.engine.enable_batching(1)
        self.assertEqual(self.engine.prefill_row, 0)
        with self.assertRaises(ValueError):
            self.engine.enable_batching(0)

    def test_spec_entries_refuse(self):
        """投机解码三件套：没有 MTP 头，也没有可重放的验证图。"""
        with self.assertRaises(NotImplementedError):
            self.engine.capture_verify(4)
        with self.assertRaises(NotImplementedError):
            self.engine.spec_step(None, 3)
        with self.assertRaises(NotImplementedError):
            self.engine.spec_generate(PROMPT, 4)
        # 统计属性返回空字典而不是抛异常：日志与 ping 会无条件读它。
        self.assertEqual(self.engine.spec_stats, {})

    def test_image_entries_refuse(self):
        """带图 prompt 直接拒绝：把它当纯文本预填充会产出看不懂的结果。"""
        marker = object()
        with self.assertRaises(NotImplementedError):
            self.engine.prefill(PROMPT, mm=marker)
        with self.assertRaises(NotImplementedError):
            self.engine.prefill(PROMPT, spec_row=0)

    def test_snapshot_covers_a_ple_model(self):
        """带 PLE 的模型现在也能快照：整份状态里含 n-gram 历史与卷积窗口。

        契约从 ``{层: (循环状态, conv 状态)}`` 扩成整份 ``ModelStates`` 之后，
        PLE 的 ``ple_context`` / ``ple_conv_state`` 有地方放了，不再需要拒绝。
        """
        self.engine.prefill(PROMPT)
        snap = self.engine.snapshot_linear_states()
        self.assertIsNotNone(snap)
        carriers = [state for state in snap.layers if state.ple_context is not None]
        self.assertTrue(carriers, "PLE 层的 n-gram 历史没有被装进快照")

    def test_foreign_row_refuses(self):
        """单序列引擎只有第 0 行：任何按行的入口都不能假装有第二行。"""
        with self.assertRaises(NotImplementedError):
            self.engine.zero_linear_states(1)
        with self.assertRaises(NotImplementedError):
            self.engine.move_linear_states(0, 1)
        with self.assertRaises(NotImplementedError):
            self.engine.snapshot_linear_states(row=3)

    def test_prompt_longer_than_the_window_refuses(self):
        """越过 ``max_seq`` 要在调度器能处理的层拒绝，而不是写穿掩码。"""
        with self.assertRaises(ValueError):
            self.engine.prefill(list(range(MAX_SEQ + 1)))
        with self.assertRaises(ValueError):
            self.engine.prefill([])
        self.engine.prefill(PROMPT)
        self.engine.pos1.fill_(MAX_SEQ)
        with self.assertRaises(ValueError):
            self.engine.logits(3)


class SnapshotTest(unittest.TestCase):
    """没有 PLE 的配置上，线性注意力状态快照可以完整往返。"""

    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(3)
        cls.model = CausalLM(tiny_config(with_ple=False)).eval()
        cls.engine = Qwen4Engine(cls.model, max_seq=MAX_SEQ)

    def setUp(self) -> None:
        self.engine.reset()

    def test_snapshot_and_restore_round_trip(self):
        """打快照、走一步、还回来：重放同一步必须给出同一组 logits。

        快照现在是整份 ``ModelStates``（GDN 两半 + PLE 历史 + QSA 的 KV/索引池），
        所以判据从"字典里的张量相等"换成"装回之后重放同一步的 logits 逐位相同"
        ——那才是前缀复用真正依赖的性质，也顺带把「位置有没有一起回滚」测进去。
        """
        engine = self.engine
        engine.prefill(PROMPT)
        snap = engine.snapshot_linear_states()
        self.assertIsNotNone(snap)
        first = engine.decode_step(STEPS[0])
        engine.restore_linear_states(snap)
        again = engine.decode_step(STEPS[0])
        self.assertTrue(torch.equal(first, again),
                        "装回快照后重放同一步的 logits 不同")

    def test_foreign_layer_index_raises_key_error(self):
        """来自另一套架构的快照必须报 ``KeyError``，不能恢复一份残缺状态。"""
        self.engine.prefill(PROMPT)
        with self.assertRaises(KeyError):
            self.engine.restore_linear_states({99: (torch.zeros(1), torch.zeros(1))})

    def test_zero_linear_states_returns_the_context_to_empty(self):
        """``zero_linear_states`` 把这一行交出去：状态整份丢掉，位置归零。"""
        self.engine.prefill(PROMPT)
        self.engine.zero_linear_states(0)
        self.assertIsNone(self.engine._states)
        self.assertIsNone(self.engine.snapshot_linear_states())


class CheckpointBuildTest(unittest.TestCase):
    """从 checkpoint 建：走 ``weights.py`` 已有的装载器，不重写加载逻辑。

    本机没有 ``safetensors``（也没有 360 GB），所以张量用一个内存里的
    :class:`DictSource` 供货——它和线上 ``StreamingSource`` 实现同一个
    ``TensorSource`` 协议，``plan`` / ``load_into`` 一行都不用改；配置也直接由
    ``weights=`` 那个字典带进来，于是这条路径不需要磁盘上的 ``config.json``。
    """

    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(4)
        cls.model = CausalLM(tiny_config(with_ple=False)).eval()
        cls.reference = reference_logits(cls.model, PROMPT)[0, -1].clone()
        cls.checkpoint = checkpoint_dict(cls.model)
        cls.plain_config = raw_config(with_ple=False)
        cls.ple_config = raw_config(with_ple=True)

    def _weights(self, raw: dict) -> dict:
        """构造 ``load_weights`` 返回的那种字典（绕过 safetensors 的 header 扫描）。

        Args:
            raw: 要交给构造器的 ``config.json`` 内容。

        Returns:
            与 :func:`models.qwen4exp.engine.load_weights` 同形状的字典。
        """
        return {"source": DictSource(self.checkpoint), "config": raw,
                "model_dir": str(ENGINE)}

    def test_build_matches_the_reference_bitwise(self):
        """装载后的前向必须与建它用的那份权重逐位一致。

        这条同时验了装载器：映射表、形状校验、``copy_`` 任何一处漏一个张量，某个参数
        就停在 ``normal_(0.02)`` 的随机值上，logits 立刻偏——而形状全对。
        """
        engine = Qwen4Engine(ENGINE, max_seq=MAX_SEQ, bench=False,
                             weights=self._weights(self.plain_config))
        self.assertIsNotNone(engine.load_report)
        self.assertGreater(engine.load_report["loaded"], 0)
        engine.reset()
        got = engine.prefill(PROMPT)
        self.assertTrue(torch.equal(got, self.reference),
                        f"max abs diff {float((got - self.reference).abs().max())}")

    def test_ple_checkpoint_without_a_table_refuses(self):
        """有 PLE 层却没给 n-gram 表：构造期就必须拒绝。

        真实的表是 102.4 GB，``stream_ngram`` 必须打开；而流式模式下没有任何表可用
        时，前向会对着 ``None`` 查表。在这里拒绝，好过让服务起完之后在第一步解码上
        抛一个与根因无关的错。
        """
        with self.assertRaises(NotImplementedError):
            Qwen4Engine(ENGINE, max_seq=MAX_SEQ, bench=False,
                        weights=self._weights(self.ple_config))

    def test_load_weights_refuses_what_this_lane_does_not_implement(self):
        """量化与视觉的装载还没接：必须抛，不能静默按 bf16 读 360 GB。

        静默按 bf16 走的后果是先花几十分钟读盘，然后在 ``Tried to allocate`` 上
        失败——那个错看不出根因是「量化没接上」。这三条都在碰磁盘之前就抛，所以
        传一个不存在的目录也成立。
        """
        with self.assertRaises(NotImplementedError):
            load_weights(ENGINE, quant=True)
        with self.assertRaises(NotImplementedError):
            load_weights(ENGINE, vision=True)
        with self.assertRaises(NotImplementedError):
            load_weights(ENGINE, weight_dtype=torch.bfloat16)


class CheckpointDirectoryTest(unittest.TestCase):
    """真正的「目录进、模型出」：``engine_api.create_model`` 必须走到 ``Qwen4Engine``。

    需要一个只含 ``config.json`` 的临时目录，所以工作区不可写时整类跳过：那种环境里
    这条路径确实验不了，而跳过会把「没验」说出来，静默少跑一条不会。
    """

    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(6)
        cls.model = CausalLM(tiny_config(with_ple=False)).eval()
        cls.reference = reference_logits(cls.model, PROMPT)[0, -1].clone()
        cls.checkpoint = checkpoint_dict(cls.model)
        cls.root: Path | None = None
        try:
            cls.root = Path(tempfile.mkdtemp(prefix="q4exp-engine-", dir=str(ENGINE)))
            (cls.root / "config.json").write_text(
                json.dumps(raw_config(with_ple=False), ensure_ascii=False),
                encoding="utf-8")
        except OSError as exc:
            # 建目录成功但写不进去时（只读挂载、受限沙箱）目录会留下，而 setUpClass
            # 抛异常时 tearDownClass 不会被调用，所以这里自己收回去。
            if cls.root is not None:
                shutil.rmtree(cls.root, ignore_errors=True)
                cls.root = None
            raise unittest.SkipTest(
                f"the workspace is not writable ({exc}); this test needs a "
                f"config.json on disk under {ENGINE}") from None

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.root is not None:
            shutil.rmtree(cls.root, ignore_errors=True)

    def _weights(self) -> dict:
        """``load_weights`` 那种字典：源是内存里的，配置从磁盘读回来。

        Returns:
            可以直接交给构造器 / ``create_model`` 的字典。
        """
        return {"source": DictSource(self.checkpoint),
                "config": json.loads((self.root / "config.json").read_text("utf-8")),
                "model_dir": str(self.root)}

    def test_detect_architecture_reads_the_written_config(self):
        """分派读的是 checkpoint 自己的 ``config.json``，不是调用方给的提示。"""
        raw = json.loads((self.root / "config.json").read_text("utf-8"))
        self.assertIn(engine_api.detect_architecture(raw), ("qwen4_exp", "qwen4exp"))

    def test_build_from_the_directory_matches_the_reference(self):
        """``model_dir`` 一条路走到底是通的：配置从文件读、权重从源装。"""
        engine = Qwen4Engine(self.root, max_seq=MAX_SEQ, bench=False,
                             weights=self._weights())
        engine.reset()
        self.assertTrue(torch.equal(engine.prefill(PROMPT), self.reference))

    def test_create_model_dispatches_to_the_family(self):
        """``engine_api.create_model`` 必须按 ``config.json`` 分派到 ``Qwen4Engine``。

        这是「能被引擎调度的模型类」的字面判据：族表登记了还不够，分派要真的走到，
        而且走到的对象要能出 logits。
        """
        model = engine_api.create_model(self.root, max_seq=MAX_SEQ, bench=False,
                                        weights=self._weights())
        self.assertIsInstance(model, Qwen4Engine)
        self.assertEqual(model.max_seq, MAX_SEQ)
        model.reset()
        self.assertTrue(torch.equal(model.prefill(PROMPT), self.reference))


class EngineLoopTest(unittest.TestCase):
    """把模型交给 ``LLMEngine``：单序列、eager，产出必须与手写贪心循环一致。"""

    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(5)
        cls.model = CausalLM(tiny_config(with_ple=False)).eval()

    def _engine(self, graph: bool) -> LLMEngine:
        """搭一个单流、不分页、不批处理的引擎。

        Args:
            graph: ``cfg.graph``。``True`` 时 ``build`` 会调 ``capture()``，而它是空
                实现，所以这条分支必须与 eager 分支产出同样的 token。

        Returns:
            已经 ``build`` 过的引擎。
        """
        cfg = EngineConfig(model_dir=str(ENGINE),
                           max_seq=MAX_SEQ, bench=False, quant=False,
                           prefix_cache=False, block_size=64, max_num_seqs=1,
                           num_blocks=4, snapshot_interval=64, chunk=8,
                           prefill_quantum=8, graph=graph)
        engine = LLMEngine(cfg, Qwen4Engine(self.model, max_seq=MAX_SEQ))
        engine.build()
        return engine

    @staticmethod
    def _greedy_reference(engine_model, n: int) -> list[int]:
        """重构前的循环：reset、prefill，然后 argmax 接 decode_step。

        Args:
            engine_model: 直接驱动的那份引擎。
            n: 生成多少 token。

        Returns:
            贪心 token 序列。
        """
        engine_model.reset()
        logits = engine_model.prefill(PROMPT)
        out: list[int] = []
        for _ in range(n):
            token = int(logits.argmax())
            out.append(token)
            logits = engine_model.decode_step(token)
        return out

    def test_generate_matches_the_greedy_loop(self):
        """引擎调度不能在 token 流里引入任何差异。"""
        model = Qwen4Engine(self.model, max_seq=MAX_SEQ)
        engine = self._engine(graph=False)
        want = self._greedy_reference(model, 5)
        got = engine.generate(PROMPT, SamplingParams(max_tokens=5, temperature=0.0))
        self.assertEqual(got["tokens"], want)
        self.assertEqual(got["stop_reason"], "length")
        self.assertEqual(got["matched"], 0)

    def test_graph_mode_runs_the_same_eager_path(self):
        """``cfg.graph=True`` 也必须走 eager：``capture`` 是空实现、``graph_ready``
        恒假，所以 ``decode_step`` 与 ``logits`` 是同一条前向，产出不能变。"""
        model = Qwen4Engine(self.model, max_seq=MAX_SEQ)
        engine = self._engine(graph=True)
        want = self._greedy_reference(model, 4)
        got = engine.generate(PROMPT, SamplingParams(max_tokens=4, temperature=0.0))
        self.assertEqual(got["tokens"], want)
        self.assertFalse(engine.model.graph_ready)


if __name__ == "__main__":
    unittest.main()
