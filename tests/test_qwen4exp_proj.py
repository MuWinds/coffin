"""``proj`` 的等价性与契约：bf16 逐位对齐、三个入口分派、量化挂载。

这一层守的是 task-4 的数值底线：把 ``nn.Linear`` 换成 :class:`~models.qwen4exp.proj.Proj`
之后，bf16 路径的结果必须与 ``F.linear`` **逐位**相同。这不是「误差很小就行」——
换的只是布局（``pack.as_kn`` 是一次纯转置），任何一位不同都说明多了一次舍入、一次
累加次序变化或者一次偷偷的反量化，而这三件都不会抛异常。

量化路径没有逐位对照物（``W4A8`` 是有损量化），所以这里守的是另外三件事：分派确实
落到了 ``linear_m`` / ``linear_prefill`` 上、反量化结果与 ``W4A8Weight.to`` 逐个相等、
以及占位权重被释放之后前向必须**报错**而不是静默退回 bf16。

Module: ``coffin/tests``。只用 CPU 与 ``reference`` 后端，不需要 DCU 或 checkpoint。

Constraints:
    * 逐位断言一律用 ``torch.equal``，不许写 ``allclose``：容差会把「多一次舍入」
      放过去，而那正是要抓的东西。
    * 量化用例自己造 ``W4A8Weight``（走 ``tools.quant_disk_w4a8`` 的真打包器），
      不读磁盘产物——本机没有量化目录。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

ENGINE = next((p for p in Path(__file__).resolve().parents
               if (p / "models" / "qwen4exp").is_dir()), Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ENGINE))

from backends import ops as bops  # noqa: E402
from models.qwen4exp import proj as proj_module  # noqa: E402
from models.qwen4exp.model import CausalLM  # noqa: E402
from models.qwen4exp.proj import (  # noqa: E402
    Proj,
    QuantizedEmbedding,
    proj_from_config,
    release_quantized_placeholders,
)
from models.qwen4exp.config import TextConfig  # noqa: E402
from models.qwen4exp.weights import DictSource, load_into, quantized_param_keys  # noqa: E402
from tools import quant_disk_w4a8  # noqa: E402
from w4a8_weights import W4A8Weight  # noqa: E402


def make_w4(source: torch.Tensor, group_size: int = 4) -> W4A8Weight:
    """把一份 ``[N, K]`` 浮点权重打包成 ``W4A8Weight``。

    Args:
        source: ``[N, K]`` fp32 权重。
        group_size: 沿 K 的分组宽度。

    Returns:
        :class:`W4A8Weight`，``(n, k)`` 与 ``source`` 一致。
    """
    packed, scales = quant_disk_w4a8.quantize_projection(source, group_size)
    return W4A8Weight(packed, scales, k=int(source.shape[1]), group_size=group_size)


class TestBf16Bitwise(unittest.TestCase):
    """bf16 权重路径：换布局之后必须与 ``F.linear`` 逐位相等。

    ``pack.as_kn`` 只是把 ``[N, K]`` 转置成 ``[K, N]``，算术一点没变，所以任何一位
    差异都来自调用方（多一次 cast、换了一种归约顺序、或者根本没走后端）。这条断言
    是整个 task-4 的数值底线：它一旦松成 ``allclose``，布局切换就会悄悄退化成
    「差不多对」。
    """

    def setUp(self) -> None:
        """造一份固定种子的 bf16 权重，作为 ``F.linear`` 对照的基准。"""
        torch.manual_seed(0)
        # 权重必须是 bf16：本族真实路径是 bf16 激活对 bf16 权重，而 ``F.linear`` 要求
        # 两个操作数同一 dtype。默认构造是 fp32，`.to(dtype)` 会把参数整体换掉。
        self.proj = Proj(64, 48).to(torch.bfloat16)
        self.weight = self.proj.weight.detach().clone()

    def assert_bitwise(self, x: torch.Tensor, note: str) -> None:
        """断言 ``Proj`` 与 ``F.linear`` 在 ``x`` 上逐位相同。

        Args:
            x: 激活，``[..., K]``，bf16。
            note: 失败信息里用来定位的形状说明。
        """
        got = self.proj(x)
        want = F.linear(x, self.weight)
        self.assertEqual(tuple(got.shape), tuple(want.shape), f"{note}: shape")
        self.assertTrue(torch.equal(got, want),
                        f"{note}: Proj diverged from F.linear by "
                        f"{(got - want).abs().max().item():.3e}")

    def test_decode_vector_is_bitwise_equal(self):
        """单 token（``[K]`` -> ``[N]``）走 ``linear`` 入口，逐位相等。"""
        self.assert_bitwise(torch.randn(64).to(torch.bfloat16), "[K]")

    def test_batched_leading_dims_are_bitwise_equal(self):
        """``[B, S, K]`` 这种前导维先摊平再还原，逐位相等。

        ``[B, S, H]`` 与 ``[B, S, 1]`` 两种形状在模型里都存在：前者是 mixer 与 MoE
        的常规输入，后者是 ``hc_count`` 维的门控投影。摊平之后它们都是 ``[M, K]``，
        结果与直接 ``F.linear`` 一致。
        """
        self.assert_bitwise(torch.randn(2, 3, 64).to(torch.bfloat16), "[2, 3, K]")
        self.assert_bitwise(torch.randn(2, 3, 1, 64).to(torch.bfloat16), "[2, 3, 1, K]")

    def test_small_m_and_prefill_rows_are_bitwise_equal(self):
        """``linear_m`` 档（2..8 行）与 ``linear_prefill`` 档（> 8 行）都逐位相等。

        两档之间只有「一次权重读取服务多少行」的差别，算术是同一个矩阵乘，所以它们
        必须给出与 ``F.linear`` 一样的位。参考后端对三个入口本来就是一个实现，这条
        断言真正钉住的是 :class:`Proj` 没有在这些档位上多做一次 cast 或一次转置。
        """
        for rows in (2, 8, 9, 40):
            with self.subTest(rows=rows):
                self.assert_bitwise(torch.randn(rows, 64).to(torch.bfloat16),
                                    f"[{rows}, K]")

    def test_non_contiguous_activation_is_bitwise_equal(self):
        """最后一维 stride 不是 1 的输入要先 ``contiguous()``，结果仍然逐位相等。

        模型里 ``[B, S, 1, H]`` 的门控输入来自 ``unflatten``，切片之后可能是非连续的；
        不先拷一份就 ``reshape`` 会拿到错位的行，而形状完全合法。
        """
        x = torch.randn(2, 6, 64).to(torch.bfloat16)[:, ::2, :]
        self.assertFalse(x.is_contiguous())
        self.assert_bitwise(x, "strided [2, 3, K]")

    def test_bias_branch_matches_nn_linear(self):
        """带 bias 的投影仍与 ``nn.Linear`` 对齐（本族真实配置不用它）。"""
        proj = Proj(64, 48, bias=True).to(torch.bfloat16)
        x = torch.randn(5, 64).to(torch.bfloat16)
        got = proj(x)
        want = F.linear(x, proj.weight, proj.bias)
        self.assertEqual(tuple(got.shape), tuple(want.shape))
        # ``nn.Linear`` 把 bias 折进 GEMM 的 epilogue，本模块加在结果上，两者相差一次
        # 舍入，所以这里只能按 bf16 的一个 ULP 比。
        self.assertLess((got - want).abs().max().item(),
                        torch.finfo(torch.bfloat16).eps * 2)

    def test_rejects_wrong_last_dim(self):
        """最后一维不是 ``K`` 时必须报错，而不是算出一个形状合法的垃圾。"""
        with self.assertRaises(ValueError):
            self.proj(torch.randn(3, 65).to(torch.bfloat16))
        # 0 维标量摊平之后会变成一个 ``[1, 1]`` 的矩阵乘，形状合法但语义全错。
        with self.assertRaises(ValueError):
            self.proj(torch.tensor(0.5, dtype=torch.bfloat16))


class TestDispatch(unittest.TestCase):
    """三个语义入口按行数分派：``linear`` / ``linear_m`` / ``linear_prefill``。

    分派规则是 task-4 明写的契约（``[K]`` / 小 ``[M, K]`` / 大 ``[C, K]``），而它决定
    预填充是走 GEMM 还是走逐行权重流式读取：真机上走错一次是 898 -> 1121 ms
    （0.80x）。参考后端三个入口是同一个实现，所以这里换成记录调用的替身——``Ops``
    是冻结的 dataclass，改不了字段，只能换掉 ``proj`` 模块看到的那个表。
    """

    def setUp(self) -> None:
        """把 ``proj`` 模块看到的算子表换成三个记录行数的替身。"""
        self.calls: list[tuple[str, int]] = []
        real = bops
        calls = self.calls

        def record(name: str, inner):
            """造一个记录行数的替身，然后转给真实入口。

            Args:
                name: 入口名。
                inner: 真实的 ``Ops`` 条目。

            Returns:
                ``(w, x) -> y`` 的替身。
            """

            def wrapper(w, x, *args, **kwargs):
                """记下入口名与行数，再算真实结果。

                Args:
                    w: 权重，原样转交。
                    x: 激活，``[K]`` 或 ``[M, K]``。
                    *args: 原样转交。
                    **kwargs: 原样转交。

                Returns:
                    真实入口的返回值。
                """
                rows = 1 if x.dim() == 1 else int(x.shape[0])
                calls.append((name, rows))
                return inner(w, x, *args, **kwargs)

            return wrapper

        fake = SimpleNamespace(
            linear=record("linear", real.linear),
            linear_m=record("linear_m", real.linear_m),
            linear_prefill=record("linear_prefill", real.linear_prefill),
        )
        patcher = patch.object(proj_module, "bops", fake)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_single_row_takes_the_vector_entry(self):
        """一行激活走 ``linear``，而且传进去的是 ``[K]`` 而不是 ``[1, K]``。"""
        proj = Proj(32, 16).to(torch.bfloat16)
        proj(torch.randn(32).to(torch.bfloat16))
        proj(torch.randn(1, 1, 32).to(torch.bfloat16))
        self.assertEqual(self.calls, [("linear", 1), ("linear", 1)])

    def test_small_batch_takes_the_linear_m_entry(self):
        """2..8 行走 ``linear_m``：一次权重读取服务全部行。"""
        proj = Proj(32, 16).to(torch.bfloat16)
        for rows in (2, 8):
            proj(torch.randn(rows, 32).to(torch.bfloat16))
        self.assertEqual(self.calls, [("linear_m", 2), ("linear_m", 8)])

    def test_prefill_takes_the_large_m_entry(self):
        """大于 8 行走 ``linear_prefill``，预填充的 128 行必须落在这里。

        这是真机实测的分界：把 128 行塞进逐 token 的路径会让 prefill 从 898 ms 变成
        1121 ms（0.80x），因为那条路径每个 token 都把整份权重读一遍。
        """
        proj = Proj(32, 16).to(torch.bfloat16)
        for rows in (9, 128):
            proj(torch.randn(rows, 32).to(torch.bfloat16))
        self.assertEqual(self.calls, [("linear_prefill", 9), ("linear_prefill", 128)])


class TestQuantizedAttach(unittest.TestCase):
    """量化挂载：占位释放、形状校验、以及「没挂上就报错」。"""

    def setUp(self) -> None:
        """造一份 ``[48, 64]`` 的 fp32 权重，并打包成 :class:`W4A8Weight`。"""
        torch.manual_seed(1)
        self.source = torch.randn(48, 64)
        self.q4 = make_w4(self.source)

    def test_set_quantized_replaces_the_placeholder(self):
        """挂上之后没有 bf16 参数，前向走 W4A8 分派，且与参考反量化一致。

        ``reference`` 后端的 ``linear`` 不认识 ``W4A8Weight``（它只认 ``QWeight``），
        所以量化路径的数值用后端的 ``linear`` 直接对 ``W4A8Weight.to`` 的结果比。
        """
        proj = Proj(64, 48, quantized=True)
        self.assertIsNone(proj.weight)
        self.assertTrue(proj.owns_quantized_weight)
        self.assertEqual(list(proj.state_dict()), [])
        proj.set_quantized(self.q4)
        x = torch.randn(3, 64).to(torch.bfloat16)
        got = proj(x)
        want = F.linear(x, self.q4.to(torch.bfloat16))
        self.assertTrue(torch.equal(got, want),
                        "quantized Proj disagrees with W4A8Weight.to")

    def test_forward_without_attach_raises(self):
        """占位已释放但还没挂量化权重时，前向必须报错。

        静默退回 bf16 会让这一步同时丢掉 6-20 倍的速度和 5.39 GiB 常驻，而形状、
        dtype 和数值都正常——只有真机账本能发现。所以宁可在第一次调用就炸。
        """
        proj = Proj(64, 48, quantized=True)
        with self.assertRaises(RuntimeError):
            proj(torch.randn(64).to(torch.bfloat16))

    def test_shape_mismatch_is_rejected(self):
        """形状与构造时声明不符必须拒绝：混用两个 checkpoint 的权重形状常常合法。"""
        proj = Proj(64, 48, quantized=True)
        with self.assertRaises(ValueError):
            proj.set_quantized(make_w4(torch.randn(32, 64)))
        with self.assertRaises(ValueError):
            proj.set_quantized(make_w4(torch.randn(48, 32)))

    def test_bias_and_quantized_is_rejected_at_construction(self):
        """``W4A8Weight`` 里没有 bias，两者同时成立必须在构造时就拒绝。"""
        with self.assertRaises(ValueError):
            Proj(64, 48, bias=True, quantized=True)

    def test_embedding_lookup_matches_w4_rows(self):
        """量化嵌入的查表结果必须等于 ``W4A8Weight.rows`` 的 bf16 版本。

        这条路径每查一个 token 解包一次（``rows`` 是 torch 实现），慢但正确。它真正
        钉住的是「没有偷偷退回 bf16 查表」：一旦退回，``Weight`` 上的 packed 缓冲区
        就不会被读，而这条断言仍然会过——所以下面还要单独断言权重确实是量化的。
        """
        table = make_w4(torch.randn(20, 32))
        embed = QuantizedEmbedding(20, 32, quantized=True)
        embed.set_quantized(table)
        ids = torch.tensor([[3, 0], [19, 7]])
        got = embed(ids)
        want = table.rows(ids.reshape(-1)).to(torch.bfloat16).reshape(2, 2, 32)
        self.assertEqual(tuple(got.shape), (2, 2, 32))
        self.assertTrue(torch.equal(got, want))
        self.assertIsNone(embed.weight)
        self.assertTrue(embed.owns_quantized_weight)

    def test_bf16_embedding_still_gathers(self):
        """非量化嵌入与 ``nn.Embedding`` 逐位相同（含 ``padding_idx`` 的普通 gather）。"""
        embed = QuantizedEmbedding(20, 32, padding_idx=0)
        ids = torch.tensor([[0, 5, 3]])
        self.assertTrue(torch.equal(embed(ids), F.embedding(ids, embed.weight, 0)))

    def test_release_placeholders_counts_modules(self):
        """``release_quantized_placeholders`` 摘掉每个投影的占位参数并报数。"""

        class Holder(torch.nn.Module):
            """两个投影加一个嵌入，模拟一层的几何。"""

            def __init__(self) -> None:
                """建两个投影加一个嵌入，覆盖三种会被摘掉占位的模块。"""
                super().__init__()
                self.a = Proj(16, 8)
                self.b = Proj(16, 4)
                self.embed = QuantizedEmbedding(10, 16)

        holder = Holder()
        self.assertEqual(release_quantized_placeholders(holder), 3)
        self.assertEqual(list(holder.state_dict()), [])
        self.assertTrue(all(module.owns_quantized_weight
                            for module in (holder.a, holder.b, holder.embed)))


class TestSetBf16(unittest.TestCase):
    """``set_bf16``：量化产物没覆盖的窄投影要能补回 bf16 并继续走 KN 路径。

    真机 artifact（作业 4492）的 526 条 ``q4`` 只命中 305 个稠密投影，169 个窄投影
    （``in_proj_a/b`` 的 N=48、``block_inject`` 的 N=4）不在产物里。它们不能退回
    ``F.linear``——那正是一步 245.2 ms 的来源——而要留着 bf16 走 ``Proj`` 的同一路：
    实测只换布局就 245.2 -> 21.2 ms，量化只再省 8.7 ms。
    """

    def test_bf16_fallback_is_bitwise_equal_to_linear(self):
        """补回的 bf16 权重走 KN 路径，单行与大 M 都与 ``F.linear`` 逐位相等。

        ``N=4``、``K=10240`` 就是 ``block_inject`` 的真实几何，也是旧路径最贵的一档。
        """
        torch.manual_seed(5)
        weight = torch.randn(4, 10240).to(torch.bfloat16)
        proj = Proj(10240, 4, quantized=True)
        self.assertIsNone(proj.weight)
        proj.set_bf16(weight)
        self.assertIs(proj.weight.dtype, torch.bfloat16)
        self.assertFalse(proj.owns_quantized_weight)
        for x in (torch.randn(10240).to(torch.bfloat16),
                  torch.randn(3, 10240).to(torch.bfloat16),
                  torch.randn(16, 10240).to(torch.bfloat16)):
            with self.subTest(shape=tuple(x.shape)):
                self.assertTrue(torch.equal(proj(x), F.linear(x, weight)))

    def test_shape_mismatch_and_double_write_are_rejected(self):
        """形状不符与「已挂量化权重又补 bf16」都必须拒绝。"""
        proj = Proj(64, 48, quantized=True)
        with self.assertRaises(ValueError):
            proj.set_bf16(torch.randn(32, 64))
        proj.set_quantized(make_w4(torch.randn(48, 64)))
        with self.assertRaises(RuntimeError):
            proj.set_bf16(torch.randn(48, 64))

    def test_released_projection_reports_both_missing_sources(self):
        """没挂量化、也没补 bf16 时，报错要点名两个方法，而不是只说一个。"""
        proj = Proj(64, 48, quantized=True)
        with self.assertRaises(RuntimeError) as caught:
            proj(torch.randn(64).to(torch.bfloat16))
        self.assertIn("set_quantized", str(caught.exception))
        self.assertIn("set_bf16", str(caught.exception))


class TestQuantizedAssembly(unittest.TestCase):
    """量化路径的装配约定：计划摘键、``to_empty``、再逐个 ``set_quantized``。

    这条链是 ``diag/qwen4/quant_attach.py`` 与真机 llamauncher 要照着写的那份顺序，
    所以它必须在纯 CPU 上有一条能跑的用例：顺序错了报出来的都是「模型参数没有源
    张量」「计划张量不是模型参数」这类离原因很远的话。
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = TextConfig(
            vocab_size=256, hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
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
            output_gate_type="sigmoid", rope_theta=10000.0, partial_rotary_factor=0.25,
            mrope_section=[3, 3, 2], eos_token_id=2, pad_token_id=0)

    def _assemble(self) -> tuple[CausalLM, dict]:
        """按真机顺序装一个量化模型：``meta`` 构造 → ``to_empty`` → ``load_into``。

        顺序是契约的一部分。``to_empty`` 只分配不初始化，所以**必须**紧跟一次
        ``load_into``：n-gram 的几个 buffer（层乘子、每头词表大小）不装进来就是
        内存残留，前向会在 ``torch.remainder`` 处以 ``ZeroDivisionError`` 炸掉，而
        那个报错完全指不到「你漏了装载」。

        Returns:
            ``(模型, 引擎键名的 bf16 状态字典)``；后者是量化权重的来源，真机上它是
            磁盘上的 ``q4`` 条目，这里用小配置现造一份。
        """
        cfg = self.cfg
        state = CausalLM(cfg).state_dict()
        with torch.device("meta"):
            model = CausalLM(cfg, quantized=True)
        provided = quantized_param_keys(model)
        model = model.to_empty(device="cpu").to(dtype=torch.bfloat16)

        def engine(name: str) -> str:
            """checkpoint 名 -> 引擎名，与 ``weights.engine_key`` 同一条规则。

            Args:
                name: checkpoint 里的完整键名。

            Returns:
                ``model.`` 前缀去掉文本塔那一段之后的键名。
            """
            prefix = "model.language_model."
            return "model." + name[len(prefix):] if name.startswith(prefix) else name

        keys = [f"model.language_model.{key[len('model.'):]}"
                for key in state if key.startswith("model.")] + ["lm_head.weight"]
        source = DictSource({key: state[engine(key)] for key in keys
                             if engine(key) not in provided})
        report = load_into(model, source, cfg)
        self.assertEqual(report["quantized"], len(provided))
        return model, state

    def test_plan_drops_exactly_the_provided_keys(self):
        """``load_into`` 在量化模型上只装非量化参数，且不报形状/覆盖错误。

        被摘掉的键数必须等于 ``quantized_param_keys`` 报的数：少了说明某个投影既没被
        装载也没被挂上（前向时才炸），多了说明计划里还留着 ``copy_`` 装不进去的键
        （``verify`` 会先报出来）。
        """
        torch.manual_seed(3)
        model, _state = self._assemble()
        provided = quantized_param_keys(model)
        self.assertEqual(len(provided),
                         sum(1 for module in model.modules()
                             if getattr(module, "owns_quantized_weight", False)))

    def test_partial_coverage_falls_back_to_bf16(self):
        """产物只覆盖一部分投影时：覆盖的挂量化，没覆盖的补回 bf16，前向仍跑通。

        这是真机 artifact 的形状（作业 4492）：526 条 ``q4`` 只命中 305 个稠密投影，
        169 个窄投影没有量化条目。两种来源必须能在同一个模型里共存——全部释放占位是
        对的（不释放就会在 ``to_empty`` 里实化 9 GiB），但没被覆盖的那些必须补回
        bf16，否则前向在第一个窄投影上就抛「没有权重」。
        """
        torch.manual_seed(6)
        model, state = self._assemble()
        provided = quantized_param_keys(model)
        # 只给前一半投影挂量化，模拟产物跳过了窄投影。
        narrowed = sorted(provided)[: len(provided) // 2]
        for name, module in model.named_modules():
            if name + ".weight" in narrowed:
                module.set_quantized(make_w4(state[name + ".weight"].float()))
        # 剩下的（含产物没覆盖的那些）补回 bf16，继续走同一套 ``Proj`` 路径。
        for name, module in model.named_modules():
            if not getattr(module, "owns_quantized_weight", False):
                continue
            if name + ".weight" in narrowed:
                continue
            module.set_bf16(state[name + ".weight"])
        # ``owns_quantized_weight`` 在挂上量化权重之后为真、在补回 bf16 之后为假，
        # 所以这两个计数加起来必须等于产物声明的键数，一个都不能漏。
        quantized_count = sum(1 for name in narrowed
                              if model.get_submodule(name[:-len(".weight")])
                              .owns_quantized_weight)
        bf16_count = sum(1 for module in model.modules()
                         if isinstance(module, (Proj, QuantizedEmbedding))
                         and not module.owns_quantized_weight)
        self.assertEqual(quantized_count, len(narrowed))
        self.assertEqual(quantized_count + bf16_count, len(provided))

        cfg = self.cfg
        model.eval()
        ple = model.model.layers[1].ple.ple_embedding
        table = torch.rand(ple.padded_vocab_size, ple.head_dim_per_ngram).to(torch.bfloat16)
        mask = torch.full((1, 1, 4, 4), torch.finfo(torch.float32).min)
        for index in range(4):
            mask[0, 0, index, :index + 1] = 0.0
        with torch.no_grad():
            logits, _ = model(
                input_ids=torch.randint(0, 256, (1, 4)), attention_mask=mask,
                conv_mask=torch.ones(1, 4, dtype=torch.bool),
                position_ids=torch.arange(4).view(1, 1, -1).expand(3, 1, -1),
                ple_tables={1: table, 2: table})
        self.assertEqual(tuple(logits.shape), (1, 4, cfg.vocab_size))
        self.assertTrue(torch.isfinite(logits).all())

    def test_forward_runs_after_every_projection_is_attached(self):
        """挂满量化权重之后整机前向跑得通，decode 一步也是。

        这一步钉的是「装配顺序」而不是数值：量化是有损的，和 bf16 逐位比对没有意义，
        但形状、dtype、以及 ``Proj`` 在无 bias 时不做多余 cast 这三件事都在这里。
        """
        torch.manual_seed(4)
        model, state = self._assemble()
        provided = quantized_param_keys(model)
        for name, module in model.named_modules():
            if not getattr(module, "owns_quantized_weight", False):
                continue
            # ``state`` 是引擎侧的 state dict，键名就是模块名加 ``.weight``，不必再
            # 走一遍 checkpoint 的前缀映射——那份映射由 ``weights.load_into`` 负责。
            module.set_quantized(make_w4(state[name + ".weight"].float()))
        attached = sum(1 for module in model.modules()
                       if getattr(module, "owns_quantized_weight", False))
        self.assertEqual(attached, len(provided))

        cfg = self.cfg
        model.eval()
        ple = model.model.layers[1].ple.ple_embedding
        table = torch.rand(ple.padded_vocab_size, ple.head_dim_per_ngram).to(torch.bfloat16)
        mask = torch.full((1, 1, 4, 4), torch.finfo(torch.float32).min)
        for index in range(4):
            mask[0, 0, index, :index + 1] = 0.0
        with torch.no_grad():
            logits, _ = model(
                input_ids=torch.randint(0, 256, (1, 4)), attention_mask=mask,
                conv_mask=torch.ones(1, 4, dtype=torch.bool),
                position_ids=torch.arange(4).view(1, 1, -1).expand(3, 1, -1),
                ple_tables={1: table, 2: table})
            step, _ = model(
                input_ids=torch.tensor([[7]]),
                attention_mask=torch.zeros(1, 1, 1, 5),
                conv_mask=torch.ones(1, 1, dtype=torch.bool),
                position_ids=torch.arange(4, 5).view(1, 1, -1).expand(3, 1, -1),
                ple_tables={1: table, 2: table})
        self.assertEqual(tuple(logits.shape), (1, 4, cfg.vocab_size))
        self.assertEqual(tuple(step.shape), (1, 1, cfg.vocab_size))
        self.assertTrue(torch.isfinite(logits).all(), "quantized prefill produced NaN/Inf")
        self.assertTrue(torch.isfinite(step).all(), "quantized decode produced NaN/Inf")


class TestFactory(unittest.TestCase):
    """``proj_from_config`` 只负责把「哪些站点带 bias」这条 checkpoint 事实收在一处。"""

    def test_bias_follows_attention_bias(self):
        """``bias=None`` 时跟 ``config.attention_bias``：QSA 的站点读它，其余显式关。"""
        cfg = TextConfig()
        self.assertEqual(proj_from_config(cfg, 16, 8, bias=None).bias, None)
        self.assertFalse(proj_from_config(cfg, 16, 8, bias=None).bias is not None)
        biased = TextConfig(attention_bias=True)
        self.assertIsNotNone(proj_from_config(biased, 16, 8, bias=None).bias)
        self.assertIsNone(proj_from_config(biased, 16, 8, bias=False).bias)


if __name__ == "__main__":
    unittest.main()
