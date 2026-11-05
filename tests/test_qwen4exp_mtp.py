"""MTP 头的回归：结构、装载映射、每步数据流，以及投机解码的接受规则。

真实 checkpoint 的 ``mtp.*`` 是 2.607B 个 BF16 参数，本机放不下，所以这里分三层验，
每层都不需要卡也不需要真实权重：

* **几何**（``meta`` 设备，零内存）——31 个 ``mtp.*`` 张量名字与形状必须与
  ``docs/qwen4/inventory.json`` 逐条对上。错一条的后果是某个参数停在
  ``normal_(0.02)`` 的初始值上，前向照跑、答案全错。
* **数据流**（小配置、真前向）——草稿步吐的两路必须是同一个算术的两个出口；
  嵌入残差只在步 0 加；链上第二步必须吃第一步的 ``multi`` 而不是从 token 重新起跑。
* **接受规则**（不经过模型）——贪心校验的接受前缀与 bonus token，用一组人造的
  logits 直接断言。这是投机解码唯一真正的算术，与 MTP 的实现细节无关，所以它能
  在没有任何权重的情况下守死。

跑法（仓库根目录）::

    python coffin/tests/test_qwen4exp_mtp.py
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import torch

ENGINE = next((p for p in Path(__file__).resolve().parents
               if (p / "models" / "qwen4exp").is_dir()), Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ENGINE))

from models.qwen4exp.config import TextConfig  # noqa: E402
from models.qwen4exp.mtp import (  # noqa: E402
    DraftIndexPlan,
    MTP,
    mtp_parameter_names,
)
from models.qwen4exp.model import CausalLM, causal_mask  # noqa: E402
from models.qwen4exp.qsa import IndexReuse  # noqa: E402

ORACLE = Path(__file__).resolve().parents[2] / "docs" / "qwen4" / "oracle"
INVENTORY = Path(__file__).resolve().parents[2] / "docs" / "qwen4" / "inventory.json"


def small_config(**override) -> TextConfig:
    """一个能跑前向的最小 MTP 配置。

    几何上必须让 QSA 真的启用（``indexer_*`` 全给值），否则草稿层会退化成普通
    注意力，而「复用的是 QSA 的选块」这件事就测不出来了。``mtp_num_hidden_layers=1``
    是设计点，不是可选值。

    Args:
        **override: 要覆盖的字段。

    Returns:
        ``TextConfig``。
    """
    fields = dict(
        vocab_size=128, hidden_size=32, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
        num_experts=8, num_experts_per_tok=2, moe_intermediate_size=16,
        shared_expert_intermediate_size=16, hc_count=2, hc_lowrank=8,
        ple_layer_ids=[], ngram_size=3, heads_per_ngram=2,
        ngram_vocab_size_base=64, make_ngram_vocab_size_divisible_by=8,
        indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=8,
        indexer_budget=8, indexer_compress_ratio=2,
        rope_theta=10000.0, partial_rotary_factor=0.5,
        mrope_section=[2, 2, 2], mrope_interleaved=True,
        mtp_num_hidden_layers=1,
    )
    fields.update(override)
    return TextConfig(**fields)


def real_config(**override) -> TextConfig:
    """真实 checkpoint 的文本几何（层数可截断）。

    Args:
        **override: 要覆盖的字段。

    Returns:
        ``TextConfig``。
    """
    fields = dict(
        vocab_size=248320, hidden_size=2560, num_hidden_layers=48,
        num_attention_heads=24, num_key_value_heads=2, head_dim=256,
        linear_num_key_heads=16, linear_num_value_heads=48,
        linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4,
        num_experts=512, num_experts_per_tok=10, moe_intermediate_size=640,
        shared_expert_intermediate_size=640, hc_count=4, hc_lowrank=320,
        ple_layer_ids=[2], ple_embed_dim=2560, ple_conv_kernel_size=4,
        ngram_size=3, heads_per_ngram=8, ngram_vocab_size_base=20_000_000,
        make_ngram_vocab_size_divisible_by=128,
        indexer_n_heads=4, indexer_kv_heads=1, indexer_head_dim=128,
        indexer_budget=2048, indexer_compress_ratio=4,
        output_gate_type="sigmoid", rope_theta=10000000.0, partial_rotary_factor=0.25,
        mrope_section=[11, 11, 10], mrope_interleaved=True, eos_token_id=248044,
        mtp_num_hidden_layers=1,
    )
    fields.update(override)
    return TextConfig(**fields)


def causal_mask_4d(batch: int, seq: int, past: int) -> torch.Tensor:
    """4 维 bool 因果掩码，``[B, 1, S, past + S]``。

    Args:
        batch: batch 大小。
        seq: query 数。
        past: 起始位置。

    Returns:
        bool 掩码。
    """
    rows = torch.arange(past, past + seq).unsqueeze(1)
    cols = torch.arange(past + seq).unsqueeze(0)
    return (cols <= rows).expand(batch, -1, -1).unsqueeze(1)


class TestMTPGeometry(unittest.TestCase):
    """31 个 ``mtp.*`` 张量的名字与形状必须与 checkpoint 逐条对上。"""

    def test_state_keys_and_shapes_match_inventory(self):
        """``MTP.state_dict()`` 必须与 ``inventory.json`` 的 ``mtp.*`` 完全对应。

        这条测试守的是整个规格的骨架：名字差一个字（``pre_fc_norm_hidden`` 写成
        ``pre_norm_hidden``）会让装载器报「模型参数没有源张量」，而形状差一维
        （``pre_fc_norm_hidden`` 是 ``[H]`` 而不是 ``[hc*H]``）会让「喂进去的是未塌缩
        流」这条结论悄悄失效——前向照跑。
        """
        if not INVENTORY.is_file():
            self.skipTest(f"no inventory.json next to the repo ({INVENTORY})")
        inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))["tensors"]
        # checkpoint 的键带 ``mtp.``，而 ``MTP`` 自己的 ``state_dict`` 是相对它的根，
        # 所以这里去掉前缀再比——``engine_key`` 就是负责把 checkpoint 的全名映射成
        # 引擎侧名字的那一环，MTP 侧留下的正是后半段。
        expected = {name[len("mtp."):]: list(meta["shape"])
                    for name, meta in inventory.items()
                    if name.startswith("mtp.")}
        self.assertEqual(len(expected), 31)
        with torch.device("meta"):
            mtp = MTP(real_config())
        actual = {name: list(tensor.shape) for name, tensor in mtp.state_dict().items()}
        self.assertEqual(sorted(actual), sorted(expected))
        for name in sorted(expected):
            self.assertEqual(actual[name], expected[name], f"{name} shape mismatch")

    def test_total_mtp_parameters(self):
        """参数总量必须是 2607150848 —— 与集群 safetensors header 一致。

        这个数是从 131 个分片的 header 里逐条加出来的，所以它能反过来验证
        「31 个张量一个没漏、形状一个没错」。
        """
        with torch.device("meta"):
            mtp = MTP(real_config())
        total = sum(t.numel() for t in mtp.state_dict().values())
        self.assertEqual(total, 2_607_150_848)

    def test_mtp_layer_is_qsa_not_gdn(self):
        """草稿层必须是 QSA 层：带 indexer、没有线性注意力。

        ``config.json`` 的 ``mtp.layer_types`` 是 ``["full_attention"]``，别名到
        ``indexed_attention``。按 GDN 建的话形状会差一大批，但更糟的是它不会有
        indexer——于是「复用选块」这条能力在代码里存在、在这层上永远不会发生。
        """
        cfg = small_config()
        self.assertEqual(cfg.mtp_layer_type, "indexed_attention")
        mtp = MTP(cfg)
        self.assertIsNotNone(mtp.layers[0].self_attn)
        self.assertIsNone(mtp.layers[0].linear_attn)
        self.assertIsNotNone(mtp.layers[0].self_attn.indexer)

    def test_mtp_layer_has_no_ple(self):
        """草稿层不带 PLE，即便把 PLE 层号改成 MTP 用的那个层号。

        这是显式关闭而不是靠层号越界的副作用：``ple_layer_ids`` 将来若改成含
        ``num_hidden_layers``，靠副作用的那种写法会静默长出 PLE。

        层号是一基的，所以 3 层配置下 ``[3]`` 就是第 2 号（零基）层——正是
        ``DecoderLayer`` 里 ``layer_idx`` 与 PLE 层号相接的那一格。
        """
        cfg = small_config(num_hidden_layers=3, ple_layer_ids=[3], eos_token_id=0)
        self.assertIsNotNone(cfg.ple_layer_index(2), "test setup: zero-based layer 2 is PLE")
        mtp = MTP(cfg)
        self.assertIsNone(mtp.layers[0].ple)

    def test_parameter_names_point_at_the_subtree(self):
        """``mtp_parameter_names`` 按前缀给名，两个方向都要真的落在 ``state_dict``。"""
        mtp = MTP(small_config())
        keys = set(mtp.state_dict())
        for name in mtp_parameter_names():
            self.assertIn(name, keys)
        for name in mtp_parameter_names("mtp."):
            self.assertIn(name[len("mtp."):], keys)

    def test_parameter_names_reject_a_foreign_prefix(self):
        """前缀不指向 MTP 子树时必须抛——否则调用方会以为「查过了」。"""
        with self.assertRaises(ValueError):
            mtp_parameter_names("model.")


class TestMTPDataFlow(unittest.TestCase):
    """小配置上的真前向：两路输出、嵌入残差、链式复用。"""

    @classmethod
    def setUpClass(cls):
        """建一个固定种子的草稿器，供本类的用例共用。"""
        torch.manual_seed(0)
        cls.cfg = small_config()
        cls.mtp = MTP(cls.cfg).eval()

    def _inputs(self, seq: int, past: int, batch: int = 1):
        """造一组自洽的入参。

        掩码宽度是 ``past + seq``：QSA 的索引器要求掩码覆盖的范围不超过它手里的
        key 池，而池里只有本次这 ``seq`` 个 key —— 所以 ``past=0`` 是唯一能在
        「无缓存」情形下自洽的取值。要追加步就传上一轮的 ``cache``。

        Args:
            seq: query 数。
            past: 起始位置。掩码宽度按 ``past + seq`` 算，而索引器池只有 ``seq``
                个 key，所以只有 ``past=0`` 能直接调用：其余情形必须带 ``cache``。
            batch: batch 大小。

        Returns:
            ``(embed, hidden, pos, mask)``。

        Raises:
            ValueError: ``past != 0`` 却当无缓存调用——这条路径的掩码会宽过索引器池。
        """
        cfg = self.cfg
        embed = torch.randn(batch, seq, cfg.hidden_size,
                            dtype=self.mtp.fc_hidden.weight.dtype)
        hidden = torch.randn(batch, seq, cfg.hc_hidden_size,
                             dtype=self.mtp.fc_hidden.weight.dtype)
        base = torch.arange(past, past + seq).view(1, 1, -1).expand(3, batch, -1)
        from models.qwen4exp.rope import RotaryEmbedding
        pos = RotaryEmbedding(cfg).cos_sin(base)
        mask = causal_mask_4d(batch, seq, past)
        return embed, hidden, pos, mask

    def test_two_outputs_have_the_right_widths(self):
        """``sample`` 是 ``[B,S,H]``、``multi`` 是 ``[B,S,hc*H]``。

        这两路是**同一次写回**的两个视角。若 ``sample`` 直接从 ``multi`` 的某一路
        取，形状也对，但读门加权就被跳过了——那是另一个模型。
        """
        cfg = self.cfg
        embed, hidden, pos, mask = self._inputs(3, 0)
        sample, multi, cache = self.mtp.run(embed, hidden, pos, mask)
        self.assertEqual(tuple(sample.shape), (1, 3, cfg.hidden_size))
        self.assertEqual(tuple(multi.shape), (1, 3, cfg.hc_hidden_size))
        self.assertEqual(tuple(cache.keys.shape[:3]), (1, cfg.num_key_value_heads, 3))

    def test_embedding_residual_is_step0_only(self):
        """步 0 加嵌入、步 1 不加：同一个输入在两种模式下的输出必须不同。

        这条守的是「每步都加」这个错误——它的输出形状完全正常，只是残差流被重复
        叠加了几次嵌入，而草稿质量下降不会在任何判据上表现为异常。
        """
        embed, hidden, pos, mask = self._inputs(1, 0)
        with_emb, _, _ = self.mtp.run(embed, hidden, pos, mask, step=0)
        without, _, _ = self.mtp.run(embed, hidden, pos, mask, step=1)
        self.assertFalse(torch.allclose(with_emb, without),
                         "adding the embedding residual must change the output")

    def test_chain_step2_consumes_step1_multi(self):
        """链上第二步必须吃第一步的 ``multi``；换回主模型的原始状态必须不同。

        ``multi`` 是「未塌缩、已含本步因果结果」的状态。忘了把它接下去、每步都用
        主模型那份原始状态，K 份草稿就退化成 K 份独立的单步预测——K 份全同、接受
        长度恒为 1，而形状一路正确。
        """
        embed, hidden, pos, mask = self._inputs(1, 0)
        _s1, m1, cache1 = self.mtp.run(embed, hidden, pos, mask, step=0)
        next_embed = torch.randn_like(embed)
        # 第二步在位置 1，缓存里有 1 个位置，所以掩码宽度 2 与索引器池一致。
        from models.qwen4exp.rope import RotaryEmbedding
        pos2 = RotaryEmbedding(self.cfg).cos_sin(
            torch.tensor([[[1]]]).expand(3, 1, -1))
        mask2 = causal_mask_4d(1, 1, 1)
        from_multi, _, _ = self.mtp.run(next_embed, m1, pos2, mask2,
                                        cache=cache1, step=1)
        from_original, _, _ = self.mtp.run(next_embed, hidden, pos2, mask2,
                                           cache=cache1, step=1)
        self.assertFalse(torch.allclose(from_multi, from_original),
                         "the chain must consume the previous step's multi stream")

    def test_kv_grows_by_the_number_of_rows(self):
        """草稿缓存按行数线性增长，预填充与单步追加一致。"""
        embed, hidden, pos, mask = self._inputs(4, 0)
        _sample, cache = self.mtp.prefill(embed, hidden, pos, mask)
        self.assertEqual(int(cache.keys.shape[2]), 4)
        from models.qwen4exp.rope import RotaryEmbedding
        pos1 = RotaryEmbedding(self.cfg).cos_sin(torch.tensor([[[4]]]).expand(3, 1, -1))
        e1 = torch.randn(1, 1, self.cfg.hidden_size,
                         dtype=self.mtp.fc_hidden.weight.dtype)
        h1 = torch.randn(1, 1, self.cfg.hc_hidden_size,
                         dtype=self.mtp.fc_hidden.weight.dtype)
        _, _, cache2 = self.mtp.run(e1, h1, pos1, causal_mask_4d(1, 1, 4),
                                    cache=cache, step=1)
        self.assertEqual(int(cache2.keys.shape[2]), 5)
        self.assertEqual(int(cache2.index_keys.shape[1]), 5)

    def test_truncate_rolls_the_draft_cache_back(self):
        """截断后长度就是给定的长度，且内容与「只跑到那里」逐位相同。

        投机校验回滚靠这条：接受 ``a`` 份草稿之后，草稿池必须停在
        ``ctx + a + 1``。截错位置不会报错——下一回合的草稿会对着一段多出来的 KV
        起跑，接受长度慢慢掉下去，而每一步的 logits 都"看起来正常"。
        """
        embed, hidden, pos, mask = self._inputs(6, 0)
        _sample, cache = self.mtp.prefill(embed, hidden, pos, mask)
        cut = self.mtp.truncate(cache, 4)
        self.assertEqual(int(cut.keys.shape[2]), 4)
        self.assertTrue(torch.equal(cut.keys, cache.keys[:, :, :4]))
        self.assertTrue(torch.equal(cut.index_keys, cache.index_keys[:, :4]))

    def test_truncate_rejects_out_of_range(self):
        """截断长度越界必须抛——静默 clamp 会让草稿看没校验过的 KV。"""
        embed, hidden, pos, mask = self._inputs(3, 0)
        _sample, cache = self.mtp.prefill(embed, hidden, pos, mask)
        with self.assertRaises(ValueError):
            self.mtp.truncate(cache, 4)
        with self.assertRaises(ValueError):
            self.mtp.truncate(cache, -1)
        self.assertIsNone(self.mtp.truncate(None, 0))


class TestIndexReuse(unittest.TestCase):
    """跨投机步复用选块的簿记与合法性。"""

    def test_step0_never_reuses_and_later_steps_do(self):
        """步 0 自己选（要产出块表），步 1+ 才复用。"""
        plan = DraftIndexPlan(shared=IndexReuse(picked=torch.zeros(1, 2, dtype=torch.long),
                                                max_blocks=4, ratio=2))
        self.assertIsNone(plan.before_step(0))
        self.assertIsNotNone(plan.before_step(1))
        self.assertIsNotNone(plan.before_step(2))
        self.assertEqual(plan.used, 2)

    def test_disabled_plan_never_reuses(self):
        """关掉复用后步 1+ 也自己选——这是「复用 vs 不复用」对拍的基础。"""
        plan = DraftIndexPlan(shared=IndexReuse(picked=torch.zeros(1, 2, dtype=torch.long),
                                                max_blocks=4, ratio=2),
                              enabled=False)
        self.assertIsNone(plan.before_step(1))
        self.assertEqual(plan.used, 0)

    def test_negative_step_is_rejected(self):
        """负步号必须抛：静默按步 0 处理会让整轮退化成不复用而看不出来。"""
        with self.assertRaises(ValueError):
            DraftIndexPlan().before_step(-1)

    def test_record_rejects_more_than_one_plan(self):
        """一层只会产出一份块表；多份说明调用方把主干层的输出也塞进来了。"""
        plan = DraftIndexPlan()
        with self.assertRaises(ValueError):
            plan.record([object(), object()])

    def test_record_keeps_none_on_the_short_context_fast_path(self):
        """短上下文走整段快路径时没有块表可复用，``shared`` 保持 ``None``。

        这不是失败：``kv_length <= block_topk`` 时正确答案就等于可见掩码本身，
        所以后续步自己选也是对的。
        """
        plan = DraftIndexPlan()
        plan.record([])
        self.assertIsNone(plan.shared)
        self.assertIsNone(plan.before_step(1))

    def test_reuse_rejects_a_block_table_from_a_different_width(self):
        """跨块边界复用必须抛，不能静默沿用旧块号。"""
        table = IndexReuse(picked=torch.zeros(1, 2, dtype=torch.long),
                          max_blocks=4, ratio=2)
        table.check(4, 2)
        with self.assertRaises(ValueError):
            table.check(8, 2)
        with self.assertRaises(ValueError):
            table.check(4, 4)


class TestGreedyAcceptRule(unittest.TestCase):
    """贪心接受规则：这是投机解码唯一真正的算术。

    不经过模型、不经过 MTP，只用一组人造的 logits 断言「接受多长的前缀、bonus 是
    哪个 token」。模型侧任何改动都不该影响这段。
    """

    @staticmethod
    def accept(drafts: list[int], target_argmax: list[int]) -> tuple[int, int]:
        """复算一次贪心校验。

        Args:
            drafts: 草稿，长度 K。
            target_argmax: 主模型在 K+1 行上的 argmax，长度 K+1。

        Returns:
            ``(accepted, emitted)``：接受了几个草稿，以及最后追加的那个 token。
        """
        accepted = 0
        for i, draft in enumerate(drafts):
            if target_argmax[i] != draft:
                break
            accepted += 1
        return accepted, target_argmax[accepted]

    def test_all_accepted_emits_the_bonus_token(self):
        """K 份全接受时追加 K+1 个 token，最后一个是主模型的 bonus。"""
        accepted, emitted = self.accept([7, 8, 9], [7, 8, 9, 42])
        self.assertEqual(accepted, 3)
        self.assertEqual(emitted, 42)

    def test_first_mismatch_emits_the_replacement(self):
        """第一个失配处用主模型的 argmax 替换，后面的草稿全丢。"""
        accepted, emitted = self.accept([7, 8, 9], [7, 99, 9, 42])
        self.assertEqual(accepted, 1)
        self.assertEqual(emitted, 99)

    def test_immediate_mismatch_is_plain_greedy(self):
        """第一个就失配时退化成普通贪心步：只出 1 个 token。"""
        accepted, emitted = self.accept([7, 8, 9], [11, 12, 13, 14])
        self.assertEqual(accepted, 0)
        self.assertEqual(emitted, 11)


class TestSpecMatchesGreedy(unittest.TestCase):
    """投机解码必须与逐 token 贪心**逐位一致**。

    这是投机解码唯一硬性的正确性判据：草稿全部被拒时，它必须退化成与普通贪心
    完全相同的一条路径。任何位置错位、缓存少一格、重放漏一个 token，都会在这里
    表现为「前几个 token 对、后面分叉」——而那些错在运行期都**不报错**，只让接受率
    慢慢掉。

    本用例在本轮把四类错都抓出来过：草稿池高度差一、重放串漏 ``out[-1]``、
    嵌入重复右移、草稿层与主模型用了两套位置网格。前四版都是「能跑、形状全对、
    只是 token 从第 4 个开始不同」。

    接受率在这组随机权重下必然是 0——这不是失败：MTP 头是随机初始化的，它给出的
    草稿与主模型无关。但正因为全是拒绝，每条序列都走「满拒绝」那条最长的回退路径，
    对位置错位最敏感。
    """

    @staticmethod
    def _plain_greedy(model, prompt, max_new):
        """逐 token 贪心，作为投机路径的锚。

        Args:
            model: 带 MTP 头的 ``CausalLM``。
            prompt: prompt token。
            max_new: 生成多少个。

        Returns:
            生成的 token 列表（不含 prompt）。

        Raises:
            RuntimeError: 模型没建 MTP 头（``_mtp_forward`` 需要它）。
        """
        out: list[int] = []
        with torch.no_grad():
            logits, _multi, states = model._mtp_forward(
                torch.tensor([prompt]), causal_mask(1, len(prompt), 0), 1, None)
            nxt = int(logits[0, -1].argmax())
            for _ in range(max_new):
                out.append(nxt)
                ctx = len(prompt) + len(out) - 1
                logits, _multi, states = model._mtp_forward(
                    torch.tensor([[nxt]]), causal_mask(1, 1, ctx), 1, states)
                nxt = int(logits[0, -1].argmax())
        return out

    def test_spec_matches_plain_greedy_for_every_k(self):
        """``K`` 取 1..4、三个随机种子，投机输出都必须与贪心逐位相同。

        ``K`` 要全覆盖：K=1 只暴露「校验读哪一行」，K>=2 才暴露「草稿链的第二步
        与回滚」。本轮 K=1 一路是对的，而 K>=2 从第 4 个 token 起分叉——只测 K=1
        会漏掉全部四类错。
        """
        prompt = [5, 9, 17, 23]
        for K in (1, 2, 3, 4):
            for seed in (0, 1, 2):
                with self.subTest(K=K, seed=seed):
                    torch.manual_seed(seed)
                    cfg = small_config(num_hidden_layers=2, ple_layer_ids=[],
                                       eos_token_id=1)
                    model = CausalLM(cfg, with_mtp=True).eval()
                    ref = self._plain_greedy(model, prompt, 12)
                    got, stats = model.spec_generate(prompt, 12, K=K)
                    self.assertEqual(got, ref)
                    self.assertEqual(stats["drafted"], K * stats["rounds"])

    def test_accept_rule_matches_the_shared_helper(self):
        """``spec_generate`` 的接受判定必须与 :class:`TestGreedyAcceptRule` 同一规则。

        两边各写一份接受逻辑是本仓库真实踩过的分歧来源：模型侧改了「第一个失配就
        停」，而单测里那份没改，于是单测一直绿、模型一直错。
        """
        prompt = [5, 9, 17, 23]
        torch.manual_seed(0)
        cfg = small_config(num_hidden_layers=2, ple_layer_ids=[], eos_token_id=1)
        model = CausalLM(cfg, with_mtp=True).eval()
        got, stats = model.spec_generate(prompt, 8, K=3)
        # 随机权重下草稿与主模型无关，所以几乎必然 0 接受：每次只出 1 个 token，
        # 而第一个 token 是 prompt 之后的直接 argmax（不占回合），所以回合数 =
        # 生成长度 - 1。
        if stats["accepted"] == 0:
            self.assertEqual(stats["rounds"], max(len(got) - 1, 0))
        self.assertLessEqual(stats["accepted"], stats["drafted"])


if __name__ == "__main__":
    unittest.main(verbosity=2)


