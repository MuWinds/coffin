"""EP 分片 + 每层一次归约：四卡路径必须与单卡路径给出同一份输出。

这个测试守的是**整个并行契约**，而不只是某个算子：

* 每个 rank 只持有 ``num_experts / world`` 个专家（显存与权重流量的来源）；
* dense 部分（mixer、超连接、共享专家）每卡全量，所以没有 dispatch；
* 每层在 MoE 的路由专家之后归约**一次**，把各 rank 的部分和加起来。

判据是「四卡拼起来 == 单卡全量」。只验证「每个 rank 都能跑完」是不够的：
漏掉一次归约、归约里少加一个 rank、或者把共享专家也一起归约（那会放大
``world`` 倍），输出仍然是有限值——只有这个等价性能发现。

真机上的归约由 ``parallel.ShmAllReduce`` 提供，它自身的正确性在
``test_qwen4exp_parallel.py`` 里用真进程加 gloo 交叉验证过。这里用进程内的
求和替身，因为被测的是「归约放在对的层、对的位置、喂对的张量」。

跑法（仓库根目录）::

    python coffin/tests/test_qwen4exp_ep.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ENGINE = next((p for p in Path(__file__).resolve().parents
               if (p / "models" / "qwen4exp").is_dir()), Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ENGINE))

from models.qwen4exp.config import TextConfig  # noqa: E402
from models.qwen4exp.model import CausalLM  # noqa: E402

#: 分片数。8 个专家切成 4 份正好每份 2 个，既能看出越界也能看出遗漏。
WORLD = 4


def base_config(**override) -> TextConfig:
    """小配置：8 个专家、4 层。

    Args:
        **override: 要覆盖的字段。

    Returns:
        ``TextConfig``。
    """
    fields = dict(
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
    fields.update(override)
    return TextConfig(**fields)


def causal_mask(seq: int) -> torch.Tensor:
    """fp32 因果掩码，可见处 0、屏蔽处 ``finfo.min``。

    Args:
        seq: 序列长度。

    Returns:
        ``[1, 1, seq, seq]``。
    """
    matrix = torch.full((1, 1, seq, seq), torch.finfo(torch.float32).min)
    for i in range(seq):
        matrix[0, 0, i, : i + 1] = 0.0
    return matrix


def make_weights(cfg: TextConfig) -> dict:
    """固定随机种子造一份全量权重，四卡路径与单卡路径共用同一份。

    Args:
        cfg: 配置。

    Returns:
        全量模型的 ``state_dict``。
    """
    torch.manual_seed(0)
    model = CausalLM(cfg)
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    del model
    return state


def shard_state(state: dict, rank: int, cfg: TextConfig) -> dict:
    """把全量权重切成某 rank 的 1/4 专家分片。

    只切两个 3D 专家张量；其余参数（dense 部分）每卡全量复制——这正是 EP 的
    契约：不切 dense、只切专家。

    Args:
        state: 全量 ``state_dict``。
        rank: 本 rank 编号。
        cfg: 配置。

    Returns:
        可以直接 ``load_state_dict`` 的字典。
    """
    per_rank = cfg.num_experts // WORLD
    lo, hi = rank * per_rank, (rank + 1) * per_rank
    sliced = dict(state)
    for name, tensor in state.items():
        if "mlp.experts.gate_up_proj" in name or "mlp.experts.down_proj" in name:
            sliced[name] = tensor[lo:hi]
    return sliced


def run(model: CausalLM, cfg: TextConfig, ids: torch.Tensor, tables: dict,
        combine=None) -> torch.Tensor:
    """跑一次前向，返回 logits。

    Args:
        model: 模型。
        cfg: 配置。
        ids: 输入 token。
        tables: PLE 的 n-gram 表。
        combine: 归约替身。

    Returns:
        模型产出的 logits。
    """
    seq = ids.shape[1]
    with torch.no_grad():
        logits, _ = model(
            input_ids=ids, attention_mask=causal_mask(seq),
            conv_mask=torch.ones(1, seq, dtype=torch.bool),
            position_ids=torch.arange(seq).view(1, 1, -1).expand(3, 1, -1),
            ple_tables=tables)
    del combine
    return logits


class TestEpCombine(unittest.TestCase):
    """分片 + 归约的位置与等价性。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = base_config()
        cls.seq = 3
        torch.manual_seed(7)
        cls.ids = torch.randint(0, 256, (1, cls.seq))
        cls.state = make_weights(cls.cfg)
        torch.manual_seed(11)
        ple_embedding = CausalLM(cls.cfg).model.layers[1].ple.ple_embedding
        cls.table = torch.rand(ple_embedding.padded_vocab_size,
                               ple_embedding.head_dim_per_ngram) * 0.02
        cls.padded_rows = ple_embedding.padded_vocab_size
        del ple_embedding
        cls.tables = {1: cls.table, 2: cls.table, 3: cls.table}

    def test_combine_called_once_per_layer_with_hidden_width(self):
        """每层恰好归约一次，入参形状是 ``[B, S, hidden_size]``。

        次数错了（每层两次、或漏层）不会报错；宽度必须是读门塌缩后的一路
        ``hidden_size``，而不是 4 路残差流的 ``hc_count * hidden_size``。

        形状是**扁平的** ``[tokens, hidden]``：MoE 内部先按 token 摊平再选专家，
        所以归约天然作用在这个布局上。decode 时 ``tokens == 1``，正好是
        ``parallel.ShmAllReduce`` 的 ``rows=1`` 情形；prefill 时 ``tokens`` 可能是
        几千，那个共享内存通道在 rows=64 时代价就到 188 ms，**prefill 必须换一条
        归约路径**（或者干脆改成分块流水），这是已知的待办而不是本测试的范围。
        """
        cfg = self.cfg
        per_rank = cfg.num_experts // WORLD
        for rank in range(WORLD):
            recorded: list[tuple[int, ...]] = []

            def combine(tensor, _recorded=recorded):
                _recorded.append(tuple(tensor.shape))
                return tensor

            model = CausalLM(cfg, local_experts=list(
                range(rank * per_rank, (rank + 1) * per_rank)), combine=combine)
            model.load_state_dict(shard_state(self.state, rank, cfg))
            model.eval()
            run(model, cfg, self.ids, self.tables)
            self.assertEqual(len(recorded), cfg.num_hidden_layers,
                             f"rank {rank}: expected one combine per layer")
            expected = (1 * self.seq, cfg.hidden_size)
            for shape in recorded:
                self.assertEqual(shape, expected,
                                 f"rank {rank}: combine fed {shape}, expected {expected}")

    def test_shards_sum_to_single_card(self):
        """四个分片的专家部分和相加 == 单卡全量。

        做法是把每层归约点的四份部分和加起来，作为"归约之后"的专家输出，再
        注入单卡前向的同一位置，最后与真实单卡输出比较。两者相等即说明：分片
        没有遗漏（少算某个专家）也没有重复（多算一次）。
        """
        cfg = self.cfg
        per_rank = cfg.num_experts // WORLD
        recorded_per_rank: list[list[torch.Tensor]] = []
        for rank in range(WORLD):
            bucket: list[torch.Tensor] = []

            def combine(tensor, _bucket=bucket):
                _bucket.append(tensor.detach().clone())
                return tensor

            model = CausalLM(cfg, local_experts=list(
                range(rank * per_rank, (rank + 1) * per_rank)), combine=combine)
            model.load_state_dict(shard_state(self.state, rank, cfg))
            model.eval()
            run(model, cfg, self.ids, self.tables)
            recorded_per_rank.append(bucket)

        merged = []
        for layer in range(cfg.num_hidden_layers):
            total = recorded_per_rank[0][layer]
            for rank in range(1, WORLD):
                total = total + recorded_per_rank[rank][layer]
            merged.append(total)

        # 单卡参照：全量专家、不归约。
        reference = CausalLM(cfg)
        reference.load_state_dict(self.state)
        reference.eval()
        expected = run(reference, cfg, self.ids, self.tables)

        # 把归约后的专家输出注入单卡前向，替代它自己算的专家部分。
        cursor = {"i": 0}

        def consume(tensor):
            out = merged[cursor["i"]]
            cursor["i"] += 1
            return out

        fused = CausalLM(cfg, combine=consume)
        fused.load_state_dict(self.state)
        fused.eval()
        got = run(fused, cfg, self.ids, self.tables)
        self.assertEqual(cursor["i"], cfg.num_hidden_layers)
        self.assertLess((got - expected).abs().max().item(), 1e-5,
                        "EP shards do not sum to the single-card expert output")

    def test_shared_expert_not_reduced(self):
        """共享专家不能进归约：否则四卡会把它放大 4 倍。

        判据：把 combine 换成"返回 0"，那么剩下的就只有共享专家那一路。单卡
        与四卡在这条路径上必须给出同一个值——dense 部分是每卡全量的。
        """
        cfg = self.cfg
        per_rank = cfg.num_experts // WORLD

        def zero(tensor):
            return torch.zeros_like(tensor)

        shared_only = CausalLM(cfg, combine=zero)
        shared_only.load_state_dict(self.state)
        shared_only.eval()
        shard_logits = run(shared_only, cfg, self.ids, self.tables)

        # 单卡：把 experts 的权重清零，等价于"没有路由专家贡献"。
        zeroed = dict(self.state)
        for name in list(zeroed):
            if "mlp.experts.gate_up_proj" in name or "mlp.experts.down_proj" in name:
                zeroed[name] = torch.zeros_like(zeroed[name])
        plain = CausalLM(cfg)
        plain.load_state_dict(zeroed)
        plain.eval()
        reference = run(plain, cfg, self.ids, self.tables)
        self.assertLess((shard_logits - reference).abs().max().item(), 1e-5,
                        "shared expert output differs between sharded and single card")
        del per_rank

    def test_shard_size_and_local_index(self):
        """每个 rank 只持有 1/4 专家，且全局号到本地下标的映射正确。"""
        cfg = self.cfg
        per_rank = cfg.num_experts // WORLD
        for rank in range(WORLD):
            shard = list(range(rank * per_rank, (rank + 1) * per_rank))
            model = CausalLM(cfg, local_experts=shard)
            experts = model.model.layers[0].mlp.experts
            self.assertEqual(experts.gate_up_proj.shape[0], per_rank)
            self.assertEqual(experts.down_proj.shape[0], per_rank)
            mapping = experts.local_index(torch.arange(cfg.num_experts))
            for global_index in range(cfg.num_experts):
                local = int(mapping[global_index])
                if global_index in shard:
                    self.assertEqual(local, global_index - rank * per_rank)
                else:
                    self.assertEqual(local, -1)


if __name__ == "__main__":
    unittest.main()
