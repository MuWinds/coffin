"""``Experts`` 的 W4A8 分组分派：数值必须与改动前逐位相同，且不许回读主机。

这是 decode 性能修复的守卫，不是新的数学。``Experts.forward`` 的量化分支原先每层
要把 ``bincount`` 的结果 ``.tolist()`` 回主机、再 ``torch.tensor(...)`` 把前缀和搬回
设备；现在前缀和全程留在设备上（``moe_grouped`` 本来就吃设备侧 int32 指针）。改的是
开销，不是数值，所以主判据是**与改动前的实现逐位相等**。

三条断言各守一件只有它能发现的事：

* **黄金对拍**：把改动前的分派体原样搬进本文件当基线，两边输出 ``torch.equal``。
  用容差是没用的——这次改动的全部风险就是"结果变了一点点"，容差一放就漏掉。
* **与逐专家循环对齐**：量化路径和 bf16 循环路径算的是同一个公式。两者不会逐位
  相同：累加次序不同（循环路径逐专家 ``index_add_``，量化路径一次累加全部槽位），
  bf16 下每次累加的舍入点也不同。这里钉的是"差别只在末位"，量级由实测给出。
* **零主机回读**：把回读接口换成会抛异常的替身，分派路径必须仍然跑通，并且交给
  内核的前缀和必须是设备侧 int32、内容正确。真机上少一次同步不会报错、只会慢，
  唯一能在 CPU 上抓住它的就是这种替身。

跑法（仓库根目录）::

    python coffin/tests/test_qwen4exp_moe_dispatch.py

Module: ``coffin/tests``。只跑 CPU 与 reference 后端，不需要 checkpoint 也不需要卡。

Constraints:
    * 量化权重必须由 ``W4A8MoEWeight`` 真包出来，不能用 bf16 张量冒充：被测的正是
      读 packed nibble 加逐组 scale 的那条路，形状对而布局错的权重会让整条路径
      看起来正常。
    * 逐专家路径的对照量是**反量化后的 bf16 权重**。``W4A8MoEWeight.to()`` 与内核
      读的是同一份 packed 数据，所以两条路径喂的是同一组数。
    * 基线实现是**拷贝**，不是 ``git show``：测试不能假设工作区里有历史版本，而集群
      上引擎树是单独同步过去的、根本没有那份历史。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

ENGINE = next((p for p in Path(__file__).resolve().parents
               if (p / "models" / "qwen4exp").is_dir()), Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ENGINE))

from models.qwen4exp.config import TextConfig  # noqa: E402
from models.qwen4exp.moe import Experts  # noqa: E402
from tools import quant_disk_w4a8  # noqa: E402
from w4a8_weights import W4A8MoEWeight  # noqa: E402

#: 专家栈的 K 轴分组。取 16 而不是真实的 128，是为了在 64 宽的 hidden 上也能分出
#: 好几组：只有跨组存在，``scale`` 沿 K 的取法错了才会露出来。
GROUP_SIZE = 16

#: 关进"一读主机就炸"替身时要换掉的协议方法。``__int__`` / ``__bool__`` 覆盖
#: ``int(tensor)`` 与 ``if tensor.any()``，``tolist`` / ``item`` 覆盖显式回读。
_HOST_READS = ("tolist", "item", "__int__", "__bool__", "__float__")

#: 量化路径与 bf16 循环路径的相对误差上限。实测在 8 个 token 的批上是 6e-5 量级
#: （两者只差 bf16 的累加次序），2e-2 留出的余量足够挡住"权重取错槽位"这类 O(1)
#: 错误，又不会因为末位舍入假红。
_LOOP_TOLERANCE = 2e-2


def small_config(**override) -> TextConfig:
    """8 个专家、2 层的小配置，够跑通分派又不用等。

    Args:
        **override: 要覆盖的字段。

    Returns:
        ``TextConfig``。
    """
    fields = dict(
        vocab_size=256, hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=32,
        linear_num_key_heads=2, linear_num_value_heads=6,
        linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
        num_experts=8, num_experts_per_tok=4, moe_intermediate_size=32,
        shared_expert_intermediate_size=32, hc_count=4, hc_lowrank=16,
        ple_layer_ids=[2], ple_embed_dim=64, ple_conv_kernel_size=4,
        ngram_size=3, heads_per_ngram=8, ngram_vocab_size_base=1000,
        make_ngram_vocab_size_divisible_by=128,
        indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=16,
        indexer_budget=16, indexer_compress_ratio=4,
        output_gate_type="sigmoid", rope_theta=10000.0, partial_rotary_factor=0.25,
        mrope_section=[3, 3, 2], eos_token_id=2,
    )
    fields.update(override)
    return TextConfig(**fields)


def quantize_stack(weights: torch.Tensor, k: int) -> W4A8MoEWeight:
    """把 ``[E, N, K]`` 的 bf16 专家栈量化成 W4A8 容器。

    逐专家调 ``quantize_projection``，与磁盘侧产物用的是同一个函数：分组量化在 K
    轴上按固定组切、N 轴逐列独立，所以逐专家的结果和整栈一次算的逐位相同。

    Args:
        weights: ``[E, N, K]``，每个专家是一个投影矩阵。
        k: 每个专家沿 K 的长度。

    Returns:
        可以在 ``set_quantized`` 里用的容器。
    """
    slices = [quant_disk_w4a8.quantize_projection(weights[e].float(), GROUP_SIZE)
              for e in range(weights.shape[0])]
    return W4A8MoEWeight(torch.stack([packed for packed, _ in slices]),
                         torch.stack([scale for _, scale in slices]),
                         k=k, group_size=GROUP_SIZE)


def subset(stack: W4A8MoEWeight, experts: list[int]) -> W4A8MoEWeight:
    """按专家号从整栈里取出一个子集，保持 packed 布局不变。

    Args:
        stack: 整栈容器。
        experts: 要保留的专家号，升序。

    Returns:
        只含这些专家的新容器，``k`` 与 ``group_size`` 沿用原栈。
    """
    index = torch.tensor(experts, dtype=torch.long)
    return W4A8MoEWeight(stack.packed.index_select(0, index).contiguous(),
                         stack.scale.index_select(0, index).contiguous(),
                         k=stack.k, group_size=stack.group_size)


def quantized_experts(cfg: TextConfig) -> tuple[Experts, W4A8MoEWeight, W4A8MoEWeight]:
    """建一个挂上 W4A8 专家栈的 ``Experts``，权重来自固定种子。

    Args:
        cfg: 配置。

    Returns:
        ``(experts, gate_up, down)``。后两者返回给逐专家路径做反量化对照。
    """
    torch.manual_seed(0)
    source = Experts(cfg)
    gate_up = quantize_stack(source.gate_up_proj.data, cfg.hidden_size)
    down = quantize_stack(source.down_proj.data, cfg.moe_intermediate_size)
    experts = Experts(cfg, quantized=True)
    experts.set_quantized(gate_up, down)
    return experts, gate_up, down


def bf16_experts(cfg: TextConfig, gate_up: torch.Tensor,
                 down: torch.Tensor) -> Experts:
    """建一个走逐专家循环的 ``Experts``，权重是反量化后的 bf16。

    量化路径下 ``gate_up_proj`` / ``down_proj`` 会被 ``set_quantized`` 删掉，所以
    这里直接替换 ``Parameter`` 的底层张量——保持它仍是 ``nn.Parameter``，前向那条
    分支才会照常走 ``F.linear``。

    Args:
        cfg: 配置。
        gate_up: ``[E, 2*I, H]``，bf16。
        down: ``[E, H, I]``，bf16。

    Returns:
        没有 ``_grouped`` 的实例。
    """
    experts = Experts(cfg)
    experts.gate_up_proj.data = gate_up
    experts.down_proj.data = down
    return experts


def legacy_dispatch(experts: Experts, hidden_states: torch.Tensor,
                    top_k_index: torch.Tensor,
                    top_k_weights: torch.Tensor) -> torch.Tensor:
    """改动前的量化分派体，原样保留，只作为逐位对拍的基线。

    与 :meth:`Experts.forward` 的区别只有前缀和那三行：先 ``.tolist()`` 回主机，再用
    ``torch.tensor(...)`` 搬回设备。其余每一行都相同，包括一次性的 ``index_add_``
    累加——这正是"改的只是开销"这句断言的依据。

    Args:
        experts: 已 ``set_quantized`` 的实例。
        hidden_states: ``[tokens, H]``。
        top_k_index: ``[tokens, top_k]``。
        top_k_weights: ``[tokens, top_k]``。

    Returns:
        ``[tokens, H]``。
    """
    final = torch.zeros_like(hidden_states)
    tokens, top_k = top_k_index.shape
    flat_index = top_k_index.reshape(-1)
    flat_weights = top_k_weights.reshape(-1)
    positions_all = torch.arange(flat_index.numel(), device=flat_index.device)
    local_index = experts.local_index(flat_index[positions_all])
    weights_sel = flat_weights[positions_all]
    token_sel = torch.div(positions_all, top_k, rounding_mode="floor")
    order = torch.argsort(local_index, stable=True)
    sorted_local = local_index[order]
    counts = torch.bincount(sorted_local, minlength=experts.num_local_experts).tolist()
    offsets = torch.tensor([0] + list(torch.cumsum(
        torch.tensor(counts, dtype=torch.int32), 0)), dtype=torch.int32,
        device=hidden_states.device)
    packed_state = hidden_states[token_sel[order]]
    up_gate = experts._grouped(packed_state, offsets)  # noqa: SLF001 - 被测的入口
    gate, up = up_gate.chunk(2, dim=-1)
    current_hidden = F.silu(gate) * up
    current_hidden = experts._grouped(current_hidden, offsets, down_proj=True)  # noqa: SLF001
    current_hidden = current_hidden * weights_sel[order, None]
    final.index_add_(0, token_sel[order], current_hidden.to(final.dtype))
    del tokens
    return final


def route(cfg: TextConfig, tokens: int,
          seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """造一份固定的 ``(hidden_states, top_k_index, top_k_weights)``。

    下标是**乱序且带重复**的：同一专家可能被好几行选中，同一 token 的多个槽位也
    未必按专家号排。整齐的输入会让"只按 token 排序"这类错误看不出来。

    Args:
        cfg: 配置。
        tokens: token 数。
        seed: 随机种子。

    Returns:
        ``(hidden_states [tokens, H], indices [tokens, top_k], weights [tokens, top_k])``。
    """
    generator = torch.Generator().manual_seed(seed)
    hidden = (torch.randn(tokens, cfg.hidden_size,
                          generator=generator) * 0.5).to(torch.bfloat16)
    indices = torch.stack([torch.randperm(cfg.num_experts, generator=generator)[
        :cfg.num_experts_per_tok] for _ in range(tokens)]).to(torch.int64)
    weights = torch.rand(tokens, cfg.num_experts_per_tok,
                         generator=generator).to(torch.bfloat16)
    return hidden, indices, weights


class DispatchIsUnchangedTest(unittest.TestCase):
    """去同步之后，量化分派的输出必须与改动前逐位相同。"""

    @classmethod
    def setUpClass(cls) -> None:
        """整个类共用一个量化实例：权重构造很贵，而每个用例都只用一次前向。"""
        cls.cfg = small_config()
        cls.experts, _, _ = quantized_experts(cls.cfg)

    def _compare(self, tokens: int, seed: int) -> None:
        """跑一份随机路由，与基线实现逐位比较。

        Args:
            tokens: token 数。
            seed: 随机种子。
        """
        hidden, indices, weights = route(self.cfg, tokens=tokens, seed=seed)
        with torch.no_grad():
            want = legacy_dispatch(self.experts, hidden, indices, weights)
            got = self.experts(hidden, indices, weights)
        self.assertTrue(torch.equal(got, want),
                        f"tokens={tokens}: grouped dispatch changed, max abs diff "
                        f"{(got.float() - want.float()).abs().max().item():.3e}")

    def test_single_token_decode_step_is_bitwise_identical(self):
        """decode 一步（``tokens == 1``）逐位相同。

        这是性能修复真正跑的形状：省掉的正是这一步里那几次主机同步，而 token 只有
        一个时任何"按 token 重排"的错误都藏不住——同一 token 的 4 个槽位必须各自
        带自己的权重累加。
        """
        self._compare(tokens=1, seed=11)

    def test_batched_tokens_are_bitwise_identical(self):
        """32 个 token 的批逐位相同：多个专家、多个 token 同时非空。

        decode 只有一行，分组内核的空段跳过、跨专家的行边界在上面那个形状里几乎
        走不到；这里让大半专家都分到几行。
        """
        self._compare(tokens=32, seed=12)

    def test_unused_trailing_experts_still_dispatch_every_row(self):
        """号大的专家一行都没被选中时，每一行仍然要落到它自己的专家上。

        decode 期 128 个专家里只有十来个非空，尾部整段空是常态而不是边缘情况。空段
        必须在分组时被跳过，而且不能让后面的行错位——行错位不会报错，只会把激活喂给
        隔壁专家的权重。

        前缀和的**长度**由 :class:`NoHostReadTest` 钉住：这个用例在 reference 后端
        上看不出长度的差别（它按 ``len(offsets) - 1`` 推专家数，短一截也照样算全），
        而在 corex 上短一截会被内核以 ``E+1`` 为由拒绝。
        """
        _, indices, _ = route(self.cfg, tokens=4, seed=15)
        indices = indices % (self.cfg.num_experts // 2)
        self.assertLess(int(indices.max()), self.cfg.num_experts - 1,
                        "this case must leave the highest expert with no rows")
        self._compare_with(indices, seed=15)

    def _compare_with(self, indices: torch.Tensor, seed: int) -> None:
        """用一份给定的路由下标与基线实现逐位比较。

        Args:
            indices: ``[tokens, top_k]`` 的专家号。
            seed: 生成激活与权重用的随机种子。
        """
        hidden, _, weights = route(self.cfg, tokens=indices.shape[0], seed=seed)
        with torch.no_grad():
            want = legacy_dispatch(self.experts, hidden, indices, weights)
            got = self.experts(hidden, indices, weights)
        self.assertTrue(torch.equal(got, want),
                        "trailing experts with no rows corrupted the dispatch, max abs diff "
                        f"{(got.float() - want.float()).abs().max().item():.3e}")


class MatchesPerExpertLoopTest(unittest.TestCase):
    """量化分组路径与 bf16 逐专家循环路径算的是同一个公式。"""

    @classmethod
    def setUpClass(cls) -> None:
        """同一个量化实例喂两条路径的其中一条，另一条用反量化后的 bf16 权重。

        两条路径必须吃同一组数，否则比较没有意义；``W4A8MoEWeight.to()`` 与内核读的
        是同一份 packed 数据。
        """
        cls.cfg = small_config()
        cls.experts, gate_up, down = quantized_experts(cls.cfg)
        cls.loop = bf16_experts(cls.cfg, gate_up.to(torch.bfloat16),
                                down.to(torch.bfloat16))

    def test_within_last_bit_of_the_per_expert_loop(self):
        """与逐专家循环的差只有 bf16 的末位舍入。

        **逐位相等在这里做不到，也不该要求**：循环路径逐专家 ``index_add_``，每次
        都把该专家的贡献舍入进 bf16 累加器；量化路径一次累加全部槽位。同一个和的
        两种加序在 bf16 下必然差末位。真要比到逐位，只能把量化路径改回逐专家累加，
        那等于用每层 128 次内核启动换一个 ulp，正好是这次优化要除掉的成本。
        """
        hidden, indices, weights = route(self.cfg, tokens=8, seed=21)
        with torch.no_grad():
            want = self.loop(hidden, indices, weights)
            got = self.experts(hidden, indices, weights)
        error = (got.float() - want.float()).abs().max() / want.float().abs().max()
        self.assertLess(error.item(), _LOOP_TOLERANCE,
                        f"grouped vs per-expert loop: relative error {error.item():.3e}")

    def test_fp32_accumulation_agrees_with_the_per_expert_loop(self):
        """把累加搬进 fp32，两条路径的差缩到 fp32 舍入量级。

        上一条把差异归因于"bf16 的累加次序"。这条把它钉死：同一组权重、同一批
        贡献，在 fp32 里按专家逐个累加与一次累加相差 1e-10 量级（bf16 下是 1e-4
        量级），而"权重取错专家"在 fp32 下依旧是 O(1) 的错。所以这一条排除了另一
        种解释——量化路径其实拿错了权重，只是碰巧接近。

        判据仍不能写成逐位：fp32 的两种加序同样不保证逐位相同，只是差得极浅。
        """
        hidden, indices, weights = route(self.cfg, tokens=1, seed=24)
        flat_index = indices.reshape(-1)
        flat_weights = weights.reshape(-1)
        top_k = indices.shape[1]
        order = torch.argsort(flat_index, stable=True)
        counts = torch.bincount(flat_index[order], minlength=self.cfg.num_experts)
        gate_up = self.experts._quant_gate_up.to(torch.float32)  # noqa: SLF001
        down = self.experts._quant_down.to(torch.float32)  # noqa: SLF001

        # 逐专家累加：每一行贡献都来自它自己那个专家。
        stepwise = torch.zeros(hidden.shape[0], self.cfg.hidden_size)
        start = 0
        for expert in range(self.cfg.num_experts):
            count = int(counts[expert])
            if count == 0:
                continue
            rows = order[start:start + count]
            start += count
            tokens_of_rows = torch.div(rows, top_k, rounding_mode="floor")
            gate, up = F.linear(hidden.float()[tokens_of_rows], gate_up[expert]).chunk(2, -1)
            contribution = F.linear(F.silu(gate) * up, down[expert])
            stepwise.index_add_(0, tokens_of_rows, contribution * flat_weights[rows, None])
        self.assertEqual(start, flat_index.numel(), "the loop did not cover every routed slot")

        # 一次累加：先把每个槽位的贡献整体算出来，再一次 index_add。``F.linear``
        # 只吃 2 维权重，逐槽位各一个专家矩阵，所以这里用 batched matmul——算术
        # 与逐专家 ``F.linear`` 相同，只是把 ``N`` 个小 GEMM 收成一次。
        grouped_state = hidden.float()[torch.div(order, top_k, rounding_mode="floor")]
        gate_up_rows = gate_up[flat_index[order]]
        up_gate = torch.bmm(gate_up_rows, grouped_state.unsqueeze(-1)).squeeze(-1)
        gate, up = up_gate.chunk(2, dim=-1)
        contributions = torch.bmm(down[flat_index[order]],
                                  (F.silu(gate) * up).unsqueeze(-1)).squeeze(-1)
        contributions = contributions * flat_weights[order, None]
        oneshot = torch.zeros_like(stepwise)
        oneshot.index_add_(0, torch.div(order, top_k, rounding_mode="floor"), contributions)

        scale = stepwise.abs().max()
        self.assertLess((oneshot - stepwise).abs().max().item() / scale.item(), 1e-6,
                        "fp32 accumulation order changed the result by more than rounding: "
                        f"{(oneshot - stepwise).abs().max().item():.3e}")
        # 顺便钉住它确实接近量化路径的 bf16 输出：差异只是把 fp32 结果窄化回去。
        error = ((stepwise.to(torch.bfloat16).float()
                  - self.experts(hidden, indices, weights).float()).abs().max() / scale)
        self.assertLess(error.item(), _LOOP_TOLERANCE,
                        f"fp32 reference deviates from the kernel path by {error.item():.3e}")


class ShardedDispatchTest(unittest.TestCase):
    """EP 分片下的量化分派。"""

    def test_shard_without_local_experts_returns_zeros(self):
        """没有本地专家的分片返回全零，且不去碰不存在的权重。

        一个 rank 可能一个槽位都分不到。那条路必须在分组内核之前就返回，否则会拿
        空激活去调内核——形状合法，但前缀和与激活行数对不上。
        """
        cfg = small_config(num_experts=4)
        torch.manual_seed(0)
        source = Experts(cfg)
        gate_up = subset(quantize_stack(source.gate_up_proj.data, cfg.hidden_size), [2, 3])
        down = subset(quantize_stack(source.down_proj.data, cfg.moe_intermediate_size),
                      [2, 3])
        shard = Experts(cfg, local_experts=[2, 3], quantized=True)
        shard.set_quantized(gate_up, down)
        hidden, _, _ = route(cfg, tokens=1, seed=13)
        indices = torch.zeros(1, cfg.num_experts_per_tok, dtype=torch.int64)
        weights = torch.ones(1, cfg.num_experts_per_tok, dtype=torch.bfloat16)
        with torch.no_grad():
            got = shard(hidden, indices, weights)
        self.assertEqual(tuple(got.shape), (1, cfg.hidden_size))
        self.assertTrue(torch.equal(got, torch.zeros_like(got)),
                        "a shard with no local expert must contribute nothing")


class NoHostReadTest(unittest.TestCase):
    """W4A8 分派路径一次主机回读都不许有。"""

    def test_dispatch_path_never_reads_back(self):
        """把回读接口换成会抛异常的替身，分组路径必须照常跑完。

        ``_grouped`` 也换成替身，还有第二个目的：把"交给内核的前缀和"钉住。它必须是
        设备侧 int32、长度 ``E+1``、内容等于按专家号统计的前缀和。不能拿真的
        reference 实现来测——它自己会 ``int(offsets[-1])``（纯 torch 兜底本来就跑在
        主机上），而 corex 内核收的是设备指针，根本没有那条回读。

        替身只在这个测试期间生效，``finally`` 里必须还原——否则同一进程里后面的
        用例会全部莫名失败。
        """
        cfg = small_config()
        experts, _, _ = quantized_experts(cfg)
        hidden, indices, weights = route(cfg, tokens=1, seed=14)
        # 只路由到前一半专家。不这么做的话，被选中的最大专家号恰好就是最后一个，
        # ``bincount`` 少了 ``minlength`` 也会碰巧给出满长度的前缀和——这个参数
        # 就白丢了，而它在 decode 期每层都真的会短一截。
        indices = indices % (cfg.num_experts // 2)
        self.assertLess(int(indices.max()), cfg.num_experts - 1,
                        "this case must leave the highest expert with no rows")
        flat_index = indices.reshape(-1)
        expected = torch.zeros(cfg.num_experts + 1, dtype=torch.int32)
        expected[1:] = torch.cumsum(
            torch.bincount(flat_index, minlength=cfg.num_experts), 0)
        seen: list[tuple[torch.dtype, torch.device]] = []

        def fake_grouped(x: torch.Tensor, offsets: torch.Tensor,
                         down_proj: bool = False) -> torch.Tensor:
            """顶替分组内核：记录并校验前缀和，返回形状正确的假输出。

            Args:
                x: 已按专家排序的激活，``[tokens, K]``。
                offsets: 前缀和，内核按 int32 设备指针读它。
                down_proj: 为真时是 down 投影（``N = H``），否则 gate_up（``N = 2I``）。

            Returns:
                ``[tokens, N]``，与真内核同形状。
            """
            seen.append((offsets.dtype, offsets.device))
            self.assertTrue(torch.equal(offsets.cpu(), expected),
                            "offsets handed to the kernel are not the expert prefix sums")
            self.assertEqual(int(offsets.numel()), cfg.num_experts + 1,
                             "offsets must have E+1 entries")
            width = cfg.hidden_size if down_proj else 2 * cfg.moe_intermediate_size
            return torch.ones(x.shape[0], width, dtype=x.dtype, device=x.device)

        def forbidden(*_args, **_kwargs):
            """被禁用的回读接口：任何一次主机回读都直接失败。"""
            raise AssertionError("the W4A8 dispatch path read a tensor back to the host")

        saved = {name: getattr(torch.Tensor, name) for name in _HOST_READS}
        original = experts._grouped  # noqa: SLF001 - 被测的分派入口
        experts._grouped = fake_grouped  # noqa: SLF001
        try:
            for name in _HOST_READS:
                setattr(torch.Tensor, name, forbidden)
            with torch.no_grad():
                got = experts(hidden, indices, weights)
        finally:
            for name, replacement in saved.items():
                setattr(torch.Tensor, name, replacement)
            experts._grouped = original  # noqa: SLF001

        self.assertEqual(tuple(got.shape), (1, cfg.hidden_size))
        self.assertEqual(len(seen), 2, "both projections must go through the grouped kernel")
        for dtype, device in seen:
            self.assertEqual(dtype, torch.int32, "the kernel indexes offsets as int32")
            self.assertEqual(device, hidden.device, "offsets must live where the activations live")


if __name__ == "__main__":
    unittest.main()
