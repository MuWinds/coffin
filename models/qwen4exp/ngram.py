"""n-gram 嵌入与 PLE：把哈希出来的短 n-gram 特征注入 4 路残差流。

这一节每个常数都参与哈希：``splitmix64`` 的四个魔数、层乘子的上界与奇数约束、
每头词表大小（质数）、拼接偏移、eos 分段规则。错一个都**不会抛异常**，只是让
检索退化成噪声，所以全部按参考实现逐位照搬，并把"为什么"写在这里而不是散在
代码里。

Module: ``coffin/models/qwen4exp``。依赖 ``config``（几何）、``norm``（三处
零中心 RMSNorm）、``proj``（两次稠密投影）和 ``util``（短卷积）；被 ``blocks``
调用。

Constraints:
    * 乘法用 int64、XOR 不能换成加法、``torch.remainder`` 不能换成 C 的 ``%``
      ——三个都会静默改变哈希分布。
    * 层乘子是 buffer 不是 parameter，checkpoint 里即使存了也不信，一律按
      ``(unigram_vocab_size, ngram_size, ple_layer_index, seed)`` 重算：它是
      纯函数，重算保证与参考实现逐位一致。
    * n-gram 上下文初值是 **eos 而不是 0**。填 0 会让 0 号 token 参与哈希，
      序列开头那几个 token 的 n-gram 就变成"假前缀 + 真 token"。
    * PLE 短卷积的状态长度是 ``(k-1) * dilation = 9``，不是 3：dilation 复用了
      ``ngram_size``，改它同时改感受野（``ARCH_SPEC.md`` §12 第 12 条）。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from models.qwen4exp.config import TextConfig
from models.qwen4exp.norm import RMSNorm
from models.qwen4exp.proj import Proj
from models.qwen4exp.util import apply_padding_mask, causal_depthwise_conv

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007

#: 层乘子的种子偏移。参考实现把它写死在 ``_build_layer_multipliers`` 里，改它
#: 等于换一张表——而表是 checkpoint 的一部分。
_SEED = 1234


def splitmix64(value: int) -> int:
    """标准 splitmix64 的一轮混合。

    每一步都要 ``& MASK64`` 做 64 位回绕。Python 大整数天然正确，但这段逻辑
    以后要翻成 C++/CUDA：那里必须用 ``unsigned long long``，否则 ``>>`` 在负数
    上是算术移位，整个结果偏掉而不会报错。

    Args:
        value: 64 位以内的整数输入。

    Returns:
        混合后的 64 位无符号整数。
    """
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def layer_multipliers(unigram_vocab_size: int, ngram_size: int, ple_layer_index: int,
                      seed: int = _SEED) -> torch.Tensor:
    """按层推导 n-gram 哈希用的 ``ngram_size`` 个奇数乘子。

    上界取 ``max_long // vocab_size`` 是为了保证 ``token_id * multiplier < 2^63``：
    真实词表下 ``248319 * 3.713e13 ≈ 9.22e18`` 刚好卡在 ``2^63 - 1`` 之下。上界
    松一点就会在词表尾溢出成负数，之后 ``remainder`` 和 XOR 全错。乘子强制奇数
    让乘法在 2 的幂下可逆，XOR 混合不会抹掉信息。

    Args:
        unigram_vocab_size: 主词表行数（不是 ``ple_embed_dim``）。
        ngram_size: 最大 n-gram 阶数。
        ple_layer_index: PLE 模块在 ``ple_layer_ids`` 里的序号，不是层号。
        seed: 配置里的 ``seed``。

    Returns:
        ``[ngram_size]`` 的 int64 张量，每项是 ``[1, 2*half_bound]`` 内的奇数。

    Raises:
        ValueError: ``ngram_size`` 非正。
    """
    if ngram_size <= 0:
        raise ValueError(f"ngram_size must be positive, got {ngram_size}.")
    max_long = (1 << 63) - 1
    multiplier_max = max_long // max(unigram_vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    values = []
    for index in range(ngram_size):
        value = (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
        values.append(2 * (splitmix64(value) % half_bound) + 1)
    return torch.tensor(values, dtype=torch.long)


def is_prime(value: int) -> bool:
    """朴素试除到 ``isqrt``。

    Args:
        value: 待判定整数。

    Returns:
        是否为质数。只用来在构造期算表，**绝不要放进前向**——没有缓存，真实
        配置下要试除 16 次、每次几千个数。
    """
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def find_nth_prime_after(start: int, count: int) -> int:
    """返回 ``start`` 之后（不含）数第 ``count`` 个质数。

    Args:
        start: 起点，返回值严格大于它。
        count: 数几个。

    Returns:
        第 ``count`` 个质数。

    Raises:
        ValueError: ``count`` 非正。
    """
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}.")
    prime = start
    for _ in range(count):
        prime += 1
        while not is_prime(prime):
            prime += 1
    return prime


class NGramEmbedding(nn.Module):
    """哈希短 n-gram 并查一张融合的嵌入表。

    表是**一个** ``[padded_vocab_size, head_dim_per_ngram]`` 的 ``nn.Embedding``，
    16 个头靠各自的行偏移拼在一起：每头的词表大小是
    ``ngram_vocab_size_base`` 之后的第 ``global_head_idx + 1`` 个质数，彼此互质、
    各行不重叠。

    Attributes:
        ngram_size: 最大阶数（3 表示同时用 2-gram 和 3-gram）。
        context_len: 需要缓存的 token 历史长度，``ngram_size - 1``。
        ngram_heads: 头总数 ``(ngram_size - 1) * heads_per_ngram``。
        ngram_embedding: 融合嵌入表。
    """

    def __init__(self, config: TextConfig, embedding_dim: int, layer_idx: int,
                 ple_layer_index: int = 0, stream: bool = False) -> None:
        """按配置推导几何并建表。

        Args:
            config: 文本配置。
            embedding_dim: 所有头拼起来的总宽度。
            layer_idx: 零基层号，只用于记录。
            ple_layer_index: PLE 模块序号，参与哈希种子与头编号。
            stream: 只建几何、**不分配嵌入表**。真实 checkpoint 的表是
                ``[320001536, 160]`` bf16 = 95 GiB，任何单机 CPU 都放不下，
                所以参考实现直接把它排除出 device map（``_no_placement_params``）；
                引擎侧必须把它当流式资源。置真时 ``ngram_embedding`` 为 ``None``，
                调用方在 ``forward`` 里显式传入 ``table``。

        Raises:
            ValueError: ``embedding_dim`` 不能被头数整除，或配置里没有 eos
                （eos 是分段哨兵）。
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.ngram_size = config.ngram_size
        self.context_len = self.ngram_size - 1
        self.heads_per_ngram = config.heads_per_ngram
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        self.ple_layer_index = ple_layer_index
        self.unigram_vocab_size = config.vocab_size
        self.ngram_vocab_size_base = config.ngram_vocab_size_base
        self.seed = config.seed
        if self.ngram_heads <= 0 or embedding_dim % self.ngram_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by the n-gram head "
                f"count ({self.ngram_heads}).")
        self.head_dim_per_ngram = embedding_dim // self.ngram_heads
        self.eos_token_id = config.eos_token
        if self.eos_token_id is None:
            raise ValueError("eos_token_id must be set for the n-gram embedding.")

        head_vocab_sizes: list[int] = []
        head_offsets: list[int] = []
        total = 0
        for head_idx in range(self.ngram_heads):
            # 每个头一张互质的子表：同一层内 16 个头不共享行，不同层的同名头
            # 也因为 global_head_idx 带层偏移而落在不同的质数区间。
            global_head_idx = self.ple_layer_index * self.ngram_heads + head_idx
            size = find_nth_prime_after(self.ngram_vocab_size_base - 1, global_head_idx + 1)
            head_vocab_sizes.append(size)
            head_offsets.append(total)
            total += size
        self.total_vocab_size = total

        divisor = config.make_ngram_vocab_size_divisible_by
        padded_vocab_size = math.ceil(total / divisor) * divisor

        # 三个 buffer 由 ``reset_buffers`` 重算覆盖，checkpoint 里存了也不信。
        self.register_buffer("layer_multipliers",
                             layer_multipliers(self.unigram_vocab_size, self.ngram_size,
                                               self.ple_layer_index, self.seed))
        self.register_buffer("ngram_heads_vocab_sizes",
                             torch.tensor(head_vocab_sizes, dtype=torch.long))
        self.register_buffer("ngram_heads_offsets",
                             torch.tensor(head_offsets, dtype=torch.long))
        self.padded_vocab_size = padded_vocab_size
        # 流式模式只留几何：95 GiB 的表由调用方按分片喂进 forward。
        self.ngram_embedding = (None if stream
                                else nn.Embedding(padded_vocab_size, self.head_dim_per_ngram))

    def reset_buffers(self) -> None:
        """按配置重算三个 buffer，覆盖 checkpoint 里可能存在的旧值。

        ``layer_multipliers`` / ``ngram_heads_vocab_sizes`` / ``ngram_heads_offsets``
        都是 ``(vocab, ngram_size, ple_layer_index, seed)`` 的纯函数，加载后重算
        比信任权重文件更安全：它们从不参与训练，只是被顺手存进了 checkpoint。
        """
        self.layer_multipliers.copy_(
            layer_multipliers(self.unigram_vocab_size, self.ngram_size,
                              self.ple_layer_index, self.seed))
        self.ngram_heads_vocab_sizes.copy_(torch.tensor(
            [self._head_size(head) for head in range(self.ngram_heads)], dtype=torch.long))
        offsets, total = [], 0
        for size in self.ngram_heads_vocab_sizes.tolist():
            offsets.append(total)
            total += size
        self.ngram_heads_offsets.copy_(torch.tensor(offsets, dtype=torch.long))

    def _head_size(self, head_idx: int) -> int:
        """第 ``head_idx`` 个头的词表大小（全局编号带上层偏移）。

        Args:
            head_idx: 本层内的头序号，范围是 ``0 .. ngram_heads - 1``。

        Returns:
            该头的词表行数，即 ``ngram_vocab_size_base`` 之后的第
            ``global_head_idx + 1`` 个质数。
        """
        global_head_idx = self.ple_layer_index * self.ngram_heads + head_idx
        return find_nth_prime_after(self.ngram_vocab_size_base - 1, global_head_idx + 1)

    def shift_right_ignore_eos(self, token_ids: torch.Tensor, shift: int) -> torch.Tensor:
        """右移 ``shift`` 位，但**不跨 eos 边界**。

        语义是"在本段内往回够得到 ``shift`` 步才取那个 token，否则填 eos"。
        eos 是硬边界：跨过它取 token 会让上一段的尾巴参与下一段的 n-gram 哈希，
        形状全对而检索结果错。引擎的前缀复用也必须在 eos 处断开上下文。

        Args:
            token_ids: ``[B, L]`` 的 token id（含历史）。
            shift: 右移位数；0 表示原样返回。

        Returns:
            ``[B, L]``，越界或跨段的槽位填 ``eos_token_id``。
        """
        if shift == 0:
            return token_ids
        batch_size, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device, dtype=torch.long)
        eos_positions = torch.where(token_ids == self.eos_token_id, positions, -1)
        # 到当前位置（含）为止最近一个 eos 的绝对下标；没有就是 -1。
        previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
        previous_eos = torch.cat(
            [eos_positions.new_full((batch_size, 1), -1), previous_eos_inclusive[:, :-1]], dim=1)
        segment_start = previous_eos + 1
        position_in_segment = positions.unsqueeze(0) - segment_start
        source_positions = positions - shift
        gather_positions = source_positions.clamp_min(0).unsqueeze(0).expand(batch_size, -1)
        shifted = token_ids.gather(dim=1, index=gather_positions)
        valid = (position_in_segment >= shift) & (source_positions.unsqueeze(0) >= 0)
        return torch.where(valid, shifted, token_ids.new_full((), self.eos_token_id))

    def ngram_ids(self, input_ids: torch.Tensor,
                  previous_context: torch.Tensor | None = None) -> torch.Tensor:
        """算出每个 token 的 ``ngram_heads`` 个表行下标。

        2-gram（前 8 头）用 ``t0*m0 XOR t1*m1``，3-gram（后 8 头）多异或一项。
        乘法在 int64 上做，``torch.remainder`` 是 Python 语义（结果恒非负）——
        换成 C 的 ``%`` 会在有负乘子时出负数下标。

        Args:
            input_ids: 当前 chunk 的 token id，``[B, S]``。
            previous_context: 历史 ``[B, context_len]``，``None`` 时全填 eos。

        Returns:
            ``[B, S, ngram_heads]`` 的 int64 下标，已加过头偏移。

        Raises:
            ValueError: ``previous_context`` 的宽度不是 ``context_len``。
        """
        input_ids = input_ids.long()
        if previous_context is None:
            previous_context = input_ids.new_full(
                (input_ids.shape[0], self.context_len), self.eos_token_id)
        elif previous_context.shape[-1] != self.context_len:
            raise ValueError(
                f"n-gram context must be {self.context_len} token ids, got "
                f"{previous_context.shape[-1]}.")

        token_history = torch.cat([previous_context, input_ids], dim=-1)
        shifted = [self.shift_right_ignore_eos(token_history, shift)
                   for shift in range(self.ngram_size)]

        blocks = []
        multipliers = self.layer_multipliers
        for ngram in range(2, self.ngram_size + 1):
            start = (ngram - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            mixed_ids = shifted[0] * multipliers[0]
            for position in range(1, ngram):
                mixed_ids = torch.bitwise_xor(mixed_ids, shifted[position] * multipliers[position])
            head_sizes = self.ngram_heads_vocab_sizes[start:end]
            head_offsets = self.ngram_heads_offsets[start:end]
            ids = torch.remainder(mixed_ids.unsqueeze(-1), head_sizes.view(1, 1, -1))
            blocks.append(ids + head_offsets.view(1, 1, -1))
        # 历史比当前 chunk 长，只保留最后 S 列（它们才对应当前 token）。
        return torch.cat(blocks, dim=-1)[:, -input_ids.shape[1]:]

    def forward(self, input_ids: torch.Tensor, previous_context: torch.Tensor | None = None,
                table: torch.Tensor | None = None) -> torch.Tensor:
        """查表并把所有头拼回 ``embedding_dim`` 宽。

        Args:
            input_ids: ``[B, S]`` 的 token id。
            previous_context: 历史 ``[B, context_len]``，见 :meth:`ngram_ids`。
            table: 覆盖用的嵌入权重；``None`` 时用模块自己的
                ``ngram_embedding.weight``。流式模式下必须给：真实表
                320001536×160 bf16 约 95 GiB，任何单机都放不下，它由调用方按
                分片取出对应行再传进来。

        Returns:
            ``[B, S, embedding_dim]``。

        Raises:
            ValueError: 既没建表也没传 ``table``。静默返回零会让 PLE 注入变成
                噪声，而形状完全正确。
        """
        ids = self.ngram_ids(input_ids, previous_context)
        if table is None:
            if self.ngram_embedding is None:
                raise ValueError(
                    "this NGramEmbedding was built in stream mode; pass the table "
                    "explicitly (the real one is ~95 GiB and cannot be allocated).")
            weight = self.ngram_embedding.weight
        else:
            weight = table
        return F.embedding(ids.to(weight.device), weight).flatten(-2)


class PLELayer(nn.Module):
    """把 n-gram 特征注入每一路残差流。

    每条残差流有一把自己的 key 与一个标量 sigmoid 门；value 是单路、复制到 4 路
    上。最终输出是"未归一化、未卷积的 gated"加上一条 dilation=3 的短卷积分支，
    是残差结构而不是替换。

    Attributes:
        short_conv_state_len: 短卷积需要缓存的历史长度，
            ``(ple_conv_kernel_size - 1) * ngram_size``。
    """

    def __init__(self, config: TextConfig, layer_idx: int, ple_layer_index: int,
                 stream_table: bool = False) -> None:
        """建 n-gram 表、三处归一化、两次投影与短卷积。

        Args:
            config: 文本配置。
            layer_idx: 零基层号。
            ple_layer_index: PLE 模块序号，参与哈希种子。
            stream_table: 只建 n-gram 几何、不分配那张 95 GiB 的表。见
                :class:`NGramEmbedding`。
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.hc_count = config.hc_count
        self.ple_embed_dim = int(config.ple_embed_dim)
        hc_hidden_size = self.hidden_size * self.hc_count
        self.hc_hidden_size = hc_hidden_size
        self.ple_embedding = NGramEmbedding(config, self.ple_embed_dim, layer_idx,
                                            ple_layer_index, stream=stream_table)
        conv_kernel_size = config.ple_conv_kernel_size
        conv_dilation = config.ngram_size
        # 状态长度必须是 (k-1)*dilation：短卷积有膨胀，一个 kernel 覆盖的时间
        # 跨度是 1 + dilation*(k-1)，少存一段就会让感受野整体前移。
        self.short_conv_state_len = (conv_kernel_size - 1) * conv_dilation
        self.conv_dilation = conv_dilation
        self.key_proj = Proj(self.ple_embed_dim, hc_hidden_size)
        self.value_proj = Proj(self.ple_embed_dim, self.hidden_size)
        self.norm_key = RMSNorm(hc_hidden_size, group_size=self.hidden_size,
                                eps=config.rms_norm_eps)
        self.norm_query = RMSNorm(hc_hidden_size, group_size=self.hidden_size,
                                  eps=config.rms_norm_eps)
        self.norm_conv = RMSNorm(hc_hidden_size, group_size=self.hidden_size,
                                 eps=config.rms_norm_eps)
        self.conv1d = nn.Conv1d(hc_hidden_size, hc_hidden_size, kernel_size=conv_kernel_size,
                                groups=hc_hidden_size, dilation=conv_dilation, bias=False)

    def short_conv(self, hidden_states: torch.Tensor,
                   history: torch.Tensor | None = None) -> torch.Tensor:
        """dilation 卷积加 silu，输入输出都是 ``[B, S, hc_hidden_size]``。

        Args:
            hidden_states: ``[B, S, hc_hidden_size]``。
            history: ``[B, hc_hidden_size, short_conv_state_len]`` 的历史，
                布局是通道优先（状态缓存就是这个布局）。

        Returns:
            ``[B, S, hc_hidden_size]``。
        """
        out = causal_depthwise_conv(hidden_states.transpose(1, 2), self.conv1d.weight,
                                    history=history, dilation=self.conv_dilation,
                                    activation="silu")
        return out.transpose(1, 2)

    def update_context(self, input_ids: torch.Tensor,
                       previous_context: torch.Tensor | None) -> torch.Tensor:
        """把 n-gram 上下文更新成本次 chunk 的最后 ``context_len`` 个 token id。

        初值必须是 **eos 而不是 0**：填 0 会让 0 号 token 参与哈希，序列开头那几个
        token 的 n-gram 就变成「假前缀 + 真 token」，形状全对、检索全错。首次前向
        且 ``S < context_len`` 时要左填 eos（通用的「取最后 N 列」会填 0）。

        Args:
            input_ids: 本次 chunk 的 token id，``[B, S]``。
            previous_context: 上一步的上下文，``[B, context_len]``；``None`` 表示
                这是第一步。

        Returns:
            ``[B, context_len]`` 的新上下文（int64）。
        """
        context_len = self.ple_embedding.context_len
        if previous_context is None:
            previous_context = input_ids.new_full(
                (input_ids.shape[0], context_len), self.ple_embedding.eos_token_id)
        history = torch.cat([previous_context, input_ids.long()], dim=-1)
        # 首次前向且序列比窗口短时，左填 eos 而不是 0。
        if history.shape[-1] < context_len:
            pad = context_len - history.shape[-1]
            history = torch.nn.functional.pad(
                history, (pad, 0), value=self.ple_embedding.eos_token_id)
        return history[:, -context_len:].contiguous()

    def update_conv_state(self, gated_value_normed: torch.Tensor,
                          history: torch.Tensor | None) -> torch.Tensor:
        """把短卷积窗口更新成「进卷积之前」的最后 ``short_conv_state_len`` 列。

        PLE 的卷积状态存的是 ``norm_conv(gated_value)`` 的输出，**进卷积之前**；
        存卷积之后的值会让热启动后每一步都偏。长度必须是
        ``(k-1) * dilation = 9``，不是 3。

        Args:
            gated_value_normed: 本次 chunk 进卷积的激活，``[B, S, hc_hidden]``。
            history: 上一步的窗口，``[B, hc_hidden, state_len]``；``None`` 表示
                从零开始。

        Returns:
            ``[B, hc_hidden, state_len]`` 的新窗口。``state_len`` 为 0 时返回空窗口
            （``k == 1`` 的退化配置，没有历史可用）。
        """
        state_len = self.short_conv_state_len
        channels = gated_value_normed.transpose(1, 2)
        if state_len == 0:
            return channels.new_zeros(channels.shape[0], channels.shape[1], 0)
        if history is None:
            history = channels.new_zeros(channels.shape[0], channels.shape[1], state_len)
        combined = torch.cat([history, channels], dim=-1)
        return combined[..., -state_len:].contiguous()

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor,
                previous_context: torch.Tensor | None = None,
                history: torch.Tensor | None = None,
                conv_mask: torch.Tensor | None = None,
                table: torch.Tensor | None = None,
                return_state: bool = False
                ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """把 PLE 增量加到 4 路残差流上。

        Args:
            hidden_states: 4 路残差流，``[B, S, hc_count * hidden_size]``。
            input_ids: ``[B, S]`` 的 token id（padding 位置在调用方已被换成 eos）。
            previous_context: n-gram 历史 ``[B, context_len]``。
            history: 短卷积历史 ``[B, hc_hidden_size, short_conv_state_len]``。
            conv_mask: ``[B, S]`` 的 keep/drop 标记；进卷积前要把 padding 清零。
            table: 覆盖 n-gram 嵌入权重，见 :meth:`NGramEmbedding.forward`。
            return_state: 一并返回更新后的 n-gram 上下文与短卷积窗口。带 cache
                的前向必须打开：漏掉这两个状态，解码第 2 步之后 PLE 就对着一个
                永远是 eos 的假上下文哈希，而形状完全正确。

        Returns:
            默认返回 ``[B, S, hc_count * hidden_size]``，**增量**（调用方负责相加）；
            ``return_state=True`` 时返回
            ``(增量, 新上下文, 新卷积窗口)``。

        Raises:
            ValueError: ``hidden_states`` 的宽度不是 ``hc_count * hidden_size``。
        """
        if hidden_states.shape[-1] != self.hc_hidden_size:
            raise ValueError(
                f"PLE expects {self.hc_hidden_size} hyper-connection features, got "
                f"{hidden_states.shape[-1]}.")
        embeddings = self.ple_embedding(input_ids, previous_context, table=table)
        key_normed = self.norm_key(self.key_proj(embeddings)).unflatten(
            -1, (self.hc_count, self.hidden_size))
        value = self.value_proj(embeddings)
        query_normed = self.norm_query(hidden_states).unflatten(
            -1, (self.hc_count, self.hidden_size))
        gate = (key_normed * query_normed).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
        # 带符号平方根：压缩大幅度、保留符号，0 附近有 1e-3 量级的下限。换成
        # sqrt(abs(g)) 会丢符号，clamp_min(g, 1e-6) 会丢负数——两者形状都对。
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        # value 是单路 2560，unsqueeze 后广播到 4 路：4 路共用同一个 value，
        # 只是每路有自己的 sigmoid 门。
        gated_value = torch.sigmoid(gate) * value.unsqueeze(-2)
        gated_value_normed = self.norm_conv(gated_value.flatten(-2))
        gated_value = gated_value.flatten(-2)
        if conv_mask is not None:
            gated_value = apply_padding_mask(gated_value, conv_mask)
            gated_value_normed = apply_padding_mask(gated_value_normed, conv_mask)
        # 残差结构：未归一化、未卷积的 gated 直接加卷积分支，不是二选一。
        out = gated_value + self.short_conv(gated_value_normed, history)
        if not return_state:
            return out
        return (out,
                self.update_context(input_ids, previous_context),
                self.update_conv_state(gated_value_normed, history))


__all__ = [
    "NGramEmbedding",
    "PLELayer",
    "find_nth_prime_after",
    "is_prime",
    "layer_multipliers",
    "splitmix64",
]
