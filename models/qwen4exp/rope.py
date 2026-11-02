"""RoPE：文本侧的 partial rotary 加 mrope 重组，以及视觉侧的轴向版。

文本 RoPE 有三个不能凭形状检查发现的坑：

* ``rotate_half`` 必须是**分半式** ``cat((-x2, x1), -1)``，不是交错式
  ``(-x1, x0)``。交错式得到的仍是单位模旋转，图只是错的。
* 只旋转前 ``rotary_dim = head_dim * partial_rotary_factor`` 维，其余原样拼接。
* mrope 重组按**频率的一维下标**取：``i % 3 == 0`` 用 T 轴，``1`` 用 H 轴、
  ``2`` 用 W 轴。纯文本三轴相同，重组退化成恒等。

Module: ``coffin/models/qwen4exp``。只依赖 ``torch`` 与 ``config``；被
``qsa``、``blocks``、``model`` 和 ``vision`` 调用。

Constraints:
    * ``position_ids`` 是**全表**：带 cache 时它包含历史位置，而 q/k 只有当前
      chunk。调用方自己切，本模块不做"猜你要哪一段"。
    * cos/sin 的 dtype 跟随输入激活，但频率表本身保持 fp32：位置 × 频率在 fp32
      里算，再 cast。
"""

from __future__ import annotations

import torch

from models.qwen4exp.config import TextConfig, VisionConfig


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """把最后一维对半分，返回 ``cat((-x2, x1), -1)``。

    参考实现用的是 HF 的分半形式，与"奇数维取负、偶数维平移"的交错形式不是同一
    个变换：两者都是单位模旋转，形状也一样，只有角度图不同。

    Args:
        x: 最后一维长度为偶数。

    Returns:
        与 ``x`` 同形状。
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                         k: torch.Tensor | None = None,
                         unsqueeze_dim: int = 1) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """对 ``q``（以及可选的 ``k``）施加 rotary。

    只有前 ``cos.shape[-1]`` 维参与旋转，其余原样拼接——partial rotary 的实现
    就是这么直接，不需要额外的掩码。

    Args:
        q: 查询，最后一维是 ``head_dim``。
        cos: 余弦表，最后一维是 ``rotary_dim``；形状是
            ``[B, S, rotary_dim]`` 或 ``[B, S, H, rotary_dim]`` 去掉头维后的样子。
        sin: 正弦表，形状与 ``cos`` 相同。
        k: 可选的键；给了就一起旋转。
        unsqueeze_dim: 在 ``cos``/``sin`` 的哪一维插一个新轴，用来对齐 ``q`` 的
            布局。文本注意力是 ``[B, H, S, D]`` 所以填 1，索引器是
            ``[B, S, H, D]`` 所以填 2。填错不会报错——广播会让头和序列轴互换，
            图就错了。

    Returns:
        只给 ``q`` 时返回旋转后的 ``q``；给了 ``k`` 时返回 ``(q, k)``。

    Raises:
        ValueError: ``cos`` 与 ``sin`` 形状不一致，或旋转维是奇数 / 超过
            ``head_dim``。
    """
    if cos.shape != sin.shape:
        raise ValueError(f"cos/sin shape mismatch: {tuple(cos.shape)} vs {tuple(sin.shape)}.")
    rotary_dim = cos.shape[-1]
    if rotary_dim % 2 != 0:
        raise ValueError(f"rotary dim must be even, got {rotary_dim}.")
    if rotary_dim > q.shape[-1]:
        raise ValueError(
            f"rotary dim {rotary_dim} exceeds the head dim {q.shape[-1]}.")
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_rope = (q[..., :rotary_dim] * cos) + (rotate_half(q[..., :rotary_dim]) * sin)
    q_rotated = torch.cat([q_rope, q[..., rotary_dim:]], dim=-1)
    if k is None:
        return q_rotated
    k_rope = (k[..., :rotary_dim] * cos) + (rotate_half(k[..., :rotary_dim]) * sin)
    return q_rotated, torch.cat([k_rope, k[..., rotary_dim:]], dim=-1)


class RotaryEmbedding:
    """partial rotary + mrope 重组的频率表。

    Attributes:
        inv_freq: ``[rotary_dim / 2]`` 的逆频率，fp32 buffer。
        mrope_section: 三段分节，默认 ``[11, 11, 10]``。
        attention_scaling: 非默认 rope 类型给的比例因子，``default`` 恒为 1。
    """

    def __init__(self, config: TextConfig) -> None:
        """按 ``rope_theta`` / ``partial_rotary_factor`` 建频率表。

        Args:
            config: 文本配置。

        Raises:
            ValueError: 旋转维是奇数，或 rope 类型不是 ``default``（其余类型要
                额外的缩放函数，引擎尚未实现，静默退回 ``default`` 会算错图）。
        """
        if config.rope_type != "default":
            raise ValueError(
                f"unsupported rope_type {config.rope_type!r}; only 'default' is implemented.")
        dim = config.rotary_dim
        if dim % 2 != 0:
            raise ValueError(f"rotary dim must be even, got {dim}.")
        self.rotary_dim = dim
        self.base = config.rope_theta
        self.attention_scaling = 1.0
        self.mrope_section = list(config.mrope_section)
        # arange(0, dim, 2) / dim 给 dim/2 个频率，与 HF 的 default rope 一致。
        #
        # ``device="cpu"`` 是必须的：模型可能在 ``torch.device("meta")`` 上下文里
        # 构造（用 meta 建模型、装载时再实体化，48 层才装得下），而 ``inv_freq``
        # 是**普通属性**不是 buffer，``to_empty`` / ``.to`` 都不会重建它。留在 meta
        # 上会在第一次前向时报 ``Cannot copy out of meta tensor``，而报错点在
        # ``cos_sin``，与"频率表没建出来"隔了一层。
        inv_freq = 1.0 / (self.base ** (
            torch.arange(0, dim, 2, dtype=torch.float32, device="cpu") / dim))
        self.inv_freq = inv_freq

    def recomposition_frequencies(self, freq: torch.Tensor) -> torch.Tensor:
        """把 ``(3, B, T, nfreq)`` 的三轴频率按 mrope 分节重组成一轴。

        Args:
            freq: 三轴频率。

        Returns:
            ``[B, T, 2 * nfreq]``：重排后的一轴频率在最后一维复制一份，这样配合
                ``rotate_half`` 的"``i`` 与 ``i + rotary_dim/2`` 配对"就得到标准
                partial-RoPE 布局。
        """
        freqs_thw = freq[0]  # 以 T 轴为底稿
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = self.mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_thw[..., idx] = freq[dim, ..., idx]
        return torch.cat((freqs_thw, freqs_thw), dim=-1)

    def cos_sin(self, position_ids: torch.Tensor,
                dtype: torch.dtype | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """按位置表算 (cos, sin)。

        Args:
            position_ids: ``[3, B, T]``（或可广播成这个形状）。纯文本时三行可以
                相同。
            dtype: 输出的 dtype；``None`` 保持 fp32。

        Returns:
            ``(cos, sin)``，形状 ``[B, T, rotary_dim]``。
        """
        position_ids = position_ids.expand(3, -1, -1)
        freqs = position_ids[..., None].float() * self.inv_freq.to(position_ids.device)
        cos = freqs.cos() * self.attention_scaling
        sin = freqs.sin() * self.attention_scaling
        sin = self.recomposition_frequencies(sin)
        cos = self.recomposition_frequencies(cos)
        if dtype is not None:
            return cos.to(dtype), sin.to(dtype)
        return cos, sin


class VisionRotaryEmbedding:
    """视觉塔的轴向 2D RoPE，旋转覆盖整个 head_dim。

    ``head_dim`` 对半分给 H 和 W，各自用 ``head_dim/4`` 个频率；重组后 cos 是
    ``[cos_h(n) | cos_w(n) | cos_h(n) | cos_w(n)]``（``n = head_dim/4``）。配合
    分半式 ``rotate_half`` 的「``i`` 与 ``i + head_dim/2`` 配对」，第 ``i`` 个二维
    平面恰好由第 ``i`` 个频率驱动：H 管第 0、2 段，W 管第 1、3 段。

    注意这**不是**「前半 H、后半 W」——实测（``tests/test_qwen4exp_vision.py``）就是
    四段交错，按前后半去核对会得到一张角度全错的图。

    坐标表的布局是 ``[N, 2]``（每行一个 token 的 ``(h, w)``），**不是**
    ``[2, N]``：重组时按 ``freq[:, 0]`` / ``freq[:, 1]`` 取两轴，也就是要求轴 1
    长度为 2。``ARCH_SPEC.md`` §8.3 和参考实现的注释都写的是 ``(2, N)``，但参考
    实现自己的 ``recomposition_frequencies`` 在 ``(2, N, nfreq)`` 上取到的是
    "第 0 个 token 的两个轴"，随后 ``cat(freq_hw, freq_hw)`` 出 ``[2, head_dim]``，
    广播到 ``q[N, H, D]`` 不成立。以代码为准：``[N, 2]``。

    Attributes:
        inv_freq: ``[head_dim / 4]`` 的逆频率。
    """

    def __init__(self, config: VisionConfig) -> None:
        """按视觉配置建频率表。

        Args:
            config: 视觉配置。
        """
        self.head_dim = config.head_dim
        self.spatial_dim = config.spatial_rotary_dim
        self.base = config.rope_theta
        self.attention_scaling = 1.0
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.spatial_dim, 2, dtype=torch.float32) / self.spatial_dim))
        self.inv_freq = inv_freq

    def recomposition_frequencies(self, freq: torch.Tensor) -> torch.Tensor:
        """把 ``(N, 2, nfreq)`` 的 H/W 频率交叠成 ``[N, 4 * nfreq]``。

        Args:
            freq: 每行一个 token，轴 1 的两项分别是 h 坐标与 w 坐标。

        Returns:
            ``[N, head_dim]`` 的频率表。

        Raises:
            ValueError: 轴 1 的长度不是 2。传 ``(2, N, nfreq)`` 进来会在这里被
                拒绝，而不是静默广播成 ``[2, 72]`` 再在 q 上炸。
        """
        if freq.dim() != 3 or freq.shape[1] != 2:
            raise ValueError(
                f"vision rope expects [N, 2, nfreq] position table, got {tuple(freq.shape)}.")
        freq_h, freq_w = freq[:, 0], freq[:, 1]
        freq_hw = torch.cat([freq_h, freq_w], dim=-1)
        return torch.cat([freq_hw, freq_hw], dim=-1)

    def cos_sin(self, position_ids: torch.Tensor,
                dtype: torch.dtype | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """按 ``[N, 2]`` 的 (h, w) 坐标算 (cos, sin)。

        Args:
            position_ids: ``[N, 2]``，每行一个 token 的 ``(h, w)``。
            dtype: 输出 dtype；``None`` 保持 fp32。

        Returns:
            ``(cos, sin)``，形状 ``[N, head_dim]``。
        """
        freqs = position_ids[..., None].float() * self.inv_freq.to(position_ids.device)
        cos = self.recomposition_frequencies(freqs.cos() * self.attention_scaling)
        sin = self.recomposition_frequencies(freqs.sin() * self.attention_scaling)
        if dtype is not None:
            return cos.to(dtype), sin.to(dtype)
        return cos, sin


__all__ = [
    "RotaryEmbedding",
    "VisionRotaryEmbedding",
    "apply_rotary_pos_emb",
    "rotate_half",
]
