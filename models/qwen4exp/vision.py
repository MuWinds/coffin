"""视觉塔：patch 嵌入、轴向 RoPE、ViT 块栈、2×2 merger。

参考实现没有给视觉塔 dump fixture（八个 fixture 全是文本侧），所以这里的对齐
基准是 ``ARCH_SPEC.md`` §8 的逐条描述加 ``coffin/models/qwen35/vision.py`` 里
已有的等价实现；能自证的部分（段划分、merge 块顺序、两个 GELU 的不同、学习
位置表的抽头求和）在测试里用数值断言守住。

四个"形状一样、结果全错"的点：

* **merger 的 LayerNorm 在 2×2 合并之前**，作用在 1152 维上（``use_postshuffle_norm=False``）。
  先合并再归一化结果完全不同。
* **两个 GELU 不是同一个**：块内 MLP 是 ``gelu_pytorch_tanh``（近似），merger
  是 ``nn.GELU()``（精确 erf）。差 ~1e-3，形状查不出。
* **RoPE 不做 partial**：轴向版按 ``head_dim`` 对半分给 H/W，转满整个 head。
* **每图每帧一段**：段划分靠 ``cu_seqlens`` 而不是掩码，跨帧互见不会报错。

Module: ``coffin/models/qwen4exp``。依赖 ``config`` 与 ``rope``；不被文本塔
直接调用——图像特征由调用方按占位符顺序 scatter 进 ``inputs_embeds``。

Constraints:
    * 学习位置表的抽头求和必须在 fp32 里累加：bf16 下四行带权的和会错 ~1e-2，
      而且事后转换救不回来（``coffin/models/qwen35/vision.py:519-523`` 记录了
      这条实测）。
    * patch 嵌入是 stride == kernel 的 Conv3d，等价于对展平的 1536 维输入做一次
      GEMM；这里保留 Conv3d 以便逐行对照参考实现，换成 GEMM 时权重形状必须
      保持 ``[embed_dim, patch_dim]`` 的行主序。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from models.qwen4exp.config import VisionConfig
from models.qwen4exp.rope import VisionRotaryEmbedding, rotate_half


def gelu_pytorch_tanh(x: torch.Tensor) -> torch.Tensor:
    """块内 MLP 用的 tanh 近似 GELU。

    与 merger 里那个 ``nn.GELU()``（精确 erf）不是同一个函数，两者差约 1e-3。

    Args:
        x: 任意形状的输入。

    Returns:
        与 ``x`` 同形状。
    """
    return F.gelu(x, approximate="tanh")


def vision_interpolation_indices_weights(
    grid_thw: torch.Tensor, num_grid_per_side: int, spatial_merge_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """算出学习位置表的双线性抽头：``(indices, weights)``。

    表是 ``side x side`` 的方阵（``side = sqrt(num_position_embeddings)``），
    用 ``linspace(0, side - 1, size)``（即 ``align_corners=True``）重采样到每张
    图的 patch 网格，每根轴取 2 个抽头。返回的行序是 **merge 块顺序**——不是
    行主序：processor 发出的 ``pixel_values`` 就按 merge 块排，位置嵌入不跟着
    排会整体错位，而形状完全一致。

    Args:
        grid_thw: ``[n_img, 3]``，每行是 ``(t, h, w)`` 的 patch 数。
        num_grid_per_side: 学习位置表的边长。
        spatial_merge_size: 2×2 合并的边长。

    Returns:
        ``(indices, weights)``：都是 ``[N, 4]``，``N = sum(t*h*w)``。``indices``
        是表内的行号，``weights`` 是 fp32 双线性权重（沿 4 个抽头求和为 1）。

    Raises:
        ValueError: 某个网格边不是 ``spatial_merge_size`` 的倍数。2×2 合并会留下
            不完整的块，merger 折不了它的行——调用方该改图像尺寸，不是改网格。
    """
    merge = spatial_merge_size
    side = num_grid_per_side
    all_indices: list[torch.Tensor] = []
    all_weights: list[torch.Tensor] = []
    for frames, height, width in grid_thw.tolist():
        frames, height, width = int(frames), int(height), int(width)
        if height % merge or width % merge:
            raise ValueError(
                f"patch grid {height}x{width} is not a multiple of merge size {merge}.")
        hs = torch.linspace(0, side - 1, height, dtype=torch.float32)
        ws = torch.linspace(0, side - 1, width, dtype=torch.float32)
        h0 = hs.to(torch.int64)
        w0 = ws.to(torch.int64)
        # 最后一行的上抽头要夹回表内，否则 align_corners 的端点会越界一行。
        h1 = (h0 + 1).clamp(max=side - 1)
        w1 = (w0 + 1).clamp(max=side - 1)
        dh = hs - h0.to(torch.float32)
        dw = ws - w0.to(torch.float32)
        indices = torch.stack([
            h0[:, None] * side + w0[None, :],
            h0[:, None] * side + w1[None, :],
            h1[:, None] * side + w0[None, :],
            h1[:, None] * side + w1[None, :],
        ], dim=-1).reshape(-1, 4)
        weights = torch.stack([
            (1 - dh)[:, None] * (1 - dw)[None, :],
            (1 - dh)[:, None] * dw[None, :],
            dh[:, None] * (1 - dw)[None, :],
            dh[:, None] * dw[None, :],
        ], dim=-1).reshape(-1, 4)
        # 行主序 (h, w) -> merge 块顺序 (block_row, block_col, in_row, in_col)。
        order = (torch.arange(height * width).view(height // merge, merge, width // merge, merge)
                 .permute(0, 2, 1, 3).reshape(-1))
        indices = indices.index_select(0, order)
        weights = weights.index_select(0, order)
        if frames > 1:
            indices = indices.repeat(frames, 1)
            weights = weights.repeat(frames, 1)
        all_indices.append(indices)
        all_weights.append(weights)
    if not all_indices:
        empty = torch.zeros((0, 4), dtype=torch.int64)
        return empty, empty.to(torch.float32)
    return torch.cat(all_indices, dim=0), torch.cat(all_weights, dim=0)


def vision_position_ids(grid_thw: torch.Tensor, spatial_merge_size: int
                        ) -> torch.Tensor:
    """逐 token 的 ``(h, w)`` patch 坐标，按 merge 块顺序、逐帧排。

    轴向 RoPE 用这两行坐标各算一套频率：第 0 行给 H、第 1 行给 W。

    Args:
        grid_thw: ``[n_img, 3]`` 的 ``(t, h, w)``。
        spatial_merge_size: 2×2 合并的边长。

    Returns:
        ``[N, 2]`` 的 int64 坐标：每行一个 token 的 ``(h, w)``。布局是
        ``[N, 2]`` 而不是 ``[2, N]``，因为轴向 RoPE 的重组按 ``freq[:, 0]`` /
        ``freq[:, 1]`` 取两轴（见 :class:`~rope.VisionRotaryEmbedding`）。

    Raises:
        ValueError: 某个网格边不是 ``spatial_merge_size`` 的倍数。
    """
    merge = spatial_merge_size
    rows: list[torch.Tensor] = []
    cols: list[torch.Tensor] = []
    for frames, height, width in grid_thw.tolist():
        frames, height, width = int(frames), int(height), int(width)
        if height % merge or width % merge:
            raise ValueError(
                f"patch grid {height}x{width} is not a multiple of merge size {merge}.")
        # 先在行主序的 (h, w) 上建坐标，再套用与 patch 相同的 merge 块置换：
        # token 顺序必须与 pixel_values 一致，否则 RoPE 转的是别的 patch 的角度。
        grid_h = torch.arange(height).repeat_interleave(width)
        grid_w = torch.arange(width).repeat(height)
        order = (torch.arange(height * width)
                 .view(height // merge, merge, width // merge, merge)
                 .permute(0, 2, 1, 3).reshape(-1))
        full_h = grid_h.index_select(0, order)
        full_w = grid_w.index_select(0, order)
        if frames > 1:
            full_h = full_h.repeat(frames)
            full_w = full_w.repeat(frames)
        rows.append(full_h)
        cols.append(full_w)
    return torch.stack([torch.cat(rows), torch.cat(cols)], dim=-1)


def vision_attention_seqlens(grid_thw: torch.Tensor) -> tuple[torch.Tensor, int]:
    """算出段边界：**每图每帧一段**。

    跨帧互见不会抛任何异常，只是图错了，所以段划分必须显式给出而不是靠掩码。

    Args:
        grid_thw: ``[n_img, 3]`` 的 ``(t, h, w)``。

    Returns:
        ``(cu_seqlens, max_seqlen)``：``cu_seqlens`` 是长度 ``n_seg + 1`` 的前缀和。
    """
    lengths: list[int] = []
    for frames, height, width in grid_thw.tolist():
        lengths.extend([int(height) * int(width)] * int(frames))
    cu = torch.tensor([0] + list(torch.cumsum(torch.tensor(lengths), 0).tolist()),
                      dtype=torch.int64)
    return cu, (max(lengths) if lengths else 0)


class VisionRotary(nn.Module):
    """轴向 RoPE 的频率表包装，只为了和参考实现的模块边界一致。"""

    def __init__(self, config: VisionConfig) -> None:
        """按视觉配置建频率表。

        Args:
            config: 视觉配置。
        """
        super().__init__()
        self.rotary = VisionRotaryEmbedding(config)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """按坐标算 ``(cos, sin)``，形状 ``[N, head_dim]``。

        Args:
            position_ids: ``[2, N]``。

        Returns:
            ``(cos, sin)``，fp32。
        """
        return self.rotary.cos_sin(position_ids)


class PatchEmbed(nn.Module):
    """把展平的 patch 投到 ViT 宽度。

    ``Conv3d(kernel=stride=(temporal, patch, patch))`` 无重叠，数学上等价于对
    展平的 ``patch_dim`` 维输入做一次 GEMM 加 bias。
    """

    def __init__(self, config: VisionConfig) -> None:
        """建 Conv3d。

        Args:
            config: 视觉配置。
        """
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        kernel = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel,
                              stride=kernel, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``[N, patch_dim] -> [N, embed_dim]``。

        Args:
            hidden_states: ``[N, in_channels * temporal * patch * patch]``。

        Returns:
            ``[N, embed_dim]``。

        Raises:
            ValueError: 输入宽度不是 ``patch_dim``。
        """
        expected = self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
        if hidden_states.shape[-1] != expected:
            raise ValueError(
                f"patch embed expects {expected}-wide rows, got {hidden_states.shape[-1]}.")
        target = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size)
        return self.proj(hidden_states.to(target)).view(-1, self.embed_dim)


class VisionMLP(nn.Module):
    """块内 MLP：``fc2(gelu_pytorch_tanh(fc1(x)))``，两层都有 bias。"""

    def __init__(self, config: VisionConfig) -> None:
        """建两个线性层。

        Args:
            config: 视觉配置。
        """
        super().__init__()
        self.linear_fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """跑 MLP。

        Args:
            hidden_state: ``[N, hidden]``。

        Returns:
            ``[N, hidden]``。
        """
        return self.linear_fc2(gelu_pytorch_tanh(self.linear_fc1(hidden_state)))


class VisionAttention(nn.Module):
    """无 GQA、非因果、按段隔离的注意力。

    16 个 Q 头对 16 个 KV 头（``num_key_value_groups = 1``）；因果性由段划分
    提供而不是掩码，所以这里对每段各自做一次完整注意力。
    """

    def __init__(self, config: VisionConfig) -> None:
        """建融合 qkv 与输出投影（都有 bias）。

        Args:
            config: 视觉配置。
        """
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim, bias=True)

    def forward(self, hidden_states: torch.Tensor,
                cu_seqlens: torch.Tensor,
                position_embeddings: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """跑一次视觉注意力。

        Args:
            hidden_states: ``[N, hidden]``。
            cu_seqlens: 段边界前缀和。
            position_embeddings: ``(cos, sin)``，``[N, head_dim]``，fp32。

        Returns:
            ``[N, hidden]``。

        Raises:
            ValueError: 段边界的最后一项不等于 ``N``。
        """
        seq_length = hidden_states.shape[0]
        if int(cu_seqlens[-1]) != seq_length:
            raise ValueError(
                f"cu_seqlens covers {int(cu_seqlens[-1])} tokens but got {seq_length}.")
        query, key, value = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, self.head_dim)
            .permute(1, 0, 2, 3).unbind(0))
        cos, sin = position_embeddings
        # RoPE 在 fp32 里做再转回：视觉塔的激活量级比文本侧小，低精度旋转会
        # 留下 ~1e-3 的固定偏差。
        orig_dtype = query.dtype
        q32, k32 = query.float(), key.float()
        cos = cos.unsqueeze(-2).float() if cos.dim() == 2 else cos.float()
        sin = sin.unsqueeze(-2).float() if sin.dim() == 2 else sin.float()
        query = ((q32 * cos) + (rotate_half(q32) * sin)).to(orig_dtype)
        key = ((k32 * cos) + (rotate_half(k32) * sin)).to(orig_dtype)

        query = query.transpose(0, 1).unsqueeze(0)
        key = key.transpose(0, 1).unsqueeze(0)
        value = value.transpose(0, 1).unsqueeze(0)
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        q_parts = torch.split(query, lengths, dim=2)
        k_parts = torch.split(key, lengths, dim=2)
        v_parts = torch.split(value, lengths, dim=2)
        outputs = []
        for q_part, k_part, v_part in zip(q_parts, k_parts, v_parts):
            weights = torch.matmul(q_part, k_part.transpose(2, 3)) * self.scaling
            weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(q_part.dtype)
            outputs.append(torch.matmul(weights, v_part))
        attn_output = torch.cat(outputs, dim=2)
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        return self.proj(attn_output)


class VisionBlock(nn.Module):
    """``x = x + attn(norm1(x))``、``x = x + mlp(norm2(x))``。

    两个 norm 都是标准 ``nn.LayerNorm``（eps=1e-6），**不是**文本侧的零中心
    RMSNorm，也没有门。
    """

    def __init__(self, config: VisionConfig) -> None:
        """建两个 LayerNorm、注意力与 MLP。

        Args:
            config: 视觉配置。
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = VisionAttention(config)
        self.mlp = VisionMLP(config)

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor,
                position_embeddings: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """跑一个 ViT 块。

        Args:
            hidden_states: ``[N, hidden]``。
            cu_seqlens: 段边界。
            position_embeddings: ``(cos, sin)``。

        Returns:
            ``[N, hidden]``。
        """
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings)
        return hidden_states + self.mlp(self.norm2(hidden_states))


class PatchMerger(nn.Module):
    """2×2 合并到文本宽度：``fc2(gelu(fc1(merge(norm(x)))))``。

    ``use_postshuffle_norm=False`` 时 LayerNorm 作用在**合并之前**的
    ``hidden_size`` 维上；先合并再归一化结果完全不同，而形状一模一样。
    """

    def __init__(self, config: VisionConfig, use_postshuffle_norm: bool = False) -> None:
        """建 LayerNorm 与两个线性层。

        Args:
            config: 视觉配置。
            use_postshuffle_norm: 是否在合并之后归一化。参考实现固定 ``False``，
                参数保留是为了让"这个开关存在"这件事显式。
        """
        super().__init__()
        self.hidden_size = config.hidden_size * config.merge_unit
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(
            self.hidden_size if use_postshuffle_norm else config.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        # merger 的 GELU 是精确 erf 版，与块内的 tanh 近似版不是同一个函数。
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """把 ``[N, hidden]`` 折成 ``[N / merge_unit, out_hidden]``。

        Args:
            x: ``[N, hidden]``。

        Returns:
            ``[N / merge_unit, out_hidden_size]``。
        """
        if self.use_postshuffle_norm:
            x = self.norm(x.view(-1, self.hidden_size))
        else:
            x = self.norm(x)
        return self.linear_fc2(self.act_fn(self.linear_fc1(x.view(-1, self.hidden_size))))


class VisionModel(nn.Module):
    """完整的视觉塔。

    Attributes:
        patch_embed: patch 投影。
        pos_embed: 学习位置表。
        rotary_pos_emb: 轴向 RoPE。
        blocks: ``depth`` 个 ViT 块。
        merger: 2×2 合并到文本宽度。
    """

    def __init__(self, config: VisionConfig) -> None:
        """建全部子模块。

        Args:
            config: 视觉配置。
        """
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.spatial_merge_unit = config.merge_unit
        self.patch_size = config.patch_size
        self.num_grid_per_side = config.num_grid_per_side
        self.patch_embed = PatchEmbed(config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.rotary_pos_emb = VisionRotary(config)
        self.blocks = nn.ModuleList([VisionBlock(config) for _ in range(config.depth)])
        self.merger = PatchMerger(config, use_postshuffle_norm=False)

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """跑视觉塔。

        Args:
            hidden_states: ``[N, patch_dim]``，行按 merge 块顺序、逐图逐帧排。
            grid_thw: ``[n_img, 3]`` 的 ``(t, h, w)``。

        Returns:
            ``(last_hidden_state, pooler_output)``：前者是 merger **之前**的
            结果（别拿错：并回文本流的是后者），形状 ``[N, hidden_size]`` 与
            ``[N / merge_unit, out_hidden_size]``。

        Raises:
            ValueError: 网格与输入行数对不上。
        """
        interp_indices, interp_weights = vision_interpolation_indices_weights(
            grid_thw, self.num_grid_per_side, self.spatial_merge_size)
        position_ids = vision_position_ids(grid_thw, self.spatial_merge_size)
        cu_seqlens, _ = vision_attention_seqlens(grid_thw)

        expected = int(torch.as_tensor(grid_thw).prod(dim=-1).sum())
        if hidden_states.shape[0] != expected:
            raise ValueError(
                f"grid_thw covers {expected} patches but got {hidden_states.shape[0]} rows.")

        hidden_states = self.patch_embed(hidden_states)
        # 抽头求和全程 fp32：bf16 下四行带权和会错 ~1e-2，事后转换救不回来。
        pos_embeds = (self.pos_embed(interp_indices).float()
                      * interp_weights[:, :, None]).sum(1)
        hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype)
        position_embeddings = self.rotary_pos_emb(position_ids)

        for block in self.blocks:
            hidden_states = block(hidden_states, cu_seqlens=cu_seqlens,
                                 position_embeddings=position_embeddings)
        return hidden_states, self.merger(hidden_states)


def scatter_image_embeds(inputs_embeds: torch.Tensor, input_ids: torch.Tensor,
                         image_embeds: torch.Tensor, image_token_id: int,
                         video_token_id: int | None = None,
                         video_embeds: torch.Tensor | None = None) -> torch.Tensor:
    """把图像/视频特征按占位符逐行覆写进文本嵌入。

    顺序契约：``image_embeds`` 的行序必须**完全等于** prompt 里 ``<|image_pad|>``
    的顺序，即每图按合并后网格的行主序。搞错顺序会得到一张"看起来合理但不对应"
    的图，而形状检查发现不了。

    Args:
        inputs_embeds: ``[B, S, hidden]``。
        input_ids: ``[B, S]``，用来定位占位符。
        image_embeds: ``[n_image_tokens, hidden]``。
        image_token_id: 图像占位符 token。
        video_token_id: 视频占位符 token；给了才处理视频。
        video_embeds: ``[n_video_tokens, hidden]``。

    Returns:
        覆写后的新张量（不原地改入参，调用方可能还要用原嵌入）。

    Raises:
        ValueError: 占位符个数乘 hidden 与特征元素数对不上。少了会被静默截断，
            多出来的行没人接收，两种都必须炸。
    """
    out = inputs_embeds.clone()
    image_mask = input_ids == image_token_id
    count = int(image_mask.sum())
    if count * inputs_embeds.shape[-1] != image_embeds.numel():
        raise ValueError(
            f"image features and image tokens do not match: {count} tokens vs "
            f"{image_embeds.shape[0]} rows.")
    out = out.masked_scatter(image_mask.unsqueeze(-1).expand_as(out),
                             image_embeds.to(out.device, out.dtype))
    if video_token_id is not None and video_embeds is not None:
        video_mask = input_ids == video_token_id
        video_count = int(video_mask.sum())
        if video_count * inputs_embeds.shape[-1] != video_embeds.numel():
            raise ValueError(
                f"video features and video tokens do not match: {video_count} tokens vs "
                f"{video_embeds.shape[0]} rows.")
        out = out.masked_scatter(video_mask.unsqueeze(-1).expand_as(out),
                                 video_embeds.to(out.device, out.dtype))
    return out


__all__ = [
    "PatchEmbed",
    "PatchMerger",
    "VisionAttention",
    "VisionBlock",
    "VisionMLP",
    "VisionModel",
    "scatter_image_embeds",
    "vision_attention_seqlens",
    "vision_interpolation_indices_weights",
    "vision_position_ids",
]
