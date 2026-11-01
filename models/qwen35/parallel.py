"""按注意力头和 MLP 通道切分 Qwen3.5 的 W4 权重与状态。

Module: ``coffin/models/qwen35``。权重加载器和模型组装调用此模块；算子通过
CoreX 整数投影完成，跨 rank 数据仅通过 Gloo 的 CPU fp32 张量交换。

Constraints:
    * 每个进程只暴露一张健康 GPU，使既有设备分配和 KV/循环状态自然局部化。
    * K 轴分片必须落在量化组边界；q/k/v 和卷积通道要分别切分后重新拼接。
    * 行并行默认共享激活 A8 scale；诊断可改用 rank 本地 scale，部分和仍用 fp32 归约。
    * 调度及前缀缓存操作须在各 rank 以相同顺序执行，不允许 CUDA graph。
"""

from __future__ import annotations

from collections import OrderedDict
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from w4a8_weights import W4A8Weight


class ParallelContext:
    """一个 rank 的主机通信上下文，进程组由启动器创建。"""

    def __init__(self, rank: int, world_size: int) -> None:
        """核对进程组身份，防止按错误的 rank 读取权重。

        Args:
            rank: 当前进程编号。
            world_size: 并行进程数。

        Raises:
            RuntimeError: Gloo 进程组尚未初始化或身份不符。
        """
        if (not dist.is_initialized() or dist.get_backend() != "gloo"
                or dist.get_world_size() != world_size or dist.get_rank() != rank):
            raise RuntimeError("tensor parallelism requires a matching CPU Gloo group")
        self.rank, self.world_size = rank, world_size
        self._buffers = OrderedDict()
        self.sum_calls = 0
        self.max_calls = 0
        self.scale_mode = os.environ.get("COFFIN_TP_SCALE_MODE", "shared")
        if self.scale_mode not in {"shared", "local", "decode_local"}:
            raise ValueError(
                "COFFIN_TP_SCALE_MODE must be 'shared', 'local', or 'decode_local'")

    def reduce(self, tensor: torch.Tensor, *, maximum: bool = False) -> torch.Tensor:
        """在 CPU 上以 fp32 做全归约，回传本 rank 当前流。

        Args:
            tensor: 本 rank 的浮点输入。
            maximum: 求最大值；否则求和。

        Returns:
            与输入同设备、同形状的 fp32 结果。
        """
        key = (tuple(tensor.shape), maximum)
        entry = self._buffers.pop(key, None)
        if entry is None:
            if len(self._buffers) >= 4:
                self._buffers.popitem(last=False)
            entry = torch.empty(tensor.shape, dtype=torch.float32, pin_memory=True)
        self._buffers[key] = entry
        host = entry
        host.copy_(tensor)
        dist.all_reduce(host, op=dist.ReduceOp.MAX if maximum else dist.ReduceOp.SUM)
        result = host.to(tensor.device)
        if maximum:
            self.max_calls += 1
        else:
            self.sum_calls += 1
        return result


_context: ParallelContext | None = None


def configure(rank: int, world_size: int) -> ParallelContext:
    """在加载模型前绑定已初始化的 Gloo 进程组。

    Args:
        rank: 当前进程编号。
        world_size: 并行进程数。

    Returns:
        此进程的上下文，之后所有层共享它。

    Raises:
        RuntimeError: 进程组不匹配，或已经绑定了上下文。
    """
    global _context
    if _context is not None:
        raise RuntimeError("tensor parallel context is already configured")
    _context = ParallelContext(rank, world_size)
    return _context


def context() -> ParallelContext | None:
    """未配置时返回 None，单卡模型保持原有组装路径。"""
    return _context


def local_config(cfg: dict, world_size: int) -> dict:
    """把注意力头数和中间通道变成局部尺寸，隐藏宽度保持全局尺寸。

    Args:
        cfg: checkpoint 文本配置。
        world_size: 分片数。

    Returns:
        独立的局部配置，不修改 checkpoint 配置。

    Raises:
        ValueError: 任意头数或中间通道不能整除并行数。
    """
    if world_size < 1:
        raise ValueError("positive world_size required")
    result = dict(cfg)
    for key in ("num_attention_heads", "num_key_value_heads", "linear_num_key_heads",
                "linear_num_value_heads", "intermediate_size"):
        value = int(cfg[key])
        if value % world_size:
            raise ValueError(f"{key}={value} is not divisible by world_size={world_size}")
        result[key] = value // world_size
    return result


def _ranges(widths: list[int], rank: int, world_size: int) -> list[tuple[int, int]]:
    """分别切分融合矩阵各段，避免把 q 头当作 v 头分给其他 rank。"""
    result = []
    offset = 0
    for width in widths:
        if width % world_size:
            raise ValueError(f"partition width {width} not divisible by {world_size}")
        size = width // world_size
        result.append((offset + rank * size, offset + (rank + 1) * size))
        offset += width
    return result


def shard_weight(key: str, value, cfg: dict, rank: int, world_size: int):
    """在 CPU 上切一项权重，完整矩阵不会先搬到 GPU。

    Args:
        key: checkpoint 参数名。
        value: 稠密张量或 packed W4 权重。
        cfg: 全局文本配置。
        rank: 分片编号。
        world_size: 分片数。

    Returns:
        本 rank 的连续张量或 W4 对象；共享参数原样返回。

    Raises:
        ValueError: rank、融合段宽度或 W4 K 轴边界不满足分片契约。
    """
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("invalid rank/world_size")
    axis, ranges = None, None
    shape = value.shape
    if ".linear_attn." in key and key.endswith("in_proj_qkv.weight"):
        kd = cfg["linear_num_key_heads"] * cfg["linear_key_head_dim"]
        vd = cfg["linear_num_value_heads"] * cfg["linear_value_head_dim"]
        axis, ranges = 0, _ranges([kd, kd, vd], rank, world_size)
        if shape[0] != 2 * kd + vd:
            raise ValueError(f"{key}: fused q/k/v shape mismatch")
    elif key.endswith((".mlp.down_proj.weight", ".self_attn.o_proj.weight",
                       ".linear_attn.out_proj.weight", ".fc.weight")):
        axis, ranges = 1, _ranges([shape[1]], rank, world_size)
    elif key.endswith(".linear_attn.conv1d.weight"):
        kd = cfg["linear_num_key_heads"] * cfg["linear_key_head_dim"]
        vd = cfg["linear_num_value_heads"] * cfg["linear_value_head_dim"]
        ranges = _ranges([kd, kd, vd], rank, world_size)
        axis = 0
    elif key.endswith((".mlp.gate_proj.weight", ".mlp.up_proj.weight",
                       ".self_attn.q_proj.weight", ".self_attn.k_proj.weight",
                       ".self_attn.v_proj.weight", ".linear_attn.in_proj_z.weight",
                       ".linear_attn.in_proj_a.weight", ".linear_attn.in_proj_b.weight",
                       ".linear_attn.A_log", ".linear_attn.dt_bias")):
        axis, ranges = 0, _ranges([shape[0]], rank, world_size)
    if axis is None:
        return value
    if not isinstance(value, W4A8Weight):
        return torch.cat([value.narrow(axis, a, b - a) for a, b in ranges], dim=axis).contiguous()
    if axis == 0:
        packed = torch.cat([value.packed[:, a:b] for a, b in ranges], dim=1).contiguous()
        scale = torch.cat([value.scale[:, a:b] for a, b in ranges], dim=1).contiguous()
        return W4A8Weight(packed, scale, k=value.k, group_size=value.group_size)
    start, stop = ranges[0]
    gs = value.group_size
    if start % gs or start % 2 or (stop != value.k and stop % gs):
        raise ValueError(f"{key}: W4 input partition must align to groups and packed pairs")
    return W4A8Weight(value.packed[start // 2:(stop + 1) // 2].contiguous(),
                       value.scale[start // gs:(stop + gs - 1) // gs].contiguous(),
                       k=stop - start, group_size=gs)


def load_weights(model_dir: str | Path, device: torch.device,
                 parallel: ParallelContext, vision: bool | None = None) -> dict:
    """逐项读取并分片离线 W4 权重，只把本 rank 的部分搬到设备。

    Args:
        model_dir: 带 W4 索引的模型目录。
        device: 本进程唯一可见的计算设备。
        parallel: 已配置的并行上下文。
        vision: 是否读视觉塔；None 沿用索引声明。

    Returns:
        与既有加载器相同键名的局部 state dict。

    Raises:
        ValueError: 格式、配置或张量布局不匹配。
        RuntimeError: 调用方要求视觉塔，但产物没有保存它。
    """
    from safetensors import safe_open
    from quantized_load import W4A8_FORMAT, _check_source_config, read_index

    model_dir = Path(model_dir)
    index = read_index(model_dir)
    if index["format"] != W4A8_FORMAT:
        raise ValueError("tensor parallel loader currently requires packed W4 checkpoint")
    _check_source_config(model_dir, index)
    cfg = json.loads((model_dir / "config.json").read_text())["text_config"]
    local_config(cfg, parallel.world_size)
    take_vision = index.get("vision", False) if vision is None else vision
    if take_vision and not index.get("vision"):
        raise RuntimeError("checkpoint has no vision tensors")
    entries = {k: v for k, v in index["tensors"].items()
               if k.startswith(("model.language_model.", "mtp.", "lm_head."))
               or (take_vision and k.startswith("model.visual."))}
    sd = {}
    for shard in sorted({v["shard"] for v in entries.values()}):
        with safe_open(str(model_dir / shard), framework="pt", device="cpu") as handle:
            for key, meta in entries.items():
                if meta["shard"] != shard:
                    continue
                value = handle.get_tensor(key)
                if meta["kind"] == "q4":
                    scale = handle.get_tensor(meta["scale_key"])
                    if list(value.shape) != meta["packed_shape"] or list(scale.shape) != meta["scale_shape"]:
                        raise ValueError(f"{key}: packed checkpoint shape mismatch")
                    value = W4A8Weight(value, scale, k=meta["k"], group_size=meta["group_size"])
                value = shard_weight(key, value, cfg, parallel.rank, parallel.world_size)
                if isinstance(value, W4A8Weight):
                    value = W4A8Weight(value.packed.to(device), value.scale.to(device),
                                       k=value.k, group_size=value.group_size)
                else:
                    value = value.to(device)
                sd[key] = value
    return sd


class RowParallelProjection:
    """归约局部 K 点积，向上层保持原有投影接口。"""

    def __init__(self, projection, parallel: ParallelContext) -> None:
        """持有局部投影及对应通信上下文。

        Args:
            projection: 从本 rank 权重构造的 GEMV。
            parallel: 模型共享的进程上下文。

        Raises:
            ValueError: 投影没有 packed W4 权重。
        """
        if not isinstance(projection.w, W4A8Weight):
            raise ValueError("row parallel projection requires packed W4 weights")
        self.w, self.N, self.K = projection.w, projection.N, projection.K
        self.parallel = parallel

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """按上下文选择激活 scale，以 fp32 归约局部整数点积后窄化。"""
        from backends.corex import _w4_rows

        if x.dim() not in (1, 2):
            raise ValueError("row parallel activation shape mismatch")
        if x.shape[-1] != self.K:
            if x.shape[-1] != self.K * self.parallel.world_size:
                raise ValueError("row parallel input width does not match shard count")
            start = self.parallel.rank * self.K
            x = x[..., start:start + self.K].contiguous()
        local_max = x.reshape(-1, self.K).float().abs().amax(-1)
        mode = self.parallel.scale_mode
        if mode not in {"shared", "local", "decode_local"}:
            raise ValueError(f"unknown activation scale mode: {self.parallel.scale_mode}")
        use_local_scale = mode == "local" or (mode == "decode_local" and x.numel() == self.K)
        scale_max = local_max if use_local_scale else self.parallel.reduce(
            local_max, maximum=True)
        scale = scale_max.clamp_min_(1e-5).div_(127)
        partial = _w4_rows(self.w, x.float(), activation_scale=scale)
        return self.parallel.reduce(partial).to(x.dtype)

    def linear_m(self, x: torch.Tensor) -> torch.Tensor:
        """批处理沿用同一全局量化和归约契约。"""
        return self(x)

    def prefill(self, x: torch.Tensor) -> torch.Tensor:
        """预填充由 CoreX 按形状选择直接消费 packed W4 的张量核。"""
        return self(x)


def row_projection(projection):
    """仅在配置张量并行时给行投影添加归约。"""
    parallel = context()
    return projection if parallel is None else RowParallelProjection(projection, parallel)
