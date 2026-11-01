"""保存并解释分组对称 W4 投影权重。

Module: ``coffin/``。由 ``quantized_load`` 构造，再交给投影算子；这里只定义
磁盘/设备侧权重的逻辑形状与解包参考，不依赖模型或后端。

Constraints:
    * 偶数 K 的低 nibble、奇数 K 的高 nibble，各存储 ``q + 8``；每组沿 K
      使用一个逐输出通道 fp32 scale。布局来自 ``quant_disk_w4a8.py``。
    * 逻辑权重为 ``[N, K]``，磁盘 packed 张量为 ``[ceil(K/2), N]``。加载时校验
      这两种形状关系，避免方阵或偶数 K 掩盖转置错误。
"""

from __future__ import annotations

import torch


class W4A8Weight:
    """持有分组 W4 权重及其逐组 scale。"""

    __slots__ = ("group_size", "k", "n", "packed", "scale")

    def __init__(self, packed: torch.Tensor, scale: torch.Tensor, *,
                 k: int, group_size: int = 128) -> None:
        if packed.dim() != 2 or packed.dtype != torch.uint8 or not packed.is_contiguous():
            raise ValueError("packed weight must be contiguous 2-D uint8 [ceil(K/2), N]")
        if scale.dim() != 2 or scale.dtype != torch.float32 or not scale.is_contiguous():
            raise ValueError("scale must be contiguous 2-D fp32 [groups, N]")
        if k <= 0 or group_size <= 0:
            raise ValueError("K and group_size must be positive")
        n = int(packed.shape[1])
        groups = (k + group_size - 1) // group_size
        if packed.shape[0] * 2 < k or tuple(scale.shape) != (groups, n):
            raise ValueError("packed weight or scale shape does not match K/group_size")
        self.packed = packed
        self.scale = scale
        self.k = int(k)
        self.n = n
        self.group_size = int(group_size)

    def nbytes(self) -> int:
        return self.packed.numel() + self.scale.numel() * self.scale.element_size()

    @property
    def shape(self) -> tuple[int, int]:
        return self.n, self.k

    @classmethod
    def cat_output(cls, weights: list["W4A8Weight"]) -> "W4A8Weight":
        first = weights[0]
        if any(w.k != first.k or w.group_size != first.group_size
               or w.packed.device != first.packed.device for w in weights):
            raise ValueError("W4A8 weights must share K, group_size, and device")
        packed = torch.cat([w.packed for w in weights], dim=1).contiguous()
        scale = torch.cat([w.scale for w in weights], dim=1).contiguous()
        return cls(packed, scale, k=first.k, group_size=first.group_size)

    def rows(self, ids: torch.Tensor) -> torch.Tensor:
        """解包指定输出行，供 embedding 等逐行读取路径使用。"""
        selected = self.packed.index_select(1, ids.reshape(-1)).t().to(torch.int16)
        q = torch.empty((ids.numel(), selected.shape[1] * 2), dtype=torch.int16,
                        device=self.packed.device)
        q[:, 0::2] = selected & 0xF
        q[:, 1::2] = (selected >> 4) & 0xF
        q = q[:, :self.k].float().sub_(8.0)
        groups = torch.arange(self.k, device=self.packed.device) // self.group_size
        row_scales = self.scale.index_select(1, ids.reshape(-1)).t()
        return q * row_scales.index_select(1, groups)

    def to(self, dtype: torch.dtype) -> torch.Tensor:
        """解包为连续 ``[N,K]`` 浮点权重，供参考后端与诊断使用。"""
        all_ids = torch.arange(self.n, device=self.packed.device)
        return self.rows(all_ids).to(dtype).contiguous()
