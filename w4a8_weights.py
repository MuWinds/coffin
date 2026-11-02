"""保存并解释分组对称 W4 投影权重。

两种容器，形状不同但字节布局完全一致：

* :class:`W4A8Weight` —— 单个投影，``[N, K]``。
* :class:`W4A8MoEWeight` —— MoE 的融合专家栈，``[E, N, K]``，第 e 个专家
  ``packed[e]`` 与单独存的它在字节上逐位相同。这不是图方便：``grouped_gemm``
  用固定偏移寻址（``configuration_qwen4_exp.py:102-106`` 的 EP plan），把 512
  个专家摊平成 512 条独立记录，既让索引膨胀到 49152 条，又破坏了它要的连续布局。

Module: ``coffin/``。由 ``quantized_load`` 构造，再交给投影算子；这里只定义
磁盘/设备侧权重的逻辑形状与解包参考，不依赖模型或后端。

Constraints:
    * 偶数 K 的低 nibble、奇数 K 的高 nibble，各存储 ``q + 8``；每组沿 K
      使用一个逐输出通道 fp32 scale。布局来自 ``quant_disk_w4a8.py``。
    * 逻辑权重为 ``[N, K]``，磁盘 packed 张量为 ``[ceil(K/2), N]``。加载时校验
      这两种形状关系，避免方阵或偶数 K 掩盖转置错误。
    * MoE 变体的 N 轴是**专家内**的输出维，专家才是第 0 轴。校验时不能把三个
      维度按 ``[N, K]`` 去套——那对 ``[512, 1280, 2560]`` 恰好也能过形状检查。
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
        """把共享 K 的多个投影沿输出轴拼成一个。

        用来把 ``gate`` / ``up`` 两个投影融成一次 GEMV：两个投影读同一份激活，
        分开调用会把激活算两遍、启动两次。

        Args:
            weights: 至少一个投影，``packed`` 为 ``[ceil(K/2), N_i]``。必须共享
                ``K``、``group_size`` 与设备。

        Returns:
            新的 :class:`W4A8Weight`，``N`` 是各段之和。

        Raises:
            ValueError: ``K``、``group_size`` 或设备不一致。拼错只是形状合法，
                但分组 scale 会整体错位，读出来的数看着合理却是错的。
        """
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


class W4A8MoEWeight:
    """一组共享同一 K 的专家，``[E, N, K]``，逐专家一个 packed 矩阵。

    Qwen4-Exp 的 MoE 把 512 个专家存成两个 3D 张量
    （``mlp.experts.gate_up_proj [512,1280,2560]``、``down_proj [512,2560,640]``）。
    量化后保持这个形状：``packed[e]`` 与"把第 e 个专家单独量化"逐位相同，因为
    分组量化在 K 轴上按 128 成组、N 轴逐列独立，两种切法的算术完全相同
    （``tools/quant_disk_qwen4.py`` 有逐位验证）。

    为什么不摊成 512 个独立键：``grouped_gemm`` 用固定偏移寻址，摊平会让第 e 个
    专家的起点算错，而形状仍然合法；同时索引会从 2 条涨到 1024 条。

    Args:
        packed: ``[E, ceil(K/2), N]`` 连续 uint8。
        scale: ``[E, groups, N]`` 连续 fp32。
        k: 每个专家沿 K 的长度。
        group_size: 每组的 K 长度，与 :class:`W4A8Weight` 同义。

    Raises:
        ValueError: 维度、dtype、连续性或形状关系不符。
    """

    __slots__ = ("group_size", "k", "n", "packed", "scale")

    def __init__(self, packed: torch.Tensor, scale: torch.Tensor, *,
                 k: int, group_size: int = 128) -> None:
        """校验专家栈的形状关系并原地持有两块缓冲。

        Args:
            packed: ``[E, ceil(K/2), N]`` 连续 uint8。
            scale: ``[E, groups, N]`` 连续 fp32。
            k: 每个专家沿 K 的长度。
            group_size: 每组的 K 长度。

        Raises:
            ValueError: 维度、dtype、连续性或形状关系不符。这些必须在构造时拦下：
                三维张量按 ``[N, K]`` 解释时 ``[512, 1280, 2560]`` 恰好也能过分组
                形状校验，于是错位要到第一次取专家才算出来。
        """
        if (packed.dim() != 3 or packed.dtype != torch.uint8
                or not packed.is_contiguous()):
            raise ValueError("packed weight must be contiguous 3-D uint8 "
                             "[experts, ceil(K/2), N]")
        if (scale.dim() != 3 or scale.dtype != torch.float32
                or not scale.is_contiguous()):
            raise ValueError("scale must be contiguous 3-D fp32 [experts, groups, N]")
        if k <= 0 or group_size <= 0:
            raise ValueError("K and group_size must be positive")
        experts, half_k, n = (int(d) for d in packed.shape)
        groups = (k + group_size - 1) // group_size
        if half_k * 2 < k or tuple(scale.shape) != (experts, groups, n):
            raise ValueError("packed weight or scale shape does not match K/group_size")
        self.packed = packed
        self.scale = scale
        self.k = int(k)
        self.n = n
        self.group_size = int(group_size)

    def __len__(self) -> int:
        """专家个数。

        Returns:
            第 0 轴的尺寸。
        """
        return int(self.packed.shape[0])

    def nbytes(self) -> int:
        """常驻字节数（packed 加 scale）。

        Returns:
            两块缓冲区的字节数之和。
        """
        return self.packed.numel() + self.scale.numel() * self.scale.element_size()

    @property
    def shape(self) -> tuple[int, int, int]:
        """``(experts, N, K)``，与磁盘上的专家栈一致。

        Returns:
            三元组，顺序与 checkpoint 里该张量的维度顺序相同。
        """
        return int(self.packed.shape[0]), self.n, self.k

    def expert(self, index: int) -> W4A8Weight:
        """把第 ``index`` 个专家取成单投影视图。

        Args:
            index: 专家下标。

        Returns:
            :class:`W4A8Weight`。``packed``/``scale`` 是**视图**，不是拷贝——
            grouped_gemm 本来就按偏移直接读这块连续内存，拷贝会把省下的显存和
            带宽还回去。

        Raises:
            IndexError: 下标越界。显式抛，因为负下标在 PyTorch 里会静默回卷。
        """
        if not 0 <= index < len(self):
            raise IndexError(f"expert {index} out of range [0, {len(self)})")
        return W4A8Weight(self.packed[index], self.scale[index],
                          k=self.k, group_size=self.group_size)

    def experts(self) -> list[W4A8Weight]:
        """按顺序取出所有专家的单投影视图。

        Returns:
            长度等于 :meth:`__len__` 的列表，元素共享同一块存储。
        """
        return [self.expert(i) for i in range(len(self))]

    def to(self, dtype: torch.dtype) -> torch.Tensor:
        """解包为连续 ``[E, N, K]`` 浮点权重，供参考后端与诊断使用。

        Args:
            dtype: 目标浮点 dtype。

        Returns:
            ``[E, N, K]`` 连续张量。逐专家解包再堆叠，所以峰值是整栈的一份浮点
            拷贝——这条路径只该出现在诊断里。
        """
        return torch.stack([w.to(dtype) for w in self.experts()])
