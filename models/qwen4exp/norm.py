"""Qwen4-Exp 用到的两种 RMSNorm：零中心的普通版与线性注意力的门控版。

它们**不是同一个类配一个开关**：``Qwen4ExpTextRMSNorm`` 的权重语义是
``1 + w``（checkpoint 存的是相对 1 的偏移、初始化全 0），
``Qwen4ExpTextRMSNormGated`` 的权重则是普通的 ``w``（初始化全 1），且把门放
在归一化**之后**。按普通增益去乘偏移量不会有任何形状错误，只会让每一处归一化
从"恒等 1"变成"恒等 0"（``ARCH_SPEC.md`` §12 第 5 条），所以这里是两个类。

Module: ``coffin/models/qwen4exp``。只依赖 ``torch``；被 ``gated_residual``、
``qsa``、``gdn``、``ngram`` 调用。

Constraints:
    * 归一化必须全程 fp32，乘完 ``1 + w`` 之后再一次性转回入参 dtype。分两步在
      bf16 上乘会在激活量级大时看到差异，而这条路径在 48 层里每层跑 8 次。
    * ``group_size`` 的语义是"把最后一维切成若干等长组、每组独立算 RMS"，
      权重仍是整条向量的宽度。残差流能稳住靠的就是 4 路各保尺度、不互相稀释。
    * 这里绝不出现 ``torch.nn.functional.rms_norm``：那是 torch 2.4+ 的 API，
      部署目标是 2.1.0。
"""

from __future__ import annotations

import torch
from torch import nn


def _silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU：``x * sigmoid(x)``。

    手写而不查激活表，好让门控归一化不引入除 ``torch`` 之外的依赖。

    Args:
        x: 任意形状的门输入。

    Returns:
        与 ``x`` 同形状。
    """
    return x * torch.sigmoid(x)


#: 门控归一化允许的激活。参考实现的配置校验也只放行这两个。
_GATE_ACTIVATIONS = {"silu": _silu, "sigmoid": torch.sigmoid}


class RMSNorm(nn.Module):
    """零中心化 RMSNorm：``y = norm(x) * (1 + w)``，可选按组归一化。

    Attributes:
        weight: 零中心权重，形状 ``[dim]``。语义是偏移量而不是增益。
        group_size: 每个 RMS 组覆盖的最后一维长度；``None`` 表示整条向量一组。
    """

    def __init__(self, dim: int, group_size: int | None = None, eps: float = 1e-6) -> None:
        """建零中心权重。

        Args:
            dim: 归一化的最后一维宽度。
            group_size: 分组长度，必须整除 ``dim``。
            eps: 加在均方上的常数。

        Raises:
            ValueError: ``group_size`` 不整除 ``dim``。分组错位不会抛异常，
                只会让每组的均方跨过边界取到别的流的能量。
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))
        self.group_size = group_size
        if group_size is not None and dim % group_size != 0:
            raise ValueError(
                f"hidden_size ({dim}) must be divisible by group_size ({group_size}).")

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """在 fp32 上算 RMS 归一化，不含权重。

        Args:
            x: 末维宽度为 ``dim`` 的 fp32 激活。

        Returns:
            与 ``x`` 同形状的归一化结果。
        """
        if self.group_size is not None:
            x = x.reshape(*x.shape[:-1], -1, self.group_size)
        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return out.flatten(-2) if self.group_size is not None else out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """归一化并把零中心权重折进去。

        Args:
            x: 任意 dtype 的激活，末维宽度为 ``dim``。

        Returns:
            与 ``x`` 同形状、同 dtype 的张量。
        """
        output = self._norm(x.float())
        # 权重是相对 1 的偏移量：checkpoint 里会出现负值，直接当增益用会把
        # 整层输出压成 0 附近。
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self) -> str:
        """把形状、eps 和分组宽度写进 ``print(module)`` 的一行。

        Returns:
            形如 ``(10240,), eps=1e-06, group_size=2560`` 的字符串。
        """
        return f"{tuple(self.weight.shape)}, eps={self.eps}, group_size={self.group_size}"


class RMSNormGated(nn.Module):
    """线性注意力的门控 RMSNorm：先归一化，再乘 ``act(gate)``。

    Attributes:
        weight: 普通增益权重，形状 ``[hidden_size]``，初始化为 1。
        variance_epsilon: 加在均方上的常数。
        activation: 门的激活名，``silu`` 或 ``sigmoid``。
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, activation: str = "silu") -> None:
        """建标准增益权重。

        Args:
            hidden_size: 归一化宽度，等于 value head 维。
            eps: 加在均方上的常数。
            activation: 门激活名。

        Raises:
            ValueError: 激活不在 ``{silu, sigmoid}`` 里。参考实现的配置校验同样
                只放行这两个，别在这里悄悄多支持一种。
        """
        super().__init__()
        if activation not in _GATE_ACTIVATIONS:
            raise ValueError(
                f"Unsupported Qwen4-Exp output gate activation: {activation!r}, "
                f"expected one of {sorted(_GATE_ACTIVATIONS)}.")
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.activation = activation

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """归一化 ``hidden_states`` 并用 ``gate`` 门控。

        三处顺序不能动：归一化在门**之前**；权重不偏移（直接乘）；门的激活在
        fp32 上算。门只影响幅度不影响归一化统计，反过来写会得到另一条曲线。

        Args:
            hidden_states: 归一化输入，末维宽度为 ``hidden_size``。
            gate: 门输入，形状与 ``hidden_states`` 一致；来自 ``in_proj_z``。

        Returns:
            与 ``hidden_states`` 同形状、同 dtype。
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * _GATE_ACTIVATIONS[self.activation](gate.to(torch.float32))
        return hidden_states.to(input_dtype)

    def extra_repr(self) -> str:
        """把宽度、eps 和门激活写进 ``print(module)`` 的一行。

        Returns:
            形如 ``(128,), eps=1e-06, activation=sigmoid`` 的字符串。
        """
        return (f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}, "
                f"activation={self.activation}")


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6,
             group_size: int | None = None) -> torch.Tensor:
    """无状态的零中心 RMSNorm，供只需要一次前向的路径使用。

    Args:
        x: 末维宽度为 ``weight.numel()`` 的激活。
        weight: 零中心权重（语义是 ``1 + w``）。
        eps: 加在均方上的常数。
        group_size: 分组长度，``None`` 表示整条向量一组。

    Returns:
        与 ``x`` 同形状、同 dtype。

    Raises:
        ValueError: ``group_size`` 不整除最后一维宽度。
    """
    dim = x.shape[-1]
    if group_size is not None and dim % group_size != 0:
        raise ValueError(f"hidden_size ({dim}) must be divisible by group_size ({group_size}).")
    xf = x.float()
    if group_size is not None:
        xf = xf.reshape(*xf.shape[:-1], -1, group_size)
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    if group_size is not None:
        out = out.flatten(-2)
    return (out * (1.0 + weight.float())).type_as(x)


__all__ = ["RMSNorm", "RMSNormGated", "rms_norm"]
