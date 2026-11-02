"""超连接（Gated Residual）：读门把 4 路残差流塌成 1 路，写门把块输出广播回 4 路。

Qwen4-Exp 的主干宽度是 ``hc_count * hidden_size``，每一层没有 pre-norm——归一化
全在这里，作用在 4 路残差流上（``group_size = hidden_size``，每路各做一次）。
一个 block 的输出被复制到 4 路残差流上、每路乘一个自己的标量门，所以写门输出是
``[B, S, hc_count]`` 而不是逐通道的。

Module: ``coffin/models/qwen4exp``。依赖同族的 ``norm``；被 ``blocks`` 和
``model`` 调用。

Constraints:
    * 读门最后是 ``.mean(dim=-2)`` 不是 ``sum``。用 sum 会让读门输入的尺度随
      ``hc_count`` 线性放大 4 倍，形状全对、数值全错（``ARCH_SPEC.md`` §6.2）。
    * 两个门都读**归一化后**的状态，且低秩投影输出先除以 ``hc_count`` 再进
      silu；写门是 ``2 * sigmoid(w / hc_count)``——漏掉 2 会让整个 block 的贡献
      减半，不除 4 会让门饱和（``ARCH_SPEC.md`` §6.3）。
    * ``use_combine=False`` 的末端 mixer 没有 ``block_inject_weight``，forward
      只返回塌缩后的一路；这条分支就是 10240 → 2560 的收口。
"""

from __future__ import annotations

import torch
from torch import nn

from models.qwen4exp.config import TextConfig
from models.qwen4exp.norm import RMSNorm


class GatedResidual(nn.Module):
    """一路读门（低秩 + swish）加一路标量写门。

    Attributes:
        hc_count: 残差流路数。
        hidden_size: 单路宽度。
        hc_norm: 作用在 ``[B, S, hc_count * hidden_size]`` 上、按 ``hidden_size``
            分组的零中心 RMSNorm。
        input_mix_weight_down: 读门的降维投影。
        input_mix_weight_up: 读门的升维投影。
        block_inject_weight: 写门；``use_combine=False`` 时为 ``None``。
    """

    def __init__(self, config: TextConfig, use_combine: bool = True) -> None:
        """建两套门。

        Args:
            config: 文本配置，用来取 ``hc_count`` / ``hidden_size`` /
                ``hc_lowrank`` / ``rms_norm_eps``。
            use_combine: 是否建写门。末端 mixer 用 ``False``，层内的两个用
                默认值。
        """
        super().__init__()
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        hc_hidden_size = self.hc_count * self.hidden_size
        self.hc_norm = RMSNorm(hc_hidden_size, group_size=self.hidden_size,
                               eps=config.rms_norm_eps)
        self.input_mix_weight_down = nn.Linear(hc_hidden_size, config.hc_lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(config.hc_lowrank, hc_hidden_size, bias=False)
        # 只有层内的两套需要写门：它们要把 block 输出写回 4 路残差流。末端
        # mixer 只做塌缩，多一个写门就是多一份永远不用的参数。
        self.block_inject_weight = (
            nn.Linear(hc_hidden_size, self.hc_count, bias=False) if use_combine else None)

    def forward(
        self, hyper_input: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """读门塌缩，可写门时一并给出 4 路状态与逐流写门。

        Args:
            hyper_input: 4 路残差流，``[B, S, hc_count * hidden_size]``。

        Returns:
            ``use_combine=False`` 时只返回 ``mixed_input``，``[B, S, hidden_size]``；
            否则返回 ``(mixed_input, hyper_input, injection_weights)``，其中
            ``injection_weights`` 是 ``[B, S, hc_count]``。

        Raises:
            ValueError: ``hyper_input`` 的最后一维不是
                ``hc_count * hidden_size``。这个错误必须在这里炸：让广播把它吃掉
                之后，读门会拿错位的流做加权，输出仍然"看起来像"激活。
        """
        if hyper_input.shape[-1] != self.hc_count * self.hidden_size:
            raise ValueError(
                f"Expected {self.hc_count * self.hidden_size} hyper-connection features, "
                f"got {hyper_input.shape[-1]}.")

        hyper_input_normed = self.hc_norm(hyper_input)
        # 先除 hc_count 再进 silu：低秩投影的输出宽度是 hc_hidden_size，除以路数
        # 把读门输入的尺度归一化到与单路状态同量级。
        input_mix_weight = torch.nn.functional.silu(
            self.input_mix_weight_down(hyper_input_normed) / self.hc_count)
        input_mix_weight = torch.sigmoid(self.input_mix_weight_up(input_mix_weight))
        input_mix_weight = input_mix_weight.unflatten(-1, (self.hc_count, self.hidden_size))
        # 门乘的是归一化后的 4 路状态，最后取均值：mean 让读门输出与 hc_count
        # 无关，换成 sum 会把尺度放大 4 倍。
        mixed_input = (
            input_mix_weight
            * hyper_input_normed.unflatten(-1, (self.hc_count, self.hidden_size))
        ).mean(dim=-2)
        if self.block_inject_weight is None:
            return mixed_input
        # 乘 2 让零输入处 w = 2*sigmoid(0) = 1，于是初始化附近等价于原始残差相加。
        injection_weights = 2 * torch.sigmoid(
            self.block_inject_weight(hyper_input_normed) / self.hc_count)
        return mixed_input, hyper_input, injection_weights


def apply_injection(hidden_states: torch.Tensor, hyper_input: torch.Tensor,
                    injection_weights: torch.Tensor) -> torch.Tensor:
    """把块输出按逐流标量门写回 4 路残差流。

    Args:
        hidden_states: 块输出，``[B, S, hidden_size]``。
        hyper_input: 写回前的 4 路状态，``[B, S, hc_count * hidden_size]``。
        injection_weights: 逐流写门，``[B, S, hc_count]``。

    Returns:
        写回后的 4 路状态，形状与 ``hyper_input`` 相同。

    Raises:
        ValueError: 三者的 batch/序列长度对不上。
    """
    if hidden_states.shape[:2] != hyper_input.shape[:2] or \
            injection_weights.shape[:2] != hidden_states.shape[:2]:
        raise ValueError(
            "residual write-back shape mismatch: "
            f"block {tuple(hidden_states.shape)}, hyper {tuple(hyper_input.shape)}, "
            f"gates {tuple(injection_weights.shape)}.")
    # 写回是"同一份块输出复制到每一路、每路乘自己的标量门"，不是只写一路。
    injection = hidden_states.unsqueeze(-2) * injection_weights.unsqueeze(-1)
    return hyper_input + injection.flatten(-2)


__all__ = ["GatedResidual", "apply_injection"]
