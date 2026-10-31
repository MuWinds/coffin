"""本 checkpoint 用到的两种约定的 RMSNorm。

``input_layernorm`` / ``post_attention_layernorm`` / ``q_norm`` / ``k_norm``
是零中心化的（实际系数是 ``1 + w``，所以存下来的权重带负值）；
``linear_attn.norm``（RMSNormGated）是标准的 ``w * x`` 再乘
``silu(gate)``。两者用混会把输出搅乱而不抛任何异常，所以它们是两个类而不是
一个开关。

Module: ``coffin/ops``。绑定后端的 ``rms_norm`` / ``add_rms_norm`` /
``rms_norm_gated`` 入口，并把 checkpoint 的权重约定折进去；``models/qwen35``
里每个 block 都调用它。

Constraints:
    * ``eps=1e-6`` 由 checkpoint 固定；改它会移动 logits。
    * 行间距作为参数传，而不是张量的属性：模型归一化的是更宽的融合缓冲区里的
      列切片（``q_norm`` 作用在转置过的头张量上），所以调用方自己
      ``.contiguous()`` 等于把本层最宽的激活完整拷一遍。只有*最后一维* 的
      stride 不是 1 才必须拷贝，因为二维 reshape 需要它。
    * 这里绝不许出现 ``torch.nn.functional.rms_norm``。它是 2.4+ 的 API，
      而部署目标是 torch 2.1.0，在那里会抛 ``AttributeError``。手写的实现放在
      参考后端里，所以表是这套数学唯一存在的地方。
    * 零中心化约定只存在于 ``RMSNorm``；它的权重绝不能拿去给带门的类用。
    * ``add_residual`` 往自己的 ``x`` 参数里累加。省下的就是这些——残差流留在
      同一个缓冲区里，不再写一遍再读一遍——这也是为什么调用方必须独占
      ``x``。它和别的东西共用一个张量属于调用方的 bug：内核分不清一个被别名
      的残差和一个私有的残差。
"""

from __future__ import annotations

import torch

from backends import ops as bops
from ops.dtypes import D


def _rows(x: torch.Tensor) -> torch.Tensor:
    """把 ``[..., n]`` 看成 ``[M, n]``，且不打乱行间距。

    Args:
        x: 最后一维是归一化宽度的激活。前面可以有任何形状；融合内核把它当作
            扁平的批行来索引。

    Returns:
        最后一维 stride 为 1 时返回二维视图，否则返回连续拷贝后的二维视图——
            转置过的操作数拿不到合法的 ``[M, n]`` 视图，拷贝是两个错误答案里
            便宜的那个。返回张量的行仍可能比 ``n`` 宽，这个间距由后端的
            ``stride`` 参数携带。
    """
    if x.stride(-1) != 1:
        x = x.contiguous()
    return x.reshape(-1, x.shape[-1])


class RMSNorm:
    """零中心化 RMSNorm，``y = norm(x) * (1 + w)``（Qwen3.5 的约定）。"""

    def __init__(self, w: torch.Tensor) -> None:
        """把 checkpoint 的偏移约定折进去，并以 bf16 保存权重。

        ``1 +`` 只在这里加一次，而不是每次调用都加：checkpoint 存的是零中心
        权重（会出现负值），在加载时补上偏移，热路径就只剩一次乘法。

        Args:
            w: checkpoint 权重，``[hidden]``，原始精度不限；折叠在 fp32 里做，
                偏移才精确。
        """
        self.g = (1.0 + w.float()).to(D).contiguous()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """归一化并缩放 ``x`` 的最后一维。

        Args:
            x: 激活，前面形状任意，``[..., hidden]``。非 bf16 的输入先转换，
                因为下游每个内核都假定 ``D``。

        Returns:
            与转换后输入相同的形状，dtype 为 bf16。
        """
        if x.dtype != D:
            x = x.to(D)
        xf = _rows(x)
        return bops.rms_norm(xf, self.g, xf.stride(0), 1e-6).view(x.shape)

    def add_residual(self, x: torch.Tensor, add: torch.Tensor,
                     out: torch.Tensor) -> torch.Tensor:
        """把残差相加折进这次归一化：先 ``x += add``，再归一化写进 ``out``。

        解码层的 pre-norm 半边是 ``h = mix(norm(x)); x = x + h``，随后 MLP 前
        还要再 ``norm(x)`` 一次。拆成两个算子就是两次 launch、两次读残差行；
        在这张卡 5.75 us 的 launch 下限面前，成本就是 launch 本身。融合起来是
        同样的一遍算术，所以 token id 不会变——相加仍然先把和舍入成 bf16，
        均方也仍然按那个舍入后的值算。

        残差**就地**累加进 ``x``。那正是层要保持活着的缓冲区：``x`` 就是残差
        流，下一层还要往里加，把和写到别处只是把拷贝换个地方。

        Args:
            x: 残差流，前面形状任意，``[..., hidden]``，bf16。**会被改写**：
                返回时里面是和。调用方必须独占它；和任何活值共用的张量（嵌入
                输出、别的缓冲区仍在读的视图）会被这次调用弄坏。
            add: 要累加的 mixer 输出，形状同 ``x``。它在任何输出写入之前就已
                读走，所以也允许就是 ``out``——层借此复用 mixer 自己的缓冲区，
                省掉一次分配。
            out: 归一化后的和要写入的稠密 bf16 目标，形状同 ``x``。不许与 ``x``
                别名：和必须活过归一化。

        Returns:
            ``out``，便于链式调用。

        Raises:
            TypeError: ``x`` 或 ``out`` 不是 bf16。残差流是就地累加的，没有更宽
                的缓冲区可退回，而在这里转换 ``x`` 会把和困在临时对象里。
            ValueError: ``out`` 不稠密，或者 ``out`` 就是 ``x``。前者会写进一个
                副本——结果无声丢失；后者会覆盖残差。
        """
        if x.dtype != D or out.dtype != D:
            raise TypeError(f"add_residual accumulates {D} in place and writes {D}; "
                            f"got x={x.dtype}, out={out.dtype}")
        # 不稠密的 ``out`` 会被 reshape 成副本，内核写进副本，调用方拿到的是没变的
        # ``out``，而且不报错。这是本次调用里唯一没法在下游被发现的布局错误。
        if out.stride(-1) != 1 or _rows(out).data_ptr() != out.data_ptr():
            raise ValueError("add_residual needs a dense destination; "
                             f"got strides {tuple(out.stride())}")
        if out is x or out.data_ptr() == x.data_ptr():
            raise ValueError("add_residual cannot write over the residual it "
                             "accumulates: the sum is still needed")
        if add.dtype != D:
            add = add.to(D)
        xf = _rows(x)
        bops.add_rms_norm(xf, _rows(add), self.g, _rows(out), xf.stride(0), 1e-6)
        return out


class RMSNormGated:
    """标准 RMSNorm 再乘以 ``silu(gate)``（Qwen3.5 线性注意力的输出归一化）。"""

    def __init__(self, w: torch.Tensor) -> None:
        """以 bf16 保存权重，不加偏移。

        Args:
            w: checkpoint 权重，``[hidden]``，本身已经是中心化的（不像
                :class:`RMSNorm` 那样把偏移折进去）。
        """
        self.w = w.to(D).contiguous()

    def __call__(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """归一化 ``x``，再乘上同形状 gate 的 silu。

        归一化是按头做的，所以权重是 ``[dv]`` 而不是摊平后的宽度：先摊平就会
        一次性除以所有头的 RMS。

        Args:
            x: 激活，``[..., dv]``；需要时转换成 bf16。
            gate: 形状同 ``x`` 的门张量。它是另一条投影，所以融合内核要求两个
                操作数按同样方式摊平。

        Returns:
            与 ``x`` 同形状，已归一化并加门。
        """
        if x.dtype != D:
            x = x.to(D)
        if gate.dtype != D:
            gate = gate.to(D)
        return bops.rms_norm_gated(_rows(x), self.w, _rows(gate), 1e-6).view(x.shape)
