"""静态批解码与一次前向校验用的批处理权重读取。

解码步受权重读取限制（每步 4 GB int8，约 700 GB/s），所以把一行权重乘到 B 个
输入向量上和乘到一个上开销几乎一样——静态批的全部收益就在这里。因此这里两个
入口都很薄：把 ``[M, K]`` 的激活块交给 ``linear_m``，由后端决定特化哪个
``M``、其余怎么切。

这些决定原来就住在这里：逐 M 的展开表、两个选择特定厂商内核的环境变量开关，
以及五次针对可选内核符号的 ``hasattr`` 探测。那些统统是内核选择，现在归后端，
所以本文件只留下真正属于模型侧的两样东西：融合投影的分组切分，以及校验路径在
挑实现之前要问的*能力*判定（``wide_kernel_ready`` / ``multi_q_ready``，答案
来自 ``caps.native`` 而不是符号是否存在）。

Module: ``coffin/models/qwen35``。依赖 ``ops/gemv`` 和 ``backends``；被
``blocks.py``（静态批路径上的每个投影）以及 ``full_attn``/``mtp``（校验宽度）
使用。

Constraints:
    * 与单条解码逐位一致是契约：静态批必须算出一模一样的数字，否则一次批处理
      请求和一次非批处理请求会给出不同结果。这件事现在由后端为每个 ``linear_m``
      入口负责，这也是本文件不再选展开方式的原因：调用方的选择只可能对某一张卡
      的内核正确。
    * 这里不许出现内核符号、设备 API 或调优开关。没有 ``linear_m`` 的后端照样
      应答这个入口（参考实现补上），所以回退路径在本模块里看不见。
    * ``WIDE_STATS`` 是不加锁改写的，这没问题，因为引擎从 ``llm_engine`` 单线程
      驱动所有投影调用。
"""

from __future__ import annotations

import torch

from backends import caps as _caps
from ops.gemv import GEMV, GEMVGroup

#: 当前选中后端的能力记录，只读一次。只给下面两个能力判定用，它们的存在是为了
#: 让校验路径在捕获图*之前*就挑好实现——这个决定必须在捕获时下，而不是每步下。
_CAPS = _caps()


def _project(proj: GEMV, x: torch.Tensor) -> torch.Tensor:
    """对一批行施加一个投影。

    Args:
        proj: 要施加的投影。
        x: 激活，``[M, K]``，bf16。

    Returns:
        ``[M, N]``。在任何 ``linear_m`` 保持单行累加次序的后端上，第 ``r`` 行
        等于对 ``x[r]`` 做单条解码的结果，批处理与非批处理路径就是按这条性质
        互相校验的（见模块约束）。
    """
    if x.shape[0] == 1:
        # bucket 1 是一行的图：走单流入口并保留行维，而不是走批入口，后者的 M
        # 在捕获时就固定了。
        return proj(x[0]).unsqueeze(0)
    return proj.linear_m(x)


def batch_proj(proj, x: torch.Tensor):
    """静态批解码各宽度用的批处理投影。

    Args:
        proj: 一个 :class:`~ops.gemv.GEMV`、一个 :class:`~ops.gemv.GEMVGroup`
            （融合的 ``qkv`` / ``in_proj_qkv/z/b/a`` / ``gate_up`` 读取）或
            :class:`~ops.embed.TiedEmbedding` 内部的 GEMV。
        x: 激活，``[M, K]``，bf16。

    Returns:
        ``[M, N]``；``proj`` 是融合组时返回按权重拼接顺序切分的一组
        ``[M, N_i]`` 张量。
    """
    if isinstance(proj, GEMVGroup):
        return _project(proj.g, x).split(proj.sizes, dim=1)
    return _project(proj, x)


def wide_kernel_ready() -> bool:
    """报告本后端有没有原生的批处理投影。

    调用方用它决定是否在捕获时选用一次前向校验：没有原生 ``linear_m`` 时那条
    路径仍然正确，但会逐行走反量化的 GEMM 读权重，所以捕获前值得问一句。

    Returns:
        当前后端用内核提供 ``linear_m`` 时为 ``True``。这是能力问题而非符号探测：
        ``caps.native`` 是后端自己声明它实现了什么。
    """
    return "linear_m" in _CAPS.native


def multi_q_ready() -> bool:
    """报告多查询分页注意力内核在本后端是否为原生实现。

    Returns:
        当前后端原生提供 ``attn_paged_multi_q`` 时为 ``True``；``False`` 表示
        校验必须回退到逐查询的注意力路径。
    """
    return "attn_paged_multi_q" in _CAPS.native


#: ``FullAttn.verify_batch_multi`` 接上内核之后就翻成 True。
MULTI_Q_WIRED = True


#: 统计 :func:`wide_proj` 把每个模块送到了哪条路："smallM" / "prefill"。留着它
#: 是因为 int8 构建里一次前向校验必须全是 smallM；否则就是走反量化路径，会表现
#: 成每步耗时回退。
WIDE_STATS: dict[str, int] = {}


def _count(kind: str) -> None:
    """记下 ``wide_proj`` 走了 ``kind`` 这条路径，供事后归因。

    Args:
        kind: 路径标签，"smallM" / "prefill" 之一；它被读取的方式见
            :data:`WIDE_STATS`。
    """
    WIDE_STATS[kind] = WIDE_STATS.get(kind, 0) + 1


def wide_proj(proj, x: torch.Tensor):
    """任意 M（``B*c``）下的批处理投影，供一次前向校验使用。

    这里对每个 M 都去问后端的 ``linear_m``，而不是只问旧版逐 M 内核实例化过的那些
    2 的幂宽度：特化哪个 M、其余怎么切都是后端的决定，所以本文件没有需要和内核
    保持同步的 M 表。和 :func:`batch_proj` 一样，它接受
    :class:`~ops.gemv.GEMVGroup` 并为它返回列表。

    Args:
        proj: 一个 :class:`~ops.gemv.GEMV` 或 :class:`~ops.gemv.GEMVGroup`。
        x: 激活，``[M, K]``，bf16。``M`` 不必是 2 的幂。

    Returns:
        ``[M, N]``，融合组则是一组按权重切分的张量。走宽路径时各行不保证与单条
        解码逐位一致；:data:`WIDE_STATS` 记录了走的哪条路径。
    """
    if isinstance(proj, GEMVGroup):
        return wide_proj(proj.g, x).split(proj.sizes, dim=1)
    _count("smallM" if x.shape[0] <= 8 else "prefill")
    return proj.linear_m(x) if x.shape[0] > 1 else proj(x).unsqueeze(0)
