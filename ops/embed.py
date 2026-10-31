"""绑定嵌入矩阵：输入行 gather 与共用的 ``lm_head`` 投影。

同一个权重承担两个完全不同的角色——按 token id 取一行，以及最终投影，后者
是一次解码步骤里最大的一笔权重读取——所以 gather 只反量化它碰到的行，投影
则走 :class:`ops.gemv.GEMV`。

Module: ``coffin/ops``。建立在 ``ops/gemv`` 之上；``models/qwen35/model.py``
用它同时做嵌入查表和 logits 投影。

Constraints:
    * 不管权重怎么存，gather 都必须返回 bf16（``D``）：KV 缓存和每个内核都是
      照 bf16 写的。
    * int8 路径按行用该行自己的 scale 反量化——整矩阵反量化要完整读一遍模型里
      最大的权重。
    * gather 和投影刻意是两个入口；把投影的内核选择从 gather 路径上摘掉，
      正是 gather 便宜的原因。
"""

from __future__ import annotations

import torch

from int8_weights import QWeight
from ops.dtypes import D
from ops.gemv import GEMV


class TiedEmbedding:
    """绑定的输入嵌入 / lm_head 矩阵，以 bf16 或 int8 持有。

    同一个权重承担两个完全不同的角色：按 token id 取一行，以及最终投影。
    量化之后，gather 只反量化碰到的那些行（几千个元素），而投影要读整个矩阵
    ——一次解码步骤里最大的一笔权重读取——所以两者分属不同路径。
    """

    def __init__(self, w, bench: bool = True) -> None:
        """包住共享权重并建好它的投影路径。

        Args:
            w: 嵌入矩阵，普通张量或 :class:`int8_weights.QWeight`。
            bench: 透传给 :class:`GEMV`；``False`` 跳过逐形状的自动调优，
                结构测试之所以便宜靠的就是它。
        """
        self.q = w if isinstance(w, QWeight) else None
        self.gemv = GEMV(w, bench)
        self.w = None if self.q is not None else w

    @property
    def shape(self) -> tuple[int, int]:
        """权重形状，无论当前是哪种存储形式都可读。

        Returns:
            ``(vocab, hidden)``；int8 分支报的是反量化后矩阵的逻辑形状，不是
            packed 缓冲区的形状。
        """
        return (self.q.n, self.q.k) if self.q is not None else tuple(self.w.shape)

    def lookup(self, ids: torch.Tensor) -> torch.Tensor:
        """把 ``[T]`` 个 token id 取成对应的 ``[T, hidden]`` bf16 嵌入。

        int8 分支只反量化选中的行：2k token 的预填充只碰到 ``[vocab, hidden]``
        矩阵里几千行，整矩阵反量化会主导这一步的耗时。

        Args:
            ids: token id，``[T]``，调用方已校验过词表范围。

        Returns:
            ``[T, hidden]`` 的 bf16 激活。
        """
        if self.q is None:
            return self.w.index_select(0, ids).to(D)
        rows, scale = self.q.rows(ids)
        return (rows.float() * scale[:, None]).to(D)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """拿一行激活对绑定矩阵做投影（解码路径）。

        Args:
            x: 激活，``[hidden]`` 或 ``[C, hidden]``。

        Returns:
            logits，``[vocab]`` 或 ``[C, vocab]``。
        """
        return self.gemv(x)

    def prefill(self, x: torch.Tensor) -> torch.Tensor:
        """拿多行激活对绑定矩阵做投影（预填充路径）。

        Args:
            x: 激活，``[C, hidden]``。

        Returns:
            logits，``[C, vocab]``；小 ``C`` 的情形为什么换一个内核见
            :meth:`ops.gemv.GEMV.prefill`。
        """
        return self.gemv.prefill(x)
