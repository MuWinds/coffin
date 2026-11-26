"""本后端的权重布局常量。

``Caps.weight_layout == "nk"``：契约、加载器、checkpoint 与本后端的全部内核共用
``[N, K]``（一行输出通道、K 轴在内）这一种布局，于是**没有任何转置**。

这里原来放着 ``KnWeight`` / ``KnQWeight`` / ``as_kn`` / ``to_kn`` 这一套
``[K, N]``（kn）打包机制，以及按对象身份记忆化的转置缓存。kn 内核整条删除之后
它们也一起删掉了：kn 那份副本对 int8 只换来约两成解码速度，而 NK 是后续要专门
优化的方向，留一份 kn 副本会把后面那条路带偏（用户原话：「涉及 KN 的一律删掉」）。

Module: ``coffin/backends/corex``。只 import ``torch``，不 import ``coffin`` 里的任何
东西。

Constraints:
    * 内核只读一种 dtype，就是 :data:``WEIGHT_DTYPE``。选 bf16 是因为预填充
      GEMM 在本卡上 bf16 和 fp16 都到 32.6 TFLOP/s，checkpoint 自己的 dtype
      加载时不必转换。
    * ``LIB_MAX_N`` 是这张卡库 GEMM 的性质，不是可调参数；超宽的 ``N`` 要沿行切。
"""
from __future__ import annotations

import torch

#: 打包权重的存储 dtype。必须与本后端的 ``Caps.weight_dtype`` 一致；内核按
#: （权重 dtype，bf16 激活）这一对编译。
WEIGHT_DTYPE = torch.bfloat16

#: 能交给库 GEMM 的最大 ``N``。本卡实测：``N = 131072`` 正确，
#: ``N = 196608`` 和 ``N = 248320``（``lm_head``）返回 NaN。预填充路径按这个
#: 宽度沿 ``N`` 分块；解码路径根本不调库，所以这个上限只约束预填充。它是这张
#: 卡 GEMM 的性质而不是可调参数，因此和布局决定放在一起。
LIB_MAX_N = 131072
