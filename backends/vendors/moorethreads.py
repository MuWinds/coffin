"""Moore Threads：MUSA 一族，走 musa 命名空间。

占位实现——没有内核，没有硬件。它和 Iluvatar、Enflame 并列列出，是因为它是本引擎瞄准
的第三家国产加速卡，也因为三家之中它的检测路径最清晰，正好当作新厂商如何声明自己的
范本。

声明是怎么定下来的

* **命名空间是 ``musa``，且刻意不改写 ``torch.cuda``。** torchada 的文档直说在 MUSA
  构建上 ``torch.cuda.is_available()`` 返回 ``False``，好让平台检测照常工作。这对本层
  是最理想的情形：没有共享命名空间的歧义，不需要名字规则，光靠命名空间就能识别厂商。
* **设备名里带着芯片型号。** ``musaInfo`` 报出的名字形如 ``MTT S4000``，下游项目按
  ``MTT`` 前缀匹配。尽管命名空间已经能决定归属，这里仍然声明了名字标记，因为它不花
  代价，而且万一厂商哪天出了改写 ``cuda`` 的兼容层，那份声明还站得住——到那时名字就是
  仅剩的依据。

其余一切都没答案：packed 数学、图捕获、扩展路径，以及 int8 权重通路是否划算。能定下
它们的探测在 ``.tmp_dsh/platforms/VENDORS.md``。

Module: ``coffin/backends/vendors/moorethreads``。声明式：只 import
``backends.vendor``，外加为 dtype 常量 import 的 ``torch``。在不存在 MUSA 构建的环境
里也必须能 import。

Constraints:
    * ``name_markers`` 是大小写不敏感的子串匹配。选 ``"mtt"`` 而不是完整产品名，是因为
      型号数字每代都变，而前缀没变过。
    * 不要声明本文件论证不了的能力；同一条规则在兄弟厂商模块里写得更长。
"""

from __future__ import annotations

import torch

from backends.vendor import Detection, VendorSpec

#: 由 :mod:`coffin.backends.discovery` 发现的模块级声明。
#: 在 ``torch.cuda`` 不被改写期间这个名字标记是多余的；一旦哪天有兼容层改了这一点，
#: 它就是唯一还管用的区别依据。便宜的保险。
SPEC = VendorSpec(
    name="moorethreads",
    detection=Detection(namespace="musa", name_markers=("mtt", "moore")),
    act_dtype=torch.bfloat16,
    weight_dtype=torch.bfloat16,
    graph=False,
    autotune=False,
    int8_weights=False,
    batch_attn=False,
    attn_split=1,
    priority=30,
    has_kernels=False,
    notes=("namespace musa, torch.cuda intentionally not redirected "
           "(is_available() is False here); device name reports MTT parts; packed "
           "math, graph capture and extension path unmeasured"),
)
