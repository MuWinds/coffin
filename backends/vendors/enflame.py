"""Enflame GCU：TopsRider / torch_gcu 一族，走 gcu 命名空间。

占位实现——没有内核，没有硬件。与 Iluvatar 那份声明不同，这里没有一手证据可依靠：
``reference/`` 里没有 Enflame 引擎，所以下面几乎每个能力都是保守默认值而不是实测值。
这个不对称值得保持可见，因为一段读起来和实测后端同样笃定的占位 docstring，就是一个
未经验证的数字变成设计约束的老路。

声明是怎么定下来的

* **命名空间是 ``gcu``，且为该厂商独有。** torch_gcu 的指南写得很清楚：把张量搬到
  设备上就是把 ``cuda`` 换成 ``gcu``，``torch.cuda.*`` 调用变成 ``torch.gcu.*``。
  所以 ``torch.cuda.is_available()`` 不会为这张卡作答，检测也就不需要名字标记：命名
  空间本身就是区别依据。
* **没有 64 位整数。** torch_gcu 会把 I64/F64 静默降级成 32 位并告警，它的文档还补
  了一句：对 64 位张量而言 ``tensor.data_ptr()`` 可能是错的。后半句的影响超出索引：
  一个地址不可信的后端无法支持重放捕获指针的图捕获。因此这里写
  ``index_dtype=torch.int32``，这也是本文件唯一一个由文档而非推断支撑的声明字段。
* 其余一切——packed 数学、图捕获、扩展路径、设备名字符串——都还没答案。能定下它们的
  探测列在 ``.tmp_dsh/platforms/VENDORS.md``。

刻意不声明的：``device_name`` 标记。没有任何公开来源说明
``torch.gcu.get_device_name()`` 返回什么，猜一个子串只会造出一条永远匹配不上的检测
规则。命名空间已经能识别厂商，诚实的声明就该省掉它。

Module: ``coffin/backends/vendors/enflame``。声明式：只 import ``backends.vendor``，
外加为 dtype 常量 import 的 ``torch``。在不存在 torch_gcu 的环境里也必须能 import
——检测用 ``getattr(torch, "gcu", None)`` 就是为了不假设它存在。

Constraints:
    * ``index_dtype`` 是 ``int32``，调用方必须把索引控制在范围内；悄悄产出错误位置
      正是这个字段要防的失效方式。
    * 没有实测就不要声明 ``weight_layout``。它保持 ``"nk"`` 默认值，而不是从 Iluvatar
      的结论抄过来：两颗芯片的带宽故事不同，抄来的数字只是一个带引用的虚构。
"""

from __future__ import annotations

import torch

from backends.vendor import Detection, VendorSpec

#: 由 :mod:`coffin.backends.discovery` 发现的模块级声明。
#: 不写 ``name_markers``：torch_gcu 暴露自己的 ``gcu`` 命名空间，命名空间已经能识别
#: 厂商，加一条名字规则只会多一种失败方式。
SPEC = VendorSpec(
    name="enflame",
    detection=Detection(namespace="gcu"),
    act_dtype=torch.bfloat16,
    weight_dtype=torch.bfloat16,
    graph=False,
    autotune=False,
    int8_weights=False,
    batch_attn=False,
    attn_split=1,
    # 有文档依据，不同于本文件其余各项：torch_gcu 没有 I64，降级时会告警，
    # 所以索引必须从一开始就声明成 32 位。
    index_dtype=torch.int32,
    priority=20,
    has_kernels=False,
    notes=("namespace gcu (torch.cuda not redirected); no I64 -- indices are int32 and "
           "data_ptr is documented as possibly wrong for 64-bit; packed math, graph "
           "capture, extension path and device-name string all unmeasured"),
)
