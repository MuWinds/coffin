"""本层所有算子共用的 dtype 与掩码常量，从后端能力记录再导出。

这些值轮不到本仓库来选：它们是硬件事实，所以唯一来源是
``backends.caps()``（``act_dtype`` / ``weight_dtype``）。本模块保留旧名字，
这样在搬走来源的同一次改动里，约 200 处 ``from ops.dtypes import D`` 的调用
点不必跟着改名。

Module: ``coffin/ops``。依赖 ``backends`` 和 ``torch``；被其余算子模块和
``models/*`` 导入。

Constraints:
    * ``D`` 和 ``WF`` 必须保持*推导*出来的名字，背后不能有字面量。这里再写
      一份字面量等于把某个平台的假设固化下来，它会和进程实际选中的后端悄悄
      不一致，而那正是这次迁移要消灭的 bug。
    * 常量在 import 时连同后端选择一起绑定：没有运行时热切换（PLATFORMS.md
      §8），所以读取它们的模块看到的值不可能在脚下改变。
    * ``NEG_INF`` 是注意力掩码的填充值；用它而不是一个大负数写字面量，
      这样 softmax 的下溢保持精确。
"""

from __future__ import annotations

from backends import caps as _caps

#: 当前选中后端的能力记录，只读一次。保持私有：需要除两种 dtype 之外的
#: 标志的调用方直接 import ``backends.caps()``，不要伸手进本模块拿第二份记录。
_CAPS = _caps()

#: 所有融合内核用的激活 dtype，包括 int8 KV 缓存那条路径。现有带内核的每个
#: 后端都是 bf16，参考实现也跑它，因为模型分配的就是它、KV 量化器也是照它写的。
D = _CAPS.act_dtype

#: 投影权重的 dtype。不一定等于 ``D``：本引擎调优用的 DCU 有 packed fp16
#: 计算但没有 packed bf16 计算，所以它的预填充 GEMM 在 fp16 权重上能跑
#: 13-20 TFLOP/s，而 bf16 上只有 2.3 TFLOP/s（宽 N 的形状拿到约 7x）。
#: 激活无论如何都是 bf16，而解码 GEMV 的拷贝循环受带宽限制，在那里把权重
#: 读成 fp16 不花代价。有 packed bf16 的型号在这里声明 bf16 即可，其余不变。
WF = _CAPS.weight_dtype

#: 掩码填充值；`-inf` 让 softmax 权重精确为零。
NEG_INF = float("-inf")
