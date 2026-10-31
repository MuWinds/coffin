"""平台层：coffin 里唯一知道自己在什么硬件上运行的地方。

模型代码在 import 时绑定本包的 ``ops`` 表，之后像调用一列设备原语那样调用它，
永远不知道是哪个平台在应答。这一层存在的全部理由就在这里：``hip_ops``、
``"cuda"`` 这个 device 字面量、``ops/dtypes.py`` 的 ``D``/``WF``、
``models/qwen35/tuning.py`` 的 ``ATTN_SPLIT`` 全是平台事实，它们搬到这里，
模型族才不用继续背着它们。

Module: ``coffin/backends``。与 ``backends.protocol`` 一起位于依赖图底部：
``ops/``、``models/``、``engine/``、``serving/``、``cache/`` 从这里 import，
本包不 import 它们任何一个。import 本包不得编译内核、分配设备张量、要求有
加速卡——纯 CPU 主机上它解析到纯 torch 的参考后端，测试套件和
``COFFIN_BACKEND=reference`` 冒烟运行就靠这一点。

Constraints:
    * 选择只做一次，在首次用到 ``ops`` 属性时确定，此后不再变化：没有运行时
      热切换（PLATFORMS.md §8）。捕获了某个 backend 引用的代码会在进程整个
      生命周期里持有它。
    * ``ops`` 是一张表，不是一个模块：一次读进模块级别名（``ops = backends.ops``）
      再调用表里的条目。每个条目都可调用——参考实现会补齐所选 backend 缺的
      部分——所以正常路径上任何调用点都不得对条目做 ``None`` 判断。
    * ``device``/``act_dtype``/``weight_dtype`` 取自 ``caps``，用来替代模型代码
      现在写死的字面量。直接 import 它们，不要重新推导。
"""

from __future__ import annotations

from backends import discovery, protocol, reference, registry, vendor
from backends.protocol import Backend, Caps, Ops, make_device
from backends.registry import (
    ENV_VAR,
    available_names,
    backend,
    caps,
    candidates,
    describe_all,
    select,
    status,
)
from backends.vendor import Detection, SpecBackend, VendorSpec

#: 所选 backend 的完整算子表。现在就解析好，调用点才能在 import 时直接绑定
#: （``from backends import ops``），解码路径上不必付一次惰性查找的分支。
#: 每个条目都可调用；backend 没有对应内核的名字由参考实现兜底。
ops: Ops = backend().ops()

#: 所选 backend 的所有分配都应该落在这个 ``torch.device`` 上。它就是要替换掉
#: ``models/`` 里 173 处 ``device="cuda"`` 字面量的那个值。
device = backend().device


def reload() -> None:
    """重新解析 backend 选择，并刷新模块级的 ``ops``。

    由 ``engine_api.reload`` 的有序重载调用，那里会就地重建引擎模块。重载不改变
    选择本身（``COFFIN_BACKEND``）：它只是让变更过的 backend 表或能力记录被重新
    拾取，不是让进程换平台。

    Returns:
        无。``ops`` 和 ``device`` 就地重新绑定；已经把它们绑进局部名字的调用方
        仍持有旧值，重载顺序把本包排在所有绑定它的模块之前就是这个原因。
    """
    global ops, device
    # 下面这条指令压制的是「访问包内私有钩子」的告警，这里是有意保留的测试入口。
    registry._reset_for_tests(None)          # noqa: SLF001 - package-internal hook
    ops = backend().ops()
    device = backend().device


#: 本包导出给引擎其余部分的名字。
__all__ = [
    "Backend", "Caps", "Detection", "ENV_VAR", "Ops", "SpecBackend", "VendorSpec",
    "available_names", "backend", "caps", "candidates", "describe_all", "device",
    "discovery", "make_device", "ops", "protocol", "reference", "registry",
    "reload", "select", "status", "vendor",
]
