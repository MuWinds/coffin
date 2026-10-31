"""平台如何声明自己：检测数据、能力策略、可选内核。

本模块就是「加一个厂商要改注册表三处」与「加一个厂商只丢进一个文件」之间的差别。
厂商用 :class:`VendorSpec` 描述自己；:func:`coffin.backends.discovery` 找到它，
:class:`SpecBackend` 把它变成 registry 驱动的对象。加平台时没有中心清单要改，
也没有每个厂商的胶水代码要写。

两类平台，一套机制

实测目标厂商时冒出一个设计必须吸收的分叉：

* **独立命名空间。** Enflame 暴露 ``torch.gcu``，Moore Threads 暴露 ``torch.musa``
  ——各自是 ``torch`` 上的属性，都不改写 ``torch.cuda``。这类平台光凭命名空间就能
  识别厂商。
* **共享命名空间。** Iluvatar 的 CoreX 实现的是 CUDA API，所以一张 Iluvatar 卡
  经由 ``torch.cuda`` 应答的方式与本引擎原生的 Hygon DCU 完全一样：
  ``torch.cuda.is_available()`` 为 True，``get_device_name()`` 返回芯片名（已在本
  仓库外核实：厂商自己的适配代码就比较
  ``torch.cuda.get_device_name() == "Iluvatar BI-V100"``，一次 ComfyUI 运行打出的
  是 ``Device: cuda:0 Iluvatar BI-V150``）。这里命名空间说明不了任何事，设备*名字*
  才是唯一的区分依据。

:func:`detect` 返回的**认领强度**而不是布尔值，一套机制才能同时覆盖两者。按名字认出
设备的厂商认领得强，只匹配到命名空间的厂商认领得弱，registry 取每个设备上最强的
认领。于是在一张 Iluvatar 卡上，ROCm 后端和 Iluvatar 后端都可能回答「这里有一个
CUDA 设备」，而 Iluvatar 那个靠强度胜出。这也意味着布局**不**取决于 CoreX 是否设置
``torch.version.hip`` 这个悬而未决的问题（有报告说它的运行时会抛
``No HIP GPUs are available``，而那个字符串只存在于 HIP 构建里，那样一来 ROCm 的
构建标记就没法当作排除条件）。按认领强度排优先级，两种情形都成立。

一份声明的三个部分

*检测*——是数据（:class:`Detection`）而不是代码，因为对每个厂商问的是同一个问题，
只有常量不同。
*策略*——最终变成 :class:`~coffin.backends.protocol.Caps` 的那些字段。它们是声明
而非实测；厂商文件不得把数据手册上的数字写成「实测」标志。
*内核*——``build_ops`` 返回一张**部分**表，只列出该厂商实现了的部分；解析器从
:mod:`coffin.backends.reference` 补齐其余部分。厂商因此可以只从一个内核（通常是
``linear``）起步再慢慢长，模型侧不需要任何回退分支。

Module: ``coffin/backends``。import ``protocol`` 和 ``reference``；绝不 import
``coffin/ops``、``models``、``engine``、``serving`` 或 ``cache``。它不得在模块作用域
import 厂商的运行时——算子表是惰性构建的，所以 import 本模块在从未见过该硬件的
主机上也能工作。

Constraints:
    * :func:`detect` 必须保持廉价且无副作用：registry 每次选择都会探测每个被发现的
      厂商，一个会初始化设备运行时的探测会让纯 CPU 的 import 变成一次驱动加载。
    * spec 绝不能为它没有证明过的平台声明可用。``has_kernels`` 的作用就是让占位实现
      与可用后端在不看算子表的情况下也能区分；占位实现会被发现和报告，但永远不会被
      自动选中。
    * 认领强度是有序的：名字匹配永远强过只匹配命名空间。交换这个顺序，就是 ROCm
      后端抢走一张 Iluvatar 卡的做法。
"""

from __future__ import annotations

import dataclasses
import functools
from dataclasses import dataclass
from typing import Callable

import torch

from backends import reference
from backends.protocol import Caps, Ops, make_device

#: :func:`detect` 返回的认领强度，按具体程度递增。``CLAIM_NONE`` 表示「不是我的
#: 卡」；高于它的任意值都表示「是我的」，两个厂商同时应答同一设备时取最高值。
CLAIM_NONE = 0
#: 只匹配到设备命名空间。当命名空间为该厂商独有时是对的（``gcu``、``musa``）；不
#: 独有时它充当默认认领者（``cuda``）。
CLAIM_NAMESPACE = 1
#: 匹配到设备名标记。这是赢下共享命名空间的唯一途径，所以从构造上就强过命名空间
#: 默认认领。
CLAIM_NAME = 2


@dataclass(frozen=True, slots=True)
class Detection:
    """回答「这是不是我这套代码写给它用的卡？」所需的数据。

    每个字段都是厂商不写代码就能陈述的常量。几个字段组合起来，才让多个厂商共享同一
    设备命名空间时检测依然无歧义。

    Attributes:
        namespace: 厂商构建暴露的 ``torch`` 设备命名空间，即 ``torch`` 上的属性名
            （``"cuda"``、``"gcu"``、``"musa"``、``"xpu"``）。它是命名空间，**不是**
            厂商：``torch.cuda`` 由 CoreX、MUSA 和 ROCm 构建分别实现。
        name_markers: 小写子串，任意一个出现在 ``get_device_name(0)`` 里即算命中。命中
            以 :data:`CLAIM_NAME` 认领该设备。为空表示该厂商接受其命名空间里剩下的
            一切，即 :data:`CLAIM_NAMESPACE`——对独有命名空间是对的，对共享命名空间
            则是「默认认领者」的位置。
        hip: 要求是 ROCm 构建（``torch.version.hip`` 已设置）。Hygon 后端拿它当*门槛*
            而不是身份判定：它排除一张普通 NVIDIA CUDA 卡，而一张同样设置了该标记的
            CoreX 卡会在认领强度上输给 Iluvatar 厂商。
    """

    namespace: str
    name_markers: tuple[str, ...] = ()
    hip: bool = False


def device_name(namespace: str, index: int = 0) -> str:
    """读取某个命名空间的设备名，不预设是哪一个命名空间。

    Args:
        namespace: 设备命名空间，即 ``torch`` 上的属性名。
        index: 设备序号。

    Returns:
        该命名空间的 ``get_device_name(index)``；命名空间或调用不可用时为 ``""``。
        在没有该硬件的主机上，空串是诚实的答案，调用方把它当作「没有名字可匹配」。
    """
    module = getattr(torch, namespace, None)
    getter = getattr(module, "get_device_name", None)
    if getter is None:
        return ""
    try:
        return str(getter(index))
    except Exception:                             # pragma: no cover - 驱动报错
        return ""


def detect(detection: Detection) -> int:
    """返回本机与某厂商检测数据的匹配强度。

    检查按代价排序：命名空间是否存在、是否可用、构建标记，最后才是读设备名（唯一
    在可用性之外还要碰驱动的一步）。任何一步失败都立刻返回 :data:`CLAIM_NONE`，
    正因如此在纯 CPU 主机上探测所有被发现的厂商依然廉价。

    Args:
        detection: 厂商的检测常量。

    Returns:
        :data:`CLAIM_NONE`、:data:`CLAIM_NAMESPACE` 或 :data:`CLAIM_NAME`。调用方
        会跨厂商比较强度，所以厂商绝不能为一个它不想占有的设备返回非零值。
    """
    module = getattr(torch, detection.namespace, None)
    if module is None:
        return CLAIM_NONE
    try:
        if not module.is_available():
            return CLAIM_NONE
    except Exception:                             # pragma: no cover - 驱动报错
        return CLAIM_NONE
    if detection.hip and getattr(torch.version, "hip", None) is None:
        # 一张非 ROCm 派生构建下的 CUDA 卡。在这里答「是」，就是这个 ROCm 后端去
        # 为一张根本没有的 ISA 编译内核的来路。
        return CLAIM_NONE
    if not detection.name_markers:
        return CLAIM_NAMESPACE
    name = device_name(detection.namespace).lower()
    if not name:
        return CLAIM_NONE
    if any(marker.lower() in name for marker in detection.name_markers):
        return CLAIM_NAME
    return CLAIM_NONE


@dataclass(frozen=True, slots=True)
class VendorSpec:
    """一个平台的声明：怎么找到它、它承诺什么、它跑什么。

    厂商文件恰好定义其中一个作为 ``SPEC``。registry 和启动横幅需要的一切都由它推导，
    所以这个文件是数据，加上内核出现之后的一个 ``build_ops`` 钩子。

    Attributes:
        name: ``COFFIN_BACKEND`` 接受、启动行打印的后端标识符。在被发现的厂商之间必须
            唯一。
        detection: 如何识别该平台，见 :class:`Detection`。
        act_dtype: 该平台融合数学所针对的激活 dtype。
        weight_dtype: 在该平台上预期最快的投影权重 dtype。这是预填充最大的一个杠杆：
            一颗有 packed fp16 而没有 packed bf16 的芯片跑 13-20 TFLOP/s，bf16 上只有
            2.3，所以这个选择是硬件事实。
        graph: 设备图捕获预期是否可用。``False`` 让解码保持即时执行，更慢但永远不会
            重放出过期输出。
        autotune: 加载时的内核基准测试预期是否可用。需要后端命名空间里的事件计时能用；
            注意 Hygon 引擎的 ``torch.cuda.Event`` 扫描在 Iluvatar 上同样有效，所以在
            那里填 ``False`` 只是白白损失而不是安全措施。
        int8_weights: 后端能否直接消费 int8 权重，而不必每次调用反量化整个矩阵。
        batch_attn: 是否存在批处理注意力内核，即一个批桶是花一次启动还是每行一次。
        attn_split: 注意力内核的 split-K 宽度。这是内核 ABI 而非调优提示：调用方据此
            确定 online-softmax 部分和缓冲区的尺寸。``1`` 表示不切分，也是内核还不存在
            之前唯一安全的声明。
        weight_layout: ``"nk"`` 或 ``"kn"``——该平台希望投影权重以哪种布局存放。这
            不是装饰：在 Iluvatar BI-V100 上，``nn.Linear`` 产出的 ``[N, K]`` 行主序
            实测 19-30 GB/s，占掉一个解码步 92% 的时间，而 ``[K, N]`` 配 ``x @ Wt``
            达到 362-410 GB/s。契约保持 ``[N, K]``（那是 checkpoint 的布局）；``"kn"``
            后端在加载时转置一次并自担成本，这也是它属于 spec 字段而不属于契约变更的
            原因。
        index_dtype: 后端可以传给内核用于位置和槽位表的整型 dtype。Enflame 的
            torch_gcu 没有 I64：它会静默降级到 32 位并告警，所以那里的后端必须声明
            ``torch.int32``，由调用方把索引控制在范围内。
        priority: 认领强度相等的厂商之间的平局裁决，越小越优。平局算错误而不是靠 import
            顺序解决，因为哪块卡被哪个后端认领不能取决于文件系统顺序。
        notes: 该平台上*未实测*的东西。它会被抄进选择错误的原因里，好让一个声明出来的
            标志不会被误当成实测结论。
        has_kernels: ``build_ops`` 是否返回真内核。占位实现会被发现、名字也能解析，但
            ``usable()`` 保持 ``False``，于是自动探测会继续往下走，而不是假装在加速。
        build_ops: 返回该厂商的部分算子表。``None`` 等价于「还没有原生算子」；解析器
            会用参考后端补齐整张表，模型照样能跑。
    """

    name: str
    detection: Detection
    act_dtype: torch.dtype = torch.bfloat16
    weight_dtype: torch.dtype = torch.bfloat16
    graph: bool = False
    autotune: bool = False
    int8_weights: bool = False
    batch_attn: bool = False
    attn_split: int = 1
    weight_layout: str = "nk"
    index_dtype: torch.dtype = torch.int64
    priority: int = 100
    notes: str = ""
    has_kernels: bool = False
    build_ops: Callable[[], Ops] | None = None

    def validate(self) -> None:
        """拒绝会产生静默错误行为的声明。

        Raises:
            ValueError: 名字或命名空间为空、``weight_layout`` 不是两种已定义布局之一、
                设了 ``has_kernels`` 却没给 ``build_ops``，或者 ``attn_split`` 非正数。
                每一条都会产出一个能被发现、却坏在只有首次调用才暴露、而且是在没人
                有过的硬件上暴露的后端。
        """
        if not self.name:
            raise ValueError("a vendor spec needs a name")
        if not self.detection.namespace:
            raise ValueError(f"{self.name}: detection.namespace is required")
        if self.weight_layout not in ("nk", "kn"):
            raise ValueError(
                f"{self.name}: weight_layout must be 'nk' or 'kn', "
                f"got {self.weight_layout!r}")
        if self.attn_split < 1:
            raise ValueError(f"{self.name}: attn_split must be >= 1")
        if self.has_kernels and self.build_ops is None:
            raise ValueError(
                f"{self.name}: has_kernels is True but no build_ops was given; "
                f"a backend that claims native ops must supply them")


class SpecBackend:
    """穿上 :class:`Backend` 接口的 :class:`VendorSpec`。

    不论手写模块（``hip``、``reference``）还是声明（``vendors/`` 下的一切），registry
    都用同一批成员驱动后端。本类就是适配器，厂商文件因此保持为数据，不用重复写管道
    代码。

    构造廉价且无副作用；算子表在首次调用 :meth:`ops` 时才构建，因为厂商链接或编译
    内核就发生在那里。
    """

    def __init__(self, spec: VendorSpec) -> None:
        """包好一份已校验的 spec。

        Args:
            spec: 厂商声明。:meth:`VendorSpec.validate` 在这里就跑，坏声明在发现阶段
                失败，而不是首次使用时才失败。

        Raises:
            ValueError: spec 非法，见 :meth:`VendorSpec.validate`。
        """
        spec.validate()
        self.spec = spec
        self.name = spec.name

    def claim(self) -> int:
        """报告本机有多像该厂商的平台。

        Returns:
            来自 :func:`detect` 的强度。registry 会跨厂商比较它，所以一张 Iluvatar
            卡由 Iluvatar 的 spec 赢下，而不是同样看到 CUDA 设备的 ROCm 后端。
        """
        return detect(self.spec.detection)

    def probe(self) -> bool:
        """报告该厂商的平台是否存在。

        Returns:
            认领非零时为 ``True``。
        """
        return self.claim() > CLAIM_NONE

    def usable(self) -> bool:
        """报告本后端现在能否跑模型。

        Returns:
            ``probe() and has_kernels``。于是占位实现永远不会被自动选中：在有这张卡的
            主机上它会被检测到并报出来，registry 继续往下回退到参考后端，而不是假借
            加速之名把模型跑得更慢。
        """
        return self.probe() and self.spec.has_kernels

    def reason(self) -> str:
        """解释 :meth:`usable` 为什么是 ``False``，供选择错误的文案使用。

        「没有设备」与「有设备但没有内核」的区别正是重点：在真实的厂商硬件上，必须
        告诉运维人员这个适配还没实现，而不是让他去查驱动问题。

        Returns:
            可用时为 ``""``，否则是一句点名缺了哪一半的话。
        """
        if self.usable():
            return ""
        if not self.probe():
            return (f"no {self.name} device visible through torch "
                    f"(namespace {self.spec.detection.namespace!r} not reporting one)")
        detail = f"; {self.spec.notes}" if self.spec.notes else ""
        return f"{self.name} detected but no kernels implemented yet{detail}"

    @functools.lru_cache(maxsize=1)
    def caps(self) -> Caps:
        """构建能力记录，并从驱动解析出设备名。

        做了记忆化，因为模型每次分配决策都会读它；每次调用都读一次设备名等于为每个
        缓冲区做一次驱动往返。

        Returns:
            模型代码用来读 dtype、设备和行为标志的记录。``native`` 是 ``build_ops``
            实际返回的内容，所以横幅里的 ``native=N/17`` 是推导出来的而不是声明出来的。
        """
        table = self._native_table()
        return Caps(
            name=self.name,
            device=self.spec.detection.namespace,
            device_index=0,
            device_name=device_name(self.spec.detection.namespace),
            arch="",
            native=frozenset(table.implemented()) if self.spec.has_kernels else frozenset(),
            act_dtype=self.spec.act_dtype,
            weight_dtype=self.spec.weight_dtype,
            graph=self.spec.graph,
            autotune=self.spec.autotune,
            attn_split=self.spec.attn_split,
            batch_attn=self.spec.batch_attn,
            int8_weights=self.spec.int8_weights,
            recurrent_state=True,
            weight_layout=self.spec.weight_layout,
            index_dtype=self.spec.index_dtype,
        )

    def _native_table(self) -> Ops:
        """返回该厂商实现了的部分，不补缺口。

        Returns:
            厂商的部分表，占位实现则为空表。与 :meth:`ops` 分开是因为
            ``caps.native`` 必须报告真正原生的内容，而不是参考实现补进来的那些。
        """
        if not self.spec.has_kernels or self.spec.build_ops is None:
            return Ops()
        return self.spec.build_ops()

    @functools.lru_cache(maxsize=1)
    def ops(self) -> Ops:
        """返回后端完整的算子表，原生条目加参考实现补齐。

        正是这个方法让只实现了一部分的平台能跑起来：只有 ``linear`` 的厂商依然回答全部
        17 个名字，因为迁移之后模型侧已经不留回退分支了。做了记忆化，因为引擎每个请求
        会访问这张表成千上万次。

        Returns:
            每个规范算子名都可调用的表。
        """
        return reference.fill(self._native_table(), self.caps())

    @functools.lru_cache(maxsize=1)
    def device(self) -> torch.device:
        """返回本后端所有分配应该落到的 ``torch.device``。

        Returns:
            ``torch.device(namespace, 0)``，惰性构建，因为构造一个 device 可能初始化
            厂商运行时。
        """
        return make_device(self.caps())

    def describe(self) -> str:
        """把本后端渲染成一行诊断信息。

        Returns:
            ``"<name>: <state> (<reason>)"``，与
            :func:`coffin.backends.registry.describe_all` 打印的格式一致。它不构建
            ``torch.device``，所以在选择之前打印也是安全的。
        """
        state = "usable" if self.usable() else ("placeholder" if self.probe() else "absent")
        reason = self.reason()
        return f"{self.name}: {state}{f' ({reason})' if reason else ''}"

    def __repr__(self) -> str:                    # pragma: no cover - debug aid
        return f"SpecBackend({self.name!r})"


def spec_fields() -> tuple[str, ...]:
    """按声明顺序返回 :class:`VendorSpec` 的字段名。

    Returns:
        dataclass 的字段名。有测试钉住它，这样给 spec 新增字段不会悄悄变成
        :class:`SpecBackend` 没处理的字段——后者按名字读每个字段，否则漏掉新字段也
        不会有任何提示。
    """
    return tuple(f.name for f in dataclasses.fields(VendorSpec))
