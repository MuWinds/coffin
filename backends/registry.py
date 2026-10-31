"""为本进程挑出唯一要跑的后端：环境变量覆盖，然后自动探测，最后参考实现。

选择只发生一次，在 import 时，此后不再重来：没有运行时热切换，换后端就是换进程。
顺序是固定且刻意平淡的——``COFFIN_BACKEND``（精确名字）优先，方便调试和 A/B；
否则逐个探测已注册的后端，最强的认领胜出，纯 torch 参考实现作为兜底的最后候选。

后端的两个来源，一份清单

*手写*后端（:data:`BUILTIN`）是需要真代码的模块：``hip`` 有内核、调优和一条编译
路径，``reference`` 是其余一切解析到的兜底。*被发现*的后端住在
``backends/vendors/`` 里，是声明式的 spec（见 :mod:`coffin.backends.discovery`），
所以加一个平台就是加一个文件，这里一个字都不用改。:func:`candidates` 把两者合并
成唯一的有序序列，选择、日志和错误文案都读它。

检测用的是**认领强度**而不是布尔值，因为同一个设备可能被多个厂商应答。Iluvatar 的
CoreX 与 Hygon DCU 一样经由 ``torch.cuda`` 应答，两者都可能看到「存在一个 CUDA
设备」；匹配到设备*名字*的厂商排在只匹配到命名空间的前面，于是 Iluvatar 的 spec
胜出。Enflame 和 Moore Threads 上命名空间本身就是唯一的（``gcu``、``musa``），所以
命名空间认领就够了。这一条规则覆盖两种情形，registry 不必知道有哪些厂商。

Module: ``coffin/backends``。与 ``protocol`` 一起位于依赖图底部：它可以 import
``torch``、``protocol``、``vendor``、``discovery`` 以及内置后端模块，不得 import
``coffin/ops``、``models``、``engine``、``serving`` 或 ``cache`` 里的任何东西。

Constraints:
    * ``select()`` 是唯一会就后端选择抛出异常的地方。它必须点名被请求的后端、试过的
      每个候选，以及每个候选的 :meth:`reason`——一个被悄悄选中的参考后端，看起来
      就和一个莫名其妙很慢的部署一模一样。
    * 显式的 ``COFFIN_BACKEND`` 绝不会落到别的后端。拼写错误或坏掉的构建必须响亮
      地失败；悄悄改跑参考实现只会把故障藏在 10 倍减速后面。
    * import 本模块不得编译、构建或探测任何东西：它只读目录列表和模块名。纯 CPU
      主机上的 ``import backends`` 必须能跑，测试套件就指望这一点。
"""

from __future__ import annotations

import importlib
import os
from typing import Sequence

from backends import discovery
from backends.protocol import Backend, Caps, Ops

#: 按名字钉住后端的环境变量；为空或未设置表示自动探测。只在 import
#: :mod:`coffin.backends` 时读一次。
ENV_VAR = "COFFIN_BACKEND"

#: 手写后端，按探测顺序排列。``backends/vendors/`` 下的一切都是被发现后合并进来的；
#: 只有需要真代码的后端放在这里。
#:
#: ``hip`` 和 ``corex`` 之所以手写而不是声明式 spec，是因为它们带着内核、内核切块、
#: autotune 和一条编译路径——这些一个声明都表达不了。它们还共用 ``cuda`` 命名空间，
#: 所以谁胜出由各自后端的 probe 决定（ROCm 看 ``torch.version.hip``，CoreX 看设备
#: 名），而不是由这里的顺序决定。``reference`` 放最后，因为它永远报可用，排在它后面
#: 的后端都到不了。
BUILTIN: tuple[str, ...] = ("hip", "corex", "reference")

#: 每个内置后端在本包内的模块名。
_BUILTIN_MODULE = {
    "hip": "backends.hip",
    "corex": "backends.corex",
    "reference": "backends.reference",
}

#: 每个内置后端的探测优先级，只用于打破两个后端同等强度认领同一设备时的平局。参考
#: 实现排最后因为它永远报可用；``hip`` 和 ``corex`` 是故意同优先级的，共享的
#: ``cuda`` 命名空间里谁都不能靠排序胜出。
_BUILTIN_PRIORITY = {"hip": 0, "corex": 0, "reference": 1000}


class Candidate:
    """一个可被选中的后端，以及它在这台主机上的认领。

    Attributes:
        name: 后端标识符，即 ``COFFIN_BACKEND`` 接受的名字。
        backend: 协议驱动的对象；手写后端是模块，声明式厂商是
            :class:`~coffin.backends.vendor.SpecBackend`。
        claim: 主机匹配的强度，来自 :func:`coffin.backends.vendor.detect`（不参与
            认领排序的手写后端为 ``0``）。越高越优。
        priority: 认领相等时的平局裁决，越小越优。
    """

    __slots__ = ("name", "backend", "claim", "priority")

    def __init__(self, name: str, backend: Backend, claim: int = 0,
                 priority: int = 0) -> None:
        """记录一个后端以及它在这台主机上的认领。

        Args:
            name: 后端标识符，即 ``COFFIN_BACKEND`` 接受的名字。
            backend: 协议驱动的模块或 :class:`~coffin.backends.vendor.SpecBackend`。
            claim: 来自 :func:`coffin.backends.vendor.detect` 的认领强度；手写后端为
                ``0``，因为它不靠认领竞争。
            priority: 认领相等时的平局裁决，越小越优。参考实现取最大值，所以永远
                排最后。
        """
        self.name = name
        self.backend = backend
        self.claim = claim
        self.priority = priority

    def rank(self) -> tuple[int, int, str]:
        """排序键：认领最强者优先，其次优先级，最后名字。

        Returns:
            ``(-claim, priority, name)``。取反是因为升序排序要把最强的排在前面；
            末尾的名字让顺序成为全序，两个并列的厂商就不会让选择结果取决于 import
            顺序。
        """
        return (-self.claim, self.priority, self.name)


def _builtins() -> tuple[Candidate, ...]:
    """加载每一个手写后端。

    Returns:
        按 :data:`BUILTIN` 声明顺序，每个一项 :class:`Candidate`。import 失败的
        内置后端会直接抛出：它们是一方模块，失败是本仓库的 bug，而不是某个恰好
        不在的平台。
    """
    out = []
    for name in BUILTIN:
        module = importlib.import_module(_BUILTIN_MODULE[name])
        out.append(Candidate(name, module, priority=_BUILTIN_PRIORITY[name]))
    return tuple(out)


def _vendors() -> tuple[Candidate, ...]:
    """加载每一个被发现的厂商后端，并评估它在这台主机上的认领。

    认领在这里只读一次，因为它是整个排序的依据；事后再问每个厂商一遍，会让主机
    状态和排序结论对不上。

    Returns:
        每个能干净 import 的厂商文件一项 :class:`Candidate`。import 失败的厂商被
        跳过：一个坏掉的平台文件不能让整层都选不出后端，它会由
        :func:`describe_all` 报出来。
    """
    out = []
    for spec, backend in discovery.iter_vendors():
        try:
            claim = backend.claim()
        except Exception:                         # pragma: no cover - 探测出错
            claim = 0
        out.append(Candidate(spec.name, backend, claim=claim,
                             priority=spec.priority))
    return tuple(out)


def candidates() -> tuple[Candidate, ...]:
    """返回所有可选后端，认领最强的在前。

    Returns:
        手写后端与发现的厂商合并后按 :meth:`Candidate.rank` 排序。并列时参考实现
        排最后，因为它的优先级最高，这正是它成为保证兜底的原因。
    """
    return tuple(sorted(_builtins() + _vendors(), key=Candidate.rank))


def _backend_named(name: str) -> Backend | None:
    """按名字解析一个后端，不探测它。

    Args:
        name: 来自 ``COFFIN_BACKEND`` 或调用方的后端标识符。

    Returns:
        后端对象；没有候选叫这个名字时返回 ``None``。先查内置模块映射，这样显式
        钉住某个后端时不必付一次厂商目录扫描的代价。

    Raises:
        ImportError: 具名内置后端存在但模块 import 不了。这里报出来而不是吞掉，
            因为一个被显式钉住却悄悄变成别的后端的情形，正是这一层要防的故障。
    """
    if name in _BUILTIN_MODULE:
        return importlib.import_module(_BUILTIN_MODULE[name])
    for spec, backend in discovery.iter_vendors():
        if spec.name == name:
            return backend
    return None


def available_names() -> tuple[str, ...]:
    """返回本次构建可选的后端名，按探测顺序。

    Returns:
        名字列表，认领最强者在先。被发现的厂商也包含在内，所以新加一个文件就成了
        合法的 ``COFFIN_BACKEND`` 取值，不必修改本模块。
    """
    return tuple(candidate.name for candidate in candidates())


def _load(name: str) -> Backend:
    """按名字解析后端，失败时连同已知名字一起抛出。

    Args:
        name: 后端标识符。

    Returns:
        后端对象。

    Raises:
        KeyError: 没有候选叫这个名字，或者某个被发现的厂商文件 import 失败。消息
            会列出可用项，因为厂商文件缺失是最常见的原因。
    """
    backend = _backend_named(name)
    if backend is None:
        raise KeyError(f"unknown backend {name!r}; known: {', '.join(available_names())}")
    return backend


def _describe(candidate: Candidate) -> str:
    """给某个候选生成一行状态，供选择错误的文案使用。

    Args:
        candidate: 后端以及它做出的认领。

    Returns:
        ``"name: state (reason)"``。只在失败路径上调用，因为生成 reason 可能
        import 一个可选依赖不存在的厂商。
    """
    try:
        present = candidate.backend.probe()
        ok = candidate.backend.usable() if present else False
        why = "" if ok else (candidate.backend.reason() or "not usable")
    except Exception as exc:                      # pragma: no cover - 探测出错
        return f"{candidate.name}: probe raised {type(exc).__name__}: {exc}"
    state = "usable" if ok else ("present, unusable" if present else "absent")
    return f"{candidate.name}: {state}{(' (' + why + ')') if why else ''}"


def select(requested: str | None = None) -> Backend:
    """为本进程挑一个后端。

    Args:
        requested: 来自 ``COFFIN_BACKEND`` 的后端名；``None`` 表示读环境变量再自动
            探测。显式传值供测试使用，这样不必改环境变量就能走到失败路径。

    Returns:
        选中的后端。

    Raises:
        RuntimeError: 显式给的名字不是已知后端，或显式指定的后端不可用。消息会逐个
            列出候选及其探测结果和 reason，因为常见原因是主机上的驱动或工具链问题，
            而不是名字写错。
        RuntimeError: 自动探测找不到任何可用后端。这只有当参考后端本身坏掉时才可能
            发生，所以那段消息就是一份 bug 报告。
    """
    explicit = requested if requested is not None else os.environ.get(ENV_VAR, "")
    explicit = (explicit or "").strip()

    if explicit:
        try:
            backend = _load(explicit)
        except KeyError as exc:
            raise RuntimeError(
                f"{ENV_VAR}={explicit!r} is not a known backend; "
                f"known: {', '.join(available_names())}. Unset it to auto-detect."
            ) from exc
        if not backend.usable():
            raise RuntimeError(
                f"{ENV_VAR}={explicit!r} was requested but is not usable on this "
                f"host: {backend.reason() or 'no reason reported'}. "
                f"Unset it to auto-detect, or set {ENV_VAR}=reference to run the "
                f"pure-torch backend.\nCandidates: "
                + "; ".join(_describe(candidate) for candidate in candidates()))
        return backend

    for candidate in candidates():
        try:
            if candidate.backend.probe() and candidate.backend.usable():
                return candidate.backend
        except Exception:                          # pragma: no cover - 探测出错
            continue

    raise RuntimeError(
        "no usable backend: the reference backend reported unusable, which is a bug "
        "in backends.reference rather than a host problem.\nCandidates: "
        + "; ".join(_describe(candidate) for candidate in candidates()))


#: 选中的后端。任何需要选择结果的地方都 import 下面某个访问器，而不是直接拿这个名字。
_backend: Backend | None = None


def backend() -> Backend:
    """返回进程级后端，首次使用时才做选择。

    Returns:
        选中的后端；每次调用都是同一个对象，因为选两次要么会重建内核，要么会让进程
        的两部分对设备产生分歧。

    Raises:
        RuntimeError: 选择失败，带上 :func:`select` 的诊断信息。
    """
    global _backend
    if _backend is None:
        _backend = select()
    return _backend


def _reset_for_tests(module: Backend | None = None) -> None:
    """强行改写记忆化的选择结果，好让测试去跑另一个后端。

    Args:
        module: 要钉住的后端，绕过 :func:`select`；``None`` 清掉记忆，下次
            :func:`backend` 调用重新选择。

    Returns:
        无。不是公开 API：它存在是因为重载过后端包的测试必须能重新钉住模块级的表。
    """
    global _backend
    _backend = module


def ops() -> Ops:
    """返回所选后端的完整算子表。

    Returns:
        表，每个名字都可调用：后端的原生条目，加上它没实现的一切的参考实现。模型
        代码在 import 时绑定的就是这个对象。

    Raises:
        RuntimeError: 选择失败。
    """
    return backend().ops()


def caps() -> Caps:
    """返回所选后端的能力记录。

    Returns:
        模型代码用来读 dtype、设备和行为标志的那条记录。

    Raises:
        RuntimeError: 选择失败。
    """
    return backend().caps()


def status() -> str:
    """渲染所选后端的一行启动横幅。

    Returns:
        ``backend=hip device=cuda:0 native=17/17 graph=1 autotune=1``。它替代传输层
        原先打印的 ``hip=...`` 字段：没有这一行，一次悄悄回退到参考实现的运行看起来
        就像模型慢，而不是内核没编出来。

    Raises:
        RuntimeError: 选择失败。
    """
    b = backend()
    c = b.caps()
    return (f"backend={c.name} device={b.device} "
            f"native={len(c.native)}/{len(Ops.NAMES)} "
            f"graph={int(c.graph)} autotune={int(c.autotune)}")


def describe_all() -> Sequence[str]:
    """为每个候选渲染一行状态，供诊断转储使用。

    Returns:
        按探测顺序，每个候选一条 ``name: state``。与 :func:`status` 不同，它会碰每个
        后端的 ``probe``，所以这是诊断调用而不是启动调用。import 失败的厂商文件会作为
        错误行出现而不是被略去：一个悄无声息消失的后端，正是这里要防的 bug。
    """
    lines = [_describe(candidate) for candidate in candidates()]
    for module_name in discovery.modules():
        try:
            discovery.load(module_name)
        except Exception as exc:
            lines.append(f"{module_name}: BROKEN ({exc})")
    return tuple(lines)
