"""厂商后端发现实现：一个平台一个文件

新增一个平台就是丢进一个 ``vendors/<name>.py``（等它需要 ``kernels/`` 目录时
再升成 ``vendors/<name>/`` 包），在里面定义模块级的 ``SPEC``。别的地方都不用改：
本模块会发现它，:mod:`coffin.backends.registry` 把它与其余后端一起排序，CLI
也接受它的名字。反过来做——每个厂商改一处元组加一处名字映射——就是平台层在
有人漏改两处之一的那一刻开始崩坏的老路。

后端发现的实现走目录扫描而不是 entry-point 插件系统，是有意的。这些平台是本仓库内的
一方文件，扫描没有部署故事可讲错，而且故障范围是局部的：import 就抛异常的
厂商文件会被报成指名道姓的错误，而不是一个来路不明的后端缺失。

Module: ``coffin/backends``。只 import ``vendor`` 和标准库；被扫描的模块本身
会被 import，所以这里是唯一执行厂商代码的地方——而且只在调用 :func:`load` 时
执行，import 时不执行。因此厂商模块必须在从未见过其硬件的主机上也能 import；
这条规则写在 ``backends/vendor.py`` 里。

Constraints:
    * import 按模块名排序，保证扫描结果确定。同优先级厂商之间的顺序不得取决于
      文件系统遍历顺序，否则哪块卡被哪个后端认领会随机器而变。
    * import 失败的模块要被报出来，不能静默跳过，除非它是文档里写明豁免的
      那几个（:data:`_IGNORED`）。一个悄无声息消失的厂商后端，正是这一层要防的
      bug。
    * 发现阶段不得编译或探测任何东西：它只 import 模块、读它们的 ``SPEC``。
      探测留到后面，在 registry 里做且只做一次。
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import Iterator

from backends.vendor import SpecBackend, VendorSpec

#: 存放厂商声明的目录，即 ``coffin/backends/vendors``。
VENDOR_DIR = Path(__file__).resolve().parent / "vendors"

#: ``vendors/`` 下不是平台的模块名。前导下划线表示私有，一个还在开发中的
#: 厂商就该先这么命名，等它能用了再改。
_IGNORED = frozenset({"__init__"})


def modules() -> tuple[str, ...]:
    """列出可 import 的厂商模块名，已排序。

    Returns:
        形如 ``"backends.vendors.iluvatar"`` 的点分模块名，每个厂商文件或包一条，
        已排序以保证扫描结果确定。目录不存在时返回空元组——对于没有任何厂商
        后端的构建，这是合法状态。
    """
    if not VENDOR_DIR.is_dir():
        return ()
    names = []
    for entry in pkgutil.iter_modules([str(VENDOR_DIR)]):
        if entry.name in _IGNORED or entry.name.startswith("_"):
            continue
        names.append(f"backends.vendors.{entry.name}")
    return tuple(sorted(names))


def load(module_name: str) -> tuple[VendorSpec, SpecBackend]:
    """import 一个厂商模块，并包好它的 ``SPEC``。

    Args:
        module_name: 来自 :func:`modules` 的点分模块名。

    Returns:
        ``(spec, backend)``。backend 在这里就构造出来，所以没有通过
        :meth:`VendorSpec.validate` 的 spec 在发现阶段就会暴露，而不是等到在某个
        没人有过的硬件上发第一个请求时才炸。

    Raises:
        RuntimeError: 模块没有模块级 ``SPEC``，或者 import 本身抛了异常。两者都会
            加上模块名再包装，因为一个坏掉的厂商文件自己的 traceback 并不会说
            它本该代表哪个平台。
        ValueError: spec 非法，由 :meth:`VendorSpec.validate` 原样透出。
    """
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise RuntimeError(
            f"vendor backend {module_name!r} failed to import: "
            f"{type(exc).__name__}: {exc}") from exc
    spec = getattr(module, "SPEC", None)
    if not isinstance(spec, VendorSpec):
        raise RuntimeError(
            f"vendor backend {module_name!r} does not define a module-level SPEC: "
            f"VendorSpec; found {type(spec).__name__}")
    return spec, SpecBackend(spec)


def discover() -> tuple[tuple[VendorSpec, SpecBackend], ...]:
    """按确定顺序 import 每一个厂商后端。

    Returns:
        每个已发现厂商一条 ``(spec, backend)``，按模块名排序。

    Raises:
        RuntimeError: 某个厂商模块坏掉了，见 :func:`load`。它只在选择失败的路径上
            被调用，好让坏文件在那里浮出水面；同时它不在 ``backends`` 的 import
            路径上，只用一个后端的容器才不会被迫保证其余后端也能 import。
    """
    return tuple(load(name) for name in modules())


def iter_vendors() -> Iterator[tuple[VendorSpec, SpecBackend]]:
    """惰性产出厂商后端，供必须容忍单个失败的调用方使用。

    Returns:
        遍历 ``(spec, backend)`` 的迭代器。与 :func:`discover` 不同，它不抛异常：
        坏掉的厂商被跳过。registry 的自动探测用它——一个不可用的平台不该妨碍
        另一个平台被选中。
    """
    for name in modules():
        try:
            yield load(name)
        except Exception:                         # pragma: no cover - 坏掉的厂商
            continue
