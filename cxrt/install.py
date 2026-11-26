"""把 ``cxrt.torchshim`` 装进 ``sys.modules['torch']``，让既有代码一行不改就能跑。

Module: ``coffin/cxrt``。模型层有 1895 处 ``torch.*``，逐个文件改 import 会把
「去掉 torch」这件事变成一次大重构；而这件事的验收标准是「``sys.modules`` 里没有
真的 ``torch``」，不是「代码里不许出现 import torch」。所以这里做一次进程级的别名：

    import cxrt.install        # 之后再 import torch 拿到的是壳

三条约束：

1. **真的 torch 在场就直接拒绝**。别名只有在「没有 torch」时才成立；如果环境里装了
   torch，静默替换会让下游以为自己用的是真 torch，这类错误最难查。
2. 不注册 ``torch.cuda`` 之类的子模块为真模块——``sys.modules`` 里写 ``torch.cuda``
   会让 ``import torch.cuda`` 走缓存，行为与壳内的属性不一致。用属性即可。
3. 装完记下装了什么，``cxrt.install.verify()`` 能自查。
"""

from __future__ import annotations

import sys
import types
from typing import Any

#: 装进 ``sys.modules`` 的名字与对应对象。
_INSTALLED: dict[str, Any] = {}


def _mod(name: str, **attrs: Any) -> types.ModuleType:
    """造一个带属性的模块对象，供 ``import torch.nn.functional`` 这类语句命中。

    Args:
        name: 模块名。
        **attrs: 要挂上去的属性。

    Returns:
        模块对象。
    """
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def _submodules(facade: Any) -> dict[str, Any]:
    """``torch`` 下面被 ``import`` 语句直接点到的子模块。

    真实代码里出现的形态是 ``import torch.nn as nn``、``import torch.nn.functional as F``、
    ``from torch.utils.cpp_extension import load``。这些语句要求 ``sys.modules`` 里有对应
    名字，光有属性不够——所以这里把它们逐个登记成模块对象。

    Args:
        facade: :mod:`cxrt.torchshim`。

    Returns:
        ``{全名: 模块}``。
    """
    nn = facade.nn
    functional = _mod("torch.nn.functional", **{
        name: getattr(nn.functional, name)
        for name in dir(nn.functional) if not name.startswith("_")
    })
    parameter = _mod("torch.nn.parameter", Parameter=nn.Parameter)
    init = _mod("torch.nn.init", constant_=lambda *a, **k: None,
                zeros_=lambda *a, **k: None, ones_=lambda *a, **k: None,
                normal_=lambda *a, **k: None, kaiming_uniform_=lambda *a, **k: None,
                calculate_gain=lambda *a, **k: 1.0)
    cpp_extension = _mod(
        "torch.utils.cpp_extension",
        load=lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("cpp_extension.load is unavailable without the vendor toolchain")),
        CUDAExtension=object,
        BuildExtension=object,
        load_inline=lambda *a, **k: None,
    )
    utils = _mod("torch.utils", cpp_extension=cpp_extension,
                 data=_mod("torch.utils.data", Dataset=object, DataLoader=object))
    cuda = _mod("torch.cuda", is_available=facade.cuda.is_available,
                synchronize=facade.cuda.synchronize,
                current_device=facade.cuda.current_device,
                device_count=facade.cuda.device_count,
                memory_allocated=facade.cuda.memory_allocated,
                empty_cache=facade.cuda.empty_cache,
                Stream=object, Event=object)
    serialization = _mod("torch.serialization", load=facade.load, save=facade.save)
    profiler = _mod("torch.profiler", profile=facade.profiler.profile,
                    ProfilerActivity=object, record_function=lambda name: None)
    backends = _mod("torch.backends", cuda=_mod(
        "torch.backends.cuda", is_available=facade.cuda.is_available, matmul=object))
    # 单进程推理：``torch.distributed`` 只需要「没初始化、world=1、集合通信是空操作」。
    class _ReduceOp:
        SUM = "sum"
        MAX = "max"
        MIN = "min"
        PRODUCT = "prod"
        AVG = "avg"

    def _noop(*args: Any, **kwargs: Any) -> None:
        return None

    distributed = _mod(
        "torch.distributed",
        is_available=lambda: False,
        is_initialized=lambda: False,
        get_rank=lambda *a, **k: 0,
        get_world_size=lambda *a, **k: 1,
        init_process_group=_noop,
        destroy_process_group=_noop,
        barrier=_noop,
        all_reduce=_noop,
        all_gather=lambda *a, **k: None,
        all_gather_object=_noop,
        broadcast=_noop,
        ReduceOp=_ReduceOp,
        group=object,
        new_group=lambda *a, **k: None,
    )
    distributed.nn = _mod("torch.distributed.nn", all_reduce=_noop)
    nn_mod = _mod("torch.nn", **{name: getattr(nn, name) for name in dir(nn)
                                 if not name.startswith("_")})
    return {
        "torch.nn": nn_mod,
        "torch.nn.functional": functional,
        "torch.nn.parameter": parameter,
        "torch.nn.init": init,
        "torch.utils": utils,
        "torch.utils.cpp_extension": cpp_extension,
        "torch.cuda": cuda,
        "torch.serialization": serialization,
        "torch.profiler": profiler,
        "torch.backends": backends,
        "torch.distributed": distributed,
        "torch.distributed.nn": distributed.nn,
    }


def verify() -> bool:
    """确认当前进程里 ``torch`` 是壳而不是真包。

    Returns:
        ``True`` 表示 ``sys.modules['torch']`` 是本项目的壳。
    """
    from cxrt import torchshim

    mod = sys.modules.get("torch")
    return mod is torchshim


def install(force: bool = False) -> Any:
    """执行别名安装。

    Args:
        force: 即使 ``sys.modules`` 里已有 ``torch`` 也覆盖（默认不覆盖）。

    Returns:
        装进去的 facade 模块。

    Raises:
        RuntimeError: 真的 torch 已导入且 ``force`` 为假。
    """
    from cxrt import torchshim

    existing = sys.modules.get("torch")
    if existing is not None and existing is not torchshim and not force:
        origin = getattr(existing, "__file__", "?")
        raise RuntimeError(
            f"a real torch is already imported ({origin}); refusing to alias it")

    sys.modules["torch"] = torchshim
    _INSTALLED["torch"] = torchshim
    # 让 ``import torch.nn`` 这类语句把壳当成包看待（否则报 "not a package"）。
    torchshim.__path__ = []                             # type: ignore[attr-defined]
    for name, mod in _submodules(torchshim).items():
        sys.modules[name] = mod
        _INSTALLED[name] = mod
        # ``import torch.distributed as dist`` 会在导入后取 ``torch.distributed`` 这个
        # 属性；只登记 sys.modules 不够，属性也得挂上（否则报 cannot import name）。
        head = name.split(".", 1)[1]
        if getattr(torchshim, head, None) is None:
            setattr(torchshim, head, mod)
    return torchshim


def modules() -> dict[str, Any]:
    """已安装的别名表（只用于排查）。"""
    return dict(_INSTALLED)
