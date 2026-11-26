"""把 ``cxrt.tt`` 与 ``cxrt.ttnn`` 组装成一个 torch 形状的模块。

Module: ``coffin/cxrt``。模型层各文件只改一行 import：

    -   import torch
    +   from cxrt import torchshim as torch

之后 ``torch.Tensor`` / ``torch.zeros`` / ``torch.nn.Module`` / ``torch.nn.functional``
这些写法全部照旧生效——因为本模块把 ``tt`` 的公开名字整体铺到自己的命名空间里，并把
``nn`` 挂成 ``ttnn``。

为什么不直接给 ``cxrt.tt`` 挂 ``nn``：那会让 ``tt`` 与 ``ttnn`` 互为导入方，在
``cxrt`` 包初始化期间容易形成半初始化模块。这层 facade 把方向固定成单向。
"""

from __future__ import annotations

from typing import Any

from cxrt import tt as _tt
from cxrt import ttnn as nn

#: ``tt`` 里所有公开名字都照搬过来；``nn`` 单独挂，避免 ``tt`` 里出现同名。
for _name, _value in list(vars(_tt).items()):
    if _name.startswith("_") or _name in {"nn"}:
        continue
    globals()[_name] = _value

#: ``torch.nn.functional`` 的等价。
#:
#: 模型里常见写法是 ``F = torch.nn.functional`` 再 ``F.pad(...)``，所以这里必须让
#: ``nn.functional`` 拿得到函数。
nn.functional = nn.functional

#: 兼容 ``torch.nn.functional as F`` 的直接导入。
F = nn.functional

#: 少数下游会 ``import torch._dynamo``；本壳不编译图，给个空壳占位。
_dynamo = _tt._dynamo_stub()

# torch 的短别名。它们在 ``tt`` 里不能定义成 ``float = float32``——那会把模块内的
# 内建 ``float`` 遮蔽掉——所以放在这一层，只有 ``torch.float`` 这个拼法能拿到。
float = _tt.float32
float64 = _tt.double
int = _tt.int32
int64 = _tt.int64
long = _tt.int64
bool = _tt.bool
half = _tt.float16


def __getattr__(name: str) -> Any:
    """把未导出的名字继续往下顺延给 ``tt``，让壳与实现不必逐个人工同步。

    Args:
        name: 属性名。

    Returns:
        ``tt`` 上的同名属性。

    Raises:
        AttributeError: 两边都没有。
    """
    try:
        return getattr(_tt, name)
    except AttributeError:
        raise AttributeError(f"module 'cxrt.torchshim' has no attribute {name!r}") from None


__all__ = sorted(n for n in vars(_tt) if not n.startswith("_")) + ["nn", "F"]
