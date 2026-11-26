"""两个和「不让 autograd 记账」有关的上下文，写成不带 torch 名字的形式。

``@torch.inference_mode()`` / ``torch.no_grad()`` 在推理链路上只做一件事：告诉
autograd「这段不要记账」。这件事本身就是 torch 的概念，所以本模块的做法是——

* 真 torch 在场：原样用它的实现，行为逐位不变；
* 不在场（或在场的是 ``cxrt.torchshim`` 这个壳）：就是个空上下文，什么都不做。

这样 ``engine/llm_engine.py`` 与 ``run_chat.py`` 不必再 ``import torch`` 来拿一个
装饰器，而「有 torch 时行为不变」这条约束仍然成立。

Module: ``coffin/cxrt``。只 import 标准库，**不 import torch**。
"""

from __future__ import annotations

import sys
from contextlib import contextmanager


@contextmanager
def inference_mode():
    """``torch.inference_mode()`` 的替代品；没有 torch（或只有壳）时是空上下文。

    可以用作装饰器（``@inference_mode()``）或 ``with`` 语句：底层是
    :class:`contextlib._GeneratorContextManager`，它自己会在重入时重建生成器。
    """
    torch = sys.modules.get("torch")
    enter = getattr(torch, "inference_mode", None) if torch is not None else None
    if enter is None:
        yield
        return
    with enter():
        yield


@contextmanager
def no_grad():
    """``torch.no_grad()`` 的替代品；语义同上。"""
    torch = sys.modules.get("torch")
    enter = getattr(torch, "no_grad", None) if torch is not None else None
    if enter is None:
        yield
        return
    with enter():
        yield
