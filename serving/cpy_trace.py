"""一次性拷贝溯源：把"这块大拷贝是谁发的"从猜测变成栈。

只在 ``COFFIN_CPY_TRACE=1`` 时启用（默认 import 都不会发生），每种
（算子、形状、调用点）只打印一次，所以对计时没有持续影响。

为什么需要它：`ixprof --print-gpu-trace` 只给内核名，而 aten 的所有拷贝共用
`direct_copy_kernel_cuda` 一个符号，网格只能算出元素数（4096 块 = 2,097,152 元素、
1024 块 = 524,288 元素这两档各 8 次/步、合计 1.45 ms），定位不到调用点。

第一版只包了 ``copy_``/``clone``，结果 decode 期一次都没命中（只有四处加载期的
权重搬迁，外加 ``ops/delta.py:167`` 的 ``(32,128,128)``），所以这两档大拷贝必定是
从 ``cat``/``to``/``contiguous``/``__setitem__``/``index_copy_`` 或 **C++ 驱动**里发出的。
第二版把前五个全包上。

Module: ``coffin.serving``（诊断用，正式路径不 import）。
"""

from __future__ import annotations

import traceback

import torch

_SEEN: set = set()
_N = [0]
_CAP = 200
_MIN = 1 << 18


def _frames() -> str:
    """取调用栈里属于本仓的最近 3 帧。"""
    lines = traceback.format_stack(limit=12)[:-1]
    keep = [ln.strip() for ln in lines if "coffin" in ln]
    return " | ".join(keep[-3:])


def _report(tag: str, shape, extra: str = "") -> None:
    if _N[0] >= _CAP:
        return
    frames = _frames()
    key = (tag, str(shape), extra, frames)
    if key in _SEEN:
        return
    _SEEN.add(key)
    _N[0] += 1
    print("[cpytrace] %s %s %s %s" % (tag, shape, extra, frames), flush=True)


_copy_ = torch.Tensor.copy_
_clone = torch.Tensor.clone
_contig = torch.Tensor.contiguous
_to = torch.Tensor.to
_setitem = torch.Tensor.__setitem__
_index_copy = torch.Tensor.index_copy_
_cat = torch.cat


def _w_copy(self, src, *a, **k):
    if self.numel() >= _MIN:
        _report("copy_", tuple(self.shape), "<- %s %s" % (tuple(src.shape), self.dtype))
    return _copy_(self, src, *a, **k)


def _w_clone(self, *a, **k):
    if self.numel() >= _MIN:
        _report("clone", tuple(self.shape), str(self.dtype))
    return _clone(self, *a, **k)


def _w_contig(self, *a, **k):
    if self.numel() >= _MIN and not self.is_contiguous():
        _report("contiguous", tuple(self.shape), str(self.dtype))
    return _contig(self, *a, **k)


def _w_to(self, *a, **k):
    if self.numel() >= _MIN:
        _report("to", tuple(self.shape), str(self.dtype))
    return _to(self, *a, **k)


def _w_setitem(self, key, value):
    if isinstance(value, torch.Tensor) and value.numel() >= _MIN:
        _report("setitem", tuple(value.shape), str(value.dtype))
    return _setitem(self, key, value)


def _w_index_copy(self, dim, index, source, *a, **k):
    if source.numel() >= _MIN:
        _report("index_copy_", tuple(source.shape), "dim=%s" % dim)
    return _index_copy(self, dim, index, source, *a, **k)


def _w_cat(tensors, *a, **k):
    total = sum(t.numel() for t in tensors) if isinstance(tensors, (list, tuple)) else 0
    if total >= _MIN:
        _report("cat", "%d×%s" % (len(tensors), tuple(tensors[0].shape)),
                "total=%d" % total)
    return _cat(tensors, *a, **k)


torch.Tensor.copy_ = _w_copy
torch.Tensor.clone = _w_clone
torch.Tensor.contiguous = _w_contig
torch.Tensor.to = _w_to
torch.Tensor.__setitem__ = _w_setitem
torch.Tensor.index_copy_ = _w_index_copy
torch.cat = _w_cat
print("[cpytrace] installed", flush=True)
