"""静态扫描：预填充投影的调用闭包里还剩哪些 torch 调用。

Module: ``coffin/tools``。只 import 标准库，不 import torch，也不需要 GPU——它读的是
``backends/corex/__init__.py`` 的语法树。

为什么需要它：``import torch`` 这件事短期内去不掉（张量本身由 torch 分配、后端契约就是
torch 张量），所以「torch 不能出现在推理链路里」只能按**算子**来验收：链路上不许有
``torch.matmul`` / ``torch.mm`` / ``F.linear`` 这类会真的算一次的东西，而
``torch.empty`` / ``torch.zeros`` 只是分配，``.to()`` 只是窄化。这个脚本把这条线画在
代码里，改坏了会立刻红。

用法::

    python tools/audit_torch_path.py                       # 默认查预填充三条入口
    python tools/audit_torch_path.py --entry linear_prefill
    python tools/audit_torch_path.py --all                 # 列出全部 torch 调用点

退出码 ``0`` 表示闭包里没有算子，``1`` 表示有。
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

#: 会真的算一次的张量算子。命中即失败。
_COMPUTE = frozenset({
    "addmm", "addbmm", "baddbmm", "bmm", "chain_matmul", "dot", "einsum", "inner",
    "matmul", "mm", "mv", "outer", "tensordot",
    "conv1d", "conv2d", "conv3d", "conv_transpose1d", "conv_transpose2d",
    "linear", "linear?", "lstm", "rnn",
    "softmax", "log_softmax", "scaled_dot_product_attention", "layer_norm",
    "group_norm", "batch_norm", "rms_norm",
    "topk", "sort", "cumsum", "argmax", "argmin", "unique",
})

#: 只是分配 / 视图 / 搬运，不算算子。
_CONTAINER = frozenset({
    "empty", "empty_like", "zeros", "zeros_like", "ones", "ones_like", "full",
    "full_like", "tensor", "as_strided", "from_numpy", "frombuffer", "cat",
    "stack", "narrow", "view", "reshape", "contiguous", "clone", "copy_",
    "device", "dtype", "no_grad", "inference_mode", "cuda",
})

#: 整数索引簿记：生成行号表、把 block 行号折成 batch 号。算得出数，但不是张量数学。
_INDEX = frozenset({"arange", "div", "floor_divide", "remainder", "searchsorted"})

#: 默认要验收的入口：预填充的分派与两条读方，加上解码的两条入口。
_DEFAULT_ENTRIES = ("linear_prefill", "_w8a8_lib_rows", "_gemm_nk_rows", "linear",
                    "_gemv_rows")

#: 默认不展开的函数：没有。``_gemv_rows`` / ``linear`` 的 nk 与稠密两支都已换到
#: ``cxrt`` 与 ``cublas``，``kn`` 遗留的那几条（``_kn_prefill_torch`` /
#: ``_w8a8_tcu_rows`` / 转置缓存）已经整条删掉，所以闭包应当整条干净——留着 skip
#: 等于给回归开后门。
_DEFAULT_SKIP: tuple[str, ...] = ()

#: 纯 torch 的**回退**实现：只在扩展/工具链不可用时才走，不在推理链路上。
#: ``_solve_blocked`` 是 ``triangular_solve`` 的回退（没有工具链的主机也要能端到端跑），
#: 有 ABI 出口 ``coffin_triangular_solve`` 时它根本不会被调用。
_FALLBACK = ("_solve_blocked",)


def _module_funcs(path: Path) -> dict[str, ast.AST]:
    """收集模块顶层的函数定义。

    Args:
        path: 源文件。

    Returns:
        函数名到定义的映射。
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef,
                                                           ast.AsyncFunctionDef))}


def _torch_calls(node: ast.AST) -> list[tuple[str, int]]:
    """找出一个函数体里的 ``torch.<name>(...)`` 与 ``F.<name>(...)`` 调用。

    Args:
        node: 函数定义。

    Returns:
        ``(算子名, 行号)`` 列表。
    """
    out: list[tuple[str, int]] = []
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        fn = sub.func
        if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name):
            if fn.value.id in ("torch", "F", "nn", "torch.nn"):
                out.append((fn.attr, getattr(sub, "lineno", 0)))
    return out


def _local_calls(node: ast.AST, known: dict[str, ast.AST]) -> list[str]:
    """找出一个函数体里调用的同模块函数名（只认 ``Name`` 形式）。

    Args:
        node: 函数定义。
        known: 模块顶层函数名表。

    Returns:
        被调用的本地函数名列表。
    """
    out = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
            if sub.func.id in known:
                out.append(sub.func.id)
    return out


def _layout_calls(node: ast.AST) -> list[tuple[str, int]]:
    """找出 ``.t().contiguous()`` 这类转置调用（kn 那套 ``pack.as_kn`` 已不存在）。

    Args:
        node: 函数定义。

    Returns:
        ``(名字, 行号)`` 列表。
    """
    out = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
            if isinstance(sub.func.value, ast.Name) and sub.func.value.id == "pack":
                out.append((sub.func.attr, getattr(sub, "lineno", 0)))
    return out


def _walk(entry: str, known: dict[str, ast.AST], skip: tuple[str, ...],
          seen: set[str]) -> list[str]:
    """展开 ``entry`` 的调用闭包（跳过 ``skip`` 与已访问的函数）。

    Args:
        entry: 入口函数名。
        known: 模块顶层函数名表。
        skip: 不展开的函数名。
        seen: 已访问集合，就地更新。

    Returns:
        闭包里除 ``entry`` 之外的函数名。
    """
    if entry in seen or entry in skip or entry not in known:
        return []
    seen.add(entry)
    out: list[str] = []
    for callee in _local_calls(known[entry], known):
        out.append(callee)
        out.extend(_walk(callee, known, skip, seen))
    return out


def main(argv: list[str] | None = None) -> int:
    """命令行入口。

    Args:
        argv: 参数列表；``None`` 时读 ``sys.argv``。

    Returns:
        进程退出码。
    """
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=str(here / "backends" / "corex" / "__init__.py"))
    ap.add_argument("--entry", default="", help="逗号分隔的入口函数名")
    ap.add_argument("--skip", default=",".join(_DEFAULT_SKIP))
    ap.add_argument("--all", action="store_true", help="列出全模块的 torch 调用点")
    args = ap.parse_args(argv)

    path = Path(args.source)
    known = _module_funcs(path)
    skip = tuple(s for s in args.skip.split(",") if s)

    bad = 0
    if args.all:
        for name, node in known.items():
            calls = _torch_calls(node)
            if calls:
                tag = "（回退：扩展不可用时才走）" if name in _FALLBACK else ""
                print(f"{name}{tag}: " + ", ".join(f"torch.{c}@{l}" for c, l in calls))
        return 0

    entries = [e for e in (args.entry or ",".join(_DEFAULT_ENTRIES)).split(",") if e]
    for entry in entries:
        if entry not in known:
            print(f"{entry}: 没有这个函数", file=sys.stderr)
            bad += 1
            continue
        seen: set[str] = set()
        closure = _walk(entry, known, skip, seen)
        reached = [entry] + [c for c in closure if c in seen]
        print(f"[{entry}] 闭包: {', '.join(sorted(reached))}")
        for fn in sorted(set(reached)):
            node = known.get(fn)
            if node is None:
                continue
            for name, line in _torch_calls(node):
                if name in _COMPUTE:
                    kind = "算子"
                elif name in _CONTAINER:
                    kind = "分配/搬运"
                elif name in _INDEX:
                    kind = "索引簿记"
                else:
                    kind = "未分类"
                print(f"  {fn}:{line} torch.{name}  ({kind})")
                if kind == "算子":
                    bad += 1
            for name, line in _layout_calls(node):
                print(f"  {fn}:{line} pack.{name}  (布局转换)")
        print()

    if bad:
        print(f"FAIL: {bad} 处算子调用仍在闭包里")
        return 1
    print("OK: 闭包里没有 torch 算子，只有分配与窄化")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
