"""``python3 tools/scan_torch_api.py`` —— 盘点推理链对 torch 的 API 依赖。

Module: ``coffin/tools``。给「把模型层也摘掉 torch」这一步用的清点工具：它不读
实现、只读**调用面**——``torch.X`` 属性链、``Tensor`` 上的方法名、以及
``torch.nn`` / ``F`` 这类子模块别名。产出一张按出现次数排序的表，``cxrt/tt.py``
就按这张表长出 API，而不是凭印象挑几个。

用法::

    python3 tools/scan_torch_api.py                 # 默认扫 models/ engine/ cache/ serving/ ops/
    python3 tools/scan_torch_api.py --top 200 --json out.json
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import sys

#: 默认扫描的子树。``backends`` 不在里面：适配层要整段替换，不是「改 import」。
DEFAULT_ROOTS = ("models", "engine", "cache", "serving", "ops")

#: 视为 torch 名字根的模块别名：``import torch as t`` 之后 ``t.randn`` 也算。
_TORCH_NAMES = {"torch"}


class _Visitor(ast.NodeVisitor):
    """收集两类事实：模块级属性链与张量方法调用。

    Attributes:
        attrs: ``{"torch.foo.bar": 次数}``。
        methods: ``{"Tensor.bar": 次数}``——只记接收者名字里含 ``x``/``t``/``q``
            这类张量的调用无法可靠判定，所以这里记的是「所有 ``.method(...)``」，
            由使用者按出现次数自己筛。
        files: 每个文件命中的次数，用于判断改造顺序。
    """

    def __init__(self) -> None:
        self.attrs: dict[str, int] = {}
        self.methods: dict[str, int] = {}
        self.files: dict[str, int] = {}

    def _bump(self, table: dict[str, int], key: str) -> None:
        table[key] = table.get(key, 0) + 1

    def visit_Attribute(self, node: ast.Attribute) -> None:
        chain = _dotted(node)
        if chain is not None:
            root, rest = chain.split(".", 1)
            if root in _TORCH_NAMES or root in {"F", "nn"}:
                self._bump(self.attrs, chain)
            elif rest:
                # ``x.new_empty`` / ``self.weight.dtype`` 这类：只留最后一段方法名，
                # 前缀随实现变化，留着会把同一件事拆成几十个键。
                self._bump(self.methods, rest)
        self.generic_visit(node)


def _dotted(node: ast.AST) -> str | None:
    """把 ``a.b.c`` 形式的表达式还原成点号串，其它返回 ``None``。

    Args:
        node: 任意 AST 节点。

    Returns:
        点号串，或 ``None``。
    """
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    elif isinstance(node, ast.Call):
        inner = _dotted(node.func)
        if inner is None:
            return None
        parts.append(inner + "()")
    else:
        return None
    return ".".join(reversed(parts))


def scan(root: pathlib.Path) -> _Visitor:
    """扫一棵子树。

    Args:
        root: 仓库内某目录。

    Returns:
        合并后的访问者。
    """
    total = _Visitor()
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        visitor = _Visitor()
        visitor.visit(tree)
        hits = sum(visitor.attrs.values()) + sum(visitor.methods.values())
        if hits:
            total.files[str(path)] = hits
        for key, n in visitor.attrs.items():
            total.attrs[key] = total.attrs.get(key, 0) + n
        for key, n in visitor.methods.items():
            total.methods[key] = total.methods.get(key, 0) + n
    return total


def main(argv: list[str] | None = None) -> int:
    """入口，返回进程退出码（总是 0：这是清点工具，不是门禁）。"""
    parser = argparse.ArgumentParser(description="盘点推理链对 torch API 的依赖")
    parser.add_argument("--root", default=".")
    parser.add_argument("--dirs", nargs="*", default=list(DEFAULT_ROOTS))
    parser.add_argument("--top", type=int, default=120)
    parser.add_argument("--json", default="")
    args = parser.parse_args(argv)

    base = pathlib.Path(args.root)
    merged = _Visitor()
    for name in args.dirs:
        sub = base / name
        if not sub.is_dir():
            print(f"[skip] {sub} 不存在")
            continue
        v = scan(sub)
        for key, n in v.attrs.items():
            merged.attrs[key] = merged.attrs.get(key, 0) + n
        for key, n in v.methods.items():
            merged.methods[key] = merged.methods.get(key, 0) + n
        for key, n in v.files.items():
            merged.files[key] = merged.files.get(key, 0) + n

    print(f"[sum] torch.* 调用面 {sum(merged.attrs.values())} 处，"
          f"属性方法面 {sum(merged.methods.values())} 处，文件 {len(merged.files)} 个")
    print("\n-- torch.* 前 %d --" % args.top)
    for key, n in sorted(merged.attrs.items(), key=lambda kv: -kv[1])[: args.top]:
        print(f"{n:6d}  {key}")
    print("\n-- 属性/方法名前 %d --" % args.top)
    for key, n in sorted(merged.methods.items(), key=lambda kv: -kv[1])[: args.top]:
        print(f"{n:6d}  {key}")
    print("\n-- 最重的文件前 20 --")
    for key, n in sorted(merged.files.items(), key=lambda kv: -kv[1])[:20]:
        print(f"{n:6d}  {key}")

    if args.json:
        pathlib.Path(args.json).write_text(
            json.dumps({"attrs": merged.attrs, "methods": merged.methods,
                        "files": merged.files}, indent=1), encoding="utf-8")
        print(f"[json] {args.json}")
    return 0


if __name__ == "__main__":                                      # pragma: no cover
    sys.exit(main())
