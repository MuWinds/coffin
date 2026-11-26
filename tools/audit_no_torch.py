"""验收「推理链路里没有真 torch」：静态清点 + 运行时身份断言。

Module: ``coffin/tools``。只 import 标准库，自己**不 import torch**——它是那条线本身
的守卫，不该依赖被守卫的东西。

两件事分开做：

* ``--static``（默认含）：把引擎里所有 ``import torch`` / ``torch.*`` 的模块清点出来，
  按「corex 推理链路 / 其它」分组打印。这一步只报告，因为「代码里怎么拼」和「进程里
  有没有真 torch」不是一回事。
* ``--runtime``：在一个**干净的子进程**里先 ``cxrt.install.install()``，再挨个 import
  推理链路上的模块，然后断言三件事：``sys.modules['torch']`` 就是 :mod:`cxrt.torchshim`、
  它的 ``__file__`` 在 ``cxrt/`` 下、``torch.Tensor`` 是 :class:`cxrt.tt.Tensor`。
  任一不成立就退出码 1——这是硬门。

用法::

    python tools/audit_no_torch.py                  # 先清点，再跑运行时断言
    python tools/audit_no_torch.py --static         # 只清点
    python tools/audit_no_torch.py --runtime --imports models.qwen35,engine_api,backends.corex

退出码 ``0`` 表示运行时断言全过（静态清点不影响退出码）。
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: 推理链路（corex 那张卡上真正会跑的）模块/包。清点时它们归一组，其余归另一组。
_CHAIN = (
    "run_chat.py", "engine_api.py", "quantized_load.py", "int8_weights.py",
    "cxrt", "ops", "models/qwen35", "backends/corex", "backends/protocol.py",
    "serving", "engine",
)

#: 运行时断言里默认要 import 的模块：模型的表层、量化加载器、corex 后端本身。
_DEFAULT_IMPORTS = ("models.qwen35", "quantized_load", "engine_api", "backends.corex",
                    "ops", "cxrt.tt")

#: 子进程脚本：先装壳，再 import，再断言。故意写成字符串——它必须在没有任何别的
#: ``import torch`` 发生过的新进程里执行。
_CHILD = """
import os
import sys

# 这个子进程验的是「壳 + 链路 module 能 import」，不是「这台主机有没有卡」：把
# ``COFFIN_BACKEND`` 之类的强制选择清掉，否则 ``models.qwen35`` 会因为
# 「corex 这张卡不可用」而拒载，看起来像 torch 的问题，其实不是。
for _name in ("COFFIN_BACKEND", "COFFIN_PREFILL_NK", "COFFIN_LAYER_DRIVER"):
    os.environ.pop(_name, None)

import cxrt.install as inst
inst.install()
import torch
import importlib

bad = []
for name in {mods!r}:
    try:
        importlib.import_module(name)
    except Exception as exc:
        bad.append((name, type(exc).__name__ + ": " + str(exc).replace("\\n", " ")[:200]))

shim = sys.modules["torch"]
print("verify:", inst.verify())
print("file:", getattr(shim, "__file__", "?"))
print("tensor:", getattr(shim, "Tensor", "?"))
print("tensor_mod:", getattr(getattr(shim, "Tensor", None), "__module__", "?"))
print("version:", getattr(shim, "__version__", "?"))
for name, err in bad:
    print("IMPORT-FAIL", name, err)
print("RESULT", "OK" if not bad else "FAIL")
sys.exit(1 if bad else 0)
"""


def _chain_modules() -> list[Path]:
    """收集引擎里的 .py 文件（跳过测试、工具、文档构件）。

    Returns:
        文件列表。
    """
    out = []
    for p in ROOT.rglob("*.py"):
        parts = p.relative_to(ROOT).parts
        if any(x in parts for x in (".git", "__pycache__", "tests", "tools")):
            continue
        out.append(p)
    return sorted(out)


def _count(imports_only: bool) -> tuple[int, int, list[tuple[str, int, int]]]:
    """清点 ``import torch`` 与 ``torch.`` 引用。

    Args:
        imports_only: 只数 import 行（``torch.`` 引用不计）。

    Returns:
        ``(import 行数总计, torch. 引用总计, [(相对路径, import 行, 引用数)])``。
    """
    import re

    total_import = 0
    total_ref = 0
    rows: list[tuple[str, int, int]] = []
    for p in _chain_modules():
        text = p.read_text(encoding="utf-8")
        n_import = len(re.findall(r"^\s*(import torch|from torch(\.\w+)* import)", text, re.M))
        n_ref = 0 if imports_only else len(re.findall(r"\btorch\.", text))
        if n_import:
            rel = str(p.relative_to(ROOT)).replace("\\", "/")
            rows.append((rel, n_import, n_ref))
            total_import += n_import
            total_ref += n_ref
    rows.sort(key=lambda r: -r[1])
    return total_import, total_ref, rows


def _in_chain(rel: str) -> bool:
    """判断一个相对路径是否属于 corex 推理链路。

    Args:
        rel: 相对仓库根的 posix 路径。

    Returns:
        属于返回 ``True``。
    """
    for entry in _CHAIN:
        if entry.endswith(".py"):
            if rel == entry:
                return True
        elif rel == entry or rel.startswith(entry + "/"):
            return True
    return False


def _static() -> int:
    """打印清点结果。

    Returns:
        0。
    """
    total_import, total_ref, rows = _count(imports_only=False)
    chain = [r for r in rows if _in_chain(r[0])]
    other = [r for r in rows if not _in_chain(r[0])]
    print(f"模块里出现 `import torch`: {len(rows)} 个，import 行 {total_import} 条，"
          f"`torch.` 引用 {total_ref} 处")
    print(f"  其中 corex 推理链路 {len(chain)} 个 / 其它 {len(other)} 个\n")
    print("  [corex 链路]")
    for rel, n_import, n_ref in chain:
        print(f"    {rel:<44} import={n_import}  refs={n_ref}")
    if other:
        print("\n  [不在链路里]")
        for rel, n_import, n_ref in other:
            print(f"    {rel:<44} import={n_import}  refs={n_ref}")
    print()
    return 0


def _runtime(imports: tuple[str, ...]) -> int:
    """在干净子进程里断言 torch 是壳。

    Args:
        imports: 要在子进程里 import 的模块名。

    Returns:
        0 表示断言全过。
    """
    code = _CHILD.format(mods=list(imports))
    proc = subprocess.run([sys.executable, "-B", "-c", code], cwd=str(ROOT),
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout.rstrip())
    fields = {}
    for line in proc.stdout.splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip() in {"verify", "file", "tensor", "tensor_mod", "version"}:
            fields[key.strip()] = value.strip()
    import_fail = [ln for ln in proc.stdout.splitlines() if ln.startswith("IMPORT-FAIL")]
    ok = (proc.returncode == 0
          and fields.get("verify") == "True"
          and fields.get("file", "").replace("\\", "/").endswith("cxrt/torchshim.py")
          and fields.get("tensor_mod") == "cxrt.tt"
          and not import_fail)
    if not ok:
        if import_fail:
            print("FAIL: 有链路模块在壳下 import 不起来——它们仍然需要真 torch")
        else:
            print("FAIL: 运行时断言不成立（真 torch 在场，或者壳没装上）")
        return 1
    print("OK: 进程里没有真 torch —— sys.modules['torch'] 是 cxrt.torchshim")
    return 0


def main(argv: list[str] | None = None) -> int:
    """命令行入口。

    Args:
        argv: 参数列表；``None`` 读 ``sys.argv``。

    Returns:
        进程退出码。
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--static", action="store_true", help="只做清点")
    ap.add_argument("--runtime", action="store_true", help="只做运行时断言")
    ap.add_argument("--imports", default=",".join(_DEFAULT_IMPORTS),
                    help="运行时断言里要 import 的模块名，逗号分隔")
    args = ap.parse_args(argv)

    do_static = args.static or not (args.static or args.runtime)
    do_runtime = args.runtime or not (args.static or args.runtime)

    rc = 0
    if do_static:
        rc |= _static()
    if do_runtime:
        rc |= _runtime(tuple(m for m in args.imports.split(",") if m))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
