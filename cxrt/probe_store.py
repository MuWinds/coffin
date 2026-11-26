"""卡上探针：用 ``cxrt.store`` 读**真实** checkpoint，确认 loader 不是只认得自造的归档。

Module: ``coffin/cxrt``。主机上没有 torch，也没有 checkpoint，所以「loader 能不能解
真的 ``torch.save`` 产物」只能在卡上验。这个模块不做推理，只回答四个问题：

1. 目标路径下有哪些文件、大小多少；
2. 每个文件解出来的顶层类型、张量数量；
3. 各张量的 dtype 分布与前几个键的形状；
4. 有没有元数据键（``__version__`` 之类）。

用法::

    python3 -m cxrt.probe_store                       # 读 $HOME/models 下的默认 checkpoint
    python3 -m cxrt.probe_store /path/to/ckpt         # 指定文件或目录
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from cxrt.tt import Tensor

#: 期望的权重后缀；目录模式下只挑这些。
_SUFFIXES = {".pt", ".pth", ".bin", ".safetensors", ".npz"}


def _describe(path: Path, limit: int) -> None:
    """读一个文件并打印摘要。

    Args:
        path: checkpoint 文件。
        limit: 打印多少个键的形状。
    """
    from cxrt import store

    size_mb = path.stat().st_size / 2**20
    print(f"\n[file] {path} ({size_mb:.1f} MiB)")
    try:
        obj = store.load(path)
    except Exception as exc:                                # noqa: BLE001 - 探针要报全
        print(f"[load] FAIL {type(exc).__name__}: {exc}")
        return
    print(f"[load] top-level type={type(obj).__name__}")
    if not isinstance(obj, dict):
        print(f"[load] value={type(obj).__name__}")
        return
    tensors = {k: v for k, v in obj.items() if isinstance(v, Tensor)}
    meta = [k for k in obj if not isinstance(obj, Tensor)]
    print(f"[load] tensors={len(tensors)} meta_keys={meta[:6]}")
    dtypes: dict[str, int] = {}
    for value in tensors.values():
        dtypes[value.dtype.name] = dtypes.get(value.dtype.name, 0) + 1
    print(f"[load] dtypes={dtypes}")
    from cxrt.tt import bfloat16, int8

    for key in sorted(tensors)[:limit]:
        value = tensors[key]
        mark = ""
        if value.dtype is int8:
            mark = " <int8>"
        elif value.dtype is bfloat16:
            mark = " <bf16>"
        print(f"[key ] {key} {tuple(value.shape)} {value.dtype.name}{mark}")


def main(argv: list[str] | None = None) -> int:
    """入口，返回退出码（有文件读失败则非零）。"""
    home = os.environ.get("HOME") or os.environ.get("USERPROFILE") or "."
    parser = argparse.ArgumentParser(description="用 cxrt.store 读真实 checkpoint")
    parser.add_argument("path", nargs="?", default="")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--scan-defaults", action="store_true",
                        help="在 ~/models 下找候选")
    args = parser.parse_args(argv)

    target = Path(args.path) if args.path else Path(home) / "models"
    if args.path and not target.exists():
        print(f"[miss] {target} 不存在")
        return 2
    if not target.exists():
        print(f"[miss] {target} 不存在；用 --path 指定")
        return 2

    failures = 0
    if target.is_dir():
        files = sorted(p for p in target.rglob("*") if p.suffix in _SUFFIXES)
        if not files:
            print(f"[miss] {target} 下没有 {'/'.join(sorted(_SUFFIXES))}")
            return 2
        for path in files[: 6]:
            _describe(path, args.limit)
        return 0
    _describe(target, args.limit)
    return failures


if __name__ == "__main__":                                  # pragma: no cover - 入口
    sys.exit(main())
