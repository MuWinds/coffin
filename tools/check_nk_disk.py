"""核对一份 ``--layout nk`` 预量化目录：形状、标尺方向、反量化误差。

Module: ``coffin/tools``。它只读两个目录，不写任何东西，所以既能当转换后的自检，
也能在拿到别人产出的目录时当验收。

**为什么必须核标尺方向**：``nk`` 是 ``int8 [N, K]``、每行一个标尺；若标尺被当成
列方向用（``[K]`` 形状或者按列广播），反量化出来的是一份尺寸完全正确、内容却整体
错位的权重，模型照样能跑完，只是每个 token 都在胡说。所以这里比的是
``int8[N, K] * scale[N, 1]`` 与源 bf16 的同名张量——两者一旦错位，相对误差会从
0.4% 量级跳到 10% 以上，一眼就能看出来。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

#: 索引文件名，与 ``quant_disk.py`` 同一个常量来源。
INDEX_NAME = "quant_index.json"


def _read_index(out: Path) -> dict:
    """读索引。

    Args:
        out: 预量化目录。

    Returns:
        索引字典。

    Raises:
        SystemExit: 目录里没有索引——那说明它不是本工具写出来的。
    """
    path = out / INDEX_NAME
    if not path.is_file():
        raise SystemExit(f"no {INDEX_NAME} under {out}")
    return json.loads(path.read_text(encoding="utf-8"))


def _find_shard(out: Path, name: str) -> Path:
    """按索引里的分片名定位文件。

    Args:
        out: 预量化目录。
        name: 分片文件名。

    Returns:
        分片路径。
    """
    return out / name


def _source_tensor(ckpt: Path, key: str) -> torch.Tensor:
    """在源 checkpoint 的任一分片里找这个键。

    Args:
        ckpt: 源 checkpoint 目录。
        key: 参数名。

    Returns:
        CPU 上的 bf16 张量。

    Raises:
        KeyError: 所有分片里都没有这个键。
    """
    for shard in sorted(ckpt.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            if key in handle.keys():
                return handle.get_tensor(key)
    raise KeyError(key)


def check(ckpt: Path, out: Path, sample: int = 3) -> dict:
    """核对整份目录，返回可打印的账目。

    Args:
        ckpt: 源 checkpoint。
        out: 预量化目录。
        sample: 逐张量比对取几个（按名字排序取前若干个量化张量）。

    Returns:
        含 ``format``、``layout``、``q8``、``dense``、``errors`` 与 ``checks`` 的字典。
    """
    index = _read_index(out)
    errors: list[str] = []
    if index.get("layout", "").find("[N, K]") < 0:
        errors.append(f"index layout is {index.get('layout')!r}, expected int8 [N, K]")
    if not index.get("format", "").endswith("nk-v1"):
        errors.append(f"index format is {index.get('format')!r}, expected *nk-v1")

    q8 = [k for k, v in index["tensors"].items() if v["kind"] == "q8"]
    dense = [k for k, v in index["tensors"].items() if v["kind"] == "dense"]
    checks = []
    for key in sorted(q8)[:sample]:
        meta = index["tensors"][key]
        shard = _find_shard(out, meta["shard"])
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            wt = handle.get_tensor(key)
            scale = handle.get_tensor(meta["scale_key"])
        n, k = int(meta["n"]), int(meta["k"])
        if tuple(wt.shape) != (n, k):
            errors.append(f"{key}: shard holds {tuple(wt.shape)}, index says {(n, k)}")
            continue
        if tuple(scale.shape) != (n,):
            errors.append(f"{key}: scale is {tuple(scale.shape)}, expected [{n}]")
            continue
        src = _source_tensor(ckpt, key).float()
        if tuple(src.shape) != (n, k):
            errors.append(f"{key}: source is {tuple(src.shape)}, expected {(n, k)}")
            continue
        got = wt.float() * scale.reshape(n, 1)
        err = (got - src).abs()
        rel = float(err.pow(2).mean().sqrt() / src.pow(2).mean().sqrt())
        checks.append({
            "key": key,
            "shape": [n, k],
            "max_abs": float(err.max()),
            "rel_rms": rel,
            "scale_min": float(scale.min()),
            "scale_max": float(scale.max()),
        })
        # 0.4% 是 int8 的实测量级；错位会把相对误差顶到 10% 以上。
        if rel > 0.02:
            errors.append(f"{key}: relative rms {rel:.4f} too large — scale likely misaligned")

    return {
        "source": str(ckpt),
        "out": str(out),
        "format": index.get("format"),
        "layout": index.get("layout"),
        "weight_dtype": index.get("weight_dtype"),
        "q8": len(q8),
        "q8_moe": sum(1 for v in index["tensors"].values() if v["kind"] == "q8-moe"),
        "dense": len(dense),
        "errors": errors,
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    """命令行入口。

    Args:
        argv: 参数列表；``None`` 取 ``sys.argv[1:]``。

    Returns:
        ``0`` 全部核对通过，``3`` 有问题。
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, help="源 checkpoint 目录")
    parser.add_argument("--out", required=True, help="预量化目录（nk 布局）")
    parser.add_argument("--sample", type=int, default=3, help="逐张量比对几个")
    args = parser.parse_args(argv)
    report = check(Path(args.checkpoint), Path(args.out), args.sample)
    print("NK_CHECK_JSON_BEGIN", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    print("NK_CHECK_JSON_END", flush=True)
    return 3 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
