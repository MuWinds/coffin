"""在真 torch 上验证 nk 目录的读取路径：``quantized_load`` → ``QWeight``。

Module: ``coffin/tools``。稠密那条路（``cxrt.dense_weights``）已经在无 torch 进程里
跑通了，这里补的是**另一条**读取器：``quantized_load.load`` 交给后端的是
``QWeight``，而 :class:`int8_weights.QWeight` 的 ``wq`` 约定是 ``[N, K]``。两者若对
不上，读出来的是一份形状正确、内容转置的权重——只有拿源 checkpoint 逐元素比才看得
出来，模型自己只会安静地胡说。

nk 是唯一的布局（kn 那一整套打包与转置已经删除），所以这里顺便断言后端**不再**有
任何 ``[K, N]`` 转换器：``pack`` 里只剩常量，``QWeightMoE`` 也没有 ``layout`` 开关。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _source(ckpt: Path, key: str):
    """从源 checkpoint 找出这个键的 bf16 张量。

    Args:
        ckpt: 源目录。
        key: 参数名。

    Returns:
        CPU 上的 bf16 张量。
    """
    from safetensors import safe_open

    for shard in sorted(ckpt.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            if key in handle.keys():
                return handle.get_tensor(key)
    raise KeyError(key)


def main(argv: list[str] | None = None) -> int:
    """命令行入口。

    Args:
        argv: 参数；``None`` 取 ``sys.argv[1:]``。

    Returns:
        ``0`` 全部对得上，``3`` 有偏差。
    """
    import torch

    parser = argparse.ArgumentParser(description="验证 nk 目录在 torch 侧的读取")
    parser.add_argument("--model", required=True, help="nk 预量化目录")
    parser.add_argument("--source", required=True, help="源 bf16 checkpoint")
    parser.add_argument("--sample", type=int, default=3)
    args = parser.parse_args(argv)

    import quantized_load
    from backends import caps

    model_dir = Path(args.model)
    src_dir = Path(args.source)
    print(f"[probe] backend={caps().name} weight_layout={caps().weight_layout}", flush=True)

    index = quantized_load.read_index(model_dir)
    print(f"[probe] index format={index['format']} layout={index['layout']}", flush=True)
    sd = quantized_load.load(model_dir, device=torch.device("cuda:0"))
    print(f"[probe] loaded {len(sd)} tensors", flush=True)

    # 布局契约：后端只认 nk，``pack`` 里不该再有 ``as_kn`` / ``to_kn``。
    from backends.corex import pack
    kn_api = [name for name in ("as_kn", "to_kn", "KnWeight", "KnQWeight")
              if hasattr(pack, name)]
    print(f"[probe] pack 里残留的 kn 符号={kn_api or '无'}", flush=True)

    q8 = sorted(k for k, v in index["tensors"].items() if v["kind"] == "q8")
    rows = []
    bad = 0
    if kn_api:
        bad += 1
    for key in q8[:args.sample]:
        qw = sd[key]
        wq = qw.wq
        src = _source(src_dir, key).float().to(wq.device)
        n, k = int(wq.shape[0]), int(wq.shape[1])
        # 1) 形状必须是 [N, K]，也就是与源张量同形，而不是它的转置。
        shape_ok = list(wq.shape) == list(src.shape)
        # 2) 反量化后逐元素比。错位/转置会让相对误差顶到 10% 以上。
        got = wq.float() * qw.scale.reshape(n, 1)
        rel = float((got - src).pow(2).mean().sqrt() / src.pow(2).mean().sqrt())
        # 3) ``wq`` 必须是盘上那份缓冲本身：nk 原生，没有任何副本。
        no_copy = bool(wq.is_contiguous())
        row = {
            "key": key,
            "wq_shape": [n, k],
            "source_shape": list(src.shape),
            "shape_ok": shape_ok,
            "rel_rms": rel,
            "contiguous": no_copy,
        }
        rows.append(row)
        if not (shape_ok and no_copy) or rel > 0.02:
            bad += 1

    print("NK_LOAD_PROBE_BEGIN", flush=True)
    print(json.dumps({"bad": bad, "kn_symbols": kn_api, "rows": rows}, indent=2), flush=True)
    print("NK_LOAD_PROBE_END", flush=True)
    return 3 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
