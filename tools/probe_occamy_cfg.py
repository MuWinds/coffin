"""看一眼 Occamy 1.0 的 checkpoint 到底长什么样：配置键、专家数、层类型、参数名前缀。

存在的理由：``docs/COREX-PORT-MATRIX.md`` §9 一度把 Occamy 判给 ``models/qwen4exp``，
但它的 ``architectures`` 是 ``Qwen3_5MoeConditionalGeneration``，落在 ``engine_api`` 的
``qwen3_5`` 键上——也就是 4B 那条已经跑通的 ``models/qwen35``。差别只在
``text_config["num_experts"]`` 让 ``blocks.py`` 把 MLP 换成 MoE 层。所以先确认三件事：

1. ``text_config`` 带 ``layer_types``（``blocks.py`` 只认这个列表，不从下标推断）；
2. 专家是逐张量 ``experts.{e}.{proj}.weight``（``tools/quant_disk.py`` 靠这个正则融合）；
3. 参数名前缀与 ``quantized_load.DEFAULT_PREFIX`` 一致，否则预量化目录与加载器对不上。

输出 ``OCCAMY_CFG_JSON_BEGIN..END``，只读，不加载任何张量。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


#: ``blocks.py`` 决定 mixer 与 MLP 的依据，缺了它整模装载会直接失败。
_NEEDED = ("layer_types", "num_hidden_layers", "hidden_size", "num_experts",
           "num_experts_per_tok", "moe_intermediate_size",
           "shared_expert_intermediate_size", "vocab_size")

#: 逐专家参数名，与 ``tools/quant_disk.py`` 的 ``_EXPERT_KEY`` 同一形状。
_PROJ = ("gate_proj", "up_proj", "down_proj")


def main(argv: list[str] | None = None) -> int:
    """命令行入口：checkpoint 目录进，一段 JSON 出。

    Args:
        argv: 参数列表；``None`` 取 ``sys.argv[1:]``。

    Returns:
        ``0``，或 ``66`` 当目录不存在。
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args(argv)

    ckpt = Path(args.checkpoint)
    if not ckpt.is_dir():
        print(f"no such dir: {ckpt}", flush=True)
        return 66
    raw = json.loads((ckpt / "config.json").read_text())
    tc = raw.get("text_config", raw)

    out: dict = {
        "architectures": raw.get("architectures"),
        "model_type": raw.get("model_type"),
        "has_vision_config": "vision_config" in raw,
        "text_keys": sorted(tc),
        "missing": [k for k in _NEEDED if k not in tc],
        "text": {k: tc[k] for k in _NEEDED if k in tc},
    }
    lt = tc.get("layer_types")
    if lt:
        out["layer_type_counts"] = dict(Counter(lt))
        out["first_full_attn"] = next((i for i, t in enumerate(lt)
                                       if "full" in str(t).lower()), None)

    index = next(iter(ckpt.glob("*.index.json")), None)
    if index is not None:
        wmap = json.loads(index.read_text())["weight_map"]
        keys = sorted(wmap)
        out["shards"] = len(set(wmap.values()))
        out["tensors"] = len(keys)
        out["prefixes"] = dict(Counter(k.split(".")[0] for k in keys))
        out["expert_keys"] = sum(1 for k in keys if ".experts." in k)
        out["sample_keys"] = keys[:8]
        mlps = sorted({k.rsplit(".experts.", 1)[0] for k in keys if ".experts." in k})
        out["expert_mlp_prefixes"] = mlps[:3]
        out["expert_per_layer"] = (len([k for k in keys if ".experts." in k and mlps[0] in k])
                                   if mlps else 0)

    print("OCCAMY_CFG_JSON_BEGIN", flush=True)
    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)
    print("OCCAMY_CFG_JSON_END", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
