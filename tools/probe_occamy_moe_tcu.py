"""在同一层 MoE 上把「行数」这一轴扫一遍：张量核分组 对 dp4a 分组 对 稠密 bf16。

存在的理由：``tools/probe_occamy_moe.py`` 在 rows=8 上量到 rel_rms=0.0056（= 量化噪声），
但真实预填充 321 token（rows=2568）出来的 chat 带着 ``�`` 这种非法字符，而单层对拍在
rows=2048 上是 **0.2037**。同一份权重、同一套胶水，差 40 倍只能说明「张量核那条路在
某个行数段是错的」，所以要按行数二分：到底从哪一档开始坏、坏的是张量核还是 dp4a。

做法：一层权重只加载一次（int8 + 反量化的 bf16 各一份），两层 ``SparseMoE`` 常驻，
然后对每个 ``rows`` 跑三遍：
  * ``tcu``  —— 原样（``_grouped_i8_tcu`` 在，会按 ``tcu_ok`` 自己选）；
  * ``dp4a`` —— 把 ``_grouped_i8_tcu`` 临时置 None，强制走 ``moe_grouped_i8_expert``；
  * 同时打每个专家实际分到的行数（min / 中位 / max），因为守卫现在是按
    ``rows >= 4 * experts`` 的**平均**预算的，挡不住分布偏斜。

判据：rel_rms 在 1e-2 量级是量化噪声；0.1 以上是数值错。

输出 ``OCCMOE_TCU_JSON_BEGIN..END``，一行一档。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ENGINE = Path(__file__).resolve().parents[1]
if str(_ENGINE) not in sys.path:
    sys.path.insert(0, str(_ENGINE))

import torch  # noqa: E402

import quantized_load  # noqa: E402
from backends import device as _device  # noqa: E402
from int8_weights import QWeight, QWeightMoE  # noqa: E402


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """相对 RMS，两端先拉成 fp32。"""
    d = (a.float() - b.float()).pow(2).mean().sqrt()
    return float(d / (b.float().pow(2).mean().sqrt() + 1e-9))


def main(argv: list[str] | None = None) -> int:
    """命令行入口。

    Args:
        argv: 参数列表；``None`` 取 ``sys.argv[1:]``。

    Returns:
        ``0``；``66`` 当模型目录不存在。
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--rows", default="8,256,512,1024,2048")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args(argv)

    root = Path(args.model)
    if not root.is_dir():
        print(f"no such dir: {root}", flush=True)
        return 66
    rows_list = [int(x) for x in str(args.rows).split(",") if x.strip()]

    layer = f"model.language_model.layers.{args.layer}."
    sd_int8 = quantized_load.load(root, device=_device, quant=True,
                                  keep=lambda k: k.startswith(layer))
    mlp = layer + "mlp"
    sd_dense = {}
    for k, v in sd_int8.items():
        if isinstance(v, QWeightMoE):
            sd_dense[k] = v.to(torch.bfloat16)
        elif isinstance(v, QWeight):
            sd_dense[k] = v.dequant(torch.bfloat16)
        else:
            sd_dense[k] = v

    from models.qwen35.moe import SparseMoE  # noqa: E402  (延后：它绑算子表)

    cfg = json.loads((root / "config.json").read_text())["text_config"]
    a = SparseMoE(sd_int8, mlp, cfg, bench=False)
    b = SparseMoE(sd_dense, mlp, cfg, bench=False)
    tcu = a._grouped_i8_tcu
    print(f"OCCMOE_TCU_META experts={a.num_experts} top_k={a.top_k} inter={a.inter} "
          f"tcu={'yes' if tcu is not None else 'no'} "
          f"expert={'yes' if a._grouped_i8_expert is not None else 'no'}", flush=True)
    print("OCCMOE_TCU_JSON_BEGIN", flush=True)

    torch.manual_seed(args.seed)
    hidden = int(cfg["hidden_size"])
    for rows in rows_list:
        x = torch.randn(rows, hidden, dtype=torch.bfloat16, device=_device) * 0.5
        # 每个专家分到多少行：守卫按 ``rows >= 4 * experts`` 的**平均**算，分布偏斜时
        # 平均够、尾部专家仍然只有几行，这正是怀疑点。
        with torch.no_grad():
            _, index = a._route(x, "prefill")
            counts = torch.bincount(index.reshape(-1).long(), minlength=a.num_experts)
            order = torch.argsort(counts)
            dist = counts[order].tolist()
            ref = b(x).float()

            a._grouped_i8_tcu = tcu
            out_tcu = a(x).float()
            r_tcu = _rel(out_tcu, ref)

            a._grouped_i8_tcu = None
            out_dp = a(x).float()
            r_dp = _rel(out_dp, ref)
            a._grouped_i8_tcu = tcu

        print(json.dumps({
            "rows": rows,
            "assignments": int(index.numel()),
            "per_expert_min": dist[0],
            "per_expert_p25": dist[len(dist) // 4],
            "per_expert_med": dist[len(dist) // 2],
            "per_expert_max": dist[-1],
            "experts_below_4": int(sum(1 for c in dist if c < 4)),
            "rel_rms_tcu": round(r_tcu, 6),
            "rel_rms_dp4a": round(r_dp, 6),
        }), flush=True)
        del x
        torch.cuda.empty_cache()

    print("OCCMOE_TCU_JSON_END", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
