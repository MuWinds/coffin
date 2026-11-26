"""把 Occamy 的一层 MoE 单独拎出来：int8 分组路径 对 同一份权重的稠密 bf16 路径。

存在的理由：4B 是纯稠密，MoE 这套胶水（门/升融合栈的 N 轴顺序、SiLU 相乘、
down 的 K 轴、共享专家与门标量）从来没在**模型层**被验证过——此前只验证了
``moe_grouped_i8_tcu`` 内核本身对 torch 参考是对的（``tools/probe_corex_moe_i8.py``，
E=8/N=512/K=2048）。内核对不代表「栈的哪一半是 gate」也对。

做法：从预量化目录里只读第 ``--layer`` 层的张量（约 0.9 GB int8），建两个
``SparseMoE``——同一个 state dict，一份保持 int8 栈，一份把 int8 反量化成 bf16 稠密栈
（后者走 ``_experts_dense``，即仓库里已有的 oracle 入口）。同一输入跑两遍，报相对
RMS。**判据**：rel_rms 在 1e-2 量级是量化噪声；到 0.1 以上就是胶水错了（哪一半是
gate、down 的方向、或 top-k 权重归一化）。

输出 ``OCCMOE_JSON_BEGIN..END``。
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
        ``0``；``66`` 当目录不存在。
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args(argv)

    root = Path(args.model)
    if not root.is_dir():
        print(f"no such dir: {root}", flush=True)
        return 66

    # 只读一层：``keep`` 在装配前就把别的层滤掉，不然整份 34.6 GB 都要过一遍。
    layer = f"model.language_model.layers.{args.layer}."
    sd_int8 = quantized_load.load(root, device=_device, quant=True,
                                  keep=lambda k: k.startswith(layer))
    mlp = layer + "mlp"

    # 同一份权重的稠密版：反量化后形状与 checkpoint 一致（gate_up [E, 2I, H]、
    # down [E, H, I]），所以 ``SparseMoE`` 会选 ``_experts_dense``。一层约 1.8 GB，
    # 两份同时在卡上没问题；两层模型构造按权重字典算，所以各拿一份自己的字典。
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

    torch.manual_seed(args.seed)
    x = torch.randn(args.rows, int(cfg["hidden_size"]), dtype=torch.bfloat16,
                    device=_device) * 0.5

    with torch.no_grad():
        out_i8 = a(x).float()
        out_de = b(x).float()

    print("OCCMOE_JSON_BEGIN", flush=True)
    print(json.dumps({
        "layer": args.layer,
        "rows": args.rows,
        "experts": a.num_experts,
        "top_k": a.top_k,
        "inter": a.inter,
        "int8_path": type(a.gate_up).__name__,
        "dense_path": type(b.gate_up).__name__,
        "rel_rms": round(_rel(out_i8, out_de), 6),
        "i8_abs_mean": round(float(out_i8.abs().mean()), 6),
        "de_abs_mean": round(float(out_de.abs().mean()), 6),
    }, indent=2), flush=True)
    print("OCCMOE_JSON_END", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
