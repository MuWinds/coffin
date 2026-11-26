"""在几组几何上单独对拍 ``moe_grouped_i8_tcu``：到底哪一维一变它就错。

来历：单层 MoE 对拍（``tools/probe_occamy_moe_tcu.py``）上，``rows=8`` 时两条路都是
0.0056（= 量化噪声），而只要张量核真被选中（``rows=256/1024/2048``，也就是 assignments
2048/8192/16384）它恒在 **0.20**，同一份权重的 dp4a 路是 0.023。也就是说这不是
「行数太少、tile 尾部没零填充」那个已知缺陷——那个缺陷只在每专家 <4 行时出现，而
rows=2048 时每专家中位数 62 行。

原内核探针（``tools/probe_corex_moe_i8.py``）在 ``E=8 / N=512 / K=2048`` 上是 0.0087，
所以差异只可能来自**几何**：真实栈是 ``E=256``、门升栈 ``N=1024``（两张 512 列的
tile）。这里把 E / N 两轴分别拉开，K 固定 2048。

    python tools/probe_corex_moe_i8_geom.py

输出 ``MOE_GEOM_JSON_BEGIN..END``，一行一档（rel_rms 与逐专家行数固定为 8）。
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

from backends import ops as bops  # noqa: E402
from int8_weights import QWeight, QWeightMoE  # noqa: E402


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """相对 RMS。"""
    d = (a.float() - b.float()).pow(2).mean().sqrt()
    return float(d / (b.float().pow(2).mean().sqrt() + 1e-9))


def _oracle(w: QWeightMoE, x: torch.Tensor, bounds: list[int]) -> torch.Tensor:
    """逐专家反量化后 ``F.linear``，按 ``bounds`` 分段写回。"""
    out = torch.empty((x.shape[0], w.n), dtype=x.dtype, device=x.device)
    for e in range(w.expert_count):
        b0, b1 = bounds[e], bounds[e + 1]
        if b0 == b1:
            continue
        dense = w.expert(e).dequant(torch.bfloat16)
        out[b0:b1] = torch.nn.functional.linear(x[b0:b1], dense)
    return out


def _one(experts: int, n: int, k: int, per: int = 0,
         counts: list[int] | None = None, tag: str = "uniform") -> dict:
    """一档几何：造随机 int8 栈，按 ``counts``（缺省每专家 ``per`` 行）切段对拍。"""
    torch.manual_seed(7)
    dense = torch.randn((experts, n, k), dtype=torch.float32, device="cuda") * 0.02
    quantized = [QWeight.quantize(dense[e]) for e in range(experts)]
    wq = torch.stack([q.wq for q in quantized]).contiguous()
    scale = torch.stack([q.scale for q in quantized]).to(torch.float32).contiguous()
    stack = QWeightMoE(wq, scale, k=k)
    del dense
    torch.cuda.empty_cache()

    if counts is None:
        bounds = [i * per for i in range(experts + 1)]
    else:
        bounds = [0]
        for c in counts:
            bounds.append(bounds[-1] + c)
    offsets = torch.tensor(bounds, dtype=torch.int32, device="cuda")
    rows = bounds[-1]
    x = torch.randn((rows, k), dtype=torch.bfloat16, device="cuda") * 0.5
    want = _oracle(stack, x, bounds)
    entry: dict = {"tag": tag, "experts": experts, "n": n, "k": k, "rows": rows}
    for name in ("moe_grouped_i8_expert", "moe_grouped_i8_tcu"):
        fn = getattr(bops, name, None)
        if fn is None:
            entry[name] = "missing"
            continue
        try:
            got = fn(stack, x, offsets)
            entry[name] = round(_rel(got, want), 5)
        except ValueError as exc:
            entry[name] = f"ValueError: {str(exc)[:120]}"
        except Exception as exc:                                # noqa: BLE001
            entry[name] = f"{type(exc).__name__}: {str(exc)[:120]}"
    del x, want, stack, wq, scale, quantized
    torch.cuda.empty_cache()
    return entry


def main(argv: list[str] | None = None) -> int:
    """命令行入口。

    Args:
        argv: 参数列表；``None`` 取 ``sys.argv[1:]``。

    Returns:
        ``0``。
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--per", type=int, default=8, help="基线那几档每专家的固定行数")
    args = ap.parse_args(argv)

    base = args.per
    cases: list[dict] = [
        # 已对拍过的基线：每专家整 8 行、段长全对齐 → 干净（0.0088）。
        dict(tag="uniform8_E8N512", experts=8, n=512, k=2048, per=base),
        # 只把某一档段长改成非 4 的倍数，其余不动：测「段长不对齐」。
        dict(tag="one3_E8N512", experts=8, n=512, k=2048,
             counts=[3, 8, 8, 8, 8, 8, 8, 8]),
        # 每段都是 4 的倍数但长短不一：测「段长不齐但对齐」。
        dict(tag="mult4_E8N512", experts=8, n=512, k=2048,
             counts=[4, 12, 8, 16, 4, 20, 8, 12]),
        # 每段长短不一且不对齐：真实路由的形态。
        dict(tag="uneven_E8N512", experts=8, n=512, k=2048,
             counts=[3, 9, 5, 11, 4, 13, 7, 6]),
        # 真实门升栈几何，先跑对齐基线。
        dict(tag="uniform8_E256N1024", experts=256, n=1024, k=2048, per=base),
        # 真实几何 + 真实形态：每段 3..20 行、不对齐。
        dict(tag="uneven_E256N1024", experts=256, n=1024, k=2048,
             counts=[3 + (i * 7) % 18 for i in range(256)]),
    ]
    print("MOE_GEOM_JSON_BEGIN", flush=True)
    for case in cases:
        try:
            print(json.dumps(_one(**case), ensure_ascii=False), flush=True)
        except Exception as exc:                                # noqa: BLE001
            print(json.dumps({**{k: v for k, v in case.items()
                                 if k in ("tag", "experts", "n", "k")},
                              "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                              }, ensure_ascii=False), flush=True)
    print("MOE_GEOM_JSON_END", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
