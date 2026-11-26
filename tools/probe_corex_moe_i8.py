"""对拍分组 int8 MoE 投影：``moe_grouped_i8`` / ``moe_grouped_i8_tcu`` vs 逐专家 oracle。

Occamy 的专家投影现在只剩 int8 这一条路（4-bit 那条连同它的内核删掉了），而
``models/qwen4exp/moe.py`` 的 ``_make_grouped`` 按每专家行数在 GEMV 与张量核两档之间
选。这个探针把两档都对着「逐专家反量化 + F.linear」比一遍，用的形状取 Occamy 的真实
量级（``N=512``、``K=2048``、8 个本地专家、每专家 1 至 32 行）。

    python tools/probe_corex_moe_i8.py

输出 ``MOE_JSON_BEGIN..END`` 之间一行 JSON；任一档 rel_rms > 2e-2 记 rc=3。

Module: ``coffin/tools``。只 import ``backends`` 与 ``int8_weights``，不需要权重。
"""

from __future__ import annotations

import json
import sys

import torch

from backends import ops as bops
from int8_weights import QWeight, QWeightMoE


def _oracle(w: QWeightMoE, x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """逐专家反量化后做一次 ``F.linear``，按 ``offsets`` 分段写回。"""
    rows, k = x.shape
    out = torch.empty((rows, w.n), dtype=x.dtype, device=x.device)
    bounds = [int(v) for v in offsets.tolist()]
    for e in range(w.expert_count):
        begin, end = bounds[e], bounds[e + 1]
        if begin == end:
            continue
        dense = w.expert(e).dequant(torch.bfloat16)
        out[begin:end] = torch.nn.functional.linear(x[begin:end], dense)
    return out


def _rows_per_expert(counts: list[int]) -> torch.Tensor:
    """把每专家的行数变成 ``[E+1]`` 前缀和（内核要的就是这个）。"""
    acc = 0
    bounds = [0]
    for c in counts:
        acc += c
        bounds.append(acc)
    return torch.tensor(bounds, dtype=torch.int32, device="cuda")


def _transpose_check(k: int) -> dict:
    """对拍 ``transpose_i8``：``[rows, K] -> [K, rows]``，逐档试 8 / 16 / 32 行。

    张量核那条 MoE 路径靠它把激活摆成 k 主序。行数很小时（8 行）如果它在填充区里
    写了非零，GEMM 的末尾几列就会拿到垃圾——那正是「8 行时 rel 0.88、32 行时 0.8%」
    最可能的来源，所以这一项要和上面对拍分开看。
    """
    from backends.corex import _moe_ops

    shim = _moe_ops().shim
    out = {}
    for rows in (8, 16, 32):
        src = torch.randint(-127, 128, (rows, k), dtype=torch.int8, device="cuda")
        dst = torch.empty((k, rows), dtype=torch.int8, device="cuda")
        rc = shim.transpose_i8(src.data_ptr(), dst.data_ptr(), rows, k, 0)
        if rc:
            raise RuntimeError(f"coffin_transpose_i8 returned {rc}")
        want = src.t().contiguous()
        exact = bool(torch.equal(dst, want))
        out[str(rows)] = {"exact": exact,
                          "mismatch": int((dst != want).sum().item())}
    return out


def main() -> None:
    """三档行数对拍两个入口，打印一行 JSON。"""
    torch.manual_seed(2)
    experts, n, k = 8, 512, 2048
    ok = True
    result = {}
    transpose = _transpose_check(k)
    if not all(v["exact"] for v in transpose.values()):
        ok = False

    dense = (torch.randn((experts, n, k), dtype=torch.float32, device="cuda") * 0.02)
    quantized = [QWeight.quantize(dense[e]) for e in range(experts)]
    wq = torch.stack([q.wq for q in quantized]).contiguous()
    scale = torch.stack([q.scale for q in quantized]).to(torch.float32).contiguous()
    stack = QWeightMoE(wq, scale, k=k)
    nk_copy = stack.nk_stack()
    same_buffer = nk_copy.data_ptr() == wq.data_ptr()

    for per in (1, 2, 3, 4, 32):
        counts = [per] * experts
        offsets = _rows_per_expert(counts)
        rows = per * experts
        x = torch.randn((rows, k), dtype=torch.bfloat16, device="cuda") * 0.5
        want = _oracle(stack, x, offsets)
        entry = {}
        for name in ("moe_grouped_i8", "moe_grouped_i8_tcu"):
            fn = getattr(bops, name, None)
            if fn is None:
                entry[name] = "missing"
                ok = False
                continue
            try:
                got = fn(stack, x, offsets)
                rel = float((got.float() - want.float()).pow(2).mean().sqrt()
                            / (want.float().pow(2).mean().sqrt() + 1e-9))
                entry[name] = round(rel, 5)
                if rel > 2e-2:
                    ok = False
            except ValueError as exc:
                # 短段被包装层挡住是**期望行为**（每专家不足 4 行时张量核那条不对，
                # 调用方按 ValueError 退回 GEMV）。只有没挡住才算失败。
                if "rows per expert" in str(exc):
                    entry[name] = "guarded"
                else:
                    entry[name] = f"ValueError: {exc}"[:200]
                    ok = False
            except Exception as exc:                            # noqa: BLE001
                entry[name] = f"{type(exc).__name__}: {exc}"[:200]
                ok = False
        entry["rows"] = rows
        result[str(per)] = entry
        del x, want

    print("MOE_JSON_BEGIN", flush=True)
    print(json.dumps({"experts": experts, "n": n, "k": k,
                      "nk_zero_copy": same_buffer, "transpose": transpose,
                      "results": result}, ensure_ascii=False), flush=True)
    print("MOE_JSON_END", flush=True)
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
