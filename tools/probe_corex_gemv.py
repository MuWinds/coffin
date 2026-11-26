"""对拍 corex 的 bf16 投影（kn GEMV 那一家）：``linear`` / ``linear_m`` / ``linear_prefill``。

现在 ``kernels/gemv.cu`` 正处于重写中，而 int8 nk 那条路（``gemv_nk_i8``）是另一支
内核。这个探针把两者分开：同一个随机权重，先问 int8 nk 对不对，再问 bf16 kn 对不对，
于是「模型输出不对」能落到具体是哪一支内核。

    python tools/probe_corex_gemv.py --rows 512 --k 2560

输出 ``GEMV_JSON_BEGIN..END`` 之间一行 JSON，任何一支 rel_rms > 2e-2 记 rc=3。

Module: ``coffin/tools``。只 import ``backends`` 与 ``int8_weights``，不需要权重。
"""

from __future__ import annotations

import argparse
import json
import sys

import torch

from backends import ops as bops
from int8_weights import QWeight


def _rel(got: torch.Tensor, want: torch.Tensor) -> float:
    """相对 RMS 误差（以 oracle 的 RMS 为尺度，避免「大值掩盖小值」）。"""
    diff = got.float() - want.float()
    return float(diff.pow(2).mean().sqrt() / (want.float().pow(2).mean().sqrt() + 1e-9))


def main() -> None:
    """按两种权重形态、三种入口对拍，打印一行 JSON。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=512)
    ap.add_argument("--k", type=int, default=2560)
    ap.add_argument("--tol", type=float, default=2e-2)
    args = ap.parse_args()

    torch.manual_seed(1)
    n, k = args.rows, args.k
    ok = True
    out = {}

    dense = (torch.randn((n, k), dtype=torch.float32, device="cuda") * 0.02).to(
        torch.bfloat16)
    int8 = QWeight.quantize(dense)
    for m in (1, 2, 8, 32, 512):
        x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
        want_i8 = torch.nn.functional.linear(
            x, int8.dequant(torch.bfloat16))
        want_bf16 = torch.nn.functional.linear(x, dense)
        entry = {}
        for name, call in (
            ("int8_linear", lambda: bops.linear(int8, x[0]) if m == 1 else bops.linear_m(int8, x)),
            ("bf16_linear", lambda: bops.linear(dense, x[0]) if m == 1 else bops.linear_m(dense, x)),
            ("bf16_prefill", lambda: bops.linear_prefill(dense, x)),
        ):
            try:
                got = call()
                if m == 1:
                    got = got.reshape(1, n)
                oracle = want_i8 if name.startswith("int8") else want_bf16
                rel = _rel(got, oracle)
                entry[name] = round(rel, 5)
                if rel > args.tol:
                    ok = False
            except Exception as exc:                            # noqa: BLE001
                entry[name] = f"{type(exc).__name__}: {exc}"[:200]
                ok = False
        out[str(m)] = entry
        del x, want_i8, want_bf16

    print("GEMV_JSON_BEGIN", flush=True)
    print(json.dumps({"n": n, "k": k, "results": out}, ensure_ascii=False), flush=True)
    print("GEMV_JSON_END", flush=True)
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
