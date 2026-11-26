"""对拍 corex 的 rms_norm：形状、逐行相对误差、以及行间是否互相串。

当前工作树里 ``kernels/norm.cu`` 正在改（图捕获曾在 ``rms_norm`` 处报
``shape '[32, 2560]' is invalid for input of size 2560``，说明内核有时只写了 1 行）。
这个探针用一个纯 torch 的式子做 oracle，分别试 M = 1 / 2 / 8 / 32，把三件事分开：

* 形状：输出是不是 ``[M, N]``；
* 数值：逐行 rel_rms；
* 串行：让每一行有一个独有的尖峰，看别的行有没有被污染（这是「只处理了第 0 行」
  或「行指针没动」的典型症状）。

    python tools/probe_corex_norm.py

输出 ``NORM_JSON_BEGIN..END`` 之间一行 JSON，任何一项不合格 rc=3。

Module: ``coffin/tools``。只 import ``backends``，不需要权重。
"""

from __future__ import annotations

import json
import sys

import torch

from backends import ops as bops


def _ref(x: torch.Tensor, g: torch.Tensor, eps: float) -> torch.Tensor:
    """零中心 RMSNorm 的参考写法（fp32 计算，回到 ``x`` 的 dtype）。"""
    xf = x.float()
    rms = xf.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
    return (xf * rms * g.float()).to(x.dtype)


def main() -> None:
    """按四档 M 对拍，打印一行 JSON。"""
    torch.manual_seed(0)
    n = 2560
    g = torch.rand(n, dtype=torch.bfloat16, device="cuda") + 0.5
    results = {}
    ok = True
    for m in (1, 2, 8, 32):
        x = torch.randn(m, n, dtype=torch.bfloat16, device="cuda") * 0.5
        # 每行一个独有尖峰：若内核只写了第 0 行，别的行会整体对不上。
        for r in range(m):
            x[r, (r * 7) % n] = 40.0 * (r + 1)
        want = _ref(x, g, 1e-6)
        got = bops.rms_norm(x, g, x.stride(0), 1e-6)
        entry = {"want": list(want.shape), "got": list(got.shape)}
        if tuple(got.shape) != tuple(want.shape):
            entry["error"] = "shape mismatch"
            ok = False
        else:
            diff = (got.float() - want.float())
            denom = want.float().pow(2).mean().sqrt() + 1e-9
            rel = float(diff.pow(2).mean().sqrt() / denom)
            entry["rel_rms"] = round(rel, 6)
            row_rel = [
                round(float((diff[i].pow(2).mean().sqrt()
                             / (want[i].float().pow(2).mean().sqrt() + 1e-9))), 6)
                for i in range(m)
            ]
            entry["row_rel"] = row_rel
            if rel > 2e-3:
                ok = False
        results[str(m)] = entry
        del x, want, got
    print("NORM_JSON_BEGIN", flush=True)
    print(json.dumps({"n": n, "results": results}, ensure_ascii=False), flush=True)
    print("NORM_JSON_END", flush=True)
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
