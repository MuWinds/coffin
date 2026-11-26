"""预填充 GEMM 的上卡探针：只 import ``cxrt``，量出 TOPS 并查「算错没有」。

用法（卡上）::

    python tools/probe_gemm_nk.py --dtype bf16 --m 2048 --k 2560 --n 18432
    python tools/probe_gemm_nk.py --dtype i8 --m 2048 --k 9216 --n 2560 --reps 20
    python tools/probe_gemm_nk.py --dtype bf16 --ones --k 4096     # 大 K 截断
    python tools/probe_gemm_nk.py --dtype i8 --computes both       # 认 computeType 世代

它查四件事，任何一件不过都不该继续往下走：

1. **链路上没有 torch**：跑完断言 ``"torch" not in sys.modules``。这比读代码可靠，
   因为 ``import torch`` 可能躲在被间接 import 的模块里。
2. **布局**：``[N, K]`` 原缓冲直接喂 ``i8_nk`` / ``bf16_nk``，与 numpy 的
   ``x @ w.T`` 对拍。不等就说明 ``lda/ldb/ldc`` 或 m/n 那次交换错了。
3. **大 K 截断**：``--ones`` 时全 1 权重 + 全 1 激活，输出必须恰好等于 ``K``。
   ixblas 记的 F-5（fp16 出、K ≥ 2049 只算前 2048）会在这条上露出来，而且是静默错值。
4. **TOPS**：``2*M*K*N / t``。int8 目标是 41.8~46.2（ixblas TN、``K % 128 == 0``），
   bf16 是 32.6~39.7。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

import cxrt as rt
from cxrt import dtypes as D


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行。

    Args:
        argv: 参数表；``None`` 时读 ``sys.argv``。

    Returns:
        ``Namespace``：dtype / m / k / n / reps / computes / ones。
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dtype", choices=("i8", "bf16", "f32"), default="bf16")
    p.add_argument("--m", type=int, default=2048, help="prefill rows")
    p.add_argument("--k", type=int, default=2560, help="inner dim; TN 要求 %%128 == 0")
    p.add_argument("--n", type=int, default=18432, help="weight columns")
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--computes", choices=("new", "legacy", "both"), default="new")
    p.add_argument("--ones", action="store_true", help="全 1 输入，查大 K 截断")
    return p.parse_args(argv)


def make_case(kind: str, m: int, k: int, n: int,
              ones: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """造一组形态与真实权重一致的输入。

    权重存成 ``[N, K]``（nk，与盘上同形），激活存成 ``[M, K]``。

    Args:
        kind: ``"i8"`` / ``"bf16"`` / ``"f32"``。
        m: 行数。
        k: 内维。
        n: 列数。
        ones: ``True`` 时全 1，用于把输出与 ``K`` 直接对上。

    Returns:
        ``(x, w, ref)``。``x``/``w`` 按 dtype 编码（bf16 是 uint16 位模式），
        ``ref`` 是 fp32 的参考结果。
    """
    if ones:
        if kind == "i8":
            x = np.ones((m, k), dtype=np.int8)
            w = np.ones((n, k), dtype=np.int8)
        else:
            x = np.ones((m, k), dtype=np.float32)
            w = np.ones((n, k), dtype=np.float32)
        ref = np.full((m, n), float(k), dtype=np.float32)
    else:
        rng = np.random.default_rng(7)
        if kind == "i8":
            x = rng.integers(-8, 8, size=(m, k), dtype=np.int8)
            w = rng.integers(-8, 8, size=(n, k), dtype=np.int8)
            ref = x.astype(np.float32) @ w.astype(np.float32).T
        else:
            x = (rng.standard_normal((m, k)) * 0.5).astype(np.float32)
            w = (rng.standard_normal((n, k)) * 0.5).astype(np.float32)
            ref = x @ w.T
    if kind == "bf16":
        return D.bf16_bits(x), D.bf16_bits(w), ref
    return x, w, ref


def device_dtype(kind: str) -> D.DType:
    """dtype 名到 :class:`cxrt.dtypes.DType`。

    Args:
        kind: ``"i8"`` / ``"bf16"`` / ``"f32"``。

    Returns:
        对应的 DType。
    """
    return {"i8": rt.I8, "bf16": rt.BF16, "f32": rt.F32}[kind]


def run_once(runtime: rt.Runtime, ops: rt.GemmOps, kind: str, x: np.ndarray, w: np.ndarray,
             m: int, k: int, n: int, reps: int) -> tuple[np.ndarray, float]:
    """跑 ``reps`` 遍，返回结果与单次时延（ms）。

    Args:
        runtime: 已打开的运行时。
        ops: 绑好的 GEMM 面。
        kind: dtype 名。
        x: ``[M, K]`` 宿主数组（已按 dtype 编码）。
        w: ``[N, K]`` 宿主数组。
        m: 行数。
        k: 内维。
        n: 列数。
        reps: 重复次数。

    Returns:
        ``(out, 单次时延 ms)``；``out`` 是 fp32 视图，bf16 时直接从位模式还原。
    """
    dt = device_dtype(kind)
    ax = rt.Array.from_numpy(runtime.cudart, runtime.alloc, x, dt)
    aw = rt.Array.from_numpy(runtime.cudart, runtime.alloc, w, dt)
    ao = rt.Array.empty(runtime.cudart, runtime.alloc, (m, n),
                        rt.I32 if kind == "i8" else rt.F32 if kind == "f32" else rt.BF16)
    stream = runtime.stream.handle

    # int8 只有一条路：nk 原缓冲（``[N, K]``）直接进 ``OP_T``（TN），一次转置都不做。
    def call() -> None:
        if kind == "i8":
            ops.i8_nk(ax, aw, ao, stream=stream)
        elif kind == "bf16":
            ops.bf16_nk(ax, aw, ao, stream=stream, out_code=rt.BF16.code)
        else:
            ops.f32_nk(ax, aw, ao, stream=stream)

    call()
    runtime.stream.sync()
    start = time.perf_counter()
    for _ in range(reps):
        call()
    runtime.stream.sync()
    ms = (time.perf_counter() - start) / reps * 1e3
    host = ao.to_numpy()
    if kind == "bf16":
        out = D.bf16_to_f32(host)
    else:
        out = host.astype(np.float32)
    for a in (ax, aw, ao):
        a.close()
    return out, ms


def main(argv: list[str] | None = None) -> int:
    """跑完四件事并打印结论。

    Args:
        argv: 参数表。

    Returns:
        退出码：``0`` 全过，``1`` 有检查没过。
    """
    args = parse_args(argv)
    flops = 2.0 * args.m * args.k * args.n
    runtime = rt.open_runtime()
    worst = 0.0
    scale = 1.0
    generations = ("new", "legacy") if args.computes == "both" else (args.computes,)

    for generation in generations:
        os.environ["COFFIN_CUBLAS_COMPUTE"] = generation
        ops = runtime.build_gemm()
        x, w, ref = make_case(args.dtype, args.m, args.k, args.n, args.ones)
        if True:
            out, ms = run_once(runtime, ops, args.dtype, x, w, args.m, args.k, args.n,
                               args.reps)
            err = float(np.abs(out - ref).max())
            worst = max(worst, err)
            scale = max(scale, float(np.abs(ref).max()))
            tag = ""
            print(f"[{generation}{'/' + tag if tag else ''}] {args.dtype} "
                  f"{args.m}x{args.k}x{args.n}  "
                  f"{flops / (ms * 1e9):.2f} TFLOP/s  {ms:.3f} ms  max_abs_err={err:.4g}")
    runtime.close()
    ok = worst <= 2e-2 * scale
    loaded = "torch" in sys.modules
    print(f"numeric={'OK' if ok else 'FAIL'} scale={scale:.4g} worst={worst:.4g} "
          f"torch_loaded={loaded}")
    return 0 if ok and not loaded else 1


if __name__ == "__main__":
    raise SystemExit(main())

