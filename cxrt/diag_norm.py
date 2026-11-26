"""``python3 -m cxrt.diag_norm`` —— 归一化五入口的逐位诊断。

Module: ``coffin/cxrt``。自检只报「差几个 ULP」，这个模块在差得离谱时把两边的位模式
打印出来，用来分辨四类原因：H2D 拷贝没到位、内核没跑、地址被池复用、以及真的算错。

用法::

    python3 -m cxrt.diag_norm
    python3 -m cxrt.diag_norm --rows 2 --cols 64 --dtype bf16

约束：只 import ``cxrt`` 与 numpy，不 import torch。
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from cxrt.array import Array
from cxrt.dtypes import BF16, BY_NAME, F32, bf16_bits, bf16_to_f32
from cxrt.errors import BuildError, RtError
from cxrt.mock import host_reference


def _fill(rt, alloc, values: np.ndarray, dtype):
    """同 ``selftest._fill``：fp32 宿主数据按 dtype 上卡。"""
    if dtype is BF16:
        return Array.from_numpy(rt, alloc, bf16_bits(values), BF16)
    return Array.from_numpy(rt, alloc, values, F32)


def _raw(array: Array) -> np.ndarray:
    """设备数组的原始位模式（bf16 是 uint16）。"""
    return array.to_numpy()


def _bits(view: np.ndarray, dtype) -> str:
    """把一行位模式排成十六进制串，便于逐位对比。"""
    if dtype is BF16:
        return " ".join(f"{int(v):04x}" for v in view)
    return " ".join(f"{v:.6g}" for v in view)


def _row_compare(label: str, got: np.ndarray, want: np.ndarray, dtype, width: int = 8) -> None:
    """打印前 ``width`` 个元素的两侧值，并给出最大差。"""
    print(f"  {label}: got ={_bits(got[:width], dtype)}")
    print(f"  {label}: want={_bits(want[:width], dtype)}")
    if dtype is BF16:
        gap = int(np.max(np.abs(got.astype(np.int32) - want.astype(np.int32))))
        print(f"  {label}: max bit gap={gap}")
    else:
        print(f"  {label}: max abs err={float(np.max(np.abs(got - want))):.3e}")


def main(argv: list[str] | None = None) -> int:
    """入口，返回进程退出码。"""
    parser = argparse.ArgumentParser(description="corex 归一化逐位诊断")
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--cols", type=int, default=64)
    parser.add_argument("--dtype", choices=sorted(BY_NAME), default="bf16")
    parser.add_argument("--arch", default="ivcore10")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--eps", type=float, default=1e-6)
    args = parser.parse_args(argv)

    import cxrt as rt

    try:
        runtime = rt.open_runtime(args.device)
    except RtError as exc:
        print(f"[env] cannot open device: {exc}")
        return 2
    print(f"[env] {runtime.stats()}")

    try:
        norm = runtime.build_norm(args.arch)
    except BuildError as exc:
        print(f"[build] {exc}")
        runtime.close()
        return 2

    dtype = BY_NAME[args.dtype]
    rng = np.random.default_rng(3)
    x = rng.standard_normal((args.rows, args.cols), dtype=np.float32)
    add = rng.standard_normal((args.rows, args.cols), dtype=np.float32) * 0.5
    gain = (1.0 + rng.standard_normal(args.cols, dtype=np.float32) * 0.02).astype(np.float32)
    try:
        xd = _fill(runtime.cudart, runtime.alloc, x, dtype)
        ad = _fill(runtime.cudart, runtime.alloc, add, dtype)
        wd = _fill(runtime.cudart, runtime.alloc, gain.reshape(1, -1), dtype).view((args.cols,))
        print(f"[ptr] x={xd.ptr} add={ad.ptr} w={wd.ptr} (aliased: "
              f"{xd.same_storage(ad)})")
        runtime.stream.sync()

        # 上传后先回读，确认 H2D 真的到位；这一步出错会让后面所有比较都变成误导。
        x_back = _raw(xd)
        a_back = _raw(ad)
        print(f"[h2d] x   got={_bits(x_back[0][:8], dtype)}")
        print(f"[h2d] x   want={_bits(bf16_bits(x)[0][:8], dtype)}" if dtype is BF16
              else f"[h2d] x want={_bits(x[0][:8], dtype)}")
        print(f"[h2d] add got={_bits(a_back[0][:8], dtype)}")

        yd = norm.add_rms_norm(xd, ad, wd, eps=args.eps, stream=runtime.stream.handle)
        runtime.stream.sync()
        got_sum = _raw(xd)
        got_y = _raw(yd)
        if dtype is BF16:
            want_sum = bf16_bits(bf16_to_f32(bf16_bits(x)) + bf16_to_f32(bf16_bits(add)))
            want_y = bf16_bits(host_reference("rms", bf16_to_f32(bf16_bits(x))
                                              + bf16_to_f32(bf16_bits(add)), gain, eps=args.eps))
        else:
            want_sum = x + add
            want_y = host_reference("rms", want_sum, gain, eps=args.eps)
        _row_compare("residual", got_sum[0], want_sum[0], dtype)
        _row_compare("output  ", got_y[0], want_y[0], dtype)
        # 逐行看谁不对：一行的间距错位会让后面每一行都错，而只有某几行错通常是
        # 宽度相关的越界/共享内存不足。
        rows = got_sum.shape[0]
        for r in range(rows):
            print(f"[row {r}] residual got ={_bits(got_sum[r][:8], dtype)}")
            print(f"[row {r}] residual want={_bits(want_sum[r][:8], dtype)}")
        print(f"[pool] {runtime.alloc.stats()}")
    finally:
        runtime.close()
    return 0


if __name__ == "__main__":                          # pragma: no cover - 入口
    sys.exit(main())
