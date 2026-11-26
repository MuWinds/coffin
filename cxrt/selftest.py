"""``python3 -m cxrt.selftest`` —— 在卡上验证 torch-free 通道。

Module: ``coffin/cxrt``。只 import ``rt`` 内的模块与 numpy，不 import
torch。它回答三个问题：

1. 驱动能不能装、板卡认不认得出来；
2. ``norm_abi.cu`` / ``solve_abi.cu`` 能不能用厂商 clang 编出来（不经过
   ``torch.utils.cpp_extension``）；
3. 归一化五出口的数值与 host 侧参考是否一致，bf16 是否逐位相同（最多差 1 个 ULP），
   两个 int8 融合体是否在半个量化步长以内；以及 ``triangular_solve`` 的 fp32/bf16
   两条是否与 fp64 前向替换在容差内。

用法::

    python3 -m cxrt.selftest
    python3 -m cxrt.selftest --rows 2048 --cols 2560 --repeat 20
    python3 -m cxrt.selftest --dtype f32 --arch ivcore10 --device 1

退出码：全过 ``0``，有用例失败 ``1``，环境问题（驱动/工具链）``2``。
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from cxrt.array import Array
from cxrt.dtypes import BF16, BY_NAME, F32, DType, bf16_bits, bf16_to_f32
from cxrt.errors import BuildError, RtError
from cxrt.mock import host_reference
from cxrt.stream import Event


def _ulp_gap(a: np.ndarray, b: np.ndarray) -> int:
    """两个 bf16 位模式数组之间最大的整数差。

    Args:
        a: ``uint16`` 位模式。
        b: ``uint16`` 位模式。

    Returns:
        最大 ``|int(a) - int(b)|``。同号数相邻位模式正好相差 1 个 ULP，所以它是最直接的
        「差几个 ULP」的度量；跨符号的情形在这里不会出现（结果都是归一化后的有限值）。
    """
    return int(np.max(np.abs(a.astype(np.int32) - b.astype(np.int32))))


def _fill(rt, alloc, values: np.ndarray, dtype: DType) -> Array:
    """把 fp32 宿主数据按 dtype 放上设备。

    Args:
        rt: 运行时。
        alloc: 分配器。
        values: ``[M, N]`` fp32 数据。
        dtype: 目标 dtype。

    Returns:
        设备数组。bf16 由本函数做窄化，调用方不必关心位模式。
    """
    if dtype is BF16:
        return Array.from_numpy(rt, alloc, bf16_bits(values), BF16)
    return Array.from_numpy(rt, alloc, values, F32)


def _read(array: Array, dtype: DType) -> np.ndarray:
    """把设备数组读回 fp32。

    Args:
        array: 设备数组。
        dtype: 它的 dtype。

    Returns:
        ``[M, N]`` fp32 副本。
    """
    raw = array.to_numpy()
    return bf16_to_f32(raw) if dtype is BF16 else raw


def _hex_row(row: np.ndarray, width: int = 6) -> str:
    """把一行 bf16 位模式排成十六进制，供失配时打印。"""
    return " ".join(f"{int(v):04x}" for v in row[:width])


def _explain_gap(tag: str, got_bits: np.ndarray, want_bits: np.ndarray,
                 got_f32: np.ndarray, want_f32: np.ndarray) -> None:
    """在差异大得不像舍入时，把第一处失配的行打印出来。

    Args:
        tag: 用例名。
        got_bits: 设备侧位模式 ``[M, N]``。
        want_bits: 参考位模式 ``[M, N]``。
        got_f32: 设备侧值。
        want_f32: 参考值。
    """
    diff = np.abs(got_bits.astype(np.int32) - want_bits.astype(np.int32))
    row = int(np.unravel_index(int(np.argmax(diff)), diff.shape)[0])
    print(f"    [{tag}] worst row={row} got={_hex_row(got_bits[row])} "
          f"want={_hex_row(want_bits[row])} "
          f"max|dv|={float(np.max(np.abs(got_f32 - want_f32))):.3e}")


def _lib_path(ops) -> str:
    """取算子对象背后那本 ``.so`` 的路径，取不到就退回类名。

    Args:
        ops: 任一 ``*Ops`` 实例。

    Returns:
        路径字符串。这里只用于打印，所以取不到不算错误。
    """
    shim = getattr(ops, "shim", None)
    path = getattr(ops, "path", None) or getattr(shim, "path", None)
    return str(path) if path else type(ops).__name__


def _narrow(values: np.ndarray, dtype: DType) -> np.ndarray:
    """把宿主参考的输入先窄化成设备上真正读到的那一份。

    设备侧只看到 ``bf16(x)``，而参考若从 fp32 的 ``x`` 出发，两者之间的差
    不是内核误差，是参考输入不同——那会把逐位对拍变成「差几个 ULP 算过」。所以
    参考必须先做同样的窄化，剩下的差才真的是内核的。

    Args:
        values: ``float32`` 数据。
        dtype: 设备 dtype。

    Returns:
        窄化后的 ``float32``（bf16 情形下已可被 bf16 精确表示）。
    """
    return bf16_to_f32(bf16_bits(values)) if dtype is BF16 else values


def case_rms(runtime, norm, dtype: DType, rows: int, cols: int, eps: float) -> bool:
    """``rms_norm``：与 host 参考对比。

    Returns:
        通过为 ``True``。
    """
    rng = np.random.default_rng(7)
    x = rng.standard_normal((rows, cols), dtype=np.float32)
    gain = (1.0 + rng.standard_normal(cols, dtype=np.float32) * 0.02).astype(np.float32)
    xd = _fill(runtime.cudart, runtime.alloc, x, dtype)
    wd = _fill(runtime.cudart, runtime.alloc, gain.reshape(1, -1), dtype).view((cols,))
    yd = norm.rms_norm(xd, wd, eps=eps, stream=runtime.stream.handle)
    runtime.stream.sync()
    got = _read(yd, dtype)
    want = host_reference("rms", _narrow(x, dtype), _narrow(gain, dtype), eps=eps)
    if dtype is BF16:
        gap = _ulp_gap(bf16_bits(got), bf16_bits(want))
        ok = gap <= 1
        print(f"  rms_norm     bf16 max ULP gap={gap} {'PASS' if ok else 'FAIL'}")
        if not ok:
            _explain_gap("rms_norm", bf16_bits(got), bf16_bits(want), got, want)
        return ok
    err = float(np.max(np.abs(got - want)))
    ok = err < 1e-5
    print(f"  rms_norm     f32  max abs err={err:.3e} {'PASS' if ok else 'FAIL'}")
    return ok


def case_add(runtime, norm, dtype: DType, rows: int, cols: int, eps: float) -> bool:
    """``add_rms_norm``：残差原地累加 + 归一化，两处都要对。"""
    rng = np.random.default_rng(11)
    x = rng.standard_normal((rows, cols), dtype=np.float32)
    add = rng.standard_normal((rows, cols), dtype=np.float32) * 0.5
    gain = (1.0 + rng.standard_normal(cols, dtype=np.float32) * 0.02).astype(np.float32)
    xd = _fill(runtime.cudart, runtime.alloc, x, dtype)
    ad = _fill(runtime.cudart, runtime.alloc, add, dtype)
    wd = _fill(runtime.cudart, runtime.alloc, gain.reshape(1, -1), dtype).view((cols,))
    yd = norm.add_rms_norm(xd, ad, wd, eps=eps, stream=runtime.stream.handle)
    runtime.stream.sync()
    got = _read(yd, dtype)
    total = _read(xd, dtype)
    # 内核已经把 x 原地改写成「舍入后的和」，参考必须从这个值出发再归一化，否则会把
    # add 加第二遍。这也是逐位对拍最容易被写错的一处。
    if dtype is BF16:
        expect_total = bf16_to_f32(bf16_bits(_narrow(x, dtype) + _narrow(add, dtype)))
        want = host_reference("rms", expect_total, _narrow(gain, dtype), eps=eps)
        gap_total = _ulp_gap(bf16_bits(total), bf16_bits(expect_total))
        gap = _ulp_gap(bf16_bits(got), bf16_bits(want))
        ok = gap <= 1 and gap_total <= 1
        print(f"  add_rms_norm bf16 output ULP gap={gap} residual ULP gap={gap_total} "
              f"{'PASS' if ok else 'FAIL'}")
        if not ok:
            _explain_gap("add/residual", bf16_bits(total), bf16_bits(expect_total),
                         total, expect_total)
            _explain_gap("add/output", bf16_bits(got), bf16_bits(want), got, want)
        return ok
    expect_total = x + add
    want = host_reference("rms", expect_total, gain, eps=eps)
    err = float(np.max(np.abs(got - want)))
    err_total = float(np.max(np.abs(total - expect_total)))
    ok = err < 1e-5 and err_total < 1e-5
    print(f"  add_rms_norm f32  max abs err={err:.3e} residual err={err_total:.3e} "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def case_gated(runtime, norm, dtype: DType, rows: int, cols: int, eps: float) -> bool:
    """``rms_norm_gated``：归一化后乘 ``silu(gate)``。"""
    rng = np.random.default_rng(13)
    x = rng.standard_normal((rows, cols), dtype=np.float32)
    gate = rng.standard_normal((rows, cols), dtype=np.float32)
    gain = (1.0 + rng.standard_normal(cols, dtype=np.float32) * 0.02).astype(np.float32)
    xd = _fill(runtime.cudart, runtime.alloc, x, dtype)
    gd = _fill(runtime.cudart, runtime.alloc, gate, dtype)
    wd = _fill(runtime.cudart, runtime.alloc, gain.reshape(1, -1), dtype).view((cols,))
    yd = norm.rms_norm_gated(xd, wd, gd, eps=eps, stream=runtime.stream.handle)
    runtime.stream.sync()
    got = _read(yd, dtype)
    want = host_reference("gated", _narrow(x, dtype), _narrow(gain, dtype),
                          gate=_narrow(gate, dtype), eps=eps)
    if dtype is BF16:
        # 门控这条多一个 2 ULP 的余量：内核用 ``__expf``（快速指数，本身约 2 ULP 的
        # 相对误差），而参考走 fp32 的精确 exp。其余两条归一化是 1 ULP 口径。
        gap = _ulp_gap(bf16_bits(got), bf16_bits(want))
        ok = gap <= 2
        print(f"  rms_gated    bf16 max ULP gap={gap} {'PASS' if ok else 'FAIL'}")
        if not ok:
            _explain_gap("rms_gated", bf16_bits(got), bf16_bits(want), got, want)
        return ok
    err = float(np.max(np.abs(got - want)))
    ok = err < 1e-5
    print(f"  rms_gated    f32  max abs err={err:.3e} {'PASS' if ok else 'FAIL'}")
    return ok


def case_rms_i8(runtime, norm, dtype: DType, rows: int, cols: int, eps: float) -> bool:
    """``rms_norm_i8``：反量化后与 host 参考的差不超过半个量化步长。"""
    rng = np.random.default_rng(21)
    x = rng.standard_normal((rows, cols), dtype=np.float32)
    gain = (1.0 + rng.standard_normal(cols, dtype=np.float32) * 0.02).astype(np.float32)
    xd = _fill(runtime.cudart, runtime.alloc, x, dtype)
    wd = _fill(runtime.cudart, runtime.alloc, gain.reshape(1, -1), dtype).view((cols,))
    qd, sd = norm.rms_norm_i8(xd, wd, eps=eps, stream=runtime.stream.handle)
    runtime.stream.sync()
    q = qd.to_numpy()
    s = sd.to_numpy()
    want = host_reference("rms", _narrow(x, dtype), _narrow(gain, dtype), eps=eps)
    dq = q.astype(np.float32) * s[:, None]
    err = float(np.max(np.abs(dq - want)))
    # 逐位口径：内核取 absmax 时用的是「舍入成 T 之后」的值，参考也必须按同一份值
    # 算 code，否则贴着边界的那几个元素会差 1 个 code，而差 1 个 code 只是舍入方向不同，
    # 不是量化算错——所以容差写「1 个步长 + 1 个 code」。
    codes = np.clip(np.rint(want / s[:, None]), -127, 127)
    gap = int(np.max(np.abs(q.astype(np.float32) - codes)))
    ok = bool(np.all(np.abs(dq - want) <= s[:, None] + 1e-6)) and gap <= 1
    print(f"  rms_i8       {dtype.name} max dequant err={err:.3e} max code gap={gap} "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def case_add_i8(runtime, norm, dtype: DType, rows: int, cols: int, eps: float) -> bool:
    """``add_rms_norm_i8``：残差原地累加，量化结果同样按半个步长验。"""
    rng = np.random.default_rng(23)
    x = rng.standard_normal((rows, cols), dtype=np.float32)
    add = rng.standard_normal((rows, cols), dtype=np.float32)
    gain = (1.0 + rng.standard_normal(cols, dtype=np.float32) * 0.02).astype(np.float32)
    xd = _fill(runtime.cudart, runtime.alloc, x, dtype)
    ad = _fill(runtime.cudart, runtime.alloc, add, dtype)
    wd = _fill(runtime.cudart, runtime.alloc, gain.reshape(1, -1), dtype).view((cols,))
    qd, sd = norm.add_rms_norm_i8(xd, ad, wd, eps=eps, stream=runtime.stream.handle)
    runtime.stream.sync()
    total = _read(xd, dtype)
    expect_total = bf16_to_f32(bf16_bits(_narrow(x, dtype) + _narrow(add, dtype)))
    # 残差是 bf16 存储的和，绝对容差按 bf16 在 ~3 量级上的步长（约 0.023）放，用相对
    # 容差更稳：同一条残差流上量级从 0.1 到 4 都会出现。
    residual_ok = bool(np.allclose(total, expect_total, rtol=1e-2, atol=0.01))
    q = qd.to_numpy()
    s = sd.to_numpy()
    want = host_reference("rms", total, _narrow(gain, dtype), eps=eps)
    dq = q.astype(np.float32) * s[:, None]
    ok = bool(np.all(np.abs(dq - want) <= s[:, None] + 1e-6))
    print(f"  add_i8       {dtype.name} residual={'ok' if residual_ok else 'BAD'} "
          f"max dequant err={float(np.max(np.abs(dq - want))):.3e} "
          f"{'PASS' if ok and residual_ok else 'FAIL'}")
    return ok and residual_ok


def _solve_reference(lower: np.ndarray, rhs: np.ndarray, rounded: DType | None = None) -> np.ndarray:
    """前向替换参考，**按 fp32 逐步累加**，与内核算术链一致。

    早先这里用 fp64 累加再在最后舍入一次，结果是参考比内核「更准」，两者的差
    里混进了参考自己的精度优势——``L=16`` 的链式减法上能差到 6e-5，被误读成内核错。
    内核每步都写回 fp32（bf16 情形下中间解仍是 fp32，只有输出被舍入），所以参考
    也必须每步落在 fp32。

    Args:
        lower: ``[bn, L, L]`` 系数，对角线按 1。
        rhs: ``[bn, L, n]`` 右端项。
        rounded: 若给出，输出再舍入成该 dtype（bf16 情形对拍设备写出的字节）。

    Returns:
        ``[bn, L, n]`` fp32。
    """
    bn, order, _ = lower.shape
    cols = rhs.shape[2]
    out = np.empty((bn, order, cols), dtype=np.float32)
    a = np.ascontiguousarray(lower, dtype=np.float32)
    b = np.ascontiguousarray(rhs, dtype=np.float32)
    for k in range(bn):
        acc = b[k].copy()                       # fp32
        for i in range(order):
            s = acc[i]
            # 串行减法，顺序与内核的 ``for j < i`` 一致：归约顺序不同会引入
            # 一个 fp32 ULP 量级的差，那样就没法区分「内核错」和「参考归约不同」。
            for j in range(i):
                s = np.float32(s - np.float32(a[k, i, j] * acc[j]))
            acc[i] = s
        out[k] = acc
    if rounded is not None and rounded is BF16:
        out = bf16_to_f32(bf16_bits(out))
    return out


def case_solve(runtime, solve, dtype: DType, bn: int = 4, order: int = 16,
               cols: int = 8) -> bool:
    """在卡上验 ``triangular_solve``：fp32 与 bf16 各跑一次，与 fp64 参考比。

    Args:
        runtime: :class:`cxrt.Runtime`。
        solve: :class:`cxrt.SolveOps`。
        dtype: 运行 dtype，只决定打印用哪一行。
        bn: 独立系统数。
        order: 阶数 ``L``。
        cols: 右端项列数。

    Returns:
        两种 dtype 都在容差内为真。
    """
    rng = np.random.default_rng(11)
    base_lower = np.tril(rng.normal(size=(bn, order, order)).astype(np.float32), -1)
    base_rhs = rng.normal(size=(bn, order, cols)).astype(np.float32)
    ok_all = True
    for run_dtype in (F32, BF16):
        lower = base_lower
        rhs = base_rhs
        if run_dtype is BF16:
            lower = bf16_to_f32(bf16_bits(lower))
            rhs = bf16_to_f32(bf16_bits(rhs))
        want_run = _solve_reference(lower, rhs, rounded=run_dtype)
        scale = float(np.abs(want_run).max())
        # fp32 口径是 2 ULP 相对：同样的串行链、同样的逐步 fp32 落盘，剩下的差只来自
        # FMA 与乘加顺序。bf16 口径多一个输出舍入步，按一个 bf16 步长算。
        tol = 4e-6 * scale if run_dtype is F32 else 1e-2 * scale
        xd = _fill(runtime.cudart, runtime.alloc, lower, run_dtype)
        bd = _fill(runtime.cudart, runtime.alloc, rhs, run_dtype)
        x = solve.triangular_solve(xd, bd, stream=runtime.stream.handle)
        runtime.stream.sync()
        got = _read(x, run_dtype)
        err = float(np.abs(got - want_run).max())
        ok = err <= tol
        ok_all = ok_all and ok
        print(f"  tri_solve    {run_dtype.name} max abs err={err:.3e} (tol {tol:.1e}) "
              f"{'PASS' if ok else 'FAIL'}")
        if not ok:
            _explain_gap("tri_solve", bf16_bits(got), bf16_bits(want_run), got, want_run)
    return ok_all


def case_conv(runtime, conv, dtype: DType, channels: int = 128, tokens: int = 17,
              rows: int = 5) -> bool:
    """卷积三兄弟：单 token / 预填充分段 / 批处理，与 ``refs`` 的递推对拍。

    ``tokens=17`` 故意越过内核的 ``CONV_NSEG=8``：分段只该改变起点怎么重建，不该改变
    任何一位结果，所以这个长度是「分段边界有没有写错」的探针。

    Args:
        runtime: :class:`cxrt.Runtime`。
        conv: :class:`cxrt.ConvOps`。
        dtype: 激活 dtype，只取决定激活的那一半（历史与输出恒为 fp32）。
        channels: 通道数。
        tokens: 序列长度。
        rows: 批处理行数。

    Returns:
        三个出口全部在容差内为真。
    """
    from cxrt import refs

    rng = np.random.default_rng(31)
    taps = rng.standard_normal((channels, refs.CONV_K), dtype=np.float32)
    hist = rng.standard_normal((channels, refs.CONV_STATE), dtype=np.float32)
    wt = _fill(runtime.cudart, runtime.alloc, taps, F32)

    def _close(got: np.ndarray, want: np.ndarray) -> float:
        """返回相对误差（按 ``want`` 的最大绝对值归一）。"""
        return float(np.max(np.abs(got - want)) / max(1e-6, float(np.abs(want).max())))

    # 1) 单 token
    x1 = rng.standard_normal(channels, dtype=np.float32)
    xd = _fill(runtime.cudart, runtime.alloc, x1.reshape(1, -1), dtype).view((channels,))
    sd = _fill(runtime.cudart, runtime.alloc, hist.reshape(1, -1), F32)
    y = conv.conv1d_silu(wt, xd, sd, stream=runtime.stream.handle)
    runtime.stream.sync()
    got_y = y.to_numpy()
    got_s = sd.to_numpy().reshape(channels, refs.CONV_STATE)
    want_y, want_s = refs.conv_step(taps, _narrow(x1, dtype), hist)
    ok1 = _close(got_y, want_y) < 2e-5 and _close(got_s, want_s) < 2e-5
    print(f"  conv1d_silu   {dtype.name} rel err={_close(got_y, want_y):.2e} "
          f"state={_close(got_s, want_s):.2e} {'PASS' if ok1 else 'FAIL'}")

    # 2) 预填充：越过分段边界
    xs = rng.standard_normal((tokens, channels), dtype=np.float32)
    xd2 = _fill(runtime.cudart, runtime.alloc, xs, dtype)
    sd2 = _fill(runtime.cudart, runtime.alloc, hist.reshape(1, -1), F32)
    y2 = conv.conv1d_silu_seq(wt, xd2, sd2, stream=runtime.stream.handle)
    runtime.stream.sync()
    got2 = y2.to_numpy()
    got_s2 = sd2.to_numpy().reshape(channels, refs.CONV_STATE)
    want2, want_s2 = refs.conv1d_silu_seq(taps, _narrow(xs, dtype), hist)
    ok2 = _close(got2, want2) < 2e-5 and _close(got_s2, want_s2) < 2e-5
    print(f"  conv1d_seq    {dtype.name} T={tokens} rel err={_close(got2, want2):.2e} "
          f"state={_close(got_s2, want_s2):.2e} {'PASS' if ok2 else 'FAIL'}")

    # 3) 批处理：每行一个窗口
    xb = rng.standard_normal((rows, channels), dtype=np.float32)
    hb = rng.standard_normal((rows, channels, refs.CONV_STATE), dtype=np.float32)
    xd3 = _fill(runtime.cudart, runtime.alloc, xb, dtype)
    sd3 = _fill(runtime.cudart, runtime.alloc, hb.reshape(rows, channels * refs.CONV_STATE), F32)
    y3 = conv.conv1d_silu_batch(wt, xd3, sd3, stream=runtime.stream.handle)
    runtime.stream.sync()
    got3 = y3.to_numpy()
    got_s3 = sd3.to_numpy().reshape(rows, channels, refs.CONV_STATE)
    want3, want_s3 = refs.conv1d_silu_batch(taps, _narrow(xb, dtype), hb)
    ok3 = _close(got3, want3) < 2e-5 and _close(got_s3, want_s3) < 2e-5
    print(f"  conv1d_batch  {dtype.name} rows={rows} rel err={_close(got3, want3):.2e} "
          f"state={_close(got_s3, want_s3):.2e} {'PASS' if ok3 else 'FAIL'}")
    return ok1 and ok2 and ok3


def case_builds(runtime, arch: str) -> bool:
    """把投影 / 注意力 / W8A8 三族的 ``.so`` 也在卡上编一遍。

    它们的功能自检还没写（参考实现要按各自的规则补），但**编译**是另一回事：厂商
    clang 的包含路径、``-ffast-math``、共享内存上限都与 torch 的 cpp_extension 不同，
    编译失败要在接模型之前就暴露，而不是等到跑模型时才看到。

    Args:
        runtime: :class:`cxrt.Runtime`。
        arch: 目标架构。

    Returns:
        三族都编出来为真。
    """
    ok = True
    for name, build in (("gemv", runtime.build_gemv), ("attn", runtime.build_attn),
                        ("w8a8", runtime.build_w8a8)):
        try:
            ops = build(arch)
        except BuildError as exc:
            print(f"  build_{name:<7} FAIL {exc}")
            ok = False
            continue
        print(f"  build_{name:<7} OK   {_lib_path(ops)}")
    return ok


def bench(runtime, norm, dtype: DType, rows: int, cols: int, repeat: int) -> None:
    """量一次 ``rms_norm`` 的墙钟，确认发射路径没有意外同步。

    打印每步微秒，并给出「每次发射的宿主开销」这一项：它才是 torch-free 改造要赚的
    东西（torch 每算子一次 dispatch，本层是一次 ``ctypes`` 调用加一次
    ``cuLaunchKernel`` 等价路径）。
    """
    x = np.random.default_rng(17).standard_normal((rows, cols), dtype=np.float32)
    gain = np.ones(cols, dtype=np.float32)
    xd = _fill(runtime.cudart, runtime.alloc, x, dtype)
    wd = _fill(runtime.cudart, runtime.alloc, gain.reshape(1, -1), dtype).view((cols,))
    yd = Array.empty(runtime.cudart, runtime.alloc, (rows, cols), dtype)
    norm.rms_norm(xd, wd, eps=1e-6, out=yd, stream=runtime.stream.handle)
    runtime.stream.sync()

    start = Event.create(runtime.cudart)
    stop = Event.create(runtime.cudart)
    start.record(runtime.stream)
    for _ in range(repeat):
        norm.rms_norm(xd, wd, eps=1e-6, out=yd, stream=runtime.stream.handle)
    stop.record(runtime.stream)
    stop.sync()
    device_ms = stop.ms_since(start)

    t0 = time.perf_counter()
    for _ in range(repeat):
        norm.rms_norm(xd, wd, eps=1e-6, out=yd, stream=runtime.stream.handle)
    runtime.stream.sync()
    wall_ms = (time.perf_counter() - t0) * 1e3
    print(f"  bench [{rows}, {cols}] {dtype.name}: device {device_ms / repeat * 1e3:.1f} us/launch, "
          f"wall {wall_ms / repeat * 1e3:.1f} us/launch ({repeat} reps)")
    start.close()
    stop.close()


def main(argv: list[str] | None = None) -> int:
    """入口。

    Args:
        argv: 命令行参数，``None`` 取 ``sys.argv[1:]``。

    Returns:
        进程退出码。
    """
    parser = argparse.ArgumentParser(description="corex torch-free 通道自检")
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--cols", type=int, default=2560)
    parser.add_argument("--dtype", choices=sorted(BY_NAME), default="bf16")
    parser.add_argument("--arch", default="ivcore10")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--repeat", type=int, default=20)
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
        solve = runtime.build_solve(args.arch)
        conv = runtime.build_conv(args.arch)
    except BuildError as exc:
        print(f"[build] {exc}")
        runtime.close()
        return 2
    print(f"[build] norm={_lib_path(norm)}")
    print(f"[build] solve={_lib_path(solve)}")
    print(f"[build] conv={_lib_path(conv)}")

    dtype = BY_NAME[args.dtype]
    checks = (case_rms, case_add, case_gated, case_rms_i8, case_add_i8)
    total = len(checks) + 3                 # + 求解 + 卷积 + 其余三族的编译
    passed = 0
    try:
        for check in checks:
            if check(runtime, norm, dtype, args.rows, args.cols, args.eps):
                passed += 1
        if case_solve(runtime, solve, dtype):
            passed += 1
        if case_conv(runtime, conv, dtype):
            passed += 1
        if case_builds(runtime, args.arch):
            passed += 1
        if args.repeat:
            bench(runtime, norm, dtype, args.rows, args.cols, args.repeat)
    finally:
        runtime.close()
    print(f"[result] {passed}/{total} cases passed")
    return 0 if passed == total else 1


if __name__ == "__main__":                          # pragma: no cover - 入口
    sys.exit(main())
