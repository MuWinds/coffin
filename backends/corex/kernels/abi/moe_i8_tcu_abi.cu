// @file moe_i8_tcu_abi.cu
// @brief 分组 int8 张量核的 C ABI 出口，外加它专用的 k-主序转置前置核；无 torch。
//
// Module: coffin/backends/corex/kernels/abi。设备代码来自
// ``../moe_i8_tcu_device.inc``——与原来的 torch 主机包装是同一份
// ``moe_grouped_i8_tcu_kernel`` / ``transpose_i8_kernel`` 与同一套块几何常量。
//
// 两条出口是**一对**：预填充那条路每次投影都要先把它自己的激活按 k 主序摆一份
// （``coffin_transpose_i8``），再喂给分组核。分开放是因为转置的源和目标形状不同，
// 合在一个出口里只会让参数表更难读。
//
// Constraints:
//   * 签名与 ``cxrt/moe.py`` 的 ``argtypes`` 必须逐字一致：多一个参数就整体错位。
//   * ``K`` 必须是 ``MI8_BK`` 的整数倍——块几何按 k 方向切块，余块没有零填充。
//   * 每专家不足 4 行的形状**不能**进这条：共享内存里的 64 行 tile 末端没有零填充，
//     短段的垃圾行会被 mma 算进有效行。这个门在调用方（Python 包装）挡，出口只做
//     几何校验，因为「每专家几行」要读 ``offsets`` 才知道，读它就要同步。

#include "cx_tensor.h"

#include <cstdint>

#include "../moe_i8_tcu_device.inc"

// @brief ``[rows, K]`` int8 -> ``[K, rows]``，64x64 分块走共享内存。
//
// @param src ``[rows, K]`` 连续 int8。
// @param dst ``[K, rows]`` 连续 int8。
// @param rows 行数。
// @param k 宽度。
// @param stream 目标流；空指针表示默认流。
// @param err 错误缓冲，可为空。
// @param errlen 错误缓冲字节数。
// @return ``0`` 成功。
extern "C" int coffin_transpose_i8(const void* src, void* dst, int rows, int k, void* stream,
                                   char* err, int errlen) {
    const cx::Err e{err, errlen};
    if (!src || !dst) return cx::fail(e, "src and dst must be non-null");
    if (rows <= 0 || k <= 0) return cx::fail(e, "rows/k must be positive, got %d/%d", rows, k);

    const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    const dim3 grid((unsigned)((rows + 63) / 64), (unsigned)((k + 63) / 64));
    transpose_i8_kernel<<<grid, 256, 0, s>>>(reinterpret_cast<const int8_t*>(src),
                                             reinterpret_cast<int8_t*>(dst), rows, k);
    return cx::finish(e);
}

// @brief 分组 W8A8 张量核：``out[r, n] = xs[r] * ws[e, n] * sum_k at[k, r] * w[e, n, k]``。
//
// @param at ``[K, rows]`` int8 连续（``coffin_transpose_i8`` 的产物）。
// @param xs 逐行激活 scale，``[rows]`` fp32。
// @param w 专家栈，``[E, N, K]`` 连续 int8。
// @param ws 专家栈 scale，``[E, N]`` fp32。
// @param offsets ``[E+1]`` 前缀和，int32。
// @param out ``[rows, N]``，dtype 由 ``out_code`` 给。
// @param out_code 目的地 dtype 码（``kBF16`` / ``kF32``）。
// @param rows 行数。
// @param k 激活宽度（也是专家栈的列长）。
// @param n 输出列数。
// @param experts 专家数（``offsets`` 的长度是它加一）。
// @param stream 目标流；空指针表示默认流。
// @param err 错误缓冲，可为空。
// @param errlen 错误缓冲字节数。
// @return ``0`` 成功。
extern "C" int coffin_moe_grouped_i8_tcu(const void* at, const void* xs, const void* w,
                                         const void* ws, const void* offsets, void* out,
                                         int out_code, int rows, int k, int n, int experts,
                                         void* stream, char* err, int errlen) {
    const cx::Err e{err, errlen};
    if (!at || !xs || !w || !ws || !offsets || !out) {
        return cx::fail(e, "at, xs, w, ws, offsets and out must be non-null");
    }
    if (out_code != cx::kF32 && out_code != cx::kBF16) {
        return cx::fail(e, "out dtype code %d is neither fp32 nor bf16", out_code);
    }
    if (rows <= 0 || k <= 0 || n <= 0 || experts <= 0) {
        return cx::fail(e, "rows/k/n/experts must be positive, got %d/%d/%d/%d", rows, k, n,
                        experts);
    }
    if (k % MI8_BK != 0) {
        return cx::fail(e, "K must be a multiple of %d, got %d", MI8_BK, k);
    }

    const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    const int8_t* atp = reinterpret_cast<const int8_t*>(at);
    const float* xsp = reinterpret_cast<const float*>(xs);
    const int8_t* wp = reinterpret_cast<const int8_t*>(w);
    const float* wsp = reinterpret_cast<const float*>(ws);
    const int* off = reinterpret_cast<const int*>(offsets);

    // TP4 下 gate_up 只有 N=256：用 512 宽 tile 会让一半 warp 空转并多占一倍共享内存。
    auto launch = [&](auto tag) {
        using XT = decltype(tag);
        if (n <= 256) {
            constexpr int TBN = 256;
            constexpr int NTH = (MI8_BM / 64) * (TBN / 64) * 64;
            constexpr size_t kShmem = 2 * (size_t)((MI8_BM / 64) * (MI8_BK / 16) * 1024
                                                   + (TBN / 64) * (MI8_BK / 16) * 1024);
            const dim3 grid((unsigned)((n + TBN - 1) / TBN), (unsigned)experts);
            moe_grouped_i8_tcu_kernel<XT, TBN><<<grid, NTH, kShmem, s>>>(
                atp, rows, wp, reinterpret_cast<XT*>(out), xsp, wsp, off, k, n);
        } else {
            constexpr int TBN = MI8_BN;
            constexpr int NTH = (MI8_BM / 64) * (TBN / 64) * 64;
            constexpr size_t kShmem = 2 * (size_t)((MI8_BM / 64) * (MI8_BK / 16) * 1024
                                                   + (TBN / 64) * (MI8_BK / 16) * 1024);
            const dim3 grid((unsigned)((n + TBN - 1) / TBN), (unsigned)experts);
            moe_grouped_i8_tcu_kernel<XT, TBN><<<grid, NTH, kShmem, s>>>(
                atp, rows, wp, reinterpret_cast<XT*>(out), xsp, wsp, off, k, n);
        }
    };
    if (out_code == cx::kBF16) launch(uint16_t{});
    else launch(float{});
    return cx::finish(e);
}
