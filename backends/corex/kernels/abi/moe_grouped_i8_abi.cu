// @file moe_grouped_i8_abi.cu
// @brief 分组 int8 投影（nk 单趟档）的 C ABI 出口：``coffin_moe_grouped_i8``，无 torch。
//
// Module: coffin/backends/corex/kernels/abi。设备代码来自
// ``../moe_grouped_i8_device.inc``——与 ``../moe_grouped_i8.cu`` 是同一份
// ``moe_grouped_i8_nk_kernel`` / ``moe_grouped_i8_reduce_kernel`` 与共用的几何常量。
//
// 这个出口只覆盖「单趟」那一档（``span == 1``，nk 宽步与 4 字节两档都算）。分片档
// （``span > 1``）要一块 ``[span, tokens, N]`` int32 的部分和缓冲，本出口照收一个指针，
// 空指针就只在 ``span > 1`` 时报错——nk 的自动片数恒为 1，所以正常调用不需要它。
//
// Constraints:
//   * 签名与 ``cxrt/moe.py`` 的 ``argtypes`` 必须逐字一致：多一个参数就整体错位。
//   * 权重只有 nk 一种布局（``[E, N, K]``）；``nk`` 参数已从签名里去掉，这条入口就是 nk。
//   * 行号 ``>= offsets[E]`` 的行由内核整块退出，``out`` 的尾行保持调用方写下的值。
//   * ``out`` 的 dtype 由调用方定（``kBF16`` / ``kF32``），内核按它选实例化并在写回时窄化。

#include "cx_tensor.h"

#include <cstdint>

#include "../moe_grouped_i8_device.inc"

// @brief 一次启动算完整栈：行按 ``offsets`` 分段，一个 warp 认领一个输出列。
//
// @param a int8 激活，``[tokens, K]`` 连续。
// @param xs 逐行激活 scale，``[tokens]`` fp32。
// @param w 专家栈，``[E, N, K]`` 连续 int8。
// @param ws 专家栈 scale，``[E, N]`` fp32。
// @param offsets ``[E+1]`` 前缀和，int32，``offsets[0] == 0``。
// @param out ``[tokens, N]``，dtype 由 ``out_code`` 给。
// @param part 可为空；``[slices, tokens, N]`` int32 部分和，仅当 ``slices > 1`` 时必需。
// @param out_code 目的地 dtype 码（``kBF16`` / ``kF32``）。
// @param tokens 行数。
// @param k 激活宽度（也是专家栈的列长）。
// @param n 输出列数。
// @param experts 专家数（``offsets`` 的长度是它加一）。
// @param slices K 方向分片数；``<= 0`` 按块数自动选（nk 恒为 1）。
// @param stream 目标流；空指针表示默认流。
// @param err 错误缓冲，可为空。
// @param errlen 错误缓冲字节数。
// @return ``0`` 成功。
extern "C" int coffin_moe_grouped_i8(const void* a, const void* xs, const void* w, const void* ws,
                                     const void* offsets, void* out, void* part, int out_code,
                                     int tokens, int k, int n, int experts, int slices,
                                     void* stream, char* err, int errlen) {
    const cx::Err e{err, errlen};
    if (!a || !xs || !w || !ws || !offsets || !out) {
        return cx::fail(e, "a, xs, w, ws, offsets and out must be non-null");
    }
    if (out_code != cx::kF32 && out_code != cx::kBF16) {
        return cx::fail(e, "out dtype code %d is neither fp32 nor bf16", out_code);
    }
    if (tokens <= 0 || k <= 0 || n <= 0 || experts <= 0) {
        return cx::fail(e, "tokens/k/n/experts must be positive, got %d/%d/%d/%d", tokens, k, n,
                        experts);
    }

    // 与主机包装层同一套块几何：宽步要 16 的倍数且 K 不超过宽档上限，浅 K 抬 warp 数并
    // 让一条 warp 认领 4 列。
    const int vb = (k % I8_NK_WIDE_BYTES == 0 && k <= I8_NK_WIDE_K) ? I8_NK_WIDE_BYTES : 4;
    const int wcol = (vb == I8_NK_WIDE_BYTES && k < I8_NK_SHALLOW_K) ? I8_NK_WCOL_SHALLOW : 1;
    const int warps = (k < I8_NK_SHALLOW_K) ? I8_NK_WARPS_SHALLOW : I8_NK_WARPS;
    const int cols = (n + warps * wcol - 1) / (warps * wcol);
    const int cols_tail = (n + I8_COLS - 1) / I8_COLS;

    int span = slices;
    if (span <= 0) span = 1;
    span = span < 1 ? 1 : (span > I8_MAX_SLICES ? I8_MAX_SLICES : span);
    span = span > k ? k : span;
    if (span > 1 && part == nullptr) {
        return cx::fail(e, "split-K needs a [slices, tokens, N] int32 partial buffer (slices=%d)",
                        span);
    }
    const int per = ((k + span - 1) / span + vb - 1) / vb * vb;

    const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    const dim3 grid((unsigned)cols, (unsigned)tokens, (unsigned)span);
    const dim3 block((unsigned)(warps * 32));
    const size_t smem = (size_t)k * sizeof(int8_t);

    int32_t* pp = reinterpret_cast<int32_t*>(part);
    const int8_t* wp = reinterpret_cast<const int8_t*>(w);
    const float* wsp = reinterpret_cast<const float*>(ws);
    const float* xsp = reinterpret_cast<const float*>(xs);
    const int* off = reinterpret_cast<const int*>(offsets);
    const int8_t* ap = reinterpret_cast<const int8_t*>(a);

    auto launch = [&](auto tag) {
        using XT = decltype(tag);
        XT* dst = reinterpret_cast<XT*>(out);
        if (vb == I8_NK_WIDE_BYTES) {
            if (wcol > 1) {
                if (k == 512 && span == 1) {
                    moe_grouped_i8_nk_kernel<XT, I8_NK_WIDE_BYTES, I8_NK_WCOL_SHALLOW, 512>
                        <<<grid, block, smem, s>>>(ap, wp, wsp, xsp, off, pp, nullptr, nullptr,
                                                   nullptr, 0, dst, k, n, experts, tokens, span,
                                                   per);
                } else {
                    moe_grouped_i8_nk_kernel<XT, I8_NK_WIDE_BYTES, I8_NK_WCOL_SHALLOW>
                        <<<grid, block, smem, s>>>(ap, wp, wsp, xsp, off, pp, nullptr, nullptr,
                                                   nullptr, 0, dst, k, n, experts, tokens, span,
                                                   per);
                }
            } else {
                moe_grouped_i8_nk_kernel<XT, I8_NK_WIDE_BYTES, 1>
                    <<<grid, block, smem, s>>>(ap, wp, wsp, xsp, off, pp, nullptr, nullptr,
                                               nullptr, 0, dst, k, n, experts, tokens, span, per);
            }
        } else {
            moe_grouped_i8_nk_kernel<XT, 4, 1>
                <<<grid, block, smem, s>>>(ap, wp, wsp, xsp, off, pp, nullptr, nullptr, nullptr,
                                           0, dst, k, n, experts, tokens, span, per);
        }
        if (span > 1) {
            const dim3 rgrid((unsigned)cols_tail, (unsigned)tokens);
            const dim3 rblock(I8_COLS);
            moe_grouped_i8_reduce_kernel<XT><<<rgrid, rblock, 0, s>>>(pp, wsp, xsp, off, dst, n,
                                                                     experts, tokens, span);
        }
    };
    if (out_code == cx::kBF16) launch(uint16_t{});
    else launch(float{});
    return cx::finish(e);
}
