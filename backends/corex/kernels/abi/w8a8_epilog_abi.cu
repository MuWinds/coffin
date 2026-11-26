// @file abi/w8a8_epilog_abi.cu
// @brief W8A8 epilogue 的 C ABI 前端：``out = narrow(c * xs[m] * ws[n])``，不链接 torch。
//
// Module: coffin/backends/corex/kernels/abi。设备代码来自 ``../w8a8_epi_device.inc``——
// 与 ``../w8a8.cu`` 是同一份，所以两条前端的算术逐位相同（两个 scale 在同一个 fp32
// 表达式里乘完再窄化一次，与稠密路径的舍入次数一致）。
//
// 为什么单独给 epilogue 一条出口：它是 int8 预填充里唯一「必须知道两个 scale」的环节，
// 而 cublas 的 ``GemmEx`` 只能给出 int32 累加器。把累加器交回 torch 做 ``mul`` 就等于
// 给推理链路续上 torch；这条出口让 ``quant_act_i8``(ABI) → ``GemmEx``(libcublas) →
// epilogue(ABI) 三步全在 torch 之外闭环。
//
// Constraints:
//   * ``c`` 必须连续：内核按 ``c + m * N`` 走行，没有给累加器留行间距。这是刻意的——
//     int32 累加器只由 ``GemmEx`` 写，那个后端产出的就是连续 ``[M, N]``。
//   * ``out`` 只需要行间距 ``stride >= cols``，允许是更宽缓冲区的列视图：沿 N 切块
//     （``lm_head`` 这类 N > 131072 的形状）时靠它把每块写回原位，不必再拷一次。
//   * 块数固定 ``EPI_BLOCKS``，与形状无关；见 ``w8a8_epi_device.inc`` 里的实测。
//   * 流由调用方透传，不查任何全局「当前流」。调用方必须传 torch 的当前流
//     （``torch.cuda.current_stream().cuda_stream``），否则两条流会在同一批字节上竞态。

#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include "cx_tensor.h"
#include "corex_scalar.h"

namespace {

//: 调用方透传的流。``nullptr`` 表示默认流；每次入口进来先写它，退出前不动。
thread_local cudaStream_t g_cx_stream = nullptr;

// @brief 设备侧代码问“用哪条流”。（epilogue 内核不走它，但保持与其它 ABI 文件同一形状，
//        免得后来给这里加第二个内核时忘了补。）
cudaStream_t cx_stream() { return g_cx_stream; }

// 内核与常量：与 ``../w8a8.cu`` 共用同一份。
#include "../w8a8_epi_device.inc"

}  // namespace

extern "C" {

// @brief 施加两个 W8A8 scale 并把 int32 累加器窄化一次。
//
// ``out`` 可以就是模型缓冲区的列视图：沿 N 切块时每块写回自己那一段，行间距由
// ``stride`` 给出。写入量与一次同形状的 bf16 拷贝相同，没有额外读写。
//
// @param cp ``[rows, cols]`` int32 累加器，连续。
// @param xsp ``[rows]`` 每 token 激活 scale（fp32）。
// @param wsp ``[cols]`` 每输出列权重 scale（fp32），与被写的列段对齐。
// @param outp 目的地，行间距 ``stride``。
// @param rows 行数。
// @param cols 本块列数（读入的累加器行宽）。
// @param stride ``out`` 的行间距，单位元素，``>= cols``。
// @param code dtype 码（bf16 或 fp32）。
// @param stream 目标流。
// @param errbuf 错误缓冲。
// @param errlen 缓冲长度。
// @return ``0`` 成功，``1`` 参数非法，``2`` 启动失败。
int coffin_w8a8_epilogue(const void* cp, const void* xsp, const void* wsp, void* outp,
                         long long rows, long long cols, long long stride, int code,
                         void* stream, char* errbuf, int errlen) {
    const cx::Err err{errbuf, errlen};
    g_cx_stream = static_cast<cudaStream_t>(stream);
    if (!cp || !xsp || !wsp || !outp) {
        return cx::fail(err, "w8a8_epilogue needs non-null c/xs/ws/out");
    }
    if (rows <= 0 || cols <= 0) return cx::fail(err, "rows=%lld cols=%lld", rows, cols);
    if (stride < cols) {
        return cx::fail(err, "out pitch %lld narrower than the row width %lld", stride,
                        cols);
    }
    // 网格维度是 unsigned：先挡住溢出，再谈能不能算。
    if (rows > 2147483647LL || cols > 2147483647LL || stride > 2147483647LL) {
        return cx::fail(err, "shape too large for 32-bit indexing: %lld x %lld @ %lld",
                        rows, cols, stride);
    }
    const dim3 grid((unsigned)EPI_BLOCKS);
    switch (code) {
        case cx::kBF16:
            w8a8_epilogue_kernel<uint16_t><<<grid, EPI_THREADS, 0, g_cx_stream>>>(
                static_cast<const int32_t*>(cp), static_cast<const float*>(xsp),
                static_cast<const float*>(wsp), static_cast<uint16_t*>(outp), (int)rows,
                (int)cols, (int)stride);
            break;
        case cx::kF32:
            w8a8_epilogue_kernel<float><<<grid, EPI_THREADS, 0, g_cx_stream>>>(
                static_cast<const int32_t*>(cp), static_cast<const float*>(xsp),
                static_cast<const float*>(wsp), static_cast<float*>(outp), (int)rows,
                (int)cols, (int)stride);
            break;
        default:
            return cx::fail(err, "unsupported dtype code %d (bf16 or fp32 only)", code);
    }
    return cx::finish(err);
}

}  // extern "C"
