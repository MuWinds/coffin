// @file abi/w8a8_abi.cu
// @brief W8A8 族激活量化的 C ABI 前端：``quant_act_i8`` / ``gather_quant_i8``，
//        不链接 torch。
//
// Module: coffin/backends/corex/kernels/abi。设备代码来自 ``w8a8_device.inc``——
// 与 ``../w8a8.cu`` 是同一份，所以两条前端的量化口径逐位相同。
//
// Constraints:
//   * 这里只搬了两条量化。``w8a8_gemm`` / ``w8a8_gemm_tcu`` / ``w8a8_epilogue`` /
//     ``w8a8_dp4a_nk`` / ``moe_grouped_*`` 要么经库 GEMM（``cublas``），要么带 MoE
//     分组偏移，属于另一档；先打通预填充的写方。
//   * 激活必须**连续**：与包装层一致（它先做 ``x.contiguous()``，而调用点给的本来就是
//     投影输出的稠密片）。带间距的版本没有需求，不做。
//   * ``gather_quant_i8`` 的行索引是设备 int32，一行一个块，整行先进共享内存再量化，
//     所以共享内存是 ``K * sizeof(T)``，调用方不需要自己算。
//   * 流由调用方透传，不查任何全局「当前流」。

#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include "cx_tensor.h"
#include "corex_scalar.h"

namespace {

//: 调用方透传的流。``nullptr`` 表示默认流；每次入口进来先写它，退出前不动。
thread_local cudaStream_t g_cx_stream = nullptr;

// @brief 设备侧代码问“用哪条流”。
// @return 本次入口透传进来的句柄。
cudaStream_t cx_stream() { return g_cx_stream; }

// 两条量化内核与 ``QUANT_THREADS``：与 ``w8a8.cu`` 共用同一份。
#include "../w8a8_device.inc"

// @brief ``[M, K]`` 输入与 ``[M]`` scale 的公共校验。
//
// @param e 错误缓冲。
// @param xp 激活起点，用于空指针检查。
// @param out_qp int8 目的地。
// @param out_sp 每行 scale 目的地。
// @param rows 行数。
// @param k 行宽。
// @param code dtype 码。
// @return ``0`` 通过，否则为 ``cx::fail`` 的返回码。
int check_quant_args(const cx::Err& e, const void* xp, const void* out_qp, const void* out_sp,
                     long long rows, long long k, int code) {
    if (!xp || !out_qp || !out_sp) return cx::fail(e, "quant needs non-null x/out_q/out_s");
    if (rows <= 0 || k <= 0) return cx::fail(e, "quant got rows=%lld K=%lld", rows, k);
    if (code != cx::kBF16 && code != cx::kF32) {
        return cx::fail(e, "activation must be bf16 or fp32, code %d", code);
    }
    return 0;
}

}  // namespace

extern "C" {

// @brief 按行（per-token）对称 int8 量化，一行一个块，一次启动。
//
// 一行 = 一个 token，``amax`` 只在 K 上归约，不需要跨块通信；``1e-5`` 下限让全零行
// 量化成零而不是 inf。这是预填充 W8A8 的写方，读方是 ``w8a8_gemm`` 与 epilogue。
//
// @param xp 激活，稠密 ``[m, k]``。
// @param out_qp int8 目的地，稠密 ``[m, k]``。
// @param out_sp 每行 fp32 scale，``[m]``。
// @param m 行数。
// @param k 行宽。
// @param code dtype 码（bf16 或 fp32）。
// @param stream 目标流；空表示默认流。
// @param errbuf 错误消息缓冲，可为空。
// @param errlen 缓冲长度。
// @return ``0`` 成功，``1`` 参数非法，``2`` 启动失败。
int coffin_quant_act_i8(const void* xp, void* out_qp, void* out_sp, long long m, long long k,
                        int code, void* stream, char* errbuf, int errlen) {
    const cx::Err err{errbuf, errlen};
    g_cx_stream = static_cast<cudaStream_t>(stream);
    const int rc = check_quant_args(err, xp, out_qp, out_sp, m, k, code);
    if (rc != 0) return rc;
    int8_t* qp = static_cast<int8_t*>(out_qp);
    float* sp = static_cast<float*>(out_sp);
    const dim3 grid((unsigned)m);
    switch (code) {
        case cx::kBF16:
            quant_act_i8_kernel<uint16_t><<<grid, QUANT_THREADS, 0, g_cx_stream>>>(
                reinterpret_cast<const uint16_t*>(xp), qp, sp, (int)k);
            break;
        case cx::kF32:
            quant_act_i8_kernel<float><<<grid, QUANT_THREADS, 0, g_cx_stream>>>(
                static_cast<const float*>(xp), qp, sp, (int)k);
            break;
        default:
            return cx::fail(err, "unsupported dtype code %d", code);
    }
    return cx::finish(err);
}

// @brief 按索引表搬行并量化：``index_select`` + ``quant_act_i8`` 合成一次启动。
//
// MoE 分派之后收拢进分组投影的那些行来自 ``index`` 指向的原行。两个内核搬的是同一批
// 字节，而中间那份 bf16 缓冲除喂量化外没有第二个读者，所以这里整行先进共享内存。
//
// @param xp 源激活，稠密 ``[m, k]``；``m`` 是源行数。
// @param indexp 设备 int32 行号表，``[rows]``。
// @param out_qp int8 目的地，稠密 ``[rows, k]``。
// @param out_sp 每行 fp32 scale，``[rows]``。
// @param m 源行数（只用于越界说明，不做逐元素检查）。
// @param rows 索引表长度，也是网格宽度。
// @param k 行宽。
// @param code dtype 码。
// @param stream 目标流。
// @param errbuf 错误缓冲。
// @param errlen 缓冲长度。
// @return ``0`` 成功，``1`` 参数非法，``2`` 启动失败。
int coffin_gather_quant_i8(const void* xp, const void* indexp, void* out_qp, void* out_sp,
                           long long m, long long rows, long long k, int code, void* stream,
                           char* errbuf, int errlen) {
    const cx::Err err{errbuf, errlen};
    g_cx_stream = static_cast<cudaStream_t>(stream);
    if (!indexp) return cx::fail(err, "gather_quant_i8 needs a row index");
    const int rc = check_quant_args(err, xp, out_qp, out_sp, rows, k, code);
    if (rc != 0) return rc;
    if (m <= 0) return cx::fail(err, "gather_quant_i8 got source rows=%lld", m);
    const size_t smem = (size_t)k * (code == cx::kBF16 ? sizeof(uint16_t) : sizeof(float));
    const dim3 grid((unsigned)rows);
    int8_t* qp = static_cast<int8_t*>(out_qp);
    float* sp = static_cast<float*>(out_sp);
    const int* ip = static_cast<const int*>(indexp);
    switch (code) {
        case cx::kBF16:
            gather_quant_i8_kernel<uint16_t><<<grid, QUANT_THREADS, smem, g_cx_stream>>>(
                reinterpret_cast<const uint16_t*>(xp), ip, qp, sp, (int)k);
            break;
        case cx::kF32:
            gather_quant_i8_kernel<float><<<grid, QUANT_THREADS, smem, g_cx_stream>>>(
                static_cast<const float*>(xp), ip, qp, sp, (int)k);
            break;
        default:
            return cx::fail(err, "unsupported dtype code %d", code);
    }
    return cx::finish(err);
}

}  // extern "C"

