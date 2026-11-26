// @file abi/gemv_abi.cu
// @brief ``gemv_device.inc`` 的 C ABI 前端：**nk 权重**的 int8 投影，不链接 torch。
//
// Module: coffin/backends/corex/kernels/abi。与 ``gemv.cu`` 包含同一份设备代码
// （``gemv_device.inc``），差别只在主机侧：这里收裸指针、dtype 码值和调用方透传的流，
// 输出缓冲区由调用方持有，错误写进调用方的缓冲并返回码——没有 ``torch::Tensor``，
// 没有 pybind，没有 ``TORCH_CHECK``。
//
// Constraints:
//   * 只暴露 ``[N, K]``（nk）权重。以后为 corex 写的适配层默认都是 nk 权重，所以
//     ``[K, N]``（kn）那两族入口没有搬到这里——它们的设备代码仍留在
//     ``gemv_device.inc`` 里给遗留的 pybind 路径用，但新的 ABI 面不接受 kn 权重，
//     免得再养一条没人走的路径。真的需要 kn 时应当先转成 nk 权重进来。
//   * 输出必须由调用方给：地址稳定是解码步能被捕获重放的前提，也是这里不需要在
//     C++ 侧分配任何东西的原因。
//   * 本文件不校验语义级别名关系（谁能与谁共享内存），只校验「内核索引得了」：
//     dtype 码、长度、``K % 4``、bf16 激活的 8 字节对齐、``threads % 32``、``width``。
//   * 分片（split-K）不存在于 nk 路径：一个输出列就是一个 32 线程组，网格天然铺满
//     ``N``。所以这里没有 ``part`` / ``counters`` 两个参数，也没有计划查询出口。

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

// 设备内核与调优规则：与 ``gemv.cu`` 共用同一份，逐位一致。
#include "../gemv_device.inc"

// @brief bf16/fp32 的激活 dtype 码判定。
// @param code 调用方给的码值。
// @return 内核实例化过这种激活时为真。
bool is_half_or_float(int code) { return code == cx::kBF16 || code == cx::kF32; }

// @brief 两个 nk 入口共用的参数校验。
// @param e 错误缓冲。
// @param x_code 激活 dtype 码。
// @param x 激活指针（只用于对齐检查）。
// @param N 输出列数。
// @param K 内维。
// @param M 行数。
// @param threads 请求的线程数。
// @param width 请求的每组列数。
// @param out_th 写回解析后的线程数。
// @param out_wd 写回解析后的每组列数。
// @return 0 通过，1 参数非法。
int check_nk(const cx::Err& e, const void* x, int x_code, long long N, long long K,
             long long M, int threads, int width, int* out_th, int* out_wd) {
    if (N <= 0 || K <= 0 || M <= 0) return cx::fail(e, "nk got M=%lld N=%lld K=%lld",
                                                   M, N, K);
    if (!is_half_or_float(x_code)) return cx::fail(e, "activation dtype code %d", x_code);
    if (K % 4 != 0) {
        return cx::fail(e, "nk GEMV needs K %% 4 == 0 (rows would not stay 4-byte aligned "
                           "across the group's loads); got K=%lld", K);
    }
    if (x_code == cx::kBF16 && (reinterpret_cast<uintptr_t>(x) % 8) != 0) {
        return cx::fail(e, "bf16 activation must be 8-byte aligned (the nk kernel reads "
                           "four adjacent halves as one word)");
    }
    const int th = threads > 0 ? threads : pick_nk_threads((int)N);
    if (th % 32 != 0) {
        return cx::fail(e, "threads must be a multiple of 32 (one group per output "
                           "column); got %d", th);
    }
    const int wd = width > 0 ? width : 1;
    if (wd != 1 && wd != 2) {
        return cx::fail(e, "width must be 1 or 2 output columns per group; got %d", wd);
    }
    *out_th = th;
    *out_wd = wd;
    return 0;
}

// @brief 按激活 dtype 分派 ``launch_nk_i8_rows``。
template <typename XT>
void run_nk(const int8_t* wp, const void* x, void* y, const float* sp, int N, int K,
            int rows, int threads, int unroll, int width) {
    launch_nk_i8_rows<XT>(wp, static_cast<const XT*>(x), static_cast<XT*>(y), sp, N, K,
                          rows, threads, unroll, width);
}

}  // namespace

extern "C" {

// @brief 解码期 W8A16 投影，权重 ``[N, K]``（nk）：一行权重一个输出列。
//
// @param wtn ``[N, K]`` int8 权重，连续。
// @param x ``[K]`` 激活，码值 ``x_code``；输出的 dtype 跟随它。
// @param x_code 激活 dtype 码（bf16 或 fp32）。
// @param scale ``[N]`` fp32 每列 scale。
// @param N 输出列数。
// @param K 内维；必须 ``K % 4 == 0``。
// @param y ``[N]`` 输出，由调用方分配。
// @param threads 每块线程数，``0`` 表示 128；必须是 32 的倍数。
// @param unroll 展开度，``0`` 表示 4。
// @param width 每组列数，``0`` 表示 1；只接受 1 或 2。
// @param stream 调用方透传的流，``nullptr`` 为默认流。
// @param err 错误缓冲，可为空。
// @param errlen 错误缓冲字节数。
// @return 0 成功，1 参数非法，2 启动失败。
int coffin_gemv_nk_i8(const void* wtn, const void* x, int x_code, const void* scale,
                      long long N, long long K, void* y, int threads, int unroll,
                      int width, void* stream, char* err, int errlen) {
    const cx::Err e{err, errlen};
    g_cx_stream = static_cast<cudaStream_t>(stream);
    if (!wtn || !x || !y || !scale) return cx::fail(e, "gemv_nk_i8 needs non-null args");
    int th = 0;
    int wd = 0;
    const int rc = check_nk(e, x, x_code, N, K, 1, threads, width, &th, &wd);
    if (rc != 0) return rc;
    const int u = unroll > 0 ? unroll : 4;
    const int8_t* wp = static_cast<const int8_t*>(wtn);
    const float* sp = static_cast<const float*>(scale);
    if (x_code == cx::kBF16) {
        run_nk<uint16_t>(wp, x, y, sp, (int)N, (int)K, 1, th, u, wd);
    } else {
        run_nk<float>(wp, x, y, sp, (int)N, (int)K, 1, th, u, wd);
    }
    return cx::finish(e);
}

// @brief 批处理 W8A16 投影，权重 ``[N, K]``。
//
// @param wtn ``[N, K]`` int8 权重。
// @param x ``[M, K]`` 激活。
// @param x_code 激活 dtype 码。
// @param scale ``[N]`` fp32 每列 scale。
// @param N 输出列数。
// @param K 内维。
// @param M 行数。
// @param y ``[M, N]`` 输出。
// @param threads 每块线程数，``0`` 表示 128。
// @param unroll 展开度，``0`` 表示 4。
// @param width 每组列数，``0`` 表示 1。
// @param stream 流。
// @param err 错误缓冲。
// @param errlen 缓冲长度。
// @return 0 成功，1 参数非法，2 启动失败。
int coffin_gemv_nk_i8_m(const void* wtn, const void* x, int x_code, const void* scale,
                        long long N, long long K, long long M, void* y, int threads,
                        int unroll, int width, void* stream, char* err, int errlen) {
    const cx::Err e{err, errlen};
    g_cx_stream = static_cast<cudaStream_t>(stream);
    if (!wtn || !x || !y || !scale) return cx::fail(e, "gemv_nk_i8_m needs non-null args");
    int th = 0;
    int wd = 0;
    const int rc = check_nk(e, x, x_code, N, K, M, threads, width, &th, &wd);
    if (rc != 0) return rc;
    const int u = unroll > 0 ? unroll : 4;
    const int8_t* wp = static_cast<const int8_t*>(wtn);
    const float* sp = static_cast<const float*>(scale);
    if (x_code == cx::kBF16) {
        run_nk<uint16_t>(wp, x, y, sp, (int)N, (int)K, (int)M, th, u, wd);
    } else {
        run_nk<float>(wp, x, y, sp, (int)N, (int)K, (int)M, th, u, wd);
    }
    return cx::finish(e);
}

}  // extern "C"
