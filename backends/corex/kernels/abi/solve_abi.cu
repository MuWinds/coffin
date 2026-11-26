// @file solve_abi.cu
// @brief 求解族的 C ABI 出口：``coffin_triangular_solve``，无 torch。
//
// Module: coffin/backends/corex/kernels/abi。设备代码来自 ``../solve_device.inc``——
// 与 ``../solve.cu`` 是同一份 ``solve_kernel``，所以两条前端逐位相同。本文件只做三件事：
// 把裸指针折成 ``cx::Tensor``（只为复用它的 dtype 码与字节宽度）、按 dtype 分派实例化、
// 把驱动错误写回调用方的缓冲。
//
// 为什么求解要先搬：它是 delta 规则唯一需要的**库**操作，入口只有一个、参数全是标量、
// 没有块表也没有分组偏移；``ops/delta.py`` 每层每块都要走它，而它原来只存在于
// ``torch.linalg`` 加载不了 cuSOLVER 的那个兜底路径里。
//
// Constraints:
//   * 签名与 ``cxrt/solve.py`` 的 ``argtypes`` 必须逐字一致。多一个参数就会整体错位一格，
//     症状是内核读垃圾。
//   * ``ldb``/``ldr`` 是**元素**为单位的批步长，不是字节，也不是「连续」的假定：调用方
//     的右端项是 ``eye(L).expand(bn, L, L)``，批步长为 0。
//   * 行布局固定：最内层步长为 1，批内行间距等于行宽（``lower`` 见 ``L``，``rhs`` 见
//     ``n``）。这不是偷懒：内核按 ``i * L`` / ``i * n`` 索引，放宽它就得再传两个步长。
//   * 流由调用方传入，不查任何全局「当前流」。没有 torch 就没有那个全局状态。

#include "cx_tensor.h"
#include "corex_scalar.h"

#include <cstdint>

#include "../solve_device.inc"

namespace {

// @brief 按 dtype 码分派一次求解启动。
//
// @param lower 系数视图。
// @param rhs 右端项视图。
// @param out 目的地视图。
// @param bn 独立系统的批大小。
// @param L 阶数。
// @param n 右端项列数。
// @param ldb ``lower`` 的批步长（元素）。
// @param ldr ``rhs``/``out`` 的批步长（元素）。
// @param stream 目标流；空指针表示默认流。
// @param err 调用方的错误缓冲。
// @return ``0`` 成功，否则为 ``cx::fail`` / ``cx::finish`` 的返回码。
int dispatch_solve(const cx::Tensor& lower, const cx::Tensor& rhs, const cx::Tensor& out,
                   int bn, int L, int n, long ldb, long ldr, void* stream,
                   const cx::Err& err) {
    int cols = 0;
    const int threads = solve_threads(n, cols);
    const size_t smem = solve_smem(L, threads);
    if (smem > SOLVE_MAX_SMEM) {
        return cx::fail(err, "L=%d needs %zu bytes of shared memory, over the %zu byte "
                        "limit; reduce the delta chunk length rather than widening it",
                        L, smem, SOLVE_MAX_SMEM);
    }
    const dim3 grid(bn, (n + cols - 1) / cols);
    const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    switch (rhs.dtype) {
        case cx::kBF16:
            solve_kernel<uint16_t><<<grid, threads, smem, s>>>(
                reinterpret_cast<const uint16_t*>(lower.data),
                reinterpret_cast<const uint16_t*>(rhs.data),
                reinterpret_cast<uint16_t*>(out.data), L, n, ldb, ldr, threads, cols);
            break;
        case cx::kF32:
            solve_kernel<float><<<grid, threads, smem, s>>>(
                reinterpret_cast<const float*>(lower.data),
                reinterpret_cast<const float*>(rhs.data),
                reinterpret_cast<float*>(out.data), L, n, ldb, ldr, threads, cols);
            break;
        default:
            return cx::fail(err, "operands must be bf16 or fp32, got dtype code %d",
                            (int)rhs.dtype);
    }
    return cx::finish(err);
}

}  // namespace

// @brief 为单位下三角 ``A`` 解 ``A X = B``（``out`` 由调用方分配）。
//
// @param lower [bn, L, L] 系数，只读严格下三角，对角线按 1 处理。
// @param rhs [bn, L, n] 右端项。
// @param out [bn, L, n] 目的地；从不与输入重叠才会被写坏。
// @param dtype 操作数 dtype 码（两个操作数与 ``out`` 相同）。
// @param bn 独立系统的批大小。
// @param L 阶数。
// @param n 右端项列数。
// @param ldb ``lower`` 的批步长（元素）。
// @param ldr ``rhs``/``out`` 的批步长（元素）。
// @param stream 目标流；空指针表示默认流。
// @param err 错误缓冲，可为空。
// @param errlen 错误缓冲字节数。
// @return ``0`` 成功。
extern "C" int coffin_triangular_solve(const void* lower, const void* rhs, void* out,
                                       int dtype, int bn, int L, int n, long long ldb,
                                       long long ldr, void* stream, char* err,
                                       int errlen) {
    const cx::Err e{err, errlen};
    if (!lower || !rhs || !out) return cx::fail(e, "operands must be non-null");
    if (bn <= 0 || L <= 0 || n <= 0) {
        return cx::fail(e, "bn/L/n must be positive, got %d/%d/%d", bn, L, n);
    }
    if (dtype != cx::kBF16 && dtype != cx::kF32) {
        return cx::fail(e, "operands must be bf16 or fp32, got dtype code %d", dtype);
    }
    if (ldb <= 0 || ldr <= 0) {
        return cx::fail(e, "ldb/ldr must be positive, got %lld/%lld", ldb, ldr);
    }
    if (ldb < (long long)L * L) {
        return cx::fail(e, "lower batch stride %lld is narrower than [%d, %d]", ldb, L, L);
    }
    if (ldr < (long long)L * n) {
        return cx::fail(e, "rhs batch stride %lld is narrower than [%d, %d]", ldr, L, n);
    }
    const cx::Tensor a = cx::Tensor::matrix(const_cast<void*>(lower), L, L, L,
                                            (cx::DType)dtype);
    const cx::Tensor b = cx::Tensor::matrix(const_cast<void*>(rhs), L, n, n,
                                            (cx::DType)dtype);
    const cx::Tensor y = cx::Tensor::matrix(out, L, n, n, (cx::DType)dtype);
    return dispatch_solve(a, b, y, bn, L, n, (long)ldb, (long)ldr, stream, e);
}
