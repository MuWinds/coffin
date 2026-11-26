// @file w8a8_dp4a_abi.cu
// @brief nk 布局 dp4a int8 GEMM 的 C ABI 出口：``coffin_w8a8_dp4a_nk``，无 torch。
//
// Module: coffin/backends/corex/kernels/abi。设备代码来自
// ``../w8a8_dp4a_device.inc``——与 ``../w8a8_dp4a.cu`` 是同一份内核与启动器。
//
// 与 pybind 入口的差别只有一处：融合输出模式**由调用方显式给**（``mode``），而不是
// 从 ``out`` 的 dtype 推。ABI 面拿的是裸指针，没有 dtype 可读。
//
// Constraints:
//   * 签名与 ``cxrt/w8a8.py`` 的 ``argtypes`` 必须逐字一致：多一个参数就整体错位。
//   * ``mode == 0``（``kOutNone``）时写 ``c``；``1`` / ``2`` 写 ``out``（bf16 / fp32），
//     且 ``out`` 必须非空、行距 ``ld_out >= N``。
//   * ``K % 4 == 0``：``__dp4a`` 一次归约 4 个相邻 k，尾部不齐会跨列读。

#include "cx_tensor.h"

#include <cstdint>

#include "../w8a8_dp4a_device.inc"

// @brief ``[N, K]`` int8 × ``[M, K]`` int8 → ``[M, N]`` int32（可选融合反量化）。
//
// @param a [M, K] int8 激活。
// @param wt [N, K] int8 权重，K 连续。
// @param c [M, N] int32 累加缓冲。
// @param out 融合输出；``mode == 0`` 时可为空。
// @param xs [M] fp32 每行激活 scale。
// @param ws [N] fp32 每列权重 scale。
// @param mode 0 = 只写 ``c``，1 = bf16 输出，2 = fp32 输出。
// @param ld_out 融合输出的行间距。
// @param rows, k, cols 几何。
// @param stream 目标流；空指针表示默认流。
// @param err 错误缓冲，可为空。
// @param errlen 错误缓冲字节数。
// @return ``0`` 成功。
extern "C" int coffin_w8a8_dp4a_nk(const void* a, const void* wt, void* c, void* out,
                                   const void* xs, const void* ws, int mode, int ld_out, int rows,
                                   int k, int cols, void* stream, char* err, int errlen) {
    const cx::Err e{err, errlen};
    if (!a || !wt || !c || !xs || !ws) {
        return cx::fail(e, "operands, accumulator and scales must be non-null");
    }
    if (rows <= 0 || k <= 0 || cols <= 0) {
        return cx::fail(e, "rows/k/cols must be positive, got %d/%d/%d", rows, k, cols);
    }
    if (k % 4 != 0) {
        return cx::fail(e, "the dp4a path needs k %% 4 == 0, got k = %d", k);
    }
    if (mode < kOutNone || mode > kOutF32) {
        return cx::fail(e, "unknown output mode %d", mode);
    }
    if (mode != kOutNone && (out == nullptr || ld_out < cols)) {
        return cx::fail(e, "a fused output needs a non-null pointer and ld_out >= cols (%d), got %d",
                        cols, ld_out);
    }
    launch_w8a8_dp4a_nk(mode, reinterpret_cast<const int8_t*>(a),
                        reinterpret_cast<const int8_t*>(wt), reinterpret_cast<int32_t*>(c),
                        reinterpret_cast<const float*>(xs), reinterpret_cast<const float*>(ws),
                        mode == kOutNone ? nullptr : out, rows, k, cols,
                        mode == kOutNone ? cols : ld_out, reinterpret_cast<cudaStream_t>(stream));
    return cx::finish(e);
}
