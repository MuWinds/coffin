// @file moe_grouped_i8_expert_abi.cu
// @brief 分组 int8「一个专家一趟」核的 C ABI 出口：``coffin_moe_grouped_i8_expert``。
//
// Module: coffin/backends/corex/kernels/abi。设备代码来自
// ``../moe_grouped_i8_expert_device.inc``——与 ``../moe_grouped_i8.cu`` 是同一份内核与
// 启动器。给 MoE 预填充用：每专家十几行时逐行核会把权重按行重读十几倍。
//
// 与 pybind 入口的差别只有一处：目的地 dtype **由调用方给**（``out_code``），因为
// ABI 面拿的是裸指针，没有 ``scalar_type()`` 可读。
//
// Constraints:
//   * 签名与 ``cxrt/moe.py`` 的 ``argtypes`` 必须逐字一致：多一个参数就整体错位。
//   * ``K % 512 == 0``：16 字节 ``int4`` 读要求列起点与步长按 16 对齐。
//   * ``offsets`` 是 ``[E + 1]`` 的 int32 前缀和，最后一格是有效行数。

#include "cx_tensor.h"

#include <cstdint>

#include "../moe_grouped_i8_expert_device.inc"

// @brief 分组 int8 投影（预填充形态）：同一专家的全部行共用一次权重读取。
//
// @param a [tokens, K] int8 激活。
// @param xs [tokens] fp32 每行激活 scale。
// @param w [E, N, K] int8 nk 专家栈。
// @param ws [E, N] fp32 每（专家，列）权重 scale。
// @param offsets [E + 1] int32 前缀和。
// @param out [tokens, N]，dtype 由 ``out_code`` 给。
// @param out_code 目的地 dtype 码（1 = bf16，0 = fp32）。
// @param tokens, k, cols, experts 几何。
// @param stream 目标流；空指针表示默认流。
// @param err 错误缓冲，可为空。
// @param errlen 错误缓冲字节数。
// @return ``0`` 成功。
extern "C" int coffin_moe_grouped_i8_expert(const void* a, const void* xs, const void* w,
                                            const void* ws, const void* offsets, void* out,
                                            int out_code, int tokens, int k, int cols, int experts,
                                            void* stream, char* err, int errlen) {
    const cx::Err e{err, errlen};
    if (!a || !xs || !w || !ws || !offsets || !out) {
        return cx::fail(e, "activation, scales, weights, offsets and destination must be non-null");
    }
    if (tokens <= 0 || k <= 0 || cols <= 0 || experts <= 0) {
        return cx::fail(e, "tokens/k/cols/experts must be positive, got %d/%d/%d/%d", tokens, k,
                        cols, experts);
    }
    if (k % I8_EXP_STEPK != 0) {
        return cx::fail(e, "prefilling grouped kernel needs k %% %d == 0, got k = %d",
                        I8_EXP_STEPK, k);
    }
    if (out_code != 0 && out_code != 1) {
        return cx::fail(e, "unknown destination dtype code %d", out_code);
    }
    launch_moe_grouped_i8_expert(reinterpret_cast<const int8_t*>(a),
                                 reinterpret_cast<const int8_t*>(w),
                                 reinterpret_cast<const float*>(ws),
                                 reinterpret_cast<const float*>(xs),
                                 reinterpret_cast<const int*>(offsets), out, out_code, tokens, k,
                                 cols, experts, reinterpret_cast<cudaStream_t>(stream));
    return cx::finish(e);
}
