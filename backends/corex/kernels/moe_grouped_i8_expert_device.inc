// @file moe_grouped_i8_expert_device.inc
// @brief 分组 int8 的「一个专家一趟」核：常量、内核、启动器。
//
// Module: coffin/backends/corex/kernels。被两条前端各包含一次：``moe_grouped_i8.cu``
// （torch 扩展的 pybind 入口）与 ``abi/moe_grouped_i8_expert_abi.cu``（``extern "C"``
// 出口）。同一份设备代码。
//
// 为什么需要这条核、以及为什么一个 warp 认领一列，见 ``moe_grouped_i8.cu`` 里那段注释：
// 预填充时每专家十几行，逐行核会把权重按行重读十几倍（805 MB -> 12.9 GB）。
//
// Constraints:
//   * ``K % I8_EXP_STEPK == 0``：16 字节的 ``int4`` 读要求列起点与步长按 16 对齐。
//   * ``offsets`` 是 **`[E + 1]`** 的 int32 前缀和；``offsets[E]`` 是有效行数。
//   * 专家区间外的尾行不会被写：``out`` 的尾行保持调用方写下的值。

#ifndef COFFIN_MOE_GROUPED_I8_EXPERT_DEVICE_INC
#define COFFIN_MOE_GROUPED_I8_EXPERT_DEVICE_INC

#include "corex_scalar.h"

#include <cstdint>
#include <cuda_runtime.h>

namespace {

constexpr int I8_EXP_THREADS = 128;
constexpr int I8_EXP_COLS = 8;

//: 一趟覆盖的行数。一个 512 token 的预填充块摊到 256 个专家正好每专家 16 行，所以一趟
//: 取 16 行才让每层每专家的权重真的只读一遍；取 8 行时 16 行被拆成两趟，权重读两遍
//: （gate_up 实测 12.0 ms / 45 GB/s，而同一份 536 MB 整份读只要 3.3 ms）。
constexpr int I8_EXP_MT = 16;

//: 一个 warp 认领的列数。2 列让每通道的累加器压到 32 个 int32：4 列时要 64 个，
//: 寄存器就顶到 96 以上，每 SM 只剩两个块。（列少则同一份激活要被更多列块重读，
//: 但那份重读落在 L1/L2 里——同一专家的 32 KB 激活块在这张卡上会被十几条 warp 同时读。）
constexpr int I8_EXP_WCOL = 2;

//: 一步的 K 跨度 = 32 条通道 x 16 字节。
constexpr int I8_EXP_STEPK = 512;

template <typename XT>
__global__ void moe_grouped_i8_expert_nk_kernel(const int8_t* __restrict__ a,
                                                const int8_t* __restrict__ wn,
                                                const float* __restrict__ ws,
                                                const float* __restrict__ xs,
                                                const int* __restrict__ offsets,
                                                XT* __restrict__ out,
                                                int K, int N) {
    const int expert = (int)blockIdx.y;
    const int r0 = offsets[expert];
    const int r1 = offsets[expert + 1];
    if (r0 >= r1) return;

    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int warps = (int)(blockDim.x >> 5);
    const int colbase = ((int)blockIdx.x * warps + warp) * I8_EXP_WCOL;
    if (colbase >= N) return;
    const int nc = min(I8_EXP_WCOL, N - colbase);

    const int8_t* ebase = wn + (size_t)expert * (size_t)N * (size_t)K;

    for (int rt = r0; rt < r1; rt += I8_EXP_MT) {
        const int rows = min(I8_EXP_MT, r1 - rt);
        int acc[I8_EXP_WCOL][I8_EXP_MT];
#pragma unroll
        for (int c = 0; c < I8_EXP_WCOL; ++c) {
#pragma unroll
            for (int m = 0; m < I8_EXP_MT; ++m) acc[c][m] = 0;
        }

        for (int kb = 0; kb < K; kb += I8_EXP_STEPK) {
            const int k0 = kb + lane * 16;
            int4 w[I8_EXP_WCOL];
#pragma unroll
            for (int c = 0; c < I8_EXP_WCOL; ++c) {
                if (c < nc) {
                    w[c] = *reinterpret_cast<const int4*>(
                        ebase + (size_t)(colbase + c) * K + k0);
                }
            }
            // 激活直接从全局读，不再经共享内存过一道：一趟 16 行要 32 KB 共享内存，
            // 那会把每 SM 的常驻块压到 1 个（8 条 warp），延迟就藏不住——实测同一条
            // 几何带共享内存只有 59 GB/s，而这份 16 行 x K 的激活块被同一专家的所有
            // 列块重读，落在 L1/L2 里，不占 HBM。
#pragma unroll
            for (int m = 0; m < I8_EXP_MT; ++m) {
                if (m < rows) {
                    const int4 av = *reinterpret_cast<const int4*>(
                        a + (size_t)(rt + m) * K + k0);
#pragma unroll
                    for (int c = 0; c < I8_EXP_WCOL; ++c) {
                        if (c < nc) {
                            acc[c][m] = __dp4a(w[c].x, av.x, acc[c][m]);
                            acc[c][m] = __dp4a(w[c].y, av.y, acc[c][m]);
                            acc[c][m] = __dp4a(w[c].z, av.z, acc[c][m]);
                            acc[c][m] = __dp4a(w[c].w, av.w, acc[c][m]);
                        }
                    }
                }
            }
        }

#pragma unroll
        for (int c = 0; c < I8_EXP_WCOL; ++c) {
            if (c >= nc) continue;
            const float wscale = ws[(size_t)expert * N + colbase + c];
#pragma unroll
            for (int m = 0; m < I8_EXP_MT; ++m) {
                if (m < rows) {
                    int v = acc[c][m];
#pragma unroll
                    for (int off = 16; off > 0; off >>= 1) {
                        v += __shfl_down_sync(0xffffffffu, v, off);
                    }
                    if (lane == 0) {
                        out[(size_t)(rt + m) * N + colbase + c] = CxScalar<XT>::from_float(
                            (float)v * xs[rt + m] * wscale);
                    }
                }
            }
        }
    }
}

}  // namespace

// @brief 启动「一个专家一趟」的分组核。
//
// @param a ``[tokens, K]`` int8 激活。
// @param wn ``[E, N, K]`` int8 nk 专家栈。
// @param ws ``[E, N]`` fp32 每（专家，列）权重 scale。
// @param xs ``[tokens]`` fp32 每行激活 scale。
// @param offsets ``[E + 1]`` int32 前缀和。
// @param out ``[tokens, N]``，dtype 由 ``out_code`` 给。
// @param out_code 目的地 dtype 码（``cx::kBF16`` = 1，``cx::kF32`` = 0）。
// @param tokens, K, N, experts 几何。
// @param stream 目标流；空指针表示默认流。
inline void launch_moe_grouped_i8_expert(const int8_t* a, const int8_t* wn, const float* ws,
                                         const float* xs, const int* offsets, void* out,
                                         int out_code, int tokens, int K, int N, int experts,
                                         cudaStream_t stream) {
    if (tokens <= 0 || K <= 0 || N <= 0 || experts <= 0) return;
    // 一个块 = 4 个 warp = 4 x WCOL 列；块小、不占共享内存，是为了让每 SM 常住的块数
    // 由寄存器而不是共享内存决定（同一条几何带 32 KB 共享内存时只剩 1 个块 / 8 条 warp）。
    const dim3 block(I8_EXP_THREADS);
    const dim3 grid((unsigned)((N + I8_EXP_COLS - 1) / I8_EXP_COLS), (unsigned)experts);
    if (out_code == 1) {
        moe_grouped_i8_expert_nk_kernel<uint16_t><<<grid, block, 0, stream>>>(
            a, wn, ws, xs, offsets, reinterpret_cast<uint16_t*>(out), K, N);
    } else {
        moe_grouped_i8_expert_nk_kernel<float><<<grid, block, 0, stream>>>(
            a, wn, ws, xs, offsets, reinterpret_cast<float*>(out), K, N);
    }
}

#endif  // COFFIN_MOE_GROUPED_I8_EXPERT_DEVICE_INC
