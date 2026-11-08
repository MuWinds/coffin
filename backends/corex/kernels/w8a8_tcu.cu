// @file w8a8_tcu.cu
// @brief 用宽指令 i8x8 的张量核 int8 GEMM，累加器留在寄存器里直接出 bf16。
//
// Module: coffin/backends/corex/kernels。绑定成 ``w8a8_gemm_tcu``，由
// ``backends/corex/__init__.py`` 的 ``_w8a8_rows`` 在宽形状上优先调用，
// 窄形状仍走库（``w8a8_gemm``）。
//
// 存在理由：厂商 libcublas 的 int8 路径只引用 ``i8x4``（4x4x4 = 64 MAC/指令），
// 而本卡还有一条 ``i8x8``（4x4x8 = 128 MAC/指令）。同一块卡上实测指令峰值
// i8x4 是 65.31 TINT8-OPS/s、i8x8 是 95.99，而库在 4B 的真实形状上只有
// 37.7-57.2。所以「库已接近上限」不成立，它被窄指令卡住了。
//
// Constraints:
//   * **权重与激活都按原生布局吃，一次转置都不做。** 厂商的 fragment 加载器要
//     「同 k 上 4 个相邻 MMA-M」和「同 MMA-N 上 4 个相邻 k」两种字；把两个操作数
//     互换（算 ``C^T = W^T @ A^T``）之后，权重 ``[K, N]`` 与激活 ``[M, K]``
//     恰好各是一种。代价只是输出变 ``[N, M]``，而 epilogue 反正要读遍整个输出。
//   * **不能用 ``wmma::mma_sync``**：它在 ``__BI__``（= ivcore10，本卡）分支里走
//     ``i8x4``，照抄厂商样例拿不到宽指令。这里直接调
//     ``__ivcorex_matrix_mad_i32x4_i8x8``，寄存器切片规则照抄
//     ``crt/iluvatar_mma.hpp`` 里 ``__MR__`` 的 ``mma_sync(fragment<accumulator,
//     64, 64, 32, ...>)`` 实现体。
//   * **不能用厂商的 64x64x32 ``wmma::load/store``**：那是 ``__MR__`` 专属，在本卡
//     上证到 lld 的 ``undefined symbol``。共享内存的散开表照抄 ``CoordToOffset<8>``，
//     fragment 加载自己写。
//   * **signed 与 unsigned 不能混。** 引擎激活是 [-127,127]，用 ``i8x8``；若哪天改成
//     [0,255] 要用 ``u8x8``，而两者零点不同（int8 的 -128 不等于 uint8 的 +128），
//     混用会静默算错而不是报错。
//   * 累加器是 int32，int8 x int8 的乘积精确落在里面。**epilogue 在寄存器里做**：
//     ``int32 -> fp32 乘两个 scale -> 一次窄化``，与 ``w8a8_epilogue`` 逐位相同，
//     但省掉整个 ``[M, N]`` int32 往返（库路径做不到，因为 ``cublasGemmEx`` 只把
//     int32 写回 global）。
//   * 每次启动都在 ``c10::cuda::getCurrentCUDAStream()`` 上。
//   * 全程不用 fp64：本设备对它返回 0。

#include "corex_i8.h"
#include "corex_ops.h"
#include "corex_scalar.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <climits>
#include <cstdint>

namespace {

using namespace nvcuda;

//: 块几何。BM 走 MMA-M（= 权重的 N 方向），BN 走 MMA-N（= 激活的 M 方向）。
//
// 这两个数不是调出来的而是算出来的：算术强度 ``2*BM*BN/(BM+BN)`` 要够大，
// 否则打满 96 TINT8-OPS/s 需要的带宽超过本卡 575 GB/s 的上限。128x128 只有
// 128 op/byte（需要 750 GB/s，物理上做不到），256x256 是 256 op/byte（需要
// 375 GB/s，留了 1.5 倍余量）。实测 128x128 在四个真实形状上都比 256x256 慢
// 30-40%，与这个估算的方向一致。
constexpr int BM = 256;
constexpr int BN = 256;
//: K 方向步长。32 的倍数（一个 warp tile 一次吃 32 个 k），64 让共享内存 64 KB。
constexpr int BK = 64;

//: 每个 warp 的输出 tile 恒为 64x64。
constexpr int WR = BM / 64;
constexpr int WC = BN / 64;
constexpr int NWARPS = WR * WC;
constexpr int NTHREADS = NWARPS * 64;

//: 共享内存每段的字节数，按「warp 列 x 16-k chunk x 1024」叠。
constexpr int SA = WR * (BK / 16) * 1024;
constexpr int SB = WC * (BK / 16) * 1024;

// @brief matrix_a（权重）的共享内存散开表。
//
// 照抄厂商 ``CoordToOffset<8, layout_t::mem_col_major>``：把 4 个相邻 m 打进一个字，
// 再按 16 字节粒度异或散开。散开不是为了好看——TCU 从共享内存取 fragment 时，
// 若 64 个 lane 的字都落在同一个 bank，读会被串行化。
//
// @param m warp 内的行号（MMA-M 方向）。
// @param k 16-k chunk 内的 k。
// @return 该元素的字节偏移。
__device__ __forceinline__ int off_a(int m, int k) {
    int const r = m >> 2;
    return (k >> 2) * 256 + (r & 3) * 64 + ((r >> 2) ^ (k >> 2)) * 16
           + (k & 3) * 4 + (m & 3);
}

// @brief matrix_b（激活）的共享内存散开表。
//
// 照抄 ``CoordToOffset<8, layout_t::mem_row_major>``。其中 ``& 255`` 不是笔误：
// 它才是让 ``(k>>2)*64`` 与 ``64*(n>>4)`` 不撞车的部分。
//
// @param k 16-k chunk 内的 k。
// @param n warp 内的列号（MMA-N 方向）。
// @return 该元素的字节偏移。
__device__ __forceinline__ int off_b(int k, int n) {
    int const i = n >> 4;
    int const c = n & 15;
    int const r = k >> 2;
    return i * 256 + ((r * 64 + c * 4 + (k & 3) + 64 * i) & 255);
}

// @brief 装 matrix_a 一个 64x16 chunk 的 4 个 word。
//
// @param a 输出，4 个 word。
// @param p 该 chunk 的共享内存起点。
__device__ __forceinline__ void ld_a(unsigned* a, const int8_t* p) {
    int const lane = __ivcorex_lane_id();
    int const col = lane & 15;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        int const row = (lane >> 4) * 4 + 16 * q;
        a[q] = *reinterpret_cast<const unsigned*>(p + off_a(row, col));
    }
}

// @brief 装 matrix_b 一个 16x64 chunk 的 4 个 word。
//
// @param b 输出，4 个 word。
// @param p 该 chunk 的共享内存起点。
__device__ __forceinline__ void ld_b(unsigned* b, const int8_t* p) {
    int const lane = __ivcorex_lane_id();
    int const row = (lane >> 4) * 4;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        int const col = (lane & 15) + 16 * q;
        b[q] = *reinterpret_cast<const unsigned*>(p + off_b(row, col));
    }
}

// @brief 用 16 条 i8x8 完成一个 64x64x32 的累加。
//
// 第 r 个行 tile 取 chunk0 的第 r 个 word 与 chunk1 的第 r 个 word 拼成 v2i32
// （K 方向一次 8 个）；列同理。这是厂商 ``__MR__`` 的 ``mma_sync`` 里的取法。
//
// @param a0 matrix_a 的 chunk0（k 0..15）。
// @param a1 matrix_a 的 chunk1（k 16..31）。
// @param b0 matrix_b 的 chunk0。
// @param b1 matrix_b 的 chunk1。
// @param acc 16 个 v4i32 的累加器，原地更新。
__device__ __forceinline__ void mma_tile(const unsigned* a0, const unsigned* a1,
                                         const unsigned* b0, const unsigned* b1,
                                         v4i32* acc) {
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        v2i32 a_tile;
        a_tile.x = (int)a0[r];
        a_tile.y = (int)a1[r];
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            v2i32 b_tile;
            b_tile.x = (int)b0[c];
            b_tile.y = (int)b1[c];
            acc[4 * r + c] = __ivcorex_matrix_mad_i32x4_i8x8(
                reinterpret_cast<const v8i8&>(a_tile),
                reinterpret_cast<const v8i8&>(b_tile), acc[4 * r + c]);
        }
    }
}

// @brief 把权重的一段从 ``[K, N]`` 搬进共享内存。
//
// 一个线程搬一个 uint32：``[K, N]`` 里固定 k 的 4 个相邻 n 连续，正好是加载器要的字，
// 所以是一次全局读对一次共享写，没有转置运算。
//
// @param sa 目的段起点。
// @param W ``[K, N]`` int8 权重。
// @param n0 本块在 N 方向的起点。
// @param k0 本块在 K 方向的起点。
// @param N 权重列数。
// @param K 权重行数。
// @param tid 本线程在线性任务表里的下标。
template <int TBK>
__device__ __forceinline__ void stage_w(int8_t* sa, const int8_t* __restrict__ W,
                                        int n0, int k0, int N, int K, int tid) {
    constexpr int J = BM / 4;
#pragma unroll
    for (int rep = 0; rep < (TBK * J + NTHREADS - 1) / NTHREADS; ++rep) {
        int const t = tid + rep * NTHREADS;
        if (t >= TBK * J) break;
        int const k = t / J;
        int const j = t % J;
        int const gk = k0 + k;
        int const gn4 = n0 + j * 4;
        unsigned v = 0;
        // 尾块（N 不是 BM 的整数倍）补零：补零不改变结果，越界读才会。
        if (gk < K && gn4 + 3 < N)
            v = *reinterpret_cast<const unsigned*>(W + (size_t)gk * N + gn4);
        int const w = j >> 4;
        int const n4 = (j & 15) * 4;
        *reinterpret_cast<unsigned*>(sa + w * (TBK / 16) * 1024 + (k >> 4) * 1024
                                     + off_a(n4, k & 15)) = v;
    }
}

// @brief 把激活的一段从 ``[M, K]`` 搬进共享内存。
//
// matrix_b 的字是「同一个 MMA-N 上 4 个相邻 k」，而 ``[M, K]`` 里固定 m 的 4 个
// 相邻 k 连续 —— 同样一次全局读对一次共享写。线性化用 ``t = m * (BK/4) + j``，
// 因为 B 侧要的是 k 相邻，只有这样才能让一个 warp 的 32 个 lane 落在连续地址上。
//
// @param sb 目的段起点。
// @param Ac ``[M, K]`` int8 激活。
// @param m0 本块在 M 方向的起点。
// @param k0 本块在 K 方向的起点。
// @param M 激活行数。
// @param K 激活行宽。
// @param tid 本线程在线性任务表里的下标。
template <int TBK>
__device__ __forceinline__ void stage_act(int8_t* sb, const int8_t* __restrict__ Ac,
                                          int m0, int k0, int M, int K, int tid) {
    constexpr int J = TBK / 4;
#pragma unroll
    for (int rep = 0; rep < (BN * J + NTHREADS - 1) / NTHREADS; ++rep) {
        int const t = tid + rep * NTHREADS;
        if (t >= BN * J) break;
        int const m = t / J;
        int const j = t % J;
        int const gk4 = k0 + j * 4;
        unsigned v = 0;
        // 两个边界都要查：M 不是 BN 的整数倍时尾部块会读到激活之外。
        if (m0 + m < M && gk4 + 3 < K)
            v = *reinterpret_cast<const unsigned*>(Ac + (size_t)(m0 + m) * K + gk4);
        int const w = m >> 6;
        int const kk = j * 4;
        *reinterpret_cast<unsigned*>(sb + w * (TBK / 16) * 1024 + (kk >> 4) * 1024
                                     + off_b(kk & 15, m & 63)) = v;
    }
}

// @brief 把 16 个 v4i32 累加器乘两个 scale、窄化一次、写进 ``[M, N]`` 输出。
//
// 累加器的 4 个分量落在同一个 16x16 tile 的 4 个相隔 4 行的位置，这是
// ``__imma_st_row_b32`` + ``CoordToOffset<32, mem_row_major>`` 给出的规格。
//
// **epilogue 在这里而不是单独一个内核**，这是自写 GEMM 才有的选择：库的
// ``cublasGemmEx`` 只把 int32 写回 global，没有任何挂点，于是那条路必须付
// 一次 ``[M, N]`` int32 的写 + 读。这里累加器就在寄存器里，直接乘完写 bf16。
//
// 舍入必须与 ``w8a8_epilogue`` 一致：两个 scale 在同一个 fp32 表达式里乘完
// 再窄化一次。分两次窄化会多一次舍入，而结果要穿过 32 层。
//
// @param acc 16 个 v4i32 累加器。
// @param out ``[M, N]`` 目的地，行距 ``stride``。
// @param xs ``[M]`` 每 token 的激活 scale。
// @param ws ``[N]`` 每输出列的权重 scale。
// @param m_base 本 warp tile 在 M 方向的起点。
// @param n_base 本 warp tile 在 N 方向的起点。
// @param M 行数。
// @param N 列数。
// @param stride ``out`` 的行间距。沿 N 切块时它大于 ``N``（``lm_head`` 的
//        ``out[:, start:stop]`` 就是这种视图），所以写回必须按 ``stride`` 走，
//        否则第二行会盖到第一行的尾部。
//
// **两个轴的归属容易写反，而写反的症状是几乎全错、看起来像「内核彻底不对」。**
//
// 实测过的判据（单位脉冲激活）：库的输出里
// ``out[m, n] = wt[m % K, n]``，**每一行都相同**；而错误的版本是「每一列都相同」。
// 一句话：**MMA-M（累加器的 ``rt``）产出的是输出的 M 方向，MMA-N（``ct``）产出 N**。
// 这里的 M/N 是**互换后的 MMA 视角**——MMA-M 是权重的 N，MMA-N 是激活的 M，所以
// 落到最终 ``[M, N]`` 输出上，``rt`` 管 M、``ct`` 管 N。
//
// 累加器那 4 个分量沿 MMA-M 每 4 个一组，所以 ``m`` 要 ``+4*q``。
template <typename XT>
__device__ __forceinline__ void store_epi(const v4i32* acc, XT* __restrict__ out,
                                          const float* __restrict__ xs,
                                          const float* __restrict__ ws,
                                          int m_base, int n_base, int M, int N,
                                          int stride) {
    int const lane = __ivcorex_lane_id();
    int const base_col = lane & 15;      // MMA-N（= 激活的 M）内的列
    int const base_row = lane >> 4;      // MMA-M（= 权重的 N）内的行
#pragma unroll
    for (int rt = 0; rt < 4; ++rt) {
#pragma unroll
        for (int ct = 0; ct < 4; ++ct) {
            v4i32 v = acc[4 * rt + ct];
            int const m = m_base + 16 * ct + base_col;
            int const n_row = n_base + 16 * rt + base_row;
            int const vals[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                int const n = n_row + 4 * q;
                if (m < M && n < N) {
                    out[(size_t)m * stride + n] = CxScalar<XT>::from_float(
                        (float)vals[q] * xs[m] * ws[n]);
                }
            }
        }
    }
}

// @brief 一次启动算 ``out[M, N] = W^T 作用在激活上``，epilogue 融合。
//
// 输出是 ``[M, N]`` 且按引擎的约定写，但内部算的是 ``C^T = W^T @ A^T``：MMA 的 M
// 是权重的 N，MMA 的 N 是激活的 M。两个操作数因此都停在原生布局里。
//
// @param W ``[K, N]`` int8 权重，连续。
// @param Ac ``[M, K]`` int8 激活，连续。
// @param out ``[M, N]`` 目的地。
// @param xs ``[M]`` 激活 scale。
// @param ws ``[N]`` 权重 scale。
// @param M 激活行数。
// @param N 权重列数。
// @param K 内维。
template <typename XT>
__global__ void gemm_tcu_kernel(const int8_t* __restrict__ W, const int8_t* __restrict__ Ac,
                                XT* __restrict__ out, const float* __restrict__ xs,
                                const float* __restrict__ ws, int M, int N, int K, int stride) {
    extern __shared__ int8_t smem[];
    int8_t* sa = smem;
    int8_t* sb = smem + 2 * SA;

    int const tid = threadIdx.x;
    int const warp = tid >> 6;
    int const wm = warp / WC;
    int const wn = warp % WC;
    // grid.x 走 N（MMA-M）、grid.y 走 M（MMA-N）。
    int const n0 = blockIdx.x * BM;
    int const m0 = blockIdx.y * BN;

    v4i32 acc[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        acc[i].x = 0; acc[i].y = 0; acc[i].z = 0; acc[i].w = 0;
    }

    int const nk = (K + BK - 1) / BK;
    stage_w<BK>(sa, W, n0, 0, N, K, tid);
    stage_act<BK>(sb, Ac, m0, 0, M, K, tid);
    __syncthreads();

    for (int kb = 0; kb < nk; ++kb) {
        int const cur = kb & 1;
        // 下一段的全局读与当前段的 mma 重叠。这条重叠是双缓冲唯一真正买到的东西：
        // 只交换「搬」和「算」的顺序而不拆开 LDG/STS，warp 会卡在 STS 上等全局内存，
        // 那段 mma 就完全没盖住任何延迟（实测 stage + mma ≈ whole 就是这个症状）。
        if (kb + 1 < nk) {
            stage_w<BK>(sa + (cur ^ 1) * SA, W, n0, (kb + 1) * BK, N, K, tid);
            stage_act<BK>(sb + (cur ^ 1) * SB, Ac, m0, (kb + 1) * BK, M, K, tid);
        }
        const int8_t* wa = sa + cur * SA + wm * (BK / 16) * 1024;
        const int8_t* wb = sb + cur * SB + wn * (BK / 16) * 1024;
        unsigned a0[4], a1[4], b0[4], b1[4];
#pragma unroll
        for (int t = 0; t < BK / 32; ++t) {
            ld_a(a0, wa + t * 2048);
            ld_a(a1, wa + t * 2048 + 1024);
            ld_b(b0, wb + t * 2048);
            ld_b(b1, wb + t * 2048 + 1024);
            mma_tile(a0, a1, b0, b1, acc);
        }
        __syncthreads();
    }

    // 一个 warp 的输出 tile 在 (n0 + 64*wm, m0 + 64*wn)：MMA-M 走 N，MMA-N 走 M，
    // 传到 store_epi 时按 (m, n) 的顺序给参数。
    store_epi<XT>(acc, out, xs, ws, m0 + 64 * wn, n0 + 64 * wm, M, N, stride);
}

// W4 每组需同时持有 int32 点积和 fp32 跨组和；缩小 tile 限制寄存器压力。
constexpr int W4_TILE = 128;
constexpr int W4_THREADS = 256;
constexpr int W4_SHARED = 2 * (BK / 16) * 1024;

// @brief 一次读取四列 packed 字节，分别把低、高 nibble 放进相邻 K 行。
//
// @tparam KB 本段的 k 步长（64 为单位置，32 为双缓冲档）。
template <int KB>
__device__ __forceinline__ void stage_w4(int8_t* sa, const uint8_t* W,
                                         int n0, int k0, int N, int K) {
    for (int t = threadIdx.x; t < KB / 2 * (W4_TILE / 4); t += blockDim.x) {
        const int k = (t / (W4_TILE / 4)) * 2;
        const int j = t % (W4_TILE / 4);
        const int n = n0 + j * 4;
        unsigned bytes = 0x88888888u;
        if (k0 + k < K && n < N) {
            if ((N & 3) == 0 && n + 3 < N) {
                bytes = *reinterpret_cast<const unsigned*>(W + (size_t)((k0 + k) / 2) * N + n);
            } else {
                for (int c = 0; c < 4; ++c) {
                    if (n + c < N) {
                        const unsigned v = W[(size_t)((k0 + k) / 2) * N + n + c];
                        bytes = (bytes & ~(255u << (8 * c))) | (v << (8 * c));
                    }
                }
            }
        }
        const int warp_row = j >> 4;
        const int n4 = (j & 15) * 4;
        for (int half = 0; half < 2; ++half) {
            unsigned word = 0;
            for (int c = 0; c < 4; ++c) {
                const int q = (int)((bytes >> (8 * c + half * 4)) & 15) - 8;
                word |= ((unsigned)q & 255) << (8 * c);
            }
            const int kk = k + half;
            *reinterpret_cast<unsigned*>(sa + warp_row * (KB / 16) * 1024
                + (kk >> 4) * 1024 + off_a(n4, kk & 15)) = word;
        }
    }
}

// @brief 激活只暂存本块的 ``MT`` 行，不生成全矩阵解包副本。
//
// @tparam MT 本块在 M 方向的 tile 行数，只能是 64 的整数倍（warp 输出 tile 的 M 恒为 64）。
template <int MT, int KB>
__device__ __forceinline__ void stage_w4_act(int8_t* sb, const int8_t* A,
                                             int m0, int k0, int M, int K) {
    for (int t = threadIdx.x; t < MT * (KB / 4); t += blockDim.x) {
        const int m = t / (KB / 4);
        const int k = (t % (KB / 4)) * 4;
        unsigned word = 0;
        if (m0 + m < M && k0 + k + 3 < K)
            word = *reinterpret_cast<const unsigned*>(A + (size_t)(m0 + m) * K + k0 + k);
        *reinterpret_cast<unsigned*>(sb + (m >> 6) * (KB / 16) * 1024
            + (k >> 4) * 1024 + off_b(k & 15, m & 63)) = word;
    }
}

// @brief 分组 W4A8 张量核 GEMM 内核，M tile 与 k 步长都是模板参数。
//
// @tparam MT 本块在 M 方向的 tile 行数（64 / 128 / 256）。它只改「块盖多少行」，
//            不改每个 warp 的输出 tile（恒为 64x64），所以每 lane 的累加器数不变；
//            改它换的是**每块的行数与权重读次数**，不是寄存器压力。
// @tparam KB 本段的 k 步长。``KB == BK`` 是单缓冲基线；``KB == 32`` 时自动走双缓冲
//            （两段 32 的共享内存合计等于单段 64，占用率不变）。
template <typename XT, int MT, int KB>
__global__ void w4_gemm_tcu(const uint8_t* W, const int8_t* A, const float* xs,
                             const float* ws, XT* out, int M, int N, int K, int gs) {
    //: 单段权重字节数；双缓冲时有两段。
    constexpr int kSegW = 2 * (KB / 16) * 1024;
    //: 单段激活字节数（MT/64 个 64 行小段）。
    constexpr int kSegA = (MT / 64) * (KB / 16) * 1024;
    //: KB != BK 即双缓冲档。
    constexpr bool kDB = (KB != BK);
    constexpr int kStages = kDB ? 2 : 1;
    extern __shared__ int8_t shared[];
    int8_t* sa = shared;
    int8_t* sb = shared + kStages * kSegW;
    //: M 方向铺几个 warp。MT=128 时 2 个（4 warp、256 线程），MT=64 时 1 个（2 warp、128 线程）。
    constexpr int kWarpsM = MT / 64;
    const int warp = threadIdx.x >> 6;
    const int wm = warp / kWarpsM, wn = warp % kWarpsM;
    const int n0 = blockIdx.x * W4_TILE, m0 = blockIdx.y * MT;
    const int lane = __ivcorex_lane_id();
    const int groups = (K + gs - 1) / gs;
    //: 一个组跨几个 KB 段；``gs`` 是 64 的倍数，KB 取 64 或 32，所以整除。
    const int per_group = gs / KB;

    float total[64] = {};
    v4i32 acc[16];
    auto reset_acc = [&]() {
#pragma unroll
        for (int i = 0; i < 16; ++i) acc[i] = {0, 0, 0, 0};
    };
    // 一组的 int32 累加收口：乘该组 scale 后加进 fp32 跨组和。两个分支共用同一段代码，
    // 否则「双缓冲」与「单缓冲」两条路的算术就不再逐位可比。
    auto flush = [&](int g) {
#pragma unroll
        for (int rt = 0; rt < 4; ++rt) {
#pragma unroll
            for (int ct = 0; ct < 4; ++ct) {
                const v4i32 v = acc[4 * rt + ct];
                const int vals[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int n = n0 + 64 * wm + 16 * rt + (lane >> 4) + 4 * q;
                    if (n < N) total[(4 * rt + ct) * 4 + q] +=
                        (float)vals[q] * ws[(size_t)g * N + n];
                }
            }
        }
    };
    reset_acc();

    if (kDB) {
        // 双缓冲：k 方向拍平成一维 chunk 序列，第 c+1 段的全局读与第 c 段的 mma 重叠；
        // 每段只有一次 __syncthreads（单缓冲是两次）。
        const int nchunks = groups * per_group;
        stage_w4<KB>(sa, W, n0, 0, N, K);
        stage_w4_act<MT, KB>(sb, A, m0, 0, M, K);
        __syncthreads();
        int cur = 0, group = 0;
        for (int c = 0; c < nchunks; ++c) {
            const int g = c / per_group;
            if (g != group) {
                flush(group);
                reset_acc();
                group = g;
            }
            const int next = c + 1;
            if (next < nchunks) {
                const int k_next = (next / per_group) * gs + (next % per_group) * KB;
                stage_w4<KB>(sa + (cur ^ 1) * kSegW, W, n0, k_next, N, K);
                stage_w4_act<MT, KB>(sb + (cur ^ 1) * kSegA, A, m0, k_next, M, K);
            }
            const int8_t* wa = sa + cur * kSegW + wm * (KB / 16) * 1024;
            const int8_t* wb = sb + cur * kSegA + wn * (KB / 16) * 1024;
            unsigned a0[4], a1[4], b0[4], b1[4];
#pragma unroll
            for (int t = 0; t < KB / 32; ++t) {
                ld_a(a0, wa + t * 2048); ld_a(a1, wa + t * 2048 + 1024);
                ld_b(b0, wb + t * 2048); ld_b(b1, wb + t * 2048 + 1024);
                mma_tile(a0, a1, b0, b1, acc);
            }
            // 一次同步同时保证：下一轮不会覆盖别人还在读的那段，且下一轮数据已就绪。
            __syncthreads();
            cur ^= 1;
        }
    } else {
        for (int g = 0; g < groups; ++g) {
            for (int k0 = g * gs; k0 < min(K, (g + 1) * gs); k0 += KB) {
                stage_w4<KB>(sa, W, n0, k0, N, K);
                stage_w4_act<MT, KB>(sb, A, m0, k0, M, K);
                __syncthreads();
                const int8_t* wa = sa + wm * (KB / 16) * 1024;
                const int8_t* wb = sb + wn * (KB / 16) * 1024;
                unsigned a0[4], a1[4], b0[4], b1[4];
#pragma unroll
                for (int t = 0; t < KB / 32; ++t) {
                    ld_a(a0, wa + t * 2048); ld_a(a1, wa + t * 2048 + 1024);
                    ld_b(b0, wb + t * 2048); ld_b(b1, wb + t * 2048 + 1024);
                    mma_tile(a0, a1, b0, b1, acc);
                }
                __syncthreads();
            }
            if (g + 1 < groups) {
                flush(g);
                reset_acc();
            }
        }
    }
    flush(groups - 1);

#pragma unroll
    for (int rt = 0; rt < 4; ++rt) {
#pragma unroll
        for (int ct = 0; ct < 4; ++ct) {
            const int m = m0 + 64 * wn + 16 * ct + (lane & 15);
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int n = n0 + 64 * wm + 16 * rt + (lane >> 4) + 4 * q;
                if (m < M && n < N) out[(size_t)m * N + n] =
                    CxScalar<XT>::from_float(total[(4 * rt + ct) * 4 + q] * xs[m]);
            }
        }
    }
}

}  // namespace

// @brief 分组 MoE 的大 M 形态：按专家分段各做一次 W4A8 张量核 GEMM。
//
// ## 为什么需要第二个 MoE 内核
//
// ``moe_grouped.cu`` 是「一列一线程、逐行串行」的 GEMV 结构，对 decode 合适：那时
// 每专家只有 1-2 行，权重读一遍就服务完了。但 **prefill 不是这个形状**。
//
// 作业 ``q4prebase3``（``seq=2048``，7.18 s）的账本：
//
//   moe_grouped_kernel   2178 ms / 96 次 = 22.7 ms/次   <- 30.4%，第一项
//   w4_gemm_tcu          1173 ms / 509 次 = 2.30 ms/次   <- 同类工作的 GEMM 路径
//
// ``seq=2048`` 时 ``tokens = 2048 x top_k(10) = 20480`` 行散在 128 个本地专家上，
// 每专家约 **160 行**。GEMV 结构下每行都要串行走完整个 ``K=2560``，权重被按段重读；
// 同样这 160 行按段喂给张量核就是标准 ``[160, 2560] x [2560, 1024]``。
//
// ## 为什么按专家逐个启动
//
// 每段的行数是数据相关的（``offsets`` 决定），而张量核 tile 是 ``W4_TILE``，一块不能
// 跨专家取激活与权重——跨了就要在 tile 内按行判断归属，那会把共享内存的装载逻辑
// 整个重写。逐段启动的代价是最多 ``E`` 次启动，在 prefill 尺度上（单次 GEMM 上百
// 微秒）可忽略；decode 走 ``moe_grouped``，不受这条影响。
//
// @param a         已按专家号排序的 int8 激活，``[tokens, K]`` 连续。
// @param xs        逐 token 激活 scale，``[tokens]`` fp32。
// @param w         专家栈 packed W4，``[E, ceil(K/2), N]`` 连续 uint8。
// @param ws        专家栈 scale，``[E, ceil(K/group), N]`` 连续 fp32。
// @param offsets   ``[E+1]`` 前缀和，int32；**本函数在主机侧读它**。
// @param out       ``[tokens, N]`` 连续输出，调用方负责先清零。
// @param group_size K 轴权重分组大小，必须是 64 的倍数。
// @return out。
// @throws c10::Error 布局、设备、dtype、形状或对齐不自洽。
void moe_grouped_gemm(torch::Tensor a, torch::Tensor xs, torch::Tensor w,
                      torch::Tensor ws, torch::Tensor offsets,
                      torch::Tensor out, int64_t group_size) {
    TORCH_CHECK(a.is_cuda() && a.dim() == 2 && a.scalar_type() == torch::kChar
                && a.is_contiguous(),
                "activation must be contiguous CUDA int8 [tokens, K]");
    TORCH_CHECK(xs.is_cuda() && xs.dim() == 1 && xs.scalar_type() == torch::kFloat
                && xs.is_contiguous() && xs.size(0) == a.size(0),
                "activation scale must be contiguous fp32 [tokens]");
    TORCH_CHECK(w.is_cuda() && w.dim() == 3 && w.scalar_type() == torch::kByte
                && w.is_contiguous(),
                "weight must be contiguous CUDA uint8 [E, K/2, N]");
    TORCH_CHECK(ws.is_cuda() && ws.dim() == 3 && ws.scalar_type() == torch::kFloat
                && ws.is_contiguous(),
                "weight scale must be contiguous fp32 [E, groups, N]");
    TORCH_CHECK(offsets.is_cuda() && offsets.dim() == 1
                && offsets.scalar_type() == torch::kInt && offsets.is_contiguous(),
                "offsets must be contiguous CUDA int32 [E+1]");
    TORCH_CHECK(out.is_cuda() && out.dim() == 2 && out.is_contiguous()
                && out.size(0) == a.size(0) && out.size(1) == ws.size(2),
                "destination must be contiguous CUDA [tokens, N]");

    const int tokens = (int)a.size(0);
    const int K = (int)a.size(1);
    const int N = (int)out.size(1);
    const int experts = (int)w.size(0);
    const int groups = (K + (int)group_size - 1) / (int)group_size;
    TORCH_CHECK(K % 4 == 0, "K must align to four for the packed nibble path");
    TORCH_CHECK(group_size > 0 && group_size % 64 == 0,
                "group_size must be a multiple of 64 for the tensor-core path");
    TORCH_CHECK((int)w.size(2) == N && (int)w.size(1) * 2 >= K,
                "weight shape mismatch");
    TORCH_CHECK((int)ws.size(0) == experts && (int)ws.size(1) == groups,
                "weight scale must be [E, ceil(K/group), N]");
    TORCH_CHECK((int)offsets.size(0) == experts + 1, "offsets must have E+1 entries");
    TORCH_CHECK(out.scalar_type() == torch::kFloat
                || out.scalar_type() == torch::kBFloat16,
                "destination must be bf16 or fp32");
    if (tokens == 0 || experts == 0) return;

    // 读 offsets 到主机：只为了决定给哪些专家启动。这是本内核唯一的主机同步，
    // 理由见上面——段内 GEMM 是百微秒级，一次同步换来跳过空段是划算的。
    const auto bounds = offsets.to(torch::kCPU);
    const int* bptr = bounds.data_ptr<int>();

    const c10::cuda::CUDAGuard guard(a.device());
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    // 每个专家的权重与 scale 在栈里是连续的一段，按专家下标直接算偏移。
    const int64_t w_stride = (int64_t)((K + 1) / 2) * N;
    const int64_t s_stride = (int64_t)groups * N;

    for (int e = 0; e < experts; ++e) {
        const int begin = bptr[e];
        const int end = bptr[e + 1];
        const int rows = end - begin;
        if (rows <= 0) continue;
        const int8_t* act = a.data_ptr<int8_t>() + (int64_t)begin * K;
        const float* ascale = xs.data_ptr<float>() + begin;
        const uint8_t* wexp = w.data_ptr<uint8_t>() + (int64_t)e * w_stride;
        const float* sexp = ws.data_ptr<float>() + (int64_t)e * s_stride;
        const dim3 grid((N + W4_TILE - 1) / W4_TILE,
                        (rows + W4_TILE - 1) / W4_TILE);
        if (out.scalar_type() == torch::kFloat) {
            float* dst = out.data_ptr<float>() + (int64_t)begin * N;
            w4_gemm_tcu<float, W4_TILE, BK><<<grid, W4_THREADS, 2 * W4_SHARED, stream>>>(
                wexp, act, ascale, sexp, dst, rows, N, K, (int)group_size);
        } else {
            uint16_t* dst = reinterpret_cast<uint16_t*>(out.data_ptr())
                + (int64_t)begin * N;
            w4_gemm_tcu<uint16_t, W4_TILE, BK><<<grid, W4_THREADS, 2 * W4_SHARED, stream>>>(
                wexp, act, ascale, sexp, dst, rows, N, K, (int)group_size);
        }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// @brief 用张量核直接消费 packed W4；group 必须按 64 对齐，K 按 4 对齐。
torch::Tensor w4a8_gemm_tcu(torch::Tensor a, torch::Tensor w, torch::Tensor xs,
                            torch::Tensor ws, torch::Tensor out, int64_t gs,
                            int64_t m_tile, int64_t db) {
    TORCH_CHECK(a.is_cuda() && a.dim() == 2 && w.dim() == 2 && out.dim() == 2,
                "CUDA 2-D activation, weight and output required");
    const int64_t M = a.size(0), K = a.size(1), N = w.size(1);
    TORCH_CHECK(K > 0 && N > 0 && K <= INT_MAX && N <= INT_MAX && M <= INT_MAX
                && K % 4 == 0 && gs > 0 && gs <= 1048576 && gs % BK == 0,
                "K must align to four and group_size to 64");
    TORCH_CHECK(a.scalar_type() == torch::kChar && w.scalar_type() == torch::kByte
                && xs.scalar_type() == torch::kFloat && ws.scalar_type() == torch::kFloat,
                "int8 activation, packed uint8 weight and fp32 scales required");
    TORCH_CHECK(w.size(0) * 2 >= K && xs.dim() == 1 && xs.numel() == M
                && ws.dim() == 2 && ws.size(0) == (K + gs - 1) / gs && ws.size(1) == N
                && out.size(0) == M && out.size(1) == N, "shape mismatch");
    TORCH_CHECK(out.scalar_type() == torch::kFloat || out.scalar_type() == torch::kBFloat16,
                "bf16 or fp32 output required");
    for (const auto& t : {a, w, xs, ws, out})
        TORCH_CHECK(t.device() == a.device() && t.is_contiguous(),
                    "all tensors must be contiguous on the same device");
    if (!M) return out;
    const c10::cuda::CUDAGuard guard(a.device());
    // M 方向 tile：64 / 128（默认）/ 256。只有 64 的整数倍可表达，因为 warp 的输出
    // tile 在 M 恒为 64；见内核的 @tparam MT。这个参数只用于扫参，生产路径不传。
    //
    // MT=256 的账：512 线程 x 128 个累加器寄存器已经顶到每 SM 寄存器文件的上限附近，
    // 所以这一档**注定溢出或掉占用率**。量它就是为了分开「权重读次数」与「寄存器压力」
    // 两个变量 —— MT=64（读双倍、压力一半）实测慢 16-29%，说明读次数是主导；
    // 若 MT=256（读一半、压力双倍）更快，就再往 512 试。
    const int mt = (m_tile == 64) ? 64 : (m_tile == 256) ? 256 : W4_TILE;
    TORCH_CHECK(mt == 64 || mt == 128 || mt == 256, "m_tile must be 0/128, 64 or 256");
    const dim3 grid((N + W4_TILE - 1) / W4_TILE, (M + mt - 1) / mt);
    // 动态共享内存：单缓冲 = 权重 2*(BK/16)*1024 + 激活 (MT/64)*(BK/16)*1024（MT=128 时
    // 正好 16 KB，与 MoE 那两处启动传的 ``2 * W4_SHARED`` 一致）；双缓冲（KB=32）两段合计
    // 与它相等，所以占用率不变 —— 这是这一档的立身之本。
    const bool use_db = (db != 0);
    const int kb = use_db ? 32 : BK;
    const int shmem = (use_db ? 2 : 1)
        * (2 * (kb / 16) * 1024 + (mt / 64) * (kb / 16) * 1024);
    const auto stream = c10::cuda::getCurrentCUDAStream();
    auto run = [&](auto tag) {
        using XT = decltype(tag);
        if (mt == 256 && use_db) {
            w4_gemm_tcu<XT, 256, 32><<<grid, 512, shmem, stream>>>(
                w.data_ptr<uint8_t>(), a.data_ptr<int8_t>(), xs.data_ptr<float>(),
                ws.data_ptr<float>(), reinterpret_cast<XT*>(out.data_ptr()), M, N, K, gs);
        } else if (mt == 256) {
            w4_gemm_tcu<XT, 256, BK><<<grid, 512, shmem, stream>>>(
                w.data_ptr<uint8_t>(), a.data_ptr<int8_t>(), xs.data_ptr<float>(),
                ws.data_ptr<float>(), reinterpret_cast<XT*>(out.data_ptr()), M, N, K, gs);
        } else if (mt == 64) {
            w4_gemm_tcu<XT, 64, BK><<<grid, 128, shmem, stream>>>(
                w.data_ptr<uint8_t>(), a.data_ptr<int8_t>(), xs.data_ptr<float>(),
                ws.data_ptr<float>(), reinterpret_cast<XT*>(out.data_ptr()), M, N, K, gs);
        } else if (use_db) {
            w4_gemm_tcu<XT, W4_TILE, 32><<<grid, W4_THREADS, shmem, stream>>>(
                w.data_ptr<uint8_t>(), a.data_ptr<int8_t>(), xs.data_ptr<float>(),
                ws.data_ptr<float>(), reinterpret_cast<XT*>(out.data_ptr()), M, N, K, gs);
        } else {
            w4_gemm_tcu<XT, W4_TILE, BK><<<grid, W4_THREADS, shmem, stream>>>(
                w.data_ptr<uint8_t>(), a.data_ptr<int8_t>(), xs.data_ptr<float>(),
                ws.data_ptr<float>(), reinterpret_cast<XT*>(out.data_ptr()), M, N, K, gs);
        }
    };
    if (out.scalar_type() == torch::kFloat) run(float{});
    else run(uint16_t{});
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor w8a8_gemm_tcu(torch::Tensor a, torch::Tensor w, torch::Tensor xs,
                            torch::Tensor ws, torch::Tensor out) {
    TORCH_CHECK(a.dim() == 2 && w.dim() == 2 && out.dim() == 2,
                "activation, weight and output must all be 2-D");
    TORCH_CHECK(a.scalar_type() == torch::kChar && w.scalar_type() == torch::kChar,
                "operands must be int8");
    TORCH_CHECK(a.is_contiguous() && w.is_contiguous(),
                "contiguous operands required");
    TORCH_CHECK(a.size(1) == w.size(0), "K mismatch: A ", a.sizes(), " W ", w.sizes());
    TORCH_CHECK(out.size(0) == a.size(0) && out.size(1) == w.size(1),
                "out ", out.sizes(), " does not match ", a.size(0), "x", w.size(1));
    TORCH_CHECK(xs.scalar_type() == torch::kFloat && xs.numel() == a.size(0),
                "xs must be fp32 with one value per row");
    TORCH_CHECK(ws.scalar_type() == torch::kFloat && ws.numel() == w.size(1),
                "ws must be fp32 with one value per column");
    TORCH_CHECK(a.is_cuda() && w.is_cuda() && out.is_cuda(),
                "all operands must be on the device");

    const int M = (int)a.size(0);
    const int K = (int)a.size(1);
    const int N = (int)w.size(1);
    if (M == 0 || N == 0) return out;

    // ``out`` 可以是列切片，行距因而大于 N：``lm_head`` 按 ``LIB_MAX_N`` 沿 N 切块时
    // 传进来的是 ``out[:, start:stop]``。**不能要求整块连续**，否则那条 248320 列的路
    // 会直接抛异常；改成接受任意行距，内核按 ``stride`` 写回。
    const int out_stride = (int)out.stride(0);
    TORCH_CHECK(out.stride(1) == 1 && out_stride >= N,
                "out must have unit inner stride and a row stride of at least N; got "
                "strides ", out.stride(0), " / ", out.stride(1));

    const dim3 grid((N + BM - 1) / BM, (M + BN - 1) / BN);
    const size_t shmem = 2 * (size_t)(SA + SB);
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    auto run = [&](auto tag) {
        using XT = decltype(tag);
        if (shmem > 48 * 1024) {
            C10_CUDA_CHECK(cudaFuncSetAttribute(gemm_tcu_kernel<XT>,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                (int)shmem));
        }
        gemm_tcu_kernel<XT><<<grid, NTHREADS, shmem, stream>>>(
            reinterpret_cast<const int8_t*>(w.data_ptr()), 
            reinterpret_cast<const int8_t*>(a.data_ptr()), 
            reinterpret_cast<XT*>(out.data_ptr()), xs.data_ptr<float>(),
            ws.data_ptr<float>(), M, N, K, out_stride);
    };
    if (out.scalar_type() == torch::kBFloat16) {
        run(uint16_t{});
    } else {
        TORCH_CHECK(out.scalar_type() == torch::kFloat, "out must be bf16 or fp32, got ",
                    out.scalar_type());
        run(float{});
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
