// @file moe_i8_tcu_device.inc
// @brief 分组 int8 张量核与它的转置前置核（设备侧）。被 abi/moe_i8_tcu_abi.cu 包含。
//
// 原文件里两条模板核写在命名空间**外**（文件作用域）。这里把它们一并收进匿名
// 命名空间：模板本来就只有本 TU 的实例化，收进来可以确保另一个 TU 不会拿到同名
// 的 ``__device_stub__``。
//
// **包含位置就是可见性边界**：include 必须在文件作用域，命名空间外的常量会先被
// 后面的代码看到。

#ifndef COFFIN_MOE_I8_TCU_DEVICE_INC
#define COFFIN_MOE_I8_TCU_DEVICE_INC

#include "corex_i8.h"
#include "corex_scalar.h"

#include <cassert>
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

// @brief 把转置激活的一段从 ``[K, rows]`` 搬进共享内存（支持任意起止行与非 4 对齐段）。
//
// ``[K, rows]`` 里固定 k 的 4 个相邻行连续；当 ``mt0`` 与 4 对齐且未触及段尾 ``m_end`` 时
// 走 32 位整字读，否则按字节拼字并对超出 ``m_end`` 的槽位补零。这样每专家任意行数
// （含 1/2/3 行与任意奇偶偏移）都能安全进张量核，不再需要外部按 4 对齐校验或 D2H 同步。
//
// @param sa    目的段起点。
// @param at    ``[K, rows]`` 转置后的 int8 激活。
// @param mt0   本块在行方向的起点（从 ``offsets[e]`` 步进）。
// @param m_end 本专家有效行上界（``min(r1, rows)``）。
// @param rows  ``at`` 的行宽（总激活行数）。
// @param k0    本块在 K 方向的起点。
// @param K     k 维长度。
// @param tid   本线程在线性任务表里的下标。
template <int TBK, int BMw, int NTH>
__device__ __forceinline__ void stage_at(int8_t* sa, const int8_t* __restrict__ at,
                                         int mt0, int m_end, int rows, int k0, int K, int tid) {
    constexpr int J = BMw / 4;
#pragma unroll
    for (int rep = 0; rep < (TBK * J + NTH - 1) / NTH; ++rep) {
        int const t = tid + rep * NTH;
        if (t >= TBK * J) break;
        int const k = t / J;
        int const j = t % J;
        int const gk = k0 + k;
        int const gn4 = mt0 + j * 4;
        unsigned v = 0;
        if (gk < K && gn4 < m_end) {
            const int8_t* p = at + (size_t)gk * (size_t)rows + gn4;
            if (((reinterpret_cast<uintptr_t>(p) & 3u) == 0u) && (gn4 + 3 < m_end)) {
                v = *reinterpret_cast<const unsigned*>(p);
            } else {
                const unsigned b0 = (unsigned)(uint8_t)p[0];
                const unsigned b1 = (gn4 + 1 < m_end) ? (unsigned)(uint8_t)p[1] : 0u;
                const unsigned b2 = (gn4 + 2 < m_end) ? (unsigned)(uint8_t)p[2] : 0u;
                const unsigned b3 = (gn4 + 3 < m_end) ? (unsigned)(uint8_t)p[3] : 0u;
                v = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
            }
        }
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
// @tparam TBK 本段的 k 步长。
// @tparam BNw 本块在 MMA-N（= 激活 M 方向）的 tile 宽。
// @tparam NTH 每块线程数。
template <int TBK, int BNw, int NTH>
__device__ __forceinline__ void stage_act(int8_t* sb, const int8_t* __restrict__ Ac,
                                          int m0, int k0, int M, int K, int tid) {
    constexpr int J = TBK / 4;
#pragma unroll
    for (int rep = 0; rep < (BNw * J + NTH - 1) / NTH; ++rep) {
        int const t = tid + rep * NTH;
        if (t >= BNw * J) break;
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





// 分组 W8A8 张量核：nk 权重栈 + 转置激活，一个专家一趟、权重只读一遍。
//
// ## 为什么不是 ``moe_grouped_i8`` 那条 SIMT 路
//
// 同一张卡上的实测（单体基准，与真实预填充同形）：
//
//   ``moe_grouped_i8_tcu``（本核）同形状              0.446 ms   19.2 TMAC/s
//   ``moe_grouped_i8_expert`` 同形状                  8.99  ms    0.96 TMAC/s
//   d2d 拷贝（带宽口径）                              605 GB/s
//
// 差 14 倍。dp4a 那条路的天花板就是「每周期每 lane 一个 4 乘」，再怎么调几何也只能
// 到 1 TMAC/s 量级；而 PP2 预填充要 4000 tps，每卡需要约 2 TMAC/s。所以专家 GEMM
// 必须走张量核。
//
// ## 为什么操作数是「互换」的（这是本内核唯一反直觉的地方）
//
// 硬件 fragment 的取法固定：mat-a 要「同 k 上 4 个相邻 MMA-M」，mat-b 要「同 MMA-N
// 上 4 个相邻 k」。nk 栈是 ``[N, K]``：固定 n 的 4 个相邻 k 连续 —— 正好是 mat-b 的
// 形状。于是这里把两个操作数互换：mat-a 吃转置后的激活 ``[K, rows]``（固定 k 的 4 个
// 相邻行连续），mat-b 吃 nk 权重。结果仍是 ``out[rows, N]``，不要转置输出。
//
// 互换的代价是累加器的轴向也要跟着换：``rt`` 走 MMA-M（= 行）、``ct`` 走 MMA-N（= 列），
// 与 ``stage_act`` 那边的轴向相反。所以下面有独立的 ``store_epi_rows``。
//
// ## 形状
//
// 一个 512 token 的预填充块摊到 256 个专家是每专家 16 行，而张量核的行 tile 最小 64
// —— 四分之三的行是空的。这份浪费换来的是 19 TMAC/s：按每层 12.9 GMAC 算，即使只跑
// 到四分之一利用率也有 4.8 TMAC/s，仍高于 4000 tps 需要的 2 TMAC/s。
constexpr int MI8_BM = 64;   // MMA-M：行
//: MMA-N：列。本核只有 4 个 warp 的根源在 M 轴——每专家 64 行就是一个 tile，撑不出
//: 第二排 warp。所以并行度只能加在列上：256 时每块 4 warp，256 片里每片 33.5 MMAC，
//: 实测 5.33 TMAC/s（同形状稠密核 19.2，它有 16 个 warp）。512 时每块 8 warp、块数
//: 减半、每块 67 MMAC，共享内存 36 KB 仍放得下。
constexpr int MI8_BN = 512;
//: k 步长。64 时每块占 40 KB 共享内存，全卡只放得下 1 块/ SM——而本核每个块只有
//: 64x256 的输出（33.5 MMAC），一趟 k 循环的搬运（20 KB）盖不住地等全局延迟，
//: 实测只有 4.19 TMAC/s（同形状稠密核 19.2，它每块 256x256 输出、16 个 warp）。
//: 减到 32 后共享内存 20 KB，同一 SM 上能压两块互相盖延迟；代价是 k 循环从 32 趟
//: 变 64 趟，每趟的搬运字节减半，总量不变。
constexpr int MI8_BK = 32;

// @brief 把累加器写回 ``[rows, N]``：``rt`` 走行、``ct`` 走列。
//
// 累加器的 4 个分量沿 MMA-M 每 4 个一组，``lane >> 4`` 也是沿 MMA-M。
template <typename XT>
__device__ __forceinline__ void store_epi_rows(const v4i32* acc, XT* __restrict__ out,
                                               const float* __restrict__ xs,
                                               const float* __restrict__ ws,
                                               int m_base, int n_base, int m_end, int N) {
    int const lane = __ivcorex_lane_id();
    int const off_m = lane >> 4;
    int const off_n = lane & 15;
#pragma unroll
    for (int rt = 0; rt < 4; ++rt) {
#pragma unroll
        for (int ct = 0; ct < 4; ++ct) {
            v4i32 v = acc[4 * rt + ct];
            int const n = n_base + 16 * ct + off_n;
            int const vals[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                int const m = m_base + 16 * rt + off_m + 4 * q;
                if (m < m_end && n < N) {
                    out[(size_t)m * N + n] = CxScalar<XT>::from_float(
                        (float)vals[q] * xs[m] * ws[n]);
                }
            }
        }
    }
}

// @brief 分组 W8A8 张量核：``out[r, n] = xs[r] * ws[e, n] * sum_k at[k, r] * w[e, n, k]``。
//
// @param at       ``[K, rows]`` int8 连续（激活的 k-主序转置）。
// @param rows     激活总行数，用于搬片时的上界。
// @param wstack   ``[E, N, K]`` int8 连续（nk 原生栈）。
// @param out      ``[rows, N]``，bf16 或 fp32。
// @param xs       ``[rows]`` 逐行激活 scale。
// @param ws       ``[E, N]`` 逐 (专家, 列) 权重 scale。
// @param offsets  ``[E+1]`` 每专家行区间。
template <typename XT, int TBN = MI8_BN>
__global__ void moe_grouped_i8_tcu_kernel(const int8_t* __restrict__ at, int rows,
                                          const int8_t* __restrict__ wstack,
                                          XT* __restrict__ out,
                                          const float* __restrict__ xs,
                                          const float* __restrict__ ws,
                                          const int* __restrict__ offsets,
                                          int K, int N) {
    constexpr int WRt = MI8_BM / 64;
    constexpr int WCt = TBN / 64;
    constexpr int NTH = WRt * WCt * 64;
    constexpr int SAt = WRt * (MI8_BK / 16) * 1024;
    constexpr int SBt = WCt * (MI8_BK / 16) * 1024;
    extern __shared__ int8_t smem[];
    int8_t* sa = smem;
    int8_t* sb = smem + 2 * SAt;

    int const tid = threadIdx.x;
    int const warp = tid >> 6;
    int const wm = warp / WCt;   // 沿 MMA-M（行）
    int const wn = warp % WCt;   // 沿 MMA-N（列）
    int const e = blockIdx.y;
    const int r0 = offsets[e];
    const int r1 = min(offsets[e + 1], rows);
    if (r0 >= r1) return;
    const int n0 = blockIdx.x * TBN;
    const int8_t* wbase = wstack + (size_t)e * (size_t)N * (size_t)K;
    // 逐 (专家, 列) 的 scale：内层的 n 是列号，专家偏移必须在这里落一次，否则每个专家
    // 都会用第 0 个专家的列 scale（小形状单专家时看不出，真实 256 专家时全错）。
    const float* ws_for_e = ws + (size_t)e * (size_t)N;

    const int nk = (K + MI8_BK - 1) / MI8_BK;
    for (int mt0 = r0; mt0 < r1; mt0 += MI8_BM) {
        v4i32 acc[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            acc[i].x = 0; acc[i].y = 0; acc[i].z = 0; acc[i].w = 0;
        }
        // mat-a = 激活 ``[K, rows]``：支持任意对齐与段尾补零。
        stage_at<MI8_BK, MI8_BM, NTH>(sa, at, mt0, r1, rows, 0, K, tid);
        // mat-b = 权重 nk ``[N, K]``：固定 n 的 4 个相邻 k 连续。
        stage_act<MI8_BK, TBN, NTH>(sb, wbase, n0, 0, N, K, tid);
        __syncthreads();

        for (int kb = 0; kb < nk; ++kb) {
            int const cur = kb & 1;
            if (kb + 1 < nk) {
                stage_at<MI8_BK, MI8_BM, NTH>(sa + (cur ^ 1) * SAt, at, mt0, r1, rows,
                                              (kb + 1) * MI8_BK, K, tid);
                stage_act<MI8_BK, TBN, NTH>(sb + (cur ^ 1) * SBt, wbase, n0,
                                            (kb + 1) * MI8_BK, N, K, tid);
            }
            const int8_t* wa = sa + cur * SAt + wm * (MI8_BK / 16) * 1024;
            const int8_t* wb = sb + cur * SBt + wn * (MI8_BK / 16) * 1024;
            unsigned a0[4], a1[4], b0[4], b1[4];
#pragma unroll
            for (int t = 0; t < MI8_BK / 32; ++t) {
                ld_a(a0, wa + t * 2048);
                ld_a(a1, wa + t * 2048 + 1024);
                ld_b(b0, wb + t * 2048);
                ld_b(b1, wb + t * 2048 + 1024);
                mma_tile(a0, a1, b0, b1, acc);
            }
            __syncthreads();
        }
        store_epi_rows<XT>(acc, out, xs, ws_for_e, mt0 + 64 * wm, n0 + 64 * wn, r1, N);
    }
}

// @brief ``[rows, K]`` int8 -> ``[K, rows]``，64x64 分块走共享内存。
//
// 逐元素的 ``x.t().contiguous()`` 在 PyTorch 里是逐元素带步长的拷贝，实测 4096x2048
// 只有 15.7 GB/s（1.07 ms），而这一步在每个专家投影前都要做一次，占了张量核那条路
// 两成的时间。按块转置后两个方向都是整行连续读写，同一份字节只要 0.1 ms 量级。
__global__ void transpose_i8_kernel(const int8_t* __restrict__ src,
                                    int8_t* __restrict__ dst, int rows, int K) {
    // 65 的填充让「按列读」那一步不撞同一个 bank。
    __shared__ int8_t tile[64][65];
    int const m0 = blockIdx.x * 64;
    int const k0 = blockIdx.y * 64;
    int const tid = threadIdx.x;
    // 源按行连续：同一行的 64 个 k 落在同一段 64 字节里。
    for (int i = tid; i < 64 * 64; i += blockDim.x) {
        int const dm = i >> 6;
        int const dk = i & 63;
        int const m = m0 + dm;
        int const k = k0 + dk;
        tile[dm][dk] = (m < rows && k < K) ? src[(size_t)m * K + k] : int8_t(0);
    }
    __syncthreads();
    // 目的按 k 行连续：同一 k 的 64 个 m 落在同一段 64 字节里。
    for (int i = tid; i < 64 * 64; i += blockDim.x) {
        int const dk = i >> 6;
        int const dm = i & 63;
        int const m = m0 + dm;
        int const k = k0 + dk;
        if (m < rows && k < K) dst[(size_t)k * rows + m] = tile[dm][dk];
    }
}

}  // namespace

#endif  // COFFIN_MOE_I8_TCU_DEVICE_INC
