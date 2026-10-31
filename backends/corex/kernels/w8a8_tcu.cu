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
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
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
// 实测过的判据（`diag/corex/ixb_tile.sh`，单位脉冲激活）：库的输出里
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

}  // namespace

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
