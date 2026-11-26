// @file w8a8_dp4a_device.inc
// @brief ``[N, K]``（nk）布局的 ``__dp4a`` int8 GEMM：几何常量、内核、启动器。
//
// Module: coffin/backends/corex/kernels。被两条前端各包含一次：``w8a8_dp4a.cu``
// （torch 扩展的 pybind 入口）与 ``abi/w8a8_dp4a_abi.cu``（``extern "C"`` 出口）。
// 同一份设备代码，所以分块几何不可能各自漂移。
//
// 布局与约束见 ``w8a8_dp4a.cu`` 的文件头：权重 ``[N, K]``（K 连续）、累加器 int32、
// ``K % 4 == 0``、内核不做边界检查（越界补零，零乘进累加器不影响结果）。
//
// Constraints:
//   * ``mode`` 是 ``kOutNone`` / ``kOutBf16`` / ``kOutF32``；``kOutNone`` 时写 ``C``。
//   * ``M == 4`` / ``M == 8`` 走更窄的编译期行块；其余走默认 32 行块。

#ifndef COFFIN_W8A8_DP4A_DEVICE_INC
#define COFFIN_W8A8_DP4A_DEVICE_INC

#include "corex_scalar.h"

#include <cstdint>
#include <cuda_runtime.h>

//: 输出模式。融合 epilogue 在 k 循环之外执行，所以这是运行时分支而不是模板参数：
//: 多一个模板实例就多一份编译单元，而这条分支每个输出元素只走一次。
enum : int { kOutNone = 0, kOutBf16 = 1, kOutF32 = 2 };

//: 默认 ``nk`` 块内几何。一个块 = ``NK_TX * NK_TY`` 个线程，每个线程一块
//: ``NK_WM x NK_WN`` 的累加器，于是块覆盖 ``NK_BM x NK_BN`` 个输出。M=4/8
//: 另用更窄的编译期几何，让每块只计算有效行，并把列块缩到 32。
//:
//: 这些数不是随手挑的：``NK_WM * NK_WN = 32`` 个 int32 累加器，加 4 条激活字和 8 条权重
//: 字，寄存器账大约 60 个——在 255 的上限里留足余量给地址计算和循环变量。32 个累加器
//: 同时意味着每个 k4 步只要 12 次 LDS 读就能喂 32 条 ``__dp4a``，读:算 = 1:2.7，LDS
//: 带宽不会成为瓶颈（这是分块的全部意义）。
//:
//: ``NK_BM = 32`` 的默认形状不适合小 M，所以 M=4/8 另用精确行块；M=1 不归这里——
//: 单行投影走 ``gemv_nk_i8``，那条路本来就是每字节一次乘加、算术藏在访存后面。
constexpr int NK_TX = 16;
constexpr int NK_TY = 8;
constexpr int NK_WM = 4;
constexpr int NK_WN = 8;
constexpr int NK_BM = NK_TY * NK_WM;
constexpr int NK_BN = NK_TX * NK_WN;
constexpr int NK_THREADS = NK_TX * NK_TY;
constexpr int NK_BK = 64;

//: LDS 分块大小。A 块 ``NK_BM x NK_BK`` 字节，B 块 ``NK_BN x NK_BK`` 字节。
constexpr int NK_AELEMS = NK_BM * NK_BK;
constexpr int NK_BELEMS = NK_BN * NK_BK;

//: A 保持普通行主序；B 在 LDS 中按 ``[j, tx, k]`` 排列。
//:
//: 只给 B 行距加 padding 并不能消掉冲突：相邻 ``tx`` 的逻辑行相差 ``NK_WN=8``，
//: 行距为 17 个 4 字节字时 bank 仍只跨 8 格，16 个 ``tx`` 实际落在 4 个 bank 上。
//: 将 ``tx`` 放到 LDS 行的内层后，相邻 ``tx`` 的 bank 跨度变为 17；它与 32 互质，
//: 一条 ``bcol[j]`` 读取的 16 个不同地址各落一个 bank。行距必须是 17 个字，否则
//: 16 个 ``tx`` 会在两个 bank 间冲突。
constexpr int NK_LD = NK_BK + 4;
constexpr int NK_AELEMS_PAD = NK_BM * NK_LD;
constexpr int NK_BELEMS_PAD = NK_BN * NK_LD;



// @brief 写一个输出元素：``out = narrow(acc * xs[m] * ws[n])``。
//
// 两个 scale 在同一个 fp32 表达式里乘完再窄化一次，与 ``w8a8_epilogue`` 的舍入次数
// 一致；分两次窄化会多一次舍入，而结果要穿过 32 层。
//
// @param dst 目的地基址。
// @param idx 元素下标。
// @param acc int32 累加值。
// @param xm 本行的激活 scale。
// @param wv 本列的权重 scale。
template <int MODE>
__device__ __forceinline__ void cx_store_scaled(void* __restrict__ dst, size_t idx, int32_t acc,
                                                float xm, float wv) {
    const float v = (float)acc * xm * wv;
    if (MODE == kOutBf16) {
        ((uint16_t*)dst)[idx] = CxScalar<uint16_t>::from_float(v);
    } else {
        ((float*)dst)[idx] = v;
    }
}

// @brief ``[N, K]`` 布局（K 连续）的 ``__dp4a`` int8 GEMM：寄存器分块 + LDS 分块。
//
// 这是预填充要的那条路径。权重按 ``[N, K]`` 存放后，4 个相邻 int8 就是 4 个相邻的 k，
// 激活侧同样连续，两边各一条 32 位加载喂一条 ``__dp4a``，每条指令四次乘加。
//
// 几何上三层，每层都为了解决一个具体的浪费：
//
//   1. **寄存器分块**：每个线程持 ``NK_WM x NK_WN = 32`` 个累加器。没有它，一个 k4 步只
//      能喂 1 条 dp4a，而读一次权重就是 1 次全局加载——读:算 = 1:1，全部时间花在发
//      访存上。32 个累加器把 12 次 LDS 读摊到 32 条 dp4a 上。
//   2. **LDS 分块**：一个块把 A 和 B 的 ``NK_BK`` 个 k 搬到共享内存，块内所有线程复用。
//      直接全局读的话每个线程要把整个 tile 读一遍，重复 ``NK_BN`` 次。
//   3. **k 分块长度 ``NK_BK = 64``**：A 块 32x64 = 2 KB、B 块 128x64 = 8 KB，共 10 KB。
//      本部件每 CU 有 64 KB LDS，所以几个块能同时驻留——LDS 占用设成上限而块数是零，
//      那会把波次压成 1，而这张卡最怕的就是块数不够。
//
// 位置计算与 CUDA 的矩阵乘教程同构：线程在块内排成 ``NK_TX x NK_TY``，每个线程负责
// ``(ty * NK_WM + i, tx * NK_WN + j)`` 那一格。
//
// @param A [M, K] int8 激活。
// @param WT [N, K] int8 权重，K 连续。
// @param C [M, N] int32 输出。
// @param xs [M] fp32 每 token 激活 scale。
// @param ws [N] fp32 每输出列权重 scale。
// @param out 融合输出；``MODE == kOutNone`` 时为 nullptr。
// @param M 行数。
// @param K 归约长度；必须是 4 的倍数（``__dp4a`` 的原子粒度）。
// @param N 输出列数。
// @param ld_out 融合输出的行间距。
template <int MODE, int TX, int TY, int WM, int WN>
__global__ void nk_kernel(const int8_t* __restrict__ A, const int8_t* __restrict__ WT,
                          int32_t* __restrict__ C, const float* __restrict__ xs,
                          const float* __restrict__ ws, void* __restrict__ out, int M, int K,
                          int N, int ld_out) {
    constexpr int NK_TX = TX;
    constexpr int NK_TY = TY;
    constexpr int NK_WM = WM;
    constexpr int NK_WN = WN;
    constexpr int NK_BM = NK_TY * NK_WM;
    constexpr int NK_BN = NK_TX * NK_WN;
    constexpr int NK_THREADS = NK_TX * NK_TY;
    constexpr int NK_AELEMS = NK_BM * NK_BK;
    constexpr int NK_BELEMS = NK_BN * NK_BK;
    constexpr int NK_AELEMS_PAD = NK_BM * NK_LD;
    constexpr int NK_BELEMS_PAD = NK_BN * NK_LD;
    __shared__ int8_t atile[NK_AELEMS_PAD];
    __shared__ int8_t btile[NK_BELEMS_PAD];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int m_blk = blockIdx.y * NK_BM;
    const int n_blk = blockIdx.x * NK_BN;
    const int tid = ty * NK_TX + tx;

    int32_t acc[NK_WM][NK_WN];
#pragma unroll
    for (int i = 0; i < NK_WM; ++i) {
#pragma unroll
        for (int j = 0; j < NK_WN; ++j) acc[i][j] = 0;
    }

    const int k4 = K & ~3;
    for (int k0 = 0; k0 < k4; k0 += NK_BK) {
        // 搬 A 块：``NK_BM`` 行 x ``NK_BK`` 字节，写进带 padding 的行距。相邻下标读相邻
        // 字节，一个 wavefront 一次 256 字节整段——这是 D1 那条「4 字节 lane」在 LDS
        // 路径上的落点。
        for (int idx = tid * 4; idx < NK_AELEMS; idx += NK_THREADS * 4) {
            const int r = idx / NK_BK;
            const int c = idx - r * NK_BK;
            const int gm = m_blk + r;
            // 越界补零而不是跳过：LDS 里的垃圾会乘进累加器，而它看起来就像一个舍入误差。
            const int32_t v = (gm < M && k0 + c + 3 < k4)
                                  ? *(const int32_t*)(A + (size_t)gm * K + k0 + c)
                                  : 0;
            *(int32_t*)(atile + r * NK_LD + c) = v;
        }
        // 全局仍按原始行主序连续读取；落入 LDS 时把逻辑行 ``tx * NK_WN + j``
        // 重排成 ``j * NK_TX + tx``，让后面的每条 B 读取在 bank 间均匀分布。
        for (int idx = tid * 4; idx < NK_BELEMS; idx += NK_THREADS * 4) {
            const int r = idx / NK_BK;
            const int c = idx - r * NK_BK;
            const int gn = n_blk + r;
            const int32_t v = (gn < N && k0 + c + 3 < k4)
                                  ? *(const int32_t*)(WT + (size_t)gn * K + k0 + c)
                                  : 0;
            const int sr = (r % NK_WN) * NK_TX + r / NK_WN;
            *(int32_t*)(btile + sr * NK_LD + c) = v;
        }
        __syncthreads();

        const int klen = min(NK_BK, k4 - k0);
        for (int l = 0; l < klen; l += 4) {
            // 先把本线程的 A 与 B 片段读进寄存器，再让 32 条 dp4a 消费它们。读 LDS 的
            // 次数（NK_WM + NK_WN = 12）远小于 dp4a 的次数（32），这就是分块挣来的。
            int32_t arow[NK_WM];
            int32_t bcol[NK_WN];
#pragma unroll
            for (int i = 0; i < NK_WM; ++i) {
                arow[i] = *(const int32_t*)(atile + (ty * NK_WM + i) * NK_LD + l);
            }
#pragma unroll
            for (int j = 0; j < NK_WN; ++j) {
                bcol[j] = *(const int32_t*)(btile + (j * NK_TX + tx) * NK_LD + l);
            }
#pragma unroll
            for (int i = 0; i < NK_WM; ++i) {
#pragma unroll
                for (int j = 0; j < NK_WN; ++j) acc[i][j] = __dp4a(arow[i], bcol[j], acc[i][j]);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < NK_WM; ++i) {
        const int m = m_blk + ty * NK_WM + i;
        if (m >= M) break;
#pragma unroll
        for (int j = 0; j < NK_WN; ++j) {
            const int n = n_blk + tx * NK_WN + j;
            if (n >= N) break;
            if (MODE == kOutNone) {
                if (C != nullptr) C[(size_t)m * N + n] = acc[i][j];
            } else {
                cx_store_scaled<MODE>(out, (size_t)m * ld_out + n, acc[i][j], xs[m], ws[n]);
            }
        }
    }
}

template <int MODE, int TX, int TY, int WM, int WN>
void nk_launch(const int8_t* A, const int8_t* WT, int32_t* C, const float* xs, const float* ws,
               void* out, int M, int K, int N, int ld_out, cudaStream_t stream) {
    constexpr int BM = TY * WM;
    constexpr int BN = TX * WN;
    const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    const dim3 block(TX, TY);
    nk_kernel<MODE, TX, TY, WM, WN>
        <<<grid, block, 0, stream>>>(A, WT, C, xs, ws, out, M, K, N, ld_out);
}

template <int TX, int TY, int WM, int WN>
void nk_launch_mode(int mode, const int8_t* A, const int8_t* WT, int32_t* C, const float* xs,
                    const float* ws, void* out, int M, int K, int N, int ld_out,
                    cudaStream_t stream) {
    if (mode == kOutBf16) {
        nk_launch<kOutBf16, TX, TY, WM, WN>(A, WT, C, xs, ws, out, M, K, N, ld_out, stream);
    } else if (mode == kOutF32) {
        nk_launch<kOutF32, TX, TY, WM, WN>(A, WT, C, xs, ws, out, M, K, N, ld_out, stream);
    } else {
        nk_launch<kOutNone, TX, TY, WM, WN>(A, WT, C, xs, ws, out, M, K, N, ld_out, stream);
    }
}

// @brief 启动 dp4a GEMM，按 M 选行块。
//
// @param mode 输出模式：``kOutNone`` 写 ``C``，否则写 ``out``。
// @param A ``[M, K]`` int8 激活。
// @param WT ``[N, K]`` int8 权重，K 连续。
// @param C ``[M, N]`` int32 累加输出；``kOutNone`` 模式下这就是结果。
// @param xs ``[M]`` fp32 每行激活 scale。
// @param ws ``[N]`` fp32 每列权重 scale。
// @param out 融合输出，可为空。
// @param M, K, N 几何；``K % 4 == 0``。
// @param ld_out 融合输出的行间距。
// @param stream 目标流；空指针表示默认流。
inline void launch_w8a8_dp4a_nk(int mode, const int8_t* A, const int8_t* WT, int32_t* C,
                                const float* xs, const float* ws, void* out, int M, int K, int N,
                                int ld_out, cudaStream_t stream) {
    // M=4/8 时改用精确行块；默认 32 行块会让无效行也完整执行 dp4a，且 128 列块
    // 给小输出形状的网格块数偏少。
    if (M == 4) {
        nk_launch_mode<8, 4, 1, 4>(mode, A, WT, C, xs, ws, out, M, K, N, ld_out, stream);
    } else if (M == 8) {
        nk_launch_mode<8, 8, 1, 4>(mode, A, WT, C, xs, ws, out, M, K, N, ld_out, stream);
    } else {
        nk_launch_mode<NK_TX, NK_TY, NK_WM, NK_WN>(mode, A, WT, C, xs, ws, out, M, K, N, ld_out,
                                                   stream);
    }
}

#endif  // COFFIN_W8A8_DP4A_DEVICE_INC
