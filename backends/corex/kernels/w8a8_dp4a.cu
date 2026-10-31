// @file w8a8_dp4a.cu
// @brief 手写 int8 GEMM：``[K, N]`` 原生布局的带宽路径、``[N, K]`` 布局的 ``__dp4a``
//        计算路径、跨层共享 grid，以及三条量天花板用的微基准。
//
// Module: coffin/backends/corex/kernels。绑定成 ``w8a8_dp4a`` / ``w8a8_dp4a_nk`` /
// ``w8a8_dp4a_groups`` / ``w8a8_dp4a_read`` / ``w8a8_dp4a_rate``。只被
// ``diag/corex/`` 的探针调用；定稿之后才由 ``backends/corex/__init__.py`` 当作
// ``w8a8_gemm`` 的一条后端。
//
// 为什么手写：厂商库的 tile、lane 宽度和 grid 全都改不了，而三条方向里有两条（4 字节
// lane / 512 列、跨层共享 grid）恰恰就是这三样。库在本卡上跑 49.07 TINT8-OPS/s
// （``W8A8_PLAN.md`` §1.2），手写内核的意义就是把这些数放进可调的几何里重新量一遍。
//
// 两个内核家族，因为**布局决定了能不能用 ``__dp4a``**，而这件事本身就是一条要记账的
// 结论：
//
//   * ``kn`` 家族：权重保持契约的 ``[K, N]``。一个 wavefront 一次读 ``64 x NTILE`` 个
//     **连续列**字节——这正是 D1 要的访存形状，也正是 GEMV 那边 442 GB/s 的图案。但
//     归约轴是 K，4 个相邻列字节属于 4 个不同输出，``__dp4a`` 会把它们加成一个数。所以
//     它每读一个字节要一次整数乘加。M=1 时一个字节本来就只喂一次乘加，算术完全藏在访存
//     后面，于是这条路径最短（不需要 LDS、不需要额外布局）；M 大时它就是 ALU 墙。
//   * ``nk`` 家族：权重按 ``[N, K]``（K 连续）存放。4 个相邻 int8 就是 4 个相邻 k，
//     激活侧同样连续，两边各一条 32 位加载喂一条 ``__dp4a``，每条指令四次乘加。代价是
//     权重多一份布局，收益是每字节 1/4 条指令。预填充要的是这一条。
//
// 带宽优化（D1）和计算优化（dp4a）作用在**不同区间**上，这一点必须分开量：M 小的时候
// 权重读取是全部，4 字节 lane 直接决定 GB/s；M 大的时候字节数相对算力是零头，指令吞吐
// 才是天花板。两个家族分开测，就是为了让这个分界线有数字而不是论证。
//
// Constraints:
//   * 累加器 int32。int8 x int8 的乘积精确落在 int32 里，而这条路径的精度预算已经
//     全部花在量化上（单层 rel_err 9.07e-3），不往里加累加误差。
//   * ``N % NTILE == 0`` 是行首对齐的前提。不满足时主机侧强制退回 ``NTILE = 1``，而
//     不是让内核跨行读并算出看着合理的数（``corex_i8.h`` 的约束）。``nk`` 家族的对齐
//     条件是 ``K % 4 == 0``。
//   * ``kn`` 家族用 ``cx_load_i8_row`` 这个共享原语，不另写一份加载：D1 的意义正是让
//     GEMM 与 GEMV 走同一套宽加载，两处各写一份就量不出同一件事。
//   * 每次启动都在 ``c10::cuda::getCurrentCUDAStream()`` 上。
//   * 内核不做边界检查：形状、dtype、连续性、对齐条件全部在主机侧判掉。
//   * 全程不用 fp64：本设备对它返回 0（COREX_PORT.md §0），设备侧 fp64 参考等于零。

#include "corex_i8.h"
#include "corex_scalar.h"
#include "w8a8_dp4a.h"

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

//: 输出模式。融合 epilogue 在 k 循环之外执行，所以这是运行时分支而不是模板参数：
//: 多一个模板实例就多一份编译单元，而这条分支每个输出元素只走一次。
enum : int { kOutNone = 0, kOutBf16 = 1, kOutF32 = 2 };

//: 跨层共享 grid 最多拼几个形状。8 个足以覆盖一层里全部窄投影，同时让整张 group 表
//: 放进内核参数（8 x 24 字节）——每次启动一次 ``cudaMemcpyAsync`` 在 M=1 的小形状上
//: 就是它全部收益的量级，所以描述符绝不能走显存。
constexpr int kMaxGroups = 8;

//: 归约类内核的线程数上限；共享归约数组按它静态分配。
constexpr int kRedThreads = 256;

//: ``nk`` 家族的块内几何。一个块 = ``NK_TX * NK_TY`` 个线程，每个线程一块
//: ``NK_WM x NK_WN`` 的累加器，于是块覆盖 ``NK_BM x NK_BN`` 个输出。
//:
//: 这些数不是随手挑的：``NK_WM * NK_WN = 32`` 个 int32 累加器，加 4 条激活字和 8 条权重
//: 字，寄存器账大约 60 个——在 255 的上限里留足余量给地址计算和循环变量。32 个累加器
//: 同时意味着每个 k4 步只要 12 次 LDS 读就能喂 32 条 ``__dp4a``，读:算 = 1:2.7，LDS
//: 带宽不会成为瓶颈（这是分块的全部意义）。
//:
//: ``NK_BM = 32`` 还有一个 M=1 上的后果：块覆盖 32 行而只有 1 行有数据，97% 的块空转。
//: 这是寄存器分块的固有代价，而 ``nk`` 本来就是给预填充写的；M=1 上该走的是 ``kn``
//: （见 ``kn_kernel`` 的注释，那里每字节一次乘加在 M=1 上完全藏在访存后面）。
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

//: LDS 行距。**必须带 padding**。
//:
//: 行距取 ``NK_BK`` 时，相邻线程（``tx`` 差一）在 B 块上的地址差是
//: ``NK_WN * NK_BK = 8 x 64 = 512`` 字节，正好是 128 个 4 字节字——也就是 LDS 的
//: 32 bank 走了整数 4 轮，于是同一个 wavefront 里 16 个 ``tx`` 全部落在同一个 bank 上：
//: 16 路冲突，每次 LDS 读要 16 拍。实测这个内核只跑到指令天花板的 1/13.7，正是这个
//: 数字；把它当成「dp4a 不行」会得出完全错误的结论。
//:
//: 加 4 字节让行距变成 544 字节 = 136 字，136 mod 32 = 8，相邻线程散开到不同 bank。
constexpr int NK_LD = NK_BK + 4;
constexpr int NK_AELEMS_PAD = NK_BM * NK_LD;
constexpr int NK_BELEMS_PAD = NK_BN * NK_LD;

//: 一个形状在内核参数里的副本。指针 + 行间距 + 列数 + 列起点 + scale，五样不可拆。
struct GroupArg {
    const int8_t* w;
    const float* ws;
    int n;
    int ld;
    int c0;
};

//: 按值传进内核的 group 表。未使用的槽 ``n == 0``。
struct GroupTable {
    GroupArg g[kMaxGroups];
    int ng;
};

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

// @brief ``[K, N]`` 原生布局的 int8 GEMM：每读一个字节做一次整数乘加。
//
// 一个 wavefront 在飞 ``64 * NTILE`` 个连续列字节，``NTILE = 4`` 就是 256 字节一条
// 连续段——D1 的落点，也是 GEMV 那边 442 GB/s 的图案。归约轴是 K，所以「每字节一次
// 乘加」这一点省不掉；这也正是它和 ``nk`` 家族的分工。
//
// @param A [M, K] int8 激活。
// @param W [K, N] int8 权重。
// @param C [M, N] int32 输出；``split > 1`` 时不写，改由 ``part`` 接收部分和。
// @param xs [M] fp32 每 token 激活 scale。
// @param ws [N] fp32 每输出列权重 scale。
// @param out 融合输出；``MODE == kOutNone`` 时为 nullptr。
// @param part [split, M, N] int32 部分和；``split == 1`` 时为 nullptr。
// @param M 行数。
// @param K 归约长度。
// @param N 输出列数。
// @param kchunk 每片的 k 长度，已按 4 对齐。
// @param split 沿 K 的切片数。
// @param ld_out 融合输出的行间距。
template <int MTILE, int NTILE, int MODE>
__global__ void kn_kernel(const int8_t* __restrict__ A, const int8_t* __restrict__ W,
                          int32_t* __restrict__ C, const float* __restrict__ xs,
                          const float* __restrict__ ws, void* __restrict__ out,
                          int32_t* __restrict__ part, int M, int K, int N, int kchunk,
                          int split, int ld_out) {
    const int n0 = (blockIdx.x * blockDim.x + threadIdx.x) * NTILE;
    if (n0 >= N) return;
    const int m0 = blockIdx.y * MTILE;
    const int mlast = min(m0 + MTILE, M);

    const int k0 = blockIdx.z * kchunk;
    const int k1 = min(K, k0 + kchunk);
    if (k0 >= k1) return;

    int32_t acc[MTILE][NTILE];
#pragma unroll
    for (int i = 0; i < MTILE; ++i) {
#pragma unroll
        for (int j = 0; j < NTILE; ++j) acc[i][j] = 0;
    }

    for (int k = k0; k < k1; ++k) {
        int8_t w[NTILE];
        cx_load_i8_row<NTILE>(W + (size_t)k * N + n0, w);
        int8_t av[MTILE];
#pragma unroll
        for (int i = 0; i < MTILE; ++i) {
            av[i] = (m0 + i < mlast) ? A[(size_t)(m0 + i) * K + k] : (int8_t)0;
        }
#pragma unroll
        for (int i = 0; i < MTILE; ++i) {
#pragma unroll
            for (int j = 0; j < NTILE; ++j) acc[i][j] += (int32_t)av[i] * (int32_t)w[j];
        }
    }

    if (split > 1) {
        const size_t plane = (size_t)M * N;
#pragma unroll
        for (int i = 0; i < MTILE; ++i) {
            if (m0 + i >= mlast) break;
#pragma unroll
            for (int j = 0; j < NTILE; ++j) {
                part[(size_t)blockIdx.z * plane + (size_t)(m0 + i) * N + n0 + j] = acc[i][j];
            }
        }
        return;
    }

#pragma unroll
    for (int i = 0; i < MTILE; ++i) {
        if (m0 + i >= mlast) break;
        const int m = m0 + i;
        if (MODE == kOutNone) {
            if (C != nullptr) {
#pragma unroll
                for (int j = 0; j < NTILE; ++j) C[(size_t)m * N + n0 + j] = acc[i][j];
            }
            continue;
        }
        const float xm = xs[m];
#pragma unroll
        for (int j = 0; j < NTILE; ++j) {
            cx_store_scaled<MODE>(out, (size_t)m * ld_out + n0 + j, acc[i][j], xm, ws[n0 + j]);
        }
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
template <int MODE>
__global__ void nk_kernel(const int8_t* __restrict__ A, const int8_t* __restrict__ WT,
                          int32_t* __restrict__ C, const float* __restrict__ xs,
                          const float* __restrict__ ws, void* __restrict__ out, int M, int K,
                          int N, int ld_out) {
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
        // 搬 B 块：``NK_BN`` 行 x ``NK_BK`` 字节，同一个形状。
        for (int idx = tid * 4; idx < NK_BELEMS; idx += NK_THREADS * 4) {
            const int r = idx / NK_BK;
            const int c = idx - r * NK_BK;
            const int gn = n_blk + r;
            const int32_t v = (gn < N && k0 + c + 3 < k4)
                                  ? *(const int32_t*)(WT + (size_t)gn * K + k0 + c)
                                  : 0;
            *(int32_t*)(btile + r * NK_LD + c) = v;
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
                bcol[j] = *(const int32_t*)(btile + (tx * NK_WN + j) * NK_LD + l);
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

// @brief split-K 归约 + epilogue。
//
// 每个线程负责一个输出元素，沿 split 累加 int32 部分和，再乘两个 scale 窄化一次。
// 只在 ``split > 1`` 时启动，所以它的启动开销不出现在单趟路径上。
//
// @param part [split, M, N] int32 部分和。
// @param xs [M] fp32 激活 scale。
// @param ws [N] fp32 权重 scale。
// @param C [M, N] int32 输出；``MODE == kOutNone`` 时写它。
// @param out 融合输出；``MODE == kOutNone`` 时为 nullptr。
// @param M/N 形状。
// @param split 切片数。
// @param ld_out 融合输出的行间距。
template <int MODE>
__global__ void reduce_kernel(const int32_t* __restrict__ part, const float* __restrict__ xs,
                              const float* __restrict__ ws, int32_t* __restrict__ C,
                              void* __restrict__ out, int M, int N, int split, int ld_out) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;
    if (col >= N) return;
    const size_t plane = (size_t)M * N;
    int32_t s = 0;
    for (int z = 0; z < split; ++z) s += part[(size_t)z * plane + (size_t)m * N + col];
    if (MODE == kOutNone) {
        C[(size_t)m * N + col] = s;
        return;
    }
    cx_store_scaled<MODE>(out, (size_t)m * ld_out + col, s, xs[m], ws[col]);
}

// @brief 跨层共享 grid：一次启动里跑多个形状的块。
//
// 块号先按各 group 的块数切开，再在 group 内按 (m 块, 列块) 展开。每个 group 带自己的
// 权重指针、行间距、列数和 scale，所以「把几份小权重拼进一次启动」不需要发明任何字节，
// 也不需要先把它们拷成一个大矩阵——那份额外拷贝正是这个想法本来要避开的成本。
//
// @param A [M, K] int8 激活，所有 group 共用。
// @param gt group 表，按值传入。
// @param C [M, total_n] int32 输出，各 group 按 ``c0`` 横向拼接。
// @param xs [M] fp32 激活 scale。
// @param out 融合输出；``MODE == kOutNone`` 时为 nullptr。
// @param M/K 形状。
// @param total_n 所有 group 列数之和，即输出行宽。
// @param mblk M 方向的块数。
// @param ld_out 融合输出的行间距。
template <int MTILE, int NTILE, int MODE>
__global__ void multi_kernel(const int8_t* __restrict__ A, GroupTable gt,
                             int32_t* __restrict__ C, const float* __restrict__ xs,
                             void* __restrict__ out, int M, int K, int total_n, int mblk,
                             int ld_out) {
    const int per_blk = blockDim.x * NTILE;
    const int m0 = (blockIdx.y % mblk) * MTILE;
    const int mlast = min(m0 + MTILE, M);
    int bid = blockIdx.x;

    // 块号 -> group。group 数不超过 8，线性扫描比一张描述符表便宜。
    int gi = 0;
    int nblk = 0;
    for (; gi < gt.ng; ++gi) {
        nblk = (gt.g[gi].n + per_blk - 1) / per_blk;
        if (bid < nblk) break;
        bid -= nblk;
    }
    if (gi >= gt.ng) return;

    const int n0 = bid * per_blk + threadIdx.x * NTILE;
    if (n0 >= gt.g[gi].n) return;
    const GroupArg gr = gt.g[gi];

    int32_t acc[MTILE][NTILE];
#pragma unroll
    for (int i = 0; i < MTILE; ++i) {
#pragma unroll
        for (int j = 0; j < NTILE; ++j) acc[i][j] = 0;
    }

    for (int k = 0; k < K; ++k) {
        int8_t w[NTILE];
        cx_load_i8_row<NTILE>(gr.w + (size_t)k * gr.ld + n0, w);
        int8_t av[MTILE];
#pragma unroll
        for (int i = 0; i < MTILE; ++i) {
            av[i] = (m0 + i < mlast) ? A[(size_t)(m0 + i) * K + k] : (int8_t)0;
        }
#pragma unroll
        for (int i = 0; i < MTILE; ++i) {
#pragma unroll
            for (int j = 0; j < NTILE; ++j) acc[i][j] += (int32_t)av[i] * (int32_t)w[j];
        }
    }

#pragma unroll
    for (int i = 0; i < MTILE; ++i) {
        if (m0 + i >= mlast) break;
        const int m = m0 + i;
        if (MODE == kOutNone) {
            if (C != nullptr) {
#pragma unroll
                for (int j = 0; j < NTILE; ++j) {
                    C[(size_t)m * total_n + gr.c0 + n0 + j] = acc[i][j];
                }
            }
            continue;
        }
        const float xm = xs[m];
#pragma unroll
        for (int j = 0; j < NTILE; ++j) {
            cx_store_scaled<MODE>(out, (size_t)m * ld_out + gr.c0 + n0 + j, acc[i][j], xm,
                                  gr.ws[n0 + j]);
        }
    }
}

// @brief 与 GEMM 同一套寻址的纯读内核：把权重读一遍，不做点积。
//
// D1 的对照项。GEMM 的实测带宽必须和**同一几何下**的纯读上限比，才能说清差额是算术
// 的还是访存的；拿别的几何的纯读数字当上限，会把两次测量的形状差异算成内核缺陷
// （``GEMV_CEILING.md`` §1 里那条「1 字节 lane 比 16 字节还快」的假结论就是这么来的）。
//
// @param W [K, N] int8 权重。
// @param acc [nblk_n, nblk_k] 每块一个和，防止整段读被优化掉。
// @param N 列数。
// @param K 行数。
// @param krows 每块负责的 k 行数。
template <int NTILE>
__global__ void read_kernel(const int8_t* __restrict__ W, long long* __restrict__ acc, int N,
                            int K, int krows) {
    const int col = (blockIdx.x * blockDim.x + threadIdx.x) * NTILE;
    if (col >= N) return;
    const int k0 = blockIdx.y * krows;
    const int k1 = min(K, k0 + krows);
    long long s = 0;
    for (int k = k0; k < k1; ++k) {
        int8_t w[NTILE];
        cx_load_i8_row<NTILE>(W + (size_t)k * N + col, w);
#pragma unroll
        for (int j = 0; j < NTILE; ++j) s += (long long)w[j];
    }
    __shared__ long long red[kRedThreads];
    red[threadIdx.x] = s;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) acc[(size_t)blockIdx.y * gridDim.x + blockIdx.x] = red[0];
}

// @brief ``__dp4a`` 与整数乘加的吞吐微基准（每秒多少次 int8 乘加）。
//
// 这条数字决定手写 GEMM 的上限，而且它是**预填充**那条路线的判据：厂商库在本卡上跑到
// 49.07 TINT8-OPS/s，而 ``__dp4a`` 一条指令只做四次 int8 乘加。如果它的吞吐换算出的
// 峰值低于 49.07，"手写 dp4a 在预填充上追平库"就不成立，与实现好坏无关。
//
// 四条独立累加链是为了让延迟不成为瓶颈——一条链上连续写同一个累加器量到的是延迟。
//
// @param acc [块数] 每块一个和。
// @param iters 每线程循环次数。
template <int MODE>
__global__ void rate_kernel(long long* __restrict__ acc, int iters) {
    uint32_t a = 0x01020304u + threadIdx.x;
    uint32_t b = 0x05060708u + blockIdx.x;
    int32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    const int b0 = (int)(threadIdx.x & 7u) + 1;
    // ``__dp4a`` 的有符号版本签名是 ``(int, int, int)``；这里把载荷先落到 int32_t，避免
    // 无符号/有符号混用造成的重载歧义。
    const int32_t ai = (int32_t)a;
    const int32_t bi = (int32_t)b;
    for (int it = 0; it < iters; ++it) {
        if (MODE == 0) {
            a0 = __dp4a(ai, bi, a0);
            a1 = __dp4a(bi, ai, a1);
            a2 = __dp4a(ai + it, bi, a2);
            a3 = __dp4a(bi + it, ai, a3);
        } else {
            a0 += (a & 0xff) * b0;
            a1 += ((a >> 8) & 0xff) * b0;
            a2 += ((a >> 16) & 0xff) * b0;
            a3 += ((a >> 24) & 0xff) * b0;
        }
    }
    __shared__ long long red[kRedThreads];
    red[threadIdx.x] = (long long)a0 + a1 + a2 + a3;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) acc[blockIdx.x] = red[0];
}

// 启动器类型。调度表是数据而不是一串 if：模板组合有几十个，写成 switch 会掩盖「哪一格
// 真的跑过」这件事，而这张表恰好就是要进扫描表的东西。
using KnFn = void (*)(const int8_t*, const int8_t*, int32_t*, const float*, const float*, void*,
                      int32_t*, int, int, int, int, int, int, cudaStream_t);

template <int MTILE, int NTILE, int MODE>
void kn_launch(const int8_t* A, const int8_t* W, int32_t* C, const float* xs, const float* ws,
               void* out, int32_t* part, int M, int K, int N, int kchunk, int split,
               int ld_out, cudaStream_t stream) {
    constexpr int kThreads = 128;
    const dim3 grid((N + kThreads * NTILE - 1) / (kThreads * NTILE), (M + MTILE - 1) / MTILE,
                    split);
    kn_kernel<MTILE, NTILE, MODE>
        <<<grid, kThreads, 0, stream>>>(A, W, C, xs, ws, out, part, M, K, N, kchunk, split,
                                        ld_out);
}

using NkFn = void (*)(const int8_t*, const int8_t*, int32_t*, const float*, const float*, void*,
                      int, int, int, int, cudaStream_t);

template <int MODE>
void nk_launch(const int8_t* A, const int8_t* WT, int32_t* C, const float* xs, const float* ws,
               void* out, int M, int K, int N, int ld_out, cudaStream_t stream) {
    const dim3 grid((N + NK_BN - 1) / NK_BN, (M + NK_BM - 1) / NK_BM);
    const dim3 block(NK_TX, NK_TY);
    nk_kernel<MODE><<<grid, block, 0, stream>>>(A, WT, C, xs, ws, out, M, K, N, ld_out);
}

// @brief 按 ``MODE`` 取 ``kn`` 启动器。
//
// @param mtile 每线程负责的 M 行数。
// @param ntile 每线程负责的相邻列数。
// @param mode 输出模式。
// @return 启动器；组合不受支持时返回 nullptr，由主机侧 TORCH_CHECK 拦下。
KnFn pick_kn(int mtile, int ntile, int mode) {
#define CX_KN_ROW(MT, NT)                                                        \
    if (mtile == (MT) && ntile == (NT)) {                                        \
        if (mode == kOutBf16) return &kn_launch<MT, NT, kOutBf16>;               \
        if (mode == kOutF32) return &kn_launch<MT, NT, kOutF32>;                 \
        return &kn_launch<MT, NT, kOutNone>;                                     \
    }
    CX_KN_ROW(1, 1) CX_KN_ROW(1, 2) CX_KN_ROW(1, 4)
    CX_KN_ROW(2, 1) CX_KN_ROW(2, 2) CX_KN_ROW(2, 4)
    CX_KN_ROW(4, 1) CX_KN_ROW(4, 2) CX_KN_ROW(4, 4)
#undef CX_KN_ROW
    return nullptr;
}

//: ``mtile`` 的合法值。每线程 4 行以上寄存器就压过收益了。
bool cx_mtile_ok(int v) { return v == 1 || v == 2 || v == 4; }

//: ``ntile`` 的合法值。8 和 16 在 GEMV 上实测更差（372-374 GB/s 对 442），这里不开。
bool cx_ntile_ok(int v) { return v == 1 || v == 2 || v == 4; }

// @brief 从输出张量读出融合模式。
//
// @param out 输出张量，可为空。
// @return ``kOutNone`` / ``kOutBf16`` / ``kOutF32``。
int out_mode_of(const c10::optional<torch::Tensor>& out) {
    if (!out.has_value()) return kOutNone;
    if (out->scalar_type() == torch::kBFloat16) return kOutBf16;
    return kOutF32;
}

//: 一组选定的几何。
struct Geom {
    int mtile;
    int ntile;
    int threads;
    int split;
    int kchunk;
};

// @brief 按 M、N 和调用方的覆盖值定下一组几何。
//
// 顺序有讲究：先定 ``ntile``（受 ``N % 4`` 约束），再定 ``mtile``（受 M 约束），最后
// 用 ``split`` 把块数补到这张卡吃得满的量级——16 个 SM 要一百多个块才有波次。
//
// @param M 行数。
// @param K 归约长度。
// @param N 输出列数。
// @param mtile/ntile/threads/split 调用方覆盖值，``<= 0`` 表示自己选。
// @return 选定的几何。
Geom cx_pick_geom(int M, int K, int N, int mtile, int ntile, int threads, int split) {
    Geom g;
    g.ntile = ntile > 0 ? ntile : (N % 4 == 0 ? 4 : (N % 2 == 0 ? 2 : 1));
    g.mtile = mtile > 0 ? mtile : (M >= 4 ? 4 : (M >= 2 ? 2 : 1));
    g.threads = threads > 0 ? threads : 128;

    if (split > 0) {
        g.split = split;
    } else {
        const int nblk = (N + g.threads - 1) / g.threads;
        const int mblk = (M + g.mtile - 1) / g.mtile;
        // 目标块数 256：16 个 SM 上十几个 wave，够把启动和尾部的开销摊掉。
        // 这里必须用 std:: 前缀：这是主机代码，``min``/``max`` 的那些无前缀版本只在
        // 设备编译单元里可见，直接写会报 no matching function。
        int want = (256 + std::max(1, nblk * mblk) - 1) / std::max(1, nblk * mblk);
        want = std::min(want, 32);
        want = std::min(want, std::max(1, K / 64));
        g.split = std::max(1, want);
    }
    // 每片按 4 对齐再回算真实片数：``__dp4a`` 的粒度是 4 个 k，片长不齐会让某一片落到
    // 别的片的 k 上——结果是对的，但边界错位，而这种错在容差里看不出来。
    const int raw = (K + g.split - 1) / g.split;
    g.kchunk = ((raw + 3) / 4) * 4;
    g.split = (K + g.kchunk - 1) / g.kchunk;
    return g;
}

}  // namespace

// 主机侧包装层。检查的是内核索引不了的布局，不是用户输入是否合理：内核不做边界检查，
// 所以错误的布局必须在这里失败，而不是读穿缓冲区。

torch::Tensor w8a8_dp4a(torch::Tensor a, torch::Tensor w, torch::Tensor c,
                        c10::optional<torch::Tensor> out, torch::Tensor xs, torch::Tensor ws,
                        c10::optional<torch::Tensor> partial, int64_t mtile, int64_t ntile,
                        int64_t ktile, int64_t threads, int64_t split) {
    TORCH_CHECK(a.dim() == 2 && w.dim() == 2 && c.dim() == 2, "A, W and C must all be 2-D");
    TORCH_CHECK(a.scalar_type() == torch::kChar && w.scalar_type() == torch::kChar,
                "operands must be int8");
    TORCH_CHECK(c.scalar_type() == torch::kInt, "accumulator must be int32");
    TORCH_CHECK(a.is_contiguous() && w.is_contiguous() && c.is_contiguous(),
                "contiguous operands required");
    const int M = (int)a.size(0);
    const int K = (int)a.size(1);
    const int N = (int)w.size(1);
    TORCH_CHECK((int)w.size(0) == K, "K mismatch: A ", a.sizes(), " W ", w.sizes());
    TORCH_CHECK((int)c.size(0) == M && (int)c.size(1) == N, "C ", c.sizes(), " does not match ",
                M, "x", N);
    TORCH_CHECK(xs.scalar_type() == torch::kFloat && xs.numel() == M,
                "xs must be fp32 with one value per row");
    TORCH_CHECK(ws.scalar_type() == torch::kFloat && ws.numel() == N,
                "ws must be fp32 with one value per column");
    if (out.has_value()) {
        TORCH_CHECK(out->scalar_type() == torch::kBFloat16 || out->scalar_type() == torch::kFloat,
                    "fused output must be bf16 or fp32, got ", out->scalar_type());
    }
    (void)ktile;

    Geom g = cx_pick_geom(M, K, N, (int)mtile, (int)ntile, (int)threads, (int)split);
    TORCH_CHECK(cx_mtile_ok(g.mtile), "mtile ", g.mtile, " is not supported (1, 2 or 4)");
    TORCH_CHECK(cx_ntile_ok(g.ntile), "ntile ", g.ntile, " is not supported (1, 2 or 4)");
    if (N % g.ntile != 0) g.ntile = 1;

    const int mode = out_mode_of(out);
    const int ld = out.has_value() ? (int)out->stride(0) : N;
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    int32_t* part = nullptr;
    if (g.split > 1 && out.has_value()) {
        TORCH_CHECK(partial.has_value(), "split > 1 with a fused output needs a partial buffer");
        TORCH_CHECK(partial->scalar_type() == torch::kInt && partial->is_contiguous(),
                    "partial must be contiguous int32");
        TORCH_CHECK(partial->numel() >= (int64_t)g.split * M * N,
                    "partial must hold split x M x N = ", g.split, " x ", M, " x ", N);
        part = partial->data_ptr<int32_t>();
    }

    KnFn kn = pick_kn(g.mtile, g.ntile, mode);
    TORCH_CHECK(kn != nullptr, "no kernel for mtile=", g.mtile, " ntile=", g.ntile);
    void* outp = out.has_value() ? (void*)out->data_ptr() : nullptr;
    kn(a.data_ptr<int8_t>(), w.data_ptr<int8_t>(), c.data_ptr<int32_t>(),
       xs.data_ptr<float>(), ws.data_ptr<float>(), outp, part, M, K, N, g.kchunk, g.split, ld,
       stream);
    if (g.split > 1 && outp != nullptr) {
        const dim3 rgrid((N + 255) / 256, M);
        if (mode == kOutBf16) {
            reduce_kernel<kOutBf16><<<rgrid, 256, 0, stream>>>(
                part, xs.data_ptr<float>(), ws.data_ptr<float>(), c.data_ptr<int32_t>(),
                outp, M, N, g.split, ld);
        } else if (mode == kOutF32) {
            reduce_kernel<kOutF32><<<rgrid, 256, 0, stream>>>(
                part, xs.data_ptr<float>(), ws.data_ptr<float>(), c.data_ptr<int32_t>(),
                outp, M, N, g.split, ld);
        } else {
            reduce_kernel<kOutNone><<<rgrid, 256, 0, stream>>>(
                part, xs.data_ptr<float>(), ws.data_ptr<float>(), c.data_ptr<int32_t>(),
                outp, M, N, g.split, ld);
        }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out.has_value() ? *out : c;
}

torch::Tensor w8a8_dp4a_nk(torch::Tensor a, torch::Tensor wt, torch::Tensor c,
                           c10::optional<torch::Tensor> out, torch::Tensor xs, torch::Tensor ws,
                           int64_t threads) {
    TORCH_CHECK(a.dim() == 2 && wt.dim() == 2 && c.dim() == 2, "A, WT and C must all be 2-D");
    TORCH_CHECK(a.scalar_type() == torch::kChar && wt.scalar_type() == torch::kChar,
                "operands must be int8");
    TORCH_CHECK(c.scalar_type() == torch::kInt, "accumulator must be int32");
    TORCH_CHECK(a.is_contiguous() && wt.is_contiguous() && c.is_contiguous(),
                "contiguous operands required");
    const int M = (int)a.size(0);
    const int K = (int)a.size(1);
    const int N = (int)wt.size(0);
    TORCH_CHECK((int)wt.size(1) == K, "WT must be [N, K]: got ", wt.sizes(), " for K = ", K);
    TORCH_CHECK((int)c.size(0) == M && (int)c.size(1) == N, "C ", c.sizes(), " must be [", M,
                ", ", N, "]");
    // ``__dp4a`` 一次归约 4 个相邻 k，K 不是 4 的倍数时尾部的处理会把下一列的字节算进来。
    TORCH_CHECK(K % 4 == 0, "the dp4a path needs K % 4 == 0, got K = ", K);
    TORCH_CHECK(xs.scalar_type() == torch::kFloat && xs.numel() == M,
                "xs must be fp32 with one value per row");
    TORCH_CHECK(ws.scalar_type() == torch::kFloat && ws.numel() == N,
                "ws must be fp32 with one value per column");
    // 块形状由寄存器分决定（``NK_TX x NK_TY``），不开放覆盖：改线程数就得改每个线程
    // 的累加器形状，那是一次重新设计而不是一个参数。传了别的值就拒绝，而不是忽略。
    TORCH_CHECK(threads <= 0 || threads == NK_THREADS, "the dp4a path is fixed at ",
                NK_THREADS, " threads per block");
    if (out.has_value()) {
        TORCH_CHECK(out->scalar_type() == torch::kBFloat16 || out->scalar_type() == torch::kFloat,
                    "fused output must be bf16 or fp32");
    }

    const int mode = out_mode_of(out);
    const int ld = out.has_value() ? (int)out->stride(0) : N;
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    void* outp = out.has_value() ? (void*)out->data_ptr() : nullptr;
    if (mode == kOutBf16) {
        nk_launch<kOutBf16>(a.data_ptr<int8_t>(), wt.data_ptr<int8_t>(),
                            c.data_ptr<int32_t>(), xs.data_ptr<float>(),
                            ws.data_ptr<float>(), outp, M, K, N, ld, stream);
    } else if (mode == kOutF32) {
        nk_launch<kOutF32>(a.data_ptr<int8_t>(), wt.data_ptr<int8_t>(),
                           c.data_ptr<int32_t>(), xs.data_ptr<float>(), ws.data_ptr<float>(),
                           outp, M, K, N, ld, stream);
    } else {
        nk_launch<kOutNone>(a.data_ptr<int8_t>(), wt.data_ptr<int8_t>(),
                            c.data_ptr<int32_t>(), xs.data_ptr<float>(),
                            ws.data_ptr<float>(), outp, M, K, N, ld, stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out.has_value() ? *out : c;
}

torch::Tensor w8a8_dp4a_groups(torch::Tensor a, std::vector<torch::Tensor> group_w,
                               std::vector<torch::Tensor> group_ws, torch::Tensor c,
                               c10::optional<torch::Tensor> out, torch::Tensor xs,
                               int64_t mtile, int64_t ntile, int64_t ktile, int64_t threads,
                               int64_t split) {
    TORCH_CHECK(a.dim() == 2 && a.scalar_type() == torch::kChar, "A must be 2-D int8");
    TORCH_CHECK(a.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(!group_w.empty() && group_w.size() == group_ws.size(),
                "need one scale vector per weight");
    TORCH_CHECK((int)group_w.size() <= kMaxGroups, "at most ", (int)kMaxGroups, " groups");
    (void)ktile;
    (void)split;
    const int M = (int)a.size(0);
    const int K = (int)a.size(1);

    GroupTable gt{};
    gt.ng = (int)group_w.size();
    int total_n = 0;
    int max_n = 0;
    for (int i = 0; i < gt.ng; ++i) {
        const torch::Tensor& wi = group_w[i];
        TORCH_CHECK(wi.dim() == 2 && wi.scalar_type() == torch::kChar && wi.is_contiguous(),
                    "group weights must be 2-D contiguous int8");
        TORCH_CHECK((int)wi.size(0) == K, "group ", i, " has K = ", wi.size(0), ", A has ", K);
        TORCH_CHECK(group_ws[i].scalar_type() == torch::kFloat &&
                        group_ws[i].numel() == wi.size(1),
                    "group ", i, " needs one fp32 scale per column");
        gt.g[i].w = wi.data_ptr<int8_t>();
        gt.g[i].ws = group_ws[i].data_ptr<float>();
        gt.g[i].n = (int)wi.size(1);
        gt.g[i].ld = (int)wi.size(1);
        gt.g[i].c0 = total_n;
        total_n += gt.g[i].n;
        max_n = std::max(max_n, gt.g[i].n);
    }
    TORCH_CHECK(c.dim() == 2 && c.scalar_type() == torch::kInt && c.is_contiguous(),
                "C must be 2-D contiguous int32");
    TORCH_CHECK((int)c.size(0) == M && (int)c.size(1) == total_n, "C ", c.sizes(), " must be [",
                M, ", ", total_n, "]");
    if (out.has_value()) {
        TORCH_CHECK(out->scalar_type() == torch::kBFloat16 || out->scalar_type() == torch::kFloat,
                    "fused output must be bf16 or fp32");
    }

    Geom g = cx_pick_geom(M, K, total_n, (int)mtile, (int)ntile, (int)threads, 1);
    TORCH_CHECK(cx_mtile_ok(g.mtile) && cx_ntile_ok(g.ntile), "unsupported group geometry");
    if (max_n % g.ntile != 0) g.ntile = 1;
    const int mode = out_mode_of(out);
    const int ld = out.has_value() ? (int)out->stride(0) : total_n;
    const int per_blk = g.threads * g.ntile;
    const int mblk = (M + g.mtile - 1) / g.mtile;

    int nblk = 0;
    for (int i = 0; i < gt.ng; ++i) nblk += (gt.g[i].n + per_blk - 1) / per_blk;
    const dim3 grid(nblk, mblk);
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    void* outp = out.has_value() ? (void*)out->data_ptr() : nullptr;
    int32_t* Cp = c.data_ptr<int32_t>();
    const int8_t* Ap = a.data_ptr<int8_t>();
    const float* xsp = xs.data_ptr<float>();

#define CX_MULTI(MT, NT, MD)                                                              \
    multi_kernel<MT, NT, MD><<<grid, g.threads, 0, stream>>>(Ap, gt, Cp, xsp, outp, M, K,  \
                                                             total_n, mblk, ld)
#define CX_MODE(MT, NT)                                      \
    do {                                                     \
        if (mode == kOutBf16) CX_MULTI(MT, NT, kOutBf16);    \
        else if (mode == kOutF32) CX_MULTI(MT, NT, kOutF32); \
        else CX_MULTI(MT, NT, kOutNone);                     \
    } while (0)
#define CX_GEO                                                \
    do {                                                      \
        if (g.mtile == 1 && g.ntile == 1) CX_MODE(1, 1);       \
        else if (g.mtile == 1 && g.ntile == 2) CX_MODE(1, 2);  \
        else if (g.mtile == 1 && g.ntile == 4) CX_MODE(1, 4);  \
        else if (g.mtile == 2 && g.ntile == 1) CX_MODE(2, 1);  \
        else if (g.mtile == 2 && g.ntile == 2) CX_MODE(2, 2);  \
        else if (g.mtile == 2 && g.ntile == 4) CX_MODE(2, 4);  \
        else if (g.mtile == 4 && g.ntile == 1) CX_MODE(4, 1);  \
        else if (g.mtile == 4 && g.ntile == 2) CX_MODE(4, 2);  \
        else CX_MODE(4, 4);                                    \
    } while (0)
    CX_GEO;
#undef CX_GEO
#undef CX_MODE
#undef CX_MULTI
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out.has_value() ? *out : c;
}

torch::Tensor w8a8_dp4a_read(torch::Tensor w, torch::Tensor out, int64_t ntile,
                             int64_t threads, int64_t blocks) {
    TORCH_CHECK(w.dim() == 2 && w.scalar_type() == torch::kChar && w.is_contiguous(),
                "W must be 2-D contiguous int8");
    const int K = (int)w.size(0);
    const int N = (int)w.size(1);
    int nt = (int)ntile;
    TORCH_CHECK(cx_ntile_ok(nt), "ntile ", nt, " is not supported (1, 2 or 4)");
    if (N % nt != 0) nt = 1;
    const int th = (int)threads;
    TORCH_CHECK(th > 0 && th <= kRedThreads, "threads must be in [1, ", kRedThreads, "]");
    const int nblk_n = (N + th * nt - 1) / (th * nt);
    int nb = (int)blocks;
    if (nb <= 0) nb = 16 * 32;
    const int nblk_k = std::max(1, nb / std::max(1, nblk_n));
    const int krows = (K + nblk_k - 1) / nblk_k;
    const int real_nblk_k = (K + krows - 1) / krows;
    TORCH_CHECK(out.scalar_type() == torch::kLong && out.is_contiguous(), "out must be int64");
    TORCH_CHECK(out.numel() >= (int64_t)nblk_n * real_nblk_k, "out must hold ",
                nblk_n * real_nblk_k, " block sums");
    const dim3 grid(nblk_n, real_nblk_k);
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    long long* ap = (long long*)out.data_ptr<int64_t>();
    const int8_t* Wp = w.data_ptr<int8_t>();
    if (nt == 1) read_kernel<1><<<grid, th, 0, stream>>>(Wp, ap, N, K, krows);
    else if (nt == 2) read_kernel<2><<<grid, th, 0, stream>>>(Wp, ap, N, K, krows);
    else read_kernel<4><<<grid, th, 0, stream>>>(Wp, ap, N, K, krows);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor w8a8_dp4a_rate(torch::Tensor out, int64_t iters, int64_t mode, int64_t blocks) {
    TORCH_CHECK(out.scalar_type() == torch::kLong, "out must be int64");
    int nb = (int)blocks;
    if (nb <= 0) nb = 16 * 32;
    TORCH_CHECK(out.numel() >= nb, "out must hold ", nb, " block sums");
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    long long* ap = (long long*)out.data_ptr<int64_t>();
    if (mode == 0) rate_kernel<0><<<dim3(nb), kRedThreads, 0, stream>>>(ap, (int)iters);
    else rate_kernel<1><<<dim3(nb), kRedThreads, 0, stream>>>(ap, (int)iters);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
