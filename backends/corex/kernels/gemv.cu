// @file gemv.cu
// @brief ``[K, N]`` 投影内核：这张卡的解码期权重读取需要的东西。
//
// Module: coffin/backends/corex/kernels。绑定成 ``gemv_kn`` / ``gemv_kn_split`` /
// ``gemv_kn_m``，只由 ``backends/corex/__init__.py`` 调用。
//
// 为什么一个输出列一个线程：权重是行主序的 ``[K, N]``，所以第 ``n`` 列是一行里带步长的
// 那根轴。持有第 ``n`` 列的线程按递增的 ``k`` 读 ``wt[k, n]``，而 64 条连续通道（本部件的
// wavefront）覆盖一行的 64 个连续列：每个 k 步正好一次合并的 128 字节事务。这就是
// ``[K, N]`` 在这里能到 168-361 GB/s、而库的 ``[N, K]`` 形式实测只有 19-30 GB/s 的全部
// 原因（COREX_PORT.md §0）。相反的分配方式——每个权重行一条通道、沿 K 归约——无法合并，
// 那正是库的做法。
//
// 改成每个线程持四个 *相邻* 列（``x4`` 变体）在本部件上反而更慢：实测慢于一个线程一列，
// 说明这里每次加载多读的字节不是瓶颈，而它省下的循环开销也不是代价所在。实测有帮助的是
// 四层深的 K 展开（``UNROLL``）。
//
// Constraints:
//   * 没有跨通道归约，所以 ``warpSize == 64`` 在这份文件里不参与。
//     每条通道自己产出完整的输出元素。
//   * k 循环被拆成编译期展开的主体加标量尾巴，所以任何 ``K`` 都正确，
//     而只有 ``K % UNROLL == 0`` 才走宽路径。
//   * 在 ``c10::cuda::getCurrentCUDAStream()`` 上启动。在 legacy 默认流上跑的内核不属于
//     ``torch.cuda.graph`` 的捕获范围，会从被捕获的图里丢掉并重放陈旧输出——那是静默的
//     正确性 bug，不是变慢。
//   * 累加用 fp32，输出按激活 dtype 写出。参考后端也舍入到激活 dtype，所以两边在 bf16
//     分辨率上一致；把累加器留在 fp32，才不会让长的 K 丢掉小量级的项。
//   * Split-K 只为窄 ``N`` 存在。它用第二次启动和一个分片缓冲区换取占用率，而
//     ``ceil(N / threads)`` 不足一个 wave 时占用率才是约束；在宽 ``N`` 上它只会给一个
//     本来就被带宽限制的形状增加流量。

#include "corex_ops.h"
#include "corex_scalar.h"

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

namespace {

// @brief 一个输出列在 K 区间上的点积，按 ``U`` 路展开。
//
// ``U`` 次加载都在第一次 FMA 之前发出，正是它给 k 循环足够的内存级并行，去掩盖步长为
// ``N`` 的访问模式。
//
// @param wt [K, N] 权重，行主序。
// @param x [K] 激活。
// @param n 本线程持有的输出列。
// @param N ``wt`` 的行间距。
// @param k0 区间首元素，含。
// @param k1 区间末元素，不含。
// @return fp32 累加的点积。
template <typename WT, typename XT, int U>
__device__ __forceinline__ float dot_kn(const WT* __restrict__ wt,
                                        const XT* __restrict__ x,
                                        int n, int N, int k0, int k1) {
    float acc = 0.f;
    int k = k0;
    for (; k + U <= k1; k += U) {
#pragma unroll
        for (int u = 0; u < U; ++u) {
            acc += CxScalar<WT>::to_float(wt[(size_t)(k + u) * N + n])
                 * CxScalar<XT>::to_float(x[k + u]);
        }
    }
    for (; k < k1; ++k) {
        acc += CxScalar<WT>::to_float(wt[(size_t)k * N + n])
             * CxScalar<XT>::to_float(x[k]);
    }
    return acc;
}

// @brief 一行激活对整份权重：``y[n] = dot(wt[:, n], x)``。
template <typename WT, typename XT, int U>
__global__ void gemv_kn_kernel(const WT* __restrict__ wt,
                               const XT* __restrict__ x, XT* __restrict__ y,
                               int N, int K) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    y[n] = CxScalar<XT>::from_float(dot_kn<WT, XT, U>(wt, x, n, N, 0, K));
}

// @brief 同一乘积的一个 K 区间，写进本分片在分片数组里的那一格。
//
// 每个分片都写自己那一格的全部 ``N`` 列，所以分片缓冲区不需要初始化，而归约顺序由 ``s``
// 决定，而不是由哪个块碰巧先跑完决定。
template <typename WT, typename XT, int U>
__global__ void gemv_kn_split_kernel(const WT* __restrict__ wt,
                                     const XT* __restrict__ x, float* __restrict__ part,
                                     int N, int K, int split, int per) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const int s = blockIdx.y;
    const int k0 = s * per;
    const int k1 = min(K, k0 + per);
    part[(size_t)s * N + n] = dot_kn<WT, XT, U>(wt, x, n, N, k0, k1);
}

// @brief 把分片部分和加进输出，每列一个线程。
template <typename XT>
__global__ void gemv_reduce_kernel(const float* __restrict__ part,
                                   XT* __restrict__ y, int N, int split) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float acc = 0.f;
    for (int s = 0; s < split; ++s) acc += part[(size_t)s * N + n];
    y[n] = CxScalar<XT>::from_float(acc);
}

// W8A16 家族：索引方式与上面三个内核完全相同，只是权重是 int8、外加一个每列的 fp32 scale。
//
// 为什么单独一个内核而不是 ``gemv_kn_kernel`` 的 ``int8_t`` 实例化：那个 scale。它在整个
// ``K`` 区间求和之后、每个输出列施加一次——逐元素施加会得到不同的舍入，更糟的是会诱使调用方
// 把它折进激活，那样结果会被缩放两次。让入口保持独立，这个契约就不可能被误满足。
//
// 每个权重读的字节数从两字节降到一字节，这就是全部意义：解码受带宽限制，而投影占了这一步
// 的大部分。

// @brief 一行激活对 int8 ``[K, N]`` 权重，每列一个线程。
template <typename XT, int U>
__global__ void gemv_kn_i8_kernel(const int8_t* __restrict__ wt,
                                  const XT* __restrict__ x, XT* __restrict__ y,
                                  const float* __restrict__ scale, int N, int K) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    y[n] = CxScalar<XT>::from_float(scale[n] * dot_kn<int8_t, XT, U>(wt, x, n, N, 0, K));
}

// @brief 同一乘积的一个 K 区间，写进本分片在分片数组里的那一格。
template <typename XT, int U>
__global__ void gemv_kn_i8_split_kernel(const int8_t* __restrict__ wt,
                                        const XT* __restrict__ x,
                                        float* __restrict__ part,
                                        int N, int K, int split, int per) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const int s = blockIdx.y;
    const int k0 = s * per;
    const int k1 = min(K, k0 + per);
    part[(size_t)s * N + n] = dot_kn<int8_t, XT, U>(wt, x, n, N, k0, k1);
}

// @brief 把分片部分和相加并施加列 scale，每列一个线程。
template <typename XT>
__global__ void gemv_reduce_i8_kernel(const float* __restrict__ part,
                                      XT* __restrict__ y,
                                      const float* __restrict__ scale,
                                      int N, int split) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float acc = 0.f;
    for (int s = 0; s < split; ++s) acc += part[(size_t)s * N + n];
    y[n] = CxScalar<XT>::from_float(scale[n] * acc);
}

// int8 路径上每个线程算四个输出列。
//
// 上面的标量内核每个权重元素读一个字节，于是一个 wavefront 的 64 条通道每个 k 步发 64 次
// 单字节加载——一条 128 字节行里的 64 个字节，却要 64 个请求。实测这使 int8 内核在同一形状
// 上被压在 235-262 GB/s，而 bf16 内核有 314 GB/s，这就是为什么把字节数减半只换来 1.5 倍
// 而不是 2 倍。
//
// 把四个相邻 *列* 当成一次 4 字节加载就解决了：列在 ``[K, N]`` 的一行内是连续的，所以
// ``wt[k*N + n .. n+3]`` 是一个对齐的字，一个 wavefront 于是覆盖 256 个连续字节，加载次数
// 降到四分之一。四个累加器是彼此独立的和，所以算术没变——只有 scale 的施加挪到最后、一次
// 管四列。
//
// 这要求 ``N % 4 == 0``；主机侧会检查，否则退回标量内核，这样一个宽度不是四的倍数的模型
// 也不会破坏契约。
template <typename XT, int U>
__global__ void gemv_kn_i8x4_kernel(const int8_t* __restrict__ wt,
                                    const XT* __restrict__ x, XT* __restrict__ y,
                                    const float* __restrict__ scale, int N, int K) {
    const int n0 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (n0 >= N) return;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    int k = 0;
    for (; k + U <= K; k += U) {
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int32_t word =
                *reinterpret_cast<const int32_t*>(&wt[(size_t)(k + u) * N + n0]);
            const float xv = CxScalar<XT>::to_float(x[k + u]);
            a0 = fmaf((float)(int8_t)(word & 0xff), xv, a0);
            a1 = fmaf((float)(int8_t)((word >> 8) & 0xff), xv, a1);
            a2 = fmaf((float)(int8_t)((word >> 16) & 0xff), xv, a2);
            a3 = fmaf((float)(int8_t)((word >> 24) & 0xff), xv, a3);
        }
    }
    for (; k < K; ++k) {
        const int32_t word = *reinterpret_cast<const int32_t*>(&wt[(size_t)k * N + n0]);
        const float xv = CxScalar<XT>::to_float(x[k]);
        a0 = fmaf((float)(int8_t)(word & 0xff), xv, a0);
        a1 = fmaf((float)(int8_t)((word >> 8) & 0xff), xv, a1);
        a2 = fmaf((float)(int8_t)((word >> 16) & 0xff), xv, a2);
        a3 = fmaf((float)(int8_t)((word >> 24) & 0xff), xv, a3);
    }
    y[n0] = CxScalar<XT>::from_float(scale[n0] * a0);
    y[n0 + 1] = CxScalar<XT>::from_float(scale[n0 + 1] * a1);
    y[n0 + 2] = CxScalar<XT>::from_float(scale[n0 + 2] * a2);
    y[n0 + 3] = CxScalar<XT>::from_float(scale[n0 + 3] * a3);
}

// @brief 同一四宽乘积的一个 K 区间，写进本分片的部分和。
template <typename XT, int U>
__global__ void gemv_kn_i8x4_split_kernel(const int8_t* __restrict__ wt,
                                          const XT* __restrict__ x,
                                          float* __restrict__ part,
                                          int N, int K, int split, int per) {
    const int n0 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (n0 >= N) return;
    const int s = blockIdx.y;
    const int k0 = s * per;
    const int k1 = min(K, k0 + per);
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    int k = k0;
    for (; k + U <= k1; k += U) {
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int32_t word =
                *reinterpret_cast<const int32_t*>(&wt[(size_t)(k + u) * N + n0]);
            const float xv = CxScalar<XT>::to_float(x[k + u]);
            a0 = fmaf((float)(int8_t)(word & 0xff), xv, a0);
            a1 = fmaf((float)(int8_t)((word >> 8) & 0xff), xv, a1);
            a2 = fmaf((float)(int8_t)((word >> 16) & 0xff), xv, a2);
            a3 = fmaf((float)(int8_t)((word >> 24) & 0xff), xv, a3);
        }
    }
    for (; k < k1; ++k) {
        const int32_t word = *reinterpret_cast<const int32_t*>(&wt[(size_t)k * N + n0]);
        const float xv = CxScalar<XT>::to_float(x[k]);
        a0 = fmaf((float)(int8_t)(word & 0xff), xv, a0);
        a1 = fmaf((float)(int8_t)((word >> 8) & 0xff), xv, a1);
        a2 = fmaf((float)(int8_t)((word >> 16) & 0xff), xv, a2);
        a3 = fmaf((float)(int8_t)((word >> 24) & 0xff), xv, a3);
    }
    float* col = part + (size_t)s * N + n0;
    col[0] = a0;
    col[1] = a1;
    col[2] = a2;
    col[3] = a3;
}

// @brief ``W`` 宽的列加载用哪种整数字来执行。
//
// ``W`` 是一个线程持有的相邻输出列数，同时也是它每个 k 步读取的权重字节数。这些字节在
// ``[K, N]`` 的一行内是连续的，所以只要行首对齐它们就是一次加载——而 ``N % W == 0`` 保证
// 了这一点，因为每一行都从缓冲区里 ``k * N`` 字节处开始。
//
// 决定事务效率的宽度，来自本部件 64 通道 wavefront 对 128 字节行的关系：W=1 时每个
// wavefront 只有 64 字节在飞（半行），W=2 正好 128，W=4 是 256。读得更宽同时会把线程数
// 除以 ``W``，而在 16 个 SM 的部件上线程是稀缺资源，所以这个取舍往哪边倒要靠实测而不是
// 规则——见 ``I8_WIDE_MIN_N`` 和 ``tune_gemv_i8_wide.py`` 里的扫描。
//
// ``W = 16`` 是两次 64 位加载，因为没有 16 字节的标量整数。把它列进来只是因为模板让它免费；
// 它到底会不会被跑到，由扫描决定。
template <int W>
struct CxWideWord;

// W = 1 是标量路径，一次读一个字节。批处理内核也会实例化它（窄投影会取这个宽度），所以
// 这个类型必须存在，即使单行内核从不经这个辅助结构走到它。
template <>
struct CxWideWord<1> {
    using type = uint8_t;
    static constexpr int kLoads = 1;
};

template <>
struct CxWideWord<2> {
    using type = uint16_t;
    static constexpr int kLoads = 1;
};

template <>
struct CxWideWord<4> {
    using type = uint32_t;
    static constexpr int kLoads = 1;
};

template <>
struct CxWideWord<8> {
    using type = uint64_t;
    static constexpr int kLoads = 1;
};

template <>
struct CxWideWord<16> {
    using type = uint64_t;
    static constexpr int kLoads = 2;
};

// @brief 从 ``p`` 开始读 ``W`` 个相邻的权重字节。
//
// 这次加载就是宽内核的全部意义：``W`` 列共用一次事务，而不是 ``W`` 次。字节靠移位取出，
// 所以这个字必须是无符号的——对有符号字做算术右移会把高位字节符号扩展进下一个字节。
//
// @param p 本线程列组的第一个权重字节；按 ``W`` 对齐。
// @param out 接收按列升序排列的 ``W`` 个有符号字节。
template <int W>
__device__ __forceinline__ void load_i8_row(const int8_t* __restrict__ p,
                                            int8_t (&out)[W]) {
    using LT = typename CxWideWord<W>::type;
    constexpr int kLoads = CxWideWord<W>::kLoads;
    constexpr int kPer = (int)sizeof(LT);
#pragma unroll
    for (int i = 0; i < kLoads; ++i) {
        const LT word = *reinterpret_cast<const LT*>(p + i * kPer);
#pragma unroll
        for (int b = 0; b < kPer; ++b) {
            out[i * kPer + b] = (int8_t)((word >> (8 * b)) & 0xff);
        }
    }
}

// @brief 一行激活对 int8 ``[K, N]`` 权重，每个线程持 ``W`` 个相邻列。
//
// ``W`` 个累加器是同一 K 区间上彼此独立的和，所以算术就是标量内核的版本按同样顺序重复
// ``W`` 次——变的只是加载宽度和最后的 scale 施加。scale 仍然每列施加一次，乘在该列完整的
// K 和上。
//
// @param wt [K, N] int8 权重，行主序，且 ``N % W == 0``。
// @param x [K] 激活。
// @param y [N] 按激活 dtype 的输出。
// @param scale [N] fp32 每列 scale。
// @param N 行间距兼列数。
// @param K 归约长度。
template <typename XT, int U, int W>
__global__ void gemv_kn_i8w_kernel(const int8_t* __restrict__ wt,
                                   const XT* __restrict__ x, XT* __restrict__ y,
                                   const float* __restrict__ scale, int N, int K) {
    const int n0 = (blockIdx.x * blockDim.x + threadIdx.x) * W;
    if (n0 >= N) return;
    float acc[W];
#pragma unroll
    for (int i = 0; i < W; ++i) acc[i] = 0.f;
    int k = 0;
    for (; k + U <= K; k += U) {
#pragma unroll
        for (int u = 0; u < U; ++u) {
            int8_t bytes[W];
            load_i8_row<W>(&wt[(size_t)(k + u) * N + n0], bytes);
            const float xv = CxScalar<XT>::to_float(x[k + u]);
#pragma unroll
            for (int i = 0; i < W; ++i) acc[i] = fmaf((float)bytes[i], xv, acc[i]);
        }
    }
    for (; k < K; ++k) {
        int8_t bytes[W];
        load_i8_row<W>(&wt[(size_t)k * N + n0], bytes);
        const float xv = CxScalar<XT>::to_float(x[k]);
#pragma unroll
        for (int i = 0; i < W; ++i) acc[i] = fmaf((float)bytes[i], xv, acc[i]);
    }
#pragma unroll
    for (int i = 0; i < W; ++i) {
        y[n0 + i] = CxScalar<XT>::from_float(scale[n0 + i] * acc[i]);
    }
}

// @brief 同一 ``W`` 宽乘积的一个 K 区间，写进本分片的部分和。
//
// 故意不乘 scale：分片里装的是原始 K 和，由 ``gemv_reduce_i8_kernel`` 在相加之后施加一次
// scale，所以 scale 不能按分片施。
//
// @param wt [K, N] int8 权重，行主序，且 ``N % W == 0``。
// @param x [K] 激活。
// @param part [split, N] fp32 部分和，写本分片那一行。
// @param N 行间距兼列数。
// @param K 归约长度。
// @param split K 区间个数。
// @param per 一个 K 区间的长度；最后一个会被夹到 ``K``。
template <typename XT, int U, int W>
__global__ void gemv_kn_i8w_split_kernel(const int8_t* __restrict__ wt,
                                         const XT* __restrict__ x,
                                         float* __restrict__ part,
                                         int N, int K, int split, int per) {
    const int n0 = (blockIdx.x * blockDim.x + threadIdx.x) * W;
    if (n0 >= N) return;
    const int s = blockIdx.y;
    const int k0 = s * per;
    const int k1 = min(K, k0 + per);
    float acc[W];
#pragma unroll
    for (int i = 0; i < W; ++i) acc[i] = 0.f;
    int k = k0;
    for (; k + U <= k1; k += U) {
#pragma unroll
        for (int u = 0; u < U; ++u) {
            int8_t bytes[W];
            load_i8_row<W>(&wt[(size_t)(k + u) * N + n0], bytes);
            const float xv = CxScalar<XT>::to_float(x[k + u]);
#pragma unroll
            for (int i = 0; i < W; ++i) acc[i] = fmaf((float)bytes[i], xv, acc[i]);
        }
    }
    for (; k < k1; ++k) {
        int8_t bytes[W];
        load_i8_row<W>(&wt[(size_t)k * N + n0], bytes);
        const float xv = CxScalar<XT>::to_float(x[k]);
#pragma unroll
        for (int i = 0; i < W; ++i) acc[i] = fmaf((float)bytes[i], xv, acc[i]);
    }
    float* col = part + (size_t)s * N + n0;
#pragma unroll
    for (int i = 0; i < W; ++i) col[i] = acc[i];
}

// @brief ``M`` 行激活共用一遍权重读取，``M`` 是编译期宽度。
//
// ``M`` 是模板参数而不是运行时循环上界，是为了让编译器把那 ``M`` 个累加器留在寄存器里而不
// 溢出到栈上：这个内核的意义就是用一次权重读取换 ``M`` 个结果，而只有 ``M`` 条链都一直活在
// k 循环里，它才划算。
template <typename WT, typename XT, int U, int M>
__global__ void gemv_kn_m_kernel(const WT* __restrict__ wt,
                                 const XT* __restrict__ x, XT* __restrict__ y,
                                 int N, int K) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.f;
    int k = 0;
    for (; k + U <= K; k += U) {
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const float w = CxScalar<WT>::to_float(wt[(size_t)(k + u) * N + n]);
#pragma unroll
            for (int m = 0; m < M; ++m) {
                acc[m] += w * CxScalar<XT>::to_float(x[(size_t)m * K + k + u]);
            }
        }
    }
    for (; k < K; ++k) {
        const float w = CxScalar<WT>::to_float(wt[(size_t)k * N + n]);
#pragma unroll
        for (int m = 0; m < M; ++m) {
            acc[m] += w * CxScalar<XT>::to_float(x[(size_t)m * K + k]);
        }
    }
#pragma unroll
    for (int m = 0; m < M; ++m) y[(size_t)m * N + n] = CxScalar<XT>::from_float(acc[m]);
}

// W8A16 的小 M 内核家族：上面那个 bf16 小 M 内核，换成 int8 权重加每列 scale。
//
// 它存在是因为 int8 路径根本没有批处理入口，而这个缺失不会以错误的形式暴露出来——
// ``_gemv_rows`` 循环调用单行内核，那份内核每行都要把整个权重读一遍。实测下来，M 行的投影
// 于是要花 M 倍于一行的时间（``M=8/M=1 = 8.19x``，五个解码形状，``probe_row_cost.py``），
// 所以一次 c-token 投机校验要付 c 遍完整权重读取，在任何草稿深度都无法打平
// （``../../../../PERF_COREX.md`` 第 14 节）。
//
// 这个内核的第一版保留标量单字节加载、也没做 split-K，理由是 M 行本该提供单行内核靠切分 K
// 才得到的那些内存级并行。实测这个想法错在两处：M=8 时只有 28.5 GB/s，而单行能到
// 395 GB/s；代价是一行的 13.87 倍，而不是想要的约 1 倍。两个修法在这份文件的别处早就成立，
// 所以这个家族直接借用，而不是再造第三种调度：
//
//   * 每个线程持 ``W`` 个相邻列，按一个字读（``load_i8_row``）。一个 wavefront 发 64 次
//     单字节加载只覆盖半条 128 字节行；W=4 时覆盖整整两条。``I8_WIDE_MIN_N`` 和它背后的
//     扫描原样适用。
//   * split-K，因为缺的正是网格：9728 列按 256 线程是 38 个块，落到 16 个 SM 上，而每线程
//     再多干活也救不了这么稀的网格。宽度由 ``pick_split_i8`` 决定，不改。
//
// 正确性要求是 *与单行入口逐位相同*，而不是落在容差内，因为批处理解码不能改变任何一个
// token（``models/qwen35/batch.py``）。所以这个家族照搬单行路径的分派，而不是自己挑调度：
// 同样由 ``pick_i8_width`` 得到 ``W``，同样由 ``pick_split_i8`` 得到分片数，片内同样的累加
// 顺序，跨片同样的归约顺序。换一个分片数会重新结合那笔 fp32 和，挪动几个 ULP——实测在分片
// 不镜像时，19456-77824 个值里有 1-5 个与逐行路径不同，而那种差异正是长生成里偶尔翻转一次
// argmax 的东西。
//
// 有一处不对称是故意的：累加风格跟随 ``W``，因为单行内核就是这么分的。
// ``gemv_kn_i8_kernel``（标量）写 ``acc += w * x``，``gemv_kn_i8w_kernel``（宽）写
// ``fmaf(w, x, acc)``。两者在最后一位上的舍入不同，所以一种固定风格不可能同时对上两边；
// 这里用同一形状在单行路径上本来会用到的那种。

// @brief M 行激活对 int8 ``[K, N]`` 权重的一组 ``W`` 宽列。
//
// @param wt [K, N] int8 权重，行主序，且 ``N % W == 0``。
// @param x [M, K] 激活，行主序。
// @param y [M, N] 按激活 dtype 的输出，行主序。
// @param scale [N] fp32 每列 scale，在该列完整 K 和之后施加一次。
// @param N 行间距兼列数。
// @param K 归约长度。
template <typename XT, int U, int M, int W>
__global__ void gemv_kn_i8_m_kernel(const int8_t* __restrict__ wt,
                                    const XT* __restrict__ x, XT* __restrict__ y,
                                    const float* __restrict__ scale, int N, int K,
                                    int rows) {
    const int n0 = (blockIdx.x * blockDim.x + threadIdx.x) * W;
    if (n0 >= N) return;
    float acc[M * W];
#pragma unroll
    for (int i = 0; i < M * W; ++i) acc[i] = 0.f;
    int k = 0;
    for (; k + U <= K; k += U) {
#pragma unroll
        for (int u = 0; u < U; ++u) {
            // 一个权重字喂 M*W 个累加器。把这次加载提到行循环之外，只是因为那个循环是模板
            // 实例化并完全展开的；运行时行循环会每行重读一遍这些字节，而那正是本内核存在要
            // 消除的 bug。
            int8_t bytes[W];
            load_i8_row<W>(&wt[(size_t)(k + u) * N + n0], bytes);
#pragma unroll
            for (int m = 0; m < M; ++m) {
                // 超过 ``rows`` 的行改为重读第 0 行而不是缓冲区里真实的行：多出来的累加器
                // 算完就丢，所以非特化宽度的调用永远不需要一个补齐过的输入张量。
                const float xv = CxScalar<XT>::to_float(
                    x[(size_t)(m < rows ? m : 0) * K + k + u]);
#pragma unroll
                for (int i = 0; i < W; ++i) {
                    // W > 1 对应宽单行内核，用 fmaf；W == 1 对应标量内核，用分开的乘法和
                    // 加法。把两种写法分开，才让各自与它们的单行孪生逐位相同。
                    if (W > 1) acc[m * W + i] = fmaf((float)bytes[i], xv, acc[m * W + i]);
                    else acc[m * W + i] += (float)bytes[i] * xv;
                }
            }
        }
    }
    for (; k < K; ++k) {
        int8_t bytes[W];
        load_i8_row<W>(&wt[(size_t)k * N + n0], bytes);
#pragma unroll
        for (int m = 0; m < M; ++m) {
            const float xv = CxScalar<XT>::to_float(x[(size_t)(m < rows ? m : 0) * K + k]);
#pragma unroll
            for (int i = 0; i < W; ++i) {
                if (W > 1) acc[m * W + i] = fmaf((float)bytes[i], xv, acc[m * W + i]);
                else acc[m * W + i] += (float)bytes[i] * xv;
            }
        }
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
        // 只存真实的行：调用方的输出正好有 ``rows`` 行，写得更宽会冲出它的末尾。
        if (m >= rows) continue;
#pragma unroll
        for (int i = 0; i < W; ++i) {
            y[(size_t)m * N + n0 + i] =
                CxScalar<XT>::from_float(scale[n0 + i] * acc[m * W + i]);
        }
    }
}

// @brief int8 小 M 乘积的一个 K 区间，写进本分片的平面。
//
// 故意不乘 scale：分片里装的是原始 K 和，由归约在相加之后施加一次 scale，与单行的分片对
// 完全一样。在这里施加会把每个切片各乘一遍，改变结果。
//
// @param wt [K, N] int8 权重，行主序，且 ``N % W == 0``。
// @param x [M, K] 激活，行主序。
// @param part [M*split, N] fp32 部分和；第 ``m`` 行的平面从 ``m * split`` 开始。
// @param N 行间距兼列数。
// @param K 归约长度。
// @param split K 区间个数。
// @param per 一个 K 区间的长度；最后一个夹到 K。
template <typename XT, int U, int M, int W>
__global__ void gemv_kn_i8_m_split_kernel(const int8_t* __restrict__ wt,
                                          const XT* __restrict__ x,
                                          float* __restrict__ part,
                                          int N, int K, int split, int per, int rows) {
    const int n0 = (blockIdx.x * blockDim.x + threadIdx.x) * W;
    if (n0 >= N) return;
    const int s = blockIdx.y;
    const int k0 = s * per;
    const int k1 = min(K, k0 + per);
    float acc[M * W];
#pragma unroll
    for (int i = 0; i < M * W; ++i) acc[i] = 0.f;
    int k = k0;
    for (; k + U <= k1; k += U) {
#pragma unroll
        for (int u = 0; u < U; ++u) {
            int8_t bytes[W];
            load_i8_row<W>(&wt[(size_t)(k + u) * N + n0], bytes);
#pragma unroll
            for (int m = 0; m < M; ++m) {
                const float xv = CxScalar<XT>::to_float(
                    x[(size_t)(m < rows ? m : 0) * K + k + u]);
#pragma unroll
                for (int i = 0; i < W; ++i) {
                    if (W > 1) acc[m * W + i] = fmaf((float)bytes[i], xv, acc[m * W + i]);
                    else acc[m * W + i] += (float)bytes[i] * xv;
                }
            }
        }
    }
    for (; k < k1; ++k) {
        int8_t bytes[W];
        load_i8_row<W>(&wt[(size_t)k * N + n0], bytes);
#pragma unroll
        for (int m = 0; m < M; ++m) {
            const float xv = CxScalar<XT>::to_float(x[(size_t)(m < rows ? m : 0) * K + k]);
#pragma unroll
            for (int i = 0; i < W; ++i) {
                if (W > 1) acc[m * W + i] = fmaf((float)bytes[i], xv, acc[m * W + i]);
                else acc[m * W + i] += (float)bytes[i] * xv;
            }
        }
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
        if (m >= rows) continue;
        float* col = part + ((size_t)m * split + s) * N + n0;
#pragma unroll
        for (int i = 0; i < W; ++i) col[i] = acc[m * W + i];
    }
}

// @brief 把分片平面相加并施加列 scale，每列一个线程。
//
// 每一行的归约顺序都是 ``s = 0 .. split-1``，这才让结合方式与单行归约完全相同，整条链因此
// 与循环调用单行入口逐位相同。
//
// @param part [M*split, N] fp32 部分和，行主序。
// @param y [M, N] 按激活 dtype 的输出，原地写入。
// @param scale [N] fp32 每列 scale。
// @param N 输出列数。
// @param split K 区间个数。
// @param M 激活行数。
template <typename XT>
__global__ void gemv_i8_m_reduce_kernel(const float* __restrict__ part,
                                        XT* __restrict__ y,
                                        const float* __restrict__ scale,
                                        int N, int split, int M, int rows) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    for (int m = 0; m < M; ++m) {
        if (m >= rows) continue;
        float acc = 0.f;
        for (int s = 0; s < split; ++s) acc += part[((size_t)m * split + s) * N + n];
        y[(size_t)m * N + n] = CxScalar<XT>::from_float(scale[n] * acc);
    }
}

// @brief SM 数量，只查询一次。Split-K 按卡自身定尺寸，而不是写死 16，所以 BI-V150 或
//        SM 数不同的分区也能拿到同样的占用率目标。
int sm_count() {
    static int cached = 0;
    if (cached == 0) {
        int device = 0;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&cached, cudaDevAttrMultiProcessorCount, device);
        if (cached <= 0) cached = 16;
    }
    return cached;
}

// @brief 每块线程数：在每个形状上都够填满一次 wavefront 对齐的事务，又不会让小的 ``N``
//        剩下半块空转的通道。
int pick_threads(int64_t N) {
    if (N >= 2048) return 256;
    if (N >= 512) return 128;
    return 64;
}

// @brief Split-K 宽度，或 ``1`` 表示跑单趟内核。
//
// 切分乘倍了块数却不碰权重流量：每个分片读每列一个不同的 K 区间，所以搬动的字节数相同，
// 变的只是同时在飞的独立加载数。在一张拷贝带宽上限 575 GB/s 的 16 SM 部件上，单趟在多数
// 形状上都让卡缺少内存级并行，所以切分是默认而不是例外。
//
// 在 BI-V100 上实测，每个配置都对过主机侧 fp64 参考。第一张表才是有用的那张——它是
// *解码步* 实际付的代价，经 ``backends.ops.linear`` 测得，所以包含了启动和部分和归约：
//
//   shape      N       K      split 1      split 4      split 8
//   mlp_up     9216    2560   0.292 ms     0.152 ms     --
//   qkv_gate   8192    2560   0.286 ms     0.160 ms     --
//   mlp_down   2560    9216   1.004 ms     --           0.245 ms
//   o_proj     2560    4096   0.329 ms     --           0.085 ms
//   lm_head    248320  2560   3.478 ms     2.961 ms     --
//
// 那张表里打进了对早先推理的两次纠正，两次都是拿一个任务换来的：
//
//   * 「只在单趟网格填不满卡时才切分」是错的。mlp_up 的 9216 列在 16 个 SM 上是 36 个
//     块——已经两个 wave 了——而四路切分仍然有 1.92x 的收益。受限的从来不是 *块* 的占用率，
//     而是在飞的加载数。
//   * 「窄形状没收益」也错了，而且是反的。mlp_down 和 o_proj 才是切分最值的形状（对单趟
//     分别是 4.1x 和 3.9x）——它们恰恰是可铺开的列数最少的那些。
//
// 切分必须尊重的是 ``K``：每个分片拿到 ``K / split``，而切片太短时部分和那一趟吃掉的好处
// 会超过收益。mlp_down 在这里所有形状里 K 最长，取 8，也是这个函数会返回的最大值。
int pick_split(int64_t N, int64_t K, int threads) {
    if (K < 1024) return 1;

    // 切片短于约 512 时，第二次启动就不划算了。
    const int64_t by_k = K / 512 > 8 ? 8 : K / 512;
    if (by_k < 2) return 1;

    // 宽形状靠列就能填满卡，所以比窄形状需要的帮助少。mlp_up/qkv_gate 实测在 4 最好、在 8
    // 变差；更窄的统统在 8 最好。阈值就放在扫描给出的交叉点上。
    const int64_t want = N >= 4096 ? 4 : 8;
    return (int)(want < by_k ? want : by_k);
}

//: int8 的 split-K 规则要维持的常驻块数。下面每个实测最优点都落在该乘积上或相差不超过一个
//: 八度；用 *块数* 而不是分片数作为目标，是因为同一个分片数在不同 ``N`` 上含义不同：s8 在
//: ``lm_head`` 上是 970 个块，在 ``o_proj`` 上是 10 个块，而两者之中只有一个需要切分。
constexpr int I8_TARGET_BLOCKS = 1024;

//: int8 分片的硬上限。超过它，分片缓冲区的流量代价就超过多出来的块所挽回的，而且归约那趟
//: 会随分片数增长。
constexpr int I8_MAX_SPLIT = 32;

//: 值得调度的最短 K 切片。一个切片就是一次没有复用的带步长遍历；低于约 64 个元素时，启动和
//: 部分和的写入会超过切片本身读的量。它同时也是分片缓冲区上限：归约要花 ``8 * split * N``
//: 字节（一个 fp32 ``[split, N]`` 先写后读），所以 ``split <= K / 64`` 把这份开销封顶在权重
//: 流量的 ``8 / 64`` = 12.5%，不管 ``K`` 和 ``N`` 是多少。这就是为什么仅凭这一个常量，就能
//: 阻止一个长 K 的形状把自己切成代价大于收益的切片。
constexpr int I8_MIN_SLICE = 64;

// @brief int8 路径的 split-K 宽度，或 ``1`` 表示跑单趟内核。
//
// 上面的 ``pick_split`` 是 bf16 规则，不动它：它的数字来自 bf16 扫描，8 这个上限编码的是
// bf16 的取舍而不是 int8 的。int8 路径每列读的字节数减半，所以要维持同样多的在飞加载大约
// 需要两倍的块——这就是同样的形状在这里要 16-32、而 bf16 只要 4-8 的原因。
//
// 在 BI-V100（16 SM，572 GB/s 上限）上按形状实测，固定 ``K = 2560``，每个候选都对过基于
// 反量化权重的主机侧 fp64 参考。中间那一列是 256 线程下的单趟块数，也就是这条规则所依赖的
// 量：
//
//   N        blocks   best split
//   4096     16       32
//   8192     32       16
//   16384    64       32
//   32768    128      8
//   65536    256      4
//   131072   512      2
//   248320   970      8        （s1 实测相差在 1% 内，所以这一行是个平底盆）
//
// 一条线精确拟合了从 128 块往上的平台区——``split * blocks ~ 1024``。最小的三个网格不在这条
// 线上：4096 被 ``I8_MAX_SPLIT`` 卡住，而 8192 和 16384 分别落在线两侧各一个八度，248320
// 则是 s1 与 s8 相差不到 1% 的平底盆。所以这是一条拟合而不是一张表：它能为下一个模型的
// ``N`` 插值，而它服务的这些形状本来就不是整数，写死一张表反正一个都盖不住。
//
// 它的已知偏差，写出来是为了不让人再把它当 bug 发现一遍：在中间那档网格上，四舍五入的除法
// 会落在实测最优下面一档（16384 拿到 16，而实测最好是 32），真实形状继承这一点——
// ``N = 18432`` 拿到 14，``N = 12352`` 拿到 21，``N = 10240`` 拿到 26，而实测最好分别是
// 16、32、32。分得偏少是安全的方向（它花的是块数，从不是正确性），而承担这一步大部分时间
// 的那些形状——两个 10 块的——都正好撞到上限。
//
// 它修掉的失败是旧的上限 8：``o_proj``（10 块）和 ``mlp_down``（10 块）被压在 8，而实测最
// 优是 32，仅这两个形状就占了那次分派改动找回的 5.9 ms 里的大部分。
//
// @param N 权重的输出列数。
// @param K 归约长度；它通过 ``I8_MIN_SLICE`` 给分片数设界。
// @param threads 每块线程数，它决定单趟块数。
// @return ``[1, I8_MAX_SPLIT]`` 里的分片宽度，且其切片总能划分 ``K``。
int pick_split_i8(int64_t N, int64_t K, int threads) {
    const int64_t blocks = threads > 0 ? (N + threads - 1) / threads : N;
    if (blocks <= 0) return 1;

    // 四舍五入的除法，让规则落在最近的块数上，而不是在每个形状上都朝更宽的分片截断。
    int64_t want = (I8_TARGET_BLOCKS + blocks / 2) / blocks;
    if (want > I8_MAX_SPLIT) want = I8_MAX_SPLIT;

    // 每个分片都得留下足够长、值得调度的切片；K/64 就是这个界，也正是它阻止一个短 K 的形状
    // 被切成启动开销盖不住的薄片。
    const int64_t by_k = K / I8_MIN_SLICE;
    if (by_k < want) want = by_k;
    return (int)(want < 1 ? 1 : want);
}

// @brief 为一种（权重，激活）dtype 组合启动单趟内核。
//
// U=1 和 U=2/U=4 一起实例化，尽管它从不是默认值：它是形状扫描里的对照组。没有它，一个 U=4
// 的测量就分不清「展开有用」和「循环开销本来就不是瓶颈」，而扫描会留下一个它从未证明有用的
// 旋钮。
template <typename WT, typename XT>
torch::Tensor launch_plain(const torch::Tensor& wt, const torch::Tensor& x,
                           torch::Tensor& y, int N, int K, int threads, int unroll) {
    const WT* wp = reinterpret_cast<const WT*>(wt.data_ptr());
    const XT* xp = reinterpret_cast<const XT*>(x.data_ptr());
    XT* yp = reinterpret_cast<XT*>(y.data_ptr());
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const int blocks = (int)((N + threads - 1) / threads);
    if (unroll == 1) {
        gemv_kn_kernel<WT, XT, 1><<<blocks, threads, 0, stream>>>(wp, xp, yp, N, K);
    } else if (unroll == 2) {
        gemv_kn_kernel<WT, XT, 2><<<blocks, threads, 0, stream>>>(wp, xp, yp, N, K);
    } else {
        gemv_kn_kernel<WT, XT, 4><<<blocks, threads, 0, stream>>>(wp, xp, yp, N, K);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

// @brief 为一种 dtype 组合启动 split-K 的两趟。
template <typename WT, typename XT>
torch::Tensor launch_split(const torch::Tensor& wt, const torch::Tensor& x,
                           torch::Tensor& y, int N, int K, int threads, int unroll,
                           int split) {
    const WT* wp = reinterpret_cast<const WT*>(wt.data_ptr());
    const XT* xp = reinterpret_cast<const XT*>(x.data_ptr());
    XT* yp = reinterpret_cast<XT*>(y.data_ptr());
    auto part = torch::empty({split, N}, x.options().dtype(torch::kFloat));
    float* pp = part.data_ptr<float>();
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const int per = (K + split - 1) / split;
    const int blocks = (int)((N + threads - 1) / threads);
    const dim3 grid(blocks, split);
    if (unroll == 1) {
        gemv_kn_split_kernel<WT, XT, 1><<<grid, threads, 0, stream>>>(wp, xp, pp, N, K,
                                                                     split, per);
    } else if (unroll == 2) {
        gemv_kn_split_kernel<WT, XT, 2><<<grid, threads, 0, stream>>>(wp, xp, pp, N, K,
                                                                     split, per);
    } else {
        gemv_kn_split_kernel<WT, XT, 4><<<grid, threads, 0, stream>>>(wp, xp, pp, N, K,
                                                                     split, per);
    }
    gemv_reduce_kernel<XT><<<blocks, threads, 0, stream>>>(pp, yp, N, split);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

// @brief 为一种 dtype 组合和一个 ``M`` 宽度启动小 M 内核。
template <typename WT, typename XT, int M>
torch::Tensor launch_m(const torch::Tensor& wt, const torch::Tensor& x, torch::Tensor& y,
                       int N, int K, int threads) {
    const WT* wp = reinterpret_cast<const WT*>(wt.data_ptr());
    const XT* xp = reinterpret_cast<const XT*>(x.data_ptr());
    XT* yp = reinterpret_cast<XT*>(y.data_ptr());
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const int blocks = (int)((N + threads - 1) / threads);
    gemv_kn_m_kernel<WT, XT, 4, M><<<blocks, threads, 0, stream>>>(wp, xp, yp, N, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

//: 列数达到或超过它时，``width == 0``（自动）路径选四宽内核。这只是 *默认分派*，不是正确性
//: 下限：通过 ``width`` 参数，宽内核在任意 ``N`` 下都够得着，而扫描驱动的正是那个参数。
//:
//: 4096 是 split 16/32 下宽度扫描给出的交叉点，而 split 很要紧：旧的 32768 下限是在
//: ``split`` 被压到 8 时测的，那时网格很稀，于是宽内核为一个 split-K 现在能补回来的线程数
//: 背了锅。在解码步真正调用的形状上重测（``tune_gemv_i8_wide.py``，16 SM，571 GB/s 上限，
//: 每一格都对过主机侧 fp64 参考）：
//:
//:   N       K      width 1     width 4     四宽赢多少
//:   18432   2560   234 GB/s    375 GB/s    1.60x
//:   12352   2560   369 GB/s    462 GB/s    1.25x
//:   10240   2560   366 GB/s    452 GB/s    1.23x
//:   248320  2560   452 GB/s    461 GB/s    1.02x
//:   2560    9216   337 GB/s    283 GB/s    —— 四宽输（0.84x）
//:   2560    4096   238 GB/s    215 GB/s    —— 四宽输（0.90x）
//:   2560    5120   269 GB/s    234 GB/s    —— 四宽输（0.87x）
//:
//: 分界很清楚，而且与字节数无关：上面每个形状都是每个元素读一个字节。它取决于模型还剩多少
//: 线程。一个 2560 列的投影在 256 线程下是 10 个块，而四宽内核把它除以四变成 2.5 个块——
//: 在 16 个 SM 上不够一个 wave。超过约 4096 列，网格就扛得住这个除法，更宽的加载胜出。所以
//: 窄形状上用 ``width = 1`` 是实测出来的选择，不是遗漏。
constexpr int I8_WIDE_MIN_N = 4096;

//: int8 内核实例化的最大 ``width``。主机侧包装层拒绝比它更大的值，所以扫描不可能要到一个
//: 构建里没有对应内核的宽度。
constexpr int I8_MAX_WIDTH = 16;

// @brief 把 ``width`` 参数解析成一个编译期分派选择。
//
// ``0`` 保留已发布的规则（达到或超过 :data:`I8_WIDE_MIN_N` 时四宽，否则标量），所以默认路径
// 在构造上就没变。显式宽度在能整除 ``N`` 时被采用，否则退回标量内核：一个宽度不是 ``W`` 整数
// 倍的模型必须保持正确，而不是跨行读，而读相邻行的字节照样会产出看着合理的数。
//
// 构建里没有实例化的宽度也在这里折回标量内核。包装层会先 TORCH_CHECK 它，所以这里是第二道
// 防线而不是调用方看到的错误——但它让算术对任何 ``int`` 都保持完整，这才让那张回退表能在没有
// 设备的情况下被测试。
//
// @param N 输出列数。
// @param width 请求的每线程列数，``0`` 表示用已发布的规则。
// @return ``1``、``2``、``4``、``8`` 或 ``16``，且大于 1 时总是 ``N`` 的约数。
inline int pick_i8_width(int N, int width) {
    int w = width;
    if (w <= 0) w = (N % 4 == 0 && N >= I8_WIDE_MIN_N) ? 4 : 1;
    // 只用已实例化的宽度：每一种都按恰好那么多字节的单个无符号字读出，而没有对应字类型的
    // 取值背后没有内核。
    if (w != 1 && w != 2 && w != 4 && w != 8 && w != I8_MAX_WIDTH) return 1;
    if (w > 1 && (N % w) != 0) return 1;
    return w;
}

// @brief 启动一个 ``W`` 和 ``U`` 都是编译期常量的单趟 int8 内核。
template <typename XT, int W, int U>
void launch_i8_plain_one(const int8_t* wp, const XT* xp, XT* yp, const float* sp,
                         int N, int K, int threads) {
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const int blocks = (int)(((N / W) + threads - 1) / threads);
    gemv_kn_i8w_kernel<XT, U, W><<<blocks, threads, 0, stream>>>(wp, xp, yp, sp, N, K);
}

// @brief 按请求的展开度启动一个 ``W`` 宽的单趟内核。
template <typename XT, int W>
void launch_i8_plain_w(const int8_t* wp, const XT* xp, XT* yp, const float* sp,
                       int N, int K, int threads, int unroll) {
    if (unroll == 2) {
        launch_i8_plain_one<XT, W, 2>(wp, xp, yp, sp, N, K, threads);
    } else {
        launch_i8_plain_one<XT, W, 4>(wp, xp, yp, sp, N, K, threads);
    }
}

// @brief 启动一个 ``W`` 和 ``U`` 都是编译期常量的 split-K 分片内核。
template <typename XT, int W, int U>
void launch_i8_split_one(const int8_t* wp, const XT* xp, float* pp, int N, int K,
                         int threads, int split, int per) {
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const int blocks = (int)(((N / W) + threads - 1) / threads);
    gemv_kn_i8w_split_kernel<XT, U, W><<<dim3(blocks, split), threads, 0, stream>>>(
        wp, xp, pp, N, K, split, per);
}

// @brief 按请求的展开度启动一个 ``W`` 宽的分片内核。
template <typename XT, int W>
void launch_i8_split_w(const int8_t* wp, const XT* xp, float* pp, int N, int K,
                       int threads, int split, int per, int unroll) {
    if (unroll == 2) {
        launch_i8_split_one<XT, W, 2>(wp, xp, pp, N, K, threads, split, per);
    } else {
        launch_i8_split_one<XT, W, 4>(wp, xp, pp, N, K, threads, split, per);
    }
}

// @brief 为一种激活 dtype、宽度和展开度启动单趟 int8 内核。
//
// ``width == 4`` 仍然走 :func:`gemv_kn_i8x4_kernel` 而不是通用的 4 宽实例化：34.23 ->
// 46.81 ms 那次测量就是对着这个内核做的，把它留在路径里，那个负面结果才可复现。
template <typename XT>
torch::Tensor launch_i8_plain(const torch::Tensor& wt, const torch::Tensor& x,
                              torch::Tensor& y, const torch::Tensor& scale,
                              int N, int K, int threads, int unroll, int width) {
    const int8_t* wp = reinterpret_cast<const int8_t*>(wt.data_ptr());
    const XT* xp = reinterpret_cast<const XT*>(x.data_ptr());
    XT* yp = reinterpret_cast<XT*>(y.data_ptr());
    const float* sp = scale.data_ptr<float>();
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const int w = pick_i8_width(N, width);
    if (w == 4) {
        const int blocks = (int)(((N / 4) + threads - 1) / threads);
        if (unroll == 2) {
            gemv_kn_i8x4_kernel<XT, 2><<<blocks, threads, 0, stream>>>(wp, xp, yp, sp, N, K);
        } else {
            gemv_kn_i8x4_kernel<XT, 4><<<blocks, threads, 0, stream>>>(wp, xp, yp, sp, N, K);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return y;
    }
    if (w == 2) {
        launch_i8_plain_w<XT, 2>(wp, xp, yp, sp, N, K, threads, unroll);
    } else if (w == 8) {
        launch_i8_plain_w<XT, 8>(wp, xp, yp, sp, N, K, threads, unroll);
    } else if (w == 16) {
        launch_i8_plain_w<XT, 16>(wp, xp, yp, sp, N, K, threads, unroll);
    } else {
        const int blocks = (int)((N + threads - 1) / threads);
        if (unroll == 2) {
            gemv_kn_i8_kernel<XT, 2><<<blocks, threads, 0, stream>>>(wp, xp, yp, sp, N, K);
        } else {
            gemv_kn_i8_kernel<XT, 4><<<blocks, threads, 0, stream>>>(wp, xp, yp, sp, N, K);
        }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

// @brief 为一种激活 dtype、宽度和展开度启动 split-K 的 int8 两趟。
template <typename XT>
torch::Tensor launch_i8_split(const torch::Tensor& wt, const torch::Tensor& x,
                              torch::Tensor& y, const torch::Tensor& scale,
                              int N, int K, int threads, int unroll, int split,
                              int width) {
    const int8_t* wp = reinterpret_cast<const int8_t*>(wt.data_ptr());
    const XT* xp = reinterpret_cast<const XT*>(x.data_ptr());
    XT* yp = reinterpret_cast<XT*>(y.data_ptr());
    const float* sp = scale.data_ptr<float>();
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const int per = (K + split - 1) / split;
    // 分片是暂存，一次为所有分片定尺寸：归约需要在任何列能用 scale 之前每个分片都写好，
    // 所以它们不能共用输出缓冲区。
    auto part = torch::empty({(int64_t)split, (int64_t)N}, x.options().dtype(torch::kFloat));
    float* pp = part.data_ptr<float>();
    const int w = pick_i8_width(N, width);
    if (w == 4) {
        const int blocks = (int)(((N / 4) + threads - 1) / threads);
        if (unroll == 2) {
            gemv_kn_i8x4_split_kernel<XT, 2><<<dim3(blocks, split), threads, 0, stream>>>(
                wp, xp, pp, N, K, split, per);
        } else {
            gemv_kn_i8x4_split_kernel<XT, 4><<<dim3(blocks, split), threads, 0, stream>>>(
                wp, xp, pp, N, K, split, per);
        }
    } else if (w == 2) {
        launch_i8_split_w<XT, 2>(wp, xp, pp, N, K, threads, split, per, unroll);
    } else if (w == 8) {
        launch_i8_split_w<XT, 8>(wp, xp, pp, N, K, threads, split, per, unroll);
    } else if (w == 16) {
        launch_i8_split_w<XT, 16>(wp, xp, pp, N, K, threads, split, per, unroll);
    } else {
        const int blocks = (int)((N + threads - 1) / threads);
        if (unroll == 2) {
            gemv_kn_i8_split_kernel<XT, 2><<<dim3(blocks, split), threads, 0, stream>>>(
                wp, xp, pp, N, K, split, per);
        } else {
            gemv_kn_i8_split_kernel<XT, 4><<<dim3(blocks, split), threads, 0, stream>>>(
                wp, xp, pp, N, K, split, per);
        }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    // 不管权重宽度是多少，归约都是每列一个线程：它每列读 ``split`` 个 float，所以代价由分片
    // 数决定，而不是由权重加载宽度决定。
    const int rblocks = (int)((N + threads - 1) / threads);
    gemv_reduce_i8_kernel<XT><<<rblocks, threads, 0, stream>>>(pp, yp, sp, N, split);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

}  // namespace

// @brief 为一种 dtype、一个 M 宽度和一个 W 启动 int8 小 M 内核。
//
// @param wp [K, N] int8 权重。
// @param xp [M, K] 激活。
// @param sp [N] fp32 每列 scale。
// @param yp [M, N] 输出。
// @param pp [M*split, N] fp32 分片缓冲区，split 为 1 时为 null。
// @param N 输出列数。
// @param K 归约长度。
// @param threads 每块线程数。
// @param split Split-K 宽度；1 表示走单趟内核。
// @param per 一个 K 区间的长度。
template <typename XT, int M, int W>
void launch_i8_m_one(const int8_t* wp, const XT* xp, const float* sp, XT* yp, float* pp,
                     int N, int K, int threads, int split, int per, int rows) {
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    // 网格跟随宽度算术：一个线程持 W 列，所以块数是 N/W 除以块线程数。保留标量网格会启动
    // W 倍的块，而其中大多数会立刻返回。
    const int blocks = (int)(((N / W) + threads - 1) / threads);
    if (split > 1) {
        gemv_kn_i8_m_split_kernel<XT, 4, M, W><<<dim3(blocks, split), threads, 0, stream>>>(
            wp, xp, pp, N, K, split, per, rows);
        // 归约是每 *列* 一个线程，所以它要的是没除过 W 的网格。沿用被 W 除过的块数会让
        // N/W 之后的每一列都没被写，而那些列读回来的是暂存缓冲区里碰巧有的东西——行一致性
        // 检查会把它报成一个量级为 1 的最大相对误差，而不是启动错误。
        const int rblocks = (int)((N + threads - 1) / threads);
        gemv_i8_m_reduce_kernel<XT><<<rblocks, threads, 0, stream>>>(
            pp, yp, sp, N, split, M, rows);
    } else {
        gemv_kn_i8_m_kernel<XT, 4, M, W><<<blocks, threads, 0, stream>>>(
            wp, xp, yp, sp, N, K, rows);
    }
}

// @brief 按扫描规则为 N 选出的 W 启动 int8 小 M 内核。
//
// W 和分片数都来自单行入口所用的那套规则，绝不来自另一套。这就是逐位一致的契约集中在一处：
// 一次 M 行的调用必须走它的各行单独跑时会走的那条算术路径，所以两个分派决定都是委托来的，
// 而不是在这里重新决定。
//
// @param wt [K, N] int8 权重。
// @param x [M, K] 激活。
// @param y 预分配的 [M, N] 输出，原样返回以便调用链串接。
// @param scale [N] fp32 每列 scale。
// @param N 输出列数。
// @param K 归约长度。
// @param threads 每块线程数。
// @param split Split-K 宽度，按单行入口的 ``pick_split_i8`` 决定。
template <typename XT, int M>
torch::Tensor launch_i8_m(const torch::Tensor& wt, const torch::Tensor& x, torch::Tensor& y,
                          int rows, const torch::Tensor& scale, int N, int K, int threads,
                          int split) {
    const int8_t* wp = reinterpret_cast<const int8_t*>(wt.data_ptr());
    const XT* xp = reinterpret_cast<const XT*>(x.data_ptr());
    XT* yp = reinterpret_cast<XT*>(y.data_ptr());
    const float* sp = scale.data_ptr<float>();
    const int per = split > 1 ? (K + split - 1) / split : K;
    // 一次为所有行准备暂存：每个分片 M 个平面，因为每行的 K 和是独立归约的。用
    // [split, N] 的缓冲区会让每行覆盖上一行的部分和，而那只会在形状上读起来正确、答案却是
    // 错的。
    torch::Tensor part;
    float* pp = nullptr;
    if (split > 1) {
        part = torch::empty({M * split, N}, x.options().dtype(torch::kFloat));
        pp = part.data_ptr<float>();
    }
    switch (pick_i8_width(N, 0)) {
        case 2: launch_i8_m_one<XT, M, 2>(wp, xp, sp, yp, pp, N, K, threads, split, per, rows); break;
        case 4: launch_i8_m_one<XT, M, 4>(wp, xp, sp, yp, pp, N, K, threads, split, per, rows); break;
        case 8: launch_i8_m_one<XT, M, 8>(wp, xp, sp, yp, pp, N, K, threads, split, per, rows); break;
        default: launch_i8_m_one<XT, M, 1>(wp, xp, sp, yp, pp, N, K, threads, split, per, rows); break;
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

// @brief 解码期投影：一个激活向量对 ``[K, N]`` 权重。
//
// @param wt [K, N] 权重，连续存放，bf16 或 fp32。fp16 被拒绝而不是重新解释：内核把 16 位
//        权重按 bf16 位模式读，所以 fp16 张量会产出看着合理的垃圾，而不是报错。
// @param x [K] 激活；bf16 或 fp32。输出的 dtype 跟随它。
// @param threads 每块线程数，``0`` 表示按 ``N`` 选。
// @param unroll K 循环展开，``2`` 或 ``4``；``0`` 表示 4。
// @param split Split-K 宽度，``0`` 表示按 ``N``/``K`` 选；``1`` 强制单趟。
// @return [N]，dtype 同 ``x``。
// @throws c10::Error 形状、dtype 或布局不是内核索引的那种——不匹配时会读穿缓冲区，而不是
//         失败。
torch::Tensor gemv_kn(torch::Tensor wt, torch::Tensor x, int64_t threads,
                      int64_t unroll, int64_t split) {
    TORCH_CHECK(wt.dim() == 2, "weight must be 2-D [K, N]");
    TORCH_CHECK(x.dim() == 1, "activation must be 1-D [K]");
    TORCH_CHECK(wt.is_contiguous() && x.is_contiguous(), "contiguous inputs required");
    const int K = (int)wt.size(0), N = (int)wt.size(1);
    TORCH_CHECK(wt.size(0) == x.size(0), "K mismatch: weight ", wt.size(0),
                " vs activation ", x.size(0));
    const bool w16 = wt.scalar_type() == torch::kBFloat16;
    TORCH_CHECK(w16 || wt.scalar_type() == torch::kFloat,
                "weight must be bf16 or fp32 (pack.to_kn converts; an fp16 weight "
                "read as bf16 would compute silently wrong values)");
    const bool x16 = x.scalar_type() == torch::kBFloat16;
    TORCH_CHECK(x16 || x.scalar_type() == torch::kFloat,
                "activation must be bf16 or fp32");
    auto y = torch::empty({N}, x.options());
    const int th = threads > 0 ? (int)threads : pick_threads(N);
    const int u = unroll > 0 ? (int)unroll : 4;
    const int sp = split > 0 ? (int)split : pick_split(N, K, th);

    if (w16 && x16) return sp > 1
        ? launch_split<uint16_t, uint16_t>(wt, x, y, N, K, th, u, sp)
        : launch_plain<uint16_t, uint16_t>(wt, x, y, N, K, th, u);
    if (w16) return sp > 1
        ? launch_split<uint16_t, float>(wt, x, y, N, K, th, u, sp)
        : launch_plain<uint16_t, float>(wt, x, y, N, K, th, u);
    return sp > 1
        ? launch_split<float, float>(wt, x, y, N, K, th, u, sp)
        : launch_plain<float, float>(wt, x, y, N, K, th, u);
}

// @brief 解码期 W8A16 投影：int8 ``[K, N]`` 权重加每列 fp32 scale。
//
// 相对 :func:`gemv_kn` 把每个权重读的字节数减半，这是一条已经被带宽限制的路径上仅剩的杠杆：
// 投影占了解码步的大部分，而卡几乎全程都在饱和（``../../../../PERF_COREX.md`` §8）。
//
// @param wt [K, N] int8 权重，连续存放。
// @param x [K] 激活，bf16 或 fp32。输出的 dtype 跟随它。
// @param scale [N] fp32 absmax scale，每个输出列一个，每列施加一次。
// @param threads 每块线程数，``0`` 表示按 ``N`` 选。
// @param unroll K 循环展开，``2`` 或 ``4``；``0`` 表示 4。
// @param split Split-K 宽度，``0`` 表示自动选；``1`` 强制单趟。
// @param width 每线程输出列数，``0`` 表示用已发布的规则。合法值是 ``1``、``2``、``4``、
//        ``8`` 和 ``16``；``N % width`` 必须为 ``0``，不满足时内核退回 ``1``。``0`` 精确
//        复现原有的分派，所以不设它什么都不会变。
// @return [N]，dtype 同 ``x``。
// @throws c10::Error 权重不是 int8、scale 长度与 ``N`` 不符、显式 ``width`` 不是已实例化的
//         值，或 dtype/布局是内核索引不了的——这几种都会读错元素而不是失败。
torch::Tensor gemv_kn_i8(torch::Tensor wt, torch::Tensor x, torch::Tensor scale,
                         int64_t threads, int64_t unroll, int64_t split,
                         int64_t width) {
    TORCH_CHECK(wt.dim() == 2, "weight must be 2-D [K, N]");
    TORCH_CHECK(x.dim() == 1, "activation must be 1-D [K]");
    TORCH_CHECK(wt.scalar_type() == torch::kChar,
                "int8 weight required (a bf16 weight here would be read one byte per "
                "value and produce plausible garbage)");
    TORCH_CHECK(scale.scalar_type() == torch::kFloat, "scale must be fp32");
    TORCH_CHECK(wt.is_contiguous() && x.is_contiguous() && scale.is_contiguous(),
                "contiguous inputs required");
    const int K = (int)wt.size(0), N = (int)wt.size(1);
    TORCH_CHECK(wt.size(0) == x.size(0), "K mismatch: weight ", wt.size(0),
                " vs activation ", x.size(0));
    TORCH_CHECK(scale.numel() == N, "scale must hold one value per output column: got ",
                scale.numel(), " for N = ", N);
    // 直接拒绝而不是悄悄圆整到一个合法宽度：扫描要一个构建里没有内核的宽度时必须失败，而不是
    // 把相邻宽度的耗时挂在错的标签下报出来。
    TORCH_CHECK(width == 0 || width == 1 || width == 2 || width == 4 || width == 8
                    || width == 16,
                "width must be 0 (auto) or one of 1, 2, 4, 8, 16; got ", width);
    const bool x16 = x.scalar_type() == torch::kBFloat16;
    TORCH_CHECK(x16 || x.scalar_type() == torch::kFloat,
                "activation must be bf16 or fp32");
    auto y = torch::empty({N}, x.options());
    const int th = threads > 0 ? (int)threads : pick_threads(N);
    const int u = unroll > 0 ? (int)unroll : 4;
    // 只用于 int8：``pick_split`` 仍是 bf16 的规则，因为它的上限是照 bf16 测出来的。
    const int sp = split > 0 ? (int)split : pick_split_i8(N, K, th);
    const int w = (int)width;
    if (x16) {
        return sp > 1 ? launch_i8_split<uint16_t>(wt, x, y, scale, N, K, th, u, sp, w)
                      : launch_i8_plain<uint16_t>(wt, x, y, scale, N, K, th, u, w);
    }
    return sp > 1 ? launch_i8_split<float>(wt, x, y, scale, N, K, th, u, sp, w)
                  : launch_i8_plain<float>(wt, x, y, scale, N, K, th, u, w);
}

// @brief 批处理 W8A16 投影：M 行激活共用一遍权重读取。
//
// 缺的那个入口是 ``_gemv_rows``：没有批处理 int8 内核时，它每行循环调用一次单行 GEMV，于是
// M 行的投影要付 M 次权重读取。在本部件上这正好是一行的 M 倍（``M=8/M=1 = 8.19x``，五个
// 解码形状），也就是一次 c-token 投机校验要付 c 个普通步、在任何草稿深度都回不了本的原因
// （``../../../../PERF_COREX.md`` 第 14 节）。
//
// 正确性契约：每一行都与同一权重、同一激活上的单行入口逐位相同。因此两个分派决定——列宽和
// split-K 宽度——都取自单行路径自己的规则，而不是在这里重新选；某个调度没有批处理内核的形状
// 退回逐行循环，而不是退回某条近似路径。
//
// @param wt [K, N] int8 权重，连续存放。
// @param x [M, K] 激活，bf16 或 fp32；输出的 dtype 跟随它。
// @param scale [N] fp32 absmax scale，每个输出列一个。
// @param threads 每块线程数，``0`` 表示按 ``N`` 选。
// @param unroll K 循环展开；启动器会定死它，所以这个参数留给扫描用。
// @return [M, N]，dtype 同 ``x``。
// @throws c10::Error 权重不是 int8、scale 长度与 N 不符，或 dtype 与布局是内核索引不了的
//         ——这几种都会读错元素而不是失败。
torch::Tensor gemv_kn_i8_m(torch::Tensor wt, torch::Tensor x, torch::Tensor scale,
                           int64_t threads, int64_t unroll) {
    TORCH_CHECK(wt.dim() == 2, "weight must be 2-D [K, N]");
    TORCH_CHECK(x.dim() == 2, "activations must be 2-D [M, K]");
    TORCH_CHECK(wt.scalar_type() == torch::kChar,
                "int8 weight required (a bf16 weight here would be read one byte per "
                "value and produce plausible garbage)");
    TORCH_CHECK(scale.scalar_type() == torch::kFloat, "scale must be fp32");
    TORCH_CHECK(wt.is_contiguous() && x.is_contiguous() && scale.is_contiguous(),
                "contiguous inputs required");
    const int K = (int)wt.size(0), N = (int)wt.size(1), M = (int)x.size(0);
    TORCH_CHECK(wt.size(0) == x.size(1), "K mismatch: weight ", wt.size(0),
                " vs activation ", x.size(1));
    TORCH_CHECK(scale.numel() == N, "scale must hold one value per output column: got ",
                scale.numel(), " for N = ", N);
    TORCH_CHECK(M >= 1 && M <= 8, "small-M kernel handles M in [1, 8], got ", M);
    const bool x16 = x.scalar_type() == torch::kBFloat16;
    TORCH_CHECK(x16 || x.scalar_type() == torch::kFloat,
                "activation must be bf16 or fp32");
    (void)unroll;  // 启动器定死 U=4；见 launch_i8_m。
    auto y = torch::empty({M, N}, x.options());
    const int th = threads > 0 ? (int)threads : pick_threads(N);
    // 只分派有内核的宽度。其余的——模型从不会要的行数——留在逐行循环上，也就是缺这个内核时
    // 每次调用都被迫走的那条代码路径。留着它，才让回退在构造上就逐位相同，而不是靠两份实现
    // 恰好一致。宽度 2、4、8 是模型真要的那些（批处理解码的桶和校验记录），所以它们各有自己的
    // 实例化。其它宽度由上一档实例化承接，多出来的行重读第 0 行：退回逐行循环会让 M=5 花五次
    // 权重读取而 M=4 只花一次，那是一个改变算法而不是改变批大小的断崖。多出来的行算的是调用方
    // 从不读取的真实值，所以它们漏不进答案。
    const int padded = M <= 2 ? 2 : (M <= 4 ? 4 : 8);
    // 分片是单行规则逐字照搬，所以部分和的结合方式与逐行循环会产出的结果一致。
    const int sp = pick_split_i8(N, K, th);
    if (x16) {
        if (padded == 2) return launch_i8_m<uint16_t, 2>(wt, x, y, M, scale, N, K, th, sp);
        if (padded == 4) return launch_i8_m<uint16_t, 4>(wt, x, y, M, scale, N, K, th, sp);
        return launch_i8_m<uint16_t, 8>(wt, x, y, M, scale, N, K, th, sp);
    }
    if (padded == 2) return launch_i8_m<float, 2>(wt, x, y, M, scale, N, K, th, sp);
    if (padded == 4) return launch_i8_m<float, 4>(wt, x, y, M, scale, N, K, th, sp);
    return launch_i8_m<float, 8>(wt, x, y, M, scale, N, K, th, sp);
}

// @brief 小 M 投影：``M`` 行激活只把权重读一遍。
//
// ``M`` 为 2/4/8 时走宽度被编译进去的内核；其它 ``M`` 一次服务一行，那是正确的，只在契约
// 点名的调度之外才会走到。
//
// @param wt [K, N] 权重，连续存放，bf16 或 fp32。
// @param x [M, K] 激活，接受的 dtype 与 ``gemv_kn`` 相同。
// @param threads 每块线程数，``0`` 表示按 ``N`` 选。
// @param unroll K 循环展开，``0`` 表示用内核默认值。
// @return [M, N]，dtype 同 ``x``。
// @throws c10::Error 形状或 dtype 是内核索引不了的。
torch::Tensor gemv_kn_m(torch::Tensor wt, torch::Tensor x, int64_t threads,
                        int64_t unroll) {
    TORCH_CHECK(wt.dim() == 2, "weight must be 2-D [K, N]");
    TORCH_CHECK(x.dim() == 2, "activations must be 2-D [M, K]");
    TORCH_CHECK(wt.is_contiguous() && x.is_contiguous(), "contiguous inputs required");
    const int K = (int)wt.size(0), N = (int)wt.size(1), M = (int)x.size(0);
    TORCH_CHECK(wt.size(0) == x.size(1), "K mismatch: weight ", wt.size(0),
                " vs activation ", x.size(1));
    TORCH_CHECK(M >= 1 && M <= 8, "small-M kernel handles M in [1, 8], got ", M);
    const bool w16 = wt.scalar_type() == torch::kBFloat16;
    TORCH_CHECK(w16 || wt.scalar_type() == torch::kFloat,
                "weight must be bf16 or fp32 (an fp16 weight read as bf16 would "
                "compute silently wrong values)");
    const bool x16 = x.scalar_type() == torch::kBFloat16;
    TORCH_CHECK(x16 || x.scalar_type() == torch::kFloat,
                "activation must be bf16 or fp32");
    const int th = threads > 0 ? (int)threads : pick_threads(N);
    auto y = torch::empty({M, N}, x.options());
    if (M != 2 && M != 4 && M != 8) {
        // 非特化宽度：每行一次启动保持契约完整，代价是每行一次启动，没有解码路径会走到这里。
        for (int m = 0; m < M; ++m) {
            auto row = gemv_kn(wt, x[m].contiguous(), th, unroll > 0 ? unroll : 4, 1);
            y[m].copy_(row);
        }
        return y;
    }
    if (w16 && x16) {
        if (M == 2) return launch_m<uint16_t, uint16_t, 2>(wt, x, y, N, K, th);
        if (M == 4) return launch_m<uint16_t, uint16_t, 4>(wt, x, y, N, K, th);
        return launch_m<uint16_t, uint16_t, 8>(wt, x, y, N, K, th);
    }
    if (w16) {
        if (M == 2) return launch_m<uint16_t, float, 2>(wt, x, y, N, K, th);
        if (M == 4) return launch_m<uint16_t, float, 4>(wt, x, y, N, K, th);
        return launch_m<uint16_t, float, 8>(wt, x, y, N, K, th);
    }
    if (M == 2) return launch_m<float, float, 2>(wt, x, y, N, K, th);
    if (M == 4) return launch_m<float, float, 4>(wt, x, y, N, K, th);
    return launch_m<float, float, 8>(wt, x, y, N, K, th);
}
