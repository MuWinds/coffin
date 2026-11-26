// @file gemv_device.inc
// @brief ``gemv.cu`` 的**设备侧**部分：nk 解码投影内核、调优规则与启动器，两条前端共用。
//
// Module: coffin/backends/corex/kernels。被 ``gemv.cu``（pybind 前端）与
// ``abi/gemv_abi.cu``（C ABI 前端）各自在 ``namespace {`` 之内包含一次。
//
// 抽出来的理由与 ``norm_device.inc`` 相同：同一份算术不许存在两处。pybind 前端
// 负责 ``torch::Tensor`` 的参数整理（已迁出的部分留在 ``gemv.cu``），C ABI 前端负责
// 指针与返回码；两者包含的内核必须逐位一致，否则「换前端数值不变」这句话就无法验证。
//
// Constraints:
//   * 本文件不许出现 ``torch`` / ``c10`` / ``at::`` / ``pybind``：它要被一个**不链接
//     torch 的**编译单元包含。启动流由包含者在包含前定义 ``cx_stream()``。
//   * 每个启动器的签名都是裸指针，调用方负责保证布局（连续、行主序）与对齐；
//     语义级校验在各自前端里做。
//   * 不许在这里 ``#include`` 任何 torch 头；需要什么标准头由包含者提供。

#pragma once

// ``CxScalar`` 是这里每个内核都要的位级转换。它自带头文件与 ``#pragma once``，所以
// 无论是 ``gemv.cu`` 还是 ``moe_block.cu``（后者只为稠密路由用这里的 nk GEMV）包含
// 本文件，都不会因为漏了这行而编译失败——那正是「.inc 只提供设备代码」的代价。
#include "corex_scalar.h"

// nk 布局的解码投影：一列一行，K 在本行内连续，32 条通道认领一列。
//
// 为什么换布局。kn 那套内核（一个线程持一列、按步长 ``N`` 读）已整体删除，只留下面这套
// nk 实现。原先的记录：kn 布局下第 ``n`` 列与第 ``n + 1`` 列相距 ``N`` 个
// 元素，于是一个 wavefront 每步发出的是 32 个单字节请求、落在 32 个不同的行上。实测这个形状
// 族卡在 202-305 GB/s（``o_proj`` 51.4 us、``mlp_down`` 77.2 us），把所有旋钮（threads /
// unroll / split / width，共 180+96 格）一起拧也只值 2-6%，因为瓶颈是"每个线程每步能搬多少
// 连续字节"，不是块数。换成"一列一行、K 连续"之后，同一份字节量：2560x4096 31.6 us
// (332 GB/s，1.63x)、2560x9216 52.4 us (450 GB/s，1.47x)、2560x5120 35.4 us (371 GB/s，
// 1.60x)、12352x2560 61.8 us (512 GB/s，1.29x)、10240x2560 52.8 us (497 GB/s，1.25x)、
// 18432x2560 119.5 us (395 GB/s，1.09x)、248320x2560 1380.7 us (460 GB/s，1.02x)。512 GB/s
// 已经贴住这张卡 D2D copy 的 597.7 GB/s。
//
// 布局怎么来的：``pack.KnQWeight.wt`` 是 ``[K, N]``，投影一次得到 ``[N, K]``（每个权重一次，
// 十来 MB，几十 us），与 ``dp4a_weight`` 共用同一份副本。kn 那份不扔，因为预填充的库 GEMM 与
// 张量核 GEMM 都按 ``[K, N]`` 吃。
//
// Constraints:
//   * 32 条通道一组，不是整个 warp。本部件实测 ``warpSize == 32``，所以 ``__shfl_down_sync``
//     的默认宽度正好是一组；哪天真换到 64 宽 wavefront 的部件，这份归约要重写而不是调参数。
//   * ``K % 4 == 0``。每组每步读 4 个字节，而第 ``n`` 行的基址是 ``n * K``，K 不是 4 的倍数
//     时这些读会跨进下一行；入口直接拒绝，不猜。
//   * 累加 fp32、scale 在整列 K 和之后乘一次——与 kn 路径同一份契约，绝不逐元素乘、绝不折进
//     激活。
//   * 归约是树形的，所以与 kn 的长链累加不逐位相同（实测相对差 1e-3 量级，是 fp32 重结合的
//     正常范围）。验收按参考实现和 greedy ids，不按 kn 逐位相等。
//   * 不分片。列数就是块数：最窄的 ``N = 2560`` 在 128 线程下也有 640 个块，远超这张卡的 16
//     个 SM，再切 K 只是多一趟归约和一块分片缓冲区。

// @brief 4 个相邻元素的激活加载：bf16 是一次 8 字节读，fp32 逐元素读。
//
// fp32 不用 ``float4``：那要求 16 字节对齐，而激活可以是别的张量的切片，行基址
// ``r * K + off`` 只保证 16 字节对齐当 ``K % 4 == 0``，指针本身没有这个承诺。逐元素读让编译器
// 自己决定要不要向量化，代价只是 fp32 解码路径（不是这条路径的目标形状）。bf16 的 8 字节读
// 只要求 8 字节对齐，那是 ``K % 4 == 0`` 就能保证的。
template <typename XT>
struct NkLoad;

template <>
struct NkLoad<uint16_t> {
    static __device__ __forceinline__ void four(const uint16_t* p, uint16_t* o) {
        *reinterpret_cast<unsigned long long*>(o) =
            *reinterpret_cast<const unsigned long long*>(p);
    }
};

template <>
struct NkLoad<float> {
    static __device__ __forceinline__ void four(const float* p, float* o) {
#pragma unroll
        for (int i = 0; i < 4; ++i) o[i] = p[i];
    }
};

// @brief ``M`` 行激活对 ``[N, K]`` int8 权重，32 条通道持一个输出列；一组 ``W`` 列。
//
// @param wtn [N, K] int8 权重，行主序；一行就是一个输出列，K 在本行内连续。
// @param x [M, K] 激活，行主序。
// @param y [M, N] 输出，按激活 dtype。
// @param scale [N] fp32 每列 scale，在该列完整 K 和之后乘一次。
// @param N 输出列数，即权重的行数。
// @param K 归约长度，且 ``K % 4 == 0``。
// @param rows 真实行数；模板 ``M`` 里多出来的行重读第 0 行，算完就丢。
// @param W 每组列数：``1`` 时一个通道组（32 条通道）认领一列，``2`` 时认领相邻两列。
//
// ``W = 2`` 让一个通道组同时拉两条独立的权重行，并把每步的激活读摊到两列上，逐字节对比：
//
//   ``W = 1``，每 128 字节权重（一列一步）：1 次权重读 + M 次激活读 + 4M 次 FMA
//   ``W = 2``，每 256 字节权重（两列一步）：2 次权重读 + M 次激活读 + 8M 次 FMA
//
// 也就是 ``M >= 2`` 时每 128 字节权重的激活读从 M 次降到 M/2 次，同时在飞的权重字节
// 翻倍。压住 ``M = 8`` 那一档（R8）的不是 HBM：每 128 字节权重只有 1 次权重读，却有 8 次
// 激活读加 32 次 FMA，L1 与发射都排在权重流前面。代价是块数减半（``N / (4W)``，每个通道组
// 一个列），所以宽度由调用方按形状决定，不写死。
//
// ``N`` 不是 ``W`` 的倍数时，越界的那几列重读第 0 行、结果丢掉（同 ``M`` 多出来的行）。
template <typename XT, int U, int M, int W>
__global__ void gemv_nk_i8_kernel(const int8_t* __restrict__ wtn,
                                  const XT* __restrict__ x, XT* __restrict__ y,
                                  const float* __restrict__ scale, int N, int K,
                                  int rows, int wideon) {
    const int lane = threadIdx.x & 31;
    const int grp = threadIdx.x >> 5;
    const int n0 = (blockIdx.x * (blockDim.x >> 5) + grp) * W;
    if (n0 >= N) return;
    const int8_t* row[W];
#pragma unroll
    for (int w = 0; w < W; ++w) {
        const int n = n0 + w;
        row[w] = wtn + (size_t)(n < N ? n : 0) * K;
    }
    float acc[M][W];
#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
        for (int w = 0; w < W; ++w) acc[m][w] = 0.f;
    // 宽步：每 lane 一次读 16 个权重字节，一个 warp 就是**一条 512 字节的连续段**。
    //
    // 动机来自设备侧读数：这是 ``N = 12352 / K = 2048`` 那一档的核，25.3 MB 的权重
    // 一发要 **90 us（281 GB/s）**；把展开度 4 -> 8 完全不动（90.35 -> 89.82，见
    // docs/DECODE-LAUNCH-LEDGER.md §12.20），所以它不缺在飞请求，缺的是**每次请求多大**：
    // 4 字节步下每 warp 一步只有 128 字节，DRAM 侧不成段。
    //
    // **实测否掉了这个假设，而且是量出来的，不是推的**（作业 076_vb16b / 077_trnk6）：
    //   * ``cudaFuncGetAttributes`` 读回 regs 24 / 49 / 77、``local=0``（没有溢出到本地内存）、
    //     occ 64 / 39 / 25 blk/sm —— 寄存器与占用率**都不是**瓶颈；
    //   * 这个核**自己的时间轴**却从 90.35 掉到 **323.11 us（3.6x）**，另两档 36.91 -> 90.97、
    //     25.00 -> 37.33；整体 24.28 -> 38.84 ms、tok/s 41.18 -> 25.74。
    // 也就是说这张卡在**这个访问模式**上不奖励更宽的每 lane 请求：每行权重是一条独立流
    // （相邻 warp 的行相隔 K = 2048 字节），4 字节 x 32 lane 给出的 128 字节段能被 DRAM
    // 交错得更满；抬到 512 字节一条反而把可交错的段数砍掉 4 倍。
    //
    // 于是保留实现、**默认关闭**：``COFFIN_GEMV_NK_WIDE=1`` 才走这条（为了以后能复现与继续
    // 实验，比如只在该档以外的形状上试）。默认仍走下面逐字未改的 4 字节步。
    const bool wide16 = wideon != 0 && W == 1 && K % (32 * 16) == 0 &&
                        (reinterpret_cast<uintptr_t>(x) & 15u) == 0;
    if (wide16) {
        const int wstep = 32 * 16;
        for (int wo = 0; wo < K; wo += wstep) {
            const int off = wo + lane * 16;
            alignas(16) unsigned wq[4];
            *reinterpret_cast<uint4*>(wq) = *reinterpret_cast<const uint4*>(row[0] + off);
#pragma unroll
            for (int b = 0; b < 4; ++b) {  // 4 个 32 位字，每个字是 4 个连续 K 元素
                const unsigned word = wq[b];
                const int sub = off + b * 4;
#pragma unroll
                for (int m = 0; m < M; ++m) {
                    alignas(8) XT x4[4];
                    NkLoad<XT>::four(x + (size_t)(m < rows ? m : 0) * K + sub, x4);
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        // 低位字节在前：与 ``wtn`` 的 int8 行内顺序一致，符号要显式补。
                        const int wv = (int)(int8_t)((word >> (8 * i)) & 0xffu);
                        acc[m][0] = fmaf((float)wv,
                                         CxScalar<XT>::to_float(x4[i]), acc[m][0]);
                    }
                }
            }
        }
    } else {
    const int step = 32 * 4;
    const int steps = K / step;
    int s = 0;
    for (; s + U <= steps; s += U) {
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int off = (s + u) * step + lane * 4;
            // 一个 32 位字就是 4 个权重字节：本布局里它们连续，所以不需要任何重排。
            alignas(4) int8_t w4[W][4];
#pragma unroll
            for (int w = 0; w < W; ++w) {
                *reinterpret_cast<int*>(w4[w]) = *reinterpret_cast<const int*>(row[w] + off);
            }
#pragma unroll
            for (int m = 0; m < M; ++m) {
                // 激活读放在 ``m`` 环里、``w`` 环外：同一行激活被 ``W`` 列复用，
                // 这正是宽度要摊的那一项。
                alignas(8) XT x4[4];
                NkLoad<XT>::four(x + (size_t)(m < rows ? m : 0) * K + off, x4);
#pragma unroll
                for (int w = 0; w < W; ++w) {
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        acc[m][w] = fmaf((float)w4[w][i], CxScalar<XT>::to_float(x4[i]),
                                         acc[m][w]);
                    }
                }
            }
        }
    }
    for (; s < steps; ++s) {
        const int off = s * step + lane * 4;
        alignas(4) int8_t w4[W][4];
#pragma unroll
        for (int w = 0; w < W; ++w) {
            *reinterpret_cast<int*>(w4[w]) = *reinterpret_cast<const int*>(row[w] + off);
        }
#pragma unroll
        for (int m = 0; m < M; ++m) {
            alignas(8) XT x4[4];
            NkLoad<XT>::four(x + (size_t)(m < rows ? m : 0) * K + off, x4);
#pragma unroll
            for (int w = 0; w < W; ++w) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    acc[m][w] = fmaf((float)w4[w][i], CxScalar<XT>::to_float(x4[i]),
                                     acc[m][w]);
                }
            }
        }
    }
    // K 不是 128 的整数倍时，尾巴按通道跨步走标量，保证任何 K 都正确。
    //
    // 试过把这里整段换成「每 lane 一次 16 字节」的宽步：第一版（16 元素临时数组）
    // **实测整体 24.28 -> 38.79 ms、文本坏成混语乱串**（作业 074_vb16）。宽步本身仍
    // 由上面的紧凑版保留，这里只是 4 字节步的兜底：任何形状都对。
    for (int k = steps * step + lane; k < K; k += 32) {
#pragma unroll
        for (int m = 0; m < M; ++m) {
#pragma unroll
            for (int w = 0; w < W; ++w) {
                acc[m][w] = fmaf((float)row[w][k],
                                 CxScalar<XT>::to_float(
                                     x[(size_t)(m < rows ? m : 0) * K + k]),
                                 acc[m][w]);
            }
        }
    }
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
#pragma unroll
        for (int w = 0; w < W; ++w) {
            float v = acc[m][w];
            for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
            const int n = n0 + w;
            if (lane == 0 && m < rows && n < N) {
                y[(size_t)m * N + n] = CxScalar<XT>::from_float(scale[n] * v);
            }
        }
    }
}

// @brief 打一行 nk 密集投影内核的 ``regs / local / occ``，``COFFIN_KATTR=1`` 门控。
//
// ``local`` 是关键：宽步第一版（每 lane 16 字节、16 元素临时数组）让整体从 24.28
// 掉到 38.79 ms、文本也坏了（作业 074_vb16）。判断「是不是临时数组把寄存器挤爆、
// 溢出到本地内存」不需要推墙钟——``cudaFuncAttributes::localSizeBytes`` 就是那个数。
// 长时间保留：以后再给这个核加临时数组，先看这一行有没有非零的 ``local``。
template <typename XT, int U, int M, int W>
void nk_gemv_report(const char* tag, int threads) {
    static bool once = false;
    if (once || std::getenv("COFFIN_KATTR") == nullptr) return;
    once = true;
    const void* fn = (const void*)&gemv_nk_i8_kernel<XT, U, M, W>;
    cudaFuncAttributes at{};
    int blocks = -1;
    if (cudaFuncGetAttributes(&at, fn) == cudaSuccess) {
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, fn, threads, 0)
                != cudaSuccess) {
            blocks = -2;
        }
    }
    std::printf("[kattr] %s regs=%d local=%zu smem=%zu threads=%d occ=%d blk/sm\n",
                tag, (int)at.numRegs, (size_t)at.localSizeBytes,
                (size_t)at.sharedSizeBytes, threads, blocks);
    std::fflush(stdout);
}

// @brief 宽步开关：``COFFIN_GEMV_NK_WIDE=1`` 才走每 lane 16 字节那一路，默认关。
//
// 默认关的理由写在 ``gemv_nk_i8_kernel`` 里那段：同一个核、同样的寄存器与占用率，
// 每 lane 从 4 字节抬到 16 字节后它自己的时间轴反而 90.35 -> 323.11 us。
static bool nk_wide_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("COFFIN_GEMV_NK_WIDE");
        return v != nullptr && *v == '1';
    }();
    return on;
}

// @brief 启动一个 ``M`` 行的 nk 单趟内核：``U``（展开度）与 ``W``（每组列数）都是编译期常量。
//
// ``W = 2`` 在分派处只与 ``U = 4`` 搭配：它是一档按形状选的宽度，不是与展开度正交的旋钮，
// 两个维度都乘上去会让这个文件的实例数从 24 涨到 48，而 ``U = 8`` 与 ``W = 2`` 同时开没有
// 测过。所以 ``unroll`` 非默认值时一律按 1 列走，调用方拿到的是正确结果而不是报错。
template <typename XT, int M, int U, int W>
void launch_nk_i8_one(const int8_t* wp, const XT* xp, XT* yp, const float* sp,
                      int N, int K, int rows, int threads) {
    const cudaStream_t stream = cx_stream();
    const int grps = threads / 32;
    const int blocks = (int)((N + W * grps - 1) / (W * grps));
    nk_gemv_report<XT, U, M, W>("gemv_nk_i8", threads);
    gemv_nk_i8_kernel<XT, U, M, W><<<blocks, threads, 0, stream>>>(
        wp, xp, yp, sp, N, K, rows, nk_wide_enabled() ? 1 : 0);
}

// @brief 按展开度与列宽启动 ``M`` 行的 nk 内核。
template <typename XT, int M>
void launch_nk_i8(const int8_t* wp, const XT* xp, XT* yp, const float* sp,
                  int N, int K, int rows, int threads, int unroll, int width) {
    if (unroll == 2) {
        launch_nk_i8_one<XT, M, 2, 1>(wp, xp, yp, sp, N, K, rows, threads);
    } else if (unroll == 8) {
        launch_nk_i8_one<XT, M, 8, 1>(wp, xp, yp, sp, N, K, rows, threads);
    } else if (width == 2) {
        launch_nk_i8_one<XT, M, 4, 2>(wp, xp, yp, sp, N, K, rows, threads);
    } else {
        launch_nk_i8_one<XT, M, 4, 1>(wp, xp, yp, sp, N, K, rows, threads);
    }
}

// @brief 按行数与激活 dtype 分派 nk 内核。
//
// 行数取 1/2/4/8 四档，与 kn 的批处理入口同一套桶：批处理解码与投机校验记录要的就是这几档，
// 其它行数由上一档承接，多出来的行重读第 0 行。
template <typename XT>
void launch_nk_i8_rows(const int8_t* wp, const XT* xp, XT* yp, const float* sp,
                       int N, int K, int rows, int threads, int unroll, int width) {
    const int padded = rows <= 1 ? 1 : (rows <= 2 ? 2 : (rows <= 4 ? 4 : 8));
    if (padded == 1) {
        launch_nk_i8<XT, 1>(wp, xp, yp, sp, N, K, rows, threads, unroll, width);
    } else if (padded == 2) {
        launch_nk_i8<XT, 2>(wp, xp, yp, sp, N, K, rows, threads, unroll, width);
    } else if (padded == 4) {
        launch_nk_i8<XT, 4>(wp, xp, yp, sp, N, K, rows, threads, unroll, width);
    } else {
        launch_nk_i8<XT, 8>(wp, xp, yp, sp, N, K, rows, threads, unroll, width);
    }
}

// @brief 每块线程数：128 就是 4 组。实测这一档在所有七个形状上都与最好档差 1-3%，而它给最窄
//        的 ``N = 2560`` 留出 640 个块，所以不按 ``N`` 再分一档。
int pick_nk_threads(int N) {
    (void)N;
    return 128;
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

//: 窄形状（``N < GEMV_NARROW_MIN_N``）的 split-K 目标块数。
//:
//: 一个分片就是网格上的 ``blocks`` 个块，所以 ``split * blocks`` 才是总块数。窄形状
//: 单趟只有 1-5 个块，而 16 SM 的部件一个 wave 就是 16 个块——**整张卡上跑着的块数
//: 比一个 wave 还少**，这就是它们慢的全部原因。
//:
//: 64 = 4 个 wave。实测（``diag/qwen4/probe_q4_gemv_dispatch.py``，M = 1）：
//:
//:   split      1       2       4       8      16      32      64
//:   N=48    200.7   174.1    94.4    54.5    35.8    29.3    29.6   （us）
//:   块数        1       2       4       8      16      32      64
//:
//: 平台在 32 块（16 SM 的 2 倍 wave），s32 与 s64 相差 1%。目标取 64 而不是 32，
//: 是因为目标块数会被 :data:`GEMV_MIN_SLICE` 再封顶一次，而封顶之后 N=48 恰好落在
//: 40 片——仍在 s32/s64 那个平底盆里。
//:
//: **这三个窄形状常量（本项、``GEMV_NARROW_SLICE``、``GEMV_NARROW_MIN_N``）是拟合，
//: 不是推导：用它们拟合 4 个窄形状（``gdn_in_proj_a``、``shared_expert_gate``、
//: ``block_inject``、``hyper_mix_down``），比例并不好。** 之所以还是这样写而不是硬编码
//: 一张表，是因为窄形状的 ``N``/``K`` 本来就不是整数（``N = 1``、``4``、``48``、``320``），
//: 写下个模型的形状表一个都盖不住。任何要动它们的人先看这段：它是一条**样本不足的
//: 拟合**，改它之前请先补至少三个不同 ``K`` 的窄形状实测。
constexpr int GEMV_NARROW_TARGET_BLOCKS = 64;

//: 窄形状每个分片能覆盖的 K 元素软上限。它管的是「长 K + 窄 N」这一类形状：
//: ``hyper_mix_down``（N = 320, K = 10240）在 5 个块上单趟要串 10240 个元素，实测
//: s8 = 177.2 us 而 s64 = 44.3 us（**4 倍**）。160 这个值让它拿到 64 片。
//:
//: 与 :data:`GEMV_MIN_SLICE` 的区别：那个是**硬**下界（低于它启动开销盖不住收益），
//: 这个是**软**上限（超过它串行链太长），所以前者用 min 夹、后者用 max 提。
constexpr int GEMV_NARROW_SLICE = 160;

//: 走「窄形状」规则的最大列数。``N >= 2048`` 时 ``pick_threads`` 给 256 线程，单趟
//: 就有 8 个块以上，补块数的需求消失，而历史实测（下面 ``pick_split`` 的注释表）给出
//: 宽形状的最优就是 4-8——所以上界以上一律沿用历史规则，一个字不改。
constexpr int GEMV_NARROW_MIN_N = 2048;

//: bf16 分片的硬上限，同时也是分片缓冲区的界。归约要花 ``8 * split * N`` 字节，
//: ``split = 64`` 时在 ``N = 248320`` 上也只有 127 MB 量级的一次性流量。
constexpr int GEMV_MAX_SPLIT = 64;

//: 值得调度的最短 K 切片。低于约 64 个元素时，启动和部分和的写入会超过切片本身读的量。
//: 它同时封顶了分片缓冲区的流量：``split <= K / 64`` 把归约开销封在权重流量的
//: ``8 / 64`` = 12.5%。
constexpr int GEMV_MIN_SLICE = 64;

// @brief Split-K 宽度，或 ``1`` 表示跑单趟内核。
//
// 切分乘倍了块数却不碰权重流量：每个分片读每列一个不同的 K 区间，所以搬动的字节数相同，
// 变的只是同时在飞的独立加载数与总块数。
//
// **两段式**，因为两类形状的最优由不同的量决定，用一条连续规则拟合会两边都不准：
//
// 宽形状（``N >= GEMV_NARROW_MIN_N``）——列数自己就能铺满卡，分片只用来摊短 K 切片。
// 这一段是**历史规则原样保留**，它的数字来自下面这张表（经 ``backends.ops.linear``
// 测得，含启动与归约）：
//
//   shape      N       K      split 1      split 4      split 8
//   mlp_up     9216    2560   0.292 ms     0.152 ms     --
//   qkv_gate   8192    2560   0.286 ms     0.160 ms     --
//   mlp_down   2560    9216   1.004 ms     --           0.245 ms
//   o_proj     2560    4096   0.329 ms     --           0.085 ms
//   lm_head    248320  2560   3.478 ms     2.961 ms     --
//
// 那张表里打进了两次纠正：「只在单趟网格填不满卡时才切分」是错的（mlp_up 的 36 个块
// 已经两个 wave，四路切分仍有 1.92x）；「窄形状没收益」也是反的（mlp_down / o_proj
// 才是切分最值的，对单趟 4.1x / 3.9x）。所以宽形状这一段**不动**：任何重写都要先有
// 新实测，而本次解码账本里的 264 次调用全是窄形状。
//
// 窄形状（``N < GEMV_NARROW_MIN_N``）——单趟只有 1-5 个块，**整张卡上的块数比一个
// wave 还少**。这一段是本次修正的重点，旧规则在这里是反的：
//
//   shape                 N     K      blocks  旧 split  实时最优  旧规则代价
//   gdn_in_proj_a/b       48    2560     1        5         32       2.79x
//   shared_expert_gate    1     2560     1        5         32       --
//   block_inject          4     10240    1        8         32       --
//   hyper_mix_down        320   10240    5        8         64       4.00x
//
// 旧规则给 ``want = 8``（``N < 4096``）再被 ``by_k = K / 512`` 上限 8 夹住，所以
// ``N = 48, K = 2560`` 只拿到 **5 片**——5 个块、320 条线程，而 16 个 SM 的容量是它的
// 两个数量级倍。实测 s5 = 78.3 us，s32 = 29.3 us。
//
// **``block_inject`` 与 ``hyper_mix_down`` 会被融合删掉，这一段对它们生效是窄形状规则
// 的必然结果，不是专门优化。** 前者（``N = 4``, ``K = 10240``, 96 次/步）算的正是
// ``hyper.cu`` 里 ``hyper_up`` 内部的逐流写门，后者（``N = 320``, ``K = 10240``,
// 24 次/步）是 ``hyper_down`` 的下投影；融合上线后这 120 次调用**不是变快，是不存在**。
// 所以按需删除：删掉这两个形状对本规则无影响（它没有任何形状特例），但要一起确认
// ``gdn_in_proj_a/b`` 与 ``shared_expert_gate`` 仍然吃到 32 片以上。
//
// 数值影响已经量化（作业 ``q4layer4``，同一份 .so、两臂只差一个环境变量、都 12 步）：
// 分派从 5/8 片改到 32/40 片之后，末层 logits 的 absmax 是 12.375 对 12.625（2.0%）、
// std 是 1.086 对 1.132（4.1%），**greedy ids 12/12 逐位一致**。逐层看，首个分叉在
// 第 2 层（第 0、1 层逐位相同，说明两次运行本身是确定性的），48 层的总放大倍数 9.2x、
// 最大单层放大 2.79x。所以差异是 **fp32 部分和累加顺序**的必然结果（split 改变的是
// 加法的结合方式，而 fp32 加法不满足结合律），不是实现错误。
//
// 性能侧：作业 ``q4disp6`` 是 234.50 ms、作业 ``q4final2``（删掉 A/B 开关之后重跑）
// 是 233.13 ms，**差 0.6%，在跨作业抖动内**，而两个核的 us/call 分别为
// ``gemv_kn_split`` 27.2 对 27.17、量化投影 46.9 对 46.88——到小数点后一位一致。
// 所以 234.50 不是单次幸运值，删开关也没有改变行为。
//
// 顺带一条排错经验：这个结论第一次没拿到，因为 A/B 对照臂用**文本手术**构造失败了两次
// （sed 改常量在数学上不可能等价；花括号配对替换函数体吞掉了相邻的 ``pick_split_i8``）。
// 两次的症状都不在操作处。构造对照臂要么用同一份产物的运行时开关，要么用两份独立源码，
// 不要用文本替换。
//
// @param N 输出列数。
// @param K 归约长度；它通过 :data:`GEMV_MIN_SLICE` 给分片数设硬下界。
// @param threads 每块线程数，它决定单趟块数。
// @return ``[1, GEMV_MAX_SPLIT]`` 里的分片数。
int pick_split(int64_t N, int64_t K, int threads) {
    if (K < 1024) return 1;
    const int64_t blocks = threads > 0 ? (N + threads - 1) / threads : N;
    if (blocks <= 0) return 1;
    const int64_t by_k = K / GEMV_MIN_SLICE;
    if (by_k < 2) return 1;

    if (N >= GEMV_NARROW_MIN_N) {
        // 宽形状：历史规则逐字保留（含 ``N >= 4096 ? 4 : 8`` 与 ``by_k`` 的 8 上限）。
        const int64_t want = N >= 4096 ? 4 : 8;
        const int64_t cap = by_k < 8 ? by_k : 8;
        return (int)(want < cap ? want : cap);
    }

    // 窄形状：两个来源各给出一个下界，取大的那个。
    //   补块数 —— 单趟块数越少，需要越多分片才能凑到 4 个 wave；
    //   摊长 K —— 每片覆盖的 K 元素超过一个软上限时，串行链就成了瓶颈。
    const int64_t by_blocks = (GEMV_NARROW_TARGET_BLOCKS + blocks - 1) / blocks;
    const int64_t by_slice = (K + GEMV_NARROW_SLICE - 1) / GEMV_NARROW_SLICE;
    int64_t want = by_blocks > by_slice ? by_blocks : by_slice;
    if (want < 2) want = 2;
    if (want > by_k) want = by_k;
    if (want > GEMV_MAX_SPLIT) want = GEMV_MAX_SPLIT;
    return (int)want;
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

// 稠密（bf16/fp32）权重的 nk 投影：一行一个输出列，K 在本行内连续。
//
// 只有一处要它：``moe_block.cu`` 里稠密的路由权重（``mlp.gate.weight`` 在
// ``DENSE_SUFFIXES`` 里——量化噪声会翻转专家选择，所以它保持 bf16）。路由形状是
// ``hidden -> experts``（2048 x 256 上下），整份权重 1 MB 级，所以这里不追带宽上限，
// 只要一条 nk 原生的路径：**不吃 ``[K, N]``**，也就不用为它留一份 kn 副本。
//
// 结构与 ``gemv_nk_i8_kernel`` 一致：32 条通道认领一列，沿 K 跨步 32，最后用
// ``__shfl_down_sync`` 在 32 条通道内归约（这台卡 warpSize 是 64，归约单位仍取 32，
// 与 i8 那个核完全相同，所以两路的加法顺序一致）。
//
// @param wtn ``[N, K]`` 稠密权重，连续。
// @param x   ``[M, K]`` 激活，连续，dtype 同权重。
// @param y   ``[M, N]`` 输出，dtype 同激活。
// @param rows 有效行数，``M`` 多出来的行重读第 0 行、结果丢掉。
template <typename XT, int M>
__global__ void gemv_nk_dense_kernel(const XT* __restrict__ wtn,
                                     const XT* __restrict__ x, XT* __restrict__ y,
                                     int N, int K, int rows) {
    const int lane = threadIdx.x & 31;
    const int grp = threadIdx.x >> 5;
    const int n = blockIdx.x * (blockDim.x >> 5) + grp;
    if (n >= N) return;
    const XT* row = wtn + (size_t)n * K;
    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.f;
    // 4 元素向量化路径（bf16 每 lane 8 字节、每组 256 字节连续读，比逐元素 2 字节读少 4 倍发射）；
    // 不满足 4 倍数或 8 字节对齐时退回标量循环，保证任意形状与切片都正确。
    const bool vec4 = (K % 4 == 0) &&
        ((reinterpret_cast<uintptr_t>(row) & 7u) == 0) &&
        ((reinterpret_cast<uintptr_t>(x) & 7u) == 0);
    if (vec4) {
        const int step = 32 * 4;
        const int steps = K / step;
        int s = 0;
        for (; s + 4 <= steps; s += 4) {
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int off = (s + u) * step + lane * 4;
                alignas(8) XT w4[4];
                NkLoad<XT>::four(row + off, w4);
#pragma unroll
                for (int m = 0; m < M; ++m) {
                    alignas(8) XT x4[4];
                    NkLoad<XT>::four(x + (size_t)(m < rows ? m : 0) * K + off, x4);
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        acc[m] = fmaf(CxScalar<XT>::to_float(w4[i]),
                                      CxScalar<XT>::to_float(x4[i]), acc[m]);
                    }
                }
            }
        }
        for (; s < steps; ++s) {
            const int off = s * step + lane * 4;
            alignas(8) XT w4[4];
            NkLoad<XT>::four(row + off, w4);
#pragma unroll
            for (int m = 0; m < M; ++m) {
                alignas(8) XT x4[4];
                NkLoad<XT>::four(x + (size_t)(m < rows ? m : 0) * K + off, x4);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    acc[m] = fmaf(CxScalar<XT>::to_float(w4[i]),
                                  CxScalar<XT>::to_float(x4[i]), acc[m]);
                }
            }
        }
        for (int k = steps * step + lane; k < K; k += 32) {
            const float wv = CxScalar<XT>::to_float(row[k]);
#pragma unroll
            for (int m = 0; m < M; ++m) {
                acc[m] = fmaf(wv, CxScalar<XT>::to_float(x[(size_t)(m < rows ? m : 0) * K + k]),
                              acc[m]);
            }
        }
    } else {
        for (int k = lane; k < K; k += 32) {
            const float wv = CxScalar<XT>::to_float(row[k]);
#pragma unroll
            for (int m = 0; m < M; ++m) {
                acc[m] = fmaf(wv, CxScalar<XT>::to_float(x[(size_t)(m < rows ? m : 0) * K + k]),
                              acc[m]);
            }
        }
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
        float v = acc[m];
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if (lane == 0 && m < rows) y[(size_t)m * N + n] = CxScalar<XT>::from_float(v);
    }
}

// @brief 按行数挑一个内核实例化启动稠密 nk 投影。
template <typename XT>
void launch_nk_dense_rows(const XT* wp, const XT* xp, XT* yp, int N, int K, int rows,
                         int threads) {
    const int padded = rows <= 1 ? 1 : (rows <= 2 ? 2 : (rows <= 4 ? 4 : 8));
    const int grps = threads > 0 ? threads : 128;
    const int blocks = (N + grps / 32 - 1) / (grps / 32);
    if (padded == 1) {
        gemv_nk_dense_kernel<XT, 1><<<blocks, grps, 0, cx_stream()>>>(wp, xp, yp, N, K, rows);
    } else if (padded == 2) {
        gemv_nk_dense_kernel<XT, 2><<<blocks, grps, 0, cx_stream()>>>(wp, xp, yp, N, K, rows);
    } else if (padded == 4) {
        gemv_nk_dense_kernel<XT, 4><<<blocks, grps, 0, cx_stream()>>>(wp, xp, yp, N, K, rows);
    } else {
        gemv_nk_dense_kernel<XT, 8><<<blocks, grps, 0, cx_stream()>>>(wp, xp, yp, N, K, rows);
    }
}
