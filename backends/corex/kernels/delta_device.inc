// @file delta_device.inc
// @brief 门控 delta（GDN）的递推与重放：两个内核加一个共用启动器。
//
// Module: coffin/backends/corex/kernels。被两条前端各包含一次：``delta.cu``（torch
// 扩展，五个 pybind 入口）与 ``abi/delta_abi.cu``（两个 ``extern "C"`` 出口）。同一份
// 设备代码。
//
// 为什么值得搬：线性注意力每层每步都要走一次递推，decode 时 ``rows = 1``，全卡只有
// ``heads`` = 32 个块在飞，2 MB 状态走一次约 64 us。它是这条链上唯一持有可变状态的内核。
//
// Constraints:
//   * ``st`` 与 ``out`` 必须**连续**：行偏移按 ``b * heads * dk * dv`` 与 ``b * heads * dv``
//     算，加宽的视图会让每一行读错地方，而症状只是批处理路径上一个很小的相对误差，
//     单行形式却是精确的。
//   * 线程按「同一 ``i`` 上连续的 ``j``」编址：一个 64 通道 wavefront 读 64 个连续
//     float，一次 256 字节事务。早先的映射每通道碰一条不同的 128 字节行、只用四个字节，
//     同样的答案 32 倍流量。
//   * ``DELTA_WAY`` 是**块数**旋钮而不是链长旋钮：解码期行数是 1、头数固定，唯一的
//     办法是把值轴切段，让网格变成 (heads, blocks_y)。
//   * ``record`` 可以为空指针：那就是不记录的路径（三个不记录的入口共用这条）。

#ifndef COFFIN_DELTA_DEVICE_INC
#define COFFIN_DELTA_DEVICE_INC

#include <cuda_runtime.h>
#include <cstdint>

constexpr int THREADS = 128;

//: 一条值列上的 ``dk`` 链由几个线程分段持有。
//:
//: 这条常量的来历：原来一个线程独占一列、把 ``dk`` 来回走两遍，靠「栈上无通信」换正确性，
//: 代价是解码期 ``rows = 1`` 时全卡只有 ``heads`` = 32 个块在飞、每块约 1 GB/s，2 MB 状态
//: 走 64.7 us（时间轴 traced 69.7）。块数没法加（头数固定、行数是 1），唯一的办法是把这条
//: 链切短。4 路后每块 512 线程、每条链从 ``2 * dk`` 缩到 ``2 * dk / 4``，两个跨线程归约各
//: 一次 ``__syncthreads()`` + 一次长度为 4 的串行求和。
constexpr int DELTA_WAY = 4;

// @brief 门控 delta 规则的一步：每列值一个线程，每个头一个 block。
//
// 每个线程独占第 ``j`` 列，并在寄存器循环里把 ``i`` 走过 ``dk``，所以两次跨 ``dk`` 的归约
// 都是那个线程本地的事，完全不发生跨通道通信。这之所以正确，只因为两个和在各列之间独立
// ——递推在一列内部耦合 ``i``，从不跨列。
//
// 网格的 ``z`` 是批行。每一行都是独立的 ``[heads, dk, dv]`` 板，所以 block 只是偏移它的
// 指针，不去碰别的行拥有的东西；没有需要补的跨行通信。实测这一点远比它替代的那次启动
// 值得：``delta_batch`` 每行一次启动，每次只有 ``heads`` = 32 个 block 在飞，而本部件每个
// block 只跑得动约 3.0 GB/s，于是一行的 6.29 MB 以 101.7 GB/s 走完（61.89 us）。把行放到
// ``z`` 轴上就能让 ``32 * rows`` 个 block 同时在飞。
//
// @param q,k [rows, heads, dk] fp32，已经由预处理做过 L2 归一化（q 还做了缩放）。
// @param v [rows, heads, dv] fp32。
// @param beta,g [rows, heads] fp32，写强度和对数衰减。
// @param st [rows, heads, dk, dv] fp32 状态，原地更新。
// @param out [rows, heads, dv] fp32，从更新后的状态写出。
// @param heads 每行的头数，也是 ``x`` 方向的网格长度；下面的行步长由它推出，而早先的
//        每行主机循环是自己施加这些步长的。
// @param dk 归约长度，同时也是相邻值列之间的步长。
// @param dv 每个头的值列数。
template <int WAY>
__global__ void delta_kernel(const float* __restrict__ q, const float* __restrict__ k,
                             const float* __restrict__ v, const float* __restrict__ beta,
                             const float* __restrict__ g, float* __restrict__ st,
                             float* __restrict__ out, float* __restrict__ record,
                             int record_slot, int record_width,
                             int heads, int dk, int dv) {
    const int head = blockIdx.x;
    // 一条列由 ``WAY`` 个线程共同持有，每个线程负责 ``dk`` 的一段。这样做的唯一理由是
    // **这条串行链**：解码期 ``rows = 1``，网格只有 ``heads`` = 32 个块，而每个线程要把
    // ``dk`` 走两遍（先求记忆再更新），实测每块只有约 1 GB/s，2 MB 状态走了 64.7 us。
    // 块数加不上去（头数固定、行数是 1），能加的只有「一条链上有几个线程」。
    const int ncol = (int)blockDim.x / WAY;
    const int j = (int)blockIdx.y * ncol + (int)threadIdx.x % ncol;
    const int part = (int)threadIdx.x / ncol;
    const int iper = (dk + WAY - 1) / WAY;
    const int i0 = part * iper;
    const int i1 = i0 + iper < dk ? i0 + iper : dk;

    // 行偏移在设备侧算，不在主机侧做指针算术：缓冲区是稠密的，行步长就是形状本身，所以
    // 内核不用间距参数、也不用第二次启动就能命名第 ``b`` 行。单行解码入口下 ``z`` 是 1，
    // 第 0 行给每个指针加零，所以那条路径读到的字节和以前完全一样。
    const size_t row = blockIdx.z;
    const size_t oq = row * heads * dk;          // q 和 k，每行 [heads, dk]
    const size_t ov = row * heads * dv;          // v 和 out，每行 [heads, dv]
    const size_t ob = row * heads;               // beta 和 g，每行 [heads]
    const size_t os = row * heads * dk * dv;     // 状态，每行 [heads, dk, dv]

    // 超出 ``dv`` 的线程绝不能在 *这里* 返回。下面的 ``__syncthreads()`` 要由块内每个线程
    // 走到，提前退出会让其余线程等一个已经不在的线程——设备侧死锁，没有 traceback，正是
    // 曾经把一张卡挂满整个任务时限的那种症状。越界的列改为走完屏障，只跳过它自己的状态和
    // 输出访问。
    const bool active = j < dv;

    extern __shared__ float sh[];
    float* ks = sh;                  // [dk] 本头的 key，所有列共用
    float* qs = sh + dk;             // [dk] 本头的 query
    // 每一列都需要整个 ``dk``，所以这份拷贝是共享的而不是每线程各来一遍。加载时线程协作，
    // 两种方式下都是连续的。
    for (int i = (int)threadIdx.x; i < dk; i += (int)blockDim.x) {
        ks[i] = k[oq + (size_t)head * dk + i];
        qs[i] = q[oq + (size_t)head * dk + i];
    }
    __shared__ float decay_sh;
    __shared__ float rate_sh;
    if (threadIdx.x == 0) {
        decay_sh = __expf(g[ob + head]);
        rate_sh = beta[ob + head];
    }
    __syncthreads();
    const float decay = decay_sh;
    const float rate = rate_sh;
    float* head_record = nullptr;
    if (record) {
        const int record_dim = dk + dv + 1;
        head_record = record +
            ((row * (size_t)record_width + record_slot) * heads + head) * record_dim;
        if (blockIdx.y == 0) {
            if (threadIdx.x == 0) head_record[dk + dv] = decay;
            for (int i = (int)threadIdx.x; i < dk; i += (int)blockDim.x) {
                head_record[i] = ks[i];
            }
        }
    }
    // 越界列只跳过它自己的状态与输出访问，**绝不提前退出**：下面新增的四处
    // ``__syncthreads()`` 必须由块内每个线程走到（同上面那条「提前退出 = 设备侧死锁」
    // 的注释）。
    const float value = active ? v[ov + (size_t)head * dv + j] : 0.f;
    float* column = st + os + (size_t)head * dk * dv + j;
    float* red = sh + 2 * dk;   // [WAY, dv] 分片部分和

    // 第一遍：本线程那一段 ``dk`` 上的记忆。``WAY`` 份部分和在共享里合一次，再回读。
    float kv_mem = 0.f;
    if (active) {
#pragma unroll 8
        for (int i = i0; i < i1; ++i) {
            kv_mem = fmaf(column[(size_t)i * dv] * decay, ks[i], kv_mem);
        }
        red[part * dv + j] = kv_mem;
    }
    __syncthreads();
    if (active && part == 0) {
        float total = red[j];   // part 0 自己那一份就在 ``red[j]``
        for (int p = 1; p < WAY; ++p) total += red[p * dv + j];
        red[j] = total;
    }
    __syncthreads();
    const float kv_total = active ? red[j] : 0.f;
    // 回读必须在别人覆盖 ``red`` 之前全部完成：第二遍的部分和要写同一个数组。
    __syncthreads();

    const float delta = (value - kv_total) * rate;
    float result = 0.f;
    if (active) {
#pragma unroll 8
        for (int i = i0; i < i1; ++i) {
            const float s = fmaf(ks[i], delta, column[(size_t)i * dv] * decay);
            column[(size_t)i * dv] = s;
            result = fmaf(s, qs[i], result);
        }
        red[part * dv + j] = result;
    }
    __syncthreads();
    if (active && part == 0) {
        float total = red[j];
        for (int p = 1; p < WAY; ++p) total += red[p * dv + j];
        if (head_record) head_record[dk + j] = delta;
        out[ov + (size_t)head * dv + j] = total;
    }
}

// @brief 在 ``rows`` 块独立状态板上启动递推。
//
// 每个（头，列分块，行）一个 block。``z`` 方向就是行，单行解码入口下为 1，所以两个入口
// 共用这同一种启动形状，解码路径是靠构造保持不变，而不是靠另一条代码路径。
//
// 网格故意取行轴而不是主机循环：本部件每个在飞的 block 跑得动约 3.0 GB/s
// （``corex-kernel-check/docs/DEVICE.md``），这里一行只有 ``heads`` = 32 个 block，所以循环不管多少行
// 都只能以 32-block 的带宽跑每一行。
//
// @param q,k [rows, heads, dk] fp32。
// @param v [rows, heads, dv] fp32。
// @param beta,g [rows, heads] fp32。
// @param st [rows, heads, dk, dv] fp32，原地更新。
// @param out [rows, heads, dv] fp32。
// @param rows 批行数；成为网格的 ``z`` 方向长度。
// @param stream 启动所用的流。
void launch_rows(const float* q, const float* k, const float* v, const float* beta,
                 const float* g, float* st, float* out, float* record, int record_slot,
                 int record_width, int rows, int heads, int dk, int dv, cudaStream_t stream) {
    // 块大小仍是 ``THREADS``，但那 128 个线程不再覆盖 128 列，而是 ``THREADS / DELTA_WAY``
    // = 32 列 × 4 路。第一版只加了路数（128 -> 512 线程、每条链缩到 1/4），实测
    // `delta_kernel<4>` 只从 69.70 掉到 59.88 us——因为瓶颈不是链长而是**每块带宽**：
    // 本部件每个在飞的 block 只跑得动约 3 GB/s（见下面 ``launch_rows`` 的注释），而解码期
    // 一行只有 ``heads`` = 32 个块。所以真正要加的是**块数**：把值轴切成 4 段后
    // 网格变成 (32, 4) = 128 个块，共享里那份 q/k 副本因此被读 4 遍——每份 2 KB，可以忽略。
    const int ncol = dv < THREADS / DELTA_WAY ? dv : THREADS / DELTA_WAY;
    const int blocks_y = (dv + ncol - 1) / ncol;
    const int threads = ncol * DELTA_WAY;
    const size_t smem = (size_t)(2 * dk + DELTA_WAY * dv) * sizeof(float);
    delta_kernel<DELTA_WAY><<<dim3(heads, blocks_y, rows), threads, smem, stream>>>(
        q, k, v, beta, g, st, out, record, record_slot, record_width, heads, dk, dv);
}

__global__ void delta_commit_kernel(const float* __restrict__ base,
                                    const float* __restrict__ record,
                                    float* __restrict__ st, int heads, int dk, int dv,
                                    int steps) {
    const int head = (int)blockIdx.x;
    const int j = (int)(blockIdx.y * blockDim.x + threadIdx.x);
    if (j >= dv) return;

    const int record_dim = dk + dv + 1;
    const size_t state_head = (size_t)head * dk * dv;
    const size_t record_head = (size_t)head * record_dim;
    // 每线程沿 i 读写一个 4 B 标量，并让相邻线程覆盖连续的 j。自动向量化的 W=4 与手写
    // ``float4`` 都只有 182.9 GB/s，4 B 标量为 435.2 GB/s；两种写法结果相同，说明惩罚来自
    // ISA 的 16 B 访问路径。大块基态备份仍由 ``copy_`` 的 DMA 完成。
#pragma unroll 8
    for (int i = 0; i < dk; ++i) {
        float value = base[state_head + (size_t)i * dv + j];
#pragma unroll 8
        for (int t = 0; t < steps; ++t) {
            const float* step = record +
                ((size_t)t * heads * record_dim) + record_head;
            const float decay = step[dk + dv];
            value = fmaf(step[i], step[dk + j], value * decay);
        }
        st[state_head + (size_t)i * dv + j] = value;
    }
}

#endif  // COFFIN_DELTA_DEVICE_INC
