// @file hyper.cu
// @brief Qwen4-Exp 的 Gated Residual 融合内核。
//
// Module: coffin/backends/corex/kernels。由 CoreX 后端的分派层调用，
// ``models/qwen4exp`` 通过它跑 read gate / write gate，不依赖模型代码。
//
// 为什么值得单独写内核：参考实现里这一层是八次 torch 调用（分组 RMSNorm、down、
// silu、up、sigmoid、加权 mean、inject、2*sigmoid），而模型有 48 层、每层两套，
// 于是一次 decode 一步就是 96 × 8 = 768 次启动。本机一次启动的下限约 5.75 us，
// 光发射就是 4.4 ms —— 与整个 dense 部分的权重带宽约 3.47 ms耗时同一量级。
// 融合成两个内核后每套两次启动，发射降到约 1.1 ms。
//
// 两个而不是一个：read gate 的输出是 up 的输入，存在真实数据依赖。合成一个要么
// 重复算 down+silu，要么在块间做全局同步，两者都更贵。
//   * ``grouped_rms_norm_kernel``：分组归一化（一行的 4 路流各算一次）
//   * ``hyper_down_kernel``：down 投影 -> ``silu(low / hc_count)``
//   * ``hyper_up_kernel``：up 投影 -> sigmoid -> 逐流加权 mean，同时出写门
//
// 归一化单独一个内核而不是并进 ``hyper_down``：后者一个块只盖一行、每线程沿 K
// 扫一遍，而归一化是对 K 的归约，两者并行维不同。合并会让要么重算归约、要么
// 串行化整个 GEMV。
//
// Constraints:
//   * 分组归一化不是整条 10240 一起算。``hc_norm`` 的 ``group_size`` 等于
//     ``hidden_size``，所以 4 路残差流各自在 2560 维上做一次 RMSNorm。整条一起算
//     会得到一个看着合理的数，而那恰好抹平了 4 路流的尺度差异——保留它正是这套
//     结构存在的理由。
//   * 归一化、门、归约全在 fp32 里算（``corex_scalar.h``：本设备 fp64 不可用，
//     fp32 是唯一可靠的累加精度）；激活与权重的存储仍是 bf16。
//   * 加权平均是mean（除 ``hc``），不是 sum。写门是 ``2 * sigmoid(dot / hc)``，
//     那个 2 让零输入时权重正好是 1，除以 ``hc`` 必须在 sigmoid **之前**。
//   * ``block_sum`` 内部有 ``__syncthreads()``，**必须由块内所有线程调用**。放进
//     ``if (threadIdx.x == 0)`` 里会死锁
//   * 不用 CUDA graph、不用 cp.async，否则会把卡给挂掉
//   * 权重按 ``[K, N]`` 给（``Caps.weight_layout == "kn"``）。``[N, K]`` 会算出
//     看着合理的错数，所以主机侧只校验 dtype / 连续性 / 二维性，转置归 Python 侧。

#include "corex_bindings.h"
#include "corex_ops.h"
#include "corex_scalar.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <climits>

namespace {

// @brief 块内 fp32 求和，每个线程都拿到同一个结果。
//
// 先 warp 内 shuffle 收成 32 个值，再由 warp 0 收口。本机一个块最多 1024 线程，
// 两级足够，且不必让每个元素都过 shared memory。
//
// @param v       本线程的部分和。
// @param scratch 至少 32 个 float 的共享缓冲，块内所有迭代复用。
// @return 块内所有线程之和。**调用方必须在所有线程上调用**，内部有 __syncthreads。
__device__ __forceinline__ float block_sum(float v, float* scratch) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    const int warps = ((int)blockDim.x + 31) >> 5;
    v = ((int)threadIdx.x < warps) ? scratch[threadIdx.x] : 0.f;
    if (warp == 0) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            v += __shfl_down_sync(0xffffffffu, v, offset);
        }
    }
    return v;
}

//: split-K 的目标块数。
//:
//: 16 SM 的部件一个 wave 就是 16 个块，而 ``hyper_*`` 三个核在 ``M == 1`` 时的单趟
//: 网格分别只有 3（down 的 ``cols``）、20（up 的 ``ceil(hidden/块大小)``）个,
//: 整张卡上跑着的块数比一个 wave 还少。这就是它们慢的全部原因，而不是权重字节数。
//:
//: 为什么是 96：``gemv.cu`` 的 ``pick_split`` 用 64（4 个 wave）并实测出平台在
//: 32 个块之后基本变平；本文件的两个核每个块更窄（down 每块 128 线程、up 每块
//: 128 个输出下标），需要更多块才能把 warp 槽填到同样的比例，所以取 6 个 wave。
constexpr int HYPER_TARGET_BLOCKS = 96;

//: 单个 K 切片值得调度的最短长度。低于它启动与部分和的写入会超过切片本身读的量。
constexpr int HYPER_MIN_SLICE = 64;

//: ``hyper_up`` 的 ``lowrank`` 切片下限，比 :data:`HYPER_MIN_SLICE` 小一半。
//:
//: 两个核的归约轴长度差 32 倍：``hyper_down`` 的 K 是 10240，``hyper_up`` 的
//: ``lowrank`` 只有 320。所以这个核的切片数被 ``lowrank / min_slice`` **直接封顶**
//: （:data:`HYPER_UP_TARGET_BLOCKS` 调大也没用，实测 200~500 都只得到同一个值）：
//:
//:   min_slice   64      32      16
//:   splits       5      10      20
//:   块数       100     200     400
//:   hyper_up   193.3   114.0   114.9  us/次     <- 已扫，32 是最优点
//:   decode      1.048x  1.099x  1.078x
//:
//: **32 是扫出来的最优点，不要再往下切。** 16 那档（400 块）实测反而退到 114.9 us：
//: 每片只剩 16 个元素，部分和的写入（``wide`` 个 fp32）超过了那 16 次乘加读的量。
//: 向上 64 那档是 193.3 us。所以这是一个单峰，不是"越大越好"。
//:
//: 它同时也是 ``hyper_up`` 当前的上限：权重 6.25 MiB 在 114.0 us 里折 57.4 GB/s，
//: 是带宽屋顶的 10%，而 ``hyper_down`` 已经到 86.6 GB/s。继续压的力气应该花在
//: ``moe_grouped`` 那 28.3 ms 的分派胶水上，不是这里。
constexpr int HYPER_UP_MIN_SLICE = 32;

//: ``hyper_up`` 的目标块数。
//:
//: 为什么和 :data:`HYPER_TARGET_BLOCKS` 分开：``hyper_down`` 每块沿 K=10240 扫一整段，
//: 一块的算术量很重；而 ``hyper_up`` 每块只沿 ``lowrank=320`` 扫，单块轻得多。
//:
//: **但它现在受 :data:`HYPER_UP_MIN_SLICE` 夹住**，实际块数由后者决定（见那里的表）。
//: 这个常量只有在前者放宽时才会起作用，留着是为了把"想要多少块"与"最少切多细"
//: 这两件事分开表达。
constexpr int HYPER_UP_TARGET_BLOCKS = 384;

// @brief 决定 ``hyper_down`` / ``hyper_up`` 的 split-K 切分数。
//
// 切分乘倍了块数却不碰权重流量：每片读每列一个不同的 K（或 lowrank）区间，所以搬动
// 的字节数相同，变的只是同时在飞的独立加载数与总块数。
//
// **这是并行度规则，不是吞吐拟合。** 判据是「块数要够填满 16 SM × 64 warp」，所以
// 从 ``需要多少块`` 反推片数，而不是照抄 ``gemv.cu`` 里那个针对宽 GEMV 拟合出来的
// 数字（``GEMV_NARROW_TARGET_BLOCKS = 64`` 那个数是在 ``N >= 2560`` 的形状上量的，
// 与本文件 ``cols == 3`` 的网格形状毫无关系）。
//
// @param rows      激活行数。
// @param cols      单趟网格的列方向块数。
// @param k         归约长度（down 是 K = 10240，up 是 lowrank = 320）。
// @param min_slice 每片至少要有多少个归约元素。
// @param target    目标块数，两个核取不同的值，见 :data:`HYPER_UP_TARGET_BLOCKS`。
// @return ``[1, ...]`` 的切分数；``1`` 表示不切。
int pick_hyper_split(int rows, int cols, int k, int min_slice, int target) {
    if (rows <= 0 || cols <= 0 || k <= 0 || min_slice <= 0 || target <= 0) return 1;
    const int single = rows * cols;
    int splits = (target + single - 1) / single;
    // 每片至少 min_slice 个元素，否则部分和的写入会把收益吃掉。
    const int by_slice = k / min_slice;
    if (splits > by_slice) splits = by_slice;
    return splits < 2 ? 1 : splits;
}

// @brief 常驻的 split-K 部分和缓冲，按需增长。
//
// 不复用 ``torch::empty``：这个缓冲在一步 decode 里被要 96 次，而它的大小随 ``splits``
// 变（切分一加大就从 7.7 KB 涨到 40 KB 量级）。``torch`` 的 caching allocator 单次
// 很便宜，但 96 次累加在 host 侧仍是可测的成本——``gemv-dispatch`` 在 ``w4a8`` 上把
// split 从 12 加到 40 之后，device 省 31.7 ms 而端到端只省 28.7 ms，多出来的就是
// 每次调用的分配。这里用一块裸 ``cudaMalloc`` 的常驻缓冲一次解决：模型几何固定，
// 它只在第一次（或换设备时）分配一次。
//
// 失败回退到 ``torch::empty`` 而不是抛错：多一次分配总比整个算子不可用要好。
//
// **两个算子的形状不同，所以只能按"容量够不够"复用，不能按"形状相同"匹配**：
// ``hyper_down`` 要 ``[splits, rows, lowrank]``（真机上约 40 KiB），``hyper_up`` 要
// ``[splits, rows, hc * hidden]``（约 400 KiB）。两者在同一层里交替出现，若要求形状
// 严格相同就会每次都重分配——那种失败的症状是"加了缓存却更慢"，而不是报错。
// 按容量复用的后果是最多增长一次（先 down 要小块、再 up 要大会增长，之后全部命中）。
//
// ``g_hyper_scratch_allocs`` 把累计分配次数暴露出来，正是为了在真机上验证这句话：
// 跑完一步应当只涨 1-2；涨到 96 或 192 就是两个形状在互顶。
struct HyperScratch {
    float* ptr = nullptr;
    int64_t bytes = 0;
    int device = -1;
};

//: 累计的 ``cudaMalloc`` 次数，供诊断读。一次 ``++int`` 的成本远低于加一个宏。
int g_hyper_scratch_allocs = 0;

HyperScratch& hyper_scratch_slot() {
    // 函数内静态：避免命名空间级对象在扩展卸载时的析构顺序问题。
    static HyperScratch slot;
    return slot;
}

// @brief 取一块至少 ``need_bytes`` 的常驻部分和缓冲。
//
// @param need_bytes 需要的字节数。
// @param device     当前设备序号；换设备会重新分配。
// @return 设备指针；``cudaMalloc`` 失败时为 ``nullptr``。
float* hyper_scratch(int64_t need_bytes, int device) {
    HyperScratch& slot = hyper_scratch_slot();
    if (slot.ptr != nullptr && slot.bytes >= need_bytes && slot.device == device) {
        return slot.ptr;
    }
    if (slot.ptr != nullptr) {
        cudaFree(slot.ptr);
        slot.ptr = nullptr;
        slot.bytes = 0;
        slot.device = -1;
    }
    void* fresh = nullptr;
    if (cudaMalloc(&fresh, need_bytes) != cudaSuccess) {
        cudaGetLastError();          // 清掉错误位，让调用方走 torch 回退
        return nullptr;
    }
    slot.ptr = static_cast<float*>(fresh);
    slot.bytes = need_bytes;
    slot.device = device;
    ++g_hyper_scratch_allocs;
    return slot.ptr;
}

//: ``hyper_up`` 每个块的线程数。128 而不是 256：这个核的并行维是 ``hidden``
//: 下标（每块 ``tile`` 个），256 线程时 ``tile`` 必须跟着变 256，块数就掉到 10——
//: 比一个 wave 还少。128 给到 20 个块，且线程完全用满（没有空转的下标槽）。
constexpr int HYPER_UP_THREADS = 128;

// @brief 分组 RMSNorm，一个块处理一整行（所有 ``hc`` 组）。
//
// 每个块盖住一行，块内依次处理各流：先对 2560 维做平方和归约，再用同一个归约值
// 缩放该流所有元素。各流之间互不干扰，所以顺序执行与并行等价，而顺序执行让共享
// 内存只需要一个归约缓冲。
//
// @param x      残差流，``[rows, hc * hidden]``。
// @param w      零中心权重，``[hc * hidden]``，fp32，前向按 ``1 + w``。
// @param out    归一化结果，``[rows, hc * hidden]``，dtype 同 ``x``。
// @param rows   行数。
// @param hidden 每流宽度。
// @param hc     流数。
// @param eps    方差下限。
template <typename XT>
__global__ void grouped_rms_norm_kernel(const XT* x, const float* w, XT* out,
                                        int rows, int hidden, int hc, float eps) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int wide = hidden * hc;
    __shared__ float scale[8];
    __shared__ float warps[32];

    const XT* xrow = x + (size_t)row * wide;
    XT* orow = out + (size_t)row * wide;
    for (int g = 0; g < hc; ++g) {
        float acc = 0.f;
        for (int i = (int)threadIdx.x; i < hidden; i += (int)blockDim.x) {
            const float v = CxScalar<XT>::to_float(xrow[g * hidden + i]);
            acc += v * v;
        }
        const float total = block_sum(acc, warps);
        if (threadIdx.x == 0) scale[g] = rsqrtf(total / (float)hidden + eps);
        // 归约里那个 __syncthreads 之后 scale[g] 才可见，而下一轮会复用 warps，
        // 所以这里还要一次屏障，否则下一轮的 block_sum 会在上一轮读到一半时写缓冲。
        __syncthreads();
        const float inv = scale[g];
        for (int i = (int)threadIdx.x; i < hidden; i += (int)blockDim.x) {
            const float v = CxScalar<XT>::to_float(xrow[g * hidden + i]);
            orow[g * hidden + i] = CxScalar<XT>::from_float(
                v * inv * (1.f + w[g * hidden + i]));
        }
        __syncthreads();
    }
}

// @brief down 投影（``[K, N]`` 权重，``N == lowrank``）+ ``silu(v / hc)``。
//
// 一个输出列一个线程：``lowrank`` 通常 320，一个块盖得住。每个线程沿 K 扫一遍，
// 读 ``wt[i * N + col]`` 这一条 strided 列——decode 期 GEMV 的标准访存模式。
//
// **K 轴用 4 条独立累加链。** 单链串行 10240 次 FMA 是一条完整的依赖链，每次都要
// 等上一次的加法结果，于是访存延迟完全暴露；4 条链让硬件能同时挂 4 个未完成的加，
// 实测在 ``gemv.cu`` 的 ``dot_kn`` 上是 1.53x。这不是微调——``hyper_down`` 在
// ``rows == 1`` 时每线程要串 1707 次迭代，单链根本盖不住访存延迟。
//
// @param normed 归一化后的激活，``[rows, K]``。
// @param wt     down 权重，``[K, N]`` 连续。
// @param out    ``[rows, N]``，已做 ``silu(x / hc)``。
// @param rows   行数。
// @param k      K。
// @param n      N。
// @param hc     读门的除数。
// @param splits K 轴的切分数，见 :func:`pick_hyper_split`。
// @param part   ``[splits, rows, n]`` fp32 部分和，``splits == 1`` 时不用。
template <typename XT>
__global__ void hyper_down_kernel(const XT* normed, const XT* wt, XT* out,
                                  int rows, int k, int n, int hc,
                                  int splits, float* part) {
    const int row = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || col >= n) return;
    const XT* rowp = normed + (size_t)row * k;
    // K 轴按 split 均分，各段独立累加。``M == 1`` 时并行度只有 N/块大小，
    // 而 N 常常只有 320 —— 一张 16 SM 的卡上只点亮 3 个 SM，其余空转。
    // 切 K 是唯一不改变数学、又能填满机器的办法。
    const int per = (k + splits - 1) / splits;
    const int begin = blockIdx.z * per;
    const int end = min(k, begin + per);
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    int i = begin;
    // 4 条链按 4 步交错：第 s 条吃 ``begin + s, begin + s + 4, ...`` 的元素。
    for (; i + 3 < end; i += 4) {
        acc0 += CxScalar<XT>::to_float(rowp[i + 0])
              * CxScalar<XT>::to_float(wt[(size_t)(i + 0) * n + col]);
        acc1 += CxScalar<XT>::to_float(rowp[i + 1])
              * CxScalar<XT>::to_float(wt[(size_t)(i + 1) * n + col]);
        acc2 += CxScalar<XT>::to_float(rowp[i + 2])
              * CxScalar<XT>::to_float(wt[(size_t)(i + 2) * n + col]);
        acc3 += CxScalar<XT>::to_float(rowp[i + 3])
              * CxScalar<XT>::to_float(wt[(size_t)(i + 3) * n + col]);
    }
    for (; i < end; ++i) {
        acc0 += CxScalar<XT>::to_float(rowp[i])
              * CxScalar<XT>::to_float(wt[(size_t)i * n + col]);
    }
    const float acc = (acc0 + acc1) + (acc2 + acc3);
    if (splits == 1) {
        const float scaled = acc / (float)hc;
        out[(size_t)row * n + col] = CxScalar<XT>::from_float(
            scaled / (1.f + __expf(-scaled)));
    } else {
        part[((size_t)blockIdx.z * rows + row) * n + col] = acc;
    }
}

// @brief 把 split-K 的部分和收成 down 的最终输出。
//
// 单独一趟而不是在部分和里用原子加：原子加在 fp32 上不可结合，归约顺序随调度变化，
// 两次运行会给出不同结果——这个内核要能当正确性基线用，就必须是确定的。
//
// @param part  ``[splits, rows, n]`` fp32 部分和。
// @param out   ``[rows, n]``，已做 ``silu(v / hc)``。
// @param splits 切分数。
template <typename XT>
__global__ void hyper_down_finish_kernel(const float* part, XT* out,
                                         int rows, int n, int hc, int splits) {
    const int row = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || col >= n) return;
    float acc = 0.f;
    for (int s = 0; s < splits; ++s) {
        acc += part[((size_t)s * rows + row) * n + col];
    }
    const float scaled = acc / (float)hc;
    out[(size_t)row * n + col] = CxScalar<XT>::from_float(
        scaled / (1.f + __expf(-scaled)));
}

// @brief 沿 ``lowrank`` 的一段算一个输出通道的点积，4 条独立累加链。
//
// 单链要串 ``t_end - t_begin`` 次 FMA，每次都得等上一次的加法结果，访存延迟因此
// 完全暴露；4 条链让硬件同时挂 4 个未完成的加。``gemv.cu`` 的 ``dot_kn`` 在同样
// 改造上实测 1.53x，这里每个输出点要付 ``lowrank`` 次，是同一个形状。
//
// @param lowp    ``low`` 的这一行，``[lowrank]``。
// @param up_wt   up 权重，``[lowrank, wide]``。
// @param wide    ``up_wt`` 的行距。
// @param col     输出通道下标。
// @param t_begin 本片的起点（含）。
// @param t_end   本片的终点（不含）。
// @return 这一段上的点积，fp32。
template <typename XT>
__device__ __forceinline__ float dot_slice(const XT* lowp, const XT* up_wt,
                                           int wide, int col,
                                           int t_begin, int t_end) {
    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
    int t = t_begin;
    for (; t + 3 < t_end; t += 4) {
        d0 += CxScalar<XT>::to_float(lowp[t + 0])
            * CxScalar<XT>::to_float(up_wt[(size_t)(t + 0) * wide + col]);
        d1 += CxScalar<XT>::to_float(lowp[t + 1])
            * CxScalar<XT>::to_float(up_wt[(size_t)(t + 1) * wide + col]);
        d2 += CxScalar<XT>::to_float(lowp[t + 2])
            * CxScalar<XT>::to_float(up_wt[(size_t)(t + 2) * wide + col]);
        d3 += CxScalar<XT>::to_float(lowp[t + 3])
            * CxScalar<XT>::to_float(up_wt[(size_t)(t + 3) * wide + col]);
    }
    for (; t < t_end; ++t) {
        d0 += CxScalar<XT>::to_float(lowp[t])
            * CxScalar<XT>::to_float(up_wt[(size_t)t * wide + col]);
    }
    return (d0 + d1) + (d2 + d3);
}

// @brief 把 ``hyper_up`` 的 split-K 部分点积收成读门输出。
//
// 必须单独一趟：``sigmoid`` 是非线性的，部分点积加齐之前不能进门，所以切片模式下
// 只能先存部分和。与 :cuda:`hyper_down_finish_kernel` 同一个理由，也与它一样用
// 确定的顺序求和（不用原子加——fp32 的原子加不可结合，两次运行会给出不同结果）。
//
// @param part   ``[splits, rows, hc * hidden]`` fp32 部分点积。
// @param normed 归一化后的激活，``[rows, wide]``。
// @param mixed  ``[rows, hidden]`` 连续目标。
// @param rows   行数。
// @param hidden 每流宽度。
// @param hc     流数。
// @param splits 切分数。
template <typename XT>
__global__ void hyper_up_finish_kernel(const float* part, const XT* normed, XT* mixed,
                                       int rows, int hidden, int hc, int splits) {
    const int row = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || i >= hidden) return;
    const int wide = hidden * hc;
    const XT* normp = normed + (size_t)row * wide;
    float acc = 0.f;
    for (int g = 0; g < hc; ++g) {
        const int col = g * hidden + i;
        float dot = 0.f;
        for (int s = 0; s < splits; ++s) {
            dot += part[((size_t)s * rows + row) * wide + col];
        }
        acc += (1.f / (1.f + __expf(-dot))) * CxScalar<XT>::to_float(normp[col]);
    }
    mixed[(size_t)row * hidden + i] = CxScalar<XT>::from_float(acc / (float)hc);
}

// @brief up 投影 + sigmoid + 逐流加权 mean，同时算写门。
//
// 一个线程负责一个**输出**下标 ``i``（``hidden`` 宽），为它遍历所有流：把
// ``gate = sigmoid(dot(low, up_wt[:, g*hidden + i]))`` 乘上该流的归一化值累加，
// 最后除以 ``hc``。这样不需要任何跨线程归约——mean 是对 4 路流取的，不是对
// 10240 个元素取的。
//
// 写门只有 ``hc`` 个输出，交给前 ``hc`` 个线程各扫一遍 ``wide`` 维；它和主循环
// 读同一份 ``normed``，放在同一个内核里省掉一次 ``[rows, 10240]`` 的读。
//
// @param normed 归一化后的激活，``[rows, wide]``。
// @param up_wt  up 权重，``[K = lowrank, N = wide]`` 连续。
// @param low    down 的输出，``[rows, lowrank]``。
// @param inj_wt 写门权重，``[K = wide, N = hc]`` 连续。
// @param mixed  读门塌缩结果，``[rows, hidden]``。
// @param inject 写门，``[rows, hc]``，fp32。
// @param rows   行数。
// @param hidden 每流宽度。
// @param hc     流数。
// @param lowrank down 的秩。
// @param tile   每个块负责的 ``hidden`` 下标数；``0`` 表示整行一个块。
//
// 为什么要 ``tile``：``hidden`` 是 2560，一块 256 线程时每线程要串行算 10 个输出、
// 每个输出 4 路流 × 320 次乘加。``M == 1`` 的 decode 一步只有**一块**在跑，也就是
// 一张 16 SM 的卡上点亮 1 个 SM。实测这样比纯 torch 慢 12 倍。按 ``hidden`` 切块
// 把并行度提到 ``ceil(hidden / tile)``，代价只是写门那一路要在每块重算。
template <typename XT>
__global__ void hyper_up_kernel(const XT* normed, const XT* up_wt, const XT* low,
                                const XT* inj_wt, XT* mixed, float* inject,
                                int rows, int hidden, int hc, int lowrank, int tile,
                                int splits, float* part) {
    // 行号取 ``blockIdx.y``：见 :func:`hyper_up` 里的网格说明。写成 ``blockIdx.x``
    // 而网格是 (行, 块) 二维时，一行内的所有块会重复算同一个下标集合。
    const int row = blockIdx.y;
    if (row >= rows) return;
    const int wide = hidden * hc;
    const int hi = (tile > 0) ? min(hidden, (int)blockIdx.x * tile + tile) : hidden;
    const int lo = (tile > 0) ? (int)blockIdx.x * tile : 0;
    const XT* normp = normed + (size_t)row * wide;
    const XT* lowp = low + (size_t)row * lowrank;
    __shared__ float gate_scratch[32];

    // ``lowrank`` 轴按 ``blockIdx.z`` 切片。**这是本核唯一还能加的并行度**：
    // 输出只有 ``hidden``（2560）个，一个线程一个输出就已经占满这一维；把 ``tile``
    // 再切小只会制造空转线程（每块下标数少于块内线程数），总线程数不变。而
    // ``hidden × hc`` 个点各要串 ``lowrank`` 次，切这一段才真的乘倍了块数。
    //
    // 切了之后不能就地 sigmoid（门是对整条点积取的），所以这一趟在切片模式下只写
    // 部分点积，由 :cuda:`hyper_up_finish_kernel` 加齐后再过 sigmoid 与逐流 mean。
    const int tper = (lowrank + splits - 1) / splits;
    const int t_begin = blockIdx.z * tper;
    const int t_end = min(lowrank, t_begin + tper);
    const bool sliced = splits > 1;
    float* mypart = sliced
        ? part + ((size_t)blockIdx.z * rows + row) * (size_t)(hc * hidden)
        : nullptr;

    if (sliced) {
        // 切片模式：只算本片的点积部分和，``[rows, hc * hidden]`` 的 fp32。
        for (int i = lo + (int)threadIdx.x; i < hi; i += (int)blockDim.x) {
            for (int g = 0; g < hc; ++g) {
                const int col = g * hidden + i;
                mypart[col] = dot_slice<XT>(lowp, up_wt, wide, col, t_begin, t_end);
            }
        }
    } else {
        // 单趟模式：一遍算完 sigmoid 与逐流 mean。``acc`` 对 ``hc`` 路取和，
        // 最后除 ``hc``——mean 是对流取的，不是对 10240 个元素取的。
        for (int i = lo + (int)threadIdx.x; i < hi; i += (int)blockDim.x) {
            float acc = 0.f;
            for (int g = 0; g < hc; ++g) {
                const int col = g * hidden + i;
                const float dot = dot_slice<XT>(lowp, up_wt, wide, col, t_begin, t_end);
                acc += (1.f / (1.f + __expf(-dot)))
                     * CxScalar<XT>::to_float(normp[col]);
            }
            mixed[(size_t)row * hidden + i] = CxScalar<XT>::from_float(acc / (float)hc);
        }
    }

    // 写门只有 ``hc`` 个输出，但每个都要扫 ``wide`` 维。
    //
    // **整个块按 ``t`` 分摊，不是每个 warp 认领一个输出。** 旧写法让 warp ``w``
    // 负责输出 ``w``、32 条 lane 步进 32，于是每条 lane 要串 ``wide / 32 = 320`` 次，
    // 而它读 ``inj_wt[t * hc + warp]``：``hc = 4`` 时 32 条 lane 全落在同一个 ``t``
    // 上、只差 ``warp`` 一个下标，**一次 8 字节的事务里只有 2 字节有用**。这个形状
    // （N = 4、K = 10240）单独调用实测 134.9 us，而本核每层两套各自付一遍。
    //
    // 改成每条线程沿 ``t`` 步进 ``blockDim.x``、并一次读出连续 ``hc`` 个权重：
    // 串行长度降到 ``wide / blockDim.x``，且一个 warp 的 32 条 lane 覆盖
    // ``32 * blockDim.x`` 个 ``t``，读的是连续元素——完全合并。
    //
    // 只让 ``blockIdx.x == 0 && blockIdx.z == 0`` 做：``tile`` 把同一行切成多个块后
    // 每块都会走到这里，而写门的结果与 ``hidden`` 下标无关，重复写只是浪费；更要紧的
    // 是它会把 10240 维的 ``normed`` 在每个块上重读一遍。切片模式还要加上
    // ``blockIdx.z`` 那一维——否则每一片都会把写门重算一次，那是纯浪费。
    if (blockIdx.x == 0 && blockIdx.z == 0) {
        float gacc[8];
#pragma unroll
        for (int g = 0; g < 8; ++g) gacc[g] = 0.f;
        for (int t = (int)threadIdx.x; t < wide; t += (int)blockDim.x) {
            const float nv = CxScalar<XT>::to_float(normp[t]);
            const XT* ip = inj_wt + (size_t)t * hc;
#pragma unroll
            for (int g = 0; g < 8; ++g) {
                if (g < hc) gacc[g] += nv * CxScalar<XT>::to_float(ip[g]);
            }
        }
        // ``block_sum`` 内部有 ``__syncthreads``，所以它必须由块内所有线程调用。
        // ``g < hc`` 对块内所有线程是同一个值（``hc`` 是内核参数），条件均匀。
#pragma unroll
        for (int g = 0; g < 8; ++g) {
            if (g < hc) {
                const float total = block_sum(gacc[g], gate_scratch);
                // 下一轮 ``block_sum`` 会复用 ``gate_scratch``，所以这里要再拦一次，
                // 否则部分线程已经在写而另一些还在读上一轮的值。
                __syncthreads();
                if ((int)threadIdx.x == 0) {
                    inject[(size_t)row * hc + g] =
                        2.f / (1.f + __expf(-total / (float)hc));
                }
            }
        }
    }
}

}  // namespace

// @brief 对残差流做分组 RMSNorm：每 ``hidden`` 个元素一组，各自归一化。
//
// @param x   残差流，``[rows, hidden * hc]`` 连续，bf16 或 fp32。
// @param w   零中心权重，``[hidden * hc]`` fp32，前向按 ``1 + w``。
// @param out 目标缓冲，``[rows, hidden * hc]`` 连续，dtype 同 ``x``。
// @param hidden 每流宽度。
// @param hc     流数，上限 8（与内核里的共享缓冲一致）。
// @param eps    方差下限。
// @return out。
// @throws c10::Error 布局、dtype、形状或组数不符合契约。
torch::Tensor hyper_norm(torch::Tensor x, torch::Tensor w, torch::Tensor out,
                         int64_t hidden, int64_t hc, double eps) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous(),
                "residual stream must be a contiguous CUDA [rows, hidden*hc]");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 || x.scalar_type() == torch::kFloat,
                "residual stream must be bf16 or fp32");
    TORCH_CHECK(hidden > 0 && hc > 0 && hc <= 8, "need 1..8 groups, got ", hc);
    TORCH_CHECK(x.size(1) == hidden * hc, "width ", x.size(1),
                " does not match hidden*hc = ", hidden * hc);
    TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.scalar_type() == torch::kFloat
                && w.numel() == hidden * hc, "norm weight must be contiguous fp32");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.sizes() == x.sizes()
                && out.scalar_type() == x.scalar_type(), "destination mismatch");

    const c10::cuda::CUDAGuard guard(x.device());
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const int rows = (int)x.size(0);
    if (rows == 0) return out;
    const int threads = 256;
    if (x.scalar_type() == torch::kBFloat16) {
        grouped_rms_norm_kernel<uint16_t><<<rows, threads, 0, stream>>>(
            reinterpret_cast<const uint16_t*>(x.data_ptr()), w.data_ptr<float>(),
            reinterpret_cast<uint16_t*>(out.data_ptr()), rows, (int)hidden, (int)hc,
            (float)eps);
    } else {
        grouped_rms_norm_kernel<float><<<rows, threads, 0, stream>>>(
            x.data_ptr<float>(), w.data_ptr<float>(), out.data_ptr<float>(),
            rows, (int)hidden, (int)hc, (float)eps);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// @brief down 投影加 ``silu(v / hc)``：``[rows, K] -> [rows, lowrank]``。
//
// @param normed 归一化后的激活，``[rows, K]`` 连续。
// @param wt     down 权重，``[K, lowrank]`` 连续，dtype 同 ``normed``。
// @param out    ``[rows, lowrank]`` 连续目标。
// @param hc     读门的除数。
// @return out。
// @throws c10::Error 布局、dtype 或形状不符合契约。
torch::Tensor hyper_down(torch::Tensor normed, torch::Tensor wt, torch::Tensor out,
                         int64_t hc) {
    TORCH_CHECK(normed.is_cuda() && normed.dim() == 2 && normed.is_contiguous(),
                "activation must be contiguous CUDA [rows, K]");
    TORCH_CHECK(wt.is_cuda() && wt.dim() == 2 && wt.is_contiguous()
                && wt.scalar_type() == normed.scalar_type(),
                "weight must be contiguous [K, lowrank] with the activation dtype");
    const int rows = (int)normed.size(0), k = (int)normed.size(1);
    const int n = (int)wt.size(1);
    TORCH_CHECK(wt.size(0) == k, "weight K ", wt.size(0), " != activation K ", k);
    TORCH_CHECK(hc > 0, "hc must be positive");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.scalar_type() == normed.scalar_type()
                && out.size(0) == rows && out.size(1) == n, "destination mismatch");

    const c10::cuda::CUDAGuard guard(normed.device());
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    if (rows == 0) return out;
    const int threads = 128;
    const int cols = (n + threads - 1) / threads;
    // 并行度 = cols × rows × splits。``n == lowrank == 320`` 时 cols 是 3，一片
    // 16 SM × 64 warp 的卡要几十个块才喂饱，所以按 :func:`pick_hyper_split` 从
    // 「需要多少块」反推片数。旧规则写死 16，在 ``rows == 1`` 时只得到 6 片 =
    // 18 个块 = 2304 条线程 = 4.5 warps/SM，实测 579 us（带宽地板的 50 倍）。
    const int splits = pick_hyper_split(rows, cols, k, HYPER_MIN_SLICE,
                                      HYPER_TARGET_BLOCKS);
    const bool need_part = splits > 1;
    // 部分和缓冲常驻，不走 ``torch::empty``：它一步要被要 96 次，而大小随 splits 变。
    // 分配失败（或没有驱动上下文）时回退到 torch 分配，宁可慢也不要算子不可用。
    float* part_ptr = nullptr;
    torch::Tensor part;
    if (need_part) {
        const int64_t need = (int64_t)splits * rows * n * (int64_t)sizeof(float);
        part_ptr = hyper_scratch(need, (int)normed.get_device());
        if (part_ptr == nullptr) {
            part = torch::empty({splits, rows, n},
                                torch::TensorOptions().dtype(torch::kFloat)
                                    .device(normed.device()));
            part_ptr = part.data_ptr<float>();
        }
    }
    const dim3 grid(cols, rows, splits);
    if (normed.scalar_type() == torch::kBFloat16) {
        hyper_down_kernel<uint16_t><<<grid, threads, 0, stream>>>(
            reinterpret_cast<const uint16_t*>(normed.data_ptr()),
            reinterpret_cast<const uint16_t*>(wt.data_ptr()),
            reinterpret_cast<uint16_t*>(out.data_ptr()), rows, k, n, (int)hc,
            splits, part_ptr);
        if (need_part) {
            hyper_down_finish_kernel<uint16_t><<<dim3(cols, rows), threads, 0, stream>>>(
                part_ptr, reinterpret_cast<uint16_t*>(out.data_ptr()),
                rows, n, (int)hc, splits);
        }
    } else {
        hyper_down_kernel<float><<<grid, threads, 0, stream>>>(
            normed.data_ptr<float>(), wt.data_ptr<float>(), out.data_ptr<float>(),
            rows, k, n, (int)hc, splits, part_ptr);
        if (need_part) {
            hyper_down_finish_kernel<float><<<dim3(cols, rows), threads, 0, stream>>>(
                part_ptr, out.data_ptr<float>(), rows, n, (int)hc, splits);
        }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// @brief up 投影加读门塌缩，同时算写门。
//
// @param normed 归一化后的激活，``[rows, hidden*hc]`` 连续。
// @param up_wt  up 权重，``[lowrank, hidden*hc]`` 连续。
// @param low    down 的输出，``[rows, lowrank]`` 连续。
// @param inj_wt 写门权重，``[hidden*hc, hc]`` 连续，dtype 同 ``normed``。
// @param mixed  ``[rows, hidden]`` 连续目标：读门塌缩结果。
// @param inject ``[rows, hc]`` 连续 fp32 目标：写门。
// @param hidden 每流宽度。
// @param hc     流数。
// @param lowrank down 的秩。
// @return mixed。
// @throws c10::Error 布局、dtype 或形状不符合契约。
torch::Tensor hyper_up(torch::Tensor normed, torch::Tensor up_wt, torch::Tensor low,
                       torch::Tensor inj_wt, torch::Tensor mixed, torch::Tensor inject,
                       int64_t hidden, int64_t hc, int64_t lowrank) {
    TORCH_CHECK(normed.is_cuda() && normed.dim() == 2 && normed.is_contiguous(),
                "activation must be contiguous CUDA [rows, hidden*hc]");
    TORCH_CHECK(normed.scalar_type() == torch::kBFloat16
                || normed.scalar_type() == torch::kFloat,
                "activation must be bf16 or fp32");
    const int rows = (int)normed.size(0);
    const int wide = (int)(hidden * hc);
    TORCH_CHECK(normed.size(1) == wide, "width ", normed.size(1), " != hidden*hc ", wide);
    TORCH_CHECK(hc > 0 && hidden > 0 && lowrank > 0, "sizes must be positive");
    TORCH_CHECK(up_wt.is_cuda() && up_wt.dim() == 2 && up_wt.is_contiguous()
                && up_wt.scalar_type() == normed.scalar_type()
                && up_wt.size(0) == lowrank && up_wt.size(1) == wide,
                "up weight must be contiguous [lowrank, hidden*hc]");
    TORCH_CHECK(low.is_cuda() && low.dim() == 2 && low.is_contiguous()
                && low.scalar_type() == normed.scalar_type()
                && low.size(0) == rows && low.size(1) == lowrank,
                "low must be contiguous [rows, lowrank]");
    TORCH_CHECK(inj_wt.is_cuda() && inj_wt.dim() == 2 && inj_wt.is_contiguous()
                && inj_wt.scalar_type() == normed.scalar_type()
                && inj_wt.size(0) == wide && inj_wt.size(1) == hc,
                "inject weight must be contiguous [hidden*hc, hc]");
    TORCH_CHECK(mixed.is_cuda() && mixed.is_contiguous()
                && mixed.scalar_type() == normed.scalar_type()
                && mixed.size(0) == rows && mixed.size(1) == hidden,
                "mixed destination mismatch");
    TORCH_CHECK(inject.is_cuda() && inject.is_contiguous()
                && inject.scalar_type() == torch::kFloat
                && inject.size(0) == rows && inject.size(1) == hc,
                "inject destination must be contiguous fp32 [rows, hc]");

    const c10::cuda::CUDAGuard guard(normed.device());
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    if (rows == 0) return mixed;
    // 写门用整块线程按 ``t`` 分摊 + 块内归约，所以 ``hc <= 8`` 只是 ``gacc[8]``
    // 那个累加器数组的界，不再要求"块内至少 hc 个 warp"——任何一个块都能算完它。
    TORCH_CHECK((int)hc <= 8, "hc must not exceed the 8 accumulators, got ", hc);
    const int threads = HYPER_UP_THREADS;
    // 块数 = ``ceil(hidden / tile)``，且每块活跃线程 = ``min(tile, threads)``。
    //
    // 这两个数**必须一起看**，否则会白扔并行度。这个核的主循环是
    // ``for (i = lo + threadIdx.x; i < hi; i += blockDim.x)``：
    //
    // * ``tile`` 大于 ``threads`` → 线程全忙，但每线程要串多个输出下标（当前的取舍）；
    // * ``tile`` 小于 ``threads`` → **每块只有 ``tile`` 条线程干活，其余空转**。
    //   按「96 个目标块」反推曾得到 ``tile = ceil(2560/96) = 27`` 配 128 线程，
    //   于是 101/128 条线程空转，实测本核 193.3 us、33.9 GB/s。
    //
    // 现在是 ``tile == threads == 128`` → 20 块、全忙，实测 134.9 us、48.6 GB/s。
    // 想再加块数只能**同时**减 ``tile`` 与 ``threads``（例如 64/64 → 40 块），
    // 但那会把每块压到 2 个 warp，反而更难盖住访存延迟——所以这一维到此为止，
    // 剩余并行度靠 :func:`pick_hyper_split` 沿 ``lowrank`` 补。
    const int tile = (int)threads;
    const int tiles = (int)((hidden + tile - 1) / tile);
    const int splits = pick_hyper_split(rows, tiles, (int)lowrank, HYPER_UP_MIN_SLICE,
                                        HYPER_UP_TARGET_BLOCKS);
    const bool need_part = splits > 1;
    float* part_ptr = nullptr;
    torch::Tensor part;
    if (need_part) {
        const int64_t need = (int64_t)splits * rows * (int64_t)(hc * hidden)
                           * (int64_t)sizeof(float);
        part_ptr = hyper_scratch(need, (int)normed.get_device());
        if (part_ptr == nullptr) {
            part = torch::empty({splits, rows, hc * hidden},
                                torch::TensorOptions().dtype(torch::kFloat)
                                    .device(normed.device()));
            part_ptr = part.data_ptr<float>();
        }
    }
    const dim3 grid((unsigned)tiles, (unsigned)rows, (unsigned)splits);
    if (normed.scalar_type() == torch::kBFloat16) {
        hyper_up_kernel<uint16_t><<<grid, threads, 0, stream>>>(
            reinterpret_cast<const uint16_t*>(normed.data_ptr()),
            reinterpret_cast<const uint16_t*>(up_wt.data_ptr()),
            reinterpret_cast<const uint16_t*>(low.data_ptr()),
            reinterpret_cast<const uint16_t*>(inj_wt.data_ptr()),
            reinterpret_cast<uint16_t*>(mixed.data_ptr()), inject.data_ptr<float>(),
            rows, (int)hidden, (int)hc, (int)lowrank, tile, splits, part_ptr);
        if (need_part) {
            const int fthreads = 128;
            const int fcols = (int)((hidden + fthreads - 1) / fthreads);
            hyper_up_finish_kernel<uint16_t><<<dim3(fcols, rows), fthreads, 0, stream>>>(
                part_ptr, reinterpret_cast<const uint16_t*>(normed.data_ptr()),
                reinterpret_cast<uint16_t*>(mixed.data_ptr()),
                rows, (int)hidden, (int)hc, splits);
        }
    } else {
        hyper_up_kernel<float><<<grid, threads, 0, stream>>>(
            normed.data_ptr<float>(), up_wt.data_ptr<float>(), low.data_ptr<float>(),
            inj_wt.data_ptr<float>(), mixed.data_ptr<float>(), inject.data_ptr<float>(),
            rows, (int)hidden, (int)hc, (int)lowrank, tile, splits, part_ptr);
        if (need_part) {
            const int fthreads = 128;
            const int fcols = (int)((hidden + fthreads - 1) / fthreads);
            hyper_up_finish_kernel<float><<<dim3(fcols, rows), fthreads, 0, stream>>>(
                part_ptr, normed.data_ptr<float>(), mixed.data_ptr<float>(),
                rows, (int)hidden, (int)hc, splits);
        }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return mixed;
}

// 绑定写在内核旁边：名字、参数名与默认值只有这一份，`bindings.cpp` 只遍历注册表。
COFFIN_BIND(hyper_norm,
            "分组 RMSNorm：hc 路残差流各自在 hidden 维上归一化",
            pybind11::arg("x"), pybind11::arg("w"), pybind11::arg("out"), pybind11::arg("hidden"),
            pybind11::arg("hc"), pybind11::arg("eps"));
COFFIN_BIND(hyper_down,
            "Gated Residual 的 down 投影加 silu 门",
            pybind11::arg("normed"), pybind11::arg("wt"), pybind11::arg("out"), pybind11::arg("hc"));
COFFIN_BIND(hyper_up,
            "Gated Residual 的 up 投影加读门塌缩与写门",
            pybind11::arg("normed"), pybind11::arg("up_wt"), pybind11::arg("low"),
            pybind11::arg("inj_wt"), pybind11::arg("mixed"), pybind11::arg("inject"),
            pybind11::arg("hidden"), pybind11::arg("hc"), pybind11::arg("lowrank"));
