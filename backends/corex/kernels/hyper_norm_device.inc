// @file hyper_norm_device.inc
// @brief 分组 RMSNorm 的设备代码（超连接 / Gated Residual 用的那一支）。
//
// Module: coffin/backends/corex/kernels。被两条前端各包含一次：``hyper.cu``（torch
// 扩展）与 ``abi/hyper_norm_abi.cu``（``extern "C"`` 出口）。同一份设备代码。
//
// 为什么它留在 corex 而不是随 ``hyper_down`` / ``hyper_up`` 一起删：那两个吃 ``[K, N]``
// （kn）权重的投影随 kn 布局整条删除，归一化不吃投影权重，所以还在 decode 的残差
// 路径上（``GatedResidual``）。搬过来是为了不在这一步留一个 torch 依赖。
//
// Constraints:
//   * 一个块处理一整行（所有 ``hc`` 组），各流顺序处理：共享缓冲只需要一个归约槽，
//     而顺序与并行等价（各流互不干扰）。
//   * 每次 ``block_sum`` 之后必须再来一次 ``__syncthreads`` 才能复用 ``warps``：
//     否则下一轮会在上一轮读到一半时写缓冲。
//   * ``hc`` 上限 8 = ``scale[8]`` 的长度；组数超了会静默写到 LDS 之外。

#ifndef COFFIN_HYPER_NORM_DEVICE_INC
#define COFFIN_HYPER_NORM_DEVICE_INC

#include "corex_scalar.h"

#include <cstdint>

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

#endif  // COFFIN_HYPER_NORM_DEVICE_INC
