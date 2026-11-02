// @file hyper.cu
// @brief Qwen4-Exp 的 Gated Residual 融合内核。
//
// Module: coffin/backends/corex/kernels。由 CoreX 后端的分派层调用，
// ``models/qwen4exp`` 通过它跑 read gate / write gate，不依赖模型代码。
//
// 为什么值得单独写内核：参考实现里这一层是八次 torch 调用（分组 RMSNorm、down、
// silu、up、sigmoid、加权 mean、inject、2*sigmoid），而模型有 48 层、每层两套，
// 于是一次 decode 一步就是 96 × 8 = 768 次启动。本机一次启动的下限约 5.75 us，
// 光发射就是 4.4 ms —— 与整个 dense 部分的权重带宽耗时（约 3.47 ms）同一量级。
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
// @param normed 归一化后的激活，``[rows, K]``。
// @param wt     down 权重，``[K, N]`` 连续。
// @param out    ``[rows, N]``，已做 ``silu(x / hc)``。
// @param rows   行数。
// @param k      K。
// @param n      N。
// @param hc     读门的除数。
// @param splits K 轴的切分数，见 :func:`hyper_down` 的说明。
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
    float acc = 0.f;
    for (int i = begin; i < end; ++i) {
        acc += CxScalar<XT>::to_float(rowp[i])
             * CxScalar<XT>::to_float(wt[(size_t)i * n + col]);
    }
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
                                int rows, int hidden, int hc, int lowrank, int tile) {
    // 行号取 ``blockIdx.y``：见 :func:`hyper_up` 里的网格说明。写成 ``blockIdx.x``
    // 而网格是 (行, 块) 二维时，一行内的所有块会重复算同一个下标集合。
    const int row = blockIdx.y;
    if (row >= rows) return;
    const int wide = hidden * hc;
    const int hi = (tile > 0) ? min(hidden, (int)blockIdx.x * tile + tile) : hidden;
    const int lo = (tile > 0) ? (int)blockIdx.x * tile : 0;
    const XT* normp = normed + (size_t)row * wide;
    const XT* lowp = low + (size_t)row * lowrank;

    for (int i = lo + (int)threadIdx.x; i < hi; i += (int)blockDim.x) {
        float acc = 0.f;
        for (int g = 0; g < hc; ++g) {
            const int col = g * hidden + i;
            float dot = 0.f;
            for (int t = 0; t < lowrank; ++t) {
                dot += CxScalar<XT>::to_float(lowp[t])
                     * CxScalar<XT>::to_float(up_wt[(size_t)t * wide + col]);
            }
            acc += (1.f / (1.f + __expf(-dot))) * CxScalar<XT>::to_float(normp[col]);
        }
        mixed[(size_t)row * hidden + i] = CxScalar<XT>::from_float(acc / (float)hc);
    }

    // 写门只有 hc 个输出，但每个都要扫 wide 维。按 warp 分派：warp ``w`` 负责
    // 输出 ``w``，32 个 lane 分摊 wide 后用 shuffle 收口。只交给前 hc 个线程会
    // 变成 4 个线程扫 10240 次、其余 252 个空转。
    //
    // **只让 ``blockIdx.x == 0`` 做**：``tile`` 把同一行切成多个块后，每块都会走到
    // 这里，而写门的结果与 ``hidden`` 下标无关，重复写只是浪费；更要紧的是它会
    // 把 10240 维的 ``normed`` 在每个块上重读一遍。
    const int warp = (int)threadIdx.x >> 5;
    const int lane = (int)threadIdx.x & 31;
    if (blockIdx.x == 0 && warp < hc) {
        float acc = 0.f;
        for (int t = lane; t < wide; t += 32) {
            acc += CxScalar<XT>::to_float(normp[t])
                 * CxScalar<XT>::to_float(inj_wt[(size_t)t * hc + warp]);
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            acc += __shfl_down_sync(0xffffffffu, acc, o);
        }
        if (lane == 0) {
            inject[(size_t)row * hc + warp] = 2.f / (1.f + __expf(-acc / (float)hc));
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
    // 并行度 = cols × rows × splits。``n == lowrank == 320`` 时 cols 是 3，
    // 一张 16 SM 的卡要 splits >= 6 才填满。这里按 16 取，多出来的块只是把
    // 每个块的 K 段切得更短，不改变总流量。
    const int splits = (rows * cols >= 16) ? 1 : (16 / (rows * cols) + 1);
    const bool need_part = splits > 1;
    auto part = need_part
        ? torch::empty({splits, rows, n},
                       torch::TensorOptions().dtype(torch::kFloat).device(normed.device()))
        : torch::Tensor();
    const dim3 grid(cols, rows, splits);
    if (normed.scalar_type() == torch::kBFloat16) {
        hyper_down_kernel<uint16_t><<<grid, threads, 0, stream>>>(
            reinterpret_cast<const uint16_t*>(normed.data_ptr()),
            reinterpret_cast<const uint16_t*>(wt.data_ptr()),
            reinterpret_cast<uint16_t*>(out.data_ptr()), rows, k, n, (int)hc,
            splits, need_part ? part.data_ptr<float>() : nullptr);
        if (need_part) {
            hyper_down_finish_kernel<uint16_t><<<dim3(cols, rows), threads, 0, stream>>>(
                part.data_ptr<float>(), reinterpret_cast<uint16_t*>(out.data_ptr()),
                rows, n, (int)hc, splits);
        }
    } else {
        hyper_down_kernel<float><<<grid, threads, 0, stream>>>(
            normed.data_ptr<float>(), wt.data_ptr<float>(), out.data_ptr<float>(),
            rows, k, n, (int)hc, splits,
            need_part ? part.data_ptr<float>() : nullptr);
        if (need_part) {
            hyper_down_finish_kernel<float><<<dim3(cols, rows), threads, 0, stream>>>(
                part.data_ptr<float>(), out.data_ptr<float>(), rows, n, (int)hc, splits);
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
    // 写门按 warp 分派，所以块内至少要 hc 个 warp；256 线程给到 8 个，够 hc<=8。
    TORCH_CHECK((int)hc <= 8, "hc must not exceed 8 warps per block, got ", hc);
    const int threads = 256;
    // 把 ``hidden`` 切成若干块，让 ``M == 1`` 时也有多块同时跑。
    // 目标并行度是 16 个块（本卡 SM 数）：块太小会让写门那次 ``wide`` 重读占比变高，
    // 块太大则填不满。``hidden == 2560`` 时每块 160 个下标，正好 16 块。
    int tile = 0;
    {
        const int want = 16;
        const int per = (int)((hidden + want - 1) / want);
        tile = (per > 0) ? per : (int)hidden;
    }
    const int tiles = (int)((hidden + tile - 1) / tile);
    const dim3 grid((unsigned)tiles, (unsigned)rows);
    if (normed.scalar_type() == torch::kBFloat16) {
        hyper_up_kernel<uint16_t><<<grid, threads, 0, stream>>>(
            reinterpret_cast<const uint16_t*>(normed.data_ptr()),
            reinterpret_cast<const uint16_t*>(up_wt.data_ptr()),
            reinterpret_cast<const uint16_t*>(low.data_ptr()),
            reinterpret_cast<const uint16_t*>(inj_wt.data_ptr()),
            reinterpret_cast<uint16_t*>(mixed.data_ptr()), inject.data_ptr<float>(),
            rows, (int)hidden, (int)hc, (int)lowrank, tile);
    } else {
        hyper_up_kernel<float><<<grid, threads, 0, stream>>>(
            normed.data_ptr<float>(), up_wt.data_ptr<float>(), low.data_ptr<float>(),
            inj_wt.data_ptr<float>(), mixed.data_ptr<float>(), inject.data_ptr<float>(),
            rows, (int)hidden, (int)hc, (int)lowrank, tile);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return mixed;
}
