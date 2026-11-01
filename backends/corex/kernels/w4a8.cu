// @file w4a8.cu
// @brief 从 packed W4 和逐 token int8 激活计算分组整数点积。
//
// Module: coffin/backends/corex/kernels。由 CoreX 投影派发调用，不依赖模型。
//
// Constraints:
//   * 每个 K 组先做 int32 累加，再乘该组 scale；不能把不同组的整数和直接相加。
//   * 解包只在寄存器中发生，常驻权重始终为 uint8 packed，不分配整矩阵 int8 副本。
//   * 不使用 graph 或异步拷贝指令，所有启动都在调用设备的当前流上。

#include "corex_ops.h"
#include "corex_scalar.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <climits>

namespace {

// @brief 行并行的各 rank 共用全局 scale，避免分片改变激活量化的数学定义。
template <typename XT>
__global__ void quant_with_scale(const XT* x, int8_t* out, const float* xs, int K) {
    const int row = blockIdx.x;
    const float inv = 1.f / xs[row];
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        const int value = __float2int_rn(CxScalar<XT>::to_float(x[(size_t)row * K + k]) * inv);
        out[(size_t)row * K + k] = (int8_t)max(-127, min(127, value));
    }
}

// @brief 同一权重加载服务 MT 行，K 分片只能在组边界切开。
template <int MT>
__global__ void w4a8_dot(const int8_t* x, const uint8_t* w, const float* ws,
                         float* partial, int M, int N, int K, int gs,
                         int groups_per_split) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const int m0 = blockIdx.y * MT;
    const int split = blockIdx.z;
    const int groups = (K + gs - 1) / gs;
    const int g1 = min(groups, (split + 1) * groups_per_split);
    float total[MT] = {};
    for (int g = split * groups_per_split; g < g1; ++g) {
        int sums[MT] = {};
        int k = g * gs;
        const int end = min(K, k + gs);
        // dp4a 的两个字必须沿 K 配对，不能把相邻输出列误当成四个归约元素。
        for (; (K & 3) == 0 && (k & 3) == 0 && k + 4 <= end; k += 4) {
            const unsigned lo = w[(size_t)(k / 2) * N + n];
            const unsigned hi = w[(size_t)(k / 2 + 1) * N + n];
            const unsigned q = ((unsigned)((int)(lo & 15) - 8) & 255)
                | (((unsigned)((int)(lo >> 4) - 8) & 255) << 8)
                | (((unsigned)((int)(hi & 15) - 8) & 255) << 16)
                | (((unsigned)((int)(hi >> 4) - 8) & 255) << 24);
#pragma unroll
            for (int r = 0; r < MT; ++r) {
                if (m0 + r < M) {
                    const int a = *reinterpret_cast<const int*>(x + (size_t)(m0 + r) * K + k);
                    sums[r] = __dp4a((int)q, a, sums[r]);
                }
            }
        }
        for (; k < end; ++k) {
            const unsigned b = w[(size_t)(k / 2) * N + n];
            const int q = (int)((b >> ((k & 1) * 4)) & 15) - 8;
#pragma unroll
            for (int r = 0; r < MT; ++r) {
                if (m0 + r < M) sums[r] += q * (int)x[(size_t)(m0 + r) * K + k];
            }
        }
        const float scale = ws[(size_t)g * N + n];
#pragma unroll
        for (int r = 0; r < MT; ++r) total[r] += (float)sums[r] * scale;
    }
#pragma unroll
    for (int r = 0; r < MT; ++r) {
        if (m0 + r < M) partial[((size_t)split * M + m0 + r) * N + n] = total[r];
    }
}

// @brief 合并组分片后仅窄化一次，避免每组先舍入到 bf16。
template <typename XT>
__global__ void w4a8_finish(const float* partial, const float* xs, XT* out,
                            int M, int N, int splits) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;
    if (n >= N) return;
    float sum = 0.f;
    for (int s = 0; s < splits; ++s) sum += partial[((size_t)s * M + m) * N + n];
    out[(size_t)m * N + n] = CxScalar<XT>::from_float(sum * xs[m]);
}

}  // namespace

// @brief 用调用方归约好的逐行 scale 量化激活，供行并行投影复用。
torch::Tensor quant_act_i8_scaled(torch::Tensor x, torch::Tensor out, torch::Tensor xs) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous()
                && x.size(0) <= 65535 && x.size(1) > 0 && x.size(1) <= INT_MAX,
                "contiguous CUDA [M,K] activation required");
    TORCH_CHECK(out.device() == x.device() && xs.device() == x.device()
                && out.sizes() == x.sizes() && out.scalar_type() == torch::kChar
                && out.is_contiguous() && xs.is_contiguous()
                && xs.dim() == 1 && xs.size(0) == x.size(0)
                && xs.scalar_type() == torch::kFloat, "quantization destination mismatch");
    TORCH_CHECK(x.scalar_type() == torch::kFloat || x.scalar_type() == torch::kBFloat16,
                "bf16 or fp32 activation required");
    if (!x.size(0)) return out;
    const c10::cuda::CUDAGuard guard(x.device());
    const auto stream = c10::cuda::getCurrentCUDAStream();
    if (x.scalar_type() == torch::kFloat)
        quant_with_scale<float><<<x.size(0), 256, 0, stream>>>(x.data_ptr<float>(),
            out.data_ptr<int8_t>(), xs.data_ptr<float>(), x.size(1));
    else
        quant_with_scale<uint16_t><<<x.size(0), 256, 0, stream>>>(
            reinterpret_cast<const uint16_t*>(x.data_ptr()), out.data_ptr<int8_t>(),
            xs.data_ptr<float>(), x.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// @brief 对分组 W4 与 A8 做投影，权重解包只发生在寄存器内。
// @param a [M,K] 连续 int8 激活。
// @param w [至少 ceil(K/2),N] 连续 uint8 packed 权重。
// @param xs [M] fp32 激活 scale。
// @param ws [ceil(K/group_size),N] fp32 权重 scale。
// @param out [M,N] 连续 bf16 或 fp32 输出。
// @param group_size K 轴权重分组大小。
// @return out。
// @throws c10::Error 布局、设备、dtype、形状或整数累加范围不符合契约。
torch::Tensor w4a8_gemv(torch::Tensor a, torch::Tensor w, torch::Tensor xs,
                        torch::Tensor ws, torch::Tensor out, int64_t group_size) {
    TORCH_CHECK(a.is_cuda() && a.dim() == 2 && a.scalar_type() == torch::kChar,
                "activation must be CUDA int8 [M,K]");
    TORCH_CHECK(w.dim() == 2 && w.scalar_type() == torch::kByte,
                "weight must be uint8 [ceil(K/2),N]");
    const auto M = a.size(0), K = a.size(1), N = w.size(1);
    TORCH_CHECK(K > 0 && N > 0 && group_size > 0 && group_size <= 1048576,
                "invalid shape or group size");
    TORCH_CHECK(K <= INT_MAX && N <= INT_MAX && M <= 65535,
                "shape exceeds kernel index range");
    TORCH_CHECK(w.size(0) * 2 >= K && ws.dim() == 2
                && ws.size(0) == (K + group_size - 1) / group_size && ws.size(1) == N,
                "weight/scale shape mismatch");
    TORCH_CHECK(xs.dim() == 1 && xs.numel() == M && xs.scalar_type() == torch::kFloat
                && ws.scalar_type() == torch::kFloat, "scales must be fp32");
    TORCH_CHECK(out.dim() == 2 && out.size(0) == M && out.size(1) == N
                && (out.scalar_type() == torch::kFloat || out.scalar_type() == torch::kBFloat16),
                "output must be bf16/fp32 [M,N]");
    for (const auto& t : {a, w, xs, ws, out}) {
        TORCH_CHECK(t.is_contiguous() && t.device() == a.device(),
                    "all tensors must be contiguous on the same CUDA device");
    }
    if (M == 0) return out;
    const c10::cuda::CUDAGuard guard(a.device());
    const int threads = 128;
    const int blocks = (N + threads - 1) / threads;
    const int groups = (K + group_size - 1) / group_size;
    const int rows_per_tile = M == 1 ? 1 : 4;
    const int row_tiles = (M + rows_per_tile - 1) / rows_per_tile;
    const int splits = std::min(groups, std::min(16, std::max(1, 256 / (blocks * row_tiles))));
    auto partial = torch::empty({splits, M, N}, xs.options());
    const auto stream = c10::cuda::getCurrentCUDAStream();
    const dim3 grid(blocks, row_tiles, splits);
    if (rows_per_tile == 1) {
        w4a8_dot<1><<<grid, threads, 0, stream>>>(a.data_ptr<int8_t>(), w.data_ptr<uint8_t>(),
            ws.data_ptr<float>(), partial.data_ptr<float>(), M, N, K, group_size,
            (groups + splits - 1) / splits);
    } else {
        w4a8_dot<4><<<grid, threads, 0, stream>>>(a.data_ptr<int8_t>(), w.data_ptr<uint8_t>(),
            ws.data_ptr<float>(), partial.data_ptr<float>(), M, N, K, group_size,
            (groups + splits - 1) / splits);
    }
    const dim3 finish_grid(blocks, M);
    if (out.scalar_type() == torch::kFloat) {
        w4a8_finish<float><<<finish_grid, threads, 0, stream>>>(partial.data_ptr<float>(),
            xs.data_ptr<float>(), out.data_ptr<float>(), M, N, splits);
    } else {
        w4a8_finish<uint16_t><<<finish_grid, threads, 0, stream>>>(partial.data_ptr<float>(),
            xs.data_ptr<float>(), reinterpret_cast<uint16_t*>(out.data_ptr()), M, N, splits);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
