// @file gemv.cu
// @brief 单 token 解码用的向量-矩阵乘，外加一个小 M 的批处理变体。
//
// Module: coffin/kernels。每层都要经过的投影内核，绑定为 ``gemv_bf16`` /
// ``gemv_f16`` / ``gemv_bf16_m`` / ``gemv_f16_m``；int8 家族在 ``gemv_int8.cu``。
//
// 为什么不用 hipBLAS：M=1 的 GEMV 在这颗芯片上没法用（55-296 GB/s，拷贝上限约
// 700 GB/s），hipBLAS 挑中的那个内核还会随形状不可预测地变，而 Triton 没有 gfx906
// 后端。一个朴素的「每行一个 warp、128 位加载」的归约能到 540-680 GB/s，所以引擎把
// 每个投影（以及 lm_head）都走这个内核。
//
// 批处理变体是为投机解码准备的：校验 k 个草稿 token 的代价必须接近一次权重读取，而不是
// k 次。gfx906 上没有 hipBLASLt / tensor core 路径，所以 hipBLAS 的小 M GEMM 大约是一
// 个解码步的 4 倍。这里每个 warp 仍然独占一个权重行，但保留 M 个累加器、把该行同时施加
// 到 M 个输入向量上，权重因此只读一次。
//
// Constraints:
//   * 权重是行主序 [N, K]；K 循环以 16 字节（8 个值）为单位走，剩下的 K % 8 用标量尾
//     循环收掉，所以任意 K 都正确，但只有 K % 8 == 0 才走向量路径。
//   * 激活只支持 bf16：内核用移位解包 bf16，传一个 fp16 激活张量会静默读到错的值。只有
//     权重可以是 fp16（``F16W``）。
//   * 必须在当前 torch 流上启动；旧的默认流不属于 ``torch.cuda.graph`` 捕获范围，在那里
//     启动的内核会从图里被丢掉。
//   * 每个 (R, U) 和 (M, U) 组合都是编译期实例化的；不在列表里的组合是 TORCH_CHECK
//     失败，而不是慢路径。

#include "hip_ops.h"

#include <c10/cuda/CUDAStream.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace {

// bf16 以 uint16 存储，它的 fp32 值就是把这个位模式左移到高半部分，所以解包是一次
// 移位而不是尾数转换。
__device__ __forceinline__ float bf16_to_float(unsigned short h) {
    return __uint_as_float(((unsigned)h) << 16);
}

__device__ __forceinline__ unsigned short float_to_bf16(float f) {
    unsigned u = __float_as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);   // 就近舍入到偶数（round to nearest even）
    return (unsigned short)(u >> 16);
}

// fp16 共用 16 位容器但布局不同：它需要真正的尾数转换。这里权重用 fp16，是因为这颗
// 芯片没有 packed bf16 数学（bf16 GEMM 跑 2.3 TFLOP/s，fp16 能到 20），而激活保持
// bf16——拷贝循环是访存受限的，多出来的一次转换不花时间。
__device__ __forceinline__ float half_to_float(unsigned short h) {
    return __half2float(*reinterpret_cast<const __half*>(&h));
}

template <bool F16W>
__device__ __forceinline__ float w_to_float(unsigned short h) {
    return F16W ? half_to_float(h) : bf16_to_float(h);
}

// 把一个 8 元素的向量乘积加到八个相互独立的部分和上，每个 lane 因此有八条 FMA 链在飞，
// 而不是一条串行依赖链。
template <bool F16W>
__device__ __forceinline__ void accumulate(float* acc, const uint4& w, const uint4& x) {
    const unsigned short* wh = reinterpret_cast<const unsigned short*>(&w);
    const unsigned short* xh = reinterpret_cast<const unsigned short*>(&x);
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] += w_to_float<F16W>(wh[j]) * bf16_to_float(xh[j]);
}

// 每个输出行一个 warp；一个 lane 以 16 字节（8 个 bf16）为单位跨 K 方向推进。
// 每 warp 的 R 行让 x 向量摊薄到 R 行上，U 路展开则在第一次 FMA 之前就发出 U 次加载，
// 短 K 的形状正需要这个来掩盖访存延迟（K=4096 时 U=1 只有 158 GB/s，U=2 达到 264 GB/s）。
//
// @param W [N, K] 行主序权重，bf16 或 fp16 由 ``F16W`` 选择。
// @param x [K] bf16 激活。``F16W`` 为真时它仍必须是 bf16：只有权重 dtype 会变。
// @param y [N] bf16 输出，由每个 warp 所负责行的 lane 0 写出。
template <int R, int U, bool F16W>
__global__ void gemv_kernel(const uint16_t* __restrict__ W,
                            const uint16_t* __restrict__ x,
                            uint16_t* __restrict__ y, int N, int K) {
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    const int row0 = warp * R;
    if (row0 >= N) return;

    const int nvec = K >> 3;
    const uint4* xv = reinterpret_cast<const uint4*>(x);
    const uint4* w0 = reinterpret_cast<const uint4*>(W + (size_t)row0 * K);

    float acc[R][8];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[r][j] = 0.f;
    float tail[R];
#pragma unroll
    for (int r = 0; r < R; ++r) tail[r] = 0.f;

    const int step = 32 * U;
    int base = 0;
    for (; base + step <= nvec; base += step) {
        uint4 xb[U];
#pragma unroll
        for (int u = 0; u < U; ++u) xb[u] = xv[base + u * 32 + lane];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (row0 + r < N) {
                const uint4* wr = w0 + (size_t)r * nvec;
#pragma unroll
                for (int u = 0; u < U; ++u) accumulate<F16W>(acc[r], wr[base + u * 32 + lane], xb[u]);
            }
        }
    }
    for (int i = base + lane; i < nvec; i += 32) {
        const uint4 xb = xv[i];
#pragma unroll
        for (int r = 0; r < R; ++r)
            if (row0 + r < N) accumulate<F16W>(acc[r], w0[(size_t)r * nvec + i], xb);
    }
    // K 不是 8 的整数倍时，剩下的元素没有向量可装。
    for (int k = (nvec << 3) + lane; k < K; k += 32) {
        const float xf = bf16_to_float(x[k]);
#pragma unroll
        for (int r = 0; r < R; ++r)
            if (row0 + r < N)
                tail[r] = fmaf(w_to_float<F16W>(W[(size_t)(row0 + r) * K + k]), xf, tail[r]);
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        float s = tail[r];
#pragma unroll
        for (int j = 0; j < 8; ++j) s += acc[r][j];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) s += __shfl_down(s, o);
        if (lane == 0 && row0 + r < N) y[row0 + r] = float_to_bf16(s);
    }
}

// 每个权重行一个 warp，每 lane 保留 M 个累加器：该行只读一次，同时施加到 M 个输入向量
// 上，小 M 的批因此只花约一次权重读取。
//
// @param X [M, K] bf16，批里每个元素一行激活向量。
// @param Y [M, N] bf16；由该 warp 的 lane 0 写出全部 M 行的第 ``warp`` 列。
template <int M, int U, bool F16W>
__global__ void gemv_m_kernel(const uint16_t* __restrict__ W,
                              const uint16_t* __restrict__ X,
                              uint16_t* __restrict__ Y, int N, int K) {
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    if (warp >= N) return;
    const int nvec = K >> 3;
    const uint4* wv = reinterpret_cast<const uint4*>(W + (size_t)warp * K);
    const uint4* xv = reinterpret_cast<const uint4*>(X);

    float acc[M][8];
#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[m][j] = 0.f;
    float tail[M];
#pragma unroll
    for (int m = 0; m < M; ++m) tail[m] = 0.f;

    const int step = 32 * U;
    int base = 0;
    for (; base + step <= nvec; base += step) {
        uint4 xb[M][U];
#pragma unroll
        for (int m = 0; m < M; ++m)
#pragma unroll
            for (int u = 0; u < U; ++u) xb[m][u] = xv[(size_t)m * nvec + base + u * 32 + lane];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const uint4 wc = wv[base + u * 32 + lane];
#pragma unroll
            for (int m = 0; m < M; ++m) accumulate<F16W>(acc[m], wc, xb[m][u]);
        }
    }
    for (int i = base + lane; i < nvec; i += 32) {
        const uint4 wc = wv[i];
#pragma unroll
        for (int m = 0; m < M; ++m) accumulate<F16W>(acc[m], wc, xv[(size_t)m * nvec + i]);
    }
    for (int k = (nvec << 3) + lane; k < K; k += 32) {
        const float wf = w_to_float<F16W>(W[(size_t)warp * K + k]);
#pragma unroll
        for (int m = 0; m < M; ++m)
            tail[m] = fmaf(wf, bf16_to_float(X[(size_t)m * K + k]), tail[m]);
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
        float s = tail[m];
#pragma unroll
        for (int j = 0; j < 8; ++j) s += acc[m][j];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) s += __shfl_down(s, o);
        if (lane == 0) Y[(size_t)m * N + warp] = float_to_bf16(s);
    }
}

}  // namespace

namespace {

// 两种权重 dtype 共用一条启动路径，只有内核里的解包不同（见 ``w_to_float``）。随版本
// 发布的每个 (R, U) 组合都是一次显式实例化，所以新增组合必须同时加在这里和 Python 侧的
// 基准测试列表里。
template <bool F16W>
void launch_gemv(const uint16_t* wp, const uint16_t* xp, uint16_t* yp, int N, int K,
                 int R, int U, int blocks, int threads, hipStream_t stream) {
#define GEMV_CASE(R_, U_)                                                            \
    if (R == R_ && U == U_) {                                                        \
        gemv_kernel<R_, U_, F16W><<<blocks, threads, 0, stream>>>(wp, xp, yp, N, K); \
        return;                                                                      \
    }
    GEMV_CASE(1, 1) GEMV_CASE(1, 2) GEMV_CASE(1, 4)
    GEMV_CASE(2, 1) GEMV_CASE(2, 2)
    GEMV_CASE(4, 1)
#undef GEMV_CASE
    TORCH_CHECK(false, "unsupported (R, U) for the GEMV kernel");
}

template <bool F16W>
torch::Tensor gemv_entry(torch::Tensor W, torch::Tensor x, int64_t R, int64_t U) {
    TORCH_CHECK(W.is_contiguous() && x.is_contiguous(), "contiguous inputs required");
    TORCH_CHECK(W.scalar_type() == (F16W ? at::kHalf : at::kBFloat16),
                "weight dtype does not match the kernel");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16, "activations stay bf16");
    const int N = W.size(0), K = W.size(1);
    // 输出跟随的是*激活*的 dtype：预填充 GEMM 的权重是 fp16，但解码的激活流仍是 bf16。
    auto y = torch::empty({N}, x.options());
    const int threads = 256, warps_per_block = threads / 32;
    const int rows_per_block = (int)R * warps_per_block;
    const int blocks = (N + rows_per_block - 1) / rows_per_block;
    const auto* wp = reinterpret_cast<const uint16_t*>(W.data_ptr());
    const auto* xp = reinterpret_cast<const uint16_t*>(x.data_ptr());
    auto* yp = reinterpret_cast<uint16_t*>(y.data_ptr());
    // 必须是当前流：旧的默认流不属于 torch.cuda.graph 捕获范围，在那里启动的内核会被
    // 静默从图里丢掉，重放出来就是垃圾。
    hipStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_gemv<F16W>(wp, xp, yp, N, K, (int)R, (int)U, blocks, threads, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

template <bool F16W>
void launch_gemv_m(const uint16_t* wp, const uint16_t* xp, uint16_t* yp, int N, int K,
                   int M, int U, int blocks, int threads, hipStream_t stream) {
    bool done = false;
#define GEMV_M_CASE(M_, U_)                                                       \
    if (!done && M == M_ && U == U_) {                                            \
        gemv_m_kernel<M_, U_, F16W><<<blocks, threads, 0, stream>>>(wp, xp, yp, N, K); \
        done = true;                                                              \
    }
    GEMV_M_CASE(2, 1) GEMV_M_CASE(2, 2) GEMV_M_CASE(2, 4)
    GEMV_M_CASE(4, 1) GEMV_M_CASE(4, 2) GEMV_M_CASE(4, 4)
    GEMV_M_CASE(8, 1) GEMV_M_CASE(8, 2)
#undef GEMV_M_CASE
    TORCH_CHECK(done, "unsupported M/U for the batched GEMV kernel");
}

template <bool F16W>
torch::Tensor gemv_m_entry(torch::Tensor W, torch::Tensor X, int64_t U) {
    TORCH_CHECK(W.is_contiguous() && X.is_contiguous(), "contiguous inputs required");
    TORCH_CHECK(W.scalar_type() == (F16W ? at::kHalf : at::kBFloat16),
                "weight dtype does not match the kernel");
    TORCH_CHECK(X.scalar_type() == at::kBFloat16, "activations stay bf16");
    const int N = W.size(0), K = W.size(1), M = X.size(0);
    TORCH_CHECK(X.size(1) == K, "X's K does not match W");
    auto Y = torch::empty({M, N}, X.options());
    const int threads = 256, warps_per_block = threads / 32;
    const int blocks = (N + warps_per_block - 1) / warps_per_block;
    const auto* wp = reinterpret_cast<const uint16_t*>(W.data_ptr());
    const auto* xp = reinterpret_cast<const uint16_t*>(X.data_ptr());
    auto* yp = reinterpret_cast<uint16_t*>(Y.data_ptr());
    hipStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_gemv_m<F16W>(wp, xp, yp, N, K, M, (int)U, blocks, threads, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return Y;
}

}  // namespace

// @brief 单 token bf16 GEMV：y[n] = dot(W[n, :], x)。
//
// @param W [N, K] bf16 行主序。
// @param x [K] bf16；与 W 一样连续存放。
// @param R 每个 warp 的输出行数（1、2 或 4）；由加载时的基准测试挑选。
// @param U K 循环展开（1、2 或 4）；同样是基准测试挑的，且它只影响延迟掩盖，不影响算术。
// @return [N] bf16。
// @throws c10::Error（TORCH_CHECK）：输入不连续、权重 dtype 不是 bf16，或 (R, U) 组合
//         没有已实例化的内核。
torch::Tensor gemv_bf16(torch::Tensor W, torch::Tensor x, int64_t R, int64_t U) {
    return gemv_entry<false>(W, x, R, U);
}

// @brief 与 ``gemv_bf16`` 相同，但权重是 fp16；预填充 GEMM 读的是同一批张量，所以权重
//        存成 fp16 来保持那条路径够快。
//
// @param W [N, K] fp16 行主序。
// @return [N] bf16：权重不是 bf16，但激活流仍然是。
// @throws c10::Error（TORCH_CHECK）：dtype 或 (R, U) 不匹配。
torch::Tensor gemv_f16(torch::Tensor W, torch::Tensor x, int64_t R, int64_t U) {
    return gemv_entry<true>(W, x, R, U);
}

// @brief 批处理的小 M bf16 GEMV：一次权重读取服务全部 M 个激活。
//
// @param X [M, K] bf16；M 必须是 2、4 或 8，即已实例化的取值。
// @param U K 循环展开。
// @return [M, N] bf16。
// @throws c10::Error（TORCH_CHECK）：M/U 没有已实例化的内核，或形状对不上；M 由调用方
//         按校验批大小挑选。
torch::Tensor gemv_bf16_m(torch::Tensor W, torch::Tensor X, int64_t U) {
    return gemv_m_entry<false>(W, X, U);
}

// @brief 权重为 fp16 的批处理小 M GEMV。
//
// @param X [M, K] bf16；M 必须是 2、4 或 8。
// @return [M, N] bf16。
// @throws c10::Error（TORCH_CHECK）：M/U 没有已实例化的内核。
torch::Tensor gemv_f16_m(torch::Tensor W, torch::Tensor X, int64_t U) {
    return gemv_m_entry<true>(W, X, U);
}
