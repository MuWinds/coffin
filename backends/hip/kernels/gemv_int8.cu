// @file gemv_int8.cu
// @brief 解码用的纯权重 int8 GEMV（W8A16）。
//
// Module: coffin/kernels。投影路径的量化那一半；fp16/bf16 家族在 ``gemv.cu``，M 取
// {16, 24, 32} 的校验变体在 ``gemv_wide.cu``，后者必须与这里的 M=8 内核逐位一致。
//
// 为什么：这颗芯片上解码是带宽受限的。投影跑到 540-680 GB/s，而拷贝上限约 700 GB/s，
// 占该步 ~48%；hipBLAS 的 M=1 GEMV 没法用，Triton 又没有 gfx906 后端。那 48% 上仅剩的
// 杠杆就是每个权重少读几个字节，于是权重以 int8 存储、每行一个 scale，每个解码步只读
// 一次。激活保持 bf16：内核是访存受限的，int8 -> float 的转换被掩盖掉了，而 A 留在 bf16
// 也省掉了第二个量化器及其误差。
//
// Constraints:
//   * 权重是 INT8 行主序 [N, K] 配每行一个 fp32 scale；列主序或按组布局会同时弄坏本内核
//     和每一个预打包 checkpoint。
//   * 结构与 ``gemv_kernel`` 刻意保持一致（每个输出行一个 warp，每 warp R 行，K 方向 U
//     路展开，每 lane 8 个累加器）：只有权重读取变了，从一次 uint4（8 个 bf16，16 B）变成
//     一次 uint2（8 个 int8，8 B），行 scale 也改成最后统一乘一次而不是逐元素乘。
//   * 只实例化列出的 (R, U) / (M, U) 组合；别的组合一律 TORCH_CHECK 失败。
//   * 尾部处理：K % 8 用标量尾循环收掉，所以任意 K 都是对的。

#include "hip_ops.h"

#include <c10/cuda/CUDAStream.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace {

__device__ __forceinline__ float bf16_to_float(unsigned short h) {
    return __uint_as_float(((unsigned)h) << 16);
}

__device__ __forceinline__ unsigned short float_to_bf16(float f) {
    unsigned u = __float_as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);   // 就近舍入到偶数（round to nearest even）
    return (unsigned short)(u >> 16);
}

// @brief 八个 int8（一个 uint2）对八个 bf16（一个 uint4）。
//
// 用移位来取字节，打包的权重才能留在寄存器里；换成 reinterpret_cast 到 char* 会把它逼
// 进 local memory。
__device__ __forceinline__ void accumulate_i8(float* acc, const uint2& w, const uint4& x) {
    const unsigned short* xh = reinterpret_cast<const unsigned short*>(&x);
    const unsigned lo = w.x, hi = w.y;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const unsigned raw = (j < 4) ? (lo >> (8 * j)) : (hi >> (8 * (j - 4)));
        const float wv = (float)(int)(signed char)(raw & 0xffu);
        acc[j] += wv * bf16_to_float(xh[j]);
    }
}

// @brief 每个输出行一个 warp，每 warp R 行，每 lane 8 条 FMA 链。
//
// @param S [N] fp32 每行反量化 scale，在归约之后统一乘一次。
// @param x [K] bf16 激活。
// @param y [N] bf16 输出；由 lane 0 写出该行乘过 scale 的和。
template <int R, int U>
__global__ void gemv_w8a16_kernel(const int8_t* __restrict__ W,
                                  const float* __restrict__ S,
                                  const uint16_t* __restrict__ x,
                                  uint16_t* __restrict__ y, int N, int K) {
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    const int row0 = warp * R;
    if (row0 >= N) return;

    const int nvec = K >> 3;
    const uint4* xv = reinterpret_cast<const uint4*>(x);
    const uint2* w0 = reinterpret_cast<const uint2*>(W + (size_t)row0 * K);

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
                const uint2* wr = w0 + (size_t)r * nvec;
#pragma unroll
                for (int u = 0; u < U; ++u) accumulate_i8(acc[r], wr[base + u * 32 + lane], xb[u]);
            }
        }
    }
    for (int i = base + lane; i < nvec; i += 32) {
        const uint4 xb = xv[i];
#pragma unroll
        for (int r = 0; r < R; ++r)
            if (row0 + r < N) accumulate_i8(acc[r], w0[(size_t)r * nvec + i], xb);
    }
    // K 不是 8 的整数倍时，剩下的元素没有向量可装。
    for (int k = (nvec << 3) + lane; k < K; k += 32) {
        const float xf = bf16_to_float(x[k]);
#pragma unroll
        for (int r = 0; r < R; ++r)
            if (row0 + r < N)
                tail[r] = fmaf((float)W[(size_t)(row0 + r) * K + k], xf, tail[r]);
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        float s = tail[r];
#pragma unroll
        for (int j = 0; j < 8; ++j) s += acc[r][j];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) s += __shfl_down(s, o);
        if (lane == 0 && row0 + r < N) {
            const unsigned idx = row0 + r;
            y[idx] = float_to_bf16(s * S[idx]);
        }
    }
}

// @brief 批处理的小 M 变体：一次读取权重行（及其 scale）服务 M 个输入向量。
//
// 投机校验因此只花大约一次权重读取。
//
// @param X [M, K] bf16；M 就是实例化的模板参数。
// @param Y [M, N] bf16；scale 在归约之后按行乘上。
template <int M, int U>
__global__ void gemv_w8a16_m_kernel(const int8_t* __restrict__ W,
                                    const float* __restrict__ S,
                                    const uint16_t* __restrict__ X,
                                    uint16_t* __restrict__ Y, int N, int K) {
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    if (warp >= N) return;
    const int nvec = K >> 3;
    const uint2* wv = reinterpret_cast<const uint2*>(W + (size_t)warp * K);
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
            const uint2 wc = wv[base + u * 32 + lane];
#pragma unroll
            for (int m = 0; m < M; ++m) accumulate_i8(acc[m], wc, xb[m][u]);
        }
    }
    for (int i = base + lane; i < nvec; i += 32) {
        const uint2 wc = wv[i];
#pragma unroll
        for (int m = 0; m < M; ++m) accumulate_i8(acc[m], wc, xv[(size_t)m * nvec + i]);
    }
    for (int k = (nvec << 3) + lane; k < K; k += 32) {
        const float wf = (float)W[(size_t)warp * K + k];
#pragma unroll
        for (int m = 0; m < M; ++m)
            tail[m] = fmaf(wf, bf16_to_float(X[(size_t)m * K + k]), tail[m]);
    }
    const float sc = S[warp];
#pragma unroll
    for (int m = 0; m < M; ++m) {
        float s = tail[m];
#pragma unroll
        for (int j = 0; j < 8; ++j) s += acc[m][j];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) s += __shfl_down(s, o);
        if (lane == 0) Y[(size_t)m * N + warp] = float_to_bf16(s * sc);
    }
}

}  // namespace

// @brief 单个解码 token 的纯权重 int8 GEMV。
//
// @param W [N, K] int8 行主序，连续存放。
// @param S [N] fp32 每行 scale，连续存放。
// @param x [K] bf16 激活，连续存放。
// @param R 每个 warp 的输出行数（1、2 或 4）；由加载时的基准测试挑选。
// @param U K 循环展开；只影响延迟掩盖，不影响算术。
// @return [N] bf16。
// @throws c10::Error（TORCH_CHECK）：输入不连续，或权重 dtype 不是 int8。
torch::Tensor gemv_w8a16(torch::Tensor W, torch::Tensor S, torch::Tensor x,
                         int64_t R, int64_t U) {
    TORCH_CHECK(W.is_contiguous() && x.is_contiguous() && S.is_contiguous(),
                "contiguous inputs required");
    TORCH_CHECK(W.scalar_type() == at::kChar, "W must be int8");
    const int N = W.size(0), K = W.size(1);
    auto y = torch::empty({N}, x.options());
    const int threads = 256, warps_per_block = threads / 32;
    const int rows_per_block = (int)R * warps_per_block;
    const int blocks = (N + rows_per_block - 1) / rows_per_block;
    const auto* wp = W.data_ptr<int8_t>();
    const auto* sp = S.data_ptr<float>();
    const auto* xp = reinterpret_cast<const uint16_t*>(x.data_ptr<at::BFloat16>());
    auto* yp = reinterpret_cast<uint16_t*>(y.data_ptr<at::BFloat16>());
    hipStream_t stream = at::cuda::getCurrentCUDAStream();
    const int rr = (int)R, uu = (int)U;
#define GEMV_W8_CASE(R_, U_)                                                      \
    if (rr == R_ && uu == U_) {                                                   \
        gemv_w8a16_kernel<R_, U_><<<blocks, threads, 0, stream>>>(wp, sp, xp, yp, N, K); \
    }
    GEMV_W8_CASE(1, 1) GEMV_W8_CASE(1, 2) GEMV_W8_CASE(1, 4)
    GEMV_W8_CASE(2, 1) GEMV_W8_CASE(2, 2)
    GEMV_W8_CASE(4, 1)
#undef GEMV_W8_CASE
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

// @brief 批处理的小 M int8 GEMV：一次权重读取服务 M 个激活向量。
//
// @param X [M, K] bf16；M 必须是 2、4 或 8，即已实例化的取值。M 取 {16, 24, 32} 的校验
//          路径走 ``gemv_wide.cu``。
// @param U K 循环展开。
// @return [M, N] bf16，与逐行调用 ``gemv_w8a16`` 逐位一致。
// @throws c10::Error（TORCH_CHECK）：M/U 没有对应的已实例化内核。
torch::Tensor gemv_w8a16_m(torch::Tensor W, torch::Tensor S, torch::Tensor X, int64_t U) {
    TORCH_CHECK(W.is_contiguous() && X.is_contiguous() && S.is_contiguous(),
                "contiguous inputs required");
    const int N = W.size(0), K = W.size(1), M = X.size(0);
    TORCH_CHECK(X.size(1) == K, "X's K does not match W");
    auto Y = torch::empty({M, N}, X.options());
    const int threads = 256, warps_per_block = threads / 32;
    const int blocks = (N + warps_per_block - 1) / warps_per_block;
    const auto* wp = W.data_ptr<int8_t>();
    const auto* sp = S.data_ptr<float>();
    const auto* xp = reinterpret_cast<const uint16_t*>(X.data_ptr<at::BFloat16>());
    auto* yp = reinterpret_cast<uint16_t*>(Y.data_ptr<at::BFloat16>());
    hipStream_t stream = at::cuda::getCurrentCUDAStream();
    const int uu = (int)U;
    bool done = false;
#define GEMV_W8_M_CASE(M_, U_)                                                    \
    if (!done && M == M_ && uu == U_) {                                           \
        gemv_w8a16_m_kernel<M_, U_><<<blocks, threads, 0, stream>>>(wp, sp, xp, yp, N, K); \
        done = true;                                                              \
    }
    GEMV_W8_M_CASE(2, 1) GEMV_W8_M_CASE(2, 2) GEMV_W8_M_CASE(2, 4)
    GEMV_W8_M_CASE(4, 1) GEMV_W8_M_CASE(4, 2) GEMV_W8_M_CASE(4, 4)
    GEMV_W8_M_CASE(8, 1) GEMV_W8_M_CASE(8, 2)
#undef GEMV_W8_M_CASE
    TORCH_CHECK(done, "unsupported M/U for gemv_w8a16_m");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return Y;
}
