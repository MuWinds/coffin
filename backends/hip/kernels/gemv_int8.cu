// @file gemv_int8.cu
// @brief Weight-only int8 GEMV (W8A16) for decode.
//
// Module: coffin/kernels. The quantised half of the projection path; the fp16/bf16
// family is ``gemv.cu`` and the M in {16, 24, 32} verify variant is
// ``gemv_wide.cu``, which must stay bit-identical to the M=8 kernel here.
//
// Why: decode on this part is bandwidth-bound. The projections run at 540-680 GB/s
// against a ~700 GB/s copy ceiling and are ~48% of the step, hipBLAS GEMV at M=1
// is unusable and Triton has no gfx906 backend. The only remaining lever on that
// 48% is reading fewer bytes per weight, so the weights are stored as int8 with a
// per-row scale and read once per decode step. Activations stay bf16: the kernel
// is memory-bound, so the int8 -> float conversion is hidden, and keeping A in
// bf16 avoids a second quantiser and its error.
//
// Constraints:
//   * Weights are INT8 row-major [N, K] with a per-row fp32 scale; a column-major
//     or per-group layout would break both this kernel and every prepacked
//     checkpoint.
//   * The structure is deliberately identical to ``gemv_kernel`` (one warp per
//     output row, R rows per warp, U-way K unroll, 8 accumulators per lane): only
//     the weight load changes, from one uint4 (8 bf16, 16 B) to one uint2 (8 int8,
//     8 B), and the row scale is applied once at the end instead of per element.
//   * Only the listed (R, U) / (M, U) pairs are instantiated; anything else is a
//     TORCH_CHECK failure.
//   * ``dk``-style tails: K % 8 is handled by a scalar tail, so any K is correct.

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
    u += 0x7fffu + ((u >> 16) & 1u);   // round to nearest even
    return (unsigned short)(u >> 16);
}

// @brief Eight int8 (one uint2) against eight bf16 (one uint4).
//
// Extracting the bytes by shift keeps the packed weight in registers; a
// reinterpret_cast to char* would force it to local memory.
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

// @brief One warp per output row, R rows per warp, 8 FMA chains per lane.
//
// @param S [N] fp32 per-row dequant scale, applied once after the reduction.
// @param x [K] bf16 activation.
// @param y [N] bf16 output; lane 0 writes the row's scaled sum.
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
    // K not a multiple of 8: the leftover elements have no vector to live in.
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

// @brief Batched small-M variant: the row (and its scale) is read once for M
//        input vectors.
//
// That is what keeps speculative verification at ~one weight pass.
//
// @param X [M, K] bf16; M is the instantiated template parameter.
// @param Y [M, N] bf16; the scale is applied per row after the reduction.
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

// @brief Weight-only int8 GEMV for a single decode token.
//
// @param W [N, K] int8 row-major, contiguous.
// @param S [N] fp32 per-row scale, contiguous.
// @param x [K] bf16 activation, contiguous.
// @param R Output rows per warp (1, 2 or 4); picked by the load-time benchmark.
// @param U K-loop unroll; only affects latency hiding, not the arithmetic.
// @return [N] bf16.
// @throws c10::Error (TORCH_CHECK) on a non-contiguous input or a weight dtype
//         that is not int8.
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

// @brief Batched small-M int8 GEMV: one weight pass for M activation vectors.
//
// @param X [M, K] bf16; M must be 2, 4 or 8, the instantiated values. The verify
//          path with M in {16, 24, 32} is ``gemv_wide.cu`` instead.
// @param U K-loop unroll.
// @return [M, N] bf16, bit-identical to calling ``gemv_w8a16`` on each row.
// @throws c10::Error (TORCH_CHECK) when M/U has no instantiated kernel.
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
