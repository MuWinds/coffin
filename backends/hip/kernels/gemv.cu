// @file gemv.cu
// @brief Vector-matrix product for single-token decode, plus a batched small-M
//        variant.
//
// Module: coffin/kernels. The projection kernel every layer routes through, bound
// as ``gemv_bf16`` / ``gemv_f16`` / ``gemv_bf16_m`` / ``gemv_f16_m``; the int8
// family lives in ``gemv_int8.cu``.
//
// Why not hipBLAS: GEMV at M=1 is useless on this part (55-296 GB/s against a
// ~700 GB/s copy ceiling), the winning hipBLAS kernel changes unpredictably from
// shape to shape, and Triton has no gfx906 backend. A plain warp-per-row reduce
// with 128-bit loads reaches 540-680 GB/s, so the engine routes every projection
// (and lm_head) through this kernel instead.
//
// The batched variant exists for speculative decoding: verifying k drafted tokens
// must cost about one weight pass, not k. hipBLAS has no hipBLASLt / tensor-core
// path for gfx906, so its small-M GEMM is ~4x a decode step. Here each warp still
// owns one weight row, but keeps M accumulators and applies the row to M input
// vectors, so the weights are read exactly once.
//
// Constraints:
//   * Weights are row-major [N, K]; the K loop walks 16-byte (8 value) units and
//     the leftover K % 8 is a scalar tail, so any K is correct but only K % 8 == 0
//     gets the vector path.
//   * bf16 activations only: the kernel unpacks bf16 by a shift, so an fp16
//     activation tensor would silently read the wrong values. Only the weights can
//     be fp16 (``F16W``).
//   * The launch must be on the current torch stream; the legacy default stream is
//     not part of a ``torch.cuda.graph`` capture and a kernel launched there is
//     dropped from the graph.
//   * Every (R, U) and (M, U) pair is instantiated at compile time; an unlisted
//     pair is a TORCH_CHECK failure, not a slow path.

#include "hip_ops.h"

#include <c10/cuda/CUDAStream.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace {

// bf16 is stored as uint16 and the fp32 value is that bit pattern shifted into
// the high half, so unpacking is a shift rather than a mantissa conversion.
__device__ __forceinline__ float bf16_to_float(unsigned short h) {
    return __uint_as_float(((unsigned)h) << 16);
}

__device__ __forceinline__ unsigned short float_to_bf16(float f) {
    unsigned u = __float_as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);   // round to nearest even
    return (unsigned short)(u >> 16);
}

// fp16 shares the 16-bit container but not the layout: it needs a real mantissa
// conversion. Weights are fp16 here because this part has no packed bf16 math
// (bf16 GEMM runs at 2.3 TFLOP/s, fp16 at 20), while activations stay bf16 --
// the copy loop is memory-bound, so the extra convert is free.
__device__ __forceinline__ float half_to_float(unsigned short h) {
    return __half2float(*reinterpret_cast<const __half*>(&h));
}

template <bool F16W>
__device__ __forceinline__ float w_to_float(unsigned short h) {
    return F16W ? half_to_float(h) : bf16_to_float(h);
}

// Adds one 8-element vector product into eight independent partial sums, so each
// lane keeps eight FMA chains in flight instead of one serial dependency.
template <bool F16W>
__device__ __forceinline__ void accumulate(float* acc, const uint4& w, const uint4& x) {
    const unsigned short* wh = reinterpret_cast<const unsigned short*>(&w);
    const unsigned short* xh = reinterpret_cast<const unsigned short*>(&x);
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] += w_to_float<F16W>(wh[j]) * bf16_to_float(xh[j]);
}

// One warp per output row; a lane strides over K in 16-byte (8 bf16) units.
// R rows per warp amortise the x vector over R rows, and U-way unrolling issues
// U loads before the first FMA, which is what short-K shapes need to hide
// memory latency (K=4096 was 158 GB/s at U=1 and 264 GB/s at U=2).
//
// @param W [N, K] row-major weights, bf16 or fp16 as ``F16W`` selects.
// @param x [K] bf16 activation. Must be bf16 even when ``F16W``: only the weight
//          dtype changes.
// @param y [N] bf16 output, written by lane 0 of each warp's row.
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
    // K not a multiple of 8: the leftover elements have no vector to live in.
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

// One warp per weight row, M accumulators per lane: the row is read once and
// applied to M input vectors, so a small-M batch costs ~one weight pass.
//
// @param X [M, K] bf16, one activation vector per row of the batch.
// @param Y [M, N] bf16; lane 0 of the warp writes column ``warp`` of all M rows.
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

// Both weight dtypes share one launch path; only the unpack inside the kernel
// differs (see ``w_to_float``). Every shipped (R, U) pair is an explicit
// instantiation, so a new pair has to be added here and to the Python benchmark
// list at the same time.
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
    // The output carries the *activation* dtype: the weights are fp16 for the
    // prefill GEMM, but the decode activation stream stays bf16.
    auto y = torch::empty({N}, x.options());
    const int threads = 256, warps_per_block = threads / 32;
    const int rows_per_block = (int)R * warps_per_block;
    const int blocks = (N + rows_per_block - 1) / rows_per_block;
    const auto* wp = reinterpret_cast<const uint16_t*>(W.data_ptr());
    const auto* xp = reinterpret_cast<const uint16_t*>(x.data_ptr());
    auto* yp = reinterpret_cast<uint16_t*>(y.data_ptr());
    // Must be the current stream: the legacy default stream is not part of a
    // torch.cuda.graph capture, so a kernel launched there is silently dropped
    // from the graph and replay produces garbage.
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

// @brief Single-token bf16 GEMV: y[n] = dot(W[n, :], x).
//
// @param W [N, K] bf16 row-major.
// @param x [K] bf16; contiguous, like W.
// @param R Output rows per warp (1, 2 or 4); chosen by the load-time benchmark.
// @param U K-loop unroll (1, 2 or 4); same choice, and it only affects latency
//          hiding, not the arithmetic.
// @return [N] bf16.
// @throws c10::Error (TORCH_CHECK) on a non-contiguous input, a weight dtype that
//         is not bf16, or an (R, U) pair with no instantiated kernel.
torch::Tensor gemv_bf16(torch::Tensor W, torch::Tensor x, int64_t R, int64_t U) {
    return gemv_entry<false>(W, x, R, U);
}

// @brief Same as ``gemv_bf16`` with fp16 weights; the prefill GEMM reads the same
//        tensors, so the weights are stored fp16 to keep that path fast.
//
// @param W [N, K] fp16 row-major.
// @return [N] bf16: the activation stream stays bf16 even though the weights do
//         not.
// @throws c10::Error (TORCH_CHECK) on a dtype or (R, U) mismatch.
torch::Tensor gemv_f16(torch::Tensor W, torch::Tensor x, int64_t R, int64_t U) {
    return gemv_entry<true>(W, x, R, U);
}

// @brief Batched small-M bf16 GEMV: one weight pass serves all M activations.
//
// @param X [M, K] bf16; M must be 2, 4 or 8, the instantiated values.
// @param U K-loop unroll.
// @return [M, N] bf16.
// @throws c10::Error (TORCH_CHECK) when M/U has no instantiated kernel or the
//         shapes do not line up; the caller picks M from the verify batch size.
torch::Tensor gemv_bf16_m(torch::Tensor W, torch::Tensor X, int64_t U) {
    return gemv_m_entry<false>(W, X, U);
}

// @brief Batched small-M GEMV with fp16 weights.
//
// @param X [M, K] bf16; M must be 2, 4 or 8.
// @return [M, N] bf16.
// @throws c10::Error (TORCH_CHECK) when M/U has no instantiated kernel.
torch::Tensor gemv_f16_m(torch::Tensor W, torch::Tensor X, int64_t U) {
    return gemv_m_entry<true>(W, X, U);
}
