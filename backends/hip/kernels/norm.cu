// @file norm.cu
// @brief Fused RMSNorm and gated RMSNorm.
//
// Module: coffin/kernels. Layer-norm entry points bound by ``hip_ops.py``; called
// from the model blocks, never from ``ops/``.
//
// Why fused: ``F.rms_norm`` on this stack decomposes into a reduce (mean of
// squares) plus separate rsqrt / mul kernels, and the engine calls it 105 times
// per decode step (in_ln, post_ln, q_norm, k_norm, the final norm and the gated
// linear-attn norm), which is ~2.6 ms of the step. One block per row fuses the
// whole thing: sum of squares, rsqrt and the weight multiply in a single pass.
//
// Constraints:
//   * A row is one block, so the norm width must fit the block limit; the thread
//     count is chosen from N by ``rms_threads`` rather than fixed.
//   * ``x`` rows are strided: q_norm runs on a ``[nh, hd]`` slice of a wider
//     buffer, so the row pitch is a parameter and must not be assumed equal to N.
//   * bf16 in and out, with the mean and rsqrt accumulated in fp32; computing the
//     sum in bf16 would lose the small-magnitude terms.
//   * The reduction is a warp shuffle plus one cross-warp combine (two
//     ``__syncthreads()``), not a shared tree: these kernels often run as a single
//     block, where the sync chain was the dominant cost.

#include "hip_ops.h"

#include <c10/cuda/CUDAStream.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace {

__device__ __forceinline__ float bf16f(unsigned short h) {
    return __uint_as_float(((unsigned)h) << 16);
}

__device__ __forceinline__ unsigned short fbf16(float f) {
    unsigned u = __float_as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);   // round to nearest even
    return (unsigned short)(u >> 16);
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down(v, o);
    return v;
}

// Warp reduction then one combine across warps: two __syncthreads() instead of
// the eight a 256-wide shared tree needs.  These kernels often run on a *single*
// block (the hidden-size norms are one row in decode), so the sync chain was the
// dominant cost -- 78 calls at 11 us each is 6% of an int8 decode step.  ``red``
// must hold at least 32 floats and is clobbered.
__device__ __forceinline__ float block_sum(float v, float* red) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarp = (blockDim.x + 31) >> 5;
    v = warp_sum(v);
    if (lane == 0) red[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = lane < nwarp ? red[lane] : 0.f;
        v = warp_sum(v);
        if (lane == 0) red[0] = v;
    }
    __syncthreads();
    return red[0];
}

// Rows are one block wide; 256 threads only pays off at large N, and the gated
// norm is [32, 128] (half the threads idle) while q_norm/k_norm are [nh, 256].
// Rounding up to a warp keeps ``block_sum``'s cross-warp combine well defined.
__host__ int rms_threads(int N) {
    return N < 256 ? ((N + 31) / 32) * 32 : 256;
}

// @brief y = x * rsqrt(mean(x^2) + eps) * w, one block per row.
//
// @param x [M, N] bf16 with row pitch ``stride`` (a slice of a wider buffer).
// @param w [N] bf16 weight.
// @param y [M, N] densely packed bf16 output.
// @param eps Added inside the rsqrt, before the reciprocal square root.
__global__ void rms_norm_kernel(const unsigned short* __restrict__ x,
                                const unsigned short* __restrict__ w,
                                unsigned short* __restrict__ y,
                                int N, long stride, float eps) {
    const int row = blockIdx.x;
    const int t = threadIdx.x;
    const unsigned short* xr = x + (size_t)row * stride;
    float ss = 0.f;
    for (int i = t; i < N; i += blockDim.x) {
        const float v = bf16f(xr[i]);
        ss += v * v;
    }
    __shared__ float red[32];
    const float inv = rsqrtf(block_sum(ss, red) / (float)N + eps);
    unsigned short* yr = y + (size_t)row * N;
    for (int i = t; i < N; i += blockDim.x)
        yr[i] = fbf16(bf16f(xr[i]) * inv * bf16f(w[i]));
}

// @brief y = rms(x) * w * silu(gate), all rows densely packed.
//
// @param x,w,gate [M, N] bf16 contiguous; the SiLU is applied to the gate after
//        the normalised weight multiply, so the gate does not affect the norm.
// @param y [M, N] bf16 output.
__global__ void rms_norm_gated_kernel(const unsigned short* __restrict__ x,
                                      const unsigned short* __restrict__ w,
                                      const unsigned short* __restrict__ gate,
                                      unsigned short* __restrict__ y,
                                      int N, float eps) {
    const int row = blockIdx.x;
    const int t = threadIdx.x;
    const unsigned short* xr = x + (size_t)row * N;
    float ss = 0.f;
    for (int i = t; i < N; i += blockDim.x) {
        const float v = bf16f(xr[i]);
        ss += v * v;
    }
    __shared__ float red[32];
    const float inv = rsqrtf(block_sum(ss, red) / (float)N + eps);
    unsigned short* yr = y + (size_t)row * N;
    const unsigned short* gr = gate + (size_t)row * N;
    for (int i = t; i < N; i += blockDim.x) {
        const float gv = bf16f(gr[i]);
        const float silu = gv / (1.f + __expf(-gv));
        yr[i] = fbf16(bf16f(xr[i]) * inv * bf16f(w[i]) * silu);
    }
}

}  // namespace

// @brief Fused RMSNorm over strided rows: y = x * rsqrt(mean(x^2) + eps) * w.
//
// @param x [M, N] bf16, read-only.
// @param w [N] bf16 weight.
// @param stride Row pitch of ``x``; pass N for a packed tensor and the wider pitch
//        when normalising a slice (q_norm / k_norm).
// @param eps Added before the rsqrt; the checkpoint's epsilon, not a tuning knob.
// @return [M, N] bf16, freshly allocated and densely packed.
torch::Tensor rms_norm(torch::Tensor x, torch::Tensor w, int64_t stride, double eps) {
    const int M = x.size(0), N = x.size(1);
    auto y = torch::empty({M, N}, x.options());
    hipStream_t stream = at::cuda::getCurrentCUDAStream();
    rms_norm_kernel<<<M, rms_threads(N), 0, stream>>>(
        reinterpret_cast<const unsigned short*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const unsigned short*>(w.data_ptr<at::BFloat16>()),
        reinterpret_cast<unsigned short*>(y.data_ptr<at::BFloat16>()),
        N, (long)stride, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

// @brief Gated RMSNorm: y = rms(x) * w * silu(gate).
//
// @param x,w,gate [M, N] bf16, every row densely packed (no stride parameter;
//        the gated norm is only ever called on a contiguous tensor).
// @return [M, N] bf16, freshly allocated.
torch::Tensor rms_norm_gated(torch::Tensor x, torch::Tensor w, torch::Tensor gate,
                             double eps) {
    const int M = x.size(0), N = x.size(1);
    auto y = torch::empty({M, N}, x.options());
    hipStream_t stream = at::cuda::getCurrentCUDAStream();
    rms_norm_gated_kernel<<<M, rms_threads(N), 0, stream>>>(
        reinterpret_cast<const unsigned short*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const unsigned short*>(w.data_ptr<at::BFloat16>()),
        reinterpret_cast<const unsigned short*>(gate.data_ptr<at::BFloat16>()),
        reinterpret_cast<unsigned short*>(y.data_ptr<at::BFloat16>()),
        N, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
