// @file norm.cu
// @brief Fused RMSNorm and gated RMSNorm for the CoreX backend.
//
// Module: coffin/backends/corex/kernels. Bound as ``rms_norm`` /
// ``rms_norm_gated`` and called only from ``backends/corex/__init__.py``.
//
// Why fused rather than the decomposition: the CoreX torch build is 2.1, which has
// no ``F.rms_norm`` at all, and the eager equivalent is a reduce plus separate
// rsqrt/multiply kernels. The model calls a norm ~105 times per decode step, so one
// block per row that sums, scales and multiplies in a single pass is what keeps the
// norm off the launch list.
//
// Constraints:
//   * The reduction is a shared-memory tree, not a ``__shfl_down_sync`` butterfly.
//     This part's warp is 64 lanes wide, and a shuffle reduction written for 32
//     would sum half the block and return a plausible but wrong scale -- the exact
//     silent failure COREX_PORT.md §0 records. A shared tree is width-agnostic, and
//     the temporary is sized from ``blockDim`` at launch, so the block size stays a
//     free parameter.
//   * ``x`` rows are pitched: ``q_norm``/``k_norm`` normalise a slice of a wider
//     buffer, so the pitch is a parameter and is never assumed equal to ``N``.
//   * bf16 in and out, fp32 for the sum and the rsqrt. Summing squares in bf16 would
//     drop the small-magnitude terms, which is the whole difference between this and
//     the reference.
//   * One block per row, so the row width must fit the block limit; rows wider than
//     the block are walked with a strided loop rather than by growing the block.
//   * The gain already carries the checkpoint's ``1 + w`` offset -- the caller folds
//     it at load. Adding it here would apply it twice and shift every logit.
//   * The gated variant normalises over the last axis only, i.e. per head. It is not
//     the flattened width: the gate scales the already-normalised result, so letting
//     it into the reduction would divide every head by the RMS of all of them.

#include "corex_ops.h"
#include "corex_scalar.h"

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

namespace {

//: Wavefront width of this part. Used only to round the block size up; the
//: reduction itself does not depend on it (see the file header).
constexpr int WAVEFRONT = 64;

//: Widest block this file launches. A row of 2560 needs 2560 lanes of work but not
//: 2560 threads: the strided loop covers the rest with better occupancy.
constexpr int MAX_THREADS = 1024;

// @brief Sum ``v`` across the block through a shared tree.
//
// @param v This thread's contribution.
// @param red Caller-provided shared scratch of ``blockDim.x`` floats; clobbered.
// @return The block-wide sum, valid in every thread after the final sync.
//
// The closing barrier is required. Every thread reads the result out of ``red[0]``, and
// nothing stops another thread from starting the next piece of shared work -- which writes
// ``red`` -- before that read lands. Each caller in this file reduces only once, so the
// barrier is not load-bearing here today, but the same helper shape in ``delta.cu`` is
// called twice per step and produced a 4.5e-02 error without it. Keeping one shape across
// every reduction in this directory is what stops that bug from being reintroduced by a
// second caller.
__device__ __forceinline__ float block_sum(float v, float* red) {
    red[threadIdx.x] = v;
    __syncthreads();
    for (int step = blockDim.x >> 1; step > 0; step >>= 1) {
        if ((int)threadIdx.x < step) red[threadIdx.x] += red[threadIdx.x + step];
        __syncthreads();
    }
    const float total = red[0];
    __syncthreads();
    return total;
}

// @brief Threads for a row of width ``N``, rounded up to the wavefront width.
//
// Rounding to 64 matters on this part: a block whose size is not a wavefront
// multiple leaves the last wavefront partly idle, and every norm in the model sits
// on the decode critical path.
//
// @param N Row width.
// @return ``min(MAX_THREADS, round_up(N, WAVEFRONT))``, never below one wavefront.
__host__ int rms_threads(int N) {
    int t = N < WAVEFRONT ? WAVEFRONT : N;
    t = ((t + WAVEFRONT - 1) / WAVEFRONT) * WAVEFRONT;
    return t > MAX_THREADS ? MAX_THREADS : t;
}

// @brief ``y = x * rsqrt(mean(x^2) + eps) * w``, one block per row.
template <typename T>
__global__ void rms_norm_kernel(const T* __restrict__ x, const T* __restrict__ w,
                                T* __restrict__ y, int N, long stride, float eps) {
    extern __shared__ float red[];
    const int row = blockIdx.x;
    const int t = threadIdx.x;
    const T* xr = x + (size_t)row * stride;
    float ss = 0.f;
    for (int i = t; i < N; i += blockDim.x) {
        const float v = CxScalar<T>::to_float(xr[i]);
        ss += v * v;
    }
    const float inv = rsqrtf(block_sum(ss, red) / (float)N + eps);
    T* yr = y + (size_t)row * N;
    for (int i = t; i < N; i += blockDim.x) {
        yr[i] = CxScalar<T>::from_float(CxScalar<T>::to_float(xr[i]) * inv
                                        * CxScalar<T>::to_float(w[i]));
    }
}

// @brief ``y = rms(x) * w * silu(gate)``, one block per row.
template <typename T>
__global__ void rms_norm_gated_kernel(const T* __restrict__ x, const T* __restrict__ w,
                                      const T* __restrict__ gate, T* __restrict__ y,
                                      int N, float eps) {
    extern __shared__ float red[];
    const int row = blockIdx.x;
    const int t = threadIdx.x;
    const T* xr = x + (size_t)row * N;
    float ss = 0.f;
    for (int i = t; i < N; i += blockDim.x) {
        const float v = CxScalar<T>::to_float(xr[i]);
        ss += v * v;
    }
    const float inv = rsqrtf(block_sum(ss, red) / (float)N + eps);
    T* yr = y + (size_t)row * N;
    const T* gr = gate + (size_t)row * N;
    for (int i = t; i < N; i += blockDim.x) {
        const float gv = CxScalar<T>::to_float(gr[i]);
        const float silu = gv / (1.f + __expf(-gv));
        yr[i] = CxScalar<T>::from_float(
            CxScalar<T>::to_float(xr[i]) * inv * CxScalar<T>::to_float(w[i]) * silu);
    }
}

}  // namespace

// @brief Fused RMSNorm over pitched rows.
//
// @param x [M, N] activations, contiguous, bf16 (as uint16) or fp32; the row pitch
//        is ``stride`` and may exceed ``N`` so a column slice normalises in place.
// @param w [N] gain, already carrying the checkpoint's ``1 + w`` offset.
// @param stride Element pitch between rows; ``0`` means ``N``.
// @param eps Variance floor, added before the rsqrt.
// @return [M, N], densely packed, in ``x``'s dtype.
// @throws c10::Error when ``x`` is not 2-D and contiguous, ``w`` is not ``N``
//         elements long, the dtypes differ, or ``stride < N`` (rows would overlap).
torch::Tensor rms_norm(torch::Tensor x, torch::Tensor w, int64_t stride, double eps) {
    TORCH_CHECK(x.dim() == 2, "activation must be 2-D [M, N]");
    // Not ``is_contiguous()``: the pitched case is exactly a column slice of a wider
    // buffer, which is never contiguous. What the kernel indexes by is the row pitch,
    // so unit innermost stride is the real requirement.
    TORCH_CHECK(x.stride(-1) == 1, "activation must have unit innermost stride");
    const int M = (int)x.size(0), N = (int)x.size(1);
    const long pitch = stride > 0 ? (long)stride : (long)N;
    TORCH_CHECK(pitch >= N, "row pitch ", pitch, " is narrower than the row ", N);
    TORCH_CHECK(w.numel() == N, "gain must have N elements, got ", w.numel());
    TORCH_CHECK(w.scalar_type() == x.scalar_type(), "gain dtype must match activation");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 || x.scalar_type() == torch::kFloat,
                "activation must be bf16 or fp32");
    auto y = torch::empty({M, N}, x.options());
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const int threads = rms_threads(N);
    const size_t shmem = (size_t)threads * sizeof(float);
    if (x.scalar_type() == torch::kBFloat16) {
        rms_norm_kernel<uint16_t><<<M, threads, shmem, stream>>>(
            reinterpret_cast<const uint16_t*>(x.data_ptr()),
            reinterpret_cast<const uint16_t*>(w.data_ptr()),
            reinterpret_cast<uint16_t*>(y.data_ptr()), N, pitch, (float)eps);
    } else {
        rms_norm_kernel<float><<<M, threads, shmem, stream>>>(
            x.data_ptr<float>(), w.data_ptr<float>(), y.data_ptr<float>(), N, pitch,
            (float)eps);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

// @brief Fused gated RMSNorm: ``y = rms(x) * w * silu(gate)``, normalised per head.
//
// @param x [M, N] activations, contiguous, bf16 or fp32.
// @param w [N] per-head gain.
// @param gate [M, N] gate in the same layout and dtype as ``x``.
// @param eps Variance floor.
// @return [M, N] in ``x``'s dtype.
// @throws c10::Error when the shapes disagree or a dtype is unsupported.
torch::Tensor rms_norm_gated(torch::Tensor x, torch::Tensor w, torch::Tensor gate,
                             double eps) {
    TORCH_CHECK(x.dim() == 2, "activation must be 2-D [M, N]");
    // The gated form has no pitch argument, so both operands are read densely and
    // contiguity is what that requires.
    TORCH_CHECK(x.is_contiguous() && gate.is_contiguous(),
                "activation and gate must be contiguous");
    TORCH_CHECK(x.sizes() == gate.sizes(), "gate shape must match the activation");
    const int M = (int)x.size(0), N = (int)x.size(1);
    TORCH_CHECK(w.numel() == N, "gain must have N elements, got ", w.numel());
    TORCH_CHECK(w.scalar_type() == x.scalar_type() && gate.scalar_type() == x.scalar_type(),
                "gain and gate dtypes must match the activation");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 || x.scalar_type() == torch::kFloat,
                "activation must be bf16 or fp32");
    auto y = torch::empty({M, N}, x.options());
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const int threads = rms_threads(N);
    const size_t shmem = (size_t)threads * sizeof(float);
    if (x.scalar_type() == torch::kBFloat16) {
        rms_norm_gated_kernel<uint16_t><<<M, threads, shmem, stream>>>(
            reinterpret_cast<const uint16_t*>(x.data_ptr()),
            reinterpret_cast<const uint16_t*>(w.data_ptr()),
            reinterpret_cast<const uint16_t*>(gate.data_ptr()),
            reinterpret_cast<uint16_t*>(y.data_ptr()), N, (float)eps);
    } else {
        rms_norm_gated_kernel<float><<<M, threads, shmem, stream>>>(
            x.data_ptr<float>(), w.data_ptr<float>(), gate.data_ptr<float>(),
            y.data_ptr<float>(), N, (float)eps);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
