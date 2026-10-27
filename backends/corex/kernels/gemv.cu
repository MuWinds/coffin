// @file gemv.cu
// @brief ``[K, N]`` projection kernels: the decode weight read this card needs.
//
// Module: coffin/backends/corex/kernels. Bound as ``gemv_kn`` / ``gemv_kn_split``
// / ``gemv_kn_m`` and called only from ``backends/corex/__init__.py``.
//
// Why one thread per output column: the weight is ``[K, N]`` row-major, so column
// ``n`` is the strided axis of one row. A thread owning column ``n`` reads
// ``wt[k, n]`` for increasing ``k``, and 64 consecutive lanes (this part's
// wavefront) cover 64 consecutive columns of one row: exactly one coalesced
// 128-byte transaction per k-step. That is the whole reason ``[K, N]`` reaches
// 168-361 GB/s here while the library's ``[N, K]`` form measured 19-30 GB/s
// (COREX_PORT.md §0). The opposite assignment -- one lane per weight row reducing
// along K -- cannot coalesce, and is what the library does.
//
// A thread holding four *adjacent* columns instead (the ``x4`` variant) is also
// worse on this part: it was measured slower than one column per thread, so the
// extra bytes per load are not the limiter here and the loop overhead it saves is
// not the cost. Four-deep K unrolling (``UNROLL``) is what was measured to help.
//
// Constraints:
//   * No cross-lane reduction, so ``warpSize == 64`` does not enter this file.
//     Every lane produces one whole output element by itself.
//   * The k-loop is split into a compile-time unrolled body plus a scalar tail, so
//     any ``K`` is correct and only ``K % UNROLL == 0`` gets the wide path.
//   * Launch on ``c10::cuda::getCurrentCUDAStream()``. A kernel on the legacy
//     default stream is not part of a ``torch.cuda.graph`` capture, so it would be
//     dropped from the decoded graph and replay stale output -- a silent
//     correctness bug, not a slowdown.
//   * Accumulation is fp32 and the output is written in the activation dtype. The
//     reference backend also rounds to the activation dtype, so the two agree to
//     bf16 resolution; keeping the accumulator fp32 is what keeps a long K from
//     losing the small-magnitude terms.
//   * Split-K exists for narrow ``N`` only. It trades a second launch and a
//     partial buffer for occupancy, which is the binding constraint when
//     ``ceil(N / threads)`` is below one wave; on a wide ``N`` it would only add
//     traffic to a shape that is already bandwidth-bound.

#include "corex_ops.h"
#include "corex_scalar.h"

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

namespace {

// @brief Dot product of one output column over a K range, unrolled ``U`` ways.
//
// The ``U`` loads are issued before the first FMA, which is what gives the k-loop
// enough memory-level parallelism to hide the stride-``N`` access pattern.
//
// @param wt [K, N] weights, row-major.
// @param x [K] activations.
// @param n Output column this thread owns.
// @param N Row stride of ``wt``.
// @param k0 First element of the range, inclusive.
// @param k1 Last element of the range, exclusive.
// @return The fp32 accumulated dot product.
template <typename WT, typename XT, int U>
__device__ __forceinline__ float dot_kn(const WT* __restrict__ wt,
                                        const XT* __restrict__ x,
                                        int n, int N, int k0, int k1) {
    float acc = 0.f;
    int k = k0;
    for (; k + U <= k1; k += U) {
#pragma unroll
        for (int u = 0; u < U; ++u) {
            acc += CxScalar<WT>::to_float(wt[(size_t)(k + u) * N + n])
                 * CxScalar<XT>::to_float(x[k + u]);
        }
    }
    for (; k < k1; ++k) {
        acc += CxScalar<WT>::to_float(wt[(size_t)k * N + n])
             * CxScalar<XT>::to_float(x[k]);
    }
    return acc;
}

// @brief One activation row against the whole weight: ``y[n] = dot(wt[:, n], x)``.
template <typename WT, typename XT, int U>
__global__ void gemv_kn_kernel(const WT* __restrict__ wt,
                               const XT* __restrict__ x, XT* __restrict__ y,
                               int N, int K) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    y[n] = CxScalar<XT>::from_float(dot_kn<WT, XT, U>(wt, x, n, N, 0, K));
}

// @brief One K range of the same product, into this split's slice of the partials.
//
// Every split writes all ``N`` columns of its own slot, so the partial buffer needs
// no initialisation and the reduction order is fixed by ``s`` rather than by when a
// block happened to finish.
template <typename WT, typename XT, int U>
__global__ void gemv_kn_split_kernel(const WT* __restrict__ wt,
                                     const XT* __restrict__ x, float* __restrict__ part,
                                     int N, int K, int split, int per) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const int s = blockIdx.y;
    const int k0 = s * per;
    const int k1 = min(K, k0 + per);
    part[(size_t)s * N + n] = dot_kn<WT, XT, U>(wt, x, n, N, k0, k1);
}

// @brief Sum the split partials into the output, one thread per column.
template <typename XT>
__global__ void gemv_reduce_kernel(const float* __restrict__ part,
                                   XT* __restrict__ y, int N, int split) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float acc = 0.f;
    for (int s = 0; s < split; ++s) acc += part[(size_t)s * N + n];
    y[n] = CxScalar<XT>::from_float(acc);
}

// @brief ``M`` activation rows share one weight pass, ``M`` a compile-time width.
//
// ``M`` is a template parameter rather than a runtime loop bound so the compiler
// keeps the ``M`` accumulators in registers instead of spilling them: the point of
// this kernel is that the weight is read once for ``M`` results, and it only pays
// if all ``M`` chains stay live across the k-loop.
template <typename WT, typename XT, int U, int M>
__global__ void gemv_kn_m_kernel(const WT* __restrict__ wt,
                                 const XT* __restrict__ x, XT* __restrict__ y,
                                 int N, int K) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.f;
    int k = 0;
    for (; k + U <= K; k += U) {
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const float w = CxScalar<WT>::to_float(wt[(size_t)(k + u) * N + n]);
#pragma unroll
            for (int m = 0; m < M; ++m) {
                acc[m] += w * CxScalar<XT>::to_float(x[(size_t)m * K + k + u]);
            }
        }
    }
    for (; k < K; ++k) {
        const float w = CxScalar<WT>::to_float(wt[(size_t)k * N + n]);
#pragma unroll
        for (int m = 0; m < M; ++m) {
            acc[m] += w * CxScalar<XT>::to_float(x[(size_t)m * K + k]);
        }
    }
#pragma unroll
    for (int m = 0; m < M; ++m) y[(size_t)m * N + n] = CxScalar<XT>::from_float(acc[m]);
}

// @brief SM count, queried once. Split-K sizes itself from the card, not from a
//        hardcoded 16, so a BI-V150 or a partition with a different SM count still
//        gets the same occupancy target.
int sm_count() {
    static int cached = 0;
    if (cached == 0) {
        int device = 0;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&cached, cudaDevAttrMultiProcessorCount, device);
        if (cached <= 0) cached = 16;
    }
    return cached;
}

// @brief Threads per block: enough to fill a wavefront-aligned transaction on every
//        shape, without leaving a small ``N`` with half a block of idle lanes.
int pick_threads(int64_t N) {
    if (N >= 2048) return 256;
    if (N >= 512) return 128;
    return 64;
}

// @brief Split-K width, or ``1`` to run the single-pass kernel.
//
// Splitting multiplies the block count without touching the weight traffic: each split
// reads a different K range of every column, so the bytes moved are the same and only
// the number of independent loads in flight changes. On a 16-SM part with a 575 GB/s
// copy ceiling, the single pass leaves the card short of memory-level parallelism on
// most shapes, so splitting is the default rather than the exception.
//
// Measured on a BI-V100 with every configuration checked against a host fp64 reference.
// The first table is the useful one -- it is what a *decode step* actually pays, taken
// through ``backends.ops.linear`` so it includes the launch and the partial reduce:
//
//   shape      N       K      split 1      split 4      split 8
//   mlp_up     9216    2560   0.292 ms     0.152 ms     --
//   qkv_gate   8192    2560   0.286 ms     0.160 ms     --
//   mlp_down   2560    9216   1.004 ms     --           0.245 ms
//   o_proj     2560    4096   0.329 ms     --           0.085 ms
//   lm_head    248320  2560   3.478 ms     2.961 ms     --
//
// Two corrections to earlier reasoning are baked into that table, both of which cost a
// job to learn:
//
//   * "Split only when the single-pass grid cannot fill the card" was wrong. mlp_up at
//     9216 columns is 36 blocks on 16 SMs -- already two waves -- and splitting it 4x
//     still gains 1.92x. Occupancy in *blocks* was never the limiter; outstanding loads
//     were.
//   * "Narrow shapes do not benefit" was also wrong, and backwards. mlp_down and o_proj
//     are the shapes where splitting is worth the most (4.1x and 3.9x against the single
//     pass) -- they are the ones with the fewest columns to spread.
//
// What the split must respect is ``K``: each split gets ``K / split`` and a slice that
// is too short loses more to the partial pass than it gains. mlp_down has the longest K
// of any shape here and takes 8, which is the most this will ever return.
int pick_split(int64_t N, int64_t K, int threads) {
    if (K < 1024) return 1;

    // A slice shorter than ~512 stops paying for the second launch.
    const int64_t by_k = K / 512 > 8 ? 8 : K / 512;
    if (by_k < 2) return 1;

    // Wide shapes fill the card with columns, so they need less help than narrow ones.
    // mlp_up/qkv_gate measured best at 4 and lost at 8; everything narrower measured
    // best at 8. The threshold sits where the sweep put the crossover.
    const int64_t want = N >= 4096 ? 4 : 8;
    return (int)(want < by_k ? want : by_k);
}

// @brief Launch the single-pass kernel for one (weight, activation) dtype pair.
//
// U=1 is instantiated as well as U=2/U=4 even though it is never a default: it is the
// control in the shape sweep. Without it, a U=4 measurement cannot distinguish "the
// unroll helps" from "the loop overhead was never the limiter", and the sweep would
// keep a knob it has not shown to matter.
template <typename WT, typename XT>
torch::Tensor launch_plain(const torch::Tensor& wt, const torch::Tensor& x,
                           torch::Tensor& y, int N, int K, int threads, int unroll) {
    const WT* wp = reinterpret_cast<const WT*>(wt.data_ptr());
    const XT* xp = reinterpret_cast<const XT*>(x.data_ptr());
    XT* yp = reinterpret_cast<XT*>(y.data_ptr());
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const int blocks = (int)((N + threads - 1) / threads);
    if (unroll == 1) {
        gemv_kn_kernel<WT, XT, 1><<<blocks, threads, 0, stream>>>(wp, xp, yp, N, K);
    } else if (unroll == 2) {
        gemv_kn_kernel<WT, XT, 2><<<blocks, threads, 0, stream>>>(wp, xp, yp, N, K);
    } else {
        gemv_kn_kernel<WT, XT, 4><<<blocks, threads, 0, stream>>>(wp, xp, yp, N, K);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

// @brief Launch the split-K pair for one dtype pair.
template <typename WT, typename XT>
torch::Tensor launch_split(const torch::Tensor& wt, const torch::Tensor& x,
                           torch::Tensor& y, int N, int K, int threads, int unroll,
                           int split) {
    const WT* wp = reinterpret_cast<const WT*>(wt.data_ptr());
    const XT* xp = reinterpret_cast<const XT*>(x.data_ptr());
    XT* yp = reinterpret_cast<XT*>(y.data_ptr());
    auto part = torch::empty({split, N}, x.options().dtype(torch::kFloat));
    float* pp = part.data_ptr<float>();
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const int blocks = (int)((N + threads - 1) / threads);
    const int per = (K + split - 1) / split;
    const dim3 grid(blocks, split);
    if (unroll == 1) {
        gemv_kn_split_kernel<WT, XT, 1><<<grid, threads, 0, stream>>>(wp, xp, pp, N, K,
                                                                     split, per);
    } else if (unroll == 2) {
        gemv_kn_split_kernel<WT, XT, 2><<<grid, threads, 0, stream>>>(wp, xp, pp, N, K,
                                                                     split, per);
    } else {
        gemv_kn_split_kernel<WT, XT, 4><<<grid, threads, 0, stream>>>(wp, xp, pp, N, K,
                                                                     split, per);
    }
    gemv_reduce_kernel<XT><<<blocks, threads, 0, stream>>>(pp, yp, N, split);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

// @brief Launch the small-M kernel for one dtype pair and one ``M`` width.
template <typename WT, typename XT, int M>
torch::Tensor launch_m(const torch::Tensor& wt, const torch::Tensor& x, torch::Tensor& y,
                       int N, int K, int threads) {
    const WT* wp = reinterpret_cast<const WT*>(wt.data_ptr());
    const XT* xp = reinterpret_cast<const XT*>(x.data_ptr());
    XT* yp = reinterpret_cast<XT*>(y.data_ptr());
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const int blocks = (int)((N + threads - 1) / threads);
    gemv_kn_m_kernel<WT, XT, 4, M><<<blocks, threads, 0, stream>>>(wp, xp, yp, N, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

}  // namespace

// @brief Decode-time projection: one activation vector against a ``[K, N]`` weight.
//
// @param wt [K, N] weight, contiguous, bf16 or fp32. fp16 is rejected rather than
//        reinterpreted: the kernels read a 16-bit weight as a bf16 bit pattern, so an
//        fp16 tensor would produce plausible garbage instead of an error.
// @param x [K] activations; bf16 or fp32. The output takes its dtype.
// @param threads Threads per block, ``0`` to pick from ``N``.
// @param unroll K-loop unroll, ``2`` or ``4``; ``0`` means 4.
// @param split Split-K width, ``0`` to pick from ``N``/``K``; ``1`` forces one pass.
// @return [N] output in ``x``'s dtype.
// @throws c10::Error when the shapes, dtypes or layouts are not what the kernel
//         indexes -- a mismatch there reads past the buffer rather than failing.
torch::Tensor gemv_kn(torch::Tensor wt, torch::Tensor x, int64_t threads,
                      int64_t unroll, int64_t split) {
    TORCH_CHECK(wt.dim() == 2, "weight must be 2-D [K, N]");
    TORCH_CHECK(x.dim() == 1, "activation must be 1-D [K]");
    TORCH_CHECK(wt.is_contiguous() && x.is_contiguous(), "contiguous inputs required");
    const int K = (int)wt.size(0), N = (int)wt.size(1);
    TORCH_CHECK(wt.size(0) == x.size(0), "K mismatch: weight ", wt.size(0),
                " vs activation ", x.size(0));
    const bool w16 = wt.scalar_type() == torch::kBFloat16;
    TORCH_CHECK(w16 || wt.scalar_type() == torch::kFloat,
                "weight must be bf16 or fp32 (pack.to_kn converts; an fp16 weight "
                "read as bf16 would compute silently wrong values)");
    const bool x16 = x.scalar_type() == torch::kBFloat16;
    TORCH_CHECK(x16 || x.scalar_type() == torch::kFloat,
                "activation must be bf16 or fp32");
    auto y = torch::empty({N}, x.options());
    const int th = threads > 0 ? (int)threads : pick_threads(N);
    const int u = unroll > 0 ? (int)unroll : 4;
    const int sp = split > 0 ? (int)split : pick_split(N, K, th);

    if (w16 && x16) return sp > 1
        ? launch_split<uint16_t, uint16_t>(wt, x, y, N, K, th, u, sp)
        : launch_plain<uint16_t, uint16_t>(wt, x, y, N, K, th, u);
    if (w16) return sp > 1
        ? launch_split<uint16_t, float>(wt, x, y, N, K, th, u, sp)
        : launch_plain<uint16_t, float>(wt, x, y, N, K, th, u);
    return sp > 1
        ? launch_split<float, float>(wt, x, y, N, K, th, u, sp)
        : launch_plain<float, float>(wt, x, y, N, K, th, u);
}

// @brief Small-M projection: ``M`` activation rows read the weight once.
//
// ``M`` in 2/4/8 goes to a kernel with that width compiled in; any other ``M`` is
// served one row at a time, which is correct and only reached outside the regimes
// the contract names.
//
// @param wt [K, N] weight, contiguous, bf16 or fp32.
// @param x [M, K] activations, same dtypes as ``gemv_kn`` accepts.
// @param threads Threads per block, ``0`` to pick from ``N``.
// @param unroll K-loop unroll, ``0`` for the kernel default.
// @return [M, N] output in ``x``'s dtype.
// @throws c10::Error on a shape or dtype the kernels cannot index.
torch::Tensor gemv_kn_m(torch::Tensor wt, torch::Tensor x, int64_t threads,
                        int64_t unroll) {
    TORCH_CHECK(wt.dim() == 2, "weight must be 2-D [K, N]");
    TORCH_CHECK(x.dim() == 2, "activations must be 2-D [M, K]");
    TORCH_CHECK(wt.is_contiguous() && x.is_contiguous(), "contiguous inputs required");
    const int K = (int)wt.size(0), N = (int)wt.size(1), M = (int)x.size(0);
    TORCH_CHECK(wt.size(0) == x.size(1), "K mismatch: weight ", wt.size(0),
                " vs activation ", x.size(1));
    TORCH_CHECK(M >= 1 && M <= 8, "small-M kernel handles M in [1, 8], got ", M);
    const bool w16 = wt.scalar_type() == torch::kBFloat16;
    TORCH_CHECK(w16 || wt.scalar_type() == torch::kFloat,
                "weight must be bf16 or fp32 (an fp16 weight read as bf16 would "
                "compute silently wrong values)");
    const bool x16 = x.scalar_type() == torch::kBFloat16;
    TORCH_CHECK(x16 || x.scalar_type() == torch::kFloat,
                "activation must be bf16 or fp32");
    const int th = threads > 0 ? (int)threads : pick_threads(N);
    auto y = torch::empty({M, N}, x.options());
    if (M != 2 && M != 4 && M != 8) {
        // Off-specialisation width: one row per launch keeps the contract whole and
        // costs a launch per row, which no decode path takes.
        for (int m = 0; m < M; ++m) {
            auto row = gemv_kn(wt, x[m].contiguous(), th, unroll > 0 ? unroll : 4, 1);
            y[m].copy_(row);
        }
        return y;
    }
    if (w16 && x16) {
        if (M == 2) return launch_m<uint16_t, uint16_t, 2>(wt, x, y, N, K, th);
        if (M == 4) return launch_m<uint16_t, uint16_t, 4>(wt, x, y, N, K, th);
        return launch_m<uint16_t, uint16_t, 8>(wt, x, y, N, K, th);
    }
    if (w16) {
        if (M == 2) return launch_m<uint16_t, float, 2>(wt, x, y, N, K, th);
        if (M == 4) return launch_m<uint16_t, float, 4>(wt, x, y, N, K, th);
        return launch_m<uint16_t, float, 8>(wt, x, y, N, K, th);
    }
    if (M == 2) return launch_m<float, float, 2>(wt, x, y, N, K, th);
    if (M == 4) return launch_m<float, float, 4>(wt, x, y, N, K, th);
    return launch_m<float, float, 8>(wt, x, y, N, K, th);
}
