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

// The W8A16 family: identical indexing to the three kernels above, with the int8 weight
// and a per-column fp32 scale.
//
// Why a separate kernel rather than the ``int8_t`` instantiation of ``gemv_kn_kernel``:
// the scale. It applies once per output column, after the whole ``K`` range is summed --
// applying it per element would round differently and, worse, invite a caller to fold it
// into the activation, which double-scales the result. Keeping the entry point distinct
// makes that contract impossible to satisfy by accident.
//
// The bytes read per weight drop from two to one, which is the entire point: decode is
// bandwidth-bound, and the projections are the majority of the step.

// @brief One activation row against an int8 ``[K, N]`` weight, one thread per column.
template <typename XT, int U>
__global__ void gemv_kn_i8_kernel(const int8_t* __restrict__ wt,
                                  const XT* __restrict__ x, XT* __restrict__ y,
                                  const float* __restrict__ scale, int N, int K) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    y[n] = CxScalar<XT>::from_float(scale[n] * dot_kn<int8_t, XT, U>(wt, x, n, N, 0, K));
}

// @brief One K range of the same product, into this split's slice of the partials.
template <typename XT, int U>
__global__ void gemv_kn_i8_split_kernel(const int8_t* __restrict__ wt,
                                        const XT* __restrict__ x,
                                        float* __restrict__ part,
                                        int N, int K, int split, int per) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const int s = blockIdx.y;
    const int k0 = s * per;
    const int k1 = min(K, k0 + per);
    part[(size_t)s * N + n] = dot_kn<int8_t, XT, U>(wt, x, n, N, k0, k1);
}

// @brief Sum the partials and apply the column scale, one thread per column.
template <typename XT>
__global__ void gemv_reduce_i8_kernel(const float* __restrict__ part,
                                      XT* __restrict__ y,
                                      const float* __restrict__ scale,
                                      int N, int split) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float acc = 0.f;
    for (int s = 0; s < split; ++s) acc += part[(size_t)s * N + n];
    y[n] = CxScalar<XT>::from_float(scale[n] * acc);
}

// Four output columns per thread, for the int8 path.
//
// The scalar kernel above reads one byte per weight element, so a wavefront's 64 lanes
// issue 64 single-byte loads per k step -- 64 bytes of a 128-byte line, one request each.
// Measured, that capped the int8 kernel at 235-262 GB/s against the bf16 kernel's 314 on
// the same shape, which is why halving the bytes only bought 1.5x instead of 2x.
//
// Reading four adjacent *columns* as one 4-byte load fixes it: columns are contiguous in
// a ``[K, N]`` row, so ``wt[k*N + n .. n+3]`` is one aligned word, a wavefront then covers
// 256 contiguous bytes, and the load count drops by four. The four accumulators are
// independent sums, so the arithmetic is unchanged -- only the scale application moves to
// four columns at the end.
//
// This requires ``N % 4 == 0``; the host checks it and falls back to the scalar kernel
// otherwise, which keeps the contract whole for a model whose width is not a multiple of
// four.
template <typename XT, int U>
__global__ void gemv_kn_i8x4_kernel(const int8_t* __restrict__ wt,
                                    const XT* __restrict__ x, XT* __restrict__ y,
                                    const float* __restrict__ scale, int N, int K) {
    const int n0 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (n0 >= N) return;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    int k = 0;
    for (; k + U <= K; k += U) {
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int32_t word =
                *reinterpret_cast<const int32_t*>(&wt[(size_t)(k + u) * N + n0]);
            const float xv = CxScalar<XT>::to_float(x[k + u]);
            a0 = fmaf((float)(int8_t)(word & 0xff), xv, a0);
            a1 = fmaf((float)(int8_t)((word >> 8) & 0xff), xv, a1);
            a2 = fmaf((float)(int8_t)((word >> 16) & 0xff), xv, a2);
            a3 = fmaf((float)(int8_t)((word >> 24) & 0xff), xv, a3);
        }
    }
    for (; k < K; ++k) {
        const int32_t word = *reinterpret_cast<const int32_t*>(&wt[(size_t)k * N + n0]);
        const float xv = CxScalar<XT>::to_float(x[k]);
        a0 = fmaf((float)(int8_t)(word & 0xff), xv, a0);
        a1 = fmaf((float)(int8_t)((word >> 8) & 0xff), xv, a1);
        a2 = fmaf((float)(int8_t)((word >> 16) & 0xff), xv, a2);
        a3 = fmaf((float)(int8_t)((word >> 24) & 0xff), xv, a3);
    }
    y[n0] = CxScalar<XT>::from_float(scale[n0] * a0);
    y[n0 + 1] = CxScalar<XT>::from_float(scale[n0 + 1] * a1);
    y[n0 + 2] = CxScalar<XT>::from_float(scale[n0 + 2] * a2);
    y[n0 + 3] = CxScalar<XT>::from_float(scale[n0 + 3] * a3);
}

// @brief One K range of the same four-wide product, into this split's partials.
template <typename XT, int U>
__global__ void gemv_kn_i8x4_split_kernel(const int8_t* __restrict__ wt,
                                          const XT* __restrict__ x,
                                          float* __restrict__ part,
                                          int N, int K, int split, int per) {
    const int n0 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (n0 >= N) return;
    const int s = blockIdx.y;
    const int k0 = s * per;
    const int k1 = min(K, k0 + per);
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    int k = k0;
    for (; k + U <= k1; k += U) {
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int32_t word =
                *reinterpret_cast<const int32_t*>(&wt[(size_t)(k + u) * N + n0]);
            const float xv = CxScalar<XT>::to_float(x[k + u]);
            a0 = fmaf((float)(int8_t)(word & 0xff), xv, a0);
            a1 = fmaf((float)(int8_t)((word >> 8) & 0xff), xv, a1);
            a2 = fmaf((float)(int8_t)((word >> 16) & 0xff), xv, a2);
            a3 = fmaf((float)(int8_t)((word >> 24) & 0xff), xv, a3);
        }
    }
    for (; k < k1; ++k) {
        const int32_t word = *reinterpret_cast<const int32_t*>(&wt[(size_t)k * N + n0]);
        const float xv = CxScalar<XT>::to_float(x[k]);
        a0 = fmaf((float)(int8_t)(word & 0xff), xv, a0);
        a1 = fmaf((float)(int8_t)((word >> 8) & 0xff), xv, a1);
        a2 = fmaf((float)(int8_t)((word >> 16) & 0xff), xv, a2);
        a3 = fmaf((float)(int8_t)((word >> 24) & 0xff), xv, a3);
    }
    float* col = part + (size_t)s * N + n0;
    col[0] = a0;
    col[1] = a1;
    col[2] = a2;
    col[3] = a3;
}

// @brief Integer word a ``W``-wide column load is performed as.
//
// ``W`` is the number of adjacent output columns one thread owns, which is also the number
// of weight bytes it reads per k step. Those bytes are contiguous within a ``[K, N]`` row,
// so they are one load as long as the row start is aligned -- which ``N % W == 0`` gives,
// since every row begins ``k * N`` bytes into the buffer.
//
// The width that matters for transaction efficiency is set by this part's 64-lane wavefront
// against a 128-byte line: W=1 puts 64 bytes in flight per wavefront (half a line), W=2 is
// exactly 128, W=4 is 256. Reading wider also divides the thread count by ``W``, and on a
// 16-SM part threads are the scarce resource, so which way that trade goes is a measurement
// rather than a rule -- see ``I8_WIDE_MIN_N`` and the sweep in ``tune_gemv_i8_wide.py``.
//
// ``W = 16`` is two 64-bit loads because there is no 16-byte scalar integer. It is included
// because the template makes it free; the sweep is what decides whether it ever runs.
template <int W>
struct CxWideWord;

// W = 1 is the scalar path, which reads one byte at a time. The batched kernel
// instantiates it too (a narrow projection takes that width), so the type has to
// exist even though the single-row kernels never reach it through this helper.
template <>
struct CxWideWord<1> {
    using type = uint8_t;
    static constexpr int kLoads = 1;
};

template <>
struct CxWideWord<2> {
    using type = uint16_t;
    static constexpr int kLoads = 1;
};

template <>
struct CxWideWord<4> {
    using type = uint32_t;
    static constexpr int kLoads = 1;
};

template <>
struct CxWideWord<8> {
    using type = uint64_t;
    static constexpr int kLoads = 1;
};

template <>
struct CxWideWord<16> {
    using type = uint64_t;
    static constexpr int kLoads = 2;
};

// @brief Read ``W`` adjacent weight bytes starting at ``p``.
//
// The load is the whole point of the wide kernels: one transaction per ``W`` columns instead
// of ``W`` of them. Bytes are extracted by shift, which is why the word has to be unsigned --
// an arithmetic right shift on a signed word would sign-extend the high byte into the next.
//
// @param p First weight byte of this thread's column group; aligned to ``W``.
// @param out Receives the ``W`` signed bytes in ascending column order.
template <int W>
__device__ __forceinline__ void load_i8_row(const int8_t* __restrict__ p,
                                            int8_t (&out)[W]) {
    using LT = typename CxWideWord<W>::type;
    constexpr int kLoads = CxWideWord<W>::kLoads;
    constexpr int kPer = (int)sizeof(LT);
#pragma unroll
    for (int i = 0; i < kLoads; ++i) {
        const LT word = *reinterpret_cast<const LT*>(p + i * kPer);
#pragma unroll
        for (int b = 0; b < kPer; ++b) {
            out[i * kPer + b] = (int8_t)((word >> (8 * b)) & 0xff);
        }
    }
}

// @brief One activation row against an int8 ``[K, N]`` weight, ``W`` adjacent columns per
//        thread.
//
// The ``W`` accumulators are independent sums over the same K range, so the arithmetic is
// the scalar kernel's repeated ``W`` times in the same order -- only the load width and the
// final scale application change. The scale still applies once per column, after that
// column's complete K sum.
//
// @param wt [K, N] int8 weights, row-major, with ``N % W == 0``.
// @param x [K] activations.
// @param y [N] output in the activation dtype.
// @param scale [N] fp32 per-column scale.
// @param N Row stride and column count.
// @param K Reduction length.
template <typename XT, int U, int W>
__global__ void gemv_kn_i8w_kernel(const int8_t* __restrict__ wt,
                                   const XT* __restrict__ x, XT* __restrict__ y,
                                   const float* __restrict__ scale, int N, int K) {
    const int n0 = (blockIdx.x * blockDim.x + threadIdx.x) * W;
    if (n0 >= N) return;
    float acc[W];
#pragma unroll
    for (int i = 0; i < W; ++i) acc[i] = 0.f;
    int k = 0;
    for (; k + U <= K; k += U) {
#pragma unroll
        for (int u = 0; u < U; ++u) {
            int8_t bytes[W];
            load_i8_row<W>(&wt[(size_t)(k + u) * N + n0], bytes);
            const float xv = CxScalar<XT>::to_float(x[k + u]);
#pragma unroll
            for (int i = 0; i < W; ++i) acc[i] = fmaf((float)bytes[i], xv, acc[i]);
        }
    }
    for (; k < K; ++k) {
        int8_t bytes[W];
        load_i8_row<W>(&wt[(size_t)k * N + n0], bytes);
        const float xv = CxScalar<XT>::to_float(x[k]);
#pragma unroll
        for (int i = 0; i < W; ++i) acc[i] = fmaf((float)bytes[i], xv, acc[i]);
    }
#pragma unroll
    for (int i = 0; i < W; ++i) {
        y[n0 + i] = CxScalar<XT>::from_float(scale[n0 + i] * acc[i]);
    }
}

// @brief One K range of the same ``W``-wide product, into this split's partials.
//
// Unscaled on purpose: the split partials carry the raw K sums and ``gemv_reduce_i8_kernel``
// applies the scale once after they are added, so the scale cannot be applied per split.
//
// @param wt [K, N] int8 weights, row-major, with ``N % W == 0``.
// @param x [K] activations.
// @param part [split, N] fp32 partial sums, this split's row.
// @param N Row stride and column count.
// @param K Reduction length.
// @param split Number of K ranges.
// @param per Length of one K range; the last one is clamped to ``K``.
template <typename XT, int U, int W>
__global__ void gemv_kn_i8w_split_kernel(const int8_t* __restrict__ wt,
                                         const XT* __restrict__ x,
                                         float* __restrict__ part,
                                         int N, int K, int split, int per) {
    const int n0 = (blockIdx.x * blockDim.x + threadIdx.x) * W;
    if (n0 >= N) return;
    const int s = blockIdx.y;
    const int k0 = s * per;
    const int k1 = min(K, k0 + per);
    float acc[W];
#pragma unroll
    for (int i = 0; i < W; ++i) acc[i] = 0.f;
    int k = k0;
    for (; k + U <= k1; k += U) {
#pragma unroll
        for (int u = 0; u < U; ++u) {
            int8_t bytes[W];
            load_i8_row<W>(&wt[(size_t)(k + u) * N + n0], bytes);
            const float xv = CxScalar<XT>::to_float(x[k + u]);
#pragma unroll
            for (int i = 0; i < W; ++i) acc[i] = fmaf((float)bytes[i], xv, acc[i]);
        }
    }
    for (; k < k1; ++k) {
        int8_t bytes[W];
        load_i8_row<W>(&wt[(size_t)k * N + n0], bytes);
        const float xv = CxScalar<XT>::to_float(x[k]);
#pragma unroll
        for (int i = 0; i < W; ++i) acc[i] = fmaf((float)bytes[i], xv, acc[i]);
    }
    float* col = part + (size_t)s * N + n0;
#pragma unroll
    for (int i = 0; i < W; ++i) col[i] = acc[i];
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

// The W8A16 small-M kernel family: the bf16 small-M kernel above, with the int8 weight and
// its per-column scale.
//
// It exists because the int8 path had no batched entry at all, and the absence was not visible
// as an error -- ``_gemv_rows`` looped the single-row kernel, which reads the whole weight once
// per row. Measured, an M-row projection then cost M times one row (``M=8/M=1 = 8.19x`` over
// five decode shapes, ``probe_row_cost.py``), so a c-token speculative verify paid c full weight
// passes and could not break even at any draft depth (``PERF_COREX.md`` section 14).
//
// The first version of this kernel kept the scalar one-byte load and ran no split-K, on the
// theory that M rows would supply the memory-level parallelism the single-row kernel gets from
// splitting K. Measured, that was wrong twice over: 28.5 GB/s at M=8 against the 395 GB/s one
// row reaches, and 13.87x the cost of one row rather than the ~1x intended. Both fixes were
// already established elsewhere in this file, so this family borrows them rather than
// inventing a third regime:
//
//   * ``W`` adjacent columns per thread, read as one word (``load_i8_row``). A wavefront
//     issuing 64 single-byte loads covers half a 128-byte line; at W=4 it covers two whole
//     lines. ``I8_WIDE_MIN_N`` and the sweep behind it apply unchanged.
//   * split-K, because the grid is what starves: 9728 columns at 256 threads is 38 blocks on
//     16 SMs, and per-thread work cannot fix a grid that thin. ``pick_split_i8`` decides the
//     width, unchanged.
//
// Correctness is *bit-identity with the single-row entry*, not a tolerance, because batched
// decode must not change a token (``models/qwen35/batch.py``). So this family mirrors the
// single-row dispatch instead of picking its own regime: the same ``W`` from
// ``pick_i8_width``, the same split from ``pick_split_i8``, the same accumulation order inside
// a slice, and the same reduce order across slices. A different split re-associates the fp32
// sum and moves a handful of ULPs -- measured, 1-5 values out of 19456-77824 differed from the
// per-row path when the split was not mirrored, which is exactly the kind of difference that
// flips an argmax once in a long generation.
//
// One asymmetry is deliberate: the accumulation style follows ``W``, because the single-row
// kernels do. ``gemv_kn_i8_kernel`` (scalar) writes ``acc += w * x`` and ``gemv_kn_i8w_kernel``
// (wide) writes ``fmaf(w, x, acc)``. Those round differently in the last bit, so one fixed
// style could not match both; this uses the style of whichever single-row kernel the same shape
// would have taken.

// @brief M activation rows against a ``W``-wide column group of an int8 ``[K, N]`` weight.
//
// @param wt [K, N] int8 weights, row-major, with ``N % W == 0``.
// @param x [M, K] activations, row-major.
// @param y [M, N] output in the activation dtype, row-major.
// @param scale [N] fp32 per-column scale, applied once after each column's full K sum.
// @param N Row stride and column count.
// @param K Reduction length.
template <typename XT, int U, int M, int W>
__global__ void gemv_kn_i8_m_kernel(const int8_t* __restrict__ wt,
                                    const XT* __restrict__ x, XT* __restrict__ y,
                                    const float* __restrict__ scale, int N, int K,
                                    int rows) {
    const int n0 = (blockIdx.x * blockDim.x + threadIdx.x) * W;
    if (n0 >= N) return;
    float acc[M * W];
#pragma unroll
    for (int i = 0; i < M * W; ++i) acc[i] = 0.f;
    int k = 0;
    for (; k + U <= K; k += U) {
#pragma unroll
        for (int u = 0; u < U; ++u) {
            // One weight word feeds M*W accumulators. The load is hoisted out of the row loop
            // only because that loop is template-instantiated and fully unrolled; a runtime
            // loop over rows would re-read the bytes once per row, which is the bug this
            // kernel exists to remove.
            int8_t bytes[W];
            load_i8_row<W>(&wt[(size_t)(k + u) * N + n0], bytes);
#pragma unroll
            for (int m = 0; m < M; ++m) {
                // A row past ``rows`` re-reads row 0 instead of the buffer's real rows: the
                // surplus accumulators are computed and then dropped, so an off-width call
                // never needs a padded input tensor.
                const float xv = CxScalar<XT>::to_float(
                    x[(size_t)(m < rows ? m : 0) * K + k + u]);
#pragma unroll
                for (int i = 0; i < W; ++i) {
                    // W > 1 is the wide single-row kernel, which uses fmaf; W == 1 is the
                    // scalar one, which uses a separate multiply and add. Keeping the two
                    // spellings apart is what makes each bit-identical to its twin.
                    if (W > 1) acc[m * W + i] = fmaf((float)bytes[i], xv, acc[m * W + i]);
                    else acc[m * W + i] += (float)bytes[i] * xv;
                }
            }
        }
    }
    for (; k < K; ++k) {
        int8_t bytes[W];
        load_i8_row<W>(&wt[(size_t)k * N + n0], bytes);
#pragma unroll
        for (int m = 0; m < M; ++m) {
            const float xv = CxScalar<XT>::to_float(x[(size_t)(m < rows ? m : 0) * K + k]);
#pragma unroll
            for (int i = 0; i < W; ++i) {
                if (W > 1) acc[m * W + i] = fmaf((float)bytes[i], xv, acc[m * W + i]);
                else acc[m * W + i] += (float)bytes[i] * xv;
            }
        }
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
        // Only the real rows are stored: the caller's output has exactly ``rows`` rows, so a
        // wider store would run off the end of it.
        if (m >= rows) continue;
#pragma unroll
        for (int i = 0; i < W; ++i) {
            y[(size_t)m * N + n0 + i] =
                CxScalar<XT>::from_float(scale[n0 + i] * acc[m * W + i]);
        }
    }
}

// @brief One K range of the int8 small-M product, into this split's partial plane.
//
// Unscaled on purpose: the partials carry the raw K sums and the reduce applies the scale once
// after they are added, exactly as the single-row split pair does. Scaling here would scale
// each slice separately and change the result.
//
// @param wt [K, N] int8 weights, row-major, with ``N % W == 0``.
// @param x [M, K] activations, row-major.
// @param part [M*split, N] fp32 partial sums; row ``m``'s planes start at ``m * split``.
// @param N Row stride and column count.
// @param K Reduction length.
// @param split Number of K ranges.
// @param per Length of one K range; the last is clamped to K.
template <typename XT, int U, int M, int W>
__global__ void gemv_kn_i8_m_split_kernel(const int8_t* __restrict__ wt,
                                          const XT* __restrict__ x,
                                          float* __restrict__ part,
                                          int N, int K, int split, int per, int rows) {
    const int n0 = (blockIdx.x * blockDim.x + threadIdx.x) * W;
    if (n0 >= N) return;
    const int s = blockIdx.y;
    const int k0 = s * per;
    const int k1 = min(K, k0 + per);
    float acc[M * W];
#pragma unroll
    for (int i = 0; i < M * W; ++i) acc[i] = 0.f;
    int k = k0;
    for (; k + U <= k1; k += U) {
#pragma unroll
        for (int u = 0; u < U; ++u) {
            int8_t bytes[W];
            load_i8_row<W>(&wt[(size_t)(k + u) * N + n0], bytes);
#pragma unroll
            for (int m = 0; m < M; ++m) {
                const float xv = CxScalar<XT>::to_float(
                    x[(size_t)(m < rows ? m : 0) * K + k + u]);
#pragma unroll
                for (int i = 0; i < W; ++i) {
                    if (W > 1) acc[m * W + i] = fmaf((float)bytes[i], xv, acc[m * W + i]);
                    else acc[m * W + i] += (float)bytes[i] * xv;
                }
            }
        }
    }
    for (; k < k1; ++k) {
        int8_t bytes[W];
        load_i8_row<W>(&wt[(size_t)k * N + n0], bytes);
#pragma unroll
        for (int m = 0; m < M; ++m) {
            const float xv = CxScalar<XT>::to_float(x[(size_t)(m < rows ? m : 0) * K + k]);
#pragma unroll
            for (int i = 0; i < W; ++i) {
                if (W > 1) acc[m * W + i] = fmaf((float)bytes[i], xv, acc[m * W + i]);
                else acc[m * W + i] += (float)bytes[i] * xv;
            }
        }
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
        if (m >= rows) continue;
        float* col = part + ((size_t)m * split + s) * N + n0;
#pragma unroll
        for (int i = 0; i < W; ++i) col[i] = acc[m * W + i];
    }
}

// @brief Sum the split planes and apply the column scale, one thread per column.
//
// The reduce order is ``s = 0 .. split-1`` for every row, which is what makes the association
// identical to the single-row reduce and therefore the whole chain bit-identical to looping the
// single-row entry.
//
// @param part [M*split, N] fp32 partial sums, row-major.
// @param y [M, N] output in the activation dtype, written in place.
// @param scale [N] fp32 per-column scale.
// @param N Output columns.
// @param split Number of K ranges.
// @param M Activation rows.
template <typename XT>
__global__ void gemv_i8_m_reduce_kernel(const float* __restrict__ part,
                                        XT* __restrict__ y,
                                        const float* __restrict__ scale,
                                        int N, int split, int M, int rows) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    for (int m = 0; m < M; ++m) {
        if (m >= rows) continue;
        float acc = 0.f;
        for (int s = 0; s < split; ++s) acc += part[((size_t)m * split + s) * N + n];
        y[(size_t)m * N + n] = CxScalar<XT>::from_float(scale[n] * acc);
    }
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

//: Blocks the int8 split-K rule keeps resident. Every measured optimum below sits at or
//: within an octave of this product, and it is a *block count* rather than a split because
//: the same split means different things at different ``N``: s8 is 970 blocks on ``lm_head``
//: and 10 blocks on ``o_proj``, and only one of those wants to split at all.
constexpr int I8_TARGET_BLOCKS = 1024;

//: Hard ceiling on the int8 split. Past this the partial buffer costs more traffic than the
//: extra blocks recover, and the reduce pass grows with the split.
constexpr int I8_MAX_SPLIT = 32;

//: Shortest K slice worth scheduling. A slice is one strided pass with no reuse; below ~64
//: elements the launch and the partial write dominate what the slice reads. It doubles as
//: the partial-buffer bound: the reduce costs ``8 * split * N`` bytes (an fp32 ``[split, N]``
//: written then read), so ``split <= K / 64`` caps that overhead at ``8 / 64`` = 12.5% of the
//: weight traffic whatever ``K`` and ``N`` are. That is why this one constant is enough to
//: stop a long-K shape from splitting itself into slices that cost more than they save.
constexpr int I8_MIN_SLICE = 64;

// @brief Split-K width for the int8 path, or ``1`` to run the single-pass kernel.
//
// ``pick_split`` above is the bf16 rule and is left alone: its numbers came from a bf16
// sweep and its cap of 8 encodes bf16's trade-off, not int8's. The int8 path reads half
// the bytes per column, so it needs roughly twice the blocks to keep the same number of
// loads in flight -- which is why the same shapes want 16-32 here where bf16 wanted 4-8.
//
// Measured per shape on a BI-V100 (16 SMs, 572 GB/s ceiling) at a fixed ``K = 2560``, with
// every candidate checked against a host fp64 reference off the dequantised weight. The
// middle column is the single-pass block count at 256 threads, which is the quantity the
// rule is a function of:
//
//   N        blocks   best split
//   4096     16       32
//   8192     32       16
//   16384    64       32
//   32768    128      8
//   65536    256      4
//   131072   512      2
//   248320   970      8        (s1 measured within 1%, so this row is a flat basin)
//
// One line fits the plateau from 128 blocks upward exactly -- ``split * blocks ~ 1024``.
// The three smallest grids do not sit on it: 4096 is capped by ``I8_MAX_SPLIT``, while
// 8192 and 16384 land one octave either side of the line, and 248320 is a flat basin
// where s1 measured within 1% of s8. So this is a fit, not a table: it interpolates for
// the next model's ``N``, and the shapes this ships against are not round numbers, so a
// hardcoded table would miss all of them anyway.
//
// Its known bias, stated so nobody rediscovers it as a bug: on the mid-range grids the
// rounded division lands one step under the measured optimum (16384 gets 16 where 32
// measured best), and the real shapes inherit that -- ``N = 18432`` gets 14, ``N = 12352``
// gets 21, ``N = 10240`` gets 26, against measured bests of 16, 32 and 32. Under-splitting
// is the safe direction (it costs block count, never correctness), and the shapes that
// carry most of the step -- the two 10-block ones -- both hit the cap exactly.
//
// The failure this fixes is the old cap of 8: ``o_proj`` (10 blocks) and ``mlp_down``
// (10 blocks) were left at 8 when the measured optimum is 32, and those two shapes alone
// account for most of the 5.9 ms the dispatch change recovers.
//
// @param N Output columns of the weight.
// @param K Reduction length; bounds the split through ``I8_MIN_SLICE``.
// @param threads Threads per block, which sets the single-pass block count.
// @return Split width in ``[1, I8_MAX_SPLIT]``, always a value whose slices partition ``K``.
int pick_split_i8(int64_t N, int64_t K, int threads) {
    const int64_t blocks = threads > 0 ? (N + threads - 1) / threads : N;
    if (blocks <= 0) return 1;

    // Rounded division, so the rule lands on the nearest block count rather than truncating
    // toward a wider split on every shape.
    int64_t want = (I8_TARGET_BLOCKS + blocks / 2) / blocks;
    if (want > I8_MAX_SPLIT) want = I8_MAX_SPLIT;

    // Each split has to leave a slice long enough to schedule; K/64 is that bound, and it
    // is what stops a short-K shape from being split into slices the launch cannot hide.
    const int64_t by_k = K / I8_MIN_SLICE;
    if (by_k < want) want = by_k;
    return (int)(want < 1 ? 1 : want);
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
    const int per = (K + split - 1) / split;
    const int blocks = (int)((N + threads - 1) / threads);
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

//: Column count at or above which the ``width == 0`` ("auto") path picks the four-wide
//: kernel. This is the *default dispatch* only, not a correctness floor: the wide kernels
//: are reachable at any ``N`` through the ``width`` argument, which is what the sweep drives.
//:
//: 4096 is where a width sweep at split 16/32 put the crossover, and the split matters:
//: the old floor of 32768 was measured with ``split`` capped at 8, which starved the grid,
//: so the wide kernel was blamed for thread counts that split-K now restores. Re-measured
//: on the shapes a decode step actually calls (``tune_gemv_i8_wide.py``, 16 SMs, 571 GB/s
//: ceiling, every cell against a host fp64 reference):
//:
//:   N       K      width 1     width 4     four-wide wins by
//:   18432   2560   234 GB/s    375 GB/s    1.60x
//:   12352   2560   369 GB/s    462 GB/s    1.25x
//:   10240   2560   366 GB/s    452 GB/s    1.23x
//:   248320  2560   452 GB/s    461 GB/s    1.02x
//:   2560    9216   337 GB/s    283 GB/s    -- four-wide LOSES (0.84x)
//:   2560    4096   238 GB/s    215 GB/s    -- four-wide LOSES (0.90x)
//:   2560    5120   269 GB/s    234 GB/s    -- four-wide LOSES (0.87x)
//:
//: The split is clean and it is not about bytes: every shape above reads the same one byte
//: per element. It is about how many threads the model leaves. A 2560-column projection is
//: 10 blocks at 256 threads, and the four-wide kernel divides that by four to 2.5 blocks --
//: under one wave on 16 SMs. Above ~4096 columns the grid survives the division and the
//: wider load wins. ``width = 1`` for the narrow shapes is therefore a measured choice, not
//: an omission.
constexpr int I8_WIDE_MIN_N = 4096;

//: Largest ``width`` the int8 kernels are instantiated for. The host wrapper rejects
//: anything above it, so a sweep cannot ask for a width the build has no kernel for.
constexpr int I8_MAX_WIDTH = 16;

// @brief Resolve the ``width`` argument into a compile-time dispatch choice.
//
// ``0`` keeps the shipped rule (four-wide at or above :data:`I8_WIDE_MIN_N`, scalar below)
// so the default path is unchanged by construction. An explicit width is honoured when it
// divides ``N`` and falls back to the scalar kernel otherwise: a model whose width is not a
// multiple of ``W`` must stay correct rather than read across a row boundary, and reading
// the neighbouring row's bytes would still produce plausible numbers.
//
// A width the build does not instantiate is also folded to the scalar kernel here. The
// wrapper TORCH_CHECKs it first, so this is the second line of defence rather than the
// error the caller sees -- but it keeps the arithmetic total for any ``int``, which is what
// lets the fallback table be tested without a device.
//
// @param N Output columns.
// @param width Requested columns per thread, ``0`` for the shipped rule.
// @return ``1``, ``2``, ``4``, ``8`` or ``16``, and always a divisor of ``N`` when above 1.
inline int pick_i8_width(int N, int width) {
    int w = width;
    if (w <= 0) w = (N % 4 == 0 && N >= I8_WIDE_MIN_N) ? 4 : 1;
    // Only the instantiated widths: each one is read as a single unsigned word of exactly
    // that many bytes, and a value with no such word type has no kernel behind it.
    if (w != 1 && w != 2 && w != 4 && w != 8 && w != I8_MAX_WIDTH) return 1;
    if (w > 1 && (N % w) != 0) return 1;
    return w;
}

// @brief Launch one single-pass int8 kernel with ``W`` and ``U`` both compile-time.
template <typename XT, int W, int U>
void launch_i8_plain_one(const int8_t* wp, const XT* xp, XT* yp, const float* sp,
                         int N, int K, int threads) {
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const int blocks = (int)(((N / W) + threads - 1) / threads);
    gemv_kn_i8w_kernel<XT, U, W><<<blocks, threads, 0, stream>>>(wp, xp, yp, sp, N, K);
}

// @brief Launch one ``W``-wide single-pass kernel at the requested unroll.
template <typename XT, int W>
void launch_i8_plain_w(const int8_t* wp, const XT* xp, XT* yp, const float* sp,
                       int N, int K, int threads, int unroll) {
    if (unroll == 2) {
        launch_i8_plain_one<XT, W, 2>(wp, xp, yp, sp, N, K, threads);
    } else {
        launch_i8_plain_one<XT, W, 4>(wp, xp, yp, sp, N, K, threads);
    }
}

// @brief Launch one split-K partial kernel with ``W`` and ``U`` both compile-time.
template <typename XT, int W, int U>
void launch_i8_split_one(const int8_t* wp, const XT* xp, float* pp, int N, int K,
                         int threads, int split, int per) {
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const int blocks = (int)(((N / W) + threads - 1) / threads);
    gemv_kn_i8w_split_kernel<XT, U, W><<<dim3(blocks, split), threads, 0, stream>>>(
        wp, xp, pp, N, K, split, per);
}

// @brief Launch one ``W``-wide split-K partial kernel at the requested unroll.
template <typename XT, int W>
void launch_i8_split_w(const int8_t* wp, const XT* xp, float* pp, int N, int K,
                       int threads, int split, int per, int unroll) {
    if (unroll == 2) {
        launch_i8_split_one<XT, W, 2>(wp, xp, pp, N, K, threads, split, per);
    } else {
        launch_i8_split_one<XT, W, 4>(wp, xp, pp, N, K, threads, split, per);
    }
}

// @brief Launch the single-pass int8 kernel for one activation dtype, width and unroll.
//
// ``width == 4`` stays on :func:`gemv_kn_i8x4_kernel` rather than the generic width-4
// instantiation: that kernel is what the 34.23 -> 46.81 ms measurement was taken against,
// and keeping it in the path is what makes that negative result reproducible.
template <typename XT>
torch::Tensor launch_i8_plain(const torch::Tensor& wt, const torch::Tensor& x,
                              torch::Tensor& y, const torch::Tensor& scale,
                              int N, int K, int threads, int unroll, int width) {
    const int8_t* wp = reinterpret_cast<const int8_t*>(wt.data_ptr());
    const XT* xp = reinterpret_cast<const XT*>(x.data_ptr());
    XT* yp = reinterpret_cast<XT*>(y.data_ptr());
    const float* sp = scale.data_ptr<float>();
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const int w = pick_i8_width(N, width);
    if (w == 4) {
        const int blocks = (int)(((N / 4) + threads - 1) / threads);
        if (unroll == 2) {
            gemv_kn_i8x4_kernel<XT, 2><<<blocks, threads, 0, stream>>>(wp, xp, yp, sp, N, K);
        } else {
            gemv_kn_i8x4_kernel<XT, 4><<<blocks, threads, 0, stream>>>(wp, xp, yp, sp, N, K);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return y;
    }
    if (w == 2) {
        launch_i8_plain_w<XT, 2>(wp, xp, yp, sp, N, K, threads, unroll);
    } else if (w == 8) {
        launch_i8_plain_w<XT, 8>(wp, xp, yp, sp, N, K, threads, unroll);
    } else if (w == 16) {
        launch_i8_plain_w<XT, 16>(wp, xp, yp, sp, N, K, threads, unroll);
    } else {
        const int blocks = (int)((N + threads - 1) / threads);
        if (unroll == 2) {
            gemv_kn_i8_kernel<XT, 2><<<blocks, threads, 0, stream>>>(wp, xp, yp, sp, N, K);
        } else {
            gemv_kn_i8_kernel<XT, 4><<<blocks, threads, 0, stream>>>(wp, xp, yp, sp, N, K);
        }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

// @brief Launch the split-K int8 pair for one activation dtype, width and unroll.
template <typename XT>
torch::Tensor launch_i8_split(const torch::Tensor& wt, const torch::Tensor& x,
                              torch::Tensor& y, const torch::Tensor& scale,
                              int N, int K, int threads, int unroll, int split,
                              int width) {
    const int8_t* wp = reinterpret_cast<const int8_t*>(wt.data_ptr());
    const XT* xp = reinterpret_cast<const XT*>(x.data_ptr());
    XT* yp = reinterpret_cast<XT*>(y.data_ptr());
    const float* sp = scale.data_ptr<float>();
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const int per = (K + split - 1) / split;
    // The partials are scratch, sized once for all splits: the reduce needs every split
    // written before any column can be scaled, so they cannot share the output buffer.
    auto part = torch::empty({(int64_t)split, (int64_t)N}, x.options().dtype(torch::kFloat));
    float* pp = part.data_ptr<float>();
    const int w = pick_i8_width(N, width);
    if (w == 4) {
        const int blocks = (int)(((N / 4) + threads - 1) / threads);
        if (unroll == 2) {
            gemv_kn_i8x4_split_kernel<XT, 2><<<dim3(blocks, split), threads, 0, stream>>>(
                wp, xp, pp, N, K, split, per);
        } else {
            gemv_kn_i8x4_split_kernel<XT, 4><<<dim3(blocks, split), threads, 0, stream>>>(
                wp, xp, pp, N, K, split, per);
        }
    } else if (w == 2) {
        launch_i8_split_w<XT, 2>(wp, xp, pp, N, K, threads, split, per, unroll);
    } else if (w == 8) {
        launch_i8_split_w<XT, 8>(wp, xp, pp, N, K, threads, split, per, unroll);
    } else if (w == 16) {
        launch_i8_split_w<XT, 16>(wp, xp, pp, N, K, threads, split, per, unroll);
    } else {
        const int blocks = (int)((N + threads - 1) / threads);
        if (unroll == 2) {
            gemv_kn_i8_split_kernel<XT, 2><<<dim3(blocks, split), threads, 0, stream>>>(
                wp, xp, pp, N, K, split, per);
        } else {
            gemv_kn_i8_split_kernel<XT, 4><<<dim3(blocks, split), threads, 0, stream>>>(
                wp, xp, pp, N, K, split, per);
        }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    // The reduce is one thread per column whatever the weight width was: it reads ``split``
    // floats per column, so its cost is set by the split, not by the weight load width.
    const int rblocks = (int)((N + threads - 1) / threads);
    gemv_reduce_i8_kernel<XT><<<rblocks, threads, 0, stream>>>(pp, yp, sp, N, split);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

}  // namespace

// @brief Launch the int8 small-M kernel for one dtype, one M width and one W.
//
// @param wp [K, N] int8 weight.
// @param xp [M, K] activations.
// @param sp [N] fp32 per-column scales.
// @param yp [M, N] output.
// @param pp [M*split, N] fp32 partial buffer, or null when split is 1.
// @param N Output columns.
// @param K Reduction length.
// @param threads Threads per block.
// @param split Split-K width; 1 takes the single-pass kernel.
// @param per Length of one K range.
template <typename XT, int M, int W>
void launch_i8_m_one(const int8_t* wp, const XT* xp, const float* sp, XT* yp, float* pp,
                     int N, int K, int threads, int split, int per, int rows) {
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    // The grid follows the wide arithmetic: a thread owns W columns, so the block count is
    // N/W over the thread block. Keeping the scalar grid instead would launch W times the
    // blocks and have most of them return immediately.
    const int blocks = (int)(((N / W) + threads - 1) / threads);
    if (split > 1) {
        gemv_kn_i8_m_split_kernel<XT, 4, M, W><<<dim3(blocks, split), threads, 0, stream>>>(
            wp, xp, pp, N, K, split, per, rows);
        // The reduce is one thread per *column*, so it needs the undivided grid. Reusing the
        // W-divided block count leaves every column past N/W unwritten, and those come back as
        // whatever the scratch buffer held -- which the row-identity check reports as a max
        // relative error of order 1, not as a launch error.
        const int rblocks = (int)((N + threads - 1) / threads);
        gemv_i8_m_reduce_kernel<XT><<<rblocks, threads, 0, stream>>>(
            pp, yp, sp, N, split, M, rows);
    } else {
        gemv_kn_i8_m_kernel<XT, 4, M, W><<<blocks, threads, 0, stream>>>(
            wp, xp, yp, sp, N, K, rows);
    }
}

// @brief Launch the int8 small-M kernel at whatever W the sweep rule picks for N.
//
// W and the split both come from the rules the single-row entry uses, never from a separate
// set. That is the bit-identity contract in one place: an M-row call has to take the same
// arithmetic path its rows would have taken one at a time, so both dispatch decisions are
// delegated rather than re-decided here.
//
// @param wt [K, N] int8 weight.
// @param x [M, K] activations.
// @param y Preallocated [M, N] output, returned unchanged for call chaining.
// @param scale [N] fp32 per-column scales.
// @param N Output columns.
// @param K Reduction length.
// @param threads Threads per block.
// @param split Split-K width, as decided by ``pick_split_i8`` for the single-row entry.
template <typename XT, int M>
torch::Tensor launch_i8_m(const torch::Tensor& wt, const torch::Tensor& x, torch::Tensor& y,
                          int rows, const torch::Tensor& scale, int N, int K, int threads,
                          int split) {
    const int8_t* wp = reinterpret_cast<const int8_t*>(wt.data_ptr());
    const XT* xp = reinterpret_cast<const XT*>(x.data_ptr());
    XT* yp = reinterpret_cast<XT*>(y.data_ptr());
    const float* sp = scale.data_ptr<float>();
    const int per = split > 1 ? (K + split - 1) / split : K;
    // Scratch for all rows at once: M planes per split, because each row's K sum reduces
    // independently. A [split, N] buffer would have every row overwrite the previous row's
    // partials, which reads back as a correct shape and a wrong answer.
    torch::Tensor part;
    float* pp = nullptr;
    if (split > 1) {
        part = torch::empty({M * split, N}, x.options().dtype(torch::kFloat));
        pp = part.data_ptr<float>();
    }
    switch (pick_i8_width(N, 0)) {
        case 2: launch_i8_m_one<XT, M, 2>(wp, xp, sp, yp, pp, N, K, threads, split, per, rows); break;
        case 4: launch_i8_m_one<XT, M, 4>(wp, xp, sp, yp, pp, N, K, threads, split, per, rows); break;
        case 8: launch_i8_m_one<XT, M, 8>(wp, xp, sp, yp, pp, N, K, threads, split, per, rows); break;
        default: launch_i8_m_one<XT, M, 1>(wp, xp, sp, yp, pp, N, K, threads, split, per, rows); break;
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

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

// @brief Decode-time W8A16 projection: int8 ``[K, N]`` weight and a per-column fp32 scale.
//
// Halves the bytes read per weight against :func:`gemv_kn`, which is the only lever left
// on a path that is already bandwidth-bound: the projections are the majority of a decode
// step and the card is saturated for essentially all of it (PERF_COREX.md §8).
//
// @param wt [K, N] int8 weight, contiguous.
// @param x [K] activations, bf16 or fp32. The output takes its dtype.
// @param scale [N] fp32 absmax scale, one per output column, applied once per column.
// @param threads Threads per block, ``0`` to pick from ``N``.
// @param unroll K-loop unroll, ``2`` or ``4``; ``0`` means 4.
// @param split Split-K width, ``0`` to pick; ``1`` forces one pass.
// @param width Output columns per thread, ``0`` for the shipped rule. Legal values are
//        ``1``, ``2``, ``4``, ``8`` and ``16``; ``N % width`` must be ``0`` and the kernel
//        falls back to ``1`` when it is not. ``0`` reproduces the pre-existing dispatch
//        exactly, so leaving it unset changes nothing.
// @return [N] output in ``x``'s dtype.
// @throws c10::Error when the weight is not int8, the scale length disagrees with ``N``,
//         an explicit ``width`` is not one of the instantiated values, or a dtype/layout
//         is one the kernels cannot index -- each of those reads the wrong elements rather
//         than failing.
torch::Tensor gemv_kn_i8(torch::Tensor wt, torch::Tensor x, torch::Tensor scale,
                         int64_t threads, int64_t unroll, int64_t split,
                         int64_t width) {
    TORCH_CHECK(wt.dim() == 2, "weight must be 2-D [K, N]");
    TORCH_CHECK(x.dim() == 1, "activation must be 1-D [K]");
    TORCH_CHECK(wt.scalar_type() == torch::kChar,
                "int8 weight required (a bf16 weight here would be read one byte per "
                "value and produce plausible garbage)");
    TORCH_CHECK(scale.scalar_type() == torch::kFloat, "scale must be fp32");
    TORCH_CHECK(wt.is_contiguous() && x.is_contiguous() && scale.is_contiguous(),
                "contiguous inputs required");
    const int K = (int)wt.size(0), N = (int)wt.size(1);
    TORCH_CHECK(wt.size(0) == x.size(0), "K mismatch: weight ", wt.size(0),
                " vs activation ", x.size(0));
    TORCH_CHECK(scale.numel() == N, "scale must hold one value per output column: got ",
                scale.numel(), " for N = ", N);
    // Rejected rather than silently rounded to a legal width: a sweep that asks for a width
    // the build has no kernel for must fail instead of reporting the neighbouring width's
    // timing under the wrong label.
    TORCH_CHECK(width == 0 || width == 1 || width == 2 || width == 4 || width == 8
                    || width == 16,
                "width must be 0 (auto) or one of 1, 2, 4, 8, 16; got ", width);
    const bool x16 = x.scalar_type() == torch::kBFloat16;
    TORCH_CHECK(x16 || x.scalar_type() == torch::kFloat,
                "activation must be bf16 or fp32");
    auto y = torch::empty({N}, x.options());
    const int th = threads > 0 ? (int)threads : pick_threads(N);
    const int u = unroll > 0 ? (int)unroll : 4;
    // int8 only: ``pick_split`` stays the bf16 rule because its cap was measured for bf16.
    const int sp = split > 0 ? (int)split : pick_split_i8(N, K, th);
    const int w = (int)width;
    if (x16) {
        return sp > 1 ? launch_i8_split<uint16_t>(wt, x, y, scale, N, K, th, u, sp, w)
                      : launch_i8_plain<uint16_t>(wt, x, y, scale, N, K, th, u, w);
    }
    return sp > 1 ? launch_i8_split<float>(wt, x, y, scale, N, K, th, u, sp, w)
                  : launch_i8_plain<float>(wt, x, y, scale, N, K, th, u, w);
}

// @brief Batched W8A16 projection: M activation rows share one weight pass.
//
// The entry ``_gemv_rows`` was missing: with no batched int8 kernel it looped the single-row
// GEMV once per row, so an M-row projection cost M weight reads. On this part that is exactly
// M times one row (``M=8/M=1 = 8.19x``, five decode shapes), and it is why a c-token
// speculative verify cost c plain steps and could not pay for itself at any draft depth
// (``PERF_COREX.md`` section 14).
//
// Correctness contract: every row is bit-identical to the single-row entry on the same weight
// and activation. Both dispatch decisions -- the column width and the split-K width -- are
// therefore taken from the single-row path's own rules rather than chosen again here, and a
// shape whose regime has no batched kernel falls back to the per-row loop instead of to an
// approximate path.
//
// @param wt [K, N] int8 weight, contiguous.
// @param x [M, K] activations, bf16 or fp32; the output takes its dtype.
// @param scale [N] fp32 absmax scale, one per output column.
// @param threads Threads per block, ``0`` to pick from ``N``.
// @param unroll K-loop unroll; the launcher pins it, so this is reserved for a sweep.
// @return [M, N] output in ``x``'s dtype.
// @throws c10::Error when the weight is not int8, the scale length disagrees with N, or a
//         dtype and layout the kernel cannot index -- each of those reads the wrong elements
//         rather than failing.
torch::Tensor gemv_kn_i8_m(torch::Tensor wt, torch::Tensor x, torch::Tensor scale,
                           int64_t threads, int64_t unroll) {
    TORCH_CHECK(wt.dim() == 2, "weight must be 2-D [K, N]");
    TORCH_CHECK(x.dim() == 2, "activations must be 2-D [M, K]");
    TORCH_CHECK(wt.scalar_type() == torch::kChar,
                "int8 weight required (a bf16 weight here would be read one byte per "
                "value and produce plausible garbage)");
    TORCH_CHECK(scale.scalar_type() == torch::kFloat, "scale must be fp32");
    TORCH_CHECK(wt.is_contiguous() && x.is_contiguous() && scale.is_contiguous(),
                "contiguous inputs required");
    const int K = (int)wt.size(0), N = (int)wt.size(1), M = (int)x.size(0);
    TORCH_CHECK(wt.size(0) == x.size(1), "K mismatch: weight ", wt.size(0),
                " vs activation ", x.size(1));
    TORCH_CHECK(scale.numel() == N, "scale must hold one value per output column: got ",
                scale.numel(), " for N = ", N);
    TORCH_CHECK(M >= 1 && M <= 8, "small-M kernel handles M in [1, 8], got ", M);
    const bool x16 = x.scalar_type() == torch::kBFloat16;
    TORCH_CHECK(x16 || x.scalar_type() == torch::kFloat,
                "activation must be bf16 or fp32");
    (void)unroll;  // The launcher pins U=4; see launch_i8_m.
    auto y = torch::empty({M, N}, x.options());
    const int th = threads > 0 ? (int)threads : pick_threads(N);
    // Only the widths that carry a kernel are dispatched. Everything else -- a row count the
    // model never asks for -- stays on the per-row loop, which is the same code path the
    // absence of this kernel used to force on every call. Keeping it is what makes the
    // fallback bit-identical by construction rather than by two implementations agreeing.
    // Widths 2, 4 and 8 are the ones the model asks for (the batched-decode buckets and the
    // verify record), so those get their own instantiations. Any other width is served by the
    // next instantiation up, with the surplus rows reading row 0 again: a fallback to a
    // per-row loop would make M=5 cost five weight reads while M=4 costs one, which is a
    // cliff that changes the algorithm rather than the batch. The surplus rows compute real
    // values the caller never reads, so they cannot leak into an answer.
    const int padded = M <= 2 ? 2 : (M <= 4 ? 4 : 8);
    // The split is the single-row rule verbatim, so the partial-sum association matches what
    // the per-row loop would have produced.
    const int sp = pick_split_i8(N, K, th);
    if (x16) {
        if (padded == 2) return launch_i8_m<uint16_t, 2>(wt, x, y, M, scale, N, K, th, sp);
        if (padded == 4) return launch_i8_m<uint16_t, 4>(wt, x, y, M, scale, N, K, th, sp);
        return launch_i8_m<uint16_t, 8>(wt, x, y, M, scale, N, K, th, sp);
    }
    if (padded == 2) return launch_i8_m<float, 2>(wt, x, y, M, scale, N, K, th, sp);
    if (padded == 4) return launch_i8_m<float, 4>(wt, x, y, M, scale, N, K, th, sp);
    return launch_i8_m<float, 8>(wt, x, y, M, scale, N, K, th, sp);
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
