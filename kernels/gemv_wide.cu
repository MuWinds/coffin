// Wide-M int8 W8A16 GEMV for speculative verify (M in {16, 24, 32}).
//
// Why this exists
// A verify step projects K+1 activation vectors per sequence row in one weight
// pass; at B=8, K=1 that is M=16 and at K=3 it is M=32.  The engine has no
// batched kernel past M=8 (``gemv_w8a16_m`` covers {2,4,8}), so verify either
// splits into several M=8 passes (the weight bytes are then read again per
// pass) or falls into the prefill GEMM path (int8 -> fp16 dequant + hipBLAS,
// 250-290 ms for the projection segment).  This kernel covers M=16/24/32 in one
// pass, without dequantisation and without hipBLAS.
//
// How it is parallelised
// M is split into groups of MW activations; one warp owns R output rows and MW
// activation vectors of those rows.  The MT/MW warps of a row group read the
// same weight rows, so the extra groups are cache hits rather than extra HBM
// passes (the group lives inside one block).  ``acc`` is R*MW*8 fp32 values, so
// register pressure stays bounded for every shipped combo (64 for all of them).
//
// Bit-exactness (the hard acceptance)
// The kernel keeps the M=8 kernel's *exact* arithmetic per (row, activation):
//
//   * K is walked in 8-int8 chunks and lane l owns chunks l, l+32, l+64, ...
//     -- the same chunk->lane mapping as ``gemv_w8a16_m_kernel``, so each lane
//     accumulates the same K positions into the same eight chains;
//   * chain j takes the j-th int8 of the chunk in ascending chunk order;
//   * the final reduction adds the chains 0..7 sequentially (after the scalar
//     K%8 tail) and then does the same __shfl_down tree, and the per-row scale
//     is applied last.
//
// A row is therefore bit-identical to calling ``gemv_w8a16_m`` on that row's
// 8-activation slice, whatever U, R or MW is used (U only changes how many
// chunks are prefetched, not their order).  The probe checks this against the
// concatenation of M=8 calls.

#include "hip_ops.h"

#include <c10/cuda/CUDAStream.h>
#include <hip/hip_fp16.h>
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

// Four int8 from one 32-bit word -> four floats (vector type: stays in VGPRs).
__device__ __forceinline__ float4 i8x4(unsigned w) {
    float4 f;
    f.x = (float)(int)(signed char)(w & 0xffu);
    f.y = (float)(int)(signed char)((w >> 8) & 0xffu);
    f.z = (float)(int)(signed char)((w >> 16) & 0xffu);
    f.w = (float)(int)(signed char)((w >> 24) & 0xffu);
    return f;
}

// Two bf16 in one 32-bit word -> two floats.
__device__ __forceinline__ float2 bf16x2(unsigned u) {
    return make_float2(bf16_to_float((unsigned short)(u & 0xffffu)),
                       bf16_to_float((unsigned short)(u >> 16)));
}

// Eight FMA chains of one (row, activation) pair, as two float4s.
struct Acc8 {
    float4 a, b;
};

// One 8-int8 weight chunk applied to one 8-bf16 activation chunk.
// ``w0`` holds weights 0..3 (chunk bytes 0..3), ``w1`` weights 4..7.
__device__ __forceinline__ void fma8(Acc8& acc, const float4& w0, const float4& w1,
                                     const uint4& xb) {
    const float2 p0 = bf16x2(xb.x);   // activations 0,1
    const float2 p1 = bf16x2(xb.y);   // 2,3
    const float2 p2 = bf16x2(xb.z);   // 4,5
    const float2 p3 = bf16x2(xb.w);   // 6,7
    acc.a.x += w0.x * p0.x;
    acc.a.y += w0.y * p0.y;
    acc.a.z += w0.z * p1.x;
    acc.a.w += w0.w * p1.y;
    acc.b.x += w1.x * p2.x;
    acc.b.y += w1.y * p2.y;
    acc.b.z += w1.z * p3.x;
    acc.b.w += w1.w * p3.y;
}

// Grid: warp ``wid`` owns activation group ``wid % GM`` and row group
// ``blockIdx.x * (wpb / GM) + wid / GM``; a row group is R consecutive rows.
template <int MT, int R, int MW, int U>
__global__ void gemv_w8a16_wide_kernel(const int8_t* __restrict__ W,
                                       const float* __restrict__ S,
                                       const uint16_t* __restrict__ X,
                                       uint16_t* __restrict__ Y, int N, int K) {
    constexpr int GM = MT / MW;
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int wpb = blockDim.x >> 5;
    const int mg = wid % GM;
    const int rg = wid / GM;
    const int rowgroup = blockIdx.x * (wpb / GM) + rg;
    const int r0 = rowgroup * R;
    if (r0 >= N) return;

    const int am0 = mg * MW;
    const int nvec = K >> 3;
    const uint2* wv = reinterpret_cast<const uint2*>(W + (size_t)r0 * K);
    const uint4* xv = reinterpret_cast<const uint4*>(X);

    Acc8 acc[R][MW];
    float tail[R][MW];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int m = 0; m < MW; ++m) {
            acc[r][m].a = make_float4(0.f, 0.f, 0.f, 0.f);
            acc[r][m].b = make_float4(0.f, 0.f, 0.f, 0.f);
            tail[r][m] = 0.f;
        }

    const int step = 32 * U;
    int base = 0;
    for (; base + step <= nvec; base += step) {
        uint2 wc[R][U];
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int u = 0; u < U; ++u)
                wc[r][u] = wv[(size_t)r * nvec + base + u * 32 + lane];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int i = base + u * 32 + lane;
            uint4 xb[MW];
#pragma unroll
            for (int m = 0; m < MW; ++m) xb[m] = xv[(size_t)(am0 + m) * nvec + i];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const float4 w0 = i8x4(wc[r][u].x);
                const float4 w1 = i8x4(wc[r][u].y);
#pragma unroll
                for (int m = 0; m < MW; ++m) fma8(acc[r][m], w0, w1, xb[m]);
            }
        }
    }
    for (int i = base + lane; i < nvec; i += 32) {
        uint4 xb[MW];
#pragma unroll
        for (int m = 0; m < MW; ++m) xb[m] = xv[(size_t)(am0 + m) * nvec + i];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint2 wc = wv[(size_t)r * nvec + i];
            const float4 w0 = i8x4(wc.x);
            const float4 w1 = i8x4(wc.y);
#pragma unroll
            for (int m = 0; m < MW; ++m) fma8(acc[r][m], w0, w1, xb[m]);
        }
    }
    // K not a multiple of 8: scalar tail, added before the chains like the M=8
    // kernel does.
    for (int k = (nvec << 3) + lane; k < K; k += 32) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (r0 + r >= N) break;
            const float wf = (float)W[(size_t)(r0 + r) * K + k];
#pragma unroll
            for (int m = 0; m < MW; ++m)
                tail[r][m] = fmaf(wf, bf16_to_float(X[(size_t)(am0 + m) * K + k]),
                                  tail[r][m]);
        }
    }

#pragma unroll
    for (int r = 0; r < R; ++r) {
        if (r0 + r >= N) break;
        const float sc = S[r0 + r];
#pragma unroll
        for (int m = 0; m < MW; ++m) {
            float s = tail[r][m];
            s += acc[r][m].a.x;
            s += acc[r][m].a.y;
            s += acc[r][m].a.z;
            s += acc[r][m].a.w;
            s += acc[r][m].b.x;
            s += acc[r][m].b.y;
            s += acc[r][m].b.z;
            s += acc[r][m].b.w;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) s += __shfl_down(s, o);
            if (lane == 0) Y[(size_t)(am0 + m) * N + r0 + r] = float_to_bf16(s * sc);
        }
    }
}

template <int MT, int R, int MW, int U>
void launch_wide(int threads, const int8_t* wp, const float* sp, const uint16_t* xp,
                 uint16_t* yp, int N, int K, hipStream_t stream) {
    constexpr int GM = MT / MW;
    const int wpb = threads / 32;
    const int rows_per_block = (wpb / GM) * R;
    const int blocks = (N + rows_per_block - 1) / rows_per_block;
    gemv_w8a16_wide_kernel<MT, R, MW, U><<<blocks, threads, 0, stream>>>(wp, sp, xp,
                                                                        yp, N, K);
}

template <int MT, int R, int MW>
bool dispatch_ru(int U, int threads, const int8_t* wp, const float* sp,
                 const uint16_t* xp, uint16_t* yp, int N, int K, hipStream_t stream) {
    constexpr int GM = MT / MW;
    const int wpb = threads / 32;
    if (wpb % GM != 0) return false;
    if (U == 2) {
        launch_wide<MT, R, MW, 2>(threads, wp, sp, xp, yp, N, K, stream);
        return true;
    }
    if (U == 4) {
        launch_wide<MT, R, MW, 4>(threads, wp, sp, xp, yp, N, K, stream);
        return true;
    }
    return false;
}

// (R, MW) combos per M.  All keep acc = R*MW*8 = 64 floats.
template <int MT>
bool dispatch(int rows, int mw, int U, int threads, const int8_t* wp, const float* sp,
              const uint16_t* xp, uint16_t* yp, int N, int K, hipStream_t stream) {
    if (mw <= 0 || MT % mw != 0 || mw > MT) return false;   // must cover all M
    if (rows == 1 && mw == 8) return dispatch_ru<MT, 1, 8>(U, threads, wp, sp, xp, yp, N, K, stream);
    if (rows == 2 && mw == 4) return dispatch_ru<MT, 2, 4>(U, threads, wp, sp, xp, yp, N, K, stream);
    if (rows == 4 && mw == 2) return dispatch_ru<MT, 4, 2>(U, threads, wp, sp, xp, yp, N, K, stream);
    return false;
}

// Default tiling per M: two rows per warp with four activations keeps four warps
// per row group (more parallelism); one row/eight activations is the fallback.
inline void default_tiling(int M, int& rows, int& mw) {
    rows = 2;
    mw = 4;
    if (M == 24) { rows = 1; mw = 8; }     // 3 warps/row group: even warp split
}

}  // namespace

torch::Tensor gemv_w8a16_wide(torch::Tensor W, torch::Tensor S, torch::Tensor X,
                              int64_t U, int64_t threads) {
    TORCH_CHECK(W.is_contiguous() && S.is_contiguous() && X.is_contiguous(),
                "contiguous inputs required");
    TORCH_CHECK(W.scalar_type() == at::kChar, "W must be int8");
    const int N = W.size(0), K = W.size(1), M = X.size(0);
    TORCH_CHECK(X.size(1) == K, "X's K does not match W");
    TORCH_CHECK(M == 16 || M == 24 || M == 32, "M must be 16, 24 or 32");
    TORCH_CHECK(K % 8 == 0, "K must be a multiple of 8");
    int rows = 2, mw = 4;
    default_tiling(M, rows, mw);
    return gemv_w8a16_wide_dbg(W, S, X, U, rows, mw, threads);
}

torch::Tensor gemv_w8a16_wide_dbg(torch::Tensor W, torch::Tensor S, torch::Tensor X,
                                  int64_t U, int64_t rows, int64_t mw,
                                  int64_t threads) {
    TORCH_CHECK(W.is_contiguous() && S.is_contiguous() && X.is_contiguous(),
                "contiguous inputs required");
    TORCH_CHECK(W.scalar_type() == at::kChar, "W must be int8");
    const int N = W.size(0), K = W.size(1), M = X.size(0);
    TORCH_CHECK(X.size(1) == K, "X's K does not match W");
    TORCH_CHECK(M == 16 || M == 24 || M == 32, "M must be 16, 24 or 32");
    TORCH_CHECK(K % 8 == 0, "K must be a multiple of 8");
    if (threads <= 0) {
        const int gm = (int)(M / mw);
        // 8 warps where they divide evenly, else the smallest multiple of gm.
        threads = 8 / gm >= 1 ? (8 / gm) * gm * 32 : gm * 32;
    }
    TORCH_CHECK(threads % 32 == 0 && threads >= 32 && threads <= 256,
                "threads must be a multiple of 32 in 32..256");
    auto Y = torch::empty({M, N}, X.options());
    hipStream_t stream = at::cuda::getCurrentCUDAStream();
    const auto* wp = W.data_ptr<int8_t>();
    const auto* sp = S.data_ptr<float>();
    const auto* xp = reinterpret_cast<const uint16_t*>(X.data_ptr<at::BFloat16>());
    auto* yp = reinterpret_cast<uint16_t*>(Y.data_ptr<at::BFloat16>());
    const int th = (int)threads;
    const int rr = (int)rows, mw_ = (int)mw, uu = (int)U;
    bool ok = false;
    if (M == 16) ok = dispatch<16>(rr, mw_, uu, th, wp, sp, xp, yp, N, K, stream);
    else if (M == 24) ok = dispatch<24>(rr, mw_, uu, th, wp, sp, xp, yp, N, K, stream);
    else ok = dispatch<32>(rr, mw_, uu, th, wp, sp, xp, yp, N, K, stream);
    TORCH_CHECK(ok, "unsupported (M, rows, mw, U, threads) for gemv_w8a16_wide");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return Y;
}

// packed fp16 variant: V_DOT2_F32_F16 does two fp16 multiplies per instruction
// with an fp32 accumulator.  int8 weights convert to fp16 exactly (all values
// in [-127,127] are representable), activations are converted bf16 -> fp16 once
// per call (exact while |x| <= 65504; clamped above that, fp16 denormals flush
// only below ~6e-8).  That halves the arithmetic instructions per weight element
// versus the scalar int8 path, which is where the wide-M kernel was losing.
namespace {

union H2U {
    __half2 h;
    unsigned u;
};

__device__ __forceinline__ float dot2(float acc, unsigned a, unsigned b) {
    float r;
    asm("v_dot2_f32_f16 %0, %1, %2, %3" : "=v"(r) : "v"(a), "v"(b), "v"(acc));
    return r;
}

// Four int8 (one 32-bit word) -> two packed half2 values (two 32-bit words).
__device__ __forceinline__ void i8x4_to_h2(unsigned w, unsigned& o0, unsigned& o1) {
    const float f0 = (float)(int)(signed char)(w & 0xffu);
    const float f1 = (float)(int)(signed char)((w >> 8) & 0xffu);
    const float f2 = (float)(int)(signed char)((w >> 16) & 0xffu);
    const float f3 = (float)(int)(signed char)((w >> 24) & 0xffu);
    H2U a, b;
    a.h = __floats2half2_rn(f0, f1);
    b.h = __floats2half2_rn(f2, f3);
    o0 = a.u;
    o1 = b.u;
}

__global__ void bf16_to_fp16_kernel(const uint16_t* __restrict__ x,
                                    uint16_t* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float f = bf16_to_float(x[i]);
    f = fminf(fmaxf(f, -65504.f), 65504.f);
    H2U h;
    h.h = __floats2half2_rn(f, f);
    y[i] = (uint16_t)(h.u & 0xffffu);
}

// Per warp: R output rows, MW activation vectors, four fp32 dot2 accumulators
// per (row, activation) pair.  Same row-group/M-group structure as the int8
// kernel; the arithmetic is packed but the per-row sums stay in fp32.
template <int MT, int R, int MW, int U>
__global__ void gemv_w8a16_dot2_kernel(const int8_t* __restrict__ W,
                                       const float* __restrict__ S,
                                       const uint16_t* __restrict__ Xh,
                                       uint16_t* __restrict__ Y, int N, int K) {
    constexpr int GM = MT / MW;
    constexpr int CH = 4;                    // fp32 chains per (row, activation)
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int wpb = blockDim.x >> 5;
    const int mg = wid % GM;
    const int rg = wid / GM;
    const int rowgroup = blockIdx.x * (wpb / GM) + rg;
    const int r0 = rowgroup * R;
    if (r0 >= N) return;

    const int am0 = mg * MW;
    const int nvec = K >> 4;                 // 16 int8 (uint4) per lane per step
    const int nvec8 = K >> 3;                // uint4 units of the fp16 activations
    const uint4* wv = reinterpret_cast<const uint4*>(W + (size_t)r0 * K);
    const uint4* xv = reinterpret_cast<const uint4*>(Xh);

    float acc[R][MW][CH];
    float tail[R][MW];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int m = 0; m < MW; ++m) {
#pragma unroll
            for (int j = 0; j < CH; ++j) acc[r][m][j] = 0.f;
            tail[r][m] = 0.f;
        }

    const int step = 32 * U;
    int base = 0;
    for (; base + step <= nvec; base += step) {
        uint4 wc[R][U];
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int u = 0; u < U; ++u)
                wc[r][u] = wv[(size_t)r * nvec + base + u * 32 + lane];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const size_t i = (size_t)(base + u * 32 + lane);
            const size_t xi = i * 2;         // two uint4 (16 fp16) per 16 weights
#pragma unroll
            for (int r = 0; r < R; ++r) {
                unsigned h[8];
                i8x4_to_h2(wc[r][u].x, h[0], h[1]);
                i8x4_to_h2(wc[r][u].y, h[2], h[3]);
                i8x4_to_h2(wc[r][u].z, h[4], h[5]);
                i8x4_to_h2(wc[r][u].w, h[6], h[7]);
#pragma unroll
                for (int m = 0; m < MW; ++m) {
                    const uint4 xa = xv[(size_t)(am0 + m) * nvec8 + xi];
                    const uint4 xb = xv[(size_t)(am0 + m) * nvec8 + xi + 1];
                    acc[r][m][0] = dot2(acc[r][m][0], h[0], xa.x);
                    acc[r][m][1] = dot2(acc[r][m][1], h[1], xa.y);
                    acc[r][m][2] = dot2(acc[r][m][2], h[2], xa.z);
                    acc[r][m][3] = dot2(acc[r][m][3], h[3], xa.w);
                    acc[r][m][0] = dot2(acc[r][m][0], h[4], xb.x);
                    acc[r][m][1] = dot2(acc[r][m][1], h[5], xb.y);
                    acc[r][m][2] = dot2(acc[r][m][2], h[6], xb.z);
                    acc[r][m][3] = dot2(acc[r][m][3], h[7], xb.w);
                }
            }
        }
    }
    for (int i = base + lane; i < nvec; i += 32) {
        const size_t xi = (size_t)i * 2;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint4 wc = wv[(size_t)r * nvec + i];
            unsigned h[8];
            i8x4_to_h2(wc.x, h[0], h[1]);
            i8x4_to_h2(wc.y, h[2], h[3]);
            i8x4_to_h2(wc.z, h[4], h[5]);
            i8x4_to_h2(wc.w, h[6], h[7]);
#pragma unroll
            for (int m = 0; m < MW; ++m) {
                const uint4 xa = xv[(size_t)(am0 + m) * nvec8 + xi];
                const uint4 xb = xv[(size_t)(am0 + m) * nvec8 + xi + 1];
                acc[r][m][0] = dot2(acc[r][m][0], h[0], xa.x);
                acc[r][m][1] = dot2(acc[r][m][1], h[1], xa.y);
                acc[r][m][2] = dot2(acc[r][m][2], h[2], xa.z);
                acc[r][m][3] = dot2(acc[r][m][3], h[3], xa.w);
                acc[r][m][0] = dot2(acc[r][m][0], h[4], xb.x);
                acc[r][m][1] = dot2(acc[r][m][1], h[5], xb.y);
                acc[r][m][2] = dot2(acc[r][m][2], h[6], xb.z);
                acc[r][m][3] = dot2(acc[r][m][3], h[7], xb.w);
            }
        }
    }
    // K not a multiple of 16: scalar fp32 tail (exact int8 -> float path).
    for (int k = (nvec << 4) + lane; k < K; k += 32) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (r0 + r >= N) break;
            const float wf = (float)W[(size_t)(r0 + r) * K + k];
#pragma unroll
            for (int m = 0; m < MW; ++m) {
                const __half xh = *reinterpret_cast<const __half*>(
                    &Xh[(size_t)(am0 + m) * K + k]);
                tail[r][m] = fmaf(wf, __half2float(xh), tail[r][m]);
            }
        }
    }

#pragma unroll
    for (int r = 0; r < R; ++r) {
        if (r0 + r >= N) break;
        const float sc = S[r0 + r];
#pragma unroll
        for (int m = 0; m < MW; ++m) {
            float s = (acc[r][m][0] + acc[r][m][1]) + (acc[r][m][2] + acc[r][m][3])
                    + tail[r][m];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) s += __shfl_down(s, o);
            if (lane == 0) Y[(size_t)(am0 + m) * N + r0 + r] = float_to_bf16(s * sc);
        }
    }
}

template <int MT, int R, int MW, int U>
void launch_dot2(int threads, const int8_t* wp, const float* sp, const uint16_t* xp,
                 uint16_t* yp, int N, int K, hipStream_t stream) {
    constexpr int GM = MT / MW;
    const int wpb = threads / 32;
    const int rows_per_block = (wpb / GM) * R;
    const int blocks = (N + rows_per_block - 1) / rows_per_block;
    gemv_w8a16_dot2_kernel<MT, R, MW, U><<<blocks, threads, 0, stream>>>(wp, sp, xp,
                                                                         yp, N, K);
}

template <int MT, int R, int MW>
bool dispatch_dot2_ru(int U, int threads, const int8_t* wp, const float* sp,
                      const uint16_t* xp, uint16_t* yp, int N, int K,
                      hipStream_t stream) {
    constexpr int GM = MT / MW;
    if ((threads / 32) % GM != 0) return false;
    if (U == 2) {
        launch_dot2<MT, R, MW, 2>(threads, wp, sp, xp, yp, N, K, stream);
        return true;
    }
    if (U == 4) {
        launch_dot2<MT, R, MW, 4>(threads, wp, sp, xp, yp, N, K, stream);
        return true;
    }
    return false;
}

template <int MT>
bool dispatch_dot2(int rows, int mw, int U, int threads, const int8_t* wp,
                   const float* sp, const uint16_t* xp, uint16_t* yp, int N, int K,
                   hipStream_t stream) {
    if (mw <= 0 || MT % mw != 0 || mw > MT) return false;   // must cover all M
    // Per-M tiling set; every combo keeps R*MW*4 <= 64 accumulator floats.
    if (MT == 1) {
        if (rows == 1 && mw == 1) return dispatch_dot2_ru<1, 1, 1>(U, threads, wp, sp, xp, yp, N, K, stream);
        if (rows == 2 && mw == 1) return dispatch_dot2_ru<1, 2, 1>(U, threads, wp, sp, xp, yp, N, K, stream);
        if (rows == 4 && mw == 1) return dispatch_dot2_ru<1, 4, 1>(U, threads, wp, sp, xp, yp, N, K, stream);
    } else if (MT == 2) {
        if (rows == 1 && mw == 2) return dispatch_dot2_ru<2, 1, 2>(U, threads, wp, sp, xp, yp, N, K, stream);
        if (rows == 2 && mw == 2) return dispatch_dot2_ru<2, 2, 2>(U, threads, wp, sp, xp, yp, N, K, stream);
        if (rows == 4 && mw == 2) return dispatch_dot2_ru<2, 4, 2>(U, threads, wp, sp, xp, yp, N, K, stream);
    } else if (MT == 4) {
        if (rows == 1 && mw == 4) return dispatch_dot2_ru<4, 1, 4>(U, threads, wp, sp, xp, yp, N, K, stream);
        if (rows == 2 && mw == 4) return dispatch_dot2_ru<4, 2, 4>(U, threads, wp, sp, xp, yp, N, K, stream);
        if (rows == 2 && mw == 2) return dispatch_dot2_ru<4, 2, 2>(U, threads, wp, sp, xp, yp, N, K, stream);
        if (rows == 1 && mw == 2) return dispatch_dot2_ru<4, 1, 2>(U, threads, wp, sp, xp, yp, N, K, stream);
        if (rows == 4 && mw == 2) return dispatch_dot2_ru<4, 4, 2>(U, threads, wp, sp, xp, yp, N, K, stream);
    } else if (MT == 8) {
        if (rows == 1 && mw == 8) return dispatch_dot2_ru<8, 1, 8>(U, threads, wp, sp, xp, yp, N, K, stream);
        if (rows == 2 && mw == 8) return dispatch_dot2_ru<8, 2, 8>(U, threads, wp, sp, xp, yp, N, K, stream);
        if (rows == 2 && mw == 4) return dispatch_dot2_ru<8, 2, 4>(U, threads, wp, sp, xp, yp, N, K, stream);
        if (rows == 4 && mw == 4) return dispatch_dot2_ru<8, 4, 4>(U, threads, wp, sp, xp, yp, N, K, stream);
        if (rows == 4 && mw == 2) return dispatch_dot2_ru<8, 4, 2>(U, threads, wp, sp, xp, yp, N, K, stream);
    } else if (MT == 16) {
        if (rows == 1 && mw == 16) return dispatch_dot2_ru<16, 1, 16>(U, threads, wp, sp, xp, yp, N, K, stream);
        if (rows == 2 && mw == 8) return dispatch_dot2_ru<16, 2, 8>(U, threads, wp, sp, xp, yp, N, K, stream);
        if (rows == 4 && mw == 4) return dispatch_dot2_ru<16, 4, 4>(U, threads, wp, sp, xp, yp, N, K, stream);
        if (rows == 1 && mw == 8) return dispatch_dot2_ru<16, 1, 8>(U, threads, wp, sp, xp, yp, N, K, stream);
    } else if (MT == 24) {
        if (rows == 2 && mw == 8) return dispatch_dot2_ru<24, 2, 8>(U, threads, wp, sp, xp, yp, N, K, stream);
        if (rows == 4 && mw == 4) return dispatch_dot2_ru<24, 4, 4>(U, threads, wp, sp, xp, yp, N, K, stream);
        if (rows == 1 && mw == 8) return dispatch_dot2_ru<24, 1, 8>(U, threads, wp, sp, xp, yp, N, K, stream);
    } else {
        if (rows == 2 && mw == 8) return dispatch_dot2_ru<32, 2, 8>(U, threads, wp, sp, xp, yp, N, K, stream);
        if (rows == 4 && mw == 4) return dispatch_dot2_ru<32, 4, 4>(U, threads, wp, sp, xp, yp, N, K, stream);
        if (rows == 1 && mw == 16) return dispatch_dot2_ru<32, 1, 16>(U, threads, wp, sp, xp, yp, N, K, stream);
    }
    return false;
}

inline void default_dot2_tiling(int M, int& rows, int& mw) {
    // These are the tiling points the probe picked as fastest per M (rows per
    // warp, activation vectors per warp; acc = rows*mw*4 = 64 floats for the
    // M>=4 cases).  Every combo keeps the per-row fp32 sum in the same order, so
    // the choice only affects speed.
    if (M == 1) { rows = 1; mw = 1; }
    else if (M == 2) { rows = 2; mw = 2; }
    else if (M == 4) { rows = 4; mw = 2; }
    else { rows = 4; mw = 4; }      // M = 8, 16, 24, 32
}

}  // namespace

torch::Tensor gemv_w8a16_wide_dot2(torch::Tensor W, torch::Tensor S, torch::Tensor X,
                                   int64_t U, int64_t threads) {
    int rows = 1, mw = 16;
    const int M = X.size(0);
    default_dot2_tiling(M, rows, mw);
    return gemv_w8a16_wide_dot2_dbg(W, S, X, U, rows, mw, threads);
}

torch::Tensor gemv_w8a16_wide_dot2_dbg(torch::Tensor W, torch::Tensor S,
                                       torch::Tensor X, int64_t U, int64_t rows,
                                       int64_t mw, int64_t threads) {
    TORCH_CHECK(W.is_contiguous() && S.is_contiguous() && X.is_contiguous(),
                "contiguous inputs required");
    TORCH_CHECK(W.scalar_type() == at::kChar, "W must be int8");
    TORCH_CHECK(X.scalar_type() == at::kHalf || X.scalar_type() == at::kBFloat16,
                "X must be fp16 or bf16");
    const int N = W.size(0), K = W.size(1), M = X.size(0);
    TORCH_CHECK(X.size(1) == K, "X's K does not match W");
    TORCH_CHECK(M >= 1 && M <= 32, "M must be in 1..32");
    TORCH_CHECK(K % 16 == 0, "K must be a multiple of 16");
    hipStream_t stream = at::cuda::getCurrentCUDAStream();

    // Activations in fp16: exact bf16 -> fp16 conversion, done once per call.
    torch::Tensor Xh;
    if (X.scalar_type() == at::kHalf) {
        Xh = X;
    } else {
        Xh = torch::empty(X.sizes(), X.options().dtype(at::kHalf));
        const int n = M * K;
        bf16_to_fp16_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
            reinterpret_cast<const uint16_t*>(X.data_ptr<at::BFloat16>()),
            reinterpret_cast<uint16_t*>(Xh.data_ptr<at::Half>()), n);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    if (threads <= 0) {
        const int gm = (int)(M / mw);
        threads = (gm >= 8) ? gm * 32 : (8 / gm) * gm * 32;
    }
    TORCH_CHECK(threads % 32 == 0 && threads >= 32 && threads <= 256,
                "threads must be a multiple of 32 in 32..256");
    auto Y = torch::empty({M, N}, X.options().dtype(at::kBFloat16));
    const auto* wp = W.data_ptr<int8_t>();
    const auto* sp = S.data_ptr<float>();
    const auto* xp = reinterpret_cast<const uint16_t*>(Xh.data_ptr<at::Half>());
    auto* yp = reinterpret_cast<uint16_t*>(Y.data_ptr<at::BFloat16>());
    const int th = (int)threads, rr = (int)rows, mw_ = (int)mw, uu = (int)U;
    bool ok = false;
    if (M == 1) ok = dispatch_dot2<1>(rr, mw_, uu, th, wp, sp, xp, yp, N, K, stream);
    else if (M == 2) ok = dispatch_dot2<2>(rr, mw_, uu, th, wp, sp, xp, yp, N, K, stream);
    else if (M == 4) ok = dispatch_dot2<4>(rr, mw_, uu, th, wp, sp, xp, yp, N, K, stream);
    else if (M == 8) ok = dispatch_dot2<8>(rr, mw_, uu, th, wp, sp, xp, yp, N, K, stream);
    else if (M == 16) ok = dispatch_dot2<16>(rr, mw_, uu, th, wp, sp, xp, yp, N, K, stream);
    else if (M == 24) ok = dispatch_dot2<24>(rr, mw_, uu, th, wp, sp, xp, yp, N, K, stream);
    else ok = dispatch_dot2<32>(rr, mw_, uu, th, wp, sp, xp, yp, N, K, stream);
    TORCH_CHECK(ok, "unsupported (M, rows, mw, U, threads) for gemv_w8a16_wide_dot2");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return Y;
}
