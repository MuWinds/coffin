// @file gemv_int4_ext.cu
// @brief Standalone probe extension for int4 weights and bandwidth ceilings.
//
// Module: coffin/diag. Built and loaded on its own (``build_int4.py``); it is NOT
// part of the engine extension, so it must not include ``hip_ops.h`` or be added
// to ``bindings.cpp``.
//
// It adds three things the engine does not have:
//   1. gemv_w4a16: weight-only int4 GEMV (group-wise symmetric, group 128, fp16
//      scale), one warp per output row, 8 nibbles per uint32 load.
//   2. read_bw: a pure streaming-read kernel, to measure the HBM read ceiling an
//      int4 GEMV would have to reach.
//   3. host_read_bw / copy_bw: mapped-pinned-memory read bandwidth (zero-copy) and
//      pinned H2D/D2H copy bandwidth.
//
// Constraints:
//   * The packing below is the checkpoint contract: element k lives in byte k/2
//     (low nibble if k even, high if odd); the stored nibble is u = q + 8 with q in
//     [-8, 7]. Groups of 128 elements share one fp16 scale (last axis scaled per
//     group within a row). Reading a differently packed tensor yields plausible
//     but wrong numbers, which is why the shape checks insist on K % 256 == 0.
//   * Everything here is diagnostic: no graph capture, no engine state, and the
//     only consumer is a probe script.
//
// The dot product folds the scale per group:
//     y[n] = sum_g  scale[n,g] * ( sum_{k in g} (u-8) * x[k] )
// and the activation sub-sums are precomputed once: XS[c] = 8 * sum of the 8
// elements of chunk c (K/8 values), so each lane's correction is exact.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

namespace {

__device__ __forceinline__ float bf16f(unsigned short h) {
    return __uint_as_float(((unsigned)h) << 16);
}

__device__ __forceinline__ unsigned short fbf16(float f) {
    unsigned u = __float_as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (unsigned short)(u >> 16);
}

// @brief One warp per output row, 8 int4 weights per 32-bit load.
//
// A "tile" is 32 chunks = 256 elements = 2 groups of 128, so lane l of tile t is
// always in group 2t + (l >> 4); no divisions in the inner loop. The group scale is
// applied to the tile partial, and the +8 offset is removed by subtracting the
// precomputed activation sum ``XS`` rather than per element.
//
// @param W4 [N, K/2] packed int4 bytes (two nibbles per byte).
// @param S  [N, K/128] fp16 per-group scales.
// @param XS [K/8] fp32, ``8 * sum`` of each 8-element activation chunk.
// @param x  [K] bf16 activation.
// @param y  [N] bf16 output.
template <int U>
__global__ void gemv_w4a16_kernel(const int8_t* __restrict__ W4,
                                  const __half* __restrict__ S,
                                  const float* __restrict__ XS,
                                  const uint16_t* __restrict__ x,
                                  uint16_t* __restrict__ y, int N, int K) {
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    if (warp >= N) return;

    const int nt = K >> 8;                 // 256-element tiles
    const int G = K >> 7;                  // 128-element groups
    const uint32_t* wv = reinterpret_cast<const uint32_t*>(W4 + (size_t)warp * (K >> 1));
    const uint4* xv = reinterpret_cast<const uint4*>(x);
    const __half* sr = S + (size_t)warp * G;

    float acc = 0.f;
    int t = 0;
    for (; t + U <= nt; t += U) {
        float part = 0.f;
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const uint32_t wc = wv[(t + u) * 32 + lane];
            const uint4 xb = xv[(t + u) * 32 + lane];
            const unsigned short* xh = reinterpret_cast<const unsigned short*>(&xb);
            float q = 0.f;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float wv4 = (float)((wc >> (4 * j)) & 0xFu);   // raw u = q+8
                q = fmaf(wv4, bf16f(xh[j]), q);
            }
            const int gi = 2 * (t + u) + (lane >> 4);
            part = fmaf(q - XS[(t + u) * 32 + lane], __half2float(sr[gi]), part);
        }
        acc += part;
    }
    for (; t < nt; ++t) {                  // tail (nt not a multiple of U)
        const uint32_t wc = wv[t * 32 + lane];
        const uint4 xb = xv[t * 32 + lane];
        const unsigned short* xh = reinterpret_cast<const unsigned short*>(&xb);
        float q = 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float wv4 = (float)((wc >> (4 * j)) & 0xFu);       // raw u = q+8
            q = fmaf(wv4, bf16f(xh[j]), q);
        }
        const int gi = 2 * t + (lane >> 4);
        acc = fmaf(q - XS[t * 32 + lane], __half2float(sr[gi]), acc);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down(acc, o);
    if (lane == 0) y[warp] = fbf16(acc);
}

// @brief Pure streaming read: the HBM read ceiling an int4 GEMV would have to hit.
//
// Every byte goes through the SM and is XOR-folded so the compiler cannot elide the
// loads; the atomic fires only on a value that never occurs, so the reduction has
// no measurable cost.
__global__ void read_bw_kernel(const uint4* __restrict__ p, long n4,
                               unsigned* __restrict__ out) {
    unsigned s = 0;
    const long stride = (long)gridDim.x * blockDim.x;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        const uint4 v = p[i];
        s += v.x ^ v.y ^ v.z ^ v.w;
    }
    if (s == 0xdeadbeefu) atomicXor(out, s);
}

}  // namespace

// @brief int4 (g128) weight-only GEMV with bf16 activations.
//
// @param W4 [N, K/2] packed nibbles; K % 256 == 0 or the tile/group mapping breaks.
// @param S [N, K/128] fp16 per-group scales.
// @param XS [K/8] fp32 precomputed activation chunk sums (see the file header).
// @param U Unroll: 1, 2, 4 or 8.
// @return [N] bf16.
// @throws c10::Error (TORCH_CHECK) when a dtype, shape or K alignment does not
//         match the packing contract; the pattern is repack the checkpoint, not
//         relax the check.
torch::Tensor gemv_w4a16(torch::Tensor W4, torch::Tensor S, torch::Tensor XS,
                         torch::Tensor x, int64_t U) {
    TORCH_CHECK(W4.is_contiguous() && S.is_contiguous() && XS.is_contiguous() &&
                x.is_contiguous(), "contiguous inputs required");
    TORCH_CHECK(W4.scalar_type() == at::kChar || W4.scalar_type() == at::kByte,
                "W4 must be packed int8/uint8 bytes");
    TORCH_CHECK(S.scalar_type() == at::kHalf, "S must be fp16");
    TORCH_CHECK(XS.scalar_type() == at::kFloat, "XS must be fp32");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be bf16");
    const int N = S.size(0), K = (int)x.size(0);
    TORCH_CHECK(K % 256 == 0, "K must be a multiple of 256 for this probe");
    TORCH_CHECK(W4.size(0) == N && W4.size(1) == K / 2, "W4 shape mismatch");
    TORCH_CHECK(S.size(1) == K / 128 && XS.size(0) == K / 8, "scale shape mismatch");
    auto y = torch::empty({N}, x.options());
    const int threads = 256, warps = threads / 32;
    const int blocks = (N + warps - 1) / warps;
    const auto* wp = reinterpret_cast<const int8_t*>(W4.data_ptr());
    const auto* sp = reinterpret_cast<const __half*>(S.data_ptr<at::Half>());
    const auto* xsp = XS.data_ptr<float>();
    const auto* xp = reinterpret_cast<const uint16_t*>(x.data_ptr<at::BFloat16>());
    auto* yp = reinterpret_cast<uint16_t*>(y.data_ptr<at::BFloat16>());
    hipStream_t stream = at::cuda::getCurrentCUDAStream();
    const int uu = (int)U;
#define W4_CASE(U_)                                                                    \
    if (uu == U_) {                                                                    \
        gemv_w4a16_kernel<U_><<<blocks, threads, 0, stream>>>(wp, sp, xsp, xp, yp, N, K); \
    }
    W4_CASE(1) W4_CASE(2) W4_CASE(4) W4_CASE(8)
#undef W4_CASE
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

// @brief HBM streaming read bandwidth over a device buffer, in GB/s.
//
// A fixed 2048x256 grid and 3 untimed warm-up launches, because the first pass
// pays for TLB and page-table warm-up and would otherwise depress the number.
//
// @param nbytes Buffer size; internally rounded down to whole uint4.
// @param iters Timed launches.
// @return GB/s, computed from the event delta over all iterations.
// @throws c10::Error (TORCH_CHECK) when an allocation or memset fails; the run is
//         then meaningless rather than merely slow.
double read_bw(int64_t nbytes, int64_t iters) {
    const long n4 = (long)(nbytes / 16);
    void* buf = nullptr;
    TORCH_CHECK(hipMalloc(&buf, (size_t)n4 * 16) == hipSuccess, "hipMalloc failed");
    TORCH_CHECK(hipMemset(buf, 1, (size_t)n4 * 16) == hipSuccess, "hipMemset failed");
    unsigned* out = nullptr;
    TORCH_CHECK(hipMalloc(&out, 4) == hipSuccess, "hipMalloc(out) failed");
    hipEvent_t e0, e1;
    hipEventCreate(&e0);
    hipEventCreate(&e1);
    hipStream_t s = at::cuda::getCurrentCUDAStream();
    // warm-up + timed loop
    for (int i = 0; i < 3; ++i)
        read_bw_kernel<<<2048, 256, 0, s>>>(reinterpret_cast<const uint4*>(buf), n4, out);
    hipEventRecord(e0, s);
    for (int i = 0; i < iters; ++i)
        read_bw_kernel<<<2048, 256, 0, s>>>(reinterpret_cast<const uint4*>(buf), n4, out);
    hipEventRecord(e1, s);
    hipEventSynchronize(e1);
    float ms = 0.f;
    hipEventElapsedTime(&ms, e0, e1);
    hipFree(buf);
    hipFree(out);
    hipEventDestroy(e0);
    hipEventDestroy(e1);
    const double gb = (double)n4 * 16.0 * iters / 1e9;
    return gb / (ms / 1e3);
}

// @brief Mapped-pinned (zero-copy) host read plus pinned H2D/D2H copy bandwidth.
//
// The three numbers answer different questions: ``mapped_read_gbps`` is what a
// device kernel gets reading host RAM over PCIe, while the copy rates are the
// staging alternative (copy in, compute, copy out) it would be compared against.
// The host buffer is freed with ``hipHostFree`` and the device buffers with
// ``hipFree``, so repeated probe runs do not leak pinned memory.
//
// @param nbytes Buffer size; internally rounded down to whole uint4.
// @param iters Timed launches / copies.
// @return A dict with ``mapped_read_gbps``, ``h2d_gbps`` and ``d2h_gbps``.
// @throws c10::Error (TORCH_CHECK) when the pinned allocation or the device-pointer
//         lookup fails; without pinned memory the measurement would be meaningless.
pybind11::dict host_read_bw(int64_t nbytes, int64_t iters) {
    const long n4 = (long)(nbytes / 16);
    void* h = nullptr;
    TORCH_CHECK(hipHostMalloc(&h, (size_t)n4 * 16, 0) == hipSuccess, "hipHostMalloc failed");
    memset(h, 1, (size_t)n4 * 16);
    void* d = nullptr;
    TORCH_CHECK(hipHostGetDevicePointer(&d, h, 0) == hipSuccess, "hipHostGetDevicePointer failed");
    unsigned* out = nullptr;
    TORCH_CHECK(hipMalloc(&out, 4) == hipSuccess, "hipMalloc(out) failed");
    hipEvent_t e0, e1;
    hipEventCreate(&e0);
    hipEventCreate(&e1);
    hipStream_t s = at::cuda::getCurrentCUDAStream();

    for (int i = 0; i < 2; ++i)
        read_bw_kernel<<<2048, 256, 0, s>>>(reinterpret_cast<const uint4*>(d), n4, out);
    hipEventRecord(e0, s);
    for (int i = 0; i < iters; ++i)
        read_bw_kernel<<<2048, 256, 0, s>>>(reinterpret_cast<const uint4*>(d), n4, out);
    hipEventRecord(e1, s);
    hipEventSynchronize(e1);
    float ms = 0.f;
    hipEventElapsedTime(&ms, e0, e1);
    const double gb = (double)n4 * 16.0 * iters / 1e9;
    const double mapped_read = gb / (ms / 1e3);

    // pinned H2D / D2H copies of the same buffer
    void* dbuf = nullptr;
    hipMalloc(&dbuf, (size_t)n4 * 16);
    hipEventRecord(e0, s);
    for (int i = 0; i < iters; ++i)
        hipMemcpyAsync(dbuf, h, (size_t)n4 * 16, hipMemcpyHostToDevice, s);
    hipEventRecord(e1, s);
    hipEventSynchronize(e1);
    hipEventElapsedTime(&ms, e0, e1);
    const double h2d = gb / (ms / 1e3);
    hipEventRecord(e0, s);
    for (int i = 0; i < iters; ++i)
        hipMemcpyAsync(h, dbuf, (size_t)n4 * 16, hipMemcpyDeviceToHost, s);
    hipEventRecord(e1, s);
    hipEventSynchronize(e1);
    hipEventElapsedTime(&ms, e0, e1);
    const double d2h = gb / (ms / 1e3);

    hipFree(dbuf);
    hipFree(out);
    hipHostFree(h);
    hipEventDestroy(e0);
    hipEventDestroy(e1);
    pybind11::dict r;
    r["mapped_read_gbps"] = mapped_read;
    r["h2d_gbps"] = h2d;
    r["d2h_gbps"] = d2h;
    return r;
}

// Probe-only module: it is loaded by ``build_int4.py`` under its own name, so
// these three names never appear in the engine extension.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemv_w4a16", &gemv_w4a16, "int4 (g128) weight-only GEMV, bf16 activation");
    m.def("read_bw", &read_bw, "HBM streaming read GB/s");
    m.def("host_read_bw", &host_read_bw, "mapped pinned host read + pinned copy GB/s");
}
