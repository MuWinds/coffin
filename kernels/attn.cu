// @file attn.cu
// @brief Fused single-token full attention over the int8 KV cache.
//
// Module: coffin/kernels. The B = 1 decode path, loaded by ``hip_ops.py`` as part
// of the torch extension; the batched counterparts live in ``batch.cu``.
//
// Why fused: the eager path materialises the whole KV cache as bf16 (two
// ``.to(D)`` copies of [nkv, S, hd]), runs two einsums plus the per-token scale
// fold, the causal mask, the softmax and the output gate as separate kernels.
// This kernel does the same in one pass: dequantise the int8 k/v on the fly, fold
// the per-token scale into the scores and probabilities, and apply the gate, so
// the cache never leaves int8 and no [nkv, S, hd] intermediate is written.
//
// Constraints:
//   * The key window is streamed in ATTN_KEY_BLOCK keys with an online softmax,
//     never staged whole: LDS is 64 KB per CU and a [S] score vector is 1 MB at
//     262144, so a staged window does not launch (see "Key blocking" below).
//   * One block per query head (times the split count); no cross-block
//     synchronisation. Slices publish partials and the combine kernel merges them.
//   * K/V stay int8 with a per-token fp32 scale; only the row id travels
//     downstream, so the dense and the paged layout share one kernel.
//   * Window length and position live on the device: the decode step is a
//     captured graph, so a host-side scalar would freeze at capture time.
//
// Key blocking: the score vector for the whole window used to live in shared
// memory -- [S] floats, S = max_seq -- which is 16 KB at 4096 and 1 MB at 262144.
// gfx906 has 64 KB of LDS per CU, and a dynamic shared allocation above 48 KB
// needs an opt-in attribute, so anything past ~12k context could not even launch.
// The window is now walked in blocks of ATTN_KEY_BLOCK with a running max/sum
// (online softmax) and a rescaled output accumulator, so the shared cost is
// constant and 256k fits. The work is unchanged: the old kernel already walked
// every valid key for the V accumulation, it just did it as one loop over the
// whole window instead of block by block.

#include "hip_ops.h"

#include <c10/cuda/CUDAStream.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace {

//: Keys per streamed block. Only bounds LDS plus one rescale of the output
//: accumulator per block; 1024 keys is ~10 KB of shared memory.
constexpr int ATTN_KEY_BLOCK = 1024;

//: Key-range splits per query head. One block per *head* is only 16 blocks, i.e.
//: ~5% of this GPU's wavefront slots, so the step was latency-bound rather than
//: bandwidth-bound (measured 3.1 us per token of context at 64k, ~24 GB/s of KV
//: traffic against a ~600 GB/s roof). Each split block produces a partial
//: (max, sum, acc) which ``attn_combine_kernel`` merges, exactly the online-softmax
//: bookkeeping above -- so the arithmetic is unchanged, only its parallelism is.
constexpr int ATTN_SPLIT = 8;

__device__ __forceinline__ float bf16f(unsigned short h) {
    return __uint_as_float(((unsigned)h) << 16);
}

__device__ __forceinline__ unsigned short fbf16(float f) {
    unsigned u = __float_as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);   // round to nearest even
    return (unsigned short)(u >> 16);
}

// Block-wide max / sum via a shared scratch of ``blockDim.x`` floats. Every
// thread of the block must call it (it ends on a ``__syncthreads()``), and
// ``red`` is clobbered.
template <bool IsMax>
__device__ __forceinline__ float block_reduce(float v, float* red) {
    const int t = threadIdx.x;
    red[t] = v;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (t < s) red[t] = IsMax ? fmaxf(red[t], red[t + s]) : red[t] + red[t + s];
        __syncthreads();
    }
    return red[0];
}

// @brief Row index of key ``s`` for kv head ``g``, dense or paged.
//
// The dense cache is head-major ([nkv, S, hd] -> row = g*S + s) and the paged
// pool is token-major ([slots, nkv, hd] -> row = slot*nkv + g), so one formula
// covers both and a null ``pos2slot`` means "dense". Everything downstream (k, v
// and both scales) is then indexed by that single row id.
//
// @param pos2slot Device-side absolute position -> pool slot map, or null.
// @param stride   S for the dense layout, nkv for the paged one.
// @return The flat row id shared by k, v and both per-token scales.
__device__ __forceinline__ int attn_row(int s, int g, const int64_t* pos2slot, int stride) {
    return pos2slot ? (int)((long)pos2slot[s] * stride + g) : g * stride + s;
}

// @brief One block per (query head, key-range slice); split-K pass of the decode
//        attention.
//
// @param pos2slot Device position -> slot map, or null for the dense layout.
// @param stride   S (dense) or nkv (paged); see ``attn_row``.
// @param part_acc [nh, split, hd] fp32 online-softmax numerator, one per slice.
// @param part_ml  [nh, split, 2] fp32 (running max, running sum).
// @param pos_ptr  Device int64 position; read here because the decode graph
//                 freezes scalar kernel arguments at capture time.
__global__ void attn_decode_kernel(const unsigned short* __restrict__ q,
                                   const unsigned short* __restrict__ gate,
                                   const int8_t* __restrict__ kc,
                                   const int8_t* __restrict__ vc,
                                   const float* __restrict__ ks,
                                   const float* __restrict__ vs,
                                   const int64_t* __restrict__ pos2slot,
                                   int stride,
                                   float* __restrict__ part_acc,
                                   float* __restrict__ part_ml,
                                   const int64_t* __restrict__ pos_ptr,
                                   int hd, int rep, float scale, int split) {
    const int h = blockIdx.x;
    const int s = blockIdx.y;        // this block's slice of the key range
    const int g = h / rep;
    const int pos = (int)*pos_ptr;   // read at replay: the graph freezes arguments
    const int t = threadIdx.x;
    const int nt = blockDim.x;
    extern __shared__ float sh[];
    float* w = sh;                                        // [ATTN_KEY_BLOCK] weights
    int* rows = (int*)(sh + ATTN_KEY_BLOCK);              // [ATTN_KEY_BLOCK] key -> row
    float* qs = sh + ATTN_KEY_BLOCK + ATTN_KEY_BLOCK;     // [hd] this head's query
    float* red = qs + hd;                                 // [nt]

    for (int d = t; d < hd; d += nt) qs[d] = bf16f(q[(size_t)h * hd + d]);

    // Only positions 0..pos exist, so the window stops there instead of scanning
    // the whole max_seq (the eager einsum paid for all of it every step).  The
    // window is then cut into `split` slices; an empty slice contributes nothing
    // (its max stays at the sentinel and its sum at 0).
    const int nvalid = pos + 1;
    const int per = (nvalid + split - 1) / split;
    const int kfrom = s * per;
    const int kto = min(kfrom + per, nvalid);
    float m = -1e30f, l = 0.f, acc = 0.f;    // running max, running sum, this dim's acc
    for (int base = kfrom; base < kto; base += ATTN_KEY_BLOCK) {
        const int nb = min(ATTN_KEY_BLOCK, kto - base);
        for (int i = t; i < nb; i += nt) rows[i] = attn_row(base + i, g, pos2slot, stride);
        __syncthreads();

        // scores: dot(q, dequant k) * per-token scale * softmax scale
        float lmax = -1e30f;
        for (int i = t; i < nb; i += nt) {
            const int8_t* kr = kc + (size_t)rows[i] * hd;
            float dot = 0.f;
            for (int d = 0; d < hd; ++d) dot += qs[d] * (float)kr[d];
            const float sc = dot * ks[rows[i]] * scale;
            w[i] = sc;
            lmax = fmaxf(lmax, sc);
        }
        const float bmax = block_reduce<true>(lmax, red);

        // online softmax: rescale what has been accumulated, then add this block.
        // A block that is entirely masked leaves bmax at the sentinel, so mnew is
        // unchanged and alpha is exactly 1 (no -inf arithmetic anywhere).
        const float mnew = fmaxf(m, bmax);
        const float alpha = __expf(m - mnew);
        float lsum = 0.f;
        for (int i = t; i < nb; i += nt) {
            const float e = __expf(w[i] - mnew);
            w[i] = e;
            lsum += e;
        }
        l = l * alpha + block_reduce<false>(lsum, red);
        acc *= alpha;
        m = mnew;

        // weighted sum of dequant v (unscaled by 1/l until the end)
        if (t < hd) {
            float a = 0.f;
            for (int i = 0; i < nb; ++i)
                a += w[i] * vs[rows[i]] * (float)vc[(size_t)rows[i] * hd + t];
            acc += a;
        }
        __syncthreads();     // rows[] and w[] are reused by the next block
    }

    // publish this slice's partial for the combine pass (the gate is applied there)
    if (t == 0) {
        part_ml[(size_t)(h * split + s) * 2 + 0] = m;
        part_ml[(size_t)(h * split + s) * 2 + 1] = l;
    }
    if (t < hd) part_acc[(size_t)(h * split + s) * hd + t] = acc;
}

// @brief Merges the per-slice partials of one head and applies the output gate.
//
// The same online-softmax combination, now across slices instead of across key
// blocks. Every thread recomputes the global max (``split`` values) rather than
// staging it in shared memory -- one warp-wide reduction would cost more than the
// redundant loads.
//
// @param gate [nh, hd] bf16 output gate; the sigmoid is folded into the
//             normalised result.
// @param out  [nh, hd] bf16, written by the first ``hd`` threads.
__global__ void attn_combine_kernel(const float* __restrict__ part_acc,
                                    const float* __restrict__ part_ml,
                                    const unsigned short* __restrict__ gate,
                                    unsigned short* __restrict__ out,
                                    int hd, int split) {
    const int h = blockIdx.x;
    const int t = threadIdx.x;
    const float* ml = part_ml + (size_t)h * split * 2;
    float m = -1e30f;
    for (int s = 0; s < split; ++s) m = fmaxf(m, ml[s * 2]);
    if (t < hd) {
        float l = 0.f, a = 0.f;
        for (int s = 0; s < split; ++s) {
            const float alpha = __expf(ml[s * 2] - m);
            l += ml[s * 2 + 1] * alpha;
            a += part_acc[(size_t)(h * split + s) * hd + t] * alpha;
        }
        const float gv = bf16f(gate[(size_t)h * hd + t]);
        out[(size_t)h * hd + t] = fbf16(a / l / (1.f + __expf(-gv)));
    }
}

}  // namespace

// @brief Launch the dense decode attention (split pass + combine pass).
//
// @param q,gate [nh, hd] bf16 contiguous.
// @param kcache,vcache [nkv, S, hd] int8 dense cache, head-major.
// @param kscale,vscale [nkv, S] fp32 per-token dequant scales.
// @param pos    Device int64 position; only positions 0..pos are visited.
// @param part_acc,part_ml Caller-owned [nh, split, hd] / [nh, split, 2] fp32;
//              ``split`` is read from ``part_acc`` so capture sees a fixed shape.
// @return ``out``, [nh, hd] bf16, written in place for call chaining.
// @throws c10::Error (TORCH_CHECK) from the kernel launch check when the device
//         rejects the grid; there is no shape validation here, the caller owns
//         the layout contract.
torch::Tensor attn_decode(torch::Tensor q, torch::Tensor gate,
                          torch::Tensor kcache, torch::Tensor vcache,
                          torch::Tensor kscale, torch::Tensor vscale,
                          torch::Tensor out, torch::Tensor pos,
                          int64_t rep, double scale,
                          torch::Tensor part_acc, torch::Tensor part_ml) {
    const int nh = q.size(0), hd = q.size(1);
    const int S = kcache.size(1);            // dense layout: row = g*S + s
    const int split = part_acc.size(1);
    hipStream_t stream = at::cuda::getCurrentCUDAStream();
    const int threads = 256;
    const size_t smem = (size_t)(2 * ATTN_KEY_BLOCK + hd + threads) * sizeof(float);
    attn_decode_kernel<<<dim3(nh, split), threads, smem, stream>>>(
        reinterpret_cast<const unsigned short*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const unsigned short*>(gate.data_ptr<at::BFloat16>()),
        kcache.data_ptr<int8_t>(), vcache.data_ptr<int8_t>(),
        kscale.data_ptr<float>(), vscale.data_ptr<float>(),
        nullptr, S,
        part_acc.data_ptr<float>(), part_ml.data_ptr<float>(),
        pos.data_ptr<int64_t>(), hd, (int)rep, (float)scale, split);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    attn_combine_kernel<<<nh, threads, 0, stream>>>(
        part_acc.data_ptr<float>(), part_ml.data_ptr<float>(),
        reinterpret_cast<const unsigned short*>(gate.data_ptr<at::BFloat16>()),
        reinterpret_cast<unsigned short*>(out.data_ptr<at::BFloat16>()), hd, split);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// @brief Launch the paged decode attention: the K/V row of a logical position is
//        indirected through ``pos2slot``.
//
// That indirection has to live on the *device* because the decode step is a
// captured graph: a host-side block table would have to be re-baked into the
// graph every time a sequence appends a block, whereas a device map is rewritten
// before the next replay with no recapture. The pool is laid out token-major
// ([slots, nkv, hd], unlike the dense cache's head-major [nkv, S, hd]) so that one
// block's slots stay contiguous across heads and a copy-on-write of a block is a
// single contiguous copy.
//
// @param kpool,vpool [num_slots, nkv, hd] int8 views of one layer's block pool.
// @param kscale,vscale [num_slots, nkv] fp32 per-slot scales.
// @param pos2slot Device int64 [S] map from absolute position to pool slot; it
//                 must be rewritten before every replay.
// @param max_pos Unused: the window is streamed and its length comes from
//                ``pos``. Kept so the caller's signature does not change.
// @return ``out``, [nh, hd] bf16, written in place for call chaining.
// @throws c10::Error (TORCH_CHECK) from the kernel launch check when the device
//         rejects the grid; the caller owns the layout contract.
torch::Tensor attn_paged_decode(torch::Tensor q, torch::Tensor gate,
                                torch::Tensor kpool, torch::Tensor vpool,
                                torch::Tensor kscale, torch::Tensor vscale,
                                torch::Tensor pos2slot, torch::Tensor out,
                                torch::Tensor pos, int64_t max_pos,
                                int64_t rep, double scale,
                                torch::Tensor part_acc, torch::Tensor part_ml) {
    (void)max_pos;      // streamed window: its length is pos + 1 on the device
    const int nh = q.size(0), hd = q.size(1);
    const int nkv = kpool.size(1);           // paged layout: row = slot*nkv + g
    const int split = part_acc.size(1);
    hipStream_t stream = at::cuda::getCurrentCUDAStream();
    const int threads = 256;
    const size_t smem = (size_t)(2 * ATTN_KEY_BLOCK + hd + threads) * sizeof(float);
    attn_decode_kernel<<<dim3(nh, split), threads, smem, stream>>>(
        reinterpret_cast<const unsigned short*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const unsigned short*>(gate.data_ptr<at::BFloat16>()),
        kpool.data_ptr<int8_t>(), vpool.data_ptr<int8_t>(),
        kscale.data_ptr<float>(), vscale.data_ptr<float>(),
        pos2slot.data_ptr<int64_t>(), nkv,
        part_acc.data_ptr<float>(), part_ml.data_ptr<float>(),
        pos.data_ptr<int64_t>(), hd, (int)rep, (float)scale, split);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    attn_combine_kernel<<<nh, threads, 0, stream>>>(
        part_acc.data_ptr<float>(), part_ml.data_ptr<float>(),
        reinterpret_cast<const unsigned short*>(gate.data_ptr<at::BFloat16>()),
        reinterpret_cast<unsigned short*>(out.data_ptr<at::BFloat16>()), hd, split);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
