// @file attn.cu
// @brief Fused single-token attention over the int8 KV cache, dense and paged.
//
// Module: coffin/backends/corex/kernels. Bound as ``attn_decode`` / ``attn_paged`` /
// ``attn_paged_batch`` / ``attn_paged_multi_q`` and called only from
// ``backends/corex/__init__.py``.
//
// Why fused: the eager path materialises the whole KV window as bf16 -- two
// ``[nkv, S, hd]`` copies -- and runs the score einsum, the causal mask, the softmax,
// the value einsum and the output gate as separate kernels. There are 8
// full-attention layers of that per token. This kernel keeps the cache in int8,
// dequantises on the fly, and applies the per-token scale, the softmax and the gate in
// one pass, so no window intermediate is ever written.
//
// Constraints:
//   * Every reduction is a shared-memory tree, never ``__shfl_down_sync``. This part's
//     warp is 64 lanes, and a butterfly written for 32 would sum half the block and
//     return a plausible but wrong softmax denominator. The tree is sized from
//     ``blockDim`` at launch, so the block size stays a free parameter.
//   * The window is streamed in blocks with an online softmax (running max and sum plus
//     a rescaled accumulator), so shared memory does not grow with context length. A
//     staged ``[S]`` score vector is 1 MB at 262144 and would not launch at all.
//   * ``pos`` is read from device memory inside the kernel, never passed by value. The
//     decode step is captured into a graph, and a scalar kernel argument freezes at
//     capture time -- the replay would then compute the same position forever, which is
//     the stale-output bug COREX_PORT.md §0 records.
//   * Split-K is read from ``part_acc``'s axis rather than hardcoded, so
//     ``caps.attn_split`` and the caller's buffer stay the single source of truth. This
//     backend currently declares ``attn_split=1`` (no split); the kernel handles either
//     width, so raising the cap later needs no kernel change.
//   * Dense and paged share one kernel through the ``pos2slot`` indirection, where a
//     null map means the dense layout. One code path means the paged answer cannot
//     drift from the dense one, which is a property the conformance check relies on.
//   * The gate is applied in the combine pass, after normalisation, so it never enters
//     the softmax and cannot change the attention weights.
//   * INT8 codes stay signed and the per-(key, head) scale is applied on the way into
//     the score, matching the reference's dequantise-then-multiply order. Applying it
//     to the probabilities instead would be a different rounding.
//   * ``B * c`` draft queries are flattened onto the grid's z axis, because a grid is
//     3-D. ``draft`` is passed so the kernel can recover ``(row, query)`` from that flat
//     index without a division on the host.

#include "corex_ops.h"
#include "corex_scalar.h"

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

namespace {

//: Keys streamed per block. Bounds shared memory to a few KB plus one rescale of the
//: output accumulator per block, and is unrelated to the attention window length.
constexpr int KEY_BLOCK = 1024;

//: Sentinel for "no key seen yet". Large and finite so ``mnew - m`` stays exact for the
//: first block; ``-inf`` would turn the ``exp(m - mnew)`` rescale into a NaN.
constexpr float NO_MAX = -1e30f;

// @brief Block-wide max or sum through a shared tree.
//
// @param v This thread's contribution.
// @param red Shared scratch of ``blockDim.x`` floats; clobbered.
// @param is_max ``true`` for a max, ``false`` for a sum.
// @return The block-wide result, valid in every thread. Every thread must call it: it
//         ends on a barrier.
//
// The closing barrier is required. ``red[0]`` is read by every thread, and the next
// ``block_reduce`` call writes ``red[threadIdx.x]`` before its own barrier, so without
// this one a fast thread clobbers the result a slow thread has not read yet. The attention
// loop calls it twice per key block -- max, then sum, with the weight array in between --
// so the race is on the first block of the first token.
__device__ __forceinline__ float block_reduce(float v, float* red, bool is_max) {
    red[threadIdx.x] = v;
    __syncthreads();
    for (int step = blockDim.x >> 1; step > 0; step >>= 1) {
        if ((int)threadIdx.x < step) {
            red[threadIdx.x] = is_max ? fmaxf(red[threadIdx.x], red[threadIdx.x + step])
                                      : red[threadIdx.x] + red[threadIdx.x + step];
        }
        __syncthreads();
    }
    const float total = red[0];
    __syncthreads();
    return total;
}

// @brief Flat row id of key ``s`` for kv head ``g``, dense or paged.
//
// The dense cache is head-major (``[nkv, S, hd]``, row ``g*S + s``) and the paged pool
// is token-major (``[slots, nkv, hd]``, row ``slot*nkv + g``). One formula covers both,
// so k, v and both per-token scales are all indexed by the same row id.
//
// @param s Position within the window.
// @param g KV head.
// @param pos2slot Device position-to-slot map, or null for the dense layout.
// @param stride ``S`` for dense, ``nkv`` for paged.
// @return The flat row id shared by k, v and both scales.
__device__ __forceinline__ int cache_row(int s, int g, const int64_t* pos2slot,
                                         int stride) {
    return pos2slot ? (int)((long)pos2slot[s] * stride + g) : g * stride + s;
}

//: Layout selector, shared by every entry point. One struct keeps the argument list of
//: :func:`slice_launch` readable and makes the null-vs-null pairing of ``pos2slot`` and
//: ``ctx_lens`` explicit.
struct AttnLayout {
    //: Device position-to-slot map, or null for the dense cache.
    const int64_t* pos2slot;
    //: Device per-row key counts, or null to derive the window from ``pos``.
    const int32_t* ctx_lens;
    //: Device position; only read when ``ctx_lens`` is null.
    const int64_t* pos;
    //: Row stride: ``S`` dense, ``nkv`` paged.
    int stride;
    //: Window cap, i.e. the slot-map row stride.
    int max_seq;
};

// @brief One block per (query head, key slice, row); one slice of the online softmax.
//
// @param q [rows * nh, hd] bf16 queries, already rotated and scaled by the caller.
// @param kc,vc int8 cache; the layout is chosen by :struct:`AttnLayout`.
// @param ks,vs fp32 per-(slot, kv-head) dequant scales.
// @param block_mask uint8 ``[B, n_blocks]`` keep mask, or null for exact attention.
// @param part_acc [rows, nh, split, hd] fp32 partial numerators.
// @param part_ml [rows, nh, split, 2] fp32 partial (max, sum).
// @param draft Queries sharing a row's history; 1 for a plain decode.
__global__ void attn_slice_kernel(const uint16_t* __restrict__ q,
                                  const uint8_t* __restrict__ block_mask,
                                  const int8_t* __restrict__ kc,
                                  const int8_t* __restrict__ vc,
                                  const float* __restrict__ ks,
                                  const float* __restrict__ vs,
                                  AttnLayout layout, float* __restrict__ part_acc,
                                  float* __restrict__ part_ml, int nh, int hd, int rep,
                                  float scale, int split, int draft, int mask_block) {
    const int h = blockIdx.x;             // query head
    const int s = blockIdx.y;             // this block's slice of the key range
    const int flat = blockIdx.z;          // batch * draft, flattened
    const int b = draft > 1 ? flat / draft : flat;
    const int j = draft > 1 ? flat % draft : 0;
    const int g = h / rep;
    const int t = threadIdx.x;
    const int nt = blockDim.x;

    // Draft query ``j`` must see ``ctx + j + 1`` keys so it is identical to decoding
    // that token alone; a plain decode sees the position itself.
    const int ctx = layout.ctx_lens ? (int)layout.ctx_lens[b]
                                    : (int)*layout.pos + 1 + j;
    const int nvalid = ctx > layout.max_seq ? layout.max_seq : ctx;

    extern __shared__ float sh[];
    float* w = sh;                          // [KEY_BLOCK] scores, then weights
    int* rows = (int*)(sh + KEY_BLOCK);     // [KEY_BLOCK] key -> cache row
    float* qs = sh + 2 * KEY_BLOCK;         // [hd] this head's query
    float* red = qs + hd;                   // [nt] reduction scratch

    const uint16_t* qrow = q + ((size_t)flat * nh + h) * hd;
    for (int d = t; d < hd; d += nt) qs[d] = CxScalar<uint16_t>::to_float(qrow[d]);

    const int per = split > 1 ? (nvalid + split - 1) / split : nvalid;
    const int kfrom = s * per;
    const int kto = kfrom + per < nvalid ? kfrom + per : nvalid;

    // The speculative mask is per key block; resolving it once per key inside the score
    // loop keeps it out of the value pass.
    const uint8_t* mask_row = block_mask ? block_mask + (size_t)b * mask_block : nullptr;

    float m = NO_MAX;
    float l = 0.f;
    float acc = 0.f;

    for (int base = kfrom; base < kto; base += KEY_BLOCK) {
        const int nb = kto - base < KEY_BLOCK ? kto - base : KEY_BLOCK;
        for (int i = t; i < nb; i += nt) {
            rows[i] = cache_row(base + i, g, layout.pos2slot, layout.stride);
        }
        __syncthreads();

        float lmax = NO_MAX;
        for (int i = t; i < nb; i += nt) {
            const int8_t* kr = kc + (size_t)rows[i] * hd;
            float dot = 0.f;
            for (int d = 0; d < hd; ++d) dot += qs[d] * (float)kr[d];
            float sc = dot * ks[rows[i]] * scale;
            if (mask_row && !mask_row[(base + i) / mask_block]) sc = NO_MAX;
            w[i] = sc;
            lmax = fmaxf(lmax, sc);
        }
        const float bmax = block_reduce(lmax, red, true);

        // Online softmax: rescale what was accumulated, then fold this block in. A fully
        // masked block leaves ``bmax`` at the sentinel, so ``mnew == m`` and ``alpha`` is
        // exactly 1 -- no -inf arithmetic and no NaN.
        const float mnew = fmaxf(m, bmax);
        const float alpha = __expf(m - mnew);
        float lsum = 0.f;
        for (int i = t; i < nb; i += nt) {
            const float e = __expf(w[i] - mnew);
            w[i] = e;
            lsum += e;
        }
        l = l * alpha + block_reduce(lsum, red, false);
        acc *= alpha;
        m = mnew;

        if (t < hd) {
            float a = 0.f;
            for (int i = 0; i < nb; ++i) {
                a += w[i] * vs[rows[i]] * (float)vc[(size_t)rows[i] * hd + t];
            }
            acc += a;
        }
        __syncthreads();     // rows[] and w[] are reused by the next block
    }

    const size_t at = ((size_t)flat * nh + h) * split + s;
    if (t == 0) {
        part_ml[at * 2 + 0] = m;
        part_ml[at * 2 + 1] = l;
    }
    if (t < hd) part_acc[at * hd + t] = acc;
}

// @brief Merge one query's per-slice partials and apply the output gate.
//
// Every thread recomputes the global max over ``split`` values instead of staging it in
// shared memory: at ``split <= 8`` one redundant scan is cheaper than a barrier, and the
// number of slices is a cap the backend controls.
//
// @param gate [rows * nh, hd] bf16 output gates.
// @param out [rows * nh, hd] bf16, written in place.
__global__ void attn_combine_kernel(const float* __restrict__ part_acc,
                                    const float* __restrict__ part_ml,
                                    const uint16_t* __restrict__ gate,
                                    uint16_t* __restrict__ out, int nh, int hd,
                                    int split) {
    const int h = blockIdx.x;
    const int flat = blockIdx.y;
    const int t = threadIdx.x;
    const size_t at = ((size_t)flat * nh + h) * split;
    float m = NO_MAX;
    for (int s = 0; s < split; ++s) m = fmaxf(m, part_ml[at * 2 + s * 2]);
    if (t >= hd) return;
    float l = 0.f;
    float a = 0.f;
    for (int s = 0; s < split; ++s) {
        const float alpha = __expf(part_ml[at * 2 + s * 2] - m);
        l += part_ml[at * 2 + s * 2 + 1] * alpha;
        a += part_acc[(at + s) * hd + t] * alpha;
    }
    const size_t io = ((size_t)flat * nh + h) * hd + t;
    const float gv = CxScalar<uint16_t>::to_float(gate[io]);
    out[io] = CxScalar<uint16_t>::from_float(a / l / (1.f + __expf(-gv)));
}

//: Threads per attention block. One row's head is walked by ``hd`` lanes and the
//: reduction needs a wavefront multiple, so the block follows the head width.
int attn_threads(int hd) {
    int t = hd > 128 ? hd : 128;
    t = ((t + 63) / 64) * 64;
    return t > 512 ? 512 : t;
}

//: Shared-memory bytes one attention block needs.
size_t attn_smem(int hd, int threads) {
    return (size_t)(2 * KEY_BLOCK + hd + threads) * sizeof(float);
}

// @brief Launch the slice and combine passes for one attention call.
//
// Every entry point funnels through here, so the dense and paged paths cannot drift in
// thread count, shared-memory size or grid geometry.
//
// @param q [flat, nh, hd] bf16 queries.
// @param gate [flat, nh, hd] bf16 gates.
// @param kc,vc int8 cache in either layout.
// @param ks,vs fp32 dequant scales.
// @param out [flat, nh, hd] bf16 output, written in place.
// @param part_acc,part_ml Caller-owned split-K scratch.
// @param layout Cache addressing mode.
// @param mask Optional uint8 ``[B, n_blocks]`` keep mask.
// @return ``out``, for call chaining.
torch::Tensor slice_launch(torch::Tensor q, torch::Tensor gate, torch::Tensor kc,
                           torch::Tensor vc, torch::Tensor ks, torch::Tensor vs,
                           torch::Tensor out, torch::Tensor part_acc,
                           torch::Tensor part_ml, AttnLayout layout,
                           const torch::Tensor* mask, int flat, int nh, int hd, int rep,
                           float scale, int draft, int mask_block) {
    // The split axis is the second-from-last one, not axis 2: the dense layout is
    // ``[nh, split, hd]`` (split == 1, hd at 2) while the batched layout is
    // ``[B, nh, split, hd]``. Reading axis 2 launched the dense path with a
    // ``hd``-wide grid and wrote far past the caller's buffer, which zeroed the
    // output instead of failing a shape check.
    const int split = (int)part_acc.size(-2);
    const int threads = attn_threads(hd);
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const uint8_t* mask_ptr = mask == nullptr
        ? nullptr : reinterpret_cast<const uint8_t*>(mask->data_ptr());
    const int blocks = mask != nullptr ? mask_block : 0;
    attn_slice_kernel<<<dim3(nh, split, flat), threads, attn_smem(hd, threads), stream>>>(
        reinterpret_cast<const uint16_t*>(q.data_ptr()), mask_ptr,
        kc.data_ptr<int8_t>(), vc.data_ptr<int8_t>(),
        ks.data_ptr<float>(), vs.data_ptr<float>(), layout,
        part_acc.data_ptr<float>(), part_ml.data_ptr<float>(),
        nh, hd, rep, scale, split, draft, blocks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    attn_combine_kernel<<<dim3(nh, flat), threads, 0, stream>>>(
        part_acc.data_ptr<float>(), part_ml.data_ptr<float>(),
        reinterpret_cast<const uint16_t*>(gate.data_ptr()),
        reinterpret_cast<uint16_t*>(out.data_ptr()), nh, hd, split);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

}  // namespace

// @brief One decode token over the dense int8 KV cache.
//
// @param q [nh, hd] bf16 query, already rotated and scaled.
// @param gate [nh, hd] bf16 output gate; sigmoid is applied in the combine pass.
// @param kcache,vcache [nkv, S, hd] int8, head-major.
// @param kscale,vscale [nkv, S] fp32 per-token dequant scales.
// @param out [nh, hd] bf16, written in place.
// @param pos 1-element int64 tensor: the token index and the causal bound. Read on the
//        device, so a captured replay sees the live position.
// @param rep Query heads per KV head.
// @param scale ``hd ** -0.5``.
// @param part_acc [nh, split, hd] fp32 split-K numerator, caller-owned.
// @param part_ml [nh, split, 2] fp32 split-K (max, sum), caller-owned.
// @return ``out``, for call chaining.
// @throws c10::Error when a shape or dtype is not what the kernel indexes.
torch::Tensor attn_decode(torch::Tensor q, torch::Tensor gate, torch::Tensor kcache,
                          torch::Tensor vcache, torch::Tensor kscale,
                          torch::Tensor vscale, torch::Tensor out, torch::Tensor pos,
                          int64_t rep, double scale, torch::Tensor part_acc,
                          torch::Tensor part_ml) {
    TORCH_CHECK(q.dim() == 2, "query must be [nh, hd]");
    TORCH_CHECK(kcache.scalar_type() == torch::kChar, "cache must be int8");
    TORCH_CHECK(pos.scalar_type() == torch::kLong, "pos must be int64");
    const int nh = (int)q.size(0);
    const int hd = (int)q.size(1);
    const int S = (int)kcache.size(1);
    AttnLayout layout{nullptr, nullptr, pos.data_ptr<int64_t>(), S, S};
    return slice_launch(q, gate, kcache, vcache, kscale, vscale, out, part_acc, part_ml,
                        layout, nullptr, 1, nh, hd, (int)rep, (float)scale, 1, 0);
}

// @brief One decode token over a layer's paged int8 pool.
//
// The position-to-slot indirection has to live on the device: the decode step is a
// captured graph, and a host-side block table would have to be re-baked into the graph
// every time a sequence appends a block, whereas a device map is rewritten before the
// next replay with no recapture.
//
// @param q [nh, hd] bf16 query.
// @param gate [nh, hd] bf16 output gate.
// @param kpool,vpool [nslots, nkv, hd] int8, token-major.
// @param kscale,vscale [nslots, nkv] fp32 per-slot scales.
// @param pos2slot [max_pos] int64 device map; only the first ``pos + 1`` entries are
//        read, so the tail may hold another sequence's slots.
// @param out [nh, hd] bf16, written in place.
// @param pos 1-element int64 tensor: the token index and the causal bound.
// @param max_pos Window length, i.e. the row stride of the slot map.
// @param rep Query heads per KV head.
// @param scale ``hd ** -0.5``.
// @param part_acc [nh, split, hd] fp32 split-K numerator.
// @param part_ml [nh, split, 2] fp32 split-K (max, sum).
// @return ``out``, for call chaining.
// @throws c10::Error when a shape or dtype is not what the kernel indexes.
torch::Tensor attn_paged(torch::Tensor q, torch::Tensor gate, torch::Tensor kpool,
                         torch::Tensor vpool, torch::Tensor kscale,
                         torch::Tensor vscale, torch::Tensor pos2slot, torch::Tensor out,
                         torch::Tensor pos, int64_t max_pos, int64_t rep, double scale,
                         torch::Tensor part_acc, torch::Tensor part_ml) {
    TORCH_CHECK(q.dim() == 2, "query must be [nh, hd]");
    TORCH_CHECK(kpool.scalar_type() == torch::kChar, "pool must be int8");
    TORCH_CHECK(pos2slot.scalar_type() == torch::kLong, "slot map must be int64");
    const int nh = (int)q.size(0);
    const int hd = (int)q.size(1);
    const int nkv = (int)kpool.size(1);
    AttnLayout layout{pos2slot.data_ptr<int64_t>(), nullptr, pos.data_ptr<int64_t>(),
                      nkv, (int)max_pos};
    return slice_launch(q, gate, kpool, vpool, kscale, vscale, out, part_acc, part_ml,
                        layout, nullptr, 1, nh, hd, (int)rep, (float)scale, 1, 0);
}

// @brief One decode token for a bucket of rows, each with its own context length.
//
// The per-row length is what lets a bucket of mixed prompt lengths share one launch: a
// row reads only the first ``context_lens[b]`` entries of its own slot map, so the masked
// tail is never visited rather than visited and discarded.
//
// @param q [B, nh, hd] bf16 queries.
// @param gate [B, nh, hd] bf16 output gates.
// @param kpool,vpool [nslots, nkv, hd] int8, token-major.
// @param kscale,vscale [nslots, nkv] fp32 per-slot scales.
// @param slots [B, max_seq] int64 per-row slot maps.
// @param context_lens [B] int32 per-row key counts.
// @param out [B, nh, hd] bf16, written in place.
// @param max_seq Window length, i.e. the row stride of ``slots``.
// @param rep Query heads per KV head.
// @param scale ``hd ** -0.5``.
// @param part_acc [B, nh, split, hd] fp32 split-K numerator.
// @param part_ml [B, nh, split, 2] fp32 split-K (max, sum).
// @return ``out``, for call chaining.
// @throws c10::Error when a shape or dtype is not what the kernel indexes.
torch::Tensor attn_paged_batch(torch::Tensor q, torch::Tensor gate, torch::Tensor kpool,
                               torch::Tensor vpool, torch::Tensor kscale,
                               torch::Tensor vscale, torch::Tensor slots,
                               torch::Tensor context_lens, torch::Tensor out,
                               int64_t max_seq, int64_t rep, double scale,
                               torch::Tensor part_acc, torch::Tensor part_ml) {
    TORCH_CHECK(q.dim() == 3, "queries must be [B, nh, hd]");
    TORCH_CHECK(context_lens.scalar_type() == torch::kInt, "context lengths must be int32");
    const int rows = (int)q.size(0);
    const int nh = (int)q.size(1);
    const int hd = (int)q.size(2);
    const int nkv = (int)kpool.size(1);
    AttnLayout layout{slots.data_ptr<int64_t>(), context_lens.data_ptr<int32_t>(), nullptr,
                      nkv, (int)max_seq};
    return slice_launch(q.reshape({rows, nh, hd}), gate, kpool, vpool, kscale, vscale,
                        out, part_acc, part_ml, layout, nullptr, rows, nh, hd, (int)rep,
                        (float)scale, 1, 0);
}

// @brief Speculative verify: ``c`` draft queries per row share one pass over the history.
//
// Row ``b``, draft ``j`` is launched as its own grid slice with a window of
// ``ctx_lens[b] + j + 1`` keys, which is exactly the window decoding that token alone
// would see. Sharing the history pass is the win; the per-query window is what keeps the
// answer identical to the non-speculative path.
//
// @param q [B, c, nh, hd] bf16 draft queries.
// @param gate [B, c, nh, hd] bf16 output gates.
// @param kpool,vpool [nslots, nkv, hd] int8, token-major.
// @param kscale,vscale [nslots, nkv] fp32 per-slot scales.
// @param slots [B, max_seq] int64 per-row slot maps.
// @param ctx_lens [B] int32 per-row history length, excluding the drafts.
// @param out [B, c, nh, hd] bf16, written in place.
// @param max_seq Window length, i.e. the row stride of ``slots``.
// @param rep Query heads per KV head.
// @param scale ``hd ** -0.5``.
// @param part_acc [B*c, nh, split, hd] fp32 split-K numerator.
// @param part_ml [B*c, nh, split, 2] fp32 split-K (max, sum).
// @param block_mask Optional uint8 ``[B, n_blocks]`` keep mask (1 = keep) over key
//        blocks; ``None`` means exact full attention.
// @param mask_block Block size the mask is expressed in.
// @return ``out``, for call chaining.
// @throws c10::Error when a shape or dtype is not what the kernel indexes.
torch::Tensor attn_paged_multi_q(torch::Tensor q, torch::Tensor gate, torch::Tensor kpool,
                                 torch::Tensor vpool, torch::Tensor kscale,
                                 torch::Tensor vscale, torch::Tensor slots,
                                 torch::Tensor ctx_lens, torch::Tensor out,
                                 int64_t max_seq, int64_t rep, double scale,
                                 torch::Tensor part_acc, torch::Tensor part_ml,
                                 torch::Tensor block_mask, int64_t mask_block) {
    TORCH_CHECK(q.dim() == 4, "draft queries must be [B, c, nh, hd]");
    TORCH_CHECK(ctx_lens.scalar_type() == torch::kInt, "context lengths must be int32");
    const int rows = (int)q.size(0);
    const int draft = (int)q.size(1);
    const int nh = (int)q.size(2);
    const int hd = (int)q.size(3);
    const int nkv = (int)kpool.size(1);
    AttnLayout layout{slots.data_ptr<int64_t>(), ctx_lens.data_ptr<int32_t>(), nullptr,
                      nkv, (int)max_seq};
    // The mask is optional in the contract; an empty tensor is how "exact attention"
    // arrives, and treating it as a mask would zero every key.
    const bool masked = block_mask.defined() && block_mask.numel() > 0;
    if (masked) {
        TORCH_CHECK(block_mask.scalar_type() == torch::kByte, "mask must be uint8");
        TORCH_CHECK((int)block_mask.size(0) == rows, "mask row count must match the batch");
    }
    return slice_launch(q.reshape({rows * draft, nh, hd}), gate, kpool, vpool, kscale,
                        vscale, out, part_acc, part_ml, layout, masked ? &block_mask
                                                                      : nullptr,
                        rows * draft, nh, hd, (int)rep, (float)scale, draft,
                        (int)mask_block);
}
