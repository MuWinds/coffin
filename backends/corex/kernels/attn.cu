// @file attn.cu
// @brief Fused single-token attention over the int8 KV cache, plus the elementwise glue
//        that feeds it: the KV quantiser and the SwiGLU product.
//
// Module: coffin/backends/corex/kernels. Bound as ``attn_decode`` / ``attn_paged`` /
// ``attn_paged_batch`` / ``attn_paged_multi_q`` / ``kv_quant`` / ``silu_mul`` and called
// only from ``backends/corex/__init__.py``.
//
// Why fused: the eager path materialises the whole KV window as bf16 -- two
// ``[nkv, S, hd]`` copies -- and runs the score einsum, the causal mask, the softmax,
// the value einsum and the output gate as separate kernels. There are 8
// full-attention layers of that per token. This kernel keeps the cache in int8,
// dequantises on the fly, and applies the per-token scale, the softmax and the gate in
// one pass, so no window intermediate is ever written.
//
// ``kv_quant`` and ``silu_mul`` live here rather than in their own translation unit because
// they are the same kind of change to the same step: each replaces a six-or-seven launch
// torch chain in a full-attention or MLP layer with one launch. ``kv_quant`` also shares
// ``block_reduce`` with the attention kernel, and the quantiser's arithmetic is half of
// what ``attn_slice_kernel``'s reader assumes, so keeping the pair in one file keeps the
// agreement between them reviewable in one place.
//
// Constraints:
//   * Every reduction is a shared-memory tree, never ``__shfl_down_sync``. This part's
//     warp is 64 lanes, and a butterfly written for 32 would sum half the block and
//     return a plausible but wrong softmax denominator. The tree is sized from
//     ``blockDim`` at launch, so the block size stays a free parameter -- but it must be a
//     power of two, because the tree halves from ``blockDim.x`` and a 192-lane block would
//     drop its tail.
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
//   * The quantiser's ``1e-5`` floor and ``127`` divisor are a *contract* with that
//     reader, not a tuning choice. ``kv_quant`` reproduces the reference chain's operation
//     order and dtype exactly -- fp32 divide by the scale, ties-to-even rounding -- and any
//     "optimisation" of them desynchronises writer and reader without failing a shape.
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

// @brief Threads for one quantisation block, as a power of two.
//
// ``block_reduce`` halves from ``blockDim.x`` down, so it only covers the block when the
// block is a power of two; a 192-thread block would reduce 192, 96, 48, 24, 12, 6, 3, 1
// and silently drop the tail. The norm file's reduction has the same requirement and
// sizes its block from the row width instead, which happens to be a power of two there.
//
// @param N Row width, which is what the block has to cover.
// @return ``min(1024, round_up_pow2(max(64, N)))`` -- never below one wavefront, capped
//         at the device's block limit.
// @throws Nothing.
__host__ int kv_quant_threads(int N) {
    int t = 64;
    while (t < N && t < 1024) t <<= 1;
    return t;
}

// @brief Quantise every row of an ``[M, N]`` slab, one row per block.
//
// The arithmetic is the int8 KV cache contract, not an implementation choice, and it is
// reproduced here in torch's own operation order and dtypes:
//
//     xf = x.float()                       // widen bf16 exactly
//     s  = max(amax(xf), 1e-5) / 127.0     // fp32 divide
//     q  = round(xf / s)                   // fp32 divide, round half to even
//     q  = clamp(q, -127, 127)             // codes stay inside the symmetric range
//
// ``attn_slice_kernel`` multiplies the stored scale back, so any deviation here -- a
// different division, a different tie rule, a floor applied after the divide -- puts the
// writer and the reader on different scales and the attention output drifts without a
// shape ever failing. ``check_kv_quant.py`` compares this against the chain bit for bit
// rather than within a tolerance, for that reason.
//
// One row per block rather than one warp per row: ``hd`` is 256 in this model, so a row is
// 512 bytes and a block per row keeps the reduction inside a single barrier while giving
// each block enough lanes to cover a row in one pass. The batch and prefill call sites
// hand it up to a few thousand rows, so the grid is wide enough to fill the card either
// way.
//
// @param x [M, N] activations, bf16 as ``uint16_t`` or fp32; rows are pitched ``pitch``.
// @param out_q [M, N] int8 codes, densely packed, written in place.
// @param out_s [M] fp32 per-row scale, written in place.
// @param M Row count, i.e. the grid's x extent.
// @param N Row width.
// @param pitch Element pitch between rows of ``x``; ``>= N``.
// @return Nothing; both destinations are caller-owned.
// @throws Nothing on the device side. The wrapper validates the layout on the host.
template <typename T>
__global__ void kv_quant_kernel(const T* __restrict__ x, int8_t* __restrict__ out_q,
                                float* __restrict__ out_s, int N, long pitch) {
    extern __shared__ float red[];
    const int row = blockIdx.x;
    const int t = threadIdx.x;
    const T* xr = x + (size_t)row * pitch;

    // The reduction is ``fmaxf``, which drops a NaN operand instead of propagating it the
    // way torch's ``amax`` does. Activations reaching the KV cache are finite -- a NaN
    // here would already have produced a NaN attention output -- so the cheaper choice is
    // the one taken, and the divergence is stated rather than left to be discovered.
    float amax = 0.f;
    for (int i = t; i < N; i += blockDim.x) {
        const float v = fabsf(CxScalar<T>::to_float(xr[i]));
        if (v > amax) amax = v;
    }
    amax = block_reduce(amax, red, true);

    // ``clamp_min`` then ``/ 127`` in that order, in fp32. The floor keeps an all-zero row
    // from dividing by zero; it costs under one int8 step, so it cannot move a finite
    // value past what int8 rounds to anyway. It is an agreement with the reader, not a
    // tolerance, which is why it is spelled out here as a constant rather than folded.
    const float s = fmaxf(amax, 1e-5f) / 127.0f;

    int8_t* qr = out_q + (size_t)row * N;
    for (int i = t; i < N; i += blockDim.x) {
        // ``__float2int_rn`` is ``cvt.rni.s32.f32``: round to nearest, ties to even. That
        // is torch's ``.round()``. ``rintf`` would say the same thing but goes through a
        // float and then a cast, and the cast's out-of-range behaviour would be a second
        // thing to reason about. |x / s| is bounded by 127 by construction, so the int is
        // in range before the clamp; the clamp stays because "bounded by construction" is
        // the kind of claim that is wrong exactly once and then silently.
        int v = __float2int_rn(CxScalar<T>::to_float(xr[i]) / s);
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        qr[i] = (int8_t)v;
    }
    if (t == 0) out_s[row] = s;
}

// @brief ``out = silu(gate) * up`` over a pitched ``[M, W]`` view, one element per lane.
//
// The fused ``gate_up`` projection returns both halves from one weight read, so the only
// work between it and the down projection is this product. As ``aten::silu`` plus
// ``aten::mul`` it is two launches and two ``[intermediate]`` temporaries per MLP; there
// are 32 MLPs in a decode step, so it is 64 launches for one multiply-add per element.
//
// **The two roundings are not an accident.** ``aten::silu`` on a bf16 tensor returns bf16,
// so the chain rounds the activation once, and ``aten::mul`` rounds the product a second
// time. Computing ``silu`` in fp32 and rounding only the product is the natural
// implementation and it disagrees with torch on roughly a quarter of all elements -- which
// changes the logits and therefore the token ids. So the activation is narrowed back to the
// storage type before the multiply, exactly as the two-op chain does. Measured: with the
// intermediate narrowing, five independent 200k-element trials are bit-identical to
// ``F.silu(gate) * up``; without it, ~24% of elements differ.
//
// **The pitches are the point.** The projection is called as a fused group and the result
// is split back into its two halves with ``.split(dim=1)``, which on a 2-D result is a
// *column* view of the wider buffer: rows are ``pitch`` elements apart and ``pitch > W``.
// A flat, contiguous-only kernel would force the caller to materialise both halves first,
// which is two more launches and two more allocations -- strictly worse than the two torch
// ops this replaces. So the block walks ``(row, column)`` instead of a flat index and the
// pitches travel as arguments. On the decode path ``W`` is the whole split and the pitch
// equals it, so the same kernel serves both without a branch.
//
// ``out`` is allowed to alias ``up``, and every lane reads ``gate[r][c]`` and ``up[r][c]``
// before writing ``out[r][c]`` with no lane touching another's index, so a shared pitch
// makes it a pure in-place update. That is what lets the call site hand the projection's
// own buffer back and allocate nothing.
//
// @param gate [M, W] gate half; unit innermost stride.
// @param up [M, W] up half, same shape; unit innermost stride.
// @param out [M, W] destination; unit innermost stride, may alias ``up``.
// @param width ``W``, the columns each row contributes.
// @param g_pitch Element pitch between rows of ``gate``.
// @param u_pitch Element pitch between rows of ``up``.
// @param o_pitch Element pitch between rows of ``out``.
// @return Nothing; ``out`` is caller-owned.
// @throws Nothing on the device side. The wrapper validates the layout on the host.
template <typename T>
__global__ void silu_mul_kernel(const T* __restrict__ gate, const T* __restrict__ up,
                                T* __restrict__ out, int width, long g_pitch,
                                long u_pitch, long o_pitch) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= width) return;
    const long r = blockIdx.y;
    const long gi = r * g_pitch + c;
    const long ui = r * u_pitch + c;
    // The sigmoid is evaluated in fp32 -- bf16 ``1 + expf(-g)`` would lose the whole
    // mantissa of the denominator -- and then narrowed to the storage type, because that
    // is what ``aten::silu`` on a bf16 tensor does and what the second rounding below is
    // defined against.
    const float g = CxScalar<T>::to_float(gate[gi]);
    const T s = CxScalar<T>::from_float(g / (1.f + __expf(-g)));
    out[r * o_pitch + c] = CxScalar<T>::from_float(CxScalar<T>::to_float(s)
                                                   * CxScalar<T>::to_float(up[ui]));
}

// @brief Rotate the leading ``rotary_dim`` channels of every q/k head, one lane per channel.
//
// RoPE as a torch chain is ten to twelve launches per full-attention layer: two ``_to_copy``
// casts of the tables, four ``mul``, two ``cat`` for ``rotate_half``, two ``add`` and two
// ``cat`` to re-join the unrotated tail. Over 8 layers that is ~80-96 of the decode step's
// launches, all on tensors of at most ``[16, 256]`` -- essentially pure launch floor. One
// launch replaces all of it.
//
// **The four roundings are the contract**, and getting this wrong is silent. ``apply_rope``
// runs ``cos.to(q.dtype)`` first and then ``x*cos + rotate_half(x)*sin`` on bf16 tensors, so
// torch narrows: cos/sin once, each product once, and the sum once. Verified on the host
// against the real ``ops/rope.py``: with all four roundings the result is bit-identical, and
// each omission is measurable --
//   * fp32 cos/sin, everything else right:  296 of 1024 elements differ;
//   * fp32 arithmetic, one rounding at the end: 435 elements differ per token.
// Both change the logits. So the tables are narrowed here per element and each of the three
// operations is narrowed individually, rather than in one fp32 expression.
//
// ``rotate_half`` pairs channel ``i`` with ``i + rotary_dim/2``; that is a contract with
// the ``cat(half, half)`` layout the tables are built in (``ops/rope.py``). Only the leading
// ``rotary_dim`` channels rotate -- the trailing ``hd - rotary_dim`` pass through untouched,
// which partial-rotary checkpoints depend on.
//
// @param q [M, nh, hd] queries, bf16 as ``uint16_t``.
// @param k [M, nkv, hd] keys, same dtype.
// @param cos,sin [M, rotary_dim] fp32, the already-gathered rows for these tokens.
// @param q_out [M, nh, hd] rotated queries, written in place; must not alias ``q``.
// @param k_out [M, nkv, hd] rotated keys, written in place; must not alias ``k``.
// @param nh,nkv Head counts; they only shape the index arithmetic.
// @param hd Head width.
// @param rot ``rotary_dim``, the leading channels that rotate.
// @return Nothing; both destinations are caller-owned.
// @throws Nothing on the device side. The wrapper validates the layout on the host.
template <typename T>
__global__ void rope_apply_kernel(const T* __restrict__ q, const T* __restrict__ k,
                                  const float* __restrict__ cos,
                                  const float* __restrict__ sin,
                                  T* __restrict__ q_out, T* __restrict__ k_out,
                                  int nh, int nkv, int hd, int rot) {
    // Grid: x covers a head's channels, y is the token, z selects q (0) or k (1). Tokens and
    // the two tensors are therefore independent, and the decode path is simply ``M == 1``.
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= hd) return;
    const int m = blockIdx.y;
    const bool is_k = blockIdx.z != 0;
    const T* src = is_k ? k : q;
    T* dst = is_k ? k_out : q_out;
    const int heads = is_k ? nkv : nh;
    const int rot_half = rot >> 1;

    // Channel ``c`` inside the rotated prefix pairs with the channel ``rot_half`` away; past
    // the prefix it is copied through. The sign belongs to the *source* channel's side of
    // the split, because ``rotate_half`` negates its second half before pairing.
    const bool rotated = c < rot;
    const int pair = c < rot_half ? c + rot_half : c - rot_half;
    const float sign = (pair >= rot_half) ? -1.f : 1.f;
    // The table row is narrowed to the storage type before it is used, because that is what
    // ``apply_rope``'s ``cos.to(q.dtype)`` does. Keeping it fp32 here is the intuitive
    // choice and it changes ~29% of the elements.
    const T cv = CxScalar<T>::from_float(rotated ? cos[(size_t)m * rot + c] : 0.f);
    const T sv = CxScalar<T>::from_float(rotated ? sin[(size_t)m * rot + c] : 0.f);

    for (int h = 0; h < heads; ++h) {
        const size_t base = ((size_t)m * heads + h) * (size_t)hd;
        if (!rotated) {
            dst[base + c] = src[base + c];
            continue;
        }
        const float x = CxScalar<T>::to_float(src[base + c]);
        const float xr = sign * CxScalar<T>::to_float(src[base + pair]);
        // Three narrowings, one per operation the torch chain performs: product, product,
        // sum. Fusing them into one fp32 expression changes the logits.
        const T a = CxScalar<T>::from_float(x * CxScalar<T>::to_float(cv));
        const T b = CxScalar<T>::from_float(xr * CxScalar<T>::to_float(sv));
        dst[base + c] = CxScalar<T>::from_float(CxScalar<T>::to_float(a)
                                               + CxScalar<T>::to_float(b));
    }
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

// @brief RoPE for ``M`` token rows into caller buffers, in one launch.
//
// Replaces the ``ops/rope.py::apply_rope`` torch chain, which is ten to twelve launches per
// full-attention layer -- two table casts, four multiplies, two ``cat`` for ``rotate_half``,
// two adds and two rejoining ``cat``s -- over tensors of at most ``[16, 256]``. Eight layers
// per decode step, so ~80-96 launches that move a few KB each. One launch here.
//
// @param q [M, nh, hd] queries, bf16.
// @param k [M, nkv, hd] keys, same dtype.
// @param cos,sin [M, rotary_dim] fp32, the rows for exactly these tokens.
// @param q_out [M, nh, hd] rotated queries, written in place; must not alias ``q``.
// @param k_out [M, nkv, hd] rotated keys, written in place; must not alias ``k``.
// @param rotary_dim Channels to rotate; the rest are copied through.
// @return ``q_out``, for call chaining.
// @throws c10::Error on a shape, dtype or layout the kernel cannot index. Aliasing is not
//         detected: the model always writes fresh buffers, and a silent in-place rotation
//         would read already-rotated values for the paired half.
torch::Tensor rope_apply(torch::Tensor q, torch::Tensor k, torch::Tensor cos,
                         torch::Tensor sin, torch::Tensor q_out, torch::Tensor k_out,
                         int64_t rotary_dim) {
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3, "q and k must be [M, heads, hd]");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && q_out.is_contiguous()
                && k_out.is_contiguous(), "q, k, q_out and k_out must be contiguous");
    TORCH_CHECK(q.sizes() == q_out.sizes() && k.sizes() == k_out.sizes(),
                "outputs must match their inputs' shapes");
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16 && k.scalar_type() == q.scalar_type()
                && q_out.scalar_type() == q.scalar_type()
                && k_out.scalar_type() == q.scalar_type(),
                "q, k and both outputs must be bf16");
    TORCH_CHECK(cos.scalar_type() == torch::kFloat && sin.scalar_type() == torch::kFloat,
                "the cos/sin rows must be fp32; narrowing them at the table would lose "
                "angle precision at long context");
    const int M = (int)q.size(0);
    const int nh = (int)q.size(1);
    const int hd = (int)q.size(2);
    const int nkv = (int)k.size(1);
    const int rot = (int)rotary_dim;
    TORCH_CHECK(k.size(0) == M && k.size(2) == hd, "q and k disagree about M or hd");
    TORCH_CHECK(rot > 0 && rot <= hd && (rot & 1) == 0,
                "rotary_dim must be even and no wider than a head, got ", rot);
    TORCH_CHECK(cos.dim() == 2 && cos.size(0) == M && cos.size(1) == rot
                && sin.sizes() == cos.sizes(),
                "cos/sin must be [M, rotary_dim] fp32");
    if (M == 0) return q_out;
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    // One lane per channel of one head: 256 lanes covers this model's head_dim in a single
    // block, and the grid's y/z axes take the tokens and the q/k pair, so nothing about the
    // launch depends on the sequence.
    const int threads = hd < 64 ? 64 : (hd > 512 ? 512 : hd);
    const dim3 grid((unsigned)((hd + threads - 1) / threads), (unsigned)M, 2u);
    rope_apply_kernel<uint16_t><<<grid, threads, 0, stream>>>(
        reinterpret_cast<const uint16_t*>(q.data_ptr()),
        reinterpret_cast<const uint16_t*>(k.data_ptr()), cos.data_ptr<float>(),
        sin.data_ptr<float>(), reinterpret_cast<uint16_t*>(q_out.data_ptr()),
        reinterpret_cast<uint16_t*>(k_out.data_ptr()), nh, nkv, hd, rot);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return q_out;
}

// @brief Symmetric int8 quantisation of ``[M, N]``, one scale per row, in one launch.
//
// Replaces the torch chain ``abs -> amax -> clamp_min -> div -> round -> clamp_ -> to``
// that ``FullAttn._quant`` runs for the int8 KV cache. The chain is six or seven
// launches; at this card's 5.75 us floor it costs about 82 us for a few microseconds of
// arithmetic, and it runs twice per full-attention layer, so about 100 launches of the
// decode step are this. One block per row, one launch.
//
// Both destinations are caller-owned and written in place. That is not only the capture
// contract: the model keeps one int8 buffer and one scale buffer per layer and reuses
// them every step, where the chain allocated five fresh tensors per call.
//
// @param x [M, N] activations, bf16 or fp32, unit innermost stride; rows may be pitched.
// @param out_q [M, N] int8, written in place.
// @param out_s [M] fp32 per-row scale, written in place.
// @return ``out_q``, for call chaining.
// @throws c10::Error on a dtype or layout the kernel cannot index, or destinations whose
//         shape disagrees with ``x``. The kernel writes both buffers without bounds
//         checks, so a mismatched destination would corrupt memory rather than fail.
torch::Tensor kv_quant(torch::Tensor x, torch::Tensor out_q, torch::Tensor out_s) {
    TORCH_CHECK(x.dim() == 2, "activation must be 2-D [M, N]");
    // The pitched case is a column slice of a wider fused buffer, which is never
    // contiguous; the row pitch is what the kernel indexes by, so unit innermost stride is
    // the real requirement.
    TORCH_CHECK(x.stride(-1) == 1, "activation must have unit innermost stride");
    const int M = (int)x.size(0), N = (int)x.size(1);
    TORCH_CHECK(out_q.scalar_type() == torch::kChar, "q destination must be int8");
    TORCH_CHECK(out_s.scalar_type() == torch::kFloat, "scale destination must be fp32");
    TORCH_CHECK(out_q.dim() == 2 && out_q.is_contiguous()
                && out_q.size(0) == M && out_q.size(1) == N,
                "q destination must be a contiguous [M, N] buffer");
    TORCH_CHECK(out_s.dim() == 1 && out_s.is_contiguous() && out_s.size(0) == M,
                "scale destination must be a contiguous [M] buffer");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 || x.scalar_type() == torch::kFloat,
                "activation must be bf16 or fp32");
    const long pitch = (long)x.stride(0);
    TORCH_CHECK(pitch >= N, "row pitch ", pitch, " is narrower than the row ", N);

    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const int threads = kv_quant_threads(N);
    const size_t shmem = (size_t)threads * sizeof(float);
    if (x.scalar_type() == torch::kBFloat16) {
        kv_quant_kernel<uint16_t><<<M, threads, shmem, stream>>>(
            reinterpret_cast<const uint16_t*>(x.data_ptr()), out_q.data_ptr<int8_t>(),
            out_s.data_ptr<float>(), N, pitch);
    } else {
        kv_quant_kernel<float><<<M, threads, shmem, stream>>>(
            x.data_ptr<float>(), out_q.data_ptr<int8_t>(), out_s.data_ptr<float>(), N,
            pitch);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out_q;
}

// @brief ``out = silu(gate) * up`` in one launch, in place when the caller wants it.
//
// Replaces ``F.silu(gate) * up``, which is ``aten::silu`` plus ``aten::mul``: two launches
// and two fresh ``[intermediate]`` temporaries per MLP, 32 MLPs per decode step. One launch
// over the product, and because ``out`` may alias ``up`` the call site hands back the
// projection's own buffer and allocates nothing.
//
// All three operands may be the halves of one wider projection output, so each carries its
// own row pitch and none has to be contiguous. That is not a general amenity: a ``.split``
// on the second axis of a ``[M, N]`` buffer -- which is exactly how the fused ``gate_up``
// result becomes ``gate`` and ``up`` -- produces a strided column view, and requiring
// contiguity would make the caller copy both halves back into dense buffers. Two launches
// to reach the same arithmetic as the two torch ops it replaces is not a saving.
//
// @param gate [M, W] or [n] gate half; unit innermost stride.
// @param up Up half, same shape and dtype as ``gate``; unit innermost stride.
// @param out Destination, same shape and dtype; may alias ``up``.
// @return ``out``, for call chaining.
// @throws c10::Error when a shape or dtype disagrees. Overlap other than ``out == up`` is
//         not detected here, because a partial overlap is a caller bug the kernel cannot
//         serve correctly and the model never forms one.
torch::Tensor silu_mul(torch::Tensor gate, torch::Tensor up, torch::Tensor out) {
    TORCH_CHECK(gate.scalar_type() == up.scalar_type() && gate.scalar_type() == out.scalar_type(),
                "gate, up and out must share a dtype");
    TORCH_CHECK(gate.sizes() == up.sizes() && gate.sizes() == out.sizes(),
                "gate, up and out must have the same shape");
    TORCH_CHECK(gate.scalar_type() == torch::kBFloat16 || gate.scalar_type() == torch::kFloat,
                "operands must be bf16 or fp32");
    // The kernels index with unit innermost stride plus an explicit row pitch. Anything
    // else -- a transposed view, say -- has no legal [M, W] walk here.
    TORCH_CHECK(gate.stride(-1) == 1 && up.stride(-1) == 1 && out.stride(-1) == 1,
                "gate, up and out must have unit innermost stride");
    const int width = (int)gate.size(-1);
    if (width == 0) return out;
    // A 1-D operand is one row: the decode path splits a [N] vector, where the pitch and the
    // width coincide and the row count is 1.
    const long rows = gate.dim() <= 1 ? 1 : (long)gate.numel() / width;
    const long g_pitch = gate.dim() <= 1 ? (long)width : (long)gate.stride(-2);
    const long u_pitch = up.dim() <= 1 ? (long)width : (long)up.stride(-2);
    const long o_pitch = out.dim() <= 1 ? (long)width : (long)out.stride(-2);
    TORCH_CHECK(g_pitch >= width && u_pitch >= width && o_pitch >= width,
                "row pitch narrower than the row width");
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    // One lane per column, one grid row per operand row. The width is the projection's
    // intermediate size (18432 in this model) so the grid is wide in x and only as tall as
    // the row count, which is 1 for decode and a few thousand for prefill. No reduction, no
    // shared memory, no barrier -- which is why this kernel can be this simple.
    const int threads = 256;
    const dim3 grid((unsigned)((width + threads - 1) / threads), (unsigned)rows);
    if (gate.scalar_type() == torch::kBFloat16) {
        silu_mul_kernel<uint16_t><<<grid, threads, 0, stream>>>(
            reinterpret_cast<const uint16_t*>(gate.data_ptr()),
            reinterpret_cast<const uint16_t*>(up.data_ptr()),
            reinterpret_cast<uint16_t*>(out.data_ptr()), width, g_pitch, u_pitch, o_pitch);
    } else {
        silu_mul_kernel<float><<<grid, threads, 0, stream>>>(
            gate.data_ptr<float>(), up.data_ptr<float>(), out.data_ptr<float>(), width,
            g_pitch, u_pitch, o_pitch);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
