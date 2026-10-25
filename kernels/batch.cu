// @file batch.cu
// @brief Batched decode kernels: one launch for B sequences instead of B launches.
//
// Module: coffin/kernels. The batch half of the decode path; ``hip_ops.py``
// loads it into the same torch extension as ``attn.cu`` / ``linear_attn.cu`` /
// ``delta.cu``, which keep the B = 1 and speculative-verify entry points.
//
// Why this file exists: the single-stream decode step runs four kernels per
// layer per sequence (paged attention, conv1d, delta prologue, delta update).
// With a static batch of B sequences the *work* is B times larger but the step
// should still cost about one launch per kernel, because at decode sizes the
// step is latency/launch-bound: the per-row kernels launched 2-4 blocks per
// layer and the grid never came close to filling the card.
//
// Constraints:
//   * A row of width 1 must reproduce the single-row kernel bit for bit -- same
//     key blocking, same online-softmax partials, same fp32 state update order.
//     The two paths are A/B-compared, so a divergence is a bug on one side.
//   * Rows never share state: one block (or warp) owns a row, and there is no
//     cross-block synchronisation.
//   * q/k/v column slices of the fused projection output take their strides
//     from the tensor; assuming contiguity there reads the wrong third.
//   * The per-row key loop is bounded by ``context_lens[b]``, that row's own
//     window, never by ``max_seq``: a batch of short sequences must not pay for
//     the longest one, which is the whole point of batching at mixed positions.
//   * LDS budget is 64 KB per CU, so the attention row window is streamed in
//     ATTN_KEY_BLOCK keys and never staged whole.
//
// Row-major batch layout, all read-only except the caller-owned outputs:
//
//   attention      q/gate [B, nh, hd] bf16, slots [B, smax] int64,
//                  context_lens [B] int32, out [B, nh, hd] bf16,
//                  part_acc [B, nh, SPLIT, hd], part_ml [B, nh, SPLIT, 2] fp32
//   conv1d         x [B, conv_dim] bf16, state [B, conv_dim, 3] fp32,
//                  out [B, conv_dim] fp32
//   delta prologue q/k [B, nk, dk], v [B, nv, dv], b/a [B, nv] bf16,
//                  q_pro/k_pro [B, nv, dk], v_pro [B, nv, dv],
//                  beta/g [B, nv] fp32
//   delta update   state [B, nv, dk, dv], out [B, nv, dv] fp32

#include "hip_ops.h"

#include <c10/cuda/CUDAStream.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace {

//: Keys per streamed block, same bound as ``attn.cu``: 1024 keys is ~10 KB of
//: LDS, and the block size is what fixes the online-softmax rescale points.
constexpr int ATTN_KEY_BLOCK = 1024;

//: Key-range splits per query head, same as the single-row attention. The
//: single-row kernel uses 8 because one block per head is only ~16 blocks; with
//: B rows the grid is already (nh, split, B), so this only bounds the partial
//: buffer the caller allocates.
constexpr int ATTN_SPLIT_MAX = 64;

//: dk slices per state column in the delta update (matches ``DELTA_SPLIT`` in
//: ``delta.cu``; a mismatch would silently change the fp32 sum order).
constexpr int DELTA_SPLIT = 4;

__device__ __forceinline__ float bf16f(unsigned short h) {
    return __uint_as_float(((unsigned)h) << 16);
}

__device__ __forceinline__ unsigned short fbf16(float f) {
    unsigned u = __float_as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);   // round to nearest even
    return (unsigned short)(u >> 16);
}

// Block-wide max / sum via a shared scratch of ``blockDim.x`` floats.
// ``red`` must hold blockDim.x entries and is clobbered; every thread of the
// block must call this, since it ends on a ``__syncthreads()``.
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

// @brief Batched paged decode attention; one (head, slice, row) per block.
//
// Identity to the single-row kernel: block (h, s, b) owns the same slice
// ``[s*per, min((s+1)*per, nvalid))`` of head ``h``'s key range, where now
// ``nvalid = context_lens[b]``; the key rows of row ``b`` are indirected through
// that row's own slot map (``slots + b*smax``).  The partial (max, sum, acc) is
// published per (b, h, s) and merged by the combine kernel, which also applies
// the output gate.  So the arithmetic per (b, h) is exactly the single-row
// arithmetic, only the grid is (nh, split, B) instead of (nh, split).
//
// @param q            Query vectors, [B, nh, hd] bf16, row pitch ``qb`` and
//                     head pitch ``qh`` (a column slice of the fused projection
//                     output is not contiguous).
// @param slots        [B, smax] int64 slot map; ``smax`` is its row stride.
// @param context_lens [B] int32; the key loop of row b stops at its own value.
// @param part_acc     [B, nh, split, hd] fp32 online-softmax numerator.
// @param part_ml      [B, nh, split, 2] fp32 (running max, running sum).
// @param split        Key-range slices per head; the caller sizes the partials.
__global__ void attn_paged_batch_kernel(
        const unsigned short* __restrict__ q, int qb, int qh,
        const int8_t* __restrict__ kpool, const int8_t* __restrict__ vpool,
        const float* __restrict__ kscale, const float* __restrict__ vscale,
        const int64_t* __restrict__ slots, int smax,
        const int* __restrict__ context_lens,
        float* __restrict__ part_acc, float* __restrict__ part_ml,
        int nh, int nkv, int hd, int rep, float scale, int split) {
    const int b = blockIdx.z;
    const int h = blockIdx.x;
    const int s = blockIdx.y;
    const int g = h / rep;
    const int t = threadIdx.x;
    const int nt = blockDim.x;
    const int nvalid = context_lens[b];              // that row's window, not the max
    const int64_t* row_slots = slots + (size_t)b * smax;

    extern __shared__ float sh[];
    float* w = sh;                                        // [ATTN_KEY_BLOCK] weights
    int* rows = (int*)(sh + ATTN_KEY_BLOCK);              // [ATTN_KEY_BLOCK] key -> row
    float* qs = sh + ATTN_KEY_BLOCK + ATTN_KEY_BLOCK;     // [hd] this head's query
    float* red = qs + hd;                                 // [nt]

    const unsigned short* qp = q + (size_t)b * qb + (size_t)h * qh;
    for (int d = t; d < hd; d += nt) qs[d] = bf16f(qp[d]);

    float m = -1e30f, l = 0.f, acc = 0.f;    // running max, running sum, this dim's acc
    if (nvalid > 0) {
        const int per = (nvalid + split - 1) / split;
        const int kfrom = s * per;
        const int kto = min(kfrom + per, nvalid);
        for (int base = kfrom; base < kto; base += ATTN_KEY_BLOCK) {
            const int nb = min(ATTN_KEY_BLOCK, kto - base);
            for (int i = t; i < nb; i += nt)
                rows[i] = (int)(row_slots[base + i] * nkv + g);
            __syncthreads();

            // scores: dot(q, dequant k) * per-token scale * softmax scale
            float lmax = -1e30f;
            for (int i = t; i < nb; i += nt) {
                const int8_t* kr = kpool + (size_t)rows[i] * hd;
                float dot = 0.f;
                for (int d = 0; d < hd; ++d) dot += qs[d] * (float)kr[d];
                const float sc = dot * kscale[rows[i]] * scale;
                w[i] = sc;
                lmax = fmaxf(lmax, sc);
            }
            const float bmax = block_reduce<true>(lmax, red);

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

            if (t < hd) {
                float a = 0.f;
                for (int i = 0; i < nb; ++i)
                    a += w[i] * vscale[rows[i]] * (float)vpool[(size_t)rows[i] * hd + t];
                acc += a;
            }
            __syncthreads();     // rows[] and w[] are reused by the next block
        }
    }

    if (t == 0) {
        const size_t ml = ((size_t)b * nh + h) * split + s;
        part_ml[ml * 2 + 0] = m;
        part_ml[ml * 2 + 1] = l;
    }
    if (t < hd) {
        const size_t pa = ((size_t)b * nh + h) * split + s;
        part_acc[pa * hd + t] = acc;
    }
}

// @brief Merges the per-slice partials of one (b, head) and applies the gate.
//
// Same combination as the single-row combine kernel, indexed by batch. Every
// thread recomputes the global max (``split`` values) instead of staging it in
// shared memory: one warp-wide reduction would cost more than the redundant
// loads.
//
// @param gate Gate vectors, [B, nh, hd] bf16 with row pitch ``gb``, head pitch
//             ``gh``; the sigmoid is folded into the normalised output.
// @param out  [B, nh, hd] bf16 with row pitch ``ob``, head pitch ``oh``.
__global__ void attn_paged_batch_combine_kernel(
        const float* __restrict__ part_acc, const float* __restrict__ part_ml,
        const unsigned short* __restrict__ gate, int gb, int gh,
        unsigned short* __restrict__ out, int ob, int oh,
        int hd, int split) {
    const int h = blockIdx.x;
    const int b = blockIdx.y;
    const int t = threadIdx.x;
    const size_t base = (size_t)b * gridDim.x + h;
    const float* ml = part_ml + base * split * 2;
    float m = -1e30f;
    for (int s = 0; s < split; ++s) m = fmaxf(m, ml[s * 2]);
    if (t >= hd) return;
    float l = 0.f, a = 0.f;
    for (int s = 0; s < split; ++s) {
        const float alpha = __expf(ml[s * 2] - m);
        l += ml[s * 2 + 1] * alpha;
        a += part_acc[base * split * hd + (size_t)s * hd + t] * alpha;
    }
    const float gv = bf16f(gate[(size_t)b * gb + (size_t)h * gh + t]);
    out[(size_t)b * ob + (size_t)h * oh + t] = fbf16(a / l / (1.f + __expf(-gv)));
}

// @brief Batched depthwise causal conv1d (kernel 4) + SiLU.
//
// Same per-channel recurrence as ``conv1d_silu_kernel``; the grid's y axis is
// the row, so every (b, c) pair has its own state ring and no cross-row traffic.
// The state shift is read-then-write inside one thread, so it needs no barrier.
//
// @param state [B, C, 3] fp32 ring of the previous three inputs, shifted in
//              place with row pitch ``lds``.
// @param ldx   Row pitch of ``x``; the conv output is a column slice of the
//              fused projection, so rows are not packed.
// @param ldo   Row pitch of ``out``.
__global__ void conv1d_silu_batch_kernel(const float* __restrict__ w,
                                         const unsigned short* __restrict__ x,
                                         float* __restrict__ state,
                                         float* __restrict__ out, int C,
                                         int ldx, int ldo, int lds) {
    const int b = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float* s = state + (size_t)b * lds + (size_t)c * 3;
    const float* wc = w + (size_t)c * 4;
    const float x3 = bf16f(x[(size_t)b * ldx + c]);
    const float acc = fmaf(s[0], wc[0], fmaf(s[1], wc[1], fmaf(s[2], wc[2], x3 * wc[3])));
    out[(size_t)b * ldo + c] = acc / (1.f + __expf(-acc));     // silu
    s[0] = s[1];
    s[1] = s[2];
    s[2] = x3;
}

// Block-wide sum of two values at once -- q and k share one reduction pass
// because they are always normalised together.
__device__ __forceinline__ float2 block_sum2(float a, float b, float2* red) {
    const int t = threadIdx.x;
    red[t] = make_float2(a, b);
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (t < s) {
            red[t].x += red[t + s].x;
            red[t].y += red[t + s].y;
        }
        __syncthreads();
    }
    return red[0];
}

// @brief Batched gated-delta prologue: per-value-head q/k L2 norm, q scale,
//        head expansion and beta/g.
//
// Block (vh, b) does one value head of row b, exactly as the single-row kernel
// does one value head of the only row; ``rep`` maps value head vh to key head
// ``vh / rep`` (repeat_interleave order).
//
// @param q_raw,k_raw fp32 [B, nk, dk] with row pitches ``ldq`` / ``ldk``; they
//                    are column slices of the conv output, so a value head's k
//                    head may be re-read ``rep`` times.
// @param qscale     Scale folded into the normalised q (softmax scale * 1/sqrt).
// @param rep        Value heads per key head.
__global__ void delta_prologue_batch_kernel(
        const float* __restrict__ q_raw, int ldq,
        const float* __restrict__ k_raw, int ldk,
        const float* __restrict__ v_raw, int ldv,
        const unsigned short* __restrict__ b_raw,
        const unsigned short* __restrict__ a_raw,
        const float* __restrict__ neg_expA,
        const float* __restrict__ dt_bias,
        float* __restrict__ q_out, float* __restrict__ k_out,
        float* __restrict__ v_out,
        float* __restrict__ beta, float* __restrict__ g,
        float qscale, int dk, int dv, int rep) {
    const int vh = blockIdx.x;
    const int bb = blockIdx.y;
    const int nv = gridDim.x;
    const int t = threadIdx.x;
    const int kh = vh / rep;
    const float* qr = q_raw + (size_t)bb * ldq + (size_t)kh * dk;
    const float* kr = k_raw + (size_t)bb * ldk + (size_t)kh * dk;

    const float qv = t < dk ? qr[t] : 0.f;
    const float kv = t < dk ? kr[t] : 0.f;
    __shared__ float2 red[256];
    const float2 ss = block_sum2(qv * qv, kv * kv, red);

    const size_t head = (size_t)bb * nv + vh;
    if (t < dk) {
        q_out[head * dk + t] = qv * rsqrtf(ss.x + 1e-6f) * qscale;
        k_out[head * dk + t] = kv * rsqrtf(ss.y + 1e-6f);
    }
    if (t < dv) v_out[head * dv + t] = v_raw[(size_t)bb * ldv + (size_t)vh * dv + t];
    if (t == 0) {
        const size_t idx = (size_t)bb * nv + vh;
        beta[idx] = 1.f / (1.f + __expf(-bf16f(b_raw[idx])));
        const float av = bf16f(a_raw[idx]) + dt_bias[vh];
        const float sp = av > 20.f ? av : __logf(1.f + __expf(av));   // softplus, no overflow
        g[idx] = neg_expA[vh] * sp;
    }
}

// @brief Batched gated-delta state update.
//
// Copy of ``delta_kernel``'s slice logic with the state offset by the row: one
// block per (head, column tile, row).  A thread owns column ``j`` of row ``b``
// and one quarter of the dk loop; nothing is shared across rows, so the only
// change is the base pointer of the state.  ``decay = exp(g)`` and ``rate =
// beta`` are per (row, head), staged in shared memory as in the single-row
// kernel.
//
// @param st  [B, nv, dk, dv] fp32 recurrent state, updated in place.
// @param out [B, nv, dv] fp32; written by the lane that owns dk slice 0.
__global__ void delta_batch_kernel(const float* __restrict__ q,
                                   const float* __restrict__ k,
                                   const float* __restrict__ v,
                                   const float* __restrict__ beta,
                                   const float* __restrict__ g,
                                   float* __restrict__ st,
                                   float* __restrict__ out, int dk, int dv) {
    const int head = blockIdx.x;
    const int bb = blockIdx.z;
    const int heads = gridDim.x;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int ii = lane & (DELTA_SPLIT - 1);              // dk slice
    const int jj = lane >> 2;                             // column within the warp
    const int cols_per_block = (blockDim.x >> 5) * (32 / DELTA_SPLIT);
    const int j = blockIdx.y * cols_per_block + warp * (32 / DELTA_SPLIT) + jj;
    const size_t hb = (size_t)bb * heads + head;

    extern __shared__ float shared[];
    float* ks = shared;
    float* qs = shared + dk;
    for (int i = threadIdx.x; i < dk; i += blockDim.x) {
        ks[i] = k[hb * dk + i];
        qs[i] = q[hb * dk + i];
    }
    __shared__ float decay, rate;
    if (threadIdx.x == 0) {
        decay = __expf(g[hb]);
        rate = beta[hb];
    }
    __syncthreads();

    if (j >= dv) return;                                  // whole 4-lane group exits together
    float* column = st + hb * (size_t)dk * dv + j;
    const float value = v[hb * dv + j];

    const int slice = dk / DELTA_SPLIT;
    const int i0 = ii * slice, i1 = i0 + slice;
    float kv_mem = 0.f;
    for (int i = i0; i < i1; ++i) {
        kv_mem = fmaf(column[(size_t)i * dv] * decay, ks[i], kv_mem);
    }
#pragma unroll
    for (int o = 1; o < DELTA_SPLIT; o <<= 1) kv_mem += __shfl_xor(kv_mem, o);

    const float delta = (value - kv_mem) * rate;
    float result = 0.f;
    for (int i = i0; i < i1; ++i) {
        const float s = fmaf(ks[i], delta, column[(size_t)i * dv] * decay);
        column[(size_t)i * dv] = s;
        result = fmaf(s, qs[i], result);
    }
#pragma unroll
    for (int o = 1; o < DELTA_SPLIT; o <<= 1) result += __shfl_xor(result, o);
    if (ii == 0) out[hb * dv + j] = result;
}

}  // namespace

// @brief Launch the batched paged attention (split pass + combine pass).
//
// @param q,gate     [B, nh, hd] bf16; inner stride must be 1 but the row/head
//                   pitches are honoured (fused-projection column slices).
// @param slots      [B, smax] int64 contiguous; smax comes from its row stride.
// @param context_lens [B] int32 contiguous, one window per row.
// @param max_seq    Unused: the window is streamed, so its length is no longer
//                   needed on the device. Kept so the caller's signature does
//                   not change.
// @param part_acc,part_ml Caller-owned fp32 partials [B, nh, split, hd] and
//                   [B, nh, split, 2]; ``split`` is read from ``part_acc``, so
//                   the shapes stay static across graph capture.
// @return ``out``, written in place and returned for call chaining.
// @throws c10::Error (TORCH_CHECK) when a shape, dtype or stride breaks the
//         kernel's indexing contract; the message names the offending tensor,
//         so fix the caller rather than the kernel.
torch::Tensor attn_paged_batch(torch::Tensor q, torch::Tensor gate,
                               torch::Tensor kpool, torch::Tensor vpool,
                               torch::Tensor kscale, torch::Tensor vscale,
                               torch::Tensor slots, torch::Tensor context_lens,
                               torch::Tensor out, int64_t max_seq,
                               int64_t rep, double scale,
                               torch::Tensor part_acc, torch::Tensor part_ml) {
    (void)max_seq;      // streamed window: the per-row length lives in context_lens
    TORCH_CHECK(q.dim() == 3 && gate.dim() == 3 && out.dim() == 3,
                "q/gate/out must be [B, nh, hd]");
    TORCH_CHECK(q.stride(2) == 1 && gate.stride(2) == 1 && out.stride(2) == 1,
                "unit inner stride required");
    const int B = q.size(0), nh = q.size(1), hd = q.size(2);
    const int nkv = kpool.size(1);
    TORCH_CHECK(gate.size(0) == B && gate.size(1) == nh && gate.size(2) == hd,
                "gate shape mismatch");
    TORCH_CHECK(out.size(0) == B && out.size(1) == nh && out.size(2) == hd,
                "out shape mismatch");
    TORCH_CHECK(slots.dim() == 2 && slots.size(0) == B && slots.stride(1) == 1,
                "slots must be [B, max_seq] int64");
    TORCH_CHECK(context_lens.dim() == 1 && context_lens.size(0) == B,
                "context_lens must be [B]");
    TORCH_CHECK(context_lens.scalar_type() == at::kInt, "context_lens must be int32");
    TORCH_CHECK(context_lens.is_contiguous() && slots.is_contiguous(),
                "context_lens/slots must be contiguous");
    TORCH_CHECK(part_acc.dim() == 4 && part_acc.size(0) == B && part_acc.size(1) == nh
                && part_acc.size(3) == hd && part_acc.is_contiguous()
                && part_ml.is_contiguous() && part_ml.dim() == 4
                && part_ml.size(0) == B && part_ml.size(1) == nh
                && part_ml.size(3) == 2 && part_acc.size(2) == part_ml.size(2),
                "part_acc/part_ml shape mismatch");
    const int split = part_acc.size(2);
    TORCH_CHECK(split >= 1 && split <= ATTN_SPLIT_MAX, "unsupported split");
    const int smax = (int)slots.stride(0);
    const int threads = 256;
    const int cthreads = hd > threads ? ((hd + 31) / 32) * 32 : threads;
    const size_t smem = (size_t)(2 * ATTN_KEY_BLOCK + hd + threads) * sizeof(float);
    hipStream_t stream = at::cuda::getCurrentCUDAStream();
    attn_paged_batch_kernel<<<dim3(nh, split, B), threads, smem, stream>>>(
        reinterpret_cast<const unsigned short*>(q.data_ptr<at::BFloat16>()),
        (int)q.stride(0), (int)q.stride(1),
        kpool.data_ptr<int8_t>(), vpool.data_ptr<int8_t>(),
        kscale.data_ptr<float>(), vscale.data_ptr<float>(),
        slots.data_ptr<int64_t>(), smax,
        context_lens.data_ptr<int32_t>(),
        part_acc.data_ptr<float>(), part_ml.data_ptr<float>(),
        nh, nkv, hd, (int)rep, (float)scale, split);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    attn_paged_batch_combine_kernel<<<dim3(nh, B), cthreads, 0, stream>>>(
        part_acc.data_ptr<float>(), part_ml.data_ptr<float>(),
        reinterpret_cast<const unsigned short*>(gate.data_ptr<at::BFloat16>()),
        (int)gate.stride(0), (int)gate.stride(1),
        reinterpret_cast<unsigned short*>(out.data_ptr<at::BFloat16>()),
        (int)out.stride(0), (int)out.stride(1), hd, split);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// @brief Launch the batched depthwise causal conv1d + SiLU.
//
// @param conv_w     [C, 4] fp32 taps (three history taps plus the current one).
// @param x          [B, C] bf16, unit inner stride.
// @param conv_state [B, C, 3] fp32 ring, shifted in place.
// @return ``out``, [B, C] fp32, written in place.
// @throws c10::Error (TORCH_CHECK) when a shape or inner stride breaks the
//         indexing contract; fix the caller.
torch::Tensor conv1d_silu_batch(torch::Tensor conv_w, torch::Tensor x,
                                torch::Tensor conv_state, torch::Tensor out) {
    TORCH_CHECK(x.dim() == 2 && x.stride(1) == 1, "x must be [B, C] with unit inner stride");
    const int B = x.size(0), C = x.size(1);
    TORCH_CHECK(out.size(0) == B && out.size(1) == C && out.stride(1) == 1,
                "out must be [B, C] fp32");
    TORCH_CHECK(conv_state.size(0) == B && conv_state.size(1) == C
                && conv_state.size(2) == 3 && conv_state.stride(2) == 1,
                "conv_state must be [B, C, 3]");
    hipStream_t stream = at::cuda::getCurrentCUDAStream();
    conv1d_silu_batch_kernel<<<dim3((C + 255) / 256, B), 256, 0, stream>>>(
        conv_w.data_ptr<float>(),
        reinterpret_cast<const unsigned short*>(x.data_ptr<at::BFloat16>()),
        conv_state.data_ptr<float>(), out.data_ptr<float>(), C,
        (int)x.stride(0), (int)out.stride(0), (int)conv_state.stride(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// @brief Launch the batched gated-delta prologue.
//
// @param q_raw,k_raw [B, nk, dk] fp32 with row strides honoured: they are column
//                   slices of the fused conv output, so the rows are not packed.
// @param v_raw      [B, nv, dv] fp32 with row stride honoured.
// @param b,a        [B, nv] bf16 raw projection outputs; ``beta = sigmoid(b)``
//                   and ``g = -exp(A) * softplus(a + dt_bias)``.
// @param q_out,k_out [B, nv, dk] fp32 contiguous, L2-normalised, q scaled.
// @param beta,g     [B, nv] fp32 contiguous, consumed by ``delta_batch``.
// @throws c10::Error (TORCH_CHECK) when a shape, dtype or stride breaks the
//         indexing contract; fix the caller.
void delta_prologue_batch(torch::Tensor q_raw, torch::Tensor k_raw, torch::Tensor v_raw,
                          torch::Tensor b, torch::Tensor a,
                          torch::Tensor neg_expA, torch::Tensor dt_bias,
                          torch::Tensor q_out, torch::Tensor k_out, torch::Tensor v_out,
                          torch::Tensor beta, torch::Tensor g,
                          double qscale, int64_t rep) {
    TORCH_CHECK(q_raw.dim() == 3 && k_raw.dim() == 3 && v_raw.dim() == 3,
                "q/k/v must be [B, heads, d]");
    TORCH_CHECK(q_raw.stride(2) == 1 && k_raw.stride(2) == 1 && v_raw.stride(2) == 1,
                "unit inner stride required");
    const int B = q_raw.size(0);
    const int nk = q_raw.size(1), dk = q_raw.size(2);
    const int nv = v_raw.size(1), dv = v_raw.size(2);
    TORCH_CHECK(q_raw.stride(1) == dk && k_raw.stride(1) == dk && v_raw.stride(1) == dv,
                "q/k/v heads must be packed");
    TORCH_CHECK(k_raw.size(0) == B && k_raw.size(1) == nk && k_raw.size(2) == dk,
                "k shape mismatch");
    TORCH_CHECK(v_raw.size(0) == B, "v batch mismatch");
    TORCH_CHECK(b.size(0) == B && b.size(1) == nv && a.size(0) == B && a.size(1) == nv,
                "b/a must be [B, nv]");
    TORCH_CHECK(q_out.is_contiguous() && k_out.is_contiguous() && v_out.is_contiguous()
                && beta.is_contiguous() && g.is_contiguous(),
                "outputs must be contiguous");
    hipStream_t stream = at::cuda::getCurrentCUDAStream();
    delta_prologue_batch_kernel<<<dim3(nv, B), 128, 0, stream>>>(
        q_raw.data_ptr<float>(), (int)q_raw.stride(0),
        k_raw.data_ptr<float>(), (int)k_raw.stride(0),
        v_raw.data_ptr<float>(), (int)v_raw.stride(0),
        reinterpret_cast<const unsigned short*>(b.data_ptr<at::BFloat16>()),
        reinterpret_cast<const unsigned short*>(a.data_ptr<at::BFloat16>()),
        neg_expA.data_ptr<float>(), dt_bias.data_ptr<float>(),
        q_out.data_ptr<float>(), k_out.data_ptr<float>(), v_out.data_ptr<float>(),
        beta.data_ptr<float>(), g.data_ptr<float>(),
        (float)qscale, dk, dv, (int)rep);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// @brief Launch the batched gated-delta state update.
//
// @param st  [B, heads, dk, dv] fp32 recurrent state, updated in place.
// @param q,k [B, heads, dk] fp32 contiguous, already normalised by the prologue.
// @param beta,g [B, heads] fp32 per-row decay and rate.
// @return ``out``, [B, heads, dv] fp32, written in place.
// @throws c10::Error (TORCH_CHECK) when an input is non-contiguous or ``dk`` is
//         not a multiple of DELTA_SPLIT (the 4-lane shuffle needs it).
torch::Tensor delta_batch(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                          torch::Tensor beta, torch::Tensor g,
                          torch::Tensor st, torch::Tensor out) {
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous()
                && beta.is_contiguous() && g.is_contiguous() && st.is_contiguous()
                && out.is_contiguous(),
                "contiguous inputs required");
    const int B = st.size(0), heads = st.size(1), dk = st.size(2), dv = st.size(3);
    TORCH_CHECK(q.size(0) == B && q.size(1) == heads && q.size(2) == dk,
                "q shape does not match the state");
    TORCH_CHECK(out.size(0) == B && out.size(1) == heads && out.size(2) == dv,
                "out shape does not match the state");
    TORCH_CHECK(dk % DELTA_SPLIT == 0, "dk must be a multiple of DELTA_SPLIT");
    const int threads = 128, cols_per_block = (threads / 32) * (32 / DELTA_SPLIT);
    const int blocks_y = (dv + cols_per_block - 1) / cols_per_block;
    hipStream_t stream = at::cuda::getCurrentCUDAStream();
    delta_batch_kernel<<<dim3(heads, blocks_y, B), threads, 2 * dk * sizeof(float),
                         stream>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
        beta.data_ptr<float>(), g.data_ptr<float>(),
        st.data_ptr<float>(), out.data_ptr<float>(), dk, dv);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// @brief Multi-query verify attention: c draft queries per row share one KV
//        history.
//
// Runs the *same* arithmetic as ``attn_paged_batch`` for every (row, query)
// pair -- identical split-K slices, identical key-block online softmax,
// identical combine -- so each query's output is bit-identical to calling the
// decode kernel for that token.  What is new is only the grid: the c queries of
// a row are adjacent ``blockIdx.z`` values, so the blocks that walk the same
// (b, head, slice) key range are launched together and the c passes over the
// history hit L2 instead of re-reading it from HBM. At long context that turns
// c HBM passes into one, which is the verify win.
//
// @param block_mask Optional uint8 [B, nblocks] keep-mask, 1 = keep; an
//                   undefined/empty tensor means exact full attention. Masked
//                   keys get score -inf (weight exactly 0), the sparsity
//                   semantics a sparse-decode caller wants -- and because the
//                   weight is 0 the V pass can skip their load.
// @param mask_block Tokens per mask block. 64 takes a shift, other values a
//                   divide, so both are specialised inside the key loop.
// @param ctx_lens   [B] int32 history length; query j sees ctx_lens[b] + j + 1
//                   keys, which is the nvalid the per-token kernel would get.
__global__ void attn_paged_multi_q_kernel(
        const unsigned short* __restrict__ q, int qb, int qh,
        const int8_t* __restrict__ kpool, const int8_t* __restrict__ vpool,
        const float* __restrict__ kscale, const float* __restrict__ vscale,
        const int64_t* __restrict__ slots, int smax,
        const int* __restrict__ ctx_lens,
        const unsigned char* __restrict__ block_mask, int nblocks, int mask_block,
        float* __restrict__ part_acc, float* __restrict__ part_ml,
        int c, int nh, int nkv, int hd, int rep, float scale, int split) {
    // Block order matters for the L2 reuse: x is the query index and z packs
    // (row, slice), so the c blocks that walk the same history slice are
    // consecutive in launch order and hit in L2.
    const int j = blockIdx.x;
    const int h = blockIdx.y;
    const int bs = blockIdx.z;
    const int b = bs / split;
    const int s = bs - b * split;
    const int z = b * c + j;
    const int g = h / rep;
    const int t = threadIdx.x;
    const int nt = blockDim.x;
    // Query j is at position ctx+j, so it sees history plus the first j+1 new
    // tokens -- the same nvalid the per-token kernel would get for that token.
    const int nvalid = ctx_lens[b] + j + 1;
    const int64_t* row_slots = slots + (size_t)b * smax;
    const unsigned char* row_mask = block_mask ? block_mask + (size_t)b * nblocks
                                               : nullptr;

    extern __shared__ float sh[];
    float* w = sh;                                        // [ATTN_KEY_BLOCK] weights
    int* rows = (int*)(sh + ATTN_KEY_BLOCK);              // [ATTN_KEY_BLOCK] key -> row
    float* qs = sh + ATTN_KEY_BLOCK + ATTN_KEY_BLOCK;     // [hd] this head's query
    float* red = qs + hd;                                 // [nt]

    const unsigned short* qp = q + (size_t)z * qb + (size_t)h * qh;
    for (int d = t; d < hd; d += nt) qs[d] = bf16f(qp[d]);

    float m = -1e30f, l = 0.f, acc = 0.f;
    if (nvalid > 0) {
        const int per = (nvalid + split - 1) / split;
        const int kfrom = s * per;
        const int kto = min(kfrom + per, nvalid);
        for (int base = kfrom; base < kto; base += ATTN_KEY_BLOCK) {
            const int nb = min(ATTN_KEY_BLOCK, kto - base);
            for (int i = t; i < nb; i += nt)
                rows[i] = (int)(row_slots[base + i] * nkv + g);
            __syncthreads();

            float lmax = -1e30f;
            for (int i = t; i < nb; i += nt) {
                float sc;
                if (row_mask && row_mask[(base + i) >> 6] == 0 && mask_block == 64) {
                    sc = -1e30f;                       // deselected block: skip the load
                } else if (row_mask && mask_block != 64
                           && row_mask[(base + i) / mask_block] == 0) {
                    sc = -1e30f;
                } else {
                    const int8_t* kr = kpool + (size_t)rows[i] * hd;
                    float dot = 0.f;
                    for (int d = 0; d < hd; ++d) dot += qs[d] * (float)kr[d];
                    sc = dot * kscale[rows[i]] * scale;
                }
                w[i] = sc;
                lmax = fmaxf(lmax, sc);
            }
            const float bmax = block_reduce<true>(lmax, red);

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

            if (t < hd) {
                float a = 0.f;
                for (int i = 0; i < nb; ++i) {
                    if (w[i] == 0.f) continue;         // masked key: adds exactly 0
                    a += w[i] * vscale[rows[i]] * (float)vpool[(size_t)rows[i] * hd + t];
                }
                acc += a;
            }
            __syncthreads();
        }
    }

    if (t == 0) {
        const size_t ml = ((size_t)z * nh + h) * split + s;
        part_ml[ml * 2 + 0] = m;
        part_ml[ml * 2 + 1] = l;
    }
    if (t < hd) {
        const size_t pa = ((size_t)z * nh + h) * split + s;
        part_acc[pa * hd + t] = acc;
    }
}

// @brief Launch the multi-query verify attention (one history pass for c
//        queries).
//
// @param q,gate,out [B, c, nh, hd] bf16 contiguous; the c decode rows of a
//                   sequence are packed next to each other.
// @param slots      [B, smax] int64 contiguous slot map, shared by the c queries
//                   of a row.
// @param ctx_lens   [B] int32 contiguous history length per row.
// @param max_seq    Unused, same reason as ``attn_paged_batch``: the window is
//                   streamed and the length comes from ``ctx_lens``.
// @param part_acc,part_ml [B*c, nh, split, hd] / [..., 2] fp32 caller-owned, so
//                   the shape is static for graph capture.
// @param block_mask Optional uint8 [B, nblocks] keep-mask; an undefined or empty
//                   tensor means exact attention. Must be contiguous.
// @param mask_block Tokens per mask block, >= 1.
// @return ``out``, written in place for call chaining.
// @throws c10::Error (TORCH_CHECK) when a shape or dtype breaks the indexing
//         contract; fix the caller.
torch::Tensor attn_paged_multi_q(torch::Tensor q, torch::Tensor gate,
                                 torch::Tensor kpool, torch::Tensor vpool,
                                 torch::Tensor kscale, torch::Tensor vscale,
                                 torch::Tensor slots, torch::Tensor ctx_lens,
                                 torch::Tensor out, int64_t max_seq,
                                 int64_t rep, double scale,
                                 torch::Tensor part_acc, torch::Tensor part_ml,
                                 torch::Tensor block_mask, int64_t mask_block) {
    (void)max_seq;      // streamed window: the length is ctx_lens[b] + j + 1
    TORCH_CHECK(q.dim() == 4 && gate.dim() == 4 && out.dim() == 4,
                "q/gate/out must be [B, c, nh, hd]");
    TORCH_CHECK(q.is_contiguous() && gate.is_contiguous() && out.is_contiguous(),
                "q/gate/out must be contiguous");
    TORCH_CHECK(q.stride(3) == 1, "unit inner stride required");
    const int B = q.size(0), c = q.size(1), nh = q.size(2), hd = q.size(3);
    const int Z = B * c;
    const int nkv = kpool.size(1);
    TORCH_CHECK(gate.sizes() == q.sizes() && out.sizes() == q.sizes(),
                "q/gate/out shape mismatch");
    TORCH_CHECK(slots.dim() == 2 && slots.size(0) == B && slots.is_contiguous(),
                "slots must be [B, max_seq] int64 and contiguous");
    TORCH_CHECK(ctx_lens.dim() == 1 && ctx_lens.size(0) == B
                && ctx_lens.scalar_type() == at::kInt && ctx_lens.is_contiguous(),
                "ctx_lens must be [B] int32 contiguous");
    TORCH_CHECK(part_acc.dim() == 4 && part_acc.size(0) == Z && part_acc.size(1) == nh
                && part_acc.size(3) == hd && part_acc.is_contiguous(),
                "part_acc must be [B*c, nh, split, hd] fp32");
    TORCH_CHECK(part_ml.dim() == 4 && part_ml.size(0) == Z && part_ml.size(1) == nh
                && part_ml.size(3) == 2 && part_ml.is_contiguous()
                && part_acc.size(2) == part_ml.size(2),
                "part_ml must be [B*c, nh, split, 2] fp32");
    TORCH_CHECK(mask_block >= 1, "mask_block must be >= 1");
    const int smax = (int)slots.stride(0);
    const unsigned char* mask_ptr = nullptr;
    int nblocks = 0;
    if (block_mask.defined() && block_mask.numel() > 0) {
        TORCH_CHECK(block_mask.dim() == 2 && block_mask.size(0) == B
                    && block_mask.scalar_type() == at::kByte
                    && block_mask.is_contiguous(),
                    "block_mask must be uint8 [B, n_blocks] contiguous");
        nblocks = (int)block_mask.size(1);
        mask_ptr = block_mask.data_ptr<unsigned char>();
    }
    const int split = part_acc.size(2);
    const int threads = 256;
    const int cthreads = hd > threads ? ((hd + 31) / 32) * 32 : threads;
    const size_t smem = (size_t)(2 * ATTN_KEY_BLOCK + hd + threads) * sizeof(float);
    hipStream_t stream = at::cuda::getCurrentCUDAStream();
    attn_paged_multi_q_kernel<<<dim3(c, nh, B * split), threads, smem, stream>>>(
        reinterpret_cast<const unsigned short*>(q.data_ptr<at::BFloat16>()),
        nh * hd, hd,
        kpool.data_ptr<int8_t>(), vpool.data_ptr<int8_t>(),
        kscale.data_ptr<float>(), vscale.data_ptr<float>(),
        slots.data_ptr<int64_t>(), smax,
        ctx_lens.data_ptr<int32_t>(), mask_ptr, nblocks, (int)mask_block,
        part_acc.data_ptr<float>(), part_ml.data_ptr<float>(),
        c, nh, nkv, hd, (int)rep, (float)scale, split);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    attn_paged_batch_combine_kernel<<<dim3(nh, Z), cthreads, 0, stream>>>(
        part_acc.data_ptr<float>(), part_ml.data_ptr<float>(),
        reinterpret_cast<const unsigned short*>(gate.data_ptr<at::BFloat16>()),
        nh * hd, hd,
        reinterpret_cast<unsigned short*>(out.data_ptr<at::BFloat16>()),
        nh * hd, hd, hd, split);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
