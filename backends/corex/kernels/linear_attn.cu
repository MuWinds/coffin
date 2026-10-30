// @file linear_attn.cu
// @brief The gated-delta-rule prologue, the depthwise causal conv1d that feeds it, and
// the fused decode chain that runs both plus the recurrence and the gated norm.
//
// Module: coffin/backends/corex/kernels. Feeds the delta recursion the way
// ``backends/hip/kernels/linear_attn.cu`` does on the DCU: this file produces the fp32
// q/k/v/beta/g that the delta step consumes, so the two must agree on the layout and on
// the 1e-6 L2 epsilon. ``delta_normed`` is the decode path of that pairing and lives here
// rather than in ``delta.cu`` because its reductions have to reproduce this file's
// ``block_sum2`` pairing exactly; ``delta.cu`` keeps the separate ``delta`` /
// ``delta_batch`` entries the batch and verify paths still use.
//
// Why fused: before the delta kernel the engine L2-normalises q and k per head, scales
// q, expands the 16 key heads to the 32 value heads, and builds
// ``beta = sigmoid(b)`` and ``g = -exp(A) * softplus(a + dt_bias)``. As torch ops that
// is ~16 small kernels per layer over 24 linear-attention layers, which at this card's
// launch overhead is a large fraction of the step. One block per value head does all of
// it and writes the fp32 the next kernel reads.
//
// Constraints:
//   * Every reduction is a shared-memory tree, not ``__shfl_down_sync``. This part's
//     warp is 64 lanes wide, so a butterfly written for 32 would normalise q and k by
//     the RMS of half their head -- wrong, and wrong by a plausible-looking amount. The
//     tree is sized from ``blockDim`` at launch.
//   * ``q`` and ``k`` are normalised in one reduction pass. They are always needed
//     together, so two passes would double the barriers for the same result.
//   * The 1e-6 epsilon in the q/k norms is part of the checkpoint's numerics, not a
//     tuning knob: changing it changes the model's output.
//   * Inputs that are column slices of the fused projection output take their row pitch
//     from the tensor (``ldx`` / ``ldb`` / ``lda``). Assuming contiguity there reads the
//     neighbouring third of the fused row, which produces plausible garbage.
//   * ``conv_state`` is caller-owned and updated in place: it is the layer's short-term
//     memory and has to be snapshotted with the recurrent state, so a kernel that
//     returned a replacement buffer would break prefix reuse.
//   * ``g`` must stay ``<= 0``. ``neg_expA`` is negative and softplus is positive, so
//     the product is a decay; a sign slip would make the recurrence grow instead, which
//     diverges rather than merely being wrong.
//   * ``delta_normed`` must be bit-identical to the ``delta_prologue`` -> ``delta`` ->
//     ``rms_norm_gated`` chain, because the decode step's greedy ids are the acceptance
//     bar and a last-ulp shift moves them. That is why it reuses ``block_sum2`` and
//     ``head_threads`` verbatim instead of growing its own reduction, and why it rounds
//     the recurrence's fp32 output to bf16 before normalising -- the chain's
//     ``RMSNormGated`` casts ``delta_out`` first, so the norm's inputs are bf16 there.
//     A block therefore needs ``dv <= head_threads(dk)``; a wider value axis keeps the
//     three-op chain, which the wrapper rejects with a shape error rather than silently
//     changing the arithmetic.

#include "corex_ops.h"
#include "corex_scalar.h"

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

namespace {

//: Epsilon of the checkpoint's q/k L2 normalisation. Part of the model's numerics.
constexpr float L2_EPS = 1e-6f;

//: Above this, ``softplus(x) == x`` to fp32 precision, so the ``exp`` is skipped. The
//: logit ``a + dt_bias`` is unbounded above, and ``exp`` of a large value overflows.
constexpr float SOFTPLUS_LINEAR = 20.f;

//: Conv kernel width. The checkpoint uses a depthwise causal conv with 4 taps, and the
//: state holds the previous 3 tokens.
constexpr int CONV_K = 4;
constexpr int CONV_STATE = CONV_K - 1;

//: Threads for a head-normalisation block: ``dk`` is 128 for this model, and the
//: reduction needs a wavefront multiple.
int head_threads(int dk) {
    int t = dk > 128 ? dk : 128;
    t = ((t + 63) / 64) * 64;
    return t > 512 ? 512 : t;
}

//: Threads ``norm.cu`` uses for a row of width ``N``, reproduced to check the fused kernel
//: can stand in for the gated norm.
//:
//: The rounding is deliberately the same expression, not a call: ``rms_norm_gated_kernel``
//: lives in another translation unit behind an anonymous namespace, so there is nothing to
//: call. The duplication is checked rather than trusted -- any drift makes ``delta_normed``
//: reject a geometry it could have served, which is a documented fallback, whereas the
//: reverse (accepting one whose reduction pairs differ) would silently change the ids.
int norm_threads(int n) {
    int t = n < 64 ? 64 : n;
    t = ((t + 63) / 64) * 64;
    return t > 1024 ? 1024 : t;
}

// @brief Block-wide sum of two values at once, through a shared tree.
//
// q and k are normalised together, so one pass suffices. Two separate reductions would
// double the barrier count for a kernel that is entirely latency-bound: the head is 128
// elements and the block has nothing else to hide the sync behind.
//
// @param a First contribution (q's sum of squares).
// @param b Second contribution (k's sum of squares).
// @param red Shared scratch of ``2 * blockDim.x`` floats; clobbered.
// @param sum_a Out-parameter receiving the block sum of ``a``.
// @param sum_b Out-parameter receiving the block sum of ``b``.
//
// The closing barrier is required. The two sums are read out of ``red`` by every thread,
// and the block then writes other shared regions that may be laid out over the same
// memory. Without the barrier a fast thread can reach that write before a slow thread has
// read the sums -- the bug that produced a 4.5e-02 error in ``delta.cu``'s equivalent
// helper.
__device__ __forceinline__ void block_sum2(float a, float b, float* red, float* sum_a,
                                           float* sum_b) {
    const int t = threadIdx.x;
    red[t] = a;
    red[blockDim.x + t] = b;
    __syncthreads();
    for (int step = blockDim.x >> 1; step > 0; step >>= 1) {
        if (t < step) {
            red[t] += red[t + step];
            red[blockDim.x + t] += red[blockDim.x + t + step];
        }
        __syncthreads();
    }
    *sum_a = red[0];
    *sum_b = red[blockDim.x];
    __syncthreads();
}

// @brief Softplus with a linear tail, so a large logit cannot overflow the exponential.
//
// @param v Logit, i.e. ``a + dt_bias``.
// @return ``log(1 + exp(v))`` for moderate ``v``, ``v`` itself when it is large.
__device__ __forceinline__ float softplus(float v) {
    return v > SOFTPLUS_LINEAR ? v : __logf(1.f + __expf(v));
}

// @brief Block-wide sum of one value, paired exactly as :func:`block_sum2` does it.
//
// The gated norm's reduction is the other half of ``delta_normed``, and it has to combine
// the same partners in the same order as ``rms_norm_gated_kernel``'s tree or the scale
// moves in the last ulp. Sharing the tree shape rather than re-deriving it is what makes
// that structural instead of a thing to keep in step by eye. ``block_sum2`` cannot serve
// here: it sums two values and the epilogue has only one, and a second dummy column
// would double the shared traffic of the kernel's last phase.
//
// @param v This thread's contribution.
// @param red Shared scratch of ``blockDim.x`` floats; clobbered.
// @return The block-wide sum, valid in every thread after the closing sync.
//
// The closing barrier is required for the same reason as in ``block_sum2``: the caller
// reads the sum out of ``red[0]`` and must not have a faster thread start overwriting
// ``red`` first. ``delta_normed_kernel`` does reach that hazard -- it reuses ``red`` for
// this reduction right after ``block_sum2`` returns -- so the barrier is load-bearing
// there and is not omitted.
__device__ __forceinline__ float block_sum_scalar(float v, float* red) {
    const int t = (int)threadIdx.x;
    red[t] = v;
    __syncthreads();
    for (int step = blockDim.x >> 1; step > 0; step >>= 1) {
        if (t < step) red[t] += red[t + step];
        __syncthreads();
    }
    const float total = red[0];
    __syncthreads();
    return total;
}

// @brief Build the fp32 q/k/v and the beta/g gates for one decode step.
//
// One block per value head: the block normalises that head's q and k (shared with the
// other value heads of the same key head, which is why each key head is normalised
// ``rep`` times -- the extra reads are L2 hits and the alternative is another launch),
// scales q, and writes beta/g from the raw logits.
//
// @param q_raw [nkeys, dk] fp32 query, per key head.
// @param k_raw [nkeys, dk] fp32 key, per key head.
// @param v_raw [nv, dv] fp32 value; copied through so the delta step reads one dtype.
// @param b,a [nv] bf16 raw logits feeding beta and g.
// @param neg_expA,dt_bias [nv] fp32 checkpoint constants for the decay gate.
// @param q_out [nv, dk] fp32 normalised and scaled query.
// @param k_out [nv, dk] fp32 normalised key.
// @param v_out [nv, dv] fp32 value.
// @param beta [nv] fp32 write strength.
// @param g [nv] fp32 log decay, ``<= 0``.
// @param qscale ``dk ** -0.5``, folded into the normalised query.
// @param rep Value heads per key head.
__global__ void prologue_kernel(const float* __restrict__ q_raw,
                                const float* __restrict__ k_raw,
                                const float* __restrict__ v_raw,
                                const uint16_t* __restrict__ b,
                                const uint16_t* __restrict__ a,
                                const float* __restrict__ neg_expA,
                                const float* __restrict__ dt_bias,
                                float* __restrict__ q_out, float* __restrict__ k_out,
                                float* __restrict__ v_out, float* __restrict__ beta,
                                float* __restrict__ g, float qscale, int dk, int dv,
                                int rep) {
    const int vh = blockIdx.x;
    const int t = threadIdx.x;
    const int kh = vh / rep;
    const float* qr = q_raw + (size_t)kh * dk;
    const float* kr = k_raw + (size_t)kh * dk;

    const float qv = t < dk ? qr[t] : 0.f;
    const float kv = t < dk ? kr[t] : 0.f;
    extern __shared__ float sh[];
    float* red = sh;                       // [2 * blockDim.x]
    float ssq = 0.f;
    float ssk = 0.f;
    block_sum2(qv * qv, kv * kv, red, &ssq, &ssk);

    if (t < dk) {
        q_out[(size_t)vh * dk + t] = qv * rsqrtf(ssq + L2_EPS) * qscale;
        k_out[(size_t)vh * dk + t] = kv * rsqrtf(ssk + L2_EPS);
    }
    if (t < dv) v_out[(size_t)vh * dv + t] = v_raw[(size_t)vh * dv + t];
    if (t == 0) {
        beta[vh] = 1.f / (1.f + __expf(-CxScalar<uint16_t>::to_float(b[vh])));
        const float av = CxScalar<uint16_t>::to_float(a[vh]) + dt_bias[vh];
        g[vh] = neg_expA[vh] * softplus(av);
    }
}

// @brief Depthwise causal conv (4 taps) followed by SiLU, for one token.
//
// Thread ``c`` owns channel ``c`` and its three-tap window, so the read-then-write of the
// state needs no synchronisation: no other thread touches that window.
//
// Grid ``y`` is the batch row. A row's inputs and window are separate slabs, so the row
// axis costs one block index and no traffic between rows; the pitches are passed in rather
// than assumed, because the batched caller reads a column slice of the fused projection
// whose row pitch is larger than ``C``.
//
// @param w [C, 4] fp32 taps, oldest first; the last tap multiplies the current input.
// @param x [B, C] bf16 projection output with row pitch ``ldx``.
// @param state [B, C, 3] fp32 previous inputs, shifted in place with row pitch ``lds``.
// @param out [B, C] fp32 conv + SiLU result with row pitch ``ldo``.
// @param C Channels, i.e. the width the ``x`` grid extent covers.
// @param ldx Row pitch of ``x``, in elements. The verify body hands in ``qkv[:, j]``, a
//        column slice of a ``[rows, c, conv_dim]`` view, so this is ``c * C`` there and a
//        kernel that assumed ``C`` would silently read the neighbouring token's channels.
// @param lds Row pitch of ``state``, in floats.
// @param ldo Row pitch of ``out``, in floats.
__global__ void conv1d_silu_kernel(const float* __restrict__ w,
                                   const uint16_t* __restrict__ x,
                                   float* __restrict__ state, float* __restrict__ out,
                                   int C, int ldx, int lds, int ldo) {
    const int row = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float* s = state + (size_t)row * lds + (size_t)c * CONV_STATE;
    const float* wc = w + (size_t)c * CONV_K;
    const float x3 = CxScalar<uint16_t>::to_float(x[(size_t)row * ldx + c]);
    const float acc = fmaf(s[0], wc[0],
                           fmaf(s[1], wc[1], fmaf(s[2], wc[2], x3 * wc[3])));
    const float y = acc / (1.f + __expf(-acc));     // SiLU
    out[(size_t)row * ldo + c] = y;
    s[0] = s[1];
    s[1] = s[2];
    s[2] = x3;
}

// @brief The same conv over a prefill chunk, carrying the window in registers.
//
// Thread ``c`` walks the sequence and keeps the three-tap ring in registers, so the pass
// reads the bf16 activation once and writes fp32 once. The eager chain it replaces
// (cat + fp32 promote + unfold + multiply + sum + SiLU) touches about six times the
// bytes.
//
// At a fixed ``t`` neighbouring threads read neighbouring channels, so a wavefront still
// covers a full cache line per step even though one thread strides by ``ldx``.
//
// @param w [C, 4] fp32 taps.
// @param x [T, C] bf16 activations; the row pitch ``ldx`` may exceed ``C`` because the
//        projections are read straight out of the fused GEMV output.
// @param state [C, 3] fp32 window, updated in place to the last three inputs.
// @param out [T, C] fp32, densely packed.
__global__ void conv1d_silu_seq_kernel(const float* __restrict__ w,
                                       const uint16_t* __restrict__ x,
                                       float* __restrict__ state, float* __restrict__ out,
                                       int C, int T, int ldx) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const float* wc = w + (size_t)c * CONV_K;
    const float w0 = wc[0];
    const float w1 = wc[1];
    const float w2 = wc[2];
    const float w3 = wc[3];
    float* s = state + (size_t)c * CONV_STATE;
    float s0 = s[0];
    float s1 = s[1];
    float s2 = s[2];
    const uint16_t* xp = x + c;
    float* o = out + c;
    for (int t = 0; t < T; ++t) {
        const float x3 = CxScalar<uint16_t>::to_float(xp[(size_t)t * ldx]);
        const float acc = fmaf(w0, s0, fmaf(w1, s1, fmaf(w2, s2, w3 * x3)));
        o[(size_t)t * C] = acc / (1.f + __expf(-acc));     // SiLU
        s0 = s1;
        s1 = s2;
        s2 = x3;
    }
    s[0] = s0;
    s[1] = s1;
    s[2] = s2;
}

// @brief Prefill head prep: the decode prologue over a whole chunk, in head-major layout.
//
// One block per (value head, token). The transposes matter more than they look: handing
// the chunked delta scan a transposed ``[H, C, d]`` view forces three strided copies
// inside it, and a blocked strided copy on this part runs far below the achievable
// bandwidth. Writing ``[H, T, d]`` here removes all three.
//
// @param qkv [T, conv_dim] fp32 conv output laid out q | k | v; the heads are column
//        slices of that row.
// @param b,a [T, nv] bf16 with row pitches ``ldb`` / ``lda``.
// @param q_out,k_out [nv, T, dk] fp32; ``v_out`` [nv, T, dv] fp32.
// @param beta,g [nv, T] fp32.
// @param kdim ``num_key_heads * dk``, the offset of the k third inside a row.
__global__ void delta_prep_seq_kernel(const float* __restrict__ qkv,
                                      const uint16_t* __restrict__ b,
                                      const uint16_t* __restrict__ a,
                                      const float* __restrict__ neg_expA,
                                      const float* __restrict__ dt_bias,
                                      float* __restrict__ q_out, float* __restrict__ k_out,
                                      float* __restrict__ v_out, float* __restrict__ beta,
                                      float* __restrict__ g, float qscale, int T, int dk,
                                      int dv, int rep, int kdim, int conv_dim, int ldb,
                                      int lda) {
    const int vh = blockIdx.x;
    const int t = blockIdx.y;
    const int i = threadIdx.x;
    const int kh = vh / rep;

    const float* row = qkv + (size_t)t * conv_dim;
    const float qv = i < dk ? row[(size_t)kh * dk + i] : 0.f;
    const float kv = i < dk ? row[kdim + (size_t)kh * dk + i] : 0.f;

    extern __shared__ float sh[];
    float* red = sh;                       // [2 * blockDim.x]
    float ssq = 0.f;
    float ssk = 0.f;
    block_sum2(qv * qv, kv * kv, red, &ssq, &ssk);

    if (i < dk) {
        q_out[((size_t)vh * T + t) * dk + i] = qv * rsqrtf(ssq + L2_EPS) * qscale;
        k_out[((size_t)vh * T + t) * dk + i] = kv * rsqrtf(ssk + L2_EPS);
    }
    if (i < dv) v_out[((size_t)vh * T + t) * dv + i] = row[2 * kdim + (size_t)vh * dv + i];
    if (i == 0) {
        beta[(size_t)vh * T + t] =
            1.f / (1.f + __expf(-CxScalar<uint16_t>::to_float(b[(size_t)t * ldb + vh])));
        const float av = CxScalar<uint16_t>::to_float(a[(size_t)t * lda + vh]) + dt_bias[vh];
        g[(size_t)vh * T + t] = neg_expA[vh] * softplus(av);
    }
}

// @brief Batched prologue: one block per (row, value head).
//
// @param q,k [B, nkeys, dk] fp32; ``v`` [B, nv, dv] fp32.
// @param b,a [B, nv] bf16.
// @param q_out,k_out [B, nv, dk] fp32; ``v_out`` [B, nv, dv] fp32.
// @param beta,g [B, nv] fp32.
__global__ void prologue_batch_kernel(const float* __restrict__ q,
                                      const float* __restrict__ k,
                                      const float* __restrict__ v,
                                      const uint16_t* __restrict__ b,
                                      const uint16_t* __restrict__ a,
                                      const float* __restrict__ neg_expA,
                                      const float* __restrict__ dt_bias,
                                      float* __restrict__ q_out, float* __restrict__ k_out,
                                      float* __restrict__ v_out, float* __restrict__ beta,
                                      float* __restrict__ g, float qscale, int nv, int dk,
                                      int dv, int rep) {
    const int row = blockIdx.y;
    const int vh = blockIdx.x;
    const int t = threadIdx.x;
    const int kh = vh / rep;
    const size_t head = ((size_t)row * (nv / rep) + kh) * dk;
    const float qv = t < dk ? q[head + t] : 0.f;
    const float kv = t < dk ? k[head + t] : 0.f;

    extern __shared__ float sh[];
    float* red = sh;
    float ssq = 0.f;
    float ssk = 0.f;
    block_sum2(qv * qv, kv * kv, red, &ssq, &ssk);

    const size_t out_head = ((size_t)row * nv + vh);
    if (t < dk) {
        q_out[out_head * dk + t] = qv * rsqrtf(ssq + L2_EPS) * qscale;
        k_out[out_head * dk + t] = kv * rsqrtf(ssk + L2_EPS);
    }
    if (t < dv) v_out[out_head * dv + t] = v[out_head * dv + t];
    if (t == 0) {
        const size_t bi = (size_t)row * nv + vh;
        beta[bi] = 1.f / (1.f + __expf(-CxScalar<uint16_t>::to_float(b[bi])));
        const float av = CxScalar<uint16_t>::to_float(a[bi]) + dt_bias[vh];
        g[bi] = neg_expA[vh] * softplus(av);
    }
}

// @brief Prologue, recurrence and gated norm for one decode token, one block per head.
//
// Why this is one kernel: the three it replaces are one memory problem, not three. The
// prologue writes ``q_pro``/``k_pro``/``v_pro`` (3 x [nv, dk] fp32) plus ``beta``/``g``,
// the recurrence reads them straight back and writes ``delta_out`` ([nv, dv] fp32), and
// the gated norm reads *that* back: about 48 KB round tripped per layer per token, 24
// times a step, purely to hand values between launches on a card whose launch floor is
// 5.75 us. Everything the prologue produces for one value head fits in that head's block
// -- ``q`` and ``k`` in ``2 * dk`` floats of shared memory (1 KB at dk = 128), ``v`` and
// the two gate scalars in registers -- and the norm's reduction is over ``dv``, which is
// exactly one block's span, so it folds into the epilogue with no extra reduction.
//
// The block shape is not a compromise between the three stages, it is the one shape all
// three already use: ``prologue_kernel`` runs ``head_threads(dk)`` threads per value
// head, ``delta_kernel`` runs ``min(dv, THREADS)``, and ``rms_norm_gated_kernel`` runs
// ``rms_threads(dv)`` -- 128, 128 and 128 for this model, where dk = dv = 128. Sharing
// that shape is what lets every reduction keep the chain's pairing, which is the
// difference between bit-identical and merely close: the decode step's greedy ids are the
// acceptance bar, and a last-ulp shift in any of these three reductions moves them.
//
// ``conv1d_silu`` deliberately stays its own launch. It is over ``conv_dim`` = 8192
// channels rather than over a head, so folding it in would put a slice of a differently
// shaped vector inside a block whose thread mapping exists to coalesce the state access.
//
// @param q,k [nkeys, dk] fp32 raw projection output; normalised here, and ``q`` also
//        scaled, so the caller applies no per-element fix-up.
// @param v [nv, dv] fp32 value.
// @param b,a [nv] bf16 raw beta/decay logits.
// @param neg_expA,dt_bias [nv] fp32 checkpoint constants for the decay gate.
// @param z [nv, dv] bf16 gate for the SiLU of the output norm.
// @param nw [dv] bf16 gated-norm gain, per value head rather than the flattened width.
// @param st [nv, dk, dv] fp32 state, updated in place.
// @param out [nv, dv] written with the normed result; the element type only selects the
//        container, because the value is the chain's bf16 result either way.
// @param qscale ``dk ** -0.5``.
// @param rep Value heads per key head.
//
// ``ZT`` is the gate's storage type, not a precision choice. The op accepts bf16, which is
// what the model's gate projection produces, and fp32, which the contract also allows; the
// instantiation is selected on the host from ``z``'s dtype. Reading an fp32 ``z`` as bf16
// would reinterpret every second half-word and gate the output with noise, so the two cases
// cannot share an instantiation.
template <typename OT, typename ZT>
__global__ void delta_normed_kernel(const float* __restrict__ q,
                                    const float* __restrict__ k,
                                    const float* __restrict__ v,
                                    const uint16_t* __restrict__ b,
                                    const uint16_t* __restrict__ a,
                                    const float* __restrict__ neg_expA,
                                    const float* __restrict__ dt_bias,
                                    const ZT* __restrict__ z,
                                    const uint16_t* __restrict__ nw,
                                    float* __restrict__ st, OT* __restrict__ out,
                                    float qscale, int dk, int dv, int rep) {
    const int vh = blockIdx.x;
    const int t = (int)threadIdx.x;
    const int kh = vh / rep;

    extern __shared__ float sh[];
    float* red = sh;                          // [2 * blockDim.x] the reduction tree
    float* qs = sh + 2 * (int)blockDim.x;     // [dk] this head's normalised query
    float* ks = qs + dk;                      // [dk] this head's normalised key

    // Head preparation: the same arithmetic, the same one-pass reduction and the same
    // tree pairing as ``prologue_kernel``.
    const float qv = t < dk ? q[(size_t)kh * dk + t] : 0.f;
    const float kv = t < dk ? k[(size_t)kh * dk + t] : 0.f;
    float ssq = 0.f;
    float ssk = 0.f;
    block_sum2(qv * qv, kv * kv, red, &ssq, &ssk);

    // Kept in shared memory rather than written out and read back: these two buffers are
    // exactly the traffic the fusion exists to remove, and ``2 * dk`` floats covers one
    // head's worth of it.
    if (t < dk) {
        qs[t] = qv * rsqrtf(ssq + L2_EPS) * qscale;
        ks[t] = kv * rsqrtf(ssk + L2_EPS);
    }
    __shared__ float rate_sh;
    __shared__ float decay_sh;
    if (t == 0) {
        rate_sh = 1.f / (1.f + __expf(-CxScalar<uint16_t>::to_float(b[vh])));
        const float av = CxScalar<uint16_t>::to_float(a[vh]) + dt_bias[vh];
        // The chain stores ``g`` in the prologue and exponentiates it in the recurrence;
        // composing the two here rounds the product once, in fp32, either way. Doing it
        // per thread instead would repeat one transcendental 128 times for no gain.
        decay_sh = __expf(neg_expA[vh] * softplus(av));
    }
    __syncthreads();

    // A thread past ``dv`` must reach every ``__syncthreads()`` in this kernel and return
    // only after the last one. Returning here -- before the norm's tree -- leaves the rest
    // of the block waiting on a thread that is gone: a device-side deadlock with no
    // traceback rather than a wrong number. That hazard is written up in ``delta.cu``,
    // whose first version had it.
    const bool active = t < dv;
    const float rate = rate_sh;
    const float decay = decay_sh;

    // The recurrence, unchanged from ``delta_kernel``: thread ``j`` owns value column
    // ``j`` outright and walks ``i`` over ``dk`` in registers, so both ``dk`` reductions
    // stay thread-local and the state read at a fixed ``i`` is contiguous in ``j`` -- the
    // only assignment that coalesces for a ``[nv, dk, dv]`` layout.
    float result = 0.f;
    if (active) {
        const float value = v[(size_t)vh * dv + t];
        float* column = st + (size_t)vh * dk * dv + t;
        float kv_mem = 0.f;
        for (int i = 0; i < dk; ++i) {
            kv_mem = fmaf(column[(size_t)i * dv] * decay, ks[i], kv_mem);
        }
        const float delta = (value - kv_mem) * rate;
        for (int i = 0; i < dk; ++i) {
            const float s = fmaf(ks[i], delta, column[(size_t)i * dv] * decay);
            column[(size_t)i * dv] = s;
            result = fmaf(s, qs[i], result);
        }
    }

    // The chain hands the gated norm a bf16 copy of ``delta_out``: ``RMSNormGated`` casts
    // its fp32 input to the activation dtype before the kernel sees it, so the norm's
    // input is already on the bf16 grid. Rounding here reproduces that. Skipping it would
    // normalise a value up to half a bf16 ulp away from what the chain normalises, which
    // is enough to move the ids -- this rounding is part of the chain's arithmetic, not an
    // artefact of its storage.
    const float xb = active
                         ? CxScalar<uint16_t>::to_float(CxScalar<uint16_t>::from_float(result))
                         : 0.f;
    // ``block_sum_scalar`` closes on a barrier, so ``red`` can be reused here by every
    // thread -- including the ones that skipped the recurrence, which is why that barrier
    // is not an optimisation to drop.
    const float inv = rsqrtf(block_sum_scalar(xb * xb, red) / (float)dv + L2_EPS);
    if (!active) return;

    // ``y = rms(x) * w * silu(gate)`` in the chain's own operand order. The products
    // associate left to right and reassociating them changes the last ulp, so the order is
    // part of the specification rather than a style choice.
    //
    // The gate is rounded to bf16 whatever its storage. ``RMSNormGated.__call__`` casts
    // both operands to the activation dtype before the kernel sees them, so an fp32 gate
    // reaching the chain would be narrowed there -- rounding here reproduces that, and for
    // a gate already stored as bf16 the round is the identity.
    const float gv = CxScalar<uint16_t>::to_float(CxScalar<uint16_t>::from_float(
        CxScalar<ZT>::to_float(z[(size_t)vh * dv + t])));
    const float silu = gv / (1.f + __expf(-gv));     // SiLU
    const float y = xb * inv * CxScalar<uint16_t>::to_float(nw[t]) * silu;
    // The chain's norm emits bf16 and ``out_proj`` consumes that, so the destination's
    // dtype is only a container: an fp32 one carries the same bf16 value rather than a
    // more precise one, which is what keeps both call sites agreeing bit for bit.
    out[(size_t)vh * dv + t] = CxScalar<OT>::from_float(
        CxScalar<uint16_t>::to_float(CxScalar<uint16_t>::from_float(y)));
}

// @brief Launch the fused decode chain: one block per value head.
//
// @param out_bf16 Whether ``out`` is bf16 (as ``uint16_t``) or fp32.
// @param z_bf16 Whether ``z`` is bf16 (as ``uint16_t``) or fp32.
// @param threads Block size. The wrapper checks it against both ``head_threads(dk)`` and
//        the block size the chain's other two reductions use, because a different size
//        would sum the same values in a different order.
void launch_normed(const float* q, const float* k, const float* v, const uint16_t* b,
                   const uint16_t* a, const float* neg_expA, const float* dt_bias,
                   const void* z, bool z_bf16, const uint16_t* nw, float* st, void* out,
                   bool out_bf16, float qscale, int nv, int dk, int dv, int rep,
                   int threads, cudaStream_t stream) {
    const size_t smem = (size_t)(2 * threads + 2 * dk) * sizeof(float);
    // Four instantiations, because the destination dtype and the gate's storage dtype are
    // independent choices. All four compute the same arithmetic -- the gate is rounded to
    // bf16 inside the kernel either way -- so this is dispatch, not four behaviours.
    if (out_bf16) {
        if (z_bf16) {
            delta_normed_kernel<uint16_t, uint16_t><<<nv, threads, smem, stream>>>(
                q, k, v, b, a, neg_expA, dt_bias, (const uint16_t*)z, nw, st,
                reinterpret_cast<uint16_t*>(out), qscale, dk, dv, rep);
        } else {
            delta_normed_kernel<uint16_t, float><<<nv, threads, smem, stream>>>(
                q, k, v, b, a, neg_expA, dt_bias, (const float*)z, nw, st,
                reinterpret_cast<uint16_t*>(out), qscale, dk, dv, rep);
        }
    } else {
        if (z_bf16) {
            delta_normed_kernel<float, uint16_t><<<nv, threads, smem, stream>>>(
                q, k, v, b, a, neg_expA, dt_bias, (const uint16_t*)z, nw, st,
                reinterpret_cast<float*>(out), qscale, dk, dv, rep);
        } else {
            delta_normed_kernel<float, float><<<nv, threads, smem, stream>>>(
                q, k, v, b, a, neg_expA, dt_bias, (const float*)z, nw, st,
                reinterpret_cast<float*>(out), qscale, dk, dv, rep);
        }
    }
}

}  // namespace

// @brief Launch the single-token depthwise causal conv1d + SiLU.
//
// One row, so the grid's ``y`` extent is 1 and every pitch is the row's own width. This is
// the same kernel the batched entry launches, with the row axis collapsed, so the two paths
// cannot drift apart in their arithmetic.
//
// @param conv_w [C, 4] fp32 taps.
// @param x [C] bf16 projection output.
// @param conv_state [C, 3] fp32 history, shifted in place.
// @param out [C] fp32, written in place.
// @throws c10::Error on a dtype or shape the kernel cannot index.
void conv1d_silu(torch::Tensor conv_w, torch::Tensor x, torch::Tensor conv_state,
                 torch::Tensor out) {
    const int C = (int)x.size(0);
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "activation must be bf16");
    TORCH_CHECK(conv_state.numel() == (int64_t)C * CONV_STATE, "conv state must be [C, 3]");
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    conv1d_silu_kernel<<<dim3((C + 255) / 256, 1), 256, 0, stream>>>(
        conv_w.data_ptr<float>(), reinterpret_cast<const uint16_t*>(x.data_ptr()),
        conv_state.data_ptr<float>(), out.data_ptr<float>(), C, C, CONV_STATE, C);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// @brief Launch the chunked prefill conv1d + SiLU.
//
// @param conv_w [C, 4] fp32 taps.
// @param x [T, C] bf16 with unit inner stride; the row pitch is honoured so a column
//        slice of the fused GEMV output can be read in place.
// @param conv_state [C, 3] fp32, updated in place so the next decode step continues the
//        same recurrence -- which is what makes a chunked prefill agree with a one-shot
//        prefill.
// @return [T, C] fp32 conv + SiLU.
// @throws c10::Error when ``x`` is not 2-D with unit inner stride; transpose the
//         caller's slice rather than relaxing this.
torch::Tensor conv1d_silu_seq(torch::Tensor conv_w, torch::Tensor x,
                              torch::Tensor conv_state) {
    TORCH_CHECK(x.dim() == 2 && x.stride(1) == 1,
                "activation must be [T, C] with unit inner stride");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "activation must be bf16");
    const int T = (int)x.size(0);
    const int C = (int)x.size(1);
    const int ldx = (int)x.stride(0);
    auto out = torch::empty({T, C}, x.options().dtype(torch::kFloat));
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    conv1d_silu_seq_kernel<<<(C + 255) / 256, 256, 0, stream>>>(
        conv_w.data_ptr<float>(), reinterpret_cast<const uint16_t*>(x.data_ptr()),
        conv_state.data_ptr<float>(), out.data_ptr<float>(), C, T, ldx);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// @brief Launch the batched single-token conv1d + SiLU: one launch for every row.
//
// The grid's ``y`` axis is the row, so the whole bucket costs one launch and keeps
// ``rows * ceil(C/256)`` blocks in flight instead of the single row's 32. Measured, the old
// per-row loop cost a flat 9.57 us per row for 368 KB of traffic per row -- 39 GB/s, which is
// grid width and not bytes -- and a 4-row verify window paid it 96 times over its 24 linear
// layers.
//
// The pitches come from the tensors rather than from ``C``, and that is load-bearing: the
// verify body hands in ``qkv[:, j]``, a column slice of a ``[rows, c, conv_dim]`` view whose
// row pitch is ``c * C``. Reading with a pitch of ``C`` made rows >= 1 take row 0's channel
// block, which is invisible at ``rows == 1`` and would have been a silent wrong answer at
// every width above it (measured: 3 of 4 rows wrong before this argument existed).
//
// @param conv_w [C, 4] fp32 taps, shared by every row.
// @param x [B, C] bf16 with row pitch honoured; a strided column slice is legal.
// @param conv_state [B, C, 3] fp32, shifted in place.
// @param out [B, C] fp32, written in place.
// @throws c10::Error on a dtype, rank or inner stride the kernel cannot index.
void conv1d_silu_batch(torch::Tensor conv_w, torch::Tensor x, torch::Tensor conv_state,
                       torch::Tensor out) {
    TORCH_CHECK(x.dim() == 2, "activation must be [B, C]");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "activation must be bf16");
    TORCH_CHECK(x.stride(1) == 1, "activation must have unit inner stride");
    const int rows = (int)x.size(0);
    const int C = (int)x.size(1);
    TORCH_CHECK(out.dim() == 2 && out.size(0) == rows && out.size(1) == C
                && out.stride(1) == 1, "output must be [B, C] with unit inner stride");
    TORCH_CHECK(conv_state.numel() == (int64_t)rows * C * CONV_STATE,
                "conv state must be [B, C, 3]");
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    conv1d_silu_kernel<<<dim3((C + 255) / 256, rows), 256, 0, stream>>>(
        conv_w.data_ptr<float>(), reinterpret_cast<const uint16_t*>(x.data_ptr()),
        conv_state.data_ptr<float>(), out.data_ptr<float>(), C,
        (int)x.stride(0), (int)conv_state.stride(0), (int)out.stride(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
// @brief Launch the single-token gated-delta prologue.
//
// @param q,k [nkeys, dk] fp32, per key head.
// @param v [nv, dv] fp32.
// @param b,a [nv] bf16 raw logits.
// @param neg_expA,dt_bias [nv] fp32 checkpoint constants.
// @param q_out,k_out [nv, dk] fp32 and ``v_out`` [nv, dv] fp32, written in place.
// @param beta,g [nv] fp32, written in place.
// @param qscale ``dk ** -0.5``.
// @param rep Value heads per key head.
// @throws c10::Error when a dtype or shape contradicts the layout.
void delta_prologue(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor b,
                    torch::Tensor a, torch::Tensor neg_expA, torch::Tensor dt_bias,
                    torch::Tensor q_out, torch::Tensor k_out, torch::Tensor v_out,
                    torch::Tensor beta, torch::Tensor g, double qscale, int64_t rep) {
    const int nv = (int)v.size(0);
    const int dk = (int)q.size(1);
    const int dv = (int)v.size(1);
    TORCH_CHECK(b.scalar_type() == torch::kBFloat16, "beta logits must be bf16");
    TORCH_CHECK(q.scalar_type() == torch::kFloat, "prologue inputs are fp32");
    const int threads = head_threads(dk);
    const size_t smem = (size_t)2 * threads * sizeof(float);
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    prologue_kernel<<<nv, threads, smem, stream>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
        reinterpret_cast<const uint16_t*>(b.data_ptr()),
        reinterpret_cast<const uint16_t*>(a.data_ptr()),
        neg_expA.data_ptr<float>(), dt_bias.data_ptr<float>(),
        q_out.data_ptr<float>(), k_out.data_ptr<float>(), v_out.data_ptr<float>(),
        beta.data_ptr<float>(), g.data_ptr<float>(), (float)qscale, dk, dv, (int)rep);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// @brief Batched prologue: one row per batch entry, each identical to the decode form
//        on that row alone.
//
// @param q,k [B, nkeys, dk] fp32.
// @param v [B, nv, dv] fp32.
// @param b,a [B, nv] bf16.
// @param neg_expA,dt_bias [nv] fp32, shared by every row.
// @param q_out,k_out [B, nv, dk] fp32 and ``v_out`` [B, nv, dv] fp32, written in place.
// @param beta,g [B, nv] fp32, written in place.
// @param qscale ``dk ** -0.5``.
// @param rep Value heads per key head.
// @throws c10::Error when a dtype or shape contradicts the layout.
void delta_prologue_batch(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                          torch::Tensor b, torch::Tensor a, torch::Tensor neg_expA,
                          torch::Tensor dt_bias, torch::Tensor q_out,
                          torch::Tensor k_out, torch::Tensor v_out, torch::Tensor beta,
                          torch::Tensor g, double qscale, int64_t rep) {
    const int rows = (int)v.size(0);
    const int nv = (int)v.size(1);
    const int dk = (int)q.size(2);
    const int dv = (int)v.size(2);
    TORCH_CHECK(b.scalar_type() == torch::kBFloat16, "beta logits must be bf16");
    const int threads = head_threads(dk);
    const size_t smem = (size_t)2 * threads * sizeof(float);
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    prologue_batch_kernel<<<dim3(nv, rows), threads, smem, stream>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
        reinterpret_cast<const uint16_t*>(b.data_ptr()),
        reinterpret_cast<const uint16_t*>(a.data_ptr()),
        neg_expA.data_ptr<float>(), dt_bias.data_ptr<float>(),
        q_out.data_ptr<float>(), k_out.data_ptr<float>(), v_out.data_ptr<float>(),
        beta.data_ptr<float>(), g.data_ptr<float>(), (float)qscale, nv, dk, dv, (int)rep);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// @brief Launch the chunked prefill gated-delta prologue into the head-major layout.
//
// @param qkv [T, conv_dim] fp32 conv output laid out q | k | v.
// @param b,a [T, nv] bf16.
// @param neg_expA,dt_bias [nv] fp32 constants.
// @param q_out,k_out [nv, T, dk] fp32 and ``v_out`` [nv, T, dv] fp32, written in place.
// @param beta,g [nv, T] fp32, written in place.
// @param qscale ``dk ** -0.5``.
// @param rep Value heads per key head.
// @throws c10::Error when ``qkv``/``b``/``a`` do not have unit inner stride; pass a
//         slice, not a transpose.
void delta_prep_seq(torch::Tensor qkv, torch::Tensor b, torch::Tensor a,
                    torch::Tensor neg_expA, torch::Tensor dt_bias, torch::Tensor q_out,
                    torch::Tensor k_out, torch::Tensor v_out, torch::Tensor beta,
                    torch::Tensor g, double qscale, int64_t rep) {
    TORCH_CHECK(qkv.stride(1) == 1 && b.stride(1) == 1 && a.stride(1) == 1,
                "unit inner stride required");
    const int T = (int)qkv.size(0);
    const int nv = (int)q_out.size(0);
    const int dk = (int)q_out.size(2);
    const int dv = (int)v_out.size(2);
    const int kdim = dk * (int)(q_out.size(0) / rep);      // num_key_heads * dk
    const int threads = head_threads(dk);
    const size_t smem = (size_t)2 * threads * sizeof(float);
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    delta_prep_seq_kernel<<<dim3(nv, T), threads, smem, stream>>>(
        qkv.data_ptr<float>(), reinterpret_cast<const uint16_t*>(b.data_ptr()),
        reinterpret_cast<const uint16_t*>(a.data_ptr()), neg_expA.data_ptr<float>(),
        dt_bias.data_ptr<float>(), q_out.data_ptr<float>(), k_out.data_ptr<float>(),
        v_out.data_ptr<float>(), beta.data_ptr<float>(), g.data_ptr<float>(), (float)qscale,
        T, dk, dv, (int)rep, kdim, (int)qkv.size(1), (int)b.stride(0), (int)a.stride(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// @brief Prologue, recurrence and gated norm for one decode token, in one launch.
//
// The result is the three-op chain's, bit for bit, so the greedy ids cannot move: the
// reductions reuse ``block_sum2`` and the block size both other stages use, the recurrence
// runs the same ``fmaf`` chain against the same fp32 values, and the norm's input is
// rounded to bf16 first because ``RMSNormGated`` casts it before the kernel sees it.
// That equivalence is why the block size is a checked invariant rather than a tuned
// parameter -- see the ``head_threads`` guard below.
//
// @param q [nkeys, dk] fp32 raw query, normalised and scaled here.
// @param k [nkeys, dk] fp32 raw key, normalised here.
// @param v [nv, dv] fp32 value.
// @param b [nv] bf16 beta logits; ``a`` [nv] bf16 decay logits.
// @param neg_expA,dt_bias [nv] fp32 checkpoint constants for the decay gate.
// @param z [nv, dv] bf16 gate for the SiLU of the output norm.
// @param nw [dv] bf16 gated-norm gain, per value head.
// @param st [nv, dk, dv] fp32 state, updated in place.
// @param out [nv, dv] bf16 or fp32. bf16 is what the chain produces and what ``out_proj``
//        reads, so pass that unless the caller needs the wider container; an fp32
//        destination holds the same bf16 value rather than a more precise one.
// @param qscale ``dk ** -0.5``.
// @param rep Value heads per key head.
// @return ``out``, for call chaining.
// @throws c10::Error when a dtype, a shape or the head geometry disagrees with the
//         kernel's indexing. The geometry case is the one worth reading: it means
//         ``dk`` or ``dv`` is wide enough that the chain would reduce with a different
//         block size, so this kernel could not reproduce its rounding and the caller
//         must keep the three-op path.
torch::Tensor delta_normed(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                           torch::Tensor b, torch::Tensor a,
                           torch::Tensor neg_expA, torch::Tensor dt_bias,
                           torch::Tensor z, torch::Tensor nw, torch::Tensor st,
                           torch::Tensor out, double qscale, int64_t rep) {
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "q/k/v must be contiguous");
    TORCH_CHECK(q.scalar_type() == torch::kFloat && k.scalar_type() == torch::kFloat
                    && v.scalar_type() == torch::kFloat
                    && neg_expA.scalar_type() == torch::kFloat
                    && dt_bias.scalar_type() == torch::kFloat
                    && st.scalar_type() == torch::kFloat,
                "q/k/v, neg_expA, dt_bias and the state are fp32");
    // The two logit tensors are read as bf16 bit patterns, so a wider dtype would be
    // reinterpreted rather than converted -- a silent wrong number, not a slow path.
    TORCH_CHECK(b.scalar_type() == torch::kBFloat16 && a.scalar_type() == torch::kBFloat16,
                "beta and decay logits must be bf16");
    // ``z`` is templated on its dtype instead, because the contract allows fp32 and the
    // kernel rounds either form to bf16 before use, matching what the chain's cast does.
    TORCH_CHECK(z.scalar_type() == torch::kBFloat16 || z.scalar_type() == torch::kFloat,
                "the norm gate must be bf16 or fp32");
    TORCH_CHECK(nw.scalar_type() == torch::kBFloat16, "the norm gain must be bf16");
    TORCH_CHECK(out.scalar_type() == torch::kBFloat16 || out.scalar_type() == torch::kFloat,
                "output must be bf16 or fp32");
    TORCH_CHECK(out.is_contiguous() && z.is_contiguous(), "out and z must be contiguous");

    const int nv = (int)st.size(0);
    const int dk = (int)st.size(1);
    const int dv = (int)st.size(2);
    TORCH_CHECK(rep > 0 && nv % (int)rep == 0, "head count does not divide by rep");
    TORCH_CHECK(q.size(0) == nv / (int)rep && q.size(1) == dk,
                "q must be [nv / rep, dk]");
    TORCH_CHECK(k.sizes() == q.sizes(), "k and q must have the same shape");
    TORCH_CHECK(v.size(0) == nv && v.size(1) == dv, "v must be [nv, dv]");
    TORCH_CHECK(z.size(0) == nv && z.size(1) == dv, "norm gate must be [nv, dv]");
    TORCH_CHECK(nw.numel() == (int64_t)dv, "norm gain must have dv elements");
    TORCH_CHECK(out.size(0) == nv && out.size(1) == dv, "out view must be [nv, dv]");

    // One block per value head, and its size is fixed by the chain it replaces rather than
    // chosen here. A block wider or narrower than ``head_threads(dk)`` would sum the q/k
    // norms with different partners than ``prologue_kernel`` does, and one that did not also
    // match the gated norm's block size would do the same to the second reduction -- either
    // way the result is close but not equal, which is what moves the greedy ids.
    const int threads = head_threads(dk);
    TORCH_CHECK(dk <= threads && dv <= threads,
                "delta_normed needs dk and dv within one block (dk=", dk, " dv=", dv,
                " threads=", threads,
                "); keep the delta_prologue/delta/rms_norm_gated chain for this model");
    TORCH_CHECK(norm_threads(dv) == threads,
                "delta_normed needs the prologue's and the gated norm's block sizes to "
                "agree (dk=", dk, " dv=", dv, " prologue=", threads,
                " norm=", norm_threads(dv), "); keep the three-op chain for this model");

    launch_normed(q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
                  reinterpret_cast<const uint16_t*>(b.data_ptr()),
                  reinterpret_cast<const uint16_t*>(a.data_ptr()),
                  neg_expA.data_ptr<float>(), dt_bias.data_ptr<float>(),
                  z.data_ptr(), z.scalar_type() == torch::kBFloat16,
                  reinterpret_cast<const uint16_t*>(nw.data_ptr()), st.data_ptr<float>(),
                  out.data_ptr(), out.scalar_type() == torch::kBFloat16, (float)qscale,
                  nv, dk, dv, (int)rep, threads, c10::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
