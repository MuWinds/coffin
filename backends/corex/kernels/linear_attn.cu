// @file linear_attn.cu
// @brief The gated-delta-rule prologue and the depthwise causal conv1d that feeds it.
//
// Module: coffin/backends/corex/kernels. Feeds the delta recursion the way
// ``backends/hip/kernels/linear_attn.cu`` does on the DCU: this file produces the fp32
// q/k/v/beta/g that the delta step consumes, so the two must agree on the layout and on
// the 1e-6 L2 epsilon.
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
// @param w [C, 4] fp32 taps, oldest first; the last tap multiplies the current input.
// @param x [C] bf16 projection output.
// @param state [C, 3] fp32 previous inputs, shifted in place.
// @param out [C] fp32 conv + SiLU result.
__global__ void conv1d_silu_kernel(const float* __restrict__ w,
                                   const uint16_t* __restrict__ x,
                                   float* __restrict__ state, float* __restrict__ out,
                                   int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float* s = state + (size_t)c * CONV_STATE;
    const float* wc = w + (size_t)c * CONV_K;
    const float x3 = CxScalar<uint16_t>::to_float(x[c]);
    const float acc = fmaf(s[0], wc[0],
                           fmaf(s[1], wc[1], fmaf(s[2], wc[2], x3 * wc[3])));
    const float y = acc / (1.f + __expf(-acc));     // SiLU
    out[c] = y;
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

}  // namespace

// @brief Launch the single-token depthwise causal conv1d + SiLU.
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
    conv1d_silu_kernel<<<(C + 255) / 256, 256, 0, stream>>>(
        conv_w.data_ptr<float>(), reinterpret_cast<const uint16_t*>(x.data_ptr()),
        conv_state.data_ptr<float>(), out.data_ptr<float>(), C);
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

// @brief Launch the batched single-token conv1d + SiLU: one window per row.
//
// @param conv_w [C, 4] fp32 taps, shared by every row.
// @param x [B, C] bf16.
// @param conv_state [B, C, 3] fp32, shifted in place.
// @param out [B, C] fp32, written in place.
// @throws c10::Error on a dtype or shape the kernel cannot index.
void conv1d_silu_batch(torch::Tensor conv_w, torch::Tensor x, torch::Tensor conv_state,
                       torch::Tensor out) {
    TORCH_CHECK(x.dim() == 2, "activation must be [B, C]");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "activation must be bf16");
    const int rows = (int)x.size(0);
    const int C = (int)x.size(1);
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    for (int b = 0; b < rows; ++b) {
        // One launch per row rather than a second grid axis: the row's window is a
        // separate ``[C, 3]`` slab and ``C`` already fills the grid, so a loop costs the
        // same number of launches and keeps the addressing trivially correct.
        conv1d_silu_kernel<<<(C + 255) / 256, 256, 0, stream>>>(
            conv_w.data_ptr<float>(),
            reinterpret_cast<const uint16_t*>(x.data_ptr()) + (size_t)b * C,
            conv_state.data_ptr<float>() + (size_t)b * C * CONV_STATE,
            out.data_ptr<float>() + (size_t)b * C, C);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
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
