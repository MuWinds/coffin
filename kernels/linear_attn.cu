// Fused prologue for the gated-delta-rule decode step, plus the depthwise
// causal conv1d that feeds it.
//
// Before the delta kernel the engine L2-normalises q and k (per head), scales q,
// expands the 16 key heads to 32 value heads, and builds beta = sigmoid(b) and
// g = -exp(A) * softplus(a + dt_bias). As torch ops that is ~16 small kernels per
// layer (mul, sum, rsqrt, sigmoid, softplus, add, repeat_interleave, casts) over
// 24 layers. One block per value head does all of it and writes fp32 q/k/v plus
// beta/g ready for delta_kernel.
//
// The conv1d (kernel 4, causal) is its own elementwise pass: cat + mul + sum +
// silu over the 8192-wide activation, once per linear layer.

#include "hip_ops.h"

#include <c10/cuda/CUDAStream.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace {

__device__ __forceinline__ float bf16f(unsigned short h) {
    return __uint_as_float(((unsigned)h) << 16);
}

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

// vh indexes the value head; its key head is vh / rep (repeat_interleave).
// q/k/v arrive fp32 (the depthwise conv1d ``cat`` with the fp32 conv state
// promotes them), b/a are the raw bf16 projection outputs.
__global__ void prologue_kernel(const float* __restrict__ q_raw,
                                const float* __restrict__ k_raw,
                                const float* __restrict__ v_raw,
                                const unsigned short* __restrict__ b,
                                const unsigned short* __restrict__ a,
                                const float* __restrict__ neg_expA,
                                const float* __restrict__ dt_bias,
                                float* __restrict__ q_out,
                                float* __restrict__ k_out,
                                float* __restrict__ v_out,
                                float* __restrict__ beta,
                                float* __restrict__ g,
                                float qscale, int dk, int dv, int rep) {
    const int vh = blockIdx.x;
    const int t = threadIdx.x;
    const int kh = vh / rep;
    const float* qr = q_raw + (size_t)kh * dk;
    const float* kr = k_raw + (size_t)kh * dk;

    const float qv = t < dk ? qr[t] : 0.f;
    const float kv = t < dk ? kr[t] : 0.f;
    __shared__ float2 red[256];
    const float2 ss = block_sum2(qv * qv, kv * kv, red);

    if (t < dk) {
        q_out[(size_t)vh * dk + t] = qv * rsqrtf(ss.x + 1e-6f) * qscale;
        k_out[(size_t)vh * dk + t] = kv * rsqrtf(ss.y + 1e-6f);
    }
    if (t < dv) v_out[(size_t)vh * dv + t] = v_raw[(size_t)vh * dv + t];
    if (t == 0) {
        beta[vh] = 1.f / (1.f + __expf(-bf16f(b[vh])));
        const float av = bf16f(a[vh]) + dt_bias[vh];
        const float sp = av > 20.f ? av : __logf(1.f + __expf(av));   // softplus
        g[vh] = neg_expA[vh] * sp;
    }
}

// Depthwise causal conv1d (kernel 4) followed by SiLU. ``state`` is the [C, 3]
// ring of previous inputs, shifted in place; thread c owns channel c, so the
// read-then-write of the three taps needs no synchronisation.
__global__ void conv1d_silu_kernel(const float* __restrict__ w,
                                   const unsigned short* __restrict__ x,
                                   float* __restrict__ state,
                                   float* __restrict__ out, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float* s = state + (size_t)c * 3;
    const float* wc = w + (size_t)c * 4;
    const float x3 = bf16f(x[c]);
    const float acc = fmaf(s[0], wc[0], fmaf(s[1], wc[1], fmaf(s[2], wc[2], x3 * wc[3])));
    out[c] = acc / (1.f + __expf(-acc));     // silu
    s[0] = s[1];
    s[1] = s[2];
    s[2] = x3;
}

// Same depthwise causal conv1d + SiLU, but for a whole prefill chunk instead of
// one token.  Thread c walks the sequence and keeps the three-tap ring in
// registers, so the pass reads the bf16 activation once and writes fp32 once.
// The eager chain it replaces (cat + fp32 promote + unfold + multiply + sum +
// SiLU) touches about six times the bytes and costs 3.6 ms per layer at C=2048,
// against ~0.15 ms of unavoidable traffic.
//
// At a fixed t, neighbouring threads read neighbouring channels, so a warp still
// covers a full 64 B cache line per step even though one thread strides by ldx.
// ``ldx`` is the row pitch of ``x``: the projections are read straight out of the
// fused GEMV output, where the q/k/v third is a column slice and not contiguous.
__global__ void conv1d_silu_seq_kernel(const float* __restrict__ w,
                                       const unsigned short* __restrict__ x,
                                       float* __restrict__ state,
                                       float* __restrict__ out, int C, int T, int ldx) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const float* wc = w + (size_t)c * 4;
    const float w0 = wc[0], w1 = wc[1], w2 = wc[2], w3 = wc[3];
    float* s = state + (size_t)c * 3;
    float s0 = s[0], s1 = s[1], s2 = s[2];
    const unsigned short* xp = x + c;
    float* o = out + c;
#pragma unroll 4
    for (int t = 0; t < T; ++t) {
        const float x3 = bf16f(xp[(size_t)t * ldx]);
        const float acc = fmaf(w0, s0, fmaf(w1, s1, fmaf(w2, s2, w3 * x3)));
        o[(size_t)t * C] = acc / (1.f + __expf(-acc));     // silu
        s0 = s1;
        s1 = s2;
        s2 = x3;
    }
    s[0] = s0;
    s[1] = s1;
    s[2] = s2;
}

}  // namespace

void conv1d_silu(torch::Tensor conv_w, torch::Tensor x, torch::Tensor conv_state,
                 torch::Tensor out) {
    const int C = x.size(0);
    hipStream_t stream = at::cuda::getCurrentCUDAStream();
    conv1d_silu_kernel<<<(C + 255) / 256, 256, 0, stream>>>(
        conv_w.data_ptr<float>(),
        reinterpret_cast<const unsigned short*>(x.data_ptr<at::BFloat16>()),
        conv_state.data_ptr<float>(), out.data_ptr<float>(), C);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void delta_prologue(torch::Tensor q_raw, torch::Tensor k_raw, torch::Tensor v_raw,
                    torch::Tensor b, torch::Tensor a,
                    torch::Tensor neg_expA, torch::Tensor dt_bias,
                    torch::Tensor q_out, torch::Tensor k_out, torch::Tensor v_out,
                    torch::Tensor beta, torch::Tensor g,
                    double qscale, int64_t rep) {
    const int nv = v_raw.size(0), dk = q_raw.size(1), dv = v_raw.size(1);
    hipStream_t stream = at::cuda::getCurrentCUDAStream();
    prologue_kernel<<<nv, 128, 0, stream>>>(
        q_raw.data_ptr<float>(),
        k_raw.data_ptr<float>(),
        v_raw.data_ptr<float>(),
        reinterpret_cast<const unsigned short*>(b.data_ptr<at::BFloat16>()),
        reinterpret_cast<const unsigned short*>(a.data_ptr<at::BFloat16>()),
        neg_expA.data_ptr<float>(), dt_bias.data_ptr<float>(),
        q_out.data_ptr<float>(), k_out.data_ptr<float>(), v_out.data_ptr<float>(),
        beta.data_ptr<float>(), g.data_ptr<float>(),
        (float)qscale, dk, dv, (int)rep);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor conv1d_silu_seq(torch::Tensor conv_w, torch::Tensor x,
                              torch::Tensor conv_state) {
    TORCH_CHECK(x.dim() == 2 && x.stride(1) == 1, "x must be [T, C] with unit inner stride");
    const int T = x.size(0), C = x.size(1), ldx = (int)x.stride(0);
    auto out = torch::empty({T, C}, x.options().dtype(at::kFloat));
    hipStream_t stream = at::cuda::getCurrentCUDAStream();
    conv1d_silu_seq_kernel<<<(C + 255) / 256, 256, 0, stream>>>(
        conv_w.data_ptr<float>(),
        reinterpret_cast<const unsigned short*>(x.data_ptr<at::BFloat16>()),
        conv_state.data_ptr<float>(), out.data_ptr<float>(), C, T, ldx);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// Same prologue as the decode step, but over a whole prefill chunk and writing
// the buffers the *chunked* scan wants.  The eager chain it replaces (split,
// reshape, repeat_interleave x2, two L2 norms, sigmoid, softplus, add, multiply,
// casts, three transposes) is ~1.7 ms per layer at C=2048, and the transposes
// matter more than they look: handing ``chunked_delta`` a transposed [H, C, d]
// view forces three strided copies inside it that cost another ~2.9 ms/layer,
// because a blocked strided copy on this part runs at ~70 GB/s rather than the
// ~700 GB/s the hardware can do.  Writing [H, T, d] here removes both.
//
// One block per (value head, token); the 128 threads cover that head's dk
// elements, which is what the two block reductions need.  q/k are read per value
// head, so with rep=2 each key head is normalised twice -- that is 2 extra reads
// of q/k (100 MB at C=2048) against ~12 removed kernel launches per layer.
namespace {

__global__ void delta_prep_seq_kernel(const float* __restrict__ qkv,
                                      const unsigned short* __restrict__ b,
                                      const unsigned short* __restrict__ a,
                                      const float* __restrict__ neg_expA,
                                      const float* __restrict__ dt_bias,
                                      float* __restrict__ q_out,
                                      float* __restrict__ k_out,
                                      float* __restrict__ v_out,
                                      float* __restrict__ beta,
                                      float* __restrict__ g,
                                      float qscale, int T, int dk, int dv, int rep,
                                      int kdim, int conv_dim, int ldb, int lda) {
    const int vh = blockIdx.x;
    const int t = blockIdx.y;
    const int i = threadIdx.x;
    const int kh = vh / rep;

    // qkv is the [T, conv_dim] conv output, laid out q | k | v.
    const float* row = qkv + (size_t)t * conv_dim;
    const float qv = i < dk ? row[(size_t)kh * dk + i] : 0.f;
    const float kv = i < dk ? row[kdim + (size_t)kh * dk + i] : 0.f;

    __shared__ float2 red[256];
    const float2 ss = block_sum2(qv * qv, kv * kv, red);

    if (i < dk) {
        q_out[((size_t)vh * T + t) * dk + i] = qv * rsqrtf(ss.x + 1e-6f) * qscale;
        k_out[((size_t)vh * T + t) * dk + i] = kv * rsqrtf(ss.y + 1e-6f);
    }
    if (i < dv) {
        v_out[((size_t)vh * T + t) * dv + i] = row[2 * kdim + (size_t)vh * dv + i];
    }
    if (i == 0) {
        beta[(size_t)vh * T + t] = 1.f / (1.f + __expf(-bf16f(b[(size_t)t * ldb + vh])));
        const float av = bf16f(a[(size_t)t * lda + vh]) + dt_bias[vh];
        const float sp = av > 20.f ? av : __logf(1.f + __expf(av));   // softplus
        g[(size_t)vh * T + t] = neg_expA[vh] * sp;
    }
}

}  // namespace

void delta_prep_seq(torch::Tensor qkv, torch::Tensor b, torch::Tensor a,
                    torch::Tensor neg_expA, torch::Tensor dt_bias,
                    torch::Tensor q_out, torch::Tensor k_out, torch::Tensor v_out,
                    torch::Tensor beta, torch::Tensor g,
                    double qscale, int64_t rep) {
    const int T = qkv.size(0);
    const int nv = q_out.size(0), dk = q_out.size(2), dv = v_out.size(2);
    const int kdim = dk * (int)(q_out.size(0) / rep);      // num_key_heads * dk
    TORCH_CHECK(qkv.stride(1) == 1 && b.stride(1) == 1 && a.stride(1) == 1,
                "unit inner stride required");
    hipStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(nv, T);
    delta_prep_seq_kernel<<<grid, 128, 0, stream>>>(
        qkv.data_ptr<float>(),
        reinterpret_cast<const unsigned short*>(b.data_ptr<at::BFloat16>()),
        reinterpret_cast<const unsigned short*>(a.data_ptr<at::BFloat16>()),
        neg_expA.data_ptr<float>(), dt_bias.data_ptr<float>(),
        q_out.data_ptr<float>(), k_out.data_ptr<float>(), v_out.data_ptr<float>(),
        beta.data_ptr<float>(), g.data_ptr<float>(),
        (float)qscale, T, dk, dv, (int)rep, kdim, (int)qkv.size(1),
        (int)b.stride(0), (int)a.stride(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
