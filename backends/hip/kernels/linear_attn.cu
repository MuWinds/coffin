// @file linear_attn.cu
// @brief gated delta 规则解码步的融合 prologue、喂给它的深度因果 conv1d，以及两者的
//        预填充变体。
//
// Module: coffin/kernels。喂给 ``delta.cu``：本文件产出 ``delta_step`` 消费的 fp32
// q/k/v/beta/g，所以两边必须在布局和 1e-6 的 L2 epsilon 上保持一致。
//
// 为什么融合：在 delta 内核之前，引擎要对 q 和 k（逐头）做 L2 归一化、给 q 乘缩放、
// 把 16 个 key 头扩展成 32 个 value 头，并构建 beta = sigmoid(b) 与
// g = -exp(A) * softplus(a + dt_bias)。作为 torch 算子这相当于每层约 16 个小内核（mul、
// sum、rsqrt、sigmoid、softplus、add、repeat_interleave、类型转换），跨 24 层累积。
// 每个 value 头一个 block 就能全部做完，并写出 delta_kernel 直接可用的 fp32 q/k/v 与
// beta/g。
//
// conv1d 是它自己的一遍逐元素：在 8192 宽的激活上做 cat + mul + sum + silu，每个线性层一次。
//
// Constraints:
//   * 每个 block（或每个通道一个线程）独占自己的输出；没有跨 block 同步，卷积状态的
//     平移是逐通道的先读后写。
//   * 作为融合投影输出列切片的输入，行距（``ldx`` / ``ldb`` / ``lda``）取自张量本身；
//     在那里假定连续会读到相邻的三分之一。
//   * prologue 运行时 q/k/v 已是 fp32：深度卷积的 ``cat`` 配上 fp32 卷积状态，会把 bf16
//     的投影输出提升成 fp32。
//   * q/k 归一化里的 1e-6 epsilon 属于 checkpoint 的数值约定；改动它就会改变模型输出。

#include "hip_ops.h"

#include <c10/cuda/CUDAStream.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace {

__device__ __forceinline__ float bf16f(unsigned short h) {
    return __uint_as_float(((unsigned)h) << 16);
}

// 一次做两个值的 block 内求和——q 和 k 总是一起归一化，所以它们共用一趟归约而不是两趟。
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

// @brief 每个 value 头一个 block：q/k 的 L2 归一化、q 缩放、头扩展、beta/g。
//
// vh 是 value 头的下标；它对应的 key 头是 ``vh / rep``（repeat_interleave 的顺序）。
// q/k/v 进来时是 fp32（深度 conv1d 的 ``cat`` 配上 fp32 卷积状态把它们提升了），b/a 是
// 原始的 bf16 投影输出。
//
// @param qscale 折进归一化后 q 的缩放，delta 内核因此不必逐元素再乘一次。
// @param rep    每个 key 头对应的 value 头数，即每个 key 头被读多少次。
// @return 无；fp32 缓冲区原地写出，供 ``delta_step`` 使用。
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
        const float sp = av > 20.f ? av : __logf(1.f + __expf(av));   // softplus；>20 时取恒等
        g[vh] = neg_expA[vh] * sp;
    }
}

// @brief 深度因果 conv1d（卷积核 4），后接 SiLU。
//
// ``state`` 是前若干输入的 [C, 3] 环形缓冲，原地平移；线程 c 独占通道 c，所以三个抽头的
// 先读后写不需要任何同步。``w`` 是 [C, 4]：三个历史抽头，加上第 4 个抽头是当前输入。
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

// @brief 同样的深度因果 conv1d + SiLU，跑在整个预填充分块上。
//
// 线程 c 沿序列前进，把三抽头的环形缓冲留在寄存器里，所以这一趟只读一次 bf16 激活、写一次
// fp32。它取代的即时执行链（cat + 提升 fp32 + unfold + 乘 + 求和 + SiLU）要碰约六倍的字
// 节数，C=2048 时每层花 3.6 ms，而不可避免的访存只有约 0.15 ms。
//
// 在固定的 t 上，相邻线程读相邻通道，所以哪怕一个线程按 ldx 跨步，一个 warp 每步仍然覆盖
// 一整条 64 B 缓存行。
//
// @param ldx ``x`` 的行距：投影直接从融合 GEMV 的输出里读，其中 q/k/v 那三分之一是列切片，
//            并不连续。
// @param out [T, C] fp32，紧密排布（调用方按连续张量分配它）。
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

// @brief 启动单 token 的深度因果 conv1d + SiLU。
//
// @param conv_w [C, 4] fp32 抽头。
// @param x      [C] bf16 投影输出。
// @param conv_state [C, 3] fp32 历史，原地平移。
// @param out    [C] fp32；原地写出，不返回任何东西。
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

// @brief 启动单 token 的 gated delta prologue。
//
// @param q_raw,k_raw [nk, dk] fp32，q 还额外乘了 ``qscale``。
// @param v_raw [nv, dv] fp32；原样拷贝过去，好让 delta 内核只读一种 dtype。
// @param b,a   [nv] bf16 的原始投影输出，分别用于构建 beta 和 g。
// @param neg_expA,dt_bias [nv] fp32 的 checkpoint 常量，用于衰减门控。
// @throws c10::Error（TORCH_CHECK）：来自内核启动检查；形状在这里是被信任的，因为调用方
//         是按模型配置构造的。
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

// @brief 启动分块预填充的 conv1d + SiLU。
//
// @param x [T, C] bf16，内层步长为 1；行距会照实际取值处理，这样融合 GEMV 输出的列切片
//          可以就地读取。
// @return [T, C] fp32。卷积状态最终停在最后三个输入上，所以下一个解码步能接着同一条递推。
// @throws c10::Error（TORCH_CHECK）：``x`` 不是二维或内层步长不为 1；调换调用方的切片顺序，
//         别把这条检查放宽。
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

// 预填充 prologue：与解码那套数学相同，只是跑在整个分块上。用 torch 算子写，C=2048 时
// 每层约 1.7 ms（split、reshape、两次 repeat_interleave、两次 L2 归一化、sigmoid、
// softplus、add、mul、类型转换、三次转置）。
//
// 那几次转置比看起来更要紧：把一个转置过的 [H, C, d] 视图交给 ``chunked_delta``，会迫使它
// 内部做三次跨步拷贝，再花掉约 2.9 ms/层，因为在这颗芯片上分块的跨步拷贝只有约 70 GB/s，
// 而硬件能做到约 700 GB/s。在这里直接写出 [H, T, d] 就同时消掉了这两笔开销。

namespace {

// @brief 每个（value 头，token）一个 block：预填充的 q/k 归一化、头扩展、beta/g，按分块
//        扫描直接读取的 [H, T, d] 布局写出。
//
// 128 个线程覆盖该头的 dk 个元素，两次 block 归约正好需要这么多。q/k 是按 value 头读的，
// 所以 rep=2 时每个 key 头会被归一化两次——多读 q/k（C=2048 时 100 MB），换来每层少约 12
// 次内核启动，而且这些多出来的读命中 L2。
//
// @param qkv [T, conv_dim] fp32 卷积输出，按 q | k | v 排布，行距为 ``conv_dim``；q/k/v 三个
//            头的元素是该行的列切片。
// @param b,a [T, nv] bf16，行距分别是 ``ldb`` / ``lda``（融合 GEMV 输出的列切片）。
// @param kdim num_key_heads * dk，即 k 那三分之一在一行内的偏移。
// @return 无；五个输出全部原地写出。
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

    // qkv 是 [T, conv_dim] 的卷积输出，按 q | k | v 排布。
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
        const float sp = av > 20.f ? av : __logf(1.f + __expf(av));   // softplus；>20 时取恒等
        g[(size_t)vh * T + t] = neg_expA[vh] * sp;
    }
}

}  // namespace

// @brief 启动分块预填充的 gated delta prologue。
//
// @param qkv [T, conv_dim] fp32 卷积输出，按 q | k | v 排布。
// @param q_out,k_out [nv, T, dk] fp32 与 ``v_out`` [nv, T, dv] fp32；这个布局正是本内核的
//        意义所在，见它上面的注释。
// @param beta,g [nv, T] fp32。
// @throws c10::Error（TORCH_CHECK）：``qkv``/``b``/``a`` 的内层步长不为 1；请传切片，不要
//         传转置。
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
