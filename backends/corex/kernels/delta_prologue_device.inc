// @file delta_prologue_device.inc
// @brief 线性注意力头前处理的设备代码：q/k 归一化、头扩展、beta/g 门，三个入口共用一套。
//
// Module: coffin/backends/corex/kernels。被两条前端各包含一次：``linear_attn.cu``（torch
// 扩展）与 ``abi/delta_prologue_abi.cu``（``extern "C"`` 出口）。同一份设备代码。
//
// 为什么值得搬：它是 prefill 每层都要走的一段，而三个入口的算术只差「行轴在哪」——
// 单 token 的网格是 ``nv``，批处理是 ``(nv, rows)``，分块预填充是 ``(nv, T)`` 并写出分块
// 扫描要读的头主序布局。因此三条内核共用 ``head_threads`` 与同一个归约实现。
//
// Constraints:
//   * ``delta_prep_seq`` 要求 ``qkv`` / ``b`` / ``a`` 的最内层步长为 1（列切片的视图），
//     行间距单独传参。
//   * ``beta`` 是写强度、``g`` 是对数衰减（``<= 0``）：两处都经过 ``softplus`` / ``-exp``
//     的口径，见内核上方的注释；少一次窄化就是整条曲线平移。
//   * ``head_threads(dk)`` 是三项归约的伙伴函数，包装层与内核必须用同一个值。

#ifndef COFFIN_DELTA_PROLOGUE_DEVICE_INC
#define COFFIN_DELTA_PROLOGUE_DEVICE_INC

#include "corex_scalar.h"

#include <cstdint>

//: checkpoint 的 q/k L2 归一化 epsilon。属于模型的数值行为。
constexpr float L2_EPS = 1e-6f;

//: 超过这个值，``softplus(x) == x`` 在 fp32 精度下成立，于是跳过 ``exp``。logit
//: ``a + dt_bias`` 上方无界，而大值的 ``exp`` 会溢出。
constexpr float SOFTPLUS_LINEAR = 20.f;

//: 头归一化 block 的线程数：这个模型 ``dk`` 是 128，而归约需要 wavefront 的整数倍。
int head_threads(int dk) {
    int t = dk > 128 ? dk : 128;
    t = ((t + 63) / 64) * 64;
    return t > 512 ? 512 : t;
}

//: ``norm.cu`` 给宽度为 ``N`` 的一行用多少线程，照抄过来是为了检查融合内核能否顶替门控
//: 归一化。
//:
//: 这里的取整故意是同一个表达式而不是一次调用：``rms_norm_gated_kernel`` 在另一个编译
//: 单元里、藏在匿名命名空间之后，没有东西可调。这份重复是被校验的而不是被信任的——任何
//: 漂移都会让 ``delta_normed`` 拒绝一个它本来能服务的几何，那是有记录的回退；反过来
//: （接受一个归约配对不同的几何）则会悄悄改变 id。
int norm_threads(int n) {
    int t = n < 64 ? 64 : n;
    t = ((t + 63) / 64) * 64;
    return t > 1024 ? 1024 : t;
}

// @brief 一次通过共享树在块内求两个值的和。
//
// q 和 k 一起归一化，所以一趟就够。分成两次归约会把这个完全受延迟限制的内核的屏障数量
// 翻倍：头只有 128 个元素，block 没有别的东西能在同步后面遮掩。
//
// @param a 第一份贡献（q 的平方和）。
// @param b 第二份贡献（k 的平方和）。
// @param red ``2 * blockDim.x`` 个 float 的共享暂存；会被覆写。
// @param sum_a 输出参数，接收 ``a`` 的块内和。
// @param sum_b 输出参数，接收 ``b`` 的块内和。
//
// 结尾的屏障是必需的。两个和由每个线程从 ``red`` 里读出去，而块随后会写别的共享区域，
// 那些区域可能铺在同一片内存上。没有屏障，快线程会在慢线程读完之前就走到那次写——
// 就是在 ``delta.cu`` 对应辅助函数里造成 4.5e-02 误差的那个 bug。
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

// @brief 尾部取线性的 softplus，让大 logit 不会把指数撑溢出。
//
// @param v logit，即 ``a + dt_bias``。
// @return ``v`` 适中时是 ``log(1 + exp(v))``，很大时就是 ``v`` 本身。
__device__ __forceinline__ float softplus(float v) {
    return v > SOFTPLUS_LINEAR ? v : __logf(1.f + __expf(v));
}

// @brief 单个值的块内和，配对方式与 :func:`block_sum2` 完全一致。
//
// 门控归一化的归约是 ``delta_normed`` 的另一半，它必须按与 ``rms_norm_gated_kernel`` 的树
// 相同的顺序组合相同的伙伴，否则 scale 会在最后一个 ulp 上移动。共用树的形状而不是重新
// 推导它，才让这件事成为结构性的，而不是靠眼睛保持一致。这里不能用 ``block_sum2``：它求
// 两个值的和，而尾声只有一个，加一列哑数据会让内核最后阶段的共享流量翻倍。
//
// @param v 本线程的贡献。
// @param red ``blockDim.x`` 个 float 的共享暂存；会被覆写。
// @return 整个块的和，在结尾同步之后每个线程都拿得到。
//
// 结尾的屏障是必需的，理由和 ``block_sum2`` 里一样：调用方从 ``red[0]`` 把和读出去，不能
// 让更快的线程先开始覆写 ``red``。``delta_normed_kernel`` 确实会撞上这个风险——它在
// ``block_sum2`` 返回之后立刻把 ``red`` 复用于这次归约——所以屏障在那里是承重的，没有
// 省略。
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

// @brief 为一个解码步构造 fp32 的 q/k/v 与 beta/g 门。
//
// 每个值头一个 block：block 归一化这个头的 q 和 k（与同一键头的其它值头共享，所以每个
// 键头会被归一化 ``rep`` 次——多出来的读取都是 L2 命中，另一条路是再来一次启动）、缩放
// q，并从原始 logits 写出 beta/g。
//
// @param q_raw [nkeys, dk] fp32 查询，每个键头一份。
// @param k_raw [nkeys, dk] fp32 键，每个键头一份。
// @param v_raw [nv, dv] fp32 值；原样拷过去，让 delta 步只读一种 dtype。
// @param b,a [nv] bf16 原始 logits，喂给 beta 和 g。
// @param neg_expA,dt_bias [nv] fp32，衰减门的 checkpoint 常量。
// @param q_out [nv, dk] fp32，归一化并缩放后的查询。
// @param k_out [nv, dk] fp32，归一化后的键。
// @param v_out [nv, dv] fp32 值。
// @param beta [nv] fp32 写强度。
// @param g [nv] fp32 对数衰减，``<= 0``。
// @param qscale ``dk ** -0.5``，折进归一化后的查询。
// @param rep 每个键头对应的值头数。
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

// @brief 预填充的头预处理：整个分块上做一遍解码预处理，输出头主序布局。
//
// 每个（值头，token）一个 block。这些转置比看起来更要紧：把转置过的 ``[H, C, d]`` 视图
// 交给分块 delta 扫描，会在它内部逼出三次带步长的拷贝，而在本部件上分块的带步长拷贝
// 远跑不到可达带宽。在这里直接写出 ``[H, T, d]`` 就把这三次全消掉了。
//
// @param qkv [T, conv_dim] fp32 卷积输出，按 q | k | v 排列；各头是这一行的列切片。
// @param b,a [T, nv] bf16，行间距 ``ldb`` / ``lda``。
// @param q_out,k_out [nv, T, dk] fp32；``v_out`` [nv, T, dv] fp32。
// @param beta,g [nv, T] fp32。
// @param kdim ``num_key_heads * dk``，k 那一段在一行内的偏移。
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

// @brief 批处理预处理：每个（行，值头）一个 block。
//
// @param q,k [B, nkeys, dk] fp32；``v`` [B, nv, dv] fp32。
// @param b,a [B, nv] bf16。
// @param q_out,k_out [B, nv, dk] fp32；``v_out`` [B, nv, dv] fp32。
// @param beta,g [B, nv] fp32。
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

#endif  // COFFIN_DELTA_PROLOGUE_DEVICE_INC
