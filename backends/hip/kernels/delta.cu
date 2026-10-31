// @file delta.cu
// @brief Gated DeltaNet 的解码步（Qwen3.5 的 24 个线性注意力层）。
//
// Module: coffin/kernels。消费 ``linear_attn.cu`` 产出的 fp32 q/k/v/beta/g，并原地更新
// 循环状态；批处理版本在 ``batch.cu``。
//
// 每个 token 的循环是
//     st = st * exp(g)[:, None, None]
//     kv = (st * k[:, :, None]).sum(-2)
//     st = st + k[:, :, None] * ((v - kv) * beta)[:, None, :]
//     out = (st * q[:, :, None]).sum(-2)
// 用一串 torch 算子写，折合每层约 140 us，即 24 层下来 3.4 ms/token，占整个解码步的
// 十分之一左右。每个线程独占 [dk, dv] 状态的第 j 列，而一列只由它自己的线程读写，因此
// 不需要任何跨线程通信，整个循环塌缩成一个内核。
//
// Constraints:
//   * ``dk`` 必须是 DELTA_SPLIT（4）的整数倍：每列的部分和要在 warp 内用三次
//     ``__shfl_xor`` 合并，换一种 dk 布局会悄悄改变 fp32 的求和顺序。
//   * 状态是原地更新的；前缀复用之前调用方必须把它和 KV 一起打快照，否则之后的一次恢复
//     会把共享的 key 配上一个已经往前走了的状态。
//   * 输入是 fp32 且连续存放：prologue 已经把 q/k 归一化和缩放好了，所以本内核不做任何
//     逐元素修补。

#include "hip_ops.h"

#include <c10/cuda/CUDAStream.h>
#include <hip/hip_runtime.h>

namespace {

//: 每个状态列由多少个 dk 切片分担。每列的部分和要在 warp 内用 ``__shfl_xor`` 合并，
//: 所以这里必须保持 2 的幂，且 ``dk % DELTA_SPLIT == 0`` 必须成立，否则状态布局就坏了。
#define DELTA_SPLIT 4

// @brief 每个（头，value 列的一块）一个 block；每列由 DELTA_SPLIT 个线程分担。
//
// 这 DELTA_SPLIT 个线程各拥有 dk 循环的一段，每列的部分和在 warp 内用两次 shuffle 合并。
//
// 早先的形状是「每列一个线程，blockDim = dv = 128」：64 个 CU 只启动了 32 个 block（半张
// 卡空转），而且每个线程要走一遍 2x128 次迭代的串行循环，每 CU 只有两个 warp 去掩盖 L2
// 延迟。实测在 int8 解码 profile 上 2294 us/步，占该步的 15.6%；把 dk 切成四份后串行链
// 缩短到 dk/4，block 数也上去了。
//
// k 和 q 被暂存进共享内存，因为每个线程都需要它们的全部 dk。第一遍刻意不把 e^g*S 写回：
// k^T(e^g S) = e^g(k^T S)，衰减留到第二遍再施加一次，于是省掉了每层每 token 对 2 MB
// 状态的一次读-改-写。
//
// @param st  [heads, dk, dv] fp32 循环状态，原地更新。
// @param out [heads, dv] fp32；由拥有 dk 切片 0 的 lane 写出，这样一列上的四个 lane 不会
//            互相竞争。
__global__ void delta_kernel(const float* __restrict__ q, const float* __restrict__ k,
                             const float* __restrict__ v, const float* __restrict__ beta,
                             const float* __restrict__ g, float* __restrict__ st,
                             float* __restrict__ out, int dk, int dv) {
    const int head = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int ii = lane & (DELTA_SPLIT - 1);              // dk 切片
    const int jj = lane >> 2;                             // warp 内的列
    const int cols_per_block = (blockDim.x >> 5) * (32 / DELTA_SPLIT);
    const int j = blockIdx.y * cols_per_block + warp * (32 / DELTA_SPLIT) + jj;

    extern __shared__ float shared[];
    float* ks = shared;
    float* qs = shared + dk;
    for (int i = threadIdx.x; i < dk; i += blockDim.x) {
        ks[i] = k[(size_t)head * dk + i];
        qs[i] = q[(size_t)head * dk + i];
    }
    __shared__ float decay, rate;
    if (threadIdx.x == 0) {
        decay = __expf(g[head]);
        rate = beta[head];
    }
    __syncthreads();

    if (j >= dv) return;                                  // 整个 4-lane 组一起退出
    float* column = st + (size_t)head * dk * dv + j;
    const float value = v[(size_t)head * dv + j];

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
    if (ii == 0) out[(size_t)head * dv + j] = result;
}

}  // namespace

// @brief 一步 gated delta 规则的解码：原地推进循环状态。
//
// @param q,k [heads, dk] fp32，已由 ``delta_prologue`` 做过 L2 归一化（q 还额外缩放
//            过），所以本内核不做任何逐元素修补。
// @param v [heads, dv] fp32。
// @param beta,g [heads] fp32 的写入速率与衰减。
// @param st [heads, dk, dv] fp32 状态，原地更新——前缀复用所依赖的快照纪律由调用方
//           负责。
// @return ``out``，[heads, dv] fp32，返回以便链式调用。
// @throws c10::Error（TORCH_CHECK）：非 beta/g 的输入不连续、头数与状态对不上，或 ``dk``
//         不是 DELTA_SPLIT 的整数倍（四 lane 的 shuffle 切不开它）。
torch::Tensor delta_step(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                         torch::Tensor beta, torch::Tensor g,
                         torch::Tensor st, torch::Tensor out) {
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous()
                && beta.is_contiguous() && g.is_contiguous(),
                "contiguous inputs required");
    const int heads = st.size(0), dk = st.size(1), dv = st.size(2);
    TORCH_CHECK(q.size(0) == heads && k.size(0) == heads && v.size(0) == heads,
                "q/k/v head count does not match the state");
    TORCH_CHECK(dk % DELTA_SPLIT == 0, "dk must be a multiple of DELTA_SPLIT");
    const int threads = 128, cols_per_block = (threads / 32) * (32 / DELTA_SPLIT);
    const int blocks_y = (dv + cols_per_block - 1) / cols_per_block;
    hipStream_t stream = at::cuda::getCurrentCUDAStream();
    delta_kernel<<<dim3(heads, blocks_y), threads, 2 * dk * sizeof(float), stream>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
        beta.data_ptr<float>(), g.data_ptr<float>(),
        st.data_ptr<float>(), out.data_ptr<float>(), dk, dv);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
