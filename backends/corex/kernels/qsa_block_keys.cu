// @file qsa_block_keys.cu
// @brief QSA 索引器的块 key：gather → 池化 → RMSNorm → 部分 RoPE，一次启动。
//
// Module: coffin/backends/corex/kernels。由 ``backends/corex`` 的
// ``qsa_block_keys`` 调用；``models/qwen4exp/qsa.py`` 的
// ``_batched_block_keys`` 用它替换那串 torch 算子。
//
// ## 为什么需要它
//
// 作业 ``q4sel1`` 把 ``QSAIndexer.select`` 的八段分开量（decode、单层）：
//
//     kv      selector  block_table  block_keys  scoring  topk   dense_mask   合计
//     4096     0.363      0.356        0.322      0.064   0.320    0.413     2.13 ms
//     8192     0.362      0.354        0.398      0.067   0.326    0.431     2.24 ms
//     32768    0.397      0.384        1.067      0.090   0.417    0.452     3.29 ms
//
// 12 个 QSA 层就是 **25.6 / 26.9 / 39.5 ms**。没有任何一段占绝对多数（最大的 38%），
// 所以要打的是**整条链**，而不是挑一段。
//
// 本文件先拿 ``block_keys``：它是长上下文下的最大单段（kv=32768 时 38%）。这一段
// 要读的数据其实极小——``max_blocks x ratio x head_dim`` 个 bf16，kv=32768 时是
// 8192 x 4 x 128 x 2 B = **8.4 MB**，按 575 GB/s 的屋顶是 **0.015 ms**，实测 1.067 ms，
// 差 71 倍。差在那一串中间量上（gather、``.float()`` 的 fp32 池化、layernorm 的
// 多次 elementwise、RoPE 的 ``cat``），每一步都是一次全量读写。
//
// ## 数值口径：必须与 torch 链逐位对齐到 bf16 容差
//
// 原链的顺序是：
//
//     gathered = keys[idx]                        # bf16
//     pooled   = layernorm(gathered.float().mean(1).to(bf16))   # fp32 求均值 → bf16
//     rotated  = rope(pooled, cos, sin)            # bf16 的 mul/add/cat
//
// 三处容易写错的语义：
//
// 1. **RMSNorm 是零中心的**：``norm.py`` 里增益是 ``1 + weight``（``weight`` 初始为
//    全零），不是常见的 ``weight``。按 ``weight`` 直接乘会得到一个恒等变换。
// 2. **均值先在 fp32 里算，然后一次舍入到 bf16**，之后归一化在 bf16 上做。
//    这里照抄：``sum(ratio 个 bf16 值) / ratio`` 在 fp32 里累加，再舍入。
// 3. **RoPE 是半切分形式**：``rotate_half(x) = cat(-x[half:], x[:half])``，
//    不是交错形式。且只有前 ``rotary_dim`` 维参与旋转，其余原样输出
//    （partial rotary）。
//
// 累加次序与 torch 的 ``mean`` 不同（它用分块归约），所以判据是**相对误差**而不是
// ``torch.equal``；bf16 机器精度是 2^-8 = 3.9e-3，判据取 2e-2。
//
// Constraints:
//   * 一个块负责一批（块行）；块内线程按 ``head_dim`` 分摊通道。
//   * ``ratio`` 必须 <= ``MAX_RATIO``；超了由包装层拒绝（本模型的 ratio = 4）。
//   * ``-1`` 补位槽统一取 0 号 token：它们的 key 是垃圾，但打分会被 ``-inf`` 压掉，
//     任何取值都不影响结果，而 clamp 换掉了逐块判空的分支。
//   * 不用 CUDA graph、不用 cp.async（``platform/scow/AGENTS.md`` 第 1、5 条）。

#include "corex_ops.h"
#include "corex_scalar.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

namespace {

//: 一个块处理的通道数上限，必须与 blockDim 一致。
constexpr int BLOCK_THREADS = 128;

//: ``compress_ratio`` 的上限。本模型是 4；留 16 倍余量，超了由包装层拒绝。
constexpr int MAX_RATIO = 16;

// @brief 块内求和（折半归约 + 共享内存）。
//
// @param sh 共享内存暂存，长度至少 ``blockDim.x``。
// @param v  本线程的值。
// @return 全块求和。
__device__ __forceinline__ float block_sum(float* sh, float v) {
    const int t = threadIdx.x;
    const int nt = blockDim.x;
    sh[t] = v;
    __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) {
        if (t < s) sh[t] += sh[t + s];
        __syncthreads();
    }
    const float out = sh[0];
    __syncthreads();
    return out;
}

// @brief 对规整表里的每个块算池化 key：gather → mean → RMSNorm → partial RoPE。
//
// @param flat_keys ``[batch * tokens, head_dim]`` bf16 的原始 key 展平视图。
// @param table     ``[rows, max_blocks, ratio]`` int32 规整块表，``-1`` 是补位。
// @param weight    ``[head_dim]`` bf16 的零中心增益（实际乘 ``1 + w``）。
// @param flat_cos  ``[batch * tokens, rotary_dim]`` bf16。
// @param flat_sin  同 ``flat_cos``。
// @param out       ``[rows * max_blocks * head_dim]`` bf16 输出。
// @param block_row ``[rows * max_blocks]`` 每个块行属于哪个 query 行，用来推 batch。
// @param rows, seq, tokens, max_blocks 几何。
// @param ratio     ``compress_ratio``。
// @param head_dim  索引器每头维数。
// @param rotary_dim 参与旋转的维数（前若干维），其余原样输出。
// @param eps       RMSNorm 的 epsilon。
__global__ void qsa_block_keys_kernel(const uint16_t* __restrict__ flat_keys,
                                      const int* __restrict__ table,
                                      const uint16_t* __restrict__ weight,
                                      const uint16_t* __restrict__ flat_cos,
                                      const uint16_t* __restrict__ flat_sin,
                                      uint16_t* __restrict__ out,
                                      const int* __restrict__ block_rows,
                                      const int* __restrict__ block_batch,
                                      int rows, int seq, int tokens, int max_blocks,
                                      int ratio, int head_dim, int rotary_dim,
                                      float eps) {
    const int total = rows * max_blocks;
    const int blk = blockIdx.x;
    if (blk >= total) return;
    const int t = threadIdx.x;
    const int nt = blockDim.x;

    extern __shared__ float sh[];
    float* red = sh;                       // [nt] 归约暂存
    float* pooled = red + nt;              // [head_dim] fp32 池化结果

    // 该块的 ``ratio`` 个 token 在展平 key 里的行号。
    const int* slot = table + (size_t)blk * ratio;
    const int batch_row = block_batch[blk];

    // 均值：每线程负责 ``head_dim / nt`` 个通道，串 ratio 个 token。
    //
    // 在 fp32 里累加，与 torch 链的 ``gathered.float().mean(1)`` 同口径；最后
    // 一次舍入到 bf16 与 ``.to(raw_keys.dtype)`` 对应。
    for (int d = t; d < head_dim; d += nt) {
        float acc = 0.f;
        for (int r = 0; r < ratio; ++r) {
            // 补位槽取 0 号 token：取值无关紧要（下游会被 -inf 压掉），
            // 但省掉了逐块判空的分支。
            const int p = slot[r] < 0 ? 0 : slot[r];
            const uint16_t* row = flat_keys + ((size_t)batch_row * tokens + (size_t)p)
                                 * (size_t)head_dim;
            acc += CxScalar<uint16_t>::to_float(row[d]);
        }
        pooled[d] = CxScalar<uint16_t>::to_float(
            CxScalar<uint16_t>::from_float(acc / (float)ratio));
    }
    __syncthreads();

    // RMSNorm：零中心增益 ``1 + w``。均方在 fp32 里求，与 norm.py 一致。
    float sq = 0.f;
    for (int d = t; d < head_dim; d += nt) sq += pooled[d] * pooled[d];
    const float ms = block_sum(red, sq) / (float)head_dim;
    const float rsqrt = rsqrtf(ms + eps);
    __syncthreads();
    for (int d = t; d < head_dim; d += nt) {
        // ``norm.py`` 的增益是 ``1 + weight``（weight 初始全零），不是 ``weight``。
        // 按 weight 直接乘会得到一个恒等变换 —— 这是本文件最容易写错的一处。
        const float gain = 1.f + CxScalar<uint16_t>::to_float(weight[d]);
        pooled[d] = CxScalar<uint16_t>::to_float(
            CxScalar<uint16_t>::from_float(pooled[d] * rsqrt * gain));
    }
    __syncthreads();

    // Partial RoPE，半切分形式：``rotate_half(x) = cat(-x[half:], x[:half])``。
    //
    // 块首位置（``table[blk][0]``）决定 cos/sin 的行；补位槽同样取 0。
    const int pos = slot[0] < 0 ? 0 : slot[0];
    const uint16_t* crow = flat_cos + ((size_t)batch_row * tokens + (size_t)pos)
                           * (size_t)rotary_dim;
    const uint16_t* srow = flat_sin + ((size_t)batch_row * tokens + (size_t)pos)
                           * (size_t)rotary_dim;
    uint16_t* dst = out + (size_t)blk * (size_t)head_dim;
    const int half = rotary_dim / 2;

    for (int d = t; d < head_dim; d += nt) {
        float v = pooled[d];
        if (d < rotary_dim) {
            // 半切分：前半用 ``-x[d+half]``，后半用 ``x[d-half]``。
            const int partner = (d < half) ? (d + half) : (d - half);
            const float other = pooled[partner];
            const float rot = (d < half) ? -other : other;
            v = v * CxScalar<uint16_t>::to_float(crow[d])
              + rot * CxScalar<uint16_t>::to_float(srow[d]);
        }
        dst[d] = CxScalar<uint16_t>::from_float(v);
    }
}

}  // namespace

// @brief QSA 块 key 的融合内核入口：gather → 池化 → RMSNorm → partial RoPE。
//
// @param flat_keys ``[batch * tokens, head_dim]`` bf16 连续。
// @param table     ``[rows, max_blocks, ratio]`` int32 连续，``-1`` 是补位。
// @param weight    ``[head_dim]`` bf16 连续，零中心增益。
// @param flat_cos  ``[batch * tokens, rotary_dim]`` bf16 连续。
// @param flat_sin  同 ``flat_cos``。
// @param block_row ``[rows * max_blocks]`` int32，块行到 query 行的映射。
// @param block_batch ``[rows * max_blocks]`` int32，块行到 batch 行的映射。
// @param out       ``[rows, max_blocks, head_dim]`` bf16 连续。
// @param eps       RMSNorm 的 epsilon。
// @return out。
// @throws c10::Error 布局、dtype、形状或 ``ratio`` 越界。
torch::Tensor qsa_block_keys(torch::Tensor flat_keys, torch::Tensor table,
                             torch::Tensor weight, torch::Tensor flat_cos,
                             torch::Tensor flat_sin, torch::Tensor block_row,
                             torch::Tensor block_batch, torch::Tensor out,
                             int64_t seq, int64_t tokens, int64_t rotary_dim,
                             double eps) {
    TORCH_CHECK(flat_keys.is_cuda() && flat_keys.dim() == 2 && flat_keys.is_contiguous()
                && flat_keys.scalar_type() == torch::kBFloat16,
                "flat_keys must be contiguous CUDA bf16 [batch*tokens, head_dim]");
    TORCH_CHECK(table.is_cuda() && table.dim() == 3 && table.is_contiguous()
                && table.scalar_type() == torch::kInt,
                "table must be contiguous CUDA int32 [rows, max_blocks, ratio]");
    TORCH_CHECK(out.is_cuda() && out.dim() == 3 && out.is_contiguous()
                && out.scalar_type() == torch::kBFloat16,
                "out must be contiguous CUDA bf16 [rows, max_blocks, head_dim]");
    TORCH_CHECK(weight.is_cuda() && weight.dim() == 1 && weight.is_contiguous()
                && weight.scalar_type() == torch::kBFloat16,
                "weight must be contiguous CUDA bf16 [head_dim]");
    TORCH_CHECK(flat_cos.is_cuda() && flat_cos.dim() == 2 && flat_cos.is_contiguous()
                && flat_cos.scalar_type() == torch::kBFloat16,
                "flat_cos must be contiguous CUDA bf16 [batch*tokens, rotary_dim]");
    TORCH_CHECK(block_row.is_cuda() && block_row.dim() == 1 && block_row.is_contiguous()
                && block_row.scalar_type() == torch::kInt,
                "block_row must be contiguous CUDA int32");

    const int rows = (int)out.size(0);
    const int max_blocks = (int)out.size(1);
    const int head_dim = (int)out.size(2);
    const int ratio = (int)table.size(2);
    TORCH_CHECK(ratio > 0 && ratio <= MAX_RATIO,
                "compress ratio ", ratio, " exceeds ", MAX_RATIO);
    TORCH_CHECK((int)weight.size(0) == head_dim, "weight must have head_dim entries");
    TORCH_CHECK((int)flat_keys.size(1) == head_dim, "flat_keys head dim mismatch");
    TORCH_CHECK((int)table.size(0) == rows && (int)table.size(1) == max_blocks,
                "table shape must match out [rows, max_blocks]");
    TORCH_CHECK((int)flat_cos.size(1) == (int)rotary_dim
                && (int)flat_sin.size(1) == (int)rotary_dim,
                "cos/sin last dim must be rotary_dim");
    TORCH_CHECK(rotary_dim > 0 && rotary_dim % 2 == 0 && rotary_dim <= head_dim,
                "rotary_dim must be even and <= head_dim");
    TORCH_CHECK((int)block_row.size(0) == rows * max_blocks
                && (int)block_batch.size(0) == rows * max_blocks,
                "block_row/block_batch must have rows*max_blocks entries");

    const int total = rows * max_blocks;
    if (total == 0) return out;

    const c10::cuda::CUDAGuard guard(out.device());
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    // 线程数取 min(head_dim, BLOCK_THREADS) 并保持 2 的幂，块内归约是折半的。
    int threads = BLOCK_THREADS;
    while (threads > head_dim && threads > 32) threads >>= 1;
    const size_t smem = (size_t)(threads + head_dim) * sizeof(float);
    qsa_block_keys_kernel<<<total, threads, smem, stream>>>(
        reinterpret_cast<const uint16_t*>(flat_keys.data_ptr()),
        table.data_ptr<int>(),
        reinterpret_cast<const uint16_t*>(weight.data_ptr()),
        reinterpret_cast<const uint16_t*>(flat_cos.data_ptr()),
        reinterpret_cast<const uint16_t*>(flat_sin.data_ptr()),
        reinterpret_cast<uint16_t*>(out.data_ptr()),
        block_row.data_ptr<int>(), block_batch.data_ptr<int>(),
        rows, (int)seq, (int)tokens, max_blocks, ratio, head_dim,
        (int)rotary_dim, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
