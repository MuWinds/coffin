// @file qsa_block_keys_device.inc
// @brief QSA 索引器块 key 的设备代码：gather → 池化 → RMSNorm → partial RoPE。
//
// Module: coffin/backends/corex/kernels。被两条前端各包含一次：``qsa_block_keys.cu``
// （torch 扩展的 pybind 入口）与 ``abi/qsa_block_keys_abi.cu``（``extern "C"`` 出口）。
// 同一份内核，所以线程数与共享内存不可能各自漂移。
//
// 数值口径见 ``qsa_block_keys.cu`` 的文件头：零中心增益 ``1 + w``、均值在 fp32 里
// 算再舍入到 bf16、半切分形式的 partial RoPE。
//
// Constraints:
//   * ``ratio <= MAX_RATIO``；超了由入口拒绝。
//   * ``rotary_dim`` 偶数且 ``<= head_dim``。
//   * 线程数取 ``min(head_dim, BLOCK_THREADS)`` 并保持 2 的幂：块内归约是折半的。

#ifndef COFFIN_QSA_BLOCK_KEYS_DEVICE_INC
#define COFFIN_QSA_BLOCK_KEYS_DEVICE_INC

#include "corex_scalar.h"

#include <cstdint>
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

// @brief 启动块 key 的融合内核。
//
// @param flat_keys ``[batch * tokens, head_dim]`` bf16。
// @param table ``[rows, max_blocks, ratio]`` int32。
// @param weight ``[head_dim]`` bf16。
// @param flat_cos,flat_sin ``[batch * tokens, rotary_dim]`` bf16。
// @param out ``[rows, max_blocks, head_dim]`` bf16。
// @param block_row,block_batch ``[rows * max_blocks]`` int32。
// @param rows, seq, tokens, max_blocks, ratio, head_dim, rotary_dim, eps 几何与缩放。
// @param stream 目标流；空指针表示默认流。
inline void launch_qsa_block_keys(const uint16_t* flat_keys, const int* table,
                                  const uint16_t* weight, const uint16_t* flat_cos,
                                  const uint16_t* flat_sin, uint16_t* out, const int* block_row,
                                  const int* block_batch, int rows, int seq, int tokens,
                                  int max_blocks, int ratio, int head_dim, int rotary_dim,
                                  float eps, cudaStream_t stream) {
    const int total = rows * max_blocks;
    if (total == 0) return;
    // 线程数取 min(head_dim, BLOCK_THREADS) 并保持 2 的幂，块内归约是折半的。
    int threads = BLOCK_THREADS;
    while (threads > head_dim && threads > 32) threads >>= 1;
    const size_t smem = (size_t)(threads + head_dim) * sizeof(float);
    qsa_block_keys_kernel<<<total, threads, smem, stream>>>(
        flat_keys, table, weight, flat_cos, flat_sin, out, block_row, block_batch,
        rows, seq, tokens, max_blocks, ratio, head_dim, rotary_dim, eps);
}

#endif  // COFFIN_QSA_BLOCK_KEYS_DEVICE_INC
