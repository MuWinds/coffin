// @file attn_sparse_device.inc
// @brief QSA 稀疏注意力的设备代码：warp 内归约 + 一条只读选中位置的内核 + 启动器。
//
// Module: coffin/backends/corex/kernels。被两条前端各包含一次：``attn_sparse.cu``
// （torch 扩展的 pybind 入口）与 ``abi/attn_sparse_abi.cu``（``extern "C"`` 出口）。
// 同一份设备代码，所以线程数、共享内存和网格几何不可能各自漂移。
//
// Constraints:
//   * 常量与内核必须**先于**任何使用它们的宿主代码可见；放进 ``namespace {}`` 时
//     包含位置就是可见性边界（见 ``docs/COREX-NO-TORCH.md`` §6.3 的教训）。
//   * ``hd`` 必须是 32 的倍数，且 ``hd / 32 <= ATT_SPARSE_LANES``。

#ifndef COFFIN_ATTN_SPARSE_DEVICE_INC
#define COFFIN_ATTN_SPARSE_DEVICE_INC

#include "corex_scalar.h"

#include <cstdint>
#include <cuda_runtime.h>

namespace {

//: 每个 warp 负责一个选中位置，warp 内的 32 条 lane 分摊 ``hd`` 个通道。
constexpr int WARP = 32;

//: 一个 block 里并发的选中位置数（即 warp 数）。8 个 warp 让 256 线程的块吃满，
//: 而 ``max_sel`` 最多 2048，所以每个 warp 串 ``2048 / 8 = 256`` 个位置 —— 但那是
//: **没有归约开销**的串行，与第一版的 4096 次同步完全不同。
constexpr int WARPS = 8;

//: 每条 lane 最多负责几个通道。``hd = 256``、``32 lane`` 时是 8。
constexpr int ATT_SPARSE_LANES = 8;

// @brief warp 内把 ``v`` 求和并广播。
//
// @param v 本 lane 的值。
// @return 全 warp 求和。
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int s = WARP / 2; s > 0; s >>= 1) v += __shfl_xor_sync(0xffffffffu, v, s);
    return v;
}

// @brief warp 内把 ``v`` 求最大值并广播。
//
// @param v 本 lane 的值。
// @return 全 warp 最大值。
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int s = WARP / 2; s > 0; s >>= 1) {
        const float o = __shfl_xor_sync(0xffffffffu, v, s);
        if (o > v) v = o;
    }
    return v;
}

// @brief 一个（行, 查询头）的稀疏精确 softmax 注意力。
//
// 每个 warp 独立处理一部分选中位置并跑在线 softmax（rescaling 的在线形式），最后
// 把各 warp 的 ``(m, l, acc)`` 按「合并两个 softmax」的公式归并。这样循环里没有
// 任何块级同步，且并行度从「一个头一个 block 的单循环」变成「头 × warp」。
//
// @param q       ``[rows * nh * hd]`` bf16。
// @param kc,vc   稠密 KV 缓存 bf16，**头主序** ``[nkv, max_seq, hd]``（未 repeat）。
// @param sel_idx 设备侧 ``[rows * max_sel]`` int32 选中位置，行内升序，无效为 -1。
// @param sel_len 设备侧 ``[rows]`` int32 每行有效位置数。
// @param ctx_row 设备侧 ``[rows]`` int32 每行的因果上界。
// @param out     ``[rows * nh * hd]`` bf16。
// @param nh, nkv, hd 几何。
// @param max_seq 每头行数（头主序行间距）。
// @param max_sel 每行槽位上限（``sel_idx`` 行间距）。
// @param scale   打分缩放，通常 ``hd ** -0.5``。
__global__ void attn_sparse_kernel(const uint16_t* __restrict__ q,
                                   const uint16_t* __restrict__ kc,
                                   const uint16_t* __restrict__ vc,
                                   const int* __restrict__ sel_idx,
                                   const int* __restrict__ sel_len,
                                   const int* __restrict__ ctx_row,
                                   uint16_t* __restrict__ out,
                                   int nh, int nkv, int hd, int max_seq,
                                   int max_sel, float scale) {
    const int h = blockIdx.x;
    const int row = blockIdx.y;
    const int rep = nh / nkv;
    const int g = h / rep;
    const int tid = threadIdx.x;
    const int warp = tid / WARP;           // 本线程属于哪个位置组
    const int lane = tid % WARP;           // 本线程在 warp 内的 lane

    const size_t out_base = ((size_t)row * nh + h) * (size_t)hd;
    const int n = sel_len[row];
    const int ctx = ctx_row[row];

    if (n <= 0 || hd <= 0) {
        for (int d = tid; d < hd; d += blockDim.x) out[out_base + d] = 0;
        return;
    }

    // 本 lane 负责的通道是 ``lane, lane + 32, ...``；跨 warp 的同一 lane 负责同一批
    // 通道，所以最后按 lane 归并各 warp 的部分和即可。
    const uint16_t* qrow = q + out_base;
    float qreg[ATT_SPARSE_LANES];
#pragma unroll
    for (int c = 0; c < ATT_SPARSE_LANES; ++c) {
        const int d = lane + c * WARP;
        qreg[c] = (d < hd) ? CxScalar<uint16_t>::to_float(qrow[d]) : 0.f;
    }

    const int* idx = sel_idx + (size_t)row * max_sel;
    const uint16_t* kbase = kc + (size_t)g * (size_t)max_seq * (size_t)hd;
    const uint16_t* vbase = vc + (size_t)g * (size_t)max_seq * (size_t)hd;

    // 在线 softmax 的三元组，每个 warp 一份。
    float m = -1e30f;
    float l = 0.f;
    float acc[ATT_SPARSE_LANES];
#pragma unroll
    for (int c = 0; c < ATT_SPARSE_LANES; ++c) acc[c] = 0.f;

    for (int j = warp; j < n; j += WARPS) {
        const int p = idx[j];
        if (p < 0 || p >= ctx) continue;
        const uint16_t* kp = kbase + (size_t)p * (size_t)hd;
        float dot = 0.f;
#pragma unroll
        for (int c = 0; c < ATT_SPARSE_LANES; ++c) {
            const int d = lane + c * WARP;
            if (d < hd) dot += qreg[c] * CxScalar<uint16_t>::to_float(kp[d]);
        }
        // 全 warp 的点积：lane 之间还没有合并过。
        dot = warp_sum(dot) * scale;

        // 在线更新：只有新分数超过当前最大值时才 rescale，避免每步都乘。
        if (dot > m) {
            const float corr = __expf(m - dot);
#pragma unroll
            for (int c = 0; c < ATT_SPARSE_LANES; ++c) acc[c] *= corr;
            l *= corr;
            m = dot;
        }
        const float w = __expf(dot - m);
        l += w;
        const uint16_t* vp = vbase + (size_t)p * (size_t)hd;
#pragma unroll
        for (int c = 0; c < ATT_SPARSE_LANES; ++c) {
            const int d = lane + c * WARP;
            if (d < hd) acc[c] += w * CxScalar<uint16_t>::to_float(vp[d]);
        }
    }

    // 每个 warp 的 (m, l, acc) 写到共享内存，再按「合并两个 softmax」归并。
    //
    // 共享内存布局：l/warp 与 m/warp 各一条，acc 是 ``[WARPS, hd]`` 的 fp32
    // —— ``8 * 256 * 4 = 8 KB``，在这张卡的 64 KB 预算内。
    extern __shared__ float sh[];
    float* sh_m = sh;                      // [WARPS]
    float* sh_l = sh_m + WARPS;            // [WARPS]
    float* sh_acc = sh_l + WARPS;          // [WARPS * hd]

    if (lane == 0) {
        sh_m[warp] = m;
        sh_l[warp] = l;
    }
    // acc 按通道写到本 warp 的槽位。同一 lane 跨 warp 负责同一批通道，所以写进去
    // 之后按通道扫描即可。
#pragma unroll
    for (int c = 0; c < ATT_SPARSE_LANES; ++c) {
        const int d = lane + c * WARP;
        if (d < hd) sh_acc[warp * hd + d] = acc[c];
    }
    __syncthreads();

    for (int d = tid; d < hd; d += blockDim.x) {
        // 全局最大。
        float gm = sh_m[0];
        for (int w = 1; w < WARPS; ++w) {
            if (sh_m[w] > gm) gm = sh_m[w];
        }
        // 按 gm 重新缩放并求和。
        float gl = 0.f;
        float ga = 0.f;
        for (int w = 0; w < WARPS; ++w) {
            const float s = __expf(sh_m[w] - gm);
            gl += sh_l[w] * s;
            ga += sh_acc[w * hd + d] * s;
        }
        out[out_base + d] = (gl > 0.f)
            ? CxScalar<uint16_t>::from_float(ga / gl)
            : CxScalar<uint16_t>::from_float(0.f);
    }
}

}  // namespace

// @brief 启动稀疏注意力。
//
// @param q,out ``[rows, nh, hd]`` bf16。
// @param kc,vc ``[nkv, max_seq, hd]`` bf16 头主序。
// @param sel_idx ``[rows, max_sel]`` int32；``sel_len``、``ctx_row`` ``[rows]`` int32。
// @param rows, nh, nkv, hd, max_seq, max_sel 几何。
// @param scale 打分缩放。
// @param stream 目标流；空指针表示默认流。
inline void launch_attn_sparse(const uint16_t* q, const uint16_t* kc, const uint16_t* vc,
                               const int* sel_idx, const int* sel_len, const int* ctx_row,
                               uint16_t* out, int rows, int nh, int nkv, int hd, int max_seq,
                               int max_sel, float scale, cudaStream_t stream) {
    const int threads = WARPS * WARP;
    const size_t smem = (size_t)(2 * WARPS + WARPS * hd) * sizeof(float);
    const dim3 grid((unsigned)nh, (unsigned)rows);
    attn_sparse_kernel<<<grid, threads, smem, stream>>>(
        q, kc, vc, sel_idx, sel_len, ctx_row, out, nh, nkv, hd, max_seq, max_sel, scale);
}

#endif  // COFFIN_ATTN_SPARSE_DEVICE_INC
