// @file moe_router_device.inc
// @brief MoE 路由融合的设备代码：``softmax -> top-k -> 归一化``，一个 32-lane 组一个 token。
//
// Module: coffin/backends/corex/kernels。被两条前端各包含一次：``moe_grouped.cu``
// （torch 扩展，``models/qwen35/moe.py`` 的 ``TopKRouter`` 走那条）与
// ``abi/moe_router_abi.cu``（``extern "C"`` 出口）。同一份设备代码，两条前端逐位相同。
//
// 为什么要它：``softmax -> topk -> div -> sum -> to`` 五条 torch 链在 decode 一步里是
// 48 层 × 5 个内核，其中 ``aten::topk`` 对 ``[1, 512]`` 取 top-10 在本卡上走 bitonic
// 排序网络、单次 38 us。挑最大的 10 个不需要排序：10 轮「找剩下的最大者」全在寄存器里。
//
// Constraints:
//   * 归约只用 32 lane 的 ``__shfl_xor_sync``：本部件硬件 warp 是 64 通道宽（见
//     ``attn.cu`` 文件头），偏移量都小于 32，两个半 warp 各算各的互不干扰；换成
//     ``__shfl_down_sync`` 的蝶形只加一半通道就返回一个看着合理的错标量。
//   * 越界的组 **clamp 而不是 return**：shuffle 的 mask 覆盖整个 64 lane 硬件 warp，
//     一个 32-lane 组提前退出会让同 warp 里另一组的 shuffle 跨不过去；只对写回加守卫。
//   * 值和下标一起过蝶形。只归约值的话，两个等概率专家在不同 lane 上会给出不同的
//     赢家，写出的下标就依赖调度——那是「同一输入两次结果不同」的来源。
//   * 每轮只让 ``lane == slot`` 的 lane 捕获结果，绝不往 ``sel[slot]`` 这种动态下标
//     的寄存器数组里写（会掉进 local memory）。

#ifndef COFFIN_MOE_ROUTER_DEVICE_INC
#define COFFIN_MOE_ROUTER_DEVICE_INC

#include "corex_scalar.h"

#include <climits>
#include <cstdint>

//: 一个 32-lane 组一个 token。本部件的硬件 warp 是 64 通道宽（见 ``attn.cu`` 文件头），
//: 所以归约只能用 32 lane 的 xor 蝶形：偏移量都小于 32，两个半 warp 各算各的，互不干扰。
constexpr int ROUTER_LANES = 32;

//: 一个 block 里的组数。decode 只用得到 1 组，prefill ``seq=512`` 时正好 8 组一个 block。
constexpr int ROUTER_GROUPS = 8;

//: 每个 lane 最多承担几个专家。``n = 512`` 时是 16；上界 1024 走 32 那一档。
constexpr int ROUTER_ELEMS = 32;

//: 补位与「已选中」用的负值。不用 ``-INFINITY``：`expf` 的入参要保持有限，
//: 而 ``NEG - m`` 有限，``__expf`` 给 0。
constexpr float ROUTER_NEG = -3.0e38f;

// @brief 一个 token 的 ``softmax(fp32) -> top-k -> 归一化``，组内 32 lane 协作。
//
// @tparam ELEMS 每个 lane 承担的专家数，编译期定死（寄存器数组不能用动态下标）。
// @tparam XT    输入/权重的存储类型：``uint16_t``（bf16）或 ``float``。
//
// @param logits     ``[tokens, n]`` 连续 logits。
// @param out_idx    ``[tokens, top_k]`` int64 目的地，按概率降序。
// @param out_w      ``[tokens, top_k]`` 目的地，dtype 同 ``logits``。
// @param tokens     行数。
// @param n          专家数（512）。
// @param top_k      每 token 取的专家数，不超过 32：每个 lane 只保存**自己那一槽**的结果。
// @param normalize  是否把 top-k 概率再归一化（``norm_topk_prob``）。
template <int ELEMS, typename XT>
__global__ void moe_router_t(const XT* __restrict__ logits, int64_t* __restrict__ out_idx,
                             XT* __restrict__ out_w, int tokens, int n, int top_k,
                             int normalize) {
    const int lane = threadIdx.x % ROUTER_LANES;
    const int group = threadIdx.x / ROUTER_LANES;
    const int real = (int)(blockIdx.x * ROUTER_GROUPS + group);
    // **clamp 而不是 return**：``__shfl_xor_sync`` 的 mask 覆盖整个 64 lane 硬件 warp，
    // 一个 32-lane 组提前退出会让同 warp 里另一组的 shuffle 跨不过去。这里把越界的组
    // 重复算最后一个 token（读的是合法内存），只对写回加守卫。
    const int token = real < tokens ? real : tokens - 1;
    const XT* row = logits + (int64_t)token * n;

    unsigned used = 0u;
    float v[ELEMS];
#pragma unroll
    for (int i = 0; i < ELEMS; ++i) {
        const int col = lane + i * ROUTER_LANES;
        if (col < n) {
            v[i] = CxScalar<XT>::to_float(row[col]);
        } else {
            v[i] = ROUTER_NEG;
            used |= 1u << i;                     // 补位不参与选择
        }
    }

    // 全量 softmax 保留在两条路径上。曾经试过「归一化时只对 top-k 做 softmax」
    // （数学等价：分母被约掉，且 softmax 单调所以 TopK 不变），隔离测量里 router
    // 42.2 -> 35.1 us，但端到端四档冒烟慢 2.6%。最可能的原因是两条路径并存把
    // ``v[ELEMS]`` 那个寄存器数组顶到溢出，多出来的 local memory 流量吃掉收益。
    float m = ROUTER_NEG;
#pragma unroll
    for (int i = 0; i < ELEMS; ++i) m = fmaxf(m, v[i]);
#pragma unroll
    for (int s = ROUTER_LANES / 2; s > 0; s >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, s));
    }

    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < ELEMS; ++i) {
        v[i] = __expf(v[i] - m);
        sum += v[i];
    }
#pragma unroll
    for (int s = ROUTER_LANES / 2; s > 0; s >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, s);
    }

    // 每轮只让「自己那一槽」的 lane 捕获结果：``lane == slot`` 时写一个标量，
    // 而不是往 ``sel[slot]`` 这种动态下标的寄存器数组里写——那会掉进 local memory。
    float mine_v = 0.f;
    int mine_i = 0;
    float top_sum = 0.f;
    for (int slot = 0; slot < top_k; ++slot) {
        float best = ROUTER_NEG;
        int bi = INT_MAX;
#pragma unroll
        for (int i = 0; i < ELEMS; ++i) {
            if ((used & (1u << i)) != 0u) continue;
            if (v[i] > best) {
                best = v[i];
                bi = lane + i * ROUTER_LANES;
            }
        }
        // 值和下标一起过蝶形。只归约值的话，两个概率相等的专家在不同 lane 上会给出
        // 不同的赢家，于是写出去的下标依赖调度 —— 那也是「同一输入两次结果不同」的来源。
#pragma unroll
        for (int s = ROUTER_LANES / 2; s > 0; s >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, best, s);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, s);
            if (ov > best || (ov == best && oi < bi)) {
                best = ov;
                bi = oi;
            }
        }
        if (bi != INT_MAX && lane == (bi % ROUTER_LANES)) used |= 1u << (bi / ROUTER_LANES);
        if (lane == slot) {
            mine_v = best;
            mine_i = bi;
        }
        top_sum += best;
    }

    if (real < tokens && lane < top_k) {
        const int64_t at = (int64_t)real * top_k + lane;
        out_idx[at] = (int64_t)mine_i;
        // 归一化路径用 top-k 自己的和；不归一化时权重就是概率 ``v / Σv``。
        const float w = normalize ? (mine_v / top_sum) : (mine_v / sum);
        out_w[at] = CxScalar<XT>::from_float(w);
    }
}


#endif  // COFFIN_MOE_ROUTER_DEVICE_INC
