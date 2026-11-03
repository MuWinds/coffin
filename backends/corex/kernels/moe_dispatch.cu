// @file moe_dispatch.cu
// @brief 把一个 token 的路由结果变成分组内核要的「前缀和 + 按专家收拢的行序」。
//
// Module: coffin/backends/corex/kernels。由 ``backends/corex`` 的 ``moe_prepare``
// 调用；``models/qwen4exp/moe.py`` 的顺序是 ``moe_prepare`` → ``moe_grouped``（两次）。
//
// ## 为什么需要它
//
// 一步 decode 里，分派这段原本是**纯 torch 的一串小算子**，每层来一遍。作业
// ``q4attrib3`` 按调用点量到（48 层 × 每层一次）：
//
//     moe.py:160  topk              48      moe.py:406  bincount   48
//     moe.py:162  div, sum    48 + 48      moe.py:414  cumsum     48
//     moe.py:163  to                48      moe.py:424  chunk      48
//     moe.py:392  nonzero, flatten 48+48    moe.py:425  mul        48
//     moe.py:398  div               48      moe.py:427  mul        48
//     moe.py:402  argsort           48      moe.py:431  to         48
//     合计 672 次内核/步
//
// 本卡实测发射地板 **5.76 µs/次**（作业 ``q4floor1``），672 × 5.76 = **3.87 ms/步**
// 只花在「把 10 个路由结果排一下序」上。一步的目标是 17-20 ms，这笔钱没有理由付。
//
// 这些算术的输入只有 ``slots = tokens * top_k`` 个整数（decode 时 **10** 个），输出
// 几百字节，全部在一个 block 里做得完 —— 分成 14 个内核纯粹是 torch 算子的粒度问题。
//
// ## 用「按桶计数的稳定排序」而不是 ``argsort``
//
// 要做的事本质上是按专家号分桶：给每个槽位算一个 ``(桶, 桶内序号)``，写成
// ``pos = offsets[桶] + 桶内序号``。单 block 内两趟扫描即可，不需要比较排序：
// 一趟计数、一趟前缀和、一趟散写。稳定排序的复杂度是 O(slots log slots)，而这个
// 是 O(slots + E)，且没有 log 因子的写放大。
//
// Constraints:
//   * **单 block**。前缀和是串行的，单 block 保证它不需要跨块扫描就能得到全局正确
//     的桶起点。``slots`` 在 prefill 时能到 20480，仍在单 block 的循环里跑完。
//   * **输出必须确定**：同一个输入两次运行给出同一份 ``pos`` / ``gathered``。
//     ``atomicAdd`` 的到达顺序不保证可复现，所以这里**不用原子操作**，改成
//     「线程 t 负责槽位 t、按 t 递增扫描」的确定性写法。桶内顺序因此只由槽位下标
//     决定，与调度无关 —— 这一条是可复现性的要求，不是保守。
//   * 不属于本 rank 的槽位（``local_table[e] < 0``）丢弃，不占桶、不写进行序。
//   * 不用 CUDA graph、不用 cp.async（``platform/scow/AGENTS.md`` 第 1、5 条）。

#include "corex_ops.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

namespace {

// @brief 单 block：计数 → 前缀和 → 稳定地算出每个槽位在分好组后的位置。
//
// 输出全部在收拢后的**行号空间**里，调用方只需两次整数索引就能完成收拢与加回：
//
//   ``offsets[e]``      第 ``e`` 个本地专家在收拢后的行序里的起点（``[E+1]``）。
//   ``token_of_pos[p]`` 收拢后第 ``p`` 行属于哪个 token。
//   ``slot_of_pos[p]``  收拢后第 ``p`` 行来自原第几个槽位（用来取路由权重）。
//   ``slot_pos[t]``     原本第 ``t`` 个槽位排在第几位；被丢弃的为 -1。
//
// **``p >= 有效行数`` 的尾部一并清成 0**：``moe_grouped`` 是按 ``offsets`` 切的，
// 尾部不参与计算，但 ``token_of_pos`` / ``slot_of_pos`` 会被整块拿去 ``index_select``
// 与 ``index_add_``，留着未初始化显存就等于往随机 token 行上加随机权重。
//
// @param expert_idx   ``[slots]`` 展平的全局专家号。
// @param local_table  ``[num_experts]`` 全局号 → 本地号，不属于本 rank 为 -1。
// @param slot_pos     ``[slots]`` 输出：收拢后的行号，丢弃的为 -1。
// @param token_of_pos ``[slots]`` 输出：收拢后每行的 token 下标。
// @param slot_of_pos  ``[slots]`` 输出：收拢后每行来自哪个槽位。
// @param offsets      ``[local_experts + 1]`` 输出：前缀和。
// @param cursor       ``[local_experts]`` 暂存：各桶已经写到第几号。
// @param slots        槽位数 ``tokens * top_k``。
// @param local_experts 本 rank 的专家数。
// @param num_experts  全局专家数。
// @param top_k         每 token 的专家数。
__global__ void moe_dispatch_kernel(const int* expert_idx, const int* local_table,
                                    int* slot_pos, int* token_of_pos,
                                    int* slot_of_pos, int* offsets, int* cursor,
                                    int slots, int local_experts, int num_experts,
                                    int top_k) {
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    // 阶段 1：清桶与两个位置表。位置表必须整体清（含有效前缀之外的尾部），
    // 理由见文件头。
    for (int e = tid; e <= local_experts; e += nthreads) offsets[e] = 0;
    for (int e = tid; e < local_experts; e += nthreads) cursor[e] = 0;
    for (int p = tid; p < slots; p += nthreads) {
        token_of_pos[p] = 0;
        slot_of_pos[p] = 0;
    }
    __syncthreads();

    // 阶段 2：计数。
    for (int t = tid; t < slots; t += nthreads) {
        const int g = expert_idx[t];
        if (g < 0 || g >= num_experts) {
            slot_pos[t] = -1;
            continue;
        }
        const int e = local_table[g];
        if (e < 0) {
            slot_pos[t] = -1;                     // 不归本 rank：留给 all-to-all
            continue;
        }
        slot_pos[t] = e;                          // 暂时存桶号，阶段 4 换成位置
        atomicAdd(&offsets[e + 1], 1);
    }
    __syncthreads();

    // 阶段 3：前缀和。桶数只有 128，串行做比并行扫描更简单且更快。
    if (tid == 0) {
        int running = 0;
        for (int e = 0; e < local_experts; ++e) {
            const int count = offsets[e + 1];
            offsets[e] = running;
            running += count;
        }
        offsets[local_experts] = running;
    }
    __syncthreads();

    // 阶段 4：把桶号换成真实位置，并填两张位置表。
    //
    // **确定性**：多个线程可能同时命中同一个桶的 ``cursor``，所以这一段按 ``t``
    // 递增**串行**做。``slots`` 在 decode 时是 10、prefill 最多 20480，单线程串行
    // 在本卡上是微秒级，而它换来的是两次运行逐位相同 —— 不确定的输出会让「这处改动
    // 到底动没动数值」无法判定，那比这点时间贵得多。
    if (tid == 0) {
        for (int t = 0; t < slots; ++t) {
            const int e = slot_pos[t];
            if (e < 0) continue;
            const int pos = offsets[e] + cursor[e];
            cursor[e] += 1;
            token_of_pos[pos] = t / top_k;
            slot_of_pos[pos] = t;
            slot_pos[t] = pos;
        }
    }
}

}  // namespace

// @brief 分派：路由结果 → ``(offsets, slot_pos, token_of_pos, slot_of_pos)``。
//
// @param expert_idx   ``[slots]`` 展平的全局专家号，int32 连续。
// @param local_table  ``[num_experts]`` 全局号 → 本地号，int32，不属于本 rank 为 -1。
// @param out_offsets  ``[local_experts + 1]`` int32 输出。
// @param out_slot_pos ``[slots]`` int32 输出，收拢后的行号，丢弃的为 -1。
// @param out_tokens   ``[slots]`` int32 输出，收拢后每行的 token 下标。
// @param out_slots    ``[slots]`` int32 输出，收拢后每行来自哪个槽位。
// @param scratch      ``[local_experts]`` int32 暂存（调用方持有、可复用）。
// @param slots, local_experts, num_experts, top_k 几何。
// @return 无。错误通过 host 侧 ``C10_CUDA_KERNEL_LAUNCH_CHECK`` 暴露。
// @throws c10::Error 布局、设备、dtype 或形状不自洽。
void moe_dispatch(torch::Tensor expert_idx, torch::Tensor local_table,
                  torch::Tensor out_offsets, torch::Tensor out_slot_pos,
                  torch::Tensor out_tokens, torch::Tensor out_slots,
                  torch::Tensor scratch, int64_t slots, int64_t local_experts,
                  int64_t num_experts, int64_t top_k) {
    TORCH_CHECK(expert_idx.is_cuda() && expert_idx.dim() == 1
                && expert_idx.scalar_type() == torch::kInt
                && expert_idx.is_contiguous(),
                "expert_idx must be contiguous CUDA int32 [slots]");
    TORCH_CHECK(local_table.is_cuda() && local_table.dim() == 1
                && local_table.scalar_type() == torch::kInt
                && local_table.is_contiguous(),
                "local_table must be contiguous CUDA int32 [num_experts]");
    TORCH_CHECK(out_offsets.is_cuda() && out_offsets.dim() == 1
                && out_offsets.scalar_type() == torch::kInt
                && out_offsets.is_contiguous()
                && out_offsets.numel() == local_experts + 1,
                "out_offsets must be contiguous CUDA int32 [local_experts + 1]");
    TORCH_CHECK(out_slot_pos.is_cuda() && out_slot_pos.dim() == 1
                && out_slot_pos.scalar_type() == torch::kInt
                && out_slot_pos.is_contiguous() && out_slot_pos.numel() == slots,
                "out_slot_pos must be contiguous CUDA int32 [slots]");
    TORCH_CHECK(out_tokens.is_cuda() && out_tokens.dim() == 1
                && out_tokens.scalar_type() == torch::kInt
                && out_tokens.is_contiguous() && out_tokens.numel() == slots,
                "out_tokens must be contiguous CUDA int32 [slots]");
    TORCH_CHECK(out_slots.is_cuda() && out_slots.dim() == 1
                && out_slots.scalar_type() == torch::kInt
                && out_slots.is_contiguous() && out_slots.numel() == slots,
                "out_slots must be contiguous CUDA int32 [slots]");
    TORCH_CHECK(scratch.is_cuda() && scratch.dim() == 1
                && scratch.scalar_type() == torch::kInt && scratch.is_contiguous()
                && scratch.numel() >= local_experts,
                "scratch must be contiguous CUDA int32 [>= local_experts]");
    TORCH_CHECK(local_experts > 0 && num_experts > 0 && top_k > 0 && slots >= 0,
                "sizes must be positive");
    if (slots == 0) return;

    const c10::cuda::CUDAGuard guard(expert_idx.device());
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    // 一个 block：前缀和是串行的，跨块要引入扫描算法，而桶数只有 128。
    // 线程数取 128 的整数倍并封顶 512：阶段 1/2 是纯整数循环，够用即可。
    const int threads = 128;
    moe_dispatch_kernel<<<1, threads, 0, stream>>>(
        expert_idx.data_ptr<int>(), local_table.data_ptr<int>(),
        out_slot_pos.data_ptr<int>(), out_tokens.data_ptr<int>(),
        out_slots.data_ptr<int>(), out_offsets.data_ptr<int>(),
        scratch.data_ptr<int>(),
        (int)slots, (int)local_experts, (int)num_experts, (int)top_k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
