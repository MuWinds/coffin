// @file moe_dispatch_device.inc
// @brief MoE 分派的设备代码：路由结果 → 前缀和 + 行序，单 block 一次算完。
//
// Module: coffin/backends/corex/kernels。被两条前端各包含一次：``moe_dispatch.cu``
// （torch 扩展）与 ``abi/moe_dispatch_abi.cu``（``extern "C"`` 出口）。同一份设备代码。
//
// 为什么它是一个内核：模型侧原本是 ``nonzero / flatten / div / argsort / bincount /
// cumsum / chunk / mul`` 等 14 个内核，合计 672 次/步，按本卡 5.76 us 的发射地板是
// 3.87 ms/步，而输入只有 ``slots`` 个整数（decode 时 10 个）。
//
// Constraints:
//   * 单 block：阶段的每一步之间都要 ``__syncthreads``，跨 block 做不到。
//   * 位置表的**尾部必须清 0**（含有效前缀之外）：两张表会被整块拿去索引。
//   * 段内排列由 ``atomicAdd`` 决定，**不确定**。合并侧按槽位号定序求和，不读排列
//     本身；若哪天改回 ``index_add_``，这里立刻变成不可复现。
//   * 桶号临时留在 ``slot_pos[t]`` 上，写者与读者是同一个线程，所以这张表不需要同步，
//     也省掉一张 ``[slots]`` 的中间表。

#ifndef COFFIN_MOE_DISPATCH_DEVICE_INC
#define COFFIN_MOE_DISPATCH_DEVICE_INC

#include <cstdint>

template <typename IT>
__global__ void moe_dispatch_kernel(const IT* expert_idx, const int* local_table,
                                    int* slot_pos, int* token_of_pos,
                                    int* slot_of_pos, int* offsets, int* cursor,
                                    int slots, int local_experts, int num_experts,
                                    int top_k) {
    // 计数器、前缀和、游标与块和全部留在 LDS 中，省掉阶段 1/2/4 的全局内存清零与全局
    // 原子加，并把阶段 3 的串行 256 步扫描换成两级并行前缀和（每块 16 个桶）。
    extern __shared__ int lds[];
    int* counts = lds;                                    // [local_experts]
    int* starts = counts + local_experts;                 // [local_experts + 1]
    int* lds_cursor = starts + local_experts + 1;         // [local_experts]
    int* chunk_off = lds_cursor + local_experts;          // [(local_experts + 15) / 16]

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    (void)cursor;

    // 阶段 1：清 LDS 计数/游标与全局位置表。位置表必须整体清（含有效前缀之外的尾部），
    // 理由见文件头。
    for (int e = tid; e < local_experts; e += nthreads) {
        counts[e] = 0;
        lds_cursor[e] = 0;
    }
    for (int p = tid; p < slots; p += nthreads) {
        token_of_pos[p] = 0;
        slot_of_pos[p] = 0;
    }
    __syncthreads();

    // 阶段 2：在 LDS 上并行计数。桶号顺手留在 ``slot_pos[t]`` 上，阶段 4 再换成真实位置。
    for (int t = tid; t < slots; t += nthreads) {
        const int g = static_cast<int>(expert_idx[t]);
        int e = -1;
        if (g >= 0 && g < num_experts) {
            e = local_table[g];
        }
        if (e < 0 || e >= local_experts) {
            slot_pos[t] = -1;                  // 不归本 rank：留给 all-to-all
            continue;
        }
        slot_pos[t] = e;                       // 暂时存桶号，阶段 4 换成位置
        atomicAdd(&counts[e], 1);
    }
    __syncthreads();

    // 阶段 3：两级并行前缀和（每段 16 桶）。256 桶只需 16 线程并行各扫 16 项 +
    // tid 0 扫 16 个段和 + 16 线程并行回写，深度从 256 降到 48 且无全局往返。
    constexpr int CHUNK = 16;
    const int nchunks = (local_experts + CHUNK - 1) / CHUNK;
    for (int c = tid; c < nchunks; c += nthreads) {
        const int b0 = c * CHUNK;
        const int b1 = b0 + CHUNK < local_experts ? b0 + CHUNK : local_experts;
        int sum = 0;
        for (int e = b0; e < b1; ++e) sum += counts[e];
        chunk_off[c] = sum;
    }
    __syncthreads();
    if (tid == 0) {
        int running = 0;
        for (int c = 0; c < nchunks; ++c) {
            const int v = chunk_off[c];
            chunk_off[c] = running;
            running += v;
        }
        starts[local_experts] = running;
        offsets[local_experts] = running;
    }
    __syncthreads();
    for (int c = tid; c < nchunks; c += nthreads) {
        const int b0 = c * CHUNK;
        const int b1 = b0 + CHUNK < local_experts ? b0 + CHUNK : local_experts;
        int running = chunk_off[c];
        for (int e = b0; e < b1; ++e) {
            starts[e] = running;
            offsets[e] = running;
            running += counts[e];
        }
    }
    __syncthreads();

    // 阶段 4：利用 LDS 游标并行分配位置。
    //
    // 每一行的位置由原子递增决定，所以段内排列不确定；合并侧按槽位号定序求和，
    // 不读排列本身。
    for (int t = tid; t < slots; t += nthreads) {
        const int e = slot_pos[t];
        if (e < 0) continue;
        const int pos = starts[e] + atomicAdd(&lds_cursor[e], 1);
        token_of_pos[pos] = t / top_k;
        slot_of_pos[pos] = t;
        slot_pos[t] = pos;
    }
}


#endif  // COFFIN_MOE_DISPATCH_DEVICE_INC
