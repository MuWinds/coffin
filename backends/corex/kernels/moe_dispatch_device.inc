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
    // 只有前缀和那段需要 LDS：计数器留在全局（``offsets[e+1]``），游标留在调用方
    // 常驻的 ``cursor`` 上。两张表加起来 2 KB 量级，一次搬进共享内存是为了让
    // 「扫一遍 512 个桶」从全局内存变成 LDS。
    extern __shared__ int lds[];
    int* counts = lds;                    // [local_experts]，从 offset[e+1] 抄来
    int* starts = counts + local_experts;    // [local_experts + 1]，前缀和

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    // 阶段 1：清计数与两个位置表。位置表必须整体清（含有效前缀之外的尾部），
    // 理由见文件头。
    for (int e = tid; e <= local_experts; e += nthreads) offsets[e] = 0;
    for (int e = tid; e < local_experts; e += nthreads) cursor[e] = 0;
    for (int p = tid; p < slots; p += nthreads) {
        token_of_pos[p] = 0;
        slot_of_pos[p] = 0;
    }
    __syncthreads();

    // 阶段 2：计数。桶号顺手留在 ``slot_pos[t]`` 上，阶段 4 再换成真实位置——
    // 这样省掉一张 ``[slots]`` 的中间表，而 ``slot_pos[t]`` 的写者与读者是同一个
    // 线程，所以这一张表不需要任何同步。
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
        atomicAdd(&offsets[e + 1], 1);
    }
    __syncthreads();

    // 阶段 3：前缀和。把计数器并行抄进 LDS，再让一个线程串行扫——桶数只有 128/512，
    // 扫完是微秒级；旧的写法是 tid 0 直接对全局内存读改写，那 512 次依赖全局延迟
    // 的迭代本身就是几十微秒
    for (int e = tid; e < local_experts; e += nthreads) counts[e] = offsets[e + 1];
    __syncthreads();
    if (tid == 0) {
        int running = 0;
        for (int e = 0; e < local_experts; ++e) {
            starts[e] = running;
            running += counts[e];
        }
        starts[local_experts] = running;
    }
    __syncthreads();
    for (int e = tid; e <= local_experts; e += nthreads) offsets[e] = starts[e];

    // 阶段 4：并行分配位置。
    //
    // 每一行的位置由原子递增决定，所以段内排列不确定
    // 允许这样做的依据是合并侧按槽位号定序求和，不读排列本身；而如果哪天有人把合并
    // 改回 ``index_add_``，这里就会立刻变成不可复现
    for (int t = tid; t < slots; t += nthreads) {
        const int e = slot_pos[t];
        if (e < 0) continue;
        const int pos = starts[e] + atomicAdd(&cursor[e], 1);
        token_of_pos[pos] = t / top_k;
        slot_of_pos[pos] = t;
        slot_pos[t] = pos;
    }
}


#endif  // COFFIN_MOE_DISPATCH_DEVICE_INC
