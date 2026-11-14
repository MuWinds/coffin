// @file moe_dispatch.cu
// @brief 把一个 token 的路由结果变成分组内核要的「前缀和 + 按专家收拢的行序」。
//
// Module: coffin/backends/corex/kernels。由 ``backends/corex`` 的 ``moe_dispatch``
// 调用；``models/qwen4exp/moe.py`` 的顺序是 ``moe_dispatch`` → ``moe_grouped``（两次）。
//
// ## 为什么需要它
//
// 一步 decode 里，分派这段原本是纯 torch 的一串小算子，每层来一遍。
//
// 这些算术的输入只有 ``slots = tokens * top_k`` 个整数（decode 时10个），输出
// 几百字节，全部在一个 block 里做得完，分成 14 个内核纯粹是 torch 算子的粒度问题。
//
// ## 打包段必须并行
//
// 旧版把「桶号 → 收拢后行号」的分配写成 ``tid == 0`` 的单线程循环：prefill 512 token
// 时 ``slots = 5120``，每次迭代 5 个相互依赖的全局访问（读 ``slot_pos[t]`` →
// 读改写 ``cursor[e]`` → 写两张位置表 → 写 ``slot_pos[t]``），5120 × 约 400 周期
// ≈ 2.0 ms/次，实测2.07 ms/次、48 层 99 ms，占 prefill 1494 ms 的 6.6%，是
// MoE 内核降下来之后的第二大单项。
//
// 改成 ``atomicAdd(&cursor[e], 1)`` 之后段内顺序变了也不影响结果：每个槽位的
// 计算只依赖它自己那一行激活，而合并侧靠 ``slot_pos`` 把行按**槽位序** gather 回来
// 再做一次定序求和（``moe.py`` 的 ``merged.sum(dim=1)``，取代了原来的 ``index_add_``），
// 所以每个 token 的结果与段内排列无关。**这条不是推测而是可测的**：
// ``probe_q4_ep4_gen.py`` 的 ``self_repeat``（同输入两遍 logits 差）就是它的判据，
// 回归时会立刻变成非零。
//
// ## 为什么仍然是单 block（没有按 slots 拆多块）
//
// 「计数 → 前缀和 → 分配」需要一个全网格屏障：前缀和必须在所有计数写完之后才能
// 算。多 block 只能靠 cooperative launch 或手写全局屏障，而后者一旦某个 block 没能
// 同时驻留就是静默死等
//
// 换成全局原子 + 共享内存前缀和之后，内核里只剩四次 ``__syncthreads()``
// （全在 block 内、每次只隔一个 O(slots) 的并行段），比拆多块的目标更划算，而且
// 没有任何挂卡风险。
//
// 原子操作留在全局内存（``offsets[e+1]`` 与 ``scratch[e]``，后者本来就是为游标
// 准备的常驻缓冲）：本仓所有内核里跑通的就只有这一种原子，不为了省那点延迟去赌
// 共享内存原子的实现。
//
// ## 用「按桶计数的稳定排序」而不是 ``argsort``
//
// 要做的事本质上是按专家号分桶：给每个槽位算一个 ``(桶, 桶内序号)``，写成
// ``pos = offsets[桶] + 桶内序号``。两趟扫描即可，不需要比较排序：一趟计数、一趟
// 前缀和、一趟散写。稳定排序的复杂度是 O(slots log slots)，而这个
// 是 O(slots + E)，且没有 log 因子的写放大。
//
// Constraints:
//   * **单 block**。前缀和需要「所有计数已完成」，跨块要引入全网格屏障，见上。
//   * 动态共享内存是 ``2 * local_experts + 1`` 个 int（128 专家 1.0 KB、512 专家
//     4.0 KB）。本卡 64 KB LDS、且超过 48 KB 的动态分配要额外声明属性，所以 host
//     侧对总量有断言，超了就在那里报错而不是让内核越界。
//   * **``offsets`` 逐位确定，三张位置表不再确定**：每个槽位写入的位置由
//     ``atomicAdd`` 的到达顺序决定，所以同一输入两次运行里 ``token_of_pos`` /
//     ``slot_of_pos`` / ``slot_pos`` 的**段内排列**会不同。这放弃的是「中间量可复现」，
//     换来的是打包段从 2.07 ms 掉到几十微秒；模型输出仍然逐位确定，因为合并侧
//     按槽位号定序求和，不依赖排列。这一条由 ``self_repeat`` 守着。
//   * 不属于本 rank 的槽位（``local_table[e] < 0``）丢弃，不占桶、不写进行序。
//   * 不用 CUDA graph、不用 cp.async

#include "corex_bindings.h"
#include "corex_ops.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

namespace {

// @brief 单 block：计数 → 共享内存前缀和 → 并行分配每个槽位的位置。
//
// 输出全部在收拢后的**行号空间**里，调用方只需两次整数索引就能完成收拢与加回：
//
//   ``offsets[e]``      第 ``e`` 个本地专家在收拢后的行序里的起点（``[E+1]``）。
//   ``token_of_pos[p]`` 收拢后第 ``p`` 行属于哪个 token。
//   ``slot_of_pos[p]``  收拢后第 ``p`` 行来自原第几个槽位（用来取路由权重）。
//   ``slot_pos[t]``     原本第 ``t`` 个槽位排在第几位；被丢弃的为 -1。
//
// **``p >= 有效行数`` 的尾部一并清成 0**：``moe_grouped`` 是按 ``offsets`` 切的，
// 尾部不参与计算，但 ``token_of_pos`` / ``slot_of_pos`` 会被整块拿去索引
// （decode 路径不切片，见 ``moe.py`` 里那句「不切：三个张量保持 slots 长度」），
// 留着未初始化显存就等于往随机 token 行上加随机权重。
//
// @param expert_idx   ``[slots]`` 展平的全局专家号。
// @param local_table  ``[num_experts]`` 全局号 → 本地号，不属于本 rank 为 -1。
// @param slot_pos     ``[slots]`` 输出：收拢后的行号，丢弃的为 -1。
// @param token_of_pos ``[slots]`` 输出：收拢后每行的 token 下标。
// @param slot_of_pos  ``[slots]`` 输出：收拢后每行来自哪个槽位。
// @param offsets      ``[local_experts + 1]`` 输入输出：先当计数器（``[e+1]`` 累加），
//                     再被前缀和整体覆盖。
// @param cursor       ``[local_experts]`` 暂存：各桶已经写到第几号。调用方常驻持有，
//                     每次进来由阶段 1 清零。
// @param slots        槽位数 ``tokens * top_k``。
// @param local_experts 本 rank 的专家数。
// @param num_experts  全局专家数。
// @param top_k         每 token 的专家数。
// @return 无。错误通过 host 侧 ``C10_CUDA_KERNEL_LAUNCH_CHECK`` 暴露。
__global__ void moe_dispatch_kernel(const int* expert_idx, const int* local_table,
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
        const int g = expert_idx[t];
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

}  // namespace

// @brief 分派：路由结果 → ``(offsets, slot_pos, token_of_pos, slot_of_pos)``。
//
// @param expert_idx   ``[slots]`` 展平的全局专家号，int32 连续。
// @param local_table  ``[num_experts]`` 全局号 → 本地号，int32，不属于本 rank 为 -1。
// @param out_offsets  ``[local_experts + 1]`` int32 输出。
// @param out_slot_pos ``[slots]`` int32 输出，收拢后的行号，丢弃的为 -1。
// @param out_tokens   ``[slots]`` int32 输出，收拢后每行的 token 下标。
// @param out_slots    ``[slots]`` int32 输出，收拢后每行来自哪个槽位。
// @param scratch      ``[>= local_experts]`` int32 暂存，用作各桶游标。调用方常驻
//                     持有（``_disp_scratch`` 按设备缓存），每次由阶段 1 清零。
// @param slots, local_experts, num_experts, top_k 几何。
// @return 无。错误通过 host 侧 ``C10_CUDA_KERNEL_LAUNCH_CHECK`` 暴露。
// @throws c10::Error 布局、设备、dtype 或形状不自洽，或共享内存预算超限。
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

    // 共享内存预算：counts[E] + starts[E+1] = 2 * local_experts + 1 个 int。
    // 48 KB 是本卡动态共享内存不用额外属性的上限（``platform/scow/AGENTS.md``），
    // 512 专家只要 4 KB，所以这条断言实际永远不会触发——留着的意义是别让某天
    // 换成一个专家更多的模型时，静默写到 LDS 之外。
    const int64_t smem_bytes = (2 * local_experts + 1) * (int64_t)sizeof(int);
    TORCH_CHECK(smem_bytes <= 48 * 1024,
                "moe_dispatch needs ", smem_bytes, " bytes of dynamic shared memory, "
                "which exceeds the 48 KB limit without extra attributes; "
                "local_experts=", local_experts);

    const c10::cuda::CUDAGuard guard(expert_idx.device());
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    // 单 block：见文件头「为什么仍然是单 block」。
    // 线程数按槽位数分两档：decode 只有 10 个槽位，开 512 个线程纯属浪费发射；
    // prefill 512 token 有 5120 个，每线程 10 个比 20 个少一半循环。
    const int threads = (slots >= 1024) ? 512 : 256;
    moe_dispatch_kernel<<<1, threads, (size_t)smem_bytes, stream>>>(
        expert_idx.data_ptr<int>(), local_table.data_ptr<int>(),
        out_slot_pos.data_ptr<int>(), out_tokens.data_ptr<int>(),
        out_slots.data_ptr<int>(), out_offsets.data_ptr<int>(),
        scratch.data_ptr<int>(),
        (int)slots, (int)local_experts, (int)num_experts, (int)top_k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// 绑定写在内核旁边：名字、参数名与默认值只有这一份，`bindings.cpp` 只遍历注册表。
COFFIN_BIND(moe_dispatch,
            "\u5206\u6d3e\uff1a\u8def\u7531\u7ed3\u679c \u2192 \u524d\u7f00\u548c + \u884c\u5e8f",
            pybind11::arg("expert_idx"), pybind11::arg("local_table"), pybind11::arg("out_offsets"),
            pybind11::arg("out_slot_pos"), pybind11::arg("out_tokens"), pybind11::arg("out_slots"),
            pybind11::arg("scratch"), pybind11::arg("slots"), pybind11::arg("local_experts"),
            pybind11::arg("num_experts"), pybind11::arg("top_k"));
