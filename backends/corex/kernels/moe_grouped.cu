// @file moe_grouped.cu
// @brief MoE 的两端：路由融合（softmax → top-k → 归一化）与稀疏专家的分组 W4A8 投影。
//
// Module: coffin/backends/corex/kernels。由 CoreX 后端的分派层调用，
// ``models/qwen4exp`` 通过它跑 MoE 的路由与两个投影。
//
// 为什么需要专门的内核：模型有 512 个专家、每 token 取 10 个，EP=4 之后每卡
// 持 128 个。decode 一步里每个专家平均只分到 1-2 行激活，若按「每 token 跑
// 10 次完整 GEMM」写，一个 2.4 MB 的专家权重会被读到 10 次——权重读取放大到
// token 数的倍数，而 decode 本来就是权重带宽受限。
//
// 按专家分组的形态下，每个专家的权重在一个步里只读一遍：所有路由到它的行
// 排在一起（调用方按专家号排序过），一次 GEMV 结构吃完整块。
//
// 分组内核有两个内核、一个入口：``moe_grouped`` 在 host 侧按「每专家几行」与
// 「每片还剩几组活」派给单趟内核或分片内核。判据与实测数字见 ``moe_grouped`` 里那段。
//
// 路由那一端（``moe_router``）与投影在同一个文件里，因为它是这条数据流的头部：
// 路由交出 ``top_idx`` / ``top_w``，调用方立刻用它们收拢激活喂给下面的分组内核。
// 它替换的是 ``softmax -> topk -> div -> sum -> to`` 这条 torch 链，理由见该内核
// 上方「为什么需要它」。
//
// Constraints:
//   * ``x`` 必须是已按专家号排序的激活，``offsets`` 是 ``[E+1]`` 前缀和。
//     内核不排序、也不检查有序性：乱序的输入会算出看着合理的错数。
//   * 空专家直接跳过。``offsets[e] == offsets[e+1]`` 时该段零行，启动一个空块
//     只会白付一次调度。
//   * 每个 K 组先做 int32 累加再乘该组 scale；不同组的整数和不能直接相加
//     （``w4a8.cu`` 的约定）。
//   * 解包只在寄存器里发生，常驻权重始终是 uint8 packed，不分配整矩阵 int8 副本。
//   * **改这个文件之前先读这三类「静默失败」**。它们的共同点是跑一遍对上了也发现不了，
//     只能靠代码推理或专门判据拦住：
//
//     1. **数值正确、调度错**。例：1-D 拍平只用 ``(blockIdx.x, blockIdx.y)`` 定位，
//        若 ``grid.z`` 仍按 ``active_count`` 开（128），每个簇会被 128 个块各算一遍 ——
//        输出值完全相同，只是白跑 100 倍以上。同类：空簇守卫写成模式相关、``cluster_step``
//        语义搞反、``span>1`` 的 ``part`` 索引与 1-D 路径混用。
//        判据：``blocks_expect`` 不随行数涨、``max_per_expert`` 在只改调度时不变。
//     2. **时间正确、归因错**。历史上四次（MoE 占比 17% 当成 55%、620 ms 当 240 ms、
//        ``rows_per_expert`` 10 当 14.3、均匀分布当 40 倍偏斜）。外推出来的账必须先有
//        形状打点（``COFFIN_MOE_SHAPE=1``）再分配优化预算，否则优化的可能是只存在于
//        探针里的形状。
//     3. **输出正常、承诺错**。例：``index_add_`` 走 CUDA ``atomicAdd``，浮点加法不满足
//        结合律 ⇒ 同一 prompt 两次前向不逐位相同（``self_repeat`` 1.27-2.10，悬了五轮）。
//        任何「同输入同输出」的承诺都要有专门判据（现在 ``moe.py`` 的融合合并用定序
//        ``sum(dim=1)``，``SELF_REPEAT`` 实测 0.0），不能靠「看起来没问题」。
//   * 不用 CUDA graph、不用 cp.async（``platform/scow/AGENTS.md`` 第 1、5 条）。
//   * 分组编号用 ``blockIdx.z``（专家）与 ``blockIdx.x``（输出列）：一个三维网格
//     让"哪块算哪个专家"由硬件分发，不需要内核里查表。

#include "corex_bindings.h"
#include "corex_ops.h"
#include "corex_scalar.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <climits>
#include <cstdint>

namespace {

// @brief 把 4 个 k 的 4 位权重（存在两个字节里）拼成 dp4a 要的 4 字节。
//
// 放在两个内核之前（单趟形态与分片形态都用它）：同一个 bit 打包规则只能有一份实现，
// 两份拷贝里改错一份的症状是「某些 k 配到了相邻输出列的权重」，算出来的数看着合理。
//
// @param lo k、k+1 两个权重所在字节（低 nibble 是 k）。
// @param hi k+2、k+3 两个权重所在字节。
// @return 字，四个字节依次是 k、k+1、k+2、k+3 的量化值（已减零点 8）。
__device__ __forceinline__ unsigned w4_dword(int lo, int hi) {
    return ((unsigned)((lo & 15) - 8) & 255)
        | (((unsigned)((lo >> 4) - 8) & 255) << 8)
        | (((unsigned)((hi & 15) - 8) & 255) << 16)
        | (((unsigned)((hi >> 4) - 8) & 255) << 24);
}

}  // namespace

namespace {

// 分组 GEMV 的**分片形态**：只在「每专家 1 行」这一档启用（见 ``moe_grouped`` 的分档）。
//
// ## 改成「行在块内、k 在行外」之前为什么慢
//
// 它的形状是「一个线程一个输出列、行方向再铺 2 层（``blockDim.y = 2``）、沿 K 串行
// 走完 40 个 weight-group，每个 k 步一次单字节全局读、一次 dp4a」。B=1 decode 时：
//
//   * 每专家恰好 1 行，所以 ``threadIdx.y == 1`` 那一半线程从第一个判断就退出；
//   * ``grid = (ceil(N/128), min(8,tokens), 128)`` 里只有约 2.5 x ceil(N/128) 个块
//     真有行（EP=4 后每层 top-10 只命中本 rank 的 ~2.5 个），其余块几乎免费退出；
//   * 于是「有活的块」只有 25（gate_up N=1280）/ 50（down N=2560）个，每个 128 个
//     有效线程 = 2 个硬件 warp。**整张卡上只有 50-100 个 warp 在跑**，而每线程是
//     1280 次「一次读一次 dp4a」的依赖链、循环体没有展开 —— 每个 warp 只有 1-2 个
//     读在飞，谁也盖不住 400+ 周期的访存延迟。
//
// 按 packed 字节算，一次 gate_up 调用只读约 4.4 MB，而实测 203 us/次 —— 有效带宽
// 约 8 GB/s 对 575 GB/s 的屋顶。空块不是成本（历史结论：92% 空块只快 2.8%），
// **在飞的读太少才是**。
//
// ## 两条修法（都在这个内核里）
//
//   1. **内层循环 8 路展开**：把 8 个 k 步的字节读先全部发出去再做 4 次 dp4a，
//      每个线程同时在飞的读从 1-2 个变成 8 个。
//   2. **按 weight-group 切 K（split-K）**：块数直接乘 ``slices``。因为 scale 是
//      逐 (group, col) 的，切片按 group 边界切就**不需要改算术**：每片把自己那几组的
//      ``int32 和 x scale`` 累加成 fp32 部分和，最后一遍归约求和再乘激活 scale。
//      部分和缓冲只有 ``slices x tokens x N`` 个 fp32（decode 时几十 KB），远小于
//      它换来的并行度。
//
// **为什么只在 1 行档启用**：作业 213 的对拍显示每专家 4 行以上时这个内核反而更慢
// （gate_up 447.5 -> 608.6 us，0.74x；64 专家 x 40 行时 0.90x）—— 部分和的写读与
// 多出来的一次归约发射是固定成本，而行数一多，旧的「行在外、k 在内」结构每个块内
// 重叠访存延迟，不需要靠切 K 换并行度。``down``（K=640、只有 5 个 weight-group）
// 连 1 行档都更慢（35.9 -> 57.3 us），所以分档里还要求 ``groups >= 16``。
//
// Constraints:
//   * ``slices == 1`` 时不写部分和、直接出 bf16/fp32，与历史上那条路是同一套算术。
//   * ``slices`` 超过 weight-group 数没有意义（空片），host 侧夹住。
//   * 行循环在**块内**、``k`` 在行循环之外：一次读到的权重字节服务 ``GEMV_ROWS`` 行，
//     权重的重复读从「x 行数」降到「x ceil(行数 / GEMV_ROWS)」。``gridDim.y`` 与
//     ``blockDim.y`` 都恒为 1，并行度全部来自 x（列）与 z（专家 x 切片）。
//   * 空片必须写 0：分到 0 个 weight-group 的片（``groups < span`` 时才会出现）仍要把
//     自己覆盖的行写 0，否则归约会加上未初始化显存。
//   * 有效行之外的尾部（EP 分片丢弃的槽位）由归约核按 ``offsets[E]`` 直接写 0：它们
//     下游要乘一个 0 权重再加回去，留未初始化显存就等于往残差上加随机数。这一条让部分
//     和缓冲可以用 ``torch::empty``，省掉「每次调用先清零」那一次发射（96 次/步）。

//: 列块宽。与历史上的 ``threads_x`` 一致，保持 128 字节的整事务。
constexpr int GEMV_COLS = 128;

//: split-K 的片数上限。再大就只是空片。
constexpr int GEMV_MAX_SLICES = 8;

//: 分片形态里每个线程一次处理几行。**这一个数就是权重重读的除数**：
//: 行数 40 时从 40 遍降到 ceil(40/8) = 5 遍。
//:
//: 为什么是 8：每行一个 int32 组累加器 + 一个 fp32 跨组和，8 行就是 16 个寄存器；
//: 再大（16 行）会挤掉激活指针与预取的 8 个权重字节，反而开始溢出。
constexpr int GEMV_ROWS = 8;

//: 每多少个 ``rows`` 一个簇。必须与内核里的 ``GEMV_ROWS`` 一致。
constexpr int kClusterRows = GEMV_ROWS;

//: ``cp`` 共享内存的容量上限（Qwen3.8 全量 512 个专家 + 1）。
constexpr int kClusterCpMax = 521;

// @brief 算每专家的**簇前缀和**：``cp[e]`` = 第 e 个专家之前的簇总数。
//
// 为什么单独一个内核：这是 ``diff -> ceil -> cumsum`` 三步的等价物。用 torch 那三个算子
// 是 3 次发射，而本函数每次 prefill 被调 96 次 ⇒ 288 次发射约 1.7 ms，会把 1-D 拍平
// 的收益吃掉一部分。这里 1 次发射、单块、``E`` 个元素。
//
// 只有 thread 0 做累加（``E = 128``，128 次整数加法约 200 周期），其余线程负责装载
// 与回写 —— 比并行扫描简单，且不会引入跨块同步。
//
// @param offsets ``[E+1]`` 每专家的行区间前缀和。
// @param cp      ``[E+1]`` 输出：每专家的簇起点，末项是总簇数。
// @param experts 专家数。
__global__ void moe_cluster_prefix_kernel(const int* __restrict__ offsets,
                                          int* __restrict__ cp, int experts) {
    extern __shared__ int cbuf[];
    for (int e = (int)threadIdx.x; e < experts; e += (int)blockDim.x) {
        const int rows = offsets[e + 1] - offsets[e];
        cbuf[e] = (rows + kClusterRows - 1) / kClusterRows;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int acc = 0;
        for (int e = 0; e < experts; ++e) {
            const int c = cbuf[e];
            cbuf[e] = acc;
            acc += c;
        }
        cbuf[experts] = acc;
        for (int e = 0; e <= experts; ++e) cp[e] = cbuf[e];
    }
}

// @brief 分片形态的分组 W4A8 GEMV：列一个线程、行在块内铺开、K 按 weight-group 切。
//
// @tparam XT 输出 dtype：``uint16_t``（bf16）或 ``float``。
//
// @param x       已排序的 int8 激活，``[tokens, K]``。
// @param xs      逐 token 激活 scale，``[tokens]`` fp32。
// @param w       专家栈 packed W4，``[E, ceil(K/2), N]``。
// @param ws      专家栈 scale，``[E, groups, N]``。
// @param offsets ``[E+1]`` 前缀和。
// @param active  ``[n_active]`` 非空专家号，``nullptr`` 表示 z 维就是专家号。
// @param part    ``[slices, tokens, N]`` fp32 部分和，``slices > 1`` 时用。
// @param out     ``[tokens, N]`` 目的地，``slices == 1`` 时直接写这里。
// @param cp          ``[E+1]`` 簇前缀和；``nullptr`` 表示走老的「每专家 N 块」路径。
// @param total       ``cp[experts]``（总簇数），只在 ``cp != nullptr`` 时有意义。
// @param K, N, gs, experts, tokens, active_count 几何。
// @param slices  K 方向分片数，``1`` 表示不切。
template <typename XT>
__global__ void moe_grouped_split_kernel(const int8_t* __restrict__ x,
                                         const float* __restrict__ xs,
                                         const uint8_t* __restrict__ w,
                                         const float* __restrict__ ws,
                                         const int* __restrict__ offsets,
                                         const int* __restrict__ active,
                                         float* __restrict__ part, XT* __restrict__ out,
                                         int K, int N, int gs, int experts, int tokens,
                                         int active_count, int slices,
                                         const int* __restrict__ cp, int total) {
    // 两种定位模式：
    //   * ``cp == nullptr``：老的「z 维是专家、y 维是行簇分段」，用于每专家行数很少的档；
    //   * ``cp != nullptr``：1-D 拍平，``blockIdx.y`` 是**全局簇号**，二分查表得出
    //     (专家, 簇)，每块恰好一个簇 —— 负载偏斜下这是唯一能完全均衡的形状。
    int expert_pos;
    int cluster_base;
    int cluster_step;
    if (cp != nullptr) {
        // cp 整块装进共享内存（E+1 个 int，512 B），二分在共享内存里做。
        __shared__ int s_cp[kClusterCpMax];
        const int ncp = experts + 1;
        for (int e = (int)threadIdx.x; e < ncp; e += (int)blockDim.x) s_cp[e] = cp[e];
        __syncthreads();
        const int target = (int)blockIdx.y;
        // 上界块会落在 ``target >= s_cp[experts]`` 上：读一次就退出，代价可忽略。
        if (target >= s_cp[experts]) return;
        // 找「最后一个 s_cp[e] <= target 的 e」：cp 单调不减，标准 lower_bound 的变体。
        int lo = 0, hi = experts;
        while (lo + 1 < hi) {
            const int mid = (lo + hi) >> 1;
            if (s_cp[mid] <= target) lo = mid; else hi = mid;
        }
        expert_pos = lo;
        cluster_base = (target - s_cp[lo]) * kClusterRows;
        // 只做这一个簇：步长取 ``rows`` 一步就跳出循环（见下面的 for）。
        cluster_step = -1;
    } else {
        const int slot = (int)blockIdx.z % active_count;
        expert_pos = (active == nullptr) ? slot : active[slot];
        cluster_base = (int)blockIdx.y * kClusterRows;
        cluster_step = (int)gridDim.y * kClusterRows;
        if (expert_pos >= experts) return;
    }
    const int slice = (cp != nullptr) ? 0 : (int)blockIdx.z / active_count;
    const int expert = expert_pos;
    const int begin = offsets[expert];
    const int rows = offsets[expert + 1] - begin;
    if (rows <= 0) return;
    // 本块没有活干（1-D 的上界块，或老路径里 y 超出该专家的簇数）：统一在这里退出。
    // 这条对两种模式都成立，所以不写成 `cp != nullptr && ...`。
    if (cluster_base >= rows) return;

    const int col = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (col >= N) return;

    const int groups = (K + gs - 1) / gs;
    // 切片按 group 边界切：scale 是逐 (group, col) 的，切在组内就要处理半组，
    // 而切在边界上每片都还是「整数和 x scale」，算术与不切时逐位可比。
    const int g0 = slice * groups / slices;
    const int g1 = (slice + 1) * groups / slices;
    const uint8_t* wexp = w + (size_t)expert * (size_t)((K + 1) / 2) * (size_t)N;
    const float* sexp = ws + (size_t)expert * (size_t)groups * (size_t)N;
    float* dst = (slices == 1) ? nullptr : part + (size_t)slice * (size_t)tokens * (size_t)N;

    // **每专家 1 行**（decode）：走「一条 4 链」的老结构。这是作业 213 拿 3.2x 的那条路，
    // 必须原样保留 —— 8 行簇在这种形状下会白占 16 个寄存器（7 个是空的），且每个 k 步
    // 都生成 8 份收尾判断，等于用多行的结构跑单行的活。实测代价是 decode 一步
    // 139.87 -> 397.35 ms（作业 248），所以这一分支不是优化而是**正确性**：
    // 它守的是 decode 的整条快路径。
    if (rows == 1) {
        const int token = begin;
        const int8_t* xrow = x + (size_t)token * K;
        float total = 0.f;
        for (int g = g0; g < g1; ++g) {
            int s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            int k = g * gs;
            const int stop = min(K, k + gs);
            for (; (K & 15) == 0 && k + 16 <= stop; k += 16) {
                const int b0 = wexp[(size_t)(k / 2) * N + col];
                const int b1 = wexp[(size_t)(k / 2 + 1) * N + col];
                const int b2 = wexp[(size_t)(k / 2 + 2) * N + col];
                const int b3 = wexp[(size_t)(k / 2 + 3) * N + col];
                const int b4 = wexp[(size_t)(k / 2 + 4) * N + col];
                const int b5 = wexp[(size_t)(k / 2 + 5) * N + col];
                const int b6 = wexp[(size_t)(k / 2 + 6) * N + col];
                const int b7 = wexp[(size_t)(k / 2 + 7) * N + col];
                s0 = __dp4a((int)w4_dword(b0, b1),
                            *reinterpret_cast<const int*>(xrow + k), s0);
                s1 = __dp4a((int)w4_dword(b2, b3),
                            *reinterpret_cast<const int*>(xrow + k + 4), s1);
                s2 = __dp4a((int)w4_dword(b4, b5),
                            *reinterpret_cast<const int*>(xrow + k + 8), s2);
                s3 = __dp4a((int)w4_dword(b6, b7),
                            *reinterpret_cast<const int*>(xrow + k + 12), s3);
            }
            for (; (K & 3) == 0 && (k & 3) == 0 && k + 4 <= stop; k += 4) {
                const int lo = wexp[(size_t)(k / 2) * N + col];
                const int hi = wexp[(size_t)(k / 2 + 1) * N + col];
                s0 = __dp4a((int)w4_dword(lo, hi),
                            *reinterpret_cast<const int*>(xrow + k), s0);
            }
            for (; k < stop; ++k) {
                const int byte = wexp[(size_t)(k / 2) * N + col];
                const int q = (int)((byte >> ((k & 1) * 4)) & 15) - 8;
                s0 += q * (int)xrow[k];
            }
            total += (float)((s0 + s1) + (s2 + s3)) * sexp[(size_t)g * N + col];
        }
        if (slices == 1) {
            out[(size_t)token * N + col] = CxScalar<XT>::from_float(total * xs[token]);
        } else {
            dst[(size_t)token * N + col] = total;
        }
        return;
    }

    // 行方向：每个线程一次处理 ``GEMV_ROWS`` 行，**k 在它们之外循环**。
    //
    // 这一层顺序就是全部：原来「行在外、k 在内」时，每走一行都要把该专家的权重从头
    // 读一遍，于是权重流量 = 名义流量 x 行数。作业 227 的实测正是这个数：rows=8 时
    // 1782 MB / 3523.9 us = 505 GB/s（已经顶在 575 GB/s 的屋顶上，其中 87.5% 是重复读）；
    // rows=40 时 40 x 1.74 MB x 128 专家 = 8.9 GB，按屋顶算是 15.5 ms，实测 17.1 ms。
    //
    // 反过来之后，一次读到的权重字节服务 ``GEMV_ROWS`` 行，重复读从「x 行数」降到
    // 「x ceil(行数 / GEMV_ROWS)」——rows=40 时是 5 次，也就是 8 倍。
    // 逐位不变：每行的整数累加次序与切片的 group 划分都没动，只是**同一份权重被多行共用**。
    // 簇按 ``grid.y`` 轮流分配：负载偏斜下（最热专家的簇数是均值的 20-40 倍）这是唯一
    // 有效的均衡手段。**划分不复制** —— 每个簇仍只被一个块处理，每行仍只写一次。
    for (int base = cluster_base; base < rows;
         base += (cluster_step < 0 ? rows : cluster_step)) {
        const int nr = min(GEMV_ROWS, rows - base);
        // 每行一个 int32 组累加器 + 一个 fp32 跨组和；GEMV_ROWS 是编译期常量，
        // 这两个数组因此留在寄存器里（动态下标的版本会掉进 local memory）。
        float total[GEMV_ROWS];
#pragma unroll
        for (int j = 0; j < GEMV_ROWS; ++j) total[j] = 0.f;
        for (int g = g0; g < g1; ++g) {
            int acc[GEMV_ROWS];
#pragma unroll
            for (int j = 0; j < GEMV_ROWS; ++j) acc[j] = 0;
            int k = g * gs;
            const int stop = min(K, k + gs);
            // 8 路展开：8 个权重字节先全发出去，4 次 dp4a 分别作用到 GEMV_ROWS 行。
            for (; (K & 15) == 0 && k + 16 <= stop; k += 16) {
                const int b0 = wexp[(size_t)(k / 2) * N + col];
                const int b1 = wexp[(size_t)(k / 2 + 1) * N + col];
                const int b2 = wexp[(size_t)(k / 2 + 2) * N + col];
                const int b3 = wexp[(size_t)(k / 2 + 3) * N + col];
                const int b4 = wexp[(size_t)(k / 2 + 4) * N + col];
                const int b5 = wexp[(size_t)(k / 2 + 5) * N + col];
                const int b6 = wexp[(size_t)(k / 2 + 6) * N + col];
                const int b7 = wexp[(size_t)(k / 2 + 7) * N + col];
                const int q0 = (int)w4_dword(b0, b1);
                const int q1 = (int)w4_dword(b2, b3);
                const int q2 = (int)w4_dword(b4, b5);
                const int q3 = (int)w4_dword(b6, b7);
#pragma unroll
                for (int j = 0; j < GEMV_ROWS; ++j) {
                    if (j >= nr) continue;         // 尾块的空行不读激活
                    const int8_t* xr = x + (size_t)(begin + base + j) * K;
                    acc[j] = __dp4a(q0, *reinterpret_cast<const int*>(xr + k), acc[j]);
                    acc[j] = __dp4a(q1, *reinterpret_cast<const int*>(xr + k + 4), acc[j]);
                    acc[j] = __dp4a(q2, *reinterpret_cast<const int*>(xr + k + 8), acc[j]);
                    acc[j] = __dp4a(q3, *reinterpret_cast<const int*>(xr + k + 12), acc[j]);
                }
            }
            // 尾段沿用同一套两种写法（4 路 dp4a 与标量），让 K 或组尾不是 16 的倍数时仍然对。
            for (; (K & 3) == 0 && (k & 3) == 0 && k + 4 <= stop; k += 4) {
                const int lo = wexp[(size_t)(k / 2) * N + col];
                const int hi = wexp[(size_t)(k / 2 + 1) * N + col];
                const int q = (int)w4_dword(lo, hi);
#pragma unroll
                for (int j = 0; j < GEMV_ROWS; ++j) {
                    if (j >= nr) continue;
                    const int8_t* xr = x + (size_t)(begin + base + j) * K;
                    acc[j] = __dp4a(q, *reinterpret_cast<const int*>(xr + k), acc[j]);
                }
            }
            for (; k < stop; ++k) {
                const unsigned byte = wexp[(size_t)(k / 2) * N + col];
                const int q = (int)((byte >> ((k & 1) * 4)) & 15) - 8;
#pragma unroll
                for (int j = 0; j < GEMV_ROWS; ++j) {
                    if (j >= nr) continue;
                    const int8_t* xr = x + (size_t)(begin + base + j) * K;
                    acc[j] += q * (int)xr[k];
                }
            }
            const float scale = sexp[(size_t)g * N + col];
#pragma unroll
            for (int j = 0; j < GEMV_ROWS; ++j) {
                if (j < nr) total[j] += (float)acc[j] * scale;
            }
        }
#pragma unroll
        for (int j = 0; j < GEMV_ROWS; ++j) {
            if (j >= nr) continue;
            const int token = begin + base + j;
            if (slices == 1) {
                out[(size_t)token * N + col] = CxScalar<XT>::from_float(total[j] * xs[token]);
            } else {
                dst[(size_t)token * N + col] = total[j];
            }
        }
    }
}

// @brief 把 ``slices`` 份部分和加起来，乘激活 scale，窄化一次写出。
//
// @tparam XT 输出 dtype。
//
// @param part    ``[slices, tokens, N]`` fp32 部分和。
// @param xs      ``[tokens]`` fp32 激活 scale。
// @param out     ``[tokens, N]`` 目的地。
// @param offsets ``[E+1]`` 前缀和，只用最后一项（有效行数）。
// @param tokens, N, slices, experts 几何。
//
// **为什么需要最后那一项**：调用方的行数 ``tokens`` 可以大于有效行数——EP 分片下不归
// 本 rank 的槽位会被丢掉，但收拢后的激活张量仍有 ``slots`` 行，尾部那几行没有任何专家
// 写过（它们下游要乘一个 0 权重再加回去，所以必须是**确定的 0**，``0 * nan`` 还是 nan）。
// 分片内核不会碰那些行，所以这里按 ``offsets[experts]`` 把尾部直接写 0。这一条让部分和
// 缓冲可以用 ``torch::empty``：省掉一次清零发射（96 次/步），而语义与旧路径「调用方先把
// out 清零」完全等价。
template <typename XT>
__global__ void moe_grouped_reduce_kernel(const float* __restrict__ part,
                                          const float* __restrict__ xs,
                                          XT* __restrict__ out,
                                          const int* __restrict__ offsets, int tokens,
                                          int N, int slices, int experts) {
    const int col = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int row = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (col >= N || row >= tokens) return;
    if (row >= offsets[experts]) {
        out[(size_t)row * N + col] = CxScalar<XT>::from_float(0.f);
        return;
    }
    float acc = 0.f;
    for (int s = 0; s < slices; ++s) {
        acc += part[((size_t)s * (size_t)tokens + (size_t)row) * (size_t)N + (size_t)col];
    }
    out[(size_t)row * N + col] = CxScalar<XT>::from_float(acc * xs[row]);
}

}  // namespace


// @brief 分组 W4A8 投影：``[tokens, K] -> [tokens, N]``，每专家独立权重。
//
// @param a        已按专家号排序的 int8 激活，``[tokens, K]`` 连续。
// @param xs       逐 token 激活 scale，``[tokens]`` fp32。
// @param w        专家栈 packed W4，``[E, ceil(K/2), N]`` 连续 uint8。
// @param ws       专家栈 scale，``[E, ceil(K/group), N]`` 连续 fp32。
// @param offsets  ``[E+1]`` 前缀和，int32，``offsets[0] == 0`` 且
//                 ``offsets[E] == tokens``。分布在前缀里的空段表示该专家没有行。
// @param out      ``[tokens, N]`` 连续，bf16 或 fp32，dtype 由它决定。
// @param group_size K 轴权重分组大小。
// @param active   可选，``[n_active]`` int32：**非空专家号**的紧凑表，由调用方用
//                 ``nonzero(offsets[1:] > offsets[:-1])`` 现成算出来。给了它，网格
//                 的 z 维就是 ``n_active`` 而不是 ``experts``，``blockIdx.z`` 先经
//                 它翻译成真正的专家号。空张量（``numel() == 0``）表示退回全量网格。
// @return out。
// @throws c10::Error 布局、设备、dtype、形状或前缀和不自洽。
//
// 本函数内部按**每专家行数**分两档，调用方不需要知道：每专家 1 行（B=1 decode，用
// ``tokens <= top_k`` 在 host 侧判定，不读 ``offsets``）走「8 路展开 + gate_up 切 8 片」
// 的内核，实测 gate_up 203.3 -> 62.9 us、down 58.5 -> 35.9 us（作业 213，逐位相同）；
// 其余走下面的单趟内核。分档理由与反例（每专家 4 行以上分片慢 26%）见启动段那段注释。
//
// 为什么需要 ``active``：decode 一个 token 只选 ``top_k``=10 个专家，而 EP=4 后
// 每卡持 128 个。按全量网格发射时 z 维是 128，其中只有约 10 个真有行，于是
// ``grid = (N/128) x min(8,tokens) x 128`` 里 92% 的 block 一进来就
// ``rows <= 0`` 退出。它们不读权重，但每个都要占一次硬件调度槽，而本卡实测一步
// 有 92 次这样的发射（``q4budget2``/``q4attrib2``），发射地板又是 5.76 us/次
// （``q4floor1``）——这笔调度开销是白付的。
//
// 判定这一条值得做的算术：``gate_up`` 的 N=1024 给出 grid (8,8,128)=8192 块，
// 有行的只有 640 块；``down`` 的 N=2560 给出 (20,8,128)=20480 块，有行的 1600 块。
// 两者都是 **92.2% 的块在空转**。
torch::Tensor moe_grouped(torch::Tensor a, torch::Tensor xs, torch::Tensor w,
                          torch::Tensor ws, torch::Tensor offsets,
                          torch::Tensor out, int64_t group_size,
                          c10::optional<torch::Tensor> active) {
    TORCH_CHECK(a.is_cuda() && a.dim() == 2 && a.scalar_type() == torch::kChar
                && a.is_contiguous(), "activation must be contiguous CUDA int8 [tokens, K]");
    TORCH_CHECK(xs.is_cuda() && xs.dim() == 1 && xs.scalar_type() == torch::kFloat
                && xs.is_contiguous() && xs.size(0) == a.size(0),
                "activation scale must be contiguous fp32 [tokens]");
    TORCH_CHECK(w.is_cuda() && w.dim() == 3 && w.scalar_type() == torch::kByte
                && w.is_contiguous(), "weight must be contiguous CUDA uint8 [E, K/2, N]");
    TORCH_CHECK(ws.is_cuda() && ws.dim() == 3 && ws.scalar_type() == torch::kFloat
                && ws.is_contiguous(), "weight scale must be contiguous fp32 [E, groups, N]");
    TORCH_CHECK(offsets.is_cuda() && offsets.dim() == 1
                && offsets.scalar_type() == torch::kInt && offsets.is_contiguous(),
                "offsets must be contiguous CUDA int32 [E+1]");
    TORCH_CHECK(out.is_cuda() && out.dim() == 2 && out.is_contiguous()
                && out.size(0) == a.size(0),
                "destination must be contiguous CUDA [tokens, N]");
    TORCH_CHECK(out.scalar_type() == torch::kBFloat16
                || out.scalar_type() == torch::kFloat, "destination must be bf16 or fp32");

    const int tokens = (int)a.size(0);
    const int K = (int)a.size(1);
    const int N = (int)out.size(1);
    const int experts = (int)w.size(0);
    const int groups = (K + (int)group_size - 1) / (int)group_size;
    TORCH_CHECK(group_size > 0 && K > 0 && N > 0 && experts > 0, "sizes must be positive");
    TORCH_CHECK((int)w.size(2) == N, "weight N ", w.size(2), " != destination N ", N);
    TORCH_CHECK((int)w.size(1) * 2 >= K, "weight K axis too short for K ", K);
    TORCH_CHECK((int)ws.size(0) == experts && (int)ws.size(1) == groups
                && (int)ws.size(2) == N, "weight scale must be [E, ceil(K/group), N]");
    TORCH_CHECK((int)offsets.size(0) == experts + 1, "offsets must have E+1 entries");
    if (tokens == 0) return out;

    // 紧凑专家表：给了就按它的长度开 z 维，否则退回全量。
    int active_count = experts;
    const int* active_ptr = nullptr;
    torch::Tensor active_dev;
    if (active.has_value() && active->defined() && active->numel() > 0) {
        TORCH_CHECK(active->is_cuda() && active->dim() == 1
                    && active->scalar_type() == torch::kInt && active->is_contiguous(),
                    "active expert map must be contiguous CUDA int32 [n_active]");
        active_count = (int)active->numel();
        // 连续化一次：调用方传进来的通常是 ``nonzero().flatten()``，本来就连续，
        // 但 ``contiguous()`` 在已连续时是零拷贝，留着它比依赖调用方更安全。
        if (active->is_contiguous()) {
            active_dev = *active;
        } else {
            active_dev = active->contiguous();
        }
        active_ptr = active_dev.data_ptr<int>();
    }

    const c10::cuda::CUDAGuard guard(a.device());
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // 分档判据：**每专家行数**。既不是总 token 数，也不是网格的 y 维。
    //
    // 两个实测事实决定了它：
    //   1. 内核里行在**块内**循环、k 在行外，所以权重流量的冗余倍数 =
    //      ``ceil(每专家行数 / GEMV_ROWS)``。rows=40 时是 5 遍（作业 247：
    //      per_expert 133.8 -> 40.0 us、call 17128 -> 5117 us，`rereads` 40 -> 5）。
    //   2. 而每专家只有 **1** 行时（decode），行簇结构白占 16 个寄存器（7 个是空的）、
    //      每个 k 步生成 8 份收尾判断 —— 实测把 decode 一步从 **139.87 拖到 397.35 ms**
    //      （作业 248）。所以内核里专门留了一条 ``rows == 1`` 的老结构分支
    //      （作业 213 的 3.2x 就是那条），**分派必须保证 1 行的形状走它**。
    //
    // host 侧判据：``tokens <= experts`` 就是「每专家至多 1 行」—— 活跃专家数不超过
    // tokens，而 experts 是本地专家数。两个区间的量级差得很开：
    //   * decode：tokens = 2-3（EP4 下 top-10 只有约 2.5 个落在本 rank）远小于 128；
    //   * 分块 prefill（512 token）：tokens = 5120、每专家 40 行，远大于 128。
    // 所以判据不会踩在边界上。**不读 offsets**：那是一次设备到主机的同步，而 decode
    // 一步有 96 次这个调用。
    //
    // 历史教训：这里曾用 ``gridDim.y = min(8, tokens) > 1`` 当判据，而 decode 的
    // tokens=10 让它是 8 —— 判据与「每专家几行」脱钩，把 decode 送进了多行结构。
    // 判据必须直接对着**每专家行数**，代理量会在边界上骗人。
    constexpr int kSplitMinGroups = 16;   // 切 K 要每片仍有一整个 weight-group 的活
    const bool single_row = tokens <= experts;
    // 1 行档：gate_up（20 组）切 8 片换并行度，down（5 组）不切 —— 作业 213 实测
    // 203.3 -> 62.9 us（span=8）与 58.5 -> 35.9 us（span=1，切 5 片反而 57.3）。
    // 多行档：``span = 1``，行在块内按 ``GEMV_ROWS`` 簇循环，权重重读 ceil(行数/8) 遍。
    const int span = (single_row && groups >= kSplitMinGroups)
                         ? std::min(GEMV_MAX_SLICES, groups)
                         : 1;
    // 行簇并行度：只在「每专家多行」那一档打开。``tokens <= experts`` 时每专家至多 1 行、
    // 只有 1 个簇，多开的 y 全是立刻退出的空块，白付调度开销。
    //
    // **抬 y 不增加带宽**：权重重读由**簇数**决定（k 循环在簇循环里面，每个簇各读一遍
    // 权重），与 ``grid.y`` 无关 —— 抬 y 只是把同一批簇分给更多块，空转块读两次
    // ``offsets`` 就退出（10x128x(K-1) 个空转块约 90 us，可忽略）。
    //
    // 取 8 的依据：G=4 实测 10.10 -> 4.02 ms/次（作业 298），热专家 348-493 行（44-62 簇）
    // 分 4 份后每块仍有 11-16 簇，仍是关键路径。抬到 8 后每块 6-8 簇。
    // 若 G=8 仍 >3 ms/次，剩下的就是**冷专家长尾**（128 个专家各 1 簇，与热专家无关），
    // 那时该把 (专家, 簇) 对拍平成 1-D grid（设备侧对簇前缀和做一次 searchsorted，
    // 多一次发射），而不是继续抬 y。
    const bool flat1d = tokens > experts;      // 多行档；单行档仍走 z 维（与 span>1 互斥）
    const unsigned cluster_y = flat1d
        ? (unsigned)((tokens + kClusterRows - 1) / kClusterRows + experts)
        : 1u;
    // **z 维在 1-D 模式下必须压到 1**：那条路只用 (blockIdx.x, blockIdx.y) 定位，
    // 保留 ``active_count`` 个 z 会让每个簇被 128 个块各算一遍 —— 结果仍然正确
    // （同一份输入写同一份输出），但白跑 100 倍以上，属于「数值对、调度错」的静默事故。
    //
    // **改动须知**：若将来要把 (专家, 簇) 从 grid.y 改回 z 维（例如为了跨专家合并块），
    // 必须**同时**把 ``flat1d`` 关掉，否则两套定位会同时生效、又回到 128 倍空转。
    // 这条约束没有编译期检查，只能靠这句话。
    const unsigned grid_z = flat1d ? 1u : (unsigned)(active_count * span);
    const dim3 sblock(GEMV_COLS, 1);
    const dim3 sgrid((unsigned)((N + GEMV_COLS - 1) / GEMV_COLS), cluster_y, grid_z);
    // 簇前缀和：1 次发射、单块、E 个元素（torch 的 diff+ceil+cumsum 是 3 次发射，
    // 96 次调用约 1.7 ms，会把 1-D 拍平的收益吃掉一部分）。
    torch::Tensor cluster_cp;
    const int* cp_ptr = nullptr;
    if (flat1d) {
        // ``empty`` 而不是 ``zeros``：内核会写满 E+1 项。
        cluster_cp = torch::empty({(int64_t)experts + 1},
                                  out.options().dtype(torch::kInt));
        const size_t cp_smem = (size_t)(experts + 1) * sizeof(int);
        moe_cluster_prefix_kernel<<<1, 128, cp_smem, stream>>>(
            offsets.data_ptr<int>(), cluster_cp.data_ptr<int>(), experts);
        cp_ptr = cluster_cp.data_ptr<int>();
    }
    // span == 1 时内核直接写 out（不写部分和），不需要额外缓冲。
    torch::Tensor part;
    float* part_ptr = nullptr;
    if (span > 1) {
        // ``empty`` 而不是 ``zeros``：有效行之外的尾部由归约核按 ``offsets[E]`` 写 0，
        // 省掉每次调用一次清零发射（96 次/步）。
        part = torch::empty({(int64_t)span, (int64_t)tokens, (int64_t)N},
                            out.options().dtype(torch::kFloat));
        part_ptr = part.data_ptr<float>();
    }
    auto launch = [&](auto tag) {
        using XT = decltype(tag);
        XT* dst = reinterpret_cast<XT*>(out.data_ptr());
        moe_grouped_split_kernel<XT><<<sgrid, sblock, 0, stream>>>(
            a.data_ptr<int8_t>(), xs.data_ptr<float>(), w.data_ptr<uint8_t>(),
            ws.data_ptr<float>(), offsets.data_ptr<int>(), active_ptr, part_ptr, dst,
            K, N, (int)group_size, experts, tokens, active_count, span,
            cp_ptr, (int)cluster_y);
        if (span > 1) {
            const dim3 rblock(128, 8);
            const dim3 rgrid((unsigned)((N + 127) / 128),
                             (unsigned)((tokens + 7) / 8));
            moe_grouped_reduce_kernel<XT><<<rgrid, rblock, 0, stream>>>(
                part_ptr, xs.data_ptr<float>(), dst, offsets.data_ptr<int>(), tokens,
                N, span, experts);
        }
    };
    if (out.scalar_type() == torch::kBFloat16) launch(uint16_t{});
    else launch(float{});
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

namespace {

// 路由：一个 32-lane 组算一个 token 的 ``softmax -> top-k -> 归一化``。
//
// ## 为什么需要它
//
// ``moe.py`` 的 ``TopKRouter.forward`` 在 ``[tokens, 512]`` 上取 top-10，而 decode
// 时 ``tokens`` 是 1。作业 ``q4attrib3`` 量到的这一族（48 层 × 每层一遍，ms 为每步）：
//
//     aten::topk   1.83 ms/步   (48 次/步, 38 us/次)   <- 只是从 512 个数里挑最大的 10 个
//     gatherTopK   0.86 + bitonicSort 0.79             <- topk 的内部实现，一个排序网络
//     aten::softmax 0.67 + aten::sum 0.63
//
// 38 us 与算术量无关：本卡上 torch 的 topk 走 bitonic 排序网络，对 512 个元素要
// 45 轮比较交换，而我们要的只是最大的 10 个。10 轮「找剩下的最大者」是 10 x 512
// 次比较，全部落在一个 warp 的寄存器里。
//
// ## 不排序怎么取 top-k
//
// 第 j 轮：本 lane 扫自己负责的 ``ELEMS`` 个值，挑出（未被选中的）最大者；32 lane
// 内用 xor 蝶形归约出赢家（值降序、下标升序）；赢家那个 lane 把自己的那一项标成
// 已用。每轮 O(ELEMS) 次比较 + 5 次 shuffle，没有全局往返、没有排序。
//
// ## 精度与语义（与参考实现逐项对齐）
//
//   * softmax 在 fp32 上做且先减行内最大值，与 ``torch.softmax`` 同一套，不会溢出。
//   * 概率相等时取**小下标**。``torch.topk`` 对等值的次序不作保证（本卡上是排序网络
//     给出的次序），这里固定成小下标，保证同一输入两次运行逐位相同。
//   * ``normalize`` 为真时权重是 top-k 概率除以它们自己的和（``norm_topk_prob``）。
//     数学上等于 ``v / Σv``，与参考实现的 ``(v/d) / Σ(v/d)`` 只差 fp32 的舍入次序。
//
// **近似说明（三处，都写明可观测影响）**：
//
//   1. ``__expf`` 代替 ``expf``。CUDA 的 ulp 误差界是 ``2 + 1.16|x|``，而 softmax 的入参
//      是 ``x - max <= 0``，实际落在 ``[-30, 0]``，所以相对误差 **<= 2e-6**。它要让
//      top-k 次序翻，需要第 10 与第 11 大的概率相差不到 2e-6 相对 —— 真实 logits 下
//      不会发生，只有构造性输入才会。
//   2. 归约次序。**max 这一处完全没有误差**（比较不产生舍入），只有 denom 的 512 项
//      fp32 求和使用的是 shuffle 蝶形而不是 torch 的块内归约：最坏界 ``n * eps = 3e-5``、
//      典型约 ``1e-6`` 相对。而且它是一个**公共因子**：归一化路径下权重是 ``v / Σv``，
//      分子分母同源，误差基本约掉，实测与参考的权重差 ~1e-7 相对。
//   3. 归一化少一次除法：参考是 ``(v/d) / Σ(v/d)``（三次舍入），这里是 ``v / Σv``
//      （一次），差 **<= 2 ulp fp32 = 1.2e-7 相对**。
//
// 以上三处都远小于权重落地时的 bf16 量化步长 **3.9e-3 相对**（8 位尾数），所以
// 「生成文本通顺、无 NaN、不退化重复」这条精度政策下不可观测。
//
// **唯一会改变最终专家选择的是等值处理**：两个概率完全相等时这里取**小下标**，
// 而 ``torch.topk`` 的次序由排序网络决定、不作保证。后果是选到另一个专家（输出因此
// 不同），但两个选择在数学上都对 —— 等值就是同一个数，权重逐位相同。
//
// 这一档**不是罕见情形**，实测（``check_router_ties.py``，bf16 randn logits）：
// ``tokens=512`` 时 **340/512 行（66.4%）** 的 top-11 内存在完全等值，``tokens=8`` 时
// 5/8。原因是 top-10 的值全挤在 2.5-3.5 这个窄区间，而 bf16 在 ``[2,4)`` 上的 ulp 是
// 0.0156 —— 这一段只有约 50 个位模式供 10 个数去分，生日悖论给出 ``1-exp(-45/50)``。
// fp32 logits 的对照是 0/5120，所以这完全是 bf16 的量化密度，不是算法。
//
// 由此定下验收口径：**不能按位置比下标**。位置 mismatch 里绝大部分由等值解释，
// 要判「选对没有」只能比值 —— 见 ``moe_router`` 那节的 ``w_set_diff``（选中的权重
// 多重集，非 0 才是真错误）。参考实现在同样的输入上本来也在做任意选择，所以
// 「与 torch 逐 token 同下标」在 bf16 logits 下按构造不可达。

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

}  // namespace

// @brief 路由融合：``softmax -> top-k -> 归一化``，一次发射。
//
// @param logits    ``[tokens, n]`` 连续 bf16 或 fp32 logits。
// @param out_idx   ``[tokens, top_k]`` int64 目的地，按概率降序。
// @param out_w     ``[tokens, top_k]`` 目的地，dtype 同 ``logits``。
// @param top_k     每 token 的专家数，``(0, 32]`` 且不超过 ``n``。
// @param normalize 是否把 top-k 概率再归一化。
// @return 无。错误经 host 侧 ``C10_CUDA_KERNEL_LAUNCH_CHECK`` 暴露。
// @throws c10::Error 布局、设备、dtype、形状或 top_k 越界。
//
// 取代的是 ``torch.softmax`` + ``torch.topk`` + ``div`` + ``sum`` + ``to`` 这条链：
// 它在 decode 一步里是 48 层 × 5 个内核，其中 38 us/次 的 topk 走的是 512 元素的
// bitonic 排序网络。这里不排序，10 轮「找剩下的最大者」全在寄存器里完成。
void moe_router(torch::Tensor logits, torch::Tensor out_idx, torch::Tensor out_w,
                int64_t top_k, bool normalize) {
    TORCH_CHECK(logits.is_cuda() && logits.dim() == 2 && logits.is_contiguous(),
                "router logits must be contiguous CUDA [tokens, experts]");
    TORCH_CHECK(logits.scalar_type() == torch::kBFloat16
                || logits.scalar_type() == torch::kFloat,
                "router logits must be bf16 or fp32");
    const int tokens = (int)logits.size(0);
    const int n = (int)logits.size(1);
    TORCH_CHECK(top_k > 0 && top_k <= 32, "top_k must be in (0, 32]");
    TORCH_CHECK(n > 0 && n <= ROUTER_LANES * ROUTER_ELEMS,
                "expert count ", n, " exceeds ", ROUTER_LANES * ROUTER_ELEMS);
    TORCH_CHECK(top_k <= n, "top_k ", top_k, " exceeds the expert count ", n);
    TORCH_CHECK(out_idx.is_cuda() && out_idx.dim() == 2 && out_idx.is_contiguous()
                && out_idx.scalar_type() == torch::kLong && out_idx.size(0) == tokens
                && out_idx.size(1) == top_k,
                "index destination must be contiguous CUDA int64 [tokens, top_k]");
    TORCH_CHECK(out_w.is_cuda() && out_w.dim() == 2 && out_w.is_contiguous()
                && out_w.scalar_type() == logits.scalar_type()
                && out_w.size(0) == tokens && out_w.size(1) == top_k,
                "weight destination must be contiguous CUDA [tokens, top_k] of the "
                "same dtype as the logits");
    if (tokens == 0) return;

    const c10::cuda::CUDAGuard guard(logits.device());
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const dim3 block(ROUTER_LANES * ROUTER_GROUPS);
    const dim3 grid((unsigned)((tokens + ROUTER_GROUPS - 1) / ROUTER_GROUPS));
    const bool narrow = n <= 16 * ROUTER_LANES;
    if (logits.scalar_type() == torch::kBFloat16) {
        const auto* src = reinterpret_cast<const uint16_t*>(logits.data_ptr());
        auto* dst = reinterpret_cast<uint16_t*>(out_w.data_ptr());
        if (narrow) {
            moe_router_t<16, uint16_t><<<grid, block, 0, stream>>>(
                src, out_idx.data_ptr<int64_t>(), dst, tokens, n, (int)top_k,
                normalize ? 1 : 0);
        } else {
            moe_router_t<32, uint16_t><<<grid, block, 0, stream>>>(
                src, out_idx.data_ptr<int64_t>(), dst, tokens, n, (int)top_k,
                normalize ? 1 : 0);
        }
    } else {
        if (narrow) {
            moe_router_t<16, float><<<grid, block, 0, stream>>>(
                logits.data_ptr<float>(), out_idx.data_ptr<int64_t>(),
                out_w.data_ptr<float>(), tokens, n, (int)top_k, normalize ? 1 : 0);
        } else {
            moe_router_t<32, float><<<grid, block, 0, stream>>>(
                logits.data_ptr<float>(), out_idx.data_ptr<int64_t>(),
                out_w.data_ptr<float>(), tokens, n, (int)top_k, normalize ? 1 : 0);
        }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// 绑定写在内核旁边：名字、参数名与默认值只有这一份，`bindings.cpp` 只遍历注册表。
COFFIN_BIND(moe_grouped,
            "\u5206\u7ec4 W4A8 \u6295\u5f71\uff1a\u6bcf\u4e2a\u4e13\u5bb6\u53ea\u7b97\u8def\u7531\u5230\u5b83\u7684\u884c",
            pybind11::arg("a"), pybind11::arg("xs"), pybind11::arg("w"), pybind11::arg("ws"),
            pybind11::arg("offsets"), pybind11::arg("out"), pybind11::arg("group_size"),
            // 紧凑非空专家表，可为 None。decode 一个 token 只碰约 10 个专家，而每卡
          // 持 128 个；按全量开 z 维会让 92% 的 block 一进来就退出。
          pybind11::arg("active") = pybind11::none());
//路由融合：替掉 ``softmax -> topk -> div -> sum -> to`` 五个内核，其中 topk 对
//``[1, 512]`` 取 top-10 在本卡上走 bitonic 排序网络、单次 38 us。
COFFIN_BIND(moe_router,
            "\u8def\u7531\uff1a\u878d\u5408 softmax -> top-k -> \u5f52\u4e00\u5316\uff0c"
          "\u4e00\u6b21\u53d1\u5c04\u5199\u51fa\u4e0b\u6807\u4e0e\u6743\u91cd",
            pybind11::arg("logits"), pybind11::arg("out_idx"), pybind11::arg("out_w"),
            pybind11::arg("top_k"), pybind11::arg("normalize"));
