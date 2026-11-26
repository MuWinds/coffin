// @file moe_grouped_i8_device.inc
// @brief 分组 int8 投影的设备代码：nk 主体核、分片归约、三种收尾，加它们共用的几何常量。
//
// Module: coffin/backends/corex/kernels。被两条前端各包含一次：``moe_grouped_i8.cu``
// （torch 扩展）与 ``abi/moe_grouped_i8_abi.cu``（``extern "C"`` 出口）。同一份设备代码。
//
// 内容自成一个匿名命名空间（常量与内核都在里面），所以两条前端各自的 ``__device_stub__``
// 都是内部链接——两个库都进同一个进程时不会互相覆盖。
//
// Constraints:
//   * 权重只有 nk 一种布局（``[E, N, K]``）；``[E, K, N]`` 的主体核已删除。
//   * ``per``（每片的 k 长度）必须是 ``VB`` 的倍数，否则 16 字节的读会跨进下一列。
//   * 分片只对 nk 的 4 字节档有意义；nk 的宽步档按单趟跑（``span == 1``），不需要分片缓冲。

#ifndef COFFIN_MOE_GROUPED_I8_DEVICE_INC
#define COFFIN_MOE_GROUPED_I8_DEVICE_INC

#include "corex_scalar.h"

#include <cstdint>
#include <cuda_runtime.h>

namespace {


//: 沿输出列的线程数，也就是一个块覆盖的列宽。
//:
//: 128 让「固定 k、连续 128 列」正好是一个 128 字节的整事务。取 256 时列块数从 8 降到
//: 4（门/升栈的 N=1024），块数不足会先掉带宽 —— 这个形状上块数比每块的线程数值钱。
constexpr int I8_COLS = 128;

//: K 方向分片上限。再大只是让每片短于一个 dp4a 展开块的宽度。
constexpr int I8_MAX_SLICES = 8;

//: 分片后每片至少保留的 k 长度。
//:
//: 128 是实测的：``down``（K=512、N=2048）在 256 的下限下只能切 2 片，8 行 x 16 个列块
//: 才 256 个块，实测 146.7 GB/s；放到 128 后切 4 片、512 个块。再往下切（64）每片只剩
//: 4 个 16-k 展开块，展开收益被尾巴吃掉。
constexpr int I8_MIN_PER = 128;

//: nk 布局下每个块里的 warp 数。一个 warp 认领一个输出列，K 在这一列内连续，所以
//: 32 条通道每步读到的 128 字节是**一整段连续内存**，正好一个整事务。
//:
//: 4 与 kn 路径的 128 线程对齐：块内 4 条独立的流，块数按 ``N / 4`` 算。``gate_up``
//: （N=1024、K=2048）在 EP4 的 R8 上是 256 个列块 x 16 个有效行 = 4096 个块，远多于这张
//: 卡的 16 个 SM，所以 nk 路径不需要再靠切 K 换并行度。
constexpr int I8_NK_WARPS = 8;

//: 浅 K 时抬到 8 个 warp。K 小于这个阈值时一个列只有不到 8 个 128 字节步，
//: 「块调度 + warp 归约的 5 次 shfl」的固定成本已经和流本身同量级，于是让一个块多认领
//: 几列、少开一倍块。``down``（K=512、N=2048）就是这一档。
constexpr int I8_NK_SHALLOW_K = 1024;
constexpr int I8_NK_WARPS_SHALLOW = 8;

//: 浅 K 档一条 warp 认领的列数。核里按 ``SUB = 32 / WCOL`` 切通道组，所以这同时决定了
//: 每条 lane 组管一列的几条通道、以及归约几级：``4`` → SUB=8、**3 级**（4/2/1），四列并行。
//
//: 归约层数是浅 K 的胜负手：一列的 K 被切成 SUB 段，算完必须 SUB 路求和；K=512 时每 lane
//: 只有 4 个 dp4a，5 级归约（SUB=32）几乎和乘加一样贵，实测 143 GB/s、加到 4 列并行后
//: 249 GB/s。8 列（SUB=4）曾试过，掉到 222——每组通道太少，一步覆盖的连续段变短。
//: 所以 4 是这条轴的最优点，不要再往上加。
// 浅 K 上一列只放得下一两轮宽步（``down`` 的 K=512 正好一轮），单列填不满访存延迟，
// 所以让一个 warp 认领多列并行。**一列的 512 字节是一整段连续内存**，多认领一列就是
// 多一条独立的长突发在飞，不改 LSU 的指令形态——这与「把通道切成 SUB 组、每组一列」
// （已测退化 67.3 → 87.6 us）是两回事。
//
// 从 4 抬到 8 实测**更差**（24.28 → 25.34 ms，队列 063_nk5）：一个 warp 认领 8 列时
// 块数从 64 掉到 32，行 × 片变成 512 个块，每块的启动与共享区加载摊不薄，而
// 多出来的在飞请求没换来带宽。回到 4。
constexpr int I8_NK_WCOL_SHALLOW = 4;

//: nk 核的行槽数（``grid.y`` 的上限）。块不再与行一一对应，每块按这个步长跨行；
//: 取 8 是因为 64 行 × 列块之后仍有一万多条 warp 摊到 16 个 SM，再少并行度开始掉。
//: 由此 ``moe_grouped_i8`` 的 ``rows_valid`` 参数实际已不再参与网格决策（保留只为兼容）。
constexpr int I8_NK_ROW_SLOTS = 8;

//: 每通道每步的字节数切到 16 的 K 上限。16 字节一步把一步覆盖的区间从 128 抬到 512，
//: 浅 K 下循环次数因此除以 4；K 一深（>= 1024）就回到 4 字节——一个 warp 一轮 128 字节
//: 是这张卡的最佳点，更宽的每通道加载会把一次事务拆成两次。
//:
//: 这个上限是实测定的，不是估的：`005_moelab2` 把条件反过来（``K >= 512`` 走 16 字节步、
//: 无上限）之后，gate_up（K=2048）从 274 GB/s 掉到 **187 GB/s**，down（K=512）没动。
//: 所以 128 字节步的默认保持不动；16 字节步只留给「K 放不下几轮 512 字节步」的浅档。
constexpr int I8_NK_WIDE_K = 1024;

//: K 是 16 的倍数时，16 字节一步读到的地址才落在同一个列内（列基址是 ``col * K``，
//: 不是 16 的倍数就会让 ``int4`` 跨进下一列）。
constexpr int I8_NK_WIDE_BYTES = 16;


// @brief 在 ``offsets`` 里二分出某一行属于哪个专家。
//
// ``offsets`` 单调不减，空专家表现为相邻两项相等；取「最后一个 ``offsets[i] <= row``
// 的 i」会自动跨过相等的项，因此空专家不会被选中。
//
// @param offsets ``[experts + 1]`` 前缀和，int32。
// @param experts 专家数。
// @param row 行号。
// @return 该行所属的专家下标。
__device__ __forceinline__ int i8_owner(const int* __restrict__ offsets,
                                        int experts, int row) {
    int lo = 0, hi = experts;
    while (lo + 1 < hi) {
        const int mid = (lo + hi) >> 1;
        if (offsets[mid] <= row) lo = mid; else hi = mid;
    }
    return lo;
}


// @brief nk 布局的主体核：``[E, N, K]`` 的专家栈，一个 warp 认领一个输出列。
//
// 与上面那个 kn 主体核只差权重的寻址方式，而这一点决定了整条链的带宽：
// ``[E, K, N]`` 里相邻 4 个 k 隔着 N 个字节，``__dp4a`` 要的那个 32 位字只能靠四次单字节
// 读再移位拼起来（``i8_dword``，4 次加载 + 6 条位运算）；``[E, N, K]`` 里它们是连续的
// 4 个字节，一次 32 位加载就是那个字，而且 32 条通道一轮读到的 128 字节是一整段连续
// 内存，正好一个整事务。
//
// 这条规律在稠密投影上已经量过：同一份字节量、同样的 32 条通道分组，nk 是 kn 的
// 1.02-1.63 倍（``gemv.cu`` 的 nk 段、``docs`` 里 2026-11-13 那份记录）。MoE 这边还多一层
// ——kn 的四次单字节读每次只搬 32 字节（一个 lane 一字节），而 nk 一次搬 128 字节。
// EP4 的 R8 实测 kn 是 89 GB/s（每层每卡 16 个有效行 x 2 MB / 360 us），远低于这张卡
// 600 GB/s 的连续读。
//
// @tparam XT 输出 dtype；@tparam VB 每通道每步的字节数（4 或 16）。
// @param wn   专家栈 ``[E, N, K]`` 连续 int8；第 ``e`` 个专家的第 ``n`` 列在第 ``n * K``。
// @param per  每片的 k 长度，必须是 ``VB`` 的倍数（否则 16 字节的读会跨进下一列）。
// 其余参数与 kn 主体核一致。
template <typename XT, int VB, int WCOL, int KCOMP = 0>
__global__ void moe_grouped_i8_nk_kernel(const int8_t* __restrict__ a,
                                         const int8_t* __restrict__ wn,
                                         const float* __restrict__ ws,
                                         const float* __restrict__ xs,
                                         const int* __restrict__ offsets,
                                         int32_t* __restrict__ part,
                                         int* __restrict__ mcnt,
                                         const void* __restrict__ rw,
                                         const int* __restrict__ slot_of_pos,
                                         int rw16,
                                         XT* __restrict__ out,
                                         int K, int N, int experts, int tokens,
                                         int slices, int per) {
    const int row = (int)blockIdx.y;
    // 与 kn 主体核同一条：不属于本 rank 的行整块退出（EP4 的 R8 上这是四分之三的块）。
    //
    // 试过把块内改成跨行循环（``grid.y`` 只给 8 个行槽、死行用 ``continue`` 跳过）：
    // gate_up 81.0 -> 77.3 是赚的，但 down 67.8 -> 73.9 是亏的，合并 148.8 -> 151.2 反而变差。
    // 原因是 ``row`` 从块常量变成循环变量后，``a + row*K`` / ``out[row*N + col]`` 这些
    // 64 位地址运算不再提到循环外；down 每 warp 只有 16 个 dp4a 却要发 582 条指令，纯指令
    // 受限，这笔重算直接吃掉 7.8 us，而 gate_up 每行 32 倍运算量摊得平。
    // 结论：这个卡上按行切块比跨行复用更划算，除非把地址运算也提到循环外（那要重排整个核）。
    if (row >= offsets[experts]) return;

    __shared__ int s_expert;
    extern __shared__ __align__(16) unsigned char s_raw[];
    int8_t* s_x = reinterpret_cast<int8_t*>(s_raw);
    if (threadIdx.x == 0) s_expert = i8_owner(offsets, experts, row);
    const int8_t* arow = a + (size_t)row * K;
    for (int i = (int)threadIdx.x; i < K; i += (int)blockDim.x) s_x[i] = arow[i];
    __syncthreads();

    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    // 一个 warp 认领 ``WCOL`` 个相邻列（``WCOL = 1`` 就是原来的一 warp 一列）。
    // 主机端的 ``cols`` 必须按 ``warps * WCOL`` 算，两边是同一套几何。
    //
    // 试过把通道切成 ``SUB = 32 / WCOL`` 组、每组认领一列（4 列并行、归约 5 级降到 3 级）：
    // down 从 67.3 us/249 GB/s 退到 **87.6 us/191.6 GB/s**。原因是那条指令一次要读 4 个
    // 相距 512 字节的 128 字节段，每条 load 变成 4 个 wavefront，LSU 上的指令占用翻四倍，
    // 而有用字节不变。也就是说这个卡上「一条指令一个连续 128 字节段」比「少两级归约」值钱，
    // 归约不是瓶颈，别再往这条路上改。
    const int colbase = ((int)blockIdx.x * (int)(blockDim.x >> 5) + warp) * WCOL;
    // 越界列**不能提前 return**：末尾那段末块收尾里有 ``__syncthreads()``，而
    // ``colbase`` 是按 warp 算的，一个 warp 退出会让其余 warp 等一个已经不在的线程
    // （设备侧死锁，没有 traceback）。把 ``nc`` 夹到 0 之后，下面每个 ``for c`` 都在
    // c = 0 处 ``break``，一行权重也不读。
    const int nc = colbase >= N ? 0 : ((colbase + WCOL <= N) ? WCOL : (N - colbase));

    const int expert = s_expert;
    const int8_t* ebase = wn + (size_t)expert * (size_t)N * (size_t)K;

    const int k0 = (int)blockIdx.z * per;
    const int k1 = min(K, k0 + per);

    // 每列各自 ``VB / 4`` 条独立链，一个 warp 一共压着 ``WCOL * ACC`` 条访存在飞。
    // 浅 K 上一列只放得下一两轮宽步，单列根本填不满延迟（``down`` 的 K=512 就是一次
    // 512 字节步就结束），只能靠多列并行把这些请求压满。
    constexpr int ACC = VB / 4;
    constexpr int STEPK = 32 * VB;
    int s[WCOL][ACC];
#pragma unroll
    for (int c = 0; c < WCOL; ++c) {
#pragma unroll
        for (int i = 0; i < ACC; ++i) s[c][i] = 0;
    }

    // ixprof 的账：K=512 时一个 warp 一生只做 16 个 dp4a，却发射了约 582 条指令——
    // 每 36 条换 1 次有效乘加。开销不在归约、不在带宽（加载管道跑到 352 GB/s 仍有余量），
    // 而在「K 是运行期量」逼出来的循环、``min``、``slices`` 判定和两段尾部兜底。
    // 所以给 K 一个编译期版本：K=512 且 span=1 时 STEPK=512 正好一步走完，整个循环、
    // 边界与尾部全部消失，列基址在展开前算好，展开体内只剩「基址 + 立即数」。
    if constexpr (KCOMP > 0) {
        static_assert(KCOMP % STEPK == 0, "KCOMP 必须是 STEPK 的整数倍");
        constexpr int STEPS = KCOMP / STEPK;
        const int8_t* wcol[WCOL];
#pragma unroll
        for (int c = 0; c < WCOL; ++c) {
            wcol[c] = ebase + (size_t)(colbase + c) * (size_t)KCOMP;
        }
#pragma unroll
        for (int t = 0; t < STEPS; ++t) {
            const int aoff = t * STEPK + lane * VB;
            const int* aq = reinterpret_cast<const int*>(s_x + aoff);
#pragma unroll
            for (int c = 0; c < WCOL; ++c) {
                if (c >= nc) break;
                const unsigned* wq = reinterpret_cast<const unsigned*>(wcol[c] + aoff);
#pragma unroll
                for (int i = 0; i < ACC; ++i) s[c][i] = __dp4a((int)wq[i], aq[i], s[c][i]);
            }
        }
    } else {
        const int step = STEPK;
        int k = k0;
        for (; k + step <= k1; k += step) {
            const int* aq = reinterpret_cast<const int*>(s_x + k + lane * VB);
#pragma unroll
            for (int c = 0; c < WCOL; ++c) {
                if (c >= nc) break;
                const unsigned* wq = reinterpret_cast<const unsigned*>(
                    ebase + (size_t)(colbase + c) * (size_t)K + k + lane * VB);
#pragma unroll
                for (int i = 0; i < ACC; ++i) s[c][i] = __dp4a((int)wq[i], aq[i], s[c][i]);
            }
        }
        // 整 128 字节的步：16 字节那条宽路径退出后，剩下的整步在这里按 4 字节走。
        for (; k + 128 <= k1; k += 128) {
#pragma unroll
            for (int c = 0; c < WCOL; ++c) {
                if (c >= nc) break;
                const unsigned* wq = reinterpret_cast<const unsigned*>(
                    ebase + (size_t)(colbase + c) * (size_t)K + k + lane * 4);
                s[c][0] = __dp4a((int)wq[0],
                                 *reinterpret_cast<const int*>(s_x + k + lane * 4), s[c][0]);
            }
        }
        // 最后不足 128 字节：每条通道最多认领一个 4 字节块，尾部不足 4 个的由紧随其后那条
        // 通道按标量兜。**不能**补齐凑整 —— 那会把下一列的字节算进这一列。
        const int rem = k1 - k;
        const int quads = rem / 4;
#pragma unroll
        for (int c = 0; c < WCOL; ++c) {
            if (c >= nc) break;
            const int8_t* wrow = ebase + (size_t)(colbase + c) * (size_t)K;
            if (lane < quads) {
                const int j = k + lane * 4;
                s[c][0] = __dp4a((int)*reinterpret_cast<const unsigned*>(wrow + j),
                                 *reinterpret_cast<const int*>(s_x + j), s[c][0]);
            } else if (lane == quads) {
                for (int j = k + lane * 4; j < k1; ++j) s[c][0] += (int)wrow[j] * (int)s_x[j];
            }
        }
    }

#pragma unroll
    for (int c = 0; c < WCOL; ++c) {
        if (c >= nc) break;
        int total = s[c][0];
#pragma unroll
        for (int i = 1; i < ACC; ++i) total += s[c][i];
        // 32 条通道各持一段 K，树形归约后由 lane 0 落盘。与 kn 的长链累加不逐位相同，
        // 相对差在 fp32 重结合的 1e-3 量级——验收按参考实现与 greedy ids。
        for (int o = 16; o > 0; o >>= 1) total += __shfl_down_sync(0xffffffffu, total, o);
        if (lane == 0) {
            const int col = colbase + c;
            // ``mcnt`` 非空时一律写 ``part``，**即使只有一片**：收尾要靠它做跨行加权，
            // 而 ``out`` 是 ``[1, N]``（逐行落盘那一支只服务 ``tokens > 1`` 的旧形态）。
            // 这条闸一开，浅 K 才敢用 ``span = 1``——那正是 down（K=512）能回到
            // 「一轮 16 字节、32 条通道哄满一个 512 字节整事务」的前提。
            if (slices == 1 && mcnt == nullptr) {
                out[(size_t)row * N + col] = CxScalar<XT>::from_float(
                    (float)total * xs[row] * ws[(size_t)expert * N + col]);
            } else {
                part[((size_t)blockIdx.z * (size_t)tokens + (size_t)row) * (size_t)N + col]
                    = total;
            }
        }
    }

    // 末块收尾：单 token 的「跨行加权累加」也在这里做掉，于是 mcnt 非空时本核同时替代
    // ``moe_grouped_i8_merge_kernel``（40 层 40 次发射）。这是把 kn 主体核末尾那一段
    // 搬到 nk 形态：kn 里一个线程一列，这里一列归一个 warp，所以由 ``lane == 0`` 收。
    //
    // 记账口径与 kn 那段逐字相同：阈值是**有效行数 x 片数**（死行在开头整块退出、不计数），
    // 计数器按列块、末块就地算完再自复位，零自旋。
    if (mcnt != nullptr) {
        __threadfence();
        __syncthreads();
        __shared__ int mlast;
        if (threadIdx.x == 0) {
            const int valid = offsets[experts];
            const int need = valid > 0 ? valid * (int)gridDim.z : 1;
            mlast = (atomicAdd(&mcnt[blockIdx.x], 1) == need - 1);
        }
        __syncthreads();
        if (!mlast) return;
        if (threadIdx.x == 0) mcnt[blockIdx.x] = 0;
        __threadfence();
        // 一列归一个 warp，**列内再把行摊到 lanes 上**：rows 一般就是 top_k = 8，于是
        // 「8 行 × slices 次读」从一个线程的串行链变成每个 lane 一次读，再做一次 5 级
        // warp 归约。原来 lane 0 串行读 16 个 int32，末块收尾的延迟全压在那一个线程上。
        const int valid = offsets[experts];
        for (int c = 0; c < WCOL; ++c) {
            const int col = colbase + c;
            if (col >= N) break;
            float acc = 0.f;
            for (int r = lane; r < (int)gridDim.y; r += 32) {
                if (r >= valid) continue;
                const int er = i8_owner(offsets, experts, r);
                int tot = 0;
                for (int s = 0; s < slices; ++s) {
                    tot += part[((size_t)s * (size_t)tokens + (size_t)r) * (size_t)N + col];
                }
                // ``slot_of_pos`` 非空时路由权重按**槽位**给出（``merge_tokens`` 那支），
                // 为空时它直接就是逐行的（``moe_grouped_i8_merge`` 那支）。
                const int sw = (slot_of_pos != nullptr) ? slot_of_pos[r] : r;
                const float w = rw16 ? CxScalar<uint16_t>::to_float(
                                           reinterpret_cast<const uint16_t*>(rw)[sw])
                                     : reinterpret_cast<const float*>(rw)[sw];
                acc += (float)tot * xs[r] * ws[(size_t)er * N + col] * w;
            }
            for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
            if (lane == 0) out[(size_t)0 * N + col] = CxScalar<XT>::from_float(acc);
        }
    }
}

// @brief 打一行「这个内核实例的寄存器数 / 每 SM 可驻留块数」，``COFFIN_KATTR=1`` 门控。
//
// 为什么需要它：nk 收尾从「lane 0 串行读 16 个 int32」改成「行摊到 lanes、warp 归约」后，
// down 自己的时间轴从 56.15 掉到 47.24 us（−16%），整体墙钟却两次都慢了 0.3 ms
// （24.28/24.39 → 24.56/24.75）。唯一说得通的机制是**寄存器**：收尾多出来的索引与归约把
// ``numRegs`` 顶过一档，主循环的可驻留块数跟着掉，省下的收尾延迟被主循环变慢吃掉。
// 这一行就是拿这个数来判断的——直接问驱动，不靠墙钟猜。
//
// 长期留着：以后再动内核形状，先看这一行有没有掉档。
template <typename XT, int VB, int WCOL, int KCOMP>
void i8_nk_report(const char* tag, int threads) {
    static bool once = false;
    if (once || std::getenv("COFFIN_KATTR") == nullptr) return;
    once = true;
    const void* fn = (const void*)&moe_grouped_i8_nk_kernel<XT, VB, WCOL, KCOMP>;
    int blocks = -1;
    cudaFuncAttributes at{};
    if (cudaFuncGetAttributes(&at, fn) == cudaSuccess) {
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, fn, threads, 0)
                != cudaSuccess) {
            blocks = -2;
        }
    }
    std::printf("[kattr] %s regs=%d smem=%zu maxthr=%d threads=%d occ=%d blk/sm\n",
                tag, (int)at.numRegs, (size_t)at.sharedSizeBytes,
                (int)at.maxThreadsPerBlock, threads, blocks);
    std::fflush(stdout);
}

// @brief 把 ``slices`` 片整数部分和加起来，乘两个 scale，窄化一次。
//
// 整数部分和可以直接相加，所以这里不需要知道每片覆盖了哪几个 k，也不需要 fp32
// 缓冲：片数再多，累加都是精确的。
//
// @tparam XT 输出 dtype。
// @param part    ``[slices, tokens, N]`` int32 部分和。
// @param ws      专家栈 scale，``[E, N]`` fp32。
// @param xs      逐 token 激活 scale，``[tokens]`` fp32。
// @param offsets ``[E+1]`` 前缀和，int32。
// @param out     ``[tokens, N]`` 目的地。
// @param N, experts, tokens, slices 几何。
template <typename XT>
__global__ void moe_grouped_i8_reduce_kernel(const int32_t* __restrict__ part,
                                             const float* __restrict__ ws,
                                             const float* __restrict__ xs,
                                             const int* __restrict__ offsets,
                                             XT* __restrict__ out,
                                             int N, int experts, int tokens,
                                             int slices) {
    const int row = (int)blockIdx.y;
    const int col = (int)blockIdx.x * I8_COLS + (int)threadIdx.x;
    if (col >= N) return;
    const int expert = i8_owner(offsets, experts, row);
    int total = 0;
    for (int s = 0; s < slices; ++s) {
        total += part[((size_t)s * (size_t)tokens + (size_t)row) * (size_t)N + col];
    }
    out[(size_t)row * N + col] = CxScalar<XT>::from_float(
        (float)total * xs[row] * ws[(size_t)expert * N + col]);
}

// @brief 归并形态的收尾核：把 ``rows`` 行按路由权重加权后累加成 **一行**。
//
// decode 时一个 token 会激活 top_k 个专家，它们在 ``part`` 里是 top_k 行。调用方真正
// 要的是这些行按路由权重加权后的**和**，所以加权与求和可以并进这一趟收尾，而不是
// 先写出一份 ``[top_k, N]`` 激活再在外面做 `mul` + `index_add_` + `sum`。
//
// 累加用 fp32：每行已经乘过自己的激活 scale 与权重 scale，加权和必须在这一层做，
// 否则 ``bf16`` 上累加 8 次会掉 3 个有效位。
//
// @tparam XT 输出 dtype：``uint16_t``（bf16）或 ``float``。
// @param part    ``[slices, rows, N]`` int32 部分和。
// @param ws      专家栈 scale，``[E, N]`` fp32。
// @param xs      逐行激活 scale，``[rows]`` fp32。
// @param offsets ``[E+1]`` 前缀和，int32。
// @param rw       逐行路由权重，``[rows]`` fp32。
// @param out      ``[1, N]`` 目的地。
// @param N, experts, rows, slices 几何。
template <typename XT>
__global__ void moe_grouped_i8_merge_kernel(const int32_t* __restrict__ part,
                                            const float* __restrict__ ws,
                                            const float* __restrict__ xs,
                                            const int* __restrict__ offsets,
                                            const float* __restrict__ rw,
                                            XT* __restrict__ out,
                                            int N, int experts, int rows,
                                            int slices) {
    const int col = (int)blockIdx.x * I8_COLS + (int)threadIdx.x;
    if (col >= N) return;
    float acc = 0.f;
    for (int row = 0; row < rows; ++row) {
        const int expert = i8_owner(offsets, experts, row);
        int total = 0;
        for (int s = 0; s < slices; ++s) {
            total += part[((size_t)s * (size_t)rows + (size_t)row) * (size_t)N + col];
        }
        acc += (float)total * xs[row] * ws[(size_t)expert * N + col] * rw[row];
    }
    out[col] = CxScalar<XT>::from_float(acc);
}

// @brief 收尾的行序形态：一个 token 的全部行按**收拢后的行号升序**加权累加。
//
// 与 ``moe_grouped_i8_merge_kernel`` 只差权重的来源。原来调用方要先
// ``index_select`` 把 ``[slots]`` 的路由权重摊成逐行表（一次发射加一次 4 KB 的显存往返），
// 再为专家并行算一遍 ``arange < valid`` 的掩码并相乘（四次发射）。这里权重改由
// ``slot_of_pos`` 直接寻址，尾部无效行按 ``offsets[E]`` 判掉。行序与加法次序都没变，
// 所以单流的结果与那条路逐位相同。
//
// @tparam XT 输出 dtype：``uint16_t``（bf16）或 ``float``。
// @tparam WT 路由权重的存储 dtype，与 ``logits`` 一致。
// @param part     ``[slices, rows, N]`` int32 部分和。
// @param ws       专家栈 scale，``[E, N]`` fp32。
// @param xs       逐行激活 scale，``[rows]`` fp32。
// @param offsets  ``[E+1]`` 前缀和，int32。
// @param rw       路由权重，按槽位号索引。
// @param slot_of_pos 收拢后行号 → 槽位号，int32。
// @param out      ``[1, N]`` 目的地。
// @param N, experts, rows, slices 几何。
template <typename XT, typename WT>
__global__ void moe_grouped_i8_merge_rows_kernel(const int32_t* __restrict__ part,
                                                 const float* __restrict__ ws,
                                                 const float* __restrict__ xs,
                                                 const int* __restrict__ offsets,
                                                 const WT* __restrict__ rw,
                                                 const int* __restrict__ slot_of_pos,
                                                 XT* __restrict__ out,
                                                 int N, int experts, int rows,
                                                 int slices) {
    const int col = (int)blockIdx.x * I8_COLS + (int)threadIdx.x;
    if (col >= N) return;
    // 有效行数留在设备上读：``offsets[E]`` 是分派内核刚写的，取回主机就是一次 D2H
    // 同步，而 decode 一步有 40 层——每层一次同步会把主机预取打断（qwen4exp 量到
    // 11.7%）。这也解释了外层那套 ``arange < valid`` 为什么全在设备上算。
    const int valid = offsets[experts];
    float acc = 0.f;
    for (int row = 0; row < rows; ++row) {
        // 尾部那些行不属于本 rank：分派内核没把它们排进行序（位置表清 0），主体核也
        // 按 ``offsets[E]`` 整块退出了它们，所以这里跳过与「权重乘 0」逐位等价，同时
        // 省掉外面那四次掩码发射。
        if (row >= valid) continue;
        const int expert = i8_owner(offsets, experts, row);
        int total = 0;
        for (int s = 0; s < slices; ++s) {
            total += part[((size_t)s * (size_t)rows + (size_t)row) * (size_t)N + col];
        }
        acc += (float)total * xs[row] * ws[(size_t)expert * N + col]
               * CxScalar<WT>::to_float(rw[slot_of_pos[row]]);
    }
    out[col] = CxScalar<XT>::from_float(acc);
}

// @brief 收尾的槽位序形态：每个 token 只累加自己的 top_k 行，写到 ``out[t]``。
//
// 多流解码原来走 ``hidden * w`` → ``buf.zero_()`` → ``buf.index_add_(0, token_of_pos)``
// 三步，外加 ``token_of_pos.long()`` 与 ``slot_of_pos.long()`` 两次 dtype 转换；一步 8 行时
// 这一串是 40 层 × 8 次发射，而它搬的数据只有 ``[8, 2048]`` bf16。这里按槽位号找出每个
// token 的行，一次累加写完。
//
// 累加用槽位序而不是行序：行序由分派的原子分配决定，按它累加等于让加法次序跟着到达
// 顺序变；槽位序是固定的，同一输入两次运行逐位一致。这也是这条路径**不用**原子累加的
// 原因——``index_add_`` 在 bf16 上的结果是不可复现的。
//
// @tparam XT 输出 dtype；@tparam WT 路由权重 dtype。
// @param part     ``[slices, rows, N]`` int32 部分和。
// @param ws       专家栈 scale，``[E, N]`` fp32。
// @param xs       逐行激活 scale，``[rows]`` fp32。
// @param offsets  ``[E+1]`` 前缀和，int32。
// @param rw       路由权重，按槽位号索引。
// @param slot_pos 槽位 → 收拢后行号，int32，丢弃的为 -1。
// @param out      ``[tokens, N]`` 目的地。
// @param N, experts, rows, slices 几何；``top_k`` 是每 token 的行数。
template <typename XT, typename WT>
__global__ void moe_grouped_i8_merge_slots_kernel(const int32_t* __restrict__ part,
                                                  const float* __restrict__ ws,
                                                  const float* __restrict__ xs,
                                                  const int* __restrict__ offsets,
                                                  const WT* __restrict__ rw,
                                                  const int* __restrict__ slot_pos,
                                                  XT* __restrict__ out,
                                                  int N, int experts, int rows,
                                                  int slices, int top_k) {
    const int col = (int)blockIdx.x * I8_COLS + (int)threadIdx.x;
    if (col >= N) return;
    const int token = (int)blockIdx.y;
    const int base = token * top_k;
    const int valid = offsets[experts];
    float acc = 0.f;
    for (int j = 0; j < top_k; ++j) {
        const int row = slot_pos[base + j];
        // 路由到别的 rank 的槽位在这里被丢掉（``-1``），与「权重乘 0」等价。
        if (row < 0 || row >= valid) continue;
        const int expert = i8_owner(offsets, experts, row);
        int total = 0;
        for (int s = 0; s < slices; ++s) {
            total += part[((size_t)s * (size_t)rows + (size_t)row) * (size_t)N + col];
        }
        acc += (float)total * xs[row] * ws[(size_t)expert * N + col]
               * CxScalar<WT>::to_float(rw[base + j]);
    }
    out[(size_t)token * N + col] = CxScalar<XT>::from_float(acc);
}


}  // namespace

#endif  // COFFIN_MOE_GROUPED_I8_DEVICE_INC
