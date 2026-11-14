// @file moe_grouped_i8.cu
// @brief 稀疏专家的分组 int8 投影：一次启动吃完整栈，行按 ``offsets`` 分段。
//
// Module: coffin/backends/corex/kernels。绑定成 ``moe_grouped_i8``，由
// ``backends/corex/__init__.py::moe_grouped_i8`` 调用；``models/qwen35/moe.py`` 在
// decode 一步（激活行数 == ``top_k``）走这里，其余行数仍走逐专家的 ``bops.linear``。
//
// 为什么需要它：int8 专家栈原来走逐专家 ``bops.linear``，一层 8 个专家 x 2 次投影 =
// 16 次发射，外加每个专家的 silu/mul/累加。单层实测 ``mlp`` 1.82 ms，其中专家部分
// 1.40 ms，而一层的专家权重只有 8 x 3 MB = 24 MB —— 按 400 GB/s 应该是 60 us。慢
// 在两处：一次 ``gemv_kn_i8`` 只覆盖一个专家的 2 MB（并行度与块数都被一个专家限住），
// 以及 16 次发射各自的地板成本。
//
// 这里把「按专家分段」交给内核自己：``blockIdx.y`` 是一个激活行，这一行属于哪个专家
// 由 ``offsets`` 二分得出；``blockIdx.z`` 是 K 方向分片。于是整层只要 2 次发射
// （主内核 + 分片归约），块数是 ``行数 x 列块 x 分片``，与专家数无关。
//
// 算术：``out[t, n] = a_scale[t] * w_scale[e, n] * sum_k a_q[t, k] * w[e, k, n]``，
// 其中 ``e`` 由 ``t`` 落在 ``[offsets[e], offsets[e+1])`` 决定。权重有两种可选布局：
// ``[E, K, N]``（K 轴在外、N 连续，与 ``gemv_kn_i8`` 的 ``kn`` 一致）和 ``[E, N, K]``
// （nk，一列一行、K 连续）。前者一个 warp 在固定 k 上读到的 128 字节正好是一个整事务，
// 但 ``__dp4a`` 要的 32 位字得靠四次单字节读拼；后者每个通道一次 32 位字、整轮 128
// 字节仍是一整段，是 EP4 的 R8 上快得多的那一支（见 ``moe_grouped_i8_nk_kernel``）。
//
// Constraints:
//   * 整数和可加是这条路径能用 int32 部分和的前提：scale 是逐 (专家, 输出列) 的、与
//     k 无关，所以分片可以先把整数和加起来、最后一次性乘两个 scale。W4A8 那边做不到
//     （它的 scale 逐 (组, 列)，每片必须各乘一次），这也是这里不需要 part 存 fp32 的原因。
//   * ``__dp4a`` 的原子粒度是 4 个 k。K 不是 4 的倍数时尾部走标量；**不能**用补齐的方式
//     凑整，那会把下一行的字节算进这一行，得到一个看着合理的错数。
//   * 一行一个块，块数就是 ``tokens``。多行时同一专家的权重会被多读一遍，所以调用方
//     只在 decode 那一档进来；内核本身对任意行数都成立，不需要额外的形状校验。
//   * 行号 ``>= offsets[E]`` 的块整块退出（EP4 下不属于本 rank 的行）。调用方本来就按
//     ``offsets[E]`` 判掉这些行，所以内核不写它们不改变结果；``nk`` 那一支与 kn 同一口径。
//   * 激活先由 ``quant_act_i8`` 按行量化（absmax/127，下限 1e-5），与预填充路径同一套
//     约定，所以两处的量化误差口径一致。
//   * 不用 CUDA graph、不用 cp.async：两者在这张卡上都能过编译，然后静默挂死。
//   * 全程不用 fp64

#include "corex_bindings.h"
#include "corex_ops.h"
#include "corex_scalar.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

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

// @brief 把四个相邻 k 的 int8 权重拼成 ``__dp4a`` 要的一个字。
//
// 权重是 ``[E, K, N]``：四个相邻 k 在同一列上隔着 N 个字节，所以拿不到一次 32 位加载，
// 只能四次单字节读再拼。``__dp4a`` 按字节把操作数当 int8 解释，因此这里只做截断、
// 不做符号处理 —— 加一次符号扩展会把 -1 变成 255。
//
// @param w 专家权重的基址（已偏移到该专家）。
// @param col 输出列。
// @param N 行宽，也就是相邻 k 之间的字节跨度。
// @param k 四个相邻 k 里的第一个。
// @return 一个 32 位字，四个字节依次是 k、k+1、k+2、k+3 的量化值。
__device__ __forceinline__ unsigned i8_dword(const int8_t* __restrict__ w, int col,
                                             int N, int k) {
    const unsigned b0 = (unsigned)(unsigned char)w[(size_t)(k + 0) * N + col];
    const unsigned b1 = (unsigned)(unsigned char)w[(size_t)(k + 1) * N + col];
    const unsigned b2 = (unsigned)(unsigned char)w[(size_t)(k + 2) * N + col];
    const unsigned b3 = (unsigned)(unsigned char)w[(size_t)(k + 3) * N + col];
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

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

// @brief 分组 int8 投影的主体：一行激活对一栈专家里属于它的那一个。
//
// @tparam XT 输出 dtype：``uint16_t``（bf16）或 ``float``。
// @param a       int8 激活，``[tokens, K]`` 连续，已由 ``quant_act_i8`` 按行量化。
// @param w       专家栈，``[E, K, N]`` 连续 int8。
// @param ws      专家栈 scale，``[E, N]`` 连续 fp32。
// @param xs      逐 token 激活 scale，``[tokens]`` fp32。
// @param offsets ``[E+1]`` 前缀和，int32。
// @param part    分片部分和，``[slices, tokens, N]`` int32；``slices == 1`` 时不用。
// @param out     ``[tokens, N]`` 目的地，``slices == 1`` 时直接写这里。
// @param K, N, experts, tokens, slices 几何，``per`` 是每片的 k 长度。
template <typename XT>
__global__ void moe_grouped_i8_kernel(const int8_t* __restrict__ a,
                                      const int8_t* __restrict__ w,
                                      const float* __restrict__ ws,
                                      const float* __restrict__ xs,
                                      const int* __restrict__ offsets,
                                      int32_t* __restrict__ part,
                                      int* __restrict__ cnt,
                                      int* __restrict__ mcnt,
                                      const void* __restrict__ rw,
                                      const int* __restrict__ slot_of_pos,
                                      int rw16,
                                      XT* __restrict__ out,
                                      int K, int N, int experts, int tokens,
                                      int slices, int per) {
    // 本片负责的 k 区间由 ``blockIdx.z`` 直接算出，不从 host 传：四片各收到一份
    // 「全区间的 k0/k1」时，每片都会把整个 K 算一遍并各写一份部分和，归约把同一个
    // 数加 ``slices`` 遍 —— 输出翻整数倍，而且只在多片时才错。
    const int k0 = (int)blockIdx.z * per;
    const int k1 = min(K, k0 + per);
    const int row = (int)blockIdx.y;
    // 不属于本 rank 的行整块退出。EP4 的 R8 上 ``rows`` 是 64 而 ``offsets[E]`` 只有 16，
    // 也就是四分之三的块白跑（它们的行全落到最后一个专家上，读的是同一份已经进 L2 的
    // 权重，所以以前不掉带宽、但白占块与指令）。``row`` 是块级的，这条早退在块内一致，
    // 可以放在任何 ``__syncthreads()`` 之前。
    if (row >= offsets[experts]) return;

    // 行属于哪个专家只让 thread 0 查，其余线程同时把激活行搬进共享内存：两件事互不
    // 依赖，串起来做会白付一次「八个依赖读」的延迟。激活行只有 K 字节（2048），
    // 放共享内存后每个 k 步的读都是一次广播，而不是 128 个线程各读一遍全局。
    __shared__ int s_expert;
    extern __shared__ __align__(16) unsigned char s_raw[];
    int8_t* s_x = reinterpret_cast<int8_t*>(s_raw);
    if (threadIdx.x == 0) s_expert = i8_owner(offsets, experts, row);
    const int8_t* arow = a + (size_t)row * K;
    for (int i = (int)threadIdx.x; i < K; i += (int)blockDim.x) s_x[i] = arow[i];
    __syncthreads();

    // 列号越界的线程在屏障**之后**才退：放在屏障之前的话，末尾那个列块里会有一半线程
    // 跳过 ``__syncthreads()``，那是未定义行为（现在只有 N 是 128 的倍数才不出事）。
    //
    // 有了末块收尾（见函数末尾）之后，越界线程也不能提前 ``return``——它们必须跟着走到
    // 那两道屏障。所以这里只记一个谓词，把计算与写回都圈进 ``if (ok)``。
    const int col = (int)blockIdx.x * I8_COLS + (int)threadIdx.x;
    const bool ok = col < N;

    const int expert = s_expert;
    const int8_t* wexp = w + (size_t)expert * (size_t)K * (size_t)N;

    if (ok) {
        // 四条独立累加链：每个 k 步只有一条 dp4a 时，16 次乘加串成一条依赖链，
        // 访存延迟完全暴露。四条链让 16 个权重字节同时在飞（与 ``moe_grouped.cu``
        // 里 1 行档的 8 路展开是同一条结论，这里按 dp4a 的 4 字节粒度展开）。
        int s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        int k = k0;
        for (; k + 16 <= k1; k += 16) {
            const int a0 = *reinterpret_cast<const int*>(s_x + k);
            const int a1 = *reinterpret_cast<const int*>(s_x + k + 4);
            const int a2 = *reinterpret_cast<const int*>(s_x + k + 8);
            const int a3 = *reinterpret_cast<const int*>(s_x + k + 12);
            const unsigned q0 = i8_dword(wexp, col, N, k);
            const unsigned q1 = i8_dword(wexp, col, N, k + 4);
            const unsigned q2 = i8_dword(wexp, col, N, k + 8);
            const unsigned q3 = i8_dword(wexp, col, N, k + 12);
            s0 = __dp4a((int)q0, a0, s0);
            s1 = __dp4a((int)q1, a1, s1);
            s2 = __dp4a((int)q2, a2, s2);
            s3 = __dp4a((int)q3, a3, s3);
        }
        for (; k + 4 <= k1; k += 4) {
            s0 = __dp4a((int)i8_dword(wexp, col, N, k),
                        *reinterpret_cast<const int*>(s_x + k), s0);
        }
        for (; k < k1; ++k) {
            s0 += (int)wexp[(size_t)k * N + col] * (int)s_x[k];
        }

        const int total = (s0 + s1) + (s2 + s3);
        if (slices == 1) {
            // 两个 scale 在同一个 fp32 表达式里乘完再窄化一次：分两次窄化多一次舍入，
            // 而这个结果要穿过 40 层。
            out[(size_t)row * N + col] = CxScalar<XT>::from_float(
                (float)total * xs[row] * ws[(size_t)expert * N + col]);
        } else {
            part[((size_t)blockIdx.z * (size_t)tokens + (size_t)row) * (size_t)N + col] = total;
        }
    }

    // 末块收尾：``cnt != nullptr`` 时本核不再需要外面那趟 ``moe_grouped_i8_reduce``。
    //
    // 每片写完 ``part`` 后各来一次 ``atomicAdd``，拿到 ``slices - 1`` 的那个列块就是本
    // (row, col) 组最后到达的，它就地读回 ``slices`` 个部分和、乘两个 scale、写 ``out``，
    // 再把计数器复位。**零自旋**：末块直接干活，不等别人，所以块数不受驻留约束。
    // 与 GEMV 那两支同一个套路，约束也相同（见 ``gemv.cu`` 里
    // ``gemv_kn_i8_split_kernel`` 的注释）。
    //
    // ``cnt == nullptr``（``moe_grouped_i8_merge_tokens`` 那一支）行为与合入前逐位一致：
    // 它要的收尾是「跨行加权累加」，需要全部行到齐，不是一个 (row, col) 组能做完的。
    if (cnt != nullptr && slices > 1) {
        __threadfence();
        __syncthreads();
        __shared__ int slast;
        const int slot = row * (int)gridDim.x + (int)blockIdx.x;
        if (threadIdx.x == 0) slast = (atomicAdd(&cnt[slot], 1) == slices - 1);
        __syncthreads();
        if (!slast) return;
        if (threadIdx.x == 0) cnt[slot] = 0;
        __threadfence();
        if (ok) {
            int tot = 0;
            for (int s = 0; s < slices; ++s) {
                tot += part[((size_t)s * (size_t)tokens + (size_t)row) * (size_t)N + col];
            }
            out[(size_t)row * N + col] = CxScalar<XT>::from_float(
                (float)tot * xs[row] * ws[(size_t)expert * N + col]);
        }
    }

    // 归并形态的末块收尾：``mcnt != nullptr`` 时本核同时替代
    // ``moe_grouped_i8_merge_rows_kernel``（单 token 的单流解码每层一次，40 层 40 次发射）。
    //
    // 这里的到达条件比上面宽一档：一列的加权和要**全部行、全部分片**都到齐，所以阈值是
    // ``行数 x 分片数``，计数器按**列块**（而不是 (行, 列) 组）计数。最后一个到达的块
    // 用一个线程算这一列块的每一列：读出 ``slices`` 个部分和、按行加权、按两个 scale 与
    // 路由权重相乘、写 ``out[col]``。
    //
    // 加权顺序与 ``moe_grouped_i8_merge_rows_kernel`` 逐位一致（行号升序、fp32 累加、
    // 每行先乘两个 scale 再乘路由权重），所以单流结果不变。
    if (mcnt != nullptr) {
        __threadfence();
        __syncthreads();
        __shared__ int mlast;
        if (threadIdx.x == 0) {
            // 阈值必须是**有效行数**而不是 ``gridDim.y``：不属于本 rank 的行在开头就整块
            // 退出了（它们没有 ``part`` 可写，也就不该计数）。EP4 上一个 token 的 8 个槽位
            // 里只有 64/256 机率落在本地，用整行数当阈值会让收尾永远攒不齐、``out`` 留着
            // 上一层的值——这是融合版第一次跑出乱码的原因。
            const int valid = offsets[experts];
            const int need = valid > 0 ? valid * (int)gridDim.z : 1;
            mlast = (atomicAdd(&mcnt[blockIdx.x], 1) == need - 1);
        }
        __syncthreads();
        if (!mlast) return;
        if (threadIdx.x == 0) mcnt[blockIdx.x] = 0;
        __threadfence();
        if (ok) {
            const int valid = offsets[experts];
            float acc = 0.f;
            for (int r = 0; r < (int)gridDim.y; ++r) {
                if (r >= valid) continue;
                const int er = i8_owner(offsets, experts, r);
                int tot = 0;
                for (int s = 0; s < slices; ++s) {
                    tot += part[((size_t)s * (size_t)tokens + (size_t)r) * (size_t)N + col];
                }
                const int sw = slot_of_pos[r];
                const float w = rw16 ? CxScalar<uint16_t>::to_float(
                                           reinterpret_cast<const uint16_t*>(rw)[sw])
                                     : reinterpret_cast<const float*>(rw)[sw];
                acc += (float)tot * xs[r] * ws[(size_t)er * N + col] * w;
            }
            out[(size_t)0 * N + col] = CxScalar<XT>::from_float(acc);
        }
    }
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
    if (colbase >= N) return;
    const int nc = (colbase + WCOL <= N) ? WCOL : (N - colbase);

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
            if (slices == 1) {
                out[(size_t)row * N + col] = CxScalar<XT>::from_float(
                    (float)total * xs[row] * ws[(size_t)expert * N + col]);
            } else {
                part[((size_t)blockIdx.z * (size_t)tokens + (size_t)row) * (size_t)N + col]
                    = total;
            }
        }
    }
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

// @brief 分组 GEMM 末块收尾用的到达计数，按 (设备, 块数) 常驻缓存。
//
// 与 ``gemv.cu`` 里 ``split_counters`` 同一件事、同一组约束：**初值必须是 0**，而**不能**
// 每次调用清零——那一次 memset 正好又变成要省掉的发射。kn 主体核在每个 (row, col) 组的
// 末块里把它写回 0 自复位，所以跨调用一直可用。
//
// 条目数等于出现过的不同 (列块 x 行) 组合，解码路径上只有个位数，所以线性扫描就够；
// 放在命名空间外是因为两个 ``moe_grouped_*`` 入口都要用它。
int* moe_grouped_counters(int device, int blocks) {
    struct Entry {
        int dev;
        int blocks;
        torch::Tensor buf;
    };
    static std::vector<Entry> cache;
    for (int i = 0; i < (int)cache.size(); ++i) {
        if (cache[i].dev == device && cache[i].blocks == blocks) {
            return cache[i].buf.data_ptr<int>();
        }
    }
    auto opts = torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA, device);
    Entry e{device, blocks, torch::zeros({blocks}, opts)};
    cache.push_back(e);
    return cache.back().buf.data_ptr<int>();
}

// @brief 分组 int8 投影：``[tokens, K] -> [tokens, N]``，每个专家一份权重。
//
// @param a       int8 激活，``[tokens, K]`` 连续 CUDA。
// @param xs      逐 token 激活 scale，``[tokens]`` fp32 连续。
// @param w       专家栈，``[E, K, N]`` 连续 int8（K 轴在外、N 连续）。
// @param ws      专家栈 scale，``[E, N]`` 连续 fp32。
// @param offsets ``[E+1]`` 前缀和，int32 连续，``offsets[0] == 0``。
// @param out     ``[tokens, N]`` 连续，bf16 或 fp32，dtype 由它决定。
// @param slices  K 方向分片数；``<= 0`` 表示按块数自动选。nk 布局下自动值就是 1 片。
// @param nk      非 0 时按 ``[E, N, K]`` 的权重栈走 nk 主体核（一列一行、K 连续）。
// @return out。
// @throws c10::Error 布局、设备、dtype 或形状不自洽。
//
// 分片的作用是换并行度：一行一个块时块数是 ``ceil(N/128) x tokens``，decode 一步
// 只有 8 行、门/升栈 N=1024，于是 64 个块 —— 在这张卡上远不够填。切 8 片后 512 块，
// 每片仍有 256 个 k，展开循环不会退化成标量尾巴。
//
// nk 那一支不需要靠分片换并行度：块数是 ``ceil(N/warp 数) x 行``，坐标本身就够多，
// 而每步读的是连续 128 字节（见 ``moe_grouped_i8_nk_kernel``）。
torch::Tensor moe_grouped_i8(torch::Tensor a, torch::Tensor xs, torch::Tensor w,
                             torch::Tensor ws, torch::Tensor offsets,
                             torch::Tensor out, int64_t slices, int64_t nk,
                             int64_t rows_valid) {
    TORCH_CHECK(a.is_cuda() && a.dim() == 2 && a.scalar_type() == torch::kChar
                && a.is_contiguous(), "activation must be contiguous CUDA int8 [tokens, K]");
    TORCH_CHECK(xs.is_cuda() && xs.dim() == 1 && xs.scalar_type() == torch::kFloat
                && xs.is_contiguous() && xs.size(0) == a.size(0),
                "activation scale must be contiguous fp32 [tokens]");
    TORCH_CHECK(w.is_cuda() && w.dim() == 3 && w.scalar_type() == torch::kChar
                && w.is_contiguous(),
                "weight must be contiguous CUDA int8, either [E, K, N] or the nk "
                "stack [E, N, K]");
    TORCH_CHECK(ws.is_cuda() && ws.dim() == 2 && ws.scalar_type() == torch::kFloat
                && ws.is_contiguous(), "weight scale must be contiguous fp32 [E, N]");
    TORCH_CHECK(offsets.is_cuda() && offsets.dim() == 1
                && offsets.scalar_type() == torch::kInt && offsets.is_contiguous(),
                "offsets must be contiguous CUDA int32 [E+1]");
    TORCH_CHECK(out.is_cuda() && out.dim() == 2 && out.is_contiguous()
                && out.size(0) == a.size(0),
                "destination must be contiguous CUDA [tokens, N]");
    TORCH_CHECK(out.scalar_type() == torch::kBFloat16 || out.scalar_type() == torch::kFloat,
                "destination must be bf16 or fp32");

    const int tokens = (int)a.size(0);
    const int K = (int)a.size(1);
    const int N = (int)out.size(1);
    const int experts = (int)w.size(0);
    const bool wide = nk != 0;
    if (wide) {
        TORCH_CHECK((int)w.size(1) == N && (int)w.size(2) == K,
                    "nk weight must be [E, N, K] matching activation and destination");
    } else {
        TORCH_CHECK((int)w.size(1) == K && (int)w.size(2) == N,
                    "weight must be [E, K, N] matching activation and destination");
    }
    TORCH_CHECK((int)ws.size(0) == experts && (int)ws.size(1) == N,
                "weight scale must be [E, N]");
    TORCH_CHECK((int)offsets.size(0) == experts + 1, "offsets must have E+1 entries");
    TORCH_CHECK(K > 0 && N > 0 && experts > 0, "sizes must be positive");
    if (tokens == 0) return out;

    // 两种布局的块几何不一样：nk 是「warp 数 x 列」（一个 warp 一列），kn 是 128 个列线程。
    // 收尾核（归约与三个收尾）仍然按 kn 的 128 列线程写，所以它们的网格单独算。
    // nk 的 16 字节路径要求每片的长度是 16 的倍数（片头不对齐会让 ``int4`` 跨进下一列）；
    // 4 字节路径只要 4 的倍数，也就是 ``__dp4a`` 本身的粒度。
    const int vb = (wide && K % I8_NK_WIDE_BYTES == 0 && K <= I8_NK_WIDE_K)
                       ? I8_NK_WIDE_BYTES : 4;
    // 浅 K（``down`` 的 K=512）一列只放得下一轮 512 字节宽步，单列根本填不满延迟；
    // 让一个 warp 认领 4 列并行，把在飞请求从 4 条抬到 16 条。只在 16 字节步这一档开。
    const int wcol = (vb == I8_NK_WIDE_BYTES && K < I8_NK_SHALLOW_K) ? I8_NK_WCOL_SHALLOW : 1;
    const int warps = (wide && K < I8_NK_SHALLOW_K) ? I8_NK_WARPS_SHALLOW : I8_NK_WARPS;
    const int cols = wide ? (N + warps * wcol - 1) / (warps * wcol)
                          : (N + I8_COLS - 1) / I8_COLS;
    const int cols_tail = (N + I8_COLS - 1) / I8_COLS;
    int span = (int)slices;
    if (span <= 0) {
        if (wide) {
            // nk 的并行度来自「列块 x 行」，已经是 1e4 量级的块数（gate_up 在 EP4 的 R8 上
            // 就有 256 个列块 x 16 个有效行），切 K 只会多一次归约和一块分片缓冲。所以单趟，
            // 主体核把结果窄化后直接写进 ``out``。
            span = 1;
        } else {
            // 自动选片数：先把块数抬到 256 以上，但每片不低于 ``I8_MIN_PER`` 个 k。
            // 这条规则保持原样：隔离测量里「每片 512 个 k」看起来更快（gate_up 249 vs 240
            // GB/s、down 111 vs 107），但端到端四档冒烟反而慢 2.6%（25.5 vs 26.3 tok/s），
            // 所以按纪律回退。隔离 benchmark 与线上口径不一致时，以后者为准。
            span = 1;
            while (span < I8_MAX_SLICES
                   && cols * tokens * (span + 1) <= 1024
                   && K / (span + 1) >= I8_MIN_PER) {
                ++span;
            }
        }
    }
    span = std::max(1, std::min(std::min(span, I8_MAX_SLICES), K));
    // nk 的 16 字节路径要求每片的长度是 16 的倍数（片头不对齐会让 ``int4`` 跨进下一列）；
    // 4 字节路径只要 4 的倍数，也就是 ``__dp4a`` 本身的粒度。
    const int per = ((K + span - 1) / span + vb - 1) / vb * vb;

    const c10::cuda::CUDAGuard guard(a.device());
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    torch::Tensor part;
    int32_t* part_ptr = nullptr;
    if (span > 1) {
        // ``empty`` 而不是 ``zeros``：每一片都覆盖全部 (row, col)，末尾列块的越界线程
        // 在归约核里同样被 ``col >= N`` 挡住，所以不存在未初始化读。
        part = torch::empty({(int64_t)span, (int64_t)tokens, (int64_t)N},
                            out.options().dtype(torch::kInt));
        part_ptr = part.data_ptr<int32_t>();
    }

    // 块与行一一对应：kn 主体核和 nk 核都按 ``tokens`` 给 y 维。
    // 行槽实验（``grid.y = 8`` + 块内跨行）已否，见 nk 核里那段注释。
    const unsigned rows_launch = (unsigned)tokens;
    const dim3 grid((unsigned)cols, rows_launch, (unsigned)span);
    const dim3 block(wide ? (unsigned)(warps * 32) : (unsigned)I8_COLS);
    const size_t smem = (size_t)K * sizeof(int8_t);
    // kn 主体核带末块收尾，于是外面的 ``moe_grouped_i8_reduce`` 可以整趟省掉（解码每层
    // 一次，40 层就是 40 次发射）。nk 主体核没有收尾，``wide`` 仍走两趟。
    int* cnt_ptr = nullptr;
    if (span > 1 && !wide) {
        cnt_ptr = moe_grouped_counters((int)a.get_device(), cols * (int)rows_launch);
    }
    auto launch = [&](auto tag) {
        using XT = decltype(tag);
        XT* dst = reinterpret_cast<XT*>(out.data_ptr());
        const int8_t* wp = w.data_ptr<int8_t>();
        if (!wide) {
            moe_grouped_i8_kernel<XT><<<grid, block, smem, stream>>>(
                a.data_ptr<int8_t>(), wp, ws.data_ptr<float>(),
                xs.data_ptr<float>(), offsets.data_ptr<int>(), part_ptr, cnt_ptr, nullptr,
                nullptr, nullptr, 0, dst,
                K, N, experts, tokens, span, per);
        } else if (vb == I8_NK_WIDE_BYTES) {
            if (wcol > 1) {
                // K=512 且不分片时给 K 一个编译期版本：循环、``min``、尾部兜底全消失。
                // 其余形状仍走运行期 K 的通用路径。
                if (K == 512 && span == 1) {
                    moe_grouped_i8_nk_kernel<XT, I8_NK_WIDE_BYTES, I8_NK_WCOL_SHALLOW, 512>
                        <<<grid, block, smem, stream>>>(
                            a.data_ptr<int8_t>(), wp, ws.data_ptr<float>(),
                            xs.data_ptr<float>(), offsets.data_ptr<int>(), part_ptr, dst,
                            K, N, experts, tokens, span, per);
                } else {
                    moe_grouped_i8_nk_kernel<XT, I8_NK_WIDE_BYTES, I8_NK_WCOL_SHALLOW>
                        <<<grid, block, smem, stream>>>(
                            a.data_ptr<int8_t>(), wp, ws.data_ptr<float>(),
                            xs.data_ptr<float>(), offsets.data_ptr<int>(), part_ptr, dst,
                            K, N, experts, tokens, span, per);
                }
            } else {
                moe_grouped_i8_nk_kernel<XT, I8_NK_WIDE_BYTES, 1>
                    <<<grid, block, smem, stream>>>(
                        a.data_ptr<int8_t>(), wp, ws.data_ptr<float>(),
                        xs.data_ptr<float>(), offsets.data_ptr<int>(), part_ptr, dst,
                        K, N, experts, tokens, span, per);
            }
        } else {
            // 4 字节步这一档固定 1 列：``wcol > 1`` 只在 ``K % 16 == 0`` 的浅档出现，
            // 与这里不可能同时成立（见 ``wcol`` 的定义）。
            moe_grouped_i8_nk_kernel<XT, 4, 1><<<grid, block, smem, stream>>>(
                a.data_ptr<int8_t>(), wp, ws.data_ptr<float>(),
                xs.data_ptr<float>(), offsets.data_ptr<int>(), part_ptr, dst,
                K, N, experts, tokens, span, per);
        }
        if (span > 1 && cnt_ptr == nullptr) {
            // 收尾核按 kn 的 128 列线程布局，块与线程都要回到那一档。
            // kn 主体核（``cnt_ptr`` 非空）已经把这一步做完了，所以只有 nk 走到这里。
            const dim3 rgrid((unsigned)cols_tail, (unsigned)tokens);
            const dim3 rblock(I8_COLS);
            moe_grouped_i8_reduce_kernel<XT><<<rgrid, rblock, 0, stream>>>(
                part_ptr, ws.data_ptr<float>(), xs.data_ptr<float>(),
                offsets.data_ptr<int>(), dst, N, experts, tokens, span);
        }
    };
    if (out.scalar_type() == torch::kBFloat16) launch(uint16_t{});
    else launch(float{});
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// @brief 归并形态的多流版本：``[rows, K] -> [tokens, N]``，每个 token 只累加自己的行。
//
// 存在的理由是外层的四到十次发射全都可以搬进来：路由权重的 gather（一次
// ``index_select``）、专家并行的有效行掩码（``arange``/``lt``/``to``/``mul`` 四次）、
// 以及多流的收尾（``zero_``/``mul``/``index_add_`` 与两次 ``.long()``）。这些算子的输入
// 输出都只有 ``[tokens, top_k]`` 量级，一步 8 行时 40 层合计 10 次 × 40 次发射，而它们
// 搬的数据不到 10 KB —— 全是发射与块数不足的成本。
//
// 权重直接按槽位号寻址，所以调用方既不必把 ``[tokens * top_k]`` 摊成逐行表，也不必把它
// 转成 fp32：内核按 ``rw`` 自己的 dtype 读。
//
// @param a           int8 激活，``[rows, K]`` 连续 CUDA，已按专家号排序。
// @param xs          逐行激活 scale，``[rows]`` fp32 连续。
// @param w           专家栈，``[E, K, N]`` 连续 int8。
// @param ws          专家栈 scale，``[E, N]`` 连续 fp32。
// @param offsets     ``[E+1]`` 前缀和，int32 连续。
// @param rw          路由权重，``[rows]`` 连续 bf16 或 fp32。
// @param slot_pos    槽位 → 行号，int32 ``[rows]`` 连续，丢弃的为 ``-1``。
// @param slot_of_pos 行号 → 槽位，int32 ``[rows]`` 连续。
// @param tokens      行数除以 ``top_k``；``1`` 时走行序以保持与单流路径逐位相同。
// @param top_k       每 token 的专家数。
// @param out         ``[tokens, N]`` 连续，bf16 或 fp32，dtype 由它决定。
// @param slices      K 方向分片数；``<= 0`` 自动，本形态至少用 2 片（理由同
//                    :func:`moe_grouped_i8_merge`）。
// @param nk          非 0 时按 ``[E, N, K]`` 的权重栈走 nk 主体核。
// @return out。
// @throws c10::Error 布局、设备、dtype 或形状不自洽。
torch::Tensor moe_grouped_i8_merge_tokens(torch::Tensor a, torch::Tensor xs,
                                          torch::Tensor w, torch::Tensor ws,
                                          torch::Tensor offsets, torch::Tensor rw,
                                          torch::Tensor slot_pos,
                                          torch::Tensor slot_of_pos,
                                          int64_t tokens, int64_t top_k,
                                          torch::Tensor out, int64_t slices,
                                          int64_t nk) {
    TORCH_CHECK(a.is_cuda() && a.dim() == 2 && a.scalar_type() == torch::kChar
                && a.is_contiguous(), "activation must be contiguous CUDA int8 [rows, K]");
    TORCH_CHECK(xs.is_cuda() && xs.dim() == 1 && xs.scalar_type() == torch::kFloat
                && xs.is_contiguous() && xs.size(0) == a.size(0),
                "activation scale must be contiguous fp32 [rows]");
    TORCH_CHECK(w.is_cuda() && w.dim() == 3 && w.scalar_type() == torch::kChar
                && w.is_contiguous(),
                "weight must be contiguous CUDA int8, either [E, K, N] or the nk "
                "stack [E, N, K]");
    TORCH_CHECK(ws.is_cuda() && ws.dim() == 2 && ws.scalar_type() == torch::kFloat
                && ws.is_contiguous(), "weight scale must be contiguous fp32 [E, N]");
    TORCH_CHECK(offsets.is_cuda() && offsets.dim() == 1
                && offsets.scalar_type() == torch::kInt && offsets.is_contiguous(),
                "offsets must be contiguous CUDA int32 [E+1]");
    TORCH_CHECK(rw.is_cuda() && rw.dim() == 1 && rw.is_contiguous()
                && (rw.scalar_type() == torch::kBFloat16
                    || rw.scalar_type() == torch::kFloat)
                && rw.size(0) == a.size(0),
                "row weight must be contiguous CUDA bf16 or fp32 [rows]");
    TORCH_CHECK(slot_pos.is_cuda() && slot_pos.dim() == 1
                && slot_pos.scalar_type() == torch::kInt && slot_pos.is_contiguous()
                && slot_pos.numel() == a.size(0),
                "slot positions must be contiguous CUDA int32 [rows]");
    TORCH_CHECK(slot_of_pos.is_cuda() && slot_of_pos.dim() == 1
                && slot_of_pos.scalar_type() == torch::kInt && slot_of_pos.is_contiguous()
                && slot_of_pos.numel() == a.size(0),
                "row slots must be contiguous CUDA int32 [rows]");
    TORCH_CHECK(out.is_cuda() && out.dim() == 2 && out.is_contiguous(),
                "destination must be contiguous CUDA [tokens, N]");
    TORCH_CHECK(out.scalar_type() == torch::kBFloat16
                || out.scalar_type() == torch::kFloat,
                "destination must be bf16 or fp32");
    TORCH_CHECK(tokens > 0 && top_k > 0 && tokens * top_k == a.size(0),
                "tokens * top_k must equal the activation rows");
    TORCH_CHECK(out.size(0) == tokens, "destination needs one row per token");

    const int rows = (int)a.size(0);
    const int K = (int)a.size(1);
    const int N = (int)out.size(1);
    const int experts = (int)w.size(0);
    const bool wide = nk != 0;
    if (wide) {
        TORCH_CHECK((int)w.size(1) == N && (int)w.size(2) == K,
                    "nk weight must be [E, N, K] matching activation and destination");
    } else {
        TORCH_CHECK((int)w.size(1) == K && (int)w.size(2) == N,
                    "weight must be [E, K, N] matching activation and destination");
    }
    TORCH_CHECK((int)ws.size(0) == experts && (int)ws.size(1) == N,
                "weight scale must be [E, N]");
    TORCH_CHECK((int)offsets.size(0) == experts + 1, "offsets must have E+1 entries");
    TORCH_CHECK(K > 0 && N > 0 && experts > 0, "sizes must be positive");

    // nk 的 16 字节路径要求每片的长度是 16 的倍数（片头不对齐会让 ``int4`` 跨进下一列）；
    // 4 字节路径只要 4 的倍数，也就是 ``__dp4a`` 本身的粒度。
    const int vb = (wide && K % I8_NK_WIDE_BYTES == 0 && K <= I8_NK_WIDE_K)
                       ? I8_NK_WIDE_BYTES : 4;
    // 浅 K（``down`` 的 K=512）一列只放得下一轮 512 字节宽步，单列根本填不满延迟；
    // 让一个 warp 认领 4 列并行，把在飞请求从 4 条抬到 16 条。只在 16 字节步这一档开。
    const int wcol = (vb == I8_NK_WIDE_BYTES && K < I8_NK_SHALLOW_K) ? I8_NK_WCOL_SHALLOW : 1;
    const int warps = (wide && K < I8_NK_SHALLOW_K) ? I8_NK_WARPS_SHALLOW : I8_NK_WARPS;
    const int cols = wide ? (N + warps * wcol - 1) / (warps * wcol)
                          : (N + I8_COLS - 1) / I8_COLS;
    // 收尾核按 kn 的 128 列线程写，网格要按那一档算。
    const int cols_tail = (N + I8_COLS - 1) / I8_COLS;
    int span = (int)slices;
    if (span <= 0) {
        if (wide) {
            // nk 的自动值就是 2：再少就退化成 1 片，而 1 片时主体核写的是 ``out[row]``
            // （逐行落盘），本形态要的是按 token 累加，所以至少留 2 片走 ``part``。
            span = 2;
        } else {
            span = 1;
            while (span < I8_MAX_SLICES
                   && cols * rows * (span + 1) <= 1024
                   && K / (span + 1) >= I8_MIN_PER) {
                ++span;
            }
        }
    }
    span = std::max(2, std::min(std::min(span, I8_MAX_SLICES), K));
    const int per = ((K + span - 1) / span + vb - 1) / vb * vb;

    const c10::cuda::CUDAGuard guard(a.device());
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // 每一片都覆盖全部 (row, col)；越界列在收尾核里被 ``col >= N`` 挡住。
    torch::Tensor part = torch::empty({(int64_t)span, (int64_t)rows, (int64_t)N},
                                      out.options().dtype(torch::kInt));
    int32_t* part_ptr = part.data_ptr<int32_t>();

    const dim3 grid((unsigned)cols, (unsigned)rows, (unsigned)span);
    const dim3 block(wide ? (unsigned)(warps * 32) : (unsigned)I8_COLS);
    const size_t smem = (size_t)K * sizeof(int8_t);
    const bool single = tokens == 1;
    // 单 token 时 kn 主体核自己就能做「跨行加权累加」的收尾（见内核末尾那段），于是
    // ``moe_grouped_i8_merge_rows`` 那一趟可以整趟省掉。计数器按**列块**计数，阈值是
    // 行 x 片；nk 主体核与多 token 形态仍走独立的收尾核。
    int* mcnt = (!wide && single && span > 1)
                    ? moe_grouped_counters((int)a.get_device(), cols) : nullptr;
    const bool rw16 = rw.scalar_type() == torch::kBFloat16;
    const dim3 mgrid((unsigned)cols_tail, (unsigned)(single ? 1 : tokens));
    const dim3 mblock(I8_COLS);
    auto launch = [&](auto tag) {
        using XT = decltype(tag);
        XT* dst = reinterpret_cast<XT*>(out.data_ptr());
        const int8_t* wp = w.data_ptr<int8_t>();
        // 主体核一律写 ``part``：本形态强制 ``span >= 2``（理由同单流归并），所以
        // ``out`` 那一支不会被走到；传 ``dst`` 只是保持签名。
        if (!wide) {
            moe_grouped_i8_kernel<XT><<<grid, block, smem, stream>>>(
                a.data_ptr<int8_t>(), wp, ws.data_ptr<float>(),
                xs.data_ptr<float>(), offsets.data_ptr<int>(), part_ptr, nullptr, mcnt,
                (const void*)rw.data_ptr(), slot_of_pos.data_ptr<int>(), rw16 ? 1 : 0,
                dst, K, N, experts, rows, span, per);
        } else if (vb == I8_NK_WIDE_BYTES) {
            if (wcol > 1) {
                moe_grouped_i8_nk_kernel<XT, I8_NK_WIDE_BYTES, I8_NK_WCOL_SHALLOW>
                    <<<grid, block, smem, stream>>>(
                        a.data_ptr<int8_t>(), wp, ws.data_ptr<float>(),
                        xs.data_ptr<float>(), offsets.data_ptr<int>(), part_ptr, dst,
                        K, N, experts, rows, span, per);
            } else {
                moe_grouped_i8_nk_kernel<XT, I8_NK_WIDE_BYTES, 1>
                    <<<grid, block, smem, stream>>>(
                        a.data_ptr<int8_t>(), wp, ws.data_ptr<float>(),
                        xs.data_ptr<float>(), offsets.data_ptr<int>(), part_ptr, dst,
                        K, N, experts, rows, span, per);
            }
        } else {
            moe_grouped_i8_nk_kernel<XT, 4, 1><<<grid, block, smem, stream>>>(
                a.data_ptr<int8_t>(), wp, ws.data_ptr<float>(),
                xs.data_ptr<float>(), offsets.data_ptr<int>(), part_ptr, dst,
                K, N, experts, rows, span, per);
        }
        // 权重按它自己的 dtype 实例化：多流路径的路由权重是 bf16，若在这里转 fp32
        // 就又是一次发射，而这个融合要省的正是那一次。
        const bool wr_bf16 = rw.scalar_type() == torch::kBFloat16;
        // ``mcnt`` 非空时 kn 主体核已经把单 token 的加权累加做完了（见内核末尾），
        // 这一趟整个省掉。注意这个跳过必须落在 ``single`` 里面而不是并进条件里：
        // 并进条件会让「single 且 mcnt 非空」掉进 ``merge_slots`` 那一支，
        // 而 merge_slots 要的是逐 token 的行表，不是本形态的语义。
        if (single) {
            if (mcnt == nullptr) {
                if (wr_bf16) {
                    moe_grouped_i8_merge_rows_kernel<XT, uint16_t><<<mgrid, mblock, 0, stream>>>(
                        part_ptr, ws.data_ptr<float>(), xs.data_ptr<float>(),
                        offsets.data_ptr<int>(),
                        reinterpret_cast<const uint16_t*>(rw.data_ptr()),
                        slot_of_pos.data_ptr<int>(), dst, N, experts, rows, span);
                } else {
                    moe_grouped_i8_merge_rows_kernel<XT, float><<<mgrid, mblock, 0, stream>>>(
                        part_ptr, ws.data_ptr<float>(), xs.data_ptr<float>(),
                        offsets.data_ptr<int>(), rw.data_ptr<float>(),
                        slot_of_pos.data_ptr<int>(), dst, N, experts, rows, span);
                }
            }
        } else if (wr_bf16) {
            moe_grouped_i8_merge_slots_kernel<XT, uint16_t><<<mgrid, mblock, 0, stream>>>(
                part_ptr, ws.data_ptr<float>(), xs.data_ptr<float>(),
                offsets.data_ptr<int>(),
                reinterpret_cast<const uint16_t*>(rw.data_ptr()),
                slot_pos.data_ptr<int>(), dst, N, experts, rows, span, (int)top_k);
        } else {
            moe_grouped_i8_merge_slots_kernel<XT, float><<<mgrid, mblock, 0, stream>>>(
                part_ptr, ws.data_ptr<float>(), xs.data_ptr<float>(),
                offsets.data_ptr<int>(), rw.data_ptr<float>(),
                slot_pos.data_ptr<int>(), dst, N, experts, rows, span, (int)top_k);
        }
    };
    if (out.scalar_type() == torch::kBFloat16) launch(uint16_t{});
    else launch(float{});
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// @brief 分组 int8 投影的归并形态：``[rows, K] -> [1, N]``，行按路由权重加权求和。
//
// 存在的理由是一次融合省掉的比投影本身还多：外面那条路要先拿到 ``[top_k, N]`` 的
// 激活，再做 ``hidden * w``、``index_add_``、``view(...).sum(1)`` 三步，实测 0.184 ms，
// 而 down 的投影只有 0.059 ms。并进收尾核之后这四步一起消失。
//
// @param a       int8 激活，``[rows, K]`` 连续 CUDA，已按专家号排序。
// @param xs      逐行激活 scale，``[rows]`` fp32 连续。
// @param w       专家栈，``[E, K, N]`` 连续 int8。
// @param ws      专家栈 scale，``[E, N]`` 连续 fp32。
// @param offsets ``[E+1]`` 前缀和，int32 连续。
// @param rw       逐行路由权重，``[rows]`` fp32 连续。
// @param out      ``[1, N]`` 连续，bf16 或 fp32，dtype 由它决定。
// @param slices   K 方向分片数；``<= 0`` 自动。归并形态至少用 2 片（见下）。
// @param nk       非 0 时按 ``[E, N, K]`` 的权重栈走 nk 主体核。
// @return out。
// @throws c10::Error 布局、设备、dtype 或形状不自洽。
//
// 为什么至少 2 片：1 片时主体核直接把结果写成 bf16 写到 ``out[row]``，而这里只有一行
// 目的地，归并就得多一趟读改写。抬到 2 片让主体核一律走 ``part``，收尾核顺手做完加权
// 求和；down（K=512）自动选片本来就会选到 3 片，这条只影响调用方硬指定 1 片的情况。
torch::Tensor moe_grouped_i8_merge(torch::Tensor a, torch::Tensor xs, torch::Tensor w,
                                   torch::Tensor ws, torch::Tensor offsets,
                                   torch::Tensor rw, torch::Tensor out, int64_t slices,
                                   int64_t nk) {
    TORCH_CHECK(a.is_cuda() && a.dim() == 2 && a.scalar_type() == torch::kChar
                && a.is_contiguous(), "activation must be contiguous CUDA int8 [rows, K]");
    TORCH_CHECK(xs.is_cuda() && xs.dim() == 1 && xs.scalar_type() == torch::kFloat
                && xs.is_contiguous() && xs.size(0) == a.size(0),
                "activation scale must be contiguous fp32 [rows]");
    TORCH_CHECK(w.is_cuda() && w.dim() == 3 && w.scalar_type() == torch::kChar
                && w.is_contiguous(),
                "weight must be contiguous CUDA int8, either [E, K, N] or the nk "
                "stack [E, N, K]");
    TORCH_CHECK(ws.is_cuda() && ws.dim() == 2 && ws.scalar_type() == torch::kFloat
                && ws.is_contiguous(), "weight scale must be contiguous fp32 [E, N]");
    TORCH_CHECK(offsets.is_cuda() && offsets.dim() == 1
                && offsets.scalar_type() == torch::kInt && offsets.is_contiguous(),
                "offsets must be contiguous CUDA int32 [E+1]");
    TORCH_CHECK(rw.is_cuda() && rw.dim() == 1 && rw.scalar_type() == torch::kFloat
                && rw.is_contiguous() && rw.size(0) == a.size(0),
                "row weight must be contiguous fp32 [rows]");
    TORCH_CHECK(out.is_cuda() && out.dim() == 2 && out.is_contiguous()
                && out.size(0) == 1, "merged destination must be contiguous [1, N]");
    TORCH_CHECK(out.scalar_type() == torch::kBFloat16 || out.scalar_type() == torch::kFloat,
                "destination must be bf16 or fp32");

    const int rows = (int)a.size(0);
    const int K = (int)a.size(1);
    const int N = (int)out.size(1);
    const int experts = (int)w.size(0);
    const bool wide = nk != 0;
    if (wide) {
        TORCH_CHECK((int)w.size(1) == N && (int)w.size(2) == K,
                    "nk weight must be [E, N, K] matching activation and destination");
    } else {
        TORCH_CHECK((int)w.size(1) == K && (int)w.size(2) == N,
                    "weight must be [E, K, N] matching activation and destination");
    }
    TORCH_CHECK((int)ws.size(0) == experts && (int)ws.size(1) == N,
                "weight scale must be [E, N]");
    TORCH_CHECK((int)offsets.size(0) == experts + 1, "offsets must have E+1 entries");
    TORCH_CHECK(K > 0 && N > 0 && experts > 0 && rows > 0, "sizes must be positive");

    // nk 的 16 字节路径要求每片的长度是 16 的倍数（片头不对齐会让 ``int4`` 跨进下一列）；
    // 4 字节路径只要 4 的倍数，也就是 ``__dp4a`` 本身的粒度。
    const int vb = (wide && K % I8_NK_WIDE_BYTES == 0 && K <= I8_NK_WIDE_K)
                       ? I8_NK_WIDE_BYTES : 4;
    // 浅 K（``down`` 的 K=512）一列只放得下一轮 512 字节宽步，单列根本填不满延迟；
    // 让一个 warp 认领 4 列并行，把在飞请求从 4 条抬到 16 条。只在 16 字节步这一档开。
    const int wcol = (vb == I8_NK_WIDE_BYTES && K < I8_NK_SHALLOW_K) ? I8_NK_WCOL_SHALLOW : 1;
    const int warps = (wide && K < I8_NK_SHALLOW_K) ? I8_NK_WARPS_SHALLOW : I8_NK_WARPS;
    const int cols = wide ? (N + warps * wcol - 1) / (warps * wcol)
                          : (N + I8_COLS - 1) / I8_COLS;
    // 收尾核按 kn 的 128 列线程写，网格要按那一档算。
    const int cols_tail = (N + I8_COLS - 1) / I8_COLS;
    int span = (int)slices;
    if (span <= 0) {
        if (wide) {
            // nk 的自动值就是 2（理由与 kn 那一支相同：1 片时主体核写逐行落盘，
            // 而这个形态只有一行目的地）。
            span = 2;
        } else {
            span = 1;
            while (span < I8_MAX_SLICES
                   && cols * rows * (span + 1) <= 1024
                   && K / (span + 1) >= I8_MIN_PER) {
                ++span;
            }
        }
    }
    span = std::max(2, std::min(std::min(span, I8_MAX_SLICES), K));
    const int per = ((K + span - 1) / span + vb - 1) / vb * vb;

    const c10::cuda::CUDAGuard guard(a.device());
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // 每一片都覆盖全部 (row, col)，末尾列块的越界线程在收尾核里同样被 ``col >= N``
    // 挡住，所以这里用 ``empty`` 而不是 ``zeros``。
    torch::Tensor part = torch::empty({(int64_t)span, (int64_t)rows, (int64_t)N},
                                      out.options().dtype(torch::kInt));
    int32_t* part_ptr = part.data_ptr<int32_t>();

    const dim3 grid((unsigned)cols, (unsigned)rows, (unsigned)span);
    const dim3 block(wide ? (unsigned)(warps * 32) : (unsigned)I8_COLS);
    const size_t smem = (size_t)K * sizeof(int8_t);
    auto launch = [&](auto tag) {
        using XT = decltype(tag);
        XT* dst = reinterpret_cast<XT*>(out.data_ptr());
        const int8_t* wp = w.data_ptr<int8_t>();
        // 主体核的 ``tokens`` 参数就是这里的 ``rows``；``part`` 的步长与之一致。
        if (!wide) {
            moe_grouped_i8_kernel<XT><<<grid, block, smem, stream>>>(
                a.data_ptr<int8_t>(), wp, ws.data_ptr<float>(),
                xs.data_ptr<float>(), offsets.data_ptr<int>(), part_ptr, nullptr, nullptr,
                nullptr, nullptr, 0, dst,
                K, N, experts, rows, span, per);
        } else if (vb == I8_NK_WIDE_BYTES) {
            if (wcol > 1) {
                moe_grouped_i8_nk_kernel<XT, I8_NK_WIDE_BYTES, I8_NK_WCOL_SHALLOW>
                    <<<grid, block, smem, stream>>>(
                        a.data_ptr<int8_t>(), wp, ws.data_ptr<float>(),
                        xs.data_ptr<float>(), offsets.data_ptr<int>(), part_ptr, dst,
                        K, N, experts, rows, span, per);
            } else {
                moe_grouped_i8_nk_kernel<XT, I8_NK_WIDE_BYTES, 1>
                    <<<grid, block, smem, stream>>>(
                        a.data_ptr<int8_t>(), wp, ws.data_ptr<float>(),
                        xs.data_ptr<float>(), offsets.data_ptr<int>(), part_ptr, dst,
                        K, N, experts, rows, span, per);
            }
        } else {
            moe_grouped_i8_nk_kernel<XT, 4, 1><<<grid, block, smem, stream>>>(
                a.data_ptr<int8_t>(), wp, ws.data_ptr<float>(),
                xs.data_ptr<float>(), offsets.data_ptr<int>(), part_ptr, dst,
                K, N, experts, rows, span, per);
        }
        const dim3 mgrid((unsigned)cols_tail);
        const dim3 mblock(I8_COLS);
        moe_grouped_i8_merge_kernel<XT><<<mgrid, mblock, 0, stream>>>(
            part_ptr, ws.data_ptr<float>(), xs.data_ptr<float>(),
            offsets.data_ptr<int>(), rw.data_ptr<float>(), dst, N, experts, rows, span);
    };
    if (out.scalar_type() == torch::kBFloat16) launch(uint16_t{});
    else launch(float{});
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// 绑定写在内核旁边：名字、参数名与默认值只有这一份，`bindings.cpp` 只遍历注册表。
COFFIN_BIND(moe_grouped_i8,
            "\u5206\u7ec4 int8 \u6295\u5f71\uff1a\u4e13\u5bb6\u6808 [E, K, N] \u6216 nk "
            "\u5806 [E, N, K] int8 + \u9010 (\u4e13\u5bb6, \u5217) scale",
            pybind11::arg("a"), pybind11::arg("xs"), pybind11::arg("w"), pybind11::arg("ws"),
            pybind11::arg("offsets"), pybind11::arg("out"), pybind11::arg("slices") = 0,
            pybind11::arg("nk") = 0, pybind11::arg("rows_valid") = -1);
COFFIN_BIND(moe_grouped_i8_merge,
            "分组 int8 投影的归并形态：各行按路由权重加权累加成一行",
            pybind11::arg("a"), pybind11::arg("xs"), pybind11::arg("w"), pybind11::arg("ws"),
            pybind11::arg("offsets"), pybind11::arg("rw"), pybind11::arg("out"),
            pybind11::arg("slices") = 0, pybind11::arg("nk") = 0);
COFFIN_BIND(moe_grouped_i8_merge_tokens,
            "归并形态的多流版本：每个 token 只累加自己的 top_k 行，权重按槽位号直取",
            pybind11::arg("a"), pybind11::arg("xs"), pybind11::arg("w"), pybind11::arg("ws"),
            pybind11::arg("offsets"), pybind11::arg("rw"), pybind11::arg("slot_pos"),
            pybind11::arg("slot_of_pos"), pybind11::arg("tokens"), pybind11::arg("top_k"),
            pybind11::arg("out"), pybind11::arg("slices") = 0, pybind11::arg("nk") = 0);
