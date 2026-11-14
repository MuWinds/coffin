// @file w4a8.cu
// @brief 从 packed W4 和逐 token int8 激活计算分组整数点积。
//
// Module: coffin/backends/corex/kernels。由 CoreX 投影派发调用，不依赖模型。
//
// Constraints:
//   * 每个 K 组先做 int32 累加，再乘该组 scale；不能把不同组的整数和直接相加。
//   * 解包只在寄存器中发生，常驻权重始终为 uint8 packed，不分配整矩阵 int8 副本。
//   * 不使用 graph 或异步拷贝指令，所有启动都在调用设备的当前流上。
//   * 一个线程负责的列数是编译期常量 ``CW``，所以 ``N`` 不整除 ``CW`` 时必须能读越界
//     之外的列而不写坏输出；见 ``w4a8_dot`` 的边界判断。
//   * 分片缓冲区是 ``[split, M, N]`` fp32：归约要能按 ``s = 0 .. split-1`` 累加，所以
//     分片不能共用输出缓冲区。

#include "corex_bindings.h"
#include "corex_ops.h"
#include "corex_scalar.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <climits>

namespace {

// @brief 行并行的各 rank 共用全局 scale，避免分片改变激活量化的数学定义。
template <typename XT>
__global__ void quant_with_scale(const XT* x, int8_t* out, const float* xs, int K) {
    const int row = blockIdx.x;
    const float inv = 1.f / xs[row];
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        const int value = __float2int_rn(CxScalar<XT>::to_float(x[(size_t)row * K + k]) * inv);
        out[(size_t)row * K + k] = (int8_t)max(-127, min(127, value));
    }
}

// @brief 同一权重加载服务 MT 行激活，K 分片只能在组边界切开。
//
// 一个线程持 ``CW`` 个 *相邻* 输出列。列在 packed 权重的一行里连续，所以 ``CW = 4``
// 时每个 k 步读的是一个对齐的 32 位字而不是 4 个单字节——一次事务覆盖 4 倍的列。这
// 与 ``gemv.cu`` 里 ``gemv_kn_i8x4_kernel`` 是同一条实测结论（int8 GEMV 的宽加载在
// ``N >= 4096`` 上赢 1.2-1.6x），只是这里的权重是半字节打包的。
//
// 为什么 dp4a 的两个字必须沿 K 配对：``__dp4a`` 把四个 int8 两两相乘再求和，而本内核
// 的归约轴是 K。把相邻输出列当成那四个元素会算出一个数学上不存在的东西——形状全对、
// 数值全错。
//
// @param x [M,K] 连续 int8 激活。
// @param w [至少 ceil(K/2),N] 连续 uint8 packed 权重。
// @param ws [ceil(K/gs),N] fp32 权重 scale。
// @param partial [split,M,N] fp32 部分和。
// @param gs 分组大小。
// @param groups_per_split 每个分片负责的组数。
template <int MT, int CW, int U>
__global__ void w4a8_dot(const int8_t* __restrict__ x, const uint8_t* __restrict__ w,
                         const float* __restrict__ ws, float* __restrict__ partial,
                         int M, int N, int K, int gs, int groups_per_split) {
    const int n0 = (blockIdx.x * blockDim.x + threadIdx.x) * CW;
    if (n0 >= N) return;
    const int m0 = blockIdx.y * MT;
    const int split = blockIdx.z;
    const int groups = (K + gs - 1) / gs;
    const int g0 = split * groups_per_split;
    const int g1 = min(groups, g0 + groups_per_split);
    float total[MT][CW];
#pragma unroll
    for (int r = 0; r < MT; ++r) {
#pragma unroll
        for (int c = 0; c < CW; ++c) total[r][c] = 0.f;
    }
    // 宽加载和逐列加载是编译期分支：``CW = 1`` 时那个 32 位字会读进相邻列之外，
    // 而 ``CW = 4`` 时逐列版本又白白放弃了一半的事务。写成 ``if (CW >= 4 && full)``
    // 让编译器把不需要的那一支整个删掉——用运行期条件的话，``CW = 1`` 的实例化里仍然
    // 存在一次 4 字节加载，它读数正确纯属侥幸。
    const bool full = CW >= 4 && (n0 + CW <= N);
    for (int g = g0; g < g1; ++g) {
        int sums[MT][CW];
#pragma unroll
        for (int r = 0; r < MT; ++r) {
#pragma unroll
            for (int c = 0; c < CW; ++c) sums[r][c] = 0;
        }
        int k = g * gs;
        const int end = min(K, k + gs);
        // dp4a 的主体要求 K 与 k 都是 4 的倍数；不满足时整组走标量尾巴，所以任何 K
        // 都正确，而不是只有 K % 4 == 0 才正确。
        if ((K & 3) == 0 && (k & 3) == 0) {
            for (; k + 4 <= end; k += 4) {
                unsigned q[CW];
                if (CW >= 4 && full) {
                    // ``CW = 4`` 时 4 个相邻列在 packed 权重里正好是连续 4 个字节，也就
                    // 是一个对齐的 32 位字。所以这个 k 步只发 2 次加载（k 行一次、k+1
                    // 行一次），而不是每列每行各一次——这正是 ``CW`` 存在的全部理由。
                    //
                    // 对齐成立的条件：``w`` 是 torch 分配的、至少 4 字节对齐的基址，
                    // ``(k/2) * N`` 是 4 的倍数（``K % 4 == 0`` 保证 ``k/2`` 是偶数，
                    // ``N % 4 == 0`` 由 ``pick_cw`` 保证），``n0 = tid * 4`` 也是 4 的
                    // 倍数。少任何一条，这里的对齐加载就是错的。
                    //
                    // 下面只能写 ``lo_row[0]`` + 按 8 位移位，**不能**写成 ``row[c / 2]``：
                    // 指针已经是 ``const unsigned*``，按 ``unsigned`` 索引会把字节偏移
                    // 放大 4 倍。那个写法形状全对、数值全错、也不会崩——本仓库已经出过
                    // 三次同一类（``solve`` 的 ``serial``、``moe_grouped`` 的偶数行、
                    // 这里第一版的字节索引），三次的共同点都是形状校验全过。
                    const unsigned* lo_row = reinterpret_cast<const unsigned*>(
                        w + (size_t)(k / 2) * N + n0);
                    const unsigned* hi_row = reinterpret_cast<const unsigned*>(
                        w + (size_t)(k / 2 + 1) * N + n0);
                    const unsigned lo_word = lo_row[0];
                    const unsigned hi_word = hi_row[0];
#pragma unroll
                    for (int c = 0; c < CW; ++c) {
                        const unsigned lo = (lo_word >> (8 * c)) & 0xffu;
                        const unsigned hi = (hi_word >> (8 * c)) & 0xffu;
                        q[c] = ((unsigned)((int)(lo & 15) - 8) & 255)
                            | (((unsigned)((int)((lo >> 4) & 15) - 8) & 255) << 8)
                            | (((unsigned)((int)(hi & 15) - 8) & 255) << 16)
                            | (((unsigned)((int)((hi >> 4) & 15) - 8) & 255) << 24);
                    }
                } else {
#pragma unroll
                    for (int c = 0; c < CW; ++c) {
                        const int n = n0 + c;
                        const unsigned b_lo = w[(size_t)(k / 2) * N + n];
                        const unsigned b_hi = w[(size_t)(k / 2 + 1) * N + n];
                        q[c] = ((unsigned)((int)(b_lo & 15) - 8) & 255)
                            | (((unsigned)((int)((b_lo >> 4) & 15) - 8) & 255) << 8)
                            | (((unsigned)((int)(b_hi & 15) - 8) & 255) << 16)
                            | (((unsigned)((int)((b_hi >> 4) & 15) - 8) & 255) << 24);
                    }
                }
#pragma unroll
                for (int r = 0; r < MT; ++r) {
                    if (m0 + r < M) {
                        const int a = *reinterpret_cast<const int*>(
                            x + (size_t)(m0 + r) * K + k);
#pragma unroll
                        for (int c = 0; c < CW; ++c) sums[r][c] = __dp4a((int)q[c], a, sums[r][c]);
                    }
                }
            }
        }
        for (; k < end; ++k) {
#pragma unroll
            for (int c = 0; c < CW; ++c) {
                const int n = n0 + c;
                if (n >= N) continue;
                const unsigned b = w[(size_t)(k / 2) * N + n];
                const int q = (int)((b >> ((k & 1) * 4)) & 15) - 8;
#pragma unroll
                for (int r = 0; r < MT; ++r) {
                    if (m0 + r < M) sums[r][c] += q * (int)x[(size_t)(m0 + r) * K + k];
                }
            }
        }
#pragma unroll
        for (int c = 0; c < CW; ++c) {
            const float scale = ws[(size_t)g * N + n0 + c];
#pragma unroll
            for (int r = 0; r < MT; ++r) total[r][c] += (float)sums[r][c] * scale;
        }
    }
#pragma unroll
    for (int c = 0; c < CW; ++c) {
        const int n = n0 + c;
        if (n >= N) continue;
#pragma unroll
        for (int r = 0; r < MT; ++r) {
            if (m0 + r < M) partial[((size_t)split * M + m0 + r) * N + n] = total[r][c];
        }
    }
}

// @brief 合并组分片后仅窄化一次，避免每组先舍入到 bf16。
//
// 每线程持 ``CW`` 个相邻列，与 ``w4a8_dot`` 的列宽一致；归约是每列一个线程，
// 所以它的块数由 ``N`` 而不是 ``N / CW`` 决定。
template <typename XT, int CW>
__global__ void w4a8_finish(const float* __restrict__ partial, const float* __restrict__ xs,
                            XT* __restrict__ out, int M, int N, int splits) {
    const int n0 = (blockIdx.x * blockDim.x + threadIdx.x) * CW;
    if (n0 >= N) return;
    const int m = blockIdx.y;
    const float xm = xs[m];
#pragma unroll
    for (int c = 0; c < CW; ++c) {
        const int n = n0 + c;
        if (n >= N) continue;
        float sum = 0.f;
        for (int s = 0; s < splits; ++s) sum += partial[((size_t)s * M + m) * N + n];
        out[(size_t)m * N + n] = CxScalar<XT>::from_float(sum * xm);
    }
}

//: 一个线程负责的输出列数上限。4 是一次对齐的 32 位加载覆盖的列数（每列 2 字节 packed），
//: 也正是 ``N = 2560``（GDN 的 qkv/z、o_proj 的输入宽）与 ``N = 512``（shared_expert）
//: 都能整除的值。``N % 4 != 0`` 的形状由 ``pick_cw`` 折回 1，所以正确性不依赖这个常量。
constexpr int W4A8_MAX_CW = 4;

//: 每块最大线程数。``N = 248320`` 在 256 线程下是 970 个块，已经远超 16 个 SM 的
//: 一个 wave，所以再大的块只会减少同时在飞的块数。
constexpr int W4A8_MAX_THREADS = 256;

// @brief 每块线程数，``0`` 表示自动。
//
// 与 ``gemv.cu`` 的 ``pick_threads`` 同一套分档，但上限是 256：``w4a8_dot`` 的一个
// 线程持 ``CW`` 列 x ``MT`` 行，寄存器压力比 bf16 的 GEMV 大，再往上会把占用率压回去。
//
// @param N 输出列数。
// @param cw 每线程列数。
// @return 每块线程数。
int pick_threads_w4(int N, int cw) {
    const int cols = (N + cw - 1) / cw;
    if (cols >= 2048) return 256;
    if (cols >= 512) return 128;
    return 64;
}

//: 宽加载（``width = 4``）自动开启的块数门槛。
//:
//: 判据是除以 width 之后还剩多少个块，不是 ``N`` 本身。实测两个方向相反的形状：
//:
//:   shape              N        width 1 块数   width 4 块数   width 4 结果
//:   gdn_in_proj_qkv    8192     32             8              输 1.71x
//:   lm_head            248320   970            243            赢 2.98x
//:
//: 16 SM 的部件在一个 wave 里能同时跑 16 个块。``width = 4`` 把块数除以 4，于是
//: 32 块降到 8 块——**掉到一个 wave 以下**，减掉的并行度比加宽省下的事务贵；而 970
//: 块降到 243 块仍然是 15 个 wave，少掉的并行度无关紧要，加宽省下的事务才是主要项。
//:
//: 门槛取 64 块（= 4 个 wave）：它在两个实测点之间（8 与 243），留了两个八度余量，
//: 而且它同时把 ``N < 16384`` 全部挡掉，于是 ``hyper_mix_up``（N = 10240）也不会误开。
constexpr int W4A8_WIDE_MIN_BLOCKS = 64;
// @brief 每个线程负责的相邻列数。
//
// ``width = 0``（自动）按 :data:`W4A8_WIDE_MIN_LANES` 判：活跃 lane 够多就开宽加载，
// 否则一列一线程。**这不是保守，是实测**：宽加载在宽网格上大赢、在窄网格上全输。
//
// ``gdn_in_proj_qkv``（N = 8192, K = 2560, M = 1）整张交叉网格，单位 us
// （``diag/qwen4/probe_q4_gemv_dispatch.py``，16 SM，每格都对过主机侧 fp64 参考）：
//
//   width      split 1   split 4   split 8   split 12   split 20   split 40
//   1           220.8      68.7      52.3      51.3       54.8       54.9
//   2           220.9      68.8      52.3      51.2       54.5       54.3
//   4           377.3     107.9      74.0      63.6       55.9       55.9
//   w4 / w1      1.71x     1.57x     1.41x     1.24x      1.02x      1.02x
//
// **``width = 4`` 在这个形状的每一格都输**，最多 1.71x。机制：一个线程持 4 列，网格
// 的块数就除以 4，而这张卡的瓶颈是**活跃 lane 数不足**（``N = 48`` 那格单独证过：
// split 1->32 涨的是活跃 lane，拿了 6.9x）。所以宽加载在窄网格上是**减 lane 换事务
// 宽度**，减掉的并行度比加宽省下的事务贵。
//
// ``lm_head``（N = 248320）方向相反，实测（同一网格，单位 us）：
//
//   width   split 1   split 4   split 8   split 12   split 16
//   1        2398.6    2368.7    2367.9    2372.3     2402.8
//   4         803.7     790.4     800.1     827.2      866.2
//   w4 / w1    2.98x     3.00x     2.96x     2.87x      2.77x
//
// 带宽地板 585.5 us，所以 width = 4 的 804 us 是地板的 73%，而 width = 1 的 2399 us
// 只有 24%。同一份权重、同一个算术，差 3 倍，唯一的变量是每线程列数。
//
// 显式 ``width`` 在 ``N`` 不是它的倍数时折回 ``1``：宽加载按 ``n0 = tid * CW`` 起算，
// 不整除时最后一列组会跨过行尾，读到的字节属于相邻行——形状全对、数值全错。
//
// @param N 输出列数。
// @param width 请求的每线程列数，``0`` 表示自动。
// @return ``1``、``2`` 或 ``4``，且大于 1 时总是 ``N`` 的约数。
int pick_cw(int N, int width) {
    int w = width;
    if (w <= 0) {
        // 自动：先按标量宽度算块数，只有当 ``width = 4`` 之后仍有
        // :data:`W4A8_WIDE_MIN_BLOCKS` 个块时才开宽加载。两个条件都必须成立——
        // ``N % 4 != 0`` 时宽加载会读跨行尾。
        const int blocks = (N + 255) / 256;
        const bool wide = ((N % W4A8_MAX_CW) == 0)
            && (blocks / W4A8_MAX_CW >= W4A8_WIDE_MIN_BLOCKS);
        w = wide ? W4A8_MAX_CW : 1;
    }
    if (w != 1 && w != 2 && w != W4A8_MAX_CW) return 1;
    if (w > 1 && (N % w) != 0) return 1;
    return w;
}

//: 分片数硬上限。归约那趟的代价随它线性增长，而每块持有的部分和是 ``8 * M * N``
//: 字节，所以 ``split`` 越大，分片缓冲区越可能把收益吃回去。
constexpr int W4A8_MAX_SPLIT = 320;

//: 分片数下限。它是 ``pick_split_w4`` 相对旧规则**唯一**的改动：旧规则的整数截断在
//: 块数超过 256 时给 0，取整后落到 1 片，而那时每个线程仍要沿 K 串行完整扫一遍。
//: ``lm_head``（970 个块）因此比带宽地板慢 4.1x。13 是「旧规则在 16 处封顶」与
//: 「实测最优落在 12-16」之间的最小值，取它不会改变任何中等宽度形状的行为。
constexpr int W4A8_MIN_SPLIT = 13;

// @brief 分片数，或 ``1`` 表示单趟。
//
// **这是原来那条规则，有意保留它**：``split = min(16, 256 / (blocks * row_tiles))``。
// 它看起来像随手写的，实际上在中等宽度的形状上恰好落在实测最优点附近，所以「优化」
// 掉它是净亏。实测（``diag/qwen4/probe_q4_gemv_dispatch.py``，M = 1，每格校验过）：
//
//   shape                 N        K     blocks(s256)  旧规则  实测最优  旧规则代价
//   gdn_in_proj_qkv       8192     2560      32           12      12       0%
//   shared_expert         512      2560       2          16      16       0%
//   shared_expert_down    2560     512        10         16      16       0%
//   lm_head               248320   2560      970          1      16      4.1x 慢
//
// 前三行「旧规则 = 实测最优」，所以它值得保留；``lm_head`` 那行是唯一被它坑的形状：
// 970 个块让 ``256 / 970 = 0``，取整后落到 1 片，而每个线程仍然要串行做完 20 组 x 32
// 次 dp4a，实测 2399 us 对上 585 us 的带宽地板（4.1x）。
//
// 所以这里只加一条：**分片数不低于 ``W4A8_MIN_SPLIT``**。它只影响「块数远超 wave 数」
// 的宽形状，对前三行一点不改。``lm_head`` 从 1 片变成 13 片之后，配合 ``width = 4``
// 是实测最优 804 us（独占 width = 1 时约 1450 us）。
//
// 为什么不干脆用「一组一片」（``split = groups``）：那会在 ``gdn_in_proj_qkv`` 上把
// 12 改成 20，实测 **慢 6%**（51.3 -> 54.8 us）。一个形状撑不起一条连续斜率，所以
// 不写进默认路径。
//
// 端到端验收（作业 ``q4disp6``，48 层 W4A8 EP=4，``COFFIN_HYPER_FUSED=off``）：
// ``w4a8_dot`` 59.9 -> 46.9 us/次、508 次合计 30.47 -> 23.79 ms，decode
// 270.85 -> 234.50 ms。同一份 .so 的双环境变量对照（作业 ``q4layer4``）显示末层
// logits 的 absmax 12.375 对 12.625、greedy ids 12/12 一致，差异是 fp32 部分和
// 累加顺序的必然结果（见 ``gemv.cu`` 的 ``pick_split`` 注释里同一份量化）。
//
// @param N 输出列数。
// @param K 归约长度。
// @param gs 分组大小。
// @param threads 每块线程数。
// @param rows 本次激活的行数块（``MT`` 的倍数），块数会乘上它。
// @return ``[1, W4A8_MAX_SPLIT]`` 里的分片数，且不超过组数。
int pick_split_w4(int N, int K, int gs, int threads, int row_tiles) {
    const int groups = (K + gs - 1) / gs;
    if (groups <= 1) return 1;
    const int blocks = threads > 0 ? (N + threads - 1) / threads : N;
    // 旧规则原样保留，包括它的整数截断——前三行的吻合正是由这个截断给出的。
    int want = 256 / (std::max(1, blocks) * std::max(1, row_tiles));
    if (want > 16) want = 16;
    if (want < W4A8_MIN_SPLIT) want = W4A8_MIN_SPLIT;
    if (want > groups) want = groups;
    if (want > W4A8_MAX_SPLIT) want = W4A8_MAX_SPLIT;
    (void)K;
    return want;
}

// @brief 每个分片内 K 循环的展开度。
//
// 这个核的 dp4a 主体一次迭代已经含 2 个 32 位字加载（8 字节）加一次激活的 32 位加载，
// 所以在飞加载数天然比 bf16 的 GEMV 多。``U = 1`` 是引入这个参数时的样子，也仍是
// 默认值：多展开只把同一个分片里的独立加载串得更长，而分片已经把串行长度压到 32 次
// 迭代了，再展开的收益要由真机扫描判定（见 ``diag/qwen4/probe_q4_gemv_dispatch.py``）。
//
// @param unroll 请求值，``0`` 表示默认。
// @return ``1``、``2`` 或 ``4``。
int pick_unroll_w4(int64_t unroll) {
    if (unroll == 2 || unroll == 4) return (int)unroll;
    return 1;
}

}  // namespace

// @brief 用调用方归约好的逐行 scale 量化激活，供行并行投影复用。
torch::Tensor quant_act_i8_scaled(torch::Tensor x, torch::Tensor out, torch::Tensor xs) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous()
                && x.size(0) <= 65535 && x.size(1) > 0 && x.size(1) <= INT_MAX,
                "contiguous CUDA [M,K] activation required");
    TORCH_CHECK(out.device() == x.device() && xs.device() == x.device()
                && out.sizes() == x.sizes() && out.scalar_type() == torch::kChar
                && out.is_contiguous() && xs.is_contiguous()
                && xs.dim() == 1 && xs.size(0) == x.size(0)
                && xs.scalar_type() == torch::kFloat, "quantization destination mismatch");
    TORCH_CHECK(x.scalar_type() == torch::kFloat || x.scalar_type() == torch::kBFloat16,
                "bf16 or fp32 activation required");
    if (!x.size(0)) return out;
    const c10::cuda::CUDAGuard guard(x.device());
    const auto stream = c10::cuda::getCurrentCUDAStream();
    if (x.scalar_type() == torch::kFloat)
        quant_with_scale<float><<<x.size(0), 256, 0, stream>>>(x.data_ptr<float>(),
            out.data_ptr<int8_t>(), xs.data_ptr<float>(), x.size(1));
    else
        quant_with_scale<uint16_t><<<x.size(0), 256, 0, stream>>>(
            reinterpret_cast<const uint16_t*>(x.data_ptr()), out.data_ptr<int8_t>(),
            xs.data_ptr<float>(), x.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// @brief 单趟内核：列宽和展开度都是编译期常量。
template <int CW, int U>
void launch_w4_dot_one(const int8_t* xp, const uint8_t* wp, const float* wsp, float* pp,
                       int M, int N, int K, int gs, int groups_per_split, int threads,
                       int row_tiles, int splits, bool mt1, cudaStream_t stream) {
    const dim3 grid((N / CW + threads - 1) / threads, row_tiles, splits);
    if (mt1) {
        w4a8_dot<1, CW, U><<<grid, threads, 0, stream>>>(xp, wp, wsp, pp, M, N, K, gs,
                                                        groups_per_split);
    } else {
        w4a8_dot<4, CW, U><<<grid, threads, 0, stream>>>(xp, wp, wsp, pp, M, N, K, gs,
                                                        groups_per_split);
    }
}

// @brief 把 ``(cw, unroll)`` 解析成编译期分派。
//
// 只有 ``cw = 1`` 和 ``cw = 4`` 有实例化，``unroll`` 也只用 1/2/4：把这个组合矩阵压到
// 一个 switch 里，才不用为每个形状都编一份核。
template <int CW>
void launch_w4_dot_cw(const int8_t* xp, const uint8_t* wp, const float* wsp, float* pp,
                      int M, int N, int K, int gs, int groups_per_split, int threads,
                      int row_tiles, int splits, bool mt1, int unroll,
                      cudaStream_t stream) {
    if (unroll == 2) {
        launch_w4_dot_one<CW, 2>(xp, wp, wsp, pp, M, N, K, gs, groups_per_split,
                                 threads, row_tiles, splits, mt1, stream);
    } else if (unroll == 4) {
        launch_w4_dot_one<CW, 4>(xp, wp, wsp, pp, M, N, K, gs, groups_per_split,
                                 threads, row_tiles, splits, mt1, stream);
    } else {
        launch_w4_dot_one<CW, 1>(xp, wp, wsp, pp, M, N, K, gs, groups_per_split,
                                 threads, row_tiles, splits, mt1, stream);
    }
}

// @brief 合并分片并窄化，块数按 ``N / CW`` 起。
template <typename XT>
void launch_w4_finish(const float* pp, const float* xsp, torch::Tensor& out, int M, int N,
                      int splits, int threads, int cw, cudaStream_t stream) {
    const int blocks = (N / cw + threads - 1) / threads;
    const dim3 grid(blocks, M);
    XT* op = out.scalar_type() == torch::kFloat ? reinterpret_cast<XT*>(out.data_ptr<float>())
                                                : reinterpret_cast<XT*>(out.data_ptr());
    if (cw == 4) {
        w4a8_finish<XT, 4><<<grid, threads, 0, stream>>>(pp, xsp, op, M, N, splits);
    } else {
        w4a8_finish<XT, 1><<<grid, threads, 0, stream>>>(pp, xsp, op, M, N, splits);
    }
}

// @brief 对分组 W4 与 A8 做投影，权重解包只发生在寄存器内。
// @param a [M,K] 连续 int8 激活。
// @param w [至少 ceil(K/2),N] 连续 uint8 packed 权重。
// @param xs [M] fp32 激活 scale。
// @param ws [ceil(K/group_size),N] fp32 权重 scale。
// @param out [M,N] 连续 bf16 或 fp32 输出。
// @param group_size K 轴权重分组大小。
// @param threads 每块线程数，``0`` 表示按 ``N`` 选。
// @param unroll 每个分片内 K 循环的展开度；``0`` 表示 1，即引入该参数之前的行为。
// @param split 分片数，``0`` 表示按 ``N``/``K`` 选；``1`` 强制单趟。
// @param width 一个线程负责的相邻输出列数，见 :func:`pick_cw`。
// @return out。
// @throws c10::Error 布局、设备、dtype、形状或整数累加范围不符合契约。
torch::Tensor w4a8_gemv(torch::Tensor a, torch::Tensor w, torch::Tensor xs,
                        torch::Tensor ws, torch::Tensor out, int64_t group_size,
                        int64_t threads, int64_t unroll, int64_t split, int64_t width) {
    TORCH_CHECK(a.is_cuda() && a.dim() == 2 && a.scalar_type() == torch::kChar,
                "activation must be CUDA int8 [M,K]");
    TORCH_CHECK(w.dim() == 2 && w.scalar_type() == torch::kByte,
                "weight must be uint8 [ceil(K/2),N]");
    const auto M = a.size(0), K = a.size(1), N = w.size(1);
    TORCH_CHECK(K > 0 && N > 0 && group_size > 0 && group_size <= 1048576,
                "invalid shape or group size");
    TORCH_CHECK(K <= INT_MAX && N <= INT_MAX && M <= 65535,
                "shape exceeds kernel index range");
    TORCH_CHECK(w.size(0) * 2 >= K && ws.dim() == 2
                && ws.size(0) == (K + group_size - 1) / group_size && ws.size(1) == N,
                "weight/scale shape mismatch");
    TORCH_CHECK(xs.dim() == 1 && xs.numel() == M && xs.scalar_type() == torch::kFloat
                && ws.scalar_type() == torch::kFloat, "scales must be fp32");
    TORCH_CHECK(out.dim() == 2 && out.size(0) == M && out.size(1) == N
                && (out.scalar_type() == torch::kFloat || out.scalar_type() == torch::kBFloat16),
                "output must be bf16/fp32 [M,N]");
    TORCH_CHECK(split <= W4A8_MAX_SPLIT, "split must be <= ", W4A8_MAX_SPLIT, ", got ", split);
    TORCH_CHECK(width == 0 || width == 1 || width == 2 || width == 4,
                "width must be 0 (auto) or one of 1, 2, 4; got ", width);
    for (const auto& t : {a, w, xs, ws, out}) {
        TORCH_CHECK(t.is_contiguous() && t.device() == a.device(),
                    "all tensors must be contiguous on the same CUDA device");
    }
    if (M == 0) return out;
    const c10::cuda::CUDAGuard guard(a.device());
    const int gs = (int)group_size;
    const int cw = pick_cw((int)N, (int)width);
    // 块数按 ``N / CW`` 起，所以列宽和线程数必须一起定：先定 ``cw`` 再定线程数，否则
    // 一个显式 ``threads`` 会让 ``N / CW`` 的商在不同机器上落到不同的块数上——扫描
    // 拿到的表就不能跨形状迁移。
    const int th = threads > 0 ? (int)threads : pick_threads_w4((int)N, cw);
    TORCH_CHECK(th > 0 && th <= W4A8_MAX_THREADS, "threads must be in [1, ",
                W4A8_MAX_THREADS, "], got ", th);
    const int u = pick_unroll_w4(unroll);
    const int rows_per_tile = M == 1 ? 1 : 4;
    const int row_tiles = ((int)M + rows_per_tile - 1) / rows_per_tile;
    const int groups = ((int)K + gs - 1) / gs;
    int splits = split > 0 ? (int)split : pick_split_w4((int)N, (int)K, gs, th, row_tiles);
    // 分片数不能超过组数：空分片会白付归约，而且 ``groups_per_split`` 向下取整之后
    // 尾部的组没人算——那是一个静默的错误答案，不是启动错误。
    if (splits > groups) splits = groups;
    if (splits < 1) splits = 1;
    const int groups_per_split = (groups + splits - 1) / splits;
    const float* xsp = xs.data_ptr<float>();
    const auto stream = c10::cuda::getCurrentCUDAStream();
    const int8_t* xp = a.data_ptr<int8_t>();
    const uint8_t* wp = w.data_ptr<uint8_t>();
    const float* wsp = ws.data_ptr<float>();

    // 分片缓冲区一次为所有分片定尺寸：归约需要在任何列能用 scale 之前每个分片都写好，
    // 所以它们不能共用输出缓冲区。``splits == 1`` 时它也走这里，用 ``[1, M, N]``
    // 的 fp32 暂存——省掉一次分支不值得让两条路径各有一份启动代码去跟内核同步。
    auto partial = torch::empty({splits, M, N}, xs.options());
    float* pp = partial.data_ptr<float>();
    if (cw == 4) {
        launch_w4_dot_cw<4>(xp, wp, wsp, pp, (int)M, (int)N, (int)K, gs,
                            groups_per_split, th, row_tiles, splits,
                            rows_per_tile == 1, u, stream);
    } else {
        launch_w4_dot_cw<1>(xp, wp, wsp, pp, (int)M, (int)N, (int)K, gs,
                            groups_per_split, th, row_tiles, splits,
                            rows_per_tile == 1, u, stream);
    }
    if (out.scalar_type() == torch::kFloat) {
        launch_w4_finish<float>(pp, xsp, out, (int)M, (int)N, splits, th, cw, stream);
    } else {
        launch_w4_finish<uint16_t>(pp, xsp, out, (int)M, (int)N, splits, th, cw, stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// 绑定写在内核旁边：名字、参数名与默认值只有这一份，`bindings.cpp` 只遍历注册表。
//调优旋钮全部带默认值 ``0``（= 沿用已发布的规则），所以不传它们时这一条与
//引入旋钮之前逐位相同。名字与 ``gemv_kn_i8`` 保持同一套：同一个概念在两个核上
//换名字，扫描脚本就没法共用一份形状表。
COFFIN_BIND(w4a8_gemv,
            "分组 W4 与 int8 激活的整数点积，int32 逐组累加",
            pybind11::arg("a"), pybind11::arg("w"), pybind11::arg("xs"), pybind11::arg("ws"),
            pybind11::arg("out"), pybind11::arg("group_size"), pybind11::arg("threads") = 0,
            pybind11::arg("unroll") = 0, pybind11::arg("split") = 0, pybind11::arg("width") = 0);
COFFIN_BIND(quant_act_i8_scaled,
            "按跨 rank 共享的逐行 scale 量化激活",
            pybind11::arg("x"), pybind11::arg("out"), pybind11::arg("xs"));
