// @file moe_grouped_i8.cu
// @brief 稀疏专家的**分组 int8** 投影：一次启动吃完整栈，行按 ``offsets`` 分段。
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
// 其中 ``e`` 由 ``t`` 落在 ``[offsets[e], offsets[e+1])`` 决定。权重按 ``[E, K, N]``
// （K 轴在外、N 连续）存放，与 ``gemv_kn_i8`` 的 ``kn`` 布局一致，所以不需要任何转置：
// 一个 warp 在固定 k 上读到的 128 字节正好是一个整事务。
//
// Constraints:
//   * 整数和可加是这条路径能用 int32 部分和的前提：scale 是逐 (专家, 输出列) 的、与
//     k 无关，所以分片可以先把整数和加起来、最后一次性乘两个 scale。W4A8 那边做不到
//     （它的 scale 逐 (组, 列)，每片必须各乘一次），这也是这里不需要 part 存 fp32 的原因。
//   * ``__dp4a`` 的原子粒度是 4 个 k。K 不是 4 的倍数时尾部走标量；**不能**用补齐的方式
//     凑整，那会把下一行的字节算进这一行，得到一个看着合理的错数。
//   * 一行一个块，块数就是 ``tokens``。多行时同一专家的权重会被多读一遍，所以调用方
//     只在 decode 那一档进来；内核本身对任意行数都成立，不需要额外的形状校验。
//   * 激活先由 ``quant_act_i8`` 按行量化（absmax/127，下限 1e-5），与预填充路径同一套
//     约定，所以两处的量化误差口径一致。
//   * 不用 CUDA graph、不用 cp.async（``platform/scow/AGENTS.md`` 第 1、5 条）：两者在
//     这张卡上都能过编译，然后静默挂死。
//   * 全程不用 fp64（本设备对它返回 0，见 ``docs/qwen4/COREX-PLATFORM.md``）。

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
                                      XT* __restrict__ out,
                                      int K, int N, int experts, int tokens,
                                      int slices, int per) {
    // 本片负责的 k 区间由 ``blockIdx.z`` 直接算出，不从 host 传：四片各收到一份
    // 「全区间的 k0/k1」时，每片都会把整个 K 算一遍并各写一份部分和，归约把同一个
    // 数加 ``slices`` 遍 —— 输出翻整数倍，而且只在多片时才错。
    const int k0 = (int)blockIdx.z * per;
    const int k1 = min(K, k0 + per);
    const int row = (int)blockIdx.y;
    const int col = (int)blockIdx.x * I8_COLS + (int)threadIdx.x;
    if (col >= N) return;

    // 行属于哪个专家只让 thread 0 查，其余线程同时把激活行搬进共享内存：两件事互不
    // 依赖，串起来做会白付一次「八个依赖读」的延迟。激活行只有 K 字节（2048），
    // 放共享内存后每个 k 步的读都是一次广播，而不是 128 个线程各读一遍全局。
    __shared__ int s_expert;
    extern __shared__ int32_t s_pad[];
    int8_t* s_x = reinterpret_cast<int8_t*>(s_pad);
    if (threadIdx.x == 0) s_expert = i8_owner(offsets, experts, row);
    const int8_t* arow = a + (size_t)row * K;
    for (int i = (int)threadIdx.x; i < K; i += (int)blockDim.x) s_x[i] = arow[i];
    __syncthreads();

    const int expert = s_expert;
    const int8_t* wexp = w + (size_t)expert * (size_t)K * (size_t)N;

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

}  // namespace

// @brief 分组 int8 投影：``[tokens, K] -> [tokens, N]``，每个专家一份权重。
//
// @param a       int8 激活，``[tokens, K]`` 连续 CUDA。
// @param xs      逐 token 激活 scale，``[tokens]`` fp32 连续。
// @param w       专家栈，``[E, K, N]`` 连续 int8（K 轴在外、N 连续）。
// @param ws      专家栈 scale，``[E, N]`` 连续 fp32。
// @param offsets ``[E+1]`` 前缀和，int32 连续，``offsets[0] == 0``。
// @param out     ``[tokens, N]`` 连续，bf16 或 fp32，dtype 由它决定。
// @param slices  K 方向分片数；``<= 0`` 表示按块数自动选。
// @return out。
// @throws c10::Error 布局、设备、dtype 或形状不自洽。
//
// 分片的作用是换并行度：一行一个块时块数是 ``ceil(N/128) x tokens``，decode 一步
// 只有 8 行、门/升栈 N=1024，于是 64 个块 —— 在这张卡上远不够填。切 8 片后 512 块，
// 每片仍有 256 个 k，展开循环不会退化成标量尾巴。
torch::Tensor moe_grouped_i8(torch::Tensor a, torch::Tensor xs, torch::Tensor w,
                             torch::Tensor ws, torch::Tensor offsets,
                             torch::Tensor out, int64_t slices) {
    TORCH_CHECK(a.is_cuda() && a.dim() == 2 && a.scalar_type() == torch::kChar
                && a.is_contiguous(), "activation must be contiguous CUDA int8 [tokens, K]");
    TORCH_CHECK(xs.is_cuda() && xs.dim() == 1 && xs.scalar_type() == torch::kFloat
                && xs.is_contiguous() && xs.size(0) == a.size(0),
                "activation scale must be contiguous fp32 [tokens]");
    TORCH_CHECK(w.is_cuda() && w.dim() == 3 && w.scalar_type() == torch::kChar
                && w.is_contiguous(), "weight must be contiguous CUDA int8 [E, K, N]");
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
    TORCH_CHECK((int)w.size(1) == K && (int)w.size(2) == N,
                "weight must be [E, K, N] matching activation and destination");
    TORCH_CHECK((int)ws.size(0) == experts && (int)ws.size(1) == N,
                "weight scale must be [E, N]");
    TORCH_CHECK((int)offsets.size(0) == experts + 1, "offsets must have E+1 entries");
    TORCH_CHECK(K > 0 && N > 0 && experts > 0, "sizes must be positive");
    if (tokens == 0) return out;

    const int cols = (N + I8_COLS - 1) / I8_COLS;
    int span = (int)slices;
    if (span <= 0) {
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
    span = std::max(1, std::min(std::min(span, I8_MAX_SLICES), K));
    // 每片长度按 4 对齐：``__dp4a`` 的粒度是 4 个 k，片头不对齐会让主循环整个跳过。
    const int per = ((K + span - 1) / span + 3) / 4 * 4;

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

    const dim3 grid((unsigned)cols, (unsigned)tokens, (unsigned)span);
    const dim3 block(I8_COLS);
    const size_t smem = (size_t)K * sizeof(int8_t);
    auto launch = [&](auto tag) {
        using XT = decltype(tag);
        XT* dst = reinterpret_cast<XT*>(out.data_ptr());
        moe_grouped_i8_kernel<XT><<<grid, block, smem, stream>>>( 
            a.data_ptr<int8_t>(), w.data_ptr<int8_t>(), ws.data_ptr<float>(),
            xs.data_ptr<float>(), offsets.data_ptr<int>(), part_ptr, dst,
            K, N, experts, tokens, span, per);
        if (span > 1) {
            const dim3 rgrid((unsigned)cols, (unsigned)tokens);
            moe_grouped_i8_reduce_kernel<XT><<<rgrid, block, 0, stream>>>( 
                part_ptr, ws.data_ptr<float>(), xs.data_ptr<float>(),
                offsets.data_ptr<int>(), dst, N, experts, tokens, span);
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
// @return out。
// @throws c10::Error 布局、设备、dtype 或形状不自洽。
//
// 为什么至少 2 片：1 片时主体核直接把结果写成 bf16 写到 ``out[row]``，而这里只有一行
// 目的地，归并就得多一趟读改写。抬到 2 片让主体核一律走 ``part``，收尾核顺手做完加权
// 求和；down（K=512）自动选片本来就会选到 3 片，这条只影响调用方硬指定 1 片的情况。
torch::Tensor moe_grouped_i8_merge(torch::Tensor a, torch::Tensor xs, torch::Tensor w,
                                   torch::Tensor ws, torch::Tensor offsets,
                                   torch::Tensor rw, torch::Tensor out, int64_t slices) {
    TORCH_CHECK(a.is_cuda() && a.dim() == 2 && a.scalar_type() == torch::kChar
                && a.is_contiguous(), "activation must be contiguous CUDA int8 [rows, K]");
    TORCH_CHECK(xs.is_cuda() && xs.dim() == 1 && xs.scalar_type() == torch::kFloat
                && xs.is_contiguous() && xs.size(0) == a.size(0),
                "activation scale must be contiguous fp32 [rows]");
    TORCH_CHECK(w.is_cuda() && w.dim() == 3 && w.scalar_type() == torch::kChar
                && w.is_contiguous(), "weight must be contiguous CUDA int8 [E, K, N]");
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
    TORCH_CHECK((int)w.size(1) == K && (int)w.size(2) == N,
                "weight must be [E, K, N] matching activation and destination");
    TORCH_CHECK((int)ws.size(0) == experts && (int)ws.size(1) == N,
                "weight scale must be [E, N]");
    TORCH_CHECK((int)offsets.size(0) == experts + 1, "offsets must have E+1 entries");
    TORCH_CHECK(K > 0 && N > 0 && experts > 0 && rows > 0, "sizes must be positive");

    const int cols = (N + I8_COLS - 1) / I8_COLS;
    int span = (int)slices;
    if (span <= 0) {
        span = 1;
        while (span < I8_MAX_SLICES
               && cols * rows * (span + 1) <= 1024
               && K / (span + 1) >= I8_MIN_PER) {
            ++span;
        }
    }
    span = std::max(2, std::min(std::min(span, I8_MAX_SLICES), K));
    const int per = ((K + span - 1) / span + 3) / 4 * 4;

    const c10::cuda::CUDAGuard guard(a.device());
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // 每一片都覆盖全部 (row, col)，末尾列块的越界线程在收尾核里同样被 ``col >= N``
    // 挡住，所以这里用 ``empty`` 而不是 ``zeros``。
    torch::Tensor part = torch::empty({(int64_t)span, (int64_t)rows, (int64_t)N},
                                      out.options().dtype(torch::kInt));
    int32_t* part_ptr = part.data_ptr<int32_t>();

    const dim3 grid((unsigned)cols, (unsigned)rows, (unsigned)span);
    const dim3 block(I8_COLS);
    const size_t smem = (size_t)K * sizeof(int8_t);
    auto launch = [&](auto tag) {
        using XT = decltype(tag);
        XT* dst = reinterpret_cast<XT*>(out.data_ptr());
        // 主体核的 ``tokens`` 参数就是这里的 ``rows``；``part`` 的步长与之一致。
        moe_grouped_i8_kernel<XT><<<grid, block, smem, stream>>>(
            a.data_ptr<int8_t>(), w.data_ptr<int8_t>(), ws.data_ptr<float>(),
            xs.data_ptr<float>(), offsets.data_ptr<int>(), part_ptr, dst,
            K, N, experts, rows, span, per);
        const dim3 mgrid((unsigned)cols);
        moe_grouped_i8_merge_kernel<XT><<<mgrid, block, 0, stream>>>(
            part_ptr, ws.data_ptr<float>(), xs.data_ptr<float>(),
            offsets.data_ptr<int>(), rw.data_ptr<float>(), dst, N, experts, rows, span);
    };
    if (out.scalar_type() == torch::kBFloat16) launch(uint16_t{});
    else launch(float{});
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}