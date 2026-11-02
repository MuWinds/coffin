// @file moe_grouped.cu
// @brief 稀疏专家的分组 W4A8 投影：每个专家只算路由到它的行。
//
// Module: coffin/backends/corex/kernels。由 CoreX 后端的分派层调用，
// ``models/qwen4exp`` 通过它跑 MoE 的两个投影。
//
// 为什么需要专门的内核：模型有 512 个专家、每 token 取 10 个，EP=4 之后每卡
// 持 128 个。decode 一步里每个专家平均只分到 1-2 行激活，若按「每 token 跑
// 10 次完整 GEMM」写，一个 2.4 MB 的专家权重会被读到 10 次——权重读取放大到
// token 数的倍数，而 decode 本来就是权重带宽受限。
//
// 按专家分组的形态下，每个专家的权重在一个步里只读一遍：所有路由到它的行
// 排在一起（调用方按专家号排序过），一次 GEMV 结构吃完整块。
//
// Constraints:
//   * ``x`` 必须是已按专家号排序的激活，``offsets`` 是 ``[E+1]`` 前缀和。
//     内核不排序、也不检查有序性：乱序的输入会算出看着合理的错数。
//   * 空专家直接跳过。``offsets[e] == offsets[e+1]`` 时该段零行，启动一个空块
//     只会白付一次调度。
//   * 每个 K 组先做 int32 累加再乘该组 scale；不同组的整数和不能直接相加
//     （``w4a8.cu`` 的约定）。
//   * 解包只在寄存器里发生，常驻权重始终是 uint8 packed，不分配整矩阵 int8 副本。
//   * 不用 CUDA graph、不用 cp.async（``platform/scow/AGENTS.md`` 第 1、5 条）。
//   * 分组编号用 ``blockIdx.z``（专家）与 ``blockIdx.x``（输出列）：一个三维网格
//     让"哪块算哪个专家"由硬件分发，不需要内核里查表。

#include "corex_ops.h"
#include "corex_scalar.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <climits>

namespace {

// @brief 一个专家的一段行 × 一个输出列块上做 W4A8 点积。
//
// 线程布局：``blockIdx.x`` 切输出列（``n``），``blockIdx.z`` 选专家，行在块内
// 由 ``blockDim.y`` 分摊。``MT`` 是每线程负责的行数，让一次权重读取服务多行——
// decode 期这几乎总是划算，因为同一专家的权重只读一次而激活很小。
//
// @param x       已排序的 int8 激活，``[tokens, K]``。
// @param xs       逐 token 的激活 scale，``[tokens]`` fp32。
// @param w        专家栈的 packed W4，``[E, ceil(K/2), N]``。
// @param ws       专家栈的 scale，``[E, groups, N]``。
// @param offsets  ``[E+1]`` 前缀和（设备指针，内核里读两次）。
// @param out      ``[tokens, N]`` bf16 或 fp32。
// @param K, N, group_size 几何。
// @param experts  专家数，用来给空专家提前退出。
template <typename XT>
__global__ void moe_grouped_kernel(const int8_t* x, const float* xs,
                                   const uint8_t* w, const float* ws,
                                   const int* offsets, XT* out,
                                   int K, int N, int gs, int experts) {
    const int expert = blockIdx.z;
    if (expert >= experts) return;
    const int begin = offsets[expert];
    const int end = offsets[expert + 1];
    const int rows = end - begin;
    if (rows <= 0) return;

    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= N) return;

    const int groups = (K + gs - 1) / gs;
    // 专家栈按 [E, ceil(K/2), N] 连续存，所以第 e 个专家的 packed 从
    // expert * half_k * N 起；scale 从 expert * groups * N 起。
    const uint8_t* wexp = w + (size_t)expert * (size_t)((K + 1) / 2) * (size_t)N;
    const float* sexp = ws + (size_t)expert * (size_t)groups * (size_t)N;

    // 沿 K 的组循环，逐组做 dp4a 再乘组 scale。同一组的整数和留在 int32 里，
    // 换成 float 只发生在乘 scale 之后——提前转换会让 4 位量化的误差累积。
    for (int r = blockIdx.y * blockDim.y; r < rows; r += gridDim.y * blockDim.y) {
        const int token = begin + r;
        const int8_t* xrow = x + (size_t)token * K;
        float total = 0.f;
        for (int g = 0; g < groups; ++g) {
            int sums = 0;
            int k = g * gs;
            const int stop = min(K, k + gs);
            // dp4a 的四字节必须沿 K 配对：两个 packed 字节各藏两个 4 位权重，
            // 拼成一个字的顺序错了会拿相邻输出列的权重去配对。
            for (; (K & 3) == 0 && (k & 3) == 0 && k + 4 <= stop; k += 4) {
                const unsigned lo = wexp[(size_t)(k / 2) * N + col];
                const unsigned hi = wexp[(size_t)(k / 2 + 1) * N + col];
                const unsigned q = ((unsigned)((int)(lo & 15) - 8) & 255)
                    | (((unsigned)((int)(lo >> 4) - 8) & 255) << 8)
                    | (((unsigned)((int)(hi & 15) - 8) & 255) << 16)
                    | (((unsigned)((int)(hi >> 4) - 8) & 255) << 24);
                const int a = *reinterpret_cast<const int*>(xrow + k);
                sums = __dp4a((int)q, a, sums);
            }
            for (; k < stop; ++k) {
                const unsigned byte = wexp[(size_t)(k / 2) * N + col];
                const int q = (int)((byte >> ((k & 1) * 4)) & 15) - 8;
                sums += q * (int)xrow[k];
            }
            total += (float)sums * sexp[(size_t)g * N + col];
        }
        out[(size_t)token * N + col] = CxScalar<XT>::from_float(total * xs[token]);
    }
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
// @return out。
// @throws c10::Error 布局、设备、dtype、形状或前缀和不自洽。
torch::Tensor moe_grouped(torch::Tensor a, torch::Tensor xs, torch::Tensor w,
                          torch::Tensor ws, torch::Tensor offsets,
                          torch::Tensor out, int64_t group_size) {
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

    const c10::cuda::CUDAGuard guard(a.device());
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    // 输出列一个线程；行方向再铺一层，让同一专家的权重读取服务多行。
    const int threads_x = 128;
    const int threads_y = 2;
    const dim3 block(threads_x, threads_y);
    const dim3 grid((unsigned)((N + threads_x - 1) / threads_x),
                    (unsigned)std::min(8, tokens),
                    (unsigned)experts);
    if (out.scalar_type() == torch::kBFloat16) {
        moe_grouped_kernel<uint16_t><<<grid, block, 0, stream>>>(
            a.data_ptr<int8_t>(), xs.data_ptr<float>(), w.data_ptr<uint8_t>(),
            ws.data_ptr<float>(), offsets.data_ptr<int>(), 
            reinterpret_cast<uint16_t*>(out.data_ptr()), K, N, (int)group_size,
            experts);
    } else {
        moe_grouped_kernel<float><<<grid, block, 0, stream>>>(
            a.data_ptr<int8_t>(), xs.data_ptr<float>(), w.data_ptr<uint8_t>(),
            ws.data_ptr<float>(), offsets.data_ptr<int>(), out.data_ptr<float>(),
            K, N, (int)group_size, experts);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
