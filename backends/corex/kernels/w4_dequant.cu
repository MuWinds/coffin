// @file w4_dequant.cu
// @brief 把 packed W4 权重**一趟**反量化成 bf16，供大 M 的厂商 GEMM 使用。
//
// Module: coffin/backends/corex/kernels。绑定成 ``w4_dequant``，由
// ``backends/corex/__init__.py`` 的 ``linear_prefill`` 在 ``rows >= 阈值`` 时调用，
// 之后走 ``torch.matmul`` 的厂商 bf16 GEMM。
//
// 为什么需要它：``w4_gemm_tcu`` 在稠密 prefill 上占内核级账 31%（397.5 ms / 506 次），
// 而它的四类改法都被实测否证（tile 64/256、LDS 双缓冲 + BK32、thin-M）。换方向后
// 三段实测给出闭环：
//
//     厂商 bf16 GEMM   q_proj 512x12288x2560   1026.6 us / 8.05 TFLOP/s   ← 合格
//     w4_gemm_tcu      同形状                  1934.8 us                   ← 被比下去
//     torch 层反量化    同形状                  2219-2315 us / ~35 GB/s      ← 唯一缺口
//
// torch 慢的原因不是布局（用真实布局 ``[K/2, N]`` 重测仍是 35 GB/s），而是它把这个操作
// 拆成 ``and/shift/to/stack/reshape/sub/mul`` 七八趟独立 elementwise，每趟都把 79 MB
// 完整读写一遍。**合成一趟就是这个内核存在的全部理由。**
//
// 账（按 400 GB/s 保守估，全 prefill 128 个投影）：packed 1.125 GB + scale 0.28 GB +
// 输出 bf16 2.25 GB ≈ 3.66 GB → 9.2 ms；剩余交给厂商 bf16 GEMM（8.5 TFLOP/s）约 133 ms
// ⇒ 合计约 142 ms，对现在 397.5 ms 是 2.8 倍。
//
// Constraints:
//   * **布局契约**（``w4a8_weights.py`` 强校验）：``packed`` 是 ``[ceil(K/2), N]``
//     ——**K 在行、N 在列**；``scale`` 是 ``[ceil(K/group), N]``。不要按 ``[N, K]`` 理解。
//   * **半字节序与零点**（``w4a8.cu`` 实证）：低半字节 = 偶 k、高半字节 = 奇 k、零点 8。
//     写成 ``(b >> ((k & 1) * 4)) & 15`` 再减 8 的那一条是唯一事实来源，本文件的
//     ``(q >> 4)`` 取高半字节必须与它一致，否则权重的相邻两行会被交换（数值看着合理）。
//   * 纯带宽内核：**无 tile、无共享内存、无 mma、无跨线程依赖**。任何分块优化在这里
//     都只会增加复杂度而不提高带宽。
//   * 输出走 ``CxScalar<uint16_t>::from_float``（就近取偶），与其它内核的窄化逐位一致。
//   * **一次调用只活一份 ``[K, N]`` bf16**（最大 63 MB），不缓存、不常驻：128 份一起
//     常驻是 4.5 GB，会把显存打爆。
//   * 每次启动都在 ``c10::cuda::getCurrentCUDAStream()`` 上。

#include "corex_bindings.h"
#include "corex_ops.h"
#include "corex_scalar.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <climits>
#include <cstdint>

namespace {

//: 每线程处理的 N 宽度。取 4 是因为 ``uchar4`` 一次正好读四列、写两个 8 字节。
//:
//: 相邻线程的 ``col`` 相差 4，所以块内 256 个线程覆盖 1024 个连续字节：packed 读、
//: scale 读、输出写三路都是合并访存。换 8 或 2 都不会更快（``uchar4`` 已是最宽事务）。
constexpr int kDequantCols = 4;

//: 块线程数。256 是「一个块覆盖 1024 列」与「占用率」的折中。
constexpr int kDequantThreads = 256;

// @brief 一趟反量化：每线程 (2 个 K) × 4 个 N。
//
// @param packed ``[ceil(K/2), N]`` uint8，每字节两个 4 位权重（低=偶 k、高=奇 k）。
// @param scale  ``[ceil(K/group), N]`` fp32，逐 (group, n)。
// @param out    ``[K, N]`` bf16 位模式（``uint16_t``），行主序。
// @param n, k   形状；``k`` 允许为奇数，此时最后一行的奇 k 越界不写。
// @param group  分组大小（128 是默认，但不写死）。
__global__ void w4_dequant_kernel(const uint8_t* __restrict__ packed,
                                 const float* __restrict__ scale,
                                 uint16_t* __restrict__ out,
                                 int n, int k, int group) {
    const int row = (int)blockIdx.y;                 // packed 行号 = k / 2
    const int col = ((int)blockIdx.x * kDequantThreads + (int)threadIdx.x) * kDequantCols;
    // ``n % 4 == 0`` 由 host 保证，所以这里只需判断整线程是否越界。
    if (col >= n) return;

    const int k0 = row * 2;                          // 偶 k
    const int k1 = k0 + 1;                           // 奇 k
    const bool has_k1 = (k1 < k);

    // 一次 4 字节读拿 8 个权重：4 列 × 2 个 k。
    const uchar4 q = *reinterpret_cast<const uchar4*>(packed + (size_t)row * n + col);
    // 逐字段取，不要把局部 ``unsigned char[4]`` reinterpret 成 ``uchar4``：局部数组只有
    // 1 字节对齐，那样写是一次可能未对齐的 4 字节 store。全局读那边是 4 字节对齐的
    // （``col`` 与 ``n`` 都是 4 的倍数），安全。
    const unsigned char qb[4] = {q.x, q.y, q.z, q.w};

    // 两个 k 通常同组（group 是 2 的倍数时只有跨界那一行不同），同组就复用一份 scale，
    // 省一半 scale 读——scale 的 DRAM 量不大（约 packed 的 6%），但它是**指令**成本的大头。
    const float* sp0 = scale + (size_t)(k0 / group) * n + col;
    const float4 s0 = *reinterpret_cast<const float4*>(sp0);
    const int g1 = (k1 / group);
    const float4 s1 = (g1 == (k0 / group))
                          ? s0
                          : *reinterpret_cast<const float4*>(scale + (size_t)g1 * n + col);
    const float sc0[4] = {s0.x, s0.y, s0.z, s0.w};
    const float sc1[4] = {s1.x, s1.y, s1.z, s1.w};

    uint16_t v0[4];
    uint16_t v1[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        // 低半字节 = 偶 k，高半字节 = 奇 k，零点 8（与 w4a8.cu 的取法逐位一致）。
        const int lo = (int)(qb[j] & 15) - 8;
        const int hi = (int)(qb[j] >> 4) - 8;
        v0[j] = CxScalar<uint16_t>::from_float((float)lo * sc0[j]);
        v1[j] = has_k1 ? CxScalar<uint16_t>::from_float((float)hi * sc1[j]) : (uint16_t)0;
    }

    // 每行 4 个 bf16 = 8 字节，一次 ``uint2`` 写出去，避免被拆成 4 次 2 字节store。
    uint2 pack0;
    pack0.x = (unsigned)v0[0] | ((unsigned)v0[1] << 16);
    pack0.y = (unsigned)v0[2] | ((unsigned)v0[3] << 16);
    *reinterpret_cast<uint2*>(out + (size_t)k0 * n + col) = pack0;
    if (has_k1) {
        uint2 pack1;
        pack1.x = (unsigned)v1[0] | ((unsigned)v1[1] << 16);
        pack1.y = (unsigned)v1[2] | ((unsigned)v1[3] << 16);
        *reinterpret_cast<uint2*>(out + (size_t)k1 * n + col) = pack1;
    }
}

}  // namespace

torch::Tensor w4_dequant(torch::Tensor packed, torch::Tensor scale, int64_t k, int64_t n,
                         int64_t group) {
    TORCH_CHECK(packed.is_cuda() && packed.dim() == 2, "packed must be 2-D CUDA");
    TORCH_CHECK(scale.is_cuda() && scale.dim() == 2, "scale must be 2-D CUDA");
    TORCH_CHECK(packed.scalar_type() == torch::kByte, "packed must be uint8");
    TORCH_CHECK(scale.scalar_type() == torch::kFloat, "scale must be fp32");
    TORCH_CHECK(packed.is_contiguous() && scale.is_contiguous(),
                "packed and scale must be contiguous");
    TORCH_CHECK(k > 0 && n > 0 && group > 0 && k <= INT_MAX && n <= INT_MAX,
                "sizes must be positive and fit int32");
    // ``packed`` 的行数**允许多于** ``ceil(K/2)``：``W4A8Weight.__init__`` 的契约只要求
    // ``packed.shape[0] * 2 >= k``，多出来的行是量化时的对齐填充。这里只读前 ``ceil(K/2)`` 行，
    // 而 ``packed`` 是连续的 ``[rows, N]``，行距恒为 ``N``，所以填充行不影响寻址。
    //
    // 写成严格相等会让 ``K = 320`` 这类「半数半字节 + 填充」的投影直接抛错 —— 实测在
    // ``gated_residual`` 的 ``input_mix_weight_up`` 上就是这样崩掉的。
    TORCH_CHECK((int64_t)packed.size(0) * 2 >= k && packed.size(1) == n,
                "packed must be [rows, N] with rows * 2 >= K");
    TORCH_CHECK((int64_t)scale.size(0) == (k + group - 1) / group && scale.size(1) == n,
                "scale must be [ceil(K/group), N]");
    // 一列四字节的事务宽度决定了这个约束；不满足就走不通这条内核（调用方应先回退）。
    TORCH_CHECK((n & 3) == 0, "N must be a multiple of 4 for the vectorized path");

    const c10::cuda::CUDAGuard guard(packed.device());
    auto out = torch::empty({k, n}, packed.options().dtype(torch::kBFloat16));
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const dim3 block(kDequantThreads, 1);
    const dim3 grid((unsigned)((n + kDequantCols * kDequantThreads - 1)
                               / (kDequantCols * kDequantThreads)),
                    (unsigned)((k + 1) / 2));
    w4_dequant_kernel<<<grid, block, 0, stream>>>(
        packed.data_ptr<uint8_t>(), scale.data_ptr<float>(),
        reinterpret_cast<uint16_t*>(out.data_ptr()), (int)n, (int)k, (int)group);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// 绑定写在内核旁边：名字、参数名与默认值只有这一份，`bindings.cpp` 只遍历注册表。
COFFIN_BIND(w4_dequant,
            "packed W4 权重一趟反量化成 bf16（大 M 的厂商 GEMM 前置）",
            pybind11::arg("packed"), pybind11::arg("scale"), pybind11::arg("k"), pybind11::arg("n"),
            pybind11::arg("group"));
