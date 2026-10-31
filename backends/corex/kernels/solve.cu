// @file solve.cu
// @brief 单位下三角求解，分块 delta 规则唯一需要的库操作。
//
// Module: coffin/backends/corex/kernels。绑定成 ``triangular_solve``，由
// ``ops/delta.py`` 经 ``backends.ops`` 调用。它存在是因为本卡的 cuSOLVER 用不了：torch
// 的 wheel 加载 ``/opt/sw_home/local/cuda/lib64/libcusolver.so``，而集群上没有这个路径
// （``/usr/local/corex/lib64/libcusolver.so`` 是有的，但没人读它），所以每条
// ``torch.linalg`` 求解路径都在 ``dlopen`` 里失败。ROCm 构建上同一个算子是一次库调用；
// 在这里只能是自己的。
//
// 为什么不用 Python 的 ``bmm`` 链：分块 delta 规则每层每个分块要解一次，一个 prompt 上
// 24 个线性注意力层就是几百次调用，每次都要从 Python 发 4-8 次启动，外加它们之间的
// 带步长拷贝。
//
// Constraints:
//   * **没有屏障。** 前向替换在右端项各列之间是独立的：``x[i][c]`` 只依赖 ``j < i``
//     的 ``x[j][c]``，所以持有一列的线程从不需要别的线程的工作。整个求解是一个线程本地
//     循环，没有 ``__syncthreads()``，这也意味着它不对 warp 宽度作任何假设——本部件的
//     warp 是 64 通道，写成归约形状的实现就得考虑这一点。
//   * 运行中的解保存在共享内存里且是 **fp32**，即使操作数是 bf16；只有写出的那一行被
//     窄化。替换是一条链：第 ``i`` 行消费它之前的每一行，所以第 0 行的一次舍入会传播
//     到全部 ``L`` 行。参考实现按操作数 dtype 求解，所以这边是更精确的一侧，而一致性
//     表允许这个方向。
//   * 对角线按 1 处理，上三角永不读取。delta 规则的 ``UT`` 是 ``I`` 加一个严格下三角
//     乘积，所以存下来的对角线会构成另一个系统，去读它会让答案依赖调用方从未写过的
//     内存。
//   * 批步长显式传入，而不是假定连续。调用方的右端项是
//     ``torch.eye(L).expand(bn, L, L)``，它的批步长是 0：一个单位块被所有系统共享。
//     实体化它要每层每个分块拷 ``bn * L * L`` 个元素，只为读一个 ``L * L`` 块；而读取
//     步长恰好表达了这件事，还不需要特例。
//   * 共享内存是 ``L * blockDim.x`` 个 float，所以 ``L`` 受每块上限约束。主机侧包装层
//     直接拒绝装不下的形状，而不是启动一个读穿自身分配的内核。
//   * ``out`` 是新分配，从不与输入重叠，所以调用方可以把同一个张量同时当 ``lower`` 和
//     ``rhs`` 传进来，求解过程不会把它中途改坏。

#include "corex_ops.h"
#include "corex_scalar.h"

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

namespace {

//: 一个块解多少条右端项列。64 是本部件的 wavefront，所以一个块的通道把 ``lower`` 的一行
//: 当成单次广播读下来，共享内存的步长也保持自然值。
constexpr int MAX_COLS = 64;

//: 动态共享内存上限。48 KB 是本部件的分界：低于它不需要额外的声明属性，所以待在它下面
//: 就能无条件启动。
constexpr size_t MAX_SMEM = 48u * 1024u;

// @brief 为单位下三角 ``A`` 解 ``A X = B``，每条右端项列一个线程。
//
// @param lower [bn, L, L] 系数；只读严格下三角部分，对角线按 1 处理。
// @param rhs [bn, L, n] 右端项。
// @param out [bn, L, n] 解，按 ``rhs`` 的 dtype 写出。
// @param L 系统阶数。
// @param n 右端项列数。
// @param ldb ``lower`` 的批步长，以元素计。
// @param ldr ``rhs``/``out`` 的批步长，以元素计。
// @param stride ``blockDim.x``：共享内存的行间距，即补齐后的线程数。
// @param cols 每块的有效列数。
template <typename T>
__global__ void solve_kernel(const T* __restrict__ lower, const T* __restrict__ rhs,
                             T* __restrict__ out, int L, int n, long ldb, long ldr,
                             int stride, int cols) {
    const int b = blockIdx.x;
    const int lane = (int)threadIdx.x;
    const int c = blockIdx.y * cols + lane;
    // 下面没有屏障，所以补齐出来的通道可以直接离开，不必等任何人。
    if (c >= n) return;

    extern __shared__ float xs[];       // [L][stride] 本列运行中的解
    const T* a = lower + (size_t)b * ldb;
    const T* rhs_b = rhs + (size_t)b * ldr;
    T* out_b = out + (size_t)b * ldr;

    for (int i = 0; i < L; ++i) {
        float acc = CxScalar<T>::to_float(rhs_b[(size_t)i * n + c]);
        const T* row = a + (size_t)i * L;
        // 第 ``i`` 行由每个通道在同一偏移上读取，所以这次加载经只读路径广播出去；共享
        // 内存的读取则是按通道的、合并的。
        for (int j = 0; j < i; ++j) {
            acc -= CxScalar<T>::to_float(row[j]) * xs[(size_t)j * stride + lane];
        }
        xs[(size_t)i * stride + lane] = acc;
        out_b[(size_t)i * n + c] = CxScalar<T>::from_float(acc);
    }
}

// @brief 按一种操作数 dtype 启动求解。
//
// @param lower [bn, L, L] 系数矩阵。
// @param rhs [bn, L, n] 右端项。
// @param out [bn, L, n] 目的地。
// @param bn 独立系统的批大小。
// @param L 阶数。
// @param n 右端项列数。
// @param ldb ``lower`` 的批步长。
// @param ldr ``rhs``/``out`` 的批步长。
// @param stream 启动所用的流；必须是当前 torch 流，否则一次捕获会把它丢掉。
template <typename T>
void launch_solve(const torch::Tensor& lower, const torch::Tensor& rhs, torch::Tensor& out,
                  int bn, int L, int n, long ldb, long ldr, cudaStream_t stream) {
    const int cols = n < MAX_COLS ? n : MAX_COLS;
    const int threads = ((cols + 63) / 64) * 64;
    const size_t smem = (size_t)L * threads * sizeof(float);
    const dim3 grid(bn, (n + cols - 1) / cols);
    solve_kernel<T><<<grid, threads, smem, stream>>>(
        reinterpret_cast<const T*>(lower.data_ptr()),
        reinterpret_cast<const T*>(rhs.data_ptr()), reinterpret_cast<T*>(out.data_ptr()),
        L, n, ldb, ldr, threads, cols);
}

}  // namespace

// @brief 为单位下三角 ``A`` 解 ``A X = B``。
//
// @param lower [bn, L, L] 单位下三角系数，bf16 或 fp32。
// @param rhs [bn, L, n] 右端项，dtype 相同。
// @return [bn, L, n]，dtype 同 ``rhs``。
// @throws c10::Error 形状不一致、dtype 没有对应实例化、操作数不是行连续，或 ``L`` 装不
//         进共享内存——这几种要么会读穿分配，要么拒绝启动，而主机侧是调用方唯一能看见
//         它的地方。
torch::Tensor triangular_solve(torch::Tensor lower, torch::Tensor rhs) {
    TORCH_CHECK(lower.dim() == 3 && rhs.dim() == 3,
                "lower and rhs must both be 3-D: [bn, L, L] and [bn, L, n]");
    TORCH_CHECK(lower.scalar_type() == rhs.scalar_type(),
                "lower and rhs must share a dtype");
    TORCH_CHECK(lower.scalar_type() == torch::kBFloat16
                || lower.scalar_type() == torch::kFloat,
                "operands must be bf16 or fp32");
    const int bn = (int)rhs.size(0);
    const int L = (int)rhs.size(1);
    const int n = (int)rhs.size(2);
    TORCH_CHECK(lower.size(0) == bn && lower.size(1) == L && lower.size(2) == L,
                "lower must be [bn, L, L] matching the right-hand side");
    // 只固定行布局：最内层步长必须是 1 才有合并的共享读取，而批步长从张量里读取。这样
    // 就接纳了 ``eye(L).expand(bn, L, L)``（批步长为 0），且不必拷贝它。
    TORCH_CHECK(rhs.stride(2) == 1 && lower.stride(2) == 1,
                "operands must have unit innermost stride");
    TORCH_CHECK(rhs.stride(1) == n && lower.stride(1) == L,
                "operands must be row-contiguous within a batch entry");
    const int cols = n < MAX_COLS ? n : MAX_COLS;
    const int threads = ((cols + 63) / 64) * 64;
    TORCH_CHECK((size_t)L * threads * sizeof(float) <= MAX_SMEM,
                "L=", L, " needs ", (size_t)L * threads * sizeof(float),
                " bytes of shared memory, over the ", MAX_SMEM,
                " byte limit; reduce the delta chunk length rather than widening it");

    auto out = torch::empty({bn, L, n}, rhs.options());
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const long ldb = (long)lower.stride(0);
    const long ldr = (long)rhs.stride(0);
    if (lower.scalar_type() == torch::kBFloat16) {
        launch_solve<uint16_t>(lower, rhs, out, bn, L, n, ldb, ldr, stream);
    } else {
        launch_solve<float>(lower, rhs, out, bn, L, n, ldb, ldr, stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
