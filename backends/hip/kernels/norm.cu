// @file norm.cu
// @brief 融合的 RMSNorm 与带门控的 RMSNorm。
//
// Module: coffin/kernels。由 ``hip_ops.py`` 绑定的层归一化入口；从模型层调用，绝不从
// ``ops/`` 调用。
//
// 为什么融合：这套栈上的 ``F.rms_norm`` 会拆成一个归约（平方均值）加若干次独立的 rsqrt
// / mul 内核，而引擎每个解码步要调它 105 次（in_ln、post_ln、q_norm、k_norm、最终
// 归一化和带门控的线性注意力归一化），折合约占该步的 2.6 ms。每行一个 block 把整件事
// 融在一起：平方和、rsqrt、乘权重一次走完。
//
// Constraints:
//   * 一行一个 block，所以归一化宽度必须放得进 block 限制；线程数不写死，由 N 经
//     ``rms_threads`` 选出。
//   * ``x`` 的行是带间距的：q_norm 跑在更宽缓冲区的一个 ``[nh, hd]`` 切片上，所以行距是
//     参数，不能假定它等于 N。
//   * 进出都是 bf16，均值和 rsqrt 在 fp32 里累加；用 bf16 求和在量级小的项上会丢精度。
//   * 归约是 warp shuffle 加一次跨 warp 合并（两次 ``__syncthreads()``），不是共享内存
//     归约树：这些内核经常只跑一个 block，那种情况下同步链就是主要开销。

#include "hip_ops.h"

#include <c10/cuda/CUDAStream.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace {

__device__ __forceinline__ float bf16f(unsigned short h) {
    return __uint_as_float(((unsigned)h) << 16);
}

__device__ __forceinline__ unsigned short fbf16(float f) {
    unsigned u = __float_as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);   // 就近舍入到偶数（round to nearest even）
    return (unsigned short)(u >> 16);
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down(v, o);
    return v;
}

// 先做 warp 内归约，再做一次跨 warp 合并：两次 __syncthreads()，而不是 256 宽共享内存
// 归约树需要的八次。这些内核经常跑在*单个* block 上（解码时隐含维度那几个归一化就是
// 一行），同步链因此成了主要开销——78 次调用每次 11 us，占一个 int8 解码步的 6%。
// ``red`` 至少要能放 32 个 float，且会被覆盖。
__device__ __forceinline__ float block_sum(float v, float* red) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarp = (blockDim.x + 31) >> 5;
    v = warp_sum(v);
    if (lane == 0) red[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = lane < nwarp ? red[lane] : 0.f;
        v = warp_sum(v);
        if (lane == 0) red[0] = v;
    }
    __syncthreads();
    return red[0];
}

// 一行就是一个 block 的宽度；256 线程只有在大 N 时才划算，而带门控的归一化是
// [32, 128]（一半线程空转），q_norm/k_norm 则是 [nh, 256]。向上取整到 warp 的整数倍，
// 是为了让 ``block_sum`` 的跨 warp 合并有良好定义。
__host__ int rms_threads(int N) {
    return N < 256 ? ((N + 31) / 32) * 32 : 256;
}

// @brief y = x * rsqrt(mean(x^2) + eps) * w，每行一个 block。
//
// @param x [M, N] bf16，行距为 ``stride``（更宽缓冲区的一个切片）。
// @param w [N] bf16 权重。
// @param y [M, N] 紧密排布的 bf16 输出。
// @param eps 在平方根倒数之前加进 rsqrt 内部。
__global__ void rms_norm_kernel(const unsigned short* __restrict__ x,
                                const unsigned short* __restrict__ w,
                                unsigned short* __restrict__ y,
                                int N, long stride, float eps) {
    const int row = blockIdx.x;
    const int t = threadIdx.x;
    const unsigned short* xr = x + (size_t)row * stride;
    float ss = 0.f;
    for (int i = t; i < N; i += blockDim.x) {
        const float v = bf16f(xr[i]);
        ss += v * v;
    }
    __shared__ float red[32];
    const float inv = rsqrtf(block_sum(ss, red) / (float)N + eps);
    unsigned short* yr = y + (size_t)row * N;
    for (int i = t; i < N; i += blockDim.x)
        yr[i] = fbf16(bf16f(xr[i]) * inv * bf16f(w[i]));
}

// @brief y = rms(x) * w * silu(gate)，所有行紧密排布。
//
// @param x,w,gate [M, N] bf16 连续存放；SiLU 加在归一化乘权重之后，所以 gate 不影响
//        归一化。
// @param y [M, N] bf16 输出。
__global__ void rms_norm_gated_kernel(const unsigned short* __restrict__ x,
                                      const unsigned short* __restrict__ w,
                                      const unsigned short* __restrict__ gate,
                                      unsigned short* __restrict__ y,
                                      int N, float eps) {
    const int row = blockIdx.x;
    const int t = threadIdx.x;
    const unsigned short* xr = x + (size_t)row * N;
    float ss = 0.f;
    for (int i = t; i < N; i += blockDim.x) {
        const float v = bf16f(xr[i]);
        ss += v * v;
    }
    __shared__ float red[32];
    const float inv = rsqrtf(block_sum(ss, red) / (float)N + eps);
    unsigned short* yr = y + (size_t)row * N;
    const unsigned short* gr = gate + (size_t)row * N;
    for (int i = t; i < N; i += blockDim.x) {
        const float gv = bf16f(gr[i]);
        const float silu = gv / (1.f + __expf(-gv));
        yr[i] = fbf16(bf16f(xr[i]) * inv * bf16f(w[i]) * silu);
    }
}

}  // namespace

// @brief 带行间距的融合 RMSNorm：y = x * rsqrt(mean(x^2) + eps) * w。
//
// @param x [M, N] bf16，只读。
// @param w [N] bf16 权重。
// @param stride ``x`` 的行距；紧密张量传 N，归一化切片（q_norm / k_norm）时传更宽的
//        间距。
// @param eps 在 rsqrt 之前加上；它是 checkpoint 的 epsilon，不是调优旋钮。
// @return [M, N] bf16，新分配且紧密排布。
torch::Tensor rms_norm(torch::Tensor x, torch::Tensor w, int64_t stride, double eps) {
    const int M = x.size(0), N = x.size(1);
    auto y = torch::empty({M, N}, x.options());
    hipStream_t stream = at::cuda::getCurrentCUDAStream();
    rms_norm_kernel<<<M, rms_threads(N), 0, stream>>>(
        reinterpret_cast<const unsigned short*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const unsigned short*>(w.data_ptr<at::BFloat16>()),
        reinterpret_cast<unsigned short*>(y.data_ptr<at::BFloat16>()),
        N, (long)stride, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

// @brief 带门控的 RMSNorm：y = rms(x) * w * silu(gate)。
//
// @param x,w,gate [M, N] bf16，每一行都紧密排布（没有 stride 参数；带门控的归一化只在
//        连续张量上被调用）。
// @return [M, N] bf16，新分配。
torch::Tensor rms_norm_gated(torch::Tensor x, torch::Tensor w, torch::Tensor gate,
                             double eps) {
    const int M = x.size(0), N = x.size(1);
    auto y = torch::empty({M, N}, x.options());
    hipStream_t stream = at::cuda::getCurrentCUDAStream();
    rms_norm_gated_kernel<<<M, rms_threads(N), 0, stream>>>(
        reinterpret_cast<const unsigned short*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const unsigned short*>(w.data_ptr<at::BFloat16>()),
        reinterpret_cast<const unsigned short*>(gate.data_ptr<at::BFloat16>()),
        reinterpret_cast<unsigned short*>(y.data_ptr<at::BFloat16>()),
        N, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
