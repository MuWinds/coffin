// @file moe_epilogue.cu
// @brief 共享专家的门标量：一次启动算完 ``sigmoid(x·w)`` 并把它乘到投影输出上。
//
// Module: coffin/kernels。由 backends/corex 作为 torch 扩展加载，只由 decode 的
// ``SparseMoE._shared`` 调用。
//
// Constraints:
//   * 归约走共享内存树，绝不用 ``__shfl_down_sync``：本部件的 warp 是 64 通道，
//     为 32 写的蝶形只加一半通道就返回一个看着合理但错误的标量。
//   * ``blockDim.x`` 必须是 2 的幂，树从 ``blockDim.x`` 开始对半砍。
//   * ``out`` 由调用方预分配、按形状复用。写成「调用方传 out」而不是原地缩放，
//     是因为投影的输出缓冲归 ``ops/gemv`` 所有，原地改它会静默污染下一次读取。
//   * 两次窄化是契约：参考链是 ``(x.float() * w).sum(-1).to(bf16)`` 再 ``sigmoid``，
//     所以点积先窄化到存储 dtype、sigmoid 的输出再窄化一次。门是逐 token 一个标量，
//     少窄化一次就是整行按另一条曲线缩放，且不会让任何形状检查失败。

#include "corex_ops.h"
#include "corex_scalar.h"

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

// @brief 逐行算 ``sigmoid(x·w)`` 并把它乘到 ``src`` 的副本上写进 ``out``。
//
// @param x     ``[rows, width]`` 或 ``[width]``，最内层步长为 1。
// @param w     ``[width]`` fp32 权重，连续。
// @param src   ``[rows, width]`` 或 ``[width]``，与 ``x`` 同 dtype，只读。
// @param out   同形状的目的地，允许与 ``src`` 重叠。
// @param width 每行的列数 ``H``。
// @param x_pitch / s_pitch / o_pitch 三个操作数各自的行间距。
// @return 无；``out`` 归调用方所有。
// @throws 设备侧不抛。包装层在主机侧校验布局与 dtype。
template <typename T>
__global__ void shared_gate_scale_kernel(const T* __restrict__ x, const float* __restrict__ w,
                                        const T* __restrict__ src, T* __restrict__ out,
                                        int width, long x_pitch, long s_pitch, long o_pitch) {
    // 定长共享内存：块大小是主机侧的常量 256，写成动态申请在这张卡的编译器上会给出
    // ``invalid device function``（模板核 + extern __shared__ 的已知问题），定长则没有。
    __shared__ float tree[256];
    const long r = blockIdx.x;
    const T* __restrict__ xr = x + r * x_pitch;
    const T* __restrict__ sr = src + r * s_pitch;
    T* __restrict__ orow = out + r * o_pitch;
    float acc = 0.f;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        acc += CxScalar<T>::to_float(xr[i]) * w[i];
    }
    tree[threadIdx.x] = acc;
    __syncthreads();
    for (int step = blockDim.x >> 1; step > 0; step >>= 1) {
        if ((int)threadIdx.x < step) {
            tree[threadIdx.x] += tree[threadIdx.x + step];
        }
        __syncthreads();
    }
    const float dotted = CxScalar<T>::to_float(CxScalar<T>::from_float(tree[0]));
    const float gate = CxScalar<T>::to_float(
        CxScalar<T>::from_float(1.f / (1.f + __expf(-dotted))));
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        orow[i] = CxScalar<T>::from_float(CxScalar<T>::to_float(sr[i]) * gate);
    }
}

torch::Tensor shared_gate_scale(torch::Tensor x, torch::Tensor w, torch::Tensor src,
                               torch::Tensor out) {
    TORCH_CHECK(w.scalar_type() == torch::kFloat, "gate weight must be fp32");
    TORCH_CHECK(x.scalar_type() == src.scalar_type() && x.scalar_type() == out.scalar_type(),
                "x, src and out must share a dtype");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 || x.scalar_type() == torch::kFloat,
                "operands must be bf16 or fp32");
    TORCH_CHECK(x.sizes() == src.sizes() && x.sizes() == out.sizes(),
                "x, src and out must have the same shape");
    TORCH_CHECK(x.stride(-1) == 1 && src.stride(-1) == 1 && out.stride(-1) == 1,
                "operands must have unit innermost stride");
    TORCH_CHECK(w.numel() == x.size(-1), "gate weight width must match the row width");
    TORCH_CHECK(w.is_contiguous(), "gate weight must be contiguous");
    const int width = (int)x.size(-1);
    if (width == 0) return out;
    // 一维操作数就是一行：decode 传的是 ``[H]``，间距与宽度重合、行数是 1。
    const long rows = x.dim() <= 1 ? 1 : (long)x.numel() / width;
    const long x_pitch = x.dim() <= 1 ? (long)width : (long)x.stride(-2);
    const long s_pitch = src.dim() <= 1 ? (long)width : (long)src.stride(-2);
    const long o_pitch = out.dim() <= 1 ? (long)width : (long)out.stride(-2);
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    // 与内核里的 ``tree[256]`` 是同一个常量：改这里必须同时改那里。
    const int threads = 256;
    const dim3 grid((unsigned)rows);
    if (x.scalar_type() == torch::kBFloat16) {
        shared_gate_scale_kernel<uint16_t><<<grid, threads, 0, stream>>>(
            reinterpret_cast<const uint16_t*>(x.data_ptr()), w.data_ptr<float>(),
            reinterpret_cast<const uint16_t*>(src.data_ptr()),
            reinterpret_cast<uint16_t*>(out.data_ptr()), width, x_pitch, s_pitch, o_pitch);
    } else {
        shared_gate_scale_kernel<float><<<grid, threads, 0, stream>>>(
            x.data_ptr<float>(), w.data_ptr<float>(), src.data_ptr<float>(),
            out.data_ptr<float>(), width, x_pitch, s_pitch, o_pitch);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
