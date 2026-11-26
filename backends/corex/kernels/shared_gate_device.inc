// @file shared_gate_device.inc
// @brief 共享专家门标量的设备代码：``sigmoid(x . w)`` 的树归约与按行缩放。
//
// Module: coffin/backends/corex/kernels。被两条前端各包含一次：``moe_epilogue.cu``
// （torch 扩展，``SparseMoE._shared`` 走的那条）与 ``abi/moe_epilog_abi.cu``
// （``extern "C"`` 出口）。同一份设备代码，两条前端逐位相同。
//
// Constraints:
//   * 归约走共享内存树，绝不用 ``__shfl_down_sync``：本部件的 warp 是 64 通道，
//     为 32 写的蝶形只加一半通道就返回一个看着合理但错误的标量。
//   * ``blockDim.x`` 必须是 2 的幂，树从 ``blockDim.x`` 开始对半砍；``tree`` 的定长
//     上界与主机侧常量 ``SGATE_THREADS`` 必须一致。
//   * 定长共享内存而不是动态申请：模板核 + ``extern __shared__`` 在这张卡的编译器上
//     会给出 ``invalid device function``。
//   * 两次窄化是契约：参考链是 ``(x.float() * w).sum(-1).to(bf16)`` 再 ``sigmoid``，
//     所以点积先窄化到存储 dtype、sigmoid 的输出再窄化一次。

#ifndef COFFIN_SHARED_GATE_DEVICE_INC
#define COFFIN_SHARED_GATE_DEVICE_INC

#include "corex_scalar.h"

#include <cstdint>

//: 门标量内核的块大小。与 ``tree`` 的定长上界是同一个常量。
#define SGATE_THREADS 256

// @brief 逐行算 ``sigmoid(x·w)`` 并把它乘到 ``src`` 的副本上写进 ``out``。
//
// @param x     ``[rows, width]`` 或 ``[width]``，最内层步长为 1。
// @param w     ``[width]`` fp32 权重，连续。
// @param src   ``[rows, width]`` 或 ``[width]``，与 ``x`` 同 dtype，只读。
// @param out   同形状的目的地，允许与 ``src`` 重叠。
// @param width 每行的列数 ``H``。
// @param x_pitch / s_pitch / o_pitch 三个操作数各自的行间距。
template <typename T>
__global__ void shared_gate_scale_kernel(const T* __restrict__ x, const float* __restrict__ w,
                                        const T* __restrict__ src, T* __restrict__ out,
                                        int width, long x_pitch, long s_pitch, long o_pitch) {
    __shared__ float tree[SGATE_THREADS];
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

#endif  // COFFIN_SHARED_GATE_DEVICE_INC
