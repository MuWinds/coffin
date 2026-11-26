// @file attn_reduce_device.inc
// @brief 块内归约：``block_reduce``。被 ``attn_device.inc`` 与 ``attn_slice_device.inc`` 共用。
//
// Module: coffin/backends/corex/kernels（设备侧）。这两个使用方共用同一份实现，因为
// 归约的**顺序**决定了结果：attn 的在线 softmax 把 ``lsum`` 按块累进来，写方和读方
// 若用了不同的结合顺序，长序列上会算出最后一个 token 的分布差别，而形状检查看不见。
//
// Constraints:
//   * 块大小必须是 2 的幂：循环从 ``blockDim.x >> 1`` 起对半砍，192 线程的块会在
//     96/48/24/12/6/3 之后丢掉尾巴。使用方因此都按行宽定块大小（``kv_quant_threads``）。
//   * ``red`` 是调用方给的共享内存，长度不小于 ``blockDim.x``；两个使用方都取
//     ``extern __shared__ float red[]``。
//   * 归约后所有通道都能读到结果，所以调用方不必再广播一次。
//   * 每次调用内部自带屏障；调用方在写入 ``red`` 之前**不能**有未完成的依赖。

#ifndef COFFIN_ATTN_REDUCE_DEVICE_INC
#define COFFIN_ATTN_REDUCE_DEVICE_INC

#include <cuda_runtime.h>

// @brief 把 ``val`` 在整块内归约，结果对块内所有通道可见。
//
// @param val    本通道的值。
// @param red    共享内存暂存，长度 >= ``blockDim.x``。
// @param is_max ``true`` 取最大（amax），``false`` 求和。
// @return 归约结果，每个通道拿到同一个值。
__device__ __forceinline__ float block_reduce(float val, float* red, bool is_max) {
    const int t = (int)threadIdx.x;
    const int n = (int)blockDim.x;
    red[t] = val;
    __syncthreads();
    for (int off = n >> 1; off > 0; off >>= 1) {
        if (t < off) {
            red[t] = is_max ? fmaxf(red[t], red[t + off]) : (red[t] + red[t + off]);
        }
        __syncthreads();
    }
    return red[0];
}

#endif  // COFFIN_ATTN_REDUCE_DEVICE_INC
