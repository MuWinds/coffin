// @file kv_store_device.inc
// @brief decode 阶段写 KV 的两个设备内核：稠密 ``[nkv, max_seq, hd]`` 缓存与 paged 池。
//
// Module: coffin/backends/corex/kernels。被两条前端各包含一次：``attn.cu``（torch
// 扩展）与 ``abi/kv_store_abi.cu``（``extern "C"`` 出口）。同一份设备代码。
//
// 为什么它们值得先搬：这两条是「同一个槽位号、四次写」被合成一次发射的产物。稠密缓存
// 那条替掉全注意力层每层的四次 ``index_copy_``（10 层 = 每步 40 次发射）；paged 那条
// 替掉 ``full_like`` 加四次 ``index_put_``（每步 50 次）。搬走的字节只有 ``nkv * hd``。
//
// Constraints:
//   * 槽位号在**设备上读**：``pos1`` 本来就是 CUDA 张量，取回主机会多一次 D2H 同步，
//     而 decode 一步 10 层正好是这条路径想消除的东西。因此带 ``slot64`` 开关。
//   * 四个目的地都要求**连续**：地址按 ``[nkv, max_seq, hd]`` 线性算，带间距会写错单元。
//   * ``kv_pool_store`` 是**可失败**的加速路径：形状不合返回非 0，调用方回退逐张量写回。

#ifndef COFFIN_KV_STORE_DEVICE_INC
#define COFFIN_KV_STORE_DEVICE_INC

#include <cstdint>

__global__ void kv_store_kernel(const int8_t* __restrict__ k,
                                const int8_t* __restrict__ v,
                                const float* __restrict__ ks,
                                const float* __restrict__ vs,
                                const void* __restrict__ slot_in,
                                int slot64, int hd, int max_seq,
                                int8_t* __restrict__ kc,
                                int8_t* __restrict__ vc,
                                float* __restrict__ ksc,
                                float* __restrict__ vsc) {
    const int h = (int)blockIdx.x;
    const int slot = slot64 ? (int)reinterpret_cast<const long long*>(slot_in)[0]
                            : reinterpret_cast<const int*>(slot_in)[0];
    const size_t dst = ((size_t)h * (size_t)max_seq + (size_t)slot) * (size_t)hd;
    const size_t src = (size_t)h * (size_t)hd;
    for (int i = (int)threadIdx.x; i < hd; i += (int)blockDim.x) {
        kc[dst + i] = k[src + i];
        vc[dst + i] = v[src + i];
    }
    if (threadIdx.x == 0) {
        ksc[(size_t)h * (size_t)max_seq + (size_t)slot] = ks[h];
        vsc[(size_t)h * (size_t)max_seq + (size_t)slot] = vs[h];
    }
}

__global__ void kv_pool_store_kernel(int layer, const void* __restrict__ slot_in, int slot64,
                                     const int8_t* __restrict__ k,
                                     const int8_t* __restrict__ v,
                                     const float* __restrict__ ks,
                                     const float* __restrict__ vs,
                                     int nslots, int head_major,
                                     int8_t* __restrict__ kc, int8_t* __restrict__ vc,
                                     float* __restrict__ ksc, float* __restrict__ vsc) {
    const int h = (int)blockIdx.x;
    const int hd = (int)blockDim.x;
    const int slot = slot64 ? (int)reinterpret_cast<const long long*>(slot_in)[0]
                            : reinterpret_cast<const int*>(slot_in)[0];
    const long long cell = head_major
        ? (long long)((layer * gridDim.x + h) * nslots + slot)
        : (long long)(((long long)layer * nslots + slot) * gridDim.x + h);
    const long long dst = cell * hd;
    const long long src = (long long)h * hd;
    for (int i = (int)threadIdx.x; i < hd; i += (int)blockDim.x) {
        kc[dst + i] = k[src + i];
        vc[dst + i] = v[src + i];
    }
    if (threadIdx.x == 0) {
        ksc[cell] = ks[h];
        vsc[cell] = vs[h];
    }
}

#endif  // COFFIN_KV_STORE_DEVICE_INC
