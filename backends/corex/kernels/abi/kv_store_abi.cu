// @file kv_store_abi.cu
// @brief decode 写 KV 的两个 C ABI 出口：``coffin_kv_store`` / ``coffin_kv_pool_store``。
//
// Module: coffin/backends/corex/kernels/abi。设备代码来自 ``../kv_store_device.inc``——
// 与 ``../attn.cu`` 是同一份，两条前端逐位相同。
//
// 为什么先搬这两条：它们都是「同一个槽位号、四次写」合成的产物。稠密缓存那条替掉全
// 注意力层每层的四次 ``index_copy_``（10 层 = 每步 40 次发射）；paged 池那条替掉
// ``full_like`` 加四次 ``index_put_``（每步 50 次）。搬的字节只有 ``nkv * hd``。
//
// Constraints:
//   * 签名与 ``cxrt/attn.py`` 的 ``argtypes`` 必须逐字一致：多一个参数就整体错位。
//   * 槽位号在设备上读，``slot64`` 选 int32 / int64 的读法：取回主机会多一次 D2H
//     同步，而 decode 一步 10 层正好是这条路径想消除的东西。
//   * 稠密那条的四个目的地必须连续，地址按 ``[nkv, max_seq, hd]`` 线性算。
//   * ``kv_pool_store`` **可失败**：形状不合返回非 0（调用方据此回退逐张量写回），
//     所以它的校验不写 ``err`` 抛错语义，只回一个码；``coffin_kv_store`` 则相反，
//     形状不对就该报错，不静默降级。

#include "cx_tensor.h"

#include <cstdint>

#include "../kv_store_device.inc"

// @brief 稠密 ``[nkv, max_seq, hd]`` 缓存的一次写入。
//
// @param k ``[nkv, hd]`` int8 key。
// @param v 同形状的 value。
// @param ks ``[nkv]`` fp32 key 每头 scale。
// @param vs 同形状的 value scale。
// @param slot ``[1]`` int32 或 int64 的槽位号，设备上读。
// @param kc ``[nkv, max_seq, hd]`` int8 缓存，原地写入。
// @param vc 同上。
// @param ksc ``[nkv, max_seq]`` fp32，原地写入。
// @param vsc 同上。
// @param nkv 头数。
// @param hd 每头维度。
// @param max_seq 缓存槽位数。
// @param slot64 槽位号是否 int64。
// @param stream 目标流；空指针表示默认流。
// @param err 错误缓冲，可为空。
// @param errlen 错误缓冲字节数。
// @return ``0`` 成功。
extern "C" int coffin_kv_store(const void* k, const void* v, const void* ks, const void* vs,
                               const void* slot, void* kc, void* vc, void* ksc, void* vsc,
                               int nkv, int hd, int max_seq, int slot64, void* stream,
                               char* err, int errlen) {
    const cx::Err e{err, errlen};
    if (!k || !v || !ks || !vs || !slot || !kc || !vc || !ksc || !vsc) {
        return cx::fail(e, "k, v, scales, slot and all caches must be non-null");
    }
    if (nkv <= 0 || hd <= 0 || max_seq <= 0) {
        return cx::fail(e, "nkv/hd/max_seq must be positive, got %d/%d/%d", nkv, hd, max_seq);
    }
    const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    const int threads = hd < 64 ? 64 : (hd > 512 ? 512 : hd);
    kv_store_kernel<<<nkv, threads, 0, s>>>(
        reinterpret_cast<const int8_t*>(k), reinterpret_cast<const int8_t*>(v),
        reinterpret_cast<const float*>(ks), reinterpret_cast<const float*>(vs), slot, slot64, hd,
        max_seq, reinterpret_cast<int8_t*>(kc), reinterpret_cast<int8_t*>(vc),
        reinterpret_cast<float*>(ksc), reinterpret_cast<float*>(vsc));
    return cx::finish(e);
}

// @brief paged 池的一次写入。**可失败**：形状或层号不合返回非 0，不写错误缓冲。
//
// @param k ``[nkv, hd]`` int8 key 行。
// @param v 同形状的 value 行。
// @param ks ``[nkv]`` fp32 key scale。
// @param vs 同形状的 value scale。
// @param slot ``[1]`` int32 或 int64 槽位号，设备上读。
// @param kc ``[layers, slots, nkv, hd]`` 或 ``[layers, nkv, slots, hd]`` int8 池。
// @param vc 同上。
// @param ksc 池的 scale（单元号无 ``hd`` 后缀）。
// @param vsc 同上。
// @param nkv 头数。
// @param hd 每头维度。
// @param nslots 池的槽位数。
// @param layer 池层轴下标。
// @param head_major 池是否按 ``[layers, nkv, slots, hd]`` 排。
// @param slot64 槽位号是否 int64。
// @param stream 目标流。
// @param err 错误缓冲，可为空（本出口只在参数为空时写）。
// @param errlen 错误缓冲字节数。
// @return ``0`` 成功；``1`` 参数不合法（调用方应回退）。
extern "C" int coffin_kv_pool_store(const void* k, const void* v, const void* ks, const void* vs,
                                    const void* slot, void* kc, void* vc, void* ksc, void* vsc,
                                    int nkv, int hd, int nslots, int layer, int head_major,
                                    int slot64, void* stream, char* err, int errlen) {
    const cx::Err e{err, errlen};
    if (!k || !v || !ks || !vs || !slot || !kc || !vc || !ksc || !vsc) {
        return cx::fail(e, "k, v, scales, slot and all pool buffers must be non-null");
    }
    if (nkv <= 0 || hd <= 0 || nslots <= 0 || layer < 0) return 1;
    const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    kv_pool_store_kernel<<<nkv, hd, 0, s>>>(
        layer, slot, slot64, reinterpret_cast<const int8_t*>(k),
        reinterpret_cast<const int8_t*>(v), reinterpret_cast<const float*>(ks),
        reinterpret_cast<const float*>(vs), nslots, head_major,
        reinterpret_cast<int8_t*>(kc), reinterpret_cast<int8_t*>(vc),
        reinterpret_cast<float*>(ksc), reinterpret_cast<float*>(vsc));
    return cx::finish(e);
}
