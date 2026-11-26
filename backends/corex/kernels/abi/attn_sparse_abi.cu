// @file attn_sparse_abi.cu
// @brief QSA 稀疏注意力的 C ABI 出口：``coffin_attn_sparse``，无 torch。
//
// Module: coffin/backends/corex/kernels/abi。设备代码来自
// ``../attn_sparse_device.inc``——与 ``../attn_sparse.cu`` 是同一份内核与启动器。
//
// 几何**全部由调用方给**（``rows`` 不在 q 的形状里读：ABI 面只拿裸指针，没有形状）。
// 校验也因此在调用方做一遍、这里再做一遍便宜的：这里只挡「会让内核读越界」的那些，
// 例如 ``hd`` 不是 32 的倍数、``nh`` 不是 ``nkv`` 的倍数。
//
// Constraints:
//   * 签名与 ``cxrt/attn.py`` 的 ``argtypes`` 必须逐字一致：多一个参数就整体错位。
//   * ``sel_idx`` / ``sel_len`` / ``ctx_row`` 是**设备**指针；主机指针会读到垃圾。
//   * ``hd`` 必须是 32 的倍数且 ``<= 32 * ATT_SPARSE_LANES``（256）。

#include "cx_tensor.h"

#include <cstdint>

#include "../attn_sparse_device.inc"

// @brief 稀疏 QSA 注意力：只读索引器选中的位置，KV 不 repeat。
//
// @param q [rows, nh, hd] bf16。
// @param kc,vc [nkv, max_seq, hd] bf16 头主序。
// @param sel_idx [rows, max_sel] int32，行内升序、无效为 -1。
// @param sel_len [rows] int32，每行有效位置数。
// @param ctx [rows] int32，每行的因果上界。
// @param out [rows, nh, hd] bf16，原地写入。
// @param scale 打分缩放。
// @param rows, nh, nkv, hd, max_seq, max_sel 几何。
// @param stream 目标流；空指针表示默认流。
// @param err 错误缓冲，可为空。
// @param errlen 错误缓冲字节数。
// @return ``0`` 成功。
extern "C" int coffin_attn_sparse(const void* q, const void* kc, const void* vc,
                                  const void* sel_idx, const void* sel_len, const void* ctx,
                                  void* out, double scale, int rows, int nh, int nkv, int hd,
                                  int max_seq, int max_sel, void* stream, char* err, int errlen) {
    const cx::Err e{err, errlen};
    if (!q || !kc || !vc || !sel_idx || !sel_len || !ctx || !out) {
        return cx::fail(e, "q, kv caches, selection tables and destination must be non-null");
    }
    if (rows <= 0 || nh <= 0 || nkv <= 0 || hd <= 0 || max_seq <= 0 || max_sel <= 0) {
        return cx::fail(e, "rows/nh/nkv/hd/max_seq/max_sel must be positive, got %d/%d/%d/%d/%d/%d",
                        rows, nh, nkv, hd, max_seq, max_sel);
    }
    if (nh % nkv != 0) {
        return cx::fail(e, "nh (%d) must be a multiple of nkv (%d)", nh, nkv);
    }
    if (hd % WARP != 0 || hd / WARP > ATT_SPARSE_LANES) {
        return cx::fail(e, "hd must be a multiple of %d and at most %d, got %d", WARP,
                        WARP * ATT_SPARSE_LANES, hd);
    }
    launch_attn_sparse(reinterpret_cast<const uint16_t*>(q), reinterpret_cast<const uint16_t*>(kc),
                       reinterpret_cast<const uint16_t*>(vc),
                       reinterpret_cast<const int*>(sel_idx), reinterpret_cast<const int*>(sel_len),
                       reinterpret_cast<const int*>(ctx), reinterpret_cast<uint16_t*>(out), rows,
                       nh, nkv, hd, max_seq, max_sel, (float)scale,
                       reinterpret_cast<cudaStream_t>(stream));
    return cx::finish(e);
}
