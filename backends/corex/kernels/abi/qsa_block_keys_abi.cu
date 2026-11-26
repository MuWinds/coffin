// @file qsa_block_keys_abi.cu
// @brief QSA 块 key 的 C ABI 出口：``coffin_qsa_block_keys``，无 torch。
//
// Module: coffin/backends/corex/kernels/abi。设备代码来自
// ``../qsa_block_keys_device.inc``——与 ``../qsa_block_keys.cu`` 是同一份内核与启动器。
//
// 这里挡的是「会让内核读越界」的几何：``ratio`` 超出 ``MAX_RATIO``、``rotary_dim``
// 不是偶数或超过 ``head_dim``。其余形状（表与输出是否一致、cos/sin 的宽度）由调用方
// 在 ``backends/corex`` 里按张量形状检查——ABI 面没有形状。
//
// Constraints:
//   * 签名与 ``cxrt/attn.py`` 的 ``argtypes`` 必须逐字一致：多一个参数就整体错位。
//   * ``table`` / ``block_row`` / ``block_batch`` 是**设备** int32 指针。

#include "cx_tensor.h"

#include <cstdint>

#include "../qsa_block_keys_device.inc"

// @brief 块 key：gather → 池化 → RMSNorm → partial RoPE。
//
// @param flat_keys [batch * tokens, head_dim] bf16。
// @param table [rows, max_blocks, ratio] int32，``-1`` 为补位。
// @param weight [head_dim] bf16 零中心增益。
// @param flat_cos,flat_sin [batch * tokens, rotary_dim] bf16。
// @param block_row,block_batch [rows * max_blocks] int32。
// @param out [rows, max_blocks, head_dim] bf16，原地写入。
// @param seq, tokens, rows, max_blocks 几何。
// @param ratio 池化宽度。
// @param head_dim 每头维数。
// @param rotary_dim 参与旋转的前若干维。
// @param eps RMSNorm 的 epsilon。
// @param stream 目标流；空指针表示默认流。
// @param err 错误缓冲，可为空。
// @param errlen 错误缓冲字节数。
// @return ``0`` 成功。
extern "C" int coffin_qsa_block_keys(const void* flat_keys, const void* table, const void* weight,
                                     const void* flat_cos, const void* flat_sin,
                                     const void* block_row, const void* block_batch, void* out,
                                     int rows, int seq, int tokens, int max_blocks, int ratio,
                                     int head_dim, int rotary_dim, double eps, void* stream,
                                     char* err, int errlen) {
    const cx::Err e{err, errlen};
    if (!flat_keys || !table || !weight || !flat_cos || !flat_sin || !block_row || !block_batch
        || !out) {
        return cx::fail(e, "keys, table, weight, cos/sin, block rows and destination must be non-null");
    }
    if (rows <= 0 || tokens <= 0 || max_blocks <= 0 || head_dim <= 0) {
        return cx::fail(e, "rows/tokens/max_blocks/head_dim must be positive, got %d/%d/%d/%d",
                        rows, tokens, max_blocks, head_dim);
    }
    if (ratio <= 0 || ratio > MAX_RATIO) {
        return cx::fail(e, "compress ratio %d must be in [1, %d]", ratio, MAX_RATIO);
    }
    if (rotary_dim <= 0 || rotary_dim > head_dim || (rotary_dim & 1) != 0) {
        return cx::fail(e, "rotary_dim %d must be even and in [1, head_dim=%d]", rotary_dim, head_dim);
    }
    launch_qsa_block_keys(reinterpret_cast<const uint16_t*>(flat_keys),
                          reinterpret_cast<const int*>(table),
                          reinterpret_cast<const uint16_t*>(weight),
                          reinterpret_cast<const uint16_t*>(flat_cos),
                          reinterpret_cast<const uint16_t*>(flat_sin),
                          reinterpret_cast<uint16_t*>(out), reinterpret_cast<const int*>(block_row),
                          reinterpret_cast<const int*>(block_batch), rows, seq, tokens, max_blocks,
                          ratio, head_dim, rotary_dim, (float)eps,
                          reinterpret_cast<cudaStream_t>(stream));
    return cx::finish(e);
}
