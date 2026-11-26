// @file attn_slice_abi.cu
// @brief 注意力的一个 C ABI 出口：``coffin_attn_slice``，无 torch。
//
// Module: coffin/backends/corex/kernels/abi。设备代码来自 ``../attn_slice_device.inc``——
// 与 ``../attn.cu`` 是同一份 ``cache_row`` / ``AttnLayout`` / 两条内核 / 一个启动器。
//
// 一个出口涵盖原来四个 pybind 入口：``attn_decode``（稠密缓存，无槽位表）、
// ``attn_paged``（单行槽位表）、``attn_paged_batch``（每行长度）、``attn_paged_multi_q``
// （每行 c 个草稿查询，可选块掩码）。四者的差别全在 ``AttnLayout`` 的步长和三个标志里，
// 所以分四个出口只会让同一套几何判断写四遍。
//
// Constraints:
//   * 签名与 ``cxrt/attn.py`` 的 ``argtypes`` 必须逐字一致：多一个参数就整体错位。
//   * ``pos2slot`` / ``ctx_lens`` / ``pos`` 里**至少有一个非空**：没有它窗口无从得知。
//     三者在设备上解引用，所以主机指针会读到垃圾而不是报错——调用方负责给设备地址。
//   * ``split`` 必须与 ``part_acc`` 的倒数第二轴一致（稠密 ``[nh, split, hd]``，批处理
//     ``[flat, nh, split, hd]``）。
//   * ``mask`` 为空表示精确注意力；非空时 ``mask_block`` 必须为正，且掩码是
//     ``uint8 [flat / draft, n_blocks]``。
//   * ``q`` / ``gate`` / ``out`` 必须是稠密的 bf16：内核按 ``[flat, nh, hd]`` 一整块索引。

#include "cx_tensor.h"

#include <cstdint>

#include "../attn_slice_device.inc"

// @brief 一次注意力调用的两趟：切片（在线 softmax 分片）与合并（加权求和 + 输出门）。
//
// @param q [flat, nh, hd] bf16 查询。
// @param gate [flat, nh, hd] bf16 输出门；``sigmoid`` 在合并那一趟施加。
// @param kc,vc int8 缓存；布局由 ``stride`` / ``stride_head`` 选择。
// @param ks,vs fp32 每（槽位，KV 头）的反量化 scale。
// @param out [flat, nh, hd] bf16 输出，原地写入。
// @param part_acc split-K 分子，``[flat, nh, split, hd]`` fp32。
// @param part_ml split-K 的（最大值，和），``[flat, nh, split, 2]`` fp32。
// @param mask 可选的 uint8 ``[flat / draft, n_blocks]`` 保留掩码（1 = 保留）。
// @param pos2slot 设备侧位置到槽位映射；稠密布局为空。
// @param ctx_lens 设备侧每行键数；为空时窗口由 ``pos`` 推出。
// @param pos 设备侧位置（单元素 int64）；``ctx_lens`` 非空时不用。
// @param scale ``hd ** -0.5``。
// @param flat 行数（批处理是 ``B``，投机校验是 ``B * c``）。
// @param nh 查询头数。
// @param hd 头宽。
// @param rep 每个 KV 头对应的查询头数。
// @param split 分片数。
// @param draft 共用一行历史的查询数；普通解码为 1。
// @param mask_block 掩码按多大的块表达；无掩码时为 0。
// @param stride 相邻槽位之间的行距（稠密与头主序是 1，token 主序是 ``nkv``）。
// @param max_seq 窗口上限，即槽位表的行间距。
// @param stride_head 相邻 KV 头之间的行距（稠密是 ``S``，token 主序是 1，头主序是 ``nslots``）。
// @param map_rows ``pos2slot`` 传的是整个 ``[B, max_seq]`` 映射（1）还是一条行的切片（0）。
// @param ctx_anchor ``ctx_lens`` 装的是锚点的最后下标（1）而不是键数（0）。
// @param stream 目标流；空指针表示默认流。
// @param err 错误缓冲，可为空。
// @param errlen 错误缓冲字节数。
// @return ``0`` 成功。
extern "C" int coffin_attn_slice(const void* q, const void* gate, const void* kc, const void* vc,
                                 const void* ks, const void* vs, void* out, void* part_acc,
                                 void* part_ml, const void* mask, const void* pos2slot,
                                 const void* ctx_lens, const void* pos, double scale, int flat,
                                 int nh, int hd, int rep, int split, int draft, int mask_block,
                                 int stride, int max_seq, int stride_head, int map_rows,
                                 int ctx_anchor, void* stream, char* err, int errlen) {
    const cx::Err e{err, errlen};
    if (!q || !gate || !kc || !vc || !ks || !vs || !out || !part_acc || !part_ml) {
        return cx::fail(e, "q, gate, kv cache, scales, destination and split buffers must be non-null");
    }
    if (flat <= 0 || nh <= 0 || hd <= 0 || rep <= 0 || split <= 0 || draft <= 0) {
        return cx::fail(e, "flat/nh/hd/rep/split/draft must be positive, got %d/%d/%d/%d/%d/%d",
                        flat, nh, hd, rep, split, draft);
    }
    if (pos2slot == nullptr && ctx_lens == nullptr && pos == nullptr) {
        return cx::fail(e, "one of pos2slot, ctx_lens or pos must be given to define the window");
    }
    if (mask != nullptr && mask_block <= 0) {
        return cx::fail(e, "a block mask needs a positive block size, got %d", mask_block);
    }
    if (stride <= 0 || stride_head <= 0 || max_seq <= 0) {
        return cx::fail(e, "stride/stride_head/max_seq must be positive, got %d/%d/%d", stride,
                        stride_head, max_seq);
    }
    const AttnLayout layout{reinterpret_cast<const int64_t*>(pos2slot),
                            reinterpret_cast<const int32_t*>(ctx_lens),
                            reinterpret_cast<const int64_t*>(pos), stride, max_seq,
                            ctx_anchor != 0, map_rows != 0, stride_head};
    launch_attn_slice(reinterpret_cast<const uint16_t*>(q), reinterpret_cast<const uint16_t*>(gate),
                      reinterpret_cast<const int8_t*>(kc), reinterpret_cast<const int8_t*>(vc),
                      reinterpret_cast<const float*>(ks), reinterpret_cast<const float*>(vs),
                      reinterpret_cast<uint16_t*>(out), reinterpret_cast<float*>(part_acc),
                      reinterpret_cast<float*>(part_ml),
                      mask == nullptr ? nullptr : reinterpret_cast<const uint8_t*>(mask), layout,
                      flat, nh, hd, rep, (float)scale, split, draft,
                      mask == nullptr ? 0 : mask_block,
                      reinterpret_cast<cudaStream_t>(stream));
    return cx::finish(e);
}
