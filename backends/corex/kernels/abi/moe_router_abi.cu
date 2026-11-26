// @file moe_router_abi.cu
// @brief MoE 路由融合的 C ABI 出口：``coffin_moe_router``，无 torch。
//
// Module: coffin/backends/corex/kernels/abi。设备代码来自
// ``../moe_router_device.inc``——与 ``../moe_grouped.cu`` 是同一份 ``moe_router_t``，
// 所以两条前端逐位相同（包括等值取小下标、clamp 越界组这些细节）。
//
// 为什么它值得搬：``aten::topk`` 对 ``[1, 512]`` 取 top-10 在本卡上一次 38 us，而
// decode 一步 48 层各来一遍。它也是 MoE 里唯一「输出两个小张量」的算子，形状固定
// （``[tokens, top_k]`` 两个稠密块），没有分组偏移也没有块表。
//
// Constraints:
//   * 签名与 ``cxrt/moe.py`` 的 ``argtypes`` 必须逐字一致：多一个参数就整体错位。
//   * logits 必须**连续**：内核按 ``row = logits + token * n`` 直接取行，带行间距的
//     视图会让它读错行。调用方负责先 ``contiguous()``。
//   * ``out_idx`` 是 int64（``torch.topk`` 的下标类型，上层直接拿去做索引）。
//   * ``top_k`` 上限 32：一个 32-lane 组里每个 lane 只存自己那一槽的结果。

#include "cx_tensor.h"
#include "corex_scalar.h"

#include <climits>
#include <cstdint>

// 内核与常量从头部文件进来。它在**文件作用域**而不是匿名命名空间里：头部自己
// ``#include <climits>`` 与 ``corex_scalar.h``，放进命名空间会把标准库内容也塞进去。
// 本文件是这份内核在 ABI 库里唯一的使用者，不会与别的出口撞符号。
#include "../moe_router_device.inc"

// @brief 启动前的形状、dtype 与 top_k 校验。
//
// @param logits 路由 logits 视图（``[tokens, n]``）。
// @param idx 下标目的地（``[tokens, top_k]`` int64）。
// @param w 权重目的地（``[tokens, top_k]``，dtype 同 logits）。
// @param tokens 行数。
// @param n 专家数。
// @param top_k 每 token 的专家数。
// @param err 错误缓冲。
// @return 合法为 ``0``。
int check_router(const cx::Tensor& logits, const cx::Tensor& idx, const cx::Tensor& w,
                 int tokens, int n, int top_k, const cx::Err& err) {
    if (!logits.data || !idx.data || !w.data) {
        return cx::fail(err, "logits, index and weight destinations must be non-null");
    }
    if (logits.dtype != cx::kBF16 && logits.dtype != cx::kF32) {
        return cx::fail(err, "router logits must be bf16 or fp32, got dtype code %d",
                        (int)logits.dtype);
    }
    if (idx.dtype != cx::kI64) {
        return cx::fail(err, "index destination must be int64, got dtype code %d",
                        (int)idx.dtype);
    }
    if (w.dtype != logits.dtype) {
        return cx::fail(err, "weight destination dtype must match the logits");
    }
    if (tokens < 0) return cx::fail(err, "tokens must not be negative, got %d", tokens);
    if (top_k <= 0 || top_k > 32) {
        return cx::fail(err, "top_k must be in (0, 32], got %d", top_k);
    }
    if (n <= 0 || n > ROUTER_LANES * ROUTER_ELEMS) {
        return cx::fail(err, "expert count %d exceeds %d", n, ROUTER_LANES * ROUTER_ELEMS);
    }
    if (top_k > n) return cx::fail(err, "top_k %d exceeds the expert count %d", top_k, n);
    if (logits.cols != n || logits.rows != tokens || logits.pitch != n) {
        return cx::fail(err, "logits must be contiguous [%d, %d], got [%lld, %lld] pitch %lld",
                        tokens, n, (long long)logits.rows, (long long)logits.cols,
                        (long long)logits.pitch);
    }
    if (idx.rows != tokens || idx.cols != top_k || idx.pitch != top_k) {
        return cx::fail(err, "index destination must be contiguous [%d, %d]", tokens, top_k);
    }
    if (w.rows != tokens || w.cols != top_k || w.pitch != top_k) {
        return cx::fail(err, "weight destination must be contiguous [%d, %d]", tokens, top_k);
    }
    return 0;
}

// @brief ``softmax -> top-k -> 归一化``，一次发射写出下标与权重。
//
// @param logits ``[tokens, n]`` 连续 bf16 / fp32。
// @param out_idx ``[tokens, top_k]`` int64 目的地。
// @param out_w ``[tokens, top_k]`` 目的地，dtype 同 logits。
// @param tokens 行数。
// @param experts 专家数 ``n``。
// @param top_k 每 token 的专家数。
// @param normalize 是否把 top-k 概率再归一化（非 0 为是）。
// @param code logits 的 dtype 码。
// @param stream 目标流；空指针表示默认流。
// @param err 错误缓冲，可为空。
// @param errlen 错误缓冲字节数。
// @return ``0`` 成功。
extern "C" int coffin_moe_router(const void* logits, void* out_idx, void* out_w, int tokens,
                                 int experts, int top_k, int normalize, int code,
                                 void* stream, char* err, int errlen) {
    const cx::Err e{err, errlen};
    const cx::DType dt = (cx::DType)code;
    const cx::Tensor src = cx::Tensor::matrix(const_cast<void*>(logits), tokens, experts,
                                              experts, dt);
    const cx::Tensor idx = cx::Tensor::matrix(out_idx, tokens, top_k, top_k, cx::kI64);
    const cx::Tensor w = cx::Tensor::matrix(out_w, tokens, top_k, top_k, dt);
    if (const int bad = check_router(src, idx, w, tokens, experts, top_k, e)) return bad;
    if (tokens == 0) return cx::finish(e);

    const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    const dim3 block(ROUTER_LANES * ROUTER_GROUPS);
    const dim3 grid((unsigned)((tokens + ROUTER_GROUPS - 1) / ROUTER_GROUPS));
    const bool narrow = experts <= 16 * ROUTER_LANES;
    int64_t* idx_ptr = reinterpret_cast<int64_t*>(idx.data);
    if (dt == cx::kBF16) {
        const uint16_t* src_ptr = reinterpret_cast<const uint16_t*>(src.data);
        uint16_t* w_ptr = reinterpret_cast<uint16_t*>(w.data);
        if (narrow) {
            moe_router_t<16, uint16_t><<<grid, block, 0, s>>>(
                src_ptr, idx_ptr, w_ptr, tokens, experts, top_k, normalize ? 1 : 0);
        } else {
            moe_router_t<32, uint16_t><<<grid, block, 0, s>>>(
                src_ptr, idx_ptr, w_ptr, tokens, experts, top_k, normalize ? 1 : 0);
        }
    } else {
        const float* src_ptr = reinterpret_cast<const float*>(src.data);
        float* w_ptr = reinterpret_cast<float*>(w.data);
        if (narrow) {
            moe_router_t<16, float><<<grid, block, 0, s>>>(
                src_ptr, idx_ptr, w_ptr, tokens, experts, top_k, normalize ? 1 : 0);
        } else {
            moe_router_t<32, float><<<grid, block, 0, s>>>(
                src_ptr, idx_ptr, w_ptr, tokens, experts, top_k, normalize ? 1 : 0);
        }
    }
    return cx::finish(e);
}
