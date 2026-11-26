// @file delta_abi.cu
// @brief 门控 delta 的两个 C ABI 出口：``coffin_delta`` / ``coffin_delta_commit``。
//
// Module: coffin/backends/corex/kernels/abi。设备代码来自 ``../delta_device.inc``——
// 与 ``../delta.cu`` 是同一份 ``delta_kernel`` / ``launch_rows`` / ``delta_commit_kernel``。
//
// 两个出口涵盖原来五个 pybind 入口：``delta`` / ``delta_batch`` 是同一个启动器在
// ``rows = 1`` 与 ``rows = B`` 上的两次调用（网格取行轴），``delta_record`` /
// ``delta_batch_record`` 只多一个 ``record`` 指针与槽位，所以合并成「``record`` 为空即
// 不记录」的一条。
//
// Constraints:
//   * 签名与 ``cxrt/attn.py`` 的 ``argtypes`` 必须逐字一致：多一个参数就整体错位。
//   * ``st`` 与 ``out`` 必须连续（行偏移按 ``b * heads * dk * dv`` 算）；``record`` 非空时
//     必须是连续的 ``[rows, width, heads, dk + dv + 1]``。
//   * 全部操作数 fp32：这个内核没有 bf16 实例，窄化发生在它之前。

#include "cx_tensor.h"

#include <cstdint>

#include "../delta_device.inc"

// @brief 推进一个（或一批）门控 delta 步，原地更新状态。
//
// @param q [rows, heads, dk] fp32。
// @param k [rows, heads, dk] fp32。
// @param v [rows, heads, dv] fp32。
// @param beta [rows, heads] fp32。
// @param g [rows, heads] fp32，对数衰减（<= 0）。
// @param st [rows, heads, dk, dv] fp32，原地更新。
// @param out [rows, heads, dv] fp32。
// @param record 可为空；非空时是 ``[rows, record_width, heads, dk + dv + 1]`` fp32。
// @param record_slot 当前 token 在记录窗口中的位置。
// @param record_width 记录窗口长度。
// @param rows 批行数。
// @param heads 头数。
// @param dk 键维度。
// @param dv 值维度。
// @param stream 目标流；空指针表示默认流。
// @param err 错误缓冲，可为空。
// @param errlen 错误缓冲字节数。
// @return ``0`` 成功。
extern "C" int coffin_delta(const void* q, const void* k, const void* v, const void* beta,
                            const void* g, void* st, void* out, void* record, int record_slot,
                            int record_width, int rows, int heads, int dk, int dv, void* stream,
                            char* err, int errlen) {
    const cx::Err e{err, errlen};
    if (!q || !k || !v || !beta || !g || !st || !out) {
        return cx::fail(e, "q, k, v, beta, g, state and destination must be non-null");
    }
    if (rows <= 0 || heads <= 0 || dk <= 0 || dv <= 0) {
        return cx::fail(e, "rows/heads/dk/dv must be positive, got %d/%d/%d/%d", rows, heads, dk,
                        dv);
    }
    if (record != nullptr && (record_slot < 0 || record_slot >= record_width)) {
        return cx::fail(e, "record slot %d is outside the window of %d", record_slot,
                        record_width);
    }
    const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    launch_rows(reinterpret_cast<const float*>(q), reinterpret_cast<const float*>(k),
                reinterpret_cast<const float*>(v), reinterpret_cast<const float*>(beta),
                reinterpret_cast<const float*>(g), reinterpret_cast<float*>(st),
                reinterpret_cast<float*>(out),
                record == nullptr ? nullptr : reinterpret_cast<float*>(record), record_slot,
                record_width, rows, heads, dk, dv, s);
    return cx::finish(e);
}

// @brief 从基态按顺序重放已记录的秩一更新，写入提交目标。
//
// @param base [heads, dk, dv] fp32 初始状态。
// @param record [width, heads, dk + dv + 1] fp32。
// @param st [heads, dk, dv] fp32 提交目标。
// @param heads 头数。
// @param dk 键维度。
// @param dv 值维度。
// @param steps 要重放的前缀长度，``0 <= steps <= width``。
// @param stream 目标流。
// @param err 错误缓冲，可为空。
// @param errlen 错误缓冲字节数。
// @return ``0`` 成功。
extern "C" int coffin_delta_commit(const void* base, const void* record, void* st, int heads,
                                   int dk, int dv, int steps, void* stream, char* err,
                                   int errlen) {
    const cx::Err e{err, errlen};
    if (!base || !record || !st) {
        return cx::fail(e, "base, record and destination must be non-null");
    }
    if (heads <= 0 || dk <= 0 || dv <= 0 || steps < 0) {
        return cx::fail(e, "heads/dk/dv must be positive and steps >= 0, got %d/%d/%d/%d",
                        heads, dk, dv, steps);
    }
    const int threads = dv < THREADS ? dv : THREADS;
    const dim3 grid((unsigned)heads, (unsigned)((dv + threads - 1) / threads));
    delta_commit_kernel<<<grid, threads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<const float*>(base), reinterpret_cast<const float*>(record),
        reinterpret_cast<float*>(st), heads, dk, dv, steps);
    return cx::finish(e);
}
