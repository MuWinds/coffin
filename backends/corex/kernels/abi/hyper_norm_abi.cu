// @file hyper_norm_abi.cu
// @brief 分组 RMSNorm 的 C ABI 出口：``coffin_hyper_norm``，无 torch。
//
// Module: coffin/backends/corex/kernels/abi。设备代码来自 ``../hyper_norm_device.inc``——
// 与 ``../hyper.cu`` 是同一份 ``grouped_rms_norm_kernel``，两条前端逐位相同。
//
// 它为什么还在：``hyper_down`` / ``hyper_up`` 那两个吃 kn 权重的投影随 kn 布局整条删除，
// 归一化不吃投影权重，所以还在 decode 的残差路径上（``GatedResidual``）。搬这一条是为了
// 不在那一步留一个 torch 依赖。
//
// Constraints:
//   * 签名与 ``cxrt/moe.py`` 里的 ``argtypes`` 无关——它在 ``cxrt.hyper`` 那一面；两边
//     必须逐字一致，多一个参数就整体错位。
//   * 操作数必须**连续**：内核按 ``row * (hidden * hc)`` 取行，带间距会读错行。
//   * ``hc`` 上限 8，与内核里的 ``scale[8]`` 是同一个常量；超了会静默写崩 LDS。

#include "cx_tensor.h"

#include <cstdint>

#include "../hyper_norm_device.inc"

// @brief 对残差流做分组 RMSNorm：每 ``hidden`` 个元素一组，各自归一化。
//
// @param x [rows, hidden * hc] 连续 bf16/fp32。
// @param w [hidden * hc] fp32 零中心权重，前向按 ``1 + w``。
// @param out 目的地，形状与 dtype 同 ``x``。
// @param rows 行数。
// @param hidden 每流宽度。
// @param hc 流数，上限 8。
// @param eps 方差下限。
// @param code 激活的 dtype 码。
// @param stream 目标流；空指针表示默认流。
// @param err 错误缓冲，可为空。
// @param errlen 错误缓冲字节数。
// @return ``0`` 成功。
extern "C" int coffin_hyper_norm(const void* x, const void* w, void* out, int rows, int hidden,
                                 int hc, double eps, int code, void* stream, char* err,
                                 int errlen) {
    const cx::Err e{err, errlen};
    if (!x || !w || !out) return cx::fail(e, "activation, weight and destination must be non-null");
    if (rows < 0 || hidden <= 0 || hc <= 0 || hc > 8) {
        return cx::fail(e, "need rows >= 0 and 1..8 groups, got rows=%d hidden=%d hc=%d", rows,
                        hidden, hc);
    }
    const cx::DType dt = (cx::DType)code;
    if (dt != cx::kBF16 && dt != cx::kF32) {
        return cx::fail(e, "residual stream must be bf16 or fp32, got dtype code %d", (int)dt);
    }
    const int64_t width = (int64_t)hidden * (int64_t)hc;
    if (rows == 0) return cx::finish(e);
    const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    const int threads = 256;
    if (dt == cx::kBF16) {
        grouped_rms_norm_kernel<uint16_t><<<rows, threads, 0, s>>>(
            reinterpret_cast<const uint16_t*>(x), reinterpret_cast<const float*>(w),
            reinterpret_cast<uint16_t*>(out), rows, hidden, hc, (float)eps);
    } else {
        grouped_rms_norm_kernel<float><<<rows, threads, 0, s>>>(
            reinterpret_cast<const float*>(x), reinterpret_cast<const float*>(w),
            reinterpret_cast<float*>(out), rows, hidden, hc, (float)eps);
    }
    (void)width;
    return cx::finish(e);
}
