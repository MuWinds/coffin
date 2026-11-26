// @file moe_epilog_abi.cu
// @brief 共享专家门标量的 C ABI 出口：``coffin_shared_gate_scale``，无 torch。
//
// Module: coffin/backends/corex/kernels/abi。设备代码来自
// ``../shared_gate_device.inc``——与 ``../moe_epilogue.cu`` 是同一份
// ``shared_gate_scale_kernel``，所以两条前端逐位相同。
//
// 为什么先搬这一条：它是 MoE 里唯一「一个标量决定整行」的算子，形状最简单（一维权重
// 加一批同形状操作数），没有分组偏移、没有块表。decode 每一步、48 层各一次。
//
// Constraints:
//   * 签名与 ``cxrt/moe.py`` 的 ``argtypes`` 必须逐字一致：多一个参数就整体错位。
//   * ``rows`` / ``width`` / 三个 pitch 都由调用方给：一维操作数（decode 的 ``[H]``）
//     的行间距必须显式写成 ``width``，否则内核按 pitch 走指针会读回第 0 行的通道块。
//   * ``out`` 允许与 ``src`` 重叠（内核先读后写同一元素）；由调用方分配。
//   * ``w`` 必须是 fp32 且连续，激活只有 bf16 / fp32 两种。

#include "cx_tensor.h"
#include "corex_scalar.h"

#include <cstdint>

#include "../shared_gate_device.inc"

namespace {

// @brief 启动前的形状与 dtype 校验。
//
// @param x 激活视图。
// @param w 门权重视图。
// @param src 被缩放的操作数。
// @param out 目的地。
// @param rows 行数。
// @param width 每行列数。
// @param width_arg 调用方声明的列数。
// @param err 错误缓冲。
// @return 合法为 ``0``。
int check_gate(const cx::Tensor& x, const cx::Tensor& w, const cx::Tensor& src,
               const cx::Tensor& out, int rows, int width, const cx::Err& err) {
    if (!x.data || !w.data || !src.data || !out.data) {
        return cx::fail(err, "activation, weight, source and destination must be non-null");
    }
    if (rows <= 0 || width <= 0) {
        return cx::fail(err, "rows/width must be positive, got %d/%d", rows, width);
    }
    if (w.dtype != cx::kF32) {
        return cx::fail(err, "gate weight must be fp32, got dtype code %d", (int)w.dtype);
    }
    if (x.dtype != cx::kBF16 && x.dtype != cx::kF32) {
        return cx::fail(err, "operands must be bf16 or fp32, got dtype code %d", (int)x.dtype);
    }
    if (src.dtype != x.dtype || out.dtype != x.dtype) {
        return cx::fail(err, "x, src and out must share a dtype");
    }
    if (w.cols < (int64_t)width) {
        return cx::fail(err, "gate weight holds %lld values, need %d",
                        (long long)w.cols, width);
    }
    if (x.cols < (int64_t)width || src.cols < (int64_t)width || out.cols < (int64_t)width) {
        return cx::fail(err, "operands must be at least %d wide, got x=%lld src=%lld out=%lld",
                        width, (long long)x.cols, (long long)src.cols, (long long)out.cols);
    }
    return 0;
}

}  // namespace

// @brief ``out = src * sigmoid(x . w)``，一次启动。
//
// @param x [rows, width] bf16/fp32，最内层步长 1。
// @param w [width] fp32 门权重，连续。
// @param src [rows, width]，同 dtype。
// @param out [rows, width]，同 dtype；允许与 ``src`` 重叠。
// @param rows 行数。
// @param width 每行列数 ``H``。
// @param x_pitch ``x`` 的行间距（元素）。
// @param s_pitch ``src`` 的行间距（元素）。
// @param o_pitch ``out`` 的行间距（元素）。
// @param code 激活的 dtype 码（``cx::kBF16`` / ``cx::kF32``）。
// @param stream 目标流；空指针表示默认流。
// @param err 错误缓冲，可为空。
// @param errlen 错误缓冲字节数。
// @return ``0`` 成功。
extern "C" int coffin_shared_gate_scale(const void* x, const void* w, const void* src,
                                        void* out, int rows, int width, long long x_pitch,
                                        long long s_pitch, long long o_pitch, int code,
                                        void* stream, char* err, int errlen) {
    const cx::Err e{err, errlen};
    if (x_pitch < (long long)width || s_pitch < (long long)width ||
        o_pitch < (long long)width) {
        return cx::fail(e, "row pitch narrower than the row width (%lld/%lld/%lld < %d)",
                        x_pitch, s_pitch, o_pitch, width);
    }
    const cx::DType dt = (cx::DType)code;
    const cx::Tensor act = cx::Tensor::matrix(const_cast<void*>(x), rows, width,
                                              (int64_t)x_pitch, dt);
    const cx::Tensor gate = cx::Tensor::matrix(const_cast<void*>(w), 1, width,
                                               (int64_t)width, cx::kF32);
    const cx::Tensor source = cx::Tensor::matrix(const_cast<void*>(src), rows, width,
                                                 (int64_t)s_pitch, dt);
    const cx::Tensor dest = cx::Tensor::matrix(out, rows, width, (int64_t)o_pitch, dt);
    if (const int bad = check_gate(act, gate, source, dest, rows, width, e)) return bad;
    const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    const dim3 grid((unsigned)rows);
    if (dt == cx::kBF16) {
        shared_gate_scale_kernel<uint16_t><<<grid, SGATE_THREADS, 0, s>>>(
            reinterpret_cast<const uint16_t*>(act.data),
            reinterpret_cast<const float*>(gate.data),
            reinterpret_cast<const uint16_t*>(source.data),
            reinterpret_cast<uint16_t*>(dest.data), width, (long)x_pitch, (long)s_pitch,
            (long)o_pitch);
    } else {
        shared_gate_scale_kernel<float><<<grid, SGATE_THREADS, 0, s>>>(
            reinterpret_cast<const float*>(act.data),
            reinterpret_cast<const float*>(gate.data),
            reinterpret_cast<const float*>(source.data),
            reinterpret_cast<float*>(dest.data), width, (long)x_pitch, (long)s_pitch,
            (long)o_pitch);
    }
    return cx::finish(e);
}
