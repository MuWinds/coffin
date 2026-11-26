// @file conv_abi.cu
// @brief 线性注意力入口卷积的 C ABI 出口：``coffin_conv1d_silu`` /
//        ``_seq`` / ``_batch``，无 torch。
//
// Module: coffin/backends/corex/kernels/abi。设备代码来自 ``../conv_device.inc``——与
// ``../linear_attn.cu`` 是同一份 ``conv1d_silu_kernel`` / ``conv1d_silu_seq_kernel``，
// 所以两条前端逐位相同。两条内核共用一份实现不是省事：批处理入口启动的就是单 token
// 那条（行轴收成 ``rows``），算术不可能各自漂移。
//
// 为什么先搬这一族：qwen35 每条线性注意力层都要走它，形状简单（一维通道 + 三元素历史
// 环），没有块表也没有分组偏移。唯一的坑是行间距——校验主体传进来的是
// ``qkv[:, j]``（``[rows, c, conv_dim]`` 的列切片），行间距是 ``c * C``，按 ``C`` 读会让
// 第 1 行以后全取到第 0 行的通道块。
//
// Constraints:
//   * 三个签名与 ``cxrt/conv.py`` 的 ``argtypes`` 必须逐字一致。多一个参数就会整体错位。
//   * 激活只有 bf16 一种：内核读 ``uint16_t``。上面的包装层也是这么拒绝 fp32 的。
//   * ``state`` 原地更新是语义而不是优化：它是这一层的短期记忆，必须与循环状态一起推进。
//   * ``out`` 由调用方分配（与 norm/gemv 的 ABI 一致），地址在整步里保持稳定。

#include "cx_tensor.h"
#include "corex_scalar.h"

#include <cstdint>

#include "../conv_device.inc"

namespace {

// @brief 单 token 与批处理共用的启动前校验。
//
// @param w 抽头视图（``[C, 4]`` fp32）。
// @param x 激活（bf16，行间距 ``ldx``）。
// @param state 历史（fp32，``[rows, C, 3]``，行间距 ``lds``）。
// @param out 目的地（fp32，``[rows, C]``，行间距 ``ldo``）。
// @param rows 批行数。
// @param C 通道数。
// @param err 错误缓冲。
// @return 合法为 ``0``。
int check_conv(const cx::Tensor& w, const cx::Tensor& x, const cx::Tensor& state,
               const cx::Tensor& out, int rows, int C, const cx::Err& err) {
    if (!w.data || !x.data || !state.data || !out.data) {
        return cx::fail(err, "taps, activation, state and destination must be non-null");
    }
    if (rows <= 0 || C <= 0) {
        return cx::fail(err, "rows/C must be positive, got %d/%d", rows, C);
    }
    if (x.dtype != cx::kBF16) {
        return cx::fail(err, "activation must be bf16, got dtype code %d", (int)x.dtype);
    }
    if (w.dtype != cx::kF32 || state.dtype != cx::kF32 || out.dtype != cx::kF32) {
        return cx::fail(err, "taps, state and destination must be fp32");
    }
    if (w.rows != C || w.cols != CONV_K) {
        return cx::fail(err, "taps must be [%d, %d], got [%lld, %lld]", C, CONV_K,
                        (long long)w.rows, (long long)w.cols);
    }
    if (x.cols != C || out.cols != C) {
        return cx::fail(err, "activation/destination must be C wide, got %lld/%lld",
                        (long long)x.cols, (long long)out.cols);
    }
    if (state.rows * state.cols < (int64_t)C * CONV_STATE) {
        return cx::fail(err, "state must hold %d values per row, got %lld",
                        C * CONV_STATE, (long long)(state.rows * state.cols));
    }
    return 0;
}

}  // namespace

// @brief 单 token 的深度因果卷积接 SiLU（一行）。
//
// @param w [C, 4] fp32 抽头。
// @param x [C] bf16 投影输出。
// @param state [C, 3] fp32 历史，原地平移。
// @param out [C] fp32 目的地。
// @param C 通道数。
// @param stream 目标流；空指针表示默认流。
// @param err 错误缓冲，可为空。
// @param errlen 错误缓冲字节数。
// @return ``0`` 成功。
extern "C" int coffin_conv1d_silu(const void* w, const void* x, void* state, void* out,
                                  int C, void* stream, char* err, int errlen) {
    const cx::Err e{err, errlen};
    const cx::Tensor taps = cx::Tensor::matrix(const_cast<void*>(w), C, CONV_K, CONV_K,
                                               cx::kF32);
    const cx::Tensor act = cx::Tensor::matrix(const_cast<void*>(x), 1, C, C, cx::kBF16);
    const cx::Tensor hist = cx::Tensor::matrix(state, 1, C * CONV_STATE, C * CONV_STATE,
                                               cx::kF32);
    const cx::Tensor dest = cx::Tensor::matrix(out, 1, C, C, cx::kF32);
    if (const int bad = check_conv(taps, act, hist, dest, 1, C, e)) return bad;
    const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    conv1d_silu_kernel<<<dim3((C + 255) / 256, 1), 256, 0, s>>>(
        reinterpret_cast<const float*>(taps.data),
        reinterpret_cast<const uint16_t*>(act.data),
        reinterpret_cast<float*>(hist.data), reinterpret_cast<float*>(dest.data),
        C, C, CONV_STATE, C);
    return cx::finish(e);
}

// @brief 预填充分块的 conv1d 加 SiLU：沿 ``T`` 分 ``CONV_NSEG`` 段，只有最后一段回写窗口。
//
// @param w [C, 4] fp32 抽头。
// @param x [T, C] bf16，行间距 ``ldx``（可以大于 ``C``）。
// @param state [C, 3] fp32，原地更新。
// @param out [T, C] fp32 稠密目的地。
// @param T token 数。
// @param C 通道数。
// @param ldx ``x`` 的行间距，以元素计。
// @param stream 目标流。
// @param err 错误缓冲，可为空。
// @param errlen 错误缓冲字节数。
// @return ``0`` 成功。
extern "C" int coffin_conv1d_silu_seq(const void* w, const void* x, void* state, void* out,
                                      int T, int C, long long ldx, void* stream, char* err,
                                      int errlen) {
    const cx::Err e{err, errlen};
    if (ldx < (long long)C) {
        return cx::fail(e, "activation row pitch %lld is narrower than %d", ldx, C);
    }
    const cx::Tensor taps = cx::Tensor::matrix(const_cast<void*>(w), C, CONV_K, CONV_K,
                                               cx::kF32);
    const cx::Tensor act = cx::Tensor::matrix(const_cast<void*>(x), T, C, (int64_t)ldx,
                                              cx::kBF16);
    const cx::Tensor hist = cx::Tensor::matrix(state, 1, C * CONV_STATE, C * CONV_STATE,
                                               cx::kF32);
    const cx::Tensor dest = cx::Tensor::matrix(out, T, C, C, cx::kF32);
    if (const int bad = check_conv(taps, act, hist, dest, T, C, e)) return bad;
    int nseg = 0;
    const int seg = conv_seg_len(T, nseg);
    const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    conv1d_silu_seq_kernel<<<dim3((C + 255) / 256, nseg), 256, 0, s>>>(
        reinterpret_cast<const float*>(taps.data),
        reinterpret_cast<const uint16_t*>(act.data),
        reinterpret_cast<float*>(hist.data), reinterpret_cast<float*>(dest.data),
        C, T, (int)ldx, seg);
    return cx::finish(e);
}

// @brief 批处理单 token 的 conv1d 加 SiLU：所有行一次启动。
//
// @param w [C, 4] fp32 抽头，所有行共用。
// @param x [B, C] bf16，行间距 ``ldx``；带步长的列切片也合法。
// @param state [B, C, 3] fp32，行间距 ``lds``，原地平移。
// @param out [B, C] fp32，行间距 ``ldo``。
// @param rows 批行数。
// @param C 通道数。
// @param ldx ``x`` 的行间距（元素）。
// @param lds ``state`` 的行间距（元素）。
// @param ldo ``out`` 的行间距（元素）。
// @param stream 目标流。
// @param err 错误缓冲，可为空。
// @param errlen 错误缓冲字节数。
// @return ``0`` 成功。
extern "C" int coffin_conv1d_silu_batch(const void* w, const void* x, void* state,
                                        void* out, int rows, int C, long long ldx,
                                        long long lds, long long ldo, void* stream, char* err,
                                        int errlen) {
    const cx::Err e{err, errlen};
    if (ldx < (long long)C || ldo < (long long)C) {
        return cx::fail(e, "row pitches must be at least %d, got ldx=%lld ldo=%lld",
                        C, ldx, ldo);
    }
    if (lds < (long long)C * CONV_STATE) {
        return cx::fail(e, "state row pitch %lld is narrower than %d",
                        lds, C * CONV_STATE);
    }
    const cx::Tensor taps = cx::Tensor::matrix(const_cast<void*>(w), C, CONV_K, CONV_K,
                                               cx::kF32);
    const cx::Tensor act = cx::Tensor::matrix(const_cast<void*>(x), rows, C, (int64_t)ldx,
                                              cx::kBF16);
    const cx::Tensor hist = cx::Tensor::matrix(state, rows, C * CONV_STATE, (int64_t)lds,
                                               cx::kF32);
    const cx::Tensor dest = cx::Tensor::matrix(out, rows, C, (int64_t)ldo, cx::kF32);
    if (const int bad = check_conv(taps, act, hist, dest, rows, C, e)) return bad;
    const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    conv1d_silu_kernel<<<dim3((C + 255) / 256, rows), 256, 0, s>>>(
        reinterpret_cast<const float*>(taps.data),
        reinterpret_cast<const uint16_t*>(act.data),
        reinterpret_cast<float*>(hist.data), reinterpret_cast<float*>(dest.data),
        C, (int)ldx, (int)lds, (int)ldo);
    return cx::finish(e);
}
