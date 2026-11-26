// @file norm_abi.cu
// @brief 归一化族的五个 C ABI 出口（含两个 int8 量化融合体），无 torch。
//
// Module: coffin/backends/corex/kernels/abi。设备代码来自 ``norm_device.inc``——
// 与 ``../norm.cu`` 是同一份（五个 ``__global__`` 都在里面），所以两条前端的数值
// 逐位相同。本文件只做三件事：把裸指针折成 ``cx::Tensor``、按 dtype 分派实例化、
// 把驱动错误写回调用方的缓冲。
//
// 为什么参数是裸指针而不是张量对象：``ctypes`` 只认扁平参数。指针 + 形状 + 间距 +
// dtype 码已经覆盖内核索引需要的全部信息，而「谁分配」这件事因此变成调用方的责任，
// 地址在整步里保持稳定（后续要接自录自发，这是前提）。
//
// Constraints:
//   * 签名与 ``rt/shim.py`` 的 ``argtypes`` 必须逐字一致。多一个参数就会整体错位一格，
//     症状是内核读垃圾。
//   * 流的来源是调用方传入的 ``void* stream``，不查任何全局「当前流」。没有 torch 就
//     没有那个全局状态，多步并发或重入时它是第一个坏掉的东西。
//   * 启动完立刻 ``cx::finish``，不要 ``cudaDeviceSynchronize``：非法参数与超限共享
//     内存会在启动时就置错误码，而同步会把 20 us 的发射变成整设备等待。

#include "cx_tensor.h"
#include "corex_scalar.h"
#include "norm_device.inc"

#include <cstdint>

namespace {

// @brief 按 dtype 码分派 ``rms_norm_kernel`` 并检查启动结果。
//
// @param x 激活视图。
// @param w 增益视图。
// @param y 目的地视图，稠密。
// @param eps 方差下限。
// @param stream 目标流；空指针表示默认流。
// @param err 调用方的错误缓冲。
// @return ``0`` 成功，否则为 ``cx::fail`` / ``cx::finish`` 的返回码。
int dispatch_rms_norm(const cx::Tensor& x, const cx::Tensor& w, const cx::Tensor& y,
                      float eps, void* stream, const cx::Err& err) {
    const int m = (int)x.rows;
    const int n = (int)x.cols;
    const int threads = rms_threads(n);
    const size_t shmem = (size_t)threads * sizeof(float);
    const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    switch (x.dtype) {
        case cx::kBF16:
            rms_norm_kernel<uint16_t><<<m, threads, shmem, s>>>(
                reinterpret_cast<const uint16_t*>(x.data),
                reinterpret_cast<const uint16_t*>(w.data),
                reinterpret_cast<uint16_t*>(y.data), n, (long)x.pitch, eps);
            break;
        case cx::kF32:
            rms_norm_kernel<float><<<m, threads, shmem, s>>>(
                reinterpret_cast<const float*>(x.data),
                reinterpret_cast<const float*>(w.data),
                reinterpret_cast<float*>(y.data), n, (long)x.pitch, eps);
            break;
        default:
            return cx::fail(err, "unsupported activation dtype code %d", (int)x.dtype);
    }
    return cx::finish(err);
}

// @brief 按 dtype 码分派 ``add_rms_norm_kernel`` 并检查启动结果。
//
// @param add 加数视图。
// @param x 残差视图，原地写。
// @param w 增益视图。
// @param y 目的地视图，稠密。
// @param eps 方差下限。
// @param stream 目标流。
// @param err 错误缓冲。
// @return ``0`` 成功。
int dispatch_add_rms_norm(const cx::Tensor& add, const cx::Tensor& x, const cx::Tensor& w,
                          const cx::Tensor& y, float eps, void* stream, const cx::Err& err) {
    const int m = (int)x.rows;
    const int n = (int)x.cols;
    const int threads = rms_threads(n);
    const size_t shmem = (size_t)threads * sizeof(float);
    const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    switch (x.dtype) {
        case cx::kBF16:
            add_rms_norm_kernel<uint16_t><<<m, threads, shmem, s>>>(
                reinterpret_cast<const uint16_t*>(add.data),
                reinterpret_cast<uint16_t*>(x.data),
                reinterpret_cast<const uint16_t*>(w.data),
                reinterpret_cast<uint16_t*>(y.data), n, (long)x.pitch, eps);
            break;
        case cx::kF32:
            add_rms_norm_kernel<float><<<m, threads, shmem, s>>>(
                reinterpret_cast<const float*>(add.data),
                reinterpret_cast<float*>(x.data),
                reinterpret_cast<const float*>(w.data),
                reinterpret_cast<float*>(y.data), n, (long)x.pitch, eps);
            break;
        default:
            return cx::fail(err, "unsupported activation dtype code %d", (int)x.dtype);
    }
    return cx::finish(err);
}

// @brief 按 dtype 码分派 ``rms_norm_gated_kernel`` 并检查启动结果。
//
// @param x 激活视图，稠密。
// @param w 按头的增益视图。
// @param gate 门视图，稠密。
// @param y 目的地视图，稠密。
// @param eps 方差下限。
// @param stream 目标流。
// @param err 错误缓冲。
// @return ``0`` 成功。
int dispatch_rms_norm_gated(const cx::Tensor& x, const cx::Tensor& w, const cx::Tensor& gate,
                            const cx::Tensor& y, float eps, void* stream,
                            const cx::Err& err) {
    const int m = (int)x.rows;
    const int n = (int)x.cols;
    const int threads = rms_threads(n);
    const size_t shmem = (size_t)threads * sizeof(float);
    const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    switch (x.dtype) {
        case cx::kBF16:
            rms_norm_gated_kernel<uint16_t><<<m, threads, shmem, s>>>(
                reinterpret_cast<const uint16_t*>(x.data),
                reinterpret_cast<const uint16_t*>(w.data),
                reinterpret_cast<const uint16_t*>(gate.data),
                reinterpret_cast<uint16_t*>(y.data), n, eps);
            break;
        case cx::kF32:
            rms_norm_gated_kernel<float><<<m, threads, shmem, s>>>(
                reinterpret_cast<const float*>(x.data),
                reinterpret_cast<const float*>(w.data),
                reinterpret_cast<const float*>(gate.data),
                reinterpret_cast<float*>(y.data), n, eps);
            break;
        default:
            return cx::fail(err, "unsupported activation dtype code %d", (int)x.dtype);
    }
    return cx::finish(err);
}

// @brief 融合归一化 + int8 量化共用的线程数与共享内存核算。
//
// 两个 int8 条目都要做这件事，而且失败条件一样，所以只有一处实现：线程数必须落在
// wavefront 的整数倍上，LDS 装得下「归约临时量 + 舍入后的一整行」。
//
// @param rows 行数，用于按 ``M`` 分档选线程数。
// @param n 行宽。
// @param code dtype 码。
// @param threads 请求的线程数；``0`` 表示用实测档位。
// @param out_threads 写回线程数。
// @param out_shmem 写回共享内存字节数。
// @param err 错误缓冲。
// @return ``0`` 通过，否则为 ``cx::fail`` 的返回码。
int i8_budget(int rows, int n, int code, int threads, int* out_threads, size_t* out_shmem,
              const cx::Err& err) {
    const int th = threads > 0 ? threads : i8_fused_threads(rows);
    if (th % WAVEFRONT != 0) {
        return cx::fail(err, "thread count must be a multiple of the %d-lane wavefront, "
                             "got %d", WAVEFRONT, th);
    }
    const size_t row_bytes = (size_t)n * (code == cx::kBF16 ? sizeof(uint16_t)
                                                            : sizeof(float));
    const size_t shmem = (size_t)th * sizeof(float) + row_bytes;
    if (shmem > 48u * 1024u) {
        return cx::fail(err, "fused rmsnorm->int8 needs %zu bytes of shared memory for "
                             "N = %d, over the 48 KB per-block budget; use rms_norm + "
                             "quant_act_i8 for widths this large", shmem, n);
    }
    *out_threads = th;
    *out_shmem = shmem;
    return 0;
}

// @brief 按 dtype 码分派 ``rms_norm_i8_kernel`` 并检查启动结果。
//
// @param x 激活视图，行间距 ``x.pitch``。
// @param w 增益视图。
// @param q int8 目的地视图，稠密 ``[m, n]``。
// @param s 每行 scale 视图，fp32 ``[m]``。
// @param eps 方差下限。
// @param threads 每块线程数；``0`` 表示按 ``M`` 分档。
// @param stream 目标流。
// @param err 错误缓冲。
// @return ``0`` 成功。
int dispatch_rms_norm_i8(const cx::Tensor& x, const cx::Tensor& w, const cx::Tensor& q,
                         const cx::Tensor& s, float eps, int threads, void* stream,
                         const cx::Err& err) {
    const int m = (int)x.rows;
    const int n = (int)x.cols;
    int th = 0;
    size_t shmem = 0;
    const int rc = i8_budget(m, n, (int)x.dtype, threads, &th, &shmem, err);
    if (rc != 0) return rc;
    const cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    int8_t* qp = static_cast<int8_t*>(q.data);
    float* sp = static_cast<float*>(s.data);
    switch (x.dtype) {
        case cx::kBF16:
            rms_norm_i8_kernel<uint16_t><<<m, th, shmem, st>>>(
                reinterpret_cast<const uint16_t*>(x.data),
                reinterpret_cast<const uint16_t*>(w.data), qp, sp, n, (long)x.pitch, eps);
            break;
        case cx::kF32:
            rms_norm_i8_kernel<float><<<m, th, shmem, st>>>(
                reinterpret_cast<const float*>(x.data),
                reinterpret_cast<const float*>(w.data), qp, sp, n, (long)x.pitch, eps);
            break;
        default:
            return cx::fail(err, "unsupported activation dtype code %d", (int)x.dtype);
    }
    return cx::finish(err);
}

// @brief 按 dtype 码分派 ``add_rms_norm_i8_kernel`` 并检查启动结果。
//
// @param add 加数视图，行间距 ``pitch``。
// @param x 残差视图，原地写。
// @param w 增益视图。
// @param q int8 目的地，稠密。
// @param s 每行 scale。
// @param eps 方差下限。
// @param threads 每块线程数；``0`` 表示按 ``M`` 分档。
// @param stream 目标流。
// @param err 错误缓冲。
// @return ``0`` 成功。
int dispatch_add_rms_norm_i8(const cx::Tensor& add, const cx::Tensor& x,
                             const cx::Tensor& w, const cx::Tensor& q, const cx::Tensor& s,
                             float eps, int threads, void* stream, const cx::Err& err) {
    const int m = (int)x.rows;
    const int n = (int)x.cols;
    int th = 0;
    size_t shmem = 0;
    const int rc = i8_budget(m, n, (int)x.dtype, threads, &th, &shmem, err);
    if (rc != 0) return rc;
    const cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    int8_t* qp = static_cast<int8_t*>(q.data);
    float* sp = static_cast<float*>(s.data);
    switch (x.dtype) {
        case cx::kBF16:
            add_rms_norm_i8_kernel<uint16_t><<<m, th, shmem, st>>>(
                reinterpret_cast<const uint16_t*>(add.data),
                reinterpret_cast<uint16_t*>(x.data),
                reinterpret_cast<const uint16_t*>(w.data), qp, sp, n, (long)x.pitch, eps);
            break;
        case cx::kF32:
            add_rms_norm_i8_kernel<float><<<m, th, shmem, st>>>(
                reinterpret_cast<const float*>(add.data),
                reinterpret_cast<float*>(x.data),
                reinterpret_cast<const float*>(w.data), qp, sp, n, (long)x.pitch, eps);
            break;
        default:
            return cx::fail(err, "unsupported activation dtype code %d", (int)x.dtype);
    }
    return cx::finish(err);
}

}  // namespace

// @brief 带行间距的 RMSNorm。
//
// @param xp 激活起点，行间距 ``pitch``。
// @param wp 增益起点，``[n]``。
// @param yp 目的地起点，稠密 ``[m, n]``。
// @param m 行数。
// @param n 列数。
// @param pitch 行间距，单位元素。
// @param eps 方差下限。
// @param code dtype 码，见 ``cx::DType``。
// @param stream 目标流；空表示默认流。
// @param errbuf 错误消息缓冲，可为空。
// @param errlen 缓冲长度。
// @return ``0`` 成功。
extern "C" int coffin_rms_norm(const void* xp, const void* wp, void* yp, int m, int n,
                               long pitch, float eps, int code, void* stream,
                               char* errbuf, int errlen) {
    const cx::Err err{errbuf, errlen};
    const cx::Tensor x = cx::Tensor::matrix(const_cast<void*>(xp), m, n, pitch, (cx::DType)code);
    const cx::Tensor w = cx::Tensor::matrix(const_cast<void*>(wp), 1, n, 0, (cx::DType)code);
    const cx::Tensor y = cx::Tensor::matrix(yp, m, n, 0, (cx::DType)code);
    if (cx::check_activation(x, "activation", err) || cx::check_gain(w, x, err)) {
        return 1;
    }
    return dispatch_rms_norm(x, w, y, eps, stream, err);
}

// @brief 残差相加 + RMSNorm，``x`` 原地累加。
//
// @param addp 加数起点，行间距 ``pitch``。
// @param xp 残差起点，行间距 ``pitch``，原地写。
// @param wp 增益起点。
// @param yp 目的地起点，稠密。
// @param m 行数。
// @param n 列数。
// @param pitch 行间距。
// @param eps 方差下限。
// @param code dtype 码。
// @param stream 目标流。
// @param errbuf 错误缓冲。
// @param errlen 缓冲长度。
// @return ``0`` 成功。
extern "C" int coffin_add_rms_norm(const void* addp, void* xp, const void* wp, void* yp,
                                   int m, int n, long pitch, float eps, int code,
                                   void* stream, char* errbuf, int errlen) {
    const cx::Err err{errbuf, errlen};
    const cx::Tensor add = cx::Tensor::matrix(const_cast<void*>(addp), m, n, pitch,
                                              (cx::DType)code);
    const cx::Tensor x = cx::Tensor::matrix(xp, m, n, pitch, (cx::DType)code);
    const cx::Tensor w = cx::Tensor::matrix(const_cast<void*>(wp), 1, n, 0, (cx::DType)code);
    const cx::Tensor y = cx::Tensor::matrix(yp, m, n, 0, (cx::DType)code);
    if (cx::check_activation(x, "activation", err) ||
        cx::check_activation(add, "addend", err) ||
        cx::check_gain(w, x, err)) {
        return 1;
    }
    return dispatch_add_rms_norm(add, x, w, y, eps, stream, err);
}

// @brief 门控 RMSNorm，稠密输入。
//
// @param xp 激活起点。
// @param wp 增益起点。
// @param gatep 门起点。
// @param yp 目的地起点。
// @param m 行数。
// @param n 列数。
// @param eps 方差下限。
// @param code dtype 码。
// @param stream 目标流。
// @param errbuf 错误缓冲。
// @param errlen 缓冲长度。
// @return ``0`` 成功。
extern "C" int coffin_rms_norm_gated(const void* xp, const void* wp, const void* gatep,
                                     void* yp, int m, int n, float eps, int code,
                                     void* stream, char* errbuf, int errlen) {
    const cx::Err err{errbuf, errlen};
    const cx::Tensor x = cx::Tensor::matrix(const_cast<void*>(xp), m, n, 0, (cx::DType)code);
    const cx::Tensor w = cx::Tensor::matrix(const_cast<void*>(wp), 1, n, 0, (cx::DType)code);
    const cx::Tensor gate = cx::Tensor::matrix(const_cast<void*>(gatep), m, n, 0,
                                               (cx::DType)code);
    const cx::Tensor y = cx::Tensor::matrix(yp, m, n, 0, (cx::DType)code);
    if (cx::check_activation(x, "activation", err) ||
        cx::check_activation(gate, "gate", err) || cx::check_gain(w, x, err)) {
        return 1;
    }
    return dispatch_rms_norm_gated(x, w, gate, y, eps, stream, err);
}

// @brief 带行间距的 RMSNorm，结果直接量化成 int8（预填充 W8A8 的激活入口）。
//
// 与「``rms_norm`` 写 bf16 再接 ``quant_act_i8``」逐位相同：中间那排归一化值舍入成
// activation dtype 之后才进 absmax，量化仍走 round 后夹到 ``[-127, 127]``。所以这条
// 路径的对拍口径是逐位一致，不是「误差很小」。
//
// @param xp 激活起点，行间距 ``pitch``（可以大于 ``n``，用于列切片原地归一化）。
// @param wp 增益起点，``[n]``。
// @param qp int8 目的地，稠密 ``[m, n]``，**必须**是 ``m * n`` 连续字节。
// @param sp 每行 scale 目的地，fp32 ``[m]``。
// @param m 行数。
// @param n 列数。
// @param pitch 行间距，单位元素；``0`` 表示 ``n``。
// @param eps 方差下限。
// @param threads 每块线程数；``0`` 表示按 ``m`` 分档。
// @param code dtype 码，见 ``cx::DType``。
// @param stream 目标流；空表示默认流。
// @param errbuf 错误消息缓冲，可为空。
// @param errlen 缓冲长度。
// @return ``0`` 成功，``1`` 参数非法，``2`` 启动失败。
extern "C" int coffin_rms_norm_i8(const void* xp, const void* wp, void* qp, void* sp,
                                  int m, int n, long pitch, float eps, int threads,
                                  int code, void* stream, char* errbuf, int errlen) {
    const cx::Err err{errbuf, errlen};
    if (!xp || !wp || !qp || !sp) return cx::fail(err, "rms_norm_i8 needs non-null x/w/q/s");
    const cx::Tensor x = cx::Tensor::matrix(const_cast<void*>(xp), m, n, pitch,
                                            (cx::DType)code);
    const cx::Tensor w = cx::Tensor::matrix(const_cast<void*>(wp), 1, n, 0, (cx::DType)code);
    const cx::Tensor q = cx::Tensor::matrix(qp, m, n, 0, cx::kI8);
    const cx::Tensor s = cx::Tensor::matrix(sp, 1, m, 0, cx::kF32);
    if (cx::check_activation(x, "activation", err) || cx::check_gain(w, x, err)) {
        return 1;
    }
    return dispatch_rms_norm_i8(x, w, q, s, eps, threads, stream, err);
}

// @brief 残差相加 + RMSNorm + int8 量化，``x`` 原地累加。解码路径的 ``add_residual``
//        走这条，预填充的 ``in_ln`` / ``post_ln`` 走 :func:`coffin_rms_norm_i8`。
//
// @param addp 加数起点，行间距 ``pitch``。
// @param xp 残差起点，行间距 ``pitch``，原地写。
// @param wp 增益起点，``[n]``。
// @param qp int8 目的地，稠密 ``[m, n]``。
// @param sp 每行 scale 目的地，fp32 ``[m]``。
// @param m 行数。
// @param n 列数。
// @param pitch 行间距。
// @param eps 方差下限。
// @param threads 每块线程数；``0`` 表示按 ``m`` 分档。
// @param code dtype 码。
// @param stream 目标流。
// @param errbuf 错误缓冲。
// @param errlen 缓冲长度。
// @return ``0`` 成功，``1`` 参数非法，``2`` 启动失败。
extern "C" int coffin_add_rms_norm_i8(const void* addp, void* xp, const void* wp, void* qp,
                                      void* sp, int m, int n, long pitch, float eps,
                                      int threads, int code, void* stream, char* errbuf,
                                      int errlen) {
    const cx::Err err{errbuf, errlen};
    if (!addp || !xp || !wp || !qp || !sp) {
        return cx::fail(err, "add_rms_norm_i8 needs non-null add/x/w/q/s");
    }
    const cx::Tensor add = cx::Tensor::matrix(const_cast<void*>(addp), m, n, pitch,
                                              (cx::DType)code);
    const cx::Tensor x = cx::Tensor::matrix(xp, m, n, pitch, (cx::DType)code);
    const cx::Tensor w = cx::Tensor::matrix(const_cast<void*>(wp), 1, n, 0, (cx::DType)code);
    const cx::Tensor q = cx::Tensor::matrix(qp, m, n, 0, cx::kI8);
    const cx::Tensor s = cx::Tensor::matrix(sp, 1, m, 0, cx::kF32);
    if (cx::check_activation(x, "activation", err) ||
        cx::check_activation(add, "addend", err) ||
        cx::check_gain(w, x, err)) {
        return 1;
    }
    return dispatch_add_rms_norm_i8(add, x, w, q, s, eps, threads, stream, err);
}
