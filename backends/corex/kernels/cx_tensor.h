// @file cx_tensor.h
// @brief 主机侧最小张量视图与错误出口：C ABI 入口（``kernels/abi/*.cu``）的唯一公共头。
//
// Module: coffin/backends/corex/kernels。只被设备编译单元包含，且**只含主机代码**：
// 一个非拥有型的 ``Tensor`` 视图（指针 + 形状 + 步长 + dtype）、dtype 码值和一套
// 写进调用方缓冲的错误上报。它不包含 ``torch/extension.h``，也不含任何 ``at::`` /
// ``c10::`` —— 这是 corex 路径上去掉 torch 之后包装层要落的类型。
//
// 为什么用「视图 + 返回码」而不是「返回值即张量 + 抛异常」：目的地由 Python 侧分配并
// 传进来，地址稳定，这样解码步才有可能被捕获重放；而异常要跨 ``ctypes`` 边界就得靠
// 每个入口自己 catch，等于每个算子多一层 try。返回码加错误缓冲是一处约定。
//
// Constraints:
//   * dtype 码值是跨语言 ABI。``rt/dtypes.py`` 里有一份同样的码表，改一处必须改两处。
//   * ``Tensor`` 不拥有内存。它由 Python 侧的 ``Array`` 折算而来，生命周期在那边。
//   * 这里的校验只覆盖「内核索引得了」这件事：二维、最内层步长为 1、dtype 受支持、
//     间距不窄于行宽。语义级的约束（谁能与谁别名）留在 ``rt/norm.py``，那里能给出带
//     数组描述的消息。
//   * 用 ``cudaGetLastError()`` 收尾而不是 ``cudaDeviceSynchronize``：异步启动失败
//     （参数非法、共享内存超限）也会立刻置错误码，而同步会把一次 20 us 的发射变成
//     一次整设备等待。

#pragma once

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

namespace cx {

//: 跨语言 dtype 码。``rt/dtypes.py`` 的 ``DType.code`` 必须与这里逐项一致。
enum DType : int {
    kF32 = 0,
    kBF16 = 1,
    kI8 = 2,
    kI32 = 3,
    kI64 = 4,
    kU8 = 5,
    kU16 = 6,
    kU32 = 7,
};

// @brief 调用方提供的错误缓冲。
//
// @param buf 目的地，可为空（空表示调用方不要消息，只要返回码）。
// @param len 字节数。
struct Err {
    char* buf;
    int len;

    // @brief 能不能写消息。
    // @return 缓冲非空且长度为正。
    bool writable() const { return buf != nullptr && len > 0; }
};

// @brief 把一条消息写进错误缓冲并返回失败码。
//
// 只在失败路径调用，所以这里用 ``vsnprintf`` 的代价可以接受：解码步的成功路径上一次
// 也不会走到它。
//
// @param err 调用方的错误缓冲。
// @param fmt ``printf`` 风格的格式串。
// @return 恒为 ``1``，方便写成 ``return cx::fail(...)``。
inline int fail(const Err& err, const char* fmt, ...) {
    if (err.writable()) {
        va_list args;
        va_start(args, fmt);
        vsnprintf(err.buf, (size_t)err.len, fmt, args);
        va_end(args);
    }
    return 1;
}

// @brief 检查内核启动是否立刻报错，并把驱动消息带回来。
//
// @param err 调用方的错误缓冲。
// @return 有错时为 ``2``，否则 ``0``。
inline int finish(const Err& err) {
    const cudaError_t code = cudaGetLastError();
    if (code == cudaSuccess) return 0;
    if (err.writable()) {
        snprintf(err.buf, (size_t)err.len, "kernel launch failed: %s (%d)",
                 cudaGetErrorString(code), (int)code);
    }
    cudaGetLastError();                          // 清掉粘滞错误，免得下次误报
    return 2;
}

// @brief 一块非拥有型二维视图：内核只需要指针、形状、间距和 dtype。
struct Tensor {
    //: 数据起点，指向矩阵第一个元素。
    void* data;
    //: 行数与列数。
    int64_t rows;
    int64_t cols;
    //: 行间距，单位元素（可以大于 ``cols``：更宽缓冲区的一列切片）。
    int64_t pitch;
    //: 存储类型。
    DType dtype;

    // @brief 造一个二维视图。
    // @param data_ 起点。
    // @param rows_ 行数。
    // @param cols_ 列数。
    // @param pitch_ 行间距，单位元素；``0`` 表示等于 ``cols``。
    // @param dtype_ 存储类型。
    // @return 视图。
    static Tensor matrix(void* data_, int64_t rows_, int64_t cols_, int64_t pitch_,
                         DType dtype_) {
        return Tensor{data_, rows_, cols_, pitch_ > 0 ? pitch_ : cols_, dtype_};
    }

    // @brief 每个元素的字节数。
    // @return 字节数；未知码值返回 ``0``。
    int64_t itemsize() const {
        switch (dtype) {
            case kF32: case kI32: case kU32: return 4;
            case kBF16: case kU16: return 2;
            case kI8: case kU8: return 1;
            case kI64: return 8;
        }
        return 0;
    }

    // @brief 是不是内核实例化的浮点类型。
    // @return bf16 或 fp32 时为真。
    bool floating16() const { return dtype == kBF16 || dtype == kF32; }
};

// @brief 内核能接受的激活布局校验。
//
// @param x 待校验视图。
// @param what 错误消息里的角色名。
// @param err 错误缓冲。
// @return 合法为 ``0``，否则为 ``cx::fail`` 的返回值。
inline int check_activation(const Tensor& x, const char* what, const Err& err) {
    if (!x.data) return fail(err, "%s has a null pointer", what);
    if (x.rows <= 0 || x.cols <= 0) {
        return fail(err, "%s must be non-empty, got [%lld, %lld]", what,
                    (long long)x.rows, (long long)x.cols);
    }
    if (x.pitch < x.cols) {
        return fail(err, "%s row pitch %lld is narrower than the row %lld", what,
                    (long long)x.pitch, (long long)x.cols);
    }
    if (x.itemsize() == 0 || !x.floating16()) {
        return fail(err, "%s dtype code %d is not bf16/fp32", what, (int)x.dtype);
    }
    return 0;
}

// @brief 增益向量的校验：元素数与 dtype 都要对上激活。
//
// @param w ``[N]`` 增益视图（``rows=1``）。
// @param x 激活视图。
// @param err 错误缓冲。
// @return 合法为 ``0``。
inline int check_gain(const Tensor& w, const Tensor& x, const Err& err) {
    if (!w.data) return fail(err, "gain has a null pointer");
    if (w.rows * w.cols != x.cols) {
        return fail(err, "gain must have N=%lld elements, got %lld", (long long)x.cols,
                    (long long)(w.rows * w.cols));
    }
    if (w.dtype != x.dtype) {
        return fail(err, "gain dtype code %d != activation %d", (int)w.dtype, (int)x.dtype);
    }
    return 0;
}

}  // namespace cx
