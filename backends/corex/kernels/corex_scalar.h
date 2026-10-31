// @file corex_scalar.h
// @brief CoreX 设备内核共用的位级 bf16 转换。
//
// Module: coffin/backends/corex/kernels。只被 ``.cu`` 文件包含，永远不被
// ``bindings.cpp`` 包含：这是设备代码，而绑定的编译单元由主机编译器编译，它没有
// ``__device__``。
//
// Constraints:
//   * bf16 用 ``uint16_t`` 位容器承载、靠移位转换，不经过 ``cuda_bf16.h``。这样转换
//     结果与 gfx906 内核逐位相同，也不需要向量类型头文件，CoreX 的 clang 工具链少
//     支持一样东西。
//   * 窄化按就近取偶，与 torch 转 bf16 的行为一致。截断式移位会让每个投影系统性偏
//     低，而这个偏差会穿过 32 层累积。
//   * 累加 dtype 一律 fp32。本设备的 fp64 是「有限支持」并返回 0（COREX_PORT.md
//     §0），所以即使它更精确也不能当累加器。

#pragma once

#include <cstdint>

// @brief 一种存储类型与内核累加所用 fp32 之间的转换。
//
// 两个特化覆盖内核需要的全部实例化：bf16 权重和激活，以及调用方可能持有张量的回退
// 路径所用的 fp32。
template <typename T>
struct CxScalar;

template <>
struct CxScalar<uint16_t> {
    // @brief 把一个 bf16 位模式拓宽成 fp32。
    static __device__ __forceinline__ float to_float(uint16_t v) {
        return __uint_as_float(((unsigned)v) << 16);
    }

    // @brief 把 fp32 窄化成 bf16，就近取偶。
    static __device__ __forceinline__ uint16_t from_float(float v) {
        unsigned u = __float_as_uint(v);
        u += 0x7fffu + ((u >> 16) & 1u);
        return (uint16_t)(u >> 16);
    }
};

template <>
struct CxScalar<float> {
    // @brief 恒等：fp32 实例化不拓宽任何东西。
    static __device__ __forceinline__ float to_float(float v) { return v; }

    // @brief 恒等：fp32 实例化不窄化任何东西。
    static __device__ __forceinline__ float from_float(float v) { return v; }
};

// @brief int8 权重存储：W8A16 的 GEMV 每个权重读一个有符号字节。
//
// 与这个特化搭配的另一半很要紧：int8 权重带着一个按列的 fp32 scale 过来，内核把它
// 一次性乘到累加后的点积上，所以这里的转换必须保持朴素整数拓宽。把别的东西折进
// ``to_float`` 会让 scale 被乘两次，或者乘到错误的位置。
template <>
struct CxScalar<int8_t> {
    // @brief 把一个有符号权重字节拓宽成 fp32。
    static __device__ __forceinline__ float to_float(int8_t v) { return (float)v; }

    // @brief 把 fp32 截断成 int8。
    //
    // 从不用来存权重——量化在主机侧经 ``int8_weights.QWeight.quantize`` 做舍入——但
    // 特化必须完整，通用内核模板才实例化得出来。
    static __device__ __forceinline__ int8_t from_float(float v) { return (int8_t)v; }
};
