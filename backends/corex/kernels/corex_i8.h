// @file corex_i8.h
// @brief int8 权重读取的宽加载原语，W8A16 GEMV 与 W8A8 GEMM 共用。
//
// Module: coffin/backends/corex/kernels。只被 ``.cu`` 文件包含，永不被
// ``bindings.cpp`` 包含：这些都是 ``__device__``，而绑定的编译单元由主机编译器编译。
//
// Constraints:
//   * 宽度只允许 1/2/4/8/16。``W`` 同时是线程持有的相邻列数和它每个 k 步读的字节数，
//     而 ``N % W == 0`` 是行首对齐的前提；不整除的模型必须退回标量内核，否则会跨行读
//     并算出看着合理的数。
//   * ``W = 16`` 是两次 64 位加载，因为没有 16 字节的标量整数。它实测比 4 更差
//     （374 vs 442 GB/s，GEMV_CEILING.md §3），列进来只是因为模板让它免费。
//   * 字必须是无符号的：对有符号字做算术右移会把高位字节符号扩展进下一个字节。

#pragma once

#include <cstdint>

// @brief ``W`` 宽的列加载用哪种整数字来执行。
//
// 决定事务效率的宽度来自本部件 64 通道 wavefront 对 128 字节行的关系：W=1 时每个
// wavefront 只有 64 字节在飞（半行），W=2 正好 128，W=4 是 256。读得更宽同时把线程数
// 除以 ``W``，而 16 个 SM 上线程是稀缺资源，所以往哪边倒要靠实测。
template <int W>
struct CxWideWord;

template <>
struct CxWideWord<1> {
    using type = uint8_t;
    static constexpr int kLoads = 1;
};

template <>
struct CxWideWord<2> {
    using type = uint16_t;
    static constexpr int kLoads = 1;
};

template <>
struct CxWideWord<4> {
    using type = uint32_t;
    static constexpr int kLoads = 1;
};

template <>
struct CxWideWord<8> {
    using type = uint64_t;
    static constexpr int kLoads = 1;
};

template <>
struct CxWideWord<16> {
    using type = uint64_t;
    static constexpr int kLoads = 2;
};

// @brief 从 ``p`` 开始读 ``W`` 个相邻的 int8 字节。
//
// @param p 本线程列组的第一个字节；按 ``W`` 对齐。
// @param out 接收按列升序排列的 ``W`` 个有符号字节。
template <int W>
__device__ __forceinline__ void cx_load_i8_row(const int8_t* __restrict__ p,
                                               int8_t (&out)[W]) {
    using LT = typename CxWideWord<W>::type;
    constexpr int kLoads = CxWideWord<W>::kLoads;
    constexpr int kPer = (int)sizeof(LT);
#pragma unroll
    for (int i = 0; i < kLoads; ++i) {
        const LT word = *reinterpret_cast<const LT*>(p + i * kPer);
#pragma unroll
        for (int b = 0; b < kPer; ++b) {
            out[i * kPer + b] = (int8_t)((word >> (8 * b)) & 0xff);
        }
    }
}
