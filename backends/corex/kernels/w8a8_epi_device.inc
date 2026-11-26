// @file w8a8_epi_device.inc
// @brief W8A8 epilogue（``out = narrow(c * xs[m] * ws[n])``）的设备代码，torch 与 ABI 两条前端共用。
//
// Module: coffin/backends/corex/kernels。``w8a8.cu`` 与 ``abi/w8a8_epilog_abi.cu`` 各自
// ``#include`` 这一份，所以两条前端的算术逐位相同——两个 scale 在同一个 fp32 表达式里
// 乘完再窄化一次，穿过 32 层时不会多一次舍入。
//
// Constraints:
//   * 本文件假定调用方已经 ``#include "corex_scalar.h"``（``CxScalar`` 是窄化的唯一出处），
//     并且把 ``#include`` 放在 ``namespace { ... }`` 内：这里没有自带的命名空间。
//   * 块数固定按部件规模给（16 SM × 8），不随形状走；按形状起块会把速率交给块调度器
//     （见下方注释里的实测）。

//: epilogue 每块线程数。
constexpr int EPI_THREADS = 256;

//: epilogue 的块数。固定按部件规模给（16 个 SM x 8），不随形状走。
//
// 这正是 epilogue 从 205 GB/s 提到 260-318 GB/s 的那一处改动：按形状起块数时
// ``gate_up`` 在 M=2048 是 147456 个块，每块只做一轮 256 个元素，速率归块调度器管。
// 8 是实测最好的那一档（每 SM 8 块足够填满流水，再多只是增加调度）。
constexpr int EPI_BLOCKS = 16 * 8;

// @brief 施加两个 W8A8 scale 并窄化一次：一块负责若干行，块内线程沿 ``N`` 扫。
//
// 为什么不是「一块一行、格点按形状起」：实测这个 epilogue 只跑到拷贝上限的 34-40%
// （205 GB/s，同样字节的一次 bf16 拷贝是 484-605），而病因是块数——``gate_up`` 在
// M=2048 时按 ``(ceil(N/256), M)`` 起 72 x 2048 = 147456 个块，每块 256 个元素，
// 等于每 1 KiB 输出一个块，速率归块调度器管。把块数压到常量后微基准上是 1.19-1.35x。
//
// 为什么是「行在外、列在内」而不是一维铺平：铺平后每个元素都要用 ``i / N`` 求行号，
// 而设备上的整数除法是 20 条指令量级的序列，这个内核的算术只有两次乘。真实预填充里
// 大量调用是小 M（``o_proj`` 一类每次只有几百行），那样每个元素一次除法就把块数带来
// 的收益全吃回去——实测端到端 epilogue 从 19.04 ms 退到 21.24 ms 就是它的代价。
//
// 用 float 倒数替代除法也不行：``total`` 最大到 2.68e8，超过 float 的 24 位整数精度
// （1.68e7），行号会整个错位。
//
// **没有**跟着上 128 位向量化：实测 ``vec4`` 和 ``rowvec4`` 只有二维版的 0.4-0.9 倍。
// 这个内核本来就是连续访存，向量化只减少了指令数，却把每块并行处理的元素数除以四。
// 这与 GEMV 上「宽加载 +24%」的结论相反，因为 GEMV 每个线程读的是不相邻的列。
//
// @param c [M, N] int32 GEMM 输出，连续存放。
// @param xs [M] 每 token 激活 scale。
// @param ws [N] 每输出列权重 scale。
// @param out 目的地，行间距 ``stride``。
// @param M 行数。
// @param N 读入的行宽。
// @param stride ``out`` 的行间距；沿 N 切块时它大于 ``N``。
template <typename XT>
__global__ void w8a8_epilogue_kernel(const int32_t* __restrict__ c,
                                     const float* __restrict__ xs,
                                     const float* __restrict__ ws,
                                     XT* __restrict__ out, int M, int N,
                                     int stride) {
    for (int m = blockIdx.x; m < M; m += gridDim.x) {
        const float xm = xs[m];
        const int32_t* cr = c + (size_t)m * N;
        // 写指针按 ``stride`` 走：沿 N 切块时 ``out`` 的行距大于本块宽度，用块宽当
        // 间距会把第二行写回第一行尾部。
        XT* orow = out + (size_t)m * stride;
        for (int col = threadIdx.x; col < N; col += blockDim.x) {
            orow[col] = CxScalar<XT>::from_float((float)cr[col] * xm * ws[col]);
        }
    }
}
