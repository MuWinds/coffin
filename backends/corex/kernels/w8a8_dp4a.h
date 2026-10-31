// @file w8a8_dp4a.h
// @brief 手写 int8 GEMM 的主机侧入口声明。
//
// Module: coffin/backends/corex/kernels。纯接口，由主机编译器编译，所以这里没有
// ``__device__``。``bindings.cpp`` 包含它，``w8a8_dp4a.cu`` 实现它。
//
// 为什么在 ``w8a8_gemm`` 之外还要一个入口：``w8a8_gemm`` 的契约是"一次 matmul"，而
// 厂商库的 tile、lane 宽度和 grid 全都改不了，三条方向里有两条（4 字节 lane / 512 列、
// 跨层共享 grid）恰恰就是这三样。这个入口把它们露出来：
//
//   * ``ntile`` —— 一个线程一次读几个相邻 int8，所以一个 wavefront 一次在飞
//     ``64 * ntile`` 字节。``ntile = 4`` 就是 D1 要的「4 字节 lane」。
//   * ``groups`` 版本让一次启动覆盖多个形状，每个形状带自己的权重指针和列窗口。
//
// 每个几何参数给 ``-1`` 表示按形状自己选。传了就原样使用，主机侧拒绝索引不了的组合，
// 而不是悄悄四舍五入——一个被静默修正的配置会让扫描表里的一整行描述一个没跑过的内核。
//
// Constraints:
//   * 权重是 ``[K, N]`` 连续 int8（``Caps.weight_layout == "kn"``）。列方向是连续字节，
//     所以行首对齐只要求 ``N % ntile == 0``；不满足时主机侧退回 ``ntile = 1``，而不是
//     让内核跨行读并算出看着合理的数。
//   * 累加器是 int32。int8 x int8 的乘积精确落在 int32 里，而这条路径的精度预算已经
//     全部花在量化上，不再往里加累加误差。
//   * 每个入口都在 ``c10::cuda::getCurrentCUDAStream()`` 上启动。legacy 默认流上启动的
//     内核不属于 torch 的捕获范围。

#pragma once

#include <torch/extension.h>

#include <cstdint>
#include <vector>

// @brief 手写 int8 GEMM：``c[m, n] = sum_k a[m, k] * w[k, n]``，权重保持 ``[K, N]``。
//
// 这是 D1 的落点：一个线程每 k 步读 ``ntile`` 个相邻列字节，``ntile = 4`` 时一次 32 位
// 加载，一个 wavefront 一次在飞 256 字节的连续段。归约轴是 K，所以这条路径每读一个
// 字节要一次整数乘加——M 小时算术藏在访存后面，M 大时它就是 ALU 墙。这条边界正是
// 预填充与解码的分界，要量出来。
//
// @param a [M, K] int8 激活，连续。
// @param w [K, N] int8 权重，连续。
// @param c [M, N] int32 输出，连续。
// @param out dtype 为 bf16/fp32 的 ``[M, N]`` 融合输出；空表示只出 int32。
// @param xs [M] fp32 每 token 激活 scale。
// @param ws [N] fp32 每输出列权重 scale。
// @param partial [split, M, N] int32 的 split-K 缓冲；``split > 1`` 且 ``out`` 非空时必填。
// @param mtile 每线程负责的 M 行数，合法值 ``1``/``2``/``4``；``-1`` 表示按 M 选。
// @param ntile 每线程负责的相邻列数，合法值 ``1``/``2``/``4``；``-1`` 表示按 N 选。
// @param ktile 保留参数；``K`` 方向的展开由 :func:`w8a8_dp4a_nk` 单独控制。
// @param threads 每块线程数；``-1`` 表示 128。
// @param split 沿 K 的切片数；``-1`` 表示按块数自动补齐。
// @return ``out`` 非空时返回它，否则返回 ``c``。
// @throws c10::Error 形状、dtype、布局或几何是内核索引不了的。
torch::Tensor w8a8_dp4a(torch::Tensor a, torch::Tensor w, torch::Tensor c,
                        c10::optional<torch::Tensor> out, torch::Tensor xs,
                        torch::Tensor ws, c10::optional<torch::Tensor> partial,
                        int64_t mtile, int64_t ntile, int64_t ktile,
                        int64_t threads, int64_t split);

// @brief 权重按 ``[N, K]``（K 连续）存放的 ``__dp4a`` int8 GEMM。
//
// 4 个相邻 int8 是 4 个相邻 k，激活侧同样连续，两边各一条 32 位加载喂一条 ``__dp4a``，
// 每条指令四次乘加。这是「每个权重字节 1/4 条指令」的那条路线，代价是权重得有一份
// ``[N, K]`` 布局——所以它的服务成本必须把转置算进去再和库比。
//
// @param a [M, K] int8 激活，连续。
// @param wt [N, K] int8 权重，K 连续；``K % 4 == 0`` 是 ``__dp4a`` 的原子粒度。
// @param c [M, N] int32 输出，连续。
// @param out dtype 为 bf16/fp32 的 ``[M, N]`` 融合输出；空表示只出 int32。
// @param xs [M] fp32 每 token 激活 scale。
// @param ws [N] fp32 每输出列权重 scale。
// @param threads 每块线程数；``0`` 或 ``-1`` 表示内核固定的值。
// @return ``out`` 非空时返回它，否则返回 ``c``。
// @throws c10::Error 形状、dtype、布局或 ``K % 4`` 不满足。
torch::Tensor w8a8_dp4a_nk(torch::Tensor a, torch::Tensor wt, torch::Tensor c,
                           c10::optional<torch::Tensor> out, torch::Tensor xs,
                           torch::Tensor ws, int64_t threads);

// @brief 一次启动跑多个形状：D3 的跨层共享 grid。
//
// 每个形状的块被拼进同一个 grid。窄形状单趟只有十个块、16 个 SM 连一个 wave 都不够；
// 把它和别的形状并进同一次启动，波次才排得满，而这不需要发明任何字节，也不需要先把
// 权重拷成一个大矩阵——那份额外拷贝正是这个想法本来要避开的成本。
//
// @param a [M, K] int8 激活，所有形状共用（同一层的 token 相同）。
// @param group_w 每个形状的 ``[K, n_i]`` int8 权重。
// @param group_ws 每个形状的 fp32 scale 向量。
// @param c [M, 各形状 n 之和] int32 输出，按 group 顺序横向拼接。
// @param out 与 ``c`` 同形的 bf16/fp32 融合输出；空表示只出 int32。
// @param xs [M] fp32 每 token 激活 scale。
// @param mtile/ntile/threads 同上，对所有形状统一。
// @param ktile 保留参数。
// @param split 保留参数；跨层 grid 目前只做单趟。
// @return ``out`` 非空时返回它，否则返回 ``c``。
// @throws c10::Error 任一形状的列数、dtype 或几何是内核索引不了的。
torch::Tensor w8a8_dp4a_groups(torch::Tensor a, std::vector<torch::Tensor> group_w,
                               std::vector<torch::Tensor> group_ws, torch::Tensor c,
                               c10::optional<torch::Tensor> out, torch::Tensor xs,
                               int64_t mtile, int64_t ntile, int64_t ktile,
                               int64_t threads, int64_t split);

// @brief 与 GEMM 同一套寻址的只读内核：把权重读一遍，不做点积。
//
// D1 的对照项。GEMM 的实测带宽必须和**同一几何下**的纯读上限比，才能说清差额是算术的
// 还是访存的；拿别的几何的纯读数字当上限，会把两次测量的形状差异算成内核缺陷
// （``GEMV_CEILING.md`` §1 里那条「1 字节 lane 比 16 字节还快」的假结论就是这么来的）。
//
// @param w [K, N] int8 权重，连续。
// @param out [块数] int64 每块一个累加和，避免整个内核被优化掉。
// @param ntile 每线程一次读的相邻字节数。
// @param threads 每块线程数。
// @param blocks 目标块数；``0`` 表示按 SM 数 x 32。
// @return ``out``。
// @throws c10::Error 布局或几何是内核索引不了的。
torch::Tensor w8a8_dp4a_read(torch::Tensor w, torch::Tensor out, int64_t ntile,
                             int64_t threads, int64_t blocks);

// @brief ``__dp4a`` 与整数乘加的吞吐微基准。
//
// 这条数字决定手写 GEMM 的上限，而它是预填充那条路线的判据：厂商库在本卡上跑到
// 49.07 TINT8-OPS/s，``__dp4a`` 一条指令只做四次 int8 乘加。如果把它换算出的峰值
// 低于 49.07，「手写 dp4a 在预填充上追平库」就不成立，与实现好坏无关。
//
// @param out [块数] int64 每块一个累加和。
// @param iters 每线程的循环次数。
// @param mode ``0`` 走 ``__dp4a``，``1`` 走 ``int`` 乘加作为对照。
// @param blocks 块数；``0`` 表示按 SM 数 x 32。
// @return ``out``。
torch::Tensor w8a8_dp4a_rate(torch::Tensor out, int64_t iters, int64_t mode,
                             int64_t blocks);
