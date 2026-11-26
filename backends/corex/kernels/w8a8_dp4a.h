// @file w8a8_dp4a.h
// @brief 手写 int8 GEMM 的主机侧入口声明（只剩 nk 一族）。
//
// Module: coffin/backends/corex/kernels。纯接口，由主机编译器编译，所以这里没有
// ``__device__``。``bindings.cpp`` 包含它，``w8a8_dp4a.cu`` 实现它。
//
// 这里原来还声明过一个吃 ``[K, N]`` 的入口 ``w8a8_dp4a``、一个跨层共享 grid 的
// ``w8a8_dp4a_groups``，以及两条纯读 / 吞吐微基准 ``w8a8_dp4a_read`` /
// ``w8a8_dp4a_rate``。它们与它们背后的 kn 内核已整条删除：本仓库只认 NK 权重，
// 不再为 kn 留任何内核或入口，也不留只服务那份副本的微基准。
//
// Constraints:
//   * 权重是 ``[N, K]`` 连续 int8，K 方向连续；``K % 4 == 0`` 是 ``__dp4a`` 的原子粒度。
//   * 累加器是 int32。int8 x int8 的乘积精确落在 int32 里，而这条路径的精度预算已经
//     全部花在量化上，不再往里加累加误差。
//   * 每个入口都在 ``c10::cuda::getCurrentCUDAStream()`` 上启动。legacy 默认流上启动的
//     内核不属于 torch 的捕获范围。

#pragma once

#include <torch/extension.h>

#include <cstdint>
#include <vector>

// @brief 权重按 ``[N, K]``（K 连续）存放的 ``__dp4a`` int8 GEMM。
//
// 4 个相邻 int8 是 4 个相邻 k，激活侧同样连续，两边各一条 32 位加载喂一条 ``__dp4a``，
// 每条指令四次乘加。这是「每个权重字节 1/4 条指令」的那条路线；因为 nk 就是契约布局、
// 也是盘上的原生布局，这份权重不需要任何转置副本。
//
// @param a [M, K] int8 激活，连续。
// @param wt [N, K] int8 权重，K 连续；``K % 4 == 0`` 是 ``__dp4a`` 的原子粒度。
// @param c [M, N] int32 输出，连续。
// @param out dtype 为 bf16/fp32 的 ``[M, N]`` 融合输出；空表示只出 int32。
// @param xs [M] fp32 每 token 激活 scale。
// @param ws [N] fp32 每输出列权重 scale。
// @param threads 每块线程数；``0`` 或 ``-1`` 按形状选择几何（M=4/8 用小块），``128`` 固定默认几何。
// @return ``out`` 非空时返回它，否则返回 ``c``。
// @throws c10::Error 形状、dtype 布局或 ``K % 4`` 不满足。
torch::Tensor w8a8_dp4a_nk(torch::Tensor a, torch::Tensor wt, torch::Tensor c,
                           c10::optional<torch::Tensor> out, torch::Tensor xs,
                           torch::Tensor ws, int64_t threads);
