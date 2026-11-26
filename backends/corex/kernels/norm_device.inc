// @file norm_device.inc
// @brief RMSNorm 三兄弟的设备代码，torch 版包装层与 C ABI shim 共用同一份。
//
// Module: coffin/backends/corex/kernels。只含设备代码：常量、共享树归约辅助、三个
// ``__global__``。它被 ``norm.cu``（torch 前端）与 ``abi/norm_abi.cu``（无 torch 前端）
// 各包含一次，所以两条前端的数值**逐位相同**——同一个函数、同一份实例化，不是两份
// 抄得像的实现。
//
// 为什么拆成 ``.inc`` 而不是头文件：它定义 ``__global__`` 与 ``__device__`` 函数，
// 只能被设备编译单元包含；``bindings.cpp``（主机编译器）绝不能看到它。``.inc`` 这个
// 后缀就是这条边界的标记。
//
// Constraints:
//   * 归约走共享内存树，不用 ``__shfl_down_sync`` 蝶形。本部件的 warp 是 64 通道，为 32
//     写的 shuffle 归约只会加一半 block 就返回一个看着合理但错误的 scale——正是
//     docs/qwen4/COREX-PLATFORM.md 记录的那种静默失败。共享树与宽度无关，临时空间在启动时按
//     ``blockDim`` 定尺寸，所以块大小保持自由参数。
//   * ``x`` 的行带间距：``q_norm``/``k_norm`` 归一化的是更宽缓冲区的一个切片，所以间距
//     是参数，永不假定等于 ``N``。
//   * 进出 bf16，求和与 rsqrt 用 fp32。用 bf16 求平方和会丢掉小量级的项，而那正是这里
//     与参考实现的全部差异。
//   * 每行一个 block，所以行宽必须装得进块上限；比块更宽的行用带步长的循环走完，而不是
//     把块放大。
//   * 这里只做设备代码。参数校验属于前端（``norm.cu`` 的 ``TORCH_CHECK`` 或
//     ``abi/norm_abi.cu`` 的 ``cx::check``），内核索引时不做边界检查。

#pragma once

#include "corex_scalar.h"

#include <cstdint>

//: 本部件的 wavefront 宽度。只用来把块大小向上取整；归约本身不依赖它（见文件头）。
constexpr int WAVEFRONT = 64;

//: 本文件启动的最宽块。2560 宽的一行需要 2560 条通道的工作量，但不需要 2560 个线程：
//: 带步长的循环用更好的占用率把剩下的覆盖掉。
constexpr int MAX_THREADS = 1024;

// @brief 通过共享树在块内对 ``v`` 求和。
//
// @param v 本线程的贡献。
// @param red 调用方提供的、``blockDim.x`` 个 float 的共享暂存；会被覆写。
// @return 整个块的和，在最后一次同步之后每个线程都拿得到。
//
// 结尾那道屏障是必需的。每个线程都要从 ``red[0]`` 把结果读出去，而没有东西能阻止另一个
// 线程在这次读取落地之前就开始下一段写 ``red`` 的共享工作。本文件里每个调用方只归约一次，
// 所以屏障今天在这里还不承重，但 ``delta.cu`` 里同形状的辅助函数每步被调两次，去掉它
// 会产生 4.5e-02 的误差。让本目录里每一次归约都保持同一个形状，才能防止第二个调用方
// 把这个 bug 重新引进来。
__device__ __forceinline__ float block_sum(float v, float* red) {
    red[threadIdx.x] = v;
    __syncthreads();
    for (int step = blockDim.x >> 1; step > 0; step >>= 1) {
        if ((int)threadIdx.x < step) red[threadIdx.x] += red[threadIdx.x + step];
        __syncthreads();
    }
    const float total = red[0];
    __syncthreads();
    return total;
}

// @brief 通过共享树在块内对 ``v`` 求最大值，形状与 :func:`block_sum` 一致。
//
// 归约形状必须和 ``block_sum`` 完全相同，理由是同一份 ``red`` 缓冲在一个内核里被
// 先后复用两次：``block_sum`` 结尾那道屏障是上一次读走 ``red[0]`` 与这一次写之间
// 唯一的隔离。让两个函数形状一致，才能防止第二个调用方把这个 bug 重新引进来。
//
// @param v 本线程的贡献。
// @param red 调用方提供的、``blockDim.x`` 个 float 的共享暂存；会被覆写。
// @return 整个块的最大值，在最后一次同步之后每个线程都拿得到。
__device__ __forceinline__ float block_max(float v, float* red) {
    red[threadIdx.x] = v;
    __syncthreads();
    for (int step = blockDim.x >> 1; step > 0; step >>= 1) {
        if ((int)threadIdx.x < step) {
            red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + step]);
        }
        __syncthreads();
    }
    const float total = red[0];
    __syncthreads();
    return total;
}

// @brief 宽度为 ``N`` 的一行要用多少线程，向上取整到 wavefront 宽度。
//
// 这个数在这里比在 ``add_rms_norm`` 里要紧得多，因为融合版一行要做两次归约、三趟
// 遍历，而块内每次归约都是一串全块屏障。线程越多，屏障的串行代价越高；线程越少，
// 每线程的循环轮数和访存并行越低。实测（``probe_rmsnorm_i8.py``，M=2048、N=2560）
// 这个取舍的最优点见 :func:`i8_fused_threads`，不是 ``MAX_THREADS``。
//
// @param N 行宽。
// @return ``min(MAX_THREADS, round_up(N, WAVEFRONT))``，至少一个 wavefront。
__host__ inline int rms_threads(int N) {
    int t = N < WAVEFRONT ? WAVEFRONT : N;
    t = ((t + WAVEFRONT - 1) / WAVEFRONT) * WAVEFRONT;
    return t > MAX_THREADS ? MAX_THREADS : t;
}

//: 融合 ``add_rms_norm_i8`` 的每块线程数。
//:
//: 与 :func:`rms_threads` 分开，因为两个内核的瓶颈不同：``add_rms_norm`` 只有一次
//: 归约，多给线程只赚并行度；融合版有两次归约和三趟遍历，**块内屏障是串行项**。
//:
//: 而且最优点随 ``M`` 走，因为 ``M`` 就是块数：块少时要靠块内线程凑并行度，块多时
//: 线程只加屏障代价。实测（N=2560，单位 ms）：
//:
//:   M      128      256      512      1024
//:   1      0.0216   0.0148   0.0124   0.0124   <- 1 个块，线程越多越好
//:   2      0.0211   0.0158   0.0130   0.0123
//:   64     0.0226   0.0176   0.0150   0.0164   <- 512 最好
//:   2048   0.1185   0.0862   0.1001   0.1868   <- 2048 个块，256 最好
//:
//: 所以按块数分档，不是一个常数。分界放在 8 和 512：块数少于 8（半个设备的 SM 数）
//: 时块内并行度是稀缺资源；多于 512 时块间并行足够，屏障代价开始主导。
//:
//: @param M 行数，也就是块数。
//: @return 该行的实测最优点线程数。
__host__ inline int i8_fused_threads(int M) {
    if (M < 8) return 1024;
    if (M < 512) return 512;
    return 256;
}

// @brief ``y = x * rsqrt(mean(x^2) + eps) * w``，每行一个 block。
template <typename T>
__global__ void rms_norm_kernel(const T* __restrict__ x, const T* __restrict__ w,
                                T* __restrict__ y, int N, long stride, float eps) {
    extern __shared__ float red[];
    const int row = blockIdx.x;
    const int t = threadIdx.x;
    const T* xr = x + (size_t)row * stride;
    float ss = 0.f;
    for (int i = t; i < N; i += blockDim.x) {
        const float v = CxScalar<T>::to_float(xr[i]);
        ss += v * v;
    }
    const float inv = rsqrtf(block_sum(ss, red) / (float)N + eps);
    T* yr = y + (size_t)row * N;
    for (int i = t; i < N; i += blockDim.x) {
        yr[i] = CxScalar<T>::from_float(CxScalar<T>::to_float(xr[i]) * inv
                                        * CxScalar<T>::to_float(w[i]));
    }
}

// @brief 先 ``x += add``，再 ``y = rms(x) * w``，每行一个 block。
//
// 残差 *原地* 累加进 ``x``，也就是这一层在整步剩下的时间里要继续用的那个缓冲区；归一化
// 结果写进单独的 ``y``。两个操作之所以融合，原因全在这里：拆开的写法把和写进 ``x``，
// 然后归一化再读一遍 ``x``，而按本卡 5.75 us 的启动下限，两趟各要一次启动。
//
// 两个细节让结果与两算子写法逐位相同而不只是接近，且两者都不是可选项：
//
//   * 和在它的平方进入归约之前先舍入成 ``T``。第二次启动读到的正是 bf16 存储的和，
//     所以这里累加 *未* 舍入的 fp32 和会算出不同的均方。
//   * 增益在第 2 趟施加，值是从 ``x`` 经 ``to_float`` 读回来的，和 ``rms_norm_kernel``
//     完全一样。把乘法折进第 1 趟会改变两次舍入的先后顺序。
//
// @param add 待累加的混合器输出，行间距与 ``x`` 一样是 ``stride``。可以与 ``y`` 重叠：
//        前端关于重叠的说明解释了为什么允许，以及为什么这两个指针因此必须保持
//        非 ``__restrict__``。
// @param x 残差流，原地累加；行间距 ``stride``。
// @param w [N] 增益，已经带着 checkpoint 的 ``1 + w`` 偏移。
// @param y [M, N] 稠密打包的输出；不得与 ``x`` 重叠。
// @param N 行宽。
// @param stride ``x`` 和 ``add`` 行之间的元素间距。
// @param eps 方差下限。
//
// ``add`` 和 ``y`` 故意 *不* 是 ``__restrict__``。前端允许它们共享存储，而
// ``__restrict__`` 会授权编译器把 ``add`` 的一次加载保留到第 1 趟对 ``y`` 的存储之外
// ——这是只在卡上才出现的误编译，因为主机侧检查一次只跑一条通道。``x`` 和 ``w`` 保持
// restrict：前端拒绝 ``y`` 与 ``x`` 重叠，而 ``w`` 是只读增益。
template <typename T>
__global__ void add_rms_norm_kernel(const T* add, T* x, const T* __restrict__ w, T* y,
                                    int N, long stride, float eps) {
    extern __shared__ float red[];
    const int row = blockIdx.x;
    const int t = threadIdx.x;
    const T* ar = add + (size_t)row * stride;
    T* xr = x + (size_t)row * stride;
    // 一个 block 独占整行，所以把和写回这里、下面再读一遍是安全的：归约自带的屏障把
    // 两趟隔开了。
    float ss = 0.f;
    for (int i = t; i < N; i += blockDim.x) {
        const float s = CxScalar<T>::to_float(xr[i]) + CxScalar<T>::to_float(ar[i]);
        const T sum = CxScalar<T>::from_float(s);
        xr[i] = sum;
        const float v = CxScalar<T>::to_float(sum);
        ss += v * v;
    }
    const float inv = rsqrtf(block_sum(ss, red) / (float)N + eps);
    T* yr = y + (size_t)row * N;
    for (int i = t; i < N; i += blockDim.x) {
        yr[i] = CxScalar<T>::from_float(CxScalar<T>::to_float(xr[i]) * inv
                                        * CxScalar<T>::to_float(w[i]));
    }
}

// @brief ``y = rms(x) * w * silu(gate)``，每行一个 block。
template <typename T>
__global__ void rms_norm_gated_kernel(const T* __restrict__ x, const T* __restrict__ w,
                                      const T* __restrict__ gate, T* __restrict__ y,
                                      int N, float eps) {
    extern __shared__ float red[];
    const int row = blockIdx.x;
    const int t = threadIdx.x;
    const T* xr = x + (size_t)row * N;
    float ss = 0.f;
    for (int i = t; i < N; i += blockDim.x) {
        const float v = CxScalar<T>::to_float(xr[i]);
        ss += v * v;
    }
    const float inv = rsqrtf(block_sum(ss, red) / (float)N + eps);
    T* yr = y + (size_t)row * N;
    const T* gr = gate + (size_t)row * N;
    for (int i = t; i < N; i += blockDim.x) {
        const float gv = CxScalar<T>::to_float(gr[i]);
        const float silu = gv / (1.f + __expf(-gv));
        yr[i] = CxScalar<T>::from_float(
            CxScalar<T>::to_float(xr[i]) * inv * CxScalar<T>::to_float(w[i]) * silu);
    }
}

// @brief 先 ``x += add``，再 ``y = rms(x) * w`` 并直接量化成 int8，每行一个 block。
//
// 这个变体存在的理由是节省字节。拆开的写法要走四趟 global（``add_rms_norm`` 读
// ``x``+``add``、写 ``x``、读 ``x``、写 bf16 ``y``；``quant_act_i8`` 再读两遍 ``y``
// 才写 int8），M=2048 × N=2560 上合计 **78.7 MB**。融合版把残差和留在 LDS 里，
// 只剩三趟：读 ``x``+``add``、写 ``x``、写 int8，**36.7 MB**。
//
// 两次归约在同一维上做——均方在归一化之前，归一化结果的 absmax 在之后。中间那排
// 归一化值按 ``T`` 存进 LDS 并**原地改写成最终值**：第二趟读进来的残差和在算完
// ``y`` 之后就没用了，同一个下标一个线程从头管到尾，所以覆写是安全的。
//
// 三处让结果与「``add_rms_norm`` 写 bf16 + ``quant_act_i8`` 读回来」逐位相同，
// 缺一不可：
//
//   * 和在进均方之前先舍入成 ``T``——拆开跑时第二次启动读到的是 bf16 存储的和；
//   * absmax 取自**舍入后**的 ``y``，不是未舍入的 fp32 值，否则 scale 会差几个 ULP，
//     让贴近边界的 code 差 1（实测 max|dq|=1.0、max|ds|~1e-4 而残差完全相同）；
//   * 量化仍走 ``round`` 后夹到 ``[-127, 127]``，与 ``quant_act_i8`` 同一套约定。
//
// @param add 待累加的混合器输出，行间距 ``stride``。
// @param x 残差流，原地累加；行间距 ``stride``。
// @param w [N] 增益，已经带着 checkpoint 的 ``1 + w`` 偏移。
// @param out_q [M, N] int8 目的地。
// @param out_s [M] 每行 fp32 scale 目的地。
// @param N 行宽。
// @param stride ``x`` 和 ``add`` 行之间的元素间距。
// @param eps 方差下限。
template <typename T>
__global__ void add_rms_norm_i8_kernel(const T* __restrict__ add, T* __restrict__ x,
                                       const T* __restrict__ w, int8_t* __restrict__ out_q,
                                       float* __restrict__ out_s, int N, long stride,
                                       float eps) {
    extern __shared__ float red[];
    // 归约临时量在前（``blockDim.x`` 个 float），舍入后的行紧随其后（N 个 T）。
    // 两个缓冲不重叠：归约只碰前一段，行只碰后一段。
    T* srow = reinterpret_cast<T*>(red + blockDim.x);
    const int row = blockIdx.x;
    const int t = threadIdx.x;
    const T* ar = add + (size_t)row * stride;
    T* xr = x + (size_t)row * stride;

    // 第一趟：残差相加（和舍入成 T 后原地写回 global，它还要活过这一层），
    // 同时累加平方和，并把和存进 LDS 供第二趟用——省掉再读一遍 global ``x``。
    float ss = 0.f;
    for (int i = t; i < N; i += blockDim.x) {
        const float s = CxScalar<T>::to_float(xr[i]) + CxScalar<T>::to_float(ar[i]);
        const T sum = CxScalar<T>::from_float(s);
        xr[i] = sum;
        srow[i] = sum;
        const float v = CxScalar<T>::to_float(sum);
        ss += v * v;
    }
    const float inv = rsqrtf(block_sum(ss, red) / (float)N + eps);

    // 第二趟：从 LDS 读回舍入过的和，算出归一化值、**舍入成 T 后就地覆写**，
    // 同时取 absmax。覆写安全：这个下标往后只需要 ``y``，不再需要那个和。
    float amax = 0.f;
    for (int i = t; i < N; i += blockDim.x) {
        const T y = CxScalar<T>::from_float(
            CxScalar<T>::to_float(srow[i]) * inv * CxScalar<T>::to_float(w[i]));
        srow[i] = y;
        amax = fmaxf(amax, fabsf(CxScalar<T>::to_float(y)));
    }
    // ``1e-5`` 的下限与 ``quant_act_i8`` 一致：全零行量化成零，而不是除出 inf。
    const float sc = fmaxf(block_max(amax, red), 1e-5f) / 127.0f;
    const float rsc = 1.0f / sc;
    if (t == 0) out_s[row] = sc;

    // 第三趟：从 LDS 读回那个已经舍入过的值并量化。这是 ``quant_act_i8`` 看到的
    // 同一批字节，所以两张表逐位相同。
    int8_t* qr = out_q + (size_t)row * N;
    for (int i = t; i < N; i += blockDim.x) {
        int q = __float2int_rn(CxScalar<T>::to_float(srow[i]) * rsc);
        q = max(-127, min(127, q));
        qr[i] = (int8_t)q;
    }
}

// @brief ``y = rms(x) * w`` 并直接量化成 int8，每行一个 block。
//
// **这是预填充真正会走的那条。** 预填充的层是 ``mix.prefill_project(self.in_ln(x))``
// 和 ``self.mlp.prefill(self.post_ln(x))`` —— 两次都是**普通** RMSNorm，残差相加是
// 另一条 ``x = x + h`` 的张量运算。``add_rms_norm_i8`` 对应的其实是解码路径的
// ``add_residual``，不是预填充；两条都得有，否则接的时候会发现融合位对不上。
//
// 与 ``rms_norm`` 写 bf16 再接 ``quant_act_i8`` 逐位相同：中间那排归一化值舍入成
// ``T`` 之后才进 absmax，量化仍走 round 后夹到 ``[-127, 127]``。
//
// @param x [M, N] 激活，行间距 ``stride``（``q_norm``/``k_norm`` 会传更宽的间距）。
// @param w [N] 增益，已经带着 checkpoint 的 ``1 + w`` 偏移。
// @param out_q [M, N] int8 目的地，稠密。
// @param out_s [M] 每行 fp32 scale 目的地。
// @param N 行宽。
// @param stride ``x`` 行之间的元素间距。
// @param eps 方差下限。
template <typename T>
__global__ void rms_norm_i8_kernel(const T* __restrict__ x, const T* __restrict__ w,
                                   int8_t* __restrict__ out_q, float* __restrict__ out_s,
                                   int N, long stride, float eps) {
    extern __shared__ float red[];
    T* srow = reinterpret_cast<T*>(red + blockDim.x);
    const int row = blockIdx.x;
    const int t = threadIdx.x;
    const T* xr = x + (size_t)row * stride;

    float ss = 0.f;
    for (int i = t; i < N; i += blockDim.x) {
        const float v = CxScalar<T>::to_float(xr[i]);
        srow[i] = xr[i];
        ss += v * v;
    }
    const float inv = rsqrtf(block_sum(ss, red) / (float)N + eps);

    // 与 ``add_rms_norm_i8`` 同一条理由：absmax 取自舍入成 ``T`` 之后的值，因为拆开跑
    // 时 ``quant_act_i8`` 读到的是 bf16 存储的那一份。取未舍入的 fp32 会让 scale 差
    // 几个 ULP，贴近边界的 code 差 1。
    float amax = 0.f;
    for (int i = t; i < N; i += blockDim.x) {
        const T y = CxScalar<T>::from_float(
            CxScalar<T>::to_float(srow[i]) * inv * CxScalar<T>::to_float(w[i]));
        srow[i] = y;
        amax = fmaxf(amax, fabsf(CxScalar<T>::to_float(y)));
    }
    const float sc = fmaxf(block_max(amax, red), 1e-5f) / 127.0f;
    const float rsc = 1.0f / sc;
    if (t == 0) out_s[row] = sc;

    int8_t* qr = out_q + (size_t)row * N;
    for (int i = t; i < N; i += blockDim.x) {
        int q = __float2int_rn(CxScalar<T>::to_float(srow[i]) * rsc);
        q = max(-127, min(127, q));
        qr[i] = (int8_t)q;
    }
}
