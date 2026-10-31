// @file gemv_wide.cu
// @brief 投机校验用的 Wide-M int8 W8A16 GEMV（M 取 {16, 24, 32}）。
//
// Module: coffin/kernels。``hip_ops.py`` 后面的校验专用快路径；它扩展了
// ``gemv_int8.cu`` 里 M <= 8 那一族，必须与之保持逐位兼容。
//
// 这个文件为什么存在：一个校验步要为每个序列行的一次权重读取投影 K+1 个激活向量；B=8、
// K=1 时就是 M=16，K=3 时是 M=32。M=8 以上引擎没有批处理内核（``gemv_w8a16_m`` 只覆盖
// {2,4,8}），所以校验要么拆成若干次 M=8（权重字节每趟重读一遍），要么落进预填充 GEMM 路径
// （int8 -> fp16 反量化 + hipBLAS，投影这一段 250-290 ms）。本内核一趟覆盖 M=16/24/32，
// 不反量化、不走 hipBLAS。
//
// 并行方式：M 被切成每组 MW 个激活；一个 warp 独占 R 个输出行和这些行里的 MW 个激活向量。
// 一个行组内的 MT/MW 个 warp 读同一批权重行，所以多出来的组命中缓存而不是多走一趟 HBM
// （组在同一个 block 内）。``acc`` 是 R*MW*8 个 fp32，所以随版本发布的每个组合寄存器压力都
// 有界（全部都是 64）。
//
// Constraints:
//   * 逐位一致是硬性验收条件：每个（行，激活）的算术就是 M=8 内核的算术，token 级一致。
//     探针拿一次 wide 调用的结果与若干次 M=8 调用拼接后的结果对比，任何算术改动都过不了。
//   * 权重是 INT8 行主序 [N, K] 配每行一个 fp32 scale；新的量化布局要同时改这个内核和它的
//     探针。
//   * block 的 grid 必须能均匀分进行组（``wpb % GM == 0``）；其余情况分派器直接拒绝，而不是
//     启动一个覆盖不全的 grid。
//
// 逐位一致，细说。内核为每个（行，激活）保持 M=8 内核*精确*的算术：
//
//   * K 以 8 个 int8 为一块遍历，lane l 拥有编号 l、l+32、l+64……的块——与
//     ``gemv_w8a16_m_kernel`` 相同的「块到 lane」映射，所以每个 lane 把同样的 K 位置累加进
//     同样的八条链；
//   * 第 j 条链按块号升序取该块的第 j 个 int8；
//   * 最后的归约按顺序相加 0..7 号链（在标量 K%8 尾数之后），然后做同一棵 __shfl_down 归约
//     树，逐行 scale 最后才乘。
//
// 因此单独看一行，无论用哪个 U、R 或 MW，结果都与对该行的 8 激活切片调用
// ``gemv_w8a16_m`` 逐位一致（U 只改变预取多少块，不改变块的顺序）。

#include "hip_ops.h"

#include <c10/cuda/CUDAStream.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace {

__device__ __forceinline__ float bf16_to_float(unsigned short h) {
    return __uint_as_float(((unsigned)h) << 16);
}

__device__ __forceinline__ unsigned short float_to_bf16(float f) {
    unsigned u = __float_as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);   // 就近舍入到偶数（round to nearest even）
    return (unsigned short)(u >> 16);
}

// 一个 32 位字里的四个 int8 -> 四个 float（用向量类型：留在 VGPR 里）。
__device__ __forceinline__ float4 i8x4(unsigned w) {
    float4 f;
    f.x = (float)(int)(signed char)(w & 0xffu);
    f.y = (float)(int)(signed char)((w >> 8) & 0xffu);
    f.z = (float)(int)(signed char)((w >> 16) & 0xffu);
    f.w = (float)(int)(signed char)((w >> 24) & 0xffu);
    return f;
}

// 一个 32 位字里的两个 bf16 -> 两个 float。
__device__ __forceinline__ float2 bf16x2(unsigned u) {
    return make_float2(bf16_to_float((unsigned short)(u & 0xffffu)),
                       bf16_to_float((unsigned short)(u >> 16)));
}

// 一个（行，激活）对的八条 FMA 链，放在两个 float4 里。
struct Acc8 {
    float4 a, b;
};

// 一个 8 个 int8 的权重块施加到一个 8 个 bf16 的激活块上。
// ``w0`` 装着权重 0..3（块里第 0..3 个字节），``w1`` 装着权重 4..7。
__device__ __forceinline__ void fma8(Acc8& acc, const float4& w0, const float4& w1,
                                     const uint4& xb) {
    const float2 p0 = bf16x2(xb.x);   // 激活 0,1
    const float2 p1 = bf16x2(xb.y);   // 2,3
    const float2 p2 = bf16x2(xb.z);   // 4,5
    const float2 p3 = bf16x2(xb.w);   // 6,7
    acc.a.x += w0.x * p0.x;
    acc.a.y += w0.y * p0.y;
    acc.a.z += w0.z * p1.x;
    acc.a.w += w0.w * p1.y;
    acc.b.x += w1.x * p2.x;
    acc.b.y += w1.y * p2.y;
    acc.b.z += w1.z * p3.x;
    acc.b.w += w1.w * p3.y;
}

// grid：warp ``wid`` 独占激活组 ``wid % GM`` 和行组
// ``blockIdx.x * (wpb / GM) + wid / GM``；一个行组是 R 个连续行。
// block 布局只是 MT、R、MW、U 的函数——随版本发布的每个组合都编译期实例化，好让
// ``acc``/``tail`` 留在 VGPR 里。
//
// @param W [N, K] int8 行主序；第 ``n`` 行随后乘 ``S[n]``。
// @param X [M, K] bf16 激活；M 即 MT，一次权重读取服务全部。
// @param Y [M, N] bf16 输出；行主序，由每个 warp 的 lane 0 写出。
template <int MT, int R, int MW, int U>
__global__ void gemv_w8a16_wide_kernel(const int8_t* __restrict__ W,
                                       const float* __restrict__ S,
                                       const uint16_t* __restrict__ X,
                                       uint16_t* __restrict__ Y, int N, int K) {
    constexpr int GM = MT / MW;
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int wpb = blockDim.x >> 5;
    const int mg = wid % GM;
    const int rg = wid / GM;
    const int rowgroup = blockIdx.x * (wpb / GM) + rg;
    const int r0 = rowgroup * R;
    if (r0 >= N) return;

    const int am0 = mg * MW;
    const int nvec = K >> 3;
    const uint2* wv = reinterpret_cast<const uint2*>(W + (size_t)r0 * K);
    const uint4* xv = reinterpret_cast<const uint4*>(X);

    Acc8 acc[R][MW];
    float tail[R][MW];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int m = 0; m < MW; ++m) {
            acc[r][m].a = make_float4(0.f, 0.f, 0.f, 0.f);
            acc[r][m].b = make_float4(0.f, 0.f, 0.f, 0.f);
            tail[r][m] = 0.f;
        }

    const int step = 32 * U;
    int base = 0;
    for (; base + step <= nvec; base += step) {
        uint2 wc[R][U];
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int u = 0; u < U; ++u)
                wc[r][u] = wv[(size_t)r * nvec + base + u * 32 + lane];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int i = base + u * 32 + lane;
            uint4 xb[MW];
#pragma unroll
            for (int m = 0; m < MW; ++m) xb[m] = xv[(size_t)(am0 + m) * nvec + i];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const float4 w0 = i8x4(wc[r][u].x);
                const float4 w1 = i8x4(wc[r][u].y);
#pragma unroll
                for (int m = 0; m < MW; ++m) fma8(acc[r][m], w0, w1, xb[m]);
            }
        }
    }
    for (int i = base + lane; i < nvec; i += 32) {
        uint4 xb[MW];
#pragma unroll
        for (int m = 0; m < MW; ++m) xb[m] = xv[(size_t)(am0 + m) * nvec + i];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint2 wc = wv[(size_t)r * nvec + i];
            const float4 w0 = i8x4(wc.x);
            const float4 w1 = i8x4(wc.y);
#pragma unroll
            for (int m = 0; m < MW; ++m) fma8(acc[r][m], w0, w1, xb[m]);
        }
    }
    // K 不是 8 的整数倍时走标量尾循环，并且与 M=8 内核一样在链求和之前加进来。
    for (int k = (nvec << 3) + lane; k < K; k += 32) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (r0 + r >= N) break;
            const float wf = (float)W[(size_t)(r0 + r) * K + k];
#pragma unroll
            for (int m = 0; m < MW; ++m)
                tail[r][m] = fmaf(wf, bf16_to_float(X[(size_t)(am0 + m) * K + k]),
                                  tail[r][m]);
        }
    }

#pragma unroll
    for (int r = 0; r < R; ++r) {
        if (r0 + r >= N) break;
        const float sc = S[r0 + r];
#pragma unroll
        for (int m = 0; m < MW; ++m) {
            float s = tail[r][m];
            s += acc[r][m].a.x;
            s += acc[r][m].a.y;
            s += acc[r][m].a.z;
            s += acc[r][m].a.w;
            s += acc[r][m].b.x;
            s += acc[r][m].b.y;
            s += acc[r][m].b.z;
            s += acc[r][m].b.w;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) s += __shfl_down(s, o);
            if (lane == 0) Y[(size_t)(am0 + m) * N + r0 + r] = float_to_bf16(s * sc);
        }
    }
}

template <int MT, int R, int MW, int U>
void launch_wide(int threads, const int8_t* wp, const float* sp, const uint16_t* xp,
                 uint16_t* yp, int N, int K, hipStream_t stream) {
    constexpr int GM = MT / MW;
    const int wpb = threads / 32;
    const int rows_per_block = (wpb / GM) * R;
    const int blocks = (N + rows_per_block - 1) / rows_per_block;
    gemv_w8a16_wide_kernel<MT, R, MW, U><<<blocks, threads, 0, stream>>>(wp, sp, xp,
                                                                        yp, N, K);
}

template <int MT, int R, int MW>
bool dispatch_ru(int U, int threads, const int8_t* wp, const float* sp,
                 const uint16_t* xp, uint16_t* yp, int N, int K, hipStream_t stream) {
    constexpr int GM = MT / MW;
    const int wpb = threads / 32;
    if (wpb % GM != 0) return false;
    if (U == 2) {
        launch_wide<MT, R, MW, 2>(threads, wp, sp, xp, yp, N, K, stream);
        return true;
    }
    if (U == 4) {
        launch_wide<MT, R, MW, 4>(threads, wp, sp, xp, yp, N, K, stream);
        return true;
    }
    return false;
}

// 每个 M 的 (R, MW) 组合。全部保持 acc = R*MW*8 = 64 个 float，也就是寄存器预算；再加一个
// 组合就会溢出或启动失败。
template <int MT>
bool dispatch(int rows, int mw, int U, int threads, const int8_t* wp, const float* sp,
              const uint16_t* xp, uint16_t* yp, int N, int K, hipStream_t stream) {
    if (mw <= 0 || MT % mw != 0 || mw > MT) return false;   // 必须覆盖全部 M
    if (rows == 1 && mw == 8) return dispatch_ru<MT, 1, 8>(U, threads, wp, sp, xp, yp, N, K, stream);
    if (rows == 2 && mw == 4) return dispatch_ru<MT, 2, 4>(U, threads, wp, sp, xp, yp, N, K, stream);
    if (rows == 4 && mw == 2) return dispatch_ru<MT, 4, 2>(U, threads, wp, sp, xp, yp, N, K, stream);
    return false;
}

// 每个 M 的默认切块：每个 warp 两行配四个激活，能让一个行组里有四个 warp（并行度更高）；
// 每 warp 一行配八个激活是兜底。
inline void default_tiling(int M, int& rows, int& mw) {
    rows = 2;
    mw = 4;
    if (M == 24) { rows = 1; mw = 8; }     // 3 个 warp 一个行组：warp 能均分
}

}  // namespace

// @brief Wide-M int8 W8A16 GEMV：一次权重读取服务 M 取 {16, 24, 32}。
//
// 按 ``default_tiling`` 挑每个 M 的切块；要从探针里扫 rows/mw 就用 ``_dbg`` 入口。
//
// @param W [N, K] int8 行主序。
// @param S [N] fp32 每行 scale，在归约之后统一乘一次。
// @param X [M, K] bf16 连续存放；M 必须是 16、24 或 32，且 K 是 8 的整数倍。
// @param U K 循环展开：每个 lane 预取 2 或 4 个块。
// @param threads 块大小，32 的整数倍且落在 32..256；warp 必须能均匀分进行组。
// @return [M, N] bf16，与对该行 8 激活切片逐个调用 ``gemv_w8a16_m`` 逐位一致。
// @throws c10::Error（TORCH_CHECK）：形状或切块不受支持；调用前先检查 M、K%8 和 warp 数。
torch::Tensor gemv_w8a16_wide(torch::Tensor W, torch::Tensor S, torch::Tensor X,
                              int64_t U, int64_t threads) {
    TORCH_CHECK(W.is_contiguous() && S.is_contiguous() && X.is_contiguous(),
                "contiguous inputs required");
    TORCH_CHECK(W.scalar_type() == at::kChar, "W must be int8");
    const int N = W.size(0), K = W.size(1), M = X.size(0);
    TORCH_CHECK(X.size(1) == K, "X's K does not match W");
    TORCH_CHECK(M == 16 || M == 24 || M == 32, "M must be 16, 24 or 32");
    TORCH_CHECK(K % 8 == 0, "K must be a multiple of 8");
    int rows = 2, mw = 4;
    default_tiling(M, rows, mw);
    return gemv_w8a16_wide_dbg(W, S, X, U, rows, mw, threads);
}

// @brief ``gemv_w8a16_wide`` 的 A/B 入口，把切块参数暴露出来。
//
// @param rows 每个 warp 的输出行数：1、2 或 4。
// @param mw   每个 warp 的激活向量数：必须能整除 MT 并覆盖全部 M。
// @param threads 块大小；<= 0 时取能均分时凑够 8 个 warp 的最小组合数的倍数。
// @return [M, N] bf16，任何切块下的算术都与 ``gemv_w8a16_wide`` 相同。
// @throws c10::Error（TORCH_CHECK）：(M, rows, mw, U, threads) 没有已实例化的内核；消息会
//         点名入口。
torch::Tensor gemv_w8a16_wide_dbg(torch::Tensor W, torch::Tensor S, torch::Tensor X,
                                  int64_t U, int64_t rows, int64_t mw,
                                  int64_t threads) {
    TORCH_CHECK(W.is_contiguous() && S.is_contiguous() && X.is_contiguous(),
                "contiguous inputs required");
    TORCH_CHECK(W.scalar_type() == at::kChar, "W must be int8");
    const int N = W.size(0), K = W.size(1), M = X.size(0);
    TORCH_CHECK(X.size(1) == K, "X's K does not match W");
    TORCH_CHECK(M == 16 || M == 24 || M == 32, "M must be 16, 24 or 32");
    TORCH_CHECK(K % 8 == 0, "K must be a multiple of 8");
    if (threads <= 0) {
        const int gm = (int)(M / mw);
        // 能均分就凑 8 个 warp，否则取组数的最小倍数。
        threads = 8 / gm >= 1 ? (8 / gm) * gm * 32 : gm * 32;
    }
    TORCH_CHECK(threads % 32 == 0 && threads >= 32 && threads <= 256,
                "threads must be a multiple of 32 in 32..256");
    auto Y = torch::empty({M, N}, X.options());
    hipStream_t stream = at::cuda::getCurrentCUDAStream();
    const auto* wp = W.data_ptr<int8_t>();
    const auto* sp = S.data_ptr<float>();
    const auto* xp = reinterpret_cast<const uint16_t*>(X.data_ptr<at::BFloat16>());
    auto* yp = reinterpret_cast<uint16_t*>(Y.data_ptr<at::BFloat16>());
    const int th = (int)threads;
    const int rr = (int)rows, mw_ = (int)mw, uu = (int)U;
    bool ok = false;
    if (M == 16) ok = dispatch<16>(rr, mw_, uu, th, wp, sp, xp, yp, N, K, stream);
    else if (M == 24) ok = dispatch<24>(rr, mw_, uu, th, wp, sp, xp, yp, N, K, stream);
    else ok = dispatch<32>(rr, mw_, uu, th, wp, sp, xp, yp, N, K, stream);
    TORCH_CHECK(ok, "unsupported (M, rows, mw, U, threads) for gemv_w8a16_wide");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return Y;
}

// @brief packed fp16 变体：``v_dot2_f32_f16`` 在一条指令里做两次 fp16 乘，累加器是 fp32。
//
// int8 权重转 fp16 是精确的（[-127,127] 内的值都能表示），激活每次调用只做一次 bf16 -> fp16
// 转换（|x| <= 65504 时精确；超过则钳制，fp16 的非规格化数只在约 6e-8 以下被冲刷掉）。相对
// 标量 int8 路径，这把每个权重元素的算术指令数减半，而 wide-M 内核之前正是输在这里。
//
// 这个变体与 int8 内核*接近*，并不逐位一致：链的打包和合并方式不同。它为校验路径的吞吐而
// 存在，那里调用方接受这点舍入差异。
namespace {

union H2U {
    __half2 h;
    unsigned u;
};

__device__ __forceinline__ float dot2(float acc, unsigned a, unsigned b) {
    float r;
    asm("v_dot2_f32_f16 %0, %1, %2, %3" : "=v"(r) : "v"(a), "v"(b), "v"(acc));
    return r;
}

// 四个 int8（一个 32 位字）-> 两个 packed half2 值（两个 32 位字）。
__device__ __forceinline__ void i8x4_to_h2(unsigned w, unsigned& o0, unsigned& o1) {
    const float f0 = (float)(int)(signed char)(w & 0xffu);
    const float f1 = (float)(int)(signed char)((w >> 8) & 0xffu);
    const float f2 = (float)(int)(signed char)((w >> 16) & 0xffu);
    const float f3 = (float)(int)(signed char)((w >> 24) & 0xffu);
    H2U a, b;
    a.h = __floats2half2_rn(f0, f1);
    b.h = __floats2half2_rn(f2, f3);
    o0 = a.u;
    o1 = b.u;
}

// 对激活做一次 bf16 -> fp16 转换，并钳制到 fp16 的范围，免得越界的激活变成 inf 毒化点积。
__global__ void bf16_to_fp16_kernel(const uint16_t* __restrict__ x,
                                    uint16_t* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float f = bf16_to_float(x[i]);
    f = fminf(fmaxf(f, -65504.f), 65504.f);
    H2U h;
    h.h = __floats2half2_rn(f, f);
    y[i] = (uint16_t)(h.u & 0xffffu);
}

// 每个 warp：R 个输出行、MW 个激活向量，每个（行，激活）对四个 fp32 dot2 累加器。行组/M 组
// 结构与 int8 内核相同；算术是打包的，但逐行的和仍是 fp32。
//
// @param Xh [M, K] fp16 激活（由 ``bf16_to_fp16_kernel`` 转换）。
// @param K  必须是 16 的整数倍：循环每个 lane 步消耗 16 个 int8。
template <int MT, int R, int MW, int U>
__global__ void gemv_w8a16_dot2_kernel(const int8_t* __restrict__ W,
                                       const float* __restrict__ S,
                                       const uint16_t* __restrict__ Xh,
                                       uint16_t* __restrict__ Y, int N, int K) {
    constexpr int GM = MT / MW;
    constexpr int CH = 4;                    // 每个（行，激活）的 fp32 链数
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int wpb = blockDim.x >> 5;
    const int mg = wid % GM;
    const int rg = wid / GM;
    const int rowgroup = blockIdx.x * (wpb / GM) + rg;
    const int r0 = rowgroup * R;
    if (r0 >= N) return;

    const int am0 = mg * MW;
    const int nvec = K >> 4;                 // 每 lane 每步 16 个 int8（uint4）
    const int nvec8 = K >> 3;                // fp16 激活的 uint4 单位数
    const uint4* wv = reinterpret_cast<const uint4*>(W + (size_t)r0 * K);
    const uint4* xv = reinterpret_cast<const uint4*>(Xh);

    float acc[R][MW][CH];
    float tail[R][MW];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int m = 0; m < MW; ++m) {
#pragma unroll
            for (int j = 0; j < CH; ++j) acc[r][m][j] = 0.f;
            tail[r][m] = 0.f;
        }

    const int step = 32 * U;
    int base = 0;
    for (; base + step <= nvec; base += step) {
        uint4 wc[R][U];
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int u = 0; u < U; ++u)
                wc[r][u] = wv[(size_t)r * nvec + base + u * 32 + lane];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const size_t i = (size_t)(base + u * 32 + lane);
            const size_t xi = i * 2;         // 每 16 个权重对应两个 uint4（16 个 fp16）
#pragma unroll
            for (int r = 0; r < R; ++r) {
                unsigned h[8];
                i8x4_to_h2(wc[r][u].x, h[0], h[1]);
                i8x4_to_h2(wc[r][u].y, h[2], h[3]);
                i8x4_to_h2(wc[r][u].z, h[4], h[5]);
                i8x4_to_h2(wc[r][u].w, h[6], h[7]);
#pragma unroll
                for (int m = 0; m < MW; ++m) {
                    const uint4 xa = xv[(size_t)(am0 + m) * nvec8 + xi];
                    const uint4 xb = xv[(size_t)(am0 + m) * nvec8 + xi + 1];
                    acc[r][m][0] = dot2(acc[r][m][0], h[0], xa.x);
                    acc[r][m][1] = dot2(acc[r][m][1], h[1], xa.y);
                    acc[r][m][2] = dot2(acc[r][m][2], h[2], xa.z);
                    acc[r][m][3] = dot2(acc[r][m][3], h[3], xa.w);
                    acc[r][m][0] = dot2(acc[r][m][0], h[4], xb.x);
                    acc[r][m][1] = dot2(acc[r][m][1], h[5], xb.y);
                    acc[r][m][2] = dot2(acc[r][m][2], h[6], xb.z);
                    acc[r][m][3] = dot2(acc[r][m][3], h[7], xb.w);
                }
            }
        }
    }
    for (int i = base + lane; i < nvec; i += 32) {
        const size_t xi = (size_t)i * 2;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint4 wc = wv[(size_t)r * nvec + i];
            unsigned h[8];
            i8x4_to_h2(wc.x, h[0], h[1]);
            i8x4_to_h2(wc.y, h[2], h[3]);
            i8x4_to_h2(wc.z, h[4], h[5]);
            i8x4_to_h2(wc.w, h[6], h[7]);
#pragma unroll
            for (int m = 0; m < MW; ++m) {
                const uint4 xa = xv[(size_t)(am0 + m) * nvec8 + xi];
                const uint4 xb = xv[(size_t)(am0 + m) * nvec8 + xi + 1];
                acc[r][m][0] = dot2(acc[r][m][0], h[0], xa.x);
                acc[r][m][1] = dot2(acc[r][m][1], h[1], xa.y);
                acc[r][m][2] = dot2(acc[r][m][2], h[2], xa.z);
                acc[r][m][3] = dot2(acc[r][m][3], h[3], xa.w);
                acc[r][m][0] = dot2(acc[r][m][0], h[4], xb.x);
                acc[r][m][1] = dot2(acc[r][m][1], h[5], xb.y);
                acc[r][m][2] = dot2(acc[r][m][2], h[6], xb.z);
                acc[r][m][3] = dot2(acc[r][m][3], h[7], xb.w);
            }
        }
    }
    // K 不是 16 的整数倍时走标量 fp32 尾循环（精确的 int8 -> float 路径）。
    for (int k = (nvec << 4) + lane; k < K; k += 32) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (r0 + r >= N) break;
            const float wf = (float)W[(size_t)(r0 + r) * K + k];
#pragma unroll
            for (int m = 0; m < MW; ++m) {
                const __half xh = *reinterpret_cast<const __half*>(
                    &Xh[(size_t)(am0 + m) * K + k]);
                tail[r][m] = fmaf(wf, __half2float(xh), tail[r][m]);
            }
        }
    }

#pragma unroll
    for (int r = 0; r < R; ++r) {
        if (r0 + r >= N) break;
        const float sc = S[r0 + r];
#pragma unroll
        for (int m = 0; m < MW; ++m) {
            float s = (acc[r][m][0] + acc[r][m][1]) + (acc[r][m][2] + acc[r][m][3])
                    + tail[r][m];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) s += __shfl_down(s, o);
            if (lane == 0) Y[(size_t)(am0 + m) * N + r0 + r] = float_to_bf16(s * sc);
        }
    }
}

template <int MT, int R, int MW, int U>
void launch_dot2(int threads, const int8_t* wp, const float* sp, const uint16_t* xp,
                 uint16_t* yp, int N, int K, hipStream_t stream) {
    constexpr int GM = MT / MW;
    const int wpb = threads / 32;
    const int rows_per_block = (wpb / GM) * R;
    const int blocks = (N + rows_per_block - 1) / rows_per_block;
    gemv_w8a16_dot2_kernel<MT, R, MW, U><<<blocks, threads, 0, stream>>>(wp, sp, xp,
                                                                         yp, N, K);
}

template <int MT, int R, int MW>
bool dispatch_dot2_ru(int U, int threads, const int8_t* wp, const float* sp,
                      const uint16_t* xp, uint16_t* yp, int N, int K,
                      hipStream_t stream) {
    constexpr int GM = MT / MW;
    if ((threads / 32) % GM != 0) return false;
    if (U == 2) {
        launch_dot2<MT, R, MW, 2>(threads, wp, sp, xp, yp, N, K, stream);
        return true;
    }
    if (U == 4) {
        launch_dot2<MT, R, MW, 4>(threads, wp, sp, xp, yp, N, K, stream);
        return true;
    }
    return false;
}

template <int MT>
bool dispatch_dot2(int rows, int mw, int U, int threads, const int8_t* wp,
                   const float* sp, const uint16_t* xp, uint16_t* yp, int N, int K,
                   hipStream_t stream) {
    if (mw <= 0 || MT % mw != 0 || mw > MT) return false;   // 必须覆盖全部 M
    // 每个 M 的切块集合；每个组合都保持 R*MW*4 <= 64 个累加器 float。
    if (MT == 1) {
        if (rows == 1 && mw == 1) return dispatch_dot2_ru<1, 1, 1>(U, threads, wp, sp, xp, yp, N, K, stream);
        if (rows == 2 && mw == 1) return dispatch_dot2_ru<1, 2, 1>(U, threads, wp, sp, xp, yp, N, K, stream);
        if (rows == 4 && mw == 1) return dispatch_dot2_ru<1, 4, 1>(U, threads, wp, sp, xp, yp, N, K, stream);
    } else if (MT == 2) {
        if (rows == 1 && mw == 2) return dispatch_dot2_ru<2, 1, 2>(U, threads, wp, sp, xp, yp, N, K, stream);
        if (rows == 2 && mw == 2) return dispatch_dot2_ru<2, 2, 2>(U, threads, wp, sp, xp, yp, N, K, stream);
        if (rows == 4 && mw == 2) return dispatch_dot2_ru<2, 4, 2>(U, threads, wp, sp, xp, yp, N, K, stream);
    } else if (MT == 4) {
        if (rows == 1 && mw == 4) return dispatch_dot2_ru<4, 1, 4>(U, threads, wp, sp, xp, yp, N, K, stream);
        if (rows == 2 && mw == 4) return dispatch_dot2_ru<4, 2, 4>(U, threads, wp, sp, xp, yp, N, K, stream);
        if (rows == 2 && mw == 2) return dispatch_dot2_ru<4, 2, 2>(U, threads, wp, sp, xp, yp, N, K, stream);
        if (rows == 1 && mw == 2) return dispatch_dot2_ru<4, 1, 2>(U, threads, wp, sp, xp, yp, N, K, stream);
        if (rows == 4 && mw == 2) return dispatch_dot2_ru<4, 4, 2>(U, threads, wp, sp, xp, yp, N, K, stream);
    } else if (MT == 8) {
        if (rows == 1 && mw == 8) return dispatch_dot2_ru<8, 1, 8>(U, threads, wp, sp, xp, yp, N, K, stream);
        if (rows == 2 && mw == 8) return dispatch_dot2_ru<8, 2, 8>(U, threads, wp, sp, xp, yp, N, K, stream);
        if (rows == 2 && mw == 4) return dispatch_dot2_ru<8, 2, 4>(U, threads, wp, sp, xp, yp, N, K, stream);
        if (rows == 4 && mw == 4) return dispatch_dot2_ru<8, 4, 4>(U, threads, wp, sp, xp, yp, N, K, stream);
        if (rows == 4 && mw == 2) return dispatch_dot2_ru<8, 4, 2>(U, threads, wp, sp, xp, yp, N, K, stream);
    } else if (MT == 16) {
        if (rows == 1 && mw == 16) return dispatch_dot2_ru<16, 1, 16>(U, threads, wp, sp, xp, yp, N, K, stream);
        if (rows == 2 && mw == 8) return dispatch_dot2_ru<16, 2, 8>(U, threads, wp, sp, xp, yp, N, K, stream);
        if (rows == 4 && mw == 4) return dispatch_dot2_ru<16, 4, 4>(U, threads, wp, sp, xp, yp, N, K, stream);
        if (rows == 1 && mw == 8) return dispatch_dot2_ru<16, 1, 8>(U, threads, wp, sp, xp, yp, N, K, stream);
    } else if (MT == 24) {
        if (rows == 2 && mw == 8) return dispatch_dot2_ru<24, 2, 8>(U, threads, wp, sp, xp, yp, N, K, stream);
        if (rows == 4 && mw == 4) return dispatch_dot2_ru<24, 4, 4>(U, threads, wp, sp, xp, yp, N, K, stream);
        if (rows == 1 && mw == 8) return dispatch_dot2_ru<24, 1, 8>(U, threads, wp, sp, xp, yp, N, K, stream);
    } else {
        if (rows == 2 && mw == 8) return dispatch_dot2_ru<32, 2, 8>(U, threads, wp, sp, xp, yp, N, K, stream);
        if (rows == 4 && mw == 4) return dispatch_dot2_ru<32, 4, 4>(U, threads, wp, sp, xp, yp, N, K, stream);
        if (rows == 1 && mw == 16) return dispatch_dot2_ru<32, 1, 16>(U, threads, wp, sp, xp, yp, N, K, stream);
    }
    return false;
}

inline void default_dot2_tiling(int M, int& rows, int& mw) {
    // 这些是探针对每个 M 实测选出最快的切块点（每 warp 的行数、每 warp 的激活向量数；
    // M>=4 时 acc = rows*mw*4 = 64 个 float）。每个组合保持逐行 fp32 求和的顺序不变，所以
    // 这个选择只影响速度。
    if (M == 1) { rows = 1; mw = 1; }
    else if (M == 2) { rows = 2; mw = 2; }
    else if (M == 4) { rows = 4; mw = 2; }
    else { rows = 4; mw = 4; }      // M = 8, 16, 24, 32
}

}  // namespace

// @brief 用 packed fp16 数学的 Wide-M GEMV，切块按 M 挑选。
//
// @param X [M, K] bf16 或 fp16；bf16 输入每次调用转换一次，M 可以是 1..32 里任意值（不只是
//          16/24/32），K 必须是 16 的整数倍。
// @return [M, N] bf16。接近 int8 内核的结果而不逐位一致：逐行仍是 fp32 求和，只是舍入不同。
// @throws c10::Error（TORCH_CHECK）：dtype、M 或 K%16 不受支持；去改调用方的形状而不是改
//         内核。
torch::Tensor gemv_w8a16_wide_dot2(torch::Tensor W, torch::Tensor S, torch::Tensor X,
                                   int64_t U, int64_t threads) {
    int rows = 1, mw = 16;
    const int M = X.size(0);
    default_dot2_tiling(M, rows, mw);
    return gemv_w8a16_wide_dot2_dbg(W, S, X, U, rows, mw, threads);
}

torch::Tensor gemv_w8a16_wide_dot2_dbg(torch::Tensor W, torch::Tensor S,
                                       torch::Tensor X, int64_t U, int64_t rows,
                                       int64_t mw, int64_t threads) {
    // A/B 入口：任何切块下的算术都相同，所以这轮扫描只是在量速度。
    TORCH_CHECK(W.is_contiguous() && S.is_contiguous() && X.is_contiguous(),
                "contiguous inputs required");
    TORCH_CHECK(W.scalar_type() == at::kChar, "W must be int8");
    TORCH_CHECK(X.scalar_type() == at::kHalf || X.scalar_type() == at::kBFloat16,
                "X must be fp16 or bf16");
    const int N = W.size(0), K = W.size(1), M = X.size(0);
    TORCH_CHECK(X.size(1) == K, "X's K does not match W");
    TORCH_CHECK(M >= 1 && M <= 32, "M must be in 1..32");
    TORCH_CHECK(K % 16 == 0, "K must be a multiple of 16");
    hipStream_t stream = at::cuda::getCurrentCUDAStream();

    // 激活转成 fp16：bf16 -> fp16 精确转换，每次调用只做一次。调用方本来就传 fp16 时跳过，
    // 于是下面的算术两种情况下完全一样。
    torch::Tensor Xh;
    if (X.scalar_type() == at::kHalf) {
        Xh = X;
    } else {
        Xh = torch::empty(X.sizes(), X.options().dtype(at::kHalf));
        const int n = M * K;
        bf16_to_fp16_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
            reinterpret_cast<const uint16_t*>(X.data_ptr<at::BFloat16>()),
            reinterpret_cast<uint16_t*>(Xh.data_ptr<at::Half>()), n);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    if (threads <= 0) {
        const int gm = (int)(M / mw);
        threads = (gm >= 8) ? gm * 32 : (8 / gm) * gm * 32;
    }
    TORCH_CHECK(threads % 32 == 0 && threads >= 32 && threads <= 256,
                "threads must be a multiple of 32 in 32..256");
    auto Y = torch::empty({M, N}, X.options().dtype(at::kBFloat16));
    const auto* wp = W.data_ptr<int8_t>();
    const auto* sp = S.data_ptr<float>();
    const auto* xp = reinterpret_cast<const uint16_t*>(Xh.data_ptr<at::Half>());
    auto* yp = reinterpret_cast<uint16_t*>(Y.data_ptr<at::BFloat16>());
    const int th = (int)threads, rr = (int)rows, mw_ = (int)mw, uu = (int)U;
    bool ok = false;
    if (M == 1) ok = dispatch_dot2<1>(rr, mw_, uu, th, wp, sp, xp, yp, N, K, stream);
    else if (M == 2) ok = dispatch_dot2<2>(rr, mw_, uu, th, wp, sp, xp, yp, N, K, stream);
    else if (M == 4) ok = dispatch_dot2<4>(rr, mw_, uu, th, wp, sp, xp, yp, N, K, stream);
    else if (M == 8) ok = dispatch_dot2<8>(rr, mw_, uu, th, wp, sp, xp, yp, N, K, stream);
    else if (M == 16) ok = dispatch_dot2<16>(rr, mw_, uu, th, wp, sp, xp, yp, N, K, stream);
    else if (M == 24) ok = dispatch_dot2<24>(rr, mw_, uu, th, wp, sp, xp, yp, N, K, stream);
    else ok = dispatch_dot2<32>(rr, mw_, uu, th, wp, sp, xp, yp, N, K, stream);
    TORCH_CHECK(ok, "unsupported (M, rows, mw, U, threads) for gemv_w8a16_wide_dot2");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return Y;
}
