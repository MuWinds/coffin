// @file w8a8.cu
// @brief W8A8 预填充投影：按行量化激活、int8 GEMM、融合 epilogue。
//
// Module: coffin/backends/corex/kernels。绑定成 ``quant_act_i8`` / ``w8a8_gemm`` /
// ``w8a8_epilogue``，由 ``backends/corex/__init__.py`` 的 ``linear_prefill`` 调用。
// 本文件只碰预填充：解码仍然是 W8A16 GEMV，因为在 M=1 上 int8 激活的固定开销一点
// 没少而字节一个没省，实测慢 42%（27B-4CARD-PLAN.md §3.4）。
//
// 存在理由，一句话：预填充每一步的 GEMM 要反复消费同一份权重，所以权重以 int8
// 常驻、激活每 token 量化一次，就把"每次调用反量化整份权重"的固定成本
// （实测 +173 ms/请求）换成三个内核。同一形状上 10.875 ms → 8.535 ms（1.27x），
// 同时权重显存减半。
//
// Constraints:
//   * 库只在 ``N <= 131072`` 上调用：本卡 cuBLAS 在 ``N = 248320``（``lm_head``）
//     返回 NaN，而那个形状正是最经不起这种错误的一个。切块的判定在 Python 侧。
//   * 累加器是 int32，不是 fp32。int8×int8 的乘积精确落在 int32 里，而这条路径的
//     精度预算（单层 rel_err 9.07e-3）已经全部花在量化上，不能再往里加累加误差。
//   * 两个 scale 在同一个 fp32 表达式里乘完再窄化一次。分两次窄化会多一次舍入，
//     而结果要穿过 32 层。
//   * 每次启动都在 ``c10::cuda::getCurrentCUDAStream()`` 上。legacy 默认流上启动的
//     内核不属于捕获范围。
//   * 全程不用 fp64：本设备对它返回 0（COREX_PORT.md §0）。

#include "corex_i8.h"
#include "corex_ops.h"
#include "corex_scalar.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>

namespace {

// 工作区在这里定义：三个算子共用同一批阈值，才能保证分块边界一致。

//: 激活量化时每块线程数。一行一个块，归约只在 K 上做，所以这个数只影响单行的带宽。
constexpr int QUANT_THREADS = 256;


//: epilogue 每块线程数。
constexpr int EPI_THREADS = 256;

//: 库 GEMM 仍然算得对的输出列上限，与 ``pack.LIB_MAX_N`` 是同一个数。
//
// 重复一份而不是从 Python 传进来，是因为这个值是卡的性质而不是调优参数：内核自己
// 拒绝它索引不了的形状，Python 侧的那一份负责决定怎么切。两边都留，切块逻辑写错
// 时才会抛异常而不是静默 NaN。
constexpr int LIB_MAX_N = 131072;

// @brief 按行（per-token）对称 int8 量化，一块一行，一次启动。
//
// 一行 = 一个 token，所以 amax 只在 K 上归约，不需要跨块通信。``1e-5`` 下限让全零
// 行量化成零而不是 inf。
//
// @param x [M, K] 激活。
// @param out_q [M, K] int8 目的地。
// @param out_s [M] 每行 fp32 scale 目的地。
// @param K 行宽。
template <typename XT>
__global__ void quant_act_i8_kernel(const XT* __restrict__ x, int8_t* __restrict__ out_q,
                                    float* __restrict__ out_s, int K) {
    const int row = blockIdx.x;
    const XT* xr = x + (size_t)row * K;
    int8_t* qr = out_q + (size_t)row * K;

    // 静态分配 256 个 float，对上 QUANT_THREADS；归约用这一块。
    __shared__ float smax[QUANT_THREADS];

    float amax = 0.f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        amax = fmaxf(amax, fabsf(CxScalar<XT>::to_float(xr[i])));
    }
    smax[threadIdx.x] = amax;
    __syncthreads();
    // 关闭屏障不可省：下一次归约（这里没有）或同一块里再进这个函数时，
    // ``smax[0] = v`` 的写必须晚于别人对它的读。
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smax[threadIdx.x] = fmaxf(smax[threadIdx.x], smax[threadIdx.x + s]);
        }
        __syncthreads();
    }
    const float sc = fmaxf(smax[0], 1e-5f) / 127.0f;
    const float inv = 1.0f / sc;
    if (threadIdx.x == 0) out_s[row] = sc;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        int q = __float2int_rn(CxScalar<XT>::to_float(xr[i]) * inv);
        q = max(-127, min(127, q));
        qr[i] = (int8_t)q;
    }
}

// @brief 施加两个 W8A8 scale 并窄化一次。
//
// 两个 scale 在同一个 fp32 表达式里乘完再窄化，与稠密路径的舍入次数一致。
//
// @param c [M, N] int32 GEMM 输出。
// @param xs [M] 每 token 激活 scale。
// @param ws [N] 每输出列权重 scale。
// @param out [M, N] 目的地。
// @param N 行宽。
// @param stride ``out`` 的行间距。
template <typename XT>
__global__ void w8a8_epilogue_kernel(const int32_t* __restrict__ c,
                                     const float* __restrict__ xs,
                                     const float* __restrict__ ws,
                                     XT* __restrict__ out, int N, int stride) {
    const int m = blockIdx.y;
    const float xm = xs[m];
    const int32_t* cr = c + (size_t)m * N;
    XT* orow = out + (size_t)m * stride;
    for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < N;
         col += blockDim.x * gridDim.x) {
        orow[col] = CxScalar<XT>::from_float((float)cr[col] * xm * ws[col]);
    }
}

}  // namespace

// 主机侧包装层。

torch::Tensor quant_act_i8(torch::Tensor x, torch::Tensor out_q, torch::Tensor out_s) {
    TORCH_CHECK(x.dim() == 2, "activation must be 2-D [M, K], got ", x.dim());
    TORCH_CHECK(out_q.is_contiguous() && out_s.is_contiguous(),
                "quant destinations must be contiguous");
    TORCH_CHECK(out_q.scalar_type() == torch::kChar, "out_q must be int8, got ",
                out_q.scalar_type());
    TORCH_CHECK(out_s.scalar_type() == torch::kFloat, "out_s must be fp32, got ",
                out_s.scalar_type());
    TORCH_CHECK(out_q.sizes() == x.sizes(), "out_q shape ", out_q.sizes(),
                " does not match x ", x.sizes());
    TORCH_CHECK(out_s.numel() == x.size(0), "out_s must hold one value per row: got ",
                out_s.numel(), " for M = ", x.size(0));

    const int M = (int)x.size(0);
    const int K = (int)x.size(1);
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const bool x32 = x.scalar_type() == torch::kFloat;
    TORCH_CHECK(x32 || x.scalar_type() == torch::kBFloat16,
                "activation must be bf16 or fp32, got ", x.scalar_type());
    auto xc = x.contiguous();
    if (x32) {
        quant_act_i8_kernel<float><<<M, QUANT_THREADS, 0, stream>>>(
            reinterpret_cast<const float*>(xc.data_ptr()), out_q.data_ptr<int8_t>(),
            out_s.data_ptr<float>(), K);
    } else {
        quant_act_i8_kernel<uint16_t><<<M, QUANT_THREADS, 0, stream>>>(
            reinterpret_cast<const uint16_t*>(xc.data_ptr()), out_q.data_ptr<int8_t>(),
            out_s.data_ptr<float>(), K);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out_q;
}

torch::Tensor w8a8_gemm(torch::Tensor a, torch::Tensor w, torch::Tensor c) {
    TORCH_CHECK(a.dim() == 2 && w.dim() == 2 && c.dim() == 2,
                "A, W and C must all be 2-D");
    TORCH_CHECK(a.scalar_type() == torch::kChar && w.scalar_type() == torch::kChar,
                "operands must be int8");
    TORCH_CHECK(c.scalar_type() == torch::kInt, "accumulator must be int32");
    TORCH_CHECK(a.is_contiguous() && w.is_contiguous() && c.is_contiguous(),
                "contiguous operands required");
    TORCH_CHECK(a.size(1) == w.size(0), "K mismatch: A ", a.sizes(), " W ", w.sizes());
    TORCH_CHECK(c.size(0) == a.size(0) && c.size(1) == w.size(1),
                "C ", c.sizes(), " does not match ", a.size(0), "x", w.size(1));
    // 库只在这张卡已验证的宽度内调用。Python 侧按 ``pack.LIB_MAX_N`` 切块，这里是
    // 第二道：切块逻辑写错时宁可抛异常，也不要让 ``lm_head`` 那条 248320 列的路
    // 悄悄走到已知返回 NaN 的形状上。
    TORCH_CHECK(w.size(1) <= LIB_MAX_N,
                "N = ", w.size(1), " exceeds the library's verified width; "
                "chunk along N in the caller");
    TORCH_CHECK(a.is_cuda() && w.is_cuda() && c.is_cuda(),
                "the int8 GEMM runs on the library, which needs all three operands "
                "on the device");

    const int M = (int)a.size(0);
    const int K = (int)a.size(1);
    const int N = (int)w.size(1);
    if (M == 0 || N == 0) return c;

    // 用 torch 的句柄而不是自建一个：句柄和当前设备、当前 stream 的关系已经由 ATen
    // 维护好，自建一个就得自己回答这两个问题，而且每进程多一份 cuBLAS 工作区。
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    TORCH_CHECK(cublasSetStream(handle, stream) == CUBLAS_STATUS_SUCCESS,
                "cublasSetStream failed on the current stream");

    const int32_t alpha = 1;
    const int32_t beta = 0;
    // 行主序的 ``C = A @ W`` 就是列主序的 ``C^T = W^T @ A^T``，所以参数顺序是
    // ``(N, M, K)`` 且权重在前。``W`` 的 ``[K, N]`` 行主序按列主序读正好是它的
    // 转置 ``[N, K]``，两边都不需要真的转置一次——这两个布局在内存里是同一份字节。
    const cublasStatus_t st = cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
        w.data_ptr<int8_t>(), CUDA_R_8I, N,
        a.data_ptr<int8_t>(), CUDA_R_8I, K,
        &beta, c.data_ptr<int32_t>(), CUDA_R_32I, N,
        CUDA_R_32I, CUBLAS_GEMM_DEFAULT);
    TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS, "cublasGemmEx(int8) failed: status ", (int)st);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return c;
}

torch::Tensor w8a8_epilogue(torch::Tensor c, torch::Tensor xs, torch::Tensor ws,
                            torch::Tensor out, int64_t stride) {
    TORCH_CHECK(c.dim() == 2, "accumulator must be 2-D [M, N], got ", c.dim());
    TORCH_CHECK(c.scalar_type() == torch::kInt, "accumulator must be int32, got ",
                c.scalar_type());
    TORCH_CHECK(c.is_contiguous(), "accumulator must be contiguous");
    TORCH_CHECK(xs.scalar_type() == torch::kFloat && xs.numel() == c.size(0),
                "xs must be fp32 with one value per row");
    TORCH_CHECK(ws.scalar_type() == torch::kFloat && ws.numel() == c.size(1),
                "ws must be fp32 with one value per column");
    const int M = (int)c.size(0);
    const int N = (int)c.size(1);
    const int st = stride > 0 ? (int)stride : N;
    TORCH_CHECK(out.size(0) == M && out.size(1) == N && st >= N,
                "out must be [M, N] with stride >= N");

    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const int blocks = (N + EPI_THREADS - 1) / EPI_THREADS;
    const dim3 grid(blocks, M);
    const bool o32 = out.scalar_type() == torch::kFloat;
    TORCH_CHECK(o32 || out.scalar_type() == torch::kBFloat16,
                "out must be bf16 or fp32, got ", out.scalar_type());
    if (o32) {
        w8a8_epilogue_kernel<float><<<grid, EPI_THREADS, 0, stream>>>(
            c.data_ptr<int32_t>(), xs.data_ptr<float>(), ws.data_ptr<float>(),
            reinterpret_cast<float*>(out.data_ptr()), N, st);
    } else {
        w8a8_epilogue_kernel<uint16_t><<<grid, EPI_THREADS, 0, stream>>>(
            c.data_ptr<int32_t>(), xs.data_ptr<float>(), ws.data_ptr<float>(),
            reinterpret_cast<uint16_t*>(out.data_ptr()), N, st);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
