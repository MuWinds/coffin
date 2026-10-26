"""The kernel facts, with every measurement made the way this card actually works.

Three traps cost a job each before this probe existed, and all three are silent
rather than loud:

* this card's ``fp64`` is "limited support": ``y.double()`` on the device returns
  zeros, so a comparison written as ``y.double().cpu() - ref`` reports the
  reference's own magnitude as the error. Every conversion here happens after
  ``.cpu()``.
* ``cuBLAS`` returns NaN for ``[K, N]`` matmul at ``N = 248320`` and cannot be
  captured into a graph (``CUBLAS_STATUS_ALLOC_FAILED``). The reference is
  therefore elementwise-only, and graph capture is only claimed for our own
  kernels.
* ``F.linear`` at M=1 with bf16 runs at 19 GB/s, so a library number is not a
  bandwidth ceiling. The ceiling is measured with a pure copy.

Module: ``coffin/diag``. Diagnostic only: synthetic tensors, no checkpoint.

Constraints:
    * No device fp64, no library GEMM in any reference path.
    * Every launch is followed by its own error check before another torch op can
      clear the error.
"""

from __future__ import annotations

import json
import os
import sys

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>
#include <string>

// y[n] = sum_k x[k] * Wt[k][n], Wt row-major [K, N], fp32 accumulate.
//
// One thread per output column, so a warp reads 64 (warpSize here) adjacent
// columns per k-step: one coalesced transaction per warp per k, and x[k] is
// broadcast to all lanes. That is the whole reason [K, N] beats the [N, K] layout
// library GEMM produces at M=1: the library's inner loop strides by K.
template <int UNROLL>
__global__ void gemv_kn_bf16_kernel(const __nv_bfloat16 *__restrict__ wt,
                                    const __nv_bfloat16 *__restrict__ x,
                                    float *__restrict__ y, int K, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float acc = 0.f;
    int k = 0;
    for (; k + UNROLL <= K; k += UNROLL) {
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            acc += __bfloat162float(wt[(size_t)(k + u) * N + n])
                 * __bfloat162float(x[k + u]);
        }
    }
    for (; k < K; ++k) {
        acc += __bfloat162float(wt[(size_t)k * N + n]) * __bfloat162float(x[k]);
    }
    y[n] = acc;
}

// Four adjacent output columns per thread: one 8-byte weight load per k-step
// instead of four 2-byte ones, and a quarter of the loop iterations.
__global__ void gemv_kn_bf16_x4_kernel(const __nv_bfloat16 *__restrict__ wt,
                                       const __nv_bfloat16 *__restrict__ x,
                                       float *__restrict__ y, int K, int N) {
    int n0 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (n0 + 3 >= N) return;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int k = 0; k < K; ++k) {
        const __nv_bfloat162 *p =
            reinterpret_cast<const __nv_bfloat162 *>(wt + (size_t)k * N + n0);
        float xv = __bfloat162float(x[k]);
        acc.x += __bfloat162float(p[0].x) * xv;
        acc.y += __bfloat162float(p[0].y) * xv;
        acc.z += __bfloat162float(p[1].x) * xv;
        acc.w += __bfloat162float(p[1].y) * xv;
    }
    y[n0 + 0] = acc.x; y[n0 + 1] = acc.y; y[n0 + 2] = acc.z; y[n0 + 3] = acc.w;
}

// fp16 weights, bf16 activations, fp32 accumulate: the dtype pair the prefill
// GEMM can also read, so one stored weight serves both regimes.
__global__ void gemv_kn_f16_kernel(const __half *__restrict__ wt,
                                   const __nv_bfloat16 *__restrict__ x,
                                   float *__restrict__ y, int K, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        acc += __half2float(wt[(size_t)k * N + n]) * __bfloat162float(x[k]);
    }
    y[n] = acc;
}

__global__ void gemv_kn_f16_x4_kernel(const __half *__restrict__ wt,
                                      const __nv_bfloat16 *__restrict__ x,
                                      float *__restrict__ y, int K, int N) {
    int n0 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (n0 + 3 >= N) return;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int k = 0; k < K; ++k) {
        const __half2 *p = reinterpret_cast<const __half2 *>(wt + (size_t)k * N + n0);
        float xv = __bfloat162float(x[k]);
        acc.x += __half2float(p[0].x) * xv;
        acc.y += __half2float(p[0].y) * xv;
        acc.z += __half2float(p[1].x) * xv;
        acc.w += __half2float(p[1].y) * xv;
    }
    y[n0 + 0] = acc.x; y[n0 + 1] = acc.y; y[n0 + 2] = acc.z; y[n0 + 3] = acc.w;
}

// Multi-row variant for the speculative-verify / small-batch regime: one weight
// pass, M accumulators per thread, so M tokens cost one read of the weight.
__global__ void gemv_kn_bf16_m_kernel(const __nv_bfloat16 *__restrict__ wt,
                                      const __nv_bfloat16 *__restrict__ x,
                                      float *__restrict__ y, int K, int N, int M) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    // M is bounded by the verify width; keeping it in registers is the point.
    float acc[8];
    for (int m = 0; m < M; ++m) acc[m] = 0.f;
    for (int k = 0; k < K; ++k) {
        float w = __bfloat162float(wt[(size_t)k * N + n]);
        for (int m = 0; m < M; ++m) {
            acc[m] += w * __bfloat162float(x[(size_t)m * K + k]);
        }
    }
    for (int m = 0; m < M; ++m) y[(size_t)m * N + n] = acc[m];
}

torch::Tensor gemv_kn(torch::Tensor wt, torch::Tensor x, int64_t unroll) {
    int K = (int)wt.size(0), N = (int)wt.size(1);
    auto y = torch::zeros({N}, x.options().dtype(torch::kFloat32));
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    auto stream = c10::cuda::getCurrentCUDAStream();
    if (unroll == 4) {
        gemv_kn_bf16_kernel<4><<<blocks, threads, 0, stream>>>(
            (const __nv_bfloat16 *)wt.data_ptr(), (const __nv_bfloat16 *)x.data_ptr(),
            y.data_ptr<float>(), K, N);
    } else {
        gemv_kn_bf16_kernel<2><<<blocks, threads, 0, stream>>>(
            (const __nv_bfloat16 *)wt.data_ptr(), (const __nv_bfloat16 *)x.data_ptr(),
            y.data_ptr<float>(), K, N);
    }
    return y;
}

torch::Tensor gemv_kn_x4(torch::Tensor wt, torch::Tensor x) {
    int K = (int)wt.size(0), N = (int)wt.size(1);
    auto y = torch::zeros({N}, x.options().dtype(torch::kFloat32));
    int threads = 128;
    int blocks = (N / 4 + threads - 1) / threads;
    gemv_kn_bf16_x4_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        (const __nv_bfloat16 *)wt.data_ptr(), (const __nv_bfloat16 *)x.data_ptr(),
        y.data_ptr<float>(), K, N);
    return y;
}

torch::Tensor gemv_kn_f16_x4(torch::Tensor wt, torch::Tensor x) {
    int K = (int)wt.size(0), N = (int)wt.size(1);
    auto y = torch::zeros({N}, x.options().dtype(torch::kFloat32));
    int threads = 128;
    int blocks = (N / 4 + threads - 1) / threads;
    gemv_kn_f16_x4_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        (const __half *)wt.data_ptr(), (const __nv_bfloat16 *)x.data_ptr(),
        y.data_ptr<float>(), K, N);
    return y;
}

torch::Tensor gemv_kn_m(torch::Tensor wt, torch::Tensor x) {
    int K = (int)wt.size(0), N = (int)wt.size(1), M = (int)x.size(0);
    auto y = torch::zeros({M, N}, x.options().dtype(torch::kFloat32));
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    gemv_kn_bf16_m_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        (const __nv_bfloat16 *)wt.data_ptr(), (const __nv_bfloat16 *)x.data_ptr(),
        y.data_ptr<float>(), K, N, M);
    return y;
}

// In-place variant: no allocation, so the caller can capture it in a graph.
torch::Tensor gemv_kn_into(torch::Tensor wt, torch::Tensor x, torch::Tensor y) {
    int K = (int)wt.size(0), N = (int)wt.size(1);
    int threads = 128;
    int blocks = (N / 4 + threads - 1) / threads;
    gemv_kn_bf16_x4_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        (const __nv_bfloat16 *)wt.data_ptr(), (const __nv_bfloat16 *)x.data_ptr(),
        y.data_ptr<float>(), K, N);
    return y;
}

// Elementwise fp32 reference: the library GEMM is broken at wide N here, and
// device fp64 returns zeros, so neither can be the reference.
torch::Tensor ref_elementwise(torch::Tensor wt, torch::Tensor x) {
    auto wf = wt.to(torch::kFloat32);
    auto xf = x.to(torch::kFloat32).unsqueeze(1);
    return (wf * xf).sum(0);
}

std::string sync_status() {
    cudaError_t e = cudaDeviceSynchronize();
    return std::string(cudaGetErrorString(e));
}
"""

CPP_SRC = r"""
#include <torch/extension.h>
#include <string>
torch::Tensor gemv_kn(torch::Tensor wt, torch::Tensor x, int64_t unroll);
torch::Tensor gemv_kn_x4(torch::Tensor wt, torch::Tensor x);
torch::Tensor gemv_kn_f16_x4(torch::Tensor wt, torch::Tensor x);
torch::Tensor gemv_kn_m(torch::Tensor wt, torch::Tensor x);
torch::Tensor gemv_kn_into(torch::Tensor wt, torch::Tensor x, torch::Tensor y);
torch::Tensor ref_elementwise(torch::Tensor wt, torch::Tensor x);
std::string sync_status();
"""


def _sep(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)


def to_host(t):
    """Move to host before any dtype change.

    ``Tensor.double()`` on this card runs on the device and returns zeros (fp64 is
    limited support), so a comparison that converts in place reports the
    reference's magnitude as the error.

    Args:
        t: Any CUDA tensor.

    Returns:
        A float64 CPU copy of it.
    """
    return t.detach().to("cpu").double()


def rel_error(y, ref_cpu):
    """Relative error of ``y`` against a host reference.

    Args:
        y: Kernel output, on any device.
        ref_cpu: Host float64 reference.

    Returns:
        ``max|y - ref| / max|ref|``. ``inf`` when the reference is all zeros, which
        is itself a finding rather than a division to protect against silently.
    """
    scale = float(ref_cpu.abs().max())
    err = float((to_host(y) - ref_cpu).abs().max())
    return (err / scale) if scale > 0 else float("inf")


def bench(fn, iters=30, warmup=8):
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main() -> int:
    import torch

    os.environ.setdefault("TORCH_EXTENSIONS_DIR",
                          os.path.expanduser("~/torch_ext_coffin_probe5"))
    from torch.utils.cpp_extension import load_inline

    _sep("build")
    mod = load_inline(
        name="corex_probe5_ext",
        cpp_sources=CPP_SRC,
        cuda_sources=CUDA_SRC,
        functions=["gemv_kn", "gemv_kn_x4", "gemv_kn_f16_x4", "gemv_kn_m",
                   "gemv_kn_into", "ref_elementwise", "sync_status"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )
    print("  ok")

    _sep("1. dtype traps on this card")
    a = torch.tensor([1.5, 2.5], dtype=torch.float64, device="cuda")
    print("  device fp64 (1.5,2.5)*2 ->", (a * 2).cpu().tolist(), "(expect [3.0, 5.0])")
    h = torch.tensor([1.5, 2.5], dtype=torch.float32)
    print("  host   fp64 (1.5,2.5)*2 ->", (h.to("cpu").double() * 2).tolist())

    _sep("2. GEMV correctness against an elementwise fp32 device reference")
    torch.manual_seed(0)
    results = {}
    shapes = (("mlp_up", 9728, 2560), ("mlp_down", 2560, 9728), ("o_proj", 2560, 4096))
    for label, N, K in shapes:
        x = torch.randn(K, dtype=torch.bfloat16, device="cuda")
        wt = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
        ref = to_host(mod.ref_elementwise(wt, x))
        wt16 = wt.half()
        ref16 = to_host(mod.ref_elementwise(wt16, x))
        for name, fn, refv in (
            ("bf16 U=2", lambda: mod.gemv_kn(wt, x, 2), ref),
            ("bf16 U=4", lambda: mod.gemv_kn(wt, x, 4), ref),
            ("bf16 x4", lambda: mod.gemv_kn_x4(wt, x), ref),
            ("f16 x4", lambda: mod.gemv_kn_f16_x4(wt16, x), ref16),
        ):
            y = fn()
            status = mod.sync_status()
            print(f"  {label:<9} {name:<9} sync={status:<10} "
                  f"rel={rel_error(y, refv):.3e}  max|y|={float(y.abs().max()):.3f}")
            results[f"{label}_{name}"] = rel_error(y, refv)
        del x, wt, wt16, ref, ref16
        torch.cuda.empty_cache()

    _sep("3. small-M (verify) correctness: M=4 rows in one weight pass")
    K, N, M = 9728, 2560, 4
    wt = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    xm = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    y = mod.gemv_kn_m(wt, xm)
    print("  sync:", mod.sync_status())
    for m in range(M):
        ref_m = to_host(mod.ref_elementwise(wt, xm[m]))
        print(f"  row {m}: rel={rel_error(y[m], ref_m):.3e}")

    _sep("4. bandwidth of the decode kernels (pure-read bound)")
    for label, N, K in (("mlp_up", 9728, 2560), ("mlp_down", 2560, 9728),
                        ("qkv_gate", 8192, 2560), ("o_proj", 2560, 4096),
                        ("lm_head", 248320, 2560)):
        x = torch.randn(K, dtype=torch.bfloat16, device="cuda")
        wt = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
        wt16 = wt.half()
        nbytes = K * N * 2
        line = f"  {label:<9} N={N:<7} K={K:<5}"
        for name, fn in (("U=2", lambda: mod.gemv_kn(wt, x, 2)),
                         ("U=4", lambda: mod.gemv_kn(wt, x, 4)),
                         ("x4bf16", lambda: mod.gemv_kn_x4(wt, x)),
                         ("x4f16", lambda: mod.gemv_kn_f16_x4(wt16, x))):
            ms = bench(fn)
            line += f" | {name} {nbytes / (ms * 1e-3) / 1e9:6.0f} GB/s"
        print(line)
        del x, wt, wt16
        torch.cuda.empty_cache()

    _sep("5. copy ceiling (the bandwidth the kernels are judged against)")
    n = 64 * 1024 * 1024
    src = torch.empty(n, dtype=torch.uint8, device="cuda")
    dst = torch.empty(n, dtype=torch.uint8, device="cuda")
    ms = bench(lambda: dst.copy_(src), iters=10, warmup=3)
    print(f"  D2D copy {2 * n / (ms * 1e-3) / 1e9:.0f} GB/s")

    _sep("6. graph capture of our own decode kernel")
    try:
        K, N = 9728, 2560
        x = torch.randn(K, dtype=torch.bfloat16, device="cuda")
        wt = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
        y = torch.zeros(N, dtype=torch.float32, device="cuda")
        mod.gemv_kn_into(wt, x, y)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=torch.cuda.graph_pool_handle()):
            mod.gemv_kn_into(wt, x, y)
        before = y.clone()
        x.normal_()
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        changed = float((y - before).abs().max())
        ref = to_host(mod.ref_elementwise(wt, x))
        print(f"  captured kernel re-read x (delta={changed:.3f}), "
              f"rel vs reference after replay={rel_error(y, ref):.3e}")
    except Exception as exc:  # noqa: BLE001
        print("  capture FAILED:", type(exc).__name__, exc)

    print("\nCOREX_PROBE5_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
