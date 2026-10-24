// Gated DeltaNet decode step (the 24 linear-attention layers of Qwen3.5).
//
// The recurrence per token is
//     st = st * exp(g)[:, None, None]
//     kv = (st * k[:, :, None]).sum(-2)
//     st = st + k[:, :, None] * ((v - kv) * beta)[:, None, :]
//     out = (st * q[:, :, None]).sum(-2)
// As a chain of torch ops that is ~140 us per layer, i.e. 3.4 ms/token over 24
// layers and about a tenth of the whole decode step. Every thread owns one
// column j of the [dk, dv] state, and a column is only ever read and written by
// its own thread, so no cross-thread communication is needed and the whole
// recurrence collapses into a single kernel.

#include "hip_ops.h"

#include <c10/cuda/CUDAStream.h>
#include <hip/hip_runtime.h>

namespace {

// One block per (head, tile of value columns); DELTA_SPLIT threads share each
// column, owning one slice of the dk loop, and the per-column partials are
// combined with two shuffles inside the warp.
//
// The earlier shape -- one thread per column, blockDim = dv = 128 -- launched 32
// blocks for 64 CUs (half the card idle) and gave every thread a 2x128-iteration
// serial walk with only two warps per CU to hide L2 latency.  It measured
// 2294 us/step on the int8 decode profile, 15.6% of the step; splitting dk four
// ways cuts the serial chain to dk/4 and the block count up.
//
// k and q are staged in shared memory because every thread needs all dk of them.
// The first pass deliberately does not write e^g*S back: k^T(e^g S) = e^g(k^T S),
// so the decay is applied again in the second pass, which removes one
// read-modify-write of the 2 MB state per layer per token.
#define DELTA_SPLIT 4

__global__ void delta_kernel(const float* __restrict__ q, const float* __restrict__ k,
                             const float* __restrict__ v, const float* __restrict__ beta,
                             const float* __restrict__ g, float* __restrict__ st,
                             float* __restrict__ out, int dk, int dv) {
    const int head = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int ii = lane & (DELTA_SPLIT - 1);              // dk slice
    const int jj = lane >> 2;                             // column within the warp
    const int cols_per_block = (blockDim.x >> 5) * (32 / DELTA_SPLIT);
    const int j = blockIdx.y * cols_per_block + warp * (32 / DELTA_SPLIT) + jj;

    extern __shared__ float shared[];
    float* ks = shared;
    float* qs = shared + dk;
    for (int i = threadIdx.x; i < dk; i += blockDim.x) {
        ks[i] = k[(size_t)head * dk + i];
        qs[i] = q[(size_t)head * dk + i];
    }
    __shared__ float decay, rate;
    if (threadIdx.x == 0) {
        decay = __expf(g[head]);
        rate = beta[head];
    }
    __syncthreads();

    if (j >= dv) return;                                  // whole 4-lane group exits together
    float* column = st + (size_t)head * dk * dv + j;
    const float value = v[(size_t)head * dv + j];

    const int slice = dk / DELTA_SPLIT;
    const int i0 = ii * slice, i1 = i0 + slice;
    float kv_mem = 0.f;
    for (int i = i0; i < i1; ++i) {
        kv_mem = fmaf(column[(size_t)i * dv] * decay, ks[i], kv_mem);
    }
#pragma unroll
    for (int o = 1; o < DELTA_SPLIT; o <<= 1) kv_mem += __shfl_xor(kv_mem, o);

    const float delta = (value - kv_mem) * rate;
    float result = 0.f;
    for (int i = i0; i < i1; ++i) {
        const float s = fmaf(ks[i], delta, column[(size_t)i * dv] * decay);
        column[(size_t)i * dv] = s;
        result = fmaf(s, qs[i], result);
    }
#pragma unroll
    for (int o = 1; o < DELTA_SPLIT; o <<= 1) result += __shfl_xor(result, o);
    if (ii == 0) out[(size_t)head * dv + j] = result;
}

}  // namespace

torch::Tensor delta_step(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                         torch::Tensor beta, torch::Tensor g,
                         torch::Tensor st, torch::Tensor out) {
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous()
                && beta.is_contiguous() && g.is_contiguous(),
                "contiguous inputs required");
    const int heads = st.size(0), dk = st.size(1), dv = st.size(2);
    TORCH_CHECK(q.size(0) == heads && k.size(0) == heads && v.size(0) == heads,
                "q/k/v head count does not match the state");
    TORCH_CHECK(dk % DELTA_SPLIT == 0, "dk must be a multiple of DELTA_SPLIT");
    const int threads = 128, cols_per_block = (threads / 32) * (32 / DELTA_SPLIT);
    const int blocks_y = (dv + cols_per_block - 1) / cols_per_block;
    hipStream_t stream = at::cuda::getCurrentCUDAStream();
    delta_kernel<<<dim3(heads, blocks_y), threads, 2 * dk * sizeof(float), stream>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
        beta.data_ptr<float>(), g.data_ptr<float>(),
        st.data_ptr<float>(), out.data_ptr<float>(), dk, dv);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
