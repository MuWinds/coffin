// @file delta.cu
// @brief The gated-delta-rule decode step: one in-place update of the recurrent state.
//
// Module: coffin/backends/corex/kernels. Consumes the fp32 q/k/v/beta/g that
// ``linear_attn.cu`` produces and updates the state in place; the batched form is
// ``delta_batch`` in this file.
//
// The recurrence per token is
//     st  = st * exp(g)[:, None, None]
//     kv  = (st * k[:, :, None]).sum(-2)
//     st  = st + k[:, :, None] * ((v - kv) * beta)[:, None, :]
//     out = (st * q[:, :, None]).sum(-2)
// As a chain of torch ops that is ~140 us per layer on the DCU, which over 24 layers is
// about a tenth of a decode step. The state has no closed-form rewind, so it is the one
// piece of layer memory that prefix reuse has to snapshot.
//
// Constraints:
//   * One block covers a tile of (head, value-columns), and inside it the threads are
//     addressed as ``(column, dk index)``. A state column is only ever read and written
//     by threads of its own column group, which is what removes cross-thread
//     communication from the recursion.
//   * The ``dk`` reduction is a per-group shared-memory tree, not ``__shfl_xor``. This
//     part's warp is 64 lanes wide, so a butterfly written for 32 would combine the wrong
//     partners and still produce a plausible number. A tree costs one barrier per step
//     and makes no assumption about the width.
//   * The state is fp32 and updated in place, and the output is read from the *updated*
//     state so a token sees its own write. Any other order changes the recurrence rather
//     than merely the rounding.
//   * The decay is re-applied on the second pass instead of being written back twice:
//     ``k^T (e^g S) == e^g (k^T S)``, which removes one read-modify-write of the state
//     per layer per token.
//   * The kernel must not allocate a replacement for ``st``: the capture path holds that
//     address fixed, so a new buffer would leave the graph writing to a dead tensor.
//   * Inputs are fp32 and contiguous. The prologue has already normalised and scaled q
//     and k, so this kernel applies no per-element fix-up.

#include "corex_ops.h"
#include "corex_scalar.h"

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

namespace {

//: Value columns per block. One column's group is ``dk`` threads, so a tile of 32 columns
//: would need 4096 threads; the launch instead fits as many whole column groups as the
//: thread budget allows, and this constant is that budget divided by the smallest head.
constexpr int MAX_THREADS = 512;

//: Threads one column group may use. A head of 128 gets 128 threads, so one block covers
//: ``MAX_THREADS / 128 = 4`` columns and every wavefront is fully used.
int group_size(int dk) {
    int t = dk < 32 ? 32 : dk;
    t = ((t + 63) / 64) * 64;
    return t > MAX_THREADS ? MAX_THREADS : t;
}

//: Whole column groups per block.
int cols_per_block(int dk) {
    const int g = group_size(dk);
    const int c = MAX_THREADS / g;
    return c < 1 ? 1 : c;
}

// @brief Sum inside one column group through a shared tree.
//
// The scratch is a flat ``[blockDim.x]`` array and a column's group owns a contiguous
// segment of it, so a group never reads another column's partial. Every thread of the
// block must call this: it ends on a barrier.
//
// @param v This thread's contribution.
// @param red Shared scratch, ``blockDim.x`` floats; clobbered.
// @param group Threads in one column group.
// @param lane This thread's index inside its own group.
// @return The sum over this thread's column group, valid in every thread of the group.
//
// The closing ``__syncthreads()`` is load-bearing, not tidiness. Without it a thread can
// return, compute, and reach the *next* ``group_sum``'s ``red[threadIdx.x] = v`` before a
// slower thread of its own group has read ``red[base]`` from this one -- and for the group
// at ``base == 0`` that write lands on exactly the slot still being read. The recurrence
// calls this twice per step, so the race is reached on the first token. Its symptom is a
// result that is wrong by a fraction of a percent rather than wildly wrong, which is easy
// to misread as a precision question.
__device__ __forceinline__ float group_sum(float v, float* red, int group, int lane) {
    const int base = (int)threadIdx.x - lane;
    red[threadIdx.x] = v;
    __syncthreads();
    for (int step = group >> 1; step > 0; step >>= 1) {
        if (lane < step) red[base + lane] += red[base + lane + step];
        __syncthreads();
    }
    const float total = red[base];
    // The barrier here is what lets every thread of the group read ``total`` safely before
    // any thread starts overwriting the scratch for a following reduction.
    __syncthreads();
    return total;
}

// @brief One step of the gated delta rule: one block per (head, tile of value columns).
//
// @param q,k [heads, dk] fp32, already L2-normalised (and q scaled) by the prologue.
// @param v [heads, dv] fp32.
// @param beta,g [heads] fp32 write strength and log decay.
// @param st [heads, dk, dv] fp32 state, updated in place.
// @param out [heads, dv] fp32, written from the updated state.
// @param group Threads per column group, i.e. ``dk`` rounded to the wavefront.
// @param cols Value columns this block covers.
__global__ void delta_kernel(const float* __restrict__ q, const float* __restrict__ k,
                             const float* __restrict__ v, const float* __restrict__ beta,
                             const float* __restrict__ g, float* __restrict__ st,
                             float* __restrict__ out, int dk, int dv, int group, int cols) {
    const int head = blockIdx.x;
    const int lane = (int)threadIdx.x % group;           // index inside the column group
    const int slot = (int)threadIdx.x / group;           // which column of this tile
    const int j = blockIdx.y * cols + slot;              // value column

    extern __shared__ float sh[];
    float* ks = sh;                  // [dk] this head's key, shared by every column
    float* qs = sh + dk;             // [dk] this head's query
    float* red = qs + dk;            // [blockDim.x] reduction scratch
    // One column group loads the shared copies. Every group needs all of ``dk``, so the
    // load is duplicated work either way; confining it to the first group avoids a
    // read-write race on the same shared locations.
    if (slot == 0) {
        for (int i = lane; i < dk; i += group) {
            ks[i] = k[(size_t)head * dk + i];
            qs[i] = q[(size_t)head * dk + i];
        }
    }
    __shared__ float decay_sh;
    __shared__ float rate_sh;
    if (threadIdx.x == 0) {
        decay_sh = __expf(g[head]);
        rate_sh = beta[head];
    }
    __syncthreads();
    const float decay = decay_sh;
    const float rate = rate_sh;
    // A column past ``dv`` still runs the reductions: ``group_sum`` ends on a block-wide
    // barrier, so returning here would leave the remaining groups waiting forever. The
    // inactive column contributes zeros and skips its own state access instead.
    const bool active = j < dv;
    const float value = active ? v[(size_t)head * dv + j] : 0.f;
    float* column = active ? st + ((size_t)head * dk) * dv + j : nullptr;

    // Decay applied on read, then again on write, so the state is touched once.
    float kv_mem = 0.f;
    if (active) {
        for (int i = lane; i < dk; i += group) {
            kv_mem = fmaf(column[(size_t)i * dv] * decay, ks[i], kv_mem);
        }
    }
    kv_mem = group_sum(kv_mem, red, group, lane);

    const float delta = (value - kv_mem) * rate;
    float result = 0.f;
    if (active) {
        for (int i = lane; i < dk; i += group) {
            const float s = fmaf(ks[i], delta, column[(size_t)i * dv] * decay);
            column[(size_t)i * dv] = s;
            result = fmaf(s, qs[i], result);
        }
    }
    result = group_sum(result, red, group, lane);
    if (active && lane == 0) out[(size_t)head * dv + j] = result;
}

// @brief Launch one row of the recurrence.
//
// @param q,k [heads, dk] fp32.
// @param v [heads, dv] fp32.
// @param beta,g [heads] fp32.
// @param st [heads, dk, dv] fp32, updated in place.
// @param out [heads, dv] fp32.
// @param stream Stream to launch on.
void launch_row(const float* q, const float* k, const float* v, const float* beta,
                const float* g, float* st, float* out, int heads, int dk, int dv,
                cudaStream_t stream) {
    const int group = group_size(dk);
    const int cols = cols_per_block(dk);
    const int threads = group * cols;
    const int blocks_y = (dv + cols - 1) / cols;
    const size_t smem = (size_t)(2 * dk + threads) * sizeof(float);
    delta_kernel<<<dim3(heads, blocks_y), threads, smem, stream>>>(
        q, k, v, beta, g, st, out, dk, dv, group, cols);
}

}  // namespace

// @brief One gated-delta-rule decode step, advancing the recurrent state in place.
//
// @param q,k [heads, dk] fp32, already L2-normalised by the prologue with q additionally
//        scaled, so this kernel applies no per-element fix-up.
// @param v [heads, dv] fp32.
// @param beta,g [heads] fp32 rate and log decay; ``g <= 0`` or the recurrence grows.
// @param st [heads, dk, dv] fp32 state, updated in place -- the caller owns the snapshot
//        discipline that prefix reuse depends on.
// @param out [heads, dv] fp32, written from the updated state.
// @return ``out``, for call chaining.
// @throws c10::Error when a non-beta/g input is non-contiguous, a dtype is not fp32, or
//         the head counts disagree with the state.
torch::Tensor delta(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor beta,
                    torch::Tensor g, torch::Tensor st, torch::Tensor out) {
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous()
                && beta.is_contiguous() && g.is_contiguous(),
                "contiguous inputs required");
    TORCH_CHECK(q.scalar_type() == torch::kFloat && st.scalar_type() == torch::kFloat,
                "delta inputs and state are fp32");
    const int heads = (int)st.size(0);
    const int dk = (int)st.size(1);
    const int dv = (int)st.size(2);
    TORCH_CHECK(q.size(0) == heads && k.size(0) == heads && v.size(0) == heads,
                "q/k/v head count does not match the state");
    launch_row(q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
               beta.data_ptr<float>(), g.data_ptr<float>(), st.data_ptr<float>(),
               out.data_ptr<float>(), heads, dk, dv,
               c10::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// @brief Batched gated-delta step: one independent state row per batch entry.
//
// Row ``b`` is identical to :func:`delta` on that row alone, which is the property the
// conformance check measures. The rows stay independent because each gets its own grid
// slice; no cross-row synchronisation exists at all.
//
// @param q,k [B, heads, dk] fp32.
// @param v [B, heads, dv] fp32.
// @param beta,g [B, heads] fp32.
// @param st [B, heads, dk, dv] fp32, updated in place.
// @param out [B, heads, dv] fp32, written in place.
// @return ``out``, for call chaining.
// @throws c10::Error when a dtype contradicts the layout.
torch::Tensor delta_batch(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                          torch::Tensor beta, torch::Tensor g, torch::Tensor st,
                          torch::Tensor out) {
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous()
                && beta.is_contiguous() && g.is_contiguous(),
                "contiguous inputs required");
    // The state and output are checked too, and that is not symmetry for its own sake.
    // The per-row offset below is ``b * heads * dk * dv``, which names row ``b`` only when
    // the batch axis has exactly that stride. A caller that handed in a widened view would
    // have every row read from the wrong place, and the symptom would be a small relative
    // error on the batched path whose single-row form is exact.
    TORCH_CHECK(st.is_contiguous() && out.is_contiguous(),
                "state and output must be contiguous: the per-row stride assumes it");
    TORCH_CHECK(q.scalar_type() == torch::kFloat && st.scalar_type() == torch::kFloat,
                "delta inputs and state are fp32");
    const int rows = (int)st.size(0);
    const int heads = (int)st.size(1);
    const int dk = (int)st.size(2);
    const int dv = (int)st.size(3);
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    for (int b = 0; b < rows; ++b) {
        // One launch per row rather than a fourth grid axis: ``dv`` and the head count
        // already fill two axes, and a handful of launches is cheaper than a grid axis the
        // kernel would have to decode back into two indices.
        launch_row(q.data_ptr<float>() + (size_t)b * heads * dk,
                   k.data_ptr<float>() + (size_t)b * heads * dk,
                   v.data_ptr<float>() + (size_t)b * heads * dv,
                   beta.data_ptr<float>() + (size_t)b * heads,
                   g.data_ptr<float>() + (size_t)b * heads,
                   st.data_ptr<float>() + (size_t)b * heads * dk * dv,
                   out.data_ptr<float>() + (size_t)b * heads * dv, heads, dk, dv, stream);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}
