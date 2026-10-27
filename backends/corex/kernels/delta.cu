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
//   * One thread owns one value column and one block covers a head, with the threads of a
//     block addressed as consecutive ``j``. That is forced by the state layout: column
//     ``j`` of a head lives at ``st[head*dk*dv + i*dv + j]``, so consecutive ``j`` at the
//     same ``i`` is the only assignment that coalesces. The earlier ``(column, dk index)``
//     mapping read one column per lane at a stride of ``dv`` floats and pulled 32x the
//     bytes it used, which is why it measured 294 us against a 7 us floor.
//   * Both ``dk`` reductions are per-thread register loops, so there is no cross-lane
//     communication and no shared scratch. This relies on the recurrence coupling ``i``
//     only within a column; a version that needed a sum across columns would have to
//     reintroduce a tree -- and a written-for-32 butterfly would combine the wrong
//     partners here, because this part's warp is 64 lanes wide.
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

//: Threads per block. Each thread owns one value column and walks the whole ``dk`` axis
//: in registers, so a block covers ``THREADS`` columns and the block count falls by 32x
//: against the old shape (32 heads x 128/THREADS column tiles).
//:
//: The mapping is the whole optimisation and it is worth stating why. The state is
//: ``[heads, dk, dv]``, so column ``j`` of a head is ``st[head*dk*dv + i*dv + j]``: fixed
//: ``i``, contiguous in ``j``. Threads are therefore addressed as consecutive ``j`` at the
//: same ``i``, which makes a 64-lane wavefront read 64 consecutive floats -- one 256-byte
//: transaction. The previous mapping put the threads of a column group on consecutive
//: ``i`` instead, so every lane in a warp touched a different 128-byte line and used four
//: bytes of it: 32x the traffic for the same answer, measured at 294 us per layer against
//: a 7.3 us floor.
//:
//: 128 keeps a whole head's value axis (``dv``) inside one block for this model, so the
//: shared copies of q and k are loaded once and reused by every column. It is also four
//: wavefronts per block on a 64-lane part, which leaves enough warps to hide latency.
constexpr int THREADS = 128;

// @brief One step of the gated delta rule: one thread per value column, one block per head.
//
// Each thread owns column ``j`` outright and walks ``i`` over ``dk`` in a register loop,
// so both reductions across ``dk`` are local to that thread and no cross-lane
// communication happens at all. That is only correct because the two sums are independent
// per column -- the recurrence couples ``i`` within a column, never across columns.
//
// @param q,k [heads, dk] fp32, already L2-normalised (and q scaled) by the prologue.
// @param v [heads, dv] fp32.
// @param beta,g [heads] fp32 write strength and log decay.
// @param st [heads, dk, dv] fp32 state, updated in place.
// @param out [heads, dv] fp32, written from the updated state.
// @param dk Reduction length, also the stride between successive value columns.
// @param dv Value columns per head.
__global__ void delta_kernel(const float* __restrict__ q, const float* __restrict__ k,
                             const float* __restrict__ v, const float* __restrict__ beta,
                             const float* __restrict__ g, float* __restrict__ st,
                             float* __restrict__ out, int dk, int dv) {
    const int head = blockIdx.x;
    const int j = (int)(blockIdx.y * blockDim.x + threadIdx.x);

    // A thread past ``dv`` must NOT return here.``__syncthreads()`` below is reached by
    // every thread of the block, so an early exit leaves the rest waiting on a thread
    // that is gone -- a device-side deadlock with no traceback, which is exactly the
    // symptom that once hung a card for a full job limit. An out-of-range column instead
    // runs the barriers and skips only its own state and output access.
    const bool active = j < dv;

    extern __shared__ float sh[];
    float* ks = sh;                  // [dk] this head's key, shared by every column
    float* qs = sh + dk;             // [dk] this head's query
    // Every column needs all of ``dk``, so the copy is shared rather than repeated per
    // thread. Threads cooperate on the load, which is contiguous either way.
    for (int i = (int)threadIdx.x; i < dk; i += (int)blockDim.x) {
        ks[i] = k[(size_t)head * dk + i];
        qs[i] = q[(size_t)head * dk + i];
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
    if (!active) return;

    const float value = v[(size_t)head * dv + j];
    float* column = st + (size_t)head * dk * dv + j;

    // Decay applied on read and again on write, so the state is touched once per step.
    // Threads at the same ``i`` are adjacent in ``j``, so each ``i`` step is one coalesced
    // transaction per wavefront: this is the access pattern the old mapping lacked.
    float kv_mem = 0.f;
    for (int i = 0; i < dk; ++i) {
        kv_mem = fmaf(column[(size_t)i * dv] * decay, ks[i], kv_mem);
    }

    const float delta = (value - kv_mem) * rate;
    float result = 0.f;
    for (int i = 0; i < dk; ++i) {
        const float s = fmaf(ks[i], delta, column[(size_t)i * dv] * decay);
        column[(size_t)i * dv] = s;
        result = fmaf(s, qs[i], result);
    }
    out[(size_t)head * dv + j] = result;
}

// @brief Launch one row of the recurrence.
//
// One block per head, ``THREADS`` value columns per block. Grid ``y`` covers the rest of
// ``dv`` when it is wider than a block, which this model's 128 does not need but a wider
// head would.
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
    const int threads = dv < THREADS ? dv : THREADS;
    const int blocks_y = (dv + threads - 1) / threads;
    const size_t smem = (size_t)(2 * dk) * sizeof(float);
    delta_kernel<<<dim3(heads, blocks_y), threads, smem, stream>>>(
        q, k, v, beta, g, st, out, dk, dv);
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
