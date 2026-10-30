// @file delta.cu
// @brief The gated-delta-rule decode step: one in-place update of the recurrent state.
//
// Module: coffin/backends/corex/kernels. Consumes the fp32 q/k/v/beta/g that
// ``linear_attn.cu`` produces and updates the state in place; the batched form is
// ``delta_batch`` in this file. ``delta_normed`` is the decode path: it takes the raw
// conv output instead and does the head preparation, the recurrence and the gated output
// norm in one launch, so nothing crosses kernel boundaries between them.
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
//   * ``delta_normed`` is bit-identical to the ``conv1d_silu`` -> ``delta_prologue`` ->
//     ``delta`` -> ``rms_norm_gated`` chain, not merely close. That forces two things the
//     other kernels in this file do not need: the block-wide reductions must reproduce the
//     chain's shared-tree pairing exactly (hence the same ``block_sum2`` shape and the same
//     block size), and the recurrence's output is rounded to the activation dtype before it
//     is normalised, because ``RMSNormGated`` casts the fp32 ``delta_out`` to bf16 before
//     its reduction. The greedy ids are the acceptance bar and this is what keeps them.

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
// Grid ``z`` is the batch row. Every row is an independent ``[heads, dk, dv]`` slab, so the
// block simply offsets its pointers and touches nothing another row owns; there is no
// cross-row communication to add. Measured, this is worth far more than the launch it
// replaces: ``delta_batch`` ran one launch per row and each launch had only ``heads`` = 32
// blocks in flight, and this part sustains ~3.0 GB/s per block, so a row's 6.29 MB moved at
// 101.7 GB/s (61.89 us). Rows on the ``z`` axis put ``32 * rows`` blocks in flight instead.
//
// @param q,k [rows, heads, dk] fp32, already L2-normalised (and q scaled) by the prologue.
// @param v [rows, heads, dv] fp32.
// @param beta,g [rows, heads] fp32 write strength and log decay.
// @param st [rows, heads, dk, dv] fp32 state, updated in place.
// @param out [rows, heads, dv] fp32, written from the updated state.
// @param heads Heads per row, which is also the ``x`` grid extent; the row strides below are
//        derived from it and are what the previous per-row host loop used to apply itself.
// @param dk Reduction length, also the stride between successive value columns.
// @param dv Value columns per head.
__global__ void delta_kernel(const float* __restrict__ q, const float* __restrict__ k,
                             const float* __restrict__ v, const float* __restrict__ beta,
                             const float* __restrict__ g, float* __restrict__ st,
                             float* __restrict__ out, int heads, int dk, int dv) {
    const int head = blockIdx.x;
    const int j = (int)(blockIdx.y * blockDim.x + threadIdx.x);

    // Row offsets, not pointer arithmetic on the host: the buffers are dense and the row
    // strides are the shapes themselves, so the kernel can name row ``b`` with no pitch
    // argument and no second launch. ``z`` is 1 for the single-row decode entry, and row 0
    // then adds zero to every pointer, so that path reads the identical bytes it always did.
    const size_t row = blockIdx.z;
    const size_t oq = row * heads * dk;          // q and k, [heads, dk] per row
    const size_t ov = row * heads * dv;          // v and out, [heads, dv] per row
    const size_t ob = row * heads;               // beta and g, [heads] per row
    const size_t os = row * heads * dk * dv;     // state, [heads, dk, dv] per row

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
        ks[i] = k[oq + (size_t)head * dk + i];
        qs[i] = q[oq + (size_t)head * dk + i];
    }
    __shared__ float decay_sh;
    __shared__ float rate_sh;
    if (threadIdx.x == 0) {
        decay_sh = __expf(g[ob + head]);
        rate_sh = beta[ob + head];
    }
    __syncthreads();
    const float decay = decay_sh;
    const float rate = rate_sh;
    if (!active) return;

    const float value = v[ov + (size_t)head * dv + j];
    float* column = st + os + (size_t)head * dk * dv + j;

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
    out[ov + (size_t)head * dv + j] = result;
}

// @brief Launch the recurrence over ``rows`` independent state slabs.
//
// One block per (head, column tile, row). The ``z`` extent is the row, and it is 1 for the
// single-row decode entry, so both entry points share this one launch shape and the decode
// path is unchanged by construction rather than by a separate code path.
//
// The grid is deliberately the row axis rather than a host loop: this part sustains ~3.0 GB/s
// per block in flight (``docs/DESIGN.md`` §2.2), a row here is only ``heads`` = 32 blocks, and
// a loop therefore runs every row at 32-block bandwidth no matter how many rows there are.
//
// @param q,k [rows, heads, dk] fp32.
// @param v [rows, heads, dv] fp32.
// @param beta,g [rows, heads] fp32.
// @param st [rows, heads, dk, dv] fp32, updated in place.
// @param out [rows, heads, dv] fp32.
// @param rows Batch rows; becomes the grid's ``z`` extent.
// @param stream Stream to launch on.
void launch_rows(const float* q, const float* k, const float* v, const float* beta,
                 const float* g, float* st, float* out, int rows, int heads, int dk, int dv,
                 cudaStream_t stream) {
    const int threads = dv < THREADS ? dv : THREADS;
    const int blocks_y = (dv + threads - 1) / threads;
    const size_t smem = (size_t)(2 * dk) * sizeof(float);
    delta_kernel<<<dim3(heads, blocks_y, rows), threads, smem, stream>>>(
        q, k, v, beta, g, st, out, heads, dk, dv);
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
    launch_rows(q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
                beta.data_ptr<float>(), g.data_ptr<float>(), st.data_ptr<float>(),
                out.data_ptr<float>(), 1, heads, dk, dv,
                c10::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// @brief Batched gated-delta step: one independent state row per batch entry.
//
// Row ``b`` is identical to :func:`delta` on that row alone, which is the property the
// conformance check measures. The rows stay independent because each gets its own grid
// slice; no cross-row synchronisation exists at all. It is one launch for every row: the
// grid's ``z`` axis is the batch, so the rows that used to cost a launch each now cost one
// launch between them and run ``rows`` times as many blocks concurrently.
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
    // The kernel's row offsets are ``b * heads * dk * dv`` and ``b * heads * dv``, which name
    // row ``b`` only when the batch axis has exactly those strides. A caller that handed in a
    // widened view would have every row read from the wrong place, and the symptom would be a
    // small relative error on the batched path whose single-row form is exact.
    TORCH_CHECK(st.is_contiguous() && out.is_contiguous(),
                "state and output must be contiguous: the per-row stride assumes it");
    TORCH_CHECK(q.scalar_type() == torch::kFloat && st.scalar_type() == torch::kFloat,
                "delta inputs and state are fp32");
    const int rows = (int)st.size(0);
    const int heads = (int)st.size(1);
    const int dk = (int)st.size(2);
    const int dv = (int)st.size(3);
    launch_rows(q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
                beta.data_ptr<float>(), g.data_ptr<float>(), st.data_ptr<float>(),
                out.data_ptr<float>(), rows, heads, dk, dv,
                c10::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
