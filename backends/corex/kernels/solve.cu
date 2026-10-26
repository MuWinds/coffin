// @file solve.cu
// @brief Unit-lower-triangular solve, the one library operation the chunked delta rule needs.
//
// Module: coffin/backends/corex/kernels. Bound as ``triangular_solve`` and called from
// ``ops/delta.py`` through ``backends.ops``. It exists because this card's cuSOLVER is
// unusable: the torch wheel loads ``/opt/sw_home/local/cuda/lib64/libcusolver.so``, a path
// that does not exist on the cluster (``/usr/local/corex/lib64/libcusolver.so`` does, but
// nothing reads it), so every ``torch.linalg`` solve path fails inside ``dlopen``. On a
// ROCm build the same op is one library call; here it has to be ours.
//
// Why not a Python ``bmm`` chain: the chunked delta rule solves once per chunk per layer,
// and 24 linear-attention layers over a prompt is a few hundred calls, each of which would
// be 4-8 launches from Python plus the strided copies between them.
//
// Constraints:
//   * **No barriers.** Forward substitution is independent across right-hand-side columns:
//     ``x[i][c]`` depends only on ``x[j][c]`` for ``j < i``, so a thread owning one column
//     never needs another thread's work. The whole solve is a thread-local loop with no
//     ``__syncthreads()``, which also means it makes no assumption about warp width --
//     this part's warp is 64 lanes, and a reduction-shaped implementation would have had
//     to care.
//   * The running solution is kept in shared memory in **fp32** even for bf16 operands,
//     and only the written row is narrowed. Substitution is a chain: row ``i`` consumes
//     every earlier row, so one rounding at row 0 propagates through all ``L`` rows. The
//     reference solves in the operand dtype, so this side is the more accurate one, which
//     is the direction the conformance table allows.
//   * The diagonal is taken as 1 and the upper triangle is never read. The delta rule's
//     ``UT`` is ``I`` plus a strictly-lower product, so a stored diagonal would be a
//     different system, and consulting it would make the answer depend on memory the
//     caller never wrote.
//   * Batch strides are passed explicitly rather than assumed contiguous. The caller's
//     right-hand side is ``torch.eye(L).expand(bn, L, L)``, whose batch stride is 0: one
//     identity block is shared by every system. Materialising it would copy ``bn * L * L``
//     elements per layer per chunk to read one ``L * L`` block, and reading the stride says
//     exactly that without a special case.
//   * Shared memory is ``L * blockDim.x`` floats, so ``L`` is bounded by the per-block
//     limit. The host wrapper rejects a shape that would not fit rather than launching a
//     kernel that reads past its allocation.
//   * ``out`` is a fresh allocation and never aliases the inputs, so a caller may pass the
//     same tensor as both ``lower`` and ``rhs`` without the solve clobbering it mid-flight.

#include "corex_ops.h"
#include "corex_scalar.h"

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

namespace {

//: Right-hand-side columns one block resolves. 64 is a wavefront on this part, so each
//: block's lanes read one row of ``lower`` as a single broadcast and the shared-memory
//: stride stays the natural one.
constexpr int MAX_COLS = 64;

//: Cap on dynamic shared memory. 48 KB is the limit that needs no opt-in attribute on this
//: part, so staying under it keeps the launch unconditional.
constexpr size_t MAX_SMEM = 48u * 1024u;

// @brief Solve ``A X = B`` for unit-lower-triangular ``A``, one thread per rhs column.
//
// @param lower [bn, L, L] coefficients; only the strictly-lower part is read, the diagonal
//        is taken as 1.
// @param rhs [bn, L, n] right-hand side.
// @param out [bn, L, n] solution, written in ``rhs``'s dtype.
// @param L System order.
// @param n Right-hand-side columns.
// @param ldb Batch stride of ``lower``, in elements.
// @param ldr Batch stride of ``rhs``/``out``, in elements.
// @param stride ``blockDim.x``: the shared-memory row pitch, i.e. the padded thread count.
// @param cols Active columns per block.
template <typename T>
__global__ void solve_kernel(const T* __restrict__ lower, const T* __restrict__ rhs,
                             T* __restrict__ out, int L, int n, long ldb, long ldr,
                             int stride, int cols) {
    const int b = blockIdx.x;
    const int lane = (int)threadIdx.x;
    const int c = blockIdx.y * cols + lane;
    // No barrier below, so a padded lane leaves without waiting on anyone.
    if (c >= n) return;

    extern __shared__ float xs[];       // [L][stride] this column's running solution
    const T* a = lower + (size_t)b * ldb;
    const T* rhs_b = rhs + (size_t)b * ldr;
    T* out_b = out + (size_t)b * ldr;

    for (int i = 0; i < L; ++i) {
        float acc = CxScalar<T>::to_float(rhs_b[(size_t)i * n + c]);
        const T* row = a + (size_t)i * L;
        // Row ``i`` is read by every lane at the same offset, so the load broadcasts out of
        // the read-only path; the shared-memory reads are per-lane and coalesced.
        for (int j = 0; j < i; ++j) {
            acc -= CxScalar<T>::to_float(row[j]) * xs[(size_t)j * stride + lane];
        }
        xs[(size_t)i * stride + lane] = acc;
        out_b[(size_t)i * n + c] = CxScalar<T>::from_float(acc);
    }
}

// @brief Launch the solve for one operand dtype.
//
// @param lower [bn, L, L] coefficient matrix.
// @param rhs [bn, L, n] right-hand side.
// @param out [bn, L, n] destination.
// @param bn Batch of independent systems.
// @param L Order.
// @param n Right-hand-side columns.
// @param ldb Batch stride of ``lower``.
// @param ldr Batch stride of ``rhs``/``out``.
// @param stream Launch stream; must be the current torch stream or a capture drops it.
template <typename T>
void launch_solve(const torch::Tensor& lower, const torch::Tensor& rhs, torch::Tensor& out,
                  int bn, int L, int n, long ldb, long ldr, cudaStream_t stream) {
    const int cols = n < MAX_COLS ? n : MAX_COLS;
    const int threads = ((cols + 63) / 64) * 64;
    const size_t smem = (size_t)L * threads * sizeof(float);
    const dim3 grid(bn, (n + cols - 1) / cols);
    solve_kernel<T><<<grid, threads, smem, stream>>>(
        reinterpret_cast<const T*>(lower.data_ptr()),
        reinterpret_cast<const T*>(rhs.data_ptr()), reinterpret_cast<T*>(out.data_ptr()),
        L, n, ldb, ldr, threads, cols);
}

}  // namespace

// @brief Solve ``A X = B`` for a unit-lower-triangular ``A``.
//
// @param lower [bn, L, L] unit-lower-triangular coefficients, bf16 or fp32.
// @param rhs [bn, L, n] right-hand side, same dtype.
// @return [bn, L, n] in ``rhs``'s dtype.
// @throws c10::Error when the shapes disagree, a dtype has no instantiation, an operand is
//         not row-contiguous, or ``L`` would not fit in shared memory -- each of those
//         either reads past an allocation or refuses to launch, and the host is the only
//         place a caller can see it.
torch::Tensor triangular_solve(torch::Tensor lower, torch::Tensor rhs) {
    TORCH_CHECK(lower.dim() == 3 && rhs.dim() == 3,
                "lower and rhs must both be 3-D: [bn, L, L] and [bn, L, n]");
    TORCH_CHECK(lower.scalar_type() == rhs.scalar_type(),
                "lower and rhs must share a dtype");
    TORCH_CHECK(lower.scalar_type() == torch::kBFloat16
                || lower.scalar_type() == torch::kFloat,
                "operands must be bf16 or fp32");
    const int bn = (int)rhs.size(0);
    const int L = (int)rhs.size(1);
    const int n = (int)rhs.size(2);
    TORCH_CHECK(lower.size(0) == bn && lower.size(1) == L && lower.size(2) == L,
                "lower must be [bn, L, L] matching the right-hand side");
    // Only the row layout is fixed: the innermost stride has to be 1 for the coalesced
    // shared reads, and the batch stride is read from the tensor. That admits
    // ``eye(L).expand(bn, L, L)``, whose batch stride is 0, without copying it.
    TORCH_CHECK(rhs.stride(2) == 1 && lower.stride(2) == 1,
                "operands must have unit innermost stride");
    TORCH_CHECK(rhs.stride(1) == n && lower.stride(1) == L,
                "operands must be row-contiguous within a batch entry");
    const int cols = n < MAX_COLS ? n : MAX_COLS;
    const int threads = ((cols + 63) / 64) * 64;
    TORCH_CHECK((size_t)L * threads * sizeof(float) <= MAX_SMEM,
                "L=", L, " needs ", (size_t)L * threads * sizeof(float),
                " bytes of shared memory, over the ", MAX_SMEM,
                " byte limit; reduce the delta chunk length rather than widening it");

    auto out = torch::empty({bn, L, n}, rhs.options());
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const long ldb = (long)lower.stride(0);
    const long ldr = (long)rhs.stride(0);
    if (lower.scalar_type() == torch::kBFloat16) {
        launch_solve<uint16_t>(lower, rhs, out, bn, L, n, ldb, ldr, stream);
    } else {
        launch_solve<float>(lower, rhs, out, bn, L, n, ldb, ldr, stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
