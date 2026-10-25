#pragma once

// @file hip_ops.h
// @brief Declarations of the HIP extension entry points bound in bindings.cpp.
//
// Module: coffin/kernels. The only header the kernels and ``hip_ops.py`` share;
// it is a pure interface, so it must stay free of kernel bodies and of anything
// that pulls a second HIP translation unit into a compile.
//
// Constraints:
//   * Every function here is declared exactly once and defined in one of the
//     ``.cu`` files; a signature change has to land on both sides or the
//     extension fails to link rather than to launch.
//   * Weights are INT8 row-major [N, K] with a per-row fp32 scale. A new
//     quantisation layout breaks ``gemv_w8a16*`` and every caller's prepacked
//     checkpoint, so it needs a new entry point.
//   * Shapes are part of the contract: the kernels index without bounds checks,
//     and the host wrappers TORCH_CHECK only what they were written to check.
//   * Partial buffers (``part_acc`` / ``part_ml``) are caller-owned so their
//     shapes stay static for graph capture; do not allocate them inside the op.

#include <torch/extension.h>

// @brief Single-token bf16 GEMV: y[n] = dot(W[n, :], x).
//
// @param W [N, K] bf16, row-major.
// @param x [K] bf16 activation.
// @param R Output rows per warp; the Python side benchmarks the (R, U) pairs once
//          per shape at load time.
// @param U K-loop unroll factor, same benchmarking reason.
// @return [N] bf16, in the activation dtype.
torch::Tensor gemv_bf16(torch::Tensor W, torch::Tensor x, int64_t R, int64_t U);

// @brief Same kernel family with fp16 weights and bf16 activations, returning bf16.
//
// gfx906 has packed fp16 math and no packed bf16 math, so the weights are fp16:
// the copy loop is bandwidth-bound either way, but the prefill GEMM that reads
// the same tensors runs ~7x faster (20 vs 2.3 TFLOP/s on wide-N shapes).
//
// @param W [N, K] fp16, row-major.
// @param x [K] bf16; the output stays bf16 so the activation stream does not
//          change dtype between layers.
// @param R Output rows per warp, same meaning as ``gemv_bf16``.
// @param U K-loop unroll factor.
// @return [N] bf16.
torch::Tensor gemv_f16(torch::Tensor W, torch::Tensor x, int64_t R, int64_t U);

// @brief Batched small-M GEMV: one warp per weight row keeps M accumulators.
//
// The row is read once for all M tokens, which is what makes k-token speculative
// verification cost about one weight pass instead of k.
//
// @param W [N, K] bf16, row-major.
// @param X [M, K] bf16.
// @param U K-loop unroll factor.
// @return [M, N] bf16.
torch::Tensor gemv_bf16_m(torch::Tensor W, torch::Tensor X, int64_t U);

// @brief Batched small-M GEMV with fp16 weights (bf16 activations in and out).
//
// Same reason as ``gemv_f16``: fp16 weights let the prefill GEMM reuse the same
// tensors without a per-layer repack.
//
// @param W [N, K] fp16, row-major.
// @param X [M, K] bf16.
// @param U K-loop unroll factor.
// @return [M, N] bf16.
torch::Tensor gemv_f16_m(torch::Tensor W, torch::Tensor X, int64_t U);

// @brief Weight-only int8 GEMV (W8A16) for decode.
//
// Reads half the weight bytes of ``gemv_bf16``, which is the only lever left once
// decode is bandwidth-bound; activations stay bf16 so no second quantiser (and its
// error) enters the path.
//
// @param W [N, K] int8, row-major.
// @param S [N] fp32 per-row dequant scale, applied once after the reduction.
// @param x [K] bf16.
// @param R Output rows per warp; the (R, U) knobs mean the same as ``gemv_bf16``.
// @param U K-loop unroll factor.
// @return [N] bf16.
torch::Tensor gemv_w8a16(torch::Tensor W, torch::Tensor S, torch::Tensor x,
                         int64_t R, int64_t U);

// @brief Batched small-M int8 GEMV: one weight pass for M activation vectors.
//
// Same shape contract as ``gemv_w8a16`` with the batch dim moved to the front.
//
// @param W [N, K] int8, row-major.
// @param S [N] fp32 per-row dequant scale.
// @param X [M, K] bf16; M must be one of the instantiated values {2, 4, 8}.
// @param U K-loop unroll factor.
// @return [M, N] bf16.
torch::Tensor gemv_w8a16_m(torch::Tensor W, torch::Tensor S, torch::Tensor X, int64_t U);

// @brief One gated-delta-rule decode step, all 24 linear-attention layers' maths.
//
// @param q,k [heads, d] fp32, already L2-normalised; q additionally scaled.
// @param v [heads, d] fp32.
// @param beta,g [heads] fp32 rate and decay, built by ``delta_prologue``.
// @param st [heads, dk, dv] fp32 recurrent state, updated in place -- the caller
//        owns the snapshot discipline that prefix reuse depends on.
// @param out [heads, dv] fp32, the same buffer returned.
// @return ``out``, [heads, dv] fp32.
torch::Tensor delta_step(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                         torch::Tensor beta, torch::Tensor g,
                         torch::Tensor st, torch::Tensor out);

// @brief Single-token full attention fused over the int8 KV cache.
//
// Dequantise k/v on the fly, fold the per-token scales into scores/probs,
// causal-mask at ``pos``, softmax and apply the output gate.
//
// @param q,gate [nh, hd] bf16; the gate's sigmoid is folded into the output.
// @param kcache,vcache [nkv, S, hd] int8 head-major; ``kscale``/``vscale`` are
//        [nkv, S] fp32.
// @param out [nh, hd] bf16, the same buffer returned.
// @param pos Device int64 tensor: the decode graph freezes scalar kernel
//        arguments at capture time, so the position has to be re-read at replay.
// @param rep Value heads per key head, i.e. the GQA group size.
// @param scale Softmax scale, already folded with 1/sqrt(head_dim) by the caller.
// @param part_acc,part_ml The online-softmax partials of the split pass,
//        [nh, split, hd] and [nh, split, 2] fp32. Caller-owned so the shapes stay
//        static for graph capture (``split = part_acc.size(1)``).
// @return ``out``, [nh, hd] bf16.
torch::Tensor attn_decode(torch::Tensor q, torch::Tensor gate,
                          torch::Tensor kcache, torch::Tensor vcache,
                          torch::Tensor kscale, torch::Tensor vscale,
                          torch::Tensor out, torch::Tensor pos,
                          int64_t rep, double scale,
                          torch::Tensor part_acc, torch::Tensor part_ml);

// @brief Same attention over a *paged* pool.
//
// @param q,gate [nh, hd] bf16.
// @param kpool,vpool One layer's view of the block pool, [num_slots, nkv, hd]
//        int8 with [num_slots, nkv] fp32 scales.
// @param pos2slot Device map from an absolute position to the pool slot holding
//        its row. It lives on the device so the captured decode graph survives a
//        block append: a host-side table would need a recapture per append.
// @param out [nh, hd] bf16, the same buffer returned.
// @param pos Device int64 position, same reason as ``attn_decode``.
// @param max_pos Unused now that the window is streamed; kept only so the
//        caller's signature does not change.
// @param rep Value heads per key head.
// @param scale Softmax scale.
// @param part_acc,part_ml Caller-owned split-pass partials, as in ``attn_decode``.
// @return ``out``, [nh, hd] bf16.
torch::Tensor attn_paged_decode(torch::Tensor q, torch::Tensor gate,
                                torch::Tensor kpool, torch::Tensor vpool,
                                torch::Tensor kscale, torch::Tensor vscale,
                                torch::Tensor pos2slot, torch::Tensor out,
                                torch::Tensor pos, int64_t max_pos,
                                int64_t rep, double scale,
                                torch::Tensor part_acc, torch::Tensor part_ml);

// @brief Fused RMSNorm: y = x * rsqrt(mean(x^2) + eps) * w.
//
// One block per row fuses the reduce, the rsqrt and the weight multiply, which is
// worth it because the engine calls it 105 times per decode step.
//
// @param x [M, N] bf16.
// @param w [N] bf16 weight.
// @param stride Row pitch of ``x``: q_norm runs on a slice, so rows are not
//        packed.
// @param eps Added to the mean square before the rsqrt; the checkpoint's epsilon,
//        not a tuning knob.
// @return [M, N] bf16.
torch::Tensor rms_norm(torch::Tensor x, torch::Tensor w, int64_t stride, double eps);

// @brief Gated RMSNorm: y = rms(x) * w * silu(gate).
//
// @param x,w,gate [M, N] bf16 contiguous; the gate is applied as SiLU after the
//        normalised weight multiply.
// @param eps Same epsilon contract as ``rms_norm``.
// @return [M, N] bf16.
torch::Tensor rms_norm_gated(torch::Tensor x, torch::Tensor w, torch::Tensor gate,
                             double eps);

// @brief Fused gated-delta-rule prologue for the decode step.
//
// L2-normalise q/k per head (q scaled by ``qscale``), expand key->value heads by
// ``rep``, convert v to fp32, and build beta = sigmoid(b) and g = -exp(A) *
// softplus(a + dt_bias). Doing this as torch ops is ~16 small kernels per layer.
//
// @param q_raw,k_raw [nk, dk] fp32, the row conv output.
// @param v_raw [nv, dv] fp32.
// @param b,a [nv] bf16 raw projection outputs, read only for the scalar factors.
// @param neg_expA,dt_bias [nv] fp32 checkpoint constants for the decay gate.
// @param q_out,k_out,v_out [nv, dk] / [nv, dv] fp32 written for ``delta_step``.
// @param beta,g [nv] fp32 rate and decay written for ``delta_step``.
// @param qscale Scale folded into the normalised q (softmax scale * 1/sqrt(d)).
// @param rep Value heads per key head.
// @return Nothing.
void delta_prologue(torch::Tensor q_raw, torch::Tensor k_raw, torch::Tensor v_raw,
                    torch::Tensor b, torch::Tensor a,
                    torch::Tensor neg_expA, torch::Tensor dt_bias,
                    torch::Tensor q_out, torch::Tensor k_out, torch::Tensor v_out,
                    torch::Tensor beta, torch::Tensor g,
                    double qscale, int64_t rep);

// @brief Fused depthwise causal conv1d (kernel 4) + SiLU for the linear-attn layers.
//
// @param conv_w [C, 4] fp32 taps.
// @param x [C] bf16 projection output.
// @param conv_state [C, 3] fp32 history, shifted in place.
// @param out [C] fp32; written through, nothing returned.
void conv1d_silu(torch::Tensor conv_w, torch::Tensor x, torch::Tensor conv_state,
                 torch::Tensor out);

// @brief Same conv over a whole prefill chunk.
//
// Reads the activation once and writes once, instead of the eager chain's six
// passes.
//
// @param conv_w [C, 4] fp32 taps, shared with ``conv1d_silu``.
// @param x [T, C] bf16.
// @param conv_state [C, 3] fp32; ends up holding the last three inputs, so the
//        next decode step continues the same recurrence.
// @return [T, C] fp32.
torch::Tensor conv1d_silu_seq(torch::Tensor conv_w, torch::Tensor x,
                              torch::Tensor conv_state);

// @brief Fused prefill head prep, writing the [H, T, d] layout the chunked scan reads.
//
// Doing it here rather than in torch removes both the ~1.7 ms/layer chain and the
// three strided transposes inside ``chunked_delta``.
//
// @param qkv [T, 2*num_k_heads*dk + num_v_heads*dv] fp32 conv output, laid out
//        q | k | v.
// @param b,a [T, nv] bf16 projection outputs; the row pitch is honoured because
//        they are column slices of the fused GEMV output.
// @param neg_expA,dt_bias [nv] fp32 checkpoint constants for the decay gate.
// @param q_out,k_out [nv, T, dk] fp32, ``v_out`` [nv, T, dv] fp32, ``beta``/``g``
//        [nv, T] fp32.
// @param qscale Scale folded into the normalised q.
// @param rep Value heads per key head.
// @return Nothing.
void delta_prep_seq(torch::Tensor qkv, torch::Tensor b, torch::Tensor a,
                    torch::Tensor neg_expA, torch::Tensor dt_bias,
                    torch::Tensor q_out, torch::Tensor k_out, torch::Tensor v_out,
                    torch::Tensor beta, torch::Tensor g,
                    double qscale, int64_t rep);

// Batched decode kernels (``batch.cu``).
//
// Same maths as the single-row kernels above, with a leading batch dim so a
// static batch of B sequences takes one launch per kernel. The single-row entry
// points stay the B = 1 / speculative-verify implementation; a row of width 1
// here reproduces them bit for bit.
//
// ``max_seq`` is the row width of ``slots`` (kept for signature compatibility
// with the contract); ``context_lens[b]`` bounds row b's key loop, so a batch
// mixes short and long windows without paying for the longest one.

// @brief Batched single-token attention over per-row paged int8 pool windows.
//
// @param slots [B, smax] int64; ``smax`` is its row stride, not ``max_seq``.
// @param context_lens [B] int32; the key loop of row b stops at its own window.
// @param part_acc,part_ml [B, nh, split, hd] / [B, nh, split, 2] fp32,
//        caller-owned for the same graph-capture reason as ``attn_decode``.
// @return ``out``, [B, nh, hd] bf16.
torch::Tensor attn_paged_batch(torch::Tensor q, torch::Tensor gate,
                               torch::Tensor kpool, torch::Tensor vpool,
                               torch::Tensor kscale, torch::Tensor vscale,
                               torch::Tensor slots, torch::Tensor context_lens,
                               torch::Tensor out, int64_t max_seq,
                               int64_t rep, double scale,
                               torch::Tensor part_acc, torch::Tensor part_ml);

// @brief Multi-query verify attention: one history pass serves c queries.
//
// Every query of a row sees the same history KV and the c queries of one row are
// adjacent blocks, so the history is read once from HBM (the other passes hit
// L2). Each query keeps the exact split-K / online-softmax / gate arithmetic of
// ``attn_paged_batch``, so it is bit-identical to decoding that token on its own.
//
// @param q,gate,out [B, c, nh, hd] bf16.
// @param ctx_lens [B] int32 history length; query j sees ctx+j+1 keys.
// @param part_acc,part_ml [B*c, nh, split, hd] / [..., 2] fp32 caller-owned.
// @param block_mask Optional uint8 [B, n_blocks] keep-mask (empty = exact).
// @param mask_block Tokens per mask block, >= 1.
// @return ``out``, [B, c, nh, hd] bf16.
torch::Tensor attn_paged_multi_q(torch::Tensor q, torch::Tensor gate,
                                 torch::Tensor kpool, torch::Tensor vpool,
                                 torch::Tensor kscale, torch::Tensor vscale,
                                 torch::Tensor slots, torch::Tensor ctx_lens,
                                 torch::Tensor out, int64_t max_seq,
                                 int64_t rep, double scale,
                                 torch::Tensor part_acc, torch::Tensor part_ml,
                                 torch::Tensor block_mask, int64_t mask_block);

// @brief Batched depthwise causal conv1d + SiLU.
//
// @param x [B, C] bf16.
// @param conv_state [B, C, 3] fp32, shifted in place.
// @return ``out``, [B, C] fp32.
torch::Tensor conv1d_silu_batch(torch::Tensor conv_w, torch::Tensor x,
                                torch::Tensor conv_state, torch::Tensor out);

// @brief Batched gated-delta prologue; outputs as ``delta_prologue`` plus a batch dim.
//
// @param q,k [B, nk, dk] fp32, ``v`` [B, nv, dv] fp32. Strided rows are allowed
//        because they are column slices of the conv output.
// @param b,a [B, nv] bf16.
// @return Nothing.
void delta_prologue_batch(torch::Tensor q_raw, torch::Tensor k_raw, torch::Tensor v_raw,
                          torch::Tensor b, torch::Tensor a,
                          torch::Tensor neg_expA, torch::Tensor dt_bias,
                          torch::Tensor q_out, torch::Tensor k_out, torch::Tensor v_out,
                          torch::Tensor beta, torch::Tensor g,
                          double qscale, int64_t rep);

// @brief Batched gated-delta update; rows are independent.
//
// @param st [B, nv, dk, dv] fp32, updated in place.
// @return ``out``, [B, nv, dv] fp32.
torch::Tensor delta_batch(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                          torch::Tensor beta, torch::Tensor g,
                          torch::Tensor st, torch::Tensor out);

// Wide-M int8 GEMV for speculative verify (``gemv_wide.cu``).

// @brief One weight pass for M in {16, 24, 32} activation vectors, no hipBLAS.
//
// The weight row is read once and applied to all M activations in fp32. The M
// values are split into groups of eight (one warp per group), and each (row,
// activation) pair keeps the same eight FMA chains and K order as
// ``gemv_w8a16_m``, so a row is bit-identical to calling the M=8 kernel on that
// row's activation slice.
//
// @param U K unroll (2 or 4).
// @param threads Block size, a multiple of 32; the warps must divide evenly into
//        row groups or the last group is not covered.
// @return [M, N] bf16.
torch::Tensor gemv_w8a16_wide(torch::Tensor W, torch::Tensor S, torch::Tensor X,
                              int64_t U, int64_t threads);

// @brief A/B entry for ``gemv_w8a16_wide`` with the tiling exposed.
//
// @param rows Output rows per warp.
// @param mw Activation vectors per warp; all combos keep 64 accumulator floats.
// @param threads Block size; 0 picks an even warp split.
// @return [M, N] bf16, identical arithmetic for any tiling.
torch::Tensor gemv_w8a16_wide_dbg(torch::Tensor W, torch::Tensor S, torch::Tensor X,
                                  int64_t U, int64_t rows, int64_t mw,
                                  int64_t threads);

// @brief Wide-M GEMV with packed fp16 math (``v_dot2_f32_f16``, two MACs per
//        instruction, fp32 accumulation).
//
// int8 weights convert to fp16 exactly; bf16 activations are converted to fp16
// once per call (exact for |x| <= 65504, clamped above that). Same M/rows/mw
// tiling knobs as the int8 kernel; the result is *close* to it (fp32 sum,
// different rounding) rather than bit-identical, which the verify path accepts in
// exchange for the halved instruction count.
//
// @return [M, N] bf16.
torch::Tensor gemv_w8a16_wide_dot2(torch::Tensor W, torch::Tensor S, torch::Tensor X,
                                   int64_t U, int64_t threads);
// @brief A/B entry for ``gemv_w8a16_wide_dot2`` with the tiling exposed.
//
// @return [M, N] bf16; the tiling affects only speed.
torch::Tensor gemv_w8a16_wide_dot2_dbg(torch::Tensor W, torch::Tensor S,
                                       torch::Tensor X, int64_t U, int64_t rows,
                                       int64_t mw, int64_t threads);
