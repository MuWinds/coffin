#pragma once

#include <torch/extension.h>

// y[n] = dot(W[n, :], x) for W [N, K] bf16 and x [K] bf16.
// R = output rows per warp, U = K-loop unroll factor; the Python side benchmarks
// the (R, U) pairs once per shape at load time.
torch::Tensor gemv_bf16(torch::Tensor W, torch::Tensor x, int64_t R, int64_t U);

// Same kernel family with fp16 weights and bf16 activations, returning bf16.
// gfx906 has packed fp16 math and no packed bf16 math, so the weights are fp16:
// the copy loop is bandwidth-bound either way, but the prefill GEMM that reads
// the same tensors runs ~7x faster (20 vs 2.3 TFLOP/s on wide-N shapes).
torch::Tensor gemv_f16(torch::Tensor W, torch::Tensor x, int64_t R, int64_t U);

// Batched small-M GEMV: ``X`` is [M, K], returns [M, N]. One warp per weight
// row keeps M accumulators, so the row is read once for all M tokens — this is
// what makes k-token speculative verification cost about one weight pass.
torch::Tensor gemv_bf16_m(torch::Tensor W, torch::Tensor X, int64_t U);

// Batched small-M fp16-weight GEMV (bf16 activations in and out).
torch::Tensor gemv_f16_m(torch::Tensor W, torch::Tensor X, int64_t U);

// Weight-only int8 GEMV (W8A16): ``W`` is [N, K] int8 with per-row fp32 scale
// ``S`` [N], ``x`` is [K] bf16 -> ``y`` [N] bf16. Reads half the weight bytes of
// ``gemv_bf16``; the (R, U) knobs mean the same thing.
torch::Tensor gemv_w8a16(torch::Tensor W, torch::Tensor S, torch::Tensor x,
                         int64_t R, int64_t U);

// Batched small-M int8 GEMV: ``X`` is [M, K] bf16, returns [M, N] bf16.
torch::Tensor gemv_w8a16_m(torch::Tensor W, torch::Tensor S, torch::Tensor X, int64_t U);

// One gated-delta-rule decode step. q, k, v are [heads, d] (q, k already
// L2-normalised, q additionally scaled), beta and g are [heads], st is the
// [heads, dk, dv] recurrent state updated in place, out is [heads, dv].
torch::Tensor delta_step(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                         torch::Tensor beta, torch::Tensor g,
                         torch::Tensor st, torch::Tensor out);

// Single-token full attention fused over the int8 KV cache: dequantise k/v on
// the fly, fold the per-token scales into scores/probs, causal-mask at ``pos``,
// softmax and apply the output gate. q/gate are [nh, hd] bf16, the caches are
// [nkv, S, hd] int8 with [nkv, S] float scales, out is [nh, hd] bf16. ``pos`` is
// a device int64 tensor (the decode graph freezes scalar kernel arguments).
//
// ``part_acc``/``part_ml`` are the online-softmax partials of the split pass:
// [nh, split, hd] and [nh, split, 2] fp32, caller-owned so the shapes stay static
// for graph capture (``split = part_acc.size(1)``).
torch::Tensor attn_decode(torch::Tensor q, torch::Tensor gate,
                          torch::Tensor kcache, torch::Tensor vcache,
                          torch::Tensor kscale, torch::Tensor vscale,
                          torch::Tensor out, torch::Tensor pos,
                          int64_t rep, double scale,
                          torch::Tensor part_acc, torch::Tensor part_ml);

// Same attention over a *paged* pool: ``kpool``/``vpool`` are one layer's view of
// the block pool, ``[num_slots, nkv, hd]`` int8 with ``[num_slots, nkv]`` float
// scales, and ``pos2slot`` maps an absolute position to the pool slot holding its
// row (device-resident, so the captured decode graph survives block appends).
// ``max_pos`` is unused now that the window is streamed, and is kept only so the
// caller's signature does not change.
torch::Tensor attn_paged_decode(torch::Tensor q, torch::Tensor gate,
                                torch::Tensor kpool, torch::Tensor vpool,
                                torch::Tensor kscale, torch::Tensor vscale,
                                torch::Tensor pos2slot, torch::Tensor out,
                                torch::Tensor pos, int64_t max_pos,
                                int64_t rep, double scale,
                                torch::Tensor part_acc, torch::Tensor part_ml);

// Fused RMSNorm: y = x * rsqrt(mean(x^2)+eps) * w, over [M, N] rows with a
// per-row ``stride`` (q_norm runs on a slice, so rows are not packed).
torch::Tensor rms_norm(torch::Tensor x, torch::Tensor w, int64_t stride, double eps);

// Gated RMSNorm: y = rms(x) * w * silu(gate), all [M, N] contiguous.
torch::Tensor rms_norm_gated(torch::Tensor x, torch::Tensor w, torch::Tensor gate,
                             double eps);

// Fused gated-delta-rule prologue: L2-normalise q/k per head (q scaled by
// ``qscale``), expand key->value heads by ``rep``, convert v to fp32, and build
// beta = sigmoid(b) and g = -exp(A) * softplus(a + dt_bias). Writes the fp32
// buffers consumed by ``delta_step``.
void delta_prologue(torch::Tensor q_raw, torch::Tensor k_raw, torch::Tensor v_raw,
                    torch::Tensor b, torch::Tensor a,
                    torch::Tensor neg_expA, torch::Tensor dt_bias,
                    torch::Tensor q_out, torch::Tensor k_out, torch::Tensor v_out,
                    torch::Tensor beta, torch::Tensor g,
                    double qscale, int64_t rep);

// Fused depthwise causal conv1d (kernel 4) + SiLU for the linear-attn layers.
// ``conv_w`` is [C, 4] fp32, ``x`` is the [C] bf16 projection output,
// ``conv_state`` is the [C, 3] fp32 history shifted in place, ``out`` is [C] fp32.
void conv1d_silu(torch::Tensor conv_w, torch::Tensor x, torch::Tensor conv_state,
                 torch::Tensor out);

// Same conv over a whole prefill chunk: ``x`` is [T, C] bf16, the result is
// [T, C] fp32 and ``conv_state`` ends up holding the last three inputs.  Reads
// the activation once and writes once, instead of the eager chain's six passes.
torch::Tensor conv1d_silu_seq(torch::Tensor conv_w, torch::Tensor x,
                              torch::Tensor conv_state);

// Fused prefill head prep: L2-normalise q/k per key head, expand key->value
// heads, scale q, carry v, and build beta/g -- into the [H, T, d] layout the
// chunked scan reads, so nothing downstream transposes.  ``qkv`` is the
// [T, 2*num_k_heads*dk + num_v_heads*dv] fp32 conv output; ``b``/``a`` are the
// bf16 projection outputs [T, nv] (row pitch honoured: they are column slices of
// the fused GEMV output).  ``q_out``/``k_out`` are [nv, T, dk], ``v_out`` is
// [nv, T, dv], ``beta``/``g`` are [nv, T].
void delta_prep_seq(torch::Tensor qkv, torch::Tensor b, torch::Tensor a,
                    torch::Tensor neg_expA, torch::Tensor dt_bias,
                    torch::Tensor q_out, torch::Tensor k_out, torch::Tensor v_out,
                    torch::Tensor beta, torch::Tensor g,
                    double qscale, int64_t rep);

// Batched decode kernels (``batch.cu``).
//
// Same maths as the single-row kernels above, with a leading batch dim so a
// static batch of B sequences takes one launch per kernel.  The single-row
// entry points stay the B = 1 / speculative-verify implementation; a row of
// width 1 here reproduces them bit for bit.
//
// ``max_seq`` is the row width of ``slots`` (kept for signature compatibility
// with the contract); ``context_lens[b]`` bounds row b's key loop, so a batch
// mixes short and long windows without paying for the longest one.

torch::Tensor attn_paged_batch(torch::Tensor q, torch::Tensor gate,
                               torch::Tensor kpool, torch::Tensor vpool,
                               torch::Tensor kscale, torch::Tensor vscale,
                               torch::Tensor slots, torch::Tensor context_lens,
                               torch::Tensor out, int64_t max_seq,
                               int64_t rep, double scale,
                               torch::Tensor part_acc, torch::Tensor part_ml);

// Multi-query verify attention: every query of a row sees the same history KV and
// the c queries of one row are adjacent blocks, so the history is read once from
// HBM (the other passes hit L2).  Each query keeps the exact split-K /
// online-softmax / gate arithmetic of ``attn_paged_batch``, so it is
// bit-identical to decoding that token on its own.  ``q``/``gate``/``out`` are
// [B, c, nh, hd]; ``ctx_lens`` [B] int32 is the history length (query j sees
// ctx+j+1 keys); ``part_acc``/``part_ml`` are [B*c, nh, split, hd] / [..., 2].
// ``block_mask`` is an optional uint8 [B, n_blocks] keep-mask (empty = exact);
// ``mask_block`` is the mask's tokens-per-block.
torch::Tensor attn_paged_multi_q(torch::Tensor q, torch::Tensor gate,
                                 torch::Tensor kpool, torch::Tensor vpool,
                                 torch::Tensor kscale, torch::Tensor vscale,
                                 torch::Tensor slots, torch::Tensor ctx_lens,
                                 torch::Tensor out, int64_t max_seq,
                                 int64_t rep, double scale,
                                 torch::Tensor part_acc, torch::Tensor part_ml,
                                 torch::Tensor block_mask, int64_t mask_block);

// Batched depthwise causal conv1d + SiLU: ``x`` [B, C] bf16,
// ``conv_state`` [B, C, 3] fp32 (shifted in place), ``out`` [B, C] fp32.
torch::Tensor conv1d_silu_batch(torch::Tensor conv_w, torch::Tensor x,
                                torch::Tensor conv_state, torch::Tensor out);

// Batched delta prologue: ``q``/``k`` [B, nk, dk], ``v`` [B, nv, dv] (strided
// rows allowed -- they are column slices of the conv output), ``b``/``a``
// [B, nv] bf16; outputs as ``delta_prologue`` with a batch dim.
void delta_prologue_batch(torch::Tensor q_raw, torch::Tensor k_raw, torch::Tensor v_raw,
                          torch::Tensor b, torch::Tensor a,
                          torch::Tensor neg_expA, torch::Tensor dt_bias,
                          torch::Tensor q_out, torch::Tensor k_out, torch::Tensor v_out,
                          torch::Tensor beta, torch::Tensor g,
                          double qscale, int64_t rep);

// Batched gated-delta update: ``state`` is [B, nv, dk, dv] fp32, updated in
// place; returns ``out`` [B, nv, dv].  Rows are independent.
torch::Tensor delta_batch(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                          torch::Tensor beta, torch::Tensor g,
                          torch::Tensor st, torch::Tensor out);

// Wide-M int8 GEMV for speculative verify (``gemv_wide.cu``).
//
// One weight pass for M in {16, 24, 32} activation vectors, no dequantisation,
// no hipBLAS: the weight row is read once and applied to all M activations in
// fp32.  The M values are split into groups of eight (one warp per group), and
// each (row, activation) pair keeps the same eight FMA chains and K order as
// ``gemv_w8a16_m``, so a row is bit-identical to calling the M=8 kernel on that
// row's activation slice.  ``U`` is the K unroll (2 or 4) and ``threads`` the
// block size (a multiple of 32; the warps must divide evenly into row groups).
torch::Tensor gemv_w8a16_wide(torch::Tensor W, torch::Tensor S, torch::Tensor X,
                              int64_t U, int64_t threads);

// A/B entry: ``rows`` output rows and ``mw`` activation vectors per warp
// (all combos keep 64 accumulator floats; threads=0 picks an even warp split).
torch::Tensor gemv_w8a16_wide_dbg(torch::Tensor W, torch::Tensor S, torch::Tensor X,
                                  int64_t U, int64_t rows, int64_t mw,
                                  int64_t threads);

// Wide-M GEMV with packed fp16 math (``v_dot2_f32_f16``, two MACs per
// instruction, fp32 accumulation).  int8 weights convert to fp16 exactly;
// bf16 activations are converted to fp16 once per call (exact for |x| <= 65504,
// clamped above that).  Same M/rows/mw/tiling knobs; the result is *close* to the
// int8 kernel (fp32 sum, different rounding) rather than bit-identical.
torch::Tensor gemv_w8a16_wide_dot2(torch::Tensor W, torch::Tensor S, torch::Tensor X,
                                   int64_t U, int64_t threads);
torch::Tensor gemv_w8a16_wide_dot2_dbg(torch::Tensor W, torch::Tensor S,
                                       torch::Tensor X, int64_t U, int64_t rows,
                                       int64_t mw, int64_t threads);
