// @file corex_ops.h
// @brief Declarations of the CoreX extension entry points bound in ``bindings.cpp``.
//
// Module: coffin/backends/corex/kernels. The only header the kernels and the binding
// share; it is a pure interface, so it must stay free of device code -- ``bindings.cpp``
// is compiled by the host compiler, which has no ``__device__``. The bf16 conversions
// live in ``corex_scalar.h`` for that reason.
//
// Constraints:
//   * Declarations here are binding names: ``backends/corex/kernels/__init__.py``
//     looks them up by string. A rename has to land on both sides in one change or the
//     extension imports with a missing symbol.
//   * Every wrapper takes the shape it indexes and TORCH_CHECKs the layout it assumes
//     (dtype, contiguity, 2-D-ness). The kernels index without bounds checks, so a
//     wrong layout must fail on the host rather than read past the buffer.
//   * Weights are ``[K, N]`` throughout -- the layout ``Caps.weight_layout == "kn"``
//     promises. A ``[N, K]`` tensor handed to these entry points produces plausible
//     numbers from the wrong elements, so the Python side owns the transpose and the
//     wrapper only checks the dtype.
//   * Every wrapper launches on the current torch stream. A kernel on the legacy
//     default stream is not part of a ``torch.cuda.graph`` capture, so it would be
//     dropped from the decode graph and replay stale output.

#pragma once

#include <torch/extension.h>

#include <cstdint>

// Projections: y[m, n] = sum_k x[m, k] * wt[k, n].

// @brief Decode-time GEMV over one activation vector, one thread per output column.
//
// @param wt [K, N] weights, contiguous, bf16, fp16 or fp32.
// @param x [K] activations, bf16 or fp32. The output takes its dtype.
// @param threads Threads per block; ``0`` picks from ``N``.
// @param unroll K-loop unroll factor, ``2`` or ``4``; ``0`` means 4.
// @param split Split-K width; ``0`` picks from ``N`` and ``K``, ``1`` forces the
//        single-pass kernel.
// @return [N] in ``x``'s dtype.
// @throws c10::Error on a shape, dtype or layout the kernels cannot index.
torch::Tensor gemv_kn(torch::Tensor wt, torch::Tensor x, int64_t threads,
                      int64_t unroll, int64_t split);

// @brief Small-M batched GEMV: one weight read produces ``M`` activation rows.
//
// @param wt [K, N] weights, contiguous, bf16, fp16 or fp32.
// @param x [M, K] activations, ``M`` in ``[1, 8]``.
// @param threads Threads per block; ``0`` picks from ``N``.
// @param unroll K-loop unroll factor; ``0`` means 4.
// @return [M, N] in ``x``'s dtype.
// @throws c10::Error on a shape, dtype or layout the kernels cannot index.
torch::Tensor gemv_kn_m(torch::Tensor wt, torch::Tensor x, int64_t threads,
                        int64_t unroll);

// @brief Decode W8A16 projection: int8 weights plus one fp32 scale per output column.
//
// Halves the bytes read per weight against :func:`gemv_kn`. The scale applies once to the
// finished dot product, never per element and never folded into the activation.
//
// @param wt [K, N] int8 weights, contiguous.
// @param x [K] activations, bf16 or fp32.
// @param scale [N] fp32 absmax scale, one per output column.
// @param threads Threads per block; ``0`` picks from ``N``.
// @param unroll K-loop unroll factor; ``0`` means 4.
// @param split Split-K width; ``0`` picks, ``1`` forces the single-pass kernel.
// @param width Adjacent output columns one thread owns, which is also the number of
//        contiguous weight bytes it loads per k step: ``2`` is one 16-bit load, ``4`` one
//        32-bit, ``8`` one 64-bit. Legal values are ``1``, ``2``, ``4``, ``8`` and ``16``;
//        any other non-zero value is rejected rather than rounded, and ``N % width`` must
//        be ``0`` (the launch falls back to ``1`` when it is not, so a weight whose width
//        is not a multiple of ``width`` stays correct instead of reading across a row
//        boundary). ``0`` means auto, which reproduces the dispatch that predates this
//        argument exactly: four-wide above a column floor, scalar below. Reading wider
//        divides the thread count by ``width``, so the best value is a measured trade-off
//        per shape rather than a constant -- see ``diag/tune_gemv_i8_wide.py``.
// @return [N] in ``x``'s dtype.
// @throws c10::Error on a non-int8 weight, a scale length other than ``N``, an unsupported
//         ``width``, or a layout the kernels cannot index.
torch::Tensor gemv_kn_i8(torch::Tensor wt, torch::Tensor x, torch::Tensor scale,
                         int64_t threads, int64_t unroll, int64_t split,
                         int64_t width);

// Normalisation.

// @brief Zero-centred RMSNorm over the last axis of ``[M, N]`` with a row pitch.
//
// @param x [M, N] activations with unit innermost stride; the rows are pitched
//        ``stride`` elements apart, which is what lets a column slice normalise in
//        place.
// @param w [N] gain, already carrying the checkpoint's ``1 + w`` offset.
// @param stride Element pitch between rows; ``0`` means ``N``.
// @param eps Variance floor, added inside the rsqrt.
// @return [M, N], densely packed, in ``x``'s dtype.
// @throws c10::Error on a dtype the kernel has no instantiation for, or a pitch
//         narrower than the row.
torch::Tensor rms_norm(torch::Tensor x, torch::Tensor w, int64_t stride, double eps);

// @brief Residual add followed by the zero-centred RMSNorm, in one launch.
//
// Every decoder layer does ``x = x + h`` and then normalises ``x``. As two ops that is two
// launches and reads the residual buffer twice; fused it is one launch and one pass, and
// the sum has a single home. A separate entry rather than an optional argument on
// :func:`rms_norm` so a backend without it falls back to the reference implementation --
// which is correct, only slower -- instead of silently dropping the add.
//
// @param x [M, N] residual stream, accumulated in place with ``add``.
// @param add [M, N] mixer output to accumulate.
// @param w [N] gain, already carrying the ``1 + w`` offset.
// @param stride Element pitch between rows of ``x``; ``0`` means ``N``.
// @param eps Variance floor.
// @param out [M, N] normalised result, written in place. It must not alias ``x``: the
//        residual sum stays live for the rest of the layer.
// @return ``out``, for call chaining.
// @throws c10::Error on a shape, dtype or aliasing the kernel cannot serve.
torch::Tensor add_rms_norm(torch::Tensor x, torch::Tensor add, torch::Tensor w,
                           torch::Tensor out, int64_t stride, double eps);

// @brief RMSNorm scaled by the SiLU of a gate, normalised per head.
//
// @param x [M, N] activations, contiguous, bf16 or fp32.
// @param w [N] per-head gain.
// @param gate [M, N] gate in the same layout and dtype as ``x``.
// @param eps Variance floor.
// @return [M, N] in ``x``'s dtype.
// @throws c10::Error on a dtype the kernel has no instantiation for, or a gate whose
//         shape differs from ``x``.
torch::Tensor rms_norm_gated(torch::Tensor x, torch::Tensor w, torch::Tensor gate,
                             double eps);

// @brief Symmetric int8 quantisation of ``[M, N]``, one scale per row, in one launch.
//
// Replaces the torch chain ``abs -> amax -> clamp_min -> div -> round -> clamp_ -> to``,
// which is six launches per call and is called twice per full-attention layer (k and v).
// At this card's launch floor the six cost about 82 us for a few microseconds of
// arithmetic; one launch costs about six.
//
// @param x [M, N] bf16 activation to quantise.
// @param out_q [M, N] int8 destination, written in place.
// @param out_s [M] fp32 destination for the per-row scale, written in place.
// @return ``out_q``, for call chaining.
// @throws c10::Error when a dtype or shape disagrees with the destinations.
torch::Tensor kv_quant(torch::Tensor x, torch::Tensor out_q, torch::Tensor out_s);

// @brief ``out = silu(gate) * up``, the SwiGLU product, in one launch.
//
// The fused ``gate_up`` projection returns both halves from one weight read, so the only
// work between it and the down projection is this product. As ``aten::silu`` +
// ``aten::mul`` it is two launches and two ``[intermediate]`` temporaries per layer, which
// at this card's launch floor costs more than the arithmetic.
//
// @param gate [M] or [M, N] gate half, contiguous.
// @param up Up half, same shape as ``gate``.
// @param out Destination of the same shape. It may alias ``up``: the product overwrites
//        it and nothing reads ``up`` afterwards, which is what lets a caller avoid both
//        the allocation and the extra read.
// @return ``out``, for call chaining.
// @throws c10::Error on a shape, dtype or overlap the kernel cannot serve.
torch::Tensor silu_mul(torch::Tensor gate, torch::Tensor up, torch::Tensor out);

// @brief Rotate the leading ``rotary_dim`` channels of every q/k head, in one launch.
//
// Replaces the torch chain ``to -> mul -> cat -> mul -> add -> cat`` that ``apply_rope``
// runs per layer: about ten launches on ``[16, 256]`` and ``[4, 256]`` tensors, where the
// arithmetic is a few thousand FMAs. At this card's launch floor that chain costs nearly a
// millisecond per step and almost none of it is work.
//
// One shape serves both paths: ``M == 1`` is decode, ``M == T`` is a prefill chunk or a
// verify window, so a single kernel keeps them numerically identical.
//
// @param q [M, nh, hd] bf16 queries.
// @param k [M, nkv, hd] bf16 keys.
// @param cos [M, rotary_dim] fp32, already gathered for the rows being rotated.
// @param sin Same layout as ``cos``.
// @param q_out Destination of the rotated queries, ``q``'s shape. Must not alias ``q``.
// @param k_out Destination of the rotated keys, ``k``'s shape. Must not alias ``k``.
// @param rotary_dim Channels to rotate; the trailing ``hd - rotary_dim`` pass through
//        untouched, which is what partial-rotary checkpoints require.
// @return ``q_out``, for call chaining.
// @throws c10::Error on a shape, dtype or aliasing the kernel cannot serve.
torch::Tensor rope_apply(torch::Tensor q, torch::Tensor k, torch::Tensor cos,
                         torch::Tensor sin, torch::Tensor q_out, torch::Tensor k_out,
                         int64_t rotary_dim);

// Attention over the int8 KV cache: dense, paged, batched and draft-verify.
//
// ``part_acc``/``part_ml`` carry the split-K width on their own shape, so a caller that
// sizes them from ``Caps.attn_split`` cannot disagree with the kernels about the ABI.

// @brief One decode token over the dense int8 KV cache.
//
// @param q [nh, hd] bf16 query, already rotated and scaled.
// @param gate [nh, hd] bf16 output gate; sigmoid is applied after normalisation.
// @param kcache,vcache [nkv, S, hd] int8, head-major.
// @param kscale,vscale [nkv, S] fp32 per-token scales.
// @param out [nh, hd] bf16, written in place.
// @param pos 1-element int64 tensor: the token index and the causal bound, read on the
//        device so a captured replay sees the live position.
// @param rep Query heads per KV head.
// @param scale ``hd ** -0.5``.
// @param part_acc [nh, split, hd] fp32 split-K numerator.
// @param part_ml [nh, split, 2] fp32 split-K (max, sum).
// @return ``out``, for call chaining.
torch::Tensor attn_decode(torch::Tensor q, torch::Tensor gate, torch::Tensor kcache,
                          torch::Tensor vcache, torch::Tensor kscale,
                          torch::Tensor vscale, torch::Tensor out, torch::Tensor pos,
                          int64_t rep, double scale, torch::Tensor part_acc,
                          torch::Tensor part_ml);

// @brief One decode token over a layer's paged int8 pool.
//
// @param kpool,vpool [nslots, nkv, hd] int8, token-major.
// @param kscale,vscale [nslots, nkv] fp32 per-slot scales.
// @param pos2slot [max_pos] int64 device map; only the first ``pos + 1`` entries are read.
// @param max_pos Window length, i.e. the row stride of the slot map.
// @return ``out``, for call chaining.
torch::Tensor attn_paged(torch::Tensor q, torch::Tensor gate, torch::Tensor kpool,
                         torch::Tensor vpool, torch::Tensor kscale,
                         torch::Tensor vscale, torch::Tensor pos2slot, torch::Tensor out,
                         torch::Tensor pos, int64_t max_pos, int64_t rep, double scale,
                         torch::Tensor part_acc, torch::Tensor part_ml);

// @brief One decode token for a bucket of rows, each with its own context length.
//
// @param q [B, nh, hd] bf16 queries.
// @param slots [B, max_seq] int64 per-row slot maps.
// @param context_lens [B] int32 per-row key counts.
// @return ``out`` [B, nh, hd], for call chaining.
torch::Tensor attn_paged_batch(torch::Tensor q, torch::Tensor gate, torch::Tensor kpool,
                               torch::Tensor vpool, torch::Tensor kscale,
                               torch::Tensor vscale, torch::Tensor slots,
                               torch::Tensor context_lens, torch::Tensor out,
                               int64_t max_seq, int64_t rep, double scale,
                               torch::Tensor part_acc, torch::Tensor part_ml);

// @brief Speculative verify: ``c`` draft queries per row share one history pass.
//
// @param q [B, c, nh, hd] bf16 draft queries.
// @param ctx_lens [B] int32 per-row history length, excluding the drafts.
// @param block_mask uint8 ``[B, n_blocks]`` keep mask; an empty tensor means exact
//        attention.
// @return ``out`` [B, c, nh, hd], for call chaining.
torch::Tensor attn_paged_multi_q(torch::Tensor q, torch::Tensor gate, torch::Tensor kpool,
                                 torch::Tensor vpool, torch::Tensor kscale,
                                 torch::Tensor vscale, torch::Tensor slots,
                                 torch::Tensor ctx_lens, torch::Tensor out,
                                 int64_t max_seq, int64_t rep, double scale,
                                 torch::Tensor part_acc, torch::Tensor part_ml,
                                 torch::Tensor block_mask, int64_t mask_block);

// Gated DeltaNet: the depthwise conv that feeds the recurrence, the head prologue, and
// the recurrence itself.

// @brief Depthwise causal conv (4 taps) + SiLU for one token, shifting the window.
//
// @param conv_w [C, 4] fp32 taps.
// @param x [C] bf16 projection output.
// @param conv_state [C, 3] fp32 history, shifted in place.
// @param out [C] fp32, written in place.
void conv1d_silu(torch::Tensor conv_w, torch::Tensor x, torch::Tensor conv_state,
                 torch::Tensor out);

// @brief The same conv over a prefill chunk, carrying the window across chunks.
//
// @param x [T, C] bf16 with unit inner stride; the row pitch is honoured.
// @param conv_state [C, 3] fp32, updated in place.
// @return [T, C] fp32.
torch::Tensor conv1d_silu_seq(torch::Tensor conv_w, torch::Tensor x,
                              torch::Tensor conv_state);

// @brief Batched depthwise causal conv + SiLU, one row per batch entry.
//
// @param x [B, C] bf16.
// @param conv_state [B, C, 3] fp32, shifted in place.
// @param out [B, C] fp32, written in place.
void conv1d_silu_batch(torch::Tensor conv_w, torch::Tensor x, torch::Tensor conv_state,
                       torch::Tensor out);

// @brief q/k L2 norm, head expansion and the beta/g gates for one decode step.
//
// @param q,k [nkeys, dk] fp32; ``v`` [nv, dv] fp32.
// @param b,a [nv] bf16 raw logits.
// @param neg_expA,dt_bias [nv] fp32 checkpoint constants.
// @param q_out,k_out [nv, dk] fp32 and ``v_out`` [nv, dv] fp32, written in place.
// @param beta,g [nv] fp32, written in place.
// @param qscale ``dk ** -0.5``.
// @param rep Value heads per key head.
void delta_prologue(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor b,
                    torch::Tensor a, torch::Tensor neg_expA, torch::Tensor dt_bias,
                    torch::Tensor q_out, torch::Tensor k_out, torch::Tensor v_out,
                    torch::Tensor beta, torch::Tensor g, double qscale, int64_t rep);

// @brief Batched prologue: a leading row dim on every argument.
void delta_prologue_batch(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                          torch::Tensor b, torch::Tensor a, torch::Tensor neg_expA,
                          torch::Tensor dt_bias, torch::Tensor q_out,
                          torch::Tensor k_out, torch::Tensor v_out, torch::Tensor beta,
                          torch::Tensor g, double qscale, int64_t rep);

// @brief Prefill head prep into the head-major layout the chunked scan reads.
//
// @param qkv [T, conv_dim] fp32 conv output laid out q | k | v.
// @param q_out,k_out [nv, T, dk] fp32 and ``v_out`` [nv, T, dv] fp32.
// @param beta,g [nv, T] fp32.
void delta_prep_seq(torch::Tensor qkv, torch::Tensor b, torch::Tensor a,
                    torch::Tensor neg_expA, torch::Tensor dt_bias, torch::Tensor q_out,
                    torch::Tensor k_out, torch::Tensor v_out, torch::Tensor beta,
                    torch::Tensor g, double qscale, int64_t rep);

// @brief One gated-delta-rule decode step, advancing the state in place.
//
// @param q,k [heads, dk] fp32, already normalised by the prologue.
// @param v [heads, dv] fp32.
// @param beta,g [heads] fp32; ``g <= 0`` or the recurrence grows.
// @param st [heads, dk, dv] fp32, updated in place.
// @param out [heads, dv] fp32, read from the updated state.
// @return ``out``, for call chaining.
torch::Tensor delta(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor beta,
                    torch::Tensor g, torch::Tensor st, torch::Tensor out);

// @brief Batched delta step: one independent state row per batch entry.
//
// @return ``out`` [B, heads, dv], for call chaining.
torch::Tensor delta_batch(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                          torch::Tensor beta, torch::Tensor g, torch::Tensor st,
                          torch::Tensor out);

// @brief Prologue, recurrence and gated norm for one decode token, in one launch.
//
// Replaces the ``conv1d_silu`` -> ``delta_prologue`` -> ``delta`` -> ``rms_norm_gated``
// chain. Those four launches write and re-read about 48 KB of fp32 scratch per layer per
// token (``q_pro``/``k_pro``/``v_pro`` = 3 x [32, 128], ``delta_out`` = [32, 128]), and at
// this card's ~5.75 us launch floor the round trips cost more than the arithmetic. One
// block per value head keeps every intermediate in shared memory and registers, and the
// gated norm's reduction is over exactly one head's ``dv``, so it folds into the epilogue
// for free.
//
// @param conv ``[nv, dv]`` fp32, the conv-and-SiLU output for this token. The q and k
//        thirds of that buffer are read by column, at ``dk`` per head, so the layout is
//        q | k | v with ``nv == nkeys * rep``; heads repeat their key head ``rep`` times.
//        Passed in three slices by the caller rather than as one row, because the conv
//        output is ``conv_dim`` wide and only its head column of each third is read here.
// @param b,a ``[nv]`` bf16 raw beta/decay logits.
// @param neg_expA,dt_bias ``[nv]`` fp32 checkpoint constants for the decay gate.
// @param z ``[nv, dv]`` fp32-or-bf16 gate projection for the SiLU of the output norm.
// @param nw ``[dv]`` bf16 gated-norm gain, per value head rather than per flattened width.
// @param st ``[nv, dk, dv]`` fp32 recurrent state, updated in place.
// @param out ``[nv, dv]`` fp32, written from the updated state and already normed.
// @param qscale ``dk ** -0.5``.
// @param rep Value heads per key head.
// @return ``out``, for call chaining.
// @throws c10::Error when a dtype, a head count or a layout contradicts the state.
torch::Tensor delta_normed(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                           torch::Tensor b, torch::Tensor a,
                           torch::Tensor neg_expA, torch::Tensor dt_bias,
                           torch::Tensor z, torch::Tensor nw, torch::Tensor st,
                           torch::Tensor out, double qscale, int64_t rep);

// Triangular solve: the one library operation the chunked delta rule needs, and the one
// this card's cuSOLVER cannot answer.

// @brief Solve ``A X = B`` for a unit-lower-triangular ``A``.
//
// @param lower [bn, L, L] unit-lower-triangular coefficients; the diagonal is taken as 1
//        and the upper triangle is never read.
// @param rhs [bn, L, n] right-hand side, same dtype as ``lower``.
// @return [bn, L, n] in ``rhs``'s dtype.
// @throws c10::Error on a shape or layout the kernel cannot index, including an ``L`` that
//         would not fit in shared memory.
torch::Tensor triangular_solve(torch::Tensor lower, torch::Tensor rhs);
