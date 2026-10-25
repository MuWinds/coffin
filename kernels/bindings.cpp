// @file bindings.cpp
// @brief pybind11 module definition: exposes the HIP kernels to ``hip_ops.py``.
//
// Module: coffin/kernels. The outermost layer of the extension, so it may include
// ``hip_ops.h`` but must define nothing else; the names here are the Python API
// surface the engine calls.
//
// Constraints:
//   * The registered names are the Python side's contract (``hip_ops.py`` looks
//     them up by string). Renaming one silently breaks the import until the
//     Python caller is updated in the same commit.
//   * The module name comes from ``TORCH_EXTENSION_NAME``: it is whatever the
//     loader named the extension, so never hard-code it.
//   * The doc strings are the only in-process API documentation; keep them short
//     and about the shape contract, not the algorithm.

#include "hip_ops.h"

#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemv_bf16", &gemv_bf16,
          "bf16 GEMV (R rows per warp, U-way K-loop unroll)");
    m.def("gemv_f16", &gemv_f16,
          "fp16-weight GEMV, bf16 activations (same layout as gemv_bf16)");
    m.def("gemv_bf16_m", &gemv_bf16_m,
          "batched small-M bf16 GEMV (M input vectors share one weight pass)");
    m.def("gemv_f16_m", &gemv_f16_m,
          "batched small-M fp16-weight GEMV");
    m.def("gemv_w8a16", &gemv_w8a16,
          "int8-weight bf16-activation GEMV (per-row scale, R rows/warp, U unroll)");
    m.def("gemv_w8a16_m", &gemv_w8a16_m,
          "batched small-M int8-weight GEMV");
    m.def("delta_step", &delta_step,
          "gated delta rule decode step, updates the state in place");
    m.def("attn_decode", &attn_decode,
          "fused single-token attention over the int8 KV cache");
    m.def("attn_paged_decode", &attn_paged_decode,
          "fused single-token attention over the paged int8 KV pool");
    m.def("rms_norm", &rms_norm, "fused RMSNorm over strided [M, N] rows");
    m.def("rms_norm_gated", &rms_norm_gated, "fused gated RMSNorm");
    m.def("delta_prologue", &delta_prologue,
          "fused q/k norm + head expand + beta/g for the gated delta rule");
    m.def("conv1d_silu", &conv1d_silu,
          "fused depthwise causal conv1d + SiLU for the linear-attn layers");
    m.def("conv1d_silu_seq", &conv1d_silu_seq,
          "same conv over a whole prefill chunk [T, C]");
    m.def("delta_prep_seq", &delta_prep_seq,
          "fused prefill q/k norm + head expand + beta/g into [H, T, d]");
    m.def("attn_paged_batch", &attn_paged_batch,
          "batched single-token attention over per-row paged int8 pool windows");
    m.def("attn_paged_multi_q", &attn_paged_multi_q,
          "multi-query verify attention: one history pass serves c queries");
    m.def("conv1d_silu_batch", &conv1d_silu_batch,
          "batched fused depthwise causal conv1d + SiLU");
    m.def("delta_prologue_batch", &delta_prologue_batch,
          "batched fused q/k norm + head expand + beta/g for the gated delta rule");
    m.def("delta_batch", &delta_batch,
          "batched gated delta rule decode step, updates [B, heads, dk, dv]");
    m.def("gemv_w8a16_wide", &gemv_w8a16_wide,
          "wide-M int8 W8A16 GEMV for verify (M in 16/24/32, one weight pass)");
    m.def("gemv_w8a16_wide_dbg", &gemv_w8a16_wide_dbg,
          "A/B entry for gemv_w8a16_wide: rows/mw tiling exposed");
    m.def("gemv_w8a16_wide_dot2", &gemv_w8a16_wide_dot2,
          "wide-M int8-weight GEMV with packed fp16 dot2 math (M in 16/24/32)");
    m.def("gemv_w8a16_wide_dot2_dbg", &gemv_w8a16_wide_dot2_dbg,
          "A/B entry for gemv_w8a16_wide_dot2: rows/mw tiling exposed");
}
