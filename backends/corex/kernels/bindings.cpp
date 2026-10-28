// @file bindings.cpp
// @brief pybind11 module definition: exposes the CoreX kernels to the host shim.
//
// Module: coffin/backends/corex/kernels. Outermost layer of the extension: it may
// include ``corex_ops.h`` but defines nothing but the module. The names here are the
// Python API surface ``backends/corex/kernels/__init__.py`` calls.
//
// Constraints:
//   * The registered names are a string contract with the Python shim; renaming one
//     breaks the import until the caller changes in the same commit.
//   * The module name comes from ``TORCH_EXTENSION_NAME`` -- it is whatever the loader
//     named the extension, never hard-coded.
//   * The docstrings are the only in-process API documentation. Keep them about the shape
//     and dtype contract, because the algorithm belongs to the ``.cu`` file.
//   * One entry per op, plus the tuning arguments the shape sweep needs. No debug or A/B
//     entry points: a knob the Python side cannot name is not a contract, and an extra
//     binding is an extra signature to keep in step with the kernel.
//   * ``conv1d_silu`` and the prologues return ``None``: their outputs are caller-owned
//     buffers, written in place so the capture path keeps their addresses.

#include "corex_ops.h"

#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Projections.
    m.def("gemv_kn", &gemv_kn, "decode GEMV over [K, N] weights: x [K] -> y [N]",
          pybind11::arg("wt"), pybind11::arg("x"), pybind11::arg("threads") = 0,
          pybind11::arg("unroll") = 0, pybind11::arg("split") = 0);
    m.def("gemv_kn_m", &gemv_kn_m,
          "small-M GEMV over [K, N] weights: x [M, K] -> y [M, N]",
          pybind11::arg("wt"), pybind11::arg("x"), pybind11::arg("threads") = 0,
          pybind11::arg("unroll") = 0);
    m.def("gemv_kn_i8", &gemv_kn_i8,
          "decode W8A16 GEMV: int8 [K, N] weights plus a per-column fp32 scale",
          pybind11::arg("wt"), pybind11::arg("x"), pybind11::arg("scale"),
          pybind11::arg("threads") = 0, pybind11::arg("unroll") = 0,
          pybind11::arg("split") = 0, pybind11::arg("width") = 0);

    // Normalisation.
    m.def("rms_norm", &rms_norm, "fused zero-centred RMSNorm over pitched [M, N] rows",
          pybind11::arg("x"), pybind11::arg("w"), pybind11::arg("stride") = 0,
          pybind11::arg("eps") = 1e-6);
    m.def("rms_norm_gated", &rms_norm_gated,
          "fused gated RMSNorm: rms(x) * w * silu(gate)", pybind11::arg("x"),
          pybind11::arg("w"), pybind11::arg("gate"), pybind11::arg("eps") = 1e-6);

    // Attention over the int8 KV cache.
    m.def("attn_decode", &attn_decode,
          "single-token attention over the dense int8 KV cache");
    m.def("attn_paged", &attn_paged,
          "single-token attention over a paged int8 pool via a device slot map");
    m.def("attn_paged_batch", &attn_paged_batch,
          "batched single-token attention, one context length per row");
    m.def("attn_paged_multi_q", &attn_paged_multi_q,
          "draft-verify attention: c queries per row share one history pass",
          pybind11::arg("q"), pybind11::arg("gate"), pybind11::arg("kpool"),
          pybind11::arg("vpool"), pybind11::arg("kscale"), pybind11::arg("vscale"),
          pybind11::arg("slots"), pybind11::arg("ctx_lens"), pybind11::arg("out"),
          pybind11::arg("max_seq"), pybind11::arg("rep"), pybind11::arg("scale"),
          pybind11::arg("part_acc"), pybind11::arg("part_ml"), pybind11::arg("block_mask"),
          pybind11::arg("mask_block") = 64);

    // Gated DeltaNet.
    m.def("conv1d_silu", &conv1d_silu,
          "depthwise causal conv1d + SiLU for one token, shifts the window in place");
    m.def("conv1d_silu_seq", &conv1d_silu_seq,
          "the same conv over a prefill chunk [T, C]");
    m.def("conv1d_silu_batch", &conv1d_silu_batch,
          "batched depthwise causal conv1d + SiLU, one window per row");
    m.def("delta_prologue", &delta_prologue,
          "q/k L2 norm, head expansion and the beta/g gates for one token");
    m.def("delta_prologue_batch", &delta_prologue_batch,
          "batched gated-delta prologue, leading row dim on every argument");
    m.def("delta_prep_seq", &delta_prep_seq,
          "prefill head prep into the head-major [H, T, d] layout");
    m.def("delta", &delta, "gated delta decode step, updates the state in place");
    m.def("delta_batch", &delta_batch,
          "batched gated delta decode step, one state row per batch entry");

    // Triangular solve: a replacement for the cuSOLVER path this build cannot load.
    m.def("triangular_solve", &triangular_solve,
          "unit-lower-triangular solve A X = B over a batch of [L, L] systems",
          pybind11::arg("lower"), pybind11::arg("rhs"));
}
