// @file bindings.cpp
// @brief pybind11 模块定义：把 CoreX 内核暴露给主机侧 shim。
//
// Module: coffin/backends/corex/kernels。扩展的最外层：它可以 include
// ``corex_ops.h``，但除了这个模块什么都不定义。这里的名字就是
// ``backends/corex/kernels/__init__.py`` 调用的 Python API 面。
//
// Constraints:
//   * 注册的名字是与 Python shim 之间的字符串契约；改一个名字会直接弄坏 import，
//     除非调用方在同一个提交里跟着改。
//   * 模块名来自 ``TORCH_EXTENSION_NAME``，也就是加载器给扩展起的名字，绝不写死。
//   * 文档字符串是进程内唯一的 API 文档。内容只说形状与 dtype 契约，因为算法属于
//     ``.cu`` 文件。
//   * 一个算子一个入口，外加形状扫描需要的调优参数。不放调试或 A/B 入口：Python
//     侧叫不出名字的旋钮不成其为契约，而多一个绑定就是多一份要跟内核保持同步的签名。
//   * ``conv1d_silu`` 和几个 prologue 返回 ``None``：它们的输出是调用方自有的缓冲区，
//     原地写入，这样捕获路径能保住它们的地址。

#include "corex_ops.h"

#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // 投影。
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
    // 和单行入口分开绑定：这里漏了注册在卡上没人调用之前是看不见的，一旦调用就显示
    // 为 AttributeError，而且离真正发起分派的地方很远（PERF_COREX 第 12.5 节记录的
    // 那个坑）。
    m.def("gemv_kn_i8_m", &gemv_kn_i8_m,
          "batched W8A16 GEMV: int8 [K, N] weights, x [M, K] -> y [M, N]",
          pybind11::arg("wt"), pybind11::arg("x"), pybind11::arg("scale"),
          pybind11::arg("threads") = 0, pybind11::arg("unroll") = 0);
    // 归一化。
    m.def("rms_norm", &rms_norm, "fused zero-centred RMSNorm over pitched [M, N] rows",
          pybind11::arg("x"), pybind11::arg("w"), pybind11::arg("stride") = 0,
          pybind11::arg("eps") = 1e-6);
    m.def("add_rms_norm", &add_rms_norm,
          "residual add followed by the zero-centred RMSNorm, in one launch",
          pybind11::arg("x"), pybind11::arg("add"), pybind11::arg("w"),
          pybind11::arg("out"), pybind11::arg("stride") = 0, pybind11::arg("eps") = 1e-6);
    m.def("rms_norm_i8", &rms_norm_i8,
          "zero-centred RMSNorm quantised straight to int8, one launch",
          pybind11::arg("x"), pybind11::arg("w"), pybind11::arg("out_q"),
          pybind11::arg("out_s"), pybind11::arg("stride") = 0,
          pybind11::arg("eps") = 1e-6, pybind11::arg("threads") = 0);
    m.def("add_rms_norm_i8", &add_rms_norm_i8,
          "residual add + zero-centred RMSNorm, quantised straight to int8, one launch",
          pybind11::arg("x"), pybind11::arg("add"), pybind11::arg("w"),
          pybind11::arg("out_q"), pybind11::arg("out_s"), pybind11::arg("stride") = 0,
          pybind11::arg("eps") = 1e-6, pybind11::arg("threads") = 0);
    m.def("rms_norm_gated", &rms_norm_gated,
          "fused gated RMSNorm: rms(x) * w * silu(gate)", pybind11::arg("x"),
          pybind11::arg("w"), pybind11::arg("gate"), pybind11::arg("eps") = 1e-6);
    m.def("quant_act_i8", &quant_act_i8,
          "per-token symmetric int8 quantisation of [M, K], one fp32 scale per row");
    m.def("w8a8_gemm", &w8a8_gemm,
          "int8 GEMM C[M, N] = A[M, K] @ W[K, N], int32 accumulate",
          pybind11::arg("a"), pybind11::arg("w"), pybind11::arg("c"));
    m.def("w8a8_epilogue", &w8a8_epilogue,
          "apply both W8A8 scales to an int32 accumulator and narrow once",
          pybind11::arg("c"), pybind11::arg("xs"), pybind11::arg("ws"),
          pybind11::arg("out"), pybind11::arg("stride") = 0);
    m.def("kv_quant", &kv_quant,
          "symmetric int8 quantisation of [M, N], one fp32 scale per row, one launch");
    m.def("silu_mul", &silu_mul,
          "out = silu(gate) * up, the SwiGLU product, out may alias up",
          pybind11::arg("gate"), pybind11::arg("up"), pybind11::arg("out"));
    m.def("rope_apply", &rope_apply,
          "rotate the leading rotary_dim channels of every q/k head, one launch",
          pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("cos"),
          pybind11::arg("sin"), pybind11::arg("q_out"), pybind11::arg("k_out"),
          pybind11::arg("rotary_dim"));

    // 在 int8 KV 缓存上做注意力。
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

    // 门控 DeltaNet。
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
    m.def("delta_normed", &delta_normed,
          "prologue + recurrence + gated norm for one decode token, one launch");

    // 三角求解：替代这份构建加载不了的 cuSOLVER 路径。
    m.def("triangular_solve", &triangular_solve,
          "unit-lower-triangular solve A X = B over a batch of [L, L] systems",
          pybind11::arg("lower"), pybind11::arg("rhs"));
}
