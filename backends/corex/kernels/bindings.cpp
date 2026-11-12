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

#include <cstring>
#include <string>
#include <tuple>

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
    // 窄 N 的 M 行投影：网格按 K 切片补齐 SM；split=0 自动选、split<=1 退化成
    // gemv_kn_m。共享专家 gate_up 的 N=512 不分片只有 2 个块。
    m.def("gemv_kn_m_split", &gemv_kn_m_split,
          "small-M GEMV with split-K over [K, N] weights: x [M, K] -> y [M, N]",
          pybind11::arg("wt"), pybind11::arg("x"), pybind11::arg("threads") = 0,
          pybind11::arg("split") = 0);
    m.def("gemv_kn_i8", &gemv_kn_i8,
          "decode W8A16 GEMV: int8 [K, N] weights plus a per-column fp32 scale",
          pybind11::arg("wt"), pybind11::arg("x"), pybind11::arg("scale"),
          pybind11::arg("threads") = 0, pybind11::arg("unroll") = 0,
          pybind11::arg("split") = 0, pybind11::arg("width") = 0);
    // 和单行入口分开绑定：这里漏了注册在卡上没人调用之前是看不见的，一旦调用就显示
    // 为 AttributeError，而且离真正发起分派的地方很远（docs/qwen4/QWEN4-PERF.md 记录的
    // 那个坑）。
    m.def("gemv_kn_i8_m", &gemv_kn_i8_m,
          "batched W8A16 GEMV: int8 [K, N] weights, x [M, K] -> y [M, N]",
          pybind11::arg("wt"), pybind11::arg("x"), pybind11::arg("scale"),
          pybind11::arg("threads") = 0, pybind11::arg("unroll") = 0);
    m.def("gemv_kn_w4", &gemv_kn_w4,
          "group-scaled packed W4 GEMV with bf16/fp32 activations",
          pybind11::arg("packed"), pybind11::arg("scale"), pybind11::arg("x"),
          pybind11::arg("k"), pybind11::arg("group_size"));
    // 调优旋钮全部带默认值 ``0``（= 沿用已发布的规则），所以不传它们时这一条与
    // 引入旋钮之前逐位相同。名字与 ``gemv_kn_i8`` 保持同一套：同一个概念在两个核上
    // 换名字，扫描脚本就没法共用一份形状表。
    m.def("w4a8_gemv", &w4a8_gemv,
          "分组 W4 与 int8 激活的整数点积，int32 逐组累加",
          pybind11::arg("a"), pybind11::arg("w"), pybind11::arg("xs"),
          pybind11::arg("ws"), pybind11::arg("out"), pybind11::arg("group_size"),
          pybind11::arg("threads") = 0, pybind11::arg("unroll") = 0,
          pybind11::arg("split") = 0, pybind11::arg("width") = 0);
    m.def("w4_dequant", &w4_dequant,
          "packed W4 权重一趟反量化成 bf16（大 M 的厂商 GEMM 前置）",
          pybind11::arg("packed"), pybind11::arg("scale"),
          pybind11::arg("k"), pybind11::arg("n"), pybind11::arg("group"));
    m.def("w4a8_gemm_tcu", &w4a8_gemm_tcu,
          "分组 packed W4 与 int8 激活的张量核投影",
          pybind11::arg("a"), pybind11::arg("w"), pybind11::arg("xs"),
          pybind11::arg("ws"), pybind11::arg("out"), pybind11::arg("group_size"),
          // M 方向 tile：0 表示默认 128，64 / 256 是另外两档；db 非 0 时走
          // 「双缓冲 + k 步长 32」那一档。两个参数都只用于扫参与稠密路径调优。
          pybind11::arg("m_tile") = 0, pybind11::arg("db") = 0);
    m.def("quant_act_i8_scaled", &quant_act_i8_scaled,
          "按跨 rank 共享的逐行 scale 量化激活",
          pybind11::arg("x"), pybind11::arg("out"), pybind11::arg("xs"));
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
    m.def("w8a8_gemm_tcu", &w8a8_gemm_tcu,
          "wide-instruction (i8x8) tensor-core int8 GEMM with the epilogue fused: "
          "out[M, N] = (A[M, K] @ W[K, N]) scaled, in one launch",
          pybind11::arg("a"), pybind11::arg("w"), pybind11::arg("xs"),
          pybind11::arg("ws"), pybind11::arg("out"));
    m.def("w8a8_dp4a_nk", &w8a8_dp4a_nk,
          "W8A8 DP4A GEMM over contiguous [N, K] int8 weights",
          pybind11::arg("a"), pybind11::arg("wt"), pybind11::arg("c"),
          pybind11::arg("out"), pybind11::arg("xs"), pybind11::arg("ws"),
          pybind11::arg("threads") = 0);
    m.def("kv_quant", &kv_quant,
          "symmetric int8 quantisation of [M, N], one fp32 scale per row, one launch");
    m.def("silu_mul", &silu_mul,
          "out = silu(gate) * up, the SwiGLU product, out may alias up",
          pybind11::arg("gate"), pybind11::arg("up"), pybind11::arg("out"));
    m.def("shared_gate_scale", &shared_gate_scale,
          "out = src * sigmoid(x . w), one launch for the shared-expert gate",
          pybind11::arg("x"), pybind11::arg("w"), pybind11::arg("src"),
          pybind11::arg("out"));
    m.def("rope_apply", &rope_apply,
          "rotate the leading rotary_dim channels of every q/k head, one launch",
          pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("cos"),
          pybind11::arg("sin"), pybind11::arg("q_out"), pybind11::arg("k_out"),
          pybind11::arg("rotary_dim"));

    // 在 int8 KV 缓存上做注意力。
    m.def("attn_decode", &attn_decode,
          "single-token attention over the dense int8 KV cache");
    m.def("attn_paged", &attn_paged,
          "\u5206\u9875\u6ce8\u610f\u529b\uff1a\u7ed9\u5b9a\u4f4d\u7f6e\u5728 int8 KV \u6c60\u4e0a\u7b97\u4e00\u4e2a\u89e3\u7801 token",
          pybind11::arg("q"), pybind11::arg("gate"), pybind11::arg("kpool"),
          pybind11::arg("vpool"), pybind11::arg("kscale"), pybind11::arg("vscale"),
          pybind11::arg("pos2slot"), pybind11::arg("out"), pybind11::arg("pos"),
          pybind11::arg("max_pos"), pybind11::arg("rep"), pybind11::arg("scale"),
          pybind11::arg("part_acc"), pybind11::arg("part_ml"),
          pybind11::arg("head_major") = false);
    m.def("attn_paged_multi_q", &attn_paged_multi_q,
          "\u6295\u673a\u6821\u9a8c\u5f62\u6001\uff1a\u6bcf\u884c c \u4e2a\u8349\u7a3f\u67e5\u8be2\u5171\u7528\u4e00\u904d\u5386\u53f2\u8bfb\u53d6",
          pybind11::arg("q"), pybind11::arg("gate"), pybind11::arg("kpool"),
          pybind11::arg("vpool"), pybind11::arg("kscale"), pybind11::arg("vscale"),
          pybind11::arg("slots"), pybind11::arg("ctx_lens"), pybind11::arg("out"),
          pybind11::arg("max_seq"), pybind11::arg("rep"), pybind11::arg("scale"),
          pybind11::arg("part_acc"), pybind11::arg("part_ml"),
          pybind11::arg("block_mask"), pybind11::arg("mask_block"),
          pybind11::arg("head_major") = false);
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
    m.def("delta_record", &delta_record,
          "记录秩一更新的 gated delta 解码步");
    m.def("delta_batch", &delta_batch,
          "batched gated delta decode step, one state row per batch entry");
    m.def("delta_batch_record", &delta_batch_record,
          "批量记录秩一更新的 gated delta 解码步");
    m.def("delta_commit", &delta_commit,
          "从初始状态重放秩一更新");
    m.def("delta_normed", &delta_normed,
          "prologue + recurrence + gated norm for one decode token, one launch");
    m.def("delta_normed_record", &delta_normed_record,
          "融合 gated delta 校验步并记录秩一更新");
    m.def("delta_normed_seq", &delta_normed_seq,
          "c 个校验 token 的融合链，一次启动跑完整个投机窗口");
    m.def("delta_normed_seq_record", &delta_normed_seq_record,
          "c 个校验 token 的融合链并逐 token 记录秩一更新");

    // 三角求解：替代这份构建加载不了的 cuSOLVER 路径。
    m.def("triangular_solve", &triangular_solve,
          "unit-lower-triangular solve A X = B over a batch of [L, L] systems",
          pybind11::arg("lower"), pybind11::arg("rhs"));

    m.def("hyper_norm", &hyper_norm,
          "分组 RMSNorm：hc 路残差流各自在 hidden 维上归一化",
          pybind11::arg("x"), pybind11::arg("w"), pybind11::arg("out"),
          pybind11::arg("hidden"), pybind11::arg("hc"), pybind11::arg("eps"));
    m.def("hyper_down", &hyper_down,
          "Gated Residual 的 down 投影加 silu 门",
          pybind11::arg("normed"), pybind11::arg("wt"), pybind11::arg("out"),
          pybind11::arg("hc"));
    m.def("moe_grouped", &moe_grouped,
          "\u5206\u7ec4 W4A8 \u6295\u5f71\uff1a\u6bcf\u4e2a\u4e13\u5bb6\u53ea\u7b97\u8def\u7531\u5230\u5b83\u7684\u884c",
          pybind11::arg("a"), pybind11::arg("xs"), pybind11::arg("w"),
          pybind11::arg("ws"), pybind11::arg("offsets"), pybind11::arg("out"),
          pybind11::arg("group_size"),
          // 紧凑非空专家表，可为 None。decode 一个 token 只碰约 10 个专家，而每卡
          // 持 128 个；按全量开 z 维会让 92% 的 block 一进来就退出。
          pybind11::arg("active") = pybind11::none());
    m.def("moe_grouped_i8", &moe_grouped_i8,
          "\u5206\u7ec4 int8 \u6295\u5f71\uff1a\u4e13\u5bb6\u6808 [E, K, N] int8 "
          "+ \u9010 (\u4e13\u5bb6, \u5217) scale",
          pybind11::arg("a"), pybind11::arg("xs"), pybind11::arg("w"),
          pybind11::arg("ws"), pybind11::arg("offsets"), pybind11::arg("out"),
          pybind11::arg("slices") = 0);
    m.def("moe_grouped_i8_merge", &moe_grouped_i8_merge,
          "分组 int8 投影的归并形态：各行按路由权重加权累加成一行",
          pybind11::arg("a"), pybind11::arg("xs"), pybind11::arg("w"),
          pybind11::arg("ws"), pybind11::arg("offsets"), pybind11::arg("rw"),
          pybind11::arg("out"), pybind11::arg("slices") = 0);
    m.def("moe_decode_block", &moe_decode_block,
          "decode 一层的 MoE 块：路由 → 共享专家 → 分派 → 两次分组 int8 → 合并 → 残差",
          pybind11::arg("x"),
          pybind11::arg("router_wt"), pybind11::arg("router_scale"),
          pybind11::arg("gu_w"), pybind11::arg("gu_scale"),
          pybind11::arg("dn_w"), pybind11::arg("dn_scale"),
          pybind11::arg("sg_wt"), pybind11::arg("sg_scale"),
          pybind11::arg("sd_wt"), pybind11::arg("sd_scale"),
          pybind11::arg("gate_w"), pybind11::arg("table"),
          pybind11::arg("idx"), pybind11::arg("rw"),
          pybind11::arg("offsets"), pybind11::arg("slot_pos"),
          pybind11::arg("token_of_pos"), pybind11::arg("slot_of_pos"),
          pybind11::arg("disp_scratch"),
          pybind11::arg("xq_gu"), pybind11::arg("xs_gu"),
          pybind11::arg("gu_out"),
          pybind11::arg("xq_dn"), pybind11::arg("xs_dn"),
          pybind11::arg("shared_hidden"), pybind11::arg("out"),
          pybind11::arg("top_k"), pybind11::arg("slices") = 0,
          pybind11::arg("with_shared") = 1);
    m.def("moe_dispatch", &moe_dispatch,
          "\u5206\u6d3e\uff1a\u8def\u7531\u7ed3\u679c \u2192 \u524d\u7f00\u548c + \u884c\u5e8f",
          pybind11::arg("expert_idx"), pybind11::arg("local_table"),
          pybind11::arg("out_offsets"), pybind11::arg("out_slot_pos"),
          pybind11::arg("out_tokens"), pybind11::arg("out_slots"),
          pybind11::arg("scratch"),
          pybind11::arg("slots"), pybind11::arg("local_experts"),
          pybind11::arg("num_experts"), pybind11::arg("top_k"));
    m.def("moe_grouped_gemm", &moe_grouped_gemm,
          "\u5206\u7ec4 MoE \u7684\u5927 M \u5f62\u6001\uff1a\u6309\u4e13\u5bb6\u5206\u6bb5\u5404\u505a\u4e00\u6b21\u5f20\u91cf\u6838 GEMM",
          pybind11::arg("a"), pybind11::arg("xs"), pybind11::arg("w"),
          pybind11::arg("ws"), pybind11::arg("offsets"), pybind11::arg("out"),
          pybind11::arg("group_size"));
    m.def("attn_paged_batch", &attn_paged_batch,
          "分页注意力的批形态：一个桶的行各带自身上下文长度，一次算一个解码 token",
          pybind11::arg("q"), pybind11::arg("gate"), pybind11::arg("kpool"),
          pybind11::arg("vpool"), pybind11::arg("kscale"), pybind11::arg("vscale"),
          pybind11::arg("slots"), pybind11::arg("context_lens"), pybind11::arg("out"),
          pybind11::arg("max_seq"), pybind11::arg("rep"), pybind11::arg("scale"),
          pybind11::arg("part_acc"), pybind11::arg("part_ml"),
          pybind11::arg("head_major") = false);
    m.def("attn_sparse", &attn_sparse,
          "QSA \u7684\u7a00\u758f\u6ce8\u610f\u529b\uff1a\u53ea\u8bfb\u7d22\u5f15\u5668\u9009\u4e2d\u7684\u4f4d\u7f6e",
          pybind11::arg("q"), pybind11::arg("kcache"), pybind11::arg("vcache"),
          pybind11::arg("sel_idx"), pybind11::arg("sel_len"), pybind11::arg("ctx"),
          pybind11::arg("out"), pybind11::arg("scale"));
    m.def("qsa_block_keys", &qsa_block_keys,
          "QSA \u5757 key\uff1agather + \u6c60\u5316 + RMSNorm + partial RoPE",
          pybind11::arg("flat_keys"), pybind11::arg("table"), pybind11::arg("weight"),
          pybind11::arg("flat_cos"), pybind11::arg("flat_sin"),
          pybind11::arg("block_row"), pybind11::arg("block_batch"),
          pybind11::arg("out"), pybind11::arg("seq"), pybind11::arg("tokens"),
          pybind11::arg("rotary_dim"), pybind11::arg("eps"));
    m.def("hyper_up", &hyper_up,
          "Gated Residual 的 up 投影加读门塌缩与写门",
          pybind11::arg("normed"), pybind11::arg("up_wt"), pybind11::arg("low"),
          pybind11::arg("inj_wt"), pybind11::arg("mixed"), pybind11::arg("inject"),
          pybind11::arg("hidden"), pybind11::arg("hc"), pybind11::arg("lowrank"));
    // 路由融合：替掉 ``softmax -> topk -> div -> sum -> to`` 五个内核，其中 topk 对
    // ``[1, 512]`` 取 top-10 在本卡上走 bitonic 排序网络、单次 38 us。
    m.def("moe_router", &moe_router,
          "\u8def\u7531\uff1a\u878d\u5408 softmax -> top-k -> \u5f52\u4e00\u5316\uff0c"
          "\u4e00\u6b21\u53d1\u5c04\u5199\u51fa\u4e0b\u6807\u4e0e\u6743\u91cd",
          pybind11::arg("logits"), pybind11::arg("out_idx"), pybind11::arg("out_w"),
          pybind11::arg("top_k"), pybind11::arg("normalize"));
    m.def("mapped_pinned_probe",
          [](int64_t detail_len) {
              std::string detail(static_cast<size_t>(detail_len > 0 ? detail_len : 256), '\0');
              const int ok = mapped_pinned_probe(detail.data(), static_cast<int>(detail.size()));
              detail.resize(std::strlen(detail.c_str()));
              return std::make_tuple(ok, detail);
          },
          "\u63a2\u6d4b\u6620\u5c04\u9501\u9875\u5185\u5b58\u662f\u5426\u53ef\u7528\uff1a"
          "\u8fd4\u56de (1/0, \u7ed3\u8bba\u6587\u672c)",
          pybind11::arg("detail_len") = 256);
    m.def("mapped_register",
          [](int64_t host_ptr, int64_t bytes) {
              std::string detail(256, '\0');
              const int64_t dev = mapped_register(
                  reinterpret_cast<void*>(static_cast<uintptr_t>(host_ptr)), bytes,
                  detail.data(), static_cast<int>(detail.size()));
              detail.resize(std::strlen(detail.c_str()));
              return std::make_tuple(dev, detail);
          },
          "\u6ce8\u518c mmap \u533a\u4e3a\u53ef\u6620\u5c04\u9501\u9875\u5185\u5b58\uff0c"
          "\u8fd4\u56de (dev_ptr, \u7ed3\u8bba\u6587\u672c)",
          pybind11::arg("host_ptr"), pybind11::arg("bytes"));
    m.def("mapped_release", &mapped_release,
          "\u89e3\u6ce8\u518c\u6620\u5c04\uff0c\u5fc5\u987b\u5728 munmap \u4e4b\u524d\u8c03\u7528",
          pybind11::arg("host_ptr"));
    m.def("mapped_stage", &mapped_stage,
          "\u672c rank \u5c40\u90e8\u548c\u5199\u8fdb\u5171\u4eab\u69fd\u5e76\u7f6e\u5230\u8fbe\u8ba1\u6570",
          pybind11::arg("x"), pybind11::arg("dev_base"), pybind11::arg("world"),
          pybind11::arg("rank"), pybind11::arg("threads"), pybind11::arg("round"));
    m.def("mapped_finish", &mapped_finish,
          "\u4ece\u6240\u6709 rank \u7684\u69fd\u6c42\u548c\u5230\u8bbe\u5907 out",
          pybind11::arg("out"), pybind11::arg("dev_base"), pybind11::arg("world"),
          pybind11::arg("blocks"), pybind11::arg("threads"));
    m.def("mapped_finish_wait", &mapped_finish_wait,
          "\u8bbe\u5907\u4fa7\u7b49\u6240\u6709 rank \u5230\u8fbe\u540e\u6c42\u548c"
          "\uff08\u4e3b\u673a\u4e0d\u9700\u8981\u540c\u6b65\u6216\u8f6e\u8be2\uff09",
          pybind11::arg("out"), pybind11::arg("dev_base"), pybind11::arg("world"),
          pybind11::arg("blocks"), pybind11::arg("threads"), pybind11::arg("round"));
    m.def("p2p_handle", &p2p_handle,
          "\u53d6\u8bbe\u5907\u7f13\u51b2\u7684 IPC \u53e5\u67c4\uff08uint8 \u5f20\u91cf\uff09",
          pybind11::arg("device_buffer"));
    m.def("p2p_open", &p2p_open,
          "\u7528\u5bf9\u7aef\u53e5\u67c4\u6620\u5c04\u5176\u663e\u5b58\uff0c"
          "\u8fd4\u56de\u8bbe\u5907\u6307\u9488",
          pybind11::arg("handle_bytes"));
    m.def("p2p_combine", &p2p_combine,
          "\u56db\u5361 all-to-all \u5f52\u7ea6\uff1a\u5199\u5bf9\u7aef\u69fd + "
          "\u81ea\u65cb\u7b49\u9f50 + \u6c42\u548c\uff0c\u4e00\u4e2a\u6838\u5b8c\u6210",
          pybind11::arg("x"), pybind11::arg("out"),
          pybind11::arg("slots0"), pybind11::arg("slots1"),
          pybind11::arg("slots2"), pybind11::arg("slots3"),
          pybind11::arg("flags0"), pybind11::arg("flags1"),
          pybind11::arg("flags2"), pybind11::arg("flags3"),
          pybind11::arg("rank"), pybind11::arg("round"),
          pybind11::arg("blocks") = 16, pybind11::arg("threads") = 256);
}

