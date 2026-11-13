// @file moe_block.cu
// @brief 把 decode 一层的 MoE 块合成一次主机侧驱动。
//
// Module: coffin/kernels。由 corex 后端作为 torch 扩展的一部分加载。
//
// Constraints:
//   * 只服务 decode 单行（`tokens == 1`）的 int8 路径；多行与 W4A8 走原来的 Python 编排。
//   * 不在这一步里分配中间缓冲：除各投影内核自己返回的输出外，全部由调用方预分配并复用，
//     否则驱动只是把 Python 的分配搬进 C++，省不下那笔钱。
//   * 数值必须与 `models/qwen35/moe.py` 的 Python 路径逐位相同，包括 `_shared` 的两次窄化。

#include <torch/extension.h>

#include "corex_ops.h"

namespace {

// @brief 取自适应分片数，0 表示交给内核按块数自动选。
constexpr int64_t kAutoSlices = 0;

// @brief 单行 vs 多行投影，与 `backends/corex/__init__.py` 的 `linear`/`linear_m` 同判据。
//
// @param wt    `[K, N]` int8 权重。
// @param x     激活：一维走单行内核，二维走批处理内核。两条路的累加方式不同，混用会
//             让这一层的输出与 Python 路径不逐位相同。
// @param scale `[N]` fp32 逐列 scale。
// @return `[N]` 或 `[M, N]`，dtype 同 `x`。
torch::Tensor project(torch::Tensor wt, torch::Tensor x, torch::Tensor scale) {
  if (x.dim() == 1) return gemv_kn_i8(wt, x, scale, 0, 0, 0, 0);
  // ``GEMV.linear_m`` 在 ``M == 1`` 时会绕回一维入口（见 ops/gemv.py）：多行内核与
  // 单行内核的累加方式不同，这里必须照抄那个判据，否则这一层与 Python 路径不逐位相同。
  if (x.size(0) == 1) return gemv_kn_i8(wt, x[0], scale, 0, 0, 0, 0).unsqueeze(0);
  return gemv_kn_i8_m(wt, x, scale, 0, 0);
}

}  // namespace

torch::Tensor moe_decode_block(torch::Tensor x,
                               torch::Tensor router_wt, torch::Tensor router_scale,
                               torch::Tensor gu_w, torch::Tensor gu_scale,
                               torch::Tensor dn_w, torch::Tensor dn_scale,
                               torch::Tensor sg_wt, torch::Tensor sg_scale,
                               torch::Tensor sd_wt, torch::Tensor sd_scale,
                               torch::Tensor gate_w, torch::Tensor table,
                               torch::Tensor idx, torch::Tensor rw,
                               torch::Tensor offsets, torch::Tensor slot_pos,
                               torch::Tensor token_of_pos, torch::Tensor slot_of_pos,
                               torch::Tensor disp_scratch,
                               torch::Tensor xq_gu, torch::Tensor xs_gu,
                               torch::Tensor gu_out,
                               torch::Tensor xq_dn, torch::Tensor xs_dn,
                               torch::Tensor shared_hidden, torch::Tensor out,
                               int64_t top_k, int64_t slices, int64_t with_shared) {
  const int64_t experts = gu_w.size(0);
  // 全局专家数取自映射表的宽度。专家并行下本 rank 只装 1/world 的专家（``experts``
  // 是局部数），而路由给的是**全局号**，分派内核的边界检查必须用全局数——写成本地数
  // 会把一半槽位当成越界丢掉。单卡 / PP / TP 下表宽等于局部数，两者相同。
  const int64_t global_experts = table.size(0);
  const int64_t inter = gu_w.size(2) / 2;
  const int64_t hidden = gu_w.size(1);

  // 路由：投影 → 融合的 softmax/top-k/归一化。`logits` 与 `moe_router` 的两个输出都按
  // Python 那条路的形状走，`idx`/`rw` 由调用方复用。
  // 路由权重可能是稠密的（本模型的 ``mlp.gate.weight`` 就在 ``DENSE_SUFFIXES`` 里：量化
  // 噪声会翻转专家选择）。``router_scale`` 为空表示稠密，走 bf16 的 ``gemv_kn``，
  // 与 ``bops.linear`` 在稠密权重上的选择一致；否则走 int8 的 W8A16。
  torch::Tensor logits =
      router_scale.numel() == 0
          ? (x.dim() == 1 ? gemv_kn(router_wt, x, 0, 0, 0)
                          : gemv_kn(router_wt, x[0], 0, 0, 0).unsqueeze(0))
          : project(router_wt, x, router_scale);
  moe_router(logits, idx, rw, top_k, true);

  // 共享专家：投影 → SwiGLU → 投影 → 门标量乘到**输出**上（位置错了不报错，只会按另一条
  // 曲线缩放）。`shared_gate_scale` 自己会做 `(x.float()*w).sum(-1).to(bf16)` 再 `sigmoid`
  // 的那两次窄化，这里不重复。
  torch::Tensor sgu = project(sg_wt, x, sg_scale);
  torch::Tensor sgate = sgu.narrow(1, 0, inter);
  torch::Tensor sup = sgu.narrow(1, inter, inter);
  silu_mul(sgate, sup, sup);
  torch::Tensor sh = project(sd_wt, sup, sd_scale);
  shared_gate_scale(x, gate_w, sh, shared_hidden);

  // 专家：一次分派算出分组与位置表，再两次分组 int8 投影。`tokens * top_k` 行全部有效，
  // 所以合并核可以直接把 top_k 行加权成一行，不需要外层再 reduce。
  // ``moe_router`` 写 int64（它自己的契约），而 ``moe_dispatch`` 只收 int32；Python 那条路
  // 也是在包装层做这一次转换的，所以这里逐位等价。
  torch::Tensor flat_index = idx.reshape({-1}).to(torch::kInt);
  moe_dispatch(flat_index, table, offsets, slot_pos, token_of_pos, slot_of_pos,
               disp_scratch, flat_index.numel(), experts, global_experts, top_k);
  torch::Tensor packed = x.index_select(0, token_of_pos);
  quant_act_i8(packed, xq_gu, xs_gu);
  torch::Tensor gu = moe_grouped_i8(xq_gu, xs_gu, gu_w, gu_scale, offsets, gu_out,
                                    slices > 0 ? slices : kAutoSlices);
  torch::Tensor gate = gu.narrow(1, 0, inter);
  torch::Tensor up = gu.narrow(1, inter, inter);
  silu_mul(gate, up, up);
  quant_act_i8(up, xq_dn, xs_dn);
  // ``moe_grouped_i8_merge`` 的逐行权重契约是 fp32；``rw`` 的 dtype 跟着 logits（bf16）。
  // Python 包装层同样做这一步转换。
  torch::Tensor flat_weights =
      rw.reshape({-1}).index_select(0, slot_of_pos).to(torch::kFloat);
  if (global_experts != experts) {
    // 专家并行：``[valid, slots)`` 这些行不属于本 rank。分派内核不把它们排进行序
    // （位置表尾部清 0），但分组内核仍按 ``slots`` 行算完了它们，于是它们带着**真实
    // 投影值**、按「槽位 0 的路由权重」加回去——恒等映射时这批行不存在，所以单卡 /
    // PP / TP 都看不见这个坑；症状是首 token 正确、其后逐 token 变胡话。
    //
    // 掩码用 ``arange < offsets[experts]`` 而不是 ``offsets[-1].item()``：后者是一次
    // D2H 同步，每层一次会把主机预取打断（qwen4exp 量到 +11.7%），而这里整个比较
    // 都留在设备上。
    auto rows = flat_index.numel();
    auto keep = at::arange(rows, flat_weights.options().dtype(torch::kInt32)) <
                offsets[experts];
    flat_weights = flat_weights * keep.to(torch::kFloat);
  }
  torch::Tensor merged = moe_grouped_i8_merge(xq_dn, xs_dn, dn_w, dn_scale, offsets,
                                              flat_weights, out,
                                              slices > 0 ? slices : kAutoSlices);
  // `merged` 就是 `out`（合并核写进去的那个）。共享专家要不要现在加，由调用方定：
  // 专家并行下本函数只返回**本 rank 的专家部分**，共享专家必须在跨 rank 求和之后
  // 再加一次——这里加了的话，四个 rank 各自带一份共享贡献，求和会把它算四遍。
  if (with_shared != 0) {
    at::add_out(out, merged, shared_hidden);
  }
  return out;
}