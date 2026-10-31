// @file corex_ops.h
// @brief 在 ``bindings.cpp`` 里绑定的 CoreX 扩展入口声明。
//
// Module: coffin/backends/corex/kernels。内核和绑定共享的唯一头文件；它是纯接口，
// 所以必须不含设备代码——``bindings.cpp`` 由主机编译器编译，那里没有 ``__device__``。
// bf16 转换因此放在 ``corex_scalar.h``。
//
// Constraints:
//   * 这里的声明就是绑定名：``backends/corex/kernels/__init__.py`` 按字符串查找它们。
//     改名必须在同一次改动里落到两边，否则扩展 import 时会缺符号。
//   * 每个包装层接收它要索引的形状，并 TORCH_CHECK 它假定的布局（dtype、连续性、
//     二维性）。内核索引时不做边界检查，所以错误的布局必须在主机侧失败，而不是读穿
//     缓冲区。
//   * 权重全程是 ``[K, N]``——``Caps.weight_layout == "kn"`` 承诺的布局。把 ``[N, K]``
//     张量交给这些入口，会拿错元素算出看着合理的数，所以转置归 Python 侧负责，包装层
//     只查 dtype。
//   * 每个包装层都在当前 torch stream 上启动。在 legacy 默认流上跑的内核不属于
//     ``torch.cuda.graph`` 的捕获范围，会被解码图丢掉并重放陈旧输出。

#pragma once

#include <torch/extension.h>

#include <cstdint>

// 投影：y[m, n] = sum_k x[m, k] * wt[k, n]。

// @brief 解码期 GEMV：一个激活向量，每个输出列一个线程。
//
// @param wt [K, N] 权重，连续存放，bf16、fp16 或 fp32。
// @param x [K] 激活，bf16 或 fp32。输出的 dtype 跟随它。
// @param threads 每块线程数；``0`` 表示按 ``N`` 选。
// @param unroll K 循环展开因子，``2`` 或 ``4``；``0`` 表示 4。
// @param split Split-K 宽度；``0`` 表示按 ``N`` 和 ``K`` 选，``1`` 强制单趟内核。
// @return [N]，dtype 同 ``x``。
// @throws c10::Error 形状、dtype 或布局是内核索引不了的。
torch::Tensor gemv_kn(torch::Tensor wt, torch::Tensor x, int64_t threads,
                      int64_t unroll, int64_t split);

// @brief 小 M 批处理 GEMV：一次权重读取产出 ``M`` 行激活。
//
// @param wt [K, N] 权重，连续存放，bf16、fp16 或 fp32。
// @param x [M, K] 激活，``M`` 在 ``[1, 8]``。
// @param threads 每块线程数；``0`` 表示按 ``N`` 选。
// @param unroll K 循环展开因子；``0`` 表示 4。
// @return [M, N]，dtype 同 ``x``。
// @throws c10::Error 形状、dtype 或布局是内核索引不了的。
torch::Tensor gemv_kn_m(torch::Tensor wt, torch::Tensor x, int64_t threads,
                        int64_t unroll);

// @brief 解码 W8A16 投影：int8 权重，外加每个输出列一个 fp32 scale。
//
// 相比 :func:`gemv_kn` 把每个权重读的字节数减半。scale 只乘一次，乘在算完的点积上，
// 绝不逐元素乘，也绝不折进激活。
//
// @param wt [K, N] int8 权重，连续存放。
// @param x [K] 激活，bf16 或 fp32。
// @param scale [N] fp32 absmax scale，每个输出列一个。
// @param threads 每块线程数；``0`` 表示按 ``N`` 选。
// @param unroll K 循环展开因子；``0`` 表示 4。
// @param split Split-K 宽度；``0`` 表示自动选，``1`` 强制单趟内核。
// @param width 一个线程负责的相邻输出列数，同时也是它每个 k 步加载的连续权重字节数：
//        ``2`` 是一次 16 位加载，``4`` 一次 32 位，``8`` 一次 64 位。合法值是 ``1``、
//        ``2``、``4``、``8`` 和 ``16``；其它非零值直接拒绝而不是四舍五入，并且
//        ``N % width`` 必须为 ``0``（不满足时启动会退回 ``1``，这样一个宽度不是
//        ``width`` 整数倍的权重仍然正确，而不是跨行读）。``0`` 表示自动，精确复现
//        引入这个参数之前的分派：列数超过门槛走四宽，否则走标量。读得更宽会把线程数
//        除以 ``width``，所以最优值是按形状实测出来的取舍而不是常量——见
//        ``diag/tune_gemv_i8_wide.py``。
// @return [N]，dtype 同 ``x``。
// @throws c10::Error 权重不是 int8、scale 长度不是 ``N``、``width`` 不受支持，或布局
//         是内核索引不了的。
torch::Tensor gemv_kn_i8(torch::Tensor wt, torch::Tensor x, torch::Tensor scale,
                         int64_t threads, int64_t unroll, int64_t split,
                         int64_t width);

// @brief 批处理 W8A16 投影：M 行激活只读一遍 int8 权重。
//
// 没有这个入口，``linear_m`` 的 int8 分支就没有内核可调，只能循环单行 GEMV，每行付
// 一次完整权重读取——在本部件上 M=8 是 M=1 的 8.19 倍（实测，五个解码形状）。这就是
// 一次 c-token 投机校验打不过 c 个普通步的原因；见 ``PERF_COREX.md`` 第 14 节。
//
// @param wt [K, N] int8 权重，连续存放。
// @param x [M, K] 激活，bf16 或 fp32。输出的 dtype 跟随它。
// @param scale [N] fp32 absmax scale，每个输出列一个。
// @param threads 每块线程数；``0`` 表示按 ``N`` 选。
// @param unroll K 循环展开；启动器会定死它，所以这个参数留给扫描用。
// @return [M, N]，dtype 同 ``x``。
// @throws c10::Error 权重不是 int8、scale 长度不是 N，或布局与 dtype 是内核索引不
//         了的。
torch::Tensor gemv_kn_i8_m(torch::Tensor wt, torch::Tensor x, torch::Tensor scale,
                            int64_t threads, int64_t unroll);

// 归一化。

// @brief 带行间距的 ``[M, N]`` 最后一轴零中心 RMSNorm。
//
// @param x [M, N] 激活，最内层步长为 1；行之间相隔 ``stride`` 个元素，这让一个列切片
//        可以原地归一化。
// @param w [N] 增益，已经带上了 checkpoint 的 ``1 + w`` 偏移。
// @param stride 行之间的元素间距；``0`` 表示 ``N``。
// @param eps 方差下限，在 rsqrt 内部相加。
// @return [M, N]，稠密打包，dtype 同 ``x``。
// @throws c10::Error dtype 没有对应实例化，或间距窄于行宽。
torch::Tensor rms_norm(torch::Tensor x, torch::Tensor w, int64_t stride, double eps);

// @brief 残差相加接零中心 RMSNorm，一次启动。
//
// 每个解码层都做 ``x = x + h`` 然后归一化 ``x``。拆成两个算子就是两次启动、两遍读
// 残差缓冲区；融合后是一次启动、一遍读取，而且这个和只有一个家。做成独立入口而不是
// :func:`rms_norm` 的一个可选参数，是为了让没有这个内核的后端回退到参考实现——正确，
// 只是慢——而不是悄悄把相加丢掉。
//
// @param x [M, N] 残差流，用 ``add`` 原地累加。
// @param add [M, N] 待累加的混合器输出。
// @param w [N] 增益，已经带上了 ``1 + w`` 偏移。
// @param stride ``x`` 行之间的元素间距；``0`` 表示 ``N``。
// @param eps 方差下限。
// @param out [M, N] 归一化结果，原地写入。不得与 ``x`` 重叠：残差和在这层剩下的部分
//        里还要活着。
// @return ``out``，便于调用链串接。
// @throws c10::Error 形状、dtype 或重叠是内核服务不了的。
torch::Tensor add_rms_norm(torch::Tensor x, torch::Tensor add, torch::Tensor w,
                           torch::Tensor out, int64_t stride, double eps);

// @brief 乘以门 SiLU 的 RMSNorm，按头归一化。
//
// @param x [M, N] 激活，连续存放，bf16 或 fp32。
// @param w [N] 按头的增益。
// @param gate [M, N] 门，布局和 dtype 同 ``x``。
// @param eps 方差下限。
// @return [M, N]，dtype 同 ``x``。
// @throws c10::Error dtype 没有对应实例化，或门的形状与 ``x`` 不一致。
torch::Tensor rms_norm_gated(torch::Tensor x, torch::Tensor w, torch::Tensor gate,
                             double eps);

// @brief ``[M, N]`` 的对称 int8 量化，每行一个 scale，一次启动。
//
// 替代那条 torch 链 ``abs -> amax -> clamp_min -> div -> round -> clamp_ -> to``：它每次
// 调用六次启动，而每个全注意力层要调两次（k 和 v）。按本卡的启动下限，这六次要花约
// 82 us，只为几微秒的算术；一次启动只花约六分之一。
//
// @param x [M, N] 待量化的 bf16 激活。
// @param out_q [M, N] int8 目的地，原地写入。
// @param out_s [M] 每行 scale 的 fp32 目的地，原地写入。
// @return ``out_q``，便于调用链串接。
// @throws c10::Error dtype 或形状与目的地不一致。
torch::Tensor kv_quant(torch::Tensor x, torch::Tensor out_q, torch::Tensor out_s);

// @brief ``out = silu(gate) * up``，即 SwiGLU 乘积，一次启动。
//
// 融合的 ``gate_up`` 投影一次读取权重就交出两半，所以它和 down 投影之间唯一的工作就是
// 这个乘积。拆成 ``aten::silu`` + ``aten::mul`` 是每层两次启动加两个
// ``[intermediate]`` 临时张量，按本卡的启动下限，代价超过算术本身。
//
// @param gate [M] 或 [M, N] 门那一半，连续存放。
// @param up 另一半，形状同 ``gate``。
// @param out 同形状的目的地。它可以与 ``up`` 重叠：乘积覆写它，之后再没人读 ``up``，
//        调用方因此既省掉分配也省掉那一次额外读取。
// @return ``out``，便于调用链串接。
// @throws c10::Error 形状、dtype 或重叠是内核服务不了的。
torch::Tensor silu_mul(torch::Tensor gate, torch::Tensor up, torch::Tensor out);

// @brief 旋转每个 q/k 头最前面 ``rotary_dim`` 个通道，一次启动。
//
// 替代 ``apply_rope`` 每层跑的那条 torch 链 ``to -> mul -> cat -> mul -> add -> cat``：
// 在 ``[16, 256]`` 和 ``[4, 256]`` 张量上约十次启动，而算术只是几千次 FMA。按本卡的
// 启动下限，那条链每个步要花掉近一毫秒，而其中几乎没多少是真正的工作。
//
// 一种形状服务两条路径：``M == 1`` 是解码，``M == T`` 是预填充分块或校验窗口，所以
// 单一内核让两者数值上完全一致。
//
// @param q [M, nh, hd] bf16 查询。
// @param k [M, nkv, hd] bf16 键。
// @param cos [M, rotary_dim] fp32，已经按被旋转的行收集好。
// @param sin 布局同 ``cos``。
// @param q_out 旋转后查询的去向，形状同 ``q``。不得与 ``q`` 重叠。
// @param k_out 旋转后键的去向，形状同 ``k``。不得与 ``k`` 重叠。
// @param rotary_dim 要旋转的通道数；尾部的 ``hd - rotary_dim`` 原样通过，这正是部分
//        旋转的 checkpoint 所要求的。
// @return ``q_out``，便于调用链串接。
// @throws c10::Error 形状、dtype 或重叠是内核服务不了的。
torch::Tensor rope_apply(torch::Tensor q, torch::Tensor k, torch::Tensor cos,
                         torch::Tensor sin, torch::Tensor q_out, torch::Tensor k_out,
                         int64_t rotary_dim);

// 在 int8 KV 缓存上做注意力：稠密、分页、批处理和草稿校验。
//
// ``part_acc``/``part_ml`` 用自身的形状携带 split-K 宽度，所以按
// ``Caps.attn_split`` 定尺寸的调用方不会和内核在 ABI 上产生分歧。

// @brief 在稠密 int8 KV 缓存上算一个解码 token。
//
// @param q [nh, hd] bf16 查询，已经旋转并缩放好。
// @param gate [nh, hd] bf16 输出门；sigmoid 在归一化之后施加。
// @param kcache,vcache [nkv, S, hd] int8，头主序。
// @param kscale,vscale [nkv, S] fp32，每个 token 一个 scale。
// @param out [nh, hd] bf16，原地写入。
// @param pos 单元素 int64 张量：token 下标兼因果上界，在设备上读取，所以被捕获的重放
//        看到的是当前的位置。
// @param rep 每个 KV 头对应的查询头数。
// @param scale ``hd ** -0.5``。
// @param part_acc [nh, split, hd] fp32，split-K 分子。
// @param part_ml [nh, split, 2] fp32，split-K 的（最大值，和）。
// @return ``out``，便于调用链串接。
torch::Tensor attn_decode(torch::Tensor q, torch::Tensor gate, torch::Tensor kcache,
                          torch::Tensor vcache, torch::Tensor kscale,
                          torch::Tensor vscale, torch::Tensor out, torch::Tensor pos,
                          int64_t rep, double scale, torch::Tensor part_acc,
                          torch::Tensor part_ml);

// @brief 在某一层的分页 int8 池上算一个解码 token。
//
// @param kpool,vpool [nslots, nkv, hd] int8，token 主序。
// @param kscale,vscale [nslots, nkv] fp32，每个槽位一个 scale。
// @param pos2slot [max_pos] int64 设备侧映射；只读前 ``pos + 1`` 项。
// @param max_pos 窗口长度，即槽位表的行间距。
// @return ``out``，便于调用链串接。
torch::Tensor attn_paged(torch::Tensor q, torch::Tensor gate, torch::Tensor kpool,
                         torch::Tensor vpool, torch::Tensor kscale,
                         torch::Tensor vscale, torch::Tensor pos2slot, torch::Tensor out,
                         torch::Tensor pos, int64_t max_pos, int64_t rep, double scale,
                         torch::Tensor part_acc, torch::Tensor part_ml);

// @brief 一个桶的行各带自身上下文长度，一次算一个解码 token。
//
// @param q [B, nh, hd] bf16 查询。
// @param slots [B, max_seq] int64，每行的槽位表。
// @param context_lens [B] int32，每行的键数。
// @return ``out`` [B, nh, hd]，便于调用链串接。
torch::Tensor attn_paged_batch(torch::Tensor q, torch::Tensor gate, torch::Tensor kpool,
                               torch::Tensor vpool, torch::Tensor kscale,
                               torch::Tensor vscale, torch::Tensor slots,
                               torch::Tensor context_lens, torch::Tensor out,
                               int64_t max_seq, int64_t rep, double scale,
                               torch::Tensor part_acc, torch::Tensor part_ml);

// @brief 投机校验：每行 ``c`` 个草稿查询共用一遍历史读取。
//
// @param q [B, c, nh, hd] bf16 草稿查询。
// @param ctx_lens [B] int32，每行的历史长度，不含草稿本身。
// @param block_mask uint8 ``[B, n_blocks]`` 保留掩码；空张量表示精确注意力。
// @return ``out`` [B, c, nh, hd]，便于调用链串接。
torch::Tensor attn_paged_multi_q(torch::Tensor q, torch::Tensor gate, torch::Tensor kpool,
                                 torch::Tensor vpool, torch::Tensor kscale,
                                 torch::Tensor vscale, torch::Tensor slots,
                                 torch::Tensor ctx_lens, torch::Tensor out,
                                 int64_t max_seq, int64_t rep, double scale,
                                 torch::Tensor part_acc, torch::Tensor part_ml,
                                 torch::Tensor block_mask, int64_t mask_block);

// 门控 DeltaNet：喂给递推的深度卷积、头预处理，以及递推本身。

// @brief 单 token 的深度因果卷积（4 抽头）加 SiLU，同时平移窗口。
//
// @param conv_w [C, 4] fp32 抽头。
// @param x [C] bf16 投影输出。
// @param conv_state [C, 3] fp32 历史，原地平移。
// @param out [C] fp32，原地写入。
void conv1d_silu(torch::Tensor conv_w, torch::Tensor x, torch::Tensor conv_state,
                 torch::Tensor out);

// @brief 同一个卷积跑在预填充分块上，窗口跨分块携带。
//
// @param x [T, C] bf16，最内层步长为 1；行间距会被正确处理。
// @param conv_state [C, 3] fp32，原地更新。
// @return [T, C] fp32。
torch::Tensor conv1d_silu_seq(torch::Tensor conv_w, torch::Tensor x,
                              torch::Tensor conv_state);

// @brief 批处理的深度因果卷积加 SiLU，每个批条目一行。
//
// @param x [B, C] bf16。
// @param conv_state [B, C, 3] fp32，原地平移。
// @param out [B, C] fp32，原地写入。
void conv1d_silu_batch(torch::Tensor conv_w, torch::Tensor x, torch::Tensor conv_state,
                       torch::Tensor out);

// @brief 一个解码步的 q/k L2 归一化、头扩展以及 beta/g 门。
//
// @param q,k [nkeys, dk] fp32；``v`` [nv, dv] fp32。
// @param b,a [nv] bf16 原始 logits。
// @param neg_expA,dt_bias [nv] fp32 checkpoint 常量。
// @param q_out,k_out [nv, dk] fp32 与 ``v_out`` [nv, dv] fp32，原地写入。
// @param beta,g [nv] fp32，原地写入。
// @param qscale ``dk ** -0.5``。
// @param rep 每个键头对应的值头数。
void delta_prologue(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor b,
                    torch::Tensor a, torch::Tensor neg_expA, torch::Tensor dt_bias,
                    torch::Tensor q_out, torch::Tensor k_out, torch::Tensor v_out,
                    torch::Tensor beta, torch::Tensor g, double qscale, int64_t rep);

// @brief 批处理版预处理：每个参数多一个前导行维度。
void delta_prologue_batch(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                          torch::Tensor b, torch::Tensor a, torch::Tensor neg_expA,
                          torch::Tensor dt_bias, torch::Tensor q_out,
                          torch::Tensor k_out, torch::Tensor v_out, torch::Tensor beta,
                          torch::Tensor g, double qscale, int64_t rep);

// @brief 预填充的头预处理，写出分块扫描读取的头主序布局。
//
// @param qkv [T, conv_dim] fp32 卷积输出，按 q | k | v 排列。
// @param q_out,k_out [nv, T, dk] fp32 与 ``v_out`` [nv, T, dv] fp32。
// @param beta,g [nv, T] fp32。
void delta_prep_seq(torch::Tensor qkv, torch::Tensor b, torch::Tensor a,
                    torch::Tensor neg_expA, torch::Tensor dt_bias, torch::Tensor q_out,
                    torch::Tensor k_out, torch::Tensor v_out, torch::Tensor beta,
                    torch::Tensor g, double qscale, int64_t rep);

// @brief 门控 delta 规则的一个解码步，原地推进状态。
//
// @param q,k [heads, dk] fp32，已经由预处理归一化过。
// @param v [heads, dv] fp32。
// @param beta,g [heads] fp32；``g <= 0``，否则递推会发散。
// @param st [heads, dk, dv] fp32，原地更新。
// @param out [heads, dv] fp32，从更新后的状态读出。
// @return ``out``，便于调用链串接。
torch::Tensor delta(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor beta,
                    torch::Tensor g, torch::Tensor st, torch::Tensor out);

// @brief 批处理 delta 步：每个批条目一行独立状态。
//
// @return ``out`` [B, heads, dv]，便于调用链串接。
torch::Tensor delta_batch(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                          torch::Tensor beta, torch::Tensor g, torch::Tensor st,
                          torch::Tensor out);

// @brief 单解码 token 的预处理、递推与门控归一化，一次启动。
//
// 替代 ``conv1d_silu`` -> ``delta_prologue`` -> ``delta`` -> ``rms_norm_gated`` 那条
// 链。那四次启动每层每个 token 要写并重读约 48 KB 的 fp32 暂存（``q_pro``/``k_pro``/
// ``v_pro`` = 3 x [32, 128]，``delta_out`` = [32, 128]），按本卡约 5.75 us 的启动下限，
// 这些往返比算术本身更贵。每个值头一个 block 把全部中间结果留在共享内存和寄存器里，
// 而门控归一化的归约正好只跨一个头的 ``dv``，所以它免费折进尾声。
//
// @param conv ``[nv, dv]`` fp32，本 token 的卷积加 SiLU 输出。该缓冲区的 q、k 两段按列
//        读取，每头 ``dk`` 个，所以布局是 q | k | v 且 ``nv == nkeys * rep``；头把它的
//        键头重复 ``rep`` 次。调用方分三段传入而不是传一整行，因为卷积输出宽
//        ``conv_dim``，而这里每段只读它的头部列。
// @param b,a ``[nv]`` bf16 原始 beta／衰减 logits。
// @param neg_expA,dt_bias ``[nv]`` fp32，衰减门的 checkpoint 常量。
// @param z ``[nv, dv]`` fp32 或 bf16，输出归一化的 SiLU 门投影。
// @param nw ``[dv]`` bf16 门控归一化增益，按值头而不是按展平宽度。
// @param st ``[nv, dk, dv]`` fp32 循环状态，原地更新。
// @param out ``[nv, dv]`` fp32，从更新后的状态写出，且已经归一化。
// @param qscale ``dk ** -0.5``。
// @param rep 每个键头对应的值头数。
// @return ``out``，便于调用链串接。
// @throws c10::Error dtype、头数或布局与状态矛盾。
torch::Tensor delta_normed(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                           torch::Tensor b, torch::Tensor a,
                           torch::Tensor neg_expA, torch::Tensor dt_bias,
                           torch::Tensor z, torch::Tensor nw, torch::Tensor st,
                           torch::Tensor out, double qscale, int64_t rep);

// 三角求解：分块 delta 规则唯一需要的库操作，也是本卡 cuSOLVER 答不了的那一个。

// @brief 为单位下三角 ``A`` 解 ``A X = B``。
//
// @param lower [bn, L, L] 单位下三角系数；对角线按 1 处理，上三角永不读取。
// @param rhs [bn, L, n] 右端项，dtype 同 ``lower``。
// @return [bn, L, n]，dtype 同 ``rhs``。
// @throws c10::Error 形状或布局是内核索引不了的，包括 ``L`` 大到装不进共享内存。
torch::Tensor triangular_solve(torch::Tensor lower, torch::Tensor rhs);
