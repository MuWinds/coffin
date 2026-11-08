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
// 一次 c-token 投机校验打不过 c 个普通步的原因；见 ``docs/qwen4/QWEN4-PERF.md``。
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

torch::Tensor gemv_kn_w4(torch::Tensor packed, torch::Tensor scale, torch::Tensor x,
                         int64_t k, int64_t group_size);

// @brief 分组 W4 与 int8 激活的整数投影，返回调用方提供的浮点输出。
//
// @param a [M,K] 连续 int8 激活，已由 ``quant_act_i8`` 归一化。
// @param w [至少 ceil(K/2),N] 连续 uint8 packed 权重。
// @param xs [M] fp32 激活 scale。
// @param ws [ceil(K/group_size),N] fp32 权重 scale。
// @param out [M,N] 连续 bf16 或 fp32 输出。
// @param group_size K 轴权重分组大小。
// @param threads 每块线程数，``0`` 表示按 ``N`` 选。
// @param unroll 每个分片内 K 循环的展开度，``1``/``2``/``4``；``0`` 表示 1。
//        ``1`` 精确复现引入这个参数之前的核，所以默认路径逐位不变。
// @param split 分片数，``0`` 表示按 ``N``/``K`` 选；``1`` 强制单趟。
// @param width 一个线程负责的相邻输出列数：``4`` 时每个 k 步读一个对齐的 32 位字
//        而不是 4 个单字节。合法值是 ``1``、``2``、``4``；``N % width`` 必须为 ``0``，
//        不满足时折回 ``1``。``0`` 表示自动，而自动目前就是 ``1``——宽加载的交叉点
//        只能由真机扫描给出，见 ``w4a8.cu`` 的 ``pick_cw``。
// @return out。
// @throws c10::Error 布局、设备、dtype、形状或整数累加范围不符合契约。
torch::Tensor w4a8_gemv(torch::Tensor a, torch::Tensor w, torch::Tensor xs,
                        torch::Tensor ws, torch::Tensor out, int64_t group_size,
                        int64_t threads, int64_t unroll, int64_t split,
                        int64_t width);
torch::Tensor w4a8_gemm_tcu(torch::Tensor a, torch::Tensor w, torch::Tensor xs,
                            torch::Tensor ws, torch::Tensor out, int64_t group_size,
                            int64_t m_tile = 0, int64_t db = 0);

// @brief 把 packed W4 权重一趟反量化成 bf16，供大 M 的厂商 GEMM 使用。
//
// torch 层的等价写法要七八趟 elementwise、每趟把 79 MB 完整读写一遍，实测只有约
// 35 GB/s；合成一趟是这个内核存在的全部理由（``w4_dequant.cu`` 头部有完整账）。
//
// @param packed ``[ceil(K/2), N]`` 连续 uint8（**K 在行、N 在列**）。
// @param scale  ``[ceil(K/group), N]`` 连续 fp32。
// @param k      输入宽度，允许奇数。
// @param n      输出列数，必须是 4 的倍数（一列四字节的事务宽度决定的）。
// @param group  分组大小。
// @return ``[K, N]`` bf16。
// @throws c10::Error 布局、dtype、形状或 ``N % 4`` 不自洽。
torch::Tensor w4_dequant(torch::Tensor packed, torch::Tensor scale, int64_t k, int64_t n,
                         int64_t group);
torch::Tensor quant_act_i8_scaled(torch::Tensor x, torch::Tensor out, torch::Tensor xs);

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

// @brief 零中心 RMSNorm，结果直接量化成 int8，一次启动——预填充 W8A8 的激活入口。
//
// 预填充的每个层有两处这条路径(in_ln 和 post_ln)，每步共 64 次，比 add_rms_norm_i8
// 更常走。与 `rms_norm` 写 bf16 再接 `quant_act_i8` **逐位相同**，但省掉两趟
// global 往返:拆开是写 bf16(2 B/元素) + 读回(2 B) + 写 int8(1 B)，融合后只剩 1 B。
//
// @param x [M, N] 激活，行间距 `stride`，可以大于 `N`。
// @param w [N] 增益，已带 `1 + w` 偏移。
// @param out_q [M, N] int8 目的地，稠密。
// @param out_s [M] 每行 fp32 scale 目的地。
// @param stride `x` 的行间距；`0` 表示 `N`。
// @param eps 方差下限。
// @param threads_override 每块线程数；`0` 用按 `M` 分档的实测最优点。
// @return `out_q`。
// @throws c10::Error dtype、形状、间距矛盾，或 LDS 装不下这一行。
torch::Tensor rms_norm_i8(torch::Tensor x, torch::Tensor w, torch::Tensor out_q,
                          torch::Tensor out_s, int64_t stride, double eps,
                          int64_t threads_override = 0);

// @brief 残差相加接零中心 RMSNorm，并把结果直接量化成 int8，一次启动。
//
// W8A8 预填充的入口：下一个消费者是 int8 GEMM，它要 int8 激活加每行一个 fp32 scale。
// 拆开的写法要走三趟 global（写 bf16 2 B → 读 bf16 2 B → 写 int8 1 B，共 5 B/元素），
// 融合后只有 1 B/元素。2048 行的分块上那是 3.8 GB 的无效往返，实测 ``quant_act_i8``
// 单独跑就要 10.34 ms/步。
//
// 口径要求：与 ``add_rms_norm`` + ``quant_act_i8`` 拆开跑**逐位相同**。和仍然先舍入成
// bf16 再进均方，absmax 仍然取自同一个归一化结果，量化仍是 round 后夹到 [-127, 127]。
//
// @param x [M, N] 残差流，**原地累加**，调用方必须独占。
// @param add [M, N] 待累加的混合器输出，行间距 ``stride``。
// @param w [N] 增益，已带 ``1 + w`` 偏移。
// @param out_q [M, N] int8 目的地，稠密。
// @param out_s [M] 每行 fp32 scale 目的地。
// @param stride ``x``/``add`` 的行间距；``0`` 表示 ``N``。
// @param eps 方差下限。
// @return ``out_q``。
// @throws c10::Error dtype、形状或间距与内核矛盾。
torch::Tensor add_rms_norm_i8(torch::Tensor x, torch::Tensor add, torch::Tensor w,
                              torch::Tensor out_q, torch::Tensor out_s, int64_t stride,
                              double eps, int64_t threads_override = 0);

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

// W8A8 预填充投影：权重与激活都是 int8，int32 累加，epilogue 里乘两个 scale 再窄化一次。

// @brief 把一个 ``[M, K]`` 激活按行（per-token）对称量化成 int8。
//
// W8A8 的第一个内核。为什么按行而不是按张量：一行就是一个 token，行与行之间没有
// 依赖，所以这个 reduction 只在 K 上做、不需要跨块通信，而按张量缩放会把某个 token
// 的溢出摊到所有 token 上。``1e-5`` 的下限让全零行得到有限 scale 而不是 inf。
//
// 与 ``kv_quant`` 是同一套量化约定（`absmax / 127`、夹 `1e-5`、值域 `[-127, 127]`），
// 故意复用而不是另立一套：两条路径的激活必须能互相替代，否则调完 prefill 再回头
// 看 decode 时，两个 "int8" 指的就不是同一个数。
//
// @param x [M, K] bf16 或 fp32 激活。
// @param out_q [M, K] int8 目的地，原地写入。
// @param out_s [M] 每行 scale 的 fp32 目的地，原地写入。
// @return ``out_q``，便于调用链串接。
// @throws c10::Error dtype、形状或连续性与目的地不一致。
torch::Tensor quant_act_i8(torch::Tensor x, torch::Tensor out_q, torch::Tensor out_s);

// @brief int8 GEMM：``C[M, N] = A[M, K] @ W[K, N]``，int32 累加。
//
// 本卡实测 int8 GEMM 49.07 TINT8-OPS/s，bf16 42.71 TFLOP/s（27B-4CARD-PLAN.md §3.2），
// 所以 W8A8 在预填充上是真快 1.27x，而权重全程 int8 还省一半显存。走库而不是手写：
// 16 个 SM 上很难超过厂商的 49 TINT8-OPS/s，``__dp4a`` 可用只说明"将来要自己写融合
// 内核有指令基础"。
//
// 库只在已知算得对的宽度内调用：``N > 131072`` 实测返回 NaN，所以调用方按 N 切块。
// 这个上限是卡的性质，不是可调参数，检查放在 Python 侧（``pack.LIB_MAX_N``），这里只
// 断言自己能索引的形状。
//
// @param a [M, K] int8 激活，行主序连续。
// @param w [K, N] int8 权重，行主序连续。
// @param c [M, N] int32 累加器目的地，原地写入。
// @return ``c``。
// @throws c10::Error dtype 不是 int8、维度不是二维，或形状对不上 K。
torch::Tensor w8a8_gemm(torch::Tensor a, torch::Tensor w, torch::Tensor c);

// @brief W8A8 的 epilogue：`out = bf16(c[m, n] * xs[m] * ws[n])`，一次窄化。
//
// 两个 scale 必须在同一个 fp32 表达式里乘完再窄化。分开两次窄化会把结果多舍入一次，
// 而这条路要喂 32 层——W8A16 那边为了同一个理由拒绝把 scale 先窄化成 bf16。
//
// 为什么是独立的 kernel 而不是 GEMM 的一部分：库的 int8 GEMM 只写 int32，没有
// epilogue 挂点，而 ``cublasLt`` 的 epilogue 不覆盖 per-token 激活 scale。三次启动
// 的代价已经计入 27B-4CARD-PLAN.md §3.3 的 8.535 ms。
//
// @param c [M, N] int32 GEMM 输出。
// @param xs [M] 每 token 的激活 scale。
// @param ws [N] 每输出列的权重 scale。
// @param out [M, N] 目的地，dtype 是调用方算术用的浮点类型（bf16 或 fp32）。
// @param stride ``out`` 的行间距；``0`` 表示等于 N。宽投影沿 N 切块时用得上。
// @return ``out``。
// @throws c10::Error 形状、dtype 或步长与内核能索引的不一致。
torch::Tensor w8a8_epilogue(torch::Tensor c, torch::Tensor xs, torch::Tensor ws,
                            torch::Tensor out, int64_t stride);

// @brief 用宽指令 i8x8 的张量核 int8 GEMM，epilogue 融合，无 int32 往返。
//
// 与 :func:`w8a8_gemm` + :func:`w8a8_epilogue` 这条库路径做同一件事，但走的是
// 指令级更宽的通路，且累加器留在寄存器里直接乘 scale 出浮点：库的
// ``cublasGemmEx`` 只把 int32 写回 global，没有任何 epilogue 挂点，所以那条路
// 必须付一次 ``[M, N]`` int32 的写 + 读，这里不用。
//
// 这条内核只在宽形状上赢库（实测 4B 的真实预填充形状：``gate_up`` 1.52x、
// ``linear_attn_qkvzba`` 1.53x、``o_proj`` 1.28x），窄 K 的 ``mlp_down``
// （K=9216）反而输 14%，所以调用方按形状分派，不是无条件替换。
//
// @param a ``[M, K]`` int8 激活，行主序连续。
// @param w ``[K, N]`` int8 权重，行主序连续。两个操作数都按引擎的原生布局吃，
//        调用方**不需要**转置。
// @param xs ``[M]`` 每 token 的 fp32 激活 scale。
// @param ws ``[N]`` 每个输出列的 fp32 权重 scale。
// @param out ``[M, N]`` 目的地，bf16 或 fp32，原地写入。
// @return ``out``。
// @throws c10::Error dtype、形状或连续性与内核索引的不一致。
torch::Tensor w8a8_gemm_tcu(torch::Tensor a, torch::Tensor w, torch::Tensor xs,
                            torch::Tensor ws, torch::Tensor out);

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

// Gated Residua）：Qwen4-Exp 每层两套，48 层就是 96 次调用。
// 参考实现里每套是八次 torch 调用，光发射就与本层权重带宽同量级，所以融合。

// @brief 对残差流做分组 RMSNorm：每 `hidden` 个元素一组，各自归一化。
//
// @param x   残差流，`[rows, hidden * hc]` 连续，bf16 或 fp32。
// @param w   零中心权重，`[hidden * hc]` fp32，前向按 `1 + w`。
// @param out 目标缓冲，`[rows, hidden * hc]` 连续，dtype 同 `x`。
// @param hidden 每流宽度。
// @param hc     流数，上限 8。
// @param eps    方差下限。
// @return out。
// @throws c10::Error 布局、dtype、形状或组数不符合契约。
torch::Tensor hyper_norm(torch::Tensor x, torch::Tensor w, torch::Tensor out,
                         int64_t hidden, int64_t hc, double eps);

// @brief down 投影加 `silu(v / hc)`：`[rows, K] -> [rows, lowrank]`。
//
// @param normed 归一化后的激活，`[rows, K]` 连续。
// @param wt     down 权重，`[K, lowrank]` 连续，dtype 同 `normed`。
// @param out    `[rows, lowrank]` 连续目标。
// @param hc     读门的除数。
// @return out。
// @throws c10::Error 布局、dtype 或形状不符合契约。
torch::Tensor hyper_down(torch::Tensor normed, torch::Tensor wt, torch::Tensor out,
                         int64_t hc);

// @brief up 投影加读门塌缩，同时算写门。
//
// @param normed 归一化后的激活，`[rows, hidden*hc]` 连续。
// @param up_wt  up 权重，`[lowrank, hidden*hc]` 连续。
// @param low    down 的输出，`[rows, lowrank]` 连续。
// @param inj_wt 写门权重，`[hidden*hc, hc]` 连续，dtype 同 `normed`。
// @param mixed  `[rows, hidden]` 连续目标：读门塌缩结果。
// @param inject `[rows, hc]` 连续 fp32 目标：写门。
// @param hidden 每流宽度。
// @param hc     流数。
// @param lowrank down 的秩。
// @return mixed。
// @throws c10::Error 布局、dtype 或形状不符合契约。
torch::Tensor hyper_up(torch::Tensor normed, torch::Tensor up_wt, torch::Tensor low,
                       torch::Tensor inj_wt, torch::Tensor mixed, torch::Tensor inject,
                       int64_t hidden, int64_t hc, int64_t lowrank);

// 稀疏专家的分组投影：每个专家只算路由到它的行。与单投影分开，因为它的并行维
// 是"专家 × 行"，切法完全不同。

// @brief 分组 W4A8 投影：``[tokens, K] -> [tokens, N]``，每专家独立权重。
//
// @param a        已按专家号排序的 int8 激活，``[tokens, K]`` 连续。
// @param xs       逐 token 激活 scale，``[tokens]`` fp32。
// @param w        专家栈 packed W4，``[E, ceil(K/2), N]`` 连续 uint8。
// @param ws       专家栈 scale，``[E, ceil(K/group), N]`` 连续 fp32。
// @param offsets  ``[E+1]`` 前缀和，int32；空段表示该专家没有行。
// @param out      ``[tokens, N]`` 连续，bf16 或 fp32。
// @param group_size K 轴权重分组大小。
// @param active   可选，``[n_active]`` int32 非空专家号；``None`` 表示按全量 E 开
//                 z 维。decode 一个 token 只碰约 10 个专家而每卡持 128 个，
//                 给了它网格的 z 维就从 128 降到 10，消掉 92% 的空转 block。
// @return out。
// @throws c10::Error 布局、设备、dtype、形状或前缀和不自洽。
//
// 本实现内部按**每专家行数**分两档，调用方不需要知道：每专家 1 行（B=1 decode，用
// ``tokens <= top_k`` 在 host 侧判定，不读 ``offsets``）走「按 weight-group 切 K +
// 部分和归约」的分片内核，实测 203.3 -> 62.9 us（3.23x，作业 213，逐位相同）；其余
// 走单趟内核 —— 分片在每专家 4 行以上反而慢 26%，因为部分和与归约是固定成本。
// 判据、实测数字与取舍见 ``moe_grouped.cu`` 里 ``moe_grouped`` 的启动段。
torch::Tensor moe_grouped(torch::Tensor a, torch::Tensor xs, torch::Tensor w,
                          torch::Tensor ws, torch::Tensor offsets,
                          torch::Tensor out, int64_t group_size,
                          c10::optional<torch::Tensor> active);

// 分派：把路由结果变成分组内核要的前缀和与行序，替掉纯 torch 的一串小算子。

// @brief 单 block 完成「计数 → 前缀和 → 稳定算位置」。
//
// 一步 decode 里这段原本是 ``topk``/``div``/``nonzero``/``argsort``/``bincount``/
// ``cumsum``/``chunk`` 等十来个 torch 算子 x 48 层 = 672 次内核。按本卡 5.76 us 的
// 发射地板是 3.87 ms/步，而这些算术的输入只有 ``slots`` 个整数。
//
// @param expert_idx   ``[slots]`` 展平的全局专家号，int32 连续。
// @param local_table  ``[num_experts]`` 全局号 → 本地号，int32，不属于本 rank 为 -1。
// @param out_offsets  ``[local_experts + 1]`` int32 输出，前缀和。
// @param out_slot_pos ``[slots]`` int32 输出，收拢后的行号，丢弃的为 -1。
// @param out_tokens   ``[slots]`` int32 输出，收拢后每行的 token 下标。
// @param out_slots    ``[slots]`` int32 输出，收拢后每行来自哪个槽位。
// @param scratch      ``[local_experts]`` int32 暂存，调用方持有可复用。
// @param slots, local_experts, num_experts, top_k 几何。
// @return 无。
// @throws c10::Error 布局、设备、dtype 或形状不自洽。
void moe_dispatch(torch::Tensor expert_idx, torch::Tensor local_table,
                  torch::Tensor out_offsets, torch::Tensor out_slot_pos,
                  torch::Tensor out_tokens, torch::Tensor out_slots,
                  torch::Tensor scratch, int64_t slots, int64_t local_experts,
                  int64_t num_experts, int64_t top_k);

// 分组 MoE 的大 M 形态：按专家分段各做一次张量核 GEMM（prefill 用）。

// @brief 分组 W4A8 GEMM：``[tokens, K] -> [tokens, N]``，按专家分段。
//
// 与 :func:`moe_grouped` 的区别只在用哪类内核：那个是 GEMV 结构、适合 decode 时
// 每专家 1-2 行；这个是每段一次张量核 GEMM、适合 prefill 时每专家上百行。
// 实测 ``seq=2048`` 时 GEMV 结构 22.7 ms/次（占 prefill 30.4%），而同类工作的
// GEMM 路径是 2.30 ms/次。
//
// @param a         已按专家号排序的 int8 激活，``[tokens, K]`` 连续。
// @param xs        逐 token 激活 scale，``[tokens]`` fp32。
// @param w         专家栈 packed W4，``[E, ceil(K/2), N]`` 连续 uint8。
// @param ws        专家栈 scale，``[E, ceil(K/group), N]`` 连续 fp32。
// @param offsets   ``[E+1]`` 前缀和，int32；**本函数在主机侧读它**。
// @param out       ``[tokens, N]`` 连续输出，调用方负责先清零。
// @param group_size K 轴分组，必须是 64 的倍数。
// @return out。
// @throws c10::Error 布局、设备、dtype、形状或对齐不自洽。
void moe_grouped_gemm(torch::Tensor a, torch::Tensor xs, torch::Tensor w,
                      torch::Tensor ws, torch::Tensor offsets,
                      torch::Tensor out, int64_t group_size);

// QSA 的稀疏注意力：只读索引器选中的位置。

// @brief 稀疏精确 softmax 注意力，KV 不物化、不扫全窗口。
//
// 替掉 ``qsa.py`` 里那三段纯 torch（``repeat_kv`` + ``matmul`` + ``softmax`` +
// ``matmul``）。那三段在 kv=32768 时是 16.68 ms/层/步，
// 其中 ``matmul`` 占 55%，而它扫的是整个 ``kv_length``，索引器预算只有 2048，
// 所以 94% 的打分作用在已经被判定为「不看」的位置上。本内核只遍历选中的位置。
//
// 顺带省掉 ``repeat_kv``：GQA 下按 ``g = h / rep`` 直接寻址 KV，不把 2 个 KV 头
// 物化成 16 份（那一条在 kv=32768 时是 1.67 ms/层/步）。
//
// @param q        ``[rows, nh, hd]`` bf16 连续。
// @param kcache   ``[nkv, max_seq, hd]`` bf16 连续（头主序，未 repeat）。
// @param vcache   同 ``kcache``。
// @param sel_idx  ``[rows, max_sel]`` int32 连续，行内升序，无效为 -1。
// @param sel_len  ``[rows]`` int32 连续。
// @param ctx      ``[rows]`` int32 连续，每行的因果上界。
// @param out      ``[rows, nh, hd]`` bf16 连续。
// @param scale    打分缩放。
// @return out。
// @throws c10::Error 布局、设备、dtype 或形状不自洽。
torch::Tensor attn_sparse(torch::Tensor q, torch::Tensor kcache, torch::Tensor vcache,
                          torch::Tensor sel_idx, torch::Tensor sel_len,
                          torch::Tensor ctx, torch::Tensor out, double scale);

// QSA 索引器的块 key 融合。

// @brief gather → 池化 → RMSNorm → partial RoPE，一次启动。
//
// 替掉 ``qsa.py`` 的 ``_batched_block_keys`` 那串 torch 算子。作业 ``q4sel1``
// 量到这一段在 kv=32768 时是 1.067 ms/层/步（12 层 12.8 ms），而它要读的数据只有
// ``max_blocks x ratio x head_dim`` 个 bf16 = 8.4 MB，按 575 GB/s 的屋顶应是
// 0.015 ms —— 差 71 倍，差在那串中间量上（gather、fp32 池化、layernorm 的多次
// elementwise、RoPE 的 cat），每一步都是一次全量读写。
//
// 三处容易写错的语义见 ``qsa_block_keys.cu`` 的文件头：RMSNorm 的增益是
// ``1 + weight``（零中心）、均值在 fp32 里算再舍入到 bf16、RoPE 是半切分形式。
//
// @param flat_keys ``[batch * tokens, head_dim]`` bf16 连续。
// @param table     ``[rows, max_blocks, ratio]`` int32 连续，``-1`` 是补位。
// @param weight    ``[head_dim]`` bf16 连续，零中心增益。
// @param flat_cos  ``[batch * tokens, rotary_dim]`` bf16 连续。
// @param flat_sin  同 ``flat_cos``。
// @param block_row ``[rows * max_blocks]`` int32，块行到 query 行的映射。
// @param block_batch ``[rows * max_blocks]`` int32，块行到 batch 行的映射。
// @param out       ``[rows, max_blocks, head_dim]`` bf16 连续。
// @param seq       每个 batch 行的 query 数。
// @param tokens    每行的总 token 数。
// @param rotary_dim 参与旋转的维数。
// @param eps       RMSNorm 的 epsilon。
// @return out。
// @throws c10::Error 布局、dtype、形状或 ``ratio`` 越界。
torch::Tensor qsa_block_keys(torch::Tensor flat_keys, torch::Tensor table,
                             torch::Tensor weight, torch::Tensor flat_cos,
                             torch::Tensor flat_sin, torch::Tensor block_row,
                             torch::Tensor block_batch, torch::Tensor out,
                             int64_t seq, int64_t tokens, int64_t rotary_dim,
                             double eps);

// 路由：从一行 logits 里选 top_k 个专家，并给出（可归一化的）权重。

// @brief 融合的 ``softmax(fp32) -> top-k -> 归一化``，一次发射写完下标与权重。
//
// 为什么新增这个入口：``moe.py`` 的 ``TopKRouter`` 这一步原本是
// ``softmax -> topk -> div -> sum -> to`` 五个 torch 内核，在 decode 一步里 48 层
// 各来一遍。其中 ``aten::topk`` 单次 38 us（作业 ``q4attrib3``，48 次/步 = 1.83 ms/步），
// 而对 ``[1, 512]`` 取 top-10 在本卡上走的是 bitonic 排序网络 —— 45 轮比较交换，
// 只为挑出最大的 10 个。这条路径给出的是「一次发射、不排序」的替代。
//
// 语义与参考实现（``torch.softmax`` + ``torch.topk`` + 归一化）对齐：fp32 softmax 先减
// 行内最大值、按概率降序、等值取小下标（``torch.topk`` 对等值次序不作保证，这里固定
// 成小下标以便同一输入两次运行逐位相同）。
//
// @param logits    ``[tokens, n]`` 连续 CUDA bf16 或 fp32 logits。
// @param out_idx   ``[tokens, top_k]`` int64 目的地，按概率降序。
// @param out_w     ``[tokens, top_k]`` 目的地，dtype 必须与 ``logits`` 相同。
// @param top_k     每 token 的专家数，``(0, 32]`` 且不超过 ``n``。
// @param normalize 是否把 top-k 概率再归一化（``norm_topk_prob``）。
// @return 无。两个目的地原地写入。
// @throws c10::Error 布局、设备、dtype、形状或 ``top_k`` 越界。
void moe_router(torch::Tensor logits, torch::Tensor out_idx, torch::Tensor out_w,
                int64_t top_k, bool normalize);

// @brief 探测本设备是否支持映射锁页内存（零拷贝），并做一次核写-主机读的往返。
//
// 设备侧归约的前提：核能直接寻址锁页缓冲。若能，跨 rank 归约可以省掉每层一次
// 阻塞 D2H（实测代价 37.98 ms，见 ``mapped_shm.cu`` 文件头）；若不能，那条路走不通。
//
// @param detail 输出缓冲，写入人类可读的结论（含失败原因或见过的魔数）。
// @param detail_len ``detail`` 的字节数。
// @return ``1`` 支持且往返成功；``0`` 不支持或往返失败。
int mapped_pinned_probe(char* detail, int detail_len);
