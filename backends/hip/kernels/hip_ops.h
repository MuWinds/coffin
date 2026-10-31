#pragma once

// @file hip_ops.h
// @brief 在 bindings.cpp 里绑定的 HIP 扩展入口的声明。
//
// Module: coffin/kernels。内核与 ``hip_ops.py`` 共享的唯一头文件；它是纯接口，所以必须
// 不含内核实现，也不含任何会把第二个 HIP 编译单元拖进构建的东西。
//
// Constraints:
//   * 这里每个函数只声明一次，并在某个 ``.cu`` 里定义；改签名必须两边一起改，否则扩展是
//     链接失败而不是启动失败。
//   * 权重是 INT8 行主序 [N, K] 配每行一个 fp32 scale。新的量化布局会弄坏
//     ``gemv_w8a16*`` 和每个调用方的预打包 checkpoint，所以它需要新的入口。
//   * 形状是契约的一部分：内核索引时不做边界检查，主机侧包装函数只 TORCH_CHECK 它本来就
//     写来检查的那些条件。
//   * 部分和缓冲区（``part_acc`` / ``part_ml``）归调用方所有，形状因此为图捕获保持静态；
//     不要在算子内部自行分配。

#include <torch/extension.h>

// @brief 单 token bf16 GEMV：y[n] = dot(W[n, :], x)。
//
// @param W [N, K] bf16，行主序。
// @param x [K] bf16 激活。
// @param R 每个 warp 的输出行数；Python 侧在加载时按形状把 (R, U) 组合基准测试一遍。
// @param U K 循环展开因子，同样是基准测试决定。
// @return [N] bf16，dtype 与激活一致。
torch::Tensor gemv_bf16(torch::Tensor W, torch::Tensor x, int64_t R, int64_t U);

// @brief 同一族内核，权重 fp16、激活 bf16，返回 bf16。
//
// gfx906 有 packed fp16 数学而没有 packed bf16 数学，所以权重用 fp16：拷贝循环两边都是
// 带宽受限，但读同一批张量的预填充 GEMM 会快约 7 倍（宽 N 形状上 20 对 2.3 TFLOP/s）。
//
// @param W [N, K] fp16，行主序。
// @param x [K] bf16；输出仍是 bf16，这样激活流在层与层之间不必换 dtype。
// @param R 每个 warp 的输出行数，含义同 ``gemv_bf16``。
// @param U K 循环展开因子。
// @return [N] bf16。
torch::Tensor gemv_f16(torch::Tensor W, torch::Tensor x, int64_t R, int64_t U);

// @brief 批处理小 M GEMV：每个权重行一个 warp，保留 M 个累加器。
//
// 该行对全部 M 个 token 只读一次，k 个 token 的投机校验因此只花约一次权重读取而不是 k 次。
//
// @param W [N, K] bf16，行主序。
// @param X [M, K] bf16。
// @param U K 循环展开因子。
// @return [M, N] bf16。
torch::Tensor gemv_bf16_m(torch::Tensor W, torch::Tensor X, int64_t U);

// @brief 权重为 fp16 的批处理小 M GEMV（进出激活都是 bf16）。
//
// 理由同 ``gemv_f16``：fp16 权重让预填充 GEMM 直接复用同一批张量，不必逐层重新打包。
//
// @param W [N, K] fp16，行主序。
// @param X [M, K] bf16。
// @param U K 循环展开因子。
// @return [M, N] bf16。
torch::Tensor gemv_f16_m(torch::Tensor W, torch::Tensor X, int64_t U);

// @brief 解码用的纯权重 int8 GEMV（W8A16）。
//
// 只读 ``gemv_bf16`` 一半的权重字节，这是解码变成带宽受限之后仅剩的杠杆；激活保持 bf16，
// 于是路径里不会混进第二个量化器（及其误差）。
//
// @param W [N, K] int8，行主序。
// @param S [N] fp32 每行反量化 scale，在归约之后统一乘一次。
// @param x [K] bf16。
// @param R 每个 warp 的输出行数；(R, U) 两个旋钮的含义同 ``gemv_bf16``。
// @param U K 循环展开因子。
// @return [N] bf16。
torch::Tensor gemv_w8a16(torch::Tensor W, torch::Tensor S, torch::Tensor x,
                         int64_t R, int64_t U);

// @brief 批处理小 M int8 GEMV：一次权重读取服务 M 个激活向量。
//
// 形状契约与 ``gemv_w8a16`` 相同，只是批维挪到了最前面。
//
// @param W [N, K] int8，行主序。
// @param S [N] fp32 每行反量化 scale。
// @param X [M, K] bf16；M 必须是已实例化的取值之一 {2, 4, 8}。
// @param U K 循环展开因子。
// @return [M, N] bf16。
torch::Tensor gemv_w8a16_m(torch::Tensor W, torch::Tensor S, torch::Tensor X, int64_t U);

// @brief 一步 gated delta 规则的解码，也就是全部 24 个线性注意力层的数学。
//
// @param q,k [heads, d] fp32，已做 L2 归一化；q 还额外缩放过。
// @param v [heads, d] fp32。
// @param beta,g [heads] fp32 的写入速率与衰减，由 ``delta_prologue`` 构建。
// @param st [heads, dk, dv] fp32 循环状态，原地更新——前缀复用所依赖的快照纪律由调用方
//        负责。
// @param out [heads, dv] fp32，返回的就是这个缓冲区。
// @return ``out``，[heads, dv] fp32。
torch::Tensor delta_step(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                         torch::Tensor beta, torch::Tensor g,
                         torch::Tensor st, torch::Tensor out);

// @brief 融合在 int8 KV 缓存上的单 token 全注意力。
//
// 边读边反量化 k/v，把逐 token 的 scale 折进分数/概率，在 ``pos`` 处做因果掩码，softmax
// 之后再施加输出门控。
//
// @param q,gate [nh, hd] bf16；gate 的 sigmoid 折进输出。
// @param kcache,vcache [nkv, S, hd] int8 头主序；``kscale``/``vscale`` 是
//        [nkv, S] fp32。
// @param out [nh, hd] bf16，返回的就是这个缓冲区。
// @param pos 设备侧 int64 张量：解码图在捕获时会把标量内核参数冻住，所以位置必须在重放时
//        重新读取。
// @param rep 每个 key 头对应的 value 头数，即 GQA 组大小。
// @param scale softmax 缩放，调用方已经把 1/sqrt(head_dim) 折进去了。
// @param part_acc,part_ml split 那一遍的 online-softmax 部分和，[nh, split, hd] 与
//        [nh, split, 2] fp32。归调用方所有，形状因此为图捕获保持静态
//        （``split = part_acc.size(1)``）。
// @return ``out``，[nh, hd] bf16。
torch::Tensor attn_decode(torch::Tensor q, torch::Tensor gate,
                          torch::Tensor kcache, torch::Tensor vcache,
                          torch::Tensor kscale, torch::Tensor vscale,
                          torch::Tensor out, torch::Tensor pos,
                          int64_t rep, double scale,
                          torch::Tensor part_acc, torch::Tensor part_ml);

// @brief 同样的注意力，但跑在*分页*池上。
//
// @param q,gate [nh, hd] bf16。
// @param kpool,vpool 某一层对块池的视图，[num_slots, nkv, hd] int8 配
//        [num_slots, nkv] fp32 scale。
// @param pos2slot 设备侧映射：绝对位置到存着该行的池槽位。它放在设备上，捕获的解码图才能
//        在追加块之后继续用：主机侧的表每追加一次就得重新捕获。
// @param out [nh, hd] bf16，返回的就是这个缓冲区。
// @param pos 设备侧 int64 位置，理由同 ``attn_decode``。
// @param max_pos 窗口改为流式处理之后已不再使用；保留它只是为了让调用方签名不变。
// @param rep 每个 key 头对应的 value 头数。
// @param scale softmax 缩放。
// @param part_acc,part_ml 调用方持有的 split 部分和，同 ``attn_decode``。
// @return ``out``，[nh, hd] bf16。
torch::Tensor attn_paged_decode(torch::Tensor q, torch::Tensor gate,
                                torch::Tensor kpool, torch::Tensor vpool,
                                torch::Tensor kscale, torch::Tensor vscale,
                                torch::Tensor pos2slot, torch::Tensor out,
                                torch::Tensor pos, int64_t max_pos,
                                int64_t rep, double scale,
                                torch::Tensor part_acc, torch::Tensor part_ml);

// @brief 融合 RMSNorm：y = x * rsqrt(mean(x^2) + eps) * w。
//
// 每行一个 block 把归约、rsqrt 和乘权重融在一起，值得这么做是因为引擎每个解码步要调它 105
// 次。
//
// @param x [M, N] bf16。
// @param w [N] bf16 权重。
// @param stride ``x`` 的行距：q_norm 跑在切片上，所以行不是紧密排布的。
// @param eps 在 rsqrt 之前加到均方上；它是 checkpoint 的 epsilon，不是调优旋钮。
// @return [M, N] bf16。
torch::Tensor rms_norm(torch::Tensor x, torch::Tensor w, int64_t stride, double eps);

// @brief 带门控的 RMSNorm：y = rms(x) * w * silu(gate)。
//
// @param x,w,gate [M, N] bf16 连续存放；门控以 SiLU 形式加在归一化乘权重之后。
// @param eps 与 ``rms_norm`` 相同的 epsilon 契约。
// @return [M, N] bf16。
torch::Tensor rms_norm_gated(torch::Tensor x, torch::Tensor w, torch::Tensor gate,
                             double eps);

// @brief 解码步的融合 gated delta 规则 prologue。
//
// 逐头对 q/k 做 L2 归一化（q 乘 ``qscale``），按 ``rep`` 把 key 头扩成 value 头，把 v 转成
// fp32，并构建 beta = sigmoid(b) 与 g = -exp(A) * softplus(a + dt_bias)。用 torch 算子做
// 这堆事相当于每层约 16 个小内核。
//
// @param q_raw,k_raw [nk, dk] fp32，即逐行的卷积输出。
// @param v_raw [nv, dv] fp32。
// @param b,a [nv] bf16 的原始投影输出，只用来取标量因子。
// @param neg_expA,dt_bias [nv] fp32 的 checkpoint 常量，用于衰减门控。
// @param q_out,k_out,v_out [nv, dk] / [nv, dv] fp32，写给 ``delta_step``。
// @param beta,g [nv] fp32 的写入速率与衰减，写给 ``delta_step``。
// @param qscale 折进归一化后 q 的缩放（softmax scale * 1/sqrt(d)）。
// @param rep 每个 key 头对应的 value 头数。
// @return 无。
void delta_prologue(torch::Tensor q_raw, torch::Tensor k_raw, torch::Tensor v_raw,
                    torch::Tensor b, torch::Tensor a,
                    torch::Tensor neg_expA, torch::Tensor dt_bias,
                    torch::Tensor q_out, torch::Tensor k_out, torch::Tensor v_out,
                    torch::Tensor beta, torch::Tensor g,
                    double qscale, int64_t rep);

// @brief 线性注意力层的融合深度因果 conv1d（卷积核 4）+ SiLU。
//
// @param conv_w [C, 4] fp32 抽头。
// @param x [C] bf16 投影输出。
// @param conv_state [C, 3] fp32 历史，原地平移。
// @param out [C] fp32；直接写穿，不返回任何东西。
void conv1d_silu(torch::Tensor conv_w, torch::Tensor x, torch::Tensor conv_state,
                 torch::Tensor out);

// @brief 同样的卷积，跑在整个预填充分块上。
//
// 只读一次激活、写一次，而不是即时执行链的六趟。
//
// @param conv_w [C, 4] fp32 抽头，与 ``conv1d_silu`` 共用。
// @param x [T, C] bf16。
// @param conv_state [C, 3] fp32；最终停在最后三个输入上，下一个解码步能接着同一条递推。
// @return [T, C] fp32。
torch::Tensor conv1d_silu_seq(torch::Tensor conv_w, torch::Tensor x,
                              torch::Tensor conv_state);

// @brief 融合的预填充头准备，写出分块扫描直接读取的 [H, T, d] 布局。
//
// 在这里做而不是在 torch 里做，同时省掉了每层约 1.7 ms 的算子链和 ``chunked_delta`` 内部
// 的三次跨步转置。
//
// @param qkv [T, 2*num_k_heads*dk + num_v_heads*dv] fp32 卷积输出，按 q | k | v 排布。
// @param b,a [T, nv] bf16 投影输出；行距会照实际取值处理，因为它们是融合 GEMV 输出的列
//        切片。
// @param neg_expA,dt_bias [nv] fp32 的 checkpoint 常量，用于衰减门控。
// @param q_out,k_out [nv, T, dk] fp32，``v_out`` [nv, T, dv] fp32，``beta``/``g``
//        [nv, T] fp32。
// @param qscale 折进归一化后 q 的缩放。
// @param rep 每个 key 头对应的 value 头数。
// @return 无。
void delta_prep_seq(torch::Tensor qkv, torch::Tensor b, torch::Tensor a,
                    torch::Tensor neg_expA, torch::Tensor dt_bias,
                    torch::Tensor q_out, torch::Tensor k_out, torch::Tensor v_out,
                    torch::Tensor beta, torch::Tensor g,
                    double qscale, int64_t rep);

// 批处理解码内核（``batch.cu``）。
//
// 与上面单行内核相同的数学，只多一个前导批维，好让 B 个序列的静态批每个内核只启动一次。
// 单行入口仍是 B = 1 / 投机校验的实现；这里宽度为 1 的一行与它们逐位一致。
//
// ``max_seq`` 是 ``slots`` 的行宽（为了与契约的签名兼容而保留）；行 b 的 key 循环以
// ``context_lens[b]`` 为界，所以一个批能混用长短窗口而不必为最长的那条买单。

// @brief 按行在分页 int8 池窗口上做批处理的单 token 注意力。
//
// @param slots [B, smax] int64；``smax`` 是它的行距，不是 ``max_seq``。
// @param context_lens [B] int32；第 b 行的 key 循环停在它自己的窗口上。
// @param part_acc,part_ml [B, nh, split, hd] / [B, nh, split, 2] fp32，归调用方所有，理由
//        同 ``attn_decode`` 的图捕获。
// @return ``out``，[B, nh, hd] bf16。
torch::Tensor attn_paged_batch(torch::Tensor q, torch::Tensor gate,
                               torch::Tensor kpool, torch::Tensor vpool,
                               torch::Tensor kscale, torch::Tensor vscale,
                               torch::Tensor slots, torch::Tensor context_lens,
                               torch::Tensor out, int64_t max_seq,
                               int64_t rep, double scale,
                               torch::Tensor part_acc, torch::Tensor part_ml);

// @brief 多查询校验注意力：一趟历史遍历服务 c 个查询。
//
// 一行的每个查询看到同一段历史 KV，且一行的 c 个查询是相邻的 block，所以这段历史只从 HBM
// 读一次（其余几趟命中 L2）。每个查询仍保持 ``attn_paged_batch`` 那套精确的 split-K /
// online-softmax / 门控算术，因此与单独解码该 token 逐位一致。
//
// @param q,gate,out [B, c, nh, hd] bf16。
// @param ctx_lens [B] int32 历史长度；查询 j 看到 ctx+j+1 个 key。
// @param part_acc,part_ml [B*c, nh, split, hd] / [..., 2] fp32，归调用方所有。
// @param block_mask 可选的 uint8 [B, n_blocks] 保留掩码（空表示精确全注意力）。
// @param mask_block 每个掩码块包含多少个 token，>= 1。
// @return ``out``，[B, c, nh, hd] bf16。
torch::Tensor attn_paged_multi_q(torch::Tensor q, torch::Tensor gate,
                                 torch::Tensor kpool, torch::Tensor vpool,
                                 torch::Tensor kscale, torch::Tensor vscale,
                                 torch::Tensor slots, torch::Tensor ctx_lens,
                                 torch::Tensor out, int64_t max_seq,
                                 int64_t rep, double scale,
                                 torch::Tensor part_acc, torch::Tensor part_ml,
                                 torch::Tensor block_mask, int64_t mask_block);

// @brief 批处理的深度因果 conv1d + SiLU。
//
// @param x [B, C] bf16。
// @param conv_state [B, C, 3] fp32，原地平移。
// @return ``out``，[B, C] fp32。
torch::Tensor conv1d_silu_batch(torch::Tensor conv_w, torch::Tensor x,
                                torch::Tensor conv_state, torch::Tensor out);

// @brief 批处理的 gated delta prologue；输出与 ``delta_prologue`` 相同，另加一个批维。
//
// @param q,k [B, nk, dk] fp32，``v`` [B, nv, dv] fp32。允许行带间距，因为它们是卷积输出的
//        列切片。
// @param b,a [B, nv] bf16。
// @return 无。
void delta_prologue_batch(torch::Tensor q_raw, torch::Tensor k_raw, torch::Tensor v_raw,
                          torch::Tensor b, torch::Tensor a,
                          torch::Tensor neg_expA, torch::Tensor dt_bias,
                          torch::Tensor q_out, torch::Tensor k_out, torch::Tensor v_out,
                          torch::Tensor beta, torch::Tensor g,
                          double qscale, int64_t rep);

// @brief 批处理的 gated delta 更新；各行相互独立。
//
// @param st [B, nv, dk, dv] fp32，原地更新。
// @return ``out``，[B, nv, dv] fp32。
torch::Tensor delta_batch(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                          torch::Tensor beta, torch::Tensor g,
                          torch::Tensor st, torch::Tensor out);

// 投机校验用的 Wide-M int8 GEMV（``gemv_wide.cu``）。

// @brief 一次权重读取服务 M 取 {16, 24, 32} 的激活向量，不走 hipBLAS。
//
// 权重行只读一次，并以 fp32 施加到全部 M 个激活上。M 个取值被分成每组八个（每组一个
// warp），每个（行，激活）对保持与 ``gemv_w8a16_m`` 相同的八条 FMA 链和 K 顺序，所以单独
// 看一行，结果与对该行的激活切片调用 M=8 内核逐位一致。
//
// @param U K 展开（2 或 4）。
// @param threads 块大小，32 的整数倍；warp 必须能均匀分进行组，否则最后一组覆盖不到。
// @return [M, N] bf16。
torch::Tensor gemv_w8a16_wide(torch::Tensor W, torch::Tensor S, torch::Tensor X,
                              int64_t U, int64_t threads);

// @brief ``gemv_w8a16_wide`` 的 A/B 入口，把切块参数暴露出来。
//
// @param rows 每个 warp 的输出行数。
// @param mw 每个 warp 的激活向量数；所有组合都保持 64 个累加器 float。
// @param threads 块大小；0 表示取均匀的 warp 切分。
// @return [M, N] bf16，任何切块下的算术都相同。
torch::Tensor gemv_w8a16_wide_dbg(torch::Tensor W, torch::Tensor S, torch::Tensor X,
                                  int64_t U, int64_t rows, int64_t mw,
                                  int64_t threads);

// @brief 用 packed fp16 数学的 Wide-M GEMV（``v_dot2_f32_f16``，一条指令两次乘加，fp32
//        累加）。
//
// int8 权重转 fp16 是精确的；bf16 激活每次调用只转一次 fp16（|x| <= 65504 时精确，超过则
// 钳制）。切块旋钮与 int8 内核相同；结果与它*接近*（都是 fp32 求和，只是舍入不同），而非
// 逐位一致——校验路径接受这一点，换来指令数减半。
//
// @return [M, N] bf16。
torch::Tensor gemv_w8a16_wide_dot2(torch::Tensor W, torch::Tensor S, torch::Tensor X,
                                   int64_t U, int64_t threads);
// @brief ``gemv_w8a16_wide_dot2`` 的 A/B 入口，把切块参数暴露出来。
//
// @return [M, N] bf16；切块只影响速度。
torch::Tensor gemv_w8a16_wide_dot2_dbg(torch::Tensor W, torch::Tensor S,
                                       torch::Tensor X, int64_t U, int64_t rows,
                                       int64_t mw, int64_t threads);
