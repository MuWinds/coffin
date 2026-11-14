// @file linear_attn.cu
// @brief 门控 delta 规则的预处理、喂给它的深度因果 conv1d，以及把两者连同递推和门控
// 归一化一起跑完的融合解码链。
//
// Module: coffin/backends/corex/kernels。像 ``backends/hip/kernels/linear_attn.cu`` 在
// DCU 上那样喂 delta 递推：本文件产出 delta 步消费的 fp32 q/k/v/beta/g，所以两者必须在
// 布局和 1e-6 的 L2 epsilon 上一致。``delta_normed`` 是这套配对里的解码路径，它放在这里
// 而不是 ``delta.cu``，因为它的归约必须精确复现本文件的 ``block_sum2`` 配对；
// ``delta.cu`` 保留批处理和校验路径仍在用的独立 ``delta`` / ``delta_batch`` 入口。
//
// 为什么融合：在 delta 内核之前，引擎要按头对 q 和 k 做 L2 归一化、缩放 q、把 16 个键头
// 扩成 32 个值头，并构造 ``beta = sigmoid(b)`` 与
// ``g = -exp(A) * softplus(a + dt_bias)``。写成 torch 算子，那是每个层约 16 个小内核，
// 24 个线性注意力层下来，按本卡的启动开销占了这一步的很大一块。每个值头一个 block 就能
// 全做完，并写出下一个内核要读的 fp32。
//
// Constraints:
//   * 每次归约都用共享内存树，不用 ``__shfl_down_sync``。本部件的 warp 是 64 通道宽，为
//     32 写的蝶形会拿半个头的 RMS 去归一化 q 和 k——错，而且错得像模像样。树的尺寸在启动
//     时按 ``blockDim`` 定。
//   * ``q`` 和 ``k`` 在同一趟归约里归一化。它们总是要一起用，两趟只会把屏障数量翻倍而结果
//     不变。
//   * q/k 归一化里的 1e-6 epsilon 是 checkpoint 数值的一部分，不是可调参数：改它就会改
//     模型的输出。
//   * 作为融合投影输出列切片的输入，行间距从张量里取（``ldx`` / ``ldb`` / ``lda``）。
//     在那里假定连续会读到融合行相邻的那三分之一，产出看着合理的垃圾。
//   * ``conv_state`` 归调用方所有、原地更新：它是这一层的短期记忆，必须和循环状态一起打
//     快照，所以返回替代缓冲区的内核会破坏前缀复用。
//   * ``g`` 必须保持 ``<= 0``。``neg_expA`` 是负的、softplus 是正的，所以乘积是衰减；
//     符号写反会让递推增长，那是发散而不只是算错。
//   * ``delta_normed`` 必须与 ``delta_prologue`` -> ``delta`` -> ``rms_norm_gated`` 链
//     逐位相同，因为解码步的贪心 id 是验收标准，最后一个 ulp 的偏移就会挪动它们。这就是它
//     逐字复用 ``block_sum2`` 和 ``head_threads`` 而不另造归约的原因，也是它把递推的 fp32
//     输出先舍入成 bf16 再归一化的原因——链里的 ``RMSNormGated`` 先转换 ``delta_out``，
//     所以那里归一化的输入是 bf16。因此一个 block 需要 ``dv <= head_threads(dk)``；值轴
//     更宽就继续走三算子链，包装层用形状错误拒绝，而不是悄悄改变算术。

#include "corex_bindings.h"
#include "corex_ops.h"
#include "corex_scalar.h"

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

namespace {

//: checkpoint 的 q/k L2 归一化 epsilon。属于模型的数值行为。
constexpr float L2_EPS = 1e-6f;

//: 超过这个值，``softplus(x) == x`` 在 fp32 精度下成立，于是跳过 ``exp``。logit
//: ``a + dt_bias`` 上方无界，而大值的 ``exp`` 会溢出。
constexpr float SOFTPLUS_LINEAR = 20.f;

//: 卷积核宽度。checkpoint 用 4 抽头的深度因果卷积，状态里存前 3 个 token。
constexpr int CONV_K = 4;
constexpr int CONV_STATE = CONV_K - 1;

//: 分块预填充的 conv1d 在 ``T`` 方向分几段。这一个线程一个通道的映射把并行度钉死在
//: ``C = conv_dim = 8192``，所以不分段时只有 32 个 block 在飞（106 GB/s，见
//: ``conv1d_silu_seq`` 的头注释）。扫参（作业 4372）在 1/2/4/8/16 五档上给出的 us/call
//: 是 968.6 / 527.0 / 307.9 / **207.6** / 229.0，所以取 8；每段起点由段内重放 3 个 token
//: 重建，五档输出与 state 全部与不分段版逐位相同。
constexpr int CONV_NSEG = 8;

//: 头归一化 block 的线程数：这个模型 ``dk`` 是 128，而归约需要 wavefront 的整数倍。
int head_threads(int dk) {
    int t = dk > 128 ? dk : 128;
    t = ((t + 63) / 64) * 64;
    return t > 512 ? 512 : t;
}

//: ``norm.cu`` 给宽度为 ``N`` 的一行用多少线程，照抄过来是为了检查融合内核能否顶替门控
//: 归一化。
//:
//: 这里的取整故意是同一个表达式而不是一次调用：``rms_norm_gated_kernel`` 在另一个编译
//: 单元里、藏在匿名命名空间之后，没有东西可调。这份重复是被校验的而不是被信任的——任何
//: 漂移都会让 ``delta_normed`` 拒绝一个它本来能服务的几何，那是有记录的回退；反过来
//: （接受一个归约配对不同的几何）则会悄悄改变 id。
int norm_threads(int n) {
    int t = n < 64 ? 64 : n;
    t = ((t + 63) / 64) * 64;
    return t > 1024 ? 1024 : t;
}

// @brief 一次通过共享树在块内求两个值的和。
//
// q 和 k 一起归一化，所以一趟就够。分成两次归约会把这个完全受延迟限制的内核的屏障数量
// 翻倍：头只有 128 个元素，block 没有别的东西能在同步后面遮掩。
//
// @param a 第一份贡献（q 的平方和）。
// @param b 第二份贡献（k 的平方和）。
// @param red ``2 * blockDim.x`` 个 float 的共享暂存；会被覆写。
// @param sum_a 输出参数，接收 ``a`` 的块内和。
// @param sum_b 输出参数，接收 ``b`` 的块内和。
//
// 结尾的屏障是必需的。两个和由每个线程从 ``red`` 里读出去，而块随后会写别的共享区域，
// 那些区域可能铺在同一片内存上。没有屏障，快线程会在慢线程读完之前就走到那次写——
// 就是在 ``delta.cu`` 对应辅助函数里造成 4.5e-02 误差的那个 bug。
__device__ __forceinline__ void block_sum2(float a, float b, float* red, float* sum_a,
                                           float* sum_b) {
    const int t = threadIdx.x;
    red[t] = a;
    red[blockDim.x + t] = b;
    __syncthreads();
    for (int step = blockDim.x >> 1; step > 0; step >>= 1) {
        if (t < step) {
            red[t] += red[t + step];
            red[blockDim.x + t] += red[blockDim.x + t + step];
        }
        __syncthreads();
    }
    *sum_a = red[0];
    *sum_b = red[blockDim.x];
    __syncthreads();
}

// @brief 尾部取线性的 softplus，让大 logit 不会把指数撑溢出。
//
// @param v logit，即 ``a + dt_bias``。
// @return ``v`` 适中时是 ``log(1 + exp(v))``，很大时就是 ``v`` 本身。
__device__ __forceinline__ float softplus(float v) {
    return v > SOFTPLUS_LINEAR ? v : __logf(1.f + __expf(v));
}

// @brief 单个值的块内和，配对方式与 :func:`block_sum2` 完全一致。
//
// 门控归一化的归约是 ``delta_normed`` 的另一半，它必须按与 ``rms_norm_gated_kernel`` 的树
// 相同的顺序组合相同的伙伴，否则 scale 会在最后一个 ulp 上移动。共用树的形状而不是重新
// 推导它，才让这件事成为结构性的，而不是靠眼睛保持一致。这里不能用 ``block_sum2``：它求
// 两个值的和，而尾声只有一个，加一列哑数据会让内核最后阶段的共享流量翻倍。
//
// @param v 本线程的贡献。
// @param red ``blockDim.x`` 个 float 的共享暂存；会被覆写。
// @return 整个块的和，在结尾同步之后每个线程都拿得到。
//
// 结尾的屏障是必需的，理由和 ``block_sum2`` 里一样：调用方从 ``red[0]`` 把和读出去，不能
// 让更快的线程先开始覆写 ``red``。``delta_normed_kernel`` 确实会撞上这个风险——它在
// ``block_sum2`` 返回之后立刻把 ``red`` 复用于这次归约——所以屏障在那里是承重的，没有
// 省略。
__device__ __forceinline__ float block_sum_scalar(float v, float* red) {
    const int t = (int)threadIdx.x;
    red[t] = v;
    __syncthreads();
    for (int step = blockDim.x >> 1; step > 0; step >>= 1) {
        if (t < step) red[t] += red[t + step];
        __syncthreads();
    }
    const float total = red[0];
    __syncthreads();
    return total;
}

// @brief 为一个解码步构造 fp32 的 q/k/v 与 beta/g 门。
//
// 每个值头一个 block：block 归一化这个头的 q 和 k（与同一键头的其它值头共享，所以每个
// 键头会被归一化 ``rep`` 次——多出来的读取都是 L2 命中，另一条路是再来一次启动）、缩放
// q，并从原始 logits 写出 beta/g。
//
// @param q_raw [nkeys, dk] fp32 查询，每个键头一份。
// @param k_raw [nkeys, dk] fp32 键，每个键头一份。
// @param v_raw [nv, dv] fp32 值；原样拷过去，让 delta 步只读一种 dtype。
// @param b,a [nv] bf16 原始 logits，喂给 beta 和 g。
// @param neg_expA,dt_bias [nv] fp32，衰减门的 checkpoint 常量。
// @param q_out [nv, dk] fp32，归一化并缩放后的查询。
// @param k_out [nv, dk] fp32，归一化后的键。
// @param v_out [nv, dv] fp32 值。
// @param beta [nv] fp32 写强度。
// @param g [nv] fp32 对数衰减，``<= 0``。
// @param qscale ``dk ** -0.5``，折进归一化后的查询。
// @param rep 每个键头对应的值头数。
__global__ void prologue_kernel(const float* __restrict__ q_raw,
                                const float* __restrict__ k_raw,
                                const float* __restrict__ v_raw,
                                const uint16_t* __restrict__ b,
                                const uint16_t* __restrict__ a,
                                const float* __restrict__ neg_expA,
                                const float* __restrict__ dt_bias,
                                float* __restrict__ q_out, float* __restrict__ k_out,
                                float* __restrict__ v_out, float* __restrict__ beta,
                                float* __restrict__ g, float qscale, int dk, int dv,
                                int rep) {
    const int vh = blockIdx.x;
    const int t = threadIdx.x;
    const int kh = vh / rep;
    const float* qr = q_raw + (size_t)kh * dk;
    const float* kr = k_raw + (size_t)kh * dk;

    const float qv = t < dk ? qr[t] : 0.f;
    const float kv = t < dk ? kr[t] : 0.f;
    extern __shared__ float sh[];
    float* red = sh;                       // [2 * blockDim.x]
    float ssq = 0.f;
    float ssk = 0.f;
    block_sum2(qv * qv, kv * kv, red, &ssq, &ssk);

    if (t < dk) {
        q_out[(size_t)vh * dk + t] = qv * rsqrtf(ssq + L2_EPS) * qscale;
        k_out[(size_t)vh * dk + t] = kv * rsqrtf(ssk + L2_EPS);
    }
    if (t < dv) v_out[(size_t)vh * dv + t] = v_raw[(size_t)vh * dv + t];
    if (t == 0) {
        beta[vh] = 1.f / (1.f + __expf(-CxScalar<uint16_t>::to_float(b[vh])));
        const float av = CxScalar<uint16_t>::to_float(a[vh]) + dt_bias[vh];
        g[vh] = neg_expA[vh] * softplus(av);
    }
}

// @brief 单 token 的深度因果卷积（4 抽头）接 SiLU。
//
// 线程 ``c`` 独占通道 ``c`` 和它的三抽头窗口，所以对状态的「先读后写」不需要同步：没有
// 别的线程碰那个窗口。
//
// 网格的 ``y`` 是批行。一行的输入和窗口是分开的板，所以行轴只花一个 block 下标、行与行
// 之间没有流量；间距是传进来的而不是假定的，因为批处理调用方读的是融合投影的列切片，其
// 行间距大于 ``C``。
//
// @param w [C, 4] fp32 抽头，最老的在前；最后一个抽头乘当前输入。
// @param x [B, C] bf16 投影输出，行间距 ``ldx``。
// @param state [B, C, 3] fp32 的历史输入，带行间距 ``lds`` 原地平移。
// @param out [B, C] fp32 的卷积加 SiLU 结果，行间距 ``ldo``。
// @param C 通道数，即 ``x`` 网格覆盖的宽度。
// @param ldx ``x`` 的行间距，以元素计。校验主体传进来的是 ``qkv[:, j]``，即
//        ``[rows, c, conv_dim]`` 视图的一列切片，所以那里它是 ``c * C``，而假定 ``C`` 的
//        内核会悄悄读到相邻 token 的通道。
// @param lds ``state`` 的行间距，以 float 计。
// @param ldo ``out`` 的行间距，以 float 计。
__global__ void conv1d_silu_kernel(const float* __restrict__ w,
                                   const uint16_t* __restrict__ x,
                                   float* __restrict__ state, float* __restrict__ out,
                                   int C, int ldx, int lds, int ldo) {
    const int row = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float* s = state + (size_t)row * lds + (size_t)c * CONV_STATE;
    const float* wc = w + (size_t)c * CONV_K;
    const float x3 = CxScalar<uint16_t>::to_float(x[(size_t)row * ldx + c]);
    const float acc = fmaf(s[0], wc[0],
                           fmaf(s[1], wc[1], fmaf(s[2], wc[2], x3 * wc[3])));
    const float y = acc / (1.f + __expf(-acc));     // SiLU
    out[(size_t)row * ldo + c] = y;
    s[0] = s[1];
    s[1] = s[2];
    s[2] = x3;
}

// @brief 同一个卷积跑在预填充分块上，窗口留在寄存器里，并沿 ``T`` 分 ``blockIdx.y`` 段。
//
// 线程 ``c`` 走过整个序列，把三抽头环留在寄存器里，所以这一趟只读一遍 bf16 激活、只写
// 一遍 fp32。它替代的那条 eager 链（cat + 提升到 fp32 + unfold + 乘法 + 求和 + SiLU）
// 要碰约六倍的字节数。
//
// 固定 ``t`` 时相邻线程读相邻通道，所以即便单个线程按 ``ldx`` 跨步，一个 wavefront 每步
// 仍然覆盖整条 cache line。
//
// **为什么 grid 的 ``y`` 轴是 ``T`` 方向的分段**：这条内核的并行度原本只有 ``C = 8192``
// 个线程（一个线程一个通道），而每个线程要串行走完 ``T = 2048`` 步 —— 在飞的 block 只有
// ``ceil(8192/256) = 32`` 个，实测 99 MB / 937 us = 106 GB/s，正好是 ``32 × 3.0 GB/s``。
// 只改 blockDim 没有用（实测 1.08x，因为线程总数恒为 8192），真正的并行度在 ``T`` 上。
//
// 分段合法是因为这条卷积是因果的：段 ``s`` 的起点状态可以**从原始输入重放**前
// ``CONV_STATE`` 个 token 重建（那 3 步的递推与主循环是同一个表达式，所以重建出的历史
// 与整条走下来逐位相同）。实测：8 是最优点：再往上分段，重放与调度开销盖过并发收益。24 层折算省 18.3 ms。
//
// @param w [C, 4] fp32 抽头。
// @param x [T, C] bf16 激活；行间距 ``ldx`` 可以大于 ``C``，因为这些投影是直接从融合
//        GEMV 的输出里读的。
// @param state [C, 3] fp32 窗口，原地更新为最近三个输入。
// @param out [T, C] fp32，稠密打包。
// @param seg 每段覆盖的 token 数，``ceil(T / nseg)``。由主机侧算好传进来。
__global__ void conv1d_silu_seq_kernel(const float* __restrict__ w,
                                       const uint16_t* __restrict__ x,
                                       float* __restrict__ state, float* __restrict__ out,
                                       int C, int T, int ldx, int seg) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int t0 = blockIdx.y * seg;
    if (t0 >= T) return;
    const int t1 = min(t0 + seg, T);
    const float* wc = w + (size_t)c * CONV_K;
    const float w0 = wc[0];
    const float w1 = wc[1];
    const float w2 = wc[2];
    const float w3 = wc[3];
    float* s = state + (size_t)c * CONV_STATE;
    const uint16_t* xp = x + c;
    float* o = out + c;
    float s0, s1, s2;
    if (blockIdx.y == 0) {
        // 第一段直接读真实窗口，不重放。
        s0 = s[0];
        s1 = s[1];
        s2 = s[2];
    } else {
        // 其余段从原始输入重放前 ``CONV_STATE`` 个 token 重建起点。它们不进 ``out``，
        // 只用来把寄存器环推到位，所以重放的代价是 O(CONV_STATE) 而不是 O(t0)。
        s0 = 0.f;
        s1 = 0.f;
        s2 = 0.f;
        for (int t = max(0, t0 - CONV_STATE); t < t0; ++t) {
            const float x3 = CxScalar<uint16_t>::to_float(xp[(size_t)t * ldx]);
            s0 = s1;
            s1 = s2;
            s2 = x3;
        }
    }
    for (int t = t0; t < t1; ++t) {
        const float x3 = CxScalar<uint16_t>::to_float(xp[(size_t)t * ldx]);
        const float acc = fmaf(w0, s0, fmaf(w1, s1, fmaf(w2, s2, w3 * x3)));
        o[(size_t)t * C] = acc / (1.f + __expf(-acc));     // SiLU
        s0 = s1;
        s1 = s2;
        s2 = x3;
    }
    // 只有最后一段把最终窗口写回。中间段不回写，否则后一段会读到被前一段覆盖的状态；
    // 而每一段自己是从原始输入重放的，本来就不需要别人给它留状态。
    if (t1 == T) {
        s[0] = s0;
        s[1] = s1;
        s[2] = s2;
    }
}

// @brief 预填充的头预处理：整个分块上做一遍解码预处理，输出头主序布局。
//
// 每个（值头，token）一个 block。这些转置比看起来更要紧：把转置过的 ``[H, C, d]`` 视图
// 交给分块 delta 扫描，会在它内部逼出三次带步长的拷贝，而在本部件上分块的带步长拷贝
// 远跑不到可达带宽。在这里直接写出 ``[H, T, d]`` 就把这三次全消掉了。
//
// @param qkv [T, conv_dim] fp32 卷积输出，按 q | k | v 排列；各头是这一行的列切片。
// @param b,a [T, nv] bf16，行间距 ``ldb`` / ``lda``。
// @param q_out,k_out [nv, T, dk] fp32；``v_out`` [nv, T, dv] fp32。
// @param beta,g [nv, T] fp32。
// @param kdim ``num_key_heads * dk``，k 那一段在一行内的偏移。
__global__ void delta_prep_seq_kernel(const float* __restrict__ qkv,
                                      const uint16_t* __restrict__ b,
                                      const uint16_t* __restrict__ a,
                                      const float* __restrict__ neg_expA,
                                      const float* __restrict__ dt_bias,
                                      float* __restrict__ q_out, float* __restrict__ k_out,
                                      float* __restrict__ v_out, float* __restrict__ beta,
                                      float* __restrict__ g, float qscale, int T, int dk,
                                      int dv, int rep, int kdim, int conv_dim, int ldb,
                                      int lda) {
    const int vh = blockIdx.x;
    const int t = blockIdx.y;
    const int i = threadIdx.x;
    const int kh = vh / rep;

    const float* row = qkv + (size_t)t * conv_dim;
    const float qv = i < dk ? row[(size_t)kh * dk + i] : 0.f;
    const float kv = i < dk ? row[kdim + (size_t)kh * dk + i] : 0.f;

    extern __shared__ float sh[];
    float* red = sh;                       // [2 * blockDim.x]
    float ssq = 0.f;
    float ssk = 0.f;
    block_sum2(qv * qv, kv * kv, red, &ssq, &ssk);

    if (i < dk) {
        q_out[((size_t)vh * T + t) * dk + i] = qv * rsqrtf(ssq + L2_EPS) * qscale;
        k_out[((size_t)vh * T + t) * dk + i] = kv * rsqrtf(ssk + L2_EPS);
    }
    if (i < dv) v_out[((size_t)vh * T + t) * dv + i] = row[2 * kdim + (size_t)vh * dv + i];
    if (i == 0) {
        beta[(size_t)vh * T + t] =
            1.f / (1.f + __expf(-CxScalar<uint16_t>::to_float(b[(size_t)t * ldb + vh])));
        const float av = CxScalar<uint16_t>::to_float(a[(size_t)t * lda + vh]) + dt_bias[vh];
        g[(size_t)vh * T + t] = neg_expA[vh] * softplus(av);
    }
}

// @brief 批处理预处理：每个（行，值头）一个 block。
//
// @param q,k [B, nkeys, dk] fp32；``v`` [B, nv, dv] fp32。
// @param b,a [B, nv] bf16。
// @param q_out,k_out [B, nv, dk] fp32；``v_out`` [B, nv, dv] fp32。
// @param beta,g [B, nv] fp32。
__global__ void prologue_batch_kernel(const float* __restrict__ q,
                                      const float* __restrict__ k,
                                      const float* __restrict__ v,
                                      const uint16_t* __restrict__ b,
                                      const uint16_t* __restrict__ a,
                                      const float* __restrict__ neg_expA,
                                      const float* __restrict__ dt_bias,
                                      float* __restrict__ q_out, float* __restrict__ k_out,
                                      float* __restrict__ v_out, float* __restrict__ beta,
                                      float* __restrict__ g, float qscale, int nv, int dk,
                                      int dv, int rep) {
    const int row = blockIdx.y;
    const int vh = blockIdx.x;
    const int t = threadIdx.x;
    const int kh = vh / rep;
    const size_t head = ((size_t)row * (nv / rep) + kh) * dk;
    const float qv = t < dk ? q[head + t] : 0.f;
    const float kv = t < dk ? k[head + t] : 0.f;

    extern __shared__ float sh[];
    float* red = sh;
    float ssq = 0.f;
    float ssk = 0.f;
    block_sum2(qv * qv, kv * kv, red, &ssq, &ssk);

    const size_t out_head = ((size_t)row * nv + vh);
    if (t < dk) {
        q_out[out_head * dk + t] = qv * rsqrtf(ssq + L2_EPS) * qscale;
        k_out[out_head * dk + t] = kv * rsqrtf(ssk + L2_EPS);
    }
    if (t < dv) v_out[out_head * dv + t] = v[out_head * dv + t];
    if (t == 0) {
        const size_t bi = (size_t)row * nv + vh;
        beta[bi] = 1.f / (1.f + __expf(-CxScalar<uint16_t>::to_float(b[bi])));
        const float av = CxScalar<uint16_t>::to_float(a[bi]) + dt_bias[vh];
        g[bi] = neg_expA[vh] * softplus(av);
    }
}

// @brief 单解码 token 的预处理、递推与门控归一化，每个头一个 block。
//
// 为什么这是一个内核：它替代的那三个是一个内存问题，不是三个。预处理写出
// ``q_pro``/``k_pro``/``v_pro``（3 x [nv, dk] fp32）加 ``beta``/``g``，递推把它们直接
// 读回来并写 ``delta_out``（[nv, dv] fp32），门控归一化又把 *那个* 读回来：每层每 token
// 往返约 48 KB，每步 24 次，纯粹是为了在一张启动下限 5.75 us 的卡上次次在启动之间递值。
// 预处理为一个值头产出的东西全装得进这个头的 block——``q`` 和 ``k`` 占 ``2 * dk`` 个
// float 的共享内存（dk = 128 时 1 KB），``v`` 和两个门的标量在寄存器里——而归一化的归约
// 只跨 ``dv``，正好是一个 block 的跨度，所以它免费折进尾声，不需要额外的归约。
//
// 块形状不是三阶段之间的折中，而是三者在用的同一个形状：``prologue_kernel`` 每个值头跑
// ``head_threads(dk)`` 个线程，``delta_kernel`` 跑 ``min(dv, THREADS)``，
// ``rms_norm_gated_kernel`` 跑 ``rms_threads(dv)``——对这个模型（dk = dv = 128）分别是
// 128、128 和 128。共用这个形状，才让每次归约都保持那条链的配对，而这正是逐位相同与仅仅
// 接近的差别：解码步的贪心 id 是验收标准，这三次归约里任何一次的最后一个 ulp 偏移都会
// 挪动它们。
//
// ``conv1d_silu`` 故意保持自己的启动。它跨的是 ``conv_dim`` = 8192 个通道而不是一个头，
// 折进来就会把一个形状不同的向量的切片塞进一个 block，而那个 block 的线程映射存在的目的
// 是合并状态访问。
//
// @param q,k [nkeys, dk] fp32 原始投影输出；在这里归一化，``q`` 还做缩放，所以调用方不做
//        任何逐元素修正。
// @param v [nv, dv] fp32 值。
// @param b,a [nv] bf16 原始 beta／衰减 logits。
// @param neg_expA,dt_bias [nv] fp32，衰减门的 checkpoint 常量。
// @param z [nv, dv] bf16，输出归一化 SiLU 用的门。
// @param nw [dv] bf16 门控归一化增益，按值头而不是按展平宽度。
// @param st [nv, dk, dv] fp32 状态，原地更新。
// @param out [nv, dv] 写入归一化后的结果；元素类型只是选择容器，因为两种情况下值都是那条
//        链的 bf16 结果。
// @param qscale ``dk ** -0.5``。
// @param rep 每个键头对应的值头数。
//
// ``ZT`` 是门的存储类型，不是精度选择。算子接受 bf16（模型的门投影产出的就是它）和
// fp32（契约也允许）；实例化在主机侧按 ``z`` 的 dtype 选。把 fp32 的 ``z`` 当 bf16 读会
// 重新解释每两个半字，用噪声去门控输出，所以这两种情况不能共用一个实例化。
// @brief 单个校验 token 的融合体：头预处理、秩一递推、门控输出归一化。
//
// 抽成设备函数是为了让单 token 入口和 c token 入口共用*同一份算术*：投机校验要求
// 「一次跑 c 个 token」与「跑 c 次单 token」逐位一致，复制一份出来就必然会漂移，而
// 漂移的表现是校验 logits 与贪心解码差一个 ulp——那正好是翻转 id 的量级。
//
// 调用方负责把该 token 的各个指针算好传进来；``st`` 是跨 token 延续的那一份，所以
// 串行调用就是递推本身。不在本函数里的只有 ``extern __shared__`` 的切分和两个
// ``__shared__`` 标量，因为它们的生命周期是 block 而不是 token。
//
// @param head_record 本 token 的更新日志起点，``nullptr`` 表示不记录。
// @param red ``[2 * blockDim.x]`` 归约树暂存。
// @param qs,ks ``[dk]`` 归一化后的查询与键，留在共享内存里。
// @param rate_sh,decay_sh 每 token 由 t==0 重算的 beta 与衰减标量。
template <typename OT, typename ZT>
__device__ __forceinline__ void delta_normed_body(
        const float* __restrict__ q,
        const float* __restrict__ k,
        const float* __restrict__ v,
        const uint16_t* __restrict__ b,
        const uint16_t* __restrict__ a,
        const float* __restrict__ neg_expA,
        const float* __restrict__ dt_bias,
        const ZT* __restrict__ z,
        const uint16_t* __restrict__ nw,
        float* __restrict__ st, OT* __restrict__ out,
        float* __restrict__ head_record,
        float qscale, int nv, int dk, int dv, int rep,
        float* red, float* qs, float* ks,
        float* rate_sh, float* decay_sh) {
    const int vh = blockIdx.x;
    const int t = (int)threadIdx.x;
    const int kh = vh / rep;

    // 头预处理：与 ``prologue_kernel`` 相同的算术、相同的单趟归约和相同的树配对。
    const float qv = t < dk ? q[(size_t)kh * dk + t] : 0.f;
    const float kv = t < dk ? k[(size_t)kh * dk + t] : 0.f;
    float ssq = 0.f;
    float ssk = 0.f;
    block_sum2(qv * qv, kv * kv, red, &ssq, &ssk);

    // 留在共享内存里而不是写出去再读回来：这两个缓冲区正是融合要消掉的那份流量，而
    // ``2 * dk`` 个 float 正好装得下一个头的量。
    if (t < dk) {
        qs[t] = qv * rsqrtf(ssq + L2_EPS) * qscale;
        ks[t] = kv * rsqrtf(ssk + L2_EPS);
        if (head_record) head_record[t] = ks[t];
    }
    if (t == 0) {
        *rate_sh = 1.f / (1.f + __expf(-CxScalar<uint16_t>::to_float(b[vh])));
        const float av = CxScalar<uint16_t>::to_float(a[vh]) + dt_bias[vh];
        // 那条链在预处理里存下 ``g``、在递推里对它取指数；在这里把两步合起来，两种写法
        // 都只在 fp32 下把乘积舍入一次。改成每线程各做一遍，只会把一次超越函数重复 128
        // 次而毫无收益。
        *decay_sh = __expf(neg_expA[vh] * softplus(av));
        if (head_record) head_record[dk + dv] = *decay_sh;
    }
    __syncthreads();

    // 超出 ``dv`` 的线程必须走到本内核里的每一个 ``__syncthreads()``，并且只在最后一个
    // 之后返回。在这里返回——也就是在归一化的树之前——会让块里其余线程等一个已经不在的
    // 线程：设备侧死锁，没有 traceback，而不是一个错数字。这个风险在 ``delta.cu`` 里写过，
    // 它的第一版就踩过。
    //
    // c token 入口上这个 ``return`` 只退出本次 token 的体，外层循环照样进下一轮，所以
    // 屏障在两边都配平。
    const bool active = t < dv;
    const float rate = *rate_sh;
    const float decay = *decay_sh;

    // 递推，和 ``delta_kernel`` 里一字未改：线程 ``j`` 独占值列 ``j``，在寄存器里把
    // ``i`` 走过 ``dk``，所以两次 ``dk`` 归约都是线程本地的，固定 ``i`` 时读到的状态在
    // ``j`` 上连续——这是 ``[nv, dk, dv]`` 布局下唯一能合并的分配方式。
    float result = 0.f;
    if (active) {
        const float value = v[(size_t)vh * dv + t];
        float* column = st + (size_t)vh * dk * dv + t;
        float kv_mem = 0.f;
        // 两段 dk 递推各 128 次迭代，实测是标量下标与控制流指令的主要来源：不展开时
        // 每 wavefront 3,453 条标量 + 337 条控制流，八倍展开后降到 1,561/116，
        // 设备时间 93.8 -> 62.5 us（-33%），状态、输出与更新日志逐位不变。再往上到
        // 十六倍只多 2.3%，寄存器却从 23 继续涨，不值得。
        // 只复制标量下标与分支，不改 4 B 列访问：16 B 宽访存在这张卡上有 2.4 倍惩罚。
#pragma unroll 8
        for (int i = 0; i < dk; ++i) {
            kv_mem = fmaf(column[(size_t)i * dv] * decay, ks[i], kv_mem);
        }
        const float delta = (value - kv_mem) * rate;
        if (head_record) head_record[dk + t] = delta;
#pragma unroll 8
        for (int i = 0; i < dk; ++i) {
            const float s = fmaf(ks[i], delta, column[(size_t)i * dv] * decay);
            column[(size_t)i * dv] = s;
            result = fmaf(s, qs[i], result);
        }
    }

    // 那条链交给门控归一化的是 ``delta_out`` 的 bf16 副本：``RMSNormGated`` 在内核看到它
    // 之前就把 fp32 输入转成激活 dtype，所以归一化的输入早就在 bf16 网格上了。在这里舍入一
    // 次复现了这一点。跳过它，归一化的值最多会比那条链归一化的值偏半个 bf16 ulp，而这就
    // 足够挪动 id——这次舍入是那条链算术的一部分，不是它存储方式的副产品。
    const float xb = active
                         ? CxScalar<uint16_t>::to_float(CxScalar<uint16_t>::from_float(result))
                         : 0.f;
    // ``block_sum_scalar`` 以一道屏障收尾，所以 ``red`` 可以被每个线程在这里复用——包括
    // 跳过递推的那些线程，这就是那道屏障不是可以随手删掉的优化。
    const float inv = rsqrtf(block_sum_scalar(xb * xb, red) / (float)dv + L2_EPS);
    if (!active) return;

    // ``y = rms(x) * w * silu(gate)``，按那条链自己的操作数顺序。乘积从左往右结合，换一种
    // 结合方式会改变最后一个 ulp，所以这个顺序是规格的一部分，不是风格选择。
    //
    // 不管门以什么形式存储，都要先舍入到 bf16。``RMSNormGated.__call__`` 在内核看到两个
    // 操作数之前就把它们转成激活 dtype，所以 fp32 的门到了那条链上也会在那里被窄化——这里
    // 的舍入复现了它，而对本来就存成 bf16 的门，这次舍入是恒等。
    const float gv = CxScalar<uint16_t>::to_float(CxScalar<uint16_t>::from_float(
        CxScalar<ZT>::to_float(z[(size_t)vh * dv + t])));
    const float silu = gv / (1.f + __expf(-gv));     // SiLU
    const float y = xb * inv * CxScalar<uint16_t>::to_float(nw[t]) * silu;
    // 那条链的归一化产出 bf16，而 ``out_proj`` 消费的就是它，所以目的地的 dtype 只是容器：
    // fp32 容器装的是同一个 bf16 值而不是更精确的值，这正是两个调用点能逐位一致的原因。
    out[(size_t)vh * dv + t] = CxScalar<OT>::from_float(
        CxScalar<uint16_t>::to_float(CxScalar<uint16_t>::from_float(y)));
}

// @brief 单 token 入口：解码步走这里，一个 block 一个值头。
template <typename OT, typename ZT>
__global__ void delta_normed_kernel(const float* __restrict__ q,
                                    const float* __restrict__ k,
                                    const float* __restrict__ v,
                                    const uint16_t* __restrict__ b,
                                    const uint16_t* __restrict__ a,
                                    const float* __restrict__ neg_expA,
                                    const float* __restrict__ dt_bias,
                                    const ZT* __restrict__ z,
                                    const uint16_t* __restrict__ nw,
                                    float* __restrict__ st, OT* __restrict__ out,
                                    float* __restrict__ record, int record_slot,
                                    float qscale, int nv, int dk, int dv, int rep) {
    extern __shared__ float sh[];
    float* red = sh;                          // [2 * blockDim.x] 归约树
    float* qs = sh + 2 * (int)blockDim.x;     // [dk] 本头归一化后的查询
    float* ks = qs + dk;                      // [dk] 本头归一化后的键
    __shared__ float rate_sh;
    __shared__ float decay_sh;
    float* head_record = record
        ? record + ((size_t)record_slot * nv + blockIdx.x) * (dk + dv + 1)
        : nullptr;
    delta_normed_body<OT, ZT>(q, k, v, b, a, neg_expA, dt_bias, z, nw, st, out,
                              head_record, qscale, nv, dk, dv, rep,
                              red, qs, ks, &rate_sh, &decay_sh);
}

// @brief c token 入口：一次启动跑完校验窗口的全部 token。
//
// 与单 token 入口唯一的差别是外面这层循环。
//
// **实测没有收益，留作负结果**（作业 5077，2026-11-12，Qwen3.5-4B int8，单流 c=4、
// max_seq 8192）：接进 ``LinearAttn.verify`` 之后 verify 一步 32.86 ms，而逐 token 的老
// 路径在三次运行里是 30.68 / 33.64 / 34.48 ms，**全部落在 ±10% 的同进程抖动内**。
//
// 原因是原先的推断错了：单 token 那次调用「1 行和 8 行都要 70–105 us、几乎不随行数变化」
// 说明的是**设备侧执行延迟**（递推要在 [nv, dk, dv] 的状态上走两趟访存），不是 host
// 提交。把 c 步串进一个 launch 只省掉 host 那侧的 3 × ~5.5 us，而块内的 c 步仍然串行、
// 仍然各付一次那笔延迟。也就是说：**这里的成本是每 token 的访存延迟，不是发射。**
//
// 保留它是因为它是「一次启动跑完 c 个 token」这个语义的现成实现，宽 M 校验核或
// 更长窗口的校验形态可能重新需要它；重开之前先证明那笔延迟被摊掉了。
//
// 串行是规格的一部分：递推的状态依赖让它无法并行，而把 c 个 token 摊进一个 kernel 不
// 改变任何一次算术。
//
// @param record_slot 第一个 token 在更新日志里的位置，后续 token 依次 +1。
// @param steps 本窗口的 token 数。
// @param stride_q,stride_k,stride_v,stride_b,stride_z 每个 token 在对应张量里的行距
//        （元素数）。允许大于行宽，因为调用方给的是融合投影 ``split`` 出来的视图。
template <typename OT, typename ZT>
__global__ void delta_normed_seq_kernel(const float* __restrict__ q,
                                        const float* __restrict__ k,
                                        const float* __restrict__ v,
                                        const uint16_t* __restrict__ b,
                                        const uint16_t* __restrict__ a,
                                        const float* __restrict__ neg_expA,
                                        const float* __restrict__ dt_bias,
                                        const ZT* __restrict__ z,
                                        const uint16_t* __restrict__ nw,
                                        float* __restrict__ st, OT* __restrict__ out,
                                        float* __restrict__ record, int record_slot,
                                        float qscale, int nv, int dk, int dv, int rep,
                                        int steps, long long stride_q,
                                        long long stride_k, long long stride_v,
                                        long long stride_b, long long stride_z) {
    extern __shared__ float sh[];
    float* red = sh;
    float* qs = sh + 2 * (int)blockDim.x;
    float* ks = qs + dk;
    __shared__ float rate_sh;
    __shared__ float decay_sh;
    const size_t vdim = (size_t)nv * dv;
    const size_t rec = (size_t)nv * (dk + dv + 1);
    for (int tk = 0; tk < steps; ++tk) {
        float* head_record = record
            ? record + (size_t)(record_slot + tk) * rec
                  + (size_t)blockIdx.x * (dk + dv + 1)
            : nullptr;
        delta_normed_body<OT, ZT>(q + (size_t)tk * (size_t)stride_q,
                                  k + (size_t)tk * (size_t)stride_k,
                                  v + (size_t)tk * (size_t)stride_v,
                                  b + (size_t)tk * (size_t)stride_b,
                                  a + (size_t)tk * (size_t)stride_b,
                                  neg_expA, dt_bias,
                                  z + (size_t)tk * (size_t)stride_z,
                                  nw, st, out + (size_t)tk * vdim, head_record,
                                  qscale, nv, dk, dv, rep,
                                  red, qs, ks, &rate_sh, &decay_sh);
    }
}

// @brief 启动融合解码链：每个值头一个 block。
//
// @param out_bf16 ``out`` 是 bf16（以 ``uint16_t`` 承载）还是 fp32。
// @param z_bf16 ``z`` 是 bf16（以 ``uint16_t`` 承载）还是 fp32。
// @param threads 块大小。包装层会把它同时对 ``head_threads(dk)`` 和那条链另外两次归约所用
//        的块大小做检查，因为不同的大小会用不同的顺序对同样的值求和。
void launch_normed(const float* q, const float* k, const float* v, const uint16_t* b,
                   const uint16_t* a, const float* neg_expA, const float* dt_bias,
                   const void* z, bool z_bf16, const uint16_t* nw, float* st, void* out,
                   bool out_bf16, float* record, int record_slot, float qscale,
                   int nv, int dk, int dv, int rep,
                   int threads, cudaStream_t stream) {
    const size_t smem = (size_t)(2 * threads + 2 * dk) * sizeof(float);
    // 四种实例化，因为目的地 dtype 和门的存储 dtype 是两个独立的选择。四种算的是同一套
    // 算术——无论如何门都会在内核里舍入成 bf16——所以这是分派，不是四种行为。
    if (out_bf16) {
        if (z_bf16) {
            delta_normed_kernel<uint16_t, uint16_t><<<nv, threads, smem, stream>>>(
                q, k, v, b, a, neg_expA, dt_bias, (const uint16_t*)z, nw, st,
                reinterpret_cast<uint16_t*>(out), record, record_slot, qscale,
                nv, dk, dv, rep);
        } else {
            delta_normed_kernel<uint16_t, float><<<nv, threads, smem, stream>>>(
                q, k, v, b, a, neg_expA, dt_bias, (const float*)z, nw, st,
                reinterpret_cast<uint16_t*>(out), record, record_slot, qscale,
                nv, dk, dv, rep);
        }
    } else {
        if (z_bf16) {
            delta_normed_kernel<float, uint16_t><<<nv, threads, smem, stream>>>(
                q, k, v, b, a, neg_expA, dt_bias, (const uint16_t*)z, nw, st,
                reinterpret_cast<float*>(out), record, record_slot, qscale,
                nv, dk, dv, rep);
        } else {
            delta_normed_kernel<float, float><<<nv, threads, smem, stream>>>(
                q, k, v, b, a, neg_expA, dt_bias, (const float*)z, nw, st,
                reinterpret_cast<float*>(out), record, record_slot, qscale,
                nv, dk, dv, rep);
        }
    }
}

// @brief 启动 c token 的融合校验链：同样一个值头一个 block，只是把 c 步串在块内。
void launch_normed_seq(const float* q, const float* k, const float* v, const uint16_t* b,
                       const uint16_t* a, const float* neg_expA, const float* dt_bias,
                       const void* z, bool z_bf16, const uint16_t* nw, float* st, void* out,
                       bool out_bf16, float* record, int record_slot, float qscale,
                       int nv, int dk, int dv, int rep, int steps,
                       long long stride_q, long long stride_k, long long stride_v,
                       long long stride_b, long long stride_z,
                       int threads, cudaStream_t stream) {
    const size_t smem = (size_t)(2 * threads + 2 * dk) * sizeof(float);
    if (out_bf16) {
        if (z_bf16) {
            delta_normed_seq_kernel<uint16_t, uint16_t><<<nv, threads, smem, stream>>>(
                q, k, v, b, a, neg_expA, dt_bias, (const uint16_t*)z, nw, st,
                reinterpret_cast<uint16_t*>(out), record, record_slot, qscale,
                nv, dk, dv, rep, steps, stride_q, stride_k, stride_v, stride_b, stride_z);
        } else {
            delta_normed_seq_kernel<uint16_t, float><<<nv, threads, smem, stream>>>(
                q, k, v, b, a, neg_expA, dt_bias, (const float*)z, nw, st,
                reinterpret_cast<uint16_t*>(out), record, record_slot, qscale,
                nv, dk, dv, rep, steps, stride_q, stride_k, stride_v, stride_b, stride_z);
        }
    } else {
        if (z_bf16) {
            delta_normed_seq_kernel<float, uint16_t><<<nv, threads, smem, stream>>>(
                q, k, v, b, a, neg_expA, dt_bias, (const uint16_t*)z, nw, st,
                reinterpret_cast<float*>(out), record, record_slot, qscale,
                nv, dk, dv, rep, steps, stride_q, stride_k, stride_v, stride_b, stride_z);
        } else {
            delta_normed_seq_kernel<float, float><<<nv, threads, smem, stream>>>(
                q, k, v, b, a, neg_expA, dt_bias, (const float*)z, nw, st,
                reinterpret_cast<float*>(out), record, record_slot, qscale,
                nv, dk, dv, rep, steps, stride_q, stride_k, stride_v, stride_b, stride_z);
        }
    }
}

}  // namespace

// @brief 启动单 token 的深度因果 conv1d 加 SiLU。
//
// 只有一行，所以网格 ``y`` 方向是 1，每个间距都是这一行自己的宽度。批处理入口启动的是
// 同一个内核，只是把行轴收成 1，所以两条路径的算术不可能各自漂移。
//
// @param conv_w [C, 4] fp32 抽头。
// @param x [C] bf16 投影输出。
// @param conv_state [C, 3] fp32 历史，原地平移。
// @param out [C] fp32，原地写入。
// @throws c10::Error dtype 或形状是内核索引不了的。
void conv1d_silu(torch::Tensor conv_w, torch::Tensor x, torch::Tensor conv_state,
                 torch::Tensor out) {
    const int C = (int)x.size(0);
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "activation must be bf16");
    TORCH_CHECK(conv_state.numel() == (int64_t)C * CONV_STATE, "conv state must be [C, 3]");
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    conv1d_silu_kernel<<<dim3((C + 255) / 256, 1), 256, 0, stream>>>(
        conv_w.data_ptr<float>(), reinterpret_cast<const uint16_t*>(x.data_ptr()),
        conv_state.data_ptr<float>(), out.data_ptr<float>(), C, C, CONV_STATE, C);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// @brief 启动分块预填充的 conv1d 加 SiLU，grid 的 ``y`` 轴沿 ``T`` 分 ``CONV_NSEG`` 段。
//
// ``C = conv_dim = 8192`` 且一个线程一个通道，所以不分段时在飞的 block 只有
// ``ceil(8192/256) = 32`` 个 —— 实测 99 MB / 937 us = 106 GB/s，正好是
// ``32 block × 3.0 GB/s``（``docs/PREFILL_52PCT.md`` §2）。线程总数是 8192，**改 blockDim
// 没有用**（实测 256/128/64 三档只差 1.08x，作业 4368），所以并发只能加在 ``T`` 上。
//
// ``CONV_NSEG = 8`` 是扫出来的（作业 4372 ``sg2``）：1/2/4/8/16 五档的 us/call 分别是
// 968.6 / 527.0 / 307.9 / 207.6 / 229.0，8 是最优点，而且**每一档的输出与 state 都与
// 不分段版逐位相同**（段起点从原始输入重放 3 个 token 重建，递推表达式与主循环是同一个）。
// 24 层折算省 18.3 ms。
//
// @param conv_w [C, 4] fp32 抽头。
// @param x [T, C] bf16，最内层步长为 1；行间距会被正确处理，所以融合 GEMV 输出的一列
//        切片可以原地读取。
// @param conv_state [C, 3] fp32，原地更新，让下一个解码步接着同一条递推——这正是分块
//        预填充能与一次性预填充一致的原因。
// @return [T, C] fp32 卷积加 SiLU。
// @throws c10::Error ``x`` 不是二维或最内层步长不是 1；请转置调用方的切片，不要放宽
//         这项检查。
torch::Tensor conv1d_silu_seq(torch::Tensor conv_w, torch::Tensor x,
                              torch::Tensor conv_state) {
    TORCH_CHECK(x.dim() == 2 && x.stride(1) == 1,
                "activation must be [T, C] with unit inner stride");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "activation must be bf16");
    const int T = (int)x.size(0);
    const int C = (int)x.size(1);
    const int ldx = (int)x.stride(0);
    auto out = torch::empty({T, C}, x.options().dtype(torch::kFloat));
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    // 段数按 ``T`` 截断：短序列（含 T < CONV_NSEG 的情形）不该发出空段。
    const int nseg = T < CONV_NSEG ? T : CONV_NSEG;
    const int seg = (T + nseg - 1) / nseg;
    conv1d_silu_seq_kernel<<<dim3((C + 255) / 256, nseg), 256, 0, stream>>>(
        conv_w.data_ptr<float>(), reinterpret_cast<const uint16_t*>(x.data_ptr()),
        conv_state.data_ptr<float>(), out.data_ptr<float>(), C, T, ldx, seg);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// @brief 启动批处理单 token conv1d 加 SiLU：所有行只用一次启动。
//
// 网格的 ``y`` 轴是行，所以整个桶只花一次启动，在飞的是 ``rows * ceil(C/256)`` 个 block
// 而不是单行的 32 个。实测旧的逐行循环每行固定 9.57 us、每行流量 368 KB——39 GB/s，那是
// 网格宽度不够而不是字节多——一个 4 行的校验窗口在它 24 个线性注意力层上要付 96 次。
//
// 间距从张量里取而不是从 ``C`` 推，这一条是承重的：校验主体传进来的是 ``qkv[:, j]``，即
// ``[rows, c, conv_dim]`` 视图的一列切片，其行间距是 ``c * C``。按 ``C`` 去读会让第 1 行
// 及以后都取到第 0 行的通道块，这在 ``rows == 1`` 时看不出来，而在宽度大于 1 时本会是一个
// 静默的错答案（实测：这个参数出现之前，4 行里有 3 行是错的）。
//
// @param conv_w [C, 4] fp32 抽头，所有行共用。
// @param x [B, C] bf16，行间距会被正确处理；带步长的列切片也合法。
// @param conv_state [B, C, 3] fp32，原地平移。
// @param out [B, C] fp32，原地写入。
// @throws c10::Error dtype、秩或最内层步长是内核索引不了的。
void conv1d_silu_batch(torch::Tensor conv_w, torch::Tensor x, torch::Tensor conv_state,
                       torch::Tensor out) {
    TORCH_CHECK(x.dim() == 2, "activation must be [B, C]");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "activation must be bf16");
    TORCH_CHECK(x.stride(1) == 1, "activation must have unit inner stride");
    const int rows = (int)x.size(0);
    const int C = (int)x.size(1);
    TORCH_CHECK(out.dim() == 2 && out.size(0) == rows && out.size(1) == C
                && out.stride(1) == 1, "output must be [B, C] with unit inner stride");
    TORCH_CHECK(conv_state.numel() == (int64_t)rows * C * CONV_STATE,
                "conv state must be [B, C, 3]");
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    conv1d_silu_kernel<<<dim3((C + 255) / 256, rows), 256, 0, stream>>>(
        conv_w.data_ptr<float>(), reinterpret_cast<const uint16_t*>(x.data_ptr()),
        conv_state.data_ptr<float>(), out.data_ptr<float>(), C,
        (int)x.stride(0), (int)conv_state.stride(0), (int)out.stride(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
// @brief 启动单 token 的门控 delta 预处理。
//
// @param q,k [nkeys, dk] fp32，每个键头一份。
// @param v [nv, dv] fp32。
// @param b,a [nv] bf16 原始 logits。
// @param neg_expA,dt_bias [nv] fp32 checkpoint 常量。
// @param q_out,k_out [nv, dk] fp32 与 ``v_out`` [nv, dv] fp32，原地写入。
// @param beta,g [nv] fp32，原地写入。
// @param qscale ``dk ** -0.5``。
// @param rep 每个键头对应的值头数。
// @throws c10::Error dtype 或形状与布局矛盾。
void delta_prologue(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor b,
                    torch::Tensor a, torch::Tensor neg_expA, torch::Tensor dt_bias,
                    torch::Tensor q_out, torch::Tensor k_out, torch::Tensor v_out,
                    torch::Tensor beta, torch::Tensor g, double qscale, int64_t rep) {
    const int nv = (int)v.size(0);
    const int dk = (int)q.size(1);
    const int dv = (int)v.size(1);
    TORCH_CHECK(b.scalar_type() == torch::kBFloat16, "beta logits must be bf16");
    TORCH_CHECK(q.scalar_type() == torch::kFloat, "prologue inputs are fp32");
    const int threads = head_threads(dk);
    const size_t smem = (size_t)2 * threads * sizeof(float);
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    prologue_kernel<<<nv, threads, smem, stream>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
        reinterpret_cast<const uint16_t*>(b.data_ptr()),
        reinterpret_cast<const uint16_t*>(a.data_ptr()),
        neg_expA.data_ptr<float>(), dt_bias.data_ptr<float>(),
        q_out.data_ptr<float>(), k_out.data_ptr<float>(), v_out.data_ptr<float>(),
        beta.data_ptr<float>(), g.data_ptr<float>(), (float)qscale, dk, dv, (int)rep);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// @brief 批处理预处理：每个批条目一行，每一行都与单独跑解码形式的结果相同。
//
// @param q,k [B, nkeys, dk] fp32。
// @param v [B, nv, dv] fp32。
// @param b,a [B, nv] bf16。
// @param neg_expA,dt_bias [nv] fp32，所有行共用。
// @param q_out,k_out [B, nv, dk] fp32 与 ``v_out`` [B, nv, dv] fp32，原地写入。
// @param beta,g [B, nv] fp32，原地写入。
// @param qscale ``dk ** -0.5``。
// @param rep 每个键头对应的值头数。
// @throws c10::Error dtype 或形状与布局矛盾。
void delta_prologue_batch(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                          torch::Tensor b, torch::Tensor a, torch::Tensor neg_expA,
                          torch::Tensor dt_bias, torch::Tensor q_out,
                          torch::Tensor k_out, torch::Tensor v_out, torch::Tensor beta,
                          torch::Tensor g, double qscale, int64_t rep) {
    const int rows = (int)v.size(0);
    const int nv = (int)v.size(1);
    const int dk = (int)q.size(2);
    const int dv = (int)v.size(2);
    TORCH_CHECK(b.scalar_type() == torch::kBFloat16, "beta logits must be bf16");
    const int threads = head_threads(dk);
    const size_t smem = (size_t)2 * threads * sizeof(float);
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    prologue_batch_kernel<<<dim3(nv, rows), threads, smem, stream>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
        reinterpret_cast<const uint16_t*>(b.data_ptr()),
        reinterpret_cast<const uint16_t*>(a.data_ptr()),
        neg_expA.data_ptr<float>(), dt_bias.data_ptr<float>(),
        q_out.data_ptr<float>(), k_out.data_ptr<float>(), v_out.data_ptr<float>(),
        beta.data_ptr<float>(), g.data_ptr<float>(), (float)qscale, nv, dk, dv, (int)rep);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// @brief 启动分块预填充的门控 delta 预处理，写出头主序布局。
//
// @param qkv [T, conv_dim] fp32 卷积输出，按 q | k | v 排列。
// @param b,a [T, nv] bf16。
// @param neg_expA,dt_bias [nv] fp32 常量。
// @param q_out,k_out [nv, T, dk] fp32 与 ``v_out`` [nv, T, dv] fp32，原地写入。
// @param beta,g [nv, T] fp32，原地写入。
// @param qscale ``dk ** -0.5``。
// @param rep 每个键头对应的值头数。
// @throws c10::Error ``qkv``/``b``/``a`` 的最内层步长不是 1；请传切片，不要传转置。
void delta_prep_seq(torch::Tensor qkv, torch::Tensor b, torch::Tensor a,
                    torch::Tensor neg_expA, torch::Tensor dt_bias, torch::Tensor q_out,
                    torch::Tensor k_out, torch::Tensor v_out, torch::Tensor beta,
                    torch::Tensor g, double qscale, int64_t rep) {
    TORCH_CHECK(qkv.stride(1) == 1 && b.stride(1) == 1 && a.stride(1) == 1,
                "unit inner stride required");
    const int T = (int)qkv.size(0);
    const int nv = (int)q_out.size(0);
    const int dk = (int)q_out.size(2);
    const int dv = (int)v_out.size(2);
    const int kdim = dk * (int)(q_out.size(0) / rep);      // num_key_heads * dk
    const int threads = head_threads(dk);
    const size_t smem = (size_t)2 * threads * sizeof(float);
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    delta_prep_seq_kernel<<<dim3(nv, T), threads, smem, stream>>>(
        qkv.data_ptr<float>(), reinterpret_cast<const uint16_t*>(b.data_ptr()),
        reinterpret_cast<const uint16_t*>(a.data_ptr()), neg_expA.data_ptr<float>(),
        dt_bias.data_ptr<float>(), q_out.data_ptr<float>(), k_out.data_ptr<float>(),
        v_out.data_ptr<float>(), beta.data_ptr<float>(), g.data_ptr<float>(), (float)qscale,
        T, dk, dv, (int)rep, kdim, (int)qkv.size(1), (int)b.stride(0), (int)a.stride(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// @brief 单解码 token 的预处理、递推与门控归一化，一次启动。
//
// 结果与那三算子链逐位相同，所以贪心 id 挪不动：归约复用 ``block_sum2`` 和另外两个阶段
// 都在用的块大小，递推对同样的 fp32 值跑同样的 ``fmaf`` 链，而归一化的输入先舍入成 bf16，
// 因为 ``RMSNormGated`` 在内核看到它之前就先转换了。正是这个等价关系让块大小成为被检查的
// 不变量而不是调出来的参数——见下面的 ``head_threads`` 检查。
//
// @param q [nkeys, dk] fp32 原始查询，在这里归一化并缩放。
// @param k [nkeys, dk] fp32 原始键，在这里归一化。
// @param v [nv, dv] fp32 值。
// @param b [nv] bf16 beta logits；``a`` [nv] bf16 衰减 logits。
// @param neg_expA,dt_bias [nv] fp32，衰减门的 checkpoint 常量。
// @param z [nv, dv] bf16，输出归一化 SiLU 用的门。
// @param nw [dv] bf16 门控归一化增益，按值头。
// @param st [nv, dk, dv] fp32 状态，原地更新。
// @param out [nv, dv] bf16 或 fp32。bf16 是那条链产出的、也是 ``out_proj`` 读的，所以除非
//        调用方需要更宽的容器，就传它；fp32 目的地装的是同一个 bf16 值而不是更精确的值。
// @param record 可选的 ``[width, nv, dk + dv + 1]`` fp32 更新日志；非空时写入归一化 key、
//        本步更新向量和衰减因子，供校验提交时重放。
// @param slot 更新日志中的位置。
// @param qscale ``dk ** -0.5``。
// @param rep 每个键头对应的值头数。
// @return ``out``，便于调用链串接。
// @throws c10::Error dtype、形状或头的几何与内核的索引方式矛盾。几何那一种是值得读的：
//         它意味着 ``dk`` 或 ``dv`` 宽到那条链会用不同的块大小做归约，于是本内核无法复现
//         它的舍入，调用方必须继续走三算子路径。
static torch::Tensor delta_normed_impl(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                       torch::Tensor b, torch::Tensor a,
                                       torch::Tensor neg_expA, torch::Tensor dt_bias,
                                       torch::Tensor z, torch::Tensor nw, torch::Tensor st,
                                       torch::Tensor out, double qscale, int64_t rep,
                                       torch::Tensor record, int64_t slot) {
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "q/k/v must be contiguous");
    TORCH_CHECK(q.scalar_type() == torch::kFloat && k.scalar_type() == torch::kFloat
                    && v.scalar_type() == torch::kFloat
                    && neg_expA.scalar_type() == torch::kFloat
                    && dt_bias.scalar_type() == torch::kFloat
                    && st.scalar_type() == torch::kFloat,
                "q/k/v, neg_expA, dt_bias and the state are fp32");
    // 两个 logit 张量是按 bf16 位模式读的，所以更宽的 dtype 会被重新解释而不是转换——那
    // 是一个静默的错数字，不是一条慢路径。
    TORCH_CHECK(b.scalar_type() == torch::kBFloat16 && a.scalar_type() == torch::kBFloat16,
                "beta and decay logits must be bf16");
    // ``z`` 反过来按 dtype 做模板，因为契约允许 fp32，而内核无论哪种形式都会先舍入成 bf16
    // 再用，和那条链的转换一致。
    TORCH_CHECK(z.scalar_type() == torch::kBFloat16 || z.scalar_type() == torch::kFloat,
                "the norm gate must be bf16 or fp32");
    TORCH_CHECK(nw.scalar_type() == torch::kBFloat16, "the norm gain must be bf16");
    TORCH_CHECK(out.scalar_type() == torch::kBFloat16 || out.scalar_type() == torch::kFloat,
                "output must be bf16 or fp32");
    TORCH_CHECK(out.is_contiguous() && z.is_contiguous(), "out and z must be contiguous");

    const int nv = (int)st.size(0);
    const int dk = (int)st.size(1);
    const int dv = (int)st.size(2);
    TORCH_CHECK(rep > 0 && nv % (int)rep == 0, "head count does not divide by rep");
    TORCH_CHECK(q.size(0) == nv / (int)rep && q.size(1) == dk,
                "q must be [nv / rep, dk]");
    TORCH_CHECK(k.sizes() == q.sizes(), "k and q must have the same shape");
    TORCH_CHECK(v.size(0) == nv && v.size(1) == dv, "v must be [nv, dv]");
    TORCH_CHECK(z.size(0) == nv && z.size(1) == dv, "norm gate must be [nv, dv]");
    TORCH_CHECK(nw.numel() == (int64_t)dv, "norm gain must have dv elements");
    TORCH_CHECK(out.size(0) == nv && out.size(1) == dv, "out view must be [nv, dv]");
    float* record_ptr = nullptr;
    if (record.defined()) {
        TORCH_CHECK(record.is_cuda() && record.device() == st.device()
                    && record.is_contiguous() && record.scalar_type() == torch::kFloat
                    && record.dim() == 3 && record.size(1) == nv
                    && record.size(2) == dk + dv + 1
                    && slot >= 0 && slot < record.size(0),
                    "record must be contiguous [width, nv, dk + dv + 1] fp32 on the state device");
        record_ptr = record.data_ptr<float>();
    }

    // 每个值头一个 block，它的大小由被替代的那条链固定，而不是在这里选。比
    // ``head_threads(dk)`` 更宽或更窄的 block 会用与 ``prologue_kernel`` 不同的伙伴去求 q/k
    // 归一化；而不匹配门控归一化块大小的那个，会对第二次归约做出同样的事——两种情况结果都是
    // 接近但不相等，而那就是挪动贪心 id 的东西。
    const int threads = head_threads(dk);
    TORCH_CHECK(dk <= threads && dv <= threads,
                "delta_normed needs dk and dv within one block (dk=", dk, " dv=", dv,
                " threads=", threads,
                "); keep the delta_prologue/delta/rms_norm_gated chain for this model");
    TORCH_CHECK(norm_threads(dv) == threads,
                "delta_normed needs the prologue's and the gated norm's block sizes to "
                "agree (dk=", dk, " dv=", dv, " prologue=", threads,
                " norm=", norm_threads(dv), "); keep the three-op chain for this model");

    launch_normed(q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
                  reinterpret_cast<const uint16_t*>(b.data_ptr()),
                  reinterpret_cast<const uint16_t*>(a.data_ptr()),
                  neg_expA.data_ptr<float>(), dt_bias.data_ptr<float>(),
                  z.data_ptr(), z.scalar_type() == torch::kBFloat16,
                  reinterpret_cast<const uint16_t*>(nw.data_ptr()), st.data_ptr<float>(),
                  out.data_ptr(), out.scalar_type() == torch::kBFloat16, record_ptr,
                  (int)slot, (float)qscale,
                  nv, dk, dv, (int)rep, threads, c10::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor delta_normed(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                           torch::Tensor b, torch::Tensor a,
                           torch::Tensor neg_expA, torch::Tensor dt_bias,
                           torch::Tensor z, torch::Tensor nw, torch::Tensor st,
                           torch::Tensor out, double qscale, int64_t rep) {
    return delta_normed_impl(q, k, v, b, a, neg_expA, dt_bias, z, nw, st, out,
                             qscale, rep, torch::Tensor(), 0);
}

torch::Tensor delta_normed_record(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                  torch::Tensor b, torch::Tensor a,
                                  torch::Tensor neg_expA, torch::Tensor dt_bias,
                                  torch::Tensor z, torch::Tensor nw, torch::Tensor st,
                                  torch::Tensor out, double qscale, int64_t rep,
                                  torch::Tensor record, int64_t slot) {
    return delta_normed_impl(q, k, v, b, a, neg_expA, dt_bias, z, nw, st, out,
                             qscale, rep, record, slot);
}

static torch::Tensor delta_normed_seq_impl(
        torch::Tensor q, torch::Tensor k, torch::Tensor v,
        torch::Tensor b, torch::Tensor a,
        torch::Tensor neg_expA, torch::Tensor dt_bias,
        torch::Tensor z, torch::Tensor nw, torch::Tensor st,
        torch::Tensor out, double qscale, int64_t rep,
        torch::Tensor record, int64_t slot) {
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3,
                "q/k/v must be [steps, heads, d]");
    TORCH_CHECK(q.scalar_type() == torch::kFloat && k.scalar_type() == torch::kFloat
                    && v.scalar_type() == torch::kFloat
                    && neg_expA.scalar_type() == torch::kFloat
                    && dt_bias.scalar_type() == torch::kFloat
                    && st.scalar_type() == torch::kFloat,
                "q/k/v, neg_expA, dt_bias and the state are fp32");
    // 与单 token 入口同一个理由：这两个 logit 张量按 bf16 位模式读，更宽的 dtype 会被
    // 重新解释而不是转换。
    TORCH_CHECK(b.scalar_type() == torch::kBFloat16 && a.scalar_type() == torch::kBFloat16,
                "beta and decay logits must be bf16");
    TORCH_CHECK(z.scalar_type() == torch::kBFloat16 || z.scalar_type() == torch::kFloat,
                "the norm gate must be bf16 or fp32");
    TORCH_CHECK(nw.scalar_type() == torch::kBFloat16, "the norm gain must be bf16");
    TORCH_CHECK(out.scalar_type() == torch::kBFloat16 || out.scalar_type() == torch::kFloat,
                "output must be bf16 or fp32");
    // q/k/v 允许行距大于行宽（调用方传的是同一块卷积输出里切出来的视图），b/a/z 同理
    // ——它们是融合投影 ``split`` 出来的，行距是融合输出的总宽。所以传行距而不是要求
    // 连续：要求连续就得在这里多做 c 次 ``contiguous()``，把内核省下的发射又付回去。
    TORCH_CHECK(q.stride(2) == 1 && k.stride(2) == 1 && v.stride(2) == 1
                    && b.stride(1) == 1 && a.stride(1) == 1 && z.stride(2) == 1,
                "the innermost dim of q/k/v/b/a/z must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

    const int steps = (int)q.size(0);
    const int nv = (int)st.size(0);
    const int dk = (int)st.size(1);
    const int dv = (int)st.size(2);
    TORCH_CHECK(steps >= 1, "steps must be >= 1");
    TORCH_CHECK(k.size(0) == steps && v.size(0) == steps && b.size(0) == steps
                    && a.size(0) == steps && z.size(0) == steps && out.size(0) == steps,
                "every argument must have the same number of steps");
    TORCH_CHECK(rep > 0 && nv % (int)rep == 0, "head count does not divide by rep");
    TORCH_CHECK(q.size(1) == nv / (int)rep && q.size(2) == dk, "q must be [steps, nv / rep, dk]");
    TORCH_CHECK(k.sizes() == q.sizes(), "k and q must have the same shape");
    TORCH_CHECK(v.size(1) == nv && v.size(2) == dv, "v must be [steps, nv, dv]");
    TORCH_CHECK(z.size(1) == nv && z.size(2) == dv, "norm gate must be [steps, nv, dv]");
    TORCH_CHECK(nw.numel() == (int64_t)dv, "norm gain must have dv elements");
    TORCH_CHECK(out.size(1) == nv && out.size(2) == dv, "out must be [steps, nv, dv]");
    float* record_ptr = nullptr;
    if (record.defined()) {
        TORCH_CHECK(record.is_cuda() && record.device() == st.device()
                    && record.is_contiguous() && record.scalar_type() == torch::kFloat
                    && record.dim() == 3 && record.size(1) == nv
                    && record.size(2) == dk + dv + 1
                    && slot >= 0 && slot + steps <= record.size(0),
                    "record must be contiguous [width, nv, dk + dv + 1] fp32 on the state device");
        record_ptr = record.data_ptr<float>();
    }
    // 块大小同样由被替代的那条链固定，理由见 ``delta_normed_impl``。
    const int threads = head_threads(dk);
    TORCH_CHECK(dk <= threads && dv <= threads,
                "delta_normed_seq needs dk and dv within one block (dk=", dk, " dv=", dv,
                " threads=", threads, ")");
    TORCH_CHECK(norm_threads(dv) == threads,
                "delta_normed_seq needs the prologue's and the gated norm's block sizes to "
                "agree (dk=", dk, " dv=", dv, " prologue=", threads,
                " norm=", norm_threads(dv), ")");

    launch_normed_seq(q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
                      reinterpret_cast<const uint16_t*>(b.data_ptr()),
                      reinterpret_cast<const uint16_t*>(a.data_ptr()),
                      neg_expA.data_ptr<float>(), dt_bias.data_ptr<float>(),
                      z.data_ptr(), z.scalar_type() == torch::kBFloat16,
                      reinterpret_cast<const uint16_t*>(nw.data_ptr()), st.data_ptr<float>(),
                      out.data_ptr(), out.scalar_type() == torch::kBFloat16, record_ptr,
                      (int)slot, (float)qscale,
                      nv, dk, dv, (int)rep, steps,
                      (long long)q.stride(0), (long long)k.stride(0),
                      (long long)v.stride(0), (long long)b.stride(0),
                      (long long)z.stride(0),
                      threads, c10::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor delta_normed_seq(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                               torch::Tensor b, torch::Tensor a,
                               torch::Tensor neg_expA, torch::Tensor dt_bias,
                               torch::Tensor z, torch::Tensor nw, torch::Tensor st,
                               torch::Tensor out, double qscale, int64_t rep) {
    return delta_normed_seq_impl(q, k, v, b, a, neg_expA, dt_bias, z, nw, st, out,
                                 qscale, rep, torch::Tensor(), 0);
}

torch::Tensor delta_normed_seq_record(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                      torch::Tensor b, torch::Tensor a,
                                      torch::Tensor neg_expA, torch::Tensor dt_bias,
                                      torch::Tensor z, torch::Tensor nw, torch::Tensor st,
                                      torch::Tensor out, double qscale, int64_t rep,
                                      torch::Tensor record, int64_t slot) {
    return delta_normed_seq_impl(q, k, v, b, a, neg_expA, dt_bias, z, nw, st, out,
                                 qscale, rep, record, slot);
}

// 绑定写在内核旁边：名字、参数名与默认值只有这一份，`bindings.cpp` 只遍历注册表。
//门控 DeltaNet。
COFFIN_BIND(conv1d_silu,
            "depthwise causal conv1d + SiLU for one token, shifts the window in place");
COFFIN_BIND(conv1d_silu_seq, "the same conv over a prefill chunk [T, C]");
COFFIN_BIND(conv1d_silu_batch, "batched depthwise causal conv1d + SiLU, one window per row");
COFFIN_BIND(delta_prologue, "q/k L2 norm, head expansion and the beta/g gates for one token");
COFFIN_BIND(delta_prologue_batch,
            "batched gated-delta prologue, leading row dim on every argument");
COFFIN_BIND(delta_prep_seq, "prefill head prep into the head-major [H, T, d] layout");
COFFIN_BIND(delta_normed, "prologue + recurrence + gated norm for one decode token, one launch");
COFFIN_BIND(delta_normed_record, "融合 gated delta 校验步并记录秩一更新");
COFFIN_BIND(delta_normed_seq, "c 个校验 token 的融合链，一次启动跑完整个投机窗口");
COFFIN_BIND(delta_normed_seq_record, "c 个校验 token 的融合链并逐 token 记录秩一更新");
