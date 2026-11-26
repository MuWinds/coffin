// @file conv_device.inc
// @brief 线性注意力入口处的深度因果卷积（4 抽头）接 SiLU：`conv1d_silu` 族的设备代码。
//        ``linear_attn.cu``（torch 前端）与 ``abi/conv_abi.cu``（C ABI 前端）各包含一次。
//
// Module: coffin/backends/corex/kernels。只含设备代码：三个常量与两个 ``__global__``。
// 批处理入口启动的就是单 token 那条内核（行轴收成 ``rows``），所以两条路径的算术
// 不可能各自漂移——这也是为什么只搬两个内核而不是三个。
//
// 为什么先搬这一族：它是 qwen35 每条线性注意力层的入口，形状简单（一维通道 + 一个
// 三元素历史环）、没有块表也没有分组偏移，唯一要小心的是**行间距**：校验主体传进来的是
// ``qkv[:, j]``，即 ``[rows, c, conv_dim]`` 视图的一列切片，其行间距是 ``c * C``。按 ``C``
// 去读会让第 1 行及以后都取到第 0 行的通道块——``rows == 1`` 时看不出来，宽度大于 1 就是
// 静默的错答案。
//
// 依赖：``corex_scalar.h`` 的 ``CxScalar``（必须在包含本文件之前引入）。

//: 卷积抽头数：4 个 token 的窗口。
constexpr int CONV_K = 4;

//: 环里保留的历史输入数。
constexpr int CONV_STATE = CONV_K - 1;

//: 预填充沿 ``T`` 的分段数。8 是扫出来的（1/2/4/8/16 五档 us/call 968.6 / 527.0 /
//: 307.9 / 207.6 / 229.0），且每一档的输出与 state 都与不分段版逐位相同。
constexpr int CONV_NSEG = 8;

// @brief 单 token 的深度因果卷积（4 抽头）接 SiLU。
//
// 线程 ``c`` 独占通道 ``c`` 和它的三抽头窗口，所以对状态的「先读后写」不需要同步：没有
// 别的线程碰那个窗口。
//
// 网格的 ``y`` 是批行。一行的输入和窗口是分开的板，所以行轴只花一个 block 下标、行与行
// 之间没有流量；间距是传进来的而不是假定的。
//
// @param w [C, 4] fp32 抽头，最老的在前；最后一个抽头乘当前输入。
// @param x [B, C] bf16 投影输出，行间距 ``ldx``。
// @param state [B, C, 3] fp32 的历史输入，带行间距 ``lds`` 原地平移。
// @param out [B, C] fp32 的卷积加 SiLU 结果，行间距 ``ldo``。
// @param C 通道数，即 ``x`` 网格覆盖的宽度。
// @param ldx ``x`` 的行间距，以元素计。
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
// 一遍 fp32。
//
// 分段合法是因为这条卷积是因果的：段 ``s`` 的起点状态从原始输入重放前 ``CONV_STATE``
// 个 token 重建（那 3 步的递推与主循环是同一个表达式，所以重建出的历史与整条走下来逐位
// 相同）。只有最后一段把最终窗口写回；中间段不回写，否则后一段会读到被前一段覆盖的状态。
//
// @param w [C, 4] fp32 抽头。
// @param x [T, C] bf16 激活；行间距 ``ldx`` 可以大于 ``C``。
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
        // 只用来把寄存器环推到位。
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
    if (t1 == T) {
        s[0] = s0;
        s[1] = s1;
        s[2] = s2;
    }
}

// @brief 预填充分段的段长与段数：段数按 ``T`` 截断，短序列不发出空段。
//
// @param T token 数。
// @param nseg 输出，实际段数。
// @return 每段覆盖的 token 数。
inline int conv_seg_len(int T, int& nseg) {
    nseg = T < CONV_NSEG ? T : CONV_NSEG;
    return (T + nseg - 1) / nseg;
}
