// @file attn_device.inc
// @brief 注意力族里四个「小算子」的设备代码：``silu_mul`` / ``silu_mul_quant`` /
//        ``rope_apply`` / ``kv_quant``，以及它们共用的 ``block_reduce`` /
//        ``kv_quant_threads``。
//
// Module: coffin/backends/corex/kernels。``attn.cu``（torch 前端）与
// ``abi/attn_abi.cu``（C ABI 前端）各包含一次，所以两条前端的算术逐位相同。
//
// 为什么只有这四个先搬出来：它们没有 split-K、没有 KV 布局、没有块表，输入就是
// 「一片带行间距的二维视图」或者「一组稠密头」，是最容易逐个对上 ABI 的那一档。
// ``attn_decode`` / ``attn_paged*`` 那几条要连 KV 布局一起定契约，留到后面单独做。
//
// 依赖：``corex_scalar.h`` 的 ``CxScalar``（必须在包含本文件之前引入）。

// @brief 在带间距的 ``[M, W]`` 视图上算 ``out = silu(gate) * up``，每条通道一个元素。
//
// 融合的 ``gate_up`` 投影一次读取权重就交出两半，所以它和 down 投影之间唯一的工作就是这个
// 乘积。拆成 ``aten::silu`` 加 ``aten::mul``，每个 MLP 是两次启动加两个
// ``[intermediate]`` 临时张量；一个解码步有 32 个 MLP，于是一个元素一次乘加要 64 次启动。
//
// **那两次舍入不是意外。** bf16 张量上的 ``aten::silu`` 返回 bf16，所以那条链对激活舍入
// 一次，``aten::mul`` 再对乘积舍入一次。用 fp32 算 ``silu`` 只对乘积舍入，是最自然的实现，
// 而它在约四分之一的元素上与 torch 不一致——那会改变 logits，进而改变 token id。所以激活
// 在乘法之前先窄化回存储类型，与那两算子链完全一样。实测：保留中间窄化时，五组独立的
// 20 万元素试验与 ``F.silu(gate) * up`` 逐位相同；不保留时约 24% 的元素不同。
//
// **间距才是关键。** 投影以融合组的形式调用，结果用 ``.split(dim=1)`` 拆回两半，在二维结果
// 上那是更宽缓冲区的 *列* 视图：行之间相隔 ``pitch`` 个元素且 ``pitch > W``。只支持稠密的
// 平坦内核会逼调用方先把两半实体化，那又是两次启动两次分配——严格劣于它替代的那两个
// torch 算子。所以 block 走 ``(row, column)`` 而不是平坦下标，间距作为参数传递。解码路径
// 上 ``W`` 就是整个切分、间距等于它，于是同一个内核两条路径都服务，不需要分支。
//
// 允许 ``out`` 与 ``up`` 重叠，而且每条通道都在写 ``out[r][c]`` 之前读 ``gate[r][c]`` 和
// ``up[r][c]``，没有通道碰别人的下标，所以共享间距时它就是纯原地更新。这正是调用点能把
// 投影自己的缓冲区交回来、什么都不分配的原因。
//
// @param gate [M, W] 门那一半；最内层步长为 1。
// @param up [M, W] 另一半，形状相同；最内层步长为 1。
// @param out [M, W] 目的地；最内层步长为 1，可以与 ``up`` 重叠。
// @param width ``W``，每行贡献的列数。
// @param g_pitch ``gate`` 行之间的元素间距。
// @param u_pitch ``up`` 行之间的元素间距。
// @param o_pitch ``out`` 行之间的元素间距。
// @return 无；``out`` 归调用方所有。
// @throws 设备侧不抛。包装层在主机侧校验布局。
template <typename T>
__global__ void silu_mul_kernel(const T* __restrict__ gate, const T* __restrict__ up,
                                T* __restrict__ out, int width, long g_pitch,
                                long u_pitch, long o_pitch) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= width) return;
    const long r = blockIdx.y;
    const long gi = r * g_pitch + c;
    const long ui = r * u_pitch + c;
    // sigmoid 在 fp32 里算——bf16 的 ``1 + expf(-g)`` 会把分母的整个尾数丢掉——然后再窄化到
    // 存储类型，因为 bf16 张量上的 ``aten::silu`` 就是这么做的，下面第二次舍入也是相对它
    // 定义的。
    const float g = CxScalar<T>::to_float(gate[gi]);
    const T s = CxScalar<T>::from_float(g / (1.f + __expf(-g)));
    out[r * o_pitch + c] = CxScalar<T>::from_float(CxScalar<T>::to_float(s)
                                                   * CxScalar<T>::to_float(up[ui]));
}

// @brief 旋转每个 q/k 头最前面 ``rotary_dim`` 个通道，每条通道一个线程。
//
// 写成 torch 链的 RoPE 每个全注意力层要十到十二次启动：两次表 ``_to_copy`` 转换、四次
// ``mul``、两次为 ``rotate_half`` 的 ``cat``、两次 ``add``，以及两次把未旋转尾部接回去的
// ``cat``。8 层下来就是解码步启动数里的约 80-96 次，全都作用在最多 ``[16, 256]`` 的张量上
// ——几乎纯粹是启动下限。一次启动全替代掉。
//
// **那四次舍入是契约**，弄错是静默的。``apply_rope`` 先跑 ``cos.to(q.dtype)``，再在 bf16
// 张量上算 ``x*cos + rotate_half(x)*sin``，所以 torch 会窄化：cos/sin 一次，每个乘积各一
// 次，和再一次。已在主机侧对着真正的 ``ops/rope.py`` 验证：四次舍入都在时结果逐位相同，
// 而每省略一次都可测——
//   * cos/sin 用 fp32、其余都对：1024 个元素里有 296 个不同；
//   * 全程 fp32 算术、最后只舍入一次：每个 token 有 435 个元素不同。
// 两者都会改变 logits。所以这里逐元素窄化表，并对三个运算各自单独窄化，而不是写成一个
// fp32 表达式。
//
// ``rotate_half`` 把通道 ``i`` 与 ``i + rotary_dim/2`` 配对；那是与建表所用
// ``cat(half, half)`` 布局之间的契约（``ops/rope.py``）。只有最前面的 ``rotary_dim`` 个
// 通道旋转——尾部的 ``hd - rotary_dim`` 原样通过，部分旋转的 checkpoint 依赖这一点。
//
// @param q [M, nh, hd] 查询，bf16（以 ``uint16_t`` 承载）。
// @param k [M, nkv, hd] 键，dtype 相同。
// @param cos,sin [M, rotary_dim] fp32，已经为这些 token 收集好的行。
// @param q_out [M, nh, hd] 旋转后的查询，原地写入；不得与 ``q`` 重叠。
// @param k_out [M, nkv, hd] 旋转后的键，原地写入；不得与 ``k`` 重叠。
// @param nh,nkv 头数；它们只影响下标算术。
// @param hd 头的宽度。
// @param rot ``rotary_dim``，要旋转的前导通道数。
// @return 无；两个目的地都归调用方所有。
// @throws 设备侧不抛。包装层在主机侧校验布局。
template <typename T>
__global__ void rope_apply_kernel(const T* __restrict__ q, const T* __restrict__ k,
                                  const float* __restrict__ cos,
                                  const float* __restrict__ sin,
                                  T* __restrict__ q_out, T* __restrict__ k_out,
                                  int nh, int nkv, int hd, int rot) {
    // 网格：x 覆盖一个头的通道，y 是 token，z 选 q（0）还是 k（1）。于是 token 和两个张量
    // 互相独立，而解码路径就是 ``M == 1``。
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= hd) return;
    const int m = blockIdx.y;
    const bool is_k = blockIdx.z != 0;
    const T* src = is_k ? k : q;
    T* dst = is_k ? k_out : q_out;
    const int heads = is_k ? nkv : nh;
    const int rot_half = rot >> 1;

    // 旋转前缀里的通道 ``c`` 与相距 ``rot_half`` 的通道配对；超出前缀的部分原样拷过去。
    // 符号属于 *源* 通道所在的那一半，因为 ``rotate_half`` 在配对之前先对它的后一半取负。
    const bool rotated = c < rot;
    const int pair = c < rot_half ? c + rot_half : c - rot_half;
    const float sign = (pair >= rot_half) ? -1.f : 1.f;
    // 表行在使用之前先窄化成存储类型，因为 ``apply_rope`` 的 ``cos.to(q.dtype)`` 就是这么
    // 做的。在这里保持 fp32 是直觉选择，而它会改变约 29% 的元素。
    const T cv = CxScalar<T>::from_float(rotated ? cos[(size_t)m * rot + c] : 0.f);
    const T sv = CxScalar<T>::from_float(rotated ? sin[(size_t)m * rot + c] : 0.f);

    for (int h = 0; h < heads; ++h) {
        const size_t base = ((size_t)m * heads + h) * (size_t)hd;
        if (!rotated) {
            dst[base + c] = src[base + c];
            continue;
        }
        const float x = CxScalar<T>::to_float(src[base + c]);
        const float xr = sign * CxScalar<T>::to_float(src[base + pair]);
        // 三次窄化，对应 torch 链做的每个运算各一次：乘积、乘积、和。把它们融合成一个 fp32
        // 表达式会改变 logits。
        const T a = CxScalar<T>::from_float(x * CxScalar<T>::to_float(cv));
        const T b = CxScalar<T>::from_float(xr * CxScalar<T>::to_float(sv));
        dst[base + c] = CxScalar<T>::from_float(CxScalar<T>::to_float(a)
                                               + CxScalar<T>::to_float(b));
    }
}

// @brief 先按「核里 silu_mul 的同一套舍入」写 bf16，再顺手量化成 int8 与逐行 scale。
//
// MoE 的 down 投影吃的是 SwiGLU 之后的那份激活的 int8 版本。原来那是两次启动：一次
// ``silu_mul`` 写 ``[slots, I]`` bf16，紧跟一次 ``quant_act_i8`` 把同一片字节读回来
// 求 amax、再写 int8。8 行 x 512 的中间量只有 8 KiB，两趟搬的是同一份。
//
// 舍入次序是契约：amax 必须按**落下去的那个 bf16** 算，否则和「先落盘再量化」差最后
// 一位。所以这里先写 ``out``，再在读回同一个地址的过程中求 amax（同一条线程序写自己
// 的那个元素，不跨线程，不需要屏障），最后按 ``1e-5`` 下限与 ``__float2int_rn`` 截断
// 到 ``[-127, 127]``——与 ``quant_act_i8_kernel`` 逐位相同。
//
// @param gate [rows, W]（或一维一行）门那一半，最内层步长 1。
// @param up 另一半，形状同 ``gate``。
// @param out 目的地，bf16 或 fp32，可以与 ``up`` 重叠（MoE 就是原地写）。
// @param out_q [rows, W] 连续 int8。
// @param out_s [rows] fp32 逐行 scale。
// @param width ``W``；``g_pitch/u_pitch/o_pitch`` 是行间距。
// @return 无；三份目的地都归调用方所有。
// @throws 设备侧不抛。包装层在主机侧校验布局。
template <typename T>
__global__ void silu_mul_quant_kernel(const T* __restrict__ gate,
                                      const T* __restrict__ up,
                                      T* __restrict__ out,
                                      int8_t* __restrict__ out_q,
                                      float* __restrict__ out_s,
                                      int width, long g_pitch, long u_pitch,
                                      long o_pitch) {
    const long r = blockIdx.x;
    const T* gr = gate + r * g_pitch;
    const T* ur = up + r * u_pitch;
    T* orow = out + r * o_pitch;
    int8_t* qr = out_q + (size_t)r * width;

    __shared__ float smax[256];
    for (int c = (int)threadIdx.x; c < width; c += (int)blockDim.x) {
        // 与 ``silu_mul_kernel`` 同一套运算与同样的两次窄化：sigmoid 在 fp32 里算、
        // 先窄化 ``s``、乘积再窄化一次。省掉任何一次都改变 logits。
        const float g = CxScalar<T>::to_float(gr[c]);
        const T s = CxScalar<T>::from_float(g / (1.f + __expf(-g)));
        orow[c] = CxScalar<T>::from_float(CxScalar<T>::to_float(s)
                                          * CxScalar<T>::to_float(ur[c]));
    }
    // amax 按落盘的值算，等价于「先跑 silu_mul 再跑 quant_act_i8」看到的输入。
    float amax = 0.f;
    for (int c = (int)threadIdx.x; c < width; c += (int)blockDim.x) {
        amax = fmaxf(amax, fabsf(CxScalar<T>::to_float(orow[c])));
    }
    smax[threadIdx.x] = amax;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smax[threadIdx.x] = fmaxf(smax[threadIdx.x], smax[threadIdx.x + s]);
        }
        __syncthreads();
    }
    const float sc = fmaxf(smax[0], 1e-5f) / 127.0f;
    const float inv = 1.0f / sc;
    if (threadIdx.x == 0) out_s[r] = sc;
    for (int c = (int)threadIdx.x; c < width; c += (int)blockDim.x) {
        int q = __float2int_rn(CxScalar<T>::to_float(orow[c]) * inv);
        q = max(-127, min(127, q));
        qr[c] = (int8_t)q;
    }
}

#include "attn_reduce_device.inc"

// @brief 一个量化块的线程数，取 2 的幂。
//
// ``block_reduce`` 从 ``blockDim.x`` 开始对半砍，所以只有块是 2 的幂时它才覆盖整个块；
// 192 线程的块会按 192、96、48、24、12、6、3、1 归约，并悄悄丢掉尾巴。norm 文件里的归约
// 有同样的要求，那里改为按行宽定块大小，而那里恰好是 2 的幂。
//
// @param N 行宽，也就是这个块必须覆盖的量。
// @return ``min(1024, round_up_pow2(max(64, N)))``——不低于一个 wavefront，也不超过设备的
//         块上限。
// @throws 无。
__host__ int kv_quant_threads(int N) {
    int t = 64;
    while (t < N && t < 1024) t <<= 1;
    return t;
}

// @brief 量化 ``[M, N]`` 板的每一行，每块一行。
//
// 这里的算术是 int8 KV 缓存的契约，不是实现选择，并且按 torch 自己的运算顺序和 dtype
// 逐条复现：
//
//     xf = x.float()                       // 精确拓宽 bf16
//     s  = max(amax(xf), 1e-5) / 127.0     // fp32 除法
//     q  = round(xf / s)                   // fp32 除法，就近取偶
//     q  = clamp(q, -127, 127)             // 码值留在对称范围内
//
// ``attn_slice_kernel`` 会把存下来的 scale 乘回去，所以这里任何偏差——换一种除法、换一种
// 平局规则、把下限放到除法之后——都会让写方和读方处在不同的 scale 上，注意力输出随之漂移
// 而任何形状检查都不会失败。``check_kv_quant.py`` 因此是逐位比较而不是在容差内比较。
//
// 一块一行而不是一行一个 warp：这个模型里 ``hd`` 是 256，一行是 512 字节，一块一行让归约
// 只跨一道屏障，同时给每个块足够的通道一趟覆盖一行。批处理和预填充的调用点会递给它多到
// 几千行，所以无论哪种情况网格都宽得能喂满这张卡。
//
// @param x [M, N] 激活，bf16（以 ``uint16_t`` 承载）或 fp32；行间距是 ``pitch``。
// @param out_q [M, N] int8 码值，稠密打包，原地写入。
// @param out_s [M] fp32 每行 scale，原地写入。
// @param M 行数，即网格的 x 方向长度。
// @param N 行宽。
// @param pitch ``x`` 行之间的元素间距；``>= N``。
// @return 无；两个目的地都归调用方所有。
// @throws 设备侧不抛。包装层在主机侧校验布局。
template <typename T>
__global__ void kv_quant_kernel(const T* __restrict__ x, int8_t* __restrict__ out_q,
                                float* __restrict__ out_s, int N, long pitch) {
    extern __shared__ float red[];
    const int row = blockIdx.x;
    const int t = threadIdx.x;
    const T* xr = x + (size_t)row * pitch;

    // 归约用 ``fmaxf``，它丢掉 NaN 操作数，而不是像 torch 的 ``amax`` 那样传播它。到达
    // KV 缓存的激活是有限的——这里出现 NaN 的话注意力输出早就已经是 NaN 了——所以选的是
    // 更便宜的那个，并且把这个差异写出来，而不是留给人去发现。
    float amax = 0.f;
    for (int i = t; i < N; i += blockDim.x) {
        const float v = fabsf(CxScalar<T>::to_float(xr[i]));
        if (v > amax) amax = v;
    }
    amax = block_reduce(amax, red, true);

    // 先 ``clamp_min`` 再 ``/ 127``，这个顺序，在 fp32 里做。下限让全零行不会除以零；它的
    // 代价不到一个 int8 步长，所以不可能把一个有限值挪过 int8 本来就会舍入到的位置。它与
    // 读方是一项约定而不是一个容差，所以这里写成显式常量而不是折进去。
    const float s = fmaxf(amax, 1e-5f) / 127.0f;

    int8_t* qr = out_q + (size_t)row * N;
    for (int i = t; i < N; i += blockDim.x) {
        // ``__float2int_rn`` 就是 ``cvt.rni.s32.f32``：就近取偶。那正是 torch 的
        // ``.round()``。``rintf`` 说的是同一件事，但要经过一个 float 再转换，而转换的越界
        // 行为会成为第二个要推理的东西。按构造 |x / s| 被 127 界住，所以在夹取之前整数就
        // 在范围内；夹取仍然保留，因为「按构造被界住」这种说法会在某一次悄悄出错，而且
        // 只错那一次。
        int v = __float2int_rn(CxScalar<T>::to_float(xr[i]) / s);
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        qr[i] = (int8_t)v;
    }
    if (t == 0) out_s[row] = s;
}
