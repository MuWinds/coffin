// @file attn.cu
// @brief 在 int8 KV 缓存上做融合的单 token 注意力，外加喂给它的逐元素胶水：KV 量化器
//        和 SwiGLU 乘积。
//
// Module: coffin/backends/corex/kernels。绑定成 ``attn_decode`` / ``attn_paged`` /
// ``attn_paged_batch`` / ``attn_paged_multi_q`` / ``kv_quant`` / ``silu_mul``，只由
// ``backends/corex/__init__.py`` 调用。
//
// 为什么融合：eager 路径把整个 KV 窗口实体化成 bf16——两份 ``[nkv, S, hd]`` 拷贝——
// 再把分数 einsum、因果掩码、softmax、值 einsum 和输出门当成分离的内核跑。每个 token
// 有 8 个全注意力层是这么干的。本内核把缓存留在 int8，边读边反量化，并在同一趟里施加
// 每 token 的 scale、softmax 和门，所以窗口中间结果从不落盘。
//
// ``kv_quant`` 和 ``silu_mul`` 放在这里而不是各自一个编译单元，因为它们是对同一步做的
// 同一类改动：各自把全注意力层或 MLP 层里一条六到七次启动的 torch 链换成一次启动。
// ``kv_quant`` 还和注意力内核共用 ``block_reduce``，而量化器的算术正是
// ``attn_slice_kernel`` 读取方所做假设的一半，所以把这对放在一个文件里，能让它们之间的
// 一致性在一处可复核。
//
// Constraints:
//   * 每次归约都是共享内存树，绝不用 ``__shfl_down_sync``。本部件的 warp 是 64 通道，
//     为 32 写的蝶形只加一半 block 就返回一个看着合理但错误的 softmax 分母。树的尺寸在
//     启动时按 ``blockDim`` 定，所以块大小保持自由参数——但必须是 2 的幂，因为树从
//     ``blockDim.x`` 开始对半砍，192 通道的块会丢掉尾巴。
//   * 窗口按块流式处理并用在线 softmax（运行最大值、运行和，外加被重新缩放的累加器），
//     所以共享内存不随上下文长度增长。把 ``[S]`` 的分数向量暂存下来，在 262144 时是
//     1 MB，根本启动不了。
//   * ``pos`` 在内核里从设备内存读取，绝不按值传入。解码步会被捕获成图，而标量内核参数
//     在捕获那一刻就冻住了——重放会永远算同一个位置，那正是 COREX_PORT.md §0 记录的
//     陈旧输出 bug。
//   * Split-K 从 ``part_acc`` 的轴上读，而不是写死，这样 ``caps.attn_split`` 和调用方的
//     缓冲区是唯一的事实来源。本后端当前声明 ``attn_split=1``（不切分）；内核两种宽度都
//     处理得了，所以将来提高上限不需要改内核。
//   * 稠密和分页通过 ``pos2slot`` 这层间接共用同一个内核，空映射即稠密布局。一条代码路径
//     意味着分页的答案不可能偏离稠密的答案，而一致性检查依赖的正是这个性质。
//   * 门在合并那一趟、归一化之后施加，所以它从不进 softmax，也改不了注意力权重。
//   * INT8 码保持有符号，每个（键，头）的 scale 在进入分数时施加，与参考实现的「先反量化
//     再相乘」顺序一致。改在概率上施加会是另一次舍入。
//   * 量化器的 ``1e-5`` 下限和 ``127`` 除数是与那个读取方的 *契约*，不是调优选择。
//     ``kv_quant`` 精确复现参考链的运算顺序和 dtype——fp32 除以 scale、就近取偶舍入——
//     任何对它们的「优化」都会让写方和读方失配，而且不会让任何形状检查失败。
//   * ``B * c`` 个草稿查询被压平到网格的 z 轴上，因为网格是三维的。传入 ``draft`` 是为了
//     让内核能从那个平坦下标还原 ``(行, 查询)``，而不必在主机侧做除法。

#include "corex_ops.h"
#include "corex_scalar.h"

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

namespace {

//: 每个 block 流式处理多少个键。把共享内存限制在几 KB 加每块一次输出累加器重缩放，
//: 与注意力窗口长度无关。
constexpr int KEY_BLOCK = 1024;

//: 「还没见过键」的哨兵值。大而有限，让第一块的 ``mnew - m`` 保持精确；``-inf`` 会把
//: ``exp(m - mnew)`` 那次重缩放变成 NaN。
constexpr float NO_MAX = -1e30f;

// @brief 通过共享树做块内最大值或求和。
//
// @param v 本线程的贡献。
// @param red ``blockDim.x`` 个 float 的共享暂存；会被覆写。
// @param is_max 求最大值时为 ``true``，求和时为 ``false``。
// @return 块内结果，每个线程都拿得到。每个线程都必须调用它：它以屏障收尾。
//
// 结尾的屏障是必需的。``red[0]`` 会被每个线程读取，而下一次 ``block_reduce`` 调用会在它
// 自己的屏障之前先写 ``red[threadIdx.x]``，所以没有这道屏障，快线程会把慢线程还没读的
// 结果覆盖掉。注意力循环每个键块调它两次——先最大值后求和，中间夹着权重数组——所以这个
// 竞态在第一个 token 的第一个块上就会出现。
__device__ __forceinline__ float block_reduce(float v, float* red, bool is_max) {
    red[threadIdx.x] = v;
    __syncthreads();
    for (int step = blockDim.x >> 1; step > 0; step >>= 1) {
        if ((int)threadIdx.x < step) {
            red[threadIdx.x] = is_max ? fmaxf(red[threadIdx.x], red[threadIdx.x + step])
                                      : red[threadIdx.x] + red[threadIdx.x + step];
        }
        __syncthreads();
    }
    const float total = red[0];
    __syncthreads();
    return total;
}

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

// @brief kv 头 ``g`` 的第 ``s`` 个键的平坦行号，稠密或分页。
//
// 稠密缓存是头主序（``[nkv, S, hd]``，行 ``g*S + s``），分页池是 token 主序
// （``[slots, nkv, hd]``，行 ``slot*nkv + g``）。一个公式覆盖两者，所以 k、v 和两个每 token
// 的 scale 全都用同一个行号索引。
//
// @param s 窗口内的位置。
// @param g KV 头。
// @param pos2slot 设备侧位置到槽位的映射，稠密布局下为 null。
// @param stride 稠密是 ``S``，分页是 ``nkv``。
// @param map_base 本次查询的槽位表行起始的元素下标。
// @return k、v 和两个 scale 共用的平坦行号。
__device__ __forceinline__ int cache_row(int s, int g, const int64_t* pos2slot,
                                         int stride, int map_base) {
    return pos2slot ? (int)((long)pos2slot[map_base + s] * stride + g) : g * stride + s;
}
//
// ``map_base`` 是参数而不是写死的 ``b * max_seq``，因为两个调用方传进来的东西不一样，而
// 只有调用方知道自己传的是哪种：单行入口传的是切片（``pos2slot[row]``，已经偏过，所以它的
// base 是 0），批处理入口传的是整个 ``[B, max_seq]`` 映射。按平坦方式读它完全忽略了行——
// 第 0 行以上的每一行都在看第 0 行的键，实测为三行批处理结果完全相同、而按行参考实现彼此
// 相差 34.5（``docs/SLOT_MAP_BUG.md``）。

//: 布局选择器，所有入口共用。一个结构体让 :func:`slice_launch` 的参数列表保持可读，也把
//: ``pos2slot`` 与 ``ctx_lens`` 的空对空搭配显式化。
struct AttnLayout {
    //: 设备侧位置到槽位的映射，稠密缓存下为 null。
    const int64_t* pos2slot;
    //: 设备侧每行键数，为 null 时窗口由 ``pos`` 推出。
    const int32_t* ctx_lens;
    //: 设备侧位置；只在 ``ctx_lens`` 为 null 时读取。
    const int64_t* pos;
    //: 行间距：稠密是 ``S``，分页是 ``nkv``。
    int stride;
    //: 窗口上限，即槽位表的行间距。
    int max_seq;
    //: ``ctx_lens`` 装的是校验锚点的 *最后下标* 而不是键数。
    //:
    //: 两个调用方在单位上不一致，而对各自的形状来说都对：``attn_paged_batch`` 传
    //: ``positions + 1``（本身已经是计数），而 ``attn_paged_multi_q`` 传
    //: ``positions[:, 0]``（锚点下标）。校验窗口的第 ``j`` 个查询必须看到
    //: ``anchor + 1 + j`` 个键——漏掉这一项就是记录在 :func:`attn_slice_kernel` 里 ``ctx``
    //: 计算处的那个 bug——而 ``j`` 在两个调用点都不在作用域内，所以转换只能在这里做。
    //:
    //: 放在最后而不是紧挨 ``ctx_lens``，是为了让每个聚合初始化保持位置含义：插在前面会悄悄
    //: 把 ``max_seq`` 重新绑上，窗口就变成 0。
    bool ctx_anchor;
    //: ``pos2slot`` 是整个 ``[B, max_seq]`` 映射还是一行的切片。
    //:
    //: ``attn_paged`` 和 ``attn_decode`` 传 ``pos2slot[row]``，一个首元素已经是该行第一个
    //: 位置的视图，所以它们的偏移是 0。``attn_paged_batch`` 和 ``attn_paged_multi_q`` 传
    //: 整个映射。没有这个标志时内核按平坦方式读，第 0 行以上的每一行都在看第 0 行的键——
    //: 实测为三行批处理结果完全相同、而按行参考实现彼此相差 34.5
    //: （``docs/SLOT_MAP_BUG.md``）。放在末尾的理由与 ``ctx_anchor`` 相同：位置式聚合
    //: 初始化不能把 ``max_seq`` 重新绑掉。
    bool map_rows;
};

// @brief 每个（查询头，键切片，行）一个 block；在线 softmax 的一片。
//
// @param q [rows * nh, hd] bf16 查询，调用方已经旋转并缩放好。
// @param kc,vc int8 缓存；布局由 :struct:`AttnLayout` 选。
// @param ks,vs fp32 每（槽位，KV 头）的反量化 scale。
// @param block_mask uint8 ``[B, n_blocks]`` 保留掩码，精确注意力时为 null。
// @param part_acc [rows, nh, split, hd] fp32 分片分子。
// @param part_ml [rows, nh, split, 2] fp32 分片（最大值，和）。
// @param draft 共用一行历史的查询数；普通解码为 1。
__global__ void attn_slice_kernel(const uint16_t* __restrict__ q,
                                  const uint8_t* __restrict__ block_mask,
                                  const int8_t* __restrict__ kc,
                                  const int8_t* __restrict__ vc,
                                  const float* __restrict__ ks,
                                  const float* __restrict__ vs,
                                  AttnLayout layout, float* __restrict__ part_acc,
                                  float* __restrict__ part_ml, int nh, int hd, int rep,
                                  float scale, int split, int draft, int mask_block) {
    const int h = blockIdx.x;             // 查询头
    const int s = blockIdx.y;             // 本 block 在键范围里的切片
    const int flat = blockIdx.z;          // batch * draft，已压平
    const int b = draft > 1 ? flat / draft : flat;
    const int j = draft > 1 ? flat % draft : 0;
    const int g = h / rep;
    const int t = threadIdx.x;
    const int nt = blockDim.x;

    // 草稿查询 ``j`` 必须看到 ``ctx_anchor + 1 + j`` 个键，才和单独解码那个 token 完全一致。
    // 在给了 ``ctx_lens`` 时漏掉 ``+ 1 + j`` 是静默的正确性 bug，不是舍入问题：那样每个草稿
    // 查询共用同一个窗口，而校验调用方传的是锚点的 *下标*，于是第一个查询少看一个键，所有
    // 查询的 softmax 分母同时算错。在一个完美草稿的 oracle 上实测：第 0 行每个 logit 都不
    // 同，而它的 argmax 恰好仍然一致，所以任何基于容差的测试都看不出来。``ctx_anchor``
    // 说明 ``ctx_lens`` 用的是哪种单位。
    const int ctx = layout.ctx_lens
                        ? (int)layout.ctx_lens[b] + (layout.ctx_anchor ? 1 : 0) + j
                        : (int)*layout.pos + 1 + j;
    const int nvalid = ctx > layout.max_seq ? layout.max_seq : ctx;

    extern __shared__ float sh[];
    float* w = sh;                          // [KEY_BLOCK] 分数，随后是权重
    int* rows = (int*)(sh + KEY_BLOCK);     // [KEY_BLOCK] 键 -> 缓存行
    float* qs = sh + 2 * KEY_BLOCK;         // [hd] 本头的查询
    float* red = qs + hd;                   // [nt] 归约暂存

    const uint16_t* qrow = q + ((size_t)flat * nh + h) * hd;
    for (int d = t; d < hd; d += nt) qs[d] = CxScalar<uint16_t>::to_float(qrow[d]);

    const int per = split > 1 ? (nvalid + split - 1) / split : nvalid;
    const int kfrom = s * per;
    const int kto = kfrom + per < nvalid ? kfrom + per : nvalid;

    // 投机掩码是按键块的；在分数循环里对每个键解一次，就不必让它进值那一趟。
    const uint8_t* mask_row = block_mask ? block_mask + (size_t)b * mask_block : nullptr;

    float m = NO_MAX;
    float l = 0.f;
    float acc = 0.f;

    // ``b`` 是这个 block 服务的行（每个单行入口上都是 0），而调用方交给我们的映射决定偏移：
    // 整个 ``[B, max_seq]`` 映射 -> 跳到这一行的板；一行的切片 -> 已经在那儿了。
    const int map_base = layout.map_rows ? b * layout.max_seq : 0;

    for (int base = kfrom; base < kto; base += KEY_BLOCK) {
        const int nb = kto - base < KEY_BLOCK ? kto - base : KEY_BLOCK;
        for (int i = t; i < nb; i += nt) {
            rows[i] = cache_row(base + i, g, layout.pos2slot, layout.stride, map_base);
        }
        __syncthreads();

        float lmax = NO_MAX;
        for (int i = t; i < nb; i += nt) {
            const int8_t* kr = kc + (size_t)rows[i] * hd;
            float dot = 0.f;
            for (int d = 0; d < hd; ++d) dot += qs[d] * (float)kr[d];
            float sc = dot * ks[rows[i]] * scale;
            if (mask_row && !mask_row[(base + i) / mask_block]) sc = NO_MAX;
            w[i] = sc;
            lmax = fmaxf(lmax, sc);
        }
        const float bmax = block_reduce(lmax, red, true);

        // 在线 softmax：先重缩放已累加的部分，再把这一块折进来。整块都被掩掉的块让 ``bmax``
        // 停在哨兵值，于是 ``mnew == m``、``alpha`` 恰好是 1——没有 -inf 算术，也没有 NaN。
        const float mnew = fmaxf(m, bmax);
        const float alpha = __expf(m - mnew);
        float lsum = 0.f;
        for (int i = t; i < nb; i += nt) {
            const float e = __expf(w[i] - mnew);
            w[i] = e;
            lsum += e;
        }
        l = l * alpha + block_reduce(lsum, red, false);
        acc *= alpha;
        m = mnew;

        if (t < hd) {
            float a = 0.f;
            for (int i = 0; i < nb; ++i) {
                a += w[i] * vs[rows[i]] * (float)vc[(size_t)rows[i] * hd + t];
            }
            acc += a;
        }
        __syncthreads();     // rows[] 和 w[] 会被下一块复用
    }

    const size_t at = ((size_t)flat * nh + h) * split + s;
    if (t == 0) {
        part_ml[at * 2 + 0] = m;
        part_ml[at * 2 + 1] = l;
    }
    if (t < hd) part_acc[at * hd + t] = acc;
}

// @brief 合并一个查询各分片的部分和，并施加输出门。
//
// 每个线程都重新算一遍跨 ``split`` 个值的全局最大值，而不是把它暂存到共享内存：``split
// <= 8`` 时一次冗余扫描比一道屏障便宜，而分片数是后端控制的上限。
//
// @param gate [rows * nh, hd] bf16 输出门。
// @param out [rows * nh, hd] bf16，原地写入。
__global__ void attn_combine_kernel(const float* __restrict__ part_acc,
                                    const float* __restrict__ part_ml,
                                    const uint16_t* __restrict__ gate,
                                    uint16_t* __restrict__ out, int nh, int hd,
                                    int split) {
    const int h = blockIdx.x;
    const int flat = blockIdx.y;
    const int t = threadIdx.x;
    const size_t at = ((size_t)flat * nh + h) * split;
    float m = NO_MAX;
    for (int s = 0; s < split; ++s) m = fmaxf(m, part_ml[at * 2 + s * 2]);
    if (t >= hd) return;
    float l = 0.f;
    float a = 0.f;
    for (int s = 0; s < split; ++s) {
        const float alpha = __expf(part_ml[at * 2 + s * 2] - m);
        l += part_ml[at * 2 + s * 2 + 1] * alpha;
        a += part_acc[(at + s) * hd + t] * alpha;
    }
    const size_t io = ((size_t)flat * nh + h) * hd + t;
    const float gv = CxScalar<uint16_t>::to_float(gate[io]);
    out[io] = CxScalar<uint16_t>::from_float(a / l / (1.f + __expf(-gv)));
}

//: 注意力块的线程数。一行的头由 ``hd`` 条通道走完，而归约需要 wavefront 的整数倍，所以块
//: 大小跟随头宽。
int attn_threads(int hd) {
    int t = hd > 128 ? hd : 128;
    t = ((t + 63) / 64) * 64;
    return t > 512 ? 512 : t;
}

//: 一个注意力块需要的共享内存字节数。
size_t attn_smem(int hd, int threads) {
    return (size_t)(2 * KEY_BLOCK + hd + threads) * sizeof(float);
}

// @brief 为一次注意力调用启动切片与合并两趟。
//
// 所有入口都汇到这里，所以稠密和分页路径在线程数、共享内存大小和网格几何上不会各自漂移。
//
// @param q [flat, nh, hd] bf16 查询。
// @param gate [flat, nh, hd] bf16 门。
// @param kc,vc 任一布局的 int8 缓存。
// @param ks,vs fp32 反量化 scale。
// @param out [flat, nh, hd] bf16 输出，原地写入。
// @param part_acc,part_ml 调用方自有的 split-K 暂存。
// @param layout 缓存寻址方式。
// @param mask 可选的 uint8 ``[B, n_blocks]`` 保留掩码。
// @return ``out``，便于调用链串接。
torch::Tensor slice_launch(torch::Tensor q, torch::Tensor gate, torch::Tensor kc,
                           torch::Tensor vc, torch::Tensor ks, torch::Tensor vs,
                           torch::Tensor out, torch::Tensor part_acc,
                           torch::Tensor part_ml, AttnLayout layout,
                           const torch::Tensor* mask, int flat, int nh, int hd, int rep,
                           float scale, int draft, int mask_block) {
    // 切分轴是倒数第二轴而不是第 2 轴：稠密布局是 ``[nh, split, hd]``（split == 1，hd 在
    // 第 2 轴），批处理布局是 ``[B, nh, split, hd]``。按第 2 轴读会让稠密路径以一个 ``hd``
    // 宽的网格启动，写到调用方缓冲区很后面去，结果是把输出清零而不是让形状检查失败。
    const int split = (int)part_acc.size(-2);
    const int threads = attn_threads(hd);
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const uint8_t* mask_ptr = mask == nullptr
        ? nullptr : reinterpret_cast<const uint8_t*>(mask->data_ptr());
    const int blocks = mask != nullptr ? mask_block : 0;
    attn_slice_kernel<<<dim3(nh, split, flat), threads, attn_smem(hd, threads), stream>>>(
        reinterpret_cast<const uint16_t*>(q.data_ptr()), mask_ptr,
        kc.data_ptr<int8_t>(), vc.data_ptr<int8_t>(),
        ks.data_ptr<float>(), vs.data_ptr<float>(), layout,
        part_acc.data_ptr<float>(), part_ml.data_ptr<float>(),
        nh, hd, rep, scale, split, draft, blocks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    attn_combine_kernel<<<dim3(nh, flat), threads, 0, stream>>>(
        part_acc.data_ptr<float>(), part_ml.data_ptr<float>(),
        reinterpret_cast<const uint16_t*>(gate.data_ptr()),
        reinterpret_cast<uint16_t*>(out.data_ptr()), nh, hd, split);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

}  // namespace

// @brief 在稠密 int8 KV 缓存上算一个解码 token。
//
// @param q [nh, hd] bf16 查询，已经旋转并缩放好。
// @param gate [nh, hd] bf16 输出门；sigmoid 在合并那一趟施加。
// @param kcache,vcache [nkv, S, hd] int8，头主序。
// @param kscale,vscale [nkv, S] fp32，每 token 的反量化 scale。
// @param out [nh, hd] bf16，原地写入。
// @param pos 单元素 int64 张量：token 下标兼因果上界。在设备上读取，所以被捕获的重放看到
//        的是当前的位置。
// @param rep 每个 KV 头对应的查询头数。
// @param scale ``hd ** -0.5``。
// @param part_acc [nh, split, hd] fp32 split-K 分子，归调用方所有。
// @param part_ml [nh, split, 2] fp32 split-K 的（最大值，和），归调用方所有。
// @return ``out``，便于调用链串接。
// @throws c10::Error 形状或 dtype 不是内核索引的那种。
torch::Tensor attn_decode(torch::Tensor q, torch::Tensor gate, torch::Tensor kcache,
                          torch::Tensor vcache, torch::Tensor kscale,
                          torch::Tensor vscale, torch::Tensor out, torch::Tensor pos,
                          int64_t rep, double scale, torch::Tensor part_acc,
                          torch::Tensor part_ml) {
    TORCH_CHECK(q.dim() == 2, "query must be [nh, hd]");
    TORCH_CHECK(kcache.scalar_type() == torch::kChar, "cache must be int8");
// 这些指针是在设备上解引用的。dtype 正确的主机张量能通过 ``scalar_type`` 检查，然后照样
// 被解引用，于是内核读到的是那个主机地址上碰巧存在的东西：每次启动都得到一个错但看着合理
// 的窗口，而且不报任何错。在 ``diag/probe_precision.py`` 里实测过一次：一个主机侧的
// ``pos`` 让一个确定性的内核看起来像有 10/20 的竞态。``TORCH_CHECK(..., .is_cuda())`` 把
// 这件事变成一次抛出。
    TORCH_CHECK(pos.scalar_type() == torch::kLong, "pos must be int64");
    TORCH_CHECK(pos.is_cuda() && pos.device() == q.device(),
                "pos must be a device tensor on the query's device: the kernel dereferences it "
                "on the device, so a host tensor is read as garbage instead of being rejected");
    const int nh = (int)q.size(0);
    const int hd = (int)q.size(1);
    const int S = (int)kcache.size(1);
    AttnLayout layout{nullptr, nullptr, pos.data_ptr<int64_t>(), S, S, false, false};
    return slice_launch(q, gate, kcache, vcache, kscale, vscale, out, part_acc, part_ml,
                        layout, nullptr, 1, nh, hd, (int)rep, (float)scale, 1, 0);
}

// @brief 在某一层的分页 int8 池上算一个解码 token。
//
// 位置到槽位这层间接必须留在设备上：解码步是被捕获的图，而主机侧的块表在序列每追加一个块时
// 都得重新烤进图里，设备侧映射则可以在下次重放之前改写，不必重新捕获。
//
// @param q [nh, hd] bf16 查询。
// @param gate [nh, hd] bf16 输出门。
// @param kpool,vpool [nslots, nkv, hd] int8，token 主序。
// @param kscale,vscale [nslots, nkv] fp32，每槽位 scale。
// @param pos2slot [max_pos] int64 设备侧映射；只读前 ``pos + 1`` 项，所以尾部可以装别的
//        序列的槽位。
// @param out [nh, hd] bf16，原地写入。
// @param pos 单元素 int64 张量：token 下标兼因果上界。
// @param max_pos 窗口长度，即槽位表的行间距。
// @param rep 每个 KV 头对应的查询头数。
// @param scale ``hd ** -0.5``。
// @param part_acc [nh, split, hd] fp32 split-K 分子。
// @param part_ml [nh, split, 2] fp32 split-K 的（最大值，和）。
// @return ``out``，便于调用链串接。
// @throws c10::Error 形状或 dtype 不是内核索引的那种。
torch::Tensor attn_paged(torch::Tensor q, torch::Tensor gate, torch::Tensor kpool,
                         torch::Tensor vpool, torch::Tensor kscale,
                         torch::Tensor vscale, torch::Tensor pos2slot, torch::Tensor out,
                         torch::Tensor pos, int64_t max_pos, int64_t rep, double scale,
                         torch::Tensor part_acc, torch::Tensor part_ml) {
    TORCH_CHECK(q.dim() == 2, "query must be [nh, hd]");
    TORCH_CHECK(kpool.scalar_type() == torch::kChar, "pool must be int8");
    TORCH_CHECK(pos2slot.scalar_type() == torch::kLong, "slot map must be int64");
    TORCH_CHECK(pos2slot.is_cuda() && pos2slot.device() == q.device(),
                "slot map must be a device tensor on the query's device");
    TORCH_CHECK(pos.is_cuda() && pos.device() == q.device(),
                "pos must be a device tensor on the query's device: the kernel dereferences it "
                "on the device, so a host tensor is read as garbage instead of being rejected");
    const int nh = (int)q.size(0);
    const int hd = (int)q.size(1);
    const int nkv = (int)kpool.size(1);
    // 这里的 ``pos2slot`` 是 ``pool.pos2slot[row]``，一个元素 0 已经是本行第一个位置的视图，
    // 所以内核不加行偏移。批处理入口传整个映射并设置标志；这个入口搞错，单次解码步就会读到
    // 相邻行的键。
    AttnLayout layout{pos2slot.data_ptr<int64_t>(), nullptr, pos.data_ptr<int64_t>(),
                      nkv, (int)max_pos, false, false};
    return slice_launch(q, gate, kpool, vpool, kscale, vscale, out, part_acc, part_ml,
                        layout, nullptr, 1, nh, hd, (int)rep, (float)scale, 1, 0);
}

// @brief 一个桶的行各带自身上下文长度，一次算一个解码 token。
//
// 每行自己的长度，才让混合 prompt 长度的桶共用一次启动：一行只读自己槽位表的前
// ``context_lens[b]`` 项，所以被掩住的尾巴根本不会被访问，而不是访问了再丢掉。
//
// @param q [B, nh, hd] bf16 查询。
// @param gate [B, nh, hd] bf16 输出门。
// @param kpool,vpool [nslots, nkv, hd] int8，token 主序。
// @param kscale,vscale [nslots, nkv] fp32，每槽位 scale。
// @param slots [B, max_seq] int64，每行的槽位表。
// @param context_lens [B] int32，每行的键数。
// @param out [B, nh, hd] bf16，原地写入。
// @param max_seq 窗口长度，即 ``slots`` 的行间距。
// @param rep 每个 KV 头对应的查询头数。
// @param scale ``hd ** -0.5``。
// @param part_acc [B, nh, split, hd] fp32 split-K 分子。
// @param part_ml [B, nh, split, 2] fp32 split-K 的（最大值，和）。
// @return ``out``，便于调用链串接。
// @throws c10::Error 形状或 dtype 不是内核索引的那种。
torch::Tensor attn_paged_batch(torch::Tensor q, torch::Tensor gate, torch::Tensor kpool,
                               torch::Tensor vpool, torch::Tensor kscale,
                               torch::Tensor vscale, torch::Tensor slots,
                               torch::Tensor context_lens, torch::Tensor out,
                               int64_t max_seq, int64_t rep, double scale,
                               torch::Tensor part_acc, torch::Tensor part_ml) {
    TORCH_CHECK(q.dim() == 3, "queries must be [B, nh, hd]");
    TORCH_CHECK(context_lens.scalar_type() == torch::kInt, "context lengths must be int32");
    TORCH_CHECK(context_lens.is_cuda() && context_lens.device() == q.device(),
                "context lengths must be a device tensor on the query's device");
    TORCH_CHECK(slots.is_cuda() && slots.device() == q.device(),
                "slot map must be a device tensor on the query's device");
    const int rows = (int)q.size(0);
    const int nh = (int)q.size(1);
    const int hd = (int)q.size(2);
    const int nkv = (int)kpool.size(1);
    // 这里的 ``context_lens`` 就是键数，所以校验锚点那次转换不适用。``slots`` 是整个
    // ``[B, max_seq]`` 映射，所以内核必须偏移到第 ``b`` 行。
    AttnLayout layout{slots.data_ptr<int64_t>(), context_lens.data_ptr<int32_t>(), nullptr,
                      nkv, (int)max_seq, false, true};
    return slice_launch(q.reshape({rows, nh, hd}), gate, kpool, vpool, kscale, vscale,
                        out, part_acc, part_ml, layout, nullptr, rows, nh, hd, (int)rep,
                        (float)scale, 1, 0);
}

// @brief 投机校验：每行 ``c`` 个草稿查询共用一遍历史读取。
//
// 第 ``b`` 行、草稿 ``j`` 作为自己的网格切片启动，窗口是 ``ctx_lens[b] + j + 1`` 个键，
// 正好是单独解码那个 token 会看到的窗口。共用历史那一趟是收益所在；每个查询自己的窗口，才
// 让答案与非投机路径完全一致。
//
// @param q [B, c, nh, hd] bf16 草稿查询。
// @param gate [B, c, nh, hd] bf16 输出门。
// @param kpool,vpool [nslots, nkv, hd] int8，token 主序。
// @param kscale,vscale [nslots, nkv] fp32，每槽位 scale。
// @param slots [B, max_seq] int64，每行的槽位表。
// @param ctx_lens [B] int32，每行的历史长度，不含草稿本身。
// @param out [B, c, nh, hd] bf16，原地写入。
// @param max_seq 窗口长度，即 ``slots`` 的行间距。
// @param rep 每个 KV 头对应的查询头数。
// @param scale ``hd ** -0.5``。
// @param part_acc [B*c, nh, split, hd] fp32 split-K 分子。
// @param part_ml [B*c, nh, split, 2] fp32 split-K 的（最大值，和）。
// @param block_mask 可选的 uint8 ``[B, n_blocks]`` 键块保留掩码（1 = 保留）；``None``
//        表示精确全注意力。
// @param mask_block 掩码按多大的块表达。
// @return ``out``，便于调用链串接。
// @throws c10::Error 形状或 dtype 不是内核索引的那种。
torch::Tensor attn_paged_multi_q(torch::Tensor q, torch::Tensor gate, torch::Tensor kpool,
                                 torch::Tensor vpool, torch::Tensor kscale,
                                 torch::Tensor vscale, torch::Tensor slots,
                                 torch::Tensor ctx_lens, torch::Tensor out,
                                 int64_t max_seq, int64_t rep, double scale,
                                 torch::Tensor part_acc, torch::Tensor part_ml,
                                 torch::Tensor block_mask, int64_t mask_block) {
    TORCH_CHECK(q.dim() == 4, "draft queries must be [B, c, nh, hd]");
    TORCH_CHECK(ctx_lens.scalar_type() == torch::kInt, "context lengths must be int32");
    TORCH_CHECK(ctx_lens.is_cuda() && ctx_lens.device() == q.device(),
                "context lengths must be a device tensor on the query's device");
    TORCH_CHECK(slots.is_cuda() && slots.device() == q.device(),
                "slot map must be a device tensor on the query's device");
    const int rows = (int)q.size(0);
    const int draft = (int)q.size(1);
    const int nh = (int)q.size(2);
    const int hd = (int)q.size(3);
    const int nkv = (int)kpool.size(1);
    // 这里的 ``ctx_lens`` 是锚点的最后下标而不是计数：第 ``j`` 个查询需要
    // ``anchor + 1 + j`` 个键，而内核按 ``ctx_anchor`` 把两项都加上。``slots`` 是整个
    // ``[B, max_seq]`` 映射，所以内核必须偏移到第 ``b`` 行。
    AttnLayout layout{slots.data_ptr<int64_t>(), ctx_lens.data_ptr<int32_t>(), nullptr,
                      nkv, (int)max_seq, true, true};
    // 契约里掩码是可选的；「精确注意力」是以空张量传来的，把它当成掩码会把每个键都清零。
    const bool masked = block_mask.defined() && block_mask.numel() > 0;
    if (masked) {
        TORCH_CHECK(block_mask.scalar_type() == torch::kByte, "mask must be uint8");
        TORCH_CHECK((int)block_mask.size(0) == rows, "mask row count must match the batch");
    }
    return slice_launch(q.reshape({rows * draft, nh, hd}), gate, kpool, vpool, kscale,
                        vscale, out, part_acc, part_ml, layout, masked ? &block_mask
                                                                      : nullptr,
                        rows * draft, nh, hd, (int)rep, (float)scale, draft,
                        (int)mask_block);
}

// @brief 把 ``M`` 个 token 的 RoPE 一次启动写进调用方的缓冲区。
//
// 替代 ``ops/rope.py::apply_rope`` 那条 torch 链：每个全注意力层十到十二次启动——两次表
// 转换、四次乘法、两次为 ``rotate_half`` 的 ``cat``、两次加法、两次重新接合的 ``cat``——
// 全都作用在最多 ``[16, 256]`` 的张量上。每个解码步八层，于是约 80-96 次启动、每次只搬几
// KB。这里一次启动。
//
// @param q [M, nh, hd] 查询，bf16。
// @param k [M, nkv, hd] 键，dtype 相同。
// @param cos,sin [M, rotary_dim] fp32，正好是这些 token 对应的行。
// @param q_out [M, nh, hd] 旋转后的查询，原地写入；不得与 ``q`` 重叠。
// @param k_out [M, nkv, hd] 旋转后的键，原地写入；不得与 ``k`` 重叠。
// @param rotary_dim 要旋转的通道数；其余原样拷过去。
// @return ``q_out``，便于调用链串接。
// @throws c10::Error 形状、dtype 或布局是内核索引不了的。重叠不会被检测：模型总是写新
//         缓冲区，而一次静默的原地旋转会让配对的那一半读到已经被旋转过的值。
torch::Tensor rope_apply(torch::Tensor q, torch::Tensor k, torch::Tensor cos,
                         torch::Tensor sin, torch::Tensor q_out, torch::Tensor k_out,
                         int64_t rotary_dim) {
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3, "q and k must be [M, heads, hd]");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && q_out.is_contiguous()
                && k_out.is_contiguous(), "q, k, q_out and k_out must be contiguous");
    TORCH_CHECK(q.sizes() == q_out.sizes() && k.sizes() == k_out.sizes(),
                "outputs must match their inputs' shapes");
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16 && k.scalar_type() == q.scalar_type()
                && q_out.scalar_type() == q.scalar_type()
                && k_out.scalar_type() == q.scalar_type(),
                "q, k and both outputs must be bf16");
    TORCH_CHECK(cos.scalar_type() == torch::kFloat && sin.scalar_type() == torch::kFloat,
                "the cos/sin rows must be fp32; narrowing them at the table would lose "
                "angle precision at long context");
    const int M = (int)q.size(0);
    const int nh = (int)q.size(1);
    const int hd = (int)q.size(2);
    const int nkv = (int)k.size(1);
    const int rot = (int)rotary_dim;
    TORCH_CHECK(k.size(0) == M && k.size(2) == hd, "q and k disagree about M or hd");
    TORCH_CHECK(rot > 0 && rot <= hd && (rot & 1) == 0,
                "rotary_dim must be even and no wider than a head, got ", rot);
    TORCH_CHECK(cos.dim() == 2 && cos.size(0) == M && cos.size(1) == rot
                && sin.sizes() == cos.sizes(),
                "cos/sin must be [M, rotary_dim] fp32");
    if (M == 0) return q_out;
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    // 一个头的每条通道一个线程：256 条线程在一个块里就覆盖这个模型的 head_dim，而网格的
    // y/z 轴分别承担 token 和 q/k 这一对，所以这次启动没有任何东西依赖序列。
    const int threads = hd < 64 ? 64 : (hd > 512 ? 512 : hd);
    const dim3 grid((unsigned)((hd + threads - 1) / threads), (unsigned)M, 2u);
    rope_apply_kernel<uint16_t><<<grid, threads, 0, stream>>>(
        reinterpret_cast<const uint16_t*>(q.data_ptr()),
        reinterpret_cast<const uint16_t*>(k.data_ptr()), cos.data_ptr<float>(),
        sin.data_ptr<float>(), reinterpret_cast<uint16_t*>(q_out.data_ptr()),
        reinterpret_cast<uint16_t*>(k_out.data_ptr()), nh, nkv, hd, rot);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return q_out;
}

// @brief ``[M, N]`` 的对称 int8 量化，每行一个 scale，一次启动。
//
// 替代 ``FullAttn._quant`` 为 int8 KV 缓存跑的那条 torch 链
// ``abs -> amax -> clamp_min -> div -> round -> clamp_ -> to``。那条链六到七次启动；按本卡
// 5.75 us 的启动下限，几微秒的算术要花约 82 us，而每个全注意力层要跑两次，于是解码步的
// 启动里约 100 次是这个。一块一行，一次启动。
//
// 两个目的地都归调用方所有、原地写入。这不只是捕获契约：模型每层保留一个 int8 缓冲区和一个
// scale 缓冲区、每步复用，而那条链每次调用要新分配五个张量。
//
// @param x [M, N] 激活，bf16 或 fp32，最内层步长为 1；行可以带间距。
// @param out_q [M, N] int8，原地写入。
// @param out_s [M] fp32 每行 scale，原地写入。
// @return ``out_q``，便于调用链串接。
// @throws c10::Error dtype 或布局是内核索引不了的，或目的地的形状与 ``x`` 不一致。内核写
//         这两个缓冲区时不做边界检查，所以目的地不匹配会破坏内存而不是失败。
torch::Tensor kv_quant(torch::Tensor x, torch::Tensor out_q, torch::Tensor out_s) {
    TORCH_CHECK(x.dim() == 2, "activation must be 2-D [M, N]");
    // 带间距的情形是更宽融合缓冲区的一列切片，它永远不连续；内核索引靠的是行间距，所以真正
    // 的要求是最内层步长为 1。
    TORCH_CHECK(x.stride(-1) == 1, "activation must have unit innermost stride");
    const int M = (int)x.size(0), N = (int)x.size(1);
    TORCH_CHECK(out_q.scalar_type() == torch::kChar, "q destination must be int8");
    TORCH_CHECK(out_s.scalar_type() == torch::kFloat, "scale destination must be fp32");
    TORCH_CHECK(out_q.dim() == 2 && out_q.is_contiguous()
                && out_q.size(0) == M && out_q.size(1) == N,
                "q destination must be a contiguous [M, N] buffer");
    TORCH_CHECK(out_s.dim() == 1 && out_s.is_contiguous() && out_s.size(0) == M,
                "scale destination must be a contiguous [M] buffer");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 || x.scalar_type() == torch::kFloat,
                "activation must be bf16 or fp32");
    const long pitch = (long)x.stride(0);
    TORCH_CHECK(pitch >= N, "row pitch ", pitch, " is narrower than the row ", N);

    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const int threads = kv_quant_threads(N);
    const size_t shmem = (size_t)threads * sizeof(float);
    if (x.scalar_type() == torch::kBFloat16) {
        kv_quant_kernel<uint16_t><<<M, threads, shmem, stream>>>(
            reinterpret_cast<const uint16_t*>(x.data_ptr()), out_q.data_ptr<int8_t>(),
            out_s.data_ptr<float>(), N, pitch);
    } else {
        kv_quant_kernel<float><<<M, threads, shmem, stream>>>(
            x.data_ptr<float>(), out_q.data_ptr<int8_t>(), out_s.data_ptr<float>(), N,
            pitch);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out_q;
}

// @brief 一次启动算 ``out = silu(gate) * up``，调用方需要时就地写入。
//
// 替代 ``F.silu(gate) * up``，也就是 ``aten::silu`` 加 ``aten::mul``：每个 MLP 两次启动加
// 两个新的 ``[intermediate]`` 临时张量，一个解码步有 32 个 MLP。这里一次启动算完乘积，而且
// 因为 ``out`` 可以与 ``up`` 重叠，调用点能把投影自己的缓冲区交回来、什么都不分配。
//
// 三个操作数都可以是同一个更宽投影输出的两半，所以各自携带自己的行间距，谁也不必连续。这不
// 是通用便利：对 ``[M, N]`` 缓冲区第二轴做 ``.split``——融合 ``gate_up`` 结果拆成 ``gate``
// 和 ``up`` 正是这么做的——产出的是带步长的列视图，强行要求连续会逼调用方把两半拷回稠密
// 缓冲区。为达到与它替代的两个 torch 算子相同的算术而多花两次启动，不叫节省。
//
// @param gate [M, W] 或 [n] 门那一半；最内层步长为 1。
// @param up 另一半，形状和 dtype 同 ``gate``；最内层步长为 1。
// @param out 目的地，形状和 dtype 相同；可以与 ``up`` 重叠。
// @return ``out``，便于调用链串接。
// @throws c10::Error 形状或 dtype 不一致。``out == up`` 之外的重叠这里不检测，因为部分
//         重叠是调用方的 bug、内核正确服务不了，而模型从来不会形成这种情形。
torch::Tensor silu_mul(torch::Tensor gate, torch::Tensor up, torch::Tensor out) {
    TORCH_CHECK(gate.scalar_type() == up.scalar_type() && gate.scalar_type() == out.scalar_type(),
                "gate, up and out must share a dtype");
    TORCH_CHECK(gate.sizes() == up.sizes() && gate.sizes() == out.sizes(),
                "gate, up and out must have the same shape");
    TORCH_CHECK(gate.scalar_type() == torch::kBFloat16 || gate.scalar_type() == torch::kFloat,
                "operands must be bf16 or fp32");
    // 内核用最内层步长 1 加显式行间距来索引。别的都——比如转置过的视图——在这里没有合法的
    // [M, W] 走法。
    TORCH_CHECK(gate.stride(-1) == 1 && up.stride(-1) == 1 && out.stride(-1) == 1,
                "gate, up and out must have unit innermost stride");
    const int width = (int)gate.size(-1);
    if (width == 0) return out;
    // 一维操作数就是一行：解码路径切分的是一个 [N] 向量，那里间距和宽度重合、行数是 1。
    const long rows = gate.dim() <= 1 ? 1 : (long)gate.numel() / width;
    const long g_pitch = gate.dim() <= 1 ? (long)width : (long)gate.stride(-2);
    const long u_pitch = up.dim() <= 1 ? (long)width : (long)up.stride(-2);
    const long o_pitch = out.dim() <= 1 ? (long)width : (long)out.stride(-2);
    TORCH_CHECK(g_pitch >= width && u_pitch >= width && o_pitch >= width,
                "row pitch narrower than the row width");
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    // 每列一条通道，每个操作数行一个网格行。宽度是投影的中间尺寸（这个模型里是 18432），所以
    // 网格在 x 方向很宽、在高度上只跟行数一样，解码是 1 行、预填充是几千行。没有归约、没有
    // 共享内存、没有屏障——这就是这个内核能这么简单的原因。
    const int threads = 256;
    const dim3 grid((unsigned)((width + threads - 1) / threads), (unsigned)rows);
    if (gate.scalar_type() == torch::kBFloat16) {
        silu_mul_kernel<uint16_t><<<grid, threads, 0, stream>>>(
            reinterpret_cast<const uint16_t*>(gate.data_ptr()),
            reinterpret_cast<const uint16_t*>(up.data_ptr()),
            reinterpret_cast<uint16_t*>(out.data_ptr()), width, g_pitch, u_pitch, o_pitch);
    } else {
        silu_mul_kernel<float><<<grid, threads, 0, stream>>>(
            gate.data_ptr<float>(), up.data_ptr<float>(), out.data_ptr<float>(), width,
            g_pitch, u_pitch, o_pitch);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
