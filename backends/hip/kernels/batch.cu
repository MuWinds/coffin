// @file batch.cu
// @brief 批处理解码内核：B 个序列一次启动，而不是 B 次。
//
// Module: coffin/kernels。解码路径的批处理那一半；``hip_ops.py`` 把它和 ``attn.cu`` /
// ``linear_attn.cu`` / ``delta.cu`` 加载进同一个 torch 扩展，那三个文件保留 B = 1 和投机
// 校验入口。
//
// 这个文件为什么存在：单流解码步每层每序列要跑四个内核（分页注意力、conv1d、delta
// prologue、delta 更新）。B 个序列的静态批把*工作量*放大 B 倍，但这一步仍然应该只花大约
// 每个内核一次启动，因为在解码尺寸下该步是延迟/启动受限的：逐行内核每层只启动 2-4 个
// block，grid 从来没接近填满这张卡。
//
// Constraints:
//   * 宽度为 1 的一行必须与单行内核逐位一致——同样的 key 分块、同样的 online-softmax
//     部分和、同样的 fp32 状态更新顺序。两条路径会做 A/B 对比，分叉就是其中一侧的 bug。
//   * 行与行绝不共享状态：一个 block（或 warp）独占一行，且没有跨 block 同步。
//   * 作为融合投影输出列切片的 q/k/v 从张量取步长；在那里假定连续会读到错误的三分之一。
//   * 逐行的 key 循环以 ``context_lens[b]`` 为界，即该行自己的窗口，绝不以 ``max_seq``
//     为界：短序列组成的批不该为最长的那条买单，这正是混合位置的批处理存在的意义。
//   * LDS 预算是每 CU 64 KB，所以注意力的行窗口按 ATTN_KEY_BLOCK 个 key 流式处理，绝不
//     整块暂存。
//
// 行主序的批布局，除调用方持有的输出外全部只读：
//
//   attention      q/gate [B, nh, hd] bf16，slots [B, smax] int64，
//                  context_lens [B] int32，out [B, nh, hd] bf16，
//                  part_acc [B, nh, SPLIT, hd]，part_ml [B, nh, SPLIT, 2] fp32
//   conv1d         x [B, conv_dim] bf16，state [B, conv_dim, 3] fp32，
//                  out [B, conv_dim] fp32
//   delta prologue q/k [B, nk, dk]，v [B, nv, dv]，b/a [B, nv] bf16，
//                  q_pro/k_pro [B, nv, dk]，v_pro [B, nv, dv]，
//                  beta/g [B, nv] fp32
//   delta update   state [B, nv, dk, dv]，out [B, nv, dv] fp32

#include "hip_ops.h"

#include <c10/cuda/CUDAStream.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace {

//: 每个流式块处理多少个 key，与 ``attn.cu`` 的上限相同：1024 个 key 约占 10 KB LDS，
//: 块大小决定了 online-softmax 的缩放点在哪里。
constexpr int ATTN_KEY_BLOCK = 1024;

//: 每个查询头的 key 范围切分数，与单行注意力相同。单行内核用 8 是因为按头一个 block
//: 只有约 16 个 block；有 B 行时 grid 已经是 (nh, split, B)，所以这个值只决定调用方要
//: 分配多大的部分和缓冲区。
constexpr int ATTN_SPLIT_MAX = 64;

//: delta 更新里每个状态列的 dk 切片数（与 ``delta.cu`` 的 ``DELTA_SPLIT`` 一致；不一致会
//: 悄悄改变 fp32 的求和顺序）。
constexpr int DELTA_SPLIT = 4;

__device__ __forceinline__ float bf16f(unsigned short h) {
    return __uint_as_float(((unsigned)h) << 16);
}

__device__ __forceinline__ unsigned short fbf16(float f) {
    unsigned u = __float_as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);   // 就近舍入到偶数（round to nearest even）
    return (unsigned short)(u >> 16);
}

// 通过 ``blockDim.x`` 个 float 的共享暂存做 block 内的最大值/求和。``red`` 必须能放
// blockDim.x 项且会被覆盖；block 里每个线程都必须调用它，因为函数结尾停在
// ``__syncthreads()``。
template <bool IsMax>
__device__ __forceinline__ float block_reduce(float v, float* red) {
    const int t = threadIdx.x;
    red[t] = v;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (t < s) red[t] = IsMax ? fmaxf(red[t], red[t + s]) : red[t] + red[t + s];
        __syncthreads();
    }
    return red[0];
}

// @brief 批处理分页解码注意力；每个（头，切片，行）一个 block。
//
// 与单行内核的同构性：block (h, s, b) 独占头 ``h`` 的 key 范围里同一个切片
// ``[s*per, min((s+1)*per, nvalid))``，只是现在 ``nvalid = context_lens[b]``；行 ``b``
// 的 key 行经它自己的槽位表（``slots + b*smax``）间接寻址。部分和 (max, sum, acc) 按
// (b, h, s) 发布，再由合并内核汇总并施加输出门控。所以每个 (b, h) 的算术与单行完全一样，
// 只是 grid 从 (nh, split) 变成 (nh, split, B)。
//
// @param q            查询向量，[B, nh, hd] bf16，行距 ``qb``、头距 ``qh``（融合投影输出的
//                     列切片并不连续）。
// @param slots        [B, smax] int64 槽位表；``smax`` 是它的行距。
// @param context_lens [B] int32；第 b 行的 key 循环停在它自己的取值上。
// @param part_acc     [B, nh, split, hd] fp32 的 online-softmax 分子。
// @param part_ml      [B, nh, split, 2] fp32（滑动最大值、滑动和）。
// @param split        每头的 key 范围切片数；部分和缓冲区由调用方按它分配。
__global__ void attn_paged_batch_kernel(
        const unsigned short* __restrict__ q, int qb, int qh,
        const int8_t* __restrict__ kpool, const int8_t* __restrict__ vpool,
        const float* __restrict__ kscale, const float* __restrict__ vscale,
        const int64_t* __restrict__ slots, int smax,
        const int* __restrict__ context_lens,
        float* __restrict__ part_acc, float* __restrict__ part_ml,
        int nh, int nkv, int hd, int rep, float scale, int split) {
    const int b = blockIdx.z;
    const int h = blockIdx.x;
    const int s = blockIdx.y;
    const int g = h / rep;
    const int t = threadIdx.x;
    const int nt = blockDim.x;
    const int nvalid = context_lens[b];              // 该行自己的窗口，而不是最大值
    const int64_t* row_slots = slots + (size_t)b * smax;

    extern __shared__ float sh[];
    float* w = sh;                                        // [ATTN_KEY_BLOCK] 权重
    int* rows = (int*)(sh + ATTN_KEY_BLOCK);              // [ATTN_KEY_BLOCK] key -> 行号
    float* qs = sh + ATTN_KEY_BLOCK + ATTN_KEY_BLOCK;     // [hd] 本头的查询
    float* red = qs + hd;                                 // [nt]

    const unsigned short* qp = q + (size_t)b * qb + (size_t)h * qh;
    for (int d = t; d < hd; d += nt) qs[d] = bf16f(qp[d]);

    float m = -1e30f, l = 0.f, acc = 0.f;    // 滑动最大值、滑动和、本维累加
    if (nvalid > 0) {
        const int per = (nvalid + split - 1) / split;
        const int kfrom = s * per;
        const int kto = min(kfrom + per, nvalid);
        for (int base = kfrom; base < kto; base += ATTN_KEY_BLOCK) {
            const int nb = min(ATTN_KEY_BLOCK, kto - base);
            for (int i = t; i < nb; i += nt)
                rows[i] = (int)(row_slots[base + i] * nkv + g);
            __syncthreads();

            // 分数：dot(q, 反量化的 k) * 逐 token scale * softmax scale
            float lmax = -1e30f;
            for (int i = t; i < nb; i += nt) {
                const int8_t* kr = kpool + (size_t)rows[i] * hd;
                float dot = 0.f;
                for (int d = 0; d < hd; ++d) dot += qs[d] * (float)kr[d];
                const float sc = dot * kscale[rows[i]] * scale;
                w[i] = sc;
                lmax = fmaxf(lmax, sc);
            }
            const float bmax = block_reduce<true>(lmax, red);

            const float mnew = fmaxf(m, bmax);
            const float alpha = __expf(m - mnew);
            float lsum = 0.f;
            for (int i = t; i < nb; i += nt) {
                const float e = __expf(w[i] - mnew);
                w[i] = e;
                lsum += e;
            }
            l = l * alpha + block_reduce<false>(lsum, red);
            acc *= alpha;
            m = mnew;

            if (t < hd) {
                float a = 0.f;
                for (int i = 0; i < nb; ++i)
                    a += w[i] * vscale[rows[i]] * (float)vpool[(size_t)rows[i] * hd + t];
                acc += a;
            }
            __syncthreads();     // rows[] 和 w[] 要给下一个块复用
        }
    }

    if (t == 0) {
        const size_t ml = ((size_t)b * nh + h) * split + s;
        part_ml[ml * 2 + 0] = m;
        part_ml[ml * 2 + 1] = l;
    }
    if (t < hd) {
        const size_t pa = ((size_t)b * nh + h) * split + s;
        part_acc[pa * hd + t] = acc;
    }
}

// @brief 合并一个（b，头）各切片的部分和，并施加门控。
//
// 与单行合并内核相同的合并方式，只是按批索引。每个线程各自重算全局最大值（``split`` 个
// 值），而不是把它暂存到共享内存：做一次 warp 归约的开销比重复读这几个值还大。
//
// @param gate 门控向量，[B, nh, hd] bf16，行距 ``gb``、头距 ``gh``；sigmoid 折进归一化后的
//             输出里。
// @param out  [B, nh, hd] bf16，行距 ``ob``、头距 ``oh``。
__global__ void attn_paged_batch_combine_kernel(
        const float* __restrict__ part_acc, const float* __restrict__ part_ml,
        const unsigned short* __restrict__ gate, int gb, int gh,
        unsigned short* __restrict__ out, int ob, int oh,
        int hd, int split) {
    const int h = blockIdx.x;
    const int b = blockIdx.y;
    const int t = threadIdx.x;
    const size_t base = (size_t)b * gridDim.x + h;
    const float* ml = part_ml + base * split * 2;
    float m = -1e30f;
    for (int s = 0; s < split; ++s) m = fmaxf(m, ml[s * 2]);
    if (t >= hd) return;
    float l = 0.f, a = 0.f;
    for (int s = 0; s < split; ++s) {
        const float alpha = __expf(ml[s * 2] - m);
        l += ml[s * 2 + 1] * alpha;
        a += part_acc[base * split * hd + (size_t)s * hd + t] * alpha;
    }
    const float gv = bf16f(gate[(size_t)b * gb + (size_t)h * gh + t]);
    out[(size_t)b * ob + (size_t)h * oh + t] = fbf16(a / l / (1.f + __expf(-gv)));
}

// @brief 批处理的深度因果 conv1d（卷积核 4）+ SiLU。
//
// 与 ``conv1d_silu_kernel`` 相同的逐通道递推；grid 的 y 轴是行，所以每个 (b, c) 都有自己
// 的状态环，没有跨行流量。状态平移在单个线程内先读后写，因此不需要任何屏障。
//
// @param state [B, C, 3] fp32，前三个输入的环形缓冲，行距 ``lds``，原地平移。
// @param ldx   ``x`` 的行距；卷积输出是融合投影的列切片，所以行不是紧密排布的。
// @param ldo   ``out`` 的行距。
__global__ void conv1d_silu_batch_kernel(const float* __restrict__ w,
                                         const unsigned short* __restrict__ x,
                                         float* __restrict__ state,
                                         float* __restrict__ out, int C,
                                         int ldx, int ldo, int lds) {
    const int b = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float* s = state + (size_t)b * lds + (size_t)c * 3;
    const float* wc = w + (size_t)c * 4;
    const float x3 = bf16f(x[(size_t)b * ldx + c]);
    const float acc = fmaf(s[0], wc[0], fmaf(s[1], wc[1], fmaf(s[2], wc[2], x3 * wc[3])));
    out[(size_t)b * ldo + c] = acc / (1.f + __expf(-acc));     // silu
    s[0] = s[1];
    s[1] = s[2];
    s[2] = x3;
}

// 一次做两个值的 block 内求和——q 和 k 总是一起归一化，所以共用一趟归约。
__device__ __forceinline__ float2 block_sum2(float a, float b, float2* red) {
    const int t = threadIdx.x;
    red[t] = make_float2(a, b);
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (t < s) {
            red[t].x += red[t + s].x;
            red[t].y += red[t + s].y;
        }
        __syncthreads();
    }
    return red[0];
}

// @brief 批处理的 gated delta prologue：逐 value 头的 q/k L2 归一化、q 缩放、头扩展和
//        beta/g。
//
// block (vh, b) 处理第 b 行的 value 头 vh，正如单行内核处理唯一那一行的某个 value 头；
// ``rep`` 把 value 头 vh 映射到 key 头 ``vh / rep``（repeat_interleave 的顺序）。
//
// @param q_raw,k_raw fp32 [B, nk, dk]，行距分别是 ``ldq`` / ``ldk``；它们是卷积输出的列
//                    切片，所以某个 value 头对应的 key 头可能被重复读 ``rep`` 次。
// @param qscale     折进归一化后 q 的缩放（softmax scale * 1/sqrt）。
// @param rep        每个 key 头对应的 value 头数。
__global__ void delta_prologue_batch_kernel(
        const float* __restrict__ q_raw, int ldq,
        const float* __restrict__ k_raw, int ldk,
        const float* __restrict__ v_raw, int ldv,
        const unsigned short* __restrict__ b_raw,
        const unsigned short* __restrict__ a_raw,
        const float* __restrict__ neg_expA,
        const float* __restrict__ dt_bias,
        float* __restrict__ q_out, float* __restrict__ k_out,
        float* __restrict__ v_out,
        float* __restrict__ beta, float* __restrict__ g,
        float qscale, int dk, int dv, int rep) {
    const int vh = blockIdx.x;
    const int bb = blockIdx.y;
    const int nv = gridDim.x;
    const int t = threadIdx.x;
    const int kh = vh / rep;
    const float* qr = q_raw + (size_t)bb * ldq + (size_t)kh * dk;
    const float* kr = k_raw + (size_t)bb * ldk + (size_t)kh * dk;

    const float qv = t < dk ? qr[t] : 0.f;
    const float kv = t < dk ? kr[t] : 0.f;
    __shared__ float2 red[256];
    const float2 ss = block_sum2(qv * qv, kv * kv, red);

    const size_t head = (size_t)bb * nv + vh;
    if (t < dk) {
        q_out[head * dk + t] = qv * rsqrtf(ss.x + 1e-6f) * qscale;
        k_out[head * dk + t] = kv * rsqrtf(ss.y + 1e-6f);
    }
    if (t < dv) v_out[head * dv + t] = v_raw[(size_t)bb * ldv + (size_t)vh * dv + t];
    if (t == 0) {
        const size_t idx = (size_t)bb * nv + vh;
        beta[idx] = 1.f / (1.f + __expf(-bf16f(b_raw[idx])));
        const float av = bf16f(a_raw[idx]) + dt_bias[vh];
        const float sp = av > 20.f ? av : __logf(1.f + __expf(av));   // softplus，不会溢出
        g[idx] = neg_expA[vh] * sp;
    }
}

// @brief 批处理的 gated delta 状态更新。
//
// 复制 ``delta_kernel`` 的切片逻辑，状态按行偏移：每个（头，列块，行）一个 block。一个线程
// 独占第 b 行的第 ``j`` 列以及 dk 循环的四分之一；行与行之间不共享任何东西，所以唯一的改动
// 就是状态的基址指针。``decay = exp(g)`` 与 ``rate = beta`` 按（行，头）取值，与单行内核
// 一样暂存在共享内存里。
//
// @param st  [B, nv, dk, dv] fp32 循环状态，原地更新。
// @param out [B, nv, dv] fp32；由拥有 dk 切片 0 的 lane 写出。
__global__ void delta_batch_kernel(const float* __restrict__ q,
                                   const float* __restrict__ k,
                                   const float* __restrict__ v,
                                   const float* __restrict__ beta,
                                   const float* __restrict__ g,
                                   float* __restrict__ st,
                                   float* __restrict__ out, int dk, int dv) {
    const int head = blockIdx.x;
    const int bb = blockIdx.z;
    const int heads = gridDim.x;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int ii = lane & (DELTA_SPLIT - 1);              // dk 切片
    const int jj = lane >> 2;                             // warp 内的列
    const int cols_per_block = (blockDim.x >> 5) * (32 / DELTA_SPLIT);
    const int j = blockIdx.y * cols_per_block + warp * (32 / DELTA_SPLIT) + jj;
    const size_t hb = (size_t)bb * heads + head;

    extern __shared__ float shared[];
    float* ks = shared;
    float* qs = shared + dk;
    for (int i = threadIdx.x; i < dk; i += blockDim.x) {
        ks[i] = k[hb * dk + i];
        qs[i] = q[hb * dk + i];
    }
    __shared__ float decay, rate;
    if (threadIdx.x == 0) {
        decay = __expf(g[hb]);
        rate = beta[hb];
    }
    __syncthreads();

    if (j >= dv) return;                                  // 整个 4-lane 组一起退出
    float* column = st + hb * (size_t)dk * dv + j;
    const float value = v[hb * dv + j];

    const int slice = dk / DELTA_SPLIT;
    const int i0 = ii * slice, i1 = i0 + slice;
    float kv_mem = 0.f;
    for (int i = i0; i < i1; ++i) {
        kv_mem = fmaf(column[(size_t)i * dv] * decay, ks[i], kv_mem);
    }
#pragma unroll
    for (int o = 1; o < DELTA_SPLIT; o <<= 1) kv_mem += __shfl_xor(kv_mem, o);

    const float delta = (value - kv_mem) * rate;
    float result = 0.f;
    for (int i = i0; i < i1; ++i) {
        const float s = fmaf(ks[i], delta, column[(size_t)i * dv] * decay);
        column[(size_t)i * dv] = s;
        result = fmaf(s, qs[i], result);
    }
#pragma unroll
    for (int o = 1; o < DELTA_SPLIT; o <<= 1) result += __shfl_xor(result, o);
    if (ii == 0) out[hb * dv + j] = result;
}

}  // namespace

// @brief 启动批处理分页注意力（分片那一遍 + 合并那一遍）。
//
// @param q,gate     [B, nh, hd] bf16；内层步长必须为 1，但行距/头距会照实际取值处理（融合
//                   投影的列切片）。
// @param slots      [B, smax] int64 连续存放；smax 来自它的行距。
// @param context_lens [B] int32 连续存放，每行一个窗口。
// @param max_seq    未使用：窗口是流式处理的，设备上不再需要它的长度。保留它是为了不改
//                   调用方签名。
// @param part_acc,part_ml 调用方持有的 fp32 部分和 [B, nh, split, hd] 与
//                   [B, nh, split, 2]；``split`` 从 ``part_acc`` 读出，所以图捕获期间形状
//                   保持静态。
// @return ``out``，原地写出并返回，便于链式调用。
// @throws c10::Error（TORCH_CHECK）：形状、dtype 或步长破坏了内核的索引契约时抛出；消息会
//         点名出问题的张量，所以应该改调用方而不是改内核。
torch::Tensor attn_paged_batch(torch::Tensor q, torch::Tensor gate,
                               torch::Tensor kpool, torch::Tensor vpool,
                               torch::Tensor kscale, torch::Tensor vscale,
                               torch::Tensor slots, torch::Tensor context_lens,
                               torch::Tensor out, int64_t max_seq,
                               int64_t rep, double scale,
                               torch::Tensor part_acc, torch::Tensor part_ml) {
    (void)max_seq;      // 流式窗口：逐行的长度存在 context_lens 里
    TORCH_CHECK(q.dim() == 3 && gate.dim() == 3 && out.dim() == 3,
                "q/gate/out must be [B, nh, hd]");
    TORCH_CHECK(q.stride(2) == 1 && gate.stride(2) == 1 && out.stride(2) == 1,
                "unit inner stride required");
    const int B = q.size(0), nh = q.size(1), hd = q.size(2);
    const int nkv = kpool.size(1);
    TORCH_CHECK(gate.size(0) == B && gate.size(1) == nh && gate.size(2) == hd,
                "gate shape mismatch");
    TORCH_CHECK(out.size(0) == B && out.size(1) == nh && out.size(2) == hd,
                "out shape mismatch");
    TORCH_CHECK(slots.dim() == 2 && slots.size(0) == B && slots.stride(1) == 1,
                "slots must be [B, max_seq] int64");
    TORCH_CHECK(context_lens.dim() == 1 && context_lens.size(0) == B,
                "context_lens must be [B]");
    TORCH_CHECK(context_lens.scalar_type() == at::kInt, "context_lens must be int32");
    TORCH_CHECK(context_lens.is_contiguous() && slots.is_contiguous(),
                "context_lens/slots must be contiguous");
    TORCH_CHECK(part_acc.dim() == 4 && part_acc.size(0) == B && part_acc.size(1) == nh
                && part_acc.size(3) == hd && part_acc.is_contiguous()
                && part_ml.is_contiguous() && part_ml.dim() == 4
                && part_ml.size(0) == B && part_ml.size(1) == nh
                && part_ml.size(3) == 2 && part_acc.size(2) == part_ml.size(2),
                "part_acc/part_ml shape mismatch");
    const int split = part_acc.size(2);
    TORCH_CHECK(split >= 1 && split <= ATTN_SPLIT_MAX, "unsupported split");
    const int smax = (int)slots.stride(0);
    const int threads = 256;
    const int cthreads = hd > threads ? ((hd + 31) / 32) * 32 : threads;
    const size_t smem = (size_t)(2 * ATTN_KEY_BLOCK + hd + threads) * sizeof(float);
    hipStream_t stream = at::cuda::getCurrentCUDAStream();
    attn_paged_batch_kernel<<<dim3(nh, split, B), threads, smem, stream>>>(
        reinterpret_cast<const unsigned short*>(q.data_ptr<at::BFloat16>()),
        (int)q.stride(0), (int)q.stride(1),
        kpool.data_ptr<int8_t>(), vpool.data_ptr<int8_t>(),
        kscale.data_ptr<float>(), vscale.data_ptr<float>(),
        slots.data_ptr<int64_t>(), smax,
        context_lens.data_ptr<int32_t>(),
        part_acc.data_ptr<float>(), part_ml.data_ptr<float>(),
        nh, nkv, hd, (int)rep, (float)scale, split);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    attn_paged_batch_combine_kernel<<<dim3(nh, B), cthreads, 0, stream>>>(
        part_acc.data_ptr<float>(), part_ml.data_ptr<float>(),
        reinterpret_cast<const unsigned short*>(gate.data_ptr<at::BFloat16>()),
        (int)gate.stride(0), (int)gate.stride(1),
        reinterpret_cast<unsigned short*>(out.data_ptr<at::BFloat16>()),
        (int)out.stride(0), (int)out.stride(1), hd, split);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// @brief 启动批处理的深度因果 conv1d + SiLU。
//
// @param conv_w     [C, 4] fp32 抽头（三个历史抽头加当前输入）。
// @param x          [B, C] bf16，内层步长为 1。
// @param conv_state [B, C, 3] fp32 环形缓冲，原地平移。
// @return ``out``，[B, C] fp32，原地写出。
// @throws c10::Error（TORCH_CHECK）：形状或内层步长破坏了索引契约；去改调用方。
torch::Tensor conv1d_silu_batch(torch::Tensor conv_w, torch::Tensor x,
                                torch::Tensor conv_state, torch::Tensor out) {
    TORCH_CHECK(x.dim() == 2 && x.stride(1) == 1, "x must be [B, C] with unit inner stride");
    const int B = x.size(0), C = x.size(1);
    TORCH_CHECK(out.size(0) == B && out.size(1) == C && out.stride(1) == 1,
                "out must be [B, C] fp32");
    TORCH_CHECK(conv_state.size(0) == B && conv_state.size(1) == C
                && conv_state.size(2) == 3 && conv_state.stride(2) == 1,
                "conv_state must be [B, C, 3]");
    hipStream_t stream = at::cuda::getCurrentCUDAStream();
    conv1d_silu_batch_kernel<<<dim3((C + 255) / 256, B), 256, 0, stream>>>(
        conv_w.data_ptr<float>(),
        reinterpret_cast<const unsigned short*>(x.data_ptr<at::BFloat16>()),
        conv_state.data_ptr<float>(), out.data_ptr<float>(), C,
        (int)x.stride(0), (int)out.stride(0), (int)conv_state.stride(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// @brief 启动批处理的 gated delta prologue。
//
// @param q_raw,k_raw [B, nk, dk] fp32，行步长照实际取值：它们是融合卷积输出的列切片，所以
//                   行不是紧密排布的。
// @param v_raw      [B, nv, dv] fp32，行步长照实际取值。
// @param b,a        [B, nv] bf16 的原始投影输出；``beta = sigmoid(b)``、
//                   ``g = -exp(A) * softplus(a + dt_bias)``。
// @param q_out,k_out [B, nv, dk] fp32 连续存放，已做 L2 归一化，q 已缩放。
// @param beta,g     [B, nv] fp32 连续存放，供 ``delta_batch`` 使用。
// @throws c10::Error（TORCH_CHECK）：形状、dtype 或步长破坏了索引契约；去改调用方。
void delta_prologue_batch(torch::Tensor q_raw, torch::Tensor k_raw, torch::Tensor v_raw,
                          torch::Tensor b, torch::Tensor a,
                          torch::Tensor neg_expA, torch::Tensor dt_bias,
                          torch::Tensor q_out, torch::Tensor k_out, torch::Tensor v_out,
                          torch::Tensor beta, torch::Tensor g,
                          double qscale, int64_t rep) {
    TORCH_CHECK(q_raw.dim() == 3 && k_raw.dim() == 3 && v_raw.dim() == 3,
                "q/k/v must be [B, heads, d]");
    TORCH_CHECK(q_raw.stride(2) == 1 && k_raw.stride(2) == 1 && v_raw.stride(2) == 1,
                "unit inner stride required");
    const int B = q_raw.size(0);
    const int nk = q_raw.size(1), dk = q_raw.size(2);
    const int nv = v_raw.size(1), dv = v_raw.size(2);
    TORCH_CHECK(q_raw.stride(1) == dk && k_raw.stride(1) == dk && v_raw.stride(1) == dv,
                "q/k/v heads must be packed");
    TORCH_CHECK(k_raw.size(0) == B && k_raw.size(1) == nk && k_raw.size(2) == dk,
                "k shape mismatch");
    TORCH_CHECK(v_raw.size(0) == B, "v batch mismatch");
    TORCH_CHECK(b.size(0) == B && b.size(1) == nv && a.size(0) == B && a.size(1) == nv,
                "b/a must be [B, nv]");
    TORCH_CHECK(q_out.is_contiguous() && k_out.is_contiguous() && v_out.is_contiguous()
                && beta.is_contiguous() && g.is_contiguous(),
                "outputs must be contiguous");
    hipStream_t stream = at::cuda::getCurrentCUDAStream();
    delta_prologue_batch_kernel<<<dim3(nv, B), 128, 0, stream>>>(
        q_raw.data_ptr<float>(), (int)q_raw.stride(0),
        k_raw.data_ptr<float>(), (int)k_raw.stride(0),
        v_raw.data_ptr<float>(), (int)v_raw.stride(0),
        reinterpret_cast<const unsigned short*>(b.data_ptr<at::BFloat16>()),
        reinterpret_cast<const unsigned short*>(a.data_ptr<at::BFloat16>()),
        neg_expA.data_ptr<float>(), dt_bias.data_ptr<float>(),
        q_out.data_ptr<float>(), k_out.data_ptr<float>(), v_out.data_ptr<float>(),
        beta.data_ptr<float>(), g.data_ptr<float>(),
        (float)qscale, dk, dv, (int)rep);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// @brief 启动批处理的 gated delta 状态更新。
//
// @param st  [B, heads, dk, dv] fp32 循环状态，原地更新。
// @param q,k [B, heads, dk] fp32 连续存放，已由 prologue 归一化。
// @param beta,g [B, heads] fp32 逐行的衰减与写入速率。
// @return ``out``，[B, heads, dv] fp32，原地写出。
// @throws c10::Error（TORCH_CHECK）：某个输入不连续，或 ``dk`` 不是 DELTA_SPLIT 的整数倍
//         （四 lane 的 shuffle 需要它）。
torch::Tensor delta_batch(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                          torch::Tensor beta, torch::Tensor g,
                          torch::Tensor st, torch::Tensor out) {
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous()
                && beta.is_contiguous() && g.is_contiguous() && st.is_contiguous()
                && out.is_contiguous(),
                "contiguous inputs required");
    const int B = st.size(0), heads = st.size(1), dk = st.size(2), dv = st.size(3);
    TORCH_CHECK(q.size(0) == B && q.size(1) == heads && q.size(2) == dk,
                "q shape does not match the state");
    TORCH_CHECK(out.size(0) == B && out.size(1) == heads && out.size(2) == dv,
                "out shape does not match the state");
    TORCH_CHECK(dk % DELTA_SPLIT == 0, "dk must be a multiple of DELTA_SPLIT");
    const int threads = 128, cols_per_block = (threads / 32) * (32 / DELTA_SPLIT);
    const int blocks_y = (dv + cols_per_block - 1) / cols_per_block;
    hipStream_t stream = at::cuda::getCurrentCUDAStream();
    delta_batch_kernel<<<dim3(heads, blocks_y, B), threads, 2 * dk * sizeof(float),
                         stream>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
        beta.data_ptr<float>(), g.data_ptr<float>(),
        st.data_ptr<float>(), out.data_ptr<float>(), dk, dv);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// @brief 多查询校验注意力：每行的 c 个草稿查询共享同一段 KV 历史。
//
// 对每个（行，查询）对跑与 ``attn_paged_batch`` *完全相同*的算术——相同的 split-K 切片、
// 相同的 key 块 online softmax、相同的合并——所以每个查询的输出与单独为那个 token 调用
// 解码内核逐位一致。真正新的只有 grid：一行的 c 个查询是相邻的 ``blockIdx.z`` 取值，于是
// 遍历同一段 (b, 头, 切片) key 范围的那些 block 会一起启动，c 趟历史遍历命中 L2 而不是从
// HBM 重读。长上下文下这把 c 趟 HBM 访存变成一趟，这就是校验的收益。
//
// @param block_mask 可选的 uint8 [B, nblocks] 保留掩码，1 表示保留；未定义或空张量表示精确
//                   的全注意力。被掩掉的 key 分数为 -inf（权重恰好为 0），正是稀疏解码
//                   调用方想要的语义——而且因为权重为 0，V 那一趟可以跳过它们的加载。
// @param mask_block 每个掩码块包含多少个 token。64 走移位，其他值走除法，所以两者都在 key
//                   循环里做了特化。
// @param ctx_lens   [B] int32 历史长度；查询 j 看到 ctx_lens[b] + j + 1 个 key，也就是逐
//                   token 内核会拿到的那个 nvalid。
__global__ void attn_paged_multi_q_kernel(
        const unsigned short* __restrict__ q, int qb, int qh,
        const int8_t* __restrict__ kpool, const int8_t* __restrict__ vpool,
        const float* __restrict__ kscale, const float* __restrict__ vscale,
        const int64_t* __restrict__ slots, int smax,
        const int* __restrict__ ctx_lens,
        const unsigned char* __restrict__ block_mask, int nblocks, int mask_block,
        float* __restrict__ part_acc, float* __restrict__ part_ml,
        int c, int nh, int nkv, int hd, int rep, float scale, int split) {
    // block 顺序对 L2 复用很关键：x 是查询下标，z 打包了（行，切片），所以遍历同一段历史
    // 切片的 c 个 block 在启动顺序上相邻，会命中 L2。
    const int j = blockIdx.x;
    const int h = blockIdx.y;
    const int bs = blockIdx.z;
    const int b = bs / split;
    const int s = bs - b * split;
    const int z = b * c + j;
    const int g = h / rep;
    const int t = threadIdx.x;
    const int nt = blockDim.x;
    // 查询 j 位于位置 ctx+j，所以它看到历史加上前 j+1 个新 token——正是逐 token 内核为那个
    // token 会拿到的 nvalid。
    const int nvalid = ctx_lens[b] + j + 1;
    const int64_t* row_slots = slots + (size_t)b * smax;
    const unsigned char* row_mask = block_mask ? block_mask + (size_t)b * nblocks
                                               : nullptr;

    extern __shared__ float sh[];
    float* w = sh;                                        // [ATTN_KEY_BLOCK] 权重
    int* rows = (int*)(sh + ATTN_KEY_BLOCK);              // [ATTN_KEY_BLOCK] key -> 行号
    float* qs = sh + ATTN_KEY_BLOCK + ATTN_KEY_BLOCK;     // [hd] 本头的查询
    float* red = qs + hd;                                 // [nt]

    const unsigned short* qp = q + (size_t)z * qb + (size_t)h * qh;
    for (int d = t; d < hd; d += nt) qs[d] = bf16f(qp[d]);

    float m = -1e30f, l = 0.f, acc = 0.f;
    if (nvalid > 0) {
        const int per = (nvalid + split - 1) / split;
        const int kfrom = s * per;
        const int kto = min(kfrom + per, nvalid);
        for (int base = kfrom; base < kto; base += ATTN_KEY_BLOCK) {
            const int nb = min(ATTN_KEY_BLOCK, kto - base);
            for (int i = t; i < nb; i += nt)
                rows[i] = (int)(row_slots[base + i] * nkv + g);
            __syncthreads();

            float lmax = -1e30f;
            for (int i = t; i < nb; i += nt) {
                float sc;
                if (row_mask && row_mask[(base + i) >> 6] == 0 && mask_block == 64) {
                    sc = -1e30f;                       // 被取消选中的块：跳过加载
                } else if (row_mask && mask_block != 64
                           && row_mask[(base + i) / mask_block] == 0) {
                    sc = -1e30f;
                } else {
                    const int8_t* kr = kpool + (size_t)rows[i] * hd;
                    float dot = 0.f;
                    for (int d = 0; d < hd; ++d) dot += qs[d] * (float)kr[d];
                    sc = dot * kscale[rows[i]] * scale;
                }
                w[i] = sc;
                lmax = fmaxf(lmax, sc);
            }
            const float bmax = block_reduce<true>(lmax, red);

            const float mnew = fmaxf(m, bmax);
            const float alpha = __expf(m - mnew);
            float lsum = 0.f;
            for (int i = t; i < nb; i += nt) {
                const float e = __expf(w[i] - mnew);
                w[i] = e;
                lsum += e;
            }
            l = l * alpha + block_reduce<false>(lsum, red);
            acc *= alpha;
            m = mnew;

            if (t < hd) {
                float a = 0.f;
                for (int i = 0; i < nb; ++i) {
                    if (w[i] == 0.f) continue;         // 被掩掉的 key：贡献恰好为 0
                    a += w[i] * vscale[rows[i]] * (float)vpool[(size_t)rows[i] * hd + t];
                }
                acc += a;
            }
            __syncthreads();
        }
    }

    if (t == 0) {
        const size_t ml = ((size_t)z * nh + h) * split + s;
        part_ml[ml * 2 + 0] = m;
        part_ml[ml * 2 + 1] = l;
    }
    if (t < hd) {
        const size_t pa = ((size_t)z * nh + h) * split + s;
        part_acc[pa * hd + t] = acc;
    }
}

// @brief 启动多查询校验注意力（c 个查询共用一趟历史遍历）。
//
// @param q,gate,out [B, c, nh, hd] bf16 连续存放；一个序列的 c 个解码行紧挨着排。
// @param slots      [B, smax] int64 连续槽位表，由一行的 c 个查询共享。
// @param ctx_lens   [B] int32 连续存放，每行的历史长度。
// @param max_seq    未使用，原因同 ``attn_paged_batch``：窗口是流式处理的，长度来自
//                   ``ctx_lens``。
// @param part_acc,part_ml [B*c, nh, split, hd] / [..., 2] fp32，归调用方所有，好让形状在
//                   图捕获时保持静态。
// @param block_mask 可选的 uint8 [B, nblocks] 保留掩码；未定义或空张量表示精确注意力。必须
//                   连续存放。
// @param mask_block 每个掩码块包含多少个 token，>= 1。
// @return ``out``，原地写出，便于链式调用。
// @throws c10::Error（TORCH_CHECK）：形状或 dtype 破坏了索引契约；去改调用方。
torch::Tensor attn_paged_multi_q(torch::Tensor q, torch::Tensor gate,
                                 torch::Tensor kpool, torch::Tensor vpool,
                                 torch::Tensor kscale, torch::Tensor vscale,
                                 torch::Tensor slots, torch::Tensor ctx_lens,
                                 torch::Tensor out, int64_t max_seq,
                                 int64_t rep, double scale,
                                 torch::Tensor part_acc, torch::Tensor part_ml,
                                 torch::Tensor block_mask, int64_t mask_block) {
    (void)max_seq;      // 流式窗口：长度是 ctx_lens[b] + j + 1
    TORCH_CHECK(q.dim() == 4 && gate.dim() == 4 && out.dim() == 4,
                "q/gate/out must be [B, c, nh, hd]");
    TORCH_CHECK(q.is_contiguous() && gate.is_contiguous() && out.is_contiguous(),
                "q/gate/out must be contiguous");
    TORCH_CHECK(q.stride(3) == 1, "unit inner stride required");
    const int B = q.size(0), c = q.size(1), nh = q.size(2), hd = q.size(3);
    const int Z = B * c;
    const int nkv = kpool.size(1);
    TORCH_CHECK(gate.sizes() == q.sizes() && out.sizes() == q.sizes(),
                "q/gate/out shape mismatch");
    TORCH_CHECK(slots.dim() == 2 && slots.size(0) == B && slots.is_contiguous(),
                "slots must be [B, max_seq] int64 and contiguous");
    TORCH_CHECK(ctx_lens.dim() == 1 && ctx_lens.size(0) == B
                && ctx_lens.scalar_type() == at::kInt && ctx_lens.is_contiguous(),
                "ctx_lens must be [B] int32 contiguous");
    TORCH_CHECK(part_acc.dim() == 4 && part_acc.size(0) == Z && part_acc.size(1) == nh
                && part_acc.size(3) == hd && part_acc.is_contiguous(),
                "part_acc must be [B*c, nh, split, hd] fp32");
    TORCH_CHECK(part_ml.dim() == 4 && part_ml.size(0) == Z && part_ml.size(1) == nh
                && part_ml.size(3) == 2 && part_ml.is_contiguous()
                && part_acc.size(2) == part_ml.size(2),
                "part_ml must be [B*c, nh, split, 2] fp32");
    TORCH_CHECK(mask_block >= 1, "mask_block must be >= 1");
    const int smax = (int)slots.stride(0);
    const unsigned char* mask_ptr = nullptr;
    int nblocks = 0;
    if (block_mask.defined() && block_mask.numel() > 0) {
        TORCH_CHECK(block_mask.dim() == 2 && block_mask.size(0) == B
                    && block_mask.scalar_type() == at::kByte
                    && block_mask.is_contiguous(),
                    "block_mask must be uint8 [B, n_blocks] contiguous");
        nblocks = (int)block_mask.size(1);
        mask_ptr = block_mask.data_ptr<unsigned char>();
    }
    const int split = part_acc.size(2);
    const int threads = 256;
    const int cthreads = hd > threads ? ((hd + 31) / 32) * 32 : threads;
    const size_t smem = (size_t)(2 * ATTN_KEY_BLOCK + hd + threads) * sizeof(float);
    hipStream_t stream = at::cuda::getCurrentCUDAStream();
    attn_paged_multi_q_kernel<<<dim3(c, nh, B * split), threads, smem, stream>>>(
        reinterpret_cast<const unsigned short*>(q.data_ptr<at::BFloat16>()),
        nh * hd, hd,
        kpool.data_ptr<int8_t>(), vpool.data_ptr<int8_t>(),
        kscale.data_ptr<float>(), vscale.data_ptr<float>(),
        slots.data_ptr<int64_t>(), smax,
        ctx_lens.data_ptr<int32_t>(), mask_ptr, nblocks, (int)mask_block,
        part_acc.data_ptr<float>(), part_ml.data_ptr<float>(),
        c, nh, nkv, hd, (int)rep, (float)scale, split);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    attn_paged_batch_combine_kernel<<<dim3(nh, Z), cthreads, 0, stream>>>(
        part_acc.data_ptr<float>(), part_ml.data_ptr<float>(),
        reinterpret_cast<const unsigned short*>(gate.data_ptr<at::BFloat16>()),
        nh * hd, hd,
        reinterpret_cast<unsigned short*>(out.data_ptr<at::BFloat16>()),
        nh * hd, hd, hd, split);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
