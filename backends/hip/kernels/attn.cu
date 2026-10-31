// @file attn.cu
// @brief int8 KV cache 上的融合单 token 全注意力。
//
// Module: coffin/kernels。B = 1 的解码路径，由 ``hip_ops.py`` 作为 torch 扩展的一部分
// 加载；批处理版本在 ``batch.cu``。
//
// 为什么融合：即时执行路径要把整个 KV 缓存实体化成 bf16（两次 ``.to(D)`` 拷贝
// [nkv, S, hd]），再跑两个 einsum，外加逐 token 的缩放折算、因果掩码、softmax 和输出
// 门控，全是独立内核。本内核一趟做完同样的事：边读边反量化 int8 的 k/v、把逐 token 的
// scale 折进分数和概率、施加门控，于是缓存始终不离开 int8，也不写出任何 [nkv, S, hd]
// 中间量。
//
// Constraints:
//   * key 窗口按 ATTN_KEY_BLOCK 个 key 流式处理并用 online softmax，绝不整块暂存：每个
//     CU 的 LDS 只有 64 KB，而在 262144 上下文下 [S] 个分数就是 1 MB，整块暂存根本启动
//     不了（见下文「key 分块」）。
//   * 每个查询头一个 block（再乘以切分数）；没有跨 block 同步。各切片发布部分和，由合并
//     内核汇总。
//   * K/V 保持 int8 加逐 token 的 fp32 scale；向下游只传行号，密集布局和分页布局因此共用
//     一个内核。
//   * 窗口长度和位置都在设备上：解码步是一张捕获的图，主机侧标量会在捕获时被冻住。
//
// key 分块：早先整个窗口的分数向量都放在共享内存里——[S] 个 float，S = max_seq——4096
// 时是 16 KB，262144 时是 1 MB。gfx906 每个 CU 只有 64 KB LDS，而超过 48 KB 的动态共享
// 分配需要额外声明属性，所以上下文一过 1.2 万左右就根本启动不了。现在窗口按
// ATTN_KEY_BLOCK 逐块遍历，配滑动最大值/和（online softmax）以及带缩放的输出累加器，
// 共享内存开销变成常数，256k 也装得下。计算量没变：老内核为了累加 V 本来就要走遍每个
// 有效 key，只是它把整个窗口当一个循环走，而不是一块一块走。

#include "hip_ops.h"

#include <c10/cuda/CUDAStream.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace {

//: 每个流式块处理多少个 key。它只限制 LDS 用量，加上每块对输出累加器的一次缩放；
//: 1024 个 key 约占 10 KB 共享内存。
constexpr int ATTN_KEY_BLOCK = 1024;

//: 每个查询头的 key 范围切分数。只按*头*分 block 只有 16 个 block，即这块 GPU
//: wavefront 槽位的约 5%，于是该步是延迟受限而不是带宽受限（实测 64k 上下文下每个 token
//: 3.1 us，KV 流量约 24 GB/s，而天花板约 600 GB/s）。每个切片 block 产出一份
//: (max, sum, acc) 部分和，由 ``attn_combine_kernel`` 合并，正是上面那套 online-softmax
//: 记账——算术没变，变的只是它的并行度。
constexpr int ATTN_SPLIT = 8;

__device__ __forceinline__ float bf16f(unsigned short h) {
    return __uint_as_float(((unsigned)h) << 16);
}

__device__ __forceinline__ unsigned short fbf16(float f) {
    unsigned u = __float_as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);   // 就近舍入到偶数（round to nearest even）
    return (unsigned short)(u >> 16);
}

// 通过 ``blockDim.x`` 个 float 的共享暂存做 block 内的最大值/求和。block 里每个线程都
// 必须调用它（函数结尾停在 ``__syncthreads()``），且 ``red`` 会被覆盖。
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

// @brief kv 头 ``g`` 下 key ``s`` 的行号，密集或分页布局通用。
//
// 密集缓存是头主序（[nkv, S, hd] -> row = g*S + s），分页池是 token 主序
// （[slots, nkv, hd] -> row = slot*nkv + g），所以一个公式同时覆盖两者，``pos2slot`` 为
// 空即表示「密集」。下游的一切（k、v 以及两个 scale）都用这一个行号索引。
//
// @param pos2slot 设备侧「绝对位置 -> 池槽位」映射；或空。
// @param stride   密集布局是 S，分页布局是 nkv。
// @return k、v 和两个逐 token scale 共用的扁平行号。
__device__ __forceinline__ int attn_row(int s, int g, const int64_t* pos2slot, int stride) {
    return pos2slot ? (int)((long)pos2slot[s] * stride + g) : g * stride + s;
}

// @brief 每个（查询头，key 范围切片）一个 block；解码注意力的 split-K 那一遍。
//
// @param pos2slot 设备侧「位置 -> 槽位」映射；密集布局时为空。
// @param stride   S（密集）或 nkv（分页），见 ``attn_row``。
// @param part_acc [nh, split, hd] fp32 的 online-softmax 分子，每个切片一份。
// @param part_ml  [nh, split, 2] fp32（滑动最大值、滑动和）。
// @param pos_ptr  设备侧 int64 位置；在这里读，因为解码图在捕获时会把标量内核参数冻住。
__global__ void attn_decode_kernel(const unsigned short* __restrict__ q,
                                   const unsigned short* __restrict__ gate,
                                   const int8_t* __restrict__ kc,
                                   const int8_t* __restrict__ vc,
                                   const float* __restrict__ ks,
                                   const float* __restrict__ vs,
                                   const int64_t* __restrict__ pos2slot,
                                   int stride,
                                   float* __restrict__ part_acc,
                                   float* __restrict__ part_ml,
                                   const int64_t* __restrict__ pos_ptr,
                                   int hd, int rep, float scale, int split) {
    const int h = blockIdx.x;
    const int s = blockIdx.y;        // 本 block 负责的 key 范围切片
    const int g = h / rep;
    const int pos = (int)*pos_ptr;   // 重放时才读：图会把标量参数冻住
    const int t = threadIdx.x;
    const int nt = blockDim.x;
    extern __shared__ float sh[];
    float* w = sh;                                        // [ATTN_KEY_BLOCK] 权重
    int* rows = (int*)(sh + ATTN_KEY_BLOCK);              // [ATTN_KEY_BLOCK] key -> 行号
    float* qs = sh + ATTN_KEY_BLOCK + ATTN_KEY_BLOCK;     // [hd] 本头的查询
    float* red = qs + hd;                                 // [nt]

    for (int d = t; d < hd; d += nt) qs[d] = bf16f(q[(size_t)h * hd + d]);

    // 只有位置 0..pos 存在，所以窗口停在那里，而不是把整个 max_seq 扫一遍（即时路径的
    // einsum 每一步都为全部买单）。随后窗口被切成 `split` 个切片；空切片不贡献任何东西
    // （它的最大值停在哨兵值上，和为 0）。
    const int nvalid = pos + 1;
    const int per = (nvalid + split - 1) / split;
    const int kfrom = s * per;
    const int kto = min(kfrom + per, nvalid);
    float m = -1e30f, l = 0.f, acc = 0.f;    // 滑动最大值、滑动和、本维累加
    for (int base = kfrom; base < kto; base += ATTN_KEY_BLOCK) {
        const int nb = min(ATTN_KEY_BLOCK, kto - base);
        for (int i = t; i < nb; i += nt) rows[i] = attn_row(base + i, g, pos2slot, stride);
        __syncthreads();

        // 分数：dot(q, 反量化的 k) * 逐 token scale * softmax scale
        float lmax = -1e30f;
        for (int i = t; i < nb; i += nt) {
            const int8_t* kr = kc + (size_t)rows[i] * hd;
            float dot = 0.f;
            for (int d = 0; d < hd; ++d) dot += qs[d] * (float)kr[d];
            const float sc = dot * ks[rows[i]] * scale;
            w[i] = sc;
            lmax = fmaxf(lmax, sc);
        }
        const float bmax = block_reduce<true>(lmax, red);

        // online softmax：先缩放已累加的部分，再把本块加进来。
        // 整块被掩掉时 bmax 停在哨兵值上，于是 mnew 不变、alpha 恰好为 1（全程没有 -inf
        // 参与运算）。
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

        // 反量化 v 的加权和（1/l 的除法留到最后再做）
        if (t < hd) {
            float a = 0.f;
            for (int i = 0; i < nb; ++i)
                a += w[i] * vs[rows[i]] * (float)vc[(size_t)rows[i] * hd + t];
            acc += a;
        }
        __syncthreads();     // rows[] 和 w[] 要给下一个块复用
    }

    // 发布本切片的部分和，供合并那一遍使用（门控在那里施加）
    if (t == 0) {
        part_ml[(size_t)(h * split + s) * 2 + 0] = m;
        part_ml[(size_t)(h * split + s) * 2 + 1] = l;
    }
    if (t < hd) part_acc[(size_t)(h * split + s) * hd + t] = acc;
}

// @brief 合并一个头各切片的部分和，并施加输出门控。
//
// 还是同一套 online-softmax 合并，只是现在跨切片而不是跨 key 块。每个线程各自重算全局
// 最大值（``split`` 个值），而不是把它暂存到共享内存——做一次 warp 归约的开销比重复读这
// 几个值还大。
//
// @param gate [nh, hd] bf16 输出门控；sigmoid 折进归一化后的结果里。
// @param out  [nh, hd] bf16，由前 ``hd`` 个线程写出。
__global__ void attn_combine_kernel(const float* __restrict__ part_acc,
                                    const float* __restrict__ part_ml,
                                    const unsigned short* __restrict__ gate,
                                    unsigned short* __restrict__ out,
                                    int hd, int split) {
    const int h = blockIdx.x;
    const int t = threadIdx.x;
    const float* ml = part_ml + (size_t)h * split * 2;
    float m = -1e30f;
    for (int s = 0; s < split; ++s) m = fmaxf(m, ml[s * 2]);
    if (t < hd) {
        float l = 0.f, a = 0.f;
        for (int s = 0; s < split; ++s) {
            const float alpha = __expf(ml[s * 2] - m);
            l += ml[s * 2 + 1] * alpha;
            a += part_acc[(size_t)(h * split + s) * hd + t] * alpha;
        }
        const float gv = bf16f(gate[(size_t)h * hd + t]);
        out[(size_t)h * hd + t] = fbf16(a / l / (1.f + __expf(-gv)));
    }
}

}  // namespace

// @brief 启动密集布局的解码注意力（分片那一遍 + 合并那一遍）。
//
// @param q,gate [nh, hd] bf16，连续存放。
// @param kcache,vcache [nkv, S, hd] int8 密集缓存，头主序。
// @param kscale,vscale [nkv, S] fp32 逐 token 反量化 scale。
// @param pos    设备侧 int64 位置；只访问位置 0..pos。
// @param part_acc,part_ml 调用方持有的 [nh, split, hd] / [nh, split, 2] fp32；``split`` 从
//              ``part_acc`` 读出，这样捕获时看到的是固定形状。
// @return ``out``，[nh, hd] bf16，原地写出，便于链式调用。
// @throws c10::Error（TORCH_CHECK）：内核启动检查在设备拒绝该 grid 时抛出；这里不做形状
//         校验，布局契约由调用方负责。
torch::Tensor attn_decode(torch::Tensor q, torch::Tensor gate,
                          torch::Tensor kcache, torch::Tensor vcache,
                          torch::Tensor kscale, torch::Tensor vscale,
                          torch::Tensor out, torch::Tensor pos,
                          int64_t rep, double scale,
                          torch::Tensor part_acc, torch::Tensor part_ml) {
    const int nh = q.size(0), hd = q.size(1);
    const int S = kcache.size(1);            // 密集布局：row = g*S + s
    const int split = part_acc.size(1);
    hipStream_t stream = at::cuda::getCurrentCUDAStream();
    const int threads = 256;
    const size_t smem = (size_t)(2 * ATTN_KEY_BLOCK + hd + threads) * sizeof(float);
    attn_decode_kernel<<<dim3(nh, split), threads, smem, stream>>>(
        reinterpret_cast<const unsigned short*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const unsigned short*>(gate.data_ptr<at::BFloat16>()),
        kcache.data_ptr<int8_t>(), vcache.data_ptr<int8_t>(),
        kscale.data_ptr<float>(), vscale.data_ptr<float>(),
        nullptr, S,
        part_acc.data_ptr<float>(), part_ml.data_ptr<float>(),
        pos.data_ptr<int64_t>(), hd, (int)rep, (float)scale, split);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    attn_combine_kernel<<<nh, threads, 0, stream>>>(
        part_acc.data_ptr<float>(), part_ml.data_ptr<float>(),
        reinterpret_cast<const unsigned short*>(gate.data_ptr<at::BFloat16>()),
        reinterpret_cast<unsigned short*>(out.data_ptr<at::BFloat16>()), hd, split);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// @brief 启动分页解码注意力：把逻辑位置的 K/V 行经 ``pos2slot`` 间接寻址。
//
// 这层间接必须在*设备*上，因为解码步是一张捕获的图：主机侧的块表在序列追加块时都得重新
// 烙进图里，而设备侧映射在下次重放前重写即可，不需要重新捕获。池按 token 主序排布
// （[slots, nkv, hd]，不同于密集缓存的头主序 [nkv, S, hd]），这样一个块的槽位在各头之间
// 是连续的，块的写时复制也只是一次连续拷贝。
//
// @param kpool,vpool [num_slots, nkv, hd] int8，某一层块池的视图。
// @param kscale,vscale [num_slots, nkv] fp32 逐槽位 scale。
// @param pos2slot 设备侧 int64 [S]，绝对位置到池槽位的映射；每次重放前都必须重写。
// @param max_pos 未使用：窗口是流式处理的，长度来自 ``pos``。保留它是为了不改调用方签名。
// @return ``out``，[nh, hd] bf16，原地写出，便于链式调用。
// @throws c10::Error（TORCH_CHECK）：内核启动检查在设备拒绝该 grid 时抛出；布局契约由调用
//         方负责。
torch::Tensor attn_paged_decode(torch::Tensor q, torch::Tensor gate,
                                torch::Tensor kpool, torch::Tensor vpool,
                                torch::Tensor kscale, torch::Tensor vscale,
                                torch::Tensor pos2slot, torch::Tensor out,
                                torch::Tensor pos, int64_t max_pos,
                                int64_t rep, double scale,
                                torch::Tensor part_acc, torch::Tensor part_ml) {
    (void)max_pos;      // 流式窗口：长度在设备上就是 pos + 1
    const int nh = q.size(0), hd = q.size(1);
    const int nkv = kpool.size(1);           // 分页布局：row = slot*nkv + g
    const int split = part_acc.size(1);
    hipStream_t stream = at::cuda::getCurrentCUDAStream();
    const int threads = 256;
    const size_t smem = (size_t)(2 * ATTN_KEY_BLOCK + hd + threads) * sizeof(float);
    attn_decode_kernel<<<dim3(nh, split), threads, smem, stream>>>(
        reinterpret_cast<const unsigned short*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const unsigned short*>(gate.data_ptr<at::BFloat16>()),
        kpool.data_ptr<int8_t>(), vpool.data_ptr<int8_t>(),
        kscale.data_ptr<float>(), vscale.data_ptr<float>(),
        pos2slot.data_ptr<int64_t>(), nkv,
        part_acc.data_ptr<float>(), part_ml.data_ptr<float>(),
        pos.data_ptr<int64_t>(), hd, (int)rep, (float)scale, split);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    attn_combine_kernel<<<nh, threads, 0, stream>>>(
        part_acc.data_ptr<float>(), part_ml.data_ptr<float>(),
        reinterpret_cast<const unsigned short*>(gate.data_ptr<at::BFloat16>()),
        reinterpret_cast<unsigned short*>(out.data_ptr<at::BFloat16>()), hd, split);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
