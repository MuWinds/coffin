// @file attn_sparse.cu
// @brief QSA 的稀疏注意力：只读索引器选中的位置，不物化 KV，不扫全窗口。
//
// Module: coffin/backends/corex/kernels。由 ``backends/corex`` 的 ``attn_sparse``
// 调用；``models/qwen4exp/qsa.py`` 在索引器出掩码之后用它替换那三段纯 torch。
//
// ## 为什么需要它
//
// ``qsa.py`` 的主注意力原来完全没走融合内核，是纯 torch：
//
//     keys   = repeat_kv(new_cache.keys, 8)     # [1,2,kv,256] -> [1,16,kv,256] 物化
//     values = repeat_kv(new_cache.values, 8)
//     w = matmul(q, keys.transpose(2,3)) * scaling   # [1,16,1,kv]
//     w = softmax(w, dim=-1, dtype=float32)
//     out = matmul(w, values)                        # [1,16,1,256]
//
// 单层、decode ``S=1``、bf16 KV 逐项拆开量：
//
//     kv      repeat_kv    matmul    attn_full
//     512       0.038 ms   0.139 ms    0.324 ms
//     2048      0.114      0.530       1.173 ms
//     8192      0.418      2.454       4.426 ms
//     32768     1.669      9.141      16.679 ms
//
// 索引器预算只有 ``budget = 2048``，所以 ``kv = 32768`` 时 94% 的打分与加权求和作用
// 在已经被判定为「不看」的位置上——先算出来、再被 ``-inf`` 压掉。本内核只遍历选中的
// 位置，读取量从 ``O(kv_length)`` 降到 ``O(min(kv_length, budget))``。
//
// ## 结构：这是第一版失败后才改对的
//
// 最开始让每个 block 串行遍历全部选中位置、每处理一个位置
// 做两次全块归约，于是 ``n = 2048`` 时有 4096 次 ``__syncthreads``，而 grid 只有
// ``16`` 个 block（一个查询头一个）。比稠密慢 1.1-14 倍：
//
//     kv=2048   dense=0.590 ms  sparse=8.735 ms   0.07x
//     kv=32768  dense=7.946 ms  sparse=8.730 ms   0.91x
//
// 稀疏版的绝对耗时完全不随 kv 变（8.73 ms 一路不变），说明瓶颈就是那个串行循环，
// 与读了多少 KV 无关。稠密路径随 kv 线性涨，所以在 kv=32768 处两者才接近碰头。
//
// 改成：**每个（查询头, 行）一个 block，块内 32 个 warp 分摊选中位置，warp 内按
// ``hd`` 再做 splat 归约（每个 lane 负责 ``hd / 32`` 个通道，份额跨 warp 求和）**。
// 关键是**循环里不再有任何 ``__syncthreads``**：每个 warp 独立跑在线 softmax，
// 只在最后把 ``m``/``l``/``acc`` 归并一次。
//
// ## 顺带省掉 ``repeat_kv``
//
// GQA 下 16 个查询头只对应 2 个 KV 头（``rep = 8``）。内核按 ``g = h / rep`` 直接
// 寻址，KV 只被读一遍、**从不物化成 16 份**。那一条在 kv=32768 时是 1.67 ms/层/步
// （12 层 20 ms），省掉是纯收益。
//
// Constraints:
//   * ``sel_idx`` 行内必须是**升序**且无效位置为 -1（索引器产出的就是升序）。
//   * 空行（``sel_len == 0``）写 0 而不是 NaN。
//   * 选中位置必须 ``< ctx_row[row]``；越界的直接跳过（防御未初始化的尾部槽）。
//   * ``hd`` 必须是 32 的倍数，且 ``hd / 32 <= ATT_SPARSE_LANES``。
//   * 不用 CUDA graph、不用 cp.async（``platform/scow/AGENTS.md`` 第 1、5 条）。

#include "corex_ops.h"
#include "corex_scalar.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

namespace {

//: 每个 warp 负责一个选中位置，warp 内的 32 条 lane 分摊 ``hd`` 个通道。
constexpr int WARP = 32;

//: 一个 block 里并发的选中位置数（即 warp 数）。8 个 warp 让 256 线程的块吃满，
//: 而 ``max_sel`` 最多 2048，所以每个 warp 串 ``2048 / 8 = 256`` 个位置 —— 但那是
//: **没有归约开销**的串行，与第一版的 4096 次同步完全不同。
constexpr int WARPS = 8;

//: 每条 lane 最多负责几个通道。``hd = 256``、``32 lane`` 时是 8。
constexpr int ATT_SPARSE_LANES = 8;

// @brief warp 内把 ``v`` 求和并广播。
//
// @param v 本 lane 的值。
// @return 全 warp 求和。
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int s = WARP / 2; s > 0; s >>= 1) v += __shfl_xor_sync(0xffffffffu, v, s);
    return v;
}

// @brief warp 内把 ``v`` 求最大值并广播。
//
// @param v 本 lane 的值。
// @return 全 warp 最大值。
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int s = WARP / 2; s > 0; s >>= 1) {
        const float o = __shfl_xor_sync(0xffffffffu, v, s);
        if (o > v) v = o;
    }
    return v;
}

// @brief 一个（行, 查询头）的稀疏精确 softmax 注意力。
//
// 每个 warp 独立处理一部分选中位置并跑在线 softmax（rescaling 的在线形式），最后
// 把各 warp 的 ``(m, l, acc)`` 按「合并两个 softmax」的公式归并。这样循环里没有
// 任何块级同步，且并行度从「一个头一个 block 的单循环」变成「头 × warp」。
//
// @param q       ``[rows * nh * hd]`` bf16。
// @param kc,vc   稠密 KV 缓存 bf16，**头主序** ``[nkv, max_seq, hd]``（未 repeat）。
// @param sel_idx 设备侧 ``[rows * max_sel]`` int32 选中位置，行内升序，无效为 -1。
// @param sel_len 设备侧 ``[rows]`` int32 每行有效位置数。
// @param ctx_row 设备侧 ``[rows]`` int32 每行的因果上界。
// @param out     ``[rows * nh * hd]`` bf16。
// @param nh, nkv, hd 几何。
// @param max_seq 每头行数（头主序行间距）。
// @param max_sel 每行槽位上限（``sel_idx`` 行间距）。
// @param scale   打分缩放，通常 ``hd ** -0.5``。
__global__ void attn_sparse_kernel(const uint16_t* __restrict__ q,
                                   const uint16_t* __restrict__ kc,
                                   const uint16_t* __restrict__ vc,
                                   const int* __restrict__ sel_idx,
                                   const int* __restrict__ sel_len,
                                   const int* __restrict__ ctx_row,
                                   uint16_t* __restrict__ out,
                                   int nh, int nkv, int hd, int max_seq,
                                   int max_sel, float scale) {
    const int h = blockIdx.x;
    const int row = blockIdx.y;
    const int rep = nh / nkv;
    const int g = h / rep;
    const int tid = threadIdx.x;
    const int warp = tid / WARP;           // 本线程属于哪个位置组
    const int lane = tid % WARP;           // 本线程在 warp 内的 lane

    const size_t out_base = ((size_t)row * nh + h) * (size_t)hd;
    const int n = sel_len[row];
    const int ctx = ctx_row[row];

    if (n <= 0 || hd <= 0) {
        for (int d = tid; d < hd; d += blockDim.x) out[out_base + d] = 0;
        return;
    }

    // 本 lane 负责的通道是 ``lane, lane + 32, ...``；跨 warp 的同一 lane 负责同一批
    // 通道，所以最后按 lane 归并各 warp 的部分和即可。
    const uint16_t* qrow = q + out_base;
    float qreg[ATT_SPARSE_LANES];
#pragma unroll
    for (int c = 0; c < ATT_SPARSE_LANES; ++c) {
        const int d = lane + c * WARP;
        qreg[c] = (d < hd) ? CxScalar<uint16_t>::to_float(qrow[d]) : 0.f;
    }

    const int* idx = sel_idx + (size_t)row * max_sel;
    const uint16_t* kbase = kc + (size_t)g * (size_t)max_seq * (size_t)hd;
    const uint16_t* vbase = vc + (size_t)g * (size_t)max_seq * (size_t)hd;

    // 在线 softmax 的三元组，每个 warp 一份。
    float m = -1e30f;
    float l = 0.f;
    float acc[ATT_SPARSE_LANES];
#pragma unroll
    for (int c = 0; c < ATT_SPARSE_LANES; ++c) acc[c] = 0.f;

    for (int j = warp; j < n; j += WARPS) {
        const int p = idx[j];
        if (p < 0 || p >= ctx) continue;
        const uint16_t* kp = kbase + (size_t)p * (size_t)hd;
        float dot = 0.f;
#pragma unroll
        for (int c = 0; c < ATT_SPARSE_LANES; ++c) {
            const int d = lane + c * WARP;
            if (d < hd) dot += qreg[c] * CxScalar<uint16_t>::to_float(kp[d]);
        }
        // 全 warp 的点积：lane 之间还没有合并过。
        dot = warp_sum(dot) * scale;

        // 在线更新：只有新分数超过当前最大值时才 rescale，避免每步都乘。
        if (dot > m) {
            const float corr = __expf(m - dot);
#pragma unroll
            for (int c = 0; c < ATT_SPARSE_LANES; ++c) acc[c] *= corr;
            l *= corr;
            m = dot;
        }
        const float w = __expf(dot - m);
        l += w;
        const uint16_t* vp = vbase + (size_t)p * (size_t)hd;
#pragma unroll
        for (int c = 0; c < ATT_SPARSE_LANES; ++c) {
            const int d = lane + c * WARP;
            if (d < hd) acc[c] += w * CxScalar<uint16_t>::to_float(vp[d]);
        }
    }

    // 每个 warp 的 (m, l, acc) 写到共享内存，再按「合并两个 softmax」归并。
    //
    // 共享内存布局：l/warp 与 m/warp 各一条，acc 是 ``[WARPS, hd]`` 的 fp32
    // —— ``8 * 256 * 4 = 8 KB``，在这张卡的 64 KB 预算内。
    extern __shared__ float sh[];
    float* sh_m = sh;                      // [WARPS]
    float* sh_l = sh_m + WARPS;            // [WARPS]
    float* sh_acc = sh_l + WARPS;          // [WARPS * hd]

    if (lane == 0) {
        sh_m[warp] = m;
        sh_l[warp] = l;
    }
    // acc 按通道写到本 warp 的槽位。同一 lane 跨 warp 负责同一批通道，所以写进去
    // 之后按通道扫描即可。
#pragma unroll
    for (int c = 0; c < ATT_SPARSE_LANES; ++c) {
        const int d = lane + c * WARP;
        if (d < hd) sh_acc[warp * hd + d] = acc[c];
    }
    __syncthreads();

    for (int d = tid; d < hd; d += blockDim.x) {
        // 全局最大。
        float gm = sh_m[0];
        for (int w = 1; w < WARPS; ++w) {
            if (sh_m[w] > gm) gm = sh_m[w];
        }
        // 按 gm 重新缩放并求和。
        float gl = 0.f;
        float ga = 0.f;
        for (int w = 0; w < WARPS; ++w) {
            const float s = __expf(sh_m[w] - gm);
            gl += sh_l[w] * s;
            ga += sh_acc[w * hd + d] * s;
        }
        out[out_base + d] = (gl > 0.f)
            ? CxScalar<uint16_t>::from_float(ga / gl)
            : CxScalar<uint16_t>::from_float(0.f);
    }
}

}  // namespace

// @brief 稀疏 QSA 注意力：只读选中的位置，KV 不物化。
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
                          torch::Tensor ctx, torch::Tensor out, double scale) {
    TORCH_CHECK(q.is_cuda() && q.dim() == 3 && q.is_contiguous()
                && q.scalar_type() == torch::kBFloat16,
                "q must be contiguous CUDA bf16 [rows, nh, hd]");
    TORCH_CHECK(kcache.is_cuda() && kcache.dim() == 3 && kcache.is_contiguous()
                && kcache.scalar_type() == torch::kBFloat16,
                "kcache must be contiguous CUDA bf16 [nkv, max_seq, hd]");
    TORCH_CHECK(vcache.is_cuda() && vcache.dim() == 3 && vcache.is_contiguous()
                && vcache.scalar_type() == torch::kBFloat16,
                "vcache must be contiguous CUDA bf16 [nkv, max_seq, hd]");
    TORCH_CHECK(sel_idx.is_cuda() && sel_idx.dim() == 2 && sel_idx.is_contiguous()
                && sel_idx.scalar_type() == torch::kInt,
                "sel_idx must be contiguous CUDA int32 [rows, max_sel]");
    TORCH_CHECK(sel_len.is_cuda() && sel_len.dim() == 1 && sel_len.is_contiguous()
                && sel_len.scalar_type() == torch::kInt
                && sel_len.size(0) == q.size(0),
                "sel_len must be contiguous CUDA int32 [rows]");
    TORCH_CHECK(ctx.is_cuda() && ctx.dim() == 1 && ctx.is_contiguous()
                && ctx.scalar_type() == torch::kInt && ctx.size(0) == q.size(0),
                "ctx must be contiguous CUDA int32 [rows]");
    TORCH_CHECK(out.is_cuda() && out.dim() == 3 && out.is_contiguous()
                && out.scalar_type() == torch::kBFloat16
                && out.size(0) == q.size(0) && out.size(1) == q.size(1)
                && out.size(2) == q.size(2),
                "out must be contiguous CUDA bf16 [rows, nh, hd]");

    const int rows = (int)q.size(0);
    const int nh = (int)q.size(1);
    const int hd = (int)q.size(2);
    const int nkv = (int)kcache.size(0);
    const int max_seq = (int)kcache.size(1);
    const int max_sel = (int)sel_idx.size(1);
    TORCH_CHECK((int)kcache.size(2) == hd && (int)vcache.size(0) == nkv
                && (int)vcache.size(1) == max_seq && (int)vcache.size(2) == hd,
                "v cache must match k cache shape");
    TORCH_CHECK(nkv > 0 && nh % nkv == 0, "nh must be a multiple of nkv");
    TORCH_CHECK(rows > 0 && nh > 0 && hd > 0, "sizes must be positive");
    TORCH_CHECK(hd % WARP == 0 && hd / WARP <= ATT_SPARSE_LANES,
                "hd must be a multiple of ", WARP, " and at most ",
                WARP * ATT_SPARSE_LANES);

    const int threads = WARPS * WARP;
    const c10::cuda::CUDAGuard guard(q.device());
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const size_t smem = (size_t)(2 * WARPS + WARPS * hd) * sizeof(float);
    const dim3 grid((unsigned)nh, (unsigned)rows);
    attn_sparse_kernel<<<grid, threads, smem, stream>>>(
        reinterpret_cast<const uint16_t*>(q.data_ptr()),
        reinterpret_cast<const uint16_t*>(kcache.data_ptr()),
        reinterpret_cast<const uint16_t*>(vcache.data_ptr()),
        sel_idx.data_ptr<int>(), sel_len.data_ptr<int>(), ctx.data_ptr<int>(),
        reinterpret_cast<uint16_t*>(out.data_ptr()), nh, nkv, hd, max_seq,
        max_sel, (float)scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
