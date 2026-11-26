// @file attn_slice_device.inc
// @brief 注意力的切片与合并两趟：在线 softmax 的一片 + 分片合并加输出门。
//
// Module: coffin/backends/corex/kernels。被两条前端各包含一次：``attn.cu``（torch 扩展）
// 与 ``abi/attn_slice_abi.cu``（``extern "C"`` 出口）。同一份设备代码。
//
// 四个公开入口（``attn_decode`` / ``attn_paged`` / ``attn_paged_batch`` /
// ``attn_paged_multi_q``）全部汇到 ``launch_attn_slice`` 一个启动器，所以稠密与分页两条
// 路在线程数、共享内存和网格几何上不可能各自漂移。
//
// Constraints:
//   * ``kc`` / ``vc`` 是 int8 缓存，``ks`` / ``vs`` 是每（槽位，KV 头）的 fp32 scale；
//     布局由 ``AttnLayout`` 的三个步长选择，内层循环里没有按布局分支。
//   * ``pos2slot`` / ``ctx_lens`` / ``pos`` 都在**设备**上解引用：主机张量能通过形状检查
//     然后被当指针读，所以这是入口边界必须挡的那一类错误。
//   * ``split`` 从 ``part_acc`` 的倒数第二轴读（稠密是 ``[nh, split, hd]``，批处理是
//     ``[B, nh, split, hd]``）；按第 2 轴读会让稠密路径以一个 ``hd`` 宽的网格启动。

#ifndef COFFIN_ATTN_SLICE_DEVICE_INC
#define COFFIN_ATTN_SLICE_DEVICE_INC

#include "corex_scalar.h"

#include <cstdint>

#include "attn_reduce_device.inc"

//: 每个 block 流式处理多少个键。把共享内存限制在几 KB 加每块一次输出累加器重缩放，
//: 与注意力窗口长度无关。
constexpr int KEY_BLOCK = 1024;

//: 「还没见过键」的哨兵值。大而有限，让第一块的 ``mnew - m`` 保持精确；``-inf`` 会把
//: ``exp(m - mnew)`` 那次重缩放变成 NaN。
constexpr float NO_MAX = -1e30f;

// @brief kv 头 ``g`` 的第 ``s`` 个键的平坦行号，稠密或分页。
//
// 行号一律是 ``slot * stride_slot + g * stride_head``；三种布局只是两个步长取不同的值：
// 稠密（``[nkv, S, hd]``）是 ``1`` 与 ``S``，分页 token 主序（``[slots, nkv, hd]``）是
// ``nkv`` 与 ``1``，分页头主序（``[nkv, slots, hd]``）是 ``1`` 与 ``nslots``。
//
// **写成两个步长而不是按布局分支**：同一份内核只改池布局实测差 1.93x（4k 单层 0.806 ->
// 0.418 ms），说明浪费在访存是否连续、不在映射本身；而每条指令多一次判断的开销在这张卡上
// 并不便宜，所以布局差异必须在启动参数里消化掉，不能落到内层循环里。
//
// @param s 窗口内的位置。
// @param g KV 头。
// @param pos2slot 设备侧位置到槽位的映射，稠密布局下为 null。
// @param stride_slot 相邻槽位之间的行距。
// @param stride_head 相邻 KV 头之间的行距。
// @param map_base 本次查询的槽位表行起始的元素下标。
// @return k、v 和两个 scale 共用的平坦行号。
__device__ __forceinline__ int cache_row(int s, int g, const int64_t* pos2slot,
                                         int stride_slot, int stride_head, int map_base) {
    const int slot = pos2slot ? (int)pos2slot[map_base + s] : s;
    return slot * stride_slot + g * stride_head;
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
    //: 相邻槽位之间的行距：稠密与头主序是 ``1``，分页 token 主序是 ``nkv``。
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
    //: 相邻 KV 头之间的行距：稠密是 ``S``，token 主序是 ``1``，头主序是 ``nslots``。
    //:
    //: 同样加在末尾：四个入口都是位置式聚合初始化，插在中间会把 ``max_seq`` 重新绑掉，
    //: 窗口随之变成 0。
    int stride_head;
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
            rows[i] = cache_row(base + i, g, layout.pos2slot, layout.stride,
                                layout.stride_head, map_base);
        }
        __syncthreads();

        float lmax = NO_MAX;
        for (int i = t; i < nb; i += nt) {
            const int8_t* kr = kc + (size_t)rows[i] * hd;
            float dot = 0.f;
            // 逐字节标量加载在这里是灾难：warp 内 32 个线程各自处理 *不同* 的键、同一列号，
            // 于是每条指令摸到 32 个不同的 32 B sector 而每 sector 只用 1 字节 —— 32x 的读取
            // 浪费。实测 seq=65536、rep=8 时该内核的有效带宽只有 17 GB/s，反推实际流量约
            // 540 GB/s（已打满 HBM），耗时 31.9 ms。改成 int4 一次取 16 字节后，每条指令的
            // 有效字节数从 1 提到 16，同一对齐前提下 sector 浪费降到 2x。
            int d = 0;
            for (; d + 16 <= hd; d += 16) {
                const int4 raw = *reinterpret_cast<const int4*>(kr + d);
                const int8_t* b = reinterpret_cast<const int8_t*>(&raw);
#pragma unroll
                for (int j = 0; j < 16; ++j) dot += qs[d + j] * (float)b[j];
            }
            for (; d < hd; ++d) dot += qs[d] * (float)kr[d];
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

// @brief 一次注意力调用的两趟启动：切片，然后合并加门。
//
// @param q [flat, nh, hd] bf16 查询。
// @param gate [flat, nh, hd] bf16 门，合并那一趟施加 ``sigmoid``。
// @param kc,vc 任一布局的 int8 缓存。
// @param ks,vs fp32 反量化 scale。
// @param out [flat, nh, hd] bf16 输出，原地写入。
// @param part_acc,part_ml 调用方的 split-K 暂存。
// @param mask 可选的 uint8 ``[flat / draft, n_blocks]`` 保留掩码，精确注意力时为 null。
// @param layout 缓存寻址方式。
// @param flat 行数（批处理是 ``B``，投机校验是 ``B * c``）。
// @param draft 共用一行历史的查询数；普通解码为 1。
// @param mask_block 掩码按多大的块表达。
// @param stream 目标流。
void launch_attn_slice(const uint16_t* q, const uint16_t* gate, const int8_t* kc,
                       const int8_t* vc, const float* ks, const float* vs, uint16_t* out,
                       float* part_acc, float* part_ml, const uint8_t* mask, AttnLayout layout,
                       int flat, int nh, int hd, int rep, float scale, int split, int draft,
                       int mask_block, cudaStream_t stream) {
    const int threads = attn_threads(hd);
    const int blocks = mask != nullptr ? mask_block : 0;
    attn_slice_kernel<<<dim3(nh, split, flat), threads, attn_smem(hd, threads), stream>>>(
        q, mask, kc, vc, ks, vs, layout, part_acc, part_ml, nh, hd, rep, scale, split, draft,
        blocks);
    attn_combine_kernel<<<dim3(nh, flat), threads, 0, stream>>>(part_acc, part_ml, gate, out, nh,
                                                               hd, split);
}

#endif  // COFFIN_ATTN_SLICE_DEVICE_INC
