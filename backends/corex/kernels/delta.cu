// @file delta.cu
// @brief 门控 delta 规则的解码步：对循环状态做一次原地更新。
//
// Module: coffin/backends/corex/kernels。消费 ``linear_attn.cu`` 产出的 fp32
// q/k/v/beta/g 并原地更新状态；批处理形式是本文件里的 ``delta_batch``。
// ``delta_normed`` 是解码路径：它接收原始卷积输出，在一次启动里做完头预处理、递推和
// 门控输出归一化，所以它们之间不跨内核边界。
//
// 每个 token 的递推是
//     st  = st * exp(g)[:, None, None]
//     kv  = (st * k[:, :, None]).sum(-2)
//     st  = st + k[:, :, None] * ((v - kv) * beta)[:, None, :]
//     out = (st * q[:, :, None]).sum(-2)
// 写成一条 torch 算子链，在 DCU 上每层约 140 us，24 层就是解码步的十分之一。状态没有
// 闭式回滚，所以它是前缀复用必须打快照的那一块层内内存。
//
// Constraints:
//   * 一个线程独占一列值，一个 block 覆盖一个头，块内线程按连续的 ``j`` 编址。这由状态
//     布局逼出来：一个头的第 ``j`` 列位于 ``st[head*dk*dv + i*dv + j]``，所以同一 ``i``
//     上连续的 ``j`` 是唯一能合并的分配方式。早先的 ``(列, dk 下标)`` 映射让每个通道按
//     ``dv`` 个 float 的步长读一列，拉进来 32 倍于实际用到的字节，这就是它实测 294 us
//     而启动下限只有 7 us 的原因。
//   * 两次 ``dk`` 归约都是每线程的寄存器循环，所以没有跨通道通信，也没有共享暂存。这依赖
//     递推只在一列内部耦合 ``i``；如果需要跨列求和，就得重新引入树——而为 32 写的蝶形在
//     这里会把错误的伙伴组合起来，因为本部件的 warp 是 64 通道宽。
//   * 状态是 fp32 且原地更新，输出从 *更新后* 的状态读出，这样一个 token 能看到自己写
//     进去的值。换任何别的顺序都会改变递推本身，而不只是舍入。
//   * 衰减在第二趟重新施加，而不是写回两次：``k^T (e^g S) == e^g (k^T S)``，这省掉了
//     每层每 token 对状态的一次读改写。
//   * 内核不得为 ``st`` 分配替代缓冲区：捕获路径要求那个地址固定，新缓冲区会让图写到一个
//     已死的张量上。
//   * 输入是 fp32 且连续。预处理已经归一化并缩放好 q 和 k，所以本内核不做任何逐元素修正。
//   * ``delta_normed`` 与 ``conv1d_silu`` -> ``delta_prologue`` -> ``delta`` ->
//     ``rms_norm_gated`` 那条链逐位相同，而不只是接近。这逼出两件本文件其它内核不需要的
//     事：块级归约必须精确复现那条链的共享树配对（因此用同样的 ``block_sum2`` 形状和同样
//     的块大小），而且递推的输出在归一化之前要先舍入成激活 dtype，因为 ``RMSNormGated``
//     在做归约之前把 fp32 的 ``delta_out`` 转成 bf16。贪心 id 是验收标准，这正是保住它们
//     的东西。

#include "corex_ops.h"
#include "corex_scalar.h"

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

namespace {

//: 每块线程数。每个线程独占一列值，并在寄存器里走完整个 ``dk`` 轴，所以一个 block 覆盖
//: ``THREADS`` 列，block 数相对旧形状（32 头 x 128/THREADS 个列分块）降到三十二分之一。
//:
//: 这个映射就是全部优化，值得说清为什么。状态是 ``[heads, dk, dv]``，一个头的第 ``j`` 列
//: 是 ``st[head*dk*dv + i*dv + j]``：``i`` 固定时在 ``j`` 上连续。所以线程按同一 ``i`` 上
//: 连续的 ``j`` 编址，一个 64 通道的 wavefront 读 64 个连续 float——一次 256 字节的事务。
//: 早先的映射把一列组的线程放在连续的 ``i`` 上，于是 warp 里每个通道都碰一条不同的
//: 128 字节行、只用到其中四个字节：同样的答案，32 倍的流量，实测每层 294 us 而启动下限
//: 是 7.3 us。
//:
//: 128 让这个模型里整个头的值轴（``dv``）待在一个 block 里，所以 q 和 k 的共享副本只加载
//: 一次、被每一列复用。在 64 通道的部件上它也是每块四个 wavefront，留下的 warp 足够掩盖
//: 延迟。
constexpr int THREADS = 128;

// @brief 门控 delta 规则的一步：每列值一个线程，每个头一个 block。
//
// 每个线程独占第 ``j`` 列，并在寄存器循环里把 ``i`` 走过 ``dk``，所以两次跨 ``dk`` 的归约
// 都是那个线程本地的事，完全不发生跨通道通信。这之所以正确，只因为两个和在各列之间独立
// ——递推在一列内部耦合 ``i``，从不跨列。
//
// 网格的 ``z`` 是批行。每一行都是独立的 ``[heads, dk, dv]`` 板，所以 block 只是偏移它的
// 指针，不去碰别的行拥有的东西；没有需要补的跨行通信。实测这一点远比它替代的那次启动
// 值得：``delta_batch`` 每行一次启动，每次只有 ``heads`` = 32 个 block 在飞，而本部件每个
// block 只跑得动约 3.0 GB/s，于是一行的 6.29 MB 以 101.7 GB/s 走完（61.89 us）。把行放到
// ``z`` 轴上就能让 ``32 * rows`` 个 block 同时在飞。
//
// @param q,k [rows, heads, dk] fp32，已经由预处理做过 L2 归一化（q 还做了缩放）。
// @param v [rows, heads, dv] fp32。
// @param beta,g [rows, heads] fp32，写强度和对数衰减。
// @param st [rows, heads, dk, dv] fp32 状态，原地更新。
// @param out [rows, heads, dv] fp32，从更新后的状态写出。
// @param heads 每行的头数，也是 ``x`` 方向的网格长度；下面的行步长由它推出，而早先的
//        每行主机循环是自己施加这些步长的。
// @param dk 归约长度，同时也是相邻值列之间的步长。
// @param dv 每个头的值列数。
__global__ void delta_kernel(const float* __restrict__ q, const float* __restrict__ k,
                             const float* __restrict__ v, const float* __restrict__ beta,
                             const float* __restrict__ g, float* __restrict__ st,
                             float* __restrict__ out, int heads, int dk, int dv) {
    const int head = blockIdx.x;
    const int j = (int)(blockIdx.y * blockDim.x + threadIdx.x);

    // 行偏移在设备侧算，不在主机侧做指针算术：缓冲区是稠密的，行步长就是形状本身，所以
    // 内核不用间距参数、也不用第二次启动就能命名第 ``b`` 行。单行解码入口下 ``z`` 是 1，
    // 第 0 行给每个指针加零，所以那条路径读到的字节和以前完全一样。
    const size_t row = blockIdx.z;
    const size_t oq = row * heads * dk;          // q 和 k，每行 [heads, dk]
    const size_t ov = row * heads * dv;          // v 和 out，每行 [heads, dv]
    const size_t ob = row * heads;               // beta 和 g，每行 [heads]
    const size_t os = row * heads * dk * dv;     // 状态，每行 [heads, dk, dv]

    // 超出 ``dv`` 的线程绝不能在 *这里* 返回。下面的 ``__syncthreads()`` 要由块内每个线程
    // 走到，提前退出会让其余线程等一个已经不在的线程——设备侧死锁，没有 traceback，正是
    // 曾经把一张卡挂满整个任务时限的那种症状。越界的列改为走完屏障，只跳过它自己的状态和
    // 输出访问。
    const bool active = j < dv;

    extern __shared__ float sh[];
    float* ks = sh;                  // [dk] 本头的 key，所有列共用
    float* qs = sh + dk;             // [dk] 本头的 query
    // 每一列都需要整个 ``dk``，所以这份拷贝是共享的而不是每线程各来一遍。加载时线程协作，
    // 两种方式下都是连续的。
    for (int i = (int)threadIdx.x; i < dk; i += (int)blockDim.x) {
        ks[i] = k[oq + (size_t)head * dk + i];
        qs[i] = q[oq + (size_t)head * dk + i];
    }
    __shared__ float decay_sh;
    __shared__ float rate_sh;
    if (threadIdx.x == 0) {
        decay_sh = __expf(g[ob + head]);
        rate_sh = beta[ob + head];
    }
    __syncthreads();
    const float decay = decay_sh;
    const float rate = rate_sh;
    if (!active) return;

    const float value = v[ov + (size_t)head * dv + j];
    float* column = st + os + (size_t)head * dk * dv + j;

    // 读的时候施加一次衰减、写的时候再施加一次，这样状态每步只被碰一次。同一 ``i`` 上的
    // 线程在 ``j`` 上相邻，所以每个 ``i`` 步对每个 wavefront 就是一次合并事务：这正是旧
    // 映射缺的访问模式。
    float kv_mem = 0.f;
    for (int i = 0; i < dk; ++i) {
        kv_mem = fmaf(column[(size_t)i * dv] * decay, ks[i], kv_mem);
    }

    const float delta = (value - kv_mem) * rate;
    float result = 0.f;
    for (int i = 0; i < dk; ++i) {
        const float s = fmaf(ks[i], delta, column[(size_t)i * dv] * decay);
        column[(size_t)i * dv] = s;
        result = fmaf(s, qs[i], result);
    }
    out[ov + (size_t)head * dv + j] = result;
}

// @brief 在 ``rows`` 块独立状态板上启动递推。
//
// 每个（头，列分块，行）一个 block。``z`` 方向就是行，单行解码入口下为 1，所以两个入口
// 共用这同一种启动形状，解码路径是靠构造保持不变，而不是靠另一条代码路径。
//
// 网格故意取行轴而不是主机循环：本部件每个在飞的 block 跑得动约 3.0 GB/s
// （``docs/DESIGN.md`` §2.2），这里一行只有 ``heads`` = 32 个 block，所以循环不管多少行
// 都只能以 32-block 的带宽跑每一行。
//
// @param q,k [rows, heads, dk] fp32。
// @param v [rows, heads, dv] fp32。
// @param beta,g [rows, heads] fp32。
// @param st [rows, heads, dk, dv] fp32，原地更新。
// @param out [rows, heads, dv] fp32。
// @param rows 批行数；成为网格的 ``z`` 方向长度。
// @param stream 启动所用的流。
void launch_rows(const float* q, const float* k, const float* v, const float* beta,
                 const float* g, float* st, float* out, int rows, int heads, int dk, int dv,
                 cudaStream_t stream) {
    const int threads = dv < THREADS ? dv : THREADS;
    const int blocks_y = (dv + threads - 1) / threads;
    const size_t smem = (size_t)(2 * dk) * sizeof(float);
    delta_kernel<<<dim3(heads, blocks_y, rows), threads, smem, stream>>>(
        q, k, v, beta, g, st, out, heads, dk, dv);
}

}  // namespace

// @brief 门控 delta 规则的一个解码步，原地推进循环状态。
//
// @param q,k [heads, dk] fp32，已经由预处理做过 L2 归一化，q 还额外缩放过，所以本内核
//        不做任何逐元素修正。
// @param v [heads, dv] fp32。
// @param beta,g [heads] fp32，速率和对数衰减；``g <= 0``，否则递推会发散。
// @param st [heads, dk, dv] fp32 状态，原地更新——前缀复用所依赖的快照纪律归调用方负责。
// @param out [heads, dv] fp32，从更新后的状态写出。
// @return ``out``，便于调用链串接。
// @throws c10::Error 非 beta/g 的输入不连续、dtype 不是 fp32，或头数与状态不一致。
torch::Tensor delta(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor beta,
                    torch::Tensor g, torch::Tensor st, torch::Tensor out) {
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous()
                && beta.is_contiguous() && g.is_contiguous(),
                "contiguous inputs required");
    TORCH_CHECK(q.scalar_type() == torch::kFloat && st.scalar_type() == torch::kFloat,
                "delta inputs and state are fp32");
    const int heads = (int)st.size(0);
    const int dk = (int)st.size(1);
    const int dv = (int)st.size(2);
    TORCH_CHECK(q.size(0) == heads && k.size(0) == heads && v.size(0) == heads,
                "q/k/v head count does not match the state");
    launch_rows(q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
                beta.data_ptr<float>(), g.data_ptr<float>(), st.data_ptr<float>(),
                out.data_ptr<float>(), 1, heads, dk, dv,
                c10::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// @brief 批处理门控 delta 步：每个批条目一行独立状态。
//
// 第 ``b`` 行和单独对那一行调 :func:`delta` 完全一致，一致性检查测的就是这个性质。各行
// 保持独立，因为每一行拿到自己的网格切片；根本不存在跨行同步。所有行只用一次启动：网格
// 的 ``z`` 轴就是批，所以以前每行一次启动的那些行现在合起来一次启动，并发在飞的 block
// 是 ``rows`` 倍。
//
// @param q,k [B, heads, dk] fp32。
// @param v [B, heads, dv] fp32。
// @param beta,g [B, heads] fp32。
// @param st [B, heads, dk, dv] fp32，原地更新。
// @param out [B, heads, dv] fp32，原地写入。
// @return ``out``，便于调用链串接。
// @throws c10::Error dtype 与布局矛盾。
torch::Tensor delta_batch(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                          torch::Tensor beta, torch::Tensor g, torch::Tensor st,
                          torch::Tensor out) {
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous()
                && beta.is_contiguous() && g.is_contiguous(),
                "contiguous inputs required");
    // 状态和输出也要检查，这不是为了对称而对称。内核的行偏移是 ``b * heads * dk * dv`` 和
    // ``b * heads * dv``，只有当批轴的步长恰好是这些值时它们才指向第 ``b`` 行。调用方传一个
    // 加宽的视图进来，每一行都会从错的地方读，而症状是批处理路径上一个很小的相对误差，
    // 可它的单行形式却是精确的。
    TORCH_CHECK(st.is_contiguous() && out.is_contiguous(),
                "state and output must be contiguous: the per-row stride assumes it");
    TORCH_CHECK(q.scalar_type() == torch::kFloat && st.scalar_type() == torch::kFloat,
                "delta inputs and state are fp32");
    const int rows = (int)st.size(0);
    const int heads = (int)st.size(1);
    const int dk = (int)st.size(2);
    const int dv = (int)st.size(3);
    launch_rows(q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
                beta.data_ptr<float>(), g.data_ptr<float>(), st.data_ptr<float>(),
                out.data_ptr<float>(), rows, heads, dk, dv,
                c10::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
