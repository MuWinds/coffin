// @file delta_normed_device.inc
// @brief 解码路径的融合链：头预处理、秩一递推、门控输出归一化，一次启动。
//
// Module: coffin/backends/corex/kernels。被两条前端各包含一次：``linear_attn.cu``
// （torch 扩展）与 ``abi/delta_normed_abi.cu``（``extern "C"`` 出口）。同一份设备代码。
//
// 为什么搬：``delta_normed`` 是解码每层每 token 的那一次线性注意力启动，它的算术必须与
// ``delta_prologue`` -> ``delta`` -> ``rms_norm_gated`` 那条链逐位相同。设备代码留在
// torch 扩展里，两条前端就各有一份，而漂移的表现是输出 id 翻一个 ulp——正是本轮要断掉
// 的东西之一。
//
// Constraints:
//   * 包含它的翻译单元必须已经包含 ``delta_prologue_device.inc``（``head_threads`` /
//     ``norm_threads`` / ``L2_EPS`` / ``SOFTPLUS_LINEAR`` 在那里定义）。
//   * ``dk`` 与 ``dv`` 要装得进同一个 block，且 ``norm_threads(dv) == head_threads(dk)``，
//     否则归约配对不同、答案会变（包装层负责检查）。
//   * ``out`` 的 dtype 与 ``z`` 的 dtype 是两个独立选择；四种实例化算同一套算术。

#ifndef COFFIN_DELTA_NORMED_DEVICE_INC
#define COFFIN_DELTA_NORMED_DEVICE_INC

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
// ``head_threads(dk)`` 个线程，``delta_kernel`` 跑 ``min(dv, THREADS) * DELTA_WAY``，
// ``rms_norm_gated_kernel`` 跑 ``rms_threads(dv)``——对这个模型（dk = dv = 128）分别是
// 128、512 和 128。共用这个形状，才让每次归约都保持那条链的配对，而这正是逐位相同与仅仅
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

#endif  // COFFIN_DELTA_NORMED_DEVICE_INC
