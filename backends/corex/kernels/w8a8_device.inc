// @file w8a8_device.inc
// @brief W8A8 族的激活量化设备代码：``quant_act_i8`` / ``gather_quant_i8``。
//
// Module: coffin/backends/corex/kernels。``w8a8.cu``（torch 前端）与
// ``abi/w8a8_abi.cu``（C 前端）各包含一次，两条前端的量化口径因此逐位相同。
//
// 两条内核的算术是 W8A8 的写方契约（读方是 epilogue 与 GEMM）：一行一个块，
// ``amax`` 只在 K 上归约、下限 ``1e-5``、``__float2int_rn`` 后夹到 ``[-127, 127]``。
// 换归约顺序或换平局规则都不会让形状检查失败，只会让整条预填充的 scale 偏移。
//
// 依赖：``corex_scalar.h`` 的 ``CxScalar``（须在包含本文件之前引入）。

//: 激活量化时每块线程数。一行一个块，归约只在 K 上做，所以这个数只影响单行的带宽。
constexpr int QUANT_THREADS = 256;

// @brief 按行（per-token）对称 int8 量化，一块一行，一次启动。
//
// 一行 = 一个 token，所以 amax 只在 K 上归约，不需要跨块通信。``1e-5`` 下限让全零
// 行量化成零而不是 inf。
//
// @param x [M, K] 激活。
// @param out_q [M, K] int8 目的地。
// @param out_s [M] 每行 fp32 scale 目的地。
// @param K 行宽。
template <typename XT>
__global__ void quant_act_i8_kernel(const XT* __restrict__ x, int8_t* __restrict__ out_q,
                                    float* __restrict__ out_s, int K) {
    const int row = blockIdx.x;
    const XT* xr = x + (size_t)row * K;
    int8_t* qr = out_q + (size_t)row * K;

    // 静态分配 256 个 float，对上 QUANT_THREADS；归约用这一块。
    __shared__ float smax[QUANT_THREADS];

    float amax = 0.f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        amax = fmaxf(amax, fabsf(CxScalar<XT>::to_float(xr[i])));
    }
    smax[threadIdx.x] = amax;
    __syncthreads();
    // 关闭屏障不可省：下一次归约（这里没有）或同一块里再进这个函数时，
    // ``smax[0] = v`` 的写必须晚于别人对它的读。
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smax[threadIdx.x] = fmaxf(smax[threadIdx.x], smax[threadIdx.x + s]);
        }
        __syncthreads();
    }
    const float sc = fmaxf(smax[0], 1e-5f) / 127.0f;
    const float inv = 1.0f / sc;
    if (threadIdx.x == 0) out_s[row] = sc;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        int q = __float2int_rn(CxScalar<XT>::to_float(xr[i]) * inv);
        q = max(-127, min(127, q));
        qr[i] = (int8_t)q;
    }
}

// @brief 按索引表搬一行、顺手量化：``index_select`` + ``quant_act_i8`` 合成一次启动。
//
// MoE 分派之后，收拢进分组投影的那些行来自 ``token_of_pos`` 指向的原行。原来那是两次
// 启动：先 ``index_select`` 把行搬进 ``[slots, K]`` 的 bf16 缓冲，再 ``quant_act_i8``
// 把它整条读一遍写成 int8 与 scale。两个内核搬的是同一批字节，而中间那份 bf16 缓冲
// 除了喂给量化没有第二个读者——一步 decode 里 40 层各一次，每次 8 行 x 4 KiB。
//
// 量化口径与 ``quant_act_i8`` **逐位相同**：amax 只在 K 上归约、下限 ``1e-5``、
// ``__float2int_rn`` 后截断到 ``[-127, 127]``。差别只有源行按索引表取，以及那一行进
// 共享内存——一行只有 ``K * sizeof(XT)`` 字节，第二趟（量化）因此不必回全局读。
//
// @param x [M, K] 源激活，bf16 或 fp32。
// @param index [rows] int32 行号表（分派写出的 ``token_of_pos``）。
// @param out_q [rows, K] int8 目的地。
// @param out_s [rows] 每行 fp32 scale 目的地。
// @param K 行宽。
template <typename XT>
__global__ void gather_quant_i8_kernel(const XT* __restrict__ x,
                                       const int* __restrict__ index,
                                       int8_t* __restrict__ out_q,
                                       float* __restrict__ out_s, int K) {
    const int row = blockIdx.x;
    const XT* src = x + (size_t)index[row] * K;
    int8_t* qr = out_q + (size_t)row * K;

    __shared__ float smax[QUANT_THREADS];
    extern __shared__ int8_t sraw[];
    XT* srow = reinterpret_cast<XT*>(sraw);
    for (int i = (int)threadIdx.x; i < K; i += (int)blockDim.x) srow[i] = src[i];
    __syncthreads();

    float amax = 0.f;
    for (int i = (int)threadIdx.x; i < K; i += (int)blockDim.x) {
        amax = fmaxf(amax, fabsf(CxScalar<XT>::to_float(srow[i])));
    }
    smax[threadIdx.x] = amax;
    __syncthreads();
    // 关闭屏障不可省：下一次归约或同一块里再进同一个内核时，``smax[0] = v`` 的写
    // 必须晚于别人对它的读。
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smax[threadIdx.x] = fmaxf(smax[threadIdx.x], smax[threadIdx.x + s]);
        }
        __syncthreads();
    }
    const float sc = fmaxf(smax[0], 1e-5f) / 127.0f;
    const float inv = 1.0f / sc;
    if (threadIdx.x == 0) out_s[row] = sc;
    for (int i = (int)threadIdx.x; i < K; i += (int)blockDim.x) {
        int q = __float2int_rn(CxScalar<XT>::to_float(srow[i]) * inv);
        q = max(-127, min(127, q));
        qr[i] = (int8_t)q;
    }
}
