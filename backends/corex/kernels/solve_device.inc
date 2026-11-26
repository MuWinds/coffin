// @file solve_device.inc
// @brief 单位下三角求解的设备代码：``solve_kernel`` 与启动形状的两个纯主机辅助，
//        torch 版包装层与 C ABI shim 共用同一份。
//
// Module: coffin/backends/corex/kernels。``solve.cu``（torch 前端）与
// ``abi/solve_abi.cu``（C ABI 前端）各包含一次，所以两条前端是同一个函数、同一份
// 实例化，不是两份抄得像的实现。
//
// 为什么先搬这一个：它是 delta 规则唯一需要的**库**操作（cuSOLVER 在这份构建里
// 加载不了），入口只有一个、参数全是标量、没有块表也没有分组偏移——是把「求解族」
// 从 torch 里摘出来的最小一步，也是 ``ops/delta.py`` 每层每块都要走的那一条。
//
// 依赖：``corex_scalar.h`` 的 ``CxScalar``（必须在包含本文件之前引入）。

//: 一个块解多少条右端项列。64 是本部件的 wavefront，所以一个块的通道把 ``lower`` 的
//: 一行当成单次广播读下来，共享内存的步长也保持自然值。
constexpr int SOLVE_MAX_COLS = 64;

//: 动态共享内存上限。48 KB 是本部件的分界：低于它不需要额外的声明属性，所以待在它
//: 下面就能无条件启动。
constexpr size_t SOLVE_MAX_SMEM = 48u * 1024u;

// @brief 为单位下三角 ``A`` 解 ``A X = B``，每条右端项列一个线程。
//
// @param lower [bn, L, L] 系数；只读严格下三角部分，对角线按 1 处理。
// @param rhs [bn, L, n] 右端项。
// @param out [bn, L, n] 解，按 ``rhs`` 的 dtype 写出。
// @param L 系统阶数。
// @param n 右端项列数。
// @param ldb ``lower`` 的批步长，以元素计。
// @param ldr ``rhs``/``out`` 的批步长，以元素计。
// @param stride ``blockDim.x``：共享内存的行间距，即补齐后的线程数。
// @param cols 每块的有效列数。
template <typename T>
__global__ void solve_kernel(const T* __restrict__ lower, const T* __restrict__ rhs,
                             T* __restrict__ out, int L, int n, long ldb, long ldr,
                             int stride, int cols) {
    const int b = blockIdx.x;
    const int lane = (int)threadIdx.x;
    const int c = blockIdx.y * cols + lane;
    // 下面没有屏障，所以补齐出来的通道可以直接离开，不必等任何人。
    if (c >= n) return;

    extern __shared__ float xs[];       // [L][stride] 本列运行中的解
    const T* a = lower + (size_t)b * ldb;
    const T* rhs_b = rhs + (size_t)b * ldr;
    T* out_b = out + (size_t)b * ldr;

    for (int i = 0; i < L; ++i) {
        float acc = CxScalar<T>::to_float(rhs_b[(size_t)i * n + c]);
        const T* row = a + (size_t)i * L;
        // 第 ``i`` 行由每个通道在同一偏移上读取，所以这次加载经只读路径广播出去；共享
        // 内存的读取则是按通道的、合并的。
        for (int j = 0; j < i; ++j) {
            acc -= CxScalar<T>::to_float(row[j]) * xs[(size_t)j * stride + lane];
        }
        xs[(size_t)i * stride + lane] = acc;
        out_b[(size_t)i * n + c] = CxScalar<T>::from_float(acc);
    }
}

// @brief 由列数算启动形状：一个块的通道数必须是 wavefront 的整数倍。
//
// @param n 右端项列数。
// @param cols 输出，每块有效列数。
// @return ``blockDim.x``。
inline int solve_threads(int n, int& cols) {
    cols = n < SOLVE_MAX_COLS ? n : SOLVE_MAX_COLS;
    return ((cols + 63) / 64) * 64;
}

// @brief 由阶数与线程数算动态共享内存字节数。
//
// @param L 系统阶数。
// @param threads ``blockDim.x``。
// @return 字节数。
inline size_t solve_smem(int L, int threads) {
    return (size_t)L * (size_t)threads * sizeof(float);
}
