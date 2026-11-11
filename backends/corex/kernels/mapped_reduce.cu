// @file mapped_reduce.cu
// @brief 设备侧跨 rank 归约：核直接读写一块被多进程共享、且被 CUDA 映射过的锁页区。
//
// Module: coffin/backends/corex/kernels。与 ``mapped_shm.cu``（只放探针）分开成两个
// 翻译单元是**为了绕过一个编译器内部崩溃**：把「裸主机指针 + torch 张量 + 设备核」
// 混在同一个 .cu 里时 ``clang-16`` 直接 ICE，拆开之后每个单元只碰一类东西。
//
// 与主机侧交换的根本区别：主机不调任何同步 API，只轮询锁页区里的到达计数。
// 主机侧交换每层要阻塞一次 D2H 等本层算完，队列排空、流水线断裂；实测 combine
// 端到端代价 37.98 ms，换 gloo 后仍有 31.0 ms。
//
// 内存布局由 Python 侧决定，本文件只按参数寻址：
//   * 前 ``world * rows * hidden`` 个 fp32 是各 rank 的槽；
//   * 紧随其后 ``world`` 个 int64 是到达计数（与主机侧同一份物理页）。
//
// Constraints:
//   * ``mmap`` 必须先建好共享，再让**每个进程各自** ``cudaHostRegister``——
//     ``cudaHostAlloc`` 的内存无法跨进程映射，用它就拿不到同一份物理页。
//   * 到达计数不能用 ``atomicAdd``：实测它写映射系统内存不生效（计数恒为 0）。
//     只有一个线程写、且写的是递增轮次值，普通 ``volatile`` 存储即可。

#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

namespace {

// @brief 把一个 rank 的局部和写进它自己的槽，随后置到达计数。
//
// bf16 按位展开成 float 再写：跨 rank 求和要可加，bf16 累加四个 rank 会在末两位掉精度。
__global__ void stage_kernel(const void* __restrict__ x, float* __restrict__ slots,
                             long long* __restrict__ counts, int rows, int hidden,
                             int rank, long long round) {
    const long long total = static_cast<long long>(rows) * hidden;
    float* dst = slots + static_cast<long long>(rank) * total;
    for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
         i < total; i += static_cast<long long>(gridDim.x) * blockDim.x) {
        const unsigned short bits = reinterpret_cast<const unsigned short*>(x)[i];
        const unsigned int u = static_cast<unsigned int>(bits) << 16;
        float v = 0.0F;
        __builtin_memcpy(&v, &u, sizeof(v));
        dst[i] = v;
    }
    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // 必须先让槽的写对其他 rank 可见，再置计数。本核**只能单 block 启动**，
        // 否则 ``blockIdx.x == 0`` 之外的 block 还没写完就已置计数，等待方会读到半截数据。
        //
        // 用 ``__threadfence()`` 而不是 ``__threadfence_system()``：后者在这个 arch 上
        // 直接让 ``clang-16`` 内部崩溃（ICE，``diagnostic msg: /tmp/mapped_reduce-*.cu``，
        // 没有任何 error 诊断行）。实测 ``__threadfence()`` 足够——作业 431 里四个槽
        // 都写到了映射区、主机读回的计时器是 ``[1,1,1,1]``、求和 ``MR_MAXERR 0.0``。
        // 若将来跨进程出现半截数据，先怀疑这里的作用域不够，而不是等待协议写错。
        __threadfence();
        *reinterpret_cast<volatile long long*>(&counts[rank]) = round;
    }
}

// @brief 从所有 rank 的槽求和，写到设备上的 ``out``（fp32）。
__global__ void finish_kernel(float* __restrict__ out, const float* __restrict__ slots,
                              int rows, int hidden, int world) {
    const long long total = static_cast<long long>(rows) * hidden;
    for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
         i < total; i += static_cast<long long>(gridDim.x) * blockDim.x) {
        float acc = 0.0F;
        for (int r = 0; r < world; ++r) {
            acc += slots[static_cast<long long>(r) * total + i];
        }
        out[i] = acc;
    }
}

// @brief 先等所有 rank 到达、再求和：把等待放在设备上，主机一次都不参与。
//
// 为什么需要它：主机侧轮询有两个都实测过的坏结局。纯忙等要被**饿 20 ms**
// （``[1, 2048]`` 一次归约 20055 us，而同一段代码先同步再轮询只要 106 us：这个
// 运行时把 kernel 下发压在 flush 点上，主机一直不放手它就不放 kernel 出去）；
// 改成「先同步再轮询」虽然降到 106 us，但每层一次 ``cudaDeviceSynchronize`` 会把
// 已经排好的下发队列清空——EP4 一步 40 层，实测因此多付约 11 ms。
//
// 自旋放在核里就没有这两个问题：主机只负责下发 stage 与 finish，两个核在同一个流上
// 背靠背执行，finish 自己等对端写完再读。每个 block 各自等一次（线程 0 自旋 +
// ``__syncthreads``），所以多 block 也不需要跨块同步。
//
// @param out      ``[rows, hidden]`` fp32 设备张量。
// @param slots    ``world * rows * hidden`` 的共享区（设备可见）。
// @param counts   紧随其后的 ``world`` 个 int64 到达计数。
// @param round    本轮轮次值；等 ``counts[r] >= round``。
__global__ void finish_wait_kernel(float* __restrict__ out,
                                   const float* __restrict__ slots,
                                   const long long* __restrict__ counts,
                                   int rows, int hidden, int world, long long round) {
    if (threadIdx.x == 0) {
        for (int r = 0; r < world; ++r) {
            // volatile：这份计数由别的进程写，编译器不能把它提到循环外。
            //
            // 不要在这里加 ``__threadfence_system()``：实测加了之后 ptxas 直接
            // 崩（``clang-16: unable to execute command: Aborted``），与文件头记的
            // 「同一翻译单元里混裸主机指针 + torch 张量 + 设备核会 ICE」是同一个病。
            // 要修「读不到对端写」只能把自旋搬进独立的翻译单元，而不是在这里加栅栏。
            while (reinterpret_cast<const volatile long long*>(counts)[r] < round) {
            }
        }
    }
    __syncthreads();
    const long long total = static_cast<long long>(rows) * hidden;
    for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
         i < total; i += static_cast<long long>(gridDim.x) * blockDim.x) {
        float acc = 0.0F;
        for (int r = 0; r < world; ++r) {
            acc += slots[static_cast<long long>(r) * total + i];
        }
        out[i] = acc;
    }
}

// @brief 记一次注册，供 ``mapped_release`` 解注册。
long long g_registered_host = 0;

}  // namespace

// @brief 把一块主机 mmap 区域注册成 CUDA 可映射锁页内存，返回设备指针。
//
// @param host_ptr 共享区的主机虚拟地址（各进程可以不同）。
// @param bytes    共享区字节数。
// @param detail   结论文本输出（失败原因，或成功时的地址说明）。
// @param detail_len ``detail`` 字节数。
// @return 设备指针；``0`` 表示失败。
int64_t mapped_register(void* host_ptr, int64_t bytes, char* detail, int detail_len) {
    auto fail = [&](const char* why, cudaError_t err) -> int64_t {
        std::snprintf(detail, static_cast<size_t>(detail_len), "%s: %s", why,
                      cudaGetErrorString(err));
        return 0;
    };
    cudaGetLastError();
    cudaError_t err = cudaHostRegister(host_ptr, static_cast<size_t>(bytes),
                                       cudaHostRegisterMapped);
    if (err != cudaSuccess) {
        return fail("cudaHostRegister(mapped) failed", err);
    }
    void* dev = nullptr;
    err = cudaHostGetDevicePointer(&dev, host_ptr, 0);
    if (err != cudaSuccess) {
        cudaHostUnregister(host_ptr);
        return fail("cudaHostGetDevicePointer failed", err);
    }
    g_registered_host = static_cast<long long>(reinterpret_cast<uintptr_t>(host_ptr));
    std::snprintf(detail, static_cast<size_t>(detail_len), "ok: dev=%p host=%p",
                  dev, host_ptr);
    return static_cast<int64_t>(reinterpret_cast<uintptr_t>(dev));
}

// @brief 解注册 ``mapped_register`` 建过的映射。
//
// 必须在 munmap 之前调用：先拆映射再解注册会让驱动指向已经不存在的页。
//
// @param host_ptr ``mapped_register`` 收到的同一个地址。
// @return ``0`` 成功，负值为 CUDA 错误码。
int mapped_release(int64_t host_ptr) {
    if (host_ptr == 0 || host_ptr != g_registered_host) {
        return -1;
    }
    const cudaError_t err = cudaHostUnregister(
        reinterpret_cast<void*>(static_cast<uintptr_t>(host_ptr)));
    g_registered_host = 0;
    return err == cudaSuccess ? 0 : -static_cast<int>(err);
}

// @brief 起 ``stage_kernel``：本 rank 的局部和写进槽，并置到达计数。
//
// **只能单 block 启动**，理由见 ``stage_kernel`` 里的注释。
//
// @param x        ``[rows, hidden]`` bf16 设备张量。
// @param dev_base 共享区的设备指针（来自 ``mapped_register``）。
// @param world    进程数。
// @param rank     本进程下标。
// @param threads  单 block 的线程数。
// @param round    本轮的轮次值；等待方比较 ``>= round``。
void mapped_stage(torch::Tensor x, int64_t dev_base, int64_t world, int64_t rank,
                  int64_t threads, int64_t round) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 2,
                "x must be a contiguous 2-D CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bf16");
    TORCH_CHECK(rank >= 0 && rank < world, "rank out of range");
    TORCH_CHECK(threads > 0 && threads <= 1024, "threads out of range");
    const int rows = static_cast<int>(x.size(0));
    const int hidden = static_cast<int>(x.size(1));
    float* slots = reinterpret_cast<float*>(static_cast<uintptr_t>(dev_base));
    long long* counts = reinterpret_cast<long long*>(
        slots + static_cast<long long>(world) * rows * hidden);
    stage_kernel<<<1, static_cast<int>(threads)>>>(
        x.data_ptr(), slots, counts, rows, hidden, static_cast<int>(rank),
        static_cast<long long>(round));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// @brief 起 ``finish_kernel``：所有槽求和到设备 ``out``。
//
// @param out      ``[rows, hidden]`` fp32 设备张量。
// @param dev_base 共享区的设备指针。
// @param world    进程数。
// @param blocks   网格块数。
// @param threads  每块线程数。
void mapped_finish(torch::Tensor out, int64_t dev_base, int64_t world, int64_t blocks,
                   int64_t threads) {
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.dim() == 2,
                "out must be a contiguous 2-D CUDA tensor");
    TORCH_CHECK(out.scalar_type() == torch::kFloat32, "out must be fp32");
    TORCH_CHECK(blocks > 0 && threads > 0 && threads <= 1024, "grid out of range");
    const int rows = static_cast<int>(out.size(0));
    const int hidden = static_cast<int>(out.size(1));
    const float* slots = reinterpret_cast<const float*>(static_cast<uintptr_t>(dev_base));
    finish_kernel<<<static_cast<int>(blocks), static_cast<int>(threads)>>>(
        out.data_ptr<float>(), slots, rows, hidden, static_cast<int>(world));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// @brief 起 ``finish_wait_kernel``：设备侧等所有 rank 到达，再求和到设备 ``out``。
//
// 主机调用它之后**不需要任何同步或轮询**：等待在核里做，所以这一步不会清空下发队列。
//
// @param out      ``[rows, hidden]`` fp32 设备张量。
// @param dev_base 共享区的设备指针。
// @param world    进程数。
// @param blocks   网格块数。
// @param threads  每块线程数。
// @param round    本轮轮次值，核等 ``counts[r] >= round``。
void mapped_finish_wait(torch::Tensor out, int64_t dev_base, int64_t world,
                        int64_t blocks, int64_t threads, int64_t round) {
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.dim() == 2,
                "out must be a contiguous 2-D CUDA tensor");
    TORCH_CHECK(out.scalar_type() == torch::kFloat32, "out must be fp32");
    TORCH_CHECK(blocks > 0 && threads > 0 && threads <= 1024, "grid out of range");
    TORCH_CHECK(round > 0, "round must be positive");
    const int rows = static_cast<int>(out.size(0));
    const int hidden = static_cast<int>(out.size(1));
    const float* slots = reinterpret_cast<const float*>(static_cast<uintptr_t>(dev_base));
    const long long* counts = reinterpret_cast<const long long*>(
        slots + static_cast<long long>(world) * rows * hidden);
    finish_wait_kernel<<<static_cast<int>(blocks), static_cast<int>(threads)>>>(
        out.data_ptr<float>(), slots, counts, rows, hidden, static_cast<int>(world),
        static_cast<long long>(round));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


