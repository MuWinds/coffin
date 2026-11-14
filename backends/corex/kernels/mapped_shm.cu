// @file mapped_shm.cu
// @brief 零拷贝（mapped）锁页内存的可用性探针，以及跨 rank 归约的设备侧原语。
//
// Module: coffin/backends/corex/kernels。
//
// 为什么需要：四卡 decode 的跨 rank 归约当前走主机侧交换（共享板或 gloo），
// 每层一次**阻塞 D2H**，主机必须停下等本层算完。实测（作业 416/420/421）：
//   * 四卡 decode 142.51 ms/步，单卡 89.15；combine 短路后 104.53
//     ⇒ combine 的端到端真实代价 **37.98 ms（26.6%）**；
//   * 换成 gloo 后 135.76 ms（−4.3%），但仍有 31.0 ms（22.9%）是阻塞造成的。
// 机制是 `pre_us`（870 µs/层，等本层设备算完）× 48 = 41.8 ms，与 37.98 吻合到 10%。
//
// 本文件先解决**前提问题**：这张卡上的锁页内存能否被设备直接寻址
// （`cudaHostAlloc(cudaHostAllocMapped)` + `cudaHostGetDevicePointer`）。
// 只有能，才谈得上「核直接写锁页缓冲 + 主机只轮询标志」的设备侧归约；
// 不能的话 31 ms 就是主机侧交换架构的固定代价，得换设计（如每 rank 算全部专家）。
//
// Constraints:
//   * `cudaHostGetDevicePointer` 在 `cudaHostAllocMapped` 未生效的设备上返回错误或
//     给出不可写地址。两者都必须**显式检查**，不能假设成功——否则核写出去的是野地址，
//     表现为静默错数或崩卡。
//   * 探针分配的内存要显式释放，且一个进程只调一次（`torch.utils.cpp_extension`
//     的静态初始化顺序不保证，所以不做全局对象）。

#include "corex_bindings.h"
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <tuple>
#include <mutex>

namespace {

//: 探针推进去的魔数，主机侧据此判断「核真的写到了这块锁页内存」。
constexpr uint32_t kMagic = 0xC0FFEE5Au;

__global__ void write_magic_kernel(uint32_t* dst) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // volatile：宿主稍后从 CPU 读同一块物理内存，编译器不能把它优化掉。
        *reinterpret_cast<volatile uint32_t*>(dst) = kMagic;
    }
}

}  // namespace

// @brief 探测本设备是否支持映射锁页内存，并返回一次真机往返的结果。
//
// 步骤：分配 mapped 锁页内存 -> 取设备指针 -> 核对两个指针确实不同（映射生效时
// 设备指针与主机指针在不同地址空间）-> 起核写魔数 -> `cudaDeviceSynchronize` ->
// 主机读回。**这次同步是探针自身的需要**（要确定核已执行完），不代表归约路径也要同步。
//
// @return ``1`` 支持且往返成功；``0`` 不支持或往返失败（此时 ``detail`` 说明原因）。
int mapped_pinned_probe(char* detail, int detail_len) {
    static std::mutex mu;
    std::lock_guard<std::mutex> lock(mu);

    auto fail = [&](const char* why, cudaError_t err) {
        std::snprintf(detail, static_cast<size_t>(detail_len), "%s: %s", why,
                      cudaGetErrorString(err));
        return 0;
    };

    // 先把上一个错误清掉，否则下面的检查会被历史错误污染。
    cudaGetLastError();

    void* host = nullptr;
    cudaError_t err = cudaHostAlloc(&host, sizeof(uint32_t), cudaHostAllocMapped);
    if (err != cudaSuccess) {
        return fail("cudaHostAlloc(mapped) failed", err);
    }

    void* dev = nullptr;
    err = cudaHostGetDevicePointer(&dev, host, 0);
    if (err != cudaSuccess) {
        cudaFreeHost(host);
        return fail("cudaHostGetDevicePointer failed", err);
    }

    *static_cast<uint32_t*>(host) = 0u;
    write_magic_kernel<<<1, 1>>>(static_cast<uint32_t*>(dev));
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFreeHost(host);
        return fail("write_magic_kernel launch failed", err);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFreeHost(host);
        return fail("write_magic_kernel execution failed", err);
    }

    const uint32_t seen = *static_cast<uint32_t*>(host);
    cudaFreeHost(host);

    if (seen != kMagic) {
        std::snprintf(detail, static_cast<size_t>(detail_len),
                      "magic mismatch: host saw 0x%08X, expected 0x%08X", seen, kMagic);
        return 0;
    }
    std::snprintf(detail, static_cast<size_t>(detail_len),
                  "ok: dev=%p differs from host, magic 0x%08X", dev, seen);
    return 1;
}

// 绑定写在内核旁边：这条要把 ``char*`` 缓冲折成 ``(int, 结论文本)``，所以绑定体是一个
// lambda，用 ``COFFIN_BIND_LAMBDA``（它不要求可调用体与算子同名）。
COFFIN_BIND_LAMBDA(mapped_pinned_probe,
                   [](int64_t detail_len) {
                       std::string detail(static_cast<size_t>(detail_len > 0 ? detail_len : 256), 0x00);
                       const int ok = mapped_pinned_probe(
                           detail.data(), static_cast<int>(detail.size()));
                       detail.resize(std::strlen(detail.c_str()));
                       return std::make_tuple(ok, detail);
                   },
                   "探测映射锁页内存是否可用："
                   "返回 (1/0, 结论文本)",
                   pybind11::arg("detail_len") = 256);
