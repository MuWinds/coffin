// @file p2p_reduce_abi.cu
// @brief 四卡设备内存 all-to-all 归约的 C ABI 出口，无 torch。
//
// Module: coffin/backends/corex/kernels/abi。由 ``cxrt/p2p.py`` 经 ctypes 装载，供
// TP4 与 EP4 共用：一个核完成「可选加共享项 → 写四卡对端槽 → 到达会合 → fp32 累加 →
// bf16 写回」，主机全程不参与、不排空流。
//
// 相对历史 torch 扩展版 ``p2p_reduce.cu`` 的三处结构修复：
//   1. 奇偶双缓冲槽区（``buf = round & 1``）：快 rank 写下一轮不会踩慢 rank 仍在读的
//      上一轮（见 ``docs/DECODE-LAUNCH-LEDGER.md`` §11.8 竞态定位）。
//   2. 多块安全发布（``gridDim.x <= 16``，每缓冲一个本地原子到达计数）：预填充多块时
//      只有最后一个写完对端槽的块才发布 flag，消除 block 0 提前放行的数据竞态；16 块
//      × 256 线程在 16 SM 的 BI-V100 上全部驻留，不会因驻留不足自旋死锁。
//   3. 槽区存 bf16（2B）并用 ``uint4`` 16 字节向量搬运，接收端在寄存器转 fp32 求和后
//      直接写 bf16 输出：与 fp32 槽区数值逐位相同，PCIe 跨卡流量减半、双缓冲显存
//      与原单缓冲 fp32 持平，并省去每层一次 ``out.to(bf16)`` 发射。
//
// Constraints:
//   * 只允许设备作用域栅栏 ``__threadfence()``：``__threadfence_system()`` 在当前
//     clang-16/ivcore10 上会触发 llc 崩溃。
//   * 固定 4 卡（``COFFIN_P2P_WORLD == 4``），且 ``blocks <= 16`` 保证全网格常驻。
//   * 签名与 ``cxrt/p2p.py`` 的 ``argtypes`` 必须逐字一致。

#include "cx_tensor.h"
#include "corex_scalar.h"

#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

#define COFFIN_P2P_WORLD 4
#define COFFIN_P2P_MAX_BLOCKS 16
#define COFFIN_P2P_SPIN_LIMIT 100000000LL
#define COFFIN_P2P_ABORT_SLOT COFFIN_P2P_WORLD
#define COFFIN_P2P_BLK_SLOT (COFFIN_P2P_WORLD + 1)

namespace {

// @brief 四卡 bf16 槽区 all-to-all 归约核（fp32 累加、直接写 bf16，可选融合 ``add``）。
//
// @param x          本 rank 贡献，``[total]`` bf16（uint16_t），连续。
// @param add        可选加数，``[total]`` bf16；非空时先做 ``bf16(x + add)`` 再写槽，
//                   用于把 MoE 的 ``experts + shared`` 合并进同一次归约发射。
// @param slots0..3  四卡各自的 ``[2, 4, max_elems]`` bf16 槽区首址。
// @param flags0..3  四卡各自的 ``[world + 2]`` int64 状态区首址。
// @param out        输出，``[total]`` bf16，连续。
// @param buf_stride 单半槽区元素数（``4 * max_elems``），保证奇偶半区严格不相交。
// @param total      本次元素数（``rows * hidden``）。
// @param rank       本进程编号（0..3）。
// @param round      全局单调轮次号（>= 1）。
__global__ void p2p_combine_bf16_kernel(const uint16_t* __restrict__ x,
                                        const uint16_t* __restrict__ add,
                                        uint16_t* slots0, uint16_t* slots1,
                                        uint16_t* slots2, uint16_t* slots3,
                                        long long* flags0, long long* flags1,
                                        long long* flags2, long long* flags3,
                                        uint16_t* __restrict__ out,
                                        long long buf_stride, long long total,
                                        int rank, long long round) {
    uint16_t* base[COFFIN_P2P_WORLD] = {slots0, slots1, slots2, slots3};
    long long* flg[COFFIN_P2P_WORLD] = {flags0, flags1, flags2, flags3};
    const int buf = (int)(round & 1LL);
    const long long boff = (long long)buf * buf_stride;
    const long long stride = (long long)gridDim.x * (long long)blockDim.x;
    const long long start = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;

    __shared__ int failed;
    if (threadIdx.x == 0) {
        failed = (flg[rank][COFFIN_P2P_ABORT_SLOT] != 0) ? 1 : 0;
    }
    __syncthreads();
    if (failed) {
        for (long long i = start; i < total; i += stride) out[i] = 0u;
        return;
    }

    // 载荷写进四个对端的本轮半区第 ``rank`` 段：按 4 字节（2 个 bf16，``uint32_t``）
    // 走 ``volatile`` 写穿，避开 ivcore10 的 16B ``uint4`` 惩罚并防止本地 L1 缓存陈旧行。
    const bool pair = ((total & 1LL) == 0LL) &&
                      ((reinterpret_cast<uintptr_t>(x) & 3u) == 0u) &&
                      (add == nullptr || (reinterpret_cast<uintptr_t>(add) & 3u) == 0u) &&
                      ((reinterpret_cast<uintptr_t>(out) & 3u) == 0u);
    if (pair) {
        const long long pairs = total >> 1;
        const uint32_t* xv = reinterpret_cast<const uint32_t*>(x);
        const uint32_t* av = reinterpret_cast<const uint32_t*>(add);
        for (long long i = start; i < pairs; i += stride) {
            uint32_t val = xv[i];
            if (av != nullptr) {
                const uint32_t b = av[i];
                const uint16_t x0 = (uint16_t)(val & 0xffffu);
                const uint16_t x1 = (uint16_t)(val >> 16);
                const uint16_t b0 = (uint16_t)(b & 0xffffu);
                const uint16_t b1 = (uint16_t)(b >> 16);
                const uint16_t o0 = CxScalar<uint16_t>::from_float(
                    CxScalar<uint16_t>::to_float(x0) + CxScalar<uint16_t>::to_float(b0));
                const uint16_t o1 = CxScalar<uint16_t>::from_float(
                    CxScalar<uint16_t>::to_float(x1) + CxScalar<uint16_t>::to_float(b1));
                val = (uint32_t)o0 | ((uint32_t)o1 << 16);
            }
#pragma unroll
            for (int p = 0; p < COFFIN_P2P_WORLD; ++p) {
                volatile uint32_t* dst = reinterpret_cast<volatile uint32_t*>(
                    base[p] + boff + (long long)rank * total);
                dst[i] = val;
            }
        }
    } else {
        for (long long i = start; i < total; i += stride) {
            uint16_t val = x[i];
            if (add != nullptr) {
                val = CxScalar<uint16_t>::from_float(
                    CxScalar<uint16_t>::to_float(val) +
                    CxScalar<uint16_t>::to_float(add[i]));
            }
#pragma unroll
            for (int p = 0; p < COFFIN_P2P_WORLD; ++p) {
                volatile uint16_t* dst = reinterpret_cast<volatile uint16_t*>(
                    base[p] + boff + (long long)rank * total);
                dst[i] = val;
            }
        }
    }

    // 载荷对本地设备出队后，再向每个对端做一次 4B volatile 回读作为 PCIe 强序刷新
    // （PCIe Spec §2.4：non-posted read 必须把此前所有 posted write 推到对端显存后才返回），
    // 弥补 ivcore10 缺少 __threadfence_system() 导致的偶发 flag 越序（~0.1%/轮）。
    __threadfence();
    __syncthreads();

    __shared__ int do_pub;
    if (gridDim.x == 1) {
        if (threadIdx.x == 0) do_pub = 1;
    } else {
        if (threadIdx.x == 0) {
            unsigned int* bcnt =
                reinterpret_cast<unsigned int*>(&flg[rank][COFFIN_P2P_BLK_SLOT]) + buf;
            const unsigned int done = atomicAdd(bcnt, 1u) + 1u;
            if (done == (unsigned int)gridDim.x) {
                *bcnt = 0u;
                do_pub = 1;
            } else {
                do_pub = 0;
            }
        }
    }
    __syncthreads();

    if (do_pub) {
        if (threadIdx.x < COFFIN_P2P_WORLD) {
            const int p = (int)threadIdx.x;
            long long pub = round;
            if (p != rank) {
                const volatile uint16_t* tail = reinterpret_cast<const volatile uint16_t*>(
                    base[p] + boff + (long long)rank * total + (total - 1));
                const uint16_t ack = *tail;
                // 运行期恒假条件制造真数据依赖：硬件必须等对端 PCIe 回读完成包返回，
                // 才会发射紧随其后的 ``dst_flg[rank] = pub`` 写。
                if (ack == 0xffffu && round < 0) pub = -1;
            }
            volatile long long* dst_flg = flg[p];
            dst_flg[rank] = pub;
        }
        __threadfence();
    }

    // 自旋等四卡到齐；每次空转带 ``__threadfence()``，保证看到对端最新载荷。
    if (threadIdx.x == 0) {
        volatile long long* mine = flg[rank];
        long long spins = 0;
        int bad = 0;
        for (int r = 0; r < COFFIN_P2P_WORLD; ++r) {
            while (mine[r] < round) {
                if (mine[COFFIN_P2P_ABORT_SLOT] != 0) {
                    bad = 1;
                    break;
                }
                __threadfence();
                if (++spins > COFFIN_P2P_SPIN_LIMIT) {
                    bad = 1;
                    break;
                }
            }
            if (bad) break;
        }
        if (bad) {
            for (int p = 0; p < COFFIN_P2P_WORLD; ++p) {
                volatile long long* dst_flg = flg[p];
                dst_flg[COFFIN_P2P_ABORT_SLOT] = round;
            }
            failed = 1;
        } else {
            failed = 0;
        }
    }
    // 无论自旋循环是否实际空转，统一补一次设备栅栏再读取对端写入的载荷。
    __threadfence();
    __syncthreads();
    if (failed) {
        for (long long i = start; i < total; i += stride) out[i] = 0u;
        return;
    }

    // 在本卡显存以 ``volatile`` 绕过 L1 缓存读取四卡载荷，fp32 累加、窄化一次写回 bf16。
    const uint16_t* my = base[rank] + boff;
    if (pair) {
        const long long pairs = total >> 1;
        const volatile uint32_t* s0 = reinterpret_cast<const volatile uint32_t*>(my);
        const volatile uint32_t* s1 = reinterpret_cast<const volatile uint32_t*>(my + total);
        const volatile uint32_t* s2 = reinterpret_cast<const volatile uint32_t*>(my + 2 * total);
        const volatile uint32_t* s3 = reinterpret_cast<const volatile uint32_t*>(my + 3 * total);
        uint32_t* ov = reinterpret_cast<uint32_t*>(out);
        for (long long i = start; i < pairs; i += stride) {
            const uint32_t v0 = s0[i];
            const uint32_t v1 = s1[i];
            const uint32_t v2 = s2[i];
            const uint32_t v3 = s3[i];
            const float sum0 = CxScalar<uint16_t>::to_float((uint16_t)(v0 & 0xffffu)) +
                               CxScalar<uint16_t>::to_float((uint16_t)(v1 & 0xffffu)) +
                               CxScalar<uint16_t>::to_float((uint16_t)(v2 & 0xffffu)) +
                               CxScalar<uint16_t>::to_float((uint16_t)(v3 & 0xffffu));
            const float sum1 = CxScalar<uint16_t>::to_float((uint16_t)(v0 >> 16)) +
                               CxScalar<uint16_t>::to_float((uint16_t)(v1 >> 16)) +
                               CxScalar<uint16_t>::to_float((uint16_t)(v2 >> 16)) +
                               CxScalar<uint16_t>::to_float((uint16_t)(v3 >> 16));
            const uint16_t r0 = CxScalar<uint16_t>::from_float(sum0);
            const uint16_t r1 = CxScalar<uint16_t>::from_float(sum1);
            ov[i] = (uint32_t)r0 | ((uint32_t)r1 << 16);
        }
    } else {
        const volatile uint16_t* myv = reinterpret_cast<const volatile uint16_t*>(my);
        for (long long i = start; i < total; i += stride) {
            float acc = 0.f;
#pragma unroll
            for (int r = 0; r < COFFIN_P2P_WORLD; ++r) {
                acc += CxScalar<uint16_t>::to_float(myv[(long long)r * total + i]);
            }
            out[i] = CxScalar<uint16_t>::from_float(acc);
        }
    }
}

}  // namespace

// @brief 直接用 ``cudaMalloc`` 分配并清零一块独立设备缓冲，保证指针位于分配段基址。
//
// 为什么不借 PyTorch 张量：PyTorch 缓存分配器会把小张量（如 48 B 的 ``flags``）切在
// 2 MB 段的非零偏移处；``cudaIpcGetMemHandle``/``OpenMemHandle`` 在本卡驱动上打开的是
// 段基址而不是子偏移，对端写 flag 会落进别的权重内存，导致首轮会合永远等不到。
extern "C" int coffin_p2p_alloc(unsigned long long nbytes, uint64_t* out_ptr,
                                char* err, int errlen) {
    const cx::Err e{err, errlen};
    if (!out_ptr || nbytes == 0) {
        return cx::fail(e, "out_ptr must be non-null and nbytes > 0");
    }
    void* p = nullptr;
    cudaError_t rc = cudaMalloc(&p, static_cast<size_t>(nbytes));
    if (rc != cudaSuccess) {
        return cx::fail(e, "cudaMalloc(%llu) failed: %s (%d)",
                        nbytes, cudaGetErrorString(rc), (int)rc);
    }
    rc = cudaMemset(p, 0, static_cast<size_t>(nbytes));
    if (rc != cudaSuccess) {
        cudaFree(p);
        return cx::fail(e, "cudaMemset failed: %s (%d)",
                        cudaGetErrorString(rc), (int)rc);
    }
    rc = cudaDeviceSynchronize();
    if (rc != cudaSuccess) {
        cudaFree(p);
        return cx::fail(e, "cudaDeviceSynchronize failed: %s (%d)",
                        cudaGetErrorString(rc), (int)rc);
    }
    *out_ptr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p));
    return 0;
}

// @brief 释放 :func:`coffin_p2p_alloc` 分配的独立设备缓冲。
extern "C" int coffin_p2p_free(uint64_t ptr, char* err, int errlen) {
    const cx::Err e{err, errlen};
    if (!ptr) return 0;
    const cudaError_t rc = cudaFree(reinterpret_cast<void*>(static_cast<uintptr_t>(ptr)));
    if (rc != cudaSuccess) {
        return cx::fail(e, "cudaFree failed: %s (%d)",
                        cudaGetErrorString(rc), (int)rc);
    }
    return 0;
}

// @brief 读回本卡 ``flags[COFFIN_P2P_ABORT_SLOT]`` 中止锁存值。
extern "C" int coffin_p2p_read_abort(uint64_t flag_ptr, long long* out_val,
                                     char* err, int errlen) {
    const cx::Err e{err, errlen};
    if (!flag_ptr || !out_val) {
        return cx::fail(e, "flag_ptr and out_val must be non-null");
    }
    const long long* dev = reinterpret_cast<const long long*>(static_cast<uintptr_t>(flag_ptr)) +
                           COFFIN_P2P_ABORT_SLOT;
    long long val = 0;
    const cudaError_t rc = cudaMemcpy(&val, dev, sizeof(val), cudaMemcpyDeviceToHost);
    if (rc != cudaSuccess) {
        return cx::fail(e, "cudaMemcpy abort slot failed: %s (%d)",
                        cudaGetErrorString(rc), (int)rc);
    }
    *out_val = val;
    return 0;
}

// @brief 取一块设备缓冲的 64 字节 IPC 句柄。
extern "C" int coffin_p2p_get_handle(const void* dev_ptr, void* out_handle64,
                                     char* err, int errlen) {
    const cx::Err e{err, errlen};
    if (!dev_ptr || !out_handle64) {
        return cx::fail(e, "dev_ptr and out_handle64 must be non-null");
    }
    cudaIpcMemHandle_t handle;
    const cudaError_t rc = cudaIpcGetMemHandle(&handle, const_cast<void*>(dev_ptr));
    if (rc != cudaSuccess) {
        return cx::fail(e, "cudaIpcGetMemHandle failed: %s (%d)",
                        cudaGetErrorString(rc), (int)rc);
    }
    std::memcpy(out_handle64, &handle, sizeof(handle));
    return 0;
}

// @brief 打开对端 IPC 句柄，返回本进程可寻址的设备指针。
extern "C" int coffin_p2p_open_handle(const void* handle64, uint64_t* out_ptr,
                                      char* err, int errlen) {
    const cx::Err e{err, errlen};
    if (!handle64 || !out_ptr) {
        return cx::fail(e, "handle64 and out_ptr must be non-null");
    }
    cudaIpcMemHandle_t handle;
    std::memcpy(&handle, handle64, sizeof(handle));
    void* peer = nullptr;
    const cudaError_t rc = cudaIpcOpenMemHandle(&peer, handle, cudaIpcMemLazyEnablePeerAccess);
    if (rc != cudaSuccess) {
        return cx::fail(e, "cudaIpcOpenMemHandle failed: %s (%d)",
                        cudaGetErrorString(rc), (int)rc);
    }
    *out_ptr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(peer));
    return 0;
}

// @brief 关闭已打开的对端 IPC 指针。
extern "C" int coffin_p2p_close_handle(uint64_t peer_ptr, char* err, int errlen) {
    const cx::Err e{err, errlen};
    if (!peer_ptr) return 0;
    const cudaError_t rc = cudaIpcCloseMemHandle(
        reinterpret_cast<void*>(static_cast<uintptr_t>(peer_ptr)));
    if (rc != cudaSuccess) {
        return cx::fail(e, "cudaIpcCloseMemHandle failed: %s (%d)",
                        cudaGetErrorString(rc), (int)rc);
    }
    return 0;
}

// @brief 发射四卡 bf16 归约核（可选加 ``add``，奇偶双缓冲，fp32 累加、直接写 bf16）。
extern "C" int coffin_p2p_combine_bf16(const void* x, const void* add, void* out,
                                       uint64_t s0, uint64_t s1, uint64_t s2, uint64_t s3,
                                       uint64_t f0, uint64_t f1, uint64_t f2, uint64_t f3,
                                       long long buf_stride, long long total,
                                       int rank, long long round, int blocks, int threads,
                                       void* stream, char* err, int errlen) {
    const cx::Err e{err, errlen};
    if (!x || !out) return cx::fail(e, "x and out must be non-null");
    if (!s0 || !s1 || !s2 || !s3 || !f0 || !f1 || !f2 || !f3) {
        return cx::fail(e, "slot and flag pointers must be non-null");
    }
    if (rank < 0 || rank >= COFFIN_P2P_WORLD) {
        return cx::fail(e, "rank %d out of [0, %d)", rank, COFFIN_P2P_WORLD);
    }
    if (round < 1 || total <= 0 || buf_stride < COFFIN_P2P_WORLD * total) {
        return cx::fail(e, "invalid round=%lld total=%lld buf_stride=%lld",
                        round, total, buf_stride);
    }
    if (blocks <= 0 || blocks > COFFIN_P2P_MAX_BLOCKS || threads <= 0 || threads > 1024) {
        return cx::fail(e, "grid (%d x %d) out of resident limit", blocks, threads);
    }
    const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    p2p_combine_bf16_kernel<<<blocks, threads, 0, s>>>(
        reinterpret_cast<const uint16_t*>(x),
        reinterpret_cast<const uint16_t*>(add),
        reinterpret_cast<uint16_t*>(static_cast<uintptr_t>(s0)),
        reinterpret_cast<uint16_t*>(static_cast<uintptr_t>(s1)),
        reinterpret_cast<uint16_t*>(static_cast<uintptr_t>(s2)),
        reinterpret_cast<uint16_t*>(static_cast<uintptr_t>(s3)),
        reinterpret_cast<long long*>(static_cast<uintptr_t>(f0)),
        reinterpret_cast<long long*>(static_cast<uintptr_t>(f1)),
        reinterpret_cast<long long*>(static_cast<uintptr_t>(f2)),
        reinterpret_cast<long long*>(static_cast<uintptr_t>(f3)),
        reinterpret_cast<uint16_t*>(out),
        buf_stride, total, rank, round);
    return cx::finish(e);
}
