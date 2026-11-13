// @file p2p_reduce.cu
// @brief 设备内存上的四卡 all-to-all 归约：一个核完成「写对端槽 + 自旋等齐 + 求和」。
//
// Module: coffin/backends/corex/kernels。由 ``backends/corex`` 作为 torch 扩展加载，
// 只有专家并行（``COFFIN_EP_COLLECTIVE=p2p``）会用到它。
//
// Constraints:
//   * 只允许设备作用域栅栏 ``__threadfence()``。系统级的 ``__threadfence_system()``
//     在这版 clang-16/ivcore10 上会让 llc 崩（``report_fatal_error`` + core dump，
//     十行最小复现见 ``.tmp_stage/pd/ice/fence.cu``），而这张卡又没有 PTX 可内联。
//     走设备内存就是为了绕开它：跨卡写只要设备作用域。
//   * 本文件**不碰裸主机指针**：``mapped_reduce.cu`` 里那种「主机 mmap + torch 张量 +
//     设备核」混在一个翻译单元里的写法会触发编译器 ICE，验证过两次。
//   * world 固定 4：指针用标量传，省掉一次设备指针数组的分配与解析。

#include <cuda_bf16.h>
#include <cuda_runtime.h>

// C10_CUDA_KERNEL_LAUNCH_CHECK 在这里，和其他核文件一致；缺了它编译起不来。
#include <c10/cuda/CUDAException.h>

#include <cstdint>
#include <cstring>

#include "corex_ops.h"

#define COFFIN_P2P_WORLD 4

/// @brief 自旋迭代上限，约等于零点几秒。超过就说明轮次真的错开了（有 rank 已经
///        退出会合，永远不会写这一轮），再等下去只会把整张卡焊死。
#define COFFIN_P2P_SPIN_LIMIT 50000000LL

/// @brief 中止槽的下标：``flags[world]``。与到达计数分开，绝不写进 ``[0, world)``，
///        否则被污染的那一轮之后 ``mine[r] < round`` 永远为假，各 rank 会带着
///        半截载荷继续算下去（静默错数比挂死更难查）。
#define COFFIN_P2P_ABORT_SLOT COFFIN_P2P_WORLD

namespace {

/// @brief 写对端槽 → 栅栏 → 置计数 → 自旋等齐 → 本地求和。
///
/// @param x       本 rank 的贡献，bf16 ``[rows, hidden]``，连续。
/// @param slots0..3 四个 rank 各自的 ``[world, rows, hidden]`` fp32 槽区。
/// @param flags0..3 四个 rank 各自的 ``[world]`` int64 到达计数。
/// @param out     求和结果，fp32 ``[rows, hidden]``。
/// @param rows    行数（decode 是 1）。
/// @param hidden  宽度。
/// @param rank    本进程编号。
/// @param round   本轮轮次，单调递增。
__global__ void p2p_combine_kernel(const __nv_bfloat16* __restrict__ x,
                                   float* slots0, float* slots1, float* slots2,
                                   float* slots3,
                                   long long* flags0, long long* flags1,
                                   long long* flags2, long long* flags3,
                                   float* __restrict__ out,
                                   int rows, int hidden, int rank, long long round) {
    float* base[COFFIN_P2P_WORLD] = {slots0, slots1, slots2, slots3};
    long long* flg[COFFIN_P2P_WORLD] = {flags0, flags1, flags2, flags3};
    const long long total = static_cast<long long>(rows) * hidden;
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    const long long start = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;

    // 中止是锁存的：一旦有人写过中止槽，本进程之后的每一轮都不再自旋，直接出零
    // 结果返回。宁可让输出变成垃圾，也不能把卡停在核心里等一个永远不来的写。
    __shared__ int failed;
    if (threadIdx.x == 0) {
        failed = (flg[rank][COFFIN_P2P_ABORT_SLOT] != 0) ? 1 : 0;
    }
    __syncthreads();
    if (failed) {
        for (long long i = start; i < total; i += stride) {
            out[i] = 0.0F;
        }
        return;                     // 全块同时走到这里，不会有线程落在 __syncthreads 上
    }

    // 本 rank 的载荷写进每个对端的第 rank 段（含自己）：这一步就是 all-to-all。
    for (int p = 0; p < COFFIN_P2P_WORLD; ++p) {
        float* dst = base[p] + static_cast<long long>(rank) * total;
        for (long long i = start; i < total; i += stride) {
            dst[i] = __bfloat162float(x[i]);
        }
    }
    // 载荷对设备可见之后才置计数。对端是设备内存，设备作用域就够。
    __threadfence();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int p = 0; p < COFFIN_P2P_WORLD; ++p) {
            flg[p][rank] = round;
        }
    }
    __syncthreads();
    // 自旋等所有 rank 到达。计数在设备内存里，volatile 读就能看到对端刚写的值。
    // 注意：这里试过用原子（atomicExch 发布 + atomicAdd 读）来绕开跨卡可见性，
    // 实测在这套 corex 卡上第 4 轮就必挂——对端内存的原子不参与会合，比 volatile
    // 更糟，所以退回 volatile。
    if (threadIdx.x == 0) {
        volatile long long* mine = flg[rank];
        long long spins = 0;
        int bad = 0;
        for (int r = 0; r < COFFIN_P2P_WORLD; ++r) {
            while (mine[r] < round) {
                // 对端把自己放出去时写的是我这个 rank 的槽：它也会同时叫醒我。
                if (mine[COFFIN_P2P_ABORT_SLOT] != 0) { bad = 1; break; }
                __threadfence();
                if (++spins > COFFIN_P2P_SPIN_LIMIT) { bad = 1; break; }
            }
            if (bad) break;
        }
        if (bad) {
            // 给所有人写我这一轮的中止号，把正在等我的人一起放出来。
            for (int p = 0; p < COFFIN_P2P_WORLD; ++p) {
                flg[p][COFFIN_P2P_ABORT_SLOT] = round;
            }
            failed = 1;
        } else {
            failed = 0;
        }
    }
    __syncthreads();
    if (failed) {
        for (long long i = start; i < total; i += stride) {
            out[i] = 0.0F;
        }
        return;
    }
    // fp32 累加、最后只窄化一次：四个 bf16 直接相加会丢两三位尾数。
    const float* my = base[rank];
    for (long long i = start; i < total; i += stride) {
        float acc = 0.0F;
        for (int r = 0; r < COFFIN_P2P_WORLD; ++r) {
            acc += my[static_cast<long long>(r) * total + i];
        }
        out[i] = acc;
    }
}

}  // namespace

torch::Tensor p2p_handle(torch::Tensor device_buffer) {
    TORCH_CHECK(device_buffer.is_cuda(), "p2p_handle needs a CUDA tensor");
    cudaIpcMemHandle_t handle;
    const cudaError_t err =
        cudaIpcGetMemHandle(&handle, device_buffer.data_ptr());
    TORCH_CHECK(err == cudaSuccess,
                "cudaIpcGetMemHandle failed: ", cudaGetErrorString(err));
    auto out = torch::empty({static_cast<int64_t>(sizeof(handle))},
                            torch::TensorOptions().dtype(torch::kUInt8));
    std::memcpy(out.data_ptr(), &handle, sizeof(handle));
    return out;
}

int64_t p2p_open(torch::Tensor handle_bytes) {
    TORCH_CHECK(handle_bytes.device().is_cpu() &&
                    handle_bytes.scalar_type() == torch::kUInt8,
                "p2p_open needs the uint8 handle tensor from p2p_handle");
    TORCH_CHECK(static_cast<size_t>(handle_bytes.numel()) == sizeof(cudaIpcMemHandle_t),
                "p2p_open got a handle of the wrong size");
    void* peer = nullptr;
    const cudaError_t err = cudaIpcOpenMemHandle(
        &peer, *reinterpret_cast<const cudaIpcMemHandle_t*>(handle_bytes.data_ptr()),
        cudaIpcMemLazyEnablePeerAccess);
    TORCH_CHECK(err == cudaSuccess,
                "cudaIpcOpenMemHandle failed: ", cudaGetErrorString(err));
    return static_cast<int64_t>(reinterpret_cast<uintptr_t>(peer));
}

void p2p_combine(torch::Tensor x, torch::Tensor out,
                 int64_t slots0, int64_t slots1, int64_t slots2, int64_t slots3,
                 int64_t flags0, int64_t flags1, int64_t flags2, int64_t flags3,
                 int64_t rank, int64_t round, int64_t blocks, int64_t threads) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 2,
                "p2p_combine needs a contiguous 2-D CUDA input");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "p2p_combine needs bf16 input");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.dim() == 2,
                "p2p_combine needs a contiguous 2-D CUDA output");
    TORCH_CHECK(out.scalar_type() == torch::kFloat32, "p2p_combine needs fp32 output");
    TORCH_CHECK(round >= 1, "p2p_combine needs a positive round");
    TORCH_CHECK(blocks > 0 && threads > 0 && threads <= 1024, "grid out of range");
    const int rows = static_cast<int>(x.size(0));
    const int hidden = static_cast<int>(x.size(1));
    TORCH_CHECK(out.size(0) == rows && out.size(1) == hidden,
                "p2p_combine shape mismatch between input and output");
    p2p_combine_kernel<<<static_cast<int>(blocks), static_cast<int>(threads)>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<float*>(static_cast<uintptr_t>(slots0)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(slots1)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(slots2)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(slots3)),
        reinterpret_cast<long long*>(static_cast<uintptr_t>(flags0)),
        reinterpret_cast<long long*>(static_cast<uintptr_t>(flags1)),
        reinterpret_cast<long long*>(static_cast<uintptr_t>(flags2)),
        reinterpret_cast<long long*>(static_cast<uintptr_t>(flags3)),
        out.data_ptr<float>(), rows, hidden, static_cast<int>(rank),
        static_cast<long long>(round));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
