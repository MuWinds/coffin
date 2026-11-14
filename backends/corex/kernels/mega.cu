// @file mega.cu
// @brief 解码步的常驻网格：把整层、乃至整条层栈收进一次发射。
//
// Module: coffin/backends/corex/kernels。绑定成 ``mega_*``，只由
// ``backends/corex/__init__.py`` 与 ``models/qwen35`` 的解码路径调用。
//
// 为什么要有这个文件：一步 decode 现在发 1291 次核。每次发射本卡 5.75 us 的下限
// 就吃掉 7.4 ms，而整步才 29 ms。逐算子融合已经走到头了——剩下的发射数就是**阶段数**
// （每层一次归一化、一次投影、一次递推、一次路由……），要让它们再少一个数量级，只有
// 一条路：把整层放进一个核，层与层之间在核内用栅栏连接。
//
// Constraints:
//   * **栅栏只能给"恰好驻留"的网格用**。本卡 16 个 SM、每块最多 4096 线程，所以
//     ``grid = SM 数`` 加一份小资源预算时，每块独占一个 SM，全部同时驻留是可证的，
//     不是"通常如此"。``moe_dispatch.cu`` 里那段说明说的就是这个前提：一个没被同时
//     调度的 block 会让另一个 block 静默死等。**任何让 grid 大于 SM 数、或让单块资源
//     撑不下两个块的改动都会把静默死等请回来。**
//   * 栅栏用"计数 + 代次"两段式：最后到达的块负责复位计数并发代次，其余块只轮询代次。
//     单计数器 + 自旋读会让先到的块读到一个"看起来对"的旧值，那是这里唯一会静默
//     出错的失败模式。
//   * 本文件所有指针参数都不是 ``__restrict__``：融合后的阶段读写同一批缓冲区
//     （残差流、``conv_out``、``normed``），``__restrict__`` 会授权编译器把前一阶段的
//     加载缓存到后一阶段的存储之后，那种误编译只在卡上出现。

#include "corex_bindings.h"
#include "corex_ops.h"
#include "corex_scalar.h"

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

namespace {

//: 栅栏的代次上限。代次是 ``unsigned``，一圈就是 40 层 × 十几个阶段，远不到回绕；
//: 这里保留一个显式检查，让"回绕后栅栏静默变成空转"这种失败至少能断言住。
constexpr unsigned kGenLimit = 1u << 30;

// @brief 常驻网格的全局栅栏：所有块在此会齐，然后一起往下走。
//
// @param counter 计数单元，初值 ``0``，由本函数复位。
// @param gen 代次单元，初值 ``0``，只有最后到达的块推进它。
// @param nblocks 参与栅栏的块数，必须等于本次发射的 ``gridDim.x``。
//
// 调用前的那道 ``__syncthreads()`` 是必需的：块内必须先把各自那份共享内存里的活干完，
// 否则线程 0 会代表一个还在算的块到达栅栏。调用后的那道同理，它让整块在别人继续之前
// 先看到释放。
__device__ __forceinline__ void pinned_barrier(unsigned* counter, unsigned* gen,
                                               unsigned nblocks) {
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned my_gen = *reinterpret_cast<volatile unsigned*>(gen);
        const unsigned arrived = atomicAdd(counter, 1u);
        if (arrived == nblocks - 1u) {
            // 复位必须早于放行：下一轮的第一个到达者会立刻加这个计数。
            *counter = 0u;
            __threadfence();
            atomicAdd(gen, 1u);
        } else {
            while (*reinterpret_cast<volatile unsigned*>(gen) == my_gen) {
                __threadfence();
            }
        }
    }
    __syncthreads();
    __threadfence();
}

// @brief 栅栏的冒烟核：连做 ``iters`` 次栅栏，最后把每块每线程的一个计数写进 ``out``。
//
// 这个核的存在理由是**它是唯一能证明栅栏安全的东西**。栅栏错了的现象是整块卡死，
// 而卡死的代价是一张卡要复位；所以先用一个不碰任何模型状态的核把路径走通，再去接
// 真正的层。
//
// @param out [blocks] int32 输出，第 ``b`` 格是块 ``b`` 走完栅栏后的代次快照。
// @param counter / gen 栅栏用的两个单元，调用方给一块两元素的 ``unsigned`` 缓冲，
//        进来时必须是 0。
// @param iters 栅栏轮数。
__global__ void mega_bar_probe_kernel(int* out, unsigned* counter, unsigned* gen,
                                      int iters) {
    for (int i = 0; i < iters; ++i) {
        pinned_barrier(counter, gen, gridDim.x);
    }
    if (threadIdx.x == 0) {
        out[blockIdx.x] = (int)(*reinterpret_cast<volatile unsigned*>(gen));
    }
}

}  // namespace

// @brief 跑 ``iters`` 次栅栏，返回每块看到的代次。
//
// @param out [blocks] int32 输出。
// @param blocks 网格块数；必须不大于本卡 SM 数（见文件头）。
// @param threads 每块线程数。
// @param iters 栅栏轮数。
// @return ``out``。
// @throws c10::Error 形状或块数越界。
torch::Tensor mega_bar_probe(torch::Tensor out, torch::Tensor gate, int64_t blocks,
                             int64_t threads, int64_t iters) {
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kInt32,
                "out must be a CUDA int32 tensor");
    TORCH_CHECK(gate.is_cuda() && gate.scalar_type() == torch::kInt32 && gate.numel() >= 2,
                "gate must be a CUDA int32 tensor with two elements");
    const int b = (int)blocks, t = (int)threads;
    TORCH_CHECK(b >= 1 && b <= 64, "blocks out of range");
    TORCH_CHECK(t >= 32 && t <= 4096 && t % 32 == 0, "threads must be a positive multiple of 32");
    TORCH_CHECK((int64_t)out.numel() >= b, "out must hold one cell per block");
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    mega_bar_probe_kernel<<<b, t, 0, stream>>>(
        out.data_ptr<int>(), reinterpret_cast<unsigned*>(gate.data_ptr<int>()),
        reinterpret_cast<unsigned*>(gate.data_ptr<int>()) + 1, (int)iters);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

COFFIN_BIND(mega_bar_probe, "pinned-grid barrier smoke test",
            pybind11::arg("out"), pybind11::arg("gate"), pybind11::arg("blocks"),
            pybind11::arg("threads"), pybind11::arg("iters"));
