// @file moe_grouped_i8_merge_abi.cu
// @brief 分组 int8 的**归并**形态两条出口，无 torch：
// ``coffin_moe_grouped_i8_merge`` / ``coffin_moe_grouped_i8_merge_tokens``。
//
// Module: coffin/backends/corex/kernels/abi。设备代码来自
// ``../moe_grouped_i8_device.inc``（主体核 + 三个收尾核都在里面）。
//
// 这两条出口比单趟那条多两样**跨调用要活下来的东西**，原来是 torch 给的：
//
// 1. ``part``：``[span, rows, N]`` int32 分片缓冲。这里按容量缓存一份设备内存，只在
//    形状变大时重新申请（``decode`` 的每层形状固定，热起来之后一次都不会再申请）。
// 2. ``mcnt``：按列块计数的到达计数，**初值必须是 0、且不能每次清零**——主体核在每个
//    (row, col) 组的末块里自复位。所以按 ``cols`` 缓存一份 ``cudaMalloc`` 的内存在首次
//    调用时清零一次，之后跨调用一直可用。
//
// Constraints:
//   * 签名与 ``cxrt/moe.py`` 的 ``argtypes`` 必须逐字一致。
//   * 归并形态强制 ``span >= 2``（主体核一律写 ``part``）；``nk`` 参数已从签名里去掉。
//   * 分片缓冲是**进程内共享**的一份：并发调用（多流）会互相踩。解码路径上这两条都在
//     同一条流上串行发射，这条约束与原来的 torch 版一致（那份 ``part`` 也是每次新申请、
//     只在同一次调用里活）。

#include "cx_tensor.h"

#include <cstdint>

#include "../moe_grouped_i8_device.inc"

namespace {

//: 分片缓冲：(设备, 字节数) → 设备内存。只增不减。
struct Scratch {
    int device;
    size_t bytes;
    void* ptr;
};

void* part_buffer(int device, size_t bytes, cudaStream_t s) {
    static Scratch cache{ -1, 0, nullptr };
    if (cache.ptr != nullptr && cache.device == device && cache.bytes >= bytes) {
        return cache.ptr;
    }
    if (cache.ptr != nullptr) cudaFree(cache.ptr);
    void* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) {
        cache = Scratch{ -1, 0, nullptr };
        return nullptr;
    }
    cache = Scratch{ device, bytes, p };
    (void)s;
    return p;
}

//: 到达计数：(设备, 列块数) → 设备内存，**首次清零一次**。
int* counter_buffer(int device, int cols) {
    struct Entry {
        int device;
        int cols;
        int* ptr;
    };
    static Entry cache[8];
    static int used = 0;
    for (int i = 0; i < used; ++i) {
        if (cache[i].device == device && cache[i].cols == cols) return cache[i].ptr;
    }
    int* p = nullptr;
    if (cudaMalloc(&p, (size_t)cols * sizeof(int)) != cudaSuccess) return nullptr;
    if (cudaMemset(p, 0, (size_t)cols * sizeof(int)) != cudaSuccess) {
        cudaFree(p);
        return nullptr;
    }
    if (used < 8) {
        cache[used].device = device;
        cache[used].cols = cols;
        cache[used].ptr = p;
        ++used;
    }
    return p;
}

//: 两条出口共用的块几何（nk 单栈，``wide`` 恒为真）。
struct Geom {
    int vb;
    int wcol;
    int warps;
    int cols;
    int cols_tail;
    int span;
    int per;
};

// @brief 按 ``(K, N, rows, slices)`` 定块几何；与原来的主机包装逐字同一套。
Geom geometry(int k, int n, int rows, int slices) {
    const int vb = (k % I8_NK_WIDE_BYTES == 0 && k <= I8_NK_WIDE_K) ? I8_NK_WIDE_BYTES : 4;
    const int wcol = (vb == I8_NK_WIDE_BYTES && k < I8_NK_SHALLOW_K) ? I8_NK_WCOL_SHALLOW : 1;
    const int warps = (k < I8_NK_SHALLOW_K) ? I8_NK_WARPS_SHALLOW : I8_NK_WARPS;
    Geom g{};
    g.vb = vb;
    g.wcol = wcol;
    g.warps = warps;
    g.cols = (vb == I8_NK_WIDE_BYTES) ? (n + warps * wcol - 1) / (warps * wcol)
                                     : (n + I8_COLS - 1) / I8_COLS;
    g.cols_tail = (n + I8_COLS - 1) / I8_COLS;
    // 自动值固定 2：1 片时主体核会把结果直接写成单行，归并仍要多一趟读改写；而按
    // 「K 装得下一轮宽步就取 1 片」试过，实测文本退化（见原主机包装的注释）。
    int span = slices;
    if (span <= 0) span = 2;
    if (span > I8_MAX_SLICES) span = I8_MAX_SLICES;
    if (span > k) span = k;
    if (span < 1) span = 1;
    g.span = span;
    g.per = ((k + span - 1) / span + vb - 1) / vb * vb;
    return g;
}

// @brief 主体核的三种模板档：宽步 + 一 warp 多列 / 宽步 + 一列 / 4 字节步。
template <typename XT>
void launch_main(const Geom& g, const int8_t* a, const int8_t* w, const float* ws,
                 const float* xs, const int* off, int32_t* part, int* mcnt, const void* rw,
                 const int* slot_of_pos, int rw16, XT* dst, int k, int n, int experts, int rows,
                 cudaStream_t s) {
    const dim3 grid((unsigned)g.cols, (unsigned)rows, (unsigned)g.span);
    const dim3 block((unsigned)(g.vb == I8_NK_WIDE_BYTES ? g.warps * 32 : I8_COLS));
    const size_t smem = (size_t)k * sizeof(int8_t);
    if (g.vb == I8_NK_WIDE_BYTES) {
        if (g.wcol > 1) {
            moe_grouped_i8_nk_kernel<XT, I8_NK_WIDE_BYTES, I8_NK_WCOL_SHALLOW>
                <<<grid, block, smem, s>>>(a, w, ws, xs, off, part, mcnt, rw, slot_of_pos, rw16,
                                           dst, k, n, experts, rows, g.span, g.per);
        } else {
            moe_grouped_i8_nk_kernel<XT, I8_NK_WIDE_BYTES, 1>
                <<<grid, block, smem, s>>>(a, w, ws, xs, off, part, mcnt, rw, slot_of_pos, rw16,
                                           dst, k, n, experts, rows, g.span, g.per);
        }
    } else {
        moe_grouped_i8_nk_kernel<XT, 4, 1><<<grid, block, smem, s>>>(
            a, w, ws, xs, off, part, mcnt, rw, slot_of_pos, rw16, dst, k, n, experts, rows,
            g.span, g.per);
    }
}

}  // namespace

// @brief 归并形态：``[rows, K] -> [1, N]``，行按路由权重加权求和。
//
// @param a int8 激活，``[rows, K]`` 连续。
// @param xs 逐行激活 scale，``[rows]`` fp32。
// @param w 专家栈，``[E, N, K]`` 连续 int8。
// @param ws 专家栈 scale，``[E, N]`` fp32。
// @param offsets ``[E+1]`` 前缀和，int32。
// @param rw 逐行路由权重，``[rows]`` fp32。
// @param out ``[1, N]``，dtype 由 ``out_code`` 给。
// @param out_code 目的地 dtype 码（``kBF16`` / ``kF32``）。
// @param rows 行数。
// @param k 激活宽度。
// @param n 输出列数。
// @param experts 专家数。
// @param slices K 方向分片数；``<= 0`` 自动（2 片）。
// @param stream 目标流；空指针表示默认流。
// @param err 错误缓冲，可为空。
// @param errlen 错误缓冲字节数。
// @return ``0`` 成功。
extern "C" int coffin_moe_grouped_i8_merge(const void* a, const void* xs, const void* w,
                                           const void* ws, const void* offsets, const void* rw,
                                           void* out, int out_code, int rows, int k, int n,
                                           int experts, int slices, void* stream, char* err,
                                           int errlen) {
    const cx::Err e{err, errlen};
    if (!a || !xs || !w || !ws || !offsets || !rw || !out) {
        return cx::fail(e, "a, xs, w, ws, offsets, rw and out must be non-null");
    }
    if (out_code != cx::kF32 && out_code != cx::kBF16) {
        return cx::fail(e, "out dtype code %d is neither fp32 nor bf16", out_code);
    }
    if (rows <= 0 || k <= 0 || n <= 0 || experts <= 0) {
        return cx::fail(e, "rows/k/n/experts must be positive, got %d/%d/%d/%d", rows, k, n,
                        experts);
    }

    const Geom g = geometry(k, n, rows, slices);
    const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return cx::fail(e, "cannot read the current device");
    int32_t* part = reinterpret_cast<int32_t*>(
        part_buffer(dev, (size_t)g.span * rows * n * sizeof(int32_t), s));
    if (part == nullptr) return cx::fail(e, "cannot allocate the split-K partial buffer");
    int* mcnt = counter_buffer(dev, g.cols);
    if (mcnt == nullptr) return cx::fail(e, "cannot allocate the arrival counters");

    const float* wsp = reinterpret_cast<const float*>(ws);
    const float* xsp = reinterpret_cast<const float*>(xs);
    const int* off = reinterpret_cast<const int*>(offsets);
    const int8_t* ap = reinterpret_cast<const int8_t*>(a);
    const int8_t* wp = reinterpret_cast<const int8_t*>(w);

    auto launch = [&](auto tag) {
        using XT = decltype(tag);
        launch_main(g, ap, wp, wsp, xsp, off, part, mcnt, rw, nullptr, 0,
                    reinterpret_cast<XT*>(out), k, n, experts, rows, s);
        // ``mcnt`` 非空时主体核已经把单 token 的加权累加做完，收尾那一趟不存在。
    };
    if (out_code == cx::kBF16) launch(uint16_t{});
    else launch(float{});
    return cx::finish(e);
}

// @brief 逐 token 归并形态：``[rows, K] -> [tokens, N]``，每 token 只累加自己那 top_k 行。
//
// @param a int8 激活，``[rows, K]`` 连续，已按专家号排序。
// @param xs 逐行激活 scale，``[rows]`` fp32。
// @param w 专家栈，``[E, N, K]`` 连续 int8。
// @param ws 专家栈 scale，``[E, N]`` fp32。
// @param offsets ``[E+1]`` 前缀和，int32。
// @param rw 路由权重，``[rows]``，dtype 由 ``rw_code`` 给（``kBF16`` / ``kF32``）。
// @param slot_pos 槽位 → 行号，int32 ``[rows]``，丢弃的为 ``-1``。
// @param slot_of_pos 行号 → 槽位，int32 ``[rows]``。
// @param out ``[tokens, N]``，dtype 由 ``out_code`` 给。
// @param out_code 目的地 dtype 码。
// @param rw_code 路由权重 dtype 码。
// @param rows 激活行数（``= tokens * top_k``）。
// @param k 激活宽度。
// @param n 输出列数。
// @param experts 专家数。
// @param tokens 解码行数。
// @param top_k 每 token 的专家数。
// @param slices K 方向分片数；``<= 0`` 自动。
// @param stream 目标流；空指针表示默认流。
// @param err 错误缓冲，可为空。
// @param errlen 错误缓冲字节数。
// @return ``0`` 成功。
extern "C" int coffin_moe_grouped_i8_merge_tokens(
        const void* a, const void* xs, const void* w, const void* ws, const void* offsets,
        const void* rw, const void* slot_pos, const void* slot_of_pos, void* out, int out_code,
        int rw_code, int rows, int k, int n, int experts, int tokens, int top_k, int slices,
        void* stream, char* err, int errlen) {
    const cx::Err e{err, errlen};
    if (!a || !xs || !w || !ws || !offsets || !rw || !slot_pos || !slot_of_pos || !out) {
        return cx::fail(e, "a, xs, w, ws, offsets, rw, slot_pos, slot_of_pos and out must be "
                           "non-null");
    }
    if (out_code != cx::kF32 && out_code != cx::kBF16) {
        return cx::fail(e, "out dtype code %d is neither fp32 nor bf16", out_code);
    }
    if (rw_code != cx::kF32 && rw_code != cx::kBF16) {
        return cx::fail(e, "row weight dtype code %d is neither fp32 nor bf16", rw_code);
    }
    if (rows <= 0 || k <= 0 || n <= 0 || experts <= 0 || tokens <= 0 || top_k <= 0) {
        return cx::fail(e, "rows/k/n/experts/tokens/top_k must be positive, "
                           "got %d/%d/%d/%d/%d/%d", rows, k, n, experts, tokens, top_k);
    }
    if (rows != tokens * top_k) {
        return cx::fail(e, "%d activation rows do not split into %d x %d", rows, tokens, top_k);
    }

    const Geom g = geometry(k, n, rows, slices);
    const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return cx::fail(e, "cannot read the current device");
    int32_t* part = reinterpret_cast<int32_t*>(
        part_buffer(dev, (size_t)g.span * rows * n * sizeof(int32_t), s));
    if (part == nullptr) return cx::fail(e, "cannot allocate the split-K partial buffer");
    const bool single = tokens == 1;
    int* mcnt = single ? counter_buffer(dev, g.cols) : nullptr;
    if (single && mcnt == nullptr) return cx::fail(e, "cannot allocate the arrival counters");

    const float* wsp = reinterpret_cast<const float*>(ws);
    const float* xsp = reinterpret_cast<const float*>(xs);
    const int* off = reinterpret_cast<const int*>(offsets);
    const int8_t* ap = reinterpret_cast<const int8_t*>(a);
    const int8_t* wp = reinterpret_cast<const int8_t*>(w);
    const int* spo = reinterpret_cast<const int*>(slot_of_pos);
    const int* sp = reinterpret_cast<const int*>(slot_pos);
    const bool rw16 = rw_code == cx::kBF16;
    const dim3 mgrid((unsigned)g.cols_tail, (unsigned)(single ? 1 : tokens));
    const dim3 mblock(I8_COLS);

    auto launch = [&](auto tag) {
        using XT = decltype(tag);
        XT* dst = reinterpret_cast<XT*>(out);
        launch_main(g, ap, wp, wsp, xsp, off, part, mcnt, rw, spo, rw16 ? 1 : 0, dst, k, n,
                    experts, rows, s);
        if (single) {
            // 单 token 时主体核已把加权累加做完，这条路只是兜底（``mcnt`` 为空时）。
            if (mcnt == nullptr) {
                if (rw16) {
                    moe_grouped_i8_merge_rows_kernel<XT, uint16_t><<<mgrid, mblock, 0, s>>>(
                        part, wsp, xsp, off, reinterpret_cast<const uint16_t*>(rw), spo, dst,
                        n, experts, rows, g.span);
                } else {
                    moe_grouped_i8_merge_rows_kernel<XT, float><<<mgrid, mblock, 0, s>>>(
                        part, wsp, xsp, off, reinterpret_cast<const float*>(rw), spo, dst, n,
                        experts, rows, g.span);
                }
            }
        } else if (rw16) {
            moe_grouped_i8_merge_slots_kernel<XT, uint16_t><<<mgrid, mblock, 0, s>>>(
                part, wsp, xsp, off, reinterpret_cast<const uint16_t*>(rw), sp, dst, n,
                experts, rows, g.span, top_k);
        } else {
            moe_grouped_i8_merge_slots_kernel<XT, float><<<mgrid, mblock, 0, s>>>(
                part, wsp, xsp, off, reinterpret_cast<const float*>(rw), sp, dst, n, experts,
                rows, g.span, top_k);
        }
    };
    if (out_code == cx::kBF16) launch(uint16_t{});
    else launch(float{});
    return cx::finish(e);
}
