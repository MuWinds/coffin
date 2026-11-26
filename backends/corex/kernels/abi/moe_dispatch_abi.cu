// @file moe_dispatch_abi.cu
// @brief MoE 分派的 C ABI 出口：``coffin_moe_dispatch``，无 torch。
//
// Module: coffin/backends/corex/kernels/abi。设备代码来自
// ``../moe_dispatch_device.inc``——与 ``../moe_dispatch.cu`` 是同一份
// ``moe_dispatch_kernel``，所以两条前端逐位相同（包括 ``atomicAdd`` 定段内顺序这条
// 已知的不确定性，见文件头）。
//
// 为什么它值得搬：模型侧原本是 14 个内核、672 次/步，按本卡 5.76 us 的发射地板是
// 3.87 ms/步，而输入只有 ``slots`` 个整数（decode 时 10 个）。
//
// Constraints:
//   * 签名与 ``cxrt/moe.py`` 的 ``argtypes`` 必须逐字一致：多一个参数就整体错位。
//   * ``expert_idx`` 允许 int32 或 int64：路由那条出口直接写 int64，窄化是一次独立
//     发射（实测 11 us/层）。分派核总共只读 ``slots`` 个元素，按入参 dtype 选模板
//     比多做一次窄化便宜。
//   * ``offsets`` 进内核时当计数器（``[e+1]`` 累加），出内核时被前缀和整体覆盖：
//     它同时是输入与输出，调用方不必预清零。
//   * 共享内存 ``(2 * local_experts + 1) * 4`` 字节；48 KB 是不加额外属性时的上限。

#include "cx_tensor.h"

#include <cstdint>

// 内核在文件作用域：头部只带 ``<cstdint>``，没有标准库头会被塞进命名空间的问题。
#include "../moe_dispatch_device.inc"

namespace {

// @brief 启动前的形状、dtype 与几何校验。
//
// @param idx 展平的全局专家号。
// @param table 全局号 → 本地号。
// @param offsets 前缀和目的地（``[E + 1]``）。
// @param pos 收拢后行号（``[slots]``）。
// @param toks 每行的 token 下标（``[slots]``）。
// @param slots_out 每行来自哪个槽位（``[slots]``）。
// @param scratch 游标暂存（``>= E``）。
// @param slots/local_experts/num_experts/top_k 几何。
// @param err 错误缓冲。
// @return 合法为 ``0``。
int check_dispatch(const cx::Tensor& idx, const cx::Tensor& table, const cx::Tensor& offsets,
                   const cx::Tensor& pos, const cx::Tensor& toks, const cx::Tensor& slots_out,
                   const cx::Tensor& scratch, int slots, int local_experts, int num_experts,
                   int top_k, const cx::Err& err) {
    if (!idx.data || !table.data || !offsets.data || !pos.data || !toks.data ||
        !slots_out.data || !scratch.data) {
        return cx::fail(err, "expert_idx, table, destinations and scratch must be non-null");
    }
    if (idx.dtype != cx::kI32 && idx.dtype != cx::kI64) {
        return cx::fail(err, "expert_idx must be int32 or int64, got dtype code %d",
                        (int)idx.dtype);
    }
    if (table.dtype != cx::kI32 || offsets.dtype != cx::kI32 || pos.dtype != cx::kI32 ||
        toks.dtype != cx::kI32 || slots_out.dtype != cx::kI32 || scratch.dtype != cx::kI32) {
        return cx::fail(err, "table, destinations and scratch must be int32");
    }
    if (local_experts <= 0 || num_experts <= 0 || top_k <= 0 || slots < 0) {
        return cx::fail(err, "sizes must be positive (slots may be 0), got %d/%d/%d/%d",
                        slots, local_experts, num_experts, top_k);
    }
    if (table.cols != num_experts) {
        return cx::fail(err, "local_table must hold %d entries, got %lld", num_experts,
                        (long long)table.cols);
    }
    if (offsets.cols != local_experts + 1) {
        return cx::fail(err, "offsets must be [%d], got %lld", local_experts + 1,
                        (long long)offsets.cols);
    }
    if (pos.cols != slots || toks.cols != slots || slots_out.cols != slots) {
        return cx::fail(err, "position tables must be [%d], got %lld/%lld/%lld", slots,
                        (long long)pos.cols, (long long)toks.cols, (long long)slots_out.cols);
    }
    if (scratch.cols < local_experts) {
        return cx::fail(err, "scratch must hold %d entries, got %lld", local_experts,
                        (long long)scratch.cols);
    }
    if (idx.rows * idx.cols < slots) {
        return cx::fail(err, "expert_idx must hold %d slots, got %lld", slots,
                        (long long)(idx.rows * idx.cols));
    }
    const int64_t nchunks = ((int64_t)local_experts + 15) / 16;
    const int64_t smem = (3 * (int64_t)local_experts + 1 + nchunks) * (int64_t)sizeof(int);
    if (smem > 48 * 1024) {
        return cx::fail(err, "dispatch needs %lld bytes of dynamic shared memory (> 48 KB "
                             "without extra attributes); local_experts=%d",
                        (long long)smem, local_experts);
    }
    return 0;
}

}  // namespace

// @brief 路由结果 → ``(offsets, slot_pos, token_of_pos, slot_of_pos)``，单 block 一次算完。
//
// @param expert_idx ``[slots]`` 展平的全局专家号，int32 或 int64。
// @param local_table ``[num_experts]`` 全局号 → 本地号，int32，不属于本 rank 为 -1。
// @param out_offsets ``[local_experts + 1]`` int32；进内核当计数器，出内核是前缀和。
// @param out_slot_pos ``[slots]`` int32，收拢后的行号，丢弃的为 -1。
// @param out_tokens ``[slots]`` int32，收拢后每行的 token 下标。
// @param out_slots ``[slots]`` int32，收拢后每行来自哪个槽位。
// @param scratch ``[>= local_experts]`` int32 游标暂存，阶段 1 会清零。
// @param slots 槽位数 ``tokens * top_k``。
// @param local_experts 本 rank 的专家数。
// @param num_experts 全局专家数。
// @param top_k 每 token 的专家数。
// @param idx_code ``expert_idx`` 的 dtype 码（``cx::kI32`` / ``cx::kI64``）。
// @param stream 目标流；空指针表示默认流。
// @param err 错误缓冲，可为空。
// @param errlen 错误缓冲字节数。
// @return ``0`` 成功。
extern "C" int coffin_moe_dispatch(const void* expert_idx, const void* local_table,
                                   void* out_offsets, void* out_slot_pos, void* out_tokens,
                                   void* out_slots, void* scratch, int slots,
                                   int local_experts, int num_experts, int top_k, int idx_code,
                                   void* stream, char* err, int errlen) {
    const cx::Err e{err, errlen};
    const cx::DType dt = (cx::DType)idx_code;
    const cx::Tensor idx = cx::Tensor::matrix(const_cast<void*>(expert_idx), 1, slots, slots, dt);
    const cx::Tensor table = cx::Tensor::matrix(const_cast<void*>(local_table), 1, num_experts,
                                                num_experts, cx::kI32);
    const cx::Tensor offsets = cx::Tensor::matrix(out_offsets, 1, local_experts + 1,
                                                  local_experts + 1, cx::kI32);
    const cx::Tensor pos = cx::Tensor::matrix(out_slot_pos, 1, slots, slots, cx::kI32);
    const cx::Tensor toks = cx::Tensor::matrix(out_tokens, 1, slots, slots, cx::kI32);
    const cx::Tensor slot_out = cx::Tensor::matrix(out_slots, 1, slots, slots, cx::kI32);
    const cx::Tensor cur = cx::Tensor::matrix(scratch, 1, local_experts, local_experts, cx::kI32);
    if (const int bad = check_dispatch(idx, table, offsets, pos, toks, slot_out, cur, slots,
                                       local_experts, num_experts, top_k, e)) {
        return bad;
    }
    if (slots == 0) return cx::finish(e);

    const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    const int64_t nchunks = ((int64_t)local_experts + 15) / 16;
    const int64_t smem = (3 * (int64_t)local_experts + 1 + nchunks) * (int64_t)sizeof(int);
    // 线程数按槽位数分两档：decode 只有 10 个槽位，开 512 个线程纯属浪费发射；
    // prefill 512 token 有 5120 个，每线程 10 个比 20 个少一半循环。
    const int threads = (slots >= 1024) ? 512 : 256;
    int* off = reinterpret_cast<int*>(offsets.data);
    int* cursor = reinterpret_cast<int*>(cur.data);
    int* pos_ptr = reinterpret_cast<int*>(pos.data);
    int* tok_ptr = reinterpret_cast<int*>(toks.data);
    int* slot_ptr = reinterpret_cast<int*>(slot_out.data);
    const int* table_ptr = reinterpret_cast<const int*>(table.data);
    if (dt == cx::kI64) {
        moe_dispatch_kernel<int64_t><<<1, threads, (size_t)smem, s>>>(
            reinterpret_cast<const int64_t*>(idx.data), table_ptr, pos_ptr, tok_ptr, slot_ptr,
            off, cursor, slots, local_experts, num_experts, top_k);
    } else {
        moe_dispatch_kernel<int><<<1, threads, (size_t)smem, s>>>(
            reinterpret_cast<const int*>(idx.data), table_ptr, pos_ptr, tok_ptr, slot_ptr, off,
            cursor, slots, local_experts, num_experts, top_k);
    }
    return cx::finish(e);
}
