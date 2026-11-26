// @file abi/attn_abi.cu
// @brief 注意力族「小算子」的 C ABI 前端：``silu_mul`` / ``silu_mul_quant`` /
//        ``rope_apply`` / ``kv_quant``，不链接 torch。
//
// Module: coffin/backends/corex/kernels/abi。设备代码来自 ``attn_device.inc``——
// 与 ``../attn.cu`` 是同一份，所以两条前端的算术逐位相同（包括 RoPE 那四次窄化与
// ``kv_quant`` 的舍入顺序）。
//
// Constraints:
//   * 这里只搬了四个入口。``attn_decode`` / ``attn_paged`` / ``attn_paged_batch`` /
//     ``attn_paged_multi_q`` / ``attn_sparse`` / ``kv_store`` / ``kv_pool_store`` 都要
//     连 KV 池布局与块表一起定契约，是另一档工作量；先把这几条不需要任何布局知识的
//     路打通。
//   * 三个操作数各自携带行间距，谁也**不必**连续：投影输出的两半正是同一个更宽
//     ``[M, N]`` 缓冲区的列视图，强行要求连续会逼调用方再拷一遍。
//   * ``out`` 允许与 ``up`` 重叠（调用点把投影缓冲区交回来就不必分配）；除此之外的
//     重叠按调用方的 bug 处理，不做检测——内核服务不了部分重叠。
//   * ``coffin_rope_apply`` 只接受 bf16。与包装层同一条理由：旋转表先按存储类型窄化，
//     这是 ``apply_rope`` 的 ``cos.to(q.dtype)``，fp16 路径从来没有被模型走过。
//   * 流由调用方透传，不查任何全局「当前流」。

#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include "cx_tensor.h"
#include "corex_scalar.h"

namespace {

//: 调用方透传的流。``nullptr`` 表示默认流；每次入口进来先写它，退出前不动。
thread_local cudaStream_t g_cx_stream = nullptr;

// @brief 设备侧代码问“用哪条流”。
// @return 本次入口透传进来的句柄。
cudaStream_t cx_stream() { return g_cx_stream; }

// 三个小内核：与 ``attn.cu`` 共用同一份。
#include "../attn_device.inc"

//: ``silu_mul`` 两个入口的线程数。每列一条通道，网格 x 是列、y 是行；没有归约、没有
//: 共享内存、没有屏障，所以这里只需要一个常数。
constexpr int SILU_THREADS = 256;

// @brief 校验一对操作数（``gate``/``up``/``out``）的布局能吃下内核的索引。
//
// @param gate_rows 行数。
// @param width 每行列数。
// @param g_pitch/u_pitch/o_pitch 三个操作数的行间距，单位元素。
// @param err 错误缓冲。
// @return ``0`` 通过，否则为 ``cx::fail`` 的返回码。
int check_strided_triplet(long long rows, long long width, long long g_pitch,
                          long long u_pitch, long long o_pitch, const cx::Err& err) {
    if (rows <= 0 || width <= 0) return cx::fail(err, "rows=%lld width=%lld", rows, width);
    if (g_pitch < width || u_pitch < width || o_pitch < width) {
        return cx::fail(err, "row pitch narrower than the row width (%lld < %lld)",
                        g_pitch < width ? g_pitch : (u_pitch < width ? u_pitch : o_pitch),
                        width);
    }
    return 0;
}

}  // namespace

extern "C" {

// @brief ``out = silu(gate) * up``，一次启动。``out`` 可以与 ``up`` 重叠。
//
// @param gatep 门那一半，行间距 ``g_pitch``。
// @param upp 另一半，行间距 ``u_pitch``。
// @param outp 目的地，行间距 ``o_pitch``。
// @param rows 行数；一维操作数算一行。
// @param width 每行列数。
// @param g_pitch/u_pitch/o_pitch 三个操作数的行间距，单位元素。
// @param code dtype 码（bf16 或 fp32），三个操作数必须一致。
// @param stream 目标流；空表示默认流。
// @param errbuf 错误消息缓冲，可为空。
// @param errlen 缓冲长度。
// @return ``0`` 成功，``1`` 参数非法，``2`` 启动失败。
int coffin_silu_mul(const void* gatep, const void* upp, void* outp, long long rows,
                    long long width, long long g_pitch, long long u_pitch,
                    long long o_pitch, int code, void* stream, char* errbuf, int errlen) {
    const cx::Err err{errbuf, errlen};
    g_cx_stream = static_cast<cudaStream_t>(stream);
    if (!gatep || !upp || !outp) return cx::fail(err, "silu_mul needs non-null gate/up/out");
    const int rc = check_strided_triplet(rows, width, g_pitch, u_pitch, o_pitch, err);
    if (rc != 0) return rc;
    const dim3 grid((unsigned)((width + SILU_THREADS - 1) / SILU_THREADS), (unsigned)rows);
    switch (code) {
        case cx::kBF16:
            silu_mul_kernel<uint16_t><<<grid, SILU_THREADS, 0, g_cx_stream>>>(
                reinterpret_cast<const uint16_t*>(gatep),
                reinterpret_cast<const uint16_t*>(upp), reinterpret_cast<uint16_t*>(outp),
                (int)width, (long)g_pitch, (long)u_pitch, (long)o_pitch);
            break;
        case cx::kF32:
            silu_mul_kernel<float><<<grid, SILU_THREADS, 0, g_cx_stream>>>(
                static_cast<const float*>(gatep), static_cast<const float*>(upp),
                static_cast<float*>(outp), (int)width, (long)g_pitch, (long)u_pitch,
                (long)o_pitch);
            break;
        default:
            return cx::fail(err, "unsupported dtype code %d (bf16 or fp32 only)", code);
    }
    return cx::finish(err);
}

// @brief ``out = silu(gate) * up``，并顺手把它量化成 int8 + 每行 scale，一次启动。
//
// ``out`` 仍是 bf16/fp32 的落盘值，``out_q``/``out_s`` 是它的量化副本：三者的运算顺序
// 与「先 ``silu_mul`` 再 ``quant_act_i8``」逐位相同，所以两个调用点可以共用同一批字节。
//
// @param gatep 门那一半，行间距 ``g_pitch``。
// @param upp 另一半，行间距 ``u_pitch``。
// @param outp 目的地，行间距 ``o_pitch``；可以与 ``upp`` 重叠。
// @param out_qp int8 目的地，稠密 ``[rows, width]``。
// @param out_sp 每行 fp32 scale，``[rows]``。
// @param rows 行数。
// @param width 每行列数。
// @param g_pitch/u_pitch/o_pitch 三个操作数的行间距。
// @param code dtype 码（bf16 或 fp32）。
// @param stream 目标流。
// @param errbuf 错误缓冲。
// @param errlen 缓冲长度。
// @return ``0`` 成功，``1`` 参数非法，``2`` 启动失败。
int coffin_silu_mul_quant(const void* gatep, const void* upp, void* outp, void* out_qp,
                          void* out_sp, long long rows, long long width, long long g_pitch,
                          long long u_pitch, long long o_pitch, int code, void* stream,
                          char* errbuf, int errlen) {
    const cx::Err err{errbuf, errlen};
    g_cx_stream = static_cast<cudaStream_t>(stream);
    if (!gatep || !upp || !outp || !out_qp || !out_sp) {
        return cx::fail(err, "silu_mul_quant needs non-null gate/up/out/out_q/out_s");
    }
    const int rc = check_strided_triplet(rows, width, g_pitch, u_pitch, o_pitch, err);
    if (rc != 0) return rc;
    int8_t* qp = static_cast<int8_t*>(out_qp);
    float* sp = static_cast<float*>(out_sp);
    const dim3 grid((unsigned)rows);
    switch (code) {
        case cx::kBF16:
            silu_mul_quant_kernel<uint16_t><<<grid, SILU_THREADS, 0, g_cx_stream>>>(
                reinterpret_cast<const uint16_t*>(gatep),
                reinterpret_cast<const uint16_t*>(upp), reinterpret_cast<uint16_t*>(outp),
                qp, sp, (int)width, (long)g_pitch, (long)u_pitch, (long)o_pitch);
            break;
        case cx::kF32:
            silu_mul_quant_kernel<float><<<grid, SILU_THREADS, 0, g_cx_stream>>>(
                static_cast<const float*>(gatep), static_cast<const float*>(upp),
                static_cast<float*>(outp), qp, sp, (int)width, (long)g_pitch,
                (long)u_pitch, (long)o_pitch);
            break;
        default:
            return cx::fail(err, "unsupported dtype code %d (bf16 or fp32 only)", code);
    }
    return cx::finish(err);
}

// @brief 旋转每个 q/k 头最前面的 ``rot`` 个通道，一次启动。
//
// 四次窄化是契约：``cos``/``sin`` 各窄化一次（对应 ``cos.to(q.dtype)``），然后每个
// 乘积一次、和再一次。少任何一次都会改变 logits，所以这里不做「优化」，只是把同一份
// 设备代码接上裸指针。
//
// @param qp 查询，``[M, nh, hd]`` 稠密 bf16。
// @param kp 键，``[M, nkv, hd]`` 稠密 bf16。
// @param cosp/sinp ``[M, rot]`` fp32，这些 token 对应的行。
// @param q_outp 旋转后的查询，``[M, nh, hd]`` 稠密。
// @param k_outp 旋转后的键，``[M, nkv, hd]`` 稠密。
// @param m token 数。
// @param nh/nkv 头数。
// @param hd 头宽。
// @param rot 要旋转的前导通道数；必须为偶数且不宽于 ``hd``。
// @param stream 目标流。
// @param errbuf 错误缓冲。
// @param errlen 缓冲长度。
// @return ``0`` 成功，``1`` 参数非法，``2`` 启动失败。
int coffin_rope_apply(const void* qp, const void* kp, const void* cosp, const void* sinp,
                      void* q_outp, void* k_outp, long long m, long long nh, long long nkv,
                      long long hd, long long rot, void* stream, char* errbuf, int errlen) {
    const cx::Err err{errbuf, errlen};
    g_cx_stream = static_cast<cudaStream_t>(stream);
    if (!qp || !kp || !cosp || !sinp || !q_outp || !k_outp) {
        return cx::fail(err, "rope_apply needs non-null q/k/cos/sin/q_out/k_out");
    }
    if (m <= 0) return cx::fail(err, "rope_apply got M=%lld", m);
    if (nh <= 0 || nkv <= 0 || hd <= 0) {
        return cx::fail(err, "rope_apply got nh=%lld nkv=%lld hd=%lld", nh, nkv, hd);
    }
    if (rot <= 0 || rot > hd || (rot & 1) != 0) {
        return cx::fail(err, "rotary_dim must be even and no wider than a head, got %lld "
                             "for hd=%lld", rot, hd);
    }
    const int threads = hd < 64 ? 64 : (hd > 512 ? 512 : (int)hd);
    const dim3 grid((unsigned)((hd + threads - 1) / threads), (unsigned)m, 2u);
    rope_apply_kernel<uint16_t><<<grid, threads, 0, g_cx_stream>>>(
        reinterpret_cast<const uint16_t*>(qp), reinterpret_cast<const uint16_t*>(kp),
        static_cast<const float*>(cosp), static_cast<const float*>(sinp),
        reinterpret_cast<uint16_t*>(q_outp), reinterpret_cast<uint16_t*>(k_outp),
        (int)nh, (int)nkv, (int)hd, (int)rot);
    return cx::finish(err);
}

// @brief ``[M, N]`` 的对称 int8 量化：每行一个 fp32 scale，一次启动。
//
// 这是 int8 KV 缓存的写方，读方是 ``attn_slice``。两边的算术是契约：``amax`` → clamp 到
// ``1e-5`` → **先除 127 再除 scale**、``round`` 就近取偶、夹到 ``[-127, 127]``。任何一处
// 换顺序或换平局规则都不会让形状检查失败，只会让注意力输出漂移，所以这条的验收口径是
// 逐位一致，不是「误差很小」。
//
// @param xp 激活，行间距 ``pitch``。
// @param out_qp int8 目的地，稠密 ``[m, n]``。
// @param out_sp 每行 fp32 scale，``[m]``。
// @param m 行数。
// @param n 行宽。
// @param pitch ``x`` 的行间距，单位元素；``0`` 表示 ``n``。
// @param code dtype 码（bf16 或 fp32）。
// @param stream 目标流；空表示默认流。
// @param errbuf 错误消息缓冲，可为空。
// @param errlen 缓冲长度。
// @return ``0`` 成功，``1`` 参数非法，``2`` 启动失败。
int coffin_kv_quant(const void* xp, void* out_qp, void* out_sp, long long m, long long n,
                    long long pitch, int code, void* stream, char* errbuf, int errlen) {
    const cx::Err err{errbuf, errlen};
    g_cx_stream = static_cast<cudaStream_t>(stream);
    if (!xp || !out_qp || !out_sp) return cx::fail(err, "kv_quant needs non-null x/q/s");
    if (m <= 0 || n <= 0) return cx::fail(err, "kv_quant got M=%lld N=%lld", m, n);
    const long p = pitch > 0 ? (long)pitch : (long)n;
    if (p < n) return cx::fail(err, "row pitch %ld is narrower than the row %lld", p, n);
    const int threads = kv_quant_threads((int)n);
    const size_t shmem = (size_t)threads * sizeof(float);
    const dim3 grid((unsigned)m);
    int8_t* qp = static_cast<int8_t*>(out_qp);
    float* sp = static_cast<float*>(out_sp);
    switch (code) {
        case cx::kBF16:
            kv_quant_kernel<uint16_t><<<grid, threads, shmem, g_cx_stream>>>(
                reinterpret_cast<const uint16_t*>(xp), qp, sp, (int)n, p);
            break;
        case cx::kF32:
            kv_quant_kernel<float><<<grid, threads, shmem, g_cx_stream>>>(
                static_cast<const float*>(xp), qp, sp, (int)n, p);
            break;
        default:
            return cx::fail(err, "unsupported dtype code %d (bf16 or fp32 only)", code);
    }
    return cx::finish(err);
}

}  // extern "C"
