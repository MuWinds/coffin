// @file delta_prologue_abi.cu
// @brief 线性注意力头前处理的三个 C ABI 出口：``coffin_delta_prologue`` /
//        ``_batch`` / ``coffin_delta_prep_seq``，无 torch。
//
// Module: coffin/backends/corex/kernels/abi。设备代码来自
// ``../delta_prologue_device.inc`` 与 ``../delta_normed_device.inc``——与
// ``../linear_attn.cu`` 是同一份内核（加 ``head_threads`` / ``norm_threads``），所以
// 两条前端逐位相同。
//
// 五个出口在**同一个翻译单元**里：两条 ``.inc`` 都定义了 ``prologue_kernel`` /
// ``delta_prep_seq_kernel`` / ``prologue_batch_kernel``，分成两个 ``.cu`` 会在链接期
// 撞 ``__device_stub__*`` 的重复定义（作业 5960 就是这么挂的）。搬走哪一批都要保住
// 「一份 include、一个 TU」。
//
// 三个出口只差「行轴在哪」：单 token 的网格是 ``nv``，批处理是 ``(nv, rows)``，分块预
// 填充是 ``(nv, T)`` 并写出分块扫描要读的头主序布局。
//
// Constraints:
//   * 签名与 ``cxrt/attn.py`` 的 ``argtypes`` 必须逐字一致：多一个参数就整体错位。
//   * ``delta_prep_seq`` 的 ``qkv`` / ``b`` / ``a`` 必须是**最内层步长 1** 的视图（列切片），
//     行间距单独传参；其余操作数按稠密地址算。
//   * ``beta`` 是写强度、``g`` 是对数衰减：两处都要走内核里的 ``softplus`` / ``-exp`` 口径。
//   * ``head_threads(dk)`` 在主机侧算块大小，与内核里的归约伙伴必须同一个值。

#include "cx_tensor.h"

#include <cstdint>

#include "../delta_prologue_device.inc"
#include "../delta_normed_device.inc"

// @brief 单 token 的 q/k 归一化、头扩展与 beta/g 门。
//
// @param q [num_k, dk] fp32。
// @param k [num_k, dk] fp32。
// @param v [num_v, dv] fp32。
// @param b [num_v] bf16 beta logits。
// @param a [num_v] bf16 衰减 logits。
// @param neg_expA [num_v] fp32。
// @param dt_bias [num_v] fp32。
// @param q_out [num_v, dk] fp32。
// @param k_out [num_v, dk] fp32。
// @param v_out [num_v, dv] fp32。
// @param beta [num_v] fp32。
// @param g [num_v] fp32。
// @param qscale ``dk ** -0.5``。
// @param rep 每个键头对应的值头数。
// @param nv 值头数。
// @param dk 键维度。
// @param dv 值维度。
// @param stream 目标流；空指针表示默认流。
// @param err 错误缓冲，可为空。
// @param errlen 错误缓冲字节数。
// @return ``0`` 成功。
extern "C" int coffin_delta_prologue(const void* q, const void* k, const void* v, const void* b,
                                     const void* a, const void* neg_expA, const void* dt_bias,
                                     void* q_out, void* k_out, void* v_out, void* beta, void* g,
                                     double qscale, int rep, int nv, int dk, int dv,
                                     void* stream, char* err, int errlen) {
    const cx::Err e{err, errlen};
    if (!q || !k || !v || !b || !a || !neg_expA || !dt_bias || !q_out || !k_out || !v_out ||
        !beta || !g) {
        return cx::fail(e, "all prologue operands must be non-null");
    }
    if (nv <= 0 || dk <= 0 || dv <= 0 || rep <= 0) {
        return cx::fail(e, "nv/dk/dv/rep must be positive, got %d/%d/%d/%d", nv, dk, dv, rep);
    }
    const int threads = head_threads(dk);
    const size_t smem = (size_t)2 * threads * sizeof(float);
    const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    prologue_kernel<<<nv, threads, smem, s>>>(
        reinterpret_cast<const float*>(q), reinterpret_cast<const float*>(k),
        reinterpret_cast<const float*>(v), reinterpret_cast<const uint16_t*>(b),
        reinterpret_cast<const uint16_t*>(a), reinterpret_cast<const float*>(neg_expA),
        reinterpret_cast<const float*>(dt_bias), reinterpret_cast<float*>(q_out),
        reinterpret_cast<float*>(k_out), reinterpret_cast<float*>(v_out),
        reinterpret_cast<float*>(beta), reinterpret_cast<float*>(g), (float)qscale, dk, dv, rep);
    return cx::finish(e);
}

// @brief 批处理版的同一个前处理，``rows`` 是网格的 y 轴。
//
// @param rows 批行数。
// 其余参数同 :func:`coffin_delta_prologue`，只是各自多一个前导行维度。
// @return ``0`` 成功。
extern "C" int coffin_delta_prologue_batch(const void* q, const void* k, const void* v,
                                           const void* b, const void* a, const void* neg_expA,
                                           const void* dt_bias, void* q_out, void* k_out,
                                           void* v_out, void* beta, void* g, double qscale,
                                           int rep, int rows, int nv, int dk, int dv,
                                           void* stream, char* err, int errlen) {
    const cx::Err e{err, errlen};
    if (!q || !k || !v || !b || !a || !neg_expA || !dt_bias || !q_out || !k_out || !v_out ||
        !beta || !g) {
        return cx::fail(e, "all prologue operands must be non-null");
    }
    if (rows <= 0 || nv <= 0 || dk <= 0 || dv <= 0 || rep <= 0) {
        return cx::fail(e, "rows/nv/dk/dv/rep must be positive, got %d/%d/%d/%d/%d", rows, nv,
                        dk, dv, rep);
    }
    const int threads = head_threads(dk);
    const size_t smem = (size_t)2 * threads * sizeof(float);
    const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    prologue_batch_kernel<<<dim3(nv, rows), threads, smem, s>>>(
        reinterpret_cast<const float*>(q), reinterpret_cast<const float*>(k),
        reinterpret_cast<const float*>(v), reinterpret_cast<const uint16_t*>(b),
        reinterpret_cast<const uint16_t*>(a), reinterpret_cast<const float*>(neg_expA),
        reinterpret_cast<const float*>(dt_bias), reinterpret_cast<float*>(q_out),
        reinterpret_cast<float*>(k_out), reinterpret_cast<float*>(v_out),
        reinterpret_cast<float*>(beta), reinterpret_cast<float*>(g), (float)qscale, nv, dk, dv,
        rep);
    return cx::finish(e);
}

// @brief 预填充的分块头前处理，写出头主序 ``[H, T, d]`` 布局。
//
// @param qkv [T, qkv_cols] fp32，最内层步长 1。
// @param b [T, num_v] bf16，最内层步长 1，行间距 ``b_pitch``。
// @param a [T, num_v] bf16，同上。
// @param neg_expA [num_v] fp32。
// @param dt_bias [num_v] fp32。
// @param q_out [nv, T, dk] fp32。
// @param k_out [nv, T, dk] fp32。
// @param v_out [nv, T, dv] fp32。
// @param beta [nv, T] fp32。
// @param g [nv, T] fp32。
// @param qscale ``dk ** -0.5``。
// @param rep 每个键头对应的值头数。
// @param T 分块长度。
// @param nv 值头数。
// @param dk 键维度。
// @param dv 值维度。
// @param kdim ``num_key_heads * dk``。
// @param qkv_cols ``qkv`` 的列数。
// @param b_pitch ``b`` 的行间距（元素）。
// @param a_pitch ``a`` 的行间距（元素）。
// @param stream 目标流。
// @param err 错误缓冲，可为空。
// @param errlen 错误缓冲字节数。
// @return ``0`` 成功。
extern "C" int coffin_delta_prep_seq(const void* qkv, const void* b, const void* a,
                                     const void* neg_expA, const void* dt_bias, void* q_out,
                                     void* k_out, void* v_out, void* beta, void* g, double qscale,
                                     int rep, int T, int nv, int dk, int dv, int kdim,
                                     int qkv_cols, int b_pitch, int a_pitch, void* stream,
                                     char* err, int errlen) {
    const cx::Err e{err, errlen};
    if (!qkv || !b || !a || !neg_expA || !dt_bias || !q_out || !k_out || !v_out || !beta || !g) {
        return cx::fail(e, "all prep operands must be non-null");
    }
    if (T <= 0 || nv <= 0 || dk <= 0 || dv <= 0 || rep <= 0 || kdim <= 0) {
        return cx::fail(e, "T/nv/dk/dv/rep/kdim must be positive, got %d/%d/%d/%d/%d/%d", T, nv,
                        dk, dv, rep, kdim);
    }
    if (qkv_cols < kdim + 2 * nv) {
        return cx::fail(e, "qkv holds %d columns, need at least %d (kdim + 2 * nv)", qkv_cols,
                        kdim + 2 * nv);
    }
    const int threads = head_threads(dk);
    const size_t smem = (size_t)2 * threads * sizeof(float);
    const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    delta_prep_seq_kernel<<<dim3(nv, T), threads, smem, s>>>(
        reinterpret_cast<const float*>(qkv), reinterpret_cast<const uint16_t*>(b),
        reinterpret_cast<const uint16_t*>(a), reinterpret_cast<const float*>(neg_expA),
        reinterpret_cast<const float*>(dt_bias), reinterpret_cast<float*>(q_out),
        reinterpret_cast<float*>(k_out), reinterpret_cast<float*>(v_out),
        reinterpret_cast<float*>(beta), reinterpret_cast<float*>(g), (float)qscale, T, dk, dv,
        rep, kdim, qkv_cols, b_pitch, a_pitch);
    return cx::finish(e);
}

namespace {

// @brief 两个 dtype 码翻成启动器要的那对 bool。
//
// 只在主机侧用：四种实例化由 ``out`` 与 ``z`` 的存储类型各选一次决定，与算术无关
// （门无论哪种形式都会在内核里舍入成 bf16）。
int decode_pair(int out_code, int z_code, bool* out_bf16, bool* z_bf16, const cx::Err& e) {
    if (out_code != cx::kF32 && out_code != cx::kBF16) {
        return cx::fail(e, "out dtype code %d is neither fp32 nor bf16", out_code);
    }
    if (z_code != cx::kF32 && z_code != cx::kBF16) {
        return cx::fail(e, "norm gate dtype code %d is neither fp32 nor bf16", z_code);
    }
    *out_bf16 = out_code == cx::kBF16;
    *z_bf16 = z_code == cx::kBF16;
    return 0;
}

// @brief 被替代那条链固定下来的块大小；内核里的归约伙伴必须同一个值。
int block_of(int dk, int dv, int* threads, const cx::Err& e) {
    const int t = head_threads(dk);
    if (dk > t || dv > t) {
        return cx::fail(e, "dk and dv must fit one block (dk=%d dv=%d threads=%d)", dk, dv, t);
    }
    if (norm_threads(dv) != t) {
        return cx::fail(e, "prologue block %d != gated-norm block %d (dk=%d dv=%d)", t,
                        norm_threads(dv), dk, dv);
    }
    *threads = t;
    return 0;
}

}  // namespace

// @brief 单解码 token 的融合链：头预处理、秩一递推、门控输出归一化。
//
// @param q [nv / rep, dk] fp32。
// @param k [nv / rep, dk] fp32。
// @param v [nv, dv] fp32。
// @param b [nv] bf16 beta logits。
// @param a [nv] bf16 衰减 logits。
// @param neg_expA [nv] fp32。
// @param dt_bias [nv] fp32。
// @param z [nv, dv] 门投影，dtype 由 ``z_code`` 给。
// @param nw [dv] bf16 门控归一化增益。
// @param st [nv, dk, dv] fp32 状态，原地更新。
// @param out [nv, dv]，dtype 由 ``out_code`` 给。
// @param qscale ``dk ** -0.5``。
// @param rep 每个键头对应的值头数。
// @param nv 值头数。
// @param dk 键维度。
// @param dv 值维度。
// @param out_code 目的地的 dtype 码（``kF32`` / ``kBF16``）。
// @param z_code 门的 dtype 码（``kF32`` / ``kBF16``）。
// @param record 可为空；非空时是 ``[record_width, nv, dk + dv + 1]`` fp32。
// @param record_slot 本 token 在记录窗口中的位置。
// @param record_width 记录窗口长度。
// @param stream 目标流；空指针表示默认流。
// @param err 错误缓冲，可为空。
// @param errlen 错误缓冲字节数。
// @return ``0`` 成功。
extern "C" int coffin_delta_normed(const void* q, const void* k, const void* v, const void* b,
                                   const void* a, const void* neg_expA, const void* dt_bias,
                                   const void* z, const void* nw, void* st, void* out,
                                   double qscale, int rep, int nv, int dk, int dv, int out_code,
                                   int z_code, void* record, int record_slot, int record_width,
                                   void* stream, char* err, int errlen) {
    const cx::Err e{err, errlen};
    if (!q || !k || !v || !b || !a || !neg_expA || !dt_bias || !z || !nw || !st || !out) {
        return cx::fail(e, "all normed operands must be non-null");
    }
    if (nv <= 0 || dk <= 0 || dv <= 0 || rep <= 0 || nv % rep != 0) {
        return cx::fail(e, "nv/dk/dv/rep must be positive and nv divisible by rep, got %d/%d/%d/%d",
                        nv, dk, dv, rep);
    }
    bool out_bf16 = false;
    bool z_bf16 = false;
    int rc = decode_pair(out_code, z_code, &out_bf16, &z_bf16, e);
    if (rc != 0) return rc;
    int threads = 0;
    rc = block_of(dk, dv, &threads, e);
    if (rc != 0) return rc;
    if (record != nullptr && (record_slot < 0 || record_slot >= record_width)) {
        return cx::fail(e, "record slot %d is outside the window of %d", record_slot,
                        record_width);
    }
    launch_normed(reinterpret_cast<const float*>(q), reinterpret_cast<const float*>(k),
                  reinterpret_cast<const float*>(v), reinterpret_cast<const uint16_t*>(b),
                  reinterpret_cast<const uint16_t*>(a), reinterpret_cast<const float*>(neg_expA),
                  reinterpret_cast<const float*>(dt_bias), z, z_bf16,
                  reinterpret_cast<const uint16_t*>(nw), reinterpret_cast<float*>(st), out,
                  out_bf16, record == nullptr ? nullptr : reinterpret_cast<float*>(record),
                  record_slot, (float)qscale, nv, dk, dv, rep, threads,
                  reinterpret_cast<cudaStream_t>(stream));
    return cx::finish(e);
}

// @brief c 个校验 token 的同一条链，一次启动跑完整个窗口。
//
// @param steps 本窗口的 token 数。
// @param stride_q,stride_k,stride_v,stride_b,stride_z 每个 token 的行距（元素数）；行距
//        可以大于行宽，因为调用方给的是融合投影 ``split`` 出来的视图。
// 其余参数同 :func:`coffin_delta_normed`，只是 ``q``/``k``/``v``/``b``/``a``/``z``/``out``
// 各多一个前导 step 维度。
// @return ``0`` 成功。
extern "C" int coffin_delta_normed_seq(const void* q, const void* k, const void* v, const void* b,
                                       const void* a, const void* neg_expA, const void* dt_bias,
                                       const void* z, const void* nw, void* st, void* out,
                                       double qscale, int rep, int nv, int dk, int dv,
                                       int out_code, int z_code, int steps, long long stride_q,
                                       long long stride_k, long long stride_v, long long stride_b,
                                       long long stride_z, void* record, int record_slot,
                                       int record_width, void* stream, char* err, int errlen) {
    const cx::Err e{err, errlen};
    if (!q || !k || !v || !b || !a || !neg_expA || !dt_bias || !z || !nw || !st || !out) {
        return cx::fail(e, "all normed-seq operands must be non-null");
    }
    if (nv <= 0 || dk <= 0 || dv <= 0 || rep <= 0 || nv % rep != 0) {
        return cx::fail(e, "nv/dk/dv/rep must be positive and nv divisible by rep, got %d/%d/%d/%d",
                        nv, dk, dv, rep);
    }
    if (steps < 1) return cx::fail(e, "steps must be >= 1, got %d", steps);
    bool out_bf16 = false;
    bool z_bf16 = false;
    int rc = decode_pair(out_code, z_code, &out_bf16, &z_bf16, e);
    if (rc != 0) return rc;
    int threads = 0;
    rc = block_of(dk, dv, &threads, e);
    if (rc != 0) return rc;
    if (record != nullptr && (record_slot < 0 || record_slot + steps > record_width)) {
        return cx::fail(e, "record window %d cannot hold steps %d from slot %d", record_width,
                        steps, record_slot);
    }
    launch_normed_seq(reinterpret_cast<const float*>(q), reinterpret_cast<const float*>(k),
                      reinterpret_cast<const float*>(v), reinterpret_cast<const uint16_t*>(b),
                      reinterpret_cast<const uint16_t*>(a),
                      reinterpret_cast<const float*>(neg_expA),
                      reinterpret_cast<const float*>(dt_bias), z, z_bf16,
                      reinterpret_cast<const uint16_t*>(nw), reinterpret_cast<float*>(st), out,
                      out_bf16, record == nullptr ? nullptr : reinterpret_cast<float*>(record),
                      record_slot, (float)qscale, nv, dk, dv, rep, steps, stride_q, stride_k,
                      stride_v, stride_b, stride_z, threads,
                      reinterpret_cast<cudaStream_t>(stream));
    return cx::finish(e);
}
