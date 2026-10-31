"""内核级等价性：每个融合内核都必须复现它替换掉的那套数学运算。

Module: ``coffin/tests``。直接 import ``ops``、``hip_ops`` 与
``int8_weights``，从不加载 checkpoint，所以它是内核本身最廉价的一层回归网；
模型级测试在这层之上再叠。

Constraints:
    * 断言是相对误差界，不是逐位相等：内核的累加顺序不同，输出又是 bf16，
      对齐的参考是 eager 的 float 实现。每个测试的容差取自上一轮复核过的实测
      误差加余量，收紧它就等于重新翻案一次测量。
    * 跨步输入属于契约，不是边缘情况：模型传给这些内核的是更宽融合缓冲区的
      列切片，所以假定行连续的内核能通过简单用例，却会把模型算坏。
    * 每个内核只测一组形状，且贴着模型的真实尺寸：最大的 GEMV 形状
      （151936 x 2560）正是子测试之间不释放缓冲区就会撑爆卡的那个。
"""

from __future__ import annotations

import unittest

import support

import torch
import torch.nn.functional as F

NEG_INF = float("-inf")


def _depthwise_conv_eager(conv_w, x, state):
    """``conv1d_silu_seq`` 所替换的 eager 链路（卷积核宽 4，state 为前 3 个 token）。

    Args:
        conv_w: 每通道的抽头，``[C, 4]``；第四个抽头对应当前 token。
        x: 待消费的 token，``[T, C]``。
        state: 之前三个 token，``[C, 3]``，最旧的在前。

    Returns:
        ``(output [T, C], new_state [C, 3])``。新 state 是 ``state`` 的最后
        三行拼在 ``x`` 之前，内核跨块边界携带的正是它。
    """
    seq = torch.cat([state.t(), x.float()], dim=0)
    new_state = seq[-3:].t().contiguous()
    win = seq.unfold(0, 4, 1)
    return F.silu((win * conv_w).sum(-1)), new_state


@support.needs_gpu
class ChunkedDeltaRuleTest(unittest.TestCase):
    """``ops.delta.chunked_delta`` 必须与逐 token 的递推等价。"""

    H, DK, DV = 8, 64, 64

    def _inputs(self, C, seed=0):
        """按模型真实的头形状造随机 delta-rule 输入，q 已预先缩放。"""
        torch.manual_seed(seed)
        q = torch.randn(self.H, C, self.DK, device="cuda")
        k = torch.randn(self.H, C, self.DK, device="cuda")
        v = torch.randn(self.H, C, self.DV, device="cuda")
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6) * self.DK ** -0.5
        k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        beta = torch.rand(self.H, C, device="cuda")
        g = -torch.rand(self.H, C, device="cuda") * 0.5
        st0 = torch.randn(self.H, self.DK, self.DV, device="cuda") * 0.01
        return q, k, v, g, beta, st0

    @staticmethod
    def _recurrent(q, k, v, g, beta, state):
        """逐 token 递推：``chunked_delta`` 优化的正是这个定义。"""
        outs = []
        for t in range(q.shape[1]):
            st = state * g[:, t].exp()[:, None, None]
            kv = (st * k[:, t, :, None]).sum(-2)
            d = (v[:, t] - kv) * beta[:, t, None]
            st = st + k[:, t, :, None] * d[:, None, :]
            state = st
            outs.append((st * q[:, t, :, None]).sum(-2))
        return torch.stack(outs, 1), state

    def test_matches_recurrent(self):
        """分块只是优化：每个输出与最终 state 都必须对上递推的结果。"""
        from ops.delta import chunked_delta

        # 实测约 1e-6；1e-3 的余量留给重排求和带来的 fp32 噪声。
        for C in (256, 200, 64, 65):          # 200/65 走的是补零路径
            q, k, v, g, beta, st0 = self._inputs(C)
            ref, ref_st = self._recurrent(q, k, v, g, beta, st0.clone())
            for chunk in (16, 32, 64, 128):
                st_out = torch.empty_like(st0)
                got = chunked_delta(q, k, v, g, beta, st0.clone(), st_out, chunk)
                got = got.transpose(0, 1)     # [C, H, dv] -> [H, C, dv]
                with self.subTest(C=C, chunk=chunk):
                    self.assertLess(support.rel_err(got, ref), 1e-3)
                    self.assertLess(support.rel_err(st_out, ref_st), 1e-3)

    def test_zero_state_and_no_decay_is_plain_causal_linear_attention(self):
        """g=0 且 beta=1 时该规则退化为因果线性注意力。

        随机用例能藏住的符号或掩码错误，在这个退化情形下会直接暴露。
        """
        from ops.delta import chunked_delta

        q, k, v, _, _, st0 = self._inputs(200)
        g = torch.zeros(self.H, 200, device="cuda")
        beta = torch.ones(self.H, 200, device="cuda")
        state = torch.zeros_like(st0)
        ref, _ = self._recurrent(q, k, v, g, beta, state.clone())
        st_out = torch.empty_like(state)
        got = chunked_delta(q, k, v, g, beta, state.clone(), st_out, 64)
        self.assertLess(support.rel_err(got.transpose(0, 1), ref), 1e-3)


@support.needs_hip
class ConvSeqTest(unittest.TestCase):
    """``bops.conv1d_silu_seq`` 必须复现 conv+SiLU，连带 state 一起。"""

    C = 8192

    def setUp(self):
        """权重固定，只凭 seed 就能复现一次失败。"""
        torch.manual_seed(0)
        self.conv_w = (torch.randn(self.C, 4, device="cuda") * 0.3).contiguous()

    def _check(self, x, st0):
        """把内核的输出与携带的 state 同 eager 链路逐项比对。"""
        from backends.hip import shim as hip_ops
        ref, ref_st = _depthwise_conv_eager(self.conv_w, x, st0.clone())
        got_st = st0.clone()
        got = bops.conv1d_silu_seq(self.conv_w, x, got_st)
        # 实测：state 逐位相同，输出约 1e-7。1e-5 的界来自输出 bf16 量化，
        # 不是累加误差。
        self.assertLess(support.rel_err(got, ref), 1e-5)
        self.assertLess(support.rel_err(got_st, ref_st), 1e-5)

    def test_matches_eager(self):
        """长度覆盖短于 4 宽窗口、恰好等于、以及跨过块边界三者。"""
        for T in (1, 2, 3, 5, 64, 257):
            with self.subTest(T=T):
                x = (torch.randn(T, self.C, device="cuda") * 0.5).to(torch.bfloat16)
                st0 = torch.randn(self.C, 3, device="cuda")
                self._check(x, st0)

    def test_honours_row_pitch_of_a_column_slice(self):
        """假定行连续的内核能通过朴素用例，接着就会读错模型的融合缓冲区。"""
        # 模型交过来的是融合 GEMV 输出的列切片：行距比 C 宽，内核必须读 ldx，
        # 不能假定行距就是 C。
        for T in (7, 200):
            with self.subTest(T=T):
                packed = (torch.randn(T, self.C + 64, device="cuda") * 0.5).to(torch.bfloat16)
                xs = packed[:, :self.C]
                self.assertEqual(xs.stride(0), self.C + 64)
                self.assertFalse(xs.is_contiguous())
                self._check(xs, torch.randn(self.C, 3, device="cuda"))

    def test_chunk_boundaries_carry_state(self):
        """分块预填充不能在块边界上把卷积的 3 token 历史重新起头。"""
        from backends.hip import shim as hip_ops
        T = 300
        x = (torch.randn(T, self.C, device="cuda") * 0.5).to(torch.bfloat16)
        st0 = torch.randn(self.C, 3, device="cuda")
        ref, ref_st = _depthwise_conv_eager(self.conv_w, x, st0.clone())
        st = st0.clone()
        parts = [bops.conv1d_silu_seq(self.conv_w, x[:100], st),
                 bops.conv1d_silu_seq(self.conv_w, x[100:180], st),
                 bops.conv1d_silu_seq(self.conv_w, x[180:], st)]
        got = torch.cat(parts, dim=0)
        self.assertLess(support.rel_err(got, ref), 1e-5)
        self.assertLess(support.rel_err(st, ref_st), 1e-5)


@support.needs_hip
class PrepSeqTest(unittest.TestCase):
    """``bops.delta_prep_seq`` 必须复现 q/k 归一化、头扩展与 beta/g。"""

    def test_matches_eager_maths_and_layout(self):
        """prep 内核同时把布局定死。

        输出一转置或跨步错位，就会把错值悄悄喂给 delta 内核。
        """
        from backends.hip import shim as hip_ops

        torch.manual_seed(0)
        nv, nk, dk, dv = 32, 16, 128, 128
        rep = nv // nk
        kdim, vdim = nk * dk, nv * dv
        conv_dim = 2 * kdim + vdim
        neg_expA = -torch.rand(nv, device="cuda") - 0.5
        dt_bias = torch.rand(nv, device="cuda") * 0.5
        qscale = dk ** -0.5
        for T in (1, 5, 64, 300):
            for pad in (0, 64):              # pad>0 表示 b/a 是跨步的列切片
                with self.subTest(T=T, strided=bool(pad)):
                    # 先转 bf16 再切片：反过来的话 .to() 会把它变回连续，
                    # 内核必须遵守的行距就丢了。
                    pb = (torch.randn(T, nv + pad, device="cuda") * 0.5).to(torch.bfloat16)
                    pa = (torch.randn(T, nv + pad, device="cuda") * 0.5).to(torch.bfloat16)
                    b, a = pb[:, :nv], pa[:, :nv]
                    self.assertEqual(b.stride(0), nv + pad)
                    qkv = torch.randn(T, conv_dim, device="cuda") * 0.5

                    qt = torch.empty(nv, T, dk, device="cuda")
                    kt = torch.empty(nv, T, dk, device="cuda")
                    vt = torch.empty(nv, T, dv, device="cuda")
                    bt = torch.empty(nv, T, device="cuda")
                    gt = torch.empty(nv, T, device="cuda")
                    bops.delta_prep_seq(qkv, b, a, neg_expA, dt_bias,
                                           qt, kt, vt, bt, gt, qscale, rep)

                    qe = qkv[:, :kdim].reshape(T, nk, dk).float()
                    ke = qkv[:, kdim:2 * kdim].reshape(T, nk, dk).float()
                    ve = qkv[:, 2 * kdim:].reshape(T, nv, dv).float()
                    qe = qe.repeat_interleave(rep, dim=1)
                    ke = ke.repeat_interleave(rep, dim=1)
                    qe = qe * torch.rsqrt((qe * qe).sum(-1, keepdim=True) + 1e-6) * qscale
                    ke = ke * torch.rsqrt((ke * ke).sum(-1, keepdim=True) + 1e-6)
                    be = b.float().sigmoid()
                    ge = neg_expA * F.softplus(a.float() + dt_bias)

                    for got, ref in ((qt.permute(1, 0, 2), qe),
                                     (kt.permute(1, 0, 2), ke),
                                     (vt.permute(1, 0, 2), ve),
                                     (bt.permute(1, 0), be),
                                     (gt.permute(1, 0), ge)):
                        self.assertLess(support.rel_err(got, ref), 1e-5)


@support.needs_hip
class F16GemvTest(unittest.TestCase):
    """fp16 权重的 GEMV 必须与 float 参考一致，并输出 bf16。"""

    SHAPES = ((2560, 2560), (8192, 2560), (2560, 4096),
              (18432, 2560), (151936, 2560))

    def test_single_vector_matches_float_reference(self):
        """GEMV 是解码热路径，因此按模型真实的权重形状钉住。

        2e-2 的界覆盖的是 fp16 存储，不是内核误差。
        """
        from backends.hip import shim as hip_ops
        for N, K in self.SHAPES:
            with self.subTest(N=N, K=K):
                wb = (torch.randn(N, K, dtype=torch.float32, device="cuda") * 0.02
                      ).to(torch.bfloat16)
                w16 = wb.to(torch.float16)
                x = (torch.randn(K, dtype=torch.float32, device="cuda") * 0.1
                     ).to(torch.bfloat16)
                ref = x.float() @ w16.float().t()
                y = hip_ops.gemv(w16, x, 1, 2)
                self.assertEqual(y.dtype, torch.bfloat16)
                self.assertLess(support.rel_err(y.float(), ref), 2e-2)
                # 151936 宽的那个形状约 1.5 GB 权重；形状之间释放掉，整个测试类
                # 才装得进卡里。
                del wb, w16, x, ref, y
                torch.cuda.empty_cache()

    def test_batched_small_m_shapes(self):
        """批量入口是另一个内核：数学相同，切块方式不同。"""
        from backends.hip import shim as hip_ops
        for N, K in self.SHAPES[:3]:
            wb = (torch.randn(N, K, dtype=torch.float32, device="cuda") * 0.02
                  ).to(torch.bfloat16)
            w16 = wb.to(torch.float16)
            for M in (2, 4, 8):
                with self.subTest(N=N, K=K, M=M):
                    X = (torch.randn(M, K, dtype=torch.float32, device="cuda") * 0.1
                         ).to(torch.bfloat16)
                    ref = X.float() @ w16.float().t()
                    y = hip_ops.gemv_m(w16, X, 2)
                    self.assertEqual(tuple(y.shape), (M, N))
                    self.assertEqual(y.dtype, torch.bfloat16)
                    self.assertLess(support.rel_err(y.float(), ref), 2e-2)
            del wb, w16
            torch.cuda.empty_cache()


@support.needs_hip
class Int8GemvTest(unittest.TestCase):
    """W8A16 GEMV：对准反量化后的权重比较，把内核误差单独隔出来。"""

    SHAPES = ((2560, 2560), (8192, 2560), (4096, 4096),
              (2560, 4096), (151936, 2560))

    def test_single_vector_tracks_dequantised_reference(self):
        """两个界各有用途。

        对反量化权重取 1e-2 隔出内核误差，对 float 权重取 2e-2 卡住量化本身的
        误差。
        """
        from backends.hip import shim as hip_ops
        from int8_weights import QWeight
        for N, K in self.SHAPES:
            with self.subTest(N=N, K=K):
                w = (torch.randn(N, K, dtype=torch.float32, device="cuda") * 0.02
                     ).to(torch.bfloat16)
                x = torch.randn(K, dtype=torch.float32, device="cuda").to(torch.bfloat16)
                q = QWeight.quantize(w)
                ref = x.float() @ w.float().t()             # 含量化误差
                dq = x.float() @ q.dequant().float().t()    # 只剩内核误差
                y = hip_ops.gemv_w8a16(q.wq, q.scale, x, 1, 2).float()
                self.assertLess(support.rel_err(y, dq), 1e-2)
                self.assertLess(support.rel_err(y, ref), 2e-2)
                del w, x, q, ref, dq, y
                torch.cuda.empty_cache()

    def test_batched_small_m_shapes(self):
        """批量 INT8 路径是独立内核，scale 广播也自己一份。

        单向量用例覆盖不到它。
        """
        from backends.hip import shim as hip_ops
        from int8_weights import QWeight
        for N, K in self.SHAPES[:3]:
            w = (torch.randn(N, K, dtype=torch.float32, device="cuda") * 0.02
                 ).to(torch.bfloat16)
            q = QWeight.quantize(w)
            for M in (2, 4, 8):
                with self.subTest(N=N, K=K, M=M):
                    X = torch.randn(M, K, dtype=torch.float32, device="cuda").to(torch.bfloat16)
                    dq = X.float() @ q.dequant().float().t()
                    y = hip_ops.gemv_w8a16_m(q.wq, q.scale, X, 2).float()
                    self.assertLess(support.rel_err(y, dq), 1e-2)
            del w, q
            torch.cuda.empty_cache()
