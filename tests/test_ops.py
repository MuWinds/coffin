"""Kernel-level functional checks (no model, no checkpoint).

Each test is one property the engine depends on, ported from the old
``diag/check_*.py`` scripts: the fused kernels must reproduce the eager maths
they replaced, the chunked delta rule must reproduce the recurrent one, and the
int8/fp16 GEMV kernels must reproduce a float reference.

Thresholds are the ones those scripts used; the REL_* values in the comments are
what the last vetted run measured.
"""

from __future__ import annotations

import unittest

import support

import torch
import torch.nn.functional as F

NEG_INF = float("-inf")


def _depthwise_conv_eager(conv_w, x, state):
    """The eager chain ``conv1d_silu_seq`` replaces (kernel 4, state = last 3)."""
    seq = torch.cat([state.t(), x.float()], dim=0)
    new_state = seq[-3:].t().contiguous()
    win = seq.unfold(0, 4, 1)
    return F.silu((win * conv_w).sum(-1)), new_state


@support.needs_gpu
class ChunkedDeltaRuleTest(unittest.TestCase):
    """``ops.delta.chunked_delta`` must equal the per-token recurrence."""

    H, DK, DV = 8, 64, 64

    def _inputs(self, C, seed=0):
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
        from ops.delta import chunked_delta

        # Measured ~1e-6; 1e-3 leaves room for bf16-free fp32 noise.
        for C in (256, 200, 64, 65):          # 200/65 exercise the padding path
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
    """``hip_ops.conv1d_silu_seq`` must reproduce conv+SiLU, state included."""

    C = 8192

    def setUp(self):
        torch.manual_seed(0)
        self.conv_w = (torch.randn(self.C, 4, device="cuda") * 0.3).contiguous()

    def _check(self, x, st0):
        import hip_ops
        ref, ref_st = _depthwise_conv_eager(self.conv_w, x, st0.clone())
        got_st = st0.clone()
        got = hip_ops.conv1d_silu_seq(self.conv_w, x, got_st)
        # Measured: state bit-identical, output ~1e-7.
        self.assertLess(support.rel_err(got, ref), 1e-5)
        self.assertLess(support.rel_err(got_st, ref_st), 1e-5)

    def test_matches_eager(self):
        for T in (1, 2, 3, 5, 64, 257):
            with self.subTest(T=T):
                x = (torch.randn(T, self.C, device="cuda") * 0.5).to(torch.bfloat16)
                st0 = torch.randn(self.C, 3, device="cuda")
                self._check(x, st0)

    def test_honours_row_pitch_of_a_column_slice(self):
        # The model hands over a column slice of the fused GEMV output: rows are
        # pitched wider than C, and the kernel must read ldx, not assume C.
        for T in (7, 200):
            with self.subTest(T=T):
                packed = (torch.randn(T, self.C + 64, device="cuda") * 0.5).to(torch.bfloat16)
                xs = packed[:, :self.C]
                self.assertEqual(xs.stride(0), self.C + 64)
                self.assertFalse(xs.is_contiguous())
                self._check(xs, torch.randn(self.C, 3, device="cuda"))

    def test_chunk_boundaries_carry_state(self):
        import hip_ops
        T = 300
        x = (torch.randn(T, self.C, device="cuda") * 0.5).to(torch.bfloat16)
        st0 = torch.randn(self.C, 3, device="cuda")
        ref, ref_st = _depthwise_conv_eager(self.conv_w, x, st0.clone())
        st = st0.clone()
        parts = [hip_ops.conv1d_silu_seq(self.conv_w, x[:100], st),
                 hip_ops.conv1d_silu_seq(self.conv_w, x[100:180], st),
                 hip_ops.conv1d_silu_seq(self.conv_w, x[180:], st)]
        got = torch.cat(parts, dim=0)
        self.assertLess(support.rel_err(got, ref), 1e-5)
        self.assertLess(support.rel_err(st, ref_st), 1e-5)


@support.needs_hip
class PrepSeqTest(unittest.TestCase):
    """``hip_ops.delta_prep_seq`` must reproduce q/k norm + head expand + beta/g."""

    def test_matches_eager_maths_and_layout(self):
        import hip_ops

        torch.manual_seed(0)
        nv, nk, dk, dv = 32, 16, 128, 128
        rep = nv // nk
        kdim, vdim = nk * dk, nv * dv
        conv_dim = 2 * kdim + vdim
        neg_expA = -torch.rand(nv, device="cuda") - 0.5
        dt_bias = torch.rand(nv, device="cuda") * 0.5
        qscale = dk ** -0.5
        for T in (1, 5, 64, 300):
            for pad in (0, 64):              # pad>0 = strided b/a (column slice)
                with self.subTest(T=T, strided=bool(pad)):
                    # bf16 *then* slice: .to() would make it contiguous again and
                    # lose the row pitch the kernel has to honour.
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
                    hip_ops.delta_prep_seq(qkv, b, a, neg_expA, dt_bias,
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
    """fp16-weight GEMV must agree with a float reference (and return bf16)."""

    SHAPES = ((2560, 2560), (8192, 2560), (2560, 4096),
              (18432, 2560), (151936, 2560))

    def test_single_vector_matches_float_reference(self):
        import hip_ops
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
                del wb, w16, x, ref, y
                torch.cuda.empty_cache()

    def test_batched_small_m_shapes(self):
        import hip_ops
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
    """W8A16 GEMV: kernel error is isolated by comparing against dequantised weights."""

    SHAPES = ((2560, 2560), (8192, 2560), (4096, 4096),
              (2560, 4096), (151936, 2560))

    def test_single_vector_tracks_dequantised_reference(self):
        import hip_ops
        from int8_weights import QWeight
        for N, K in self.SHAPES:
            with self.subTest(N=N, K=K):
                w = (torch.randn(N, K, dtype=torch.float32, device="cuda") * 0.02
                     ).to(torch.bfloat16)
                x = torch.randn(K, dtype=torch.float32, device="cuda").to(torch.bfloat16)
                q = QWeight.quantize(w)
                ref = x.float() @ w.float().t()             # includes quant error
                dq = x.float() @ q.dequant().float().t()    # kernel error only
                y = hip_ops.gemv_w8a16(q.wq, q.scale, x, 1, 2).float()
                self.assertLess(support.rel_err(y, dq), 1e-2)
                self.assertLess(support.rel_err(y, ref), 2e-2)
                del w, x, q, ref, dq, y
                torch.cuda.empty_cache()

    def test_batched_small_m_shapes(self):
        import hip_ops
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
