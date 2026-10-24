"""Gated DeltaNet linear attention: the mixer of 24 of the 32 layers.

All state (conv window, fp32 recurrent state) lives in fixed buffers updated in
place, so the decode step is graph-capturable.  Decode runs the fused prologue
and delta kernels; prefill uses the chunked form in ``ops.delta``; verify runs
the decode kernels per token and copies the state after each one, because a
rejected draft has to be rolled back.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import hip_ops
from models.qwen35.batch import batch_proj
from models.qwen35.tuning import CKPT
from ops.delta import chunked_delta
from ops.dtypes import D
from ops.gemv import _proj
from ops.norm import RMSNormGated


class LinearAttn:
    """Gated DeltaNet linear attention, single-token decode step.

    All state lives in fixed buffers updated in place so the step is
    graph-capturable.
    """

    def __init__(self, sd: dict, p: str, cfg: dict, bench: bool) -> None:
        self.num_v = cfg["linear_num_value_heads"]      # 32
        self.num_k = cfg["linear_num_key_heads"]        # 16
        self.dk = cfg["linear_key_head_dim"]            # 128
        self.dv = cfg["linear_value_head_dim"]          # 128
        self.rep = self.num_v // self.num_k             # 2
        self.kdim = self.dk * self.num_k                # 2048
        self.vdim = self.dv * self.num_v                # 4096
        self.conv_dim = self.kdim * 2 + self.vdim       # 8192
        self.kernel = cfg["linear_conv_kernel_dim"]     # 4

        self.qkvzb_a = _proj(sd, [f"{p}.in_proj_qkv", f"{p}.in_proj_z",
                                  f"{p}.in_proj_b", f"{p}.in_proj_a"], bench)
        self.out_proj = _proj(sd, [f"{p}.out_proj"], bench)
        self.norm = RMSNormGated(sd[f"{p}.norm.weight"])
        self.conv_w = sd[f"{p}.conv1d.weight"].reshape(self.conv_dim, self.kernel).float()
        self.A_log = sd[f"{p}.A_log"].float()
        self.dt_bias = sd[f"{p}.dt_bias"].float()
        self.neg_expA = -self.A_log.exp()
        self.qscale = self.dk ** -0.5

        # Per-row state and scratch for the batch path: the leading dim is the
        # batch row (capacity 1 = the single-stream engine, which indexes the
        # prefill row).  ``prefill_row`` is set by ``Qwen35.enable_batching`` to
        # the last row, which the single-stream prefill/verify paths use -- so a
        # prefill between decode steps cannot clobber a decode row's state.
        self._bcap = 1
        self.prefill_row = 0
        rows = 1
        self.conv_state = torch.zeros(rows, self.conv_dim, self.kernel - 1, dtype=torch.float32, device="cuda")
        self.conv_out = torch.zeros(rows, self.conv_dim, dtype=torch.float32, device="cuda")
        self.state = torch.zeros(rows, self.num_v, self.dk, self.dv, dtype=torch.float32, device="cuda")
        self.delta_out = torch.zeros(rows, self.num_v, self.dv, dtype=torch.float32, device="cuda")

        # fp32 scratch written by ``delta_prologue`` and read by ``delta_step``.
        self.q_pro = torch.zeros(rows, self.num_v, self.dk, dtype=torch.float32, device="cuda")
        self.k_pro = torch.zeros(rows, self.num_v, self.dk, dtype=torch.float32, device="cuda")
        self.v_pro = torch.zeros(rows, self.num_v, self.dv, dtype=torch.float32, device="cuda")
        self.beta = torch.zeros(rows, self.num_v, dtype=torch.float32, device="cuda")
        self.g = torch.zeros(rows, self.num_v, dtype=torch.float32, device="cuda")

        # per-token state snapshots for speculative-verify rollback (one slot per
        # verified token; the recurrent state cannot be rewound otherwise).
        self.ckpt_state = torch.zeros(CKPT, self.num_v, self.dk, self.dv, dtype=torch.float32, device="cuda")
        self.ckpt_conv = torch.zeros(CKPT, self.conv_dim, self.kernel - 1, dtype=torch.float32, device="cuda")
        # Batched speculative verify keeps one snapshot per (row, verified token);
        # allocated by ``enable_verify_batch`` only when batch speculative decoding
        # is switched on, because the fp32 state is 2 MB per layer per snapshot.
        self.ckpt_state_b: torch.Tensor | None = None
        self.ckpt_conv_b: torch.Tensor | None = None

    def enable_verify_batch(self, capacity: int, width: int) -> None:
        """Room for ``width`` rollback snapshots of every batch row."""
        if self.ckpt_state_b is not None and self.ckpt_state_b.shape[:2] == (capacity, width):
            return
        self.ckpt_state_b = torch.zeros(capacity, width, self.num_v, self.dk, self.dv,
                                        dtype=torch.float32, device="cuda")
        self.ckpt_conv_b = torch.zeros(capacity, width, self.conv_dim, self.kernel - 1,
                                       dtype=torch.float32, device="cuda")

    def snapshot_batch(self, rows: int, slot: int) -> None:
        self.ckpt_state_b[:rows, slot].copy_(self.state[:rows])
        self.ckpt_conv_b[:rows, slot].copy_(self.conv_state[:rows])

    def verify_batch_step(self, qkv: torch.Tensor, z: torch.Tensor,
                          b: torch.Tensor, a: torch.Tensor, slot: int) -> torch.Tensor:
        """One verified token for ``B`` rows, from already-projected qkv/z/b/a.

        Runs the same conv / prologue / delta kernels as the decode step (one
        batched launch each, so each row keeps the greedy numbers) and snapshots
        the recurrent state after the token -- ``slot`` is its index inside the
        verify width, i.e. the rollback point if the drafts up to here are
        accepted.
        """
        rows = qkv.shape[0]
        hip_ops.conv1d_silu_batch(self.conv_w, qkv, self.conv_state[:rows],
                                  self.conv_out[:rows])
        conv = self.conv_out[:rows]
        q = conv[:, :self.kdim].reshape(rows, self.num_k, self.dk)
        k = conv[:, self.kdim:2 * self.kdim].reshape(rows, self.num_k, self.dk)
        v = conv[:, 2 * self.kdim:].reshape(rows, self.num_v, self.dv)
        hip_ops.delta_prologue_batch(q, k, v, b.contiguous(), a.contiguous(),
                                     self.neg_expA, self.dt_bias,
                                     self.q_pro[:rows], self.k_pro[:rows],
                                     self.v_pro[:rows], self.beta[:rows],
                                     self.g[:rows], self.qscale, self.rep)
        hip_ops.delta_batch(self.q_pro[:rows], self.k_pro[:rows],
                            self.v_pro[:rows], self.beta[:rows], self.g[:rows],
                            self.state[:rows], self.delta_out[:rows])
        self.snapshot_batch(rows, slot)
        # One gated-norm launch for every row (the RMSNormGated entry flattens
        # (row, head) pairs internally); the per-row loop here cost ~8 launches
        # per token and dominated the one-forward verify.
        zz = z.reshape(rows, self.num_v, self.dv).contiguous()
        proj_in = self.norm(self.delta_out[:rows], zz)
        return proj_in.reshape(rows, self.vdim)

    def restore_batch(self, row: int, slot: int) -> None:
        self.state[row].copy_(self.ckpt_state_b[row, slot])
        self.conv_state[row].copy_(self.ckpt_conv_b[row, slot])

    def enable_batching(self, capacity: int, prefill_row: int) -> None:
        """Give the recurrent state and scratch one row per batch entry."""
        capacity = max(int(capacity), 1)
        if capacity == 1 and self.prefill_row == 0:
            return
        rows = capacity
        self._bcap = capacity
        self.prefill_row = int(prefill_row)
        self.conv_state = torch.zeros(rows, self.conv_dim, self.kernel - 1,
                                      dtype=torch.float32, device="cuda")
        self.conv_out = torch.zeros(rows, self.conv_dim, dtype=torch.float32, device="cuda")
        self.state = torch.zeros(rows, self.num_v, self.dk, self.dv,
                                 dtype=torch.float32, device="cuda")
        self.delta_out = torch.zeros(rows, self.num_v, self.dv,
                                     dtype=torch.float32, device="cuda")
        self.q_pro = torch.zeros(rows, self.num_v, self.dk, dtype=torch.float32, device="cuda")
        self.k_pro = torch.zeros(rows, self.num_v, self.dk, dtype=torch.float32, device="cuda")
        self.v_pro = torch.zeros(rows, self.num_v, self.dv, dtype=torch.float32, device="cuda")
        self.beta = torch.zeros(rows, self.num_v, dtype=torch.float32, device="cuda")
        self.g = torch.zeros(rows, self.num_v, dtype=torch.float32, device="cuda")

    def batch(self, x: torch.Tensor, *_) -> torch.Tensor:
        """``x`` [B, hidden] -> [B, hidden]: batched projections, per-row recurrence.

        The delta rule is sequential over positions and stateful per sequence, so
        the recurrent kernels stay per row in the maths -- but with the stage-3
        batch kernels a whole bucket costs one launch per stage (conv, prologue,
        update) instead of B, each row carrying its own state/conv row.  Without
        them this falls back to calling the single-row kernels once per row; both
        paths must stay bit-identical per row.
        """
        B = x.shape[0]
        if not hip_ops.available():
            raise RuntimeError("batched linear attention needs hip_ops")
        qkv, z, b, a = batch_proj(self.qkvzb_a, x)
        z = z.reshape(B, self.num_v, self.dv)
        proj_in = torch.empty(B, self.vdim, dtype=D, device="cuda")
        use_batch_kernel = hasattr(hip_ops, "conv1d_silu_batch")
        if use_batch_kernel:
            # The batch kernels expect exactly [B, ...] buffers; the state/scratch
            # rows live in capacity-sized buffers, so hand them row prefixes.
            conv_out = self.conv_out[:B]
            hip_ops.conv1d_silu_batch(self.conv_w, qkv, self.conv_state[:B], conv_out)
            # Column slices of the [B, conv_dim] result: their row stride is
            # conv_dim, which the batched prologue accepts (only the two inner
            # strides have to be compact).
            q = conv_out[:, :self.kdim].reshape(B, self.num_k, self.dk)
            k = conv_out[:, self.kdim:2 * self.kdim].reshape(B, self.num_k, self.dk)
            v = conv_out[:, 2 * self.kdim:].reshape(B, self.num_v, self.dv)
            hip_ops.delta_prologue_batch(q, k, v, b.contiguous(), a.contiguous(),
                                         self.neg_expA, self.dt_bias,
                                         self.q_pro[:B], self.k_pro[:B], self.v_pro[:B],
                                         self.beta[:B], self.g[:B], self.qscale, self.rep)
            hip_ops.delta_batch(self.q_pro[:B], self.k_pro[:B], self.v_pro[:B],
                                self.beta[:B], self.g[:B], self.state[:B],
                                self.delta_out[:B])
            for i in range(B):
                proj_in[i] = self.norm(self.delta_out[i],
                                       z[i].reshape(self.num_v, self.dv)).reshape(self.vdim)
        else:
            for i in range(B):
                hip_ops.conv1d_silu(self.conv_w, qkv[i], self.conv_state[i],
                                    self.conv_out[i])
                q, k, v = self.conv_out[i].split([self.kdim, self.kdim, self.vdim])
                hip_ops.delta_prologue(q.reshape(self.num_k, self.dk),
                                       k.reshape(self.num_k, self.dk),
                                       v.reshape(self.num_v, self.dv), b[i], a[i],
                                       self.neg_expA, self.dt_bias,
                                       self.q_pro[i], self.k_pro[i], self.v_pro[i],
                                       self.beta[i], self.g[i],
                                       self.qscale, self.rep)
                hip_ops.delta(self.q_pro[i], self.k_pro[i], self.v_pro[i],
                              self.beta[i], self.g[i], self.state[i], self.delta_out[i])
                proj_in[i] = self.norm(self.delta_out[i],
                                       z[i].reshape(self.num_v, self.dv)).reshape(self.vdim)
        return batch_proj(self.out_proj, proj_in)

    def __call__(self, x: torch.Tensor, *_) -> torch.Tensor:
        qkv, z, b, a = self.qkvzb_a(x)
        z = z.reshape(self.num_v, self.dv)
        pr = self.prefill_row
        conv_state, conv_out = self.conv_state[pr], self.conv_out[pr]

        # depthwise causal conv1d (kernel 4) over the last 3 states + current,
        # fused with the SiLU into one kernel (was cat + copy_ + mul + sum + silu)
        if hip_ops.available():
            hip_ops.conv1d_silu(self.conv_w, qkv, conv_state, conv_out)
            qkv = conv_out
        else:
            win = torch.cat([conv_state, qkv[:, None]], dim=1)   # [C, 4]
            conv_state.copy_(win[:, 1:])
            qkv = F.silu((win * self.conv_w).sum(-1))                 # [conv_dim]

        q, k, v = qkv.split([self.kdim, self.kdim, self.vdim], dim=0)
        q = q.reshape(self.num_k, self.dk)
        k = k.reshape(self.num_k, self.dk)
        v = v.reshape(self.num_v, self.dv)

        # fused q/k norm + head expand + beta/g: one kernel instead of ~16 small
        # elementwise ops (mul, sum, rsqrt, sigmoid, softplus, add, repeat_interleave).
        q_pro, k_pro, v_pro = self.q_pro[pr], self.k_pro[pr], self.v_pro[pr]
        beta, g = self.beta[pr], self.g[pr]
        if hip_ops.available():
            hip_ops.delta_prologue(q, k, v, b, a, self.neg_expA, self.dt_bias,
                                   q_pro, k_pro, v_pro, beta, g,
                                   self.qscale, self.rep)
            out = hip_ops.delta(q_pro, k_pro, v_pro, beta, g,
                                self.state[pr], self.delta_out[pr])
        else:
            if self.rep > 1:
                q = q.repeat_interleave(self.rep, dim=0)
                k = k.repeat_interleave(self.rep, dim=0)
            beta = b.sigmoid().float()
            g = self.neg_expA * F.softplus(a.float() + self.dt_bias)  # <= 0
            q = q.float(); k = k.float()
            q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) * self.qscale
            k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
            out = hip_ops.delta(q, k, v.float(), beta, g, self.state[pr], self.delta_out[pr])

        out = self.norm(out, z).reshape(self.vdim)
        return self.out_proj(out)

    def verify(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` [C, hidden] -> [C, hidden] with per-token state snapshots.

        Same maths as ``prefill`` for the linear layers, but the C tokens go
        through the *decode* kernels (fused prologue + delta), so the recurrent
        state after each token can be copied into ``ckpt_state``/``ckpt_conv``
        and restored when a draft is rejected. The projections are batched
        (``gemv_m``), so C tokens cost about one weight pass.
        """
        C = x.shape[0]
        qkv, z, b, a = self.qkvzb_a.prefill(x)          # [C, conv_dim], [C, vdim], [C, nv]x2
        # The recurrent part is per-token, but ``out_proj`` is a weight read: batch
        # it, otherwise C tokens re-read the projection C times.
        proj_in = torch.empty(C, self.vdim, dtype=D, device="cuda")
        pr = self.prefill_row
        conv_state, conv_out = self.conv_state[pr], self.conv_out[pr]
        q_pro, k_pro, v_pro = self.q_pro[pr], self.k_pro[pr], self.v_pro[pr]
        beta, g = self.beta[pr], self.g[pr]
        state, delta_out = self.state[pr], self.delta_out[pr]
        for i in range(C):
            hip_ops.conv1d_silu(self.conv_w, qkv[i], conv_state, conv_out)
            q, k, v = conv_out.split([self.kdim, self.kdim, self.vdim])
            hip_ops.delta_prologue(q.reshape(self.num_k, self.dk), k.reshape(self.num_k, self.dk),
                                   v.reshape(self.num_v, self.dv), b[i], a[i],
                                   self.neg_expA, self.dt_bias,
                                   q_pro, k_pro, v_pro, beta, g,
                                   self.qscale, self.rep)
            hip_ops.delta(q_pro, k_pro, v_pro, beta, g, state, delta_out)
            # The gated norm is per value head: its weight is [dv], not [vdim], so
            # the state must stay [num_v, dv]. Flattening it first asks the kernel
            # for num_v*dv weights and runs off the end of that [dv] tensor.
            proj_in[i] = self.norm(delta_out,
                                   z[i].reshape(self.num_v, self.dv)).reshape(self.vdim)
            self.ckpt_state[i].copy_(state)
            self.ckpt_conv[i].copy_(conv_state)
        return self.out_proj.prefill(proj_in)

    def prefill_project(self, x: torch.Tensor):
        """Projection half of prefill: ``[C, hidden]`` -> raw ``(qkv, z, b, a)``.

        Split out so a group prefill can run one GEMM over the concatenated
        prompts of several sequences and slice the result afterwards.
        """
        return self.qkvzb_a.prefill(x)

    def prefill_state(self, qkv: torch.Tensor, z: torch.Tensor,
                      b: torch.Tensor, a: torch.Tensor,
                      row: int | None = None, chunk: int = 64) -> torch.Tensor:
        """Recurrent half of prefill: raw projections -> ``[C, vdim]`` (pre-out_proj).

        Runs the causal conv, the fused head prep and the chunked delta scan
        against ``row``'s conv/recurrent state, so several sequences can share one
        batched projection and still keep independent state.
        """
        if row is None:
            row = self.prefill_row
        C = qkv.shape[0]
        conv_state = self.conv_state[row]

        if hip_ops.available():
            # conv + head prep as two kernels, and -- the reason the prep is worth
            # fusing -- q/k/v come out already in the [H, C, d] layout the chunked
            # scan reads, instead of as transposed views it has to copy.
            qkv = hip_ops.conv1d_silu_seq(self.conv_w, qkv, conv_state)
            qt = torch.empty(self.num_v, C, self.dk, dtype=torch.float32, device="cuda")
            kt = torch.empty(self.num_v, C, self.dk, dtype=torch.float32, device="cuda")
            vt = torch.empty(self.num_v, C, self.dv, dtype=torch.float32, device="cuda")
            bt = torch.empty(self.num_v, C, dtype=torch.float32, device="cuda")
            gt = torch.empty(self.num_v, C, dtype=torch.float32, device="cuda")
            hip_ops.delta_prep_seq(qkv, b, a, self.neg_expA, self.dt_bias,
                                   qt, kt, vt, bt, gt, self.qscale, self.rep)
        else:
            # depthwise causal conv1d over the sequence, carrying the last 3 tokens
            seq = torch.cat([conv_state.t(), qkv.float()], dim=0)  # [3+C, conv_dim]
            conv_state.copy_(seq[-self.kernel + 1:].t())
            win = seq.unfold(0, self.kernel, 1)                        # [C, conv_dim, k]
            qkv = F.silu((win * self.conv_w).sum(-1))                  # [C, conv_dim]

            q, k, v = qkv.split([self.kdim, self.kdim, self.vdim], dim=-1)
            q = q.reshape(C, self.num_k, self.dk).float()
            k = k.reshape(C, self.num_k, self.dk).float()
            v = v.reshape(C, self.num_v, self.dv).float()
            if self.rep > 1:
                q = q.repeat_interleave(self.rep, dim=1)
                k = k.repeat_interleave(self.rep, dim=1)
            q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) * self.qscale
            k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
            qt, kt, vt = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
            bt = b.sigmoid().float().transpose(0, 1)                   # [Hv, C]
            gt = (self.neg_expA * F.softplus(a.float() + self.dt_bias)).transpose(0, 1)

        out = chunked_delta(qt, kt, vt, gt, bt, self.state[row], self.state[row], chunk)
        out = out.reshape(C, self.num_v, self.dv)
        return self.norm(out, z.reshape(C, self.num_v, self.dv)).reshape(C, self.vdim)

    def prefill(self, x: torch.Tensor, _positions=None, _ctx=None, _tiles=None,
                chunk: int = 64) -> torch.Tensor:
        """``x`` [C, hidden] -> [C, hidden]; updates conv state and delta state."""
        proj = self.prefill_project(x)
        out = self.prefill_state(*proj, row=self.prefill_row, chunk=chunk)
        return self.out_proj.prefill(out)

    def reset(self) -> None:
        self.conv_state.zero_()
        self.state.zero_()
