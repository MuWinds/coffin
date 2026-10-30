"""Gated DeltaNet linear attention: the mixer of 24 of the 32 layers.

Module: ``coffin/models/qwen35``. The other per-layer attention kind
(``full_attn`` is the 8 paged-KV layers); ``blocks.Layer`` picks one per layer
index and ``model.py`` builds all 32. Depends downward on ``backends`` for the
mixer ops and the device, on ``ops.delta`` for the chunked prefill scan and
``ops`` for dtypes / proj / norm, plus ``models/qwen35/batch`` and ``tuning``.
Nothing outside this family may import it directly.

All state (conv window, fp32 recurrent state) lives in fixed buffers updated in
place, so the decode step is graph-capturable.  Decode runs the fused prologue
and delta entries; prefill uses the chunked form in ``ops.delta``; verify runs
the decode entries per token and copies the state after each one, because a
rejected draft has to be rolled back.

Constraints:
    * The recurrent state ``[num_v, dk, dv]`` is 2 MB per layer in fp32 and has
      no closed-form rewind. Every path that can be rolled back (single-stream
      verify, batched verify) therefore copies the state before/after each
      token into ``ckpt_*``; without that copy a rejected draft leaves the mixer
      permanently ahead of the KV.
    * State must be snapshotted *together* with the full-attention KV when the
      prefix engine exports a snapshot, or a reused prefix pairs keys from one
      request with a recurrence that never saw them.
    * Buffers are capacity-sized with the batch row leading, and ``prefill_row``
      (the last row) is where the single-stream prefill/verify runs -- so a
      prefill between decode steps cannot clobber a decode row's state.
    * fp32 throughout the recurrence: the delta rule accumulates outer products
      across the whole context, and bf16 state diverges over a few thousand
      tokens.
    * No path here branches on whether a kernel exists. The eager twin of the
      prologue/delta chain is the reference backend's implementation, reached
      through the op table, so there is one maths definition and not two.
    * All mutation is single-threaded, driven by ``llm_engine``; there is no
      locking and none may be added.
"""

from __future__ import annotations

import torch

from backends import device as _device
from backends import ops as bops
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
        """Load one layer's DeltaNet weights and allocate its recurrent state.

        Args:
            sd: Full state dict; this layer reads the ``{p}.*`` keys only.
            p: Checkpoint prefix of the layer, e.g. ``model.layers.0.linear_attn``.
            cfg: Model config. The head split (32 value heads, 16 key heads),
                ``linear_key_head_dim`` / ``linear_value_head_dim`` (128 each)
                and ``linear_conv_kernel_dim`` (4) come from here.
            bench: Threaded into ``_proj`` to select the benchmark weight layout.
        """
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
        # The conv kernel arrives as [conv_dim, 1, 4]; the last dim stays last so
        # the fused op can read the four taps as a contiguous stride-1 run.
        self.conv_w = sd[f"{p}.conv1d.weight"].reshape(self.conv_dim, self.kernel).float()
        # ``A_log`` / ``dt_bias`` are stored in fp32 by the checkpoint and are
        # exponentiated here: doing that in a low dtype loses the decay rate.
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
        self.conv_state = torch.zeros(rows, self.conv_dim, self.kernel - 1,
                                      dtype=torch.float32, device=_device)
        self.conv_out = torch.zeros(rows, self.conv_dim, dtype=torch.float32,
                                    device=_device)
        self.state = torch.zeros(rows, self.num_v, self.dk, self.dv,
                                 dtype=torch.float32, device=_device)
        self.delta_out = torch.zeros(rows, self.num_v, self.dv, dtype=torch.float32,
                                     device=_device)

        # fp32 scratch written by ``delta_prologue`` and read by ``delta``.
        self.q_pro = torch.zeros(rows, self.num_v, self.dk, dtype=torch.float32,
                                 device=_device)
        self.k_pro = torch.zeros(rows, self.num_v, self.dk, dtype=torch.float32,
                                 device=_device)
        self.v_pro = torch.zeros(rows, self.num_v, self.dv, dtype=torch.float32,
                                 device=_device)
        self.beta = torch.zeros(rows, self.num_v, dtype=torch.float32, device=_device)
        self.g = torch.zeros(rows, self.num_v, dtype=torch.float32, device=_device)
        # Destination of the fused prologue+recurrence+norm entry. It is a separate
        # buffer from ``delta_out`` rather than sharing it because the fused entry
        # writes the *normed* result: ``delta_out`` stays the pre-norm scratch the
        # three-stage spelling (and the verify path) still needs.
        #
        # bf16, matching what ``RMSNormGated`` returns and what ``out_proj`` reads in the
        # three-op chain. An fp32 container here would change the activation dtype reaching
        # the projection, which is a numeric change even though the value would be equal.
        self.normed_out = torch.zeros(rows, self.num_v, self.dv, dtype=D, device=_device)

        # Per-token state snapshots for speculative-verify rollback (one slot per
        # verified token; the recurrent state cannot be rewound otherwise).
        # ``CKPT`` bounds the verify width for the same reason.
        self.ckpt_state = torch.zeros(CKPT, self.num_v, self.dk, self.dv,
                                      dtype=torch.float32, device=_device)
        self.ckpt_conv = torch.zeros(CKPT, self.conv_dim, self.kernel - 1,
                                     dtype=torch.float32, device=_device)
        # Batched speculative verify keeps one snapshot per (row, verified token);
        # allocated by ``enable_verify_batch`` only when batch speculative decoding
        # is switched on, because the fp32 state is 2 MB per layer per snapshot.
        self.ckpt_state_b: torch.Tensor | None = None
        self.ckpt_conv_b: torch.Tensor | None = None

    def enable_verify_batch(self, capacity: int, width: int) -> None:
        """Room for ``width`` rollback snapshots of every batch row.

        Args:
            capacity: Decode rows.
            width: Verified tokens per row, i.e. snapshot slots. Must not exceed
                ``CKPT`` when the single-stream path also has to hold them.

        Returns:
            Nothing. Idempotent for an unchanged shape, so a second call after
            the first ``enable_verify_batch`` does not reallocate under a graph.
        """
        if self.ckpt_state_b is not None and self.ckpt_state_b.shape[:2] == (capacity, width):
            return
        self.ckpt_state_b = torch.zeros(capacity, width, self.num_v, self.dk, self.dv,
                                        dtype=torch.float32, device=_device)
        self.ckpt_conv_b = torch.zeros(capacity, width, self.conv_dim, self.kernel - 1,
                                       dtype=torch.float32, device=_device)

    def snapshot_batch(self, rows: int, slot: int) -> None:
        """Save the live recurrent (and conv) state of ``rows`` into one slot.

        Args:
            rows: Leading rows to copy; the rest are untouched, so a partially
                filled bucket does not pay for idle rows.
            slot: Snapshot index inside the verify width, i.e. how many drafted
                tokens have been accepted at this point.

        Returns:
            Nothing. This is the only way the batched verify can rewind, so the
            call has to happen after every verified token, before the next one
            mutates the state in place.
        """
        self.ckpt_state_b[:rows, slot].copy_(self.state[:rows])
        self.ckpt_conv_b[:rows, slot].copy_(self.conv_state[:rows])

    def verify_batch_step(self, qkv: torch.Tensor, z: torch.Tensor,
                          b: torch.Tensor, a: torch.Tensor, slot: int) -> torch.Tensor:
        """One verified token for ``B`` rows, from already-projected qkv/z/b/a.

        Runs the same conv / prologue / delta entries as the decode step (one
        batched launch each, so each row keeps the greedy numbers) and snapshots
        the recurrent state after the token -- ``slot`` is its index inside the
        verify width, i.e. the rollback point if the drafts up to here are
        accepted.

        Args:
            qkv: Fused q/k/v projection for the token, ``[B, conv_dim]``.
            z: Gate projection, ``[B, vdim]``.
            b: Beta logits, ``[B, num_v]``.
            a: Decay logits, ``[B, num_v]``.
            slot: Snapshot index, ``0 <= slot < width``.

        Returns:
            ``[B, vdim]`` pre-``out_proj`` output, so the caller runs that weight
            read once for the whole bucket.
        """
        rows = qkv.shape[0]
        bops.conv1d_silu_batch(self.conv_w, qkv, self.conv_state[:rows],
                               self.conv_out[:rows])
        conv = self.conv_out[:rows]
        q = conv[:, :self.kdim].reshape(rows, self.num_k, self.dk)
        k = conv[:, self.kdim:2 * self.kdim].reshape(rows, self.num_k, self.dk)
        v = conv[:, 2 * self.kdim:].reshape(rows, self.num_v, self.dv)
        bops.delta_prologue_batch(q, k, v, b.contiguous(), a.contiguous(),
                                  self.neg_expA, self.dt_bias,
                                  self.q_pro[:rows], self.k_pro[:rows],
                                  self.v_pro[:rows], self.beta[:rows],
                                  self.g[:rows], self.qscale, self.rep)
        bops.delta_batch(self.q_pro[:rows], self.k_pro[:rows],
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
        """Rewind one row's recurrent state to a snapshot taken during verify.

        Args:
            row: Decode row to restore.
            slot: Snapshot index that was accepted; everything after it is
                dropped by moving the state back, which is why the verify loop
                snapshots unconditionally rather than on accept.

        Returns:
            Nothing. Only the state and conv window move -- the KV cache is
            position-indexed and rewound by the position counter.
        """
        self.state[row].copy_(self.ckpt_state_b[row, slot])
        self.conv_state[row].copy_(self.ckpt_conv_b[row, slot])

    def enable_batching(self, capacity: int, prefill_row: int) -> None:
        """Give the recurrent state and scratch one row per batch entry.

        Args:
            capacity: Rows the batch needs, including the prefill row.
            prefill_row: Row reserved for single-stream prefill and verify, so a
                prefill between decode steps never clobbers a decode row's state.

        Returns:
            Nothing. A no-op when the geometry already matches, which keeps the
            captured graph's buffer addresses valid.
        """
        capacity = max(int(capacity), 1)
        if capacity == 1 and self.prefill_row == 0:
            return
        rows = capacity
        self._bcap = capacity
        self.prefill_row = int(prefill_row)
        self.conv_state = torch.zeros(rows, self.conv_dim, self.kernel - 1,
                                      dtype=torch.float32, device=_device)
        self.conv_out = torch.zeros(rows, self.conv_dim, dtype=torch.float32,
                                    device=_device)
        self.state = torch.zeros(rows, self.num_v, self.dk, self.dv,
                                 dtype=torch.float32, device=_device)
        self.delta_out = torch.zeros(rows, self.num_v, self.dv,
                                     dtype=torch.float32, device=_device)
        self.q_pro = torch.zeros(rows, self.num_v, self.dk, dtype=torch.float32,
                                 device=_device)
        self.k_pro = torch.zeros(rows, self.num_v, self.dk, dtype=torch.float32,
                                 device=_device)
        self.v_pro = torch.zeros(rows, self.num_v, self.dv, dtype=torch.float32,
                                 device=_device)
        self.beta = torch.zeros(rows, self.num_v, dtype=torch.float32, device=_device)
        self.g = torch.zeros(rows, self.num_v, dtype=torch.float32, device=_device)
        self.normed_out = torch.zeros(rows, self.num_v, self.dv, dtype=D, device=_device)

    def batch(self, x: torch.Tensor, *_) -> torch.Tensor:
        """``x`` [B, hidden] -> [B, hidden]: batched projections, per-row recurrence.

        The delta rule is sequential over positions and stateful per sequence, so
        the recurrent maths stays per row -- but with the batched entries a whole
        bucket costs one launch per stage (conv, prologue, update) instead of B,
        each row carrying its own state/conv row. Both paths must stay
        bit-identical per row.

        Args:
            x: Decode-row inputs, ``[B, hidden]``.
            *_: Absorbed layer-argument placeholder, so ``Layer.batch`` can call
                every mixer with the same signature.

        Returns:
            ``[B, hidden]`` layer output after the batched ``out_proj``.
        """
        B = x.shape[0]
        qkv, z, b, a = batch_proj(self.qkvzb_a, x)
        z = z.reshape(B, self.num_v, self.dv)
        proj_in = torch.empty(B, self.vdim, dtype=D, device=_device)
        # The batch entries expect exactly [B, ...] buffers; the state/scratch rows
        # live in capacity-sized buffers, so hand them row prefixes.
        conv_out = self.conv_out[:B]
        bops.conv1d_silu_batch(self.conv_w, qkv, self.conv_state[:B], conv_out)
        # Column slices of the [B, conv_dim] result: their row stride is
        # conv_dim, which the batched prologue accepts (only the two inner
        # strides have to be compact).
        q = conv_out[:, :self.kdim].reshape(B, self.num_k, self.dk)
        k = conv_out[:, self.kdim:2 * self.kdim].reshape(B, self.num_k, self.dk)
        v = conv_out[:, 2 * self.kdim:].reshape(B, self.num_v, self.dv)
        bops.delta_prologue_batch(q, k, v, b.contiguous(), a.contiguous(),
                                  self.neg_expA, self.dt_bias,
                                  self.q_pro[:B], self.k_pro[:B], self.v_pro[:B],
                                  self.beta[:B], self.g[:B], self.qscale, self.rep)
        bops.delta_batch(self.q_pro[:B], self.k_pro[:B], self.v_pro[:B],
                         self.beta[:B], self.g[:B], self.state[:B],
                         self.delta_out[:B])
        # The gated norm is per (row, value head) and its weight is [dv], so it
        # cannot be folded into the batched entries -- this loop is one launch per
        # row, not per head.
        for i in range(B):
            proj_in[i] = self.norm(self.delta_out[i],
                                   z[i].reshape(self.num_v, self.dv)).reshape(self.vdim)
        return batch_proj(self.out_proj, proj_in)

    def __call__(self, x: torch.Tensor, *_) -> torch.Tensor:
        """Single-token decode step, state updated in place.

        Args:
            x: The layer input for this token, ``[hidden]``.
            *_: Absorbed layer-argument placeholder, so ``Layer.__call__`` can
                call both mixer kinds with the same signature.

        Returns:
            ``[hidden]`` layer output after ``out_proj``.
        """
        qkv, z, b, a = self.qkvzb_a(x)
        z = z.reshape(self.num_v, self.dv)
        pr = self.prefill_row
        conv_state, conv_out = self.conv_state[pr], self.conv_out[pr]

        # Depthwise causal conv1d (kernel 4) over the last 3 states plus current,
        # fused with the SiLU into one op (was cat + copy_ + mul + sum + silu).
        bops.conv1d_silu(self.conv_w, qkv, conv_state, conv_out)
        qkv = conv_out

        q, k, v = qkv.split([self.kdim, self.kdim, self.vdim], dim=0)
        q = q.reshape(self.num_k, self.dk)
        k = k.reshape(self.num_k, self.dk)
        v = v.reshape(self.num_v, self.dv)

        # The three-op spelling, not the fused ``delta_normed``. The fusion is bit-identical
        # and removes 48 launches per step, and it is still slower: measured on the card it
        # costs +5.3 us per layer of device time in isolation and +13.1 us per layer in the
        # real step (20.285 -> 20.615 ms over 24 layers), because one kernel has to serve
        # the recurrence's register-loop geometry and two block reductions at once. The
        # launch slots it saves are worth less than that. See ../../../PERF_COREX.md §12.
        q_pro, k_pro, v_pro = self.q_pro[pr], self.k_pro[pr], self.v_pro[pr]
        beta, g = self.beta[pr], self.g[pr]
        bops.delta_prologue(q, k, v, b, a, self.neg_expA, self.dt_bias,
                            q_pro, k_pro, v_pro, beta, g,
                            self.qscale, self.rep)
        out = bops.delta(q_pro, k_pro, v_pro, beta, g,
                         self.state[pr], self.delta_out[pr])

        out = self.norm(out, z).reshape(self.vdim)
        return self.out_proj(out)

    def verify(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` [C, hidden] -> [C, hidden] with per-token state snapshots.

        Same maths as ``prefill`` for the linear layers, but the C tokens go
        through the *decode* entries (fused prologue + delta), so the recurrent
        state after each token can be copied into ``ckpt_state``/``ckpt_conv``
        and restored when a draft is rejected. The projections are batched, so C
        tokens cost about one weight pass.

        Args:
            x: The drafted tokens' inputs, ``[C, hidden]``, ``C <= CKPT``.

        Returns:
            ``[C, hidden]`` layer output after ``out_proj``. The snapshot in
            ``ckpt_state[i]`` is the state *after* token ``i``, so it doubles as
            the resume point if the first ``i+1`` drafts are accepted.
        """
        C = x.shape[0]
        # [C, conv_dim], [C, vdim], [C, nv]x2
        qkv, z, b, a = self.qkvzb_a.prefill(x)
        # The recurrent part is per-token, but ``out_proj`` is a weight read: batch
        # it, otherwise C tokens re-read the projection C times.
        proj_in = torch.empty(C, self.vdim, dtype=D, device=_device)
        pr = self.prefill_row
        conv_state, conv_out = self.conv_state[pr], self.conv_out[pr]
        q_pro, k_pro, v_pro = self.q_pro[pr], self.k_pro[pr], self.v_pro[pr]
        beta, g = self.beta[pr], self.g[pr]
        state, delta_out = self.state[pr], self.delta_out[pr]
        for i in range(C):
            bops.conv1d_silu(self.conv_w, qkv[i], conv_state, conv_out)
            q, k, v = conv_out.split([self.kdim, self.kdim, self.vdim])
            bops.delta_prologue(q.reshape(self.num_k, self.dk),
                                k.reshape(self.num_k, self.dk),
                                v.reshape(self.num_v, self.dv), b[i], a[i],
                                self.neg_expA, self.dt_bias,
                                q_pro, k_pro, v_pro, beta, g,
                                self.qscale, self.rep)
            bops.delta(q_pro, k_pro, v_pro, beta, g, state, delta_out)
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

        Args:
            x: Prefill inputs, ``[C, hidden]``.

        Returns:
            ``(qkv, z, b, a)`` before the conv and the delta prep, i.e. what
            :meth:`prefill_state` consumes after the caller's optional
            concatenation and slicing.
        """
        return self.qkvzb_a.prefill(x)

    def prefill_state(self, qkv: torch.Tensor, z: torch.Tensor,
                      b: torch.Tensor, a: torch.Tensor,
                      row: int | None = None, chunk: int = 64) -> torch.Tensor:
        """Recurrent half of prefill: raw projections -> ``[C, vdim]`` (pre-out_proj).

        Runs the causal conv, the fused head prep and the chunked delta scan
        against ``row``'s conv/recurrent state, so several sequences can share one
        batched projection and still keep independent state.

        Args:
            qkv: Fused q/k/v projection, ``[C, conv_dim]``.
            z: Gate projection, ``[C, vdim]``.
            b: Beta logits, ``[C, num_v]``.
            a: Decay logits, ``[C, num_v]``.
            row: State row to advance; ``None`` uses the staging row.
            chunk: Tokens per chunk in the delta scan. The scan is exact for any
                split, so this trades scratch size against launch count.

        Returns:
            ``[C, vdim]`` gated-norm output, before ``out_proj``.
        """
        if row is None:
            row = self.prefill_row
        C = qkv.shape[0]
        conv_state = self.conv_state[row]

        # conv + head prep as two ops, and -- the reason the prep is worth fusing
        # -- q/k/v come out already in the [H, C, d] layout the chunked scan reads,
        # instead of as transposed views it has to copy.
        qkv = bops.conv1d_silu_seq(self.conv_w, qkv, conv_state)
        qt = torch.empty(self.num_v, C, self.dk, dtype=torch.float32, device=_device)
        kt = torch.empty(self.num_v, C, self.dk, dtype=torch.float32, device=_device)
        vt = torch.empty(self.num_v, C, self.dv, dtype=torch.float32, device=_device)
        bt = torch.empty(self.num_v, C, dtype=torch.float32, device=_device)
        gt = torch.empty(self.num_v, C, dtype=torch.float32, device=_device)
        bops.delta_prep_seq(qkv, b, a, self.neg_expA, self.dt_bias,
                            qt, kt, vt, bt, gt, self.qscale, self.rep)

        out = chunked_delta(qt, kt, vt, gt, bt, self.state[row], self.state[row], chunk)
        out = out.reshape(C, self.num_v, self.dv)
        return self.norm(out, z.reshape(C, self.num_v, self.dv)).reshape(C, self.vdim)

    def prefill(self, x: torch.Tensor, _positions=None, _ctx=None, _tiles=None,
                chunk: int = 64) -> torch.Tensor:
        """``x`` [C, hidden] -> [C, hidden]; updates conv state and delta state.

        Args:
            x: Prefill inputs, ``[C, hidden]``.
            _positions: Ignored. Present because ``Layer.prefill`` calls both
                mixer kinds with one signature; the linear mixer is causal by
                construction and needs no position table.
            _ctx: Ignored, same reason.
            _tiles: Ignored, same reason.
            chunk: Tokens per chunk in the delta scan.

        Returns:
            ``[C, hidden]`` layer output after ``out_proj``, in ``x`` order.
        """
        proj = self.prefill_project(x)
        out = self.prefill_state(*proj, row=self.prefill_row, chunk=chunk)
        return self.out_proj.prefill(out)

    def reset(self) -> None:
        """Clear the recurrence between requests.

        Returns:
            Nothing. Both buffers must go back to zero: a leftover conv window
            would let the next request's first tokens see the previous request's
            tail, which no position counter can undo.
        """
        self.conv_state.zero_()
        self.state.zero_()
