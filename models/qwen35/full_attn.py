"""GQA full attention with an output gate, partial RoPE and an int8 KV cache.

Module: ``coffin/models/qwen35``. One of the two per-layer attention kinds the
family composes (the other is ``linear_attn``); ``blocks.Layer`` picks one per
layer index and ``model.py`` builds all 32. Depends downward on ``hip_ops`` for
the kernels, ``ops`` for dtypes/rope/norm/proj, and ``models/qwen35/batch`` and
``tuning`` for the batched projections and split counts. Nothing outside this
family may import it directly.

Eight of the 32 layers are this kind.  The decode step always spans the whole
``max_seq`` window and masks entries past a device-side position counter, so
its shape never changes and the graph is captured once; prefill uses online
softmax over key blocks so the score block stays ~34 MB whatever the context
length.  The split-K partial buffers live here because the captured graph needs
stable addresses.

Constraints:
    * The KV cache is int8 with a separate fp32 per-(head, token) scale. The
      quantiser is ``_quant`` for one token and ``_quant_rows`` /
      ``_quant_seq`` for a batch; all three must stay bit-identical, because
      the kernel dequantises with the same scale the writer stored.
    * Decode buffers have static shapes and stable addresses. Anything the
      captured graph writes must be allocated before ``capture()`` and never
      reassigned afterwards; ``enable_batching`` and ``attach_pool`` are the
      only resizers, and both must run pre-capture.
    * Recurrent state does not exist here -- the KV is position-indexed, so
      rollback is a position counter -- but the cache must still be snapshotted
      and restored with it for prefix reuse, or a shared KV pairs a restored
      position counter with someone else's tail.
    * All mutation is single-threaded, driven by ``llm_engine``; there is no
      locking and none may be added.
"""

from __future__ import annotations

import torch

import hip_ops
from models.qwen35.batch import batch_proj
from models.qwen35.tuning import ATTN_SPLIT, PREFILL_ATTN_KB
from ops.dtypes import D, NEG_INF, WF
from ops.gemv import _proj
from ops.norm import RMSNorm
from ops.rope import apply_rope


class FullAttn:
    """GQA attention with an output gate, partial rotary and an int8 KV cache.

    Attention spans the whole ``max_seq`` window; entries past the current
    position are masked, so the shapes stay static for graph capture.
    """

    def __init__(self, sd: dict, p: str, cfg: dict, bench: bool, max_seq: int,
                 paged: bool = False) -> None:
        """Load one layer's attention weights and size its KV buffers.

        Args:
            sd: Full state dict; this layer reads the ``{p}.*`` keys only.
            p: Checkpoint prefix of the layer, e.g. ``model.layers.3.self_attn``.
            cfg: Model config; ``num_attention_heads`` (16),
                ``num_key_value_heads`` (4), ``head_dim`` (256) and
                ``rope_parameters.partial_rotary_factor`` (0.25 -> 64 rotated
                dims) come from here.
            bench: Thread the flag into ``_proj``, which selects the benchmark
                weight layout.
            max_seq: Window in tokens; the dense cache is ``[nkv, max_seq, hd]``.
            paged: Skip the dense cache entirely because a pool will own the KV.
                See the comment on the allocation below -- this is a memory
                decision, not a mode switch.
        """
        self.nh = cfg["num_attention_heads"]        # 16
        self.nkv = cfg["num_key_value_heads"]       # 4
        self.hd = cfg["head_dim"]                   # 256
        self.rep = self.nh // self.nkv
        self.rot = int(self.hd * cfg["rope_parameters"]["partial_rotary_factor"])  # 64
        self.scale = self.hd ** -0.5
        self.max_seq = max_seq

        self.qkv = _proj(sd, [f"{p}.q_proj", f"{p}.k_proj", f"{p}.v_proj"], bench)
        self.o_proj = _proj(sd, [f"{p}.o_proj"], bench)
        self.q_norm = RMSNorm(sd[f"{p}.q_norm.weight"])
        self.k_norm = RMSNorm(sd[f"{p}.k_norm.weight"])

        # int8 KV cache: symmetric per-(head, token) quantisation. With a paged
        # pool attached these private buffers are dropped and the pool is used
        # instead (see ``attach_pool``).  ``paged=True`` skips allocating them in
        # the first place: at a 256k window they are 4 GB of memory that the pool
        # then needs back, and the allocator cannot reliably reuse the freed
        # 0.5 GB-per-layer segments for the pool's single large tensors.
        if not paged:
            self.kcache = torch.zeros(self.nkv, max_seq, self.hd, dtype=torch.int8,
                                      device="cuda")
            self.vcache = torch.zeros(self.nkv, max_seq, self.hd, dtype=torch.int8,
                                      device="cuda")
            self.kscale = torch.zeros(self.nkv, max_seq, dtype=torch.float32, device="cuda")
            self.vscale = torch.zeros(self.nkv, max_seq, dtype=torch.float32, device="cuda")
        else:
            self.kcache = self.vcache = self.kscale = self.vscale = None
        # Per-row buffers for the batch path: the leading dim is the batch row
        # (capacity 1 = the single-stream engine, whose kernels get the row-0
        # views below).  ``enable_batching`` reallocates them with more rows;
        # ``prefill_row`` is the row the single-stream prefill/verify path uses,
        # kept apart from the decode rows by ``Qwen35.enable_batching``.
        self._bcap = 1
        self.prefill_row = 0
        self.attn_out = torch.zeros(1, self.nh, self.hd, dtype=D, device="cuda")
        # Per-key-slice partials for the split decode attention (see attn.cu);
        # kept here because the decode step is a captured graph and needs fixed
        # buffers.
        self.part_acc = torch.zeros(1, self.nh, ATTN_SPLIT, self.hd, dtype=torch.float32,
                                    device="cuda")
        self.part_ml = torch.zeros(1, self.nh, ATTN_SPLIT, 2, dtype=torch.float32,
                                   device="cuda")
        self._bind_single_views()
        self.pool = None
        self.pidx = 0

    def _bind_single_views(self) -> None:
        """Row views the single-stream (prefill / verify) path writes.

        Re-bound, never reallocated: the captured decode/prefill graphs were
        recorded against these exact addresses, so ``enable_batching`` calls
        this after resizing the parents instead of binding views per call.
        """
        pr = self.prefill_row
        self.attn_out1 = self.attn_out[pr]
        self.part_acc1 = self.part_acc[pr]
        self.part_ml1 = self.part_ml[pr]

    def enable_batching(self, capacity: int, prefill_row: int) -> None:
        """Give the attention buffers one row per batch entry (before capture).

        Args:
            capacity: Rows the batch needs, already including the prefill row.
            prefill_row: Row reserved for single-stream prefill and verify. The
                batched decode must never write it, so the model keeps it out of
                the row pool.

        Returns:
            Nothing. A no-op when the buffers already have this capacity and the
            staging row is unchanged, which keeps the captured graph addresses
            valid across an engine that runs both modes.
        """
        capacity = max(int(capacity), 1)
        if capacity == 1 and self.prefill_row == 0:
            return
        self._bcap = capacity
        self.prefill_row = int(prefill_row)
        self.attn_out = torch.zeros(capacity, self.nh, self.hd, dtype=D, device="cuda")
        self.part_acc = torch.zeros(capacity, self.nh, ATTN_SPLIT, self.hd,
                                    dtype=torch.float32, device="cuda")
        self.part_ml = torch.zeros(capacity, self.nh, ATTN_SPLIT, 2,
                                   dtype=torch.float32, device="cuda")
        self._bind_single_views()

    def attach_pool(self, pool, idx: int) -> None:
        """Serve this layer's KV from ``pool`` slot ``idx`` instead of the dense cache.

        Called before any graph is captured: the captured decode step bakes in the
        pool tensors, so switching afterwards would leave a graph writing to the
        wrong place. The private dense buffers are released.

        Args:
            pool: The paged KV pool (``paged_cache``); its slot maps are read on
                every step, its tensors are written in place.
            idx: This layer's slot inside the pool, i.e. its position in the
                per-layer dimension of ``pool.k`` / ``pool.v``.

        Returns:
            Nothing. After this call the dense cache attributes are ``None``;
            any code path that still reads them will fail loudly rather than
            quietly write to a stale buffer.
        """
        self.pool = pool
        self.pidx = idx
        self.kcache = None
        self.vcache = None
        self.kscale = None
        self.vscale = None

    def _pool_views(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """This layer's slices of the pool, in the order the kernels take them.

        Returns:
            ``(k, v, kscale, vscale)`` views, already indexed by ``pidx``. They
            point into the pool, so writing through them is the cache write.
        """
        p = self.pool
        return p.k[self.pidx], p.v[self.pidx], p.kscale[self.pidx], p.vscale[self.pidx]

    @staticmethod
    def _quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """[nkv, hd] bf16 -> (int8, per-head scale).

        The ``1e-5`` floor on the abs-max stops an all-zero K/V row from
        dividing by zero; it costs under one int8 step, so it cannot move the
        value past what int8 already rounds to.

        Args:
            x: One token's K or V, ``[nkv, hd]``, in the compute dtype.

        Returns:
            ``(q, s)``: int8 codes in ``[-127, 127]`` and the positive fp32 scale
            per head that the kernel multiplies back.
        """
        s = x.float().abs().amax(-1).clamp_min(1e-5) / 127.0
        q = (x.float() / s[:, None]).round().clamp_(-127, 127).to(torch.int8)
        return q, s

    def __call__(self, x: torch.Tensor, pos1, cos_tbl, sin_tbl, mask_idx,
                 rope_delta=None) -> torch.Tensor:
        """Single-token decode step for one row.

        Args:
            x: The layer input for this token, ``[hidden]``.
            pos1: 1-element long tensor with the token index. It addresses the KV
                slot and the cache write, and the mask is ``> pos1``.
            cos_tbl: RoPE cos table, indexed by position.
            sin_tbl: RoPE sin table, indexed by position.
            mask_idx: Device-side position counter used by the CPU fallback mask.
            rope_delta: Optional mRoPE offset added to ``pos1`` for the rotation
                only. Zero (``None``) for text, so the text path stays bit for
                bit what it was before images existed.

        Returns:
            ``[hidden]`` the gated attention output after ``o_proj``.
        """
        qg, k, v = self.qkv(x)
        qg = qg.reshape(self.nh, 2 * self.hd)
        q, gate = qg[:, :self.hd], qg[:, self.hd:]
        q = self.q_norm(q)
        k = self.k_norm(k.reshape(self.nkv, self.hd))
        v = v.reshape(self.nkv, self.hd)

        # ``pos1`` stays the *token index* everywhere else in this method (it is
        # what addresses the KV slot); only the rotation reads the mRoPE position,
        # which is ``token_index + rope_delta`` once a prompt has an image.  With
        # no image ``rope_delta`` is 0 and the lookup is the old one, bit for bit.
        idx = pos1 if rope_delta is None else pos1 + rope_delta
        cos = cos_tbl.index_select(0, idx)
        sin = sin_tbl.index_select(0, idx)
        q, k = apply_rope(q, k, cos, sin, self.rot)

        qk, sk = self._quant(k)
        qv, sv = self._quant(v)
        if self.pool is not None:
            self.pool.write_slot(self.pidx, self.pool.slots_for(pos1, self.prefill_row),
                                 qk, sk, qv, sv)
        else:
            self.kcache.index_copy_(1, pos1, qk.unsqueeze(1))
            self.vcache.index_copy_(1, pos1, qv.unsqueeze(1))
            self.kscale.index_copy_(1, pos1, sk.unsqueeze(1))
            self.vscale.index_copy_(1, pos1, sv.unsqueeze(1))

        # One fused kernel per path: dequant k/v, fold the scales, apply the
        # causal mask, softmax, gate, PV. The fallbacks below only run without
        # the extension and must produce the same numbers.
        if self.pool is not None and hip_ops.available():
            kp, vp, kps, vps = self._pool_views()
            out = hip_ops.attn_paged(q.contiguous(), gate.contiguous(), kp, vp, kps, vps,
                                     self.pool.pos2slot[self.prefill_row], self.attn_out1,
                                     pos1, self.max_seq, self.rep, self.scale,
                                     self.part_acc1, self.part_ml1)
            return self.o_proj(out.reshape(-1))
        if hip_ops.available():
            out = hip_ops.attn(q.contiguous(), gate.contiguous(), self.kcache,
                               self.vcache, self.kscale, self.vscale,
                               self.attn_out1, pos1, self.rep, self.scale,
                               self.part_acc1, self.part_ml1)
            return self.o_proj(out.reshape(-1))

        kcache = self.pool.k[self.pidx] if self.pool is not None else self.kcache
        kk = kcache.to(D)
        qe = q.reshape(self.nkv, self.rep, self.hd)
        scores = torch.einsum("ghd,gsd->ghs", qe, kk).float()
        scores = scores * self.kscale[:, None, :] * self.scale
        mask = (mask_idx[None, :] > pos1[:, None])[None, :, :]
        scores = scores.masked_fill(mask, NEG_INF)
        probs = scores.softmax(-1).to(D)

        pv = probs * self.vscale[:, None, :].to(D)
        out = torch.einsum("ghs,gsd->ghd", pv, self.vcache.to(D)).reshape(self.nh, self.hd)
        out = out * gate.sigmoid()
        return self.o_proj(out.reshape(-1))

    @staticmethod
    def _quant_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """[B, nkv, hd] bf16 -> (int8, per-(row, head) scale), row-wise == ``_quant``.

        Args:
            x: One token per batch row, ``[B, nkv, hd]``.

        Returns:
            ``(q, s)``: int8 codes ``[B, nkv, hd]`` and fp32 scales ``[B, nkv]``.
            Each row is quantised independently, so the result for row ``b`` is
            bit-identical to :meth:`_quant` on that row alone.
        """
        s = x.float().abs().amax(-1).clamp_min(1e-5) / 127.0
        q = (x.float() / s.unsqueeze(-1)).round().clamp_(-127, 127).to(torch.int8)
        return q, s

    def batch(self, x: torch.Tensor, positions: torch.Tensor, ctx) -> torch.Tensor:
        """``x`` [B, hidden] -> [B, hidden]: batched projections, per-row attention.

        Two of the three stages here are batched: the projections (one weight read
        for all rows, ``models/qwen35/batch.py``) and -- when the extension has
        them -- the stage-3 batch kernels, so attention is one launch for the
        whole bucket instead of B.  Each row still resolves positions through its
        own ``pos2slot`` row and stops at its own ``context_lens[b]``, and the
        kernels are numerically identical to the single-row ones.  Without the
        batch kernels this falls back to calling the single-token kernel once per
        row on row slices; both paths must stay bit-identical per row.

        ``ctx`` may carry a 5th entry: the per-row ``rope_delta`` ``[B]`` long
        (mRoPE text offset of a request that has an image).  It only moves the
        cos/sin lookup; ``positions`` keeps addressing the KV slot.

        Args:
            x: Decode-row inputs, ``[B, hidden]``.
            positions: Current token index per row, ``[B]`` long.
            ctx: ``(cos_tbl, sin_tbl, mask_idx[, context_lens[, rope_delta]])``.
                ``context_lens`` is ``[B]`` int32 and selects the batched kernel;
                ``rope_delta`` is ``[B]`` long and moves only the rotation.

        Returns:
            ``[B, hidden]`` gated attention output after the batched ``o_proj``.

        Raises:
            RuntimeError: The extension is unavailable or no pool is attached.
                The batched path has no dense-cache fallback, so the caller must
                either attach a pool before enabling batching or stay on
                single-stream decode.
        """
        cos_tbl, sin_tbl, mask_idx = ctx[0], ctx[1], ctx[2]
        ctx_lens = ctx[3] if len(ctx) > 3 else None
        rope_delta = ctx[4] if len(ctx) > 4 else None
        B = x.shape[0]
        if not hip_ops.available() or self.pool is None:
            raise RuntimeError("batched attention needs hip_ops and a paged pool")
        qg, k, v = batch_proj(self.qkv, x)
        qg = qg.reshape(B, self.nh, 2 * self.hd)
        q, gate = qg[..., :self.hd], qg[..., self.hd:]
        q = self.q_norm(q)
        k = self.k_norm(k.reshape(B, self.nkv, self.hd))
        v = v.reshape(B, self.nkv, self.hd)

        # [B, 1, rot]: the singleton is the token axis the block kernels expect.
        rope_pos = positions if rope_delta is None else positions + rope_delta[:B]
        cos = cos_tbl.index_select(0, rope_pos).unsqueeze(1)
        sin = sin_tbl.index_select(0, rope_pos).unsqueeze(1)
        q, k = apply_rope(q, k, cos, sin, self.rot)

        kp, vp, kps, vps = self._pool_views()
        use_batch_kernel = ctx_lens is not None and hasattr(hip_ops, "attn_paged_batch")
        if use_batch_kernel:
            # One launch for the whole bucket: quantise every row, write every
            # row's K/V through its own slot, then one paged-attention pass.
            qk, sk = self._quant_rows(k)
            qv, sv = self._quant_rows(v)
            slots = self.pool.pos2slot[:B].gather(1, positions.view(B, 1)).squeeze(1)
            self.pool.write_rows(self.pidx, slots, qk, sk, qv, sv)
            hip_ops.attn_paged_batch(q, gate, kp, vp, kps, vps,
                                     self.pool.pos2slot[:B], ctx_lens[:B],
                                     self.attn_out[:B], self.max_seq, self.rep, self.scale,
                                     self.part_acc[:B], self.part_ml[:B])
        else:
            for b in range(B):
                qk, sk = self._quant(k[b])
                qv, sv = self._quant(v[b])
                pos_b = positions[b:b + 1]
                self.pool.write_slot(self.pidx, self.pool.slots_for(pos_b, b),
                                     qk, sk, qv, sv)
                hip_ops.attn_paged(q[b].contiguous(), gate[b].contiguous(), kp, vp, kps, vps,
                                   self.pool.pos2slot[b], self.attn_out[b], pos_b,
                                   self.max_seq, self.rep, self.scale,
                                   self.part_acc[b], self.part_ml[b])
        # Only the first ``B`` rows of the capacity-sized buffer were written; the
        # slice is what keeps the projection's K at nh*hd.
        return batch_proj(self.o_proj, self.attn_out[:B].reshape(B, -1))

    @staticmethod
    def _quant_seq(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """[C, nkv, hd] bf16 -> (int8, per-(token, head) scale).

        Args:
            x: Several tokens' K or V, ``[C, nkv, hd]``, in the compute dtype.

        Returns:
            ``(q, s)`` with int8 codes ``[C, nkv, hd]`` and fp32 scales
            ``[C, nkv]``. Same per-token reduction as :meth:`_quant`, so a
            prefill chunk and a per-token write quantise identically.
        """
        s = x.float().abs().amax(-1).clamp_min(1e-5) / 127.0
        q = (x.float() / s[..., None]).round().clamp_(-127, 127).to(torch.int8)
        return q, s

    def prefill_project(self, x: torch.Tensor):
        """Projection half of prefill: ``[C, hidden]`` -> raw ``(qg, k, v)``.

        Split out so a group prefill can run one GEMM over the concatenated
        prompts of several sequences and slice the result afterwards.

        Args:
            x: Prefill inputs, ``[C, hidden]``.

        Returns:
            ``(qg, k, v)`` before the q/k norms and rope, i.e. the raw projection
            the caller may concatenate and later hand to :meth:`prefill_attend`.
        """
        return self.qkv.prefill(x)

    def enable_multi_q(self, capacity: int, width: int) -> None:
        """Buffers for the multi-query verify: one ``(row, token)`` query pair row.

        Args:
            capacity: Decode rows.
            width: Drafted tokens per row (``c``); a row and its ``c`` queries
                become ``c`` entries of the leading dim.

        Returns:
            Nothing. Buffers are allocated here and only here, because the
            verify graph captures these addresses.
        """
        self._mq_width = int(width)
        n = capacity * int(width)
        self.mq_out = torch.zeros(n, self.nh, self.hd, dtype=D, device="cuda")
        self.mq_part_acc = torch.zeros(n, self.nh, ATTN_SPLIT, self.hd,
                                       dtype=torch.float32, device="cuda")
        self.mq_part_ml = torch.zeros(n, self.nh, ATTN_SPLIT, 2,
                                      dtype=torch.float32, device="cuda")

    def verify_batch_multi(self, qg: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                           positions: torch.Tensor, ctx, rows: int) -> torch.Tensor:
        """One multi-query attention call for the whole one-forward verify.

        ``qg``/``k``/``v`` are the layer projection over ``rows*c`` tokens
        (row-major), ``positions`` is ``[rows, c]``.  Every token's K/V is written
        at its own slot (one indexed write for all ``rows*c`` tokens), then
        ``attn_paged_multi_q`` serves all ``c`` queries of every row from one pass
        over that row's history KV.  Returns the pre-``o_proj`` attention output
        ``[rows*c, nh*hd]``; the caller runs the wide projection once.

        Args:
            qg: Packed query and gate, ``[rows*c, 2*nh*hd]``.
            k: Keys, ``[rows*c, nkv*hd]``.
            v: Values, ``[rows*c, nkv*hd]``.
            positions: Token indices, ``[rows, c]`` long, row-major inside a row.
            ctx: ``(cos_tbl, sin_tbl[, rope_delta_rows])``; a 5th entry is the
                per-row mRoPE offset, applied to the rotation only.
            rows: Row count ``b``; ``c`` is read from ``positions``.

        Returns:
            ``[rows*c, nh*hd]`` gated attention output before ``o_proj``, so the
            caller can run that weight read once for the whole verify batch.
        """
        cos_tbl, sin_tbl = ctx[0], ctx[1]
        b = rows
        c = positions.shape[1]
        n = b * c
        qg = qg.view(b, c, self.nh, 2 * self.hd)
        q, gate = qg[..., :self.hd], qg[..., self.hd:]
        q = self.q_norm(q)
        k = self.k_norm(k.view(b, c, self.nkv, self.hd))
        v = v.view(b, c, self.nkv, self.hd)
        # The rotation's position is ``token index + rope_delta``; the KV slot and
        # the per-row window stay at the token index.
        rope_pos = positions if len(ctx) < 5 else positions + ctx[4][:b, None]
        cos = cos_tbl.index_select(0, rope_pos.reshape(-1)).view(b, c, 1, self.rot)
        sin = sin_tbl.index_select(0, rope_pos.reshape(-1)).view(b, c, 1, self.rot)
        q, k = apply_rope(q, k, cos, sin, self.rot)

        # Every verified token's K/V goes in, then the multi-query attention reads
        # each row's history once for all c queries (the kernel masks query j at
        # pos j, so one pass serves the whole speculative block).
        kq, ks = self._quant_seq(k.reshape(n, self.nkv, self.hd))
        vq, vs = self._quant_seq(v.reshape(n, self.nkv, self.hd))
        slots = torch.stack([self.pool.pos2slot[i].index_select(0, positions[i])
                             for i in range(b)]).reshape(n)
        self.pool.write_rows(self.pidx, slots, kq, ks, vq, vs)

        kp, vp, kps, vps = self._pool_views()
        lens = positions[:, 0].to(torch.int32).contiguous()
        out = self.mq_out[:n].view(b, c, self.nh, self.hd)
        from models.qwen35.batch import _count
        _count("multi_q")
        hip_ops.attn_paged_multi_q(q.contiguous(), gate.contiguous(), kp, vp, kps, vps,
                                   self.pool.pos2slot[:b], lens, out,
                                   self.max_seq, self.rep, self.scale,
                                   self.mq_part_acc[:n], self.mq_part_ml[:n])
        return out.reshape(n, self.nh * self.hd)

    def verify_batch_step(self, qg: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                          positions: torch.Tensor, ctx) -> torch.Tensor:
        """One verified token for ``B`` rows, from already-projected q/k/v.

        Same attention as the decode step (the fused paged kernel, so the numbers
        agree with greedy), one batched launch for the whole bucket: the verify
        graph projects all ``B*c`` tokens once and then walks the ``c`` token
        positions through this entry point.  ``ctx`` may carry the per-row
        ``rope_delta`` as a 5th entry (mRoPE after an image; ``positions`` keeps
        addressing the KV slot).

        Args:
            qg: Already-projected query and gate for the ``c`` tokens of each
                row, ``[B*c, 2*nh*hd]``.
            k: Projected keys, ``[B*c, nkv*hd]``.
            v: Projected values, ``[B*c, nkv*hd]``.
            positions: Token indices, ``[B, c]`` long; used both to place the KV
                and, plus ``rope_delta``, to look up the rotation.
            ctx: ``(cos_tbl, sin_tbl, mask_idx)``; a 5th entry is the per-row
                mRoPE offset.

        Returns:
            ``[B, hidden]`` gated output after ``o_proj`` -- one row per verified
            token, which the caller compares against the draft to accept or
            reject.
        """
        cos_tbl, sin_tbl, mask_idx = ctx[0], ctx[1], ctx[2]
        b = qg.shape[0]
        qg = qg.reshape(b, self.nh, 2 * self.hd)
        q, gate = qg[..., :self.hd], qg[..., self.hd:]
        q = self.q_norm(q)
        k = self.k_norm(k.reshape(b, self.nkv, self.hd))
        v = v.reshape(b, self.nkv, self.hd)
        rope_pos = positions if len(ctx) < 5 else positions + ctx[4][:b]
        cos = cos_tbl.index_select(0, rope_pos).unsqueeze(1)
        sin = sin_tbl.index_select(0, rope_pos).unsqueeze(1)
        q, k = apply_rope(q, k, cos, sin, self.rot)
        qk, sk = self._quant_rows(k)
        qv, sv = self._quant_rows(v)
        kp, vp, kps, vps = self._pool_views()
        slots = self.pool.pos2slot[:b].gather(1, positions.view(b, 1)).squeeze(1)
        self.pool.write_rows(self.pidx, slots, qk, sk, qv, sv)
        # ``attn_paged_batch`` takes a length, not a last index, so +1 turns the
        # token position into a count. In place, because this is a captured path.
        lens = positions.to(torch.int32).add_(1)
        hip_ops.attn_paged_batch(q, gate, kp, vp, kps, vps,
                                 self.pool.pos2slot[:b], lens,
                                 self.attn_out[:b], self.max_seq, self.rep, self.scale,
                                 self.part_acc[:b], self.part_ml[:b])
        return self.o_proj.prefill(self.attn_out[:b].reshape(b, -1))

    def prefill_attend(self, qg: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                       positions: torch.Tensor, ctx, tiles: list | None = None,
                       row: int | None = None, rope=None) -> torch.Tensor:
        """Attention half of prefill: raw ``(qg, k, v)`` -> gated ``[C, nh*hd]``.

        Writes the C tokens into ``row``'s KV slots and attends over that row's
        own window.  ``o_proj`` is left to the caller, so a group prefill can run
        it once over every sequence's output (it is a weight read).

        ``rope`` is an optional explicit ``(cos, sin)`` pair of ``[C, rot]`` rows
        (``ops.mrope.MRopeMapper``), used by a chunk that contains image tokens:
        there the three position axes differ per token, so a single table lookup
        cannot produce the rotation, and the causal/KV side still uses
        ``positions``.  ``None`` keeps the 1-D table path unchanged.  When the
        argument is ``None`` but ``ctx`` carries a 4th entry, that entry is the
        rope pair: the prefill call chain (``Layer.prefill`` ->
        ``prefill_state``) only forwards ``ctx``, and this is the same pass-through
        style ``FullAttn.batch`` already uses for ``ctx_lens``.

        Args:
            qg: Packed query and gate, ``[C, 2*nh*hd]``.
            k: Keys, ``[C, nkv*hd]``.
            v: Values, ``[C, nkv*hd]``.
            positions: Absolute token indices of the chunk, ``[C]`` long; they
                address both the KV slots and, without explicit ``rope``, the
                rotation table.
            ctx: ``(cos_tbl, sin_tbl, mask_idx[, rope])``.
            tiles: ``(q0, q1, p0, kend)`` query tiles from ``Qwen35.prefill``.
                ``None`` treats the whole chunk as one tile, correct but with the
                wasted upper-triangle work described in :meth:`prefill`.
            row: Slot-map row to write. Defaults to the staging row.
            rope: Explicit ``(cos, sin)`` of ``[C, rot]`` for a chunk holding
                image tokens, where a single table lookup cannot produce the
                three-axis rotation.

        Returns:
            ``[C, nh*hd]`` gated attention output, before ``o_proj``.
        """
        if row is None:
            row = self.prefill_row
        cos_tbl, sin_tbl, mask_idx = ctx[0], ctx[1], ctx[2]
        if rope is None and len(ctx) > 3:
            rope = ctx[3]
        C = qg.shape[0]
        # The tile list is authoritative for the key range when present: its
        # bounds already account for a prefix the caller sliced off the front.
        if tiles is None:
            p0, p1 = int(positions[0]), int(positions[-1]) + 1
        else:
            p0, p1 = tiles[0][2], tiles[-1][3]
        qg = qg.reshape(C, self.nh, 2 * self.hd)
        q, gate = qg[..., :self.hd], qg[..., self.hd:]
        q = self.q_norm(q)
        k = self.k_norm(k.reshape(C, self.nkv, self.hd))
        v = v.reshape(C, self.nkv, self.hd)

        if rope is None:
            # [C, 1, rot]: the singleton broadcasts over heads.
            cos = cos_tbl.index_select(0, positions).unsqueeze(1)
            sin = sin_tbl.index_select(0, positions).unsqueeze(1)
        else:
            cos = rope[0].to(cos_tbl.dtype).unsqueeze(1)
            sin = rope[1].to(sin_tbl.dtype).unsqueeze(1)
        q, k = apply_rope(q, k, cos, sin, self.rot)

        qk, sk = self._quant_seq(k)
        qv, sv = self._quant_seq(v)
        if self.pool is not None:
            # Paged: scatter the C new tokens into the pool, then read back a dense
            # view for the attention itself. Prefill is compute-bound, so the
            # materialised view costs little and keeps this path identical to the
            # dense one (the reference paged implementation does the same).
            slots = self.pool.slots_for(positions, row)
            self.pool.write_range(self.pidx, slots, qk, sk, qv, sv)
            kk, vv, kkp, vvp = self.pool.gather(self.pidx, p1, row=row)
        else:
            self.kcache[:, p0:p1] = qk.transpose(0, 1)
            self.vcache[:, p0:p1] = qv.transpose(0, 1)
            self.kscale[:, p0:p1] = sk.transpose(0, 1)
            self.vscale[:, p0:p1] = sv.transpose(0, 1)
            # [nkv, S, hd] int8: only the prefix up to p1 is addressed by this
            # chunk, so the tail is never touched and needs no zeroing.
            kk = self.kcache[:, :p1]
            vv = self.vcache[:, :p1]
            kkp = self.kscale[:, :p1]
            vvp = self.vscale[:, :p1]

        # The two attention matmuls run on fp16 operands for the same reason the
        # projections do: this part has packed fp16 math and no packed bf16 math,
        # so the same einsum is ~2-3x faster (the accumulators stay fp32 on both
        # sides).  The rest of the block keeps bf16, which is what the KV cache
        # quantiser and the gates are written against.  The widening is done *per
        # key block*, not once for the whole prefix: widening the prefix once costs
        # 4 KB per token of context (1.07 GB at 256k) and that buffer has to be
        # live for the whole layer call, which is what a long prompt cannot afford.
        if tiles is None:
            tiles = [(0, C, p0, p1)]
        KB = PREFILL_ATTN_KB
        out = torch.empty(C, self.nh, self.hd, dtype=D, device="cuda")
        for q0, q1, _p0, kend in tiles:
            nq = q1 - q0
            qe = q[q0:q1].reshape(nq, self.nkv, self.rep, self.hd).permute(1, 2, 0, 3)
            qf = qe.to(WF)
            qpos = positions[q0:q1]
            # Online softmax over key blocks: acc is the unnormalised output, m the
            # running max and l the running denominator, all [nkv, rep, nq, .]
            acc = m = l = None
            for k0 in range(0, kend, KB):
                k1 = min(k0 + KB, kend)
                kb16 = kk[:, k0:k1].to(WF)
                scores = torch.einsum("ghcd,gsd->ghcs", qf, kb16).float()
                # One fused scale instead of two multiplies, and the causal mask in
                # place: the score block is the largest tensor in the layer, so every
                # extra pass over it is worth avoiding.  `tiles` carries absolute
                # positions, so the mask and the cache slice line up directly.
                scores *= (kkp[:, k0:k1] * self.scale)[:, None, None, :]
                scores.masked_fill_(
                    (mask_idx[None, k0:k1] > qpos[:, None])[None, None, :, :], NEG_INF)
                bmax = scores.amax(-1, keepdim=True)
                if acc is None:
                    # The first block always contains key 0, which every query can
                    # see, so bmax is finite here and no -inf arithmetic can leak in.
                    m = bmax
                    p = (scores - m).exp()
                    l = p.sum(-1, keepdim=True)
                else:
                    mnew = torch.maximum(m, bmax)
                    alpha = (m - mnew).exp()
                    p = (scores - mnew).exp()
                    l = l * alpha + p.sum(-1, keepdim=True)
                    acc *= alpha
                    m = mnew
                # Fold the per-token V scale into the weights (as the untiled
                # version does) so the einsum runs on raw int8-widened V.
                pw = (p * vvp[:, None, None, k0:k1]).to(WF)
                pv = torch.einsum("ghcs,gsd->ghcd", pw, vv[:, k0:k1].to(WF))
                acc = pv if acc is None else acc + pv
            # acc/l is the normalised tile output in (nkv, rep, nq, hd); the
            # permute restores (nq, nh, hd) before the gate is applied.
            out[q0:q1] = (acc / l).permute(2, 0, 1, 3).reshape(nq, self.nh, self.hd).to(D)
        out = out * gate.sigmoid()
        return out.reshape(C, -1)

    def prefill(self, x: torch.Tensor, positions: torch.Tensor, ctx,
                tiles: list | None = None, row: int | None = None,
                rope=None) -> torch.Tensor:
        """``x`` [C, hidden] -> [C, hidden]; writes C tokens into the KV cache.

        With ``tiles`` (``(q0, q1, kend)`` triples from ``Qwen35.prefill``) the
        attention runs query-tile by query-tile over a key range that stops at the
        tile's own end.  That is not just an L2 trick: the untiled version builds a
        full ``[nkv, rep, C, p1]`` score block and masks off the upper triangle, so
        half of its matmul and softmax work at pos=0 (a quarter at pos=2048) is
        thrown away.  Tiling cuts the score/softmax/PV traffic by the same fraction
        and makes the block small enough to stay in cache.  ``row`` selects the
        slot-map row (default: the staging row); the batched drafter passes each
        sequence's own row.  ``rope`` is the optional explicit 3-D ``(cos, sin)``
        for a chunk with image tokens (see :meth:`prefill_attend`).

        Args:
            x: Prefill inputs, ``[C, hidden]``.
            positions: Absolute token indices of the chunk, ``[C]`` long.
            ctx: ``(cos_tbl, sin_tbl, mask_idx[, rope])``.
            tiles: ``(q0, q1, p0, kend)`` query tiles, from ``Qwen35.prefill``.
            row: Slot-map row to write; ``None`` uses the staging row.
            rope: Explicit 3-D ``(cos, sin)`` for a chunk containing image tokens.

        Returns:
            ``[C, hidden]`` layer output after ``o_proj``, in the same order as
            ``x`` so the residual add lines up.
        """
        proj = self.prefill_project(x)
        out = self.prefill_attend(*proj, positions, ctx, tiles,
                                  self.prefill_row if row is None else row, rope)
        return self.o_proj.prefill(out)

    def verify(self, x: torch.Tensor, positions, cos_tbl, sin_tbl, mask_idx,
               rope_delta=None) -> torch.Tensor:
        """``x`` [C, hidden] -> [C, hidden] for C drafted tokens at ``positions``.

        The projections are batched; each token then runs through the same fused
        attention as the decode step, writing its own KV slot. Rollback is free
        here: the cache is position-indexed, so rewinding is just the position
        counter.  ``rope_delta`` (optional, 0-d or ``[1]`` long) is the request's
        mRoPE offset: ``positions`` keeps addressing KV slots, the rotation reads
        ``positions + rope_delta``.

        Rolling back by the position counter only works because a rejected draft
        leaves nothing behind: the next prefill overwrites those slots. That is
        why the loop writes each token's KV before attending -- a rejected token
        is never read again.

        Args:
            x: The drafted tokens' inputs, ``[C, hidden]``.
            positions: Token indices of the drafted tokens, ``[C]`` long.
            cos_tbl: RoPE cos table.
            sin_tbl: RoPE sin table.
            mask_idx: Device-side position counter for the mask.
            rope_delta: Optional mRoPE offset, 0-d or ``[1]`` long; moves only the
                rotation.

        Returns:
            ``[C, hidden]`` layer output, one row per drafted token, which the
            sampler compares against the draft to accept the longest prefix.
        """
        C = x.shape[0]
        qg, k, v = self.qkv.prefill(x)
        qg = qg.reshape(C, self.nh, 2 * self.hd)
        q, gate = qg[..., :self.hd], qg[..., self.hd:]
        q = self.q_norm(q)
        k = self.k_norm(k.reshape(C, self.nkv, self.hd))
        v = v.reshape(C, self.nkv, self.hd)
        rope_pos = positions if rope_delta is None else positions + rope_delta.reshape(())
        out = torch.empty(C, self.nh, self.hd, dtype=D, device="cuda")
        kp, vp, kps, vps = self._pool_views() if self.pool is not None else (None,) * 4
        for i in range(C):
            qi, ki, vi = q[i], k[i], v[i]
            cos = cos_tbl.index_select(0, rope_pos[i:i + 1])
            sin = sin_tbl.index_select(0, rope_pos[i:i + 1])
            qi, ki = apply_rope(qi, ki, cos, sin, self.rot)
            qk, sk = self._quant(ki)
            qv, sv = self._quant(vi)
            if self.pool is not None:
                self.pool.write_slot(self.pidx,
                                     self.pool.slots_for(positions[i:i + 1],
                                                         self.prefill_row),
                                     qk, sk, qv, sv)
                hip_ops.attn_paged(qi.contiguous(), gate[i].contiguous(), kp, vp, kps, vps,
                                   self.pool.pos2slot[self.prefill_row], self.attn_out1,
                                   positions[i:i + 1],
                                   self.max_seq, self.rep, self.scale,
                                   self.part_acc1, self.part_ml1)
            else:
                self.kcache.index_copy_(1, positions[i:i + 1], qk.unsqueeze(1))
                self.vcache.index_copy_(1, positions[i:i + 1], qv.unsqueeze(1))
                self.kscale.index_copy_(1, positions[i:i + 1], sk.unsqueeze(1))
                self.vscale.index_copy_(1, positions[i:i + 1], sv.unsqueeze(1))
                hip_ops.attn(qi.contiguous(), gate[i].contiguous(), self.kcache, self.vcache,
                             self.kscale, self.vscale, self.attn_out1, positions[i:i + 1],
                             self.rep, self.scale, self.part_acc1, self.part_ml1)
            out[i] = self.attn_out1
        return self.o_proj.prefill(out.reshape(C, -1))

    def reset(self) -> None:
        """Clear the dense KV cache between requests.

        Returns:
            Nothing. Zeroing is not needed for correctness of the next prefill --
            positions are overwritten before they are read -- but a stale int8
            scale from a previous request would be dequantised by any path that
            reads past the current length, so the buffers start neutral.

        Only the dense cache is cleared: with a pool attached the session owns
        the slots, and zeroing here would wipe a prefix another sequence shares.
        """
        if self.pool is not None:
            return
        self.kcache.zero_()
        self.vcache.zero_()
        self.kscale.zero_()
        self.vscale.zero_()
