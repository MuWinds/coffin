"""Multi-token-prediction head: the speculative decoder's drafter.

Module: ``coffin/models/qwen35``. Sits on top of the family's own blocks: it
reuses ``FullAttn`` (as ``mtp.layers.0``), ``MLP``, the tied embedding and the
batched projections, and is driven by ``model.py`` (``spec_generate`` /
``spec_step`` / ``spec_step_batch``). It owns a second paged pool for the
drafter KV and must not outlive the model that built it.

One full-attention layer that, given the main model's hidden state at position
``i`` and the embedding of token ``i+1``, predicts ``i+2``.  Drafting runs as a
captured graph over fixed buffers with the chained argmax resolved on the
device, so K drafts are K replays and the host never sees a token between them.

Constraints:
    * The draft chain lives entirely in device buffers (``h_in``, ``tok_in``,
      ``logits_out``, ``b_*``). A host read between replays costs a full sync
      per drafted token (~0.5 ms at batch 1) and destroys the point of
      speculation, so the chained argmax must stay on the device.
    * Graphs bake in buffer addresses and the pool tensors. ``capture_draft`` /
      ``capture_draft_batch`` must run after every allocation, and the buffers
      they write may never be reallocated while the graph is live.
    * Capture goes through ``ops.runtime``, so a backend that cannot capture (its
      operator table mixes library and hand-written kernels) runs the same body
      eagerly instead. The chained argmax still stays on the device either way,
      which is what keeps the no-host-round-trip constraint true on both paths.
    * The drafter KV is rewritten at the main model's draft positions and rolled
      back by the main verify, so it only needs the same window; a prompt past
      ``batch_len`` is truncated and reported through ``check_len`` rather than
      written past the pool.
    * ``self.pos`` is the KV address and ``rope_delta`` the mRoPE offset. mRoPE
      moves the rotation only -- the same split the main attention uses, and
      mixing them up silently rotates by the wrong position after an image.
    * All mutation is single-threaded, driven by ``llm_engine``; there is no
      locking and none may be added.
"""

from __future__ import annotations

import torch

from backends import device as _device
from models.qwen35.batch import batch_proj
from models.qwen35.blocks import MLP
from models.qwen35.full_attn import FullAttn
from models.qwen35.tuning import PREFILL_ATTN_TILE, PREFILL_MTP_CHUNK
from ops import runtime
from ops.dtypes import D
from ops.embed import TiedEmbedding
from ops.gemv import _proj
from ops.norm import RMSNorm
from cache.paged_cache import PagedKVCache


class MTP:
    """Multi-token-prediction head: one full-attention layer that, given the main
    model's hidden ``h_i`` and the embedding of token ``i+1``, predicts ``i+2``.

    Drafting chains its own outputs (``h`` from the previous step feeds the
    next), and the next token is ``logits.argmax()`` on the device, so a whole
    draft never syncs to the host. The main model then verifies the draft.
    """

    def __init__(self, sd: dict, cfg: dict, bench: bool, max_seq: int) -> None:
        """Load the MTP head and allocate the single-stream draft buffers.

        Args:
            sd: Full state dict; this head reads the ``mtp.*`` keys plus the tied
                ``model.language_model.embed_tokens.weight``.
            cfg: Model config; ``hidden_size`` and the attention geometry pass
                straight through to the embedded ``FullAttn`` layer.
            bench: Threaded into ``_proj`` to select the benchmark weight layout.
            max_seq: Context window handed to the embedded full-attention layer,
                which sizes its dense (non-paged) KV cache from it.
        """
        p = "mtp"
        self.embed = TiedEmbedding(sd["model.language_model.embed_tokens.weight"], bench)
        self.pre_norm_emb = RMSNorm(sd[f"{p}.pre_fc_norm_embedding.weight"])
        self.pre_norm_hid = RMSNorm(sd[f"{p}.pre_fc_norm_hidden.weight"])
        self.fc = _proj(sd, [f"{p}.fc"], bench)
        self.in_ln = RMSNorm(sd[f"{p}.layers.0.input_layernorm.weight"])
        self.attn = FullAttn(sd, f"{p}.layers.0.self_attn", cfg, bench, max_seq)
        self.post_ln = RMSNorm(sd[f"{p}.layers.0.post_attention_layernorm.weight"])
        self.mlp = MLP(sd, f"{p}.layers.0.mlp", bench)
        self.norm = RMSNorm(sd[f"{p}.norm.weight"])
        self.pos = torch.zeros(1, dtype=torch.long, device=_device)
        #: mRoPE offset of the request being drafted, matching ``Qwen35.rope_delta``:
        #: the drafter's rotation reads ``pos + rope_delta`` while ``pos`` keeps
        #: addressing the KV slot.  The caller (``Qwen35.spec_step`` /
        #: ``spec_step_batch``) copies the row's value in before a draft, because
        #: the captured draft graph can only add a buffer, not branch on one.
        self.rope_delta = torch.zeros(1, dtype=torch.long, device=_device)

        # Drafting runs as a captured graph. The chain (h, token) -> (h', token')
        # lives entirely in these device buffers, so K drafts are K replays with
        # no host round trip and no per-op launch overhead; the chained argmax is
        # resolved on the device between replays.
        hidden = cfg["hidden_size"]
        self.tok_in = torch.zeros(1, dtype=torch.long, device=_device)
        self.h_in = torch.zeros(hidden, dtype=D, device=_device)
        self.logits_out = torch.zeros(self.embed.shape[0], dtype=D, device=_device)
        self._ctx = None
        self._graph: runtime.Capture | None = None

        # Batched drafter (one row per sequence).  Kept entirely separate from the
        # single-stream buffers above so the speculative path stays untouched.
        self._sd, self._cfg, self._bench = sd, cfg, bench
        # Keep the attention tensors alive even when the caller clears ``sd`` after
        # the model is built (the batch drafter builds its paged twin lazily).
        self._attn_sd = {k: v for k, v in sd.items()
                         if k.startswith("mtp.layers.0.self_attn.")}
        self._max_seq = max_seq
        self.batch_capacity = 0
        self.batch_buckets: tuple[int, ...] = ()
        self.batch_len = 0               # drafter positions a row can address
        self.pool: PagedKVCache | None = None
        self.attn_b: FullAttn | None = None
        self.b_pos = self.b_tok = self.b_h = self.b_logits = None
        self.b_h_prev = None
        self._ctx_lens = None
        self._draft_graphs: dict[int, runtime.Capture] = {}

    def reset(self) -> None:
        """Rewind the single-stream drafter to position 0.

        Returns:
            Nothing. The MTP KV is position-indexed and never read past ``pos``,
            so moving the counter back is a complete rollback; the embedded
            attention layer's own cache is cleared too.
        """
        self.attn.reset()
        self.pos.zero_()
        self.rope_delta.zero_()

    def _draft_impl(self) -> None:
        """One draft step over fixed buffers; the tail token feeds the next.

        Returns:
            Nothing. Reads ``tok_in`` / ``h_in`` / ``pos`` and writes back the new
            hidden, the logits and the advanced position -- all in place, because
            this body is what the captured graph replays.
        """
        cos_tbl, sin_tbl, mask_idx = self._ctx[0], self._ctx[1], self._ctx[2]
        e = self.embed.lookup(self.tok_in).view(-1)
        x = self.fc(torch.cat([self.pre_norm_emb(e), self.pre_norm_hid(self.h_in)], dim=-1))
        h = x + self.attn(self.in_ln(x), self.pos, cos_tbl, sin_tbl, mask_idx,
                          self.rope_delta)
        h = h + self.mlp(self.post_ln(h))
        h = self.norm(h)
        self.h_in.copy_(h)
        self.logits_out.copy_(self.embed(h))
        self.pos.add_(1)

    def capture_draft(self, ctx, warmup: int = 3) -> None:
        """Capture one draft step; :meth:`draft` chains K replays of it.

        Args:
            ctx: The main model's ``(cos_tbl, sin_tbl, mask_idx)``. Held, not
                copied, because the graph reads those exact tensors.
            warmup: Replays on a side stream before capturing. The kernels need
                their shapes warmed or capture records an allocation it cannot
                replay.

        Returns:
            Nothing. ``_graph`` holds the captured step; a second call replaces
            it, and the position counters are reset on both sides of the capture
            so the first replay starts at 0.
        """
        self._ctx = ctx
        self.reset()
        self._graph = runtime.capture(self._draft_impl, warmup)
        self.reset()

    def draft(self, h_prev: torch.Tensor, tok0: torch.Tensor, K: int,
              out_toks: torch.Tensor) -> None:
        """Chain ``K`` drafts from ``(h_prev, tok0)``; writes K+1 ids into ``out_toks``.

        ``self.pos`` must already sit at the position of ``h_prev``.

        Args:
            h_prev: The main model's hidden state for the last accepted token.
            tok0: That token's id, the first input of the chain.
            K: Drafts to produce; ``K`` replays of the captured step.
            out_toks: Long buffer of at least ``K+1`` entries. ``out_toks[0]`` is
                ``tok0`` and ``out_toks[i]`` the ``i``-th draft, which is the
                layout the verify step indexes.

        Raises:
            AssertionError: The graph was not captured. ``capture_draft`` must run
                after the model is built and before the first draft.
        """
        assert self._graph is not None, "draft graph not captured"
        self.h_in.copy_(h_prev)
        self.tok_in.copy_(tok0.reshape(1))
        # Slice, not index: ``out_toks[0]`` on a 1-D tensor is a 0-dim view and
        # ``copy_`` from a [1] tensor into it fails the shape check.
        out_toks[0:1].copy_(self.tok_in)
        for i in range(1, K + 1):
            self._graph.replay()
            self.tok_in.copy_(self.logits_out.argmax())
            out_toks[i:i + 1].copy_(self.tok_in)

    def step(self, h_prev: torch.Tensor, tok: torch.Tensor, ctx) -> tuple[torch.Tensor, torch.Tensor]:
        """One eager draft step at ``self.pos``; returns (h', logits) and advances pos.

        ``ctx`` is the main model's ``(cos_tbl, sin_tbl, mask_idx)`` (an optional
        4th entry is ignored); the rotation adds :attr:`rope_delta`, which the
        caller keeps in step with the main model's row.

        Args:
            h_prev: The main model's hidden state at the current position.
            tok: The token id to draft from.
            ctx: The main model's ``(cos_tbl, sin_tbl, mask_idx[, rope])``; the
                rope entry is ignored here because this path handles no image
                chunk.

        Returns:
            ``(h, logits)`` for the predicted next token. The caller samples from
            ``logits``; the drafter does not commit a token on this path.
        """
        cos_tbl, sin_tbl, mask_idx = ctx[0], ctx[1], ctx[2]
        e = self.embed.lookup(tok.reshape(1)).view(-1)
        x = self.fc(torch.cat([self.pre_norm_emb(e), self.pre_norm_hid(h_prev)], dim=-1))
        h = x + self.attn(self.in_ln(x), self.pos, cos_tbl, sin_tbl, mask_idx,
                          self.rope_delta)
        h = h + self.mlp(self.post_ln(h))
        h = self.norm(h)
        self.pos.add_(1)
        return h, self.embed(h)

    def prefill(self, H: torch.Tensor, next_toks: torch.Tensor, positions: torch.Tensor,
                ctx, rope=None) -> None:
        """Fill the MTP KV cache over a run of prompt positions.

        ``H[i]`` is the main model's hidden state at ``positions[i]`` and
        ``next_toks[i]`` is the token that follows it -- the MTP head is trained on
        exactly that pair, so the caller passes the already-shifted tokens (there is
        no shift inside, or a chunked caller would shift twice).

        ``rope`` is an optional explicit ``(cos, sin)`` pair of ``[n, rot]`` rows
        for a prompt region that contains image tokens (``ops.mrope``): the
        drafter is a full-attention layer over the same positions as the main
        model, so it has to rotate identically.  ``None`` keeps the 1-D table
        path; either way ``positions`` addresses the KV slots and ``ctx`` stays
        the main model's tuple.

        The queries are tiled and the keys streamed like the main attention
        (``PREFILL_ATTN_TILE`` / ``PREFILL_ATTN_KB``): without the query tiles the
        score block is ``[nkv, rep, n, kend]``, which is 1 GiB at 16k tokens and
        17 GB at 64k.

        Args:
            H: Main-model hidden states, ``[n, hidden]``.
            next_toks: Token ids following each row of ``H``, ``[n]`` long.
            positions: Absolute positions of those tokens, ``[n]`` long.
            ctx: The main model's ``(cos_tbl, sin_tbl, mask_idx[, rope])``.
            rope: Explicit 3-D ``(cos, sin)`` for an image-bearing region.

        Returns:
            Nothing. The cache is filled and ``self.pos`` is moved past the last
            position, so the next draft continues from there.
        """
        n = min(int(positions.shape[0]), int(H.shape[0]), int(next_toks.shape[0]))
        if n <= 0:
            return
        # The caller may put the rope pair on ctx instead of the argument.
        if len(ctx) > 3:
            if rope is None:
                rope = ctx[3]
            ctx = ctx[:3]
        e = self.embed.lookup(next_toks[:n])
        # ``fc`` must go through the batched path: the fused entry takes a single
        # [K] vector, so calling it with [n, 2*hidden] silently reads row 0 only.
        x = self.fc.prefill(torch.cat([self.pre_norm_emb(e), self.pre_norm_hid(H[:n])],
                                      dim=-1))
        Q = PREFILL_ATTN_TILE
        for c0 in range(0, n, PREFILL_MTP_CHUNK):
            c1 = min(c0 + PREFILL_MTP_CHUNK, n)
            n_loc = c1 - c0
            base = int(positions[c0])
            # Tiles are absolute: ``(q0, q1, p0, kend)`` in chunk coordinates, so
            # the last tile's causal bound is the chunk end.
            tiles = ([(q0, min(q0 + Q, n_loc), base, base + min(q0 + Q, n_loc))
                      for q0 in range(0, n_loc, Q)]
                     if n_loc > Q else None)
            rope_c = None if rope is None else (rope[0][c0:c1], rope[1][c0:c1])
            x[c0:c1] = x[c0:c1] + self.attn.prefill(self.in_ln(x[c0:c1]),
                                                    positions[c0:c1], ctx, tiles,
                                                    rope=rope_c)
            x[c0:c1] = x[c0:c1] + self.mlp.prefill(self.post_ln(x[c0:c1]))
            # The final norm is not applied per chunk: the caller consumes the
            # hidden state through ``_draft_impl``, which normalises once at the
            # end. Computing it here would cost a full [n, hidden] pass and change
            # nothing the next replay reads.
            _ = self.norm(x[c0:c1])
        self.pos.fill_(int(positions[n - 1]) + 1)

    def enable_spec_batch(self, capacity: int, buckets: tuple[int, ...],
                          ctx, block_size: int = 64,
                          max_len: int = 16384) -> None:
        """Give the drafter per-row buffers, a paged KV pool and captured graphs.

        The drafter's KV is rewritten at the same positions as the main model's
        drafts and rolled back by the main verify, so it only needs the same
        context window: ``min(max_seq, max_len)`` positions per row (the cap keeps
        the pool small enough to coexist with the decode graphs; longer contexts
        are reported to the caller by :meth:`check_len`).

        Args:
            capacity: Decode rows.
            buckets: Batch sizes to capture a draft graph for; one graph each,
                and every bucket must be ``<= capacity``.
            ctx: The main model's ``(cos_tbl, sin_tbl, mask_idx)``.
            block_size: Paged-KV block size; the per-row block count
                ``batch_len / block_size`` is rounded down, so a row always
                addresses whole blocks.
            max_len: Upper bound on drafter positions per row.

        Returns:
            Nothing. Idempotent for an unchanged ``(capacity, buckets)``; a
            change drops the old pool and graphs first, because a captured graph
            cannot be pointed at a new pool.
        """
        capacity = max(int(capacity), 1)
        if (self.pool is not None and self.batch_capacity == capacity
                and self.batch_buckets == tuple(buckets)):
            return
        if self.pool is not None:
            # Rebuilding for different buckets (e.g. a second test class on the
            # same model): drop the old drafter state before allocating the new.
            self.pool = None
            self.attn_b = None
            self._draft_graphs = {}
        self._ctx = ctx
        self.batch_capacity = capacity
        self.batch_buckets = tuple(buckets)
        hidden = self._cfg["hidden_size"]
        bs = int(block_size)
        bpr = max(1, min(int(max_len), int(self._max_seq)) // bs)
        self.batch_len = bpr * bs
        nkv, hd = self.attn.nkv, self.attn.hd
        self.pool = PagedKVCache(1, nkv, hd, num_blocks=capacity * bpr,
                                 block_size=bs, max_seq=self.batch_len,
                                 rows=capacity)
        # Static block tables: row ``r`` owns blocks ``[r*bpr, (r+1)*bpr)``, so
        # the draft path runs no allocator and no table mutation.
        for row in range(capacity):
            self.pool.assign_table([row * bpr + i for i in range(bpr)], 0, row)
        self.attn_b = FullAttn(self._attn_sd, "mtp.layers.0.self_attn", self._cfg,
                               self._bench, self.batch_len, paged=True)
        self.attn_b.attach_pool(self.pool, 0)
        self.attn_b.enable_batching(capacity, 0)
        self.b_pos = torch.zeros(capacity, dtype=torch.long, device=_device)
        self.b_tok = torch.zeros(capacity, dtype=torch.long, device=_device)
        self.b_h = torch.zeros(capacity, hidden, dtype=D, device=_device)
        self.b_h_prev = torch.zeros(capacity, hidden, dtype=D, device=_device)
        self.b_logits = torch.zeros(capacity, self.embed.shape[0], dtype=D, device=_device)
        self._ctx_lens = torch.ones(capacity, dtype=torch.int32, device=_device)
        #: Per-row mRoPE offset of the batched drafter, mirrored from the main
        #: model's ``rope_delta`` by the caller (``Qwen35.spec_step_batch``); 0 for
        #: a text-only bucket, so the default bucket is unchanged.
        self.b_rope_delta = torch.zeros(capacity, dtype=torch.long, device=_device)
        self.capture_draft_batch()

    def check_len(self, max_position: int) -> None:
        """Raise if a row's context would run past the drafter pool.

        Args:
            max_position: The row's last position, i.e. context length minus one.

        Raises:
            RuntimeError: The position is at or past ``batch_len``. The pool has
                no slot to write, so the caller must disable speculation for this
                request or raise ``max_len``; truncating silently would draft
                against a KV that disagrees with the main model.
        """
        if self.pool is not None and max_position >= self.batch_len:
            raise RuntimeError(
                f"batch spec needs context < {self.batch_len} (got {max_position}); "
                "disable spec for this request or raise max_len")

    def _draft_batch_impl(self, rows: int) -> None:
        """One draft step for ``rows`` rows: the chain tail feeds the next replay.

        Args:
            rows: Active rows, i.e. the leading slice of every ``b_*`` buffer.

        Returns:
            Nothing. All state moves in place, because this body is the captured
            bucket graph.
        """
        cos_tbl, sin_tbl, mask_idx = self._ctx[0], self._ctx[1], self._ctx[2]
        # +1 in place: the entry takes a length, not a last index, and this runs
        # inside the captured graph, so it cannot allocate.
        self._ctx_lens[:rows].copy_(
            self.b_pos[:rows].to(torch.int32).add_(1))
        e = self.embed.lookup(self.b_tok[:rows])
        x = batch_proj(self.fc, torch.cat([self.pre_norm_emb(e),
                                           self.pre_norm_hid(self.b_h[:rows])], dim=-1))
        h = x + self.attn_b.batch(self.in_ln(x), self.b_pos[:rows],
                                  (cos_tbl, sin_tbl, mask_idx, self._ctx_lens,
                                   self.b_rope_delta))
        h = h + self.mlp.batch(self.post_ln(h))
        h = self.norm(h)
        self.b_h[:rows].copy_(h)
        self.b_logits[:rows].copy_(batch_proj(self.embed.gemv, h))
        self.b_pos[:rows].add_(1)

    def capture_draft_batch(self) -> None:
        """One graph per drafter bucket; :meth:`draft_batch` chains K replays.

        Returns:
            Nothing. The position counters are zeroed before and after capture so
            every bucket graph starts at position 0.

        Raises:
            AssertionError: No pool exists yet, i.e. ``enable_spec_batch`` was not
                called after the model was built.
        """
        assert self.pool is not None, "enable_spec_batch() first"
        self._draft_graphs = {}
        for b in self.batch_buckets:
            impl = lambda b=b: self._draft_batch_impl(b)
            self.b_pos.zero_()
            self._draft_graphs[b] = runtime.capture(impl, warmup=2)
        self.b_pos.zero_()

    def draft_batch(self, bucket: int, K: int, out_toks: torch.Tensor) -> None:
        """Chain ``K`` drafts for every row of ``bucket``; writes ``[bucket, K+1]``.

        ``b_h``/``b_tok``/``b_pos`` must be staged by the caller; the chained
        argmax stays on the device between replays, like the single-stream draft.

        Args:
            bucket: Batch size whose graph to replay. It must be one of the
                captured buckets: a larger one would overrun the buffers, a
                smaller one would leave rows undrafted.
            K: Drafts to produce per row.
            out_toks: Long buffer ``[bucket, K+1]``; column 0 is the input token,
                column ``i`` the ``i``-th draft.
        """
        g = self._draft_graphs[bucket]
        out_toks[:, 0].copy_(self.b_tok[:bucket])
        for i in range(1, K + 1):
            g.replay()
            self.b_tok[:bucket].copy_(self.b_logits[:bucket].argmax(-1))
            out_toks[:, i].copy_(self.b_tok[:bucket])

    @torch.no_grad()
    def prefill_row(self, row: int, H: torch.Tensor, next_toks: torch.Tensor,
                    positions: torch.Tensor, ctx, rope=None) -> None:
        """Fill one row's paged drafter KV over a run of prompt positions.

        Same maths as :meth:`prefill`, but the attention writes/reads ``row``'s
        slot map instead of the single-stream dense cache.  ``rope`` is the same
        optional explicit 3-D ``(cos, sin)`` pair (image tokens in this region).

        Args:
            row: Decode row whose block table addresses the drafter pool.
            H: Main-model hidden states, ``[n, hidden]``.
            next_toks: Token ids following each row of ``H``, ``[n]`` long.
            positions: Absolute positions of those tokens, ``[n]`` long.
            ctx: The main model's ``(cos_tbl, sin_tbl, mask_idx[, rope])``.
            rope: Explicit 3-D ``(cos, sin)`` for an image-bearing region.

        Returns:
            Nothing. A run that would cross ``batch_len`` is truncated to the part
            that fits, so the row's later positions are dropped and the caller
            falls back through :meth:`check_len` instead of writing past the pool.

        Raises:
            AssertionError: ``enable_spec_batch`` has not built the paged drafter.
        """
        assert self.attn_b is not None, "enable_spec_batch() first"
        n = min(int(positions.shape[0]), int(H.shape[0]), int(next_toks.shape[0]))
        if n <= 0:
            return
        # The caller may put the rope pair on ctx instead of the argument.
        if len(ctx) > 3:
            if rope is None:
                rope = ctx[3]
            ctx = ctx[:3]
        # The drafter pool only addresses positions in [0, batch_len): a prompt
        # longer than the drafter window must not write past it.  Prefill the
        # prefix that fits and let spec_step_batch report the overflow via
        # check_len(), so the caller falls back to a plain decode step instead of
        # crashing inside the attention kernel.
        n = min(n, self.batch_len - int(positions[0]))
        if n <= 0:
            return
        e = self.embed.lookup(next_toks[:n])
        x = self.fc.prefill(torch.cat([self.pre_norm_emb(e),
                                       self.pre_norm_hid(H[:n])], dim=-1))
        Q = PREFILL_ATTN_TILE
        for c0 in range(0, n, PREFILL_MTP_CHUNK):
            c1 = min(c0 + PREFILL_MTP_CHUNK, n)
            n_loc = c1 - c0
            base = int(positions[c0])
            tiles = ([(q0, min(q0 + Q, n_loc), base, base + min(q0 + Q, n_loc))
                      for q0 in range(0, n_loc, Q)]
                     if n_loc > Q else None)
            rope_c = None if rope is None else (rope[0][c0:c1], rope[1][c0:c1])
            x[c0:c1] = x[c0:c1] + self.attn_b.prefill(
                self.in_ln(x[c0:c1]), positions[c0:c1], ctx, tiles, row=row,
                rope=rope_c)
            x[c0:c1] = x[c0:c1] + self.mlp.prefill(self.post_ln(x[c0:c1]))

    def close(self) -> None:
        """Drop the captured draft graphs; they pin their own private memory pools.

        Returns:
            Nothing. The graphs hold device memory no allocator can reclaim while
            they exist, so a model reload must call this first or the next model
            sees less free VRAM than this one used.
        """
        self._graph = None
        self._draft_graphs = {}
