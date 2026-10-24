"""Multi-token-prediction head: the speculative decoder's drafter.

One full-attention layer that, given the main model's hidden state at position
``i`` and the embedding of token ``i+1``, predicts ``i+2``.  Drafting runs as a
captured graph over fixed buffers with the chained argmax resolved on the
device, so K drafts are K replays and the host never sees a token between them.
"""

from __future__ import annotations

import torch

from models.qwen35.batch import batch_proj
from models.qwen35.blocks import MLP
from models.qwen35.full_attn import FullAttn
from models.qwen35.tuning import PREFILL_ATTN_TILE, PREFILL_MTP_CHUNK
from ops.dtypes import D
from ops.embed import TiedEmbedding
from ops.gemv import _proj
from ops.norm import RMSNorm
from paged_cache import PagedKVCache


class MTP:
    """Multi-token-prediction head: one full-attention layer that, given the main
    model's hidden ``h_i`` and the embedding of token ``i+1``, predicts ``i+2``.

    Drafting chains its own outputs (``h`` from the previous step feeds the
    next), and the next token is ``logits.argmax()`` on the device, so a whole
    draft never syncs to the host. The main model then verifies the draft.
    """

    def __init__(self, sd: dict, cfg: dict, bench: bool, max_seq: int) -> None:
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
        self.pos = torch.zeros(1, dtype=torch.long, device="cuda")

        # Drafting runs as a captured graph. The chain (h, token) -> (h', token')
        # lives entirely in these device buffers, so K drafts are K replays with
        # no host round trip and no per-op launch overhead; the chained argmax is
        # resolved on the device between replays.
        hidden = cfg["hidden_size"]
        self.tok_in = torch.zeros(1, dtype=torch.long, device="cuda")
        self.h_in = torch.zeros(hidden, dtype=D, device="cuda")
        self.logits_out = torch.zeros(self.embed.shape[0], dtype=D, device="cuda")
        self._ctx = None
        self._graph = None

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
        self._draft_graphs: dict[int, torch.cuda.CUDAGraph] = {}

    def reset(self) -> None:
        self.attn.reset()
        self.pos.zero_()

    def _draft_impl(self) -> None:
        """One draft step over fixed buffers; the tail token feeds the next."""
        cos_tbl, sin_tbl, mask_idx = self._ctx
        e = self.embed.lookup(self.tok_in).view(-1)
        x = self.fc(torch.cat([self.pre_norm_emb(e), self.pre_norm_hid(self.h_in)], dim=-1))
        h = x + self.attn(self.in_ln(x), self.pos, cos_tbl, sin_tbl, mask_idx)
        h = h + self.mlp(self.post_ln(h))
        h = self.norm(h)
        self.h_in.copy_(h)
        self.logits_out.copy_(self.embed(h))
        self.pos.add_(1)

    def capture_draft(self, ctx, warmup: int = 3) -> None:
        """Capture one draft step; :meth:`draft` chains K replays of it."""
        self._ctx = ctx
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                self._draft_impl()
        torch.cuda.current_stream().wait_stream(s)
        self.reset()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._draft_impl()
        self._graph = g
        self.reset()

    def draft(self, h_prev: torch.Tensor, tok0: torch.Tensor, K: int,
              out_toks: torch.Tensor) -> None:
        """Chain ``K`` drafts from ``(h_prev, tok0)``; writes K+1 ids into ``out_toks``.

        ``self.pos`` must already sit at the position of ``h_prev``.
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
        """One eager draft step at ``self.pos``; returns (h', logits) and advances pos."""
        cos_tbl, sin_tbl, mask_idx = ctx
        e = self.embed.lookup(tok.reshape(1)).view(-1)
        x = self.fc(torch.cat([self.pre_norm_emb(e), self.pre_norm_hid(h_prev)], dim=-1))
        h = x + self.attn(self.in_ln(x), self.pos, cos_tbl, sin_tbl, mask_idx)
        h = h + self.mlp(self.post_ln(h))
        h = self.norm(h)
        self.pos.add_(1)
        return h, self.embed(h)

    def prefill(self, H: torch.Tensor, next_toks: torch.Tensor, positions: torch.Tensor,
                ctx) -> None:
        """Fill the MTP KV cache over a run of prompt positions.

        ``H[i]`` is the main model's hidden state at ``positions[i]`` and
        ``next_toks[i]`` is the token that follows it -- the MTP head is trained on
        exactly that pair, so the caller passes the already-shifted tokens (there is
        no shift inside, or a chunked caller would shift twice).

        The queries are tiled and the keys streamed like the main attention
        (``PREFILL_ATTN_TILE`` / ``PREFILL_ATTN_KB``): without the query tiles the
        score block is ``[nkv, rep, n, kend]``, which is 1 GiB at 16k tokens and
        17 GB at 64k.
        """
        n = min(int(positions.shape[0]), int(H.shape[0]), int(next_toks.shape[0]))
        if n <= 0:
            return
        e = self.embed.lookup(next_toks[:n])
        # ``fc`` must go through the batched path: the fused GEMV takes a single
        # [K] vector, so calling it with [n, 2*hidden] silently reads row 0 only.
        x = self.fc.prefill(torch.cat([self.pre_norm_emb(e), self.pre_norm_hid(H[:n])],
                                      dim=-1))
        Q = PREFILL_ATTN_TILE
        for c0 in range(0, n, PREFILL_MTP_CHUNK):
            c1 = min(c0 + PREFILL_MTP_CHUNK, n)
            n_loc = c1 - c0
            base = int(positions[c0])
            tiles = ([(q0, min(q0 + Q, n_loc), base, base + min(q0 + Q, n_loc))
                      for q0 in range(0, n_loc, Q)]
                     if n_loc > Q else None)
            x[c0:c1] = x[c0:c1] + self.attn.prefill(self.in_ln(x[c0:c1]),
                                                    positions[c0:c1], ctx, tiles)
            x[c0:c1] = x[c0:c1] + self.mlp.prefill(self.post_ln(x[c0:c1]))
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
        for row in range(capacity):
            self.pool.assign_table([row * bpr + i for i in range(bpr)], 0, row)
        self.attn_b = FullAttn(self._attn_sd, "mtp.layers.0.self_attn", self._cfg,
                               self._bench, self.batch_len, paged=True)
        self.attn_b.attach_pool(self.pool, 0)
        self.attn_b.enable_batching(capacity, 0)
        self.b_pos = torch.zeros(capacity, dtype=torch.long, device="cuda")
        self.b_tok = torch.zeros(capacity, dtype=torch.long, device="cuda")
        self.b_h = torch.zeros(capacity, hidden, dtype=D, device="cuda")
        self.b_h_prev = torch.zeros(capacity, hidden, dtype=D, device="cuda")
        self.b_logits = torch.zeros(capacity, self.embed.shape[0], dtype=D, device="cuda")
        self._ctx_lens = torch.ones(capacity, dtype=torch.int32, device="cuda")
        self.capture_draft_batch()

    def check_len(self, max_position: int) -> None:
        """Raise if a row's context would run past the drafter pool."""
        if self.pool is not None and max_position >= self.batch_len:
            raise RuntimeError(
                f"batch spec needs context < {self.batch_len} (got {max_position}); "
                "disable spec for this request or raise max_len")

    def _draft_batch_impl(self, rows: int) -> None:
        """One draft step for ``rows`` rows: the chain tail feeds the next replay."""
        cos_tbl, sin_tbl, mask_idx = self._ctx
        self._ctx_lens[:rows].copy_(
            self.b_pos[:rows].to(torch.int32).add_(1))
        e = self.embed.lookup(self.b_tok[:rows])
        x = batch_proj(self.fc, torch.cat([self.pre_norm_emb(e),
                                           self.pre_norm_hid(self.b_h[:rows])], dim=-1))
        h = x + self.attn_b.batch(self.in_ln(x), self.b_pos[:rows],
                                  (cos_tbl, sin_tbl, mask_idx, self._ctx_lens))
        h = h + self.mlp.batch(self.post_ln(h))
        h = self.norm(h)
        self.b_h[:rows].copy_(h)
        self.b_logits[:rows].copy_(batch_proj(self.embed.gemv, h))
        self.b_pos[:rows].add_(1)

    def capture_draft_batch(self) -> None:
        """One graph per drafter bucket; :meth:`draft_batch` chains K replays."""
        assert self.pool is not None, "enable_spec_batch() first"
        self._draft_graphs = {}
        for b in self.batch_buckets:
            impl = lambda b=b: self._draft_batch_impl(b)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    impl()
            torch.cuda.current_stream().wait_stream(s)
            self.b_pos.zero_()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                impl()
            self._draft_graphs[b] = g
        self.b_pos.zero_()

    def draft_batch(self, bucket: int, K: int, out_toks: torch.Tensor) -> None:
        """Chain ``K`` drafts for every row of ``bucket``; writes ``[bucket, K+1]``.

        ``b_h``/``b_tok``/``b_pos`` must be staged by the caller; the chained
        argmax stays on the device between replays, like the single-stream draft.
        """
        g = self._draft_graphs[bucket]
        out_toks[:, 0].copy_(self.b_tok[:bucket])
        for i in range(1, K + 1):
            g.replay()
            self.b_tok[:bucket].copy_(self.b_logits[:bucket].argmax(-1))
            out_toks[:, i].copy_(self.b_tok[:bucket])

    @torch.no_grad()
    def prefill_row(self, row: int, H: torch.Tensor, next_toks: torch.Tensor,
                    positions: torch.Tensor, ctx) -> None:
        """Fill one row's paged drafter KV over a run of prompt positions.

        Same maths as :meth:`prefill`, but the attention writes/reads ``row``'s
        slot map instead of the single-stream dense cache.
        """
        assert self.attn_b is not None, "enable_spec_batch() first"
        n = min(int(positions.shape[0]), int(H.shape[0]), int(next_toks.shape[0]))
        if n <= 0:
            return
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
            x[c0:c1] = x[c0:c1] + self.attn_b.prefill(
                self.in_ln(x[c0:c1]), positions[c0:c1], ctx, tiles, row=row)
            x[c0:c1] = x[c0:c1] + self.mlp.prefill(self.post_ln(x[c0:c1]))

    def close(self) -> None:
        """Drop the captured draft graphs; they pin their own private memory pools."""
        self._graph = None
        self._draft_graphs = {}
