"""The Qwen3.5-4B text engine: weights to layers to captured graphs.

The assembly point of the family: it owns the layer list, the rope tables, the
fixed graph buffers and the three execution paths -- decode (one graph replay),
prefill (batched GEMMs with tiled causal attention) and verify (a captured
C-token forward for speculative decoding).  The primitives it is built from
live in ``ops/``; the per-layer blocks are next to this module.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from models.qwen35.batch import (MULTI_Q_WIRED, batch_proj, multi_q_ready,
                                 wide_kernel_ready, wide_proj)
from models.qwen35.blocks import Layer
from models.qwen35.full_attn import FullAttn
from models.qwen35.linear_attn import LinearAttn
from models.qwen35.mtp import MTP
from models.qwen35.tuning import CKPT, PREFILL_ATTN_TILE
from models.qwen35.weights import load_weights
from ops.dtypes import D, WF
from ops.embed import TiedEmbedding
from ops.mrope import mrope_cos_sin, scatter_image_embeds
from ops.norm import RMSNorm
from ops.rope import make_inv_freq, make_rope_tables
from paged_cache import PagedKVCache


class Qwen35:
    def __init__(self, model_dir: str | Path, max_seq: int = 4096, bench: bool = True,
                 weights: dict | None = None, paged: bool = False) -> None:
        """``paged=True`` builds the full-attention layers without their private
        dense KV caches, for the caller that is going to call
        :meth:`enable_paging` (PrefixCacheEngine does). The MTP layer is not paged
        -- its cache is not part of a prefix snapshot -- so it keeps its own."""
        model_dir = Path(model_dir)
        self.raw_cfg = json.loads((model_dir / "config.json").read_text())
        self.cfg = self.raw_cfg["text_config"]
        self.max_seq = max_seq
        sd = weights if weights is not None else load_weights(model_dir)

        self.embed = TiedEmbedding(sd["model.language_model.embed_tokens.weight"], bench)
        self.norm = RMSNorm(sd["model.language_model.norm.weight"])
        self.layers = [Layer(sd, i, self.cfg, bench, max_seq, paged)
                       for i in range(self.cfg["num_hidden_layers"])]
        self.lm_head = self.embed
        self.mtp = MTP(sd, self.cfg, bench, max_seq) if "mtp.fc.weight" in sd else None
        #: Vision tensors stay in the resident dict but are only turned into a
        #: tower on the first image request (see ``enable_vision``); a text-only
        #: server therefore carries the 0.62 GiB but never builds the module.
        self.vision_sd = {k: v for k, v in sd.items()
                          if k.startswith("model.visual.")}
        self.vision = None
        if weights is None:                 # caller-owned dict stays resident
            sd.clear()
        torch.cuda.empty_cache()

        rp = self.cfg["rope_parameters"]
        self.rot = int(self.cfg["head_dim"] * rp["partial_rotary_factor"])
        #: The one frequency axis: the 1-D decode tables and the 3-D mRoPE tables
        #: both derive from it, so a text-only prompt gets identical cos/sin on
        #: either path.
        self.inv_freq = make_inv_freq(self.cfg["head_dim"], self.rot,
                                      rp["rope_theta"], "cuda")
        self.cos_tbl, self.sin_tbl = make_rope_tables(
            self.cfg["head_dim"], self.rot, rp["rope_theta"], max_seq, "cuda",
            inv_freq=self.inv_freq)

        # graph-friendly fixed buffers.  ``pos``/``token_in``/``logits_out`` grow a
        # leading row dim when ``enable_batching`` is used; with the default
        # capacity of 1 they keep their single-stream shapes exactly.  When batching
        # is on, row ``prefill_row`` (the last one) is reserved for the single-stream
        # prefill / verify paths, so prefilling a new request never touches a decode
        # row (see ``enable_batching``).
        self.batch_capacity = 1
        self.decode_capacity = 1
        self.prefill_row = 0
        #: ``False`` in the batched engine: speculative decoding is single-stream,
        #: so the MTP head's prefill is dead work there and is skipped.
        self.mtp_prefill = True
        self.pool = None
        self._batch_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._scratch_blocks: list[int] = []
        self._row_tables: dict[int, tuple] = {}
        self._sentinel_token = 0
        self.pos = torch.zeros(1, dtype=torch.long, device="cuda")
        self.pos1 = self.pos[:1]
        self.mask_idx = torch.arange(max_seq, dtype=torch.long, device="cuda")
        #: mRoPE offset of the request being decoded, per row: the mRoPE position
        #: of token ``i`` is ``i + rope_delta``.  0 for a text-only request (where
        #: the three axes collapse to 1-D and this is the identity), and the
        #: constant ``MMInput.rope_delta`` after a prompt that contains an image --
        #: the image block ends at ``max(h, w) / merge`` while consuming
        #: ``h*w/merge**2`` tokens, so from then on the position runs ahead of the
        #: token index.  Keeping it a *buffer add* inside the captured graphs is
        #: what lets decode stay one replay: only the cos/sin row moves, every KV
        #: slot is still addressed by the token index.
        self.rope_delta = torch.zeros(1, dtype=torch.long, device="cuda")
        self.rope_delta1 = self.rope_delta[:1]
        self.token_in = torch.zeros(1, dtype=torch.long, device="cuda")
        self.logits_out = torch.zeros(self.embed.shape[0], dtype=D, device="cuda")
        self.logits_single = self.logits_out
        self._graph = None

        # batched speculative decoding (per-row drafter + per-row verify), all
        # allocated by ``enable_spec_batch``; off by default so the single-stream
        # spec path and the plain batched path are untouched.
        self.batch_spec = False
        self.spec_width = 0
        self.spec_buckets: tuple[int, ...] = ()
        self.verify_mode = "token"      # "once" = one forward for all B*c tokens
        self.sparse_topk = 0            # >0: selective KV (needs masked multi-q)
        self.verify_toks_b = None
        self.verify_pos_b = None
        self.verify_hidden_b = None
        self.verify_logits_b = None
        self.spec_toks_b = None
        self.spec_logits_b = None
        self.h_prev_b = None
        self._verify_batch_graphs: dict[int, torch.cuda.CUDAGraph] = {}

        # speculative-verify state (graph captured by ``capture_verify``)
        hidden = self.cfg["hidden_size"]
        self.h_prev = torch.zeros(hidden, dtype=D, device="cuda")
        self.verify_c = 4
        self.verify_off = torch.arange(self.verify_c, dtype=torch.long, device="cuda")
        self.verify_toks = torch.zeros(self.verify_c, dtype=torch.long, device="cuda")
        self.verify_pos = torch.zeros(self.verify_c, dtype=torch.long, device="cuda")
        self.verify_hidden = torch.zeros(self.verify_c, hidden, dtype=D, device="cuda")
        self.verify_logits = torch.zeros(self.verify_c, self.embed.shape[0], dtype=D, device="cuda")
        self._verify_graph = None

        # acceptance statistics of the last ``spec_generate``
        self.n_drafted = 0
        self.n_accepted = 0
        self.n_spec_steps = 0

    def enable_batching(self, capacity: int) -> None:
        """Allocate a leading row dim of ``capacity`` on every per-token buffer.

        Stage 1 runs ``B`` sequences through one forward: the projections are
        batched (one weight read for all rows, see ``models/qwen35/batch.py``)
        while the stateful mixers keep calling the single-stream kernels once per
        row on row slices.  Every buffer the captured graph writes therefore has
        to exist -- with a stable address -- before capture.

        ``capacity`` *includes* the single-stream staging row: decode rows are
        ``0 .. capacity-2`` and row ``capacity-1`` is the row ``prefill`` writes
        (``prefill_row``, also what ``pos1`` points at).  That split is what lets
        continuous batching prefill a new request between decode steps without
        clobbering a live decode row's recurrent state.  ``capacity=1`` keeps the
        single-stream buffers untouched, so the default path is unchanged.

        Must run before ``capture()``/``capture_batch()``.
        """
        capacity = int(capacity)
        if capacity < 1:
            raise ValueError(f"batch capacity must be >= 1 (got {capacity})")
        if self._graph is not None or self._verify_graph is not None:
            raise RuntimeError("enable_batching() must be called before capture()")
        self.batch_capacity = capacity
        if capacity == 1:
            self.decode_capacity = 1
            self.prefill_row = 0
            return
        # One row is the prefill/verify staging row; the rest are decode rows.
        self.decode_capacity = capacity - 1
        self.prefill_row = capacity - 1
        self.mtp_prefill = False        # speculative decoding is single-stream only
        self.pos = torch.zeros(capacity, dtype=torch.long, device="cuda")
        self.pos1 = self.pos[self.prefill_row:self.prefill_row + 1]
        self.rope_delta = torch.zeros(capacity, dtype=torch.long, device="cuda")
        self.rope_delta1 = self.rope_delta[self.prefill_row:self.prefill_row + 1]
        self.token_in = torch.zeros(capacity, dtype=torch.long, device="cuda")
        self.logits_out = torch.zeros(capacity, self.embed.shape[0], dtype=D, device="cuda")
        self.logits_single = self.logits_out[self.prefill_row]
        # Per-row attention window length (``pos+1``) for the stage-3 batched
        # attention kernel; [B] int32 because the kernel reads it on the device.
        self.ctx_lens = torch.ones(capacity, dtype=torch.int32, device="cuda")
        for layer in self.layers:
            layer.mix.enable_batching(capacity, self.prefill_row)
        self._row_tables.clear()
        if self.pool is not None:
            self._prepare_pool_rows()

    def _prepare_pool_rows(self) -> None:
        """Grow the slot map and reserve one scratch block per row.

        The captured graphs run every row of their bucket, including rows the
        caller does not fill (a bucket is padded with sentinel rows).  Reserve a
        block per row *before* capture and point that row's map at it, so the
        capture-time warmup cannot scribble over a block a real sequence owns --
        the same trick ``serve.py`` uses for its single scratch block.  The
        prefill row gets one too, because ``capture()`` warms its prefill path.
        """
        pool = self.pool
        if pool is None or self.batch_capacity <= 1:
            return
        pool.set_rows(self.batch_capacity)
        if self._scratch_blocks:
            return
        for row in range(self.batch_capacity):
            blk = pool.alloc()
            self._scratch_blocks.append(blk)
            pool.assign(blk, 0, row)

    def enable_paging(self, num_blocks: int = 512, block_size: int = 64):
        """Serve the full-attention layers from a shared paged int8 block pool.

        Must run before :meth:`capture`: the captured decode step bakes in the pool
        tensors (and the device-resident slot map), so switching afterwards would
        leave a graph writing somewhere else. Returns the pool, which the caller
        drives through a :class:`~radix_cache.RadixCache`.

        Caller contract: **assign a block table before any prefill or capture**.
        ``pos2slot`` starts as all zeros, so running ``prefill()``/``capture()``
        on a freshly paged model makes every position address pool slot 0 and
        successive prefills read each other's KV.  The symptom looks like a
        non-deterministic model but is a caller error; ``PrefixCacheEngine``
        satisfies the contract by allocating blocks and calling ``assign_table``.
        """
        if self._graph is not None:
            raise RuntimeError("enable_paging() must be called before capture()")
        full = [layer.mix for layer in self.layers if isinstance(layer.mix, FullAttn)]
        # Release the private dense caches *before* the pool is allocated. At a 256k
        # window each is ~4.5 GB, and holding both at once does not fit beside 4 GB
        # of int8 weights -- they are dead the moment the pool exists, so dropping
        # them first turns "OOM at startup" into "no extra peak at all".
        for mix in full:
            mix.kcache = mix.vcache = mix.kscale = mix.vscale = None
        torch.cuda.empty_cache()
        pool = PagedKVCache(len(full), self.cfg["num_key_value_heads"], self.cfg["head_dim"],
                            num_blocks=num_blocks, block_size=block_size,
                            max_seq=self.max_seq, rows=max(self.batch_capacity, 1))
        for i, mix in enumerate(full):
            mix.attach_pool(pool, i)
        self.pool = pool
        torch.cuda.empty_cache()
        if self.batch_capacity > 1:
            self._prepare_pool_rows()
        return pool

    def enable_vision(self, quant: bool = False, dtype=D) -> "VisionTower":
        """Attach the vision tower, built lazily so text-only serving pays nothing.

        The tower is not part of the decode graph: an image is encoded once per
        request, before its tokens are prefilled, so it can be a plain eager
        module.  Its weights live in the same resident dict as the text ones and
        are released by :meth:`close`.
        """
        from models.qwen35.vision import VisionTower

        if self.pool is None:
            raise RuntimeError("enable_vision() needs the paged model (serve path)")
        cfg = self.raw_cfg.get("vision_config")
        if not cfg:
            raise RuntimeError("this checkpoint has no vision_config")
        if self.vision is None:
            self.vision = VisionTower(self.vision_sd, cfg, device="cuda",
                                      quant=quant, dtype=dtype)
        return self.vision

    def image_embeds(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images into LLM token embeddings ``[Σtokens, hidden]``.

        ``pixel_values`` is the processor's flattened patch tensor
        ``[Σ t*h*w, C*tps*p*p]`` and ``grid_thw`` its ``(t, h, w)`` patch grids, in
        prompt order.  The result is in that same order, which is exactly the order
        of the prompt's ``<|image_pad|>`` tokens, so the caller can splice it in
        without permuting anything.
        """
        if self.vision is None:
            raise RuntimeError("vision tower not enabled (enable_vision())")
        pv = pixel_values.to(device="cuda", dtype=WF)
        gt = grid_thw.to(device="cuda", dtype=torch.long)
        with torch.no_grad():
            return self.vision(pv, gt).to(D)

    def snapshot_linear_states(self, row: int = 0) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        """Clone every linear-attention layer's recurrent and conv state for ``row``.

        Prefix sharing needs this: a cached KV prefix is only reusable together
        with the recurrent state that produced it, because those 24 layers carry
        the prefix forward as a fixed-size state rather than as KV.  ``row``
        defaults to 0; a batched caller passes the row whose sequence it wants to
        snapshot (the prefill row, or a decode row).
        """
        out: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for i, layer in enumerate(self.layers):
            mix = layer.mix
            if isinstance(mix, LinearAttn):
                out[i] = (mix.state[row].clone(), mix.conv_state[row].clone())
        return out

    def restore_linear_states(self, states, row: int = 0) -> None:
        """Adopt a snapshot taken by :meth:`snapshot_linear_states` into ``row``."""
        for i, (st, conv) in states.items():
            mix = self.layers[i].mix
            mix.state[row].copy_(st)
            mix.conv_state[row].copy_(conv)

    def move_linear_states(self, src_row: int, dst_row: int) -> None:
        """Copy row ``src_row``'s recurrent state into ``dst_row``.

        The bridge between the two halves of a batched step: a new request is
        prefilled on the staging row (``prefill_row``), then its recurrent state
        is moved into the free decode row that will carry on generating.  The
        row's :attr:`rope_delta` travels with it -- it belongs to the sequence,
        not to the row, and losing it would rotate the next tokens of an
        image prompt at the wrong angle.
        """
        if src_row == dst_row:
            return
        for layer in self.layers:
            mix = layer.mix
            if isinstance(mix, LinearAttn):
                mix.state[dst_row].copy_(mix.state[src_row])
                mix.conv_state[dst_row].copy_(mix.conv_state[src_row])
        self.rope_delta[dst_row].copy_(self.rope_delta[src_row])

    def zero_linear_states(self, row: int) -> None:
        """Clear one row's recurrent state (before a fresh sequence occupies it)."""
        for layer in self.layers:
            mix = layer.mix
            if isinstance(mix, LinearAttn):
                mix.state[row].zero_()
                mix.conv_state[row].zero_()
        self.rope_delta[row].zero_()

    def _forward(self) -> None:
        """One token on the single-stream path (the prefill/staging row).

        This is the graph ``capture()`` records for the single-stream engine and
        the eager reference the batched path is compared against; it writes the
        staging row, never a decode row.

        The rotation reads ``pos + rope_delta`` while the KV slot stays at ``pos``
        (see ``rope_delta``); with a text-only request the buffer is 0 and this is
        exactly the old one-term lookup.
        """
        pr = self.prefill_row
        x = self.embed.lookup(self.token_in[pr:pr + 1]).view(-1)
        ctx = (self.pos1, self.cos_tbl, self.sin_tbl, self.mask_idx,
               self.rope_delta1)
        for layer in self.layers:
            x = layer(x, ctx)
        x = self.norm(x)
        self.logits_single.copy_(self.lm_head(x))
        self.pos1.add_(1)

    def _forward_batch(self, rows: int) -> None:
        """One token for each of ``rows`` sequences in a single forward.

        Projections take the M=rows batched GEMV path (the row count is static in
        a captured graph); attention and the delta-net recurrence call the
        existing single-token kernels once per row, on row slices of their
        per-row buffers, so each row's numbers match a solo decode exactly.

        ``positions`` stays the raw token index (it addresses the KV slot and the
        context window); the 5th ``ctx`` entry carries the per-row ``rope_delta``
        so only the rotation is shifted (mRoPE after an image).
        """
        x = self.embed.lookup(self.token_in[:rows])           # [rows, hidden]
        positions = self.pos[:rows]
        ctx = (self.cos_tbl, self.sin_tbl, self.mask_idx, self.ctx_lens,
               self.rope_delta)
        for layer in self.layers:
            x = layer.batch(x, positions, ctx)
        x = self.norm(x)
        self.logits_out[:rows].copy_(batch_proj(self.embed.gemv, x))
        self.pos[:rows].add_(1)

    def reset(self) -> None:
        self.pos.zero_()
        self.rope_delta.zero_()
        for layer in self.layers:
            layer.reset()
        if self.mtp is not None:
            self.mtp.reset()
        if self.batch_spec:
            self.h_prev_b.zero_()
            self.mtp.b_pos.zero_()

    def logits(self, token: int) -> torch.Tensor:
        """Un-captured single step (prefill / debugging)."""
        pr = self.prefill_row
        self.token_in[pr:pr + 1].fill_(token)
        self._forward()
        return self.logits_single

    @torch.no_grad()
    def prefill(self, tokens: list[int], chunk: int = 2048,
                spec_row: int | None = None,
                mm: "MMInput | None" = None) -> torch.Tensor:
        """Process a whole prompt in batched GEMMs; returns the last token's logits.

        ``chunk`` is the GEMM's M, and it is worth a lot: rocBLAS on fp16 reaches
        ~13 TFLOP/s at M=256 and ~20 at M>=1024, and a bigger chunk also means
        fewer 4-3-token-sized elementwise ops per token.  Measured at n=2048:
        865 tok/s at 256, 1035 at 512, 1094 at 2048.  It only costs activation
        memory (attention scores are [nkv, rep, C, p1] fp32), which is why this
        is not simply ``max_seq``.

        Falls back to the one-token path for very short prompts, where the fixed
        setup cost outweighs the batching win.

        ``mm`` (an :class:`mm_input.MMInput`) is the request's image payload.  It
        changes exactly two things, both per chunk and both in *absolute* prompt
        coordinates (``tokens`` is the whole prompt, so chunk ``[c0, c1)`` is
        prompt ``[pos0+c0, pos0+c1)``):

        * the gather buffer's image rows are overwritten with the tower's
          embeddings (``mm.spans`` overlap, so a chunk that starts inside an image
          copies only its own rows -- no chunk-boundary assumption);
        * the full-attention layers rotate with the explicit 3-D cos/sin of those
          tokens (``ops.mrope.mrope_cos_sin`` over ``mm.slice_positions``) instead
          of the 1-D table.

        What it deliberately does *not* change: the causal mask, the KV slot and
        the context window keep using the token index, which is why decode
        afterwards only needs the scalar ``rope_delta``.  ``mm=None`` is the old
        path, bit for bit.
        """
        n = len(tokens)
        pos0 = int(self.pos1)
        if n == 0:
            raise ValueError("empty prompt")
        if n > self.max_seq - pos0:
            raise ValueError(f"prompt of {n} exceeds max_seq={self.max_seq}")
        if n < 4:
            # tiny prompts are memory-bound: hipBLAS at M<4 is slower than the
            # dedicated GEMV kernel, so just walk them through the decode path.
            # ``rope_delta`` is a decode-graph input, so it has to be in place
            # before the first ``logits`` call.
            if mm is not None:
                self.rope_delta[self.prefill_row] = int(mm.rope_delta)
            lg = None
            for t in tokens:
                lg = self.logits(t)
            return lg

        toks = torch.tensor(tokens, dtype=torch.long, device="cuda")
        ctx = (self.cos_tbl, self.sin_tbl, self.mask_idx)
        Q = PREFILL_ATTN_TILE
        last = None
        for c0 in range(0, n, chunk):
            c1 = min(c0 + chunk, n)
            x = self.embed.lookup(toks[c0:c1])
            positions = torch.arange(pos0 + c0, pos0 + c1, device="cuda")
            # Query tiles for the causal attention: each tile only reads keys up to
            # its own last position, so the masked-out half of the score block is
            # never built.  Host-side ints, so no device sync per layer.  The key
            # range and the query range are both *absolute* positions, which is
            # what the cache slice and mask_idx are indexed with.
            n_loc, base = c1 - c0, pos0 + c0
            tiles = ([(q0, min(q0 + Q, n_loc), base, base + min(q0 + Q, n_loc))
                      for q0 in range(0, n_loc, Q)]
                     if n_loc > Q else None)
            layer_ctx = ctx
            rope = None
            if mm is not None:
                x = scatter_image_embeds(x, mm.embeds, mm.spans, pos0 + c0, pos0 + c1)
                a, b = pos0 + c0, pos0 + c1
                rope = mrope_cos_sin(mm.slice_positions(a, b), self.inv_freq)
                layer_ctx = ctx + (rope,)
            for layer in self.layers:
                x = layer.prefill(x, positions, layer_ctx, tiles)
            if self.mtp is not None and self.mtp_prefill:
                # Feed the MTP head chunk by chunk instead of holding every chunk's
                # hidden state to concatenate at the end: `H` for a 256k prompt is
                # 1.34 GB bf16 and the cat copy doubles it.  The last prompt token
                # has no next token, so each chunk stops one short of its end.
                # Its own attention is a full-attention layer over the same
                # positions, so an image chunk hands it its own rope slice (it
                # covers ``m1 - c0`` tokens, a strict prefix of the chunk's).
                m1 = min(c1, n - 1)
                if m1 > c0:
                    self.mtp.prefill(x[:m1 - c0], toks[c0 + 1:m1 + 1],
                                     torch.arange(pos0 + c0, pos0 + m1, device="cuda"),
                                     layer_ctx,
                                     rope=None if rope is None
                                     else (rope[0][:m1 - c0], rope[1][:m1 - c0]))
            elif self.batch_spec and self.mtp is not None and spec_row is not None:
                # Batched spec: same chunk, into that row's paged drafter KV.
                m1 = min(c1, n - 1)
                if m1 > c0:
                    self.mtp.prefill_row(
                        spec_row, x[:m1 - c0], toks[c0 + 1:m1 + 1],
                        torch.arange(pos0 + c0, pos0 + m1, device="cuda"), layer_ctx,
                        rope=None if rope is None
                        else (rope[0][:m1 - c0], rope[1][:m1 - c0]))
            last = x[-1]
        self.pos1.fill_(pos0 + n)
        # From here on every decode step of this request rotates at
        # ``token_index + rope_delta``; a text-only prompt (or ``mm=None``) writes
        # 0, which is the identity the captured graph was built with.
        self.rope_delta[self.prefill_row] = int(mm.rope_delta) if mm is not None else 0

        if self.mtp is not None and self.mtp_prefill and n >= 2:
            self.h_prev.copy_(last)
        elif self.batch_spec and spec_row is not None:
            self.h_prev_b[spec_row].copy_(last)

        return self.lm_head(self.norm(last))

    @torch.no_grad()
    def prefill_group(self, seqs: list, snapshot_interval: int | None = None):
        """Prefill several prompts in one forward.

        Returns ``(logits_list, states_at_list)``: one next-token logits tensor per
        sequence and, per sequence, the block-boundary recurrent snapshots taken
        while prefilling (``{absolute_position: snapshot_linear_states(row)}``) for
        the radix tree to insert.  ``states_at_list[i]`` is empty when
        ``snapshot_interval`` is ``None``.

        ``seqs`` are duck-typed: each needs ``token_ids`` (the whole prompt),
        ``matched`` (how many leading prompt tokens the radix tree already holds;
        default 0), ``row`` (the decode row it keeps for its lifetime) and
        ``block_table`` (its KV blocks, already allocated -- matched blocks shared
        with the tree, suffix blocks fresh).  The caller is responsible for having
        restored the matched boundary's recurrent state into ``row`` already; this
        method only clears the row when ``matched == 0``.  Nothing is staged on the
        prefill row: KV and recurrent state land directly in each sequence's decode
        row, so no state has to be moved afterwards.

        Only the projection/MLP stages are shared: at every layer all sequences'
        hidden states are concatenated, the projection runs once over ``C_total``
        tokens (a bigger GEMM, and one weight dequant per layer instead of one per
        sequence), the result is sliced per sequence for the stateful attention /
        delta scan on that sequence's own positions and rows, and the output
        projections plus the MLP run batched again.  Prefill is eager (no captured
        graph), so the dynamic ``C_total`` costs nothing.

        With ``snapshot_interval`` set, the suffix of each sequence is walked in
        block-aligned ``snapshot_interval`` chunks (token-major: every chunk goes
        through *all* layers before the next chunk starts) so the recurrent state
        can be snapshotted at block boundaries, exactly like the single-sequence
        prefix path.  Without it the whole suffix is one chunk, which also keeps
        ``k == 1`` bit-identical to ``prefill``.

        This buys the latency of a *batch* of prompts rather than their sum: the
        weights are read once per layer.  It is not bit-identical to running the
        prompts one by one -- a different GEMM M changes the fp32 accumulation
        order -- so the contract is "same argmax, logits within a few bf16 ULPs"
        (the single-sequence ``prefill`` path is unchanged and stays exact).
        """
        if self.pool is None:
            raise RuntimeError("prefill_group needs a paged pool (enable_paging())")
        n = len(seqs)
        if n == 0:
            return [], []
        lens = [len(s.token_ids) for s in seqs]
        if min(lens) < 1:
            raise ValueError("empty prompt")
        if max(lens) > self.max_seq:
            raise ValueError(f"prompt of {max(lens)} exceeds max_seq={self.max_seq}")
        matched = [int(getattr(s, "matched", 0) or 0) for s in seqs]
        for length, m in zip(lens, matched):
            if not 0 <= m < length:
                raise ValueError(f"matched={m} outside 0..{length - 1}")
        rows = [int(s.row) for s in seqs]
        if len(set(rows)) != n:
            raise ValueError(f"duplicate decode rows: {rows}")
        if min(rows) < 0 or max(rows) >= self.decode_capacity:
            raise ValueError(f"row outside 0..{self.decode_capacity - 1}: {rows}")

        ctx = (self.cos_tbl, self.sin_tbl, self.mask_idx)
        bs = self.pool.bs
        suffix = [length - m for length, m in zip(lens, matched)]
        #: Per-sequence image payload, read off the sequence so a multimodal
        #: request needs no extra argument here (the runner already carries it on
        #: ``Sequence.mm``).  All-``None`` -- the text-only engine -- takes exactly
        #: the old code path: no rope pair, no embed overwrite, delta 0.
        mm_list = [getattr(s, "mm", None) for s in seqs]
        for s, row, m, mm_i in zip(seqs, rows, matched, mm_list):
            self.pool.assign_table(list(s.block_table), 0, row)
            if m == 0:
                self.zero_linear_states(row)      # fresh sequence
                # matched > 0: the caller restored the boundary's state into this
                # row, and the rope offset with it -- rewriting it here would be
                # the same value, but a re-used row of a *text* request must go
                # back to 0, so it is written unconditionally.
            self.rope_delta[row] = int(mm_i.rope_delta) if mm_i is not None else 0

        # Token-major schedule.  With prefix reuse the suffix is split at
        # block-aligned snapshot boundaries: the block-aligned head (the part the
        # tree can reuse) is walked in ``snapshot_interval`` chunks with one
        # snapshot each; the sub-block tail goes last and only its logits matter.
        states_at: list[dict[int, dict]] = [{} for _ in range(n)]
        plans: list[list[tuple[int, int, int]]] = []
        # Last token index of the snapshottable head of each sequence (a whole
        # number of blocks above ``matched``); the rest is the tail.
        head = [max(0, (length // bs) * bs - m) for length, m in zip(lens, matched)]
        if snapshot_interval is None:
            plans.append([(i, 0, suffix[i]) for i in range(n)])
        else:
            step = max(bs, int(snapshot_interval) // bs * bs)
            n_head = max((h + step - 1) // step for h in head)
            for j in range(n_head):
                rng = []
                for i in range(n):
                    c0 = min(j * step, head[i])
                    c1 = min(c0 + step, head[i])
                    if c1 > c0:
                        rng.append((i, c0, c1))
                if rng:
                    plans.append(rng)
            tail = [(i, head[i], suffix[i]) for i in range(n) if head[i] < suffix[i]]
            if tail:
                plans.append(tail)

        last_hidden: list[torch.Tensor | None] = [None] * n
        Q = PREFILL_ATTN_TILE
        for chunk in plans:
            parts: list[torch.Tensor] = []
            positions: list[torch.Tensor] = []
            ropes: list[tuple | None] = []
            tiles: list[list | None] = []
            starts = [0]
            for i, c0, c1 in chunk:
                length = c1 - c0
                base = matched[i] + c0
                ids = list(seqs[i].token_ids[base:base + length])
                emb = self.embed.lookup(torch.as_tensor(
                    ids, dtype=torch.long, device="cuda"))
                mm_i = mm_list[i]
                if mm_i is not None:
                    emb = scatter_image_embeds(emb, mm_i.embeds, mm_i.spans,
                                               base, base + length)
                    ropes.append(mrope_cos_sin(mm_i.slice_positions(base, base + length),
                                               self.inv_freq))
                else:
                    ropes.append(None)
                parts.append(emb)
                positions.append(torch.arange(base, base + length,
                                              dtype=torch.long, device="cuda"))
                # Absolute key range: a tile only reads keys up to its own end.
                tiles.append(
                    [(q0, min(q0 + Q, length), base, base + min(q0 + Q, length))
                     for q0 in range(0, length, Q)]
                    if length > Q else None)
                starts.append(starts[-1] + length)
            x = torch.cat(parts, dim=0)

            for layer in self.layers:
                mix, proj = layer.prefill_group_project(x)
                pieces = []
                for k, (i, _c0, _c1) in enumerate(chunk):
                    sl = slice(starts[k], starts[k + 1])
                    proj_k = tuple(t[sl] for t in proj)
                    # The rope pair rides on ``ctx`` (the same pass-through style
                    # ``FullAttn.batch`` uses for ``ctx_lens``): the layer helpers
                    # forward ``ctx`` untouched, so a sequence with an image gets
                    # its own 3-D rotation and a text sequence gets the 1-D table.
                    ctx_k = ctx if ropes[k] is None else ctx + (ropes[k],)
                    pieces.append(layer.prefill_group_state(
                        mix, proj_k, positions[k], ctx_k, tiles[k], rows[i]))
                out = layer.prefill_group_out(mix, torch.cat(pieces, dim=0))
                x = x + out
                x = x + layer.mlp.prefill(layer.post_ln(x))

            for k, (i, c0, c1) in enumerate(chunk):
                if c1 == suffix[i]:
                    last_hidden[i] = x[starts[k + 1] - 1].clone()
                # Batched spec: feed this chunk's main hidden states to that row's
                # drafter KV (H at position p, next token p+1) -- the same pair the
                # single-sequence prefill hands to ``MTP.prefill``.
                if self.batch_spec and self.mtp is not None:
                    base = matched[i] + c0
                    length = c1 - c0
                    if length > 1:
                        h = x[starts[k]:starts[k + 1] - 1]
                        nxt = torch.as_tensor(
                            list(seqs[i].token_ids[base + 1:base + length]),
                            dtype=torch.long, device="cuda")
                        pos = torch.arange(base, base + length - 1,
                                           dtype=torch.long, device="cuda")
                        self.mtp.prefill_row(rows[i], h, nxt, pos, ctx)
                    if c1 == suffix[i]:
                        self.h_prev_b[rows[i]].copy_(x[starts[k + 1] - 1])
                # A reusable snapshot only exists on a block boundary inside the
                # block-aligned head (same rule as the single-sequence path).
                if snapshot_interval is not None and c1 % bs == 0 and c1 <= head[i]:
                    states_at[i][matched[i] + c1] = \
                        self.snapshot_linear_states(row=rows[i])

        # One lm_head read for every sequence: [n, hidden] -> [n, vocab].  The
        # batched GEMV path handles n in 2/4/8; other counts use the single GEMV
        # per row (the usual bucket sizes are what matter here).
        last = torch.stack([self.norm(h) for h in last_hidden])
        if n in (2, 4, 8):
            logits = self.embed.prefill(last)
            out = [logits[i] for i in range(n)]
        else:
            out = [self.lm_head(last[i]) for i in range(n)]
        return out, states_at

    def capture(self, warmup: int = 3) -> None:
        """Capture ``_forward`` into a CUDA graph for one-launch decode steps."""
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                self._forward()
        torch.cuda.current_stream().wait_stream(s)
        self.reset()
        self.token_in.zero_()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._forward()
        self._graph = g
        # Warm the batched-prefill path as well: its first call otherwise pays for
        # rocBLAS handle creation, the triangular solver and pool growth.
        self.prefill([1, 2, 3, 4, 5, 6, 7, 8] * 4)
        self.reset()
        if self.mtp is not None:
            self.capture_verify()
            self.mtp.capture_draft((self.cos_tbl, self.sin_tbl, self.mask_idx))
            self.reset()

    @torch.inference_mode()
    def set_rows(self, tokens, positions, slot_rows=None, row_indices=None) -> None:
        """Stage one decode step of the active sequences.

        ``tokens``/``positions`` are the per-row input token and absolute
        position; by default they fill rows ``0..n-1``.  A continuous-batching
        caller whose live sequences sit on non-contiguous rows passes
        ``row_indices`` with the same length, so a finished sequence's row can be
        left as a hole (``decode_batch(pad_rows=...)`` fills holes with the
        sentinel).  ``slot_rows`` optionally gives each row's block table, which
        is copied into that row's ``pos2slot`` map (unchanged tables are skipped).
        Everything here is an eager host/device write outside the captured graph.
        """
        if self.batch_capacity <= 1:
            raise RuntimeError("set_rows() needs enable_batching(capacity>1)")
        tokens = list(tokens)
        positions = list(positions)
        n = len(tokens)
        if len(positions) != n:
            raise ValueError(f"{n} tokens vs {len(positions)} positions")
        if row_indices is None:
            rows = list(range(n))
        else:
            rows = [int(r) for r in row_indices]
            if len(rows) != n:
                raise ValueError(f"{n} rows vs {len(rows)} row_indices")
        if any(r < 0 or r >= self.decode_capacity for r in rows):
            raise ValueError(f"row out of 0..{self.decode_capacity - 1}: {rows}")
        if rows == list(range(n)):
            self.token_in[:n].copy_(torch.as_tensor(tokens, dtype=torch.long, device="cuda"))
            pos_t = torch.as_tensor(positions, dtype=torch.long, device="cuda")
            self.pos[:n].copy_(pos_t)
            # Absolute position P means positions 0..P exist, i.e. P+1 keys: the
            # batched attention kernel loops only over this row's own window.
            self.ctx_lens[:n].copy_(pos_t.to(torch.int32).add_(1))
        else:
            for r, tok, pos in zip(rows, tokens, positions):
                self.token_in[r] = tok
                self.pos[r] = pos
                self.ctx_lens[r] = pos + 1
        if self.pool is not None and slot_rows is not None:
            for row, blocks in zip(rows, slot_rows):
                blocks = list(blocks)
                key = tuple(blocks)
                if self._row_tables.get(row) == key:
                    continue
                self.pool.assign_table(blocks, 0, row)
                self._row_tables[row] = key

    def capture_batch(self, buckets: tuple[int, ...] = (1, 2, 4, 8),
                      warmup: int = 3) -> None:
        """Capture one decode graph per bucket, all sharing one memory pool.

        Each graph is a fixed-``rows`` forward, so the projection shapes and every
        kernel launch inside it are static -- which is what makes the batched
        weight reads possible in the first place.  The graphs share a single
        CUDA graph memory pool: they are mutually exclusive (one replay at a
        time), so the pool only has to be as large as the largest bucket rather
        than the sum over buckets.

        Bucket 1 is captured through the *batched* path as well (a decode row),
        not through ``_forward``; ``_forward`` belongs to the single-stream
        engine (``capture()``/``decode_step``) and writes the staging row.
        """
        if self.batch_capacity <= 1:
            raise RuntimeError("capture_batch() needs enable_batching(capacity>1)")
        if self.pool is None:
            raise RuntimeError("capture_batch() needs a paged pool (enable_paging())")
        for b in buckets:
            if b not in (1, 2, 4, 8):
                raise ValueError(f"bucket {b} not in (1, 2, 4, 8)")
        order = sorted(set(int(b) for b in buckets) | {1})
        pool_handle = torch.cuda.graph_pool_handle()
        self._batch_graphs = {}
        if self.ctx_lens is not None:
            self.ctx_lens.fill_(1)       # capture-time windows are all pos 0
        self.reset()
        for b in order:
            if b > self.decode_capacity:
                continue
            impl = lambda b=b: self._forward_batch(b)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(warmup):
                    impl()
            torch.cuda.current_stream().wait_stream(s)
            self.reset()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool_handle):
                impl()
            self._batch_graphs[b] = g
        self.reset()

    @torch.inference_mode()
    def decode_batch(self, rows: int, bucket: int, pad_rows=()) -> torch.Tensor:
        """Replay ``bucket``'s graph; returns the logits of all ``bucket`` rows.

        The runner keeps each sequence on a fixed decode row, so the live rows of
        a bucket need not be contiguous.  ``rows`` is the high-water mark (active
        rows are ``< rows``); ``pad_rows`` lists the holes inside it, and rows
        ``rows..bucket-1`` are always holes.  Every hole is filled with the
        sentinel sequence (its own scratch block + slot-map row, token 0, position
        0, context length 1); its logits are computed and discarded.  The caller
        samples by ``seq.row`` from the returned ``[bucket, vocab]`` view.
        """
        graph = self._batch_graphs.get(bucket)
        if graph is None:
            raise ValueError(f"no graph captured for bucket {bucket}")
        if not 1 <= rows <= bucket:
            raise ValueError(f"rows={rows} outside 1..{bucket}")
        pad = {int(r) for r in pad_rows}
        if any(r < 0 or r >= bucket for r in pad):
            raise ValueError(f"pad_rows outside 0..{bucket - 1}: {sorted(pad)}")
        pad.update(range(rows, bucket))
        if pad:
            for row in sorted(pad):
                if self.pool is not None:
                    self.pool.assign(self._scratch_blocks[row], 0, row)
                self.token_in[row] = self._sentinel_token
                self.pos[row] = 0
                # A hole must rotate at a valid table row: row 0 + a stale negative
                # delta from the previous occupant would index before the table.
                self.rope_delta[row] = 0
                if self.ctx_lens is not None:
                    self.ctx_lens[row] = 1
        graph.replay()
        return self.logits_out[:bucket]

    def enable_spec_batch(self, K: int, buckets: tuple[int, ...] = (1, 2, 4, 8),
                          block_size: int | None = None,
                          max_drafter_len: int = 16384,
                          sparse_topk: int = 0,
                          verify_mode: str | None = None) -> None:
        """Allocate the per-row drafter/verify state and capture their graphs.

        ``K`` drafts per step means verifying ``c = K + 1`` tokens per row.  The
        drafter gets its own small paged KV pool (one identity block table per
        decode row) and a bucket graph that advances every row once per replay.

        Two verify implementations share the same buffers and the same per-step
        interface (:meth:`spec_step_batch`):

        * ``"once"`` -- one forward for all ``B*c`` tokens: wide-M projections and
          the multi-query paged attention kernel (task-12); the fast path;
        * ``"token"`` -- ``c`` sequential batched-decode forwards; slower, but it
          only needs the stage-3 kernels and stays as the fallback.

        ``verify_mode=None`` picks ``"once"`` only when the wide-M GEMV and the
        multi-query attention kernel are both present.  ``sparse_topk > 0`` is the
        opt-in selective-KV verify; it needs the masked multi-query kernel and is
        not wired yet.

        The ``c`` rollback snapshots per row are the expensive part (24 layers x
        2 MB fp32 per snapshot); ``enable_verify_batch`` sizes them for exactly
        this width.  Contexts beyond the drafter pool length raise from
        :meth:`MTP.check_len`, so the caller can fall back to plain decode.
        """
        if self.batch_capacity <= 1:
            raise RuntimeError("enable_spec_batch() needs enable_batching(capacity>1)")
        if self.pool is None:
            raise RuntimeError("enable_spec_batch() needs a paged pool")
        if self.mtp is None:
            raise RuntimeError("enable_spec_batch() needs the MTP head in the checkpoint")
        if int(sparse_topk) > 0:
            raise NotImplementedError(
                "sparse Top-K verify needs the masked attn_paged_multi_q kernel "
                "(task-12); pass sparse_topk=0 for now")
        if verify_mode is None:
            # Default stays the c-token verify.  The one-forward verify is opt-in
            # (``Q35_SPEC_VERIFY=once``): at B=8 even 4k context measured below
            # greedy (task-13 conclusion), so it is kept as the reference path
            # until a wide-M kernel that beats the M=8 one per element exists.
            verify_mode = os.environ.get("Q35_SPEC_VERIFY", "token")
            if verify_mode not in ("once", "token"):
                verify_mode = "token"
            if verify_mode == "once" and not (wide_kernel_ready() and multi_q_ready()
                                              and MULTI_Q_WIRED):
                verify_mode = "token"
        if verify_mode not in ("once", "token"):
            raise ValueError(f"verify_mode must be 'once' or 'token' (got {verify_mode!r})")
        self.verify_mode = verify_mode
        self.sparse_topk = int(sparse_topk)
        # Drop any graphs captured for a previous configuration before allocating
        # the (possibly differently sized) buffers for this one.
        self._verify_batch_graphs = {}
        width = int(K) + 1
        if width < 2 or width > CKPT:
            raise ValueError(f"K must be in 1..{CKPT - 1} (got {K})")
        cap = self.decode_capacity
        buckets = tuple(b for b in buckets if b <= cap)
        if not buckets:
            raise ValueError("no bucket fits the decode capacity")
        self.batch_spec = True
        self.spec_width = width
        self.spec_buckets = buckets
        hidden = self.cfg["hidden_size"]
        vocab = self.embed.shape[0]
        self.verify_toks_b = torch.zeros(cap, width, dtype=torch.long, device="cuda")
        self.verify_pos_b = torch.zeros(cap, width, dtype=torch.long, device="cuda")
        self.verify_hidden_b = torch.zeros(cap, width, hidden, dtype=D, device="cuda")
        self.verify_logits_b = torch.zeros(cap, width, vocab, dtype=D, device="cuda")
        self.spec_toks_b = torch.zeros(cap, width, dtype=torch.long, device="cuda")
        self.spec_logits_b = torch.zeros(cap, vocab, dtype=D, device="cuda")
        self.h_prev_b = torch.zeros(cap, hidden, dtype=D, device="cuda")
        bs = int(block_size or self.pool.bs)
        self.mtp.enable_spec_batch(
            self.batch_capacity, buckets,
            ctx=(self.cos_tbl, self.sin_tbl, self.mask_idx),
            block_size=bs, max_len=max_drafter_len)
        for layer in self.layers:
            mix = layer.mix
            if isinstance(mix, LinearAttn):
                mix.enable_verify_batch(cap, width)
            elif isinstance(mix, FullAttn) and hasattr(mix, "enable_multi_q"):
                mix.enable_multi_q(cap, width)
        self.capture_verify_batch()

    def _verify_batch_once(self, rows: int, prof: dict | None = None) -> None:
        """One forward for every row's ``c`` tokens (task-13 fast path).

        Layout: ``x`` is row-major ``[rows*c, hidden]`` (a row's c verified tokens
        are contiguous).  Per layer, the projection and the MLP run once with the
        wide-M kernels over all rows*c tokens; the two stateful halves keep the
        greedy-decode kernels:

        * full attention -- all c K/V rows are written first, then one
          multi-query paged-attention call serves all c queries from that row's
          history (kernels task-12).  Writing all of them before the call is safe:
          the kernel masks each query at its own position;
        * linear attention -- the conv/delta recurrence is sequential over the c
          tokens (state dependency), one batch launch per token, and the state is
          snapshotted right after token j for the rollback.

        When the multi-query kernel is missing this falls back to the per-token
        attention entry (correct, just not the point of this path).
        """
        c = self.spec_width
        total = rows * c
        hidden = self.cfg["hidden_size"]
        # The 5th entry is the per-row mRoPE offset: every verified token rotates
        # at ``pos + rope_delta`` while its KV slot stays at ``pos``.  Constant
        # within a request, so it is a buffer the captured graph reads.
        ctx = (self.cos_tbl, self.sin_tbl, self.mask_idx, self.ctx_lens,
               self.rope_delta)
        poss = self.verify_pos_b[:rows]
        x = self.embed.lookup(self.verify_toks_b[:rows].reshape(total))
        multi_q = multi_q_ready()
        prof = prof if prof is not None else None

        def lap(tag, start):
            if prof is not None:
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                prof.setdefault(tag, []).append((start, end))
            return start

        for layer in self.layers:
            mix = layer.mix
            s = torch.cuda.Event(enable_timing=True) if prof is not None else None
            if s is not None:
                s.record()
            h = layer.in_ln(x)
            if isinstance(mix, LinearAttn):
                qkv, z, b, a = wide_proj(mix.qkvzb_a, h)
                if s is not None:
                    lap("proj", s)
                    s = torch.cuda.Event(enable_timing=True)
                    s.record()
                qkv = qkv.view(rows, c, -1)
                z = z.view(rows, c, -1)
                b = b.view(rows, c, -1)
                a = a.view(rows, c, -1)
                pieces = [mix.verify_batch_step(qkv[:, j], z[:, j], b[:, j], a[:, j], j)
                          for j in range(c)]
                out = wide_proj(mix.out_proj,
                                torch.stack(pieces, dim=1).reshape(total, -1))
                if s is not None:
                    lap("recur", s)
                    s = torch.cuda.Event(enable_timing=True)
                    s.record()
            else:
                qg, k, v = wide_proj(mix.qkv, h)
                if s is not None:
                    lap("proj", s)
                    s = torch.cuda.Event(enable_timing=True)
                    s.record()
                if multi_q:
                    # multi-query kernel returns the pre-out_proj attention output
                    pre = mix.verify_batch_multi(qg, k, v, poss, ctx, rows)
                    out = wide_proj(mix.o_proj, pre)
                else:
                    # fallback: the per-token entry already includes o_proj
                    qg = qg.view(rows, c, -1)
                    k = k.view(rows, c, -1)
                    v = v.view(rows, c, -1)
                    pieces = [mix.verify_batch_step(qg[:, j], k[:, j], v[:, j],
                                                    poss[:, j], ctx)
                              for j in range(c)]
                    out = torch.stack(pieces, dim=1).reshape(total, -1)
                if s is not None:
                    lap("attn", s)
                    s = torch.cuda.Event(enable_timing=True)
                    s.record()
            x = x + out
            x = x + layer.mlp.wide(layer.post_ln(x))
            if s is not None:
                lap("mlp", s)
        s = torch.cuda.Event(enable_timing=True) if prof is not None else None
        if s is not None:
            s.record()
        x = self.norm(x)
        logits = wide_proj(self.embed.gemv, x)
        self.verify_hidden_b[:rows].copy_(x.view(rows, c, hidden))
        self.verify_logits_b[:rows].copy_(
            logits.view(rows, c, self.embed.shape[0]))
        if s is not None:
            lap("lmhead", s)

    def _verify_batch_impl(self, rows: int) -> None:
        """``c`` sequential batched-decode steps, snapshotting state after each.

        Each step is the batched greedy decode forward, so a row's verify logits
        are exactly the logits the plain greedy step would produce at that token;
        the per-token recurrent snapshots are the rollback points.

        (A single multi-token forward -- all ``B*c`` tokens in one projection pass
        -- was implemented and measured, but at ``M = B*c = 16`` the projection
        falls back to the prefill path with its full int8->fp16 dequant (~250-290
        ms fixed cost on this part), 6.9x *slower* than these ``c`` batched-decode
        steps.  A wide-M batched GEMV (one weight pass for M=16/32) is what would
        make the single-pass verify worthwhile; until that kernel exists this is
        the faster form.)
        """
        ctx_base = (self.cos_tbl, self.sin_tbl, self.mask_idx)
        c = self.spec_width
        ctx = ctx_base + (self.ctx_lens, self.rope_delta)
        for j in range(c):
            toks = self.verify_toks_b[:rows, j]
            poss = self.verify_pos_b[:rows, j]
            self.ctx_lens[:rows].copy_(poss.to(torch.int32).add_(1))
            x = self.embed.lookup(toks)
            for layer in self.layers:
                x = layer.batch(x, poss, ctx)
            x = self.norm(x)
            self.verify_hidden_b[:rows, j].copy_(x)
            self.verify_logits_b[:rows, j].copy_(batch_proj(self.embed.gemv, x))
            for layer in self.layers:
                mix = layer.mix
                if isinstance(mix, LinearAttn):
                    mix.snapshot_batch(rows, j)

    def set_verify_mode(self, mode: str) -> None:
        """Switch the captured verify implementation (``"once"`` / ``"token"``).

        Both use the same buffers and the same ``spec_step_batch`` interface, so
        the graphs can simply be re-captured; used for A/B measurement and by the
        engine when it wants the cheap fallback.
        """
        if mode not in ("once", "token"):
            raise ValueError(f"verify_mode must be 'once' or 'token' (got {mode!r})")
        if mode == self.verify_mode:
            return
        self.verify_mode = mode
        self.capture_verify_batch()

    def capture_verify_batch(self, warmup: int = 2) -> None:
        """One verify graph per bucket (all share a graph memory pool).

        The graph records whichever implementation ``verify_mode`` selected; the
        per-step interface and the rollback buffers are identical for both.
        """
        if not self.batch_spec:
            raise RuntimeError("enable_spec_batch() first")
        handle = torch.cuda.graph_pool_handle()
        self._verify_batch_graphs = {}
        body = (self._verify_batch_once if self.verify_mode == "once"
                else self._verify_batch_impl)
        for b in self.spec_buckets:
            impl = lambda b=b: body(b)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(warmup):
                    impl()
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=handle):
                impl()
            self._verify_batch_graphs[b] = g

    @torch.inference_mode()
    def spec_step_batch(self, tokens: list[int], positions: list[int],
                        slot_rows: list[list[int]], bucket: int, K: int,
                        row_indices=None,
                        pad_rows=()) -> tuple[list[list[int]], list[torch.Tensor], list[int]]:
        """One batched speculative step over ``tokens``/``positions``.

        ``tokens[i]`` is the token already sampled for sequence ``i`` (the one the
        drafter starts from) and ``positions[i]`` its absolute position;
        ``slot_rows`` are the main model's block tables for those rows (the same
        arguments :meth:`set_rows` takes).  ``row_indices`` places entry ``i`` on
        its fixed decode row (default ``i``) and ``pad_rows`` marks holes inside
        ``0..bucket-1``; ``bucket`` selects the captured graphs and every row of
        it runs, the unused rows as sentinels.

        Returns ``(accepted_tokens, logits_next, accepted_counts)``: per input
        sequence the accepted tokens *including* ``tokens[i]``, the logits that
        follow the accepted prefix (sample the next token from those), and how
        many draft tokens were accepted.  The caller advances the sequence by
        ``len(accepted_tokens[i])`` and stages the next step.
        """
        if not self.batch_spec:
            raise RuntimeError("enable_spec_batch() first")
        n = len(tokens)
        active_rows = (list(range(n)) if row_indices is None
                       else [int(r) for r in row_indices])
        if len(active_rows) != n or len(set(active_rows)) != n:
            raise ValueError(f"{n} tokens vs row_indices {active_rows}")
        if n < 1 or max(active_rows) >= bucket or bucket not in self._verify_batch_graphs:
            raise ValueError(f"bad rows/bucket ({active_rows}, {bucket})")
        self.mtp.check_len(max(positions) + K + 1)
        width = K + 1
        if width > self.spec_width:
            raise ValueError(f"K={K} needs width {width} > {self.spec_width}")

        self.set_rows([tokens[i] for i in range(n)],
                      [positions[i] for i in range(n)],
                      [slot_rows[i] for i in range(n)],
                      row_indices=active_rows)
        # Sentinels for every row the bucket graph executes but we do not use.
        pad = set(int(r) for r in pad_rows)
        pad |= {r for r in range(bucket) if r not in active_rows}
        for row in sorted(pad):
            self.pool.assign(self._scratch_blocks[row], 0, row)
            self.token_in[row] = self._sentinel_token
            self.pos[row] = 0
            self.ctx_lens[row] = 1
            self.rope_delta[row] = 0          # forward-only valid table row

        # Stage the drafter: h at the previous accepted token, its token, and the
        # drafter position (main position - 1: the MTP predicts the token after h).
        # Each row's mRoPE offset is copied across so the drafter rotates exactly
        # like the main model; a padded row gets 0 (it runs on the sentinel).
        mtp = self.mtp
        mtp.b_rope_delta.zero_()
        for i, row in enumerate(active_rows):
            mtp.b_tok[row] = int(tokens[i])
            mtp.b_h[row].copy_(self.h_prev_b[row])
            mtp.b_pos[row] = int(positions[i]) - 1
            mtp.b_rope_delta[row] = self.rope_delta[row]
        for row in sorted(pad):
            mtp.b_tok[row] = 0
            mtp.b_h[row].zero_()
            mtp.b_pos[row] = 0

        mtp.draft_batch(bucket, K, self.spec_toks_b[:bucket])
        self.verify_toks_b[:bucket, :width].copy_(self.spec_toks_b[:bucket, :width])
        for i, row in enumerate(active_rows):
            self.verify_pos_b[row, :width].copy_(
                torch.arange(int(positions[i]), int(positions[i]) + width,
                             device="cuda"))
        for row in sorted(pad):
            self.verify_pos_b[row, :width].copy_(
                torch.arange(width, device="cuda"))
        self._verify_batch_graphs[bucket].replay()

        # Acceptance per live row: logits[j].argmax() must equal the next draft.
        pred = self.verify_logits_b[:bucket, :K].argmax(-1)      # [bucket, K]
        match = (pred == self.verify_toks_b[:bucket, 1:width]).tolist()
        accepted_tokens: list[list[int]] = []
        accepted_counts: list[int] = []
        logits_next: list[torch.Tensor] = []
        for row in active_rows:
            j = 0
            while j < K and match[row][j]:
                j += 1
            accepted_counts.append(j)
            accepted_tokens.append(self.verify_toks_b[row, :j + 1].tolist())
            logits_next.append(self.verify_logits_b[row, j].clone())
            for layer in self.layers:
                mix = layer.mix
                if isinstance(mix, LinearAttn):
                    mix.restore_batch(row, j)
            self.h_prev_b[row].copy_(self.verify_hidden_b[row, j])
        return accepted_tokens, logits_next, accepted_counts

    def _verify_impl(self) -> None:
        x = self.embed.lookup(self.verify_toks)
        # 4th entry: this request's mRoPE offset (a buffer, so the graph replays
        # unchanged; ``verify_pos`` keeps addressing the KV slots).
        ctx = (self.cos_tbl, self.sin_tbl, self.mask_idx, self.rope_delta1)
        for layer in self.layers:
            x = layer.verify(x, self.verify_pos, ctx)
        self.verify_hidden.copy_(x)
        self.verify_logits.copy_(self.lm_head.prefill(self.norm(x)))

    def capture_verify(self, c: int = 4, warmup: int = 3) -> None:
        """Capture the c-token verification forward into its own graph.

        Verifying k drafts has to cost about one weight pass. Eagerly it costs
        ~k decode steps, mostly CPU launch overhead from the per-token recurrent
        update, so the whole forward is captured once and replayed per step.
        ``c`` is also the draft width, so ``K = c - 1`` drafts are verified per
        replay. The batched GEMV accepts only M in {2, 4, 8}, and the linear-attn
        layers keep ``CKPT`` state snapshots for the rollback, so ``c`` must be one
        of those widths.
        """
        if c not in (2, 4, 8) or c > CKPT:
            raise ValueError(f"verify width must be 2, 4 or 8 (got {c})")
        self.verify_c = c
        hidden = self.cfg["hidden_size"]
        self.verify_off = torch.arange(c, dtype=torch.long, device="cuda")
        self.verify_toks = torch.zeros(c, dtype=torch.long, device="cuda")
        self.verify_pos = self.verify_off.clone()
        self.verify_hidden = torch.zeros(c, hidden, dtype=D, device="cuda")
        self.verify_logits = torch.zeros(c, self.embed.shape[0], dtype=D, device="cuda")
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                self._verify_impl()
        torch.cuda.current_stream().wait_stream(s)
        self.reset()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._verify_impl()
        self._verify_graph = g
        self.reset()

    @torch.inference_mode()
    def spec_step(self, logits_prev: torch.Tensor, K: int) -> tuple[list[int], torch.Tensor]:
        """Draft K tokens with the MTP, verify all K+1 at once, accept the
        matching prefix and rewind the recurrent state to it.

        Returns ``(accepted_token_ids, logits_after_prefix)``. The main model's
        own next token is always accepted, so at least one token advances.

        Both halves are graph replays and every position/payload stays on the
        device, so the only host round trip per step is the acceptance test --
        which the caller needs anyway to emit the tokens.
        """
        assert self.mtp is not None and self._verify_graph is not None, "MTP not available"
        assert K + 1 <= self.verify_c, f"K+1={K + 1} exceeds verify width {self.verify_c}"

        # draft: t0 is the main model's own next token, then K chained MTP steps.
        # The drafter rotates at the same mRoPE position as the main model, so the
        # request's offset is copied across before the first replay.
        self.mtp.pos.copy_(self.pos1).sub_(1)
        self.mtp.rope_delta.copy_(self.rope_delta1)
        t0 = logits_prev.argmax()
        self.mtp.draft(self.h_prev, t0, K, self.verify_toks[:K + 1])

        # verify: one replay of the c-token graph
        self.verify_pos.copy_(self.verify_off).add_(self.pos1)
        self._verify_graph.replay()

        match = (self.verify_logits[:K].argmax(-1) == self.verify_toks[1:K + 1]).tolist()
        j = 0
        while j < K and match[j]:
            j += 1
        toks = self.verify_toks[:j + 1].tolist()

        # commit the accepted prefix, rewind everything else
        self.n_drafted += K
        self.n_accepted += j
        self.n_spec_steps += 1
        self.pos1.add_(j + 1)
        self.h_prev.copy_(self.verify_hidden[j])
        self.mtp.pos.copy_(self.pos1).sub_(1)
        for layer in self.layers:
            mix = layer.mix
            if isinstance(mix, LinearAttn):
                mix.state[self.prefill_row].copy_(mix.ckpt_state[j])
                mix.conv_state[self.prefill_row].copy_(mix.ckpt_conv[j])
        return toks, self.verify_logits[j]

    @torch.inference_mode()
    def spec_generate(self, prompt: list[int], max_new: int, K: int = 3,
                      eos: set[int] | None = None) -> list[int]:
        """Greedy speculative decode; returns up to ``max_new`` new token ids."""
        eos = eos or set()
        self.n_drafted = self.n_accepted = self.n_spec_steps = 0
        logits = self.prefill(prompt)
        out: list[int] = []
        while len(out) < max_new:
            if int(self.pos1) + self.verify_c > self.max_seq:
                break
            toks, logits = self.spec_step(logits, K)
            for t in toks:
                if t in eos:
                    return out
                out.append(t)
                if len(out) >= max_new:
                    break
        return out

    @property
    def spec_stats(self) -> dict:
        """Acceptance report for the last :meth:`spec_generate` run.

        ``accept_rate`` is the fraction of *drafted* tokens the main model
        agreed with; ``tokens_per_step`` is what the speedup is proportional to,
        since every step costs about one verify pass.
        """
        steps = max(self.n_spec_steps, 1)
        tokens = self.n_accepted + self.n_spec_steps      # drafts + one bonus each
        return {
            "steps": self.n_spec_steps,
            "drafted": self.n_drafted,
            "accepted": self.n_accepted,
            "tokens": tokens,
            "accept_rate": round(self.n_accepted / max(self.n_drafted, 1), 4),
            "tokens_per_step": round(tokens / steps, 3),
        }

    def decode_step(self, token: int) -> torch.Tensor:
        """Set the input token, replay the single-stream graph, return its logits."""
        pr = self.prefill_row
        self.token_in[pr:pr + 1].fill_(token)
        self._graph.replay()
        return self.logits_single

    @property
    def graph_ready(self) -> bool:
        return self._graph is not None

    def close(self) -> None:
        """Drop the captured graphs and the layer objects.

        ``serve.py`` rebuilds the model after reloading the code; a live
        ``CUDAGraph`` pins its private memory pool (about half the model's
        footprint), so simply dropping the model reference is not enough.
        """
        self._graph = None
        self._verify_graph = None
        self._batch_graphs = {}
        self._verify_batch_graphs = {}
        if self.mtp is not None:
            self.mtp.close()
        self.layers = []
        self.embed = None
        self.lm_head = None
        self.mtp = None
        self.vision = None
        self.vision_sd = {}
