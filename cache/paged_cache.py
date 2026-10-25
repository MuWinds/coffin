"""Paged KV storage with block sharing and copy-on-write.

Module: ``coffin/`` (top level, model-agnostic). ``radix_cache`` keys prefixes
over the blocks handed out here and ``block_manager`` owns both; model-family
attention code reaches the pool only through the slot map. This module must not
import a model family, and nothing here may assume a tokenizer or a layer count.

The attention window at decode time is a grow-only prefix of the sequence, so
the KV cache does not have to be one contiguous ``[nkv, max_seq, hd]`` buffer
per layer. Fixed-size blocks buy the two things the dense layout cannot: two
sequences that share a prompt prefix point at the *same* physical blocks
(refcount + copy-on-write instead of a defensive copy, which is what makes
:mod:`radix_cache` worth having), and a finished sequence returns its blocks to
the pool, so KV memory tracks live sequences instead of ``max_seq`` x concurrent
sequences.

All full-attention layers of the model share one int8 pool
``[L, num_slots, nkv, hd]`` with a per-(head, token) fp32 scale -- the same int8
scheme the dense cache uses, so the dequantising attention kernel is unchanged
apart from how it addresses a row. A sequence owns a *slot map*: one int64 pool
slot per absolute position, kept on the device as ``pos2slot``. Block ``b``
covers positions ``[b*bs, (b+1)*bs)`` and occupies slots ``[b*bs, (b+1)*bs)``,
so ``slot = block*bs + (pos % bs)``.

Constraints:
    * ``pos2slot`` stays in device memory because the decode step is
      graph-captured: attention resolves positions by indexing this tensor,
      where arguments baked in at capture time would need one graph per
      sequence and per position. One captured graph then serves every sequence
      at every position.
    * The slot map is ``[rows, max_seq]``, never 1-D. Row 0 is the
      single-stream map that ``slots_for`` and ``assign`` default to, so those
      call sites stay unchanged; batched decode gives each batch row its own map
      row and hands the same kernels a row view. The row stride is part of what
      a captured kernel bakes in, so rows are fixed once ``capture()`` has run.
    * No locking. ``llm_engine``'s step loop is the only writer, and a block id
      is valid only while a refcount holds it.
"""

from __future__ import annotations

import torch

D = torch.bfloat16


class OutOfBlocks(RuntimeError):
    """The pool has no unreferenced block left to hand out.

    Raised rather than returning a sentinel because the caller cannot recover:
    by the time an allocation fails the sequence has already been admitted, so
    the request must be failed and its blocks released. Sizing the pool so this
    never fires is ``BlockManager.reserve``'s job, not the caller's.
    """


class PagedKVCache:
    """Shared int8 block pool for every full-attention layer.

    ``num_blocks`` x ``block_size`` slots are available to *all* sequences and a
    sequence's blocks are refcounted, so blocks shared through the radix tree
    survive until the last holder drops them -- that shared reference is the
    whole point of the pool, and dropping one too many corrupts a live
    sequence's KV.
    """

    def __init__(self, n_layers: int, nkv: int, hd: int, num_blocks: int = 512,
                 block_size: int = 64, max_seq: int = 4096,
                 device: str = "cuda", rows: int = 1) -> None:
        """Provision the whole pool up front, as a VRAM commitment.

        Args:
            n_layers: Full-attention layers sharing the pool; each gets its own
                slice of every block.
            nkv: Key/value heads per layer. GQA packs them here, so this is
                smaller than the query-head count.
            hd: Head dimension.
            num_blocks: Physical blocks. ``-1`` is resolved by the config layer
                before it reaches here.
            block_size: Tokens per block; also the radix match granularity.
            max_seq: Longest position the slot map can address. A sequence
                longer than this cannot be represented even if blocks are free.
            device: Where the int8 pool and the slot map live.
            rows: Slot-map rows. ``1`` for single-stream; batched decode passes
                the batch capacity, and row ``capacity-1`` is the prefill row.
        """
        self.n_layers = n_layers
        self.nkv = nkv
        self.hd = hd
        self.num_blocks = num_blocks
        self.bs = block_size
        self.max_seq = max_seq
        self.num_slots = num_blocks * block_size
        shape = (n_layers, self.num_slots, nkv, hd)
        self.k = torch.zeros(shape, dtype=torch.int8, device=device)
        self.v = torch.zeros(shape, dtype=torch.int8, device=device)
        self.kscale = torch.zeros((n_layers, self.num_slots, nkv), dtype=torch.float32,
                                  device=device)
        self.vscale = torch.zeros((n_layers, self.num_slots, nkv), dtype=torch.float32,
                                  device=device)

        #: position -> pool slot, one map row per batch row (device-resident).
        #: Row 0 is the single-stream map; rows 1.. are the batched decode rows.
        self.pos2slot = torch.zeros(rows, max_seq, dtype=torch.int64, device=device)

        self.free: list[int] = list(range(num_blocks - 1, -1, -1))
        self.refcount: list[int] = [0] * num_blocks
        self.n_cows = 0
        self.n_allocs = 0

    @property
    def num_free(self) -> int:
        return len(self.free)

    @property
    def num_used(self) -> int:
        return self.num_blocks - len(self.free)

    def bytes_per_block(self) -> int:
        """Resident bytes of one block across every full-attention layer.

        The `2 * n` term is the int8 K/V payload (`bs * nkv * hd` bytes each),
        and `2 * bs * nkv * 4` is the fp32 per-(head, token) scale for K and V.
        The scale is not optional bookkeeping: the attention kernel reads it to
        dequantise, and an fp16 scale would lose precision on the ~1e-2 range
        these weights use.

        Returns:
            Bytes for one block, all layers included.
        """
        n = self.bs * self.nkv * self.hd
        return self.n_layers * (2 * n + 2 * self.bs * self.nkv * 4)

    def memory_bytes(self) -> int:
        """Total resident bytes of the pool at its current block count.

        Returns:
            ``bytes_per_block() * num_blocks``, i.e. the whole int8 cache.
        """
        return self.bytes_per_block() * self.num_blocks

    def alloc(self) -> int:
        """Take one unreferenced block and hand it out with refcount 1.

        Returns:
            The block id. It is the caller's to release with :meth:`decref`, and
            is only valid while a refcount holds it.

        Raises:
            OutOfBlocks: no unreferenced block remains. The pool is the KV
                budget of every live request, so the caller must evict cached
                prefixes (:meth:`radix_cache.RadixCache.reserve`) and retry.
        """
        if not self.free:
            raise OutOfBlocks(f"no free KV blocks (used {self.num_used}/{self.num_blocks})")
        b = self.free.pop()
        self.refcount[b] = 1
        self.n_allocs += 1
        return b

    def incref(self, b: int) -> None:
        """Add a reference to ``b``, for sharing a cached block with one more holder.

        Args:
            b: Block id already owned by someone -- the radix tree or another
                sequence. Incref never allocates, so it cannot fail.
        """
        self.refcount[b] += 1

    def decref(self, b: int) -> None:
        """Drop one reference to ``b``, returning it to the free list at zero.

        Args:
            b: Block id the caller holds. Releasing a block it does not own
                drives the count negative and frees a block someone still reads,
                so every :meth:`alloc` / :meth:`incref` must have exactly one
                matching call.
        """
        self.refcount[b] -= 1
        if self.refcount[b] == 0:
            self.free.append(b)

    def free_blocks(self, blocks: list[int]) -> None:
        """Drop one reference on each of ``blocks``.

        Args:
            blocks: Ids the caller holds, typically a sequence's whole table.
                Blocks the radix tree still caches keep their tree reference,
                so their KV stays valid after the sequence is retired.
        """
        for b in blocks:
            self.decref(b)

    def copy_block(self, src: int, dst: int) -> None:
        """Copy every layer's slots for ``src`` into ``dst`` (copy-on-write).

        Args:
            src: Block the sequence is currently sharing.
            dst: Freshly allocated private block for the same positions. The
                caller opens its write window at the block boundary and points
                the slot map at ``dst``; writing into ``src`` instead would
                corrupt every other sequence that shares it.
        """
        bs = self.bs
        self.k[:, dst * bs:(dst + 1) * bs].copy_(self.k[:, src * bs:(src + 1) * bs])
        self.v[:, dst * bs:(dst + 1) * bs].copy_(self.v[:, src * bs:(src + 1) * bs])
        self.kscale[:, dst * bs:(dst + 1) * bs].copy_(self.kscale[:, src * bs:(src + 1) * bs])
        self.vscale[:, dst * bs:(dst + 1) * bs].copy_(self.vscale[:, src * bs:(src + 1) * bs])
        self.n_cows += 1

    @property
    def rows(self) -> int:
        """Slot-map rows, i.e. batch capacity. ``1`` means single-stream only.

        Returns:
            Row count of ``pos2slot``; the batched model sizes it to its bucket
            capacity, with the last row reserved for prefill.
        """
        return self.pos2slot.shape[0]

    def set_rows(self, rows: int) -> None:
        """Grow or shrink the slot map to ``rows`` rows, preserving row 0.

        Must run before any graph is captured: the captured attention kernels
        bake in the row views they are handed, and the row stride of
        ``pos2slot`` is part of that. Growing reallocates the map, which is why
        this cannot happen once a graph exists.

        Args:
            rows: Rows wanted; clamped to at least 1, and a no-op when the count
                already matches. Rows past the old capacity start zeroed, which
                is why the caller must assign a table before the row is used.
        """
        rows = max(int(rows), 1)
        if rows == self.rows:
            return
        new = torch.zeros(rows, self.max_seq, dtype=torch.int64,
                          device=self.pos2slot.device)
        keep = min(rows, self.rows)
        new[:keep].copy_(self.pos2slot[:keep])
        self.pos2slot = new

    def assign(self, block: int, start: int, row: int = 0) -> None:
        """Point positions ``[start, start+bs)`` of ``row`` at ``block``.

        Args:
            block: Block whose slots those positions now live in. The caller
                must already hold a reference; ``assign`` only rewrites the map.
            start: Token position the block starts at, normally a multiple of
                the block size. A start past ``max_seq`` writes nothing.
            row: Slot-map row; row 0 is the single-stream path.
        """
        lo, hi = start, min(start + self.bs, self.max_seq)
        if hi <= lo:
            return
        self.pos2slot[row, lo:hi] = block * self.bs + torch.arange(
            hi - lo, device=self.pos2slot.device)

    def assign_table(self, blocks: list[int], start: int = 0, row: int = 0) -> None:
        """Point consecutive blocks into one map row, block-aligned.

        The paged pool holds all zeros on the map until this runs, so a prefill
        that skipped it makes every position address pool slot 0. Call it after
        ``enable_paging()`` and before ``capture()`` or ``prefill()``.

        Args:
            blocks: A sequence's block table, in position order.
            start: Position the first block begins at; blocks follow one block
                size apart.
            row: Slot-map row to rewrite.
        """
        for i, b in enumerate(blocks):
            self.assign(b, start + i * self.bs, row)

    @staticmethod
    def _quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantise ``[..., hd]`` to int8 with a per-row scale.

        Args:
            x: bf16 or fp32 rows. ``1e-5`` floors the scale so an all-zero row
                divides by a positive number instead of producing NaN.

        Returns:
            The int8 tensor and the fp32 scale, such that
            ``q.float() * s[..., None]`` reproduces ``x`` to int8 precision.
        """
        xf = x.float()
        s = xf.abs().amax(-1).clamp_min(1e-5) / 127.0
        q = (xf / s.unsqueeze(-1)).round().clamp_(-127, 127).to(torch.int8)
        return q, s

    def write_slot(self, layer: int, slot: torch.Tensor, k: torch.Tensor,
                   sk: torch.Tensor, v: torch.Tensor, sv: torch.Tensor) -> None:
        """Write one token of one layer at ``slot`` (int8 already quantised).

        Args:
            layer: Index into the pool's layer axis, i.e. the layer's position
                among the *full-attention* layers, not the model's block index.
            slot: Device scalar pool slot to write, normally
                ``slots_for(pos1)``.
            k: ``[nkv, hd]`` int8 key row.
            sk: ``[nkv]`` fp32 key dequant scale.
            v: ``[nkv, hd]`` int8 value row.
            sv: ``[nkv]`` fp32 value dequant scale.
        """
        li = torch.full_like(slot, layer)
        self.k.index_put_((li, slot), k.unsqueeze(0))
        self.v.index_put_((li, slot), v.unsqueeze(0))
        self.kscale.index_put_((li, slot), sk.unsqueeze(0))
        self.vscale.index_put_((li, slot), sv.unsqueeze(0))

    def write_range(self, layer: int, slots: torch.Tensor, k: torch.Tensor,
                    sk: torch.Tensor, v: torch.Tensor, sv: torch.Tensor) -> None:
        """Write ``T`` tokens of one layer at ``slots`` ([T]).

        Args:
            layer: Index among the full-attention layers.
            slots: ``[T]`` pool slots, resolved from the same slot map row the
                tokens are being written into.
            k: ``[T, nkv, hd]`` int8 keys.
            sk: ``[T, nkv]`` fp32 key scales.
            v: ``[T, nkv, hd]`` int8 values.
            sv: ``[T, nkv]`` fp32 value scales.
        """
        li = torch.full_like(slots, layer)
        self.k.index_put_((li, slots), k)
        self.v.index_put_((li, slots), v)
        self.kscale.index_put_((li, slots), sk)
        self.vscale.index_put_((li, slots), sv)

    def write_rows(self, layer: int, slots: torch.Tensor, k: torch.Tensor,
                   sk: torch.Tensor, v: torch.Tensor, sv: torch.Tensor) -> None:
        """Write one token of one layer for ``B`` rows at their own ``slots`` ([B]).

        The batched counterpart of :meth:`write_slot`: ``k``/``v`` are
        ``[B, nkv, hd]`` int8 and the scales ``[B, nkv]``, so all rows go in with
        one indexed write per pool tensor instead of ``B`` separate launches.

        Args:
            layer: Index among the full-attention layers.
            slots: ``[B]`` pool slot per batch row, gathered through that row's
                own slot-map row.
            k: ``[B, nkv, hd]`` int8 keys.
            sk: ``[B, nkv]`` fp32 key scales.
            v: ``[B, nkv, hd]`` int8 values.
            sv: ``[B, nkv]`` fp32 value scales.
        """
        li = torch.full_like(slots, layer)
        self.k.index_put_((li, slots), k)
        self.v.index_put_((li, slots), v)
        self.kscale.index_put_((li, slots), sk)
        self.vscale.index_put_((li, slots), sv)

    def slots_for(self, positions: torch.Tensor, row: int = 0) -> torch.Tensor:
        """Pool slots the given positions currently map to in ``row``.

        Args:
            positions: Device int64 positions; a captured graph passes the
                position counter here rather than a baked-in constant.
            row: Slot-map row; row 0 is the single-stream path.

        Returns:
            One pool slot per position, in the same order.
        """
        return self.pos2slot[row].index_select(0, positions)

    def gather(self, layer: int, n_tokens: int, row: int = 0) -> tuple[torch.Tensor, torch.Tensor,
                                                                      torch.Tensor, torch.Tensor]:
        """Dense K/V for the active sequence's first ``n_tokens`` positions.

        Returned as ``[nkv, S, hd]`` (+ ``[nkv, S]`` scales) to match the dense
        cache's head-major layout, so the prefill/verify attention code is the same
        for both. Those paths are batched and compute-bound, so materialising a
        dense view of the pool is the cheap option there -- the same choice the
        reference paged implementation makes.

        Args:
            layer: Index among the full-attention layers.
            n_tokens: Prefix length to materialise; read from the start of the
                row's slot map, so a shorter prefix than the sequence is fine.
            row: Slot-map row of the sequence being prefilled.

        Returns:
            ``(k, v, kscale, vscale)`` as head-major dense views of those
            positions, in position order.
        """
        slots = self.pos2slot[row][:n_tokens]
        return (self.k[layer].index_select(0, slots).transpose(0, 1),
                self.v[layer].index_select(0, slots).transpose(0, 1),
                self.kscale[layer].index_select(0, slots).transpose(0, 1),
                self.vscale[layer].index_select(0, slots).transpose(0, 1))


class SeqKV:
    """One sequence's view of the pool: a block list and a slot map.

    Every block in ``blocks`` is a reference this object holds, so :meth:`free`
    is the only correct way to retire it. ``num_tokens`` is the committed length
    -- KV the model has already written -- and a ``reserve``/``commit`` window on
    top of it is the region about to be written.
    """

    def __init__(self, cache: PagedKVCache, blocks: list[int] | None = None,
                 num_tokens: int = 0) -> None:
        """Wrap a block list the caller already holds a reference to.

        Args:
            cache: Pool those blocks belong to.
            blocks: Block table, adopted as-is; no ``incref`` happens here, so
                the caller's references transfer to this object.
            num_tokens: Tokens already written into those blocks. Not rounded to
                a block boundary: a prefix match can end mid-block.
        """
        self.cache = cache
        self.blocks: list[int] = list(blocks or [])
        self.num_tokens = num_tokens
        self._pending = 0
        self._start = num_tokens

    def __len__(self) -> int:
        """Committed length, the positions a decode step may attend over.

        Returns:
            Tokens whose KV is already written; a reservation in flight is not
            included, since it is not readable yet.
        """
        return self.num_tokens

    @property
    def bs(self) -> int:
        """Block size of the pool, in tokens.

        Returns:
            Tokens per block, read through to the pool because a sequence's
            block table is meaningless without it.
        """
        return self.cache.bs

    @property
    def total_tokens(self) -> int:
        """Length including the reservation that is not written yet.

        Returns:
            Committed length plus the open write window, i.e. how many tokens
            the caller has already planned to hold.
        """
        return self.num_tokens + self._pending

    def reserve(self, t: int, radix=None) -> None:
        """Allocate blocks for ``t`` new tokens and open a write window.

        A partial tail block that is still shared with the tree (or another
        sequence) is copied before the window opens, so the writes that follow
        cannot corrupt the shared prefix -- that copy is the copy-on-write the
        whole pool design exists for.

        Args:
            t: Tokens about to be written. ``0`` closes the window without
                allocating or copying, which is the no-op a decode step wants
                when it generated nothing.
            radix: Not consulted; kept for call-site compatibility. Eviction
                belongs to ``BlockManager.reserve``, which the engine calls
                before this allocates, so the free list is already stocked.

        Raises:
            OutOfBlocks: the pool ran dry mid-allocation. Only reachable when
                the caller skipped the eviction step; the sequence must then be
                failed and freed rather than retried.
        """
        if t <= 0:
            self._start, self._pending = self.num_tokens, 0
            return
        bs = self.bs
        cache = self.cache
        if self.blocks and self.num_tokens % bs != 0:
            tail = self.blocks[-1]
            if cache.refcount[tail] > 1:
                new = cache.alloc()
                cache.copy_block(tail, new)
                cache.decref(tail)
                self.blocks[-1] = new
                cache.assign(new, self.num_tokens - self.num_tokens % bs)
        total = self.num_tokens + t
        need = (total + bs - 1) // bs - len(self.blocks)
        for _ in range(need):
            blk = cache.alloc()
            self.blocks.append(blk)
            cache.assign(blk, (len(self.blocks) - 1) * bs)
        self._start, self._pending = self.num_tokens, t

    def commit(self) -> None:
        """Close the write window, counting the reserved tokens as written."""
        self.num_tokens += self._pending
        self._pending = 0

    def commit_upto(self, n_tokens: int) -> None:
        """Set the sequence length from the model's own position counter.

        Speculative decoding advances by a variable number of tokens per step, so
        the length is taken from the model (which is the authority on how much KV
        it has written) rather than from the reserve/commit bookkeeping.

        Args:
            n_tokens: The model's position counter after the step. Passing a
                value shorter than the committed length would drop KV the model
                still attends over, so callers read ``model.pos1``.
        """
        self.num_tokens = int(n_tokens)
        self._pending = 0

    def free(self) -> None:
        """Return every block to the pool and empty the sequence.

        A second call is a no-op because the table is cleared: releasing the same
        ids twice would corrupt the free list and hand a live block to the next
        allocation.
        """
        self.cache.free_blocks(self.blocks)
        self.blocks = []
        self.num_tokens = 0
