"""Paged KV storage with block sharing and copy-on-write.

The attention window at decode time is a grow-only prefix of the sequence, so the
KV cache does not have to be one contiguous ``[nkv, max_seq, hd]`` buffer per
layer. Splitting it into fixed-size blocks buys two things the dense layout
cannot give:

* two sequences that share a prompt prefix can point at the *same* physical
  blocks (refcount + copy-on-write instead of a defensive copy); this is what
  makes :mod:`radix_cache` worth having;
* a finished sequence returns its blocks to a pool, so KV memory tracks live
  sequences instead of ``max_seq`` x concurrent sequences.

Layout:
All full-attention layers of the model share one pool ``[L, num_slots, nkv, hd]``
of int8 with a per-(head, token) fp32 scale -- the same int8 scheme the dense
cache uses, so the dequantising attention kernel is unchanged apart from how it
addresses a row.

A sequence owns a *slot map*: one int64 pool slot per absolute position, kept on
the device as ``pos2slot``. The decode step is graph-captured, so attention has
to resolve positions through device memory rather than through arguments baked in
at capture time; with the map on the device, one captured graph works for every
sequence and every position. A block ``b`` covers positions ``[b*bs, (b+1)*bs)``
and occupies slots ``[b*bs, (b+1)*bs)``, so slot = ``block*bs + (pos % bs)``.

``pos2slot`` is two-dimensional -- ``[rows, max_seq]`` with ``rows=1`` by
default.  The single-stream engine only ever uses row 0 (``slots_for(pos)`` and
``assign(block, start)`` default to it, so every old call site keeps working);
the batched decode path gives each row of the batch its own slot-map row and
passes the row view to the same kernels.
"""

from __future__ import annotations

import torch

D = torch.bfloat16


class OutOfBlocks(RuntimeError):
    """Raised when the block pool cannot satisfy an allocation."""


class PagedKVCache:
    """Shared int8 block pool for every full-attention layer.

    ``num_blocks`` x ``block_size`` slots are available to *all* sequences; a
    sequence's blocks are refcounted, so blocks shared through the radix tree are
    only reclaimed when the last holder drops them.
    """

    def __init__(self, n_layers: int, nkv: int, hd: int, num_blocks: int = 512,
                 block_size: int = 64, max_seq: int = 4096,
                 device: str = "cuda", rows: int = 1) -> None:
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
        n = self.bs * self.nkv * self.hd
        return self.n_layers * (2 * n + 2 * self.bs * self.nkv * 4)

    def memory_bytes(self) -> int:
        return self.bytes_per_block() * self.num_blocks

    def alloc(self) -> int:
        if not self.free:
            raise OutOfBlocks(f"no free KV blocks (used {self.num_used}/{self.num_blocks})")
        b = self.free.pop()
        self.refcount[b] = 1
        self.n_allocs += 1
        return b

    def incref(self, b: int) -> None:
        self.refcount[b] += 1

    def decref(self, b: int) -> None:
        self.refcount[b] -= 1
        if self.refcount[b] == 0:
            self.free.append(b)

    def free_blocks(self, blocks: list[int]) -> None:
        for b in blocks:
            self.decref(b)

    def copy_block(self, src: int, dst: int) -> None:
        """Copy every layer's slots for ``src`` into ``dst`` (copy-on-write)."""
        bs = self.bs
        self.k[:, dst * bs:(dst + 1) * bs].copy_(self.k[:, src * bs:(src + 1) * bs])
        self.v[:, dst * bs:(dst + 1) * bs].copy_(self.v[:, src * bs:(src + 1) * bs])
        self.kscale[:, dst * bs:(dst + 1) * bs].copy_(self.kscale[:, src * bs:(src + 1) * bs])
        self.vscale[:, dst * bs:(dst + 1) * bs].copy_(self.vscale[:, src * bs:(src + 1) * bs])
        self.n_cows += 1

    @property
    def rows(self) -> int:
        """Number of slot-map rows (batch capacity; 1 = single-stream)."""
        return self.pos2slot.shape[0]

    def set_rows(self, rows: int) -> None:
        """Grow (or shrink) the slot map to ``rows`` rows, preserving row 0.

        Must run before any graph is captured: the captured attention kernels
        bake in the row views they are handed, and the row stride of
        ``pos2slot`` is part of that.  Growing is done by reallocating the map,
        which is why this cannot happen once a graph exists.
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
        """Point positions ``[start, start+bs)`` of ``row`` at ``block``."""
        lo, hi = start, min(start + self.bs, self.max_seq)
        if hi <= lo:
            return
        self.pos2slot[row, lo:hi] = block * self.bs + torch.arange(
            hi - lo, device=self.pos2slot.device)

    def assign_table(self, blocks: list[int], start: int = 0, row: int = 0) -> None:
        for i, b in enumerate(blocks):
            self.assign(b, start + i * self.bs, row)

    @staticmethod
    def _quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """[..., hd] bf16/fp32 -> (int8, per-row scale)."""
        xf = x.float()
        s = xf.abs().amax(-1).clamp_min(1e-5) / 127.0
        q = (xf / s.unsqueeze(-1)).round().clamp_(-127, 127).to(torch.int8)
        return q, s

    def write_slot(self, layer: int, slot: torch.Tensor, k: torch.Tensor,
                   sk: torch.Tensor, v: torch.Tensor, sv: torch.Tensor) -> None:
        """Write one token of one layer at ``slot`` (int8 already quantised)."""
        li = torch.full_like(slot, layer)
        self.k.index_put_((li, slot), k.unsqueeze(0))
        self.v.index_put_((li, slot), v.unsqueeze(0))
        self.kscale.index_put_((li, slot), sk.unsqueeze(0))
        self.vscale.index_put_((li, slot), sv.unsqueeze(0))

    def write_range(self, layer: int, slots: torch.Tensor, k: torch.Tensor,
                    sk: torch.Tensor, v: torch.Tensor, sv: torch.Tensor) -> None:
        """Write ``T`` tokens of one layer at ``slots`` ([T])."""
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
        one indexed write per pool tensor.
        """
        li = torch.full_like(slots, layer)
        self.k.index_put_((li, slots), k)
        self.v.index_put_((li, slots), v)
        self.kscale.index_put_((li, slots), sk)
        self.vscale.index_put_((li, slots), sv)

    def slots_for(self, positions: torch.Tensor, row: int = 0) -> torch.Tensor:
        return self.pos2slot[row].index_select(0, positions)

    def gather(self, layer: int, n_tokens: int, row: int = 0) -> tuple[torch.Tensor, torch.Tensor,
                                                                      torch.Tensor, torch.Tensor]:
        """Dense K/V for the active sequence's first ``n_tokens`` positions.

        Returned as ``[nkv, S, hd]`` (+ ``[nkv, S]`` scales) to match the dense
        cache's head-major layout, so the prefill/verify attention code is the same
        for both. Those paths are batched and compute-bound, so materialising a
        dense view of the pool is the cheap option there -- the same choice the
        reference paged implementation makes.
        """
        slots = self.pos2slot[row][:n_tokens]
        return (self.k[layer].index_select(0, slots).transpose(0, 1),
                self.v[layer].index_select(0, slots).transpose(0, 1),
                self.kscale[layer].index_select(0, slots).transpose(0, 1),
                self.vscale[layer].index_select(0, slots).transpose(0, 1))


class SeqKV:
    """One sequence's view of the pool: a block list and a slot map."""

    def __init__(self, cache: PagedKVCache, blocks: list[int] | None = None,
                 num_tokens: int = 0) -> None:
        self.cache = cache
        self.blocks: list[int] = list(blocks or [])
        self.num_tokens = num_tokens
        self._pending = 0
        self._start = num_tokens

    def __len__(self) -> int:
        return self.num_tokens

    @property
    def bs(self) -> int:
        return self.cache.bs

    @property
    def total_tokens(self) -> int:
        return self.num_tokens + self._pending

    def reserve(self, t: int, radix=None) -> None:
        """Allocate blocks for ``t`` new tokens and open a write window.

        A partial tail block that is still shared with the tree (or another
        sequence) is copied before the window opens, so the writes that follow
        cannot corrupt the shared prefix.
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
        self.num_tokens += self._pending
        self._pending = 0

    def commit_upto(self, n_tokens: int) -> None:
        """Set the sequence length from the model's own position counter.

        Speculative decoding advances by a variable number of tokens per step, so
        the length is taken from the model (which is the authority on how much KV
        it has written) rather than from the reserve/commit bookkeeping.
        """
        self.num_tokens = int(n_tokens)
        self._pending = 0

    def free(self) -> None:
        self.cache.free_blocks(self.blocks)
        self.blocks = []
        self.num_tokens = 0
