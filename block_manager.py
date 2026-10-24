"""Block ownership: the paged pool, its refcounts and the radix prefix tree.

The pool (``paged_cache.PagedKVCache``) already knew how to hand out blocks and
share them; the radix tree (``radix_cache.RadixCache``) already knew which
prefixes are worth keeping.  What was missing is the object that owns *both* and
answers the questions a scheduler asks: can this request be admitted, which
blocks can it reuse, and who still holds the blocks it is about to release.

Phase 0 keeps the single-stream behaviour: one sequence owns the pool at a time
and the radix tree is consulted before a prefill.  The scheduler API here
(``can_allocate`` / ``allocate`` / ``free`` / ``assign``) is the shape the
continuous-batching scheduler will drive in phase 2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from paged_cache import PagedKVCache
    from radix_cache import RadixCache


#: Default block granularity; ``serve.py`` exposes it as ``--block-size``.
DEFAULT_BLOCK_SIZE = 64


class BlockManager:
    """The pool plus its prefix tree, and the refcount bookkeeping between them."""

    def __init__(self, pool: "PagedKVCache", radix: "RadixCache | None" = None,
                 block_size: int = DEFAULT_BLOCK_SIZE) -> None:
        self.pool = pool
        self.radix = radix
        self.block_size = block_size

    @classmethod
    def attach(cls, model, num_blocks: int, block_size: int = DEFAULT_BLOCK_SIZE,
               snapshots: int = 8, with_radix: bool = True) -> "BlockManager":
        """Create the paged pool on ``model`` and wrap it with a radix tree.

        ``with_radix=False`` is the batched-decode pool: same storage and block
        refcounts, but no prefix reuse (the batched scheduler allocates a
        sequence's blocks itself until phase 2 teaches it per-sequence prefixes).
        """
        from radix_cache import RadixCache

        pool = model.enable_paging(num_blocks, block_size)
        radix = RadixCache(pool, snapshots) if with_radix else None
        return cls(pool, radix, block_size)

    # per-sequence allocation (batched path)
    def ensure_blocks(self, seq, num_tokens: int) -> list[int]:
        """Allocate blocks until ``seq`` can hold ``num_tokens`` positions."""
        need = (num_tokens + self.block_size - 1) // self.block_size
        while len(seq.block_table) < need:
            seq.block_table.append(self.pool.alloc())
        return seq.block_table

    def release(self, seq) -> None:
        """Drop ``seq``'s reference on every block it holds."""
        self.free(seq.block_table)
        seq.block_table.clear()

    def reserve(self, blocks_needed: int) -> None:
        """Evict cached prefixes until ``blocks_needed`` free blocks exist."""
        if self.radix is not None:
            self.radix.reserve(blocks_needed)

    def match_prefix(self, token_ids: list[int], require_states: bool = False,
                     mm=None):
        """Longest cached block prefix of ``token_ids`` (blocks, length, states).

        ``mm`` (``mm_input.MMInput | None``) is the request's image payload; the
        radix tree salts its block keys with it so a prompt carrying a *different*
        image cannot reuse this prefix's KV.  ``None`` is the plain text path and
        costs nothing extra.
        """
        if self.radix is None:
            return [], 0, None
        return self.radix.match_prefix(token_ids, require_states=require_states,
                                       mm=mm)

    def insert(self, token_ids: list[int], blocks: list[int], states,
               mm=None) -> None:
        if self.radix is not None:
            self.radix.insert(token_ids, blocks, states, mm=mm)

    def assign(self, blocks: list[int], row: int = 0) -> None:
        """Point the pool's slot map for ``row`` at ``blocks``."""
        if row == 0:
            self.pool.assign_table(blocks, 0)
        else:
            self.pool.assign_table(blocks, 0, row=row)

    def free(self, blocks: list[int]) -> None:
        self.pool.free_blocks(blocks)

    def clear_prefix_cache(self) -> None:
        """Drop every cached prefix (A/B cold-start measurement, tests)."""
        if self.radix is not None:
            self.radix.clear()

    @property
    def num_blocks(self) -> int:
        return self.pool.num_blocks

    @property
    def num_used(self) -> int:
        return self.pool.num_used

    @property
    def num_free(self) -> int:
        return self.pool.num_free

    def stats(self) -> dict:
        """Prefix-cache counters, as reported by ``serve.py``'s ``ping``."""
        out = {"blocks_used": self.num_used, "blocks_total": self.num_blocks}
        if self.radix is not None:
            out.update({
                "cached_blocks": self.radix.num_cached_blocks(),
                "snapshots": self.radix.num_snapshots(),
                "snapshot_mib": round(self.radix.snapshot_bytes() / 2**20, 1),
                "evicted_blocks": self.radix.num_evicted_blocks,
            })
        return out
