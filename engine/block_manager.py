"""Block ownership: the paged pool, its refcounts and the radix prefix tree.

Module: ``coffin/`` (top level), used by ``llm_engine`` and ``prefix_engine``.
The pool (``paged_cache.PagedKVCache``) hands out and shares blocks; the radix
tree (``radix_cache.RadixCache``) knows which prefixes are worth keeping. This
is the object that owns both and answers what a scheduler actually asks: can
this request be admitted, which blocks can it reuse, and who still holds the
blocks it is about to release.

Constraints:
    * A block id is only valid while it is referenced by the radix tree, by a
      ``Sequence.block_table``, or both. ``free`` drops the pool reference, so
      calling it on a block the tree still caches is a corruption bug, not a
      leak.
    * ``assign`` only rewrites the pool's slot map; it does not take a
      reference. The caller must already own the blocks.
    * No locking. The engine's step loop is the only writer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from cache.paged_cache import PagedKVCache
    from cache.radix_cache import RadixCache


#: Default block granularity; the runtime CLI exposes it as ``--block-size``.
#: 64 matches the snapshot interval floor, so a recurrent-state snapshot never
#: lands mid-block.
DEFAULT_BLOCK_SIZE = 64


class BlockManager:
    """The pool plus its prefix tree, and the refcount bookkeeping between them."""

    def __init__(self, pool: "PagedKVCache", radix: "RadixCache | None" = None,
                 block_size: int = DEFAULT_BLOCK_SIZE) -> None:
        """Wrap an already-built pool.

        Args:
            pool: The paged KV pool, normally built by :meth:`attach` calling
                ``model.enable_paging``.
            radix: Prefix tree over the same pool, or ``None`` for the
                batched-decode pool that does not share prefixes.
            block_size: Tokens per block. Must agree with the pool's own block
                size, since it is used to translate token counts into blocks.
        """
        self.pool = pool
        self.radix = radix
        self.block_size = block_size

    @classmethod
    def attach(cls, model, num_blocks: int, block_size: int = DEFAULT_BLOCK_SIZE,
               snapshots: int = 8, with_radix: bool = True) -> "BlockManager":
        """Create the paged pool on ``model`` and wrap it with a radix tree.

        Args:
            model: A ``PagedModel``; ``enable_paging`` must exist on it.
            num_blocks: Physical blocks in the pool. ``-1`` is resolved by the
                config layer, not here.
            block_size: Tokens per block.
            snapshots: Recurrent-state snapshot budget handed to the radix tree.
            with_radix: ``False`` builds the batched-decode pool — same storage
                and refcounts, no prefix reuse. The batched scheduler allocates
                a sequence's blocks itself until per-sequence prefixes exist.

        Returns:
            The manager owning the new pool.
        """
        from cache.radix_cache import RadixCache

        pool = model.enable_paging(num_blocks, block_size)
        radix = RadixCache(pool, snapshots) if with_radix else None
        return cls(pool, radix, block_size)

    # Per-sequence allocation, used by the batched path.
    def ensure_blocks(self, seq, num_tokens: int) -> list[int]:
        """Allocate blocks until ``seq`` can hold ``num_tokens`` positions.

        Only ever grows the table. Shrinking here would drop the tail of a
        sequence that is still decoding, so release is a separate call.

        Args:
            seq: Sequence whose ``block_table`` is extended in place.
            num_tokens: Positions the sequence must be able to hold.

        Returns:
            The sequence's block table, extended as needed.

        Raises:
            OutOfBlocks: The pool is empty. The caller must evict the prefix
                cache and, if that is not enough, preempt a sequence; the request
                cannot grow here.
        """
        need = (num_tokens + self.block_size - 1) // self.block_size
        while len(seq.block_table) < need:
            seq.block_table.append(self.pool.alloc())
        return seq.block_table

    def release(self, seq) -> None:
        """Drop ``seq``'s reference on every block it holds.

        The tree keeps its own reference on anything it caches, so a preempting
        sequence can call this and still leave every block-aligned prefix reusable
        by the next admission. A block with no other holder returns to the pool.

        Args:
            seq: Sequence being retired or preempted. Its table is cleared, so a
                second call frees nothing rather than freeing the same ids twice.
        """
        self.free(seq.block_table)
        seq.block_table.clear()

    def reserve(self, blocks_needed: int) -> None:
        """Evict cached prefixes until ``blocks_needed`` free blocks exist.

        Args:
            blocks_needed: Blocks the next prefill requires. Called *before*
                allocating so eviction can still pick unreferenced leaves;
                afterwards the request itself would occupy the only candidates.
        """
        if self.radix is not None:
            self.radix.reserve(blocks_needed)

    def match_prefix(self, token_ids: list[int], require_states: bool = False,
                     mm=None):
        """Longest cached block prefix of ``token_ids``.

        Args:
            token_ids: Full request tokens.
            require_states: Require a recurrent-state snapshot at the hit point
                (mandatory for the hybrid model, where KV alone is not enough).
            mm: The request's image payload (``mm_input.MMInput | None``). The
                radix tree salts block keys with it, so a prompt carrying a
                *different* image cannot reuse this prefix's KV. ``None`` is the
                plain text path and costs nothing extra.

        Returns:
            ``(blocks, length, states)``; ``([], 0, None)`` when no prefix tree
            is attached.
        """
        if self.radix is None:
            return [], 0, None
        return self.radix.match_prefix(token_ids, require_states=require_states,
                                       mm=mm)

    def insert(self, token_ids: list[int], blocks: list[int], states,
               mm=None) -> None:
        """Record a prefilled prefix in the tree.

        Args:
            token_ids: Tokens the prefix covers, block-aligned.
            blocks: Physical blocks holding that prefix's KV.
            states: Recurrent-state snapshots for the block boundaries, or
                ``None`` when none were captured.
            mm: Media payload to salt the keys with; must match what
                :meth:`match_prefix` will be called with.
        """
        if self.radix is not None:
            self.radix.insert(token_ids, blocks, states, mm=mm)

    def assign(self, blocks: list[int], row: int = 0) -> None:
        """Point the pool's slot map for ``row`` at ``blocks``.

        Row 0 is the single-stream path and skips the row argument entirely, so
        that path pays no extra dispatch on the hot decode loop.

        Args:
            blocks: Block ids to map. The caller must already hold references.
            row: Batch row to map them onto.
        """
        if row == 0:
            self.pool.assign_table(blocks, 0)
        else:
            self.pool.assign_table(blocks, 0, row=row)

    def free(self, blocks: list[int]) -> None:
        """Return blocks to the pool.

        Args:
            blocks: Ids to release. Must not include blocks the radix tree still
                caches — evict them through the tree instead.
        """
        self.pool.free_blocks(blocks)

    def clear_prefix_cache(self) -> None:
        """Drop every cached prefix.

        Used by A/B cold-start measurements and by tests that need a known-cold
        engine without rebuilding the pool.
        """
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
        """Prefix-cache counters, as reported by the runtime ``ping``.

        Returns:
            Block totals always; the radix counters only when a tree is
            attached. ``snapshot_mib`` is rounded to one decimal because it is
            read by humans watching the 16 GB budget.
        """
        out = {"blocks_used": self.num_used, "blocks_total": self.num_blocks}
        if self.radix is not None:
            out.update({
                "cached_blocks": self.radix.num_cached_blocks(),
                "snapshots": self.radix.num_snapshots(),
                "snapshot_mib": round(self.radix.snapshot_bytes() / 2**20, 1),
                "evicted_blocks": self.radix.num_evicted_blocks,
            })
        return out
