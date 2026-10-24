"""Radix-tree prefix cache (RadixAttention) over the paged KV blocks.

Every *block-aligned* token prefix that has been seen is stored in a radix tree
together with the physical blocks holding its KV. A new request takes the longest
cached prefix (``refcount++`` on those blocks) and only prefills the suffix, so a
shared system prompt is computed once instead of once per request.

Hybrid models:
Qwen3.5 interleaves 8 *full attention* layers (paged KV, shareable) with 24
*linear attention* layers whose state is a function of the whole prefix. Reusing
a KV prefix is therefore only correct if the matching recurrent state is restored
too -- the KV alone would pair the shared keys with a state that never saw them.
Each node carries ``block_states``: an optional snapshot of the linear-attention
conv/recurrent states for every block along its edge, so a match can be replayed
from any block boundary that has one. ``match_prefix(..., require_states=True)``
truncates the match to the deepest boundary that carries a snapshot, and the
caller restores that state.

Snapshots are big (24 layers x [32,128,128] fp32 ~ 50 MB), so a snapshot budget
bounds them: the node end of every stored prefix gets one, and intermediate block
boundaries only while the budget lasts. A prefix that was inserted without a
snapshot still matches -- the caller just falls back to recomputing the part it
cannot restore, never to a silently wrong state.

Why block-aligned: KV is only shared at whole-block granularity, exactly like
vLLM's automatic prefix caching and SGLang's RadixCache. Children are keyed by
the full first block, so two prefixes that diverge inside their first block stay
distinct edges.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from paged_cache import PagedKVCache

#: A linear-attention snapshot: model layer index -> (recurrent, conv) state.
LinearStates = dict[int, tuple[Any, Any]]


class RadixNode:
    __slots__ = ("parent", "children", "key", "blocks", "refcount",
                 "last_access", "block_states")

    def __init__(self, parent: Optional["RadixNode"] = None, key: tuple = (),
                 blocks: Optional[list[int]] = None) -> None:
        self.parent = parent
        self.children: dict[tuple, "RadixNode"] = {}
        self.key: tuple = tuple(key)              # token ids along this edge
        self.blocks: list[int] = list(blocks or [])
        self.refcount = 0                         # live sequences on this node
        self.last_access = time.time()
        self.block_states: list[Optional[LinearStates]] = [None] * len(self.blocks)

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def num_states(self) -> int:
        return sum(1 for s in self.block_states if s is not None)


class RadixCache:
    """Prefix cache over a :class:`~paged_cache.PagedKVCache`'s blocks."""

    def __init__(self, cache: PagedKVCache, max_state_snapshots: int = 8) -> None:
        self.cache = cache
        self.bs = cache.bs
        self.root = RadixNode()
        self.max_state_snapshots = max_state_snapshots
        self.hit_tokens = 0
        self.total_query_tokens = 0
        self.num_evicted_blocks = 0

    def match_prefix(self, tokens: list[int], require_states: bool = False
                     ) -> tuple[list[int], int, Optional[LinearStates]]:
        """Longest cached prefix of ``tokens``.

        Returns ``(blocks, matched_len, states)``. ``matched_len`` is a multiple
        of ``block_size``, and the blocks are already refcounted for the caller:
        adopt them into a :class:`~paged_cache.SeqKV` and free them when done.
        """
        self.total_query_tokens += len(tokens)
        node, i = self.root, 0
        out: list[int] = []
        now = time.time()
        cp_len, cp_n, cp_states = 0, 0, None      # deepest usable checkpoint

        while i + self.bs <= len(tokens):
            child = node.children.get(tuple(tokens[i:i + self.bs]))
            if child is None:
                break
            nb = 0
            m = len(child.blocks)
            while nb < m and (i + (nb + 1) * self.bs) <= len(tokens):
                seg = tuple(tokens[i + nb * self.bs:i + (nb + 1) * self.bs])
                if seg != child.key[nb * self.bs:(nb + 1) * self.bs]:
                    break
                nb += 1
            for b in child.blocks[:nb]:
                self.cache.incref(b)
                out.append(b)
            if require_states and nb > 0:
                for k in range(nb - 1, -1, -1):
                    if child.block_states[k] is not None:
                        cp_len = i + (k + 1) * self.bs
                        cp_n = len(out) - (nb - (k + 1))
                        cp_states = child.block_states[k]
                        break
            i += nb * self.bs
            if nb < m:
                break                              # partial edge: stop here
            child.refcount += 1
            child.last_access = now
            node = child

        if require_states and cp_n < len(out):
            # Release the blocks we cannot justify a state for.
            for b in out[cp_n:]:
                self.cache.decref(b)
            out = out[:cp_n]
            i = cp_len

        self.hit_tokens += i
        return out, i, cp_states

    def insert(self, tokens: list[int], blocks: list[int],
               states_at: Optional[dict[int, LinearStates]] = None) -> None:
        """Insert a block-aligned prefix and its blocks.

        The tree takes its own reference on ``blocks``; the caller keeps its own
        until its sequence is freed. ``states_at`` maps an absolute token
        position (a multiple of ``block_size``) to the linear-attention states
        observed there.
        """
        states_at = states_at or {}
        n_full = (len(tokens) // self.bs) * self.bs
        if n_full == 0:
            return
        tokens = list(tokens[:n_full])
        blocks = list(blocks[:n_full // self.bs])
        node, i = self.root, 0
        while i < len(tokens):
            child = node.children.get(tuple(tokens[i:i + self.bs]))
            if child is None:
                self._new_leaf(node, tokens[i:], blocks[i // self.bs:], i // self.bs,
                               states_at)
                return
            m = len(child.blocks)
            nb = 0
            while nb < m and (i + (nb + 1) * self.bs) <= len(tokens):
                seg = tuple(tokens[i + nb * self.bs:i + (nb + 1) * self.bs])
                if seg != child.key[nb * self.bs:(nb + 1) * self.bs]:
                    break
                nb += 1
            if nb == m and (i + m * self.bs) < len(tokens):
                i += m * self.bs
                node = child
                continue
            if nb == m:
                self._adopt_states(child, i, nb, states_at)
                return                             # tokens end exactly at the child
            mid = self._split(child, nb)
            rest_tok = tokens[i + nb * self.bs:]
            rest_blk = blocks[i // self.bs + nb:]
            if rest_tok:
                self._new_leaf(mid, rest_tok, rest_blk, (i + nb * self.bs) // self.bs,
                               states_at)
            return

    def _adopt_states(self, node: RadixNode, i: int, nb: int,
                      states_at: dict[int, LinearStates]) -> None:
        for k in range(nb):
            pos = i + (k + 1) * self.bs
            if node.block_states[k] is None and pos in states_at:
                node.block_states[k] = states_at[pos]
                self._trim_states()

    def _new_leaf(self, parent: RadixNode, key: list[int], blocks: list[int],
                  start_block: int, states_at: dict[int, LinearStates]) -> RadixNode:
        node = RadixNode(parent=parent, key=key, blocks=blocks)
        for b in node.blocks:
            self.cache.incref(b)
        node.block_states = [states_at.get((start_block + k + 1) * self.bs)
                             for k in range(len(node.blocks))]
        parent.children[tuple(key[:self.bs])] = node
        self._trim_states()
        return node

    def _split(self, child: RadixNode, nb: int) -> RadixNode:
        """Split ``child`` after ``nb`` blocks; return the new upper node."""
        mid = RadixNode(parent=child.parent, key=child.key[:nb * self.bs],
                        blocks=child.blocks[:nb])
        mid.refcount = child.refcount
        mid.last_access = child.last_access
        mid.block_states = child.block_states[:nb]
        child.parent.children[mid.key[:self.bs]] = mid
        child.parent = mid
        child.key = child.key[nb * self.bs:]
        child.blocks = child.blocks[nb:]
        child.block_states = child.block_states[nb:]
        mid.children[child.key[:self.bs]] = child
        return mid

    def _trim_states(self) -> None:
        """Drop the LRU snapshots past the budget (blocks themselves are kept)."""
        if self.max_state_snapshots <= 0:
            return
        holders = []
        stack = [self.root]
        while stack:
            n = stack.pop()
            for k, s in enumerate(n.block_states):
                if s is not None:
                    holders.append((n.last_access, n, k))
            stack.extend(n.children.values())
        if len(holders) <= self.max_state_snapshots:
            return
        holders.sort(key=lambda h: h[0])
        for _, node, k in holders[:len(holders) - self.max_state_snapshots]:
            node.block_states[k] = None

    def _evictable_leaf(self) -> Optional[RadixNode]:
        best, best_t = None, float("inf")
        stack = [self.root]
        while stack:
            n = stack.pop()
            for c in n.children.values():
                if (c.is_leaf and c.refcount == 0
                        and all(self.cache.refcount[b] == 1 for b in c.blocks)):
                    if c.last_access < best_t:
                        best, best_t = c, c.last_access
                elif not c.is_leaf:
                    stack.append(c)
        return best

    def evict(self, blocks_needed: int) -> int:
        """Free LRU, unshared leaves until ``blocks_needed`` blocks are free."""
        freed = 0
        while self.cache.num_free < blocks_needed:
            leaf = self._evictable_leaf()
            if leaf is None:
                break
            for b in leaf.blocks:
                self.cache.decref(b)
            freed += len(leaf.blocks)
            self.num_evicted_blocks += len(leaf.blocks)
            del leaf.parent.children[leaf.key[:self.bs]]
        return freed

    def reserve(self, blocks_needed: int) -> None:
        """Make room for an allocation, evicting cached prefixes if needed."""
        if self.cache.num_free < blocks_needed:
            self.evict(blocks_needed)

    def clear(self) -> None:
        """Drop the whole tree, releasing every cached block.

        Only valid while no sequence holds blocks it adopted from the tree (the
        engine calls it when idle); used by ``LLMEngine.clear_prefix_cache`` for
        A/B measurements of a cold start.
        """
        stack = [self.root]
        while stack:
            node = stack.pop()
            for b in node.blocks:
                self.cache.decref(b)
            stack.extend(node.children.values())
        self.root = RadixNode()

    def num_cached_blocks(self) -> int:
        n = 0
        stack = [self.root]
        while stack:
            node = stack.pop()
            n += len(node.blocks)
            stack.extend(node.children.values())
        return n

    def num_snapshots(self) -> int:
        n = 0
        stack = [self.root]
        while stack:
            node = stack.pop()
            n += node.num_states()
            stack.extend(node.children.values())
        return n

    def snapshot_bytes(self) -> int:
        """Host-side estimate of the resident snapshot memory (rough)."""
        return self.num_snapshots() * 24 * (32 * 128 * 128 * 4 + 8192 * 3 * 4)

    def depth(self) -> int:
        best = 0
        stack = [(self.root, 0)]
        while stack:
            node, d = stack.pop()
            best = max(best, d)
            stack.extend((c, d + 1) for c in node.children.values())
        return best

    @property
    def hit_rate(self) -> float:
        if self.total_query_tokens == 0:
            return 0.0
        return self.hit_tokens / self.total_query_tokens
