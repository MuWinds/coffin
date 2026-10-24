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

Media salt (images)
-------------------
Token ids are not the whole story once the prompt can carry an image: the
``<|image_pad|>`` placeholder run is identical for *every* image, while the KV
computed for it depends on the pixels.  A cache keyed on token ids alone would
hand a second, different image the first image's KV -- silently wrong output that
grows worse the longer the prompt is.

Every block therefore also carries a *media signature*: the set of media ids of
the tokens in it, where a token covered by image ``g`` gets a stable int derived
from that image's content digest (``MMInput.digest``) and its index in the
request, and every other token gets :data:`TEXT_MEDIA_ID` (0).  A text block's
signature is ``(0,)``, so its child key is the plain token tuple -- the text-only
path keeps exactly the old key (and the old behaviour, and the old cost: the
signature work is skipped entirely when no ``mm`` is passed).  An image block's
key is ``(tokens, signature)``, so two prompts with different images diverge at
the first block that contains image tokens.

Salting only the image blocks is enough, because a radix path is a *path*: blocks
after the image have signature ``(0,)`` again (no image token in them), but the
only way to reach them is through the image block that already diverged.  This is
the same argument vLLM makes for hashing the mm hash into the prefix hash, except
that only the blocks that actually contain mm tokens need it.  An id is also
salted with the image's index in the request, so the same picture appearing twice
never claims to be the same KV as itself in another position.

One asymmetry is worth spelling out, because it is the difference between a
working salt and a silent leak: an edge inserted *without* media information
(``mm=None``, i.e. the text-only path or any caller that predates this) has no
signatures, and "no signatures" must not mean "reusable by anything".  A block
inside such an edge may have been computed for an image nobody recorded, so a
query that *does* carry media may only adopt the blocks it can prove are
image-free.  See :meth:`RadixCache._sig_ok`.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Optional

from paged_cache import PagedKVCache
from mm_input import IMAGE_TOKEN_ID

#: A linear-attention snapshot: model layer index -> (recurrent, conv) state.
LinearStates = dict[int, tuple[Any, Any]]

#: Media id of a token no image covers (also the whole signature of a text block).
TEXT_MEDIA_ID = 0


def media_id(digest: str, index: int) -> int:
    """Stable positive int64 for image ``index`` of a request with ``digest``.

    Stable across requests (same bytes + same position -> same id, so the second
    turn of a conversation still hits), distinct per image content, and never 0
    (0 is reserved for text so a text block's signature stays ``(0,)``).
    """
    h = hashlib.sha256(f"{digest}#{index}".encode("utf-8")).digest()
    return (int.from_bytes(h[:8], "little") & ((1 << 63) - 1)) | 1


def media_ids_for(mm, num_tokens: int) -> Optional[list[int]]:
    """Per-token media id for the first ``num_tokens`` tokens of a prompt.

    ``None`` means "no media information": the caller has a text-only prompt (or
    no ``MMInput`` at all) and the tree keeps its plain token keys.  The result
    aligns with the *absolute* prompt index, so a caller that matches
    ``tokens[:-1]`` still passes the prefix of the very same list.
    """
    if mm is None:
        return None
    spans = getattr(mm, "spans", None)
    if not spans:
        return None
    digest = getattr(mm, "digest", "") or ""
    if not digest:
        # Refusing is the safe failure: without a content digest two different
        # images of the same size would share block keys and the cache would
        # return one image's KV for the other.  ``serve.py`` always fills it in
        # (``mm_input.image_digest``); an empty one with spans is a caller bug.
        raise ValueError(
            "MMInput has image spans but an empty digest; the radix tree cannot "
            "salt its block keys (use mm_input.image_digest)")
    ids = [TEXT_MEDIA_ID] * num_tokens
    for index, (start, end) in enumerate(spans):
        mid = media_id(digest, index)
        a, b = max(int(start), 0), min(int(end), num_tokens)
        if a < b:
            ids[a:b] = [mid] * (b - a)
    return ids


def media_signature(media: Optional[list[int]], start: int, bs: int) -> tuple[int, ...]:
    """Signature of the block at token ``start``: the sorted set of media ids."""
    if media is None:
        return (TEXT_MEDIA_ID,)
    ids = sorted(set(media[start:start + bs]))
    return tuple(ids) if ids else (TEXT_MEDIA_ID,)


class RadixNode:
    __slots__ = ("parent", "children", "key", "blocks", "refcount",
                 "last_access", "block_states", "sigs")

    def __init__(self, parent: Optional["RadixNode"] = None, key: tuple = (),
                 blocks: Optional[list[int]] = None) -> None:
        self.parent = parent
        self.children: dict[tuple, "RadixNode"] = {}
        self.key: tuple = tuple(key)              # token ids along this edge
        self.blocks: list[int] = list(blocks or [])
        self.refcount = 0                         # live sequences on this node
        self.last_access = time.time()
        self.block_states: list[Optional[LinearStates]] = [None] * len(self.blocks)
        #: Per-block media signature (``None`` = the edge holds no image token, so
        #: its child keys are plain token tuples and every request matches it).
        self.sigs: Optional[list[tuple[int, ...]]] = None

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

    def match_prefix(self, tokens: list[int], require_states: bool = False,
                     mm=None) -> tuple[list[int], int, Optional[LinearStates]]:
        """Longest cached prefix of ``tokens``.

        Returns ``(blocks, matched_len, states)``. ``matched_len`` is a multiple
        of ``block_size``, and the blocks are already refcounted for the caller:
        adopt them into a :class:`~paged_cache.SeqKV` and free them when done.

        ``mm`` (an ``mm_input.MMInput``) salts the block keys with the media each
        block contains; ``None`` keeps the plain token keys, so a text request
        pays nothing for the salt.  ``tokens`` is the *absolute* prompt prefix
        (the callers pass ``tokens[:-1]``), which is what makes the spans line up.
        """
        self.total_query_tokens += len(tokens)
        bs = self.bs
        media = media_ids_for(mm, len(tokens))
        node, i = self.root, 0
        out: list[int] = []
        now = time.time()
        cp_len, cp_n, cp_states = 0, 0, None      # deepest usable checkpoint

        while i + bs <= len(tokens):
            child = node.children.get(self._child_key(tokens, media, i))
            if child is None:
                break
            nb = 0
            m = len(child.blocks)
            while nb < m and (i + (nb + 1) * bs) <= len(tokens):
                seg = tuple(tokens[i + nb * bs:i + (nb + 1) * bs])
                if seg != child.key[nb * bs:(nb + 1) * bs]:
                    break
                if media is not None and not self._sig_ok(
                        media_signature(media, i + nb * bs, bs), child, nb, seg):
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

    def _sig_ok(self, sig: tuple[int, ...], node: "RadixNode", k: int,
                block_tokens: tuple) -> bool:
        """May block ``k`` of ``node``'s edge be reused by a query with ``sig``?

        Three cases, and the middle one is the subtle one:

        * the edge was inserted *with* media info and carries signatures: they
          have to be equal;
        * the edge was inserted *without* media info (``sigs is None``: the
          text-only path, or a caller that predates ``mm``).  Its blocks may well
          have been computed for an image we know nothing about, so the only
          branch that is provably safe is a block that holds no image token at
          all.  Accepting it unconditionally is exactly the silent cross-image
          hit this module exists to prevent;
        * a salted edge is stored under a salted child key, so a plain-token
          lookup never reaches it in the first place.
        """
        if node.sigs is None:
            return sig == (TEXT_MEDIA_ID,) and IMAGE_TOKEN_ID not in block_tokens
        return sig == node.sigs[k]

    def _child_key(self, tokens: list[int], media: Optional[list[int]], start: int
                   ) -> tuple:
        """Key of the child edge whose first block starts at ``start``."""
        key = tuple(tokens[start:start + self.bs])
        if media is None:
            return key
        sig = media_signature(media, start, self.bs)
        # A block with no image token keeps the *plain* key, whether or not the
        # request carried images: that is what lets a text prompt and an image
        # prompt share a text prefix (and keeps the text-only format unchanged).
        return key if sig == (TEXT_MEDIA_ID,) else key + sig

    def _edge_key(self, node: RadixNode) -> tuple:
        """The key ``node.parent.children`` stores ``node`` under."""
        key = tuple(node.key[:self.bs])
        if node.sigs is not None and node.sigs[0] != (TEXT_MEDIA_ID,):
            key += node.sigs[0]
        return key

    def insert(self, tokens: list[int], blocks: list[int],
               states_at: Optional[dict[int, LinearStates]] = None,
               mm=None) -> None:
        """Insert a block-aligned prefix and its blocks.

        The tree takes its own reference on ``blocks``; the caller keeps its own
        until its sequence is freed. ``states_at`` maps an absolute token
        position (a multiple of ``block_size``) to the linear-attention states
        observed there.  ``mm`` salts the block keys exactly like
        :meth:`match_prefix` does (pass the same ``MMInput`` to both, or the
        insert populates a branch no match will ever look at).
        """
        states_at = states_at or {}
        n_full = (len(tokens) // self.bs) * self.bs
        if n_full == 0:
            return
        tokens = list(tokens[:n_full])
        blocks = list(blocks[:n_full // self.bs])
        media = media_ids_for(mm, n_full)
        node, i = self.root, 0
        while i < len(tokens):
            child = node.children.get(self._child_key(tokens, media, i))
            if child is None:
                self._new_leaf(node, tokens[i:], blocks[i // self.bs:], i // self.bs,
                               states_at, media, i)
                return
            m = len(child.blocks)
            nb = 0
            while nb < m and (i + (nb + 1) * self.bs) <= len(tokens):
                seg = tuple(tokens[i + nb * self.bs:i + (nb + 1) * self.bs])
                if seg != child.key[nb * self.bs:(nb + 1) * self.bs]:
                    break
                if media is not None and not self._sig_ok(
                        media_signature(media, i + nb * self.bs, self.bs), child, nb,
                        seg):
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
                               states_at, media, i + nb * self.bs)
            return

    def _adopt_states(self, node: RadixNode, i: int, nb: int,
                      states_at: dict[int, LinearStates]) -> None:
        for k in range(nb):
            pos = i + (k + 1) * self.bs
            if node.block_states[k] is None and pos in states_at:
                node.block_states[k] = states_at[pos]
                self._trim_states()

    def _new_leaf(self, parent: RadixNode, key: list[int], blocks: list[int],
                  start_block: int, states_at: dict[int, LinearStates],
                  media: Optional[list[int]] = None, start_token: int = 0) -> RadixNode:
        node = RadixNode(parent=parent, key=key, blocks=blocks)
        for b in node.blocks:
            self.cache.incref(b)
        node.block_states = [states_at.get((start_block + k + 1) * self.bs)
                             for k in range(len(node.blocks))]
        node.sigs = self._signatures(media, start_token, len(node.blocks))
        parent.children[self._edge_key(node)] = node
        self._trim_states()
        return node

    def _signatures(self, media: Optional[list[int]], start_token: int,
                    n_blocks: int) -> Optional[list[tuple[int, ...]]]:
        """Per-block media signature of an edge, or ``None`` for a text-only one.

        ``None`` is the fast path: a text-only edge (no image token in it) keeps
        the old plain-token behaviour *and* keeps working for requests that were
        matched without any ``mm`` at all.
        """
        if media is None:
            return None
        sigs = [media_signature(media, start_token + k * self.bs, self.bs)
                for k in range(n_blocks)]
        return None if all(s == (TEXT_MEDIA_ID,) for s in sigs) else sigs

    def _split(self, child: RadixNode, nb: int) -> RadixNode:
        """Split ``child`` after ``nb`` blocks; return the new upper node."""
        bs = self.bs
        mid = RadixNode(parent=child.parent, key=child.key[:nb * bs],
                        blocks=child.blocks[:nb])
        mid.refcount = child.refcount
        mid.last_access = child.last_access
        mid.block_states = child.block_states[:nb]
        mid.sigs = child.sigs[:nb] if child.sigs is not None else None
        child.parent.children[self._edge_key(mid)] = mid
        child.parent = mid
        child.key = child.key[nb * bs:]
        child.blocks = child.blocks[nb:]
        child.block_states = child.block_states[nb:]
        if child.sigs is not None:
            child.sigs = child.sigs[nb:]
            if all(s == (TEXT_MEDIA_ID,) for s in child.sigs):
                child.sigs = None
        mid.children[self._edge_key(child)] = child
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
            del leaf.parent.children[self._edge_key(leaf)]
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
