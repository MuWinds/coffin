"""Radix-tree prefix cache (RadixAttention) over the paged KV blocks.

Module: ``coffin/`` (top level, model-agnostic). Keys the blocks that
``paged_cache`` hands out; ``block_manager`` owns both and is the object callers
go through, while ``prefix_engine`` (single-stream) and ``model_runner``
(batched) are the two paths that match and insert. This module may not import a
model family: everything it needs about one is the state snapshot it is handed
and the image spans on an ``MMInput``.

Every *block-aligned* token prefix that has been seen is stored in a radix tree
together with the physical blocks holding its KV. A new request takes the
longest cached prefix (``refcount++`` on those blocks) and only prefills the
suffix, so a shared system prompt is computed once instead of once per request.

Constraints:
    * Only whole blocks are stored and matched. KV is shared at block
      granularity exactly like vLLM's automatic prefix caching and SGLang's
      RadixCache, and children are keyed by the full first block, so two
      prefixes that diverge inside their first block stay distinct edges. A
      half-block hit would leave the block table holding a partial block that
      copy-on-write cannot refcount.
    * On the hybrid model a KV-only hit is silently wrong, so the match is
      truncated to the deepest boundary that carries a recurrent-state snapshot
      (see *Hybrid models* below). Dropping a snapshot costs reuse, never
      correctness.
    * Block keys carry the media signature of the block (see *Media salt*
      below), and an edge inserted without media information may only be reused
      for blocks proven image-free. Getting this wrong hands one image's KV to
      a different image.
    * No locking. ``llm_engine``'s step loop is the only writer, and it drives
      both callers in turn; ``refcount`` counts live sequences, not blocks.

Hybrid models:
Qwen3.5 interleaves 8 *full attention* layers (paged KV, shareable) with 24
*linear attention* layers whose state is a function of the whole prefix. Reusing
a KV prefix is therefore only correct if the matching recurrent state is
restored too -- the KV alone would pair the shared keys with a state that never
saw them. Each node carries ``block_states``: an optional snapshot of the
linear-attention conv/recurrent states for every block along its edge, so a
match can be replayed from any block boundary that has one.
``match_prefix(..., require_states=True)`` truncates the match to the deepest
boundary that carries a snapshot, and the caller restores that state.

Snapshots are big (24 layers x [32,128,128] fp32 ~ 50 MB), so a snapshot budget
bounds them: the node end of every stored prefix gets one, and intermediate
block boundaries only while the budget lasts. A prefix that was inserted without
a snapshot still matches -- the caller drops the unmatched blocks and recomputes
them, never resumes from a state it could not restore.

Media salt (images):
Token ids are not the whole story once the prompt can carry an image: the
``<|image_pad|>`` placeholder run is identical for *every* image, while the KV
computed for it depends on the pixels. A cache keyed on token ids alone would
hand a second, different image the first image's KV -- silently wrong output
that grows worse the longer the prompt is.

Every block therefore also carries a *media signature*: the set of media ids of
the tokens in it, where a token covered by image ``g`` gets a stable int derived
from that image's content digest (``MMInput.digest``) and its index in the
request, and every other token gets :data:`TEXT_MEDIA_ID` (0). A text block's
signature is ``(0,)``, so its child key is the plain token tuple -- the
text-only path keeps exactly the old key, the old behaviour and the old cost:
the signature work is skipped entirely when no ``mm`` is passed. An image
block's key is ``(tokens, signature)``, so two prompts with different images
diverge at the first block that contains image tokens.

Salting only the image blocks is enough, because a radix path is a *path*:
blocks after the image have signature ``(0,)`` again (no image token in them),
but the only way to reach them is through the image block that already
diverged. This is the same argument vLLM makes for hashing the mm hash into the
prefix hash, except that only the blocks that actually contain mm tokens need
it. An id is also salted with the image's index in the request, so the same
picture appearing twice never claims to be the same KV as itself in another
position.

One asymmetry is the difference between a working salt and a silent leak: an
edge inserted *without* media information (``mm=None``, i.e. the text-only path
or any caller that predates this) has no signatures, and "no signatures" must
not mean "reusable by anything". A block inside such an edge may have been
computed for an image nobody recorded, so a query that *does* carry media may
only adopt the blocks it can prove are image-free. See
:meth:`RadixCache._sig_ok`.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Optional

from .paged_cache import PagedKVCache
from utils.mm_input import IMAGE_TOKEN_ID

#: A linear-attention snapshot: model layer index -> (recurrent, conv) state.
LinearStates = dict[int, tuple[Any, Any]]

#: Media id of a token no image covers (also the whole signature of a text block).
TEXT_MEDIA_ID = 0


def media_id(digest: str, index: int) -> int:
    """Stable positive int64 for image ``index`` of a request with ``digest``.

    Stable across requests (same bytes + same position -> same id, so the second
    turn of a conversation still hits), distinct per image content, and never 0
    (0 is reserved for text so a text block's signature stays ``(0,)``).

    Args:
        digest: Content digest of the image (``MMInput.digest``), so two
            different images never share an id.
        index: The image's position among the request's images. Salting with it
            keeps the same picture at a different position from claiming the
            same KV.

    Returns:
        An int64-range positive id, with the top bit cleared so it stays inside
        int64 and the low bit forced to 1 so it can never collide with
        :data:`TEXT_MEDIA_ID`.
    """
    h = hashlib.sha256(f"{digest}#{index}".encode("utf-8")).digest()
    return (int.from_bytes(h[:8], "little") & ((1 << 63) - 1)) | 1


def media_ids_for(mm, num_tokens: int) -> Optional[list[int]]:
    """Per-token media id for the first ``num_tokens`` tokens of a prompt.

    ``None`` means "no media information": the caller has a text-only prompt (or
    no ``MMInput`` at all) and the tree keeps its plain token keys. The result
    aligns with the *absolute* prompt index, so a caller that matches
    ``tokens[:-1]`` still passes the prefix of the very same list.

    Args:
        mm: The request's ``mm_input.MMInput``, or ``None`` for a text request.
            Only ``spans`` and ``digest`` are read.
        num_tokens: Prefix length to produce ids for. Spans are clipped to it,
            so the match over ``tokens[:-1]`` sees the same ids as the insert
            over the full prompt.

    Returns:
        One media id per token in prompt order, or ``None`` when the prompt
        carries no media information at all.

    Raises:
        ValueError: ``mm`` has image spans but an empty digest. Refusing is the
            safe failure: without a content digest two different images of the
            same size would share block keys and the cache would return one
            image's KV for the other. The runtime service always fills it in
            (``mm_input.image_digest``), so this is a caller bug.
    """
    if mm is None:
        return None
    spans = getattr(mm, "spans", None)
    if not spans:
        return None
    digest = getattr(mm, "digest", "") or ""
    if not digest:
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
    """Signature of the block at token ``start``: the sorted set of media ids.

    A set, not a sequence: a block's reuse depends on which images it contains,
    not on where inside the block each one starts, so two prompts that pad their
    placeholders differently still hit the same block.

    Args:
        media: Per-token media ids from :func:`media_ids_for`, or ``None`` for a
            request with no media information.
        start: Absolute token the block begins at.
        bs: Block size in tokens; the signature covers ``[start, start+bs)``.

    Returns:
        ``(0,)`` for a block no image covers, otherwise the sorted media ids it
        contains.
    """
    if media is None:
        return (TEXT_MEDIA_ID,)
    ids = sorted(set(media[start:start + bs]))
    return tuple(ids) if ids else (TEXT_MEDIA_ID,)


class RadixNode:
    """One edge of the tree: the tokens it spans and the blocks holding their KV.

    A node's ``key`` is the whole token run from its parent down to it, so an
    edge can cover many blocks; ``blocks`` runs parallel to it, one entry per
    block, and so do ``block_states`` and (when the edge contains an image)
    ``sigs``. Splitting an edge preserves that parallel structure.
    """

    __slots__ = ("parent", "children", "key", "blocks", "refcount",
                 "last_access", "block_states", "sigs")

    def __init__(self, parent: Optional["RadixNode"] = None, key: tuple = (),
                 blocks: Optional[list[int]] = None) -> None:
        """Create an edge; the caller links it into ``parent.children``.

        Args:
            parent: Node this edge hangs off. ``None`` only for the root.
            key: Token ids along the edge. Its first block is what the parent
                keys the child by.
            blocks: Physical blocks holding that KV, one per block of ``key``.
                No reference is taken here; the inserting call does that.
        """
        self.parent = parent
        self.children: dict[tuple, "RadixNode"] = {}
        #: Token ids along this edge; ``len(key) == len(blocks) * bs``.
        self.key: tuple = tuple(key)
        self.blocks: list[int] = list(blocks or [])
        #: Live sequences parked on this node; the only thing that keeps an edge
        #: from being evicted.
        self.refcount = 0
        self.last_access = time.time()
        self.block_states: list[Optional[LinearStates]] = [None] * len(self.blocks)
        #: Per-block media signature. ``None`` means the edge holds no image
        #: token, so its child keys are plain token tuples and every request
        #: matches it.
        self.sigs: Optional[list[tuple[int, ...]]] = None

    @property
    def is_leaf(self) -> bool:
        """Whether no prefix continues past this node.

        Returns:
            ``True`` for a node with no children, which is the only kind
            eviction may reclaim.
        """
        return not self.children

    def num_states(self) -> int:
        """How many block boundaries on this edge still hold a snapshot.

        Returns:
            Snapshot count, what the tree-wide budget is charged against.
        """
        return sum(1 for s in self.block_states if s is not None)


class RadixCache:
    """Prefix cache over a :class:`~paged_cache.PagedKVCache`'s blocks."""

    def __init__(self, cache: PagedKVCache, max_state_snapshots: int = 8) -> None:
        """Build an empty tree over ``cache``'s blocks.

        Args:
            cache: The pool whose blocks are shared. The tree takes its own
                reference on every block it stores, so the pool must be the same
                object the engine allocates from.
            max_state_snapshots: How many recurrent-state snapshots the tree may
                keep. One snapshot is ~50 MB (24 layers of [32,128,128] fp32),
                so this is a VRAM budget, and ``0`` disables snapshots entirely
                (prefix reuse then only ever helps a text-only or non-hybrid
                model).
        """
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
        pays nothing for the salt. ``tokens`` is the *absolute* prompt prefix
        (the callers pass ``tokens[:-1]``), which is what makes the spans line up.

        Args:
            tokens: Prompt prefix to match. Shorter than one block means no hit
                at all; the walk only ever considers whole blocks.
            require_states: Truncate the match to the deepest block boundary
                holding a recurrent-state snapshot, and release the blocks past
                it. Mandatory on the hybrid model, where the KV alone would pair
                shared keys with a state that never saw them.
            mm: The request's image payload, or ``None`` for text only. Must be
                the same object :meth:`insert` will be given.

        Returns:
            ``(blocks, matched_len, states)``: the refcounted block list, the
            matched length in tokens (``0`` with an empty block list means no
            hit), and the snapshot at ``matched_len`` or ``None`` when
            ``require_states`` is off or no boundary had one. ``matched_len``
            is the authority -- ``blocks`` has been trimmed to match it.
        """
        self.total_query_tokens += len(tokens)
        bs = self.bs
        media = media_ids_for(mm, len(tokens))
        node, i = self.root, 0
        out: list[int] = []
        now = time.time()

        # Deepest block boundary on the walked path that has a snapshot; the
        # match is cut back to it once the walk ends, so it is tracked here
        # rather than by rewinding the tree.
        cp_len, cp_n, cp_states = 0, 0, None

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
                # Newest boundary first: the longest matching prefix is the one
                # worth keeping, and the linear-attention state at it is what
                # the caller will restore.
                for k in range(nb - 1, -1, -1):
                    if child.block_states[k] is not None:
                        cp_len = i + (k + 1) * self.bs
                        cp_n = len(out) - (nb - (k + 1))
                        cp_states = child.block_states[k]
                        break
            i += nb * self.bs
            if nb < m:
                # Partial edge: the prefix diverges inside this edge, so nothing
                # below it can match and the walk ends here.
                break
            child.refcount += 1
            child.last_access = now
            node = child

        if require_states and cp_n < len(out):
            # Release the blocks we cannot justify a state for: keeping them
            # would hand the caller KV whose recurrent state it cannot restore,
            # which is the silently-wrong-state case the snapshot exists to
            # prevent. Recomputing those blocks costs time, never correctness.
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
          text-only path, or a caller that predates ``mm``). Its blocks may well
          have been computed for an image we know nothing about, so the only
          branch that is provably safe is a block that holds no image token at
          all. Accepting it unconditionally is exactly the silent cross-image
          hit this module exists to prevent;
        * a salted edge is stored under a salted child key, so a plain-token
          lookup never reaches it in the first place.

        Args:
            sig: The query's signature for this block.
            node: Edge being considered.
            k: Index of the block within ``node``'s edge.
            block_tokens: The query's token ids for the block, checked against
                ``IMAGE_TOKEN_ID`` for an edge that carries no signatures.

        Returns:
            ``True`` only when reuse is provable; anything unknown is ``False``,
            because the cost of a false ``True`` is wrong output rather than a
            cache miss.
        """
        if node.sigs is None:
            return sig == (TEXT_MEDIA_ID,) and IMAGE_TOKEN_ID not in block_tokens
        return sig == node.sigs[k]

    def _child_key(self, tokens: list[int], media: Optional[list[int]], start: int
                   ) -> tuple:
        """Key of the child edge whose first block starts at ``start``.

        Args:
            tokens: Prompt tokens being matched or inserted.
            media: Per-token media ids, or ``None`` on the text-only path.
            start: Absolute token the child's first block begins at.

        Returns:
            The first block's token tuple, extended with its media signature
            only when the block contains an image token.
        """
        key = tuple(tokens[start:start + self.bs])
        if media is None:
            return key
        sig = media_signature(media, start, self.bs)
        # A block with no image token keeps the *plain* key, whether or not the
        # request carried images: that is what lets a text prompt and an image
        # prompt share a text prefix (and keeps the text-only format unchanged).
        return key if sig == (TEXT_MEDIA_ID,) else key + sig

    def _edge_key(self, node: RadixNode) -> tuple:
        """The key ``node.parent.children`` stores ``node`` under.

        ``_child_key``'s format applied to an existing node, so a lookup and the
        link that created the node always agree; a split re-derives both halves
        with this rather than remembering the original key.

        Args:
            node: Node being linked under its parent.

        Returns:
            Its first block's tokens, plus that block's signature when it holds
            an image token.
        """
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
        observed there. ``mm`` salts the block keys exactly like
        :meth:`match_prefix` does (pass the same ``MMInput`` to both, or the
        insert populates a branch no match will ever look at).

        Args:
            tokens: Tokens the prefix covers. A non-block-aligned tail is dropped
                before anything is stored, since no match could ever land there.
            blocks: Physical blocks holding that KV, starting at position 0 and
                parallel to ``tokens``. The tree increfs what it keeps.
            states_at: Absolute position -> snapshot observed there. Missing
                boundaries stay ``None`` and are not reusable as a hit point; a
                position that already holds a snapshot is not overwritten, so
                the first observation wins.
            mm: Media payload to salt the keys with; must match what
                :meth:`match_prefix` will be called with.
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
                # Edge fully consumed and more tokens remain: descend. Its
                # refcount is not touched -- insert stores prefixes, it does not
                # claim that a sequence is parked on them.
                i += m * self.bs
                node = child
                continue
            if nb == m:
                # Tokens end exactly at the child: nothing new to store, but the
                # snapshots the caller captured here may still fill gaps.
                self._adopt_states(child, i, nb, states_at)
                return
            mid = self._split(child, nb)
            rest_tok = tokens[i + nb * self.bs:]
            rest_blk = blocks[i // self.bs + nb:]
            if rest_tok:
                self._new_leaf(mid, rest_tok, rest_blk, (i + nb * self.bs) // self.bs,
                               states_at, media, i + nb * self.bs)
            return

    def _adopt_states(self, node: RadixNode, i: int, nb: int,
                      states_at: dict[int, LinearStates]) -> None:
        """Fill in snapshots an existing edge is missing, without overwriting.

        Args:
            node: Edge whose prefix the inserted tokens ended on.
            i: Absolute token position the edge starts at.
            nb: Blocks of the edge that the insert covered.
            states_at: Absolute position -> snapshot. Positions the node already
                has are skipped: a stale re-observation must not replace the
                state that a live sequence is restoring.
        """
        for k in range(nb):
            pos = i + (k + 1) * self.bs
            if node.block_states[k] is None and pos in states_at:
                node.block_states[k] = states_at[pos]
                self._trim_states()

    def _new_leaf(self, parent: RadixNode, key: list[int], blocks: list[int],
                  start_block: int, states_at: dict[int, LinearStates],
                  media: Optional[list[int]] = None, start_token: int = 0) -> RadixNode:
        """Create an edge for a prefix nothing in the tree covers yet.

        Args:
            parent: Node the new edge hangs off.
            key: Tokens along the edge, block-aligned.
            blocks: Blocks holding that KV; increfed here, since the tree now
                owns a reference.
            start_block: Index of ``blocks[0]`` within the inserted prefix, used
                to look the tail's snapshots up by absolute position.
            states_at: Absolute position -> snapshot.
            media: Per-token media ids of the inserted prefix, or ``None``.
            start_token: Absolute token the edge begins at, so the signatures
                are taken from the right part of ``media``.

        Returns:
            The new node, already linked under ``parent``.
        """
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

        Args:
            media: Per-token media ids of the prefix, or ``None``.
            start_token: Absolute token the edge starts at.
            n_blocks: Blocks along the edge.

        Returns:
            One signature per block, or ``None`` when no block holds an image
            token -- the representation that lets an edge be matched by callers
            with no media information.
        """
        if media is None:
            return None
        sigs = [media_signature(media, start_token + k * self.bs, self.bs)
                for k in range(n_blocks)]
        return None if all(s == (TEXT_MEDIA_ID,) for s in sigs) else sigs

    def _split(self, child: RadixNode, nb: int) -> RadixNode:
        """Split ``child`` after ``nb`` blocks; return the new upper node.

        The insert's tokens diverge inside ``child``, so the shared head becomes
        its own node and the remaining tail stays under it. Every parallel list
        is cut at the same point, and refcount/last_access are inherited by the
        head, which is now the node a matching sequence parks on.

        Args:
            child: Edge being split.
            nb: Blocks kept in the upper node; at least 1, at most
                ``len(child.blocks) - 1``.

        Returns:
            The new upper node, now linked under ``child``'s former parent.
        """
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
            # A tail with no image token goes back to the ``None`` form so a
            # text-only request can still match it.
            if all(s == (TEXT_MEDIA_ID,) for s in child.sigs):
                child.sigs = None
        mid.children[self._edge_key(child)] = child
        return mid

    def _trim_states(self) -> None:
        """Drop the LRU snapshots past the budget (blocks themselves are kept).

        Each snapshot is ~50 MB, so the budget is a VRAM cap and has to bite on
        an insert rather than on a read. Dropping one lowers the reuse a later
        match can reach, never its correctness: a boundary without a snapshot is
        no longer a hit point. The oldest access goes first because a
        conversation grows forward, so the oldest boundaries are the ones it is
        least likely to return to.
        """
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
        """The least-recently-used leaf whose blocks nobody else holds.

        A leaf is reclaimable only when no sequence is parked on it
        (``refcount == 0``) and no other reference survives in the pool
        (``cache.refcount[b] == 1``, the tree's own). Evicting a block still
        shared with a sequence would corrupt that sequence's KV, so those leaves
        are skipped even though they are leaves.

        Returns:
            The eviction candidate, or ``None`` when the whole tree is pinned.
        """
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
        """Free LRU, unshared leaves until ``blocks_needed`` blocks are free.

        Args:
            blocks_needed: Free blocks the next allocation wants.

        Returns:
            How many blocks were actually freed. Fewer than requested means the
            tree has nothing left that is safely reclaimable, so the caller's
            allocation is about to fail or has to wait for a sequence to retire
            -- it must not treat this as success.
        """
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
        """Make room for an allocation, evicting cached prefixes if needed.

        Called before the allocation, never after: once the request itself holds
        blocks, the leaves it would have to give up are no longer evictable.

        Args:
            blocks_needed: Free blocks the next prefill or decode step requires.
        """
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
        """Blocks the tree is holding on to, shared or not.

        Returns:
            Total block count across every edge; this is the part of the pool
            that eviction can reclaim when no sequence references it.
        """
        n = 0
        stack = [self.root]
        while stack:
            node = stack.pop()
            n += len(node.blocks)
            stack.extend(node.children.values())
        return n

    def num_snapshots(self) -> int:
        """Resident recurrent-state snapshots, the quantity the budget caps.

        Returns:
            Snapshot count across the tree, each one ~50 MB of host memory.
        """
        n = 0
        stack = [self.root]
        while stack:
            node = stack.pop()
            n += node.num_states()
            stack.extend(node.children.values())
        return n

    def snapshot_bytes(self) -> int:
        """Host-side estimate of the resident snapshot memory (rough).

        The per-snapshot figure is a fixed 24 linear-attention layers of
        [32,128,128] fp32 recurrent state plus 8192x3 fp32 conv state; it is
        derived from the Qwen3.5 geometry, so another family's hybrid model
        would need its own constant here. Reported to humans watching the 16 GB
        budget, never used to size anything.

        Returns:
            Estimated bytes, rounded down to the snapshot count's precision.
        """
        return self.num_snapshots() * 24 * (32 * 128 * 128 * 4 + 8192 * 3 * 4)

    def depth(self) -> int:
        """Length of the longest stored prefix chain, in edges.

        Returns:
            Edge count from the root to the deepest node: a diagnostic for how
            far reuse actually reaches, not a bound on anything.
        """
        best = 0
        stack = [(self.root, 0)]
        while stack:
            node, d = stack.pop()
            best = max(best, d)
            stack.extend((c, d + 1) for c in node.children.values())
        return best

    @property
    def hit_rate(self) -> float:
        """Fraction of queried tokens served from the tree so far.

        Returns:
            ``hit_tokens / total_query_tokens``, or ``0.0`` before the first
            query -- a ratio, so it is only meaningful over a workload, never
            per request.
        """
        if self.total_query_tokens == 0:
            return 0.0
        return self.hit_tokens / self.total_query_tokens
