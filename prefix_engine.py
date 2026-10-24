"""Radix-cached serving on top of the paged KV pool.

Ties the three pieces together for one request:

    match the prefix cache -> adopt its blocks and restore the recurrent state
    -> prefill only the unseen suffix -> insert the new prefix into the tree
    -> decode -> release the sequence's reference on the blocks

What this buys: a prefix that has been seen before -- a system prompt, a tool
preamble, a few-shot header, a regenerated answer -- is computed once instead of
once per request. The reusable unit is the *block-aligned* prefix, so with
``block_size=64`` a 512-token system prompt turns the next request's prefill into
a one-block suffix.

Single-stream by construction: the decode step is a CUDA graph over one token and
one slot map, so the engine serves one sequence at a time and the pool's slot map
belongs to the active sequence. Batched multi-sequence decode would need a
different step (and a different graph), which is a separate piece of work.

Speculative decoding composes with this, but the MTP drafter's own KV cache is
not part of a prefix snapshot: after a prefix hit the drafter sees a stale
context, so its drafts get worse (acceptance drops) while the *output* stays
exactly greedy, because every draft is verified by the main model. Restoring the
MTP cache at block boundaries would fix the acceptance rate and is a small
follow-up (it is 4 MB per boundary against the ~50 MB recurrent snapshot).

``prefill`` also requires at least one token to be recomputed: a prefix that
matches the entire prompt would leave nothing to run, and the logits of the last
token have to come from somewhere. The match is therefore taken over
``tokens[:-1]``, which bounds the reuse at "all but the last block" and is what
makes a full cache hit return valid logits.
"""

from __future__ import annotations

import time

import torch

from paged_cache import PagedKVCache, SeqKV
from block_manager import BlockManager
from radix_cache import RadixCache


class PrefixCacheEngine:
    """Sequential request serving with radix prefix reuse over a paged pool."""

    def __init__(self, model, num_blocks: int = 512, block_size: int = 64,
                 max_state_snapshots: int = 8, snapshot_interval: int = 512,
                 bm: BlockManager | None = None) -> None:
        self.model = model
        #: The pool and the radix tree are owned by the block manager; this engine
        #: only orchestrates one request's prefill/decode over them.
        self.bm = bm or BlockManager.attach(model, num_blocks, block_size, max_state_snapshots)
        self.pool: PagedKVCache = self.bm.pool
        self.radix: RadixCache = self.bm.radix
        #: Snapshot every this many tokens (rounded up to a whole block).  The
        #: reusable unit is a block-aligned prefix, but snapshotting *every* block
        #: boundary is wasteful twice over: each snapshot is ~50 MB (24 layers of
        #: [32,128,128] fp32), so all 256 boundaries of a 16k prompt would be
        #: 12.8 GB; and one ``model.prefill`` per 64 tokens pays the whole
        #: GEMM/TRSM/scheduling setup for a tiny M.  Boundary granularity only
        #: costs the tokens between the join point and the prompt, which for a
        #: conversation growing by a few hundred tokens per turn is nothing.
        self.snapshot_interval = snapshot_interval
        self.keep_states = max_state_snapshots
        self.seq: SeqKV | None = None
        self.last_matched = 0
        self.n_requests = 0
        self.n_prompt_tokens = 0
        self.n_hit_tokens = 0

    @torch.inference_mode()
    def prefill(self, tokens: list[int], mm=None) -> torch.Tensor:
        """Adopt a cached prefix, prefill the rest, insert the result.

        Returns exactly what a full prefill of ``tokens`` returns: the logits for
        the next token.

        ``mm`` (``mm_input.MMInput | None``) is the request's image payload.  It
        does two things: the prefix *match* is salted with the image digest, so a
        prompt with a different image cannot adopt this image's KV, and the
        suffix prefill gets it too, so the model splices the image embeddings in
        and uses the 3-D mRoPE positions.  ``matched`` still only ever lands on a
        block boundary; it may fall *inside* an image span, which is fine because
        the embeddings are sliced by absolute prompt index.
        """
        if not tokens:
            raise ValueError("empty prompt")
        model, pool, bm = self.model, self.pool, self.bm
        bs = pool.bs
        bm.reserve((len(tokens) + bs - 1) // bs + 1)         # evict LRU if needed

        blocks, matched, states = bm.match_prefix(tokens[:-1], require_states=True,
                                                  mm=mm)
        if self.seq is not None:
            self.seq.free()
        seq = SeqKV(pool, blocks, matched)
        self.seq = seq
        bm.assign(blocks, 0)
        if states is not None:
            model.restore_linear_states(states)
        else:
            model.reset()                                    # also clears pos1 + MTP
        model.pos1.fill_(matched)

        states_at: dict[int, dict] = {}
        if matched and states is not None:
            states_at[matched] = states                      # re-insertable as-is
        rest = tokens[matched:]
        seq.reserve(len(rest))
        logits = None
        # A snapshot is only reusable if it sits on a *block* boundary (that is the
        # unit the radix matches on), so the prompt is split at its last block
        # boundary: the aligned head is prefilled in `snapshot_interval` slices, and
        # each slice's end gets a snapshot; the tail is prefilled last and only its
        # logits matter.  Slicing wider than one block matters for speed (a 64-token
        # ``model.prefill`` pays the whole GEMM/TRSM/scheduling setup for a tiny M --
        # 16k tokens used to be 256 of those calls) and the live snapshot set is
        # capped at the radix budget, because each snapshot is ~50 MB (24 layers of
        # [32,128,128] fp32) and keeping every boundary of a 16k prompt would need
        # 12.8 GB.  The deepest boundaries are the ones a growing conversation hits,
        # so the oldest are dropped first.
        n_total = len(tokens)
        head_len = max(0, (n_total // bs) * bs - matched)
        step = max(bs, self.snapshot_interval // bs * bs)
        for c0 in range(0, head_len, step):
            c1 = min(c0 + step, head_len)
            logits = model.prefill(rest[c0:c1], mm=mm)
            states_at[matched + c1] = model.snapshot_linear_states()
            while len(states_at) > self.keep_states:
                droppable = [k for k in states_at if k != matched]
                if not droppable:
                    break
                del states_at[min(droppable)]
        if head_len < len(rest):
            logits = model.prefill(rest[head_len:], mm=mm)
        seq.commit_upto(model.pos1)

        self.last_matched = matched
        self.n_requests += 1
        self.n_prompt_tokens += len(tokens)
        self.n_hit_tokens += matched
        radix = self.radix
        radix.insert(tokens, seq.blocks, states_at, mm=mm)
        return logits

    @torch.inference_mode()
    def decode_iter(self, logits: torch.Tensor, max_new: int,
                    eos: set[int] | None = None, temperature: float = 0.0):
        """Yield the generated token ids one at a time (``decode`` in list form).

        Each step opens and closes its one-token write window first, which is what
        makes a partial tail block copy-on-write while it is still shared with the
        tree instead of overwriting a cached prefix.
        """
        eos = eos or set()
        model, seq = self.model, self.seq
        out = 0
        while out < max_new:
            seq.reserve(1)
            if temperature > 0:
                p = torch.softmax(logits.float() / temperature, dim=-1)
                nxt = int(torch.multinomial(p, 1))
            else:
                nxt = int(logits.argmax())
            if nxt in eos:
                break
            yield nxt
            out += 1
            logits = model.decode_step(nxt)
            seq.commit_upto(model.pos1)

    @torch.inference_mode()
    def decode(self, logits: torch.Tensor, max_new: int, eos: set[int] | None = None,
               spec_k: int = 0) -> list[int]:
        """Decode from the current sequence; returns the new token ids."""
        return list(self.decode_iter(logits, max_new, eos))

    @torch.inference_mode()
    def generate(self, tokens: list[int], max_new: int, eos: set[int] | None = None,
                 spec_k: int = 0, mm=None) -> dict:
        """One full request; returns the tokens and the prefill/decode split."""
        t0 = time.time()
        logits = self.prefill(tokens, mm=mm)
        t_pre = time.time() - t0
        t1 = time.time()
        out = self.decode(logits, max_new, eos, spec_k)
        return {"tokens": out, "prefill_s": t_pre, "decode_s": time.time() - t1,
                "n_prompt": len(tokens), "n_new": len(out), "matched": self.last_matched}

    def release(self) -> None:
        """Drop the active sequence's reference on its blocks."""
        if self.seq is not None:
            self.seq.free()
            self.seq = None

    def stats(self) -> dict:
        return {
            "requests": self.n_requests,
            "prompt_tokens": self.n_prompt_tokens,
            "hit_tokens": self.n_hit_tokens,
            "hit_rate": round(self.n_hit_tokens / self.n_prompt_tokens, 4)
            if self.n_prompt_tokens else 0.0,
            "blocks_used": self.pool.num_used, "blocks_total": self.pool.num_blocks,
            "cached_blocks": self.radix.num_cached_blocks(),
            "snapshots": self.radix.num_snapshots(),
            "snapshot_mib": round(self.radix.snapshot_bytes() / 2**20, 1),
            "cows": self.pool.n_cows, "evicted_blocks": self.radix.num_evicted_blocks,
        }
