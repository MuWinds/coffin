"""Tuning knobs measured on this model and this part.

The defaults are the best points the probes named next to each constant found;
the env overrides exist so a run can A/B something without editing the engine.

Module: ``coffin/models/qwen35``. Leaf module of the family -- imports only
``os`` -- read by ``blocks``, ``full_attn``, ``linear_attn`` and ``mtp`` when they
size their kernels and buffers.

Constraints:
    * ``ATTN_SPLIT`` must equal ``ATTN_SPLIT`` in ``kernels/attn.cu``: this side
      owns the partial buffers the kernel writes, so a mismatch corrupts attention
      instead of failing a shape check.
    * ``CKPT`` sizes the speculative rollback state and bounds the verify width,
      so it must agree with the verify graph captured in ``model.py``.
    * These are measured values, not free parameters; re-run the named probe
      before changing a default.
"""

from __future__ import annotations

import os


#: Verified tokens per speculative step; the verify graph is captured for it.
CKPT = 8

#: Query-tile width for causal attention in prefill. Each tile reads keys only up
#: to its own end, so the upper triangle of the score block is never built: at
#: pos=0 that is half the score/softmax/PV work, at pos=2048 a quarter. Smaller
#: tiles save more work but add launches (the whole attention is 8 layers of 2
#: einsums); 512 measured best on this part (diag/probe_prefill_split.py).
PREFILL_ATTN_TILE = int(os.environ.get("PREFILL_ATTN_TILE", "512"))

#: Keys per streamed block inside a query tile. The score block used to be
#: ``[nkv, rep, nq, p1]`` -- O(context length) memory for every tile, which is
#: 8.6 GB at 256k and cannot coexist with the weights on a 16 GB card. Blocks are
#: accumulated with a running max (online softmax), so the block stays
#: ``[nkv, rep, nq, PREFILL_ATTN_KB]`` (~34 MB at the defaults) whatever the
#: context, at the cost of one rescale of the accumulator per block.
PREFILL_ATTN_KB = int(os.environ.get("PREFILL_ATTN_KB", "1024"))

#: Tokens per chunk in ``MTP.prefill``. Its attention is the same blocked one as
#: the main layers, but it is fed the whole prompt (to keep the draft cache in
#: step), so it tiles and chunks its own input instead of relying on the caller's
#: prefill chunk -- 2048 keeps its score blocks at ~134 MB whatever the prompt.
PREFILL_MTP_CHUNK = int(os.environ.get("PREFILL_MTP_CHUNK", "2048"))

#: Key-range splits per query head in the decode attention kernel; must match
#: ``ATTN_SPLIT`` in ``kernels/attn.cu`` (this side owns the partial buffers).
ATTN_SPLIT = int(os.environ.get("ATTN_SPLIT", "8"))
