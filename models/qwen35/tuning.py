"""Tuning knobs measured on this model and this part.

The defaults are the best points the probes named next to each constant found;
the env overrides exist so a run can A/B something without editing the engine.

Module: ``coffin/models/qwen35``. Leaf module of the family -- imports ``os`` and
``backends`` -- read by ``blocks``, ``full_attn``, ``linear_attn`` and ``mtp`` when
they size their kernels and buffers.

Constraints:
    * Nothing here may carry a *backend* number. ``ATTN_SPLIT`` used to live in this
      file with a comment saying it must equal the constant in the attention
      kernel, which was the backend ABI leaking into the model family: on a
      backend that does not split at all, the caller would still size its partial
      buffers for 8 and the kernel would write past them. It is now read from
      ``caps.attn_split`` and re-exported under the old name, so the buffer sizes
      follow whatever the selected backend actually does.
    * ``CKPT`` sizes the speculative rollback state and bounds the verify width,
      so it must agree with the verify graph captured in ``model.py``.
    * These are measured values, not free parameters; re-run the named probe
      before changing a default.
"""

from __future__ import annotations

import os

from backends import caps as _caps

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
#: ``[nkv, rep, nq, PREFILL_ATTN_KB]`` whatever the context, at the cost of one
#: rescale of the accumulator per block.
#:
#: Raised from 1024 to 4096 after measuring it on the BI-V100. A bigger block means fewer
#: online-softmax rescales, and prefill there is compute-bound at 45-51 TFLOP/s, so the
#: rescale is the cost while the extra memory is affordable: at the 512-token tile used in
#: practice the live score block grows 128 MB, against a 32 GB card and ~4.5 GB of int8
#: weights. At n=8192 each step up paid -- 512 -> 2048 was 10.5% faster, 2048 -> 4096 a
#: further 4.9% -- and the direction held at every tile width and in two independent
#: sweeps (``diag/corex/probe_prefill_sweep.py``, PERF_COREX.md §13.7). A 2048-token
#: prompt gains nothing (720.0 -> 720.2 ms) because it is a single tile, so the block loop
#: barely runs there; the win is on long prompts.
PREFILL_ATTN_KB = int(os.environ.get("PREFILL_ATTN_KB", "4096"))

#: Tokens per chunk in ``MTP.prefill``. Its attention is the same blocked one as
#: the main layers, but it is fed the whole prompt (to keep the draft cache in
#: step), so it tiles and chunks its own input instead of relying on the caller's
#: prefill chunk -- 2048 keeps its score blocks at ~134 MB whatever the prompt.
PREFILL_MTP_CHUNK = int(os.environ.get("PREFILL_MTP_CHUNK", "2048"))

#: Key-range splits per query head in the decode attention kernel. This is a
#: kernel ABI, not a tuning choice, so it is the *backend's* value: this side owns
#: the ``part_acc``/``part_ml`` buffers the kernel writes, and the width has to be
#: the one the kernel indexes them with. Read from ``caps.attn_split``; kept under
#: the old name so every buffer-sizing call site stayed unchanged.
ATTN_SPLIT = _caps().attn_split
