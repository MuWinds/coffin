"""Engine configuration: the knobs that used to be scattered through ``serve.py``.

One object is built from the CLI flags (or by a test) and passed down to
:class:`~llm_engine.LLMEngine`, :class:`~scheduler.Scheduler` and
:class:`~model_runner.ModelRunner`, so that none of them has to know how the
server was invoked.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class EngineConfig:
    """Static configuration of one loaded engine.

    ``num_blocks`` defaults to the number of blocks a single sequence can fill
    (``max_seq / block_size``), which is what the single-stream engine needs;
    batched serving will want a pool shared by several sequences instead, so the
    field is explicit rather than derived at the allocation site.
    """

    model_dir: str
    #: Context window.  The KV pool and the rope tables are built for it up front.
    max_seq: int = 262144
    #: Tuner flag of the model: measure a handful of kernel variants at load time.
    bench: bool = True
    #: Keep the 2D weights in int8 (4 GiB resident) instead of fp16 (8 GiB).
    quant: bool = True
    #: Serve full-attention layers from the shared paged int8 pool.
    prefix_cache: bool = True
    #: Tokens per KV block.  Must divide ``max_seq`` cleanly enough for the radix
    #: tree to land on block boundaries; 64 matches the snapshot interval floor.
    block_size: int = 64
    #: Physical blocks in the pool; ``-1`` means "one sequence's worth".
    num_blocks: int = -1
    #: How many linear-attention recurrent-state snapshots the radix tree may keep.
    snapshots: int = 8
    #: Snapshot every N tokens (rounded up to a whole block).
    snapshot_interval: int = 512
    #: Drafts per speculative step (0 = off).  Needs the MTP head in the checkpoint.
    spec_k: int = 0
    #: Allow speculation in the *batched* engine.  Off by default: the batched
    #: verify costs one weight pass per candidate token, so on this card it is
    #: slower than a plain batched decode step (see NOTES.md) even though the
    #: output is identical.  Single-stream spec ignores this flag.
    spec_batch: bool = False
    #: Approximate KV selection for the batched verify: keep the top-k historical
    #: KV blocks per row (picked from the drafter's lookahead and the target's last
    #: attention) plus a recent window, instead of reading the whole context.
    #: 0 = exact full attention.  Implies ``spec_batch``.
    spec_sparse_k: int = 0
    #: Only speculate in the batched engine once a sequence has at least this many
    #: cached tokens: below it the greedy step is cheaper (the batched verify pays
    #: an extra weight pass), above it the shared-KV verify wins.  0 = always.
    spec_min_ctx: int = 0
    #: Prefill GEMM M.  rocBLAS tops out around 2048 tokens on this card.
    chunk: int = 2048
    #: Sequences decoded in one step.  Phase 0/1 of the batching work keeps this
    #: at one; the scheduler never admits more than it can serve.
    max_num_seqs: int = 1
    #: Token budget of one scheduling step (prefill chunking + decode tokens).
    max_num_batched_tokens: int = 2048
    #: Hard wall-clock cap of one generation, seconds (0 = none).
    max_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.num_blocks < 0:
            self.num_blocks = (self.max_seq + self.block_size - 1) // self.block_size
