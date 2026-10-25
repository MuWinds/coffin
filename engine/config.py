"""Engine configuration: every knob the server, scheduler and runner share.

Module: ``coffin/`` (top level, imports nothing else from the engine). One
``EngineConfig`` is built from CLI flags or by a test and passed down to
``llm_engine`` / ``scheduler`` / ``model_runner``, so none of them has to know
how the server was invoked. ``entrypoints/cli/runtime.py`` is the only place
that parses runtime flags.

Constraints:
    * Fields are read once at construction by the layers they configure; a few
      (``num_blocks``) are resolved in ``__post_init__``. Mutating a field after
      the engine is built will not re-provision anything that was sized from it.
    * Defaults describe the 16 GB BI-V100 target. Raising ``max_seq``,
      ``num_blocks`` or ``vision`` changes resident VRAM and must be checked
      against that budget.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class EngineConfig:
    """Static configuration of one loaded engine.

    The defaults are the ones the benchmark numbers in ``NOTES.md`` were
    measured with; anything tuned for a different card belongs in the CLI layer
    of ``entrypoints/cli/runtime.py`` rather than here.
    """

    model_dir: str
    #: Context window. The KV pool and the rope tables are built for it up front,
    #: so this is a VRAM commitment, not a limit checked at runtime.
    max_seq: int = 262144
    #: Tuner flag of the model: measure a handful of kernel variants at load time
    #: and keep the fastest. Costs a few seconds of start-up, so tests turn it off.
    bench: bool = True
    #: Keep the 2D weights in int8 (4 GiB resident) instead of fp16 (8 GiB).
    #: On a 16 GB card this is the difference between fitting the pool or not.
    quant: bool = True
    #: Serve full-attention layers from the shared paged int8 pool.
    prefix_cache: bool = True
    #: Tokens per KV block. Must divide ``max_seq`` cleanly enough for the radix
    #: tree to land on block boundaries; 64 matches the snapshot interval floor.
    block_size: int = 64
    #: Physical blocks in the pool; ``-1`` means "one sequence's worth", resolved
    #: in ``__post_init__`` from ``max_seq`` and ``block_size``.
    num_blocks: int = -1
    #: How many linear-attention recurrent-state snapshots the radix tree may
    #: keep. A snapshot is ~50 MB, so this is the cache's VRAM budget in disguise.
    snapshots: int = 8
    #: Snapshot every N tokens (rounded up to a whole block).
    snapshot_interval: int = 512
    #: Drafts per speculative step (0 = off). Needs the MTP head in the checkpoint.
    spec_k: int = 0
    #: Allow speculation in the *batched* engine. Off by default: the batched
    #: verify costs one weight pass per candidate token, so on this card it is
    #: slower than a plain batched decode step (see NOTES.md) even though the
    #: output is identical. Single-stream spec ignores this flag.
    spec_batch: bool = False
    #: Approximate KV selection for the batched verify: keep the top-k historical
    #: KV blocks per row (picked from the drafter's lookahead and the target's last
    #: attention) plus a recent window, instead of reading the whole context.
    #: 0 = exact full attention. Implies ``spec_batch``.
    spec_sparse_k: int = 0
    #: Only speculate in the batched engine once a sequence has at least this many
    #: cached tokens: below it the greedy step is cheaper (the batched verify pays
    #: an extra weight pass), above it the shared-KV verify wins. 0 = always.
    spec_min_ctx: int = 0
    #: Prefill GEMM M. rocBLAS tops out around 2048 tokens on this card.
    chunk: int = 2048
    #: Sequences decoded in one step. The scheduler never admits more than it can
    #: serve, and row ``capacity-1`` stays reserved for a prefill.
    max_num_seqs: int = 1
    #: Token budget of one scheduling step (prefill chunking + decode tokens).
    max_num_batched_tokens: int = 2048
    #: Hard wall-clock cap of one generation, seconds (0 = none).
    max_seconds: float = 0.0

    # Multimodal (image) serving.
    #: Serve images at all. ``False`` (``--no-vision``) keeps the vision tower
    #: unloaded and answers an image request with a readable 503 instead of the
    #: silent garbage a text-only prefill of image placeholders would produce.
    vision: bool = True
    #: Keep the tower's 2-D weights int8 (+ per-row fp32 scale) instead of fp16:
    #: ~0.31 GiB resident instead of ~0.62 GiB, at a small per-request cost.
    vision_int8: bool = False
    #: The only real VRAM/latency knob: the processor downsamples anything larger.
    #: 1 MP is ~1024 image tokens -- about the same prefill cost as 1k text tokens.
    image_max_pixels: int = 1048576
    #: Official floor: a smaller image is *upscaled* to this, so an image never
    #: costs fewer than 64 tokens.
    image_min_pixels: int = 65536
    #: Images per request (four is 4k image tokens, already a large prefill).
    max_images: int = 4
    #: Decoded bytes of one image (a base64 part is 4/3 of that in the request).
    max_image_bytes: int = 8388608
    #: Base64 characters of all image parts of one request (before decoding).
    max_image_total_bytes: int = 16777216

    def __post_init__(self) -> None:
        """Resolve sentinel defaults into concrete sizes.

        ``num_blocks = -1`` means "one sequence's worth", which is all the
        single-stream engine needs. Batched serving wants a pool shared by
        several sequences, so the field stays explicit rather than being derived
        at the allocation site.
        """
        if self.num_blocks < 0:
            self.num_blocks = (self.max_seq + self.block_size - 1) // self.block_size
