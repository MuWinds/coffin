"""Resident inference runtime.

Module: ``coffin.serving``. This class provisions the model and delegates
lifecycle, generation and chat behavior to focused mixins.

Constraints:
    * The model thread remains the only writer of engine state.
    * The socket protocol stays line-oriented and unchanged.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from pathlib import Path

import torch

import engine_api
from engine import llm_engine
from engine.config import EngineConfig
from engine.sequence import SamplingParams

from .runtime_parts import (
    RuntimeChatMixin,
    RuntimeGenerationMixin,
    RuntimeLifecycleMixin,
)
from .runtime_support import MAX_SEQ, MODEL, MMError, SOCK, _eos_ids


class Server(
    RuntimeLifecycleMixin,
    RuntimeGenerationMixin,
    RuntimeChatMixin,
):
        def __init__(self, model_dir: str, max_seq: int, bench: bool, spec_k: int = 0,
                     int8: bool = True, prefix_cache: bool = True, block_size: int = 64,
                     snapshots: int = 8, snapshot_interval: int = 512,
                     max_seconds: float = 0.0, max_num_seqs: int = 1,
                     num_blocks: int = 0, spec_batch: bool = False,
                     spec_sparse_k: int = 0, spec_min_ctx: int = 0,
                     fast_gemv: bool = False, vision: bool = True,
                     vision_int8: bool = False, image_max_pixels: int = 1048576,
                     image_min_pixels: int = 65536, max_images: int = 4,
                     max_image_bytes: int = 8388608) -> None:
            """Load the checkpoint, freeze ``EngineConfig`` and build the engine.

            Args:
                model_dir: Checkpoint directory; its ``config.json`` picks the
                    architecture and its tokenizer renders the chat template.
                max_seq: Context window shared by prompt and output. The KV cache is
                    allocated up front from this, so it is the dominant VRAM term.
                bench: Keep the legacy single-stream bench path available.
                spec_k: Drafts per speculative step, 0 to disable. Silently forced to
                    0 when the prefix cache is on, because a snapshot does not carry
                    the drafter's cache.
                int8: Quantise the 2D weights to int8. The default because the 256k
                    window and fp16 weights do not both fit in 16 GB.
                prefix_cache: Enable the paged radix prefix cache.
                block_size: KV block granularity in tokens; prefix reuse works at
                    whole-block boundaries only.
                snapshots: Recurrent-state snapshot budget, ~50 MB each.
                snapshot_interval: Snapshot every N tokens, rounded to a block.
                max_seconds: Wall-clock cap on one generation, 0 for none.
                max_num_seqs: Rows in a batched decode step; 1 keeps the original
                    single-stream engine.
                num_blocks: KV pool size in blocks, 0 for
                    ``max_seq / block_size * max_num_seqs``.
                spec_batch: Speculate in the batched engine. Opt-in because its
                    verify costs one weight pass per candidate token.
                spec_sparse_k: Approximate batched verify: keep the top-k historical
                    KV blocks per row. Implies ``spec_batch``.
                spec_min_ctx: Only speculate in the batched engine past this many
                    cached tokens, 0 to always.
                fast_gemv: Use the packed-fp16 projection kernels, forwarded through
                    ``Q35_FAST_GEMV`` because the choice must precede the model
                    module import.
                vision: Load the checkpoint's vision tower when it has one.
                vision_int8: Keep the tower's 2D weights int8, dequantised one GEMM
                    at a time per request.
                image_max_pixels: Largest image the processor serves; the real
                    VRAM/latency knob, anything larger is downsampled.
                image_min_pixels: Smallest image after the processor's upscale, so an
                    image never costs fewer than 64 tokens.
                max_images: Images accepted in one request.
                max_image_bytes: Decoded bytes allowed per image.

            Raises:
                RuntimeError: The checkpoint's ``config.json`` names no registered
                    architecture, or the weights do not load. Both are startup
                    failures: fix the ``--model`` path, do not retry in a loop.
            """
            from transformers import AutoTokenizer

            self.model_dir = model_dir
            # what this server is serving, for the startup log and ``ping``
            self.info = engine_api.describe(model_dir)
            self.max_seq = max_seq
            self.bench = bench
            self.spec_k = spec_k
            self.int8 = int8
            self.prefix_cache = prefix_cache
            self.block_size = block_size
            self.snapshots = snapshots
            self.snapshot_interval = snapshot_interval
            # hard wall-clock cap per generation (0 = none); a request may lower it
            self.max_seconds = float(max_seconds)
            self.pc = None
            self.n_req = 0
            self.n_reload = 0

            # images
            self.vision = bool(vision)
            self.vision_int8 = bool(vision_int8)
            self.image_max_pixels = int(image_max_pixels)
            self.image_min_pixels = int(image_min_pixels)
            self.max_images = int(max_images)
            self.max_image_bytes = int(max_image_bytes)
            #: The checkpoint's own processor (``Qwen3VLProcessor``: image processor +
            #: tokenizer). Loaded lazily on the first image request, not at startup:
            #: it costs ~2 s and text-only serving must not pay it -- and a text-only
            #: container that never sees an image never needs transformers' image
            #: stack at all. ``_processor_failed`` remembers a load failure so a
            #: broken checkpoint does not retry (and stall) on every request.
            self.processor = None
            self._processor_failed: str | None = None
            self.n_image_req = 0
            #: One engine thread drives every forward pass; connections submit
            #: requests and drain a per-request queue, so several can be in flight at
            #: once and the scheduler batches their decode steps.
            self._lock = threading.RLock()
            self._cv = threading.Condition(self._lock)
            self._outputs: dict[int, "queue.Queue"] = {}
            self._stop = False
            self._paused = False
            self._loop_thread: threading.Thread | None = None
            batched = max(1, int(max_num_seqs)) > 1
            #: Batched speculation is opted into separately: its verify costs one
            #: weight pass per candidate token, so it is slower than a plain batched
            #: decode step on this card even though the output is identical.
            self.spec_batch = bool((spec_batch or spec_sparse_k > 0) and batched)
            self.spec_sparse_k = int(spec_sparse_k) if self.spec_batch else 0
            self.spec_min_ctx = int(spec_min_ctx)
            #: Opt-in packed-fp16 projection kernels (models/qwen35/batch.py). The
            #: choice has to happen before the model module is imported, so it is
            #: passed through the environment.
            self.fast_gemv = bool(fast_gemv)
            if self.fast_gemv:
                os.environ["Q35_FAST_GEMV"] = "1"
            if prefix_cache and spec_k and not self.spec_batch:
                # A prefix snapshot does not carry the MTP drafter's cache, so a hit
                # would leave the drafter with a stale context (the output stays
                # exact, the acceptance rate drops). Rather than silently degrade,
                # spec stays off while the prefix cache is on.
                print("[serve] --spec-k is ignored while the prefix cache is on")
                self.spec_k = 0
            if spec_k and batched and not self.spec_batch:
                print("[serve] batched speculation is off; pass --spec-batch to enable it "
                      "(measured 0.79x plain batch decode on this card)")
            #: The single source of truth for engine knobs; the LLM engine, its
            #: scheduler and its runner all read from it.
            blocks = int(num_blocks) if num_blocks else (max_seq // block_size) * max(1, max_num_seqs)
            self.cfg = EngineConfig(
                model_dir=model_dir, max_seq=max_seq, bench=bench, quant=int8,
                prefix_cache=prefix_cache, block_size=block_size,
                num_blocks=blocks, snapshots=snapshots,
                snapshot_interval=snapshot_interval,
                spec_k=self.spec_k, spec_batch=self.spec_batch,
                spec_sparse_k=self.spec_sparse_k,
                spec_min_ctx=self.spec_min_ctx,
                max_seconds=self.max_seconds,
                max_num_seqs=max(1, int(max_num_seqs)))
            t0 = time.time()
            # stays resident; int8 halves it (4.0 GiB instead of 8.1), which is what
            # leaves room for a long-context KV cache on a 16 GB card
            self.weights = engine_api.load_weights(
                model_dir, quant=int8, **self._load_weight_kwargs())
            self.load_s = time.time() - t0
            self.tok = AutoTokenizer.from_pretrained(model_dir)
            self.eos = _eos_ids(self.tok)
            if self.cfg.spec_k and self.cfg.spec_batch:
                # Batched MTP adds ~1.25 GiB of rollback snapshots plus the drafter's
                # paged KV; refuse it up front rather than OOM inside capture.
                free, _total = torch.cuda.mem_get_info()
                if free < 1.5 * 2**30:
                    print(f"[serve] --spec-batch disabled: only {free / 2**30:.2f} GiB free "
                          f"(batched MTP needs ~1.5 GiB)")
                    self.cfg.spec_k = 0
                    self.cfg.spec_batch = False
            self._build()
