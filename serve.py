"""Persistent inference server: keep the weights in VRAM and answer one JSON
line per request over a unix socket.

Module: ``coffin`` top level, above ``engine_api`` and ``llm_engine``. This file
owns the transport, the prompt rendering and the reply shape; everything below
it is reached only through the ``engine_api.Model`` and
``llm_engine.LLMEngine`` contract, so a new model family needs no edit here.

Constraints:
    * Absolute imports only (``import engine_api``). The entry script puts this
      directory on ``sys.path``, which makes ``ops`` / ``models`` top-level
      packages next to ``hip_ops``.
    * One engine thread drives every forward pass. Connection threads only
      submit requests and drain a per-request queue; anything that touches the
      model directly has to pause the loop first (see ``handle``).
    * The wire format is one JSON line per request and one per reply. The single
      exception is ``chat_stream``, which sends one line per token so
      ``openai_server.py`` can forward them as SSE frames.
    * Which model is served comes from the checkpoint's ``config.json`` through
      ``engine_api.create_model``; this file must not assume a family.
    * Weights are loaded once and rebuilt from the resident tensors, so an engine
      edit costs a ``reload`` (~seconds) instead of a 9 GB disk read (~1 min).
    * An image request is decoded, validated and processor-expanded before the
      prompt is prefilled, so a bad image is a clean rejection rather than a
      half-generated answer.

Server (defaults: 256k window, int8 weights, see NOTES §9):

    python serve.py --serve --model /root/private_data/models/Qwen3.5-4B

Client (instant once the server is up):

    python serve.py ping
    python serve.py reload
    python serve.py generate --prompt "The capital of France is" -n 32
    python serve.py chat --user "北京天气怎么样？" --tools tools_demo.json -n 256
    python serve.py stop

For an OpenAI-compatible HTTP endpoint (``/v1/chat/completions``, SSE streaming)
run ``openai_server.py`` next to it; it is a pure client of this socket.

Chat prompts are rendered with the checkpoint's own jinja template (via
``transformers``), so tool schemas and thinking mode match exactly.
"""

from __future__ import annotations

import argparse
import base64
import importlib
import io
import json
import os
import queue
import socket
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

import hip_ops
import engine_api
import llm_engine
from config import EngineConfig
from sequence import SamplingParams
from tool_call import parse_tool_calls, split_thinking

#: Deployment default. Any checkpoint whose config.json maps onto an
#: architecture registered in ``engine_api`` works here; override with --model or
#: $ENGINE_MODEL.
MODEL = os.environ.get("ENGINE_MODEL", "/root/private_data/models/Qwen3.5-4B")
#: Kept at the old path so the existing clients (openai_server.py, agent.py, the
#: probes) keep connecting; override with --sock or $ENGINE_SOCK to run a second
#: model next to the first.
SOCK = os.environ.get("ENGINE_SOCK", "/tmp/qwen35.sock")
#: The checkpoint's own limit (``max_position_embeddings``).  Only 8 of the 32
#: layers are full attention and their KV is int8, so a 256k window costs ~4.4 GB
#: of cache -- see the note in ``diag/probe_longctx.py``.
MAX_SEQ = 262144


def _eos_ids(tok) -> set[int]:
    """Every id this checkpoint can stop on, not just ``eos_token_id``.

    The chat template ends an assistant turn with ``<|im_end|>``, which is a
    different id from the model's ``eos_token_id`` on this checkpoint. Stopping
    on the model's own id alone would let a finished turn keep generating into
    the next one.

    Args:
        tok: The loaded tokenizer; its vocabulary decides which special tokens
            resolve.

    Returns:
        The stop ids, always containing ``eos_token_id``. A special token the
        vocabulary does not know resolves to a negative id and is dropped, since
        a sequence can never emit it.
    """
    ids = {tok.eos_token_id}
    for t in ("<|im_end|>", "<|endoftext|>"):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0:
            ids.add(i)
    return ids


# Image input: decode -> the checkpoint's own processor -> MMInput.
#
# Everything below runs once per image request, before the prompt is prefilled.
# The division of labour is the whole design:
#
# * PIL (this file) decodes and validates the bytes. It is the only place that
#   can bound *decode* memory: a 30k x 30k PNG is a valid PNG, and PIL would
#   expand it to 2.7 GB of RGB before the processor ever downsamples it.
# * ``Qwen3VLProcessor`` (transformers, already in the container) does the
#   smart_resize / patchify / merge-block flattening and expands the prompt's
#   ``<|image_pad|>`` markers to one token per merged visual patch. Re-implementing
#   that is the exact "values right, order wrong" risk VISION §3.2 warns about, so
#   the engine consumes its output verbatim.
# * ``mm_input`` turns the token spans and grids into the 3-D mRoPE positions and
#   the ``rope_delta`` the decode graph reads.

#: Hard ceiling on one image's *decoded* pixel count (not the served resolution):
#: bounds PIL's expand-to-RGB memory. 16 MP admits a 12 MP camera photo, which the
#: processor then downsamples to ``image_max_pixels``; above it the request is
#: refused before any large allocation.
MAX_DECODE_PIXELS = 16 * 1024 * 1024


class MMError(Exception):
    """An image request this server refuses, carrying the HTTP status to answer.

    The status travels with the exception because the refusal happens deep below
    the transport: the op handlers copy it into the reply, and
    ``openai_server.py`` maps it onto the HTTP status instead of calling every
    failure a 500.
    """

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = int(status)


def _decode_image(b64: str, mime: str, max_bytes: int):
    """A ``data:`` URL's payload -> an RGB ``PIL.Image``, oriented and validated.

    Refused with ``MMError`` (400 or 413, never a traceback) when the payload is
    not decodable, over ``max_bytes``, or over ``MAX_DECODE_PIXELS``. PIL's own
    decompression-bomb guard is part of that: the container leaves
    ``MAX_IMAGE_PIXELS`` at its default, so a bomb raises
    ``DecompressionBombError`` here rather than OOM-killing the server.

    Args:
        b64: Base64 payload of the ``data:`` URL, without the ``data:`` header.
        mime: The declared media type, used only in the error text.
        max_bytes: Cap on the decoded byte count for this request.

    Returns:
        ``(image, raw)``: the decoded RGB image and its original bytes. The raw
        bytes feed the request digest that salts the prefix-cache keys, so two
        different images can never share a cached KV prefix.

    Raises:
        MMError: The payload is not base64 (400), decodes to nothing (400),
            exceeds ``max_bytes`` (413), exceeds ``MAX_DECODE_PIXELS`` (413),
            trips the decompression-bomb guard (413), or is not a readable image
            (400). The caller forwards the status; retrying cannot help.
    """
    from PIL import Image, ImageOps, UnidentifiedImageError

    try:
        raw = base64.b64decode(b64, validate=True)
    except Exception as exc:  # noqa: BLE001 - any decode failure is the client's
        raise MMError(f"image payload is not valid base64: {exc}") from None
    if not raw:
        raise MMError("image payload decodes to zero bytes")
    if len(raw) > max_bytes:
        raise MMError(f"image is {len(raw):,} bytes decoded, over the "
                      f"{max_bytes:,}-byte (per-image) limit; downscale it first", 413)
    try:
        img = Image.open(io.BytesIO(raw))
        w, h = img.size
        if w * h > MAX_DECODE_PIXELS:
            raise MMError(
                f"image is {w}x{h} = {w * h / 1e6:.1f} MP; the decoder accepts at most "
                f"{MAX_DECODE_PIXELS / 1e6:.0f} MP (it is downsampled to the serving "
                f"size afterwards, but it has to be decoded first)", 413)
        # JPEGs can be greyscale/CMYK and PNGs paletted or 16-bit; PIL converts
        # all of them. ``load()`` is what actually decodes, so a truncated file
        # fails here with a readable message rather than inside the processor.
        img.load()

        # A phone photo stores "rotate 90" in EXIF, and a model that reads a
        # sideways "SCNET" gets the wrong answer.
        img = ImageOps.exif_transpose(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
    except MMError:
        raise
    except Image.DecompressionBombError as exc:
        raise MMError(f"image is too large to decode safely: {exc}", 413) from None
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise MMError(f"image could not be decoded ({mime}): {exc}") from None
    return img, raw


class Server:
    """One resident engine plus the request queues that feed it.

    Lifetime is the process: the weights and the captured graphs are built once
    here (``_build``) and survive until ``stop``, which is what makes a request
    cost a prefill rather than a 9 GB load. ``_reload`` rebuilds the same state
    from the resident tensors when the engine code changed.

    Every knob is a constructor argument rather than a module global because the
    engine's config has to be frozen before the model module is imported, and the
    diagnostic ops in ``handle`` read the live attributes back.
    """

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

    def _build(self) -> None:
        """Create the model and the engine, then bring up the vision tower.

        Split out of ``__init__`` because ``_reload`` runs exactly this step
        again against the tensors that are already resident. ``build_s`` is
        reported by ``ping`` and the startup log, so a slow build is visible
        without a profiler.
        """
        t0 = time.time()
        # The checkpoint config decides which family implementation is built;
        # ``serve`` itself only knows the engine_api.Model contract.
        self.model = engine_api.create_model(self.model_dir, max_seq=self.max_seq,
                                            bench=self.bench, weights=self.weights,
                                            paged=self.prefix_cache)
        # The engine owns the prefix cache, the block pool, the scheduler and the
        # captured graphs; ``serve`` keeps only the transport and rendering.
        self.engine = llm_engine.LLMEngine(self.cfg, self.model)
        self.engine.eos = set(self.eos)
        self.engine.build()
        self.pc = self.engine.pc
        # The tower is built here, after the engine's pools and graphs exist: it
        # is an eager module that never enters the decode graph, so it can come
        # up last and still be resident for the first image request.
        # ``--no-vision`` skips it, which is the only way to get the 0.62 GiB
        # back.
        if self.vision and self.model.vision_sd and self.model.vision is None:
            try:
                t_v = time.time()
                self.model.enable_vision(quant=self.vision_int8)
                print(f"[serve] vision tower ready in {time.time() - t_v:.1f}s "
                      f"(int8={self.vision_int8}, "
                      f"{len(self.model.vision_sd)} tensors)", flush=True)
            except Exception as exc:  # noqa: BLE001 - text serving must still work
                print(f"[serve] vision tower unavailable: {type(exc).__name__}: {exc}",
                      flush=True)
                self.vision = False
        elif not self.vision:
            self.model.vision_sd = {}
        self.build_s = time.time() - t0

    def _load_weight_kwargs(self) -> dict:
        """How to load this server's weights: text quant plus the vision policy.

        The tower goes through the same resident dict as the text weights (that
        is what makes ``reload`` rebuild from VRAM), but it has its own memory
        policy: fp16 by default, int8 with ``--vision-int8``, not loaded at all
        with ``--no-vision``.

        Returns:
            Extra keyword arguments for ``engine_api.load_weights``. Empty when
            the checkpoint declares no ``vision_config``, so a text-only
            checkpoint keeps the original call signature.
        """
        raw = json.loads((Path(self.model_dir) / "config.json").read_text())
        if not raw.get("vision_config"):
            return {}
        return {"vision": self.vision, "vision_quant": self.vision_int8}

    def _vision_ready(self) -> bool:
        """Whether the vision tower was actually loaded, so an image can be served.

        Returns:
            ``True`` only when the tower is in VRAM. ``--no-vision`` and a failed
            ``enable_vision`` both report ``False``; the caller answers 503
            rather than prefilling a prompt whose images have no embeddings.
        """
        if not self.vision:
            return False
        raw = json.loads((Path(self.model_dir) / "config.json").read_text())
        return bool(raw.get("vision_config"))

    def _ensure_processor(self):
        """Load ``Qwen3VLProcessor`` on first use (lazy, ~2 s, text-only pays 0).

        Lazy rather than eager because it is the only transformers piece with a
        heavy import cost besides the tokenizer, and a text-only deployment never
        touches it. The cost lands on the first image request, where it is one
        more second against a prefill already in the hundreds of ms and is
        reported in the request log, instead of on every cold start.

        Returns:
            The processor. It stays cached on the instance, so every later image
            request skips the load.

        Raises:
            MMError: 503 when the processor cannot be loaded. The failure string
                is remembered, so a broken checkpoint fails fast instead of
                retrying the load on every request.
        """
        if self.processor is not None:
            return self.processor
        if self._processor_failed is not None:
            raise MMError(f"the image processor is unavailable on this server: "
                          f"{self._processor_failed}", 503)
        try:
            from transformers import AutoProcessor

            self.processor = AutoProcessor.from_pretrained(self.model_dir)
        except Exception as exc:  # noqa: BLE001 - reported as a 503, not a crash
            self._processor_failed = f"{type(exc).__name__}: {exc}"
            raise MMError(f"could not load the image processor: "
                          f"{self._processor_failed}", 503) from None
        print(f"[serve] image processor ready: {type(self.processor).__name__} "
              f"({type(self.processor.image_processor).__name__})", flush=True)
        return self.processor

    def _prepare_mm(self, images: list[dict], text: str):
        """An image request's parts -> ``MMInput`` for the engine.

        Runs the whole image path before the prompt is prefilled, so every
        rejection is a clean status instead of a half-generated answer:

        1. PIL decode, orientation and RGB conversion per image
           (``_decode_image``);
        2. ``Qwen3VLProcessor`` over ``text`` (the rendered prompt, whose
           ``{"type": "image"}`` markers already became
           ``<|vision_start|><|image_pad|><|vision_end|>``): it downsamples each
           image into ``[image_min_pixels, image_max_pixels]``, patchifies it and
           expands the ``<|image_pad|>`` runs to one token per merged patch;
        3. a self-check that the expanded count is exactly ``Σ grid.prod() / 4``,
           which turns a processor-version drift into a 400 rather than an index
           mismatch deep inside the model;
        4. the vision tower -> embeddings, and ``mm_input`` -> 3-D positions.

        Args:
            images: The request's image parts, each ``{"b64", "mime"}``.
            text: The rendered prompt, with the template's image markers already
                expanded into the vision delimiters.

        Returns:
            ``(token_ids, mm)``. ``token_ids`` are the processor's own ids: on an
            image prompt the tokenizer and the processor agree, but taking them
            from the processor is what guarantees the placeholders and
            ``pixel_values`` line up (the count is asserted, so a disagreement
            cannot pass silently).

        Raises:
            MMError: The server has no vision tower (503), ``images`` is empty or
                over ``--max-images`` (400), an image is malformed or fails to
                decode (the status from ``_decode_image``), the processor raises
                (400), or any of the three alignment checks disagrees (400). All
                are client-visible refusals; none should be retried unchanged.
        """
        from mm_input import (IMAGE_TOKEN_ID, MMInput, find_image_spans,
                              image_digest, mrope_positions)

        proc = self._ensure_processor()
        if not self._vision_ready():
            raise MMError("this server cannot serve images (no vision tower was "
                          "loaded; see --no-vision)", 503)
        if not isinstance(images, list) or not images:
            raise MMError("`images` must be a non-empty array")
        if len(images) > self.max_images:
            raise MMError(f"too many images: {len(images)} > --max-images "
                          f"{self.max_images}")

        pil: list = []
        chunks: list[bytes] = []
        for i, item in enumerate(images):
            if not isinstance(item, dict) or "b64" not in item:
                raise MMError(f"image {i}: expected {{'b64': ..., 'mime': ...}}")
            img, raw = _decode_image(str(item["b64"]), str(item.get("mime") or ""),
                                     self.max_image_bytes)
            pil.append(img)
            chunks.append(raw)

        t0 = time.time()
        try:
            out = proc(text=[text], images=pil, return_tensors="pt",
                       size={"shortest_edge": self.image_min_pixels,
                             "longest_edge": self.image_max_pixels})
        except MMError:
            raise
        except Exception as exc:  # noqa: BLE001 - a bad image must not 500 the server
            raise MMError(f"image preprocessing failed: {type(exc).__name__}: {exc}") from None
        prep_ms = (time.time() - t0) * 1000

        ids = out["input_ids"][0].tolist()
        grid = out["image_grid_thw"].long()
        pixel_values = out["pixel_values"]
        # The processor merges 2x2 visual patches, so one merged patch is one
        # prompt token: grid.prod() counts patches, dividing by 4 counts tokens.
        want = int((grid.prod(-1) // 4).sum())

        # Three cheap checks that together pin the whole alignment:
        #
        # 1. the processor's tokenizer must agree with the one this file renders
        #    prompts with -- if they were different tokenizers every id here would
        #    be off and the model would answer something unrelated;
        # 2. the prompt's expansion must be *exactly* "each image marker becomes
        #    grid.prod()/4 tokens". ``tok.encode(text)`` is the unexpanded
        #    rendering (one ``<|image_pad|>`` per image), so the identity
        #    ``len(ids) == len(unexpanded) - n_images + Σ grid.prod()/4`` is the
        #    processor's own contract; a version drift breaks it here instead of
        #    surfacing as an embedding/KV index mismatch inside the model;
        # 3. the resulting placeholder runs are what the model overwrites, so
        #    their count and order must match the grids the tower encodes.
        plain_ids = self.tok.encode(text)
        n_plain_pads = sum(1 for t in plain_ids if t == IMAGE_TOKEN_ID)
        if n_plain_pads != len(images):
            raise MMError(
                f"the rendered prompt has {n_plain_pads} image placeholders but the "
                f"request carries {len(images)} images; the template and the image "
                f"parts disagree")
        if len(ids) != len(plain_ids) - len(images) + want:
            raise MMError(
                f"image expansion does not match the processor's grid: {len(ids)} "
                f"prompt tokens vs {len(plain_ids)} unexpanded - {len(images)} "
                f"placeholders + {want} image tokens")
        spans = find_image_spans(ids)
        got = sum(e - s for s, e in spans)
        if len(spans) != len(images) or got != want:
            raise MMError(
                f"image placeholder mismatch: the prompt has {len(spans)} image runs "
                f"covering {got} tokens, the processor produced {len(images)} images "
                f"needing {want} tokens (image_pad count must equal "
                f"sum(grid.prod()/4))")

        embeds = self.model.image_embeds(pixel_values, grid)
        positions, rope_delta = mrope_positions(spans, grid, len(ids))
        mm = MMInput(embeds=embeds, grid_thw=grid, positions=positions,
                     rope_delta=int(rope_delta),
                     digest=image_digest(chunks), spans=spans)
        self.n_image_req += 1
        print(f"[serve] mm: {len(images)} img grid={grid.tolist()} "
              f"img_tokens={want} n_prompt={len(ids)} rope_delta={rope_delta} "
              f"prep={prep_ms:.0f}ms ", flush=True)
        return ids, mm

    def _reload(self) -> None:
        """Re-import the engine (and the kernel extension) and rebuild from VRAM.

        The old model (buffers plus its captured graph pool) is torn down first,
        otherwise the two models do not both fit in 16 GB. The graph is nulled
        explicitly inside ``close()``: a live ``CUDAGraph`` pins its private
        memory pool, and that pool is ~half the model's footprint (all the
        grouped-projection copies), so merely dropping ``self.model`` is not
        enough.
        """
        import gc

        def _mem(tag: str) -> None:
            print(f"[reload] {tag}: alloc={torch.cuda.memory_allocated() / 2**30:.2f} "
                  f"reserved={torch.cuda.memory_reserved() / 2**30:.2f} GiB", flush=True)

        _mem("before")
        old = self.model
        self.engine = None
        self.model = None
        if old is not None:
            old.close()          # nulls the graphs (they pin their pools) + layers
        del old
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        _mem("after teardown")

        importlib.reload(hip_ops)

        # Every engine module, dependencies first: reloading in the other order
        # would leave old classes bound to the new tensors.
        engine_api.reload()
        self.processor = None    # re-imported transformers classes: rebuild lazily
        self._processor_failed = None
        self.n_reload += 1
        self._build()
        _mem("after build")

    # engine thread and request submission
    def start(self) -> None:
        """Run the serving loop on its own thread.

        Idempotent: a second call is a no-op, so ``serve`` can start the loop
        before it accepts connections without tracking the state itself.
        """
        if self._loop_thread is not None:
            return
        self._loop_thread = threading.Thread(target=self._engine_loop,
                                             name="engine-loop", daemon=True)
        self._loop_thread.start()

    def stop(self) -> None:
        """Ask the engine thread to return and wait up to 5 s for it.

        The thread also polls the condition with a 0.2 s timeout, so a missed
        notification costs one step rather than a hang. A join that times out is
        not an error: the thread is a daemon and the process is exiting anyway.
        """
        self._stop = True
        with self._cv:
            self._cv.notify_all()
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=5.0)
            self._loop_thread = None

    def _engine_loop(self) -> None:
        """One ``engine.step()`` per iteration, dispatching tokens to requests.

        The 0.2 s wait bounds the cost of a lost notification: the loop must also
        notice ``_paused`` (a diagnostic is touching the model) and ``_stop``,
        neither of which may be waited on indefinitely.
        """
        while not self._stop:
            with self._cv:
                while not self._stop:
                    engine = self.engine
                    if engine is not None and not self._paused \
                            and engine.scheduler.has_work():
                        break
                    self._cv.wait(0.2)
                if self._stop:
                    return
            try:
                with self._lock:
                    events = self.engine.step()
            except Exception as exc:  # keep serving; report to every active request
                print(f"[serve] engine step failed: {exc}", flush=True)
                self._fail(str(exc))
                self.engine.reset()
                continue
            for seq, tok, done in events:
                with self._lock:
                    out_q = self._outputs.get(seq.seq_id)
                if out_q is not None:
                    out_q.put({"token": tok, "done": done, "error": None,
                               "prefill_s": seq.prefill_s})
                if done:
                    with self._lock:
                        self._outputs.pop(seq.seq_id, None)

    def _fail(self, message: str) -> None:
        """Report one engine failure to every in-flight request and drop the queues.

        A failed ``step()`` has already lost the batch's state, so each waiting
        connection is answered with the error rather than left blocked on a queue
        nothing will feed again. The dict is swapped under the lock so a request
        submitted concurrently cannot land in the discarded copy.

        Args:
            message: The failure text, sent verbatim to every waiting connection.
        """
        with self._lock:
            pending, self._outputs = self._outputs, {}
        for out_q in pending.values():
            out_q.put({"token": None, "done": True, "error": message,
                       "prefill_s": 0.0})

    def _submit(self, ids: list[int], params: SamplingParams, mm=None):
        """Queue one request and return its sequence plus the token queue.

        ``mm`` rides on the :class:`~sequence.Sequence` (``seq.mm``), which is what
        the runner, the block manager and the radix tree all read: the tree salts
        its block keys with ``mm.digest`` so two different images can never share a
        cached KV prefix. The queue is registered *inside* the same lock that the
        engine loop's ``step()`` takes, so a prefilled sequence always has a queue
        waiting for its tokens.

        Args:
            ids: Prompt token ids, already rendered and (for an image request)
                already expanded to the processor's layout.
            params: Sampling and stopping policy for this request.
            mm: The request's ``MMInput``, or ``None`` for a text prompt.

        Returns:
            ``(seq, out_q)``: the engine-side sequence and the queue that will
            receive ``{"token", "done", "error", "prefill_s"}`` events.
        """
        out_q: queue.Queue = queue.Queue()
        with self._lock:
            seq = self.engine.add_request(ids, params, mm=mm)
            self._outputs[seq.seq_id] = out_q
            self._cv.notify()
        return seq, out_q

    def _blocking_generate(self, prompt_ids: list[int], max_new: int,
                           temperature: float, deadline: float = 0.0, mm=None):
        """Submit a request and collect its tokens until the engine says done.

        Args:
            prompt_ids: Prompt token ids.
            max_new: Output budget in tokens, already clamped by ``_room``.
            temperature: 0 for greedy.
            deadline: Epoch-seconds cutoff, 0 for none.
            mm: The request's ``MMInput``, or ``None``.

        Returns:
            ``(tokens, prefill_s, decode_s)``. ``decode_s`` excludes the prefill
            and is floored at 0 because the two timings are read from different
            clocks and can invert by a few microseconds on a short answer.

        Raises:
            RuntimeError: The engine reported an error for this request. The
                failed step has already been reset, so the caller may retry a new
                request, but not this sequence.
        """
        seq, out_q = self._submit(prompt_ids,
                                  self._sampling(max_new, temperature, deadline), mm=mm)
        t0 = time.time()
        out: list[int] = []
        while True:
            ev = out_q.get()
            if ev["error"]:
                raise RuntimeError(ev["error"])
            if ev["token"] is not None:
                out.append(ev["token"])
            if ev["done"]:
                break
        total = time.time() - t0
        pre = seq.prefill_s
        return out, pre, max(total - pre, 0.0)

    def _room(self, n_prompt: int, max_new) -> tuple[int, bool]:
        """Clamp the output budget to what the context window has left.

        The window is shared by prompt *and* output (262144 tokens total), and the
        decode step has no bounds check of its own: it writes the KV cache at
        ``pos`` and the rope tables are ``[max_seq]``, so a budget larger than the
        room would run off the end of both. Long-context clients routinely ask for
        a full window without looking at their prompt length, so the engine clamps
        instead of faulting.

        Args:
            n_prompt: Prompt length in tokens, after image expansion.
            max_new: Requested output budget; ``None`` or negative means "as much
                as fits", the caller stating no preference.

        Returns:
            ``(budget, window_full)``. ``budget`` is what the request may actually
            generate, never negative -- a prompt that exactly fills the window
            generates nothing. ``window_full`` says the *request* was cut by the
            window rather than finishing on its own. Both travel back in the
            reply so the client can tell "done" from "ran out of room".
        """
        room = max(0, self.max_seq - n_prompt)
        if max_new is None or int(max_new) < 0:
            return room, False
        return max(0, min(int(max_new), room)), int(max_new) > room

    def _sampling(self, max_new: int, temperature: float,
                  deadline: float = 0.0) -> SamplingParams:
        """Build the per-request decoding policy the engine's scheduler sees.

        Args:
            max_new: Output budget, after ``_room`` clamping.
            temperature: 0 for greedy, which is what enables the spec path.
            deadline: Epoch-seconds cutoff, 0 for none.

        Returns:
            The sampling parameters, with this server's full stop-id set attached
            so every path stops on the template's own end-of-turn token.
        """
        return SamplingParams(temperature=temperature, max_tokens=max_new,
                              eos=set(self.eos), deadline=deadline)

    def _deadline(self, req: dict) -> float:
        """Wall-clock cap for this generation (epoch seconds, 0 = none).

        The request may only shorten the server-wide ``--max-seconds``, never
        extend it: the cap exists to cover a client that goes quiet without
        closing its socket, so a client cannot opt out of it.

        Args:
            req: The request dict; ``max_seconds`` overrides the server default.

        Returns:
            The absolute cutoff, or 0 when no cap applies.
        """
        secs = float(req.get("max_seconds") or self.max_seconds or 0.0)
        return time.time() + secs if secs > 0 else 0.0

    def _cached_tokens(self) -> int:
        """How much of the last prompt came out of the prefix cache.

        Returns:
            The matched prefix length, 0 when nothing was reused. Read after the
            request finished, because a batched step updates it per sequence and
            only the reply needs it.
        """
        if self.pc is not None:
            return self.pc.last_matched
        return int(getattr(self.engine, "last_matched", 0))

    @torch.inference_mode()
    def _generate(self, prompt_ids: list[int], max_new: int, temperature: float,
                  deadline: float = 0.0, mm=None) -> tuple[list[int], float, float]:
        """Non-streaming generation through the engine's batched path.

        Args:
            prompt_ids: Prompt token ids.
            max_new: Output budget, after ``_room`` clamping.
            temperature: 0 for greedy.
            deadline: Epoch-seconds cutoff, 0 for none.
            mm: The request's ``MMInput``, or ``None``.

        Returns:
            ``(tokens, prefill_s, decode_s)``, as ``_blocking_generate`` returns
            them.

        Raises:
            RuntimeError: The engine reported an error for this request.
        """
        return self._blocking_generate(prompt_ids, max_new, temperature, deadline, mm=mm)

    @torch.inference_mode()
    def _spec_generate(self, prompt_ids: list[int], max_new: int, k: int,
                       deadline: float = 0.0,
                       mm=None) -> tuple[list[int], float, float]:
        """Greedy speculative decode, driving the model instead of the scheduler.

        This bypasses the engine's request queue: it resets the model, prefills,
        then loops on ``spec_step`` directly. That is what lets the ``spec`` op
        A/B it against the plain path, and it means the verify graph must already
        be captured for ``k + 1`` tokens (``capture_verify``) or the first step
        raises.

        Only greedy is supported: the accept/reject test compares the main
        model's argmax against the draft, which is exact for greedy and not for
        sampling.

        Args:
            prompt_ids: Prompt token ids.
            max_new: Output budget, after ``_room`` clamping.
            k: Drafts per step; the verify graph must already be captured for
                ``k + 1`` tokens.
            deadline: Epoch-seconds cutoff, 0 for none.
            mm: The request's ``MMInput``, or ``None``. Supported, not refused:
                ``prefill(mm=...)`` writes the request's ``rope_delta`` and both
                the drafter and the verify graph read it per step (``spec_step``
                copies it into ``mtp.rope_delta``), so an image prompt speculates
                exactly like a text one.

        Returns:
            ``(tokens, prefill_s, decode_s)``. ``decode_s`` covers the spec loop
            only, so it is directly comparable with the plain path's decode time.
        """
        m = self.model
        m.reset()
        t0 = time.time()
        logits = m.prefill(prompt_ids, mm=mm)
        t_pre = time.time() - t0
        m.n_drafted = m.n_accepted = m.n_spec_steps = 0
        out: list[int] = []
        t1 = time.time()
        while len(out) < max_new:
            if int(m.pos1) + m.verify_c > m.max_seq:
                break
            if deadline and time.time() >= deadline:
                break
            toks, logits = m.spec_step(logits, k)
            stop = False
            for t in toks:
                if t in self.eos:
                    stop = True
                    break
                out.append(t)
                if len(out) >= max_new:
                    break
            if stop:
                break
        return out[:max_new], t_pre, time.time() - t1

    def _render_with_text(self, messages: list[dict], tools, enable_thinking: bool,
                          forced_prefix: str = "") -> tuple[list[int], str]:
        """Render a conversation with the checkpoint's own jinja template.

        Using the checkpoint's template rather than a hand-built string is what
        keeps the tool schemas, the thinking mode and the special tokens exactly
        as the model was trained to see them.

        Args:
            messages: Conversation turns in OpenAI chat format.
            tools: Tool schemas, or ``None``; the template renders them into the
                system turn, so a text prompt with tools is not a plain prompt.
            enable_thinking: Whether the template opens the reasoning block.
            forced_prefix: Template opener appended to the assistant turn, used to
                implement ``tool_choice: required``.

        Returns:
            ``(token_ids, text)``. The text is kept because the image path has to
            hand the *rendered* prompt to the processor, and because the caller
            inspects its tail to see whether thinking was opened.
        """
        text = self.tok.apply_chat_template(
            messages, tools=tools, enable_thinking=enable_thinking,
            tokenize=False, add_generation_prompt=True)
        if forced_prefix:
            # ``tool_choice: required`` (OpenAI's ``ChatCompletionToolChoiceOption``)
            # is implemented by continuing the assistant turn: the rendered prompt
            # already ends at the assistant header, so appending the template's
            # opener leaves the model no way out but to finish the call.
            text += forced_prefix
        return self.tok.encode(text), text

    def _render(self, messages: list[dict], tools, enable_thinking: bool,
                forced_prefix: str = "") -> list[int]:
        """Token ids of a rendered conversation; the text form is not needed.

        Args:
            messages: Conversation turns in OpenAI chat format.
            tools: Tool schemas, or ``None``.
            enable_thinking: Whether the template opens the reasoning block.
            forced_prefix: Template opener appended to the assistant turn.

        Returns:
            The prompt token ids, ready to submit.
        """
        return self._render_with_text(messages, tools, enable_thinking, forced_prefix)[0]

    @staticmethod
    def _split(raw: str, enable_thinking: bool) -> tuple[str, str]:
        """Split reasoning from answer, knowing whether thinking was asked for.

        ``split_thinking`` cannot tell "no reasoning" from "reasoning that was cut
        off before the closing marker": with thinking on and no ``</think>`` in the
        text, everything generated so far *is* reasoning. Returning it as the
        answer is how a thinking model's monologue ends up looking like a reply.

        Args:
            raw: The full generated text, forced prefix included.
            enable_thinking: Whether the caller asked the template for reasoning.

        Returns:
            ``(thinking, answer)``. Both are empty strings when the model produced
            neither; the answer is empty for a truncated reasoning block.
        """
        thinking, answer = split_thinking(raw)
        if enable_thinking and not thinking and "</think>" not in raw:
            thinking, answer = answer, ""
        return thinking, answer

    def _render_request(self, req: dict, thinking_on: bool, prefix: str):
        """Render one request's prompt -> ``(token_ids, text, MMInput|None)``.

        The text-only path is render, encode, no ``mm``. A request carrying
        ``images`` renders the *same* way (the template turns its
        ``{"type": "image"}`` markers into the vision delimiters) and then hands
        the rendered text to the processor, which is the only difference -- that
        keeps one rendering rule for both paths, so adding vision could never
        change how a text-only prompt renders.

        Args:
            req: The request dict: ``messages``, optional ``tools`` and ``images``.
            thinking_on: Whether the template should open the reasoning block.
            prefix: The forced tool-call opener, empty when the caller did not ask
                for ``tool_choice: required``.

        Returns:
            ``(token_ids, text, mm)``; ``mm`` is ``None`` for a text request. For
            an image request the token ids are the processor's expanded ones, not
            the tokenizer's.

        Raises:
            MMError: The request carries ``images`` while a forced tool call is
                active (400), or the image path refused (the status from
                ``_prepare_mm``). The two are mutually exclusive because the
                forced call needs the assistant turn to open the call, which
                leaves no room for the vision delimiters.
        """
        messages = req.get("messages", [])
        images = req.get("images") or []
        ids, prompt = self._render_with_text(messages, req.get("tools"), thinking_on,
                                             prefix)
        if not images:
            return ids, prompt, None
        if prefix:
            raise MMError("tool_choice=required forces the assistant turn to open "
                          "with a call; it cannot be combined with images")
        mm_ids, mm = self._prepare_mm(images, prompt)
        return mm_ids, prompt, mm

    @torch.inference_mode()
    def stream_chat(self, req: dict):
        """``chat`` as a generator: one reply line per step, for SSE clients.

        Yields ``type: "start"`` (prompt length and whether the prompt itself
        opened the reasoning block), then one ``type: "token"`` per decoded piece,
        then a final ``type: "done"`` with exactly the fields of the ``chat``
        reply. That three-line framing is the whole contract ``openai_server.py``
        forwards, so a new field belongs in ``done``, never in a new line type.

        ``text`` on a token line is *incremental*: only the new token's ids are
        decoded each step, so streaming stays linear in the answer length. The
        previous step's ids are kept only while they end in the middle of a
        multi-byte character, which is never more than a couple.

        A refused request is yielded as ``type: "error"`` with its status,
        *before* the ``start`` line: SSE has already committed a 200 by then, so
        ``openai_server`` maps that line's status into the stream event instead of
        pretending the request succeeded.

        Speculative decode is deliberately not used here: the spec loop only
        returns after a whole K+1 verification window, which is the opposite of
        what a stream wants.

        Args:
            req: The request dict: ``messages``, optional ``tools``, ``images``,
                ``max_new``, ``temperature``, ``enable_thinking``,
                ``tool_prefix`` and ``max_seconds``.

        Raises:
            RuntimeError: The engine reported an error for this request. It is
                raised after the ``start`` line, so the transport closes the
                stream rather than sending a ``done`` that claims success.
        """
        prefix = str(req.get("tool_prefix") or "")

        # A forced call must start the answer, so reasoning is off for that turn:
        # the template would otherwise open ``<think>`` where the call has to go.
        thinking_on = bool(req.get("enable_thinking", False)) and not prefix
        try:
            ids, prompt, mm = self._render_request(req, thinking_on, prefix)
        except MMError as exc:
            yield {"type": "error", "ok": False, "error": str(exc),
                   "status": exc.status}
            return
        n, window_full = self._room(len(ids), req.get("max_new"))
        temp = float(req.get("temperature", 0.0))
        deadline = self._deadline(req)
        yield {"type": "start", "n_prompt": len(ids), "max_new": n,
               "prompt_opens_think": prompt.rstrip().endswith("<think>"),
               "forced_tool": bool(prefix),
               "n_images": mm.num_images if mm is not None else 0,
               "image_tokens": len(mm) if mm is not None else 0}
        t0 = time.time()
        seq, out_q = self._submit(ids, self._sampling(n, temp, deadline), mm=mm)
        out: list[int] = []
        hold: list[int] = []             # token ids held back: partial UTF-8
        t1 = time.time()
        while True:
            ev = out_q.get()
            if ev["error"]:
                raise RuntimeError(ev["error"])
            nxt = ev["token"]
            if nxt is not None:
                out.append(nxt)

                # Incremental decode: decoding the whole list every step is O(n^2)
                # and dominates a long answer (a 20k-token reply would spend
                # minutes in the tokenizer). A single token can end in the middle
                # of a UTF-8 sequence, so its ids are held back and decoded
                # together with the next ones until the character completes.
                held = hold + [nxt] if hold else [nxt]
                text = self.tok.decode(held, skip_special_tokens=True)
                if text.endswith("\ufffd"):
                    hold = held
                else:
                    hold = []
                    if text:
                        yield {"type": "token", "id": nxt, "text": text}
            if ev["done"]:
                break
        if hold:                         # the sequence never completed (truncated)
            text = self.tok.decode(hold, skip_special_tokens=True)
            if text:
                yield {"type": "token", "id": hold[-1], "text": text}
        t_dec = time.time() - t1
        t_pre = seq.prefill_s
        raw = self.tok.decode(out, skip_special_tokens=False)

        # The forced opener lives in the *prompt*, so it is put back in front of
        # the generated text before the call is parsed -- otherwise the block
        # would look like `<function=...>` with no enclosing `<tool_call>`.
        full = prefix + raw
        thinking, answer = self._split(full, thinking_on)
        calls = parse_tool_calls(answer)
        yield {"type": "done", "ok": True, "text": answer, "thinking": thinking,
               "raw": full, "tokens": out,
               "tool_calls": [c.to_openai_dict() for c in calls],
               "n_prompt": len(ids), "n_new": len(out), "max_new": n,
               "window_full": window_full, "cached_tokens": self._cached_tokens(),
               "prefill_ms": round(t_pre * 1000, 1),
               "decode_ms": round(t_dec * 1000, 1),
               "decode_tok_s": round(len(out) / t_dec, 2) if t_dec else None,
               # why generation stopped, when the engine decided it itself
               "stopped": ("deadline" if deadline and time.time() >= deadline else None),
               "spec_k": 0, "spec": None}

    def handle(self, req: dict) -> dict:
        """Serve one non-streaming op.

        Diagnostics (bench/profile/ops/spec/reload/reset) touch the model
        directly, so they are refused while requests are in flight and pause the
        engine loop while they run: ``step`` and the diagnostic take the same
        lock. Pausing is what makes them safe at all -- a reload tears the model
        down, and a profiler capture would otherwise interleave with a live step.

        Args:
            req: One request dict; ``op`` defaults to ``ping``.

        Returns:
            The reply dict, always with ``ok``. A refusal is ``ok: False`` plus an
            ``error`` string, not an exception: the transport turns that into a
            normal JSON line the client can read.
        """
        op = req.get("op", "ping")
        debug = op in ("reload", "reset", "bench", "profile", "ops", "spec")
        if not debug:
            return self._handle_impl(req)
        engine = self.engine
        if op in ("bench", "profile", "ops", "spec") and engine is not None \
                and engine.batching:
            return {"ok": False, "error": "diagnostic ops need the single-stream "
                                          "engine (run without --max-num-seqs)"}
        if engine is not None and (engine.scheduler.waiting or engine.scheduler.running):
            return {"ok": False, "error": "engine busy: diagnostic ops need an idle engine"}
        with self._lock:
            self._paused = True
            try:
                return self._handle_impl(req)
            finally:
                self._paused = False
                self._cv.notify_all()

    def _handle_impl(self, req: dict) -> dict:
        """Dispatch one op and build its reply.

        Every reply carries the same timing fields (``prefill_ms``,
        ``decode_ms``, ``decode_tok_s``, ``cached_tokens``) so a probe can compare
        paths without special-casing them. No locking here: the caller holds it.

        Args:
            req: The request dict; its keys depend on ``op``.

        Returns:
            The reply dict; ``{"ok": False, "error": ...}`` for an unknown op.
        """
        op = req.get("op", "ping")
        self.n_req += 1
        if op == "ping":
            out = {"ok": True, **self.info,
                   "graph": self.model.graph_ready, "max_seq": self.max_seq,
                   "load_s": round(self.load_s, 1), "build_s": round(self.build_s, 1),
                   "n_req": self.n_req, "n_reload": self.n_reload,
                   "hip": hip_ops.available(), "int8": self.int8,
                   "vram_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
                   "batch": bool(self.engine is not None and self.engine.batching),
                   "decode_capacity": int(getattr(self.model, "decode_capacity", 1)),
                   # what image requests this server accepts; the OpenAI shell reads
                   # ``vision`` to answer 503 up front instead of forwarding a request
                   # the engine will reject anyway
                   "vision": bool(self.vision and self.model.vision_sd),
                   "vision_int8": self.vision_int8,
                   "image_max_pixels": self.image_max_pixels,
                   "image_min_pixels": self.image_min_pixels,
                   "max_images": self.max_images,
                   "n_image_req": self.n_image_req,
                   "processor": ("loaded" if self.processor is not None
                                 else ("failed" if self._processor_failed else "lazy"))}

            if self.engine is not None:
                out["queue"] = {"waiting": len(self.engine.scheduler.waiting),
                                "running": len(self.engine.scheduler.running)}
            if self.pc is not None:
                out["prefix"] = self.pc.stats()
            elif self.engine is not None and self.engine.bm is not None \
                    and self.engine.bm.radix is not None:
                out["prefix"] = self.engine.bm.stats()
            return out
        if op == "reload":
            self._reload()
            return {"ok": True, "build_s": round(self.build_s, 1), "n_reload": self.n_reload,
                    "hip": hip_ops.available(), "hip_err": hip_ops.last_error()}
        if op == "reset":
            self.engine.reset()
            return {"ok": True}
        if op == "stop":
            return {"ok": True, "stopping": True}
        if op == "generate":
            ids = req.get("tokens") or self.tok.encode(req.get("prompt", ""))
            n, window_full = self._room(len(ids), req.get("max_new"))
            temp = float(req.get("temperature", 0.0))
            deadline = self._deadline(req)
            k = int(req.get("spec_k", self.spec_k))
            spec = None
            if k and temp == 0:
                out, t_pre, t_dec = self._spec_generate(ids, n, k, deadline)
                spec = self.model.spec_stats
            else:
                out, t_pre, t_dec = self._generate(ids, n, temp, deadline)
            return {"ok": True, "tokens": out, "text": self.tok.decode(out),
                    "n_prompt": len(ids), "n_new": len(out), "max_new": n,
                    "window_full": window_full, "cached_tokens": self._cached_tokens(),
                    "prefill_ms": round(t_pre * 1000, 1), "decode_ms": round(t_dec * 1000, 1),
                    "decode_tok_s": round(len(out) / t_dec, 2) if t_dec else None,
                    "stopped": ("deadline" if deadline and time.time() >= deadline else None),
                    "spec_k": k if spec else 0, "spec": spec}
        if op == "chat":
            prefix = str(req.get("tool_prefix") or "")
            thinking_on = bool(req.get("enable_thinking", False)) and not prefix
            try:
                ids, _prompt, mm = self._render_request(req, thinking_on, prefix)
            except MMError as exc:
                return {"ok": False, "error": str(exc), "status": exc.status}
            n, window_full = self._room(len(ids), req.get("max_new"))
            temp = float(req.get("temperature", 0.0))
            deadline = self._deadline(req)
            k = int(req.get("spec_k", self.spec_k))
            spec = None
            if k and temp == 0:
                out, t_pre, t_dec = self._spec_generate(ids, n, k, deadline, mm=mm)
                spec = self.model.spec_stats
            else:
                out, t_pre, t_dec = self._generate(ids, n, temp, deadline, mm=mm)
            raw = self.tok.decode(out, skip_special_tokens=False)
            full = prefix + raw
            thinking, answer = self._split(full, thinking_on)
            calls = parse_tool_calls(answer)
            return {"ok": True, "text": answer, "thinking": thinking, "raw": full,
                    "tool_calls": [c.to_openai_dict() for c in calls], "forced_tool": bool(prefix),
                    "n_prompt": len(ids), "n_new": len(out), "max_new": n,
                    "window_full": window_full, "cached_tokens": self._cached_tokens(),
                    "prefill_ms": round(t_pre * 1000, 1),
                    "decode_tok_s": round(len(out) / t_dec, 2) if t_dec else None,
                    "stopped": ("deadline" if deadline and time.time() >= deadline else None),
                    "spec_k": k if spec else 0, "spec": spec,
                    # what the image path actually did, so a probe can report it
                    "n_images": mm.num_images if mm is not None else 0,
                    "image_tokens": len(mm) if mm is not None else 0}
        if op == "spec":
            # A/B the two decode paths on the same prompt. Both are warmed before
            # timing, and the spec output is compared against plain greedy because
            # a speculative loop that is only "about as good" is not a decoder.
            n = int(req.get("n", 48))
            ids = self.tok.encode(req.get("prompt") or "The quick brown fox")
            k = int(req.get("spec_k", self.spec_k or 1))
            self.model.capture_verify(k + 1)
            self._generate(ids, 8, 0.0)
            plain, _, t_a = self._generate(ids, n, 0.0)
            self._spec_generate(ids, 8, k)
            spec, _, t_b = self._spec_generate(ids, n, k)
            same = sum(a == b for a, b in zip(plain, spec))
            return {"ok": True, "n": n, "k": k, "n_prompt": len(ids),
                    "plain_tok_s": round(len(plain) / t_a, 2),
                    "spec_tok_s": round(len(spec) / t_b, 2),
                    "speedup": round((len(spec) / t_b) / (len(plain) / t_a), 3),
                    "exact": f"{same}/{min(len(plain), len(spec))}",
                    "accept": self.model.spec_stats}
        if op == "bench":
            n = int(req.get("n", 30))
            ids = self.tok.encode(req.get("prompt") or "The quick brown fox")
            self.model.reset()
            logits = self.model.prefill(ids)
            for _ in range(3):  # untimed: the first replay pays graph cache warming
                logits = self.model.decode_step(int(logits.argmax()))
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(n):
                logits = self.model.decode_step(int(logits.argmax()))
            torch.cuda.synchronize()
            dt = time.time() - t0
            return {"ok": True, "n": n, "ms_per_step": round(dt / n * 1000, 2),
                    "decode_tok_s": round(n / dt, 2)}
        if op == "profile":
            n = int(req.get("n", 30))
            ids = self.tok.encode(req.get("prompt") or "The quick brown fox")
            self.model.reset()
            logits = self.model.prefill(ids)
            for _ in range(3):
                logits = self.model.decode_step(int(logits.argmax()))
            torch.cuda.synchronize()
            with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                for _ in range(n):
                    logits = self.model.decode_step(int(logits.argmax()))
                torch.cuda.synchronize()
            ka = prof.key_averages()
            total = sum(r.self_device_time_total for r in ka)
            rows = sorted(ka, key=lambda r: r.self_device_time_total, reverse=True)[:18]
            return {"ok": True, "total_us": round(total / n, 1),
                    "rows": [{"kernel": r.key[:60], "calls": r.count // n,
                              "us": round(r.self_device_time_total / n, 1),
                              "pct": round(100 * r.self_device_time_total / total, 1)}
                             for r in rows]}
        if op == "ops":
            # eager (un-graphed) forward so each aten op is individually visible,
            # grouped by op name with its self device time -> per-op attribution
            n = int(req.get("n", 5))
            ids = self.tok.encode(req.get("prompt") or "The quick brown fox")
            self.model.reset()
            tok = int(self.model.prefill(ids).argmax())
            with torch.inference_mode(), torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA]) as prof:
                for _ in range(n):
                    self.model.logits(tok)
                torch.cuda.synchronize()
            ka = prof.key_averages()
            total = sum(r.self_device_time_total for r in ka)
            rows = sorted(ka, key=lambda r: r.self_device_time_total, reverse=True)[:25]
            return {"ok": True, "n": n, "total_us": round(total / n, 1),
                    "rows": [{"op": r.key[:70], "calls": r.count // n,
                              "us": round(r.self_device_time_total / n, 1),
                              "pct": round(100 * r.self_device_time_total / total, 1)}
                             for r in rows]}
        return {"ok": False, "error": f"unknown op {op!r}"}


def serve(model_dir: str, max_seq: int, bench: bool, sock_path: str,
          spec_k: int = 0, int8: bool = True, prefix_cache: bool = True,
          block_size: int = 64, snapshots: int = 8, snapshot_interval: int = 512,
          max_seconds: float = 0.0, max_num_seqs: int = 1,
          num_blocks: int = 0, spec_batch: bool = False,
          spec_sparse_k: int = 0, spec_min_ctx: int = 0,
          fast_gemv: bool = False, vision: bool = True, vision_int8: bool = False,
          image_max_pixels: int = 1048576, image_min_pixels: int = 65536,
          max_images: int = 4, max_image_bytes: int = 8388608) -> None:
    """Build the server, bind the unix socket and serve until asked to stop.

    The argument list mirrors ``Server.__init__``; see it for the meaning of each
    knob. Everything here is process lifetime, not per request.

    Args:
        model_dir: Checkpoint directory.
        max_seq: Context window, shared by prompt and output.
        bench: Keep the legacy single-stream bench path available.
        sock_path: Unix socket to bind; an existing file at that path is removed
            first, so a stale socket from a killed run does not block startup.
        spec_k: Drafts per speculative step, 0 to disable.
        int8: Quantise the 2D weights.
        prefix_cache: Enable the paged radix prefix cache.
        block_size: KV block granularity in tokens.
        snapshots: Recurrent-state snapshot budget.
        snapshot_interval: Snapshot every N tokens, rounded to a block.
        max_seconds: Wall-clock cap on one generation, 0 for none.
        max_num_seqs: Rows in a batched decode step.
        num_blocks: KV pool size, 0 to derive it from the window.
        spec_batch: Speculate in the batched engine.
        spec_sparse_k: Approximate batched verify, top-k historical blocks.
        spec_min_ctx: Skip batched speculation below this cached length.
        fast_gemv: Use the packed-fp16 projection kernels.
        vision: Load the checkpoint's vision tower when it has one.
        vision_int8: Keep the tower's 2D weights int8.
        image_max_pixels: Largest image the processor serves.
        image_min_pixels: Smallest image after the processor's upscale.
        max_images: Images accepted in one request.
        max_image_bytes: Decoded bytes allowed per image.
    """
    srv_model = Server(model_dir, max_seq, bench, spec_k, int8, prefix_cache, block_size,
                       snapshots, snapshot_interval, max_seconds, max_num_seqs,
                       num_blocks, spec_batch, spec_sparse_k, spec_min_ctx, fast_gemv,
                       vision, vision_int8, image_max_pixels, image_min_pixels,
                       max_images, max_image_bytes)
    info = srv_model.info
    print(f"[serve] model {info['model']} ({info['arch']}) at {info['model_path']}",
          flush=True)
    prefix_on = (srv_model.pc is not None
                 or (srv_model.engine.bm is not None
                     and srv_model.engine.bm.radix is not None))
    print(f"[serve] weights {srv_model.load_s:.1f}s + build {srv_model.build_s:.1f}s  "
          f"eos={sorted(srv_model.eos)}  hip={hip_ops.available()}  "
          f"int8={int8}  max_seq={max_seq}  "
          f"prefix_cache={'paged' if prefix_on else 'off'}"
          f"{' (bs=%d, snapshots=%d)' % (block_size, snapshots) if prefix_on else ''}"
          f"  batch={srv_model.cfg.max_num_seqs if srv_model.engine.batching else 'off'}"
          f"  blocks={srv_model.cfg.num_blocks}"
          f"  spec_k={srv_model.cfg.spec_k}"
          f"{' (batched)' if srv_model.cfg.spec_batch else ''}"
          f"  max_seconds={max_seconds or 'none'}"
          f"  vision={'off' if not srv_model.vision else ('int8' if srv_model.vision_int8 else 'fp16')}"
          f" (max {srv_model.image_max_pixels} px, {srv_model.max_images} imgs)",
          flush=True)
    if os.path.exists(sock_path):
        os.unlink(sock_path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    srv.listen(8)
    print(f"[serve] listening on {sock_path}", flush=True)
    srv_model.start()

    # The accept timeout is how the loop notices ``_stop``: a blocking accept
    # would keep the socket open after a ``stop`` request until a new client
    # happened to connect.
    srv.settimeout(0.5)
    while not srv_model._stop:
        try:
            conn, _ = srv.accept()
        except socket.timeout:
            continue

        # One thread per connection: the engine thread batches whatever requests
        # are in flight, so a slow SSE client no longer blocks the next request.
        threading.Thread(target=_serve_connection, args=(srv_model, conn),
                         daemon=True).start()
    srv_model.stop()
    srv.close()
    if os.path.exists(sock_path):
        os.unlink(sock_path)


def _serve_connection(srv_model: "Server", conn: socket.socket) -> None:
    """Serve one JSON-line request (or one SSE stream) on its own thread.

    The reply goes out as a single line, so the client's read-until-newline loop
    terminates. A ``chat_stream`` op instead writes one line per stream event,
    which is the framing ``openai_server.py`` turns into SSE frames.

    Args:
        srv_model: The shared server; every thread sees the same one.
        conn: The accepted connection. It is closed here in every case, including
            a malformed request, so a bad client cannot leak a descriptor.
    """
    try:
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = conn.recv(65536)
            if not chunk:
                break
            buf += chunk
        req = json.loads(buf.decode() or "{}")
        t0 = time.time()
        if req.get("op") == "chat_stream":
            # One reply line per token, so the client can forward them as SSE.
            for line in srv_model.stream_chat(req):
                conn.sendall((json.dumps(line, ensure_ascii=False) + "\n").encode())
            print(f"[serve] chat_stream -> {round((time.time() - t0) * 1000, 1)}ms",
                  flush=True)
            return
        rep = srv_model.handle(req)
        rep["wall_ms"] = round((time.time() - t0) * 1000, 1)
        conn.sendall((json.dumps(rep, ensure_ascii=False) + "\n").encode())
        print(f"[serve] {req.get('op')} -> {rep.get('wall_ms')}ms", flush=True)
        if rep.get("stopping"):
            srv_model.stop()
    except Exception as exc:  # keep the server alive on a bad request
        # An ``MMError`` that escaped a path which could not answer cleanly (a
        # stream whose 200 is already committed) is still reported with its status
        # so the HTTP shell can map it instead of calling everything a 500.
        rep = {"ok": False, "error": str(exc)}
        if isinstance(exc, MMError):
            rep["status"] = exc.status
        try:
            conn.sendall((json.dumps(rep) + "\n").encode())
        except Exception:
            pass
        print(f"[serve] error: {exc}", flush=True)
    finally:
        conn.close()


def call(sock_path: str, req: dict) -> dict:
    """Send one request to a running server and return its reply.

    The client half of the wire format: one JSON line out, one back.

    Args:
        sock_path: The engine socket the server bound.
        req: The request dict; ``op`` selects the handler.

    Returns:
        The decoded reply. It may carry ``ok: False``; this function does not
        raise on a refusal, because the CLI prints the reply either way.

    Raises:
        OSError: The server is not listening on ``sock_path``. Start it first;
            a client-side retry cannot fix a missing server.
        json.JSONDecodeError: The server closed without sending a full line.
    """
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(sock_path)
    s.sendall((json.dumps(req) + "\n").encode())
    buf = b""
    while not buf.endswith(b"\n"):
        chunk = s.recv(65536)
        if not chunk:
            break
        buf += chunk
    s.close()
    return json.loads(buf.decode())


def main() -> None:
    """CLI entry point: serve in this process, or act as a client of one.

    ``--serve`` owns the socket and the model; every other invocation is a thin
    client that prints the reply, so the two roles never share a code path and a
    client cannot accidentally load weights.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("cmd", nargs="?", default="ping",
                    choices=["ping", "reload", "reset", "stop", "generate", "chat",
                             "bench", "profile", "ops", "spec"])
    ap.add_argument("--model", default=MODEL,
                    help="checkpoint directory; the architecture is read from its "
                         "config.json (default: $ENGINE_MODEL or the Qwen3.5-4B path)")
    ap.add_argument("--sock", default=SOCK, help="engine socket (default: $ENGINE_SOCK)")
    ap.add_argument("--max-seq", type=int, default=MAX_SEQ,
                    help="context window; the KV cache is allocated up front, so a "
                         "256k window costs ~4.4 GB of VRAM (8 int8 full-attn layers)")
    ap.add_argument("--no-prefix-cache", dest="prefix_cache", action="store_false",
                    help="disable the paged radix prefix cache. It is what makes a "
                         "repeated prefix (an agent resending its conversation) cheap: "
                         "only the unseen suffix is prefilled")
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--snapshots", type=int, default=8,
                    help="recurrent-state snapshot budget (~50 MB each); the join "
                         "points a later request can reuse")
    ap.add_argument("--snapshot-interval", type=int, default=512,
                    help="snapshot every N tokens (rounded to a block)")
    ap.add_argument("--fp16", dest="int8", action="store_false",
                    help="keep the 2D weights in fp16 instead of int8: ~4 GiB more VRAM "
                         "and a slower decode, in exchange for ~0.3%% better perplexity "
                         "(NOTES §6). Default is int8, because the 256k window and the "
                         "weights have to share 16 GB.")
    ap.add_argument("--no-bench", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=0.0,
                    help="hard wall-clock cap on one generation, seconds (0 = none). "
                         "This is the only cancellation that also covers a client that "
                         "goes quiet without closing its socket -- otherwise the single "
                         "stream keeps generating for a request nobody is reading")
    ap.add_argument("--spec-k", type=int, default=0,
                    help="drafts per speculative step (0 = off); the verify graph is "
                         "captured for K+1 tokens")
    ap.add_argument("--spec-batch", action="store_true",
                    help="also speculate in the batched engine. Off by default: on "
                         "this card the batched verify costs one weight pass per "
                         "candidate token, so it is ~0.79x a plain batch decode step "
                         "(output is identical either way)")
    ap.add_argument("--spec-min-ctx", type=int, default=0,
                    help="only speculate in the batched engine once a sequence has "
                         "this many cached tokens (0 = always); below it a plain "
                         "batched decode step is cheaper")
    ap.add_argument("--fast-gemv", action="store_true",
                    help="use the packed-fp16 (v_dot2) projection kernels in the "
                         "batched decode: ~1.18x the projection segment on this card, "
                         "but 1-2 ulp off the single-stream result (not bit-identical)")
    ap.add_argument("--spec-sparse-k", type=int, default=0,
                    help="approximate batched verify: keep only the top-k historical "
                         "KV blocks per row (drafter lookahead + last attention) plus "
                         "a recent window; 0 = exact full attention. Implies "
                         "--spec-batch; token output is no longer guaranteed equal")
    ap.add_argument("--max-num-seqs", type=int, default=1,
                    help="sequences decoded in one batched step (1 = the original "
                         "single-stream engine). Requests still queue; the scheduler "
                         "admits up to this many and batches their decode steps.")
    ap.add_argument("--num-blocks", type=int, default=0,
                    help="KV pool blocks shared by all sequences (0 = "
                         "max_seq/block_size * max_num_seqs). Lower it when the "
                         "window is long: the pool is allocated up front.")

    # images
    ap.add_argument("--no-vision", dest="vision", action="store_false",
                    help="do not load the vision tower (saves ~0.62 GiB); image "
                         "requests then answer 503 instead of being served")
    ap.add_argument("--vision-int8", action="store_true",
                    help="keep the vision tower's 2-D weights int8 (0.31 GiB instead "
                         "of 0.62), dequantised one GEMM at a time per request")
    ap.add_argument("--image-max-pixels", type=int, default=1048576,
                    help="largest image the processor will serve, in pixels "
                         "(default 1 MP = ~1024 image tokens). This is the only real "
                         "VRAM/latency knob: anything bigger is downsampled")
    ap.add_argument("--image-min-pixels", type=int, default=65536,
                    help="smallest image after the processor's upscale (official "
                         "value; an image never costs fewer than 64 tokens)")
    ap.add_argument("--max-images", type=int, default=4,
                    help="images accepted in one request (four 1 MP images are "
                         "already ~4k image tokens)")
    ap.add_argument("--max-image-bytes", type=int, default=8388608,
                    help="decoded bytes allowed per image (default 8 MiB)")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--user", default="")
    ap.add_argument("--system", default=None)
    ap.add_argument("--tools", default=None, help="JSON file with an OpenAI tools array")
    ap.add_argument("--think", action="store_true", help="enable thinking mode")
    ap.add_argument("-n", "--max-new", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    if args.serve:
        serve(args.model, args.max_seq, not args.no_bench, args.sock, args.spec_k,
              args.int8, args.prefix_cache, args.block_size, args.snapshots,
              args.snapshot_interval, args.max_seconds, args.max_num_seqs,
              args.num_blocks, args.spec_batch, args.spec_sparse_k,
              args.spec_min_ctx, args.fast_gemv, args.vision, args.vision_int8,
              args.image_max_pixels, args.image_min_pixels, args.max_images,
              args.max_image_bytes)
        return

    if args.cmd in ("ping", "reload", "reset", "stop"):
        print(json.dumps(call(args.sock, {"op": args.cmd}), ensure_ascii=False, indent=2))
    elif args.cmd == "spec":
        r = call(args.sock, {"op": "spec", "n": args.max_new, "prompt": args.prompt,
                             "spec_k": args.spec_k or 0})
        print(f"plain {r['plain_tok_s']} tok/s -> spec(K={r['k']}) {r['spec_tok_s']} tok/s  "
              f"= {r['speedup']}x   exact {r['exact']}")
        print(f"accept {r['accept']}")
    elif args.cmd == "bench":
        r = call(args.sock, {"op": "bench", "n": args.max_new, "prompt": args.prompt})
        print(f"decode: {r['ms_per_step']} ms/step  {r['decode_tok_s']} tok/s  (n={r['n']})")
    elif args.cmd == "profile":
        r = call(args.sock, {"op": "profile", "n": args.max_new, "prompt": args.prompt})
        print(f"{'kernel':<50}{'calls':>7}{'us/step':>10}{'%':>7}")
        for row in r["rows"]:
            print(f"{row['kernel']:<50}{row['calls']:>7}{row['us']:>10.1f}{row['pct']:>7.1f}")
        print(f"{'TOTAL':<50}{'':>7}{r['total_us']:>10.1f}")
    elif args.cmd == "ops":
        r = call(args.sock, {"op": "ops", "n": args.max_new, "prompt": args.prompt})
        print(f"{'op':<50}{'calls':>7}{'us/step':>10}{'%':>7}")
        for row in r["rows"]:
            print(f"{row['op']:<50}{row['calls']:>7}{row['us']:>10.1f}{row['pct']:>7.1f}")
        print(f"{'TOTAL':<50}{'':>7}{r['total_us']:>10.1f}")
    elif args.cmd == "generate":
        r = call(args.sock, {"op": "generate", "prompt": args.prompt,
                             "max_new": args.max_new, "temperature": args.temperature})
        print(r["text"])
        print(f"[{r['n_prompt']} prompt tok, {r['n_new']} new, "
              f"prefill {r['prefill_ms']}ms, decode {r['decode_tok_s']} tok/s]")
    elif args.cmd == "chat":
        msgs = []
        if args.system:
            msgs.append({"role": "system", "content": args.system})
        msgs.append({"role": "user", "content": args.user})
        tools = json.loads(Path(args.tools).read_text()) if args.tools else None
        r = call(args.sock, {"op": "chat", "messages": msgs, "tools": tools,
                             "enable_thinking": args.think, "max_new": args.max_new,
                             "temperature": args.temperature})
        if r.get("thinking"):
            print(f"[thinking] {r['thinking']}")
        print(r["text"])
        for c in r["tool_calls"]:
            print(f"[tool_call] {c['function']['name']} {c['function']['arguments']}")
        print(f"[{r['n_prompt']} prompt tok, {r['n_new']} new, "
              f"prefill {r['prefill_ms']}ms, decode {r['decode_tok_s']} tok/s]")


if __name__ == "__main__":
    main()
