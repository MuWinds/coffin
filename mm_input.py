"""Multimodal (image) input plumbing shared by the server, the model and the pool.

One image request is described by an :class:`MMInput`: the image embeddings the
vision tower produced, laid out in the exact order the LLM's ``<|image_pad|>``
tokens appear in the prompt, plus the 3-D (T,H,W) mRoPE positions of the whole
prompt and the ``rope_delta`` that maps token index -> mRoPE position.

Module: ``coffin/mm_input.py``. Sits between the vision tower and the text
model: ``models/qwen35/vision.py`` produces the embeds, ``models/qwen35/model.py``
consumes ``MMInput`` on prefill, ``radix_cache.py`` reads ``digest`` to key its
blocks, and ``serve.py`` / ``openai_server.py`` build one per image request. It
imports ``torch`` only, so both the model and the cache layer can depend on it.

Why the prompt's token ids alone are not enough: text-only mRoPE collapses to
plain 1-D RoPE, so the engine could index a 1-D cos/sin table with the token
index and never care. With an image in the prompt that stops being true twice
over.

* The image's tokens are *not* laid out linearly: three axes advance together
  and the image block ends at ``current_pos + max(h, w) // spatial_merge``, so
  the token index and the mRoPE position diverge by a constant after the image.
  That constant is :attr:`MMInput.rope_delta`.
* Inside the image the three axes differ (T is constant, H/W run over the merged
  grid), so a single 1-D table lookup cannot produce the right cos/sin.

``rope_delta`` is what lets decode stay a single captured graph: for text tokens
all three axes advance by one, so the position is ``token_index + rope_delta``
and the existing 1-D tables still work. Only the prefill chunks that actually
contain image tokens need the explicit 3-D positions.

Cache safety: two prompts with the same ``input_ids`` can embed *different*
images (the placeholder runs are identical by construction).
:attr:`MMInput.digest` is the content hash that the radix tree salts block keys
with, so a cached KV prefix is only reused for the same image bytes. An empty
digest means "text only".

Constraints:
    * ``embeds`` is indexed by absolute prompt position through
      :attr:`MMInput.spans`, never by image-local offset. A chunked prefill
      starts mid-image, so a local offset would inject the wrong rows.
    * ``positions`` carries ``[3, prompt_len]`` for the *whole* prompt, because
      the positions of a chunk cannot be derived from that chunk's tokens.
    * Row 0/1/2 of ``positions`` are the T/H/W axes; ``ops/mrope.py`` interleaves
      them with stride 3 as ``[11,11,10]``, so a transposed or swapped row
      rotates image tokens only -- a numeric mismatch that no text test catches.
    * Block ids derive from the token ids and the digest: changing either
      invalidates cached KV for every prompt that shares the block.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import torch

#: ``<|image_pad|>`` in the Qwen3.5 vocabulary (config ``image_token_id``).
IMAGE_TOKEN_ID = 248056
#: ``<|vision_start|>`` / ``<|vision_end|>``; they are ordinary text tokens for
#: attention and position purposes, only the template uses them as delimiters.
VISION_START_ID = 248053
VISION_END_ID = 248054


def image_digest(chunks: list[bytes]) -> str:
    """Stable content hash of every image payload in one request.

    Hashing the *encoded image bytes* (not the pixels) is deliberate: it is what
    the client sent, it needs no decode, and two different PNGs of the same
    picture are treated as different media (which is the conservative choice for
    a KV cache).

    Args:
        chunks: Encoded image payloads in prompt order, as received.

    Returns:
        A hex SHA-256 over the payload lengths and bytes; ``""`` for no images
        (``MMInput`` then means text-only). Each length is hashed before its
        bytes so two payloads cannot splice into the same hash.
    """
    h = hashlib.sha256()
    for chunk in chunks:
        h.update(len(chunk).to_bytes(8, "little"))
        h.update(chunk)
    return h.hexdigest()


@dataclass
class MMInput:
    """Everything the model needs to know about one request's images.

    A text-only request is normally represented by ``mm=None`` at the call
    sites rather than by an empty instance, so the fields below are the image
    case. ``embeds`` and ``positions`` live in VRAM and are read once by the
    prefill chunk that needs them; nothing here is cached across requests.
    """

    #: ``[total_image_tokens, hidden]`` bf16, in prompt order.
    embeds: torch.Tensor
    #: ``[num_images, 3]`` ``(t, h, w)`` patch grids, in prompt order.
    grid_thw: torch.Tensor
    #: ``[3, prompt_len]`` mRoPE positions of the whole prompt (T, H, W rows).
    positions: torch.Tensor
    #: ``max(positions) + 1 - prompt_len``; 0 for a text-only prompt.
    rope_delta: int = 0
    #: Content hash of the image payloads (``""`` when text-only).
    digest: str = ""
    #: Absolute ``[start, end)`` token spans of each image, in prompt order.
    spans: list[tuple[int, int]] = field(default_factory=list)

    def __len__(self) -> int:
        """Embedding rows held, so a chunk can clamp its slice to them.

        Returns:
            The row count, or 0 when no image was decoded for this request.
        """
        return int(self.embeds.shape[0]) if self.embeds is not None else 0

    @property
    def num_images(self) -> int:
        """Image count, derived from the grids rather than tracked separately.

        Returns:
            The number of ``(t, h, w)`` grids, which is also how many entries
            ``spans`` has; 0 for a text-only request.
        """
        return int(self.grid_thw.shape[0]) if self.grid_thw is not None else 0

    @property
    def has_image(self) -> bool:
        """Whether any prompt token takes an embedding from the tower.

        Returns:
            ``True`` when the request carries at least one image.
        """
        return self.num_images > 0

    def slice_embeds(self, start: int, end: int) -> torch.Tensor:
        """Embeddings for prompt positions ``[start, end)`` (absolute).

        Args:
            start: First prompt position of the chunk, inclusive.
            end: One past the last prompt position of the chunk.

        Returns:
            Rows in prompt order, concatenated across images when the chunk
            straddles two of them. A chunk with no image tokens returns an empty
            ``[0, hidden]`` view instead of ``None``, so the caller can always
            index it.
        """
        out = []
        for (s, e) in self.spans:
            a, b = max(s, start), min(e, end)
            if a < b:
                out.append(self.embeds[a - s:b - s])
        if not out:
            return self.embeds[:0]
        return out[0] if len(out) == 1 else torch.cat(out, dim=0)

    def slice_positions(self, start: int, end: int) -> torch.Tensor:
        """mRoPE positions ``[3, end-start]`` for prompt positions ``[start, end)``.

        Args:
            start: First prompt position of the chunk, inclusive.
            end: One past the last prompt position of the chunk.

        Returns:
            The three axis rows for that slice; the caller needs the full
            ``[3, prompt_len]`` tensor to have been built first, since a slice
            cannot be reconstructed from its own tokens.
        """
        return self.positions[:, start:end]


def find_image_spans(input_ids, image_token_id: int = IMAGE_TOKEN_ID
                     ) -> list[tuple[int, int]]:
    """Absolute ``[start, end)`` runs of ``image_token_id`` in ``input_ids``.

    The chat template emits one placeholder per merged image patch, so a run is
    one image; the boundaries are what :class:`MMInput` and the embed scatter
    address, and a wrong boundary shifts every later image by a patch.

    Args:
        input_ids: Prompt token ids, as a sequence of ints.
        image_token_id: Id to treat as the placeholder; overridable so a
            different checkpoint vocabulary can reuse this.

    Returns:
        One ``(start, end)`` per run, in prompt order; empty when the prompt has
        no placeholder tokens, which is the text-only case.
    """
    spans: list[tuple[int, int]] = []
    start = -1
    for i, t in enumerate(input_ids):
        if t == image_token_id:
            if start < 0:
                start = i
        elif start >= 0:
            spans.append((start, i))
            start = -1
    if start >= 0:
        spans.append((start, len(input_ids)))
    return spans


def mrope_positions(spans: list[tuple[int, int]], grids, prompt_len: int,
                    spatial_merge: int = 2, device="cuda") -> tuple[torch.Tensor, int]:
    """3-D mRoPE positions of a prompt with image spans, in prompt order.

    Mirrors ``Qwen3_5Model.get_rope_index`` for the batch-1, no-padding case:
    text runs put all three axes at the same value (and advance the cursor by
    one per token); an image run puts T at the cursor and runs H/W over the
    ``(h, w)`` *merged* grid, then advances the cursor by
    ``max(h, w) // spatial_merge``.  ``rope_delta`` is then
    ``max(positions) + 1 - prompt_len``.

    Args:
        spans: Image spans from :func:`find_image_spans`, in prompt order.
        grids: ``(t, h, w)`` patch grids from the image processor, in the same
            order as ``spans``; a tensor or a list works.
        prompt_len: Total prompt tokens, generated tokens excluded.
        spatial_merge: Patch merge factor of the tower; the grids are divided by
            it to get the LLM-side grid.
        device: Where the returned positions tensor goes.

    Returns:
        ``(positions, rope_delta)``: ``[3, prompt_len]`` int64 positions and the
        scalar offset such that decode position is ``token_index + rope_delta``.
        A text-only prompt returns all three rows equal and ``rope_delta == 0``.

    Raises:
        ValueError: The placeholder run length disagrees with the merged grid.
            That is a template/processor mismatch rather than a bad request, so
            it is rejected here instead of producing silently misaligned
            positions.
    """
    import numpy as np

    # ``cur_tok`` indexes the prompt (and the ``pos`` columns); ``cur_pos`` is the
    # mRoPE position the next token gets. The two are NOT the same after an
    # image: it consumes ``max(h, w) // merge`` positions per axis while taking
    # up ``h*w/merge^2`` token slots, so they diverge by exactly ``rope_delta``
    # from the image onwards. Using the token index as the position (the first
    # version of this function) silently zeroes ``rope_delta`` and rotates every
    # post-image token to the wrong angle.
    pos = np.empty((3, prompt_len), dtype=np.int64)
    cur_tok = 0
    cur_pos = 0
    for gi, (start, end) in enumerate(spans):
        if start > cur_tok:                                  # text run
            n = start - cur_tok
            pos[:, cur_tok:start] = np.arange(n, dtype=np.int64) + cur_pos
            cur_tok, cur_pos = start, cur_pos + n
        grid = grids[gi].tolist() if hasattr(grids[gi], "tolist") else grids[gi]
        _t, gh, gw = (int(x) for x in grid)
        llm_h, llm_w = gh // spatial_merge, gw // spatial_merge
        n_img = end - start
        if n_img != llm_h * llm_w * max(_t, 1):
            raise ValueError(
                f"image {gi}: {n_img} placeholder tokens but grid {grid} implies "
                f"{llm_h * llm_w * max(_t, 1)}")
        t_row = np.full((llm_h * llm_w,), cur_pos, dtype=np.int64)
        h_row = np.repeat(np.arange(llm_h, dtype=np.int64), llm_w) + cur_pos
        w_row = np.tile(np.arange(llm_w, dtype=np.int64), llm_h) + cur_pos
        pos[0, start:end] = t_row
        pos[1, start:end] = h_row
        pos[2, start:end] = w_row
        cur_tok = end
        cur_pos = int(max(t_row.max(), h_row.max(), w_row.max())) + 1
    if cur_tok < prompt_len:                                 # trailing text
        n = prompt_len - cur_tok
        pos[:, cur_tok:] = np.arange(n, dtype=np.int64) + cur_pos
    phi = int(pos.max()) + 1 - prompt_len
    return torch.from_numpy(pos).to(device), phi
