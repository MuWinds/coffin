"""Multimodal (image) input plumbing shared by the server, the model and the pool.

One image request is described by an :class:`MMInput`: the image embeddings the
vision tower produced, laid out in the exact order the LLM's ``<|image_pad|>``
tokens appear in the prompt, plus the 3-D (T,H,W) mRoPE positions of the whole
prompt and the ``rope_delta`` that maps token index -> mRoPE position.

Why the prompt's token ids alone are not enough
-----------------------------------------------
Text-only mRoPE collapses to plain 1-D RoPE, so the engine could index a 1-D
cos/sin table with the token index and never care.  With an image in the prompt
that stops being true twice over:

* the image's tokens are *not* laid out linearly: three axes advance together and
  the image block ends at ``current_pos + max(h, w) // spatial_merge``, so the
  token index and the mRoPE position diverge by a constant after the image.  That
  constant is :attr:`MMInput.rope_delta`.
* inside the image the three axes differ (T is constant, H/W run over the merged
  grid), so a single 1-D table lookup cannot produce the right cos/sin.

``rope_delta`` is what lets decode stay a single captured graph: for text tokens
all three axes advance by one, so the position is ``token_index + rope_delta`` and
the existing 1-D tables still work.  Only the prefill chunks that actually
contain image tokens need the explicit 3-D positions.

Cache safety
------------
Two prompts with the same ``input_ids`` can embed *different* images (the
placeholder runs are identical by construction).  :attr:`MMInput.digest` is the
content hash that the radix tree salts block keys with, so a cached KV prefix is
only reused for the same image bytes.  An empty digest means "text only".
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
    """
    h = hashlib.sha256()
    for chunk in chunks:
        h.update(len(chunk).to_bytes(8, "little"))
        h.update(chunk)
    return h.hexdigest()


@dataclass
class MMInput:
    """Everything the model needs to know about one request's images."""

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
        return int(self.embeds.shape[0]) if self.embeds is not None else 0

    @property
    def num_images(self) -> int:
        return int(self.grid_thw.shape[0]) if self.grid_thw is not None else 0

    @property
    def has_image(self) -> bool:
        return self.num_images > 0

    def slice_embeds(self, start: int, end: int) -> torch.Tensor:
        """Embeddings for prompt positions ``[start, end)`` (absolute)."""
        out = []
        for (s, e) in self.spans:
            a, b = max(s, start), min(e, end)
            if a < b:
                out.append(self.embeds[a - s:b - s])
        if not out:
            return self.embeds[:0]
        return out[0] if len(out) == 1 else torch.cat(out, dim=0)

    def slice_positions(self, start: int, end: int) -> torch.Tensor:
        """mRoPE positions ``[3, end-start]`` for prompt positions ``[start, end)``."""
        return self.positions[:, start:end]


def find_image_spans(input_ids, image_token_id: int = IMAGE_TOKEN_ID
                     ) -> list[tuple[int, int]]:
    """Absolute ``[start, end)`` runs of ``image_token_id`` in ``input_ids``."""
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

    ``grids`` are the ``(t, h, w)`` patch grids from the image processor, in the
    same order as ``spans``.
    """
    import numpy as np

    # Two cursors, and they are NOT the same after an image: ``cur_tok`` indexes
    # the prompt (and the ``pos`` columns), ``cur_pos`` is the mRoPE position the
    # next token gets.  An image consumes ``max(h, w) // merge`` positions per
    # axis while taking up ``h*w/merge^2`` token slots, so the two diverge by
    # exactly ``rope_delta`` from the image onwards.  Using the token index as
    # the position (the first version of this function) silently zeroes
    # ``rope_delta`` and rotates every post-image token to the wrong angle.
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
