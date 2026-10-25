"""3-D mRoPE: rotary tables for prompts that contain images.

Text-only mRoPE collapses to plain 1-D RoPE, which is why every decode path in
this engine indexes a ``[max_seq, rotary_dim]`` 1-D table with the token index
(``ops/rope.py``). A prompt with an image breaks that in two ways:

* the image's tokens are laid out in a 2-D (H, W) merged grid, so the three
  position axes (T, H, W) differ *inside* the image -- one table lookup cannot
  express that;
* after the image the image block ended at ``current_pos + max(h, w) / merge``
  while only ``h*w/merge**2`` tokens were consumed, so from then on the mRoPE
  position is ``token_index + rope_delta`` with a constant, negative
  ``rope_delta`` (:attr:`mm_input.MMInput.rope_delta`).

So there are exactly two things here:

* :func:`mrope_cos_sin` -- the interleaved 3-axis composition (the layout
  ``mrope_interleaved=true`` in the checkpoint means, and the only one
  ``Qwen3_5TextRotaryEmbedding`` implements). Half the frequencies come from T,
  then H and W alternate in, so ``i % 3 == 0 -> T``, ``1 -> H``, ``2 -> W``.
  ``cos`` is finally ``cat(half, half)`` -- the *same* layout the 1-D table uses,
  so ``rotate_half`` pairs ``i`` with ``i + rotary_dim/2`` and a text-only prompt
  yields bit-identical rows to ``ops.rope.make_rope_tables``;
* :func:`scatter_image_embeds` -- the pure-tensor splice that puts the tower's
  embeddings into a prefill chunk's token embeddings at *absolute* prompt indices.

The positions themselves (``[3, prompt_len]`` + ``rope_delta``) come from
:func:`mm_input.mrope_positions`, the integration surface;
``tests/test_mrope.py`` asserts that helper element-wise against the reference
``get_rope_index``, so this module can stay a pure "given positions, make angles"
layer.

Decode never comes through here: after prefill the three axes advance together,
so ``pos + rope_delta`` indexes the 1-D table and the captured graphs stay
captured.

Module: ``coffin/ops``. Depends on ``ops/rope`` for the shared frequency axis;
used by ``models/qwen35/model.py`` when a prefill chunk carries image embeds.

Constraints:
    * The stride-3 interleave and the 11/11/10 section split are fixed by the
      checkpoint's ``rope_parameters``; recomputing them differently changes the
      angles of image tokens only, which is easy to miss in a numeric test.
    * ``cos``/``sin`` must keep the ``cat(half, half)`` layout so ``rotate_half``
      pairs the same frequencies as the 1-D path.
    * ``scatter_image_embeds`` mutates ``x`` in place: the caller owns that
      buffer, and returning a copy instead would break the caller's aliasing
      assumptions.
"""

from __future__ import annotations

import torch

#: ``rope_parameters.mrope_section`` of Qwen3.5: 11/11/10, summing to
#: ``rotary_dim/2 = 32`` (the H/W sections are 11 and 10 because T takes the
#: multiples of three, i.e. 11 of the 32 frequencies).
MROPE_SECTION = (11, 11, 10)


def mrope_cos_sin(positions: torch.Tensor, inv_freq: torch.Tensor,
                  mrope_section: tuple[int, int, int] = MROPE_SECTION
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose ``[3, C]`` position axes into per-token cos/sin for image prompts.

    The composition is the reference one: T fills every frequency, then H
    overwrites ``slice(1, 3*section_h, 3)`` and W ``slice(2, 3*section_w, 3)`` --
    a stride-3 interleave in which frequency ``i`` comes from axis ``i % 3``
    (T, H, W, ...), H reaching index 31 and W index 29. Duplicating the 32
    composed angles to 64 is what makes the partial-rotary ``rotate_half`` pairs
    match the 1-D table exactly.

    Args:
        positions: Position axes, ``[3, C]`` with rows T, H, W; a single
            position triple as ``[3]`` is accepted too.
        inv_freq: ``[half]`` fp32 frequency axis from
            :func:`ops.rope.make_inv_freq`. Passing that shared axis is what
            keeps a text-only prompt bit-identical to the 1-D table path.
        mrope_section: Frequencies per axis, summing to ``half``.

    Returns:
        ``(cos, sin)``, each ``[C, 2*half]`` with the duplicated half layout.
    """
    pos = positions
    if pos.dim() == 1:                      # a single position triple
        pos = pos.reshape(3, 1)
    freqs = pos.to(torch.float32)[..., None] * inv_freq.to(pos.device)
    # freqs: [3, C, half]; T is the base, H/W overwrite their interleaved slots.
    out = freqs[0].clone()
    for dim, offset in ((1, 1), (2, 2)):
        length = mrope_section[dim] * 3
        idx = slice(offset, length, 3)
        out[..., idx] = freqs[dim][..., idx]
    c, s = out.cos(), out.sin()
    return torch.cat([c, c], -1), torch.cat([s, s], -1)


def scatter_image_embeds(x: torch.Tensor, embeds: torch.Tensor,
                         spans: list[tuple[int, int]], start: int, end: int) -> torch.Tensor:
    """Overwrite one prefill chunk's image rows with the vision tower's embeddings.

    Only the overlap of each span with the chunk is copied, and the offset into
    ``embeds`` is the span's own position in the concatenation -- so a chunk that
    starts or ends inside an image gets exactly its own rows, which is what makes
    chunked prefill, and a cached prefix boundary, work unchanged.

    Args:
        x: The chunk's token embeddings, ``[C, hidden]``, covering the absolute
            prompt positions ``[start, end)``. Mutated in place: the caller owns
            this freshly gathered buffer, and copying would only duplicate it.
        embeds: Every image's embeddings concatenated in prompt order.
        spans: Absolute ``[s, e)`` token runs of those images, also in prompt
            order, so the running offset into ``embeds`` stays aligned.
        start: First absolute prompt position of the chunk, inclusive.
        end: One past the last absolute prompt position of the chunk.

    Returns:
        ``x`` itself, for chaining. Unchanged when there is no image in the
        chunk, which is the text-only fast path.
    """
    if embeds is None or not spans:
        return x
    base = 0
    for s, e in spans:
        a, b = max(s, start), min(e, end)
        if a < b:
            x[a - start:b - start].copy_(
                embeds[base + (a - s):base + (b - s)].to(x.dtype))
        base += e - s
    return x
