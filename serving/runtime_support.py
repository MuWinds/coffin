"""Shared runtime constants and multimodal validation helpers.

Module: ``coffin.serving``. This module has no engine lifecycle state; it only
contains process defaults and image validation used by the runtime mixins.
"""

from __future__ import annotations

import base64
import io
import os

MODEL = os.environ.get("ENGINE_MODEL", "/root/private_data/models/Qwen3.5-4B")
SOCK = os.environ.get("ENGINE_SOCK", "/tmp/qwen35.sock")
MAX_SEQ = 262144
MAX_DECODE_PIXELS = 16 * 1024 * 1024

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


class MMError(Exception):
    """An image request this server refuses, carrying the HTTP status to answer.

    The status travels with the exception because the refusal happens deep below
    the transport: the op handlers copy it into the reply, and
    The OpenAI entry point maps it onto the HTTP status instead of calling every
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
