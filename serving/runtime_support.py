"""Shared runtime constants, device-runtime shims and multimodal validation helpers.

Module: ``coffin.serving``. This module has no engine lifecycle state; it only
contains process defaults, the device-runtime accessors every mixin shares, and
image validation used by the runtime mixins.

Constraints:
    * No vendor spelling: the device-runtime namespace and every VRAM/timing call
      are derived from ``backends.caps().device``, never from a literal. That is
      what lets the same server code start under ``COFFIN_BACKEND=reference`` on a
      machine with no accelerator, where the namespace simply does not exist.
    * Importing this module must not select a backend: the CLI reads ``MODEL`` /
      ``SOCK`` from here before it applies ``--backend``, so ``backends`` is
      imported inside the accessors rather than at module scope.
"""

from __future__ import annotations

import base64
import io
import os

import torch

MODEL = os.environ.get("ENGINE_MODEL", "/root/private_data/models/Qwen3.5-4B")
SOCK = os.environ.get("ENGINE_SOCK", "/tmp/qwen35.sock")
MAX_SEQ = 262144
MAX_DECODE_PIXELS = 16 * 1024 * 1024


def device_runtime():
    """Return torch's device-runtime namespace for the selected backend.

    Every timing barrier and VRAM readout in the serving layer needs the same
    namespace, and on this card that namespace is the one carrying the vendor
    spelling. Reading it off ``caps.device`` keeps that spelling inside the
    backend package, and returning ``None`` when there is no usable device is
    what lets the banner and the diagnostics run on a host with no accelerator
    instead of raising out of an unguarded device call.

    Presence of the namespace is not enough to decide. ``torch.cpu`` exists in
    every build and answers ``is_available()`` with ``True`` -- it means "the CPU
    is always there", not "a device is attached" -- and a vendor build that
    implements the CUDA API exposes its device module on a machine with no card.
    Both would pass a ``hasattr`` test and then fail the first ``synchronize``, so
    the test is the backend's own device name plus an availability probe.

    Returns:
        The ``torch.<caps.device>`` module, or ``None`` when the selected backend
        runs on the CPU, torch has no namespace by that name, or it reports no
        available device.
    """
    import backends

    caps = backends.caps()
    if caps.device == "cpu":
        return None
    runtime = getattr(torch, caps.device, None)
    if runtime is None:
        return None
    probe = getattr(runtime, "is_available", None)
    if probe is None:
        return None
    try:
        return runtime if probe() else None
    except Exception:  # noqa: BLE001 - a broken driver must not break the banner
        return None


def sync() -> None:
    """Block until every queued device work item has completed.

    Needed around a timing window: a host-side clock around asynchronous kernel
    launches measures the launch, not the work, so ``bench`` and ``profile`` would
    report a few microseconds per step. A CPU backend has no queue and no
    namespace, so this is a no-op there rather than an early return a caller could
    forget.
    """
    runtime = device_runtime()
    if runtime is not None:
        runtime.synchronize()


def empty_cache() -> None:
    """Release the caching allocator's free segments back to the driver.

    Called between a teardown and the next build: the pool and the captured
    graphs are the two largest allocations in the process, and the allocator
    otherwise keeps their segments, which is exactly the VRAM the rebuild needs.
    """
    runtime = device_runtime()
    if runtime is not None:
        runtime.empty_cache()


def allocated_gib() -> float:
    """Bytes the caching allocator currently holds live, in GiB.

    Returns:
        The total across all allocator pools, ``0.0`` on a backend without an
        allocator namespace. Rounded by the caller, since every readout is for a
        human watching a memory budget rather than for a comparison.
    """
    runtime = device_runtime()
    if runtime is None:
        return 0.0
    return runtime.memory_allocated() / 2**30


def reserved_gib() -> float:
    """Bytes the caching allocator has reserved from the driver, in GiB.

    Returns:
        The reserved total, ``0.0`` on a backend without an allocator namespace.
        Reported next to :func:`allocated_gib` because the difference is the
        fragmentation a rebuild has to fit into.
    """
    runtime = device_runtime()
    if runtime is None:
        return 0.0
    return runtime.memory_reserved() / 2**30


def free_bytes() -> int:
    """Free device memory the driver reports, in bytes.

    Returns:
        The free byte count, or ``-1`` when the backend has no query for it, so a
        caller can refuse to act on an unknown budget instead of treating it as
        zero.
    """
    runtime = device_runtime()
    if runtime is None:
        return -1
    try:
        free, _total = runtime.mem_get_info()
    except Exception:  # noqa: BLE001 - a backend may expose the namespace without it
        return -1
    return int(free)


def profiler_activities(*, device: bool = False) -> list:
    """Profiler activity list for the selected backend.

    Args:
        device: Also trace device kernels. Passed explicitly because the CPU half
            is what attributes a decode step to an aten op, and a device-only
            trace has neither the launch names nor the host time.

    Returns:
        ``[CPU]``, or ``[CPU, CUDA]`` when ``device`` is set and the backend is
        actually a device backend. The enum member exists in every torch build, so
        a CPU host would accept it and then report nothing -- which is why the
        backend is consulted instead of the symbol's presence.
    """
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device and device_runtime() is not None:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return activities


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
