"""Turn the checkpoint's ``[N, K]`` projections into the ``[K, N]`` layout this card streams at full bandwidth.

``Caps.weight_layout == "kn"`` is a promise the backend keeps on the loader's
behalf: the op contract stays in the checkpoint's ``[N, K]`` layout, and the one
transpose happens here instead of on every call. The layout is worth 14-22x on
decode on a BI-V100 (``[N, K]`` measured 19-30 GB/s against 362-410 GB/s for
``[K, N]``, COREX_PORT.md §0), so this file is a performance-critical part of the
backend rather than a reshaping helper.

Module: ``coffin/backends/corex``. Imports ``torch`` and the leaf module
``int8_weights``, and nothing else from ``coffin``. ``backends/corex/__init__.py``
calls :func:`as_kn` on every projection; a model loader that wants the resident
set to stay at one copy calls :func:`repack_state_dict` once at load.

Constraints:
    * Peak VRAM, not speed, decides how the transpose is done. Transposing a whole
      state dict with ``w.t().contiguous()`` holds both layouts at once and turns
      8.8 GB of bf16 weights into 17.6 GB; :func:`repack_state_dict` converts one
      tensor at a time and drops its source immediately, so the transient is twice
      the *largest single* weight (1.3 GB for ``lm_head``) and the resident set
      stays at one copy. An in-place transpose is not an option: it does not exist
      for a non-square matrix, and the chunked version has the same src+dst peak.
    * A raw ``[N, K]`` tensor is always read as the checkpoint layout. That is why
      the packed form is a wrapper type (:class:`KnWeight`) instead of a bare
      ``[K, N]`` tensor: shape alone cannot distinguish the two, and a square
      weight would transpose silently in the wrong direction.
    * :func:`as_kn` memoises per source tensor so a caller that ignores
      ``weight_layout`` still gets one transpose per weight rather than one per
      decode step. The cache holds only the packed copy and is keyed by object
      identity, so a weight is packed again only if it is a fresh tensor.
    * The kernel reads one dtype; :data:`WEIGHT_DTYPE` is that dtype. bf16 because
      the prefill GEMM reaches the same 32.6 TFLOP/s on bf16 and fp16 here, so the
      checkpoint's own dtype needs no conversion at load.
"""

from __future__ import annotations

import weakref
from typing import Any

import torch

from int8_weights import QWeight, is_quantizable

#: Storage dtype of a packed weight. Must agree with ``Caps.weight_dtype`` of this
#: backend; the kernels are compiled for the pair (weight dtype, bf16 activation).
WEIGHT_DTYPE = torch.bfloat16

#: Widest ``N`` that may be handed to the library GEMM. Measured on this card:
#: ``N = 131072`` is correct, ``N = 196608`` and ``N = 248320`` (``lm_head``) return
#: NaN. The prefill path chunks along ``N`` at this width; the decode path does not
#: call the library at all, so this bound shapes prefill only. It is a property of the
#: card's GEMM, not a tuning knob, which is why it lives next to the layout decision.
LIB_MAX_N = 131072

#: Ceiling on the identity cache. A model has a few hundred projections, so the
#: limit only ever trips in a diagnostic that packs freshly generated tensors in a
#: loop; the sweep then drops entries whose source has already been collected.
_CACHE_LIMIT = 4096

#: ``(id(source), dtype) -> (weakref to source, packed weight)``. Collected
#: eagerly through the weakref callback, so a weight that the loader drops does not
#: pin its packed copy.
_CACHE: dict[tuple[int, torch.dtype], tuple[weakref.ref, "KnWeight"]] = {}


class KnQWeight:
    """An ``[N, K]`` projection stored as int8 ``[K, N]`` plus one fp32 scale per column.

    The int8 counterpart of :class:`KnWeight`, and it exists for one reason: decode is
    bandwidth-bound, so halving the bytes read per weight is the only lever left on the
    projections, which dominate the step (``../../../PERF_COREX.md`` §8).

    The scale is per *output column*, which is the same thing as the checkpoint's per-row
    scale -- rows of ``[N, K]`` are columns of ``[K, N]``. It is stored alongside the
    matrix rather than multiplied into it, because applying it once per column after the
    ``K`` sum is both cheaper and what the quantiser's error model assumes.
    """

    __slots__ = ("scale", "wt", "n", "k")

    #: Distinguishes this form from :class:`KnWeight` without a type check at every call.
    int8 = True

    def __init__(self, wt: torch.Tensor, scale: torch.Tensor) -> None:
        """Wrap an already-transposed int8 matrix and its column scales.

        Args:
            wt: ``[K, N]`` contiguous int8 matrix.
            scale: ``[N]`` fp32 scales, one per output column.

        Raises:
            ValueError: ``wt`` is not 2-D int8, the scale is not 1-D fp32, or their
                lengths disagree. Each of those produces plausible numbers from the wrong
                elements instead of failing.
        """
        if wt.dim() != 2:
            raise ValueError(f"packed weight must be 2-D, got {tuple(wt.shape)}")
        if not wt.is_contiguous():
            raise ValueError("packed weight must be contiguous")
        if wt.dtype != torch.int8:
            raise ValueError(f"packed int8 weight must be int8, got {wt.dtype}")
        if scale.dim() != 1 or scale.dtype != torch.float32:
            raise ValueError("scale must be 1-D fp32")
        self.k, self.n = int(wt.shape[0]), int(wt.shape[1])
        if int(scale.numel()) != self.n:
            raise ValueError(f"scale holds {scale.numel()} values for N = {self.n}")
        self.wt = wt
        self.scale = scale

    def nbytes(self) -> int:
        """Resident size in bytes: the int8 matrix plus its fp32 column scales.

        Returns:
            ``K * N + 4 * N``.
        """
        return self.wt.numel() + self.scale.numel() * 4

    def to(self, dtype: torch.dtype) -> torch.Tensor:
        """Return the checkpoint layout ``[N, K]`` as a plain dequantised tensor.

        The escape hatch for paths this backend serves with the bf16 kernels or the
        reference implementation. It is a real dequantisation, so it costs the full fp
        size of the weight -- callers on the decode path must not use it.

        Args:
            dtype: Dtype the caller's arithmetic runs in.

        Returns:
            A fresh ``[N, K]`` contiguous tensor.
        """
        return (self.wt.float() * self.scale.view(1, -1)).to(dtype).t().contiguous()

    def dequant_kn(self, dtype: torch.dtype = WEIGHT_DTYPE) -> "KnWeight":
        """Dequantise in place of a transpose, keeping the ``[K, N]`` layout.

        The ``[K, N]`` sibling of :meth:`to`: this is what the bf16 kernels need, so a
        caller that cannot use the int8 path converts with one multiply instead of a
        transpose.

        Written as cast-then-scale, not scale-then-cast. The obvious spelling
        ``(wt.float() * scale).to(dtype)`` materialises a ``[K, N]`` fp32 temporary, twice
        the size of the answer it is immediately narrowed to; over this model's 128
        projections that is a *fixed* 174.8 ms per prefill request, which was 3.1x the
        whole prefill at n=128 (``../../../PERF_COREX.md`` §13.2b).

        ``mul_`` on a bf16 destination with an fp32 scale computes in fp32 and rounds
        once, the same single rounding the old spelling did, so the weights stay
        bit-identical. Narrowing the scale first (``scale.to(dtype)``) is 1.88x instead of
        1.70x but moves every weight by one bf16 ULP, and a weight feeds 32 layers; that
        is not a trade to make for 10% of one operator.

        Args:
            dtype: Storage dtype of the result.

        Returns:
            A :class:`KnWeight` over the dequantised matrix.

        Raises:
            TypeError: ``dtype`` is not a float type. Scaling into an integer destination
                wraps instead of raising, which surfaces as a corrupt checkpoint much
                further downstream.
        """
        if not dtype.is_floating_point:
            raise TypeError(f"dequant_kn needs a float dtype, got {dtype}")
        out = self.wt.to(dtype)
        out.mul_(self.scale.view(1, -1))
        return KnWeight(out)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"KnQWeight(K={self.k}, N={self.n}, int8)"


class KnWeight:
    """An ``[N, K]`` projection stored as the contiguous ``[K, N]`` matrix the kernels index.

    The wrapper, rather than a bare tensor, is what keeps the layout unambiguous:
    the op contract hands out ``[N, K]``, and a square weight would otherwise be
    transposed a second time without any shape change to give it away.
    """

    __slots__ = ("wt", "n", "k")

    #: Both packed forms answer these, so a caller can ask "is this int8" and "what is its
    #: scale" without an isinstance against every subclass.
    int8 = False
    scale = None

    def __init__(self, wt: torch.Tensor) -> None:
        """Wrap a matrix that is already transposed; nothing is copied here.

        Args:
            wt: ``[K, N]`` contiguous matrix. Ownership stays with the caller, so
                passing a view would make the kernels read the wrong geometry.

        Raises:
            ValueError: ``wt`` is not 2-D or not contiguous. Both produce plausible
                numbers from the wrong elements, which is the failure mode this
                class exists to prevent.
        """
        if wt.dim() != 2:
            raise ValueError(f"packed weight must be 2-D, got {tuple(wt.shape)}")
        if not wt.is_contiguous():
            raise ValueError("packed weight must be contiguous")
        self.wt = wt
        self.k, self.n = int(wt.shape[0]), int(wt.shape[1])

    def nbytes(self) -> int:
        """Resident size in bytes, for the loader's VRAM accounting.

        Returns:
            ``K * N * element_size``: the packed matrix and nothing else, because
            the source it was made from is expected to be dropped.
        """
        return self.wt.numel() * self.wt.element_size()

    def to(self, dtype: torch.dtype) -> torch.Tensor:
        """Return the checkpoint layout ``[N, K]`` in ``dtype``, as a plain tensor.

        This is the escape hatch for the one path that must not see the packed form:
        an op this backend has no kernel for is served by :mod:`backends.reference`,
        and its ``linear`` does ``w.to(dtype)`` on whatever it is handed. Without this
        method that fallback raises ``AttributeError`` in the middle of a decode step,
        which reads as a corrupt checkpoint rather than as an unimplemented kernel.

        The transpose is not cached: this path has already accepted a copy, and keeping
        one per weight would undo the single-copy property the packer exists for.

        Args:
            dtype: Dtype the caller's arithmetic runs in.

        Returns:
            A fresh ``[N, K]`` contiguous tensor.
        """
        return self.wt.t().to(dtype).contiguous()

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"KnWeight(K={self.k}, N={self.n}, {self.wt.dtype})"


def is_packed(w: Any) -> bool:
    """Report whether a weight is already in this backend's layout.

    Args:
        w: A projection weight of any accepted storage form.

    Returns:
        ``True`` for a :class:`KnWeight`. A plain tensor is always ``False``: the
        raw form is the contract's ``[N, K]``.
    """
    return isinstance(w, (KnWeight, KnQWeight))


def to_kn(w: torch.Tensor | QWeight | KnWeight | KnQWeight,
          dtype: torch.dtype = WEIGHT_DTYPE) -> KnWeight | KnQWeight:
    """Transpose one projection into the layout the kernels read.

    Args:
        w: ``[N, K]`` weight: a plain tensor in any float dtype, an
            ``int8_weights.QWeight`` (dequantised first, because this backend
            declares ``int8_weights=False``), or a :class:`KnWeight`, which is
            returned unchanged so the function is idempotent.
        dtype: Storage dtype of the packed matrix; matches ``caps.weight_dtype``.

    Returns:
        The ``[K, N]`` weight. It is a fresh allocation except when ``w`` is
        already packed.

    Raises:
        ValueError: ``w`` is not 2-D. A 1-D norm weight reaching here means a
            caller packed the wrong tensor, and transposing it would fail later
            with a shape error far from the cause.
    """
    if isinstance(w, (KnWeight, KnQWeight)):
        return w
    if isinstance(w, QWeight):
        # Already int8 in the checkpoint layout, so the cheap move is to keep it int8 and
        # transpose one byte per element rather than expanding to two bytes first. The
        # transpose peak is half of what the bf16 path pays for the same weight.
        wq = w.wq.t().contiguous()
        return KnQWeight(wq, w.scale.contiguous())
    if w.dim() != 2:
        raise ValueError(f"projection weight must be 2-D, got {tuple(w.shape)}")
    return KnWeight(w.to(dtype).t().contiguous())


def _evictor(key: tuple[int, torch.dtype]) -> Any:
    """Build the weakref callback that drops one cache entry.

    Args:
        key: Cache key the callback is responsible for.

    Returns:
        A callback that removes that key. Closing over the key rather than over the
        weight keeps the callback from extending the weight's lifetime.
    """

    def _drop(_dead: weakref.ref) -> None:
        _CACHE.pop(key, None)

    return _drop


def _sweep() -> None:
    """Drop cache entries whose source tensor has already been collected.

    The weakref callback normally removes an entry as soon as its source dies, but
    a callback delayed by the collector leaves a dead entry behind; the sweep makes
    the ceiling a hard bound rather than a suggestion.
    """
    for key, (ref, _packed) in list(_CACHE.items()):
        if ref() is None:
            _CACHE.pop(key, None)


def as_kn(w: torch.Tensor | QWeight | KnWeight | KnQWeight,
          dtype: torch.dtype = WEIGHT_DTYPE) -> KnWeight | KnQWeight:
    """Return the packed form of a weight, transposing it once per source object.

    The op table calls this on every projection. The cost of a transpose is
    milliseconds and the decode step is a fraction of one, so doing it per call
    would dominate the step; memoising by identity makes the first call pay and
    every later one a dictionary hit.

    Args:
        w: Projection weight in contract layout, or an already packed one.
        dtype: Storage dtype of the packed matrix.

    Returns:
        The packed weight. The returned object is the cache's own, so a caller must
        not mutate ``wt`` in place; a weight that is not weak-referenceable is
        packed per call instead of cached, which is correct but slow.

    Raises:
        ValueError: ``w`` is not 2-D; see :func:`to_kn`.
    """
    if isinstance(w, (KnWeight, KnQWeight)):
        return w
    source = w.wq if isinstance(w, QWeight) else w
    key = (id(source), dtype)
    hit = _CACHE.get(key)
    if hit is not None and hit[0]() is source:
        return hit[1]
    packed = to_kn(w, dtype)
    try:
        ref = weakref.ref(source, _evictor(key))
    except TypeError:
        # A tensor subclass with no ``__weakref__`` slot. Caching it would pin it
        # forever, so this one weight is repacked per call.
        return packed
    if len(_CACHE) >= _CACHE_LIMIT:
        _sweep()
    _CACHE[key] = (ref, packed)
    return packed


def repack_state_dict(sd: dict, dtype: torch.dtype = WEIGHT_DTYPE) -> dict[str, int]:
    """Transpose every projection of a loaded state dict in place, one at a time.

    This is the loader's entry point for ``weight_layout="kn"``: it keeps the
    resident set at one copy of the weights by dropping each source as its packed
    replacement takes its place, and it uses the same 2-D-ness rule as the int8
    quantiser, so the tensors it touches are exactly the projections and never the
    norms, ``A_log`` or the conv kernel.

    Args:
        sd: State dict of bf16 tensors and/or ``QWeight`` objects, consumed and
            rewritten in place. Tensors that are not projections are left alone.
        dtype: Storage dtype of the packed matrices.

    Returns:
        ``{"packed": n, "bytes_before": b, "bytes_after": a}`` for the loader's
        startup log. The byte counts cover only the converted entries, so the
        difference is what the conversion changed, not the model's total.

    Raises:
        ValueError: A projection is not 2-D; see :func:`to_kn`.
    """
    packed_count = 0
    bytes_before = 0
    bytes_after = 0
    for key in list(sd):
        w = sd[key]
        if isinstance(w, KnWeight):
            continue
        if not isinstance(w, QWeight) and not (
                isinstance(w, torch.Tensor) and is_quantizable(key, w)):
            continue
        bytes_before += w.nbytes() if isinstance(w, QWeight) else (
            w.numel() * w.element_size())
        packed = to_kn(w, dtype)
        sd[key] = packed
        packed_count += 1
        bytes_after += packed.nbytes()
        del w
    return {"packed": packed_count, "bytes_before": bytes_before,
            "bytes_after": bytes_after}


def unpacked_keys(sd: dict) -> list[str]:
    """Report which entries of a state dict are still in the checkpoint layout.

    Args:
        sd: State dict, as loaded or already repacked.

    Returns:
        The projection keys not yet packed, in dict order. Empty means the loader
        did its part; a non-empty answer after loading means every call for that
        weight pays a cached transpose instead of using the resident one.
    """
    return [key for key, w in sd.items()
            if not isinstance(w, KnWeight)
            and (isinstance(w, QWeight)
                 or (isinstance(w, torch.Tensor) and is_quantizable(key, w)))]
