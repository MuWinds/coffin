"""The gfx906 (Hygon DCU) backend: the contract adapter over this package's kernels.

S1 of the platform migration. The kernels and their host shim now live inside this
package (``backends/hip/kernels/`` and ``backends/hip/shim.py``); this module is
the *contract* adapter that makes them look like one of the platform backends, and
it is the only place in ``coffin`` allowed to know the arch string, the ROCm
environment variables and the split-K width.

All four of the shim's ``gemv*`` families collapse to the three ``linear*`` entries
here: ``R``/``U`` (rows per block, K-unroll), the wide-M kernels and the ``_dbg``
A/B entry points are kernel tiling detail, so they become an internal choice keyed
by shape instead of arguments in the contract.

Module: ``coffin/backends/hip``. Imports ``coffin.backends.protocol``,
``coffin.backends.reference`` (to fill ops with no kernel) and, *lazily*,
``coffin.backends.hip.shim`` -- lazily so importing this package on a host without
a ROCm toolchain does not set the build environment variables or import
``torch.utils.cpp_extension``.

Constraints:
    * Importing the shim has module-level side effects: it sets
      ``PYTORCH_ROCM_ARCH`` and ``TORCH_EXTENSIONS_DIR`` and imports the CUDA/HIP
      extension loader, which reads them at import time. It is therefore done
      inside :func:`_hip`, never at module scope: a CPU-only box that merely asks
      the registry to enumerate backends must not be reconfigured by the attempt,
      and a process that selected a CoreX card must not inherit a gfx906 arch.
    * The extension is built on first :func:`ops()` call (or first :func:`usable`
      call), not at import: a broken toolchain must degrade to the eager path
      instead of taking the process down, and the failure has to be memoised so it
      costs one attempt per process.
    * ``ATTN_SPLIT`` here must equal ``ATTN_SPLIT`` in ``kernels/attn.cu``: this
      side owns the split-K partial buffers, so a mismatch corrupts attention
      rather than failing a shape check. It moves to ``backends/hip/tuning.py``
      in S3 together with the buffer sizing.
    * Every entry keeps the eager path reachable: an op with no kernel returns
      ``None`` and :func:`ops` fills it from the reference backend, so a partially
      built extension still runs the model.
"""

from __future__ import annotations

import functools
from typing import Any

import torch
import torch.nn.functional as F

from backends import reference
from backends.protocol import Caps, Ops, OpFn, make_device
from int8_weights import QWeight

#: Backend identifier, i.e. what ``COFFIN_BACKEND=hip`` selects.
NAME = "hip"

#: Module-level alias, so this module satisfies :class:`~coffin.backends.protocol.Backend`.
name = NAME

#: Split-K width of ``kernels/attn.cu``. This is a kernel ABI: the caller allocates
#: ``part_acc``/``part_ml`` from it and the kernel indexes them with the same
#: width. It moves to ``backends/hip/tuning.py`` in S3.
ATTN_SPLIT = 8

#: Arch this backend's kernels are compiled for. Read from the driver at selection
#: time (:func:`caps`) and used as the fallback when the driver does not report
#: one, so the build flags never have to hardcode it at a call site.
DEFAULT_ARCH = "gfx906"

#: Row-block / K-unroll pairs the GEMV kernel instantiates. These are kernel tiling
#: parameters, not contract: they stay inside this module and are indexed by shape
#: in :data:`_TUNING`.
_GEMV_CFGS: tuple[tuple[int, int], ...] = ((1, 1), (2, 1), (1, 2), (2, 2), (1, 4), (4, 1))

#: The static default for a shape nobody has tuned yet. ``(1, 2)`` is what the
#: contract's old ``gemv`` default was, and it is within a few percent of the
#: per-shape optimum on the model's real shapes.
_DEFAULT_CFG = (1, 2)

#: Shape-keyed kernel choice, filled by :func:`_cfg`. Keyed by ``(N, K, is_int8)``
#: rather than by weight tensor identity so two layers with the same geometry share
#: one entry. It is a backend-internal cache: the model must never read or pass it.
_TUNING: dict[tuple[int, int, bool], tuple[int, int]] = {}


def _hip() -> Any:
    """Import this backend's kernel shim on first use.

    The shim lives beside this file (``backends/hip/shim.py``) rather than at the
    package root, so importing it is a private act of this backend. It sets
    ``PYTORCH_ROCM_ARCH`` / ``TORCH_EXTENSIONS_DIR`` at import, which is a
    process-wide side effect that must not happen just because something imported
    the backend package: on a CoreX host that would point the extension loader at
    a gfx906 arch.

    Returns:
        The shim module.

    Raises:
        ImportError: The shim or the torch extension loader is missing. The caller
            treats that as "no kernels", which is a supported state.
    """
    from backends.hip import shim
    return shim


def _is_rocm_build() -> bool:
    """Report whether this torch build links a ROCm/HIP runtime at all.

    ``torch.cuda.is_available()`` is not a vendor test. Several non-NVIDIA
    accelerators expose themselves through the *CUDA* namespace because they
    implement that API: the Iluvatar CoreX stack is one -- its own engine sets
    ``device="cuda"`` unconditionally -- and a Moore Threads or Hygon card is
    another. Probing on the namespace alone would let this backend claim a
    foreign card, report itself usable, and then either fail inside the first
    kernel launch or, worse, compile for ``--offload-arch=gfx906`` and run.

    ``torch.version.hip`` is the marker that separates the two: it is set to the
    HIP version on every ROCm build and is ``None`` on a CUDA build and on every
    CUDA-compatible vendor build this engine is aimed at. ``gcnArchName`` in
    ``get_device_properties`` is the second, device-side marker used by
    :func:`_arch`.

    Returns:
        ``True`` when the torch build itself is a HIP build.
    """
    return getattr(torch.version, "hip", None) is not None


def probe() -> bool:
    """Report whether a ROCm/HIP device is visible to torch.

    Only reads the driver: no kernel import, no build, no device reset. This runs
    on every host during auto-detect, including a CPU-only one.

    Both halves matter: a HIP build with no device attached is ``False``, and a
    CUDA-compatible card from another vendor is ``False`` too (see
    :func:`_is_rocm_build`). Auto-detect then falls through to that vendor's
    backend, or to the reference.

    Returns:
        ``True`` only when the build is a HIP build *and* a device is present.
    """
    try:
        return _is_rocm_build() and bool(torch.cuda.is_available())
    except Exception:                             # pragma: no cover - driver error
        return False


def _build_error() -> str | None:
    """Return the memoised kernel-build failure, attempting the build once.

    Returns:
        ``"<ExceptionType>: <message>"`` from the failed build, or ``None`` when
        the extension is loaded. The attempt is memoised inside the shim, so
        calling this again costs nothing.
    """
    try:
        module = _hip()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    if module.available():
        return None
    return module.last_error() or "the kernel extension did not build"


def usable() -> bool:
    """Report whether the kernels are built or cached, so the table can run native.

    Returns:
        ``False`` when there is no device or the extension failed to build; the
        registry then falls through to the next candidate, so a host with the
        driver but no working toolchain still serves the model through the
        reference backend.
    """
    return probe() and _build_error() is None


def reason() -> str:
    """Explain why :func:`usable` is ``False``, for the selection error text.

    Returns:
        ``""`` when usable, otherwise the device state or the build exception.
    """
    if not probe():
        return "no ROCm/HIP device visible (torch.cuda.is_available() is False)"
    err = _build_error()
    return "" if err is None else f"kernel build failed: {err}"


@functools.lru_cache(maxsize=1)
def _device_name() -> str:
    """Read the device name once; it is used in the banner and to pick the arch.

    Returns:
        ``torch.cuda.get_device_name(0)``, or ``""`` when there is no device. The
        call initialises the HIP runtime, so it is memoised and only reached once
        :func:`probe` has already said a device exists.
    """
    if not probe():
        return ""
    try:
        return str(torch.cuda.get_device_name(0))
    except Exception:                             # pragma: no cover - driver error
        return ""


@functools.lru_cache(maxsize=1)
def _arch() -> str:
    """Report the target ISA, read from the driver rather than hardcoded.

    ``gcnArchName`` is a ROCm-only field, which is why it doubles as the second
    half of the vendor test: reaching this function with a non-ROCm card under
    the ``cuda`` namespace means :func:`probe` already said no, and this returns
    ``""`` rather than inventing ``gfx906`` and compiling for it.

    Returns:
        ``get_device_properties(0).gcnArchName`` (``"gfx906"`` on the DCU) with
        the ``:sramecc``-style suffix stripped, or :data:`DEFAULT_ARCH` when the
        driver is ROCm but reports no arch. Build flags are derived from this
        value, so a different part needs a different string, not a second build
        path.
    """
    if not probe():
        return ""
    try:
        props = torch.cuda.get_device_properties(0)
        arch = getattr(props, "gcnArchName", "") or ""
    except Exception:                             # pragma: no cover - driver error
        arch = ""
    return str(arch).split(":")[0] or DEFAULT_ARCH


@functools.lru_cache(maxsize=1)
def caps() -> Caps:
    """Return this backend's capability record, probing the device once.

    Returns:
        A record with ``device="cuda"`` (torch's ROCm backend keeps that name),
        fp16 projection weights because gfx906 has packed fp16 math only, graph
        capture and load-time autotune both enabled, and ``recurrent_state=True``
        because snapshot/restore is plain ``copy_`` on this backend.
    """
    natives = frozenset(Ops.NAMES) if usable() else frozenset()
    return Caps(
        name=NAME,
        device="cuda",
        device_index=0,
        device_name=_device_name(),
        arch=_arch(),
        native=natives,
        act_dtype=torch.bfloat16,
        # fp16, not bf16: this part has packed fp16 math and no packed bf16 math, so
        # the prefill GEMM reaches 13-20 TFLOP/s on fp16 weights against
        # 2.3 TFLOP/s on bf16. Activations stay bf16 because the int8 KV cache and
        # every fused kernel are written against it.
        weight_dtype=torch.float16,
        graph=True,
        autotune=True,
        attn_split=ATTN_SPLIT,
        batch_attn=True,
        int8_weights=True,
        recurrent_state=True,
    )


@functools.lru_cache(maxsize=1)
def _resolve_device() -> torch.device:
    """Return the device every allocation for this backend goes on.

    Returns:
        ``torch.device("cuda", 0)``. Model code uses this instead of the
        ``device="cuda"`` literal.
    """
    return make_device(caps())


def __getattr__(name: str) -> Any:
    """Resolve the lazily-computed module attributes of the :class:`Backend` protocol.

    ``device`` cannot be a module-level constant: building the capability record
    reads the device name, which initialises the HIP runtime, and importing this
    module on a CPU-only host must not do that. Python only calls this hook for
    names the module does not define, so the normal attribute path is unaffected.

    Args:
        name: Attribute being looked up.

    Returns:
        The resolved attribute.

    Raises:
        AttributeError: The name is not one of the lazily-resolved attributes.
    """
    if name == "device":
        return _resolve_device()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _cfg(w: torch.Tensor | QWeight) -> tuple[int, int]:
    """Return the ``(rows_per_block, k_unroll)`` this shape should use.

    The choice is the backend's private business: it is cached by shape, never
    passed by a caller and never stored in the model family. :data:`_TUNING` is
    filled by the load-time sweep (S3) when ``caps().autotune`` is on; until then
    every shape takes :data:`_DEFAULT_CFG`.

    Args:
        w: The projection weight, plain or int8.

    Returns:
        The ``(R, U)`` pair to pass to the kernel.
    """
    n, k = (w.n, w.k) if isinstance(w, QWeight) else w.shape
    return _TUNING.get((n, k, isinstance(w, QWeight)), _DEFAULT_CFG)


def _split_k(part_acc: torch.Tensor | None, part_ml: torch.Tensor | None,
             nh: int, hd: int, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the split-K scratch, allocating it when the caller did not.

    Args:
        part_acc: Caller-owned accumulator, ``[nh, ATTN_SPLIT, hd]`` fp32, or
            ``None``.
        part_ml: Caller-owned running max/denominator, ``[nh, ATTN_SPLIT, 2]``
            fp32, or ``None``.
        nh: Query heads.
        hd: Head dim.
        like: A tensor whose device the allocation follows.

    Returns:
        ``(part_acc, part_ml)``. Allocating here is correct but not
        graph-capturable, which is why the model's capture path always passes its
        own buffers.
    """
    if part_acc is None:
        part_acc = torch.zeros(nh, ATTN_SPLIT, hd, dtype=torch.float32,
                               device=like.device)
    if part_ml is None:
        part_ml = torch.zeros(nh, ATTN_SPLIT, 2, dtype=torch.float32,
                              device=like.device)
    return part_acc, part_ml


def _split_k_batch(part_acc: torch.Tensor | None, part_ml: torch.Tensor | None,
                   rows: int, nh: int, hd: int,
                   like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row variant of :func:`_split_k` for the batched attention entries.

    Args:
        part_acc: Caller-owned ``[rows, nh, ATTN_SPLIT, hd]`` fp32, or ``None``.
        part_ml: Caller-owned ``[rows, nh, ATTN_SPLIT, 2]`` fp32, or ``None``.
        rows: Leading row count of the buffers.
        nh: Query heads.
        hd: Head dim.
        like: A tensor whose device the allocation follows.

    Returns:
        ``(part_acc, part_ml)``, allocated when absent.
    """
    if part_acc is None:
        part_acc = torch.zeros(rows, nh, ATTN_SPLIT, hd, dtype=torch.float32,
                               device=like.device)
    if part_ml is None:
        part_ml = torch.zeros(rows, nh, ATTN_SPLIT, 2, dtype=torch.float32,
                              device=like.device)
    return part_acc, part_ml


# Projections


def linear(w: torch.Tensor | QWeight, x: torch.Tensor) -> torch.Tensor:
    """Decode-time weight read: one or a few activation rows against ``w``.

    Routes to the int8 kernel for a ``QWeight`` and the fp16/bf16 kernel for a
    plain tensor; a 2-D activation delegates to :func:`linear_m`, because the
    single-row kernel is compiled for one vector only.

    Args:
        w: ``[N, K]`` row-major weight, plain tensor or ``QWeight``.
        x: Activations, ``[K]`` or ``[C, K]``.

    Returns:
        ``[N]`` or ``[C, N]`` in ``x``'s dtype.

    Raises:
        RuntimeError: The extension is unavailable and this entry was reached
            directly. Through :func:`ops` it is filled by the reference instead,
            so this only happens if a caller reached past the table.
    """
    if x.dim() != 1:
        return linear_m(w, x)
    module = _hip()
    r, u = _cfg(w)
    if isinstance(w, QWeight):
        return module.gemv_w8a16(w.wq, w.scale, x, r, u)
    return module.gemv(w, x, r, u)


def linear_m(w: torch.Tensor | QWeight, x: torch.Tensor) -> torch.Tensor:
    """Small-M batch projection: one weight pass for every row of ``x``.

    Picks between the small-M kernel (``M`` in 2/4/8), the wide-M kernel for
    ``M`` in 16/24/32, and the prefill GEMM otherwise. All three are numerically
    equivalent to calling :func:`linear` per row, so the choice is invisible to
    the caller -- which is the reason one contract name covers them.

    Args:
        w: ``[N, K]`` row-major weight, plain tensor or ``QWeight``.
        x: Activations, ``[M, K]``.

    Returns:
        ``[M, N]`` in ``x``'s dtype.

    Raises:
        RuntimeError: The extension is unavailable.
    """
    module = _hip()
    m = x.shape[0]
    x = x.contiguous()
    int8 = isinstance(w, QWeight)
    if m in (2, 4, 8):
        _, u = _cfg(w)
        if int8:
            return module.gemv_w8a16_m(w.wq, w.scale, x, u)
        return module.gemv_m(w, x, u)
    if int8 and m in (16, 24, 32):
        _, u = _cfg(w)
        return module.gemv_w8a16_wide(w.wq, w.scale, x, u)
    return linear_prefill(w, x)


def linear_prefill(w: torch.Tensor | QWeight, x: torch.Tensor) -> torch.Tensor:
    """Prefill-chunk projection (large M), where a GEMM beats streaming the weight.

    Args:
        w: ``[N, K]`` row-major weight, plain tensor or ``QWeight``.
        x: Activations, ``[C, K]``.

    Returns:
        ``[C, N]`` in ``x``'s dtype. An int8 weight is dequantised to
        :attr:`Caps.weight_dtype` for the call: prefill is compute-bound, so one
        materialisation is cheaper than many dequantising kernel launches.
    """
    weight_dtype = caps().weight_dtype
    dense = w.dequant(weight_dtype) if isinstance(w, QWeight) else w.to(weight_dtype)
    return F.linear(x.to(weight_dtype), dense).to(x.dtype)


# Normalisation


def rms_norm(x: torch.Tensor, w: torch.Tensor, stride: int = 0,
             eps: float = 1e-6) -> torch.Tensor:
    """Zero-centred RMSNorm over the last axis of ``[M, N]``, via the fused kernel.

    Args:
        x: Rows to normalise, ``[M, N]``; may be a column slice, in which case the
            rows are pitched wider than ``N``.
        w: Gain, ``[N]``, already carrying the ``1 + w`` offset.
        stride: Element pitch between rows; ``0`` reads it from ``x``.
        eps: Variance floor.

    Returns:
        Normalised rows, same shape and dtype as ``x``.

    Raises:
        RuntimeError: The extension is unavailable.
    """
    module = _hip()
    stride = x.stride(0) if stride == 0 else stride
    return module.rms_norm(x, w, stride, eps)


def rms_norm_gated(x: torch.Tensor, w: torch.Tensor, gate: torch.Tensor,
                   eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm (per head) scaled by ``silu(gate)``, via the fused kernel.

    Args:
        x: Activations, ``[..., dv]``.
        w: Per-head gain, ``[dv]``: not the flattened width, because the norm is
            per head.
        gate: Gate tensor shaped like ``x``.
        eps: Variance floor.

    Returns:
        Same shape and dtype as ``x``.

    Raises:
        RuntimeError: The extension is unavailable.
    """
    module = _hip()
    return module.rms_norm_gated(x, w, gate, eps)


# Attention


def attn_decode(q: torch.Tensor, gate: torch.Tensor, kcache: torch.Tensor,
                vcache: torch.Tensor, kscale: torch.Tensor, vscale: torch.Tensor,
                out: torch.Tensor, pos: torch.Tensor, rep: int, scale: float,
                part_acc: torch.Tensor | None = None,
                part_ml: torch.Tensor | None = None) -> torch.Tensor:
    """One decode token over the dense int8 KV cache, via the fused split-K kernel.

    Args:
        q: Query, ``[nh, hd]``, already rotated and scaled by the caller.
        gate: Output gate, ``[nh, hd]``; the kernel applies ``sigmoid``.
        kcache: int8 keys, ``[nkv, max_seq, hd]``.
        vcache: int8 values, ``[nkv, max_seq, hd]``.
        kscale: fp32 K scale, ``[nkv, max_seq]``.
        vscale: fp32 V scale, ``[nkv, max_seq]``.
        out: Output buffer, ``[nh, hd]``, written in place.
        pos: 1-element long tensor; the token index and the causal bound. The
            kernel masks past it rather than shrinking the loop, which is what
            keeps the captured shape static.
        rep: Query heads per KV head.
        scale: ``hd ** -0.5``.
        part_acc: Split-K accumulator, ``[nh, ATTN_SPLIT, hd]`` fp32. ``None``
            allocates, which is correct but not graph-capturable.
        part_ml: Split-K running max/denominator, ``[nh, ATTN_SPLIT, 2]`` fp32.

    Returns:
        ``out``, for call chaining.

    Raises:
        RuntimeError: The extension is unavailable.
    """
    module = _hip()
    pa, pm = _split_k(part_acc, part_ml, q.shape[0], q.shape[1], q)
    return module.attn(q.contiguous(), gate.contiguous(), kcache, vcache,
                       kscale, vscale, out, pos, rep, scale, pa, pm)


def attn_paged(q: torch.Tensor, gate: torch.Tensor, kpool: torch.Tensor,
               vpool: torch.Tensor, kscale: torch.Tensor, vscale: torch.Tensor,
               pos2slot: torch.Tensor, out: torch.Tensor, pos: torch.Tensor,
               max_pos: int, rep: int, scale: float,
               part_acc: torch.Tensor | None = None,
               part_ml: torch.Tensor | None = None) -> torch.Tensor:
    """One decode token over a layer's paged int8 pool.

    Args:
        q: Query, ``[nh, hd]``.
        gate: Output gate, ``[nh, hd]``.
        kpool: This layer's int8 K pool, ``[nslots, nkv, hd]``.
        vpool: This layer's int8 V pool, ``[nslots, nkv, hd]``.
        kscale: fp32 K scale, ``[nslots, nkv]``.
        vscale: fp32 V scale, ``[nslots, nkv]``.
        pos2slot: This row's slot map; only the first ``pos + 1`` entries are read.
        out: Output buffer, ``[nh, hd]``, written in place.
        pos: 1-element long tensor; the token index and the causal bound.
        max_pos: Window length, i.e. the row stride of the slot map.
        rep: Query heads per KV head.
        scale: ``hd ** -0.5``.
        part_acc: Split-K accumulator, ``[nh, ATTN_SPLIT, hd]`` fp32.
        part_ml: Split-K running max/denominator, ``[nh, ATTN_SPLIT, 2]`` fp32.

    Returns:
        ``out``, for call chaining.

    Raises:
        RuntimeError: The extension is unavailable.
    """
    module = _hip()
    pa, pm = _split_k(part_acc, part_ml, q.shape[0], q.shape[1], q)
    return module.attn_paged(q.contiguous(), gate.contiguous(), kpool, vpool,
                             kscale, vscale, pos2slot, out, pos, max_pos, rep,
                             scale, pa, pm)


def attn_paged_batch(q: torch.Tensor, gate: torch.Tensor, kpool: torch.Tensor,
                     vpool: torch.Tensor, kscale: torch.Tensor, vscale: torch.Tensor,
                     slots: torch.Tensor, context_lens: torch.Tensor,
                     out: torch.Tensor, max_seq: int, rep: int, scale: float,
                     part_acc: torch.Tensor | None = None,
                     part_ml: torch.Tensor | None = None) -> torch.Tensor:
    """One decode token for a bucket of rows, one launch for the whole bucket.

    Args:
        q: Queries, ``[B, nh, hd]``.
        gate: Output gates, ``[B, nh, hd]``.
        kpool: This layer's int8 K pool, ``[nslots, nkv, hd]``.
        vpool: This layer's int8 V pool, ``[nslots, nkv, hd]``.
        kscale: fp32 K scale, ``[nslots, nkv]``.
        vscale: fp32 V scale, ``[nslots, nkv]``.
        slots: Per-row slot maps, ``[B, max_seq]``.
        context_lens: Per-row key count, ``[B]`` int32.
        out: Output buffer, ``[B, nh, hd]``, written in place.
        max_seq: Window length, i.e. the row stride of ``slots``.
        rep: Query heads per KV head.
        scale: ``hd ** -0.5``.
        part_acc: Split-K accumulator, ``[B, nh, ATTN_SPLIT, hd]`` fp32.
        part_ml: Split-K running max/denominator, ``[B, nh, ATTN_SPLIT, 2]`` fp32.

    Returns:
        ``out``, for call chaining.

    Raises:
        RuntimeError: The extension is unavailable.
    """
    module = _hip()
    rows, nh, hd = q.shape[0], q.shape[1], q.shape[2]
    pa, pm = _split_k_batch(part_acc, part_ml, rows, nh, hd, q)
    return module.attn_paged_batch(q, gate, kpool, vpool, kscale, vscale, slots,
                                   context_lens, out, max_seq, rep, scale, pa, pm)


def attn_paged_multi_q(q: torch.Tensor, gate: torch.Tensor, kpool: torch.Tensor,
                       vpool: torch.Tensor, kscale: torch.Tensor, vscale: torch.Tensor,
                       slots: torch.Tensor, ctx_lens: torch.Tensor,
                       out: torch.Tensor, max_seq: int, rep: int, scale: float,
                       part_acc: torch.Tensor | None = None,
                       part_ml: torch.Tensor | None = None,
                       block_mask: torch.Tensor | None = None,
                       mask_block: int = 64) -> torch.Tensor:
    """Speculative verify: ``c`` draft queries per row share one pass over the history KV.

    Args:
        q: Draft queries, ``[B, c, nh, hd]``.
        gate: Output gates, same shape.
        kpool: This layer's int8 K pool, ``[nslots, nkv, hd]``.
        vpool: This layer's int8 V pool, ``[nslots, nkv, hd]``.
        kscale: fp32 K scale, ``[nslots, nkv]``.
        vscale: fp32 V scale, ``[nslots, nkv]``.
        slots: Per-row slot maps, ``[B, max_seq]``.
        ctx_lens: Per-row history length, ``[B]`` int32.
        out: Output buffer, ``[B, c, nh, hd]``, written in place.
        max_seq: Window length, i.e. the row stride of ``slots``.
        rep: Query heads per KV head.
        scale: ``hd ** -0.5``.
        part_acc: Split-K accumulator, ``[B*c, nh, ATTN_SPLIT, hd]`` fp32.
        part_ml: Split-K running max/denominator, ``[B*c, nh, ATTN_SPLIT, 2]`` fp32.
        block_mask: Optional uint8 ``[B, n_blocks]`` keep-mask (1 = keep).
        mask_block: Block size the mask is expressed in.

    Returns:
        ``out``, for call chaining.

    Raises:
        RuntimeError: The extension is unavailable.
    """
    module = _hip()
    rows, nh, hd = q.shape[0] * q.shape[1], q.shape[2], q.shape[3]
    pa, pm = _split_k_batch(part_acc, part_ml, rows, nh, hd, q)
    return module.attn_paged_multi_q(q, gate, kpool, vpool, kscale, vscale, slots,
                                     ctx_lens, out, max_seq, rep, scale, pa, pm,
                                     block_mask, mask_block)


# Gated DeltaNet


def conv1d_silu(conv_w: torch.Tensor, x: torch.Tensor, conv_state: torch.Tensor,
                out: torch.Tensor) -> torch.Tensor:
    """Depthwise causal conv (kernel 4) + SiLU for one token, shifting ``conv_state``.

    Args:
        conv_w: Per-channel taps, ``[C, 4]`` fp32.
        x: This token's projection, ``[C]``.
        conv_state: Previous ``3`` tokens, ``[C, 3]`` fp32; shifted in place.
        out: Output buffer, ``[C]`` fp32, written in place.

    Returns:
        ``out``, for call chaining.

    Raises:
        RuntimeError: The extension is unavailable.
    """
    return _hip().conv1d_silu(conv_w, x, conv_state, out)


def conv1d_silu_seq(conv_w: torch.Tensor, x: torch.Tensor,
                    conv_state: torch.Tensor) -> torch.Tensor:
    """The same conv over a prefill chunk, carrying ``conv_state`` across chunks.

    Args:
        conv_w: Per-channel taps, ``[C, 4]`` fp32.
        x: Chunk projections, ``[T, C]``.
        conv_state: Previous ``3`` tokens, ``[C, 3]`` fp32; shifted in place.

    Returns:
        ``[T, C]`` fp32 conv+SiLU output.

    Raises:
        RuntimeError: The extension is unavailable.
    """
    return _hip().conv1d_silu_seq(conv_w, x, conv_state)


def conv1d_silu_batch(conv_w: torch.Tensor, x: torch.Tensor,
                      conv_state: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Batched depthwise causal conv + SiLU, one row per batch entry.

    Args:
        conv_w: Per-channel taps, ``[C, 4]`` fp32.
        x: One token per row, ``[B, C]``.
        conv_state: Per-row windows, ``[B, C, 3]`` fp32; shifted in place.
        out: Output buffer, ``[B, C]`` fp32, written in place.

    Returns:
        ``out``, for call chaining.

    Raises:
        RuntimeError: The extension is unavailable.
    """
    return _hip().conv1d_silu_batch(conv_w, x, conv_state, out)


def delta_prologue(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   b: torch.Tensor, a: torch.Tensor,
                   neg_expA: torch.Tensor, dt_bias: torch.Tensor,
                   q_out: torch.Tensor, k_out: torch.Tensor, v_out: torch.Tensor,
                   beta: torch.Tensor, g: torch.Tensor,
                   qscale: float, rep: int) -> None:
    """Fused q/k norm, head expand and beta/g build for one token.

    Args:
        q: Query per key head, ``[num_k, dk]``.
        k: Key per key head, ``[num_k, dk]``.
        v: Value per value head, ``[num_v, dv]``.
        b: Beta logits, ``[num_v]``.
        a: Decay logits, ``[num_v]``.
        neg_expA: ``-exp(A_log)``, ``[num_v]`` fp32.
        dt_bias: Decay bias, ``[num_v]`` fp32.
        q_out: Output query, ``[num_v, dk]`` fp32.
        k_out: Output key, ``[num_v, dk]`` fp32.
        v_out: Output value, ``[num_v, dv]`` fp32.
        beta: Output write strength, ``[num_v]`` fp32.
        g: Output log decay ``<= 0``, ``[num_v]`` fp32.
        qscale: ``dk ** -0.5``.
        rep: Value heads per key head.

    Returns:
        Nothing. Every output buffer is written in place.

    Raises:
        RuntimeError: The extension is unavailable.
    """
    _hip().delta_prologue(q, k, v, b, a, neg_expA, dt_bias, q_out, k_out, v_out,
                          beta, g, qscale, rep)


def delta_prologue_batch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         b: torch.Tensor, a: torch.Tensor,
                         neg_expA: torch.Tensor, dt_bias: torch.Tensor,
                         q_out: torch.Tensor, k_out: torch.Tensor,
                         v_out: torch.Tensor, beta: torch.Tensor,
                         g: torch.Tensor, qscale: float, rep: int) -> None:
    """Batched :func:`delta_prologue`, a leading row dim on every argument.

    Args:
        q: Query per key head, ``[B, num_k, dk]``.
        k: Key per key head, ``[B, num_k, dk]``.
        v: Value per value head, ``[B, num_v, dv]``.
        b: Beta logits, ``[B, num_v]``.
        a: Decay logits, ``[B, num_v]``.
        neg_expA: ``-exp(A_log)``, ``[num_v]``, shared by every row.
        dt_bias: Decay bias, ``[num_v]``, shared by every row.
        q_out: Output query, ``[B, num_v, dk]`` fp32.
        k_out: Output key, ``[B, num_v, dk]`` fp32.
        v_out: Output value, ``[B, num_v, dv]`` fp32.
        beta: Output write strength, ``[B, num_v]`` fp32.
        g: Output log decay ``<= 0``, ``[B, num_v]`` fp32.
        qscale: ``dk ** -0.5``.
        rep: Value heads per key head.

    Returns:
        Nothing. Buffers are written in place.

    Raises:
        RuntimeError: The extension is unavailable.
    """
    _hip().delta_prologue_batch(q, k, v, b, a, neg_expA, dt_bias, q_out, k_out,
                                v_out, beta, g, qscale, rep)


def delta_prep_seq(qkv: torch.Tensor, b: torch.Tensor, a: torch.Tensor,
                   neg_expA: torch.Tensor, dt_bias: torch.Tensor,
                   q_out: torch.Tensor, k_out: torch.Tensor, v_out: torch.Tensor,
                   beta: torch.Tensor, g: torch.Tensor,
                   qscale: float, rep: int) -> None:
    """Prefill head prep, writing the head-major layout the chunked scan reads.

    Args:
        qkv: Conv output, ``[T, conv_dim]`` fp32.
        b: Beta logits, ``[T, num_v]``.
        a: Decay logits, ``[T, num_v]``.
        neg_expA: ``-exp(A_log)``, ``[num_v]`` fp32.
        dt_bias: Decay bias, ``[num_v]`` fp32.
        q_out: Output query, ``[num_v, T, dk]`` fp32.
        k_out: Output key, ``[num_v, T, dk]`` fp32.
        v_out: Output value, ``[num_v, T, dv]`` fp32.
        beta: Output write strength, ``[num_v, T]`` fp32.
        g: Output log decay ``<= 0``, ``[num_v, T]`` fp32.
        qscale: ``dk ** -0.5``.
        rep: Value heads per key head.

    Returns:
        Nothing. Every output buffer is written in place.

    Raises:
        RuntimeError: The extension is unavailable.
    """
    _hip().delta_prep_seq(qkv, b, a, neg_expA, dt_bias, q_out, k_out, v_out,
                          beta, g, qscale, rep)


def delta(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor,
          g: torch.Tensor, st: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """One gated delta decode step, updating ``st`` in place.

    Args:
        q: Normalised query, ``[num_v, dk]`` fp32.
        k: Normalised key, ``[num_v, dk]`` fp32.
        v: Value, ``[num_v, dv]`` fp32.
        beta: Write strength, ``[num_v]`` fp32.
        g: Log decay ``<= 0``, ``[num_v]`` fp32.
        st: Recurrent state ``[num_v, dk, dv]`` fp32, updated in place.
        out: Output buffer ``[num_v, dv]`` fp32, written in place.

    Returns:
        ``out``, for call chaining.

    Raises:
        RuntimeError: The extension is unavailable.
    """
    return _hip().delta(q, k, v, beta, g, st, out)


def delta_batch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                beta: torch.Tensor, g: torch.Tensor, st: torch.Tensor,
                out: torch.Tensor) -> torch.Tensor:
    """Batched gated delta step, one state row per batch entry.

    Args:
        q: Normalised query, ``[B, num_v, dk]`` fp32.
        k: Normalised key, ``[B, num_v, dk]`` fp32.
        v: Value, ``[B, num_v, dv]`` fp32.
        beta: Write strength, ``[B, num_v]`` fp32.
        g: Log decay ``<= 0``, ``[B, num_v]`` fp32.
        st: Recurrent state ``[B, num_v, dk, dv]`` fp32, updated in place.
        out: Output buffer ``[B, num_v, dv]`` fp32, written in place.

    Returns:
        ``out``, for call chaining.

    Raises:
        RuntimeError: The extension is unavailable.
    """
    return _hip().delta_batch(q, k, v, beta, g, st, out)


#: The 17 names this backend serves with a kernel. ``_NATIVE`` is what
#: :func:`caps` reports; it is spelled out so a missing kernel shows up as a
#: shorter ``native=N/17`` in the banner instead of silently degrading.
_NATIVE: dict[str, OpFn] = {
    "linear": linear,
    "linear_m": linear_m,
    "linear_prefill": linear_prefill,
    "rms_norm": rms_norm,
    "rms_norm_gated": rms_norm_gated,
    "attn_decode": attn_decode,
    "attn_paged": attn_paged,
    "attn_paged_batch": attn_paged_batch,
    "attn_paged_multi_q": attn_paged_multi_q,
    "conv1d_silu": conv1d_silu,
    "conv1d_silu_seq": conv1d_silu_seq,
    "conv1d_silu_batch": conv1d_silu_batch,
    "delta_prologue": delta_prologue,
    "delta_prologue_batch": delta_prologue_batch,
    "delta_prep_seq": delta_prep_seq,
    "delta": delta,
    "delta_batch": delta_batch,
}

#: Table with every kernel this backend has; the resolver fills anything absent.
_OPS = Ops.from_mapping(_NATIVE)


@functools.lru_cache(maxsize=1)
def ops() -> Ops:
    """Return this backend's total op table, with the extension loaded.

    The kernels are built (or attached to a cached build) on the first call, and
    the result is memoised: the failure memo lives in the shim, so a broken
    toolchain costs one attempt per process. Entries without a kernel are filled
    from the reference backend, which is what keeps the table total.

    Returns:
        A table with all 17 entries callable: this backend's kernels where they
        exist, the reference implementation for the rest. With no device, or with
        a failed build, the table is all reference, so a diagnostic caller can
        still inspect it; :func:`usable` is what stops such a table from ever
        reaching a request.
    """
    # ``probe()`` first, so a host with no device never reaches ``_build_error`` and
    # never imports the shim that sets the build environment variables.
    if not probe() or _build_error() is not None:
        return reference.fill(Ops(), caps())
    return reference.fill(_OPS, caps())


def reason_table() -> str:
    """Return the native/fallback split, for a diagnostic dump.

    Returns:
        One line naming which ops run native and which fall back to the
        reference, e.g. for ``test_backends`` to assert the split.
    """
    table = _OPS
    return (f"native={', '.join(table.implemented())}; "
            f"fallback={', '.join(table.missing()) or '(none)'}")
