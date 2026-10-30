"""The Iluvatar CoreX (BI-V100) backend: hand-written ``[K, N]`` kernels behind the op contract.

This is the second native backend in ``coffin``. The same production code runs on
the Hygon DCU and on this card; what changes is the op table, the weight layout and
the dtype policy, and all three belong to this package. Two measured facts shape
everything here (COREX_PORT.md §0): the card streams ``[K, N]`` weights at
168-361 GB/s while the checkpoint's ``[N, K]`` form reaches 19-30 GB/s, and its
``cuBLAS`` returns NaN on the ``lm_head`` width.

Module: ``coffin/backends/corex``. Imports ``backends.protocol``,
``backends.reference`` (to fill ops with no kernel yet), ``int8_weights`` and its own
``pack`` / ``kernels`` submodules -- nothing from ``coffin/ops``, ``models``,
``engine``, ``serving`` or ``cache``. Importing it must not build anything: the
extension is loaded on the first :func:`ops` or :func:`usable` call, so a CPU-only
host can enumerate backends without being asked for a toolchain.

Constraints:
    * ``caps.weight_layout == "kn"`` is a promise: every projection this backend runs
      wants ``[K, N]``. The contract's ``w`` stays ``[N, K]`` (that is the
      checkpoint), so :func:`pack.as_kn` transposes once per weight and memoises it;
      a loader that calls :func:`pack.repack_state_dict` pays it once and keeps one
      resident copy.
    * No library GEMM above ``pack.LIB_MAX_N`` output columns. The card measured NaN
      at ``N = 248320`` and correct results at ``N = 131072``, so the prefill path
      chunks along ``N`` and the decode path never calls the library at all. This is
      the difference between a wrong ``lm_head`` and a slow one.
    * Every kernel launch is on ``c10::cuda::getCurrentCUDAStream()`` (the ``.cu``
      files do it). A launch on the legacy stream is not part of a
      ``torch.cuda.graph`` capture, so a captured decode step would replay stale
      output -- a correctness bug that reads as a numerical one.
    * ``caps.graph`` and ``caps.batch_attn`` are derived from the op table rather than
      declared. A capture is only safe when no op falls back to the reference, because
      that fallback reaches ``cuBLAS``, which cannot be captured here; likewise
      ``batch_attn`` is "``attn_paged_batch`` has a kernel". Both become ``True`` on
      their own as the inventory lands, so neither can drift into a lie.
    * No fp64 anywhere. This device returns zeros for it, so every accumulator and every
      comparison stays at fp32 or narrower.

The kernel inventory grows one op at a time; an op with no kernel is left out of
:data:`_NATIVE` and :func:`ops` fills it from the reference backend, which is why a
partially implemented platform still runs the whole model.
"""

from __future__ import annotations

import functools
from typing import Any

import torch

from backends import reference
from backends.corex import kernels, pack
from backends.protocol import Backend, Caps, Ops, OpFn, make_device
from int8_weights import QWeight

#: Backend identifier, i.e. what ``COFFIN_BACKEND=corex`` selects.
NAME = "corex"

#: Module-level alias, so this module satisfies :class:`~coffin.backends.protocol.Backend`.
name = NAME

#: ISA the extension is compiled for when the driver reports nothing usable.
#: ``ivcore10`` is the BI-V100 target (COREX_PORT.md §0); a card that reports a
#: different arch string gets that one instead, so the build flag never has to be
#: edited for a new part.
DEFAULT_ARCH = "ivcore10"

#: Device-name markers that identify this vendor inside the shared ``cuda``
#: namespace. Matched case-insensitively as substrings: the vendor's own code
#: compares against the exact string ``"Iluvatar BI-V100"`` and therefore breaks on
#: a BI-V150, which is the mistake this tuple exists to avoid.
_NAME_MARKERS: tuple[str, ...] = ("iluvatar", "bi-v100", "bi-v150", "corex")

#: Activation count above which a projection is compute-bound and a library GEMM
#: beats streaming the weight. Below it the GEMV reads the weight once and the GEMM
#: reads it once *per tile*, which is why the two regimes are separate ops at all.
_PREFILL_MIN_M = 64

#: Rows per kernel launch when a shape has to avoid the library. :func:`_gemv_rows`
#: batches up to this many rows into one weight pass, matching what the small-M
#: kernel is specialised for.
_ROW_CHUNK = 8


def _is_corex_host() -> tuple[bool, str]:
    """Identify this vendor inside torch's shared ``cuda`` namespace.

    ``torch.cuda.is_available()`` is not a vendor test: ROCm, CoreX and a plain
    NVIDIA driver all answer it, and the ``cuda`` device type says nothing about who
    made the card. On a shared namespace the device *name* is the only
    discriminator, which is why the driver is asked for it here and why the ROCm
    backend additionally requires its own build marker.

    Returns:
        ``(matched, device_name)``. A failed driver call is ``(False, "")``: "cannot
        tell" must not become "mine", or a broken driver would be claimed as this
        backend and then fail inside a kernel launch.
    """
    if getattr(torch.version, "hip", None) is not None:
        # A ROCm build. Even if a CoreX card answered through it, the HIP backend is
        # the one that can build for a HIP ISA, so this backend steps aside.
        return (False, "")
    try:
        if not torch.cuda.is_available():
            return (False, "")
        device_name = str(torch.cuda.get_device_name(0))
    except Exception:                             # pragma: no cover - driver error
        return (False, "")
    lowered = device_name.lower()
    return (any(marker in lowered for marker in _NAME_MARKERS), device_name)


def probe() -> bool:
    """Report whether this host has an Iluvatar device.

    Only reads the driver: no extension build, no allocation, no device reset. The
    registry probes every candidate during auto-detect, including on a CPU-only box.

    Returns:
        ``True`` only when a CUDA-namespace device is visible *and* its name matches
        this vendor. Answering ``True`` on name alone would let this backend claim a
        foreign card and then compile for an ISA that card lacks.
    """
    return _is_corex_host()[0]


@functools.lru_cache(maxsize=1)
def _device_name() -> str:
    """Read the board name once, for the banner and the arch fallback.

    Returns:
        ``torch.cuda.get_device_name(0)``, or ``""`` when there is no device. The
        call initialises the runtime, so it is memoised and only reached once
        :func:`probe` has already reported this vendor.
    """
    if not probe():
        return ""
    return _is_corex_host()[1]


@functools.lru_cache(maxsize=1)
def _arch() -> str:
    """Report the ISA to compile for, preferring what the driver exposes.

    The arch is a build flag, so an invented value is worse than the measured
    default: a string the compiler does not recognise fails the build, and one it
    silently accepts would target the wrong ISA. The driver is therefore only used
    when it hands back a CoreX-shaped name; otherwise this returns
    :data:`DEFAULT_ARCH`, which is what the card was measured with
    (COREX_PORT.md §0).

    Returns:
        An ISA string for ``--cuda-gpu-arch=``, or ``""`` on a host that is not a
        CoreX host at all (a build for an absent card is never wanted).
    """
    if not probe():
        return ""
    try:
        props = torch.cuda.get_device_properties(0)
    except Exception:                             # pragma: no cover - driver error
        return DEFAULT_ARCH
    for field in ("arch", "gcnArchName", "device_arch"):
        value = getattr(props, field, None)
        if isinstance(value, str) and value.lower().startswith("ivcore"):
            return value.split(":")[0]
    return DEFAULT_ARCH


def _build_error() -> str | None:
    """Return the memoised extension-build failure, attempting the build once.

    Returns:
        ``"<ExceptionType>: <message>"`` from a failed build, ``None`` when the
        extension is loaded. The attempt is memoised inside
        :mod:`backends.corex.kernels`, so asking again costs nothing.
    """
    if kernels.available():
        return None
    return kernels.last_error() or "backends.corex.kernels.available() is False"


def usable() -> bool:
    """Report whether this backend can execute the model right now.

    Returns:
        ``True`` only when the device is present *and* the extension built. The
        registry falls through to the next candidate otherwise, so a host with the
        driver but no toolchain still serves the model through the reference backend
        instead of taking the process down at the first decode step.
    """
    return probe() and _build_error() is None


def reason() -> str:
    """Explain in one line why :func:`usable` is ``False``.

    Returns:
        ``""`` when usable, otherwise the device state or the build exception. Without
        it a reference fallback reads as a mysteriously slow deployment rather than a
        missing kernel build.
    """
    if not probe():
        seen = _is_corex_host()[1]
        detail = f" (got {seen!r})" if seen else ""
        return f"no Iluvatar CoreX device visible through torch.cuda{detail}"
    err = _build_error()
    return "" if err is None else f"kernel build failed: {err}"


def _kernels() -> Any:
    """Return the loaded extension, building it on first use.

    Returns:
        The extension module with the ``gemv_kn`` / ``gemv_kn_m`` / ``rms_norm`` /
        ``rms_norm_gated`` entry points.

    Raises:
        RuntimeError: The build failed or ``TORCH_EXTENSIONS_DIR`` is unset. Through
            :func:`ops` the affected entry is never reached on such a host, because
            the whole table is the reference implementation; a direct call means a
            caller reached past the table.
    """
    return kernels.load_kernels(_arch() or DEFAULT_ARCH)


def _compute_dtype(x: torch.Tensor) -> torch.Tensor:
    """Widen an activation to a dtype the kernels are instantiated for.

    The kernels are compiled for bf16 and fp32. Anything else (an fp16 activation
    from a caller holding old tensors) widens to fp32, which is lossless for every
    16-bit float, and the result is narrowed back by the caller. The reference
    backend is deliberately not the fallback here: its ``rms_norm`` calls
    ``F.rms_norm``, which does not exist on this cluster's torch 2.1.

    Args:
        x: Activation tensor of any float dtype.

    Returns:
        ``x`` itself when it is bf16 or fp32, otherwise an fp32 copy.
    """
    if x.dtype in (torch.bfloat16, torch.float32):
        return x
    return x.float()


def _restore(y: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Narrow a kernel result back to the caller's dtype.

    Args:
        y: Kernel output, bf16 or fp32.
        dtype: Dtype the caller passed in and expects back.

    Returns:
        ``y`` itself when it already has ``dtype``, otherwise a copy in ``dtype``.
    """
    return y if y.dtype == dtype else y.to(dtype)


# Projections


def linear(w: torch.Tensor | QWeight | pack.KnWeight, x: torch.Tensor) -> torch.Tensor:
    """Decode-time projection: one activation row against ``[N, K]`` ``w``.

    Args:
        w: ``[N, K]`` projection weight -- a plain tensor, a ``QWeight`` (kept int8
            and served by the W8A16 kernel, which is the point of declaring
            ``int8_weights``), or a packed weight the loader already transposed.
        x: Activations, ``[K]`` or ``[C, K]``.

    Returns:
        ``[N]`` or ``[C, N]`` in ``x``'s dtype.

    Raises:
        RuntimeError: The extension is unavailable and this entry was reached
            directly; through :func:`ops` the reference implementation answers
            instead.
    """
    if x.dim() != 1:
        return _gemv_rows(w, x)
    packed = pack.as_kn(w)
    module = _kernels()
    dtype = x.dtype
    act = _compute_dtype(x).contiguous()
    if packed.int8:
        return _restore(module.gemv_kn_i8(packed.wt, act, packed.scale), dtype)
    return _restore(module.gemv_kn(packed.wt, act), dtype)


def linear_m(w: torch.Tensor | QWeight | pack.KnWeight, x: torch.Tensor) -> torch.Tensor:
    """Small-M projection: ``M`` rows of activations read the weight once.

    Args:
        w: ``[N, K]`` projection weight in any accepted storage form.
        x: Activations, ``[M, K]``.

    Returns:
        ``[M, N]`` in ``x``'s dtype. Large ``M`` routes to :func:`linear_prefill`,
        because past the small-M regime a GEMM is the cheaper shape even though it
        reads the weight once per tile.

    Raises:
        RuntimeError: The extension is unavailable.
    """
    if x.shape[0] > _ROW_CHUNK:
        return linear_prefill(w, x)
    return _gemv_rows(w, x)


def _gemv_rows(w: torch.Tensor | QWeight | pack.KnWeight,
               x: torch.Tensor) -> torch.Tensor:
    """Project up to :data:`_ROW_CHUNK` rows, in chunks, through the self-written GEMV.

    A separate helper because three entries need it: the single-row entry when the
    caller passes a batch, the small-M entry, and the wide-``N`` prefill path that
    must not touch the library. Chunking at :data:`_ROW_CHUNK` keeps every launch on
    a specialisation the kernel actually has.

    Args:
        w: ``[N, K]`` projection weight in any accepted storage form.
        x: Activations, ``[C, K]``, ``C >= 1``.

    Returns:
        ``[C, N]`` in ``x``'s dtype.
    """
    packed = pack.as_kn(w)
    module = _kernels()
    dtype = x.dtype
    act = _compute_dtype(x).contiguous()
    rows = act.shape[0]
    if packed.int8:
        # One weight pass per chunk. Until gemv_kn_i8_m existed this looped the single-row
        # kernel, so a row batch read the whole weight once per row: measured, M=8 cost
        # 8.19x M=1 (probe_row_cost.py, five decode shapes), which is why a c-token
        # speculative verify paid c full weight passes and could not break even at any draft
        # depth (../../../PERF_COREX.md section 14).
        #
        # The kernel takes a width it was instantiated for, and this helper is reached with
        # any row count up to _ROW_CHUNK (a prefill chunk that fell back here, for instance),
        # so the chunking the bf16 branch below does has to happen here too: handing 24 rows
        # to a kernel whose TORCH_CHECK says [1, 8] raises, which is what this branch did
        # before the chunk loop was added.
        if rows <= _ROW_CHUNK:
            return _restore(module.gemv_kn_i8_m(packed.wt, act, packed.scale), dtype)
        out = torch.empty((rows, packed.n), dtype=act.dtype, device=act.device)
        for start in range(0, rows, _ROW_CHUNK):
            stop = min(rows, start + _ROW_CHUNK)
            out[start:stop] = module.gemv_kn_i8_m(packed.wt, act[start:stop].contiguous(),
                                                  packed.scale)
        return _restore(out, dtype)
    if rows == 1:
        return _restore(module.gemv_kn(packed.wt, act[0]), dtype)
    if rows <= _ROW_CHUNK:
        return _restore(module.gemv_kn_m(packed.wt, act), dtype)
    out = torch.empty((rows, packed.n), dtype=act.dtype, device=act.device)
    for start in range(0, rows, _ROW_CHUNK):
        stop = min(rows, start + _ROW_CHUNK)
        chunk = act[start:stop]
        if chunk.shape[0] == 1:
            out[start] = module.gemv_kn(packed.wt, chunk[0])
        else:
            out[start:stop] = module.gemv_kn_m(packed.wt, chunk.contiguous())
    return _restore(out, dtype)


def linear_prefill(w: torch.Tensor | QWeight | pack.KnWeight,
                   x: torch.Tensor) -> torch.Tensor:
    """Prefill-chunk projection (large ``M``), where a GEMM beats streaming the weight.

    The library is only used inside the width this card is known to compute
    correctly: above :data:`pack.LIB_MAX_N` output columns its GEMM returned NaN at
    ``N = 248320`` (the ``lm_head``), which is the single most damaging shape to hand
    it. A wide-``N`` prefill therefore runs the self-written ``[K, N]`` GEMV chunked
    over rows: bandwidth-bound rather than compute-bound, but finite.

    Args:
        w: ``[N, K]`` projection weight in any accepted storage form.
        x: Activations, ``[C, K]``.

    Returns:
        ``[C, N]`` in ``x``'s dtype.

    Raises:
        RuntimeError: The extension is unavailable and a wide-``N`` shape needed it.
    """
    packed = pack.as_kn(w)
    rows = x.shape[0]
    if packed.int8:
        # int8 exists for the decode GEMV, where halving the bytes dominates. A prefill
        # GEMM consumes the weight many times per tile and the library cannot take an
        # int8 operand here, so the wide path dequantises once and uses the bf16 GEMM.
        # That keeps one storage form in VRAM instead of two copies of every weight.
        packed = packed.dequant_kn(pack.WEIGHT_DTYPE)
    if rows < _PREFILL_MIN_M:
        # Below the prefill regime the GEMM reads the weight once per tile and loses
        # to the GEMV, which reads it exactly once for the whole call.
        return _gemv_rows(w, x)
    dtype = x.dtype
    act = _compute_dtype(x)
    weight_dtype = packed.wt.dtype
    # The GEMM reads both operands in one dtype, and this card reaches the same
    # TFLOP/s on bf16 as on fp16 (32.6 at M=2048), so the weights' own dtype decides
    # and the activations follow it rather than the other way round.
    lhs = act.to(weight_dtype)
    if packed.n <= pack.LIB_MAX_N:
        return _restore(torch.matmul(lhs, packed.wt), dtype)
    out = torch.empty((rows, packed.n), dtype=weight_dtype, device=x.device)
    for start in range(0, packed.n, pack.LIB_MAX_N):
        stop = min(packed.n, start + pack.LIB_MAX_N)
        # A column slice of a ``[K, N]`` matrix is not contiguous; the copy is
        # unavoidable and is amortised over ``M`` rows. The result is copied out
        # rather than written through ``out=`` because a column slice of ``out`` is
        # not contiguous either, and ``matmul`` rejects that as an output.
        block = packed.wt[:, start:stop].contiguous()
        out[:, start:stop].copy_(torch.matmul(lhs, block))
    return _restore(out, dtype)


# Normalisation


def _pitch_rows(x: torch.Tensor, width: int) -> torch.Tensor:
    """View ``x`` as ``[rows, width]`` at its own row pitch, without a copy.

    ``x.reshape(-1, width)`` is the obvious spelling and it is wrong here: a column
    slice of a wider buffer is not contiguous, so reshape silently copies, and the
    pitch handed to the kernel then describes the copy rather than the buffer. The
    answer stays numerically right, which is what makes the wasted bandwidth easy to
    miss.

    Args:
        x: Activations whose last axis is ``width``.
        width: Last-axis length.

    Returns:
        A ``[rows, width]`` view when the rows are uniformly pitched -- the case the
        contract describes -- otherwise a reshaped (possibly copied) tensor.
    """
    if x.dim() == 2 and x.stride(-1) == 1:
        return torch.as_strided(x, (x.shape[0], width), (x.stride(0), 1))
    return x.reshape(-1, width)


def rms_norm(x: torch.Tensor, w: torch.Tensor, stride: int = 0,
             eps: float = 1e-6) -> torch.Tensor:
    """Zero-centred RMSNorm over the last axis of ``x``, via the fused kernel.

    Args:
        x: Activations, ``[..., N]``; may be a column slice of a wider buffer, in
            which case the rows are pitched further apart than ``N``.
        w: Gain, ``[N]``, already carrying the checkpoint's ``1 + w`` offset.
        stride: Element pitch between rows; ``0`` reads it from ``x``.
        eps: Variance floor.

    Returns:
        Same shape and dtype as ``x``.

    Raises:
        RuntimeError: The extension is unavailable.
    """
    module = _kernels()
    dtype = x.dtype
    width = x.shape[-1]
    act = _compute_dtype(x)
    rows = _pitch_rows(act, width)
    gain = w if w.dtype == act.dtype else w.to(act.dtype)
    y = module.rms_norm(rows, gain.contiguous(),
                        rows.stride(0) if stride == 0 else stride, eps)
    return _restore(y.view(x.shape), dtype)


def rms_norm_gated(x: torch.Tensor, w: torch.Tensor, gate: torch.Tensor,
                   eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm (per head) scaled by ``silu(gate)``, via the fused kernel.

    Args:
        x: Activations, ``[..., dv]``.
        w: Per-head gain, ``[dv]``: not the flattened width, because the norm is per
            head.
        gate: Gate tensor shaped like ``x``.
        eps: Variance floor.

    Returns:
        Same shape and dtype as ``x``.

    Raises:
        RuntimeError: The extension is unavailable.
    """
    module = _kernels()
    dtype = x.dtype
    width = x.shape[-1]
    act = _compute_dtype(x)
    gate_c = _compute_dtype(gate)
    gain = w if w.dtype == act.dtype else w.to(act.dtype)
    y = module.rms_norm_gated(act.reshape(-1, width), gain.contiguous(),
                              gate_c.reshape(-1, width), eps)
    return _restore(y.view(x.shape), dtype)


def add_rms_norm(x: torch.Tensor, add: torch.Tensor, w: torch.Tensor,
                 out: torch.Tensor, stride: int = 0,
                 eps: float = 1e-6) -> torch.Tensor:
    """Residual add into ``x`` then the zero-centred RMSNorm, in one launch.

    The upstream spelling is two launches and two reads of the residual stream;
    fusing them is worth a launch per call at this card's 5.75 us floor and a
    read of the whole ``[hidden]`` vector.

    Args:
        x: Residual stream, ``[..., N]``, accumulated in place.
        add: Mixer output with ``x``'s shape.
        w: Gain, ``[N]``, already carrying the checkpoint's ``1 + w`` offset.
        out: Destination for the normalised sum, ``x``'s shape. Must not alias
            ``x``: the residual has to survive the norm.
        stride: Element pitch between rows; ``0`` reads it from ``x``.
        eps: Variance floor.

    Returns:
        ``out``, for call chaining.

    Raises:
        RuntimeError: The extension is unavailable.
    """
    module = _kernels()
    width = x.shape[-1]
    act = _compute_dtype(x)
    act_add = _compute_dtype(add)
    rows = _pitch_rows(act, width)
    rows_add = _pitch_rows(act_add, width)
    gain = w if w.dtype == act.dtype else w.to(act.dtype)
    y = module.add_rms_norm(rows, rows_add, gain.contiguous(), out, 
                            rows.stride(0) if stride == 0 else stride, eps)
    return y.view(x.shape) if y.dtype == out.dtype else y.to(out.dtype).view(out.shape)


def kv_quant(x: torch.Tensor, out_q: torch.Tensor,
             out_s: torch.Tensor) -> torch.Tensor:
    """Symmetric int8 quantisation of ``[M, N]``, one fp32 scale per row, one launch.

    Replaces the seven-op torch chain ``_quant`` uses, which costs about 82 us per
    call against this card's 5.75 us launch floor and runs twice per
    full-attention layer.

    Args:
        x: Activations to quantise, ``[M, N]``.
        out_q: int8 destination, ``[M, N]``, written in place.
        out_s: fp32 destination for the per-row scale, ``[M]``, written in place.

    Returns:
        ``out_q``, for call chaining.

    Raises:
        RuntimeError: The extension is unavailable.
    """
    module = _kernels()
    act = _compute_dtype(x)
    width = act.shape[-1]
    return module.kv_quant(_pitch_rows(act, width), out_q, out_s)


def silu_mul(gate: torch.Tensor, up: torch.Tensor,
             out: torch.Tensor) -> torch.Tensor:
    """``out = silu(gate) * up``, the SwiGLU product, in one launch.

    Nothing is copied to make an operand contiguous, and that is load-bearing rather than
    tidy. ``GEMVGroup.__call__`` hands the two halves back as ``.split(sizes, dim=1)`` of one
    fused result, so on every 2-D path (prefill, batch, wide) ``gate`` and ``up`` are
    *column views* of a wider buffer and are not contiguous. Densifying them here would add
    two ``_to_copy`` launches per MLP to save two -- three launches where ``F.silu(gate) *
    up`` was two, and the regression would appear only on the prefill paths, which are not
    the ones being measured. The kernel takes a per-operand row pitch instead, so a strided
    column view is served in place.

    Args:
        gate: Gate half of the fused projection, ``[...]`` with unit innermost stride.
        up: Up half, same shape as ``gate``, unit innermost stride (need not be contiguous).
        out: Destination, same shape. May alias ``up``.

    Returns:
        ``out``, for call chaining.

    Raises:
        RuntimeError: The extension is unavailable.
    """
    module = _kernels()
    if gate.dim() == 1:
        return module.silu_mul(gate, up, out)
    width = gate.shape[-1]
    return module.silu_mul(_pitch_rows(gate, width), _pitch_rows(up, width),
                           _pitch_rows(out, width))


def rope_apply(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor,
               sin: torch.Tensor, q_out: torch.Tensor, k_out: torch.Tensor,
               rotary_dim: int) -> torch.Tensor:
    """Rotate the leading ``rotary_dim`` channels of every q/k head, in one launch.

    Args:
        q: Queries, ``[M, nh, hd]``.
        k: Keys, ``[M, nkv, hd]``.
        cos: Cosine for the rows being rotated, ``[M, rotary_dim]`` fp32.
        sin: Sine, same layout as ``cos``.
        q_out: Destination of the rotated queries, ``q``'s shape.
        k_out: Destination of the rotated keys, ``k``'s shape.
        rotary_dim: Channels to rotate.

    Returns:
        ``q_out``, for call chaining.

    Raises:
        RuntimeError: The extension is unavailable.
    """
    return _kernels().rope_apply(q, k, cos, sin, q_out, k_out, rotary_dim)


# Attention


def _split_k(part_acc: torch.Tensor | None, part_ml: torch.Tensor | None,
             shape: tuple[int, ...], like: torch.Tensor) -> tuple[torch.Tensor,
                                                                  torch.Tensor]:
    """Return the split-K scratch, allocating it when the caller did not.

    Allocating here is correct but not graph-capturable: the capture path holds buffer
    addresses fixed, so a fresh allocation per replay would leave the graph writing to a
    dead tensor. The model's capture path therefore always passes its own buffers.

    Args:
        part_acc: Caller-owned numerator, or ``None``.
        part_ml: Caller-owned running max/denominator, or ``None``.
        shape: Leading dimensions before the split axis, e.g. ``(nh,)`` or ``(B, nh)``.
        like: A tensor whose device the allocation follows.

    Returns:
        ``(part_acc, part_ml)`` sized for this backend's ``attn_split``.
    """
    split = caps().attn_split
    if part_acc is None:
        part_acc = torch.zeros(*shape, split, like.shape[-1], dtype=torch.float32,
                               device=like.device)
    if part_ml is None:
        part_ml = torch.zeros(*shape, split, 2, dtype=torch.float32, device=like.device)
    return part_acc, part_ml


def attn_decode(q: torch.Tensor, gate: torch.Tensor, kcache: torch.Tensor,
                vcache: torch.Tensor, kscale: torch.Tensor, vscale: torch.Tensor,
                out: torch.Tensor, pos: torch.Tensor, rep: int, scale: float,
                part_acc: torch.Tensor | None = None,
                part_ml: torch.Tensor | None = None) -> torch.Tensor:
    """One decode token over the dense int8 KV cache, via the fused kernel.

    Args:
        q: Query, ``[nh, hd]``, already rotated and scaled by the caller.
        gate: Output gate, ``[nh, hd]``; the kernel applies ``sigmoid``.
        kcache: int8 keys, ``[nkv, max_seq, hd]``.
        vcache: int8 values, ``[nkv, max_seq, hd]``.
        kscale: fp32 K scale, ``[nkv, max_seq]``.
        vscale: fp32 V scale, ``[nkv, max_seq]``.
        out: Output buffer, ``[nh, hd]``, written in place.
        pos: 1-element long tensor; the token index and the causal bound. Read on the
            device, so a captured replay sees the live position instead of the one
            frozen at capture.
        rep: Query heads per KV head.
        scale: ``hd ** -0.5``.
        part_acc: Split-K accumulator, ``[nh, split, hd]`` fp32; ``None`` allocates.
        part_ml: Split-K running max/denominator, ``[nh, split, 2]`` fp32.

    Returns:
        ``out``, for call chaining.

    Raises:
        RuntimeError: The extension is unavailable.
    """
    module = _kernels()
    pa, pm = _split_k(part_acc, part_ml, (q.shape[0],), q)
    return module.attn_decode(q.contiguous(), gate.contiguous(), kcache, vcache, kscale,
                              vscale, out, pos, rep, scale, pa, pm)


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
        part_acc: Split-K accumulator, ``[nh, split, hd]`` fp32.
        part_ml: Split-K running max/denominator, ``[nh, split, 2]`` fp32.

    Returns:
        ``out``, for call chaining.

    Raises:
        RuntimeError: The extension is unavailable.
    """
    module = _kernels()
    pa, pm = _split_k(part_acc, part_ml, (q.shape[0],), q)
    return module.attn_paged(q.contiguous(), gate.contiguous(), kpool, vpool, kscale,
                             vscale, pos2slot, out, pos, max_pos, rep, scale, pa, pm)


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
        part_acc: Split-K accumulator, ``[B, nh, split, hd]`` fp32.
        part_ml: Split-K running max/denominator, ``[B, nh, split, 2]`` fp32.

    Returns:
        ``out``, for call chaining.

    Raises:
        RuntimeError: The extension is unavailable.
    """
    module = _kernels()
    rows, nh = q.shape[0], q.shape[1]
    pa, pm = _split_k(part_acc, part_ml, (rows, nh), q)
    # Dense operands, made so here rather than trusted from the caller. The kernel indexes ``q`` and
    # ``gate`` through ``data_ptr()`` as one flat ``[B, nh, hd]`` block, so a column view silently
    # reads the wrong bytes -- and the natural way to build the gate is exactly such a view
    # (``qg[..., hd:]`` of a fused ``[.., 2*hd]`` projection, whose last stride is 1 but whose row
    # pitch is ``2*hd``). Measured: with the strided gate this entry disagreed with
    # ``attn_paged_multi_q`` on 3834/4096 elements of layer 3 while the query and the pool bytes were
    # bit-identical, which read as a kernel defect across several sessions
    # (``docs/STAGE_LOCALISED.md``). The copy is a no-op when the caller already passes dense
    # tensors, which is why it belongs at the boundary instead of at every call site.
    return module.attn_paged_batch(q.contiguous(), gate.contiguous(), kpool, vpool, kscale,
                                   vscale, slots, context_lens.contiguous(), out, max_seq,
                                   rep, scale, pa, pm)


def attn_paged_multi_q(q: torch.Tensor, gate: torch.Tensor, kpool: torch.Tensor,
                       vpool: torch.Tensor, kscale: torch.Tensor, vscale: torch.Tensor,
                       slots: torch.Tensor, ctx_lens: torch.Tensor, out: torch.Tensor,
                       max_seq: int, rep: int, scale: float,
                       part_acc: torch.Tensor | None = None,
                       part_ml: torch.Tensor | None = None,
                       block_mask: torch.Tensor | None = None,
                       mask_block: int = 64) -> torch.Tensor:
    """Speculative verify: ``c`` draft queries per row share one pass over the history.

    Args:
        q: Draft queries, ``[B, c, nh, hd]``.
        gate: Output gates, same shape.
        kpool: This layer's int8 K pool, ``[nslots, nkv, hd]``.
        vpool: This layer's int8 V pool, ``[nslots, nkv, hd]``.
        kscale: fp32 K scale, ``[nslots, nkv]``.
        vscale: fp32 V scale, ``[nslots, nkv]``.
        slots: Per-row slot maps, ``[B, max_seq]``.
        ctx_lens: Per-row history length, ``[B]`` int32; query ``j`` sees
            ``ctx_lens[b] + j + 1`` keys, so it is identical to decoding that token
            alone.
        out: Output buffer, ``[B, c, nh, hd]``, written in place.
        max_seq: Window length, i.e. the row stride of ``slots``.
        rep: Query heads per KV head.
        scale: ``hd ** -0.5``.
        part_acc: Split-K accumulator, ``[B*c, nh, split, hd]`` fp32.
        part_ml: Split-K running max/denominator, ``[B*c, nh, split, 2]`` fp32.
        block_mask: Optional uint8 ``[B, n_blocks]`` keep-mask (1 = keep).
        mask_block: Block size the mask is expressed in.

    Returns:
        ``out``, for call chaining.

    Raises:
        RuntimeError: The extension is unavailable.
    """
    module = _kernels()
    rows, draft, nh = q.shape[0], q.shape[1], q.shape[2]
    pa, pm = _split_k(part_acc, part_ml, (rows * draft, nh), q)
    mask = block_mask if block_mask is not None else torch.empty(
        0, dtype=torch.uint8, device=q.device)
    # Dense operands at the boundary, for the reason given on ``attn_paged_batch``: the kernel
    # indexes the gate through ``data_ptr()`` as one flat block, and every caller builds it as a
    # column slice of a fused projection.
    return module.attn_paged_multi_q(q.contiguous(), gate.contiguous(), kpool, vpool, kscale,
                                     vscale, slots, ctx_lens.contiguous(), out, max_seq,
                                     rep, scale, pa, pm, mask, mask_block)


# Gated DeltaNet


def conv1d_silu(conv_w: torch.Tensor, x: torch.Tensor, conv_state: torch.Tensor,
                out: torch.Tensor) -> torch.Tensor:
    """Depthwise causal conv (kernel 4) + SiLU for one token, shifting the window.

    Args:
        conv_w: Per-channel taps, ``[C, 4]`` fp32.
        x: This token's projection, ``[C]``.
        conv_state: The previous 3 tokens, ``[C, 3]`` fp32; shifted in place.
        out: Output buffer, ``[C]`` fp32, written in place.

    Returns:
        ``out``, for call chaining.

    Raises:
        RuntimeError: The extension is unavailable.
    """
    _kernels().conv1d_silu(conv_w, x, conv_state, out)
    return out


def conv1d_silu_seq(conv_w: torch.Tensor, x: torch.Tensor,
                    conv_state: torch.Tensor) -> torch.Tensor:
    """The same conv over a prefill chunk, carrying the window across chunks.

    Args:
        conv_w: Per-channel taps, ``[C, 4]`` fp32.
        x: Chunk projections, ``[T, C]``.
        conv_state: The previous 3 tokens, ``[C, 3]`` fp32; updated in place.

    Returns:
        ``[T, C]`` fp32 conv + SiLU output.

    Raises:
        RuntimeError: The extension is unavailable.
    """
    return _kernels().conv1d_silu_seq(conv_w, x, conv_state)


def conv1d_silu_batch(conv_w: torch.Tensor, x: torch.Tensor, conv_state: torch.Tensor,
                      out: torch.Tensor) -> torch.Tensor:
    """Batched depthwise causal conv + SiLU, one window per row.

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
    _kernels().conv1d_silu_batch(conv_w, x, conv_state, out)
    return out


def delta_prologue(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, b: torch.Tensor,
                   a: torch.Tensor, neg_expA: torch.Tensor, dt_bias: torch.Tensor,
                   q_out: torch.Tensor, k_out: torch.Tensor, v_out: torch.Tensor,
                   beta: torch.Tensor, g: torch.Tensor, qscale: float,
                   rep: int) -> None:
    """Fused q/k norm, head expansion and the beta/g gates for one token.

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
        Nothing; every output buffer is written in place.

    Raises:
        RuntimeError: The extension is unavailable.
    """
    _kernels().delta_prologue(q, k, v, b, a, neg_expA, dt_bias, q_out, k_out, v_out,
                              beta, g, qscale, rep)


def delta_prologue_batch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         b: torch.Tensor, a: torch.Tensor, neg_expA: torch.Tensor,
                         dt_bias: torch.Tensor, q_out: torch.Tensor, k_out: torch.Tensor,
                         v_out: torch.Tensor, beta: torch.Tensor, g: torch.Tensor,
                         qscale: float, rep: int) -> None:
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
        Nothing; buffers are written in place.

    Raises:
        RuntimeError: The extension is unavailable.
    """
    _kernels().delta_prologue_batch(q, k, v, b, a, neg_expA, dt_bias, q_out, k_out,
                                    v_out, beta, g, qscale, rep)


def delta_prep_seq(qkv: torch.Tensor, b: torch.Tensor, a: torch.Tensor,
                   neg_expA: torch.Tensor, dt_bias: torch.Tensor, q_out: torch.Tensor,
                   k_out: torch.Tensor, v_out: torch.Tensor, beta: torch.Tensor,
                   g: torch.Tensor, qscale: float, rep: int) -> None:
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
        Nothing; every output buffer is written in place.

    Raises:
        RuntimeError: The extension is unavailable.
    """
    _kernels().delta_prep_seq(qkv, b, a, neg_expA, dt_bias, q_out, k_out, v_out, beta, g,
                              qscale, rep)


def delta(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor,
          g: torch.Tensor, st: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """One gated delta decode step, updating the recurrent state in place.

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
    return _kernels().delta(q, k, v, beta, g, st, out)


def delta_batch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor,
                g: torch.Tensor, st: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Batched gated delta step, one state row per batch entry.

    Args:
        q: Normalised query, ``[B, num_v, dk]`` fp32.
        k: Normalised key, ``[B, num_v, dk]`` fp32.
        v: Value, ``[B, num_v, dv]`` fp32.
        beta: Write strength, ``[B, num_v]`` fp32.
        g: Log decay ``<= 0``, ``[B, num_v]`` fp32.
        st: Recurrent state ``[B, num_v, dk, dv]`` fp32, updated in place.
        out: Output buffer, ``[B, num_v, dv]`` fp32, written in place.

    Returns:
        ``out``, for call chaining.

    Raises:
        RuntimeError: The extension is unavailable.
    """
    return _kernels().delta_batch(q, k, v, beta, g, st, out)


def delta_normed(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                 b: torch.Tensor, a: torch.Tensor, neg_expA: torch.Tensor,
                 dt_bias: torch.Tensor, z: torch.Tensor, nw: torch.Tensor,
                 st: torch.Tensor, out: torch.Tensor, qscale: float,
                 rep: int) -> torch.Tensor:
    """Head prep, recurrence and gated output norm for one decode token, one launch.

    Replaces the ``delta_prologue`` -> ``delta`` -> ``rms_norm_gated`` chain, which
    wrote and re-read about 48 KB of fp32 scratch per layer per token to hand
    three values between kernels. See :func:`delta_normed` in ``kernels/delta.cu``
    for why one block per value head is enough to keep all of it on chip.

    Args:
        q: Raw query, one row per key head, ``[num_k, dk]`` fp32.
        k: Raw key, ``[num_k, dk]`` fp32.
        v: Value, ``[num_v, dv]`` fp32.
        b: Beta logits, ``[num_v]`` bf16.
        a: Decay logits, ``[num_v]`` bf16.
        neg_expA: ``-exp(A_log)``, ``[num_v]`` fp32.
        dt_bias: Decay bias, ``[num_v]`` fp32.
        z: Gate projection for the output norm, ``[num_v, dv]``.
        nw: Gated-norm gain, ``[dv]`` bf16, per value head.
        st: Recurrent state, ``[num_v, dk, dv]`` fp32, updated in place.
        out: Output buffer, ``[num_v, dv]`` fp32, written in place with the normed
            result.
        qscale: ``dk ** -0.5``.
        rep: Value heads per key head.

    Returns:
        ``out``, for call chaining.

    Raises:
        RuntimeError: The extension is unavailable.
    """
    return _kernels().delta_normed(q, k, v, b, a, neg_expA, dt_bias, z, nw, st, out,
                                   qscale, rep)


#: Rows per block in the triangular solve, largest first. Measured against block
#: sizes 32/64/128/256 at L=512 on this card: 256 was fastest (2.48 ms) and 128 was
#: pathological (9.3 ms), so the choice matters more than the algorithm. Every block
#: size tried was correct to 1e-07, so this is purely a speed knob -- which is why a
#: smaller block is picked when the row count does not divide by the best one.
_SOLVE_BLOCKS = (256, 128, 64, 32, 16, 8, 4, 2, 1)


def _solve_block(rows: int) -> int:
    """Pick the largest block size that divides ``rows`` exactly.

    The row count is the delta rule's chunk length, and it is not always a power of
    two large enough for the fastest block: a short prompt prefills with a chunk of
    64, which has no 256-row block. Refusing that case made the whole prefill fail,
    so the block adapts instead.

    Args:
        rows: Number of rows in the triangular system.

    Returns:
        The largest selector from :data:`_SOLVE_BLOCKS` that divides ``rows``.

    Raises:
        ValueError: ``rows`` is zero or negative, which is a caller bug rather than
            a shape this can adapt to.
    """
    if rows <= 0:
        raise ValueError(f"triangular_solve: row count must be positive, got {rows}")
    for block in _SOLVE_BLOCKS:
        if rows % block == 0:
            return block
    return 1


def _solve_blocked(lower: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Blocked-substitution fallback for :func:`triangular_solve`, in pure torch.

    Used only when the extension is unavailable. It is kept rather than deleted because
    it needs no toolchain at all, so a host that can enumerate the backend and run the
    reference path can still exercise this op end to end.

    Args:
        lower: ``[bn, L, L]`` unit-lower-triangular coefficient matrix. Only the strict
            lower triangle is read.
        rhs: ``[bn, L, n]`` right-hand side.

    Returns:
        ``[bn, L, n]`` in ``rhs``'s dtype. Verified against a host fp64 solve at 1.1e-07
        relative over L = 512/1024/2048.

    Raises:
        ValueError: The row count is not positive. Any positive count is handled: the
            block size adapts, so a short prefill chunk of 64 rows works.
    """
    bn, rows, cols = rhs.shape
    block = _solve_block(rows)
    out = torch.empty_like(rhs)
    eye = torch.eye(block, dtype=rhs.dtype, device=rhs.device)
    for start in range(0, rows, block):
        stop = start + block
        acc = rhs[:, start:stop, :]
        if start:
            # Everything already solved contributes one GEMM per block, which is
            # what makes the sequential depth rows/block instead of rows.
            acc = acc - torch.bmm(lower[:, start:stop, :start], out[:, :start, :])
        diag = lower[:, start:stop, start:stop]
        # ``eye`` is broadcast rather than materialised per block: bn copies of a
        # 256x256 fp32 block would be 64 MB at the batch sizes prefill reaches.
        inv = eye.expand(bn, block, block)
        twice = 2.0 * inv
        for _ in range(max(1, (block - 1).bit_length())):
            inv = torch.bmm(inv, twice - torch.bmm(diag, inv))
        out[:, start:stop, :] = torch.bmm(inv, acc)
    return out


def triangular_solve(lower: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Solve ``A X = B`` for unit-lower-triangular ``A``, without any library call.

    ``torch.linalg.solve_triangular`` and every other ``linalg`` solve route is unusable
    on this platform: the CoreX torch build loads cuSOLVER from a hardcoded
    ``/opt/sw_home/local/cuda/lib64/libcusolver.so`` that does not exist on this cluster,
    so the call fails inside ``dlopen`` rather than raising a normal "unsupported" error.
    Five routes were measured and all five failed that way, which is why the operation
    belongs in the contract: the fallback is a platform fact, not model maths.

    The kernel is a plain forward substitution with one thread per right-hand-side
    column. Substitution is independent across columns, so it needs no barrier and makes
    no assumption about warp width -- and it solves a whole chunk in one launch, where
    the blocked form below needs ``rows / block`` GEMMs plus ``log2(block)`` batched
    matmuls per block to build each diagonal inverse.

    Args:
        lower: ``[bn, L, L]`` unit-lower-triangular coefficient matrix. The diagonal is
            taken as 1 and the strict upper triangle is never read.
        rhs: ``[bn, L, n]`` right-hand side.

    Returns:
        ``[bn, L, n]`` in ``rhs``'s dtype.

    Raises:
        RuntimeError: The operands are not a shape the kernel can index, including an
            ``L`` too large for one block's shared memory. The blocked fallback is not
            used for a bad shape: silently switching implementations on a shape error
            would hide the caller's bug behind a slower path.
    """
    try:
        module = _kernels()
    except RuntimeError:
        # No toolchain on this host: the pure-torch substitution still answers, which
        # keeps a structural or reference-path caller working.
        return _solve_blocked(lower, rhs)
    return module.triangular_solve(lower.contiguous(), rhs.contiguous())


#: Ops this backend serves with a kernel, in :class:`Ops` field order. Everything
#: absent is filled from the reference backend by :func:`ops`, which is why a
#: partially implemented platform still answers every name.
#:
#: This dict is hand-maintained, and that is a trap worth naming: an adapter, a header
#: declaration, a binding, a protocol field and a reference implementation can all land
#: while the op is still missing here -- and then it silently runs the *reference*
#: implementation on this backend. A measurement taken in that state reads as a clean
#: zero, which is the same "stale cache is invisible" failure this repo documents twice
#: in another disguise. ``tests/test_backends.py`` pins this against the adapters that
#: exist, so a new kernel cannot be added without its registration.
_NATIVE: dict[str, OpFn] = {
    "linear": linear,
    "linear_m": linear_m,
    "linear_prefill": linear_prefill,
    "rms_norm": rms_norm,
    "add_rms_norm": add_rms_norm,
    "rms_norm_gated": rms_norm_gated,
    "kv_quant": kv_quant,
    "silu_mul": silu_mul,
    "rope_apply": rope_apply,
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
    "delta_normed": delta_normed,
    "triangular_solve": triangular_solve,
}

#: Table with every kernel this backend has; the resolver fills anything absent.
_OPS = Ops.from_mapping(_NATIVE)

#: Names this backend has a kernel for on *some* host. Host-independent by design:
#: what is runnable *here* is ``caps().native``.
NATIVE: frozenset[str] = frozenset(_NATIVE)

#: Whether the captured decode step has been *measured* to work on this card. Kept as an
#: explicit constant rather than inferred, because the two questions are different: a
#: complete op table says every kernel exists, and this says a replay of the **whole
#: decode step** was observed to re-read its inputs and match the eager path.
#:
#: Set to ``False`` on 2026-10-26 after a real failure, and the way it failed is the
#: lesson. ``backends/corex/diag/check_graph.py`` had captured only ``linear`` and
#: ``rms_norm`` -- two ops, in isolation -- and reported ``capture rel=2.551e-03,
#: replay rel=2.744e-03, output moved by 2.910e+02``. That is a true measurement of
#: those two ops and it was taken to mean the decode step was capturable. It is not the
#: same claim: capturing op by op exercises neither the 32-layer call graph, nor the
#: paged-attention window, nor the int8 pool, nor the in-place recurrent state.
#:
#: The first attempt to replay a real decode step then **hung the device**. ``capture()``
#: itself succeeded (2396 ms), the log printed ``decode path: graph replay``, and nothing
#: followed: the job burned its full 25-minute wall clock and was killed as TIMEOUT, with
#: no traceback and no ``.err`` file. That signature is a kernel spinning or waiting on the
#: device, not a segfault -- a segfault exits immediately and leaves a trace.
#:
#: The hang was a **calling-order** bug, not a capture bug: the probe prefilled the prompt
#: and *then* called ``capture()``, which begins by resetting the model and running its own
#: prefill. That wipes the prompt's KV cache and recurrent state, so the replay drove
#: attention over a window that was never written. ``run_chat.generate`` does it the other
#: way round -- ``reset()``, ``capture()``, ``reset()``, then ``prefill()`` -- and so does
#: ``diag/corex/probe_chat.py``.
#:
#: With the order fixed, the whole decode step was measured. The acceptance run is
#: the historical graph acceptance probe on gpu-node-07, 2026-10-26 -- six
#: capture+prefill+replay rounds in one process, every one completing with the *same*
#: tokens ``[5, 6, 24218, 10]`` and finite logits:
#:
#:   backend=corex native=18/18 graph=1, loaded in 35.1 s
#:   capture ok in 3.1s / 2.0s / ...        graph_ready=True (every round)
#:   prefill ok, logits finite=True
#:   decode ok, tokens=[5, 6, 24218, 10] finite=True     (x6)
#:   completed attempts: [0, 1, 2, 3, 4, 5] of 6
#:
#: Identical output across repeated capture+replay is what proves the replay re-reads its
#: inputs, and six repetitions prove it is not one lucky run. Speed against the eager path
#: on the same prompt: 52.41 ms/step vs 77.09 ms/step, so **1.47x**.
#:
#: **One hang was observed and it was not the graph.** A single run on gpu-node-06 stopped
#: after ``graph replay`` with no traceback, and minutes later that *same node* could not
#: complete an 8x8 matmul -- its context was already spinning. Node-06 had hosted the
#: earlier failed capture. So the operational rule is: check a node with an 8x8 matmul
#: before blaming a run, because a wedged context and a wedged kernel produce the same
#: log, and nothing in the log distinguishes them. Every launcher under
#: ``diag/corex/`` now does that check first and exits 64 without touching the
#: model.
#:
#: ``check_graph.py`` still covers two ops only and its header says so; it is the cheap
#: regression check, not the acceptance gate.
#:
#: **Set back to ``False`` on 2026-10-27, and it must stay ``False``.**
#:
#: Three independent reasons, any one sufficient:
#:
#: 1. *It randomly kills cards.* Identical code on comparable nodes produced both outcomes
#:    -- a complete step at 52.41 ms and a run that stopped dead after ``graph replay`` with
#:    no traceback and no stderr. The wedge is unkillable: the process sits in an
#:    uninterruptible state while the device spins, so the card is gone until the node is
#:    reset. Four cards on this partition were lost this way during the port. Nothing that
#:    is only sometimes safe is worth enabling on shared hardware.
#: 2. *It is no longer faster.* The 1.47x above was measured against an eager path that had
#:    not been optimised. After the split-K fix (50.53 -> 43.46 ms) and the delta rewrite
#:    (43.46 -> 37.98 ms), replay measures 46.97 ms on the same prompt -- **0.93x, a
#:    regression**. A graph removes launch overhead; once the kernels got faster, what is
#:    left of that overhead is smaller than the cost of replaying.
#: 3. *The overhead it targeted is better removed elsewhere.* ``launch`` count and host
#:    time are attacked directly in ``../../../PERF_COREX.md`` §8, on the eager path, where a bug
#:    raises an error instead of consuming the device.
#:
#: Do not re-enable this to chase the old 1.47x. The repeat-run evidence above shows the
#: graph *mechanism* works on this driver when it works at all; it does not show the trade
#: is acceptable, and it was measured against a baseline that no longer exists. Any future
#: attempt has to beat 37.98 ms eager with a replay that is provably stable across runs,
#: and it must be done on cards that can afford to be lost.
_GRAPH_VERIFIED = False

#: Ops the captured decode step calls. A device graph is only safe when every one of
#: these is a self-written kernel: a reference fallback reaches ``cuBLAS``, which
#: cannot be captured on this card, and the capture then fails or -- worse -- replays
#: stale output. ``linear_prefill`` is deliberately *not* in this set: it is the
#: prefill regime, which is never captured, and it is the one entry that does call the
#: library (chunked, to stay inside the width this card computes correctly).
_DECODE_OPS: frozenset[str] = frozenset({
    "linear", "linear_m", "rms_norm", "add_rms_norm", "rms_norm_gated",
    "kv_quant", "silu_mul", "rope_apply", "attn_decode",
    "attn_paged", "attn_paged_batch", "attn_paged_multi_q", "conv1d_silu",
    "conv1d_silu_seq", "conv1d_silu_batch", "delta_prologue", "delta_prologue_batch",
    "delta_prep_seq", "delta", "delta_batch", "delta_normed",
})


@functools.lru_cache(maxsize=1)
def caps() -> Caps:
    """Return this backend's capability record for this host.

    Returns:
        A record with ``device="cuda"`` (CoreX implements that namespace), bf16 for
        both dtypes because the prefill GEMM reaches the same 32.6 TFLOP/s on bf16 as
        on fp16 so no conversion is worth its traffic, ``weight_layout="kn"`` because
        that layout is worth 14-22x on decode here, and ``attn_split=1`` because the
        attention kernels do not split K yet -- it is a kernel ABI, and a value the
        kernels do not implement would corrupt attention rather than fail.

        ``graph`` and ``batch_attn`` are derived, not declared. ``graph`` is true only
        once every op the decode step touches has a kernel of its own
        (:data:`_DECODE_OPS`), because the reference fallback reaches ``cuBLAS`` and
        that cannot be captured here. Deriving both means the banner cannot claim a
        capability the table does not have.

        ``graph`` additionally requires :data:`_GRAPH_VERIFIED`: a complete table is
        necessary but not sufficient, because a capture also has to be *measured* to
        re-read its inputs on this card. Until that check passes, a ``True`` here would
        promise a graph that replays stale output, which is the failure mode the flag
        exists to prevent.
    """
    native = frozenset(_OPS.implemented()) if usable() else frozenset()
    return Caps(
        name=NAME,
        device="cuda",
        device_index=0,
        device_name=_device_name(),
        arch=_arch(),
        native=native,
        act_dtype=torch.bfloat16,
        weight_dtype=pack.WEIGHT_DTYPE,
        graph=_GRAPH_VERIFIED and _DECODE_OPS <= native,
        autotune=True,
        attn_split=1,
        batch_attn="attn_paged_batch" in native,
        int8_weights=True,
        recurrent_state=True,
        weight_layout="kn",
    )


@functools.lru_cache(maxsize=1)
def _resolve_device() -> torch.device:
    """Return the device every allocation for this backend goes on.

    Returns:
        ``torch.device("cuda", 0)``, resolved lazily so that merely importing this
        module does not initialise the runtime.
    """
    return make_device(caps())


def __getattr__(attr: str) -> Any:
    """Resolve the lazily-computed module attributes of the :class:`Backend` protocol.

    ``device`` cannot be a module constant: building the capability record reads the
    device name, which initialises the driver, and importing this module on a
    CPU-only host must not do that. Python only calls this hook for names the module
    does not define, so the normal attribute path is unaffected.

    Args:
        attr: Attribute being looked up.

    Returns:
        The resolved attribute.

    Raises:
        AttributeError: The name is not one of the lazily-resolved attributes.
    """
    if attr == "device":
        return _resolve_device()
    raise AttributeError(f"module {__name__!r} has no attribute {attr!r}")


@functools.lru_cache(maxsize=1)
def ops() -> Ops:
    """Return this backend's total op table.

    The extension is built (or attached to a cached build) on the first call, and the
    result is memoised because the engine reaches the table thousands of times per
    request. Entries with no kernel are filled from the reference backend, which is
    what keeps the table total and lets a call site stay branch-free.

    Returns:
        A table where all 17 names are callable: this backend's kernels where they
        exist, the reference implementation for the rest. With no device, or a failed
        build, the table is all reference, so a diagnostic caller can still inspect
        it; :func:`usable` is what keeps such a table away from a request.
    """
    # ``probe()`` first: a host with no card must not reach the build path at all,
    # because a failed attempt would compile nothing but still memoise an error that
    # reads like a toolchain problem.
    if not usable():
        return reference.fill(Ops(), caps())
    return reference.fill(_OPS, caps())


def describe() -> str:
    """Render this backend as one diagnostic line.

    Returns:
        ``"corex: <state> device=... arch=... native=N/17"``, the line the startup
        banner prints and a diagnostic dump can grep. Unlike ``backends.status()`` it
        also reports the device name and the ISA the kernels were built for, which are
        the two facts a wrong-arch build has to be diagnosed from.
    """
    state = "usable" if usable() else ("present, unusable" if probe() else "absent")
    record = caps()
    suffix = "" if usable() else f" ({reason()})"
    return (f"{NAME}: {state}{suffix} device={record.device}:{record.device_index} "
            f"name={record.device_name!r} arch={record.arch!r} "
            f"native={len(record.native)}/{len(Ops.NAMES)} "
            f"layout={record.weight_layout} graph={int(record.graph)}")


def reason_table() -> str:
    """Return the native/fallback split, for a diagnostic dump.

    Returns:
        One line naming which ops run native and which come from the reference, so a
        partly finished backend reports its own progress instead of looking either
        complete or absent.
    """
    return (f"native={', '.join(_OPS.implemented())}; "
            f"fallback={', '.join(_OPS.missing()) or '(none)'}")


#: The protocol is satisfied by the module itself; naming the protocol here keeps a
#: signature drift from going unnoticed by a type checker.
_PROTOCOL: type[Backend] = Backend
