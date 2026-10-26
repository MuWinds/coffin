"""Build and expose the HIP kernels in ``backends/hip/kernels/``.

Module: ``coffin/backends/hip``, bottom of the dependency graph next to
``backends/hip/__init__.py``, which is the only importer. The kernels are C++/HIP
sources compiled once into a cached extension under ``TORCH_EXTENSIONS_DIR``;
this module is only the build/loader shim, so it must stay free of model, op-table
and protocol dependencies.

Constraints:
    * Launch on ``at::cuda::getCurrentCUDAStream()`` (the legacy default stream is
      not part of a ``torch.cuda.graph`` capture, so kernels launched there
      vanish from the graph and the decode step silently reads stale outputs).
    * Keep every shape static: the decode step is captured once and replayed for
      every position, so a shape that varies with the token index cannot be
      captured.
    * Buffers the kernels write (split partials, KV state, conv state) are
      caller-owned and must stay at fixed addresses for the same reason.
    * The extension targets gfx906 only (``PYTORCH_ROCM_ARCH`` /
      ``--offload-arch``). A different part gets its own backend, not another arch
      string here.
    * ``TORCH_EXTENSIONS_DIR`` must be isolated per deployment. A shared cache
      dir makes every new checkout trigger a full HIP rebuild and fight over the
      same build lock; concurrent deploys then wait forever on it.
    * The two ``os.environ`` writes stay above the ``torch`` import, because the
      extension loader reads them at import time. They are process-wide side
      effects: keeping them inside a backend (rather than at the package root)
      is what stops ``PYTORCH_ROCM_ARCH=gfx906`` from leaking into a process that
      selected a different backend.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx906")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/root/private_data/torch_ext")

import torch
from torch.utils.cpp_extension import load

_KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
_SOURCES = ("bindings.cpp", "gemv.cu", "gemv_int8.cu", "delta.cu", "attn.cu", "norm.cu",
            "linear_attn.cu", "batch.cu", "gemv_wide.cu")

_mod = None
_error: str | None = None


def _load():
    """Compile (or attach to the cached build of) the kernel extension once.

    The result is memoised in ``_mod`` and the failure in ``_error``, so a broken
    toolchain costs one build attempt per process instead of one per call -- the
    engine calls into this module thousands of times per request.

    Returns:
        The extension module, or ``None`` when the build failed. Callers must
        treat ``None`` as "use the eager path", never as an empty module. The
        build exception itself is kept in ``_error`` for :func:`last_error`,
        because a missing toolchain has to degrade to the hipBLAS fallback
        rather than take the process down at import.
    """
    global _mod, _error
    if _mod is None and _error is None:
        try:
            _mod = load(
                name="hip_ops_dcu",
                sources=[str(_KERNEL_DIR / name) for name in _SOURCES],
                extra_include_paths=[str(_KERNEL_DIR)],
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3", "--offload-arch=gfx906"],
                verbose=False,
            )
        except Exception as exc:  # pragma: no cover - depends on the toolchain
            _error = f"{type(exc).__name__}: {exc}"
    return _mod


def available() -> bool:
    """True if the extension built (or was already cached).

    Returns:
        ``False`` when the toolchain is missing or the build failed. Callers
        branch on this to pick the eager fallback; the value is stable for the
        life of the process because the load is memoised.
    """
    return _load() is not None


def last_error() -> str | None:
    """Report why the extension is unavailable, for the startup log.

    Returns:
        ``"<ExceptionType>: <message>"`` from the failed build, or ``None`` when
        the build succeeded or was never attempted. The engine logs this once at
        boot; without it a silent fallback to hipBLAS looks like a slow model
        rather than a broken deployment.
    """
    return _error


def gemv(w: torch.Tensor, x: torch.Tensor, R: int = 1, U: int = 2) -> torch.Tensor:
    """``w`` [N, K] fp16 or bf16 contiguous, ``x`` [K] bf16, returns ``y`` [N] bf16.

    Weights are fp16 in practice (see ``models/qwen35/weights.py``); the bf16
    entry point stays for a caller that holds bf16 tensors.

    Args:
        w: Weight matrix ``[N, K]``, row-major, fp16 or bf16, contiguous.
        x: Activation vector ``[K]`` in bf16.
        R: Rows of output computed per thread block; the tuning knob from
            ``models/qwen35/tuning.py``.
        U: K unroll factor (2 or 4). Higher halves the loop overhead and needs
            the K dim to stay a multiple of it.

    Returns:
        ``[N]`` bf16 output.

    Raises:
        RuntimeError: The extension is unavailable. Fall back to
            ``ops/gemv``'s eager path; the numbers must match.
    """
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    if w.dtype == torch.float16:
        return module.gemv_f16(w, x, R, U)
    return module.gemv_bf16(w, x, R, U)


def gemv_m(w: torch.Tensor, x: torch.Tensor, U: int = 2) -> torch.Tensor:
    """``w`` [N, K] fp16/bf16, ``x`` [M, K] bf16 (M in 2/4/8), returns ``y`` [M, N].

    Args:
        w: Weight matrix ``[N, K]``, row-major, contiguous.
        x: Activation matrix ``[M, K]`` bf16. ``M`` must be 2, 4 or 8: the kernel
            is compiled for those widths, so any other M silently reads the wrong
            number of rows.
        U: K unroll factor (2 or 4).

    Returns:
        ``[M, N]`` bf16 output, one weight pass for all M rows.

    Raises:
        RuntimeError: The extension is unavailable.
    """
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    if w.dtype == torch.float16:
        return module.gemv_f16_m(w, x, U)
    return module.gemv_bf16_m(w, x, U)


def gemv_w8a16(w: torch.Tensor, scale: torch.Tensor, x: torch.Tensor,
               R: int = 1, U: int = 2) -> torch.Tensor:
    """``w`` [N, K] int8 with per-row ``scale`` [N], ``x`` [K] bf16 -> [N] bf16.

    Args:
        w: int8 weight matrix ``[N, K]``, row-major, dequantised in-kernel.
        scale: Per-row fp32 dequantisation scale, ``[N]``.
        x: Activation vector ``[K]`` bf16.
        R: Rows of output per thread block.
        U: K unroll factor (2 or 4).

    Returns:
        ``[N]`` bf16 output. Materialising the weight as bf16 instead would cost
        ``[N, K]`` resident bytes, which is what the int8 layout exists to avoid.

    Raises:
        RuntimeError: The extension is unavailable.
    """
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    return module.gemv_w8a16(w, scale, x, R, U)


def gemv_w8a16_m(w: torch.Tensor, scale: torch.Tensor, x: torch.Tensor,
                 U: int = 2) -> torch.Tensor:
    """``w`` [N, K] int8, ``x`` [M, K] bf16 (M in 2/4/8) -> [M, N] bf16.

    Args:
        w: int8 weight matrix ``[N, K]``.
        scale: Per-row fp32 scale, ``[N]``.
        x: Activation matrix ``[M, K]`` bf16, ``M`` in 2/4/8.
        U: K unroll factor (2 or 4).

    Returns:
        ``[M, N]`` bf16 output.

    Raises:
        RuntimeError: The extension is unavailable.
    """
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    return module.gemv_w8a16_m(w, scale, x, U)


def delta(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor,
          g: torch.Tensor, st: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Gated delta rule decode step; updates ``st`` in place and returns ``out``.

    Args:
        q: Normalised query per value head, ``[num_v, dk]`` fp32.
        k: Normalised key per value head, ``[num_v, dk]`` fp32.
        v: Value per value head, ``[num_v, dv]`` fp32.
        beta: Per-head write strength in ``[0, 1]``, ``[num_v]`` fp32.
        g: Per-head log decay, ``<= 0``, ``[num_v]`` fp32.
        st: Recurrent state ``[num_v, dk, dv]`` fp32, updated in place. It is the
            layer's only cross-token memory, so a caller that needs rollback must
            snapshot it first.
        out: Output buffer ``[num_v, dv]`` fp32, written in place.

    Returns:
        ``out`` itself, for call chaining.

    Raises:
        RuntimeError: The extension is unavailable.
    """
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    return module.delta_step(q, k, v, beta, g, st, out)


def delta_prologue(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   b: torch.Tensor, a: torch.Tensor,
                   neg_expA: torch.Tensor, dt_bias: torch.Tensor,
                   q_out: torch.Tensor, k_out: torch.Tensor, v_out: torch.Tensor,
                   beta: torch.Tensor, g: torch.Tensor,
                   qscale: float, rep: int) -> None:
    """L2-normalise q/k, expand key->value heads, build beta/g into fp32 buffers.

    One kernel replaces ~16 small elementwise ops (mul, sum, rsqrt, sigmoid,
    softplus, add, repeat_interleave), which at decode were most of the mixer's
    launch count.

    Args:
        q: Query per key head, ``[num_k, dk]``.
        k: Key per key head, ``[num_k, dk]``.
        v: Value per value head, ``[num_v, dv]``.
        b: Beta logits, ``[num_v]``; sigmoid is applied here.
        a: Decay logits, ``[num_v]``; softplus plus ``dt_bias`` is applied here.
        neg_expA: ``-exp(A_log)``, ``[num_v]`` fp32.
        dt_bias: Decay bias, ``[num_v]`` fp32.
        q_out: Output query, ``[num_v, dk]`` fp32, head-expanded and scaled.
        k_out: Output key, ``[num_v, dk]`` fp32.
        v_out: Output value, ``[num_v, dv]`` fp32.
        beta: Output write strength, ``[num_v]`` fp32.
        g: Output log decay ``<= 0``, ``[num_v]`` fp32.
        qscale: ``dk ** -0.5``, applied to the query only.
        rep: Value heads per key head, i.e. how far to expand.

    Returns:
        Nothing. Every output buffer is written in place.

    Raises:
        RuntimeError: The extension is unavailable.
    """
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    module.delta_prologue(q, k, v, b, a, neg_expA, dt_bias,
                          q_out, k_out, v_out, beta, g, qscale, rep)


def conv1d_silu(conv_w: torch.Tensor, x: torch.Tensor, conv_state: torch.Tensor,
                out: torch.Tensor) -> torch.Tensor:
    """Fused depthwise causal conv1d (kernel 4) + SiLU; shifts ``conv_state``.

    Args:
        conv_w: Depthwise kernel taps ``[conv_dim, 4]`` fp32.
        x: Current token's projection, ``[conv_dim]``.
        conv_state: Previous 3 tokens, ``[conv_dim, 3]`` fp32. Shifted in place,
            so it is the mixer's short-term memory and must be snapshotted with
            the recurrent state.
        out: Output buffer ``[conv_dim]`` fp32, written in place.

    Returns:
        ``out`` itself, for call chaining.

    Raises:
        RuntimeError: The extension is unavailable. The eager path in
            ``linear_attn.__call__`` does cat + copy_ + mul + sum + silu instead.
    """
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    return module.conv1d_silu(conv_w, x, conv_state, out)


def conv1d_silu_seq(conv_w: torch.Tensor, x: torch.Tensor,
                    conv_state: torch.Tensor) -> torch.Tensor:
    """Same conv over a prefill chunk: ``x`` [T, C] bf16 -> [T, C] fp32.

    Args:
        conv_w: Depthwise kernel taps ``[C, 4]`` fp32.
        x: Chunk projections ``[T, C]``, one row per token.
        conv_state: Conv window carried across chunks, ``[C, 3]`` fp32, shifted in
            place so a chunked prefill and a one-shot prefill agree exactly.

    Returns:
        ``[T, C]`` fp32 conv+SiLU output. fp32 because the delta scan downstream
        accumulates across the chunk.

    Raises:
        RuntimeError: The extension is unavailable.
    """
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    return module.conv1d_silu_seq(conv_w, x, conv_state)


def delta_prep_seq(qkv: torch.Tensor, b: torch.Tensor, a: torch.Tensor,
                   neg_expA: torch.Tensor, dt_bias: torch.Tensor,
                   q_out: torch.Tensor, k_out: torch.Tensor, v_out: torch.Tensor,
                   beta: torch.Tensor, g: torch.Tensor,
                   qscale: float, rep: int) -> None:
    """Fused prefill head prep, writing q/k/v as [H, T, d] and beta/g as [H, T].

    The head-major layout is the point of the fusion: the chunked delta scan
    reads ``[H, T, d]`` directly, so producing it here saves a transposed copy of
    the whole chunk.

    Args:
        qkv: Conv output ``[T, conv_dim]`` fp32, the fused q/k/v projection.
        b: Beta logits, ``[T, num_v]``.
        a: Decay logits, ``[T, num_v]``.
        neg_expA: ``-exp(A_log)``, ``[num_v]`` fp32.
        dt_bias: Decay bias, ``[num_v]`` fp32.
        q_out: Output query, ``[num_v, T, dk]`` fp32.
        k_out: Output key, ``[num_v, T, dk]`` fp32.
        v_out: Output value, ``[num_v, T, dv]`` fp32.
        beta: Output write strength, ``[num_v, T]`` fp32.
        g: Output log decay ``<= 0``, ``[num_v, T]`` fp32.
        qscale: ``dk ** -0.5``, applied to the query only.
        rep: Value heads per key head.

    Returns:
        Nothing. Every output buffer is written in place.

    Raises:
        RuntimeError: The extension is unavailable.
    """
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    module.delta_prep_seq(qkv, b, a, neg_expA, dt_bias, q_out, k_out, v_out,
                          beta, g, qscale, rep)


def attn(q: torch.Tensor, gate: torch.Tensor, kcache: torch.Tensor, vcache: torch.Tensor,
         kscale: torch.Tensor, vscale: torch.Tensor, out: torch.Tensor,
         pos: torch.Tensor, rep: int, scale: float,
         part_acc: torch.Tensor, part_ml: torch.Tensor) -> torch.Tensor:
    """Fused single-token attention over the int8 KV cache; returns ``out``.

    ``part_acc``/``part_ml`` are the per-key-slice partials of the online softmax
    (``[nh, ATTN_SPLIT, hd]`` fp32 and ``[nh, ATTN_SPLIT, 2]`` fp32): the kernel
    runs ``ATTN_SPLIT`` blocks per head, so the caller owns these buffers and keeps
    them static for graph capture.

    Args:
        q: Query for this token, ``[nh, hd]``, already scaled and rotated.
        gate: Output gate, ``[nh, hd]``; the kernel applies ``sigmoid``.
        kcache: int8 keys ``[nkv, max_seq, hd]``.
        vcache: int8 values ``[nkv, max_seq, hd]``.
        kscale: Per-(head, token) fp32 K scale, ``[nkv, max_seq]``.
        vscale: Per-(head, token) fp32 V scale, ``[nkv, max_seq]``.
        out: Output buffer ``[nh, hd]``, written in place.
        pos: 1-element long tensor with this token's index; also the causal bound,
            so entries past it are masked rather than excluded from the loop.
        rep: Query heads per KV head (GQA group size).
        scale: ``hd ** -0.5``, folded into the K scale by the kernel.
        part_acc: Split-K accumulator, ``[nh, ATTN_SPLIT, hd]`` fp32, caller-owned.
        part_ml: Split-K running max/denominator, ``[nh, ATTN_SPLIT, 2]`` fp32.

    Returns:
        ``out`` itself, for call chaining.

    Raises:
        RuntimeError: The extension is unavailable.
    """
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    return module.attn_decode(q, gate, kcache, vcache, kscale, vscale, out, pos, rep,
                              scale, part_acc, part_ml)


def attn_paged(q: torch.Tensor, gate: torch.Tensor, kpool: torch.Tensor,
               vpool: torch.Tensor, kscale: torch.Tensor, vscale: torch.Tensor,
               pos2slot: torch.Tensor, out: torch.Tensor, pos: torch.Tensor,
               max_pos: int, rep: int, scale: float,
               part_acc: torch.Tensor, part_ml: torch.Tensor) -> torch.Tensor:
    """Same fused attention over one layer of the paged int8 pool.

    Args:
        q: Query for this token, ``[nh, hd]``.
        gate: Output gate, ``[nh, hd]``.
        kpool: This layer's int8 K pool, ``[nslots, nkv, hd]``.
        vpool: This layer's int8 V pool, ``[nslots, nkv, hd]``.
        kscale: Per-slot fp32 K scale, ``[nslots, nkv]``.
        vscale: Per-slot fp32 V scale, ``[nslots, nkv]``.
        pos2slot: This row's slot map, read only for the first ``pos + 1``
            entries.
        out: Output buffer ``[nh, hd]``, written in place.
        pos: 1-element long tensor with this token's index; the causal bound.
        max_pos: Window length, i.e. the stride of the slot the kernel scans to.
        rep: Query heads per KV head.
        scale: ``hd ** -0.5``.
        part_acc: Split-K accumulator, ``[nh, ATTN_SPLIT, hd]`` fp32.
        part_ml: Split-K running max/denominator, ``[nh, ATTN_SPLIT, 2]`` fp32.

    Returns:
        ``out`` itself, for call chaining.

    Raises:
        RuntimeError: The extension is unavailable.
    """
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    return module.attn_paged_decode(q, gate, kpool, vpool, kscale, vscale,
                                    pos2slot, out, pos, max_pos, rep, scale,
                                    part_acc, part_ml)


def rms_norm(x: torch.Tensor, w: torch.Tensor, stride: int, eps: float = 1e-6) -> torch.Tensor:
    """Fused ``x * rsqrt(mean(x^2)+eps) * w`` over [M, N] rows.

    Args:
        x: Input rows ``[M, N]``.
        w: Gain, ``[N]`` (or strided, see ``stride``).
        stride: Element stride between consecutive rows of ``x``, so a view into a
            wider buffer normalises the same rows without a copy.
        eps: Variance floor, matching the checkpoint's ``layer_norm_eps``.

    Returns:
        Normalised tensor with the same shape as ``x``.

    Raises:
        RuntimeError: The extension is unavailable.
    """
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    return module.rms_norm(x, w, stride, eps)


def rms_norm_gated(x: torch.Tensor, w: torch.Tensor, gate: torch.Tensor,
                   eps: float = 1e-6) -> torch.Tensor:
    """Fused ``rms(x) * w * silu(gate)``.

    Args:
        x: Input ``[num_v, dv]``, one row per value head.
        w: Per-head gain ``[dv]``. It is not ``[num_v*dv]``: the norm is
            per-head, so passing the flattened width runs off the end of the
            tensor.
        gate: Gate projection ``[num_v, dv]``; silu is applied here.
        eps: Variance floor.

    Returns:
        Gated normalised tensor, same shape as ``x``.

    Raises:
        RuntimeError: The extension is unavailable.
    """
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    return module.rms_norm_gated(x, w, gate, eps)


# Batched decode entry points for ``kernels/batch.cu``.
#
# One launch for B sequences instead of B launches; the single-row entry points
# above stay the B = 1 / speculative-verify path.  A row of the batch kernels is
# numerically identical to the matching single-row kernel, and the attention
# kernel loops each row only over that row's own ``context_lens[b]`` keys.


def attn_paged_batch(q: torch.Tensor, gate: torch.Tensor,
                     kpool: torch.Tensor, vpool: torch.Tensor,
                     kscale: torch.Tensor, vscale: torch.Tensor,
                     slots: torch.Tensor, context_lens: torch.Tensor,
                     out: torch.Tensor, max_seq: int, rep: int, scale: float,
                     part_acc: torch.Tensor, part_ml: torch.Tensor) -> torch.Tensor:
    """Batched paged attention over per-row int8 pool windows.

    ``q``/``gate``/``out`` are ``[B, nh, hd]`` (bf16), ``kpool``/``vpool`` are
    ``[nslots, nkv, hd]`` int8 with ``[nslots, nkv]`` fp32 scales, ``slots`` is
    the ``[B, max_seq]`` device slot map (only the first ``context_lens[b]``
    entries of row ``b`` are read) and ``context_lens`` is ``[B]`` int32.
    ``part_acc``/``part_ml`` are the split-K online-softmax partials,
    ``[B, nh, split, hd]`` and ``[B, nh, split, 2]`` fp32, caller-owned so the
    shapes stay static for graph capture.

    Args:
        q: Queries ``[B, nh, hd]`` bf16, already rotated and scaled.
        gate: Output gates ``[B, nh, hd]`` bf16.
        kpool: This layer's int8 K pool, ``[nslots, nkv, hd]``.
        vpool: This layer's int8 V pool, ``[nslots, nkv, hd]``.
        kscale: Per-slot fp32 K scale, ``[nslots, nkv]``.
        vscale: Per-slot fp32 V scale, ``[nslots, nkv]``.
        slots: Per-row device slot map, ``[B, max_seq]``.
        context_lens: Per-row context length ``[B]`` int32. Row ``b`` reads only
            the first ``context_lens[b]`` entries of its own slot map, which is
            what lets a bucket with mixed prompt lengths share one launch.
        out: Output buffer ``[B, nh, hd]``, written in place.
        max_seq: Window length, i.e. the row stride of ``slots``.
        rep: Query heads per KV head.
        scale: ``hd ** -0.5``.
        part_acc: Split-K accumulator, ``[B, nh, split, hd]`` fp32.
        part_ml: Split-K running max/denominator, ``[B, nh, split, 2]`` fp32.

    Returns:
        ``out`` itself, for call chaining.

    Raises:
        RuntimeError: The extension is unavailable.
    """
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    return module.attn_paged_batch(q, gate, kpool, vpool, kscale, vscale, slots,
                                   context_lens, out, max_seq, rep, scale,
                                   part_acc, part_ml)


def attn_paged_multi_q(q: torch.Tensor, gate: torch.Tensor,
                       kpool: torch.Tensor, vpool: torch.Tensor,
                       kscale: torch.Tensor, vscale: torch.Tensor,
                       slots: torch.Tensor, ctx_lens: torch.Tensor,
                       out: torch.Tensor, max_seq: int, rep: int, scale: float,
                       part_acc: torch.Tensor, part_ml: torch.Tensor,
                       block_mask: torch.Tensor | None = None,
                       mask_block: int = 64) -> torch.Tensor:
    """Verify attention: ``c`` draft queries per row share one history KV pass.

    ``q``/``gate``/``out`` are ``[B, c, nh, hd]`` bf16, ``ctx_lens`` is the
    history length per row (query ``j`` sees ``ctx + j + 1`` keys, i.e. history
    plus the first ``j+1`` new tokens).  Each query gets the same split-K /
    online-softmax / gate arithmetic as ``attn_paged_batch``, so it is
    bit-identical to decoding that token on its own.  ``part_acc``/``part_ml``
    are ``[B*c, nh, split, hd]`` / ``[B*c, nh, split, 2]`` fp32.

    ``block_mask`` (uint8 ``[B, n_blocks]``, 1 = keep, block size ``mask_block``)
    selects which KV blocks take part; ``None`` is exact full attention.

    Args:
        q: Draft queries ``[B, c, nh, hd]`` bf16, already rotated and scaled.
        gate: Output gates, same shape.
        kpool: This layer's int8 K pool, ``[nslots, nkv, hd]``.
        vpool: This layer's int8 V pool, ``[nslots, nkv, hd]``.
        kscale: Per-slot fp32 K scale, ``[nslots, nkv]``.
        vscale: Per-slot fp32 V scale, ``[nslots, nkv]``.
        slots: Per-row device slot map, ``[B, max_seq]``.
        ctx_lens: Per-row history length, ``[B]`` int32. Query ``j`` sees
            ``ctx_lens[b] + j + 1`` keys, i.e. history plus the new tokens up to
            and including its own.
        out: Output buffer ``[B, c, nh, hd]``, written in place.
        max_seq: Window length, i.e the row stride of ``slots``.
        rep: Query heads per KV head.
        scale: ``hd ** -0.5``.
        part_acc: Split-K accumulator, ``[B*c, nh, split, hd]`` fp32.
        part_ml: Split-K running max/denominator, ``[B*c, nh, split, 2]`` fp32.
        block_mask: Optional uint8 block keep-mask, ``[B, n_blocks]``.
        mask_block: Block size the mask is expressed in.

    Returns:
        ``out`` itself, for call chaining.

    Raises:
        RuntimeError: The extension is unavailable.
    """
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    if block_mask is None:
        block_mask = torch.empty(0, dtype=torch.uint8, device=q.device)
    return module.attn_paged_multi_q(q, gate, kpool, vpool, kscale, vscale,
                                     slots, ctx_lens, out, max_seq, rep, scale,
                                     part_acc, part_ml, block_mask, mask_block)


def conv1d_silu_batch(conv_w: torch.Tensor, x: torch.Tensor,
                      conv_state: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Batched fused depthwise causal conv1d + SiLU.

    ``x`` [B, C] bf16, ``conv_state`` [B, C, 3] fp32 (shifted in place),
    ``out`` [B, C] fp32.

    Args:
        conv_w: Depthwise kernel taps ``[C, 4]`` fp32.
        x: One token per row, ``[B, C]``.
        conv_state: Per-row conv window ``[B, C, 3]`` fp32, shifted in place.
        out: Output buffer ``[B, C]`` fp32, written in place.

    Returns:
        ``out`` itself, for call chaining.

    Raises:
        RuntimeError: The extension is unavailable.
    """
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    return module.conv1d_silu_batch(conv_w, x, conv_state, out)


def delta_prologue_batch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         b: torch.Tensor, a: torch.Tensor,
                         neg_expA: torch.Tensor, dt_bias: torch.Tensor,
                         q_out: torch.Tensor, k_out: torch.Tensor,
                         v_out: torch.Tensor, beta: torch.Tensor,
                         g: torch.Tensor, qscale: float, rep: int) -> None:
    """Batched q/k norm + head expand + beta/g (see ``delta_prologue``).

    Args:
        q: Query per key head, ``[B, num_k, dk]``.
        k: Key per key head, ``[B, num_k, dk]``.
        v: Value per value head, ``[B, num_v, dv]``.
        b: Beta logits, ``[B, num_v]``.
        a: Decay logits, ``[B, num_v]``.
        neg_expA: ``-exp(A_log)``, ``[num_v]`` fp32, shared by every row.
        dt_bias: Decay bias, ``[num_v]`` fp32, shared by every row.
        q_out: Output query, ``[B, num_v, dk]`` fp32.
        k_out: Output key, ``[B, num_v, dk]`` fp32.
        v_out: Output value, ``[B, num_v, dv]`` fp32.
        beta: Output write strength, ``[B, num_v]`` fp32.
        g: Output log decay ``<= 0``, ``[B, num_v]`` fp32.
        qscale: ``dk ** -0.5``.
        rep: Value heads per key head.

    Returns:
        Nothing. Every output buffer is written in place; each row is
        bit-identical to :func:`delta_prologue` on that row.

    Raises:
        RuntimeError: The extension is unavailable.
    """
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    module.delta_prologue_batch(q, k, v, b, a, neg_expA, dt_bias,
                                q_out, k_out, v_out, beta, g, qscale, rep)


def delta_batch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                beta: torch.Tensor, g: torch.Tensor,
                st: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Batched gated delta update; updates ``st`` [B, nv, dk, dv] in place.

    Args:
        q: Normalised query per value head, ``[B, num_v, dk]`` fp32.
        k: Normalised key per value head, ``[B, num_v, dk]`` fp32.
        v: Value per value head, ``[B, num_v, dv]`` fp32.
        beta: Write strength in ``[0, 1]``, ``[B, num_v]``.
        g: Log decay ``<= 0``, ``[B, num_v]``.
        st: Per-row recurrent state ``[B, num_v, dk, dv]`` fp32, updated in place.
        out: Output buffer ``[B, num_v, dv]`` fp32, written in place.

    Returns:
        ``out`` itself, for call chaining.

    Raises:
        RuntimeError: The extension is unavailable.
    """
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    return module.delta_batch(q, k, v, beta, g, st, out)


def gemv_w8a16_wide(w: torch.Tensor, scale: torch.Tensor, x: torch.Tensor,
                    U: int = 2, threads: int = 0) -> torch.Tensor:
    """Wide-M int8 W8A16 GEMV: ``w`` [N, K] int8, ``scale`` [N], ``x`` [M, K] bf16.

    ``M`` must be 16, 24 or 32; one weight pass, no dequantisation.  Every row is
    bit-identical to calling ``gemv_w8a16_m`` on that row's 8-activation slice.
    ``U`` is the K unroll (2 or 4); ``threads=0`` picks a block size that splits
    the warps evenly into activation groups (192 for M=24, 256 otherwise).

    Args:
        w: int8 weight matrix ``[N, K]``, row-major.
        scale: Per-row fp32 scale, ``[N]``.
        x: Activations ``[M, K]`` bf16, ``M`` in {16, 24, 32}.
        U: K unroll factor (2 or 4).
        threads: Threads per block; ``0`` selects the even-split default above.

    Returns:
        ``[M, N]`` bf16 output. Materialising the weight instead would cost
        ``[N, K]`` resident bytes per call.

    Raises:
        RuntimeError: The extension is unavailable.
    """
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    return module.gemv_w8a16_wide(w, scale, x, U, threads)


def gemv_w8a16_wide_dbg(w: torch.Tensor, scale: torch.Tensor, x: torch.Tensor,
                        U: int, rows: int, mw: int, threads: int = 0) -> torch.Tensor:
    """A/B entry: ``rows`` output rows and ``mw`` activation vectors per warp.

    Args:
        w: int8 weight matrix ``[N, K]``.
        scale: Per-row fp32 scale, ``[N]``.
        x: Activations ``[M, K]`` bf16.
        U: K unroll factor (2 or 4).
        rows: Output rows each block covers, overriding the tuned default.
        mw: Activation vectors per warp, overriding the tuned default.
        threads: Threads per block; ``0`` keeps the compiled default.

    Returns:
        ``[M, N]`` bf16 output with the same maths as
        :func:`gemv_w8a16_wide`; only the launch shape differs, so the two are
        bit-identical and can gate each other in a benchmark.

    Raises:
        RuntimeError: The extension is unavailable.
    """
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    return module.gemv_w8a16_wide_dbg(w, scale, x, U, rows, mw, threads)


def gemv_w8a16_wide_dot2(w: torch.Tensor, scale: torch.Tensor, x: torch.Tensor,
                         U: int = 2, threads: int = 0) -> torch.Tensor:
    """Wide-M int8-weight GEMV with packed fp16 ``v_dot2_f32_f16`` math.

    ``M`` must be 16, 24 or 32; fp32 accumulation, activations converted to fp16
    once per call.  Numerically close to ``gemv_w8a16_wide`` (not bit-identical).

    Args:
        w: int8 weight matrix ``[N, K]``, row-major.
        scale: Per-row fp32 scale, ``[N]``.
        x: Activations ``[M, K]`` bf16, ``M`` in {16, 24, 32}.
        U: K unroll factor (2 or 4).
        threads: Threads per block; ``0`` keeps the compiled default.

    Returns:
        ``[M, N]`` bf16 output. The packed ``v_dot2_f32_f16`` path halves the
        arithmetic instructions per weight element versus the scalar int8 path,
        but it is not bit-identical to :func:`gemv_w8a16_wide`, so it may only be
        enabled where the caller accepts that rounding difference.

    Raises:
        RuntimeError: The extension is unavailable.
    """
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    return module.gemv_w8a16_wide_dot2(w, scale, x, U, threads)


def gemv_w8a16_wide_dot2_dbg(w: torch.Tensor, scale: torch.Tensor, x: torch.Tensor,
                             U: int, rows: int, mw: int,
                             threads: int = 0) -> torch.Tensor:
    """A/B entry: ``rows`` output rows and ``mw`` activation vectors per warp.

    Args:
        w: int8 weight matrix ``[N, K]``.
        scale: Per-row fp32 scale, ``[N]``.
        x: Activations ``[M, K]`` bf16.
        U: K unroll factor (2 or 4).
        rows: Output rows each block covers, overriding the tuned default.
        mw: Activation vectors per warp, overriding the tuned default.
        threads: Threads per block; ``0`` keeps the compiled default.

    Returns:
        ``[M, N]`` bf16 output, the same maths as
        :func:`gemv_w8a16_wide_dot2` with only the launch shape changed.

    Raises:
        RuntimeError: The extension is unavailable.
    """
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    return module.gemv_w8a16_wide_dot2_dbg(w, scale, x, U, rows, mw, threads)

