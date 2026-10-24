"""Build and expose the HIP kernels in ``kernels/``.

The kernels themselves are C++/HIP sources (``gemv.cu``, ``delta.cu``) compiled
once into a cached extension under ``TORCH_EXTENSIONS_DIR``; this module is only
the build/loader shim. If the build fails the engine falls back to hipBLAS.

Two constraints the kernels must respect, both learned the hard way: launch on
``at::cuda::getCurrentCUDAStream()`` (the legacy default stream is not part of a
``torch.cuda.graph`` capture, so kernels launched there vanish from the graph),
and keep every shape static (the decode step is captured once and replayed for
every position).
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
    """True if the extension built (or was already cached)."""
    return _load() is not None


def last_error() -> str | None:
    return _error


def gemv(w: torch.Tensor, x: torch.Tensor, R: int = 1, U: int = 2) -> torch.Tensor:
    """``w`` [N, K] fp16 or bf16 contiguous, ``x`` [K] bf16, returns ``y`` [N] bf16.

    Weights are fp16 in practice (see ``models/qwen35/weights.py``); the bf16
    entry point stays for a caller that holds bf16 tensors.
    """
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    if w.dtype == torch.float16:
        return module.gemv_f16(w, x, R, U)
    return module.gemv_bf16(w, x, R, U)


def gemv_m(w: torch.Tensor, x: torch.Tensor, U: int = 2) -> torch.Tensor:
    """``w`` [N, K] fp16/bf16, ``x`` [M, K] bf16 (M in 2/4/8), returns ``y`` [M, N]."""
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    if w.dtype == torch.float16:
        return module.gemv_f16_m(w, x, U)
    return module.gemv_bf16_m(w, x, U)


def gemv_w8a16(w: torch.Tensor, scale: torch.Tensor, x: torch.Tensor,
               R: int = 1, U: int = 2) -> torch.Tensor:
    """``w`` [N, K] int8 with per-row ``scale`` [N], ``x`` [K] bf16 -> [N] bf16."""
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    return module.gemv_w8a16(w, scale, x, R, U)


def gemv_w8a16_m(w: torch.Tensor, scale: torch.Tensor, x: torch.Tensor,
                 U: int = 2) -> torch.Tensor:
    """``w`` [N, K] int8, ``x`` [M, K] bf16 (M in 2/4/8) -> [M, N] bf16."""
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    return module.gemv_w8a16_m(w, scale, x, U)


def delta(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor,
          g: torch.Tensor, st: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Gated delta rule decode step; updates ``st`` in place and returns ``out``."""
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
    """L2-normalise q/k, expand key->value heads, build beta/g into fp32 buffers."""
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    module.delta_prologue(q, k, v, b, a, neg_expA, dt_bias,
                          q_out, k_out, v_out, beta, g, qscale, rep)


def conv1d_silu(conv_w: torch.Tensor, x: torch.Tensor, conv_state: torch.Tensor,
                out: torch.Tensor) -> torch.Tensor:
    """Fused depthwise causal conv1d (kernel 4) + SiLU; shifts ``conv_state``."""
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    return module.conv1d_silu(conv_w, x, conv_state, out)


def conv1d_silu_seq(conv_w: torch.Tensor, x: torch.Tensor,
                    conv_state: torch.Tensor) -> torch.Tensor:
    """Same conv over a prefill chunk: ``x`` [T, C] bf16 -> [T, C] fp32."""
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    return module.conv1d_silu_seq(conv_w, x, conv_state)


def delta_prep_seq(qkv: torch.Tensor, b: torch.Tensor, a: torch.Tensor,
                   neg_expA: torch.Tensor, dt_bias: torch.Tensor,
                   q_out: torch.Tensor, k_out: torch.Tensor, v_out: torch.Tensor,
                   beta: torch.Tensor, g: torch.Tensor,
                   qscale: float, rep: int) -> None:
    """Fused prefill head prep, writing q/k/v as [H, T, d] and beta/g as [H, T]."""
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
    """Same fused attention over one layer of the paged int8 pool."""
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    return module.attn_paged_decode(q, gate, kpool, vpool, kscale, vscale,
                                    pos2slot, out, pos, max_pos, rep, scale,
                                    part_acc, part_ml)


def rms_norm(x: torch.Tensor, w: torch.Tensor, stride: int, eps: float = 1e-6) -> torch.Tensor:
    """Fused ``x * rsqrt(mean(x^2)+eps) * w`` over [M, N] rows."""
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    return module.rms_norm(x, w, stride, eps)


def rms_norm_gated(x: torch.Tensor, w: torch.Tensor, gate: torch.Tensor,
                   eps: float = 1e-6) -> torch.Tensor:
    """Fused ``rms(x) * w * silu(gate)``."""
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
    """Batched q/k norm + head expand + beta/g (see ``delta_prologue``)."""
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    module.delta_prologue_batch(q, k, v, b, a, neg_expA, dt_bias,
                                q_out, k_out, v_out, beta, g, qscale, rep)


def delta_batch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                beta: torch.Tensor, g: torch.Tensor,
                st: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Batched gated delta update; updates ``st`` [B, nv, dk, dv] in place."""
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
    """
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    return module.gemv_w8a16_wide(w, scale, x, U, threads)


def gemv_w8a16_wide_dbg(w: torch.Tensor, scale: torch.Tensor, x: torch.Tensor,
                        U: int, rows: int, mw: int, threads: int = 0) -> torch.Tensor:
    """A/B entry: ``rows`` output rows and ``mw`` activation vectors per warp."""
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    return module.gemv_w8a16_wide_dbg(w, scale, x, U, rows, mw, threads)


def gemv_w8a16_wide_dot2(w: torch.Tensor, scale: torch.Tensor, x: torch.Tensor,
                         U: int = 2, threads: int = 0) -> torch.Tensor:
    """Wide-M int8-weight GEMV with packed fp16 ``v_dot2_f32_f16`` math.

    ``M`` must be 16, 24 or 32; fp32 accumulation, activations converted to fp16
    once per call.  Numerically close to ``gemv_w8a16_wide`` (not bit-identical).
    """
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    return module.gemv_w8a16_wide_dot2(w, scale, x, U, threads)


def gemv_w8a16_wide_dot2_dbg(w: torch.Tensor, scale: torch.Tensor, x: torch.Tensor,
                             U: int, rows: int, mw: int,
                             threads: int = 0) -> torch.Tensor:
    """A/B entry: ``rows`` output rows and ``mw`` activation vectors per warp."""
    module = _load()
    if module is None:
        raise RuntimeError(f"hip_ops unavailable: {_error}")
    return module.gemv_w8a16_wide_dot2_dbg(w, scale, x, U, rows, mw, threads)

