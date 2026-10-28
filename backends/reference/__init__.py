"""The pure-torch backend: the reference implementation of every op, and the spec the kernels are checked against.

This backend is the safety net that makes the platform abstraction total. Every
op in :class:`~coffin.backends.protocol.Ops` has a plain-torch implementation
here, so a platform with no native kernel still runs the model -- slower, but
with no `if backend.available()` fork anywhere in the model code. New platforms
are required to match these numbers before they are tuned for speed
(PLATFORMS.md §6), so this file is the specification, not a fallback of last
resort.

The maths is lifted from the eager branches that are currently spread over
``models/qwen35/{full_attn,linear_attn}.py``, ``ops/gemv.py`` and
``ops/norm.py``. Moving them here is what lets those branches be deleted.

Module: ``coffin/backends/reference``. Leaf of the backend subtree: it imports
``coffin.backends.protocol``, ``torch`` and the leaf module ``int8_weights``, and
nothing else from ``coffin``. ``backends/{hip,intel}`` import :func:`fill` to
complete their own tables, so this module must never import them back.

Constraints:
    * Device-agnostic by construction: no ``"cuda"``, no device literal, no
      allocation outside the caller's ``out`` buffers. Reference ops run on
      whatever device their inputs are on, which is what makes a CPU-only smoke
      test (`COFFIN_BACKEND=reference`) possible.
    * Numbers are not bit-identical to the kernels and must not be forced to be:
      the reference accumulates in fp32 where a kernel may stay in the activation
      dtype (e.g. attention dequantises the int8 KV in fp32 before the matmul).
      Conformance compares against the reference with a tolerance table, so being
      the *more* accurate side is the point.
    * Buffers named ``out``/``conv_state``/``*_out`` are caller-owned and written
      with ``copy_``. Replacing them would break the graph-capture contract of
      the surrounding model code, which holds their addresses fixed.
    * The int8 KV cache, ``QWeight`` weights and the fp32 recurrent state are
      model-side storage formats; the reference must accept them as they come
      rather than re-quantising, or a reference run would not be the same model.
"""

from __future__ import annotations

import functools

import torch
import torch.nn.functional as F

from backends.protocol import Caps, Ops, make_device
from int8_weights import QWeight

#: Mask fill value; ``-inf`` makes the softmax weight exactly zero.
NEG_INF = float("-inf")

#: Backend identifier, i.e. what ``COFFIN_BACKEND`` accepts.
NAME = "reference"


# Activation dtype of the model's compute path. The reference runs in bf16
# because that is what the model allocates and what the int8 KV quantiser is
# written against; on a CPU-only host this is slow but correct, and correctness on
# a host with no accelerator is the whole reason this backend exists.
_CAPS = Caps(
    name=NAME,
    device="cpu",
    device_name="torch-eager",
    native=frozenset(),
    act_dtype=torch.bfloat16,
    weight_dtype=torch.bfloat16,
    graph=False,
    autotune=False,
    attn_split=1,
    batch_attn=True,
    int8_weights=False,
    recurrent_state=True,
)


def _linear(w: torch.Tensor | QWeight, x: torch.Tensor,
            weight_dtype: torch.dtype) -> torch.Tensor:
    """Project activations with a row-major weight, for any row count.

    Args:
        w: ``[N, K]`` row-major weight, plain tensor or ``QWeight``.
        x: Activations, ``[K]`` or ``[C, K]``.
        weight_dtype: Dtype an int8 weight dequantises to when the activations are
            no wider: on this backend that is the caps ``weight_dtype``.

    Returns:
        ``[N]`` or ``[C, N]`` in ``x``'s dtype. The computation runs in the wider
        of the two operands, so an int8 weight dequantises straight into fp32
        when the caller is in fp32 (no bf16 round trip), while a bf16 model call
        stays in bf16 exactly like the kernel. One GEMM covers the single-row and
        the prefill regimes: the reference has no kernel choice to make, which is
        the distinction the three ``linear*`` names hide from the caller.
    """
    compute = torch.promote_types(x.dtype, weight_dtype)
    dense = w.dequant(compute) if isinstance(w, QWeight) else w.to(compute)
    return F.linear(x.to(compute), dense).to(x.dtype)


def _rms_norm(x: torch.Tensor, w: torch.Tensor, stride: int = 0,
              eps: float = 1e-6) -> torch.Tensor:
    """Zero-centred RMSNorm over the last axis of a 2-D activation block.

    Written out by hand instead of calling ``F.rms_norm``: that entry point is
    torch 2.4+, and the CoreX deployment on this cluster runs torch 2.1, where the
    attribute does not exist and every reference call raises. The arithmetic is
    the same one the fused kernels implement -- an fp32 mean square, an fp32
    rsqrt, then the gain multiply in the activation dtype.

    Args:
        x: Rows to normalise, ``[M, N]``, at any row pitch.
        w: Gain, ``[N]``, already carrying the checkpoint's ``1 + w`` offset.
        stride: Element pitch between rows. Unused here: the row-major read of
            ``x`` already gives the pitch. It stays in the signature so the
            contract matches a native kernel that needs it for column slices.
        eps: Variance floor; the checkpoint's ``layer_norm_eps``.

    Returns:
        The normalised block, same shape and dtype as ``x``.
    """
    del stride
    xf = x.to(torch.float32)
    scale = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * scale).to(x.dtype) * w.to(x.dtype)


def _rms_norm_gated(x: torch.Tensor, w: torch.Tensor, gate: torch.Tensor,
                    eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm scaled by ``silu(gate)``, normalised per head.

    Args:
        x: Activations, ``[..., dv]``; the last axis is the per-head width.
        w: Per-head gain, ``[dv]``. It is not the flattened width: normalising
            across the flattened width would divide by the RMS of every head.
        gate: Gate tensor shaped like ``x``; silu is applied here.
        eps: Variance floor.

    Returns:
        Same shape and dtype as ``x``.
    """
    return _rms_norm(x, w, 0, eps) * F.silu(gate)


def _add_rms_norm(x: torch.Tensor, add: torch.Tensor, w: torch.Tensor,
                  out: torch.Tensor, stride: int = 0,
                  eps: float = 1e-6) -> torch.Tensor:
    """Residual add into ``x``, then the zero-centred RMSNorm of the sum into ``out``.

    The upstream spelling is ``x = x + add`` followed by a norm of ``x``: two
    launches and two reads of the residual. This is the same arithmetic with the
    sum materialised once, which is what a fused kernel does, so the reference and
    the kernel agree by construction rather than by tolerance.

    Args:
        x: Residual stream, ``[M, N]``; accumulated in place.
        add: Mixer output to add, same shape as ``x``.
        w: Gain, ``[N]``, carrying the ``1 + w`` offset.
        out: Normalised result, ``[M, N]``, written in place. Must not alias ``x``:
            the residual has to survive the norm.
        stride: Element pitch of ``x`` between rows. Unused here for the same
            reason as in :func:`_rms_norm`.
        eps: Variance floor.

    Returns:
        ``out``, for call chaining.
    """
    del stride
    x.add_(add)
    out.copy_(_rms_norm(x, w, 0, eps))
    return out


def _silu_mul(gate: torch.Tensor, up: torch.Tensor,
              out: torch.Tensor) -> torch.Tensor:
    """``out = silu(gate) * up``, the SwiGLU product, into a caller-owned buffer.

    Args:
        gate: Gate half of the fused projection, ``[...]``.
        up: Up half, same shape as ``gate``.
        out: Destination, same shape. May alias ``up``: the product overwrites it
            and nothing reads ``up`` afterwards.

    Returns:
        ``out``, for call chaining.
    """
    out.copy_(F.silu(gate) * up)
    return out


def _rope_apply(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor,
                sin: torch.Tensor, q_out: torch.Tensor, k_out: torch.Tensor,
                rotary_dim: int) -> torch.Tensor:
    """Rotate the leading ``rotary_dim`` channels of every q/k head into caller buffers.

    ``rotate_half`` pairs channel ``i`` with ``i + rotary_dim/2``, which is the layout the
    ``cat(half, half)`` cos/sin tables are built in. Anything else rotates the wrong pairs
    without a shape error.

    Args:
        q: Queries, ``[M, nh, hd]``.
        k: Keys, ``[M, nkv, hd]``.
        cos: Cosine for the rows being rotated, ``[M, rotary_dim]`` fp32.
        sin: Sine, same layout as ``cos``.
        q_out: Destination for the rotated queries, ``q``'s shape.
        k_out: Destination for the rotated keys, ``k``'s shape.
        rotary_dim: Channels to rotate; the rest pass through.

    Returns:
        ``q_out``, for call chaining.
    """
    c = cos.to(q.dtype).unsqueeze(1)          # [M, 1, rotary_dim]
    s = sin.to(q.dtype).unsqueeze(1)
    for src, dst in ((q, q_out), (k, k_out)):
        rot = src[..., :rotary_dim]
        half = rotary_dim // 2
        rh = torch.cat([-rot[..., half:], rot[..., :half]], dim=-1)
        dst[..., :rotary_dim].copy_(rot * c + rh * s)
        dst[..., rotary_dim:].copy_(src[..., rotary_dim:])
    return q_out


def _kv_quant(x: torch.Tensor, out_q: torch.Tensor,
              out_s: torch.Tensor) -> torch.Tensor:
    """Symmetric int8 quantisation of ``[M, N]`` with one scale per row.

    This is the chain ``FullAttn._quant`` performs for the int8 KV cache, and it is
    an op rather than model-side torch for one reason: the cache is read by a
    kernel that undoes exactly this quantisation, so the ``1e-5`` floor and the
    ``127`` divisor are an agreement between the writer and the reader. Two
    spellings of it that differ in the last ulp would leave the cache and the
    attention kernel disagreeing about the scale.

    Args:
        x: Activations to quantise, ``[M, N]``.
        out_q: int8 destination, ``[M, N]``, written in place.
        out_s: fp32 destination for the per-row scale, ``[M]``, written in place.

    Returns:
        ``out_q``, for call chaining.
    """
    xf = x.to(torch.float32)
    s = xf.abs().amax(dim=-1).clamp_min(1e-5) / 127.0
    out_s.copy_(s)
    out_q.copy_((xf / s.unsqueeze(-1)).round().clamp_(-127, 127).to(torch.int8))
    return out_q


def _dequant_cache(k: torch.Tensor, kscale: torch.Tensor) -> torch.Tensor:
    """Widen and rescale an int8 KV cache slab to fp32.

    Args:
        k: int8 codes, ``[nkv, S, hd]``.
        kscale: Per-(head, token) fp32 scale, ``[nkv, S]``.

    Returns:
        ``[nkv, S, hd]`` fp32. The scale is applied while widening rather than to
        the attention scores, so the reference does its arithmetic on
        dequantised values -- the more accurate of the two orders, and the one a
        future kernel is measured against.
    """
    return k.to(torch.float32) * kscale.to(torch.float32)[:, :, None]


def _attn_math(q: torch.Tensor, gate: torch.Tensor, k: torch.Tensor,
               v: torch.Tensor, kscale: torch.Tensor, vscale: torch.Tensor,
               rep: int, scale: float, keep: torch.Tensor | None = None) -> torch.Tensor:
    """Causal gated attention over one row's int8 KV window.

    Args:
        q: Query for this token, ``[nh, hd]``, already rotated and scaled.
        gate: Output gate, ``[nh, hd]``; sigmoid is applied here.
        k: int8 keys, ``[nkv, S, hd]``.
        v: int8 values, ``[nkv, S, hd]``.
        kscale: fp32 K scale, ``[nkv, S]``.
        vscale: fp32 V scale, ``[nkv, S]``.
        rep: Query heads per KV head (``nh // nkv``).
        scale: ``hd ** -0.5``. The caller has already folded it into nothing, so
            it is applied here and not by ``q``.
        keep: Optional bool ``[S]`` key mask, ``True`` = participate; ``None`` is
            exact full attention. A row whose mask keeps no key has no defined
            softmax denominator, same as the kernel.

    Returns:
        ``[nh, hd]`` gated output in ``q``'s dtype.
    """
    nh, hd = q.shape
    nkv = k.shape[0]
    qf = q.float().reshape(nkv, rep, hd)
    kf = _dequant_cache(k, kscale)
    vf = _dequant_cache(v, vscale)
    scores = torch.einsum("ghd,gsd->ghs", qf, kf) * scale
    if keep is not None:
        scores = scores.masked_fill(~keep[None, None, :], NEG_INF)
    probs = scores.softmax(-1)
    out = torch.einsum("ghs,gsd->ghd", probs, vf).reshape(nh, hd)
    return (out * gate.float().sigmoid()).to(q.dtype)


def _slice_window(pos: torch.Tensor, total: int) -> int:
    """Number of cache entries a token at ``pos`` can see.

    Args:
        pos: 1-element long tensor holding the token index.
        total: Entries available in the cache.

    Returns:
        ``pos + 1`` clamped into ``[1, total]``. Reading only the visible window
        is numerically identical to masking the tail (masked softmax weights are
        exactly zero) and it keeps a CPU reference run from touching the whole
        256k window every step.
    """
    return max(1, min(int(pos.reshape(-1)[0]) + 1, total))


def _attn_decode(q: torch.Tensor, gate: torch.Tensor, kcache: torch.Tensor,
                 vcache: torch.Tensor, kscale: torch.Tensor, vscale: torch.Tensor,
                 out: torch.Tensor, pos: torch.Tensor, rep: int, scale: float,
                 part_acc: torch.Tensor | None = None,
                 part_ml: torch.Tensor | None = None) -> torch.Tensor:
    """One decode token over the dense int8 KV cache.

    Args:
        q: Query, ``[nh, hd]``.
        gate: Output gate, ``[nh, hd]``.
        kcache: int8 keys, ``[nkv, max_seq, hd]``.
        vcache: int8 values, ``[nkv, max_seq, hd]``.
        kscale: fp32 K scale, ``[nkv, max_seq]``.
        vscale: fp32 V scale, ``[nkv, max_seq]``.
        out: Output buffer, ``[nh, hd]``, written in place.
        pos: 1-element long tensor; the token index and the causal bound.
        rep: Query heads per KV head.
        scale: ``hd ** -0.5``.
        part_acc: Split-K accumulator. Ignored: the reference does not split.
        part_ml: Split-K running max/denominator. Ignored, same reason.

    Returns:
        ``out``, for call chaining.
    """
    n = _slice_window(pos, kcache.shape[1])
    res = _attn_math(q, gate, kcache[:, :n], vcache[:, :n],
                     kscale[:, :n], vscale[:, :n], rep, scale)
    out.copy_(res)
    return out


def _attn_paged(q: torch.Tensor, gate: torch.Tensor, kpool: torch.Tensor,
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
        pos2slot: This row's slot map, ``[max_pos]``; only the first ``pos + 1``
            entries are read, so the tail may still hold another sequence's slots.
        out: Output buffer, ``[nh, hd]``, written in place.
        pos: 1-element long tensor; the token index and the causal bound.
        max_pos: Window length, i.e. the stride of a slot map row.
        rep: Query heads per KV head.
        scale: ``hd ** -0.5``.
        part_acc: Split-K accumulator. Ignored.
        part_ml: Split-K running max/denominator. Ignored.

    Returns:
        ``out``, for call chaining.
    """
    n = _slice_window(pos, min(max_pos, pos2slot.numel()))
    slots = pos2slot[:n].long()
    k = kpool.index_select(0, slots).permute(1, 0, 2)
    v = vpool.index_select(0, slots).permute(1, 0, 2)
    ks = kscale.index_select(0, slots).permute(1, 0)
    vs = vscale.index_select(0, slots).permute(1, 0)
    out.copy_(_attn_math(q, gate, k, v, ks, vs, rep, scale))
    return out


def _attn_paged_batch(q: torch.Tensor, gate: torch.Tensor, kpool: torch.Tensor,
                      vpool: torch.Tensor, kscale: torch.Tensor, vscale: torch.Tensor,
                      slots: torch.Tensor, context_lens: torch.Tensor,
                      out: torch.Tensor, max_seq: int, rep: int, scale: float,
                      part_acc: torch.Tensor | None = None,
                      part_ml: torch.Tensor | None = None) -> torch.Tensor:
    """One decode token for a bucket of rows, each with its own context length.

    Args:
        q: Queries, ``[B, nh, hd]``.
        gate: Output gates, ``[B, nh, hd]``.
        kpool: This layer's int8 K pool, ``[nslots, nkv, hd]``.
        vpool: This layer's int8 V pool, ``[nslots, nkv, hd]``.
        kscale: fp32 K scale, ``[nslots, nkv]``.
        vscale: fp32 V scale, ``[nslots, nkv]``.
        slots: Per-row slot maps, ``[B, max_seq]``.
        context_lens: Per-row key count, ``[B]`` int32. Row ``b`` reads only the
            first ``context_lens[b]`` entries of its own map.
        out: Output buffer, ``[B, nh, hd]``, written in place.
        max_seq: Window length, i.e. the row stride of ``slots``.
        rep: Query heads per KV head.
        scale: ``hd ** -0.5``.
        part_acc: Split-K accumulator. Ignored.
        part_ml: Split-K running max/denominator. Ignored.

    Returns:
        ``out``, for call chaining.
    """
    for b in range(q.shape[0]):
        n = max(0, min(int(context_lens[b]), max_seq))
        if n == 0:
            # A row with no context has no attention to compute; the kernel writes
            # the gated bias of an empty softmax, which is zero because the output
            # gate multiplies a zero weighted sum.
            out[b].zero_()
            continue
        win = slots[b, :n].long()
        k = kpool.index_select(0, win).permute(1, 0, 2)
        v = vpool.index_select(0, win).permute(1, 0, 2)
        ks = kscale.index_select(0, win).permute(1, 0)
        vs = vscale.index_select(0, win).permute(1, 0)
        out[b].copy_(_attn_math(q[b], gate[b], k, v, ks, vs, rep, scale))
    return out


def _attn_paged_multi_q(q: torch.Tensor, gate: torch.Tensor, kpool: torch.Tensor,
                        vpool: torch.Tensor, kscale: torch.Tensor, vscale: torch.Tensor,
                        slots: torch.Tensor, ctx_lens: torch.Tensor,
                        out: torch.Tensor, max_seq: int, rep: int, scale: float,
                        part_acc: torch.Tensor | None = None,
                        part_ml: torch.Tensor | None = None,
                        block_mask: torch.Tensor | None = None,
                        mask_block: int = 64) -> torch.Tensor:
    """Speculative verify: ``c`` draft queries per row share one pass over that row's history.

    Args:
        q: Draft queries, ``[B, c, nh, hd]``, already rotated and scaled.
        gate: Output gates, same shape as ``q``.
        kpool: This layer's int8 K pool, ``[nslots, nkv, hd]``.
        vpool: This layer's int8 V pool, ``[nslots, nkv, hd]``.
        kscale: fp32 K scale, ``[nslots, nkv]``.
        vscale: fp32 V scale, ``[nslots, nkv]``.
        slots: Per-row slot maps, ``[B, max_seq]``.
        ctx_lens: Per-row history length, ``[B]`` int32. Query ``j`` sees
            ``ctx_lens[b] + j + 1`` keys: the history plus the drafts up to and
            including itself, so each query is bit-identical to decoding that
            token alone.
        out: Output buffer, ``[B, c, nh, hd]``, written in place.
        max_seq: Window length, i.e. the row stride of ``slots``.
        rep: Query heads per KV head.
        scale: ``hd ** -0.5``.
        part_acc: Split-K accumulator. Ignored.
        part_ml: Split-K running max/denominator. Ignored.
        block_mask: Optional uint8 ``[B, n_blocks]`` keep-mask (1 = keep), block
            size ``mask_block``. ``None`` or an empty tensor is exact attention.
        mask_block: Block size the mask is expressed in.

    Returns:
        ``out``, for call chaining.
    """
    bsz, c = q.shape[0], q.shape[1]
    for b in range(bsz):
        base = int(ctx_lens[b])
        keep_full = None
        if block_mask is not None and block_mask.numel() > 0:
            keep_full = block_mask[b].bool().repeat_interleave(int(mask_block))
        for j in range(c):
            n = min(base + j + 1, max_seq)
            if n <= 0:
                out[b, j].zero_()
                continue
            keep = None
            if keep_full is not None:
                keep = keep_full[:n]
                if keep.numel() < n:
                    keep = F.pad(keep, (0, n - keep.numel()))
            win = slots[b, :n].long()
            k = kpool.index_select(0, win).permute(1, 0, 2)
            v = vpool.index_select(0, win).permute(1, 0, 2)
            ks = kscale.index_select(0, win).permute(1, 0)
            vs = vscale.index_select(0, win).permute(1, 0)
            out[b, j].copy_(_attn_math(q[b, j], gate[b, j], k, v, ks, vs,
                                       rep, scale, keep))
    return out


def _conv1d_silu(conv_w: torch.Tensor, x: torch.Tensor, conv_state: torch.Tensor,
                 out: torch.Tensor) -> torch.Tensor:
    """Depthwise causal conv1d (kernel 4) fused with SiLU, for one token.

    Args:
        conv_w: Per-channel taps, ``[C, K]`` fp32, oldest tap first.
        x: This token's projection, ``[C]``.
        conv_state: The previous ``K - 1`` tokens, ``[C, K - 1]`` fp32; shifted in
            place so it is the mixer's short-term memory across steps.
        out: Output buffer, ``[C]`` fp32, written in place.

    Returns:
        ``out``, for call chaining.
    """
    win = torch.cat([conv_state, x.to(conv_state.dtype).reshape(-1, 1)], dim=1)
    conv_state.copy_(win[:, 1:])
    out.copy_(F.silu((win * conv_w).sum(-1)))
    return out


def _conv1d_silu_seq(conv_w: torch.Tensor, x: torch.Tensor,
                     conv_state: torch.Tensor) -> torch.Tensor:
    """The same depthwise conv over a prefill chunk, carrying the window across chunks.

    Args:
        conv_w: Per-channel taps, ``[C, K]`` fp32.
        x: Chunk projections, ``[T, C]``.
        conv_state: Previous ``K - 1`` tokens, ``[C, K - 1]`` fp32; updated in
            place to the last ``K - 1`` rows of ``state + x``, which is what makes
            a chunked prefill agree with a one-shot prefill.

    Returns:
        ``[T, C]`` fp32 conv+SiLU output; fp32 because the delta scan downstream
        accumulates across the chunk.
    """
    kernel = conv_w.shape[1]
    seq = torch.cat([conv_state.t(), x.to(torch.float32)], dim=0)      # [K-1+T, C]
    conv_state.copy_(seq[-(kernel - 1):].t())
    win = seq.unfold(0, kernel, 1)                                     # [T, C, K]
    return F.silu((win * conv_w).sum(-1))


def _conv1d_silu_batch(conv_w: torch.Tensor, x: torch.Tensor,
                       conv_state: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Batched :func:`_conv1d_silu`: ``B`` rows, one conv window per row.

    Args:
        conv_w: Per-channel taps, ``[C, K]`` fp32, shared by every row.
        x: One token per row, ``[B, C]``.
        conv_state: Per-row windows, ``[B, C, K - 1]`` fp32; shifted in place.
        out: Output buffer, ``[B, C]`` fp32, written in place.

    Returns:
        ``out``, for call chaining.
    """
    for b in range(x.shape[0]):
        _conv1d_silu(conv_w, x[b], conv_state[b], out[b])
    return out


def _head_prep(q: torch.Tensor, k: torch.Tensor, b: torch.Tensor,
               a: torch.Tensor, neg_expA: torch.Tensor, dt_bias: torch.Tensor,
               qscale: float, rep: int) -> tuple[torch.Tensor, torch.Tensor,
                                                 torch.Tensor, torch.Tensor]:
    """Normalise, head-expand and gate one step's query/key/beta/decay.

    Args:
        q: Query per key head, ``[..., num_k, dk]``.
        k: Key per key head, ``[..., num_k, dk]``.
        b: Beta logits, ``[..., num_v]``.
        a: Decay logits, ``[..., num_v]``.
        neg_expA: ``-exp(A_log)``, ``[num_v]`` fp32.
        dt_bias: Decay bias, ``[num_v]`` fp32.
        qscale: ``dk ** -0.5``, applied to the query only.
        rep: Value heads per key head.

    Returns:
        ``(q_norm, k_norm, beta, g)`` with the q/k head axis expanded to
        ``num_v``. ``g`` is ``<= 0`` by construction: a decay per step, never a
        growth. The RMS floor is ``1e-6`` because the checkpoint's normalisation
        uses it and a zero-norm head would otherwise divide by zero.
    """
    qf = q.float()
    kf = k.float()
    if rep > 1:
        qf = qf.repeat_interleave(rep, dim=-2)
        kf = kf.repeat_interleave(rep, dim=-2)
    qn = qf * torch.rsqrt((qf * qf).sum(-1, keepdim=True) + 1e-6) * qscale
    kn = kf * torch.rsqrt((kf * kf).sum(-1, keepdim=True) + 1e-6)
    beta = b.float().sigmoid()
    g = neg_expA * F.softplus(a.float() + dt_bias)
    return qn, kn, beta, g


def _delta_prologue(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    b: torch.Tensor, a: torch.Tensor,
                    neg_expA: torch.Tensor, dt_bias: torch.Tensor,
                    q_out: torch.Tensor, k_out: torch.Tensor, v_out: torch.Tensor,
                    beta: torch.Tensor, g: torch.Tensor,
                    qscale: float, rep: int) -> None:
    """:func:`_head_prep` for one token, written into the caller's fp32 buffers.

    Args:
        q: Query per key head, ``[num_k, dk]``.
        k: Key per key head, ``[num_k, dk]``.
        v: Value per value head, ``[num_v, dv]``; already at the value-head
            count, so it is only widened.
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
        Nothing. Every output buffer is written in place, because the caller keeps
        these buffers at fixed addresses for graph capture.
    """
    qn, kn, bt, gt = _head_prep(q, k, b, a, neg_expA, dt_bias, qscale, rep)
    q_out.copy_(qn)
    k_out.copy_(kn)
    v_out.copy_(v.float())
    beta.copy_(bt)
    g.copy_(gt)


def _delta_prologue_batch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                          b: torch.Tensor, a: torch.Tensor,
                          neg_expA: torch.Tensor, dt_bias: torch.Tensor,
                          q_out: torch.Tensor, k_out: torch.Tensor,
                          v_out: torch.Tensor, beta: torch.Tensor,
                          g: torch.Tensor, qscale: float, rep: int) -> None:
    """Batched :func:`_delta_prologue`: a leading row dim on every argument.

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
        Nothing. Buffers are written in place; row ``b`` is identical to
        :func:`_delta_prologue` on that row.
    """
    qn, kn, bt, gt = _head_prep(q, k, b, a, neg_expA, dt_bias, qscale, rep)
    q_out.copy_(qn)
    k_out.copy_(kn)
    v_out.copy_(v.float())
    beta.copy_(bt)
    g.copy_(gt)


def _delta_prep_seq(qkv: torch.Tensor, b: torch.Tensor, a: torch.Tensor,
                    neg_expA: torch.Tensor, dt_bias: torch.Tensor,
                    q_out: torch.Tensor, k_out: torch.Tensor, v_out: torch.Tensor,
                    beta: torch.Tensor, g: torch.Tensor,
                    qscale: float, rep: int) -> None:
    """Prefill head prep: split the fused conv output and emit the head-major layout.

    Args:
        qkv: Conv output, ``[T, conv_dim]``, holding q/k/v end to end.
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
        Nothing. The head-major outputs are the layout the chunked delta scan
        reads directly, which is why this op exists instead of a transpose at the
        call site.
    """
    num_v, T, dk = q_out.shape
    dv = v_out.shape[2]
    num_k = num_v // rep
    kdim = num_k * dk
    qf = qkv[:, :kdim].reshape(T, num_k, dk)
    kf = qkv[:, kdim:2 * kdim].reshape(T, num_k, dk)
    vf = qkv[:, 2 * kdim:].reshape(T, num_v, dv)
    qn, kn, bt, gt = _head_prep(qf, kf, b, a, neg_expA, dt_bias, qscale, rep)
    q_out.copy_(qn.transpose(0, 1).contiguous())
    k_out.copy_(kn.transpose(0, 1).contiguous())
    v_out.copy_(vf.float().transpose(0, 1).contiguous())
    beta.copy_(bt.transpose(0, 1).contiguous())
    g.copy_(gt.transpose(0, 1).contiguous())


def _delta_step(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                beta: torch.Tensor, g: torch.Tensor, st: torch.Tensor,
                out: torch.Tensor) -> torch.Tensor:
    """One gated delta update of a per-head recurrent state, in place.

    The rule is ``st <- st * exp(g)``, then ``st <- st + k (v - st^T k)^T beta``,
    and the output is ``st^T q`` read *after* the update, so a token sees its own
    write.

    Args:
        q: Normalised query, ``[num_v, dk]``.
        k: Normalised key, ``[num_v, dk]``.
        v: Value, ``[num_v, dv]``.
        beta: Write strength in ``[0, 1]``, ``[num_v]``.
        g: Log decay ``<= 0``, ``[num_v]``.
        st: Recurrent state ``[num_v, dk, dv]`` fp32, updated in place.
        out: Output buffer ``[num_v, dv]`` fp32, written in place.

    Returns:
        ``out``, for call chaining.
    """
    st.mul_(g.float().exp()[:, None, None])
    kv = (st * k.float()[:, :, None]).sum(-2)                  # [num_v, dv]
    d = (v.float() - kv) * beta.float()[:, None]
    st.add_(k.float()[:, :, None] * d[:, None, :])
    out.copy_((st * q.float()[:, :, None]).sum(-2))
    return out


def _delta(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor,
           g: torch.Tensor, st: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Single-row :func:`_delta_step`, the decode path of the linear mixer.

    Args:
        q: Normalised query, ``[num_v, dk]``.
        k: Normalised key, ``[num_v, dk]``.
        v: Value, ``[num_v, dv]``.
        beta: Write strength, ``[num_v]``.
        g: Log decay, ``[num_v]``.
        st: Recurrent state ``[num_v, dk, dv]`` fp32, updated in place. It has no
            closed-form rewind, so a caller that must roll back snapshots it.
        out: Output buffer ``[num_v, dv]`` fp32, written in place.

    Returns:
        ``out``, for call chaining.
    """
    return _delta_step(q, k, v, beta, g, st, out)


def _delta_batch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                 beta: torch.Tensor, g: torch.Tensor, st: torch.Tensor,
                 out: torch.Tensor) -> torch.Tensor:
    """Batched :func:`_delta`: one state row per batch entry, kept independent.

    Args:
        q: Normalised query, ``[B, num_v, dk]``.
        k: Normalised key, ``[B, num_v, dk]``.
        v: Value, ``[B, num_v, dv]``.
        beta: Write strength, ``[B, num_v]``.
        g: Log decay, ``[B, num_v]``.
        st: Recurrent state ``[B, num_v, dk, dv]`` fp32, updated in place.
        out: Output buffer ``[B, num_v, dv]`` fp32, written in place.

    Returns:
        ``out``, for call chaining. Row ``b`` is identical to :func:`_delta` on
        that row, which is the property the batched kernel is measured against.
    """
    for b in range(q.shape[0]):
        _delta_step(q[b], k[b], v[b], beta[b], g[b], st[b], out[b])
    return out


def _delta_normed(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  b: torch.Tensor, a: torch.Tensor,
                  neg_expA: torch.Tensor, dt_bias: torch.Tensor,
                  z: torch.Tensor, nw: torch.Tensor, st: torch.Tensor,
                  out: torch.Tensor, qscale: float, rep: int) -> torch.Tensor:
    """Head prep, recurrence and gated norm for one decode token.

    The three-stage spelling allocates ``q_pro``/``k_pro``/``v_pro``/``beta``/``g``
    as fp32 scratch, then ``delta_out``, and re-reads each: about 48 KB round
    tripped per layer per token. This is the same arithmetic with every
    intermediate kept local, which is what the fused kernel does, so the two agree
    by construction and this doubles as the specification.

    Args:
        q: Query per key head, ``[num_k, dk]``.
        k: Key per key head, ``[num_k, dk]``.
        v: Value per value head, ``[num_v, dv]``.
        b: Beta logits, ``[num_v]``.
        a: Decay logits, ``[num_v]``.
        neg_expA: ``-exp(A_log)``, ``[num_v]`` fp32.
        dt_bias: Decay bias, ``[num_v]`` fp32.
        z: Gate projection for the output norm, ``[num_v, dv]``.
        nw: Gated-norm gain, ``[dv]``, per value head rather than flattened.
        st: Recurrent state ``[num_v, dk, dv]`` fp32, updated in place.
        out: Output buffer ``[num_v, dv]`` fp32, written in place with the normed
            result.
        qscale: ``dk ** -0.5``.
        rep: Value heads per key head.

    Returns:
        ``out``, for call chaining.
    """
    qn, kn, beta, g = _head_prep(q, k, b, a, neg_expA, dt_bias, qscale, rep)
    vf = v.float()
    dh = torch.empty_like(vf)
    _delta_step(qn, kn, vf, beta, g, st, dh)
    out.copy_(_rms_norm_gated(dh, nw, z.float(), 1e-6))
    return out


def _triangular_solve(lower: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Solve ``A X = B`` for a unit-lower-triangular ``A``.

    The chunked delta rule needs ``A^-1 B`` once per chunk per layer, where ``A``
    is ``I`` plus strictly-lower products and is therefore unit triangular.

    This is the specification, so it states the library call the op exists to hide:
    on ROCm/CUDA builds ``torch.linalg.solve_triangular`` is the fastest correct
    answer and this is what a backend should match. On CoreX it is unavailable --
    that build loads cuSOLVER from a hardcoded path that does not exist on the
    cluster -- which is precisely why the operation belongs in the contract instead
    of in ``ops/delta.py``: the fallback is a platform fact, not model maths.

    Args:
        lower: ``[bn, L, L]`` unit-lower-triangular coefficient matrix. The upper
            triangle and the diagonal are ignored, as in the library call.
        rhs: ``[bn, L, n]`` right-hand side.

    Returns:
        ``[bn, L, n]`` in ``rhs``'s dtype.
    """
    return torch.linalg.solve_triangular(lower, rhs, upper=False,
                                         unitriangular=True)


def build(caps: Caps | None = None) -> Ops:
    """Build the reference op table.

    Args:
        caps: Capability record whose ``weight_dtype`` the ``linear*`` ops use to
            dequantise int8 weights. ``None`` uses this module's own record; a
            native backend passes its own, so a merged table dequantises to that
            backend's weight dtype rather than to the reference's.

    Returns:
        A table with all 18 entries callable. The reference never leaves a gap:
        a missing entry here would be a hole in the safety net that every other
        backend relies on.
    """
    c = _CAPS if caps is None else caps
    linear = functools.partial(_linear, weight_dtype=c.weight_dtype)
    return Ops(
        linear=linear,
        linear_m=linear,
        linear_prefill=linear,
        rms_norm=_rms_norm,
        rms_norm_gated=_rms_norm_gated,
        add_rms_norm=_add_rms_norm,
        kv_quant=_kv_quant,
        silu_mul=_silu_mul,
        rope_apply=_rope_apply,
        attn_decode=_attn_decode,
        attn_paged=_attn_paged,
        attn_paged_batch=_attn_paged_batch,
        attn_paged_multi_q=_attn_paged_multi_q,
        conv1d_silu=_conv1d_silu,
        conv1d_silu_seq=_conv1d_silu_seq,
        conv1d_silu_batch=_conv1d_silu_batch,
        delta_prologue=_delta_prologue,
        delta_prologue_batch=_delta_prologue_batch,
        delta_prep_seq=_delta_prep_seq,
        delta=_delta,
        delta_batch=_delta_batch,
        delta_normed=_delta_normed,
        triangular_solve=_triangular_solve,
    )


def fill(partial: Ops, caps: Caps) -> Ops:
    """Complete a native backend's table with reference implementations.

    Args:
        partial: A native backend's table; ``None`` marks an op it has no kernel
            for.
        caps: The *caller's* capability record, so the reference entries fill in
            with that backend's dtypes and device rather than this module's.

    Returns:
        A total table: native entries are kept, everything else is the reference
        implementation. This is the one place where "no kernel here" is turned
        into "run the spec", which is what keeps model code free of availability
        checks.
    """
    base = build(caps)
    return Ops(**{name: (getattr(partial, name) or getattr(base, name))
                  for name in Ops.NAMES})


#: The reference backend's own total table, memoised because every process that
#: falls back needs the same one and rebuilding it per call would allocate 18
#: closures on the decode path.
_OPS = build(_CAPS)

#: Backend identifier, mirroring :data:`NAME` for the ``Backend`` protocol.
name = NAME

#: Device every allocation for this backend goes on. The reference is CPU-bound by
#: definition, but the attribute exists so ``backends.device()`` works uniformly.
device = make_device(_CAPS)


def probe() -> bool:
    """The reference backend is always present: it is plain torch.

    Returns:
        Always ``True``, which is what makes it the guaranteed last candidate in
        the selection order.
    """
    return True


def usable() -> bool:
    """Always usable: torch is already imported by the time this is asked.

    Returns:
        Always ``True``. Nothing has to be built, which is the property that lets
        ``import coffin.backends`` work on a host with no accelerator.
    """
    return True


def reason() -> str:
    """The reference backend cannot be unusable.

    Returns:
        Always ``""``.
    """
    return ""


def ops() -> Ops:
    """Return the reference table.

    Returns:
        The memoised all-18 table. Every entry is callable.
    """
    return _OPS


def caps() -> Caps:
    """Return the reference capability record.

    Returns:
        A record with ``device="cpu"``, no native ops, no graph capture and no
        autotune, so model code takes its generic path everywhere.
    """
    return _CAPS


#: Names this backend serves natively. Empty by definition; listed so a caller can
#: tell "reference" apart from a native backend without importings its module.
NATIVE: frozenset[str] = frozenset()

#: Exported surface of the backend module, matching what ``registry`` calls.
__all__ = ["NAME", "NATIVE", "build", "caps", "device", "fill", "name", "ops",
           "probe", "reason", "usable"]
