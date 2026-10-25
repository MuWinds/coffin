"""Chunked gated delta rule: the linear-attention recurrence for prefill.

Everything that depends only on values inside a chunk -- the decay matrix, the
UT system, the two triangular solves -- is one batched op over
``[H * nchunks]``; only the state scan is sequential, and it is affine in the
state, so it is one ``baddbmm`` per chunk. Matches the recurrent form to about
1e-6 relative error (``diag/check_chunked.py``).

Module: ``coffin/ops``. Pure ``torch``; used by ``models/qwen35/linear_attn.py``
for prefill and must never import from ``models/``.

Constraints:
    * The whole kernel exists to stay bandwidth-bound. Splitting the batched
      phases back into a per-chunk Python loop costs ~25 launch-bound kernels per
      chunk per layer, which measured 15% of prefill at n=2048 -- do not
      "simplify" the structure without re-running the probe.
    * Flattened batches are head-major, so chunk ``i`` lives at batch index
      ``i, nc+i, 2*nc+i, ...``; the strided slices in the state scan depend on
      that layout.
    * ``out_state`` is written in place: the caller keeps the recurrent state
      tensor across chunks and prefix reuse snapshots it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ops.dtypes import NEG_INF


#: Strictly-upper masks, one per chunk length, built once instead of per chunk.
#: The decay matrix is a difference of cumulative sums over a chunk, so its upper
#: triangle has to be killed before the exp; allocating and filling that mask for
#: every chunk of every layer was pure overhead.
_UPPER_MASK: dict[tuple[int, int | None], torch.Tensor] = {}


def _upper_mask(n: int, device: torch.device) -> torch.Tensor:
    """Return the cached strictly-upper ``[n, n]`` bool mask for ``device``.

    Args:
        n: Chunk length; one mask is cached per length and device index because
            the decay matrix is rebuilt every chunk but the mask never changes.
        device: Device the mask must live on.

    Returns:
        A shared ``[n, n]`` bool tensor, ``True`` above the diagonal. Treat it as
        read-only: every chunk of every layer gets the same object.
    """
    key = (n, device.index)
    m = _UPPER_MASK.get(key)
    if m is None:
        m = torch.ones(n, n, dtype=torch.bool, device=device).triu(1)
        _UPPER_MASK[key] = m
    return m


def chunked_delta(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor,
                  beta: torch.Tensor, state: torch.Tensor, out_state: torch.Tensor,
                  chunk: int = 64) -> torch.Tensor:
    """Chunked form of the gated delta rule, for prefill over many tokens.

    The output is token-major -- ``[C, H, dv]``, the layout the layers carry their
    activations in -- so the caller needs no transposing copy. Sequence padding is
    absorbed here: chunks are padded to a whole number of chunks with
    ``g = beta = 0``, which makes a padded row contribute nothing to the state
    (``v_beta = k_beta = 0``, so its solved values are zero) while leaving
    ``cum_decay`` untouched.

    Two phases, like the reference ``torch_chunk_gated_delta_rule``:

    1. everything that depends only on values *inside* a chunk -- the decay
       matrix, the UT system, the two triangular solves, the decayed q/k -- is
       computed for all chunks at once as one batched op over ``[H * nchunks]``;
    2. a sequential scan over chunks carries the recurrent state. This step is
       affine in the state (see below), so it is one batched matmul per chunk,
       and the outputs are four batched matmuls over all chunks at once.

    Args:
        q: Queries, ``[H, C, d]``, already L2-normalised and scaled.
        k: Keys, ``[H, C, d]``, already L2-normalised.
        v: Values, ``[H, C, dv]``.
        g: Per-token log decay, ``[H, C]``.
        beta: Per-token write gate, ``[H, C]``.
        state: Recurrent state carried in from earlier chunks, ``[H, dk, dv]``.
        out_state: Destination for the updated state, ``[H, dk, dv]``. Written in
            place so the caller's state tensor survives, which is what prefix
            reuse snapshots.
        chunk: Tokens per chunk. Drives the shape of the batched system; the
            default is the length the probes were measured at.

    Returns:
        Outputs, ``[C, H, dv]``, truncated back to the real token count when
        padding was added.

    Raises:
        RuntimeError: ``state``/``out_state`` disagree with ``q``/``v`` in the
            head or head-dim axes, surfaced by the matmuls. The caller must size
            the state from the same heads it feeds here.
    """
    H, C, dk = q.shape
    dv = v.shape[-1]
    nc = (C + chunk - 1) // chunk
    if nc * chunk != C:
        pad = nc * chunk - C
        q = F.pad(q, (0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
        g = F.pad(g, (0, pad))
        beta = F.pad(beta, (0, pad))
    L = chunk

    # ``.contiguous()`` on the reshapes is not cosmetic.  The caller holds the
    # activation as [C, H, d] and hands over transposed views, and splitting the
    # chunk dimension of a transposed view *is* expressible without a copy -- so
    # reshape returns a strided 4-D view.  Elementwise ops on that miss the
    # vectorized path and run about 5x slower (measured: one kernel at 392 us
    # instead of 79 us, 26.1 vs 13.8 ms per layer at C=2048).  The copy itself is
    # one blocked pass (~7 us per tensor) and only happens when it is needed.
    qc = q.reshape(H, nc, L, dk).contiguous()
    kc = k.reshape(H, nc, L, dk).contiguous()
    vc = v.reshape(H, nc, L, dv).contiguous()
    cum = g.reshape(H, nc, L).contiguous().cumsum(-1)                 # [H, nc, L]
    pd = (cum.unsqueeze(-1) - cum.unsqueeze(-2)) \
        .masked_fill(_upper_mask(L, q.device), NEG_INF).exp()         # [H, nc, L, L]

    bc = beta.reshape(H, nc, L).contiguous()
    kb = kc * bc.unsqueeze(-1)
    vb = vc * bc.unsqueeze(-1)
    ut = (kb @ kc.transpose(-1, -2)) * pd
    ica = (qc @ kc.transpose(-1, -2)) * pd
    ecum = cum.exp()                                                  # reused twice below
    dkb = kb * ecum.unsqueeze(-1)

    # Both solves share ``ut``, and rocBLAS TRSM costs about linearly in the
    # number of right-hand sides, so inverting once and applying the inverse with
    # two batched matmuls beats two TRSMs with 128 columns each.  Same maths:
    # A^-1 B either way, to fp32 round-off.
    bn = H * nc
    ut2 = ut.reshape(bn, L, L)
    eye = torch.eye(L, dtype=ut.dtype, device=ut.device).expand(bn, L, L)
    ut_inv = torch.linalg.solve_triangular(ut2, eye, upper=False, unitriangular=True)
    nv = torch.bmm(ut_inv, vb.reshape(bn, L, dv)).reshape(H, nc, L, dv)
    kcd = torch.bmm(ut_inv, dkb.reshape(bn, L, dk)).reshape(H, nc, L, dk)

    q2 = qc * ecum.unsqueeze(-1)
    k2 = kc * (cum[..., -1:] - cum).exp().unsqueeze(-1)
    cd = cum[..., -1].exp()                                           # [H, nc]

    # Phase 2 is affine in the state, not merely linear.  Substituting
    # ``v_new_i = nv_i - kcd_i @ st_i`` into the state update gives
    #     st_{i+1} = (cd_i * I - k2_i^T kcd_i) st_i + k2_i^T nv_i  =  A_i st_i + B_i
    # so the sequential part costs *one* baddbmm per chunk instead of three small
    # ones plus a broadcast multiply, and -- because we then know the state that
    # enters every chunk -- the outputs collapse into four batched matmuls over
    # [H * nchunks] instead of five ops per chunk.  On one layer at n=2048 this is
    # 4.3 ms against 9.7 ms for the per-chunk form (diag/probe_delta_scan_struct.py),
    # at ~1.8x the FLOPs but a quarter of the launches, which is the right trade on
    # a part where a small kernel costs 8-15 us to launch.
    #
    # ``bn`` is defined above (phase 1); ``A``/``B`` are flattened head-major, so
    # chunk ``i`` lives at batch indices ``i, nc+i, 2*nc+i, ...`` -- the strided
    # slices below are what a strided-batched rocBLAS GEMM is for.
    k2t = k2.transpose(-1, -2).reshape(bn, dk, L)
    A = torch.bmm(k2t, kcd.reshape(bn, L, dk)).neg_()                  # [bn, dk, dk]
    A.diagonal(dim1=-2, dim2=-1).add_(cd.reshape(bn, 1))               # + cd_i * I
    B = torch.bmm(k2t, nv.reshape(bn, L, dv))                          # [bn, dk, dv]

    sts = torch.empty(H, nc, dk, dv, dtype=v.dtype, device=v.device)
    st = state
    for i in range(nc):
        sts[:, i] = st
        st = torch.baddbmm(B[i::nc], A[i::nc], st)
    out_state.copy_(st)

    v_new = nv - torch.bmm(kcd.reshape(bn, L, dk), sts.reshape(bn, dk, dv)) \
        .reshape(H, nc, L, dv)
    out = torch.baddbmm(torch.bmm(q2.reshape(bn, L, dk), sts.reshape(bn, dk, dv)),
                        ica.reshape(bn, L, L), v_new.reshape(bn, L, dv)) \
        .reshape(H, nc, L, dv)
    # Drop the chunk padding again; the caller only wants the real tokens.  The
    # permute+reshape is the one reordering copy in this function, and the
    # [C, H, dv] result lets the caller reshape to [C, Hv, dv] for free (a *view*,
    # since [C, H, dv] is contiguous).  Note [H, nc, L, dv] must not be reshaped
    # straight to [nc*L, H, dv]: that is a legal view of contiguous memory and
    # would silently reorder the axes.
    return out.permute(1, 2, 0, 3).reshape(nc * L, H, dv)[:C]
