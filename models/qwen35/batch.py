"""Batched weight reads for the static-batch decode and the one-forward verify.

A decode step is bound by reading the weights (4 GB of int8 per step, ~700 GB/s),
so applying one weight row to B input vectors costs almost the same as applying it
to one -- that is the whole win of a static batch. Both entry points here are
therefore thin: they hand a ``[M, K]`` activation block to ``linear_m`` and let the
backend decide which ``M`` it specialises and how it splits the rest.

What used to live here was that decision: the per-M unroll table, two
environment switches that selected a specific vendor kernel, and five ``hasattr``
probes for optional kernel symbols. All of it was kernel selection, which is the
backend's job now, so this file keeps only the two things that are actually
model-side: the group split for a fused projection, and the *capability*
predicates the verify path asks before choosing an implementation
(``wide_kernel_ready`` / ``multi_q_ready``, answered from ``caps.native`` rather
than from symbol presence).

Module: ``coffin/models/qwen35``. Depends on ``ops/gemv`` and ``backends``; used by
``blocks.py`` (every projection in the static-batch path) and by
``full_attn``/``mtp`` for the verify widths.

Constraints:
    * Bit-identity with solo decode is the contract: the static batch must produce
      the same numbers as one-at-a-time decode, or a batched request and an
      unbatched one disagree. That is now the backend's responsibility for each
      ``linear_m`` entry, which is why this file no longer picks an unroll: a
      caller-side choice could only be right for one card's kernels.
    * Nothing here may name a kernel symbol, a device API or a tuning flag. A
      backend without ``linear_m`` still answers the entry (the reference fills
      it), so the fallback is invisible to this module.
    * ``WIDE_STATS`` is mutated without locking, which is fine because the engine
      drives all projection calls single-threaded from ``llm_engine``.
"""

from __future__ import annotations

import torch

from backends import caps as _caps
from ops.gemv import GEMV, GEMVGroup

#: The selected backend's capability record, read once. Used only by the two
#: capability predicates below, which exist so the verify path can pick an
#: implementation *before* capturing a graph -- a decision that has to be made at
#: capture time, not per step.
_CAPS = _caps()


def _project(proj: GEMV, x: torch.Tensor) -> torch.Tensor:
    """Apply one projection to a batch of rows.

    Args:
        proj: The projection to apply.
        x: Activations, ``[M, K]``, bf16.

    Returns:
        ``[M, N]``. Row ``r`` equals a solo decode of ``x[r]`` on any backend whose
        ``linear_m`` keeps the single-row accumulation order, which is the property
        the batched and unbatched paths are held to (see the module constraints).
    """
    if x.shape[0] == 1:
        # Bucket 1 is a one-row graph: the single-stream entry keeps the row dim
        # rather than taking the batch entry, whose M is fixed at capture time.
        return proj(x[0]).unsqueeze(0)
    return proj.linear_m(x)


def batch_proj(proj, x: torch.Tensor):
    """Batched projection for the static-batch decode widths.

    Args:
        proj: A :class:`~ops.gemv.GEMV`, a :class:`~ops.gemv.GEMVGroup` (the fused
            ``qkv`` / ``in_proj_qkv/z/b/a`` / ``gate_up`` reads) or a
            :class:`~ops.embed.TiedEmbedding`'s inner GEMV.
        x: Activations, ``[M, K]``, bf16.

    Returns:
        ``[M, N]``, or a list of per-weight ``[M, N_i]`` tensors when ``proj`` is
        a fused group, split in the order the weights were concatenated.
    """
    if isinstance(proj, GEMVGroup):
        return _project(proj.g, x).split(proj.sizes, dim=1)
    return _project(proj, x)


def wide_kernel_ready() -> bool:
    """Report whether the backend has a native batched projection.

    Callers use this to choose the one-forward verify at capture time: without a
    native ``linear_m`` that path is still correct but reads the weights through a
    dequantising GEMM per row, which is why it is worth asking before capturing.

    Returns:
        ``True`` when the selected backend serves ``linear_m`` with a kernel. This
        is a capability question, not a symbol probe: ``caps.native`` is the
        backend's own statement of what it implements.
    """
    return "linear_m" in _CAPS.native


def multi_q_ready() -> bool:
    """Report whether the multi-query paged attention kernel is native here.

    Returns:
        ``True`` when the selected backend serves ``attn_paged_multi_q`` natively;
        ``False`` means verify must fall back to the per-query attention path.
    """
    return "attn_paged_multi_q" in _CAPS.native


#: Flipped to True once ``FullAttn.verify_batch_multi`` is wired to the kernel.
MULTI_Q_WIRED = True


#: Counts where :func:`wide_proj` sent each module: "smallM" / "prefill". Kept
#: because the one-forward verify must be all-smallM in an int8 build; anything
#: else is the dequant path and shows up as a step-time regression.
WIDE_STATS: dict[str, int] = {}


def _count(kind: str) -> None:
    """Record that ``wide_proj`` took the ``kind`` path, for later attribution.

    Args:
        kind: Path label, one of "smallM" / "prefill"; see :data:`WIDE_STATS` for
            how it is read.
    """
    WIDE_STATS[kind] = WIDE_STATS.get(kind, 0) + 1


def wide_proj(proj, x: torch.Tensor):
    """Batched projection at an arbitrary M (``B*c``), for the one-forward verify.

    The backend's ``linear_m`` is asked for every M here rather than only for the
    power-of-two widths the old per-M kernels instantiated: it owns the decision of
    which M it specialises and how to split the rest, so this file has no M table to
    keep in sync with a kernel. Like :func:`batch_proj` it accepts a
    :class:`~ops.gemv.GEMVGroup` and returns a list for it.

    Args:
        proj: A :class:`~ops.gemv.GEMV` or :class:`~ops.gemv.GEMVGroup`.
        x: Activations, ``[M, K]``, bf16. ``M`` need not be a power of two.

    Returns:
        ``[M, N]``, or a list of per-weight tensors for a fused group. Rows carry
        no bit-identity guarantee with solo decode when a wide path is used;
        :data:`WIDE_STATS` records which path ran.
    """
    if isinstance(proj, GEMVGroup):
        return wide_proj(proj.g, x).split(proj.sizes, dim=1)
    _count("smallM" if x.shape[0] <= 8 else "prefill")
    return proj.linear_m(x) if x.shape[0] > 1 else proj(x).unsqueeze(0)
