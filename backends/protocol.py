"""The contract every device backend implements: one flat semantic op table plus capability flags.

The engine needs exactly two things from a platform: a callable per *semantic*
device op, and a handful of behavioural facts the model code branches on. A flat
table keyed by op name (rather than one class per kernel) keeps call sites
mechanical -- ``ops.linear(w, x)`` instead of ``hip_ops.gemv(w, x, R, U)`` -- and
makes the table total: every backend answers every name, falling back to the
pure-torch reference for an op it has no native kernel for, so model code has one
path instead of an ``if hip_ops.available()`` fork per call.

Module: ``coffin/backends``. Bottom of the dependency graph next to
``coffin/backends/registry.py``: it may import ``torch`` and the leaf module
``int8_weights`` (which itself imports only ``torch``). It may not import
``coffin/ops``, ``models``, ``engine``, ``serving`` or ``cache`` -- those import
from here, never the reverse, and the backend layer has to stay importable with
no model, no checkpoint and no compiled extension.

Constraints:
    * Contract parameters are model-semantic quantities only: shapes, dtypes, the
      quantised storage format, positions, causal lengths. Anything that exists
      because of how *this* card tiles a kernel (rows per block, K-unroll, wide-M
      variants, debug A/B entry points) stays inside the backend and must never
      appear as an op name or an argument. That is why the eight ``hip_ops.gemv*``
      entries collapse to three ``linear*`` names: a new platform must not have to
      accept tuning knobs that mean nothing there.
    * ``w`` may be a plain ``torch.Tensor`` or an ``int8_weights.QWeight``; the
      int8 dispatch happens *inside* the backend, so the caller passes whichever
      storage the loader produced. ``QWeight`` is the model-side storage format,
      not a platform capability.
    * In-place semantics are part of the contract: ``out``, ``conv_state``,
      ``st`` and the prologue outputs are caller-owned buffers that the
      graph-capture path keeps at fixed addresses. A backend writes through them
      and returns the same tensor rather than allocating a replacement.
    * Signatures are identical across backends, including the arguments the
      pure-torch path ignores: a call site must not branch on which backend is
      active.
    * ``Caps`` carries only facts the model branches on. Whether an op is
      *natively* implemented is ``native_ops``; the fallback answer is always
      "call it anyway", so the table itself never tells a caller to skip work.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Callable, ClassVar, Mapping, Protocol, runtime_checkable

import torch


#: An op entry. Signatures are documented per :class:`Ops` field; backends bind
#: closures and partials, so the callable type stays opaque here.
OpFn = Callable[..., Any]


#: Canonical op names, in :class:`Ops` field order and grouped the way
#: PLATFORMS.md §3.2 tables them. Kept in sync with the dataclass by a test.
_OP_NAMES: tuple[str, ...] = (
    # Projections: one name per activation-count regime, not per kernel.
    "linear",            # 1 or a few tokens
    "linear_m",          # small-M batch, one weight pass
    "linear_prefill",    # prefill-chunk GEMM (large M)
    # Normalisation.
    "rms_norm",
    "rms_norm_gated",
    # Full attention over the int8 KV cache.
    "attn_decode",
    "attn_paged",
    "attn_paged_batch",
    "attn_paged_multi_q",
    # Gated DeltaNet: depthwise conv + SiLU, then the recurrence.
    "conv1d_silu",
    "conv1d_silu_seq",
    "conv1d_silu_batch",
    "delta_prologue",
    "delta_prologue_batch",
    "delta_prep_seq",
    "delta",
    "delta_batch",
    # Triangular solve: the chunked delta rule inverts a unit-lower-triangular
    # system once per chunk, and that is a *library* capability, not model maths.
    "triangular_solve",
)


@dataclass(frozen=True, slots=True)
class Caps:
    """What a backend is, on this host: identity, dtypes, and the facts model code branches on.

    Every field here replaces something that is currently hardcoded outside the
    backend: the ``"cuda"`` literal, ``ops/dtypes.py``'s ``D``/``WF``, the
    ``hip_ops.available()`` checks, ``ATTN_SPLIT`` from
    ``models/qwen35/tuning.py``, and the ``hasattr(hip_ops, ...)`` probes.
    """

    #: Backend identifier: what ``COFFIN_BACKEND`` accepts and what log lines print.
    name: str
    #: torch device *type* the backend executes on, i.e. the attribute name on
    #: ``torch``: ``"cuda"``, ``"gcu"``, ``"musa"``, ``"xpu"``, ``"cpu"``. It is a
    #: ``torch.device``-compatible string; use :func:`make_device` (or
    #: ``backends.device()``) for the resolved ``torch.device``.
    #:
    #: This is a namespace, **not** a vendor. ``"cuda"`` is implemented by ROCm, by
    #: Iluvatar CoreX and by MUSA's compat layer alike, so a value of ``"cuda"`` says
    #: nothing about whose card it is -- :attr:`device_name` carries that.
    device: str
    #: Ordinal within :attr:`device`, i.e. the ``:N`` of ``torch.device(type, N)``.
    device_index: int = 0
    #: Board name from the driver (``"Z200SM_80"``, ``"Iluvatar BI-V150"``,
    #: ``"MTT S4000"``), or ``""`` when unknown.
    #:
    #: Not a log-only field: inside a **shared** device namespace it is the only thing
    #: that distinguishes vendors, so it is what backend detection keys on. Do not
    #: demote it to decoration, and do not reintroduce an ``is_available()`` check as
    #: the platform test.
    device_name: str = ""
    #: Target ISA the kernels were built for, read from the driver rather than
    #: hardcoded (``"gfx906"`` on this DCU), or ``""`` when the backend does not
    #: target a named arch. Build flags derive from this, so a different part
    #: needs a different value, not a second build path.
    arch: str = ""
    #: Op names this backend serves with a compiled kernel. Its complement runs
    #: the reference implementation, which is why a model call site never needs to
    #: know the difference. Empty for the pure-torch reference; the startup line
    #: prints ``len(native)/len(Ops.NAMES)``, which is the migration's progress
    #: bar for a new platform.
    native: frozenset[str] = frozenset()
    #: Activation dtype the kernels and the int8 KV cache are written against.
    #: Replaces ``ops.dtypes.D``.
    act_dtype: torch.dtype = torch.float32
    #: Projection-weight dtype. Replaces ``ops.dtypes.WF``; on gfx906 it is fp16
    #: because that part has packed fp16 math and no packed bf16 math, while on a
    #: part with packed bf16 it should follow :attr:`act_dtype` instead.
    weight_dtype: torch.dtype = torch.float32
    #: The decode step can be captured into a device graph and replayed. A backend
    #: whose kernels launch outside the capturing stream must report ``False``:
    #: such a graph replays stale outputs instead of failing, which is a silent
    #: correctness bug rather than a performance one.
    graph: bool = False
    #: Kernels may be benchmarked per shape at load time to pick a config. ``False``
    #: means the backend takes a static default, for a platform whose timing
    #: primitives do not exist (``torch.cuda.Event`` on xpu) or where the sweep
    #: would cost more than it saves.
    autotune: bool = False
    #: Split-K width of the attention kernels. This is a kernel ABI, not a tuning
    #: knob: the caller allocates ``part_acc``/``part_ml`` from it, and a mismatch
    #: corrupts attention rather than failing a shape check. ``1`` means the
    #: backend does not split, so it needs no caller-owned partials.
    attn_split: int = 1
    #: Batched paged attention is native, so a batch bucket costs one attention
    #: launch instead of one per row. Distinct from the table entry existing:
    #: every backend has ``attn_paged_batch``, but only some run it native.
    batch_attn: bool = False
    #: The backend reads ``int8_weights.QWeight`` natively. When ``False`` the
    #: ``linear*`` ops dequantise to :attr:`weight_dtype` and GEMM, which is
    #: correct but reads twice the bytes per weight element.
    int8_weights: bool = False
    #: The backend owns snapshot/restore of the linear-attention recurrent state
    #: for prefix reuse. The hybrid model cannot reuse a prefix without it: shared
    #: KV paired with a recurrence that never saw those tokens answers wrongly.
    #: ``False`` forces prefix reuse off for this backend rather than silently
    #: producing wrong text.
    #:
    #: True for every torch backend in practice: the state is a dense fp32 tensor and
    #: snapshot/restore is a ``clone()``/``copy_()`` on the engine side, with no kernel
    #: involved. A backend should report ``False`` only if that copy is unsupported or
    #: not trusted, not merely because it has no native snapshot kernel -- doing so
    #: throws away the main token-latency win for nothing.
    recurrent_state: bool = False
    #: Layout this backend wants its projection weights stored in: ``"nk"`` (the
    #: checkpoint's ``[N, K]`` row-major, this engine's current convention) or
    #: ``"kn"`` (``[K, N]``, i.e. ``x @ Wt``).
    #:
    #: Not cosmetic. On an Iluvatar BI-V100 the ``[N, K]`` layout that ``nn.Linear``
    #: produces measured 19-30 GB/s and accounted for 92% of a decode step, while
    #: ``[K, N]`` reached 362-410 GB/s: one layout decision, 14-22x. The op contract
    #: stays ``[N, K]`` because that is the checkpoint; a ``"kn"`` backend transposes
    #: once at load time and owns both the cost and the extra buffer.
    weight_layout: str = "nk"
    #: Integer dtype the backend can pass to kernels for positions, slot maps and
    #: block indices.
    #:
    #: ``int64`` is the default because it is what this engine's index tensors are.
    #: Some platforms have no 64-bit integer path -- Enflame's torch_gcu silently
    #: downcasts to 32-bit and warns, which for a ``data_ptr`` also makes the reported
    #: address untrustworthy -- so they must declare ``torch.int32`` and the caller
    #: keeps indices inside that range instead of discovering the truncation later.
    index_dtype: torch.dtype = torch.int64


@dataclass(frozen=True, slots=True)
class Ops:
    """One backend's flat table of device ops; the field name is the contract.

    The table is *total*: every backend answers all 18 names. A native backend
    returns its kernel; and the resolver in :mod:`coffin.backends` fills any name
    the backend leaves ``None`` with the reference implementation. ``None`` is
    therefore only meaningful on a native backend's own table, where it means
    "no kernel here" -- never a gap that reaches model code.

    Every entry takes the same positional arguments on every backend. Buffers
    named ``out``, ``conv_state``, ``st`` and the ``*_out`` prologue outputs are
    caller-owned and written in place, because the decode step is captured into a
    graph whose buffer addresses must not move.
    """

    #: ``linear(w, x) -> y``: decode-time weight read. ``x`` is ``[K]`` or
    #: ``[C, K]`` and ``y`` is ``[N]`` or ``[C, N]`` in ``x``'s dtype; ``w`` is
    #: ``[N, K]`` row-major, a ``torch.Tensor`` or a ``QWeight``. One name covers
    #: both storage formats and every single-row kernel variant, so no call site
    #: carries a tuning choice.
    linear: OpFn | None = None
    #: ``linear_m(w, x) -> y``: small-M batch (``x`` ``[M, K]``, ``y`` ``[M, N]``)
    #: that reads the weight once for all ``M`` rows. The backend decides which
    #: ``M`` it specialises and how to split the rest; the caller only knows it
    #: has more than one row.
    linear_m: OpFn | None = None
    #: ``linear_prefill(w, x) -> y``: the large-M regime, where a GEMM beats a
    #: weight-streaming kernel. On int8 weights a backend without a dequantising
    #: GEMM materialises ``w`` at :attr:`Caps.weight_dtype` for the call, which is
    #: the right trade because prefill is compute-bound.
    linear_prefill: OpFn | None = None
    #: ``rms_norm(x, w, stride=0, eps=1e-6) -> y``: zero-centred RMSNorm over the
    #: last axis of ``[M, N]`` (effective coefficient ``1 + w``, folded by the
    #: caller). ``stride`` is the element pitch between rows so a view into a wider
    #: buffer normalises in place without a copy; ``0`` means "read it from ``x``".
    #: A backend that does not take a separate pitch ignores the argument.
    rms_norm: OpFn | None = None
    #: ``rms_norm_gated(x, w, gate, eps=1e-6) -> y``: ``rms(x) * w * silu(gate)``,
    #: normalised per head, so ``w`` is ``[dv]`` and not the flattened width.
    rms_norm_gated: OpFn | None = None
    #: ``attn_decode(q, gate, kcache, vcache, kscale, vscale, out, pos, rep, scale,
    #: part_acc=None, part_ml=None) -> out``: one token over the dense int8 KV
    #: cache. ``pos`` is a 1-element long tensor that addresses the slot and is
    #: also the causal bound (entries past it are masked, not skipped, so the shape
    #: stays static). ``gate`` is applied as ``sigmoid``. ``part_acc``/``part_ml``
    #: are the caller-owned split-K scratch sized from :attr:`Caps.attn_split`;
    #: passing ``None`` lets the backend allocate, which is correct but not
    #: graph-capturable.
    attn_decode: OpFn | None = None
    #: ``attn_paged(q, gate, kpool, vpool, kscale, vscale, pos2slot, out, pos,
    #: max_pos, rep, scale, part_acc=None, part_ml=None) -> out``: the same
    #: attention over one layer of the paged int8 pool. ``kpool``/``vpool`` are
    #: ``[nslots, nkv, hd]`` int8, ``kscale``/``vscale`` ``[nslots, nkv]`` fp32,
    #: ``pos2slot`` the row's ``[max_pos]`` slot map of which the first
    #: ``pos + 1`` entries are read. Must agree with :attr:`attn_decode` when
    #: ``pos2slot`` is the identity map.
    attn_paged: OpFn | None = None
    #: ``attn_paged_batch(q, gate, kpool, vpool, kscale, vscale, slots,
    #: context_lens, out, max_seq, rep, scale, part_acc=None, part_ml=None) -> out``:
    #: one launch for ``B`` rows, row ``b`` reading only the first
    #: ``context_lens[b]`` entries of its own ``slots[b]``, which is what lets a
    #: bucket of mixed prompt lengths share a launch. Each row must stay
    #: numerically identical to :attr:`attn_paged` on that row alone.
    attn_paged_batch: OpFn | None = None
    #: ``attn_paged_multi_q(q, gate, kpool, vpool, kscale, vscale, slots, ctx_lens,
    #: out, max_seq, rep, scale, part_acc=None, part_ml=None, block_mask=None,
    #: mask_block=64) -> out``: speculative verify, ``q`` ``[B, c, nh, hd]``, where
    #: the ``c`` draft queries of a row share one pass over that row's history.
    #: Query ``j`` sees ``ctx_lens[b] + j + 1`` keys, i.e. it is bit-identical to
    #: decoding that token alone. ``block_mask`` (uint8 ``[B, n_blocks]``, 1 = keep)
    #: restricts which KV blocks take part; ``None`` is exact full attention.
    attn_paged_multi_q: OpFn | None = None
    #: ``conv1d_silu(conv_w, x, conv_state, out) -> out``: depthwise causal conv
    #: (kernel 4) + SiLU for one token. ``conv_w`` is ``[C, 4]`` fp32 with the
    #: current token last; ``conv_state`` is ``[C, 3]`` fp32 and is shifted in
    #: place, so it is the mixer's short-term memory and has to be snapshotted
    #: with the recurrent state.
    conv1d_silu: OpFn | None = None
    #: ``conv1d_silu_seq(conv_w, x, conv_state) -> y``: the same conv over a
    #: prefill chunk ``[T, C]`` -> ``[T, C]`` fp32, carrying ``conv_state`` across
    #: chunks so a chunked prefill and a one-shot prefill agree exactly. fp32
    #: because the delta scan downstream accumulates over the chunk.
    conv1d_silu_seq: OpFn | None = None
    #: ``conv1d_silu_batch(conv_w, x, conv_state, out) -> out``: ``B`` rows, ``x``
    #: ``[B, C]``, ``conv_state`` ``[B, C, 3]``; each row identical to
    #: :attr:`conv1d_silu` on that row alone.
    conv1d_silu_batch: OpFn | None = None
    #: ``delta_prologue(q, k, v, b, a, neg_expA, dt_bias, q_out, k_out, v_out,
    #: beta, g, qscale, rep) -> None``: one step of head preparation --
    #: L2-normalise q/k, expand key heads to value heads (``rep``), and build
    #: ``beta = sigmoid(b)`` and ``g = neg_expA * softplus(a + dt_bias)`` into fp32
    #: out buffers. ``q`` gets ``qscale`` and the norm; ``k`` gets the norm only.
    #: ``g`` must stay ``<= 0``, a decay per step and never a growth.
    delta_prologue: OpFn | None = None
    #: ``delta_prologue_batch(...) -> None``: batched :attr:`delta_prologue`,
    #: a leading row dim on ``q``/``k``/``v``/``b``/``a`` and the outputs;
    #: ``neg_expA``/``dt_bias`` stay 1-D, shared by every row.
    delta_prologue_batch: OpFn | None = None
    #: ``delta_prep_seq(qkv, b, a, neg_expA, dt_bias, q_out, k_out, v_out, beta, g,
    #: qscale, rep) -> None``: prefill head prep. Writes q/k/v head-major as
    #: ``[num_v, T, d]`` and beta/g as ``[num_v, T]``, which is the layout the
    #: chunked delta scan reads directly, so no transposed copy of the chunk is
    #: needed.
    delta_prep_seq: OpFn | None = None
    #: ``delta(q, k, v, beta, g, st, out) -> out``: one gated-delta decode step.
    #: Updates ``st`` ``[num_v, dk, dv]`` fp32 in place and writes ``out``
    #: ``[num_v, dv]`` fp32; ``out`` is computed from the *updated* state. The
    #: recurrence is the layer's only cross-token memory and has no closed-form
    #: rewind, so a caller that needs rollback snapshots ``st`` first.
    delta: OpFn | None = None
    #: ``delta_batch(q, k, v, beta, g, st, out) -> out``: as :attr:`delta` with a
    #: leading row dim on every argument; each row must stay identical to
    #: :attr:`delta` on that row alone.
    delta_batch: OpFn | None = None
    #: ``triangular_solve(lower, rhs) -> out``: solve ``A X = B`` where ``A`` is
    #: ``[bn, L, L]`` unit-lower-triangular and ``B`` is ``[bn, L, n]``, returning
    #: ``[bn, L, n]`` in ``rhs``'s dtype.
    #:
    #: This is an op because a *library* has to answer it, not the model. The
    #: chunked delta rule needs ``A^-1 B`` once per chunk per layer, and the
    #: obvious spelling -- ``torch.linalg.solve_triangular`` -- is unavailable on
    #: CoreX: that build's cuSOLVER is loaded from a hardcoded
    #: ``/opt/sw_home/local/cuda/lib64/libcusolver.so`` which does not exist on the
    #: cluster, so every ``linalg`` solve path fails inside ``dlopen``. A blocked
    #: forward substitution written with ``bmm`` has no library dependency at all
    #: and measured 1.1e-07 against a host fp64 solve.
    #:
    #: ``unitriangular`` is fixed ``True`` rather than exposed: the delta rule's
    #: ``UT`` system always has a unit diagonal (it is ``I`` plus strictly-lower
    #: products), so a flag would be a knob no caller could meaningfully set.
    triangular_solve: OpFn | None = None

    #: Canonical op names, in field order.
    NAMES: ClassVar[tuple[str, ...]] = _OP_NAMES

    def __post_init__(self) -> None:
        """Reject an entry that is neither a callable nor ``None``.

        Raises:
            TypeError: An entry is not callable. Registering a non-callable would
                surface as a confusing ``TypeError`` at the first decode step
                rather than at table construction, and only for one shape.
        """
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None and not callable(value):
                raise TypeError(
                    f"op {field.name!r} must be a callable or None, "
                    f"got {type(value).__name__}")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, OpFn | None]) -> Ops:
        """Build a table from a partial ``{name: callable}`` mapping.

        Args:
            mapping: Op names to callables; ``None`` values are redundant but
                accepted. Every name absent from ``mapping`` becomes ``None``.

        Returns:
            The table. Absent entries are what the resolver fills from the
            reference backend, so a native backend only lists what it implements.

        Raises:
            ValueError: ``mapping`` names an op outside :attr:`NAMES`. A typo
                would otherwise produce a table that silently lacks that op, which
                looks exactly like a platform without the kernel.
        """
        unknown = sorted(set(mapping) - set(_OP_NAMES))
        if unknown:
            raise ValueError(
                f"unknown op(s): {', '.join(unknown)}; "
                f"known: {', '.join(_OP_NAMES)}")
        return cls(**{name: mapping.get(name) for name in _OP_NAMES})

    def as_dict(self) -> dict[str, OpFn | None]:
        """Return the table as ``{name: callable-or-None}`` in canonical order.

        Returns:
            One entry per op name, including the ``None`` ones. Callers iterate this
            rather than ``dataclasses.fields`` so the type checker keeps the
            callable-or-``None`` shape.
        """
        return {name: getattr(self, name) for name in _OP_NAMES}

    def implemented(self) -> tuple[str, ...]:
        """List the op names this table answers itself.

        Returns:
            The names with a callable, in canonical order. On a native backend
            this is its kernel inventory; on the reference it is all 18.
        """
        return tuple(name for name, fn in self.as_dict().items() if fn is not None)

    def missing(self) -> tuple[str, ...]:
        """List the op names this table leaves ``None``.

        Returns:
            The names the resolver must fill from the reference backend, in
            canonical order. Empty means the table is already total.
        """
        return tuple(name for name, fn in self.as_dict().items() if fn is None)


@runtime_checkable
class Backend(Protocol):
    """A platform: how to find it, what it can do, and its op table.

    The registry only ever touches these members, so a backend is a module with
    these six attributes rather than a class hierarchy. Every backend is
    instantiated once per process: selecting a backend is an import-time decision,
    and there is no runtime hot swap.
    """

    #: Identifier accepted by ``COFFIN_BACKEND`` and printed in the startup line.
    name: str

    def probe(self) -> bool:
        """Report whether the hardware this backend targets is on this host.

        Must be cheap and side-effect free -- no kernel build, no allocation, no
        device reset -- because the registry probes every candidate when
        auto-detecting. ``False`` means "not this machine", not "broken".

        Returns:
            ``True`` when the device or driver the backend needs is visible.
        """

    def usable(self) -> bool:
        """Report whether this backend can execute the model right now.

        Stricter than :meth:`probe`: the kernel build (if any) must succeed too.
        The registry falls through to the next candidate while auto-detecting, so
        this is the gate that keeps a host with the driver but no usable toolchain
        on the eager path instead of taking the process down.

        Returns:
            ``True`` when :meth:`ops` yields a table the model can run on.
        """

    def reason(self) -> str:
        """Explain in one line why :meth:`usable` is ``False``.

        Returns:
            The diagnostic to show in the selection error and the startup log;
            ``""`` when usable. This is what keeps a failed kernel build from
            reading as a slow model.
        """

    def ops(self) -> Ops:
        """Return this backend's op table.

        May be expensive on the first call (it builds or attaches the compiled
        kernels), and must be memoised: the engine calls it thousands of times per
        request through the module-level table. Never partially populated -- an op
        without a native kernel is ``None`` and the resolver fills it.

        Returns:
            The table. An unusable backend still returns a table (all ``None``) so
            the caller can log and inspect it instead of handling an exception
            per call site.
        """

    def caps(self) -> Caps:
        """Return the capability flags for this backend on this host."""

    @property
    def device(self) -> torch.device:
        """The ``torch.device`` every allocation for this backend should use.

        Returns:
            ``torch.device(caps.device, caps.device_index)``. Model code replaces
            its ``device="cuda"`` literals with this, so a new platform changes
            one value instead of 173 call sites.
        """


def make_device(caps: Caps) -> torch.device:
    """Resolve a :class:`Caps` record into the ``torch.device`` to allocate on.

    Args:
        caps: The backend's capability record.

    Returns:
        ``torch.device(caps.device, caps.device_index)``; the index is dropped for
        ``"cpu"`` because ``torch.device("cpu", 0)`` is not accepted on every
        supported torch version.

    Raises:
        ValueError: ``caps.device`` is not a device type torch knows. Failing here
            beats failing at the first allocation halfway through a model build.
    """
    if caps.device == "cpu":
        return torch.device("cpu")
    return torch.device(caps.device, caps.device_index)
