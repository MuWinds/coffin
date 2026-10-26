"""How a platform declares itself: detection data, capability policy, optional kernels.

This module is the difference between "add a vendor by editing the registry in three
places" and "add a vendor by dropping in one file". A vendor describes itself with a
:class:`VendorSpec`; :func:`coffin.backends.discovery` finds it, and
:class:`SpecBackend` turns it into the object the registry drives. No central list is
edited when a platform is added, and no per-vendor glue code is written.

Two kinds of platform, one mechanism
------------------------------------
Measuring the target vendors turned up a split that the design has to absorb:

* **Independent namespace.** Enflame exposes ``torch.gcu`` and Moore Threads exposes
  ``torch.musa`` -- separate attributes on ``torch``, and neither redirects
  ``torch.cuda``. For these, the namespace alone identifies the vendor.
* **Shared namespace.** Iluvatar's CoreX implements the CUDA API, so an Iluvatar
  card answers through ``torch.cuda`` exactly like the Hygon DCU this engine was
  written on: ``torch.cuda.is_available()`` is True and ``get_device_name()`` returns
  the part (verified outside this repo: the vendor's own adaptation code compares
  ``torch.cuda.get_device_name() == "Iluvatar BI-V100"``, and a ComfyUI run logs
  ``Device: cuda:0 Iluvatar BI-V150``). Here the namespace says nothing and the
  device *name* is the only discriminator.

:func:`detect` returns a **claim strength** rather than a boolean, which is what lets
one mechanism cover both. A vendor that recognised the device by name claims it
strongly; a vendor that only matched the namespace claims it weakly; the registry
takes the strongest claim per device. So on an Iluvatar card both the ROCm backend
and the Iluvatar backend may answer "there is a CUDA device here", and the Iluvatar
one wins on strength. That also means the layout does **not** depend on settling
whether CoreX sets ``torch.version.hip`` -- an open question (one report says its
runtime raises ``No HIP GPUs are available``, a string that only exists in HIP builds,
which would make the ROCm build marker useless as an exclusion). Priority by claim
strength is correct either way.

The three parts of a declaration
--------------------------------
*Detection* -- data (:class:`Detection`), not code, because the question is the same
for every vendor and only the constants differ.
*Policy* -- the fields that become :class:`~coffin.backends.protocol.Caps`. These are
declarations, not measurements; a vendor file must not turn a datasheet number into a
"measured" flag.
*Kernels* -- ``build_ops`` returns a **partial** table naming only what the vendor
implements; the resolver fills the rest from :mod:`coffin.backends.reference`. That is
what lets a vendor start with one kernel (usually ``linear``) and grow, without the
model needing a fallback branch.

Module: ``coffin/backends``. Imports ``protocol`` and ``reference``; never
``coffin/ops``, ``models``, ``engine``, ``serving`` or ``cache``. It must not import a
vendor's runtime at module scope -- the ops table is built lazily, so importing this
module works on a host that has never seen the hardware.

Constraints:
    * :func:`detect` must stay cheap and side-effect free: the registry probes every
      discovered vendor on every selection, and a probe that initialises a device
      runtime turns a CPU-only import into a driver load.
    * A spec may never claim usability for a platform it has not proven. ``has_kernels``
      exists so a placeholder is distinguishable from a working backend without
      inspecting its op table; placeholders are discovered and reported, never
      auto-selected.
    * Claim strength is ordered: a name match always beats a namespace-only match.
      Swapping that order is how the ROCm backend would steal an Iluvatar card.
"""

from __future__ import annotations

import dataclasses
import functools
from dataclasses import dataclass
from typing import Callable

import torch

from backends import reference
from backends.protocol import Caps, Ops, make_device

#: Claim strengths returned by :func:`detect`, in increasing specificity.
#: ``CLAIM_NONE`` means "not my card"; anything above it means "mine", and the
#: highest value wins when two vendors answer for the same device.
CLAIM_NONE = 0
#: Matched only the device namespace. Correct when the namespace is unique to the
#: vendor (``gcu``, ``musa``), and the default owner when it is not (``cuda``).
CLAIM_NAMESPACE = 1
#: Matched a device-name marker. This is the only way to win a shared namespace, so
#: it outranks the namespace default by construction.
CLAIM_NAME = 2


@dataclass(frozen=True, slots=True)
class Detection:
    """The data needed to answer "is this the card I was written for?".

    Every field is a constant a vendor can state without writing code. The combination
    is what makes detection unambiguous when several vendors share a device namespace.

    Attributes:
        namespace: ``torch`` device namespace the vendor's build exposes, i.e. the
            attribute name on ``torch`` (``"cuda"``, ``"gcu"``, ``"musa"``, ``"xpu"``).
            This is a namespace, **not** a vendor: ``torch.cuda`` is implemented by
            CoreX, MUSA and ROCm builds alike.
        name_markers: Lower-cased substrings, any of which may appear in
            ``get_device_name(0)``. A match claims the device at :data:`CLAIM_NAME`.
            Empty means the vendor accepts whatever is left in its namespace at
            :data:`CLAIM_NAMESPACE` -- correct for a unique namespace, and for a
            shared one it is the "default owner" position.
        hip: Require a ROCm build (``torch.version.hip`` set). Used by the Hygon
            backend as a *gate*, not as an identity test: it excludes a plain NVIDIA
            CUDA card, while a CoreX card that also sets the marker loses to
            the Iluvatar vendor on claim strength.
    """

    namespace: str
    name_markers: tuple[str, ...] = ()
    hip: bool = False


def device_name(namespace: str, index: int = 0) -> str:
    """Read the device name for a namespace, without assuming which one it is.

    Args:
        namespace: Device namespace, i.e. the attribute name on ``torch``.
        index: Device ordinal.

    Returns:
        ``get_device_name(index)`` for that namespace, or ``""`` when the namespace or
        the call is unavailable. Empty is the honest answer on a host without the
        hardware, and callers treat it as "no name to match".
    """
    module = getattr(torch, namespace, None)
    getter = getattr(module, "get_device_name", None)
    if getter is None:
        return ""
    try:
        return str(getter(index))
    except Exception:                             # pragma: no cover - driver error
        return ""


def detect(detection: Detection) -> int:
    """Return how strongly this host matches a vendor's detection data.

    Checks are ordered by cost: namespace presence, availability, build marker, then
    the device-name read (the only one that touches the driver beyond availability).
    A failure at any step returns :data:`CLAIM_NONE` immediately, which is what keeps
    probing every discovered vendor cheap on a CPU-only host.

    Args:
        detection: The vendor's detection constants.

    Returns:
        :data:`CLAIM_NONE`, :data:`CLAIM_NAMESPACE`, or :data:`CLAIM_NAME`. The caller
        compares strengths across vendors, so a vendor must never return a non-zero
        value for a device it does not want to own.
    """
    module = getattr(torch, detection.namespace, None)
    if module is None:
        return CLAIM_NONE
    try:
        if not module.is_available():
            return CLAIM_NONE
    except Exception:                             # pragma: no cover - driver error
        return CLAIM_NONE
    if detection.hip and getattr(torch.version, "hip", None) is None:
        # A CUDA card from a vendor whose build is not ROCm-derived. Answering "yes"
        # here is how a ROCm backend ends up compiling for an ISA the card lacks.
        return CLAIM_NONE
    if not detection.name_markers:
        return CLAIM_NAMESPACE
    name = device_name(detection.namespace).lower()
    if not name:
        return CLAIM_NONE
    if any(marker.lower() in name for marker in detection.name_markers):
        return CLAIM_NAME
    return CLAIM_NONE


@dataclass(frozen=True, slots=True)
class VendorSpec:
    """One platform's declaration: how to find it, what it promises, what it runs.

    A vendor file defines exactly one of these as ``SPEC``. Everything the registry and
    the banner need is derived from it, so the file is data plus, once kernels exist, a
    ``build_ops`` hook.

    Attributes:
        name: Backend identifier accepted by ``COFFIN_BACKEND`` and printed in the
            startup line. Must be unique across discovered vendors.
        detection: How to recognise the platform; see :class:`Detection`.
        act_dtype: Activation dtype this platform's fused math is written against.
        weight_dtype: Projection-weight dtype expected to be fastest there. The single
            largest prefill lever: a part with packed fp16 and no packed bf16 runs
            13-20 TFLOP/s against 2.3 on bf16, so the choice is a hardware fact.
        graph: Whether device-graph capture is expected to work. ``False`` keeps decode
            eager, which is slower but never replays stale outputs.
        autotune: Whether load-time kernel benchmarking is expected to work. Needs
            working event timing in the backend's namespace; note the Hygon engine's
            ``torch.cuda.Event`` sweep works on Iluvatar too, so ``False`` there would
            be an unnecessary loss rather than a safety measure.
        int8_weights: Whether the backend can consume int8 weights without
            dequantising a whole matrix per call.
        batch_attn: Whether batched attention kernels exist, i.e. whether a batch
            bucket costs one launch instead of one per row.
        attn_split: Split-K width of the attention kernel. A kernel ABI, not a tuning
            hint: the caller sizes the online-softmax partial buffers from it. ``1``
            means unsplit and is the only safe declaration before a kernel exists.
        weight_layout: ``"nk"`` or ``"kn"`` -- the layout this platform wants its
            projection weights stored in. This is not cosmetic: on the Iluvatar BI-V100
            the ``[N, K]`` row-major form that ``nn.Linear`` produces measured 19-30
            GB/s and accounted for 92% of a decode step, while ``[K, N]`` with
            ``x @ Wt`` reached 362-410 GB/s. The contract keeps ``[N, K]`` (it is the
            checkpoint layout); a ``"kn"`` backend transposes once at load and owns the
            cost, which is why this is a spec field and not a contract change.
        index_dtype: Integer dtype the backend can pass to kernels for positions and
            slot maps. Enflame's torch_gcu has no I64: it silently downcasts to 32-bit and
            warns, so a backend there must declare ``torch.int32`` and the caller keeps
            indices in range.
        priority: Tie-break among vendors with equal claim strength, lower first. Ties
            are an error rather than resolved by import order, because which backend
            claimed a card must not depend on filesystem order.
        notes: What is *unmeasured* about this platform. Copied into the selection
            reason so a declared flag cannot be mistaken for a measured one.
        has_kernels: Whether ``build_ops`` returns real kernels. A placeholder is
            discovered and its name resolves, but ``usable()`` stays ``False`` so
            auto-detect falls through instead of pretending to accelerate.
        build_ops: Returns the vendor's partial op table. ``None`` is equivalent to "no
            native op yet"; the resolver fills the table from the reference backend, so
            the model still runs.
    """

    name: str
    detection: Detection
    act_dtype: torch.dtype = torch.bfloat16
    weight_dtype: torch.dtype = torch.bfloat16
    graph: bool = False
    autotune: bool = False
    int8_weights: bool = False
    batch_attn: bool = False
    attn_split: int = 1
    weight_layout: str = "nk"
    index_dtype: torch.dtype = torch.int64
    priority: int = 100
    notes: str = ""
    has_kernels: bool = False
    build_ops: Callable[[], Ops] | None = None

    def validate(self) -> None:
        """Reject a declaration that would silently misbehave.

        Raises:
            ValueError: The name or namespace is empty, ``weight_layout`` is not one of
                the two defined layouts, ``has_kernels`` is set with no ``build_ops``,
                or ``attn_split`` is not positive. Each of those produces a backend
                that is discoverable but broken in a way that only shows up at the
                first call, on hardware nobody has.
        """
        if not self.name:
            raise ValueError("a vendor spec needs a name")
        if not self.detection.namespace:
            raise ValueError(f"{self.name}: detection.namespace is required")
        if self.weight_layout not in ("nk", "kn"):
            raise ValueError(
                f"{self.name}: weight_layout must be 'nk' or 'kn', "
                f"got {self.weight_layout!r}")
        if self.attn_split < 1:
            raise ValueError(f"{self.name}: attn_split must be >= 1")
        if self.has_kernels and self.build_ops is None:
            raise ValueError(
                f"{self.name}: has_kernels is True but no build_ops was given; "
                f"a backend that claims native ops must supply them")


class SpecBackend:
    """A :class:`VendorSpec` wearing the :class:`Backend` interface.

    The registry drives backends through the same members whether they are a
    hand-written module (``hip``, ``reference``) or a declaration (everything under
    ``vendors/``). This class is the adapter, so a vendor file stays data and never
    repeats plumbing.

    Construction is cheap and side-effect free; the op table is built on first
    :meth:`ops` call, because that is where a vendor links or compiles its kernels.
    """

    def __init__(self, spec: VendorSpec) -> None:
        """Wrap a validated spec.

        Args:
            spec: The vendor declaration. :meth:`VendorSpec.validate` runs here so a
                bad declaration fails at discovery, not at first use.

        Raises:
            ValueError: The spec is invalid; see :meth:`VendorSpec.validate`.
        """
        spec.validate()
        self.spec = spec
        self.name = spec.name

    def claim(self) -> int:
        """Report how strongly this host looks like the vendor's platform.

        Returns:
            The strength from :func:`detect`. The registry compares it across vendors,
            so an Iluvatar card is won by the Iluvatar spec rather than by the ROCm
            backend that also sees a CUDA device.
        """
        return detect(self.spec.detection)

    def probe(self) -> bool:
        """Report whether this vendor's platform is present.

        Returns:
            ``True`` for any non-zero claim.
        """
        return self.claim() > CLAIM_NONE

    def usable(self) -> bool:
        """Report whether this backend can run the model right now.

        Returns:
            ``probe() and has_kernels``. A placeholder is therefore never
            auto-selected: on a host that has the card it is detected and reported,
            and the registry falls through to the reference backend rather than
            running the model slower under the pretence of acceleration.
        """
        return self.probe() and self.spec.has_kernels

    def reason(self) -> str:
        """Explain why :meth:`usable` is ``False``, for the selection error text.

        The distinction between "no device" and "device, no kernels" is the point: on
        real vendor hardware the operator must be told the integration is
        unimplemented, not sent looking for a driver problem.

        Returns:
            ``""`` when usable, otherwise a sentence naming the missing half.
        """
        if self.usable():
            return ""
        if not self.probe():
            return (f"no {self.name} device visible through torch "
                    f"(namespace {self.spec.detection.namespace!r} not reporting one)")
        detail = f"; {self.spec.notes}" if self.spec.notes else ""
        return f"{self.name} detected but no kernels implemented yet{detail}"

    @functools.lru_cache(maxsize=1)
    def caps(self) -> Caps:
        """Build the capability record, resolving the device name from the driver.

        Memoised because the model reads it on every allocation decision; a device-name
        read per call would be a driver round trip per buffer.

        Returns:
            The record model code reads for dtypes, device and behavioural flags.
            ``native`` is what ``build_ops`` actually returned, so the banner's
            ``native=N/17`` is derived rather than declared.
        """
        table = self._native_table()
        return Caps(
            name=self.name,
            device=self.spec.detection.namespace,
            device_index=0,
            device_name=device_name(self.spec.detection.namespace),
            arch="",
            native=frozenset(table.implemented()) if self.spec.has_kernels else frozenset(),
            act_dtype=self.spec.act_dtype,
            weight_dtype=self.spec.weight_dtype,
            graph=self.spec.graph,
            autotune=self.spec.autotune,
            attn_split=self.spec.attn_split,
            batch_attn=self.spec.batch_attn,
            int8_weights=self.spec.int8_weights,
            recurrent_state=True,
            weight_layout=self.spec.weight_layout,
            index_dtype=self.spec.index_dtype,
        )

    def _native_table(self) -> Ops:
        """Return what the vendor implements, without filling the gaps.

        Returns:
            The vendor's partial table, or an empty one for a placeholder. Kept separate
            from :meth:`ops` because ``caps.native`` must report what is really native,
            not what the reference filled in.
        """
        if not self.spec.has_kernels or self.spec.build_ops is None:
            return Ops()
        return self.spec.build_ops()

    @functools.lru_cache(maxsize=1)
    def ops(self) -> Ops:
        """Return the backend's total op table, native entries plus reference fill.

        This is the method that makes a partially-implemented platform work: a vendor
        that has only ``linear`` still answers all 17 names, because the model has no
        fallback branch left after migration. Memoised because the engine reaches the
        table thousands of times per request.

        Returns:
            A table where every canonical op name is callable.
        """
        return reference.fill(self._native_table(), self.caps())

    @functools.lru_cache(maxsize=1)
    def device(self) -> torch.device:
        """Return the ``torch.device`` allocations for this backend belong on.

        Returns:
            ``torch.device(namespace, 0)``, built lazily because forming a device can
            initialise a vendor runtime.
        """
        return make_device(self.caps())

    def describe(self) -> str:
        """Render this backend as one diagnostic line.

        Returns:
            ``"<name>: <state> (<reason>)"``, matching what
            :func:`coffin.backends.registry.describe_all` prints. Does not build a
            ``torch.device``, so it is safe to print before selection.
        """
        state = "usable" if self.usable() else ("placeholder" if self.probe() else "absent")
        reason = self.reason()
        return f"{self.name}: {state}{f' ({reason})' if reason else ''}"

    def __repr__(self) -> str:                    # pragma: no cover - debug aid
        return f"SpecBackend({self.name!r})"


def spec_fields() -> tuple[str, ...]:
    """Return :class:`VendorSpec`'s field names, in declaration order.

    Returns:
        The dataclass field names. A test pins this so a field added to the spec does not
        silently go unhandled by :class:`SpecBackend`, which reads every field by name
        and would otherwise drop a new one without complaint.
    """
    return tuple(f.name for f in dataclasses.fields(VendorSpec))
