"""Choose the one backend this process runs on: env override, then auto-detect, then reference.

Selection happens once, at import time, and is never revisited: there is no runtime hot
swap, so a backend change means a new process. The order is fixed and deliberately
boring -- ``COFFIN_BACKEND`` (exact name) wins for debugging and A/B, otherwise every
registered backend is probed and the best claim wins, with the pure-torch reference as
the guaranteed last candidate.

Two sources of backends, one list
---------------------------------
*Hand-written* backends (:data:`BUILTIN`) are modules that need real code: ``hip`` has
kernels, tuning and a build path, and ``reference`` is the fallback everything else
resolves to. *Discovered* backends live in ``backends/vendors/`` as declarative specs
(see :mod:`coffin.backends.discovery`), so adding a platform means adding one file and
editing nothing here. :func:`candidates` merges the two into the single ordered sequence
that selection, logging and the error text all read.

Detection is by **claim strength**, not by a boolean, because more than one vendor can
answer for the same device. Iluvatar's CoreX answers through ``torch.cuda`` exactly like
the Hygon DCU, so both may see "a CUDA device is present"; the vendor that matched the
device *name* outranks the one that only matched the namespace, and the Iluvatar spec
wins. On Enflame and Moore Threads the namespace itself is unique (``gcu``, ``musa``), so the
namespace claim is enough. That single rule covers both cases without the registry
knowing which vendors exist.

Module: ``coffin/backends``. Sits next to ``protocol`` at the bottom of the dependency
graph: it may import ``torch``, ``protocol``, ``vendor``, ``discovery`` and the built-in
backend modules, and nothing from ``coffin/ops``, ``models``, ``engine``, ``serving`` or
``cache``.

Constraints:
    * ``select()`` is the only place that raises about backend selection. It must name
      the requested backend, every candidate tried, and each candidate's
      :meth:`reason`, because a silently-chosen reference backend looks exactly like a
      mysteriously slow deployment.
    * An explicit ``COFFIN_BACKEND`` never falls through to another backend. A typo or a
      broken build must fail loudly; quietly running reference instead would hide the
      failure behind a 10x slowdown.
    * Importing this module must not build, compile or probe anything: it reads a
      directory listing and module names. ``import backends`` on a CPU-only host has to
      work, and it is what the test suite relies on.
"""

from __future__ import annotations

import importlib
import os
from typing import Sequence

from backends import discovery
from backends.protocol import Backend, Caps, Ops

#: Environment variable that pins the backend by name; empty/unset means auto-detect.
#: Read once, at import of :mod:`coffin.backends`.
ENV_VAR = "COFFIN_BACKEND"

#: Hand-written backends, in probe order. Everything under ``backends/vendors/`` is
#: discovered and merged in; only backends that need real code live here.
#:
#: ``hip`` and ``corex`` are hand-written rather than declarative vendor specs because
#: they carry kernels, kernel tiling, autotune and a build path -- none of which a
#: declaration expresses. They also share the ``cuda`` namespace, so which one wins is
#: decided by each backend's own probe (``torch.version.hip`` for ROCm, the device name
#: for CoreX), not by ordering here. ``reference`` is last because it always reports
#: usable, so anything after it would be unreachable.
BUILTIN: tuple[str, ...] = ("hip", "corex", "reference")

#: Module name of each built-in backend inside this package.
_BUILTIN_MODULE = {
    "hip": "backends.hip",
    "corex": "backends.corex",
    "reference": "backends.reference",
}

#: Probe priority of each built-in, used only to break a tie between two backends that
#: claimed the same device equally strongly. The reference is last because it always
#: answers usable; ``hip`` and ``corex`` sit at the same priority on purpose, since
#: neither may win a shared ``cuda`` namespace by ordering alone.
_BUILTIN_PRIORITY = {"hip": 0, "corex": 0, "reference": 1000}


class Candidate:
    """One selectable backend, with the claim it made on this host.

    Attributes:
        name: Backend identifier, i.e. what ``COFFIN_BACKEND`` accepts.
        backend: The object the protocol drives; a module for hand-written backends, a
            :class:`~coffin.backends.vendor.SpecBackend` for a declared vendor.
        claim: How strongly this host matched, from
            :func:`coffin.backends.vendor.detect` (``0`` for a hand-written backend that
            does not take part in claim ordering). Higher wins.
        priority: Tie-break among equal claims, lower first.
    """

    __slots__ = ("name", "backend", "claim", "priority")

    def __init__(self, name: str, backend: Backend, claim: int = 0,
                 priority: int = 0) -> None:
        """Record one backend and the claim it made on this host.

        Args:
            name: Backend identifier, i.e. what ``COFFIN_BACKEND`` accepts.
            backend: The module or :class:`~coffin.backends.vendor.SpecBackend` the
                protocol drives.
            claim: Claim strength from :func:`coffin.backends.vendor.detect`; ``0`` for
                a hand-written backend, which does not compete on claim.
            priority: Tie-break among equal claims, lower first. The reference carries
                the highest value so it always sorts last.
        """
        self.name = name
        self.backend = backend
        self.claim = claim
        self.priority = priority

    def rank(self) -> tuple[int, int, str]:
        """Sort key: strongest claim first, then priority, then name.

        Returns:
            ``(-claim, priority, name)``. Claim is negated because sorting ascending
            puts the strongest first; the trailing name keeps the order total, so two
            vendors that tie cannot make selection depend on import order.
        """
        return (-self.claim, self.priority, self.name)


def _builtins() -> tuple[Candidate, ...]:
    """Load every hand-written backend.

    Returns:
        One :class:`Candidate` per :data:`BUILTIN`, in declaration order. A built-in
        that fails to import propagates: these are first-party modules, so a failure is
        a bug in this repository rather than a platform that happens to be absent.
    """
    out = []
    for name in BUILTIN:
        module = importlib.import_module(_BUILTIN_MODULE[name])
        out.append(Candidate(name, module, priority=_BUILTIN_PRIORITY[name]))
    return tuple(out)


def _vendors() -> tuple[Candidate, ...]:
    """Load every discovered vendor backend and evaluate its claim on this host.

    The claim is read here, once, because it is the whole basis of the ordering; asking
    each vendor again later would allow the host and the ordering to disagree.

    Returns:
        One :class:`Candidate` per vendor file that imports cleanly. A vendor that
        fails to import is skipped: one broken platform file must not make the layer
        unselectable, and :func:`describe_all` is where it gets reported.
    """
    out = []
    for spec, backend in discovery.iter_vendors():
        try:
            claim = backend.claim()
        except Exception:                         # pragma: no cover - probe error
            claim = 0
        out.append(Candidate(spec.name, backend, claim=claim,
                             priority=spec.priority))
    return tuple(out)


def candidates() -> tuple[Candidate, ...]:
    """Return every selectable backend, strongest claim first.

    Returns:
        Hand-written backends and discovered vendors merged and sorted by
        :meth:`Candidate.rank`. The reference sorts last among equals because its
        priority is the highest, which is what makes it the guaranteed fallback.
    """
    return tuple(sorted(_builtins() + _vendors(), key=Candidate.rank))


def _backend_named(name: str) -> Backend | None:
    """Resolve one backend by name, without probing it.

    Args:
        name: Backend identifier from ``COFFIN_BACKEND`` or a caller.

    Returns:
        The backend object, or ``None`` when no candidate has that name. Tries the
        built-in module map first so an explicit pin does not pay for a vendor
        directory scan.

    Raises:
        ImportError: The named built-in exists but its module cannot import. Reported
            rather than swallowed, because an explicitly pinned backend that silently
            became something else is the failure this layer exists to prevent.
    """
    if name in _BUILTIN_MODULE:
        return importlib.import_module(_BUILTIN_MODULE[name])
    for spec, backend in discovery.iter_vendors():
        if spec.name == name:
            return backend
    return None


def available_names() -> tuple[str, ...]:
    """Return the backend names this build can select, in probe order.

    Returns:
        The names, strongest claim first. Discovered vendors are included, so a name
        added as a file is immediately a valid ``COFFIN_BACKEND`` value without editing
        this module.
    """
    return tuple(candidate.name for candidate in candidates())


def _load(name: str) -> Backend:
    """Resolve a backend by name or raise with the known names.

    Args:
        name: Backend identifier.

    Returns:
        The backend object.

    Raises:
        KeyError: No candidate has that name, or a discovered vendor file failed to
            import. The message lists what is available, since a missing vendor file is
            the usual cause.
    """
    backend = _backend_named(name)
    if backend is None:
        raise KeyError(f"unknown backend {name!r}; known: {', '.join(available_names())}")
    return backend


def _describe(candidate: Candidate) -> str:
    """One-line status for a candidate, for the selection error text.

    Args:
        candidate: The backend and the claim it made.

    Returns:
        ``"name: state (reason)"``. Only called on the failure path, because building
        the reason may import a vendor whose optional dependency is absent.
    """
    try:
        present = candidate.backend.probe()
        ok = candidate.backend.usable() if present else False
        why = "" if ok else (candidate.backend.reason() or "not usable")
    except Exception as exc:                      # pragma: no cover - probe error
        return f"{candidate.name}: probe raised {type(exc).__name__}: {exc}"
    state = "usable" if ok else ("present, unusable" if present else "absent")
    return f"{candidate.name}: {state}{(' (' + why + ')') if why else ''}"


def select(requested: str | None = None) -> Backend:
    """Pick the backend for this process.

    Args:
        requested: Backend name from ``COFFIN_BACKEND``, or ``None`` to read the
            environment and then auto-detect. Passing a value explicitly is what tests
            use to exercise the failure paths without touching the environment.

    Returns:
        The selected backend.

    Raises:
        RuntimeError: An explicit name is not a known backend, or the explicitly named
            backend is not usable. The message lists every candidate with its probe
            result and reason, since the usual cause is a driver or toolchain problem on
            the host rather than a wrong name.
        RuntimeError: Auto-detection found no usable backend. Only reachable if the
            reference backend itself is broken, so the message is a bug report.
    """
    explicit = requested if requested is not None else os.environ.get(ENV_VAR, "")
    explicit = (explicit or "").strip()

    if explicit:
        try:
            backend = _load(explicit)
        except KeyError as exc:
            raise RuntimeError(
                f"{ENV_VAR}={explicit!r} is not a known backend; "
                f"known: {', '.join(available_names())}. Unset it to auto-detect."
            ) from exc
        if not backend.usable():
            raise RuntimeError(
                f"{ENV_VAR}={explicit!r} was requested but is not usable on this "
                f"host: {backend.reason() or 'no reason reported'}. "
                f"Unset it to auto-detect, or set {ENV_VAR}=reference to run the "
                f"pure-torch backend.\nCandidates: "
                + "; ".join(_describe(candidate) for candidate in candidates()))
        return backend

    for candidate in candidates():
        try:
            if candidate.backend.probe() and candidate.backend.usable():
                return candidate.backend
        except Exception:                          # pragma: no cover - probe error
            continue

    raise RuntimeError(
        "no usable backend: the reference backend reported unusable, which is a bug "
        "in backends.reference rather than a host problem.\nCandidates: "
        + "; ".join(_describe(candidate) for candidate in candidates()))


#: The selected backend. Anything that needs the selection result imports one of the
#: accessors below instead of reaching for this name.
_backend: Backend | None = None


def backend() -> Backend:
    """Return the process-wide backend, selecting it on first use.

    Returns:
        The selected backend; the same object on every call, because selecting twice
        would either rebuild kernels or make two parts of the process disagree about the
        device.

    Raises:
        RuntimeError: Selection failed, with the diagnostic from :func:`select`.
    """
    global _backend
    if _backend is None:
        _backend = select()
    return _backend


def _reset_for_tests(module: Backend | None = None) -> None:
    """Force the memoised selection, so a test can exercise another backend.

    Args:
        module: Backend to pin, bypassing :func:`select`; ``None`` clears the memo so the
            next :func:`backend` call selects again.

    Returns:
        Nothing. Not public API: it exists because a test that reloaded the backend
        package has to be able to re-pin the module-level table.
    """
    global _backend
    _backend = module


def ops() -> Ops:
    """Return the selected backend's total op table.

    Returns:
        The table, with every name callable: the backend's native entries plus reference
        implementations for everything it does not implement. This is the object model
        code binds at import time.

    Raises:
        RuntimeError: Selection failed.
    """
    return backend().ops()


def caps() -> Caps:
    """Return the selected backend's capability record.

    Returns:
        The record model code reads for dtypes, device and behavioural flags.

    Raises:
        RuntimeError: Selection failed.
    """
    return backend().caps()


def status() -> str:
    """Render the one-line startup banner for the selected backend.

    Returns:
        ``backend=hip device=cuda:0 native=17/17 graph=1 autotune=1``. It replaces the
        ``hip=...`` field the transport used to print: without this line a run that
        silently fell back to the reference backend looks like a slow model instead of a
        missing kernel build.

    Raises:
        RuntimeError: Selection failed.
    """
    b = backend()
    c = b.caps()
    return (f"backend={c.name} device={b.device} "
            f"native={len(c.native)}/{len(Ops.NAMES)} "
            f"graph={int(c.graph)} autotune={int(c.autotune)}")


def describe_all() -> Sequence[str]:
    """Render one status line per candidate, for a diagnostic dump.

    Returns:
        One ``name: state`` string per candidate, in probe order. Unlike :func:`status`
        this touches every backend's ``probe``, so it is a diagnostic call, not a startup
        one. A vendor file that fails to import is included as an error line rather than
        omitted: a backend that vanishes without a word is the bug this guards against.
    """
    lines = [_describe(candidate) for candidate in candidates()]
    for module_name in discovery.modules():
        try:
            discovery.load(module_name)
        except Exception as exc:
            lines.append(f"{module_name}: BROKEN ({exc})")
    return tuple(lines)
