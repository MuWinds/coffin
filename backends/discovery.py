"""Find the vendor backends: one file per platform, no central list to edit.

Adding a platform is dropping in ``vendors/<name>.py`` (or a ``vendors/<name>/``
package, once it needs a ``kernels/`` directory) that defines a module-level ``SPEC``.
Nothing else changes: this module discovers it, :mod:`coffin.backends.registry` orders
it with the rest, and the CLI accepts its name. The alternative -- a tuple and a name
map edited per vendor -- is how a platform layer starts failing the moment someone
forgets one of the two places.

Discovery is a directory scan rather than an entry-point plugin system on purpose.
The platforms are first-party files in this repository, so a scan has no deployment
story to get wrong, and it keeps the failure mode local: a vendor file that raises on
import is reported as an error naming the file, not as a mysterious missing backend.

Module: ``coffin/backends``. Imports ``vendor`` and the standard library; the scanned
modules are imported, so this is the one place that executes vendor code -- and it does
so only when :func:`load` is called, not at import. A vendor module must therefore be
importable on a host that has never seen its hardware; that is a documented rule in
``backends/vendor.py``.

Constraints:
    * Imports are sorted by module name so the scan is deterministic. Ordering among
      equal-priority vendors must not depend on filesystem iteration, because which
      backend claims a card would then differ between machines.
    * A module that fails to import is reported, not skipped silently, unless it is
      one of the documented opt-outs (:data:`_IGNORED`). A vendor backend that
      vanished without a word is exactly the bug this layer exists to prevent.
    * Discovery must not build or probe anything: it imports modules and reads their
      ``SPEC``. Probing happens later, once, in the registry.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import Iterator

from backends.vendor import SpecBackend, VendorSpec

#: Directory holding the vendor declarations, i.e. ``coffin/backends/vendors``.
VENDOR_DIR = Path(__file__).resolve().parent / "vendors"

#: Module names under ``vendors/`` that are not platforms. A leading underscore means
#: private, which is what an in-progress vendor should be called until it is real.
_IGNORED = frozenset({"__init__"})


def modules() -> tuple[str, ...]:
    """List the importable vendor module names, sorted.

    Returns:
        Dotted module names such as ``"backends.vendors.iluvatar"``, one per vendor
        file or package, sorted so the scan is deterministic. Empty when the directory
        does not exist, which is a valid state for a build with no vendor backends.
    """
    if not VENDOR_DIR.is_dir():
        return ()
    names = []
    for entry in pkgutil.iter_modules([str(VENDOR_DIR)]):
        if entry.name in _IGNORED or entry.name.startswith("_"):
            continue
        names.append(f"backends.vendors.{entry.name}")
    return tuple(sorted(names))


def load(module_name: str) -> tuple[VendorSpec, SpecBackend]:
    """Import one vendor module and wrap its ``SPEC``.

    Args:
        module_name: Dotted module name from :func:`modules`.

    Returns:
        ``(spec, backend)``. The backend is constructed here, so a spec that fails
        :meth:`VendorSpec.validate` is reported at discovery rather than at the first
        request on the hardware nobody has.

    Raises:
        RuntimeError: The module has no module-level ``SPEC``, or importing it raised.
            Both are wrapped with the module name, because the traceback of a broken
            vendor file does not say which platform it was supposed to be.
        ValueError: The spec is invalid; propagated from :meth:`VendorSpec.validate`.
    """
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise RuntimeError(
            f"vendor backend {module_name!r} failed to import: "
            f"{type(exc).__name__}: {exc}") from exc
    spec = getattr(module, "SPEC", None)
    if not isinstance(spec, VendorSpec):
        raise RuntimeError(
            f"vendor backend {module_name!r} does not define a module-level SPEC: "
            f"VendorSpec; found {type(spec).__name__}")
    return spec, SpecBackend(spec)


def discover() -> tuple[tuple[VendorSpec, SpecBackend], ...]:
    """Import every vendor backend, in deterministic order.

    Returns:
        ``(spec, backend)`` per discovered vendor, sorted by module name.

    Raises:
        RuntimeError: A vendor module is broken; see :func:`load`. Called on the
            failure path of selection so a broken vendor file surfaces there, and
            kept out of the import path of ``backends`` so a container that only
            needs one backend does not depend on the others importing.
    """
    return tuple(load(name) for name in modules())


def iter_vendors() -> Iterator[tuple[VendorSpec, SpecBackend]]:
    """Yield the vendor backends lazily, for a caller that must tolerate one failure.

    Returns:
        An iterator over ``(spec, backend)``. Unlike :func:`discover` this does not
        raise: a broken vendor is skipped. Used by the registry's auto-detect, where
        one unusable platform must not prevent another from being selected.
    """
    for name in modules():
        try:
            yield load(name)
        except Exception:                         # pragma: no cover - broken vendor
            continue
