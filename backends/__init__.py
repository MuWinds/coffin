"""The platform layer: the one place in coffin that knows what hardware it is running on.

Model code binds this package's ``ops`` table at import and calls it like a module
of device primitives; it never learns which platform answered. That is the whole
point of the layer: ``hip_ops``, the ``"cuda"`` device literal, ``ops/dtypes.py``'s
``D``/``WF`` and ``models/qwen35/tuning.py``'s ``ATTN_SPLIT`` are all platform
facts, and they move here so the model family can stop carrying them.

Module: ``coffin/backends``. Bottom of the dependency graph with
``backends.protocol``: ``ops/``, ``models/``, ``engine/``, ``serving/`` and
``cache/`` import from here; this package imports none of them. Importing it must
not build a kernel, allocate a device tensor or require an accelerator -- on a
CPU-only host it resolves to the pure-torch reference backend, which is how the
test suite and ``COFFIN_BACKEND=reference`` smoke runs work.

Constraints:
    * The selection is made once, at import of the ``ops`` property's first use,
      and never changes: there is no runtime hot swap (PLATFORMS.md §8). Code that
      captures a backend reference keeps it for the process's life.
    * ``ops`` is a table, not a module: read it once into a module-level name
      (``ops = backends.ops``) and call the entries. Every entry is callable --
      the reference fills whatever the selected backend lacks -- so no call site
      may test an entry for ``None`` in the normal path.
    * ``device``/``act_dtype``/``weight_dtype`` come from ``caps`` and replace the
      literals model code writes today. Import them, do not re-derive them.
"""

from __future__ import annotations

from backends import discovery, protocol, reference, registry, vendor
from backends.protocol import Backend, Caps, Ops, make_device
from backends.registry import (
    ENV_VAR,
    available_names,
    backend,
    caps,
    candidates,
    describe_all,
    select,
    status,
)
from backends.vendor import Detection, SpecBackend, VendorSpec

#: The selected backend's total op table, resolved now so a call site can bind it
#: at import (``from backends import ops``) without paying a lazy-lookup branch on
#: the decode path. Every entry is callable; a name the backend has no kernel for
#: is the reference implementation.
ops: Ops = backend().ops()

#: The ``torch.device`` every allocation for the selected backend belongs on. This
#: is the value that replaces the 173 ``device="cuda"`` literals in ``models/``.
device = backend().device


def reload() -> None:
    """Re-resolve the backend selection and refresh the module-level ``ops``.

    Called by ``engine_api.reload``'s ordered reload, which rebuilds the engine
    modules in place. The selection itself (``COFFIN_BACKEND``) is unchanged by a
    reload: this exists so a changed backend table or capability record is picked
    up, not so a process can switch platform.

    Returns:
        Nothing. ``ops`` and ``device`` are rebound in place; a caller that bound
        either into a local name keeps the old value, which is why the reload
        order puts this package before everything that binds it.
    """
    global ops, device
    registry._reset_for_tests(None)          # noqa: SLF001 - package-internal hook
    ops = backend().ops()
    device = backend().device


#: Names this package exports to the rest of the engine.
__all__ = [
    "Backend", "Caps", "Detection", "ENV_VAR", "Ops", "SpecBackend", "VendorSpec",
    "available_names", "backend", "caps", "candidates", "describe_all", "device",
    "discovery", "make_device", "ops", "protocol", "reference", "registry",
    "reload", "select", "status", "vendor",
]
