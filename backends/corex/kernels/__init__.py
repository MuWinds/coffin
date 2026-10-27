"""Build, or attach to a cached build of, the CoreX kernel extension.

The extension is a torch JIT extension built from ``backends/corex/kernels/*.cu``
with ``torch.utils.cpp_extension.load``. Nothing is compiled at import: this module
is imported by ``backends/corex/__init__.py`` on every host that merely enumerates
backend names, including a CPU-only one with no CoreX toolchain, so the build is
lazy and its failure is memoised.

Module: ``coffin/backends/corex``. Imports ``torch``, the standard library and the
extension loader -- and nothing from ``coffin`` at all. The kernels it builds are
C++/CUDA sources next to it; the op table above it calls :func:`load_kernels`.

Constraints:
    * ``TORCH_EXTENSIONS_DIR`` must come from the environment, never from a literal
      here. Every deployment has to point at a per-deployment cache: a shared one
      makes concurrent jobs contend on a single build baton and hang, and a cache
      inside the source tree makes a checkout unwritable on the cluster.
    * The build target is ``--cuda-gpu-arch=ivcore10`` (this part's ISA, read from
      the driver as ``Caps.arch``). A different part needs a different arch string
      passed in, not a second build path.
    * The failure memo is the whole point of :func:`load_kernels`: a broken toolchain
      costs one attempt per process, then costs nothing, and the op table falls back
      to the reference backend instead of the process dying mid-request.
    * Importing this module must not import ``torch.utils.cpp_extension`` at module
      scope. The loader reads environment variables at import time, and a host that
      never builds must not be reconfigured by an enumeration.
    * The extension is compiled per arch, so the build directory is keyed by the arch
      string: an ``ivcore10`` object file must never be attached to a run targeting a
      different ISA.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

#: Directory holding the ``.cu``/``.cpp`` sources this module builds.
KERNEL_DIR = Path(__file__).resolve().parent

#: Translation units of the extension, in link order. Listing them here rather than
#: globbing keeps a stray ``.cu`` (a probe, an experiment) out of the production build
#: and keeps the list reviewable.
_SOURCES = ("bindings.cpp", "gemv.cu", "norm.cu", "attn.cu", "linear_attn.cu",
            "delta.cu", "solve.cu")

#: ISA fallback when the driver does not report one. ``ivcore10`` is the BI-V100's arch
#: (``COREX_PORT.md`` §0); a card that reports something else gets that value instead,
#: which is why the loader takes ``arch`` as an argument.
DEFAULT_ARCH = "ivcore10"

#: Extension name. The suffix is the build-scheme version: bump it when a source change
#: must not attach to an already-cached object tree. It is bumped with every kernel batch
#: precisely because a stale cache is invisible -- the old objects load and answer.
_EXTENSION = "coffin_corex_ops_v7"

#: Loaded extension, or ``None`` before the first attempt.
_module: Any = None

#: ``"<ExceptionType>: <message>"`` from a failed build, or ``None``. Memoised so a
#: broken toolchain costs one attempt per process.
_error: str | None = None


def _flags(arch: str) -> dict[str, list[str]]:
    """Return the compiler flags for one target ISA.

    Args:
        arch: Target ISA, i.e. ``Caps.arch`` (``"ivcore10"`` on a BI-V100).

    Returns:
        ``{"cflags": [...], "cuda_cflags": [...]}``. ``-O3`` on both sides because
        the GEMV is a bare load/FMA loop with nothing to hide behind, and the
        arch flag has to reach the device compiler or the build silently targets
        whatever the driver defaults to.
    """
    return {
        "cflags": ["-O3"],
        "cuda_cflags": ["-O3", f"--cuda-gpu-arch={arch}"],
    }


def _clear_stale_build_lock(cache: str) -> None:
    """Remove a build lock left behind by a killed process.

    ``torch.utils.cpp_extension`` serialises builds with a baton file at
    ``<cache>/<name>/lock``. Its ``wait()`` polls until the file disappears, and nothing
    ages it out: a job killed between creating that file and removing it leaves every later
    job on the cluster blocked forever, with no output on either stream, because the wait
    happens inside an import.

    This was not theoretical. Cancelling three submits whose imports were still resolving
    the backend wedged the shared cache for every subsequent run, and the failure looked
    exactly like a hang in device code -- the stack dump was the only thing that named it.

    Only a *stale* lock is removed. A fresh one belongs to a build that is genuinely
    running, and deleting that would let two builds write the same object tree.

    Args:
        cache: Value of ``TORCH_EXTENSIONS_DIR``.

    Returns:
        Nothing. Recovery is best effort: an unremovable lock is left to ``load``.
    """
    lock = Path(cache) / _EXTENSION / "lock"
    try:
        age = time.time() - lock.stat().st_mtime
    except OSError:
        return
    # Ten minutes by default: longer than any successful build of these seven units, far
    # shorter than the wall clock a blocked job would otherwise burn. The threshold is
    # overridable because recovering from a wedge that is only seconds old needs a smaller
    # value, and waiting ten minutes to clear a known-bad lock is its own bug.
    limit = float(os.environ.get("COFFIN_BUILD_LOCK_MAX_AGE", "600"))
    if age < limit:
        return
    try:
        lock.unlink()
    except OSError:
        return


def load_kernels(arch: str = DEFAULT_ARCH) -> Any:
    """Build or attach the extension, once per process.

    Args:
        arch: Target ISA from the driver. Only the first call's value is used: the
            loaded module is memoised, so a process cannot hold two builds and a
            second arch would silently reuse the first one's objects.

    Returns:
        The extension module.

    Raises:
        RuntimeError: The build failed or ``TORCH_EXTENSIONS_DIR`` is unset. Raised
            rather than returning ``None`` so the caller has one place to catch and
            one message to log; ``backends/corex/__init__.py`` turns it into the
            backend's ``reason()`` and the reference fallback.
    """
    global _module, _error
    if _module is not None:
        return _module
    if _error is not None:
        raise RuntimeError(_error)
    cache = os.environ.get("TORCH_EXTENSIONS_DIR", "").strip()
    if not cache:
        # A default here would put build artifacts in a shared or unwritable place
        # and the two failure modes look identical from the outside.
        _error = ("TORCH_EXTENSIONS_DIR is not set; the CoreX kernels need a "
                  "per-deployment extension cache (a shared one serialises every "
                  "deployment behind one build lock)")
        raise RuntimeError(_error)
    try:
        from torch.utils.cpp_extension import load

        _clear_stale_build_lock(cache)
        flags = _flags(arch or DEFAULT_ARCH)
        _module = load(
            name=_EXTENSION,
            sources=[str(KERNEL_DIR / name) for name in _SOURCES],
            extra_include_paths=[str(KERNEL_DIR)],
            extra_cflags=flags["cflags"],
            extra_cuda_cflags=flags["cuda_cflags"],
            verbose=False,
        )
    except Exception as exc:                      # pragma: no cover - needs a toolchain
        _error = f"{type(exc).__name__}: {exc}"
        raise RuntimeError(_error) from exc
    return _module


def available() -> bool:
    """Report whether the extension is loaded or loadable, without raising.

    Returns:
        ``True`` when :func:`load_kernels` has succeeded or can succeed now. On a
        host with no toolchain this attempts the build once and then answers
        ``False`` from the memo, which is what makes
        ``backends/corex.usable()`` cheap to call repeatedly.
    """
    if _module is not None:
        return True
    try:
        load_kernels()
    except Exception:                             # noqa: BLE001 - degradation path
        return False
    return True


def last_error() -> str | None:
    """Return the memoised build failure, for the backend's ``reason()``.

    Returns:
        ``"<ExceptionType>: <message>"`` from the failed build, or ``None`` when the
        build succeeded or was never attempted. Without this the startup log would
        show a reference fallback and no cause.
    """
    return _error


def errors() -> list[str]:
    """Return the build failure as a one-element list, or an empty list.

    Returns:
        ``[last_error()]`` when a build failed. It exists so a diagnostic can print
        ``"\n".join(...)`` without a ``None`` check, which is the shape the diag
        scripts in ``backends/corex/diag`` use.
    """
    err = last_error()
    return [] if err is None else [err]


def reset() -> None:
    """Forget the memoised module and error, so the next call rebuilds.

    Returns:
        Nothing. Exists for a diagnostic that has just created
        ``TORCH_EXTENSIONS_DIR`` or fixed the toolchain, and for tests that need the
        "never attempted" state back.
    """
    global _module, _error
    _module = None
    _error = None
