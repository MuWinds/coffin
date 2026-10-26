"""Does the engine import and run on this cluster's torch at all?

The cluster's torch is 2.1.0 with a CoreX runtime, while the engine was written
against 2.7 on ROCm. Several APIs the engine uses today (``F.rms_norm``,
``torch.compile`` helpers, the flex-attention namespace) do not exist in 2.1, and
a missing symbol only surfaces when the importing module is reached -- which for
this codebase is deep inside ``model.py``, after a 9 GB weight load.

So this probe imports every module in the tree, in dependency order, and reports
each failure separately instead of stopping at the first. Modules that need a
device are imported with ``COFFIN_BACKEND=reference`` so the import itself does
not require kernels; the device path is exercised afterwards by a tiny reference
forward.

Module: ``coffin/diag``. Diagnostic only: no weights, no checkpoint.

Constraints:
    * Import failures are collected, not raised: the point is the list, and a
      first-failure abort hides the other nine.
    * Runs on the CPU by design, so it can be used on the login node as soon as
      that node's torch imports.
"""

from __future__ import annotations

import importlib
import os
import pkgutil
import sys
import traceback

#: Modules in dependency order. Anything importable is checked; the ordering only
#: decides which error a reader sees first.
ORDER = (
    "backends.protocol", "backends.reference", "backends.vendor",
    "backends.discovery", "backends.registry", "backends",
    "backends.hip", "backends.corex",
    "ops.dtypes", "ops.gemv", "ops.embed", "ops.norm", "ops.rope", "ops.mrope",
    "ops.delta", "ops",
    "int8_weights",
    "models.qwen35.tuning", "models.qwen35.weights", "models.qwen35.batch",
    "models.qwen35.vision", "models.qwen35.linear_attn", "models.qwen35.full_attn",
    "models.qwen35.blocks", "models.qwen35.mtp", "models.qwen35.model",
    "models.qwen35", "models",
    "cache.paged_cache", "cache.radix_cache", "cache",
    "engine.config", "engine.sequence", "engine.block_manager",
    "engine.prefix_engine", "engine.model_runner", "engine.scheduler",
    "engine.llm_engine", "engine",
    "utils.mm_input", "utils.tool_call", "utils",
    "serving.runtime_support", "serving.runtime_transport", "serving.runtime_parts",
    "serving.runtime_service", "serving",
    "entrypoints.openai", "entrypoints.cli.runtime", "entrypoints.cli.openai",
    "engine_api",
    "agent",
)


def _package_root() -> str:
    """Locate the directory that must sit on ``sys.path`` for ``import ops`` to work.

    The engine is imported as a set of *top-level* packages (``ops``, ``models``,
    ``backends``), so the sys.path root is ``coffin/`` itself -- three levels above
    this file at ``coffin/diag/corex_probe/probe_import.py``. Getting this wrong is
    silent: every import fails with ``No module named 'ops'``, which reads like a
    missing package rather than a wrong root.

    Returns:
        Absolute path of the ``coffin`` directory.
    """
    here = os.path.abspath(__file__)
    return os.path.dirname(os.path.dirname(os.path.dirname(here)))


def _discovered() -> list[str]:
    """Every importable module under the package root, for the "did I miss one" case.

    Returns:
        Sorted dotted module names other than the ones in :data:`ORDER`, so a
        module added since this list was written still gets checked. ``diag`` and
        ``tests`` are skipped: they are probe scripts and test modules, and several
        of them build an extension or read a checkpoint at import time.
    """
    root = _package_root()
    skip = {"diag", "tests"}
    found = []
    for info in pkgutil.walk_packages([root]):
        if info.name.split(".")[0] in skip:
            continue
        found.append(info.name)
    return sorted(found)


def main() -> int:
    sys.path.insert(0, _package_root())
    os.environ.setdefault("COFFIN_BACKEND", "reference")

    import torch

    print(f"torch {torch.__version__}  cuda={torch.version.cuda}  "
          f"hip={getattr(torch.version, 'hip', None)}  "
          f"available={torch.cuda.is_available()}", flush=True)
    print(f"COFFIN_BACKEND={os.environ['COFFIN_BACKEND']}", flush=True)

    # APIs the engine is known to use today that torch 2.1 does not have.
    print("\n--- API presence ---", flush=True)
    import torch.nn.functional as F
    checks = {
        "F.rms_norm": hasattr(F, "rms_norm"),
        "torch.compile": hasattr(torch, "compile"),
        "torch.library.custom_op": hasattr(getattr(torch, "library", None), "custom_op"),
        "F.scaled_dot_product_attention": hasattr(F, "scaled_dot_product_attention"),
        "torch.nn.attention.flex_attention": False,
        "torch.cuda.CUDAGraph": hasattr(torch.cuda, "CUDAGraph"),
        "torch.cuda.graph_pool_handle": hasattr(torch.cuda, "graph_pool_handle"),
        "torch.cuda.Event": hasattr(torch.cuda, "Event"),
        "torch.nn.attention": hasattr(torch.nn, "attention"),
    }
    try:
        importlib.import_module("torch.nn.attention.flex_attention")
        checks["torch.nn.attention.flex_attention"] = True
    except Exception:  # noqa: BLE001 - absence is the answer
        pass
    for name, present in checks.items():
        print(f"  {'OK ' if present else 'MISSING'} {name}", flush=True)

    names = list(ORDER)
    known = set(names)
    names += [m for m in _discovered() if m not in known]

    print(f"\n--- importing {len(names)} modules ---", flush=True)
    failures = []
    for name in names:
        if not name:
            continue
        try:
            importlib.import_module(name)
            print(f"  ok    {name}", flush=True)
        except Exception as exc:  # noqa: BLE001 - collected, not raised
            failures.append((name, exc))
            tb = traceback.format_exc().strip().splitlines()
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}", flush=True)
            for line in tb[-6:]:
                print(f"        {line}", flush=True)

    print("\n--- reference-backend op smoke (CPU tensors) ---", flush=True)
    ops_ok = []
    try:
        import backends

        print("  " + backends.status(), flush=True)
        table = backends.ops
        missing = [n for n in table.__class__.NAMES if getattr(table, n) is None]
        print(f"  table total: {len(table.__class__.NAMES) - len(missing)}/"
              f"{len(table.__class__.NAMES)} callable", flush=True)

        w = torch.randn(64, 32, dtype=torch.float32)
        x = torch.randn(32, dtype=torch.float32)
        y = table.linear(w, x)
        print(f"  linear      -> {tuple(y.shape)} {y.dtype}", flush=True)
        ops_ok.append("linear")

        z = table.rms_norm(torch.randn(4, 32), torch.randn(32), 0, 1e-6)
        print(f"  rms_norm    -> {tuple(z.shape)}", flush=True)
        ops_ok.append("rms_norm")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()

    print("\n--- summary ---", flush=True)
    print(f"import failures: {len(failures)}")
    for name, exc in failures:
        print(f"  {name}: {type(exc).__name__}: {exc}")
    print(f"ops smoke: {ops_ok}")
    print("\nCOREX_IMPORT_DONE")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
