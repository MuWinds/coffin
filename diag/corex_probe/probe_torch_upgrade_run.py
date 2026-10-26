"""Decisive test: can a newer torch run on this node's driver, in user space?

The survey established the pieces and the doubt. ``torch 2.10.0+corex.4.5.0``
exists as a cp312 wheel; the loaded kernel module reports driver 3.2.1; and the
4.5 user-space ``libnccl`` fails to load against that driver with
``undefined symbol: ixdrvDeviceGetAttribute``. That last failure is the whole
question: if it is only about which directory is on ``LD_LIBRARY_PATH``, the
upgrade is a user-space matter; if the *loaded driver* genuinely lacks the symbol,
no amount of library paths helps and the driver (a kernel module, so an
administrator change on a shared node) has to move first.

This probe answers that by trying to import the staged 4.5 stack under several
library paths and reporting which one gets furthest. Nothing is installed: the
wheel is unpacked into the user's own home, and every attempt runs in a
subprocess so a crash in one configuration cannot poison the next.

Module: ``coffin/diag``. Diagnostic only.

Constraints:
    * Never modify ``/usr/local``, ``/etc`` or a loaded module. Everything stays
      inside ``$HOME``; a driver change on a shared GPU node takes the machine out
      for every user and is not something a job may attempt.
    * Each attempt is one subprocess with an explicit ``LD_LIBRARY_PATH``, because
      the dynamic loader reads it once at process start: retrying inside one
      interpreter would silently test the first environment again.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

HOME = os.path.expanduser("~")

#: Candidate library roots. ``~/corex45/img/usr/local/corex-4.5.0`` is an already
#: unpacked 4.5 toolchain from an earlier attempt; it is the only 4.5 user space on
#: the machine, so it is what a user-space-only upgrade would run on.
CANDIDATE_LIBS = {
    "host 3.2.x": "/usr/local/corex/lib64",
    "staged 4.5": os.path.join(HOME, "corex45/img/usr/local/corex-4.5.0/lib64"),
    "4.5 + host 3.2.x": (
        os.path.join(HOME, "corex45/img/usr/local/corex-4.5.0/lib64")
        + ":/usr/local/corex/lib64"),
}

#: Interpreters to try. py312 is the only one whose ABI matches the cp312 wheel.
CANDIDATE_PYTHONS = {
    "py310 (working engine)": "/data/software/Anaconda3/2024.10-1/envs/py310/bin/python",
    "py312 (staged wheel)": os.path.join(HOME, ".conda/envs/py312/bin/python"),
}

#: The one-line question. Importing torch is enough: the CoreX build links its
#: runtime at import, so an ABI mismatch surfaces there rather than at first use.
PROBE = textwrap.dedent("""
    import sys
    try:
        import torch
    except Exception as exc:
        print("IMPORT_FAIL", type(exc).__name__, str(exc).splitlines()[0][:180])
        sys.exit(0)
    print("TORCH", torch.__version__, "cuda", torch.version.cuda)
    try:
        print("AVAILABLE", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("DEVICE", torch.cuda.get_device_name(0))
            x = torch.randn(64, 64, dtype=torch.bfloat16, device="cuda")
            y = x @ x
            torch.cuda.synchronize()
            print("MATMUL_OK", bool(torch.isfinite(y).all()))
    except Exception as exc:
        print("DEVICE_FAIL", type(exc).__name__, str(exc).splitlines()[0][:180])
""")


def attempt(python: str, libs: str) -> str:
    """Import torch in a fresh process with the given library path.

    Args:
        python: Interpreter to run.
        libs: Value for ``LD_LIBRARY_PATH``.

    Returns:
        The subprocess output, or a one-line failure description.
    """
    if not os.path.exists(python):
        return f"<no interpreter at {python}>"
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = libs
    # A stale extension cache from another toolchain would load the wrong .so.
    env["TORCH_EXTENSIONS_DIR"] = os.path.join(HOME, "torch_ext_upgrade_check")
    try:
        out = subprocess.run([python, "-c", PROBE], capture_output=True, text=True,
                             timeout=180, env=env)
    except subprocess.TimeoutExpired:
        return "<timed out after 180s>"
    except Exception as exc:  # noqa: BLE001
        return f"<failed: {type(exc).__name__}: {exc}>"
    return ((out.stdout or "") + (out.stderr or "")).strip()


def main() -> int:
    print("=" * 78)
    print("driver: what the kernel module reports, and what symbols it exports")
    print("=" * 78)
    for cmd in (["bash", "-lc", "ixsmi 2>/dev/null | head -3"],
                ["bash", "-lc",
                 "nm -D --defined-only /usr/local/corex/lib64/libcuda.so.1 "
                 "2>/dev/null | grep -c ixdrv"],
                ["bash", "-lc", "lsmod | grep -E '^iluvatar'"]):
        out = subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
        print(out or "  <none>")

    print()
    print("=" * 78)
    print("attempt matrix: interpreter x library path")
    print("=" * 78)
    results = {}
    for py_name, python in CANDIDATE_PYTHONS.items():
        for lib_name, libs in CANDIDATE_LIBS.items():
            key = f"{py_name} + {lib_name}"
            body = attempt(python, libs)
            results[key] = body
            verdict = "FAIL"
            if "TORCH" in body:
                verdict = "IMPORTS"
                if "MATMUL_OK True" in body:
                    verdict = "WORKS"
                elif "AVAILABLE True" in body:
                    verdict = "IMPORTS, NO USABLE DEVICE"
            print(f"\n--- {key}: {verdict}")
            for line in body.splitlines()[:6]:
                print(f"    {line}")

    print()
    print("=" * 78)
    print("summary")
    print("=" * 78)
    for key, body in results.items():
        first = next((l for l in body.splitlines()
                      if l.startswith(("TORCH", "IMPORT_FAIL"))), "<no output>")
        print(f"  {key:<44} {first[:120]}")
    print(json.dumps({"note": "nothing installed; every attempt was a subprocess"},
                     indent=2))
    print("\nTORCH_UPGRADE_TEST_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
