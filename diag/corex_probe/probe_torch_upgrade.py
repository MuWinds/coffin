"""What would it take to run a newer torch on this cluster?

The engine needs APIs that torch 2.1 lacks (``F.rms_norm`` is the first one hit),
so the cheapest fix would be a newer torch. Whether that is possible turns on
three things this probe measures and nothing else does:

1. **Driver version.** A vendor's user-space libraries are built against a driver
   ABI. If the installed ``ixdriver`` is older than the toolchain's libs, loading
   them fails with an undefined ``ixdrv*`` symbol -- which looks like a broken
   wheel and is really a driver mismatch. The driver is a kernel module: a normal
   user cannot change it, and on a shared node should not try.
2. **Which toolchain releases exist.** The vendor publishes versioned download
   roots; listing them shows whether a build exists that pairs a newer torch with
   the driver already installed (that would be a safe, user-space-only upgrade).
3. **Which interpreter each wheel targets.** A cp312 wheel cannot install into the
   py310 environment the working engine uses, so a torch upgrade can force a
   Python upgrade too.

Everything here is read-only: no installs, no driver changes, no writes outside
the job's own directory. That is deliberate -- the card is shared, and a driver
mishap takes the node down for everybody.

Module: ``coffin/diag``. Diagnostic only.

Constraints:
    * Never run ``pip install`` or touch ``/usr/local``; report and let a human
      decide.
    * Keep the whole probe under a minute of wall clock, so it can be re-run
      freely while deciding.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys


def run(cmd: list[str], timeout: float = 20.0) -> str:
    """Run a command and return its combined output, never raising.

    Args:
        cmd: argv to execute.
        timeout: Seconds before the call is abandoned.

    Returns:
        stdout and stderr merged, with a marker line when the command failed.
        Probes must not die because one command is missing on this node.
    """
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        body = (out.stdout or "") + (out.stderr or "")
        if out.returncode != 0:
            body += f"\n  [exit {out.returncode}]"
        return body.strip()
    except FileNotFoundError:
        return "  <not present>"
    except Exception as exc:  # noqa: BLE001 - a probe reports, it does not raise
        return f"  <failed: {type(exc).__name__}: {exc}>"


def sep(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)


def main() -> int:
    sep("1. driver: what the kernel module actually is")
    print("ixsmi header:")
    print(run(["bash", "-lc", "ixsmi 2>&1 | head -6"]))
    print("\nixdriver package:")
    print(run(["bash", "-lc",
               "rpm -qa 2>/dev/null | grep -i -E 'ix|iluvatar' | head -20"]))
    print("\nkernel module:")
    print(run(["bash", "-lc", "lsmod | grep -i -E '^ix|iluvatar' | head"]))
    print(run(["bash", "-lc", "modinfo ixdriver 2>&1 | head -12"]))
    print("\ndriver ABI symbols (does the loaded driver export ixdrvDeviceGetAttribute?):")
    print(run(["bash", "-lc",
               "for f in /usr/local/corex*/lib64/libcuda.so.1; do "
               "echo \"-- $f\"; strings $f 2>/dev/null | grep -c ixdrvDeviceGetAttribute; "
               "done"]))
    print("\nlibcuda actually linked by the working torch:")
    print(run(["bash", "-lc",
               "/data/software/Anaconda3/2024.10-1/envs/py310/bin/python -c \"import torch,os;"
               "print(torch.__version__)\" 2>&1 | tail -2"]))

    sep("2. toolchain releases installed on this node")
    print(run(["bash", "-lc", "ls -d /usr/local/corex* 2>/dev/null"]))
    for path in ("/usr/local/corex/release-corex.txt",
                 "/usr/local/corex-3.2.1/corex-3.2.1",
                 "/usr/local/corex-3.2.3/release-corex.txt"):
        print(f"\n-- {path}")
        print(run(["bash", "-lc", f"head -5 {path} 2>&1"]))

    sep("3. every python environment and its torch, if any")
    roots = ["/data/software/Anaconda3/2024.10-1/envs",
             os.path.expanduser("~/.conda/envs"),
             os.path.expanduser("~/venvs")]
    for root in roots:
        print(f"\n-- {root}")
        listing = run(["bash", "-lc", f"ls {root} 2>/dev/null"])
        print(listing)
        for name in listing.split():
            if name.startswith("["):
                continue
            site = (f"{root}/{name}/lib/python3.10/site-packages",
                    f"{root}/{name}/lib/python3.12/site-packages",
                    f"{root}/{name}/lib/python3.11/site-packages")
            for s in site:
                ver = run(["bash", "-lc", f"cat {s}/torch/version.py 2>/dev/null | head -4"])
                if ver and "not present" not in ver:
                    print(f"  {name} [{s.split('python')[1][:4]}]: "
                          f"{' '.join(ver.split())[:160]}")

    sep("4. wheels staged in the home directory")
    print(run(["bash", "-lc",
               "ls -la ~/whl ~/whl45 2>/dev/null | head -30"]))
    print("\nwheel python tags:")
    print(run(["bash", "-lc",
               "for w in ~/whl45/*.whl ~/whl/*.whl; do "
               "echo \"$(basename $w) -> $(unzip -p $w '*.dist-info/WHEEL' 2>/dev/null "
               "| grep -i tag | head -1)\"; done 2>/dev/null"]))

    sep("5. is a newer toolchain self-consistent? (the 4.5 user-space libs vs this driver)")
    corex45 = os.path.expanduser("~/corex45/img/usr/local/corex-4.5.0/lib64")
    print(f"-- {corex45} exists: {os.path.isdir(corex45)}")
    if os.path.isdir(corex45):
        print("libnccl needs from the driver:")
        print(run(["bash", "-lc",
                   f"ldd {corex45}/libnccl.so.2 2>&1 | grep -i 'not found' | head -5"]))
        print("undefined symbols in libnccl against the installed driver:")
        print(run(["bash", "-lc",
                   "cd ~/corex45/img/usr/local/corex-4.5.0/lib64 && "
                   "LD_LIBRARY_PATH=$PWD:/usr/local/corex/lib64 "
                   "python3 -c \"import ctypes;ctypes.CDLL('./libnccl.so.2')\" 2>&1 | tail -3"]))

    sep("6. can a newer torch even be imported here?")
    py312 = os.path.expanduser("~/.conda/envs/py312/bin/python")
    if os.path.exists(py312):
        print(run(["bash", "-lc", f"{py312} -V 2>&1"]))
        print(run(["bash", "-lc", f"{py312} -c 'import torch;print(torch.__version__)' 2>&1 | tail -3"]))
    else:
        print("  no py312 env")

    sep("summary")
    print(json.dumps({
        "note": "read-only survey; nothing installed or modified",
    }, indent=2))
    print("\nTORCH_UPGRADE_SURVEY_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
