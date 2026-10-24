#!/usr/bin/env python3
"""Push a file through the *non-daemon* E-Shell path.

``tools/push.py`` goes through the long-lived daemon, which serialises every
command: while another agent holds the daemon with a long command, pushes queue.
This helper opens a fresh connection per call instead and puts the whole file in
one base64 command (the daemon-stall workaround for this session).

Usage: python coffin/diag/push_nodaemon.py <local> <remote>
"""
from __future__ import annotations

import base64
import pathlib
import subprocess
import sys

PLATFORM = pathlib.Path(__file__).resolve().parent.parent.parent / "platform"
GROUP = "b7dfbc1d-a37c-4926-af69-52adb535c3ad"


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("usage: push_nodaemon.py <local> <remote>")
    local, remote = sys.argv[1], sys.argv[2]
    b64 = base64.b64encode(pathlib.Path(local).read_bytes()).decode()
    if len(b64) > 28000:
        sys.exit(f"too large for one command: {len(b64)} b64 chars")
    cmd = f"mkdir -p \"$(dirname '{remote}')\" && printf %s '{b64}' | base64 -d > '{remote}' && wc -c '{remote}'"
    rc = subprocess.run(["node", "shell.mjs", "--group", GROUP, "--no-daemon",
                         "--exec", cmd], cwd=str(PLATFORM)).returncode
    sys.exit(rc)


if __name__ == "__main__":
    main()
