"""Where exactly does ``import backends`` block? Dump the stack, do not guess.

``import backends`` resolves the backend selection, builds the op table and reads the
capability record, so it is the one import whose cost is not obvious from outside. Several
runs stopped between the marker printed just before it and the one just after, with nothing
on stderr: no build output, no traceback, no error. That is either a slow file lock or a
blocked device query, and the two need different answers.

``faulthandler.dump_traceback_later`` is the tool for it: after a timeout it writes every
thread's stack to stderr, which the launcher merges into stdout, so the blocking frame is
named instead of inferred.

Module: ``coffin/diag``. Runs without needing a model.

Constraints:
    * The dump goes to stderr; run it with stderr merged into stdout or the output is lost.
    * The timeout is short. The point is to catch a block, not to wait for completion.
"""

from __future__ import annotations

import faulthandler
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> int:
    """Import ``backends`` under a stack-dump watchdog.

    Returns:
        ``0`` when the import completes. A block never returns; the watchdog writes the
        stacks that say where it stopped.
    """
    print(f"  cwd={os.getcwd()}", flush=True)
    print(f"  sys.path[0]={sys.path[0]}", flush=True)
    print(f"  COFFIN_BACKEND={os.environ.get('COFFIN_BACKEND')}", flush=True)
    print(f"  TORCH_EXTENSIONS_DIR={os.environ.get('TORCH_EXTENSIONS_DIR')}", flush=True)

    print("  [1] importing torch", flush=True)
    t0 = time.time()
    import torch

    print(f"  [2] torch ready in {time.time() - t0:.1f}s", flush=True)

    # 20 s is far longer than a cached attach and far shorter than a rebuild, so a dump here
    # means a block rather than a build.
    faulthandler.dump_traceback_later(20, repeat=True, file=sys.stderr)
    print("  [3] importing backends (watchdog armed at 20 s)", flush=True)
    t0 = time.time()
    import backends

    faulthandler.cancel_dump_traceback_later()
    print(f"  [4] backends ready in {time.time() - t0:.1f}s", flush=True)
    print(f"  status: {backends.status()}", flush=True)
    print(f"  caps  : graph={int(backends.caps().graph)} "
          f"device={backends.caps().device}", flush=True)
    print("IMPORT_CHECK_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
