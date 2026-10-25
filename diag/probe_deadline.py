"""Does the engine's wall-clock deadline actually cut a runaway generation?

The point of the deadline is the case no socket-level cancellation covers: a client
that goes quiet without closing the connection.  Nothing here talks to a client at
all -- the request runs on the engine socket with a 8 s cap and a 100k-token budget,
so "did it stop itself" is the only thing being measured.

    cd /root/private_data/engine && MKL_THREADING_LAYER=GNU python3 diag/probe_deadline.py

Module: ``coffin/diag``. Drives a real engine over the serve socket, so it needs
a running server at ``$ENGINE_SOCK`` (see ``serve.py``).

Constraints:
    * The budget is deliberately absurd (8 s, 100k tokens) so the only possible
      outcome is a self-inflicted stop; a generous budget would pass by accident.
    * Talks to the engine socket rather than in-process on purpose -- that is the
      path a silent client actually takes.
"""

from __future__ import annotations

import json
import socket
import sys
import time

SOCK = "/tmp/qwen35.sock"
CAP = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0


def call(req: dict, timeout: float = 300.0) -> dict:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect(SOCK)
    f = s.makefile("rb")
    try:
        s.sendall((json.dumps(req, ensure_ascii=False) + "\n").encode())
        return json.loads(f.readline().decode())
    finally:
        f.close()
        s.close()


def main() -> None:
    req = {"op": "generate", "prompt": "数到一千：1、2、3、", "max_new": 100000,
           "max_seconds": CAP, "temperature": 0.0}
    t0 = time.time()
    r = call(req)
    dt = time.time() - t0
    print(f"max_seconds={CAP}  wall={dt:.1f}s  n_new={r.get('n_new')}  "
          f"budget={r.get('max_new')}  stopped={r.get('stopped')!r}")
    ok = r.get("stopped") == "deadline" and CAP - 2 <= dt <= CAP + 20
    print("PASS: the cap fired and said so" if ok
          else "FAIL: the generation was not cut by the deadline")
    print(f"decode_tok_s={r.get('decode_tok_s')}")


if __name__ == "__main__":
    main()
