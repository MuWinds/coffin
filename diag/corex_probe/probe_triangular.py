"""Which triangular-solve route works on this card?

``ops/delta.py`` needs ``A^-1 B`` for a unit-lower-triangular ``A`` once per chunk
per layer in prefill, and it calls ``torch.linalg.solve_triangular`` for it. On the
CoreX build that call dies inside ``dlopen``: the wheel's cuSOLVER is looked for at
a hardcoded ``/opt/sw_home/local/cuda/lib64/libcusolver.so`` which does not exist
on this cluster. So the question is not "tune the solve" but "which route still
works", and picking one needs measurement, not guesswork.

Candidates, cheapest first:

* ``torch.triangular_solve`` -- the pre-``linalg`` spelling; a different dispatch
  path, and it may be pure ATen on this build.
* ``torch.linalg.inv`` -- inverts the whole matrix; if it works, it is one call.
* ``torch.linalg.solve_ex`` / ``torch.linalg.solve`` -- LU-based, still solver.
* A blocked forward substitution written with ``bmm`` -- no solver at all, so it
  cannot dlopen anything. Costs ``L/block`` sequential matmuls; for the chunk
  lengths in use that is 4-8 steps, which is the fallback's price.
* Neumann doubling on ``(I + N)`` -- no solver, but ``log2(L)`` full-sized matmuls
  per step, so it is only worth it if the blocked form is too slow.

Module: ``coffin/diag``. Runs on the card.

Constraints:
    * Every candidate is verified against a host fp64 solve, because "it returned a
      tensor" is not "it is right": this card's fp64 is broken, so the reference is
      computed on the CPU.
    * Shapes are the ones the model actually uses: chunk length 512, 16 heads, and
      the two right-hand-side widths (``dv`` = 128 for v, ``dk`` = 128 for k).
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _sep(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)


def _bench(fn, iters=5, warmup=2):
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def blocked_forward_sub(lower, rhs, block=64):
    """Solve ``A X = B`` for unit-lower-triangular ``A`` without a solver.

    Blocked forward substitution: split the ``L`` rows into blocks of ``block``.
    For each block ``i``, ``X_i = B_i - A[i, :i] @ X[:i]``, then the diagonal block
    is unit lower triangular and small enough to invert by repeated squaring.

    Args:
        lower: ``[bn, L, L]`` unit-lower-triangular coefficient matrix.
        rhs: ``[bn, L, n]`` right-hand side.
        block: Rows per block. Must divide ``L``; 64 keeps the per-block inverse at
            512 KB and the sequential depth at ``L / block``.

    Returns:
        ``[bn, L, n]`` solution. No library call involved, so it cannot dlopen a
        missing solver.
    """
    import torch

    bn, L, n = rhs.shape
    nb = L // block
    out = torch.empty_like(rhs)
    for i in range(nb):
        lo = i * block
        hi = lo + block
        acc = rhs[:, lo:hi, :].clone()
        if i:
            acc = acc - torch.bmm(lower[:, lo:hi, :lo], out[:, :lo, :])
        # Diagonal block: unit lower triangular, inverted by squaring. With
        # ``D = I + N`` and N strictly lower, ``(I+N)^-1 = I - N + N^2 - ...`` and
        # the series terminates after ``block`` terms; Newton doubling reaches the
        # same exact point in log2(block) matmuls.
        d = lower[:, lo:hi, lo:hi]
        eye = torch.eye(block, dtype=d.dtype, device=d.device).expand(bn, block, block)
        inv = eye.clone()
        nrm = eye - d                     # I - D
        # X <- X (2I - D X), starting from D = I + N so the error is -N.
        for _ in range(max(1, (block - 1).bit_length())):
            inv = torch.bmm(inv, 2 * eye - torch.bmm(d, inv))
        out[:, lo:hi, :] = torch.bmm(inv, acc)
    return out


def main() -> int:
    import torch

    _sep("0. environment")
    print(f"  torch {torch.__version__}  cuda={torch.version.cuda}", flush=True)
    print(f"  cusolver present under corex: "
          f"{os.path.exists('/usr/local/corex/lib64/libcusolver.so')}", flush=True)
    print(f"  the path the wheel dlopens: "
          f"{os.path.exists('/opt/sw_home/local/cuda/lib64/libcusolver.so')}", flush=True)

    L = 512
    bn = 16          # heads; one chunk per head is the shape at prefill chunk 512
    n = 128          # dv (and dk) for this checkpoint
    torch.manual_seed(0)

    # A unit-lower-triangular coefficient matrix of the form the delta rule builds:
    # A = I + (strictly lower), and well conditioned because it comes from a decay
    # matrix with entries in (0, 1].
    strict = torch.rand(bn, L, L, dtype=torch.float32)
    strict = torch.tril(strict, diagonal=-1)
    a = (torch.eye(L) + 0.02 * strict).to("cuda")
    b = torch.randn(bn, L, n, dtype=torch.float32, device="cuda")

    # Host fp64 reference: the card's fp64 returns zeros, so this must be the CPU.
    ref = torch.linalg.solve_triangular(a.cpu().double(), b.cpu().double(),
                                        upper=False, unitriangular=True)

    def _rel(x):
        xh = x.detach().to("cpu").double()
        scale = float(ref.abs().max())
        return float((xh - ref).abs().max()) / scale if scale else float("inf")

    _sep("1. candidate routes (correctness first, timing second)")
    rows = []

    candidates = {
        "linalg.solve_triangular": lambda: torch.linalg.solve_triangular(
            a, b, upper=False, unitriangular=True),
        "triangular_solve": lambda: torch.triangular_solve(
            b, a, upper=False, unitriangular=True).solution,
        "linalg.solve (LU)": lambda: torch.linalg.solve(a, b),
        "linalg.solve_ex": lambda: torch.linalg.solve_ex(a, b)[0],
        "linalg.inv + bmm": lambda: torch.bmm(torch.linalg.inv(a), b),
        "blocked fwd-sub": lambda: blocked_forward_sub(a, b),
    }
    for name, fn in candidates.items():
        try:
            t0 = time.time()
            out = fn()
            torch.cuda.synchronize()
            rel = _rel(out)
            ms = _bench(fn)
            print(f"  {name:<26} rel={rel:.3e}  {ms:8.3f} ms  "
                  f"(first call {time.time() - t0:.1f}s)", flush=True)
            rows.append((name, rel, ms))
        except Exception as exc:                  # noqa: BLE001 - the answer is the list
            print(f"  {name:<26} FAILED: {type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:120]}", flush=True)

    _sep("2. blocked forward substitution at the block sizes worth trying")
    for block in (32, 64, 128, 256):
        if L % block:
            continue
        try:
            fn = lambda blk=block: blocked_forward_sub(a, b, blk)
            out = fn()
            torch.cuda.synchronize()
            rel = _rel(out)
            ms = _bench(fn, iters=3, warmup=1)
            print(f"  block={block:<4} rel={rel:.3e}  {ms:8.3f} ms", flush=True)
        except Exception as exc:                  # noqa: BLE001
            print(f"  block={block:<4} FAILED: {type(exc).__name__}: {exc}", flush=True)

    _sep("3. model-real shape: chunk 2048 as well")
    for L2 in (512, 1024, 2048):
        strict2 = torch.tril(torch.rand(4, L2, L2, dtype=torch.float32), diagonal=-1)
        a2 = (torch.eye(L2) + 0.02 * strict2).to("cuda")
        b2 = torch.randn(4, L2, n, dtype=torch.float32, device="cuda")
        ref2 = torch.linalg.solve_triangular(a2.cpu().double(), b2.cpu().double(),
                                             upper=False, unitriangular=True)
        try:
            out = blocked_forward_sub(a2, b2)
            torch.cuda.synchronize()
            scale = float(ref2.abs().max())
            rel = float((out.detach().to("cpu").double() - ref2).abs().max()) / scale
            ms = _bench(lambda: blocked_forward_sub(a2, b2), iters=3, warmup=1)
            print(f"  L={L2:<5} blocked rel={rel:.3e}  {ms:8.3f} ms", flush=True)
        except Exception as exc:                  # noqa: BLE001
            print(f"  L={L2:<5} blocked FAILED: {type(exc).__name__}: {exc}", flush=True)
        del a2, b2, ref2

    _sep("summary")
    print("  working routes:", [name for name, rel, _ in rows if rel < 1e-4], flush=True)
    print("\nCOREX_TRIANGULAR_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
