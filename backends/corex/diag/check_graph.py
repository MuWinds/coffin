"""Capture and replay ``linear`` and ``rms_norm`` in isolation.

**This script does NOT show that the whole decode step is capturable.** Read the next
paragraph before treating a green run as permission to set ``_GRAPH_VERIFIED = True``.

A per-op capture exercises neither the 32-layer call graph, nor the paged-attention
window, nor the int8 KV pool, nor the in-place recurrent-state updates. Those are exactly
where a capture breaks. This was already misread once: a green run here was taken as
evidence that the decode step could be captured, ``_GRAPH_VERIFIED`` was set to ``True``,
and the first real ``capture()`` + replay of a decode step **hung the device** until the
scheduler killed the job at its wall-clock limit -- no traceback, no ``.err`` file. A hang
is worse than a crash: it holds the card until someone intervenes.

So the scope of this file is deliberately narrow, and the narrow scope is the point:

* It answers "does this one kernel replay correctly on this card" -- useful when a single
  op is suspect.
* It does **not** answer "can the decode step be captured", and no combination of the two
  per-op checks below implies it.

The bar for ``caps.graph`` is a whole decode step: the same prompt decoded eagerly and
through a replay must produce identical tokens, and the process must survive. That check
does not live here, because building it inside ``backends/`` would need a checkpoint
import and ``backends/`` may not import an engine layer. ``_GRAPH_VERIFIED`` stays
``False`` until that measurement exists and is recorded next to the constant.

The checks themselves are built carefully, because "the captured call returned a finite
tensor" proves nothing:

* the tensor the backend allocates *inside* the capture is the one a replay writes. The
  value captured before ``torch.cuda.graph`` is a different allocation and stays frozen,
  so holding the wrong reference would report a stale buffer as a replay failure -- or,
  worse, a stale buffer as a pass;
* a replay that re-read a frozen copy of the input is still finite, so the check changes
  the input, replays, and compares the replayed output against a reference computed *from
  the new input*.

Module: ``coffin/backends/corex/diag``. Diagnostic only; runs on the card.

Constraints:
    * No fp64 on the device and no library GEMM in any reference: the card returns zeros
      for the first and NaN above ``N = 131072`` for the second.
    * The reference for ``linear`` is a host fp64 dot product, and for ``rms_norm`` a host
      formula. The reference backend cannot be the oracle for the norms here: its
      ``rms_norm`` calls ``F.rms_norm``, which does not exist in this cluster's torch 2.1.
    * No dependency on an engine layer, by the R5 gate: this file must stay importable
      with no checkpoint present.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: ``coffin/`` is the ``sys.path`` root: the engine imports ``backends``, ``ops``
#: and ``models`` as top-level packages. Running this file by path puts its own
#: directory on ``sys.path`` instead, so the root is added explicitly. Three levels
#: up from ``backends/corex/diag/check_graph.py``.
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _host(t):
    """Move a tensor to the host before widening it.

    Args:
        t: Any tensor.

    Returns:
        A float64 CPU copy. The widening happens after the transfer because the
        device's fp64 path returns zeros.
    """
    return t.detach().to("cpu").double()


def _rel(y, ref) -> float:
    """Relative error against a host reference.

    Args:
        y: Kernel output, on any device.
        ref: Host float64 reference of the same shape.

    Returns:
        ``max|y - ref| / max|ref|``, or ``inf`` when the reference is all zeros.
    """
    scale = float(ref.abs().max())
    err = float((_host(y) - ref).abs().max())
    return (err / scale) if scale > 0 else float("inf")


def _sep(title: str) -> None:
    """Print a section header.

    Args:
        title: Section name.
    """
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)


def _stage(name: str) -> None:
    """Announce a step before running it.

    A device-side hang or fault can end the process with no traceback, so the log is the
    only record of how far it got. Printing and flushing *before* the work is what turns
    "no output after this line" into a location.

    Args:
        name: Stage label.
    """
    print(f"  [stage] {name}", flush=True)


def _capture(torch, call):
    """Capture ``call`` into a fresh graph and return ``(graph, captured_output)``.

    Args:
        torch: The torch module.
        call: Zero-argument callable that runs the op on fixed input buffers.

    Returns:
        The graph and the tensor the call allocated inside the capture, which is the
        buffer a replay writes into.

    Raises:
        RuntimeError: Capture failed. Propagated with the op name added by the caller.
    """
    graph = torch.cuda.CUDAGraph()
    held = []
    with torch.cuda.graph(graph, pool=torch.cuda.graph_pool_handle()):
        held.append(call())
    return graph, held[0]


def main() -> int:
    """Capture, replay and verify two ops in isolation.

    Returns:
        ``0`` when both replays re-read their input and matched the reference, ``1``
        otherwise. A diagnostic that always exits 0 makes the job state meaningless.

    Note:
        A ``0`` here is evidence about ``linear`` and ``rms_norm`` only. It is not
        evidence that the decode step is capturable; see the module docstring.
    """
    import torch

    import backends.corex as corex
    from backends.corex import pack

    _sep("0. backend state")
    print(" ", corex.describe(), flush=True)
    # Deliberately not gated on ``corex.usable()`` alone: this script runs on the card
    # either way, and a capture attempt on an unusable backend is the more useful failure.
    table = corex.ops()
    failures: list[str] = []

    _sep("1. linear: capture one GEMV, change the activation, replay")
    n, k = 9728, 2560
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(k, dtype=torch.bfloat16, device="cuda")
    packed = pack.as_kn(w)
    torch.cuda.synchronize()
    # Warm the allocator outside the capture: a first-touch allocation inside the
    # capture is a different code path and would test the allocator, not the kernel.
    table.linear(w, x)
    torch.cuda.synchronize()
    _stage("linear capture")
    try:
        graph, out = _capture(torch, lambda: table.linear(w, x))
    except Exception as exc:                      # noqa: BLE001 - the answer either way
        print(f"  capture FAILED: {type(exc).__name__}: {exc}", flush=True)
        return 1
    torch.cuda.synchronize()
    # A capture only *records* the kernel: nothing ran, so the output buffer holds
    # whatever the allocator handed over. Its value at this point says nothing about
    # the kernel, which is why the first measurement is a replay -- of the capture's
    # own input, before it is changed. Reading the buffer here instead would report
    # the uninitialised contents and fail every correct backend.
    _stage("linear replay on captured input")
    graph.replay()
    torch.cuda.synchronize()
    ref_captured = _host(x) @ _host(packed.wt)
    rel_captured = _rel(out, ref_captured)
    _stage("linear replay on changed input")
    x.normal_()
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    ref_replay = _host(x) @ _host(packed.wt)
    rel_replay = _rel(out, ref_replay)
    moved = float((_host(out) - ref_captured).abs().max())
    print(f"  capture rel={rel_captured:.3e}  replay rel={rel_replay:.3e}  "
          f"output moved by {moved:.3e}", flush=True)
    if not (rel_captured < 2e-2 and rel_replay < 2e-2 and moved > 0):
        failures.append("linear graph replay")

    _sep("2. rms_norm: capture and replay, same discipline")
    m = 8
    xs = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")
    g = torch.randn(n, dtype=torch.bfloat16, device="cuda")
    torch.cuda.synchronize()
    table.rms_norm(xs, g, 0, 1e-6)
    torch.cuda.synchronize()
    _stage("rms_norm capture")
    try:
        norm_graph, norm_out = _capture(torch, lambda: table.rms_norm(xs, g, 0, 1e-6))
    except Exception as exc:                      # noqa: BLE001
        print(f"  capture FAILED: {type(exc).__name__}: {exc}", flush=True)
        return 1
    torch.cuda.synchronize()
    _stage("rms_norm replay")
    xs.normal_()
    torch.cuda.synchronize()
    norm_graph.replay()
    torch.cuda.synchronize()
    ref = (_host(xs) * torch.rsqrt(_host(xs).pow(2).mean(-1, keepdim=True) + 1e-6)
           * _host(g))
    rel = _rel(norm_out, ref)
    print(f"  replay rel={rel:.3e}", flush=True)
    if not rel < 2e-2:
        failures.append("rms_norm graph replay")

    _sep("summary")
    print(f"  failures: {failures or '(none)'}", flush=True)
    print("  scope: two ops in isolation; this says nothing about the decode step",
          flush=True)
    print(f"  per-op capture: {'OK' if not failures else 'FAILED'}", flush=True)
    print("\nCOREX_GRAPH_DONE", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
