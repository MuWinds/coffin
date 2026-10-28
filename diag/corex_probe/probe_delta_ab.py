"""A/B the fused delta kernel against the three-op chain it replaces, in one process.

The launch-floor argument says a fused kernel should win: replacing three launches with one
removes two launch slots, and on this card a launch costs ~5.75 us whether it does work or
not. The chain it replaces is:

    delta_prologue -> delta -> rms_norm_gated

But the fusion is not free. ``delta_kernel`` is fast because one thread owns one value
column and walks ``dk`` in registers, so it has **no cross-lane communication at all**. The
prologue and the gated norm each need a block reduction. A single kernel must pick one
geometry for all three, and if that compromise costs more device time than the two launch
slots it saves, the fusion is a pessimisation -- which is a real possibility on a part this
small (16 SMs), not a theoretical one.

Comparing the two across separate jobs cannot answer that: individual kernel times are
stable to a few percent between runs, but the *wall* differs by ~2% between nodes, so a
0.3 ms effect is inside the noise. This probe measures both paths in one process, on the
same buffers, with the same clocks, alternating them so a thermal drift hits both equally.

It reports device time per call for each path and the delta, and it checks the fused
kernel's output against the chain's bit for bit -- a fusion that wins but changes the
numbers is not eligible.

Module: ``coffin/diag/corex_probe``. Runs on the card. No model needed.

Constraints:
    * Both paths run on the same fp32 state buffer *object* but the probe resets it between
      measurements, because ``delta`` updates the state in place and the two paths would
      otherwise be racing to different recurrent histories.
    * Device events, not wall clock: the question is what the card spends, and the host is
      not the critical path here.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

#: The model's real linear-attention geometry, from config.json.
NUM_V = 32
NUM_K = 16
DK = 128
DV = 128
REP = 2
QSCALE = DK ** -0.5

#: Layers that carry this mixer: 24 of 32.
LAYERS = 24

#: Repetitions per measurement.
ITERS = 50

#: Alternating rounds, so a drift in clocks or temperature lands on both paths.
ROUNDS = 5


def _time(fn, iters: int) -> float:
    """Time a callable with device events.

    Args:
        fn: Zero-argument callable performing one fused call or one chain.
        iters: Calls per sample.

    Returns:
        Mean milliseconds per call.
    """
    import torch

    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> int:
    """Run the A/B and print the per-call and per-step deltas.

    Returns:
        ``0`` on success.
    """
    import torch

    import backends
    from backends import ops as bops

    print(f"  python {sys.version.split()[0]}  host={os.uname().nodename}", flush=True)
    print(f"  {backends.status()}", flush=True)
    print(f"  native delta_normed: {'delta_normed' in backends.caps().native}", flush=True)

    dev = "cuda"
    dt = torch.bfloat16

    # Inputs, at the real geometry. Fresh each round by re-filling, so neither path sees a
    # state the other advanced.
    conv = torch.randn(NUM_V * DV * 3, dtype=torch.float32, device=dev) * 0.1
    qr = torch.empty(NUM_K, DK, dtype=torch.float32, device=dev)
    kr = torch.empty(NUM_K, DK, dtype=torch.float32, device=dev)
    vr = torch.empty(NUM_V, DV, dtype=torch.float32, device=dev)
    # The chain's prologue takes the raw conv row and slices it, mirroring LinearAttn.
    kdim = DK * NUM_K
    vdim = DV * NUM_V
    src = torch.randn(kdim * 2 + vdim, dtype=torch.float32, device=dev) * 0.1
    b = torch.randn(NUM_V, dtype=dt, device=dev) * 0.1
    a = torch.randn(NUM_V, dtype=dt, device=dev) * 0.1
    z = torch.randn(NUM_V, DV, dtype=dt, device=dev) * 0.1
    neg_expA = -torch.rand(NUM_V, dtype=torch.float32, device=dev) - 0.5
    dt_bias = torch.randn(NUM_V, dtype=torch.float32, device=dev) * 0.1
    nw = torch.randn(DV, dtype=dt, device=dev) * 0.02 + 1.0

    state = torch.zeros(NUM_V, DK, DV, dtype=torch.float32, device=dev)

    q_pro = torch.zeros(NUM_V, DK, dtype=torch.float32, device=dev)
    k_pro = torch.zeros(NUM_V, DK, dtype=torch.float32, device=dev)
    v_pro = torch.zeros(NUM_V, DV, dtype=torch.float32, device=dev)
    beta = torch.zeros(NUM_V, dtype=torch.float32, device=dev)
    g = torch.zeros(NUM_V, dtype=torch.float32, device=dev)
    d_out = torch.zeros(NUM_V, DV, dtype=torch.float32, device=dev)
    normed = torch.zeros(NUM_V, DV, dtype=dt, device=dev)

    q_slice = torch.empty(NUM_K, DK, dtype=torch.float32, device=dev)
    k_slice = torch.empty(NUM_K, DK, dtype=torch.float32, device=dev)
    v_slice = torch.empty(NUM_V, DV, dtype=torch.float32, device=dev)

    def chain() -> None:
        """The three-op spelling, against the same buffers."""
        q_slice.copy_(src[:kdim].view(NUM_K, DK))
        k_slice.copy_(src[kdim:2 * kdim].view(NUM_K, DK))
        v_slice.copy_(src[2 * kdim:].view(NUM_V, DV))
        bops.delta_prologue(q_slice, k_slice, v_slice, b, a, neg_expA, dt_bias,
                            q_pro, k_pro, v_pro, beta, g, QSCALE, REP)
        bops.delta(q_pro, k_pro, v_pro, beta, g, state, d_out)
        # The chain's norm input is cast to the activation dtype first: ``delta`` writes fp32
        # scratch and ``rms_norm_gated`` in the model receives an already-bf16 tensor, so
        # casting here is the faithful spelling rather than a convenience.
        bops.rms_norm_gated(d_out.to(dt), nw, z, 1e-6)

    def fused() -> None:
        """The single fused entry, against the same buffers."""
        q_slice.copy_(src[:kdim].view(NUM_K, DK))
        k_slice.copy_(src[kdim:2 * kdim].view(NUM_K, DK))
        v_slice.copy_(src[2 * kdim:].view(NUM_V, DV))
        bops.delta_normed(q_slice, k_slice, v_slice, b, a, neg_expA, dt_bias,
                          z, nw, state, normed, QSCALE, REP)

    # The copies above are part of both paths equally and would otherwise dominate the
    # measurement at this size; time them once and subtract, so the number quoted is the
    # mixer's own cost. Stated rather than silently folded in.
    def copies_only() -> None:
        """Just the three slice copies, which both paths pay."""
        q_slice.copy_(src[:kdim].view(NUM_K, DK))
        k_slice.copy_(src[kdim:2 * kdim].view(NUM_K, DK))
        v_slice.copy_(src[2 * kdim:].view(NUM_V, DV))

    chain_ms = []
    fused_ms = []
    copy_ms = []
    for r in range(ROUNDS):
        state.zero_()
        chain_ms.append(_time(chain, ITERS))
        state.zero_()
        fused_ms.append(_time(fused, ITERS))
        copy_ms.append(_time(copies_only, ITERS))
        print(f"  round {r}: chain {chain_ms[-1] * 1000:7.2f} us  "
              f"fused {fused_ms[-1] * 1000:7.2f} us  "
              f"copies {copy_ms[-1] * 1000:5.2f} us", flush=True)

    # The minimum of each, which is the least-contended sample: a shared node makes every
    # round slower and only the fastest one shows what the kernel costs.
    c = min(chain_ms) - min(copy_ms)
    f = min(fused_ms) - min(copy_ms)
    print(f"\n  chain  (3 launches): {c * 1000:7.2f} us/call", flush=True)
    print(f"  fused  (1 launch)  : {f * 1000:7.2f} us/call", flush=True)
    per_layer = (f - c) * 1000
    print(f"  fused - chain      : {per_layer:+7.2f} us/call", flush=True)
    print(f"  x{LAYERS} layers       : {per_layer * LAYERS / 1000:+.3f} ms/step", flush=True)
    print(f"  launch slots saved : 2 per layer = {2 * LAYERS} launches/step", flush=True)
    print(f"  wall at 5.75 us floor: {-2 * LAYERS * 5.75 / 1000:+.3f} ms/step", flush=True)
    print(f"  NET                : "
          f"{(per_layer * LAYERS - 2 * LAYERS * 5.75) / 1000:+.3f} ms/step", flush=True)

    # Bit-exactness: a faster spelling that changes the numbers is not eligible.
    state.zero_()
    copies_only()
    bops.delta_prologue(q_slice, k_slice, v_slice, b, a, neg_expA, dt_bias,
                        q_pro, k_pro, v_pro, beta, g, QSCALE, REP)
    bops.delta(q_pro, k_pro, v_pro, beta, g, state, d_out)
    ref = bops.rms_norm_gated(d_out.to(dt), nw, z, 1e-6).float().clone()
    ref_state = state.clone()

    state.zero_()
    fused()
    got = normed.float().clone()
    got_state = state.clone()

    print(f"\n  output bit-identical : {bool(torch.equal(ref, got))}", flush=True)
    print(f"  state  bit-identical : {bool(torch.equal(ref_state, got_state))}", flush=True)
    if not torch.equal(ref, got):
        print(f"  max|d| = {float((ref - got).abs().max())}", flush=True)

    print("\nDELTA_AB_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
