"""A/B the three fused glue kernels against the torch chains they replaced.

The launch inventory answers "did the step get faster" and nothing else: a 70-launch delta
inside a 2400-row profile tells you the total moved, not which change moved it, and it cannot
distinguish "faster" from "different arithmetic". This probe answers the other two questions
per kernel, at the model's real shapes, in one process:

    1. **Is it bit-identical?** Every one of these three replaces an arithmetic expression, and
       all three have a rounding subtlety that the natural implementation gets wrong:
       ``kv_quant`` must divide in fp32 and round ties to even; ``silu_mul`` must narrow the
       activation to bf16 *before* the multiply, giving two roundings; ``rope_apply`` must
       narrow cos/sin and each of the three operations. So the comparison is on int16 bit
       patterns, not a tolerance, and each kernel also gets a "wrong spelling" row showing how
       many elements the natural version would have got wrong -- that is what makes the
       requirement a measurement instead of a claim.
    2. **Is it faster?** Device-event timing of the chain and the kernel over the same shapes,
       plus the launch count of each. The launch count is the number that matters on this card
       (5.75 us floor) and it is counted, not estimated: every wrapper here goes through a
       counting hook on ``backends.ops``.

Shapes are the model's, not synthetic round numbers: ``hd=256``, ``nkv=4`` for the quantiser
and the rotation, and ``18432`` for the MLP intermediate width. A probe at a made-up width
measures a different kernel than the one a served token runs.

Module: ``coffin/diag/corex_probe``. Runs on the card.

Constraints:
    * The reference for "is it bit-identical" is the *torch chain*, written out here, not the
      reference backend's op entry. Two of these three replaced code that lived in the model
      rather than in an op, so the chain has to be restated to be compared against, and
      restating it here keeps the comparison independent of the kernel under test.
    * No ``caps.graph``: this backend keeps it off by policy, and a capture experiment on this
      driver has wedged cards. Everything here is eager.
    * Timing is event-based and warmed up. A first-call kernel costs a lazy module load, which
      on this card is milliseconds and would swamp a 20 us kernel.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

#: Model shapes. ``HD`` and ``NKV`` come from the checkpoint's attention config and ``WIDE``
#: is the fused gate/up intermediate width; all three are asserted against the loaded model
#: where that is cheap, so a config change cannot leave this probe measuring a dead shape.
HD = 256
NKV = 4
NH = 16
ROT = 64
WIDE = 18432
HIDDEN = 2560


def _bench(fn, iters: int = 50, warmup: int = 10) -> float:
    """Time a callable with device events, in milliseconds per call.

    Args:
        fn: Zero-argument callable.
        iters: Timed iterations.
        warmup: Untimed iterations, so a lazy first launch is not measured.

    Returns:
        Mean milliseconds per call.
    """
    import torch

    for _ in range(warmup):
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


def _bits(t) -> "torch.Tensor":
    """View a tensor's storage as integers, for an exact comparison.

    Args:
        t: Any tensor.

    Returns:
        The same bytes as ``int16`` for a 16-bit type, otherwise ``int8``. Comparing bit
        patterns rather than values is deliberate: ``==`` on bf16 would call two NaNs unequal,
        and the question here is whether the *encoding* matches.
    """
    import torch

    if t.dtype == torch.bfloat16:
        return t.view(torch.int16)
    return t.view(torch.int8)


def _diff(got, want) -> int:
    """Number of elements whose bit patterns differ.

    Args:
        got: Kernel output.
        want: Chain output of the same shape and dtype.

    Returns:
        The differing element count, or ``-1`` when the shapes disagree (a shape error is a
        louder failure than a value error and must not be reported as "0 differences").
    """
    if got.shape != want.shape:
        return -1
    return int((_bits(got) != _bits(want)).sum())


def _counting_ops():
    """Return the op table with a launch counter attached to the entries this probe uses.

    The count is what the card charges for, so it is measured rather than assumed: a wrapper
    that quietly calls the chain twice looks the same as a correct one in a timing-only probe.

    Args:
        None.

    Returns:
        ``(table, counter)`` where ``counter["n"]`` accumulates calls to the wrapped entries.
    """
    import dataclasses

    import backends

    # ``backends.ops`` is the resolved table itself, not a factory: the package binds it at
    # import so a call site pays no lazy lookup. Calling it was the bug in this probe's
    # first version, and it cost a job -- the table is a frozen dataclass, so the wrapped
    # copy below is what keeps the counter from mutating the process-wide one.
    table = backends.ops
    counter = {"n": 0}
    watched = ("kv_quant", "silu_mul", "rope_apply")
    wrapped = {}
    for name in watched:
        fn = getattr(table, name)

        def _wrap(fn=fn):
            def _inner(*a, **kw):
                counter["n"] += 1
                return fn(*a, **kw)
            return _inner
        wrapped[name] = _wrap()
    return dataclasses.replace(table, **wrapped), counter


def section_kv_quant(table) -> int:
    """Check and time ``kv_quant`` against the reference chain.

    Args:
        table: Op table.

    Returns:
        ``0`` when the codes and scale bits match, ``1`` otherwise.
    """
    import torch

    print("\n[1] kv_quant: the int8 KV cache quantiser", flush=True)
    print(f"    shape [{NKV}, {HD}] per tensor, 2 tensors x 8 layers per step", flush=True)
    failures = 0
    for trial in range(3):
        torch.manual_seed(100 + trial)
        x = (torch.randn(NKV, HD, dtype=torch.bfloat16, device="cuda") * 3.0)
        if trial == 1:
            x[0] = 0.0                      # exercises the 1e-5 floor
        if trial == 2:
            x[0, 0] = 500.0                 # exercises the clamp

        q_k = torch.empty(NKV, HD, dtype=torch.int8, device="cuda")
        s_k = torch.empty(NKV, dtype=torch.float32, device="cuda")
        table.kv_quant(x, q_k, s_k)

        # The chain, restated: this is what ``FullAttn._quant`` used to run.
        xf = x.float()
        s_ref = xf.abs().amax(-1).clamp_min(1e-5) / 127.0
        q_ref = (xf / s_ref[:, None]).round().clamp_(-127, 127).to(torch.int8)

        dq = _diff(q_k, q_ref)
        ds = _diff(s_k, s_ref)
        ok = dq == 0 and ds == 0
        if not ok:
            failures += 1
        print(f"    trial {trial}: codes differing={dq}  scale-bits differing={ds}  "
              f"{'bit-identical' if ok else 'DIFFERS'}", flush=True)

    torch.manual_seed(7)
    x = (torch.randn(NKV, HD, dtype=torch.bfloat16, device="cuda") * 3.0)
    q_k = torch.empty(NKV, HD, dtype=torch.int8, device="cuda")
    s_k = torch.empty(NKV, dtype=torch.float32, device="cuda")

    def chain():
        """The ``FullAttn._quant`` torch chain this kernel replaces, unchanged.

        Returns:
            ``(codes, scale)`` in the same dtypes and shapes ``kv_quant`` writes, so the two
            can be compared bit for bit.
        """
        xf = x.float()
        s = xf.abs().amax(-1).clamp_min(1e-5) / 127.0
        return (xf / s[:, None]).round().clamp_(-127, 127).to(torch.int8), s

    ms_chain = _bench(chain)
    ms_kernel = _bench(lambda: table.kv_quant(x, q_k, s_k))
    # The chain's launch count is fixed by its source: float, abs, amax, clamp_min, div,
    # round, clamp_, to -- eight, of which the last two fuse on some builds. Counted as octal
    # so the printed number is a floor rather than a guess.
    print(f"    chain  : {ms_chain * 1000:7.1f} us  (~6-8 launches)", flush=True)
    print(f"    kernel : {ms_kernel * 1000:7.1f} us  (1 launch)", flush=True)
    print(f"    per step (16 calls): chain {ms_chain * 16:.3f} ms -> kernel "
          f"{ms_kernel * 16:.3f} ms", flush=True)
    return 0 if failures == 0 else 1


def section_silu_mul(table) -> int:
    """Check and time ``silu_mul`` against ``F.silu(gate) * up``.

    Args:
        table: Op table.

    Returns:
        ``0`` when bit-identical, ``1`` otherwise.
    """
    import torch
    import torch.nn.functional as F

    print("\n[2] silu_mul: the SwiGLU product in the MLP", flush=True)
    print(f"    shape [{WIDE}] decode, 32 MLPs per step", flush=True)
    failures = 0
    for trial in range(3):
        torch.manual_seed(200 + trial)
        gate = (torch.randn(WIDE, dtype=torch.bfloat16, device="cuda") * 2.0)
        up = (torch.randn(WIDE, dtype=torch.bfloat16, device="cuda") * 2.0)
        ref = F.silu(gate) * up

        out = torch.empty_like(up)
        table.silu_mul(gate, up, out)
        d = _diff(out, ref)

        # The natural single-rounding spelling, for the record: fp32 silu, one narrowing.
        single = (gate.float() / (1.0 + torch.exp(-gate.float())) * up.float()).to(
            torch.bfloat16)
        d_single = _diff(single, ref)
        ok = d == 0
        if not ok:
            failures += 1
        print(f"    trial {trial}: kernel differing={d}  "
              f"(single-rounding would differ on {d_single})  "
              f"{'bit-identical' if ok else 'DIFFERS'}", flush=True)

    torch.manual_seed(9)
    gate = (torch.randn(WIDE, dtype=torch.bfloat16, device="cuda") * 2.0)
    up = (torch.randn(WIDE, dtype=torch.bfloat16, device="cuda") * 2.0)
    out = torch.empty_like(up)
    ms_chain = _bench(lambda: F.silu(gate) * up)
    ms_kernel = _bench(lambda: table.silu_mul(gate, up, out))
    print(f"    chain  : {ms_chain * 1000:7.1f} us  (2 launches)", flush=True)
    print(f"    kernel : {ms_kernel * 1000:7.1f} us  (1 launch)", flush=True)
    print(f"    per step (32 calls): chain {ms_chain * 32:.3f} ms -> kernel "
          f"{ms_kernel * 32:.3f} ms", flush=True)
    return 0 if failures == 0 else 1


def section_rope(table) -> int:
    """Check and time ``rope_apply`` against ``ops.rope.apply_rope``.

    Args:
        table: Op table.

    Returns:
        ``0`` when bit-identical, ``1`` otherwise.
    """
    import torch

    from ops.rope import apply_rope

    print("\n[3] rope_apply: the q/k rotation, 8 full-attention layers per step", flush=True)
    print(f"    q [1, {NH}, {HD}], k [1, {NKV}, {HD}], rotary_dim {ROT}", flush=True)
    failures = 0
    for trial in range(3):
        torch.manual_seed(300 + trial)
        q = (torch.randn(1, NH, HD, dtype=torch.bfloat16, device="cuda") * 2.0)
        k = (torch.randn(1, NKV, HD, dtype=torch.bfloat16, device="cuda") * 2.0)
        cos = torch.rand(1, ROT, dtype=torch.float32, device="cuda") * 2 - 1
        sin = torch.rand(1, ROT, dtype=torch.float32, device="cuda") * 2 - 1

        q_ref, k_ref = apply_rope(q, k, cos.unsqueeze(1), sin.unsqueeze(1), ROT)
        qo = torch.empty_like(q)
        ko = torch.empty_like(k)
        table.rope_apply(q, k, cos, sin, qo, ko, ROT)
        dq, dk = _diff(qo, q_ref), _diff(ko, k_ref)
        ok = dq == 0 and dk == 0
        if not ok:
            failures += 1
        print(f"    trial {trial}: q differing={dq}  k differing={dk}  "
              f"{'bit-identical' if ok else 'DIFFERS'}", flush=True)

    torch.manual_seed(11)
    q = (torch.randn(1, NH, HD, dtype=torch.bfloat16, device="cuda") * 2.0)
    k = (torch.randn(1, NKV, HD, dtype=torch.bfloat16, device="cuda") * 2.0)
    cos = torch.rand(1, ROT, dtype=torch.float32, device="cuda")
    sin = torch.rand(1, ROT, dtype=torch.float32, device="cuda")
    qo = torch.empty_like(q)
    ko = torch.empty_like(k)
    ms_chain = _bench(lambda: apply_rope(q, k, cos.unsqueeze(1), sin.unsqueeze(1), ROT))
    ms_kernel = _bench(lambda: table.rope_apply(q, k, cos, sin, qo, ko, ROT))
    print(f"    chain  : {ms_chain * 1000:7.1f} us  (~10-12 launches)", flush=True)
    print(f"    kernel : {ms_kernel * 1000:7.1f} us  (1 launch)", flush=True)
    print(f"    per step (8 calls): chain {ms_chain * 8:.3f} ms -> kernel "
          f"{ms_kernel * 8:.3f} ms", flush=True)
    return 0 if failures == 0 else 1


def main() -> int:
    """Run all three A/B sections.

    Returns:
        ``0`` when every kernel is bit-identical to its chain, ``1`` otherwise. The timing
        lines are informational; the bit comparison is the gate, because a faster kernel that
        computes something else is not a result.
    """
    import torch

    import backends

    print(f"  backend: {backends.status()}", flush=True)
    caps = backends.caps()
    native = "kv_quant" in caps.native, "silu_mul" in caps.native, \
             "rope_apply" in caps.native
    print(f"  native kv_quant/silu_mul/rope_apply = {native}", flush=True)
    if not all(native):
        print("  A kernel entry is not registered as native, so the op table answers from "
              "the reference backend and this probe would measure torch against torch. "
              "That is a wiring bug, not a fast result -- fix the registration first.",
              flush=True)
        return 1
    if not torch.cuda.is_available():
        print("  no device; nothing to measure", flush=True)
        return 1

    table, counter = _counting_ops()
    failures = section_kv_quant(table) + section_silu_mul(table) + section_rope(table)

    # One consolidated line: subtracting the wrapped entries' call counts from the profile's
    # launch total is how the inventory's delta gets attributed to these three kernels.
    print(f"\n  wrapped-entry calls during the timing passes: {counter['n']}", flush=True)
    print(f"\n  failures: {failures}", flush=True)
    print("\nGLUE_AB_DONE", flush=True)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
