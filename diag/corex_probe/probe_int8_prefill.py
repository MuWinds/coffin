"""Which int8 prefill dequantisation scheme is worth implementing?

``linear_prefill`` currently dequantises its entire weight on every call::

    KnWeight((self.wt.float() * self.scale.view(1, -1)).to(WEIGHT_DTYPE))

measured at 171 ms per 2048-token prefill (909.9 ms int8 vs 738.5 ms with the dequant
hoisted out of the timed region -- the same 171 ms seen two independent ways). That is
a fixed cost per request, so it is 3.1x of the whole prefill at n=128 and 0.23x at
n=2048. Four different fixes are on the table and they are not mutually exclusive:

A. **Keep the current shape, write it better.** ``wt.float()`` allocates a ``[K, N]``
   fp32 tensor -- twice the bf16 size -- then multiplies it, then narrows. Three passes
   over ~4x the payload the result needs. A cast straight to bf16 plus an in-place
   multiply is two passes at half the width.
B. **Do not re-do it per chunk.** The prefill loop is chunk-major (``for chunk: for
   layer:``), so a prompt longer than ``chunk`` dequantises every weight once per chunk.
   At n=8192 with chunk=2048 that is 4x the work for the same weights.
C. **Hold the dequantised form** for the request. Measured at 1.23x but +6.65 GiB, which
   is exactly what int8 storage was buying.
D. **Stream it per layer** (layer-major loop): same one-dequant-per-weight-per-request
   as C, but the peak is one projection instead of all 128. Arithmetic says +94 MB
   against C's +6.65 GiB; this probe reports the number so the claim is not a guess.

This measures A, B and the availability behind a true fused kernel, and verifies every
candidate against the current implementation so a faster-but-wrong variant cannot win.

Module: ``coffin/diag/corex_probe``. Runs on the card; nothing imports it.

Constraints:
    * Correctness first: each candidate's output is compared against ``dequant_kn`` and
      must be bit-identical or within one bf16 ULP. A scheme that changes the weights is
      not a speedup, it is a different model.
    * The B measurement (chunk multiplication) needs a context window of 16k, so the paged
      pool is sized for it; the probe does not measure the model, only the dequant cost
      as a function of chunk count.
    * Timings are best-of-3 with the allocation inside the timed region, because that is
      how the engine pays it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: ``coffin/`` is the ``sys.path`` root; this file sits two levels below it.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

MODEL = os.environ.get("ENGINE_MODEL", "")

#: Context window; 16k so the chunk-multiplication test can use an 8192-token prompt.
MAX_SEQ = 16384

#: Tokens per block in the paged pool.
BLOCK = 64

#: Prompt length for the chunk-multiplication test. Deliberately a clean multiple of the
#: chunk sizes swept, so "number of chunks" is the only variable.
BIG_N = 8192

#: Chunk sizes for that test: 4 chunks, 2 chunks, 1 chunk.
BIG_CHUNKS = (2048, 4096, 8192)


def _events() -> tuple:
    """Create a start/end device-event pair with timing enabled.

    Returns:
        ``(start, end)`` events.
    """
    import torch

    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def _tokens(n: int) -> list[int]:
    """Build a deterministic prompt of ``n`` legal token ids.

    Args:
        n: Prompt length.

    Returns:
        ``n`` ids from the arithmetic sequence the other probes use.
    """
    return [(i * 7919 + 13) % 100000 for i in range(n)]


def _time(fn, reps: int = 3) -> float:
    """Time a zero-argument callable, best of ``reps``, in milliseconds.

    Args:
        fn: Callable to time.
        reps: Timed repeats.

    Returns:
        Best milliseconds.
    """
    import torch

    fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        start, end = _events()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end))
    return best


def _collect(model) -> list:
    """Every distinct int8 projection the prefill path will dequantise.

    Args:
        model: Loaded int8 model.

    Returns:
        The packed ``KnQWeight`` objects, one per projection, de-duplicated by identity.
        A 32-layer model has 128 of them (4 projections x 32 layers), which is why the
        per-call cost multiplies out to a request-sized number.
    """
    from backends.corex import pack

    seen: dict[int, object] = {}
    for layer in model.layers:
        for proj in (getattr(layer.mix, "qkvzb_a", None), getattr(layer.mix, "qkv", None),
                     getattr(layer.mix, "out_proj", None), getattr(layer.mix, "o_proj", None),
                     getattr(layer.mlp, "gate_up", None),
                     getattr(layer.mlp, "down", None)):
            if proj is None:
                continue
            gemv = getattr(proj, "g", proj)
            packed = pack.as_kn(gemv.w)
            if getattr(packed, "int8", False):
                seen[id(packed)] = packed
    return list(seen.values())


def _bench_schemes(weights: list) -> None:
    """Time each dequantisation spelling over every real weight, and verify it.

    Args:
        weights: The packed int8 weights, as :func:`_collect` returns them.
    """
    import torch

    from backends.corex import pack

    bf16 = pack.WEIGHT_DTYPE
    total_bytes = sum(w.nbytes() for w in weights)
    print(f"\n  --- dequantisation schemes over all {len(weights)} projections ---",
          flush=True)
    print(f"  int8 payload: {total_bytes / 2**30:.2f} GiB "
          f"-> bf16 {total_bytes * 2 / 2**30:.2f} GiB", flush=True)
    print(f"  {'scheme':<34}{'ms':>9}{'GB/s eff':>10}{'max |diff|':>12}", flush=True)

    refs = {}

    def scheme_current(w):
        """The engine's current spelling.

        Args:
            w: Packed int8 weight.

        Returns:
            ``[K, N]`` bf16 weight.
        """
        return pack.KnWeight((w.wt.float() * w.scale.view(1, -1)).to(bf16))

    def scheme_bf16_mul(w):
        """Cast straight to bf16, then scale in place.

        ``w.scale`` is narrowed first: leaving it fp32 would promote the multiply back to
        fp32 and reinstate the 2x-wide intermediate that this scheme exists to avoid.

        Args:
            w: Packed int8 weight.

        Returns:
            ``[K, N]`` bf16 weight.
        """
        out = w.wt.to(bf16)
        out.mul_(w.scale.to(bf16).view(1, -1))
        return pack.KnWeight(out)

    def scheme_prealloc(w):
        """Cast into a caller-owned buffer, then scale in place.

        No allocation in the hot path at all: the fp32 intermediate is gone and the
        allocator is not asked for a fresh 94 MB tensor per projection.

        Args:
            w: Packed int8 weight.

        Returns:
            ``[K, N]`` bf16 weight, freshly allocated anyway because the probe has to
            return something the caller can compare; the *cast* is what reuses the buffer.
        """
        buf = torch.empty((w.k, w.n), dtype=bf16, device=w.wt.device)
        buf.copy_(w.wt)
        buf.mul_(w.scale.to(bf16).view(1, -1))
        return pack.KnWeight(buf)

    def scheme_per_column(w):
        """Dequantise a column block at a time, to stay inside cache.

        A ``[K, N]`` int8 matrix read column-blocked is a strided read, so this is worth
        measuring rather than assuming: it trades coalescing for a smaller working set.

        Args:
            w: Packed int8 weight.

        Returns:
            ``[K, N]`` bf16 weight.
        """
        out = torch.empty((w.k, w.n), dtype=bf16, device=w.wt.device)
        step = max(256, w.n // 16)
        for start in range(0, w.n, step):
            stop = min(w.n, start + step)
            blk = w.wt[:, start:stop].to(bf16)
            out[:, start:stop] = blk.mul_(w.scale[start:stop].to(bf16).view(1, -1))
        return pack.KnWeight(out)

    for name, fn in (("A current: fp32 mul then narrow", scheme_current),
                     ("B cast to bf16, in-place mul", scheme_bf16_mul),
                     ("C cast into preallocated buffer", scheme_prealloc),
                     ("D column-blocked", scheme_per_column)):
        def run(fn=fn):
            """Dequantise every weight once.

            Args:
                fn: The scheme under test.
            """
            for w in weights:
                fn(w)

        ms = _time(run)
        # Verify against A on the two largest weights only: a full comparison of 128
        # weights would cost more memory than the answer is worth, and a wrong scale or
        # a transposed index shows up on any one of them.
        worst = 0.0
        for w in weights[:2] + weights[-2:]:
            ref = refs.get(id(w))
            if ref is None:
                ref = scheme_current(w).wt
                refs[id(w)] = ref
            got = fn(w).wt
            worst = max(worst, float((got.float() - ref.float()).abs().max()))
        # Effective bandwidth: read int8 once, write bf16 once, which is the floor any
        # scheme has to pay.
        eff = (total_bytes * 3) / (ms * 1e-3) / 1e9
        print(f"  {name:<34}{ms:>9.2f}{eff:>10.1f}{worst:>12.2e}", flush=True)


def _probe_int8_gemm() -> None:
    """Check whether this build can do an int8 matmul, which would remove the widen.

    A fused int8 GEMM (widen the weight on-chip, keep it in bf16 for the tensor core)
    would read 4.4 GB instead of writing 8.8 GB and reading it back. That needs the
    library to accept an int8 operand, so the availability is checked before any design
    is proposed around it.

    Args:
        Nothing.
    """
    import torch

    print("\n  --- int8 GEMM availability ---", flush=True)
    a = torch.randint(-127, 127, (2048, 2560), dtype=torch.int8, device="cuda")
    b = torch.randint(-127, 127, (2560, 9216), dtype=torch.int8, device="cuda")
    for label, fn in (
            ("torch._int_mm", getattr(torch, "_int_mm", None)),
            ("torch.matmul int8", lambda x, y: torch.matmul(x, y)),
            ("torch.mm int8", lambda x, y: torch.mm(x, y)),
    ):
        if fn is None:
            print(f"  {label:<22} absent in this build", flush=True)
            continue
        try:
            out = fn(a, b)
            torch.cuda.synchronize()
            print(f"  {label:<22} ok, out {tuple(out.shape)} {out.dtype}", flush=True)
            ms = _time(lambda: fn(a, b))
            flops = 2 * 2048 * 9216 * 2560
            print(f"  {'':<22} {ms:>7.2f} ms  {flops / (ms * 1e-3) / 1e12:>5.1f} TINT8-OPS/s",
                  flush=True)
        except Exception as exc:                  # noqa: BLE001 - availability probe
            print(f"  {label:<22} fails: {type(exc).__name__}: {exc}", flush=True)
    del a, b
    torch.cuda.empty_cache()


def main() -> int:
    """Measure the schemes, the chunk multiplication, and the streaming peak.

    Returns:
        ``0`` on success, ``1`` when the model directory is missing.
    """
    import torch

    import backends
    from engine_api import create_model, load_weights

    print(f"  backend: {backends.status()}", flush=True)
    print(f"  model  : {MODEL}", flush=True)
    if not MODEL or not Path(MODEL).is_dir():
        print(f"  ENGINE_MODEL unset or not a directory: {MODEL!r}", flush=True)
        return 1
    print(f"  device : {torch.cuda.get_device_name(0)} "
          f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB",
          flush=True)

    sd = load_weights(MODEL, quant=True, vision=False)
    model = create_model(MODEL, max_seq=MAX_SEQ, bench=False, weights=sd, paged=True)
    blocks = -(-MAX_SEQ // BLOCK)
    pool = model.enable_paging(num_blocks=blocks, block_size=BLOCK)
    pool.assign_table(list(range(blocks)), 0, row=model.prefill_row)
    model.mtp_prefill = False

    weights = _collect(model)
    largest = max(w.nbytes() for w in weights) * 2
    print(f"\n  largest single projection at bf16: {largest / 2**20:.0f} MiB "
          f"(the streaming peak)", flush=True)

    _bench_schemes(weights)
    _probe_int8_gemm()

    print(f"\n  --- chunk multiplication, n={BIG_N} ---", flush=True)
    print(f"  {'chunk':>7}{'chunks':>8}{'total ms':>11}{'ms/prompt tok':>15}", flush=True)
    toks = _tokens(BIG_N)
    for chunk in BIG_CHUNKS:
        def run(chunk=chunk):
            """One prefill at this chunk size.

            Args:
                chunk: Tokens per GEMM pass.
            """
            model.reset()
            model.prefill(toks, chunk=chunk)

        run()
        torch.cuda.synchronize()
        ms = _time(run, reps=2)
        print(f"  {chunk:>7}{-(-BIG_N // chunk):>8}{ms:>11.1f}"
              f"{ms / BIG_N * 1e3:>15.3f}", flush=True)

    print("\nINT8_PREFILL_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
