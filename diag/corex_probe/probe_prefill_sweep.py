"""Sweep the two prefill attention knobs, which were tuned on the DCU and never here.

``PREFILL_ATTN_TILE`` (512) and ``PREFILL_ATTN_KB`` (1024) come from
``diag/probe_prefill_split.py`` on the Hygon DCU. Both parts of that measurement have
changed underneath them:

* The tile decides how much of the causal upper triangle is skipped. Its optimum is a
  trade between saved score-block work and added launches, and this card's per-launch
  cost differs from the DCU's by more than an order of magnitude in the other direction --
  prefill on the BI-V100 is compute-bound at 45-51 TFLOP/s, so a launch is cheap here
  relative to a block of math.
* ``PREFILL_ATTN_KB`` sets the online-softmax block: every block costs one rescale of the
  accumulator, so bigger blocks mean fewer rescales but a bigger live score block. The
  comment quotes ~34 MB at the defaults for the DCU's head geometry; the same formula on
  this model is what the probe prints, because the number that matters is the one that
  will actually be allocated.

Both constants are read from the environment **at import time** (``tuning.py``), so one
process can only measure one combination. The sweep therefore re-executes this file with
each combination in ``os.environ`` and parses one machine-readable line back, which is the
only way to vary them without editing the tree between jobs.

Module: ``coffin/diag/corex_probe``. Runs on the card; nothing imports it.

Constraints:
    * Every point is the *same* prompt and the same model instance lifetime: the child
      process builds once and times, so a point costs one model load (~35 s on this card
      when the extension is already built).
    * The prompt is long enough that tiling actually happens: a 2048-token prompt with a
      512-token tile produces four tiles, and with 256-token tiles eight. At 128 tokens
      the sweep would measure noise.
    * ``mtp_prefill`` is off in every child: it is worth 1.26x (PERF_COREX.md §13.5b) and
      leaving it on would add a constant to every point without changing the ranking.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

#: ``coffin/`` is the ``sys.path`` root; this file sits two levels below it.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

MODEL = os.environ.get("ENGINE_MODEL", "")

#: Context window; covers the longest prompt below.
MAX_SEQ = 8192

#: Tokens per block in the paged pool.
BLOCK = 64

#: Prompt lengths to sweep over. 2048 is where the headline figures are quoted; 4096 and
#: 8192 make the tiling and the block count matter more, and they answer the separate
#: question of whether the prefill chunk should exceed 2048.
SIZES = (2048, 4096, 8192)

#: Query-tile widths to try. 512 is the DCU default; the rest bracket it by 4x either way.
TILES = (512, 1024)

#: Key-block widths to try. 1024 is the DCU default. 256 is dropped from the grid: it
#: quadruples the number of online-softmax rescales per tile for a working set that is
#: already only ~34 MB, so it cannot plausibly win, and each point costs a model load.
KBS = (2048, 4096)

#: The chunk used while sweeping the attention knobs, held fixed so only one thing varies.
#: 2048 is the engine default.
FIXED_CHUNK = 2048

#: Extra chunk sizes for the separate chunk-vs-prompt-length question.
CHUNKS = (2048, 4096)

#: Marker the child prints its result on, so the parent can find it in a noisy log.
RESULT_MARK = "SWEEP_RESULT "


def _tokens(n: int) -> list[int]:
    """Build a deterministic prompt of ``n`` legal token ids.

    Args:
        n: Prompt length.

    Returns:
        ``n`` ids from the arithmetic sequence the other probes use.
    """
    return [(i * 7919 + 13) % 100000 for i in range(n)]


def _child(env: dict) -> int:
    """Run one measurement point in this process, configured from ``env``.

    The model is built here rather than in the parent because ``tuning.py`` binds the two
    knobs at import, and a second import in the same process would see the first values.

    Args:
        env: Environment the point was launched with; read for the knob values only, so
            the printed result can say what it measured.

    Returns:
        ``0`` on success, ``1`` when the model directory is missing.
    """
    import torch

    import backends
    from engine_api import create_model, load_weights

    tile = int(env.get("PREFILL_ATTN_TILE", "512"))
    kb = int(env.get("PREFILL_ATTN_KB", "1024"))
    if not MODEL or not Path(MODEL).is_dir():
        print(f"  ENGINE_MODEL unset or not a directory: {MODEL!r}", flush=True)
        return 1
    print(f"  {backends.status()}  tile={tile} kb={kb}", flush=True)

    sd = load_weights(MODEL, quant=False, vision=False)
    model = create_model(MODEL, max_seq=MAX_SEQ, bench=False, weights=sd, paged=True)
    blocks = -(-MAX_SEQ // BLOCK)
    pool = model.enable_paging(num_blocks=blocks, block_size=BLOCK)
    pool.assign_table(list(range(blocks)), 0, row=model.prefill_row)
    model.mtp_prefill = False

    for n in SIZES:
        toks = _tokens(n)
        model.reset()
        model.prefill(toks, chunk=FIXED_CHUNK)
        torch.cuda.synchronize()
        best = float("inf")
        for _ in range(2):
            model.reset()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model.prefill(toks, chunk=FIXED_CHUNK)
            end.record()
            torch.cuda.synchronize()
            best = min(best, start.elapsed_time(end))
        print(f"{RESULT_MARK}{json.dumps({'kind': 'attn', 'tile': tile, 'kb': kb, 'n': n, 'ms': best})}",
              flush=True)

    for chunk in CHUNKS:
        for n in SIZES:
            if chunk > n:
                continue
            toks = _tokens(n)
            model.reset()
            model.prefill(toks, chunk=chunk)
            torch.cuda.synchronize()
            best = float("inf")
            for _ in range(2):
                model.reset()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                model.prefill(toks, chunk=chunk)
                end.record()
                torch.cuda.synchronize()
                best = min(best, start.elapsed_time(end))
            print(f"{RESULT_MARK}{json.dumps({'kind': 'chunk', 'chunk': chunk, 'n': n, 'ms': best})}",
                  flush=True)

    print("SWEEP_CHILD_DONE", flush=True)
    return 0


def _run_child(tile: int, kb: int, timeout: float = 900.0) -> list[dict]:
    """Launch one measurement point as its own process and collect its results.

    Args:
        tile: ``PREFILL_ATTN_TILE`` for this point.
        kb: ``PREFILL_ATTN_KB`` for this point.
        timeout: Wall-clock bound on the child, so one bad point cannot eat the job.

    Returns:
        The result records the child printed, in order. Empty when it failed, which the
        caller reports rather than silently averaging away.
    """
    env = dict(os.environ)
    env["PREFILL_ATTN_TILE"] = str(tile)
    env["PREFILL_ATTN_KB"] = str(kb)
    env["PYTHONUNBUFFERED"] = "1"
    try:
        proc = subprocess.run([sys.executable, "-B", "-u", __file__, "--child"],
                              env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"  tile={tile} kb={kb} TIMED OUT after {timeout:.0f}s", flush=True)
        return []
    if proc.returncode != 0:
        tail = (proc.stdout or "").strip().splitlines()[-6:]
        print(f"  tile={tile} kb={kb} FAILED rc={proc.returncode}", flush=True)
        for line in tail:
            print(f"    {line}", flush=True)
        return []
    out = []
    for line in (proc.stdout or "").splitlines():
        if line.startswith(RESULT_MARK):
            try:
                out.append(json.loads(line[len(RESULT_MARK):]))
            except json.JSONDecodeError:
                continue
    return out


def _report(rows: list[dict]) -> None:
    """Print the attention sweep and the chunk sweep as tables.

    Args:
        rows: Every result record collected from the children.
    """
    attn = [r for r in rows if r.get("kind") == "attn"]
    chunk = [r for r in rows if r.get("kind") == "chunk"]
    sizes = sorted({r["n"] for r in attn})
    combos = sorted({(r["tile"], r["kb"]) for r in attn})

    print(f"\n{'=' * 78}\nattention tile x key-block, chunk={FIXED_CHUNK}\n{'=' * 78}",
          flush=True)
    print(f"  {'tile':>6}{'kb':>7}" + "".join(f"{f'n={n} ms':>12}" for n in sizes)
          + f"{'best tok/s':>12}", flush=True)
    best = None
    for tile, kb in combos:
        cells = []
        tps = []
        for n in sizes:
            hit = next((r for r in attn if r["tile"] == tile and r["kb"] == kb
                        and r["n"] == n), None)
            if hit is None:
                cells.append(f"{'-':>12}")
                tps.append(None)
            else:
                cells.append(f"{hit['ms']:>12.1f}")
                tps.append(n / hit["ms"] * 1e3)
        # Rank on the largest prompt: that is where the tile decides the most work, and
        # where a small-prompt win from a bigger block would otherwise dominate.
        score = tps[-1] if tps and tps[-1] else 0.0
        line = f"  {tile:>6}{kb:>7}" + "".join(cells) + f"{score:>12.0f}"
        print(line, flush=True)
        if best is None or score > best[0]:
            best = (score, tile, kb)
    if best:
        print(f"\n  best at n={sizes[-1]}: tile={best[1]} kb={best[2]} "
              f"{best[0]:.0f} tok/s", flush=True)

    print(f"\n{'=' * 78}\nchunk vs prompt length (tile/kb at their swept best)\n{'=' * 78}",
          flush=True)
    print(f"  {'n':>6}{'chunk':>7}{'chunks':>8}{'ms':>10}{'tok/s':>9}", flush=True)
    for n in sorted({r["n"] for r in chunk}):
        for c in sorted({r["chunk"] for r in chunk if r["n"] == n}):
            hit = next(r for r in chunk if r["n"] == n and r["chunk"] == c)
            print(f"  {n:>6}{c:>7}{-(-n // c):>8}{hit['ms']:>10.1f}"
                  f"{n / hit['ms'] * 1e3:>9.0f}", flush=True)


def main() -> int:
    """Run the child for every combination, or the child itself under ``--child``.

    Returns:
        ``0`` on success, ``1`` when the model directory is missing.
    """
    if "--child" in sys.argv:
        return _child(dict(os.environ))

    print(f"  model : {MODEL}", flush=True)
    if not MODEL or not Path(MODEL).is_dir():
        print(f"  ENGINE_MODEL unset or not a directory: {MODEL!r}", flush=True)
        return 1
    print(f"  sizes : {SIZES}  chunks {CHUNKS}", flush=True)
    print(f"  tiles : {TILES}  key blocks {KBS}", flush=True)
    print(f"  points: {len(TILES) * len(KBS)} attention combos, "
          f"one child process each", flush=True)

    rows: list[dict] = []
    for tile in TILES:
        for kb in KBS:
            print(f"\n--- tile={tile} kb={kb} ---", flush=True)
            got = _run_child(tile, kb)
            print(f"  {len(got)} result rows", flush=True)
            rows.extend(got)

    if not rows:
        print("\nno measurements: every child failed", flush=True)
        return 1
    _report(rows)
    print("\nPREFILL_SWEEP_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
