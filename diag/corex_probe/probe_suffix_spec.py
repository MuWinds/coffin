"""Can suffix (prompt-lookup) speculation pay on a card where capture is forbidden?

``LLMEngine.build`` refuses ``--spec-k`` when capture is off, because an MTP verify step is a
captured replay by construction. CoreX reports ``caps.graph == 0`` by policy -- the SCOW
driver's replay is not trusted and must stay unused -- so on this card speculation is off and
the only lever left is *what* the drafter is. Suffix decoding replaces the MTP head with a
lookup in the request's own token history: a draft costs a dict lookup, no weight read and no
launch, and the target model still verifies it, so the emitted tokens cannot change.

Whether that pays is decided by one ratio, not by the drafter:

    a step costs ``step_ms`` and emits one token;
    a verify costs ``verify_ms`` and emits ``1 + accepted`` tokens.

Suffix novelty is only the numerator. ``verify_ms / step_ms`` is the denominator, and it is a
property of the *weight path*: the int8 entry (``_gemv_rows`` in the CoreX backend) launches one
GEMV **per row** because no small-M int8 kernel exists, so a ``c``-token verify costs roughly
``c`` decode steps and no drafter can ever recover it. The bf16 entry has ``gemv_kn_m``, which
reads the weight once for all rows. So this probe measures the ratio on both weight paths,
because that difference is the whole answer.

Two independent checks keep the numbers honest:

* **The oracle control** stages the *true* next tokens as candidates. Every draft must then be
  accepted and the emitted stream must equal plain greedy token for token. If it does not, the
  verify path disagrees with eager decode and no suffix number from that configuration means
  anything -- that is a measurement bug, not a speed result.
* **The offline ceiling** plays the drafter over the target's own greedy output, so the
  drafter's quality is reported separately from the verify path's cost.

Module: ``coffin/diag/corex_probe``. Runs on the card through the SCOW client; nothing imports
it.

Constraints:
    * Eager only. Nothing here starts a capture; the decode and verify bodies are the ones the
      engine takes when ``caps.graph`` is 0.
    * Greedy only, and a draft is committed only after the target reproduces it, so a differing
      token stream is reported as a bug rather than sampled around.
    * One prompt per phase and no radix cache: the question is the step/verify ratio, and a
      prefix cache would only change the prefill.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import zlib
from pathlib import Path

#: ``coffin/`` is the ``sys.path`` root; this file sits two levels below it.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

MODEL = os.environ.get("ENGINE_MODEL", "")

#: Tokens per KV block; matches the engine's default so the block math below is the same one
#: the scheduler does.
BLOCK = 64

#: Context window for the pool. The prompts are ~100 tokens and the phases generate at most
#: a few hundred more, so a small window is the right size: the dense KV caches the paged pool
#: replaces are sized from it, and at 4096 the bf16 weights plus those caches do not fit beside
#: each other on a 32 GiB card.
MAX_SEQ = 1024

#: Tokens generated per phase. Long enough for the drafter's index to have material, short
#: enough that the whole matrix of configurations still fits in one job.
N_GEN = 64

#: Draft depths the offline sweep reports. The verify width is ``K + 1`` and the rollback ring
#: is sized by ``CKPT`` (8), so this stays at or below 7.
K_SWEEP = (1, 2, 3, 4, 6)

#: Draft depth of the on-card loop: the widest depth the sweep is expected to justify without
#: making the verify wide enough to be dismissed on cost alone.
K_LOOP = 4

#: Texts the acceptance is measured on. They are deliberately different registers: suffix
#: decoding is a repetition detector, so prose and code must not be averaged into one number
#: that describes neither. The third is the control -- if the mechanism is wired right it has to
#: score near 1.0, and if it does not, the first two mean nothing.
CORPUS = (
    ("prose",
     "The decode step of a large language model is bound by the memory bandwidth needed to "
     "read its weights, not by arithmetic. Every step reads the whole parameter set once, so "
     "the time per token barely depends on the batch size: two sequences decode almost as "
     "fast as one, and eight cost a fraction more. That is why continuous batching is worth "
     "so much on a serving system. The kernels themselves are simple matrix-vector products; "
     "what makes a step slow is the number of launches and the size of the reads."),
    ("code",
     "def match_prefix(self, tokens, require_states=False):\n"
     "    node = self.root\n"
     "    matched = 0\n"
     "    for tok in tokens:\n"
     "        if tok not in node.children:\n"
     "            return matched, node.node_id\n"
     "        node = node.children[tok]\n"
     "        matched += 1\n"
     "    return matched, node.node_id\n"),
    ("repeat",
     "step 1: read the weights. step 2: apply the weights. step 3: write the cache. "
     "step 1: read the weights. step 2: apply the weights. step 3: write the cache. "
     "step 1: read the weights. step 2: apply the weights. step 3: write the cache."),
)


def _sep(title: str) -> None:
    """Print a section header.

    Args:
        title: Text of the banner.

    Returns:
        Nothing. Flushed, because a job log is only readable once the run ends.
    """
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)


def _crc(ids: list[int]) -> int:
    """Fingerprint a token stream so two runs can be compared inside a log.

    Args:
        ids: Token ids in emission order.

    Returns:
        CRC32 of their decimal spelling.
    """
    return zlib.crc32(",".join(str(i) for i in ids).encode())


def _first_diff(got: list[int], want: list[int]) -> str:
    """Describe the first position where two token streams disagree.

    Args:
        got: Stream produced by the path under test.
        want: Stream the path must reproduce.

    Returns:
        A one-line description, or ``"none"`` when the streams are equal.
    """
    for i, (a, b) in enumerate(zip(got, want)):
        if a != b:
            return f"token {i}: {a} != {b}"
    return "none" if len(got) == len(want) else f"length {len(got)} != {len(want)}"


def _prefill(model, prompt: list[int], blocks: list[int]):
    """Prefill one prompt on the staging row and hand its state to decode row 0.

    The staging row is the only row a prefill may write, and the recurrent state has to be
    moved to the row that will decode: a decode row that never saw the prompt's delta state
    would generate from an empty recurrence.

    Args:
        model: Loaded paged model with batching enabled.
        prompt: Prompt token ids.
        blocks: Block table covering the prompt plus every generated token.

    Returns:
        Logits of the last prompt position.
    """
    from ops import runtime

    stage = model.prefill_row
    model.reset()
    model.pos[stage:stage + 1].fill_(0)
    model.pool.assign_table(blocks, 0, row=stage)
    logits = model.prefill(prompt, chunk=2048)
    model.move_linear_states(stage, 0)
    model.pool.assign_table(blocks, 0, row=0)
    runtime.synchronize()
    return logits


def _stage(model, pending: int, pos: int, blocks: list[int]) -> None:
    """Write one row's next token and position into the decode buffers.

    Args:
        model: Loaded paged model with batching enabled.
        pending: Token the row is about to feed, already emitted by the caller.
        pos: Its absolute position, which addresses both the KV slot and the window.
        blocks: The row's block table.

    Returns:
        Nothing. The batched forward reads these buffers rather than taking arguments.
    """
    model.set_rows([int(pending)], [int(pos)], [blocks], row_indices=[0])


def _plain_step(model, pending: int, pos: int, blocks: list[int]) -> int:
    """Run one plain batched decode step and return its greedy token.

    Args:
        model: Loaded paged model with batching enabled.
        pending: Token to feed at ``pos``.
        pos: Absolute position of ``pending``.
        blocks: The row's block table.

    Returns:
        The argmax token of the next position.
    """
    _stage(model, pending, pos, blocks)
    logits = model.decode_batch_eager(1, 1)
    return int(logits[0].argmax())


def _verify_step(model, pending: int, cands: list[int], pos: int, blocks: list[int]):
    """Verify candidate continuations in one verify forward and rewind to the accepted prefix.

    Mirrors what ``spec_step_batch`` does with the MTP drafter, minus the drafter: the verify
    buffers are staged from the candidates instead of from ``h_prev_b``, and the verify body is
    replayed through whatever capture wrapper it has. The width is fixed by the buffers and the
    rollback ring, so a shorter candidate list is padded; the padding is never accepted because
    the accept loop stops at the end of the real candidates.

    Args:
        model: Loaded paged model with the speculative buffers enabled.
        pending: Token at ``pos``, already emitted.
        cands: Candidate continuations, at most ``spec_width - 1`` tokens.
        pos: Absolute position of ``pending``.
        blocks: The row's block table.

    Returns:
        ``(accepted, next_token)``: how many of ``cands`` the target itself would have
        produced, and the token to emit after them.
    """
    import torch

    from models.qwen35.linear_attn import LinearAttn

    width = model.spec_width
    k = width - 1
    row = [int(pending)] + [int(t) for t in cands[:k]]
    row += [0] * (width - len(row))
    _stage(model, pending, pos, blocks)
    model.verify_toks_b[0, :width].copy_(torch.tensor(row, dtype=torch.long))
    model.verify_pos_b[0, :width].copy_(
        torch.arange(int(pos), int(pos) + width, dtype=torch.long))
    # Private on purpose: the public entry stages the MTP drafter from ``h_prev_b``, and with a
    # lookup drafter the staging above is the whole difference. Replaying the same body keeps
    # the measurement on the shipped verify implementation rather than a copy of it.
    model._verify_batch_graphs[1].replay()  # noqa: SLF001 - see above

    pred = model.verify_logits_b[0, :k].argmax(-1).tolist()
    accepted = 0
    while accepted < len(cands) and pred[accepted] == int(cands[accepted]):
        accepted += 1
    # Slot ``j`` holds the recurrent state right after verify token ``j``, and the accepted
    # prefix covers tokens ``0..accepted``, so the state to keep is slot ``accepted``. KV past
    # it stays in the pool and is overwritten by the next step's write.
    for layer in model.layers:
        mix = layer.mix
        if isinstance(mix, LinearAttn):
            mix.restore_batch(0, accepted)
    return accepted, int(model.verify_logits_b[0, accepted].argmax())


def _greedy(model, prompt: list[int], blocks: list[int],
            n_gen: int) -> tuple[list[int], float]:
    """Generate ``n_gen`` tokens one plain batched step at a time.

    Runs twice: the first pass pays the one-off kernel and allocator costs that a decode
    benchmark must not attribute to the step, and the second pass is the one reported.

    Args:
        model: Loaded paged model with batching enabled.
        prompt: Prompt token ids.
        blocks: Block table covering prompt plus generated tokens.
        n_gen: Tokens to generate.

    Returns:
        ``(ids, ms_per_token)``: the emitted ids and the second pass's speed.
    """
    from ops import runtime

    ids: list[int] = []
    ms_per_token = 0.0
    for _attempt in range(2):
        logits = _prefill(model, prompt, blocks)
        pending = int(logits.argmax())
        out = [pending]
        pos = len(prompt)
        runtime.synchronize()
        t0 = time.perf_counter()
        while len(out) < n_gen:
            pending = _plain_step(model, pending, pos, blocks)
            out.append(pending)
            pos += 1
        runtime.synchronize()
        ids = out
        ms_per_token = (time.perf_counter() - t0) * 1e3 / max(1, len(out) - 1)
    return ids, ms_per_token


def _suffix(model, prompt: list[int], blocks: list[int], n_gen: int, k: int,
            max_match: int):
    """Generate ``n_gen`` tokens with suffix drafts verified by the target.

    Args:
        model: Loaded paged model with the speculative buffers enabled.
        prompt: Prompt token ids.
        blocks: Block table covering prompt plus generated tokens.
        n_gen: Tokens to generate.
        k: Drafts offered per step, equal to ``spec_width - 1``.
        max_match: Longest suffix the lookup compares.

    Returns:
        ``(ids, ms_per_token, offered, accepted, verify_steps, plain_steps)``, timed on the
        second (warmed) pass.
    """
    from ops import runtime
    from ops.suffix import SuffixDrafter

    ids: list[int] = []
    ms_per_token = 0.0
    offered = accepted = verify_steps = plain_steps = 0
    for _attempt in range(2):
        drafter = SuffixDrafter(max_match=max_match)
        logits = _prefill(model, prompt, blocks)
        pending = int(logits.argmax())
        out = [pending]
        pos = len(prompt)
        # The drafter's history is the model's own stream, prompt included: feeding drafts back
        # would make the index self-fulfilling and the acceptance number meaningless.
        drafter.reset(list(prompt))
        drafter.observe([pending])
        offered = accepted = verify_steps = plain_steps = 0
        runtime.synchronize()
        t0 = time.perf_counter()
        # A step that would run past the requested length is clamped: overrunning keeps emitting
        # valid tokens but makes the stream longer than the baseline, and a length difference
        # then hides whatever the comparison was supposed to catch.
        while len(out) < n_gen:
            cands = drafter.draft(min(k, n_gen - len(out) - 1))
            if not cands:
                nxt = _plain_step(model, pending, pos, blocks)
                plain_steps += 1
                out.append(nxt)
                drafter.observe([nxt])
                pos += 1
                pending = nxt
                continue
            offered += len(cands)
            hit, nxt = _verify_step(model, pending, cands, pos, blocks)
            accepted += hit
            verify_steps += 1
            emitted = list(cands[:hit]) + [nxt]
            out.extend(emitted)
            drafter.observe(emitted)
            pos += hit + 1
            pending = nxt
        runtime.synchronize()
        del out[n_gen:]
        ids = out
        ms_per_token = (time.perf_counter() - t0) * 1e3 / max(1, len(out) - 1)
    return ids, ms_per_token, offered, accepted, verify_steps, plain_steps


def _oracle(model, prompt: list[int], blocks: list[int], plain: list[int],
            k: int) -> tuple[list[int], float, int, int]:
    """Run the verify loop with the *true* next tokens as candidates.

    The control that separates "the drafter is weak" from "the verify path is wrong". With
    perfect candidates every draft must be accepted and the stream must equal plain greedy token
    for token; anything else is a staging, rollback or kernel bug, and would make the measured
    suffix numbers meaningless.

    Args:
        model: Loaded paged model with the speculative buffers enabled.
        prompt: Prompt token ids.
        blocks: Block table covering prompt plus generated tokens.
        plain: The plain greedy stream, prompt included; the source of the candidates.
        k: Drafts offered per step, equal to ``spec_width - 1``.

    Returns:
        ``(ids, ms_per_token, verify_steps, accepted)`` for the oracle run.
    """
    from ops import runtime

    logits = _prefill(model, prompt, blocks)
    pending = int(logits.argmax())
    out = [pending]
    pos = len(prompt)
    verify_steps = accepted = 0
    runtime.synchronize()
    t0 = time.perf_counter()
    while len(out) < len(plain):
        t = len(out) - 1
        # Clamped like the suffix loop: offering candidates past the baseline end makes the
        # stream longer than the baseline, and a length mismatch would then mask a real
        # disagreement instead of reporting it.
        cands = list(plain[t + 1:t + 1 + min(k, len(plain) - len(out))])
        if not cands:
            break
        hit, nxt = _verify_step(model, pending, cands, pos, blocks)
        accepted += hit
        verify_steps += 1
        out.extend(list(cands[:hit]) + [nxt])
        pos += hit + 1
        pending = nxt
    runtime.synchronize()
    del out[len(plain):]
    ms = (time.perf_counter() - t0) * 1e3 / max(1, len(out) - 1)
    return out, ms, verify_steps, accepted


def _offline_acceptance(drafter_cls, prompt_len: int,
                        ids: list[int]) -> dict[int, tuple[int, int]]:
    """Count what a suffix drafter would have offered and earned on a known token stream.

    Played over the target's own greedy output, so it isolates the drafter's quality from the
    verify path: this is the ceiling, the on-card loop below is the reality.

    The history handed to the drafter ends at the *pending* token ``ids[t]``, because that is
    what a real step has -- the drafter predicts what follows the token the model just emitted.
    Feeding history that stops one token short would compare each draft against the wrong
    position and report zero acceptance for a drafter that is working.

    Args:
        drafter_cls: Drafter class to instantiate.
        prompt_len: How much of ``ids`` is prompt; drafting starts after it.
        ids: The complete token stream (prompt plus generated tokens).

    Returns:
        ``{k: (accepted, offered)}`` for every depth in ``K_SWEEP``.
    """
    counts: dict[int, list[int]] = {k: [0, 0] for k in K_SWEEP}
    for k in K_SWEEP:
        drafter = drafter_cls(max_match=4)
        drafter.reset(ids[:prompt_len])
        for t in range(prompt_len, len(ids) - 1):
            drafter.observe([ids[t]])
            cands = drafter.draft(k)
            counts[k][1] += len(cands)
            for i, token in enumerate(cands):
                if t + 1 + i >= len(ids) or ids[t + 1 + i] != token:
                    break
                counts[k][0] += 1
    return {k: (v[0], v[1]) for k, v in counts.items()}


def _verify_cost(model) -> float:
    """Time one ``c``-token verify forward on the buffers the last real step left behind.

    Replaying with stale staging measures the body's cost exactly: the entry reads buffers it
    does not own and writes the same fixed outputs, which is what a real step pays.

    Args:
        model: Loaded paged model with the speculative buffers enabled.

    Returns:
        Mean milliseconds per verify forward.
    """
    from ops import runtime

    body = model._verify_batch_graphs[1]  # noqa: SLF001 - the eager verify entry
    for _ in range(3):
        body.replay()
    runtime.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        body.replay()
    runtime.synchronize()
    return (time.perf_counter() - t0) * 1e2


def _run_weight_path(label: str, quant: bool, prompts: dict[str, list[int]],
                     gen: int) -> list[tuple]:
    """Measure every configuration for one weight path.

    Args:
        label: Name of the weight path, for the log.
        quant: Load the projections as int8 when true.
        prompts: ``{corpus label: prompt ids}``.
        gen: Tokens to generate per phase.

    Returns:
        One row per (weight path, verify mode): the numbers the verdict is built from.
    """
    import torch

    import backends
    from engine_api import create_model, load_weights
    from ops.suffix import SuffixDrafter

    _sep(f"weight path {label}  (quant={quant})")
    t0 = time.perf_counter()
    sd = load_weights(MODEL, quant=quant, vision=False)
    model = create_model(MODEL, max_seq=MAX_SEQ, bench=False, weights=sd, paged=True)
    blocks_need = -(-MAX_SEQ // BLOCK)
    pool = model.enable_paging(num_blocks=blocks_need + 8, block_size=BLOCK)
    # Two rows: one decode row plus the prefill staging row the engine always keeps free.
    model.enable_batching(2)
    blocks = [pool.alloc() for _ in range(blocks_need)]
    print(f"  loaded in {time.perf_counter() - t0:.1f} s  mtp={model.mtp is not None}  "
          f"decode_capacity={model.decode_capacity}  "
          f"vram={torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)

    plain: dict[str, list[int]] = {}
    plain_ms: dict[str, float] = {}
    for corpus, prompt in prompts.items():
        ids, ms = _greedy(model, prompt, blocks, gen)
        plain[corpus], plain_ms[corpus] = ids, ms
        print(f"  plain step {corpus:<7} {ms:>6.2f} ms/token {1e3 / ms:>6.2f} tok/s  "
              f"crc={_crc(ids)}  head={ids[:6]}", flush=True)
    step_ms = sum(plain_ms.values()) / len(plain_ms)

    from ops.suffix import SuffixDrafter

    print("\n  --- suffix acceptance over the model's own output (drafter ceiling) ---",
          flush=True)
    for corpus, prompt in prompts.items():
        counts = _offline_acceptance(SuffixDrafter, len(prompt), prompt + plain[corpus])
        line = "  ".join(f"K={k}: {a / o:.2f} ({a}/{o})" if o else f"K={k}: none"
                         for k, (a, o) in counts.items())
        print(f"    {corpus:<7} {line}", flush=True)
    print("    (target text is the model's own greedy stream above; a degenerate repetition"
          "\n     scores high here, and that number describes the output, not the idea.)",
          flush=True)

    if model.mtp is None:
        print("  no MTP head: the verify buffers cannot be enabled, stopping this path",
              flush=True)
        del model, sd, pool, blocks
        torch.cuda.empty_cache()
        return []

    rows: list[tuple] = []
    for mode, depth in [(m, d) for d in (1, 2, 4) for m in ("token",)] + [("once", K_LOOP)]:
        t0 = time.perf_counter()
        model.enable_spec_batch(depth, buckets=(1,), verify_mode=mode)
        v_ms = _verify_cost(model)
        print(f"\n  verify_mode={model.verify_mode:<5}  K={depth} width={model.spec_width}  "
              f"setup {time.perf_counter() - t0:.1f} s  cost {v_ms:.2f} ms = "
              f"{v_ms / step_ms:.2f}x a plain step", flush=True)

        print(f"  --- oracle control (perfect drafts; must reproduce greedy exactly) ---",
              flush=True)
        oracle_ok = True
        for corpus, prompt in prompts.items():
            ids, ms, vsteps, acc = _oracle(model, prompt, blocks, plain[corpus], depth)
            same = ids == plain[corpus]
            oracle_ok &= same
            print(f"    {corpus:<7} {ms:>6.2f} ms/token ({plain_ms[corpus] / ms:>4.2f}x)  "
                  f"verify={vsteps} accepted={acc}  greedy_equal={same}  "
                  f"diff={_first_diff(ids, plain[corpus])}", flush=True)

        speedups = []
        rates = []
        for corpus, prompt in prompts.items():
            ids, ms, offered, accepted, vsteps, psteps = _suffix(
                model, prompt, blocks, gen, depth, 4)
            rate = accepted / offered if offered else 0.0
            rates.append(rate)
            speedups.append(plain_ms[corpus] / ms)
            print(f"    {corpus:<7} {ms:>6.2f} ms/token ({plain_ms[corpus] / ms:>4.2f}x)  "
                  f"accept {rate:.2f} ({accepted}/{offered})  verify={vsteps} plain={psteps}  "
                  f"greedy_equal={ids == plain[corpus]}  "
                  f"diff={_first_diff(ids, plain[corpus])}", flush=True)
        rows.append((label, f"{model.verify_mode}/K={depth}", step_ms, v_ms, v_ms / step_ms,
                     sum(rates) / len(rates), sum(speedups) / len(speedups), oracle_ok))

    del model, sd, pool, blocks
    torch.cuda.empty_cache()
    return rows


def main() -> int:
    """Measure the step/verify ratio and the suffix trade on both weight paths.

    Returns:
        ``0`` on success, ``1`` when the checkpoint directory is missing.
    """
    import backends
    from tokenizers import Tokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=int, default=N_GEN, help="tokens per phase")
    ap.add_argument("--paths", default="int8",
                    help="comma-separated weight paths to measure; bf16 needs ~32 GiB on its "
                         "own and OOMs next to the dense caches a 1024-token pool replaces")
    args = ap.parse_args()

    print(f"  backend: {backends.status()}", flush=True)
    print(f"  model  : {MODEL}", flush=True)
    if not MODEL or not Path(MODEL).is_dir():
        print(f"  ENGINE_MODEL unset or not a directory: {MODEL!r}", flush=True)
        return 1

    tok = Tokenizer.from_file(str(Path(MODEL) / "tokenizer.json"))
    prompts = {label: tok.encode(text).ids for label, text in CORPUS}
    print("  prompts: " + ", ".join(f"{k}={len(v)}" for k, v in prompts.items()), flush=True)

    all_rows: list[tuple] = []
    for label in [p.strip() for p in args.paths.split(",") if p.strip()]:
        all_rows += _run_weight_path(label, label == "int8", prompts, args.gen)

    _sep("verdict")
    print(f"  {'weights':<7}{'verify':<8}{'step ms':>9}{'verify ms':>11}{'ratio':>8}"
          f"{'accept':>8}{'speedup':>9}  oracle", flush=True)
    for weights, mode, step_ms, v_ms, ratio, rate, speedup, oracle_ok in all_rows:
        print(f"  {weights:<7}{mode:<8}{step_ms:>9.2f}{v_ms:>11.2f}{ratio:>8.2f}"
              f"{rate:>8.2f}{speedup:>8.2f}x  {'exact' if oracle_ok else 'MISMATCH'}",
              flush=True)
    print("\n  A verify is worth running only when ratio < 1 + accepted. Suffix acceptance is"
          "\n  the ceiling measured in section 2; anything at or above 1.0x here means the"
          "\n  weight path cannot feed a verify cheaper than c plain steps.", flush=True)
    print("\nSUFFIX_SPEC_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
