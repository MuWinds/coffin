"""Determinism inside ONE model instance (the real serving shape).

Answers whether the same prompt on the same loaded model yields the same logits,
which is the property a serving deployment needs. The serving process builds the
model exactly once and then prefills many requests, so the determinism that
matters is: same prompt, same model instance, repeated prefill -> identical
logits. Building two models in one process is a different (and much less stable)
question, because the second build allocates against an allocator that already
holds the first build's caching artifacts.

Read the ``PASS``/``FAIL`` on each line: ``rel`` is the relative L2 difference
against the first run and must stay at or below 1e-6, and ``argmax`` must agree,
because a bit-level drift that does not move the argmax is still a drift but a
different argmax changes sampled text. ``--mode single`` is the serving question;
``--mode multi`` rebuilds per run and is expected to be less stable — treat its
failures as allocator sensitivity, not as a determinism bug in the model.

Usage (container, from the engine dir):
    flock /root/private_data/gpu.lock -c 'python3 diag/det_probe.py --mode single'
    flock /root/private_data/gpu.lock -c 'python3 diag/det_probe.py --mode single --baseline b.pt'

``--write-baseline`` records a run and a later ``--baseline`` compares against it
at the looser 1e-3 tolerance, which is what a cross-session regression check can
hold.

Module: ``coffin/diag``. Standalone probe: imports ``engine_api`` plus
transformers, and nothing under ``coffin/`` imports it.

Constraints:
    * ``engine_api`` is imported only after ``sys.path`` is fixed up inside
      ``main``, because this script runs from ``diag/`` while the engine's own
      modules use absolute imports rooted at ``coffin/``.
    * ``flock`` is required: the comparison is a bit-level difference, and a
      co-tenant's load does not change correctness but does change timings, so
      the lock keeps a slow run distinguishable from a wrong one.
    * The prompts mix English, code and Chinese on purpose; the tokenizer's
      multi-byte paths are where a prefill bug usually appears first.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import torch

MD = "/root/private_data/models/Qwen3.5-4B"
PROMPTS = [
    "The capital of France is",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n",
    "用户：北京天气怎么样？\n助手：",
    "1, 1, 2, 3, 5, 8, 13,",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MD)
    ap.add_argument("--max-seq", type=int, default=4096)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--write-baseline", default=None)
    ap.add_argument("--mode", choices=("single", "multi"), default="single",
                    help="single = one model, repeated prefill (serving shape); "
                         "multi = rebuild the model per run (allocator-sensitive)")
    args = ap.parse_args()

    root = pathlib.Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    import engine_api
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    ids = [tok.encode(p) for p in PROMPTS]
    print("prompt lens", [len(x) for x in ids], flush=True)

    if args.mode == "single":
        model = engine_api.create_model(args.model, max_seq=args.max_seq,
                                        bench=False, paged=True)
        model.enable_paging(-(-args.max_seq // 64), 64)
        model.capture()
        base = None
        for r in range(args.repeat):
            for i, seq in enumerate(ids):
                model.reset()
                lg = model.prefill(seq).float().cpu()
                if base is None:
                    base = {}
                if i not in base:
                    base[i] = lg
                else:
                    rel = ((lg - base[i]).norm() / base[i].norm()).item()
                    same = int(lg.argmax()) == int(base[i].argmax())
                    print(f"repeat{r} prompt{i} rel {rel:.2e} "
                          f"argmax {int(lg.argmax())}/{int(base[i].argmax())} "
                          f"{'PASS' if rel <= 1e-6 and same else 'FAIL'}", flush=True)
        model.close()
        torch.cuda.empty_cache()
        first = [base[i] for i in sorted(base)]
    else:
        first = []
        for r in range(args.repeat):
            model = engine_api.create_model(args.model, max_seq=args.max_seq,
                                            bench=False, paged=True)
            model.enable_paging(-(-args.max_seq // 64), 64)
            model.capture()
            lg = []
            for seq in ids:
                model.reset()
                lg.append(model.prefill(seq).float().cpu())
            model.close()
            del model
            torch.cuda.empty_cache()
            print(f"build {r} done", flush=True)
            first.append(lg)
        base = first[0]
        for r in range(1, len(first)):
            for i, (a, b) in enumerate(zip(first[r], base)):
                rel = ((a - b).norm() / b.norm()).item()
                same = int(a.argmax()) == int(b.argmax())
                print(f"rebuild{r} prompt{i} rel {rel:.2e} "
                      f"argmax {int(a.argmax())}/{int(b.argmax())} "
                      f"{'PASS' if rel <= 1e-6 and same else 'FAIL'}")

    if args.write_baseline:
        torch.save(base if args.mode == "single" else first[0], args.write_baseline)
        print("baseline written", args.write_baseline)

    if args.baseline:
        ref = torch.load(args.baseline, map_location="cpu")
        if isinstance(ref, list):
            pairs = list(zip(base, ref))
        else:
            pairs = [(base[i], ref[i]) for i in sorted(ref)]
        for i, (got, exp) in enumerate(pairs):
            rel = ((got - exp).norm() / exp.norm()).item()
            same = int(got.argmax()) == int(exp.argmax())
            print(f"baseline prompt{i} rel {rel:.2e} argmax "
                  f"{int(got.argmax())}/{int(exp.argmax())} "
                  f"{'PASS' if same and rel <= 1e-3 else 'FAIL'}")
    print("DET_DONE")


if __name__ == "__main__":
    main()
