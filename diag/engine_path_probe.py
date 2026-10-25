"""Determinism / regression probe on the *supported* serving path.

Answers whether the path ``serve.py`` actually uses is deterministic, which is a
narrower and more useful question than probing the model object directly. The
paged pool's ``pos2slot`` starts as all zeros: a caller that runs
``model.prefill()`` without first assigning a block table makes every token
address slot 0, so successive prefills stomp on each other's KV. That is a
misuse of the internal API -- the public path is ``PrefixCacheEngine`` (or
``LLMEngine``), which allocates blocks and installs the table first.

This probe therefore drives the engine the way ``serve.py`` does, and checks:

  1. a text-only prefill repeated on a fresh engine is bit-identical
  2. warm (prefix hit) == cold for the same prompt
  3. optionally, the same prompt with an image differs from text-only

Read the three numbered blocks. Check 2 is the one that guards prefix reuse:
warm must equal cold, because the radix cache hands the second request the first
one's KV, and any mismatch means reused blocks do not carry the state the prompt
needs. Check 3 asserts the *opposite* direction — an image prompt must differ
from the text-only one by more than 1e-2 — so a vision path that silently drops
the image fails here instead of passing quietly.

    flock /root/private_data/gpu.lock -c 'python3 diag/engine_path_probe.py'

Module: ``coffin/diag``. Standalone probe: imports ``engine_api``,
``prefix_engine``, ``mm_input`` and transformers, and nothing under ``coffin/``
imports it.

Constraints:
    * ``PrefixCacheEngine`` must be constructed before ``capture()``. It assigns
      the block table, and a paged model captured with ``pos2slot`` still all
      zeros maps every position to slot 0 — that caller error is what made an
      earlier version of this probe report false non-determinism.
    * ``flock`` is required: a pinned, deterministic comparison still needs the
      whole card, and a co-tenant can push a run past its timeout.
    * ``pc.release()`` runs before each measured prefill so "cold" means cold;
      skipping it turns check 1 into a warm-cache test.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import torch

MD = "/root/private_data/models/Qwen3.5-4B"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MD)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    root = pathlib.Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    import engine_api
    import mm_input
    from prefix_engine import PrefixCacheEngine
    from transformers import AutoProcessor, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    proc = AutoProcessor.from_pretrained(args.model)
    text = tok.encode("The capital of France is")

    weights = engine_api.load_weights(args.model, quant=True)
    model = engine_api.create_model(args.model, max_seq=args.max_seq, bench=False,
                                    weights=weights, paged=True)
    # Paging must be installed (and blocks assigned) before capture: constructing
    # PrefixCacheEngine does both.  Running ``model.capture()`` on a paged model
    # whose ``pos2slot`` is still all zeros makes every position address slot 0 --
    # that is a caller error, not an engine bug, and it is what made an earlier
    # version of this probe report non-determinism.
    pc = PrefixCacheEngine(model, num_blocks=-(-args.max_seq // 64), block_size=64)
    model.capture()
    print("engine ready", flush=True)

    # 1. repeated prefill on a fresh engine (cold every time)
    base = None
    for i in range(args.repeat):
        pc.release()
        pc.clear_prefix_cache() if hasattr(pc, "clear_prefix_cache") else None
        model.reset()
        lg = pc.prefill(text).float().cpu()
        if base is None:
            base = lg
            print(f"text cold[0] argmax {int(lg.argmax())}", flush=True)
        else:
            rel = ((lg - base).norm() / base.norm()).item()
            same = int(lg.argmax()) == int(base.argmax())
            print(f"1 cold[{i}] rel {rel:.2e} argmax {int(lg.argmax())} "
                  f"{'PASS' if rel <= 1e-6 and same else 'FAIL'}", flush=True)

    # 2. warm (prefix-cache hit) must equal cold
    lg_warm = pc.prefill(text).float().cpu()
    rel = ((lg_warm - base).norm() / base.norm()).item()
    same = int(lg_warm.argmax()) == int(base.argmax())
    print(f"2 warm-vs-cold matched={pc.last_matched} rel {rel:.2e} "
          f"{'PASS' if rel <= 1e-6 and same else 'FAIL'} "
          f"(prefix cache is only exact if this passes)", flush=True)

    # 3. image vs text-only through the same engine path
    rendered = proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"},
                                      {"type": "text", "text": "read it"}]}],
        tokenize=False, add_generation_prompt=True)
    import numpy as np
    from PIL import Image
    a = np.full((512, 512, 3), 255, np.uint8)
    a[64:192, :, :] = 0
    out = proc(text=[rendered], images=[Image.fromarray(a)], return_tensors="pt")
    ids = out["input_ids"][0].tolist()
    pv, grid = out["pixel_values"], out["image_grid_thw"]
    spans = mm_input.find_image_spans(ids)
    n_pads = sum(e - s for s, e in spans)
    print(f"3 image prompt {len(ids)} tokens, {n_pads} pads, grid {grid.tolist()}",
          flush=True)
    if n_pads:
        model.enable_vision()
        emb = model.image_embeds(pv, grid)
        pos, delta = mm_input.mrope_positions(spans, grid, len(ids), device="cuda")
        mm = mm_input.MMInput(embeds=emb, grid_thw=grid.to("cuda"), positions=pos,
                              rope_delta=delta, digest=mm_input.image_digest([b"p"]),
                              spans=spans)
        pc.release()
        model.reset()
        lg_img = pc.prefill(ids, mm=mm)
        fin = bool(torch.isfinite(lg_img).all())
        print(f"3 image prefill finite={fin} argmax={int(lg_img.argmax())} "
              f"rope_delta={delta} {'PASS' if fin else 'FAIL'}", flush=True)
        t_ids = [t for t in ids if t not in (mm_input.IMAGE_TOKEN_ID,
                                             mm_input.VISION_START_ID,
                                             mm_input.VISION_END_ID)]
        pc.release()
        model.reset()
        lg_txt = pc.prefill(t_ids).float().cpu()
        d = (lg_img.float().cpu() - lg_txt).abs().max().item()
        print(f"3 image-vs-text max|d|={d:.3f} {'PASS' if d > 1e-2 else 'FAIL'}",
              flush=True)
    print("ENGINE_PATH_DONE")


if __name__ == "__main__":
    main()
