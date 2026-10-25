#!/usr/bin/env python3
"""Lead-side integration probe: how far does the vision path actually run?

Answers which rung of the vision stack is broken, rather than whether the stack as
a whole works. Run inside the DCU container (``engine_vision`` deployment):

    flock /root/private_data/gpu.lock -c 'python3 diag/probe_vision_e2e.py'

It walks the stack bottom-up and stops at the first broken rung, so a failure
names the layer that is wrong instead of dumping one traceback from the top:

    1. checkpoint config + vision tensors
    2. processor -> input_ids / pixel_values / grid_thw
    3. engine vision tower vs the transformers oracle (rel_err)
    4. MMInput construction (spans, 3D positions, rope_delta)
    5. real model: image_embeds + prefill(ids, mm=...) + greedy decode
    6. without images: prefill logits must be bit-identical to the baseline file

Read the ``=== n name ===`` banners in order: the last one printed before a
failure is the rung to investigate, and the ``PASS``/``FAIL`` lines carry their
own gate so no interpretation is needed. Rung 3 prints ``oracle fixture skipped``
when the fixture does not match the image or size — that is a skip, not a pass.

The last rung is the regression gate: ``--baseline`` writes the text-only logits
of a fixed prompt on the *unpatched* engine, and every later run compares
against it.

Module: ``coffin/diag``. Standalone probe: imports ``engine_api``, ``mm_input``
and transformers, and nothing under ``coffin/`` imports it.

Constraints:
    * ``engine_api`` is imported after ``sys.path`` is fixed up inside ``main``,
      because this script lives in ``diag/`` while the engine modules use
      absolute imports rooted at ``coffin/``.
    * ``flock`` is required: the checkpoint, the reference tower and the model
      will not fit beside a co-tenant on the same card.
    * Rung 6 needs a baseline written by a run on the *unpatched* tree. Without
      that file it prints a skip and the regression gate has not run.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

MD = "/root/private_data/models/Qwen3.5-4B"
PROMPT = [5, 6, 7, 8, 9, 10]


def _section(name: str) -> None:
    """Print a rung banner so the last one before a failure names the broken layer.

    Args:
        name: Rung label. Banners go to stdout with ``flush=True`` so they are
            not lost when a later rung raises.
    """
    print(f"\n=== {name} ===", flush=True)


def main() -> None:
    """Walk rungs 1-6 in order and print each verdict."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MD)
    ap.add_argument("--max-seq", type=int, default=4096)
    ap.add_argument("--baseline", default="/root/private_data/engine_vision/text_baseline.pt")
    ap.add_argument("--write-baseline", action="store_true")
    ap.add_argument("--image", default=None, help="PNG to send through the stack")
    args = ap.parse_args()

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    import engine_api
    import mm_input
    from PIL import Image
    import numpy as np

    cfg = json.loads((pathlib.Path(args.model) / "config.json").read_text())
    vcfg = cfg.get("vision_config")
    _section("1 checkpoint")
    print("vision_config:", bool(vcfg), "depth", (vcfg or {}).get("depth"))
    wm = json.loads((pathlib.Path(args.model) / "model.safetensors.index.json")
                    .read_text())["weight_map"]
    n_vis = sum(1 for k in wm if k.startswith("model.visual."))
    print("vision tensors:", n_vis)

    _section("2 processor")
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(args.model)
    if args.image:
        img = Image.open(args.image).convert("RGB")
    else:
        a = np.zeros((512, 512, 3), np.uint8)
        a[:] = (255, 255, 255)
        a[100:180, 100:400] = 0
        img = Image.fromarray(a)
    text = proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is in the image?"}]}],
        tokenize=False, add_generation_prompt=True)
    out = proc(text=[text], images=[img], return_tensors="pt")
    ids = out["input_ids"][0].tolist()
    pv, grid = out["pixel_values"], out["image_grid_thw"]
    spans = mm_input.find_image_spans(ids)
    print("prompt tokens", len(ids), "image pads", sum(e - s for s, e in spans),
          "grid", grid.tolist())

    _section("3 engine vision tower vs oracle")
    model = engine_api.create_model(args.model, max_seq=args.max_seq, bench=False, paged=True)
    paged = model.enable_paging(num_blocks=-(-args.max_seq // 64), block_size=64)
    model.enable_vision(quant=False)
    emb = model.image_embeds(pv, grid)
    print("engine embeds", tuple(emb.shape), emb.dtype)
    oracle_path = pathlib.Path("/root/private_data/engine_vision/oracle_512.pt")
    if oracle_path.exists() and not args.image:
        o = torch.load(oracle_path, map_location="cpu")
        if tuple(o["grid"].tolist()) == tuple(grid.tolist()):
            e = emb.float().cpu()
            ref = o["embeds"].float()
            rel = ((e - ref).norm() / ref.norm()).item()
            print(f"oracle rel_err {rel:.2e} (gate 2e-3): {'PASS' if rel <= 2e-3 else 'FAIL'}")
    else:
        print("oracle fixture skipped (different image/size or missing)")

    _section("4 MMInput")
    pos, delta = mm_input.mrope_positions(spans, grid, len(ids), device="cpu")
    mm = mm_input.MMInput(embeds=emb, grid_thw=grid, positions=pos.to("cuda"),
                          rope_delta=delta, digest=mm_input.image_digest([b"probe"]),
                          spans=spans)
    print("positions", tuple(mm.positions.shape), "rope_delta", delta, "spans", spans)

    _section("5 real model prefill with image")
    model.reset()
    logits = model.prefill(ids, mm=mm)
    print("logits finite", bool(torch.isfinite(logits).all()), "argmax", int(logits.argmax()))
    tok = int(logits.argmax())
    for _ in range(3):
        logits = model.decode_step(tok)
        tok = int(logits.argmax())
    print("decoded ids", tok)

    _section("6 text-only regression")
    model.reset()
    base_logits = model.prefill(PROMPT)
    if args.write_baseline:
        torch.save(base_logits.cpu(), args.baseline)
        print("baseline written", args.baseline)
    elif pathlib.Path(args.baseline).exists():
        ref = torch.load(args.baseline, map_location="cpu").float()
        rel = ((base_logits.float().cpu() - ref).norm() / ref.norm()).item()
        print(f"text-only logits rel_err {rel:.2e} (gate 1e-6): "
              f"{'PASS' if rel <= 1e-6 else 'FAIL'}")
    else:
        print("no baseline file; rerun with --write-baseline on the unpatched tree")

    print("\nPROBE_DONE")


if __name__ == "__main__":
    main()
