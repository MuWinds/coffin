"""End-to-end acceptance for the vision path on the real checkpoint.

Four checks, in the order that isolates failures:

  1. text regression -- the text-only prefill logits must match the baseline the
     *unpatched* tree produced (bit-level: vision must be a no-op without images)
  2. image prefill   -- same prompt with an image: finite logits, and different
     from the same ids without the image (the image actually reaches the model)
  3. decode after image -- positions advance through ``pos + rope_delta``; the
     three decoded tokens must differ from the text-only run
  4. cache salt      -- the same ids with image B must not reuse image A's prefix

Runs in one process, one model, a few requests: the point is correctness, not
throughput.  ``--no-vision`` gives the text-only side of the comparison.

    flock /root/private_data/gpu.lock -c 'python3 diag/e2e_vision.py'
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

MD = "/root/private_data/models/Qwen3.5-4B"


def image(size: int, seed: int):
    from PIL import Image

    rng = np.random.default_rng(seed)
    a = np.full((size, size, 3), 255, np.uint8)
    band = size // 8
    for i in range(8):
        if i % 2 == 0:
            a[i * band:(i + 1) * band, :, :] = rng.integers(0, 90, 3, dtype=np.uint8)
    return Image.fromarray(a)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MD)
    ap.add_argument("--max-seq", type=int, default=4096)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--write-baseline", default=None)
    args = ap.parse_args()

    root = pathlib.Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    import engine_api
    import mm_input
    from transformers import AutoProcessor, AutoTokenizer

    cfg = json.loads((pathlib.Path(args.model) / "config.json").read_text())
    tok = AutoTokenizer.from_pretrained(args.model)
    proc = AutoProcessor.from_pretrained(args.model)

    # The image placeholder MUST come from the chat template: the processor only
    # expands an existing ``<|image_pad|>``, and silently drops an image whose
    # placeholder is missing (no error, prompt length unchanged).  That failure
    # mode is why this probe asserts ``image_pads > 0`` below.
    rendered = proc.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "图里有什么？请简短回答。"}]}],
        tokenize=False, add_generation_prompt=True)
    text_ids = tok.encode(rendered)

    def encode_with_image(img):
        out = proc(text=[rendered], images=[img], return_tensors="pt")
        return out["input_ids"][0].tolist(), out["pixel_values"], out["image_grid_thw"]

    ids_a, pv_a, grid_a = encode_with_image(image(512, 0))
    ids_b, pv_b, grid_b = encode_with_image(image(512, 1))
    spans = mm_input.find_image_spans(ids_a)
    n_pads = sum(e - s for s, e in spans)
    if n_pads != int(grid_a.prod()) // 4:
        raise SystemExit(
            f"image was not spliced into the prompt: {n_pads} pads vs grid "
            f"{grid_a.tolist()} ({int(grid_a.prod()) // 4} expected)")
    pos, delta = mm_input.mrope_positions(spans, grid_a, len(ids_a), device="cuda")
    print(f"prompt_with_image={len(ids_a)} image_pads={n_pads} "
          f"grid={grid_a.tolist()} rope_delta={delta}", flush=True)
    assert len(ids_b) == len(ids_a) and bool((grid_b == grid_a).all())

    # The text-only side of every comparison: the same prompt with the image
    # placeholders stripped out (and the surrounding markers, which the template
    # only emits for an image), so the two runs differ solely by the image.
    text_only_ids = [t for t in ids_a
                     if t not in (mm_input.IMAGE_TOKEN_ID,
                                  mm_input.VISION_START_ID, mm_input.VISION_END_ID)]

    weights = engine_api.load_weights(args.model, quant=True)
    model = engine_api.create_model(args.model, max_seq=args.max_seq, bench=False,
                                    weights=weights, paged=True)
    model.enable_paging(-(-args.max_seq // 64), 64)
    model.enable_vision()
    model.capture()
    print("model ready", flush=True)

    def mm_for(pv, grid, ids):
        emb = model.image_embeds(pv, grid)
        p, d = mm_input.mrope_positions(mm_input.find_image_spans(ids), grid,
                                        len(ids), device="cuda")
        return mm_input.MMInput(embeds=emb, grid_thw=grid.to("cuda"),
                                positions=p, rope_delta=d,
                                digest=mm_input.image_digest([b"e2e"]),
                                spans=mm_input.find_image_spans(ids))

    # 1. text regression: a text-only prompt (no image anywhere) must reproduce
    #    the logits this same prompt produced before the vision work landed.
    model.reset()
    text_logits = model.prefill(text_only_ids).float().cpu()
    if args.write_baseline:
        torch.save(text_logits, args.write_baseline)
        print("text baseline written", args.write_baseline)
    elif args.baseline and pathlib.Path(args.baseline).exists():
        ref = torch.load(args.baseline, map_location="cpu").float()
        rel = ((text_logits - ref).norm() / ref.norm()).item()
        print(f"1 text-regression rel={rel:.2e} "
              f"{'PASS' if rel <= 1e-6 else 'FAIL'} (gate 1e-6)", flush=True)

    # 2. image prefill
    model.reset()
    mm_a = mm_for(pv_a, grid_a, ids_a)
    logits_a = model.prefill(ids_a, mm=mm_a)
    finite = bool(torch.isfinite(logits_a).all())
    with torch.no_grad():
        diff = (logits_a.float() - text_logits.to("cuda")).abs().max().item()
    print(f"2 image-prefill finite={finite} argmax={int(logits_a.argmax())} "
          f"max|img-text|={diff:.3f} "
          f"{'PASS' if finite and diff > 1e-2 else 'FAIL'}", flush=True)

    # 3. decode after image (rope_delta in play)
    toks = [int(logits_a.argmax())]
    lg = logits_a
    for _ in range(2):
        lg = model.decode_step(toks[-1])
        toks.append(int(lg.argmax()))
    model.reset()
    lg_t = model.prefill(text_only_ids)
    t_toks = [int(lg_t.argmax())]
    for _ in range(2):
        lg_t = model.decode_step(t_toks[-1])
        t_toks.append(int(lg_t.argmax()))
    print(f"3 decode img={toks} text={t_toks} "
          f"{'PASS' if toks != t_toks else 'FAIL'}", flush=True)

    # 4. cache salt: image B must not reuse image A's prefix
    model.reset()
    mm_a2 = mm_for(pv_a, grid_a, ids_a)
    model.prefill(ids_a, mm=mm_a2)
    model.reset()
    mm_b = mm_for(pv_b, grid_b, ids_b)
    logits_b = model.prefill(ids_b, mm=mm_b)
    same = bool(torch.equal(
        logits_b.float().cpu().argmax().reshape(1),
        torch.tensor([int(logits_b.argmax())])))
    diff_ab = (logits_b.float().cpu() - logits_a.float().cpu()).abs().max().item()
    print(f"4 salt: max|logits(A)-logits(B)|={diff_ab:.3f} "
          f"{'PASS' if diff_ab > 1e-2 else 'FAIL'}", flush=True)
    print("E2E_DONE")


if __name__ == "__main__":
    main()
