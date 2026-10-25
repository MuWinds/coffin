"""Oracle parity for the engine's vision tower (container, real weights).

Answers whether the engine's tower computes the reference function, at both
precisions. The authoritative check: load the checkpoint's ``model.visual.*`` into
the engine's :class:`VisionTower`, run it on the processor's ``pixel_values``, and
compare with what transformers 4.57.6's ``Qwen3VLVisionModel`` produces from the
*same* weights and the same input. bf16 oracle vs fp16 engine, so the gate is
relative error, not bit equality.

Read each size row's ``rel_err`` against its printed gate: 2e-3 for the fp16
tower, 2e-2 for the int8 one. The looser int8 gate is the quantisation cost, not
sloppiness — a run that fails the int8 gate while passing the fp16 gate has a
quantisation bug, not a formula bug, so re-run with ``--vision-int8`` off to tell
the two apart. ``vit_ms`` and ``peak_GiB`` are the cost side of the same trade.

    flock /root/private_data/gpu.lock -c 'python3 diag/probe_vision_oracle.py'
    ... --vision-int8      # same, with the resident int8 tower

Module: ``coffin/diag``. Standalone probe: imports ``engine_api`` and
``models/qwen35/vision.py`` directly, plus transformers for the oracle.

Constraints:
    * The checkpoint keys are ``model.visual.``-prefixed while the reference module
      wants them bare; the oracle strips the prefix on load. A change to either
      naming convention silently loads nothing and the parity result becomes
      meaningless.
    * The oracle is bf16 and the engine fp16, so bit equality is impossible by
      construction; the relative-error gate is the only valid comparison.
    * ``flock`` is required: the tower and the reference model together do not fit
      beside a co-tenant on this card.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

MD = "/root/private_data/models/Qwen3.5-4B"


def oracle(processor, pixel_values, grid_thw, model_dir):
    """Run transformers' vision tower on the same checkpoint tensors and inputs.

    Args:
        processor: The AutoProcessor; kept for signature symmetry with the engine
            side, which needs it, though the oracle only uses pixel values.
        pixel_values: Processor output, cast to bf16 for the reference.
        grid_thw: Image grid, needed to unpatch.
        model_dir: Checkpoint directory for ``config.json`` and the index.

    Returns:
        Reference embeddings as a float32 CPU tensor.

    Raises:
        RuntimeError: The state dict does not match the reference architecture;
            the message lists missing and unexpected keys. A mismatch means the
            installed transformers disagrees with the checkpoint, so the parity
            number would compare against the wrong formula.
    """
    from safetensors import safe_open
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    cfg = json.loads((pathlib.Path(model_dir) / "config.json").read_text())
    vcfg = dict(cfg["vision_config"])
    vcfg.pop("model_type", None)
    vm = Qwen3VLVisionModel(Qwen3VLVisionConfig(**vcfg))
    vm = vm.to("cuda", dtype=torch.bfloat16).eval()

    wm = json.loads((pathlib.Path(model_dir) / "model.safetensors.index.json")
                    .read_text())["weight_map"]
    sd = {}
    for shard in sorted({wm[k] for k in wm if k.startswith("model.visual.")}):
        with safe_open(str(pathlib.Path(model_dir) / shard), "pt") as f:
            for k in f.keys():
                if k.startswith("model.visual."):
                    sd[k[len("model.visual."):]] = f.get_tensor(k)
    missing, unexpected = vm.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"oracle load mismatch: {missing} {unexpected}")
    del sd
    torch.cuda.empty_cache()
    with torch.no_grad():
        out = vm(pixel_values.to("cuda", dtype=torch.bfloat16),
                 grid_thw=grid_thw.to("cuda"))
    emb = out[0] if isinstance(out, (tuple, list)) else out.pooler_output
    return emb.float().cpu()


def main() -> None:
    """Compare the engine tower against the bf16 oracle at three image sizes."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MD)
    ap.add_argument("--vision-int8", action="store_true")
    args = ap.parse_args()

    root = pathlib.Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    import engine_api
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(args.model)
    text = proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"},
                                      {"type": "text", "text": "x"}]}],
        tokenize=False, add_generation_prompt=True)

    def make(h, w):
        """Build the probe image: two axis gradients with one solid red square.

        Args:
            h: Image height.
            w: Image width.

        Returns:
            A PIL image. The gradients give the patch embedder a smooth field to
            get wrong in a detectable way, and the red square gives a hard edge
            whose position reveals a patch-order or crop error.
        """
        a = np.zeros((h, w, 3), np.uint8)
        a[..., 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
        a[..., 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
        a[h // 4:h // 2, w // 4:w // 2] = [255, 0, 0]
        from PIL import Image
        return Image.fromarray(a)

    sd = engine_api.load_weights(args.model, quant=True,
                                 vision_quant=args.vision_int8)
    n_vis = sum(1 for k in sd if k.startswith("model.visual."))
    print("vision tensors loaded:", n_vis, "int8:", args.vision_int8, flush=True)

    from models.qwen35.vision import VisionTower
    cfg = json.loads((pathlib.Path(args.model) / "config.json").read_text())
    tower = VisionTower({k: v for k, v in sd.items()
                         if k.startswith("model.visual.")},
                        cfg["vision_config"], device="cuda",
                        quant=args.vision_int8)
    print("tower:", tower.summary(), flush=True)

    ok = True
    for hw in ((256, 256), (512, 512), (1024, 1024)):
        out = proc(text=[text], images=[make(*hw)], return_tensors="pt")
        pv, grid = out["pixel_values"], out["image_grid_thw"]
        ref = oracle(proc, pv, grid, args.model)

        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        with torch.no_grad():
            got = tower(pv.to("cuda", dtype=torch.float16),
                        grid.to("cuda")).float().cpu()
        ms = (time.time() - t0) * 1000
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        rel = ((got - ref).norm() / ref.norm()).item()
        gate = 2e-3 if not args.vision_int8 else 2e-2
        good = rel <= gate
        ok &= good
        print(f"{hw[0]}x{hw[1]} seq={pv.shape[0]} emb={tuple(got.shape)} "
              f"rel_err={rel:.2e} ({'PASS' if good else 'FAIL'} gate {gate}) "
              f"vit_ms={ms:.0f} peak_GiB={peak:.2f}", flush=True)
        del ref, got, out
        torch.cuda.empty_cache()

    print("ORACLE_DONE", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
