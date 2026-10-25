"""Dump the fp32 vision-tower oracle used by tests/test_vision.py.

Answers what the engine's vision tower must reproduce numerically, by pinning an
independent oracle once so the unit tests do not need transformers or a GPU.

Why fp32: the checkpoint is bf16, and a 24-layer ViT at bf16 drifts ~1e-1 by the
last block (activations reach ~1e3, and each block adds ~8e-3 relative error).
That makes a bf16 oracle useless as a numeric gate -- it cannot tell a correct
implementation from one with a 1e-2 formula error. fp32 is the reference the
engine's tower is compared against (gate 1e-5), and the fp16 tower is then held
to ~3e-2, which is what fp16 itself costs.

    flock /root/private_data/gpu.lock -c 'python3 diag/make_vision_fixture.py'

Writes /root/private_data/engine_vision/oracle_fp32.pt with, for a handful of
images: the processor's pixel_values/grid_thw, and HF's fp32 output embeddings.
Small on purpose (256² and 512², plus a non-square 384x768) so it can live next
to the tests; the 1024² case is measured by the probe, not pinned as a fixture.

A successful run ends with ``FIXTURE_DONE`` and a byte count; the tests then
consume the file, so regenerating it invalidates any stored test expectation.

Module: ``coffin/diag``. Standalone fixture generator: imports transformers,
safetensors and torch, and nothing under ``coffin/`` imports it. The tests read
only the ``.pt`` it writes.

Constraints:
    * fp32 on the oracle, always. The whole point is a reference far tighter than
      the engine's own precision, so a bf16 fixture would defeat the gate.
    * ``strict=False`` plus an explicit mismatch check: the reference architecture
      expects a few buffers the checkpoint omits, and the check still catches a
      genuinely wrong architecture.
    * Deterministic image content, not noise. With noise every patch permutation
      still looks plausible, so a patch-order bug would pass.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch

MD = "/root/private_data/models/Qwen3.5-4B"
OUT = "/root/private_data/engine_vision/oracle_fp32.pt"

#: (height, width) — a small grid, a 1 MP grid, and a non-square phone-ish one.
SIZES = ((256, 256), (512, 512), (384, 768))


def pattern(h: int, w: int) -> Image:  # noqa: F821 - imported below
    """Build a deterministic, structured image (gradients + blocks), not noise.

    Args:
        h: Image height in pixels.
        w: Image width in pixels.

    Returns:
        A PIL image with a per-axis gradient plus a checker of hard-edged
        blocks. Random noise would hide a patch-order bug: with noise every
        permutation still looks "statistically fine". Blocks and hard edges do
        not, which is why this pattern is the fixture input.

    The ``Image`` return annotation is resolved only when the local
    ``from PIL import Image`` runs, hence the ``noqa`` above.
    """
    from PIL import Image as _Image

    a = np.zeros((h, w, 3), np.uint8)
    a[..., 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    a[..., 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
    a[..., 2] = 64
    bh, bw = max(h // 8, 1), max(w // 8, 1)
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 0:
                continue
            a[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw] = [
                (i * 31) % 256, (j * 47) % 256, ((i + j) * 29) % 256]
    return _Image.fromarray(a)


def main() -> None:
    """Build the fp32 oracle for every fixture size and write ``oracle_fp32.pt``."""
    from safetensors import safe_open
    from transformers import AutoProcessor
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    proc = AutoProcessor.from_pretrained(MD)
    text = proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"},
                                      {"type": "text", "text": "x"}]}],
        tokenize=False, add_generation_prompt=True)

    cfg = json.loads((pathlib.Path(MD) / "config.json").read_text())
    vcfg = dict(cfg["vision_config"])
    vcfg.pop("model_type", None)
    vm = Qwen3VLVisionModel(Qwen3VLVisionConfig(**vcfg))
    vm = vm.to("cuda", dtype=torch.float32).eval()

    wm = json.loads((pathlib.Path(MD) / "model.safetensors.index.json")
                    .read_text())["weight_map"]
    sd = {}
    for shard in sorted({wm[k] for k in wm if k.startswith("model.visual.")}):
        with safe_open(str(pathlib.Path(MD) / shard), "pt") as f:
            for k in f.keys():
                if k.startswith("model.visual."):
                    sd[k[len("model.visual."):]] = f.get_tensor(k).float()
    missing, unexpected = vm.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"oracle load mismatch: {missing} {unexpected}")
    del sd
    torch.cuda.empty_cache()

    cases = {}
    for hw in SIZES:
        out = proc(text=[text], images=[pattern(*hw)], return_tensors="pt")
        pv = out["pixel_values"].to("cuda", dtype=torch.float32)
        grid = out["image_grid_thw"].to("cuda")
        with torch.no_grad():
            res = vm(pv, grid_thw=grid)
        emb = res[0] if isinstance(res, (tuple, list)) else res.pooler_output
        cases[f"{hw[0]}x{hw[1]}"] = {
            "pixel_values": out["pixel_values"].to(torch.float32).cpu(),
            "grid_thw": out["image_grid_thw"].cpu(),
            "embeds": emb.float().cpu(),
        }
        print(f"{hw[0]}x{hw[1]} grid={out['image_grid_thw'].tolist()} "
              f"emb={tuple(emb.shape)}", flush=True)
        del pv, grid, res, emb
        torch.cuda.empty_cache()

    pathlib.Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    torch.save(cases, OUT)
    print("wrote", OUT, pathlib.Path(OUT).stat().st_size, "bytes")
    print("FIXTURE_DONE")


if __name__ == "__main__":
    main()
