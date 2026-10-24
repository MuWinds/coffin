"""Canonical vision-tower acceptance: formula parity + clean performance.

Runs the engine's tower against the fp32 transformers oracle and reports the
three gates agreed with the implementation owner:

  A. engine fp32 vs HF fp32   rel <= 2e-5   (formula correctness)
  B. engine fp16 vs HF fp32   rel <= 3e-2   (what fp16 itself costs)
  C. engine fp16 vs HF bf16   rel <= 1.5e-1 (smoke: the bf16 oracle's own noise)

Deliberately measures performance *after* dropping the text weights, so the
reported peak belongs to the tower and not to the resident model.

    flock /root/private_data/gpu.lock -c 'python3 diag/vision_accept.py'
    ... --sizes 256 512 1024 --vision-int8
"""
from __future__ import annotations

import argparse
import gc
import json
import pathlib
import statistics
import sys
import time

import numpy as np
import torch

MD = "/root/private_data/models/Qwen3.5-4B"


def make_image(h: int, w: int):
    """Structured image (gradients + hard-edged blocks): a patch-order bug shows."""
    from PIL import Image

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
    return Image.fromarray(a)


def build_oracle(model_dir: str, dtype: torch.dtype, sd_cpu: dict):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    cfg = json.loads((pathlib.Path(model_dir) / "config.json").read_text())
    vcfg = dict(cfg["vision_config"])
    vcfg.pop("model_type", None)
    vm = Qwen3VLVisionModel(Qwen3VLVisionConfig(**vcfg))
    vm = vm.to("cuda", dtype=dtype).eval()
    sd = {k: v.to(dtype) for k, v in sd_cpu.items()}
    missing, unexpected = vm.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"oracle load mismatch: {missing} {unexpected}")
    return vm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MD)
    ap.add_argument("--sizes", type=int, nargs="*", default=[256, 512])
    ap.add_argument("--vision-int8", action="store_true")
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()

    root = pathlib.Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    import engine_api
    from models.qwen35.vision import VisionTower
    from safetensors import safe_open
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(args.model)
    text = proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"},
                                      {"type": "text", "text": "x"}]}],
        tokenize=False, add_generation_prompt=True)

    # Vision tensors only, kept on the host under their short names (the module
    # names, without the checkpoint's ``model.visual.`` prefix).
    wm = json.loads((pathlib.Path(args.model) / "model.safetensors.index.json")
                    .read_text())["weight_map"]
    sd_cpu = {}
    for shard in sorted({wm[k] for k in wm if k.startswith("model.visual.")}):
        with safe_open(str(pathlib.Path(args.model) / shard), "pt") as f:
            for k in f.keys():
                if k.startswith("model.visual."):
                    sd_cpu[k[len("model.visual."):]] = f.get_tensor(k)
    n_vis = len(sd_cpu)
    print(f"vision tensors {n_vis}", flush=True)

    vcfg = json.loads((pathlib.Path(args.model) / "config.json").read_text())["vision_config"]
    # The tower expects the checkpoint-prefixed keys, so hand it a prefixed copy.
    tower_fp16 = VisionTower({f"model.visual.{k}": v for k, v in sd_cpu.items()},
                             vcfg, device="cuda", quant=args.vision_int8)
    tower_fp32 = VisionTower({f"model.visual.{k}": v for k, v in sd_cpu.items()},
                             vcfg, device="cuda", dtype=torch.float32,
                             compute_dtype=torch.float32, quant=False)
    print("towers built", flush=True)

    ok = True
    for size in args.sizes:
        out = proc(text=[text], images=[make_image(size, size)], return_tensors="pt")
        pv, grid = out["pixel_values"], out["image_grid_thw"]
        g = grid.to("cuda")
        print(f"\n-- {size}x{size} grid={grid.tolist()} patch_tokens={pv.shape[0]}",
              flush=True)

        ref_fp32 = build_oracle(args.model, torch.float32, sd_cpu)
        with torch.no_grad():
            r32 = ref_fp32(pv.to("cuda", dtype=torch.float32), grid_thw=g)
        r32 = (r32[0] if isinstance(r32, (tuple, list)) else r32.pooler_output)
        r32 = r32.float().cpu()
        ref_bf16 = build_oracle(args.model, torch.bfloat16, sd_cpu)
        with torch.no_grad():
            rb = ref_bf16(pv.to("cuda", dtype=torch.bfloat16), grid_thw=g)
        rb = (rb[0] if isinstance(rb, (tuple, list)) else rb.pooler_output)
        rb = rb.float().cpu()
        del ref_fp32, ref_bf16
        gc.collect()
        torch.cuda.empty_cache()

        def rel(a, b):
            return ((a.float().cpu() - b.float().cpu()).norm()
                    / b.float().cpu().norm()).item()

        with torch.no_grad():
            e32 = tower_fp32(pv.to("cuda", dtype=torch.float32), g).float().cpu()
        a = rel(e32, r32)
        with torch.no_grad():
            e16 = tower_fp16(pv.to("cuda", dtype=torch.float16), g).float().cpu()
        b = rel(e16, r32)
        c = rel(e16, rb)
        torch.cuda.empty_cache()

        ga = a <= 2e-5
        gb = b <= 3e-2
        gc_ = c <= 1.5e-1
        ok &= ga and gb and gc_
        print(f"   A engine_fp32 vs HF_fp32   rel={a:.3e}  "
              f"{'PASS' if ga else 'FAIL'} (gate 2e-5)", flush=True)
        print(f"   B engine_fp16 vs HF_fp32   rel={b:.3e}  "
              f"{'PASS' if gb else 'FAIL'} (gate 3e-2)", flush=True)
        print(f"   C engine_fp16 vs HF_bf16   rel={c:.3e}  "
              f"{'PASS' if gc_ else 'FAIL'} (gate 1.5e-1, smoke)", flush=True)

        # clean performance: only the tower's weights are resident now
        del e32, e16, r32, rb
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(args.iters):
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                tower_fp16(pv.to("cuda", dtype=torch.float16), g)
            torch.cuda.synchronize()
            times.append((time.time() - t0) * 1000)
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        print(f"   perf fp16 median={statistics.median(times):.0f} ms "
              f"(min {min(times):.0f}) peak_GiB={peak:.2f}", flush=True)

    print("\nVISION_ACCEPT", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
