"""Checkpoint loading for Qwen3.5.

Reads the text-model and MTP tensors onto the GPU as contiguous tensors; the
vision tower is read in the same pass when the checkpoint has one (``
config.json:vision_config``) and the caller has not opted out, because the
resident dict is what ``Qwen35.enable_vision`` builds the tower from later.
The MTP head is kept because speculative decoding needs it.  With ``quant`` the
2-D matrices arrive as int8 ``QWeight`` and the fp16 tensor is dropped as soon
as it has been quantised, so peak memory is the larger of the two rather than
their sum.

The vision tensors are a separate memory policy from the text ones:

* they keep their ``model.visual.`` prefix (``Qwen35`` filters them back out of
  the dict into ``vision_sd``), and they are never routed through the text int8
  path -- the tower is compute-bound, so quantising it only helps when the
  0.62 GiB of fp16 weights is the thing that does not fit;
* that case is ``vision_quant``, which stores them as int8 + per-row scale using
  the same :class:`~int8_weights.QWeight` (0.31 GiB resident).  The tower
  dequantises one GEMM at a time, so the transient is one fp16 matrix rather
  than a second copy of the tower.

``vision=None`` means "auto": load the tower iff the checkpoint declares one.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open

from int8_weights import QWeight, is_quantizable
from models.qwen35.vision import VISUAL_PREFIX, is_vision_quantizable
from ops.dtypes import D, WF


def load_weights(model_dir: str | Path, prefix: str = "model.language_model.",
                 quant: bool = False, weight_dtype: torch.dtype = WF,
                 vision: bool | None = None,
                 vision_quant: bool = False) -> dict:
    """Read the text-model, MTP and vision tensors from disk onto the GPU.

    This is the slow part (~1 min for 9 GB); a caller that wants to rebuild the
    model after a code change can hold the returned dict in VRAM and pass it back
    through ``Qwen35(..., weights=...)`` to skip the reload.

    With ``quant`` the text 2-D matrices arrive as
    :class:`int8_weights.QWeight` instead of fp16: half the bytes to read per
    decode step, and half the resident footprint. Quantising during the load
    (rather than after) means the fp16 tensor is dropped as soon as it has been
    quantised, so peak memory is the larger of the two rather than their sum.
    ``vision_quant`` does the same for the tower's GEMM weights, independently.

    The text 2-D matrices are the GEMV-managed projections, and they are cast to
    ``weight_dtype`` (fp16 by default); everything else -- norms, ``A_log``,
    ``dt_bias``, ``conv1d.weight`` -- is consumed by the bf16 activation kernels
    and stays bf16.  The vision tower's tensors arrive in ``weight_dtype``
    (or int8 with ``vision_quant``).
    """
    model_dir = Path(model_dir)
    with (model_dir / "config.json").open() as f:
        cfg = json.load(f)
    take_vision = bool(cfg.get("vision_config")) if vision is None else bool(vision)

    prefixes = (prefix, "mtp.")
    sd: dict[str, torch.Tensor] = {}
    n_vision = 0
    for p in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(p), "pt") as f:
            for k in f.keys():
                if k.startswith(prefixes):
                    t = f.get_tensor(k).to("cuda")
                    if not is_quantizable(k, t):
                        sd[k] = t.to(D).contiguous()
                        continue
                    t = t.to(weight_dtype).contiguous()
                    sd[k] = QWeight.quantize(t) if quant else t
                elif take_vision and k.startswith(VISUAL_PREFIX):
                    t = f.get_tensor(k).to("cuda", dtype=weight_dtype).contiguous()
                    if vision_quant and is_vision_quantizable(k, t):
                        sd[k] = QWeight.quantize(t)
                    else:
                        sd[k] = t
                    n_vision += 1
    if take_vision and n_vision == 0:
        raise RuntimeError(
            f"config.json declares a vision_config but no {VISUAL_PREFIX}* tensor "
            f"was found under {model_dir}")
    return sd
