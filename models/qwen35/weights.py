"""Checkpoint loading for Qwen3.5: text, MTP and vision tensors onto the GPU.

Reads the text-model and MTP tensors as contiguous tensors; the vision tower is
read in the same pass when the checkpoint has one (``config.json:
vision_config``) and the caller has not opted out, because the resident dict is
what ``Qwen35.enable_vision`` builds the tower from later. The MTP head is kept
because speculative decoding needs it.

The vision tensors are a separate memory policy from the text ones:

* they keep their ``model.visual.`` prefix (``Qwen35`` filters them back out of
  the dict into ``vision_sd``), and they are never routed through the text int8
  path -- the tower is compute-bound, so quantising it only helps when the
  0.62 GiB of fp16 weights is the thing that does not fit;
* that case is ``vision_quant``, which stores them as int8 + per-row scale using
  the same :class:`~int8_weights.QWeight` (0.31 GiB resident). The tower
  dequantises one GEMM at a time, so the transient is one fp16 matrix rather
  than a second copy of the tower.

Module: ``coffin/models/qwen35``. Depends on ``backends`` (for the device the
tensors land on and the projection dtype), ``ops/dtypes`` and ``int8_weights``,
plus its sibling ``vision``; called by ``engine_api.load_weights``, and its dict
is handed straight to ``Qwen35``.

Constraints:
    * Only 2-D matrices go through ``is_quantizable``: norms, ``A_log``,
      ``dt_bias`` and ``conv1d.weight`` are consumed by bf16 activation kernels
      and must stay bf16, or those kernels silently run in the wrong precision.
    * Quantise during the load, not after: the fp16 tensor is dropped as soon as
      it is quantised, so peak memory is the larger of the two forms rather than
      their sum -- which is what lets the 9 GB checkpoint fit a 16 GB card.
    * Keys are matched by prefix string (``model.language_model.``, ``mtp.``,
      ``model.visual.``); they come from the checkpoint layout, so renaming one
      here breaks loading rather than a shape check.
    * ``vision=None`` means "auto": load the tower iff the checkpoint declares
      one; an explicit ``True`` on a checkpoint without vision tensors is an
      error, not an empty tower.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from backends import device as _device
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
    decode step, and half the resident footprint. ``vision_quant`` does the same
    for the tower's GEMM weights, independently.

    Args:
        model_dir: Checkpoint directory holding ``config.json`` and the
            ``*.safetensors`` shards.
        prefix: Parameter-name prefix of the text model in the checkpoint.
        quant: Store the text 2-D projection matrices as int8 ``QWeight``.
        weight_dtype: dtype for the projections and the vision tensors; defaults to
            ``caps.weight_dtype``, which is fp16 on the DCU this engine was tuned
            on because that part has packed fp16 math.
        vision: Whether to load the tower. ``None`` (auto) loads it iff
            ``config.json`` declares a ``vision_config``.
        vision_quant: Store the tower's GEMM weights as int8 + per-row scale.
            Worth it only when the fp16 tower is what does not fit.

    Returns:
        State dict keyed by checkpoint parameter name; text/MTP keys stripped of
        ``prefix``, vision keys still carrying ``VISUAL_PREFIX`` so the caller can
        split them out.

    Raises:
        RuntimeError: The checkpoint declares a ``vision_config`` but no
            ``VISUAL_PREFIX`` tensor was found. Report the mismatched checkpoint
            rather than starting an engine that will fail on its first image.
    """
    from safetensors import safe_open

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
                    t = f.get_tensor(k).to(_device)
                    if not is_quantizable(k, t):
                        sd[k] = t.to(D).contiguous()
                        continue
                    t = t.to(weight_dtype).contiguous()
                    sd[k] = QWeight.quantize(t) if quant else t
                elif take_vision and k.startswith(VISUAL_PREFIX):
                    t = f.get_tensor(k).to(_device, dtype=weight_dtype).contiguous()
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
