"""Checkpoint loading for Qwen3.5.

Reads the text-model and MTP tensors onto the GPU as contiguous tensors and
skips the vision tower (the engine is text-only for now); the MTP head is kept
because speculative decoding needs it.  With ``quant`` the 2-D matrices arrive
as int8 ``QWeight`` and the fp16 tensor is dropped as soon as it has been
quantised, so peak memory is the larger of the two rather than their sum.
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open

from int8_weights import QWeight, is_quantizable
from ops.dtypes import D, WF


def load_weights(model_dir: str | Path, prefix: str = "model.language_model.",
                 quant: bool = False, weight_dtype: torch.dtype = WF) -> dict:
    """Read the text-model and MTP tensors from disk onto the GPU (contiguous).

    This is the slow part (~1 min for 9 GB); a caller that wants to rebuild the
    model after a code change can hold the returned dict in VRAM and pass it back
    through ``Qwen35(..., weights=...)`` to skip the reload. The vision tower is
    skipped; the MTP head is kept (speculative decoding needs it).

    With ``quant`` the 2-D matrices arrive as :class:`int8_weights.QWeight`
    instead of fp16: half the bytes to read per decode step, and half the resident
    footprint. Quantising during the load (rather than after) means the fp16
    tensor is dropped as soon as it has been quantised, so peak memory is the
    larger of the two rather than their sum.

    The 2-D matrices are the GEMV-managed projections, and they are cast to
    ``weight_dtype`` (fp16 by default); everything else -- norms, ``A_log``,
    ``dt_bias``, ``conv1d.weight`` -- is consumed by the bf16 activation kernels
    and stays bf16.
    """
    prefixes = (prefix, "mtp.")
    sd: dict[str, torch.Tensor] = {}
    for p in sorted(Path(model_dir).glob("*.safetensors")):
        with safe_open(str(p), "pt") as f:
            for k in f.keys():
                if not k.startswith(prefixes):
                    continue
                t = f.get_tensor(k).to("cuda")
                if not is_quantizable(k, t):
                    sd[k] = t.to(D).contiguous()
                    continue
                t = t.to(weight_dtype).contiguous()
                sd[k] = QWeight.quantize(t) if quant else t
    return sd
