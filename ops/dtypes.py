"""Dtypes shared by every op.

Everything the fused kernels touch is bf16 (``D``): the activations and the
int8 KV cache are written against it.  Projection weights are fp16 (``WF``)
because this part has packed fp16 math and no packed bf16 math -- the same GEMM
runs 13-20 TFLOP/s on fp16 weights but 2.3-7 on bf16 -- and the decode GEMV is
bandwidth-bound, so reading them as fp16 costs nothing there.  ``NEG_INF`` is
the mask fill value.
"""

from __future__ import annotations

import torch


D = torch.bfloat16
#: dtype of the projection weights.  Not bf16: this part has packed fp16 math and
#: no packed bf16 math, so the prefill GEMM runs 2.3 TFLOP/s on bf16 weights and
#: 13-20 TFLOP/s on fp16 ones (wide-N shapes gain ~7x).  Activations stay bf16 --
#: every kernel and the KV cache are written against that -- and the copy loop of
#: the decode GEMV is bandwidth-bound, so reading the weights as fp16 costs
#: nothing there.
WF = torch.float16
NEG_INF = float("-inf")
