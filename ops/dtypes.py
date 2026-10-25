"""The three dtype/mask constants every op in this layer agrees on.

Module: ``coffin/ops``. Depends on nothing above ``torch``; imported by every
other op module and by ``models/*``.

Constraints:
    * ``D`` (bf16) is the activation dtype: the fused kernels and the int8 KV
      cache are both written against it, so a new kernel must accept bf16
      activations rather than promoting them.
    * ``WF`` (fp16) is the projection-weight dtype, kept separate on purpose --
      see the comment on the constant for the measured reason.
    * ``NEG_INF`` is the attention-mask fill value; use it instead of a large
      negative literal so the softmax underflow stays exact.
"""

from __future__ import annotations

import torch


#: Activation dtype for every fused kernel, including the int8 KV cache path.
D = torch.bfloat16

#: Projection-weight dtype. Not bf16 on purpose: this part has packed fp16 math
#: and no packed bf16 math, so the prefill GEMM reaches 13-20 TFLOP/s on fp16
#: weights against 2.3 TFLOP/s on bf16 (wide-N shapes gain ~7x). Activations stay
#: bf16 and the decode GEMV copy loop is bandwidth-bound, so reading the weights
#: as fp16 costs nothing there.
WF = torch.float16

#: Mask fill value; `-inf` makes the softmax weight exactly zero.
NEG_INF = float("-inf")
