"""The dtype and mask constants every op in this layer agrees on, re-exported from the backend caps.

The values are not this repository's to choose: they are hardware facts, so the
single source is ``backends.caps()`` (``act_dtype`` / ``weight_dtype``). This
module keeps the old names so that the ~200 ``from ops.dtypes import D`` call
sites did not have to be renamed in the same change that moved the source.

Module: ``coffin/ops``. Depends on ``backends`` and ``torch``; imported by every
other op module and by ``models/*``.

Constraints:
    * ``D`` and ``WF`` must stay *derived* names with no literal behind them. A
      second literal here would be a platform assumption that silently disagrees
      with the backend the process actually selected, which is the exact bug this
      migration removed.
    * The constants are bound at import, together with the backend selection
      itself: there is no runtime hot swap (PLATFORMS.md §8), so a module that
      reads them is reading a value that cannot change under it.
    * ``NEG_INF`` is the attention-mask fill value; use it instead of a large
      negative literal so the softmax underflow stays exact.
"""

from __future__ import annotations

from backends import caps as _caps

#: The selected backend's capability record, read once. Kept private: callers
#: that need a flag other than the two dtypes import ``backends.caps()`` directly
#: rather than reaching into this module for a second copy of the record.
_CAPS = _caps()

#: Activation dtype for every fused kernel, including the int8 KV cache path.
#: bf16 on every backend that has kernels today, and the reference runs it too
#: because that is what the model allocates and what the KV quantiser is written
#: against.
D = _CAPS.act_dtype

#: Projection-weight dtype. Not necessarily ``D``: the DCU this engine was tuned
#: on has packed fp16 math and no packed bf16 math, so its prefill GEMM reaches
#: 13-20 TFLOP/s on fp16 weights against 2.3 TFLOP/s on bf16 (wide-N shapes gain
#: ~7x). Activations stay bf16 either way, and the decode GEMV copy loop is
#: bandwidth-bound, so reading the weights as fp16 costs nothing there. A part with
#: packed bf16 declares bf16 here instead and nothing else changes.
WF = _CAPS.weight_dtype

#: Mask fill value; `-inf` makes the softmax weight exactly zero.
NEG_INF = float("-inf")
