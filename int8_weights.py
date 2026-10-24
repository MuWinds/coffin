"""Weight-only int8 quantisation (W8A16).

Decode on the DCU is bandwidth-bound: the projections run at 540-680 GB/s against
a ~700 GB/s copy ceiling and account for ~48% of a step, hipBLAS GEMV at M=1 is
unusable, and Triton has no gfx906 backend. Halving the bytes every projection
reads is therefore the largest remaining lever, and it is cheap to take because
the dequantise can hide behind the memory traffic: the weights are stored as
int8 with one fp32 scale per output row, and the kernel applies that scale once
to the accumulated dot product instead of per element.

Scheme: symmetric, per output row (per output channel), absmax.

    s[n]     = max_k |w[n, k]| / 127
    wq[n, k] = round(w[n, k] / s[n]) clamped to [-127, 127]

Rows are independent, so the error stays local to a row and the scale folds into
the kernel's final multiply. Activations stay bf16 -- quantising them too (W8A8)
would need a second quantiser on the critical path and would start to move the
error into the attention scores.

``embed_tokens`` is included: it is the tied lm_head, and its matmul is the
single largest weight read of a decode step. ``conv1d.weight`` stays bf16 (its
kernel consumes fp32 taps and it is 32 KB), as do all the 1-D norms.
"""

from __future__ import annotations

import torch

D = torch.bfloat16


def is_quantizable(key: str, w: torch.Tensor) -> bool:
    """True for the 2-D matrices that are read as weights at decode time."""
    return w.dim() == 2 and not key.endswith("conv1d.weight")


class QWeight:
    """An ``[N, K]`` weight held as int8 plus a per-row fp32 scale."""

    __slots__ = ("wq", "scale", "n", "k")

    def __init__(self, wq: torch.Tensor, scale: torch.Tensor) -> None:
        self.wq = wq                      # [N, K] int8, contiguous
        self.scale = scale                # [N] fp32, contiguous
        self.n, self.k = wq.shape

    @classmethod
    def quantize(cls, w: torch.Tensor) -> "QWeight":
        wf = w.float()
        scale = wf.abs().amax(-1).clamp_min(1e-8) / 127.0
        q = (wf / scale[:, None]).round().clamp_(-127, 127).to(torch.int8)
        return cls(q.contiguous(), scale.contiguous())

    def dequant(self, dtype: torch.dtype = D) -> torch.Tensor:
        """Materialise weights in ``dtype``; used by the prefill GEMM path only.

        fp16 is what prefill asks for: this part has no packed bf16 math, so the
        same GEMM runs ~7x faster on fp16 weights.
        """
        return (self.wq.float() * self.scale[:, None]).to(dtype)

    def rows(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather rows: returns ``(int8 [T, K], fp32 scale [T])`` for embeddings."""
        return self.wq.index_select(0, idx), self.scale.index_select(0, idx)

    def nbytes(self) -> int:
        return self.wq.numel() + self.scale.numel() * 4

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"QWeight({self.n}x{self.k} int8)"


def quantize_state_dict(sd: dict) -> dict:
    """Replace every quantisable 2-D weight in ``sd`` with a :class:`QWeight`.

    Mutates ``sd`` in place and drops each bf16 tensor as soon as it has been
    quantised (so the resident copy shrinks instead of doubling: 9 GB of bf16
    weights become ~4.5 GB, which is what leaves room on a 16 GB card for a longer
    context and a paged KV pool). Returns byte-count stats.

    ``load_weights`` quantises during the load instead and does not use this; it
    stays for a caller that already holds bf16 tensors in VRAM.
    """
    n_before = n_after = 0
    for k in list(sd):
        w = sd[k]
        if not isinstance(w, torch.Tensor) or not is_quantizable(k, w):
            continue
        n_before += w.numel() * 2
        q = QWeight.quantize(w)
        n_after += q.nbytes()
        sd[k] = q
        del w
    return {"bf16_bytes": n_before, "int8_bytes": n_after}


def quantized_keys(sd: dict) -> list[str]:
    return [k for k, v in sd.items() if isinstance(v, QWeight)]
