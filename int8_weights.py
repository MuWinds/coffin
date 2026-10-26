"""Weight-only int8 quantisation (W8A16), the encoder that feeds the GEMV kernels.

Module: ``coffin/int8_weights.py``. Leaf of the import graph: it imports only
``torch``. ``models/qwen35/weights.py`` quantises through it while loading a
checkpoint, ``ops/gemv.py`` consumes the resulting ``QWeight``, and both
``ops/embed.py`` and ``models/qwen35/vision.py`` understand the same object.

Why: decode is bandwidth-bound. On the accelerator this engine was first written
for, the projections ran at 540-680 GB/s against a ~700 GB/s copy ceiling and
accounted for ~48% of a step; the library's GEMV at M=1 was unusable (55-296
GB/s, and it changed kernel per shape), and Triton had no backend for that part.
Halving the bytes every projection reads is therefore the largest remaining
lever, and it is cheap to take because the dequantise can hide behind the memory
traffic: the weights are stored as int8 with one fp32 scale per output row, and
the kernel applies that scale once to the accumulated dot product instead of per
element.

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

Constraints:
    * Weights stay row-major ``[N, K]``, the layout the checkpoint ships and the
      HIP kernels index; only the dtype changes here, never the layout.
    * Stored weights are int8 with one fp32 scale per output row, and the math
      depends on that pairing: dequantising without the row scale, or scaling per
      tensor, reintroduces the drift this scheme avoids.
    * ``dequant`` materialises a dense copy. It exists for the prefill GEMM path
      only; calling it per decode step would undo the bandwidth saving.
    * ``quantize_state_dict`` mutates its dict in place and drops each bf16
      tensor as it goes, because peak VRAM during a load is the binding
      constraint on a 16 GB card.
"""

from __future__ import annotations

import torch

# Local copy of the activation dtype rather than an import of ``ops.dtypes``:
# ``ops/`` imports this module, so reaching back into it would close a cycle.
D = torch.bfloat16


def is_quantizable(key: str, w: torch.Tensor) -> bool:
    """Whether one checkpoint tensor is a projection this scheme may take.

    The test is 2-D-ness plus the ``conv1d.weight`` exclusion, not a key list:
    every 2-D matrix in this checkpoint is read as a ``[N, K]`` weight at decode
    time, whereas the convolution's kernel consumes fp32 taps and is 32 KB, so
    quantising it would cost accuracy for no bandwidth.

    Args:
        key: Checkpoint tensor name; the suffix decides the exclusion.
        w: The tensor as loaded, before any cast.

    Returns:
        ``True`` when ``w`` should be stored as a :class:`QWeight`.
    """
    return w.dim() == 2 and not key.endswith("conv1d.weight")


class QWeight:
    """An ``[N, K]`` weight held as int8 plus a per-row fp32 scale.

    The two tensors travel together: ``ops/gemv.py`` and ``ops/embed.py`` read
    ``wq`` and ``scale`` directly rather than going through a method, so any
    replacement object must expose the same pair.
    """

    __slots__ = ("wq", "scale", "n", "k")

    def __init__(self, wq: torch.Tensor, scale: torch.Tensor) -> None:
        """Wrap an already-quantised matrix; it is not re-quantised here.

        Args:
            wq: ``[N, K]`` int8 values. Public attributes, not copies: the
                caller must hand over owned, contiguous storage.
            scale: ``[N]`` fp32 absmax scales, one per output row.
        """
        self.wq = wq                      # [N, K] int8, contiguous
        self.scale = scale                # [N] fp32, contiguous
        self.n, self.k = wq.shape

    @classmethod
    def quantize(cls, w: torch.Tensor) -> "QWeight":
        """Quantise a dense ``[N, K]`` weight, symmetric per output row.

        The absmax scale is clamped at ``1e-8`` so an all-zero row divides
        instead of producing ``inf``/NaN; such a row then quantises to zeros and
        decodes back to zeros.

        Args:
            w: Dense weight, row-major ``[N, K]``, any float dtype.

        Returns:
            The int8 rows and their absmax scales, both made contiguous so the
            HIP kernels can take a flat pointer over them.
        """
        wf = w.float()
        scale = wf.abs().amax(-1).clamp_min(1e-8) / 127.0
        q = (wf / scale[:, None]).round().clamp_(-127, 127).to(torch.int8)
        return cls(q.contiguous(), scale.contiguous())

    def dequant(self, dtype: torch.dtype = D) -> torch.Tensor:
        """Materialise weights in ``dtype``; used by the prefill GEMM path only.

        fp16 is what prefill asks for: this part has no packed bf16 math, so the
        same GEMM runs ~7x faster on fp16 weights.

        Args:
            dtype: Output dtype; the caller picks it to match the GEMM it feeds.

        Returns:
            A fresh dense ``[N, K]`` tensor. It is not cached, so calling this
            in the decode loop would rebuild the weight every step.
        """
        return (self.wq.float() * self.scale[:, None]).to(dtype)

    def rows(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather rows: returns ``(int8 [T, K], fp32 scale [T])`` for embeddings.

        Args:
            idx: Row indices to gather, in the order the caller wants them.

        Returns:
            The matching int8 rows and their scales, both still scaled per row
            so the caller can dequantise only the rows it looked up.
        """
        return self.wq.index_select(0, idx), self.scale.index_select(0, idx)

    def nbytes(self) -> int:
        """Resident size in bytes, for the loader's VRAM accounting.

        Returns:
            The int8 rows plus their fp32 row scales. The dense weight this
            replaced would have cost twice the rows' element count in bf16.
        """
        return self.wq.numel() + self.scale.numel() * 4

    # Excluded from coverage: nothing in the runtime calls a repr, so a test that
    # exercised it would only be testing f-strings.
    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"QWeight({self.n}x{self.k} int8)"


def quantize_state_dict(sd: dict) -> dict:
    """Replace every quantisable 2-D weight in ``sd`` with a :class:`QWeight`.

    Mutates ``sd`` in place and drops each bf16 tensor as soon as it has been
    quantised (so the resident copy shrinks instead of doubling: 9 GB of bf16
    weights become ~4.5 GB, which is what leaves room on a 16 GB card for a longer
    context and a paged KV pool).

    ``load_weights`` quantises during the load instead and does not use this; it
    stays for a caller that already holds bf16 tensors in VRAM.

    Args:
        sd: State dict, consumed and rewritten in place.

    Returns:
        A byte-count report, ``{"bf16_bytes": ..., "int8_bytes": ...}``, for the
        loader's startup log. It counts only the tensors that were converted, so
        the pair difference is the VRAM the conversion actually freed.
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
    """Report which entries of a state dict are int8, for the loader's log.

    Args:
        sd: State dict, either as loaded or already converted by
            :func:`quantize_state_dict`.

    Returns:
        The quantised names in dict order; empty for a fully dense state dict.
    """
    return [k for k, v in sd.items() if isinstance(v, QWeight)]
