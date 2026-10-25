"""Qwen3.5-4B vision tower: the 24-layer ViT that turns pixels into LLM tokens.

Module: ``coffin/models/qwen35``. Leaf of the model family: it imports only
``torch`` and ``int8_weights``, and is built by ``models.qwen35.model`` out of
``weights.load_weights`` tensors. It must not import ``ops``, ``engine_api`` or
the text-side modules, so ``tests/test_vision.py`` can build a tiny tower with
no checkpoint and no engine around it.

Shape flow (``N = Σ t*h*w`` patch tokens, ``n_img = N / 4`` LLM tokens)::

    pixel_values [N, 1536]        processor output, rows in merge-block order
      -> patch_embed              F.linear (= Conv3d k=s=(2,16,16))   -> [N, 1024]
      -> + pos_embed resample     48x48 learned table -> per-image (h,w) grid
      -> 24 x block               LN -> attn(axial rope, per-image) -> LN -> MLP
      -> merger                   LN(1024) -> 4-in-1 view(4096) -> fc1 -> GELU
                                  (exact) -> fc2                       -> [N/4, 2560]

Constraints:
    * Compute is fp16 on the DCU, not bf16: this part has packed fp16 math and
      no packed bf16 math (13-20 TFLOP/s vs 2.3-7 for the same GEMM, the same
      reason the text projections load as ``ops.dtypes.WF``), and the tower is
      compute-bound.
    * The fp32 islands must stay fp32 -- the pos-embed tap sum, the rotary
      angle, the attention softmax. They are the ones HF has too, because a
      bf16/fp16 tap sum is 1e-2 wrong (``.tmp_dsh/vision/reference_spec.md``
      §14.3).
    * Token order is merge-block order everywhere except inside the attention:
      the processor emits ``pixel_values`` rows that way, and both the pos-embed
      permute and the merger's 4-in-1 view depend on it. A row-major stream here
      silently produces a plausible but wrong image.
    * Attention is per-frame, per-image and non-causal; segments are isolated by
      slicing, never by a mask, because the tower has no mask input and an
      ``[N, N]`` additive mask would be its largest tensor.
    * Never dequantise the whole int8 tower at once: one GEMM at a time keeps
      the peak transient at a single fp16 matrix (33 MB for the merger's fc1).

Measured against ``transformers`` 4.57.6 ``Qwen3VLVisionModel`` on the real
checkpoint (canonical acceptance, DCU, one image per size)::

    size   engine(fp32)  engine(fp16)  engine(fp16)     time   tower-only
           vs HF(fp32)   vs HF(fp32)   vs HF(bf16)              peak
    256^2      8.7e-6       7.1e-3       6.4e-2         67 ms   ~0.67 GiB
    512^2      1.2e-5       1.3e-2       7.9e-2        129 ms   ~0.71 GiB
    1024^2     1.4e-5       1.4e-2       9.0e-2        756 ms   ~0.85 GiB

The first column is the correctness gate: formula error is O(1), so 1e-5 sits a
million times below anything a wrong merge order, rope layout or LN placement
could hide.  The second is fp16's intrinsic cost, and it matches HF's *own* fp16
run against its fp32 run (1.4e-2) -- the dtype, not the implementation.  The
third is a smoke column only: bf16 rounds ~8e-3 per layer and this tower stacks
24 layers of activations reaching 1e3, so a bf16 oracle is 1e-1 from fp32 by
itself and cannot gate anything at 1e-3.

1024^2 is 756 ms against a 450 ms target.  The tower sustains 5.7 TFLOP/s there
(44% of this part's M=256 packed-fp16 ceiling) and the rate is flat from 512^2
to 1024^2, so the gap is per-op efficiency in the attention's ``score -> fp32
softmax -> PV`` round trip rather than launch count: at n=4096 attention is
38.5% of the 4.28 TFLOP forward and needs 6.4 G exp() calls.
``.tmp_dsh/vision2/budget.py`` prints that budget; the levers are fp16 softmax,
a larger ``Q_TILE``, and a fused attention kernel.

The int8 mode is for the memory-constrained case only: the tower is
compute-bound, so ``vision_quant`` buys 0.315 GiB instead of 0.621 GiB of
resident weights and pays for it in both error and time (W8A16 on top of a tower
already 1.4e-2 from fp32 -- measure with ``probe_parity.py --int8`` rather than
assuming it is free).
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F

from int8_weights import QWeight

#: Checkpoint prefix of every tensor this module reads.
VISUAL_PREFIX = "model.visual."

#: Query-tile width of the vision attention.  The score tile ``[heads, Q_TILE,
#: n]`` is the largest activation of the whole tower -- at 1 MP (n = 4096) the
#: untiled block is 0.5 GiB fp16 and 1 GiB once softmax lifts it to fp32, which
#: is why tiling is a hard requirement rather than a tuning knob.  256 keeps the
#: transient at 33 MB fp16 + 67 MB fp32 per tile (peak well inside 0.4 GiB) at
#: 16 tiles x 2 matmuls per image per layer, i.e. ~750 launches for 24 layers.
Q_TILE = int(os.environ.get("VISION_Q_TILE", "256"))

#: Token tile for the rotary application, which is done in fp32 like HF.  Keeps
#: its temporaries at ~4 MB instead of ``N*heads*head_dim*4`` (16 MB at 1 MP).
_ROPE_TILE = 1024

#: Rows per chunk when dequantising an int8 weight.  ``QWeight.dequant`` makes an
#: fp32 copy of the whole matrix (67 MB for a 4096x4096); chunking bounds that
#: transient at 8 MB so the fp16 result (33 MB) stays the peak.
_DEQUANT_ROWS = 512


def is_vision_quantizable(key: str, w: torch.Tensor) -> bool:
    """Select the tower's GEMM weights, the only ones int8 helps.

    ``pos_embed.weight`` is a row-gather table read by index (not a GEMM) and
    ``patch_embed.proj.weight`` is a 5-D conv kernel; both stay fp16.  Quantising
    them would put a dequantise on the gather / first-layer path for a 7.8 MB win,
    so the rule is "2-D ``.weight``, not the gather table".

    In numbers: 24 blocks x 4 matmuls + the merger's 2 = 98 of the 297 tensors
    are quantised, which is 329.25 M of the tower's 333.51 M parameters.  Resident
    that is 0.307 GiB int8 + 0.008 GiB of fp16 1-D/norm/gather leftovers, i.e.
    **0.315 GiB against fp16's 0.621 GiB**.

    Args:
        key: Checkpoint name of the tensor, used to exclude the gather table.
        w: The loaded tensor; only its rank is inspected.

    Returns:
        ``True`` when the caller should keep this weight as int8. The decision
        is policy, not measurement: an unmoved call site here means the tower
        loads twice its needed resident bytes.
    """
    return w.dim() == 2 and key.endswith(".weight") and "pos_embed" not in key


#: 2-D matmuls per block: ``attn.qkv``, ``attn.proj``, ``mlp.linear_fc1``,
#: ``mlp.linear_fc2``.  The biases and both LayerNorms are 1-D; the merger adds
#: ``linear_fc1`` + ``linear_fc2``.  24*4 + 2 = 98 quantisable weights.
GEMMS_PER_BLOCK = 4


def _cfg_get(cfg, key, default=None):
    """Read one field from either the raw ``config.json`` dict or a config object.

    Args:
        cfg: A ``dict`` from ``config.json`` or an object with attributes; both
            shapes reach this module, depending on who builds the tower.
        key: Field name.
        default: Value when the field is absent, which is the normal case for
            every key the checkpoint does not spell out.

    Returns:
        The field value, or ``default``.
    """
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class _VisionConfig:
    """``config.json:vision_config`` with the defaults transformers fills in.

    The checkpoint has no ``rope_parameters`` for the vision tower; transformers
    4.57.6 fills ``{rope_type: "axial", rope_theta: 10000.0}`` from
    ``Qwen3_5VisionConfig.default_rope_type`` / ``RotaryEmbeddingConfigMixin``.
    The defaults below are that fill, so a checkpoint that omits a field lands on
    the values HF would have used rather than on an invented geometry.

    Raises at construction time for any checkpoint geometry the tower cannot
    express -- those are configuration errors, and failing here names the field
    instead of surfacing as a wrong image several layers later.
    """

    def __init__(self, cfg) -> None:
        """Normalise one ``vision_config`` into the tower's fixed vocabulary.

        Args:
            cfg: ``config.json:vision_config`` as a dict or an object.

        Raises:
            ValueError: The checkpoint asks for deepstack visual indexes, a
                non-axial rope, an unsupported activation, or a pos-embed table
                that is not a perfect square. Each is a checkpoint this engine
                does not implement; reject it at load rather than decode.
        """
        def g(key, default=None):
            """Read one config field through ``_cfg_get``, keeping the call sites short.

            Args:
                key: Field name in ``vision_config``.
                default: Value when the checkpoint omits the field, i.e. the
                    transformers fill for that key.

            Returns:
                The field value, or ``default``.
            """
            return _cfg_get(cfg, key, default)

        self.depth = int(g("depth", 24))
        self.hidden_size = int(g("hidden_size", 1024))
        self.intermediate_size = int(g("intermediate_size", 4096))
        self.num_heads = int(g("num_heads") or g("num_attention_heads") or 16)
        self.in_channels = int(g("in_channels", 3))
        self.patch_size = int(g("patch_size", 16))
        self.spatial_merge_size = int(g("spatial_merge_size", 2))
        self.temporal_patch_size = int(g("temporal_patch_size", 2))
        self.num_position_embeddings = int(g("num_position_embeddings", 2304))
        self.out_hidden_size = int(g("out_hidden_size", 2560))
        self.hidden_act = str(g("hidden_act", "gelu_pytorch_tanh"))
        self.layer_norm_eps = float(g("layer_norm_eps", 1e-6))
        rope = g("rope_parameters") or {}
        self.rope_type = str(rope.get("rope_type", "axial"))
        self.rope_theta = float(rope.get("rope_theta", 10000.0))
        deepstack = g("deepstack_visual_indexes") or []
        if deepstack:
            raise ValueError(
                f"this checkpoint asks for deepstack_visual_indexes={deepstack}; "
                "the engine's tower returns last_hidden/merger output only")
        if self.rope_type != "axial":
            raise ValueError(f"vision rope_type must be axial, got {self.rope_type!r}")
        if self.hidden_act not in ("gelu_pytorch_tanh", "gelu"):
            raise ValueError(f"unsupported vision hidden_act {self.hidden_act!r}")
        self.head_dim = self.hidden_size // self.num_heads

        side = math.isqrt(self.num_position_embeddings)
        if side * side != self.num_position_embeddings:
            # Silently rounding would mis-lay the whole pos-embed table (spec §14.6).
            raise ValueError(
                f"num_position_embeddings={self.num_position_embeddings} is not a "
                "perfect square; the learned pos-embed grid is unknown")
        self.num_grid_per_side = side
        self.merge_unit = self.spatial_merge_size * self.spatial_merge_size

    @property
    def patch_dim(self) -> int:
        """Width of one ``pixel_values`` row: ``C * tps * p * p`` (1536 here).

        Returns:
            The flattened patch width the processor emits, which ``forward``
            validates the input against before any GEMM runs.
        """
        return self.in_channels * self.temporal_patch_size * self.patch_size ** 2


class _Block:
    """One ViT block's tensors: two LayerNorms, a packed attention, an MLP."""

    __slots__ = ("norm1_w", "norm1_b", "norm2_w", "norm2_b",
                 "qkv_w", "qkv_b", "proj_w", "proj_b",
                 "fc1_w", "fc1_b", "fc2_w", "fc2_b")


class VisionTower:
    """The checkpoint's ``model.visual.*`` tensors as a runnable fp16 tower.

    ``sd`` may key the tensors either with the ``model.visual.`` prefix (what
    :func:`models.qwen35.weights.load_weights` returns) or stripped; both are
    accepted.  With ``quant=True`` -- or when ``sd`` already holds
    :class:`~int8_weights.QWeight` matrices, which is what a ``vision_quant``
    load produces -- the GEMM weights are int8 + per-row fp32 scale and are
    dequantised one GEMM at a time.

    ``dtype`` is what :meth:`forward` returns (the model casts it to bf16 for the
    text side anyway).  The tower itself computes in fp16 on CUDA whatever
    ``dtype`` says, for the packed-math reason in the module docstring; on CPU
    (the numeric tests) it computes in ``dtype`` so fp32 works.  Pass
    ``compute_dtype=`` to override that rule explicitly.
    """

    def __init__(self, sd: dict, cfg, dtype: torch.dtype = torch.float16,
                 device="cuda", quant: bool = False,
                 compute_dtype: torch.dtype | None = None) -> None:
        """Bind one checkpoint to ``device`` and pick the compute dtype.

        Args:
            sd: ``model.visual.*`` tensors, keyed with or without the
                ``model.visual.`` prefix. A ``vision_quant`` load already holds
                :class:`~int8_weights.QWeight` matrices, which turns on the int8
                path by presence rather than by flag.
            cfg: ``config.json:vision_config`` as a dict or an object.
            dtype: What :meth:`forward` returns. The model casts it to bf16 for
                the text side, so this is a boundary dtype, not a compute one.
            device: Target device, string or ``torch.device``.
            quant: Force int8 GEMM weights even when ``sd`` is all fp16. Costs
                the 1.4e-2 fp16 error plus W8A16 on top, so it is for the
                memory-constrained case only.
            compute_dtype: Override the default rule, which is fp16 on CUDA for
                the packed-math reason in the module docstring and ``dtype`` on
                CPU so the fp32 numeric tests can run.

        Raises:
            KeyError: The state dict is missing tensors the forward pass reads;
                the error names the count and the first four so a partial
                download is obvious.
            ValueError: ``cfg`` describes a tower this engine does not implement
                (see :class:`_VisionConfig`).
        """
        self.cfg = _VisionConfig(cfg)
        c = self.cfg
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        on_cuda = self.device.type == "cuda"
        if compute_dtype is None:
            compute_dtype = (torch.float16
                             if on_cuda and dtype not in (torch.float32, torch.float64)
                             else dtype)
        self.compute_dtype = compute_dtype
        self.dtype = dtype
        self.quant = bool(quant) or any(isinstance(v, QWeight) for v in sd.values())
        self.scaling = c.head_dim ** -0.5

        self._sd = {k[len(VISUAL_PREFIX):] if k.startswith(VISUAL_PREFIX) else k: v
                    for k, v in sd.items()}
        self._check_keys()

        # patch embed: the checkpoint's Conv3d k=s=(2,16,16) is exactly a GEMM
        # over 1536 inputs once the kernel is reshaped, so no conv runs here.
        self.patch_w = self._tensor("patch_embed.proj.weight").reshape(
            c.hidden_size, c.patch_dim)
        self.patch_b = self._tensor("patch_embed.proj.bias")

        # Learned pos-embed table: a row-gather read by index, never quantised.
        self.pos_embed = self._tensor("pos_embed.weight")

        # Axial rope: head_dim//4 = 16 frequencies per axis, base 10000. Kept
        # fp32 because the angle is one of the fp32 islands.
        spatial_dim = c.head_dim // 2
        self.inv_freq = 1.0 / (
            c.rope_theta ** (torch.arange(0, spatial_dim, 2, dtype=torch.float32,
                                          device=self.device) / spatial_dim))

        self.blocks: list[_Block] = []
        for i in range(c.depth):
            p = f"blocks.{i}."
            blk = _Block()
            blk.norm1_w = self._tensor(p + "norm1.weight")
            blk.norm1_b = self._tensor(p + "norm1.bias")
            blk.norm2_w = self._tensor(p + "norm2.weight")
            blk.norm2_b = self._tensor(p + "norm2.bias")
            blk.qkv_w = self._tensor(p + "attn.qkv.weight", quantizable=True)
            blk.qkv_b = self._tensor(p + "attn.qkv.bias")
            blk.proj_w = self._tensor(p + "attn.proj.weight", quantizable=True)
            blk.proj_b = self._tensor(p + "attn.proj.bias")
            blk.fc1_w = self._tensor(p + "mlp.linear_fc1.weight", quantizable=True)
            blk.fc1_b = self._tensor(p + "mlp.linear_fc1.bias")
            blk.fc2_w = self._tensor(p + "mlp.linear_fc2.weight", quantizable=True)
            blk.fc2_b = self._tensor(p + "mlp.linear_fc2.bias")
            self.blocks.append(blk)

        # Merger applies LayerNorm(1024) *before* the 2x2 shuffle, not after.
        self.merger_norm_w = self._tensor("merger.norm.weight")
        self.merger_norm_b = self._tensor("merger.norm.bias")
        self.merger_fc1_w = self._tensor("merger.linear_fc1.weight", quantizable=True)
        self.merger_fc1_b = self._tensor("merger.linear_fc1.bias")
        self.merger_fc2_w = self._tensor("merger.linear_fc2.weight", quantizable=True)
        self.merger_fc2_b = self._tensor("merger.linear_fc2.bias")
        self.merge_dim = c.hidden_size * c.merge_unit

        # (h,w)-keyed resampling/rope tables: they depend on the patch grid only,
        # never on the pixels, so a server pays for each new image size once
        # (spec §14.4).
        self._pos_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self._hw_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self._rope_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        del self._sd

    def _check_keys(self) -> None:
        """Reject a state dict that does not hold every tensor the forward reads.

        Checked up front because the alternative is an ``AttributeError`` from a
        ``_Block.__slots__`` field half-way through the first image, after the
        weights are already resident. The error names the count and the first
        four missing keys, which is enough to tell a partial download from a
        wrong checkpoint.

        Raises:
            KeyError: Any required key is absent. Re-load the tower; a partial
                state dict cannot be patched at run time.
        """
        c = self.cfg
        want = {"patch_embed.proj.weight", "patch_embed.proj.bias", "pos_embed.weight",
                "merger.norm.weight", "merger.norm.bias",
                "merger.linear_fc1.weight", "merger.linear_fc1.bias",
                "merger.linear_fc2.weight", "merger.linear_fc2.bias"}
        for i in range(c.depth):
            for suffix in ("norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias",
                           "attn.qkv.weight", "attn.qkv.bias",
                           "attn.proj.weight", "attn.proj.bias",
                           "mlp.linear_fc1.weight", "mlp.linear_fc1.bias",
                           "mlp.linear_fc2.weight", "mlp.linear_fc2.bias"):
                want.add(f"blocks.{i}.{suffix}")
        missing = sorted(want - set(self._sd))
        if missing:
            raise KeyError(
                f"vision state dict is missing {len(missing)} tensors, first: {missing[:4]}")

    def _tensor(self, key: str, quantizable: bool = False):
        """Move one tensor to ``device`` in the compute dtype, or keep it int8.

        1-D tensors (LayerNorm weights, biases) stay in the compute dtype: they
        are consumed elementwise by ``F.layer_norm`` / added to a fp16 output, so
        a dequantise would add a launch per block to save 0.3 MB in total.

        Args:
            key: State-dict key with the ``model.visual.`` prefix already stripped.
            quantizable: Allow int8 when :attr:`quant` is on. Only the GEMM
                weights carry this flag; see :func:`is_vision_quantizable`.

        Returns:
            The tensor on ``device``, either in the compute dtype or as a
            :class:`~int8_weights.QWeight` that ``_linear`` dequantises per call.
        """
        t = self._sd[key]
        if isinstance(t, QWeight):
            return t
        t = t.to(device=self.device, dtype=self.compute_dtype).contiguous()
        if quantizable and self.quant:
            q = QWeight.quantize(t)
            del t
            return q
        return t

    def _dequant(self, q: QWeight) -> torch.Tensor:
        """Expand one int8 matrix to the compute dtype, in row chunks.

        Chunking is what keeps the peak transient at one chunk plus the fp16
        result instead of a whole fp32 copy (see ``_DEQUANT_ROWS``); the row
        slices are contiguous, so the extra loop costs no gather.

        Args:
            q: int8 weight, ``[n, k]`` with a per-row fp32 scale.

        Returns:
            The dense ``[n, k]`` matrix in the compute dtype, ready for
            ``F.linear``. Caller-owned; freed on the next GEMM.
        """
        if q.n <= _DEQUANT_ROWS:
            return q.dequant(self.compute_dtype)
        out = torch.empty((q.n, q.k), dtype=self.compute_dtype, device=self.device)
        for r0 in range(0, q.n, _DEQUANT_ROWS):
            r1 = min(r0 + _DEQUANT_ROWS, q.n)
            out[r0:r1] = (q.wq[r0:r1].to(torch.float32)
                          * q.scale[r0:r1, None]).to(self.compute_dtype)
        return out

    def _linear(self, x: torch.Tensor, w, b) -> torch.Tensor:
        """``F.linear`` that accepts either a dense weight or an int8 ``QWeight``.

        Args:
            x: Activations, ``[N, in]``.
            w: Dense weight, or a :class:`~int8_weights.QWeight` dequantised
                just for this call.
            b: Bias in the compute dtype, or ``None``.

        Returns:
            ``[N, out]`` in the activation dtype.
        """
        if isinstance(w, QWeight):
            w = self._dequant(w)
        return F.linear(x, w, b)

    def _ln(self, x: torch.Tensor, w, b) -> torch.Tensor:
        """LayerNorm with the checkpoint's ``layer_norm_eps`` and HF's biased variance.

        Args:
            x: Activations, ``[N, hidden]``.
            w: Gain, ``[hidden]``.
            b: Bias, ``[hidden]``.

        Returns:
            ``[N, hidden]`` normalised in place-free fashion.
        """
        return F.layer_norm(x, (x.shape[-1],), w, b, self.cfg.layer_norm_eps)

    def _pos_taps(self, h: int, w: int) -> tuple[torch.Tensor, torch.Tensor]:
        """``(h*w, 4)`` gather indices and fp32 weights, in row-major patch order.

        Reproduces ``Qwen3VLVisionModel.fast_pos_embed_interpolate``: the table is
        a ``side x side`` grid resampled with ``torch.linspace(0, side-1, size)``
        (align_corners=True) and a 2-tap hat kernel per axis.  The merge-block
        permute is applied to the *result* (see :meth:`_pos_offsets`), which is
        the same thing HF's ``view/permute/flatten`` does and keeps the tap
        tables identical for every image of a given size.

        Args:
            h: Patch-grid rows of one image.
            w: Patch-grid columns of one image.

        Returns:
            ``(idx, wt)`` for the row-major grid: four table offsets per patch
            in ``idx`` and their fp32 bilinear weights in ``wt``, which sum to 1.
            Both are cached per ``(h, w)`` because they depend on the grid only,
            never on the pixels.
        """
        hit = self._pos_cache.get((h, w))
        if hit is not None:
            return hit
        side = self.cfg.num_grid_per_side
        hs = torch.linspace(0, side - 1, h, device=self.device, dtype=torch.float32)
        ws = torch.linspace(0, side - 1, w, device=self.device, dtype=torch.float32)
        h0 = hs.to(torch.int64)                       # >= 0, so trunc == floor
        w0 = ws.to(torch.int64)
        h1 = (h0 + 1).clamp(max=side - 1)
        w1 = (w0 + 1).clamp(max=side - 1)
        dh = hs - h0.to(torch.float32)
        dw = ws - w0.to(torch.float32)
        idx = torch.stack([
            h0[:, None] * side + w0[None, :],
            h0[:, None] * side + w1[None, :],
            h1[:, None] * side + w0[None, :],
            h1[:, None] * side + w1[None, :],
        ], dim=-1).reshape(-1, 4)
        wt = torch.stack([
            (1 - dh)[:, None] * (1 - dw)[None, :],
            (1 - dh)[:, None] * dw[None, :],
            dh[:, None] * (1 - dw)[None, :],
            dh[:, None] * dw[None, :],
        ], dim=-1).reshape(-1, 4)
        if len(self._pos_cache) > 32:
            self._pos_cache.clear()
        self._pos_cache[(h, w)] = (idx, wt)
        return idx, wt

    def _pos_offsets(self, gt: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Per-token learned pos-embed, summed in fp32 (spec §14.3), in token order.

        The accumulator stays fp32 for the whole tap sum because a fp16/bf16 tap
        sum is 1e-2 wrong: four table rows with bilinear weights are added and
        then permuted, and the rounding of the intermediate sum cannot be undone
        by casting the result later.

        Args:
            gt: ``grid_thw``, ``[n_img, 3]`` rows of ``(t, h, w)`` patch counts.
            dtype: Dtype to return, normally the compute dtype of ``x``.

        Returns:
            ``[N, hidden]`` positional embedding, already permuted into
            merge-block order and repeated over frames, so it adds to the patch
            embeddings row for row.

        Raises:
            ValueError: A grid side is not a multiple of ``spatial_merge_size``.
                The 2x2 merge would leave a partial block whose rows the merger
                cannot fold, so the caller must resize the image, not the grid.
        """
        c = self.cfg
        m, hidden = c.spatial_merge_size, c.hidden_size
        parts = []
        for t, h, w in gt.tolist():
            t, h, w = int(t), int(h), int(w)
            if h % m or w % m:
                raise ValueError(f"patch grid {h}x{w} is not a multiple of merge {m}")
            idx, wt = self._pos_taps(h, w)
            acc = torch.zeros(h * w, hidden, dtype=torch.float32, device=self.device)
            for k in range(4):
                gathered = self.pos_embed.index_select(0, idx[:, k]).to(torch.float32)
                acc.addcmul_(gathered, wt[:, k:k + 1])
                del gathered
            # Row-major (h, w) -> merge-block order (block_row, block_col,
            # in_row, in_col). This is the order pixel_values already use, so
            # skipping the permute misaligns every patch from here on.
            acc = (acc.view(h // m, m, w // m, m, hidden)
                   .permute(0, 2, 1, 3, 4).reshape(h * w, hidden))
            if t > 1:
                acc = acc.repeat(t, 1)
            parts.append(acc)
        out = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
        return out.to(dtype)

    def _hw_positions(self, h: int, w: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Patch-grid (row, col) of every token, in merge-block order.

        Args:
            h: Patch-grid rows of one image.
            w: Patch-grid columns of one image.

        Returns:
            ``(rows, cols)`` as int64 ``[h*w]`` in merge-block order. The rope
            angles are derived from these, so they must use the same order as
            the token stream; cached per ``(h, w)``.
        """
        hit = self._hw_cache.get((h, w))
        if hit is not None:
            return hit
        m = self.cfg.spatial_merge_size
        gh, gw = h // m, w // m
        block_rows = torch.arange(gh, device=self.device)
        block_cols = torch.arange(gw, device=self.device)
        intra = torch.arange(m, device=self.device)
        rows = (block_rows[:, None, None, None] * m
                + intra[None, None, :, None]).expand(gh, gw, m, m).reshape(-1)
        cols = (block_cols[None, :, None, None] * m
                + intra[None, None, None, :]).expand(gh, gw, m, m).reshape(-1)
        if len(self._hw_cache) > 32:
            self._hw_cache.clear()
        self._hw_cache[(h, w)] = (rows, cols)
        return rows, cols

    def _rope(self, gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """fp32 cos/sin ``[N, head_dim]`` for the whole batch, in token order.

        Layout is ``[cos_h | cos_w | cos_h | cos_w]`` (16 frequencies each): with
        ``rotate_half`` pairing ``i`` with ``i + head_dim/2`` that is 32
        independent 2D rotations, H for the first 16 and W for the next 16.

        Args:
            gt: ``grid_thw``, ``[n_img, 3]``. The whole tensor keys the cache,
                so two requests with the same images share one table.

        Returns:
            ``(cos, sin)``, each fp32 ``[N, head_dim]``, already repeated over
            frames and concatenated across images in ``gt`` row order. fp32
            because the angle is one of the fp32 islands HF also keeps.
        """
        key = tuple(int(v) for v in gt.reshape(-1).tolist())
        hit = self._rope_cache.get(key)
        if hit is not None:
            return hit
        cos_parts, sin_parts = [], []
        for t, h, w in gt.tolist():
            rows, cols = self._hw_positions(int(h), int(w))
            fh = rows.to(torch.float32)[:, None] * self.inv_freq
            fw = cols.to(torch.float32)[:, None] * self.inv_freq
            ch, sh = fh.cos(), fh.sin()
            cw, sw = fw.cos(), fw.sin()
            cos = torch.cat([ch, cw, ch, cw], dim=-1)
            sin = torch.cat([sh, sw, sh, sw], dim=-1)
            if int(t) > 1:
                cos, sin = cos.repeat(int(t), 1), sin.repeat(int(t), 1)
            cos_parts.append(cos)
            sin_parts.append(sin)
        out = (cos_parts[0] if len(cos_parts) == 1 else torch.cat(cos_parts, dim=0),
               sin_parts[0] if len(sin_parts) == 1 else torch.cat(sin_parts, dim=0))
        if len(self._rope_cache) > 16:
            self._rope_cache.clear()
        self._rope_cache[key] = out
        return out

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Swap the two halves of the last dim with a sign flip on the first.

        Interleaved rope (``-x2, x1``) instead of HF's ``rotate_half`` variant
        would still produce unit-norm rotations, so nothing downstream would
        raise -- the image would just be wrong. This is HF's split-half form.

        Args:
            x: fp32 tensor whose last dim is ``head_dim``.

        Returns:
            Same shape, rotated half.
        """
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor,
                    cos: torch.Tensor, sin: torch.Tensor):
        """Rotate q/k in fp32 then cast back -- HF's ``apply_rotary_pos_emb_vision``.

        Tiled over tokens so the fp32 copies stay ~4 MB instead of N*heads*64*4.

        Args:
            q: Queries, ``[N, heads, head_dim]`` in token order.
            k: Keys, same shape.
            cos: fp32 ``[N, head_dim]`` from :meth:`_rope`.
            sin: fp32 ``[N, head_dim]`` from :meth:`_rope`.

        Returns:
            ``(q_out, k_out)`` with the input shapes and dtypes restored. The
            buffers are preallocated so the tile loop never grows the peak.
        """
        q_dtype, k_dtype = q.dtype, k.dtype
        q_out = torch.empty_like(q)
        k_out = torch.empty_like(k)
        for i0 in range(0, q.shape[0], _ROPE_TILE):
            i1 = min(i0 + _ROPE_TILE, q.shape[0])
            # [tile, 1, head_dim]: the singleton broadcasts over heads.
            c = cos[i0:i1].unsqueeze(1)
            s = sin[i0:i1].unsqueeze(1)
            qf = q[i0:i1].to(torch.float32)
            q_out[i0:i1] = (qf * c + self._rotate_half(qf) * s).to(q_dtype)
            del qf
            kf = k[i0:i1].to(torch.float32)
            k_out[i0:i1] = (kf * c + self._rotate_half(kf) * s).to(k_dtype)
            del kf, c, s
        return q_out, k_out

    def _segments(self, gt: torch.Tensor) -> list[tuple[int, int]]:
        """``[start, end)`` of every attention segment: one per frame, per image.

        Frames attend within themselves only, so a 2-frame image produces two
        segments, not one. Sliding these wrong lets one frame see another, which
        no numeric test would flag as an error.

        Args:
            gt: ``grid_thw``, ``[n_img, 3]``.

        Returns:
            Half-open ``(start, end)`` pairs covering ``[0, N)`` in order.
        """
        segs: list[tuple[int, int]] = []
        off = 0
        for t, h, w in gt.tolist():
            n = int(h) * int(w)
            for f in range(int(t)):
                segs.append((off + f * n, off + (f + 1) * n))
            off += int(t) * n
        return segs

    def cu_seqlens(self, gt: torch.Tensor) -> torch.Tensor:
        """``int32`` cumulative segment lengths (``[0, ..., N]``), FA-varlen style.

        Args:
            gt: ``grid_thw``, ``[n_img, 3]``.

        Returns:
            ``[n_seg + 1]`` int32 offsets on ``device``. int32 because that is
            what a varlen attention kernel takes; the tower itself slices with
            :meth:`_segments` and only reports this shape for parity checks.
        """
        bounds = [0] + [e for _s, e in self._segments(gt)]
        return torch.tensor(bounds, dtype=torch.int32, device=self.device)

    def _attn(self, blk: _Block, x: torch.Tensor, segs: list[tuple[int, int]],
              cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Packed non-causal attention, one query-tile of scores at a time.

        Segments are isolated by slicing, not by a mask (the tower has no mask
        input; a ``[N, N]`` additive mask would also be the biggest tensor in the
        tower).  Softmax runs in fp32 and is cast back to the compute dtype before
        the ``p @ v``, exactly like HF's eager path, so a query tile reproduces the
        full-score-block result to within fp32 summation order.

        The softmax is ``torch.softmax(..., dtype=fp32)`` rather than a hand-rolled
        max/exp/div: one fused kernel instead of three passes over a fp32 copy of
        the tile, and it is literatim what HF's ``eager_attention_forward`` does.

        Args:
            blk: The block whose ``qkv`` / ``proj`` weights this pass uses.
            x: Pre-norm activations, ``[N, hidden]``.
            segs: Segments from :meth:`_segments`.
            cos: fp32 ``[N, head_dim]`` rope table.
            sin: fp32 ``[N, head_dim]`` rope table.

        Returns:
            ``[N, hidden]`` attention output, ready to add to the residual. Tiles
            cover each segment separately, so the output is exact per segment
            while the score block never materialises whole.
        """
        c = self.cfg
        n = x.shape[0]
        heads, hd = c.num_heads, c.head_dim
        qkv = self._linear(x, blk.qkv_w, blk.qkv_b)
        q, k, v = qkv.view(n, 3, heads, hd).permute(1, 0, 2, 3).unbind(0)
        del qkv
        q, k = self._apply_rope(q, k, cos, sin)
        # [heads, n, hd]: batch the heads so each tile matmul is one GEMM.
        q = q.transpose(0, 1).contiguous()
        k = k.transpose(0, 1).contiguous()
        v = v.transpose(0, 1).contiguous()
        out = torch.empty_like(q)
        tile = max(1, min(Q_TILE, n))
        for s, e in segs:
            # [heads, hd, seg]: transposed once per segment, not per tile.
            kt = k[:, s:e].transpose(-1, -2).contiguous()
            vs = v[:, s:e]
            for q0 in range(s, e, tile):
                q1 = min(q0 + tile, e)
                scores = torch.matmul(q[:, q0:q1], kt) * self.scaling
                p = torch.softmax(scores, dim=-1, dtype=torch.float32)
                out[:, q0:q1] = torch.matmul(p.to(self.compute_dtype), vs)
                del scores, p
            del kt, vs
        del q, k, v
        # (heads, n, hd) -> (n, heads*hd): the head dim has to come back *inside*
        # the feature dim before proj, or the projection reads a permuted vector.
        return self._linear(out.transpose(0, 1).reshape(n, heads * hd),
                            blk.proj_w, blk.proj_b)

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor,
                grid_thw: torch.Tensor) -> torch.Tensor:
        """``pixel_values [N, 1536]`` + ``grid_thw [n_img, 3]`` -> ``[N/4, 2560]``.

        The output row order is the prompt's ``<|image_pad|>`` order: images in
        ``grid_thw`` row order, each one row-major over its *merged* grid, which
        is what ``masked_scatter`` (and the engine's span overwrite) expects.

        Args:
            pixel_values: Processor output, ``[N, 1536]``, rows in merge-block
                order. The width must be ``patch_dim``.
            grid_thw: ``[n_img, 3]`` rows of ``(t, h, w)`` patch counts.

        Returns:
            ``[N / 4, 2560]`` image embeddings in the tower's output dtype.
            ``N / 4`` because the merger folds each 2x2 patch block into one
            row; the callers replace the prompt's image-pad rows with these, so
            a count mismatch is an immediate shape error at scatter time.

        Raises:
            ValueError: ``grid_thw`` is not ``[n, 3]``, the row count disagrees
                with the grid, or the row width is not ``patch_dim``. All three
                are processor/engine mismatches; the caller should reject the
                request rather than feed a misaligned image to GEMMs.
        """
        c = self.cfg
        gt = torch.as_tensor(grid_thw, dtype=torch.long, device=self.device)
        if gt.dim() != 2 or gt.shape[1] != 3:
            raise ValueError(f"grid_thw must be [n, 3], got {tuple(gt.shape)}")
        pv = pixel_values.to(device=self.device, dtype=self.compute_dtype)
        n_expect = int(gt.prod(dim=-1).sum().item())
        if pv.shape[0] != n_expect:
            raise ValueError(f"pixel_values has {pv.shape[0]} rows, grid implies {n_expect}")
        if pv.shape[1] != c.patch_dim:
            raise ValueError(f"pixel_values row width {pv.shape[1]} != {c.patch_dim}")

        x = F.linear(pv, self.patch_w, self.patch_b)
        del pv
        x = x + self._pos_offsets(gt, x.dtype)
        cos, sin = self._rope(gt)
        segs = self._segments(gt)

        # Both GELUs are deliberately different: the blocks use the checkpoint's
        # ``gelu_pytorch_tanh`` approximation, the merger uses HF's exact erf
        # ``nn.GELU``. Swapping them costs ~1e-3 and no shape check would catch it.
        for blk in self.blocks:
            x = x + self._attn(blk, self._ln(x, blk.norm1_w, blk.norm1_b), segs, cos, sin)
            h = self._ln(x, blk.norm2_w, blk.norm2_b)
            h = self._linear(h, blk.fc1_w, blk.fc1_b)
            h = F.gelu(h, approximate="tanh")
            x = x + self._linear(h, blk.fc2_w, blk.fc2_b)
            del h

        y = self._ln(x, self.merger_norm_w, self.merger_norm_b)
        del x
        # 2x2 merge-block -> one row: the halving of the token count the prompt's
        # image-pad span already accounts for.
        y = y.reshape(-1, self.merge_dim)
        y = self._linear(y, self.merger_fc1_w, self.merger_fc1_b)
        # Default ``F.gelu`` is the exact erf form nn.GELU uses; passing
        # approximate="tanh" here would be the block activation, not the merger's.
        y = F.gelu(y)
        y = self._linear(y, self.merger_fc2_w, self.merger_fc2_b)
        return y.to(self.dtype)

    def __call__(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """Alias of :meth:`forward` so the tower reads like a module in the model.

        Args:
            pixel_values: ``[N, 1536]`` processor output.
            grid_thw: ``[n_img, 3]`` patch grids.

        Returns:
            ``[N / 4, 2560]`` image embeddings, same as :meth:`forward`.
        """
        return self.forward(pixel_values, grid_thw)

    def num_params(self) -> int:
        """Count the tower's parameters from the config, without touching storage.

        Exists because the fp16 and int8 modes hold the same parameter count but
        different bytes; a log line that quotes parameters must not have to walk
        (and so materialise) ``QWeight`` matrices.

        Returns:
            Total parameter count of the architecture the config describes.
        """
        c = self.cfg
        n = c.hidden_size * c.patch_dim + c.hidden_size                    # patch embed
        n += c.num_position_embeddings * c.hidden_size                     # pos embed
        h, i = c.hidden_size, c.intermediate_size
        per_block = (3 * h * h + 3 * h) + (h * h + h) + (h * i + i) + (i * h + h) + 4 * h
        n += c.depth * per_block
        merge = h * c.merge_unit
        n += 2 * h + (merge * merge + merge) + (c.out_hidden_size * merge + c.out_hidden_size)
        return n

    def nbytes(self) -> int:
        """Resident bytes of the weights (int8 mode is ~half of fp16).

        Returns:
            Bytes actually held, so int8 counts 1 byte per weight plus the fp32
            per-row scale -- the number the residency budget needs, not the
            parameter count times an element size.
        """
        total = 0
        tensors = [self.patch_w, self.patch_b, self.pos_embed,
                   self.merger_norm_w, self.merger_norm_b,
                   self.merger_fc1_w, self.merger_fc1_b,
                   self.merger_fc2_w, self.merger_fc2_b]
        for blk in self.blocks:
            for name in _Block.__slots__:
                tensors.append(getattr(blk, name))
        for t in tensors:
            total += t.nbytes() if isinstance(t, QWeight) else t.numel() * t.element_size()
        return total

    def summary(self) -> dict:
        """Startup log payload: geometry, dtypes and resident weight size.

        Returns:
            A JSON-friendly dict. ``q_tile`` is the effective tile and
            ``q_tile_env`` the variable that overrides it, so a mismatched
            ``VISION_Q_TILE`` in the deployment is visible in the log.
        """
        return {"params": self.num_params(), "weight_GiB": self.nbytes() / 1024 ** 3,
                "quant": self.quant, "dtype": str(self.dtype),
                "compute_dtype": str(self.compute_dtype), "device": str(self.device),
                "q_tile": Q_TILE, "q_tile_env": "VISION_Q_TILE"}
