"""One transformer layer: a mixer (linear or full attention) plus the MLP.

Which mixer a layer gets comes from ``cfg["layer_types"]``; the checkpoint
interleaves 24 linear-attention layers with 8 full-attention ones.

Module: ``coffin/models/qwen35``. Depends on ``ops`` (norm, gemv) and its
siblings ``full_attn``, ``linear_attn`` and ``batch``; assembled into a stack by
``model.py``.

Constraints:
    * The two mixer classes are interchangeable only through the methods below
      (``prefill_project`` / ``prefill_state`` / ``verify`` / ``batch``), so a new
      mixer has to implement the same surface; ``isinstance`` checks route the
      linear-attention-only cases.
    * Linear-attention layers carry recurrent state, full-attention layers carry
      KV. Every entry point here must reach the right one, or a chunked prefill
      silently advances only half the model.
    * The layer type is read from the checkpoint's ``layer_types`` list, never
      inferred from the layer index -- the interleaving is a model property.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.qwen35.batch import batch_proj, wide_proj
from models.qwen35.full_attn import FullAttn
from models.qwen35.linear_attn import LinearAttn
from ops.gemv import _proj
from ops.norm import RMSNorm


class MLP:
    """SwiGLU feed-forward block: ``down(silu(gate) * up)``.

    ``gate`` and ``up`` share one input, so they are built as a single fused
    :class:`~ops.gemv.GEMVGroup` and split afterwards; one wide weight read beats
    two narrow ones on every path (decode, prefill, batch).
    """

    def __init__(self, sd: dict, p: str, bench: bool) -> None:
        """Build the fused gate/up projection and the down projection.

        Args:
            sd: Checkpoint state dict.
            p: Parameter prefix of this MLP inside the state dict.
            bench: Forwarded to the projections; ``False`` skips the autotune.
        """
        self.gate_up = _proj(sd, [f"{p}.gate_proj", f"{p}.up_proj"], bench)
        self.down = _proj(sd, [f"{p}.down_proj"], bench)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block to one or a few rows (decode path).

        Args:
            x: Activations, ``[hidden]`` or ``[B, hidden]``, bf16.

        Returns:
            Same leading shape as ``x`` with the hidden width.
        """
        gate, up = self.gate_up(x)
        return self.down(F.silu(gate) * up)

    def prefill(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block to a prefill chunk.

        Args:
            x: Activations, ``[C, hidden]``, bf16.

        Returns:
            ``[C, hidden]``.
        """
        gate, up = self.gate_up.prefill(x)
        return self.down.prefill(F.silu(gate) * up)

    def batch(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block with the bit-exact batched GEMV path.

        Args:
            x: Activations, ``[B, hidden]``, one row per decoding sequence.

        Returns:
            ``[B, hidden]``, each row identical to the solo-decode result.
        """
        gate, up = batch_proj(self.gate_up, x)
        return batch_proj(self.down, F.silu(gate) * up)

    def wide(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block at an arbitrary row count (the one-forward verify).

        Args:
            x: Activations, ``[M, hidden]``; ``M`` is the verify width ``B*c``.

        Returns:
            ``[M, hidden]``; rows need not be bit-identical to solo decode once a
            wide kernel is in play.
        """
        gate, up = wide_proj(self.gate_up, x)
        return wide_proj(self.down, F.silu(gate) * up)


class Layer:
    """One decoder layer: pre-norm mixer, residual add, pre-norm MLP, residual add.

    The mixer is chosen by the checkpoint's ``layer_types`` entry for this index:
    ``linear_attention`` gets :class:`~models.qwen35.linear_attn.LinearAttn`,
    anything else :class:`~models.qwen35.full_attn.FullAttn`.
    """

    def __init__(self, sd: dict, i: int, cfg: dict, bench: bool, max_seq: int,
                 paged: bool = False) -> None:
        """Build the norms, the MLP and this layer's mixer.

        Args:
            sd: Checkpoint state dict.
            i: Layer index, used both for the parameter prefix and to look up
                ``layer_types[i]``.
            cfg: Model config dict from ``config.json``.
            bench: Forwarded to the projections; ``False`` skips the autotune.
            max_seq: Longest sequence the engine will run, used to size the
                full-attention mask and RoPE tables.
            paged: Build the mixer against the paged KV pool instead of a
                contiguous cache. Must match how the runner drives the model.
        """
        p = f"model.language_model.layers.{i}"
        self.type = cfg["layer_types"][i]
        self.in_ln = RMSNorm(sd[f"{p}.input_layernorm.weight"])
        self.post_ln = RMSNorm(sd[f"{p}.post_attention_layernorm.weight"])
        self.mlp = MLP(sd, f"{p}.mlp", bench)
        if self.type == "linear_attention":
            self.mix = LinearAttn(sd, f"{p}.linear_attn", cfg, bench)
        else:
            self.mix = FullAttn(sd, f"{p}.self_attn", cfg, bench, max_seq, paged)

    def __call__(self, x: torch.Tensor, ctx) -> torch.Tensor:
        """Run one decode step through the layer.

        Args:
            x: Hidden state, ``[hidden]``.
            ctx: Mixer inputs (cache slots, positions and the layer's own state);
                forwarded to the mixer as positional arguments.

        Returns:
            ``[hidden]`` output state.
        """
        h = self.mix(self.in_ln(x), *ctx)
        x = x + h
        return x + self.mlp(self.post_ln(x))

    def prefill(self, x: torch.Tensor, positions: torch.Tensor, ctx,
                tiles: list | None = None) -> torch.Tensor:
        """Run one prefill chunk through the layer, for a single sequence.

        The input projection is hoisted out of :meth:`prefill_state` so the
        stateful half can run per sequence while the GEMM stays one call.

        Args:
            x: Chunk activations, ``[C, hidden]``.
            positions: Absolute positions of the chunk, ``[C]``.
            ctx: Mixer inputs for this chunk.
            tiles: Pre-split query tiles, or ``None`` to let the mixer tile the
                chunk itself.

        Returns:
            ``[C, hidden]`` output activations.
        """
        proj = self.mix.prefill_project(self.in_ln(x))
        h = self.prefill_state(proj, positions, ctx, tiles, self.mix.prefill_row)
        x = x + h
        return x + self.mlp.prefill(self.post_ln(x))

    def prefill_state(self, proj, positions: torch.Tensor, ctx,
                      tiles: list | None, row: int) -> torch.Tensor:
        """Mixer half of a prefill chunk for one sequence on ``row``.

        Args:
            proj: Raw projection output for this chunk, as returned by the
                mixer's ``prefill_project``.
            positions: Absolute positions of the chunk, ``[C]``.
            ctx: Mixer inputs for this chunk.
            tiles: Pre-split query tiles, or ``None``.
            row: Recurrence/KV row this sequence owns in the batch layout; the
                prefill scratch row is the last one.

        Returns:
            ``[C, hidden]`` pre-residual mixer output.
        """
        mix = self.mix
        if isinstance(mix, LinearAttn):
            pre = mix.prefill_state(*proj, row=row)
            return mix.out_proj.prefill(pre)
        pre = mix.prefill_attend(*proj, positions, ctx, tiles, row)
        return mix.o_proj.prefill(pre)

    # group prefill: the projection is one GEMM over every sequence's tokens, the
    # stateful half runs per sequence with its own positions / KV row / recurrence
    # row, and the output projection is batched again over the concatenation.
    def prefill_group_project(self, x: torch.Tensor):
        """Project a concatenation of several sequences' prefill chunks.

        Args:
            x: Activations for every sequence's chunk, ``[C_total, hidden]``.

        Returns:
            ``(mixer, projection tuple)``: the mixer instance, so the per-sequence
            calls below do not have to re-dispatch on the layer type, plus the
            raw projection output.
        """
        mix = self.mix
        return mix, mix.prefill_project(self.in_ln(x))

    def prefill_group_state(self, mix, proj, positions: torch.Tensor, ctx,
                            tiles: list | None, row: int) -> torch.Tensor:
        """Run one sequence's slice of the group projection.

        Args:
            mix: Mixer returned by :meth:`prefill_group_project`.
            proj: That sequence's slice of the group projection.
            positions: Absolute positions of the slice, ``[C_i]``.
            ctx: Mixer inputs for this slice.
            tiles: Pre-split query tiles, or ``None``.
            row: Row this sequence owns in the batch layout.

        Returns:
            ``[C_i, D]``, before the output projection. Kept pre-projection so
            :meth:`prefill_group_out` can batch the GEMM over all sequences.
        """
        if isinstance(mix, LinearAttn):
            return mix.prefill_state(*proj, row=row)
        return mix.prefill_attend(*proj, positions, ctx, tiles, row)

    def prefill_group_out(self, mix, pre_out: torch.Tensor) -> torch.Tensor:
        """Apply the output projection once over all sequences' mixer outputs.

        Args:
            mix: Mixer returned by :meth:`prefill_group_project`.
            pre_out: Concatenated pre-projection outputs, ``[C_total, D]``.

        Returns:
            ``[C_total, hidden]``.
        """
        if isinstance(mix, LinearAttn):
            return mix.out_proj.prefill(pre_out)
        return mix.o_proj.prefill(pre_out)

    def verify(self, x: torch.Tensor, positions: torch.Tensor, ctx) -> torch.Tensor:
        """Replay the layer for the ``c`` speculative tokens of one step.

        Args:
            x: Hidden states for the verify window, ``[c, hidden]``.
            positions: Absolute positions of those tokens, ``[c]``.
            ctx: Mixer inputs for the window; the linear-attention mixer gets no
                positions because it carries its state instead.

        Returns:
            ``[c, hidden]`` output states.
        """
        if self.type == "linear_attention":
            h = self.mix.verify(self.in_ln(x))
        else:
            h = self.mix.verify(self.in_ln(x), positions, *ctx)
        x = x + h
        return x + self.mlp.prefill(self.post_ln(x))

    def batch(self, x: torch.Tensor, positions: torch.Tensor, ctx) -> torch.Tensor:
        """Run the layer for the static-batch decode step.

        Projections run batched (one weight read for all rows); the stateful
        mixers run the existing per-row kernels so every row keeps the exact
        numbers of a solo decode.

        Args:
            x: One row per decoding sequence, ``[B, hidden]``.
            positions: Each row's next absolute position, ``[B]``.
            ctx: Mixer inputs for the batch.

        Returns:
            ``[B, hidden]`` output states.
        """
        if self.type == "linear_attention":
            h = self.mix.batch(self.in_ln(x))
        else:
            h = self.mix.batch(self.in_ln(x), positions, ctx)
        x = x + h
        return x + self.mlp.batch(self.post_ln(x))

    def reset(self) -> None:
        """Clear this layer's recurrent state so the model can take a new request."""
        self.mix.reset()
