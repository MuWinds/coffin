"""One transformer layer: a mixer (linear or full attention) plus the MLP.

Which mixer a layer gets comes from ``cfg["layer_types"]``; the checkpoint
interleaves 24 linear-attention layers with 8 full-attention ones.
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
    def __init__(self, sd: dict, p: str, bench: bool) -> None:
        self.gate_up = _proj(sd, [f"{p}.gate_proj", f"{p}.up_proj"], bench)
        self.down = _proj(sd, [f"{p}.down_proj"], bench)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x)
        return self.down(F.silu(gate) * up)

    def prefill(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up.prefill(x)
        return self.down.prefill(F.silu(gate) * up)

    def batch(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` [B, hidden] -> [B, hidden] with the M=B batched GEMV path."""
        gate, up = batch_proj(self.gate_up, x)
        return batch_proj(self.down, F.silu(gate) * up)

    def wide(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` [M, hidden] for arbitrary M (the one-forward verify's B*c rows)."""
        gate, up = wide_proj(self.gate_up, x)
        return wide_proj(self.down, F.silu(gate) * up)


class Layer:
    def __init__(self, sd: dict, i: int, cfg: dict, bench: bool, max_seq: int,
                 paged: bool = False) -> None:
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
        h = self.mix(self.in_ln(x), *ctx)
        x = x + h
        return x + self.mlp(self.post_ln(x))

    def prefill(self, x: torch.Tensor, positions: torch.Tensor, ctx,
                tiles: list | None = None) -> torch.Tensor:
        proj = self.mix.prefill_project(self.in_ln(x))
        h = self.prefill_state(proj, positions, ctx, tiles, self.mix.prefill_row)
        x = x + h
        return x + self.mlp.prefill(self.post_ln(x))

    def prefill_state(self, proj, positions: torch.Tensor, ctx,
                      tiles: list | None, row: int) -> torch.Tensor:
        """Mixer half of a prefill chunk for one sequence on ``row``."""
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
        """``[C_total, hidden]`` -> ``(mix, raw projection tuple)``."""
        mix = self.mix
        return mix, mix.prefill_project(self.in_ln(x))

    def prefill_group_state(self, mix, proj, positions: torch.Tensor, ctx,
                            tiles: list | None, row: int) -> torch.Tensor:
        """One sequence's slice of the group projection -> pre-out_proj ``[C_i, D]``."""
        if isinstance(mix, LinearAttn):
            return mix.prefill_state(*proj, row=row)
        return mix.prefill_attend(*proj, positions, ctx, tiles, row)

    def prefill_group_out(self, mix, pre_out: torch.Tensor) -> torch.Tensor:
        """The batched output projection, once over all sequences' outputs."""
        if isinstance(mix, LinearAttn):
            return mix.out_proj.prefill(pre_out)
        return mix.o_proj.prefill(pre_out)

    def verify(self, x: torch.Tensor, positions: torch.Tensor, ctx) -> torch.Tensor:
        if self.type == "linear_attention":
            h = self.mix.verify(self.in_ln(x))
        else:
            h = self.mix.verify(self.in_ln(x), positions, *ctx)
        x = x + h
        return x + self.mlp.prefill(self.post_ln(x))

    def batch(self, x: torch.Tensor, positions: torch.Tensor, ctx) -> torch.Tensor:
        """``x`` [B, hidden] -> [B, hidden] for the static-batch decode step.

        Projections run batched (one weight read for all rows); the stateful
        mixers run the existing per-row kernels so every row keeps the exact
        numbers of a solo decode.
        """
        if self.type == "linear_attention":
            h = self.mix.batch(self.in_ln(x))
        else:
            h = self.mix.batch(self.in_ln(x), positions, ctx)
        x = x + h
        return x + self.mlp.batch(self.post_ln(x))

    def reset(self) -> None:
        self.mix.reset()
