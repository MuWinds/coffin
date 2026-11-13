"""一个 transformer 层：一个 mixer（线性注意力或全注意力）加 MLP。

层拿到哪种 mixer 由 ``cfg["layer_types"]`` 决定；checkpoint 把 24 层线性注意力和
8 层全注意力交错排列。

Module: ``coffin/models/qwen35``。依赖 ``ops``（归一化、gemv）和同级的
``full_attn``、``linear_attn``、``batch``；由 ``model.py`` 组装成层栈。

Constraints:
    * 两个 mixer 类只能通过下面这些方法互换（``prefill_project`` /
      ``prefill_state`` / ``verify`` / ``batch``），所以新的 mixer 必须实现同一套
      接口；``isinstance`` 判断负责把线性注意力专属的分支导过去。
    * 线性注意力层携带循环状态，全注意力层携带 KV。这里的每个入口都必须摸到对的
      那一个，否则分块预填充只会悄悄推进半个模型。
    * 层类型从 checkpoint 的 ``layer_types`` 列表读取，绝不从层下标推断——
      交错方式是模型属性。
"""

from __future__ import annotations

import torch

from backends import ops as bops
from models.qwen35.batch import batch_proj, wide_proj
from models.qwen35.full_attn import FullAttn
from models.qwen35.linear_attn import LinearAttn
from models.qwen35.moe import SparseMoE
from models.qwen35.parallel import row_projection
from ops.gemv import _proj
from ops.norm import RMSNorm


class MLP:
    """SwiGLU 前馈块：``down(silu(gate) * up)``。

    ``gate`` 和 ``up`` 共享同一个输入，所以建成一个融合的
    :class:`~ops.gemv.GEMVGroup` 再切分；在每条路径（解码、预填充、批处理）上，
    一次宽权重读取都胜过两次窄的。
    """

    def __init__(self, sd: dict, p: str, bench: bool) -> None:
        """构建融合的 gate/up 投影和 down 投影。

        Args:
            sd: checkpoint state dict。
            p: 这个 MLP 在 state dict 里的参数前缀。
            bench: 透传给投影；``False`` 跳过自动调优。
        """
        self.gate_up = _proj(sd, [f"{p}.gate_proj", f"{p}.up_proj"], bench)
        self.down = row_projection(_proj(sd, [f"{p}.down_proj"], bench))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """把块作用在一行或少数几行上（解码路径）。

        ``silu(gate) * up`` 作为一个算子直接写进 ``up`` 自己的缓冲区，而不是写成
        ``F.silu(gate) * up``。拆成两次 torch 调用就是两次 launch，每层还要两个
        新的 ``[intermediate]`` 临时量；融合入口只 launch 一次，每个操作数只读
        一次，并就地覆写 ``up``，那之后没有东西再读它。一次解码步骤有 32 个 MLP，
        而这张卡的 launch 下限约 5.75 us，两次 launch 比算术本身还贵。按契约
        ``out`` 可以与 ``up`` 别名，所以这次回写是合法的，不是取巧。

        Args:
            x: 激活，``[hidden]`` 或 ``[B, hidden]``，bf16。

        Returns:
            与 ``x`` 相同的前导形状，宽度为 hidden。
        """
        gate, up = self.gate_up(x)
        return self.down(bops.silu_mul(gate, up, up))

    def prefill(self, x: torch.Tensor) -> torch.Tensor:
        """把块作用在一个预填充分块上。

        和 :meth:`__call__` 是同一个融合乘积。预填充是计算受限的，launch 占比更小，
        但这里算术和布局都完全相同，另写一种拼法只会多出一处需要保持数值一致的
        地方。

        Args:
            x: 激活，``[C, hidden]``，bf16。

        Returns:
            ``[C, hidden]``。
        """
        gate, up = self.gate_up.prefill(x)
        return self.down.prefill(bops.silu_mul(gate, up, up))

    def batch(self, x: torch.Tensor) -> torch.Tensor:
        """用逐位精确的批处理 GEMV 路径作用这个块。

        Args:
            x: 激活，``[B, hidden]``，每个解码序列一行。

        Returns:
            ``[B, hidden]``，每行与单条解码结果一致。
        """
        gate, up = batch_proj(self.gate_up, x)
        return batch_proj(self.down, bops.silu_mul(gate, up, up))

    def wide(self, x: torch.Tensor) -> torch.Tensor:
        """在任意行数下作用这个块（一次前向校验）。

        Args:
            x: 激活，``[M, hidden]``；``M`` 是校验宽度 ``B*c``。

        Returns:
            ``[M, hidden]``；一旦用上宽内核，各行不必与单条解码逐位相同。
        """
        gate, up = wide_proj(self.gate_up, x)
        return wide_proj(self.down, bops.silu_mul(gate, up, up))


class Layer:
    """一个解码层：pre-norm mixer、残差相加、pre-norm MLP、残差相加。

    mixer 由 checkpoint 里 ``layer_types`` 对应本下标的项决定：
    ``linear_attention`` 拿 :class:`~models.qwen35.linear_attn.LinearAttn`，其余
    一律 :class:`~models.qwen35.full_attn.FullAttn`。
    """

    def __init__(self, sd: dict, i: int, cfg: dict, bench: bool, max_seq: int,
                 paged: bool = False) -> None:
        """构建归一化、MLP 和本层的 mixer。

        Args:
            sd: checkpoint state dict。
            i: 层下标，既用于拼参数前缀，也用于查 ``layer_types[i]``。
            cfg: 来自 ``config.json`` 的模型配置字典。
            bench: 透传给投影；``False`` 跳过自动调优。
            max_seq: 引擎会跑的最长序列，用来定全注意力掩码和 RoPE 表的尺寸。
            paged: 让 mixer 针对分页 KV 池构建，而不是连续缓存。必须和运行器
                驱动模型的方式一致。
        """
        p = f"model.language_model.layers.{i}"
        self.type = cfg["layer_types"][i]
        self.in_ln = RMSNorm(sd[f"{p}.input_layernorm.weight"])
        self.post_ln = RMSNorm(sd[f"{p}.post_attention_layernorm.weight"])
        # 稠密层和 MoE 层只在这里分叉：``num_experts`` 是 checkpoint 的属性，
        # 而两个块暴露的四个入口（``__call__`` / ``prefill`` / ``batch`` / ``wide``）
        # 完全一致，所以层里再往下就没有分支了。
        self.mlp = (SparseMoE(sd, f"{p}.mlp", cfg, bench)
                    if cfg.get("num_experts") else MLP(sd, f"{p}.mlp", bench))
        if self.type == "linear_attention":
            self.mix = LinearAttn(sd, f"{p}.linear_attn", cfg, bench)
        else:
            self.mix = FullAttn(sd, f"{p}.self_attn", cfg, bench, max_seq, paged)

    def _add_then_norm(self, ln: RMSNorm, x: torch.Tensor, add: torch.Tensor,
                       out: torch.Tensor) -> torch.Tensor:
        """``x += add`` 紧接 ``ln(x)``：折成一次发射，返回归一化后的和。

        拆开跑是两次发射、三趟 global（相加读 ``x`` 读 ``add`` 写 ``x``，归一化再
        读 ``x`` 写结果）；融合版两趟、一次发射。这张卡一次发射的下限是 5.75 us，
        而 ``[hidden]`` 上的一次相加实测 13.2 us 已经全在发射里，所以省下来的就是
        一次发射本身。

        Args:
            ln: 消费这次相加的归一化，``in_ln`` 或 ``post_ln``。
            x: 残差流，``[..., hidden]``。**会被就地改写**成和，调用方必须独占它。
            add: 要累加的输出，形状同 ``x``。
            out: 归一化后的和要写入的稠密缓冲区。可以就是 ``add``——相加在写之前
                已经把它读走，于是省掉一次分配；编码捕获路径因此不需要新缓冲区。

        Returns:
            ``[..., hidden]``，归一化后的和。

        激活不是 bf16 时退回拆开的两步：残差是就地累加的，没有更宽的缓冲区可退，
        在这里转换会把和困在临时对象里。
        """
        if x.dtype != ln.g.dtype or add.dtype != ln.g.dtype:
            return ln(x + add)
        return ln.add_residual(x, add, out)

    def norm_after_add(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """``x += h`` 紧接 ``post_ln(x)``，归一化写进 ``h`` 本身——相加在写之前
        已经把它读走，于是连缓冲区都不用分配。

        预填充、校验、批解码三条入口每层都先加残差、紧接着归一化，拆开跑是两次
        发射、三趟 global；解码的 ``__call__`` 早就是这条，这三条是补上的同一处。

        Args:
            x: 残差流，``[..., hidden]``。**会被就地改写**成和。
            h: 要累加的 mixer 输出，形状同 ``x``。

        Returns:
            ``[..., hidden]``，归一化后的和。
        """
        return self._add_then_norm(self.post_ln, x, h, h)

    def step(self, x: torch.Tensor, ctx, carry: torch.Tensor | None = None):
        """让一个解码步穿过本层，并把跨层的那处残差相加折进本层 ``in_ln``。

        层与层之间唯一的相加是上一层的 MLP 输出，而它唯一的消费者是本层的
        ``in_ln``：拆开跑是「相加 + 归一化」两次发射（``[hidden]`` 上实测 13.2 +
        15.8 us），折成 ``add_rms_norm`` 只有 17.9 us。33 层一步步省 32 次发射，
        约 0.36 ms，占每步 17.3 ms 的 2%。

        残差流 ``x`` 一路就地累加，所以同一块缓冲区要跨层传下去；本层 MLP 的输出
        不加，交给下一层——加在哪一层的 ``in_ln`` 里不影响数字，因为相加仍然先把和
        舍入成 bf16，均方也仍按那个舍入后的值算。

        Args:
            x: 残差流，``[hidden]``，bf16；返回时已含本层 mixer 那处相加。
            ctx: mixer 的输入（缓存槽位、位置以及本层自己的状态）。
            carry: 上一层留下的 MLP 输出，尚未加进 ``x``；``None`` 表示 ``x``
                已经是完整的残差流（第一层）。

        Returns:
            ``(x, mlp 输出)``：残差流与本层 MLP 的结果，后者留给下一层累加。
        """
        normed = (self.in_ln(x) if carry is None
                  else self._add_then_norm(self.in_ln, x, carry, carry))
        h = self.mix(normed, *ctx)
        return x, self.mlp(self.post_ln.add_residual(x, h, h))

    def __call__(self, x: torch.Tensor, ctx) -> torch.Tensor:
        """让一个完整解码步穿过本层（自己不持有残差流的调用方用这个）。

        Args:
            x: 隐含状态，``[hidden]``。
            ctx: mixer 的输入（缓存槽位、位置以及本层自己的状态）；按位置参数
                转交给 mixer。

        Returns:
            ``[hidden]`` 的输出状态。
        """
        x, carry = self.step(x, ctx)
        return x + carry

    def prefill(self, x: torch.Tensor, positions: torch.Tensor, ctx,
                tiles: list | None = None) -> torch.Tensor:
        """让一个预填充分块穿过本层，只处理单个序列。

        输入投影被提到 :meth:`prefill_state` 外面，这样有状态的那一半可以逐序列
        运行，而 GEMM 仍然只有一次调用。

        Args:
            x: 分块激活，``[C, hidden]``。
            positions: 该分块的绝对位置，``[C]``。
            ctx: 该分块对应的 mixer 输入。
            tiles: 预先切好的查询块；``None`` 表示让 mixer 自己切分这个分块。

        Returns:
            ``[C, hidden]`` 的输出激活。
        """
        proj = self.mix.prefill_project(self.in_ln(x))
        h = self.prefill_state(proj, positions, ctx, tiles, self.mix.prefill_row)
        return x + self.mlp.prefill(self.norm_after_add(x, h))

    def prefill_state(self, proj, positions: torch.Tensor, ctx,
                      tiles: list | None, row: int) -> torch.Tensor:
        """在 ``row`` 上为单个序列执行预填充分块的 mixer 半边。

        Args:
            proj: 该分块的原始投影输出，即 mixer 的 ``prefill_project`` 返回的
                东西。
            positions: 该分块的绝对位置，``[C]``。
            ctx: 该分块对应的 mixer 输入。
            tiles: 预先切好的查询块，或 ``None``。
            row: 该序列在批布局里占用的递推/KV 行；预填充临时行是最后一行。

        Returns:
            ``[C, hidden]``，加残差之前的 mixer 输出。
        """
        mix = self.mix
        if isinstance(mix, LinearAttn):
            pre = mix.prefill_state(*proj, row=row)
            return mix.out_proj.prefill(pre)
        pre = mix.prefill_attend(*proj, positions, ctx, tiles, row)
        return mix.o_proj.prefill(pre)

    # 组预填充：投影是对所有序列 token 的一次 GEMM，有状态的一半逐序列运行、各用
    # 自己的位置 / KV 行 / 递推行，输出投影再对拼接结果做一次批处理。
    def prefill_group_project(self, x: torch.Tensor):
        """对若干个序列预填充分块的拼接做投影。

        Args:
            x: 所有序列分块的激活，``[C_total, hidden]``。

        Returns:
            ``(mixer, 投影元组)``：mixer 实例，好让下面的逐序列调用不必再按层
            类型重新分派；外加原始投影输出。
        """
        mix = self.mix
        return mix, mix.prefill_project(self.in_ln(x))

    def prefill_group_state(self, mix, proj, positions: torch.Tensor, ctx,
                            tiles: list | None, row: int) -> torch.Tensor:
        """运行某个序列在组投影里的那一片。

        Args:
            mix: :meth:`prefill_group_project` 返回的 mixer。
            proj: 该序列在组投影里的切片。
            positions: 该切片的绝对位置，``[C_i]``。
            ctx: 该切片对应的 mixer 输入。
            tiles: 预先切好的查询块，或 ``None``。
            row: 该序列在批布局里占用的行。

        Returns:
            ``[C_i, D]``，位于输出投影之前。保持未投影状态，:meth:`prefill_group_out`
            才能把所有序列的 GEMM 合成一次。
        """
        if isinstance(mix, LinearAttn):
            return mix.prefill_state(*proj, row=row)
        return mix.prefill_attend(*proj, positions, ctx, tiles, row)

    def prefill_group_out(self, mix, pre_out: torch.Tensor) -> torch.Tensor:
        """对所有序列的 mixer 输出一次性施加输出投影。

        Args:
            mix: :meth:`prefill_group_project` 返回的 mixer。
            pre_out: 未投影输出的拼接，``[C_total, D]``。

        Returns:
            ``[C_total, hidden]``。
        """
        if isinstance(mix, LinearAttn):
            return mix.out_proj.prefill(pre_out)
        return mix.o_proj.prefill(pre_out)

    def step_verify(self, x: torch.Tensor, positions: torch.Tensor, ctx,
                    carry: torch.Tensor | None = None):
        """为一步里的 ``c`` 个投机 token 重放本层，跨层相加折进 ``in_ln``。

        与 :meth:`step` 同一条折叠，只是行数是校验窗口宽度 ``c``；残差流同样就地
        累加、跨层传同一块缓冲区，归一化写进 ``carry`` 本身（它在相加读走之后就没
        人用了），所以捕获图里也不会多出缓冲区。

        Args:
            x: 校验窗口的隐含状态，``[c, hidden]``；返回时已含本层 mixer 那处相加。
            positions: 那些 token 的绝对位置，``[c]``。
            ctx: 窗口对应的 mixer 输入；线性注意力 mixer 不接收位置，因为它携带
                的是自己的状态。
            carry: 上一层留下的 MLP 输出，尚未加进 ``x``；``None`` 是第一层。

        Returns:
            ``(x, mlp 输出)``，后者留给下一层累加。
        """
        normed = (self.in_ln(x) if carry is None
                  else self._add_then_norm(self.in_ln, x, carry, carry))
        if self.type == "linear_attention":
            h = self.mix.verify(normed)
        else:
            h = self.mix.verify(normed, positions, *ctx)
        return x, self.mlp.prefill(self.post_ln.add_residual(x, h, h))

    def verify(self, x: torch.Tensor, positions: torch.Tensor, ctx) -> torch.Tensor:
        """为一步里的 ``c`` 个投机 token 重放本层。

        Args:
            x: 校验窗口的隐含状态，``[c, hidden]``。
            positions: 那些 token 的绝对位置，``[c]``。
            ctx: 窗口对应的 mixer 输入；线性注意力 mixer 不接收位置，因为它携带
                的是自己的状态。

        Returns:
            ``[c, hidden]`` 的输出状态。
        """
        x, carry = self.step_verify(x, positions, ctx)
        return x + carry

    def step_batch(self, x: torch.Tensor, positions: torch.Tensor, ctx,
                   carry: torch.Tensor | None = None,
                   verify_slot: int | None = None):
        """为静态批解码步运行本层，跨层相加折进 ``in_ln``。

        与 :meth:`step` 同一条折叠，行数是批宽 ``B``；投影仍走批处理，有状态的
        mixer 仍逐行跑现有内核。

        Args:
            x: 每个解码序列一行，``[B, hidden]``；返回时已含本层 mixer 那处相加。
            positions: 每行下一个绝对位置，``[B]``。
            ctx: 该批对应的 mixer 输入。
            carry: 上一层留下的 MLP 输出，尚未加进 ``x``；``None`` 是第一层。
            verify_slot: 仅校验时传入当前日志位置，让线性注意力保存秩一更新。

        Returns:
            ``(x, mlp 输出)``，后者留给下一层累加。
        """
        normed = (self.in_ln(x) if carry is None
                  else self._add_then_norm(self.in_ln, x, carry, carry))
        if self.type == "linear_attention":
            h = self.mix.batch(normed, record_slot=verify_slot)
        else:
            h = self.mix.batch(normed, positions, ctx)
        return x, self.mlp.batch(self.post_ln.add_residual(x, h, h))

    def batch(self, x: torch.Tensor, positions: torch.Tensor, ctx,
              verify_slot: int | None = None) -> torch.Tensor:
        """为静态批解码步运行本层。

        投影走批处理（所有行共用一次权重读取）；有状态的 mixer 仍跑现有的逐行
        内核，每一行因此保持单条解码的精确数字。

        Args:
            x: 每个解码序列一行，``[B, hidden]``。
            positions: 每行下一个绝对位置，``[B]``。
            ctx: 该批对应的 mixer 输入。
            verify_slot: 仅校验时传入当前日志位置，让线性注意力保存秩一更新。

        Returns:
            ``[B, hidden]`` 的输出状态。
        """
        x, carry = self.step_batch(x, positions, ctx, verify_slot=verify_slot)
        return x + carry

    def reset(self) -> None:
        """清空本层的循环状态，让模型可以接新请求。"""
        self.mix.reset()
