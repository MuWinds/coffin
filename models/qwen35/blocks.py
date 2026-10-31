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
        self.down = _proj(sd, [f"{p}.down_proj"], bench)

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
        self.mlp = MLP(sd, f"{p}.mlp", bench)
        if self.type == "linear_attention":
            self.mix = LinearAttn(sd, f"{p}.linear_attn", cfg, bench)
        else:
            self.mix = FullAttn(sd, f"{p}.self_attn", cfg, bench, max_seq, paged)

    def __call__(self, x: torch.Tensor, ctx) -> torch.Tensor:
        """让一个解码步穿过本层。

        Args:
            x: 隐含状态，``[hidden]``。
            ctx: mixer 的输入（缓存槽位、位置以及本层自己的状态）；按位置参数
                转交给 mixer。

        Returns:
            ``[hidden]`` 的输出状态。
        """
        h = self.mix(self.in_ln(x), *ctx)
        # 残差相加折进消费它的归一化：``post_ln`` 就地累加 ``x += h``，并把和归一化
        # 写进 ``h``——相加刚刚就读走了 mixer 的输出，那个值随即失效。一次 launch
        # 代替两次，也不用分配缓冲区。
        #
        # 只有这一处相加在这里折叠。本层的*另一处*相加 ``x + mlp(...)`` 由下一层的
        # ``in_ln`` 消费，而不是本层任何东西，要折它就得有跨层携带的缓冲区，那属于
        # ``model._forward``，不在这里。
        normed = self.post_ln.add_residual(x, h, h)
        return x + self.mlp(normed)

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
        x = x + h
        return x + self.mlp.prefill(self.post_ln(x))

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
        if self.type == "linear_attention":
            h = self.mix.verify(self.in_ln(x))
        else:
            h = self.mix.verify(self.in_ln(x), positions, *ctx)
        x = x + h
        return x + self.mlp.prefill(self.post_ln(x))

    def batch(self, x: torch.Tensor, positions: torch.Tensor, ctx) -> torch.Tensor:
        """为静态批解码步运行本层。

        投影走批处理（所有行共用一次权重读取）；有状态的 mixer 仍跑现有的逐行
        内核，每一行因此保持单条解码的精确数字。

        Args:
            x: 每个解码序列一行，``[B, hidden]``。
            positions: 每行下一个绝对位置，``[B]``。
            ctx: 该批对应的 mixer 输入。

        Returns:
            ``[B, hidden]`` 的输出状态。
        """
        if self.type == "linear_attention":
            h = self.mix.batch(self.in_ln(x))
        else:
            h = self.mix.batch(self.in_ln(x), positions, ctx)
        x = x + h
        return x + self.mlp.batch(self.post_ln(x))

    def reset(self) -> None:
        """清空本层的循环状态，让模型可以接新请求。"""
        self.mix.reset()
