"""门控 DeltaNet 线性注意力：32 层里 24 层的 mixer。

Module: ``coffin/models/qwen35``。另一种逐层注意力在这里是 ``full_attn``，那 8 层
用分页 KV；``blocks.Layer`` 按层下标二选一，``model.py`` 建出全部 32 层。向下依赖
``backends``（mixer 算子和设备）、``ops.delta``（分块预填充扫描）、``ops`
（dtype / 投影 / 归一化），以及 ``models/qwen35/batch`` 和 ``tuning``。本族之外不许
直接 import 它。

全部状态（卷积窗口、fp32 循环状态）都在固定缓冲区里就地更新，所以解码步可以被
图捕获。解码跑融合的 prologue 和 delta 入口；预填充用 ``ops.delta`` 里的分块
形式；校验逐 token 跑解码入口，并在每个 token 之后拷贝状态，因为被拒绝的草稿
必须回滚。

Constraints:
    * 循环状态 ``[num_v, dk, dv]`` 在 fp32 下每层 2 MB，没有闭式回退办法。所以
      每个能回滚的路径（单流校验、批量校验）都在每个 token 前后把状态拷进
      ``ckpt_*``；少了这份拷贝，被拒的草稿会让 mixer 永久领先于 KV。
    * 前缀引擎导出快照时，状态必须和全注意力的 KV *一起*打快照，否则复用的前缀
      会把一个请求的 key 配上从未见过它们的递推状态。
    * 缓冲区按容量定尺寸、批行打头，单流预填充/校验跑在 ``prefill_row``（最后
      一行）上——解码步之间的预填充因此踩不到某个解码行的状态。
    * 递推全程 fp32：delta 规则在整个上下文上累加外积，bf16 状态几千个 token
      之后就会发散。
    * 这里没有任何路径按「内核是否存在」分支。prologue/delta 链的即时执行版本
      就是参考后端的实现，经算子表到达，所以数学定义只有一份而不是两份。
    * 所有改写都是单线程的，由 ``llm_engine`` 驱动；没有加锁，也不许加。
"""

from __future__ import annotations

import torch

from backends import device as _device
from backends import ops as bops
from models.qwen35.batch import batch_proj
from models.qwen35.tuning import CKPT
from ops.delta import chunked_delta
from ops.dtypes import D
from ops.gemv import _proj
from ops.norm import RMSNormGated


class LinearAttn:
    """门控 DeltaNet 线性注意力的单 token 解码步。

    全部状态都在固定缓冲区里就地更新，所以这一步可以被图捕获。
    """

    def __init__(self, sd: dict, p: str, cfg: dict, bench: bool) -> None:
        """加载一层的 DeltaNet 权重并分配它的循环状态。

        Args:
            sd: 完整 state dict；本层只读 ``{p}.*`` 这些键。
            p: 该层在 checkpoint 里的前缀，例如
                ``model.layers.0.linear_attn``。
            cfg: 模型配置。头的分配（32 个 value 头、16 个 key 头）、
                ``linear_key_head_dim`` / ``linear_value_head_dim``（各 128）以及
                ``linear_conv_kernel_dim``（4）都来自这里。
            bench: 透传给 ``_proj``，用于选择基准权重布局。
        """
        self.num_v = cfg["linear_num_value_heads"]      # 32
        self.num_k = cfg["linear_num_key_heads"]        # 16
        self.dk = cfg["linear_key_head_dim"]            # 128
        self.dv = cfg["linear_value_head_dim"]          # 128
        self.rep = self.num_v // self.num_k             # 2
        self.kdim = self.dk * self.num_k                # 2048
        self.vdim = self.dv * self.num_v                # 4096
        self.conv_dim = self.kdim * 2 + self.vdim       # 8192
        self.kernel = cfg["linear_conv_kernel_dim"]     # 4

        self.qkvzb_a = _proj(sd, [f"{p}.in_proj_qkv", f"{p}.in_proj_z",
                                  f"{p}.in_proj_b", f"{p}.in_proj_a"], bench)
        self.out_proj = _proj(sd, [f"{p}.out_proj"], bench)
        self.norm = RMSNormGated(sd[f"{p}.norm.weight"])
        # 卷积核到达时是 [conv_dim, 1, 4]；最后一维保持在最后，融合算子才能把四个
        # 抽头当作一段连续的 stride-1 数据读。
        self.conv_w = sd[f"{p}.conv1d.weight"].reshape(self.conv_dim, self.kernel).float()
        # ``A_log`` / ``dt_bias`` 在 checkpoint 里就是 fp32，在这里做指数运算：用低
        # 精度做会丢掉衰减率。
        self.A_log = sd[f"{p}.A_log"].float()
        self.dt_bias = sd[f"{p}.dt_bias"].float()
        self.neg_expA = -self.A_log.exp()
        self.qscale = self.dk ** -0.5

        # 批处理路径的逐行状态与临时缓冲：首维是批行（容量 1 即单流引擎，它索引的
        # 是预填充行）。``prefill_row`` 由 ``Qwen35.enable_batching`` 设成最后一行，
        # 单流预填充/校验路径就用它——解码步之间的预填充因此踩不到某个解码行的
        # 状态。
        self._bcap = 1
        self.prefill_row = 0
        rows = 1
        self.conv_state = torch.zeros(rows, self.conv_dim, self.kernel - 1,
                                      dtype=torch.float32, device=_device)
        self.conv_out = torch.zeros(rows, self.conv_dim, dtype=torch.float32,
                                    device=_device)
        self.state = torch.zeros(rows, self.num_v, self.dk, self.dv,
                                 dtype=torch.float32, device=_device)
        self.delta_out = torch.zeros(rows, self.num_v, self.dv, dtype=torch.float32,
                                     device=_device)

        # fp32 临时缓冲，由 ``delta_prologue`` 写入、``delta`` 读取。
        self.q_pro = torch.zeros(rows, self.num_v, self.dk, dtype=torch.float32,
                                 device=_device)
        self.k_pro = torch.zeros(rows, self.num_v, self.dk, dtype=torch.float32,
                                 device=_device)
        self.v_pro = torch.zeros(rows, self.num_v, self.dv, dtype=torch.float32,
                                 device=_device)
        self.beta = torch.zeros(rows, self.num_v, dtype=torch.float32, device=_device)
        self.g = torch.zeros(rows, self.num_v, dtype=torch.float32, device=_device)
        # 融合 prologue+递推+归一化入口的目标缓冲。它和 ``delta_out`` 分开而不是共用，
        # 因为融合入口写的是*归一化后*的结果：``delta_out`` 要留给三算子拼法（以及
        # 校验路径）当归一化前的临时缓冲。
        #
        # bf16，与 ``RMSNormGated`` 的返回以及三算子链里 ``out_proj`` 读入的东西一致。
        # 这里用 fp32 容器会改变到达投影的激活 dtype，即便数值相等也算数值改动。
        self.normed_out = torch.zeros(rows, self.num_v, self.dv, dtype=D, device=_device)

        # 投机校验回滚用的逐 token 状态快照（每个校验 token 一个槽位；否则循环状态
        # 无法倒回）。``CKPT`` 限定校验宽度也是同一个理由。
        self.ckpt_state = torch.zeros(CKPT, self.num_v, self.dk, self.dv,
                                      dtype=torch.float32, device=_device)
        self.ckpt_conv = torch.zeros(CKPT, self.conv_dim, self.kernel - 1,
                                     dtype=torch.float32, device=_device)
        # 批量投机校验按 (行, 校验 token) 保存一份快照；只有打开批量投机解码时才由
        # ``enable_verify_batch`` 分配，因为 fp32 状态每层每份快照就是 2 MB。
        self.ckpt_state_b: torch.Tensor | None = None
        self.ckpt_conv_b: torch.Tensor | None = None

    def enable_verify_batch(self, capacity: int, width: int) -> None:
        """为每个批行的 ``width`` 份回滚快照腾出空间。

        Args:
            capacity: 解码行数。
            width: 每行的校验 token 数，即快照槽位数。单流路径也要持有时，不得
                超过 ``CKPT``。

        Returns:
            无。形状不变时是幂等的，所以第一次 ``enable_verify_batch`` 之后再调一次
            不会在图下面重新分配。
        """
        if self.ckpt_state_b is not None and self.ckpt_state_b.shape[:2] == (capacity, width):
            return
        self.ckpt_state_b = torch.zeros(capacity, width, self.num_v, self.dk, self.dv,
                                        dtype=torch.float32, device=_device)
        self.ckpt_conv_b = torch.zeros(capacity, width, self.conv_dim, self.kernel - 1,
                                       dtype=torch.float32, device=_device)

    def snapshot_batch(self, rows: int, slot: int) -> None:
        """把 ``rows`` 行当前的循环（以及卷积）状态存进一个槽位。

        Args:
            rows: 要拷贝的前导行数；其余不动，所以没装满的 bucket 不必为闲置行
                付代价。
            slot: 校验宽度内的快照下标，也就是此刻已经接受了多少个草稿 token。

        Returns:
            无。这是批量校验唯一能倒回的手段，所以必须在每个校验 token 之后、下一
            个 token 就地改写状态之前调用。
        """
        self.ckpt_state_b[:rows, slot].copy_(self.state[:rows])
        self.ckpt_conv_b[:rows, slot].copy_(self.conv_state[:rows])

    def verify_batch_step(self, qkv: torch.Tensor, z: torch.Tensor,
                          b: torch.Tensor, a: torch.Tensor, slot: int) -> torch.Tensor:
        """从已投影好的 qkv/z/b/a 出发，为 ``B`` 行处理一个校验 token。

        跑的是和解码步相同的 conv / prologue / delta 入口（各自一次批处理 launch，
        所以每行仍保持贪心解码的数字），并在该 token 之后给循环状态打快照——
        ``slot`` 就是它在校验宽度内的下标，也就是到此为止的草稿都被接受时的回滚点。

        Args:
            qkv: 该 token 的融合 q/k/v 投影，``[B, conv_dim]``。
            z: 门投影，``[B, vdim]``。
            b: beta logits，``[B, num_v]``。
            a: 衰减 logits，``[B, num_v]``。
            slot: 快照下标，``0 <= slot < width``。

        Returns:
            ``[B, vdim]``，``out_proj`` 之前的输出，好让调用方对整个 bucket 只做
            一次那笔权重读取。
        """
        rows = qkv.shape[0]
        bops.conv1d_silu_batch(self.conv_w, qkv, self.conv_state[:rows],
                               self.conv_out[:rows])
        conv = self.conv_out[:rows]
        q = conv[:, :self.kdim].reshape(rows, self.num_k, self.dk)
        k = conv[:, self.kdim:2 * self.kdim].reshape(rows, self.num_k, self.dk)
        v = conv[:, 2 * self.kdim:].reshape(rows, self.num_v, self.dv)
        bops.delta_prologue_batch(q, k, v, b.contiguous(), a.contiguous(),
                                  self.neg_expA, self.dt_bias,
                                  self.q_pro[:rows], self.k_pro[:rows],
                                  self.v_pro[:rows], self.beta[:rows],
                                  self.g[:rows], self.qscale, self.rep)
        bops.delta_batch(self.q_pro[:rows], self.k_pro[:rows],
                         self.v_pro[:rows], self.beta[:rows], self.g[:rows],
                         self.state[:rows], self.delta_out[:rows])
        self.snapshot_batch(rows, slot)
        # 所有行只发一次门控归一化（RMSNormGated 入口内部把 (行, 头) 对摊平）；原来
        # 这里的逐行循环每 token 要约 8 次 launch，主导了一次前向校验的耗时。
        zz = z.reshape(rows, self.num_v, self.dv).contiguous()
        proj_in = self.norm(self.delta_out[:rows], zz)
        return proj_in.reshape(rows, self.vdim)

    def restore_batch(self, row: int, slot: int) -> None:
        """把某一行的循环状态倒回校验期间打下的快照。

        Args:
            row: 要恢复的解码行。
            slot: 被接受的快照下标；它之后的东西靠把状态移回去就丢掉了，这也是
                校验循环无条件打快照、而不是只在接受时打的原因。

        Returns:
            无。只有状态和卷积窗口会动——KV 缓存是按位置索引的，靠位置计数器
            倒回。
        """
        self.state[row].copy_(self.ckpt_state_b[row, slot])
        self.conv_state[row].copy_(self.ckpt_conv_b[row, slot])

    def enable_batching(self, capacity: int, prefill_row: int) -> None:
        """让循环状态和临时缓冲按批条目各占一行。

        Args:
            capacity: 批需要的行数，包含预填充行。
            prefill_row: 留给单流预填充和校验的行，解码步之间的预填充因此踩不到
                某个解码行的状态。

        Returns:
            无。几何形状已经匹配时是空操作，捕获图的缓冲区地址因此保持有效。
        """
        capacity = max(int(capacity), 1)
        if capacity == 1 and self.prefill_row == 0:
            return
        rows = capacity
        self._bcap = capacity
        self.prefill_row = int(prefill_row)
        self.conv_state = torch.zeros(rows, self.conv_dim, self.kernel - 1,
                                      dtype=torch.float32, device=_device)
        self.conv_out = torch.zeros(rows, self.conv_dim, dtype=torch.float32,
                                    device=_device)
        self.state = torch.zeros(rows, self.num_v, self.dk, self.dv,
                                 dtype=torch.float32, device=_device)
        self.delta_out = torch.zeros(rows, self.num_v, self.dv,
                                     dtype=torch.float32, device=_device)
        self.q_pro = torch.zeros(rows, self.num_v, self.dk, dtype=torch.float32,
                                 device=_device)
        self.k_pro = torch.zeros(rows, self.num_v, self.dk, dtype=torch.float32,
                                 device=_device)
        self.v_pro = torch.zeros(rows, self.num_v, self.dv, dtype=torch.float32,
                                 device=_device)
        self.beta = torch.zeros(rows, self.num_v, dtype=torch.float32, device=_device)
        self.g = torch.zeros(rows, self.num_v, dtype=torch.float32, device=_device)
        self.normed_out = torch.zeros(rows, self.num_v, self.dv, dtype=D, device=_device)

    def batch(self, x: torch.Tensor, *_) -> torch.Tensor:
        """``x`` [B, hidden] -> [B, hidden]：批处理投影，逐行递推。

        delta 规则沿位置串行、且按序列有状态，所以递推数学仍是逐行的——但用上批
        处理入口之后，整个 bucket 每个阶段（conv、prologue、更新）只 launch 一次
        而不是 B 次，每行各用自己的状态/卷积行。两条路径必须逐行逐位一致。

        Args:
            x: 解码行输入，``[B, hidden]``。
            *_: 吸收掉的层参数占位，好让 ``Layer.batch`` 用同一签名调用每种
                mixer。

        Returns:
            ``[B, hidden]``，经批处理 ``out_proj`` 之后的层输出。
        """
        B = x.shape[0]
        qkv, z, b, a = batch_proj(self.qkvzb_a, x)
        z = z.reshape(B, self.num_v, self.dv)
        proj_in = torch.empty(B, self.vdim, dtype=D, device=_device)
        # 批处理入口要求刚好 [B, ...] 的缓冲区；状态/临时行住在按容量定尺寸的缓冲区
        # 里，所以把行前缀交给它们。
        conv_out = self.conv_out[:B]
        bops.conv1d_silu_batch(self.conv_w, qkv, self.conv_state[:B], conv_out)
        # [B, conv_dim] 结果的列切片：行 stride 是 conv_dim，批处理 prologue 接受
        # 这种布局（只有两个内层 stride 必须紧凑）。
        q = conv_out[:, :self.kdim].reshape(B, self.num_k, self.dk)
        k = conv_out[:, self.kdim:2 * self.kdim].reshape(B, self.num_k, self.dk)
        v = conv_out[:, 2 * self.kdim:].reshape(B, self.num_v, self.dv)
        bops.delta_prologue_batch(q, k, v, b.contiguous(), a.contiguous(),
                                  self.neg_expA, self.dt_bias,
                                  self.q_pro[:B], self.k_pro[:B], self.v_pro[:B],
                                  self.beta[:B], self.g[:B], self.qscale, self.rep)
        bops.delta_batch(self.q_pro[:B], self.k_pro[:B], self.v_pro[:B],
                         self.beta[:B], self.g[:B], self.state[:B],
                         self.delta_out[:B])
        # 门控归一化按 (行, value 头) 做，权重是 [dv]，所以折不进批处理入口——这个
        # 循环是每行一次 launch，不是每个头一次。
        for i in range(B):
            proj_in[i] = self.norm(self.delta_out[i],
                                   z[i].reshape(self.num_v, self.dv)).reshape(self.vdim)
        return batch_proj(self.out_proj, proj_in)

    def __call__(self, x: torch.Tensor, *_) -> torch.Tensor:
        """单 token 解码步，状态就地更新。

        Args:
            x: 该 token 的层输入，``[hidden]``。
            *_: 吸收掉的层参数占位，好让 ``Layer.__call__`` 用同一签名调用两种
                mixer。

        Returns:
            ``[hidden]``，经 ``out_proj`` 之后的层输出。
        """
        qkv, z, b, a = self.qkvzb_a(x)
        z = z.reshape(self.num_v, self.dv)
        pr = self.prefill_row
        conv_state, conv_out = self.conv_state[pr], self.conv_out[pr]

        # 在最近 3 个状态加当前输入上做 depthwise 因果 conv1d（核宽 4），与 SiLU 融合
        # 成一个算子（原来是 cat + copy_ + mul + sum + silu）。
        bops.conv1d_silu(self.conv_w, qkv, conv_state, conv_out)
        qkv = conv_out

        q, k, v = qkv.split([self.kdim, self.kdim, self.vdim], dim=0)
        q = q.reshape(self.num_k, self.dk)
        k = k.reshape(self.num_k, self.dk)
        v = v.reshape(self.num_v, self.dv)

        # 用三算子拼法，不用融合的 ``delta_normed``。那个融合是逐位一致的，还能省掉
        # 每步 48 次 launch，但它仍然更慢：在卡上实测，单测时每层多 5.3 us 设备时间，
        # 真实步骤里每层多 13.1 us（24 层累计 20.285 -> 20.615 ms），因为一个内核要同时
        # 伺候递推的寄存器循环几何和两次块内归约。它省下的 launch 名额不值这个价。
        # 见 ../../../PERF_COREX.md §12。
        q_pro, k_pro, v_pro = self.q_pro[pr], self.k_pro[pr], self.v_pro[pr]
        beta, g = self.beta[pr], self.g[pr]
        bops.delta_prologue(q, k, v, b, a, self.neg_expA, self.dt_bias,
                            q_pro, k_pro, v_pro, beta, g,
                            self.qscale, self.rep)
        out = bops.delta(q_pro, k_pro, v_pro, beta, g,
                         self.state[pr], self.delta_out[pr])

        out = self.norm(out, z).reshape(self.vdim)
        return self.out_proj(out)

    def verify(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` [C, hidden] -> [C, hidden]，并逐 token 打状态快照。

        线性层上的数学和 ``prefill`` 相同，但那 C 个 token 走的是*解码*入口
        （融合 prologue + delta），所以每个 token 之后的循环状态可以拷进
        ``ckpt_state``/``ckpt_conv``，草稿被拒时再恢复。投影是批处理的，C 个 token
        因此大约只花一次权重读取。

        Args:
            x: 草稿 token 的输入，``[C, hidden]``，``C <= CKPT``。

        Returns:
            ``[C, hidden]``，经 ``out_proj`` 之后的层输出。``ckpt_state[i]`` 里的
            快照是第 ``i`` 个 token *之后*的状态，所以前 ``i+1`` 份草稿都被接受时，
            它同时就是续跑点。
        """
        C = x.shape[0]
        # [C, conv_dim], [C, vdim], [C, nv]x2
        qkv, z, b, a = self.qkvzb_a.prefill(x)
        # 递推部分逐 token，但 ``out_proj`` 是一笔权重读取：要批处理它，否则 C 个
        # token 会把投影重读 C 遍。
        proj_in = torch.empty(C, self.vdim, dtype=D, device=_device)
        pr = self.prefill_row
        conv_state, conv_out = self.conv_state[pr], self.conv_out[pr]
        q_pro, k_pro, v_pro = self.q_pro[pr], self.k_pro[pr], self.v_pro[pr]
        beta, g = self.beta[pr], self.g[pr]
        state, delta_out = self.state[pr], self.delta_out[pr]
        for i in range(C):
            bops.conv1d_silu(self.conv_w, qkv[i], conv_state, conv_out)
            q, k, v = conv_out.split([self.kdim, self.kdim, self.vdim])
            bops.delta_prologue(q.reshape(self.num_k, self.dk),
                                k.reshape(self.num_k, self.dk),
                                v.reshape(self.num_v, self.dv), b[i], a[i],
                                self.neg_expA, self.dt_bias,
                                q_pro, k_pro, v_pro, beta, g,
                                self.qscale, self.rep)
            bops.delta(q_pro, k_pro, v_pro, beta, g, state, delta_out)
            # 门控归一化按 value 头做：权重是 [dv] 而不是 [vdim]，所以状态必须保持
            # [num_v, dv]。先摊平会让内核去要 num_v*dv 个权重，然后冲出那个 [dv]
            # 张量的末尾。
            proj_in[i] = self.norm(delta_out,
                                   z[i].reshape(self.num_v, self.dv)).reshape(self.vdim)
            self.ckpt_state[i].copy_(state)
            self.ckpt_conv[i].copy_(conv_state)
        return self.out_proj.prefill(proj_in)

    def prefill_project(self, x: torch.Tensor):
        """预填充的投影半边：``[C, hidden]`` -> 原始 ``(qkv, z, b, a)``。

        拆出来是为了让组预填充对若干个序列拼接后的 prompt 只跑一次 GEMM，事后再
        切分结果。

        Args:
            x: 预填充输入，``[C, hidden]``。

        Returns:
            卷积和 delta 预处理之前的 ``(qkv, z, b, a)``，也就是调用方做完可选的
            拼接与切片之后 :meth:`prefill_state` 消费的东西。
        """
        return self.qkvzb_a.prefill(x)

    def prefill_state(self, qkv: torch.Tensor, z: torch.Tensor,
                      b: torch.Tensor, a: torch.Tensor,
                      row: int | None = None, chunk: int = 64) -> torch.Tensor:
        """预填充的递推半边：原始投影 -> ``[C, vdim]``（out_proj 之前）。

        在 ``row`` 的卷积/循环状态上跑因果卷积、融合的头预处理和分块 delta 扫描，
        于是多个序列可以共用一次批处理投影，同时各自保持独立状态。

        Args:
            qkv: 融合 q/k/v 投影，``[C, conv_dim]``。
            z: 门投影，``[C, vdim]``。
            b: beta logits，``[C, num_v]``。
            a: 衰减 logits，``[C, num_v]``。
            row: 要推进的状态行；``None`` 用暂存行。
            chunk: delta 扫描里每个分块的 token 数。扫描对任何切分都是精确的，
                所以这里是在临时缓冲大小和 launch 次数之间取舍。

        Returns:
            ``[C, vdim]``，``out_proj`` 之前的门控归一化输出。
        """
        if row is None:
            row = self.prefill_row
        C = qkv.shape[0]
        conv_state = self.conv_state[row]

        # conv 和头预处理是两个算子，而（这也是头预处理值得融合的原因）q/k/v 出来的
        # 时候已经是分块扫描要读的 [H, C, d] 布局，而不是它还得拷贝的转置视图。
        qkv = bops.conv1d_silu_seq(self.conv_w, qkv, conv_state)
        qt = torch.empty(self.num_v, C, self.dk, dtype=torch.float32, device=_device)
        kt = torch.empty(self.num_v, C, self.dk, dtype=torch.float32, device=_device)
        vt = torch.empty(self.num_v, C, self.dv, dtype=torch.float32, device=_device)
        bt = torch.empty(self.num_v, C, dtype=torch.float32, device=_device)
        gt = torch.empty(self.num_v, C, dtype=torch.float32, device=_device)
        bops.delta_prep_seq(qkv, b, a, self.neg_expA, self.dt_bias,
                            qt, kt, vt, bt, gt, self.qscale, self.rep)

        out = chunked_delta(qt, kt, vt, gt, bt, self.state[row], self.state[row], chunk)
        out = out.reshape(C, self.num_v, self.dv)
        return self.norm(out, z.reshape(C, self.num_v, self.dv)).reshape(C, self.vdim)

    def prefill(self, x: torch.Tensor, _positions=None, _ctx=None, _tiles=None,
                chunk: int = 64) -> torch.Tensor:
        """``x`` [C, hidden] -> [C, hidden]；更新卷积状态和 delta 状态。

        Args:
            x: 预填充输入，``[C, hidden]``。
            _positions: 忽略。它存在是因为 ``Layer.prefill`` 用同一个签名调用两种
                mixer；线性 mixer 构造上就是因果的，不需要位置表。
            _ctx: 忽略，理由同上。
            _tiles: 忽略，理由同上。
            chunk: delta 扫描里每个分块的 token 数。

        Returns:
            ``[C, hidden]``，经 ``out_proj`` 之后的层输出，顺序与 ``x`` 一致。
        """
        proj = self.prefill_project(x)
        out = self.prefill_state(*proj, row=self.prefill_row, chunk=chunk)
        return self.out_proj.prefill(out)

    def reset(self) -> None:
        """在请求之间清空递推状态。

        Returns:
            无。两个缓冲区都必须归零：残留的卷积窗口会让下一个请求的首批 token
            看到上一个请求的尾巴，那是任何位置计数器都补救不回来的。
        """
        self.conv_state.zero_()
        self.state.zero_()
