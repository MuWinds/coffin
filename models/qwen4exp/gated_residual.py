"""超连接（Gated Residual）：读门把 4 路残差流塌成 1 路，写门把块输出广播回 4 路。

Qwen4-Exp 的主干宽度是 ``hc_count * hidden_size``，每一层没有 pre-norm——归一化
全在这里，作用在 4 路残差流上（``group_size = hidden_size``，每路各做一次）。
一个 block 的输出被复制到 4 路残差流上、每路乘一个自己的标量门，所以写门输出是
``[B, S, hc_count]`` 而不是逐通道的。

Module: ``coffin/models/qwen4exp``。依赖同族的 ``norm`` 与 ``proj``；被 ``blocks``
和 ``model`` 调用。另外 import ``backends`` 判断本后端有没有那三个超连接内核。

Constraints:
    * 读门最后是 ``.mean(dim=-2)`` 不是 ``sum``。用 sum 会让读门输入的尺度随
      ``hc_count`` 线性放大 4 倍，形状全对、数值全错（``ARCH_SPEC.md`` §6.2）。
    * 两个门都读**归一化后**的状态，且低秩投影输出先除以 ``hc_count`` 再进
      silu；写门是 ``2 * sigmoid(w / hc_count)``——漏掉 2 会让整个 block 的贡献
      减半，不除 4 会让门饱和（``ARCH_SPEC.md`` §6.3）。
    * ``use_combine=False`` 的末端 mixer 没有 ``block_inject_weight``，forward
      只返回塌缩后的一路；这条分支就是 10240 → 2560 的收口。
    * 融合路径只在后端**原生**提供那三个算子、**且**接线已验证（``_HYPER_WIRED``）、
      **且**当前值得开启（``_HYPER_FASTER``）时才走。判据必须是 ``caps().native``：
      参考后端会把每个名字都补齐（它的实现就是这条链的规范），拿算子表里有没有这一格
      当开关等于在没有内核的平台上把八次 torch 调用换成三次候选调用，只会更慢。
    * ``_HYPER_WIRED`` 与 ``_HYPER_FASTER`` 是两件事，不能合成一个常量：前者说语义
      对，后者说值得开。当前前者为真、后者为假——原因与实测数字见
      ``_HYPER_FASTER`` 的注释（``hyper.cu`` 的并行度，不是模型侧）。
    * 融合路径要把三处中间结果物化成调用方持有的缓冲区，并且**每步复用**：读门和
      写门读的是同一份归一化结果，重算一次就是多一整个 ``[B, S, 10240]`` 的读写。
"""

from __future__ import annotations

import os

import torch
from torch import nn

from backends import caps as _caps
from backends import ops as bops
from models.qwen4exp.config import TextConfig
from models.qwen4exp.norm import RMSNorm
from models.qwen4exp.proj import Proj

#: 融合超连接缺一不可的三个内核条目。
#:
#: 命中判定用 ``caps().native`` 而不是"算子表里有没有这个字段"：表在任何后端上都是
#: 满的（参考实现把每个名字都补齐了），拿它当开关等于在没有内核的平台上接上一串
#: 候选调用。理由与 ``norm._has_native`` / ``gdn._DECODE_OPS`` 相同。
_HYPER_OPS = frozenset({"hyper_norm", "hyper_down", "hyper_up"})

#: 融合超连接是否已**接线正确并验证过语义**。
#:
#: 2026-11-02 真机（作业 4548/4551，gpu-node-06，48 层 W4A8 EP=4 单卡，真 n-gram 表）
#: ``COFFIN_HYPER_FUSED=off`` 对 ``on`` 两臂对照：
#:
#:   判据            结果
#:   logits absmax   on/off = 1.0101        （折错 1+w 会是约 4 倍档）
#:   四个出口张量    std 差 <2%，inject 差 <1.2%
#:   贪心 ids        on 臂与 q4base2 基线**逐位相同**，off 臂从第 3 位起分叉
#:
#: 最后一条是这轮最重要的发现：融合核用 fp32 累加、只舍入一次，而纯 torch 链在 bf16
#: 上分次舍入——**融合在数值上比它替掉的那条链更接近参考实现**。所以这条路径的价值
#: 不只是"可能省时间"。
_HYPER_WIRED = True

#: 融合超连接当前是否**真的更快**。接线正确与值得开启是两件事，所以是两个常量。
#:
#: **当前为 ``True``**。2026-11-03 真机（gpu-node-06，48 层 W4A8 EP=4 单卡，真
#: n-gram 表）三轮迭代，每轮都是同树两臂独立进程：
#:
#:   作业       off 臂      on 臂       收益      hyper_up    hyper_down
#:   4551       232.82      304.98     0.78x     510.6 us    579.3 us   （慢）
#:   4556       232.82      222.10     1.048x    193.3 us     75.8 us
#:   4563       236.70      215.45     1.099x    114.0 us     75.7 us
#:
#: 4551 那一轮是 net 负收益，原因**不在模型侧**：``hyper.cu`` 的 ``splits`` 公式在
#: ``rows == 1`` 时只给出 6 片、grid=(3,1,6)=2304 条线程 = **4.5 warps/SM**（屋顶 64），
#: 有效带宽只有 11-13 GB/s。修法在 ``hyper.cu``：按「需要多少块才喂满
#: 16 SM × 64 warp」反推片数（``pick_hyper_split``）、内层点积拆 4 条独立累加链
#: （``dot_slice``）、写门改成整块线程按 ``t`` 分摊。两核合计 104.6 -> 18.2 ms/步。
#:
#: **``hyper_up`` 到此为止，不要继续硬压。** 它现在 114.0 us、58.4 GB/s（屋顶的
#: 10.2%），而 ``hyper_down`` 是 75.7 us、86.6 GB/s。差距不是并行度问题：
#: ``hyper_up`` 的 MAC 只有 ``hyper_down`` 的 1/4（2560×320 对 320×10240），却多背了
#: sigmoid、逐流 mean、写门三段非线性；写门要扫 ``wide``=10240 维而只有 ``hc``=4 个
#: 输出，是 M=1 下单块的结构性限制（它在旧分派下单独调用就是 134.9 us）。继续加
#: ``splits`` 只会把部分和的写入摊上来，不会再降。
#:
#: **两臂的 ids 会翻转，这不是 bug，不要靠调参去凑。**
#:
#: 两条等价的算术路径（融合核 fp32 累加、只舍入一次；纯 torch 链在 bf16 上分次
#: 舍入）在**最后一位**上不必逐位相同，于是贪婪选择的吸引子会翻转——而且**翻转方向
#: 随其它改动变化**，不是"某一条路更接近参考实现"：
#:
#:   作业 4556  off 落 [220, 20, 220, 20, ...]  on 落 [220, 19, 15, ...]
#:   作业 4563  off 落 [220, 20, 15, 15, ...]   on 落 [220, 20, 220, 20, ...]
#:
#: （4556 那轮我曾据此推断"融合更接近参考实现"，被 4563 那轮的对调否掉了。）
#:
#: 判据是 ``HYPER_SAMPLE`` 四个出口张量的量级与末层 logits 的 absmax 比，**不是 ids
#: 相等**。去调参凑 ids 会把一个正确的实现调成另一个正确的实现，而掩盖真正的错误。
_HYPER_FASTER = True

#: ``COFFIN_HYPER_FUSED`` 的取值：``on`` 强制走融合、``off`` 强制走纯 torch 链。
#: **它存在的理由是逃生口，不是实验开关。** 默认路径由 :data:`_HYPER_FASTER` 决定，
#: 所以不设这个变量时行为完全由代码里的结论控制。留着它是为了两种真实情形：
#:
#: * ``hyper.cu`` 在这张卡上又变慢（``_HYPER_FASTER`` 翻成 ``False`` 之前想先量一下），
#:   在**不重编译**的前提下拿回那条已知可用的 torch 链；
#: * 怀疑数值问题时做一次隔离，确认症状是否跟着融合走。
#:
#: ``on`` 会**绕过** :data:`_HYPER_FASTER`。这是刻意的：一旦那个常量被翻成 ``False``，
#: 验收探针就再也进不了融合分支，而"``.cu`` 改完到底快没快"正是要靠那条分支回答的。
#: 所以 ``on`` 表示"别被那个结论挡住"，不表示"它一定更快"。
#:
#: 它是 import 期读的常量，所以两条路必须在**独立进程**里跑——同进程改不了，
#: 而同进程改了却没生效的日志读起来像"两臂一样快"。
_HYPER_MODE = os.environ.get("COFFIN_HYPER_FUSED", "").strip().lower()

#: 融合超连接只在**解码量级**的行数上开启（``rows * hc * hidden`` 的激活很小）。
#:
#: 三个 ``hyper_*`` 内核是照着 ``M == 1`` 手工调的：``pick_hyper_split`` 从「要多少块
#: 才喂满 16 SM」反推 split-K 片数，``hyper_up`` 的写门按整块线程分摊。这套调优序列
#: 在 decode 上实测有效（``hyper_up`` 510.6 → 114.0 us、``hyper_down`` 579.3 → 75.7 us，
#: 见 ``_HYPER_FASTER`` 上的三轮记录）。**但同一个核在 prefill 上不是这条曲线。**
#:
#: 作业 ``q4prebase2`` 的 prefill 设备账本（``seq=2048``，9.66 s）暴露了这件事：
#:
#:   moe_grouped   2221 ms / 96 次 =  23.1 ms/次
#:   hyper_down    1550 ms / 96 次 =  16.1 ms/次   <-- 31.9% 在这一对融合核上
#:   hyper_up      1536 ms / 96 次 =  16.0 ms/次
#:   w4_gemm_tcu    925 ms / 317 次 =  2.9 ms/次
#:
#: 稠密投影走的是张量核 GEMM（2.9 ms/次），而这两个手工 GEMV 核每次要 16 ms —— 慢 5.5
#: 倍。原因就是它们没有「大 M 走 GEMM」的形态：``hyper_down`` 是 ``[rows,10240] x
#: [10240,320]``、``hyper_up`` 是 ``[rows,320] x [320,10240]``，在 ``rows=2048`` 时
#: 都是标准 GEMM，而内核还在按「一块一列、K 轴切片累加」的 GEMV 结构跑。
#:
#: 所以按行数分档：**超过这个数就退回纯 torch 链**——那条链里的 ``Proj`` 会走到
#: ``linear_prefill`` → ``w4a8_gemm_tcu``/``matmul``，正是 prefill 该走的路。分界取
#: :data:`proj._SMALL_ROWS`（8）之上的一档，因为 decode 的批量也是 1-8 行，而 16 行
#: 以上已经确定是 prefill 的领地。
_HYPER_FUSED_MAX_ROWS = 16

#: 读门用「逐路乘加」而不是 ``(w * x).mean(-2)`` 的最小行数（``B * S``）。
#:
#: 两种写法算术完全一样（都是 ``sum_c w_c * x_c / hc``），差在物化与归约路径上。
#: 实测一次（作业 335，``[B=1, S=512, hc=4, H=2560]`` bf16，纯设备时间）：
#:
#:     (w * x).mean(-2)     279.46 us   ← 还要先物化 10 MB 的乘积，再走沿 dim=-2
#:                                        的 4 元跨步归约；字节量只有 12.9 MB（屋顶 22 us）
#:     逐路乘加              162.15 us   ← 同样的算术摊在 [B,S,H] 的 2.6 MB 连续张量上
#:     转成 [B,S,H,4] 再 mean  853.17 us ← 布局转换那次拷贝把收益吃光
#:
#: 差额随行数线性缩小，而逐路乘加固定多 3 次发射（约 15 us），交叉点约 66 行，
#: 所以取 64：decode（1 行）继续走 ``mean``，prefill 分块（512 行）走逐路乘加。
_MEAN_CHAIN_MIN_ROWS = 64


class GatedResidual(nn.Module):
    """一路读门（低秩 + swish）加一路标量写门。

    Attributes:
        hc_count: 残差流路数。
        hidden_size: 单路宽度。
        hc_norm: 作用在 ``[B, S, hc_count * hidden_size]`` 上、按 ``hidden_size``
            分组的零中心 RMSNorm。
        input_mix_weight_down: 读门的降维投影。
        input_mix_weight_up: 读门的升维投影。
        block_inject_weight: 写门；``use_combine=False`` 时为 ``None``。
    """

    def __init__(self, config: TextConfig, use_combine: bool = True) -> None:
        """建两套门。

        Args:
            config: 文本配置，用来取 ``hc_count`` / ``hidden_size`` /
                ``hc_lowrank`` / ``rms_norm_eps``。
            use_combine: 是否建写门。末端 mixer 用 ``False``，层内的两个用
                默认值。
        """
        super().__init__()
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        hc_hidden_size = self.hc_count * self.hidden_size
        self.hc_norm = RMSNorm(hc_hidden_size, group_size=self.hidden_size,
                               eps=config.rms_norm_eps)
        self.input_mix_weight_down = Proj(hc_hidden_size, config.hc_lowrank)
        self.input_mix_weight_up = Proj(config.hc_lowrank, hc_hidden_size)
        # 只有层内的两套需要写门：它们要把 block 输出写回 4 路残差流。末端
        # mixer 只做塌缩，多一个写门就是多一份永远不用的参数。
        #
        # 写门只有 hc_count = 4 行，是整步里最窄的一档（4 x 10240）：行主序下实测
        # 1.792 ms，换 ``[K, N]`` 后 0.134 ms（作业 4489）。48 层乘两套就是 96 个，
        # 它们和宽投影走同一条路。
        self.block_inject_weight = (
            Proj(hc_hidden_size, self.hc_count) if use_combine else None)
        # 融合路径的中间缓冲区，按 (前导形状, 设备, dtype) 惰性分配。不能注册成
        # buffer：``.to(dtype=...)`` 会把它们一起搬走，而 ``hyper_up`` 的写门目的地
        # 契约是 fp32，转窄之后内核会拒绝启动。惰性分配之后地址不再变，所以第一次
        # 解码步之后再捕获图是安全的（与 ``gdn._decode_scratch`` 同一套理由）。
        self._fused_bufs: dict = {}
        #: ``hc_norm.weight``（**未加 1**）的 fp32 常驻副本，见 :meth:`_hyper_gain`。
        self._gain_cache: tuple | None = None

    def _fused_ready(self) -> bool:
        """本后端是否真的能跑融合超连接。

        每次进 forward 都重算：``caps()`` 自己带缓存，而模块可能在 ``to()`` / 重新
        装载之后换了设备，一次性缓存的结果会指向上一台机器上分配过的缓冲区。

        ``use_combine=False`` 的末端 mixer 不走融合：它的第二、三个内核要用写门权重，
        而末端没有写门。整机只有一套末端（它是 ``CausalModel`` 的收口），而层内是
        96 套，所以省发射的收益全在前者之外，为它造一份假写门权重不值得。

        Returns:
            三个内核都在 ``caps().native`` 里、有写门、接线已验证、且当前值得开启时
            为 ``True``。

        ``_HYPER_FASTER`` 也参与判定：接线正确不等于值得开，见那个常量上的实测记录。
        所以即使后端原生支持、即使 ``COFFIN_HYPER_FUSED=on``，它仍可能是 ``False``
        ——那时 ``on`` 只表示"别被这个开关额外挡住"，最终仍由 ``_HYPER_FASTER`` 决定。

        Raises:
            ValueError: ``COFFIN_HYPER_FUSED`` 是个不认识的值。静默回退会让一次 A/B
                看起来像"两臂一样快"，而实际两臂跑的是同一条路——与
                ``COFFIN_I8_GEMM`` 同一条理由。
        """
        if _HYPER_MODE == "off":
            return False
        if _HYPER_MODE not in ("", "on"):
            raise ValueError(
                f"COFFIN_HYPER_FUSED={_HYPER_MODE!r} is not one of 'on' / 'off'")
        # 显式 ``on`` 允许绕过 ``_HYPER_FASTER``，那是为了在 ``.cu`` 优化落地前就能
        # 重跑 A/B 验收——否则一旦置 ``False``，验收探针就再也进不了融合分支，
        # 而"改完 .cu 到底快没快"正是要靠它回答的。
        if _HYPER_MODE == "on":
            return (_HYPER_WIRED and self.block_inject_weight is not None
                    and _HYPER_OPS <= _caps().native)
        return (_HYPER_WIRED and _HYPER_FASTER
                and self.block_inject_weight is not None
                and _HYPER_OPS <= _caps().native)

    def _fused_ok_for_rows(self, rows: int) -> bool:
        """本后端的融合超连接在这个行数下是否值得走。

        三个 ``hyper_*`` 核是照 ``M == 1`` 调的，在 prefill 的行数上会输给 GEMM：
        作业 ``q4prebase2`` 的 ``seq=2048`` 账本里 ``hyper_down``/``hyper_up`` 各
        16.1 ms/次（96 次共 3086 ms，占 31.9%），而同一份工作走 ``w4_gemm_tcu``
        只要 2.9 ms/次。见 :data:`_HYPER_FUSED_MAX_ROWS`。

        Args:
            rows: 本次前导维展平后的行数。

        Returns:
            行数在融合核的调优区间内为 ``True``；否则应由纯 torch 链（其 ``Proj``
            会在大 M 上走 ``linear_prefill``）接手。
        """
        return rows <= _HYPER_FUSED_MAX_ROWS

    def _hyper_gain(self) -> torch.Tensor:
        """返回 ``hc_norm.weight`` 的 fp32 副本，权重变了就重拷。

        **不折 ``1 + w``**。``hyper_norm`` 内核自己在 fp32 里算 ``v * inv * (1 + w)``
        （``hyper.cu`` 的 ``grouped_rms_norm_kernel``），与这里传 ``_folded_gain()``
        再让内核加一次 1 的效果是 ``2 + w``：形状全对、数值全错，而且它只影响 48 层
        的每一处归一化，logits 会整体偏移。这正是 ``norm.py`` 文件头点名的那个陷阱
        的镜像——那边是「别人已经折好了」，这边是「内核自己会折」。

        缓存是必须的而不是优化：``.float()`` 每次调用就是一次逐元素启动，96 套乘一次
        就是 96 次，正好把融合省下来的一部分还回去。失效靠 ``weight._version``，与
        ``RMSNorm._folded_gain`` 同一条约定。

        Returns:
            ``[hc_count * hidden_size]`` 的 fp32 零中心权重，驻留原设备。
        """
        version = (self.hc_norm.weight._version, self.hc_norm.weight.device)
        if self._gain_cache is not None and self._gain_cache[0] == version:
            return self._gain_cache[1]
        with torch.no_grad():
            gain = self.hc_norm.weight.detach().float()
        self._gain_cache = (version, gain)
        return gain

    def _fused_scratch(self, lead: tuple[int, ...], device: torch.device,
                       dtype: torch.dtype) -> dict:
        """取（必要时分配）融合超连接的三块中间缓冲区。

        Args:
            lead: 前导形状，即 ``hyper_input.shape[:-1]``。
            device: 缓冲区所在设备。
            dtype: 归一化结果的 dtype，跟随入参。

        Returns:
            ``normed`` / ``low`` / ``mixed`` / ``inject`` 四个键：读门塌缩结果、低秩
            输出、写门。前三个跟随激活 dtype，写门目的地按内核契约是 fp32。
        """
        key = (lead, device, dtype)
        cached = self._fused_bufs.get(key)
        if cached is not None:
            return cached
        hc_hidden = self.hc_count * self.hidden_size
        lowrank = self.input_mix_weight_down.out_features
        buf = {
            "normed": torch.empty(*lead, hc_hidden, dtype=dtype, device=device),
            "low": torch.empty(*lead, lowrank, dtype=dtype, device=device),
            "mixed": torch.empty(*lead, self.hidden_size, dtype=dtype, device=device),
            # 写门目的地按内核契约必须是 fp32，而模型侧契约是激活 dtype——两者不一致是
            # 承重的，见 :meth:`_fused_forward` 里的收窄。
            "inject": torch.empty(*lead, self.hc_count, dtype=torch.float32,
                                  device=device),
            "inject_act": torch.empty(*lead, self.hc_count, dtype=dtype,
                                      device=device),
        }
        self._fused_bufs[key] = buf
        return buf

    def _fused_forward(self, hyper_input: torch.Tensor):
        """用三个融合内核跑一遍超连接，一次 forward 共三次启动。

        与纯 torch 链的分工差异只有一处：归一化、``silu(v / hc)``、``sigmoid`` 门、
        逐流加权 mean 与写门全都留在内核里，``torch`` 侧只负责把权重交出去。

        Args:
            hyper_input: 4 路残差流，``[..., hc_count * hidden_size]`` 连续。

        Returns:
            ``(mixed_input, hyper_input, injection_weights)``，语义与 :meth:`forward`
            的 ``use_combine=True`` 分支完全相同。

        Raises:
            ValueError: ``hyper_input`` 不是连续布局。内核按 ``ptr + row * width``
                直接寻址，带间距的视图会被内核的 TORCH_CHECK 拒绝，而先
                ``contiguous()`` 会静默拷贝一份 10240 宽的张量——那正好是这条路径
                要省掉的开销。
        """
        if not hyper_input.is_contiguous():
            raise ValueError("fused hyper path needs a contiguous residual stream")
        lead = hyper_input.shape[:-1]
        device, dtype = hyper_input.device, hyper_input.dtype
        buf = self._fused_scratch(lead, device, dtype)
        lowrank = self.input_mix_weight_down.out_features
        bops.hyper_norm(hyper_input, self._hyper_gain(), buf["normed"],
                        self.hidden_size, self.hc_count, self.hc_norm.eps)
        bops.hyper_down(buf["normed"], self.input_mix_weight_down._source,
                        buf["low"], self.hc_count)
        bops.hyper_up(buf["normed"], self.input_mix_weight_up._source, buf["low"],
                      self.block_inject_weight._source, buf["mixed"], buf["inject"],
                      self.hidden_size, self.hc_count, lowrank)
        # 写门必须收窄回激活 dtype 再返回。内核的契约是 fp32（`hyper_up.cu` 的
        # ``TORCH_CHECK(inject.scalar_type() == torch::kFloat)``），但**模型侧的契约是
        # 激活 dtype**：``apply_injection`` 拿它去乘 ``[B, S, hidden]`` 的块输出，而
        # 那个乘积直接并进 4 路残差流。漏掉这一次舍入会把整条残差流抬成 fp32，
        # 下一层的 ``b = in_proj_b(fp32)`` 跟着变宽，于是 GDN 的融合核在第 1 层之后
        # 抛 ``fused decode needs bf16 beta/decay logits``（作业 4547 的 on 臂）。
        #
        # 这与 ``gdn._fused_decode`` 末尾那次 ``.to(dtype)`` 是同一条规则，也是
        # ``gdn.py`` 在 ``_DECODE_FUSED_VERIFIED`` 上记过的同一个失败：**内核输出缓冲的
        # dtype 不决定模型侧的 dtype，适配器负责收口。**
        inject_act = buf["inject_act"]
        inject_act.copy_(buf["inject"])
        return buf["mixed"], hyper_input, inject_act

    def combine_and_mix(self, hyper_input: torch.Tensor,
                        block_output: torch.Tensor,
                        injection_weights: torch.Tensor | None
                        ) -> tuple[torch.Tensor, torch.Tensor]:
        """一次算出「未塌缩的 4 路状态」与「塌缩后的单路」，给 MTP 头收口。

        MTP 头每步要吐两样东西：塌缩后的 ``[B, S, H]`` 喂 lm_head，未塌缩的
        ``[B, S, hc*H]`` 喂下一个 draft step。这两样是**同一次写回**的两个视角，
        分开算会把 ``apply_injection`` 的 ``[B,S,hc,H]`` 中间量做两遍。

        Args:
            hyper_input: 写回前的 4 路状态，``[B, S, hc_count * hidden_size]``。
            block_output: 末段 MoE 的输出，``[B, S, hidden_size]``。
            injection_weights: 逐流写门，``[B, S, hc_count]``；``None`` 表示**单位
                权重**（每路都乘 1）。MTP 的嵌入残差就走这一支。

        Returns:
            ``(mixed, multi)``：``mixed`` 是读门塌缩后的 ``[B, S, hidden_size]``，
            ``multi`` 是写回后的 4 路状态 ``[B, S, hc_count * hidden_size]``。

        Raises:
            ValueError: 前导形状对不上；或本模块带写门（``use_combine=True``）却收到
                ``None``。层内的两个 mixer 必须用学到的写门，把 ``None`` 当单位权重
                会静默丢掉那 96 套参数。

        ``hyper_input`` 必须连续，理由与 :meth:`_fused_forward` 相同：融合核按
        ``ptr + row * width`` 寻址。
        """
        if self.block_inject_weight is not None and injection_weights is None:
            raise ValueError(
                "combine_and_mix() with injection_weights=None is only for the "
                "use_combine=False mixer (no learned write gate to apply)")
        if block_output.shape[:2] != hyper_input.shape[:2]:
            raise ValueError(
                "combine_and_mix shape mismatch: "
                f"block {tuple(block_output.shape)}, hyper {tuple(hyper_input.shape)}.")
        if injection_weights is None:
            # 单位权重：块输出复制到每一路、不乘门。拓扑与 apply_injection 相同
            # （hc 外、H 内），只是那个乘子恒为 1。
            injection = block_output.unsqueeze(-2).expand(
                *block_output.shape[:-1], self.hc_count, self.hidden_size)
            multi = hyper_input + injection.flatten(-2)
        else:
            multi = apply_injection(block_output, hyper_input, injection_weights)
        return self(multi), multi

    def forward(
        self, hyper_input: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        """读门塌缩，可写门时一并给出 4 路状态与逐流写门。

        后端原生提供 ``hyper_norm`` / ``hyper_down`` / ``hyper_up`` 且
        ``_HYPER_WIRED`` 且当前值得开启时走融合核（三次启动），否则走下面那条纯
        torch 链（八次）。两条路的语义完全相同，见 ``hyper.cu`` 的 Constraints。

        Args:
            hyper_input: 4 路残差流，``[B, S, hc_count * hidden_size]``。

        Returns:
            ``use_combine=False`` 时只返回 ``mixed_input``，``[B, S, hidden_size]``；
            否则返回 ``(mixed_input, hyper_input, injection_weights)``，其中
            ``injection_weights`` 是 ``[B, S, hc_count]``。

        Raises:
            ValueError: ``hyper_input`` 的最后一维不是
                ``hc_count * hidden_size``。这个错误必须在这里炸：让广播把它吃掉
                之后，读门会拿错位的流做加权，输出仍然"看起来像"激活。
        """
        if hyper_input.shape[-1] != self.hc_count * self.hidden_size:
            raise ValueError(
                f"Expected {self.hc_count * self.hidden_size} hyper-connection features, "
                f"got {hyper_input.shape[-1]}.")

        # 融合路径在语义上与下面那条链逐项等价（三条都写在 hyper.cu 的 Constraints
        # 里），所以换路只减少发射、不改变算术顺序之外的任何东西。
        #
        # 但**只在解码量级的行数上**换：三个融合核是照 M == 1 调的，prefill 的大 M
        # 上它们每次 16 ms，而同一份工作走张量核 GEMM 只要 2.9 ms（见
        # :data:`_HYPER_FUSED_MAX_ROWS` 上 q4prebase2 的账本）。超出行数时退回纯
        # torch 链，那条链里的 ``Proj`` 会自己走 ``linear_prefill``。
        if self._fused_ready() and self._fused_ok_for_rows(hyper_input.numel()
                                                            // hyper_input.shape[-1]):
            return self._fused_forward(hyper_input)

        hyper_input_normed = self.hc_norm(hyper_input)
        # 先除 hc_count 再进 silu：低秩投影的输出宽度是 hc_hidden_size，除以路数
        # 把读门输入的尺度归一化到与单路状态同量级。
        input_mix_weight = torch.nn.functional.silu(
            self.input_mix_weight_down(hyper_input_normed) / self.hc_count)
        input_mix_weight = torch.sigmoid(self.input_mix_weight_up(input_mix_weight))
        input_mix_weight = input_mix_weight.unflatten(-1, (self.hc_count, self.hidden_size))
        # 门乘的是归一化后的 4 路状态，最后取均值：mean 让读门输出与 hc_count
        # 无关，换成 sum 会把尺度放大 4 倍。
        #
        # 行数够多时逐路乘加：见 :data:`_MEAN_CHAIN_MIN_ROWS` 的实测（同一次
        # ``[1,512,4,2560]`` 上 279.46 -> 162.15 us）。decode 只有 1 行，多出来的发射
        # 比省下的带宽贵，所以那一档保持 ``.mean(-2)``。
        streams = hyper_input_normed.unflatten(-1, (self.hc_count, self.hidden_size))
        rows = streams.numel() // streams.shape[-1]
        if rows >= _MEAN_CHAIN_MIN_ROWS:
            # ``[..., c, :]`` 是 [B,S,H] 的连续视图，读得整齐；乘积不再单独物化。
            mixed_input = input_mix_weight[..., 0, :] * streams[..., 0, :]
            for channel in range(1, self.hc_count):
                mixed_input.addcmul_(input_mix_weight[..., channel, :],
                                     streams[..., channel, :])
            mixed_input.mul_(1.0 / self.hc_count)
        else:
            mixed_input = (input_mix_weight * streams).mean(dim=-2)
        if self.block_inject_weight is None:
            return mixed_input
        # 乘 2 让零输入处 w = 2*sigmoid(0) = 1，于是初始化附近等价于原始残差相加。
        injection_weights = 2 * torch.sigmoid(
            self.block_inject_weight(hyper_input_normed) / self.hc_count)
        return mixed_input, hyper_input, injection_weights


def apply_injection(hidden_states: torch.Tensor, hyper_input: torch.Tensor,
                    injection_weights: torch.Tensor) -> torch.Tensor:
    """把块输出按逐流标量门写回 4 路残差流。

    Args:
        hidden_states: 块输出，``[B, S, hidden_size]``。
        hyper_input: 写回前的 4 路状态，``[B, S, hc_count * hidden_size]``。
        injection_weights: 逐流写门，``[B, S, hc_count]``。

    Returns:
        写回后的 4 路状态，形状与 ``hyper_input`` 相同。

    Raises:
        ValueError: 三者的 batch/序列长度对不上。
    """
    if hidden_states.shape[:2] != hyper_input.shape[:2] or \
            injection_weights.shape[:2] != hidden_states.shape[:2]:
        raise ValueError(
            "residual write-back shape mismatch: "
            f"block {tuple(hidden_states.shape)}, hyper {tuple(hyper_input.shape)}, "
            f"gates {tuple(injection_weights.shape)}.")
    # 写回是"同一份块输出复制到每一路、每路乘自己的标量门"，不是只写一路。
    injection = hidden_states.unsqueeze(-2) * injection_weights.unsqueeze(-1)
    return hyper_input + injection.flatten(-2)


__all__ = ["GatedResidual", "apply_injection"]
