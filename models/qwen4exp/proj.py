"""稠密投影与词表嵌入：按激活行数挑后端入口，不再直接调 ``F.linear``。

这个文件存在的理由是一次真机账本（作业 4481，48 层 W4A8 单卡 decode 1149.61 ms）：
设备时间 1093 ms 里 ``GemmEx_kernel::gemmh`` 737 次占 900.14 ms（82.4%），MoE 分组
内核只有 24.46 ms（2.2%）。慢的不是专家而是稠密投影，而根因是布局——checkpoint 的
``[N, K]`` 行主序在本卡 M=1 下实测只有 3.2-31.9 GB/s，``[K, N]`` 加自写 GEMV 是
193-433 GB/s。同一份权重、同一个算术，逐形状快 11.8-17.9 倍。窄投影更极端：216 个
``N <= 48`` 的投影一步 245.2 ms，只换布局（权重仍是 bf16）就降到 21.2 ms（作业 4489），
所以 bf16 路径也必须走后端，不能退回 ``F.linear``。

Module: ``coffin/models/qwen4exp``。族里唯一 import ``backends`` 的地方：布局转换与
查表 dtype 都是平台事实，只能由后端回答。被 ``gdn`` / ``ngram`` / ``gated_residual`` /
``moe`` / ``model`` 使用，本模块不 import 它们。``moe`` 用的是 :func:`dispatch_linear`
而不是 :class:`Proj`——路由矩阵是 ``nn.Parameter`` 而不是模块，但它必须走同一套派发。

Constraints:
    * 布局转换只许发生一次，而且只许由后端做。``bops.linear*`` 对裸 bf16 权重内部走
      ``pack.as_kn`` 并按对象身份记忆化；这里再转一次等于把权重多读一遍，而那正是
      窄投影账本里 245.2 -> 21.2 ms 的来源。
    * 权重有三种状态：bf16 ``nn.Parameter``、``W4A8Weight``、以及「由量化产物提供但
      还没挂上」。第三种必须让前向立刻报错，绝不静默退回 bf16 查表/GEMM——那会同时
      丢掉 6-20 倍的速度和 5.39 GiB 常驻，而形状与数值都正常，只有账本能发现。
    * ``reshape(-1, K)`` 只在最后一维 stride 为 1 时是视图；其余情况先 ``contiguous()``，
      否则 ``reshape`` 会静默拷贝一份激活。
    * 行数分界只决定「语义入口」，不决定内核：``linear`` 收 ``[K]``、``linear_m`` 收小
      ``[M, K]``、``linear_prefill`` 收大 ``[C, K]``，每个后端还会在内部再判一次
      （corex 的 ``_ROW_CHUNK = 8``、``_PREFILL_MIN_M = 64``）。在这里按形状另写一套
      调优判断，只会对某一张卡正确。
    * 量化路径的模型必须在 ``torch.device("meta")`` 下构造：占位参数虽然在构造末尾就被
      摘掉，但摘之前已经建过一份，48 层的稠密投影是 9 GiB 量级（``moe.Experts`` 记着
      同一次 OOM）。
    * 同激活投影的融合组只在 decode 的**单行**前向上生效，且只在权重全部由 W4A8 供
      货的成员之间生效；BF16 回填的窄投影走 ``_gemv_rows``，那条路的权重布局与内核
      都不同，拼进去只会把两条路搅在一起。组命不中时一次都不许多算，命中率长期不到
      一半就自愈停用并释放拼接权重（见 :meth:`_FusedGroup.try_forward`）。
"""

from __future__ import annotations

import os
import weakref

import torch
import torch.nn.functional as F
from torch import nn

from backends import caps as _caps
from backends import ops as bops
from w4a8_weights import W4A8Weight

#: 量化路径查表输出用的 dtype。与 ``ops/embed.TiedEmbedding.lookup`` 同一条约定：
#: 不管权重怎么存，激活一律是后端声明的 ``act_dtype``——KV 缓存和每个内核都是照
#: 它写的，这里另挑一个 dtype 只会在下游某个算子处报一次错。
_ACT_DTYPE = _caps().act_dtype

#: 「少量行」的上界，也是 ``linear_m`` 与 ``linear_prefill`` 的分界。8 是模型侧事实
#: 而不是内核参数：一次前向校验最宽就是 8 个 token（``CKPT`` 的 capture 宽度），所以
#: 2..8 行是「所有行共用一次权重读取」的语义，再多就交给 GEMM。两个后端的换手点都
#: 落在这附近，所以这里只挑入口，具体走哪个内核仍由后端决定。
_SMALL_ROWS = 8


def dispatch_linear(w, x: torch.Tensor, n: int | None = None) -> torch.Tensor:
    """把一次 ``[..., K] -> [..., N]`` 投影按行数派给后端的三个语义入口。

    单独成一个函数，是因为 :class:`Proj` 之外还有一处必须走同一条路：MoE 的路由
    矩阵。它在一步 decode 里是 48 次 ``[1, 2560] x [2560, 512]``，原先直接
    ``F.linear(x, weight)``，吃的是 checkpoint 的 ``[N, K]`` 行主序，本卡上实测
    共 **30.234 ms / 一步**（作业 q4land6 的 ``aten::linear`` 48 次、629.9 us 一次，
    占那一步设备时间的 11.1%）。而权重只有 2.62 MB，按本卡 575 GB/s 的上限读一遍
    是 4.6 us —— 差的三个数量级全在布局与派发上，不在算术上。

    它和 :meth:`Proj.forward` 共用这一段的理由不是省几行：两处的分派规则必须
    一起改，否则某天调 ``_SMALL_ROWS`` 会只生效一半，而那半边的症状是「某个投影
    慢回去了」，账本上只表现为一行数字变大。

    Args:
        w: 权重，契约布局 ``[N, K]``；也可以是后端已打包的形式（``W4A8Weight``、
            ``pack.KnWeight``），它们自己带 ``n``。
        x: 激活，``[..., K]``，最后一维的 stride 必须是 1，否则这里先拷一份。
        n: 输出宽度，只在激活为空批时用来造结果。``None`` 表示从 ``w`` 推断。

    Returns:
        与 ``x`` 同前导形状、最后一维为 ``N`` 的张量，dtype 同 ``x``。

    Raises:
        ValueError: ``n`` 既没给、``w`` 又没有可读的 ``n`` 或二维形状。空批时
            悄无声息地返回一个宽度错的结果，会让错误在很远的加法处才暴露。

    """
    lead = x.shape[:-1]
    # decode 一步里本函数被调约 485 次（``q4attrib3`` 按调用点量到），而每一步
    # ``stride`` / ``reshape`` / ``unsqueeze`` 都是一次 Python 分发。这里把
    # 「已经是 [1, K] 连续」这条最常见的情形短路掉：它占 decode 的绝大多数调用，
    # 而那种输入既不需要拷贝也不需要 reshape。
    if x.dim() == 2 and x.shape[0] == 1 and x.stride(-1) == 1 and x.is_contiguous():
        # 解码入口收 ``[K]``；契约就是这么分的，不要传 ``[1, K]``。
        return bops.linear(w, x[0]).unsqueeze(0)
    if x.stride(-1) != 1:
        x = x.contiguous()
    flat = x.reshape(-1, x.shape[-1])
    if flat.shape[0] == 0:
        # 空批不该进内核：多数内核至少要求一行，而空激活在数学上就是个空结果。
        width = n
        if width is None:
            width = getattr(w, "n", None)
        if width is None:
            if w.dim() != 2:
                raise ValueError(
                    f"cannot infer the output width of {type(w).__name__}; pass n")
            width = int(w.shape[0])
        return x.new_empty(*lead, width)
    if flat.shape[0] == 1:
        # 解码入口收 ``[K]``；这里不要传 ``[1, K]``，契约就是这么分的。
        out = bops.linear(w, flat[0]).unsqueeze(0)
    elif flat.shape[0] <= _SMALL_ROWS:
        out = bops.linear_m(w, flat)
    else:
        out = bops.linear_prefill(w, flat)
    return out.reshape(*lead, out.shape[-1])


#: 因为「量化产物没覆盖组内成员」而停用的组数，见 :func:`fusion_stats`。
_UNCOVERED_GROUPS = 0

#: ``Proj.forward`` 的单调调用序号。
#:
#: 融合组的信箱靠它判断「这次切片是不是同一次求值留下的」。用它而不是张量身份或
#: 版本号，是因为 decode 的激活经常是**同一块缓冲区**每步被重写（``gated_residual``
#: 的 ``_fused_bufs`` 就是复用的），身份与 ``_version`` 都分不出相邻两步；而序号在
#: 同一组的成员之间只差 1，跨步至少差几十。``0`` 只在测试里直接构造组时出现。
_SERIAL = 0


class _FusedGroup:
    """一组吃同一份激活的投影：合成一次量化 + 一次 N 拼接 GEMV。

    存在理由是账本，不是整洁：一步 decode 有 314 次 W4A8 投影（每步
    ``w4a8_dot`` 计数，作业 q4attrib3 的 3 步合计 942），而其中三簇天然吃同一份
    ``hidden_states``：

    * ``qsa.Attention.forward`` 里的 ``q_proj`` / ``k_proj`` / ``v_proj``（``K=2560``）；
    * ``gdn.GatedDeltaNet.forward`` 里的 ``in_proj_qkv`` 与 ``in_proj_z``（``K=2560``）。
      ``in_proj_a/b``
      在真机产物里没有被量化（``N=48``，装载器走 ``set_bf16``），于是
      ``gdn._fused_in_proj`` 的四路拼接整条失效——本组正好把那两个真正被量化的
      补回来；
    * ``moe.MLP.forward`` 的 ``gate_proj`` 与 ``up_proj``（``K=2560``、``N=512``，两者相邻
      调用；宿主类名是通用的 ``MLP``，所以档位判定按属性名而不是类名）。

    分开跑等于把同一份激活量化 3 次 / 2 次，并且各发一次 ``w4a8_dot`` 与
    ``w4a8_finish``。合成一次之后，量化次数、发射次数、主机侧的 ``_w4_rows``
    调用次数同时下降——这是这段代码唯一要买的东西。实测（作业 187，纯内核）：
    QSA 那三路 220.60 -> 89.26 us/层，且**逐位相同**（``absmax`` 差 0）。

    ``indexer.index_qk_proj`` 也吃同一份激活，但它挂在 ``QSAIndexer`` 下面而不是
    ``Attention`` 下面，按「只认同一个父模块的直接子模块」就收不进来。收它要往
    上找最小公共祖先，那会把跨层的同名投影放进同一个桶——它们吃不同的激活，
    代价是每次求值白算整组。省下的约 0.1 ms/步不值得这条风险。

    拼接由 :meth:`W4A8Weight.cat_output` 完成，沿 N 轴逐列独立，所以数值上只是
    「同一份算术算到更宽的 N」：int32 累加逐组精确，唯一的变化是 ``split-K`` 的
    分片数与 fp32 部分和的累加顺序（``pick_split_w4`` 依赖 ``N``），落在 fp32
    归约噪声里。

    Attributes:
        members: 组的成员，顺序即 N 轴上的排列顺序。
        boxes: 与 ``members`` 一一对应的 :class:`W4A8Weight`；``merged`` 由它们拼成。
        offsets: ``(起点, 宽度)``，按成员下标索引。
        merged: 拼好的 ``[K, N_total]`` 权重，惰性建一次。
        window: 信箱的有效窗口，单位是 ``Proj.forward`` 调用序号之差。
        hits / misses: 命中与「不得不自己算一遍」的次数，用于自愈。
        disabled: 长期命中率低于一半时置真，此后该组退回逐投影路径。
    """

    __slots__ = ("members", "boxes", "offsets", "merged", "keys", "mailbox",
                 "window", "hits", "misses", "disabled")

    def __init__(self, members: list["Proj"]) -> None:
        """登记成员；此时权重还没挂上，所以一次都不读它。

        Args:
            members: 同一模块下共享 ``in_features`` 的投影，至少两个。
        """
        self.members = list(members)
        self.boxes: list = []
        self.offsets: list[tuple[int, int]] = []
        self.merged = None
        self.keys: tuple | None = None
        self.mailbox = None
        self.window = len(members) + 1
        self.hits = 0
        self.misses = 0
        self.disabled = False

    def _reconcile(self) -> object | None:
        """按实际挂上的权重收窄组，惰性拼出合并权重。

        组在 ``release_quantized_placeholders`` 时建，那时每个投影的权重都还是
        ``None``：量化产物只覆盖它认识的那批键，剩下的由 ``Proj.set_bf16`` 补回，
        所以「这个组里到底有几个人真的是 W4A8」只有装载完成后才知道。第一次前向
        时核一次，之后权重不再变（换了对象的话 :attr:`keys` 会让它重建）。

        Returns:
            合并后的 :class:`W4A8Weight`；成员不足两个、``K``/``group_size``/设备
            不一致时返回 ``None``，此时整组停用而不是退化成更慢的路径。

        Raises:
            ValueError: 成员共享 ``K`` 但 ``group_size`` 或设备不同。拼接会在
                scale 轴上整体错位，形状却完全合法。
        """
        if self.disabled:
            return None
        kept = []
        boxes = []
        for proj in self.members:
            weight = proj._quantized
            if isinstance(weight, W4A8Weight):
                kept.append(proj)
                boxes.append(weight)
        if len(boxes) < 2:
            # 不是自愈那条路：量化产物只覆盖了组里的一个（或零个）成员，拼了也没用。
            # 这种情况 48 层会有上百个组，逐个打日志只会把真信号淹掉，所以只计数，
            # 由 :func:`fusion_stats` 的 ``uncovered`` 报出来。
            global _UNCOVERED_GROUPS
            _UNCOVERED_GROUPS += 1
            self.disabled = True
            self.mailbox = None
            return None
        keys = tuple(id(box) for box in boxes)
        if self.merged is not None and self.keys == keys:
            return self.merged
        first = boxes[0]
        for box in boxes[1:]:
            if (box.k, box.group_size) != (first.k, first.group_size):
                raise ValueError(
                    "fused projection group shares in_features but not K/group_size; "
                    "concatenating the scales would silently shift every group")
        offs = []
        offset = 0
        for box in boxes:
            offs.append((offset, box.n))
            offset += box.n
        # 被摘掉的成员必须清掉自己的组指针：它们留在 ``self.members`` 里的那份
        # 槽位已经指向别人的列段，继续走融合路径就会读到别人的输出。
        kept_ids = {id(proj) for proj in kept}
        for proj in self.members:
            if id(proj) not in kept_ids:
                proj._group = None
                proj._group_slot = -1
        self.members = kept
        self.boxes = boxes
        self.offsets = offs
        self.keys = keys
        self.merged = W4A8Weight.cat_output(boxes)
        self.window = len(kept) + 1
        for slot, proj in enumerate(kept):
            proj._group_slot = slot
        return self.merged

    def _retire(self, reason: str) -> None:
        """停用这个组并释放拼接权重，同时留一行日志。

        静默停用与「命中了但没收益」在账本上长得一模一样，而这条路径一旦被分错组
        就会白算整组，所以它必须留痕。

        Args:
            reason: 停用原因，写进日志。
        """
        if not self.disabled:
            print(f"[proj] 融合组停用（{reason}）：N={sum(n for _, n in self.offsets)}",
                  flush=True)
        self.disabled = True
        self.merged = None
        self.boxes = []
        self.mailbox = None

    def _mailbox_hit(self, x: torch.Tensor):
        """取出本次求值已经算好的整条输出；没有就返回 ``None``。

        Args:
            x: 成员收到的激活，``[..., K]``。

        Returns:
            ``[1, N_total]`` 的合并结果，或 ``None``。
        """
        box = self.mailbox
        if box is None:
            return None
        serial, ptr, shape, dtype, _src, merged_out, served = box
        if _SERIAL - serial > self.window:
            # 窗口外一律当过期：留着它只会在下一次求值里被误当成同一步的结果。
            self.mailbox = None
            return None
        if x.data_ptr() != ptr or tuple(x.shape) != shape or x.dtype != dtype:
            return None
        if served >= len(self.members):
            # 组里每个成员都取过一次了。下一次同组求值必须重新算，否则复用到的
            # 是**上一步**（或上一层）的结果——decode 的残差流缓冲区是复用的，
            # 地址与形状都不足以区分两次求值。
            self.mailbox = None
            return None
        self.mailbox = (serial, ptr, shape, dtype, box[4], merged_out, served + 1)
        # 第 5 项是激活本身，故意留的强引用：它的存储活着，这块地址才不会被别的
        # 张量复用，于是「同一个 data_ptr」等价于「同一份数据」。
        return merged_out

    def _store(self, x: torch.Tensor, out: torch.Tensor) -> None:
        """记住本次求值的合并输出，供同组其余成员按序取用。

        Args:
            x: 本次求值用的激活。
            out: ``[1, N_total]`` 的合并结果。
        """
        self.mailbox = (_SERIAL, x.data_ptr(), tuple(x.shape), x.dtype, x, out, 1)

    def _piece(self, out: torch.Tensor, slot: int, lead: tuple[int, ...]
               ) -> torch.Tensor:
        """从合并输出里切出一个成员的列段，并恢复它的前导形状。

        Args:
            out: ``[1, N_total]``。
            slot: 成员下标。
            lead: 调用方的前导形状，元素个数必须是 1。

        Returns:
            ``[*lead, N_i]``。列段本身是非连续视图，但前导维全是 1，所以
            ``reshape`` 是免费的视图，``qsa.py`` 下游的 ``.view(...)`` 也仍然成立
            （作业 187 的 ``VIEW_OK`` 就是量这一条）。
        """
        offset, width = self.offsets[slot]
        return out[:, offset:offset + width].reshape(*lead, width)

    def try_forward(self, proj: "Proj", x: torch.Tensor):
        """尝试走融合路径，返回 ``None`` 表示调用方应当照旧自己算。

        Args:
            proj: 调用方。
            x: 激活，``[..., K]``，前导维元素个数必须是 1（decode）。

        Returns:
            ``[*lead, N_i]``，或 ``None``。
        """
        if self.disabled or proj.bias is not None:
            return None
        slot = proj._group_slot
        if slot < 0:
            return None
        out = self._mailbox_hit(x)
        if out is not None:
            self.hits += 1
            return self._piece(out, slot, x.shape[:-1])
        merged = self._reconcile()
        if merged is None:
            return None
        flat = x.reshape(1, -1) if x.dim() != 2 else x
        out = bops.linear_m(merged, flat)
        self.misses += 1
        # 自愈：组若被分错（成员其实吃不同的激活），每次调用都会是一次 miss 并
        # 顺带算完整组。命中率长期不到一半就停用并释放拼接权重，宁可退回逐投影，
        # 也不留一条静默更慢的路径。
        if self.misses > 16 + 2 * self.hits:
            self._retire(f"命中 {self.hits} 次 / 未命中 {self.misses} 次")
            return None
        self._store(x, out)
        return self._piece(out, slot, x.shape[:-1])


#: ``COFFIN_W4_GROUP`` 认识的档位名，可以逗号组合。
_TIER_NAMES = frozenset({"qsa", "gdn", "shared", "gatedresidual"})

#: ``all``（默认）实际启用的档位。
#:
#: ``gatedresidual`` 故意不在里面：qwen4exp 的 decode 走 ``hyper_*`` 融合核，
#: ``GatedResidual`` 的两个读写门投影一步被调用 **0 次**，而拼接权重要常驻约
#: 0.9 GB ——B=1 时无所谓，但它会直接压低下一轮的 batch 上限（四卡 EP4 每卡 20 GB
#: / 31.72 GiB）。保留这段分组逻辑是为了 ``hyper_*`` 一旦回退时的免费加速，
#: 所以要它得显式点名 ``COFFIN_W4_GROUP=gatedresidual``。
_DEFAULT_TIERS = frozenset({"qsa", "gdn", "shared"})


def _group_kind(module: nn.Module, names: list[str]) -> str:
    """把一个组归到档位名上，供 ``COFFIN_W4_GROUP`` 逐类开关。

    判据是**成员的属性名**，不是父模块的类名。类名在这里不可靠：QSA 的宿主叫
    ``Attention``、GDN 的叫 ``GatedDeltaNet``、而 shared expert 的宿主是一个通用的
    ``MLP``——按类名分类时第三种会被判成"未知"而静默不融合（本仓库已经因为"按类名
    猜"踩过一次）。属性名则是模型自己的契约，``state_dict`` 里也是它们。

    Args:
        module: 承载组内成员的父模块，只用于兜底命名。
        names: 组内成员在这个父模块下的属性名。

    Returns:
        ``qsa`` / ``gdn`` / ``shared`` / ``gatedresidual``，或未归类时父模块的小写
        类名（那种组只能靠显式点名启用，不会跟着 ``all`` 打开）。
    """
    joined = " ".join(names)
    if "block_inject" in joined or "input_mix" in joined:
        return "gatedresidual"
    if "in_proj" in joined:
        return "gdn"
    if "q_proj" in joined or "index_qk" in joined:
        return "qsa"
    if "gate_proj" in joined or "up_proj" in joined or "down_proj" in joined:
        return "shared"
    return type(module).__name__.lower()


def _enabled_kinds() -> set[str]:
    """读 ``COFFIN_W4_GROUP``，返回要启用的档位集合。

    为什么要有档位而不是一个布尔：拼接权重是**常驻**的，三档合计约 1.09 GB
    （``gdn`` 那组最宽，803 MB）。decode B=1 时三档都值得开；大 batch 下显存吃紧，
    最先该关的就是 ``gdn``，而那时只改一个环境变量比改代码快。

    Returns:
        空集合表示全关；否则是要启用的档位。

    Raises:
        ValueError: 出现不认识的档位名。静默当成「全关」会让一次 A/B 看起来像两边
            一样快，而实际两边跑的是同一条路。
    """
    raw = os.environ.get("COFFIN_W4_GROUP", "").strip().lower()
    if not raw or raw in {"1", "on", "yes", "all"}:
        return set(_DEFAULT_TIERS)
    if raw in {"0", "off", "no"}:
        return set()
    kinds = {part.strip() for part in raw.split(",") if part.strip()}
    unknown = sorted(kinds - _TIER_NAMES)
    if unknown:
        raise ValueError(
            f"COFFIN_W4_GROUP={raw!r} 里有不认识的档位 {unknown}；"
            f"可用的是 all / 0 / {sorted(_TIER_NAMES)} 的任意组合")
    return kinds


def _plan_fused_groups(root: nn.Module) -> int:
    """给「同一父模块下吃同一份激活的投影」登记融合组，返回建成的组数。

    分组只能在模块树上做，因为真正的调用点在 ``qsa.py`` / ``gdn.py`` / ``moe.py`` 里，
    而这三个文件不在本次改动范围内；但同激活投影在代码里都是**同一个父模块的相邻
    兄弟调用**，于是「父模块 + 相同的 ``in_features``」就是分组键：

    * ``Attention``（QSA）下的 ``{q_proj, k_proj, v_proj}``；
    * ``GatedDeltaNet``（GDN）下的 ``{in_proj_qkv, in_proj_z, in_proj_b, in_proj_a}``
      ——后两个在真机产物里没有被量化，第一次前向时会被摘到只剩两个，见
      :meth:`_FusedGroup._reconcile`；
    * ``MLP``（``SparseMoeBlock.shared_expert``，宽 ``512``）下的
      ``{gate_proj, up_proj}``（``down_proj`` 的 ``in_features`` 是中间维，自然被排除）。

    **只认直接子模块。** 往上找「最小公共祖先」还能多收一个
    ``indexer.index_qk_proj``（它也吃同一份激活），但那会顺带把跨层的同名投影放进
    同一个桶里——它们吃的是不同的激活，代价是每组每次求值都白算一遍整组。

    Args:
        root: 已构造完成的模型。

    Returns:
        建成的组数，供装载日志核对。
    """
    kinds = _enabled_kinds()
    if not kinds:
        return 0
    groups = 0
    for module in root.modules():
        children = [(name, child) for name, child in module.named_children()
                    if isinstance(child, Proj)]
        if len(children) < 2:
            continue
        # 整个父模块先判一次档位：同一父模块下的组必然同类，省掉每个桶一次判断。
        if _group_kind(module, [name for name, _ in children]) not in kinds:
            continue
        buckets: dict[int, list[Proj]] = {}
        for _name, proj in children:
            buckets.setdefault(proj.in_features, []).append(proj)
        for members in buckets.values():
            if len(members) < 2 or any(proj.bias is not None for proj in members):
                continue
            group = _FusedGroup(members)
            for slot, proj in enumerate(members):
                proj._group = group
                proj._group_slot = slot
            groups += 1
    return groups


def fusion_stats() -> dict:
    """报告同激活投影融合组的累计状态，供探针核对「组真的被用上了」。

    没有这个出口的话，「组没被命中」和「组命中了但没有收益」在账本上长得一模一样
    ——两者都只表现为「发射次数没降」。

    Returns:
        ``{"groups": 组数, "active": 仍启用的组数, "uncovered": 因产物没覆盖成员而停用
        的组数, "hits": 命中, "misses": 未命中, "widths": 每组的 N_total 列表}``。
        ``uncovered`` 与「命中率不足」是两件事：前者说明 artifact 没量化，后者说明
        分组分错了，处理方式完全不同。
    """
    groups = []
    seen: set[int] = set()
    for module in list(_ALL_PROJ):
        group = getattr(module, "_group", None)
        if group is None or id(group) in seen:
            continue
        seen.add(id(group))
        groups.append(group)
    return {
        "groups": len(groups),
        "active": sum(1 for g in groups if not g.disabled),
        "uncovered": _UNCOVERED_GROUPS,
        "hits": sum(g.hits for g in groups),
        "misses": sum(g.misses for g in groups),
        "widths": [g.offsets[-1][0] + g.offsets[-1][1] for g in groups
                   if g.offsets],
    }


#: 所有建过的 :class:`Proj`，只给 :func:`fusion_stats` 用。弱集合的理由与
#: ``pack`` 的身份缓存相同：诊断不能把一份已经释放的模型钉在显存里。
_ALL_PROJ: "weakref.WeakSet" = weakref.WeakSet()


class Proj(nn.Linear):
    """稠密投影 ``[..., K] -> [..., N]``，按激活行数挑后端的语义入口。

    继承 ``nn.Linear`` 只为拿到参数创建、初始化、dtype/设备搬移和 ``state_dict``
    命名——checkpoint 里的键名是 ``...weight``，``weights.load_into`` 与 oracle
    fixture 都按它逐条对账，换一套命名等于把所有装载路径重写一遍。前向整个换掉。

    Attributes:
        owns_quantized_weight: 权重由量化产物就地提供时为真。装载器据此把该键从计划
            里摘掉，见 ``weights.load_into``；bf16 路径恒为假，那些键照旧逐个校验。
    """

    #: 类默认值让装载器不必先判断类型，见 :attr:`owns_quantized_weight`。
    owns_quantized_weight = False

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 *, quantized: bool = False) -> None:
        """建投影；``quantized`` 表示权重稍后由 :meth:`set_quantized` 挂上。

        Args:
            in_features: 输入宽度 ``K``。
            out_features: 输出宽度 ``N``。
            bias: 是否带 bias。本族真实配置是 ``attention_bias=False``，只有 QSA 的
                四个投影按参考实现读这个开关。
            quantized: 建完就把 bf16 占位权重摘掉，改为等待量化产物。这样
                ``to_empty`` 不会为一份马上要丢的矩阵分配显存。

        Raises:
            ValueError: 宽度非正，或 ``quantized`` 与 ``bias`` 同时成立。后者不提供：
                ``W4A8Weight`` 里没有 bias，挂上去之后这个 bias 会停在 ``meta`` 上，
                而报错位置会离这里很远。
        """
        if in_features <= 0 or out_features <= 0:
            raise ValueError(
                f"projection widths must be positive, got {in_features} -> {out_features}.")
        if quantized and bias:
            raise ValueError(
                "quantized projections carry no bias: W4A8Weight holds only the packed "
                "matrix, so a bias parameter would stay on the meta device")
        if quantized:
            # 占位参数只活这一次构造：建在 meta 上，省下一次「马上要丢的 [N, K] 分配」。
            with torch.device("meta"):
                super().__init__(in_features, out_features, bias=bias)
            self._quantized: W4A8Weight | None = None
            self.release_placeholder()
        else:
            super().__init__(in_features, out_features, bias=bias)
            self._quantized: W4A8Weight | None = None
        #: 所属的同激活融合组与自己在组内的槽位；``None`` / ``-1`` 表示不参与。
        #: 由 :func:`_plan_fused_groups` 在模型构造末尾填，此时权重还没挂上，
        #: 所以组内成员要到第一次前向才收敛（见 :meth:`_FusedGroup._reconcile`）。
        self._group: _FusedGroup | None = None
        self._group_slot = -1
        _ALL_PROJ.add(self)

    @property
    def _source(self):
        """交给后端算子的权重对象。

        Returns:
            :class:`W4A8Weight` 或 bf16 ``nn.Parameter``；两者都不在时抛错，而不是
            返回 ``None`` 让某个 ``F.linear`` 用一个莫名其妙的 ``NoneType`` 报错。

        Raises:
            RuntimeError: 占位权重已摘掉，且既没有挂量化权重也没有用
                :meth:`set_bf16` 补回 bf16 权重。
        """
        if self._quantized is not None:
            return self._quantized
        if self.weight is None:
            raise RuntimeError(
                f"{self.out_features}x{self.in_features} projection has no weight: its "
                f"bf16 placeholder was released for the quantized path, but neither "
                f"set_quantized() nor set_bf16() was called")
        return self.weight

    def release_placeholder(self) -> None:
        """把 bf16 占位权重从 ``state_dict`` 里摘掉，标记为由量化产物提供。

        为什么不是「保留一份同形状的 meta 参数、量化之后再置 None」：``to_empty`` 是
        ``empty_like`` 每个 meta 参数，它不看这个参数之后会不会被删，于是 48 层的稠密
        投影会在装载前先实化 9 GiB 再扔掉。专家栈已经在同一个坑里 OOM 过一次
        （``moe.Experts``），所以键必须从计划里消失，见 ``weights.load_into``。
        """
        if getattr(self, "weight", None) is not None:
            del self.weight
        # 补成 None 让任何误用立刻报错，而不是悄悄拿回一份旧的 bf16 张量。
        self.weight = None
        self.owns_quantized_weight = True

    def set_quantized(self, weight: W4A8Weight) -> None:
        """把量化权重就地挂上，前向从此走后端的 W4A8 分派。

        Args:
            weight: :class:`W4A8Weight`，逻辑形状 ``[N, K]``。它必须已经在目标设备上：
                ``W4A8Weight`` 不是 ``nn.Module`` 也不是 buffer，``model.to(device)``
                搬不动它。

        Raises:
            TypeError: 传进来的不是 :class:`W4A8Weight`。误传一个普通张量会让下面
                ``owns_quantized_weight`` 的记账与实际权重不符。
            ValueError: 形状与构造时声明的不一致。混用两个 checkpoint 的投影权重会让
                每一层用上别人的矩阵，而形状往往仍然合法。
        """
        if not isinstance(weight, W4A8Weight):
            raise TypeError(f"set_quantized expects W4A8Weight, got {type(weight).__name__}")
        if (weight.n, weight.k) != (self.out_features, self.in_features):
            raise ValueError(
                f"quantized weight is {weight.n}x{weight.k}, this projection is "
                f"{self.out_features}x{self.in_features}")
        self._quantized = weight
        if getattr(self, "weight", None) is not None:
            self.release_placeholder()

    def set_bf16(self, weight: torch.Tensor) -> None:
        """把一份普通 bf16 权重装回已经释放过占位的投影。

        量化产物并不覆盖每一个投影：真机 artifact（作业 4492）的 526 条 ``q4`` 只命中
        305 个稠密投影，``in_proj_a/b``（N=48）与 ``block_inject``（N=4）等 169 个窄
        投影不在里面，因为量化工具跳过了它们。它们不能因此退回 ``F.linear``——那正是
        一步 245.2 ms 的来源——而应当留着 bf16 权重继续走 :class:`Proj` 的同一条路：
        ``bops.linear`` 对裸 bf16 张量内部做 ``pack.as_kn`` 并按对象身份记忆化，实测
        245.2 -> 21.2 ms，无损。量化只再省 8.7 ms，所以「产物没覆盖就用 bf16」几乎不
        损失什么，却省掉一轮重新量化。

        为什么需要「补」而不是「一开始就留着」：量化的键集合只有在读过
        ``quant_index.json`` 之后才知道，而模型构造发生在读索引之前；占位若一直留着，
        ``to_empty`` 就会为那 305 个马上要被替换的投影先实化一份 bf16 矩阵（48 层是
        9 GiB 量级）。所以构造时一律释放，加载器再把没被覆盖的补回来。

        Args:
            weight: ``[N, K]`` 权重。dtype 与设备由调用方决定（装载器知道模型驻留在
                哪张卡上），这里只校验形状、不做隐式转换：``bf16`` 与 ``fp32`` 混用会
                在第一次 ``F.linear`` 报一次与这里毫无关系的 dtype 错误。

        Raises:
            ValueError: 形状与构造时声明的不一致。
            RuntimeError: 已经挂着量化权重。一条投影只有一个权重来源，两边都写等于
                让哪一边生效取决于调用顺序。
        """
        if self._quantized is not None:
            raise RuntimeError(
                "projection already carries a quantized weight; set_bf16 would make "
                "the effective weight depend on call order")
        if tuple(weight.shape) != (self.out_features, self.in_features):
            raise ValueError(
                f"bf16 weight is {tuple(weight.shape)}, this projection is "
                f"({self.out_features}, {self.in_features})")
        self.weight = nn.Parameter(weight.detach(), requires_grad=False)
        # 权重来源从「量化供货」改回「自己持有」：计划核对与下一次装载都按这个标志走。
        self.owns_quantized_weight = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """投影，最后一维 ``K`` -> ``N``，前导维任意。

        前导维先摊平再还原，是因为后端的三个入口都只认 ``[K]`` / ``[M, K]``，而
        ``models/qwen4exp`` 里既有 ``[B, S, H]`` 也有 ``[B, S, 1]``。摊平是视图：
        只有最后一维 stride 不是 1 时才拷一份。

        单行前向且本投影属于一个融合组（:func:`_plan_fused_groups`）时先试
        :meth:`_FusedGroup.try_forward`：同组第一个成员把整组算出来，其余成员从
        同一次结果里切自己的列段。这不是透明的小优化——它把「同一份激活量化 N 次、
        发 N 次 GEMV」降成一次，见 :class:`_FusedGroup` 的账本。

        Args:
            x: 激活，``[..., K]``。

        Returns:
            与入参同前导形状、最后一维为 ``N`` 的张量，dtype 同 ``x``。

        Raises:
            ValueError: ``x`` 的最后一维不是 ``K``，或 ``x`` 是 0 维标量。这两种在
                摊平之后都会变成一个形状合法但与权重不匹配的矩阵乘，报错点会离调用
                点很远。
        """
        if x.dim() == 0 or x.shape[-1] != self.in_features:
            raise ValueError(
                f"projection expects [..., {self.in_features}], got {tuple(x.shape)}")
        lead = x.shape[:-1]
        # 这个序号只喂给融合组的信箱窗口，别处不读它。**每次前向都推进**，包括不
        # 参与融合的投影：窗口要能区分「同一组在一次求值内的相邻调用」与「同一个组
        # 在相邻两层里的两次求值」，而后者之间隔着的正是那些不融合的投影调用。
        global _SERIAL
        _SERIAL += 1
        group = self._group
        # decode 才有融合价值：前导维多于一个元素时，下游 ``.view(...)`` 会要求
        # 连续性，而融合结果是宽的拼接缓冲，切片不再能 ``view`` 成原形状。
        if group is not None and lead.numel() == 1:
            fused = group.try_forward(self, x)
            if fused is not None:
                return fused
        out = dispatch_linear(self._source, x, self.out_features)
        if self.bias is not None:
            # bias 加在收敛后的激活 dtype 上，而 ``nn.Linear`` 把它折进 GEMM 的 epilogue
            # 在 fp32 里加，两者相差一次舍入。本族真实配置无 bias，这条分支只为让
            # ``attention_bias=True`` 的配置仍能跑。
            out = out + self.bias
        return out.reshape(*lead, self.out_features)


class QuantizedEmbedding(nn.Embedding):
    """主词表嵌入；权重可以是 bf16 表，也可以是由量化产物提供的 W4 行。

    继承 ``nn.Embedding`` 的理由与 :class:`Proj` 相同：``model.embed_tokens.weight``
    是 checkpoint 的键名，查表语义（含 ``padding_idx`` 只是普通 gather）也照旧。

    这里每查一个 token 就解包一次：``W4A8Weight.rows`` 是 torch 实现，会为每行重建
    分组下标并做一次 fp32 乘法。它慢，但它正确，而且是**唯一**不退回 bf16 查表的
    写法。真正的修法是把解包融进下一层投影的 prologue（嵌入永远紧接着一次投影），
    那是单独一步，不该混进这次布局切换里。

    Attributes:
        owns_quantized_weight: 同 :attr:`Proj.owns_quantized_weight`。
    """

    #: 类默认值让装载器不必先判断类型。
    owns_quantized_weight = False

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 padding_idx: int | None = None, *,
                 quantized: bool = False) -> None:
        """建嵌入表；``quantized`` 表示表稍后由 :meth:`set_quantized` 挂上。

        Args:
            num_embeddings: 词表行数。
            embedding_dim: 每行宽度。
            padding_idx: padding 行下标；查表语义与 ``nn.Embedding`` 一致。
            quantized: 建完就把 bf16 表摘掉，改为等待量化产物。

        Raises:
            ValueError: 行数或宽度非正。
        """
        if num_embeddings <= 0 or embedding_dim <= 0:
            raise ValueError(
                f"embedding sizes must be positive, got {num_embeddings}x{embedding_dim}.")
        if quantized:
            with torch.device("meta"):
                super().__init__(num_embeddings, embedding_dim, padding_idx)
            self._quantized: W4A8Weight | None = None
            self.release_placeholder()
        else:
            super().__init__(num_embeddings, embedding_dim, padding_idx)
            self._quantized: W4A8Weight | None = None

    def release_placeholder(self) -> None:
        """把 bf16 表从 ``state_dict`` 里摘掉，标记为由量化产物提供。"""
        if getattr(self, "weight", None) is not None:
            del self.weight
        self.weight = None
        self.owns_quantized_weight = True

    def set_quantized(self, weight: W4A8Weight) -> None:
        """把量化嵌入表就地挂上。

        Args:
            weight: :class:`W4A8Weight`，逻辑形状 ``[vocab, hidden]``，已在目标设备上。

        Raises:
            TypeError: 传进来的不是 :class:`W4A8Weight`。
            ValueError: 行数或宽度与构造时声明的不一致。行数错位会让查到的每一行都是
                别的 token，而形状完全合法。
        """
        if not isinstance(weight, W4A8Weight):
            raise TypeError(f"set_quantized expects W4A8Weight, got {type(weight).__name__}")
        if (weight.n, weight.k) != (self.num_embeddings, self.embedding_dim):
            raise ValueError(
                f"quantized embedding is {weight.n}x{weight.k}, this table is "
                f"{self.num_embeddings}x{self.embedding_dim}")
        self._quantized = weight
        if getattr(self, "weight", None) is not None:
            self.release_placeholder()

    def set_bf16(self, weight: torch.Tensor) -> None:
        """把一份普通 bf16 嵌入表装回已经释放过占位的模块。

        存在的理由与 :meth:`Proj.set_bf16` 相同：量化产物不保证覆盖每一个嵌入表，
        而没被覆盖的那个不能因此没有权重。

        Args:
            weight: ``[num_embeddings, embedding_dim]``，dtype 由调用方决定。

        Raises:
            ValueError: 形状与构造时声明的不一致。
            RuntimeError: 已经挂着量化权重。
        """
        if self._quantized is not None:
            raise RuntimeError(
                "embedding already carries a quantized weight; set_bf16 would make "
                "the effective weight depend on call order")
        if tuple(weight.shape) != (self.num_embeddings, self.embedding_dim):
            raise ValueError(
                f"bf16 embedding is {tuple(weight.shape)}, this table is "
                f"({self.num_embeddings}, {self.embedding_dim})")
        self.weight = nn.Parameter(weight.detach(), requires_grad=False)
        self.owns_quantized_weight = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """按 token id 取行，``[...] -> [..., embedding_dim]``。

        Args:
            input: token id，整型，形状任意。

        Returns:
            ``[..., embedding_dim]``。量化路径返回后端声明的 ``act_dtype``。

        Raises:
            RuntimeError: 占位表已摘掉但 :meth:`set_quantized` 还没调用。
        """
        if self._quantized is None:
            if self.weight is None:
                raise RuntimeError(
                    "embedding has no weight: its bf16 table was released for the "
                    "quantized path, but set_quantized() was never called")
            return super().forward(input)
        ids = input.reshape(-1)
        rows = self._quantized.rows(ids).to(_ACT_DTYPE)
        return rows.reshape(*input.shape, self.embedding_dim)


def proj_from_config(config, in_features: int, out_features: int,
                     bias: bool | None = None, quantized: bool = False) -> Proj:
    """按配置造一个稠密投影。

    存在的理由是把「哪些站点带 bias」这条 checkpoint 事实收在一处：QSA 的四个投影在
    参考实现里读 ``attention_bias``，GDN / PLE / 门控的投影则一律没有 bias，而这两类
    在调用点上长得完全一样。

    Args:
        config: :class:`~models.qwen4exp.config.TextConfig`。
        in_features: 输入宽度 ``K``。
        out_features: 输出宽度 ``N``。
        bias: 是否带 bias。``None`` 表示按 ``config.attention_bias`` 决定，即 QSA 的
            站点；其余站点显式传 ``False``。
        quantized: 传给 :class:`Proj`；表示权重稍后由量化产物挂上。

    Returns:
        建好的 :class:`Proj`。

    Raises:
        ValueError: 宽度非正，或 ``quantized`` 与 ``bias`` 同时成立；见
            :class:`Proj`。
    """
    if bias is None:
        bias = bool(config.attention_bias)
    return Proj(in_features, out_features, bias=bias, quantized=quantized)


def release_quantized_placeholders(root: nn.Module) -> int:
    """把一棵子树上所有稠密投影的 bf16 占位权重摘掉。

    量化路径的构造入口只有 ``model.TextModel`` / ``model.CausalLM`` 两个，而投影散在
    ``gdn`` / ``ngram`` / ``gated_residual`` 里、由 ``blocks.DecoderLayer`` 建。与其让
    ``quantized`` 这个开关穿过每一层构造函数（``blocks.py`` 并不在本次改动范围内），
    不如在模型建好之后走一遍：名字与 ``state_dict`` 的对应关系才是装载器关心的东西，
    而它在构造完成时已经定型。

    Args:
        root: 已构造完成的模型。

    Returns:
        摘掉占位参数的模块个数，供装载日志核对「量化到底接上了几个投影」。

    Raises:
        ValueError: 某个模块已经挂着 :class:`W4A8Weight` 却被要求摘占位参数。那说明
            调用顺序反了；静默继续会把刚挂上的权重和占位参数一起丢掉。
    """
    released = 0
    for module in root.modules():
        if not isinstance(module, (Proj, QuantizedEmbedding)):
            continue
        if getattr(module, "_quantized", None) is not None:
            raise ValueError(
                "release_quantized_placeholders ran after set_quantized; attach the "
                "quantized weights only to a model built with the placeholder released")
        if getattr(module, "weight", None) is not None:
            module.release_placeholder()
            released += 1
        else:
            module.owns_quantized_weight = True
    # 占位摘完正好是分组的时机：``Proj`` 全都在树上了，而权重还没有一个挂上来，
    # 所以分组只看结构（父模块 + ``in_features``），不看谁被量化产物覆盖——那是
    # 第一次前向才收敛的事，见 ``_FusedGroup._reconcile``。
    _plan_fused_groups(root)
    return released


__all__ = [
    "Proj",
    "fusion_stats",
    "QuantizedEmbedding",
    "dispatch_linear",
    "proj_from_config",
    "release_quantized_placeholders",
]
