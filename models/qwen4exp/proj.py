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
"""

from __future__ import annotations

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
    return released


__all__ = [
    "Proj",
    "QuantizedEmbedding",
    "dispatch_linear",
    "proj_from_config",
    "release_quantized_placeholders",
]
