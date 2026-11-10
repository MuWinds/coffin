"""只量化权重的 int8 量化（W8A16），喂给 GEMV 内核的编码器。

Module: ``coffin/int8_weights.py``。导入图的叶子：只 import ``torch``。
``models/qwen35/weights.py`` 在加载 checkpoint 时经它量化，``ops/gemv.py`` 消费
产出的 ``QWeight``，``ops/embed.py`` 和 ``models/qwen35/vision.py`` 认的是同一个
对象。

为什么这么做：解码受带宽限制。在本引擎最初针对的加速器上，投影跑在 540-680 GB/s，
而对着一道约 700 GB/s 的拷贝天花板，占了单步的约 48%；库里的 M=1 GEMV 完全不能用
（55-296 GB/s，而且按形状换内核），Triton 对那颗芯片又没有后端。于是把每个投影要读
的字节数减半成了剩下最大的杠杆，而且代价便宜，因为反量化能藏在访存背后：权重存成
int8、每输出行一个 fp32 scale，内核把这一个 scale 的乘法施加在累加完的点积上，而不是
逐元素施加。

方案：对称、按输出行（按输出通道）、absmax。

    s[n]     = max_k |w[n, k]| / 127
    wq[n, k] = round(w[n, k] / s[n]) clamped to [-127, 127]

行与行独立，所以误差局限在一行之内，而 scale 折进内核的最后一次乘法。激活保持
bf16——连激活一起量化（W8A8）要在关键路径上再加一个量化器，还会开始把误差推进
注意力分数。

``embed_tokens`` 也在其中：它就是绑定的 lm_head，它的矩阵乘是一次解码步里最大的一笔
权重读取。``conv1d.weight`` 保持 bf16（它的内核消费 fp32 抽头，而且只有 32 KB），
所有一维归一化权重同样保留。

Constraints:
    * 权重保持行主序 ``[N, K]``，那是 checkpoint 发出的、HIP 内核索引的布局；这里
      只换 dtype，绝不换布局。唯一一处例外是 :meth:`QWeight.from_kn`：它接受内核
      布局 ``[K, N]``（也就是离线预量化写到盘上的那个形状）并交出 ``[N, K]`` 转置
      视图，目的是让 ``pack.to_kn`` 的那次 ``.contiguous()`` 成为空操作。
    * 存储的是 int8 加每输出行一个 fp32 scale，数学依赖这一配对：不带行 scale 反量化，
      或者按张量缩放，都会重新引入本方案避开的漂移。
    * ``dequant`` 会实体化一份稠密拷贝。它只为预填充 GEMM 路径存在；在解码循环里
      每次调用会把这笔带宽收益吐回去。
    * ``quantize_state_dict`` 就地改写它的字典，并在过程中逐个丢掉 bf16 张量，因为
      加载时的显存峰值是 16 GB 卡上的绑定约束。
"""

from __future__ import annotations

import torch

#: 本地复制一份激活 dtype，而不是 import ``ops.dtypes``：``ops/`` 会 import 本
#: 模块，反向伸手过去就成环了。
D = torch.bfloat16

#: 每份 fp32 临时量覆盖的量化行数。5120 宽的矩阵取 4096 行是 80 MB 临时量，相对
#: 它喂出的 int8 输出很小，行数又足够多到让分块开销消失。这个值只对很宽的矩阵才
#: 重要：27B checkpoint 上不分块的 ``lm_head`` 需要 4.7 GiB 的 fp32 拷贝，加载时在
#: 32 GiB 的卡上就会挂掉。
_QUANT_CHUNK_ROWS = 4096


def is_quantizable(key: str, w: torch.Tensor) -> bool:
    """判断某个 checkpoint 张量是不是本方案可以接手的投影。

    判据是「二维」加上排除 ``conv1d.weight``，不是一张键名单：这个 checkpoint 里
    每个二维矩阵在解码时都按 ``[N, K]`` 权重读取，而卷积的内核消费 fp32 抽头且只有
    32 KB，量化它只会白丢精度、换不来带宽。

    Args:
        key: checkpoint 张量名；后缀决定它是否被排除。
        w: 加载得到的张量，尚未做任何类型转换。

    Returns:
        ``w`` 应当存成 :class:`QWeight` 时为 ``True``。
    """
    return w.dim() == 2 and not key.endswith("conv1d.weight")


class QWeightMoE:
    """一组共享 K 的 int8 专家，``[E, K, N]`` int8 加 ``[E, N]`` fp32 scale。

    布局是 ``kn``（K 轴在外、N 轴连续），与 ``W4A8MoEWeight`` 的 packed 一致，也正好是
    本卡 ``gemv_kn_i8`` 要的形状：``expert(e)`` 交出的 ``QWeight`` 由 ``from_kn`` 构造，
    它内部只取一个转置视图，``pack.to_kn`` 的 ``.contiguous()`` 因此是空操作。

    反过来（存 ``[E, N, K]``）不会报错，但每次 ``bops.linear`` 都要真转置一遍 2-4 MB
    的 int8 权重：``pack.as_kn`` 的缓存键是 ``id(w.wq)``，而每次新建的视图对象 id 都不同，
    缓存必然 miss。实测这一步让每专家一次调用贵到 289 us（4.62 ms/层，decode 224 ms/token）。

    MoE 的专家在 checkpoint 里是逐个二维张量（``experts.{e}.gate_proj.weight``），
    逐专家建 :class:`QWeight` 会让解码一步变成 40 层 x 8 槽 x 2 次小调用，光是主机侧
    启动就够把权重带宽吃回去。融合成一整栈之后，按专家取出来的视图与单独存的那份
    逐位相同（scale 本来就是逐输出行独立的，两种切法算术一致），而调用方只需要一次
    ``index_select`` 就能把命中的专家凑齐。

    Constraints:
        * ``wq`` 必须是 ``[E, K, N]`` 连续 int8，``scale`` 必须是 ``[E, N]`` 连续 fp32。
          三维张量按 ``[N, K]`` 解释时形状校验照样能过，于是错位要到第一次取专家才
          算出来——所以这里逐维校验。
        * 第 ``e`` 个专家的视图是**视图**不是拷贝：``w8a8`` 内核本来就按偏移读这块
          连续内存，拷贝会把省下的带宽还回去。
    """

    __slots__ = ("expert_count", "k", "n", "wq", "scale", "_cache")

    def __init__(self, wq: torch.Tensor, scale: torch.Tensor, *, k: int) -> None:
        """校验形状关系并原地持有两块缓冲。

        Args:
            wq: ``[E, K, N]`` 连续 int8 权重（K 轴在外、N 连续）。
            scale: ``[E, N]`` 连续 fp32，每专家每输出行一个 absmax scale。
            k: 每个专家沿 K 的长度。

        Raises:
            ValueError: 维度、dtype、连续性或形状关系不符。
        """
        if wq.dim() != 3 or wq.dtype != torch.int8 or not wq.is_contiguous():
            raise ValueError("moe int8 weight must be contiguous 3-D int8 [E, K, N]")
        if scale.dim() != 2 or scale.dtype != torch.float32 or not scale.is_contiguous():
            raise ValueError("moe int8 scale must be contiguous 2-D fp32 [E, N]")
        experts, k_dim, n = (int(d) for d in wq.shape)
        if k <= 0 or k_dim != k:
            raise ValueError(f"moe int8 weight holds K={k_dim} but k={k}")
        if tuple(scale.shape) != (experts, n):
            raise ValueError(f"moe int8 scale must be {(experts, n)}, got {tuple(scale.shape)}")
        self.wq = wq
        self.scale = scale
        self.expert_count = experts
        self.n = n
        self.k = int(k)
        #: 逐专家的 ``QWeight`` 缓存。每次新建视图会让 ``pack.as_kn`` 的 ``id()`` 键
        #: 全部失效，转置成本就落在解码路径上；这里让同一个专家永远交同一个对象。
        self._cache = None

    def __len__(self) -> int:
        """专家个数。"""
        return self.expert_count

    def expert(self, index: int) -> "QWeight":
        """把第 ``index`` 个专家取成单投影视图。

        Args:
            index: 专家下标。

        Returns:
            共享同一块存储的 :class:`QWeight`。

        Raises:
            IndexError: 下标越界。
        """
        if not 0 <= index < self.expert_count:
            raise IndexError(f"expert {index} out of range [0, {self.expert_count})")
        cache = self._cache
        if cache is None:
            cache = self._cache = [None] * self.expert_count
        view = cache[index]
        if view is None:
            view = QWeight.from_kn(self.wq[index], self.scale[index])
            cache[index] = view
        return view

    def to(self, dtype: torch.dtype) -> torch.Tensor:
        """解包为 ``[E, N, K]`` 浮点栈，仅供诊断与参考后端。"""
        return torch.stack([self.expert(i).dequant(dtype) for i in range(self.expert_count)])


class QWeight:
    """以 int8 加逐行 fp32 scale 持有的 ``[N, K]`` 权重。

    两个张量总是结伴出现：``ops/gemv.py`` 和 ``ops/embed.py`` 直接读 ``wq`` 和
    ``scale`` 而不是走方法，所以任何替代对象都必须暴露同一对属性。
    """

    __slots__ = ("wq", "scale", "n", "k")

    def __init__(self, wq: torch.Tensor, scale: torch.Tensor) -> None:
        """包住一个已量化好的矩阵；这里不会再量化一次。

        Args:
            wq: ``[N, K]`` 的 int8 值。是公开属性而不是拷贝：调用方必须交出自己
                拥有、连续的存储。
            scale: ``[N]`` 的 fp32 absmax scale，每输出行一个。
        """
        self.wq = wq                      # [N, K] int8，连续
        self.scale = scale                # [N] fp32，连续
        self.n, self.k = wq.shape

    @classmethod
    def from_kn(cls, wt: torch.Tensor, scale: torch.Tensor) -> "QWeight":
        """包住一份已经是 ``[K, N]`` 的 int8 权重，不复制也不转置。

        ``backends/corex/pack.to_kn`` 消费 ``QWeight`` 的方式是 ``w.wq.t()`` 再
        ``.contiguous()``。这里交出的 ``wq`` 是 ``wt`` 的转置视图（strides
        ``(1, N)``），所以那一步取回来正是 ``wt`` 自己，``.contiguous()`` 空转不复制。
        离线预量化到盘上时就写成 ``[K, N]``，省掉的正是这一次转置。

        Args:
            wt: ``[K, N]`` 连续存放的 int8 权重，内核读取的布局。
            scale: ``[N]`` 个 fp32 scale，每个输出列一个。

        Returns:
            ``wq`` 为转置视图的 ``QWeight``。

        Raises:
            ValueError: ``wt`` 不是二维 int8、``scale`` 不是一维 fp32，或两者长度对
                不上。拿错元素算出来的数看着合理，所以在这里拒绝。
        """
        if wt.dim() != 2 or wt.dtype != torch.int8:
            raise ValueError(f"packed int8 weight must be 2-D int8, got "
                             f"{tuple(wt.shape)} {wt.dtype}")
        if scale.dim() != 1 or scale.dtype != torch.float32:
            raise ValueError("scale must be 1-D fp32")
        k, n = int(wt.shape[0]), int(wt.shape[1])
        if int(scale.numel()) != n:
            raise ValueError(f"scale holds {scale.numel()} values for N = {n}")
        if not wt.is_contiguous():
            raise ValueError("packed [K, N] weight must be contiguous")
        obj = cls.__new__(cls)
        obj.wq = wt.t()                   # [N, K] 视图，strides (1, N)
        obj.scale = scale
        obj.n, obj.k = n, k
        return obj

    @classmethod
    def quantize(cls, w: torch.Tensor, chunk_rows: int = _QUANT_CHUNK_ROWS) -> "QWeight":
        """量化一个稠密的 ``[N, K]`` 权重，按输出行对称。

        absmax scale 在 ``1e-8`` 处被夹住，全零行于是照常做除法而不会产生
        ``inf``/NaN；这样的行量化成零，解码回来也是零。

        行与行独立，所以工作按行分块做。这不是吞吐决策——它是宽权重能被量化的前提：
        27B checkpoint 上 ``lm_head`` 的 fp32 拷贝是 ``248320 * 5120 * 4`` B = 4.7 GiB
        一次性活着，还叠在已经产出的 int8 行之上，模型加载期间直接抛
        ``OutOfMemoryError``。分块把这份临时量限制在 ``chunk_rows * K * 4``，与矩阵
        高度无关。

        Args:
            w: 稠密权重，行主序 ``[N, K]``，任意浮点 dtype。
            chunk_rows: 每份 fp32 临时量覆盖的行数。大到让内核吃饱，小到让临时量
                停在几百 MB。

        Returns:
            int8 行和它们的 absmax scale，两者都做成连续，HIP 内核才能在上面取一个
            扁平指针。
        """
        n = w.shape[0]
        if n <= chunk_rows:
            wf = w.float()
            scale = wf.abs().amax(-1).clamp_min(1e-8) / 127.0
            q = (wf / scale[:, None]).round().clamp_(-127, 127).to(torch.int8)
            return cls(q.contiguous(), scale.contiguous())

        # 目标只分配一次、按块填充：先建 N 个分块张量再拼接，等于每块和整体都要同时活着。
        q = torch.empty(w.shape, dtype=torch.int8, device=w.device)
        scale = torch.empty(n, dtype=torch.float32, device=w.device)
        for start in range(0, n, chunk_rows):
            stop = min(n, start + chunk_rows)
            wf = w[start:stop].float()
            s = wf.abs().amax(-1).clamp_min(1e-8) / 127.0
            q[start:stop] = (wf / s[:, None]).round().clamp_(-127, 127)
            scale[start:stop] = s
        return cls(q, scale)

    def dequant(self, dtype: torch.dtype = D) -> torch.Tensor:
        """把权重实体化成 ``dtype``；只给预填充 GEMM 路径用。

        预填充要的是 fp16：这颗芯片没有 packed bf16 计算，同一个 GEMM 在 fp16 权重
        上快约 7x。

        Args:
            dtype: 输出 dtype；调用方按它要喂的 GEMM 来选。

        Returns:
            一份新的稠密 ``[N, K]`` 张量。它不被缓存，所以在解码循环里调用等于每步
            重建一次权重。
        """
        return (self.wq.float() * self.scale[:, None]).to(dtype)

    def rows(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """按行 gather：给嵌入返回 ``(int8 [T, K], fp32 scale [T])``。

        Args:
            idx: 要 gather 的行下标，顺序随调用方。

        Returns:
            对应的 int8 行和它们的 scale，仍按行带 scale，调用方因此只反量化它查到
            的那些行。
        """
        return self.wq.index_select(0, idx), self.scale.index_select(0, idx)

    def nbytes(self) -> int:
        """常驻字节数，供加载器做显存核算。

        Returns:
            int8 行加上它们的 fp32 行 scale。被它替换掉的稠密权重若用 bf16，光元素
            数就要两倍字节。
        """
        return self.wq.numel() + self.scale.numel() * 4

    # 排除在覆盖率之外：运行时没有任何地方调 repr，为它写测试只是在测 f-string。
    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"QWeight({self.n}x{self.k} int8)"


def quantize_state_dict(sd: dict) -> dict:
    """把 ``sd`` 里每个可量化的二维权重替换成 :class:`QWeight`。

    就地改写 ``sd``，并且一个 bf16 张量量化完就立刻丢掉（常驻拷贝因此是缩小而不是
    翻倍：9 GB 的 bf16 权重变成约 4.5 GB，16 GB 卡上腾出的这块地方正好给更长的上下文
    和分页 KV 池）。

    ``load_weights`` 改为在加载过程中量化、不用这个函数；它留给已经把 bf16 张量放在
    显存里的调用方。

    Args:
        sd: state dict，会被消费并就地改写。

    Returns:
        字节数报告 ``{"bf16_bytes": ..., "int8_bytes": ...}``，供加载器启动日志使用。
        它只统计被转换的张量，所以两者之差就是转换实际释放的显存。
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
    """报告 state dict 里哪些条目是 int8，供加载器打日志。

    Args:
        sd: state dict，刚加载的或已被 :func:`quantize_state_dict` 转换过的均可。

    Returns:
        按字典顺序排列的已量化名字；全稠密的 state dict 返回空。
    """
    return [k for k, v in sd.items() if isinstance(v, QWeight)]
