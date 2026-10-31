"""解码期的权重读取：用少数几行激活去乘 ``[N, K]`` 投影。

这里原本是一个内核选择器：六种 (rows-per-warp, K-unroll) 候选、按权重形状做
设备计时扫描，以及同一个矩阵乘的四种纯 torch 回退写法。那些都是某一张卡的内核
事实，所以现在搬进后端，藏在三个语义入口后面——``linear`` / ``linear_m`` /
``linear_prefill``——本文件只判断*一次调用处在哪个区间*。:class:`GEMV` 保留
名字和它逐实例持有的权重，因为约 200 个调用点在加载时各构造一个；
:class:`GEMVGroup` 仍然把共用同一输入的投影拼起来，因为不管什么平台，一次宽
权重读取都胜过几次窄的。

Module: ``coffin/ops``。实际内核依赖 ``backends.ops``，量化存储形式来自
``int8_weights``；``models/qwen35`` 里每个投影都经过它。

Constraints:
    * 权重保持行主序 ``[N, K]``——checkpoint 发出来的布局，也是契约接受的布局。
      想要 ``[K, N]`` 的后端在加载时于 ``caps.weight_layout`` 背后转置；在这里
      转置，会对所有不转置的后端产生看着合理、实际错误的数字。
    * 本文件不许调用计时原语、读设备 API，也不许按「内核是否存在」分支。某个
      形状是否拿到自动调优的内核由 ``caps.autotune`` 在后端内部决定；算子表是
      完备的，调用点永远不需要问。
    * int8 权重只量化权重：激活仍是 ``D``，每行按自己的 scale 缩放。对
      :class:`int8_weights.QWeight` 的分派发生在后端内部，所以调用方加载器产出
      什么存储就传什么。
    * ``bench=False`` 必须能在没有 GPU 的情况下用：结构测试在没有 checkpoint、
      没有设备的容器里构造这些对象。
"""

from __future__ import annotations

import torch

from backends import ops as bops
from int8_weights import QWeight


class GEMV:
    """按解码期向量乘矩阵的方式求值的 ``[N, K]`` 权重。

    ``w`` 要么是 bf16/fp16 张量，要么是 :class:`int8_weights.QWeight`
    （只量化权重的 int8）。两种形式都原样交给后端：量化格式是模型侧的存储，
    不是平台能力。
    """

    def __init__(self, w, bench: bool = True) -> None:
        """保存投影权重并记下它的形状。

        Args:
            w: 投影权重，行主序 ``[N, K]``，普通张量或
                :class:`QWeight`.
            bench: 为调用点兼容性而留着，除此之外不用。逐形状的内核扫描搬进了
                后端，在那里受 ``caps.autotune`` 控制并按形状缓存；模型侧的标志
                根本传不到那里。这个参数会一直留到所有调用方（``_proj``、
                ``TiedEmbedding``、各模型构造函数）都丢掉它为止，而到那时删它
                纯粹是签名改动。
        """
        self.q = w if isinstance(w, QWeight) else None
        self.N, self.K = (self.q.n, self.q.k) if self.q is not None else w.shape
        #: 权重，按算子表接受的形式给：普通张量，或者带着 packed 缓冲区和 scale 的
        #: ``QWeight``。算子表按类型分派，所以这里是两种存储形式重新合一的地方，
        #: 调用点永远看不出差别。
        self.w = w

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """投影一行或少数几行激活，``[K]`` / ``[C, K]`` -> ``[N]`` / ``[C, N]``。

        Args:
            x: 激活，``[K]`` 或 ``[C, K]``，bf16。

        Returns:
            投影后的激活，dtype 与前导形状随输入。
        """
        return bops.linear(self.w, x)

    def linear_m(self, x: torch.Tensor) -> torch.Tensor:
        """小批量的每一行共用一次权重读取，``[M, K]`` -> ``[M, N]``。

        Args:
            x: 激活，``[M, K]``，bf16，``M > 1``.

        Returns:
            ``[M, N]``，dtype 同 ``x``。切块方式和特化的 ``M`` 由后端选，所以
            批量解码和投机校验用这个入口，而不是围绕
            :meth:`__call__` 循环。
        """
        return bops.linear_m(self.w, x)

    def prefill(self, x: torch.Tensor) -> torch.Tensor:
        """投影一个预填充分块，``[C, K]`` -> ``[C, N]``。

        小的 ``C`` 走 :meth:`linear_m`，因为那个区间里所有行共用一遍权重读取
        胜过 GEMM：校验路径以
        ``C <= 8`` 落到这里，权重只重读一次而不是 ``C`` 次。
        ``8`` 是模型 capture 过的最宽校验窗口（``CKPT``），所以这个界限是
        模型事实，不是内核切块参数。超过它之后，
        迄今测过的每个后端上 GEMM 都更快；而 int8 权重会在这次调用里按 caps
        的权重 dtype 实体化——预填充是计算受限的，
        那里的取舍就是这样做。

        Args:
            x: 激活，``[C, K]``，bf16。

        Returns:
            投影后的激活，``[C, N]``，bf16。
        """
        if 2 <= x.shape[0] <= 8:
            return bops.linear_m(self.w, x)
        return bops.linear_prefill(self.w, x)


class GEMVGroup:
    """若干个共享*同一*输入的 ``[N_i, K]`` 投影，融合成一次 GEMV。

    读一个宽权重矩阵胜过读几个窄的：launch 更少，而且大 N 下读取效率更高，于是
    比如 ``in_proj_qkv/z/b/a`` 或 ``gate/up`` 变成一次调用，输出随后再切分。
    这在每个平台上都成立，所以拼接留在这里，不搬进后端。


    """

    def __init__(self, ws: list, bench: bool = True) -> None:
        """Concatenate the weights along the output axis and build the fused GEMV.

        融合矩阵一次分配、就地填充，而不是用 ``torch.cat`` 搭出来。``cat`` 要求输出
        和两个输入同时活着，于是在 27B 的 checkpoint 上，每层的 ``gate``/``up``
        对会临时占三份自己；六十四个这样的临时量让缓存分配器攥着 4.2 GiB
        *已保留但未分配*的段，它们拼不进下一次 170 MiB 的请求，构建最终以 31.7 GiB
        的卡上 27.0 GiB 常驻内存挂掉。拷进预分配好的目标，每层只多一份拷贝，
        源还能立刻释放。

        Args:
            ws: 共享一个输入的投影，各自行主序 ``[N_i, K]``。必须同属一种存储
                类别（全 int8 或全普通），且共享
                K；拼接会用形状错误拦住不合规的组合。
            bench: 透传给 :class:`GEMV`。
        """
        if isinstance(ws[0], QWeight):
            self.sizes = [w.n for w in ws]
            k = ws[0].k
            total = sum(self.sizes)
            wq = torch.empty((total, k), dtype=ws[0].wq.dtype, device=ws[0].wq.device)
            scale = torch.empty(total, dtype=ws[0].scale.dtype, device=ws[0].scale.device)
            row = 0
            for w in ws:
                wq[row:row + w.n].copy_(w.wq)
                scale[row:row + w.n].copy_(w.scale)
                row += w.n
            fused = QWeight(wq, scale)
        else:
            self.sizes = [w.shape[0] for w in ws]
            first = ws[0]
            total = sum(self.sizes)
            buf = torch.empty((total, first.shape[1]), dtype=first.dtype,
                              device=first.device)
            row = 0
            for w in ws:
                buf[row:row + w.shape[0]].copy_(w)
                row += w.shape[0]
            fused = buf
        self.g = GEMV(fused, bench)

    def __call__(self, x: torch.Tensor) -> list[torch.Tensor]:
        """投影共享输入，再把结果按权重切回去。

        Args:
            x: 激活，``[K]`` 或 ``[C, K]``，bf16。

        Returns:
            按给定顺序每个输入权重一个张量，沿输出轴切回各自原来的尺寸。
        """
        return self.g(x).split(self.sizes, dim=0)

    def linear_m(self, x: torch.Tensor) -> list[torch.Tensor]:
        """:meth:`__call__` 的批处理版本，沿输出轴切分。

        Args:
            x: 激活，``[M, K]``，bf16。

        Returns:
            按给定顺序每个输入权重一个 ``[M, N_i]`` 张量。
        """
        return self.g.linear_m(x).split(self.sizes, dim=1)

    def prefill(self, x: torch.Tensor) -> list[torch.Tensor]:
        """:meth:`__call__` 的预填充分块版本，沿输出轴切分。

        Args:
            x: 激活，``[C, K]``，bf16。

        Returns:
            按给定顺序每个输入权重一个 ``[C, N_i]`` 张量。
        """
        return self.g.prefill(x).split(self.sizes, dim=1)


def _proj(sd: dict, prefixes: list[str], bench: bool):
    """为 checkpoint 的一个或多个权重前缀构建投影。

    共用一个辅助函数，单前缀和多前缀就走同一条构造路径；只有一个前缀时跳过
    融合，因为那只会多出一次切分。

    Args:
        sd: 按参数名索引的 checkpoint state dict。
        prefixes: 参数前缀，不含 ``.weight`` 后缀，顺序与调用方
            解包结果一致。
        bench: 透传给 :class:`GEMV`；现在已无用，见它的 ``__init__``。

    Returns:
        单个前缀时是 :class:`GEMV`，否则是 :class:`GEMVGroup`。
    """
    ws = [sd[p + ".weight"] for p in prefixes]
    return GEMV(ws[0], bench) if len(ws) == 1 else GEMVGroup(ws, bench)
