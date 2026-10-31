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
      只换 dtype，绝不换布局。
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
