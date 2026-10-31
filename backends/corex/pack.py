"""把 checkpoint 的 ``[N, K]`` 投影转成本卡全带宽流式读取的 ``[K, N]`` 布局。

``Caps.weight_layout == "kn"`` 是后端替加载器兑现的一句承诺：算子契约仍然是
checkpoint 的 ``[N, K]`` 布局，唯一一次转置挪到这里做，而不是每次调用都做。
在 BI-V100 上这个布局对解码值 14-22 倍（``[N, K]`` 实测 19-30 GB/s，
``[K, N]`` 是 362-410 GB/s，COREX_PORT.md §0），所以本文件是后端的性能关键
部分，不是一个整理形状的辅助工具。

Module: ``coffin/backends/corex``。只 import ``torch`` 和叶子模块
``int8_weights``，不 import ``coffin`` 里的任何东西。
``backends/corex/__init__.py`` 对每个投影调用 :func:``as_kn``；希望常驻集
保持单份拷贝的模型加载器在加载时调用一次 :func:``repack_state_dict``。

Constraints:
    * 决定转置怎么写的是峰值显存，而不是速度。用 ``w.t().contiguous()`` 转整个
      state dict 会同时持有两份布局，8.8 GB 的 bf16 权重变成 17.6 GB；
      :func:``repack_state_dict`` 一次转一个张量并立刻丢掉源，所以瞬时峰值是
      *最大单个* 权重的两倍（``lm_head`` 是 1.3 GB），常驻集始终只有一份。
      原地转置不是选项：非方阵没有这种操作，分块写法同样有 src+dst 的峰值。
    * 裸 ``[N, K]`` 张量一律按 checkpoint 布局解读。打包形式用包装类型
      （:class:``KnWeight``）而不是裸 ``[K, N]`` 张量就是为了这个：只看形状
      区分不了两者，方阵权重会不声不响地朝反方向转置。
    * :func:``as_kn`` 按源张量记忆化，这样忽略 ``weight_layout`` 的调用方也是
      每个权重只转置一次，而不是每个解码步一次。缓存只装打包后的拷贝，键是
      对象身份，所以只有新张量才会重新打包。
    * 内核只读一种 dtype，就是 :data:``WEIGHT_DTYPE``。选 bf16 是因为预填充
      GEMM 在本卡上 bf16 和 fp16 都到 32.6 TFLOP/s，checkpoint 自己的 dtype
      加载时不必转换。
"""

from __future__ import annotations

import weakref
from typing import Any

import torch

from int8_weights import QWeight, is_quantizable

#: 打包权重的存储 dtype。必须与本后端的 ``Caps.weight_dtype`` 一致；内核按
#: （权重 dtype，bf16 激活）这一对编译。
WEIGHT_DTYPE = torch.bfloat16

#: 能交给库 GEMM 的最大 ``N``。本卡实测：``N = 131072`` 正确，
#: ``N = 196608`` 和 ``N = 248320``（``lm_head``）返回 NaN。预填充路径按这个
#: 宽度沿 ``N`` 分块；解码路径根本不调库，所以这个上限只约束预填充。它是这张
#: 卡 GEMM 的性质而不是可调参数，因此和布局决定放在一起。
LIB_MAX_N = 131072

#: 身份缓存的上限。一个模型只有几百个投影，所以只有不断把新建张量打包的循环
#: 诊断才会撞到上限；届时清扫会丢掉源已被回收的条目。
_CACHE_LIMIT = 4096

#: ``(id(source), dtype) -> (源的 weakref, 打包后的权重)``。通过 weakref 回调
#: 及时回收，加载器丢掉的权重不会把它的打包副本钉住。
_CACHE: dict[tuple[int, torch.dtype], tuple[weakref.ref, "KnWeight"]] = {}


class KnQWeight:
    """以 int8 ``[K, N]`` 加每列一个 fp32 scale 存储的 ``[N, K]`` 投影。

    这是 :class:``KnWeight`` 的 int8 对应物，存在的理由只有一个：解码受带宽
    限制，把每个权重读的字节数减半是投影上仅剩的杠杆，而投影占了这个步骤的
    大头（``../../../PERF_COREX.md`` §8）。

    scale 按 *输出列* 给，和 checkpoint 的按行 scale 是一回事——``[N, K]`` 的
    行就是 ``[K, N]`` 的列。它存在矩阵旁边而不是预先乘进去：在 ``K`` 求和之后
    每列乘一次既更便宜，也正是量化器的误差模型所假设的。
    """

    __slots__ = ("scale", "wt", "n", "k")

    #: 用来区分本形式与 :class:``KnWeight``，不必每次调用都做类型检查。
    int8 = True

    def __init__(self, wt: torch.Tensor, scale: torch.Tensor) -> None:
        """包装一个已经转置好的 int8 矩阵及其列 scale。

        Args:
            wt: ``[K, N]`` 连续存放的 int8 矩阵。
            scale: ``[N]`` 个 fp32 scale，每个输出列一个。

        Raises:
            ValueError: ``wt`` 不是二维 int8、scale 不是一维 fp32，或两者长度
                对不上。这几种都会拿错元素算出看着合理的数，而不是直接失败。
        """
        if wt.dim() != 2:
            raise ValueError(f"packed weight must be 2-D, got {tuple(wt.shape)}")
        if not wt.is_contiguous():
            raise ValueError("packed weight must be contiguous")
        if wt.dtype != torch.int8:
            raise ValueError(f"packed int8 weight must be int8, got {wt.dtype}")
        if scale.dim() != 1 or scale.dtype != torch.float32:
            raise ValueError("scale must be 1-D fp32")
        self.k, self.n = int(wt.shape[0]), int(wt.shape[1])
        if int(scale.numel()) != self.n:
            raise ValueError(f"scale holds {scale.numel()} values for N = {self.n}")
        self.wt = wt
        self.scale = scale

    def nbytes(self) -> int:
        """常驻字节数：int8 矩阵加它的 fp32 列 scale。

        Returns:
            ``K * N + 4 * N``。
        """
        return self.wt.numel() + self.scale.numel() * 4

    def to(self, dtype: torch.dtype) -> torch.Tensor:
        """按 checkpoint 布局 ``[N, K]`` 返回一个反量化后的普通张量。

        给由 bf16 内核或参考实现承担的路径留的逃生口。这是真反量化，代价是权重的
        完整浮点大小——解码路径上的调用方不许用它。

        Args:
            dtype: 调用方算术所用的 dtype。

        Returns:
            新的 ``[N, K]`` 连续存放张量。
        """
        return (self.wt.float() * self.scale.view(1, -1)).to(dtype).t().contiguous()

    def dequant_kn(self, dtype: torch.dtype = WEIGHT_DTYPE) -> "KnWeight":
        """不转置，只反量化，保持 ``[K, N]`` 布局。

        :meth:``to`` 的 ``[K, N]`` 版本：bf16 内核要的就是它，所以用不了 int8
        路径的调用方用一次乘法就能转换，不必转置。

        写成先 cast 再 scale，不是先 scale 再 cast。直觉写法
        ``(wt.float() * scale).to(dtype)`` 会实体化一个 ``[K, N]`` 的 fp32 临时
        张量，是它马上要被窄化成的结果的两倍大；在这个模型的 128 个投影上，那是
        每个预填充请求 *固定的* 174.8 ms，在 n=128 时是整个预填充的 3.1 倍
        （``../../../PERF_COREX.md`` §13.2b）。

        在 bf16 目标上用 fp32 scale 做 ``mul_`` 以 fp32 计算、只舍入一次，和旧
        写法是同一次舍入，权重保持逐位相同。先窄化 scale
        （``scale.to(dtype)``）是 1.88x 而不是 1.70x，但会把每个权重挪动一个
        bf16 ULP，而一个权重会喂 32 层；为一个算子的 10% 做这种取舍不值得。

        Args:
            dtype: 结果的存储 dtype。

        Returns:
            覆盖反量化矩阵的 :class:``KnWeight``。

        Raises:
            TypeError: ``dtype`` 不是浮点类型。往整数目标里缩放会回绕而不报错，
                那会在下游很远的地方表现为 checkpoint 损坏。
        """
        if not dtype.is_floating_point:
            raise TypeError(f"dequant_kn needs a float dtype, got {dtype}")
        out = self.wt.to(dtype)
        out.mul_(self.scale.view(1, -1))
        return KnWeight(out)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"KnQWeight(K={self.k}, N={self.n}, int8)"


class KnWeight:
    """以内核索引的连续 ``[K, N]`` 矩阵存储的 ``[N, K]`` 投影。

    用包装类型而不是裸张量，是为了让布局不产生歧义：算子契约交出的是
    ``[N, K]``，方阵权重否则会被再转置一次，形状上看不出任何异常。
    """

    __slots__ = ("wt", "n", "k")

    #: 两种打包形式都回答这两项，调用方不必对每个子类做 isinstance 就能问
    #: 「这是不是 int8」和「它的 scale 是多少」。
    int8 = False
    scale = None

    def __init__(self, wt: torch.Tensor) -> None:
        """包装一个已经转置好的矩阵；这里不拷贝任何东西。

        Args:
            wt: ``[K, N]`` 连续矩阵。所有权仍归调用方，传视图进来会让内核读到
                错误的几何形状。

        Raises:
            ValueError: ``wt`` 不是二维或不是连续存放。两者都会拿错元素算出看着
                合理的数，而本类存在的目的就是挡掉这种失败模式。
        """
        if wt.dim() != 2:
            raise ValueError(f"packed weight must be 2-D, got {tuple(wt.shape)}")
        if not wt.is_contiguous():
            raise ValueError("packed weight must be contiguous")
        self.wt = wt
        self.k, self.n = int(wt.shape[0]), int(wt.shape[1])

    def nbytes(self) -> int:
        """常驻字节数，供加载器核算显存。

        Returns:
            ``K * N * element_size``：只有打包后的矩阵，因为它来自的源预期已被
            丢掉。
        """
        return self.wt.numel() * self.wt.element_size()

    def to(self, dtype: torch.dtype) -> torch.Tensor:
        """以 ``dtype`` 按 checkpoint 布局 ``[N, K]`` 返回普通张量。

        这是给唯一一个绝不能看到打包形式的路径留的逃生口：本后端没有内核的算子由
        :mod:``backends.reference`` 承担，它的 ``linear`` 会对拿到的任何东西做
        ``w.to(dtype)``。没有这个方法，回退会在解码步中途抛
        ``AttributeError``，看起来像 checkpoint 损坏，而不是内核未实现。

        转置结果不进缓存：这条路径已经接受了一次拷贝，按权重缓存会把打包器存在
        的意义（单份拷贝）抵消掉。

        Args:
            dtype: 调用方算术所用的 dtype。

        Returns:
            新的 ``[N, K]`` 连续存放张量。
        """
        return self.wt.t().to(dtype).contiguous()

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"KnWeight(K={self.k}, N={self.n}, {self.wt.dtype})"


def is_packed(w: Any) -> bool:
    """报告一个权重是否已经处于本后端的布局。

    Args:
        w: 任意可接受存储形式的投影权重。

    Returns:
        :class:``KnWeight`` 返回 ``True``。裸张量永远是 ``False``：裸形式是契约
        规定的 ``[N, K]``。
    """
    return isinstance(w, (KnWeight, KnQWeight))


def to_kn(w: torch.Tensor | QWeight | KnWeight | KnQWeight,
          dtype: torch.dtype = WEIGHT_DTYPE) -> KnWeight | KnQWeight:
    """把一个投影转置成内核读取的布局。

    Args:
        w: ``[N, K]`` 权重：任意浮点 dtype 的普通张量、一个
            ``int8_weights.QWeight``（先反量化，因为本后端声明
            ``int8_weights=False``），或一个 :class:``KnWeight``（原样返回，
            因此本函数是幂等的）。
        dtype: 打包矩阵的存储 dtype，与 ``caps.weight_dtype`` 一致。

    Returns:
        ``[K, N]`` 权重。除非 ``w`` 已经打包，否则是新分配。

    Raises:
        ValueError: ``w`` 不是二维。一维归一化权重走到这里，说明调用方打包错了
            张量，转置它只会在离原因很远的地方报形状错误。
    """
    if isinstance(w, (KnWeight, KnQWeight)):
        return w
    if isinstance(w, QWeight):
        # 在 checkpoint 布局里已经是 int8，所以便宜的做法是保持 int8、按每元素
        # 一字节转置，而不是先扩成两字节。同一个权重，转置峰值只有 bf16 路径的
        # 一半。
        wq = w.wq.t().contiguous()
        return KnQWeight(wq, w.scale.contiguous())
    if w.dim() != 2:
        raise ValueError(f"projection weight must be 2-D, got {tuple(w.shape)}")
    return KnWeight(w.to(dtype).t().contiguous())


def _evictor(key: tuple[int, torch.dtype]) -> Any:
    """构造丢弃单个缓存条目的 weakref 回调。

    Args:
        key: 该回调负责的缓存键。

    Returns:
        删除该键的回调。闭包只捕获 key 而不捕获权重，避免回调延长权重的生命周期。
    """

    def _drop(_dead: weakref.ref) -> None:
        _CACHE.pop(key, None)

    return _drop


def _sweep() -> None:
    """丢掉源张量已被回收的缓存条目。

    weakref 回调通常在源一死就删掉条目，但被回收器延迟的回调会留下死条目；清扫
    让这个上限成为硬边界而不是建议。
    """
    for key, (ref, _packed) in list(_CACHE.items()):
        if ref() is None:
            _CACHE.pop(key, None)


def as_kn(w: torch.Tensor | QWeight | KnWeight | KnQWeight,
          dtype: torch.dtype = WEIGHT_DTYPE) -> KnWeight | KnQWeight:
    """返回权重的打包形式，按源对象只转置一次。

    算子表对每个投影都调用它。一次转置是毫秒级，解码步只有一毫秒的零头，每次
    调用都做会主导这个步骤；按对象身份记忆化让第一次调用付账，之后每次都是字典
    命中。

    Args:
        w: 契约布局下的投影权重，或已经打包的权重。
        dtype: 打包矩阵的存储 dtype。

    Returns:
        打包后的权重。返回的对象归缓存所有，调用方不得原地改 ``wt``；不能被弱
        引用的权重每次调用都重新打包，正确但慢。

    Raises:
        ValueError: ``w`` 不是二维；见 :func:``to_kn``。
    """
    if isinstance(w, (KnWeight, KnQWeight)):
        return w
    source = w.wq if isinstance(w, QWeight) else w
    key = (id(source), dtype)
    hit = _CACHE.get(key)
    if hit is not None and hit[0]() is source:
        return hit[1]
    packed = to_kn(w, dtype)
    try:
        ref = weakref.ref(source, _evictor(key))
    except TypeError:
        # 没有 ``__weakref__`` 槽位的张量子类。缓存它会被永久钉住，所以这一个
        # 权重每次调用都重新打包。
        return packed
    if len(_CACHE) >= _CACHE_LIMIT:
        _sweep()
    _CACHE[key] = (ref, packed)
    return packed


def repack_state_dict(sd: dict, dtype: torch.dtype = WEIGHT_DTYPE) -> dict[str, int]:
    """就地把已加载 state dict 的每个投影转置，一次一个。

    这是加载器对 ``weight_layout="kn"`` 的入口：每份打包副本就位时立刻丢掉对应
    的源，让常驻集只有一份权重；它沿用和 int8 量化器一样的二维判定，所以碰到的
    张量正好是投影，永远不是归一化权重、``A_log`` 或 conv 内核。

    Args:
        sd: bf16 张量和/或 ``QWeight`` 对象的 state dict，会被消费并原地改写。
            不是投影的张量保持不动。
        dtype: 打包矩阵的存储 dtype。

    Returns:
        ``{"packed": n, "bytes_before": b, "bytes_after": a}``，供加载器的启动
        日志用。字节数只覆盖被转换的条目，所以差值反映的是转换改了什么，而不是
        模型总量。

    Raises:
        ValueError: 某个投影不是二维；见 :func:``to_kn``。
    """
    packed_count = 0
    bytes_before = 0
    bytes_after = 0
    for key in list(sd):
        w = sd[key]
        if isinstance(w, KnWeight):
            continue
        if not isinstance(w, QWeight) and not (
                isinstance(w, torch.Tensor) and is_quantizable(key, w)):
            continue
        bytes_before += w.nbytes() if isinstance(w, QWeight) else (
            w.numel() * w.element_size())
        packed = to_kn(w, dtype)
        sd[key] = packed
        packed_count += 1
        bytes_after += packed.nbytes()
        del w
    return {"packed": packed_count, "bytes_before": bytes_before,
            "bytes_after": bytes_after}


def unpacked_keys(sd: dict) -> list[str]:
    """报告 state dict 里还有哪些条目处于 checkpoint 布局。

    Args:
        sd: state dict，刚加载的或已重打包的都行。

    Returns:
        尚未打包的投影键，按字典顺序。空表示加载器做完了自己那部分；加载后仍非空
        意味着每次用到该权重都要付一次缓存转置，而不是用常驻的那份。
    """
    return [key for key, w in sd.items()
            if not isinstance(w, KnWeight)
            and (isinstance(w, QWeight)
                 or (isinstance(w, torch.Tensor) and is_quantizable(key, w)))]
