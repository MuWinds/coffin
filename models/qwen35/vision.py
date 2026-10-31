"""Qwen3.5-4B 的视觉塔：把像素变成 LLM token 的 24 层 ViT。

Module: ``coffin/models/qwen35``。模型族的叶子：只 import ``torch``、
``int8_weights`` 和 ``backends``（设备与权重 dtype 都是平台事实），由
``models.qwen35.model`` 用 ``weights.load_weights`` 的张量建起来。它不许 import
``ops``、``engine_api`` 或文本侧的模块，这样 ``tests/test_vision.py`` 才能在没有
checkpoint、没有引擎的情况下建一座小塔。

形状流转（``N = Σ t*h*w`` 个 patch token，``n_img = N / 4`` 个 LLM token）::

    pixel_values [N, 1536]        processor 输出，行按 merge 块顺序排列
      -> patch_embed              F.linear (= Conv3d k=s=(2,16,16))   -> [N, 1024]
      -> + pos_embed resample     48x48 学到的表 -> 每图的 (h,w) 网格
      -> 24 x block               LN -> attn(轴向 rope, 每图) -> LN -> MLP
      -> merger                   LN(1024) -> 4 合 1 视图(4096) -> fc1 -> GELU
                                  (精确) -> fc2                        -> [N/4, 2560]

Constraints:
    * 计算用 fp16，因为这颗芯片有 packed fp16 计算、没有 packed bf16 计算
      （同一个 GEMM 13-20 TFLOP/s 对 2.3-7，文本投影按 ``caps.weight_dtype`` 加载
      也是这个原因），而视觉塔是计算受限的。caps 报告 bf16 权重的芯片，这里也用
      bf16 算：规则是 caps 的值，不是厂商名。
    * fp32 孤岛必须保持 fp32——pos-embed 的抽头求和、旋转角、注意力 softmax。这些
      也是 HF 保留 fp32 的地方，因为 bf16/fp16 的抽头求和会错 1e-2
      （``.tmp_dsh/vision/reference_spec.md`` §14.3）。
    * 除注意力内部之外，token 顺序处处是 merge 块顺序：processor 就是这么发
      ``pixel_values`` 的，pos-embed 的 permute 和 merger 的 4 合 1 视图都依赖它。
      在这里改成行主序，会悄悄产出一张看似合理、实际错误的图。
    * 注意力逐帧、逐图、非因果；各段靠切片隔离，绝不用掩码，因为视觉塔没有掩码
      输入，而 ``[N, N]`` 的加性掩码会变成它最大的张量。
    * 绝不要一次性反量化整个 int8 塔：一次一个 GEMM 把峰值临时量压在一个 fp16 矩阵上
      （merger 的 fc1 是 33 MB）。

在真实 checkpoint 上与 ``transformers`` 4.57.6 的 ``Qwen3VLVisionModel`` 实测对比
（canonical 验收、DCU、每个尺寸一张图）::

    size   engine(fp32)  engine(fp16)  engine(fp16)     time   tower-only
           vs HF(fp32)   vs HF(fp32)   vs HF(bf16)              peak
    256^2      8.7e-6       7.1e-3       6.4e-2         67 ms   ~0.67 GiB
    512^2      1.2e-5       1.3e-2       7.9e-2        129 ms   ~0.71 GiB
    1024^2     1.4e-5       1.4e-2       9.0e-2        756 ms   ~0.85 GiB

第一列是正确性门禁：公式误差是 O(1) 量级，所以 1e-5 比任何错误的 merge 顺序、rope
布局或 LN 位置所能藏身的高度还低一百万倍。第二列是 fp16 的内在代价，它和 HF *自己*
的 fp16 对 fp32 的差异一致（1.4e-2）——那是 dtype 的代价，不是实现的。第三列只是
冒烟列：bf16 每层约舍入 8e-3，而这座塔叠了 24 层、激活到达 1e3 量级，bf16 判据本身
离 fp32 就有 1e-1，管不了 1e-3 级别的门禁。

1024^2 是 756 ms，目标是 450 ms。塔在那里维持 5.7 TFLOP/s（本芯片 M=256 的
packed-fp16 峰值的 44%），而且从 512^2 到 1024^2 速率是平的，所以差距出在注意力
``score -> fp32 softmax -> PV`` 往返里逐算子的效率上，而不是 launch 次数：n=4096 时
注意力占 4.28 TFLOP 前向的 38.5%，需要 6.4 G 次 exp() 调用。
``.tmp_dsh/vision2/budget.py`` 会打印这份预算；可动的杠杆是 fp16 softmax、更大的
``Q_TILE``，以及融合的注意力内核。

int8 模式只用于显存吃紧的情形：塔是计算受限的，所以 ``vision_quant`` 用 0.315 GiB
换下 0.621 GiB 的常驻权重，代价同时落在误差和时间上（本来离 fp32 就 1.4e-2 的塔上
再叠 W8A16——用 ``probe_parity.py --int8`` 实测，别假设它是免费的）。
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F

from backends import caps as _caps
from backends import device as _device
from int8_weights import QWeight

#: 本模块读取的每个张量的 checkpoint 前缀。
VISUAL_PREFIX = "model.visual."

#: 视觉注意力的查询切块宽度。score 切块 ``[heads, Q_TILE, n]`` 是整座塔最大的
#: 激活——1 MP（n = 4096）时不分块的块是 0.5 GiB fp16，softmax 把它抬到 fp32 后
#: 是 1 GiB，所以切块是硬性要求而不是调优旋钮。取 256 时每块临时量是 33 MB fp16
#: 加 67 MB fp32（峰值稳在 0.4 GiB 以内），每图每层 16 块、2 次矩阵乘，24 层合计
#: 约 750 次 launch。
Q_TILE = int(os.environ.get("VISION_Q_TILE", "256"))

#: 施加旋转时的 token 切块宽度，和 HF 一样在 fp32 里做。它把临时量控制在约 4 MB，
#: 而不是 ``N*heads*head_dim*4``（1 MP 时 16 MB）。
_ROPE_TILE = 1024

#: 反量化 int8 权重时每个分块的行数。``QWeight.dequant`` 会为整个矩阵做一份 fp32
#: 拷贝（4096x4096 时 67 MB）；分块把这个临时量压到 8 MB，峰值因此仍落在 fp16
#: 结果（33 MB）上。
_DEQUANT_ROWS = 512


def is_vision_quantizable(key: str, w: torch.Tensor) -> bool:
    """挑出视觉塔的 GEMM 权重，那是 int8 唯一帮得上忙的部分。

    ``pos_embed.weight`` 是按索引读的行 gather 表（不是 GEMM），
    ``patch_embed.proj.weight`` 是 5 维卷积核；两者都保持 fp16。量化它们等于为了
    省 7.8 MB 而在 gather / 第一层路径上塞进一次反量化，所以规则是「二维
    ``.weight``，且不是 gather 表」。

    算个数：24 个块 × 4 个矩阵乘 + merger 的 2 个 = 297 个张量里量化 98 个，占视觉塔
    333.51 M 参数中的 329.25 M。常驻是 0.307 GiB 的 int8 加 0.008 GiB 的 fp16
    一维/归一化/gather 残留，也就是**0.315 GiB 对 fp16 的 0.621 GiB**。

    Args:
        key: 张量在 checkpoint 里的名字，用来排除 gather 表。
        w: 已加载的张量；只看它的秩。

    Returns:
        调用方应把这个权重存成 int8 时为 ``True``。这个决定是策略而非实测：这里漏改
        一个调用点，视觉塔就要多占一倍常驻字节。
    """
    return w.dim() == 2 and key.endswith(".weight") and "pos_embed" not in key


#: 每个块的二维矩阵乘：``attn.qkv``、``attn.proj``、``mlp.linear_fc1``、
#: ``mlp.linear_fc2``。各 bias 和两个 LayerNorm 都是一维；merger 再加
#: ``linear_fc1`` + ``linear_fc2``。24*4 + 2 = 98 个可量化权重。
GEMMS_PER_BLOCK = 4


def _cfg_get(cfg, key, default=None):
    """从原始 ``config.json`` 字典或配置对象里读一个字段。

    Args:
        cfg: 来自 ``config.json`` 的 ``dict``，或带属性的对象；两种形态都会到达本
            模块，取决于谁构建视觉塔。
        key: 字段名。
        default: 字段缺失时的值，对 checkpoint 没写明的每个键来说这都是常态。

    Returns:
        字段值，或者 ``default``。
    """
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class _VisionConfig:
    """``config.json:vision_config``，外加 transformers 会填的默认值。

    checkpoint 里没有视觉塔的 ``rope_parameters``；transformers 4.57.6 会从
    ``Qwen3_5VisionConfig.default_rope_type`` / ``RotaryEmbeddingConfigMixin``
    填入 ``{rope_type: "axial", rope_theta: 10000.0}``。下面的默认值就是那份填充，
    所以省略了某个字段的 checkpoint 落到的仍是 HF 会用的值，而不是一套自创的几何。

    构造时对任何视觉塔表达不了的 checkpoint 几何直接抛出——那些是配置错误，在这里
    失败能点名字段，而不是几层之后冒出一张错误的图。
    """

    def __init__(self, cfg) -> None:
        """把一份 ``vision_config`` 归一化到视觉塔固定的词表里。

        Args:
            cfg: ``config.json:vision_config``，字典或对象。

        Raises:
            ValueError: checkpoint 要求 deepstack 视觉索引、非轴向 rope、不支持的
                激活函数，或 pos-embed 表不是完全平方数。每一种都是本引擎没实现的
                checkpoint；在加载时拒绝，而不是等到解码。
        """
        def g(key, default=None):
            """经 ``_cfg_get`` 读一个配置字段，让调用点保持简短。

            Args:
                key: ``vision_config`` 里的字段名。
                default: checkpoint 省略该字段时的值，也就是 transformers 为该键
                    填的默认值。

            Returns:
                字段值，或者 ``default``。
            """
            return _cfg_get(cfg, key, default)

        self.depth = int(g("depth", 24))
        self.hidden_size = int(g("hidden_size", 1024))
        self.intermediate_size = int(g("intermediate_size", 4096))
        self.num_heads = int(g("num_heads") or g("num_attention_heads") or 16)
        self.in_channels = int(g("in_channels", 3))
        self.patch_size = int(g("patch_size", 16))
        self.spatial_merge_size = int(g("spatial_merge_size", 2))
        self.temporal_patch_size = int(g("temporal_patch_size", 2))
        self.num_position_embeddings = int(g("num_position_embeddings", 2304))
        self.out_hidden_size = int(g("out_hidden_size", 2560))
        self.hidden_act = str(g("hidden_act", "gelu_pytorch_tanh"))
        self.layer_norm_eps = float(g("layer_norm_eps", 1e-6))
        rope = g("rope_parameters") or {}
        self.rope_type = str(rope.get("rope_type", "axial"))
        self.rope_theta = float(rope.get("rope_theta", 10000.0))
        deepstack = g("deepstack_visual_indexes") or []
        if deepstack:
            raise ValueError(
                f"this checkpoint asks for deepstack_visual_indexes={deepstack}; "
                "the engine's tower returns last_hidden/merger output only")
        if self.rope_type != "axial":
            raise ValueError(f"vision rope_type must be axial, got {self.rope_type!r}")
        if self.hidden_act not in ("gelu_pytorch_tanh", "gelu"):
            raise ValueError(f"unsupported vision hidden_act {self.hidden_act!r}")
        self.head_dim = self.hidden_size // self.num_heads

        side = math.isqrt(self.num_position_embeddings)
        if side * side != self.num_position_embeddings:
            # 悄悄取整会把整张 pos-embed 表摆错位置（spec §14.6）。
            raise ValueError(
                f"num_position_embeddings={self.num_position_embeddings} is not a "
                "perfect square; the learned pos-embed grid is unknown")
        self.num_grid_per_side = side
        self.merge_unit = self.spatial_merge_size * self.spatial_merge_size

    @property
    def patch_dim(self) -> int:
        """一行 ``pixel_values`` 的宽度：``C * tps * p * p``（这里是 1536）。

        Returns:
            processor 发出的展平 patch 宽度；``forward`` 在任何 GEMM 跑起来之前
            用它校验输入。
        """
        return self.in_channels * self.temporal_patch_size * self.patch_size ** 2


class _Block:
    """One ViT block's 张量：两个 LayerNorm、一组打包注意力、一个 MLP。"""

    __slots__ = ("norm1_w", "norm1_b", "norm2_w", "norm2_b",
                 "qkv_w", "qkv_b", "proj_w", "proj_b",
                 "fc1_w", "fc1_b", "fc2_w", "fc2_b")


class VisionTower:
    """把 checkpoint 的 ``model.visual.*`` 张量变成一座可运行的 fp16 视觉塔。

    ``sd`` 既可以用 ``model.visual.`` 前缀给键（即
    :func:`models.qwen35.weights.load_weights` 返回的形式），也可以去掉前缀；两种
    都接受。``quant=True`` 时——或者 ``sd`` 本身就持有
    :class:`~int8_weights.QWeight` 矩阵（``vision_quant`` 加载出来的正是这样）——
    GEMM 权重是 int8 加逐行 fp32 scale，一次反量化一个 GEMM。

    ``dtype`` 是 :meth:`forward` 返回的类型（模型反正会把文本侧转成 bf16）。
    在 CUDA 上，视觉塔自身不管 ``dtype`` 说什么都按 fp16 算，理由是模块 docstring
    里的 packed 数学；在 CPU 上（数值测试）它按 ``dtype`` 算，fp32 才能跑。传
    ``compute_dtype=`` 可以显式覆盖这条规则。
    """

    def __init__(self, sd: dict, cfg, dtype: torch.dtype = torch.float16,
                 device=None, quant: bool = False,
                 compute_dtype: torch.dtype | None = None) -> None:
        """把一个 checkpoint 绑到 ``device`` 上并选定计算 dtype。

        Args:
            sd: ``model.visual.*`` 张量，键带或不带 ``model.visual.`` 前缀均可。
                ``vision_quant`` 加载的结果已经持有
                :class:`~int8_weights.QWeight` 矩阵，于是 int8 路径靠「存在」而不是
                「标志」被打开。
            cfg: ``config.json:vision_config``，字典或对象。
            dtype: :meth:`forward` 返回的类型。模型会把它转成 bf16 给文本侧，所以
                这是边界 dtype，不是计算 dtype。
            device: 目标设备，字符串或 ``torch.device``。``None`` 用当前选中后端的
                设备，除 CPU 数值测试之外模型每条路径都这么传。
            quant: 即使 ``sd`` 全是 fp16 也强制 int8 GEMM 权重。要付 1.4e-2 的 fp16
                误差再加 W8A16 的代价，所以只用于显存吃紧的情形。
            compute_dtype: 覆盖默认规则——芯片有 packed fp16 计算时默认 fp16
                （即文本投影已经遵循的 ``caps.weight_dtype``），否则用 ``dtype``，
                fp32 数值测试才能跑。

        Raises:
            KeyError: state dict 缺少前向要读的张量；报错会点出缺失数量和前四个，
                一次不完整的下载一眼就能看出来。
            ValueError: ``cfg`` 描述的视觉塔本引擎没实现（见
                :class:`_VisionConfig`）。
        """
        self.cfg = _VisionConfig(cfg)
        c = self.cfg
        if device is None:
            device = _device
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        packed_fp16 = _caps().weight_dtype == torch.float16
        if compute_dtype is None:
            compute_dtype = (torch.float16
                             if packed_fp16 and dtype not in (torch.float32, torch.float64)
                             else dtype)
        self.compute_dtype = compute_dtype
        self.dtype = dtype
        self.quant = bool(quant) or any(isinstance(v, QWeight) for v in sd.values())
        self.scaling = c.head_dim ** -0.5

        self._sd = {k[len(VISUAL_PREFIX):] if k.startswith(VISUAL_PREFIX) else k: v
                    for k, v in sd.items()}
        self._check_keys()

        # patch embed：checkpoint 的 Conv3d k=s=(2,16,16) 把卷积核 reshape 之后就是
        # 对 1536 个输入的 GEMM，所以这里不跑卷积。
        self.patch_w = self._tensor("patch_embed.proj.weight").reshape(
            c.hidden_size, c.patch_dim)
        self.patch_b = self._tensor("patch_embed.proj.bias")

        # 学到的 pos-embed 表：按索引读的行 gather，永不量化。
        self.pos_embed = self._tensor("pos_embed.weight")

        # 轴向 rope：每轴 head_dim//4 = 16 个频率，底数 10000。保持 fp32，因为角度是
        # fp32 孤岛之一。
        spatial_dim = c.head_dim // 2
        self.inv_freq = 1.0 / (
            c.rope_theta ** (torch.arange(0, spatial_dim, 2, dtype=torch.float32,
                                          device=self.device) / spatial_dim))

        self.blocks: list[_Block] = []
        for i in range(c.depth):
            p = f"blocks.{i}."
            blk = _Block()
            blk.norm1_w = self._tensor(p + "norm1.weight")
            blk.norm1_b = self._tensor(p + "norm1.bias")
            blk.norm2_w = self._tensor(p + "norm2.weight")
            blk.norm2_b = self._tensor(p + "norm2.bias")
            blk.qkv_w = self._tensor(p + "attn.qkv.weight", quantizable=True)
            blk.qkv_b = self._tensor(p + "attn.qkv.bias")
            blk.proj_w = self._tensor(p + "attn.proj.weight", quantizable=True)
            blk.proj_b = self._tensor(p + "attn.proj.bias")
            blk.fc1_w = self._tensor(p + "mlp.linear_fc1.weight", quantizable=True)
            blk.fc1_b = self._tensor(p + "mlp.linear_fc1.bias")
            blk.fc2_w = self._tensor(p + "mlp.linear_fc2.weight", quantizable=True)
            blk.fc2_b = self._tensor(p + "mlp.linear_fc2.bias")
            self.blocks.append(blk)

        # Merger 在 2x2 重排*之前*做 LayerNorm(1024)，而不是之后。
        self.merger_norm_w = self._tensor("merger.norm.weight")
        self.merger_norm_b = self._tensor("merger.norm.bias")
        self.merger_fc1_w = self._tensor("merger.linear_fc1.weight", quantizable=True)
        self.merger_fc1_b = self._tensor("merger.linear_fc1.bias")
        self.merger_fc2_w = self._tensor("merger.linear_fc2.weight", quantizable=True)
        self.merger_fc2_b = self._tensor("merger.linear_fc2.bias")
        self.merge_dim = c.hidden_size * c.merge_unit

        # 以 (h,w) 为键的重采样/rope 表：它们只依赖 patch 网格，和像素无关，所以
        # 服务端每遇到一个新图像尺寸只付一次代价（spec §14.4）。
        self._pos_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self._hw_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self._rope_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        del self._sd

    def _check_keys(self) -> None:
        """拒绝缺少前向所需任何张量的 state dict。

        提前检查，因为替代方案是在第一张图跑到一半、权重已经常驻之后，从
        ``_Block.__slots__`` 的某个字段上炸出 ``AttributeError``。报错会点出缺失
        数量和前四个缺失键，足以分辨是下载不全还是拿错了 checkpoint。

        Raises:
            KeyError: 任何必需的键缺失。重新加载视觉塔；残缺的 state dict 没法在
                运行时打补丁。
        """
        c = self.cfg
        want = {"patch_embed.proj.weight", "patch_embed.proj.bias", "pos_embed.weight",
                "merger.norm.weight", "merger.norm.bias",
                "merger.linear_fc1.weight", "merger.linear_fc1.bias",
                "merger.linear_fc2.weight", "merger.linear_fc2.bias"}
        for i in range(c.depth):
            for suffix in ("norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias",
                           "attn.qkv.weight", "attn.qkv.bias",
                           "attn.proj.weight", "attn.proj.bias",
                           "mlp.linear_fc1.weight", "mlp.linear_fc1.bias",
                           "mlp.linear_fc2.weight", "mlp.linear_fc2.bias"):
                want.add(f"blocks.{i}.{suffix}")
        missing = sorted(want - set(self._sd))
        if missing:
            raise KeyError(
                f"vision state dict is missing {len(missing)} tensors, first: {missing[:4]}")

    def _tensor(self, key: str, quantizable: bool = False):
        """把一个张量按计算 dtype 搬到 ``device``，或者保持 int8。

        一维张量（LayerNorm 权重、bias）保持计算 dtype：它们被 ``F.layer_norm``
        逐元素消费、或者加到一个 fp16 输出上，反量化只会为了总共省 0.3 MB 而每块
        多一次 launch。

        Args:
            key: 已经去掉 ``model.visual.`` 前缀的 state dict 键。
            quantizable: :attr:`quant` 打开时允许 int8。只有 GEMM 权重带这个
                标志；见 :func:`is_vision_quantizable`。

        Returns:
            位于 ``device`` 上的张量，或是计算 dtype，或是 ``_linear`` 每次调用
            反量化的 :class:`~int8_weights.QWeight`。
        """
        t = self._sd[key]
        if isinstance(t, QWeight):
            return t
        t = t.to(device=self.device, dtype=self.compute_dtype).contiguous()
        if quantizable and self.quant:
            q = QWeight.quantize(t)
            del t
            return q
        return t

    def _dequant(self, q: QWeight) -> torch.Tensor:
        """按行分块，把一个 int8 矩阵展开成计算 dtype。

        分块把峰值临时量压到一个分块加 fp16 结果，而不是整份 fp32 拷贝（见
        ``_DEQUANT_ROWS``）；行切片是连续的，多出来的循环不产生 gather。

        Args:
            q: int8 权重，``[n, k]``，带逐行 fp32 scale。

        Returns:
            计算 dtype 的稠密 ``[n, k]`` 矩阵，可直接给 ``F.linear``。归调用方所有；
            下一个 GEMM 时释放。
        """
        if q.n <= _DEQUANT_ROWS:
            return q.dequant(self.compute_dtype)
        out = torch.empty((q.n, q.k), dtype=self.compute_dtype, device=self.device)
        for r0 in range(0, q.n, _DEQUANT_ROWS):
            r1 = min(r0 + _DEQUANT_ROWS, q.n)
            out[r0:r1] = (q.wq[r0:r1].to(torch.float32)
                          * q.scale[r0:r1, None]).to(self.compute_dtype)
        return out

    def _linear(self, x: torch.Tensor, w, b) -> torch.Tensor:
        """``F.linear``，既接受稠密权重，也接受 int8 ``QWeight``。

        Args:
            x: 激活，``[N, in]``。
            w: 稠密权重，或只为这一次调用反量化的
                :class:`~int8_weights.QWeight`。
            b: 计算 dtype 的 bias，或 ``None``。

        Returns:
            ``[N, out]``，dtype 与激活一致。
        """
        if isinstance(w, QWeight):
            w = self._dequant(w)
        return F.linear(x, w, b)

    def _ln(self, x: torch.Tensor, w, b) -> torch.Tensor:
        """用 checkpoint 的 ``layer_norm_eps`` 和 HF 的有偏方差做 LayerNorm。

        Args:
            x: 激活，``[N, hidden]``。
            w: 增益，``[hidden]``。
            b: bias，``[hidden]``。

        Returns:
            ``[N, hidden]``，归一化后新分配的结果。
        """
        return F.layer_norm(x, (x.shape[-1],), w, b, self.cfg.layer_norm_eps)

    def _pos_taps(self, h: int, w: int) -> tuple[torch.Tensor, torch.Tensor]:
        """行主序 patch 顺序下的 ``(h*w, 4)`` gather 下标和 fp32 权重。

        复现 ``Qwen3VLVisionModel.fast_pos_embed_interpolate``：表是 ``side x side``
        的网格，用 ``torch.linspace(0, side-1, size)``（align_corners=True）重采样，
        每根轴一个 2 抽头的帽形核。merge 块的 permute 施加在*结果*上（见
        :meth:`_pos_offsets`），和 HF 的 ``view/permute/flatten`` 做的是同一件事，
        给定尺寸的每张图因此共用同一套抽头表。

        Args:
            h: 一张图的 patch 网格行数。
            w: 一张图的 patch 网格列数。

        Returns:
            行主序网格的 ``(idx, wt)``：``idx`` 里每个 patch 四个表偏移，``wt`` 里
            是对应的 fp32 双线性权重，和为 1。两者都按 ``(h, w)`` 缓存，因为它们
            只依赖网格，与像素无关。
        """
        hit = self._pos_cache.get((h, w))
        if hit is not None:
            return hit
        side = self.cfg.num_grid_per_side
        hs = torch.linspace(0, side - 1, h, device=self.device, dtype=torch.float32)
        ws = torch.linspace(0, side - 1, w, device=self.device, dtype=torch.float32)
        h0 = hs.to(torch.int64)                       # >= 0，所以截断即下取整
        w0 = ws.to(torch.int64)
        h1 = (h0 + 1).clamp(max=side - 1)
        w1 = (w0 + 1).clamp(max=side - 1)
        dh = hs - h0.to(torch.float32)
        dw = ws - w0.to(torch.float32)
        idx = torch.stack([
            h0[:, None] * side + w0[None, :],
            h0[:, None] * side + w1[None, :],
            h1[:, None] * side + w0[None, :],
            h1[:, None] * side + w1[None, :],
        ], dim=-1).reshape(-1, 4)
        wt = torch.stack([
            (1 - dh)[:, None] * (1 - dw)[None, :],
            (1 - dh)[:, None] * dw[None, :],
            dh[:, None] * (1 - dw)[None, :],
            dh[:, None] * dw[None, :],
        ], dim=-1).reshape(-1, 4)
        if len(self._pos_cache) > 32:
            self._pos_cache.clear()
        self._pos_cache[(h, w)] = (idx, wt)
        return idx, wt

    def _pos_offsets(self, gt: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """逐 token 的学到 pos-embed，在 fp32 里求和（spec §14.3），按 token 顺序。

        整个抽头求和过程的累加器都保持 fp32，因为 fp16/bf16 的抽头求和会错 1e-2：
        带双线性权重的四行表被相加再 permute，中间和的舍入没法靠事后转换结果挽回。

        Args:
            gt: ``grid_thw``，``[n_img, 3]`` 行，每行是 ``(t, h, w)`` 的 patch 数。
            dtype: 返回的 dtype，通常是 ``x`` 的计算 dtype。

        Returns:
            ``[N, hidden]`` 位置嵌入，已经 permute 成 merge 块顺序并按帧重复，
            因此可以逐行加到 patch 嵌入上。

        Raises:
            ValueError: 某个网格边不是 ``spatial_merge_size`` 的倍数。2x2 合并会留下
                一个不完整的块，merger 折不了它的行，所以调用方要改图像尺寸，
                不是改网格。
        """
        c = self.cfg
        m, hidden = c.spatial_merge_size, c.hidden_size
        parts = []
        for t, h, w in gt.tolist():
            t, h, w = int(t), int(h), int(w)
            if h % m or w % m:
                raise ValueError(f"patch grid {h}x{w} is not a multiple of merge {m}")
            idx, wt = self._pos_taps(h, w)
            acc = torch.zeros(h * w, hidden, dtype=torch.float32, device=self.device)
            for k in range(4):
                gathered = self.pos_embed.index_select(0, idx[:, k]).to(torch.float32)
                acc.addcmul_(gathered, wt[:, k:k + 1])
                del gathered
            # 行主序 (h, w) -> merge 块顺序 (block_row, block_col, in_row, in_col)。
            # 这正是 pixel_values 已经在用的顺序，跳过 permute 会让此后每个 patch
            # 都错位。
            acc = (acc.view(h // m, m, w // m, m, hidden)
                   .permute(0, 2, 1, 3, 4).reshape(h * w, hidden))
            if t > 1:
                acc = acc.repeat(t, 1)
            parts.append(acc)
        out = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
        return out.to(dtype)

    def _hw_positions(self, h: int, w: int) -> tuple[torch.Tensor, torch.Tensor]:
        """每个 token 的 patch 网格 (行, 列)，按 merge 块顺序。

        Args:
            h: 一张图的 patch 网格行数。
            w: 一张图的 patch 网格列数。

        Returns:
            merge 块顺序下的 ``(rows, cols)``，int64 ``[h*w]``。rope
            angles are derived from these, so they must use the same order as
            the token stream; cached per ``(h, w)``.
        """
        hit = self._hw_cache.get((h, w))
        if hit is not None:
            return hit
        m = self.cfg.spatial_merge_size
        gh, gw = h // m, w // m
        block_rows = torch.arange(gh, device=self.device)
        block_cols = torch.arange(gw, device=self.device)
        intra = torch.arange(m, device=self.device)
        rows = (block_rows[:, None, None, None] * m
                + intra[None, None, :, None]).expand(gh, gw, m, m).reshape(-1)
        cols = (block_cols[None, :, None, None] * m
                + intra[None, None, None, :]).expand(gh, gw, m, m).reshape(-1)
        if len(self._hw_cache) > 32:
            self._hw_cache.clear()
        self._hw_cache[(h, w)] = (rows, cols)
        return rows, cols

    def _rope(self, gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """整个批的 fp32 cos/sin ``[N, head_dim]``，按 token 顺序。

        布局是 ``[cos_h | cos_w | cos_h | cos_w]``（各 16 个频率）：配合把 ``i`` 与
        ``i + head_dim/2`` 配对的 ``rotate_half``，就是 32 个独立的二维旋转，前 16 个
        用 H、后 16 个用 W。

        Args:
            gt: ``grid_thw``，``[n_img, 3]``。整个张量作为缓存键，所以图像相同的
                两次请求共用一张表。

        Returns:
            ``(cos, sin)``，各自 fp32 ``[N, head_dim]``，已按帧重复、并跨图像按
            ``gt`` 的行序拼接。用 fp32 是因为角度也是 HF 保留的 fp32 孤岛之一。
        """
        key = tuple(int(v) for v in gt.reshape(-1).tolist())
        hit = self._rope_cache.get(key)
        if hit is not None:
            return hit
        cos_parts, sin_parts = [], []
        for t, h, w in gt.tolist():
            rows, cols = self._hw_positions(int(h), int(w))
            fh = rows.to(torch.float32)[:, None] * self.inv_freq
            fw = cols.to(torch.float32)[:, None] * self.inv_freq
            ch, sh = fh.cos(), fh.sin()
            cw, sw = fw.cos(), fw.sin()
            cos = torch.cat([ch, cw, ch, cw], dim=-1)
            sin = torch.cat([sh, sw, sh, sw], dim=-1)
            if int(t) > 1:
                cos, sin = cos.repeat(int(t), 1), sin.repeat(int(t), 1)
            cos_parts.append(cos)
            sin_parts.append(sin)
        out = (cos_parts[0] if len(cos_parts) == 1 else torch.cat(cos_parts, dim=0),
               sin_parts[0] if len(sin_parts) == 1 else torch.cat(sin_parts, dim=0))
        if len(self._rope_cache) > 16:
            self._rope_cache.clear()
        self._rope_cache[key] = out
        return out

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """交换最后一维的前后两半，并给前半变号。

        用交错式 rope（``-x2, x1``）替代 HF 的 ``rotate_half`` 变体，得到的仍是
        单位模旋转，下游什么都不会报错——图只是错的。这里是 HF 的分半形式。

        Args:
            x: 最后一维是 ``head_dim`` 的 fp32 张量。

        Returns:
            同形状，半区完成旋转。
        """
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor,
                    cos: torch.Tensor, sin: torch.Tensor):
        """在 fp32 里旋转 q/k 再转回去——即 HF 的 ``apply_rotary_pos_emb_vision``。

        按 token 切块，fp32 拷贝因此维持在约 4 MB，而不是 N*heads*64*4。

        Args:
            q: 查询，``[N, heads, head_dim]``，按 token 顺序。
            k: 键，同形状。
            cos: 来自 :meth:`_rope` 的 fp32 ``[N, head_dim]``。
            sin: 来自 :meth:`_rope` 的 fp32 ``[N, head_dim]``。

        Returns:
            ``(q_out, k_out)``，形状与 dtype 恢复为输入。缓冲区是预分配的，切块
            循环因此不会推高峰值。
        """
        q_dtype, k_dtype = q.dtype, k.dtype
        q_out = torch.empty_like(q)
        k_out = torch.empty_like(k)
        for i0 in range(0, q.shape[0], _ROPE_TILE):
            i1 = min(i0 + _ROPE_TILE, q.shape[0])
            # [tile, 1, head_dim]：那个单元素维在头上广播。
            c = cos[i0:i1].unsqueeze(1)
            s = sin[i0:i1].unsqueeze(1)
            qf = q[i0:i1].to(torch.float32)
            q_out[i0:i1] = (qf * c + self._rotate_half(qf) * s).to(q_dtype)
            del qf
            kf = k[i0:i1].to(torch.float32)
            k_out[i0:i1] = (kf * c + self._rotate_half(kf) * s).to(k_dtype)
            del kf, c, s
        return q_out, k_out

    def _segments(self, gt: torch.Tensor) -> list[tuple[int, int]]:
        """每个注意力段的 ``[start, end)``：每图每帧一段。

        帧只在自己内部做注意力，所以 2 帧的图像产生两段而不是一段。这些段滑错位会
        让一帧看到另一帧，而没有任何数值测试会把它标成错误。

        Args:
            gt: ``grid_thw``，``[n_img, 3]``。

        Returns:
            按顺序覆盖 ``[0, N)`` 的半开 ``(start, end)`` 区间。
        """
        segs: list[tuple[int, int]] = []
        off = 0
        for t, h, w in gt.tolist():
            n = int(h) * int(w)
            for f in range(int(t)):
                segs.append((off + f * n, off + (f + 1) * n))
            off += int(t) * n
        return segs

    def cu_seqlens(self, gt: torch.Tensor) -> torch.Tensor:
        """``int32`` 的累积段长（``[0, ..., N]``），FA-varlen 风格。

        Args:
            gt: ``grid_thw``，``[n_img, 3]``。

        Returns:
            ``device`` 上 ``[n_seg + 1]`` 个 int32 偏移。用 int32 是因为 varlen
            注意力内核就要这个；视觉塔自己用 :meth:`_segments` 切片，这个形状只用在
            对比检查里上报。
        """
        bounds = [0] + [e for _s, e in self._segments(gt)]
        return torch.tensor(bounds, dtype=torch.int32, device=self.device)

    def _attn(self, blk: _Block, x: torch.Tensor, segs: list[tuple[int, int]],
              cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """打包的非因果注意力，一次处理一个查询切块的 score。

        各段靠切片隔离，不用掩码（视觉塔没有掩码输入；一张 ``[N, N]`` 的加性掩码
        也会成为塔里最大的张量）。Softmax 在 fp32 里跑，并在 ``p @ v`` 之前转回
        计算 dtype，和 HF 的 eager 路径完全一致，所以一个查询切块复现出的结果与
        完整 score 块相比，只差 fp32 求和次序。

        softmax 用 ``torch.softmax(..., dtype=fp32)`` 而不是手写 max/exp/div：
        一个融合内核替代对切块 fp32 拷贝的三趟遍历，而且它逐字就是 HF
        ``eager_attention_forward`` 做的事。

        Args:
            blk: 这一趟用其 ``qkv`` / ``proj`` 权重的块。
            x: pre-norm 激活，``[N, hidden]``。
            segs: 来自 :meth:`_segments` 的段。
            cos: fp32 ``[N, head_dim]`` rope 表。
            sin: fp32 ``[N, head_dim]`` rope 表。

        Returns:
            ``[N, hidden]`` 注意力输出，可直接加进残差。切块逐段覆盖，所以输出在
            每段内是精确的，同时 score 块从不整体实体化。
        """
        c = self.cfg
        n = x.shape[0]
        heads, hd = c.num_heads, c.head_dim
        qkv = self._linear(x, blk.qkv_w, blk.qkv_b)
        q, k, v = qkv.view(n, 3, heads, hd).permute(1, 0, 2, 3).unbind(0)
        del qkv
        q, k = self._apply_rope(q, k, cos, sin)
        # [heads, n, hd]：把头并成一批，每个切块的矩阵乘因此是一次 GEMM。
        q = q.transpose(0, 1).contiguous()
        k = k.transpose(0, 1).contiguous()
        v = v.transpose(0, 1).contiguous()
        out = torch.empty_like(q)
        tile = max(1, min(Q_TILE, n))
        for s, e in segs:
            # [heads, hd, seg]：每段转置一次，而不是每个切块一次。
            kt = k[:, s:e].transpose(-1, -2).contiguous()
            vs = v[:, s:e]
            for q0 in range(s, e, tile):
                q1 = min(q0 + tile, e)
                scores = torch.matmul(q[:, q0:q1], kt) * self.scaling
                p = torch.softmax(scores, dim=-1, dtype=torch.float32)
                out[:, q0:q1] = torch.matmul(p.to(self.compute_dtype), vs)
                del scores, p
            del kt, vs
        del q, k, v
        # (heads, n, hd) -> (n, heads*hd)：head 维必须在 proj 之前回到特征维*内部*，
        # 否则投影读到的是一个被重排过的向量。
        return self._linear(out.transpose(0, 1).reshape(n, heads * hd),
                            blk.proj_w, blk.proj_b)

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor,
                grid_thw: torch.Tensor) -> torch.Tensor:
        """``pixel_values [N, 1536]`` 加 ``grid_thw [n_img, 3]`` -> ``[N/4, 2560]``。

        输出行序就是 prompt 里 ``<|image_pad|>`` 的顺序：图像按 ``grid_thw`` 的行序
        排列，每张在其*合并后*的网格上按行主序展开，这正是 ``masked_scatter``（以及
        引擎的区间覆写）期待的布局。

        Args:
            pixel_values: processor 输出，``[N, 1536]``，行按 merge 块顺序。宽度必须
                等于 ``patch_dim``。
            grid_thw: ``[n_img, 3]`` 行，每行是 ``(t, h, w)`` 的 patch 数。

        Returns:
            ``[N / 4, 2560]`` 的图像嵌入，dtype 是视觉塔的输出 dtype。``N / 4`` 是
            因为 merger 把每个 2x2 patch 块折成一行；调用方用这些行替换 prompt 里的
            图像填充行，所以数量对不上会在 scatter 时立刻触发形状错误。

        Raises:
            ValueError: ``grid_thw`` 不是 ``[n, 3]``、行数与网格不符，或行宽不等于
                ``patch_dim``。三种都是 processor 与引擎不匹配；调用方应当拒绝该请求，
                而不是把一份错位的图像喂给 GEMM。
        """
        c = self.cfg
        gt = torch.as_tensor(grid_thw, dtype=torch.long, device=self.device)
        if gt.dim() != 2 or gt.shape[1] != 3:
            raise ValueError(f"grid_thw must be [n, 3], got {tuple(gt.shape)}")
        pv = pixel_values.to(device=self.device, dtype=self.compute_dtype)
        n_expect = int(gt.prod(dim=-1).sum().item())
        if pv.shape[0] != n_expect:
            raise ValueError(f"pixel_values has {pv.shape[0]} rows, grid implies {n_expect}")
        if pv.shape[1] != c.patch_dim:
            raise ValueError(f"pixel_values row width {pv.shape[1]} != {c.patch_dim}")

        x = F.linear(pv, self.patch_w, self.patch_b)
        del pv
        x = x + self._pos_offsets(gt, x.dtype)
        cos, sin = self._rope(gt)
        segs = self._segments(gt)

        # 两个 GELU 刻意不同：各块用 checkpoint 的 ``gelu_pytorch_tanh`` 近似，merger
        # 用 HF 精确的 erf ``nn.GELU``。对调会带来约 1e-3 的差异，而且没有任何形状
        # 检查能发现。
        for blk in self.blocks:
            x = x + self._attn(blk, self._ln(x, blk.norm1_w, blk.norm1_b), segs, cos, sin)
            h = self._ln(x, blk.norm2_w, blk.norm2_b)
            h = self._linear(h, blk.fc1_w, blk.fc1_b)
            h = F.gelu(h, approximate="tanh")
            x = x + self._linear(h, blk.fc2_w, blk.fc2_b)
            del h

        y = self._ln(x, self.merger_norm_w, self.merger_norm_b)
        del x
        # 2x2 的 merge 块 -> 一行：token 数减半，prompt 里图像填充区间已经按这个算过。
        y = y.reshape(-1, self.merge_dim)
        y = self._linear(y, self.merger_fc1_w, self.merger_fc1_b)
        # 默认的 ``F.gelu`` 就是 nn.GELU 用的精确 erf 形式；这里传
        # approximate="tanh" 用的就是各块的激活，而不是 merger 的。
        y = F.gelu(y)
        y = self._linear(y, self.merger_fc2_w, self.merger_fc2_b)
        return y.to(self.dtype)

    def __call__(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """:meth:`forward` 的别名，让视觉塔在模型里读起来像一个模块。

        Args:
            pixel_values: ``[N, 1536]`` 的 processor 输出。
            grid_thw: ``[n_img, 3]`` 的 patch 网格。

        Returns:
            ``[N / 4, 2560]`` 图像嵌入，与 :meth:`forward` 相同。
        """
        return self.forward(pixel_values, grid_thw)

    def num_params(self) -> int:
        """只按配置数视觉塔的参数，不碰存储。

        它存在的理由是 fp16 和 int8 两种模式参数量相同、字节数不同；一行引用参数的
        日志不该被迫遍历（从而实体化）``QWeight`` 矩阵。

        Returns:
            配置所描述架构的参数总数。
        """
        c = self.cfg
        n = c.hidden_size * c.patch_dim + c.hidden_size                    # patch 嵌入
        n += c.num_position_embeddings * c.hidden_size                     # 位置嵌入
        h, i = c.hidden_size, c.intermediate_size
        per_block = (3 * h * h + 3 * h) + (h * h + h) + (h * i + i) + (i * h + h) + 4 * h
        n += c.depth * per_block
        merge = h * c.merge_unit
        n += 2 * h + (merge * merge + merge) + (c.out_hidden_size * merge + c.out_hidden_size)
        return n

    def nbytes(self) -> int:
        """权重的常驻字节数（int8 模式约为 fp16 的一半）。

        Returns:
            实际占用的字节：int8 按每个权重 1 字节再加上逐行 fp32 scale 计——常驻预算
            要的是这个数，而不是参数量乘以一个元素大小。
        """
        total = 0
        tensors = [self.patch_w, self.patch_b, self.pos_embed,
                   self.merger_norm_w, self.merger_norm_b,
                   self.merger_fc1_w, self.merger_fc1_b,
                   self.merger_fc2_w, self.merger_fc2_b]
        for blk in self.blocks:
            for name in _Block.__slots__:
                tensors.append(getattr(blk, name))
        for t in tensors:
            total += t.nbytes() if isinstance(t, QWeight) else t.numel() * t.element_size()
        return total

    def summary(self) -> dict:
        """启动日志的内容：几何、dtype 和常驻权重大小。

        Returns:
            可直接 JSON 化的字典。``q_tile`` 是生效的切块宽度，``q_tile_env`` 是
            覆盖它的环境变量，部署里 ``VISION_Q_TILE`` 配错在日志里一眼可见。
        """
        return {"params": self.num_params(), "weight_GiB": self.nbytes() / 1024 ** 3,
                "quant": self.quant, "dtype": str(self.dtype),
                "compute_dtype": str(self.compute_dtype), "device": str(self.device),
                "q_tile": Q_TILE, "q_tile_env": "VISION_Q_TILE"}
