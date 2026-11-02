"""把 Qwen3.8-Flash-Next 的 360 GB bf16 checkpoint 转成 W4A8 分 rank 目录。

这是 ``coffin/tools/quant_disk_w4a8.py`` 在 Qwen4-Exp 上的专用版本。格式逐字复用它
——``uint8 [ceil(K/2), N]`` 打包 + ``fp32 [groups, N]`` 分组 scale、``q+8`` nibble
偏置、``group_size=128``——所以 ``coffin/quantized_load.py`` 认识这份索引，能把 q4
条目读回成 :class:`~w4a8_weights.W4A8Weight`。新增的是它没有的三件事：

1. **逐张量决策**。Qwen4 里不是"二维就该量化"：MoE 的 3D 融合张量要逐 expert 切、
   N-gram 表要整块踢出 GPU、门控/归一化/卷积要保精度。判定规则连同理由写在
   :func:`classify` 里，且只信 ``inventory.json`` 记录的形状，不硬编码层号或列宽。
2. **按 rank 切好的输出**。``ep4``（专家并行，``PORT_PLAN.md`` §3 选定的拓扑）与
   ``tp4``（张量并行，备选）各写一棵目录树，每个 rank 一棵，里面只放这个 rank 要读的
   字节。这里不做运行时：怎么并发是引擎的事，本工具只保证盘上已经切好、切点和理由
   都记在索引里。
3. **N-gram 表单独落盘并标记 host-resident**。51.2 B 参数 / 102.4 GB bf16 的表不进
   GPU，转成 int8 行 + 逐行 fp32 scale 写在自己的目录里。混进 GPU 权重会让每卡常驻
   直接超过 34 GB。

Module: ``coffin/tools``。只 import ``torch`` 与本包的叶子模块
（``quant_disk``、``quant_disk_w4a8``）；引擎任何代码都不 import 它。

Constraints:
    * 逐张量决策只能从 checkpoint 的形状推。形状对得上但语义不同（``[1,2560]`` 的
      ``shared_expert_gate`` 与 ``[2560,640]`` 的 down_proj）量化后都不报错，只会让某
      一层的输出缓慢漂移——"跑得通但结果不对"是最贵的一类错误，所以每个分支的
      docstring 都写了它防的是什么。
    * MoE 的 3D 张量**逐 expert** 量化，每个 expert 保持 ``[K, N]`` 行主序，堆成
      ``[E, ceil(K/2), N]``。绝不摊平成一个 2D 矩阵：``grouped_gemm`` 用固定偏移寻址，
      摊平会让第 e 个专家的起点算错，而形状仍然合法。
    * ``A_log``、``dt_bias``、所有 norm weight、``conv1d.weight`` 绝不进入 W4 路径。
      它们被逐元素内核消费（``ARCH_SPEC.md`` §11.1、``models/qwen35/weights.py:26-29``），
      量化后在错误精度下静默运行。
    * 切分在**量化之前**做：先切稠密切片、再对切片量化。倒过来会让一个 packed 字节的
      两个 nibble 分属两个 rank，或者让 K 切点落在 128 的组边界之外。同理也先切再转
      dtype：MoE 的 ``gate_up_proj`` 是 3.36 GB，先整份转 fp32 再切会白占一份副本。
    * 源 checkpoint 只读。输出目录必须是空的或是本工具上一次的续跑现场，避免两次转换
      的分片混进一个索引。
    * 一个源分片可能同时含 N-gram 表与普通权重（131 个分片里有 1 个），所以写出端按
      **目标角色**分流，而不是一个源分片对一个输出分片。
    * 每个源分片处理完就落盘并写一次进度。WebShell 会断，360 GB 的活必须能接着跑：
      进度文件只记"哪些源分片已经完整处理"，重跑时跳过它们。

用法::

    python -m tools.quant_disk_qwen4 --checkpoint /path/Qwen3.8-Flash-Next \\
        --out /path/qwen4-w4a8 --inventory docs/qwen4/inventory.json \\
        --layout ep4
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import torch

# 本文件在 ``coffin/tools/`` 下，直接跑脚本时 ``sys.path[0]`` 是工具目录而不是引擎根，
# 所以先把引擎根插进去；下面这条 import 必须留在这几行之后，否则 ``tools`` 包导入不到。
_ENGINE = Path(__file__).resolve().parents[1]
if str(_ENGINE) not in sys.path:
    sys.path.insert(0, str(_ENGINE))

from tools.quant_disk import (  # noqa: E402
    _WEIGHT_DTYPES, _config_digest, _dtype_name)
from tools.quant_disk_w4a8 import _NIBBLE_OFFSET, quantize_projection  # noqa: E402

#: 索引格式标识。与 ``quantized_load.W4A8_FORMAT`` 同值——加载器读的是这个字符串。
FORMAT = "coffin-w4a8-kn-v1"

#: 顶层清单的格式标识。刻意与 :data:`FORMAT` 不同：顶层目录不是一份可加载的权重目录
#: （它下面有 rank 子目录），拿它去 ``quantized_load.load`` 应该在第一步就报"格式不
#: 认识"，而不是读到半份权重。
MANIFEST_FORMAT = "coffin-qwen4-w4a8-manifest-v1"

#: 索引文件名，与 ``quantized_load.INDEX_NAME`` 同值。
INDEX_NAME = "quant_index.json"

#: N-gram 表的独立索引名。刻意不叫 ``quant_index.json``：那张表不是 GPU 权重，任何按
#: 目录名自动发现权重目录的路径（``quantized_load.is_quantized_dir``）都不该把它当成
#: 一个模型目录。
NGRAM_INDEX_NAME = "ngram_index.json"

#: 续跑进度文件名。放在输出根目录，与两份索引并列。
PROGRESS_NAME = "quant_progress.json"

#: N-gram 表的名字。128 个分片张量共享同一张逻辑表。
NGRAM_RE = re.compile(r"\.ple\.ple_embedding\.ngram_embedding\.shard_(\d+)\.weight$")

#: MoE 的 3D 融合张量。第 0 维是 expert，是 EP 的切分轴。
MOE_RE = re.compile(r"\.mlp\.experts\.(gate_up_proj|down_proj)$")

#: 视觉塔前缀。整塔保持 bf16，也不参与 rank 切分（每卡一份完整副本）。
VISUAL_PREFIX = "model.visual."

#: 输出宽度小于这个值的二维矩阵保持 bf16。它们是门和逐头标量：
#: ``block_inject_weight`` 是 ``[4,10240]``、``shared_expert_gate`` 是 ``[1,2560]``、
#: ``in_proj_a/b`` 是 ``[48,2560]``。全部乘在残差流或递推状态上，W4 的相对误差会直接
#: 改变动态，而它们的字节数加起来不到 20 MB，换不来带宽。
NARROW_N = 64

#: 额外保持 bf16 的二维矩阵后缀。router 决定 top-10 路由，量化噪声会翻转选择，而翻转
#: 后的输出分布仍然合理、内容却是错的——最难查的一类 bug。
DENSE_SUFFIXES = ("mlp.gate.weight",)

#: 整段复制的模块。超连接在 10240 宽的残差流上做 ``.mean(dim=-2)`` 的读门与逐流写门
#: （``ARCH_SPEC.md`` §6），把 4 路流切开就要跨 rank 求均值；PLE 注入同样是 10240 宽
#: 的直接相加（§7.7）。两者 W4 后加起来不到 0.4 GB，切了省不下带宽却引入跨 rank 语义。
REPLICATED_MODULES = ("hyper_connection", ".ple.")

#: 支持的分片布局。``ep4`` 按 expert 切，``tp4`` 按 N/K 轴切。
_LAYOUTS = ("ep4", "tp4")

#: 并行度。两种布局都是 4 卡，与 ``PORT_PLAN.md`` §2.2 的显存账一致。
_WORLD = 4

#: 本工具写的分片文件名。
_SHARD_NAME = "quant-w4a8-{layout}-r{rank}-{i:03d}.safetensors"

#: 一个输出分片的目标字节数。3 GiB 与源分片同量级，便于对照排查。进程每处理完一个源
#: 分片就把所有缓冲落盘（续跑的最小单位），所以实际分片是"源分片的贡献"与这个上限中
#: 较小的那个。
_SHARD_BYTES = 3 << 30

#: N-gram 表按行分块处理的块大小。一张表 2500012 行 × 160 维，整份转 fp32 是 1.6 GB；
#: 分块让峰值只看这个常数，与表高无关。
_NGRAM_CHUNK_ROWS = 1 << 18

#: N-gram 码的 dtype。对称 int8、无偏置——与 W4A8 的 ``q+8`` nibble 不同，这里一行一个
#: scale、一行一次反量化，加偏置只会在 host 侧查表时多出一次减法。
_NGRAM_CODES_DTYPE = torch.int8

#: MoE 张量按专家分块量化的块大小。
#:
#: 这条常数是被环境逼出来的，不是调优。本集群**登录节点会在约 2 GB RSS 上 SIGKILL**
#: 进程（实测 1 GB 分配成功、2 GB 被杀；而 ``ulimit -v/-m`` 与 cgroup ``memory.max``
#: 都是 unlimited，是个外部看门狗）。提交到计算节点的作业有约 35 GB，所以这条常数只在
#: 登录节点上决定性——但登录节点正是最容易被随手拿来跑一下的地方。
#:
#: 8 个专家一份时，``[8,1280,2560]`` 的 bf16 是 52 MB、转 fp32 是 105 MB、int16 码
#: 105 MB，峰值约 270 MB，任何节点都安全。再大就只是在省算子开销（实测 32 个一批比
#: 8 个一批快约 20%），而在 2 GB 的悬崖边上那 20% 不值得。
_MOE_CHUNK_EXPERTS = 8

#: ``in_proj_qkv`` 的三段融合宽度 ``[key(2048), key(2048), value(6144)]``，加起来正是
#: ``conv_dim=10240``。TP 切分必须按段切：按 10240/4=2560 整段切会让 rank 0 拿到
#: "全部 key1 + 半个 key2"，q/k/v 的头部对齐关系被破坏，前向仍然跑得通、注意力全错。
_QKV_SEGMENTS = (2048, 2048, 6144)


def _bytes_per_element(dtype_name: str) -> int:
    """把索引里的 dtype 名字换算成每元素字节数。

    Args:
        dtype_name: ``"bfloat16"`` 这类名字。

    Returns:
        字节数。未知名字按 2 字节算：本 checkpoint 里非 bf16 的只有 280 字节的 I64
        buffer，它们在 dense 路径上、字节账差几十字节不影响任何判断。
    """
    return {"float32": 4, "float16": 2, "bfloat16": 2, "int8": 1, "uint8": 1}.get(
        dtype_name, 2)


def _elem(shape: list[int]) -> int:
    """形状的元素个数。

    Args:
        shape: 形状。

    Returns:
        元素数。
    """
    total = 1
    for dim in shape:
        total *= dim
    return total


def _packed_shapes(n: int, k: int, group_size: int) -> tuple[int, int]:
    """算出 ``[N,K]`` 量化后的 packed 行数与组数。

    Args:
        n: 输出宽度。
        k: 输入宽度。
        group_size: 分组大小。

    Returns:
        ``(packed 行数 = ceil(padded_K/2), 组数)``。K 先向上取整到组边界、再取偶，与
        ``quant_disk_w4a8.quantize_projection`` 的填充规则逐字一致——这里算错一位，
        理论字节账和盘上实测就会差出几十 MB，而那种差异很容易被当成"正常浮动"。
    """
    groups = (k + group_size - 1) // group_size
    padded_k = groups * group_size
    return (padded_k + 1) // 2, groups


def _inventory_shape(inventory: dict, key: str) -> list[int]:
    """从 inventory 取一个张量的权威形状。

    Args:
        inventory: ``inventory.json`` 里 ``tensors`` 那一层。
        key: checkpoint 张量名。

    Returns:
        形状列表。

    Raises:
        KeyError: inventory 里没有这个键。缺键必须当场失败：跳过它会让输出目录少一个
            张量，而缺张量要到加载端的形状断言才暴露，报出来的位置离原因很远。
    """
    meta = inventory.get(key)
    if meta is None:
        raise KeyError(f"{key}: not in the inventory; the inventory is the authority on "
                       f"shapes, so a key it does not know cannot be quantised")
    return [int(x) for x in meta["shape"]]


def classify(key: str, shape: list[int]) -> str:
    """判断一个 checkpoint 张量在输出目录里走哪种落盘形态。

    规则按"判错了不报错"的代价从高到低排。

    Args:
        key: checkpoint 张量名。
        shape: 张量形状，来自 inventory 而不是 checkpoint 自己报的。

    Returns:
        ``"ngram"``（host 侧查表）、``"moe"``（逐 expert 的 W4）、``"q4"``、或
        ``"dense"``（原样 bf16）。
    """
    if NGRAM_RE.search(key):
        return "ngram"
    if MOE_RE.search(key):
        return "moe"
    # 视觉塔整塔保 bf16。它只有 0.9 GB，而且是计算受限的（一次只过一个 GEMM），量化
    # 换不到带宽；更重要的是塔的每个线性层都有 bias（§11.1），而这里没有量化 bias 的
    # 路径，混在一起会让"有 bias 的 q4 投影"这个组合在加载端无处安放。
    if key.startswith(VISUAL_PREFIX):
        return "dense"
    # conv1d 的核消费 fp32 抽头，形状是 [C,1,K] 的 3D，两个理由都让它离开二维路径。
    if key.endswith("conv1d.weight"):
        return "dense"
    if key.endswith(DENSE_SUFFIXES):
        return "dense"
    if len(shape) != 2:
        # A_log / dt_bias / 各种 norm weight / patch_embed 的 Conv3d / I64 buffer。
        return "dense"
    if shape[0] < NARROW_N:
        return "dense"
    return "q4"


def group_of(key: str) -> str:
    """把张量名归到模块组，供字节统计按组汇总。

    Args:
        key: checkpoint 张量名。

    Returns:
        组名。层级编号被抹平，于是 48 层的同类张量汇总成一行。
    """
    if key.startswith(VISUAL_PREFIX):
        return "视觉塔"
    if key.startswith("mtp."):
        return "MTP"
    if NGRAM_RE.search(key):
        return "N-gram 表 (host)"
    if key.endswith(("embed_tokens.weight", "lm_head.weight")):
        return "embed + lm_head"
    if ".mlp.experts." in key:
        return "MoE experts"
    if ".mlp." in key:
        return "共享专家 + router"
    if ".linear_attn." in key:
        return "GDN"
    if ".self_attn." in key:
        return "QSA"
    if ".ple." in key:
        return "PLE 投影"
    if "hyper_connection" in key:
        return "Gated Residual"
    return "其它"


def _is_replicated(key: str) -> bool:
    """报告某个张量是否在 rank 之间整份复制。

    Args:
        key: checkpoint 张量名。

    Returns:
        ``True`` 表示每个 rank 各持一份完整副本。超连接与 PLE 必须整份复制：跨 rank
        求 10240 宽残差流的均值要把 all-reduce 塞进每层，而它们 W4 后不到 0.4 GB。
    """
    return any(token in key for token in REPLICATED_MODULES)


def _segments_for(n: int) -> list[int]:
    """把输出宽度拆成融合段，供 TP 按段切。

    Args:
        n: 张量输出宽度。

    Returns:
        段宽列表。只有 ``in_proj_qkv`` 的 10240 是三段融合，其余按整份处理。
    """
    return list(_QKV_SEGMENTS) if n == sum(_QKV_SEGMENTS) else [n]


def _split_axis(key: str, shape: list[int]) -> str:
    """决定一个张量在 ``tp4`` 下沿哪根轴切。

    Args:
        key: checkpoint 张量名。
        shape: 张量形状。

    Returns:
        ``"n"``（列并行）、``"k"``（行并行）或 ``"replicate"``（整份复制）。

    Raises:
        ValueError: 张量的 N 与 K 都不能被并行度整除。静默地把除不尽的余数留给 rank 0
            会让各 rank 形状不同，而错位要到 GEMM 才暴露。
    """
    if _is_replicated(key):
        return "replicate"
    kind = classify(key, shape)
    if kind == "dense":
        return "replicate"
    if kind == "moe":
        # gate_up_proj 是 [E, 2I, H]，列并行沿 N；down_proj 是 [E, H, I]，行并行沿 K。
        return "n" if key.endswith("gate_up_proj") else "k"
    if key.endswith(("embed_tokens.weight", "lm_head.weight")):
        return "n"
    if key.endswith((".mlp.shared_expert.gate_proj.weight",
                     ".mlp.shared_expert.up_proj.weight",
                     ".linear_attn.in_proj_qkv.weight",
                     ".linear_attn.in_proj_z.weight",
                     ".self_attn.q_proj.weight")):
        return "n"
    if key.endswith((".mlp.shared_expert.down_proj.weight",
                     ".linear_attn.out_proj.weight",
                     ".self_attn.o_proj.weight")):
        return "k"
    n, k = int(shape[-2]), int(shape[-1])
    if n % _WORLD and k % _WORLD:
        raise ValueError(f"{key}: neither N={n} nor K={k} is divisible by {_WORLD}")
    # 剩下的是 KV 头（nkv=2 < 4，切了就劈开半个头）与 indexer 的 640 宽投影
    # （5×128，除以 4 落在 128 的软边界上）。两份都很小，整份复制比切错便宜。
    return "replicate"


def _rank_slices(key: str, shape: list[int], layout: str, rank: int
                 ) -> tuple[list[tuple[int, int, int]], list[tuple[int, int]] | None]:
    """算出一个 rank 对某个张量要取的稠密切片。

    Args:
        key: checkpoint 张量名。
        shape: 原始形状。
        layout: ``"ep4"`` 或 ``"tp4"``。
        rank: 当前 rank。

    Returns:
        ``(slices, segments)``。``slices`` 是 ``(dim, start, stop)`` 列表，``dim`` 是
        张量自身维度下标；多个条目表示要在该维上把各段拼起来。``segments`` 只对列并行
        切 ``in_proj_qkv`` 这类融合张量非空，记录每段在源张量上对应的行区间，写进索引
        让加载端还原头部对齐。

    Raises:
        ValueError: 布局与张量形状不兼容（除不尽）。让不兼容的切分默默产出短一截的
            张量，会变成 GEMM 里的错位而不是这里的一条异常。
    """
    kind = classify(key, shape)
    if layout == "ep4":
        # EP 只切专家，其余权重整份复制：PORT_PLAN §3 选的就是这个拓扑，MoE 之外的
        # 部分每卡全量，好处是除 MoE 的 all-to-all 之外一次集合通信都不需要。
        if kind != "moe":
            return [], None
        experts = int(shape[0])
        if experts % _WORLD:
            raise ValueError(f"{key}: {experts} experts is not divisible by EP={_WORLD}; "
                             f"grouped_gemm 的固定偏移寻址要求各 rank 专家数相同")
        per_rank = experts // _WORLD
        return [(0, rank * per_rank, (rank + 1) * per_rank)], None

    axis = _split_axis(key, shape)
    if axis == "replicate":
        return [], None
    last = len(shape) - 1
    if axis == "n":
        rows = int(shape[-2])
        widths = _segments_for(rows) if (kind == "q4" and last == 1) else [rows]
        slices: list[tuple[int, int, int]] = []
        segments: list[tuple[int, int]] = []
        offset = 0
        for width in widths:
            if width % _WORLD:
                raise ValueError(f"{key}: segment width {width} is not divisible by "
                                 f"TP={_WORLD}; the fused layout would be misaligned")
            step = width // _WORLD
            start = offset + rank * step
            slices.append((last - 1, start, start + step))
            segments.append((start, start + step))
            offset += width
        return slices, (segments if len(segments) > 1 else None)
    width = int(shape[last])
    if width % _WORLD:
        raise ValueError(f"{key}: K={width} is not divisible by TP={_WORLD}")
    step = width // _WORLD
    return [(last, rank * step, (rank + 1) * step)], None


def _read_slice(slicer: object, slices: list[tuple[int, int, int]],
                shape: list[int]) -> torch.Tensor:
    """按 ``(dim, start, stop)`` 列表从 ``safetensors`` 切片器读出本 rank 的稠密切片。

    只读需要的那一段，而不是 ``get_tensor`` 整份。``embed_tokens`` 是 1.27 GB，整份读
    进内存就超过本集群登录节点约 2 GB 的 RSS 上限；而每个 rank 只要它的 1/4。

    Args:
        slicer: ``safe_open(...).get_slice(key)`` 返回的 ``PySafeSlice``。
        slices: :func:`_rank_slices` 给出的切片列表；为空表示整份都要。
        shape: 张量的完整形状，用于在空切片时定位整份。

    Returns:
        本 rank 的稠密切片。多段时在那一维上拼接。

    Raises:
        ValueError: 多段切片落在不同维度上。不同维度上拼段会让张量形状看起来仍然合法，
            而内容按错误的轴重排。
    """
    dims = len(shape)
    if not slices:
        return slicer[tuple(slice(None) for _ in range(dims))]
    if len(slices) == 1:
        dim, start, stop = slices[0]
        index = [slice(None)] * dims
        index[dim] = slice(start, stop)
        return slicer[tuple(index)]
    unique = {dim for dim, _, _ in slices}
    if len(unique) != 1:
        raise ValueError(f"multi-segment slices must share one dim, got {sorted(unique)}")
    dim = unique.pop()
    parts = []
    for _, start, stop in slices:
        index = [slice(None)] * dims
        index[dim] = slice(start, stop)
        parts.append(slicer[tuple(index)])
    return torch.cat(parts, dim=dim)


def _apply_slices(tensor: torch.Tensor, slices: list[tuple[int, int, int]]) -> torch.Tensor:
    """按 ``(dim, start, stop)`` 列表切一个张量；多段时在那一维上拼接。

    切片先于 dtype 转换调用，所以这里拿到的是 checkpoint 里的原始张量：``narrow`` 是
    视图，只有多段拼接才复制，而那种情况最多是 ``in_proj_qkv`` 的三段。

    Args:
        tensor: 原始张量。
        slices: :func:`_rank_slices` 给出的切片列表。

    Returns:
        本 rank 的切片。``slices`` 为空时返回原张量本身。

    Raises:
        ValueError: 多段切片的维度不一致。不同维度上拼段会让张量形状看起来仍然合法，
            而内容按错误的轴重排。
    """
    if not slices:
        return tensor
    if len(slices) == 1:
        dim, start, stop = slices[0]
        return tensor.narrow(dim, start, stop - start)
    dims = {dim for dim, _, _ in slices}
    if len(dims) != 1:
        raise ValueError(f"multi-segment slices must share one dim, got {sorted(dims)}")
    dim = dims.pop()
    return torch.cat([tensor.narrow(dim, start, stop - start) for _, start, stop in slices],
                     dim=dim)


def quantize_moe(dense: torch.Tensor, group_size: int,
                 chunk_experts: int = _MOE_CHUNK_EXPERTS
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """把一个 3D MoE 融合张量**分批**量化，再沿 expert 维拼成连续张量。

    每个 expert 是 ``[N, K]``（``gate_up_proj[e]`` 为 ``[2I,H]``、``down_proj[e]`` 为
    ``[H,I]``），所以直接套 ``quantize_projection`` 的 ``[N,K]`` 约定，不转置。所有
    expert 的 packed 形状相同，拼成 ``[E, ceil(K/2), N]``——``grouped_gemm`` 按固定偏移
    寻址，要的正是这个布局。

    为什么按批而不是整份算：整份 ``.float()`` 对 ``[512,1280,2560]`` 是 13.4 GB 的中间量，
    而本集群登录节点约 2 GB RSS 就 SIGKILL。分批之后峰值只与 ``chunk_experts`` 有关，
    与总专家数无关；每批的算术完全独立（分组 absmax 只沿 K、逐输出列），所以结果与整份
    逐位相同——``diag/qwen4/quant_selftest.py`` 里有断言。

    Args:
        dense: ``[E, N, K]`` 的稠密融合权重，已在 CPU 上。
        group_size: K 轴分组大小。
        chunk_experts: 每批处理的专家数。

    Returns:
        ``(packed [E, ceil(padded_K/2), N] uint8, scale [E, groups, N] fp32)``，两者都连续。

    Raises:
        ValueError: 输入不是三维，K 不是 ``group_size`` 的整数倍，或 ``chunk_experts``
            非正。二维输入走这里说明 :func:`classify` 的判据被绕过了，而按单 expert 处理
            会静默产出一份少 511 个专家的权重。K 不是整数倍时分组会跨 expert 边界取
            absmax，那样 scale 属于两个不同的 expert、加载端按 ``[E, groups, N]`` 索引
            就会整体错位。
    """
    if dense.dim() != 3:
        raise ValueError(f"moe fused weight must be 3-D, got {tuple(dense.shape)}")
    if chunk_experts <= 0:
        raise ValueError(f"chunk_experts must be positive, got {chunk_experts}")
    experts, n, k = (int(x) for x in dense.shape)
    if k % group_size:
        raise ValueError(f"moe fused weight has K={k}, not a multiple of group_size="
                         f"{group_size}; the group absmax would span expert boundaries")
    rows, groups = _packed_shapes(n, k, group_size)
    packed = torch.empty((experts, rows, n), dtype=torch.uint8)
    scale = torch.empty((experts, groups, n), dtype=torch.float32)
    for start in range(0, experts, chunk_experts):
        stop = min(experts, start + chunk_experts)
        piece_packed, piece_scale = _quantize_moe_block(dense[start:stop], group_size)
        packed[start:stop] = piece_packed
        scale[start:stop] = piece_scale
        del piece_packed, piece_scale
    return packed, scale


def quantize_moe_streaming(slicer: object, experts: int, shape: list[int],
                           group_size: int,
                           chunk_experts: int = _MOE_CHUNK_EXPERTS
                           ) -> tuple[torch.Tensor, torch.Tensor]:
    """沿 expert 维**分段读**再量化，峰值与总专家数无关。

    这是 MoE 路径真正该用的入口。``quantize_moe`` 要求整份稠密张量已在内存里，而
    ``gate_up_proj`` 是 3.36 GB——读那一步就超过本集群登录节点约 2 GB 的 RSS 上限
    （实测 1 GB 分配成功、2 GB 被 SIGKILL，而 ulimit 与 cgroup 都是 unlimited，是外部
    看门狗）。``safe_open().get_slice()[a:b]`` 只读第 0 维的那一段，于是峰值只与
    ``chunk_experts`` 有关。

    Args:
        slicer: ``safe_open(...).get_slice(key)`` 返回的 ``PySafeSlice``。
        experts: 张量第 0 维的长度。
        shape: 完整形状 ``[E, N, K]``。
        group_size: K 轴分组大小。
        chunk_experts: 每批读取并量化的专家数。

    Returns:
        ``(packed [E, ceil(padded_K/2), N], scale [E, groups, N])``。

    Raises:
        ValueError: 形状或分组不合法。这些在这里挡下来，比让下游按错位的 scale 建模型好。
    """
    _, n, k = (int(x) for x in shape)
    if k % group_size:
        raise ValueError(f"moe fused weight has K={k}, not a multiple of group_size="
                         f"{group_size}")
    rows, groups = _packed_shapes(n, k, group_size)
    packed = torch.empty((experts, rows, n), dtype=torch.uint8)
    scale = torch.empty((experts, groups, n), dtype=torch.float32)
    for start in range(0, experts, chunk_experts):
        stop = min(experts, start + chunk_experts)
        block = slicer[start:stop]
        if list(block.shape) != [stop - start, n, k]:
            raise ValueError(f"slice [{start}:{stop}] returned shape {list(block.shape)}, "
                             f"expected {[stop - start, n, k]}")
        piece_packed, piece_scale = _quantize_moe_block(block, group_size)
        packed[start:stop] = piece_packed
        scale[start:stop] = piece_scale
        del block, piece_packed, piece_scale
    return packed, scale


def _quantize_moe_block(dense: torch.Tensor, group_size: int
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    """量化一小批 ``[C, N, K]`` 的专家，返回 ``[C, ceil(K/2), N]`` 与 ``[C, groups, N]``。

    算术与 ``quant_disk_w4a8.quantize_projection`` 逐位一致：每组 absmax/7 作 scale，
    round 后 clamp 到 [-8,7]，加 8 平移成无符号 nibble，低 nibble 存偶数 K。

    Args:
        dense: ``[C, N, K]`` 稠密权重，``C`` 是一批专家数。
        group_size: K 轴分组大小。

    Returns:
        ``(packed, scale)``。

    Raises:
        ValueError: 输入不是三维，或 K 不是 ``group_size`` 的整数倍。
    """
    if dense.dim() != 3:
        raise ValueError(f"expected 3-D, got {tuple(dense.shape)}")
    chunks, n, k = (int(x) for x in dense.shape)
    if k % group_size:
        raise ValueError(f"K={k} is not a multiple of group_size={group_size}")
    groups = k // group_size
    values = dense.float().reshape(chunks, n, groups, group_size)
    scale = values.abs().amax(dim=-1).clamp_min_(1e-8).div_(7.0)
    signed = (values / scale[..., None]).round_().clamp_(-8, 7)
    del values
    codes = (signed.to(torch.int16) + _NIBBLE_OFFSET).reshape(chunks, n, k)
    del signed
    pairs = codes.reshape(chunks, n, k // 2, 2)
    packed = (pairs[..., 0] | (pairs[..., 1] << 4)).to(torch.uint8)
    del codes, pairs
    return packed.permute(0, 2, 1).contiguous(), scale.permute(0, 2, 1).contiguous()


def quantize_ngram_streaming(slicer: object, rows: int, dim: int,
                             chunk_rows: int = _NGRAM_CHUNK_ROWS
                             ) -> tuple[torch.Tensor, torch.Tensor]:
    """沿行分段读一张 N-gram 表并量化成 int8 加逐行 fp32 scale。

    为什么必须分段：整张表是 2500012 x 160 bf16 = 800 MB，``.float()`` 再放大一倍到
    1.6 GB，加输出 400 MB——超过本集群登录节点约 2 GB 的 RSS 上限（实测 1 GB 分配成功、
    2 GB 被 SIGKILL，而 ulimit 与 cgroup 都是 unlimited，是外部看门狗）。分段之后峰值只
    看 ``chunk_rows``，与表高无关。

    与 W4A8 不同，这里**不分组**：一行就是一个 160 维向量、查一行解一行，分组在
    160/128 的尾块上只会多算一次除法而换不到精度。逐行 scale 也让 host 侧查表变成
    "取一行 int8、乘一个标量"，不需要跨行状态。

    Args:
        slicer: ``safe_open(...).get_slice(key)`` 返回的 ``PySafeSlice``。
        rows: 表的行数。
        dim: 表的宽度。
        chunk_rows: 每次读取并处理的行数。

    Returns:
        ``(int8 [rows, dim], fp32 [rows])``。全零行被夹到 ``1e-8``，于是量化成零、解回
        来还是零，而不是 NaN。

    Raises:
        ValueError: 分段读回来的形状与声明不符。形状不符说明索引用错了切片，继续按错位
            的行往下写，得到的是一张"每行都还在、但错行"的表——查表不会报错。
    """
    codes = torch.empty((rows, dim), dtype=_NGRAM_CODES_DTYPE)
    scale = torch.empty(rows, dtype=torch.float32)
    for start in range(0, rows, chunk_rows):
        stop = min(rows, start + chunk_rows)
        block = slicer[start:stop]
        if list(block.shape) != [stop - start, dim]:
            raise ValueError(f"ngram slice [{start}:{stop}] returned shape "
                             f"{list(block.shape)}, expected {[stop - start, dim]}")
        values = block.float()
        row_scale = values.abs().amax(-1).clamp_min_(1e-8).div_(127.0)
        codes[start:stop] = (values / row_scale[:, None]).round_().clamp_(-127, 127)
        scale[start:stop] = row_scale
        del block, values, row_scale
    return codes, scale


def quantize_ngram(tensor: torch.Tensor, group_size: int = 0
                   ) -> tuple[torch.Tensor, torch.Tensor]:
    """把一整张已读入内存的 N-gram 表量化成 int8 加逐行 fp32 scale。

    与 :func:`quantize_ngram_streaming` 算术相同，只是输入已经在内存里。保留它是因为
    单元测试和小样本验证需要一个不依赖 ``safetensors`` 的入口；真 checkpoint 的路径
    走 streaming 那条——整张表 800 MB，整份读进来就会超过登录节点的 RSS 上限。

    与 W4A8 不同，这里**不分组**：一行就是一个 160 维向量、查一行解一行，分组在
    160/128 的尾块上只会多算一次除法而换不到精度。逐行 scale 也让 host 侧查表变成
    "取一行 int8、乘一个标量"，不需要跨行状态。

    Args:
        tensor: ``[rows, dim]`` 的整张表。
        group_size: 未使用，仅为让调用点与 W4 路径的形状对称。

    Returns:
        ``(int8 [rows, dim], fp32 [rows])``。全零行被夹到 ``1e-8``，于是量化成零、解回
        来还是零，而不是 NaN。

    Raises:
        ValueError: 输入不是二维。
    """
    del group_size
    if tensor.dim() != 2:
        raise ValueError(f"ngram table must be 2-D, got {tuple(tensor.shape)}")
    rows, dim = int(tensor.shape[0]), int(tensor.shape[1])
    codes = torch.empty((rows, dim), dtype=_NGRAM_CODES_DTYPE)
    scale = torch.empty(rows, dtype=torch.float32)
    for start in range(0, rows, _NGRAM_CHUNK_ROWS):
        stop = min(rows, start + _NGRAM_CHUNK_ROWS)
        values = tensor[start:stop].float()
        row_scale = values.abs().amax(-1).clamp_min_(1e-8).div_(127.0)
        codes[start:stop] = (values / row_scale[:, None]).round_().clamp_(-127, 127)
        scale[start:stop] = row_scale
        # 每轮显式丢掉 fp32 临时量。不删的话它们要等下一次循环覆盖才释放，而这张表
        # 有 10 轮，叠起来的峰值会超过单轮的两倍。
        del values, row_scale
    return codes, scale


class ShardWriter:
    """把 checkpoint 切分结果按 rank 分组写盘时持有的 safetensors 写出端。

    一个 ``ShardWriter`` 对应一个输出目录（一个 rank、一种布局）。它按目标字节上限滚动
    开新分片，并记下每个张量落到哪一片——索引里 ``shard`` 字段就取自这里。

    建它的时候目录必须已经存在：目录结构由 :func:`convert` 一次建好，写出端途中自己
    建目录的话，两个写出端可能同时认领同一个不存在的路径。

    Args:
        out: 输出目录。
        layout: 布局名，只用于分片文件名。
        rank: rank 编号，只用于分片文件名。
        start_index: 分片序号起点。续跑时从已有分片数接着编，重头编会覆盖上一轮的分片。
        limit_bytes: 单个分片的目标上限。

    Raises:
        OSError: 输出目录不存在。这条显式失败比让 ``save_file`` 报一个路径错误有用：
            后者会说"没有这个文件"，而原因在被调用的上一行。
    """

    def __init__(self, out: Path, *, layout: str, rank: int, start_index: int = 0,
                 limit_bytes: int = _SHARD_BYTES) -> None:
        if not out.is_dir():
            raise OSError(f"shard writer needs an existing directory: {out}")
        self.out = out
        self.layout = layout
        self.rank = rank
        self.limit = limit_bytes
        self._payload: dict[str, torch.Tensor] = {}
        self._size = 0
        self._index = start_index
        self.files: dict[str, dict] = {}
        self.totals: Counter = Counter()

    def add(self, key: str, tensor: torch.Tensor, *, bucket: str) -> str:
        """把张量放进当前分片，必要时先落盘再开新分片。

        Args:
            key: 写进 safetensors 的键名。
            tensor: 要写的张量，必须在 CPU 上连续。
            bucket: 记账桶名，``"w4"`` / ``"scale"`` / ``"dense"`` / ``"int8"``。

        Returns:
            这个张量落在哪个分片文件名里。
        """
        nbytes = tensor.numel() * tensor.element_size()
        if self._payload and self._size + nbytes > self.limit:
            self.flush()
        name = self._name()
        self._payload[key] = tensor
        self._size += nbytes
        self.totals[bucket] += nbytes
        return name

    def flush(self) -> None:
        """把当前分片写盘并清空缓冲。"""
        if not self._payload:
            return
        from safetensors.torch import save_file

        name = self._name()
        save_file(self._payload, str(self.out / name), metadata={"format": FORMAT})
        self.files[name] = {"tensors": len(self._payload), "bytes": self._size}
        self._payload = {}
        self._size = 0
        self._index += 1

    def close(self) -> None:
        """写掉最后一片。"""
        self.flush()

    def _name(self) -> str:
        """当前分片文件名。

        Returns:
            形如 ``quant-w4a8-ep4-r0-000.safetensors``。
        """
        return _SHARD_NAME.format(layout=self.layout, rank=self.rank, i=self._index)


def _emit_moe_streaming(slicer: object, key: str, shape: list[int],
                        layouts: tuple[str, ...], rank_dirs: dict, writers: dict,
                        indexes: dict, group_size: int) -> None:
    """沿 expert 维分段读、分段量化一个 MoE 张量，逐个 rank 写进各自的目录。

    内存账：每一步只持有 ``_MOE_CHUNK_EXPERTS`` 个专家的 bf16 块（8 个时约 52 MB）、
    它的 fp32 中间量（约 105 MB）与量化结果（约 105 MB），加上每个目标 rank 目录**一个**
    张量的输出缓冲（``[128, ceil(K/2), N]``，gate_up 约 210 MB）。峰值与总专家数无关。

    为什么不一次性量化完再分给各 rank：量化结果是 ``[512, 640, 1280]`` 的 uint8 加
    ``[512, 20, 640]`` 的 fp32，合计约 1.0 GB；叠在读块与写出端 payload 上仍然会撞到
    登录节点约 2 GB 的上限。逐步流的版本把这个常数也去掉。

    Args:
        slicer: ``safe_open(...).get_slice(key)`` 返回的 ``PySafeSlice``。
        key: checkpoint 参数名。
        shape: 完整 3D 形状 ``[E, N, K]``。
        layouts: 本次要写出的布局。
        rank_dirs: ``(layout, rank)`` 到目录的映射，只为算目标清单。
        writers: ``(layout, rank)`` 到写出端的映射。
        indexes: ``(layout, rank)`` 到索引条目字典的映射，就地写入。
        group_size: K 轴分组大小。

    Raises:
        ValueError: expert 数不能被并行度整除，或分段读回来的形状与声明不符。切不匀时
            每个 rank 的专家数不同，``grouped_gemm`` 的固定偏移寻址不再成立，而运行时
            只表现为"某些 token 用错专家"。
    """
    experts_total, n, k = (int(x) for x in shape)
    per_rank = experts_total // _WORLD
    if per_rank * _WORLD != experts_total:
        raise ValueError(f"{key}: {experts_total} experts is not divisible by EP={_WORLD}")
    rows, groups = _packed_shapes(n, k, group_size)
    for rank in range(_WORLD):
        targets = [(layout, rank) for layout in layouts if (layout, rank) in writers]
        if not targets:
            continue
        # 每个目标预先分配本 rank 的完整输出缓冲，再按块填进去。预分配而不是"攒一批
        # 再 concat"，是因为 concat 的瞬间会同时活着两份。
        packed_buf = torch.empty((per_rank, rows, n), dtype=torch.uint8)
        scale_buf = torch.empty((per_rank, groups, n), dtype=torch.float32)
        # ``rank_lo`` 是这个 rank 在整个张量上的起点，同时也是输出缓冲里的偏移基点：
        # 块是绝对下标（``chunk_lo`` 从 rank_lo 数起），写进缓冲要减去 rank_lo。漏掉这个
        # 减法会让 rank 1..3 每个块都写到缓冲开头，最后每个 rank 只有最后一块的数据、
        # 其余是未初始化的——而形状完全合法。
        rank_lo = rank * per_rank
        for offset, chunk_lo in enumerate(range(rank_lo, rank_lo + per_rank,
                                                _MOE_CHUNK_EXPERTS)):
            chunk_hi = min(rank_lo + per_rank, chunk_lo + _MOE_CHUNK_EXPERTS)
            block = slicer[chunk_lo:chunk_hi]
            if list(block.shape) != [chunk_hi - chunk_lo, n, k]:
                raise ValueError(f"{key}: slice [{chunk_lo}:{chunk_hi}] returned "
                                 f"{list(block.shape)}, expected {[chunk_hi - chunk_lo, n, k]}")
            piece_packed, piece_scale = _quantize_moe_block(block, group_size)
            del block
            start = chunk_lo - rank_lo
            stop = start + (chunk_hi - chunk_lo)
            packed_buf[start:stop] = piece_packed
            scale_buf[start:stop] = piece_scale
            del piece_packed, piece_scale
        source_experts = (rank_lo, rank_lo + per_rank)
        for dest in targets:
            _emit_moe(writers[dest], key, packed_buf, scale_buf, indexes[dest],
                      full_shape=shape, group_size=group_size,
                      source_experts=source_experts)
        del packed_buf, scale_buf


def _emit_q4(writer: ShardWriter, key: str, dense: torch.Tensor, group_size: int,
             entries: dict[str, dict], *, full_shape: list[int],
             segments: list[tuple[int, int]] | None,
             built: tuple[torch.Tensor, torch.Tensor] | None = None) -> tuple:
    """量化一个二维切片并写进分片，同时登记索引条目。

    Args:
        writer: 目标目录的写出端。
        key: checkpoint 参数名，直接用作 safetensors 键名。
        dense: 本 rank 的 ``[N,K]`` 稠密切片。
        group_size: K 轴分组大小。
        entries: 这个 rank 的索引 ``tensors`` 字典，就地写入。
        full_shape: 原始张量的完整形状，写进索引供加载端核对。
        segments: 列并行切融合张量时每段的源行区间；整份取用时为 ``None``。
        built: 已经量化好的 ``(packed, scale)``。同一份稠密切片会被最多 8 个目标目录
            需要（EP=4 的 4 个复制 rank × 2 种布局），传进来就复用，不再算第二遍。

    Returns:
        ``(packed, scale)``，供下一个目标目录复用。
    """
    packed, scale = built or quantize_projection(dense, group_size)
    n, k = int(dense.shape[0]), int(dense.shape[1])
    entry = {
        "kind": "q4",
        "shape": [n, k],
        "full_shape": full_shape,
        "n": n,
        "k": k,
        "group_size": group_size,
        "groups": int(scale.shape[0]),
        "packed_shape": list(packed.shape),
        "scale_shape": list(scale.shape),
        "scale_key": key + "#scale",
        "shard": writer.add(key, packed, bucket="w4"),
    }
    writer.add(key + "#scale", scale, bucket="scale")
    if segments is None:
        entry["replicated"] = True
    else:
        # 融合张量按段切之后，加载端要知道每段在源张量上的行区间才能把 q/k/v 的头部
        # 对齐接回去。靠"段宽是 2048/2048/6144"再推一遍是可行的，但那等于把切分算术
        # 抄到加载端，两处一旦漂移就是静默的头部错位。
        entry["source_segments"] = [list(s) for s in segments]
    entries[key] = entry
    return packed, scale


def _emit_moe(writer: ShardWriter, key: str, packed: torch.Tensor, scale: torch.Tensor,
              entries: dict[str, dict], *, full_shape: list[int], group_size: int,
              source_experts: tuple[int, int] | None) -> None:
    """把**已经是本 rank 那一段**的 MoE 量化结果写进一个 rank 目录。

    这里不再切 buffer：调用方（:func:`_emit_moe_streaming`）交进来的就是本 rank 的
    完整一段，``source_experts`` 只用于索引元数据，告诉加载端"这几个专家是整个张量的
    哪一段"。

    为什么要把这两件事分开：上一个版本用 ``source_experts`` 同时在缓冲上做切片，于是
    rank 1..3 拿着 128 行的缓冲去切 ``[128:256]``，得到 0 行——索引里 ``experts=0``、
    形状合法，而权重是空的。这种错不会在任何形状校验里报出来。

    Args:
        writer: 目标目录的写出端。
        key: checkpoint 参数名。
        packed: 本 rank 的 ``[per_rank, ceil(K/2), N]`` uint8。
        scale: 本 rank 的 ``[per_rank, groups, N]`` fp32。
        entries: 这个 rank 的索引 ``tensors`` 字典，就地写入。
        full_shape: 原始 3D 形状。
        group_size: K 轴分组大小。
        source_experts: 本段在整个张量上对应的 expert 区间；整份持有时为 ``None``。
    """
    experts = int(packed.shape[0])
    groups = int(scale.shape[1])
    # N 和 K 只能从原始形状取。packed 是 ``[E, ceil(K/2), N]``，直接按位置解包会把
    # ceil(K/2) 当成 N、把 N 当成 K——两个数都是合法整数，写进索引后加载端按它们
    # 校验形状会一路通过，而分组 scale 的索引整体错位。
    n, k = int(full_shape[1]), int(full_shape[2])
    entry = {
        "kind": "q4-moe",
        "shape": [experts, n, k],
        "full_shape": full_shape,
        "experts": experts,
        "expert_axis": 0,
        "n": n,
        "k": k,
        "group_size": group_size,
        "groups": groups,
        "packed_shape": list(packed.shape),
        "scale_shape": list(scale.shape),
        "scale_key": key + "#scale",
        "shard": writer.add(key, packed, bucket="w4"),
        "layout": "expert-major [E, ceil(K/2), N] packed + [E, groups, N] fp32 scale",
    }
    writer.add(key + "#scale", scale, bucket="scale")
    if source_experts is None:
        entry["replicated"] = True
    else:
        entry["source_experts"] = list(source_experts)
    entries[key] = entry


def _emit_dense(writer: ShardWriter, key: str, dense: torch.Tensor,
                entries: dict[str, dict], *,
                built: torch.Tensor | None = None) -> torch.Tensor:
    """把一个保持高精度的张量写进分片。

    Args:
        writer: 目标目录的写出端。
        key: checkpoint 参数名。
        dense: CPU 上的张量，dtype 已经是目标 dtype。
        entries: 这个 rank 的索引 ``tensors`` 字典，就地写入。
        built: 已经规整过的连续张量，供多个目标目录复用；``None`` 时现场 ``contiguous``。

    Returns:
        写下去的那个张量，供下一个目标目录复用。
    """
    value = built if built is not None else dense.contiguous()
    entries[key] = {
        "kind": "dense",
        "shape": list(value.shape),
        "dtype": _dtype_name(value.dtype),
        "shard": writer.add(key, value, bucket="dense"),
    }
    return value


def _discover_shards(checkpoint: Path, only: str | None, limit: int) -> list[Path]:
    """列出要处理的源分片。

    Args:
        checkpoint: 源 checkpoint 目录。
        only: 可选的 glob 过滤，用于只跑已经到盘的分片。
        limit: 最多处理多少个；``0`` 表示全部。

    Returns:
        按文件名排序的分片列表。

    Raises:
        SystemExit: 一个分片都没有。空输入会写出一份空索引，它读回去是一个零权重模型，
            比直接失败更难查。
    """
    shards = sorted(checkpoint.glob(only or "*.safetensors"))
    if limit:
        shards = shards[:limit]
    if not shards:
        raise SystemExit(f"no shard under {checkpoint} matched {only or '*.safetensors'}")
    return shards


def _copy_sidecars(checkpoint: Path, out: Path) -> list[str]:
    """把 checkpoint 里的小文件（config.json / 分词器）复制到输出目录。

    ``quantized_load._check_source_config`` 用 ``config.json`` 的 sha256 核对权重，所以
    这份文件必须与索引里记的摘要来自同一个 checkpoint。

    注意 ``model.safetensors.index.json`` 不复制：它是**原始 HF 分片索引**，会把读它的
    人引到一批不存在的源分片上，而本工具产出的是 ``quant_index.json``。

    Args:
        checkpoint: 源目录，只读。
        out: 目标目录。

    Returns:
        复制过的文件名。
    """
    copied: list[str] = []
    for path in sorted(checkpoint.iterdir()):
        if not path.is_file():
            continue
        if path.name.endswith((".safetensors", ".safetensors.index.json", ".bin", ".pt")):
            continue
        if path.name in {INDEX_NAME, NGRAM_INDEX_NAME, PROGRESS_NAME}:
            continue
        shutil.copyfile(path, out / path.name)
        copied.append(path.name)
    return copied


def _dir_bytes(path: Path) -> int:
    """目录里所有普通文件的总字节数。

    Args:
        path: 要统计的目录。

    Returns:
        字节数。实测占用用它报，而不是拿索引里算出来的理论值：两者应当相等，一旦不等
        就说明有分片没写全。
    """
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _read_progress(out: Path) -> dict | None:
    """读续跑进度。

    Args:
        out: 输出根目录。

    Returns:
        进度字典；没有进度文件时返回 ``None``（全新一轮转换）。
    """
    path = out / PROGRESS_NAME
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_progress(out: Path, progress: dict) -> None:
    """原子地写续跑进度。

    Args:
        out: 输出根目录。
        progress: 要写的字典。先写临时文件再 ``os.replace``，这样进程在写一半时被杀
            也不会留下半个 JSON——而半个 JSON 会让下一次续跑直接解析失败。
    """
    path = out / PROGRESS_NAME
    tmp = out / (PROGRESS_NAME + ".tmp")
    tmp.write_text(json.dumps(progress, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _resume_state(out: Path, layouts: tuple[str, ...], group_size: int,
                  weight_dtype: torch.dtype, checkpoint: Path) -> tuple[dict, dict]:
    """读回上一次的现场，供续跑从断点接上。

    Args:
        out: 输出根目录。
        layouts: 本次要写出的布局。
        group_size: 分组大小。
        weight_dtype: 量化前的 dtype。
        checkpoint: 源 checkpoint。

    Returns:
        ``(indexes, writer_offsets)``。``indexes`` 是各 ``(layout, rank)`` 与 ``"ngram"``
        已有的张量条目；``writer_offsets`` 是各目录下一个可用的分片序号。

    Raises:
        SystemExit: 上次的 group_size / dtype / 源目录 / 布局与本次不同。混着写会让同一
            份索引里出现两套量化参数，而那种权重既不是 A 也不是 B，反量化出来只是
            "看起来像权重"的数字。
    """
    old = _read_progress(out)
    if old is None:
        return {}, {}
    for field, mine, theirs in (
            ("group_size", group_size, old.get("group_size")),
            ("weight_dtype", _dtype_name(weight_dtype), old.get("weight_dtype")),
            ("source", str(checkpoint.resolve()), old.get("source"))):
        if mine != theirs:
            raise SystemExit(f"{out}: previous run used {field}={theirs}, this one asks "
                             f"for {mine}; use a fresh directory")
    if set(old.get("layouts") or []) != set(layouts):
        raise SystemExit(f"{out}: previous run wrote layouts {old.get('layouts')}, this "
                         f"one asks for {list(layouts)}; layouts share the output tree")

    indexes: dict = {}
    offsets: dict = {}
    for layout in layouts:
        for rank in range(_WORLD):
            dest = out / f"rank-{layout}-{rank}"
            index_path = dest / INDEX_NAME
            entries = (json.loads(index_path.read_text(encoding="utf-8"))["tensors"]
                       if index_path.is_file() else {})
            # 只保留索引里点过名的分片。上一轮如果在某个源分片中途崩掉，写出端可能已经
            # flush 出几个分片文件，但它们没进索引——那些文件里是一份"半截"的贡献，
            # 留着会让加载端读到同一张量的两份拷贝中的错的那一份。清掉，让这个源分片
            # 整片重算。
            listed = set((json.loads(index_path.read_text(encoding="utf-8")).get("files") or {})
                         if index_path.is_file() else {})
            for stale in dest.glob("*.safetensors"):
                if stale.name not in listed:
                    stale.unlink()
            indexes[(layout, rank)] = entries
            offsets[(layout, rank)] = len(listed)
    ngram_dir = out / "ngram"
    ngram_index = ngram_dir / NGRAM_INDEX_NAME
    if ngram_index.is_file():
        doc = json.loads(ngram_index.read_text(encoding="utf-8"))
        indexes["ngram"] = doc["tensors"]
        listed = set(doc.get("files") or {})
    else:
        indexes["ngram"], listed = {}, set()
    for stale in ngram_dir.glob("*.safetensors"):
        if stale.name not in listed:
            stale.unlink()
    offsets["ngram"] = len(listed)
    return indexes, offsets


def _done_shards(inventory: dict, seen: set[str], shards: list[Path]) -> set[str]:
    """从磁盘上的索引反推哪些源分片已经完整处理过。

    刻意不用进度文件里的 ``done_shards`` 列表做判据。那个文件和索引是两次独立的写，
    进程可能在两者之间被杀：那时索引里有这个分片的全部张量、进度文件里没有，按进度
    文件判断就会把整片重算一遍——而重算会在索引里撞上"张量已存在"，直接报错退出，
    于是续跑永远起不来。反过来按索引判断是幂等的：索引就是**唯一**的事实来源。

    Args:
        inventory: ``inventory.json`` 的 ``tensors`` 层，用来知道每个张量属于哪个源分片。
        seen: 已经在索引里出现过的张量名集合。
        shards: 本次要处理的源分片。

    Returns:
        已完整处理的源分片文件名集合。
    """
    by_shard: dict[str, set[str]] = {}
    for key, meta in inventory.items():
        by_shard.setdefault(meta["shard"], set()).add(key)
    return {path.name for path in shards
            if by_shard.get(path.name) and by_shard[path.name] <= seen}


def convert(checkpoint: Path, out: Path, *, inventory: dict, layouts: tuple[str, ...],
            group_size: int, weight_dtype: torch.dtype, shards: list[Path],
            complete: bool, resume: bool = True) -> dict:
    """把一个 checkpoint 目录转成分 rank 的 W4A8 目录树，支持断点续跑。

    Args:
        checkpoint: 源 checkpoint，只读。
        out: 输出根目录，里面会建 ``ngram/``、``rank-<layout>-<n>/`` 与顶层清单。
        inventory: ``inventory.json`` 的 ``tensors`` 层，逐张量形状的权威来源。
        layouts: 要写出的分片布局。
        group_size: K 轴分组大小。
        weight_dtype: 量化前把权重窄到的 dtype，必须与运行时一致。
        shards: 要处理的源分片。
        complete: 源分片是否齐全。不齐全时索引记 ``complete: false``。
        resume: 允许从 ``quant_progress.json`` 接着跑。

    Returns:
        写进顶层清单的那个字典。

    Raises:
        SystemExit: checkpoint 里某个张量的形状与 inventory 不符。那说明这份 inventory
            属于另一个 release，而按错的形状量化出来的权重每一层都会错位。
        RuntimeError: 分片里出现了上一次已处理过的张量。那说明源分片集合变了（例如
            重新下载后文件名不同），继续写会让索引里出现两张同名但数据不同的条目。
    """
    from safetensors import safe_open

    out.mkdir(parents=True, exist_ok=True)
    (out / "ngram").mkdir(exist_ok=True)
    rank_dirs: dict[tuple[str, int], Path] = {}
    for layout in layouts:
        for rank in range(_WORLD):
            dest = out / f"rank-{layout}-{rank}"
            dest.mkdir(exist_ok=True)
            rank_dirs[(layout, rank)] = dest

    done_index, offsets = (_resume_state(out, layouts, group_size, weight_dtype, checkpoint)
                           if resume else ({}, {}))
    indexes = dict(done_index)
    for layout in layouts:
        for rank in range(_WORLD):
            indexes.setdefault((layout, rank), {})
            offsets.setdefault((layout, rank), 0)
    indexes.setdefault("ngram", {})
    offsets.setdefault("ngram", 0)

    writers = {key: ShardWriter(dest, layout=key[0], rank=key[1],
                               start_index=offsets.get(key, 0))
               for key, dest in rank_dirs.items()}
    ngram_writer = ShardWriter(out / "ngram", layout="ngram", rank=0,
                              start_index=offsets.get("ngram", 0))

    seen: set[str] = set()
    for entries in indexes.values():
        seen.update(entries)
    done = _done_shards(inventory, seen, shards)
    started = time.time()
    resumed = len(done)

    for index, source in enumerate(shards, 1):
        if source.name in done:
            if index % 10 == 0 or index == len(shards):
                print(f"  [{index}/{len(shards)}] {source.name} 跳过（已完成）  "
                      f"{time.time() - started:.0f}s", flush=True)
            continue
        with safe_open(str(source), framework="pt", device="cpu") as handle:
            for key in sorted(handle.keys()):
                if key in seen:
                    raise RuntimeError(
                        f"{key}: already written by an earlier run; the source shard set "
                        f"changed under the same output directory")
                shape = _inventory_shape(inventory, key)
                kind = classify(key, shape)
                seen.add(key)

                if kind == "ngram":
                    # 同样必须分段读：一张表 2500012 × 160 的 bf16 是 800 MB，加上
                    # ``quantize_ngram`` 里转 fp32 的那份就是 1.6 GB，再叠上 packed 输出，
                    # 超过登录节点约 2 GB 的 RSS 上限。按行切之后峰值只看 _NGRAM_CHUNK_ROWS。
                    rows, dim = int(shape[0]), int(shape[1])
                    codes, scale = quantize_ngram_streaming(
                        handle.get_slice(key), rows, dim)
                    indexes["ngram"][key] = {
                        "kind": "int8-host",
                        "host_resident": True,
                        "shape": list(shape),
                        "rows": rows,
                        "dim": dim,
                        "scale_key": key + "#scale",
                        "shard": ngram_writer.add(key, codes, bucket="int8"),
                        "dequant": "row.astype(fp32) * scale[row], symmetric int8, no offset",
                    }
                    ngram_writer.add(key + "#scale", scale, bucket="scale")
                    del codes, scale
                    continue

                if kind == "moe":
                    # MoE 走"按需读、写一点丢一点"，**不**保留整张量化结果。512 个专家的
                    # packed + scale 是约 1.0 GB，看着不大，但它要叠在读块、fp32 中间量和
                    # 写出端 payload 上，而登录节点的上限只有约 2 GB（实测被 SIGKILL）。
                    #
                    # 做法：沿 expert 维按 _MOE_CHUNK_EXPERTS 读一块、量化一块，这一块
                    # 完全落进某一个 rank（EP 下每 rank 是 512/4=128 个专家，8 整除 128），
                    # 于是量化完立刻交给该 rank 的写出端、立刻释放。
                    experts_total = int(shape[0])
                    per_rank = experts_total // _WORLD
                    if per_rank * _WORLD != experts_total:
                        raise SystemExit(
                            f"{key}: {experts_total} experts is not divisible by "
                            f"EP={_WORLD}")
                    if per_rank % _MOE_CHUNK_EXPERTS:
                        raise SystemExit(
                            f"{key}: {per_rank} experts per rank is not divisible by the "
                            f"MoE chunk size {_MOE_CHUNK_EXPERTS}; a read chunk would "
                            f"straddle two rank directories")
                    _emit_moe_streaming(handle.get_slice(key), key, shape, layouts,
                                        rank_dirs, writers, indexes, group_size)
                    continue

                # 同一份稠密切片会被多个目标目录同时需要：EP=4 下除 MoE 外每个张量在
                # 4 个 rank 上完全相同，两种布局又要各写一遍，最多 8 个目标。量化一次、
                # 把同一批 CPU 张量交给 8 个写出端，省掉 7/8 的算术。
                #
                # 切片也**按需读**：``embed_tokens`` 是 1.27 GB，整份读进来就超过登录
                # 节点约 2 GB 的 RSS 上限（它只有 131 个分片里的两三个，但正是最容易被
                # OOM 打断的那几个）。先算出各目标要的切片，再把同一批切片合并成一次
                # ``get_slice`` 读，读到的量只与切片大小有关。
                plan: dict[tuple, list[tuple[str, int]]] = {}
                for layout in layouts:
                    for rank in range(_WORLD):
                        slices, segments = _rank_slices(key, shape, layout, rank)
                        signature = (tuple(slices), tuple(segments or ()))
                        plan.setdefault(signature, []).append((layout, rank))
                if not plan:
                    plan[((), ())] = []
                slicer = handle.get_slice(key) if plan else None
                for (slices, segments_sig), targets in plan.items():
                    slice_list = [tuple(s) for s in slices]
                    piece = _read_slice(slicer, slice_list, shape).to(weight_dtype)
                    segments = [tuple(s) for s in segments_sig] or None
                    built: tuple | None = None
                    for layout, rank in targets:
                        writer = writers[(layout, rank)]
                        entries = indexes[(layout, rank)]
                        if kind == "q4":
                            built = _emit_q4(writer, key, piece, group_size, entries,
                                             full_shape=shape, segments=segments,
                                             built=built)
                        else:
                            built = _emit_dense(writer, key, piece, entries, built=built)
                    del piece

        # 每处理完一个源分片就落盘 + 记进度：WebShell 会断，360 GB 的活必须能接着跑。
        # 落盘单位是源分片而不是"攒够 3 GiB"，后者会让崩溃后的续跑不知道缓冲区里
        # 那些没写盘的张量算不算数。
        #
        # 索引也必须跟着写：续跑读的是索引里的 ``tensors``，只写分片不写索引会让下一次
        # 启动以为那些分片里的张量从未出现过，接着把它们再写一遍（索引条目重复、盘上
        # 出现两份同名张量）。索引很小（131 个分片 × 8 份 × 几十 KB），每片写一次的代价
        # 远低于让续跑从头再来。
        for writer in writers.values():
            writer.flush()
        ngram_writer.flush()
        done.add(source.name)
        _write_all_indexes(out, layouts, rank_dirs, indexes, writers, ngram_writer,
                           weight_dtype=weight_dtype, group_size=group_size,
                           checkpoint=checkpoint, complete=complete)
        _write_progress(out, {
            "source": str(checkpoint.resolve()),
            "weight_dtype": _dtype_name(weight_dtype),
            "group_size": group_size,
            "layouts": list(layouts),
            "done_shards": sorted(done),
            "tensors": len(seen),
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        print(f"  [{index}/{len(shards)}] {source.name}  tensors={len(seen)}  "
              f"{time.time() - started:.0f}s", flush=True)

    for writer in writers.values():
        writer.close()
    ngram_writer.close()

    missing = sorted(set(inventory) - seen)
    if missing and complete:
        raise SystemExit(f"{len(missing)} tensors in the inventory are missing from the "
                         f"checkpoint, e.g. {missing[0]}")

    # 计数从索引重算而不是沿用本轮累加器：续跑时本轮只看到一部分源分片，沿用累加器
    # 会让清单里的 kinds / modules 只覆盖后半段，而那正是排查"哪类张量被漏掉"要看的数。
    kinds, modules = _count_kinds(indexes)

    layout_reports: dict[str, dict] = {}
    for layout in layouts:
        per_rank = {}
        for rank in range(_WORLD):
            dest = (layout, rank)
            files = _merge_files(writers[dest].files, rank_dirs[dest], INDEX_NAME)
            _write_rank_index(rank_dirs[dest], layout=layout, rank=rank,
                              entries=indexes[dest], files=files,
                              weight_dtype=weight_dtype, group_size=group_size,
                              checkpoint=checkpoint, complete=complete)
            per_rank[str(rank)] = {
                "dir": rank_dirs[dest].name,
                "tensors": len(indexes[dest]),
                "files": len(files),
                "bytes": _dir_bytes(rank_dirs[dest]),
                "totals": _totals_of(indexes[dest]),
            }
        layout_reports[layout] = per_rank
    _write_ngram_index(out / "ngram", indexes["ngram"], ngram_writer)

    report = {
        "format": MANIFEST_FORMAT,
        "generated_by": "coffin/tools/quant_disk_qwen4.py",
        "source": str(checkpoint.resolve()),
        "source_config_sha256": _config_digest(checkpoint),
        "weight_dtype": _dtype_name(weight_dtype),
        "group_size": group_size,
        "activation_format": "int8 per token, runtime kernel required",
        "scheme": "symmetric per-output-channel K-group absmax; q=round(w/(maxabs/7)), "
                  "clamped to [-8,7], nibble code=q+8",
        "layout": "uint8 [ceil(K/2), N] 低 nibble 偶数 K; fp32 [ceil(K/group_size), N]; "
                  "MoE 逐 expert 堆成 [E, ceil(K/2), N] / [E, groups, N]",
        "world": _WORLD,
        "complete": bool(complete),
        "shards_done": len(done),
        "shards_total": len(shards),
        "missing_count": len(missing),
        "missing_tensors": missing[:16],
        "rank_dirs": {layout: {rank: info["dir"] for rank, info in per.items()}
                      for layout, per in layout_reports.items()},
        "layouts": layout_reports,
        "ngram": {"tensors": len(indexes["ngram"]),
                  "files": len(list((out / "ngram").glob("*.safetensors"))),
                  "bytes": _dir_bytes(out / "ngram")},
        "kinds": dict(sorted(kinds.items())),
        "modules": {name: dict(counter) for name, counter in sorted(modules.items())},
        "rank_index_hint": "quant_index.json 只存在于 rank-*/ 与 ngram/ 里；顶层这份是"
                           "清单，不是可加载目录",
        "sidecars": _copy_sidecars(checkpoint, out),
    }
    (out / INDEX_NAME).write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    _write_progress(out, {
        "source": str(checkpoint.resolve()),
        "weight_dtype": _dtype_name(weight_dtype),
        "group_size": group_size,
        "layouts": list(layouts),
        "done_shards": sorted(done),
        "tensors": len(seen),
        "finished": True,
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    if resumed:
        print(f"  resumed: {resumed} 个源分片上一轮已完成，本次跳过", flush=True)
    return report


def _count_kinds(indexes: dict) -> tuple[Counter, dict[str, Counter]]:
    """从索引条目重算逐类与逐模块的计数。

    Args:
        indexes: ``{"ngram": {...}, (layout, rank): {...}}`` 的条目字典集合。

    Returns:
        ``(kinds, modules)``。``kinds`` 按落盘形态计数，``modules`` 按模块组计数并带上
        该组的字节账。从索引重算而不是在循环里累加，是为了让续跑后的清单覆盖全部
        张量而不是只有本轮处理的那一批。
    """
    kinds: Counter = Counter()
    modules: dict[str, Counter] = {}
    for entries in indexes.values():
        for key, entry in entries.items():
            kind = entry["kind"]
            kinds[kind] += 1
            bucket = modules.setdefault(group_of(key), Counter())
            bucket["tensors"] += 1
            if kind in {"q4", "q4-moe"}:
                bucket["w4_bytes"] += _elem(entry["packed_shape"])
                bucket["scale_bytes"] += _elem(entry["scale_shape"]) * 4
            elif kind == "dense":
                bucket["dense_bytes"] += _elem(entry["shape"]) * _bytes_per_element(
                    entry["dtype"])
            elif kind == "int8-host":
                bucket["host_int8_bytes"] += entry["rows"] * entry["dim"]
                bucket["host_scale_bytes"] += entry["rows"] * 4
    return kinds, {name: counter for name, counter in sorted(modules.items())}


def _write_all_indexes(out: Path, layouts: tuple[str, ...],
                       rank_dirs: dict, indexes: dict, writers: dict,
                       ngram_writer: ShardWriter, *, weight_dtype: torch.dtype,
                       group_size: int, checkpoint: Path, complete: bool) -> None:
    """把全部 rank 目录与 N-gram 目录的索引重写一遍。

    每个源分片处理完调一次，所以这里必须便宜：8 份 rank 索引各几十 KB、一份 ngram
    索引在收尾前是空字典。用整写而不是增量追加，是因为索引本身是一份 JSON 快照，
    增量改需要在磁盘上做部分重写，而那正是"半个 JSON"的来源。

    Args:
        out: 输出根目录。
        layouts: 本次写出的布局。
        rank_dirs: ``(layout, rank)`` 到目录的映射。
        indexes: ``(layout, rank)`` 与 ``"ngram"`` 到条目字典的映射。
        writers: ``(layout, rank)`` 到写出端的映射。
        ngram_writer: N-gram 的写出端。
        weight_dtype: 量化前 dtype。
        group_size: 分组大小。
        checkpoint: 源 checkpoint。
        complete: 源分片是否齐全。
    """
    for layout in layouts:
        for rank in range(_WORLD):
            dest = (layout, rank)
            files = _merge_files(writers[dest].files, rank_dirs[dest], INDEX_NAME)
            _write_rank_index(rank_dirs[dest], layout=layout, rank=rank,
                              entries=indexes[dest], files=files,
                              weight_dtype=weight_dtype, group_size=group_size,
                              checkpoint=checkpoint, complete=complete)
    _write_ngram_index(out / "ngram", indexes["ngram"], ngram_writer)


def _merge_files(this_run: dict, dest: Path, index_name: str) -> dict:
    """把本次写出的分片与上一次续跑留下的分片合起来。

    Args:
        this_run: 本次写出端记的 ``{name: stats}``。
        dest: rank 目录。
        index_name: 目录里索引的文件名。

    Returns:
        ``{分片名: 统计}``，包含目录里已有的全部分片。续跑时上一轮写出的分片也必须
        列进来，否则新索引会漏报旧分片，而加载器按索引找分片，漏报等于丢权重。
    """
    files = dict(this_run)
    existing = {p.name for p in dest.glob("*.safetensors")}
    index_path = dest / index_name
    old = {}
    if index_path.is_file():
        old = json.loads(index_path.read_text(encoding="utf-8")).get("files") or {}
    for name in sorted(existing):
        if name in files:
            continue
        if name in old:
            files[name] = old[name]
            continue
        files[name] = {"tensors": -1, "bytes": (dest / name).stat().st_size}
    return dict(sorted(files.items()))


def _totals_of(entries: dict[str, dict]) -> dict:
    """按索引条目算一个目录的字节账。

    Args:
        entries: ``tensors`` 字典。

    Returns:
        ``{"w4_bytes", "scale_bytes", "dense_bytes", "host_int8_bytes",
        "host_scale_bytes", "tensors"}``。它是理论值，与盘上实测对照即可看出有没有
        分片写漏。
    """
    totals = {"w4_bytes": 0, "scale_bytes": 0, "dense_bytes": 0,
              "host_int8_bytes": 0, "host_scale_bytes": 0, "tensors": len(entries)}
    for entry in entries.values():
        kind = entry["kind"]
        if kind in {"q4", "q4-moe"}:
            totals["w4_bytes"] += _elem(entry["packed_shape"])
            totals["scale_bytes"] += _elem(entry["scale_shape"]) * 4
        elif kind == "dense":
            totals["dense_bytes"] += _elem(entry["shape"]) * _bytes_per_element(
                entry["dtype"])
        elif kind == "int8-host":
            totals["host_int8_bytes"] += entry["rows"] * entry["dim"]
            totals["host_scale_bytes"] += entry["rows"] * 4
    return totals


def _write_rank_index(dest: Path, *, layout: str, rank: int, entries: dict[str, dict],
                      files: dict, weight_dtype: torch.dtype, group_size: int,
                      checkpoint: Path, complete: bool) -> None:
    """写一个 rank 目录自己的 ``quant_index.json``。

    这份索引的 schema 与 ``quant_disk_w4a8.py`` 写的完全一致——``quantized_load`` 认的
    就是它——额外多了 ``shard_layout``/``rank``/``world`` 以及每个张量的 ``full_shape``：
    加载端据此知道手里的 ``[N,K]`` 是整个张量的哪一片，不必去读顶层清单。

    Args:
        dest: rank 目录。
        layout: 布局名。
        rank: rank 编号。
        entries: 这个 rank 的张量条目。
        files: 目录里的分片统计。
        weight_dtype: 量化前 dtype。
        group_size: 分组大小。
        checkpoint: 源 checkpoint，用于算配置摘要。
        complete: 源分片是否齐全。
    """
    report = {
        "format": FORMAT,
        "generated_by": "coffin/tools/quant_disk_qwen4.py",
        "source": str(checkpoint.resolve()),
        "source_config_sha256": _config_digest(checkpoint),
        "weight_dtype": _dtype_name(weight_dtype),
        "group_size": group_size,
        "scheme": "symmetric per-output-channel K-group absmax; q=round(w/(maxabs/7)), "
                  "clamped to [-8,7], nibble code=q+8",
        "layout": "uint8 [ceil(K/2), N] 低 nibble 偶数 K; fp32 [ceil(K/group_size), N]",
        "vision": any(k.startswith(VISUAL_PREFIX) for k in entries),
        "shard_layout": layout,
        "rank": rank,
        "world": _WORLD,
        "complete": bool(complete),
        "files": files,
        "totals": _totals_of(entries),
        "tensors": entries,
    }
    (dest / INDEX_NAME).write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")


def _write_ngram_index(dest: Path, entries: dict[str, dict], writer: ShardWriter) -> None:
    """写 N-gram 表的独立索引。

    Args:
        dest: ``ngram/`` 目录。
        entries: 逐张量的条目。
        writer: 写出端，用于取分片统计。
    """
    rows = sum(e["rows"] for e in entries.values())
    report = {
        "format": "coffin-qwen4-ngram-host-v1",
        "generated_by": "coffin/tools/quant_disk_qwen4.py",
        "host_resident": True,
        "note": "51.2 B 参数的查表，按宿主内存常驻设计，不参与 GPU 权重的 W4A8 打包",
        "scheme": "per-row symmetric int8: s[r] = max|row|/127, code = round(row/s[r])",
        "widths": sorted({e["dim"] for e in entries.values()}),
        "files": writer.files,
        "totals": _totals_of(entries),
        "tensors": entries,
    }
    report["totals"]["rows"] = rows
    (dest / NGRAM_INDEX_NAME).write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")


def dry_run(inventory: dict, layouts: tuple[str, ...], group_size: int) -> dict:
    """只按 inventory 算字节账，不读也不写任何权重。

    Args:
        inventory: ``inventory.json`` 的 ``tensors`` 层。
        layouts: 要估算的布局。
        group_size: 分组大小。

    Returns:
        ``{"layouts": {layout: [逐 rank 的 Counter]}, "ngram": Counter}``。
    """
    per_layout: dict[str, list[Counter]] = {layout: [Counter() for _ in range(_WORLD)]
                                            for layout in layouts}
    host = Counter()
    for key, meta in inventory.items():
        shape = [int(x) for x in meta["shape"]]
        kind = classify(key, shape)
        if kind == "ngram":
            rows, dim = shape
            host["int8_bytes"] += rows * dim
            host["scale_bytes"] += rows * 4
            continue
        for layout in layouts:
            for rank in range(_WORLD):
                slices, _ = _rank_slices(key, shape, layout, rank)
                piece = list(shape)
                for dim, start, stop in slices:
                    piece[dim] = stop - start
                counter = per_layout[layout][rank]
                if kind == "moe":
                    experts, n, k = piece
                    packed, groups = _packed_shapes(n, k, group_size)
                    counter["w4_bytes"] += experts * packed * n
                    counter["scale_bytes"] += experts * groups * n * 4
                elif kind == "q4":
                    n, k = piece
                    packed, groups = _packed_shapes(n, k, group_size)
                    counter["w4_bytes"] += packed * n
                    counter["scale_bytes"] += groups * n * 4
                else:
                    counter["dense_bytes"] += _elem(piece) * 2
    return {"layouts": {layout: [dict(c) for c in ranks]
                        for layout, ranks in per_layout.items()},
            "ngram": dict(host)}


def main(argv: list[str] | None = None) -> int:
    """命令行入口：checkpoint 目录进，分 rank 的 W4A8 目录树出。

    Args:
        argv: 参数列表；``None`` 取 ``sys.argv[1:]``。

    Returns:
        ``0`` 表示转换完成。

    Raises:
        SystemExit: 源目录不存在、输出目录非空且不是续跑现场、或参数取值不合法。
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, help="源 checkpoint 目录（只读）")
    parser.add_argument("--out", required=True, help="输出根目录；空目录，或上一次的续跑现场")
    parser.add_argument("--inventory", required=True,
                        help="docs/qwen4/inventory.json，逐张量形状的权威来源")
    parser.add_argument("--layout", action="append", choices=_LAYOUTS, default=None,
                        help="要写出的分片布局，可重复；默认 ep4")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--weight-dtype", default="bf16", choices=sorted(_WEIGHT_DTYPES))
    parser.add_argument("--only-shards", default=None,
                        help="只处理匹配这个 glob 的源分片，用于先跑子集验证格式")
    parser.add_argument("--limit-shards", type=int, default=0,
                        help="只处理前 N 个源分片；0 表示全部")
    parser.add_argument("--allow-partial", action="store_true",
                        help="源分片不齐时仍然写出，并在索引里记 complete=false")
    parser.add_argument("--no-resume", action="store_true",
                        help="忽略已有进度文件，从头开始（输出目录必须为空）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只按 inventory 算字节账并打印，不读也不写任何权重")
    parser.add_argument("--threads", type=int, default=0,
                        help="torch 的 CPU 线程数；0 表示用 torch 的默认值")
    args = parser.parse_args(argv)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_dir():
        raise SystemExit(f"not a checkpoint directory: {checkpoint}")
    out = Path(args.out)
    if out.resolve() == checkpoint.resolve():
        raise SystemExit("--out must not be the checkpoint itself; it is read-only")
    inventory = json.loads(Path(args.inventory).read_text(encoding="utf-8"))["tensors"]
    if args.group_size <= 0:
        raise SystemExit("--group-size must be positive")
    layouts = tuple(args.layout) if args.layout else ("ep4",)
    weight_dtype = _WEIGHT_DTYPES[args.weight_dtype]
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    print(f"  source  : {checkpoint}", flush=True)
    print(f"  out     : {out}", flush=True)
    print(f"  layouts : {', '.join(layouts)}", flush=True)
    print(f"  dtype   : {_dtype_name(weight_dtype)}  group_size={args.group_size}  "
          f"threads={torch.get_num_threads()}", flush=True)

    if args.dry_run:
        report = dry_run(inventory, layouts, args.group_size)
        print("QUANT_DRY_RUN_JSON_BEGIN", flush=True)
        print(json.dumps(report, indent=1, ensure_ascii=False), flush=True)
        print("QUANT_DRY_RUN_JSON_END", flush=True)
        return 0

    progress = None if args.no_resume else _read_progress(out)
    if out.exists() and any(out.iterdir()) and progress is None:
        raise SystemExit(f"output directory must be empty (no {PROGRESS_NAME} to resume "
                         f"from): {out}")
    shards = _discover_shards(checkpoint, args.only_shards, args.limit_shards)
    print(f"  shards  : {len(shards)}  ({shards[0].name} .. {shards[-1].name})", flush=True)
    if progress:
        print(f"  resume  : {len(progress.get('done_shards') or [])} 个源分片已完成",
              flush=True)
    started = time.time()
    report = convert(checkpoint, out, inventory=inventory, layouts=layouts,
                     group_size=args.group_size, weight_dtype=weight_dtype,
                     shards=shards, complete=not args.allow_partial)
    elapsed = time.time() - started
    written = _dir_bytes(out)
    summary = {
        "seconds": round(elapsed, 1),
        "written_bytes": written,
        "written_GB": round(written / 1e9, 3),
        "complete": report["complete"],
        "shards_done": report["shards_done"],
        "shards_total": report["shards_total"],
        "missing_count": report["missing_count"],
        "kinds": report["kinds"],
        "layouts": {layout: {rank: info["bytes"] for rank, info in per.items()}
                    for layout, per in report["layouts"].items()},
        "rank_totals": {layout: {rank: info["totals"] for rank, info in per.items()}
                        for layout, per in report["layouts"].items()},
        "ngram_bytes": report["ngram"]["bytes"],
    }
    print(f"  wrote {written / 1e9:.2f} GB in {elapsed:.1f}s", flush=True)
    print("QUANT_DISK_JSON_BEGIN", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print("QUANT_DISK_JSON_END", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
