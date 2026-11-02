"""把 checkpoint 的张量装进 qwen4exp 的模型，并按 EP 切分专家。

这一层不碰硬件、也不碰 ``safetensors``：它只认一个张量源
源可以是内存字典（测试用）、safetensors 分片目录（集群用），或者 W4A8 量化产物。
把"从哪读"和"怎么装"分开，是唯一能在纯 CPU 开发机上验证加载路径的办法——真实
权重是 360 GB，本机连一个分片都放不下。

装的过程中做三件事，每一件错了都不会当场报错、只会在前向里变成垃圾：

1. **名字映射**：checkpoint 的文本塔在 ``model.language_model.`` 下（多模态包装），
   而引擎的 ``CausalLM.model`` 是文本塔、``lm_head`` 在顶层。映射是纯字符串操作，
   但漏掉一条前缀就会让某个张量悄悄保留随机初始化值——所以落不到的键和没被喂
   的槽位都必须报出来。
2. **专家切片**：``gate_up_proj`` / ``down_proj`` 是 3D 融合张量，EP 分片沿第 0 维
   切。切错轴形状仍然对（``[128, 1280, 2560]`` 对 ``[512, 2560, 640]`` 的第 0 维都
   切得动），所以这里按**参数名**决定切哪一维，不看形状。
3. **n-gram 分片拼接**：checkpoint 把那张 320001536 行的表切成 128 个
   ``shard_k.weight``，必须按 ``k`` 的顺序拼成一张。拼错顺序不会报错——哈希查到的
   是另一行，检索质量下降而已。

Module: ``coffin/models/qwen4exp``。族内的 I/O 边界：它 import ``config`` 和
``model``，但反过来不被它们 import。不 import ``transformers``。

Constraints:
    * 加载完必须**全量校验**：模型要的每个张量都拿到、拿到的每个张量都有归属、
      形状逐一相等。缺一条都会让某个参数停在 ``nn.Parameter`` 的初始值上，而
      ``normal_(0.02)`` 的随机值不会让任何数值变成 NaN——只会让答案错。
    * 大表（n-gram 95 GiB）不加载，只校验几何。见 ``stream_ngram``。
    * 这个模块不允许出现设备假设：源返回什么就装什么，转 dtype 与转设备由调用方
      通过 ``prepare`` 决定。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import torch

from models.qwen4exp.config import TextConfig

#: 多模态 checkpoint 里文本塔的前缀（``Qwen4ExpForConditionalGeneration`` 的布局）。
TEXT_PREFIXES = ("model.language_model.", "model.")

#: 引擎里文本塔的前缀：``CausalLM.model`` 是 TextModel。
ENGINE_PREFIX = "model."

#: 3D 融合专家张量：checkpoint 里第 0 维就是专家号，EP 分片沿它切。
EXPERT_TENSORS = (".mlp.experts.gate_up_proj", ".mlp.experts.down_proj")

#: n-gram 表在 checkpoint 里被切成这个数量的分片（``split_ngram_parts`` 的默认值）。
#: 分片名是 ``...ngram_embedding.shard_{k}.weight``，必须按 ``k`` 拼。
NGRAM_SHARD_MARKER = ".ngram_embedding.shard_"


class TensorSource(Protocol):
    """一个能按 checkpoint 名取张量的只读源。

    实现者负责"从哪读"（内存、safetensors、量化产物），本模块负责"怎么装"。
    """

    def keys(self) -> Iterable[str]:
        """列出源里所有可用的张量名（checkpoint 布局，带 ``model.`` 前缀）。"""

    def get(self, name: str) -> torch.Tensor:
        """取出一个张量。

        Args:
            name: checkpoint 里的完整名字。

        Returns:
            该张量。dtype 与设备由实现者决定。
        """


class DictSource:
    """内存字典源：测试与 fixture 用。

    它是 ``decoder_layer.pt`` 那 34 个张量能直接喂进模型的原因——单层 fixture 的
    ``weights`` 键是**层内相对名**（``linear_attn.in_proj_qkv.weight``），本类负责
    加上前缀，让它与 checkpoint 布局一致。
    """

    def __init__(self, tensors: dict[str, torch.Tensor], prefix: str = "") -> None:
        """包住一个字典。

        Args:
            tensors: 名字到张量的映射。
            prefix: 给每个名字加的前缀；单层 fixture 用 ``"model.layers.1."``。
        """
        self._tensors = {prefix + name: tensor for name, tensor in tensors.items()}

    def keys(self) -> Iterable[str]:
        """返回所有名字。"""
        return self._tensors.keys()

    def get(self, name: str) -> torch.Tensor:
        """按名取张量。

        Args:
            name: 加了前缀的完整名字。

        Returns:
            该张量。

        Raises:
            KeyError: 名字不在源里。
        """
        return self._tensors[name]


def engine_key(checkpoint_key: str) -> str | None:
    """把 checkpoint 的键名映射成引擎模型的键名。

    ``model.language_model.layers.0...`` -> ``model.layers.0...``；
    ``lm_head.weight`` 原样保留（它在 ``CausalLM`` 顶层）。
    ``mtp.*`` 与 ``model.visual.*`` 不属于文本塔，返回 ``None``。

    前缀按**最长匹配**去掉：``model.language_model.`` 必须在 ``model.`` 之前试，
    否则会映射成 ``language_model.layers...``。

    Args:
        checkpoint_key: checkpoint 里的完整键名。

    Returns:
        引擎侧键名，或 ``None`` 表示这个张量不由文本塔持有。

    Raises:
        ValueError: 键名带 n-gram 分片后缀。分片必须在 :func:`plan` 里先按几何
            合并成一张表，单条拼接没有意义。
    """
    if NGRAM_SHARD_MARKER in checkpoint_key:
        raise ValueError(
            f"{checkpoint_key} is an n-gram shard; aggregate the shards before mapping.")
    if checkpoint_key.startswith("mtp.") or checkpoint_key.startswith("model.visual."):
        return None
    if checkpoint_key.startswith("model.language_model."):
        return ENGINE_PREFIX + checkpoint_key[len("model.language_model."):]
    if checkpoint_key.startswith("model."):
        return checkpoint_key + ""
    return checkpoint_key


@dataclass
class LoadPlan:
    """一次加载的完整账目：装什么、跳过什么、切几片。

    Attributes:
        assignments: 引擎参数名 -> checkpoint 名（或 n-gram 聚合键）。
        experts: 引擎参数名 -> 切片的全局专家号列表；``None`` 表示不切。
        skipped: 源里有、但文本塔不认的键（``mtp.*``、视觉塔……），只记录。
        ngram: 引擎参数名 -> 分片组，每项是 ``(split 后张量名列表, 行数)``。
    """

    assignments: dict[str, str] = field(default_factory=dict)
    experts: dict[str, Sequence[int] | None] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
    ngram: dict[str, list[tuple[list[str], int]]] = field(default_factory=dict)

    def total(self) -> int:
        """这次计划要装的张量数。"""
        return len(self.assignments) + len(self.ngram)


def _ngram_groups(keys: Iterable[str]) -> dict[str, list[tuple[str, int]]]:
    """把 ``...ngram_embedding.shard_{k}.weight`` 按前缀分组并按 k 排序。

    Args:
        keys: 源里所有键名。

    Returns:
        ``{前缀: [(shard_key, k), ...]}``，每组内按 ``k`` 升序。
    """
    groups: dict[str, list[tuple[str, int]]] = {}
    for key in keys:
        index = key.find(NGRAM_SHARD_MARKER)
        if index < 0:
            continue
        stem = key[:index]
        rest = key[index + len(NGRAM_SHARD_MARKER):]
        digits = rest.split(".", 1)[0]
        if not digits.isdigit():
            continue
        groups.setdefault(stem, []).append((key, int(digits)))
    for items in groups.values():
        items.sort(key=lambda item: item[1])
    return groups


def plan(source: TensorSource, config: TextConfig,
         local_experts: Sequence[int] | None = None,
         with_lm_head: bool = True,
         subtree: str | None = None,
         max_layers: int | None = None,
         skip_experts: bool = False) -> LoadPlan:
    """算出每个引擎参数该从源里取哪个张量。

    纯规划，不读张量内容——所以本机（没有 safetensors、放不下真实权重）也能验证
    映射与切片规则对不对。

    Args:
        source: 张量源。
        config: 文本配置。
        local_experts: EP 分片时本 rank 持有的全局专家号；``None`` 表示全量。
        with_lm_head: 是否装 ``lm_head``。tied 权重的 checkpoint 里没有这个张量。
        subtree: 只规划这个前缀下的参数，比如 ``"model.layers.1."``。装单层或只装
            末端 mixer 时必须给，否则会拿全模型的账目去核对单层的 ``state_dict``。
        max_layers: 只规划前 N 层。裁层跑通时（比如先装 4 层验装载与形状）源里
            仍有 48 层，不限制会让规划范围大过模型，校验报"计划张量不是模型参数"。
        skip_experts: 跳过专家栈。量化路径下模型里的专家是 ``W4A8MoEWeight`` 而不
            是 ``nn.Parameter``，bf16 的那两个张量根本不存在，装它只会白读 60 GB。

    Returns:
        该装什么、切什么、跳过什么的账目。

    Raises:
        ValueError: 同一个引擎参数被两个 checkpoint 张量映射到。名字冲突说明前缀
            表有歧义，静默取后者会让某层悄悄用上别的层的权重。
    """
    result = LoadPlan()
    local = None if local_experts is None else list(local_experts)
    ngram_groups = _ngram_groups(source.keys())
    seen: dict[str, str] = {}

    def wanted(target: str) -> bool:
        """这个引擎参数是否在本次规划范围内。

        三条过滤按代价从低到高排：先按层号砍掉裁层剩下的层（省掉最多），再砍
        专家栈（量化路径下模型里没有这些参数），最后按 ``subtree`` 收窄到单层。
        顺序不影响结果，只影响每次调用少做几次字符串比较。

        Args:
            target: 引擎侧的参数名。

        Returns:
            是否要装它。
        """
        if max_layers is not None:
            # 层号从 ``model.layers.<n>.`` 里取；非层参数（嵌入、末端 mixer、
            # lm_head）总是要。
            marker = "model.layers."
            if marker in target:
                rest = target.split(marker, 1)[1]
                digits = rest.split(".", 1)[0]
                if digits.isdigit() and int(digits) >= max_layers:
                    return False
        if skip_experts and any(target.endswith(marker) for marker in EXPERT_TENSORS):
            return False
        if subtree is not None and not target.startswith(subtree):
            return False
        if not with_lm_head and target == "lm_head.weight":
            return False
        return True

    for key in source.keys():
        if NGRAM_SHARD_MARKER in key:
            continue
        target = engine_key(key)
        if target is None or not wanted(target):
            result.skipped.append(key)
            continue
        if target in seen:
            raise ValueError(
                f"both {seen[target]} and {key} map to engine parameter {target}")
        seen[target] = key
        result.assignments[target] = key
        sharded = any(target.endswith(marker) for marker in EXPERT_TENSORS)
        result.experts[target] = local if sharded else None

    for stem, shards in ngram_groups.items():
        target = engine_key(stem + ".weight")
        if target is None or not wanted(target):
            result.skipped.extend(item[0] for item in shards)
            continue
        result.ngram[target] = [(item[0], item[1]) for item in shards]

    return result


def verify(plan_result: LoadPlan, state_dict: dict[str, torch.Tensor],
           subtree: str | None = None, skip_experts: bool = False) -> list[str]:
    """核对计划覆盖了模型的每一个参数。

    流式模式下 n-gram 表不是模型参数（它由调用方按分片传入），所以账目里的
    n-gram 键不参与键名比较，只由 :func:`load_into` 用几何对账。

    Args:
        plan_result: :func:`plan` 的结果。
        state_dict: 目标模型的 ``state_dict``（可以用 meta 设备构造，零内存）。
        subtree: 只核对这个前缀下的参数；``None`` 表示整个模型。
        skip_experts: 不要求专家栈有源张量（量化路径）。

    Returns:
        问题描述列表；空列表表示全部对上。调用方把它当断言用。

    Raises:
        ValueError: 目标里出现了计划外的参数名（映射表写了一种不存在的张量）。
    """
    problems: list[str] = []
    planned = set(plan_result.assignments)
    expected = {key for key in state_dict
                if subtree is None or key.startswith(subtree)}
    # n-gram 表走 ``plan_result.ngram`` 单独一条路（由 128 个分片拼出来），所以不在
    # ``assignments`` 里；但它在模型里**仍然是参数**，于是会被算进 ``missing``。
    # 单层验收就是在这里报"模型参数没有源张量"的，而那个参数的源其实就在 ngram
    # 账目里。排除掉它，让两条路径各自核对：普通张量比键名，n-gram 比拼出的形状。
    stitched = set(plan_result.ngram)
    missing = sorted(expected - planned - stitched)
    # 量化路径下 ``skip_experts`` 让专家栈不进计划，而模型里那两个参数也被
    # ``set_quantized`` 删掉了，所以这里通常已经对得上。留着这条过滤是为了让
    # "先装载、后挂量化权重"的顺序也能通过——那时参数还在。
    if skip_experts:
        missing = [key for key in missing
                   if not any(key.endswith(marker) for marker in EXPERT_TENSORS)]
    if missing:
        problems.append(
            f"{len(missing)} model parameters have no source tensor, e.g. {missing[:5]}")
    extra = sorted(planned - set(state_dict))
    if extra:
        raise ValueError(
            f"{len(extra)} planned tensors are not model parameters, e.g. {extra[:5]}")
    return problems


def slice_experts(tensor: torch.Tensor, local_experts: Sequence[int] | None,
                  total_experts: int) -> torch.Tensor:
    """按全局专家号切出本 rank 的那几行。

    沿第 0 维切，因为 ``gate_up_proj`` / ``down_proj`` 的第 0 维是专家号。
    换一个轴切出来的形状往往仍然合法（三个维度都大于 128），这正是它危险的地方：
    切错轴不会报错，只会让每个专家用上别的专家的权重。

    Args:
        tensor: 3D 专家张量，``[num_experts, ...]``。
        local_experts: 要保留的全局专家号；``None`` 表示全量。
        total_experts: 全局专家总数，用来校验。

    Returns:
        切好的张量；``local_experts`` 为 ``None`` 时原样返回。

    Raises:
        ValueError: 张量第 0 维不是 ``total_experts``，或 ``local_experts`` 越界 /
            未排序（未排序时切出来的顺序与路由的全局号对不上）。
    """
    if local_experts is None:
        return tensor
    if tensor.shape[0] != total_experts:
        raise ValueError(
            f"expert tensor has {tensor.shape[0]} rows, config says {total_experts}")
    indices = list(local_experts)
    if indices != sorted(indices):
        raise ValueError("local_experts must be ascending: routing maps global ids "
                         "to this order positionally")
    if indices[0] < 0 or indices[-1] >= total_experts:
        raise ValueError(f"local_experts out of range: {indices[:3]}...")
    # 索引必须跟着张量走。``StreamingSource(device=...)`` 读出来的张量在卡上，
    # 而 ``torch.tensor(indices)`` 默认落在 CPU，``index_select`` 会报"张量在两个
    # 设备上"——这一步就是 GPU 装载时最先撞到的那个错。
    index = torch.tensor(indices, dtype=torch.long, device=tensor.device)
    return tensor.index_select(0, index)


def load_into(model: torch.nn.Module, source: TensorSource, config: TextConfig,
              local_experts: Sequence[int] | None = None,
              prepare: callable | None = None,
              with_lm_head: bool = True,
              subtree: str | None = None,
              max_layers: int | None = None,
              skip_experts: bool = False) -> dict:
    """把源里的张量装进模型，并返回加载账目。

    Args:
        model: 目标模型（``CausalLM`` 或单层 ``DecoderLayer`` 的有前缀版本）。
        source: 张量源。
        config: 文本配置。
        local_experts: EP 分片时本 rank 持有的全局专家号。
        prepare: 每个张量装载前的变换，``(name, tensor) -> tensor``，用来统一
            dtype / 设备；``None`` 表示原样装。
        with_lm_head: 是否装 ``lm_head``。
        subtree: 只装这个前缀下的参数，比如 ``"model.layers.1."``。装单层时必须
            给，否则会拿全模型的账目去核对单层的 ``state_dict``。
        max_layers: 只装前 N 层；裁层跑通时用，见 :func:`plan`。
        skip_experts: 不装专家栈；量化路径下见 :func:`plan`。

    Returns:
        一个字典，含 ``loaded`` / ``skipped`` / ``ngram`` 三个计数，便于在日志里
        核对"装了多少个"。张量本身不进返回值，它们已经进了模型。

    Raises:
        KeyError: 源里缺模型需要的张量。
        ValueError: 形状不符、映射冲突，或计划外的参数名。
    """
    plan_result = plan(source, config, local_experts, with_lm_head, subtree=subtree,
                       max_layers=max_layers, skip_experts=skip_experts)
    target_state = model.state_dict()
    problems = verify(plan_result, target_state, subtree=subtree,
                      skip_experts=skip_experts)
    if problems:
        raise KeyError("; ".join(problems))

    loaded = 0
    with torch.no_grad():
        for name, checkpoint_key in plan_result.assignments.items():
            tensor = source.get(checkpoint_key)
            # 先切专家再比形状：源里是**全局**的 3D 融合张量，模型里只有本 rank
            # 那几行，反过来比会把一份正确的分片判成形状不符。
            local = plan_result.experts.get(name)
            if local is not None:
                tensor = slice_experts(tensor, local, config.num_experts)
            expected = target_state[name].shape
            if tuple(tensor.shape) != tuple(expected):
                raise ValueError(
                    f"{name}: source shape {tuple(tensor.shape)} != model shape "
                    f"{tuple(expected)}")
            if prepare is not None:
                tensor = prepare(name, tensor)
            target_state[name].copy_(tensor.to(target_state[name].dtype))
            loaded += 1

        for name, shards in plan_result.ngram.items():
            # ``stream_ngram=True`` 时模型里**没有**这个参数（表由调用方按分片传入，
            # 102 GB 不该进显存也不该进 host 内存），而规划阶段仍然会在源里看到那
            # 128 个分片。不跳过就会一块块往卡上搬，直到 OOM——实测报在
            # ``Tried to allocate 764.00 MiB``，而真正的原因是这里根本不该读。
            if name not in target_state:
                continue
            table = _gather_ngram(source, shards)
            expected = target_state[name].shape
            if tuple(table.shape) != tuple(expected):
                raise ValueError(
                    f"{name}: stitched {tuple(table.shape)} != model {tuple(expected)}")
            if prepare is not None:
                table = prepare(name, table)
            target_state[name].copy_(table.to(target_state[name].dtype))
            loaded += 1

    return {"loaded": loaded, "skipped": len(plan_result.skipped),
            "ngram": len(plan_result.ngram), "experts": local_experts}


def _gather_ngram(source: TensorSource,
                  shards: list[tuple[str, int]]) -> torch.Tensor:
    """按 ``k`` 顺序把 n-gram 分片拼成一张表。

    Args:
        source: 张量源。
        shards: ``[(key, k), ...]``，已按 k 升序。

    Returns:
        拼好的 ``[total_rows, width]``。

    Raises:
        ValueError: 分片的 k 不连续从 0 开始（缺一片会静默错位整张表），或宽度
            不一致。
    """
    expected_index = 0
    parts: list[torch.Tensor] = []
    width = None
    for key, index in shards:
        if index != expected_index:
            raise ValueError(
                f"n-gram shards are not contiguous from 0: expected index "
                f"{expected_index}, got {index} ({key})")
        expected_index += 1
        part = source.get(key)
        if width is None:
            width = part.shape[1]
        elif part.shape[1] != width:
            raise ValueError(f"{key}: shard width {part.shape[1]} != {width}")
        parts.append(part)
    if not parts:
        raise ValueError("no n-gram shards to stitch")
    return torch.cat(parts, dim=0)


def read_config(model_dir: str | Path) -> dict:
    """读 checkpoint 目录里的 ``config.json``。

    Args:
        model_dir: 权重目录。

    Returns:
        解析后的顶层配置字典。
    """
    return json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))


def text_config_of(raw: dict) -> TextConfig:
    """从顶层配置里取出文本配置。

    Args:
        raw: :func:`read_config` 的返回值。

    Returns:
        校验完成的 ``TextConfig``。
    """
    return TextConfig.from_dict(raw.get("text_config", raw))


@dataclass
class LoadReport:
    """加载结果的摘要，给日志和测试断言用。

    Attributes:
        loaded: 装进去的参数量。
        skipped: 源里有但没装的键数（``mtp.*``、视觉塔）。
        ngram: 拼起来的 n-gram 表数量。
        experts: 本 rank 的专家号；``None`` 表示全量。
    """

    loaded: int
    skipped: int
    ngram: int
    experts: Sequence[int] | None = None

    def as_dict(self) -> dict:
        """转成可以直接 ``json.dumps`` 的字典。

        Returns:
            四个字段的普通字典。
        """
        return {"loaded": self.loaded, "skipped": self.skipped,
                "ngram": self.ngram,
                "experts": None if self.experts is None else list(self.experts)}


def iter_checkpoint_keys(model_dir: str | Path) -> Iterator[str]:
    """只读 header，列出 checkpoint 里所有张量名。

    用 ``safetensors`` 的 header 解析，不打开数据段——360 GB 的目录也能秒级列完。
    没装 ``safetensors`` 时抛 ``ImportError``，调用方据此跳过需要真实 checkpoint
    的路径（本机就是这样）。

    Args:
        model_dir: 权重目录。

    Returns:
        键名迭代器。

    Raises:
        ImportError: 环境里没有 ``safetensors``。
    """
    from safetensors import safe_open

    for path in sorted(Path(model_dir).glob("*.safetensors")):
        with safe_open(str(path), "pt") as handle:
            yield from handle.keys()


__all__ = [
    "DictSource",
    "ENGINE_PREFIX",
    "EXPERT_TENSORS",
    "LoadPlan",
    "LoadReport",
    "NGRAM_SHARD_MARKER",
    "TEXT_PREFIXES",
    "TensorSource",
    "engine_key",
    "iter_checkpoint_keys",
    "load_into",
    "plan",
    "read_config",
    "slice_experts",
    "text_config_of",
    "verify",
]
