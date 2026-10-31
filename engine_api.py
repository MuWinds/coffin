"""引擎面向服务的接口：一个 checkpoint 进，一个会话出。

Module: ``coffin/``（依赖图顶端，不感知模型族）。这是唯一知道有哪些模型族存在
的模块；其他所有引擎模块都通过这里声明的 ``Model`` / ``PagedModel`` 协议与某个
模型族打交道。运行时和诊断代码因此从不经由这层分派去 import 模型族，新增第二个
模型族也只需在本文件末尾加一次 ``register()``，而不用去改调用方。

引擎按职责切分：

    ops/                 任何模型共用的设备原语（GEMV、归一化、RoPE、分块
                         delta 规则、绑定嵌入）
    models/<family>/     一个架构族：块、MTP 头、权重加载器，以及持有图的模型类
    engine_api.py        本文件：``Model`` 契约、checkpoint -> 模型的分派，以及
                         运行时服务用的 reload

分派靠 checkpoint 配置而不是靠开关：:func:`create_model` 读 ``config.json``，把
登记过的键去匹配 ``model_type`` / ``architectures``。所以键 ``"qwen3_5"`` 既能
匹配 ``Qwen3_5ForConditionalGeneration``，也能匹配光秃秃的 ``qwen3_5``。对元数据
说谎的 checkpoint，用 ``architecture=`` 覆盖检测结果。

Constraints:
    * 导入一律是绝对导入（``from models.qwen35 import ...``），因为入口脚本把
      ``coffin/`` 本身放进了 ``sys.path``；``ops`` 和 ``models`` 是与
      ``hip_ops`` 并列的顶层包，不是本模块的子包。
    * 每个引擎模块都必须列进 ``_RELOAD_ORDER``。否则 :func:`reload` 会拒绝执行，
      因为漏掉的模块会继续提供 reload 之前的类对象，而它的导入方已经拿到新的——
      这种类型不匹配只在运行时才暴露。
    * 模型类在文件*末尾*导入（所以才要 ``# noqa: E402``）：紧随其后的
      ``register()`` 调用需要 ``_Entry`` 和 ``_REGISTRY`` 已存在，所以这是唯一
      允许 import 模型族的模块，而且必须在自己的分派面定义完之后才 import。
    * 这里不分配任何模型内存。权重由 :func:`load_weights` 送来，reload 时原样
      传回，重建因此只要几秒，而不是再读一遍 9 GB。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Protocol

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch


class Model(Protocol):
    """一个已加载的模型，一次处理一个请求。

    分词器、socket 和请求循环归服务端所有；token id 以下的一切都归模型管。服务层
    直接读取的属性有：

    ``max_seq``
        KV cache 和 RoPE 表按它构造的上下文窗口。
    ``pos1``
        长度为 1 的设备张量，装着当前位置。服务端和 ``PrefixCacheEngine`` 靠读写
        它来移动窗口，所以它属于契约，而不是私有细节。
    ``mtp``
        草稿模型对象；checkpoint 没有 MTP 头时为 ``None``。
    ``verify_c``
        验证图宽度：``spec_generate`` 每步最多接受 ``verify_c - 1`` 个草稿。
    """

    max_seq: int
    pos1: "torch.Tensor"
    verify_c: int
    mtp: Any | None

    def reset(self) -> None:
        """把每一项「按序列」的状态倒回空上下文。"""

    def prefill(self, tokens: list[int], chunk: int = 2048) -> "torch.Tensor":
        """处理一段 prompt，返回下一个 token 的 logits。

        Args:
            tokens: 要跑的 prompt token，从当前 ``pos1`` 开始。
            chunk: 把预填充切成多大的 GEMM M；这张卡上 rocBLAS 每次调用在 2048
                token 附近到顶。

        Returns:
            最后一个位置的 logits，也就是下一个 token 的抽样分布。

        Raises:
            RuntimeError: 模型没有分页池，这次预填充产出的 KV 无处安放。先调
                ``enable_paging``。
            ValueError: prompt 为空、超过 ``max_seq``，或者会从预留行之后开始。
        """

    def decode_step(self, token: int) -> "torch.Tensor":
        """沿 capture 好的解码图推进一个 token。

        Args:
            token: 要追加的 token；图会把它的 KV 写在 ``pos1`` 处并推进计数器。

        Returns:
            ``token`` 之后那个位置的 logits。

        Raises:
            RuntimeError: 图从来没 capture 过，没有可回放的东西；服务之前先调
                ``capture``。
        """

    def logits(self, token: int) -> "torch.Tensor":
        """不做 capture 的单步（预填充兜底 / 调试）。

        Args:
            token: 要追加的 token，与 ``decode_step`` 完全一样。

        Returns:
            ``decode_step`` 会产出的同一份 logits，只是不走图。用于 capture 不
            适用的场合：图没覆盖的形状，或者与 captured 路径做数值对比。

        Raises:
            RuntimeError: 没有分页池，这一步没有槽位可写。
        """

    def capture(self, warmup: int = 3) -> None:
        """把解码步（以及存在时的草稿模型）capture 一次。

        Args:
            warmup: capture 之前不计时的回放次数，用来把分配器和内核状态稳定下来；
                3 次就足以让第一个真实步不会在已经 capture 的区域内部触发分配。

        Raises:
            RuntimeError: 从来没开分页。captured 的内核把池张量和槽位表烧了进去，
                没有池就 capture 只会得到一张写到空处的图。
        """

    @property
    def graph_ready(self) -> bool:
        """:meth:`capture` 产出可回放的解码图之后为真。"""

    def capture_verify(self, c: int = 4, warmup: int = 3) -> None:
        """capture 宽度为 ``c`` 个 token 的验证前向。

        Args:
            c: 验证宽度，即草稿数加锚 token。它限定了 :meth:`spec_generate`
                能接受的 ``K``。
            warmup: capture 之前不计时的回放次数，理由同 :meth:`capture`。
        """

    def spec_generate(self, prompt: list[int], max_new: int, K: int = 3,
                      eos: set[int] | None = None) -> list[int]:
        """从一段 prompt 开始做贪心投机解码。

        Args:
            prompt: prompt 的 token，从空上下文开始处理。
            max_new: 生成 token 数的上限。
            K: 每步的草稿数，至多 ``verify_c - 1``。只有草稿模型的接受率能一直
                保持高位时，调大才划算。
            eos: 停止 token id；遇到 EOS 结束生成且不返回该 token。

        Returns:
            生成的 token id。与不投机的贪心输出完全一致，因为每份草稿都由主模型
            验证过。
        """

    @property
    def spec_stats(self) -> dict:
        """最近一次 :meth:`spec_generate` 的接受率报告。"""

    def close(self) -> None:
        """丢掉图和层对象，好让 reload 能重建。"""


class PagedModel(Model, Protocol):
    """在 ``Model`` 之上补齐前缀缓存需要的能力。

    混合（线性注意力）模型并不是只用 KV 承载前缀：24 层把它以定长循环状态的
    形式带下去，所以一段缓存的块前缀只有连同产生它的状态一起才能复用。这三个
    方法就是 ``PrefixCacheEngine`` 在 ``Model`` 之外还需要的东西。
    """

    def enable_paging(self, num_blocks: int = 512, block_size: int = 64):
        """挂上一个共享的分页 KV 池；必须在 :meth:`capture` 之前跑。

        Args:
            num_blocks: 池里的块数，也就是所有活请求合起来的 KV 预算。
            block_size: 每块的 token 数；同时也是基数树共享前缀的粒度。

        Returns:
            这个池；调用方通过一个 :class:`~radix_cache.RadixCache` 来驱动它。

        Raises:
            RuntimeError: 分页已经开过了；再开一个池会让第一个池的块永远无法
                寻址。
        """

    def snapshot_linear_states(self) -> dict[int, tuple["torch.Tensor", "torch.Tensor"]]:
        """克隆每一层线性注意力的 (循环状态, conv 状态)。

        Returns:
            层下标 -> (循环状态, conv 状态) 张量，都是脱离计算图的克隆，活状态
            因此可以继续前进，而快照仍是它在那个边界上的记录。
        """

    def restore_linear_states(self, states) -> None:
        """采用一份由 :meth:`snapshot_linear_states` 拍下的快照。

        Args:
            states: 要装入的快照；它必须描述本模型自己的线性注意力层，因为层下标
                就是按它做键的。

        Raises:
            KeyError: ``states`` 里出现了本模型没有的层下标，也就是来自另一个
                架构的快照。宁可拒绝，也不要恢复一份残缺状态、把共享 KV 静默
                配上错误的循环状态。
        """


@dataclass(frozen=True)
class _Entry:
    model: type
    load_weights: Callable[..., dict]


#: checkpoint 配置键 -> 实现。键按小写子串去匹配拼接后的 ``model_type`` /
#: ``architectures`` 字符串。
_REGISTRY: dict[str, _Entry] = {}


def register(keys: tuple[str, ...], model: type,
             load_weights: Callable[..., dict]) -> None:
    """让 ``model`` 成为配置里写着 ``keys`` 的那些 checkpoint 的实现。

    每个模型族在本文件末尾调用一次。同一个键后来的注册会覆盖前面的，reload 本
    模块因此才能生效。

    Args:
        keys: ``model_type`` / ``architectures`` 的小写子串。要短但要有辨识度：
            匹配就是一次普通子串判断，键太泛会把别的族的 checkpoint 也认领走。
        model: 该族的模型类；:func:`create_model` 调的就是它的构造签名。
        load_weights: 该族的权重加载器，:func:`load_weights` 会用 checkpoint
            路径调它。
    """
    for key in keys:
        _REGISTRY[key] = _Entry(model, load_weights)


def registered() -> list[str]:
    """这个引擎构建能服务的架构键。

    Returns:
        排好序的键，也就是 ``architecture=`` 能接受的取值，以及 checkpoint 一个
        都匹配不上时写进报错信息里的内容。
    """
    return sorted(_REGISTRY)


def _arch_strings(config: dict) -> list[str]:
    """从 checkpoint 配置里取出小写的架构线索。

    顶层和 ``text_config`` 都要读，因为多模态 checkpoint 把文本模型包了一层，只在
    嵌套段里才点它的名。

    Args:
        config: 解析后的 ``config.json``。

    Returns:
        两个段里的 ``model_type`` 和每一个 ``architectures`` 条目，按这个顺序，
        统一转成小写以便做子串匹配。
    """
    out: list[str] = []
    for section in (config, config.get("text_config") or {}):
        if not isinstance(section, dict):
            continue
        if section.get("model_type"):
            out.append(str(section["model_type"]))
        out.extend(str(a) for a in (section.get("architectures") or []))
    return [s.lower() for s in out]


def detect_architecture(config: dict) -> str:
    """返回某个 checkpoint ``config.json`` 对应的注册键。

    猜不出来就抛异常：用错误的实现去服务一个 checkpoint，得到的是垃圾而不是
    错误。

    Args:
        config: 解析后的 ``config.json``。

    Returns:
        注册表里存在的键，可以传给 ``architecture=`` 和 ``create_model``。

    Raises:
        RuntimeError: 没有任何登记项匹配这份配置，按几何特征的兜底也不适用。
            调用方只能换一个 checkpoint 或把该模型族登记进来；没有安全的默认
            选项。
    """
    text = " ".join(_arch_strings(config))
    for key in _REGISTRY:
        if key in text:
            return key
    # 多模态 checkpoint 可能把文本模型藏在包装配置后面；Qwen3.5 文本部分的几何
    # 足够有辨识度，可以当最后手段用。
    text_config = config.get("text_config") or {}
    if "linear_num_value_heads" in text_config:
        return "qwen3_5"
    raise RuntimeError(
        "no engine implementation matches this checkpoint: "
        f"model_type/architectures={text!r}; registered={registered()}")


def _entry_for(model_dir: str | Path, architecture: str | None = None) -> _Entry:
    """把一个 checkpoint 目录解析成它的注册项。

    Args:
        model_dir: 含有 ``config.json`` 的目录。
        architecture: 覆盖检测结果的显式键，用于元数据说谎的 checkpoint。

    Returns:
        该键对应的、装着模型类和权重加载器的注册项。

    Raises:
        RuntimeError: 解析出的键没有登记。报错里会列出已登记的键，因为单说一句
            「未知架构」等于什么都没给调用方。
    """
    config = json.loads((Path(model_dir) / "config.json").read_text())
    key = architecture or detect_architecture(config)
    try:
        return _REGISTRY[key]
    except KeyError:
        raise RuntimeError(
            f"unknown architecture {key!r}; registered={registered()}") from None


def create_model(model_dir: str | Path, max_seq: int = 4096, bench: bool = True,
                 weights: dict | None = None, paged: bool = False,
                 architecture: str | None = None, quant: bool = False) -> Model:
    """按这个 checkpoint 的要求加载模型实现。

    ``weights`` 是 :func:`load_weights` 得到的常驻张量字典，reload 时原样透传，
    重建因此省掉约 1 分钟的磁盘读取。

    ``quant`` 只在 ``weights`` 为 ``None``、也就是本次调用自己去加载 checkpoint
    时生效。要这个参数，是因为手里已经有那份字典的调用方早就自己选定了存储形式，
    而手里没有的调用方否则只能吃加载器默认的 bf16——常驻体积翻倍，正是 27B 放得进
    一张卡和放不进去的差别。同时传 ``quant`` 和 ``weights`` 不算错误，也不做转换：
    字典原样使用。

    Args:
        model_dir: checkpoint 目录。
        max_seq: KV 池和 RoPE 表按它定尺寸的上下文窗口。这是一笔显存承诺，不是
            运行时检查。
        bench: 加载时跑该族的内核变体调优。要多花几秒启动时间；测试里关掉。
        weights: 用来构建的、已经加载好的张量。传 ``None`` 就让该族自己去磁盘读。
        paged: 构造分页 KV 变体。前缀缓存和批处理路径需要它；不分页就
            ``capture`` 会把稠密布局烧进去。
        architecture: 覆盖配置检测的注册键。
        quant: 在本次调用自己加载 checkpoint 时，把文本投影存成 int8。传了
            ``weights`` 时忽略。

    Returns:
        一个 ``Model``；只有 ``paged`` 为真时才该把它当 ``PagedModel`` 用。

    Raises:
        RuntimeError: checkpoint 匹配不上任何已登记的族，或者点名的族没登记。
            换一个受支持的 checkpoint，别去猜实现。
    """
    entry = _entry_for(model_dir, architecture)
    return entry.model(model_dir, max_seq=max_seq, bench=bench,
                       weights=weights, paged=paged, quant=quant)


def load_weights(model_dir: str | Path, quant: bool = False,
                 weight_dtype: "torch.dtype | None" = None,
                 architecture: str | None = None, vision: bool | None = None,
                 vision_quant: bool = False) -> dict:
    """把这个 checkpoint 的权重读上 GPU，供常驻服务使用。

    单独拆出来而不是塞进 :func:`create_model`，是因为 reload 用显存里已有的张量
    重建模型，而不是再把 9 GB 从磁盘读一遍。

    ``vision``/``vision_quant`` 只控制视觉塔的张量：``None`` 表示「checkpoint 声明
    了 ``vision_config`` 就加载」，而 ``vision_quant`` 把塔的 GEMM 权重存成 int8
    （这是独立于文本模型的另一套内存策略，后者由 ``quant`` 管）。

    Args:
        model_dir: checkpoint 目录。
        quant: 把文本模型的二维投影权重存成 int8。在 16 GB 的卡上，这就是 KV
            池放不放得下的差别。
        weight_dtype: 仍然保持稠密的张量所用的计算 dtype。``None`` 用该族自己的
            默认值。
        architecture: 覆盖配置检测的注册键。
        vision: 是否加载视觉塔。``False`` 表示不加载，此时图像请求必须被拒绝，
            而不是静默地错误预填充。
        vision_quant: 把塔的 GEMM 权重存成 int8（常驻约 0.31 GiB 而不是
            0.62 GiB），代价是每个请求多一点开销。

    Returns:
        常驻的张量字典，按 checkpoint 参数名索引，现在以及之后每次 reload 都交给
        :func:`create_model`。

    Raises:
        RuntimeError: checkpoint 匹配不上任何已登记的族，也就没有加载器知道它的
            布局。
    """
    entry = _entry_for(model_dir, architecture)
    kwargs: dict[str, Any] = {} if weight_dtype is None else {"weight_dtype": weight_dtype}
    return entry.load_weights(Path(model_dir), quant=quant, vision=vision,
                              vision_quant=vision_quant, **kwargs)


def describe(model_dir: str | Path, architecture: str | None = None) -> dict:
    """启动时要记下来的 checkpoint 信息：路径、目录名和架构。

    服务启动时打印它，``ping`` 也返回它，OpenAI 外壳再把它读回去，这样谁都不用
    假设加载的是哪个模型——checkpoint 配置是唯一的事实来源。

    Args:
        model_dir: checkpoint 目录。
        architecture: 覆盖检测的注册键。

    Returns:
        ``model_path``（原样）、``model``（目录的 basename，``/v1/models`` 默认
        就报它）和 ``arch``（实际在用的注册键）。

    Raises:
        RuntimeError: checkpoint 匹配不上任何已登记的族，也就没有 ``arch`` 可报。
    """
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    return {"model_path": str(model_dir), "model": model_dir.name,
            "arch": architecture or detect_architecture(config)}


#: 依赖顺序：一个模块排在它 import 的所有东西之后，这样模型模块顶部的
#: ``from ops.gemv import GEMV`` 拿到的就是刚刚 reload 过的 ops 模块所定义的那个
#: 类对象。
_RELOAD_ORDER = (
    "backends.protocol", "backends.reference", "backends.vendor",
    "backends.discovery", "backends.hip", "backends.corex",
    "backends.vendors", "backends.registry", "backends",
    "ops.dtypes", "ops.runtime", "ops.gemv", "ops.embed", "ops.norm", "ops.rope",
    "ops.mrope", "ops.delta", "ops.suffix",
    "ops",
    "models.qwen35.tuning", "models.qwen35.weights", "models.qwen35.batch",
    "models.qwen35.vision",
    "models.qwen35.linear_attn",
    "models.qwen35.full_attn", "models.qwen35.blocks", "models.qwen35.mtp",
    "models.qwen35.model", "models.qwen35", "models",
    "utils.mm_input", "utils.tool_call", "utils",
    "cache.paged_cache", "cache.radix_cache", "cache",
    "engine.config", "engine.sequence", "engine.block_manager",
    "engine.prefix_engine", "engine.model_runner", "engine.scheduler",
    "engine.llm_engine", "engine",
    "engine_api",
)


def _engine_module(name: str) -> bool:
    """一个 ``sys.modules`` 条目是否属于本引擎。

    Args:
        name: 完全限定的模块名。

    Returns:
        属于引擎自己的包时返回 ``True``，:func:`reload` 据此发现那些被加载了却
        没被加进 ``_RELOAD_ORDER`` 的模块。
    """
    return (
        name in ("ops", "models", "engine", "cache", "utils")
        or name.startswith(("ops.", "models.", "engine.", "cache.", "utils."))
        or name in _RELOAD_ORDER
    )


def reload() -> None:
    """按依赖先后重新 import 每一个引擎模块。

    运行时卸载模型之后会调它。只 reload 顶层模块的话，实现模块会停在旧版本上。
    调用方持有的模块对象是就地更新的，所以此后的 ``engine_api.create_model``
    就是新函数。

    Raises:
        RuntimeError: 有已加载的引擎模块没列进 ``_RELOAD_ORDER``。那种模块会继续
            提供 reload 之前的类对象，而它的导入方已经用上新的。把它加进列表即可；
            报错信息会点名是哪个模块。
    """
    import importlib
    import sys

    stale = {name for name in sys.modules if _engine_module(name)} - set(_RELOAD_ORDER)
    if stale:
        raise RuntimeError(
            f"engine_api.reload(): add {sorted(stale)} to _RELOAD_ORDER")
    for name in _RELOAD_ORDER:
        module = sys.modules.get(name)
        if module is not None:
            importlib.reload(module)


# 唯一允许向上的那条边：engine_api 伸进某个模型族，反方向永远不许。放在分派代码
# 下面，是为了让本模块自己的接口面先定义完再拉进模型族，也因为它紧随其后的注册
# 需要 ``register`` 和 ``_Entry`` 已存在。这一行以下的文件都会触发 ``# noqa: E402``。
from models.qwen35 import Qwen35, load_weights as _qwen35_load_weights  # noqa: E402

register(("qwen3_5", "qwen35"), Qwen35, _qwen35_load_weights)
