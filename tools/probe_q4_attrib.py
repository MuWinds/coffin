"""一步 decode 的小内核到底是谁发的：按 ``file:line`` 归因，而不是按内核名猜。

## 为什么需要这个探针

`Q4EXP-PERF-ROUND2.md` 的账本：decode 一步 device 合计 259.2 ms，top12 有名有姓的
内核只有 108.7 ms，**缺口 150.5 ms** 落在 4230 个 ``elementwise_kernel`` /
``reduce_kernel`` / ``CatArrayBatchedCopy`` 上，平均 35.6 µs/个，而空核地板 6.5 µs。
按内核名只能知道"是哪一类算子"，不能知道"是谁调的"，而同一类算子在不同调用点的
处置完全不同（同一个 ``copy_`` 可能是必要布局搬运，也可能是可以删掉的冗余转换）。

`COREX-GLUE-PLAN.md` §4 因此把 ``with_stack=True`` 的那次测量列为动手前提。本探针
就是那次测量，并且额外给一份**主机侧**的 aten 调用点计数——它不依赖 profiler，
在 CoreX 这种 profiler 偶尔不稳的栈上也能出结果。

## 两种口径，不能互相替代

* ``--mode count``：``TorchFunctionMode`` 在 Python 层拦每一次 aten 调用，按
  ``file:line`` 累计次数与主机耗时。回答"谁发了 655 次 ``mul``"。
* ``--mode profile``：``torch.profiler(with_stack=True)`` 给每个 CUDA 事件带
  Python 调用栈，按栈聚合设备时间。回答"那 150 ms 具体属于哪几行"。

两个都要跑：次数多不等于贵（6 µs 的空核 × 655 = 4 ms），贵也不等于该改
（可能是必要的 dtype 边界）。

## 判据（本脚本打印，人工核对）

1. ``enqueue_ms`` 与 ``wall_ms``：接近则主机受限，先减发射；远小则设备受限。
2. 设备时间 top 表的覆盖率：top 项占比之和若远小于总和，缺口就是小内核。
3. 主机侧 top 表的 ``count`` 与 profiler 的 kernel 次数要对得上同一个量级。

Module: ``coffin/tools``。只在有卡的机器上跑；本地 Windows 用 ``--synthetic`` 冒烟。

Constraints:
    * 不用 CUDA graph、不用 cp.async（``platform/scow/AGENTS.md`` 第 1、5 条）。
    * 模型按 ``--layers`` 裁层跑：真实 48 层 W4A8 要 18.4 GiB 常驻，4 层 bf16 只要
      约 5 GiB，能在同一张卡上快速迭代。**裁层只改层数与层类型表，不改任何层的
      内部结构**，所以每层的算子构成与 48 层版本一致。
    * ``stream_ngram`` 打开时不装 n-gram 表（102.4 GB 进不了显存）；本探针把 PLE
      层关掉（``ple_layer_ids=[]``），PLE 的代价单独用 host 表探针量。
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

#: 本文件在 ``coffin/tools/`` 下，父目录就是引擎根。
_ENGINE = Path(__file__).resolve().parents[1]
if str(_ENGINE) not in sys.path:
    sys.path.insert(0, str(_ENGINE))

from models.qwen4exp.checkpoint import StreamingSource  # noqa: E402
from models.qwen4exp.model import CausalLM, causal_mask  # noqa: E402
from models.qwen4exp.weights import load_into, read_config, text_config_of  # noqa: E402

#: 分页池的 token 容量；``0`` 表示走稠密缓存（每步 ``torch.cat`` 一次历史）。
#: 放在模块级，是为了让统计/剖析那几个"只跑一遍"的函数不必各自加参数。
_PAGED_CAPACITY = 0
#: prefill 一次前向覆盖多少 token；``0`` 表示一整个 prompt 一次跑完。
_PREFILL_CHUNK = 0


def log(message: str) -> None:
    """打一行带时间戳的日志。

    Args:
        message: 正文。
    """
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def build_config(raw: dict, layers: int, keep_ple: bool = False) -> object:
    """按裁层要求改一份配置。

    只动三处：层数、层类型表的前缀、PLE 层表清空。层内部结构一概不改，否则量到
    的就不是 48 层那条路径的单层构成。

    Args:
        raw: checkpoint 的 ``config.json`` 解析结果。
        layers: 保留多少层。
        keep_ple: 是否保留 PLE 层。留 ``False`` 时把 ``ple_layer_ids`` 清空——纯
            归因探针不需要那 102.4 GB 的表；真机生成探针必须留 ``True``。

    Returns:
        ``TextConfig``。
    """
    raw = copy.deepcopy(raw)
    text = raw.setdefault("text_config", {})
    types = list(text.get("layer_types") or [])
    if types:
        text["layer_types"] = types[:layers]
    text["num_hidden_layers"] = layers
    if not keep_ple:
        text["ple_layer_ids"] = []
    return text_config_of(raw)


def build_model(config, quantized: bool, rank: int | None, world: int,
                device: torch.device, combine: "callable | None" = None) -> CausalLM:
    """建一个裁层后的 ``CausalLM``。

    量化路径**不**在这里落设备：``Experts`` 的专家栈在 ``quantized=True`` 下是 meta
    张量，直接 ``to(device)`` 会为每一层分配一份 4.87 GB 的 bf16 专家栈然后立刻扔掉
    （48 层 60.4 GB，4 层也要 19.5 GB）。必须先 :func:`attach_quantized` 把 meta 参数
    摘掉，再整体搬家。

    Args:
        config: 文本配置。
        quantized: 稠密投影是否走 W4A8（此时不建 bf16 占位）。
        rank: EP 分片的本 rank 编号；``None`` 表示单卡全量。
        world: EP 进程数。
        device: 目标设备。
        combine: 路由专家输出的跨 rank 归约闭包；``None`` 表示单卡。

    Returns:
        空权重的模型；量化路径下它的专家栈仍是 meta，需要调用方接着挂产物。
    """
    local_experts = None
    if rank is not None:
        per = config.num_experts // world
        local_experts = list(range(rank * per, (rank + 1) * per))
    model = CausalLM(config, local_experts=local_experts, stream_ngram=True,
                     quantized=quantized, combine=combine, with_mtp=False)
    return model if quantized else model.to(device)


#: 形如 ``model.language_model.layers.<N>...`` 里的层号；不是分层张量时返回 ``-1``。
_LAYER_RE = None


def _layer_index(key: str) -> int:
    """从 checkpoint 键里取层号，供裁层过滤用。

    Args:
        key: checkpoint 张量名。

    Returns:
        层号；键里没有 ``.layers.<N>.`` 时返回 ``-1``（嵌入、lm_head 等）。
    """
    global _LAYER_RE
    if _LAYER_RE is None:
        import re

        _LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
    match = _LAYER_RE.search(key)
    return int(match.group(1)) if match else -1


def attach_quantized(model: CausalLM, rank_dir: Path, device: torch.device,
                     layers: int) -> dict:
    """把 W4A8 产物挂到模型的 ``Proj`` 与 ``Experts`` 上。

    产物里的键是 checkpoint 名（``model.language_model.layers.N...``），而模块名是
    引擎名；映射交给 ``weights.engine_key``，不要在这里另写一份规则——两份规则
    迟早会分叉，而分叉的表现是"某些投影悄悄用回了 bf16"。

    只读 ``< layers`` 层的张量：48 层 rank 目录是 20 GB，全读进来再扔掉既慢又占显存，
    而裁层探针用不到它们。

    Args:
        model: 已建好的量化模型（专家栈是 meta）。
        rank_dir: 这个 rank 的产物目录。
        device: 目标设备。
        layers: 模型实际有多少层。

    Returns:
        计数：挂上几个模块、缺条目的模块名前几个。
    """
    from models.qwen4exp.moe import Experts
    from models.qwen4exp.proj import Proj, QuantizedEmbedding
    from models.qwen4exp.weights import engine_key
    from quantized_load import load as load_quantized

    tensors = load_quantized(rank_dir, device=device, quant=True,
                             keep=lambda key: _layer_index(key) < layers)
    by_engine: dict[str, object] = {}
    for key, value in tensors.items():
        name = engine_key(key)
        if name is not None:
            by_engine[name] = value

    attached = 0
    missing: list[str] = []
    for name, module in model.named_modules():
        if isinstance(module, Experts):
            up = by_engine.get(f"{name}.gate_up_proj")
            down = by_engine.get(f"{name}.down_proj")
            if up is None or down is None:
                missing.append(name)
                continue
            module.set_quantized(up, down)
            attached += 1
        elif isinstance(module, (Proj, QuantizedEmbedding)):
            # 产物按 checkpoint 参数名索引，所以键里带 ``.weight``；而这里拿到的是
            # 模块名。少这层后缀不会报错，只会让每个投影都被判成"缺条目"，然后
            # 在第一次前向时以"bf16 表被释放了"的形式炸出来——与本因隔一层。
            weight = by_engine.get(f"{name}.weight")
            if weight is None:
                weight = by_engine.get(name)
            if weight is None:
                missing.append(name)
                continue
            # 产物里不是每个投影都有 q4 条目：量化工具按张量语义分类，窄投影、门控和
            # 归一化的伴生矩阵会留成 dense。这些要 ``set_bf16``，交给它去重建参数；
            # 一律走 ``set_quantized`` 会以 "expects W4A8Weight, got Tensor" 炸在
            # 一个看起来完全无关的层上。
            if isinstance(weight, torch.Tensor):
                module.set_bf16(weight)
            else:
                module.set_quantized(weight)
            attached += 1
    return {"attached": attached, "n_missing": len(missing), "missing": missing[:8]}


def decode_once(model: CausalLM, token: int, states, past: int,
                device: torch.device, ple_tables: dict | None = None
                ) -> tuple[torch.Tensor, object]:
    """跑一步 decode。

    Args:
        model: 模型。
        token: 本步的 token id。
        states: 上一步留下的状态；``None`` 表示从头开始。
        past: 已经处理过的 token 数，决定掩码宽度与位置。
        device: 设备。

    Returns:
        ``(logits, new_states)``。
    """
    ids = torch.tensor([[int(token)]], dtype=torch.long, device=device)
    # ``causal_mask`` 返回的是 CPU 上的 bool 掩码（服务层自己负责搬），探针这里
    # 必须自己搬：QSA 的 `_additive_mask` 会把它和 cuda 上的分数相加。
    mask = causal_mask(1, 1, past).to(device)
    conv = torch.ones((1, 1), dtype=torch.bool, device=device)
    logits, new_states = model(input_ids=ids, attention_mask=mask, conv_mask=conv,
                               states=states, ple_tables=ple_tables,
                               logits_to_keep=1)
    return logits, new_states


def prefill(model: CausalLM, tokens: list[int], device: torch.device,
            ple_tables: dict | None = None):
    """跑一段 prompt，返回末位 logits 与状态。

    分页池与分块预填充由模块级的 ``_PAGED_CAPACITY`` / ``_PREFILL_CHUNK`` 控制：
    统计与剖析那几个函数只是"跑一遍"，不看缓存怎么摆，所以开关放在模块级而不是
    给每个调用点加参数。

    Args:
        model: 模型。
        tokens: prompt token。
        device: 设备。
        ple_tables: host n-gram 表，``None`` 表示不用 PLE。

    Returns:
        ``(logits, states)``。
    """
    if _PAGED_CAPACITY and model.model._paged_max_seq is None:
        model.model.enable_paged_kv(_PAGED_CAPACITY)
    states = None
    logits = None
    step = _PREFILL_CHUNK if _PREFILL_CHUNK > 0 else len(tokens)
    for start in range(0, len(tokens), step):
        chunk = tokens[start:start + step]
        ids = torch.tensor([chunk], dtype=torch.long, device=device)
        mask = causal_mask(1, len(chunk), start).to(device)
        conv = torch.ones((1, len(chunk)), dtype=torch.bool, device=device)
        logits, states = model(input_ids=ids, attention_mask=mask, conv_mask=conv,
                               ple_tables=ple_tables, states=states,
                               logits_to_keep=1)
    return logits, states


class A(torch.overrides.TorchFunctionMode):  # noqa: N801 - 只在本文件用
    """按 ``coffin`` 内的调用点累计 aten 调用的次数与主机耗时。"""

    def __init__(self) -> None:
        """初始化两张表。"""
        super().__init__()
        self.count: Counter = Counter()
        self.millis: Counter = Counter()
        self._depth = 0

    def _site(self) -> str:
        """在调用栈里找最近的两帧 coffin 源码。

        ``__torch_function__`` 的下一帧永远是 torch 内部，所以必须往上找到属于本仓
        的帧。只取最内一帧不够用：``Proj.forward`` 是每个投影都会走的薄壳，按它聚合
        会把 509 次投影全并成一行，看不出「谁在调它」。所以取两级，
        ``内层 <- 外层``，外层才是可下手的调用点。

        Returns:
            ``内层:行 <- 外层:行``；找不到时返回 ``"<torch>"``。
        """
        found: list[str] = []
        frame = sys._getframe(2)
        while frame is not None and len(found) < 2:
            name = frame.f_code.co_filename
            if "_engine" in name or "coffin" in name or "models" in name:
                try:
                    rel = Path(name).resolve().relative_to(_ENGINE)
                    found.append(f"{rel.as_posix()}:{frame.f_lineno}")
                except (ValueError, OSError):
                    found.append(f"{Path(name).name}:{frame.f_lineno}")
            frame = frame.f_back
        if not found:
            return "<torch>"
        return " <- ".join(found)

    def __torch_function__(self, func, types, args=(), kwargs=None):  # noqa: D401
        """记录一次调用再原样转交。"""
        kwargs = {} if kwargs is None else kwargs
        name = getattr(func, "__name__", None) or repr(func)
        site = self._site()
        key = f"{site} {name}"
        self.count[key] += 1
        if self._depth:                       # 嵌套调用只记最外层耗时
            return func(*args, **kwargs)
        self._depth += 1
        started = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            self._depth -= 1
            self.millis[key] += (time.perf_counter() - started) * 1e3


def print_table(title: str, rows: list[tuple[str, float, float]], unit: str) -> None:
    """按数值降序打一张两列的表。

    Args:
        title: 表头。
        rows: ``(名字, 主数值, 次数值)``。
        unit: 主数值的单位。
    """
    print(f"\n### {title}")
    print(f"| 调用点 | {unit} | 次数 |")
    print("| --- | ---: | ---: |")
    for name, value, count in rows:
        print(f"| `{name}` | {value:.3f} | {int(count)} |")


def run_count_pass(model: CausalLM, tokens: list[int], steps: int,
                   device: torch.device, ple_tables: dict | None = None) -> dict:
    """用 ``TorchFunctionMode`` 归因一次 decode 的 aten 调用。

    Args:
        model: 模型。
        tokens: 冷启动用的 prompt。
        steps: decode 步数。
        device: 设备。

    Returns:
        ``{"wall_ms": 每步均值, "top_count": [(点, 次数, 毫秒)], "total_calls": n}``。
    """
    logits, states = prefill(model, tokens, device, ple_tables)
    token = int(logits[0, -1].argmax())
    mode = A()
    started = time.perf_counter()
    with mode:
        for step in range(steps):
            logits, states = decode_once(model, token, states, len(tokens) + step,
                                         device, ple_tables)
            token = int(logits[0, -1].argmax())
    wall = (time.perf_counter() - started) * 1e3 / steps
    rows = [(key, ms, mode.count[key]) for key, ms in mode.millis.most_common(40)]
    by_count = [(key, mode.millis[key], count)
                for key, count in mode.count.most_common(40)]
    return {"wall_ms": wall, "top_time": rows, "top_count": by_count,
            "total_calls": sum(mode.count.values())}


def run_profile_pass(model: CausalLM, tokens: list[int], steps: int,
                     device: torch.device, ple_tables: dict | None = None) -> dict:
    """用 ``torch.profiler(with_stack=True)`` 归因一次 decode 的设备时间。

    Args:
        model: 模型。
        tokens: 冷启动用的 prompt。
        steps: decode 步数。
        device: 设备。

    Returns:
        ``{"device_ms": 每步设备时间, "by_kernel": [(内核, ms, 次数)],
        "sites": [(调用点, ms, 次数)], "total_ms": n}``。
    """
    logits, states = prefill(model, tokens, device, ple_tables)
    token = int(logits[0, -1].argmax())
    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities, with_stack=True) as prof:
        for step in range(steps):
            logits, states = decode_once(model, token, states, len(tokens) + step,
                                         device, ple_tables)
            token = int(logits[0, -1].argmax())
        if device.type == "cuda":
            torch.cuda.synchronize()

    kernels: Counter = Counter()
    counts: Counter = Counter()
    sites: Counter = Counter()
    site_counts: Counter = Counter()
    total = 0.0
    for event in prof.events():
        device_us = float(getattr(event, "cuda_time_total", 0.0) or 0.0)
        if device_us <= 0:
            continue
        name = event.key
        kernels[name] += device_us / 1e3 / steps
        counts[name] += event.count
        total += device_us / 1e3 / steps
        stack = getattr(event, "stack", None) or []
        site = _stack_site(stack)
        if site is not None:
            sites[site] += device_us / 1e3 / steps
            site_counts[site] += event.count
    return {
        "total_ms": total,
        "by_kernel": [(name, ms, counts[name]) for name, ms in kernels.most_common(30)],
        "sites": [(site, ms, site_counts[site]) for site, ms in sites.most_common(30)],
        "launches": int(sum(counts.values())),
    }


def _stack_site(stack) -> str | None:
    """从 profiler 的调用栈里取出最近的一帧 coffin 源码。

    Args:
        stack: ``event.stack``，元素可能是元组也可能是字符串，随 torch 版本而变。

    Returns:
        ``相对路径:行号:函数名``；栈里没有本仓帧时返回 ``None``。
    """
    for entry in stack:
        if isinstance(entry, (tuple, list)) and len(entry) >= 3:
            filename, lineno, func = entry[0], entry[1], entry[2]
        else:
            text = str(entry)
            if "models/qwen4exp" not in text and "coffin" not in text:
                continue
            return text[-120:]
        if "models/qwen4exp" not in str(filename) and "coffin" not in str(filename):
            continue
        try:
            rel = Path(filename).resolve().relative_to(_ENGINE)
            return f"{rel.as_posix()}:{lineno}:{func}"
        except (ValueError, OSError):
            return f"{Path(filename).name}:{lineno}:{func}"
    return None


def synthetic_setup(layers: int, device: torch.device) -> tuple[CausalLM, dict]:
    """不读盘建一个极小模型，供无卡/无权重时冒烟。

    Args:
        layers: 层数。
        device: 设备。

    Returns:
        ``(model, raw_config)``。
    """
    from models.qwen4exp.config import TextConfig

    raw = {
        "text_config": {
            "hidden_size": 128, "hc_count": 4, "hc_lowrank": 16,
            "head_dim": 64, "num_attention_heads": 4, "num_key_value_heads": 1,
            # 64 维头 × 0.25 部分旋转 = 16 个旋转维 = 8 对，mrope 三段之和必须是 8。
            "partial_rotary_factor": 0.25, "mrope_section": [3, 3, 2],
            "mrope_interleaved": True,
            "linear_key_head_dim": 32, "linear_value_head_dim": 32,
            "linear_num_key_heads": 2, "linear_num_value_heads": 2,
            "linear_conv_kernel_dim": 4, "num_experts": 8, "num_experts_per_tok": 2,
            "moe_intermediate_size": 32, "shared_expert_intermediate_size": 32,
            "num_hidden_layers": layers, "vocab_size": 256,
            "layer_types": ["linear_attention", "linear_attention",
                            "linear_attention", "full_attention"][:layers],
            "ple_layer_ids": [], "ngram_size": 3, "ple_embed_dim": 128,
            "indexer_head_dim": 32, "indexer_n_heads": 1, "indexer_kv_heads": 1,
            "indexer_compress_ratio": 2, "indexer_budget": 4,
            "max_position_embeddings": 512, "rms_norm_eps": 1e-6,
        }
    }
    config = text_config_of(raw)
    model = CausalLM(config, stream_ngram=True, with_mtp=False).to(device)
    return model, raw


def compare_chunk_prefill(model: CausalLM, tokens: list[int], chunk: int,
                          device: torch.device,
                          ple_tables: dict | None = None) -> dict:
    """同一段 prompt 用「一次跑完」与「分块跑完」两条路，比较末位 logits。

    为什么不能只比 token：单进程只装了 1/4 的专家，logits 本身就是噪声，
    argmax 会因为 1e-3 的数值差翻转，那种翻转说明不了任何事。比 logits 才能
    区分「分块算错」（差到量级 1）和「fp 累加次序不同」（相对 1e-6）。

    Args:
        model: 模型。
        tokens: prompt token。
        chunk: 分块大小。
        device: 设备。
        ple_tables: host n-gram 表。

    Returns:
        ``{"oneshot_absmax", "chunked_absmax", "max_abs_diff", "rel"}``。
    """
    global _PREFILL_CHUNK
    saved = _PREFILL_CHUNK
    _PREFILL_CHUNK = 0
    oneshot, _ = prefill(model, tokens, device, ple_tables)
    _PREFILL_CHUNK = max(int(chunk), 1)
    chunked, _ = prefill(model, tokens, device, ple_tables)
    _PREFILL_CHUNK = saved
    gap = float((oneshot - chunked).abs().max())
    scale = float(oneshot.abs().max())
    return {"oneshot_absmax": round(scale, 6),
            "chunked_absmax": round(float(chunked.abs().max()), 6),
            "max_abs_diff": gap,
            "rel": gap / max(scale, 1e-12)}


def run_generate(model: CausalLM, tokens: list[int], max_new: int,
                 device: torch.device, ple_tables: dict | None = None) -> dict:
    """贪心生成若干个 token，并报 prefill 与每步 decode 的毫秒。

    这是"能不能真的出 token"的最小验收：它走的是与 serving 相同的前向路径
    （含 PLE host 表的 prefetch/wait），只是没有调度器与 HTTP 外壳。

    Args:
        model: 模型。
        tokens: prompt。
        max_new: 生成多少个 token（含 prefill 后的第一个）。
        device: 设备。
        ple_tables: host n-gram 表；``None`` 表示不带 PLE。

    Returns:
        ``{"tokens": [...], "prefill_ms": 毫秒, "decode_ms": 每步均值毫秒}``。
    """
    started = time.perf_counter()
    logits, states = prefill(model, tokens, device, ple_tables)
    if device.type == "cuda":
        torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - started) * 1e3
    token = int(logits[0, -1].argmax())
    out = [token]
    steps: list[float] = []
    for step in range(max(0, max_new - 1)):
        t0 = time.perf_counter()
        logits, states = decode_once(model, token, states, len(tokens) + step,
                                     device, ple_tables)
        if device.type == "cuda":
            torch.cuda.synchronize()
        steps.append((time.perf_counter() - t0) * 1e3)
        token = int(logits[0, -1].argmax())
        out.append(token)
    return {"tokens": out, "prefill_ms": prefill_ms,
            "decode_ms": sum(steps) / len(steps) if steps else 0.0}


def run_batched_generate(model: CausalLM, tokens: list[int], max_new: int,
                         batch: int, device: torch.device,
                         ple_tables: dict | None = None) -> dict:
    """用引擎的静态批跑一段生成，报告每步毫秒与聚合吞吐。

    为什么走 ``Qwen4Engine`` 而不是直接堆一个 ``[B, 1]`` 前向：批的收益必须能在
    真实服务路径上兑现，否则测出来的是"探针能批处理"，不是"引擎能批处理"。

    Args:
        model: 已装载的模型。
        tokens: 每行共用的 prompt（静态批要求等长）。
        max_new: 生成多少个 token。
        batch: 行数。
        device: 设备。
        ple_tables: host n-gram 表。

    Returns:
        ``{"tokens", "prefill_ms", "decode_ms", "tok_per_s", "batch"}``。
    """
    from models.qwen4exp.engine import Qwen4Engine

    capacity = max(int(batch), 2)
    span = len(tokens) + max_new + 8
    engine = Qwen4Engine(model, max_seq=span, ple_tables=ple_tables)
    engine.enable_paging(num_blocks=1, block_size=max(64, span))
    engine.enable_batching(capacity)

    class _Seq:
        """``prefill_group`` 只读 ``token_ids``，所以最小形状就是这个。"""

        def __init__(self, ids: list[int]) -> None:
            self.token_ids = ids

    seqs = [_Seq(list(tokens)) for _ in range(capacity)]
    started = time.perf_counter()
    rows, _ = engine.prefill_group(seqs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - started) * 1e3
    picked = [int(row.argmax()) for row in rows]
    steps: list[float] = []
    for step in range(max(0, max_new - 1)):
        t0 = time.perf_counter()
        engine.set_rows(picked, [len(tokens) + step] * capacity,
                        row_indices=list(range(capacity)))
        out = engine.decode_batch_eager(capacity, capacity)
        if device.type == "cuda":
            torch.cuda.synchronize()
        steps.append((time.perf_counter() - t0) * 1e3)
        picked = [int(out[i].argmax()) for i in range(capacity)]
    mean_ms = sum(steps) / len(steps) if steps else 0.0
    return {"tokens": picked, "prefill_ms": prefill_ms, "decode_ms": mean_ms,
            "tok_per_s": capacity * 1000.0 / max(mean_ms, 1e-9),
            "batch": capacity}


def main(argv: list[str] | None = None) -> int:
    """跑归因并打印账本。

    Args:
        argv: 命令行参数。

    Returns:
        ``0`` 正常完成。
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", default="", help="checkpoint 目录")
    parser.add_argument("--layers", type=int, default=4, help="裁层后的层数")
    parser.add_argument("--mode", default="bf16", choices=("bf16", "w4a8"))
    parser.add_argument("--rank-dir", default="", help="w4a8 模式下这个 rank 的产物目录")
    parser.add_argument("--rank", type=int, default=None, help="EP rank；不给表示全量专家")
    parser.add_argument("--world", type=int, default=4)
    parser.add_argument("--prompt-len", type=int, default=8)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--passes", default="count,profile", help="要跑哪几遍")
    parser.add_argument("--synthetic", action="store_true", help="不读盘，用极小随机模型")
    parser.add_argument("--keep-ple", action="store_true",
                        help="保留 PLE 层（真机生成探针用；归因探针不要开）")
    parser.add_argument("--host-ple", action="store_true",
                        help="从 checkpoint 分片装 host n-gram 表并接进前向")
    parser.add_argument("--gen", type=int, default=0,
                        help="生成多少个 token 并报每步毫秒；0 表示只跑归因")
    parser.add_argument("--paged", type=int, default=0,
                        help="打开块粒度 KV 池，参数是池的 token 容量；0 表示走稠密缓存")
    parser.add_argument("--prefill-chunk", type=int, default=0,
                        help="prefill 按多少个 token 分块；0 表示一次跑完")
    parser.add_argument("--compare-chunk", type=int, default=0,
                        help="用这个分块大小与一次性 prefill 比末位 logits；0 表示不比")
    parser.add_argument("--batch", type=int, default=0,
                        help="用引擎静态批跑生成，参数是行数；0 表示单行")
    parser.add_argument("--json-out", default="", help="把结果写成 JSON")
    args = parser.parse_args(argv)

    global _PAGED_CAPACITY, _PREFILL_CHUNK
    _PAGED_CAPACITY = int(args.paged)
    _PREFILL_CHUNK = int(args.prefill_chunk)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device} mode={args.mode} layers={args.layers} "
        f"passes={args.passes} synthetic={args.synthetic}")

    if args.synthetic:
        model, raw = synthetic_setup(args.layers, device)
        # CPU 上 bf16 的逐元素路径比 fp32 还慢且没有代表性，冒烟统一用 fp32。
        model = model.to(torch.float32)
        tokens = list(range(2, 2 + args.prompt_len))
        ple_tables = None
    else:
        if not args.model_dir:
            parser.error("--model-dir 必填（或用 --synthetic）")
        model_dir = Path(args.model_dir)
        raw = read_config(model_dir)
        # 带 PLE 的模型必须留着 ple_layer_ids，否则装 host 表时找不到层。
        config = build_config(raw, args.layers,
                              keep_ple=args.keep_ple or args.host_ple)
        quantized = args.mode == "w4a8"
        model = build_model(config, quantized, args.rank, args.world, device)
        if quantized:
            if not args.rank_dir:
                parser.error("w4a8 模式要给 --rank-dir")
            report = attach_quantized(model, Path(args.rank_dir), device, args.layers)
            log(f"量化挂载: {report}")
            # 量化挂载会摘掉 meta 专家栈，此后才可以把剩下的非量化参数搬上卡。
            model = model.to(device)
        else:
            source = StreamingSource(model_dir, device="cpu")
            t0 = time.perf_counter()
            report = load_into(model, source, config, max_layers=args.layers,
                               prepare=lambda name, t: t.to(device=device,
                                                            dtype=torch.bfloat16))
            log(f"装载 {report} 用时 {time.perf_counter() - t0:.1f}s")
        ple_tables = None
        if args.host_ple:
            from models.qwen4exp.weights import load_host_ngram_table

            t0 = time.perf_counter()
            # ``device`` 决定 wait() 把行送到哪里；表本身与中转缓冲始终在 host 上。
            # 不传就会默认成 CPU，然后 PLE 层的投影拿到 CPU 激活并在第一个 Proj 上
            # 报 "W4 weight and activation must share a device"。
            table = load_host_ngram_table(model_dir, config, device=device)
            # 表的键是**零基层号**，而 ``ple_layer_ids`` 是 1 基层号（第 2 层 =
            # index 1）。拿配置的号直接当键会指到一个没有 PLE 的层，然后在前向
            # 开头被 prefetch 的校验拒掉。以模型里真实的 PLE 层为准。
            layers_with_ple = [idx for idx, layer in enumerate(model.model.layers)
                               if layer.ple is not None]
            ple_tables = {idx: table for idx in layers_with_ple}
            log(f"host n-gram 表: 0 基层 {layers_with_ple} 装载 "
                f"{time.perf_counter() - t0:.2f}s（mmap 惰性，不读数据）")
        tokens = list(range(2, 2 + args.prompt_len))

    if args.batch > 1 and args.gen > 0:
        batched = run_batched_generate(model, tokens, args.gen, args.batch, device,
                                       ple_tables)
        print(f"BATCHED {json.dumps(batched, ensure_ascii=False)}")
        if not args.passes:
            print("ATTRIB_DONE")
            return 0

    if args.compare_chunk > 0:
        gap = compare_chunk_prefill(model, tokens, args.compare_chunk, device,
                                    ple_tables)
        print(f"CHUNK_CMP {json.dumps(gap, ensure_ascii=False)}")
        if not args.passes and args.gen <= 0:
            print("ATTRIB_DONE")
            return 0

    if args.gen > 0:
        # 生成冒烟：只关心「能不能跑、一步多少毫秒、出不出有限 token」。
        generated = run_generate(model, tokens, args.gen, device, ple_tables)
        print(f"\nGEN tokens={generated['tokens']}")
        print(f"GEN prefill_ms={generated['prefill_ms']:.2f} "
              f"decode_ms_per_step={generated['decode_ms']:.2f} "
              f"tok_per_s={1000.0 / max(generated['decode_ms'], 1e-9):.2f}")
        if not args.passes:
            print("ATTRIB_DONE")
            return 0

    logits, _ = prefill(model, tokens, device, ple_tables)
    torch.cuda.synchronize() if device.type == "cuda" else None
    log(f"prompt 末位 logits: shape={tuple(logits.shape)} "
        f"absmax={float(logits.abs().max()):.4f} "
        f"finite={bool(torch.isfinite(logits).all())}")

    result: dict = {"mode": args.mode, "layers": args.layers,
                    "device": str(device), "synthetic": args.synthetic,
                    "host_ple": bool(args.host_ple)}
    passes = {p.strip() for p in args.passes.split(",") if p.strip()}
    if "count" in passes:
        counted = run_count_pass(model, tokens, args.steps, device, ple_tables)
        result["count"] = counted
        log(f"count 遍: 一步 {counted['wall_ms']:.2f} ms, "
            f"aten 调用 {counted['total_calls']} 次/步")
        print_table("主机侧耗时 top（TorchFunctionMode）",
                    counted["top_time"], "ms/步")
        print_table("主机侧次数 top（TorchFunctionMode）",
                    counted["top_count"], "ms/步")
    if "profile" in passes:
        try:
            prof = run_profile_pass(model, tokens, args.steps, device, ple_tables)
        except Exception as exc:                    # noqa: BLE001 - profiler 在 CoreX 上可能不稳
            log(f"profiler 失败，跳过: {exc!r}")
            prof = None
        if prof is not None:
            result["profile"] = prof
            log(f"profile 遍: 设备 {prof['total_ms']:.2f} ms/步, "
                f"内核 {prof['launches']} 次/步")
            print_table("设备时间 top（按内核）", prof["by_kernel"], "ms/步")
            print_table("设备时间 top（按 Python 调用点）", prof["sites"], "ms/步")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
        log(f"结果写到 {args.json_out}")
    print("ATTRIB_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
