"""四进程 EP4 的真机生成冒烟：每卡一个 rank 分片，MoE 之后跨 rank 归约。

## 它验的是什么

`Q4EXP-PERF-ROUND2.md` §4 量到四卡 EP 只有 0.96x，但那条路径的 prefill 归约用的是
共享内存 barrier：`rows=64` 时 **194 ms/次**（48 层 ⇒ 9.3 s），gloo 同形状 730 µs/次。
所以四卡要能服务，第一件事不是调拓扑，而是**把 prefill 的归约从共享内存换掉**
（`parallel.DecodeShmCombine` 按行数分路：decode 走共享内存，prefill 走 gloo）。

验收三件事：

1. 四个 rank 的 dense 权重（2.25 GB W4）各自完整、专家 128 个一片；
2. 每层 MoE 的 `combine` 真的把四片专家之和归约回来——判据是四个 rank 的
   `combine_out` 相同，且 `combine_in` **四个 rank 都非零**（各持 1/4 专家）；
3. prefill 与 decode 的毫秒数，以及归约那一路的分项（shm / gloo 分开记）。

## 为什么还要再从 checkpoint 装一遍 bf16 参数

`attach_quantized` 只覆盖**量化产物里有的东西**：`Proj`、`QuantizedEmbedding`、`Experts`。
路由矩阵 `TopKRouter.weight`、各处 `RMSNorm` 的缩放、`hc_norm`、GDN 的 `dt_bias` /
`A_log` 都不在其中，而 `CausalLM` 构造时它们的初值是 `torch.zeros` / `torch.ones`
（不是装载出来的）。

零路由矩阵不是「差一点精度」，是**退化**：`softmax(0)` 在 512 个专家上完全相等，
`topk` 对全等向量返回固定的下标 `0..9`（CPU 与 CUDA 都是取最小下标），于是每个
token 都路由到全局专家 0-9，而它们全部在 rank 0 的分片里（分片起点 `0/128/256/384`）。
rank 1-3 一个本地专家都轮不到，`combine` 的输入恒为 0——这正是
`Q4EXP-PERF-ROUND2.md` §2.4 记的「四卡数值坏了」那条读数。**它不是引擎的 bug，
是探针少装了一半权重**；同一份读数里「首层 MoE 输出四个 rank 各不相同」也对得上：
rank 0 有专家贡献、其余三个只有共享专家。

**这改变了性能分母**：退化路由下 rank 0 一个人干 10 个专家、rank 1-3 完全不干活，
那个 187.72 ms/步 不是服务基线。所以本探针把两档都量出来，让「修正分母」与
「真优化」分开记账：

* ``EP4_ARM.correct``：装齐 dense 参数，每个 rank 约 2-3 个非空专家；
* ``EP4_ARM.degenerate``：在同一个进程里把路由矩阵清零后重跑（只差这一个变量），
  复现旧读数里的「rank 1-3 的 combine_in 恰好为 0」。

## 两个归约字段的口径（不要跨版本相减）

``combine_us_shm`` / ``combine_us_gloo`` 是**单次归约的纯载荷耗时**（墙钟，含主机侧一次
D2H 与一次 H2D，不含诊断用的额外集合通信）。``combine_ms`` = 单次 shm 微秒 × 每步层数，
即「decode 一步里归约合计多少毫秒」；``combine_ms_gloo`` 是同一个式子的 gloo 版本，
**两者相减才是换共享内存的收益**。早期版本的 ``combine_ms`` 是整个作业累计的毫秒，
两个口径不能相减——这一条写在这里是因为它已经被踩过一次。

## 为什么不用 `parallel.ShmAllReduce` 直接归约 prefill

它的 docstring 自己写了原因：rows=64 时「Python 层 4 次全量拷贝吃掉 188 ms」。
prefill 的载荷是 `[S, 2560]`、decode 是 `[1, 2560]`，两者差了三个数量级，所以按行数
分路：`DecodeShmCombine` 只把 `rows<=1` 送去共享板。

Module: ``coffin/tools``。只在有 4 张卡的机器上跑；由 launcher 起 4 个进程。

Constraints:
    * 不用 CUDA graph、不用 cp.async（``platform/scow/AGENTS.md`` 第 1、5 条）。
    * 归约只在**路由专家**那一路：共享专家是 dense 的，每卡都算了完整一份，把它也
      all-reduce 会让它变成 world 倍——这一点由模型的调用点保证（见 ``moe.py``）。
    * 四个 rank 必须按同一顺序调用集合通信。任何 rank 提前退出都会让其余 rank 卡在
      归约上，所以入口用 try/finally 做 destroy；共享板的自旋上限到点抛
      ``TimeoutError``，不会永久空转。
    * 摘要行里的每个值都必须能被 ``184_bench.sh`` 的 ``eval`` 读回，所以非有限的
      浮点要转成字符串，见 :func:`_num`。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

import torch

_ENGINE = Path(__file__).resolve().parents[1]
if str(_ENGINE) not in sys.path:
    sys.path.insert(0, str(_ENGINE))

from models.qwen4exp.checkpoint import StreamingSource  # noqa: E402
from models.qwen4exp.model import causal_mask  # noqa: E402
from models.qwen4exp.moe import TopKRouter  # noqa: E402
from models.qwen4exp.parallel import (  # noqa: E402
    DecodeShmCombine, prepare_shm_boards,
)
from models.qwen4exp.weights import load_host_ngram_table, load_into, read_config  # noqa: E402
from tools.probe_q4_attrib import (  # noqa: E402
    attach_quantized, build_config, build_model, decode_once, log, prefill,
)
# 下划线开头是有意的：``_NarrowSource`` 是**补装 bf16 参数时收窄源**的唯一实现。
# 这条规则一度在两个探针里各写了一份（``_EngineSource`` 与 ``_NarrowSource``），
# 迟早会分叉，而分叉的症状是「一边装得上、一边装不上」，看起来像权重坏了。
from tools.probe_q4_attrib import _NarrowSource  # noqa: E402

def _num(value: float, digits: int = 6):
    """把浮点收拾成 ``eval`` 能读回的形态。

    ``184_bench.sh`` 用 ``eval``（空 ``__builtins__``）解析摘要行，``nan`` / ``inf``
    会让它抛 ``NameError`` 并把整段结果丢掉——症状是「这一轮什么都没打」，而不是
    「数值坏了」。所以非有限值改成字符串。

    Args:
        value: 待收拾的浮点。
        digits: 保留几位。

    Returns:
        有限值时是 ``float``，否则是 ``"nan"`` / ``"inf"`` / ``"-inf"``。
    """
    value = float(value)
    if not math.isfinite(value):
        return "nan" if value != value else ("inf" if value > 0 else "-inf")
    return round(value, digits)


def _broken_arm(name: str, exc: BaseException) -> dict:
    """某一格跑失败时的占位，让摘要仍然打出来。

    数字位一律填 ``-1`` 而不是 0：0 会被读成「这一步不花钱」，而 -1 一眼就知道是失败。
    上一轮（作业 189）退化格在最后一步抛了异常，结果**前面已经量到的三格一个都没
    打出来**，整轮作业白花。

    Args:
        name: 档名。
        exc: 捕获到的异常。

    Returns:
        与 :func:`run_arm` 同键的结果字典。
    """
    return {"name": name, "shm_rows": -1, "decode_ms": -1.0, "prefill_ms": -1.0,
            "tokens": [], "text": f"<failed: {exc!r}>", "layers": [],
            "combine_in": [], "combine_out": [], "nonzero_layers": -1,
            "zero_layers": [], "rows": [], "shm_us": -1.0, "gloo_us": -1.0,
            "shm_calls": 0, "gloo_calls": 0}


def load_dense_leftovers(model, model_dir: Path, device: torch.device,
                         layers: int) -> dict:
    """把量化产物没覆盖的 bf16 参数（路由矩阵、各 norm、hc_norm、GDN 常数）补齐。

    Args:
        model: 已挂好量化产物的模型。
        model_dir: 源 checkpoint 目录。
        device: 目标设备。
        layers: 模型实际层数，用来裁掉源里剩下的层。

    Returns:
        ``load_into`` 的账目：``loaded`` / ``skipped`` / ``ngram`` / ``quantized``。

    Raises:
        KeyError: 源里缺模型需要的张量。缺一个路由矩阵就会让整台机器回到退化路由，
            所以这里必须抛，不能吞。
    """
    source = _NarrowSource(StreamingSource(model_dir, device="cpu"),
                           set(model.state_dict()))
    return load_into(model, source, model.config, max_layers=layers,
                     skip_experts=True,
                     prepare=lambda _name, tensor: tensor.to(device=device,
                                                             dtype=torch.bfloat16))


def load_tokenizer(model_dir: Path):
    """从 checkpoint 目录装一个 HF ``tokenizers`` 分词器。

    验收要求「生成的是一段通顺英文」，那就必须真分词、真还原；拿 ``range(2, 10)``
    当 prompt 只能验形状，验不了"输出正常"。

    Args:
        model_dir: checkpoint 目录，里面有 ``tokenizer.json``。

    Returns:
        分词器；``tokenizers`` 不在或文件缺失时返回 ``None``（此时调用方退回按 id
        的数字 prompt，并把这件事写进日志）。
    """
    try:
        from tokenizers import Tokenizer
    except ImportError:
        return None
    path = Path(model_dir) / "tokenizer.json"
    if not path.exists():
        return None
    return Tokenizer.from_file(str(path))


def text_of(tokenizer, ids: list[int]) -> str:
    """把 id 还原成文本，供验收时肉眼核对。

    Args:
        tokenizer: :func:`load_tokenizer` 的返回值，可以为 ``None``。
        ids: token id 列表。

    Returns:
        文本；没有分词器时退回收到的 id 列表本身。
    """
    if tokenizer is None:
        return f"<ids>{ids}"
    return tokenizer.decode(ids, skip_special_tokens=True)


def tag_combine_layers(model, combine) -> list:
    """给每层 MoE 挂 pre-hook，把层号写进 ``combine.tag``。

    combine 的调用点在层的内部，它自己不知道层号；没有层号的 `combine.trace` 只能
    说「某三层的输入是 0」，说不出**是哪三层**——`QWEN4-PARALLEL.md` §4 把这条列为
    缺口的正是它。钩子必须在前向之前就装好（两档对照都在它之后跑），整轮只装一次。

    Args:
        model: 模型。
        combine: 归约闭包；``None`` 时什么都不装。

    Returns:
        需要调用方在收尾时 ``remove()`` 的钩子列表。
    """
    if combine is None:
        return []
    handles = []
    for idx, layer in enumerate(model.model.layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue

        def mark(_m, _i, layer_idx=idx):  # noqa: ANN001 - hook 签名
            combine.tag = layer_idx

        handles.append(mlp.register_forward_pre_hook(mark))
    return handles


def tap_layers(model, tokens: list[int], device: torch.device,
               ple_tables: dict | None, batch: int = 1) -> dict:
    """跑一次带输出打点的前向，返回各段激活的 absmax。

    Args:
        model: 模型。
        tokens: prompt token。
        device: 设备。
        ple_tables: host n-gram 表。
        batch: 行数；与 ``--batch`` 一致，这样打点覆盖的是被测的那条路径。

    Returns:
        打点字典：``embed`` / ``layer0`` / ``layer0_mlp`` / 可选 ``ple1``。
    """
    taps: list[dict[str, float]] = []
    handles = []

    def tap(name: str, module) -> None:
        """给一个模块挂 hook，把每次调用的输出 absmax 追加到当前这一份快照里。"""

        def hook(_m, _i, output, name=name):  # noqa: ANN001 - torch hook 签名
            value = output[0] if isinstance(output, tuple) else output
            if not taps or name in taps[-1]:
                taps.append({})
            taps[-1][name] = _num(float(value.abs().max()), 4)

        handles.append(module.register_forward_hook(hook))

    tap("embed", model.model.embed_tokens)
    tap("layer0", model.model.layers[0])
    tap("layer0_mlp", model.model.layers[0].mlp)
    if len(model.model.layers) > 1 and model.model.layers[1].ple is not None:
        tap("ple1", model.model.layers[1].ple)
    try:
        prefill_rows(model, tokens, batch, device, ple_tables, causal_mask)
    finally:
        for handle in handles:
            handle.remove()
    return taps[0] if taps else {}


def prefill_rows(model, tokens: list[int], batch: int, device: torch.device,
                 ple_tables: dict | None, masks) -> tuple[torch.Tensor, object]:
    """把同一段 prompt 复制 ``batch`` 行跑 prefill。

    ``batch=1`` 时与 ``probe_q4_attrib.prefill`` 逐位等价（同一个 ``causal_mask``
    形状、同一个 ``conv_mask``），所以单行口径不会因为批支持而被改掉。

    Args:
        model: 模型。
        tokens: prompt token（各行共用，静态批的前提）。
        batch: 行数。
        device: 设备。
        ple_tables: host n-gram 表。
        masks: :func:`causal_mask`，由调用方传进来避免每次重新 import。

    Returns:
        ``(logits[B, 1, V], states)``。
    """
    ids = torch.tensor([tokens] * batch, dtype=torch.long, device=device)
    mask = masks(batch, len(tokens), 0).to(device)
    conv = torch.ones((batch, len(tokens)), dtype=torch.bool, device=device)
    return model(input_ids=ids, attention_mask=mask, conv_mask=conv,
                 ple_tables=ple_tables, states=None, logits_to_keep=1)


def decode_rows(model, row_tokens: list[int], states, past: int, batch: int,
                device: torch.device, ple_tables: dict | None,
                masks) -> tuple[torch.Tensor, object]:
    """跑一步 ``[B, 1]`` 的 decode。

    每行喂自己的 token：静态批里各行已经生成的内容不同，共用一个 token 只在第一步
    成立。``past`` 是各行共同的历史长度（等长 prompt 的静态批，这点成立）。

    Args:
        model: 模型。
        row_tokens: 长度 ``batch``，每行这一步的输入 token。
        states: 上一步留下的状态。
        past: 已处理过的 token 数。
        batch: 行数。
        device: 设备。
        ple_tables: host n-gram 表。
        masks: :func:`causal_mask`。

    Returns:
        ``(logits[B, 1, V], new_states)``。
    """
    ids = torch.tensor(row_tokens, dtype=torch.long, device=device).view(batch, 1)
    mask = masks(batch, 1, past).to(device)
    conv = torch.ones((batch, 1), dtype=torch.bool, device=device)
    return model(input_ids=ids, attention_mask=mask, conv_mask=conv,
                 states=states, ple_tables=ple_tables, logits_to_keep=1)


def bench_decode(model, tokens: list[int], device: torch.device,
                 ple_tables: dict | None, gen: int, batch: int = 1) -> dict:
    """跑「prefill + 若干 decode 步」，给出每步毫秒与各行生成 token。

    单独成一个函数是为了让几个对照臂走**同一段代码**：分开写两份的话，「量了几步」
    与「取哪一步的 logits」这两处很容易只在一处改对。

    Args:
        model: 模型。
        tokens: prompt token（各行共用）。
        device: 设备。
        ple_tables: host n-gram 表。
        gen: 生成多少个 token（含 prefill 后的第一个）。
        batch: 行数；``1`` 时走与单行完全相同的路径。

    Returns:
        ``{"decode_ms", "tokens", "rows", "prefill_ms", "logits"}``：
        ``tokens`` 是首行的生成序列（单行时就是全部），``rows`` 是每一步**每行**的
        token（用来核对批内各行是否一致），``logits`` 是 prefill 末位那一片
        ``[B, 1, V]``，供跨 rank 一致性比较。
    """
    started = time.perf_counter()
    logits, states = prefill_rows(model, tokens, batch, device, ple_tables, causal_mask)
    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - started) * 1e3
    row_tokens = [int(v) for v in logits[:, -1].float().argmax(dim=-1)]
    out = [row_tokens[0]]
    rows = [list(row_tokens)]
    steps: list[float] = []
    for step in range(max(0, gen - 1)):
        t0 = time.perf_counter()
        step_logits, states = decode_rows(model, row_tokens, states,
                                          len(tokens) + step, batch, device,
                                          ple_tables, causal_mask)
        torch.cuda.synchronize()
        steps.append((time.perf_counter() - t0) * 1e3)
        row_tokens = [int(v) for v in step_logits[:, -1].float().argmax(dim=-1)]
        rows.append(list(row_tokens))
        out.append(row_tokens[0])
    return {"decode_ms": _num(sum(steps) / len(steps) if steps else 0.0, 2),
            "tokens": out, "rows": rows, "prefill_ms": _num(prefill_ms, 2),
            "logits": logits}


def diag_router(model, tokens: list[int], device: torch.device) -> dict:
    """看一眼路由矩阵本身：全零就是「rank 1-3 专家路为 0」的根因。

    Args:
        model: 模型。
        tokens: prompt token，用来算一遍真实的路由结果。
        device: 设备。

    Returns:
        ``{"gate_absmax", "gate_zero", "top1", "identity_pick", "local_hits",
        "norm_absmax"}``。
    """
    gate = model.model.layers[0].mlp.gate
    experts = model.model.layers[0].mlp.experts
    ids = torch.tensor([tokens], dtype=torch.long, device=device)
    with torch.no_grad():
        flat = model.model.embed_tokens(ids).reshape(-1, model.config.hidden_size)
        _, _, chosen = gate(flat)
    local = experts.local_index(chosen) >= 0
    # 只看 "weight" 且名字带 norm 的那些，跳过还在 meta 上的（量化路径下不该有，
    # 但真有了就读不出来，``float(meta.abs().max())`` 会在这里炸得莫名其妙）。
    norms = [value for name, value in model.named_parameters()
             if name.endswith("weight") and "norm" in name
             and value.device.type != "meta"]
    norm_absmax = max((float(value.abs().max()) for value in norms), default=0.0)
    # 退化的指纹是「每个 token 都恰好选中下标 0..top_k-1」，不是「10 个专家里最小
    # 等于最大」——后者对正常路由也恒为假，判不出来。
    top_k = int(chosen.shape[-1])
    identity = [int(v) for v in chosen[0].tolist()] == list(range(top_k))
    return {"gate_absmax": _num(float(gate.weight.abs().max())),
            "gate_zero": bool(float(gate.weight.abs().max()) == 0.0),
            "top1": [int(v) for v in chosen[0].tolist()],
            "identity_pick": bool(identity),
            "local_hits": int(local.sum()),
            "norm_absmax": _num(norm_absmax, 4)}


def diag_layer_trace(model, tokens: list[int], device: torch.device,
                     ple_tables: dict | None, runs: int = 3) -> dict:
    """逐层 absmax 与「同输入连跑几遍」的逐层差，定位第一个发散的层号。

    跑三遍而不是两遍：两遍只能回答"一不一致"，三遍能回答**是第一次调用特殊、还是每次
    都特殊** —— 前者指向未初始化缓冲/暖机（``ab≠0`` 而 ``bc=0``），后者才是真不确定。

    Args:
        model: 模型。
        tokens: prompt token。
        device: 设备。
        ple_tables: host n-gram 表。
        runs: 跑几遍。固定三遍，``ab`` 与 ``bc`` 各需要一对。

    Returns:
        ``{"first_div", "first_div_bc", "gaps_ab", "gaps_bc", "amps", "amp_last"}``。
    """
    snaps: list[dict[int, torch.Tensor]] = [{} for _ in range(runs)]
    which = [0]
    handles = []

    def make(idx: int):
        """给第 ``idx`` 层造一个记录 hook，写进当前那一遍的快照里。"""

        def hook(_m, _i, output):  # noqa: ANN001 - torch hook 签名
            value = output[0] if isinstance(output, tuple) else output
            snaps[which[0]][idx] = value.detach().float()

        return hook

    for idx, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(make(idx)))
    try:
        for run in range(runs):
            which[0] = run
            prefill(model, tokens, device, ple_tables)
    finally:
        for handle in handles:
            handle.remove()
    ab = {idx: _num(float((snaps[0][idx] - snaps[1][idx]).abs().max()))
          for idx in snaps[0]}
    bc = {idx: _num(float((snaps[1][idx] - snaps[2][idx]).abs().max()))
          for idx in snaps[1]}
    amps = {idx: _num(float(snaps[0][idx].abs().max()), 4) for idx in snaps[0]}
    div_ab = [idx for idx in sorted(ab) if float(ab[idx]) != 0.0]
    div_bc = [idx for idx in sorted(bc) if float(bc[idx]) != 0.0]
    return {"first_div": div_ab[0] if div_ab else -1,
            "first_div_bc": div_bc[0] if div_bc else -1,
            "gaps_ab": {idx: ab[idx] for idx in div_ab[:6]},
            "gaps_bc": {idx: bc[idx] for idx in div_bc[:6]},
            "amps": {idx: amps[idx] for idx in list(amps)[:6]},
            "amp_last": amps[max(amps)] if amps else 0.0}


def diag_no_combine_repeat(model, tokens: list[int], device: torch.device,
                           ple_tables: dict | None, batch: int = 1) -> dict:
    """关掉跨 rank 归约再跑两遍：仍然不同就说明漂移不在通信上。

    单卡路径没有集合通信，所以这一档是「通信 vs 计算」的分水岭：有漂移去查内核与
    缓冲，为 0 就去查归约那一路（含 ``copy_`` 与 dtype 收口）。

    Args:
        model: 模型。
        tokens: prompt token。
        device: 设备。
        ple_tables: host n-gram 表。
        batch: 行数；与 ``--batch`` 一致，这样比的是被测的那条路径。

    Returns:
        ``{"gap"}``：两遍 logits 的 max|d|。
    """
    saved = model.model.combine
    model.model.combine = None
    try:
        first = prefill_rows(model, tokens, batch, device, ple_tables,
                             causal_mask)[0].float().clone()
        second = prefill_rows(model, tokens, batch, device, ple_tables,
                              causal_mask)[0].float().clone()
    finally:
        model.model.combine = saved
    return {"gap": _num(float((first - second).abs().max()))}


def diag_deterministic(model, tokens: list[int], device: torch.device,
                       ple_tables: dict | None) -> dict:
    """打开确定性模式跑两遍，并收集 torch 点名的「没有确定性实现」的算子。

    用 ``warn_only=True`` 而不是直接抛：直接抛会在第一个 ``index_add_`` 上就断（它本身
    已被证明可复现），后面什么都拿不到。警告文案会**点名**是哪个算子，那正是要找的东西。

    Args:
        model: 模型。
        tokens: prompt token。
        device: 设备。
        ple_tables: host n-gram 表。

    Returns:
        ``{"gap", "warned"}``：两遍 logits 的 max|d|，与警告里出现的算子名。
    """
    # 这个环境变量要在 CUDA 上下文建立之前设才生效；这里设只是让下一次进程启动生效。
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            first = prefill(model, tokens, device, ple_tables)[0].float().clone()
            second = prefill(model, tokens, device, ple_tables)[0].float().clone()
    finally:
        torch.use_deterministic_algorithms(False)
    named = sorted({str(item.message).split(",")[0][:110] for item in caught})
    return {"gap": _num(float((first - second).abs().max())), "warned": named[:6]}


def diag_index_add(device: torch.device, hidden: int = 2560,
                   dtype: torch.dtype = torch.bfloat16) -> dict:
    """重复下标的 ``index_add_`` 是否可复现：MoE 把 top_k 个槽位加回 token 靠它。

    这是本机**唯一**一处浮点原子累加，仓库记录的正常不确定度是 1.7e-07。量它的原因
    是要把「1.7e-07 那档噪声」与「2.17 那档两个不同输入的差」区分开。

    Args:
        device: 设备。
        hidden: 归约宽度。
        dtype: 累加 dtype，与模型激活一致。

    Returns:
        ``{"spread_bf16", "spread_fp32", "same"}``：重复下标累加 N 次的最大逐元素
        跨度，以及是否逐位相同。
    """
    rows, rounds = 10, 32
    src = torch.randn(rows, hidden, dtype=dtype, device=device) * 8.0
    index = torch.zeros(rows, dtype=torch.long, device=device)
    sums = []
    for _ in range(rounds):
        dst = torch.zeros(1, hidden, dtype=dtype, device=device)
        dst.index_add_(0, index, src)
        sums.append(dst)
    stacked = torch.stack(sums).float()
    spread = float((stacked.max(dim=0).values - stacked.min(dim=0).values).max())
    same = bool(torch.equal(sums[0], sums[-1]))
    return {"spread_bf16": _num(spread), "same": same}


def main(argv: list[str] | None = None) -> int:
    """跑验收与诊断，打印可 grep 的结论行。

    Args:
        argv: 命令行参数。

    Returns:
        ``0`` 正常完成。
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--quant-root", required=True,
                        help="含 rank-ep4-{0..3} 的产物根目录")
    parser.add_argument("--world", type=int, default=4)
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--prompt-len", type=int, default=8)
    parser.add_argument("--gen", type=int, default=4)
    parser.add_argument("--skip-ple", action="store_true",
                        help="不装 host n-gram 表（只量通信与权重路径时用）")
    parser.add_argument("--no-dense", action="store_true",
                        help="不从 checkpoint 补 bf16 参数，复现退化路由那一档")
    parser.add_argument("--diag", action="store_true",
                        help="打印路由矩阵、逐层发散、index_add 可复现性")
    parser.add_argument("--prompt-text", default="",
                        help="真 prompt 文本；给了就按 checkpoint 的分词器切分，"
                             "并把生成物还原成文本贴进日志")
    parser.add_argument("--shm-path", default="",
                        help="共享板文件前缀；默认按 MASTER_PORT 派生，作业之间不串")
    parser.add_argument("--shm-rows", type=int, default=1,
                        help="行数不超过它就走共享内存板；decode 恒为 1")
    parser.add_argument("--batch", type=int, default=1,
                        help="每 rank 的批行数。>1 时用 [B,1] 输入、combine 载荷变成 "
                             "[B, 2560]，共享板自动按 1..B 建")
    args = parser.parse_args(argv)

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", str(args.world)))
    device = torch.device("cuda")
    torch.cuda.set_device(0)

    import torch.distributed as dist

    dist.init_process_group("gloo")
    # 集合通信自检：四个 1 全归约必须是 world。不先验这一条，后面所有"归约看起来
    # 生效了"的推断都建立在一个未验证的假设上。
    probe = torch.ones(4, dtype=torch.float32)
    dist.all_reduce(probe)
    log(f"[rank {rank}] all_reduce 自检: {float(probe[0])}（应为 {world}）")
    if abs(float(probe[0]) - world) > 1e-6:
        raise RuntimeError(f"all_reduce self-check failed on rank {rank}")

    shm_path = args.shm_path or os.path.join(
        tempfile.gettempdir(), f"coffin-ep4-{os.environ.get('MASTER_PORT', '0')}")
    # 批行数决定共享板要建到哪一档：判据是「行数不超过 shm_rows 就走板」，而 B>1 时
    # decode 的载荷是 ``[B, 2560]``，所以上限至少要到 B，否则 B 那一档的板不存在，
    # 归约会退到 gloo（正确但慢 4 倍），而那种退化只会表现为「时间不对」。
    batch = max(int(args.batch), 1)
    boards_max = max(int(args.shm_rows), batch)
    # 清空必须在任何 rank 打开板之前完成，而唯一没有并发者的时刻就是现在：
    # 先由 rank 0 建好并清零，再 barrier，四个 rank 才轮到构造。
    # 建的是 ``1..boards_max`` 的**每一档**：判据是"行数不超过就走上限"，所以
    # ``--shm-rows 8`` 时实际可能出现 1..8 里任何一档，少建一档就是运行中才报错。
    if rank == 0:
        boards = prepare_shm_boards(shm_path, world, range(1, boards_max + 1))
        log(f"[rank 0] 共享板: {boards}")
    dist.barrier()
    combine = DecodeShmCombine(shm_path, rank, world, dist,
                              dtype=torch.bfloat16, shm_rows=0,
                              # 上限取层数：一次 prefill 正好 48 次调用把它填满，于是
                              # decode 那 48 次**一次都不会被记录**。记录本身要在主机上
                              # 做一次求和，摊在 33 µs 的单次读数上是 10% 量级的假象，
                              # 而 decode 的单次微秒正是这一轮的验收数。
                              trace_limit=max(int(args.layers), 8),
                              # 等待打点常开：每次多两次 perf_counter，摊在 33 µs 上是
                              # 千分之几，而「有没有在等最慢 rank」是每一轮都要看的账，
                              # 不该因为漏加 --diag 就只能重跑一次作业。
                              timing_limit=4096)
    summary: dict = {"rank": rank, "world": world}

    try:
        model_dir = Path(args.model_dir)
        quant_root = Path(args.quant_root)
        raw = read_config(model_dir)
        config = build_config(raw, args.layers, keep_ple=not args.skip_ple)
        log(f"[rank {rank}] 建模型: {config.num_hidden_layers} 层, "
            f"{config.num_experts} 专家, 本 rank {config.num_experts // world} 个")
        model = build_model(config, quantized=True, rank=rank, world=world,
                            device=device, combine=combine)
        report = attach_quantized(model, quant_root / f"rank-ep4-{rank}", device,
                                  args.layers)
        log(f"[rank {rank}] 量化挂载: attached={report['attached']} "
            f"missing={report['n_missing']}")
        if report["n_missing"]:
            raise RuntimeError(f"rank {rank} 缺 {report['n_missing']} 个量化条目: "
                               f"{report['missing']}")
        model = model.to(device)

        dense: dict = {"skipped": True}
        if not args.no_dense:
            started = time.perf_counter()
            try:
                dense = load_dense_leftovers(model, model_dir, device, args.layers)
                dense["seconds"] = _num(time.perf_counter() - started, 2)
                log(f"[rank {rank}] bf16 补齐: {dense}")
            except Exception as exc:  # noqa: BLE001 - 装载失败要继续跑完诊断
                # 不吞，但也不当场抛：这一轮作业要同时给出「为什么装不上」与
                # 「退化路由长什么样」，后者正是任务 A 的根因证据。失败会同时出现在
                # 日志这一行、summary 的 ``all_dense``、以及 ``EP4_ROUTER`` 的
                # ``gate_zero`` 里——三处可见，不会变成静默的错数。
                dense = {"error": repr(exc),
                         "seconds": _num(time.perf_counter() - started, 2)}
                log(f"[rank {rank}] bf16 补齐失败，本轮按退化路由继续: {exc!r}")

        ple_tables = None
        if not args.skip_ple:
            t0 = time.perf_counter()
            table = load_host_ngram_table(model_dir, config, device=device)
            layers_with_ple = [idx for idx, layer in enumerate(model.model.layers)
                               if layer.ple is not None]
            ple_tables = {idx: table for idx in layers_with_ple}
            log(f"[rank {rank}] host n-gram 表: 层 {layers_with_ple} "
                f"装载 {time.perf_counter() - t0:.2f}s")

        experts = model.model.layers[0].mlp.experts
        own = list(experts.local_experts or [])
        log(f"[rank {rank}] local_experts: {own[:2]} … {own[-1:]} 共 {len(own)} 个")

        router = diag_router(model, list(range(2, 2 + args.prompt_len)), device)
        log(f"[rank {rank}] 路由: {router}")

        # 两个 rank 必须同时开始前向：归约是同步集合通信，先到的会等。
        dist.barrier()
        tokenizer = load_tokenizer(model_dir)
        if args.prompt_text:
            if tokenizer is None:
                raise RuntimeError(
                    "--prompt-text needs <model-dir>/tokenizer.json; 没有分词器就"
                    "没法保证验收文本可比")
            tokens = list(tokenizer.encode(args.prompt_text).ids)
        else:
            tokens = list(range(2, 2 + args.prompt_len))
        log(f"[rank {rank}] prompt ids({len(tokens)}): {tokens} "
            f"text={text_of(tokenizer, tokens)!r}")
        arms: dict[str, dict] = {}
        # 层号钩子在整个对照期间都留着：四个格子都要按层号记账。
        marks = tag_combine_layers(model, combine)

        def run_arm(name: str, shm_rows: int) -> dict:
            """清空摘要后跑一档，返回该档的毫秒数与归约账目。

            Args:
                name: 档名，原样写回结果。
                shm_rows: 这一档的 shm 判据。``0`` 表示强制全部走 gloo，那就是
                    「换成共享内存之前」的那一格。

            Returns:
                该档的 ``decode_ms`` / ``prefill_ms`` / ``tokens`` / 生成文本、按层
                的归约输入与输出和，以及这一档在 shm 与 gloo 上各花的单次微秒。
            """
            saved_rows = combine.shm_rows
            combine.shm_rows = shm_rows
            try:
                combine.trace.clear()
                combine.timing.clear()
                combine.tag = -1
                shm_before, gloo_before = combine.millis_shm, combine.millis_gloo
                calls_before = (combine.calls_shm, combine.calls_gloo)
                measured = bench_decode(model, tokens, device, ple_tables,
                                        args.gen, batch)
                timing = list(combine.timing)
            finally:
                combine.shm_rows = saved_rows
            trace = list(combine.trace)[:max(1, args.layers)]
            shm_calls = combine.calls_shm - calls_before[0]
            gloo_calls = combine.calls_gloo - calls_before[1]
            inputs = [_num(t[2], 3) for t in trace]
            outs = [_num(t[3], 3) for t in trace]
            # 判「整层为零」要用**未舍入**的原值：0.0004 舍入到三位也是 0.0，而这里
            # 要区分的是「恰好 0.0」（专家路真的没算）与「小值」（算对了只是互相抵消）。
            zeros = [t[0] for t in trace if float(t[2]) == 0.0]
            # 只留前几层与"整层为零"的层号：48 层三个数一铺就是 4 KB 一行，日志尾部
            # 会被它挤掉，而判据只需要「有没有零层」与「零在哪几层」。
            return {"name": name,
                    "shm_rows": shm_rows,
                    "decode_ms": measured["decode_ms"],
                    "prefill_ms": measured["prefill_ms"],
                    "tokens": measured["tokens"],
                    "tokens_by_row": measured["rows"],
                    "text": text_of(tokenizer, measured["tokens"]),
                    "layers": [t[0] for t in trace[:8]],
                    "combine_in": inputs[:8],
                    "combine_out": outs[:4],
                    "nonzero_layers": len(inputs) - len(zeros),
                    "zero_layers": zeros[:12],
                    "rows": sorted({t[1] for t in trace}),
                    "shm_us": _num(1e3 * (combine.millis_shm - shm_before)
                                   / max(shm_calls, 1), 2),
                    "gloo_us": _num(1e3 * (combine.millis_gloo - gloo_before)
                                    / max(gloo_calls, 1), 2),
                    "shm_calls": shm_calls,
                    "gloo_calls": gloo_calls,
                    # 归约等待打点原样带走：跨 rank 的「到达时刻极差」要等 gather 之后
                    # 才能算，因为那需要四个 rank 的时间戳。
                    "timing": timing,
                    # 立刻拷一份：lm_head 的输出如果是被复用过的缓冲，留着引用会在
                    # 后面几次前向之后读到别人的数值，而"两次前向一样"就成了自证。
                    "logits": measured["logits"].float().clone()}

        # 四格账：路由（正/零）× 归约（shm/gloo）。**前两格是硬收益，后两格是分母
        # 修正**，两笔不能相加成"优化"：退化路由下 rank 0 一个人干 10 个专家、
        # rank 1-3 完全不干活，那个数不是服务基线。
        # 「shm」那一格用 boards_max 而不是 --shm-rows：B>1 时 decode 的载荷是 B 行，
        # 判据必须覆盖它，否则批路径会静默退回 gloo（正确但慢数倍）。
        arms["correct_shm"] = run_arm("correct_shm", boards_max)
        arms["correct_gloo"] = run_arm("correct_gloo", 0)
        logits = arms["correct_shm"].pop("logits")
        arms["correct_gloo"].pop("logits", None)
        for name in ("correct_shm", "correct_gloo"):
            log(f"[rank {rank}] {name}: decode {arms[name]['decode_ms']} ms, "
                f"text={arms[name]['text']!r}")

        taps = tap_layers(model, tokens, device, ple_tables, batch)
        log(f"[rank {rank}] 打点: {taps}")

        # 同一进程、同一输入、同一权重，**紧挨着**跑三遍再比。旧写法拿 decode 循环
        # 覆盖掉的 ``logits`` 去减一次 prefill 的结果，比的是两个不同输入，2.17 那个
        # "同进程两次前向不一致"就是这么来的。
        # 每次调用完立刻拷一份快照：如果某个上游缓冲被复用，留引用会让两次比对的是
        # 同一块内存，差值恒为 0——那是自证，不是判据。
        # 三遍而不是两遍：``ab`` 与 ``bc`` 分开报，"第一次特殊"与"每次都特殊"是两种病。
        snaps = [prefill_rows(model, tokens, batch, device, ple_tables,
                              causal_mask)[0].float().clone() for _ in range(3)]
        gap_ab = float((snaps[0] - snaps[1]).abs().max())
        gap_bc = float((snaps[1] - snaps[2]).abs().max())
        self_repeat = gap_ab
        repeat_report = {"ab": _num(gap_ab), "bc": _num(gap_bc),
                         "absmax": _num(float(snaps[2].abs().max()), 4)}
        print(f"EP4_REPEAT3 {rank} " + repr(repeat_report), flush=True)
        log(f"[rank {rank}] 同输入三遍 logits 差: ab={gap_ab:.6f} bc={gap_bc:.6f}")

        # 关掉归约再跑两遍：单卡路径没有集合通信，这一档分清「通信」与「计算」。
        no_combine = diag_no_combine_repeat(model, tokens, device, ple_tables, batch)
        print(f"EP4_REPEAT_NO_COMBINE {rank} " + repr(no_combine), flush=True)
        log(f"[rank {rank}] 关归约两遍差: {no_combine['gap']}")

        # 归约是不是空操作：把 combine 摘掉再跑一遍同一个 prompt，比较 logits absmax。
        # 两臂都是 **prefill 同一个 prompt**，才有可比性。
        with_absmax = float(snaps[2].abs().max())
        model.model.combine = None
        logits_plain = prefill_rows(model, tokens, batch, device, ple_tables,
                                   causal_mask)[0]
        model.model.combine = combine
        without_absmax = float(logits_plain.abs().max())
        log(f"[rank {rank}] logits absmax: 有归约 {with_absmax:.4f} / "
            f"无归约 {without_absmax:.4f}")

        layer_trace: dict = {}
        index_add: dict = {}
        if args.diag:
            layer_trace = diag_layer_trace(model, tokens, device, ple_tables)
            index_add = diag_index_add(device)
            log(f"[rank {rank}] 逐层: {layer_trace}")
            log(f"[rank {rank}] index_add: {index_add}")

        # 退化格：只清零路由矩阵，别的都不动。这样"两档之差"只归因于路由本身，
        # 而不是"少装了一半权重"这件事的混合效应。
        #
        # 必须包 ``no_grad``：``TopKRouter.weight`` 是 requires_grad 的叶参数，原地
        # ``zero_()`` / ``copy_()`` 在 grad 模式下抛 "a leaf Variable that requires grad
        # is being used in an in-place operation"。作业 189 就是死在这一行，而它是最
        # 后一步，于是前面三格的数一个都没打出来。
        gates = [module for module in model.modules() if isinstance(module, TopKRouter)]
        saved = [module.weight.detach().clone() for module in gates]
        try:
            with torch.no_grad():
                for module in gates:
                    module.weight.zero_()
            arms["degenerate_shm"] = run_arm("degenerate_shm", boards_max)
            arms["degenerate_gloo"] = run_arm("degenerate_gloo", 0)
        except Exception as exc:  # noqa: BLE001 - 对照格失败不能连累主结果
            # 这一格是**对照**不是验收项：挂了也要把前三格打出来。
            log(f"[rank {rank}] 退化格失败: {exc!r}")
            for name in ("degenerate_shm", "degenerate_gloo"):
                arms.setdefault(name, _broken_arm(name, exc))
        finally:
            with torch.no_grad():
                for module, value in zip(gates, saved):
                    module.weight.copy_(value)
            for name in ("degenerate_shm", "degenerate_gloo"):
                arms[name].pop("logits", None)
        for handle in marks:
            handle.remove()
        for name in ("degenerate_shm", "degenerate_gloo"):
            log(f"[rank {rank}] {name}: decode {arms[name]['decode_ms']} ms, "
                f"text={arms[name]['text']!r}")

        # 每一行的 logits 头都收一份：跨 rank 一致性要按行比，只看第 0 行会漏掉
        # 「只有某一行的 batch 路径算错」这种情形。
        probe_logits = logits[:, -1].float()
        head_rows = [[_num(v, 4) for v in probe_logits[row, :8]]
                     for row in range(int(probe_logits.shape[0]))]
        signature = {
            "head": head_rows,
            "absmax": _num(float(logits.abs().max()), 4),
            "top5": [int(v) for v in probe_logits[0].topk(5).indices],
            "own_first": int(own[0]) if own else -1,
            "with_absmax": _num(with_absmax, 4),
            "without_absmax": _num(without_absmax, 4),
            "taps": taps,
            "router": router,
            "dense": dense,
            "prompt": text_of(tokenizer, tokens),
            "self_repeat": _num(self_repeat),
            "self_repeat_bc": repeat_report["bc"],
            "repeat_no_combine": no_combine,
            "arms": arms,
            "layer_trace": layer_trace,
            "index_add": index_add,
        }
        sigs: list[dict] = [None] * world  # type: ignore[list-item]
        dist.all_gather_object(sigs, signature)
        dist.barrier()

        if rank == 0:
            # 跨 rank 一致性按**每一行的 logits 头**比：只看第 0 行会漏掉「只有某一行
            # 的批路径算错」这种情形，而那正是批支持最可能出的错。
            spread = max(abs(head[row][0] - sigs[0]["head"][row][0])
                         for head in (s["head"] for s in sigs)
                         for row in range(len(head)))

            def stats(values: list[float]) -> list[float]:
                """给一串微秒数报 ``[min, median, max]``，空表返回空表。"""
                if not values:
                    return []
                ordered = sorted(values)
                return [_num(ordered[0], 1), _num(ordered[len(ordered) // 2], 1),
                        _num(ordered[-1], 1)]

            def by_layer(values: list[float], layers: int) -> list[float]:
                """按「层号 = 序号 % 层数」聚成每层的中位数。

                一次 decode 步里 48 次归约按层号顺序发生，所以第 ``i`` 次 decode 归约
                属于第 ``i % 48`` 层；每层因此有 ``步数`` 个样本，取中位数代表该层。
                """
                buckets: list[list[float]] = [[] for _ in range(layers)]
                for index, value in enumerate(values):
                    buckets[index % layers].append(value)
                return [_num(sorted(item)[len(item) // 2], 1) if item else 0.0
                        for item in buckets]

            # 归约等待打点：``timing`` 里每项是
            # ``(tag, rows, 路, 到达时刻, 等待µs, 到达前µs)``；``路==0`` 是共享板（decode），
            # ``路==1`` 是 gloo。等待＝到达之后到 barrier 松开，那段就是**等最慢 rank**。
            timings = [s["arms"]["correct_shm"]["timing"] for s in sigs]
            width = min(len(item) for item in timings) if timings else 0
            decode_idx = [i for i in range(width) if timings[0][i][2] == 0]
            gloo_idx = [i for i in range(width) if timings[0][i][2] == 1]
            waits = {r: [timings[r][i][4] for i in decode_idx] for r in range(world)}
            pres = {r: [timings[r][i][5] for i in decode_idx] for r in range(world)}
            spread_us = [1e6 * (max(timings[r][i][3] for r in range(world))
                                - min(timings[r][i][3] for r in range(world)))
                         for i in decode_idx]
            print("EP4_WAIT " + repr({
                "calls": len(decode_idx),
                "gloo_calls": len(gloo_idx),
                # 每个 rank 自己花在等待上的时间（µs）
                "wait_us": {r: stats(waits[r]) for r in range(world)},
                # 到达之前那一段＝等本层设备算完（阻塞 D2H），它不属于通信
                "pre_us": {r: stats(pres[r]) for r in range(world)},
                # 四个 rank 的**到达时刻极差**：这才是「等最慢 rank」的真实幅度
                "arrive_spread_us": stats(spread_us),
                "wait_by_layer": by_layer(waits[0], int(args.layers)),
                "spread_by_layer": by_layer(spread_us, int(args.layers)),
                "gloo_wait_us": stats([timings[0][i][4] for i in gloo_idx]),
            }), flush=True)

            def grid(arm: str) -> dict:
                """把四格里某一格的四个 rank 读数摊平。"""
                return {"decode_ms": [s["arms"][arm]["decode_ms"] for s in sigs],
                        "text": [s["arms"][arm]["text"] for s in sigs]}

            print("EP4_GRID " + repr({
                # 四格：路由（正/零）× 归约（shm/gloo）。前两格之差是硬收益，
                # 上下两行之差是分母修正。
                "correct_shm": grid("correct_shm"),
                "correct_gloo": grid("correct_gloo"),
                "degenerate_shm": grid("degenerate_shm"),
                "degenerate_gloo": grid("degenerate_gloo"),
                "shm_us": [s["arms"]["correct_shm"]["shm_us"] for s in sigs],
                "gloo_us": [s["arms"]["correct_gloo"]["gloo_us"] for s in sigs],
                "prompt": sigs[0]["prompt"],
            }), flush=True)
            print("EP4_ARM_CORRECT " + repr({
                "decode_ms": [s["arms"]["correct_shm"]["decode_ms"] for s in sigs],
                "tokens": [s["arms"]["correct_shm"]["tokens"] for s in sigs],
                "tokens_by_row": [s["arms"]["correct_shm"]["tokens_by_row"]
                                  for s in sigs],
                "text": [s["arms"]["correct_shm"]["text"] for s in sigs],
                "nonzero_layers": [s["arms"]["correct_shm"]["nonzero_layers"]
                                   for s in sigs],
                "zero_layers": [s["arms"]["correct_shm"]["zero_layers"] for s in sigs],
                "combine_in": [s["arms"]["correct_shm"]["combine_in"] for s in sigs],
                "combine_out": [s["arms"]["correct_shm"]["combine_out"] for s in sigs],
                "rows": [s["arms"]["correct_shm"]["rows"] for s in sigs],
            }), flush=True)
            print("EP4_ARM_DEGENERATE " + repr({
                "decode_ms": [s["arms"]["degenerate_shm"]["decode_ms"] for s in sigs],
                "text": [s["arms"]["degenerate_shm"]["text"] for s in sigs],
                "nonzero_layers": [s["arms"]["degenerate_shm"]["nonzero_layers"]
                                   for s in sigs],
                "zero_layers": [s["arms"]["degenerate_shm"]["zero_layers"]
                                for s in sigs],
                "combine_in": [s["arms"]["degenerate_shm"]["combine_in"] for s in sigs],
                "tokens": [s["arms"]["degenerate_shm"]["tokens"] for s in sigs],
            }), flush=True)
            print("EP4_ROUTER " + repr({
                "gate_absmax": [s["router"]["gate_absmax"] for s in sigs],
                "gate_zero": [s["router"]["gate_zero"] for s in sigs],
                "top1": [s["router"]["top1"] for s in sigs],
                "identity_pick": [s["router"]["identity_pick"] for s in sigs],
                "local_hits": [s["router"]["local_hits"] for s in sigs],
                "norm_absmax": [s["router"]["norm_absmax"] for s in sigs],
            }), flush=True)
            print("EP4_SUMMARY " + repr({
                "prefill_ms": arms["correct_shm"]["prefill_ms"],
                "decode_ms": arms["correct_shm"]["decode_ms"],
                "tok_per_s": _num(1000.0 / max(float(arms["correct_shm"]["decode_ms"]),
                                               1e-9), 2),
                "combine_calls": combine.calls,
                "combine_calls_shm": combine.calls_shm,
                "combine_calls_gloo": combine.calls_gloo,
                # 一档 decode 步里的归约合计：单次微秒 × 每步的层数。两个口径并列给，
                # 因为「换共享内存省了多少」只能是这两条式子相减，而不是和历史上的
                # 「整个作业累计毫秒」相减。
                "combine_ms": _num(arms["correct_shm"]["shm_us"] * args.layers / 1e3, 3),
                "combine_ms_gloo": _num(arms["correct_gloo"]["gloo_us"]
                                        * args.layers / 1e3, 3),
                "combine_us_shm": arms["correct_shm"]["shm_us"],
                "combine_us_gloo": arms["correct_gloo"]["gloo_us"],
                "combine_ms_total": _num(combine.millis, 2),
                "batch": batch,
                "tokens": arms["correct_shm"]["tokens"],
                "tokens_by_row": arms["correct_shm"]["tokens_by_row"],
                "text": arms["correct_shm"]["text"],
                "prompt": text_of(tokenizer, tokens),
                "logits_head_spread": _num(spread),
                "all_top5": [s["top5"] for s in sigs],
                "all_absmax": [s["absmax"] for s in sigs],
                "all_own_first": [s["own_first"] for s in sigs],
                "with_vs_without": [(s["with_absmax"], s["without_absmax"])
                                    for s in sigs],
                "all_taps": [s["taps"] for s in sigs],
                "combine_in": arms["correct_shm"]["combine_in"],
                "combine_rows": arms["correct_shm"]["rows"],
                "combine_layers": arms["correct_shm"]["layers"],
                "self_repeat": _num(self_repeat),
                "self_repeat_bc": repeat_report["bc"],
                "repeat_no_combine": [s["repeat_no_combine"] for s in sigs],
                "all_self_repeat": [s["self_repeat"] for s in sigs],
                "all_combine_in": [s["arms"]["correct_shm"]["combine_in"]
                                   for s in sigs],
                "all_combine_out": [s["arms"]["correct_shm"]["combine_out"]
                                    for s in sigs],
                "all_decode_ms": [s["arms"]["correct_shm"]["decode_ms"]
                                  for s in sigs],
                "all_shm_calls": [s["arms"]["correct_shm"]["shm_calls"]
                                  for s in sigs],
                "all_gloo_calls": [s["arms"]["correct_shm"]["gloo_calls"]
                                   for s in sigs],
                "all_dense": [s["dense"] for s in sigs],
                "layer_trace": layer_trace,
                "index_add": index_add,
            }), flush=True)
        else:
            log(f"[rank {rank}] prefill {arms['correct_shm']['prefill_ms']} ms, "
                f"decode {arms['correct_shm']['decode_ms']} ms")

        # 放在**摘要之后**：它要开确定性模式，万一某个算子在 warn_only 下行为异常，
        # 也伤不到已经打出来的结论行。它自己也是立刻打印的，所以单独炸了也留得下。
        determinism = diag_deterministic(model, tokens, device, ple_tables)
        print(f"EP4_DETERMINISM {rank} " + repr(determinism), flush=True)
    finally:
        combine.close()
        dist.destroy_process_group()
    print("EP4_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

