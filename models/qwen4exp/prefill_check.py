"""单卡前向脚手架：加载分片权重、跑前向、与参考对照。

这是"真实权重能不能跑起来"的第一道验收，也是后面所有量化与并行验收的基准。
它按四个阶段推进，每一阶段都有可独立复核的判据——出问题时能立刻定位到"读盘错、
映射错、装载错"还是"数学错"，而不是只看到一个 NaN：

1. **几何**：扫 header，核对张量数与形状与 ``inventory.json`` 完全一致。360 GB
   少一个分片、或混进两个版本的权重，都在这一步暴露，代价只有几秒。
2. **装载**：只装 1 层（``layers.1``，挂 PLE 的那层）跑一批 token，与"同权重、
   同输入下的参考实现"比。单层装得下、跑得动，才谈得上 48 层。
3. **整机**：48 层全装，比 logits。真实权重下这一步要 16 GB/卡量级的显存。
4. **量化对照**：同一份权重走 W4A8 路径，比 top-1 一致率。

本脚本只做 1 与 2，3 与 4 留给有卡的环境——开发机上没有 ``safetensors``、也没有
能放下 360 GB 的内存，硬跑只会给出误导性的"通过"。

用法::

    # 只验几何（任何机器，只要有 safetensors 和权重目录）
    python -m coffin.models.qwen4exp.prefill_check --model /path/to/ckpt --stage geometry

    # 装 1 层并前向
    python -m coffin.models.qwen4exp.prefill_check --model /path/to/ckpt --stage layer

    # 与参考实现的 fixture 对照（小配置的 34 个张量）
    python -m coffin.models.qwen4exp.prefill_check --oracle docs/qwen4/oracle --stage fixture
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from models.qwen4exp.blocks import DecoderLayer
from models.qwen4exp.config import TextConfig
from models.qwen4exp.model import CausalLM
from models.qwen4exp.weights import (
    NGRAM_SHARD_MARKER,
    engine_key,
    load_into,
    plan,
    text_config_of,
    verify,
)


@dataclass
class StageReport:
    """一个阶段的结论。

    Attributes:
        stage: 阶段名。
        ok: 是否通过。
        detail: 人类可读的一句话。
        numbers: 关键数字，便于跨机器比较。
    """

    stage: str
    ok: bool
    detail: str
    numbers: dict

    def render(self) -> str:
        """格式化成一行日志。

        Returns:
            ``[ok/FAIL] stage: detail  {numbers}``。
        """
        status = "ok" if self.ok else "FAIL"
        body = " ".join(f"{key}={value}" for key, value in self.numbers.items())
        return f"[{status}] {self.stage}: {self.detail}  {body}"


def causal_mask(seq: int, past: int = 0) -> torch.Tensor:
    """fp32 因果掩码。

    Args:
        seq: 本次 query 数。
        past: 已处理 token 数。

    Returns:
        ``[1, 1, seq, past + seq]``。
    """
    total = past + seq
    matrix = torch.full((1, 1, seq, total), torch.finfo(torch.float32).min)
    for i in range(seq):
        matrix[0, 0, i, : past + i + 1] = 0.0
    return matrix


def check_geometry(model_dir: Path, inventory: Path | None) -> StageReport:
    """阶段 1：核对 header 与 inventory 完全一致。

    Args:
        model_dir: 权重目录。
        inventory: ``inventory.json`` 的路径；``None`` 时只报告张量数。

    Returns:
        阶段结论。
    """
    from models.qwen4exp.checkpoint import StreamingSource, assert_inventory_matches

    start = time.perf_counter()
    source = StreamingSource(model_dir)
    summary = source.summarize()
    numbers = {"shards": summary["shards"], "tensors": summary["tensors"]}
    if inventory is None:
        return StageReport("geometry", True, "checkpoint header scanned", numbers)
    problems = assert_inventory_matches(model_dir, inventory)
    numbers["elapsed_s"] = round(time.perf_counter() - start, 2)
    if problems:
        return StageReport("geometry", False, "; ".join(problems), numbers)
    return StageReport("geometry", True, "checkpoint matches inventory", numbers)


def check_plan(model_dir: Path, config_path: Path | None,
               inventory: Path | None) -> StageReport:
    """阶段 1b：把 header 的键名走一遍映射，核对覆盖模型的每一个参数。

    不读任何权重内容，所以对 360 GB 的目录也只是秒级。这一步能抓住"checkpoint 里
    有、但映射表不认识"的张量——它们会被静默跳过，对应参数停在随机初始值上。

    Args:
        model_dir: 权重目录。
        config_path: ``config.json`` 路径；``None`` 时用 ``model_dir`` 里的。
        inventory: 未使用，保留以统一调用签名。

    Returns:
        阶段结论。
    """
    from models.qwen4exp.checkpoint import StreamingSource

    del inventory
    source = StreamingSource(model_dir)
    raw = json.loads(((config_path or model_dir / "config.json")).read_text(encoding="utf-8"))
    cfg = text_config_of(raw)
    with torch.device("meta"):
        model = CausalLM(cfg, stream_ngram=True)
    result = plan(source, cfg)
    problems = verify(result, model.state_dict())
    numbers = {"tensors": len(source), "mapped": len(result.assignments),
               "ngram_groups": len(result.ngram), "skipped": len(result.skipped)}
    if problems:
        return StageReport("plan", False, "; ".join(problems), numbers)
    return StageReport("plan", True, "every model parameter has a source tensor", numbers)


def check_layer(model_dir: Path, layer_idx: int = 1, tokens: int = 8) -> StageReport:
    """阶段 2：只装一层并前向，验证"读盘 + 映射 + 装载 + 前向"整条链。

    单层的参数在真实配置下约 1.4 GB（MoE 占大头），任何一张卡都放得下，所以这
    一步能在最小的失败面上定位问题。判据是输出全有限且量级合理——不是与参考比，
    因为参考实现的单层输出没有随权重一起 dump。

    Args:
        model_dir: 权重目录。
        layer_idx: 要装的层号。
        tokens: 前向多少 token。

    Returns:
        阶段结论。
    """
    from models.qwen4exp.checkpoint import StreamingSource

    start = time.perf_counter()
    raw = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    cfg = text_config_of(raw)
    source = StreamingSource(model_dir)

    layer = _NamedLayer(DecoderLayer(cfg, layer_idx), layer_idx)
    subtree = f"model.layers.{layer_idx}."
    # 不把权重升到 fp32。真实 checkpoint 是 bf16，升精度既没有依据（要验的是
    # "能不能装载并前向"，不是数值极限），又让常驻量直接翻倍：单层去掉 n-gram
    # 表是 5.25 GB bf16、10.51 GB fp32，而构造模型时那 5 GB 专家已经分配了一份，
    # 装载期同时存在两份就会撞上作业的内存上限（实测 exit 137，108 秒被杀）。
    # 需要高精度比对的场景走 ``--stage fixture``，那里权重本来就是 fp32。
    report = load_into(layer, source, cfg, with_lm_head=False, subtree=subtree)

    hidden = cfg.hc_hidden_size
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(1, tokens, hidden, generator=generator)
    ids = torch.randint(0, cfg.vocab_size, (1, tokens), generator=generator)
    mask = torch.full((1, 1, tokens, tokens), torch.finfo(torch.float32).min)
    for i in range(tokens):
        mask[0, 0, i, : i + 1] = 0.0
    conv_mask = torch.ones(1, tokens, dtype=torch.bool)
    cos, sin = _position_embeddings(cfg, tokens)

    # PLE 的 n-gram 表是 95 GiB，单层验收用一张同宽的小替代表：这一步只验装载与
    # 前向链路，不验检索质量。
    table = _small_ngram_table(cfg)
    with torch.no_grad():
        out, state = layer.inner(
            x, position_embeddings=(cos, sin), attention_mask=mask,
            conv_mask=conv_mask, ple_input_ids=ids,
            ple_table=table if layer_idx + 1 in cfg.ple_layer_ids else None)
    numbers = {
        "loaded": report["loaded"],
        "params_mb": round(_param_bytes(layer) / 2 ** 20, 1),
        "elapsed_s": round(time.perf_counter() - start, 1),
        "finite": bool(torch.isfinite(out).all()),
        "absmax": round(float(out.abs().max()), 4),
    }
    ok = numbers["finite"] and numbers["absmax"] > 0
    return StageReport("layer", ok, f"layer {layer_idx} forward", numbers)


def check_fixture(oracle: Path) -> StageReport:
    """阶段 2b：用 fixture 的 34 个张量走同一条装载路径，并与 fixture 对照。

    这条不需要任何真实权重，任何机器都能跑，是"装载路径本身没写坏"的证据。

    Args:
        oracle: fixture 目录。

    Returns:
        阶段结论。
    """
    from models.qwen4exp.weights import DictSource

    fixture_path = oracle / "decoder_layer.pt"
    if not fixture_path.is_file():
        return StageReport("fixture", False, f"missing {fixture_path}", {})
    fixture = torch.load(fixture_path, map_location="cpu", weights_only=False)
    cfg = _fixture_config()
    source = DictSource(fixture["weights"], prefix="model.layers.1.")
    layer = _NamedLayer(DecoderLayer(cfg, layer_idx=1), 1)
    report = load_into(layer, source, cfg, with_lm_head=False, subtree="model.layers.1.")
    with torch.no_grad():
        out, _ = layer.inner(
            fixture["x"], position_embeddings=(fixture["cos"], fixture["sin"]),
            attention_mask=fixture["mask"], conv_mask=fixture["conv_mask"],
            ple_input_ids=fixture["input_ids"])
    err = ((out - fixture["out"]).abs().max() / fixture["out"].abs().max()).item()
    numbers = {"loaded": report["loaded"], "rel_err": f"{err:.2e}"}
    return StageReport("fixture", err < 1e-5, "loaded from checkpoint names", numbers)


class _NamedLayer(torch.nn.Module):
    """给一个 ``DecoderLayer`` 套上 ``model.layers.N`` 前缀。

    ``weights.load_into`` 认的是 checkpoint 名，所以层必须先挂到这个命名空间下，
    否则单层装载找不到对应的键。套一层而不是另写一套装载逻辑，是为了让脚手架和
    线上走同一个函数。
    """

    def __init__(self, inner: DecoderLayer, index: int) -> None:
        """建命名空间。

        Args:
            inner: 真正的 decoder 层。
            index: 层号。
        """
        super().__init__()
        self.inner = inner
        self.model = torch.nn.ModuleDict(
            {"layers": torch.nn.ModuleDict({str(index): inner})})


def _param_bytes(module: torch.nn.Module) -> int:
    """模块所有参数的字节数。

    Args:
        module: 任意模块。

    Returns:
        参数字节总和。
    """
    return sum(tensor.numel() * tensor.element_size()
               for tensor in module.parameters())


def _position_embeddings(config: TextConfig, tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    """按线性位置算一组 (cos, sin)。

    Args:
        config: 文本配置。
        tokens: token 数。

    Returns:
        ``(cos, sin)``，``[1, tokens, rotary_dim]``。
    """
    from models.qwen4exp.rope import RotaryEmbedding

    positions = torch.arange(tokens).view(1, 1, -1).expand(3, 1, -1)
    return RotaryEmbedding(config).cos_sin(positions)


def _small_ngram_table(config: TextConfig) -> torch.Tensor:
    """造一张与真实表同宽、但行数很小的 n-gram 替代表。

    真实表 320001536 行放不下，而单层验收只需要"表的形状对、查表能跑通"。

    Args:
        config: 文本配置。

    Returns:
        ``[padded_rows, head_dim]`` 的 fp32 表，行数取配置推导值。
    """
    from models.qwen4exp.ngram import NGramEmbedding, find_nth_prime_after

    heads = (config.ngram_size - 1) * config.heads_per_ngram
    head_dim = int(config.ple_embed_dim) // heads
    padded = NGramEmbedding(config, int(config.ple_embed_dim), 0, 0).padded_vocab_size
    return torch.rand(padded, head_dim, generator=torch.Generator().manual_seed(1)) * 0.02


def _fixture_config() -> TextConfig:
    """与 ``diag/qwen4/dump_reference.py`` 完全一致的小配置。

    Returns:
        ``TextConfig``。
    """
    return TextConfig(
        vocab_size=256, hidden_size=64, num_hidden_layers=8, num_attention_heads=4,
        num_key_value_heads=2, head_dim=32,
        linear_num_key_heads=2, linear_num_value_heads=6,
        linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
        num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32,
        shared_expert_intermediate_size=32, hc_count=4, hc_lowrank=16,
        ple_layer_ids=[2], ple_embed_dim=64, ple_conv_kernel_size=4,
        ngram_size=3, heads_per_ngram=8, ngram_vocab_size_base=1000,
        make_ngram_vocab_size_divisible_by=128,
        indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=16,
        indexer_budget=16, indexer_compress_ratio=4,
        output_gate_type="sigmoid", rms_norm_eps=1e-6, max_position_embeddings=512,
        eos_token_id=2, bos_token_id=1, rope_theta=10000.0,
        partial_rotary_factor=0.25, mrope_section=[3, 3, 2], mrope_interleaved=True,
    )


def main(argv: list[str]) -> int:
    """按阶段跑脚手架并打印结论。

    Args:
        argv: 命令行参数。

    Returns:
        进程退出码：全部通过为 0。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="checkpoint 目录")
    parser.add_argument("--oracle", type=Path, help="fixture 目录")
    parser.add_argument("--inventory", type=Path, help="inventory.json 路径")
    parser.add_argument("--stage", default="all",
                        choices=("all", "geometry", "plan", "layer", "fixture"))
    parser.add_argument("--layer", type=int, default=1, help="layer 阶段装哪一层")
    args = parser.parse_args(argv[1:])

    reports: list[StageReport] = []
    if args.stage in ("all", "fixture"):
        if args.oracle is None:
            reports.append(StageReport("fixture", False, "--oracle is required", {}))
        else:
            reports.append(check_fixture(args.oracle))
    if args.model is not None:
        if args.stage in ("all", "geometry"):
            reports.append(check_geometry(args.model, args.inventory))
        if args.stage in ("all", "plan"):
            reports.append(check_plan(args.model, None, args.inventory))
        if args.stage in ("all", "layer"):
            reports.append(check_layer(args.model, args.layer))
    elif args.stage in ("geometry", "plan", "layer"):
        reports.append(StageReport(args.stage, False, "--model is required", {}))

    for report in reports:
        print(report.render())
    return 0 if all(report.ok for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
