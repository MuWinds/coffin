"""批大小对一步 decode 的影响：B=1/2/4/8 的步时间与聚合 tok/s。

## 为什么量这个而不是继续调内核

`Q4EXP-ROUND3.md` §4 的归因：一步 48 层的设备时间里，权重类内核（`w4a8_dot` 24 ms、
`moe_grouped` 19.5 ms）已经跑到 **274 / 360 GB/s**，是屋顶（575）的 48–63%；剩下的大头
是**每个投影一次发射**的固定成本（一步约 1300 次内核）。也就是说：

* 想靠"再快一点的单核"拿到 2x 很难——它已经在带宽的同一量级上；
* 而 **B 条请求共用同一份权重** 时，权重只读一遍，固定成本被摊薄。上一轮（`Q4EXP-PERF-ROUND2.md`
  §5）在旧树上量到 B=8 是 2.79x，这一轮换了融合派发与 host PLE 表，必须重测。

判据（本脚本打印，人工核对）：

1. B=8 的步时间相对 B=1 涨多少（涨不到 2 倍就是强批处理收益）；
2. 聚合 tok/s 随 B 是否单调上升；
3. 每行 logits 与"同一行单独跑"在 bf16 末位内一致——批量路径算错时聚合吞吐照样好看。

Module: ``coffin/tools``。只在有卡的机器上跑。

Constraints:
    * 不用 CUDA graph、不用 cp.async（``platform/scow/AGENTS.md`` 第 1、5 条）。
    * 全批共用同一段 prompt 长度：ModelStates 的批维是稠密张量，不等长请求要分页池，
      那是 M3 的事。这里只回答"批本身值不值"。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

_ENGINE = Path(__file__).resolve().parents[1]
if str(_ENGINE) not in sys.path:
    sys.path.insert(0, str(_ENGINE))

from tools.probe_q4_attrib import (  # noqa: E402
    attach_quantized, build_config, build_model, log,
)
from models.qwen4exp.model import causal_mask  # noqa: E402
from models.qwen4exp.weights import load_host_ngram_table, read_config  # noqa: E402


def batch_prefill(model, tokens: list[int], batch: int, device: torch.device,
                  ple_tables=None):
    """把同一段 prompt 复制 ``batch`` 行跑一次前向。

    Args:
        model: 模型。
        tokens: 单行的 prompt。
        batch: 行数。
        device: 设备。
        ple_tables: host n-gram 表。

    Returns:
        ``(logits, states)``；``logits`` 是 ``[batch, vocab]``。
    """
    seq = len(tokens)
    ids = torch.tensor([tokens] * batch, dtype=torch.long, device=device)
    mask = causal_mask(batch, seq, 0).to(device)
    conv = torch.ones((batch, seq), dtype=torch.bool, device=device)
    logits, states = model(input_ids=ids, attention_mask=mask, conv_mask=conv,
                           ple_tables=ple_tables, logits_to_keep=1)
    return logits, states


def batch_decode(model, logits: torch.Tensor, states, past: int, batch: int,
                 device: torch.device, ple_tables=None):
    """用上一步的 argmax 作为下一步的输入，跑一步 batched decode。

    Args:
        model: 模型。
        logits: 上一步的 ``[batch, vocab]``。
        states: 状态。
        past: 已处理 token 数。
        batch: 行数。
        device: 设备。
        ple_tables: host n-gram 表。

    Returns:
        ``(logits, states)``。
    """
    # ``logits_to_keep=1`` 让 logits 是 ``[B, 1, vocab]``；取 `[:, -1]` 才是
    # ``[B, vocab]``，直接 ``argmax(-1, keepdim=True)`` 会多出一维再喂回模型
    # （报在 ``batch, seq, _ = inputs_embeds.shape`` 的 "too many values to unpack"）。
    next_ids = logits[:, -1].argmax(dim=-1, keepdim=True).to(torch.long)
    mask = causal_mask(batch, 1, past).to(device)
    conv = torch.ones((batch, 1), dtype=torch.bool, device=device)
    return model(input_ids=next_ids, attention_mask=mask, conv_mask=conv,
                 states=states, ple_tables=ple_tables, logits_to_keep=1)


def measure(model, tokens: list[int], batch: int, device: torch.device,
            steps: int, ple_tables=None) -> dict:
    """量一次 batched prefill 与 ``steps`` 步 batched decode。

    Args:
        model: 模型。
        tokens: 单行 prompt。
        batch: 行数。
        device: 设备。
        steps: decode 步数。
        ple_tables: host n-gram 表。

    Returns:
        ``{"prefill_ms", "decode_ms", "tok_per_s", "row_logits"}``。
    """
    started = time.perf_counter()
    logits, states = batch_prefill(model, tokens, batch, device, ple_tables)
    if device.type == "cuda":
        torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - started) * 1e3
    steps_ms: list[float] = []
    for step in range(steps):
        t0 = time.perf_counter()
        logits, states = batch_decode(model, logits, states, len(tokens) + step,
                                      batch, device, ple_tables)
        if device.type == "cuda":
            torch.cuda.synchronize()
        steps_ms.append((time.perf_counter() - t0) * 1e3)
    decode_ms = sum(steps_ms) / len(steps_ms)
    return {"prefill_ms": prefill_ms, "decode_ms": decode_ms,
            "tok_per_s": batch * 1000.0 / decode_ms,
            "row_logits": logits[:, -1].detach().float().cpu()}


def main(argv: list[str] | None = None) -> int:
    """按 B 扫一遍并打印表格。

    Args:
        argv: 命令行参数。

    Returns:
        ``0`` 正常完成。
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--rank-dir", required=True)
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world", type=int, default=4)
    parser.add_argument("--batches", default="1,2,4,8")
    parser.add_argument("--prompt-len", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--skip-ple", action="store_true")
    args = parser.parse_args(argv)

    device = torch.device("cuda")
    model_dir = Path(args.model_dir)
    raw = read_config(model_dir)
    config = build_config(raw, args.layers, keep_ple=not args.skip_ple)
    model = build_model(config, quantized=True, rank=args.rank, world=args.world,
                        device=device)
    report = attach_quantized(model, Path(args.rank_dir), device, args.layers)
    log(f"量化挂载: attached={report['attached']} missing={report['n_missing']}")
    model = model.to(device)

    ple_tables = None
    if not args.skip_ple:
        table = load_host_ngram_table(model_dir, config, device=device)
        ple_tables = {idx: table for idx, layer in enumerate(model.model.layers)
                      if layer.ple is not None}
        log(f"host n-gram 表: 层 {list(ple_tables)}")

    tokens = list(range(2, 2 + args.prompt_len))
    # 单行基线用的是**同样的步数**：不同步数的 logits 本来就不该相等（位置不同），
    # 拿 steps=1 与 steps=3 比会把"位置不同"误判成"批量算错"。
    base = measure(model, tokens, 1, device, args.steps, ple_tables)["row_logits"][0]
    step = float(base.abs().max()) * 2 ** -8
    print("\n| B | prefill ms | decode ms/步 | 聚合 tok/s | 相对 B=1 | 每 token | 行间 Δ | 对单行 Δ |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    first = None
    for batch in [int(v) for v in args.batches.split(",") if v.strip()]:
        got = measure(model, tokens, batch, device, args.steps, ple_tables)
        if first is None:
            first = got["decode_ms"]
        rows = got["row_logits"]
        # 行间一致性：同一段 prompt 的 B 行应给出同一组 logits。判据按 bf16 量化步长
        # 归一化——split-K 的累加次序在各行之间可能有 1 ulp 差异，那是无害的。
        row_gap = float((rows[0] - rows[-1]).abs().max()) / max(step, 1e-9)
        # 与单行基线：同上，按 bf16 步长归一化。
        gap = float((rows[0] - base).abs().max()) / max(step, 1e-9)
        print(f"| {batch} | {got['prefill_ms']:.1f} | {got['decode_ms']:.2f} | "
              f"{got['tok_per_s']:.2f} | {first / got['decode_ms']:.2f}x | "
              f"{got['decode_ms'] / batch / (first / 1):.3f}x | "
              f"{row_gap:.1f} 步 | {gap:.1f} 步 |")
    print("BATCH_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
