"""定位"批路径与单行不等价"发生在哪一层 / 哪一类内核。

`Q4EXP-ROUND3.md` §11.3 量到：同一段 prompt 复制成 B 行后，logits 与单行基线差
180–324 个 bf16 量化步长。两个可能的来源，处置完全不同：

1. **行间串扰**（批维处理错）——判据是同一批内两行（逐元素相同的输入）输出不同。
   第一次跑这个探针，8 层上每层的行间差都是 **0.000e+00**，所以这条排除掉了。
2. **批大小依赖的内核路径**——B=1 走 `linear`/`delta`，B>1 走 `linear_m`/`delta_batch`，
   两者算术不同；48 层的混合模型会把这点差放大（`Q4EXP-PERF-ROUND2.md` §2.3 量到
   放大 9.2x）。这条用 ``--compare-single`` 逐层定位。

Module: ``coffin/tools``。只在有卡的机器上跑。

Constraints:
    * 不用 CUDA graph、不用 cp.async。
    * 行间比较时两行喂同一段 token：任何"行间不同"都只能来自批维处理。
"""

from __future__ import annotations

import argparse
import sys
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


def main(argv: list[str] | None = None) -> int:
    """逐层比较行间差，可选比较 B=1 与 B=2 的同一行。

    Args:
        argv: 命令行参数。

    Returns:
        ``0`` 正常完成。
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--rank-dir", required=True)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world", type=int, default=4)
    parser.add_argument("--prompt-len", type=int, default=8)
    parser.add_argument("--skip-ple", action="store_true")
    parser.add_argument("--compare-single", action="store_true",
                        help="额外跑一遍 B=1，逐层比较 B=2 的 row0 与它（找批大小依赖）")
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

    #: 每层一行：``(层号, 层类型, 行间差, 该层 absmax, row0 的 float 副本)``。
    records: list[tuple[int, str, float, float, torch.Tensor]] = []
    for idx, layer in enumerate(model.model.layers):
        def hook(module, inputs, output, idx=idx):  # noqa: ANN001 - torch hook 签名
            hidden = output[0] if isinstance(output, tuple) else output
            rows = hidden.float()
            records.append((idx, getattr(module, "layer_type", "?"),
                            float((rows[0] - rows[-1]).abs().max()),
                            float(rows.abs().max()), rows[0].detach().cpu()))

        layer.register_forward_hook(hook)

    tokens = list(range(2, 2 + args.prompt_len))

    def run_once(batch: int) -> torch.Tensor:
        """按给定批大小跑一遍 prefill，返回末位 logits 的 float 副本。"""
        ids = torch.tensor([tokens] * batch, dtype=torch.long, device=device)
        mask = causal_mask(batch, len(tokens), 0).to(device)
        conv = torch.ones((batch, len(tokens)), dtype=torch.bool, device=device)
        with torch.no_grad():
            logits, _ = model(input_ids=ids, attention_mask=mask, conv_mask=conv,
                              ple_tables=ple_tables, logits_to_keep=1)
        return logits[:, -1].float()

    if args.compare_single:
        records.clear()
        one = run_once(1)
        first = list(records)
        records.clear()
        two = run_once(2)
        second = list(records)
        print("\n| 层 | 类型 | B=1 vs B=2(row0) | 该层 absmax | 相对 |")
        print("| ---: | --- | ---: | ---: | ---: |")
        for before, after in zip(first, second):
            gap = float((before[4] - after[4]).abs().max())
            scale = float(before[4].abs().max())
            print(f"| {before[0]} | {before[1]} | {gap:.3e} | {scale:.3e} | "
                  f"{gap / max(scale, 1e-12):.2e} |")
        gap = float((one[0] - two[0]).abs().max())
        print(f"\nlogits（row0）B=1 vs B=2: {gap:.3e}  absmax={float(one[0].abs().max()):.3e}")
        print("SINGLE_COMPARE_DONE")
        print("BATCH_DIVERGENCE_DONE")
        return 0

    records.clear()
    logits = run_once(2)
    print("\n| 层 | 类型 | 行间最大差 | 该层 absmax | 相对 |")
    print("| ---: | --- | ---: | ---: | ---: |")
    first_bad = None
    for idx, kind, gap, scale, _ in records:
        rel = gap / max(scale, 1e-12)
        flag = ""
        if gap > 0 and first_bad is None:
            first_bad = idx
            flag = " ← 首个非零"
        print(f"| {idx} | {kind} | {gap:.3e} | {scale:.3e} | {rel:.2e}{flag} |")
    out = logits
    print(f"\nlogits 行间最大差: {float((out[0] - out[1]).abs().max()):.3e}  "
          f"absmax={float(out[0].abs().max()):.3e}")
    print(f"首个非零层: {first_bad}")
    print("BATCH_DIVERGENCE_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
