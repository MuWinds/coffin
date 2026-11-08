"""按「算子名 + 输入形状/dtype」归因一步 decode 里的胶水算子。

## 为什么要这个口径

48 层 rank-0 的设备账本（`probe_q4_attrib.py --passes profile`）显示一步有 **32668 次内核**，
其中 `aten::copy_` 3006、`aten::_to_copy` 2139、`aten::to` 2034、`aten::mul` 1749、
`aten::cat` 585、`aten::fill_` 507 —— 这些加起来约占内核数的 36%。而调用点归因
（TorchFunctionMode 的 `file:line`）只保留了 top-40，胶水项全被 `index_add_` 挤掉。

形状比调用点更能定位问题：`copy_ [1,2560] bf16` 与 `copy_ [1,40,2560]` 是两个完全不同的来源，
而前者一眼就能认出是「每次投影新建输出」那类。所以这里按 (算子, 形状, dtype) 聚合。

Module: ``coffin/tools``。只在有卡的机器上跑。

Constraints:
    * 不做 stack 归因：`sys._getframe` 在 74577 次调用上太贵（会让量化失真一个量级）。
    * 只统计目标算子集合，其余原样透传，避免把墙钟拉长到失去参考价值。
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import torch

_ENGINE = Path(__file__).resolve().parents[1]
if str(_ENGINE) not in sys.path:
    sys.path.insert(0, str(_ENGINE))

from tools.probe_q4_attrib import (  # noqa: E402
    attach_quantized, build_config, build_model, decode_once, log, prefill,
)
from models.qwen4exp.weights import read_config  # noqa: E402

#: 目标算子：设备账本里占内核数的那些逐元素/搬运类。
TARGET = {
    "copy_", "to", "_to_copy", "contiguous", "mul", "add", "sub", "div", "cat",
    "fill_", "zero_", "silu", "sigmoid", "index_select", "index_add_", "masked_fill",
    "transpose", "view", "reshape", "permute", "narrow", "slice", "expand", "repeat",
    "unsqueeze", "squeeze", "topk", "softmax", "sum", "mean", "amax", "rsqrt",
}

#: 开 ``--sites`` 时按调用点归因的那一组：它们才是真发内核的搬运/转换，
#: 视图类（reshape/unsqueeze/view）不产生内核，按调用点归因没有意义。
SITE_OPS = {
    "copy_", "to", "_to_copy", "contiguous", "cat", "fill_", "index_add_",
    "index_select", "masked_fill", "silu", "sigmoid",
}


def _shape_of(obj) -> str:
    """把参数压成一个短形状串；非张量取类型名。"""
    if isinstance(obj, torch.Tensor):
        return f"{tuple(obj.shape)}{str(obj.dtype).replace('torch.', '')}"
    if isinstance(obj, (list, tuple)):
        return "(" + ",".join(_shape_of(x) for x in obj[:3]) + ")"
    return type(obj).__name__


class GlueCounter(torch.overrides.TorchFunctionMode):
    """按 (算子, 形状签名) 累计次数与主机耗时。

    ``with_sites`` 打开时额外按「算子 + coffin 内两级调用点」聚合并关闭形状键：
    取调用栈只为这一小组算子付钱（几百到几千次），不会像全量归因那样把墙钟拉长一个量级。
    """

    def __init__(self, with_sites: bool = False) -> None:
        """建表。"""
        super().__init__()
        self.count: Counter = Counter()
        self.millis: Counter = Counter()
        self.with_sites = with_sites
        self._depth = 0

    def _site(self) -> str:
        """在调用栈里找最近的两帧 coffin 源码，返回 ``内层 <- 外层``。"""
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
        return " <- ".join(found) if found else "<torch>"

    def __torch_function__(self, func, types, args=(), kwargs=None):  # noqa: D401
        """记录目标算子后原样转交。"""
        kwargs = {} if kwargs is None else kwargs
        name = getattr(func, "__name__", None) or repr(func)
        if name not in TARGET:
            return func(*args, **kwargs)
        if self.with_sites:
            if name not in SITE_OPS:
                return func(*args, **kwargs)
            key = f"{name} {self._site()}"
        else:
            key = f"{name} {_shape_of(args)}"
        self.count[key] += 1
        if self._depth:
            return func(*args, **kwargs)
        self._depth += 1
        started = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            self._depth -= 1
            self.millis[key] += (time.perf_counter() - started) * 1e3


def main(argv: list[str] | None = None) -> int:
    """跑一遍并打印两张表。"""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--quant-root", required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world", type=int, default=4)
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--prompt-len", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--top", type=int, default=70)
    parser.add_argument("--sites", action="store_true",
                        help="按 coffin 调用点归因（只覆盖真发内核的那组算子）")
    args = parser.parse_args(argv)

    device = torch.device("cuda")
    torch.cuda.set_device(0)
    raw = read_config(Path(args.model_dir))
    config = build_config(raw, args.layers, keep_ple=False)
    model = build_model(config, True, args.rank, args.world, device)
    report = attach_quantized(model, Path(args.quant_root) / f"rank-ep4-{args.rank}",
                              device, args.layers)
    log(f"量化挂载: attached={report['attached']} missing={report['n_missing']}")
    model = model.to(device)

    tokens = list(range(2, 2 + args.prompt_len))
    logits, states = prefill(model, tokens, device, None)
    token = int(logits[0, -1].argmax())

    mode = GlueCounter(with_sites=args.sites)
    started = time.perf_counter()
    with mode:
        for step in range(args.steps):
            logits, states = decode_once(model, token, states,
                                         len(tokens) + step, device, None)
            token = int(logits[0, -1].argmax())
    torch.cuda.synchronize()
    wall = (time.perf_counter() - started) * 1e3 / args.steps
    total = sum(mode.count.values())
    print(f"\nGLUE wall_ms={wall:.2f} glue_calls_per_step={total / args.steps:.0f}")

    by_op: Counter = Counter()
    for key, count in mode.count.items():
        by_op[key.split(" ")[0]] += count
    print("\n### 按算子")
    print("| 算子 | 次数/步 |")
    print("| --- | ---: |")
    for name, count in by_op.most_common(30):
        print(f"| `{name}` | {count / args.steps:.0f} |")

    print(f"\n### 按形状（前 {args.top}）")
    print("| 算子 形状 | 次数/步 |")
    print("| --- | ---: |")
    for key, count in mode.count.most_common(args.top):
        print(f"| `{key}` | {count / args.steps:.1f} |")
    print("GLUE_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
