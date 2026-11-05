"""四进程 EP4 的真机生成冒烟：每卡一个 rank 分片，MoE 之后 gloo 归约。

## 它验的是什么

`Q4EXP-PERF-ROUND2.md` §4 量到四卡 EP 只有 0.96x，但那条路径的 prefill 归约用的是
共享内存 barrier：`rows=64` 时 **194 ms/次**（48 层 ⇒ 9.3 s），gloo 同形状 730 µs/次。
所以四卡要能服务，第一件事不是调拓扑，而是**把 prefill 的归约从共享内存换掉**。

本探针用 gloo 做那一步归约（decode 与 prefill 同一条路），验收三件事：

1. 四个 rank 的 dense 权重（2.25 GB W4）各自完整、专家 128 个一片；
2. 每层 MoE 的 `combine` 真的把四片专家之和归约回来——判据是**四个 rank 生成的
   token 逐位相同**（少归约或归约错时，各 rank 会给出不同的贪婪 token）；
3. prefill 与 decode 的毫秒数，用来和单进程 162 ms/步那条基线对照。

## 为什么不用 `parallel.ShmAllReduce`

它的 docstring 自己写了原因：rows=64 时「Python 层 4 次全量拷贝吃掉 188 ms」。那是
调用方的开销，不是通信的；换 gloo 之后每层一次 TCP all_reduce 是 0.4–0.7 ms，
48 层 20–35 ms，比 9.3 s 少两个数量级。等引擎侧有了"prefill 单独用一条通信路径"的
能力，再回头考虑共享内存（它 decode 侧 19 µs/次确实更快）。

Module: ``coffin/tools``。只在有 4 张卡的机器上跑；由 launcher 起 4 个进程。

Constraints:
    * 不用 CUDA graph、不用 cp.async（``platform/scow/AGENTS.md`` 第 1、5 条）。
    * 归约只在**路由专家**那一路：共享专家是 dense 的，每卡都算了完整一份，把它也
      all-reduce 会让它变成 world 倍——这一点由模型的调用点保证（见 ``moe.py``）。
    * 四个 rank 必须按同一顺序调用集合通信。任何 rank 提前退出都会让其余 rank 卡在
      ``all_reduce`` 上，所以入口用 try/finally 做 destroy。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

_ENGINE = Path(__file__).resolve().parents[1]
if str(_ENGINE) not in sys.path:
    sys.path.insert(0, str(_ENGINE))

from tools.probe_q4_attrib import (  # noqa: E402
    attach_quantized, build_config, build_model, decode_once, log, prefill,
)
from models.qwen4exp.weights import load_host_ngram_table, read_config  # noqa: E402


class GlooCombine:
    """把模型里那一路路由专家输出做跨 rank 求和。

    模型的 ``combine`` 收的是**设备上**的 ``[tokens, hidden]``（bf16），语义是
    ``all_reduce(SUM)``。gloo 只能算 CPU 张量，所以这里做「落地 → 归约 → 回传」，
    中转缓冲用锁页内存，形状只有 ``[tokens, 2560]``。
    """

    def __init__(self, dist, hidden: int, dtype: torch.dtype) -> None:
        """按最大的 token 数预分配锁页缓冲。

        Args:
            dist: 已初始化的 ``torch.distributed`` 模块。
            hidden: 归约宽度，qwen4_exp 的 ``hidden_size``。
            dtype: 回传给模型的 dtype（与激活一致）。
        """
        self.dist = dist
        self.hidden = hidden
        self.dtype = dtype
        self.host = torch.empty((4096, hidden), dtype=torch.float32, pin_memory=True)
        self.calls = 0
        self.millis = 0.0
        self.trace: list[tuple[float, float]] = []

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """归约一次并返回同形状、同 dtype、同设备的张量。"""
        rows = int(x.shape[0])
        if rows > self.host.shape[0]:
            self.host = torch.empty((rows, self.hidden), dtype=torch.float32,
                                    pin_memory=True)
        buf = self.host[:rows]
        started = time.perf_counter()
        buf.copy_(x.float(), non_blocking=False)
        local_sum = float(buf.sum())
        self.dist.all_reduce(buf, op=self.dist.ReduceOp.SUM)
        if len(self.trace) < 3:
            # 前三次的进出摘要：输入和四个 rank 各不相同是**对的**（各持 1/4 专家），
            # 输出和四个 rank 必须相同。两者都能一眼区分「归约没生效」与「归约生效
            # 但输入本身就不对」。
            self.trace.append((local_sum, float(buf.sum())))
        out = buf.to(x.dtype).to(x.device, non_blocking=False)
        self.millis += (time.perf_counter() - started) * 1e3
        self.calls += 1
        return out


def main(argv: list[str] | None = None) -> int:
    """跑三个阶段的验收并打印一行可 grep 的结论。

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
    args = parser.parse_args(argv)

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", str(args.world)))
    device = torch.device("cuda")
    torch.cuda.set_device(0)

    import torch.distributed as dist

    dist.init_process_group("gloo")
    combine = GlooCombine(dist, 2560, torch.bfloat16)
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

        ple_tables = None
        if not args.skip_ple:
            t0 = time.perf_counter()
            table = load_host_ngram_table(model_dir, config, device=device)
            layers_with_ple = [idx for idx, layer in enumerate(model.model.layers)
                               if layer.ple is not None]
            ple_tables = {idx: table for idx in layers_with_ple}
            log(f"[rank {rank}] host n-gram 表: 层 {layers_with_ple} "
                f"装载 {time.perf_counter() - t0:.2f}s")

        # 三个 rank 必须同时开始前向：gloo 的 all_reduce 是同步集合通信，先到的会等。
        dist.barrier()
        tokens = list(range(2, 2 + args.prompt_len))
        started = time.perf_counter()
        logits, states = prefill(model, tokens, device, ple_tables)
        torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - started) * 1e3
        token = int(logits[0, -1].argmax())
        out = [token]
        steps: list[float] = []
        for step in range(max(0, args.gen - 1)):
            t0 = time.perf_counter()
            logits, states = decode_once(model, token, states, len(tokens) + step,
                                         device, ple_tables)
            torch.cuda.synchronize()
            steps.append((time.perf_counter() - t0) * 1e3)
            token = int(logits[0, -1].argmax())
            out.append(token)
        decode_ms = sum(steps) / len(steps) if steps else 0.0

        summary.update({"prefill_ms": round(prefill_ms, 2),
                        "decode_ms": round(decode_ms, 2),
                        "tokens": out,
                        "combine_calls": combine.calls,
                        "combine_ms": round(combine.millis, 2)})
        # 每个 rank 的 token 都必须逐位相同：这是"四片专家真的被加齐了"的唯一硬判据。
        gathered: list[list[int]] = [None] * world  # type: ignore[list-item]
        dist.all_gather_object(gathered, out)
        agree = all(item == gathered[0] for item in gathered)
        summary["agree"] = agree
        traces: list[list[tuple[float, float]]] = [None] * world  # type: ignore[list-item]
        dist.all_gather_object(traces, combine.trace)
        if rank == 0:
            print("EP4_SUMMARY " + repr({
                "prefill_ms": summary["prefill_ms"],
                "decode_ms": summary["decode_ms"],
                "tok_per_s": round(1000.0 / max(decode_ms, 1e-9), 2),
                "combine_calls": summary["combine_calls"],
                "combine_ms": summary["combine_ms"],
                "tokens": summary["tokens"],
                "agree": agree,
                "combine_in": [round(t[0], 3) for t in combine.trace],
                "combine_out": [round(t[1], 3) for t in combine.trace],
                "all_combine_out": [[round(t[1], 3) for t in item] for item in traces],
            }), flush=True)
        else:
            log(f"[rank {rank}] prefill {prefill_ms:.1f} ms, decode {decode_ms:.2f} ms, "
                f"combine {combine.millis:.1f} ms")
        if not agree:
            raise RuntimeError(f"四个 rank 的 token 不一致: {gathered}")
    finally:
        dist.destroy_process_group()
    print("EP4_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
