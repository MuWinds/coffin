"""单机入口：加载一个 checkpoint，贪心生成，打印结果。

这是一张卡上一个请求的调试前门——没有服务端、没有调度器、没有分页缓存。它存在的
目的是回答「这次构建还能不能产出文本、有多快」，那是改完内核之后要跑的检查；服务
路径是 ``entrypoints/cli/runtime.py``，任何真实部署用的都是它。

    python run_chat.py --raw "The capital of France is" -n 32
    python run_chat.py --chat "introduce yourself" -n 64
    python run_chat.py --bench -n 128

Module: ``coffin/run_chat.py``。导入图的最顶层，与运行时 CLI 平级：它 import
``engine_api``（面向模型的表层接口），从 checkpoint 目录读分词器，没有任何引擎
模块反向导入它。模型选择交给 ``create_model``，所以这个脚本从不点名模型族——
由 checkpoint 的 ``config.json`` 决定。

Constraints:
    * 解码是贪心的，``argmax`` 不带采样器。同一个 prompt 因此每次运行都可复现，
      那正是冒烟测试的意义；``--bench`` 的数字要和早先的运行比较，带采样器就没法比。
    * ``--bench`` 和 ``--no-graph`` 改变的是被测量的路径本身，不只是输出：解码
      循环要么重放捕获图，要么即时调用 ``logits``，两种成本不同。
    * 这里没有任何东西是线程安全或可重入的。一个进程、一个模型、一次一个请求；
      并发是运行时服务的职责。
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from engine_api import Model, create_model, describe

#: 部署默认值；engine_api 里注册过的 checkpoint 都能用。
MODEL = os.environ.get("ENGINE_MODEL", "/root/private_data/models/Qwen3.5-4B")


def build_prompt(tok: Tokenizer, args) -> list[int]:
    """把 CLI 的 ``--raw``/``--chat``/``--system`` 变成 prompt token id。

    标签是用分词器编码的，不是当字面量拼进去：``<|im_start|>``/``<|im_end|>`` 在这个
    词表里是单个特殊 token，字符串拼接会把它们当成普通文本切分。

    Args:
        tok: checkpoint 的分词器。
        args: 解析后的 CLI 参数；``raw`` 绕过模板，``chat`` 和 ``system`` 填充模板。

    Returns:
        prompt 的 id，末尾是打开的 ``assistant`` 回合，生成因此是接着 assistant
        回复往下写，而不是新起一条消息。

    Raises:
        AttributeError: ``raw`` 和 ``chat`` 都没给；得到的 ``None`` 会在拼接时失败。
            缺 prompt 是 CLI 用法错误，在这里失败胜过给模型送一个空 prompt。
    """
    if args.raw:
        return tok.encode(args.raw).ids
    im_start = tok.encode("<|im_start|>").ids
    im_end = tok.encode("<|im_end|>").ids
    ids: list[int] = []
    if args.system:
        ids += im_start + tok.encode("system\n" + args.system).ids + im_end + tok.encode("\n").ids
    ids += im_start + tok.encode("user\n" + args.chat).ids + im_end + tok.encode("\n").ids
    ids += im_start + tok.encode("assistant\n").ids
    return ids


@torch.inference_mode()
def generate(model: Model, tok: Tokenizer, prompt: list[int], n: int, eos: int,
             graph: bool = True) -> tuple[list[int], float, float]:
    """预填充 ``prompt``，然后贪心解码最多 ``n`` 个 token。

    两条解码路径，每次调用时决定：``decode_step`` 只在图捕获之后才存在，所以第一次
    调用会捕获它，其后的每个 token 都重放它。``graph=False`` 时每一步都即时调用
    ``logits``——更慢，但会跑到捕获所掩盖的那段代码，这也是 ``--no-graph`` 作为诊断
    开关存在的理由。

    Args:
        model: 已加载的模型；它的状态会被消费并留在脏状态。
        tok: 该 checkpoint 的分词器，留给要用它解码的调用方。
        prompt: 要预填充的 prompt id。
        n: 最多生成多少个 token，含 ``eos``。
        eos: 停止 id；生成在它之前结束，不会把它吐出来。
        graph: 模型能提供捕获图时就用它。

    Returns:
        ``(tokens, prefill_s, decode_s)``：生成的 id（不含停止 token），以及两个
        分开上报的墙钟时间——预填充受限的改动和解码受限的改动在合并数字里看起来
        一模一样。
    """
    model.reset()
    if graph and not model.graph_ready:
        model.capture()
        model.reset()
    step = model.decode_step if (graph and model.graph_ready) else model.logits

    t0 = time.time()
    logits = model.prefill(prompt)       # 对整个 prompt 做批处理 GEMM
    t_prefill = time.time() - t0

    out: list[int] = []
    t1 = time.time()
    for _ in range(n):
        nxt = int(logits.argmax())
        if nxt == eos:
            break
        out.append(nxt)
        logits = step(nxt)
    dt = time.time() - t1
    return out, t_prefill, dt


def main() -> None:
    """解析 CLI、加载 checkpoint，然后打印回复或者计时。

    ``--bench`` 跑一个固定 prompt 并报告吞吐而不是文本，连续的构建因此是在同一份
    输入上比较。

    Raises:
        OSError: checkpoint 目录缺 ``tokenizer.json`` 或权重；把 ``--model`` 指向一个
            完整的 checkpoint。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--raw", help="raw prompt (no chat template)")
    ap.add_argument("--chat", help="user message (chat template)")
    ap.add_argument("--system", default=None)
    ap.add_argument("-n", "--max-new", type=int, default=64)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--max-seq", type=int, default=4096)
    ap.add_argument("--no-bench", action="store_true",
                    help="skip the backend's load-time kernel selection (its own autotune)")
    ap.add_argument("--no-graph", action="store_true",
                    help="disable the device-graph decode path and run eager")
    args = ap.parse_args()

    tok = Tokenizer.from_file(str(Path(args.model) / "tokenizer.json"))

    # 这个 checkpoint 的 ``<|im_end|>``。服务端会从分词器推导停止集合，这个冒烟测试
    # 则把它需要的那个 id 写死：这里写错的表现是回复越过自己的回合还在继续，而那
    # 正是该被注意到的回归。
    eos = 248044

    print("loading weights ...", flush=True)
    t0 = time.time()
    model = create_model(args.model, max_seq=args.max_seq, bench=not args.no_bench)
    info = describe(args.model)
    print(f"loaded {info['model']} ({info['arch']}) in {time.time()-t0:.1f}s", flush=True)
    graph = not args.no_graph

    if args.bench:
        # 故意用固定 prompt：token 数是上报 tok/s 的一部分，prompt 一变，连续几次
        # 运行就没法比较了。
        prompt = tok.encode("The quick brown fox").ids
        out, tp, dt = generate(model, tok, prompt, args.max_new, eos, graph)
        print(f"prefill: {len(prompt)} tokens in {tp*1000:.0f}ms ({len(prompt)/tp:.0f} tok/s)")
        print(f"decode : {len(out)} tokens in {dt:.2f}s = {len(out)/dt:.2f} tok/s"
              f"  (graph={graph and model.graph_ready})")
        return

    prompt = build_prompt(tok, args)
    print(f"prompt tokens: {len(prompt)}")
    out, tp, dt = generate(model, tok, prompt, args.max_new, eos, graph)
    text = tok.decode(out, skip_special_tokens=False)
    print("output:")
    print(text)
    print(f"prefill {len(prompt)} tok in {tp*1000:.0f}ms ({len(prompt)/tp:.0f} tok/s); "
          f"decode {len(out)} tok in {dt:.2f}s ({len(out)/dt:.2f} tok/s)"
          f"  (graph={graph and model.graph_ready})")


if __name__ == "__main__":
    main()
