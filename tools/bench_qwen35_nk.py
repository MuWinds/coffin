"""量 Qwen3.5-4B 在 nk int8 契约下的稳态吞吐与显存。

``run_chat.py --bench`` 只跑一个 4 token 的 prompt 和一次解码，回答的是「这一步
有没有变慢」；这个脚本回答的是「模型能跑到多少」：预填充按 128 / 512 / 2048 / 4096
四档分别计时（预填充是算力受限、解码是带宽受限，两个数字不能混），每档之后再用同一
个上下文解码若干 token，最后报峰值显存。

    python tools/bench_qwen35_nk.py --model ~/models/Qwen3.5-4B-int8-nk --int8

输出是 ``BENCH_JSON_BEGIN`` / ``BENCH_JSON_END`` 之间的一行 JSON，方便别处解析。

Module: ``coffin/tools``。import ``engine_api`` 与 ``run_chat``（复用它的贪心解码
循环），不反向被引擎导入。

Constraints:
    * 一律 ``graph=False``：当前工作树里 ``ops/norm.py`` 与内核侧的 rms_norm 正在改，
      图捕获在 ``capture() -> prefill -> rms_norm`` 上必挂（``shape '[32, 2560]' is
      invalid for input of size 2560``）。关图测的是「立即可用的这条路」，数字偏保守，
      但只要两次运行都用同一个开关就可以互相比较。
    * 每档之间 ``reset()``：不清状态会把上一档的 KV 带进下一档，预填充的耗时就不再
      是这一档的长度。
    * ``torch.cuda.reset_peak_memory_stats`` 只在加载完成后调一次，所以
      ``peak_bytes`` 是「加载 + 最重的那一次预填充」的峰值，正是 OOM 判据。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from engine_api import create_model, describe

#: 四档预填充长度。512 与 2048 是服务端最常见的两个入站长度；4096 是这张卡上
#: 4B 的 ``max_seq`` 量级，用来划线看 KV 与激活的增长。
PREFILL_LENS = (128, 512, 2048, 4096)

#: 每档之后解码多少 token。取 32：足够摊掉第一次调用的编译与缓存冷启动，
#: 又不至于让一档跑满几十秒。
DECODE_TOKENS = 32


def _fixed_ids(n: int, vocab: int = 248320) -> list[int]:
    """造一段固定、可复现的 prompt。

    只用 ``n % 1024 + 7`` 这类小 id 是为了避开特殊 token 与图像占位符：预填充的
    形状才是被测量的东西，内容不该影响它。
    """
    base = [((i * 37) % 1021) + 7 for i in range(min(n, 1024))]
    out = (base * ((n // len(base)) + 1))[:n]
    assert all(0 <= t < vocab for t in out)
    return out


def main() -> None:
    """按档预填充 + 解码，打印一行 JSON。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--max-seq", type=int, default=4096)
    ap.add_argument("--decode", type=int, default=DECODE_TOKENS)
    ap.add_argument("--lens", default=",".join(str(v) for v in PREFILL_LENS))
    args = ap.parse_args()

    if not Path(args.model).is_dir():
        print(f"no such model dir: {args.model}", flush=True)
        raise SystemExit(66)

    t0 = time.time()
    model = create_model(args.model, max_seq=args.max_seq, bench=False,
                         quant=args.int8)
    load_s = time.time() - t0
    info = describe(args.model)
    torch.cuda.reset_peak_memory_stats()

    result = {
        "model": info["model"],
        "arch": info["arch"],
        "graph": False,
        "load_s": round(load_s, 3),
        "prefill": [],
    }
    eos = 248044
    for n in [int(v) for v in args.lens.split(",") if v.strip()]:
        ids = _fixed_ids(n)
        model.reset()
        t1 = time.time()
        logits = model.prefill(ids)
        dt = time.time() - t1
        decode_s, produced = 0.0, 0
        t2 = time.time()
        for _ in range(args.decode):
            nxt = int(logits.argmax())
            if nxt == eos:
                break
            produced += 1
            logits = model.logits(nxt)
        decode_s = time.time() - t2
        result["prefill"].append({
            "len": n,
            "prefill_ms": round(dt * 1000, 1),
            "prefill_tok_s": round(n / dt, 2),
            "decode_tok_s": round(produced / decode_s, 2) if decode_s > 0 else None,
            "decode_n": produced,
        })
        model.reset()

    result["peak_bytes"] = int(torch.cuda.max_memory_allocated())
    result["peak_gib"] = round(result["peak_bytes"] / 2**30, 2)
    print("BENCH_JSON_BEGIN", flush=True)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    print("BENCH_JSON_END", flush=True)


if __name__ == "__main__":
    main()
