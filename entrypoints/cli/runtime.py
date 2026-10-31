"""常驻推理运行时的命令行客户端。

Module: ``coffin.entrypoints.cli``。参数解析与模型构造隔开，导入本模块不会
连带着构造一个巨大的参数解析器。

Constraints:
    * 这里是唯一解析运行时开关的地方，所以也是 ``--backend`` 唯一能生效的
      地方：后端在 import ``coffin.backends`` 时就已经选定了，开关必须在
      绑定算子表的 serving 模块被导入*之前*写进 ``COFFIN_BACKEND``。下面
      那些 import 之所以延迟到解析之后，就是为此。
    * ``--no-graph`` 不能做成环境变量：这个决定必须在引擎 capture 之前
      传到 ``EngineConfig``，而且不支持 capture 的后端优先级必须高于开关。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from serving.runtime_support import MAX_SEQ, MODEL, SOCK

def main() -> None:
    """命令行入口：要么在本进程里起服务，要么作为客户端去连一个服务。

    ``--serve`` 独占 socket 和模型；其余调用都是薄客户端，只把回包打印出来。
    两种角色不共用代码路径，客户端就不可能顺手把权重加载起来。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("cmd", nargs="?", default="ping",
                    choices=["ping", "reload", "reset", "stop", "generate", "chat",
                             "bench", "profile", "ops", "spec"])
    ap.add_argument("--backend", default="",
                    help="device backend to run on (default: auto-detect, then the "
                         "pure-torch reference). Sets COFFIN_BACKEND; the backend is "
                         "selected once at import, so this flag only takes effect "
                         "with --serve in the same process")
    ap.add_argument("--no-graph", dest="graph", action="store_false",
                    help="run the decode step eagerly instead of replaying a captured "
                         "device graph. Slower per token, and the only safe choice on "
                         "a backend whose graph replay is not trusted; a backend "
                         "declaring no graph support is eager regardless")
    ap.add_argument("--model", default=MODEL,
                    help="checkpoint directory; the architecture is read from its "
                         "config.json (default: $ENGINE_MODEL or the Qwen3.5-4B path)")
    ap.add_argument("--sock", default=SOCK, help="engine socket (default: $ENGINE_SOCK)")
    ap.add_argument("--max-seq", type=int, default=MAX_SEQ,
                    help="context window; the KV cache is allocated up front, so a "
                         "256k window costs ~4.4 GB of VRAM (8 int8 full-attn layers)")
    ap.add_argument("--no-prefix-cache", dest="prefix_cache", action="store_false",
                    help="disable the paged radix prefix cache. It is what makes a "
                         "repeated prefix (an agent resending its conversation) cheap: "
                         "only the unseen suffix is prefilled")
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--snapshots", type=int, default=8,
                    help="recurrent-state snapshot budget (~50 MB each); the join "
                         "points a later request can reuse")
    ap.add_argument("--snapshot-interval", type=int, default=512,
                    help="snapshot every N tokens (rounded to a block)")
    ap.add_argument("--fp16", dest="int8", action="store_false",
                    help="keep the 2D weights in fp16 instead of int8: ~4 GiB more VRAM "
                         "and a slower decode, in exchange for ~0.3%% better perplexity "
                         "(NOTES §6). Default is int8, because the 256k window and the "
                         "weights have to share 16 GB.")
    ap.add_argument("--no-bench", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=0.0,
                    help="hard wall-clock cap on one generation, seconds (0 = none). "
                         "This is the only cancellation that also covers a client that "
                         "goes quiet without closing its socket -- otherwise the single "
                         "stream keeps generating for a request nobody is reading")
    ap.add_argument("--spec-k", type=int, default=0,
                    help="drafts per speculative step (0 = off); the verify graph is "
                         "captured for K+1 tokens")
    ap.add_argument("--spec-batch", action="store_true",
                    help="also speculate in the batched engine. Off by default: on "
                         "this card the batched verify costs one weight pass per "
                         "candidate token, so it is ~0.79x a plain batch decode step "
                         "(output is identical either way)")
    ap.add_argument("--spec-min-ctx", type=int, default=0,
                    help="only speculate in the batched engine once a sequence has "
                         "this many cached tokens (0 = always); below it a plain "
                         "batched decode step is cheaper")
    ap.add_argument("--spec-sparse-k", type=int, default=0,
                    help="approximate batched verify: keep only the top-k historical "
                         "KV blocks per row (drafter lookahead + last attention) plus "
                         "a recent window; 0 = exact full attention. Implies "
                         "--spec-batch; token output is no longer guaranteed equal")
    ap.add_argument("--max-num-seqs", type=int, default=1,
                    help="sequences decoded in one batched step (1 = the original "
                         "single-stream engine). Requests still queue; the scheduler "
                         "admits up to this many and batches their decode steps.")
    ap.add_argument("--num-blocks", type=int, default=0,
                    help="KV pool blocks shared by all sequences (0 = "
                         "max_seq/block_size * max_num_seqs). Lower it when the "
                         "window is long: the pool is allocated up front.")
    ap.add_argument("--prefill-quantum", type=int, default=2048,
                    help="prompt tokens one request may prefill per scheduling step "
                         "(0 = the step's whole token budget). This is what keeps a "
                         "long prompt from blocking a short one; it is not the GEMM "
                         "tile (that is --max-num-batched-tokens' chunk)")

    # 图像相关开关集中在这里，因为它们互相关联：像素上限决定图像 token 数，
    # 而图像 token 数才是真正吃显存和延迟的那一项。
    ap.add_argument("--no-vision", dest="vision", action="store_false",
                    help="do not load the vision tower (saves ~0.62 GiB); image "
                         "requests then answer 503 instead of being served")
    ap.add_argument("--vision-int8", action="store_true",
                    help="keep the vision tower's 2-D weights int8 (0.31 GiB instead "
                         "of 0.62), dequantised one GEMM at a time per request")
    ap.add_argument("--image-max-pixels", type=int, default=1048576,
                    help="largest image the processor will serve, in pixels "
                         "(default 1 MP = ~1024 image tokens). This is the only real "
                         "VRAM/latency knob: anything bigger is downsampled")
    ap.add_argument("--image-min-pixels", type=int, default=65536,
                    help="smallest image after the processor's upscale (official "
                         "value; an image never costs fewer than 64 tokens)")
    ap.add_argument("--max-images", type=int, default=4,
                    help="images accepted in one request (four 1 MP images are "
                         "already ~4k image tokens)")
    ap.add_argument("--max-image-bytes", type=int, default=8388608,
                    help="decoded bytes allowed per image (default 8 MiB)")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--user", default="")
    ap.add_argument("--system", default=None)
    ap.add_argument("--tools", default=None, help="JSON file with an OpenAI tools array")
    ap.add_argument("--think", action="store_true", help="enable thinking mode")
    ap.add_argument("-n", "--max-new", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    # 必须在任何模块绑定算子表之前写入：``coffin.backends`` 在 import 时就选好
    # 后端，之后再赋值不会改变任何东西。
    if args.backend:
        os.environ["COFFIN_BACKEND"] = args.backend
    from serving.runtime_transport import call, serve

    if args.serve:
        serve(args.model, args.max_seq, not args.no_bench, args.sock, args.spec_k,
              args.int8, args.prefix_cache, args.block_size, args.snapshots,
              args.snapshot_interval, args.max_seconds, args.max_num_seqs,
              args.num_blocks, args.spec_batch, args.spec_sparse_k,
              args.spec_min_ctx, args.vision, args.vision_int8,
              args.image_max_pixels, args.image_min_pixels, args.max_images,
              args.max_image_bytes, graph=args.graph,
              prefill_quantum=args.prefill_quantum)
        return

    if args.cmd in ("ping", "reload", "reset", "stop"):
        print(json.dumps(call(args.sock, {"op": args.cmd}), ensure_ascii=False, indent=2))
    elif args.cmd == "spec":
        r = call(args.sock, {"op": "spec", "n": args.max_new, "prompt": args.prompt,
                             "spec_k": args.spec_k or 0})
        print(f"plain {r['plain_tok_s']} tok/s -> spec(K={r['k']}) {r['spec_tok_s']} tok/s  "
              f"= {r['speedup']}x   exact {r['exact']}")
        print(f"accept {r['accept']}")
    elif args.cmd == "bench":
        r = call(args.sock, {"op": "bench", "n": args.max_new, "prompt": args.prompt})
        print(f"decode: {r['ms_per_step']} ms/step  {r['decode_tok_s']} tok/s  (n={r['n']})")
    elif args.cmd == "profile":
        r = call(args.sock, {"op": "profile", "n": args.max_new, "prompt": args.prompt})
        print(f"{'kernel':<50}{'calls':>7}{'us/step':>10}{'%':>7}")
        for row in r["rows"]:
            print(f"{row['kernel']:<50}{row['calls']:>7}{row['us']:>10.1f}{row['pct']:>7.1f}")
        print(f"{'TOTAL':<50}{'':>7}{r['total_us']:>10.1f}")
    elif args.cmd == "ops":
        r = call(args.sock, {"op": "ops", "n": args.max_new, "prompt": args.prompt})
        print(f"{'op':<50}{'calls':>7}{'us/step':>10}{'%':>7}")
        for row in r["rows"]:
            print(f"{row['op']:<50}{row['calls']:>7}{row['us']:>10.1f}{row['pct']:>7.1f}")
        print(f"{'TOTAL':<50}{'':>7}{r['total_us']:>10.1f}")
    elif args.cmd == "generate":
        r = call(args.sock, {"op": "generate", "prompt": args.prompt,
                             "max_new": args.max_new, "temperature": args.temperature})
        print(r["text"])
        print(f"[{r['n_prompt']} prompt tok, {r['n_new']} new, "
              f"prefill {r['prefill_ms']}ms, decode {r['decode_tok_s']} tok/s]")
    elif args.cmd == "chat":
        msgs = []
        if args.system:
            msgs.append({"role": "system", "content": args.system})
        msgs.append({"role": "user", "content": args.user})
        tools = json.loads(Path(args.tools).read_text()) if args.tools else None
        r = call(args.sock, {"op": "chat", "messages": msgs, "tools": tools,
                             "enable_thinking": args.think, "max_new": args.max_new,
                             "temperature": args.temperature})
        if r.get("thinking"):
            print(f"[thinking] {r['thinking']}")
        print(r["text"])
        for c in r["tool_calls"]:
            print(f"[tool_call] {c['function']['name']} {c['function']['arguments']}")
        print(f"[{r['n_prompt']} prompt tok, {r['n_new']} new, "
              f"prefill {r['prefill_ms']}ms, decode {r['decode_tok_s']} tok/s]")


if __name__ == "__main__":
    main()
