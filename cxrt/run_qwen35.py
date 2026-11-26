"""无 torch 跑一次 Qwen3.5-4B 的贪心解码：``python3 -m cxrt.run_qwen35``。

Module: ``coffin/cxrt``。这条路径的每一步都是「不用真 torch」的：

1. ``cxrt.install`` 把 ``sys.modules['torch']`` 换成壳；
2. ``cxrt.dense_weights`` 把 w8a8 的 int8 权重折成 bf16 稠密表（用 ``np.memmap`` 读）；
3. ``models.qwen35.Qwen35`` 收下这份 ``weights`` 建模型（走模型自己的稠密路径）；
4. ``prefill`` → ``logits`` → ``argmax`` 逐个出 token，最后用 checkpoint 的
   ``tokenizer.json`` 解码。

它是「模型能不能跑」的验收前门：不看吞吐、不建图、不缓存，先把 token 打出来。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

#: 默认 checkpoint；作业脚本用 ``--model`` 覆盖。
DEFAULT_CKPT = Path.home() / "models" / "Qwen3.5-4B-int8-nk"

#: ``<|im_end|>`` 的 id。与 ``run_chat.py`` 里的那个常量同一来源：这个 checkpoint 的
#: 停止 token 是它，写错的表现是回复越过自己的回合还在继续。
EOS = 248044


def main(argv: list[str] | None = None) -> int:
    """入口，返回进程退出码。

    Args:
        argv: 命令行参数；``None`` 取 ``sys.argv[1:]``。

    Returns:
        ``0`` 表示至少产出了一个 token。
    """
    parser = argparse.ArgumentParser(description="无 torch 跑 Qwen3.5-4B")
    parser.add_argument("--model", default=str(DEFAULT_CKPT))
    parser.add_argument("--raw", default="The capital of France is")
    parser.add_argument("-n", "--max-new", type=int, default=4)
    parser.add_argument("--max-seq", type=int, default=256)
    parser.add_argument("--limit-gib", type=float, default=None,
                        help="只折这么多 GiB（形状探针用）")
    args = parser.parse_args(argv)

    import cxrt.install as inst
    inst.install()
    import torch  # noqa: F401 - 取的就是壳

    from tokenizers import Tokenizer

    from cxrt import dense_weights
    from models.qwen35 import Qwen35

    model_dir = Path(args.model)
    print(f"[run] model={model_dir} seq={args.max_seq}", flush=True)

    t0 = time.time()
    sd = dense_weights.load_dense(model_dir, limit_gib=args.limit_gib)
    print(f"[run] dense tensors={len(sd)} in {time.time() - t0:.1f}s", flush=True)

    t1 = time.time()
    model = Qwen35(model_dir, max_seq=args.max_seq, bench=False, weights=sd)
    print(f"[run] built in {time.time() - t1:.1f}s "
          f"layers={len(model.layers)} hidden={model.hidden}", flush=True)

    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    prompt = tok.encode(args.raw).ids
    print(f"[run] prompt tokens={len(prompt)}", flush=True)

    model.reset()
    t2 = time.time()
    logits = model.prefill(prompt)
    print(f"[run] prefill {len(prompt)} tok in {time.time() - t2:.1f}s", flush=True)

    out: list[int] = []
    t3 = time.time()
    for _ in range(args.max_new):
        nxt = int(logits.argmax().item())
        if nxt == EOS:
            break
        out.append(nxt)
        logits = model.logits(nxt)
    text = tok.decode(out, skip_special_tokens=False)
    print(f"[run] decode {len(out)} tok in {time.time() - t3:.1f}s", flush=True)
    print(f"[text] {text!r}", flush=True)
    print(f"[ids] {out}", flush=True)
    return 0 if out else 2


if __name__ == "__main__":                                  # pragma: no cover - 入口
    sys.exit(main())
