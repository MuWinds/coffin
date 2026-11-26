"""连上已经起好的 runtime socket，原样打出发电/对话的响应。

存在的理由：`entrypoints/cli/runtime.py` 的 `generate` / `chat` 两条命令直接读
`r["text"]`，服务端一旦回的是错误字典（`{"ok": false, "error": ...}`）就变成
`KeyError: 'text'`——真正的原因被这个键访问吞掉了。这里不做键假设，把响应逐字段
打出来，所以「为什么没出 token」是能看见的。

用法（服务已起）：
    python tools/probe_occamy_gen.py --sock /tmp/occamy_tp4.sock

输出 ``OCCGEN_JSON_BEGIN..END``，每步一条；失败也照样打出来，退出码仍为 0，
因为要的是证据而不是「命令成功」。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ENGINE = Path(__file__).resolve().parents[1]
if str(_ENGINE) not in sys.path:
    sys.path.insert(0, str(_ENGINE))

from serving.runtime_transport import call  # noqa: E402


def _show(tag: str, payload: dict) -> None:
    """打一条请求的结果，超长字段截断，保住形状与失败原因。"""
    out = {}
    for k, v in payload.items():
        if isinstance(v, str) and len(v) > 400:
            out[k] = v[:400] + f"…({len(v)} chars)"
        elif isinstance(v, list) and len(v) > 16:
            out[k] = v[:16] + [f"…{len(v) - 16} more"]
        else:
            out[k] = v
    print(f"[{tag}] " + json.dumps(out, ensure_ascii=False), flush=True)


def main(argv: list[str] | None = None) -> int:
    """命令行入口：连 socket 跑一句 generate 与一句 chat。

    Args:
        argv: 参数列表；``None`` 取 ``sys.argv[1:]``。

    Returns:
        ``0`` 无论成败（证据都打出来了），``66`` 当 socket 不存在。
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sock", default="/tmp/occamy_tp4.sock")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--repeat", type=int, default=1,
                    help="把 prompt 重复多少遍，用来量长上下文的预填充；实际 token 数看 n_prompt")
    ap.add_argument("--user", default="用一句话介绍你自己")
    ap.add_argument("-n", "--max-new", type=int, default=24)
    ap.add_argument("--ngenerate", type=int, default=1,
                    help="同一个 prompt 连发几次 generate：第一次的 prefill_ms 含内核"
                         "首次加载，稳态要看第二次起")
    ap.add_argument("--repeat-step", type=int, default=1,
                    help="每多发一次把重复次数加这么多，避开前缀缓存命中")
    args = ap.parse_args(argv)

    if not Path(args.sock).exists():
        print(f"no socket: {args.sock}", flush=True)
        return 66

    def _prompt_of(i: int) -> str:
        """第 i 次 generate 用的 prompt。

        前缀缓存会把完全相同的 prompt 认掉 320/321 个 token（``occ15``/``occ17`` 实测：
        第二次起 ``cached_tokens=320``、``prefill_ms`` 只剩 104~204 ms），那就量不到
        稳态的整段预填充。只改末尾还不够——前缀命中的是整个公共前缀，320 个 token 一样
        全中。所以每次在**开头**插一个短标记，整段都成了新 token。
        """
        r = args.repeat + i * args.repeat_step
        body = args.prompt if r <= 1 else (args.prompt + " ") * r
        return body if i == 0 else f"Case {i}. " + body

    print("OCCGEN_JSON_BEGIN", flush=True)
    _show("ping", call(args.sock, {"op": "ping"}))
    # 同一个 prompt 连发多次：第一次的 ``prefill_ms`` 里混着内核首次加载/编译，稳态
    # 要第二次才看得见（``occtp13`` 的层计时显示首个分块 11436 ms、同尺寸的后续分块
    # 338 ms，差的正是首次加载）。``--ngenerate`` 就是给这个 A/B 用的。
    for i in range(max(1, args.ngenerate)):
        _show(f"generate[{i}]", call(args.sock, {"op": "generate",
                                                 "prompt": _prompt_of(i),
                                                 "max_new": args.max_new,
                                                 "temperature": 0.0}))
    _show("chat", call(args.sock, {"op": "chat",
                                   "messages": [{"role": "user", "content": args.user}],
                                   "tools": None, "enable_thinking": False,
                                   "max_new": args.max_new, "temperature": 0.0}))
    print("OCCGEN_JSON_END", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
