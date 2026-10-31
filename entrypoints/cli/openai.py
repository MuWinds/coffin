"""OpenAI 兼容 HTTP 适配器的命令行入口。

Module: ``coffin.entrypoints.cli``。只负责解析开关、等常驻引擎就绪、把 HTTP
层挂起来；模型和 KV 都在引擎进程里，不在本进程。

Constraints:
    * 启动时轮询 ``ping``：HTTP 层可能先于引擎起来，读不到引擎信息就没法
      向 ``/v1/models`` 报告模型 id。
    * 引擎是单流的，没有取消通道；``--default-max-tokens`` /
      ``--max-decode-seconds`` 的默认值就是为这种情况兜底的。
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import ThreadingHTTPServer

from entrypoints.openai import app as _app
from entrypoints.openai.client import EngineClient


def main() -> None:
    """等常驻引擎可用，配置好 HTTP 适配器后开始服务。

    先轮询 ``ping`` 拿到引擎信息（模型 id、checkpoint 路径、是否有视觉能力），
    这些是 ``/v1/models`` 和 ``/health`` 的唯一数据来源。

    Raises:
        SystemExit: 连续三次 ping 都失败。此时报错退出比带着空配置起 HTTP
            服务好——后者只会把每个请求都变成 500。
    """
    ap = argparse.ArgumentParser(description="OpenAI-compatible HTTP API in front of "
                                     "the resident engine")
    ap.add_argument("--sock", default=_app.SOCK,
                    help="engine unix socket (default: $ENGINE_SOCK or the runtime default)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--served-model-name", default="",
                    help="model id reported by /v1/models (default: the engine's "
                         "checkpoint directory name)")
    ap.add_argument("--model", default="",
                    help="override the checkpoint path reported by /health "
                         "(default: whatever the resident engine loaded)")
    ap.add_argument("--default-max-tokens", type=int, default=8192,
                    help="output budget when the client states none (0 = the whole "
                         "window). The engine is single-stream with no cancel path, so "
                         "an unbounded request that nobody waits for keeps later "
                         "requests queued until it ends by itself")
    ap.add_argument("--enable-thinking", action="store_true",
                    help="let the model reason by default (per-request override wins)")
    ap.add_argument("--max-decode-seconds", type=float, default=180.0,
                    help="wall-clock cap the engine enforces on one generation "
                         "(0 = none). This is the cutoff that also fires when a client "
                         "goes quiet without closing its socket: the single stream "
                         "would otherwise keep decoding for a request nobody reads, and "
                         "every later request queues behind it")
    ap.add_argument("--timeout", type=float, default=900.0,
                    help="per-request engine timeout, seconds")
    ap.add_argument("--max-images", type=int, default=_app.MAX_IMAGES,
                    help=f"images per request (default {_app.MAX_IMAGES}); the engine "
                         f"re-checks against its own --max-images")
    ap.add_argument("--max-image-bytes", type=int, default=_app.MAX_IMAGE_BYTES,
                    help="decoded bytes allowed per image (default 8 MiB)")
    ap.add_argument("--max-image-total-bytes", type=int,
                    default=_app.MAX_IMAGE_TOTAL_BYTES,
                    help="base64 characters of all image parts of one request "
                         "(default 16 MiB)")
    args = ap.parse_args()

    _app.MAX_IMAGES = int(args.max_images)
    _app.MAX_IMAGE_BYTES = int(args.max_image_bytes)
    _app.MAX_IMAGE_TOTAL_BYTES = int(args.max_image_total_bytes)

    engine = EngineClient(args.sock, timeout=args.timeout)
    for _ in range(3):
        try:
            info = engine.call({"op": "ping"})
            break
        except Exception as exc:  # noqa: BLE001
            print(f"[server] engine not reachable on {args.sock}: {exc}", flush=True)
            time.sleep(2)
    else:
        raise SystemExit(f"engine socket {args.sock} is not answering -- "
                         "start the runtime CLI first")

    served = args.served_model_name or info.get("model") or "default"
    model_path = args.model or info.get("model_path") or ""
    _app.STATE.update({
        "engine": engine,
        "model_path": model_path,
        "served_name": served,
        "enable_thinking": args.enable_thinking,
        "default_max_tokens": args.default_max_tokens or -1,
        "max_decode_seconds": args.max_decode_seconds,
        "vision": bool(info.get("vision", True)),
        "image_limits": {
            "max_images": _app.MAX_IMAGES,
            "max_image_bytes": _app.MAX_IMAGE_BYTES,
            "max_image_total_bytes": _app.MAX_IMAGE_TOTAL_BYTES,
        },
        "created": int(time.time()),
    })
    print(f"[server] engine up: {json.dumps(info, ensure_ascii=False)}", flush=True)
    srv = ThreadingHTTPServer((args.host, args.port), _app.Handler)
    srv.daemon_threads = True
    print(f"[server] listening on http://{args.host}:{args.port}/v1 "
          f"(model id {served}, arch {info.get('arch')}, thinking={args.enable_thinking}, "
          f"max_decode_seconds={args.max_decode_seconds or 'none'}, "
          f"vision={_app.STATE['vision']}, images<={_app.MAX_IMAGES}x"
          f"{_app.MAX_IMAGE_BYTES / 2**20:.0f}MiB)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
