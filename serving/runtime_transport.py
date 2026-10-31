"""常驻推理运行时的 unix socket 传输层。

Module: ``coffin.serving``。socket 的 accept/read/write 行为归本模块；模型生命周期
仍在 ``runtime_service.Server`` 里。

Constraints:
    * 启动横幅是后端静默回退唯一会被看见的地方，所以它原样打印
      ``backends.status()``，而不是自己拼一份字段列表：``backend=``、``device=``、
      ``native=``、``graph=``、``autotune=`` 这些字段就是运维诊断时看的东西，手工
      拼出来的横幅会在后端新增开关的那一刻就开始漂移。
    * 这里不许碰设备：会分配或同步的传输层在没有任何加速卡的主机上根本绑不了
      socket，而参考后端恰恰就是必须在那样的机器上跑通的情况。
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time

import backends

from .runtime_service import Server
from .runtime_support import MMError

def serve(model_dir: str, max_seq: int, bench: bool, sock_path: str,
          spec_k: int = 0, int8: bool = True, prefix_cache: bool = True,
          block_size: int = 64, snapshots: int = 8, snapshot_interval: int = 512,
          max_seconds: float = 0.0, max_num_seqs: int = 1,
          num_blocks: int = 0, spec_batch: bool = False,
          spec_sparse_k: int = 0, spec_min_ctx: int = 0,
          vision: bool = True, vision_int8: bool = False,
          image_max_pixels: int = 1048576, image_min_pixels: int = 65536,
          max_images: int = 4, max_image_bytes: int = 8388608,
          graph: bool = True, prefill_quantum: int = 2048) -> None:
    """建好服务、绑定 unix socket，然后一直服务到被要求停止。

    参数表与 ``Server.__init__`` 一一对应，各开关的含义见那里。这里的一切都是
    进程级的，不是按请求的。

    Args:
        model_dir: checkpoint 目录。
        max_seq: 上下文窗口，prompt 和输出共享。
        bench: 保留旧的单流 bench 路径可用。
        sock_path: 要绑的 unix socket；该路径上已有的文件会先被删掉，一次被杀掉
            的运行留下的陈旧 socket 因此不会挡住启动。
        spec_k: 每个投机步的草稿数，0 表示关闭。
        int8: 把二维权重量化。
        prefix_cache: 启用分页基数前缀缓存。
        block_size: KV 块的 token 粒度。
        snapshots: 循环状态快照预算。
        snapshot_interval: 每 N 个 token 打一次快照，取整到块。
        max_seconds: 一次生成的墙钟上限，0 表示不限。
        max_num_seqs: 批解码步里的行数。
        num_blocks: KV 池大小，0 表示按窗口推算。
        spec_batch: 在批处理引擎里投机。
        spec_sparse_k: 近似的批验证，只用历史块里的 top-k。
        spec_min_ctx: 缓存长度低于它就跳过批投机。
        vision: checkpoint 有视觉塔时加载它。
        vision_int8: 视觉塔的二维权重保持 int8。
        image_max_pixels: 处理器服务的最大图像。
        image_min_pixels: 处理器放大之后的最小图像。
        max_images: 一个请求接受的图像数。
        max_image_bytes: 单张图像允许的解码字节数。
        graph: 把解码步 capture 成设备图。``False`` 改跑 eager 步；这是给
            还不信任图回放的后端留的逃生口，而它必须在图 capture 之前就传到引擎，
            所以它走构造参数，而不是环境变量。
        prefill_quantum: 一个请求每步最多预填充多少 prompt token。调度器用它把长
            prompt 和短 prompt 交错开；0 表示退回该步的全部 token 预算。
    """
    srv_model = Server(model_dir, max_seq, bench, spec_k, int8, prefix_cache, block_size,
                       snapshots, snapshot_interval, max_seconds, max_num_seqs,
                       num_blocks, spec_batch, spec_sparse_k, spec_min_ctx,
                       vision, vision_int8, image_max_pixels, image_min_pixels,
                       max_images, max_image_bytes, graph=graph,
                       prefill_quantum=prefill_quantum)
    info = srv_model.info
    print(f"[serve] model {info['model']} ({info['arch']}) at {info['model_path']}",
          flush=True)
    prefix_on = (srv_model.pc is not None
                 or (srv_model.engine.bm is not None
                     and srv_model.engine.bm.radix is not None))
    print(f"[serve] {backends.status()}", flush=True)
    print(f"[serve] weights {srv_model.load_s:.1f}s + build {srv_model.build_s:.1f}s  "
          f"eos={sorted(srv_model.eos)}  "
          f"int8={int8}  max_seq={max_seq}  "
          f"prefix_cache={'paged' if prefix_on else 'off'}"
          f"{' (bs=%d, snapshots=%d)' % (block_size, snapshots) if prefix_on else ''}"
          f"  batch={srv_model.cfg.max_num_seqs if srv_model.engine.batching else 'off'}"
          f"  blocks={srv_model.cfg.num_blocks}"
          f"  spec_k={srv_model.cfg.spec_k}"
          f"{' (batched)' if srv_model.cfg.spec_batch else ''}"
          f"  max_seconds={max_seconds or 'none'}"
          f"  vision={'off' if not srv_model.vision else ('int8' if srv_model.vision_int8 else 'fp16')}"
          f" (max {srv_model.image_max_pixels} px, {srv_model.max_images} imgs)",
          flush=True)
    if os.path.exists(sock_path):
        os.unlink(sock_path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    srv.listen(8)
    print(f"[serve] listening on {sock_path}", flush=True)
    srv_model.start()

    # accept 超时是循环发现 ``_stop`` 的唯一办法：阻塞式 accept 会让 socket 在
    # 收到 ``stop`` 之后继续开着，直到恰好有新的客户端连上来。
    srv.settimeout(0.5)
    while not srv_model._stop:
        try:
            conn, _ = srv.accept()
        except socket.timeout:
            continue

        # 每条连接一个线程：引擎线程会把在途请求凑成批，慢的 SSE 客户端因此不再
        # 挡住下一个请求。
        threading.Thread(target=_serve_connection, args=(srv_model, conn),
                         daemon=True).start()
    srv_model.stop()
    srv.close()
    if os.path.exists(sock_path):
        os.unlink(sock_path)


def _serve_connection(srv_model: "Server", conn: socket.socket) -> None:
    """在自己的线程上服务一个 JSON 行请求（或一条 SSE 流）。

    回复作为一整行发出去，客户端「读到换行」的循环因此才能结束。``chat_stream`
    操作则每个流事件写一行，由 OpenAI 入口把它们变成 SSE 帧。

    Args:
        srv_model: 共享的服务对象；所有线程看到的是同一个。
        conn: 已接受的连接。无论哪种情况都在这里关掉，包括请求格式错误时，
            所以坏客户端不会泄漏描述符。
    """
    try:
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = conn.recv(65536)
            if not chunk:
                break
            buf += chunk
        req = json.loads(buf.decode() or "{}")
        t0 = time.time()
        if req.get("op") == "chat_stream":
            # 每个 token 一行回复，客户端才能把它们当作 SSE 转发出去。
            for line in srv_model.stream_chat(req):
                conn.sendall((json.dumps(line, ensure_ascii=False) + "\n").encode())
            print(f"[serve] chat_stream -> {round((time.time() - t0) * 1000, 1)}ms",
                  flush=True)
            return
        rep = srv_model.handle(req)
        rep["wall_ms"] = round((time.time() - t0) * 1000, 1)
        conn.sendall((json.dumps(rep, ensure_ascii=False) + "\n").encode())
        print(f"[serve] {req.get('op')} -> {rep.get('wall_ms')}ms", flush=True)
        if rep.get("stopping"):
            srv_model.stop()
    except Exception as exc:  # keep the server alive on a bad request
        # 从无法干净答复的路径里逃出来的 ``MMError``（比如 200 已经发出去的流），
        # 仍然带着自己的状态码上报，HTTP 外壳据此映射，而不是一律当成 500。
        rep = {"ok": False, "error": str(exc)}
        if isinstance(exc, MMError):
            rep["status"] = exc.status
        try:
            conn.sendall((json.dumps(rep) + "\n").encode())
        except Exception:
            pass
        print(f"[serve] error: {exc}", flush=True)
    finally:
        conn.close()


def call(sock_path: str, req: dict) -> dict:
    """向正在运行的服务发一个请求，返回它的回复。

    线协议的客户端部分：发一行 JSON，收回一行。

    Args:
        sock_path: 服务绑定的引擎 socket。
        req: 请求字典；``op`` 决定走哪个处理分支。

    Returns:
        解码后的回复。它可能带 ``ok: False``；本函数不会因为被拒绝而抛异常，
        因为命令行无论如何都要把回复打出来。

    Raises:
        OSError: 服务没在 ``sock_path`` 上监听。先把服务起起来；客户端重试
            修不好一个不存在的服务。
        json.JSONDecodeError: 服务在一整行还没发完时就关了连接。
    """
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(sock_path)
    s.sendall((json.dumps(req) + "\n").encode())
    buf = b""
    while not buf.endswith(b"\n"):
        chunk = s.recv(65536)
        if not chunk:
            break
        buf += chunk
    s.close()
    return json.loads(buf.decode())
