"""Unix-socket transport for the resident inference runtime.

Module: ``coffin.serving``. This module owns socket accept/read/write behavior;
model lifecycle remains in ``runtime_service.Server``.

Constraints:
    * The startup banner is the only place a silent backend fallback would be
      visible, so it prints ``backends.status()`` verbatim instead of assembling
      its own field list: the fields (``backend=``, ``device=``, ``native=``,
      ``graph=``, ``autotune=``) are the ones an operator diagnoses with, and a
      hand-built banner drifts from the backend the moment one gains a flag.
    * Nothing here may touch the device: a transport that allocates or
      synchronises cannot bind its socket on a host with no accelerator, and the
      reference backend is exactly the case that has to work there.
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
    """Build the server, bind the unix socket and serve until asked to stop.

    The argument list mirrors ``Server.__init__``; see it for the meaning of each
    knob. Everything here is process lifetime, not per request.

    Args:
        model_dir: Checkpoint directory.
        max_seq: Context window, shared by prompt and output.
        bench: Keep the legacy single-stream bench path available.
        sock_path: Unix socket to bind; an existing file at that path is removed
            first, so a stale socket from a killed run does not block startup.
        spec_k: Drafts per speculative step, 0 to disable.
        int8: Quantise the 2D weights.
        prefix_cache: Enable the paged radix prefix cache.
        block_size: KV block granularity in tokens.
        snapshots: Recurrent-state snapshot budget.
        snapshot_interval: Snapshot every N tokens, rounded to a block.
        max_seconds: Wall-clock cap on one generation, 0 for none.
        max_num_seqs: Rows in a batched decode step.
        num_blocks: KV pool size, 0 to derive it from the window.
        spec_batch: Speculate in the batched engine.
        spec_sparse_k: Approximate batched verify, top-k historical blocks.
        spec_min_ctx: Skip batched speculation below this cached length.
        vision: Load the checkpoint's vision tower when it has one.
        vision_int8: Keep the tower's 2D weights int8.
        image_max_pixels: Largest image the processor serves.
        image_min_pixels: Smallest image after the processor's upscale.
        max_images: Images accepted in one request.
        max_image_bytes: Decoded bytes allowed per image.
        graph: Capture the decode step into a device graph. ``False`` runs the
            eager step instead; it is the escape hatch for a backend whose graph
            replay is not trusted yet, and it must reach the engine before the
            graph is captured, which is why it travels as a constructor argument
            rather than an environment variable.
        prefill_quantum: Prompt tokens one request may prefill per step. The
            scheduler uses it to interleave long prompts with short ones; 0 falls
            back to the step's whole token budget.
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

    # The accept timeout is how the loop notices ``_stop``: a blocking accept
    # would keep the socket open after a ``stop`` request until a new client
    # happened to connect.
    srv.settimeout(0.5)
    while not srv_model._stop:
        try:
            conn, _ = srv.accept()
        except socket.timeout:
            continue

        # One thread per connection: the engine thread batches whatever requests
        # are in flight, so a slow SSE client no longer blocks the next request.
        threading.Thread(target=_serve_connection, args=(srv_model, conn),
                         daemon=True).start()
    srv_model.stop()
    srv.close()
    if os.path.exists(sock_path):
        os.unlink(sock_path)


def _serve_connection(srv_model: "Server", conn: socket.socket) -> None:
    """Serve one JSON-line request (or one SSE stream) on its own thread.

    The reply goes out as a single line, so the client's read-until-newline loop
    terminates. A ``chat_stream`` op instead writes one line per stream event,
    which the OpenAI entry point turns into SSE frames.

    Args:
        srv_model: The shared server; every thread sees the same one.
        conn: The accepted connection. It is closed here in every case, including
            a malformed request, so a bad client cannot leak a descriptor.
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
            # One reply line per token, so the client can forward them as SSE.
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
        # An ``MMError`` that escaped a path which could not answer cleanly (a
        # stream whose 200 is already committed) is still reported with its status
        # so the HTTP shell can map it instead of calling everything a 500.
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
    """Send one request to a running server and return its reply.

    The client half of the wire format: one JSON line out, one back.

    Args:
        sock_path: The engine socket the server bound.
        req: The request dict; ``op`` selects the handler.

    Returns:
        The decoded reply. It may carry ``ok: False``; this function does not
        raise on a refusal, because the CLI prints the reply either way.

    Raises:
        OSError: The server is not listening on ``sock_path``. Start it first;
            a client-side retry cannot fix a missing server.
        json.JSONDecodeError: The server closed without sending a full line.
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
